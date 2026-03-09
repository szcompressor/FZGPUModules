// compressor_inverse_dag.cpp — builds the inverse (decompression) DAG from a
// forward-topology description.
#include "pipeline/compressor.h"
#include "log.h"
#include <stdexcept>
#include <cstdint>
#include <vector>

namespace fz {

std::pair<std::unique_ptr<CompressionDAG>, std::unordered_map<Stage*, int>>
Pipeline::buildInverseDAG(
    const std::vector<FwdStageDesc>&          fwd_stages,
    const PipelineOutputMap&                  pipeline_outputs,
    MemoryPool*                               pool,
    MemoryStrategy                            strategy,
    const std::unordered_map<Stage*, size_t>& source_sizes,
    bool                                      enable_profiling
) {
    auto inv_dag = std::make_unique<CompressionDAG>(pool, strategy);
    if (enable_profiling) inv_dag->enableProfiling(true);

    std::unordered_map<Stage*, DAGNode*> inv_nodes;

    // Compute total uncompressed size across all sources for initial buffer hints.
    size_t total_uncompressed = 0;
    for (const auto& [s, sz] : source_sizes) total_uncompressed += sz;
    if (total_uncompressed == 0) total_uncompressed = 1;  // safety guard

    // Step 1: Add stages in REVERSE forward order.
    // assignLevels() needs parents before children; inverse parents = forward leaves.
    for (int i = static_cast<int>(fwd_stages.size()) - 1; i >= 0; i--) {
        Stage*   stage    = fwd_stages[i].stage;
        DAGNode* node     = inv_dag->addStage(stage, stage->getName());
        size_t   num_out  = stage->getNumOutputs();
        auto     out_names = stage->getOutputNames();
        for (size_t j = 0; j < num_out; j++) {
            std::string n = (j < out_names.size()) ? out_names[j] : std::to_string(j);
            inv_dag->addUnconnectedOutput(node, total_uncompressed,
                                          static_cast<int>(j),
                                          stage->getName() + "." + n);
        }
        inv_nodes[stage] = node;
    }

    // Build buf_to_consumer: fwd_buf_id → {index in fwd_stages, input position}
    std::unordered_map<int, std::pair<int,int>> buf_to_consumer;
    for (int i = 0; i < static_cast<int>(fwd_stages.size()); i++) {
        for (int j = 0; j < static_cast<int>(fwd_stages[i].input_buf_ids.size()); j++) {
            int bid = fwd_stages[i].input_buf_ids[j];
            if (bid >= 0) buf_to_consumer[bid] = {i, j};
        }
    }

    // Step 2: Wire inverse inputs in forward order.
    for (int i = 0; i < static_cast<int>(fwd_stages.size()); i++) {
        Stage*   curr     = fwd_stages[i].stage;
        DAGNode* inv_node = inv_nodes.at(curr);

        for (int fwd_out_buf_id : fwd_stages[i].output_buf_ids) {
            auto cons_it = buf_to_consumer.find(fwd_out_buf_id);
            if (cons_it != buf_to_consumer.end()) {
                // Intermediate buffer — produced in the inverse by the consumer's inverse node.
                int      cons_idx = cons_it->second.first;
                int      cons_pos = cons_it->second.second;
                Stage*   consumer = fwd_stages[cons_idx].stage;
                DAGNode* inv_prod = inv_nodes.at(consumer);

                bool ok = inv_dag->connectExistingOutput(inv_prod, inv_node, cons_pos);
                if (!ok) {
                    throw std::runtime_error(
                        "buildInverseDAG: connectExistingOutput failed for output " +
                        std::to_string(cons_pos) + " of stage '" +
                        consumer->getName() + "'");
                }
                FZ_LOG(DEBUG, "Inverse edge: %s.out[%d] -> %s (fwd_buf=%d)",
                       consumer->getName().c_str(), cons_pos,
                       curr->getName().c_str(), fwd_out_buf_id);
            } else {
                // Pipeline-output buffer — inject as external input to this inv_node.
                auto pe_it = pipeline_outputs.find(fwd_out_buf_id);
                if (pe_it == pipeline_outputs.end()) {
                    throw std::runtime_error(
                        "buildInverseDAG: compressed buffer not found for fwd_buf_id=" +
                        std::to_string(fwd_out_buf_id));
                }
                void*  d_ptr = pe_it->second.first;
                size_t sz    = pe_it->second.second;

                inv_dag->setInputBuffer(inv_node, sz,
                                        "inv_ext_" + std::to_string(fwd_out_buf_id));
                int ext_buf_id = inv_node->input_buffer_ids.back();
                inv_dag->setExternalPointer(ext_buf_id, d_ptr);

                FZ_LOG(DEBUG, "Inverse external input: fwd_buf=%d %.2f KB -> stage '%s'",
                       fwd_out_buf_id, sz / 1024.0, curr->getName().c_str());
            }
        }
    }

    // Step 3: Every stage present in source_sizes is a forward source and
    // therefore an inverse sink.  Mark its first output buffer persistent.
    std::unordered_map<Stage*, int> inv_result_map;  // source stage -> result buf id
    for (const auto& fwd_desc : fwd_stages) {
        if (!source_sizes.count(fwd_desc.stage)) continue;  // not a source
        Stage*   src_stage = fwd_desc.stage;
        DAGNode* inv_sink  = inv_nodes.at(src_stage);
        if (inv_sink->output_buffer_ids.empty()) {
            throw std::runtime_error(
                "buildInverseDAG: inv_sink '" + src_stage->getName() +
                "' has no output buffers");
        }
        int res_buf_id = inv_sink->output_buffer_ids[0];
        inv_dag->setBufferPersistent(res_buf_id, true);
        inv_result_map[src_stage] = res_buf_id;
        FZ_LOG(DEBUG, "Inverse sink: stage '%s', result_buf_id=%d",
               src_stage->getName().c_str(), res_buf_id);
    }
    if (inv_result_map.empty()) {
        throw std::runtime_error(
            "buildInverseDAG: no source stages found in forward topology");
    }

    // Step 4: Finalize — assigns levels and streams.
    inv_dag->finalize();

    // Step 5: Propagate estimated buffer sizes; override each result buffer
    // with the exact per-source uncompressed size.
    for (const auto& level_nodes : inv_dag->getLevels()) {
        for (auto* node : level_nodes) {
            std::vector<size_t> in_sizes;
            for (int buf_id : node->input_buffer_ids) {
                in_sizes.push_back(inv_dag->getBufferSize(buf_id));
            }
            auto est = node->stage->estimateOutputSizes(in_sizes);
            for (size_t k = 0;
                 k < node->output_buffer_ids.size() && k < est.size(); k++) {
                inv_dag->updateBufferSize(node->output_buffer_ids[k], est[k]);
            }
        }
    }
    // Override each source's result buffer with its exact known size.
    for (const auto& fwd_desc : fwd_stages) {
        if (!source_sizes.count(fwd_desc.stage)) continue;
        auto sz_it  = source_sizes.find(fwd_desc.stage);
        auto buf_it = inv_result_map.find(fwd_desc.stage);
        if (sz_it != source_sizes.end() && buf_it != inv_result_map.end()) {
            inv_dag->updateBufferSize(buf_it->second, sz_it->second);
        }
    }

    if (strategy == MemoryStrategy::PREALLOCATE) {
        inv_dag->preallocateBuffers();
    }

    FZ_LOG(DEBUG, "Inverse DAG: %zu levels, max_parallelism=%d, %zu result buf(s), strategy=%s",
           inv_dag->getLevels().size(),
           inv_dag->getMaxParallelism(),
           inv_result_map.size(),
           strategy == MemoryStrategy::MINIMAL   ? "MINIMAL"   :
           strategy == MemoryStrategy::PIPELINE  ? "PIPELINE"  : "PREALLOCATE");

    return {std::move(inv_dag), std::move(inv_result_map)};
}

} // namespace fz
