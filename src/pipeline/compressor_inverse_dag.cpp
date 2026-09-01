// compressor_inverse_dag.cpp — builds the inverse (decompression) DAG from a
// forward-topology description.
#include "pipeline/compressor.h"
#include "advanced/fusion_registry.h"
#include "log.h"
#include <stdexcept>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <unordered_set>

namespace fz {

std::pair<std::unique_ptr<CompressionDAG>, std::unordered_map<Stage*, int>>
Pipeline::buildInverseDAG(
    const std::vector<FwdStageDesc>&          fwd_stages,
    const PipelineOutputMap&                  pipeline_outputs,
    MemoryPool*                               pool,
    MemoryStrategy                            strategy,
    const std::unordered_map<Stage*, size_t>& source_sizes,
    bool                                      enable_profiling,
    bool                                      enable_inverse_fusion
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

    // Install evidence-gated inverse implementations over linear inverse chains.
    // Registry matching, not the topology walker, owns semantic eligibility.
    if (enable_inverse_fusion) {
        std::vector<CompressionDAG::FusedGroupExec> installed;
        std::unordered_set<DAGNode*> claimed;
        for (DAGNode* start : inv_dag->getNodes()) {
            if (!start || !start->dependencies.empty() || claimed.count(start)) continue;
            std::vector<DAGNode*> chain;
            for (DAGNode* cur = start; cur && !claimed.count(cur);) {
                chain.push_back(cur);
                if (cur->dependents.size() != 1 ||
                    cur->dependents[0]->dependencies.size() != 1) break;
                cur = cur->dependents[0];
            }
            for (size_t begin = 0; begin + 1 < chain.size();) {
                const FusedImpl* selected = nullptr;
                size_t selected_end = begin;
                for (size_t end = chain.size(); end >= begin + 2; --end) {
                    std::vector<Stage*> stages;
                    for (size_t i = begin; i < end; ++i) stages.push_back(chain[i]->stage);
                    if (const FusedImpl* impl = findFusedImpl(stages, false)) {
                        selected = impl;
                        selected_end = end;
                        break;
                    }
                    if (end == begin + 2) break;
                }
                if (!selected) { ++begin; continue; }
                CompressionDAG::FusedGroupExec fg;
                fg.head = chain[begin];
                fg.tail = chain[selected_end - 1];
                fg.impl = selected;
                for (size_t i = begin; i < selected_end; ++i) {
                    fg.members.push_back(chain[i]);
                    fg.stages.push_back(chain[i]->stage);
                    claimed.insert(chain[i]);
                }
                FZ_LOG(INFO, "Inverse fusion: installed '%s' over %zu stages",
                       selected->name, fg.stages.size());
                installed.push_back(std::move(fg));
                begin = selected_end;
            }
        }
        if (!installed.empty()) {
            inv_dag->setFusedGroups(std::move(installed));
        }
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
    // Grow each source's result buffer to its exact known uncompressed size.
    //
    // Deliberately max(), not an override. The recorded source size is exact
    // for what the sink stage *reports*, but a stage may legitimately *write*
    // more than it reports: GPULZ zero-pads a partial tail chunk and its decode
    // kernel writes the whole padded extent, then reports the pre-padding size
    // so downstream element counts stay right. Overriding with the smaller
    // number shrank the buffer below what the kernel writes, and in PREALLOCATE
    // the overrun landed in the packed region behind it -- which is the *input*
    // compressed buffer, so the tail chunk's write raced the other chunks'
    // reads of the same stream and they decoded to garbage. It presented as
    // whole chunks coming back zero-filled, far from the tail: CESM-2D
    // qcodes/PRECT lost 14,480 bytes across 372 of 6,329 chunks, silently, with
    // status ok. Taking the max keeps the exact size where the estimate is
    // smaller (its original purpose) without ever undercutting the stage.
    for (const auto& fwd_desc : fwd_stages) {
        if (!source_sizes.count(fwd_desc.stage)) continue;
        auto sz_it  = source_sizes.find(fwd_desc.stage);
        auto buf_it = inv_result_map.find(fwd_desc.stage);
        if (sz_it != source_sizes.end() && buf_it != inv_result_map.end()) {
            const size_t est_sz = inv_dag->getBufferSize(buf_it->second);
            inv_dag->updateBufferSize(buf_it->second,
                                      std::max(est_sz, sz_it->second));
        }
    }

    if (strategy == MemoryStrategy::PREALLOCATE) {
        inv_dag->preallocateBuffers();
    }

    FZ_LOG(DEBUG, "Inverse DAG: %zu levels, max_parallelism=%d, %zu result buf(s), strategy=%s",
           inv_dag->getLevels().size(),
           inv_dag->getMaxParallelism(),
           inv_result_map.size(),
           strategy == MemoryStrategy::MINIMAL ? "MINIMAL" : "PREALLOCATE");

    return {std::move(inv_dag), std::move(inv_result_map)};
}

} // namespace fz
