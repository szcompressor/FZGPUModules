// compressor_exec.cpp — in-memory compress / decompress execution
#include "pipeline/compressor.h"
#include "log.h"
#include "cuda_check.h"
#include <chrono>
#include <stdexcept>
#include <algorithm>
#include <cstdint>
#include <vector>

namespace fz {

// ── Helper: build per-level timing summary from per-stage CUDA event timings ──
static std::vector<LevelTimingResult> buildLevelTimings(
    const std::vector<StageTimingResult>& stages
) {
    std::unordered_map<int, LevelTimingResult> level_map;
    for (const auto& st : stages) {
        auto& lv = level_map[st.level];
        lv.level = st.level;
        lv.parallelism++;
        lv.elapsed_ms = std::max(lv.elapsed_ms, st.elapsed_ms);
    }
    std::vector<LevelTimingResult> levels;
    for (auto& [lvl, lv] : level_map) levels.push_back(lv);
    std::sort(levels.begin(), levels.end(),
              [](const LevelTimingResult& a, const LevelTimingResult& b) {
                  return a.level < b.level;
              });
    return levels;
}

// =====

void Pipeline::compress(
    const std::vector<InputSpec>& inputs,
    void** d_output,
    size_t* output_size,
    cudaStream_t stream
) {
    if (!is_finalized_) {
        throw std::runtime_error("Pipeline must be finalized before execution");
    }
    if (inputs.size() != input_nodes_.size()) {
        throw std::runtime_error(
            "compress(): expected " + std::to_string(input_nodes_.size()) +
            " input(s), got " + std::to_string(inputs.size()));
    }

    if (was_compressed_) {
        dag_->reset(stream);
        was_compressed_ = false;
    }

    // Wire each source's external input pointer and record per-source sizes.
    input_size_ = 0;
    source_input_sizes_.assign(input_nodes_.size(), 0);
    for (const auto& spec : inputs) {
        auto node_it = stage_to_node_.find(spec.source);
        if (node_it == stage_to_node_.end()) {
            throw std::runtime_error(
                "compress(): InputSpec source stage '" +
                (spec.source ? spec.source->getName() : "<null>") +
                "' not found in pipeline");
        }
        DAGNode* src_node = node_it->second;
        auto idx_it = std::find(input_nodes_.begin(), input_nodes_.end(), src_node);
        if (idx_it == input_nodes_.end()) {
            throw std::runtime_error(
                "compress(): InputSpec source stage '" + spec.source->getName() +
                "' is not a pipeline source stage");
        }
        size_t idx = static_cast<size_t>(std::distance(input_nodes_.begin(), idx_it));
        dag_->setExternalPointer(input_buffer_ids_[idx], const_cast<void*>(spec.d_data));
        source_input_sizes_[idx] = spec.size;
        input_size_ += spec.size;
    }

    // Host-side wall-clock start (covers everything including output gathering)
    auto t_host_start = std::chrono::steady_clock::now();

    // Clear any stale metadata upfront; repopulated only on success so a
    // failed compress() never leaves buffer_metadata_ in a partially-written state.
    buffer_metadata_.clear();

    // Execute DAG + post-processing.
    // On any failure: reset pool allocations and clear half-written state.
    std::vector<StageTimingResult> stage_timings;
    auto t_dag_start = std::chrono::steady_clock::now();
    auto t_dag_end   = t_dag_start;  // updated inside try block on success
    try {
        // Execute DAG — time this portion separately for dag_elapsed_ms
        dag_->execute(stream);
        t_dag_end = std::chrono::steady_clock::now();

        // required for postStreamSync() and cuda events
        FZ_CUDA_CHECK(cudaStreamSynchronize(stream));

        // Allow stages to finalize host-side state from GPU results
        // (e.g. Lorenzo reads back actual outlier count and trims output sizes).
        for (auto& stage_ptr : stages_) {
            stage_ptr->postStreamSync(stream);
        }

        // If profiling: collect CUDA event timings (stream already synced above)
        stage_timings = profiling_enabled_ ? dag_->collectTimings() : std::vector<StageTimingResult>{};

        // Capture buffer metadata for file serialization
        for (size_t i = 0; i < output_buffer_ids_.size(); i++) {
            int buffer_id = output_buffer_ids_[i];
            const auto& buffer_info = dag_->getBufferInfo(buffer_id);
            DAGNode* producer = output_nodes_[i];

            BufferMetadata meta;
            meta.buffer_id = buffer_id;
            meta.allocated_size = buffer_info.size;
            meta.producer = producer;
            meta.output_index = buffer_info.producer_output_index;

            // Get output name
            auto output_names = producer->stage->getOutputNames();
            int output_idx = buffer_info.producer_output_index;
            meta.name = (output_idx >= 0 && output_idx < static_cast<int>(output_names.size()))
                        ? output_names[output_idx]
                        : "output";

            // Get actual size from stage
            auto sizes_by_name = producer->stage->getActualOutputSizesByName();
            auto it = sizes_by_name.find(meta.name);
            meta.actual_size = (it != sizes_by_name.end()) ? it->second : buffer_info.size;

            buffer_metadata_.push_back(meta);
        }

        // Get output buffer and size
        if (needs_concat_) {
            concatOutputs(d_output, output_size, stream);
        } else {
            *d_output = dag_->getBuffer(output_buffer_ids_[0]);

            auto sizes_by_name = output_nodes_[0]->stage->getActualOutputSizesByName();
            auto output_names  = output_nodes_[0]->stage->getOutputNames();

            *output_size = 0;
            if (!output_names.empty() && sizes_by_name.count(output_names[0])) {
                *output_size = sizes_by_name.at(output_names[0]);
            }
        }
    } catch (...) {
        // Restore clean state so subsequent calls don't observe partial output
        dag_->reset(stream);
        buffer_metadata_.clear();
        input_size_ = 0;
        was_compressed_ = false;
        throw;
    }

    // Mark that a successful compress() has been completed.
    is_compressed_  = true;
    was_compressed_ = true;

    // Host-side wall-clock end
    auto t_host_end = std::chrono::steady_clock::now();

    float host_ms = std::chrono::duration<float, std::milli>(t_host_end - t_host_start).count();
    float dag_ms  = std::chrono::duration<float, std::milli>(t_dag_end  - t_dag_start ).count();

    // Build profiling result
    if (profiling_enabled_) {
        PipelinePerfResult r;
        r.is_compress     = true;
        r.host_elapsed_ms = host_ms;
        r.dag_elapsed_ms  = dag_ms;
        r.input_bytes     = input_size_;
        r.output_bytes    = *output_size;
        r.stages          = std::move(stage_timings);

        // Build per-level aggregates
        r.levels = buildLevelTimings(r.stages);

        last_perf_result_ = std::move(r);
    }

    FZ_LOG(INFO, "Compress complete: %zu -> %zu bytes (host=%.2f ms, dag=%.2f ms, DAG=%.2f GB/s, pipeline=%.2f GB/s)",
           input_size_, *output_size, host_ms, dag_ms,
           profiling_enabled_ ? last_perf_result_.throughput_gbs() : 0.0f,
           profiling_enabled_ ? last_perf_result_.pipeline_throughput_gbs() : 0.0f);
}

void Pipeline::compress(
    const void* d_input,
    size_t input_size,
    void** d_output,
    size_t* output_size,
    cudaStream_t stream
) {
    if (!is_finalized_) {
        throw std::runtime_error("Pipeline must be finalized before execution");
    }
    if (input_nodes_.size() > 1) {
        throw std::runtime_error(
            "Pipeline has " + std::to_string(input_nodes_.size()) +
            " source stages; use compress(const std::vector<InputSpec>&) for multi-source pipelines");
    }
    compress({InputSpec{input_nodes_[0]->stage, d_input, input_size}},
             d_output, output_size, stream);
}

// ========== In-Memory Decompression ==========

std::vector<std::pair<void*, size_t>> Pipeline::decompressMulti(
    const void* d_input,
    size_t      input_size,
    cudaStream_t stream
) {
    if (!is_finalized_) {
        throw std::runtime_error("Pipeline not finalized");
    }
    if (buffer_metadata_.empty()) {
        throw std::runtime_error("decompressMulti() requires compress() to have been called first");
    }

    auto t_host_start = std::chrono::steady_clock::now();
    FZ_LOG(INFO, "Decompressing (%zu source(s))", input_nodes_.size());

    // ── Build compressed-data pointer map ────────────────────────────────────
    // Determines which device pointer feeds each compressed buffer in the
    // inverse DAG.  Three cases are handled: see decompress() header doc.
    std::unordered_map<int, void*> compressed_ptrs;
    if (d_input != nullptr) {
        if (!needs_concat_) {
            if (!buffer_metadata_.empty()) {
                compressed_ptrs[buffer_metadata_[0].buffer_id] =
                    const_cast<void*>(d_input);
            }
        } else {
            size_t byte_offset = sizeof(uint32_t);
            for (const auto& meta : buffer_metadata_) {
                byte_offset += sizeof(uint64_t);
                compressed_ptrs[meta.buffer_id] =
                    static_cast<uint8_t*>(const_cast<void*>(d_input)) + byte_offset;
                byte_offset += meta.actual_size;
            }
        }
    } else {
        for (const auto& meta : buffer_metadata_) {
            compressed_ptrs[meta.buffer_id] = dag_->getBuffer(meta.buffer_id);
        }
    }

    // ── Switch all stages to inverse mode ────────────────────────────────────
    for (auto& s : stages_) s->setInverse(true);

    // ── Assemble forward-topology description ────────────────────────────────
    PipelineOutputMap po_map;
    for (const auto& meta : buffer_metadata_) {
        auto it   = compressed_ptrs.find(meta.buffer_id);
        void* ptr = (it != compressed_ptrs.end())
            ? it->second
            : dag_->getBuffer(meta.buffer_id);  // safety fallback
        po_map[meta.buffer_id] = {ptr, meta.actual_size};
    }

    std::vector<FwdStageDesc> fwd_topology;
    fwd_topology.reserve(stages_.size());
    for (const auto& level_nodes : dag_->getLevels()) {
        for (auto* fwd_node : level_nodes) {
            FwdStageDesc d;
            d.stage          = fwd_node->stage;
            d.output_buf_ids = fwd_node->output_buffer_ids;
            d.input_buf_ids  = fwd_node->input_buffer_ids;
            fwd_topology.push_back(std::move(d));
        }
    }

    // ── Build per-source uncompressed sizes from what compress() recorded ────
    std::unordered_map<Stage*, size_t> source_sizes;
    for (size_t i = 0; i < input_nodes_.size(); i++) {
        size_t sz = (i < source_input_sizes_.size() && source_input_sizes_[i] > 0)
                    ? source_input_sizes_[i]
                    : input_size_;  // fallback: total (correct for single-source)
        source_sizes[input_nodes_[i]->stage] = sz;
    }

    // ── Build and finalize the inverse DAG ───────────────────────────────────
    auto [inv_dag, inv_result_map] = buildInverseDAG(
        fwd_topology, po_map, mem_pool_.get(), strategy_,
        source_sizes, profiling_enabled_);

    // ── Execute ───────────────────────────────────────────────────────────────
    auto t_dag_start = std::chrono::steady_clock::now();
    inv_dag->execute(stream);
    auto t_dag_end = std::chrono::steady_clock::now();
    FZ_CUDA_CHECK(cudaStreamSynchronize(stream));

    for (auto& stage_ptr : stages_) {
        stage_ptr->postStreamSync(stream);
    }

    auto stage_timings = profiling_enabled_
        ? inv_dag->collectTimings()
        : std::vector<StageTimingResult>{};

    // ── Extract results in source-discovery order (matches input_nodes_) ─────
    // For each source stage, refine the buffer size from postStreamSync output,
    // then cudaMalloc a persistent copy.
    std::vector<std::pair<void*, size_t>> results;
    results.reserve(input_nodes_.size());

    for (size_t i = 0; i < input_nodes_.size(); i++) {
        Stage* src_stage = input_nodes_[i]->stage;
        auto   buf_it    = inv_result_map.find(src_stage);
        if (buf_it == inv_result_map.end()) {
            // Shouldn't happen — guard against missing entries
            throw std::runtime_error(
                "decompressMulti: no inverse result buffer for source stage '" +
                src_stage->getName() + "'");
        }
        int    res_buf_id  = buf_it->second;
        void*  d_inv_ptr   = inv_dag->getBuffer(res_buf_id);
        size_t actual_size = inv_dag->getBufferSize(res_buf_id);

        // Prefer the stage's post-execution reported size (e.g. Lorenzo trimming).
        auto post_sizes = src_stage->getActualOutputSizesByName();
        auto post_names = src_stage->getOutputNames();
        if (!post_names.empty() && post_sizes.count(post_names[0])) {
            actual_size = post_sizes.at(post_names[0]);
        }

        void* d_final = nullptr;
        cudaError_t err = cudaMalloc(&d_final, actual_size);
        if (err != cudaSuccess) {
            // Free any already-allocated outputs before unwinding
            for (auto& r : results) cudaFree(r.first);
            inv_dag->reset(stream);
            for (size_t j = 0; j < input_nodes_.size(); j++)
                mem_pool_->free(inv_dag->getBuffer(
                    inv_result_map.at(input_nodes_[j]->stage)), stream);
            for (auto& s : stages_) s->setInverse(false);
            throw std::runtime_error("cudaMalloc for decompress output failed");
        }
        FZ_CUDA_CHECK(cudaMemcpyAsync(d_final, d_inv_ptr, actual_size,
                                      cudaMemcpyDeviceToDevice, stream));
        results.push_back({d_final, actual_size});
    }
    FZ_CUDA_CHECK(cudaStreamSynchronize(stream));

    // ── Cleanup ───────────────────────────────────────────────────────────────
    // Free each result buffer in the pool before resetting the DAG.
    for (const auto& [src_stage, res_buf_id] : inv_result_map) {
        mem_pool_->free(inv_dag->getBuffer(res_buf_id), stream);
    }
    inv_dag->reset(stream);
    for (auto& s : stages_) s->setInverse(false);

    // ── Profiling ─────────────────────────────────────────────────────────────
    auto t_host_end = std::chrono::steady_clock::now();
    float host_ms = std::chrono::duration<float, std::milli>(t_host_end - t_host_start).count();
    float dag_ms  = std::chrono::duration<float, std::milli>(t_dag_end  - t_dag_start ).count();

    size_t total_output = 0;
    for (const auto& r : results) total_output += r.second;

    if (profiling_enabled_) {
        PipelinePerfResult r;
        r.is_compress     = false;
        r.host_elapsed_ms = host_ms;
        r.dag_elapsed_ms  = dag_ms;
        r.input_bytes     = input_size;
        r.output_bytes    = total_output;
        r.stages          = std::move(stage_timings);
        r.levels          = buildLevelTimings(r.stages);
        last_perf_result_ = std::move(r);
    }

    FZ_LOG(INFO, "Decompress complete (DAG-native): %zu bytes total, %zu source(s) "
           "(host=%.2f ms, dag=%.2f ms, DAG=%.2f GB/s, pipeline=%.2f GB/s)",
           total_output, results.size(), host_ms, dag_ms,
           profiling_enabled_ ? last_perf_result_.throughput_gbs() : 0.0f,
           profiling_enabled_ ? last_perf_result_.pipeline_throughput_gbs() : 0.0f);

    return results;
}

void Pipeline::decompress(
    const void* d_input,
    size_t input_size,
    void** d_output,
    size_t* output_size,
    cudaStream_t stream
) {
    auto results = decompressMulti(d_input, input_size, stream);

    if (results.size() == 1) {
        // Single-source — return raw buffer directly (backward compatible).
        *d_output    = results[0].first;
        *output_size = results[0].second;
    } else {
        // Multi-source — concatenate in the same format as compress() multi-output:
        //   [num_bufs:u32][size1:u64][data1][size2:u64][data2]...
        uint32_t num_bufs  = static_cast<uint32_t>(results.size());
        size_t   total     = sizeof(uint32_t);
        for (const auto& r : results) total += sizeof(uint64_t) + r.second;

        void* d_final = nullptr;
        cudaError_t err = cudaMalloc(&d_final, total);
        if (err != cudaSuccess) {
            for (auto& r : results) cudaFree(r.first);
            throw std::runtime_error("cudaMalloc for multi-source decompress output failed");
        }

        uint8_t* dst = static_cast<uint8_t*>(d_final);
        FZ_CUDA_CHECK(cudaMemcpyAsync(dst, &num_bufs, sizeof(uint32_t),
                                      cudaMemcpyHostToDevice, stream));
        dst += sizeof(uint32_t);
        for (const auto& r : results) {
            uint64_t sz = static_cast<uint64_t>(r.second);
            FZ_CUDA_CHECK(cudaMemcpyAsync(dst, &sz, sizeof(uint64_t),
                                          cudaMemcpyHostToDevice, stream));
            dst += sizeof(uint64_t);
            FZ_CUDA_CHECK(cudaMemcpyAsync(dst, r.first, r.second,
                                          cudaMemcpyDeviceToDevice, stream));
            dst += r.second;
        }
        FZ_CUDA_CHECK(cudaStreamSynchronize(stream));

        // Individual buffers have been copied into the concat buffer; free them.
        for (auto& r : results) cudaFree(r.first);

        *d_output    = d_final;
        *output_size = total;

        // Update profiling output_bytes to reflect the final concat size.
        if (profiling_enabled_) last_perf_result_.output_bytes = total;
    }
}

} // namespace fz
