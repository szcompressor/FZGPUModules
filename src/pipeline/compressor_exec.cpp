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

// Must match the helper in compressor.cpp — rounds up to next multiple of 16.
static inline size_t align16(size_t x) { return (x + 15u) & ~15u; }

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

    // Wire inputs (normal mode) or copy to the fixed graph input slot (graph mode).
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

        // Guard: null device pointer is always a programming error.
        if (spec.d_data == nullptr) {
            throw std::runtime_error(
                "compress(): InputSpec for source stage '" + spec.source->getName() +
                "' has a null device pointer");
        }

        // Guard: data must not exceed the hint used to size buffers at
        // finalize().  A larger payload means output buffers were
        // under-allocated from estimateOutputSizes(), which leads to silent
        // data truncation (the stage only sees the hint-sized slice) or, for
        // stages with exact estimates, a buffer overwrite.
        {
            auto hint_it = per_source_hints_.find(spec.source);
            size_t hint = (hint_it != per_source_hints_.end())
                ? hint_it->second : input_size_hint_;
            if (hint > 0 && spec.size > hint) {
                throw std::runtime_error(
                    "compress(): InputSpec for source stage '" + spec.source->getName() +
                    "' data size (" + std::to_string(spec.size) +
                    " bytes) exceeds the finalize-time buffer size hint (" +
                    std::to_string(hint) + " bytes); "
                    "re-construct the pipeline with a larger input size hint");
            }
        }

        if (graph_captured_) {
            // Graph mode: the DAG's input buffer already has a stable pointer
            // (d_graph_input_, baked into the graph at captureGraph() time).
            // Copy user data into that slot; zero any tail padding so stages
            // that process aligned chunks see clean data past the real payload.
            FZ_CUDA_CHECK(cudaMemcpyAsync(d_graph_input_, spec.d_data, spec.size,
                                          cudaMemcpyDeviceToDevice, stream));
            if (spec.size < d_graph_input_size_) {
                FZ_CUDA_CHECK(cudaMemsetAsync(
                    static_cast<uint8_t*>(d_graph_input_) + spec.size,
                    0, d_graph_input_size_ - spec.size, stream));
            }
        } else {
            dag_->setExternalPointer(input_buffer_ids_[idx], const_cast<void*>(spec.d_data));
            dag_->updateBufferSize(input_buffer_ids_[idx], spec.size);
        }
        source_input_sizes_[idx] = spec.size;
        input_size_ += spec.size;
    }

    // Re-estimate buffer sizes from runtime inputs when no static hint was
    // given at finalize() time.  Skipped in graph mode (hint is required).
    if (!graph_captured_ && input_size_hint_ == 0 && per_source_hints_.empty()) {
        propagateBufferSizes(true);
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
        // Execute DAG (or replay the instantiated graph).
        // The graph path eliminates per-call CPU dispatch overhead: all kernel
        // launches and D2D copies recorded at captureGraph() time are replayed
        // in one driver call.  Post-graph work (sync, postStreamSync, output
        // collection) runs identically to the normal path.
        if (graph_captured_) {
            FZ_CUDA_CHECK(cudaGraphLaunch(graph_exec_, stream));
        } else {
            dag_->execute(stream);
        }
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

        // Get output buffer and size.
        //
        // OWNERSHIP CONTRACT: both paths below hand the caller a pointer into
        // the Pipeline's internal memory pool (d_concat_buffer_ or the DAG
        // pool).  The caller must NOT call cudaFree() on *d_output.
        // This is intentionally zero-copy.  Contrast with decompress(), which
        // always cudaMalloc's a fresh buffer that the caller IS responsible for
        // freeing.
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

    // Graph mode: d_graph_input_ is already sized at the padded capacity
    // (allocated in finalize()).  Pass the user's raw pointer directly to the
    // InputSpec overload; it will D2D-copy into d_graph_input_ and zero any
    // tail bytes before calling cudaGraphLaunch().  No d_pad_buf_ needed.
    if (graph_captured_) {
        original_input_size_ = (input_alignment_bytes_ > 1 &&
                                 input_size % input_alignment_bytes_ != 0)
                               ? input_size : 0;
        compress({InputSpec{input_nodes_[0]->stage, d_input, input_size}},
                 d_output, output_size, stream);
        return;
    }

    // Transparently pad the input to the required chunk boundary.
    const void* d_source  = d_input;
    size_t      source_sz = input_size;
    original_input_size_ = 0;  // reset: no padding applied yet
    if (input_alignment_bytes_ > 1 && input_size % input_alignment_bytes_ != 0) {
        const size_t padded = ((input_size + input_alignment_bytes_ - 1)
                               / input_alignment_bytes_) * input_alignment_bytes_;
        if (padded > d_pad_buf_size_) {
            if (d_pad_buf_) {
                mem_pool_->free(d_pad_buf_, stream);
                d_pad_buf_ = nullptr;
            }
            d_pad_buf_ = mem_pool_->allocate(padded, stream,
                                             "pipeline_input_pad", /*persistent=*/true);
            if (!d_pad_buf_) {
                throw std::runtime_error("Failed to allocate pipeline input pad buffer");
            }
            d_pad_buf_size_ = padded;
        }
        FZ_CUDA_CHECK(cudaMemcpyAsync(d_pad_buf_, d_input, input_size,
                                      cudaMemcpyDeviceToDevice, stream));
        FZ_CUDA_CHECK(cudaMemsetAsync(static_cast<uint8_t*>(d_pad_buf_) + input_size,
                                      0, padded - input_size, stream));
        FZ_LOG(INFO,
            "Input padded: %zu \u2192 %zu bytes (+%zu bytes for %zu-byte chunk alignment)",
            input_size, padded, padded - input_size, input_alignment_bytes_);
        d_source  = d_pad_buf_;
        source_sz = padded;
        original_input_size_ = input_size;  // remember for decompress() trimming
    }

    compress({InputSpec{input_nodes_[0]->stage, d_source, source_sz}},
             d_output, output_size, stream);
    // source_input_sizes_[0] is left as source_sz (padded) so decompressMulti()
    // allocates buffers large enough for the inverse pass to write into.
    // The caller-visible size is trimmed back in decompress() using original_input_size_.
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
            // Layout written by writeConcatBuffer():
            //   [u32 n][u64 s0]..[u64 sN-1][padding→16B][data0 slot (padded)][data1 slot]...
            // Walk past the padded header, then advance by each padded slot size.
            const size_t n          = buffer_metadata_.size();
            const size_t hdr_padded = align16(sizeof(uint32_t) + n * sizeof(uint64_t));
            size_t byte_offset      = hdr_padded;
            for (const auto& meta : buffer_metadata_) {
                compressed_ptrs[meta.buffer_id] =
                    static_cast<uint8_t*>(const_cast<void*>(d_input)) + byte_offset;
                byte_offset += align16(meta.actual_size);
            }
        }
    } else {
        for (const auto& meta : buffer_metadata_) {
            compressed_ptrs[meta.buffer_id] = dag_->getBuffer(meta.buffer_id);
        }
    }

    // ── Switch all stages to inverse mode ────────────────────────────────────
    for (auto& s : stages_) {
        s->saveState();
        s->setInverse(true);
    }

    // ── Build pipeline-output map (fwd_buf_id → {ptr, size}) ─────────────────
    PipelineOutputMap po_map;
    for (const auto& meta : buffer_metadata_) {
        auto it   = compressed_ptrs.find(meta.buffer_id);
        void* ptr = (it != compressed_ptrs.end())
            ? it->second
            : dag_->getBuffer(meta.buffer_id);  // safety fallback
        po_map[meta.buffer_id] = {ptr, meta.actual_size};
    }

    // ── Build per-source uncompressed sizes from what compress() recorded ────
    std::unordered_map<Stage*, size_t> source_sizes;
    for (size_t i = 0; i < input_nodes_.size(); i++) {
        size_t sz = (i < source_input_sizes_.size() && source_input_sizes_[i] > 0)
                    ? source_input_sizes_[i]
                    : input_size_;  // fallback: total (correct for single-source)
        source_sizes[input_nodes_[i]->stage] = sz;
    }

    // ── Inverse DAG: reuse cached instance or build and cache ─────────────────
    // The topology is fixed after finalize() so we only need to rebuild when
    // the source sizes change (i.e. a different-sized input was compressed).
    bool cache_valid = (inv_cache_ != nullptr);
    if (cache_valid) {
        for (const auto& [stage, sz] : source_sizes) {
            auto it = inv_cache_->source_sizes.find(stage);
            if (it == inv_cache_->source_sizes.end() || it->second != sz) {
                cache_valid = false;
                FZ_LOG(DEBUG, "decompressMulti: inv DAG cache invalidated (source size changed)");
                break;
            }
        }
    }

    if (!cache_valid) {
        // ── Build forward topology description ─────────────────────────────
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

        auto [inv_dag_up, inv_result_map_new] = buildInverseDAG(
            fwd_topology, po_map, mem_pool_.get(), strategy_,
            source_sizes, profiling_enabled_);

        // ── Build fwd_buf_id → inv external buffer id map ──────────────────
        // Stored so future calls can update the external pointers directly
        // without iterating all nodes.  The tag format is "inv_ext_<fwd_buf_id>"
        // (set by buildInverseDAG) which lets us reconstruct the mapping here.
        std::unordered_map<int, int> fwd_to_inv_ext_buf;
        for (auto* node : inv_dag_up->getNodes()) {
            for (int buf_id : node->input_buffer_ids) {
                const auto& info = inv_dag_up->getBufferInfo(buf_id);
                if (info.is_external && info.tag.size() > 8 &&
                    info.tag.compare(0, 8, "inv_ext_") == 0) {
                    try {
                        int fwd_buf_id = std::stoi(info.tag.substr(8));
                        fwd_to_inv_ext_buf[fwd_buf_id] = buf_id;
                    } catch (...) {}
                }
            }
        }

        inv_cache_ = std::make_unique<InvDAGCache>();
        inv_cache_->inv_dag            = std::move(inv_dag_up);
        inv_cache_->inv_result_map     = std::move(inv_result_map_new);
        inv_cache_->fwd_to_inv_ext_buf = std::move(fwd_to_inv_ext_buf);
        inv_cache_->source_sizes       = source_sizes;

        FZ_LOG(DEBUG, "decompressMulti: built and cached inverse DAG (%zu ext buffers mapped)",
               inv_cache_->fwd_to_inv_ext_buf.size());
    } else {
        // ── Cache hit: update compressed-data pointers and reset ───────────
        for (const auto& [fwd_buf_id, inv_buf_id] : inv_cache_->fwd_to_inv_ext_buf) {
            auto it = po_map.find(fwd_buf_id);
            if (it != po_map.end()) {
                inv_cache_->inv_dag->setExternalPointer(inv_buf_id, it->second.first);
                inv_cache_->inv_dag->updateBufferSize(inv_buf_id, it->second.second);
            }
        }
        // Restore ref-counts and reset non-persistent intermediate buffers.
        // Persistent result buffers and PREALLOCATE intermediates are kept.
        inv_cache_->inv_dag->reset(stream);
        inv_cache_->inv_dag->enableProfiling(profiling_enabled_);

        FZ_LOG(DEBUG, "decompressMulti: reusing cached inverse DAG");
    }

    CompressionDAG& inv_dag        = *inv_cache_->inv_dag;
    const auto&     inv_result_map = inv_cache_->inv_result_map;

    // ── Pre-allocate caller output buffers and wire as external DAG outputs ───
    // §4: the inv DAG writes directly into these caller-owned buffers,
    // eliminating the post-execute pool-buf → cudaMalloc D2D copy.
    // setExternalPointer() handles pool→external transparently: it frees any
    // prior pool/persistent allocation and marks the slot external so
    // allocateBuffer() skips it during execute().
    // ── Release previous pool-managed outputs (if any) ───────────────────────
    if (pool_managed_decomp_) {
        for (void* p : d_decomp_outputs_) {
            if (p && mem_pool_) mem_pool_->free(p, stream);
        }
        d_decomp_outputs_.clear();
    }

    std::vector<std::pair<void*, size_t>> results;
    results.reserve(input_nodes_.size());
    for (size_t i = 0; i < input_nodes_.size(); i++) {
        Stage* src_stage  = input_nodes_[i]->stage;
        auto   buf_it     = inv_result_map.find(src_stage);
        if (buf_it == inv_result_map.end()) {
            if (pool_managed_decomp_) {
                for (auto& r : results) mem_pool_->free(r.first, stream);
            } else {
                for (auto& r : results) cudaFree(r.first);
            }
            for (auto& s : stages_) { s->setInverse(false); s->restoreState(); }
            throw std::runtime_error(
                "decompressMulti: no inverse result buffer for source stage '" +
                src_stage->getName() + "'");
        }
        int    res_buf_id  = buf_it->second;
        size_t actual_size = inv_dag.getBufferSize(res_buf_id);

        void* d_final = nullptr;
        if (pool_managed_decomp_) {
            // Allocate from pool (persistent) — zero-copy, inv DAG writes directly here.
            // Caller must NOT cudaFree(); pointer is valid until the next decompress() call.
            d_final = mem_pool_->allocate(actual_size, stream, "decomp_output", /*persistent=*/true);
            if (!d_final) {
                for (auto& r : results) mem_pool_->free(r.first, stream);
                for (auto& s : stages_) { s->setInverse(false); s->restoreState(); }
                throw std::runtime_error("pool allocation for decompress output failed");
            }
            d_decomp_outputs_.push_back(d_final);
        } else {
            cudaError_t err = cudaMalloc(&d_final, actual_size);
            if (err != cudaSuccess) {
                for (auto& r : results) cudaFree(r.first);
                for (auto& s : stages_) { s->setInverse(false); s->restoreState(); }
                throw std::runtime_error("cudaMalloc for decompress output failed");
            }
        }
        inv_dag.setExternalPointer(res_buf_id, d_final);
        results.push_back({d_final, actual_size});
    }

    // ── Execute ───────────────────────────────────────────────────────────────
    auto t_dag_start = std::chrono::steady_clock::now();
    inv_dag.execute(stream);
    auto t_dag_end = std::chrono::steady_clock::now();
    FZ_CUDA_CHECK(cudaStreamSynchronize(stream));

    for (auto& stage_ptr : stages_) {
        stage_ptr->postStreamSync(stream);
    }

    auto stage_timings = profiling_enabled_
        ? inv_dag.collectTimings()
        : std::vector<StageTimingResult>{};

    // ── Refine output sizes from postStreamSync (no copy needed) ─────────────
    // The inv DAG already wrote into d_final directly. Just update the size
    // field if the stage reports a smaller actual size post-execution.
    for (size_t i = 0; i < input_nodes_.size(); i++) {
        Stage* src_stage = input_nodes_[i]->stage;
        auto post_sizes  = src_stage->getActualOutputSizesByName();
        auto post_names  = src_stage->getOutputNames();
        if (!post_names.empty() && post_sizes.count(post_names[0])) {
            results[i].second = post_sizes.at(post_names[0]);
        }
    }

    // ── Cleanup ───────────────────────────────────────────────────────────────
    // Result buffers are now external (caller-owned via cudaMalloc) — reset()
    // skips external buffers so they are not freed here.
    // Intermediate non-external buffers are freed by reset() in MINIMAL mode.
    // In PREALLOCATE mode reset() is a no-op for non-external allocations.
    inv_dag.reset(stream);

    for (auto& s : stages_) {
        s->setInverse(false);
        s->restoreState();
    }

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
        // If compress() transparently padded the input, trim the reported output
        // size back to the original (unpadded) byte count.  The tail bytes
        // (zero-padded) are harmless but should not be visible to the caller.
        if (original_input_size_ > 0 && *output_size > original_input_size_)
            *output_size = original_input_size_;
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
        if (pool_managed_decomp_) {
            for (void* p : d_decomp_outputs_) { if (p && mem_pool_) mem_pool_->free(p, stream); }
            d_decomp_outputs_.clear();
        } else {
            for (auto& r : results) cudaFree(r.first);
        }

        *d_output    = d_final;
        *output_size = total;

        // Update profiling output_bytes to reflect the final concat size.
        if (profiling_enabled_) last_perf_result_.output_bytes = total;
    }
}

} // namespace fz
