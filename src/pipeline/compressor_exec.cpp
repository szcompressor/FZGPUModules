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
    const void* d_input,
    size_t      input_size,
    void**      d_output,
    size_t*     output_size,
    cudaStream_t stream
) {
    if (!is_finalized_) {
        throw std::runtime_error("Pipeline must be finalized before execution");
    }
    if (input_nodes_.size() != 1) {
        throw std::runtime_error(
            "compress(): pipeline has " + std::to_string(input_nodes_.size()) +
            " source stage(s); only single-source pipelines are supported");
    }
    if (d_input == nullptr) {
        throw std::runtime_error(
            "compress(): null device pointer passed as input");
    }

    if (was_compressed_) {
        dag_->reset(stream);
        was_compressed_ = false;
    }

    // Guard: data must not exceed the hint used to size buffers at finalize().
    if (input_size_hint_ > 0 && input_size > input_size_hint_) {
        throw std::runtime_error(
            "compress(): input size (" + std::to_string(input_size) +
            " bytes) exceeds the finalize-time buffer size hint (" +
            std::to_string(input_size_hint_) + " bytes); "
            "re-construct the pipeline with a larger input size hint");
    }

    // Transparently pad the input to the required chunk boundary.
    const void* d_source  = d_input;
    size_t      source_sz = input_size;
    original_input_size_ = 0;

    if (graph_captured_) {
        // Graph mode: copy user data into the stable graph input buffer,
        // zero any tail padding so stages see clean data past the real payload.
        original_input_size_ = (input_alignment_bytes_ > 1 &&
                                 input_size % input_alignment_bytes_ != 0)
                               ? input_size : 0;
        FZ_CUDA_CHECK(cudaMemcpyAsync(d_graph_input_, d_input, input_size,
                                      cudaMemcpyDeviceToDevice, stream));
        if (input_size < d_graph_input_size_) {
            FZ_CUDA_CHECK(cudaMemsetAsync(
                static_cast<uint8_t*>(d_graph_input_) + input_size,
                0, d_graph_input_size_ - input_size, stream));
        }
        d_source  = d_graph_input_;
        source_sz = d_graph_input_size_;
    } else if (input_alignment_bytes_ > 1 && input_size % input_alignment_bytes_ != 0) {
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
                throw std::runtime_error(
                    "Failed to allocate pipeline input pad buffer (" +
                    std::to_string(padded) + " bytes); pool may be exhausted");
            }
            d_pad_buf_size_ = padded;
        }
        FZ_CUDA_CHECK(cudaMemcpyAsync(d_pad_buf_, d_input, input_size,
                                      cudaMemcpyDeviceToDevice, stream));
        FZ_CUDA_CHECK(cudaMemsetAsync(static_cast<uint8_t*>(d_pad_buf_) + input_size,
                                      0, padded - input_size, stream));
        FZ_LOG(INFO,
            "Input padded: %zu → %zu bytes (+%zu bytes for %zu-byte chunk alignment)",
            input_size, padded, padded - input_size, input_alignment_bytes_);
        d_source  = d_pad_buf_;
        source_sz = padded;
        original_input_size_ = input_size;
    }

    dag_->setExternalPointer(input_buffer_ids_[0], const_cast<void*>(d_source));
    dag_->updateBufferSize(input_buffer_ids_[0], source_sz);
    source_input_sizes_.assign(1, source_sz);
    input_size_ = source_sz;

    // Re-estimate buffer sizes from runtime inputs when no static hint was given.
    // Skipped in graph mode (hint is required).
    if (!graph_captured_ && input_size_hint_ == 0) {
        propagateBufferSizes(true);
    }

    auto t_host_start = std::chrono::steady_clock::now();

    buffer_metadata_.clear();

    std::vector<StageTimingResult> stage_timings;
    auto t_dag_start = std::chrono::steady_clock::now();
    auto t_dag_end   = t_dag_start;
    try {
        if (graph_captured_) {
            FZ_CUDA_CHECK(cudaGraphLaunch(graph_exec_, stream));
        } else {
            dag_->execute(stream);
        }
        t_dag_end = std::chrono::steady_clock::now();

        FZ_CUDA_CHECK(cudaStreamSynchronize(stream));

        for (auto& stage_ptr : stages_) {
            stage_ptr->postStreamSync(stream);
        }

        stage_timings = profiling_enabled_ ? dag_->collectTimings() : std::vector<StageTimingResult>{};

        for (size_t i = 0; i < output_buffer_ids_.size(); i++) {
            int buffer_id = output_buffer_ids_[i];
            const auto& buffer_info = dag_->getBufferInfo(buffer_id);
            DAGNode* producer = output_nodes_[i];

            BufferMetadata meta;
            meta.buffer_id = buffer_id;
            meta.allocated_size = buffer_info.size;
            meta.producer = producer;
            meta.output_index = buffer_info.producer_output_index;

            auto output_names = producer->stage->getOutputNames();
            int output_idx = buffer_info.producer_output_index;
            meta.name = (output_idx >= 0 && output_idx < static_cast<int>(output_names.size()))
                        ? output_names[output_idx]
                        : "output";

            auto sizes_by_name = producer->stage->getActualOutputSizesByName();
            auto it = sizes_by_name.find(meta.name);
            meta.actual_size = (it != sizes_by_name.end()) ? it->second : buffer_info.size;

            buffer_metadata_.push_back(meta);
        }

        // OWNERSHIP CONTRACT: pool-owned, caller must NOT cudaFree().
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
        dag_->reset(stream);
        buffer_metadata_.clear();
        input_size_ = 0;
        was_compressed_ = false;
        throw;
    }

    is_compressed_  = true;
    was_compressed_ = true;

    auto t_host_end = std::chrono::steady_clock::now();

    float host_ms = std::chrono::duration<float, std::milli>(t_host_end - t_host_start).count();
    float dag_ms  = std::chrono::duration<float, std::milli>(t_dag_end  - t_dag_start ).count();

    if (profiling_enabled_) {
        PipelinePerfResult r;
        r.is_compress     = true;
        r.host_elapsed_ms = host_ms;
        r.dag_elapsed_ms  = dag_ms;
        r.input_bytes     = input_size_;
        r.output_bytes    = *output_size;
        r.stages          = std::move(stage_timings);
        r.levels = buildLevelTimings(r.stages);
        last_perf_result_ = std::move(r);
    }

    FZ_LOG(INFO, "Compress complete: %zu -> %zu bytes (host=%.2f ms, dag=%.2f ms, DAG=%.2f GB/s, pipeline=%.2f GB/s)",
           input_size_, *output_size, host_ms, dag_ms,
           profiling_enabled_ ? last_perf_result_.throughput_gbs() : 0.0f,
           profiling_enabled_ ? last_perf_result_.pipeline_throughput_gbs() : 0.0f);
}

void Pipeline::decompress(
    const void* d_input,
    size_t      input_size,
    void**      d_output,
    size_t*     output_size,
    cudaStream_t stream
) {
    if (!is_finalized_) {
        throw std::runtime_error("Pipeline not finalized");
    }
    if (buffer_metadata_.empty()) {
        throw std::runtime_error("decompress() requires compress() to have been called first");
    }

    auto t_host_start = std::chrono::steady_clock::now();
    FZ_LOG(INFO, "Decompressing");

    // Determine which device pointer feeds each compressed buffer in the
    // inverse DAG.
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

    for (auto& s : stages_) {
        s->saveState();
        s->setInverse(true);
    }

    PipelineOutputMap po_map;
    for (const auto& meta : buffer_metadata_) {
        auto it   = compressed_ptrs.find(meta.buffer_id);
        void* ptr = (it != compressed_ptrs.end())
            ? it->second
            : dag_->getBuffer(meta.buffer_id);
        po_map[meta.buffer_id] = {ptr, meta.actual_size};
    }

    // Single source — use the recorded input size.
    Stage* src_stage = input_nodes_[0]->stage;
    size_t src_sz = (source_input_sizes_.size() > 0 && source_input_sizes_[0] > 0)
                    ? source_input_sizes_[0]
                    : input_size_;
    std::unordered_map<Stage*, size_t> source_sizes = {{src_stage, src_sz}};

    // ── Inverse DAG: reuse cached instance or build and cache ─────────────────
    bool cache_valid = (inv_cache_ != nullptr);
    if (cache_valid) {
        auto it = inv_cache_->source_sizes.find(src_stage);
        if (it == inv_cache_->source_sizes.end() || it->second != src_sz) {
            cache_valid = false;
            FZ_LOG(DEBUG, "decompress: inv DAG cache invalidated (source size changed)");
        }
    }

    if (!cache_valid) {
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

        FZ_LOG(DEBUG, "decompress: built and cached inverse DAG (%zu ext buffers mapped)",
               inv_cache_->fwd_to_inv_ext_buf.size());
    } else {
        for (const auto& [fwd_buf_id, inv_buf_id] : inv_cache_->fwd_to_inv_ext_buf) {
            auto it = po_map.find(fwd_buf_id);
            if (it != po_map.end()) {
                inv_cache_->inv_dag->setExternalPointer(inv_buf_id, it->second.first);
                inv_cache_->inv_dag->updateBufferSize(inv_buf_id, it->second.second);
            }
        }
        inv_cache_->inv_dag->reset(stream);
        inv_cache_->inv_dag->enableProfiling(profiling_enabled_);

        FZ_LOG(DEBUG, "decompress: reusing cached inverse DAG");
    }

    CompressionDAG& inv_dag        = *inv_cache_->inv_dag;
    const auto&     inv_result_map = inv_cache_->inv_result_map;

    // ── Pre-allocate output buffer and wire as external DAG output ────────────
    if (pool_managed_decomp_) {
        for (void* p : d_decomp_outputs_) {
            if (p && mem_pool_) mem_pool_->free(p, stream);
        }
        d_decomp_outputs_.clear();
    }

    auto buf_it = inv_result_map.find(src_stage);
    if (buf_it == inv_result_map.end()) {
        for (auto& s : stages_) { s->setInverse(false); s->restoreState(); }
        throw std::runtime_error(
            "decompress: no inverse result buffer for source stage '" +
            src_stage->getName() + "'");
    }
    int    res_buf_id  = buf_it->second;
    size_t actual_size = inv_dag.getBufferSize(res_buf_id);

    void* d_final = nullptr;
    if (pool_managed_decomp_) {
        if (actual_size > 0) {
            d_final = mem_pool_->allocate(actual_size, stream, "decomp_output", /*persistent=*/true);
            if (!d_final) {
                for (auto& s : stages_) { s->setInverse(false); s->restoreState(); }
                throw std::runtime_error(
                    "Pool allocation for decompress output failed (" +
                    std::to_string(actual_size) + " bytes); pool may be exhausted");
            }
            d_decomp_outputs_.push_back(d_final);
        }
    } else {
        cudaError_t err = cudaMalloc(&d_final, actual_size);
        if (err != cudaSuccess) {
            for (auto& s : stages_) { s->setInverse(false); s->restoreState(); }
            throw std::runtime_error(
                "cudaMalloc for decompress output failed (" +
                std::to_string(actual_size) + " bytes): " +
                cudaGetErrorString(err));
        }
    }
    inv_dag.setExternalPointer(res_buf_id, d_final);

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

    // Refine output size from postStreamSync.
    auto post_sizes = src_stage->getActualOutputSizesByName();
    auto post_names = src_stage->getOutputNames();
    if (!post_names.empty() && post_sizes.count(post_names[0])) {
        actual_size = post_sizes.at(post_names[0]);
    }

    inv_dag.reset(stream);

    for (auto& s : stages_) {
        s->setInverse(false);
        s->restoreState();
    }

    *d_output    = d_final;
    *output_size = actual_size;

    // If compress() transparently padded the input, trim the reported output
    // size back to the original (unpadded) byte count.
    if (original_input_size_ > 0 && *output_size > original_input_size_)
        *output_size = original_input_size_;

    auto t_host_end = std::chrono::steady_clock::now();
    float host_ms = std::chrono::duration<float, std::milli>(t_host_end - t_host_start).count();
    float dag_ms  = std::chrono::duration<float, std::milli>(t_dag_end  - t_dag_start ).count();

    if (profiling_enabled_) {
        PipelinePerfResult r;
        r.is_compress     = false;
        r.host_elapsed_ms = host_ms;
        r.dag_elapsed_ms  = dag_ms;
        r.input_bytes     = input_size;
        r.output_bytes    = *output_size;
        r.stages          = std::move(stage_timings);
        r.levels          = buildLevelTimings(r.stages);
        last_perf_result_ = std::move(r);
    }

    FZ_LOG(INFO, "Decompress complete: %zu bytes (host=%.2f ms, dag=%.2f ms, DAG=%.2f GB/s, pipeline=%.2f GB/s)",
           *output_size, host_ms, dag_ms,
           profiling_enabled_ ? last_perf_result_.throughput_gbs() : 0.0f,
           profiling_enabled_ ? last_perf_result_.pipeline_throughput_gbs() : 0.0f);
}

// ── getMaxCompressedSize ──────────────────────────────────────────────────────

size_t Pipeline::getMaxCompressedSize(size_t input_bytes) const {
    if (!is_finalized_) {
        throw std::runtime_error(
            "getMaxCompressedSize() requires a finalized pipeline");
    }

    // Walk the DAG level-by-level, propagating input_bytes through each stage's
    // estimateOutputSizes(). The final value is the worst-case compressed size.
    // This mirrors propagateBufferSizes() but operates on a local copy so the
    // actual DAG buffer state is unchanged.
    size_t current = input_bytes;
    for (const auto& level_nodes : dag_->getLevels()) {
        size_t level_max = 0;
        for (auto* node : level_nodes) {
            // All nodes at a given level see the same "current" bytes
            // (single-source pipeline).
            auto est = node->stage->estimateOutputSizes({current});
            for (size_t sz : est) level_max = std::max(level_max, sz);
        }
        if (level_max > 0) current = level_max;
    }

    // Add 5% margin for stage-internal size-tracking rounding and
    // concat-format headers (4B num_bufs + 8B per output × n_outputs).
    const size_t header_overhead =
        sizeof(uint32_t) + output_buffer_ids_.size() * sizeof(uint64_t);
    return static_cast<size_t>(current * 1.05) + header_overhead;
}

// ── compress (user-owned output) ──────────────────────────────────────────────

void Pipeline::compress(
    const void* d_input,
    size_t      input_size,
    void*       d_output_buf,
    size_t      output_buf_capacity,
    size_t*     actual_output_size,
    cudaStream_t stream
) {
    if (graph_mode_enabled_) {
        throw std::runtime_error(
            "compress() with user-owned output is incompatible with CUDA Graph mode. "
            "Use the pool-owned overload (void** d_output) when graph mode is enabled.");
    }
    if (d_output_buf == nullptr) {
        throw std::runtime_error(
            "compress(): d_output_buf must not be null for user-owned output");
    }

    // Run the standard pool-owned compress to get the compressed data
    // into the pool buffer, then D2D-copy into the caller's buffer.
    void*  d_pool_out  = nullptr;
    size_t pool_out_sz = 0;
    compress(d_input, input_size, &d_pool_out, &pool_out_sz, stream);

    if (pool_out_sz > output_buf_capacity) {
        throw std::runtime_error(
            "compress() user-owned output: actual compressed size (" +
            std::to_string(pool_out_sz) +
            " bytes) exceeds the provided buffer capacity (" +
            std::to_string(output_buf_capacity) +
            " bytes). Allocate a larger buffer or use "
            "getMaxCompressedSize() for a guaranteed upper bound.");
    }

    FZ_CUDA_CHECK(cudaMemcpyAsync(d_output_buf, d_pool_out, pool_out_sz,
                                  cudaMemcpyDeviceToDevice, stream));
    FZ_CUDA_CHECK(cudaStreamSynchronize(stream));
    *actual_output_size = pool_out_sz;

    FZ_LOG(INFO, "compress (user-owned output): copied %zu bytes to caller buffer", pool_out_sz);
}

// ── decompress (user-owned output buffer) ────────────────────────────────────

void Pipeline::decompress(
    const void* d_input,
    size_t      input_size,
    void*       d_output_buf,
    size_t      output_buf_capacity,
    size_t*     actual_output_size,
    cudaStream_t stream
) {
    if (d_output_buf == nullptr) {
        throw std::runtime_error(
            "decompress(): d_output_buf must not be null for user-owned output");
    }

    // Run the standard decompress into a temporary pool/malloc'd buffer,
    // then D2D-copy into the caller's buffer and free the temporary.
    const bool saved_pool = pool_managed_decomp_;
    pool_managed_decomp_ = false;    // always get a fresh cudaMalloc'd pointer here

    void*  d_tmp  = nullptr;
    size_t tmp_sz = 0;
    try {
        decompress(d_input, input_size, &d_tmp, &tmp_sz, stream);
    } catch (...) {
        pool_managed_decomp_ = saved_pool;
        throw;
    }
    pool_managed_decomp_ = saved_pool;

    if (tmp_sz > output_buf_capacity) {
        cudaFree(d_tmp);
        throw std::runtime_error(
            "decompress() user-owned output: actual decompressed size (" +
            std::to_string(tmp_sz) +
            " bytes) exceeds the provided buffer capacity (" +
            std::to_string(output_buf_capacity) +
            " bytes). Allocate a larger buffer (the original uncompressed "
            "size is available from the file header or your compress() call).");
    }

    FZ_CUDA_CHECK(cudaMemcpyAsync(d_output_buf, d_tmp, tmp_sz,
                                  cudaMemcpyDeviceToDevice, stream));
    FZ_CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaFree(d_tmp);
    *actual_output_size = tmp_sz;

    FZ_LOG(INFO, "decompress (user-owned output): copied %zu bytes to caller buffer", tmp_sz);
}

} // namespace fz
