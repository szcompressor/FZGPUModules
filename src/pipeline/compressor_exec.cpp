// compressor_exec.cpp — in-memory compress / decompress execution
#include "pipeline/compressor.h"
#include "pipeline_utils.h"
#include "dag_event_timer.h"
#include "log.h"
#include "cuda_check.h"
#include <chrono>
#include <stdexcept>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

namespace fz {

// =====

std::pair<const void*, size_t> Pipeline::prepareInputSource(
    const void* d_input, size_t input_size, cudaStream_t stream)
{
    original_input_size_ = 0;

    if (graph_captured_) {
        original_input_size_ = (input_alignment_bytes_ > 1 &&
                                 input_size % input_alignment_bytes_ != 0)
                               ? input_size : 0;
        FZ_CUDA_CHECK(cudaMemcpyAsync(d_graph_input_.ptr, d_input, input_size,
                                      cudaMemcpyDeviceToDevice, stream));
        if (input_size < d_graph_input_size_) {
            FZ_CUDA_CHECK(cudaMemsetAsync(
                static_cast<uint8_t*>(d_graph_input_.ptr) + input_size,
                0, d_graph_input_size_ - input_size, stream));
        }
        return {d_graph_input_.ptr, d_graph_input_size_};
    }

    if (input_alignment_bytes_ > 1 && input_size % input_alignment_bytes_ != 0) {
        const size_t padded = ((input_size + input_alignment_bytes_ - 1)
                               / input_alignment_bytes_) * input_alignment_bytes_;
        if (padded > d_pad_buf_.capacity) {
            if (!d_pad_buf_.allocate(mem_pool_.get(), padded, stream,
                                     "pipeline_input_pad", /*persistent=*/true)) {
                throw std::runtime_error(
                    "Failed to allocate pipeline input pad buffer (" +
                    std::to_string(padded) + " bytes); pool may be exhausted");
            }
        }
        FZ_CUDA_CHECK(cudaMemcpyAsync(d_pad_buf_.ptr, d_input, input_size,
                                      cudaMemcpyDeviceToDevice, stream));
        FZ_CUDA_CHECK(cudaMemsetAsync(static_cast<uint8_t*>(d_pad_buf_.ptr) + input_size,
                                      0, padded - input_size, stream));
        FZ_LOG(INFO, "Input padded: %zu → %zu bytes (+%zu bytes for %zu-byte chunk alignment)",
               input_size, padded, padded - input_size, input_alignment_bytes_);
        original_input_size_ = input_size;
        return {d_pad_buf_.ptr, padded};
    }

    return {d_input, input_size};
}

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
        throw std::runtime_error("compress(): null device pointer passed as input");
    }

    if (was_compressed_) {
        dag_->reset(stream);
        was_compressed_ = false;
    }

    if (input_size_hint_ > 0 && input_size > input_size_hint_) {
        throw std::runtime_error(
            "compress(): input size (" + std::to_string(input_size) +
            " bytes) exceeds the finalize-time buffer size hint (" +
            std::to_string(input_size_hint_) + " bytes); "
            "re-construct the pipeline with a larger input size hint");
    }

    auto [d_source, source_sz] = prepareInputSource(d_input, input_size, stream);

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

    // Device wall time via CUDA events (profiling path only); host markers are a
    // fallback for the log line when profiling is off.
    DagEventTimer dag_timer(profiling_enabled_);
    std::vector<StageTimingResult> stage_timings;
    auto t_dag_start = std::chrono::steady_clock::now();
    auto t_dag_end   = t_dag_start;
    try {
        dag_timer.recordStart(stream);
        if (graph_captured_) {
            FZ_CUDA_CHECK(cudaGraphLaunch(graph_exec_, stream));
        } else {
            dag_->execute(stream);
        }
        dag_timer.recordStop(stream);
        t_dag_end = std::chrono::steady_clock::now();

        FZ_CUDA_CHECK(cudaStreamSynchronize(stream));

        for (auto& stage_ptr : stages_) {
            stage_ptr->postStreamSync(stream);
        }

        // Per-stage CUDA-event timing is unavailable during a graph replay: the
        // start/completion events are recorded by nodes baked into the captured
        // graph, and cudaEventElapsedTime() across graph-recorded events is not
        // supported (returns cudaErrorInvalidValue on every node).  The whole-
        // pipeline dag_elapsed_ms from the outer DagEventTimer (events recorded
        // on `stream` outside the graph) is still valid.  Skip the per-stage pass
        // here so we never issue the failing query.
        stage_timings = (profiling_enabled_ && !graph_captured_)
                            ? dag_->collectTimings()
                            : std::vector<StageTimingResult>{};

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
    // dag_ms is true device wall time when profiling (CUDA events); otherwise a
    // rough host-side enqueue estimate just for the log line.
    float dag_ms  = profiling_enabled_
        ? dag_timer.elapsedMs()
        : std::chrono::duration<float, std::milli>(t_dag_end - t_dag_start).count();

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

void Pipeline::buildStaticBufferMetadata() {
    buffer_metadata_.clear();
    buffer_metadata_.reserve(output_buffer_ids_.size());
    for (size_t i = 0; i < output_buffer_ids_.size(); i++) {
        int         buffer_id   = output_buffer_ids_[i];
        const auto& buffer_info = dag_->getBufferInfo(buffer_id);
        DAGNode*    producer    = output_nodes_[i];

        BufferMetadata meta;
        meta.buffer_id      = buffer_id;
        meta.allocated_size = buffer_info.size;
        meta.actual_size    = buffer_info.size;  // placeholder; real size read from blob
        meta.producer       = producer;
        meta.output_index   = buffer_info.producer_output_index;

        auto output_names = producer->stage->getOutputNames();
        int  output_idx   = buffer_info.producer_output_index;
        meta.name = (output_idx >= 0 && output_idx < static_cast<int>(output_names.size()))
                    ? output_names[output_idx] : "output";

        buffer_metadata_.push_back(std::move(meta));
    }
}

std::vector<size_t> Pipeline::readConcatSegmentSizes(
    const void* d_blob, size_t n, cudaStream_t stream) const
{
    const size_t hdr = sizeof(uint32_t) + n * sizeof(uint64_t);
    std::vector<uint8_t> host(hdr);
    FZ_CUDA_CHECK(cudaMemcpyAsync(host.data(), d_blob, hdr,
                                  cudaMemcpyDeviceToHost, stream));
    FZ_CUDA_CHECK(cudaStreamSynchronize(stream));

    uint32_t count = 0;
    std::memcpy(&count, host.data(), sizeof(uint32_t));
    if (count != n) {
        throw std::runtime_error(
            "decompress: concat header segment count (" + std::to_string(count) +
            ") does not match this pipeline's output count (" + std::to_string(n) +
            ") — blob was produced by a different pipeline");
    }

    std::vector<size_t> sizes(n);
    const uint8_t* p = host.data() + sizeof(uint32_t);
    for (size_t i = 0; i < n; i++) {
        uint64_t v = 0;
        std::memcpy(&v, p, sizeof(uint64_t));
        p += sizeof(uint64_t);
        sizes[i] = static_cast<size_t>(v);
    }
    return sizes;
}

void Pipeline::prepareInverse(size_t uncompressed_size) {
    if (!is_finalized_) {
        throw std::runtime_error("prepareInverse() requires a finalized pipeline");
    }
    if (uncompressed_size == 0) {
        throw std::runtime_error("prepareInverse() requires a non-zero uncompressed size");
    }
    buildStaticBufferMetadata();
    source_input_sizes_.assign(1, uncompressed_size);
    input_size_ = uncompressed_size;
    FZ_LOG(DEBUG, "prepareInverse: pipeline ready for external-blob decompress "
                  "(%zu output buffer(s), uncompressed=%zu bytes)",
           buffer_metadata_.size(), uncompressed_size);
}

void Pipeline::decompressCore(
    const void* d_input,
    size_t      input_size,
    void*       caller_output,
    size_t      caller_capacity,
    bool        synchronize,
    void**      d_output,
    size_t*     output_size,
    cudaStream_t stream
) {
    if (!is_finalized_) {
        throw std::runtime_error("Pipeline not finalized");
    }
    if (buffer_metadata_.empty()) {
        throw std::runtime_error(
            "decompress() requires compress() or prepareInverse(uncompressed_size) "
            "to have been called first");
    }

    auto t_host_start = std::chrono::steady_clock::now();
    FZ_LOG(INFO, "Decompressing");

    // Per-segment compressed sizes. For an external blob (d_input != nullptr) these
    // are read from the blob's own self-describing concat header — authoritative and
    // independent of any prior compress(), so blobs of differing sizes decode
    // correctly and prepareInverse()'s placeholder sizes are never used. The
    // single-segment case needs no header: the whole blob is the one segment. Only
    // the live-DAG path (d_input == nullptr, immediately after compress()) falls
    // back to the freshly-produced buffer_metadata_ sizes.
    const size_t n_seg = buffer_metadata_.size();
    std::vector<size_t> seg_sizes(n_seg);
    if (d_input != nullptr && needs_concat_) {
        seg_sizes = readConcatSegmentSizes(d_input, n_seg, stream);
    } else if (d_input != nullptr) {
        seg_sizes[0] = input_size;  // single segment == whole blob
    } else {
        for (size_t i = 0; i < n_seg; i++) seg_sizes[i] = buffer_metadata_[i].actual_size;
    }

    // Map each compressed buffer ID to a device pointer from d_input or the live DAG.
    std::unordered_map<int, void*> compressed_ptrs;
    if (d_input != nullptr) {
        if (!needs_concat_) {
            compressed_ptrs[buffer_metadata_[0].buffer_id] = const_cast<void*>(d_input);
        } else {
            size_t byte_offset = ConcatLayout::headerSize(n_seg);
            for (size_t i = 0; i < n_seg; i++) {
                compressed_ptrs[buffer_metadata_[i].buffer_id] =
                    static_cast<uint8_t*>(const_cast<void*>(d_input)) + byte_offset;
                byte_offset += ConcatLayout::slotSize(seg_sizes[i]);
            }
        }
    } else {
        for (const auto& meta : buffer_metadata_)
            compressed_ptrs[meta.buffer_id] = dag_->getBuffer(meta.buffer_id);
    }

    for (auto& s : stages_) {
        s->saveState();
        s->setInverse(true);
    }

    PipelineOutputMap po_map;
    for (size_t i = 0; i < n_seg; i++) {
        const auto& meta = buffer_metadata_[i];
        auto it = compressed_ptrs.find(meta.buffer_id);
        po_map[meta.buffer_id] = {
            (it != compressed_ptrs.end()) ? it->second : dag_->getBuffer(meta.buffer_id),
            seg_sizes[i]
        };
    }

    Stage* src_stage = input_nodes_[0]->stage;
    size_t src_sz    = (source_input_sizes_.size() > 0 && source_input_sizes_[0] > 0)
                       ? source_input_sizes_[0] : input_size_;

    buildOrReuseInvCache(po_map, src_stage, src_sz, stream);

    CompressionDAG& inv_dag        = *inv_cache_->inv_dag;
    const auto&     inv_result_map = inv_cache_->inv_result_map;

    // Reclaim prior pool-managed decompress outputs only when WE allocate the
    // output below (caller-supplied output is caller-owned and not pool-tracked).
    if (caller_output == nullptr && pool_managed_decomp_) {
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

    // Resolve the output pointer.  A caller-supplied buffer is written into
    // directly — no cudaMalloc / D2D copy / cudaFree (all device-wide barriers
    // that would prevent cross-stream overlap).  Otherwise allocate as before.
    void* d_final = nullptr;
    if (caller_output != nullptr) {
        if (actual_size > caller_capacity) {
            for (auto& s : stages_) { s->setInverse(false); s->restoreState(); }
            throw std::runtime_error(
                "decompress() user-owned output: actual decompressed size (" +
                std::to_string(actual_size) + " bytes) exceeds the provided buffer "
                "capacity (" + std::to_string(caller_capacity) + " bytes). Allocate a "
                "larger buffer (the original uncompressed size is available from the "
                "file header or your compress() call).");
        }
        d_final = caller_output;
    } else if (pool_managed_decomp_) {
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

    const bool do_profile = profiling_enabled_ && synchronize;
    DagEventTimer dag_timer(do_profile);
    auto t_dag_start = std::chrono::steady_clock::now();
    dag_timer.recordStart(stream);
    inv_dag.execute(stream);
    dag_timer.recordStop(stream);
    auto t_dag_end = std::chrono::steady_clock::now();

    std::vector<StageTimingResult> stage_timings;
    if (synchronize) {
        // Correctness/convenience barrier: result is ready and stage metadata
        // (sizes, counts) can be read back.  Skipped on the async path, where the
        // caller owns synchronization and the planned size is authoritative.
        FZ_CUDA_CHECK(cudaStreamSynchronize(stream));

        for (auto& stage_ptr : stages_) {
            stage_ptr->postStreamSync(stream);
        }

        if (do_profile) stage_timings = inv_dag.collectTimings();

        // Refine output size from postStreamSync.
        auto post_sizes = src_stage->getActualOutputSizesByName();
        auto post_names = src_stage->getOutputNames();
        if (!post_names.empty() && post_sizes.count(post_names[0])) {
            actual_size = post_sizes.at(post_names[0]);
        }
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
    // True device wall time via CUDA events when profiling; host fallback otherwise.
    // dag_timer events are only valid after a stream sync, so device time is only
    // meaningful on the synchronous path.
    float dag_ms  = do_profile
        ? dag_timer.elapsedMs()
        : std::chrono::duration<float, std::milli>(t_dag_end - t_dag_start).count();

    if (do_profile) {
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

    FZ_LOG(INFO, "Decompress %s: %zu bytes (host=%.2f ms, dag=%.2f ms)",
           synchronize ? "complete" : "enqueued", *output_size, host_ms, dag_ms);
}

// ── Public decompress overloads (thin wrappers over decompressCore) ───────────

void Pipeline::decompress(
    const void* d_input,
    size_t      input_size,
    void**      d_output,
    size_t*     output_size,
    cudaStream_t stream
) {
    decompressCore(d_input, input_size, /*caller_output=*/nullptr, /*caller_capacity=*/0,
                   /*synchronize=*/true, d_output, output_size, stream);
}

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
    // Inject the caller buffer as the inverse result pointer — no temp alloc,
    // no D2D copy, no free.  Still synchronous (result ready on return).
    void* result = nullptr;
    decompressCore(d_input, input_size, d_output_buf, output_buf_capacity,
                   /*synchronize=*/true, &result, actual_output_size, stream);
}

void Pipeline::decompressInto(
    const void* d_input,
    size_t      input_size,
    void*       d_output_buf,
    size_t      output_buf_capacity,
    size_t*     actual_output_size,
    cudaStream_t stream
) {
    if (d_output_buf == nullptr) {
        throw std::runtime_error("decompressInto(): d_output_buf must not be null");
    }
    if (strategy_ != MemoryStrategy::PREALLOCATE) {
        throw std::runtime_error(
            "decompressInto() requires PREALLOCATE memory strategy (internal inverse "
            "buffers must be allocated once, not per call, for stream-concurrent decode)");
    }
    void* result = nullptr;
    decompressCore(d_input, input_size, d_output_buf, output_buf_capacity,
                   /*synchronize=*/false, &result, actual_output_size, stream);
}

void Pipeline::buildOrReuseInvCache(
    const PipelineOutputMap& po_map,
    Stage*       src_stage,
    size_t       src_sz,
    cudaStream_t stream)
{
    std::unordered_map<Stage*, size_t> source_sizes = {{src_stage, src_sz}};

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
                        fwd_to_inv_ext_buf[std::stoi(info.tag.substr(8))] = buf_id;
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

    // Add 5% margin for stage-internal size-tracking rounding 
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

// ── Explicit-ownership span API ──────────────────────────────────────────────
//
// Thin wrappers over the pointer overloads: the ownership contract moves into
// the return type, the behavior is unchanged. See memory/public_api_evolution.md
// and docs/api_reference.md.

BorrowedDeviceBuffer Pipeline::compress(ConstDeviceSpan input, cudaStream_t stream) {
    void*  d_out = nullptr;
    size_t out_sz = 0;
    compress(input.data, input.bytes, &d_out, &out_sz, stream);
    return BorrowedDeviceBuffer(d_out, out_sz);
}

size_t Pipeline::compressInto(ConstDeviceSpan input, DeviceSpan output, cudaStream_t stream) {
    size_t written = 0;
    compress(input.data, input.bytes, output.data, output.bytes, &written, stream);
    return written;
}

BorrowedDeviceBuffer Pipeline::decompressBorrowed(ConstDeviceSpan input, cudaStream_t stream) {
    // Always borrow, whatever setPoolManagedDecompOutput() says: the ownership
    // is stated by the return type, not by pipeline state.
    const bool prev = pool_managed_decomp_;
    pool_managed_decomp_ = true;
    void*  d_out = nullptr;
    size_t out_sz = 0;
    try {
        decompress(input.data, input.bytes, &d_out, &out_sz, stream);
    } catch (...) {
        pool_managed_decomp_ = prev;
        throw;
    }
    pool_managed_decomp_ = prev;
    return BorrowedDeviceBuffer(d_out, out_sz);
}

OwnedDeviceBuffer Pipeline::decompressOwned(ConstDeviceSpan input, cudaStream_t stream) {
    // Always own, whatever setPoolManagedDecompOutput() says.
    const bool prev = pool_managed_decomp_;
    pool_managed_decomp_ = false;
    void*  d_out = nullptr;
    size_t out_sz = 0;
    try {
        decompress(input.data, input.bytes, &d_out, &out_sz, stream);
    } catch (...) {
        pool_managed_decomp_ = prev;
        throw;
    }
    pool_managed_decomp_ = prev;

    // Record the device the allocation belongs to so the deleter frees it there
    // even if the caller switches devices before the buffer dies.
    int device = 0;
    FZ_CUDA_CHECK(cudaGetDevice(&device));
    return OwnedDeviceBuffer(d_out, out_sz, device);
}

size_t Pipeline::decompressInto(ConstDeviceSpan input, DeviceSpan output, cudaStream_t stream) {
    size_t written = 0;
    decompress(input.data, input.bytes, output.data, output.bytes, &written, stream);
    return written;
}

size_t Pipeline::decompressIntoAsync(ConstDeviceSpan input, DeviceSpan output, cudaStream_t stream) {
    size_t planned = 0;
    decompressInto(input.data, input.bytes, output.data, output.bytes, &planned, stream);
    return planned;
}

} // namespace fz
