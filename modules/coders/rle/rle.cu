#include "coders/rle/rle.h"
#include "log.h"
#include "backend/api.h"
#include "backend/cub.h"
#include "backend/algorithms.h"
#include "mem/mempool.h"
#include "cuda_check.h"

namespace fz {

/**
 * RLE Decompression Kernel (Inverse)
 *
 * Expands (value, run_length) pairs back to original sequence.
 * Each thread handles one run, writing multiple output values.
 *
 * Input format: [num_runs] [value1, count1, value2, count2, ...]
 * Output: Expanded sequence [value1×count1, value2×count2, ...]
 */
template<typename T>
__global__ void rle_decompress_kernel(
    const T* __restrict__ compressed_values,     // [num_runs]
    const uint32_t* __restrict__ run_lengths,    // [num_runs]
    const uint32_t* __restrict__ run_offsets,    // [num_runs] prefix sum of run_lengths
    T* __restrict__ output,
    const uint32_t num_runs
) {
    uint32_t run_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (run_idx < num_runs) {
        T value = compressed_values[run_idx];
        uint32_t start = (run_idx == 0) ? 0 : run_offsets[run_idx - 1];
        uint32_t end = run_offsets[run_idx];

        // Write this value 'count' times
        for (uint32_t i = start; i < end; i++) {
            output[i] = value;
        }
    }
}

/**
 * RLE Compression Kernel (Forward) - Phase 1
 * Mark positions where the value changes (run boundaries).
 */
template<typename T>
__global__ void rle_mark_boundaries_kernel(
    const T* __restrict__ input,
    uint8_t* __restrict__ is_boundary,
    const size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        is_boundary[idx] = (idx == 0 || input[idx] != input[idx - 1]) ? 1 : 0;
    }
}

/**
 * Scatter each boundary element's position into a compact positions array.
 * boundary_scan is the inclusive prefix sum of is_boundary; element i
 * belongs to run (boundary_scan[i] - 1).
 */
__global__ void scatter_boundary_positions_kernel(
    const uint8_t* __restrict__ is_boundary,
    const uint32_t* __restrict__ boundary_scan,
    uint32_t* __restrict__ boundary_positions,
    const size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && is_boundary[idx]) {
        boundary_positions[boundary_scan[idx] - 1] = static_cast<uint32_t>(idx);
    }
}

/**
 * Extract run values and lengths into persistent scratch arrays.
 *
 * Takes d_num_runs as a device pointer (boundary_scan[n-1]) so the host
 * never needs to read num_runs before launching — the kernel bounds-checks
 * against *d_num_runs itself.  This eliminates the D2H sync that blocked
 * the GPU pipeline in the previous implementation and makes this kernel
 * CUDA Graph-capturable.
 */
template<typename T>
__global__ void rle_extract_runs_kernel(
    const T* __restrict__ input,
    const uint32_t* __restrict__ boundary_positions,
    const uint32_t* __restrict__ d_num_runs,  // device pointer — boundary_scan[n-1]
    T* __restrict__ values_scratch,
    uint32_t* __restrict__ lengths_scratch,
    const size_t n
) {
    const uint32_t num_runs = *d_num_runs;
    const uint32_t run_id   = blockIdx.x * blockDim.x + threadIdx.x;
    if (run_id >= num_runs) return;

    const uint32_t start_pos = boundary_positions[run_id];
    const uint32_t end_pos   = (run_id + 1 < num_runs)
                                   ? boundary_positions[run_id + 1]
                                   : static_cast<uint32_t>(n);

    values_scratch[run_id]  = input[start_pos];
    lengths_scratch[run_id] = end_pos - start_pos;
}

/**
 * Pack scratch arrays into the compact output wire format:
 *   [num_runs: u32][pad to alignof(T)][values: T×num_runs, 4B-aligned][run_lengths: u32×num_runs]
 *
 * Uses *d_num_runs (device pointer) for all layout arithmetic — no host
 * involvement required.  Grid is launched at the worst-case size (n
 * elements); threads past num_runs return immediately.
 *
 * The header write (i==0) and the values/lengths writes touch disjoint
 * byte ranges and do not race:
 *   header   → [0, 4)
 *   values   → [values_offset, values_offset + num_runs*sizeof(T))
 *   lengths  → [values_offset + values_aligned, values_offset + values_aligned + num_runs*4)
 * where values_offset = rleValuesOffset<T>() (4 for T ≤ 4 bytes, 8 for 8-byte T
 * — required so the reinterpret_cast<T*> below never issues a misaligned wide
 * load for uint64_t/int64_t).
 */
template<typename T>
__global__ void rle_pack_kernel(
    const T* __restrict__ values_scratch,
    const uint32_t* __restrict__ lengths_scratch,
    uint8_t* __restrict__ output_base,
    const uint32_t* __restrict__ d_num_runs,
    const size_t n
) {
    constexpr uint32_t values_offset = static_cast<uint32_t>(rleValuesOffset<T>());
    const uint32_t num_runs     = *d_num_runs;
    const uint32_t values_bytes = num_runs * static_cast<uint32_t>(sizeof(T));
    const uint32_t values_aligned = (values_bytes + 3u) & ~3u;
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i == 0) {
        *reinterpret_cast<uint32_t*>(output_base) = num_runs;
        // Zero the header-to-values pad (present when alignof(T) > 4) and the
        // alignment padding between the values section and the lengths section.
        // Neither range is written by the per-run threads below, leaving them
        // uninitialized.  Zero them here so the full [0, actual_output_size_)
        // range is always initialized.
        for (uint32_t b = sizeof(uint32_t); b < values_offset; b++)
            output_base[b] = 0;
        uint8_t* pad_base = output_base + values_offset + values_bytes;
        for (uint32_t b = 0; b < values_aligned - values_bytes; b++)
            pad_base[b] = 0;
    }
    if (i < num_runs) {
        reinterpret_cast<T*>(output_base + values_offset)[i] = values_scratch[i];
        reinterpret_cast<uint32_t*>(output_base + values_offset + values_aligned)[i]
            = lengths_scratch[i];
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Chunked path
//
// One thread block owns one chunk end to end, so the device-wide scan and the
// four-kernel dependency chain of the global path both disappear: encode is a
// single kernel plus a scan over num_chunks (not n) plus a compaction pass.
// ═════════════════════════════════════════════════════════════════════════════

/// Threads per chunk-owning block.  256 keeps the BlockScan cheap while still
/// covering a 16 KB / 2-byte chunk in 32 tile iterations.
static constexpr int RLE_CHUNK_BLOCK = 256;

/**
 * Encode every chunk independently.
 *
 * Block `c` walks its chunk in `RLE_CHUNK_BLOCK`-wide tiles, marking run
 * boundaries and compacting them with a block-local exclusive scan.  Runs are
 * written into the chunk's *worst-case* scratch slot (`c * elems_per_chunk`),
 * so no inter-block coordination is needed; a later pass compacts them.
 *
 * Run lengths need the *next* boundary, which may live in a later tile, so the
 * boundary positions are staged in `starts_scratch` and differenced once the
 * whole chunk has been scanned.
 */
template<typename T>
__global__ void rle_chunk_encode_kernel(
    const T* __restrict__ input,
    const size_t n,
    const uint32_t elems_per_chunk,
    T* __restrict__ values_scratch,
    uint32_t* __restrict__ lengths_scratch,
    uint32_t* __restrict__ starts_scratch,
    uint32_t* __restrict__ runs_per_chunk
) {
    using BlockScanT = cub::BlockScan<uint32_t, RLE_CHUNK_BLOCK>;
    __shared__ typename BlockScanT::TempStorage scan_tmp;
    __shared__ uint32_t s_run_base;

    const uint32_t c     = blockIdx.x;
    const size_t   start = static_cast<size_t>(c) * elems_per_chunk;
    if (start >= n) return;
    const size_t   avail = n - start;
    const uint32_t len   = (avail < static_cast<size_t>(elems_per_chunk))
                               ? static_cast<uint32_t>(avail) : elems_per_chunk;

    if (threadIdx.x == 0) s_run_base = 0;
    __syncthreads();

    for (uint32_t tile = 0; tile < len; tile += RLE_CHUNK_BLOCK) {
        const uint32_t i = tile + threadIdx.x;
        uint32_t flag = 0;
        T        v    = T();
        if (i < len) {
            v    = input[start + i];
            // i == 0 forces a boundary at the chunk head: that is exactly what
            // makes chunks independently decodable, and the CR cost of it.
            flag = (i == 0 || v != input[start + i - 1]) ? 1u : 0u;
        }

        uint32_t pos = 0, tile_runs = 0;
        BlockScanT(scan_tmp).ExclusiveSum(flag, pos, tile_runs);

        if (flag) {
            const uint32_t r = s_run_base + pos;
            values_scratch[start + r] = v;
            starts_scratch[start + r] = i;
        }
        __syncthreads();                        // scan_tmp reuse + s_run_base RAW
        if (threadIdx.x == 0) s_run_base += tile_runs;
        __syncthreads();
    }

    const uint32_t runs = s_run_base;
    if (threadIdx.x == 0) runs_per_chunk[c] = runs;
    for (uint32_t r = threadIdx.x; r < runs; r += RLE_CHUNK_BLOCK) {
        const uint32_t end = (r + 1 < runs) ? starts_scratch[start + r + 1] : len;
        lengths_scratch[start + r] = end - starts_scratch[start + r];
    }
}

/**
 * Finish the chunk-offset table in the output header: write `num_chunks` and
 * the total run count that terminates the exclusive-scan offsets.
 */
__global__ void rle_chunk_finalize_offsets_kernel(
    uint8_t* __restrict__ output_base,
    const uint32_t* __restrict__ runs_per_chunk,
    const uint32_t num_chunks
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    uint32_t* hdr = reinterpret_cast<uint32_t*>(output_base);
    hdr[0] = num_chunks;
    // hdr[1 .. num_chunks] hold the exclusive scan written by CUB.
    hdr[1 + num_chunks] = hdr[num_chunks] + runs_per_chunk[num_chunks - 1];
}

/**
 * Compact each chunk's runs out of its worst-case scratch slot into the packed
 * output, and zero the alignment padding.
 *
 * `total_runs` is read from the on-device offset table, so the layout
 * arithmetic never round-trips through the host — the whole forward chunked
 * path stays CUDA Graph-capturable.
 */
template<typename T>
__global__ void rle_chunk_compact_kernel(
    const T* __restrict__ values_scratch,
    const uint32_t* __restrict__ lengths_scratch,
    uint8_t* __restrict__ output_base,
    const uint32_t elems_per_chunk,
    const uint32_t num_chunks,
    const uint32_t values_offset
) {
    const uint32_t* offsets    = reinterpret_cast<const uint32_t*>(output_base) + 1;
    const uint32_t  total_runs = offsets[num_chunks];
    const uint32_t  values_bytes   = total_runs * static_cast<uint32_t>(sizeof(T));
    const uint32_t  values_aligned = (values_bytes + 3u) & ~3u;

    T*        out_values  = reinterpret_cast<T*>(output_base + values_offset);
    uint32_t* out_lengths = reinterpret_cast<uint32_t*>(
        output_base + values_offset + values_aligned);

    const uint32_t c = blockIdx.x;
    if (c >= num_chunks) return;

    if (c == 0 && threadIdx.x == 0) {
        // Neither the offset table nor the per-run writes cover these pads.
        for (uint32_t b = (num_chunks + 2) * sizeof(uint32_t); b < values_offset; b++)
            output_base[b] = 0;
        uint8_t* pad = output_base + values_offset + values_bytes;
        for (uint32_t b = 0; b < values_aligned - values_bytes; b++) pad[b] = 0;
    }

    const uint32_t dst  = offsets[c];
    const uint32_t runs = offsets[c + 1] - dst;
    const size_t   src  = static_cast<size_t>(c) * elems_per_chunk;
    for (uint32_t r = threadIdx.x; r < runs; r += RLE_CHUNK_BLOCK) {
        out_values[dst + r]  = values_scratch[src + r];
        out_lengths[dst + r] = lengths_scratch[src + r];
    }
}

/**
 * Expand one chunk per block.  Within-chunk output positions come from a
 * block-local scan of the run lengths, so no prefix sum over the whole stream
 * (and no readback of the total element count) is needed.
 */
template<typename T>
__global__ void rle_chunk_decode_kernel(
    const uint8_t* __restrict__ input_base,
    T* __restrict__ output,
    const size_t n,
    const uint32_t elems_per_chunk,
    const uint32_t num_chunks,
    const uint32_t values_offset
) {
    using BlockScanT = cub::BlockScan<uint32_t, RLE_CHUNK_BLOCK>;
    __shared__ typename BlockScanT::TempStorage scan_tmp;
    __shared__ uint32_t s_out_base;

    const uint32_t* offsets    = reinterpret_cast<const uint32_t*>(input_base) + 1;
    const uint32_t  total_runs = offsets[num_chunks];
    const uint32_t  values_bytes   = total_runs * static_cast<uint32_t>(sizeof(T));
    const uint32_t  values_aligned = (values_bytes + 3u) & ~3u;

    const T* values = reinterpret_cast<const T*>(input_base + values_offset);
    const uint32_t* lengths = reinterpret_cast<const uint32_t*>(
        input_base + values_offset + values_aligned);

    const uint32_t c = blockIdx.x;
    if (c >= num_chunks) return;
    const size_t   start = static_cast<size_t>(c) * elems_per_chunk;
    const size_t   avail = n - start;
    const uint32_t len   = (avail < static_cast<size_t>(elems_per_chunk))
                               ? static_cast<uint32_t>(avail) : elems_per_chunk;

    const uint32_t base = offsets[c];
    const uint32_t runs = offsets[c + 1] - base;

    if (threadIdx.x == 0) s_out_base = 0;
    __syncthreads();

    for (uint32_t tile = 0; tile < runs; tile += RLE_CHUNK_BLOCK) {
        const uint32_t r = tile + threadIdx.x;
        const uint32_t run_len = (r < runs) ? lengths[base + r] : 0u;

        uint32_t pos = 0, tile_total = 0;
        BlockScanT(scan_tmp).ExclusiveSum(run_len, pos, tile_total);

        if (r < runs) {
            const T v = values[base + r];
            uint32_t o = s_out_base + pos;
            for (uint32_t k = 0; k < run_len && o + k < len; k++)
                output[start + o + k] = v;
        }
        __syncthreads();
        if (threadIdx.x == 0) s_out_base += tile_total;
        __syncthreads();
    }
}

// ── Kernel launcher (inverse / decompression) ────────────────────────────────
template<typename T>
void launchRLEDecompressKernel(
    const T* compressed_values,
    const uint32_t* run_lengths,
    const uint32_t* run_offsets,
    T* output,
    uint32_t num_runs,
    cudaStream_t stream
) {
    const int block_size = 256;
    const int grid_size  = (num_runs + block_size - 1) / block_size;
    rle_decompress_kernel<T><<<grid_size, block_size, 0, stream>>>(
        compressed_values, run_lengths, run_offsets, output, num_runs
    );
}

// ── Destructor ────────────────────────────────────────────────────────────────
template<typename T>
RLEStage<T>::~RLEStage() {
    auto dev_free = [&](void* p) {
        if (!p) return;
        if (fwd_from_pool_ && fwd_scratch_pool_) fwd_scratch_pool_->free(p, 0);
        else cudaFree(p);
    };
    dev_free(d_is_boundary_);
    dev_free(d_boundary_scan_);
    dev_free(d_boundary_positions_);
    dev_free(d_values_scratch_);
    dev_free(d_lengths_scratch_);
    if (h_num_runs_) { cudaFreeHost(h_num_runs_); h_num_runs_ = nullptr; }
}

// ── execute() ─────────────────────────────────────────────────────────────────
template<typename T>
void RLEStage<T>::execute(
    cudaStream_t stream,
    MemoryPool* pool,
    const std::vector<void*>& inputs,
    const std::vector<void*>& outputs,
    const std::vector<size_t>& sizes
) {
    if (inputs.empty() || outputs.empty() || sizes.empty()) {
        throw std::runtime_error("RLEStage: Invalid inputs/outputs");
    }

    if (is_inverse_ && isChunked()) {
        // ── DECOMPRESSION, chunked ───────────────────────────────────────────
        // The element count and chunk size both come from the serialized stage
        // header, so nothing has to be read back from the stream.
        const size_t n = cached_num_elements_;
        if (n == 0) { actual_output_sizes_ = {0}; return; }

        const uint32_t epc = elemsPerChunk();
        const uint32_t nc  = static_cast<uint32_t>(numChunks(n * sizeof(T)));
        const uint32_t values_offset =
            static_cast<uint32_t>(rleChunkedValuesOffset<T>(nc));
        FZ_LOG(TRACE, "RLE decode (chunked): %u chunks -> %zu elems", nc, n);

        rle_chunk_decode_kernel<T><<<nc, RLE_CHUNK_BLOCK, 0, stream>>>(
            static_cast<const uint8_t*>(inputs[0]), static_cast<T*>(outputs[0]),
            n, epc, nc, values_offset);
        FZ_CUDA_CHECK(cudaGetLastError());

        actual_output_sizes_ = {n * sizeof(T)};

    } else if (is_inverse_) {
        // ── DECOMPRESSION ────────────────────────────────────────────────────
        // Read num_runs from the first 4 bytes of the compressed stream.
        // This D2H sync is unavoidable with the current compact wire format;
        // decompression is not intended to be CUDA Graph-capturable.
        uint32_t num_runs;
        FZ_CUDA_CHECK(cudaMemcpyAsync(&num_runs, inputs[0], sizeof(uint32_t),
                       cudaMemcpyDeviceToHost, stream));
        FZ_CUDA_CHECK(cudaStreamSynchronize(stream));
        FZ_LOG(TRACE, "RLE decode: %u runs -> %u elems", num_runs, cached_num_elements_);

        if (num_runs == 0) {
            actual_output_sizes_ = {0};
            return;
        }

        const uint8_t* input_base = static_cast<const uint8_t*>(inputs[0]);
        const T* compressed_values = reinterpret_cast<const T*>(
            input_base + rleValuesOffset<T>());

        const size_t values_bytes   = num_runs * sizeof(T);
        const size_t values_aligned = (values_bytes + 3) & ~3;
        const uint32_t* run_lengths = reinterpret_cast<const uint32_t*>(
            input_base + rleValuesOffset<T>() + values_aligned);

        // Prefix sum of run_lengths → run_offsets (for scattered decompression)
        uint32_t* d_run_offsets = nullptr;
        if (pool) {
            d_run_offsets = static_cast<uint32_t*>(
                pool->allocate(num_runs * sizeof(uint32_t), stream, "rle_run_offsets"));
        } else {
            FZ_CUDA_CHECK(cudaMalloc(&d_run_offsets, num_runs * sizeof(uint32_t)));
        }

        auto d_temp = fz::backend::withTempStorage(pool, stream, "rle_cub_decomp_temp",
            [&](void* tmp, size_t& bytes) {
                cub::DeviceScan::InclusiveSum(tmp, bytes,
                                              run_lengths, d_run_offsets, num_runs, stream);
            });

        // Read total output size (last element of prefix sum)
        uint32_t total_output_size;
        FZ_CUDA_CHECK(cudaMemcpyAsync(&total_output_size, d_run_offsets + num_runs - 1,
                       sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
        FZ_CUDA_CHECK(cudaStreamSynchronize(stream));

        launchRLEDecompressKernel<T>(
            compressed_values, run_lengths, d_run_offsets,
            static_cast<T*>(outputs[0]), num_runs, stream);

        if (pool) {
            pool->free(d_run_offsets, stream);
        } else {
            FZ_CUDA_CHECK_WARN(cudaStreamSynchronize(stream));
            FZ_CUDA_CHECK_WARN(cudaFree(d_run_offsets));
        }
        fz::backend::freeTempStorage(pool, d_temp, stream);

        actual_output_sizes_ = {total_output_size * sizeof(T)};

    } else {
        // ── COMPRESSION (forward, CUDA Graph-capturable) ─────────────────────
        const size_t byte_size    = sizes[0];
        const size_t n            = byte_size / sizeof(T);
        cached_num_elements_      = static_cast<uint32_t>(n);

        if (n == 0) {
            uint32_t zero = 0;
            FZ_CUDA_CHECK(cudaMemcpyAsync(outputs[0], &zero, sizeof(uint32_t),
                           cudaMemcpyHostToDevice, stream));
            actual_output_sizes_ = {sizeof(uint32_t)};
            fwd_sync_pending_    = false;
            return;
        }

        // ── Grow persistent scratch if needed ────────────────────────────────
        // Triggered only on the first call or when a larger dataset is seen.
        if (n > fwd_scratch_n_) {
            // Free previous allocations
            auto dev_free = [&](void* p) {
                if (!p) return;
                if (fwd_from_pool_ && fwd_scratch_pool_) fwd_scratch_pool_->free(p, 0);
                else {
                    FZ_CUDA_CHECK_WARN(cudaStreamSynchronize(stream));
                    cudaFree(p);
                }
            };
            dev_free(d_is_boundary_);      d_is_boundary_       = nullptr;
            dev_free(d_boundary_scan_);    d_boundary_scan_     = nullptr;
            dev_free(d_boundary_positions_); d_boundary_positions_ = nullptr;
            dev_free(d_values_scratch_);   d_values_scratch_    = nullptr;
            dev_free(d_lengths_scratch_);  d_lengths_scratch_   = nullptr;

            if (pool) {
                d_is_boundary_ = static_cast<uint8_t*>(pool->allocate(
                    n, stream, "rle_is_boundary", /*persistent=*/true));
                d_boundary_scan_ = static_cast<uint32_t*>(pool->allocate(
                    n * sizeof(uint32_t), stream, "rle_boundary_scan", /*persistent=*/true));
                d_boundary_positions_ = static_cast<uint32_t*>(pool->allocate(
                    n * sizeof(uint32_t), stream, "rle_boundary_positions", /*persistent=*/true));
                d_values_scratch_ = static_cast<T*>(pool->allocate(
                    n * sizeof(T), stream, "rle_values_scratch", /*persistent=*/true));
                d_lengths_scratch_ = static_cast<uint32_t*>(pool->allocate(
                    n * sizeof(uint32_t), stream, "rle_lengths_scratch", /*persistent=*/true));
                fwd_scratch_pool_ = pool;
                fwd_from_pool_    = true;
            } else {
                FZ_CUDA_CHECK(cudaMalloc(&d_is_boundary_,       n));
                FZ_CUDA_CHECK(cudaMalloc(&d_boundary_scan_,     n * sizeof(uint32_t)));
                FZ_CUDA_CHECK(cudaMalloc(&d_boundary_positions_,n * sizeof(uint32_t)));
                FZ_CUDA_CHECK(cudaMalloc(&d_values_scratch_,    n * sizeof(T)));
                FZ_CUDA_CHECK(cudaMalloc(&d_lengths_scratch_,   n * sizeof(uint32_t)));
                fwd_scratch_pool_ = nullptr;
                fwd_from_pool_    = false;
            }

            if (!h_num_runs_) {
                FZ_CUDA_CHECK(cudaHostAlloc(&h_num_runs_, sizeof(uint32_t),
                                            cudaHostAllocDefault));
            }
            fwd_scratch_n_ = n;
        }

        const T*    input      = static_cast<const T*>(inputs[0]);
        uint8_t*    out_base   = static_cast<uint8_t*>(outputs[0]);

        if (isChunked()) {
            // ── COMPRESSION, chunked (CUDA Graph-capturable) ─────────────────
            const uint32_t epc = elemsPerChunk();
            const uint32_t nc  = static_cast<uint32_t>(numChunks(byte_size));
            const uint32_t values_offset =
                static_cast<uint32_t>(rleChunkedValuesOffset<T>(nc));

            // d_boundary_scan_ (n × u32, n ≥ nc) doubles as the per-chunk run
            // counts; d_boundary_positions_ holds the per-chunk run starts.
            uint32_t* d_runs_per_chunk = d_boundary_scan_;

            rle_chunk_encode_kernel<T><<<nc, RLE_CHUNK_BLOCK, 0, stream>>>(
                input, n, epc, d_values_scratch_, d_lengths_scratch_,
                d_boundary_positions_, d_runs_per_chunk);
            FZ_CUDA_CHECK(cudaGetLastError());

            // Exclusive scan of the run counts, written straight into the
            // output's offset table (entries [1 .. nc]).
            uint32_t* d_offsets = reinterpret_cast<uint32_t*>(out_base) + 1;
            {
                auto d_tmp = fz::backend::withTempStorage(pool, stream, "rle_chunk_scan_tmp",
                    [&](void* tmp, size_t& bytes) {
                        cub::DeviceScan::ExclusiveSum(tmp, bytes,
                                                      d_runs_per_chunk, d_offsets,
                                                      static_cast<int>(nc), stream);
                    });
                fz::backend::freeTempStorage(pool, d_tmp, stream);
            }

            rle_chunk_finalize_offsets_kernel<<<1, 1, 0, stream>>>(
                out_base, d_runs_per_chunk, nc);
            FZ_CUDA_CHECK(cudaGetLastError());

            rle_chunk_compact_kernel<T><<<nc, RLE_CHUNK_BLOCK, 0, stream>>>(
                d_values_scratch_, d_lengths_scratch_, out_base,
                epc, nc, values_offset);
            FZ_CUDA_CHECK(cudaGetLastError());

            // total_runs terminates the offset table; async D2H for the size.
            FZ_CUDA_CHECK(cudaMemcpyAsync(h_num_runs_,
                           reinterpret_cast<uint32_t*>(out_base) + 1 + nc,
                           sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
            fwd_last_stream_  = stream;
            fwd_sync_pending_ = true;
            return;
        }

        const int   block_size = 256;
        const int   grid_size  = static_cast<int>((n + block_size - 1) / block_size);

        // ── Phase 1: mark run boundaries ─────────────────────────────────────
        rle_mark_boundaries_kernel<T><<<grid_size, block_size, 0, stream>>>(
            input, d_is_boundary_, n);
        FZ_CUDA_CHECK(cudaGetLastError());

        // ── Phase 2: inclusive prefix sum → d_boundary_scan_ ─────────────────
        // d_boundary_scan_[n-1] == num_runs (used as device-side num_runs ptr)
        {
            auto d_tmp = fz::backend::withTempStorage(pool, stream, "rle_cub_scan_tmp",
                [&](void* tmp, size_t& bytes) {
                    cub::DeviceScan::InclusiveSum(tmp, bytes,
                                                  d_is_boundary_, d_boundary_scan_,
                                                  static_cast<int>(n), stream);
                });
            fz::backend::freeTempStorage(pool, d_tmp, stream);
        }

        // d_num_runs_ptr points into d_boundary_scan_ — no D2H needed
        const uint32_t* d_num_runs_ptr = d_boundary_scan_ + (n - 1);

        // ── Phase 3: scatter boundary positions ──────────────────────────────
        scatter_boundary_positions_kernel<<<grid_size, block_size, 0, stream>>>(
            d_is_boundary_, d_boundary_scan_, d_boundary_positions_, n);
        FZ_CUDA_CHECK(cudaGetLastError());

        // ── Phase 4: extract run values + lengths into scratch ────────────────
        rle_extract_runs_kernel<T><<<grid_size, block_size, 0, stream>>>(
            input, d_boundary_positions_, d_num_runs_ptr,
            d_values_scratch_, d_lengths_scratch_, n);
        FZ_CUDA_CHECK(cudaGetLastError());

        // ── Phase 5: pack scratch → compact wire format in outputs[0] ─────────
        // Grid is the same worst-case size; threads past *d_num_runs return early.
        rle_pack_kernel<T><<<grid_size, block_size, 0, stream>>>(
            d_values_scratch_, d_lengths_scratch_,
            out_base, d_num_runs_ptr, n);
        FZ_CUDA_CHECK(cudaGetLastError());

        // ── Async D2H of num_runs ─────────────────────────────────────────────
        // h_num_runs_ is pinned; the copy completes when the stream is synced.
        // fwd_last_stream_ lets completePendingSync() sync the right stream
        // when getActualOutputSizesByName() is called before postStreamSync().
        FZ_CUDA_CHECK(cudaMemcpyAsync(h_num_runs_, d_num_runs_ptr, sizeof(uint32_t),
                       cudaMemcpyDeviceToHost, stream));
        fwd_last_stream_  = stream;
        fwd_sync_pending_ = true;
    }
}

// ── postStreamSync() ─────────────────────────────────────────────────────────
// Called by the pipeline after stream synchronization.  Delegates to
// completePendingSync() which is also callable from the const getters.
template<typename T>
void RLEStage<T>::postStreamSync(cudaStream_t /*stream*/) {
    completePendingSync();
}

// Explicit template instantiations
template class RLEStage<uint8_t>;
template class RLEStage<uint16_t>;
template class RLEStage<uint32_t>;
template class RLEStage<uint64_t>;
template class RLEStage<int8_t>;
template class RLEStage<int16_t>;
template class RLEStage<int32_t>;
template class RLEStage<int64_t>;

template void launchRLEDecompressKernel<uint8_t>(const uint8_t*, const uint32_t*, const uint32_t*, uint8_t*, uint32_t, cudaStream_t);
template void launchRLEDecompressKernel<uint16_t>(const uint16_t*, const uint32_t*, const uint32_t*, uint16_t*, uint32_t, cudaStream_t);
template void launchRLEDecompressKernel<uint32_t>(const uint32_t*, const uint32_t*, const uint32_t*, uint32_t*, uint32_t, cudaStream_t);
template void launchRLEDecompressKernel<uint64_t>(const uint64_t*, const uint32_t*, const uint32_t*, uint64_t*, uint32_t, cudaStream_t);
template void launchRLEDecompressKernel<int8_t>(const int8_t*, const uint32_t*, const uint32_t*, int8_t*, uint32_t, cudaStream_t);
template void launchRLEDecompressKernel<int16_t>(const int16_t*, const uint32_t*, const uint32_t*, int16_t*, uint32_t, cudaStream_t);
template void launchRLEDecompressKernel<int32_t>(const int32_t*, const uint32_t*, const uint32_t*, int32_t*, uint32_t, cudaStream_t);
template void launchRLEDecompressKernel<int64_t>(const int64_t*, const uint32_t*, const uint32_t*, int64_t*, uint32_t, cudaStream_t);

} // namespace fz
