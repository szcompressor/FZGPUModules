#include "encoders/RLE/rle.h"
#include "log.h"
#include <cuda_runtime.h>
#include <cub/cub.cuh>
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
 *   [num_runs: u32][values: T×num_runs, 4B-aligned][run_lengths: u32×num_runs]
 *
 * Uses *d_num_runs (device pointer) for all layout arithmetic — no host
 * involvement required.  Grid is launched at the worst-case size (n
 * elements); threads past num_runs return immediately.
 *
 * The header write (i==0) and the values/lengths writes touch disjoint
 * byte ranges and do not race:
 *   header   → [0, 4)
 *   values   → [4, 4 + num_runs*sizeof(T))
 *   lengths  → [4 + values_aligned, 4 + values_aligned + num_runs*4)
 */
template<typename T>
__global__ void rle_pack_kernel(
    const T* __restrict__ values_scratch,
    const uint32_t* __restrict__ lengths_scratch,
    uint8_t* __restrict__ output_base,
    const uint32_t* __restrict__ d_num_runs,
    const size_t n
) {
    const uint32_t num_runs     = *d_num_runs;
    const uint32_t values_bytes = num_runs * static_cast<uint32_t>(sizeof(T));
    const uint32_t values_aligned = (values_bytes + 3u) & ~3u;
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i == 0) {
        *reinterpret_cast<uint32_t*>(output_base) = num_runs;
        // Zero the alignment padding between the values section and the lengths
        // section.  values_aligned may be up to 3 bytes larger than values_bytes
        // (4-byte alignment).  Those pad bytes are never written by the per-run
        // threads below, leaving them uninitialized.  Zero them here so the full
        // [0, actual_output_size_) range is always initialized.
        uint8_t* pad_base = output_base + sizeof(uint32_t) + values_bytes;
        for (uint32_t b = 0; b < values_aligned - values_bytes; b++)
            pad_base[b] = 0;
    }
    if (i < num_runs) {
        reinterpret_cast<T*>(output_base + sizeof(uint32_t))[i] = values_scratch[i];
        reinterpret_cast<uint32_t*>(output_base + sizeof(uint32_t) + values_aligned)[i]
            = lengths_scratch[i];
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

    if (is_inverse_) {
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
            input_base + sizeof(uint32_t));

        const size_t values_bytes   = num_runs * sizeof(T);
        const size_t values_aligned = (values_bytes + 3) & ~3;
        const uint32_t* run_lengths = reinterpret_cast<const uint32_t*>(
            input_base + sizeof(uint32_t) + values_aligned);

        // Prefix sum of run_lengths → run_offsets (for scattered decompression)
        uint32_t* d_run_offsets = nullptr;
        if (pool) {
            d_run_offsets = static_cast<uint32_t*>(
                pool->allocate(num_runs * sizeof(uint32_t), stream, "rle_run_offsets"));
        } else {
            FZ_CUDA_CHECK(cudaMallocAsync(&d_run_offsets, num_runs * sizeof(uint32_t), stream));
        }

        void*  d_temp  = nullptr;
        size_t tmp_sz  = 0;
        cub::DeviceScan::InclusiveSum(d_temp, tmp_sz,
                                      run_lengths, d_run_offsets, num_runs, stream);
        d_temp = pool ? pool->allocate(tmp_sz, stream, "rle_cub_decomp_temp")
                      : nullptr;
        if (!pool) FZ_CUDA_CHECK(cudaMallocAsync(&d_temp, tmp_sz, stream));
        cub::DeviceScan::InclusiveSum(d_temp, tmp_sz,
                                      run_lengths, d_run_offsets, num_runs, stream);

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
            pool->free(d_temp, stream);
        } else {
            FZ_CUDA_CHECK_WARN(cudaFreeAsync(d_run_offsets, stream));
            FZ_CUDA_CHECK_WARN(cudaFreeAsync(d_temp, stream));
        }

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
                else cudaFree(p);
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
        const int   block_size = 256;
        const int   grid_size  = static_cast<int>((n + block_size - 1) / block_size);

        // ── Phase 1: mark run boundaries ─────────────────────────────────────
        rle_mark_boundaries_kernel<T><<<grid_size, block_size, 0, stream>>>(
            input, d_is_boundary_, n);
        FZ_CUDA_CHECK(cudaGetLastError());

        // ── Phase 2: inclusive prefix sum → d_boundary_scan_ ─────────────────
        // d_boundary_scan_[n-1] == num_runs (used as device-side num_runs ptr)
        {
            void*  d_tmp   = nullptr;
            size_t tmp_sz  = 0;
            cub::DeviceScan::InclusiveSum(d_tmp, tmp_sz,
                                          d_is_boundary_, d_boundary_scan_,
                                          static_cast<int>(n), stream);
            d_tmp = pool ? pool->allocate(tmp_sz, stream, "rle_cub_scan_tmp")
                         : nullptr;
            if (!pool) FZ_CUDA_CHECK(cudaMallocAsync(&d_tmp, tmp_sz, stream));
            cub::DeviceScan::InclusiveSum(d_tmp, tmp_sz,
                                          d_is_boundary_, d_boundary_scan_,
                                          static_cast<int>(n), stream);
            if (pool && d_tmp) pool->free(d_tmp, stream);
            else if (d_tmp)    FZ_CUDA_CHECK_WARN(cudaFreeAsync(d_tmp, stream));
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
template class RLEStage<int32_t>;

template void launchRLEDecompressKernel<uint8_t>(const uint8_t*, const uint32_t*, const uint32_t*, uint8_t*, uint32_t, cudaStream_t);
template void launchRLEDecompressKernel<uint16_t>(const uint16_t*, const uint32_t*, const uint32_t*, uint16_t*, uint32_t, cudaStream_t);
template void launchRLEDecompressKernel<uint32_t>(const uint32_t*, const uint32_t*, const uint32_t*, uint32_t*, uint32_t, cudaStream_t);
template void launchRLEDecompressKernel<int32_t>(const int32_t*, const uint32_t*, const uint32_t*, int32_t*, uint32_t, cudaStream_t);

} // namespace fz
