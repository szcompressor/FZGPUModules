#include "encoders/diff/diff.h"
#include "transforms/negabinary/negabinary.h"
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cub/block/block_scan.cuh>
#include "mem/mempool.h"
#include "cuda_check.h"

namespace fz {

// ─── Forward kernel: difference with optional chunking + negabinary output ────
//
// chunk_elems == 0  → whole array is one chunk (only idx 0 is a boundary).
// chunk_elems  > 0  → first element of each chunk is stored as-is.
//
// When TOut != T the computed difference is negabinary-encoded before writing.
template<typename T, typename TOut>
__global__ void diffKernel(const T* __restrict__ in,
                            TOut* __restrict__ out,
                            size_t n,
                            size_t chunk_elems)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    bool is_boundary = (idx == 0) ||
                       (chunk_elems > 0 && (idx % chunk_elems == 0));

    T diff = is_boundary ? in[idx] : (in[idx] - in[idx - 1]);

    if constexpr (std::is_same_v<T, TOut>) {
        out[idx] = diff;
    } else {
        out[idx] = Negabinary<T>::encode(diff);
    }
}

// ─── Negabinary decode pass: TOut[] → T[] ────────────────────────────────────
//
// Used as the first step of the inverse pass when TOut != T.
template<typename T, typename TOut>
__global__ void negabinaryDecodePassKernel(const TOut* __restrict__ in,
                                            T* __restrict__ out,
                                            size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = Negabinary<T>::decode(in[idx]);
}

// ─── Chunked inclusive prefix-sum kernel ─────────────────────────────────────
//
// One CUDA block per chunk.  Tiles through the chunk in strides of BLOCK_DIM
// using cub::BlockScan, carrying the running prefix across tiles.
//
// Operates in-place: data[] is read as differences, written as cumsum.
// The caller must have already placed the (decoded) differences into data[]
// before launching this kernel.
//
// Must be launched with exactly BLOCK_DIM threads per block.
template<typename T, int BLOCK_DIM>
__global__ void cumsumChunkedKernel(T* __restrict__ data,
                                     size_t n,
                                     size_t chunk_elems)
{
    using BlockScan = cub::BlockScan<T, BLOCK_DIM>;
    __shared__ typename BlockScan::TempStorage temp;

    size_t base    = static_cast<size_t>(blockIdx.x) * chunk_elems;
    if (base >= n) return;
    size_t local_n = min(chunk_elems, n - base);

    T prefix = T(0);
    for (size_t tile = 0; tile < local_n; tile += BLOCK_DIM) {
        size_t tid    = threadIdx.x;
        size_t g_idx  = base + tile + tid;
        bool   valid  = (tile + tid) < local_n;

        T val = (valid && g_idx < n) ? data[g_idx] : T(0);

        T scan_out, agg;
        BlockScan(temp).InclusiveSum(val, scan_out, agg);
        __syncthreads();

        if (valid) data[g_idx] = scan_out + prefix;
        prefix += agg;
        __syncthreads();
    }
}

// ─── Helper: forward diff launch ─────────────────────────────────────────────
template<typename T, typename TOut>
static void launchDiff(const T* in, TOut* out, size_t n,
                       size_t chunk_elems, cudaStream_t stream)
{
    constexpr int kBlock = 256;
    int grid = static_cast<int>((n + kBlock - 1) / kBlock);
    diffKernel<T, TOut><<<grid, kBlock, 0, stream>>>(in, out, n, chunk_elems);
}

// ─── Helper: CUB global inclusive sum (no chunking) ──────────────────────────
template<typename T>
static void launchGlobalCumsum(const T* in, T* out, size_t n,
                                cudaStream_t stream, MemoryPool* pool)
{
    int    n_int      = static_cast<int>(n);
    size_t temp_bytes = 0;
    cub::DeviceScan::InclusiveSum(nullptr, temp_bytes, in, out, n_int, stream);

    void* d_temp = nullptr;
    if (pool) {
        d_temp = pool->allocate(temp_bytes, stream, "diff_cub_temp");
    } else {
        FZ_CUDA_CHECK(cudaMallocAsync(&d_temp, temp_bytes, stream));
    }

    cub::DeviceScan::InclusiveSum(d_temp, temp_bytes, in, out, n_int, stream);

    if (pool) {
        pool->free(d_temp, stream);
    } else {
        FZ_CUDA_CHECK_WARN(cudaFreeAsync(d_temp, stream));
    }
}

// ─── Helper: block-scan chunked inclusive sum (in-place) ─────────────────────
template<typename T>
static void launchChunkedCumsum(T* data, size_t n,
                                 size_t chunk_elems, cudaStream_t stream)
{
    constexpr int kBlock    = 256;
    size_t        num_chunks = (n + chunk_elems - 1) / chunk_elems;
    cumsumChunkedKernel<T, kBlock><<<static_cast<int>(num_chunks), kBlock,
                                     0, stream>>>(data, n, chunk_elems);
}

// ─── execute() ───────────────────────────────────────────────────────────────
template<typename T, typename TOut>
void DifferenceStage<T, TOut>::execute(
    cudaStream_t stream,
    MemoryPool* pool,
    const std::vector<void*>& inputs,
    const std::vector<void*>& outputs,
    const std::vector<size_t>& sizes
) {
    if (inputs.empty() || outputs.empty() || sizes.empty())
        throw std::runtime_error("DifferenceStage: Invalid inputs/outputs");

    size_t byte_size   = sizes[0];
    size_t n           = byte_size / sizeof(T);
    size_t chunk_elems = (chunk_size_ > 0) ? (chunk_size_ / sizeof(T)) : size_t(0);

    if (n == 0) { actual_output_size_ = 0; return; }

    if (!is_inverse_) {
        // ── Forward: difference ± negabinary encode ───────────────────────────
        launchDiff<T, TOut>(
            static_cast<const T*>(inputs[0]),
            static_cast<TOut*>(outputs[0]),
            n, chunk_elems, stream);
    } else {
        // ── Inverse: (decode negabinary) → cumulative sum ─────────────────────
        //
        // if constexpr guards ensure Negabinary<T> is only instantiated for the
        // signed→unsigned pairs; unreachable code is excluded at compile time.
        if constexpr (!std::is_same_v<T, TOut>) {
            // Step 1: decode each TOut element back to T into a scratch buffer.
            T* d_decoded = nullptr;
            if (pool) {
                d_decoded = static_cast<T*>(
                    pool->allocate(n * sizeof(T), stream, "diff_nb_decode_tmp"));
            } else {
                FZ_CUDA_CHECK(cudaMallocAsync(
                    reinterpret_cast<void**>(&d_decoded), n * sizeof(T), stream));
            }

            {
                constexpr int kBlock = 256;
                int grid = static_cast<int>((n + kBlock - 1) / kBlock);
                negabinaryDecodePassKernel<T, TOut><<<grid, kBlock, 0, stream>>>(
                    static_cast<const TOut*>(inputs[0]), d_decoded, n);
            }

            // Step 2: cumsum on decoded values → output buffer.
            T* out_ptr = static_cast<T*>(outputs[0]);
            if (chunk_elems == 0) {
                launchGlobalCumsum<T>(d_decoded, out_ptr, n, stream, pool);
            } else {
                cudaMemcpyAsync(out_ptr, d_decoded, n * sizeof(T),
                                cudaMemcpyDeviceToDevice, stream);
                launchChunkedCumsum<T>(out_ptr, n, chunk_elems, stream);
            }

            if (pool) {
                pool->free(d_decoded, stream);
            } else {
                FZ_CUDA_CHECK_WARN(cudaFreeAsync(d_decoded, stream));
            }
        } else {
            // No negabinary — TOut == T.
            const T* in_ptr  = static_cast<const T*>(inputs[0]);
            T*       out_ptr = static_cast<T*>(outputs[0]);

            if (chunk_elems == 0) {
                launchGlobalCumsum<T>(in_ptr, out_ptr, n, stream, pool);
            } else {
                cudaMemcpyAsync(out_ptr, in_ptr, n * sizeof(T),
                                cudaMemcpyDeviceToDevice, stream);
                launchChunkedCumsum<T>(out_ptr, n, chunk_elems, stream);
            }
        }
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(
            std::string("DifferenceStage kernel launch failed: ")
            + cudaGetErrorString(err));

    actual_output_size_ = byte_size;
}

// ─── Explicit instantiations ──────────────────────────────────────────────────

// Same-type (TOut = T) — original API
template class DifferenceStage<float>;
template class DifferenceStage<double>;
template class DifferenceStage<int32_t>;
template class DifferenceStage<int64_t>;
template class DifferenceStage<uint16_t>;
template class DifferenceStage<uint8_t>;
template class DifferenceStage<uint32_t>;

// Negabinary-fused (TOut = unsigned counterpart of T)
template class DifferenceStage<int8_t,  uint8_t>;
template class DifferenceStage<int16_t, uint16_t>;
template class DifferenceStage<int32_t, uint32_t>;
template class DifferenceStage<int64_t, uint64_t>;

// Kernels used by negabinary-fused instantiations
template __global__ void negabinaryDecodePassKernel<int8_t,  uint8_t> (const  uint8_t*,  int8_t*, size_t);
template __global__ void negabinaryDecodePassKernel<int16_t, uint16_t>(const uint16_t*, int16_t*, size_t);
template __global__ void negabinaryDecodePassKernel<int32_t, uint32_t>(const uint32_t*, int32_t*, size_t);
template __global__ void negabinaryDecodePassKernel<int64_t, uint64_t>(const uint64_t*, int64_t*, size_t);

} // namespace fz
