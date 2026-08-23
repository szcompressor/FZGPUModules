#include "predictors/diff/diff.h"
#include "stage/stage_registry.h"
#include <cstring>
#include <algorithm>
#include <stdexcept>
#include <string>
#include "transforms/negabinary/negabinary.h"
#include "transforms/zigzag/zigzag.h"
#include "log.h"
#include "backend/api.h"
#include "backend/cub.h"
#include "mem/mempool.h"
#include "cuda_check.h"

namespace fz {

// ─── Forward kernel: difference with optional chunking + fused output ────────
//
// chunk_elems == 0  → whole array is one chunk (only idx 0 is a boundary).
// chunk_elems  > 0  → first element of each chunk is stored as-is.
//
// When TOut != T the computed difference is encoded (negabinary or zigzag,
// selected by Mode) before writing.
//
// Uses a grid-stride loop so each thread processes multiple elements, matching
// the PFPL reference pattern (d_DIFFNB/d_DIFFMS).  This improves instruction-
// level parallelism and reduces grid launch overhead for large arrays.
template<typename T, typename TOut, FusionMode Mode>
__global__ void diffKernel(const T* __restrict__ in,
                            TOut* __restrict__ out,
                            size_t n,
                            size_t chunk_elems)
{
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n;
         idx += stride)
    {
        // chunk_elems is always a power-of-2 (chunk_size is 2^k and element sizes
        // are powers of 2), so the boundary test reduces to a cheap bitwise AND
        // instead of an expensive runtime integer division (IDIV).
        bool is_boundary = (idx == 0) ||
                           (chunk_elems > 0 && ((idx & (chunk_elems - 1)) == 0));

        T diff = is_boundary ? in[idx] : (in[idx] - in[idx - 1]);

        if constexpr (std::is_same_v<T, TOut>) {
            out[idx] = diff;
        } else if constexpr (Mode == FusionMode::NEGABINARY) {
            out[idx] = Negabinary<T>::encode(diff);
        } else {
            out[idx] = Zigzag<T>::encode(diff);
        }
    }
}

// ─── Fused decode pass: TOut[] → T[] ─────────────────────────────────────────
//
// Used as the first step of the inverse pass when TOut != T. Undoes whichever
// transform Mode selected. Grid-stride loop for multi-element-per-thread
// throughput.
template<typename T, typename TOut, FusionMode Mode>
__global__ void fusionDecodePassKernel(const TOut* __restrict__ in,
                                        T* __restrict__ out,
                                        size_t n)
{
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n;
         idx += stride)
    {
        if constexpr (Mode == FusionMode::NEGABINARY) {
            out[idx] = Negabinary<T>::decode(in[idx]);
        } else {
            out[idx] = Zigzag<T>::decode(in[idx]);
        }
    }
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
template<typename T, typename TOut, FusionMode Mode>
static void launchDiff(const T* in, TOut* out, size_t n,
                       size_t chunk_elems, cudaStream_t stream)
{
    // chunk_elems must be a power-of-2 for the bitwise boundary test in the kernel.
    // All supported chunk_size values (16384, 8192, …) and element sizes are
    // powers-of-2, so this should never fire in practice.
    if (chunk_elems > 0 && (chunk_elems & (chunk_elems - 1)) != 0)
        throw std::runtime_error("DifferenceStage: chunk_elems must be a power-of-2");
    constexpr int kBlock = 512;
    // Target ~8 elements per thread for good ILP (matches PFPL's stride pattern
    // where TPB=512 threads iterate over 4096 elements per chunk).
    int fullGrid = static_cast<int>((n + kBlock - 1) / kBlock);
    int grid = fullGrid < (fullGrid / 8 + 1) ? fullGrid : (fullGrid / 8 + 1);
    if (grid < 1) grid = 1;
    diffKernel<T, TOut, Mode><<<grid, kBlock, 0, stream>>>(in, out, n, chunk_elems);
}

// ─── Helper: block-scan chunked inclusive sum (in-place) ─────────────────────
template<typename T>
static void launchChunkedCumsum(T* data, size_t n,
                                 size_t chunk_elems, cudaStream_t stream)
{
    constexpr int kBlock    = 512;
    size_t        num_chunks = (n + chunk_elems - 1) / chunk_elems;
    cumsumChunkedKernel<T, kBlock><<<static_cast<int>(num_chunks), kBlock,
                                     0, stream>>>(data, n, chunk_elems);
}

// ─── Helper: global inclusive sum (no chunking, no extra device allocation) ───
//
// For the global case (chunk_elems == 0) we use a two-step approach:
//   1. Copy in → out (if in != out) so that cumsumChunkedKernel can work
//      in-place.
//   2. Run cumsumChunkedKernel with chunk_elems = n, which treats the whole
//      array as a single chunk handled by one CUDA block.  This uses only
//      shared memory (CUB BlockScan) — no extra device allocation.
//
// For small-to-medium arrays this is efficient.  For very large arrays the
// single-block tiling degrades in performance but remains correct.  CUB
// DeviceScan is preferred for large n, but CUB temp allocation causes
// compute-sanitizer shadow-memory races when the enclosing pool is destroyed
// immediately after (the decompressFromFile pattern), so we avoid it here.
template<typename T>
static void launchGlobalCumsum(const T* in, T* out, size_t n, cudaStream_t stream)
{
    if (n == 0) return;
    // Copy input to output buffer so cumsumChunkedKernel can work in-place.
    if (in != out) {
        FZ_CUDA_CHECK(cudaMemcpyAsync(out, in, n * sizeof(T),
                                     cudaMemcpyDeviceToDevice, stream));
    }
    // One block handles the whole array in tiles of kBlock elements.
    launchChunkedCumsum<T>(out, n, n, stream);
}

// ─── execute() ───────────────────────────────────────────────────────────────
template<typename T, typename TOut, FusionMode Mode>
void DifferenceStage<T, TOut, Mode>::execute(
    cudaStream_t stream,
    MemoryPool* pool,
    const std::vector<void*>& inputs,
    const std::vector<void*>& outputs,
    const std::vector<size_t>& sizes
) {
    if (inputs.empty() || outputs.empty() || sizes.empty())
        throw std::runtime_error(
            "DifferenceStage: inputs, outputs, and sizes vectors must all be non-empty");

    size_t byte_size   = sizes[0];
    size_t n           = byte_size / sizeof(T);
    size_t chunk_elems = (chunk_size_ > 0) ? (chunk_size_ / sizeof(T)) : size_t(0);

    if (n == 0) { actual_output_size_ = 0; return; }

    if (!is_inverse_) {
        // ── Forward: difference ± fused negabinary/zigzag encode ──────────────
        launchDiff<T, TOut, Mode>(
            static_cast<const T*>(inputs[0]),
            static_cast<TOut*>(outputs[0]),
            n, chunk_elems, stream);
    } else {
        // ── Inverse: (decode negabinary/zigzag) → cumulative sum ──────────────
        //
        // if constexpr guards ensure Negabinary<T>/Zigzag<T> are only instantiated
        // for the signed→unsigned pairs; unreachable code is excluded at compile time.
        if constexpr (!std::is_same_v<T, TOut>) {
            // Step 1: decode each TOut element back to T into a scratch buffer.
            T* d_decoded = nullptr;
            if (pool) {
                d_decoded = static_cast<T*>(
                    pool->allocate(n * sizeof(T), stream, "diff_nb_decode_tmp"));
            } else {
                FZ_CUDA_CHECK(cudaMalloc(
                    reinterpret_cast<void**>(&d_decoded), n * sizeof(T)));
            }

            {
                constexpr int kBlock = 512;
                int fullGrid = static_cast<int>((n + kBlock - 1) / kBlock);
                int grid = fullGrid < (fullGrid / 8 + 1) ? fullGrid : (fullGrid / 8 + 1);
                if (grid < 1) grid = 1;
                fusionDecodePassKernel<T, TOut, Mode><<<grid, kBlock, 0, stream>>>(
                    static_cast<const TOut*>(inputs[0]), d_decoded, n);
            }

            // Step 2: cumsum on decoded values → output buffer.
            T* out_ptr = static_cast<T*>(outputs[0]);
            if (chunk_elems == 0) {
                launchGlobalCumsum<T>(d_decoded, out_ptr, n, stream);
            } else {
                cudaMemcpyAsync(out_ptr, d_decoded, n * sizeof(T),
                                cudaMemcpyDeviceToDevice, stream);
                launchChunkedCumsum<T>(out_ptr, n, chunk_elems, stream);
            }

            if (pool) {
                pool->free(d_decoded, stream);
            } else {
                FZ_CUDA_CHECK_WARN(cudaStreamSynchronize(stream));
                FZ_CUDA_CHECK_WARN(cudaFree(d_decoded));
            }
        } else {
            // No fusion — TOut == T.
            const T* in_ptr  = static_cast<const T*>(inputs[0]);
            T*       out_ptr = static_cast<T*>(outputs[0]);

            if (chunk_elems == 0) {
                launchGlobalCumsum<T>(in_ptr, out_ptr, n, stream);
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
    FZ_LOG(TRACE, "Difference %s: %.1f KB, %zu elems, chunk=%zu",
           is_inverse_ ? "cumsum" : "diff",
           byte_size / 1024.0, n,
           chunk_elems > 0 ? chunk_elems : n);
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

// Negabinary-fused (TOut = unsigned counterpart of T; Mode defaults to NEGABINARY)
template class DifferenceStage<int8_t,  uint8_t>;
template class DifferenceStage<int16_t, uint16_t>;
template class DifferenceStage<int32_t, uint32_t>;
template class DifferenceStage<int64_t, uint64_t>;

// Zigzag-fused (TOut = unsigned counterpart of T, Mode = ZIGZAG)
template class DifferenceStage<int8_t,  uint8_t,  FusionMode::ZIGZAG>;
template class DifferenceStage<int16_t, uint16_t, FusionMode::ZIGZAG>;
template class DifferenceStage<int32_t, uint32_t, FusionMode::ZIGZAG>;
template class DifferenceStage<int64_t, uint64_t, FusionMode::ZIGZAG>;

// Kernels used by negabinary-fused instantiations
template __global__ void fusionDecodePassKernel<int8_t,  uint8_t,  FusionMode::NEGABINARY>(const  uint8_t*,  int8_t*, size_t);
template __global__ void fusionDecodePassKernel<int16_t, uint16_t, FusionMode::NEGABINARY>(const uint16_t*, int16_t*, size_t);
template __global__ void fusionDecodePassKernel<int32_t, uint32_t, FusionMode::NEGABINARY>(const uint32_t*, int32_t*, size_t);
template __global__ void fusionDecodePassKernel<int64_t, uint64_t, FusionMode::NEGABINARY>(const uint64_t*, int64_t*, size_t);

// Kernels used by zigzag-fused instantiations
template __global__ void fusionDecodePassKernel<int8_t,  uint8_t,  FusionMode::ZIGZAG>(const  uint8_t*,  int8_t*, size_t);
template __global__ void fusionDecodePassKernel<int16_t, uint16_t, FusionMode::ZIGZAG>(const uint16_t*, int16_t*, size_t);
template __global__ void fusionDecodePassKernel<int32_t, uint32_t, FusionMode::ZIGZAG>(const uint32_t*, int32_t*, size_t);
template __global__ void fusionDecodePassKernel<int64_t, uint64_t, FusionMode::ZIGZAG>(const uint64_t*, int64_t*, size_t);

} // namespace fz

// ── FZM-header reconstruction (self-registered; see stage_registry.h) ─────────
namespace {
fz::Stage* Difference_fromHeader(const uint8_t* config, size_t config_size) {
    using fz::DataType; using fz::FusionMode; using fz::DifferenceStage; using fz::Stage;
    if (config_size >= 2) {
        DataType tin_dt  = static_cast<DataType>(config[0]);
        DataType tout_dt = static_cast<DataType>(config[1]);
        FusionMode mode = FusionMode::NEGABINARY;
        if (config_size >= 7) mode = static_cast<FusionMode>(config[6]);
        Stage* stage = nullptr;
        if (tin_dt == DataType::INT8 && tout_dt == DataType::UINT8)
            stage = (mode == FusionMode::ZIGZAG)
                ? static_cast<Stage*>(new DifferenceStage<int8_t, uint8_t, FusionMode::ZIGZAG>())
                : static_cast<Stage*>(new DifferenceStage<int8_t, uint8_t, FusionMode::NEGABINARY>());
        else if (tin_dt == DataType::INT16 && tout_dt == DataType::UINT16)
            stage = (mode == FusionMode::ZIGZAG)
                ? static_cast<Stage*>(new DifferenceStage<int16_t, uint16_t, FusionMode::ZIGZAG>())
                : static_cast<Stage*>(new DifferenceStage<int16_t, uint16_t, FusionMode::NEGABINARY>());
        else if (tin_dt == DataType::INT32 && tout_dt == DataType::UINT32)
            stage = (mode == FusionMode::ZIGZAG)
                ? static_cast<Stage*>(new DifferenceStage<int32_t, uint32_t, FusionMode::ZIGZAG>())
                : static_cast<Stage*>(new DifferenceStage<int32_t, uint32_t, FusionMode::NEGABINARY>());
        else if (tin_dt == DataType::INT64 && tout_dt == DataType::UINT64)
            stage = (mode == FusionMode::ZIGZAG)
                ? static_cast<Stage*>(new DifferenceStage<int64_t, uint64_t, FusionMode::ZIGZAG>())
                : static_cast<Stage*>(new DifferenceStage<int64_t, uint64_t, FusionMode::NEGABINARY>());
        else if (tin_dt == DataType::FLOAT32)  stage = new DifferenceStage<float>();
        else if (tin_dt == DataType::FLOAT64)  stage = new DifferenceStage<double>();
        else if (tin_dt == DataType::UINT8)    stage = new DifferenceStage<uint8_t>();
        else if (tin_dt == DataType::UINT16)   stage = new DifferenceStage<uint16_t>();
        else if (tin_dt == DataType::UINT32)   stage = new DifferenceStage<uint32_t>();
        else if (tin_dt == DataType::INT32)    stage = new DifferenceStage<int32_t>();
        else if (tin_dt == DataType::INT64)    stage = new DifferenceStage<int64_t>();
        else throw std::runtime_error("Unsupported Difference data type: "
                + std::to_string(static_cast<int>(tin_dt)));
        stage->deserializeHeader(config, config_size);
        return stage;
    } else if (config_size >= 1) {
        DataType dt = static_cast<DataType>(config[0]);
        switch (dt) {
            case DataType::FLOAT32:  return new DifferenceStage<float>();
            case DataType::FLOAT64:  return new DifferenceStage<double>();
            case DataType::UINT8:    return new DifferenceStage<uint8_t>();
            case DataType::UINT16:   return new DifferenceStage<uint16_t>();
            case DataType::UINT32:   return new DifferenceStage<uint32_t>();
            case DataType::INT32:    return new DifferenceStage<int32_t>();
            case DataType::INT64:    return new DifferenceStage<int64_t>();
            default: throw std::runtime_error("Unsupported Difference data type: "
                        + std::to_string(static_cast<int>(dt)));
        }
    }
    return new DifferenceStage<float>();
}
}  // namespace
FZ_REGISTER_STAGE_FACTORY(fz::StageType::DIFFERENCE, Difference_fromHeader);
