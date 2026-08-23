#include "coders/bitpack/bitpack_stage.h"
#include "stage/stage_registry.h"
#include <cstring>
#include <algorithm>
#include <stdexcept>
#include <string>
#include "backend/algorithms.h"
#include "mem/mempool.h"
#include "cuda_check.h"
#include "backend/cub.h"
#include "backend/api.h"
#include <stdexcept>
#include <string>

namespace fz {

// ─────────────────────────────────────────────────────────────────────────────
// Device kernels
//
// Restriction: nbits must be a power of two and divide 8*sizeof(T).
// Three cases by nbits relative to 8:
//
//   nbits < 8  (sub-byte): multiple elements share one output byte.
//              One thread per OUTPUT BYTE packs all (8/nbits) elements
//              that belong to it.  No thread contention on any output byte.
//
//   nbits == 8: one element per byte.  One thread per element.
//
//   nbits > 8  (multi-byte): one element spans (nbits/8) bytes.
//              One thread per element; no byte is shared.
//
// Every encode path applies the shift transform `(v - base) >> shift` before
// masking, and every decode path inverts it with `(packed << shift) + base`.
// base == 0 && shift == 0 is the identity, so the default path costs one
// subtract and one shift per element.
//
// ─────────────────────────────────────────────────────────────────────────────

template<typename T>
__device__ __forceinline__ T bitpackFwdTransform(T v, T base, uint8_t shift) {
    return static_cast<T>(static_cast<T>(v - base) >> shift);
}

template<typename T>
__device__ __forceinline__ T bitpackInvTransform(T v, T base, uint8_t shift) {
    return static_cast<T>(static_cast<T>(v << shift) + base);
}

// Sub-byte encode: one thread per output byte.
template<typename T>
__global__ void bitpackEncodeSubByteKernel(
    const T*  __restrict__ in,
    uint8_t*  __restrict__ out,
    size_t n_out_bytes,    // number of output bytes
    size_t n_elements,
    uint8_t nbits,
    T       base,
    uint8_t shift)
{
    const size_t byte_idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (byte_idx >= n_out_bytes) return;

    const uint8_t elems_per_byte = 8 / nbits;
    const T mask = static_cast<T>((T(1) << nbits) - T(1));

    uint8_t packed = 0;
    const size_t base_elem = byte_idx * elems_per_byte;
    for (uint8_t k = 0; k < elems_per_byte; ++k) {
        const size_t elem_idx = base_elem + k;
        if (elem_idx < n_elements) {
            const T v = bitpackFwdTransform<T>(in[elem_idx], base, shift);
            packed |= static_cast<uint8_t>((v & mask) << (k * nbits));
        }
    }
    out[byte_idx] = packed;
}

// Sub-byte decode: one thread per output element.
template<typename T>
__global__ void bitpackDecodeSubByteKernel(
    const uint8_t* __restrict__ in,
    T*             __restrict__ out,
    size_t n,
    uint8_t nbits,
    T       base,
    uint8_t shift)
{
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const uint8_t elems_per_byte = 8 / nbits;
    const uint8_t mask = static_cast<uint8_t>((1u << nbits) - 1u);
    const size_t byte_idx  = idx / elems_per_byte;
    const uint8_t slot     = static_cast<uint8_t>(idx % elems_per_byte);
    const T v = static_cast<T>((in[byte_idx] >> (slot * nbits)) & mask);
    out[idx] = bitpackInvTransform<T>(v, base, shift);
}

// Multi-byte encode/decode (nbits >= 8): one thread per element.
template<typename T>
__global__ void bitpackEncodeMultiByteKernel(
    const T*  __restrict__ in,
    uint8_t*  __restrict__ out,
    size_t n,
    uint8_t nbits,
    T       base,
    uint8_t shift)
{
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const T mask = static_cast<T>((nbits == 8 * sizeof(T))
                                  ? ~T(0)
                                  : (T(1) << nbits) - T(1));
    T v = bitpackFwdTransform<T>(in[idx], base, shift) & mask;
    const int bytes = nbits / 8;
    const size_t byte_offset = idx * bytes;
    for (int b = 0; b < bytes; ++b) {
        out[byte_offset + b] = static_cast<uint8_t>(v & 0xFF);
        if constexpr (sizeof(T) > 1) v >>= 8;
    }
}

template<typename T>
__global__ void bitpackDecodeMultiByteKernel(
    const uint8_t* __restrict__ in,
    T*             __restrict__ out,
    size_t n,
    uint8_t nbits,
    T       base,
    uint8_t shift)
{
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const int bytes = nbits / 8;
    const size_t byte_offset = idx * bytes;
    T val = T(0);
    for (int b = 0; b < bytes; ++b)
        val |= static_cast<T>(in[byte_offset + b]) << (8 * b);
    out[idx] = bitpackInvTransform<T>(val, base, shift);
}

// Auto-shift scan: OR-reduce of (v - base) across the whole input.  The
// trailing-zero count of the result is the largest right shift that drops no
// information.  A uint32_t accumulator covers every supported T.
template<typename T>
__global__ void bitpackOrReduceKernel(
    const T* __restrict__ in,
    size_t n,
    T base,
    unsigned int* __restrict__ out)
{
    __shared__ unsigned int s_or;
    if (threadIdx.x == 0) s_or = 0u;
    __syncthreads();

    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    unsigned int local = 0u;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += stride)
        local |= static_cast<unsigned int>(static_cast<T>(in[i] - base));

    atomicOr(&s_or, local);
    __syncthreads();
    if (threadIdx.x == 0 && s_or != 0u) atomicOr(out, s_or);
}

// ─────────────────────────────────────────────────────────────────────────────
// Auto-detect helpers
// ─────────────────────────────────────────────────────────────────────────────

// Trailing-zero count of a non-zero 32-bit word.
static uint8_t ctz32(uint32_t v) {
    uint8_t c = 0;
    while ((v & 1u) == 0u) { v >>= 1; ++c; }
    return c;
}

// Smallest power-of-two nbits that can represent max_val.
template<typename T>
static uint8_t nbits_for_max(T max_val) {
    const uint8_t full = static_cast<uint8_t>(8 * sizeof(T));
    for (uint8_t b = 1; b < full; b = static_cast<uint8_t>(b * 2)) {
        if (max_val < (T(1) << b)) return b;
    }
    return full;
}

// Largest lossless right shift for `in`, given the frame-of-reference base.
template<typename T>
static uint8_t detectShift(
    const T* in, size_t n, T base, MemoryPool* pool, cudaStream_t stream)
{
    // Scratch goes through the pool so all device memory stays tracked; fall
    // back to cudaMalloc if the pool returns null (vGPU / fallback mode).
    unsigned int* d_or = static_cast<unsigned int*>(
        pool->allocate(sizeof(unsigned int), stream, "bitpack_or"));
    const bool pooled = (d_or != nullptr);
    if (!pooled) FZ_CUDA_CHECK(cudaMalloc(&d_or, sizeof(unsigned int)));
    FZ_CUDA_CHECK(cudaMemsetAsync(d_or, 0, sizeof(unsigned int), stream));

    constexpr int kBlock = 256;
    int grid = static_cast<int>((n + kBlock - 1) / kBlock);
    if (grid > 4096) grid = 4096;   // the grid-stride loop covers the remainder
    bitpackOrReduceKernel<T><<<grid, kBlock, 0, stream>>>(in, n, base, d_or);

    unsigned int h_or = 0u;
    FZ_CUDA_CHECK(cudaMemcpyAsync(&h_or, d_or, sizeof(unsigned int),
                                  cudaMemcpyDeviceToHost, stream));
    FZ_CUDA_CHECK(cudaStreamSynchronize(stream));

    if (pooled) pool->free(d_or, stream); else FZ_CUDA_CHECK(cudaFree(d_or));

    // h_or == 0 means every element equals base — nothing to shift.
    return (h_or == 0u) ? uint8_t(0) : ctz32(h_or);
}

// ─────────────────────────────────────────────────────────────────────────────
// Launcher helpers
// ─────────────────────────────────────────────────────────────────────────────

template<typename T>
static void launchEncode(
    const T* in, uint8_t* out, size_t n, uint8_t nbits,
    T base, uint8_t shift, cudaStream_t stream)
{
    if (n == 0) return;
    constexpr int kBlock = 256;

    if (nbits < 8) {
        // One thread per output byte — no two threads touch the same output byte.
        const size_t n_out_bytes = (n * nbits + 7) / 8;
        const int grid = static_cast<int>((n_out_bytes + kBlock - 1) / kBlock);
        bitpackEncodeSubByteKernel<T><<<grid, kBlock, 0, stream>>>(
            in, out, n_out_bytes, n, nbits, base, shift);
    } else {
        // One thread per element (nbits >= 8, so no shared-byte contention)
        const int grid = static_cast<int>((n + kBlock - 1) / kBlock);
        bitpackEncodeMultiByteKernel<T><<<grid, kBlock, 0, stream>>>(
            in, out, n, nbits, base, shift);
    }
}

template<typename T>
static void launchDecode(
    const uint8_t* in, T* out, size_t n, uint8_t nbits,
    T base, uint8_t shift, cudaStream_t stream)
{
    if (n == 0) return;
    constexpr int kBlock = 256;
    const int grid = static_cast<int>((n + kBlock - 1) / kBlock);

    if (nbits < 8) {
        bitpackDecodeSubByteKernel<T><<<grid, kBlock, 0, stream>>>(
            in, out, n, nbits, base, shift);
    } else {
        bitpackDecodeMultiByteKernel<T><<<grid, kBlock, 0, stream>>>(
            in, out, n, nbits, base, shift);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// BitpackStage::execute
// ─────────────────────────────────────────────────────────────────────────────

template<typename T>
void BitpackStage<T>::execute(
    cudaStream_t stream,
    MemoryPool* pool,
    const std::vector<void*>& inputs,
    const std::vector<void*>& outputs,
    const std::vector<size_t>& sizes)
{
    if (inputs.empty() || outputs.empty() || sizes.empty())
        throw std::runtime_error("BitpackStage: inputs, outputs, and sizes must be non-empty");

    const size_t in_bytes = sizes[0];

    if (!is_inverse_) {
        // ── Forward: T[] → uint8_t[] ─────────────────────────────────────────
        const size_t n = in_bytes / sizeof(T);
        if (n == 0) { actual_output_size_ = 0; num_elements_ = 0; return; }

        const T* d_in = static_cast<const T*>(inputs[0]);

        // One device-wide CUB reduce plus a blocking readback into a host
        // scalar.  Scratch goes through the pool so all device memory stays
        // tracked; fall back to cudaMalloc if the pool returns null (vGPU /
        // fallback mode).  Every auto-* mode is already excluded from graph
        // capture, so the sync here is safe.
        auto reduceScalar = [&](auto reduce_fn, const char* tag) -> T {
            T* d_val = static_cast<T*>(pool->allocate(sizeof(T), stream, tag));
            const bool pooled = (d_val != nullptr);
            if (!pooled) FZ_CUDA_CHECK(cudaMalloc(&d_val, sizeof(T)));

            auto d_tmp = fz::backend::withTempStorage(pool, stream, "bitpack_cub_tmp",
                [&](void* tmp, size_t& bytes) { reduce_fn(tmp, bytes, d_val); });

            T h_val = T(0);
            FZ_CUDA_CHECK(cudaMemcpyAsync(&h_val, d_val, sizeof(T),
                                          cudaMemcpyDeviceToHost, stream));
            FZ_CUDA_CHECK(cudaStreamSynchronize(stream));

            if (pooled) pool->free(d_val, stream); else FZ_CUDA_CHECK(cudaFree(d_val));
            fz::backend::freeTempStorage(pool, d_tmp, stream);
            return h_val;
        };

        // Order matters: base first (shift is measured on v - base), then
        // shift, then nbits (measured on the fully transformed range).
        if (auto_base_) {
            base_ = reduceScalar([&](void* tmp, size_t& bytes, T* d_out) {
                cub::DeviceReduce::Min(tmp, bytes, d_in, d_out,
                                       static_cast<int>(n), stream);
            }, "bitpack_min");
        }

        if (auto_shift_) {
            shift_ = detectShift<T>(d_in, n, base_, pool, stream);
        }

        if (auto_detect_) {
            const T h_max = reduceScalar([&](void* tmp, size_t& bytes, T* d_out) {
                cub::DeviceReduce::Max(tmp, bytes, d_in, d_out,
                                       static_cast<int>(n), stream);
            }, "bitpack_max");
            nbits_ = nbits_for_max<T>(
                static_cast<T>(static_cast<T>(h_max - base_) >> shift_));
        }

        num_elements_ = n;
        launchEncode<T>(
            d_in,
            static_cast<uint8_t*>(outputs[0]),
            n, nbits_, base_, shift_, stream);

        actual_output_size_ = (n * nbits_ + 7) / 8;
    } else {
        // ── Inverse: uint8_t[] → T[] ─────────────────────────────────────────
        // num_elements_, shift_, and base_ were restored from the file header
        // by deserializeHeader.
        const size_t n = num_elements_;
        if (n == 0) { actual_output_size_ = 0; return; }

        launchDecode<T>(
            static_cast<const uint8_t*>(inputs[0]),
            static_cast<T*>(outputs[0]),
            n, nbits_, base_, shift_, stream);

        actual_output_size_ = n * sizeof(T);
    }

    FZ_CUDA_CHECK(cudaGetLastError());
}

// ─────────────────────────────────────────────────────────────────────────────
// Explicit template instantiations
// ─────────────────────────────────────────────────────────────────────────────

template class BitpackStage<uint8_t>;
template class BitpackStage<uint16_t>;
template class BitpackStage<uint32_t>;

template __global__ void bitpackEncodeSubByteKernel<uint8_t> (const  uint8_t*, uint8_t*, size_t, size_t, uint8_t,  uint8_t, uint8_t);
template __global__ void bitpackEncodeSubByteKernel<uint16_t>(const uint16_t*, uint8_t*, size_t, size_t, uint8_t, uint16_t, uint8_t);
template __global__ void bitpackEncodeSubByteKernel<uint32_t>(const uint32_t*, uint8_t*, size_t, size_t, uint8_t, uint32_t, uint8_t);

template __global__ void bitpackEncodeMultiByteKernel<uint8_t> (const  uint8_t*, uint8_t*, size_t, uint8_t,  uint8_t, uint8_t);
template __global__ void bitpackEncodeMultiByteKernel<uint16_t>(const uint16_t*, uint8_t*, size_t, uint8_t, uint16_t, uint8_t);
template __global__ void bitpackEncodeMultiByteKernel<uint32_t>(const uint32_t*, uint8_t*, size_t, uint8_t, uint32_t, uint8_t);

template __global__ void bitpackDecodeSubByteKernel<uint8_t> (const uint8_t*,  uint8_t*, size_t, uint8_t,  uint8_t, uint8_t);
template __global__ void bitpackDecodeSubByteKernel<uint16_t>(const uint8_t*, uint16_t*, size_t, uint8_t, uint16_t, uint8_t);
template __global__ void bitpackDecodeSubByteKernel<uint32_t>(const uint8_t*, uint32_t*, size_t, uint8_t, uint32_t, uint8_t);

template __global__ void bitpackDecodeMultiByteKernel<uint8_t> (const uint8_t*,  uint8_t*, size_t, uint8_t,  uint8_t, uint8_t);
template __global__ void bitpackDecodeMultiByteKernel<uint16_t>(const uint8_t*, uint16_t*, size_t, uint8_t, uint16_t, uint8_t);
template __global__ void bitpackDecodeMultiByteKernel<uint32_t>(const uint8_t*, uint32_t*, size_t, uint8_t, uint32_t, uint8_t);

template __global__ void bitpackOrReduceKernel<uint8_t> (const  uint8_t*, size_t,  uint8_t, unsigned int*);
template __global__ void bitpackOrReduceKernel<uint16_t>(const uint16_t*, size_t, uint16_t, unsigned int*);
template __global__ void bitpackOrReduceKernel<uint32_t>(const uint32_t*, size_t, uint32_t, unsigned int*);

} // namespace fz

// ── FZM-header reconstruction (self-registered; see stage_registry.h) ─────────
namespace {
fz::Stage* Bitpack_fromHeader(const uint8_t* config, size_t config_size) {
    using fz::DataType; using fz::BitpackStage; using fz::Stage;
    DataType dt = (config_size > 0) ? static_cast<DataType>(config[0]) : DataType::UINT16;
    Stage* stage = nullptr;
    if      (dt == DataType::UINT8)  stage = new BitpackStage<uint8_t>();
    else if (dt == DataType::UINT16) stage = new BitpackStage<uint16_t>();
    else if (dt == DataType::UINT32) stage = new BitpackStage<uint32_t>();
    else throw std::runtime_error("Unsupported BitpackStage DataType: "
            + std::to_string(static_cast<int>(dt)));
    stage->deserializeHeader(config, config_size);
    return stage;
}
}  // namespace
FZ_REGISTER_STAGE_FACTORY(fz::StageType::BITPACK, Bitpack_fromHeader);
