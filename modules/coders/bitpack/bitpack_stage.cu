#include "coders/bitpack/bitpack_stage.h"
#include "mem/mempool.h"
#include "cuda_check.h"
#include <cuda_runtime.h>
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
// ─────────────────────────────────────────────────────────────────────────────

// Sub-byte encode: one thread per output byte.
template<typename T>
__global__ void bitpackEncodeSubByteKernel(
    const T*  __restrict__ in,
    uint8_t*  __restrict__ out,
    size_t n_out_bytes,    // number of output bytes
    size_t n_elements,
    uint8_t nbits)
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
            packed |= static_cast<uint8_t>((in[elem_idx] & mask) << (k * nbits));
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
    uint8_t nbits)
{
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const uint8_t elems_per_byte = 8 / nbits;
    const uint8_t mask = static_cast<uint8_t>((1u << nbits) - 1u);
    const size_t byte_idx  = idx / elems_per_byte;
    const uint8_t slot     = static_cast<uint8_t>(idx % elems_per_byte);
    out[idx] = static_cast<T>((in[byte_idx] >> (slot * nbits)) & mask);
}

// Multi-byte encode/decode (nbits >= 8): one thread per element.
template<typename T>
__global__ void bitpackEncodeMultiByteKernel(
    const T*  __restrict__ in,
    uint8_t*  __restrict__ out,
    size_t n,
    uint8_t nbits)
{
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const T mask = static_cast<T>((nbits == 8 * sizeof(T))
                                  ? ~T(0)
                                  : (T(1) << nbits) - T(1));
    T v = in[idx] & mask;
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
    uint8_t nbits)
{
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const int bytes = nbits / 8;
    const size_t byte_offset = idx * bytes;
    T val = T(0);
    for (int b = 0; b < bytes; ++b)
        val |= static_cast<T>(in[byte_offset + b]) << (8 * b);
    out[idx] = val;
}

// ─────────────────────────────────────────────────────────────────────────────
// Launcher helpers
// ─────────────────────────────────────────────────────────────────────────────

template<typename T>
static void launchEncode(
    const T* in, uint8_t* out, size_t n, uint8_t nbits, cudaStream_t stream)
{
    if (n == 0) return;
    constexpr int kBlock = 256;

    if (nbits < 8) {
        // One thread per output byte — no two threads touch the same output byte.
        const size_t n_out_bytes = (n * nbits + 7) / 8;
        const int grid = static_cast<int>((n_out_bytes + kBlock - 1) / kBlock);
        bitpackEncodeSubByteKernel<T><<<grid, kBlock, 0, stream>>>(in, out, n_out_bytes, n, nbits);
    } else {
        // One thread per element (nbits >= 8, so no shared-byte contention)
        const int grid = static_cast<int>((n + kBlock - 1) / kBlock);
        bitpackEncodeMultiByteKernel<T><<<grid, kBlock, 0, stream>>>(in, out, n, nbits);
    }
}

template<typename T>
static void launchDecode(
    const uint8_t* in, T* out, size_t n, uint8_t nbits, cudaStream_t stream)
{
    if (n == 0) return;
    constexpr int kBlock = 256;
    const int grid = static_cast<int>((n + kBlock - 1) / kBlock);

    if (nbits < 8) {
        bitpackDecodeSubByteKernel<T><<<grid, kBlock, 0, stream>>>(in, out, n, nbits);
    } else {
        bitpackDecodeMultiByteKernel<T><<<grid, kBlock, 0, stream>>>(in, out, n, nbits);
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
    (void)pool;

    if (inputs.empty() || outputs.empty() || sizes.empty())
        throw std::runtime_error("BitpackStage: inputs, outputs, and sizes must be non-empty");

    const size_t in_bytes = sizes[0];

    if (!is_inverse_) {
        // ── Forward: T[] → uint8_t[] ─────────────────────────────────────────
        const size_t n = in_bytes / sizeof(T);
        if (n == 0) { actual_output_size_ = 0; num_elements_ = 0; return; }

        num_elements_ = n;
        launchEncode<T>(
            static_cast<const T*>(inputs[0]),
            static_cast<uint8_t*>(outputs[0]),
            n, nbits_, stream);

        actual_output_size_ = (n * nbits_ + 7) / 8;
    } else {
        // ── Inverse: uint8_t[] → T[] ─────────────────────────────────────────
        // num_elements_ was restored from the file header by deserializeHeader.
        const size_t n = num_elements_;
        if (n == 0) { actual_output_size_ = 0; return; }

        launchDecode<T>(
            static_cast<const uint8_t*>(inputs[0]),
            static_cast<T*>(outputs[0]),
            n, nbits_, stream);

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

template __global__ void bitpackEncodeSubByteKernel<uint8_t> (const  uint8_t*, uint8_t*, size_t, size_t, uint8_t);
template __global__ void bitpackEncodeSubByteKernel<uint16_t>(const uint16_t*, uint8_t*, size_t, size_t, uint8_t);
template __global__ void bitpackEncodeSubByteKernel<uint32_t>(const uint32_t*, uint8_t*, size_t, size_t, uint8_t);

template __global__ void bitpackEncodeMultiByteKernel<uint8_t> (const  uint8_t*, uint8_t*, size_t, uint8_t);
template __global__ void bitpackEncodeMultiByteKernel<uint16_t>(const uint16_t*, uint8_t*, size_t, uint8_t);
template __global__ void bitpackEncodeMultiByteKernel<uint32_t>(const uint32_t*, uint8_t*, size_t, uint8_t);

template __global__ void bitpackDecodeSubByteKernel<uint8_t> (const uint8_t*,  uint8_t*, size_t, uint8_t);
template __global__ void bitpackDecodeSubByteKernel<uint16_t>(const uint8_t*, uint16_t*, size_t, uint8_t);
template __global__ void bitpackDecodeSubByteKernel<uint32_t>(const uint8_t*, uint32_t*, size_t, uint8_t);

template __global__ void bitpackDecodeMultiByteKernel<uint8_t> (const uint8_t*,  uint8_t*, size_t, uint8_t);
template __global__ void bitpackDecodeMultiByteKernel<uint16_t>(const uint8_t*, uint16_t*, size_t, uint8_t);
template __global__ void bitpackDecodeMultiByteKernel<uint32_t>(const uint8_t*, uint32_t*, size_t, uint8_t);

} // namespace fz
