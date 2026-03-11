/**
 * modules/transforms/bitshuffle/bitshuffle_stage.cu
 *
 * GPU implementation of BitshuffleStage using the butterfly (warp-shuffle)
 * algorithm from the LC framework (Burtscher et al., BSD-3 licensed).
 *
 * Algorithm overview
 * ------------------
 * Rather than extracting one bit per thread per bit-plane loop iteration
 * (__ballot_sync), the butterfly approach does the W×N bit-matrix transpose
 * in a single pass over the elements using W register-level butterfly stages:
 *
 *   For element width W = 32 bits (4-byte) — adapted from d_BIT_4:
 *     Five __shfl_xor_sync stages (distances 16, 8, 4, 2, 1) interleaved with
 *     __byte_perm / nibble / bit-pair / bit shuffles transform each thread's
 *     register so that after the butterfly sublane s holds the contribution of
 *     bit-plane s from all 32 elements in its warp group.
 *
 *   For element width W = 64 bits (8-byte) — adapted from d_BIT_8:
 *     Each thread holds two uint64 values (elements i and i+32).  A cross-
 *     register 32-bit swap forms the first stage, followed by five
 *     __shfl_xor_sync stages (16, 8, 4, 2, 1) operating on both registers.
 *
 *   For element widths 1 and 2 bytes:
 *     The compact __ballot_sync approach is used (correct and adequate for
 *     non-primary element widths).
 *
 * Output layout (all widths): MSB-first — bit-plane W-1 (MSBit) is at plane
 * index 0; bit-plane 0 (LSBit) is at plane index W-1.  This matches the
 * natural output of the 4/8-byte butterfly where sublane 0 collects the
 * highest bit of each element.
 *   Plane p occupies words  p * (N_chunk/32)  through  (p+1)*(N_chunk/32) - 1
 *   where N_chunk = block_size_bytes / element_width.
 *
 * Block mapping: one CUDA block per chunk.
 *   Butterfly kernels (4/8 byte): blockDim.x = 1024.
 *   Ballot kernels (1/2 byte):    blockDim.x = min(N_chunk, 1024).
 *
 * Chunk-size constraint: block_size must be a multiple of 1024 * element_width.
 * This ensures N_chunk is always a multiple of 1024, so every stride iteration
 * has full warps for the __shfl_xor_sync calls.
 */

#include "transforms/bitshuffle/bitshuffle_stage.h"
#include "mem/mempool.h"
#include "cuda_check.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <algorithm>

namespace fz {

// ─────────────────────────────────────────────────────────────────────────────
// 4-byte (32-bit) butterfly kernels — ported from d_BIT_4 / d_iBIT_4
// (LC framework, Burtscher et al.)
//
// Each thread processes element i = threadIdx.x + k*blockDim.x (grid-stride).
// After the 5-stage butterfly, thread at position (sublane = i % 32) holds the
// contribution of bit-plane sublane for all 32 elements in that warp group.
//
// Output: plane p at in-chunk positions  p*(N_chunk/32) .. (p+1)*(N_chunk/32)-1
// ─────────────────────────────────────────────────────────────────────────────

__global__ void bitshuffleEncodeKernel32(
    const uint32_t* __restrict__ in,
    uint32_t*       __restrict__ out,
    uint32_t N_chunk)
{
    const int      tid      = (int)threadIdx.x;
    const int      sublane  = tid % 32;
    const uint32_t in_base  = blockIdx.x * N_chunk;
    const uint32_t out_base = blockIdx.x * N_chunk;
    const int      npp      = (int)(N_chunk / 32u);  // words per bit-plane

    for (int i = tid; i < (int)N_chunk; i += (int)blockDim.x) {
        unsigned int a = in[in_base + i];

        unsigned int q = __shfl_xor_sync(0xFFFFFFFFu, a, 16);
        a = ((sublane & 16) == 0)
            ? __byte_perm(a, q, (3u<<12)|(2u<<8)|(7u<<4)|6u)
            : __byte_perm(a, q, (5u<<12)|(4u<<8)|(1u<<4)|0u);

        q = __shfl_xor_sync(0xFFFFFFFFu, a, 8);
        a = ((sublane & 8) == 0)
            ? __byte_perm(a, q, (3u<<12)|(7u<<8)|(1u<<4)|5u)
            : __byte_perm(a, q, (6u<<12)|(2u<<8)|(4u<<4)|0u);

        q = __shfl_xor_sync(0xFFFFFFFFu, a, 4);
        unsigned int mask = 0x0F0F0F0Fu;
        a = ((sublane & 4) == 0)
            ? ((a & ~mask) | ((q >> 4) & mask))
            : (((q << 4) & ~mask) | (a & mask));

        q = __shfl_xor_sync(0xFFFFFFFFu, a, 2);
        mask = 0x33333333u;
        a = ((sublane & 2) == 0)
            ? ((a & ~mask) | ((q >> 2) & mask))
            : (((q << 2) & ~mask) | (a & mask));

        q = __shfl_xor_sync(0xFFFFFFFFu, a, 1);
        mask = 0x55555555u;
        a = ((sublane & 1) == 0)
            ? ((a & ~mask) | ((q >> 1) & mask))
            : (((q << 1) & ~mask) | (a & mask));

        // LSB-first: plane sublane at word offset i/32 + sublane*npp
        out[out_base + i / 32 + sublane * npp] = a;
    }
}

__global__ void bitshuffleDecodeKernel32(
    const uint32_t* __restrict__ in,
    uint32_t*       __restrict__ out,
    uint32_t N_chunk)
{
    const int      tid      = (int)threadIdx.x;
    const int      sublane  = tid % 32;
    const uint32_t in_base  = blockIdx.x * N_chunk;
    const uint32_t out_base = blockIdx.x * N_chunk;
    const int      npp      = (int)(N_chunk / 32u);

    for (int i = tid; i < (int)N_chunk; i += (int)blockDim.x) {
        // Read from plane-organised layout (butterfly is self-inverse)
        unsigned int a = in[in_base + i / 32 + sublane * npp];

        unsigned int q = __shfl_xor_sync(0xFFFFFFFFu, a, 16);
        a = ((sublane & 16) == 0)
            ? __byte_perm(a, q, (3u<<12)|(2u<<8)|(7u<<4)|6u)
            : __byte_perm(a, q, (5u<<12)|(4u<<8)|(1u<<4)|0u);

        q = __shfl_xor_sync(0xFFFFFFFFu, a, 8);
        a = ((sublane & 8) == 0)
            ? __byte_perm(a, q, (3u<<12)|(7u<<8)|(1u<<4)|5u)
            : __byte_perm(a, q, (6u<<12)|(2u<<8)|(4u<<4)|0u);

        q = __shfl_xor_sync(0xFFFFFFFFu, a, 4);
        unsigned int mask = 0x0F0F0F0Fu;
        a = ((sublane & 4) == 0)
            ? ((a & ~mask) | ((q >> 4) & mask))
            : (((q << 4) & ~mask) | (a & mask));

        q = __shfl_xor_sync(0xFFFFFFFFu, a, 2);
        mask = 0x33333333u;
        a = ((sublane & 2) == 0)
            ? ((a & ~mask) | ((q >> 2) & mask))
            : (((q << 2) & ~mask) | (a & mask));

        q = __shfl_xor_sync(0xFFFFFFFFu, a, 1);
        mask = 0x55555555u;
        a = ((sublane & 1) == 0)
            ? ((a & ~mask) | ((q >> 1) & mask))
            : (((q << 1) & ~mask) | (a & mask));

        out[out_base + i] = a;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 8-byte (64-bit) butterfly kernels — ported from d_BIT_8 / d_iBIT_8
// (LC framework, Burtscher et al.)
//
// Each thread processes TWO consecutive elements: positions i and i+32.
// The first butterfly stage is a cross-register swap of upper/lower 32 bits
// (not a warp shuffle); the remaining 5 stages use __shfl_xor_sync on both
// registers in parallel.
//
// Start index per thread: subwarp*64 + sublane  (subwarp = tid/32)
// Stride: blockDim.x * 2
//
// Output: plane p (0..31 via sublane, 32..63 via sublane+32) at positions
//   p * (N_chunk/64) .. (p+1)*(N_chunk/64) - 1   (uint64 words)
// ─────────────────────────────────────────────────────────────────────────────

__global__ void bitshuffleEncodeKernel64(
    const uint64_t* __restrict__ in,
    uint64_t*       __restrict__ out,
    uint32_t N_chunk)
{
    const int      tid      = (int)threadIdx.x;
    const int      subwarp  = tid / 32;
    const int      sublane  = tid % 32;
    const uint32_t in_base  = blockIdx.x * N_chunk;
    const uint32_t out_base = blockIdx.x * N_chunk;
    const int      npp      = (int)(N_chunk / 64u);  // uint64 words per bit-plane

    for (int i = subwarp * 64 + sublane; i < (int)N_chunk; i += (int)(blockDim.x * 2)) {
        unsigned long long a0 = in[in_base + i];
        unsigned long long a1 = in[in_base + i + 32];

        // Stage 0 — cross-register 32-bit half-swap
        unsigned long long b0 = a1, b1 = a0;
        unsigned long long m = 0x00000000FFFFFFFFull;
        a0 = (a0 & ~m) | (b0 >> 32);
        a1 = (a1 &  m) | (b1 << 32);

        unsigned long long q0, q1;

        q0 = __shfl_xor_sync(0xFFFFFFFFu, a0, 16);
        q1 = __shfl_xor_sync(0xFFFFFFFFu, a1, 16);
        m = 0x0000FFFF0000FFFFull;
        a0 = ((sublane & 16) == 0) ? ((a0 & ~m) | ((q0 >> 16) & m)) : ((a0 & m) | ((q0 << 16) & ~m));
        a1 = ((sublane & 16) == 0) ? ((a1 & ~m) | ((q1 >> 16) & m)) : ((a1 & m) | ((q1 << 16) & ~m));

        q0 = __shfl_xor_sync(0xFFFFFFFFu, a0, 8);
        q1 = __shfl_xor_sync(0xFFFFFFFFu, a1, 8);
        m = 0x00FF00FF00FF00FFull;
        a0 = ((sublane & 8) == 0) ? ((a0 & ~m) | ((q0 >> 8) & m)) : ((a0 & m) | ((q0 << 8) & ~m));
        a1 = ((sublane & 8) == 0) ? ((a1 & ~m) | ((q1 >> 8) & m)) : ((a1 & m) | ((q1 << 8) & ~m));

        q0 = __shfl_xor_sync(0xFFFFFFFFu, a0, 4);
        q1 = __shfl_xor_sync(0xFFFFFFFFu, a1, 4);
        m = 0x0F0F0F0F0F0F0F0Full;
        a0 = ((sublane & 4) == 0) ? ((a0 & ~m) | ((q0 >> 4) & m)) : ((a0 & m) | ((q0 << 4) & ~m));
        a1 = ((sublane & 4) == 0) ? ((a1 & ~m) | ((q1 >> 4) & m)) : ((a1 & m) | ((q1 << 4) & ~m));

        q0 = __shfl_xor_sync(0xFFFFFFFFu, a0, 2);
        q1 = __shfl_xor_sync(0xFFFFFFFFu, a1, 2);
        m = 0x3333333333333333ull;
        a0 = ((sublane & 2) == 0) ? ((a0 & ~m) | ((q0 >> 2) & m)) : ((a0 & m) | ((q0 << 2) & ~m));
        a1 = ((sublane & 2) == 0) ? ((a1 & ~m) | ((q1 >> 2) & m)) : ((a1 & m) | ((q1 << 2) & ~m));

        q0 = __shfl_xor_sync(0xFFFFFFFFu, a0, 1);
        q1 = __shfl_xor_sync(0xFFFFFFFFu, a1, 1);
        m = 0x5555555555555555ull;
        a0 = ((sublane & 1) == 0) ? ((a0 & ~m) | ((q0 >> 1) & m)) : ((a0 & m) | ((q0 << 1) & ~m));
        a1 = ((sublane & 1) == 0) ? ((a1 & ~m) | ((q1 >> 1) & m)) : ((a1 & m) | ((q1 << 1) & ~m));

        // LSB-first: planes 0..31 via sublane, planes 32..63 via sublane+32
        out[out_base + i / 64 + sublane        * npp] = a0;
        out[out_base + i / 64 + (sublane + 32) * npp] = a1;
    }
}

__global__ void bitshuffleDecodeKernel64(
    const uint64_t* __restrict__ in,
    uint64_t*       __restrict__ out,
    uint32_t N_chunk)
{
    const int      tid      = (int)threadIdx.x;
    const int      subwarp  = tid / 32;
    const int      sublane  = tid % 32;
    const uint32_t in_base  = blockIdx.x * N_chunk;
    const uint32_t out_base = blockIdx.x * N_chunk;
    const int      npp      = (int)(N_chunk / 64u);

    for (int i = subwarp * 64 + sublane; i < (int)N_chunk; i += (int)(blockDim.x * 2)) {
        unsigned long long a0 = in[in_base + i / 64 + sublane        * npp];
        unsigned long long a1 = in[in_base + i / 64 + (sublane + 32) * npp];

        // Same butterfly (self-inverse)
        unsigned long long b0 = a1, b1 = a0;
        unsigned long long m = 0x00000000FFFFFFFFull;
        a0 = (a0 & ~m) | (b0 >> 32);
        a1 = (a1 &  m) | (b1 << 32);

        unsigned long long q0, q1;

        q0 = __shfl_xor_sync(0xFFFFFFFFu, a0, 16);
        q1 = __shfl_xor_sync(0xFFFFFFFFu, a1, 16);
        m = 0x0000FFFF0000FFFFull;
        a0 = ((sublane & 16) == 0) ? ((a0 & ~m) | ((q0 >> 16) & m)) : ((a0 & m) | ((q0 << 16) & ~m));
        a1 = ((sublane & 16) == 0) ? ((a1 & ~m) | ((q1 >> 16) & m)) : ((a1 & m) | ((q1 << 16) & ~m));

        q0 = __shfl_xor_sync(0xFFFFFFFFu, a0, 8);
        q1 = __shfl_xor_sync(0xFFFFFFFFu, a1, 8);
        m = 0x00FF00FF00FF00FFull;
        a0 = ((sublane & 8) == 0) ? ((a0 & ~m) | ((q0 >> 8) & m)) : ((a0 & m) | ((q0 << 8) & ~m));
        a1 = ((sublane & 8) == 0) ? ((a1 & ~m) | ((q1 >> 8) & m)) : ((a1 & m) | ((q1 << 8) & ~m));

        q0 = __shfl_xor_sync(0xFFFFFFFFu, a0, 4);
        q1 = __shfl_xor_sync(0xFFFFFFFFu, a1, 4);
        m = 0x0F0F0F0F0F0F0F0Full;
        a0 = ((sublane & 4) == 0) ? ((a0 & ~m) | ((q0 >> 4) & m)) : ((a0 & m) | ((q0 << 4) & ~m));
        a1 = ((sublane & 4) == 0) ? ((a1 & ~m) | ((q1 >> 4) & m)) : ((a1 & m) | ((q1 << 4) & ~m));

        q0 = __shfl_xor_sync(0xFFFFFFFFu, a0, 2);
        q1 = __shfl_xor_sync(0xFFFFFFFFu, a1, 2);
        m = 0x3333333333333333ull;
        a0 = ((sublane & 2) == 0) ? ((a0 & ~m) | ((q0 >> 2) & m)) : ((a0 & m) | ((q0 << 2) & ~m));
        a1 = ((sublane & 2) == 0) ? ((a1 & ~m) | ((q1 >> 2) & m)) : ((a1 & m) | ((q1 << 2) & ~m));

        q0 = __shfl_xor_sync(0xFFFFFFFFu, a0, 1);
        q1 = __shfl_xor_sync(0xFFFFFFFFu, a1, 1);
        m = 0x5555555555555555ull;
        a0 = ((sublane & 1) == 0) ? ((a0 & ~m) | ((q0 >> 1) & m)) : ((a0 & m) | ((q0 << 1) & ~m));
        a1 = ((sublane & 1) == 0) ? ((a1 & ~m) | ((q1 >> 1) & m)) : ((a1 & m) | ((q1 << 1) & ~m));

        out[out_base + i]      = a0;
        out[out_base + i + 32] = a1;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 1-byte and 2-byte — __ballot_sync approach (LSB-first layout)
// ─────────────────────────────────────────────────────────────────────────────

template<typename T>
__global__ void bitshuffleEncodeKernelBallot(
    const T*  __restrict__ in,
    uint32_t* __restrict__ out,
    uint32_t N_chunk)
{
    constexpr int  W        = static_cast<int>(sizeof(T) * 8);
    const uint32_t npp      = N_chunk / 32u;
    const uint32_t in_base  = blockIdx.x * N_chunk;
    const uint32_t out_base = blockIdx.x * (uint32_t)W * npp;
    const int      lane     = (int)threadIdx.x % 32;

    for (uint32_t i = threadIdx.x; i < N_chunk; i += blockDim.x) {
        const uint32_t wg = i / 32u;
        T val = in[in_base + i];
#pragma unroll
        for (int b = 0; b < W; b++) {
            uint32_t ballot = __ballot_sync(0xFFFFFFFFu,
                                             static_cast<uint32_t>((val >> b) & T(1)));
            // MSB-first to match 4/8-byte butterfly convention: bit (W-1) at plane 0
            if (lane == 0)
                out[out_base + (uint32_t)(W - 1 - b) * npp + wg] = ballot;
        }
    }
}

template<typename T>
__global__ void bitshuffleDecodeKernelBallot(
    const uint32_t* __restrict__ in,
    T*              __restrict__ out,
    uint32_t N_chunk)
{
    constexpr int  W        = static_cast<int>(sizeof(T) * 8);
    const uint32_t npp      = N_chunk / 32u;
    const uint32_t in_base  = blockIdx.x * (uint32_t)W * npp;
    const uint32_t out_base = blockIdx.x * N_chunk;
    const int      lane     = (int)threadIdx.x % 32;

    for (uint32_t i = threadIdx.x; i < N_chunk; i += blockDim.x) {
        const uint32_t wg = i / 32u;
        T val = T(0);
#pragma unroll
        for (int b = 0; b < W; b++) {
            // MSB-first: bit (W-1) is at plane 0, so plane index = (W-1-b)
            uint32_t word = in[in_base + (uint32_t)(W - 1 - b) * npp + wg];
            T bit = static_cast<T>((word >> lane) & 1u);
            val |= static_cast<T>(bit << b);
        }
        out[out_base + i] = val;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// BitshuffleStage::execute
// ─────────────────────────────────────────────────────────────────────────────

void BitshuffleStage::execute(
    cudaStream_t stream,
    MemoryPool* pool,
    const std::vector<void*>& inputs,
    const std::vector<void*>& outputs,
    const std::vector<size_t>& sizes)
{
    (void)pool;

    if (inputs.empty() || outputs.empty() || sizes.empty())
        throw std::runtime_error("BitshuffleStage: invalid inputs/outputs");

    const size_t in_bytes = sizes[0];
    if (in_bytes == 0) { actual_output_size_ = 0; return; }

    const size_t N_chunk = validateConfig();

    if (in_bytes % block_size_ != 0)
        throw std::runtime_error(
            "BitshuffleStage: input size (" + std::to_string(in_bytes) +
            " bytes) is not a multiple of block_size (" +
            std::to_string(block_size_) + " bytes)");

    const int grid = static_cast<int>(in_bytes / block_size_);

    if (!is_inverse_) {
        switch (element_width_) {
            case 1: {
                const int bdim = static_cast<int>(std::min(N_chunk, size_t(1024)));
                bitshuffleEncodeKernelBallot<uint8_t>
                    <<<grid, bdim, 0, stream>>>(
                    static_cast<const uint8_t*>(inputs[0]),
                    static_cast<uint32_t*>(outputs[0]),
                    static_cast<uint32_t>(N_chunk));
                break;
            }
            case 2: {
                const int bdim = static_cast<int>(std::min(N_chunk, size_t(1024)));
                bitshuffleEncodeKernelBallot<uint16_t>
                    <<<grid, bdim, 0, stream>>>(
                    static_cast<const uint16_t*>(inputs[0]),
                    static_cast<uint32_t*>(outputs[0]),
                    static_cast<uint32_t>(N_chunk));
                break;
            }
            case 4:
                bitshuffleEncodeKernel32
                    <<<grid, 1024, 0, stream>>>(
                    static_cast<const uint32_t*>(inputs[0]),
                    static_cast<uint32_t*>(outputs[0]),
                    static_cast<uint32_t>(N_chunk));
                break;
            case 8:
                bitshuffleEncodeKernel64
                    <<<grid, 1024, 0, stream>>>(
                    static_cast<const uint64_t*>(inputs[0]),
                    static_cast<uint64_t*>(outputs[0]),
                    static_cast<uint32_t>(N_chunk));
                break;
            default:
                throw std::runtime_error("BitshuffleStage: unsupported element_width");
        }
    } else {
        switch (element_width_) {
            case 1: {
                const int bdim = static_cast<int>(std::min(N_chunk, size_t(1024)));
                bitshuffleDecodeKernelBallot<uint8_t>
                    <<<grid, bdim, 0, stream>>>(
                    static_cast<const uint32_t*>(inputs[0]),
                    static_cast<uint8_t*>(outputs[0]),
                    static_cast<uint32_t>(N_chunk));
                break;
            }
            case 2: {
                const int bdim = static_cast<int>(std::min(N_chunk, size_t(1024)));
                bitshuffleDecodeKernelBallot<uint16_t>
                    <<<grid, bdim, 0, stream>>>(
                    static_cast<const uint32_t*>(inputs[0]),
                    static_cast<uint16_t*>(outputs[0]),
                    static_cast<uint32_t>(N_chunk));
                break;
            }
            case 4:
                bitshuffleDecodeKernel32
                    <<<grid, 1024, 0, stream>>>(
                    static_cast<const uint32_t*>(inputs[0]),
                    static_cast<uint32_t*>(outputs[0]),
                    static_cast<uint32_t>(N_chunk));
                break;
            case 8:
                bitshuffleDecodeKernel64
                    <<<grid, 1024, 0, stream>>>(
                    static_cast<const uint64_t*>(inputs[0]),
                    static_cast<uint64_t*>(outputs[0]),
                    static_cast<uint32_t>(N_chunk));
                break;
            default:
                throw std::runtime_error("BitshuffleStage: unsupported element_width");
        }
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(
            std::string("BitshuffleStage kernel launch failed: ") +
            cudaGetErrorString(err));

    actual_output_size_ = in_bytes;
}

// ─────────────────────────────────────────────────────────────────────────────
// Explicit instantiations for ballot kernels
// ─────────────────────────────────────────────────────────────────────────────

template __global__ void bitshuffleEncodeKernelBallot<uint8_t> (const  uint8_t*, uint32_t*, uint32_t);
template __global__ void bitshuffleEncodeKernelBallot<uint16_t>(const uint16_t*, uint32_t*, uint32_t);

template __global__ void bitshuffleDecodeKernelBallot<uint8_t> (const uint32_t*,  uint8_t*, uint32_t);
template __global__ void bitshuffleDecodeKernelBallot<uint16_t>(const uint32_t*, uint16_t*, uint32_t);

} // namespace fz
