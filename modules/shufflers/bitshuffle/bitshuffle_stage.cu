/**
 * modules/shufflers/bitshuffle/bitshuffle_stage.cu
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

#include "shufflers/bitshuffle/bitshuffle_stage.h"
#include "mem/mempool.h"
#include "cuda_check.h"
#include "log.h"
#include "backend/api.h"
#include "backend/warp.h"
#include <stdexcept>
#include <string>
#include <algorithm>
#include <cstdlib>

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

// The 5-stage register butterfly shared by the scattered and smem-staged 32-bit
// kernels. Self-inverse (encode and decode apply the identical transform); the
// only difference between the two directions is the global memory layout.
__device__ __forceinline__ unsigned butterfly32(unsigned a, int sublane)
{
    unsigned q = fz::backend::shflXor(a, 16, 32);
    a = ((sublane & 16) == 0)
        ? __byte_perm(a, q, (3u<<12)|(2u<<8)|(7u<<4)|6u)
        : __byte_perm(a, q, (5u<<12)|(4u<<8)|(1u<<4)|0u);
    q = fz::backend::shflXor(a, 8, 32);
    a = ((sublane & 8) == 0)
        ? __byte_perm(a, q, (3u<<12)|(7u<<8)|(1u<<4)|5u)
        : __byte_perm(a, q, (6u<<12)|(2u<<8)|(4u<<4)|0u);
    q = fz::backend::shflXor(a, 4, 32);
    unsigned mask = 0x0F0F0F0Fu;
    a = ((sublane & 4) == 0) ? ((a & ~mask) | ((q >> 4) & mask))
                            : (((q << 4) & ~mask) | (a & mask));
    q = fz::backend::shflXor(a, 2, 32);
    mask = 0x33333333u;
    a = ((sublane & 2) == 0) ? ((a & ~mask) | ((q >> 2) & mask))
                            : (((q << 2) & ~mask) | (a & mask));
    q = fz::backend::shflXor(a, 1, 32);
    mask = 0x55555555u;
    a = ((sublane & 1) == 0) ? ((a & ~mask) | ((q >> 1) & mask))
                            : (((q << 1) & ~mask) | (a & mask));
    return a;
}

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

        unsigned int q = fz::backend::shflXor(a, 16, 32);
        a = ((sublane & 16) == 0)
            ? __byte_perm(a, q, (3u<<12)|(2u<<8)|(7u<<4)|6u)
            : __byte_perm(a, q, (5u<<12)|(4u<<8)|(1u<<4)|0u);

        q = fz::backend::shflXor(a, 8, 32);
        a = ((sublane & 8) == 0)
            ? __byte_perm(a, q, (3u<<12)|(7u<<8)|(1u<<4)|5u)
            : __byte_perm(a, q, (6u<<12)|(2u<<8)|(4u<<4)|0u);

        q = fz::backend::shflXor(a, 4, 32);
        unsigned int mask = 0x0F0F0F0Fu;
        a = ((sublane & 4) == 0)
            ? ((a & ~mask) | ((q >> 4) & mask))
            : (((q << 4) & ~mask) | (a & mask));

        q = fz::backend::shflXor(a, 2, 32);
        mask = 0x33333333u;
        a = ((sublane & 2) == 0)
            ? ((a & ~mask) | ((q >> 2) & mask))
            : (((q << 2) & ~mask) | (a & mask));

        q = fz::backend::shflXor(a, 1, 32);
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

        unsigned int q = fz::backend::shflXor(a, 16, 32);
        a = ((sublane & 16) == 0)
            ? __byte_perm(a, q, (3u<<12)|(2u<<8)|(7u<<4)|6u)
            : __byte_perm(a, q, (5u<<12)|(4u<<8)|(1u<<4)|0u);

        q = fz::backend::shflXor(a, 8, 32);
        a = ((sublane & 8) == 0)
            ? __byte_perm(a, q, (3u<<12)|(7u<<8)|(1u<<4)|5u)
            : __byte_perm(a, q, (6u<<12)|(2u<<8)|(4u<<4)|0u);

        q = fz::backend::shflXor(a, 4, 32);
        unsigned int mask = 0x0F0F0F0Fu;
        a = ((sublane & 4) == 0)
            ? ((a & ~mask) | ((q >> 4) & mask))
            : (((q << 4) & ~mask) | (a & mask));

        q = fz::backend::shflXor(a, 2, 32);
        mask = 0x33333333u;
        a = ((sublane & 2) == 0)
            ? ((a & ~mask) | ((q >> 2) & mask))
            : (((q << 2) & ~mask) | (a & mask));

        q = fz::backend::shflXor(a, 1, 32);
        mask = 0x55555555u;
        a = ((sublane & 1) == 0)
            ? ((a & ~mask) | ((q >> 1) & mask))
            : (((q << 1) & ~mask) | (a & mask));

        out[out_base + i] = a;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Shared-memory-staged 32-bit kernels. The scattered kernels above write each
// bit-plane word to global at stride npp (512 B for the default chunk) — the 32
// lanes of a warp hit 32 different sectors, so every store is fully uncoalesced
// (32 sectors/request, 12.5% byte efficiency, L2-transaction bound at ~9% DRAM;
// see docs/codebase_notes.md CN-BSHUF-SMEM). These stage the permuted chunk in
// shared memory so the *global* traffic is contiguous both ways.
//
// The smem plane layout is padded to `npp + 1` words per plane: a warp writes
// s[sublane*pstride + col] with col warp-uniform and sublane = lane, i.e. stride
// pstride. Since npp is a multiple of 32 (N_chunk is a multiple of 1024), pstride
// is coprime to 32 → all 32 lanes land in distinct banks, no conflict. Output is
// byte-identical to the scattered kernels.
// ─────────────────────────────────────────────────────────────────────────────

__global__ void bitshuffleEncodeKernel32Smem(
    const uint32_t* __restrict__ in,
    uint32_t*       __restrict__ out,
    uint32_t N_chunk)
{
    extern __shared__ unsigned s[];
    const int      tid      = (int)threadIdx.x;
    const int      sublane  = tid % 32;
    const uint32_t in_base  = blockIdx.x * N_chunk;
    const uint32_t out_base = blockIdx.x * N_chunk;
    const int      npp      = (int)(N_chunk / 32u);   // words per bit-plane
    const int      pstride  = npp + 1;                // padded plane stride

    // Coalesced read → butterfly → conflict-free smem scatter into plane layout.
    for (int i = tid; i < (int)N_chunk; i += (int)blockDim.x)
        s[sublane * pstride + i / 32] = butterfly32(in[in_base + i], sublane);
    __syncthreads();

    // Coalesced flush: global word j is plane (j/npp), column (j%npp).
    for (int j = tid; j < (int)N_chunk; j += (int)blockDim.x) {
        const int p = j / npp;
        out[out_base + j] = s[p * pstride + (j - p * npp)];
    }
}

__global__ void bitshuffleDecodeKernel32Smem(
    const uint32_t* __restrict__ in,
    uint32_t*       __restrict__ out,
    uint32_t N_chunk)
{
    extern __shared__ unsigned s[];
    const int      tid      = (int)threadIdx.x;
    const int      sublane  = tid % 32;
    const uint32_t in_base  = blockIdx.x * N_chunk;
    const uint32_t out_base = blockIdx.x * N_chunk;
    const int      npp      = (int)(N_chunk / 32u);
    const int      pstride  = npp + 1;

    // Coalesced load of the plane-organised input into the padded smem layout.
    for (int j = tid; j < (int)N_chunk; j += (int)blockDim.x) {
        const int p = j / npp;
        s[p * pstride + (j - p * npp)] = in[in_base + j];
    }
    __syncthreads();

    // s[sublane*pstride + i/32] == in[in_base + i/32 + sublane*npp] (the value the
    // scattered decode gathered); butterfly is self-inverse → coalesced store.
    for (int i = tid; i < (int)N_chunk; i += (int)blockDim.x)
        out[out_base + i] = butterfly32(s[sublane * pstride + i / 32], sublane);
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

        q0 = fz::backend::shflXor(a0, 16, 32);
        q1 = fz::backend::shflXor(a1, 16, 32);
        m = 0x0000FFFF0000FFFFull;
        a0 = ((sublane & 16) == 0) ? ((a0 & ~m) | ((q0 >> 16) & m)) : ((a0 & m) | ((q0 << 16) & ~m));
        a1 = ((sublane & 16) == 0) ? ((a1 & ~m) | ((q1 >> 16) & m)) : ((a1 & m) | ((q1 << 16) & ~m));

        q0 = fz::backend::shflXor(a0, 8, 32);
        q1 = fz::backend::shflXor(a1, 8, 32);
        m = 0x00FF00FF00FF00FFull;
        a0 = ((sublane & 8) == 0) ? ((a0 & ~m) | ((q0 >> 8) & m)) : ((a0 & m) | ((q0 << 8) & ~m));
        a1 = ((sublane & 8) == 0) ? ((a1 & ~m) | ((q1 >> 8) & m)) : ((a1 & m) | ((q1 << 8) & ~m));

        q0 = fz::backend::shflXor(a0, 4, 32);
        q1 = fz::backend::shflXor(a1, 4, 32);
        m = 0x0F0F0F0F0F0F0F0Full;
        a0 = ((sublane & 4) == 0) ? ((a0 & ~m) | ((q0 >> 4) & m)) : ((a0 & m) | ((q0 << 4) & ~m));
        a1 = ((sublane & 4) == 0) ? ((a1 & ~m) | ((q1 >> 4) & m)) : ((a1 & m) | ((q1 << 4) & ~m));

        q0 = fz::backend::shflXor(a0, 2, 32);
        q1 = fz::backend::shflXor(a1, 2, 32);
        m = 0x3333333333333333ull;
        a0 = ((sublane & 2) == 0) ? ((a0 & ~m) | ((q0 >> 2) & m)) : ((a0 & m) | ((q0 << 2) & ~m));
        a1 = ((sublane & 2) == 0) ? ((a1 & ~m) | ((q1 >> 2) & m)) : ((a1 & m) | ((q1 << 2) & ~m));

        q0 = fz::backend::shflXor(a0, 1, 32);
        q1 = fz::backend::shflXor(a1, 1, 32);
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

        q0 = fz::backend::shflXor(a0, 16, 32);
        q1 = fz::backend::shflXor(a1, 16, 32);
        m = 0x0000FFFF0000FFFFull;
        a0 = ((sublane & 16) == 0) ? ((a0 & ~m) | ((q0 >> 16) & m)) : ((a0 & m) | ((q0 << 16) & ~m));
        a1 = ((sublane & 16) == 0) ? ((a1 & ~m) | ((q1 >> 16) & m)) : ((a1 & m) | ((q1 << 16) & ~m));

        q0 = fz::backend::shflXor(a0, 8, 32);
        q1 = fz::backend::shflXor(a1, 8, 32);
        m = 0x00FF00FF00FF00FFull;
        a0 = ((sublane & 8) == 0) ? ((a0 & ~m) | ((q0 >> 8) & m)) : ((a0 & m) | ((q0 << 8) & ~m));
        a1 = ((sublane & 8) == 0) ? ((a1 & ~m) | ((q1 >> 8) & m)) : ((a1 & m) | ((q1 << 8) & ~m));

        q0 = fz::backend::shflXor(a0, 4, 32);
        q1 = fz::backend::shflXor(a1, 4, 32);
        m = 0x0F0F0F0F0F0F0F0Full;
        a0 = ((sublane & 4) == 0) ? ((a0 & ~m) | ((q0 >> 4) & m)) : ((a0 & m) | ((q0 << 4) & ~m));
        a1 = ((sublane & 4) == 0) ? ((a1 & ~m) | ((q1 >> 4) & m)) : ((a1 & m) | ((q1 << 4) & ~m));

        q0 = fz::backend::shflXor(a0, 2, 32);
        q1 = fz::backend::shflXor(a1, 2, 32);
        m = 0x3333333333333333ull;
        a0 = ((sublane & 2) == 0) ? ((a0 & ~m) | ((q0 >> 2) & m)) : ((a0 & m) | ((q0 << 2) & ~m));
        a1 = ((sublane & 2) == 0) ? ((a1 & ~m) | ((q1 >> 2) & m)) : ((a1 & m) | ((q1 << 2) & ~m));

        q0 = fz::backend::shflXor(a0, 1, 32);
        q1 = fz::backend::shflXor(a1, 1, 32);
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
            uint32_t ballot = fz::backend::ballotSync32(
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
// Shared-memory staging control for the 32-bit path.
// ─────────────────────────────────────────────────────────────────────────────

// Device opt-in shared-memory ceiling (H100 ~227 KB), queried once.
static int bitshufMaxSmem() {
    static int v = [] {
        int dev = 0; cudaGetDevice(&dev);
        int s = 48 * 1024;
        cudaDeviceGetAttribute(&s, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
        return s;
    }();
    return v;
}

// Staged 32-bit path on by default; FZ_BITSHUF_SMEM=0 forces the scattered kernel
// (kept for A/B measurement — see CN-BSHUF-SMEM).
static bool bitshufUseSmem() {
    static bool v = [] {
        const char* e = std::getenv("FZ_BITSHUF_SMEM");
        return !(e && e[0] == '0');
    }();
    return v;
}

// Padded smem footprint for a 32-plane chunk of npp = N_chunk/32 words per plane.
static size_t bitshuf32SmemBytes(size_t N_chunk) {
    const size_t npp = N_chunk / 32u;
    return 32u * (npp + 1u) * sizeof(uint32_t);
}

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

    const size_t full_bytes = (in_bytes / block_size_) * block_size_;
    const size_t tail_bytes = in_bytes - full_bytes;
    const int grid = static_cast<int>(full_bytes / block_size_);

    if (grid > 0) {
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
                case 4: {
                    const size_t smem = bitshuf32SmemBytes(N_chunk);
                    if (bitshufUseSmem() && smem <= (size_t)bitshufMaxSmem()) {
                        if (smem > 48u * 1024u)
                            cudaFuncSetAttribute(bitshuffleEncodeKernel32Smem,
                                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
                        bitshuffleEncodeKernel32Smem
                            <<<grid, 1024, smem, stream>>>(
                            static_cast<const uint32_t*>(inputs[0]),
                            static_cast<uint32_t*>(outputs[0]),
                            static_cast<uint32_t>(N_chunk));
                    } else {
                        bitshuffleEncodeKernel32
                            <<<grid, 1024, 0, stream>>>(
                            static_cast<const uint32_t*>(inputs[0]),
                            static_cast<uint32_t*>(outputs[0]),
                            static_cast<uint32_t>(N_chunk));
                    }
                    break;
                }
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
                case 4: {
                    const size_t smem = bitshuf32SmemBytes(N_chunk);
                    if (bitshufUseSmem() && smem <= (size_t)bitshufMaxSmem()) {
                        if (smem > 48u * 1024u)
                            cudaFuncSetAttribute(bitshuffleDecodeKernel32Smem,
                                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
                        bitshuffleDecodeKernel32Smem
                            <<<grid, 1024, smem, stream>>>(
                            static_cast<const uint32_t*>(inputs[0]),
                            static_cast<uint32_t*>(outputs[0]),
                            static_cast<uint32_t>(N_chunk));
                    } else {
                        bitshuffleDecodeKernel32
                            <<<grid, 1024, 0, stream>>>(
                            static_cast<const uint32_t*>(inputs[0]),
                            static_cast<uint32_t*>(outputs[0]),
                            static_cast<uint32_t>(N_chunk));
                    }
                    break;
                }
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
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(
            std::string("BitshuffleStage kernel launch failed: ") +
            cudaGetErrorString(err));

    if (tail_bytes > 0) {
        const auto* in_tail = static_cast<const uint8_t*>(inputs[0]) + full_bytes;
        auto* out_tail = static_cast<uint8_t*>(outputs[0]) + full_bytes;
        FZ_CUDA_CHECK(cudaMemcpyAsync(
            out_tail,
            in_tail,
            tail_bytes,
            cudaMemcpyDeviceToDevice,
            stream));
    }

    actual_output_size_ = in_bytes;
    FZ_LOG(TRACE, "Bitshuffle %s: %.1f KB, block=%zu ew=%d",
           is_inverse_ ? "decode" : "encode",
           in_bytes / 1024.0, block_size_, static_cast<int>(element_width_));
}

// ─────────────────────────────────────────────────────────────────────────────
// Explicit instantiations for ballot kernels
// ─────────────────────────────────────────────────────────────────────────────

template __global__ void bitshuffleEncodeKernelBallot<uint8_t> (const  uint8_t*, uint32_t*, uint32_t);
template __global__ void bitshuffleEncodeKernelBallot<uint16_t>(const uint16_t*, uint32_t*, uint32_t);

template __global__ void bitshuffleDecodeKernelBallot<uint8_t> (const uint32_t*,  uint8_t*, uint32_t);
template __global__ void bitshuffleDecodeKernelBallot<uint16_t>(const uint32_t*, uint16_t*, uint32_t);

} // namespace fz
