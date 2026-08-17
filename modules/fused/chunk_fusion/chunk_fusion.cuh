#pragma once

/**
 * @file modules/fused/chunk_fusion/chunk_fusion.cuh
 * @brief Chunk-cooperative fusion HARNESS — the generalizable "connecting code"
 *        that composes per-stage __device__ ops into one CTA-per-chunk kernel.
 *
 * This is the shared skeleton for the chunk-cooperative fusion strategy (one CTA
 * owns a 16 KB chunk, intermediates in shared memory, `__syncthreads` between
 * stages) — the LC-style counterpart to the warp-register driver in fused_block/.
 *
 * The DESIGN (see docs/codebase_notes.md CN-CHUNK-FUSE):
 *   - Each fusable stage contributes a small __device__ OP (below): a Map op that
 *     loads+transforms global input, a stencil/fixed transform op that maps one
 *     smem buffer to another, or a variable-length CODER op (the sink) with the
 *     uniform LC signature. The ops are the hand-written per-stage logic.
 *   - The HARNESS (`chunk_fused_kernel`) is the stage-agnostic glue: it loads the
 *     chunk, ping-pongs the transform ops through two smem buffers with the right
 *     syncs, runs the coder, and emits the per-chunk compressed bytes + size. It
 *     does not know or care which ops it is composing.
 *   - Swapping the coder (RZE -> RRE) or the transform set re-composes the kernel
 *     with no new glue: `chunk_fused_kernel<Quant, RRECoder, Diff, Bitshuffle>`.
 *
 * A future NVRTC path emits THIS harness as source for an arbitrary op list,
 * #including the op headers — the ops stay hand-written, only the glue is
 * generated. Template composition here is the compile-time precursor.
 */

#include "fused/chunk_fusion/chunk_geometry.h"   // chunk geometry constants (no host deps)
#include "coders/lc_common/lc_chunk_components.cuh"
#include "transforms/negabinary/negabinary.h"
#include <cstdint>

namespace fz {
namespace fused {
namespace chunk {

using byte = uint8_t;

// 5-stage register butterfly (copy of bitshuffle_stage.cu butterfly32).
__device__ __forceinline__ unsigned butterfly32(unsigned a, int sublane) {
    unsigned q = __shfl_xor_sync(0xffffffffu, a, 16, 32);
    a = ((sublane&16)==0) ? __byte_perm(a,q,(3u<<12)|(2u<<8)|(7u<<4)|6u)
                          : __byte_perm(a,q,(5u<<12)|(4u<<8)|(1u<<4)|0u);
    q = __shfl_xor_sync(0xffffffffu, a, 8, 32);
    a = ((sublane&8)==0) ? __byte_perm(a,q,(3u<<12)|(7u<<8)|(1u<<4)|5u)
                         : __byte_perm(a,q,(6u<<12)|(2u<<8)|(4u<<4)|0u);
    q = __shfl_xor_sync(0xffffffffu, a, 4, 32); unsigned m=0x0F0F0F0Fu;
    a = ((sublane&4)==0) ? ((a&~m)|((q>>4)&m)) : (((q<<4)&~m)|(a&m));
    q = __shfl_xor_sync(0xffffffffu, a, 2, 32); m=0x33333333u;
    a = ((sublane&2)==0) ? ((a&~m)|((q>>2)&m)) : (((q<<2)&~m)|(a&m));
    q = __shfl_xor_sync(0xffffffffu, a, 1, 32); m=0x55555555u;
    a = ((sublane&1)==0) ? ((a&~m)|((q>>1)&m)) : (((q<<1)&~m)|(a&m));
    return a;
}

// ── Map op: linear/NOA quant with inplace outliers + TCMS(zigzag) codes. ─────
// Loads global floats and writes codes to smem. Out-of-radius / over-threshold
// values are stored as raw IEEE-754 bits (matches quantizer_abs_fwd_inplace_kernel).
struct QuantInplaceZigzag {
    struct Params { float ebx2_r; uint32_t radius; float threshold; };
    __device__ static void load(const float* __restrict__ in, size_t base, int cnt,
                                uint32_t* __restrict__ s, Params p) {
        for (int i = threadIdx.x; i < cnt; i += TPB) {
            const float x = in[base + i];
            const int   q = __float2int_rn(x * p.ebx2_r);
            uint32_t c;
            if (q > -(int)p.radius && q < (int)p.radius && fabsf(x) < p.threshold)
                c = (uint32_t)((q << 1) ^ (q >> 31));
            else
                c = __float_as_uint(x);   // raw IEEE-754 bits (NVRTC-portable bit-cast)
            s[i] = c;
        }
    }
};

// ── Stencil op: chunk-local difference (boundary = elem 0) + negabinary. ─────
struct DiffNegabinary {
    __device__ static void apply(const uint32_t* __restrict__ s_in, uint32_t* __restrict__ s_out,
                                 int cnt, bool /*full*/) {
        for (int i = threadIdx.x; i < cnt; i += TPB) {
            const int ci = (int)s_in[i];
            const int d  = (i == 0) ? ci : (ci - (int)s_in[i-1]);
            s_out[i] = Negabinary<int32_t>::encode(d);
        }
    }
};

// ── Fixed-length cooperative op: 32-bit bitshuffle. The partial tail chunk is
// copied through (the staged bitshuffle memcpys its sub-chunk tail). ─────────
struct Bitshuffle32 {
    __device__ static void apply(const uint32_t* __restrict__ s_in, uint32_t* __restrict__ s_out,
                                 int cnt, bool full) {
        if (full) {
            const int lane = threadIdx.x & 31;
            for (int i = threadIdx.x; i < NELEM; i += TPB)
                s_out[i/32 + lane*NPP] = butterfly32(s_in[i], lane);
        } else {
            for (int i = threadIdx.x; i < cnt; i += TPB) s_out[i] = s_in[i];
        }
    }
};

// ── Coder ops: the swappable variable-length sink. Uniform LC signature. ─────
struct RZECoder {
    __device__ static bool encode(int& csize, byte* in, byte* out, byte* temp) {
        return lc_detail::d_RZE<byte, CHUNK_BYTES>(csize, in, out, temp);
    }
};
struct RRECoder {
    __device__ static bool encode(int& csize, byte* in, byte* out, byte* temp) {
        return lc_detail::d_RRE<byte, CHUNK_BYTES>(csize, in, out, temp);
    }
};

// ── Transform chain: apply ops in order, ping-ponging a<->b, return the final
// buffer. Ops must be size-preserving (diff, bitshuffle are). ────────────────
template<class... Ts> struct Chain;
template<> struct Chain<> {
    __device__ static uint32_t* apply(uint32_t* a, uint32_t*, int, bool) { return a; }
};
template<class T, class... R> struct Chain<T, R...> {
    __device__ static uint32_t* apply(uint32_t* a, uint32_t* b, int cnt, bool full) {
        T::apply(a, b, cnt, full);
        __syncthreads();
        return Chain<R...>::apply(b, a, cnt, full);
    }
};

// ── Harness body: one CTA per chunk. Quant (Map) -> Transforms... -> Coder (sink).
// Stage-agnostic: it composes whatever ops it is given. Factored out of the
// __global__ entry so the NVRTC codegen path can wrap it in an `extern "C"`
// kernel (a __global__ cannot call another __global__). Both the compile-time
// template kernel below and the runtime-generated kernel call this same body —
// the ops stay single-sourced; only the composing glue differs. ──────────────
template<class QuantOp, class Coder, class... Transforms>
__device__ __forceinline__ void
chunk_fused_body(const float* __restrict__ in, size_t n,
                 typename QuantOp::Params qp,
                 byte* __restrict__ scratch, uint32_t* __restrict__ sizes) {
    __shared__ __align__(16) uint32_t sA[NELEM];
    __shared__ __align__(16) uint32_t sB[NELEM];
    __shared__ __align__(16) byte     sTemp[TEMP_BYTES];

    const uint32_t cid  = blockIdx.x;
    const size_t   base = (size_t)cid * NELEM;
    const int      cnt  = (int)min((size_t)NELEM, n - base);
    const bool     full = (cnt == NELEM);

    QuantOp::load(in, base, cnt, sA, qp);
    __syncthreads();

    uint32_t* cur = Chain<Transforms...>::apply(sA, sB, cnt, full);
    uint32_t* alt = (cur == sA) ? sB : sA;

    const int in_size = full ? CHUNK_BYTES : cnt * 4;
    if (!full) {   // zero-pad the sub-chunk so the coder's word reads see zeros
        for (int i = threadIdx.x + in_size; i < CHUNK_BYTES; i += TPB)
            reinterpret_cast<byte*>(cur)[i] = 0;
        __syncthreads();
    }

    int  csize = in_size;
    bool good  = Coder::encode(csize, reinterpret_cast<byte*>(cur),
                               reinterpret_cast<byte*>(alt), sTemp);
    __syncthreads();

    byte* out = scratch + (size_t)cid * CHUNK_BYTES;
    if (good && csize < in_size) {
        for (int i = threadIdx.x; i < csize; i += TPB) out[i] = reinterpret_cast<byte*>(alt)[i];
        if (threadIdx.x == 0) sizes[cid] = (uint32_t)csize;
    } else {
        for (int i = threadIdx.x; i < in_size; i += TPB) out[i] = reinterpret_cast<byte*>(cur)[i];
        if (threadIdx.x == 0) sizes[cid] = (1u << 31) | (uint32_t)in_size;
    }
}

// Compile-time template entry (the FZ_FUSION path without NVRTC). The NVRTC path
// generates an equivalent `extern "C"` kernel over the same body.
template<class QuantOp, class Coder, class... Transforms>
__global__ void __launch_bounds__(TPB)
chunk_fused_kernel(const float* __restrict__ in, size_t n,
                   typename QuantOp::Params qp,
                   byte* __restrict__ scratch, uint32_t* __restrict__ sizes) {
    chunk_fused_body<QuantOp, Coder, Transforms...>(in, n, qp, scratch, sizes);
}

} // namespace chunk
} // namespace fused
} // namespace fz
