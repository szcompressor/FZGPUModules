#pragma once

/**
 * @file modules/fused/chunk_fusion/chunk_fusion.cuh
 * @brief Chunk-cooperative fusion HARNESS — the generalizable "connecting code"
 *        that composes per-stage __device__ ops into one CTA-per-chunk kernel.
 *
 * This is the shared skeleton for the chunk-cooperative fusion strategy (one CTA
 * owns a chunk, intermediates in shared memory, `__syncthreads` between
 * stages) — the LC-style counterpart to the warp-register driver in fused_block/.
 * Chunk size is a template parameter (`ChunkBytes`, see `Geom<>` in
 * chunk_geometry.h): every op whose geometry actually depends on it
 * (`Bitshuffle32`, the coder ops) is itself templated on `ChunkBytes`; ops that
 * only ever touch a runtime element count are not.
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
 *     with no new glue: `chunk_fused_kernel<16384, Quant, RRECoder<16384>, Diff,
 *     Bitshuffle32<16384>>`.
 *
 * The compile-time template kernel above is a fallback (`FZ_FUSION_NVRTC=0`) kept
 * for the profiling harness; the production path (the generic registry runner,
 * `runChunkCooperative` in fusion_registry.cpp) always composes THIS harness as
 * source for an arbitrary op list at runtime via NVRTC — see
 * nvrtc_chunk_fusion.h/.cpp and docs/codebase_notes.md CN-NVRTC-FUSE. The ops stay
 * hand-written; only the composing glue (including which `ChunkBytes` to bake in)
 * is generated.
 */

#include "fused/chunk_fusion/chunk_geometry.h"   // chunk geometry constants (no host deps)
#include "fused/chunk_fusion/chunk_op_params.h"  // shared POD op Params (host/device agree)
#include "coders/lc_common/lc_chunk_components.cuh"
#include "coders/lc_common/lc_clog_components.cuh"   // d_CLOG / d_HCLOG
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
    using Params = QuantInplaceZigzagParams;   // shared POD (chunk_op_params.h)
    // Every Map op takes a `const void* pp` pointing at its slice of the packed
    // params blob (parametric ops cast it, stateless ignore it — the blob is
    // exactly-sized so a tail op may get a one-past-end pointer, never
    // dereferenced) and a `ChunkSideCtx` for escaping outputs (this variant emits
    // none — outliers go inline — so it ignores side).
    __device__ static void load(const float* __restrict__ in, size_t base, int cnt,
                                uint32_t* __restrict__ s, const void* pp,
                                const ChunkSideCtx& /*side*/) {
        const Params p = *static_cast<const Params*>(pp);
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

// ── Map op: NOA/ABS quant with SPLIT outliers (3-port). Codes stream stays clean —
// a `0` sentinel at each outlier position, TCMS(zigzag) elsewhere — and outliers are
// appended to a side list of (global index, value) pairs via a GLOBAL atomic counter
// shared across all chunk CTAs. Reproduces quantizer_abs_fwd_kernel<...,Zigzag=true>
// byte-for-byte in the codes stream, so the staged 3-port inverse decodes it unchanged.
// The clean codes stream compresses far better than inline raw float bits on
// outlier-heavy fields, which is the whole point of the split variant.
struct QuantSplitOutlier {
    using Params = QuantSplitOutlierParams;   // == QuantInplaceZigzagParams layout
    __device__ static void load(const float* __restrict__ in, size_t base, int cnt,
                                uint32_t* __restrict__ s, const void* pp,
                                const ChunkSideCtx& side) {
        const Params p = *static_cast<const Params*>(pp);
        const int lane = threadIdx.x & 31;
        for (int i = threadIdx.x; i < cnt; i += TPB) {
            const float x = in[base + i];
            const int   q = __float2int_rn(x * p.ebx2_r);
            if (q > -(int)p.radius && q < (int)p.radius && fabsf(x) < p.threshold) {
                s[i] = (uint32_t)((q << 1) ^ (q >> 31));   // zigzag(q)
                continue;
            }
            s[i] = 0u;                                      // outlier sentinel
            // Warp-aggregated append: the outlier lanes of this warp claim a contiguous
            // block of slots with ONE global atomicAdd by the leader (lowest outlier
            // lane), then each takes its rank within the block — instead of one global
            // atomicAdd per outlier. Cuts global-counter contention up to 32x. Only
            // outlier lanes reach here, so __activemask() is exactly the outlier set
            // (tail-safe: partial final chunks just have fewer active lanes).
            const unsigned outmask   = __activemask();
            const int      leader    = __ffs(outmask) - 1;
            const unsigned rank      = __popc(outmask & ((1u << lane) - 1));
            uint32_t       warp_base = 0;
            if (lane == leader)
                warp_base = atomicAdd(side.out_count, __popc(outmask));
            warp_base = __shfl_sync(outmask, warp_base, leader);   // broadcast base slot
            const uint32_t slot = warp_base + rank;
            if (slot < side.max) {
                side.out_idxs[slot] = (uint32_t)(base + i);   // global element index
                side.out_vals[slot] = x;
            }
        }
    }
};

// ── Stencil op: chunk-local difference (boundary = elem 0) + negabinary. ─────
struct DiffNegabinary {
    using Params = EmptyParams;   // stateless → 0 params bytes
    __device__ static void apply(const uint32_t* __restrict__ s_in, uint32_t* __restrict__ s_out,
                                 int cnt, bool /*full*/, const void* /*pp*/) {
        for (int i = threadIdx.x; i < cnt; i += TPB) {
            const int ci = (int)s_in[i];
            const int d  = (i == 0) ? ci : (ci - (int)s_in[i-1]);
            s_out[i] = Negabinary<int32_t>::encode(d);
        }
    }
};

// ── Fixed-length cooperative op: 32-bit bitshuffle. The partial tail chunk is
// copied through (the staged bitshuffle memcpys its sub-chunk tail). Templated
// on chunk size — its plane-stride math (NELEM/NPP) depends on it, unlike the
// Map/stencil ops above which only ever touch a runtime element count. ───────
template <int ChunkBytes>
struct Bitshuffle32 {
    using Params = EmptyParams;
    static constexpr int NELEM = Geom<ChunkBytes>::NELEM;
    static constexpr int NPP   = Geom<ChunkBytes>::NPP;
    __device__ static void apply(const uint32_t* __restrict__ s_in, uint32_t* __restrict__ s_out,
                                 int cnt, bool full, const void* /*pp*/) {
        if (full) {
            const int lane = threadIdx.x & 31;
            for (int i = threadIdx.x; i < NELEM; i += TPB)
                s_out[i/32 + lane*NPP] = butterfly32(s_in[i], lane);
        } else {
            for (int i = threadIdx.x; i < cnt; i += TPB) s_out[i] = s_in[i];
        }
    }
};

// ── Coder ops: the swappable variable-length sink. Uniform LC signature.
// Templated on chunk size — each is a thin wrapper handing it through to the
// already chunk-size-generic `d_RZE<T,ChunkBytes>`-style LC primitives
// (lc_chunk_components.cuh), which is where the underlying capability already
// lives; only this glue needed to stop hardcoding CHUNK_BYTES. ──────────────
template <int ChunkBytes>
struct RZECoder {
    using Params = EmptyParams;
    __device__ static bool encode(int& csize, byte* in, byte* out, byte* temp, const void* /*pp*/) {
        return lc_detail::d_RZE<byte, ChunkBytes>(csize, in, out, temp);
    }
    __device__ static void decode(int& csize, byte* in, byte* out, byte* temp) {
        lc_detail::d_iRZE<byte, ChunkBytes>(csize, in, out, temp);
    }
};
template <int ChunkBytes>
struct RRECoder {
    using Params = EmptyParams;
    __device__ static bool encode(int& csize, byte* in, byte* out, byte* temp, const void* /*pp*/) {
        return lc_detail::d_RRE<byte, ChunkBytes>(csize, in, out, temp);
    }
};
// RARE/RAZE (auto-k generalizations of RRE/RZE) — same uniform LC signature, so they
// drop into the harness as coder ops with no new glue. Their stages just declare
// getFusedOp() and any Map->Transform*->{RARE|RAZE} chain fuses via the generic runner.
template <int ChunkBytes>
struct RARECoder {
    using Params = EmptyParams;
    __device__ static bool encode(int& csize, byte* in, byte* out, byte* temp, const void* /*pp*/) {
        return lc_detail::d_RARE<byte, ChunkBytes>(csize, in, out, temp);
    }
};
template <int ChunkBytes>
struct RAZECoder {
    using Params = EmptyParams;
    __device__ static bool encode(int& csize, byte* in, byte* out, byte* temp, const void* /*pp*/) {
        return lc_detail::d_RAZE<byte, ChunkBytes>(csize, in, out, temp);
    }
};

// CLOG / HCLOG — LC leading-zero + bit-packing coders (byte-word, matching each stage's
// word_size==1 dispatch → d_CLOG<uint8_t>). Same uniform LC signature, so they drop in with
// no new glue: their stages declare getFusedOp() and any Map->Transform*->{CLOG|HCLOG} fuses.
template <int ChunkBytes>
struct CLOGCoder {
    using Params = EmptyParams;
    __device__ static bool encode(int& csize, byte* in, byte* out, byte* temp, const void* /*pp*/) {
        return lc_detail::d_CLOG<uint8_t, ChunkBytes>(csize, in, out, temp);
    }
};
template <int ChunkBytes>
struct HCLOGCoder {
    using Params = EmptyParams;
    __device__ static bool encode(int& csize, byte* in, byte* out, byte* temp, const void* /*pp*/) {
        return lc_detail::d_HCLOG<uint8_t, ChunkBytes>(csize, in, out, temp);
    }
};

// ── Packed-params-blob offset arithmetic. Each op contributes sizeof(Params),
// or 0 if stateless (EmptyParams). The blob is ops in execution order:
// [QuantOp][Transforms...][Coder]. Offsets are resolved at compile time. ─────
template<class Op> struct OpParamBytes {
    static constexpr int value =
        FzSame<typename Op::Params, EmptyParams>::value ? 0 : (int)sizeof(typename Op::Params);
};
template<class...> struct SumParamBytes { static constexpr int value = 0; };
template<class T, class... R> struct SumParamBytes<T, R...> {
    static constexpr int value = OpParamBytes<T>::value + SumParamBytes<R...>::value;
};

// ── Transform chain: apply ops in order, ping-ponging a<->b, threading each op
// its params slice (running byte offset into the blob). Returns the final
// buffer. Ops must be size-preserving (diff, bitshuffle are). ────────────────
template<class... Ts> struct Chain;
template<> struct Chain<> {
    __device__ static uint32_t* apply(uint32_t* a, uint32_t*, int, bool, const byte*, int) { return a; }
};
template<class T, class... R> struct Chain<T, R...> {
    __device__ static uint32_t* apply(uint32_t* a, uint32_t* b, int cnt, bool full,
                                      const byte* params, int off) {
        T::apply(a, b, cnt, full, params + off);
        __syncthreads();
        return Chain<R...>::apply(b, a, cnt, full, params, off + OpParamBytes<T>::value);
    }
};

// ── Harness body: one CTA per chunk. Quant (Map) -> Transforms... -> Coder (sink).
// Stage-agnostic: it composes whatever ops it is given. Factored out of the
// __global__ entry so the NVRTC codegen path can wrap it in an `extern "C"`
// kernel (a __global__ cannot call another __global__). Both the compile-time
// template kernel below and the runtime-generated kernel call this same body —
// the ops stay single-sourced; only the composing glue differs. ──────────────
// `params` is the packed per-op Params blob, ops in execution order
// ([QuantOp][Transforms...][Coder]); each op is handed its slice at a
// compile-time offset. Stateless ops ignore it.
// `side` carries escaping outputs (e.g. an outlier list); the Map op uses it or
// ignores it. Defaulted so callers that never fuse a side-output op need not pass it.
template<int ChunkBytes, class QuantOp, class Coder, class... Transforms>
__device__ __forceinline__ void
chunk_fused_body(const float* __restrict__ in, size_t n,
                 const byte* __restrict__ params,
                 byte* __restrict__ scratch, uint32_t* __restrict__ sizes,
                 ChunkSideCtx side = ChunkSideCtx{}) {
    constexpr int NELEM = Geom<ChunkBytes>::NELEM;
    __shared__ __align__(16) uint32_t sA[NELEM];
    __shared__ __align__(16) uint32_t sB[NELEM];
    __shared__ __align__(16) byte     sTemp[TEMP_BYTES];

    const uint32_t cid  = blockIdx.x;
    const size_t   base = (size_t)cid * NELEM;
    const int      cnt  = (int)min((size_t)NELEM, n - base);
    const bool     full = (cnt == NELEM);

    QuantOp::load(in, base, cnt, sA, params /* + 0: quant is first in the blob */, side);
    __syncthreads();

    constexpr int kTransOff = OpParamBytes<QuantOp>::value;
    uint32_t* cur = Chain<Transforms...>::apply(sA, sB, cnt, full, params, kTransOff);
    uint32_t* alt = (cur == sA) ? sB : sA;

    const int in_size = full ? ChunkBytes : cnt * 4;
    if (!full) {   // zero-pad the sub-chunk so the coder's word reads see zeros
        for (int i = threadIdx.x + in_size; i < ChunkBytes; i += TPB)
            reinterpret_cast<byte*>(cur)[i] = 0;
        __syncthreads();
    }

    constexpr int kCoderOff = OpParamBytes<QuantOp>::value + SumParamBytes<Transforms...>::value;
    int  csize = in_size;
    bool good  = Coder::encode(csize, reinterpret_cast<byte*>(cur),
                               reinterpret_cast<byte*>(alt), sTemp, params + kCoderOff);
    __syncthreads();

    byte* out = scratch + (size_t)cid * ChunkBytes;
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
template<int ChunkBytes, class QuantOp, class Coder, class... Transforms>
__global__ void __launch_bounds__(TPB)
chunk_fused_kernel(const float* __restrict__ in, size_t n,
                   const byte* __restrict__ params,
                   byte* __restrict__ scratch, uint32_t* __restrict__ sizes,
                   ChunkSideCtx side = ChunkSideCtx{}) {
    chunk_fused_body<ChunkBytes, QuantOp, Coder, Transforms...>(in, n, params, scratch, sizes, side);
}

// ── Inverse chunk harness (initial RZE evidence-gated path). ─────────────────
// One CTA consumes one packed LC chunk, keeps every intermediate in the same
// shared-memory ping-pong buffers as compression, and writes reconstructed
// floats once. `offsets` are payload-relative exclusive offsets computed from
// the archive's flagged size table by the launcher.
template<int ChunkBytes, class Coder>
__device__ __forceinline__ void
chunk_inverse_pfpl_body(const byte* __restrict__ archive,
                        const uint32_t* __restrict__ entries,
                        const uint32_t* __restrict__ offsets,
                        size_t output_bytes, float ebx2,
                        bool inplace_outliers, uint32_t quant_radius,
                        float* __restrict__ out) {
    constexpr int NELEM = Geom<ChunkBytes>::NELEM;
    constexpr int NPP   = Geom<ChunkBytes>::NPP;
    __shared__ __align__(16) uint32_t sA[NELEM];
    __shared__ __align__(16) uint32_t sB[NELEM];
    __shared__ __align__(16) byte     sTemp[TEMP_BYTES];

    const uint32_t cid = blockIdx.x;
    const size_t base_bytes = static_cast<size_t>(cid) * ChunkBytes;
    if (base_bytes >= output_bytes) return;
    const int out_bytes = static_cast<int>(min(static_cast<size_t>(ChunkBytes),
                                               output_bytes - base_bytes));
    const int cnt = out_bytes / static_cast<int>(sizeof(uint32_t));
    const bool full = out_bytes == ChunkBytes;
    const uint32_t entry = entries[cid];
    const uint32_t stored = entry & 0x7fffffffu;
    const bool raw = (entry & 0x80000000u) != 0;
    const uint32_t nchunks = static_cast<uint32_t>(
        (output_bytes + ChunkBytes - 1) / ChunkBytes);
    const size_t header = 8u + static_cast<size_t>(nchunks) * sizeof(uint32_t);
    const byte* payload = archive + header + offsets[cid];

    for (uint32_t i = threadIdx.x; i < stored; i += blockDim.x)
        reinterpret_cast<byte*>(sA)[i] = payload[i];
    __syncthreads();

    uint32_t* cur = sA;
    uint32_t* alt = sB;
    if (!raw) {
        int csize = static_cast<int>(stored);
        Coder::decode(csize, reinterpret_cast<byte*>(sA),
                      reinterpret_cast<byte*>(sB), sTemp);
        __syncthreads();
        cur = sB;
        alt = sA;
    }

    // Bitshuffle is self-inverse but its read/write indexing is reversed.
    if (full) {
        const int lane = threadIdx.x & 31;
        for (int i = threadIdx.x; i < NELEM; i += blockDim.x) {
            const unsigned a = cur[i / 32 + lane * NPP];
            alt[i] = butterfly32(a, lane);
        }
        __syncthreads();
        uint32_t* tmp = cur; cur = alt; alt = tmp;
    }

    // Inverse negabinary + chunk-local inclusive scan. Each 512-value tile is
    // scanned warp-wise, with 16 warp totals and one carry in the coder scratch
    // (the LC decoder is finished with it by this point).
    int32_t* qout = reinterpret_cast<int32_t*>(alt);
    int32_t* scan = reinterpret_cast<int32_t*>(sTemp);
    if (threadIdx.x == 0) scan[16] = 0;
    __syncthreads();
    for (int tile = 0; tile < cnt; tile += TPB) {
        const int idx = tile + threadIdx.x;
        const bool valid = idx < cnt;
        int32_t v = valid ? Negabinary<int32_t>::decode(cur[idx]) : 0;
        const int lane = threadIdx.x & 31;
        const int warp = threadIdx.x >> 5;
        for (int delta = 1; delta < 32; delta <<= 1) {
            const int32_t up = __shfl_up_sync(0xffffffffu, v, delta);
            if (lane >= delta) v += up;
        }
        if (lane == 31) scan[warp] = v;
        __syncthreads();
        if (warp == 0) {
            int32_t w = lane < 16 ? scan[lane] : 0;
            for (int delta = 1; delta < 32; delta <<= 1) {
                const int32_t up = __shfl_up_sync(0xffffffffu, w, delta);
                if (lane >= delta) w += up;
            }
            if (lane < 16) scan[lane] = w;
        }
        __syncthreads();
        const int32_t q = v + (warp ? scan[warp - 1] : 0) + scan[16];
        if (valid) qout[idx] = q;
        __syncthreads();
        if (threadIdx.x == 0) scan[16] += scan[15];
        __syncthreads();
    }

    const size_t base_elem = base_bytes / sizeof(uint32_t);
    for (int i = threadIdx.x; i < cnt; i += blockDim.x) {
        // Difference reconstructs the quantizer's *zigzag code* stream. Undo
        // that final map before scaling back to the reconstructed float.
        const uint32_t code = static_cast<uint32_t>(qout[i]);
        if (inplace_outliers && (code >> 1) >= quant_radius) {
            out[base_elem + static_cast<size_t>(i)] = __uint_as_float(code);
        } else {
            const int32_t q = static_cast<int32_t>(
                (code >> 1) ^ (0u - (code & 1u)));
            out[base_elem + static_cast<size_t>(i)] = static_cast<float>(q) * ebx2;
        }
    }
}

template<int ChunkBytes, class Coder>
__global__ void __launch_bounds__(TPB)
chunk_inverse_pfpl_kernel(const byte* __restrict__ archive,
                          const uint32_t* __restrict__ entries,
                          const uint32_t* __restrict__ offsets,
                          size_t output_bytes, float ebx2,
                          bool inplace_outliers, uint32_t quant_radius,
                          float* __restrict__ out) {
    chunk_inverse_pfpl_body<ChunkBytes, Coder>(archive, entries, offsets, output_bytes, ebx2,
                                               inplace_outliers, quant_radius, out);
}

} // namespace chunk
} // namespace fused
} // namespace fz
