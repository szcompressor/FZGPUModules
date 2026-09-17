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
 *   - Each fusable stage contributes a small __device__ OP (below): an Elementwise op that
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
#include "transforms/zigzag/zigzag.h"
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

// ── Elementwise op: linear/NOA quant with inplace outliers + TCMS(zigzag) codes. ─
// Loads global floats and writes codes to smem. Out-of-radius / over-threshold
// values are stored as raw IEEE-754 bits (matches quantizer_abs_fwd_inplace_kernel).
struct QuantInplaceZigzag {
    using Params = QuantInplaceZigzagParams;   // shared POD (chunk_op_params.h)
    // Every Elementwise op takes a `const void* pp` pointing at its slice of the packed
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

// ── Elementwise op: NOA/ABS quant with SPLIT outliers (3-port). Codes stream stays clean —
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

// ── Stencil op: chunk-local difference (boundary = elem 0), PLAIN -- no final
// encode. Same shape as DiffNegabinary but leaves the signed delta as-is
// (reinterpreted through uint32_t bits, not encoded): GolombRiceCoder does its
// OWN zigzag internally (matching the standalone GolombRiceStage kernel, which
// has no upstream transform to rely on and zigzags inline), so a transform that
// zigzagged here too would double-encode. This is the fused equivalent of a
// plain same-type DifferenceStage<T> (T==TOut, no Mode fusion). ─────────────
struct DiffPlain {
    using Params = EmptyParams;
    __device__ static void apply(const uint32_t* __restrict__ s_in, uint32_t* __restrict__ s_out,
                                 int cnt, bool /*full*/, const void* /*pp*/) {
        for (int i = threadIdx.x; i < cnt; i += TPB) {
            const int ci = (int)s_in[i];
            const int d  = (i == 0) ? ci : (ci - (int)s_in[i-1]);
            s_out[i] = static_cast<uint32_t>(d);
        }
    }
};

// ── Fixed-length cooperative op: 32-bit bitshuffle. The partial tail chunk is
// copied through (the staged bitshuffle memcpys its sub-chunk tail). Templated
// on chunk size — its plane-stride math (NELEM/NPP) depends on it, unlike the
// Elementwise/stencil ops above which only ever touch a runtime element count. ──
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
// getFusedOp() and any Elementwise->Transform*->{RARE|RAZE} chain fuses via the generic runner.
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
// no new glue: their stages declare getFusedOp() and any Elementwise->Transform*->{CLOG|HCLOG} fuses.
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

// ── GolombRiceCoder: chunk-cooperative fused encode of modules/coders/golomb_rice.
// ENCODE ONLY (matches RRE/RARE/RAZE/CLOG/HCLOG above -- decode fusion for this
// path is future work, same as the staged GolombRiceStage's own decode). Must
// reproduce modules/coders/golomb_rice/golomb_rice_stage.cu's
// golombRiceEncodeKernel BYTE-FOR-BYTE (same k selection, same kIntervalsPerChunk
// restart-interval offsets, same header layout) so the ordinary, unfused
// GolombRiceStage::execute() inverse can decode an archive this fused encoder
// produced -- it never gets its own inverse.
//
// `in` here is `cur`, the chunk-local PLAIN signed difference a DiffPlain
// transform produced (bit-reinterpreted through uint32_t). This coder zigzag-
// encodes it internally (see grZigzag() below) -- exactly matching the
// standalone GolombRiceStage kernel's own inline zigzagEncode<T>() step. An
// upstream transform that ALSO zigzagged would double-encode; that was tried
// first and caught by the byte-identity test (fused archive came out smaller
// than staged -- double-zigzag of an always-nonnegative value just doubles
// it, inflating every Rice cost by ~1 bit).
//
// Uses `temp` (TEMP_BYTES=4096) for all scratch instead of declaring new
// __shared__ arrays -- the harness already budgets sA/sB (16 KB each) for
// `in`/`out`; needed scratch (~1.8 KB, see the layout comment below) fits
// comfortably in the coder's `temp` slice with room to spare.
namespace golomb_rice_detail {

constexpr uint32_t kGrEscapeQ           = 24u;
constexpr uint32_t kGrRawBits           = 32u;   // bitWidth<int32_t>()
constexpr uint32_t kGrMaxK              = 24u;   // kMaxCandidate<int32_t>()
constexpr uint32_t kGrIntervalsPerChunk = TPB / 32u;   // 16 at TPB=512
constexpr uint32_t kGrHeaderBytes       = 4u + 4u * kGrIntervalsPerChunk;   // 68

// Scatter the low `nbits` of `value` into a flat little-endian bit array
// starting at `bit_off`. Identical to golomb_rice_stage.cu's orBits() --
// atomicOr because two elements' disjoint bit ranges can still share one
// 32-bit WORD.
__device__ __forceinline__ void grOrBits(uint32_t* words, uint64_t bit_off,
                                         uint64_t value, uint32_t nbits) {
    while (nbits > 0) {
        const uint32_t widx = static_cast<uint32_t>(bit_off >> 5);
        const uint32_t boff = static_cast<uint32_t>(bit_off & 31u);
        const uint32_t take = min(nbits, 32u - boff);
        const uint32_t mask = (take == 32u) ? 0xFFFFFFFFu : ((1u << take) - 1u);
        const uint32_t chunk = static_cast<uint32_t>(value & mask);
        atomicOr(&words[widx], chunk << boff);
        value  >>= take;
        bit_off += take;
        nbits   -= take;
    }
}

// Exact bit cost of one value under Rice parameter k, with the escape cap --
// identical to golomb_rice_stage.cu's riceCost<kGrEscapeQ, kGrRawBits>().
__device__ __forceinline__ uint32_t grRiceCost(uint32_t u, uint32_t k) {
    const uint32_t q = u >> k;
    return (q < kGrEscapeQ) ? (q + 1u + k) : (kGrEscapeQ + kGrRawBits);
}

// zigzag(signed int32 delta -> unsigned). GolombRiceCoder does this itself
// (its input is DiffPlain's plain signed delta, bit-reinterpreted through
// uint32_t) -- exactly mirroring the standalone GolombRiceStage kernel's own
// inline zigzagEncode<T>() step, which exists precisely because that kernel
// has no upstream transform to rely on. An upstream transform that ALSO
// zigzagged (e.g. a hypothetical DiffZigzag) would double-encode -- caught
// by the byte-identity test (it was; see the fix history in this file's
// commit / chunk_local_entropy_coder_design.md).
__device__ __forceinline__ uint32_t grZigzag(uint32_t bits) {
    return Zigzag<int32_t>::encode(static_cast<int32_t>(bits));
}

} // namespace golomb_rice_detail

template <int ChunkBytes>
struct GolombRiceCoder {
    using Params = EmptyParams;
    __device__ static bool encode(int& csize, byte* in, byte* out, byte* temp, const void* /*pp*/) {
        using namespace golomb_rice_detail;
        // Chunk geometry is a template parameter (ChunkBytes) like every sibling
        // coder above; NELEM/CHUNK_BYTES were global constants before the
        // Geom<> templating refactor.
        constexpr int       NELEM      = Geom<ChunkBytes>::NELEM;
        constexpr int       CHUNK_BYTES = ChunkBytes;
        constexpr uint32_t kMaxK   = kGrMaxK;
        constexpr int       nwarps = TPB >> 5;
        constexpr int       LOCAL  = NELEM / TPB;

        const int       live  = csize / 4;      // csize enters as in_size (bytes)
        const int       tid   = threadIdx.x;
        const int       warp  = tid >> 5, lane = tid & 31;
        // `in` holds DiffPlain's plain signed per-chunk deltas (bit-reinterpreted
        // through uint32_t, NOT zigzag-encoded yet) -- grZigzag() below does that.
        const uint32_t* deltas = reinterpret_cast<const uint32_t*>(in);

        // `temp` (TEMP_BYTES=4096 B) layout, all uint32 slots:
        //   [0 .. nwarps*(kMaxK+1))            s_warp_cost[warp][k]   (400 slots)
        //   [.. + (kMaxK+1))                   s_total_cost[k]        ( 25 slots)
        //   [.. + 1)                           s_k                    (  1 slot )
        //   [.. + nwarps)                      s_warp_bytelen[warp]   ( 16 slots)
        //   [.. + nwarps)                      s_warp_bytebase[warp]  ( 16 slots)
        //   [.. + 1)                           s_total_payload_bytes  (  1 slot )
        // Total 459 slots = 1836 B, comfortably under 4096 B.
        uint32_t* t              = reinterpret_cast<uint32_t*>(temp);
        uint32_t* s_warp_cost    = t;
        uint32_t* s_total_cost   = s_warp_cost + nwarps * (kMaxK + 1);
        uint32_t* s_k            = s_total_cost + (kMaxK + 1);
        uint32_t* s_warp_bytelen = s_k + 1;
        uint32_t* s_warp_bytebase= s_warp_bytelen + nwarps;
        uint32_t* s_total_bytes  = s_warp_bytebase + nwarps;

        // ---- Phase 1: exact cost search over k in [0, kMaxK] ----
        uint32_t local_cost[kMaxK + 1];
        #pragma unroll
        for (uint32_t k = 0; k <= kMaxK; ++k) local_cost[k] = 0u;
        for (int j = 0; j < LOCAL; ++j) {
            const int i = tid * LOCAL + j;
            if (i >= live) continue;
            const uint32_t u = grZigzag(deltas[i]);
            #pragma unroll
            for (uint32_t k = 0; k <= kMaxK; ++k) local_cost[k] += grRiceCost(u, k);
        }
        #pragma unroll
        for (uint32_t k = 0; k <= kMaxK; ++k) {
            uint32_t c = local_cost[k];
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) c += __shfl_down_sync(0xffffffffu, c, off, 32);
            if (lane == 0) s_warp_cost[warp * (kMaxK + 1) + k] = c;
        }
        __syncthreads();
        if (warp == 0) {
            #pragma unroll
            for (uint32_t k = 0; k <= kMaxK; ++k) {
                uint32_t c = (lane < nwarps) ? s_warp_cost[lane * (kMaxK + 1) + k] : 0u;
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1) c += __shfl_down_sync(0xffffffffu, c, off, 32);
                if (lane == 0) s_total_cost[k] = c;
            }
            if (lane == 0) {
                uint32_t best = 0;
                for (uint32_t k = 1; k <= kMaxK; ++k)
                    if (s_total_cost[k] < s_total_cost[best]) best = k;
                *s_k = best;
            }
        }
        __syncthreads();
        const uint32_t k_opt = *s_k;

        // ---- Phase 2: per-thread length -> WARP-LOCAL exclusive scan (each
        // warp is one restart interval, byte-aligned independently). ----
        uint32_t local_len[LOCAL];
        uint32_t thread_total = 0;
        for (int j = 0; j < LOCAL; ++j) {
            const int i = tid * LOCAL + j;
            const uint32_t len = (i < live) ? grRiceCost(grZigzag(deltas[i]), k_opt) : 0u;
            local_len[j] = thread_total;
            thread_total += len;
        }
        uint32_t warp_excl = thread_total;
        #pragma unroll
        for (int d = 1; d < 32; d <<= 1) {
            const uint32_t up = __shfl_up_sync(0xffffffffu, warp_excl, d, 32);
            if (lane >= d) warp_excl += up;
        }
        const uint32_t warp_bit_total = __shfl_sync(0xffffffffu, warp_excl, 31, 32);
        warp_excl -= thread_total;
        const uint32_t warp_byte_len = (warp_bit_total + 7u) >> 3;
        if (lane == 0) s_warp_bytelen[warp] = warp_byte_len;
        __syncthreads();
        if (warp == 0) {
            uint32_t v = (lane < nwarps) ? s_warp_bytelen[lane] : 0u;
            uint32_t excl = v;
            #pragma unroll
            for (int d = 1; d < 32; d <<= 1) {
                const uint32_t up = __shfl_up_sync(0xffffffffu, excl, d, 32);
                if (lane >= d) excl += up;
            }
            const uint32_t grand_total = __shfl_sync(0xffffffffu, excl, nwarps - 1, 32);
            excl -= v;
            if (lane < nwarps) s_warp_bytebase[lane] = excl;
            if (lane == 0) *s_total_bytes = grand_total;
        }
        __syncthreads();
        const uint64_t thread_base = static_cast<uint64_t>(s_warp_bytebase[warp]) * 8ull + warp_excl;
        const uint32_t total_payload_bytes = *s_total_bytes;

        const uint32_t packed_bytes = kGrHeaderBytes + total_payload_bytes;
        const uint32_t orig_bytes   = static_cast<uint32_t>(live) * 4u;

        // Never write more than `out`'s CHUNK_BYTES capacity, and let the
        // harness's own csize<in_size check decide the raw-copy fallback --
        // bail out with no writes whenever packing wouldn't help or wouldn't
        // fit (GolombRice's escape-bounded worst case CAN exceed CHUNK_BYTES,
        // unlike the byte-level LC coders above).
        if (packed_bytes >= orig_bytes || packed_bytes > static_cast<uint32_t>(CHUNK_BYTES))
            return false;

        // ---- Phase 3: zero the payload region, then atomicOr-pack. ----
        byte*     outb  = out;
        uint32_t* words = reinterpret_cast<uint32_t*>(outb + kGrHeaderBytes);
        const uint32_t payload_words = (total_payload_bytes + 3) / 4 + 1;   // +1 word tail safety
        for (uint32_t w = tid; w < payload_words; w += TPB) words[w] = 0u;
        if (tid == 0) *reinterpret_cast<uint32_t*>(outb) = k_opt;
        if (lane == 0) reinterpret_cast<uint32_t*>(outb + 4)[warp] = s_warp_bytebase[warp];
        __syncthreads();

        for (int j = 0; j < LOCAL; ++j) {
            const int i = tid * LOCAL + j;
            if (i >= live) continue;
            const uint32_t u   = grZigzag(deltas[i]);
            const uint32_t q   = u >> k_opt;
            const uint64_t off = thread_base + local_len[j];
            if (q < kGrEscapeQ) {
                const uint32_t r = u & ((k_opt == 32u) ? 0xFFFFFFFFu : ((1u << k_opt) - 1u));
                const uint64_t codeword = ((1ull << q) - 1ull) | (static_cast<uint64_t>(r) << (q + 1u));
                grOrBits(words, off, codeword, q + 1u + k_opt);
            } else {
                const uint64_t codeword = ((1ull << kGrEscapeQ) - 1ull) | (static_cast<uint64_t>(u) << kGrEscapeQ);
                grOrBits(words, off, codeword, kGrEscapeQ + kGrRawBits);
            }
        }
        __syncthreads();
        csize = static_cast<int>(packed_bytes);
        return true;
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

// ── Harness body: one CTA per chunk. Quant (Elementwise) -> Transforms... -> Coder (sink).
// Stage-agnostic: it composes whatever ops it is given. Factored out of the
// __global__ entry so the NVRTC codegen path can wrap it in an `extern "C"`
// kernel (a __global__ cannot call another __global__). Both the compile-time
// template kernel below and the runtime-generated kernel call this same body —
// the ops stay single-sourced; only the composing glue differs. ──────────────
// `params` is the packed per-op Params blob, ops in execution order
// ([QuantOp][Transforms...][Coder]); each op is handed its slice at a
// compile-time offset. Stateless ops ignore it.
// `side` carries escaping outputs (e.g. an outlier list); the Elementwise op uses it or
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
