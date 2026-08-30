#pragma once

/**
 * @file modules/fused/fused_block/warp_ti_fusion.cuh
 * @brief Thread-INDEPENDENT warp fusion harness (cuSZp-style layout).
 *
 * The warp-COOPERATIVE harness (warp_fusion.cuh) maps 32 lanes to the 32 elements of ONE
 * block: prediction is a cross-lane __shfl chain and packing is a __ballot bit-transpose.
 * That cross-lane traffic is what ncu measured as compute-bound (71% SM) — below native
 * cuSZp's memory-bound ceiling — and it is why a tile-cooperative predictor (FSZ's
 * AdaptiveLorenzo, a 256-element 4-mode decision) cannot warp-fuse in that layout.
 *
 * This harness is the cuSZp layout: CTA = 1 warp, and each THREAD owns a whole 32-element
 * block. The predictor runs SERIALLY in-thread (prev in a register, no cross-lane comm);
 * the coder packs its own block in registers. A warp is 32 threads each on a DIFFERENT
 * block, so there is no cross-lane ballot/shfl at all — only the byte-offset prefix-sum is
 * cooperative, and that reuses `warp_decoupled_lookback` from warp_fusion.cuh.
 *
 * BYTE-IDENTITY: we target the AdaptiveBitpack byte stream so the STAGED inverse round-trips
 * (milestone 1 reuses it — no new decoder). The per-block codes are numerically identical to
 * the warp-cooperative Lorenzo1D (serial vs shuffled, same values), so a byte-identical coder
 * is possible. The one thing that differs from cuSZp is the byte ORDER: cuSZp writes the
 * stream thread-major, but AB is LINEAR block order, so this harness assigns offsets in linear
 * order via a 2-D warp prefix (per-row scan + cumulative row sums), not cuSZp's thread-major
 * scan. See memory/cost_based_fusion_optimizer.md (THREAD-INDEPENDENT PATH — BUILD PLAN).
 */

#include "fused/fused_block/warp_fusion.cuh"   // warp_decoupled_lookback, LB_AGG/LB_PREFIX
#include "fused/fused_block/warp_op_params.h"  // Lorenzo1DParams (shared with the warp-coop path)
#include <cstdint>

namespace fz { namespace fused { namespace warp_ti {

// ── Predictor policy interface ───────────────────────────────────────────────
// A thread-independent predictor fills d[32] with the signed codes for the 32-element block
// starting at global element index `base` — serial, register-resident, no cross-lane comm.
//   struct P {
//     static P fromParams(const float* in, size_t n, const void* pp);
//     __device__ void predict(size_t base, int (&d)[32]) const;   // d[i] = code for element base+i
//   };

// Serial 1-D Lorenzo (cuSZp2 predictor). Byte-identical codes to the warp-cooperative
// Lorenzo1DPredictor: prev resets per block, d[0]=q0, d[i]=qi-q_{i-1}.
struct ThreadLorenzo1DPredictor {
    const float* in;
    size_t       n;
    float        inv2eb;
    __device__ static ThreadLorenzo1DPredictor fromParams(const float* in, size_t n, const void* pp) {
        return ThreadLorenzo1DPredictor{in, n, static_cast<const warp::Lorenzo1DParams*>(pp)->inv2eb};
    }
    __device__ __forceinline__ void predict(size_t base, int (&d)[32]) const {
        int prev = 0;
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            const size_t g = base + static_cast<size_t>(i);
            const int q = (g < n) ? __float2int_rn(in[g] * inv2eb) : 0;
            d[i] = q - prev;
            prev = q;
        }
    }
};

// ── Coder policy interface (Phase 2) ─────────────────────────────────────────
// A thread-independent coder consumes one thread's 32 block codes and works in-register
// (no cross-lane ops). Byte-identical to AdaptiveBitpack so the staged inverse decodes it.
//   struct C {
//     static constexpr uint32_t meta_bytes;
//     // Compute this block's payload byte length; write its meta bytes. Returns the length.
//     __device__ static uint32_t cost(const int (&d)[32], uint32_t word_bytes, uint32_t count,
//                                      uint8_t* meta);
//     // Write this block's payload at `out` (length == the cost() return).
//     __device__ static void pack(const int (&d)[32], uint32_t word_bytes, uint32_t count,
//                                 const uint8_t* meta, uint8_t* out);
//   };

// Thread-independent AdaptiveBitpack: one thread owns all 32 block codes, so the warp-
// cooperative ballot/shfl of AdaptiveBitpackCoder becomes serial in-register loops. Emits
// the IDENTICAL byte stream (same meta/selector/sign/plane layout, 32-bit = 4-byte words),
// so the staged AdaptiveBitpack inverse decodes it. 32-element blocks only ⇒ word_bytes==4.
struct ThreadFixedRateCoder {
    static constexpr uint32_t meta_bytes = 2;

    // Writes meta[0..1] and returns this block's payload byte length. Mirrors
    // AdaptiveBitpackCoder::cost<1>: plain (fixed-rate over all) vs outlier (elem0 raw + rate
    // over the rest), whichever is cheaper.
    __device__ static __forceinline__ uint32_t cost(const int (&d)[32], uint32_t word_bytes,
                                                    uint32_t count, uint8_t* __restrict__ meta) {
        uint32_t acc_all = 0u, acc_rest = 0u;
        for (uint32_t i = 0; i < count; ++i) {
            const uint32_t av = warp::absU_i32(d[i]);
            acc_all |= av;
            if (i > 0) acc_rest |= av;
        }
        const uint32_t mag0     = (count > 0) ? warp::absU_i32(d[0]) : 0u;
        const int      fr_all   = warp::bitWidth32(acc_all);
        const int      fr_rest  = warp::bitWidth32(acc_rest);
        const uint32_t ob_bytes = static_cast<uint32_t>((warp::bitWidth32(mag0) + 7) / 8);
        const uint32_t cost_plain = (fr_all  > 0) ? word_bytes * (fr_all  + 1u) : 0u;
        const uint32_t cost_out   = ob_bytes + ((fr_rest > 0) ? word_bytes * (fr_rest + 1u) : word_bytes);
        if (cost_plain <= cost_out) { meta[0] = static_cast<uint8_t>(fr_all); meta[1] = 0; return cost_plain; }
        meta[0] = static_cast<uint8_t>(fr_rest);
        meta[1] = static_cast<uint8_t>(1u | ((ob_bytes - 1u) << 1));
        return cost_out;
    }

    // Writes the payload. Mirrors AdaptiveBitpackCoder::pack<1> byte-for-byte: sign mask (4 B
    // LE, bit i = element i) then r bit-planes (4 B each); outlier prepends elem0's magnitude
    // and drops elem0 from the planes.
    __device__ static __forceinline__ void pack(const int (&d)[32], uint32_t word_bytes,
                                                uint32_t count, const uint8_t* __restrict__ meta,
                                                uint8_t* __restrict__ out) {
        const int     r      = meta[0];
        const uint8_t sel    = meta[1];
        const bool    is_out = (sel & 1u) != 0;

        if (!is_out) {
            if (r == 0) return;
            uint32_t sm = 0u;
            for (uint32_t i = 0; i < count; ++i) if (d[i] < 0) sm |= (1u << i);
            for (uint32_t k = 0; k < 4u; ++k) out[k] = static_cast<uint8_t>((sm >> (8u * k)) & 0xFFu);
            for (int p = 0; p < r; ++p) {
                uint32_t pm = 0u;
                for (uint32_t i = 0; i < count; ++i)
                    if ((warp::absU_i32(d[i]) >> p) & 1u) pm |= (1u << i);
                for (uint32_t k = 0; k < 4u; ++k)
                    out[word_bytes * (1u + p) + k] = static_cast<uint8_t>((pm >> (8u * k)) & 0xFFu);
            }
            return;
        }
        // Outlier: [ob_bytes elem0 magnitude LE][sign of all elems][r planes for elems 1..].
        const uint32_t ob_bytes = ((sel >> 1) & 3u) + 1u;
        const uint32_t mag0     = (count > 0) ? warp::absU_i32(d[0]) : 0u;
        for (uint32_t k = 0; k < ob_bytes; ++k) out[k] = static_cast<uint8_t>((mag0 >> (8u * k)) & 0xFFu);
        uint8_t* sign   = out + ob_bytes;
        uint8_t* planes = out + ob_bytes + word_bytes;
        uint32_t sm = 0u;
        for (uint32_t i = 0; i < count; ++i) if (d[i] < 0) sm |= (1u << i);
        for (uint32_t k = 0; k < 4u; ++k) sign[k] = static_cast<uint8_t>((sm >> (8u * k)) & 0xFFu);
        for (int p = 0; p < r; ++p) {
            uint32_t pm = 0u;
            for (uint32_t i = 1; i < count; ++i)
                if ((warp::absU_i32(d[i]) >> p) & 1u) pm |= (1u << i);
            for (uint32_t k = 0; k < 4u; ++k)
                planes[word_bytes * p + k] = static_cast<uint8_t>((pm >> (8u * k)) & 0xFFu);
        }
    }
};

// ── Harness (Phase 3) ────────────────────────────────────────────────────────
// CTA = 1 warp. Thread `lane` owns block (jb,lane) = linear index warp_block_base + jb*32 + lane
// for jb in [0,BlocksPerWarp). Holds all codes, computes LINEAR-order byte offsets with a 2-D
// warp prefix, gets the warp's base via the decoupled look-back, then packs. `meta`/`payload`
// point at the AdaptiveBitpack meta region and payload region of the archive.
template<int BlocksPerWarp, class Coder, class Pred>
__device__ __forceinline__ void fused_ti_body(
    Pred pred, size_t n, uint32_t word_bytes, size_t num_blocks,
    uint8_t* __restrict__ meta, uint8_t* __restrict__ payload,
    uint32_t* __restrict__ g_state, uint32_t* __restrict__ g_agg,
    uint32_t* __restrict__ g_incl, size_t num_warps)
{
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t w    = blockIdx.x;                 // CTA = 1 warp ⇒ global warp id
    if (static_cast<size_t>(w) >= num_warps) return;

    const size_t warp_block_base = static_cast<size_t>(w) * BlocksPerWarp * 32u;

    // ── Phase A: predict + cost every block this thread owns (one per row jb).
    int      d[BlocksPerWarp][32];   // held codes (local memory) — no recompute in pack
    uint32_t bcost[BlocksPerWarp];   // this lane's block cost per row
    #pragma unroll 1
    for (int jb = 0; jb < BlocksPerWarp; ++jb) {
        const size_t b = warp_block_base + static_cast<size_t>(jb) * 32u + lane;
        if (b < num_blocks) {
            const size_t base = b * 32u;
            const uint32_t count = static_cast<uint32_t>(min(size_t{32}, n - base));
            pred.predict(base, d[jb]);
            bcost[jb] = Coder::cost(d[jb], word_bytes, count, meta + Coder::meta_bytes * b);
        } else {
            bcost[jb] = 0u;
        }
    }

    // ── Phase B: LINEAR-order 2-D prefix. block(jb,lane) sits at linear pos jb*32+lane, so its
    // exclusive byte offset within the warp = (sum of rows < jb) + (intra-row exclusive prefix).
    uint32_t blk_excl[BlocksPerWarp];
    uint32_t warp_total = 0u;        // cumulative sum of full rows processed so far
    #pragma unroll 1
    for (int jb = 0; jb < BlocksPerWarp; ++jb) {
        const uint32_t v = bcost[jb];
        uint32_t inc = v;
        #pragma unroll
        for (int off = 1; off < 32; off <<= 1) {
            const uint32_t t = __shfl_up_sync(0xffffffffu, inc, off);
            if (lane >= static_cast<uint32_t>(off)) inc += t;
        }
        const uint32_t rowsum = __shfl_sync(0xffffffffu, inc, 31);
        blk_excl[jb] = warp_total + (inc - v);   // rows<jb + intra-row exclusive
        warp_total  += rowsum;
    }

    // ── Phase C: decoupled look-back over per-warp aggregates → this warp's base byte offset.
    if (lane == 0u) { g_agg[w] = warp_total; __threadfence(); g_state[w] = warp::LB_AGG; }
    const uint32_t warp_base = warp::warp_decoupled_lookback(w,
        reinterpret_cast<volatile uint32_t*>(g_state),
        reinterpret_cast<volatile uint32_t*>(g_agg),
        reinterpret_cast<volatile uint32_t*>(g_incl), lane);
    if (lane == 0u) { g_incl[w] = warp_base + warp_total; __threadfence(); g_state[w] = warp::LB_PREFIX; }

    // ── Phase D: pack every held block at its resolved linear offset.
    #pragma unroll 1
    for (int jb = 0; jb < BlocksPerWarp; ++jb) {
        const size_t b = warp_block_base + static_cast<size_t>(jb) * 32u + lane;
        if (b < num_blocks) {
            const size_t base = b * 32u;
            const uint32_t count = static_cast<uint32_t>(min(size_t{32}, n - base));
            Coder::pack(d[jb], word_bytes, count, meta + Coder::meta_bytes * b,
                        payload + warp_base + blk_excl[jb]);
        }
    }
}

} } } // namespace fz::fused::warp_ti
