#pragma once

/**
 * @file modules/fused/fused_block/warp_fusion.cuh
 * @brief Warp-register fusion HARNESS — the NVRTC-compilable device pieces of the
 *        block-local "predict+quant+fixed-rate-coder" family (cuSZp2 / cuSZp3).
 *
 * The warp-register counterpart to chunk_fusion.cuh. One warp owns one block and
 * runs quant + predictor delta + adaptive-bitpack in registers (no DRAM round-trip
 * for the intermediate codes). This header holds ONLY device code so both the
 * compile-time template path (fused_block.cu) and the runtime NVRTC path
 * (nvrtc_warp_fusion.cpp) compile the SAME bodies — they stay byte-identical.
 *
 * The DESIGN (parallels the chunk strategy):
 *   - Each predictor is a small POD POLICY with a `delta(lane, b, m)` that produces
 *     the signed int code for one element inline from the float input. Its runtime
 *     config (inv2eb, dims, tile shape) is a `Params` POD the host packs into a
 *     blob; `in`/`n` (only known at launch) are injected via `fromParams`.
 *   - The HARNESS (`fused_rate_body` / `fused_pack_body`) is predictor-agnostic:
 *     rate computes each block's fixed bit-rate + cost, pack emits the bit-planes.
 *     Templated on ElemsPerLane (= block_size/32) and the predictor policy.
 *   - The host-side CUB exclusive-scan of the per-block costs and the length
 *     read-back stay in the launcher (host orchestration, NOT in these kernels), so
 *     the NVRTC surface is just intrinsics — no cub/thrust.
 */

#include "fused/fused_block/warp_op_params.h"   // shared POD predictor Params (host/device agree)
#include <cstdint>   // NVRTC receives a stub with the fixed-width aliases

namespace fz {
namespace fused {
namespace warp {

__device__ __forceinline__ uint32_t absU_i32(int v) {
    return static_cast<uint32_t>(v < 0 ? -v : v);
}
__device__ __forceinline__ int bitWidth32(uint32_t x) { return x ? (32 - __clz(x)) : 0; }

// ── Predictor policies ──────────────────────────────────────────────────────
// delta() returns the signed int code (delta) for element `lane + 32*m` of warp-
// block `b`, quantising the float input inline. Out-of-range / padding elements
// return 0 (matching the staged predictor). Small PODs passed by value — no global
// state, no spills. `fromParams` reconstructs the policy from the launch input +
// the packed config blob (the uniform NVRTC factory).

// cuSZp2 / SZp: linear-ABS quant + 1-D Lorenzo delta, reset per BLOCK (32*epl
// elements). All lanes call delta() so the intra-warp shuffles are collective.
// For epl == 1 this is exactly cuSZp2's per-32 reset; for epl > 1 the delta chain
// runs across the whole block, element `32*m + lane` of block `b`, resetting only
// at (lane 0, m 0). Byte-identical to staged Quantizer(linear) -> Lorenzo(32*epl).
struct Lorenzo1DPredictor {
    const float* in;
    size_t n;
    float inv2eb;
    uint32_t epl;
    __device__ static Lorenzo1DPredictor fromParams(const float* in, size_t n, const void* pp) {
        const Lorenzo1DParams p = *static_cast<const Lorenzo1DParams*>(pp);
        return Lorenzo1DPredictor{in, n, p.inv2eb, p.epl ? p.epl : 1u};
    }
    __device__ __forceinline__ int q_at(size_t gidx) const {
        return (gidx < n) ? __float2int_rn(in[gidx] * inv2eb) : 0;
    }
    __device__ __forceinline__ int delta(uint32_t lane, size_t b, int m) const {
        const size_t base = b * (32ull * epl) + 32ull * static_cast<uint32_t>(m);
        const int q      = q_at(base + lane);
        const int q_up   = __shfl_up_sync(0xffffffffu, q, 1);            // predecessor in this row
        int q_row_prev   = (m > 0 && lane == 31u) ? q_at(base - 1ull) : 0;  // last elem of previous row
        q_row_prev       = __shfl_sync(0xffffffffu, q_row_prev, 31);
        if (lane != 0u)  return q - q_up;
        return (m == 0) ? q : (q - q_row_prev);
    }

    // INVERSE: block-reset inclusive prefix sum of the per-lane delta array `d`
    // (elements `32*m + lane` of the block), in place. Exactly reverses delta():
    // a warp scan per 32-lane row, then a running carry across rows. Integer add
    // is associative so this matches the staged Hillis-Steele scan bit-for-bit.
    template<int EPL>
    __device__ static void undelta(int (&d)[EPL], uint32_t lane) {
        int carry = 0;
        #pragma unroll
        for (int m = 0; m < EPL; ++m) {
            int s = d[m];
            #pragma unroll
            for (int off = 1; off < 32; off <<= 1) {
                const int y = __shfl_up_sync(0xffffffffu, s, off);
                if (lane >= static_cast<uint32_t>(off)) s += y;
            }
            d[m] = s + carry;
            carry += __shfl_sync(0xffffffffu, s, 31);   // row total (pre-carry) to all lanes
        }
    }
};

// cuSZp3: linear-ABS quant + 2-D separable tiled Lorenzo (tz == 1). Each element
// re-quantises its own left/up predecessor from the float field (a pure map, so the
// neighbour code equals what the staged quantizer produced). Mirrors
// tiled_lorenzo_delta_kernel exactly. tile_elems == tx*ty == block_size.
struct TiledLorenzo2DPredictor {
    const float* in;
    float inv2eb;
    uint32_t dx, dy, tx, ty, ntx;
    __device__ static TiledLorenzo2DPredictor fromParams(const float* in, size_t /*n*/, const void* pp) {
        const TiledLorenzo2DParams p = *static_cast<const TiledLorenzo2DParams*>(pp);
        return TiledLorenzo2DPredictor{in, p.inv2eb, p.dx, p.dy, p.tx, p.ty, p.ntx};
    }
    __device__ __forceinline__ int delta(uint32_t lane, size_t b, int m) const {
        const uint32_t local = lane + 32u * static_cast<uint32_t>(m);   // element within tile
        const uint32_t lx = local % tx;
        const uint32_t ly = local / tx;                                 // tz==1 ⇒ ly < ty
        const uint32_t tix = static_cast<uint32_t>(b % ntx);
        const uint32_t tiy = static_cast<uint32_t>(b / ntx);
        const uint32_t gx = tix * tx + lx;
        const uint32_t gy = tiy * ty + ly;
        if (gx >= dx || gy >= dy) return 0;                             // padding
        const size_t gidx = static_cast<size_t>(gy) * dx + gx;
        const int cur = __float2int_rn(in[gidx] * inv2eb);
        int pred;
        if (lx > 0)      pred = __float2int_rn(in[gidx - 1] * inv2eb);          // X-delta
        else if (ly > 0) pred = __float2int_rn(in[gidx - dx] * inv2eb);         // Y-delta
        else             pred = 0;                                             // tile origin
        return cur - pred;
    }
};

// ── Transform ops (optional, between predictor and coder) ────────────────────
// A warp transform is a size-preserving register→register map on the per-lane delta
// array `d[EPL]`, applied after the predictor and before the coder (the register
// analogue of a chunk transform). Element-wise transforms ignore `lane`; ones that
// need neighbours use warp shuffles. Its staged counterpart must invert it on decode
// (decompress is not fused), so the fused op must match the staged transform exactly.

// Zigzag (TCMS): signed → unsigned interleave, per element. Matches ZigzagStage<int32_t>.
struct ZigzagTransform {
    template<int EPL>
    __device__ static void apply(int (&d)[EPL], uint32_t /*lane*/) {
        #pragma unroll
        for (int m = 0; m < EPL; ++m) {
            const uint32_t z = (static_cast<uint32_t>(d[m]) << 1) ^ static_cast<uint32_t>(d[m] >> 31);
            d[m] = static_cast<int>(z);
        }
    }
};

// Apply a transform pack to d[] in order (no-op for the empty chain).
template<int EPL>
__device__ __forceinline__ void applyTransforms(int (&)[EPL], uint32_t) {}
template<int EPL, class T, class... R>
__device__ __forceinline__ void applyTransforms(int (&d)[EPL], uint32_t lane) {
    T::template apply<EPL>(d, lane);
    applyTransforms<EPL, R...>(d, lane);
}

// cuSZp3 3-D (PROTOTYPE): linear-ABS quant + 3-D separable tiled Lorenzo (tz > 1).
// Mirrors tiled_lorenzo_delta_kernel's 3-D path exactly (X-delta if lx>0, else Y if
// ly>0, else Z if lz>0, else tile origin), re-quantising each neighbour from the float
// field so the delta equals the staged code delta (byte-identical). tile_elems==64 ⇒
// local ∈ [0,64). NOTE: like the 2-D predictor this re-reads neighbours from GLOBAL and
// the fused kernel recomputes it in BOTH the rate and pack passes.
struct TiledLorenzo3DPredictor {
    const float* in;
    float inv2eb;
    uint32_t dx, dy, dz, tx, ty, tz, ntx, nty;
    __device__ static TiledLorenzo3DPredictor fromParams(const float* in, size_t /*n*/, const void* pp) {
        const TiledLorenzo3DParams p = *static_cast<const TiledLorenzo3DParams*>(pp);
        return TiledLorenzo3DPredictor{in, p.inv2eb, p.dx, p.dy, p.dz, p.tx, p.ty, p.tz, p.ntx, p.nty};
    }
    __device__ __forceinline__ int delta(uint32_t lane, size_t b, int m) const {
        const uint32_t local = lane + 32u * static_cast<uint32_t>(m);   // element within tile
        const uint32_t lx = local % tx;
        const uint32_t ly = (local / tx) % ty;
        const uint32_t lz = local / (tx * ty);
        const uint32_t tix = static_cast<uint32_t>(b % ntx);
        const uint32_t tiy = static_cast<uint32_t>((b / ntx) % nty);
        const uint32_t tiz = static_cast<uint32_t>(b / (static_cast<size_t>(ntx) * nty));
        const uint32_t gx = tix * tx + lx;
        const uint32_t gy = tiy * ty + ly;
        const uint32_t gz = tiz * tz + lz;
        if (gx >= dx || gy >= dy || gz >= dz) return 0;                 // padding
        const size_t gidx = (static_cast<size_t>(gz) * dy + gy) * dx + gx;
        const int cur = __float2int_rn(in[gidx] * inv2eb);
        int pred;
        if (lx > 0)      pred = __float2int_rn(in[gidx - 1] * inv2eb);                         // X
        else if (ly > 0) pred = __float2int_rn(in[gidx - dx] * inv2eb);                        // Y
        else if (lz > 0) pred = __float2int_rn(in[gidx - static_cast<size_t>(dx) * dy] * inv2eb); // Z
        else             pred = 0;                                                             // origin
        return cur - pred;
    }
};

// ── Coder policies (the swappable Cooperative sink) ──────────────────────────
// A warp coder is the variable-length tail of the register chain. Its two halves
// mirror the two-pass driver: `cost()` (called by ALL lanes — it warp-reduces the
// per-lane deltas) writes the block's `meta` + byte `cost`; `pack()` writes the
// block's payload at `base` given that meta. `meta_bytes` is the per-block meta the
// driver reserves. Both take the per-lane delta array `d[EPL]` the predictor produced
// — quant/predict stay in the predictor, the coder only consumes deltas, so swapping
// the coder re-composes the kernel with no other change (the LC-of the chunk coders).

// AdaptiveBitpack: per block, pick the cheaper of a plain fixed-rate bit-plane pack
// (rate over ALL elements) and an "outlier" pack (element 0 stored raw, rate over the
// rest). meta = [rate][selector]; selector bit0 = outlier, bits1-2 = ob_bytes-1.
struct AdaptiveBitpackCoder {
    static constexpr uint32_t meta_bytes = 2;

    template<int EPL>
    __device__ static void cost(const int (&d)[EPL], uint32_t lane, uint32_t word_bytes,
                                size_t count, uint8_t* __restrict__ meta_b,
                                uint32_t* __restrict__ cost_b) {
        uint32_t acc_all = 0, acc_rest = 0;
        #pragma unroll
        for (int m = 0; m < EPL; ++m) {
            const size_t idx = static_cast<size_t>(lane) + 32u * m;
            const uint32_t av = (idx < count) ? absU_i32(d[m]) : 0u;
            acc_all |= av;
            if (idx > 0) acc_rest |= av;
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            acc_all  |= __shfl_xor_sync(0xffffffffu, acc_all, off);
            acc_rest |= __shfl_xor_sync(0xffffffffu, acc_rest, off);
        }
        if (lane == 0) {
            const uint32_t mag0 = (count > 0) ? absU_i32(d[0]) : 0u;
            const int fr_all = bitWidth32(acc_all), fr_rest = bitWidth32(acc_rest);
            const uint32_t ob_bytes = static_cast<uint32_t>((bitWidth32(mag0) + 7) / 8);
            const uint32_t cost_plain = (fr_all > 0) ? word_bytes * (fr_all + 1u) : 0u;
            const uint32_t cost_out   = ob_bytes + ((fr_rest > 0) ? word_bytes * (fr_rest + 1u) : word_bytes);
            if (cost_plain <= cost_out) { meta_b[0]=fr_all;  meta_b[1]=0; *cost_b=cost_plain; }
            else { meta_b[0]=fr_rest; meta_b[1]=static_cast<uint8_t>(1u|((ob_bytes-1u)<<1)); *cost_b=cost_out; }
        }
    }

    template<int EPL>
    __device__ static void pack(const int (&d)[EPL], uint32_t lane, uint32_t word_bytes,
                                size_t count, const uint8_t* __restrict__ meta_b,
                                uint8_t* __restrict__ base) {
        const int r = meta_b[0]; const uint8_t sel = meta_b[1];
        const bool is_out = (sel & 1u) != 0;
        uint32_t av[EPL]; bool active[EPL];
        #pragma unroll
        for (int m = 0; m < EPL; ++m) {
            const size_t idx = static_cast<size_t>(lane) + 32u * m;
            active[m] = idx < count;
            av[m]     = active[m] ? absU_i32(d[m]) : 0u;
        }
        if (!is_out) {
            if (r == 0) return;
            #pragma unroll
            for (int m = 0; m < EPL; ++m) {
                const uint32_t sm = __ballot_sync(0xffffffffu, active[m] && d[m] < 0);
                if (lane < 4) base[4u*m + lane] = static_cast<uint8_t>((sm >> (8u*lane)) & 0xFFu);
            }
            for (int p = 0; p < r; ++p) {
                #pragma unroll
                for (int m = 0; m < EPL; ++m) {
                    const uint32_t pm = __ballot_sync(0xffffffffu, active[m] && ((av[m] >> p) & 1u));
                    if (lane < 4)
                        base[word_bytes*(1u+p) + 4u*m + lane] = static_cast<uint8_t>((pm >> (8u*lane)) & 0xFFu);
                }
            }
            return;
        }
        // Outlier block: [ob_bytes elem0 magnitude LE][sign region][r planes for elems 1..].
        const uint32_t ob_bytes = ((sel >> 1) & 3u) + 1u;
        if (lane == 0) {
            const uint32_t mag0 = absU_i32(d[0]);
            for (uint32_t k = 0; k < ob_bytes; ++k)
                base[k] = static_cast<uint8_t>((mag0 >> (8u*k)) & 0xffu);
        }
        uint8_t* sign   = base + ob_bytes;
        uint8_t* planes = base + ob_bytes + word_bytes;
        #pragma unroll
        for (int m = 0; m < EPL; ++m) {
            const uint32_t sm = __ballot_sync(0xffffffffu, active[m] && d[m] < 0);
            if (lane < 4) sign[4u*m + lane] = static_cast<uint8_t>((sm >> (8u*lane)) & 0xFFu);
        }
        for (int p = 0; p < r; ++p) {
            #pragma unroll
            for (int m = 0; m < EPL; ++m) {
                const bool plane_active = active[m] && (static_cast<size_t>(lane) + 32u*m) > 0;
                const uint32_t pm = __ballot_sync(0xffffffffu, plane_active && ((av[m] >> p) & 1u));
                if (lane < 4)
                    planes[word_bytes*p + 4u*m + lane] = static_cast<uint8_t>((pm >> (8u*lane)) & 0xFFu);
            }
        }
    }

    // INVERSE: decode a plain- OR outlier-mode block into `d`. Mirrors
    // decode_unpack_outlier_kernel. meta = [rate][selector]; selector bit0 =
    // outlier, bits1-2 = ob_bytes-1. Plain path == PlainRateCoder::decode; the
    // outlier path pulls element 0's magnitude from the raw leading bytes and
    // reads the r planes after the ob_bytes + sign region.
    template<int EPL>
    __device__ static void decode(const uint8_t* __restrict__ meta_b,
                                  const uint8_t* __restrict__ base,
                                  uint32_t word_bytes, size_t count,
                                  uint32_t lane, int (&d)[EPL]) {
        const int r = meta_b[0];
        const uint8_t sel = meta_b[1];
        const bool is_out = (sel & 1u) != 0;
        const uint32_t q = lane >> 3, j = lane & 7u;
        if (!is_out) {
            #pragma unroll
            for (int m = 0; m < EPL; ++m) {
                const size_t idx = static_cast<size_t>(lane) + 32u * m;
                if (idx >= count || r == 0) { d[m] = 0; continue; }
                const uint32_t k = 4u * m + q;
                const uint8_t sgn = base[k];
                uint32_t av = 0;
                for (int p = 0; p < r; ++p)
                    av |= ((static_cast<uint32_t>(base[word_bytes*(1u+p) + k]) >> j) & 1u) << p;
                d[m] = ((sgn >> j) & 1u) ? static_cast<int>(0u - av) : static_cast<int>(av);
            }
            return;
        }
        const uint32_t ob_bytes = ((sel >> 1) & 3u) + 1u;
        const uint8_t* sign   = base + ob_bytes;
        const uint8_t* planes = base + ob_bytes + word_bytes;
        #pragma unroll
        for (int m = 0; m < EPL; ++m) {
            const size_t idx = static_cast<size_t>(lane) + 32u * m;
            if (idx >= count) { d[m] = 0; continue; }
            if (idx == 0) {
                uint32_t mag0 = 0;
                for (uint32_t k = 0; k < ob_bytes; ++k)
                    mag0 |= static_cast<uint32_t>(base[k]) << (8u * k);
                d[m] = (sign[0] & 1u) ? static_cast<int>(0u - mag0) : static_cast<int>(mag0);
                continue;
            }
            const uint32_t k = 4u * m + q;
            const uint8_t sgn = sign[k];
            uint32_t av = 0;
            for (int p = 0; p < r; ++p)
                av |= ((static_cast<uint32_t>(planes[word_bytes*p + k]) >> j) & 1u) << p;
            d[m] = ((sgn >> j) & 1u) ? static_cast<int>(0u - av) : static_cast<int>(av);
        }
    }
};

// PlainBitpack: AdaptiveBitpack with the outlier mode disabled — every block is packed
// plain (fixed rate over ALL elements, no raw element-0 escape). Its output is exactly
// an AdaptiveBitpack archive whose blocks all select plain mode (selector byte 0), so
// the existing AdaptiveBitpack inverse decodes it unchanged. It never beats
// AdaptiveBitpack on size (AB picks the per-block min), which makes it the honest
// baseline coder for A/B-ing the swappable-coder path. The demonstrator for "swap more
// than the predictor": a different Cooperative op composed into the same warp chain.
struct PlainBitpackCoder {
    static constexpr uint32_t meta_bytes = 2;   // [rate][0] — always plain, AB-decodable

    template<int EPL>
    __device__ static void cost(const int (&d)[EPL], uint32_t lane, uint32_t word_bytes,
                                size_t count, uint8_t* __restrict__ meta_b,
                                uint32_t* __restrict__ cost_b) {
        uint32_t acc_all = 0;
        #pragma unroll
        for (int m = 0; m < EPL; ++m) {
            const size_t idx = static_cast<size_t>(lane) + 32u * m;
            acc_all |= (idx < count) ? absU_i32(d[m]) : 0u;
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            acc_all |= __shfl_xor_sync(0xffffffffu, acc_all, off);
        if (lane == 0) {
            const int fr = bitWidth32(acc_all);
            meta_b[0] = static_cast<uint8_t>(fr); meta_b[1] = 0;
            *cost_b = (fr > 0) ? word_bytes * (fr + 1u) : 0u;
        }
    }

    template<int EPL>
    __device__ static void pack(const int (&d)[EPL], uint32_t lane, uint32_t word_bytes,
                                size_t count, const uint8_t* __restrict__ meta_b,
                                uint8_t* __restrict__ base) {
        const int r = meta_b[0];
        if (r == 0) return;
        uint32_t av[EPL]; bool active[EPL];
        #pragma unroll
        for (int m = 0; m < EPL; ++m) {
            const size_t idx = static_cast<size_t>(lane) + 32u * m;
            active[m] = idx < count;
            av[m]     = active[m] ? absU_i32(d[m]) : 0u;
        }
        #pragma unroll
        for (int m = 0; m < EPL; ++m) {
            const uint32_t sm = __ballot_sync(0xffffffffu, active[m] && d[m] < 0);
            if (lane < 4) base[4u*m + lane] = static_cast<uint8_t>((sm >> (8u*lane)) & 0xFFu);
        }
        for (int p = 0; p < r; ++p) {
            #pragma unroll
            for (int m = 0; m < EPL; ++m) {
                const uint32_t pm = __ballot_sync(0xffffffffu, active[m] && ((av[m] >> p) & 1u));
                if (lane < 4)
                    base[word_bytes*(1u+p) + 4u*m + lane] = static_cast<uint8_t>((pm >> (8u*lane)) & 0xFFu);
            }
        }
    }
};

// PlainRateCoder: the TRUE staged plain-mode AdaptiveBitpack format — a single
// rate byte per block (meta_bytes == 1, no selector byte), payload = sign region
// + `rate` bit-planes. Identical cost + payload layout to PlainBitpackCoder; the
// ONLY difference is the 1-byte meta stride, which matches AdaptiveBitpackStage
// with outlier_selection == false (its inverse reads a 1-byte rate region). This
// is the coder SZp's composed chain (Quantizer -> Lorenzo -> AdaptiveBitpack,
// plain) fuses through, byte-identical to the staged plain path.
struct PlainRateCoder {
    static constexpr uint32_t meta_bytes = 1;   // [rate]

    template<int EPL>
    __device__ static void cost(const int (&d)[EPL], uint32_t lane, uint32_t word_bytes,
                                size_t count, uint8_t* __restrict__ meta_b,
                                uint32_t* __restrict__ cost_b) {
        uint32_t acc_all = 0;
        #pragma unroll
        for (int m = 0; m < EPL; ++m) {
            const size_t idx = static_cast<size_t>(lane) + 32u * m;
            acc_all |= (idx < count) ? absU_i32(d[m]) : 0u;
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            acc_all |= __shfl_xor_sync(0xffffffffu, acc_all, off);
        if (lane == 0) {
            const int fr = bitWidth32(acc_all);
            meta_b[0] = static_cast<uint8_t>(fr);
            *cost_b = (fr > 0) ? word_bytes * (fr + 1u) : 0u;
        }
    }

    template<int EPL>
    __device__ static void pack(const int (&d)[EPL], uint32_t lane, uint32_t word_bytes,
                                size_t count, const uint8_t* __restrict__ meta_b,
                                uint8_t* __restrict__ base) {
        const int r = meta_b[0];
        if (r == 0) return;
        uint32_t av[EPL]; bool active[EPL];
        #pragma unroll
        for (int m = 0; m < EPL; ++m) {
            const size_t idx = static_cast<size_t>(lane) + 32u * m;
            active[m] = idx < count;
            av[m]     = active[m] ? absU_i32(d[m]) : 0u;
        }
        #pragma unroll
        for (int m = 0; m < EPL; ++m) {
            const uint32_t sm = __ballot_sync(0xffffffffu, active[m] && d[m] < 0);
            if (lane < 4) base[4u*m + lane] = static_cast<uint8_t>((sm >> (8u*lane)) & 0xFFu);
        }
        for (int p = 0; p < r; ++p) {
            #pragma unroll
            for (int m = 0; m < EPL; ++m) {
                const uint32_t pm = __ballot_sync(0xffffffffu, active[m] && ((av[m] >> p) & 1u));
                if (lane < 4)
                    base[word_bytes*(1u+p) + 4u*m + lane] = static_cast<uint8_t>((pm >> (8u*lane)) & 0xFFu);
            }
        }
    }

    // INVERSE: decode block `base` (payload at the block's scanned offset) into
    // the per-lane delta array `d`. Mirrors decode_unpack_kernel_warp's O(rate)
    // gather: lane l reads byte group q = l>>3, bit j = l&7 of each of the `r`
    // plane words + the sign word. `r` comes from the 1-byte meta.
    template<int EPL>
    __device__ static void decode(const uint8_t* __restrict__ meta_b,
                                  const uint8_t* __restrict__ base,
                                  uint32_t word_bytes, size_t count,
                                  uint32_t lane, int (&d)[EPL]) {
        const int r = meta_b[0];
        const uint32_t q = lane >> 3, j = lane & 7u;
        #pragma unroll
        for (int m = 0; m < EPL; ++m) {
            const size_t idx = static_cast<size_t>(lane) + 32u * m;
            if (idx >= count || r == 0) { d[m] = 0; continue; }
            const uint32_t k = 4u * m + q;
            const uint8_t sgn = base[k];
            uint32_t av = 0;
            for (int p = 0; p < r; ++p)
                av |= ((static_cast<uint32_t>(base[word_bytes*(1u+p) + k]) >> j) & 1u) << p;
            // two's-complement negate via unsigned wrap — matches applySign<int32_t>
            d[m] = ((sgn >> j) & 1u) ? static_cast<int>(0u - av) : static_cast<int>(av);
        }
    }
};

// ── Harness bodies (predictor- AND coder-agnostic) ───────────────────────────
// Factored out of the __global__ entries so the NVRTC path can wrap them in an
// extern "C" kernel; the template path below calls them too. The predictor produces
// the per-lane deltas `d[EPL]`; the coder consumes them. Both passes recompute the
// deltas (cheaper than spilling them to DRAM — the reason warp fusion is fast).
// Template order `<EPL, Coder, Pred>` lets the codegen name EPL+Coder explicitly and
// deduce Pred from the argument.
template<int ElemsPerLane, class Coder, class Pred, class... Transforms>
__device__ __forceinline__ void fused_rate_body(
    Pred pred, size_t n, uint32_t word_bytes, size_t num_blocks,
    uint8_t* __restrict__ meta, uint32_t* __restrict__ cost)
{
    const uint32_t lane          = threadIdx.x & 31u;
    const uint32_t warp_in_block = threadIdx.x >> 5;
    const uint32_t warps_per_cta = blockDim.x >> 5;
    const size_t b = static_cast<size_t>(blockIdx.x) * warps_per_cta + warp_in_block;
    if (b >= num_blocks) return;

    constexpr uint32_t block_size = static_cast<uint32_t>(ElemsPerLane) * 32u;
    const size_t start = b * block_size;
    const size_t count = min(static_cast<size_t>(block_size), n - start);

    int d[ElemsPerLane];
    #pragma unroll
    for (int m = 0; m < ElemsPerLane; ++m) d[m] = pred.delta(lane, b, m);
    applyTransforms<ElemsPerLane, Transforms...>(d, lane);
    Coder::template cost<ElemsPerLane>(d, lane, word_bytes, count,
                                       meta + Coder::meta_bytes * b, &cost[b]);
}

template<int ElemsPerLane, class Coder, class Pred, class... Transforms>
__device__ __forceinline__ void fused_pack_body(
    Pred pred, size_t n, uint32_t word_bytes, size_t num_blocks,
    const uint8_t* __restrict__ meta, const uint32_t* __restrict__ offset,
    uint8_t* __restrict__ payload)
{
    const uint32_t lane          = threadIdx.x & 31u;
    const uint32_t warp_in_block = threadIdx.x >> 5;
    const uint32_t warps_per_cta = blockDim.x >> 5;
    const size_t b = static_cast<size_t>(blockIdx.x) * warps_per_cta + warp_in_block;
    if (b >= num_blocks) return;

    constexpr uint32_t block_size = static_cast<uint32_t>(ElemsPerLane) * 32u;
    const size_t start = b * block_size;
    const size_t count = min(static_cast<size_t>(block_size), n - start);

    int d[ElemsPerLane];
    #pragma unroll
    for (int m = 0; m < ElemsPerLane; ++m) d[m] = pred.delta(lane, b, m);
    applyTransforms<ElemsPerLane, Transforms...>(d, lane);
    Coder::template pack<ElemsPerLane>(d, lane, word_bytes, count,
                                       meta + Coder::meta_bytes * b, payload + offset[b]);
}

// ── Single-pass decoupled-lookback path ─────────────────────────────────────
// The two-pass path (rate → host CUB scan → pack) recomputes the predictor delta
// in BOTH kernels because the device-wide byte-offset prefix-sum forces a launch
// boundary, and register-resident deltas don't survive it. This path collapses
// cost + scan + pack into ONE kernel: each warp claims a logical tile via an
// atomic counter (so logical order is decoupled from block scheduling → forward
// progress), computes its deltas ONCE, publishes its byte cost as an aggregate,
// derives its own exclusive byte offset by a warp-windowed look-back over
// predecessor tiles, then packs from the still-live registers. Same policies,
// same costs/offsets/packing as two-pass ⇒ byte-identical (reuses staged inverse).
// This is cuSZp's single-pass design, reconstructed over our modular ops.
//
// Status protocol (per tile, three global arrays, device-scope fences): a tile
// first publishes g_agg then flips g_state to AGG(1); after its look-back it
// publishes g_incl then flips g_state to PREFIX(2). Separate agg/incl slots avoid
// any overwrite race on a value a peer may still be reading. A reader spins on
// g_state, fences, then reads incl (if PREFIX) or agg (if AGG). p<0 lanes act as
// PREFIX-0 so the window terminates cleanly at tile 0.
enum : uint32_t { LB_NONE = 0u, LB_AGG = 1u, LB_PREFIX = 2u };

__device__ __forceinline__ uint32_t warp_decoupled_lookback(
    uint32_t tile, volatile uint32_t* __restrict__ g_state,
    volatile uint32_t* __restrict__ g_agg, volatile uint32_t* __restrict__ g_incl,
    uint32_t lane)
{
    uint32_t excl = 0u;
    int look = static_cast<int>(tile) - 1;   // highest predecessor index
    while (look >= 0) {
        const int p = look - static_cast<int>(lane);   // lane 0 = highest p in window
        uint32_t st, vv;
        if (p >= 0) {
            while ((st = g_state[p]) == LB_NONE) { /* spin: predecessor not ready */ }
            __threadfence();
            vv = (st == LB_PREFIX) ? g_incl[p] : g_agg[p];
        } else {
            st = LB_PREFIX;   // out-of-range predecessor contributes an inclusive 0
            vv = 0u;
        }
        const uint32_t prefix_ballot = __ballot_sync(0xffffffffu, st == LB_PREFIX);
        uint32_t add;
        bool done;
        if (prefix_ballot) {
            const int fpl = __ffs(prefix_ballot) - 1;   // lowest lane w/ PREFIX = highest such p
            add  = (lane <= static_cast<uint32_t>(fpl)) ? vv : 0u;  // aggs above it + its inclusive
            done = true;
        } else {
            add  = vv;      // whole window is AGG — sum it and slide down
            done = false;
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            add += __shfl_xor_sync(0xffffffffu, add, off);
        excl += add;
        if (done) break;
        look -= 32;
    }
    return excl;   // identical on all lanes
}

// Single kernel, cuSZp-style COARSE WARP granularity. CTA = 1 warp; each warp OWNS a run
// of BlocksPerWarp consecutive blocks (BlocksPerWarp*32*EPL elements), computes+holds all
// their deltas ONCE (no recompute, no 2nd input read), sums their byte costs, does ONE
// warp-cooperative decoupled look-back over the per-WARP aggregate (scan elements =
// num_blocks/BlocksPerWarp — coarse, so the look-back latency is hidden behind the warp's
// large work), then packs every held block at base+intra-warp offset. This is the port of
// cuSZp2/cuSZp3's fused kernel structure into our generic policy harness: the predictor/
// coder stay swappable, only the scan/granularity is fixed. Byte-identical to two-pass.
// (num_warps = grid; g_counter unused here — tile id is blockIdx, forward progress holds
// because each warp publishes its aggregate before its own look-back, cuSZp-style.)
template<int ElemsPerLane, int BlocksPerWarp, class Coder, class Pred, class... Transforms>
__device__ __forceinline__ void fused_single_pass_body(
    Pred pred, size_t n, uint32_t word_bytes, size_t num_blocks,
    uint8_t* __restrict__ meta, uint8_t* __restrict__ payload,
    uint32_t* __restrict__ /*g_counter*/, uint32_t* __restrict__ g_state,
    uint32_t* __restrict__ g_agg, uint32_t* __restrict__ g_incl, size_t num_warps)
{
    __shared__ uint32_t s_base[BlocksPerWarp];   // exclusive byte offset of each block in the warp

    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t w    = blockIdx.x;            // CTA = 1 warp ⇒ global warp id = blockIdx
    if (static_cast<size_t>(w) >= num_warps) return;

    const size_t   b0         = static_cast<size_t>(w) * BlocksPerWarp;
    constexpr uint32_t block_size = static_cast<uint32_t>(ElemsPerLane) * 32u;

    // ── Phase 1: compute + hold deltas for all my blocks; sum warp cost; block offsets.
    int      dheld[BlocksPerWarp][ElemsPerLane];   // held across the look-back (local mem)
    uint32_t warp_total = 0u;
    #pragma unroll 1
    for (int jb = 0; jb < BlocksPerWarp; ++jb) {
        const size_t b = b0 + static_cast<size_t>(jb);
        uint32_t c = 0u;
        if (b < num_blocks) {
            const size_t start = b * block_size;
            const size_t count = min(static_cast<size_t>(block_size), n - start);
            #pragma unroll
            for (int m = 0; m < ElemsPerLane; ++m) dheld[jb][m] = pred.delta(lane, b, m);
            applyTransforms<ElemsPerLane, Transforms...>(dheld[jb], lane);
            Coder::template cost<ElemsPerLane>(dheld[jb], lane, word_bytes, count,
                                               meta + Coder::meta_bytes * b, &c);
            c = __shfl_sync(0xffffffffu, c, 0);
        }
        if (lane == 0u) s_base[jb] = warp_total;   // exclusive within this warp
        warp_total += c;
    }
    __syncwarp();

    // ── Phase 2: warp-cooperative decoupled look-back over per-warp aggregates → base.
    if (lane == 0u) { g_agg[w] = warp_total; __threadfence(); g_state[w] = LB_AGG; }
    const uint32_t base = warp_decoupled_lookback(w,
        reinterpret_cast<volatile uint32_t*>(g_state),
        reinterpret_cast<volatile uint32_t*>(g_agg),
        reinterpret_cast<volatile uint32_t*>(g_incl), lane);
    if (lane == 0u) { g_incl[w] = base + warp_total; __threadfence(); g_state[w] = LB_PREFIX; }

    // ── Phase 3: pack every held block at its resolved offset (registers, no recompute).
    #pragma unroll 1
    for (int jb = 0; jb < BlocksPerWarp; ++jb) {
        const size_t b = b0 + static_cast<size_t>(jb);
        if (b < num_blocks) {
            const size_t start = b * block_size;
            const size_t count = min(static_cast<size_t>(block_size), n - start);
            Coder::template pack<ElemsPerLane>(dheld[jb], lane, word_bytes, count,
                                               meta + Coder::meta_bytes * b,
                                               payload + base + s_base[jb]);
        }
    }
}

// ── Inverse (decompress) harness ────────────────────────────────────────────
// One warp owns one block. Reverses the forward chain, register-resident: the
// coder decodes the block's payload into the per-lane delta array `d`, transforms
// are undone in reverse, the predictor prefix-sums the deltas back to quant codes
// (block-reset), and the codes are dequantized to floats. The per-block payload
// offsets come from a preceding cost pass + CUB scan (host-orchestrated, like the
// forward two-pass) so this kernel never syncs. Bit-exact vs the staged
// AB-decode -> Lorenzo-prefix-sum -> linear-dequant kernels.
template<int ElemsPerLane, class Coder, class Pred, class... Transforms>
__device__ __forceinline__ void fused_unpack_body(
    size_t n, uint32_t word_bytes, size_t num_blocks, float ebx2,
    const uint8_t* __restrict__ meta, const uint32_t* __restrict__ offset,
    const uint8_t* __restrict__ payload, float* __restrict__ out)
{
    const uint32_t lane          = threadIdx.x & 31u;
    const uint32_t warp_in_block = threadIdx.x >> 5;
    const uint32_t warps_per_cta = blockDim.x >> 5;
    const size_t b = static_cast<size_t>(blockIdx.x) * warps_per_cta + warp_in_block;
    if (b >= num_blocks) return;

    constexpr uint32_t block_size = static_cast<uint32_t>(ElemsPerLane) * 32u;
    const size_t start = b * block_size;
    const size_t count = min(static_cast<size_t>(block_size), n - start);

    int d[ElemsPerLane];
    Coder::template decode<ElemsPerLane>(meta + Coder::meta_bytes * b,
                                         payload + offset[b], word_bytes, count, lane, d);
    // Transforms are size-preserving register maps; the forward applied
    // Transforms... in order after the predictor, so the inverse chain undoes
    // them before the predictor. (No inverse-transform ops registered yet — the
    // pack is empty for the cuSZp/SZp chains.)
    Pred::template undelta<ElemsPerLane>(d, lane);
    #pragma unroll
    for (int m = 0; m < ElemsPerLane; ++m) {
        const size_t idx = static_cast<size_t>(lane) + 32u * m;
        if (idx < count) out[start + idx] = static_cast<float>(d[m]) * ebx2;
    }
}

// ── Compile-time template kernels (the non-NVRTC path). The NVRTC path generates
// equivalent extern "C" kernels over the same bodies. ────────────────────────
template<int ElemsPerLane, class Coder, class Pred, class... Transforms>
__global__ void fused_rate_kernel(
    Pred pred, size_t n, uint32_t word_bytes, size_t num_blocks,
    uint8_t* __restrict__ meta, uint32_t* __restrict__ cost)
{
    fused_rate_body<ElemsPerLane, Coder, Pred, Transforms...>(pred, n, word_bytes, num_blocks, meta, cost);
}

template<int ElemsPerLane, class Coder, class Pred, class... Transforms>
__global__ void fused_unpack_kernel(
    size_t n, uint32_t word_bytes, size_t num_blocks, float ebx2,
    const uint8_t* __restrict__ meta, const uint32_t* __restrict__ offset,
    const uint8_t* __restrict__ payload, float* __restrict__ out)
{
    fused_unpack_body<ElemsPerLane, Coder, Pred, Transforms...>(
        n, word_bytes, num_blocks, ebx2, meta, offset, payload, out);
}

template<int ElemsPerLane, class Coder, class Pred, class... Transforms>
__global__ void fused_pack_kernel(
    Pred pred, size_t n, uint32_t word_bytes, size_t num_blocks,
    const uint8_t* __restrict__ meta, const uint32_t* __restrict__ offset,
    uint8_t* __restrict__ payload)
{
    fused_pack_body<ElemsPerLane, Coder, Pred, Transforms...>(pred, n, word_bytes, num_blocks, meta, offset, payload);
}

} // namespace warp
} // namespace fused
} // namespace fz
