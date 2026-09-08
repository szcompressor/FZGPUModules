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
        if (base + 32u <= n) {
            // Vectorized 128-bit loads (the block is 32 contiguous floats, in+base is 16 B
            // aligned since base%4==0). Fewer load instructions + better MLP than 32 scalars;
            // same float values ⇒ byte-identical codes.
            const float4* in4 = reinterpret_cast<const float4*>(in + base);
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float4 v = in4[j];
                int q;
                q = __float2int_rn(v.x * inv2eb); d[j*4+0] = q - prev; prev = q;
                q = __float2int_rn(v.y * inv2eb); d[j*4+1] = q - prev; prev = q;
                q = __float2int_rn(v.z * inv2eb); d[j*4+2] = q - prev; prev = q;
                q = __float2int_rn(v.w * inv2eb); d[j*4+3] = q - prev; prev = q;
            }
        } else {   // partial tail block
            #pragma unroll
            for (int i = 0; i < 32; ++i) {
                const size_t g = base + static_cast<size_t>(i);
                const int q = (g < n) ? __float2int_rn(in[g] * inv2eb) : 0;
                d[i] = q - prev;
                prev = q;
            }
        }
    }
};

// ── Tiled predictors (cuSZp3-shaped): one thread owns a whole 64-element TILE ─
// (8x8 in 2-D, 4x4x4 in 3-D — the same tile geometry as TiledLorenzo{2D,3D}Predictor
// in warp_fusion.cuh), traversed serially inside the thread. Block index `base/64`
// IS the tile index (block_size == tile_elems by construction, see tiled_lorenzo_stage.h).
//
// LOAD-ONCE: the warp-cooperative TiledLorenzo{2D,3D}Predictor re-reads and re-quantizes
// each element's neighbour from global (`in[gidx-1]` etc.), because different LANES own
// different elements and cannot share registers. Here ONE thread owns the WHOLE tile, so
// the neighbour is simply the previous LOOP ITERATION's already-computed `cur` — a
// register read, not a second global load + requantize. This is exactly what native
// cuSZp3 does per-thread (see cuSZp_kernels_3D_f32.cu's prevQuant_{x,y,z} chain) and it
// is mathematically IDENTICAL to the warp-cooperative delta() (same deterministic
// __float2int_rn on the same float value), so it is byte-identical by construction —
// no new inverse needed; this predictor pairs with the SAME ThreadFixedRateCoderN<64>
// coder below to emit the identical AdaptiveBitpack archive the warp-cooperative /
// staged tiled paths produce, which the existing tiled inverse already decodes.
//
// Traversal order is z outer, y middle, x inner (matching the tile's fast-x layout),
// with three chained registers:
//   prevX — value at (lx-1,ly,lz): updated every element, consumed when lx>0
//   prevY — value at (0,ly-1,lz): updated only when lx==0, consumed when lx==0,ly>0
//   prevZ — value at (0,0,lz-1):  updated only when lx==0&&ly==0, consumed at the tile's
//           leading (0,0,lz>0) edge
// Once one axis goes out of range (padding), every later element on that axis is also
// out of range (dims are monotonic in tile-local coordinates), so an update skipped for
// a padding element is never consulted by a later valid one.
struct ThreadTiledLorenzo2DPredictor {
    const float* in;
    float inv2eb;
    uint32_t dx, dy, tx, ty, ntx;
    __device__ static ThreadTiledLorenzo2DPredictor fromParams(const float* in, size_t /*n*/, const void* pp) {
        const warp::TiledLorenzo2DParams p = *static_cast<const warp::TiledLorenzo2DParams*>(pp);
        return ThreadTiledLorenzo2DPredictor{in, p.inv2eb, p.dx, p.dy, p.tx, p.ty, p.ntx};
    }
    __device__ __forceinline__ void predict(size_t base, int (&d)[64]) const {
        const uint32_t t   = static_cast<uint32_t>(base / 64u);   // block index == tile index
        const uint32_t tix = t % ntx, tiy = t / ntx;
        int prevY = 0;
        for (uint32_t ly = 0; ly < ty; ++ly) {
            const uint32_t gy = tiy * ty + ly;
            int prevX = 0;
            for (uint32_t lx = 0; lx < tx; ++lx) {
                const uint32_t local = ly * tx + lx;
                const uint32_t gx = tix * tx + lx;
                if (gx >= dx || gy >= dy) { d[local] = 0; continue; }
                const int cur = __float2int_rn(in[static_cast<size_t>(gy) * dx + gx] * inv2eb);
                const int pred = (lx > 0) ? prevX : (ly > 0) ? prevY : 0;
                d[local] = cur - pred;
                prevX = cur;
                if (lx == 0) prevY = cur;
            }
        }
    }
};

struct ThreadTiledLorenzo3DPredictor {
    const float* in;
    float inv2eb;
    uint32_t dx, dy, dz, tx, ty, tz, ntx, nty;
    __device__ static ThreadTiledLorenzo3DPredictor fromParams(const float* in, size_t /*n*/, const void* pp) {
        const warp::TiledLorenzo3DParams p = *static_cast<const warp::TiledLorenzo3DParams*>(pp);
        return ThreadTiledLorenzo3DPredictor{in, p.inv2eb, p.dx, p.dy, p.dz, p.tx, p.ty, p.tz, p.ntx, p.nty};
    }
    __device__ __forceinline__ void predict(size_t base, int (&d)[64]) const {
        const uint32_t t   = static_cast<uint32_t>(base / 64u);
        const uint32_t tix = t % ntx, tiy = (t / ntx) % nty, tiz = t / (ntx * nty);
        int prevZ = 0;
        for (uint32_t lz = 0; lz < tz; ++lz) {
            const uint32_t gz = tiz * tz + lz;
            int prevY = 0;
            for (uint32_t ly = 0; ly < ty; ++ly) {
                const uint32_t gy = tiy * ty + ly;
                int prevX = 0;
                for (uint32_t lx = 0; lx < tx; ++lx) {
                    const uint32_t local = (lz * ty + ly) * tx + lx;
                    const uint32_t gx = tix * tx + lx;
                    if (gx >= dx || gy >= dy || gz >= dz) { d[local] = 0; continue; }
                    const size_t gidx = (static_cast<size_t>(gz) * dy + gy) * dx + gx;
                    const int cur = __float2int_rn(in[gidx] * inv2eb);
                    const int pred = (lx > 0) ? prevX : (ly > 0) ? prevY : (lz > 0) ? prevZ : 0;
                    d[local] = cur - pred;
                    prevX = cur;
                    if (lx == 0) { prevY = cur; if (ly == 0) prevZ = cur; }
                }
            }
        }
    }
};

// ── Identity tiled predictors (cuSZp3 FIXED mode): same tile geometry, no delta —
// d[local] = cur. Genuinely load-once by construction (no chain state at all). ─────
struct ThreadTiledLorenzoIdentity2DPredictor {
    const float* in;
    float inv2eb;
    uint32_t dx, dy, tx, ty, ntx;
    __device__ static ThreadTiledLorenzoIdentity2DPredictor fromParams(const float* in, size_t /*n*/, const void* pp) {
        const warp::TiledLorenzo2DParams p = *static_cast<const warp::TiledLorenzo2DParams*>(pp);
        return ThreadTiledLorenzoIdentity2DPredictor{in, p.inv2eb, p.dx, p.dy, p.tx, p.ty, p.ntx};
    }
    __device__ __forceinline__ void predict(size_t base, int (&d)[64]) const {
        const uint32_t t   = static_cast<uint32_t>(base / 64u);
        const uint32_t tix = t % ntx, tiy = t / ntx;
        for (uint32_t ly = 0; ly < ty; ++ly) {
            const uint32_t gy = tiy * ty + ly;
            for (uint32_t lx = 0; lx < tx; ++lx) {
                const uint32_t local = ly * tx + lx;
                const uint32_t gx = tix * tx + lx;
                d[local] = (gx >= dx || gy >= dy) ? 0
                    : __float2int_rn(in[static_cast<size_t>(gy) * dx + gx] * inv2eb);
            }
        }
    }
};

struct ThreadTiledLorenzoIdentity3DPredictor {
    const float* in;
    float inv2eb;
    uint32_t dx, dy, dz, tx, ty, tz, ntx, nty;
    __device__ static ThreadTiledLorenzoIdentity3DPredictor fromParams(const float* in, size_t /*n*/, const void* pp) {
        const warp::TiledLorenzo3DParams p = *static_cast<const warp::TiledLorenzo3DParams*>(pp);
        return ThreadTiledLorenzoIdentity3DPredictor{in, p.inv2eb, p.dx, p.dy, p.dz, p.tx, p.ty, p.tz, p.ntx, p.nty};
    }
    __device__ __forceinline__ void predict(size_t base, int (&d)[64]) const {
        const uint32_t t   = static_cast<uint32_t>(base / 64u);
        const uint32_t tix = t % ntx, tiy = (t / ntx) % nty, tiz = t / (ntx * nty);
        for (uint32_t lz = 0; lz < tz; ++lz) {
            const uint32_t gz = tiz * tz + lz;
            for (uint32_t ly = 0; ly < ty; ++ly) {
                const uint32_t gy = tiy * ty + ly;
                for (uint32_t lx = 0; lx < tx; ++lx) {
                    const uint32_t local = (lz * ty + ly) * tx + lx;
                    const uint32_t gx = tix * tx + lx;
                    if (gx >= dx || gy >= dy || gz >= dz) { d[local] = 0; continue; }
                    const size_t gidx = (static_cast<size_t>(gz) * dy + gy) * dx + gx;
                    d[local] = __float2int_rn(in[gidx] * inv2eb);
                }
            }
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

// Thread-independent AdaptiveBitpack, generalized to N elements (N a multiple of 32; the
// warp-cooperative side supports EPL up to kMaxWarpElemsPerLane==4, i.e. N up to 128 — this
// coder generalizes the same way). One thread owns all N block codes, so the warp-cooperative
// ballot/shfl of AdaptiveBitpackCoder becomes NW=N/32 serial in-register 32-bit sub-words per
// sign/plane region — byte-for-byte the SAME layout AdaptiveBitpackCoder::pack<EPL> emits
// (each region is NW consecutive 4-byte LE words, one per group of 32 elements), so the
// staged/warp-cooperative inverse decodes either producer's archive identically.
// N=32 (NW=1) is byte-for-byte what the original hand-written ThreadFixedRateCoder emitted.
template<int N>
struct ThreadFixedRateCoderN {
    static_assert(N % 32 == 0 && N <= 128, "ThreadFixedRateCoderN: N must be a multiple of 32, up to 128");
    static constexpr uint32_t meta_bytes = 2;
    static constexpr int NW = N / 32;

    // Writes meta[0..1] and returns this block's payload byte length. Mirrors
    // AdaptiveBitpackCoder::cost<EPL>: plain (fixed-rate over all) vs outlier (elem0 raw + rate
    // over the rest), whichever is cheaper.
    __device__ static __forceinline__ uint32_t cost(const int (&d)[N], uint32_t word_bytes,
                                                    uint32_t count, uint8_t* __restrict__ meta) {
        uint32_t acc_all = 0u, acc_rest = 0u;
        #pragma unroll
        for (uint32_t i = 0; i < static_cast<uint32_t>(N); ++i) {
            const uint32_t av = (i < count) ? warp::absU_i32(d[i]) : 0u;
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

    // Writes the payload. Mirrors AdaptiveBitpackCoder::pack<EPL> byte-for-byte: sign region
    // (NW 4-byte LE sub-words, bit j of sub-word m = element 32*m+j) then r bit-plane regions
    // (same NW-sub-word shape each); outlier prepends elem0's magnitude and drops elem0 from
    // the planes.
    __device__ static __forceinline__ void pack(const int (&d)[N], uint32_t word_bytes,
                                                uint32_t count, const uint8_t* __restrict__ meta,
                                                uint8_t* __restrict__ out) {
        const int     r      = meta[0];
        const uint8_t sel    = meta[1];
        const bool    is_out = (sel & 1u) != 0;

        if (!is_out) {
            if (r == 0) return;
            #pragma unroll
            for (int m = 0; m < NW; ++m) {
                uint32_t sm = 0u;
                #pragma unroll
                for (int j = 0; j < 32; ++j) {
                    const uint32_t i = static_cast<uint32_t>(m * 32 + j);
                    if (i < count && d[i] < 0) sm |= (1u << j);
                }
                #pragma unroll
                for (uint32_t k = 0; k < 4u; ++k)
                    out[4u * m + k] = static_cast<uint8_t>((sm >> (8u * k)) & 0xFFu);
            }
            for (int p = 0; p < r; ++p) {
                #pragma unroll
                for (int m = 0; m < NW; ++m) {
                    uint32_t pm = 0u;
                    #pragma unroll
                    for (int j = 0; j < 32; ++j) {
                        const uint32_t i = static_cast<uint32_t>(m * 32 + j);
                        if (i < count && ((warp::absU_i32(d[i]) >> p) & 1u)) pm |= (1u << j);
                    }
                    #pragma unroll
                    for (uint32_t k = 0; k < 4u; ++k)
                        out[word_bytes * (1u + p) + 4u * m + k] =
                            static_cast<uint8_t>((pm >> (8u * k)) & 0xFFu);
                }
            }
            return;
        }
        // Outlier: [ob_bytes elem0 magnitude LE][sign region][r plane regions for elems 1..].
        const uint32_t ob_bytes = ((sel >> 1) & 3u) + 1u;
        const uint32_t mag0     = (count > 0) ? warp::absU_i32(d[0]) : 0u;
        for (uint32_t k = 0; k < ob_bytes; ++k) out[k] = static_cast<uint8_t>((mag0 >> (8u * k)) & 0xFFu);
        uint8_t* sign   = out + ob_bytes;
        uint8_t* planes = out + ob_bytes + word_bytes;
        #pragma unroll
        for (int m = 0; m < NW; ++m) {
            uint32_t sm = 0u;
            #pragma unroll
            for (int j = 0; j < 32; ++j) {
                const uint32_t i = static_cast<uint32_t>(m * 32 + j);
                if (i < count && d[i] < 0) sm |= (1u << j);
            }
            #pragma unroll
            for (uint32_t k = 0; k < 4u; ++k)
                sign[4u * m + k] = static_cast<uint8_t>((sm >> (8u * k)) & 0xFFu);
        }
        for (int p = 0; p < r; ++p) {
            #pragma unroll
            for (int m = 0; m < NW; ++m) {
                uint32_t pm = 0u;
                #pragma unroll
                for (int j = 0; j < 32; ++j) {
                    const uint32_t i = static_cast<uint32_t>(m * 32 + j);
                    if (i > 0 && i < count && ((warp::absU_i32(d[i]) >> p) & 1u)) pm |= (1u << j);
                }
                #pragma unroll
                for (uint32_t k = 0; k < 4u; ++k)
                    planes[word_bytes * p + 4u * m + k] = static_cast<uint8_t>((pm >> (8u * k)) & 0xFFu);
            }
        }
    }
};

/// Back-compat alias: the original hand-written 32-element coder, now the N=32
/// instantiation of the generalized template (byte-identical output).
using ThreadFixedRateCoder = ThreadFixedRateCoderN<32>;

// Thread-independent PlainRateCoder: fixed-rate over ALL elements, no outlier escape.
// Mirrors warp::PlainRateCoder exactly (1-byte meta = [rate], sign region + r plane
// regions, no outlier branch) — this is what AdaptiveBitpackStage emits when
// outlier_selection is false (cuszp2/cuszp3 "plain" mode), so TI could not fuse those
// chains before this coder existed (tiSupportedChain required AdaptiveBitpackCoder
// specifically). Same NW=N/32 sub-word byte layout as ThreadFixedRateCoderN.
template<int N>
struct ThreadPlainRateCoderN {
    static_assert(N % 32 == 0 && N <= 128, "ThreadPlainRateCoderN: N must be a multiple of 32, up to 128");
    static constexpr uint32_t meta_bytes = 1;
    static constexpr int NW = N / 32;

    __device__ static __forceinline__ uint32_t cost(const int (&d)[N], uint32_t word_bytes,
                                                    uint32_t count, uint8_t* __restrict__ meta) {
        uint32_t acc_all = 0u;
        #pragma unroll
        for (uint32_t i = 0; i < static_cast<uint32_t>(N); ++i)
            acc_all |= (i < count) ? warp::absU_i32(d[i]) : 0u;
        const int fr = warp::bitWidth32(acc_all);
        meta[0] = static_cast<uint8_t>(fr);
        return (fr > 0) ? word_bytes * (fr + 1u) : 0u;
    }

    __device__ static __forceinline__ void pack(const int (&d)[N], uint32_t word_bytes,
                                                uint32_t count, const uint8_t* __restrict__ meta,
                                                uint8_t* __restrict__ out) {
        const int r = meta[0];
        if (r == 0) return;
        #pragma unroll
        for (int m = 0; m < NW; ++m) {
            uint32_t sm = 0u;
            #pragma unroll
            for (int j = 0; j < 32; ++j) {
                const uint32_t i = static_cast<uint32_t>(m * 32 + j);
                if (i < count && d[i] < 0) sm |= (1u << j);
            }
            #pragma unroll
            for (uint32_t k = 0; k < 4u; ++k)
                out[4u * m + k] = static_cast<uint8_t>((sm >> (8u * k)) & 0xFFu);
        }
        for (int p = 0; p < r; ++p) {
            #pragma unroll
            for (int m = 0; m < NW; ++m) {
                uint32_t pm = 0u;
                #pragma unroll
                for (int j = 0; j < 32; ++j) {
                    const uint32_t i = static_cast<uint32_t>(m * 32 + j);
                    if (i < count && ((warp::absU_i32(d[i]) >> p) & 1u)) pm |= (1u << j);
                }
                #pragma unroll
                for (uint32_t k = 0; k < 4u; ++k)
                    out[word_bytes * (1u + p) + 4u * m + k] =
                        static_cast<uint8_t>((pm >> (8u * k)) & 0xFFu);
            }
        }
    }
};

// ── Harness (Phase 3) ────────────────────────────────────────────────────────
// CTA = 1 warp. Thread `lane` owns block (jb,lane) = linear index warp_block_base + jb*32 + lane
// for jb in [0,BlocksPerThread). Holds all codes, computes LINEAR-order byte offsets with a 2-D
// warp prefix, gets the warp's base via the decoupled look-back, then packs. `meta`/`payload`
// point at the AdaptiveBitpack meta region and payload region of the archive.
//
// `BlockSize` (elements per block: 32 for Lorenzo1D, 64 for the tiled cuSZp3 predictors) is
// purely the size of each thread's held-codes array and the count/offset arithmetic below —
// the WARP still always has exactly 32 lanes, each owning a full BlockSize-element block, so
// the lane-dispatch and cross-lane prefix-sum (Phase B/C) are unchanged by it.
template<int BlockSize, int BlocksPerThread, class Coder, class Pred>
__device__ __forceinline__ void fused_ti_body(
    Pred pred, size_t n, uint32_t word_bytes, size_t num_blocks,
    uint8_t* __restrict__ meta, uint8_t* __restrict__ payload,
    uint32_t* __restrict__ g_state, uint32_t* __restrict__ g_agg,
    uint32_t* __restrict__ g_incl, size_t num_warps)
{
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t w    = blockIdx.x;                 // CTA = 1 warp ⇒ global warp id
    if (static_cast<size_t>(w) >= num_warps) return;

    const size_t warp_block_base = static_cast<size_t>(w) * BlocksPerThread * 32u;

    // ── Phase A: predict + cost every block this thread owns (one per row jb).
    int      d[BlocksPerThread][BlockSize];   // held codes (local memory) — no recompute in pack
    uint32_t bcost[BlocksPerThread];          // this lane's block cost per row
    #pragma unroll 1
    for (int jb = 0; jb < BlocksPerThread; ++jb) {
        const size_t b = warp_block_base + static_cast<size_t>(jb) * 32u + lane;
        if (b < num_blocks) {
            const size_t base = b * static_cast<size_t>(BlockSize);
            const uint32_t count = static_cast<uint32_t>(min(size_t{BlockSize}, n - base));
            pred.predict(base, d[jb]);
            bcost[jb] = Coder::cost(d[jb], word_bytes, count, meta + Coder::meta_bytes * b);
        } else {
            bcost[jb] = 0u;
        }
    }

    // ── Phase B: LINEAR-order 2-D prefix. block(jb,lane) sits at linear pos jb*32+lane, so its
    // exclusive byte offset within the warp = (sum of rows < jb) + (intra-row exclusive prefix).
    uint32_t blk_excl[BlocksPerThread];
    uint32_t warp_total = 0u;        // cumulative sum of full rows processed so far
    #pragma unroll 1
    for (int jb = 0; jb < BlocksPerThread; ++jb) {
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
    for (int jb = 0; jb < BlocksPerThread; ++jb) {
        const size_t b = warp_block_base + static_cast<size_t>(jb) * 32u + lane;
        if (b < num_blocks) {
            const size_t base = b * static_cast<size_t>(BlockSize);
            const uint32_t count = static_cast<uint32_t>(min(size_t{BlockSize}, n - base));
            Coder::pack(d[jb], word_bytes, count, meta + Coder::meta_bytes * b,
                        payload + warp_base + blk_excl[jb]);
        }
    }
}

} } } // namespace fz::fused::warp_ti
