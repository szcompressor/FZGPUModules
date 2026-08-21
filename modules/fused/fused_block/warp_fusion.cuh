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

// cuSZp2: linear-ABS quant + 1-D Lorenzo delta, reset per 32-block. All lanes call
// delta() so the intra-warp __shfl_up_sync is collective.
struct Lorenzo1DPredictor {
    const float* in;
    size_t n;
    float inv2eb;
    __device__ static Lorenzo1DPredictor fromParams(const float* in, size_t n, const void* pp) {
        return Lorenzo1DPredictor{in, n, static_cast<const Lorenzo1DParams*>(pp)->inv2eb};
    }
    __device__ __forceinline__ int delta(uint32_t lane, size_t b, int /*m*/) const {
        const size_t gidx = b * 32u + lane;
        const bool active = gidx < n;
        const int q = active ? __float2int_rn(in[gidx] * inv2eb) : 0;
        const int qprev = __shfl_up_sync(0xffffffffu, q, 1);
        return (lane == 0) ? q : (q - qprev);
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

// ── Harness bodies (predictor-agnostic) ──────────────────────────────────────
// Factored out of the __global__ entries so the NVRTC path can wrap them in an
// extern "C" kernel; the template path below calls them too.
template<int ElemsPerLane, class Pred>
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
    uint32_t acc_all = 0, acc_rest = 0;
    #pragma unroll
    for (int m = 0; m < ElemsPerLane; ++m) {
        const size_t idx = static_cast<size_t>(lane) + 32u * m;
        d[m] = pred.delta(lane, b, m);
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
        if (cost_plain <= cost_out) { meta[2*b]=fr_all;  meta[2*b+1]=0; cost[b]=cost_plain; }
        else { meta[2*b]=fr_rest; meta[2*b+1]=static_cast<uint8_t>(1u|((ob_bytes-1u)<<1)); cost[b]=cost_out; }
    }
}

template<int ElemsPerLane, class Pred>
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

    const int r = meta[2*b]; const uint8_t sel = meta[2*b+1];
    const bool is_out = (sel & 1u) != 0;
    constexpr uint32_t block_size = static_cast<uint32_t>(ElemsPerLane) * 32u;
    const size_t start = b * block_size;
    const size_t count = min(static_cast<size_t>(block_size), n - start);
    uint8_t* base = payload + offset[b];

    int d[ElemsPerLane];
    uint32_t av[ElemsPerLane];
    bool active[ElemsPerLane];
    #pragma unroll
    for (int m = 0; m < ElemsPerLane; ++m) {
        const size_t idx = static_cast<size_t>(lane) + 32u * m;
        d[m]      = pred.delta(lane, b, m);
        active[m] = idx < count;
        av[m]     = active[m] ? absU_i32(d[m]) : 0u;
    }

    if (!is_out) {
        if (r == 0) return;
        #pragma unroll
        for (int m = 0; m < ElemsPerLane; ++m) {
            const uint32_t sm = __ballot_sync(0xffffffffu, active[m] && d[m] < 0);
            if (lane < 4) base[4u*m + lane] = static_cast<uint8_t>((sm >> (8u*lane)) & 0xFFu);
        }
        for (int p = 0; p < r; ++p) {
            #pragma unroll
            for (int m = 0; m < ElemsPerLane; ++m) {
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
    for (int m = 0; m < ElemsPerLane; ++m) {
        const uint32_t sm = __ballot_sync(0xffffffffu, active[m] && d[m] < 0);
        if (lane < 4) sign[4u*m + lane] = static_cast<uint8_t>((sm >> (8u*lane)) & 0xFFu);
    }
    for (int p = 0; p < r; ++p) {
        #pragma unroll
        for (int m = 0; m < ElemsPerLane; ++m) {
            const bool plane_active = active[m] && (static_cast<size_t>(lane) + 32u*m) > 0;
            const uint32_t pm = __ballot_sync(0xffffffffu, plane_active && ((av[m] >> p) & 1u));
            if (lane < 4)
                planes[word_bytes*p + 4u*m + lane] = static_cast<uint8_t>((pm >> (8u*lane)) & 0xFFu);
        }
    }
}

// ── Compile-time template kernels (the non-NVRTC path). The NVRTC path generates
// equivalent extern "C" kernels over the same bodies. ────────────────────────
template<int ElemsPerLane, class Pred>
__global__ void fused_rate_kernel(
    Pred pred, size_t n, uint32_t word_bytes, size_t num_blocks,
    uint8_t* __restrict__ meta, uint32_t* __restrict__ cost)
{
    fused_rate_body<ElemsPerLane, Pred>(pred, n, word_bytes, num_blocks, meta, cost);
}

template<int ElemsPerLane, class Pred>
__global__ void fused_pack_kernel(
    Pred pred, size_t n, uint32_t word_bytes, size_t num_blocks,
    const uint8_t* __restrict__ meta, const uint32_t* __restrict__ offset,
    uint8_t* __restrict__ payload)
{
    fused_pack_body<ElemsPerLane, Pred>(pred, n, word_bytes, num_blocks, meta, offset, payload);
}

} // namespace warp
} // namespace fused
} // namespace fz
