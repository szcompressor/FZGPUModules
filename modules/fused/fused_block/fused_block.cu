// Parametric fused block-local compress (predict+quant+fixed-rate outlier coder).
// The rate/pack kernels ARE the AdaptiveBitpack outlier warp kernels
// (encode_{rate,pack}_outlier_kernel_warp), but the per-element int code is
// produced on the fly by a predictor policy instead of read from a materialised
// codes array — so quant + predictor + coder collapse into two kernels with no
// DRAM round-trip for the intermediate codes. ElemsPerLane = block_size/32 and
// the predictor are compile-time parameters, so a warp owns one block and the
// tiny per-lane register arrays don't spill. Byte-identical to the staged path;
// see docs/codebase_notes.md CN-FUSE-PROOF / CN-FUSE-EXEC.

#include "fused/fused_block/fused_block.h"
#include "coders/adaptive_bitpack/adaptive_bitpack_kernels.h"
#include "mem/mempool.h"
#include "cuda_check.h"

#include "backend/cub.h"
#include "backend/algorithms.h"
#include <cub/device/device_scan.cuh>

namespace fz {
namespace fused {

namespace ab = fz::adaptive_bitpack;

__device__ __forceinline__ uint32_t absU_i32(int v) {
    return static_cast<uint32_t>(v < 0 ? -v : v);
}
__device__ __forceinline__ int bitWidth32(uint32_t x) { return x ? (32 - __clz(x)) : 0; }

// ── Predictor policies ──────────────────────────────────────────────────────
// Each returns the signed int code (delta) for element `lane + 32*m` of warp-
// block `b`, quantising the float input inline. Out-of-range / padding elements
// return 0 (matching the staged predictor, which writes 0 there). Passed by value
// to the kernels (small PODs), so no global state and no register spills.

// cuSZp2: linear-ABS quant + 1-D Lorenzo delta, reset per 32-block. All lanes
// call delta() so the intra-warp __shfl_up_sync is collective.
struct Lorenzo1DPredictor {
    const float* in;
    size_t n;
    float inv2eb;
    __device__ __forceinline__ int delta(uint32_t lane, size_t b, int /*m*/) const {
        const size_t gidx = b * 32u + lane;
        const bool active = gidx < n;
        const int q = active ? __float2int_rn(in[gidx] * inv2eb) : 0;
        const int qprev = __shfl_up_sync(0xffffffffu, q, 1);
        return (lane == 0) ? q : (q - qprev);
    }
};

// cuSZp3: linear-ABS quant + 2-D separable tiled Lorenzo (tz == 1). Each element
// re-quantises its own left/up predecessor from the float field (a pure map, so
// the neighbour code equals what the staged quantizer produced). Mirrors
// tiled_lorenzo_delta_kernel exactly: X-delta if lx>0, else Y-delta if ly>0, else
// tile origin (pred 0). tile_elems == tx*ty == block_size, so local ∈ [0, 64).
struct TiledLorenzo2DPredictor {
    const float* in;
    float inv2eb;
    uint32_t dx, dy, tx, ty, ntx;
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

// ── Fused kernels (AdaptiveBitpack outlier warp encode, predictor-sourced) ───
template<int ElemsPerLane, class Pred>
__global__ void fused_rate_kernel(
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
__global__ void fused_pack_kernel(
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

// ── Core launcher ────────────────────────────────────────────────────────────
// Runs the two kernels + the CUB exclusive-scan offsets, then reads back the
// archive length. Outlier mode only (meta_bytes = 2); both cuSZp2 and cuSZp3 use
// it. Not graph-capturable (the length read-back syncs); fusion disables graph
// mode in planAndInstallFusion().
template<int ElemsPerLane, class Pred>
static size_t launchFusedBlockCore(
    Pred pred, size_t n_ab, uint32_t word_bytes, size_t num_blocks,
    uint8_t* d_out, MemoryPool* pool, cudaStream_t stream)
{
    if (num_blocks == 0) return 0;
    const size_t meta_region = 2u * num_blocks;   // outlier meta_bytes == 2

    auto* d_cost   = static_cast<uint32_t*>(pool->allocate(sizeof(uint32_t)*num_blocks, stream, "fused_cost"));
    auto* d_offset = static_cast<uint32_t*>(pool->allocate(sizeof(uint32_t)*num_blocks, stream, "fused_offset"));

    const int WPB = 8, THREADS = WPB * 32;
    const int grid = static_cast<int>((num_blocks + WPB - 1) / WPB);
    uint8_t* d_meta    = d_out;
    uint8_t* d_payload = d_out + meta_region;

    fused_rate_kernel<ElemsPerLane, Pred><<<grid, THREADS, 0, stream>>>(
        pred, n_ab, word_bytes, num_blocks, d_meta, d_cost);
    FZ_CUDA_CHECK(cudaGetLastError());

    auto d_tmp = fz::backend::withTempStorage(pool, stream, "fused_cub",
        [&](void* tmp, size_t& bytes) {
            cub::DeviceScan::ExclusiveSum(tmp, bytes, d_cost, d_offset, num_blocks, stream);
        });

    fused_pack_kernel<ElemsPerLane, Pred><<<grid, THREADS, 0, stream>>>(
        pred, n_ab, word_bytes, num_blocks, d_meta, d_offset, d_payload);
    FZ_CUDA_CHECK(cudaGetLastError());

    uint32_t h_off = 0, h_cost = 0;
    FZ_CUDA_CHECK(cudaMemcpyAsync(&h_off,  d_offset + num_blocks-1, 4, cudaMemcpyDeviceToHost, stream));
    FZ_CUDA_CHECK(cudaMemcpyAsync(&h_cost, d_cost   + num_blocks-1, 4, cudaMemcpyDeviceToHost, stream));
    FZ_CUDA_CHECK(cudaStreamSynchronize(stream));

    fz::backend::freeTempStorage(pool, d_tmp, stream);
    pool->free(d_offset, stream);
    pool->free(d_cost, stream);
    return meta_region + static_cast<size_t>(h_off) + h_cost;
}

// ── Public launchers ─────────────────────────────────────────────────────────
size_t launchFusedCuszp2Compress(
    const float* d_in, size_t n, float abs_eb, uint32_t block_size, bool outlier,
    uint8_t* d_out, MemoryPool* pool, cudaStream_t stream)
{
    if (n == 0 || block_size != 32u || !outlier) return 0;   // cuSZp2 shape only
    ab::Config cfg = ab::configure(n, block_size, outlier);
    Lorenzo1DPredictor pred{d_in, n, 1.0f / (2.0f * abs_eb)};
    return launchFusedBlockCore<1>(pred, n, cfg.word_bytes, cfg.num_blocks, d_out, pool, stream);
}

size_t launchFusedCuszp3Compress(
    const float* d_in, size_t dx, size_t dy, float abs_eb, uint32_t tx, uint32_t ty,
    uint8_t* d_out, MemoryPool* pool, cudaStream_t stream)
{
    const uint32_t tile_elems = tx * ty;
    if (dx == 0 || dy == 0 || tile_elems != 64u) return 0;   // block-64 2-D driver only
    const uint32_t ntx = static_cast<uint32_t>((dx + tx - 1) / tx);
    const uint32_t nty = static_cast<uint32_t>((dy + ty - 1) / ty);
    const size_t num_tiles = static_cast<size_t>(ntx) * nty;
    const size_t n_ab = num_tiles * tile_elems;              // tile-major padded count
    ab::Config cfg = ab::configure(n_ab, tile_elems, /*outlier=*/true);
    TiledLorenzo2DPredictor pred{d_in, 1.0f / (2.0f * abs_eb),
                                 static_cast<uint32_t>(dx), static_cast<uint32_t>(dy), tx, ty, ntx};
    return launchFusedBlockCore<2>(pred, n_ab, cfg.word_bytes, cfg.num_blocks, d_out, pool, stream);
}

} // namespace fused
} // namespace fz
