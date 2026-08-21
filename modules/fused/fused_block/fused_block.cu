// Parametric fused block-local compress (predict+quant+fixed-rate outlier coder).
// The per-element int code is produced on the fly by a predictor policy instead of
// read from a materialised codes array, so quant + predictor + coder collapse into
// two kernels with no DRAM round-trip for the intermediate codes. ElemsPerLane =
// block_size/32 and the predictor are compile-time parameters. Byte-identical to the
// staged path; see docs/codebase_notes.md CN-FUSE-PROOF / CN-FUSE-EXEC.
//
// The device pieces (predictor policies + the two warp encode bodies) live in
// warp_fusion.cuh so the runtime NVRTC path compiles the SAME bodies. This file is
// the host orchestration: the CUB exclusive-scan of per-block costs, the length
// read-back, and the two public shape launchers.
//
// The registry runner now composes the warp kernels through the NVRTC path
// (nvrtc_warp_fusion.*); these compile-time launchers remain as the nvcc-instantiated
// reference (they compile-check the bodies against the host toolchain, and are the
// oracle the byte-identity was proven against) — the same role the chunk strategy's
// template path plays alongside its NVRTC generator.

#include "fused/fused_block/fused_block.h"
#include "fused/fused_block/warp_fusion.cuh"
#include "coders/adaptive_bitpack/adaptive_bitpack_kernels.h"
#include "mem/mempool.h"
#include "cuda_check.h"

#include "backend/cub.h"
#include "backend/algorithms.h"
#include <cub/device/device_scan.cuh>

namespace fz {
namespace fused {

namespace ab = fz::adaptive_bitpack;
using warp::Lorenzo1DPredictor;
using warp::TiledLorenzo2DPredictor;

// ── Core launcher ────────────────────────────────────────────────────────────
// Runs the two warp kernels + the CUB exclusive-scan offsets, then reads back the
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

    // The compile-time reference uses the AdaptiveBitpack coder (the byte-identity
    // oracle for cuszp2/3); the NVRTC path composes whichever coder the chain names.
    warp::fused_rate_kernel<ElemsPerLane, warp::AdaptiveBitpackCoder, Pred><<<grid, THREADS, 0, stream>>>(
        pred, n_ab, word_bytes, num_blocks, d_meta, d_cost);
    FZ_CUDA_CHECK(cudaGetLastError());

    auto d_tmp = fz::backend::withTempStorage(pool, stream, "fused_cub",
        [&](void* tmp, size_t& bytes) {
            cub::DeviceScan::ExclusiveSum(tmp, bytes, d_cost, d_offset, num_blocks, stream);
        });

    warp::fused_pack_kernel<ElemsPerLane, warp::AdaptiveBitpackCoder, Pred><<<grid, THREADS, 0, stream>>>(
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
