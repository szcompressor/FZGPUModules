#pragma once

/**
 * @file modules/fused/fused_block/fused_block.h
 * @brief Fused compress for the block-local "predict+quant+fixed-rate-coder"
 *        family (cuSZp2 and cuSZp3).
 *
 * Collapses Quantizer(linear,ABS) + a block-local predictor + AdaptiveBitpack
 * (outlier) into per-warp work: one warp owns one block and runs quant +
 * predictor delta + adaptive-bitpack in registers, so the int codes are never
 * materialised to DRAM. The rate/pack kernels are the AdaptiveBitpack outlier
 * warp kernels parameterised on ElemsPerLane (= block_size/32) and a predictor
 * policy, so the two shapes share one code path. Measured 2-3x over the staged
 * pipeline; see docs/codebase_notes.md CN-FUSE-PROOF / CN-FUSE-EXEC. The output
 * archive is byte-identical to the staged AdaptiveBitpack output, so decode is
 * unchanged.
 *
 * Registry entries in the "predict+quant+fixedcoder" driver:
 *   - cuSZp2: Lorenzo(1-D, block=32)         → launchFusedCuszp2Compress
 *   - cuSZp3: TiledLorenzo(2-D 8x8, block=64) → launchFusedCuszp3Compress
 * Neither is graph-capturable — each synchronises to read back the archive
 * length — so fusion disables graph mode.
 */

#include "backend/types.h"
#include <cstddef>
#include <cstdint>

namespace fz {
class MemoryPool;

namespace fused {

/**
 * Run the fused cuSZp2 compress.
 * @param d_in        device float input (n elements)
 * @param n           element count
 * @param abs_eb      absolute error bound (2*eb reconstruction spacing)
 * @param block_size  32 only, for now
 * @param outlier     per-block outlier selection (true for cuSZp2)
 * @param d_out       device archive buffer (>= worst-case AdaptiveBitpack size)
 * @param pool        pool for look-back scratch (cost/offset/CUB temp)
 * @param stream      CUDA stream (synchronised internally to read the length)
 * @return total archive length in bytes (meta region + packed payload)
 */
size_t launchFusedCuszp2Compress(
    const float* d_in, size_t n, float abs_eb, uint32_t block_size, bool outlier,
    uint8_t* d_out, MemoryPool* pool, fz::stream_t stream);

/**
 * Run the fused cuSZp3 compress (linear-ABS quant + 2-D separable tiled Lorenzo +
 * outlier AdaptiveBitpack with block == tile_elems). Only the 8x8 (tile_elems ==
 * 64), 2-D shape is supported; other shapes return 0 so the caller falls back to
 * the staged path. The archive is byte-identical to the staged cuSZp3 output.
 * @param d_in    device float field (dx*dy elements, row-major, x fastest)
 * @param dx,dy   field dimensions (tz == 1)
 * @param abs_eb  absolute error bound
 * @param tx,ty   tile extents (tx*ty must be 64)
 * @param d_out   device archive buffer
 * @param pool    scratch pool (cost/offset/CUB temp)
 * @param stream  CUDA stream (synchronised internally to read the length)
 * @return total archive length in bytes (meta region + packed payload)
 */
size_t launchFusedCuszp3Compress(
    const float* d_in, size_t dx, size_t dy, float abs_eb, uint32_t tx, uint32_t ty,
    uint8_t* d_out, MemoryPool* pool, fz::stream_t stream);

} // namespace fused
} // namespace fz
