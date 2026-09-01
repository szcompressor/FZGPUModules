#pragma once

/**
 * @file modules/fused/fused_block/warp_op_params.h
 * @brief POD parameter blocks for the warp-register predictor policies.
 *
 * Shared VERBATIM between the device policies (warp_fusion.cuh, which `using` these)
 * and the host predictor stages that pack these bytes into the fused params blob
 * (which the generated kernel `reinterpret_cast`s). Keep them plain POD and
 * dependency-free (no device code), so both the host stage compile (g++) and the
 * device compile (nvcc / NVRTC) agree on the layout — the LC-of chunk_op_params.h.
 *
 * CONVENTION: every warp predictor's Params begins with `float inv2eb` at offset 0.
 * The predictor stage cannot know the error bound (the quantizer owns it), so it
 * packs 0 there; the generic runner overwrites those 4 bytes with `1/(2*abs_eb)`
 * resolved from the primed quantizer bound before uploading the blob.
 */

#include <cstdint>

namespace fz {
namespace fused {
namespace warp {

/// Largest block a warp-register fused kernel accepts, in elements-per-lane
/// (block_size = 32 * EPL). Shared by the predictor + coder stages (which gate
/// their fused-op declarations on it) and the launcher (which forces the
/// two-pass path for EPL > 2 — the single-pass body holds BlocksPerWarp*EPL
/// deltas in local memory).
///
/// Set to 4 (block 128) by measurement: on CLDHGH 3600x1800 / H100 / eb 1e-3 the
/// fused Quantizer->Lorenzo(1-D)->AdaptiveBitpack compress throughput climbs
/// 201 (EPL 1) -> 245 (2) -> 255 (3) -> 275 (4) GB/s and then PLATEAUS
/// (~266-271 GB/s for EPL 5-8) while per-lane register/local-memory pressure
/// keeps rising. EPL 4 also covers every natural block size (32 = cuSZp2,
/// 64 = cuSZp3 / 1-D, 128 = SZp). Bump only with a measured throughput win.
constexpr uint32_t kMaxWarpElemsPerLane = 4;   // block_size <= 128

/// cuSZp2 / SZp: linear-ABS quant + 1-D Lorenzo, block reset every 32*epl
/// elements. `epl` (= block_size/32) lets the policy address element `32*m+lane`
/// of block `b` for epl > 1; the delta chain runs across the whole block.
struct Lorenzo1DParams { float inv2eb; uint32_t epl; };

/// cuSZp3: linear-ABS quant + 2-D separable tiled Lorenzo (tz == 1).
struct TiledLorenzo2DParams {
    float    inv2eb;
    uint32_t dx, dy;   ///< field extents (x fastest)
    uint32_t tx, ty;   ///< tile extents (tx*ty == block_size == 64)
    uint32_t ntx;      ///< number of tiles along x (= ceil(dx/tx))
};

/// cuSZp3: linear-ABS quant + 3-D separable tiled Lorenzo (tz > 1). PROTOTYPE.
struct TiledLorenzo3DParams {
    float    inv2eb;
    uint32_t dx, dy, dz;   ///< field extents (x fastest)
    uint32_t tx, ty, tz;   ///< tile extents (tx*ty*tz == block_size == 64)
    uint32_t ntx, nty;     ///< tiles along x, y (= ceil(dx/tx), ceil(dy/ty))
};

} // namespace warp
} // namespace fused
} // namespace fz
