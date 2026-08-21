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

/// cuSZp2: linear-ABS quant + 1-D Lorenzo. Only inv2eb (dims are implicit in n).
struct Lorenzo1DParams { float inv2eb; };

/// cuSZp3: linear-ABS quant + 2-D separable tiled Lorenzo (tz == 1).
struct TiledLorenzo2DParams {
    float    inv2eb;
    uint32_t dx, dy;   ///< field extents (x fastest)
    uint32_t tx, ty;   ///< tile extents (tx*ty == block_size == 64)
    uint32_t ntx;      ///< number of tiles along x (= ceil(dx/tx))
};

} // namespace warp
} // namespace fused
} // namespace fz
