#pragma once

/**
 * @file modules/fused/fused_block/nvrtc_warp_fusion.h
 * @brief Runtime NVRTC codegen for the warp-register fusion harness (cuSZp family).
 *
 * The compile-time path (fused_block.cu) instantiates the rate/pack kernels for a
 * fixed <ElemsPerLane, Predictor> chosen in C++ and dispatches on the predictor name.
 * This path does the same at RUNTIME: a WarpFusionSpec (predictor policy type name +
 * ElemsPerLane) is turned into source that wraps `fused_rate_body`/`fused_pack_body`,
 * compiled+cached via the shared NVRTC JIT, and launched with the host-side CUB
 * exclusive-scan of per-block costs in between. Adding a predictor becomes writing its
 * policy in warp_fusion.cuh and having its stage declare the op — no new template
 * instantiation, no per-shape launcher, no string dispatch in the registry.
 */

#include "backend/types.h"
#include <cstddef>
#include <cstdint>
#include <string>

namespace fz {
class MemoryPool;

namespace fused {

/// The warp-chain fingerprint the codegen composes: the predictor device-policy type
/// name (defined in warp_fusion.cuh) and ElemsPerLane (= block_size / 32, a
/// compile-time template arg of the kernels). Defaults spell cuSZp2.
struct WarpFusionSpec {
    std::string predictor      = "Lorenzo1DPredictor";
    int         elems_per_lane = 1;
};

/// The CUDA source the codegen emits for `spec` (exposed for tests/inspection).
std::string generateWarpFusionSource(const WarpFusionSpec& spec);

/**
 * Generic warp-register fused compress — the entry the registry runner uses with no
 * per-predictor shape. NVRTC-composes `spec` (predictor + ElemsPerLane), uploads the
 * predictor params blob (`pred_params`/`params_bytes`; a leading `float inv2eb` at
 * offset 0, already resolved by the caller), runs the rate kernel, the CUB
 * exclusive-scan of per-block costs, and the pack kernel, then reads back the archive
 * length. `n_ab` is the padded block-covering element count; block_size = 32 *
 * elems_per_lane. Byte-identical to the pre-instantiated launchers. Returns the
 * archive length, or 0 if `n_ab == 0`.
 */
size_t launchNvrtcWarpFused(
    const WarpFusionSpec& spec, const float* d_in, size_t n_ab,
    const uint8_t* pred_params, size_t params_bytes,
    uint8_t* d_out, MemoryPool* pool, fz::stream_t stream);

} // namespace fused
} // namespace fz
