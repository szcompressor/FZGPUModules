#pragma once

#include <cstdint>

namespace fz {

/**
 * @brief How a stage accesses its input — the property that decides whether it
 *        can be fused into a single kernel with its neighbours.
 *
 * Fusion keeps a block's data register/shared-resident across a chain of stages
 * instead of materialising each intermediate to DRAM. Whether that is possible
 * depends only on a stage's data-access pattern, not on what it computes:
 *
 *  - `Map`         element-wise, `out[i] = f(in[i])`. Composes with anything.
 *  - `BlockLocal`  bounded, resettable neighbourhood inside a fixed-size block
 *                  (e.g. 1-D Lorenzo delta with a per-block reset). Fusable with
 *                  other block-local / map stages of the same block size.
 *  - `Cooperative` warp/block reduce+scan producing variable-length output — a
 *                  fixed-length coder. Fusable as the *tail* of a block-local
 *                  chain (its per-block work consumes the block still in
 *                  registers); the cross-block offset prefix is handled by the
 *                  fused driver.
 *  - `Unfusable`   opaque kernel or a genuine global dependency (entropy coder
 *                  with a global codebook, whole-array scan). A fusion barrier.
 *
 * See docs/codebase_notes.md CN-FUSE-PROOF for the measured motivation.
 */
enum class FusionAccess : uint8_t {
    Unfusable = 0,
    Map,
    BlockLocal,
    Cooperative,
};

/**
 * @brief A stage's fusion contract. Stages that can participate in a fused
 *        kernel override `Stage::getFusionSpec()` to return a non-`Unfusable`
 *        spec; the default is `Unfusable` (a barrier), so a stage is only ever
 *        fused if it opts in.
 */
struct FusionSpec {
    /// Access pattern class.
    FusionAccess access = FusionAccess::Unfusable;
    /// Reset/tile period in elements for `BlockLocal`/`Cooperative`; 0 = N/A.
    /// Block-local and cooperative members of one fused group must agree on this.
    uint32_t block_size = 0;

    bool fusable() const { return access != FusionAccess::Unfusable; }
};

} // namespace fz
