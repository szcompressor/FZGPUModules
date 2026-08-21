#pragma once

#include "backend/types.h"   // fz::stream_t
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace fz {

class MemoryPool;

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

/**
 * @brief Which fused-kernel execution model a stage's device-op belongs to.
 *
 * A fused group is composed of ops that all share one strategy — the generic
 * runner routes by this, and the codegen has a per-strategy backend. The two are
 * deliberately different execution models (see the two-axis taxonomy):
 *  - `ChunkCooperative` one CTA owns a fixed byte-chunk, intermediates in shared
 *    memory, `__syncthreads` between ops (LC/PFPL-style).
 *  - `WarpRegister`     one warp owns a ≤64-element block, intermediates in
 *    registers and shuffles, no barriers (cuSZp-style).
 */
enum class FusionStrategy : uint8_t { ChunkCooperative, WarpRegister };

/**
 * @brief A stage's contribution to a generated fused kernel — the device-op it
 *        maps to, where its source lives, and its runtime parameter bytes.
 *
 * The generic runner collects one `FusedOpDecl` per stage in a fused group (after
 * priming), packs the `params` blobs in group order, and hands the ordered
 * op-name list to the codegen. This is how a stage declares its fused identity
 * without the runner hard-coding any pipeline shape. Default-constructed (empty
 * `op_name`) means "not a fused op" — the stage does not participate.
 *
 * `params` is the raw bytes of the op's POD `Params` struct; the generated kernel
 * `reinterpret_cast`s the packed blob to that type, so the host-packed layout MUST
 * match the device struct exactly (share the POD definition — see
 * modules/fused/chunk_fusion/chunk_op_params.h). Stateless ops leave it empty.
 */
struct FusedOpDecl {
    FusionStrategy       strategy = FusionStrategy::ChunkCooperative;
    std::string          op_name;         ///< device-op type name, e.g. "DiffNegabinary"
    std::string          include_header;  ///< header that defines it (for the codegen #include)
    std::vector<uint8_t> params;          ///< POD Params bytes; empty for stateless ops

    /// Warp-register predictors only (0/unused otherwise): the shape a generic warp
    /// runner needs so it does not have to downcast the predictor stage. The kernel
    /// is templated on `elems_per_lane` (= block_size/32), and the blocks cover
    /// `n_ab` padded elements. The predictor packs its config into `params` with a
    /// leading `float inv2eb` slot (offset 0) the runner fills from the resolved bound.
    uint32_t             elems_per_lane = 0;
    size_t               n_ab = 0;

    bool valid() const { return !op_name.empty(); }
};

/**
 * @brief Minimal context a fused runner hands a stage so it can establish the
 *        forward-computed state its OWN inverse will later read.
 *
 * A fused pipeline reuses the same stage objects for decompress but bypasses their
 * forward `execute()`, so any state normally computed there (e.g. a quantizer's
 * resolved error bound from a value-range scan) must be primed explicitly, or the
 * inverse reconstructs with defaults. See `Stage::primeFusedForwardState`.
 */
struct FusedPrimeContext {
    const void*  d_input     = nullptr;  ///< device input buffer
    size_t       input_bytes = 0;        ///< its size in bytes
    MemoryPool*  pool        = nullptr;  ///< scratch pool
    fz::stream_t stream      = nullptr;  ///< stream to prime on
};

} // namespace fz
