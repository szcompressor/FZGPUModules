#pragma once

/*
 * ADVANCED API — no source-compatibility promise. The types here (CompressionDAG,
 * DAGNode, BufferInfo, the fusion planner/registry) are pipeline internals exposed
 * for advanced/experimental use; they may change or be removed in any release.
 * Most users only need <fzgpumodules.h> / pipeline/compressor.h. See the API tiers
 * in docs/api_reference.md.
 */

/**
 * @file advanced/fusion_planner.h
 * @brief Identifies maximal fusion-legality domains in a finalized DAG.
 *
 * The planner is pure analysis: it walks the compress DAG and returns the
 * linear chains within which fused execution plans may be considered. It does
 * not require one implementation to cover the whole maximal chain: a later pass
 * enumerates contiguous subspans, finds registered/generated implementations,
 * and selects non-overlapping profitable candidates. Unmatched stages keep
 * running staged.
 *
 * A group is a maximal chain where every stage opts into fusion
 * (`Stage::getFusionSpec()`), the chain is strictly linear (no fan-in/out
 * inside it), conventional block-local/cooperative members share one block
 * size, and a `Cooperative` coder, if present, terminates the chain (nothing
 * fuses past a variable-length coder). A `TileAdaptive` selector instead owns a
 * larger tile made of equal immediate downstream coder units.
 * See docs/codebase_notes.md CN-FUSE-PROOF.
 */

#include "stage/fusion.h"
#include <cstdint>
#include <string>
#include <vector>

namespace fz {

class Stage;
class CompressionDAG;

/// Geometry accumulated while extending one candidate fusion chain.
struct FusionGeometry {
    uint32_t block_size = 0;         ///< legacy same-size block/chunk path
    uint32_t selector_tile_size = 0; ///< TileAdaptive selector unit
    uint32_t coder_unit_size = 0;    ///< coder units nested in selector tile

    bool tileAdaptive() const { return selector_tile_size != 0; }
};

/// Exact reason a stage spec cannot extend an accumulated group geometry.
enum class FusionCompatibility : uint8_t {
    Compatible = 0,
    UnfusableStage,
    InvalidTileGeometry,
    TileAfterBlockLocal,
    MultipleTileSelectors,
    TileInteriorStageUnsupported,
    StandardBlockMismatch,
    TileCoderUnitMismatch,
};

/// Validate and, on success, extend `geometry` with `next`.
FusionCompatibility extendFusionGeometry(
    FusionGeometry& geometry, const FusionSpec& next);

const char* fusionCompatibilityName(FusionCompatibility result);

/// One maximal fusion-legality domain, in producer→consumer order.
struct FusionGroup {
    std::vector<Stage*>      stages;       ///< the fused chain, front = producer
    std::vector<std::string> stage_names;  ///< node names, parallel to `stages`
    uint32_t                 block_size = 0;   ///< shared block size (0 if all Map)
    uint32_t                 selector_tile_size = 0; ///< TileAdaptive unit, else 0
    uint32_t                 coder_unit_size = 0;    ///< nested coder unit, else 0
    bool                     has_tile_adaptive = false;
    bool                     has_coder  = false;///< ends in a Cooperative coder
};

/// Return every maximal legal group (size >= 2) in `dag`. `dag` should be
/// finalized. Groups are disjoint search domains; installation may choose
/// smaller contiguous subspans, and all unselected stages run staged.
std::vector<FusionGroup> planFusionGroups(const CompressionDAG& dag);

} // namespace fz
