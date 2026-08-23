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
 * @brief Identifies maximal runs of fusable stages in a finalized DAG.
 *
 * The planner is pure analysis: it walks the compress DAG and returns the
 * linear chains of stages that a fused kernel could collapse into one launch
 * (predictor + quantizer + block-local coder, etc.). It does not change
 * execution — a later pass substitutes a fused implementation for a group when
 * one is registered/generated, and unmatched groups keep running staged.
 *
 * A group is a maximal chain where every stage opts into fusion
 * (`Stage::getFusionSpec()`), the chain is strictly linear (no fan-in/out
 * inside it), all block-local/cooperative members share one block size, and a
 * `Cooperative` coder, if present, terminates the chain (nothing fuses past a
 * variable-length coder). See docs/codebase_notes.md CN-FUSE-PROOF.
 */

#include "stage/fusion.h"
#include <cstdint>
#include <string>
#include <vector>

namespace fz {

class Stage;
class CompressionDAG;

/// One fusable chain, in producer→consumer order.
struct FusionGroup {
    std::vector<Stage*>      stages;       ///< the fused chain, front = producer
    std::vector<std::string> stage_names;  ///< node names, parallel to `stages`
    uint32_t                 block_size = 0;   ///< shared block size (0 if all Map)
    bool                     has_coder  = false;///< ends in a Cooperative coder
};

/// Return every maximal fusable group (size ≥ 2) in `dag`. `dag` should be
/// finalized. Groups are disjoint; stages not in any group run staged.
std::vector<FusionGroup> planFusionGroups(const CompressionDAG& dag);

} // namespace fz
