#include "advanced/fusion_planner.h"
#include "advanced/dag.h"
#include "stage/stage.h"

#include <unordered_set>

namespace fz {

namespace {

// A fused edge requires a strictly linear producer→consumer link: `prod` feeds
// exactly one stage and `cons` is fed by exactly one stage, and both opt into
// fusion. (Fan-in/out inside a group would break register-resident composition.)
bool linearFusableEdge(const DAGNode* prod, const DAGNode* cons) {
    if (!prod->stage || !cons->stage) return false;
    if (!prod->stage->getFusionSpec().fusable()) return false;
    if (!cons->stage->getFusionSpec().fusable()) return false;
    if (prod->dependents.size() != 1 || prod->dependents[0] != cons) return false;
    if (cons->dependencies.size() != 1 || cons->dependencies[0] != prod) return false;
    return true;
}

} // namespace

FusionCompatibility extendFusionGeometry(
    FusionGeometry& geometry, const FusionSpec& next)
{
    if (!next.fusable()) return FusionCompatibility::UnfusableStage;
    if (next.access == FusionAccess::Map) {
        return geometry.tileAdaptive()
            ? FusionCompatibility::TileInteriorStageUnsupported
            : FusionCompatibility::Compatible;
    }

    if (next.access == FusionAccess::TileAdaptive) {
        if (next.block_size == 0 || next.coder_unit_size == 0 ||
            next.block_size % next.coder_unit_size != 0) {
            return FusionCompatibility::InvalidTileGeometry;
        }
        if (geometry.block_size != 0)
            return FusionCompatibility::TileAfterBlockLocal;
        if (geometry.tileAdaptive())
            return FusionCompatibility::MultipleTileSelectors;
        geometry.selector_tile_size = next.block_size;
        geometry.coder_unit_size = next.coder_unit_size;
        return FusionCompatibility::Compatible;
    }

    if (geometry.tileAdaptive()) {
        if (next.access != FusionAccess::Cooperative)
            return FusionCompatibility::TileInteriorStageUnsupported;
        if (next.block_size != geometry.coder_unit_size)
            return FusionCompatibility::TileCoderUnitMismatch;
        return FusionCompatibility::Compatible;
    }

    if (next.block_size != 0) {
        if (geometry.block_size != 0 && geometry.block_size != next.block_size)
            return FusionCompatibility::StandardBlockMismatch;
        geometry.block_size = next.block_size;
    }
    return FusionCompatibility::Compatible;
}

const char* fusionCompatibilityName(FusionCompatibility result) {
    switch (result) {
        case FusionCompatibility::Compatible: return "compatible";
        case FusionCompatibility::UnfusableStage: return "unfusable_stage";
        case FusionCompatibility::InvalidTileGeometry: return "invalid_tile_geometry";
        case FusionCompatibility::TileAfterBlockLocal: return "tile_after_block_local";
        case FusionCompatibility::MultipleTileSelectors: return "multiple_tile_selectors";
        case FusionCompatibility::TileInteriorStageUnsupported:
            return "tile_interior_stage_unsupported";
        case FusionCompatibility::StandardBlockMismatch: return "standard_block_mismatch";
        case FusionCompatibility::TileCoderUnitMismatch: return "tile_coder_unit_mismatch";
    }
    return "unknown";
}

std::vector<FusionGroup> planFusionGroups(const CompressionDAG& dag) {
    std::vector<FusionGroup> groups;
    const auto& nodes = dag.getNodes();
    std::unordered_set<const DAGNode*> consumed;

    for (DAGNode* start : nodes) {
        if (!start->stage || consumed.count(start)) continue;
        const FusionSpec sspec = start->stage->getFusionSpec();
        if (!sspec.fusable()) continue;

        // Only begin a chain at a true head: a node whose single predecessor is
        // NOT a linear fusable edge into it (otherwise it is mid-chain and will
        // be picked up when its predecessor's chain is walked).
        if (start->dependencies.size() == 1 &&
            linearFusableEdge(start->dependencies[0], start)) {
            continue;
        }
        // A coder as the very first stage is a group of one — nothing to fuse.
        if (sspec.access == FusionAccess::Cooperative) continue;

        FusionGroup g;
        DAGNode* cur = start;
        FusionGeometry geometry;
        if (extendFusionGeometry(geometry, sspec) != FusionCompatibility::Compatible)
            continue;
        bool coder = false;
        for (;;) {
            g.stages.push_back(cur->stage);
            g.stage_names.push_back(cur->name);
            const FusionSpec cs = cur->stage->getFusionSpec();
            consumed.insert(cur);
            if (cs.access == FusionAccess::Cooperative) { coder = true; break; }  // coder terminates

            if (cur->dependents.size() != 1) break;
            DAGNode* nxt = cur->dependents[0];
            if (!linearFusableEdge(cur, nxt)) break;
            FusionGeometry extended = geometry;
            if (extendFusionGeometry(extended, nxt->stage->getFusionSpec()) !=
                FusionCompatibility::Compatible) break;
            geometry = extended;
            cur = nxt;
        }

        if (g.stages.size() >= 2) {
            g.block_size = geometry.block_size;
            g.selector_tile_size = geometry.selector_tile_size;
            g.coder_unit_size = geometry.coder_unit_size;
            g.has_tile_adaptive = geometry.tileAdaptive();
            g.has_coder  = coder;
            groups.push_back(std::move(g));
        }
    }
    return groups;
}

} // namespace fz
