#include "advanced/fusion_planner.h"
#include "advanced/dag.h"
#include "stage/stage.h"

#include <algorithm>
#include <unordered_set>

namespace fz {

namespace {

bool declaresAuxOutput(const Stage* stage, int output_index) {
    const auto declarations = stage->getFusedAuxOutputs();
    return std::any_of(
        declarations.begin(), declarations.end(),
        [&](const FusedAuxOutputDecl& d) {
            return d.valid() && d.output_index == output_index;
        });
}

DAGNode* nodeById(const CompressionDAG& dag, int id) {
    for (DAGNode* node : dag.getNodes())
        if (node && node->id == id) return node;
    return nullptr;
}

// A fused edge follows output port 0, the main representation carried by every
// registered/generated specialization. The main path remains strictly linear.
// Other connected output ports may cross the region boundary only when the
// producer declares how the fused runner materializes and sizes them.
bool linearFusableEdge(
    const CompressionDAG& dag, const DAGNode* prod, const DAGNode* cons)
{
    if (!prod->stage || !cons->stage) return false;
    if (!prod->stage->getFusionSpec().fusable()) return false;
    if (!cons->stage->getFusionSpec().fusable()) return false;
    if (cons->dependencies.size() != 1 || cons->dependencies[0] != prod) return false;

    const auto main_it = prod->output_index_to_buffer_id.find(0);
    if (main_it == prod->output_index_to_buffer_id.end()) return false;
    const BufferInfo& main = dag.getBufferInfo(main_it->second);
    if (main.consumer_stage_ids.size() != 1 ||
        main.consumer_stage_ids.front() != cons->id) return false;

    for (const auto& [output_index, buffer_id] : prod->output_index_to_buffer_id) {
        if (output_index == 0) continue;
        const BufferInfo& output = dag.getBufferInfo(buffer_id);
        if (!output.consumer_stage_ids.empty() &&
            !declaresAuxOutput(prod->stage, output_index)) return false;
    }
    return true;
}

DAGNode* mainFusableSuccessor(const CompressionDAG& dag, DAGNode* prod) {
    const auto it = prod->output_index_to_buffer_id.find(0);
    if (it == prod->output_index_to_buffer_id.end()) return nullptr;
    const BufferInfo& main = dag.getBufferInfo(it->second);
    if (main.consumer_stage_ids.size() != 1) return nullptr;
    DAGNode* cons = nodeById(dag, main.consumer_stage_ids.front());
    return cons && linearFusableEdge(dag, prod, cons) ? cons : nullptr;
}

} // namespace

FusionCompatibility extendFusionGeometry(
    FusionGeometry& geometry, const FusionSpec& next)
{
    if (!next.fusable()) return FusionCompatibility::UnfusableStage;
    if (next.access == FusionAccess::Elementwise) {
        return geometry.hasTileSelector()
            ? FusionCompatibility::TileInteriorStageUnsupported
            : FusionCompatibility::Compatible;
    }

    if (next.access == FusionAccess::TileSelector) {
        if (next.block_size == 0 || next.coder_unit_size == 0 ||
            next.block_size % next.coder_unit_size != 0) {
            return FusionCompatibility::InvalidTileGeometry;
        }
        if (geometry.block_size != 0)
            return FusionCompatibility::TileAfterRegionLocal;
        if (geometry.hasTileSelector())
            return FusionCompatibility::MultipleTileSelectors;
        geometry.selector_tile_size = next.block_size;
        geometry.coder_unit_size = next.coder_unit_size;
        return FusionCompatibility::Compatible;
    }

    if (geometry.hasTileSelector()) {
        if (next.access != FusionAccess::SegmentCodec)
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
        case FusionCompatibility::TileAfterRegionLocal: return "tile_after_region_local";
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
            linearFusableEdge(dag, start->dependencies[0], start)) {
            continue;
        }
        // A coder as the very first stage is a group of one — nothing to fuse.
        if (sspec.access == FusionAccess::SegmentCodec) continue;

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
            if (cs.access == FusionAccess::SegmentCodec) { coder = true; break; }  // codec terminates

            DAGNode* nxt = mainFusableSuccessor(dag, cur);
            if (!nxt) break;
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
            g.has_tile_selector = geometry.hasTileSelector();
            g.has_coder  = coder;
            groups.push_back(std::move(g));
        }
    }
    return groups;
}

} // namespace fz
