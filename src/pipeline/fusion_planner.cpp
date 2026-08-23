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

// Block sizes agree if neither constrains (Map, block 0) or both name the same.
bool blockCompatible(uint32_t group_bs, const FusionSpec& s) {
    if (s.access == FusionAccess::Map || s.block_size == 0) return true;
    return group_bs == 0 || group_bs == s.block_size;
}

} // namespace

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
        uint32_t bs = (sspec.block_size ? sspec.block_size : 0u);
        bool coder = false;
        for (;;) {
            g.stages.push_back(cur->stage);
            g.stage_names.push_back(cur->name);
            const FusionSpec cs = cur->stage->getFusionSpec();
            if (cs.block_size) bs = cs.block_size;
            consumed.insert(cur);
            if (cs.access == FusionAccess::Cooperative) { coder = true; break; }  // coder terminates

            if (cur->dependents.size() != 1) break;
            DAGNode* nxt = cur->dependents[0];
            if (!linearFusableEdge(cur, nxt)) break;
            if (!blockCompatible(bs, nxt->stage->getFusionSpec())) break;
            cur = nxt;
        }

        if (g.stages.size() >= 2) {
            g.block_size = bs;
            g.has_coder  = coder;
            groups.push_back(std::move(g));
        }
    }
    return groups;
}

} // namespace fz
