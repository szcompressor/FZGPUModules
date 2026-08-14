#pragma once

/**
 * @file include/pipeline/fusion_registry.h
 * @brief Registry of fused implementations, keyed by the shape of a fusion group.
 *
 * The fusion planner (fusion_planner.h) finds *candidate* fusable chains. This
 * registry answers "is there a fused kernel for this exact chain?". A hit lets
 * the DAG executor run one fused kernel in place of the group's staged
 * execute()s; a miss leaves the group staged. Adding a new fused configuration =
 * registering one `FusedImpl` (later: NVRTC-generated ones keyed by fingerprint).
 *
 * See docs/codebase_notes.md CN-FUSE-PROOF / CN-FUSE-PLAN.
 */

#include "backend/types.h"
#include <cstddef>
#include <vector>

namespace fz {

class Stage;
class MemoryPool;

/// Everything a fused runner needs to compress one group in place.
struct FusedRunContext {
    const std::vector<Stage*>* stages;   ///< group stages, producer→consumer
    const void* d_input;                 ///< the group's input buffer
    size_t      input_bytes;             ///< bytes of d_input
    void*       d_output;                ///< the group's output buffer (>= worst case)
    MemoryPool* pool;                    ///< pool for the runner's scratch
    fz::stream_t stream;                 ///< stream (runner may synchronise)
};

/// A registered fused implementation and its matcher.
struct FusedImpl {
    const char* name;
    /// True if this impl handles `group` exactly (types + relevant config).
    bool   (*matches)(const std::vector<Stage*>& group);
    /// Run the fused compress; return the archive length written to d_output.
    size_t (*run)(const FusedRunContext& ctx);
};

/// First registered impl whose matcher accepts `group`, or nullptr.
const FusedImpl* findFusedImpl(const std::vector<Stage*>& group);

} // namespace fz
