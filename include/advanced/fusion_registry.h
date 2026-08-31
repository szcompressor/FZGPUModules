#pragma once

/*
 * ADVANCED API — no source-compatibility promise. The types here (CompressionDAG,
 * DAGNode, BufferInfo, the fusion planner/registry) are pipeline internals exposed
 * for advanced/experimental use; they may change or be removed in any release.
 * Most users only need <fzgpumodules.h> / pipeline/compressor.h. See the API tiers
 * in docs/api_reference.md.
 */

/**
 * @file advanced/fusion_registry.h
 * @brief Registry of fused implementations, keyed by the shape of a fusion group.
 *
 * The fusion planner (fusion_planner.h) finds maximal legality domains. The
 * installer queries this registry for each contiguous subspan: "is there a
 * fused kernel for this exact chain?". An eligible hit lets
 * the DAG executor run one fused kernel in place of the group's staged
 * execute()s; misses and unselected overlaps remain staged. Normal Auto selection
 * also requires the implementation's profitability gate. Adding a new fused
 * configuration means registering one `FusedImpl` (later: NVRTC-generated ones
 * keyed by fingerprint).
 *
 * See docs/codebase_notes.md CN-FUSE-PROOF / CN-FUSE-PLAN.
 */

#include "backend/types.h"
#include "stage/fusion.h"
#include <cstddef>
#include <vector>

namespace fz {

class Stage;
class MemoryPool;

/// A group member's escaping output port — a side output (e.g. an outlier list)
/// the fused kernel produces in addition to the main archive. These become
/// pipeline leaf outputs and are auto-concatenated by the pipeline, so a fused op
/// can emit outliers without moving that work out of the fused kernel. The runner
/// writes `d_ptr` and reports the bytes it wrote in `size`; the DAG then sizes the
/// buffer from it. Empty for the common single-output case.
struct FusedSideOutput {
    Stage* producer;       ///< the group member that owns this output port
    int    output_index;   ///< which of the producer's output ports this is
    void*  d_ptr;          ///< pre-allocated buffer for the port
    size_t capacity;       ///< its allocated size in bytes
    size_t size;           ///< OUT: bytes the runner wrote (DAG sets the buffer size from this)
    FusedAuxOutputDecl declaration; ///< semantic identity/sizing rule, if declared
};

/// Everything a fused runner needs to compress one group in place.
struct FusedRunContext {
    const std::vector<Stage*>* stages;   ///< group stages, producer→consumer
    const void* d_input;                 ///< the group's input buffer
    size_t      input_bytes;             ///< bytes of d_input
    void*       d_output;                ///< the group's MAIN output buffer (tail port 0, >= worst case)
    MemoryPool* pool;                    ///< pool for the runner's scratch
    fz::stream_t stream;                 ///< stream (runner may synchronise)
    /// Escaping side outputs of group members (e.g. outlier lists), in member/port
    /// order. nullptr or empty for single-output pipelines. The runner writes each
    /// `d_ptr` and fills each `size`; a runner that produces none leaves them alone.
    std::vector<FusedSideOutput>* side_outputs = nullptr;
};

/// A registered fused implementation and its matcher.
struct FusedImpl {
    const char* name;
    /// Eligible for normal `FusionPolicy::Auto` selection. False keeps a legal
    /// implementation available only under `Force` while it is being evaluated.
    bool auto_enabled;
    /// True if this impl handles `group` exactly (types + relevant config).
    bool   (*matches)(const std::vector<Stage*>& group);
    /// Run the fused compress; return the archive length written to d_output.
    size_t (*run)(const FusedRunContext& ctx);
};

/// First registered impl whose matcher accepts `group`, or nullptr. Experimental
/// implementations are returned only when `include_experimental` is true.
const FusedImpl* findFusedImpl(
    const std::vector<Stage*>& group, bool include_experimental = false);

} // namespace fz
