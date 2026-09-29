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
 * installer queries this registry for each contiguous subspan: "does a registered
 * specialization strategy accept these stage declarations?". An eligible hit lets
 * the DAG executor generate and run a fused kernel in place of the group's staged
 * execute()s; misses and unselected overlaps remain staged. Normal Auto selection
 * considers only auto-enabled strategies: registration is the current evidence
 * gate, not a general predictive per-input profitability test. Compatible device
 * operations can form new generated chains within a strategy without adding a
 * per-chain registry entry.
 *
 * See docs/codebase_notes.md CN-FUSE-PROOF / CN-FUSE-PLAN.
 */

#include "backend/types.h"
#include "stage/fusion.h"
#include <cstddef>
#include <string>
#include <vector>

namespace fz {

class Stage;
class MemoryPool;

/// A group member's escaping output port — a side output (e.g. an outlier list)
/// the fused kernel produces in addition to the main archive. It may be a
/// pipeline leaf or continue through staged consumers outside the specialized
/// region. The runner writes `d_ptr` and reports the bytes it wrote in `size`;
/// the DAG then sizes the boundary buffer. Empty for the common single-output
/// case.
struct FusedSideOutput {
    Stage* producer;       ///< the group member that owns this output port
    int    output_index;   ///< which of the producer's output ports this is
    void*  d_ptr;          ///< pre-allocated buffer for the port
    size_t capacity;       ///< its allocated size in bytes
    size_t size;           ///< OUT: bytes the runner wrote (DAG sets the buffer size from this)
    FusedAuxOutputDecl declaration; ///< semantic identity/sizing rule, if declared
};

/// An input entering a fused group from outside its main linear chain. Forward
/// fusion currently only needs one main input, while inverse groups may also
/// consume archive side streams (for example quantizer outlier values/indices).
struct FusedSideInput {
    Stage* consumer;       ///< group member that owns this input
    int    input_index;    ///< position in that stage's inverse input list
    const void* d_ptr;     ///< externally produced/device-resident input
    size_t size;           ///< bytes available at d_ptr
};

/// Everything a fused runner needs to compress one group in place.
struct FusedRunContext {
    const std::vector<Stage*>* stages;   ///< group stages, producer→consumer
    const void* d_input;                 ///< the group's input buffer
    size_t      input_bytes;             ///< bytes of d_input (logical, chunk-aligned)
    /// Bytes of real data at d_input when it is shorter than input_bytes: the pipeline
    /// skipped the zero-padded copy (FusedImpl::accepts_unpadded_input). Reading at or
    /// past it is out of bounds; the runner must treat that tail as zeros. 0 = input_bytes.
    size_t      input_valid_bytes = 0;
    void*       d_output;                ///< the group's MAIN output buffer (tail port 0, >= worst case)
    size_t      output_capacity = 0;     ///< allocated bytes available at d_output
    MemoryPool* pool;                    ///< pool for the runner's scratch
    fz::stream_t stream;                 ///< stream (runner may synchronise)
    /// Escaping side outputs of group members (e.g. outlier lists), in member/port
    /// order. nullptr or empty for single-output pipelines. The runner writes each
    /// `d_ptr` and fills each `size`; a runner that produces none leaves them alone.
    std::vector<FusedSideOutput>* side_outputs = nullptr;
    /// Non-main inputs entering any member from outside the fused chain. Used by
    /// inverse fusion for side archives; empty for ordinary linear compression.
    const std::vector<FusedSideInput>* side_inputs = nullptr;
    /// Optional per-group diagnostic populated by runners whose implementation
    /// selects among materially different internal execution paths at runtime.
    std::string* execution_path = nullptr;
};

/// A registered fused implementation and its matcher.
struct FusedImpl {
    const char* name;
    /// Eligible for normal `FusionPolicy::Auto` selection. False keeps a legal
    /// implementation available only under `Force` while it is being evaluated.
    bool auto_enabled;
    /// True if this strategy handles the group's declarations and geometry.
    bool   (*matches)(const std::vector<Stage*>& group);
    /// Run the fused compress; return the archive length written to d_output.
    size_t (*run)(const FusedRunContext& ctx);
    /// True if `run` honours FusedRunContext::input_valid_bytes (reads nothing past it
    /// and treats the rest of input_bytes as zeros), so the pipeline may hand it the
    /// caller's unpadded input instead of a zero-padded copy.
    bool accepts_unpadded_input = false;
};

/// First registered impl whose matcher accepts `group`, or nullptr. Experimental
/// implementations are returned only when `include_experimental` is true.
const FusedImpl* findFusedImpl(
    const std::vector<Stage*>& group, bool include_experimental = false);

} // namespace fz
