#include "advanced/fusion_registry.h"
#include "stage/stage.h"

#include "quantizers/quantizer/quantizer.h"
#include "predictors/lorenzo/lorenzo_stage.h"
#include "predictors/tiled_lorenzo/tiled_lorenzo_stage.h"
#include "coders/adaptive_bitpack/adaptive_bitpack_stage.h"
#include "fused/fused_block/nvrtc_warp_fusion.h"
#include "fused/chunk_fusion/nvrtc_chunk_fusion.h"

#include <cstdint>
#include <cstring>
#include <vector>

namespace fz {

namespace {

// ── Warp-register (cuSZp): Quantizer(linear,ABS) -> block-local predictor ->
//    AdaptiveBitpack(outlier). One generic entry replaces the former per-shape
//    cuszp2/cuszp3 matchers: the stages declare WarpRegister fused-ops (the
//    predictor op name selects the fused driver), the planner guarantees the
//    Map -> BlockLocal -> Cooperative ordering, and the runner dispatches on the
//    predictor. Like the chunk strategy this is now NVRTC-composed: the predictor
//    stage declares its device-policy op + packed params + shape (elems_per_lane,
//    n_ab) via getFusedOp(), and the generic runner builds a WarpFusionSpec and
//    calls one launcher — no per-predictor dispatch or pre-instantiated launcher.
//    Adding a predictor, a transform, or a coder = write its policy in warp_fusion.cuh
//    + declare it on the stage. The chain is Quant(Map) -> Predictor(BlockLocal) ->
//    {Map|BlockLocal transforms}* -> Coder(Cooperative), all WarpRegister — the register
//    analogue of the chunk chain.
bool matchesWarpRegister(const std::vector<Stage*>& g) {
    if (g.size() < 3) return false;                // Quant -> Predictor -> ... -> Coder
    for (Stage* s : g) {
        const FusedOpDecl op = s->getFusedOp();
        if (!op.valid() || op.strategy != FusionStrategy::WarpRegister) return false;
    }
    if (g.front()->getFusionSpec().access != FusionAccess::Map)          return false; // quant
    if (g[1]->getFusionSpec().access      != FusionAccess::BlockLocal)   return false; // predictor
    if (g.back()->getFusionSpec().access  != FusionAccess::Cooperative)  return false; // coder
    // Interior stages (between predictor and coder) are register→register transforms.
    for (size_t i = 2; i + 1 < g.size(); ++i) {
        const FusionAccess a = g[i]->getFusionSpec().access;
        if (a != FusionAccess::Map && a != FusionAccess::BlockLocal) return false;
    }
    return true;
}

size_t runWarpRegister(const FusedRunContext& ctx) {
    const auto& g = *ctx.stages;

    // Prime forward-computed state the reused inverse stages read (the quant abs eb).
    const FusedPrimeContext pc{ ctx.d_input, ctx.input_bytes, ctx.pool,
                                static_cast<fz::stream_t>(ctx.stream) };
    for (Stage* s : g) s->primeFusedForwardState(pc);

    auto* q = static_cast<QuantizerStage<float, uint32_t>*>(g.front());
    auto* a = static_cast<AdaptiveBitpackStage<int32_t>*>(g.back());
    // Resolved absolute bound after priming — ABS: = error_bound; NOA: = eb*range
    // (padding-excluded, see primeComputedAbsEb). The fused kernel quantizes with this
    // and the reused inverse reconstructs with the same computed_abs_eb_, so ABS and
    // NOA share one uniform-step fused path.
    const float eb     = static_cast<float>(q->getComputedAbsEb());
    const float inv2eb = 1.0f / (2.0f * eb);

    // Build the warp spec + params blob from the stages' own declarations — no
    // per-predictor dispatch. Chain positions: g[0] quant (absorbed into the predictor
    // policy), g[1] predictor, g[2..n-2] transforms, g[n-1] coder. The predictor packs
    // its geometry with a leading inv2eb slot (offset 0) it cannot fill (the quantizer
    // owns the bound); patch it here.
    const FusedOpDecl decl = g[1]->getFusedOp();
    fused::WarpFusionSpec spec;
    spec.predictor      = decl.op_name;
    spec.coder          = g.back()->getFusedOp().op_name;   // swappable Cooperative sink
    spec.elems_per_lane = static_cast<int>(decl.elems_per_lane);
    for (size_t i = 2; i + 1 < g.size(); ++i)              // register→register transforms
        spec.transforms.push_back(g[i]->getFusedOp().op_name);
    std::vector<uint8_t> blob = decl.params;
    if (blob.size() >= sizeof(float)) std::memcpy(blob.data(), &inv2eb, sizeof(float));

    // n_ab: the predictor's padded block-covering count, or the input element count
    // when it declares 0 (1-D needs no padding).
    const size_t n_ab = decl.n_ab ? decl.n_ab : ctx.input_bytes / sizeof(float);
    const size_t archive_bytes = fused::launchNvrtcWarpFused(
        spec, static_cast<const float*>(ctx.d_input), n_ab,
        blob.data(), blob.size(),
        static_cast<uint8_t*>(ctx.d_output), ctx.pool, static_cast<fz::stream_t>(ctx.stream));

    // The archive masquerades as the staged AdaptiveBitpack output: set the tail
    // stage's execute-time state (num_elements = the padded tile-major count) so
    // buildHeader() and the DAG's output sizing see a normal AB result.
    a->setFusedResult(n_ab, archive_bytes);
    return archive_bytes;
}

// ── Generic chunk-cooperative fusion. Composes ANY linear
//    Map -> Transform* -> Coder chain of ChunkCooperative device-ops from the
//    stages' own getFusedOp() declarations — no per-pipeline shape hard-coded.
//    PFPL (Quant-inplace-zigzag -> Difference-negabinary -> Bitshuffle -> {RZE|
//    RRE|RARE|RAZE...}) is just one instance; a novel compatible chain a user
//    assembles fuses with zero new registry code. The planner already guarantees
//    the group is strictly linear, same-block-size, and coder-terminated; this
//    checks every member is a ChunkCooperative op with a single Map head and
//    Coder tail (the harness's Map op is the global-memory loader).
bool matchesChunkCooperative(const std::vector<Stage*>& g) {
    if (g.size() < 2) return false;                 // need a Map head + a Coder tail
    int maps = 0, coders = 0;
    for (Stage* s : g) {
        const FusedOpDecl op = s->getFusedOp();
        if (!op.valid() || op.strategy != FusionStrategy::ChunkCooperative) return false;
        switch (s->getFusionSpec().access) {
            case FusionAccess::Map:         ++maps;   break;
            case FusionAccess::Cooperative: ++coders; break;
            default:                                  break;
        }
    }
    return maps == 1 && coders == 1 &&
           g.front()->getFusionSpec().access == FusionAccess::Map &&
           g.back()->getFusionSpec().access  == FusionAccess::Cooperative;
}

size_t runChunkCooperative(const FusedRunContext& ctx) {
    const auto& g = *ctx.stages;

    // 1. Prime each stage's forward-computed state its own inverse will read (the
    //    runner bypasses execute()) — e.g. the quantizer's NOA value-range scan.
    const FusedPrimeContext pc{ ctx.d_input, ctx.input_bytes, ctx.pool,
                                static_cast<fz::stream_t>(ctx.stream) };
    for (Stage* s : g) s->primeFusedForwardState(pc);

    // 2. Assemble the fused spec + packed params blob from the ops themselves, by
    //    role (FusionSpec.access). Stage order IS execution order, so the blob is
    //    naturally [Map][Transforms...][Coder] — the order the kernel expects.
    fused::ChunkFusionSpec spec;
    spec.transforms.clear();
    std::vector<uint8_t> blob;
    Stage* coder = nullptr;
    for (Stage* s : g) {
        const FusedOpDecl op = s->getFusedOp();
        switch (s->getFusionSpec().access) {
            case FusionAccess::Map:         spec.quant_op = op.op_name;            break;
            case FusionAccess::Cooperative: spec.coder    = op.op_name; coder = s; break;
            default:                        spec.transforms.push_back(op.op_name); break;
        }
        blob.insert(blob.end(), op.params.begin(), op.params.end());
    }

    // 3. Split-outlier producer? The quant Map declares "QuantSplitOutlier" and its
    //    outlier ports arrive as named escaping side outputs. Hand the pre-allocated buffers to the
    //    launcher; it fills them and reports the outlier count. Absent side outputs
    //    (the common single-output chain), these stay null and the launcher no-ops it.
    FusedSideOutput* so_idxs = nullptr;
    FusedSideOutput* so_vals = nullptr;
    if (ctx.side_outputs) {
        for (auto& so : *ctx.side_outputs) {
            if (so.declaration.name == "outlier_vals")      so_vals = &so;
            else if (so.declaration.name == "outlier_idxs") so_idxs = &so;
        }
    }
    uint32_t* d_side_idxs = nullptr;
    float*    d_side_vals = nullptr;
    uint32_t  side_max    = 0;
    if (so_idxs && so_vals) {
        d_side_idxs = static_cast<uint32_t*>(so_idxs->d_ptr);
        d_side_vals = static_cast<float*>(so_vals->d_ptr);
        const uint32_t cap_i = static_cast<uint32_t>(so_idxs->capacity / sizeof(uint32_t));
        const uint32_t cap_v = static_cast<uint32_t>(so_vals->capacity / sizeof(float));
        side_max = cap_i < cap_v ? cap_i : cap_v;   // both hold `max` elements
    }

    // 4. NVRTC-compose + launch + shared scan/pack tail (fills side buffers if any).
    const size_t n = ctx.input_bytes / sizeof(float);
    uint32_t outlier_count = 0;
    const size_t archive_bytes = fused::launchGenericChunkFusion(
        spec, static_cast<const float*>(ctx.d_input), n, blob.data(), blob.size(),
        static_cast<uint8_t*>(ctx.d_output), ctx.pool, static_cast<fz::stream_t>(ctx.stream),
        d_side_idxs, d_side_vals, side_max, &outlier_count);

    // 5. Size the outlier side buffers from the readback count (clamped to capacity —
    //    overflow past `side_max` is dropped, same as the staged max_outliers path) and
    //    report the byte counts back to the producer, so its serializeHeader records the
    //    outlier count the reused inverse will scatter (execute() was bypassed).
    if (so_idxs && so_vals) {
        const uint32_t written = outlier_count < side_max ? outlier_count : side_max;
        so_idxs->size = static_cast<size_t>(written) * sizeof(uint32_t);
        so_vals->size = static_cast<size_t>(written) * sizeof(float);
        so_idxs->producer->setFusedSideOutput(so_idxs->output_index, so_idxs->size);
        so_vals->producer->setFusedSideOutput(so_vals->output_index, so_vals->size);
    }

    // 6. Tail coder: prime its inverse output sizing (see CN-CHUNK-WIRE). Original
    //    (uncompressed) coder input = n*4 bytes = ctx.input_bytes.
    if (coder) coder->setFusedArchiveResult(archive_bytes, ctx.input_bytes);
    return archive_bytes;
}

const FusedImpl kBuiltins[] = {
    { "warp-register",  true,  &matchesWarpRegister,     &runWarpRegister     },
    { "chunk-coop",     true,  &matchesChunkCooperative, &runChunkCooperative },
};

} // namespace

const FusedImpl* findFusedImpl(
    const std::vector<Stage*>& group, bool include_experimental) {
    for (const auto& impl : kBuiltins)
        if ((impl.auto_enabled || include_experimental) && impl.matches(group))
            return &impl;
    return nullptr;
}

} // namespace fz
