#include "pipeline/fusion_registry.h"
#include "stage/stage.h"

#include "quantizers/quantizer/quantizer.h"
#include "predictors/lorenzo/lorenzo_stage.h"
#include "predictors/tiled_lorenzo/tiled_lorenzo_stage.h"
#include "coders/adaptive_bitpack/adaptive_bitpack_stage.h"
#include "fused/fused_block/fused_block.h"
#include "fused/chunk_fusion/nvrtc_chunk_fusion.h"

#include <cstdint>
#include <vector>

namespace fz {

namespace {

// ── Warp-register (cuSZp): Quantizer(linear,ABS) -> block-local predictor ->
//    AdaptiveBitpack(outlier). One generic entry replaces the former per-shape
//    cuszp2/cuszp3 matchers: the stages declare WarpRegister fused-ops (the
//    predictor op name selects the fused driver), the planner guarantees the
//    Map -> BlockLocal -> Cooperative ordering, and the runner dispatches on the
//    predictor. Adding a predictor = a policy + a launcher + one dispatch case,
//    not a new registry entry. Unlike the chunk strategy this is not NVRTC-
//    composed: the warp driver is a fixed kernel skeleton with few knobs, so the
//    coder is always AdaptiveBitpack and the launchers take typed args.
bool matchesWarpRegister(const std::vector<Stage*>& g) {
    if (g.size() != 3) return false;               // Quant -> Predictor -> AdaptiveBitpack
    for (Stage* s : g) {
        const FusedOpDecl op = s->getFusedOp();
        if (!op.valid() || op.strategy != FusionStrategy::WarpRegister) return false;
    }
    return g[0]->getFusionSpec().access == FusionAccess::Map &&
           g[1]->getFusionSpec().access == FusionAccess::BlockLocal &&
           g[2]->getFusionSpec().access == FusionAccess::Cooperative;
}

size_t runWarpRegister(const FusedRunContext& ctx) {
    const auto& g = *ctx.stages;

    // Prime forward-computed state the reused inverse stages read (the quant abs eb).
    const FusedPrimeContext pc{ ctx.d_input, ctx.input_bytes, ctx.pool,
                                static_cast<fz::stream_t>(ctx.stream) };
    for (Stage* s : g) s->primeFusedForwardState(pc);

    auto* q = static_cast<QuantizerStage<float, uint32_t>*>(g[0]);
    auto* a = static_cast<AdaptiveBitpackStage<int32_t>*>(g[2]);
    const float eb = static_cast<float>(q->getErrorBound());

    // Dispatch on the declared predictor op → the matching fused driver. The
    // driver is a compile-time <EPL, Predictor> template, so this maps the op name
    // to its pre-instantiated launcher (the residual O(predictor) glue).
    const std::string pred = g[1]->getFusedOp().op_name;
    size_t archive_bytes = 0, n_ab = 0;
    if (pred == "Lorenzo1DPredictor") {
        const size_t n = ctx.input_bytes / sizeof(float);
        archive_bytes = fused::launchFusedCuszp2Compress(
            static_cast<const float*>(ctx.d_input), n, eb,
            a->getBlockSize(), a->getOutlierSelection(),
            static_cast<uint8_t*>(ctx.d_output), ctx.pool,
            static_cast<fz::stream_t>(ctx.stream));
        n_ab = n;
    } else if (pred == "TiledLorenzo2DPredictor") {
        auto* l = static_cast<TiledLorenzoStage<int32_t>*>(g[1]);
        const auto dims = l->getDims();
        const auto tile = l->getTileShape();
        archive_bytes = fused::launchFusedCuszp3Compress(
            static_cast<const float*>(ctx.d_input), dims[0], dims[1], eb,
            tile[0], tile[1], static_cast<uint8_t*>(ctx.d_output),
            ctx.pool, static_cast<fz::stream_t>(ctx.stream));
        const size_t ntx = (dims[0] + tile[0] - 1) / tile[0];
        const size_t nty = (dims[1] + tile[1] - 1) / tile[1];
        n_ab = ntx * nty * tile[0] * tile[1];
    } else {
        return 0;   // unknown predictor — matcher shouldn't have let this through
    }

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

    // 3. NVRTC-compose + launch + shared scan/pack tail.
    const size_t n = ctx.input_bytes / sizeof(float);
    const size_t archive_bytes = fused::launchGenericChunkFusion(
        spec, static_cast<const float*>(ctx.d_input), n, blob.data(), blob.size(),
        static_cast<uint8_t*>(ctx.d_output), ctx.pool, static_cast<fz::stream_t>(ctx.stream));

    // 4. Tail coder: prime its inverse output sizing (see CN-CHUNK-WIRE). Original
    //    (uncompressed) coder input = n*4 bytes = ctx.input_bytes.
    if (coder) coder->setFusedArchiveResult(archive_bytes, ctx.input_bytes);
    return archive_bytes;
}

const FusedImpl kBuiltins[] = {
    { "warp-register", &matchesWarpRegister,     &runWarpRegister     },
    { "chunk-coop",    &matchesChunkCooperative, &runChunkCooperative },
};

} // namespace

const FusedImpl* findFusedImpl(const std::vector<Stage*>& group) {
    for (const auto& impl : kBuiltins)
        if (impl.matches(group)) return &impl;
    return nullptr;
}

} // namespace fz
