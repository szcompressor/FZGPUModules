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

// ── cuSZp2: Quantizer(linear,ABS) + Lorenzo(1-D,32) + AdaptiveBitpack(32,outlier).
bool matchesCuszp2(const std::vector<Stage*>& g) {
    if (g.size() != 3) return false;
    auto* q  = dynamic_cast<QuantizerStage<float, uint32_t>*>(g[0]);
    auto* l  = dynamic_cast<LorenzoStage<int32_t>*>(g[1]);
    auto* a  = dynamic_cast<AdaptiveBitpackStage<int32_t>*>(g[2]);
    if (!q || !l || !a) return false;
    if (!q->getFusionSpec().fusable() || q->getErrorBoundMode() != ErrorBoundMode::ABS) return false;
    if (l->getBlockSize() != 32u) return false;
    if (a->getBlockSize() != 32u || !a->getOutlierSelection()) return false;
    return true;
}

size_t runCuszp2(const FusedRunContext& ctx) {
    const auto& g = *ctx.stages;
    auto* q = static_cast<QuantizerStage<float, uint32_t>*>(g[0]);
    auto* a = static_cast<AdaptiveBitpackStage<int32_t>*>(g[2]);
    const float eb = static_cast<float>(q->getErrorBound());
    const size_t n = ctx.input_bytes / sizeof(float);

    // The reused stages are used verbatim for decompress; establish the
    // forward-computed state their inverse depends on (the quant's abs eb —
    // otherwise the inverse reconstructs with the default bound).
    q->primeAbsEbForFusion();

    const size_t archive_bytes = fused::launchFusedCuszp2Compress(
        static_cast<const float*>(ctx.d_input), n, eb,
        a->getBlockSize(), a->getOutlierSelection(),
        static_cast<uint8_t*>(ctx.d_output), ctx.pool,
        static_cast<fz::stream_t>(ctx.stream));

    // The archive masquerades as the staged AdaptiveBitpack output: set the tail
    // stage's execute-time state so buildHeader() (num_elements) and the DAG's
    // output sizing see a normal AB result.
    a->setFusedResult(n, archive_bytes);
    return archive_bytes;
}

// ── cuSZp3: Quantizer(linear,ABS) + TiledLorenzo(2-D 8x8) + AdaptiveBitpack(64,outlier).
bool matchesCuszp3(const std::vector<Stage*>& g) {
    if (g.size() != 3) return false;
    auto* q = dynamic_cast<QuantizerStage<float, uint32_t>*>(g[0]);
    auto* l = dynamic_cast<TiledLorenzoStage<int32_t>*>(g[1]);
    auto* a = dynamic_cast<AdaptiveBitpackStage<int32_t>*>(g[2]);
    if (!q || !l || !a) return false;
    if (!q->getFusionSpec().fusable() || q->getErrorBoundMode() != ErrorBoundMode::ABS) return false;
    const auto tile = l->getTileShape();
    if (tile[2] != 1u || tile[0] * tile[1] != 64u) return false;   // 2-D, block-64 driver only
    if (a->getBlockSize() != 64u || !a->getOutlierSelection()) return false;
    return true;
}

size_t runCuszp3(const FusedRunContext& ctx) {
    const auto& g = *ctx.stages;
    auto* q = static_cast<QuantizerStage<float, uint32_t>*>(g[0]);
    auto* l = static_cast<TiledLorenzoStage<int32_t>*>(g[1]);
    auto* a = static_cast<AdaptiveBitpackStage<int32_t>*>(g[2]);
    const float eb  = static_cast<float>(q->getErrorBound());
    const auto dims = l->getDims();
    const auto tile = l->getTileShape();

    q->primeAbsEbForFusion();   // see runCuszp2 — the inverse reuses this stage

    const size_t archive_bytes = fused::launchFusedCuszp3Compress(
        static_cast<const float*>(ctx.d_input), dims[0], dims[1], eb,
        tile[0], tile[1], static_cast<uint8_t*>(ctx.d_output),
        ctx.pool, static_cast<fz::stream_t>(ctx.stream));

    // The tail AB masquerades over the padded tile-major element count.
    const size_t ntx  = (dims[0] + tile[0] - 1) / tile[0];
    const size_t nty  = (dims[1] + tile[1] - 1) / tile[1];
    const size_t n_ab = ntx * nty * tile[0] * tile[1];
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
    { "cuszp2",     &matchesCuszp2,           &runCuszp2           },
    { "cuszp3",     &matchesCuszp3,           &runCuszp3           },
    { "chunk-coop", &matchesChunkCooperative, &runChunkCooperative },
};

} // namespace

const FusedImpl* findFusedImpl(const std::vector<Stage*>& group) {
    for (const auto& impl : kBuiltins)
        if (impl.matches(group)) return &impl;
    return nullptr;
}

} // namespace fz
