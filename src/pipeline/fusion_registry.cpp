#include "pipeline/fusion_registry.h"
#include "stage/stage.h"

#include "quantizers/quantizer/quantizer.h"
#include "predictors/lorenzo/lorenzo_stage.h"
#include "predictors/tiled_lorenzo/tiled_lorenzo_stage.h"
#include "predictors/diff/diff.h"
#include "shufflers/bitshuffle/bitshuffle_stage.h"
#include "coders/adaptive_bitpack/adaptive_bitpack_stage.h"
#include "coders/rze/rze_stage.h"
#include "coders/rre/rre_stage.h"
#include "fused/fused_block/fused_block.h"
#include "fused/chunk_fusion/chunk_fusion.h"

#include <cmath>
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

// ── PFPL (chunk-cooperative): Quantizer(inplace,zigzag,ABS/NOA) + Difference
//    (int32->uint32, negabinary, chunk 16 KB) + Bitshuffle(ew4,16 KB) + {RZE|RRE}.
//    The coder is swappable — the same fused harness composes either.
bool matchesPfpl(const std::vector<Stage*>& g) {
    if (g.size() != 4) return false;
    auto* q = dynamic_cast<QuantizerStage<float, uint32_t>*>(g[0]);
    auto* d = dynamic_cast<DifferenceStage<int32_t, uint32_t>*>(g[1]);
    auto* b = dynamic_cast<BitshuffleStage*>(g[2]);
    if (!q || !d || !b) return false;
    if (!q->getFusionSpec().fusable() || !q->getInplaceOutliers() || !q->getZigzagCodes())
        return false;
    const auto em = q->getErrorBoundMode();
    if (em != ErrorBoundMode::ABS && em != ErrorBoundMode::NOA) return false;
    if (!d->getFusionSpec().fusable() || !b->getFusionSpec().fusable()) return false;
    auto* rze = dynamic_cast<RZEStage*>(g[3]);
    auto* rre = dynamic_cast<RREStage*>(g[3]);
    if (rze) return rze->getFusionSpec().fusable();
    if (rre) return rre->getFusionSpec().fusable();
    return false;
}

size_t runPfpl(const FusedRunContext& ctx) {
    const auto& g = *ctx.stages;
    auto* q   = static_cast<QuantizerStage<float, uint32_t>*>(g[0]);
    auto* rze = dynamic_cast<RZEStage*>(g[3]);
    auto* rre = dynamic_cast<RREStage*>(g[3]);
    const size_t n = ctx.input_bytes / sizeof(float);

    // Prime the quant's forward-computed abs bound (covers the NOA value-range
    // scan) so the fused kernel's scale AND the reused inverse/header agree.
    q->primeComputedAbsEb(ctx.d_input, n, ctx.pool, static_cast<fz::stream_t>(ctx.stream));
    const float ebx2_r    = 1.0f / (2.0f * static_cast<float>(q->getComputedAbsEb()));
    const uint32_t radius = static_cast<uint32_t>(q->getQuantRadius());
    const float threshold = q->getOutlierThreshold();

    const auto coder = rze ? fused::ChunkCoderKind::RZE : fused::ChunkCoderKind::RRE;
    const size_t archive_bytes = fused::launchFusedChunkPfpl(
        coder, static_cast<const float*>(ctx.d_input), n, ebx2_r, radius, threshold,
        static_cast<uint8_t*>(ctx.d_output), ctx.pool, static_cast<fz::stream_t>(ctx.stream));

    // Original (uncompressed) coder input = the bitshuffle output = n*4 bytes; the
    // inverse sizes its output from this.
    if (rze) rze->setFusedResult(archive_bytes, ctx.input_bytes);
    else     rre->setFusedResult(archive_bytes, ctx.input_bytes);
    return archive_bytes;
}

const FusedImpl kBuiltins[] = {
    { "cuszp2", &matchesCuszp2, &runCuszp2 },
    { "cuszp3", &matchesCuszp3, &runCuszp3 },
    { "pfpl",   &matchesPfpl,   &runPfpl   },
};

} // namespace

const FusedImpl* findFusedImpl(const std::vector<Stage*>& group) {
    for (const auto& impl : kBuiltins)
        if (impl.matches(group)) return &impl;
    return nullptr;
}

} // namespace fz
