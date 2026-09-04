#include "advanced/fusion_registry.h"
#include "stage/stage.h"

#include "quantizers/quantizer/quantizer.h"
#include "predictors/lorenzo/lorenzo_stage.h"
#include "predictors/tiled_lorenzo/tiled_lorenzo_stage.h"
#include "coders/adaptive_bitpack/adaptive_bitpack_stage.h"
#include "coders/rze/rze_stage.h"
#include "shufflers/bitshuffle/bitshuffle_stage.h"
#include "predictors/diff/diff.h"
#include "fused/fused_block/nvrtc_warp_fusion.h"
#include "fused/chunk_fusion/nvrtc_chunk_fusion.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
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

    // 5. Size the outlier side buffers from the readback count and report the byte
    //    counts back to the producer, so its serializeHeader records the count the
    //    reused inverse will scatter (execute() was bypassed).  The staged quantizer
    //    rejects a capacity overflow: entries beyond the allocated side buffers are
    //    not recoverable, so silently clamping here would create an archive that
    //    decodes cleanly but violates its error bound.
    if (so_idxs && so_vals) {
        auto* quant = dynamic_cast<QuantizerStage<float, uint32_t>*>(g.front());
        if (outlier_count > side_max && quant && quant->getOutlierCapacity() != 0.0f) {
            const float actual_pct = n > 0
                ? 100.0f * static_cast<float>(outlier_count) / static_cast<float>(n)
                : 0.0f;
            const float capacity_pct = n > 0
                ? 100.0f * static_cast<float>(side_max) / static_cast<float>(n)
                : 0.0f;
            char msg[512];
            std::snprintf(
                msg, sizeof(msg),
                "QuantizerStage: outlier overflow — %u of %zu elements (%.1f%%) "
                "fell outside the quantizer radius, but outlier_capacity reserves "
                "only %.1f%%. Dropping the excess would violate the error bound. "
                "Raise outlier_capacity to at least %.2f, widen quant_radius, or "
                "loosen the error bound. Set outlier_capacity = 0 to opt into "
                "dropping outliers deliberately.",
                outlier_count, n, actual_pct, capacity_pct,
                std::min(1.0f, actual_pct * 1.1f / 100.0f));
            throw std::runtime_error(msg);
        }
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

// ── Generic chunk-cooperative inverse, first admitted shape: PFPL/RZE. ────────
// Stage order is inverse execution order: Coder -> reverse transforms -> Map.
// The kernel harness remains chunk-based and stage-agnostic internally; this
// matcher is deliberately evidence-gated to the exact RZE/PFPL operation set
// until additional inverse device ops have correctness/performance coverage.
bool matchesChunkCooperativeInverse(const std::vector<Stage*>& g) {
    if (g.size() != 4) return false;
    auto* rze = dynamic_cast<RZEStage*>(g[0]);
    auto* bs  = dynamic_cast<BitshuffleStage*>(g[1]);
    auto* df  = dynamic_cast<DifferenceStage<int32_t, uint32_t>*>(g[2]);
    auto* q   = dynamic_cast<QuantizerStage<float, uint32_t>*>(g[3]);
    return rze && bs && df && q &&
           rze->isInverse() && bs->isInverse() && df->isInverse() && q->isInverse() &&
           rze->getWordSize() == 1 && rze->getChunkSize() == 16384u &&
           bs->getElementWidth() == 4u && bs->getBlockSize() == 16384u &&
           df->getChunkSize() == 16384u && q->supportsChunkInverseFusion();
}

size_t runChunkCooperativeInverse(const FusedRunContext& ctx) {
    const auto& g = *ctx.stages;
    auto* rze = static_cast<RZEStage*>(g[0]);
    auto* q   = static_cast<QuantizerStage<float, uint32_t>*>(g[3]);

    const float* outlier_vals = nullptr;
    const uint32_t* outlier_idxs = nullptr;
    if (ctx.side_inputs) {
        for (const FusedSideInput& in : *ctx.side_inputs) {
            if (in.consumer != q) continue;
            if (in.input_index == 1) outlier_vals = static_cast<const float*>(in.d_ptr);
            else if (in.input_index == 2)
                outlier_idxs = static_cast<const uint32_t*>(in.d_ptr);
        }
    }

    const size_t output_bytes = static_cast<size_t>(rze->getCachedOrigBytes());
    const size_t written = fused::launchFusedChunkPfplInverse(
        static_cast<const uint8_t*>(ctx.d_input), ctx.input_bytes, output_bytes,
        2.0f * static_cast<float>(q->getComputedAbsEb()),
        q->getInplaceOutliers(), static_cast<uint32_t>(q->getQuantRadius()),
        outlier_vals, outlier_idxs, q->getActualOutlierCount(),
        static_cast<float*>(ctx.d_output), ctx.pool,
        static_cast<fz::stream_t>(ctx.stream));
    q->setFusedInverseResult(written);
    return written;
}

// ── Warp-register inverse (cuSZp / SZp decompress). Reverses the warp forward
//    chain in one warp-per-block kernel: coder decode -> block-local predictor
//    undelta (block-reset prefix sum) -> linear dequant, register-resident, no
//    DRAM round-trip for the intermediate codes/deltas. Inverse stage order is
//    Coder⁻¹ -> Predictor⁻¹ -> Quantizer⁻¹ (buildInverseDAG walks forward-reverse).
//    Bit-exact vs the staged inverse kernels.
//
//    ROLE-BASED, mirroring matchesWarpRegister/runWarpRegister on the compress
//    side: the stages declare WarpRegister inverse fused-ops
//    (getInverseFusionSpec/getInverseFusedOp) in the Cooperative / BlockLocal /
//    Map roles, and this matcher/runner read those declarations by role and build
//    a WarpFusionSpec from op names — no dynamic_cast to concrete stage types and
//    no hardcoded predictor/coder. A new warp predictor/coder that declares
//    forward+inverse ops fuses in BOTH directions with no edits here.
//    CORE scope: interior register transforms are not yet composed on the inverse
//    (the inverse harness has no invert() ops registered), so the chain is
//    exactly Coder -> Predictor -> Quant (size 3). See the CHANGELOG note.
bool matchesWarpRegisterInverse(const std::vector<Stage*>& g) {
    if (g.size() != 3) return false;                 // CORE: coder -> predictor -> quant
    for (Stage* s : g) {
        if (!s->isInverse()) return false;
        const FusedOpDecl op = s->getInverseFusedOp();
        if (!op.valid() || op.strategy != FusionStrategy::WarpRegister) return false;
    }
    const FusionSpec coder = g.front()->getInverseFusionSpec();  // reverse-forward order
    const FusionSpec pred  = g[1]->getInverseFusionSpec();
    const FusionSpec quant = g.back()->getInverseFusionSpec();
    if (coder.access != FusionAccess::Cooperative) return false;
    if (pred.access  != FusionAccess::BlockLocal)  return false;
    if (quant.access != FusionAccess::Map)         return false;
    // Coder and predictor must agree on the warp block size (Map/quant carries 0).
    return coder.block_size != 0 && pred.block_size == coder.block_size;
}

size_t runWarpRegisterInverse(const FusedRunContext& ctx) {
    const auto& g = *ctx.stages;
    Stage* coder = nullptr;
    Stage* predictor = nullptr;
    Stage* quant = nullptr;
    for (Stage* s : g) {
        switch (s->getInverseFusionSpec().access) {
            case FusionAccess::Cooperative: coder     = s; break;
            case FusionAccess::BlockLocal:  predictor = s; break;
            case FusionAccess::Map:         quant     = s; break;
            default: break;
        }
    }
    if (!coder || !predictor || !quant) return 0;   // matcher guarantees this

    const FusedOpDecl pdecl = predictor->getInverseFusedOp();
    fused::WarpFusionSpec spec;
    spec.coder          = coder->getInverseFusedOp().op_name;
    spec.predictor      = pdecl.op_name;
    spec.elems_per_lane = static_cast<int>(coder->getInverseFusedOp().elems_per_lane);

    // n_elems: the coder's block-covering count. 1-D: the natural count. Tiled
    // (cuSZp3): the padded tile-major count — so n_out (the predictor's natural
    // dx*dy*dz) is the real output size, and the predictor's geometry blob drives
    // the tile→natural scatter. A 1-D predictor declares no inverse op/params.
    const size_t n_elems = coder->getFusedInverseElementCount();
    const size_t n_out   = predictor->getFusedInverseElementCount();   // 0 for 1-D
    const float  ebx2    = static_cast<float>(quant->getFusedInverseDequantStep());

    const size_t written = fused::launchNvrtcWarpInverseFused(
        spec, static_cast<const uint8_t*>(ctx.d_input), ctx.input_bytes,
        n_elems, n_out, ebx2, pdecl.params.data(), pdecl.params.size(),
        static_cast<float*>(ctx.d_output), ctx.pool,
        static_cast<fz::stream_t>(ctx.stream));

    quant->setFusedInverseResult(written);
    return written;
}

const FusedImpl kBuiltins[] = {
    { "warp-register",  true,  &matchesWarpRegister,     &runWarpRegister     },
    { "chunk-coop",     true,  &matchesChunkCooperative, &runChunkCooperative },
    { "chunk-coop-inverse", true, &matchesChunkCooperativeInverse,
                                      &runChunkCooperativeInverse },
    { "warp-register-inverse", true, &matchesWarpRegisterInverse,
                                      &runWarpRegisterInverse },
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
