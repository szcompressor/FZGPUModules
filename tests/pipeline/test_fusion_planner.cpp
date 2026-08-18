// Unit tests for the fusion planner: does it identify the fusable stage chains
// in a DAG that a fused kernel could collapse? Pure analysis — no execution.

#include "fzgpumodules.h"
#include "pipeline/fusion_planner.h"
#include "fused/chunk_fusion/nvrtc_chunk_fusion.h"
#include "fused/chunk_fusion/chunk_op_params.h"

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include <functional>
#include <vector>

using namespace fz;

namespace {

// Build the cuszp2 fast pipeline: Quantizer(linear) -> Lorenzo(32) -> AdaptiveBitpack(32).
void buildCuszp2(Pipeline& p, size_t n) {
    p.setDims(n, 1, 1);
    auto* q = p.addStage<QuantizerStage<float, uint32_t>>();
    q->setErrorBound(1e-3f); q->setErrorBoundMode(ErrorBoundMode::ABS); q->setLinearMode(true);
    auto* l = p.addStage<LorenzoStage<int32_t>>(); l->setBlockSize(32);
    p.connect(l, q, "codes");
    auto* a = p.addStage<AdaptiveBitpackStage<int32_t>>();
    a->setBlockSize(32); a->setOutlierSelection(true);
    p.connect(a, l);
}

// Build the cuszp3 fast pipeline: Quantizer(linear) -> TiledLorenzo(8x8) ->
// AdaptiveBitpack(block=64). Matches examples/presets/cuszp3_outlier.toml.
void buildCuszp3(Pipeline& p, size_t dx, size_t dy) {
    p.setDims(dx, dy, 1);
    auto* q = p.addStage<QuantizerStage<float, uint32_t>>();
    q->setErrorBound(1e-3f); q->setErrorBoundMode(ErrorBoundMode::ABS); q->setLinearMode(true);
    auto* tl = p.addStage<TiledLorenzoStage<int32_t>>(); tl->setTileShape(8, 8);
    p.connect(tl, q, "codes");
    auto* a = p.addStage<AdaptiveBitpackStage<int32_t>>();
    a->setBlockSize(64); a->setOutlierSelection(true);
    p.connect(a, tl);
}

} // namespace

// The whole cuszp2 front is one block-local fusable group ending in the coder.
TEST(FusionPlanner, Cuszp2ChainIsOneGroup) {
    Pipeline p(1024 * sizeof(float), MemoryStrategy::PREALLOCATE, 2.0f);
    buildCuszp2(p, 1024);
    p.finalize();

    auto groups = planFusionGroups(*p.getDAG());
    ASSERT_EQ(groups.size(), 1u);
    EXPECT_EQ(groups[0].stages.size(), 3u);
    EXPECT_EQ(groups[0].block_size, 32u);
    EXPECT_TRUE(groups[0].has_coder);
}

// A quantizer in the default (outlier) mode is not a pure Map, so it is not
// fusable and the chain does not form.
TEST(FusionPlanner, OutlierQuantizerIsNotFusable) {
    Pipeline p(1024 * sizeof(float), MemoryStrategy::PREALLOCATE, 2.0f);
    p.setDims(1024, 1, 1);
    auto* q = p.addStage<QuantizerStage<float, uint32_t>>();
    q->setErrorBound(1e-3f); q->setErrorBoundMode(ErrorBoundMode::ABS);  // outlier mode (not linear)
    auto* l = p.addStage<LorenzoStage<int32_t>>(); l->setBlockSize(32);
    p.connect(l, q, "codes");
    p.finalize();

    // Lorenzo alone can't form a group of >=2 without a fusable neighbour.
    auto groups = planFusionGroups(*p.getDAG());
    for (const auto& g : groups)
        EXPECT_GE(g.stages.size(), 2u);
    // The quantizer must not appear in any group.
    for (const auto& g : groups)
        for (auto* s : g.stages)
            EXPECT_NE(s, static_cast<Stage*>(q));
}

// Stages declare their own fusion access correctly.
TEST(FusionPlanner, StageFusionSpecs) {
    QuantizerStage<float, uint32_t> qlin;
    qlin.setLinearMode(true);
    EXPECT_EQ(qlin.getFusionSpec().access, FusionAccess::Map);

    QuantizerStage<float, uint32_t> qout;  // default: outlier
    EXPECT_EQ(qout.getFusionSpec().access, FusionAccess::Unfusable);

    LorenzoStage<int32_t> lz; lz.setBlockSize(32);
    EXPECT_EQ(lz.getFusionSpec().access, FusionAccess::BlockLocal);
    EXPECT_EQ(lz.getFusionSpec().block_size, 32u);

    LorenzoStage<int32_t> lznd;  // N-D default (block_size 0) — not fused yet
    EXPECT_EQ(lznd.getFusionSpec().access, FusionAccess::Unfusable);

    AdaptiveBitpackStage<int32_t> ab; ab.setBlockSize(32);
    EXPECT_EQ(ab.getFusionSpec().access, FusionAccess::Cooperative);
    EXPECT_EQ(ab.getFusionSpec().block_size, 32u);
}

// End-to-end: fusion (Auto) must produce a byte-identical archive to the staged
// path and round-trip within the error bound. This is the correctness gate for
// the executor substitution.
TEST(FusionPlanner, EndToEndFusedMatchesStaged) {
    const size_t n = 1u << 20;   // 1 Mi elements
    const float  eb = 1e-3f;
    std::vector<float> h(n);
    for (size_t i = 0; i < n; ++i)                     // smooth, in-range for eb=1e-3
        h[i] = 0.5f * std::sin(i * 0.001f) + 0.2f * std::cos(i * 0.017f);
    const size_t bytes = n * sizeof(float);
    float* d_in = nullptr; ASSERT_EQ(cudaMalloc(&d_in, bytes), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_in, h.data(), bytes, cudaMemcpyHostToDevice), cudaSuccess);

    auto compressCopy = [&](FusionPolicy pol, std::vector<uint8_t>& out) -> size_t {
        Pipeline p(bytes, MemoryStrategy::PREALLOCATE, 2.0f);
        p.setFusionPolicy(pol);
        buildCuszp2(p, n);
        p.finalize();
        EXPECT_EQ(p.getFusedGroupCount(), pol == FusionPolicy::Auto ? 1u : 0u);
        void* d_comp = nullptr; size_t sz = 0;
        p.compress(d_in, bytes, &d_comp, &sz, 0);
        cudaDeviceSynchronize();
        out.resize(sz);
        EXPECT_EQ(cudaMemcpy(out.data(), d_comp, sz, cudaMemcpyDeviceToHost), cudaSuccess);
        return sz;
    };

    std::vector<uint8_t> staged, fused;
    size_t sz_staged = compressCopy(FusionPolicy::Off, staged);
    size_t sz_fused  = compressCopy(FusionPolicy::Auto, fused);

    ASSERT_EQ(sz_staged, sz_fused) << "fused archive size differs from staged";
    EXPECT_EQ(staged, fused) << "fused archive is not byte-identical to staged";

    // Round-trip each on its own pipeline (compress+decompress together) and
    // compare, isolating whether the staged baseline itself round-trips.
    auto roundtrip = [&](FusionPolicy pol, std::vector<float>& recon) {
        Pipeline p(bytes, MemoryStrategy::PREALLOCATE, 2.0f);
        p.setFusionPolicy(pol);
        buildCuszp2(p, n);
        p.finalize();
        void* d_comp = nullptr; size_t sz = 0;
        p.compress(d_in, bytes, &d_comp, &sz, 0);
        void* d_decomp = nullptr; size_t dsz = 0;
        p.decompress(d_comp, sz, &d_decomp, &dsz, 0);
        cudaDeviceSynchronize();
        recon.assign(n, 0.0f);
        cudaMemcpy(recon.data(), d_decomp, bytes, cudaMemcpyDeviceToHost);
    };
    auto maxErr = [&](const std::vector<float>& r) {
        double m = 0; for (size_t i = 0; i < n; ++i) m = std::max(m, (double)std::abs(r[i]-h[i]));
        return m;
    };
    std::vector<float> rs, rf;
    roundtrip(FusionPolicy::Off, rs);
    roundtrip(FusionPolicy::Auto, rf);
    EXPECT_LE(maxErr(rs), eb * 1.001) << "staged baseline exceeds bound (data/config issue)";
    EXPECT_LE(maxErr(rf), eb * 1.001) << "fused round-trip exceeds bound";
    EXPECT_EQ(rs, rf) << "fused reconstruction differs from staged";

    cudaFree(d_in);
}

// PFPL (chunk-cooperative): Quantizer(NOA,inplace,zigzag) -> Difference(negabinary)
// -> Bitshuffle(ew4) -> {RZE|RRE}. The coder is a build-time choice; the SAME
// registry entry / harness fuses either. useRre swaps the coder.
static void buildPfpl(Pipeline& p, size_t n, bool useRre) {
    p.setDims(n, 1, 1);
    auto* q = p.addStage<QuantizerStage<float, uint32_t>>();
    q->setErrorBound(1e-3f); q->setErrorBoundMode(ErrorBoundMode::NOA);
    q->setQuantRadius(32768); q->setZigzagCodes(true); q->setInplaceOutliers(true);
    auto* d = p.addStage<DifferenceStage<int32_t, uint32_t>>(); d->setChunkSize(16384);
    p.connect(d, q, "codes");
    auto* b = p.addStage<BitshuffleStage>(); b->setElementWidth(4); b->setBlockSize(16384);
    p.connect(b, d);
    if (useRre) { auto* c = p.addStage<RREStage>(); c->setWordSize(1); c->setChunkSize(16384); p.connect(c, b); }
    else        { auto* c = p.addStage<RZEStage>(); c->setWordSize(1); c->setChunkSize(16384); p.connect(c, b); }
}

// The PFPL chain fuses into one 4-stage chunk-cooperative group ending in the coder.
TEST(FusionPlanner, PfplChainIsOneGroup) {
    Pipeline p(4096 * sizeof(float), MemoryStrategy::PREALLOCATE, 2.0f);
    buildPfpl(p, 4096, /*useRre=*/false);
    p.finalize();
    auto groups = planFusionGroups(*p.getDAG());
    ASSERT_EQ(groups.size(), 1u);
    EXPECT_EQ(groups[0].stages.size(), 4u);
    EXPECT_EQ(groups[0].block_size, 16384u);
    EXPECT_TRUE(groups[0].has_coder);
}

// End-to-end: Auto must be byte-identical to staged and round-trip within bound —
// run for BOTH coders through the same registry entry (the generalization gate).
// Generic chunk-fusion end-to-end check: fused (Auto) must be byte-identical to
// staged (Off) and round-trip identically. `build` assembles the chain — reused
// for PFPL and for novel shapes the registry has no hand-written entry for.
static void chunkFusionEndToEnd(const std::function<void(Pipeline&, size_t)>& build) {
    const size_t n = 1u << 20;   // 1 Mi elems = whole 16 KB chunks
    const float  eb = 1e-3f;
    std::vector<float> h(n);
    double lo = 1e30, hi = -1e30;
    for (size_t i = 0; i < n; ++i) {
        h[i] = 0.6f*std::sin(i*0.001f) + 0.3f*std::cos(i*0.017f) + 0.05f*std::sin(i*0.13f);
        lo = std::min(lo, (double)h[i]); hi = std::max(hi, (double)h[i]);
    }
    const double bound = eb * (hi - lo) * 1.001;   // NOA
    const size_t bytes = n * sizeof(float);
    float* d_in = nullptr; ASSERT_EQ(cudaMalloc(&d_in, bytes), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_in, h.data(), bytes, cudaMemcpyHostToDevice), cudaSuccess);

    auto compressCopy = [&](FusionPolicy pol, std::vector<uint8_t>& out) -> size_t {
        Pipeline p(bytes, MemoryStrategy::PREALLOCATE, 2.0f);
        p.setFusionPolicy(pol);
        build(p, n);
        p.finalize();
        EXPECT_EQ(p.getFusedGroupCount(), pol == FusionPolicy::Auto ? 1u : 0u);
        void* d_comp = nullptr; size_t sz = 0;
        p.compress(d_in, bytes, &d_comp, &sz, 0);
        cudaDeviceSynchronize();
        out.resize(sz);
        EXPECT_EQ(cudaMemcpy(out.data(), d_comp, sz, cudaMemcpyDeviceToHost), cudaSuccess);
        return sz;
    };
    std::vector<uint8_t> staged, fused;
    size_t ss = compressCopy(FusionPolicy::Off, staged);
    size_t sf = compressCopy(FusionPolicy::Auto, fused);
    ASSERT_EQ(ss, sf) << "fused archive size differs from staged";
    EXPECT_EQ(staged, fused) << "fused archive not byte-identical to staged";

    auto roundtrip = [&](FusionPolicy pol, std::vector<float>& recon) {
        Pipeline p(bytes, MemoryStrategy::PREALLOCATE, 2.0f);
        p.setFusionPolicy(pol);
        build(p, n);
        p.finalize();
        void* d_comp = nullptr; size_t sz = 0;
        p.compress(d_in, bytes, &d_comp, &sz, 0);
        void* d_decomp = nullptr; size_t dsz = 0;
        p.decompress(d_comp, sz, &d_decomp, &dsz, 0);
        cudaDeviceSynchronize();
        recon.assign(n, 0.0f);
        cudaMemcpy(recon.data(), d_decomp, bytes, cudaMemcpyDeviceToHost);
    };
    auto maxErr = [&](const std::vector<float>& r) {
        double m = 0; for (size_t i = 0; i < n; ++i) m = std::max(m, (double)std::abs(r[i]-h[i]));
        return m;
    };
    std::vector<float> rs, rf;
    roundtrip(FusionPolicy::Off, rs);
    roundtrip(FusionPolicy::Auto, rf);
    EXPECT_LE(maxErr(rs), bound) << "staged baseline exceeds bound";
    EXPECT_LE(maxErr(rf), bound) << "fused round-trip exceeds bound";
    EXPECT_EQ(rs, rf) << "fused reconstruction differs from staged";
    cudaFree(d_in);
}

TEST(FusionPlanner, PfplRzeEndToEndFusedMatchesStaged) {
    chunkFusionEndToEnd([](Pipeline& p, size_t n){ buildPfpl(p, n, /*useRre=*/false); });
}
TEST(FusionPlanner, PfplRreEndToEndFusedMatchesStaged) {
    chunkFusionEndToEnd([](Pipeline& p, size_t n){ buildPfpl(p, n, /*useRre=*/true); });
}

// A chain the registry has NO hand-written entry for: Quantizer(inplace,zigzag) ->
// Difference(negabinary) -> RRE, with the Bitshuffle stage dropped. It fuses only
// because the generic runner assembles the kernel from the stages' getFusedOp()
// declarations — the payoff of Phase C (a novel composition, zero new registry code).
static void buildNoBitshuffle(Pipeline& p, size_t n) {
    p.setDims(n, 1, 1);
    auto* q = p.addStage<QuantizerStage<float, uint32_t>>();
    q->setErrorBound(1e-3f); q->setErrorBoundMode(ErrorBoundMode::NOA);
    q->setQuantRadius(32768); q->setZigzagCodes(true); q->setInplaceOutliers(true);
    auto* d = p.addStage<DifferenceStage<int32_t, uint32_t>>(); d->setChunkSize(16384);
    p.connect(d, q, "codes");
    auto* c = p.addStage<RREStage>(); c->setWordSize(1); c->setChunkSize(16384);
    p.connect(c, d);
}

TEST(FusionPlanner, GenericRunnerFusesNovelShapeNoRegistryEntry) {
    // First confirm the planner groups the 3-stage chain and it has no bitshuffle.
    Pipeline pg(4096 * sizeof(float), MemoryStrategy::PREALLOCATE, 2.0f);
    buildNoBitshuffle(pg, 4096);
    pg.finalize();
    auto groups = planFusionGroups(*pg.getDAG());
    ASSERT_EQ(groups.size(), 1u);
    EXPECT_EQ(groups[0].stages.size(), 3u);   // Quant -> Diff -> RRE (no Bitshuffle)

    // Then the generic runner must fuse it byte-identically + round-trip.
    chunkFusionEndToEnd(buildNoBitshuffle);
}

// New coder, zero registry glue: PFPL-shaped chain terminated by a coder the
// registry has NEVER hand-matched (RARE / RAZE, the auto-k generalizations of
// RRE/RZE). Fuses only because RAREStage/RAZEStage now declare getFusedOp() and
// the generic runner composes them — "add a coder op once, it composes into any
// chain." (Phase D payoff.)
template <class Coder>
static void buildPfplCoder(Pipeline& p, size_t n) {
    p.setDims(n, 1, 1);
    auto* q = p.addStage<QuantizerStage<float, uint32_t>>();
    q->setErrorBound(1e-3f); q->setErrorBoundMode(ErrorBoundMode::NOA);
    q->setQuantRadius(32768); q->setZigzagCodes(true); q->setInplaceOutliers(true);
    auto* d = p.addStage<DifferenceStage<int32_t, uint32_t>>(); d->setChunkSize(16384);
    p.connect(d, q, "codes");
    auto* b = p.addStage<BitshuffleStage>(); b->setElementWidth(4); b->setBlockSize(16384);
    p.connect(b, d);
    auto* c = p.addStage<Coder>(); c->setWordSize(1); c->setChunkSize(16384);
    p.connect(c, b);
}

TEST(FusionPlanner, GenericRunnerFusesRareCoderNoRegistryEntry) {
    chunkFusionEndToEnd(buildPfplCoder<RAREStage>);
}
TEST(FusionPlanner, GenericRunnerFusesRazeCoderNoRegistryEntry) {
    chunkFusionEndToEnd(buildPfplCoder<RAZEStage>);
}

// ── Phase A: each PFPL stage declares its fused device-op via getFusedOp(), so a
// generic runner can assemble the chain from the stages themselves (no per-shape
// registry). Locks the op names / strategy / params sizes the codegen consumes.
TEST(FusionPlanner, PfplStagesDeclareFusedOps) {
    Pipeline p(4096 * sizeof(float), MemoryStrategy::PREALLOCATE, 2.0f);
    buildPfpl(p, 4096, /*useRre=*/false);
    p.finalize();
    auto groups = planFusionGroups(*p.getDAG());
    ASSERT_EQ(groups.size(), 1u);
    ASSERT_EQ(groups[0].stages.size(), 4u);

    const char* kNames[] = {"QuantInplaceZigzag", "DiffNegabinary", "Bitshuffle32", "RZECoder"};
    for (size_t i = 0; i < 4; ++i) {
        FusedOpDecl op = groups[0].stages[i]->getFusedOp();
        EXPECT_TRUE(op.valid()) << "stage " << i << " declares no fused op";
        EXPECT_EQ(op.strategy, FusionStrategy::ChunkCooperative);
        EXPECT_EQ(op.op_name, kNames[i]);
        EXPECT_FALSE(op.include_header.empty());
    }
    // Only the Map (quant) op is parametric; its params match the shared POD.
    EXPECT_EQ(groups[0].stages[0]->getFusedOp().params.size(),
              sizeof(fused::chunk::QuantInplaceZigzagParams));
    EXPECT_TRUE(groups[0].stages[1]->getFusedOp().params.empty());
    EXPECT_TRUE(groups[0].stages[3]->getFusedOp().params.empty());

    // RRE tail declares its own coder op name (the swappable sink).
    Pipeline pr(4096 * sizeof(float), MemoryStrategy::PREALLOCATE, 2.0f);
    buildPfpl(pr, 4096, /*useRre=*/true);
    pr.finalize();
    auto gr = planFusionGroups(*pr.getDAG());
    ASSERT_EQ(gr.size(), 1u);
    EXPECT_EQ(gr[0].stages[3]->getFusedOp().op_name, "RRECoder");
}

// ── NVRTC codegen contract (host-only): the stage-chain fingerprint composes
// into the "connecting code" that wraps chunk_fused_body<QuantOp,Coder,Trs...>.
// The end-to-end byte-identity of the generated kernel is covered by running the
// two PFPL tests above under FZ_FUSION_NVRTC=1; here we just lock the mapping
// from a ChunkFusionSpec to its template-argument list.
TEST(FusionPlanner, NvrtcCodegenComposesSpecOps) {
    fused::ChunkFusionSpec spec;   // PFPL defaults
    const std::string src = fused::generateChunkFusionSource(spec);
    // The generated glue names every op from the spec, in order, as template args.
    EXPECT_NE(src.find("chunk_fused_body< QuantInplaceZigzag, RZECoder, "
                       "DiffNegabinary, Bitshuffle32 >"), std::string::npos);
    EXPECT_NE(src.find("extern \"C\" __global__ void"), std::string::npos);
    EXPECT_NE(src.find("#include \"fused/chunk_fusion/chunk_fusion.cuh\""),
              std::string::npos);

    // Swapping the coder (the swappable sink) is a data change — no new C++.
    spec.coder = fused::chunkCoderOpName(fused::ChunkCoderKind::RRE);
    EXPECT_NE(fused::generateChunkFusionSource(spec).find("RRECoder"),
              std::string::npos);

    // Dropping a transform re-composes a shorter chain from the same generator.
    spec.transforms = {"DiffNegabinary"};
    const std::string shorter = fused::generateChunkFusionSource(spec);
    EXPECT_NE(shorter.find("RRECoder, DiffNegabinary >"), std::string::npos);
    EXPECT_EQ(shorter.find("Bitshuffle32"), std::string::npos);
}

// The cuszp3 front (Quantizer -> TiledLorenzo(8x8) -> AdaptiveBitpack(64)) is one
// block-local fusable group with block_size = tile_elems = 64.
TEST(FusionPlanner, Cuszp3ChainIsOneGroup) {
    Pipeline p(64 * 64 * sizeof(float), MemoryStrategy::PREALLOCATE, 2.0f);
    buildCuszp3(p, 64, 64);
    p.finalize();

    auto groups = planFusionGroups(*p.getDAG());
    ASSERT_EQ(groups.size(), 1u);
    EXPECT_EQ(groups[0].stages.size(), 3u);
    EXPECT_EQ(groups[0].block_size, 64u);
    EXPECT_TRUE(groups[0].has_coder);
}

// TiledLorenzo declares BlockLocal(tile_elems) forward, Unfusable inverse.
TEST(FusionPlanner, TiledLorenzoFusionSpec) {
    TiledLorenzoStage<int32_t> tl; tl.setTileShape(8, 8);
    EXPECT_EQ(tl.getFusionSpec().access, FusionAccess::BlockLocal);
    EXPECT_EQ(tl.getFusionSpec().block_size, 64u);

    TiledLorenzoStage<int32_t> tli; tli.setTileShape(8, 8); tli.setInverse(true);
    EXPECT_EQ(tli.getFusionSpec().access, FusionAccess::Unfusable);
}

// End-to-end for cuszp3: fusion must be byte-identical to staged and round-trip
// within the bound. Uses non-tile-aligned dims to exercise edge-tile padding.
TEST(FusionPlanner, Cuszp3EndToEndFusedMatchesStaged) {
    const size_t dx = 300, dy = 180;   // neither is a multiple of 8 -> padded tiles
    const size_t n  = dx * dy;
    const float  eb = 1e-3f;
    std::vector<float> h(n);
    for (size_t j = 0; j < dy; ++j)
        for (size_t i = 0; i < dx; ++i)
            h[j*dx+i] = 0.5f*std::sin(i*0.03f) + 0.3f*std::cos(j*0.05f) + 0.1f*std::sin((i+j)*0.011f);
    const size_t bytes = n * sizeof(float);
    float* d_in = nullptr; ASSERT_EQ(cudaMalloc(&d_in, bytes), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_in, h.data(), bytes, cudaMemcpyHostToDevice), cudaSuccess);

    auto compressCopy = [&](FusionPolicy pol, std::vector<uint8_t>& out) -> size_t {
        Pipeline p(bytes, MemoryStrategy::PREALLOCATE, 2.0f);
        p.setFusionPolicy(pol);
        buildCuszp3(p, dx, dy);
        p.finalize();
        EXPECT_EQ(p.getFusedGroupCount(), pol == FusionPolicy::Auto ? 1u : 0u);
        void* d_comp = nullptr; size_t sz = 0;
        p.compress(d_in, bytes, &d_comp, &sz, 0);
        cudaDeviceSynchronize();
        out.resize(sz);
        EXPECT_EQ(cudaMemcpy(out.data(), d_comp, sz, cudaMemcpyDeviceToHost), cudaSuccess);
        return sz;
    };

    std::vector<uint8_t> staged, fused;
    size_t sz_staged = compressCopy(FusionPolicy::Off, staged);
    size_t sz_fused  = compressCopy(FusionPolicy::Auto, fused);
    ASSERT_EQ(sz_staged, sz_fused) << "fused archive size differs from staged";
    EXPECT_EQ(staged, fused)       << "fused archive is not byte-identical to staged";

    auto roundtrip = [&](FusionPolicy pol, std::vector<float>& recon) {
        Pipeline p(bytes, MemoryStrategy::PREALLOCATE, 2.0f);
        p.setFusionPolicy(pol);
        buildCuszp3(p, dx, dy);
        p.finalize();
        void* d_comp = nullptr; size_t sz = 0;
        p.compress(d_in, bytes, &d_comp, &sz, 0);
        void* d_decomp = nullptr; size_t dsz = 0;
        p.decompress(d_comp, sz, &d_decomp, &dsz, 0);
        cudaDeviceSynchronize();
        recon.assign(n, 0.0f);
        cudaMemcpy(recon.data(), d_decomp, bytes, cudaMemcpyDeviceToHost);
    };
    auto maxErr = [&](const std::vector<float>& r) {
        double m = 0; for (size_t i = 0; i < n; ++i) m = std::max(m, (double)std::abs(r[i]-h[i]));
        return m;
    };
    std::vector<float> rs, rf;
    roundtrip(FusionPolicy::Off, rs);
    roundtrip(FusionPolicy::Auto, rf);
    EXPECT_LE(maxErr(rs), eb * 1.001) << "staged baseline exceeds bound (data/config issue)";
    EXPECT_LE(maxErr(rf), eb * 1.001) << "fused round-trip exceeds bound";
    EXPECT_EQ(rs, rf) << "fused reconstruction differs from staged";

    cudaFree(d_in);
}
