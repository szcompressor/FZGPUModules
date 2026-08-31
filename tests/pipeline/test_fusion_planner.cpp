// Unit tests for the fusion planner: does it identify the fusable stage chains
// in a DAG that a fused kernel could collapse? Pure analysis — no execution.

#include "fzgpumodules.h"
#include "advanced/fusion_planner.h"
#include "fused/chunk_fusion/nvrtc_chunk_fusion.h"
#include "fused/chunk_fusion/chunk_op_params.h"
#include "fused/fused_block/nvrtc_warp_fusion.h"
#include "fused/fused_block/warp_op_params.h"

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include <functional>
#include <vector>

using namespace fz;

namespace {

// Planner-only Map that deliberately has no generated op. It extends the
// maximal legality chain but cannot belong to an executable specialization,
// exercising selection of a valid interior subspan.
class PlannerOnlyMapStage final : public Stage {
public:
    void execute(fz::stream_t, MemoryPool*, const std::vector<void*>&,
                 const std::vector<void*>&, const std::vector<size_t>& sizes) override {
        actual_ = sizes.empty() ? 0 : sizes[0];
    }
    std::string getName() const override { return "PlannerOnlyMap"; }
    size_t getNumInputs() const override { return 1; }
    size_t getNumOutputs() const override { return 1; }
    std::vector<size_t> estimateOutputSizes(
        const std::vector<size_t>& inputs) const override {
        return {inputs.empty() ? 0 : inputs[0]};
    }
    std::unordered_map<std::string, size_t>
    getActualOutputSizesByName() const override { return {{"output", actual_}}; }
    uint16_t getStageTypeId() const override { return 0xffffu; }
    uint8_t getOutputDataType(size_t) const override {
        return static_cast<uint8_t>(DataType::FLOAT32);
    }
    uint8_t getInputDataType(size_t) const override {
        return static_cast<uint8_t>(DataType::FLOAT32);
    }
    FusionSpec getFusionSpec() const override {
        return FusionSpec{FusionAccess::Map, 0};
    }
private:
    size_t actual_ = 0;
};

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

void buildFsz(Pipeline& p, size_t n, bool outlier) {
    p.setDims(n, 1, 1);
    auto* q = p.addStage<QuantizerStage<float, uint32_t>>();
    q->setErrorBound(1e-3f);
    q->setErrorBoundMode(ErrorBoundMode::ABS);
    q->setLinearMode(true);
    auto* al = p.addStage<AdaptiveLorenzoStage<int32_t>>();
    auto* ab = p.addStage<AdaptiveBitpackStage<int32_t>>();
    ab->setBlockSize(32);
    ab->setOutlierSelection(outlier);
    p.connect(al, q, "codes");
    p.connect(ab, al);
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

TEST(FusionPlanner, AutoSelectsExecutableSubspanOfMaximalChain) {
    Pipeline p(1024 * sizeof(float), MemoryStrategy::PREALLOCATE, 3.0f);
    p.setFusionPolicy(FusionPolicy::Auto);
    p.setDims(1024, 1, 1);
    auto* prefix = p.addStage<PlannerOnlyMapStage>();
    auto* q = p.addStage<QuantizerStage<float, uint32_t>>();
    q->setErrorBound(1e-3f);
    q->setErrorBoundMode(ErrorBoundMode::ABS);
    q->setLinearMode(true);
    auto* l = p.addStage<LorenzoStage<int32_t>>();
    l->setBlockSize(32);
    auto* a = p.addStage<AdaptiveBitpackStage<int32_t>>();
    a->setBlockSize(32);
    a->setOutlierSelection(true);
    p.connect(q, prefix);
    p.connect(l, q, "codes");
    p.connect(a, l);
    p.finalize();

    const auto maximal = planFusionGroups(*p.getDAG());
    ASSERT_EQ(maximal.size(), 1u);
    EXPECT_EQ(maximal[0].stages.size(), 4u);
    // The four-stage span has no registry match; the interior q->lz->AB span does.
    EXPECT_EQ(p.getFusedGroupCount(), 1u);
    const auto& info = p.getFusionInfo();
    EXPECT_EQ(info.policy, FusionPolicy::Auto);
    EXPECT_EQ(info.legal_group_count, 1u);
    ASSERT_EQ(info.installed_groups.size(), 1u);
    EXPECT_EQ(info.installed_groups[0].implementation, "warp-register");
    EXPECT_EQ(info.installed_groups[0].stages.size(), 3u);
    EXPECT_TRUE(info.fallback_reason.empty());
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

// Encoded-size oracles are semantic declarations, independent of whether a
// particular coder configuration currently participates in kernel fusion.
TEST(FusionPlanner, AdaptiveBitpackEncodingOracleDeclarations) {
    AdaptiveBitpackStage<int32_t> plain;
    plain.setBlockSize(32);
    const EncodingOracleDecl p = plain.getEncodingOracle();
    ASSERT_TRUE(p.valid());
    EXPECT_EQ(p.kind, EncodingOracleKind::PlainFixedRateBitpack);
    EXPECT_EQ(p.input_data_type, static_cast<uint8_t>(DataType::INT32));
    EXPECT_EQ(p.unit_elems, 32u);
    EXPECT_TRUE(p.exact);
    EXPECT_TRUE(p.additive);
    EXPECT_FALSE(plain.getFusedOp().valid())
        << "plain staged semantics must not imply current warp-fusion eligibility";

    plain.setOutlierSelection(true);
    const EncodingOracleDecl adaptive = plain.getEncodingOracle();
    ASSERT_TRUE(adaptive.valid());
    EXPECT_EQ(adaptive.kind, EncodingOracleKind::AdaptiveFixedRateBitpack);
}

TEST(FusionPlanner, FinalizeBindsPlainOracleToAdaptiveLorenzo) {
    Pipeline p(4096 * sizeof(int32_t), MemoryStrategy::PREALLOCATE);
    auto* al = p.addStage<AdaptiveLorenzoStage<int32_t>>();
    auto* ab = p.addStage<AdaptiveBitpackStage<int32_t>>();
    ab->setBlockSize(32);
    p.connect(ab, al);

    EXPECT_FALSE(al->hasBoundEncodingOracle());
    p.finalize();
    EXPECT_TRUE(al->hasBoundEncodingOracle());
    EXPECT_EQ(al->getBoundEncodingOracleKind(),
              EncodingOracleKind::PlainFixedRateBitpack);
}

TEST(FusionPlanner, FinalizeBindsAdaptiveOutlierOracleToAdaptiveLorenzo) {
    Pipeline p(4096 * sizeof(int32_t), MemoryStrategy::PREALLOCATE);
    auto* al = p.addStage<AdaptiveLorenzoStage<int32_t>>();
    auto* ab = p.addStage<AdaptiveBitpackStage<int32_t>>();
    ab->setBlockSize(32);
    ab->setOutlierSelection(true);
    p.connect(ab, al);
    p.finalize();

    EXPECT_TRUE(al->hasBoundEncodingOracle());
    EXPECT_EQ(al->getBoundEncodingOracleKind(),
              EncodingOracleKind::AdaptiveFixedRateBitpack);
}

TEST(FusionPlanner, TileAdaptiveGeometryReportsExactLegalityFailures) {
    FusionGeometry geometry;
    EXPECT_EQ(extendFusionGeometry(
                  geometry, FusionSpec{FusionAccess::TileAdaptive, 250, 32}),
              FusionCompatibility::InvalidTileGeometry);

    geometry = {};
    ASSERT_EQ(extendFusionGeometry(
                  geometry, FusionSpec{FusionAccess::TileAdaptive, 256, 32}),
              FusionCompatibility::Compatible);
    EXPECT_EQ(geometry.selector_tile_size, 256u);
    EXPECT_EQ(geometry.coder_unit_size, 32u);
    EXPECT_EQ(extendFusionGeometry(
                  geometry, FusionSpec{FusionAccess::Map, 0}),
              FusionCompatibility::TileInteriorStageUnsupported);
    EXPECT_EQ(extendFusionGeometry(
                  geometry, FusionSpec{FusionAccess::Cooperative, 64}),
              FusionCompatibility::TileCoderUnitMismatch);

    geometry = {};
    ASSERT_EQ(extendFusionGeometry(
                  geometry, FusionSpec{FusionAccess::BlockLocal, 32}),
              FusionCompatibility::Compatible);
    EXPECT_EQ(extendFusionGeometry(
                  geometry, FusionSpec{FusionAccess::BlockLocal, 64}),
              FusionCompatibility::StandardBlockMismatch);
    EXPECT_EQ(extendFusionGeometry(
                  geometry, FusionSpec{FusionAccess::TileAdaptive, 256, 32}),
              FusionCompatibility::TileAfterBlockLocal);
    EXPECT_STREQ(fusionCompatibilityName(
                     FusionCompatibility::TileCoderUnitMismatch),
                 "tile_coder_unit_mismatch");
}

TEST(FusionPlanner, FszLegalChainHasNoUnprofitableExecutionPlan) {
    auto make = [](FusionPolicy policy) {
        auto p = std::make_unique<Pipeline>(
            4096 * sizeof(float), MemoryStrategy::PREALLOCATE, 2.0f);
        p->setFusionPolicy(policy);
        p->setDims(4096, 1, 1);
        auto* q = p->addStage<QuantizerStage<float, uint32_t>>();
        q->setErrorBound(1e-3f);
        q->setErrorBoundMode(ErrorBoundMode::ABS);
        q->setLinearMode(true);
        auto* al = p->addStage<AdaptiveLorenzoStage<int32_t>>();
        auto* ab = p->addStage<AdaptiveBitpackStage<int32_t>>();
        ab->setBlockSize(32);
        p->connect(al, q, "codes");
        p->connect(ab, al);
        p->finalize();
        return p;
    };

    auto automatic = make(FusionPolicy::Auto);
    const auto groups = planFusionGroups(*automatic->getDAG());
    ASSERT_EQ(groups.size(), 1u);
    EXPECT_EQ(groups[0].stages.size(), 3u);
    EXPECT_TRUE(groups[0].has_tile_adaptive);
    EXPECT_EQ(groups[0].selector_tile_size, 256u);
    EXPECT_EQ(groups[0].coder_unit_size, 32u);
    EXPECT_EQ(groups[0].block_size, 0u);
    EXPECT_TRUE(groups[0].has_coder);
    EXPECT_EQ(automatic->getFusedGroupCount(), 0u);
    EXPECT_EQ(automatic->getFusionInfo().fallback_reason,
              "no_profitable_implementation");

    auto forced = make(FusionPolicy::Force);
    EXPECT_EQ(forced->getFusedGroupCount(), 0u);
}

TEST(FusionPlanner, FszAutoFallbackMatchesStagedAndRoundTrips) {
    const size_t n = (1u << 16) + 37;
    const size_t bytes = n * sizeof(float);
    std::vector<float> input(n);
    for (size_t i = 0; i < n; ++i)
        input[i] = 50.0f + 0.4f * std::sin(i * 0.013f) +
                   0.08f * std::cos(i * 0.071f);
    float* d_input = nullptr;
    ASSERT_EQ(cudaMalloc(&d_input, bytes), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice), cudaSuccess);

    for (bool outlier : {false, true}) {
        auto compressCopy = [&](FusionPolicy policy, std::vector<uint8_t>& archive) {
            Pipeline p(bytes, MemoryStrategy::PREALLOCATE, 4.0f);
            p.setFusionPolicy(policy);
            buildFsz(p, n, outlier);
            p.finalize();
            EXPECT_EQ(p.getFusedGroupCount(), 0u);
            void* d_compressed = nullptr;
            size_t compressed_bytes = 0;
            p.compress(d_input, bytes, &d_compressed, &compressed_bytes, 0);
            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
            archive.resize(compressed_bytes);
            EXPECT_EQ(cudaMemcpy(archive.data(), d_compressed, compressed_bytes,
                                 cudaMemcpyDeviceToHost), cudaSuccess);
        };

        std::vector<uint8_t> staged, fused;
        compressCopy(FusionPolicy::Off, staged);
        compressCopy(FusionPolicy::Auto, fused);
        ASSERT_EQ(fused.size(), staged.size()) << "outlier=" << outlier;
        EXPECT_EQ(fused, staged) << "outlier=" << outlier;

        Pipeline p(bytes, MemoryStrategy::PREALLOCATE, 4.0f);
        p.setFusionPolicy(FusionPolicy::Auto);
        buildFsz(p, n, outlier);
        p.finalize();
        void* d_compressed = nullptr; size_t compressed_bytes = 0;
        p.compress(d_input, bytes, &d_compressed, &compressed_bytes, 0);
        void* d_reconstructed = nullptr; size_t reconstructed_bytes = 0;
        p.decompress(d_compressed, compressed_bytes, &d_reconstructed,
                     &reconstructed_bytes, 0);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        ASSERT_EQ(reconstructed_bytes, bytes);
        std::vector<float> reconstructed(n);
        ASSERT_EQ(cudaMemcpy(reconstructed.data(), d_reconstructed, bytes,
                             cudaMemcpyDeviceToHost), cudaSuccess);
        double max_error = 0.0;
        for (size_t i = 0; i < n; ++i)
            max_error = std::max(max_error,
                                 static_cast<double>(std::abs(reconstructed[i] - input[i])));
        EXPECT_LE(max_error, 1.01e-3) << "outlier=" << outlier;
    }
    cudaFree(d_input);
}

TEST(FusionPlanner, AdaptiveAuxiliaryOutputsAreDeclarative) {
    AdaptiveLorenzoStage<int32_t> al;
    EncodingOracleDecl oracle;
    oracle.kind = EncodingOracleKind::PlainFixedRateBitpack;
    oracle.op_name = "PlainBitpackCoder";
    oracle.input_data_type = static_cast<uint8_t>(DataType::INT32);
    oracle.unit_elems = 32;
    oracle.exact = true;
    oracle.additive = true;
    ASSERT_TRUE(al.bindDownstreamEncodingOracle(oracle));

    const auto al_aux = al.getFusedAuxOutputs();
    ASSERT_EQ(al_aux.size(), 2u);
    EXPECT_EQ(al_aux[0].name, "modes");
    EXPECT_EQ(al_aux[0].size_kind, FusedAuxSizeKind::FixedBitsPerUnit);
    EXPECT_EQ(al_aux[0].unit_elems, 256u);
    EXPECT_EQ(al_aux[0].bits_per_unit, 2u);
    EXPECT_EQ(al_aux[1].name, "means");
    EXPECT_EQ(al_aux[1].size_kind, FusedAuxSizeKind::CompactedElements);
    EXPECT_EQ(al_aux[1].count_group, 1u);

    QuantizerStage<float, uint32_t> quant;
    quant.setErrorBoundMode(ErrorBoundMode::ABS);
    quant.setZigzagCodes(true);
    const auto q_aux = quant.getFusedAuxOutputs();
    ASSERT_EQ(q_aux.size(), 2u);
    EXPECT_EQ(q_aux[0].name, "outlier_vals");
    EXPECT_EQ(q_aux[1].name, "outlier_idxs");
    EXPECT_EQ(q_aux[0].count_group, q_aux[1].count_group);
}

TEST(FusionPlanner, AdaptiveSelectorRejectsNonAdditiveOrMismatchedOracle) {
    AdaptiveLorenzoStage<int32_t> al;
    EncodingOracleDecl oracle;
    oracle.kind = EncodingOracleKind::PlainFixedRateBitpack;
    oracle.op_name = "PlainBitpackCoder";
    oracle.input_data_type = static_cast<uint8_t>(DataType::INT32);
    oracle.unit_elems = 32;
    oracle.exact = true;

    oracle.additive = false;
    EXPECT_FALSE(al.bindDownstreamEncodingOracle(oracle));
    oracle.additive = true;
    oracle.unit_elems = 64;
    EXPECT_FALSE(al.bindDownstreamEncodingOracle(oracle));
    EXPECT_EQ(al.getFusionSpec().access, FusionAccess::Unfusable);
}

TEST(FusionPlanner, AmbiguousAdaptiveConsumerTopologyStaysUnfusable) {
    Pipeline p(4096 * sizeof(int32_t), MemoryStrategy::PREALLOCATE);
    auto* al = p.addStage<AdaptiveLorenzoStage<int32_t>>();
    auto* ab0 = p.addStage<AdaptiveBitpackStage<int32_t>>();
    auto* ab1 = p.addStage<AdaptiveBitpackStage<int32_t>>();
    ab0->setBlockSize(32);
    ab1->setBlockSize(32);
    p.connect(ab0, al);
    p.connect(ab1, al);
    p.finalize();

    EXPECT_FALSE(al->hasBoundEncodingOracle());
    EXPECT_EQ(al->getFusionSpec().access, FusionAccess::Unfusable);
    for (const auto& group : planFusionGroups(*p.getDAG()))
        for (Stage* stage : group.stages)
            EXPECT_NE(stage, static_cast<Stage*>(al));
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

// Warp-register fusion under NOA (not just ABS): cuszp2 with a range-relative bound
// must fuse, stay byte-identical to staged, and satisfy the bound — the runner primes
// the NOA range scan and passes the resolved abs_eb into the uniform-step fused kernel.
// Positive data + non-32-aligned tail also exercises the padding-exclusion fix.
TEST(FusionPlanner, Cuszp2NoaEndToEndFusedMatchesStaged) {
    const size_t n  = (1u << 20) + 777;
    const float  eb = 1e-3f;
    std::vector<float> h(n);
    double lo = 1e30, hi = -1e30;
    for (size_t i = 0; i < n; ++i) {
        h[i] = 1.0f + 0.5f*std::sin(i*0.001f) + 0.2f*std::cos(i*0.017f);   // strictly > 0
        lo = std::min(lo, (double)h[i]); hi = std::max(hi, (double)h[i]);
    }
    const double bound = eb * (hi - lo) * 1.001;
    const size_t bytes = n * sizeof(float);
    float* d_in = nullptr; ASSERT_EQ(cudaMalloc(&d_in, bytes), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_in, h.data(), bytes, cudaMemcpyHostToDevice), cudaSuccess);

    auto build = [&](Pipeline& p) {
        p.setDims(n, 1, 1);
        auto* q = p.addStage<QuantizerStage<float, uint32_t>>();
        q->setErrorBound(eb); q->setErrorBoundMode(ErrorBoundMode::NOA); q->setLinearMode(true);
        auto* l = p.addStage<LorenzoStage<int32_t>>(); l->setBlockSize(32);
        p.connect(l, q, "codes");
        auto* a = p.addStage<AdaptiveBitpackStage<int32_t>>();
        a->setBlockSize(32); a->setOutlierSelection(true);
        p.connect(a, l);
    };
    auto compressCopy = [&](FusionPolicy pol, std::vector<uint8_t>& out) -> size_t {
        Pipeline p(bytes, MemoryStrategy::PREALLOCATE, 2.0f);
        p.setFusionPolicy(pol); build(p); p.finalize();
        EXPECT_EQ(p.getFusedGroupCount(), pol == FusionPolicy::Auto ? 1u : 0u);
        void* d_comp = nullptr; size_t sz = 0;
        p.compress(d_in, bytes, &d_comp, &sz, 0);
        cudaDeviceSynchronize();
        out.resize(sz);
        cudaMemcpy(out.data(), d_comp, sz, cudaMemcpyDeviceToHost);
        return sz;
    };
    std::vector<uint8_t> staged, fused;
    size_t ss = compressCopy(FusionPolicy::Off,  staged);
    size_t sf = compressCopy(FusionPolicy::Auto, fused);
    ASSERT_EQ(ss, sf) << "NOA fused archive size differs from staged";
    EXPECT_EQ(staged, fused) << "NOA fused archive not byte-identical to staged";

    auto roundtrip = [&](FusionPolicy pol, std::vector<float>& r) {
        Pipeline p(bytes, MemoryStrategy::PREALLOCATE, 2.0f);
        p.setFusionPolicy(pol); build(p); p.finalize();
        void* d_comp = nullptr; size_t sz = 0;
        p.compress(d_in, bytes, &d_comp, &sz, 0);
        void* d_decomp = nullptr; size_t dsz = 0;
        p.decompress(d_comp, sz, &d_decomp, &dsz, 0);
        cudaDeviceSynchronize();
        r.assign(n, 0.0f);
        cudaMemcpy(r.data(), d_decomp, bytes, cudaMemcpyDeviceToHost);
    };
    auto maxErr = [&](const std::vector<float>& r) {
        double m = 0; for (size_t i = 0; i < n; ++i) m = std::max(m, (double)std::abs(r[i]-h[i]));
        return m;
    };
    std::vector<float> rs, rf;
    roundtrip(FusionPolicy::Off,  rs);
    roundtrip(FusionPolicy::Auto, rf);
    EXPECT_LE(maxErr(rs), bound) << "staged NOA exceeds bound";
    EXPECT_LE(maxErr(rf), bound) << "fused NOA exceeds bound";
    EXPECT_EQ(rs, rf) << "fused NOA reconstruction differs from staged";
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

// PFPL with SPLIT outliers (3-port): Quantizer(ABS,zigzag, inplace=OFF) -> Difference
// -> Bitshuffle -> RZE. The quant Map now emits a clean codes stream (0 at outlier
// positions) plus an escaping (index,value) outlier list — the `QuantSplitOutlier`
// fused op that consumes the multi-output plumbing. Unlike the inplace/single-output
// chains, the archive is NOT byte-identical to staged: the outlier list is filled by a
// global atomicAdd, so its order is nondeterministic (true of the staged path too).
// What must hold is that fusion engages, the codes-side survives, and the fused
// round-trip reconstructs identically to staged and within bound (the outlier SCATTER
// is order-independent). Data: small representable base with sparse forced-outlier
// spikes (|x| >= threshold), ~0.2% outliers, well under the 5% capacity.
static void buildPfplSplit(Pipeline& p, size_t n) {
    p.setDims(n, 1, 1);
    auto* q = p.addStage<QuantizerStage<float, uint32_t>>();
    q->setErrorBound(1e-3f); q->setErrorBoundMode(ErrorBoundMode::ABS);
    q->setQuantRadius(32768); q->setZigzagCodes(true);
    q->setInplaceOutliers(false);                 // 3-port split outliers
    q->setOutlierThreshold(1.0f);                 // |x| >= 1 forced to the side list
    auto* d = p.addStage<DifferenceStage<int32_t, uint32_t>>(); d->setChunkSize(16384);
    p.connect(d, q, "codes");
    auto* b = p.addStage<BitshuffleStage>(); b->setElementWidth(4); b->setBlockSize(16384);
    p.connect(b, d);
    auto* c = p.addStage<RZEStage>(); c->setWordSize(1); c->setChunkSize(16384);
    p.connect(c, b);
}

TEST(FusionPlanner, PfplSplitOutlierFusesAndRoundTrips) {
    const size_t n  = 1u << 20;
    const float  eb = 1e-3f;
    std::vector<float> h(n);
    size_t n_out = 0;
    for (size_t i = 0; i < n; ++i) {
        h[i] = 0.2f * std::sin(i * 0.001f);       // representable (|x| < 1)
        if (i % 512 == 0) { h[i] = 5.0f; ++n_out; }  // forced outlier (|x| >= threshold)
    }
    ASSERT_GT(n_out, 0u);
    const double bound = eb * 1.001;              // ABS
    const size_t bytes = n * sizeof(float);
    float* d_in = nullptr; ASSERT_EQ(cudaMalloc(&d_in, bytes), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_in, h.data(), bytes, cudaMemcpyHostToDevice), cudaSuccess);

    // Confirm the chain fuses into one group (the split quant is still a Map head).
    { Pipeline pg(bytes, MemoryStrategy::PREALLOCATE, 2.0f);
      buildPfplSplit(pg, n); pg.finalize();
      auto groups = planFusionGroups(*pg.getDAG());
      ASSERT_EQ(groups.size(), 1u);
      EXPECT_EQ(groups[0].stages.size(), 4u); }

    auto roundtrip = [&](FusionPolicy pol, std::vector<float>& recon) {
        Pipeline p(bytes, MemoryStrategy::PREALLOCATE, 2.0f);
        p.setFusionPolicy(pol);
        buildPfplSplit(p, n);
        p.finalize();
        EXPECT_EQ(p.getFusedGroupCount(), pol == FusionPolicy::Auto ? 1u : 0u);
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
    roundtrip(FusionPolicy::Off,  rs);
    roundtrip(FusionPolicy::Auto, rf);
    EXPECT_LE(maxErr(rs), bound) << "staged split-outlier baseline exceeds bound";
    EXPECT_LE(maxErr(rf), bound) << "fused split-outlier round-trip exceeds bound";
    EXPECT_EQ(rs, rf) << "fused split-outlier reconstruction differs from staged";
    // The outlier scatter must have restored the spikes exactly (they are lossless).
    for (size_t i = 0; i < n; i += 512)
        EXPECT_FLOAT_EQ(rf[i], 5.0f) << "outlier at " << i << " not restored";
    cudaFree(d_in);
}

// Regression: the fused NOA range scan must exclude the chunk-aligned zero-padding
// tail. If it scans the padding, those zeros lower the min and inflate the range →
// too-large abs_eb → the fused path quantizes too loosely and violates the bound on
// real data (found on CESM CLDHGH; fixed via the logical-grid clamp in
// primeComputedAbsEb). Needs BOTH a partial final 16 KB chunk (non-aligned n) AND
// all-positive data so padding-0 shifts the min — which is why the aligned,
// zero-centred end-to-end tests above never caught it.
TEST(FusionPlanner, PfplNoaPartialChunkExcludesPadding) {
    const size_t n  = (1u << 20) + 777;   // partial final chunk (not a 4096 multiple)
    const float  eb = 1e-3f;
    std::vector<float> h(n);
    double lo = 1e30, hi = -1e30;
    for (size_t i = 0; i < n; ++i) {
        h[i] = 1.0f + 0.4f*std::sin(i*0.001f) + 0.1f*std::cos(i*0.017f);   // strictly > 0
        lo = std::min(lo, (double)h[i]); hi = std::max(hi, (double)h[i]);
    }
    const double bound = eb * (hi - lo) * 1.001;   // NOA over REAL data (padding excluded)
    const size_t bytes = n * sizeof(float);
    float* d_in = nullptr; ASSERT_EQ(cudaMalloc(&d_in, bytes), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_in, h.data(), bytes, cudaMemcpyHostToDevice), cudaSuccess);

    auto roundtrip = [&](FusionPolicy pol, std::vector<float>& recon) {
        Pipeline p(bytes, MemoryStrategy::PREALLOCATE, 2.0f);
        p.setFusionPolicy(pol);
        buildPfpl(p, n, /*useRre=*/false);   // inplace-zigzag NOA quant + Diff + Bitshuffle + RZE
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
    roundtrip(FusionPolicy::Off,  rs);
    roundtrip(FusionPolicy::Auto, rf);
    EXPECT_LE(maxErr(rs), bound) << "staged baseline exceeds bound";
    EXPECT_LE(maxErr(rf), bound) << "fused exceeds bound — padding inflated the NOA range";
    EXPECT_EQ(rs, rf) << "fused real-data reconstruction diverges from staged";
    cudaFree(d_in);
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

// Phase E: the warp-register (cuSZp) stages declare WarpRegister fused-ops too, so
// cuszp2 and cuszp3 route through ONE generic registry entry that dispatches on the
// predictor op name — no per-shape matcher/runner.
TEST(FusionPlanner, WarpStagesDeclareFusedOps) {
    Pipeline p(4096 * sizeof(float), MemoryStrategy::PREALLOCATE, 2.0f);
    buildCuszp2(p, 4096);
    p.finalize();
    auto g = planFusionGroups(*p.getDAG());
    ASSERT_EQ(g.size(), 1u);
    ASSERT_EQ(g[0].stages.size(), 3u);
    const char* kNames[] = {"LinearQuant", "Lorenzo1DPredictor", "AdaptiveBitpackCoder"};
    for (size_t i = 0; i < 3; ++i) {
        FusedOpDecl op = g[0].stages[i]->getFusedOp();
        EXPECT_TRUE(op.valid()) << "warp stage " << i << " declares no fused op";
        EXPECT_EQ(op.strategy, FusionStrategy::WarpRegister);
        EXPECT_EQ(op.op_name, kNames[i]);
    }
    // The predictor stage carries the shape the generic NVRTC runner needs (EPL +
    // packed params with a leading inv2eb slot) so the runner does not downcast it.
    const FusedOpDecl pred2 = g[0].stages[1]->getFusedOp();
    EXPECT_EQ(pred2.elems_per_lane, 1u);
    EXPECT_EQ(pred2.params.size(), sizeof(fused::warp::Lorenzo1DParams));
    EXPECT_EQ(pred2.include_header, "fused/fused_block/warp_fusion.cuh");

    // cuSZp3 shares the strategy with a different predictor op — same generic entry.
    Pipeline p3(300 * 180 * sizeof(float), MemoryStrategy::PREALLOCATE, 2.0f);
    buildCuszp3(p3, 300, 180);
    p3.finalize();
    auto g3 = planFusionGroups(*p3.getDAG());
    ASSERT_EQ(g3.size(), 1u);
    const FusedOpDecl pred3 = g3[0].stages[1]->getFusedOp();
    EXPECT_EQ(pred3.strategy, FusionStrategy::WarpRegister);
    EXPECT_EQ(pred3.op_name, "TiledLorenzo2DPredictor");
    EXPECT_EQ(pred3.elems_per_lane, 2u);
    EXPECT_EQ(pred3.params.size(), sizeof(fused::warp::TiledLorenzo2DParams));
    // n_ab = padded tile-major count: ceil(300/8)*8 * ceil(180/8)*8 = 304 * 184.
    EXPECT_EQ(pred3.n_ab, size_t{304} * 184);
}

// Warp NVRTC codegen contract (host-only): the WarpFusionSpec composes into the two
// extern-C kernels wrapping fused_rate_body/fused_pack_body, parameterised on the
// predictor policy type and ElemsPerLane. End-to-end byte-identity is covered by the
// cuszp2/cuszp3 tests; here we lock the spec → source mapping (adding a predictor is a
// data change to the spec, no new launcher/dispatch).
TEST(FusionPlanner, WarpNvrtcCodegenComposesSpecOps) {
    fused::WarpFusionSpec s2;   // cuSZp2 defaults: Lorenzo1DPredictor, AdaptiveBitpackCoder, EPL=1
    const std::string src2 = fused::generateWarpFusionSource(s2);
    EXPECT_NE(src2.find("Lorenzo1DPredictor pred = Lorenzo1DPredictor::fromParams"),
              std::string::npos);
    EXPECT_NE(src2.find("fused_rate_body<1, AdaptiveBitpackCoder, Lorenzo1DPredictor>"), std::string::npos);
    EXPECT_NE(src2.find("fused_pack_body<1, AdaptiveBitpackCoder, Lorenzo1DPredictor>"), std::string::npos);
    EXPECT_NE(src2.find("#include \"fused/fused_block/warp_fusion.cuh\""), std::string::npos);
    EXPECT_NE(src2.find("fz_fused_warp_rate"), std::string::npos);
    EXPECT_NE(src2.find("fz_fused_warp_pack"), std::string::npos);

    // Swapping the predictor + EPL is a data change — no new C++.
    fused::WarpFusionSpec s3;
    s3.predictor = "TiledLorenzo2DPredictor"; s3.coder = "AdaptiveBitpackCoder"; s3.elems_per_lane = 2;
    const std::string src3 = fused::generateWarpFusionSource(s3);
    EXPECT_NE(src3.find("TiledLorenzo2DPredictor::fromParams"), std::string::npos);
    EXPECT_NE(src3.find("fused_rate_body<2, AdaptiveBitpackCoder, TiledLorenzo2DPredictor>"),
              std::string::npos);
    EXPECT_EQ(src3.find("Lorenzo1DPredictor"), std::string::npos);

    // Swapping the CODER is likewise a data change — the whole point of this update.
    fused::WarpFusionSpec s4;
    s4.predictor = "Lorenzo1DPredictor"; s4.coder = "PlainBitpackCoder"; s4.elems_per_lane = 1;
    const std::string src4 = fused::generateWarpFusionSource(s4);
    EXPECT_NE(src4.find("fused_rate_body<1, PlainBitpackCoder, Lorenzo1DPredictor>"), std::string::npos);
    EXPECT_NE(src4.find("fused_pack_body<1, PlainBitpackCoder, Lorenzo1DPredictor>"), std::string::npos);
    EXPECT_EQ(src4.find("AdaptiveBitpackCoder"), std::string::npos);

    // Composing a TRANSFORM between predictor and coder: another data change, no new C++.
    fused::WarpFusionSpec s5;
    s5.predictor = "Lorenzo1DPredictor"; s5.transforms = {"ZigzagTransform"};
    s5.coder = "AdaptiveBitpackCoder"; s5.elems_per_lane = 1;
    const std::string src5 = fused::generateWarpFusionSource(s5);
    EXPECT_NE(src5.find("fused_rate_body<1, AdaptiveBitpackCoder, Lorenzo1DPredictor, ZigzagTransform>"),
              std::string::npos);
}

// Warp transform chain: a register→register op (Zigzag/TCMS) composed BETWEEN the
// predictor and the coder — proving the warp path fuses more than a fixed 3-stage
// shape (Quant→Predictor→Transform*→Coder), like the chunk path. The staged pipeline
// (Quant→Lorenzo→Zigzag→AdaptiveBitpack) is the byte-identity oracle: the fused kernel
// applies the same zigzag to the deltas before packing. Zigzag runs byte-transparent so
// its uint32 output feeds AdaptiveBitpack<int32> unchecked (the LC "TCMS" role).
TEST(FusionPlanner, WarpTransformChainZigzagFusesMatchesStaged) {
    const size_t n = 1u << 20;
    const float  eb = 1e-3f;
    std::vector<float> h(n);
    for (size_t i = 0; i < n; ++i)
        h[i] = 0.6f*std::sin(i*0.001f) + 0.3f*std::cos(i*0.017f);
    const double bound = eb * 1.001;   // ABS
    const size_t bytes = n * sizeof(float);
    float* d_in = nullptr; ASSERT_EQ(cudaMalloc(&d_in, bytes), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_in, h.data(), bytes, cudaMemcpyHostToDevice), cudaSuccess);

    auto build = [&](Pipeline& p) {
        p.setDims(n, 1, 1);
        auto* q = p.addStage<QuantizerStage<float, uint32_t>>();
        q->setErrorBound(eb); q->setErrorBoundMode(ErrorBoundMode::ABS); q->setLinearMode(true);
        auto* l = p.addStage<LorenzoStage<int32_t>>(); l->setBlockSize(32);
        p.connect(l, q, "codes");
        auto* z = p.addStage<ZigzagStage<int32_t>>(); z->setByteTransparent(true);
        p.connect(z, l);
        auto* a = p.addStage<AdaptiveBitpackStage<int32_t>>();
        a->setBlockSize(32); a->setOutlierSelection(true);
        p.connect(a, z);
    };

    // Planner groups the 4-stage chain as one warp group.
    { Pipeline pg(bytes, MemoryStrategy::PREALLOCATE, 2.0f); build(pg); pg.finalize();
      auto gr = planFusionGroups(*pg.getDAG());
      ASSERT_EQ(gr.size(), 1u); EXPECT_EQ(gr[0].stages.size(), 4u); }

    auto compressCopy = [&](FusionPolicy pol, std::vector<uint8_t>& out) -> size_t {
        Pipeline p(bytes, MemoryStrategy::PREALLOCATE, 2.0f);
        p.setFusionPolicy(pol);
        build(p); p.finalize();
        EXPECT_EQ(p.getFusedGroupCount(), pol == FusionPolicy::Auto ? 1u : 0u);
        void* d_comp = nullptr; size_t sz = 0;
        p.compress(d_in, bytes, &d_comp, &sz, 0);
        cudaDeviceSynchronize();
        out.resize(sz);
        cudaMemcpy(out.data(), d_comp, sz, cudaMemcpyDeviceToHost);
        return sz;
    };
    std::vector<uint8_t> staged, fused;
    const size_t ss = compressCopy(FusionPolicy::Off, staged);
    const size_t sf = compressCopy(FusionPolicy::Auto, fused);
    ASSERT_EQ(ss, sf) << "fused archive size differs from staged";
    EXPECT_EQ(staged, fused) << "fused (with transform) archive not byte-identical to staged";

    auto roundtrip = [&](FusionPolicy pol, std::vector<float>& recon) {
        Pipeline p(bytes, MemoryStrategy::PREALLOCATE, 2.0f);
        p.setFusionPolicy(pol);
        build(p); p.finalize();
        void* d_comp = nullptr; size_t sz = 0;
        p.compress(d_in, bytes, &d_comp, &sz, 0);
        void* d_dec = nullptr; size_t dsz = 0;
        p.decompress(d_comp, sz, &d_dec, &dsz, 0);
        cudaDeviceSynchronize();
        recon.assign(n, 0.0f);
        cudaMemcpy(recon.data(), d_dec, bytes, cudaMemcpyDeviceToHost);
    };
    auto maxErr = [&](const std::vector<float>& r) {
        double m = 0; for (size_t i = 0; i < n; ++i) m = std::max(m, (double)std::abs(r[i]-h[i]));
        return m;
    };
    std::vector<float> rs, rf;
    roundtrip(FusionPolicy::Off,  rs);
    roundtrip(FusionPolicy::Auto, rf);
    EXPECT_LE(maxErr(rf), bound) << "fused transform-chain round-trip exceeds bound";
    EXPECT_EQ(rs, rf) << "fused reconstruction differs from staged";
    cudaFree(d_in);
}

// cuSZp3 3-D (PROTOTYPE): Quantizer(linear) -> TiledLorenzo(4x4x4) -> AdaptiveBitpack(64)
// on a 3-D field must fuse (TiledLorenzo3DPredictor warp op) byte-identical to staged and
// round-trip. Dims not multiples of 4 -> padded tiles (exercises the padding path).
TEST(FusionPlanner, Cuszp3_3D_FusesMatchesStaged) {
    const size_t dx = 34, dy = 30, dz = 28;   // none a multiple of 4 -> padded 3-D tiles
    const size_t n  = dx * dy * dz;
    const float  eb = 1e-3f;
    std::vector<float> h(n);
    for (size_t z = 0; z < dz; ++z)
      for (size_t y = 0; y < dy; ++y)
        for (size_t x = 0; x < dx; ++x)
          h[(z*dy+y)*dx+x] = 0.5f*std::sin(x*0.03f)+0.3f*std::cos(y*0.05f)+0.2f*std::sin(z*0.02f);
    const size_t bytes = n * sizeof(float);
    float* d_in = nullptr; ASSERT_EQ(cudaMalloc(&d_in, bytes), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_in, h.data(), bytes, cudaMemcpyHostToDevice), cudaSuccess);

    auto build = [&](Pipeline& p) {
        p.setDims(dx, dy, dz);
        auto* q = p.addStage<QuantizerStage<float, uint32_t>>();
        q->setErrorBound(eb); q->setErrorBoundMode(ErrorBoundMode::ABS); q->setLinearMode(true);
        auto* tl = p.addStage<TiledLorenzoStage<int32_t>>(); tl->setTileShape(4, 4, 4);
        p.connect(tl, q, "codes");
        auto* a = p.addStage<AdaptiveBitpackStage<int32_t>>();
        a->setBlockSize(64); a->setOutlierSelection(true);
        p.connect(a, tl);
    };
    { Pipeline pg(bytes, MemoryStrategy::PREALLOCATE, 2.0f); build(pg); pg.finalize();
      auto gr = planFusionGroups(*pg.getDAG());
      ASSERT_EQ(gr.size(), 1u); EXPECT_EQ(gr[0].stages.size(), 3u);
      EXPECT_EQ(gr[0].stages[1]->getFusedOp().op_name, "TiledLorenzo3DPredictor"); }

    auto compressCopy = [&](FusionPolicy pol, std::vector<uint8_t>& out) -> size_t {
        Pipeline p(bytes, MemoryStrategy::PREALLOCATE, 2.0f); p.setFusionPolicy(pol);
        build(p); p.finalize();
        EXPECT_EQ(p.getFusedGroupCount(), pol == FusionPolicy::Auto ? 1u : 0u);
        void* d_comp = nullptr; size_t sz = 0;
        p.compress(d_in, bytes, &d_comp, &sz, 0); cudaDeviceSynchronize();
        out.resize(sz); cudaMemcpy(out.data(), d_comp, sz, cudaMemcpyDeviceToHost);
        return sz;
    };
    std::vector<uint8_t> staged, fused;
    ASSERT_EQ(compressCopy(FusionPolicy::Off, staged), compressCopy(FusionPolicy::Auto, fused));
    EXPECT_EQ(staged, fused) << "fused 3-D archive not byte-identical to staged";

    auto roundtrip = [&](FusionPolicy pol, std::vector<float>& recon) {
        Pipeline p(bytes, MemoryStrategy::PREALLOCATE, 2.0f); p.setFusionPolicy(pol);
        build(p); p.finalize();
        void* d_comp=nullptr; size_t sz=0; p.compress(d_in, bytes, &d_comp, &sz, 0);
        void* d_dec=nullptr; size_t dsz=0; p.decompress(d_comp, sz, &d_dec, &dsz, 0);
        cudaDeviceSynchronize(); recon.assign(n,0.0f);
        cudaMemcpy(recon.data(), d_dec, bytes, cudaMemcpyDeviceToHost);
    };
    std::vector<float> rs, rf; roundtrip(FusionPolicy::Off, rs); roundtrip(FusionPolicy::Auto, rf);
    double m=0; for (size_t i=0;i<n;++i) m=std::max(m,(double)std::abs(rf[i]-h[i]));
    EXPECT_LE(m, eb*1.001) << "fused 3-D round-trip exceeds bound";
    EXPECT_EQ(rs, rf) << "fused 3-D reconstruction differs from staged";
    cudaFree(d_in);
}

// The swappable-coder payoff: fuse the SAME warp chain with a different Cooperative
// sink (PlainBitpackCoder) composed in by a data change — no new launcher/dispatch,
// mirroring how RARE/RAZE proved chunk generality. PlainBitpack emits an
// AdaptiveBitpack-decodable archive (all blocks plain-mode), so the existing inverse
// round-trips it; it never beats AdaptiveBitpack on size (AB picks the per-block min).
TEST(FusionPlanner, WarpSwappableCoderPlainBitpack) {
    const size_t n = 1u << 20;
    const float  eb = 1e-3f;
    std::vector<float> h(n);
    for (size_t i = 0; i < n; ++i)
        h[i] = 0.6f*std::sin(i*0.001f) + 0.3f*std::cos(i*0.017f) + 0.05f*std::sin(i*0.13f);
    const double bound = eb * 1.001;   // ABS
    const size_t bytes = n * sizeof(float);
    float* d_in = nullptr; ASSERT_EQ(cudaMalloc(&d_in, bytes), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_in, h.data(), bytes, cudaMemcpyHostToDevice), cudaSuccess);

    auto run = [&](const char* coder, std::vector<float>& recon) -> size_t {
        Pipeline p(bytes, MemoryStrategy::PREALLOCATE, 2.0f);
        p.setFusionPolicy(FusionPolicy::Auto);
        p.setDims(n, 1, 1);
        auto* q = p.addStage<QuantizerStage<float, uint32_t>>();
        q->setErrorBound(eb); q->setErrorBoundMode(ErrorBoundMode::ABS); q->setLinearMode(true);
        auto* l = p.addStage<LorenzoStage<int32_t>>(); l->setBlockSize(32);
        p.connect(l, q, "codes");
        auto* a = p.addStage<AdaptiveBitpackStage<int32_t>>();
        a->setBlockSize(32); a->setOutlierSelection(true);
        a->setFusedCoder(coder);
        p.connect(a, l);
        p.finalize();
        EXPECT_EQ(p.getFusedGroupCount(), 1u) << "chain must fuse with coder " << coder;
        void* d_comp = nullptr; size_t sz = 0;
        p.compress(d_in, bytes, &d_comp, &sz, 0);
        void* d_dec = nullptr; size_t dsz = 0;
        p.decompress(d_comp, sz, &d_dec, &dsz, 0);
        cudaDeviceSynchronize();
        recon.assign(n, 0.0f);
        cudaMemcpy(recon.data(), d_dec, bytes, cudaMemcpyDeviceToHost);
        return sz;
    };
    auto maxErr = [&](const std::vector<float>& r) {
        double m = 0; for (size_t i = 0; i < n; ++i) m = std::max(m, (double)std::abs(r[i]-h[i]));
        return m;
    };
    std::vector<float> rAB, rPlain;
    const size_t szAB    = run("AdaptiveBitpackCoder", rAB);
    const size_t szPlain = run("PlainBitpackCoder",    rPlain);
    EXPECT_LE(maxErr(rAB),    bound) << "adaptive-coder fused round-trip exceeds bound";
    EXPECT_LE(maxErr(rPlain), bound) << "plain-coder fused round-trip exceeds bound";
    EXPECT_GE(szPlain, szAB) << "plain must never beat adaptive (AB picks the per-block min)";
    cudaFree(d_in);
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
