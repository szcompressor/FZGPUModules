#include <gtest/gtest.h>
#include <cstdint>
#include <numeric>
#include <vector>

#include "predictors/lorenzo/lorenzo_stage.h"
#include "helpers/stage_harness.h"
#include "helpers/fz_test_utils.h"

using namespace fz;
using namespace fz_test;

// ── RoundTrip1D ────────────────────────────────────────────────────────────
TEST(LorenzoStage, RoundTrip1D) {
    const size_t N = 4096;
    const size_t in_bytes = N * sizeof(int32_t);

    // Ramp input — predictable deltas, good for delta coding
    std::vector<int32_t> h_input(N);
    std::iota(h_input.begin(), h_input.end(), 0);

    Pipeline p(in_bytes, MemoryStrategy::PREALLOCATE);
    auto* stage = p.addStage<LorenzoStage<int32_t>>();
    stage->setDims(N);
    p.finalize();

    CudaStream cs;
    auto res = pipeline_round_trip<int32_t>(p, h_input, cs.stream);

    EXPECT_EQ(res.max_error, 0.0f);
}

// ── RoundTrip2D ────────────────────────────────────────────────────────────
TEST(LorenzoStage, RoundTrip2D) {
    const size_t NX = 64, NY = 64;
    const size_t N = NX * NY;
    const size_t in_bytes = N * sizeof(int32_t);

    std::vector<int32_t> h_input(N);
    for (size_t i = 0; i < N; ++i)
        h_input[i] = static_cast<int32_t>(i % 200);

    Pipeline p(in_bytes, MemoryStrategy::PREALLOCATE);
    auto* stage = p.addStage<LorenzoStage<int32_t>>();
    stage->setDims(NX, NY);
    p.finalize();

    CudaStream cs;
    auto res = pipeline_round_trip<int32_t>(p, h_input, cs.stream);

    EXPECT_EQ(res.max_error, 0.0f);
}

// ── RoundTrip3D ────────────────────────────────────────────────────────────
TEST(LorenzoStage, RoundTrip3D) {
    const size_t NX = 16, NY = 16, NZ = 16;
    const size_t N = NX * NY * NZ;
    const size_t in_bytes = N * sizeof(int32_t);

    std::vector<int32_t> h_input(N);
    for (size_t i = 0; i < N; ++i)
        h_input[i] = static_cast<int32_t>(i % 50);

    Pipeline p(in_bytes, MemoryStrategy::PREALLOCATE);
    auto* stage = p.addStage<LorenzoStage<int32_t>>();
    stage->setDims(NX, NY, NZ);
    p.finalize();

    CudaStream cs;
    auto res = pipeline_round_trip<int32_t>(p, h_input, cs.stream);

    EXPECT_EQ(res.max_error, 0.0f);
}

// ── SerializeDeserialize ──────────────────────────────────────────────────
TEST(LorenzoStage, SerializeDeserialize) {
    LorenzoStage<int32_t> original;
    original.setDims(128, 64, 2);

    uint8_t buf[128] = {};
    size_t written = original.serializeHeader(0, buf, sizeof(buf));
    EXPECT_EQ(written, sizeof(LorenzoConfig));

    LorenzoStage<int32_t> restored;
    restored.deserializeHeader(buf, written);

    EXPECT_EQ(original.getDims(), restored.getDims());
    EXPECT_EQ(original.ndim(), restored.ndim());
}

// ── StageTypeId ────────────────────────────────────────────────────────────
TEST(LorenzoStage, StageTypeId) {
    EXPECT_EQ(LorenzoStage<int32_t>().getStageTypeId(),
              static_cast<uint16_t>(StageType::LORENZO));
}

// ── GraphCompatible ───────────────────────────────────────────────────────
TEST(LorenzoStage, GraphCompatible) {
    EXPECT_TRUE(LorenzoStage<int32_t>().isGraphCompatible());
}

// ── Int16RoundTrip ────────────────────────────────────────────────────────
TEST(LorenzoStage, Int16RoundTrip) {
    const size_t N = 1024;
    const size_t in_bytes = N * sizeof(int16_t);

    std::vector<int16_t> h_input(N);
    for (size_t i = 0; i < N; ++i)
        h_input[i] = static_cast<int16_t>(i % 100);

    Pipeline p(in_bytes, MemoryStrategy::PREALLOCATE);
    auto* stage = p.addStage<LorenzoStage<int16_t>>();
    stage->setDims(N);
    p.finalize();

    CudaStream cs;
    auto res = pipeline_round_trip<int16_t>(p, h_input, cs.stream);

    EXPECT_EQ(res.max_error, 0.0f);
}

// ── Quantizer → Lorenzo round-trip (cuSZp-style, no entropy coding) ──────────
#include "predictors/quantizer/quantizer.h"

TEST(LorenzoStage, QuantizerLorenzoPipelineRoundTrip) {
    // Verifies that LorenzoStage can be chained after QuantizerStage.
    // Lorenzo receives the quantizer's integer codes and delta-codes them;
    // inverse reconstructs the codes, which Quantizer inverse maps back to floats.
    const size_t N = 4096;
    const size_t in_bytes = N * sizeof(float);
    auto h_input = make_smooth_data<float>(N);

    Pipeline p(in_bytes, MemoryStrategy::PREALLOCATE);

    auto* quant = p.addStage<QuantizerStage<float, uint32_t>>();
    quant->setErrorBound(0.01f);
    quant->setErrorBoundMode(ErrorBoundMode::ABS);
    quant->setQuantRadius(32768);
    quant->setOutlierCapacity(0.1f);
    quant->setZigzagCodes(false);

    auto* lrz = p.addStage<LorenzoStage<int32_t>>();
    lrz->setDims(N);
    p.connect(lrz, quant, "codes");

    p.finalize();

    CudaBuffer<float> d_in(N);
    CudaStream cs;
    d_in.upload(h_input, cs.stream);
    cudaStreamSynchronize(cs.stream);

    // Use nullptr decompress so the pipeline reads buffers from its internal
    // DAG — required for multi-output stages like Quantizer (4 ports) where
    // concat parsing via d_comp is not used.
    void* d_comp = nullptr; size_t comp_sz = 0;
    p.compress(d_in.void_ptr(), in_bytes, &d_comp, &comp_sz, cs.stream);
    cudaStreamSynchronize(cs.stream);

    void* d_dec = nullptr; size_t dec_sz = 0;
    p.decompress(nullptr, comp_sz, &d_dec, &dec_sz, cs.stream);
    cudaStreamSynchronize(cs.stream);

    ASSERT_EQ(dec_sz, in_bytes);
    std::vector<float> h_recon(N);
    cudaMemcpy(h_recon.data(), d_dec, dec_sz, cudaMemcpyDeviceToHost);

    float max_err = max_abs_error(h_input, h_recon);
    EXPECT_LT(max_err, 0.011f);
}
