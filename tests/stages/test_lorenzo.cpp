/**
 * tests/stages/test_lorenzo.cpp
 *
 * Unit tests for LorenzoStage<T> (lossless delta predictor) and
 * LorenzoQuantStage<TIn, TCode> (fused lossy predictor + quantizer).
 *
 *   LZ1  LorenzoStage/RoundTrip1D                — 1-D int32 ramp, exact reconstruction
 *   LZ2  LorenzoStage/RoundTrip2D                — 2-D int32 grid, exact reconstruction
 *   LZ3  LorenzoStage/RoundTrip3D                — 3-D int32 grid, exact reconstruction
 *   LZ4  LorenzoStage/SerializeDeserialize       — config round-trip via header bytes
 *   LZ5  LorenzoStage/StageTypeId               — getStageTypeId() == StageType::LORENZO
 *   LZ6  LorenzoStage/GraphCompatible            — forward stage returns isGraphCompatible()=true
 *   LZ7  LorenzoStage/Int16RoundTrip             — int16_t type instantiation, exact reconstruction
 *   LZ8  LorenzoQuantStage/DeterministicRecon    — two independent pipelines produce identical output
 *   LZ9  LorenzoStage/QuantizerLorenzoPipeline   — Quantizer→Lorenzo chained round-trip
 */

#include <gtest/gtest.h>
#include <cstdint>
#include <numeric>
#include <vector>

#include "predictors/lorenzo/lorenzo_stage.h"
#include "fused/lorenzo_quant/lorenzo_quant.h"
#include "quantizers/quantizer/quantizer.h"
#include "helpers/stage_harness.h"
#include "helpers/fz_test_utils.h"

using namespace fz;
using namespace fz_test;

// ─────────────────────────────────────────────────────────────────────────────
// LZ1: RoundTrip1D — 1-D int32 ramp reconstructs exactly
// ─────────────────────────────────────────────────────────────────────────────
TEST(LorenzoStage, RoundTrip1D) {
    const size_t N = 4096;
    const size_t in_bytes = N * sizeof(int32_t);

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

// ─────────────────────────────────────────────────────────────────────────────
// LZ2: RoundTrip2D — 2-D int32 grid reconstructs exactly
// ─────────────────────────────────────────────────────────────────────────────
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

// ─────────────────────────────────────────────────────────────────────────────
// LZ3: RoundTrip3D — 3-D int32 grid reconstructs exactly
// ─────────────────────────────────────────────────────────────────────────────
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

// ─────────────────────────────────────────────────────────────────────────────
// LZ4: SerializeDeserialize — dims survive serializeHeader/deserializeHeader
// ─────────────────────────────────────────────────────────────────────────────
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

// ─────────────────────────────────────────────────────────────────────────────
// LZ5: StageTypeId — getStageTypeId() returns StageType::LORENZO
// ─────────────────────────────────────────────────────────────────────────────
TEST(LorenzoStage, StageTypeId) {
    EXPECT_EQ(LorenzoStage<int32_t>().getStageTypeId(),
              static_cast<uint16_t>(StageType::LORENZO));
}

// ─────────────────────────────────────────────────────────────────────────────
// LZ6: GraphCompatible — forward stage returns isGraphCompatible() = true
// ─────────────────────────────────────────────────────────────────────────────
TEST(LorenzoStage, GraphCompatible) {
    EXPECT_TRUE(LorenzoStage<int32_t>().isGraphCompatible());
}

// ─────────────────────────────────────────────────────────────────────────────
// LZ7: Int16RoundTrip — int16_t instantiation, exact reconstruction
// ─────────────────────────────────────────────────────────────────────────────
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

// ─────────────────────────────────────────────────────────────────────────────
// LZ8: DeterministicReconstruction — two independent LorenzoQuantStage
//       pipelines compress the same input and produce element-wise identical
//       reconstructions, guarding against non-deterministic GPU atomics.
// ─────────────────────────────────────────────────────────────────────────────
TEST(LorenzoQuantStage, DeterministicReconstruction) {
    const size_t N  = 4096;
    const float  EB = 1e-2f;
    auto h_input = make_smooth_data<float>(N);

    auto make_pipeline = [&]() {
        auto p = std::make_unique<Pipeline>(N * sizeof(float), MemoryStrategy::PREALLOCATE);
        auto* lq = p->addStage<LorenzoQuantStage<float, uint16_t>>();
        lq->setErrorBound(EB);
        lq->setQuantRadius(512);
        lq->setOutlierCapacity(0.2f);
        lq->setDims(N);
        p->finalize();
        return p;
    };

    CudaStream cs;
    auto p1 = make_pipeline();
    auto p2 = make_pipeline();

    auto res1 = pipeline_round_trip<float>(*p1, h_input, cs.stream);
    auto res2 = pipeline_round_trip<float>(*p2, h_input, cs.stream);

    ASSERT_EQ(res1.data.size(), N);
    ASSERT_EQ(res2.data.size(), N);

    for (size_t i = 0; i < N; i++) {
        EXPECT_EQ(res1.data[i], res2.data[i])
            << "Mismatch at element " << i;
    }
    EXPECT_LE(res1.max_error, EB * 1.01);
}

// ─────────────────────────────────────────────────────────────────────────────
// LZ9: QuantizerLorenzoPipeline — Quantizer→Lorenzo chained round-trip.
//       Verifies that LorenzoStage can follow QuantizerStage in a pipeline:
//       Lorenzo delta-codes the quantizer's integer codes; the inverse path
//       undoes the delta then maps codes back to floats.
//
//       Uses decompress(nullptr, ...) because multi-output QuantizerStage
//       leaves compressed data in the pipeline's internal pool rather than
//       a single concatenated buffer — the standard pipeline_round_trip
//       pattern is not applicable for this topology.
// ─────────────────────────────────────────────────────────────────────────────
TEST(LorenzoStage, QuantizerLorenzoPipelineRoundTrip) {
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
