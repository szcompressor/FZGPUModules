/**
 * tests/pipeline/test_sperr_gpu_bounded.cpp
 *
 * Integration tests for the DAG-integrated GPU SPERR bound-guarantee pipeline:
 * Cdf97Stage -> QuantizerStage -> Cdf97OutlierCorrectStage -> Speck2DStage,
 * with Cdf97Stage and Cdf97OutlierCorrectStage both bound directly to the
 * pipeline's raw input via Pipeline::bindExternalInput() -- no fan-out/
 * duplicate-copy stage. See
 * `modules/coders/outlier_correct/outlier_correct_stage.h` for the port
 * shape/mechanism and `Pipeline::bindExternalInput()`'s doc comment for why
 * no such stage is needed, and `memory/speck_gpu_design.md` sec.9 for the
 * full design history (the earlier TeeStage-based version this replaced).
 *
 *   SB1  BoundGuaranteed                 — round trip actually respects the bound (in-DAG decompress)
 *   SB2  RepeatedCompressDecompressStable — same class of reuse the Speck2DStage cached_n_ bug hid in
 *   SB3  TighterBoundStillGuaranteed     — a second, tighter bound on the same field
 *   SB4  Cdf97OutlierCorrectStageMetadata — port counts, type id, output names
 */

#include <gtest/gtest.h>
#include "helpers/fz_test_utils.h"
#include "fzgpumodules.h"

#include <cmath>
#include <random>
#include <vector>

using namespace fz;
using namespace fz_test;

namespace {

std::vector<float> make_clustered_field(int nx, int ny, uint64_t seed) {
    std::mt19937 rng(seed);
    std::vector<float> v((size_t)nx * ny);
    for (int y = 0; y < ny; ++y)
        for (int x = 0; x < nx; ++x) {
            double s = std::exp(-6.0 * ((double)x / nx + (double)y / ny));
            std::uniform_real_distribution<double> u(-1.0, 1.0);
            v[(size_t)y * nx + x] = (float)(u(rng) * (0.1 + s));
        }
    return v;
}

// Pipeline is non-copyable/non-movable (unique_ptr members) -- caller
// constructs it in place and passes it by reference to wire stages into it.
void build_bounded_pipeline(Pipeline& p, int nx, int ny, float bound) {
    p.setDims(nx, ny, 1);

    auto* dwt = p.addStage<Cdf97Stage<float>>();   // pure source: auto-discovered, no
                                                    // bindExternalInput() call needed

    auto* quant = p.addStage<QuantizerStage<float, uint32_t>>();
    quant->setErrorBound(bound);
    quant->setErrorBoundMode(ErrorBoundMode::ABS);
    quant->setLinearMode(true);

    auto* corr = p.addStage<Cdf97OutlierCorrectStage>();
    corr->setErrorBound(bound);
    p.bindExternalInput(corr);   // corr.input[0] = raw field -- BEFORE the connect()
                                 // below so the external buffer lands on port 0.

    auto* speck = p.addStage<Speck2DStage>();

    p.connect(quant, dwt);
    p.connect(corr, quant, "codes");     // corr.input[1] = codes
    p.connect(speck, corr, "codes");     // Speck2D consumes corr's codes passthrough

    p.setPrimarySource(corr);   // decompress() returns corr's corrected field, not dwt's own
    p.finalize();
}

} // namespace

TEST(SperrGpuBounded, BoundGuaranteed) {
    CudaStream stream;
    const int nx = 128, ny = 128;
    const float bound = 1e-3f;
    auto h_field = make_clustered_field(nx, ny, 11);
    const size_t bytes = h_field.size() * sizeof(float);

    CudaBuffer<float> d_in(h_field.size());
    d_in.upload(h_field, stream);
    stream.sync();

    Pipeline p(bytes, MemoryStrategy::PREALLOCATE, /*pool_mult=*/8.0f);
    build_bounded_pipeline(p, nx, ny, bound);

    void* d_comp = nullptr; size_t comp_sz = 0;
    p.compress(d_in.void_ptr(), bytes, &d_comp, &comp_sz, stream);
    ASSERT_GT(comp_sz, 0u);

    void* d_dec = nullptr; size_t dec_sz = 0;
    p.decompress(nullptr, comp_sz, &d_dec, &dec_sz, stream);
    ASSERT_EQ(dec_sz, bytes);

    std::vector<float> recon(h_field.size());
    FZ_TEST_CUDA(cudaMemcpy(recon.data(), d_dec, bytes, cudaMemcpyDeviceToHost));

    float max_err = 0.0f;
    for (size_t i = 0; i < h_field.size(); ++i) max_err = std::max(max_err, std::fabs(recon[i] - h_field[i]));
    EXPECT_LE(max_err, bound * 1.0001f) << "reconstructed error " << max_err << " exceeds the guaranteed bound " << bound;
}

TEST(SperrGpuBounded, RepeatedCompressDecompressStable) {
    // The exact reuse pattern (same Pipeline object, in-DAG decompress via
    // d_input=nullptr, no file/header round trip) that hid the Speck2DStage
    // cached_n_ bug and the QuantizerStage inverse-header bug -- both were
    // invisible in a single-shot or file-based round trip.
    CudaStream stream;
    const int nx = 96, ny = 80;
    const float bound = 1e-2f;
    auto h_field = make_clustered_field(nx, ny, 22);
    const size_t bytes = h_field.size() * sizeof(float);

    CudaBuffer<float> d_in(h_field.size());
    d_in.upload(h_field, stream);
    stream.sync();

    Pipeline p(bytes, MemoryStrategy::PREALLOCATE, /*pool_mult=*/8.0f);
    build_bounded_pipeline(p, nx, ny, bound);

    for (int iter = 0; iter < 3; ++iter) {
        void* d_comp = nullptr; size_t comp_sz = 0;
        p.compress(d_in.void_ptr(), bytes, &d_comp, &comp_sz, stream);
        ASSERT_GT(comp_sz, 0u) << "iteration " << iter;

        void* d_dec = nullptr; size_t dec_sz = 0;
        p.decompress(nullptr, comp_sz, &d_dec, &dec_sz, stream);
        ASSERT_EQ(dec_sz, bytes) << "iteration " << iter;

        std::vector<float> recon(h_field.size());
        FZ_TEST_CUDA(cudaMemcpy(recon.data(), d_dec, bytes, cudaMemcpyDeviceToHost));
        float max_err = 0.0f;
        for (size_t i = 0; i < h_field.size(); ++i) max_err = std::max(max_err, std::fabs(recon[i] - h_field[i]));
        EXPECT_LE(max_err, bound * 1.0001f) << "iteration " << iter << ": error " << max_err;
    }
}

TEST(SperrGpuBounded, TighterBoundStillGuaranteed) {
    CudaStream stream;
    const int nx = 100, ny = 60;
    const float bound = 1e-4f;
    auto h_field = make_clustered_field(nx, ny, 33);
    const size_t bytes = h_field.size() * sizeof(float);

    CudaBuffer<float> d_in(h_field.size());
    d_in.upload(h_field, stream);
    stream.sync();

    Pipeline p(bytes, MemoryStrategy::PREALLOCATE, /*pool_mult=*/8.0f);
    build_bounded_pipeline(p, nx, ny, bound);

    void* d_comp = nullptr; size_t comp_sz = 0;
    p.compress(d_in.void_ptr(), bytes, &d_comp, &comp_sz, stream);
    void* d_dec = nullptr; size_t dec_sz = 0;
    p.decompress(nullptr, comp_sz, &d_dec, &dec_sz, stream);
    ASSERT_EQ(dec_sz, bytes);

    std::vector<float> recon(h_field.size());
    FZ_TEST_CUDA(cudaMemcpy(recon.data(), d_dec, bytes, cudaMemcpyDeviceToHost));
    float max_err = 0.0f;
    for (size_t i = 0; i < h_field.size(); ++i) max_err = std::max(max_err, std::fabs(recon[i] - h_field[i]));
    EXPECT_LE(max_err, bound * 1.0001f);
}

TEST(SperrGpuBounded, Cdf97OutlierCorrectStageMetadata) {
    Cdf97OutlierCorrectStage s;
    EXPECT_EQ(s.getName(), "Cdf97OutlierCorrect");
    EXPECT_EQ(s.getStageTypeId(), static_cast<uint16_t>(StageType::CDF97_OUTLIER_CORRECT));
    EXPECT_EQ(s.getNumInputs(), 2u);
    EXPECT_EQ(s.getNumOutputs(), 2u);
    auto fwd_names = s.getOutputNames();
    EXPECT_EQ(fwd_names[0], "correction");
    EXPECT_EQ(fwd_names[1], "codes");
    s.setInverse(true);
    auto inv_names = s.getOutputNames();
    EXPECT_EQ(inv_names[0], "field");
    EXPECT_EQ(inv_names[1], "codes");
}
