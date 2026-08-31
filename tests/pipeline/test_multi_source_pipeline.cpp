/**
 * tests/pipeline/test_multi_source_pipeline.cpp
 *
 * Integration tests for Pipeline's multi-source capability (setPrimarySource()):
 * more than one stage with zero forward inputs, each reading the pipeline's
 * external buffer directly, no TeeStage/duplicate-copy node required.
 *
 * Two independent, unconnected source stages (Cdf97Stage and QuantizerStage)
 * both bind to the same external float buffer. Their forward+inverse
 * behaviors are numerically distinguishable (DWT round trip is near-exact;
 * quantization is lossy at the configured bound), which lets these tests
 * confirm decompress() actually returns the *designated* primary source's
 * answer, not just *a* plausible one -- and that sharing one physical buffer
 * across two DAG input ports doesn't corrupt either stage's read (both must
 * see the untouched original field).
 *
 *   MS1  PrimaryDwtGivesNearExactReconstruction
 *   MS2  PrimaryQuantizerGivesLossyReconstruction
 *   MS3  NoAliasingCorruptionBetweenSources   — cross-check against single-source runs
 *   MS4  UnsetPrimaryDefaultsToFirstAddedSourceWithWarning
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

std::vector<float> make_field(int nx, int ny, uint64_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    std::vector<float> v((size_t)nx * ny);
    for (auto& x : v) x = (float)u(rng);
    return v;
}

float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float m = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}

// Two unconnected sources sharing one external buffer: dwt (near-exact
// round trip) and quant (lossy at `bound`). Neither is downstream of the
// other -- both have zero forward inputs from other stages, so both qualify
// as sources (Pipeline::getSourceStages(): node->dependencies.empty()).
void build_two_source_pipeline(Pipeline& p, int nx, int ny, float bound,
                                Cdf97Stage<float>** out_dwt = nullptr,
                                QuantizerStage<float, uint32_t>** out_quant = nullptr) {
    p.setDims(nx, ny, 1);
    auto* dwt = p.addStage<Cdf97Stage<float>>();
    auto* quant = p.addStage<QuantizerStage<float, uint32_t>>();
    quant->setErrorBound(bound);
    quant->setErrorBoundMode(ErrorBoundMode::ABS);
    quant->setLinearMode(true);
    // Deliberately no connect() calls between them -- both bind directly to
    // the pipeline's external input.
    p.finalize();
    if (out_dwt) *out_dwt = dwt;
    if (out_quant) *out_quant = quant;
}

} // namespace

TEST(MultiSourcePipeline, PrimaryDwtGivesNearExactReconstruction) {
    CudaStream stream;
    const int nx = 96, ny = 64;
    const float bound = 1e-2f;
    auto h_field = make_field(nx, ny, 41);
    const size_t bytes = h_field.size() * sizeof(float);

    CudaBuffer<float> d_in(h_field.size());
    d_in.upload(h_field, stream);
    stream.sync();

    Pipeline p(bytes, MemoryStrategy::PREALLOCATE, /*pool_mult=*/8.0f);
    Cdf97Stage<float>* dwt = nullptr;
    build_two_source_pipeline(p, nx, ny, bound, &dwt, nullptr);
    p.setPrimarySource(dwt);

    void* d_comp = nullptr; size_t comp_sz = 0;
    p.compress(d_in.void_ptr(), bytes, &d_comp, &comp_sz, stream);
    void* d_dec = nullptr; size_t dec_sz = 0;
    p.decompress(nullptr, comp_sz, &d_dec, &dec_sz, stream);
    ASSERT_EQ(dec_sz, bytes);

    std::vector<float> recon(h_field.size());
    FZ_TEST_CUDA(cudaMemcpy(recon.data(), d_dec, bytes, cudaMemcpyDeviceToHost));

    // CDF 9/7 forward+inverse is a near-identity round trip (float rounding
    // only) -- orders of magnitude tighter than the quantizer's bound would
    // ever produce. That gap is what lets this test tell "we picked dwt" from
    // "we picked quant" without inspecting internals.
    float err = max_abs_diff(recon, h_field);
    EXPECT_LT(err, 1e-4f) << "expected DWT-quality round trip, got quantizer-quality error " << err;
}

TEST(MultiSourcePipeline, PrimaryQuantizerGivesLossyReconstruction) {
    CudaStream stream;
    const int nx = 96, ny = 64;
    const float bound = 1e-2f;
    auto h_field = make_field(nx, ny, 41);  // same seed as MS1: same input, different primary
    const size_t bytes = h_field.size() * sizeof(float);

    CudaBuffer<float> d_in(h_field.size());
    d_in.upload(h_field, stream);
    stream.sync();

    Pipeline p(bytes, MemoryStrategy::PREALLOCATE, /*pool_mult=*/8.0f);
    QuantizerStage<float, uint32_t>* quant = nullptr;
    build_two_source_pipeline(p, nx, ny, bound, nullptr, &quant);
    p.setPrimarySource(quant);

    void* d_comp = nullptr; size_t comp_sz = 0;
    p.compress(d_in.void_ptr(), bytes, &d_comp, &comp_sz, stream);
    void* d_dec = nullptr; size_t dec_sz = 0;
    p.decompress(nullptr, comp_sz, &d_dec, &dec_sz, stream);
    ASSERT_EQ(dec_sz, bytes);

    std::vector<float> recon(h_field.size());
    FZ_TEST_CUDA(cudaMemcpy(recon.data(), d_dec, bytes, cudaMemcpyDeviceToHost));

    float err = max_abs_diff(recon, h_field);
    EXPECT_GT(err, 1e-4f) << "expected quantizer-quality (lossy, ~bound) error, got DWT-quality " << err;
    EXPECT_LE(err, bound * 1.0001f) << "quantizer's own bound must still hold";
}

TEST(MultiSourcePipeline, NoAliasingCorruptionBetweenSources) {
    // Compare the multi-source pipeline's dwt-primary reconstruction against
    // a plain single-source Cdf97Stage-only pipeline fed the identical input.
    // If sharing one external pointer across two DAG input-buffer IDs
    // corrupted either stage's read (e.g. one clobbering the other's view),
    // this diverges; compute-sanitizer racecheck on this same test path
    // additionally covers the device-side race itself.
    CudaStream stream;
    const int nx = 64, ny = 64;
    const float bound = 5e-2f;
    auto h_field = make_field(nx, ny, 77);
    const size_t bytes = h_field.size() * sizeof(float);

    CudaBuffer<float> d_in(h_field.size());
    d_in.upload(h_field, stream);
    stream.sync();

    // Multi-source: dwt + quant sharing the buffer, primary = dwt.
    std::vector<float> multi_recon(h_field.size());
    {
        Pipeline p(bytes, MemoryStrategy::PREALLOCATE, /*pool_mult=*/8.0f);
        Cdf97Stage<float>* dwt = nullptr;
        build_two_source_pipeline(p, nx, ny, bound, &dwt, nullptr);
        p.setPrimarySource(dwt);

        void* d_comp = nullptr; size_t comp_sz = 0;
        p.compress(d_in.void_ptr(), bytes, &d_comp, &comp_sz, stream);
        void* d_dec = nullptr; size_t dec_sz = 0;
        p.decompress(nullptr, comp_sz, &d_dec, &dec_sz, stream);
        ASSERT_EQ(dec_sz, bytes);
        FZ_TEST_CUDA(cudaMemcpy(multi_recon.data(), d_dec, bytes, cudaMemcpyDeviceToHost));
    }

    // Single-source: dwt alone, same input.
    std::vector<float> single_recon(h_field.size());
    {
        Pipeline p(bytes, MemoryStrategy::PREALLOCATE, /*pool_mult=*/8.0f);
        p.setDims(nx, ny, 1);
        p.addStage<Cdf97Stage<float>>();
        p.finalize();

        void* d_comp = nullptr; size_t comp_sz = 0;
        p.compress(d_in.void_ptr(), bytes, &d_comp, &comp_sz, stream);
        void* d_dec = nullptr; size_t dec_sz = 0;
        p.decompress(nullptr, comp_sz, &d_dec, &dec_sz, stream);
        ASSERT_EQ(dec_sz, bytes);
        FZ_TEST_CUDA(cudaMemcpy(single_recon.data(), d_dec, bytes, cudaMemcpyDeviceToHost));
    }

    EXPECT_EQ(max_abs_diff(multi_recon, single_recon), 0.0f)
        << "sharing the external buffer across two source stages changed dwt's own result";
}

TEST(MultiSourcePipeline, UnsetPrimaryDefaultsToFirstAddedSourceWithWarning) {
    // No setPrimarySource() call: must not throw, must deterministically
    // return the first-added source's (dwt's) answer -- matches the
    // documented default in setPrimarySource()'s doc comment.
    CudaStream stream;
    const int nx = 48, ny = 48;
    const float bound = 1e-2f;
    auto h_field = make_field(nx, ny, 5);
    const size_t bytes = h_field.size() * sizeof(float);

    CudaBuffer<float> d_in(h_field.size());
    d_in.upload(h_field, stream);
    stream.sync();

    Pipeline p(bytes, MemoryStrategy::PREALLOCATE, /*pool_mult=*/8.0f);
    build_two_source_pipeline(p, nx, ny, bound, nullptr, nullptr);  // dwt added first

    void* d_comp = nullptr; size_t comp_sz = 0;
    p.compress(d_in.void_ptr(), bytes, &d_comp, &comp_sz, stream);
    void* d_dec = nullptr; size_t dec_sz = 0;
    p.decompress(nullptr, comp_sz, &d_dec, &dec_sz, stream);
    ASSERT_EQ(dec_sz, bytes);

    std::vector<float> recon(h_field.size());
    FZ_TEST_CUDA(cudaMemcpy(recon.data(), d_dec, bytes, cudaMemcpyDeviceToHost));
    EXPECT_LT(max_abs_diff(recon, h_field), 1e-4f) << "default primary should be dwt (first-added)";
}
