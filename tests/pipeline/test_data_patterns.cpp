/**
 * tests/test_data_patterns.cpp
 *
 * Correctness tests for a variety of input data patterns run through a
 * full Lorenzo-only compress → decompress pipeline.
 *
 * The goal is to hit edge cases in the quantizer and outlier logic that
 * smooth sinusoidal data alone does not cover.
 *
 * Each test checks:
 *   - The pipeline does not crash.
 *   - The decompressed data has max absolute error ≤ error_bound.
 *   - (Where noted) additional properties like compression ratio.
 */

#include <gtest/gtest.h>
#include "helpers/fz_test_utils.h"
#include "fzgpumodules.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

using namespace fz;
using namespace fz_test;

// ─────────────────────────────────────────────────────────────────────────────
// Helper: build a Lorenzo pipeline, compress, decompress, return max error.
// comp_sz_out (optional) receives the compressed byte count.
// ─────────────────────────────────────────────────────────────────────────────
static float run_pattern(
    const std::vector<float>& h_input,
    float                     error_bound,
    int                       quant_radius     = 32768,
    float                     outlier_capacity = 0.25f,
    size_t*                   comp_sz_out      = nullptr)
{
    const size_t N       = h_input.size();
    const size_t in_bytes = N * sizeof(float);

    CudaStream stream;
    CudaBuffer<float> d_in(N);
    d_in.upload(h_input, stream);
    stream.sync();

    Pipeline pipeline(in_bytes, MemoryStrategy::MINIMAL);
    auto* lrz = pipeline.addStage<LorenzoStage<float, uint16_t>>();
    lrz->setErrorBound(error_bound);
    lrz->setQuantRadius(quant_radius);
    lrz->setOutlierCapacity(outlier_capacity);
    pipeline.setPoolManagedDecompOutput(false);
    pipeline.finalize();

    void*  d_comp = nullptr;
    size_t comp_sz = 0;
    pipeline.compress(d_in.void_ptr(), in_bytes, &d_comp, &comp_sz, stream);
    EXPECT_GT(comp_sz, 0u) << "Compressed output is empty";
    if (comp_sz_out) *comp_sz_out = comp_sz;

    void*  d_dec = nullptr;
    size_t dec_sz = 0;
    pipeline.decompress(nullptr, 0, &d_dec, &dec_sz, stream);

    EXPECT_NE(d_dec, nullptr) << "Decompressed pointer is null";
    EXPECT_EQ(dec_sz, in_bytes)  << "Decompressed size mismatch";

    std::vector<float> h_recon(N, 0.0f);
    if (d_dec && dec_sz == in_bytes)
        cudaMemcpy(h_recon.data(), d_dec, in_bytes, cudaMemcpyDeviceToHost);
    if (d_dec) cudaFree(d_dec);

    return max_abs_error(h_input, h_recon);
}

// ─────────────────────────────────────────────────────────────────────────────
// DP1: All-zero input
//   - max_error should be 0 (zero is predicted perfectly by Lorenzo)
//   - compressed size should be very small
// ─────────────────────────────────────────────────────────────────────────────
TEST(DataPatterns, AllZeros) {
    constexpr size_t N  = 1 << 14;
    constexpr float  EB = 1e-2f;

    size_t comp_sz = 0;
    float max_err = run_pattern(std::vector<float>(N, 0.0f), EB,
                                32768, 0.25f, &comp_sz);

    EXPECT_FLOAT_EQ(max_err, 0.0f) << "All-zero input should reconstruct exactly";
    EXPECT_LT(comp_sz, N * sizeof(float))
        << "All-zero data should compress to less than raw size";
}

// ─────────────────────────────────────────────────────────────────────────────
// DP2: All-same non-zero constant
//   - max_error should be 0 for interior elements (perfect prediction)
// ─────────────────────────────────────────────────────────────────────────────
TEST(DataPatterns, AllSameConstant) {
    constexpr size_t N  = 1 << 14;
    constexpr float  EB = 1e-2f;
    constexpr float  C  = 3.14159f;

    float max_err = run_pattern(std::vector<float>(N, C), EB);
    EXPECT_LE(max_err, EB) << "Constant input should reconstruct within EB";
}

// ─────────────────────────────────────────────────────────────────────────────
// DP3: Linear ramp
//   - Lorenzo predicts adjacent differences; constant slope = zero diff errors
// ─────────────────────────────────────────────────────────────────────────────
TEST(DataPatterns, LinearRamp) {
    constexpr size_t N  = 1 << 14;
    constexpr float  EB = 1e-2f;

    auto h_input = make_ramp<float>(N, 0.001f);  // ramp in [0, ~16]

    float max_err = run_pattern(h_input, EB);
    EXPECT_LE(max_err, EB * 1.01f)
        << "Linear ramp max_err=" << max_err << " exceeds EB=" << EB;
}

// ─────────────────────────────────────────────────────────────────────────────
// DP5: White noise in [-1, 1]
//   - Nearly all values become outliers (large prediction errors)
//   - Round-trip should still satisfy the error bound
// ─────────────────────────────────────────────────────────────────────────────
TEST(DataPatterns, WhiteNoise) {
    constexpr size_t N  = 1 << 14;
    constexpr float  EB = 1e-2f;

    // Use outlier_capacity=1.0f because noise produces ~100% outliers
    float max_err = run_pattern(make_random_floats(N, 42), EB, 32768, 1.0f);
    EXPECT_LE(max_err, EB * 1.01f)
        << "White noise max_err=" << max_err << " exceeds EB=" << EB;
}

// ─────────────────────────────────────────────────────────────────────────────
// DP7: N = 1 (single element) — must not crash
// ─────────────────────────────────────────────────────────────────────────────
TEST(DataPatterns, SingleElement) {
    constexpr float EB = 1e-2f;
    float max_err = run_pattern({42.0f}, EB, 512, 1.0f);
    EXPECT_LE(max_err, EB * 1.01f) << "Single element max_err=" << max_err;
}

// ─────────────────────────────────────────────────────────────────────────────
// DP8: N = 2, 3, 4 boundary cases
// ─────────────────────────────────────────────────────────────────────────────
TEST(DataPatterns, TwoElements) {
    constexpr float EB = 1e-2f;
    float max_err = run_pattern({1.0f, 2.0f}, EB, 512, 1.0f);
    EXPECT_LE(max_err, EB * 1.01f);
}

TEST(DataPatterns, ThreeElements) {
    constexpr float EB = 1e-2f;
    float max_err = run_pattern({1.0f, 2.0f, 3.0f}, EB, 512, 1.0f);
    EXPECT_LE(max_err, EB * 1.01f);
}

TEST(DataPatterns, FourElements) {
    constexpr float EB = 1e-2f;
    float max_err = run_pattern({1.0f, 2.0f, 4.0f, 8.0f}, EB, 512, 1.0f);
    EXPECT_LE(max_err, EB * 1.01f);
}

// ─────────────────────────────────────────────────────────────────────────────
// DP9: Large dataset (4 M floats = 16 MB) — correctness at scale, no OOM
// ─────────────────────────────────────────────────────────────────────────────
TEST(DataPatterns, FourMillionFloats) {
    constexpr size_t N  = 1 << 22;  // 4 M
    constexpr float  EB = 1e-2f;

    auto h_input = make_sine_floats(N, 0.001f, 50.0f);

    float max_err = run_pattern(h_input, EB);
    EXPECT_LE(max_err, EB * 1.01f)
        << "4M-float dataset max_err=" << max_err << " exceeds EB=" << EB;
}

// ─────────────────────────────────────────────────────────────────────────────
// DP10: Very tight error bound (EB = 1e-5)
//   Uses smooth data in a small range so quantization codes stay in uint16_t
//   range at fine precision.
// ─────────────────────────────────────────────────────────────────────────────
TEST(DataPatterns, TightErrorBound) {
    constexpr size_t N  = 1 << 14;
    constexpr float  EB = 1e-5f;

    // Data in [-1, 1]; per-step prediction error ~ 0.05 * amplitude = 0.05
    // Quantization radius 32768 * 1e-5 = 0.33 covers typical errors → codes path
    auto h_input = make_sine_floats(N, 0.05f);  // range [-1, 1]

    // outlier_capacity=0.5f to handle any that escape the quantizer
    float max_err = run_pattern(h_input, EB, 32768, 0.5f);
    EXPECT_LE(max_err, EB * 1.01f)
        << "Tight EB=1e-5 max_err=" << max_err;
}

// ─────────────────────────────────────────────────────────────────────────────
// DP11: Very loose error bound (EB = 10)
//   More values quantize to fewer distinct codes → improved compression ratio.
// ─────────────────────────────────────────────────────────────────────────────
TEST(DataPatterns, LooseErrorBound) {
    constexpr size_t N   = 1 << 14;
    constexpr float  EB  = 10.0f;
    constexpr float  EB_tight = 1e-2f;

    auto h_input = make_sine_floats(N, 0.01f, 50.0f);

    size_t comp_sz_loose = 0, comp_sz_tight = 0;
    float  max_err_loose = run_pattern(h_input, EB, 512, 0.2f, &comp_sz_loose);
    float  max_err_tight = run_pattern(h_input, EB_tight, 512, 0.2f, &comp_sz_tight);

    EXPECT_LE(max_err_loose, EB * 1.01f)
        << "Loose EB max_err=" << max_err_loose;

    // Loose bound should compress at least as well (likely better) than tight
    EXPECT_LE(comp_sz_loose, comp_sz_tight)
        << "Loose EB should not produce a larger output than tight EB";
}

// ─────────────────────────────────────────────────────────────────────────────
// L8: outlier_capacity = 0  (intentional lossy mode — all outliers dropped)
//
// When outlier_capacity is explicitly set to 0 the Lorenzo stage allocates no
// space for outlier (index / error) arrays.  Any element whose quantization
// code overflows the quant_radius is silently dropped: its reconstruction
// error will exceed error_bound but the pipeline must not crash.
//
// Test with white-noise data (high outlier rate) and a very tight error bound
// to guarantee that a non-trivial number of outliers would normally be present.
// ─────────────────────────────────────────────────────────────────────────────
TEST(DataPatterns, ZeroOutlierCapacityNoCrash) {
    CudaStream stream;
    constexpr size_t N        = 4096;
    constexpr float  EB       = 1e-4f;  // tight bound → many outliers in random data
    const size_t     in_bytes = N * sizeof(float);

    // White-noise data has high Shannon entropy → many elements become outliers
    // under a tight error bound.
    auto h_input = make_random_floats(N, 42);

    CudaBuffer<float> d_in(N);
    d_in.upload(h_input, stream);
    stream.sync();

    Pipeline p(in_bytes, MemoryStrategy::MINIMAL);
    auto* lrz = p.addStage<LorenzoStage<float, uint16_t>>();
    lrz->setErrorBound(EB);
    lrz->setOutlierCapacity(0.0f);  // intentionally drop all outliers
    p.setPoolManagedDecompOutput(false);
    p.finalize();

    void*  d_comp  = nullptr;
    size_t comp_sz = 0;
    ASSERT_NO_THROW(
        p.compress(d_in.void_ptr(), in_bytes, &d_comp, &comp_sz, stream))
        << "compress() with outlier_capacity=0 must not throw";
    EXPECT_GT(comp_sz, 0u);

    // Decompression should also complete without crashing even though some
    // elements will have reconstruction error > EB.
    void*  d_dec  = nullptr;
    size_t dec_sz = 0;
    ASSERT_NO_THROW(
        p.decompress(nullptr, comp_sz, &d_dec, &dec_sz, stream))
        << "decompress() with zero-capacity outliers must not throw";
    ASSERT_NE(d_dec, nullptr);
    EXPECT_EQ(dec_sz, in_bytes);

    // We do NOT check max_abs_error here because dropped outliers will violate
    // the error bound — that is the documented behaviour of capacity=0.
    cudaFree(d_dec);
}
