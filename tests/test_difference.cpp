/**
 * tests/test_difference.cpp
 *
 * Unit tests for DifferenceStage  (forward + inverse round-trip).
 *
 * The Difference stage is a lossless transform:
 *   Forward : output[0] = input[0];  output[i] = input[i] - input[i-1]
 *   Inverse : output[0] = input[0];  output[i] = input[i] + output[i-1]
 *
 * Key properties verified here:
 *   1. Forward transform changes values (not a no-op).
 *   2. Forward followed by Inverse reconstructs the original exactly.
 *   3. Output size equals input size (lossless / size-preserving).
 */

#include <gtest/gtest.h>
#include "helpers/fz_test_utils.h"
#include "encoders/diff/diff.h"

#include <cmath>
#include <numeric>
#include <vector>

using namespace fz;
using namespace fz_test;

// ─────────────────────────────────────────────────────────────────────────────
// Helper: allocate output buffer using estimateOutputSizes, run stage, return
//         host copy of the output.
// ─────────────────────────────────────────────────────────────────────────────
template <typename T>
static std::vector<T> run_diff_stage(Stage& stage,
                                     const std::vector<T>& h_input,
                                     cudaStream_t           stream,
                                     fz::MemoryPool&        pool) {
    size_t n = h_input.size();
    size_t in_bytes = n * sizeof(T);

    // Host → device input
    CudaBuffer<T> d_in(n);
    d_in.upload(h_input, stream);

    // Estimate and allocate output
    auto est = stage.estimateOutputSizes({in_bytes});
    EXPECT_FALSE(est.empty());
    size_t out_bytes = est[0];
    size_t out_n = out_bytes / sizeof(T);

    CudaBuffer<T> d_out(out_n);

    // Execute
    std::vector<void*> inputs  = {d_in.void_ptr()};
    std::vector<void*> outputs = {d_out.void_ptr()};
    std::vector<size_t> sizes  = {in_bytes};

    stage.execute(stream, &pool, inputs, outputs, sizes);
    cudaStreamSynchronize(stream);

    // Trim to actual size
    auto actual_map = stage.getActualOutputSizesByName();
    size_t actual_bytes = actual_map.count("output") ? actual_map.at("output") : out_bytes;
    size_t actual_n = actual_bytes / sizeof(T);

    auto h_out = d_out.download(stream);
    h_out.resize(actual_n);
    return h_out;
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: float round-trip
// ─────────────────────────────────────────────────────────────────────────────
TEST(DifferenceStage, FloatRoundTrip) {
    CudaStream stream;
    auto pool = make_test_pool(1024 * sizeof(float));

    std::vector<float> h_input = {1.0f, 3.0f, 6.0f, 10.0f, 15.0f, 21.0f, 28.0f};

    // ─── Forward pass ───
    DifferenceStage<float> fwd_stage;
    auto h_diff = run_diff_stage(fwd_stage, h_input, stream, *pool);

    ASSERT_EQ(h_diff.size(), h_input.size()) << "Forward pass must preserve element count";

    // First element unchanged; rest are differences
    EXPECT_FLOAT_EQ(h_diff[0], h_input[0]);
    for (size_t i = 1; i < h_input.size(); i++) {
        EXPECT_FLOAT_EQ(h_diff[i], h_input[i] - h_input[i - 1])
            << "Mismatch at index " << i;
    }

    // ─── Inverse pass ───
    DifferenceStage<float> inv_stage;
    inv_stage.setInverse(true);
    auto h_reconstructed = run_diff_stage(inv_stage, h_diff, stream, *pool);

    ASSERT_EQ(h_reconstructed.size(), h_input.size());
    for (size_t i = 0; i < h_input.size(); i++) {
        EXPECT_FLOAT_EQ(h_reconstructed[i], h_input[i])
            << "Reconstruction mismatch at index " << i;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: uint16_t round-trip (typical after Lorenzo quantization codes)
// ─────────────────────────────────────────────────────────────────────────────
TEST(DifferenceStage, Uint16RoundTrip) {
    CudaStream stream;
    auto pool = make_test_pool(4096 * sizeof(uint16_t));

    // Build a ramp of 256 uint16_t values
    std::vector<uint16_t> h_input(256);
    for (size_t i = 0; i < h_input.size(); i++)
        h_input[i] = static_cast<uint16_t>(i * 3 + 100);

    DifferenceStage<uint16_t> fwd;
    auto h_diff = run_diff_stage(fwd, h_input, stream, *pool);
    ASSERT_EQ(h_diff.size(), h_input.size());

    DifferenceStage<uint16_t> inv;
    inv.setInverse(true);
    auto h_recon = run_diff_stage(inv, h_diff, stream, *pool);

    ASSERT_EQ(h_recon.size(), h_input.size());
    for (size_t i = 0; i < h_input.size(); i++) {
        EXPECT_EQ(h_recon[i], h_input[i]) << "Mismatch at index " << i;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: constant input → differences are all zero after first element
// ─────────────────────────────────────────────────────────────────────────────
TEST(DifferenceStage, ConstantInput) {
    CudaStream stream;
    auto pool = make_test_pool(512 * sizeof(float));

    std::vector<float> h_input(512, 3.14f);

    DifferenceStage<float> stage;
    auto h_diff = run_diff_stage(stage, h_input, stream, *pool);

    ASSERT_EQ(h_diff.size(), h_input.size());
    EXPECT_FLOAT_EQ(h_diff[0], 3.14f);
    for (size_t i = 1; i < h_diff.size(); i++) {
        EXPECT_FLOAT_EQ(h_diff[i], 0.0f) << "Expected zero diff at index " << i;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: large random-ish data round-trip
// ─────────────────────────────────────────────────────────────────────────────
TEST(DifferenceStage, LargeRoundTrip) {
    CudaStream stream;
    constexpr size_t N = 1 << 16;  // 64 K floats
    auto pool = make_test_pool(N * sizeof(float) * 4);

    std::vector<float> h_input(N);
    for (size_t i = 0; i < N; i++)
        h_input[i] = std::sin(static_cast<float>(i) * 0.01f) * 100.0f;

    DifferenceStage<float> fwd;
    auto h_diff = run_diff_stage(fwd, h_input, stream, *pool);

    DifferenceStage<float> inv;
    inv.setInverse(true);
    auto h_recon = run_diff_stage(inv, h_diff, stream, *pool);

    ASSERT_EQ(h_recon.size(), h_input.size());
    float max_err = 0.0f;
    for (size_t i = 0; i < N; i++) {
        float err = std::abs(h_recon[i] - h_input[i]);
        max_err = std::max(max_err, err);
    }
    // Float32 prefix-sum accumulates ~1 ULP per step.  Over 64 K elements with
    // values up to ~100 the worst-case accumulated error is O(N * eps * max_val)
    // ≈ 64K * 1.2e-7 * 100 ≈ 7.6e-4, so 1e-3 is the appropriate bound.
    EXPECT_LT(max_err, 1e-3f) << "Max reconstruction error too large: " << max_err;
}
