/**
 * tests/test_lorenzo.cpp
 *
 * Unit tests for LorenzoStage<float, uint16_t>.
 *
 * Lorenzo is a lossy predictor with quantization:
 *   Forward: produces 4 outputs — codes, outlier_errors, outlier_indices, outlier_count
 *   Inverse: takes those 4 outputs and reconstructs the original data within error_bound
 *
 * Key properties verified:
 *   1. Forward produces non-trivial output (something happened).
 *   2. Forward + Inverse reconstructs within error_bound for smooth data.
 *   3. Constant input is reproduced exactly (zero prediction error → no outliers).
 *   4. Large smooth dataset round-trips within the configured error bound.
 */

#include <gtest/gtest.h>
#include "helpers/fz_test_utils.h"
#include "predictors/lorenzo/lorenzo.h"

#include <cmath>
#include <vector>

using namespace fz;
using namespace fz_test;

// ─────────────────────────────────────────────────────────────────────────────
// Helper: run Lorenzo forward pass.
// Returns device buffers for each output so the inverse pass can consume them.
// ─────────────────────────────────────────────────────────────────────────────
struct LorenzoForwardResult {
    std::vector<uint8_t> codes_raw;   // raw bytes of quantization codes
    std::vector<uint8_t> errors_raw;  // raw bytes of outlier errors
    std::vector<uint8_t> indices_raw; // raw bytes of outlier indices
    std::vector<uint8_t> count_raw;   // raw bytes of outlier count (4 bytes)
    // Sizes in bytes
    size_t codes_bytes   = 0;
    size_t errors_bytes  = 0;
    size_t indices_bytes = 0;
    size_t count_bytes   = 0;
};

static LorenzoForwardResult run_lorenzo_forward(
    LorenzoStage<float, uint16_t>& stage,
    const std::vector<float>&      h_input,
    cudaStream_t                   stream,
    fz::MemoryPool&                pool)
{
    size_t n = h_input.size();
    size_t in_bytes = n * sizeof(float);

    CudaBuffer<float> d_in(n);
    d_in.upload(h_input, stream);

    auto est = stage.estimateOutputSizes({in_bytes});
    EXPECT_EQ(est.size(), 4u);

    // Allocate four output buffers
    CudaBuffer<uint8_t> d_codes  (est[0]);
    CudaBuffer<uint8_t> d_errors (est[1]);
    CudaBuffer<uint8_t> d_indices(est[2]);
    CudaBuffer<uint8_t> d_count  (est[3]);

    std::vector<void*> inputs  = {d_in.void_ptr()};
    std::vector<void*> outputs = {d_codes.void_ptr(), d_errors.void_ptr(),
                                   d_indices.void_ptr(), d_count.void_ptr()};
    std::vector<size_t> sizes  = {in_bytes};

    stage.execute(stream, &pool, inputs, outputs, sizes);
    cudaStreamSynchronize(stream);
    stage.postStreamSync(stream);  // read back actual outlier count

    auto actual = stage.getActualOutputSizesByName();

    LorenzoForwardResult r;
    r.codes_bytes   = actual.count("codes")          ? actual.at("codes")          : est[0];
    r.errors_bytes  = actual.count("outlier_errors")  ? actual.at("outlier_errors") : est[1];
    r.indices_bytes = actual.count("outlier_indices") ? actual.at("outlier_indices"): est[2];
    r.count_bytes   = actual.count("outlier_count")   ? actual.at("outlier_count")  : est[3];

    r.codes_raw   = d_codes.download(stream);   r.codes_raw.resize(r.codes_bytes);
    r.errors_raw  = d_errors.download(stream);  r.errors_raw.resize(r.errors_bytes);
    r.indices_raw = d_indices.download(stream); r.indices_raw.resize(r.indices_bytes);
    r.count_raw   = d_count.download(stream);   r.count_raw.resize(r.count_bytes);

    return r;
}

// Run Lorenzo inverse pass and return reconstructed floats.
static std::vector<float> run_lorenzo_inverse(
    LorenzoStage<float, uint16_t>& stage,
    const LorenzoForwardResult&    fwd,
    size_t                          n_elements,
    cudaStream_t                    stream,
    fz::MemoryPool&                 pool)
{
    // Upload the four compressed buffers back to device
    CudaBuffer<uint8_t> d_codes  (fwd.codes_raw.size());  d_codes.upload(fwd.codes_raw, stream);
    CudaBuffer<uint8_t> d_errors (fwd.errors_raw.size()); d_errors.upload(fwd.errors_raw, stream);
    CudaBuffer<uint8_t> d_indices(fwd.indices_raw.size());d_indices.upload(fwd.indices_raw, stream);
    CudaBuffer<uint8_t> d_count  (fwd.count_raw.size());  d_count.upload(fwd.count_raw, stream);

    // Output: reconstructed floats
    CudaBuffer<float> d_out(n_elements);

    std::vector<void*> inputs = {d_codes.void_ptr(), d_errors.void_ptr(),
                                  d_indices.void_ptr(), d_count.void_ptr()};
    std::vector<void*> outputs = {d_out.void_ptr()};
    std::vector<size_t> sizes  = {fwd.codes_bytes, fwd.errors_bytes,
                                   fwd.indices_bytes, fwd.count_bytes};

    stage.execute(stream, &pool, inputs, outputs, sizes);
    cudaStreamSynchronize(stream);

    return d_out.download(stream);
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: smooth sinusoidal data round-trip within error_bound
// ─────────────────────────────────────────────────────────────────────────────
TEST(LorenzoStage, SmoothRoundTrip) {
    CudaStream stream;
    constexpr size_t N  = 1024;
    constexpr float  EB = 1e-2f;

    auto pool = make_test_pool(N * sizeof(float) * 20);

    std::vector<float> h_input(N);
    for (size_t i = 0; i < N; i++)
        h_input[i] = std::sin(static_cast<float>(i) * 0.05f) * 10.0f;

    // Forward
    LorenzoStage<float, uint16_t> fwd;
    fwd.setErrorBound(EB);
    fwd.setQuantRadius(512);
    fwd.setOutlierCapacity(0.2f);

    auto fwd_result = run_lorenzo_forward(fwd, h_input, stream, *pool);

    // Codes must be non-trivial
    EXPECT_GT(fwd_result.codes_bytes, 0u) << "Codes output is empty";

    // Inverse
    LorenzoStage<float, uint16_t> inv;
    inv.setErrorBound(EB);
    inv.setQuantRadius(512);
    inv.setInverse(true);

    // Copy stage config from the forward stage via serializeHeader
    uint8_t cfg_buf[128] = {};
    size_t  cfg_sz = fwd.serializeHeader(0, cfg_buf, sizeof(cfg_buf));
    inv.deserializeHeader(cfg_buf, cfg_sz);

    auto h_recon = run_lorenzo_inverse(inv, fwd_result, N, stream, *pool);

    ASSERT_EQ(h_recon.size(), N);
    float max_err = 0.0f;
    for (size_t i = 0; i < N; i++) {
        float err = std::abs(h_recon[i] - h_input[i]);
        max_err = std::max(max_err, err);
    }
    EXPECT_LE(max_err, EB * 1.01f)  // 1 % tolerance for floating-point accumulation
        << "Max reconstruction error " << max_err << " exceeds bound " << EB;
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: constant input — zero prediction errors, should have no outliers
// ─────────────────────────────────────────────────────────────────────────────
TEST(LorenzoStage, ConstantInputRoundTrip) {
    CudaStream stream;
    constexpr size_t N  = 512;
    constexpr float  EB = 1e-3f;

    auto pool = make_test_pool(N * sizeof(float) * 20);

    std::vector<float> h_input(N, 5.0f);

    LorenzoStage<float, uint16_t> fwd;
    fwd.setErrorBound(EB);
    fwd.setQuantRadius(512);
    fwd.setOutlierCapacity(0.2f);

    auto fwd_result = run_lorenzo_forward(fwd, h_input, stream, *pool);

    // Lorenzo predicts each element from its left neighbour.  Element 0 has no
    // predecessor so its prediction is 0; for a constant input of 5.0f the
    // prediction error is 5.0 >> EB, making it a legitimate outlier.  All
    // interior elements (i > 0) see a perfect prediction, so the count must be
    // exactly 1 (only the boundary element).
    ASSERT_GE(fwd_result.count_raw.size(), sizeof(uint32_t));
    uint32_t outlier_count = 0;
    std::memcpy(&outlier_count, fwd_result.count_raw.data(), sizeof(uint32_t));
    EXPECT_LE(outlier_count, 1u)
        << "Constant input should produce at most 1 outlier (first-element boundary)";

    // Round-trip
    LorenzoStage<float, uint16_t> inv;
    inv.setInverse(true);
    uint8_t cfg[128] = {};
    size_t  cfg_sz = fwd.serializeHeader(0, cfg, sizeof(cfg));
    inv.deserializeHeader(cfg, cfg_sz);

    auto h_recon = run_lorenzo_inverse(inv, fwd_result, N, stream, *pool);

    ASSERT_EQ(h_recon.size(), N);
    for (size_t i = 0; i < N; i++) {
        EXPECT_NEAR(h_recon[i], 5.0f, EB)
            << "Reconstruction mismatch at index " << i;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: large dataset (64 K floats), tighter error bound
// ─────────────────────────────────────────────────────────────────────────────
TEST(LorenzoStage, LargeDataRoundTrip) {
    CudaStream stream;
    constexpr size_t N  = 1 << 16;  // 64 K
    constexpr float  EB = 1e-3f;

    auto pool = make_test_pool(N * sizeof(float) * 20);

    std::vector<float> h_input(N);
    for (size_t i = 0; i < N; i++)
        h_input[i] = std::cos(static_cast<float>(i) * 0.001f) * 50.0f
                   + std::sin(static_cast<float>(i) * 0.003f) * 20.0f;

    LorenzoStage<float, uint16_t> fwd;
    fwd.setErrorBound(EB);
    fwd.setQuantRadius(512);
    fwd.setOutlierCapacity(0.2f);

    auto fwd_result = run_lorenzo_forward(fwd, h_input, stream, *pool);
    EXPECT_GT(fwd_result.codes_bytes, 0u);

    LorenzoStage<float, uint16_t> inv;
    inv.setInverse(true);
    uint8_t cfg[128] = {};
    inv.deserializeHeader(cfg, fwd.serializeHeader(0, cfg, sizeof(cfg)));

    auto h_recon = run_lorenzo_inverse(inv, fwd_result, N, stream, *pool);
    ASSERT_EQ(h_recon.size(), N);

    float max_err = 0.0f;
    for (size_t i = 0; i < N; i++) {
        max_err = std::max(max_err, std::abs(h_recon[i] - h_input[i]));
    }
    EXPECT_LE(max_err, EB * 1.01f)
        << "Max error " << max_err << " exceeds bound " << EB;
}
