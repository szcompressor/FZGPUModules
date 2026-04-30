/**
 * tests/test_quantizer.cpp
 *
 * Unit tests for QuantizerStage<float, uint16_t> (ABS and NOA modes)
 * and QuantizerStage<float, uint32_t> (REL mode).
 *
 * QuantizerStage quantises input *values* directly (unlike LorenzoQuantStage which
 * quantises prediction residuals).  Values that would exceed the code range
 * are stored losslessly as outliers.
 *
 *   ABS  — uniform step quantisation:  |x - x_hat| <= eb
 *   NOA  — norm-of-absolute:           abs_eb  = user_eb * (max - min)
 *   REL  — pointwise relative:         |x - x_hat| / |x| <= eb
 *
 * Forward outputs (index → name):
 *   [0] codes          — quantisation codes  (TCode[n])
 *   [1] outlier_vals   — original values at outlier positions (TInput[k])
 *   [2] outlier_idxs   — positions of outliers (uint32_t[k])
 *   [3] outlier_count  — number of outliers    (uint32_t scalar)
 */

#include <gtest/gtest.h>
#include "helpers/fz_test_utils.h"
#include "quantizers/quantizer/quantizer.h"
#include "fused/lorenzo_quant/lorenzo_quant.h"   // for ErrorBoundMode
#include "fzgpumodules.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

using namespace fz;
using namespace fz_test;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

struct QuantizerForwardResult {
    std::vector<uint8_t> codes_raw;   // raw bytes of quantisation codes
    std::vector<uint8_t> vals_raw;    // raw bytes of outlier values
    std::vector<uint8_t> idxs_raw;    // raw bytes of outlier indices
    std::vector<uint8_t> count_raw;   // raw bytes of outlier count (4 bytes)
    // Actual sizes in bytes
    size_t codes_bytes = 0;
    size_t vals_bytes  = 0;
    size_t idxs_bytes  = 0;
    size_t count_bytes = 0;
};

// Run a Quantizer forward pass.  Works for both uint16_t and uint32_t code types.
template<typename TCode>
static QuantizerForwardResult run_quantizer_forward(
    QuantizerStage<float, TCode>& stage,
    const std::vector<float>&     h_input,
    cudaStream_t                  stream,
    fz::MemoryPool&               pool)
{
    size_t n = h_input.size();
    size_t in_bytes = n * sizeof(float);

    CudaBuffer<float> d_in(n);
    d_in.upload(h_input, stream);

    auto est = stage.estimateOutputSizes({in_bytes});
    EXPECT_EQ(est.size(), 4u);

    CudaBuffer<uint8_t> d_codes(est[0]);
    CudaBuffer<uint8_t> d_vals (est[1]);
    CudaBuffer<uint8_t> d_idxs (est[2]);
    CudaBuffer<uint8_t> d_count(est[3]);

    std::vector<void*>  inputs  = {d_in.void_ptr()};
    std::vector<void*>  outputs = {d_codes.void_ptr(), d_vals.void_ptr(),
                                    d_idxs.void_ptr(), d_count.void_ptr()};
    std::vector<size_t> sizes   = {in_bytes};

    stage.execute(stream, &pool, inputs, outputs, sizes);
    cudaStreamSynchronize(stream);
    stage.postStreamSync(stream);

    auto actual = stage.getActualOutputSizesByName();

    QuantizerForwardResult r;
    r.codes_bytes = actual.count("codes")         ? actual.at("codes")         : est[0];
    r.vals_bytes  = actual.count("outlier_vals")  ? actual.at("outlier_vals")  : est[1];
    r.idxs_bytes  = actual.count("outlier_idxs")  ? actual.at("outlier_idxs")  : est[2];
    r.count_bytes = actual.count("outlier_count") ? actual.at("outlier_count") : est[3];

    r.codes_raw = d_codes.download_bytes(r.codes_bytes, stream);
    r.vals_raw  = d_vals.download_bytes(r.vals_bytes,  stream);
    r.idxs_raw  = d_idxs.download_bytes(r.idxs_bytes,  stream);
    r.count_raw = d_count.download_bytes(r.count_bytes, stream);

    return r;
}

// Run a Quantizer inverse pass; returns reconstructed floats.
template<typename TCode>
static std::vector<float> run_quantizer_inverse(
    QuantizerStage<float, TCode>& stage,
    const QuantizerForwardResult& fwd,
    size_t                         n_elements,
    cudaStream_t                   stream,
    fz::MemoryPool&                pool)
{
    CudaBuffer<uint8_t> d_codes(fwd.codes_raw.size()); d_codes.upload(fwd.codes_raw, stream);
    CudaBuffer<uint8_t> d_vals (fwd.vals_raw.size());  d_vals.upload(fwd.vals_raw,  stream);
    CudaBuffer<uint8_t> d_idxs (fwd.idxs_raw.size());  d_idxs.upload(fwd.idxs_raw,  stream);
    CudaBuffer<uint8_t> d_count(fwd.count_raw.size()); d_count.upload(fwd.count_raw, stream);

    CudaBuffer<float> d_out(n_elements);

    std::vector<void*>  inputs  = {d_codes.void_ptr(), d_vals.void_ptr(),
                                    d_idxs.void_ptr(), d_count.void_ptr()};
    std::vector<void*>  outputs = {d_out.void_ptr()};
    std::vector<size_t> sizes   = {fwd.codes_bytes, fwd.vals_bytes,
                                    fwd.idxs_bytes,  fwd.count_bytes};

    stage.execute(stream, &pool, inputs, outputs, sizes);
    cudaStreamSynchronize(stream);

    return d_out.download(stream);
}

// ─────────────────────────────────────────────────────────────────────────────
// ABS mode
// ─────────────────────────────────────────────────────────────────────────────

TEST(QuantizerABS, SmoothRoundTrip) {
    CudaStream stream;
    constexpr size_t N  = 1024;
    constexpr float  EB = 1e-2f;

    auto pool = make_test_pool(N * sizeof(float) * 20);

    auto h_input = make_sine_floats(N, 0.05f, 10.0f);

    // quant_radius must cover the full data range: radius > amplitude / (2*EB) = 10/0.02 = 500
    // Use 32768 (the stage default) which covers |x| up to 655 with EB=1e-2.
    QuantizerStage<float, uint16_t> fwd;
    fwd.setErrorBound(EB);
    fwd.setQuantRadius(32768);
    fwd.setOutlierCapacity(0.1f);

    auto fwd_result = run_quantizer_forward(fwd, h_input, stream, *pool);
    EXPECT_GT(fwd_result.codes_bytes, 0u) << "Forward produced no codes";

    QuantizerStage<float, uint16_t> inv;
    inv.setInverse(true);
    uint8_t cfg[128] = {};
    inv.deserializeHeader(cfg, fwd.serializeHeader(0, cfg, sizeof(cfg)));

    auto h_recon = run_quantizer_inverse(inv, fwd_result, N, stream, *pool);

    ASSERT_EQ(h_recon.size(), N);
    float max_err = max_abs_error(h_input, h_recon);
    EXPECT_LE(max_err, EB * 1.01f)
        << "ABS smooth round-trip max_err=" << max_err << " exceeds bound " << EB;
}

TEST(QuantizerABS, ConstantInput) {
    // Unlike Lorenzo, the quantizer has no boundary effect: a constant input
    // should produce zero outliers and perfect reconstruction.
    CudaStream stream;
    constexpr size_t N   = 512;
    constexpr float  EB  = 1e-3f;
    constexpr float  VAL = 5.0f;

    auto pool = make_test_pool(N * sizeof(float) * 20);

    std::vector<float> h_input(N, VAL);

    // quant_radius must cover VAL: radius > VAL / (2*EB) = 5 / 0.002 = 2500. Use 32768.
    QuantizerStage<float, uint16_t> fwd;
    fwd.setErrorBound(EB);
    fwd.setQuantRadius(32768);
    fwd.setOutlierCapacity(0.1f);

    auto fwd_result = run_quantizer_forward(fwd, h_input, stream, *pool);

    ASSERT_GE(fwd_result.count_raw.size(), sizeof(uint32_t));
    uint32_t outlier_count = 0;
    std::memcpy(&outlier_count, fwd_result.count_raw.data(), sizeof(uint32_t));
    EXPECT_EQ(outlier_count, 0u)
        << "Constant input should produce zero outliers (no boundary issue)";

    QuantizerStage<float, uint16_t> inv;
    inv.setInverse(true);
    uint8_t cfg[128] = {};
    inv.deserializeHeader(cfg, fwd.serializeHeader(0, cfg, sizeof(cfg)));

    auto h_recon = run_quantizer_inverse(inv, fwd_result, N, stream, *pool);

    ASSERT_EQ(h_recon.size(), N);
    for (size_t i = 0; i < N; i++)
        EXPECT_NEAR(h_recon[i], VAL, EB) << "Reconstruction mismatch at index " << i;
}

TEST(QuantizerABS, LargeDataRoundTrip) {
    CudaStream stream;
    constexpr size_t N  = 1 << 16;  // 64 K
    constexpr float  EB = 1e-3f;

    auto pool = make_test_pool(N * sizeof(float) * 20);

    std::vector<float> h_input(N);
    for (size_t i = 0; i < N; i++)
        h_input[i] = std::cos(static_cast<float>(i) * 0.001f) * 50.0f
                   + std::sin(static_cast<float>(i) * 0.003f) * 20.0f;

    // Amplitude ~70 (±50 + ±20).  With EB=1e-3, max_representable = 32767*2e-3 = 65.5.
    // Peaks above 65.5 become outliers (reconstructed exactly), covered by 10% capacity.
    QuantizerStage<float, uint16_t> fwd;
    fwd.setErrorBound(EB);
    fwd.setQuantRadius(32768);
    fwd.setOutlierCapacity(0.1f);

    auto fwd_result = run_quantizer_forward(fwd, h_input, stream, *pool);
    EXPECT_GT(fwd_result.codes_bytes, 0u);

    QuantizerStage<float, uint16_t> inv;
    inv.setInverse(true);
    uint8_t cfg[128] = {};
    inv.deserializeHeader(cfg, fwd.serializeHeader(0, cfg, sizeof(cfg)));

    auto h_recon = run_quantizer_inverse(inv, fwd_result, N, stream, *pool);

    ASSERT_EQ(h_recon.size(), N);
    float max_err = max_abs_error(h_input, h_recon);
    EXPECT_LE(max_err, EB * 1.01f)
        << "Large data ABS max_err=" << max_err << " exceeds bound " << EB;
}

// ─────────────────────────────────────────────────────────────────────────────
// NOA mode:  abs_eb = user_eb * (max(data) - min(data))
// ─────────────────────────────────────────────────────────────────────────────

TEST(QuantizerNOA, SmoothRoundTrip) {
    CudaStream stream;
    constexpr size_t N        = 1024;
    constexpr float  USER_EB  = 0.01f;   // 1 % of value range

    auto pool = make_test_pool(N * sizeof(float) * 20);

    auto h_input = make_sine_floats(N, 0.05f, 10.0f);

    // Compute the expected absolute bound on the host for verification.
    float vmin = *std::min_element(h_input.begin(), h_input.end());
    float vmax = *std::max_element(h_input.begin(), h_input.end());
    float expected_abs_eb = USER_EB * (vmax - vmin);

    QuantizerStage<float, uint16_t> fwd;
    fwd.setErrorBound(USER_EB);
    fwd.setErrorBoundMode(ErrorBoundMode::NOA);
    fwd.setQuantRadius(512);
    fwd.setOutlierCapacity(0.1f);

    auto fwd_result = run_quantizer_forward(fwd, h_input, stream, *pool);
    EXPECT_GT(fwd_result.codes_bytes, 0u) << "Forward produced no codes";

    QuantizerStage<float, uint16_t> inv;
    inv.setInverse(true);
    uint8_t cfg[128] = {};
    inv.deserializeHeader(cfg, fwd.serializeHeader(0, cfg, sizeof(cfg)));

    auto h_recon = run_quantizer_inverse(inv, fwd_result, N, stream, *pool);

    ASSERT_EQ(h_recon.size(), N);
    float max_err = max_abs_error(h_input, h_recon);
    // 2 % tolerance: the stage computes vmax-vmin from the GPU data which may
    // differ by at most one ULP from our host reference.
    EXPECT_LE(max_err, expected_abs_eb * 1.02f)
        << "NOA smooth round-trip max_err=" << max_err
        << " exceeds expected_abs_eb=" << expected_abs_eb;
}

TEST(QuantizerNOA, ConstantInput) {
    // value_range for constant data = 0, so abs_eb = 0, which means every
    // element is an inlier (code 0 = value exactly).  After round-trip the
    // reconstruction should be perfect (all outliers or all losslessly coded).
    CudaStream stream;
    constexpr size_t N   = 256;
    constexpr float  EB  = 0.01f;
    constexpr float  VAL = 3.14f;

    auto pool = make_test_pool(N * sizeof(float) * 20);

    std::vector<float> h_input(N, VAL);

    QuantizerStage<float, uint16_t> fwd;
    fwd.setErrorBound(EB);
    fwd.setErrorBoundMode(ErrorBoundMode::NOA);
    fwd.setQuantRadius(512);
    fwd.setOutlierCapacity(1.0f);  // generous capacity — all outliers acceptable

    auto fwd_result = run_quantizer_forward(fwd, h_input, stream, *pool);

    QuantizerStage<float, uint16_t> inv;
    inv.setInverse(true);
    uint8_t cfg[128] = {};
    inv.deserializeHeader(cfg, fwd.serializeHeader(0, cfg, sizeof(cfg)));

    auto h_recon = run_quantizer_inverse(inv, fwd_result, N, stream, *pool);

    ASSERT_EQ(h_recon.size(), N);
    for (size_t i = 0; i < N; i++)
        EXPECT_FLOAT_EQ(h_recon[i], VAL)
            << "NOA constant input mismatch at index " << i;
}

// ─────────────────────────────────────────────────────────────────────────────
// REL mode:  |x - x_hat| / |x| <= eb  (log2-space quantisation, PFPL exact)
// REL packs |log_bin| into extended-range codes, so we use uint32_t.
// ─────────────────────────────────────────────────────────────────────────────

TEST(QuantizerREL, SmoothRoundTrip) {
    CudaStream stream;
    constexpr size_t N   = 1024;
    constexpr float  EPS = 0.01f;   // 1 % relative error

    auto pool = make_test_pool(N * sizeof(float) * 20);

    // Use |sin| + 0.1 to avoid zeros/denormals completely.
    std::vector<float> h_input(N);
    for (size_t i = 0; i < N; i++)
        h_input[i] = (std::abs(std::sin(static_cast<float>(i) * 0.05f)) + 0.1f) * 10.0f;

    QuantizerStage<float, uint32_t> fwd;
    fwd.setErrorBound(EPS);
    fwd.setErrorBoundMode(ErrorBoundMode::REL);
    fwd.setQuantRadius(512);
    fwd.setOutlierCapacity(0.1f);

    auto fwd_result = run_quantizer_forward(fwd, h_input, stream, *pool);
    EXPECT_GT(fwd_result.codes_bytes, 0u) << "REL forward produced no codes";

    QuantizerStage<float, uint32_t> inv;
    inv.setInverse(true);
    uint8_t cfg[128] = {};
    inv.deserializeHeader(cfg, fwd.serializeHeader(0, cfg, sizeof(cfg)));

    auto h_recon = run_quantizer_inverse(inv, fwd_result, N, stream, *pool);

    ASSERT_EQ(h_recon.size(), N);
    uint32_t outlier_count = 0;
    if (fwd_result.count_raw.size() >= sizeof(uint32_t))
        std::memcpy(&outlier_count, fwd_result.count_raw.data(), sizeof(uint32_t));

    // Build a set of outlier indices for exact-match checking.
    std::vector<uint32_t> outlier_idxs;
    if (fwd_result.idxs_raw.size() >= outlier_count * sizeof(uint32_t)) {
        outlier_idxs.resize(outlier_count);
        std::memcpy(outlier_idxs.data(), fwd_result.idxs_raw.data(),
                    outlier_count * sizeof(uint32_t));
    }
    std::sort(outlier_idxs.begin(), outlier_idxs.end());

    // Non-outlier elements: verify relative error bound.
    // Outlier elements: verify exact reconstruction.
    float max_rel_err = 0.0f;
    for (size_t i = 0; i < N; i++) {
        bool is_outlier = std::binary_search(outlier_idxs.begin(), outlier_idxs.end(),
                                              static_cast<uint32_t>(i));
        if (is_outlier) {
            // Outliers are stored losslessly; allow 0-ULP tolerance.
            EXPECT_FLOAT_EQ(h_recon[i], h_input[i])
                << "REL outlier not exactly reconstructed at index " << i;
        } else {
            float orig_abs = std::abs(h_input[i]);
            if (orig_abs > 0.0f) {
                float rel = std::abs(h_recon[i] - h_input[i]) / orig_abs;
                max_rel_err = std::max(max_rel_err, rel);
            }
        }
    }
    // 1.5 % tolerance above eps to account for the fast log2/pow2 approximation.
    EXPECT_LE(max_rel_err, EPS * 1.5f)
        << "REL round-trip max relative error=" << max_rel_err
        << " exceeds eps=" << EPS;
}

TEST(QuantizerREL, ZerosGoToOutliers) {
    // Zeros, denormals and infinities cannot be represented in log2-space.
    // They must be sent to the outlier list and reconstructed exactly.
    CudaStream stream;
    constexpr size_t N   = 256;
    constexpr float  EPS = 0.01f;

    auto pool = make_test_pool(N * sizeof(float) * 20);

    // Mix: half normal positive values, quarter zeros, quarter negatives.
    std::vector<float> h_input(N);
    for (size_t i = 0; i < N; i++) {
        if (i % 4 == 0)      h_input[i] = 0.0f;
        else if (i % 4 == 1) h_input[i] = -std::sin(static_cast<float>(i) * 0.1f + 0.5f) * 5.0f;
        else                 h_input[i] =  std::sin(static_cast<float>(i) * 0.1f + 0.5f) * 5.0f;
    }

    QuantizerStage<float, uint32_t> fwd;
    fwd.setErrorBound(EPS);
    fwd.setErrorBoundMode(ErrorBoundMode::REL);
    fwd.setQuantRadius(512);
    fwd.setOutlierCapacity(0.5f);

    auto fwd_result = run_quantizer_forward(fwd, h_input, stream, *pool);

    ASSERT_GE(fwd_result.count_raw.size(), sizeof(uint32_t));
    uint32_t outlier_count = 0;
    std::memcpy(&outlier_count, fwd_result.count_raw.data(), sizeof(uint32_t));

    // The N/4 zeros must all appear as outliers.
    EXPECT_GE(outlier_count, N / 4)
        << "REL: not all zeros went to outlier list";

    // Full round-trip.
    QuantizerStage<float, uint32_t> inv;
    inv.setInverse(true);
    uint8_t cfg[128] = {};
    inv.deserializeHeader(cfg, fwd.serializeHeader(0, cfg, sizeof(cfg)));

    auto h_recon = run_quantizer_inverse(inv, fwd_result, N, stream, *pool);

    ASSERT_EQ(h_recon.size(), N);
    // Every zero must be reconstructed exactly.
    for (size_t i = 0; i < N; i += 4)
        EXPECT_FLOAT_EQ(h_recon[i], 0.0f)
            << "REL: zero not reconstructed exactly at index " << i;
}

// ─────────────────────────────────────────────────────────────────────────────
// Header serialisation / deserialisation
// ─────────────────────────────────────────────────────────────────────────────

TEST(QuantizerABS, SerializeDeserialize) {
    QuantizerStage<float, uint16_t> src;
    src.setErrorBound(5e-3f);
    src.setQuantRadius(1024);
    src.setOutlierCapacity(0.05f);

    uint8_t cfg[128] = {};
    size_t sz = src.serializeHeader(0, cfg, sizeof(cfg));
    EXPECT_EQ(sz, sizeof(QuantizerConfig));

    QuantizerStage<float, uint16_t> dst;
    dst.setInverse(true);
    dst.deserializeHeader(cfg, sz);

    EXPECT_FLOAT_EQ(dst.getErrorBound(), 5e-3f);
    EXPECT_EQ(dst.getQuantRadius(), 1024);
    EXPECT_EQ(dst.getErrorBoundMode(), ErrorBoundMode::ABS);
}

TEST(QuantizerNOA, SerializeDeserialize) {
    QuantizerStage<float, uint16_t> src;
    src.setErrorBound(0.02f);
    src.setErrorBoundMode(ErrorBoundMode::NOA);
    src.setQuantRadius(2048);

    uint8_t cfg[128] = {};
    size_t sz = src.serializeHeader(0, cfg, sizeof(cfg));
    EXPECT_EQ(sz, sizeof(QuantizerConfig));

    QuantizerStage<float, uint16_t> dst;
    dst.setInverse(true);
    dst.deserializeHeader(cfg, sz);

    EXPECT_FLOAT_EQ(dst.getErrorBound(), 0.02f);
    EXPECT_EQ(dst.getQuantRadius(), 2048);
    EXPECT_EQ(dst.getErrorBoundMode(), ErrorBoundMode::NOA);
}

TEST(QuantizerREL, SerializeDeserialize) {
    QuantizerStage<float, uint32_t> src;
    src.setErrorBound(0.01f);
    src.setErrorBoundMode(ErrorBoundMode::REL);
    src.setQuantRadius(512);

    uint8_t cfg[128] = {};
    size_t sz = src.serializeHeader(0, cfg, sizeof(cfg));

    QuantizerStage<float, uint32_t> dst;
    dst.setInverse(true);
    dst.deserializeHeader(cfg, sz);

    EXPECT_FLOAT_EQ(dst.getErrorBound(), 0.01f);
    EXPECT_EQ(dst.getErrorBoundMode(), ErrorBoundMode::REL);
}

// ─────────────────────────────────────────────────────────────────────────────
// ZigzagCodes tests
// These verify the setZigzagCodes(true) path of QuantizerStage:
//   1. ABS mode round-trip reconstructs within error bound with zigzag on.
//   2. NOA mode round-trip works with zigzag on.
//   3. REL mode is unaffected by zigzag (log-space codes are not zigzag-encoded).
//   4. The zigzag_codes flag is preserved through serializeHeader/deserializeHeader.
// ─────────────────────────────────────────────────────────────────────────────

TEST(QuantizerZigzag, ABSRoundTrip) {
    CudaStream stream;
    constexpr size_t N  = 1024;
    constexpr float  EB = 1e-2f;

    auto pool = make_test_pool(N * sizeof(float) * 20);

    auto h_input = make_sine_floats(N, 0.05f, 10.0f);

    QuantizerStage<float, uint16_t> fwd;
    fwd.setErrorBound(EB);
    fwd.setQuantRadius(32768);
    fwd.setOutlierCapacity(0.1f);
    fwd.setZigzagCodes(true);
    EXPECT_TRUE(fwd.getZigzagCodes());

    auto fwd_result = run_quantizer_forward(fwd, h_input, stream, *pool);
    EXPECT_GT(fwd_result.codes_bytes, 0u);

    // Inverse — restore zigzag flag through header
    QuantizerStage<float, uint16_t> inv;
    inv.setInverse(true);
    uint8_t cfg[128] = {};
    size_t cfg_sz = fwd.serializeHeader(0, cfg, sizeof(cfg));
    inv.deserializeHeader(cfg, cfg_sz);
    EXPECT_TRUE(inv.getZigzagCodes())
        << "zigzag_codes flag must survive serializeHeader/deserializeHeader";

    auto h_recon = run_quantizer_inverse(inv, fwd_result, N, stream, *pool);

    ASSERT_EQ(h_recon.size(), N);
    float max_err = max_abs_error(h_input, h_recon);
    EXPECT_LE(max_err, EB * 1.01f)
        << "ZigzagCodes ABS round-trip max_err=" << max_err << " exceeds bound " << EB;
}

TEST(QuantizerZigzag, NOARoundTrip) {
    CudaStream stream;
    constexpr size_t N        = 1024;
    constexpr float  USER_EB  = 0.01f;

    auto pool = make_test_pool(N * sizeof(float) * 20);

    auto h_input = make_sine_floats(N, 0.04f, 8.0f);

    float vmin = *std::min_element(h_input.begin(), h_input.end());
    float vmax = *std::max_element(h_input.begin(), h_input.end());
    float expected_abs_eb = USER_EB * (vmax - vmin);

    QuantizerStage<float, uint16_t> fwd;
    fwd.setErrorBound(USER_EB);
    fwd.setErrorBoundMode(ErrorBoundMode::NOA);
    fwd.setQuantRadius(32768);
    fwd.setOutlierCapacity(0.1f);
    fwd.setZigzagCodes(true);

    auto fwd_result = run_quantizer_forward(fwd, h_input, stream, *pool);
    EXPECT_GT(fwd_result.codes_bytes, 0u);

    QuantizerStage<float, uint16_t> inv;
    inv.setInverse(true);
    uint8_t cfg[128] = {};
    inv.deserializeHeader(cfg, fwd.serializeHeader(0, cfg, sizeof(cfg)));

    auto h_recon = run_quantizer_inverse(inv, fwd_result, N, stream, *pool);

    ASSERT_EQ(h_recon.size(), N);
    float max_err = max_abs_error(h_input, h_recon);
    EXPECT_LE(max_err, expected_abs_eb * 1.02f)
        << "ZigzagCodes NOA round-trip max_err=" << max_err
        << " exceeds expected_abs_eb=" << expected_abs_eb;
}

TEST(QuantizerZigzag, RELModeUnaffected) {
    // In REL mode the codes are log-space packed (always unsigned), so the zigzag
    // flag should have no effect.  Both zigzag=true and zigzag=false must produce
    // identical reconstructions when the mode is REL.
    CudaStream stream;
    constexpr size_t N   = 512;
    constexpr float  EPS = 0.01f;

    auto pool = make_test_pool(N * sizeof(float) * 20);

    // Use values well away from zero so none become REL special-case outliers.
    std::vector<float> h_input(N);
    for (size_t i = 0; i < N; i++)
        h_input[i] = (std::abs(std::sin(static_cast<float>(i) * 0.06f)) + 1.0f) * 5.0f;

    auto run_rel = [&](bool use_zigzag) {
        QuantizerStage<float, uint32_t> fwd;
        fwd.setErrorBound(EPS);
        fwd.setErrorBoundMode(ErrorBoundMode::REL);
        fwd.setQuantRadius(512);
        fwd.setOutlierCapacity(0.1f);
        fwd.setZigzagCodes(use_zigzag);

        auto fwd_result = run_quantizer_forward(fwd, h_input, stream, *pool);

        QuantizerStage<float, uint32_t> inv;
        inv.setInverse(true);
        uint8_t cfg[128] = {};
        inv.deserializeHeader(cfg, fwd.serializeHeader(0, cfg, sizeof(cfg)));

        return run_quantizer_inverse(inv, fwd_result, N, stream, *pool);
    };

    auto h_no_zz   = run_rel(false);
    auto h_with_zz = run_rel(true);

    ASSERT_EQ(h_no_zz.size(), N);
    ASSERT_EQ(h_with_zz.size(), N);

    // The zigzag flag does not affect REL codes, so both reconstructions should
    // be identical element-by-element.
    float max_diff = 0.0f;
    for (size_t i = 0; i < N; i++)
        max_diff = std::max(max_diff, std::abs(h_no_zz[i] - h_with_zz[i]));
    EXPECT_EQ(max_diff, 0.0f)
        << "REL mode: zigzag and non-zigzag should produce identical reconstructions; "
        << "max_diff=" << max_diff;
}

TEST(QuantizerZigzag, HeaderSerialization) {
    // zigzag_codes=true and zigzag_codes=false must each survive a round-trip
    // through serializeHeader/deserializeHeader without cross-contamination.
    QuantizerStage<float, uint16_t> src_on;
    src_on.setErrorBound(1e-3f);
    src_on.setQuantRadius(32768);
    src_on.setZigzagCodes(true);

    QuantizerStage<float, uint16_t> src_off;
    src_off.setErrorBound(1e-3f);
    src_off.setQuantRadius(32768);
    src_off.setZigzagCodes(false);

    uint8_t buf_on[128]  = {};
    uint8_t buf_off[128] = {};
    size_t sz_on  = src_on.serializeHeader(0, buf_on,  sizeof(buf_on));
    size_t sz_off = src_off.serializeHeader(0, buf_off, sizeof(buf_off));

    EXPECT_EQ(sz_on,  sizeof(QuantizerConfig));
    EXPECT_EQ(sz_off, sizeof(QuantizerConfig));

    QuantizerStage<float, uint16_t> dst_on, dst_off;
    dst_on.setInverse(true);
    dst_off.setInverse(true);
    dst_on.deserializeHeader(buf_on,   sz_on);
    dst_off.deserializeHeader(buf_off, sz_off);

    EXPECT_TRUE(dst_on.getZigzagCodes())
        << "zigzag_codes=true was not recovered after Quantizer serialization";
    EXPECT_FALSE(dst_off.getZigzagCodes())
        << "zigzag_codes=false was not recovered after Quantizer serialization";
}

// ─────────────────────────────────────────────────────────────────────────────
//  QuantizerTypeMatrix — exercises every (TInput, TCode) instantiation
//
//  The existing suite only tests QuantizerStage<float, uint32_t> (ABS/NOA/REL).
//  These tests cover:
//    <float, uint16_t>  — half-width codes (much more common in practice)
//    <float, uint16_t>  + zigzag flag
//
//  (Double-precision variants share the same kernel paths; they would require
//  a typed run_quantizer_forward<double, TCode> helper and are left for a
//  future test when that helper is added.)
// ─────────────────────────────────────────────────────────────────────────────

TEST(QuantizerTypeMatrix, FloatUint16_ABSRoundTrip) {
    // ABS mode with uint16_t codes: max representable quantization index is
    // ±32767.  We use a moderate quant_radius to stay well inside that range.
    CudaStream stream;
    constexpr size_t N  = 2048;
    constexpr float  EB = 0.01f;
    auto pool = make_test_pool(N * sizeof(float) * 20);

    auto h_input = make_sine_floats(N, 0.05f, 10.0f);

    QuantizerStage<float, uint16_t> fwd;
    fwd.setErrorBound(EB);
    fwd.setErrorBoundMode(ErrorBoundMode::ABS);
    fwd.setQuantRadius(512);
    fwd.setOutlierCapacity(0.1f);

    auto fwd_result = run_quantizer_forward<uint16_t>(fwd, h_input, stream, *pool);

    QuantizerStage<float, uint16_t> inv;
    inv.setInverse(true);
    uint8_t cfg[128] = {};
    inv.deserializeHeader(cfg, fwd.serializeHeader(0, cfg, sizeof(cfg)));

    auto h_recon = run_quantizer_inverse<uint16_t>(inv, fwd_result, N, stream, *pool);

    ASSERT_EQ(h_recon.size(), N);
    EXPECT_LE(max_abs_error(h_input, h_recon), EB + 1e-6f)
        << "float/uint16 ABS round-trip max error exceeds error_bound";
}

TEST(QuantizerTypeMatrix, FloatUint16_ConstantExact) {
    // Constant input should produce a single unique code for all elements
    // (no outliers, since the value lies inside the quantization range).
    // With EB=1e-3 and quant_radius=512 the representable range is
    //   ±(512 * 2 * 1e-3) = ±1.024
    // We use 0.5f which maps to code round(0.5 / 0.002) = 250, well inside range.
    CudaStream stream;
    constexpr size_t N   = 2048;
    constexpr float  EB  = 1e-3f;
    constexpr float  VAL = 0.5f;   // inside representable range (±1.024)
    auto pool = make_test_pool(N * sizeof(float) * 20);

    auto h_input = make_constant<float>(N, VAL);

    QuantizerStage<float, uint16_t> fwd;
    fwd.setErrorBound(EB);
    fwd.setErrorBoundMode(ErrorBoundMode::ABS);
    fwd.setQuantRadius(512);
    fwd.setOutlierCapacity(0.05f);

    auto fwd_result = run_quantizer_forward<uint16_t>(fwd, h_input, stream, *pool);

    // Check that all codes carry the same value (one unique quantization bin).
    ASSERT_GE(fwd_result.codes_bytes, N * sizeof(uint16_t));
    std::vector<uint16_t> h_codes(N);
    std::memcpy(h_codes.data(), fwd_result.codes_raw.data(), N * sizeof(uint16_t));
    uint16_t first = h_codes[0];
    EXPECT_NE(first, uint16_t{0}) << "Constant non-zero input should not map to code 0";
    bool all_same = std::all_of(h_codes.begin(), h_codes.end(),
                                 [first](uint16_t v){ return v == first; });
    EXPECT_TRUE(all_same) << "Constant input should produce identical codes";

    QuantizerStage<float, uint16_t> inv;
    inv.setInverse(true);
    uint8_t cfg[128] = {};
    inv.deserializeHeader(cfg, fwd.serializeHeader(0, cfg, sizeof(cfg)));

    auto h_recon = run_quantizer_inverse<uint16_t>(inv, fwd_result, N, stream, *pool);

    ASSERT_EQ(h_recon.size(), N);
    EXPECT_LE(max_abs_error(h_input, h_recon), EB + 1e-6f)
        << "float/uint16 constant round-trip max error exceeds error_bound";
}

TEST(QuantizerTypeMatrix, FloatUint16_ZigzagCodes) {
    // Zigzag remaps signed codes to unsigned: verify that enabling it does
    // not degrade reconstruction quality for the uint16_t code type.
    CudaStream stream;
    constexpr size_t N  = 2048;
    constexpr float  EB = 0.01f;
    auto pool = make_test_pool(N * sizeof(float) * 20);

    auto h_input = make_sine_floats(N, 0.05f, 10.0f);

    auto run_one = [&](bool use_zigzag) {
        QuantizerStage<float, uint16_t> fwd;
        fwd.setErrorBound(EB);
        fwd.setErrorBoundMode(ErrorBoundMode::ABS);
        fwd.setQuantRadius(512);
        fwd.setOutlierCapacity(0.1f);
        fwd.setZigzagCodes(use_zigzag);

        auto fwd_result = run_quantizer_forward<uint16_t>(fwd, h_input, stream, *pool);

        QuantizerStage<float, uint16_t> inv;
        inv.setInverse(true);
        uint8_t cfg[128] = {};
        inv.deserializeHeader(cfg, fwd.serializeHeader(0, cfg, sizeof(cfg)));

        return run_quantizer_inverse<uint16_t>(inv, fwd_result, N, stream, *pool);
    };

    auto h_off = run_one(false);
    auto h_on  = run_one(true);

    ASSERT_EQ(h_off.size(), N);
    ASSERT_EQ(h_on.size(),  N);

    EXPECT_LE(max_abs_error(h_input, h_off), EB + 1e-6f)
        << "float/uint16 zigzag=false round-trip exceeds error_bound";
    EXPECT_LE(max_abs_error(h_input, h_on),  EB + 1e-6f)
        << "float/uint16 zigzag=true  round-trip exceeds error_bound";

    // Zigzag and non-zigzag must give identical reconstructions
    float max_diff = max_abs_error(h_off, h_on);
    EXPECT_EQ(max_diff, 0.0f)
        << "float/uint16: zigzag=true vs false should reconstruct identically; "
        << "max_diff=" << max_diff;
}

// ─────────────────────────────────────────────────────────────────────────────
// Inplace outlier helpers
//
// When setInplaceOutliers(true) the stage has 1 output (codes) not 4.
// Raw float bits of outliers are stored directly in the codes array;
// the sentinel check (code >> 1) >= quant_radius distinguishes them.
// ─────────────────────────────────────────────────────────────────────────────

struct InplaceFwdResult {
    std::vector<uint8_t> codes_raw;
    size_t codes_bytes = 0;
};

static InplaceFwdResult run_quantizer_forward_inplace(
    QuantizerStage<float, uint32_t>& stage,
    const std::vector<float>& h_input,
    cudaStream_t stream,
    fz::MemoryPool& pool)
{
    size_t n = h_input.size();
    size_t in_bytes = n * sizeof(float);

    CudaBuffer<float> d_in(n);
    d_in.upload(h_input, stream);

    auto est = stage.estimateOutputSizes({in_bytes});
    EXPECT_EQ(est.size(), 1u) << "inplace mode should estimate 1 output";

    CudaBuffer<uint8_t> d_codes(est[0]);
    std::vector<void*>  inputs  = {d_in.void_ptr()};
    std::vector<void*>  outputs = {d_codes.void_ptr()};
    std::vector<size_t> sizes   = {in_bytes};

    stage.execute(stream, &pool, inputs, outputs, sizes);
    cudaStreamSynchronize(stream);
    stage.postStreamSync(stream);

    auto actual = stage.getActualOutputSizesByName();
    InplaceFwdResult r;
    r.codes_bytes = actual.count("codes") ? actual.at("codes") : est[0];
    r.codes_raw   = d_codes.download_bytes(r.codes_bytes, stream);
    return r;
}

static std::vector<float> run_quantizer_inverse_inplace(
    QuantizerStage<float, uint32_t>& stage,
    const InplaceFwdResult& fwd,
    size_t n_elements,
    cudaStream_t stream,
    fz::MemoryPool& pool)
{
    CudaBuffer<uint8_t> d_codes(fwd.codes_raw.size());
    d_codes.upload(fwd.codes_raw, stream);
    CudaBuffer<float> d_out(n_elements);

    std::vector<void*>  inputs  = {d_codes.void_ptr()};
    std::vector<void*>  outputs = {d_out.void_ptr()};
    std::vector<size_t> sizes   = {fwd.codes_bytes};

    stage.execute(stream, &pool, inputs, outputs, sizes);
    cudaStreamSynchronize(stream);
    return d_out.download(stream);
}

// ─────────────────────────────────────────────────────────────────────────────
// InplaceOutliers — setInplaceOutliers(true) paths (ABS and NOA)
// ─────────────────────────────────────────────────────────────────────────────

TEST(QuantizerInplace, ABSRoundTrip) {
    // Inplace mode: all outlier bits go into the codes array, no scatter buffers.
    // Uses QuantizerStage<float, uint32_t> as required (sizeof match).
    CudaStream stream;
    constexpr size_t N  = 1024;
    constexpr float  EB = 1e-2f;
    auto pool = make_test_pool(N * sizeof(float) * 20);

    // Use values in range [-20, 20]; quant_radius=1<<22 covers |x|/(2*EB) = 20/0.02 = 1000.
    auto h_input = make_sine_floats(N, 0.05f, 20.0f);

    QuantizerStage<float, uint32_t> fwd;
    fwd.setErrorBound(EB);
    fwd.setErrorBoundMode(ErrorBoundMode::ABS);
    fwd.setQuantRadius(1 << 22);
    fwd.setZigzagCodes(true);
    fwd.setInplaceOutliers(true);

    EXPECT_EQ(fwd.getNumOutputs(), 1) << "inplace mode must expose 1 output";

    auto fwd_result = run_quantizer_forward_inplace(fwd, h_input, stream, *pool);
    EXPECT_EQ(fwd_result.codes_bytes, N * sizeof(uint32_t));

    // Deserialize reconstructs inplace_outliers=true automatically.
    QuantizerStage<float, uint32_t> inv;
    inv.setInverse(true);
    uint8_t cfg[128] = {};
    inv.deserializeHeader(cfg, fwd.serializeHeader(0, cfg, sizeof(cfg)));
    EXPECT_TRUE(inv.getInplaceOutliers())
        << "inplace_outliers must survive header serialization";
    EXPECT_EQ(inv.getNumInputs(), 1) << "inverse inplace mode must accept 1 input";

    auto h_recon = run_quantizer_inverse_inplace(inv, fwd_result, N, stream, *pool);

    ASSERT_EQ(h_recon.size(), N);
    float max_err = max_abs_error(h_input, h_recon);
    EXPECT_LE(max_err, EB * 1.01f)
        << "ABS inplace round-trip max_err=" << max_err << " exceeds bound " << EB;
}

TEST(QuantizerInplace, NOARoundTrip) {
    CudaStream stream;
    constexpr size_t N       = 1024;
    constexpr float  USER_EB = 0.01f;
    auto pool = make_test_pool(N * sizeof(float) * 20);

    auto h_input = make_sine_floats(N, 0.04f, 8.0f);

    float vmin = *std::min_element(h_input.begin(), h_input.end());
    float vmax = *std::max_element(h_input.begin(), h_input.end());
    float expected_abs_eb = USER_EB * (vmax - vmin);

    QuantizerStage<float, uint32_t> fwd;
    fwd.setErrorBound(USER_EB);
    fwd.setErrorBoundMode(ErrorBoundMode::NOA);
    fwd.setQuantRadius(32768);
    fwd.setZigzagCodes(true);
    fwd.setInplaceOutliers(true);

    auto fwd_result = run_quantizer_forward_inplace(fwd, h_input, stream, *pool);
    EXPECT_EQ(fwd_result.codes_bytes, N * sizeof(uint32_t));

    QuantizerStage<float, uint32_t> inv;
    inv.setInverse(true);
    uint8_t cfg[128] = {};
    inv.deserializeHeader(cfg, fwd.serializeHeader(0, cfg, sizeof(cfg)));

    auto h_recon = run_quantizer_inverse_inplace(inv, fwd_result, N, stream, *pool);

    ASSERT_EQ(h_recon.size(), N);
    float max_err = max_abs_error(h_input, h_recon);
    EXPECT_LE(max_err, expected_abs_eb * 1.02f)
        << "NOA inplace round-trip max_err=" << max_err
        << " exceeds expected_abs_eb=" << expected_abs_eb;
}

TEST(QuantizerInplace, AllOutliersRoundTrip) {
    // When every value exceeds quant_radius * 2 * EB, every element is an outlier.
    // With inplace mode each element's raw bits go into codes[i]; inverse recovers exactly.
    CudaStream stream;
    constexpr size_t N  = 512;
    constexpr float  EB = 1e-3f;
    auto pool = make_test_pool(N * sizeof(float) * 20);

    // quant_radius=8 → representable range is ±(8 * 2e-3) = ±0.016.
    // Use a constant value of 5.0 — well outside the representable range,
    // so every element is an outlier.  (Sine data has zero-crossings that
    // fall *inside* the range and would get quantized, not stored as outliers.)
    auto h_input = make_constant<float>(N, 5.0f);

    QuantizerStage<float, uint32_t> fwd;
    fwd.setErrorBound(EB);
    fwd.setErrorBoundMode(ErrorBoundMode::ABS);
    fwd.setQuantRadius(8);
    fwd.setZigzagCodes(true);
    fwd.setInplaceOutliers(true);

    auto fwd_result = run_quantizer_forward_inplace(fwd, h_input, stream, *pool);

    QuantizerStage<float, uint32_t> inv;
    inv.setInverse(true);
    uint8_t cfg[128] = {};
    inv.deserializeHeader(cfg, fwd.serializeHeader(0, cfg, sizeof(cfg)));

    auto h_recon = run_quantizer_inverse_inplace(inv, fwd_result, N, stream, *pool);

    ASSERT_EQ(h_recon.size(), N);
    // All outliers must be reconstructed bit-exactly.
    for (size_t i = 0; i < N; i++)
        EXPECT_FLOAT_EQ(h_recon[i], h_input[i])
            << "All-outlier inplace: mismatch at index " << i;
}

TEST(QuantizerInplace, SerializeDeserialize) {
    // The inplace_outliers flag and outlier_threshold must survive a header round-trip.
    QuantizerStage<float, uint32_t> src;
    src.setErrorBound(1e-3f);
    src.setErrorBoundMode(ErrorBoundMode::ABS);
    src.setQuantRadius(1 << 22);
    src.setZigzagCodes(true);
    src.setInplaceOutliers(true);
    src.setOutlierThreshold(500.0f);

    uint8_t cfg[128] = {};
    size_t sz = src.serializeHeader(0, cfg, sizeof(cfg));

    QuantizerStage<float, uint32_t> dst;
    dst.setInverse(true);
    dst.deserializeHeader(cfg, sz);

    EXPECT_TRUE(dst.getInplaceOutliers())
        << "inplace_outliers=true not recovered after serialization";
    EXPECT_FLOAT_EQ(dst.getOutlierThreshold(), 500.0f)
        << "outlier_threshold not recovered after serialization";
}

TEST(QuantizerInplace, BackwardCompatOldHeader) {
    // A header written with the old 28-byte QuantizerConfig (before inplace/threshold
    // fields were added) must deserialize with safe defaults:
    //   inplace_outliers = false
    //   outlier_threshold = infinity
    //
    // Actual old QuantizerConfig layout (28 bytes):
    //   [0..3]   abs_error_bound  (float)
    //   [4..7]   user_error_bound (float)  <- getErrorBound() reads this
    //   [8..11]  value_base       (float)
    //   [12..15] quant_radius     (uint32_t)
    //   [16..19] num_elements     (uint32_t)
    //   [20..23] outlier_count    (uint32_t)
    //   [24]     input_type       (uint8_t)
    //   [25]     code_type        (uint8_t)
    //   [26]     eb_mode          (uint8_t)
    //   [27]     zigzag_codes     (uint8_t)
    struct OldConfig {
        float    abs_error_bound;
        float    user_error_bound;
        float    value_base;
        uint32_t quant_radius;
        uint32_t num_elements;
        uint32_t outlier_count;
        uint8_t  input_type;
        uint8_t  code_type;
        uint8_t  eb_mode;
        uint8_t  zigzag_codes;
    };
    static_assert(sizeof(OldConfig) == 28, "OldConfig must be 28 bytes");

    OldConfig old{};
    old.abs_error_bound  = 1e-3f;
    old.user_error_bound = 1e-3f;  // getErrorBound() reads user_error_bound
    old.value_base       = 0.0f;
    old.quant_radius     = 32768;
    old.num_elements     = 0;
    old.outlier_count    = 0;
    old.input_type       = 3;  // DataType::FLOAT32
    old.code_type        = 5;  // DataType::UINT32
    old.eb_mode          = 0;  // ErrorBoundMode::ABS
    old.zigzag_codes     = 1;

    QuantizerStage<float, uint32_t> dst;
    dst.setInverse(true);
    dst.deserializeHeader(reinterpret_cast<uint8_t*>(&old), sizeof(old));

    EXPECT_FALSE(dst.getInplaceOutliers())
        << "Old header: inplace_outliers should default to false";
    EXPECT_TRUE(std::isinf(dst.getOutlierThreshold()) && dst.getOutlierThreshold() > 0)
        << "Old header: outlier_threshold should default to +infinity";
    EXPECT_FLOAT_EQ(dst.getErrorBound(), 1e-3f);
    EXPECT_EQ(dst.getQuantRadius(), 32768u);
}

// ─────────────────────────────────────────────────────────────────────────────
// OutlierThreshold — setOutlierThreshold() in ABS, NOA, REL modes
//
// Threshold forces any element with |x| >= threshold to the outlier path,
// regardless of whether it would otherwise fit in the quantization range.
// ─────────────────────────────────────────────────────────────────────────────

TEST(QuantizerThreshold, ABSForcesOutliersAboveThreshold) {
    // Values are in [0, 10]. Set threshold=5 so the upper half become outliers,
    // even though they'd fit in the quantization range (quant_radius=1<<22).
    CudaStream stream;
    constexpr size_t N         = 1024;
    constexpr float  EB        = 1e-2f;
    constexpr float  THRESHOLD = 5.0f;
    auto pool = make_test_pool(N * sizeof(float) * 20);

    // Uniform sweep from 0.5 to 9.5: about half are >= 5.
    std::vector<float> h_input(N);
    for (size_t i = 0; i < N; i++)
        h_input[i] = 0.5f + (static_cast<float>(i) / N) * 9.0f;

    size_t expected_outliers = 0;
    for (float v : h_input) if (std::fabs(v) >= THRESHOLD) ++expected_outliers;
    ASSERT_GT(expected_outliers, 0u) << "Test data has no values >= threshold";

    QuantizerStage<float, uint32_t> fwd;
    fwd.setErrorBound(EB);
    fwd.setErrorBoundMode(ErrorBoundMode::ABS);
    fwd.setQuantRadius(1 << 22);
    fwd.setZigzagCodes(true);
    fwd.setOutlierCapacity(0.6f);
    fwd.setOutlierThreshold(THRESHOLD);
    fwd.setInplaceOutliers(false);  // use scatter mode so we can count outliers

    auto fwd_result = run_quantizer_forward(fwd, h_input, stream, *pool);

    ASSERT_GE(fwd_result.count_raw.size(), sizeof(uint32_t));
    uint32_t outlier_count = 0;
    std::memcpy(&outlier_count, fwd_result.count_raw.data(), sizeof(uint32_t));
    EXPECT_EQ(outlier_count, static_cast<uint32_t>(expected_outliers))
        << "ABS threshold: expected " << expected_outliers
        << " outliers, got " << outlier_count;

    // Full round-trip must still be within EB.
    QuantizerStage<float, uint32_t> inv;
    inv.setInverse(true);
    uint8_t cfg[128] = {};
    inv.deserializeHeader(cfg, fwd.serializeHeader(0, cfg, sizeof(cfg)));
    auto h_recon = run_quantizer_inverse(inv, fwd_result, N, stream, *pool);

    ASSERT_EQ(h_recon.size(), N);
    float max_err = max_abs_error(h_input, h_recon);
    EXPECT_LE(max_err, EB * 1.01f)
        << "ABS threshold round-trip max_err=" << max_err << " exceeds EB";
}

TEST(QuantizerThreshold, InplaceABSRoundTrip) {
    // Same threshold test but with inplace outliers; confirms the two features
    // compose correctly.
    CudaStream stream;
    constexpr size_t N         = 1024;
    constexpr float  EB        = 1e-2f;
    constexpr float  THRESHOLD = 5.0f;
    auto pool = make_test_pool(N * sizeof(float) * 20);

    std::vector<float> h_input(N);
    for (size_t i = 0; i < N; i++)
        h_input[i] = 0.5f + (static_cast<float>(i) / N) * 9.0f;

    QuantizerStage<float, uint32_t> fwd;
    fwd.setErrorBound(EB);
    fwd.setErrorBoundMode(ErrorBoundMode::ABS);
    fwd.setQuantRadius(1 << 22);
    fwd.setZigzagCodes(true);
    fwd.setInplaceOutliers(true);
    fwd.setOutlierThreshold(THRESHOLD);

    auto fwd_result = run_quantizer_forward_inplace(fwd, h_input, stream, *pool);
    EXPECT_EQ(fwd_result.codes_bytes, N * sizeof(uint32_t));

    QuantizerStage<float, uint32_t> inv;
    inv.setInverse(true);
    uint8_t cfg[128] = {};
    inv.deserializeHeader(cfg, fwd.serializeHeader(0, cfg, sizeof(cfg)));
    EXPECT_FLOAT_EQ(inv.getOutlierThreshold(), THRESHOLD);

    auto h_recon = run_quantizer_inverse_inplace(inv, fwd_result, N, stream, *pool);
    ASSERT_EQ(h_recon.size(), N);

    // Values below threshold must be within EB; values at/above must be exact.
    for (size_t i = 0; i < N; i++) {
        if (std::fabs(h_input[i]) >= THRESHOLD) {
            EXPECT_FLOAT_EQ(h_recon[i], h_input[i])
                << "Inplace threshold outlier not exact at index " << i;
        } else {
            EXPECT_NEAR(h_recon[i], h_input[i], EB * 1.01f)
                << "Inplace threshold inlier out of bound at index " << i;
        }
    }
}

TEST(QuantizerThreshold, RELForcesOutliersAboveThreshold) {
    // In REL mode the threshold still routes |x| >= THRESHOLD to the scatter
    // outlier list, even when the log-bin would normally fit.
    CudaStream stream;
    constexpr size_t N         = 512;
    constexpr float  EPS       = 0.01f;
    constexpr float  THRESHOLD = 4.0f;
    auto pool = make_test_pool(N * sizeof(float) * 20);

    // Values span [1, 8]; roughly half are above THRESHOLD.
    std::vector<float> h_input(N);
    for (size_t i = 0; i < N; i++)
        h_input[i] = 1.0f + (static_cast<float>(i) / N) * 7.0f;

    size_t expected_outliers = 0;
    for (float v : h_input) if (std::fabs(v) >= THRESHOLD) ++expected_outliers;
    ASSERT_GT(expected_outliers, 0u);

    QuantizerStage<float, uint32_t> fwd;
    fwd.setErrorBound(EPS);
    fwd.setErrorBoundMode(ErrorBoundMode::REL);
    fwd.setQuantRadius(512);
    fwd.setOutlierCapacity(0.6f);
    fwd.setOutlierThreshold(THRESHOLD);

    auto fwd_result = run_quantizer_forward(fwd, h_input, stream, *pool);

    ASSERT_GE(fwd_result.count_raw.size(), sizeof(uint32_t));
    uint32_t outlier_count = 0;
    std::memcpy(&outlier_count, fwd_result.count_raw.data(), sizeof(uint32_t));
    EXPECT_GE(outlier_count, static_cast<uint32_t>(expected_outliers))
        << "REL threshold: at least " << expected_outliers
        << " outliers expected, got " << outlier_count;

    // Full round-trip.
    QuantizerStage<float, uint32_t> inv;
    inv.setInverse(true);
    uint8_t cfg[128] = {};
    inv.deserializeHeader(cfg, fwd.serializeHeader(0, cfg, sizeof(cfg)));
    auto h_recon = run_quantizer_inverse(inv, fwd_result, N, stream, *pool);

    ASSERT_EQ(h_recon.size(), N);
    // Outlier elements must be exact; non-outlier relative error must be <= EPS.
    std::vector<uint32_t> outlier_idxs;
    outlier_idxs.resize(outlier_count);
    std::memcpy(outlier_idxs.data(), fwd_result.idxs_raw.data(),
                outlier_count * sizeof(uint32_t));
    std::sort(outlier_idxs.begin(), outlier_idxs.end());

    for (size_t i = 0; i < N; i++) {
        bool is_outlier = std::binary_search(outlier_idxs.begin(), outlier_idxs.end(),
                                              static_cast<uint32_t>(i));
        if (is_outlier) {
            EXPECT_FLOAT_EQ(h_recon[i], h_input[i])
                << "REL threshold outlier not exact at index " << i;
        } else {
            float rel = std::abs(h_recon[i] - h_input[i]) / std::abs(h_input[i]);
            EXPECT_LE(rel, EPS * 1.5f)
                << "REL threshold inlier rel_err=" << rel << " at index " << i;
        }
    }
}

TEST(QuantizerThreshold, InfThresholdEquivalentToNoThreshold) {
    // Explicitly setting threshold=inf must produce identical output to not
    // setting it at all (default).
    CudaStream stream;
    constexpr size_t N  = 512;
    constexpr float  EB = 1e-2f;
    auto pool = make_test_pool(N * sizeof(float) * 20);

    auto h_input = make_sine_floats(N, 0.05f, 8.0f);

    auto run_one = [&](bool set_explicit_inf) {
        QuantizerStage<float, uint32_t> fwd;
        fwd.setErrorBound(EB);
        fwd.setErrorBoundMode(ErrorBoundMode::ABS);
        fwd.setQuantRadius(1 << 22);
        fwd.setZigzagCodes(true);
        fwd.setInplaceOutliers(true);
        if (set_explicit_inf)
            fwd.setOutlierThreshold(std::numeric_limits<float>::infinity());

        auto fwd_result = run_quantizer_forward_inplace(fwd, h_input, stream, *pool);

        QuantizerStage<float, uint32_t> inv;
        inv.setInverse(true);
        uint8_t cfg[128] = {};
        inv.deserializeHeader(cfg, fwd.serializeHeader(0, cfg, sizeof(cfg)));
        return run_quantizer_inverse_inplace(inv, fwd_result, N, stream, *pool);
    };

    auto h_default = run_one(false);
    auto h_inf     = run_one(true);

    ASSERT_EQ(h_default.size(), N);
    ASSERT_EQ(h_inf.size(), N);

    float max_diff = max_abs_error(h_default, h_inf);
    EXPECT_EQ(max_diff, 0.0f)
        << "inf threshold should give identical output to default; max_diff=" << max_diff;
}

TEST(QuantizerThreshold, SerializeDeserialize) {
    // threshold value must round-trip through the header exactly.
    QuantizerStage<float, uint32_t> src;
    src.setErrorBound(1e-3f);
    src.setErrorBoundMode(ErrorBoundMode::ABS);
    src.setQuantRadius(1 << 22);
    src.setOutlierThreshold(1234.5f);

    uint8_t cfg[128] = {};
    size_t sz = src.serializeHeader(0, cfg, sizeof(cfg));

    QuantizerStage<float, uint32_t> dst;
    dst.setInverse(true);
    dst.deserializeHeader(cfg, sz);

    EXPECT_FLOAT_EQ(dst.getOutlierThreshold(), 1234.5f)
        << "outlier_threshold not recovered after serialization";
    // inplace_outliers was not set, so it should default to false
    EXPECT_FALSE(dst.getInplaceOutliers());
}

// ─────────────────────────────────────────────────────────────────────────────
//  QuantizerTypeMatrix — double-precision variants
//
//  QuantizerStage<double, uint16_t> and <double, uint32_t> are instantiated in
//  quantizer.cu.  These tests verify the double-precision kernel path runs
//  correctly end-to-end using the full Pipeline API.
// ─────────────────────────────────────────────────────────────────────────────

// QD1: QuantizerStage<double, uint16_t> ABS round-trip via Pipeline
TEST(QuantizerTypeMatrix, DoubleUint16_PipelineRoundTrip) {
    CudaStream stream;
    constexpr size_t N  = 1 << 12;   // 4 K doubles
    constexpr double EB = 1e-2;
    const size_t in_bytes = N * sizeof(double);

    std::vector<double> h_input(N);
    for (size_t i = 0; i < N; i++)
        h_input[i] = std::sin(static_cast<double>(i) * 0.01) * 50.0
                   + std::cos(static_cast<double>(i) * 0.003) * 20.0;

    CudaBuffer<double> d_in(N);
    d_in.upload(h_input, stream);
    stream.sync();

    Pipeline pipeline(in_bytes, MemoryStrategy::MINIMAL, 5.0f);
    auto* q = pipeline.addStage<QuantizerStage<double, uint16_t>>();
    q->setErrorBound(static_cast<float>(EB));
    q->setErrorBoundMode(ErrorBoundMode::ABS);
    // quant_radius=8192: max |q| = round(70/0.02)=3500 < 8192, so no outliers
    // for this smooth sinusoidal data (avoids scatter-buffer overflow).
    q->setQuantRadius(8192);
    q->setOutlierCapacity(0.1f);
    pipeline.setPoolManagedDecompOutput(false);
    pipeline.finalize();

    void*  d_comp  = nullptr;
    size_t comp_sz = 0;
    ASSERT_NO_THROW(
        pipeline.compress(d_in.void_ptr(), in_bytes, &d_comp, &comp_sz, stream)
    ) << "QuantizerStage<double,uint16_t> compress must not throw";
    stream.sync();
    ASSERT_GT(comp_sz, 0u);

    void*  d_dec  = nullptr;
    size_t dec_sz = 0;
    ASSERT_NO_THROW(
        pipeline.decompress(nullptr, 0, &d_dec, &dec_sz, stream)
    ) << "QuantizerStage<double,uint16_t> decompress must not throw";
    stream.sync();
    ASSERT_NE(d_dec, nullptr);
    ASSERT_EQ(dec_sz, in_bytes);

    std::vector<double> h_recon(N);
    cudaMemcpy(h_recon.data(), d_dec, in_bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_dec);

    double max_err = 0.0;
    for (size_t i = 0; i < N; i++)
        max_err = std::max(max_err, std::abs(h_recon[i] - h_input[i]));
    EXPECT_LE(max_err, EB * 1.01)
        << "QuantizerStage<double,uint16_t> round-trip max_err=" << max_err;
}

// QD2: QuantizerStage<double, uint32_t> ABS round-trip via Pipeline
TEST(QuantizerTypeMatrix, DoubleUint32_PipelineRoundTrip) {
    CudaStream stream;
    constexpr size_t N  = 1 << 12;   // 4 K doubles
    constexpr double EB = 1e-3;
    const size_t in_bytes = N * sizeof(double);

    std::vector<double> h_input(N);
    for (size_t i = 0; i < N; i++)
        h_input[i] = std::sin(static_cast<double>(i) * 0.01) * 50.0
                   + std::cos(static_cast<double>(i) * 0.003) * 20.0;

    CudaBuffer<double> d_in(N);
    d_in.upload(h_input, stream);
    stream.sync();

    Pipeline pipeline(in_bytes, MemoryStrategy::MINIMAL, 5.0f);
    auto* q = pipeline.addStage<QuantizerStage<double, uint32_t>>();
    q->setErrorBound(static_cast<float>(EB));
    q->setErrorBoundMode(ErrorBoundMode::ABS);
    q->setQuantRadius(1 << 20);  // wide range for uint32 codes
    q->setOutlierCapacity(0.05f);
    pipeline.setPoolManagedDecompOutput(false);
    pipeline.finalize();

    void*  d_comp  = nullptr;
    size_t comp_sz = 0;
    ASSERT_NO_THROW(
        pipeline.compress(d_in.void_ptr(), in_bytes, &d_comp, &comp_sz, stream)
    ) << "QuantizerStage<double,uint32_t> compress must not throw";
    stream.sync();
    ASSERT_GT(comp_sz, 0u);

    void*  d_dec  = nullptr;
    size_t dec_sz = 0;
    ASSERT_NO_THROW(
        pipeline.decompress(nullptr, 0, &d_dec, &dec_sz, stream)
    ) << "QuantizerStage<double,uint32_t> decompress must not throw";
    stream.sync();
    ASSERT_NE(d_dec, nullptr);
    ASSERT_EQ(dec_sz, in_bytes);

    std::vector<double> h_recon(N);
    cudaMemcpy(h_recon.data(), d_dec, in_bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_dec);

    double max_err = 0.0;
    for (size_t i = 0; i < N; i++)
        max_err = std::max(max_err, std::abs(h_recon[i] - h_input[i]));
    EXPECT_LE(max_err, EB * 1.01)
        << "QuantizerStage<double,uint32_t> round-trip max_err=" << max_err;
}
