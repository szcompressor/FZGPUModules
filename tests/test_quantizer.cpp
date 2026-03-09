/**
 * tests/test_quantizer.cpp
 *
 * Unit tests for QuantizerStage<float, uint16_t> (ABS and NOA modes)
 * and QuantizerStage<float, uint32_t> (REL mode).
 *
 * QuantizerStage quantises input *values* directly (unlike LorenzoStage which
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
#include "predictors/quantizer/quantizer.h"
#include "predictors/lorenzo/lorenzo.h"   // for ErrorBoundMode

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

    r.codes_raw = d_codes.download(stream); r.codes_raw.resize(r.codes_bytes);
    r.vals_raw  = d_vals.download(stream);  r.vals_raw.resize(r.vals_bytes);
    r.idxs_raw  = d_idxs.download(stream);  r.idxs_raw.resize(r.idxs_bytes);
    r.count_raw = d_count.download(stream); r.count_raw.resize(r.count_bytes);

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

    std::vector<float> h_input(N);
    for (size_t i = 0; i < N; i++)
        h_input[i] = std::sin(static_cast<float>(i) * 0.05f) * 10.0f;

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

    std::vector<float> h_input(N);
    for (size_t i = 0; i < N; i++)
        h_input[i] = std::sin(static_cast<float>(i) * 0.05f) * 10.0f;

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
