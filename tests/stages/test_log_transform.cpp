/**
 * tests/stages/test_log_transform.cpp
 *
 * GPU unit tests for LogTransformStage<float> — the log-space transform that
 * converts a point-wise relative bound into an absolute one (Liang et al.,
 * IEEE CLUSTER 2018).
 *
 *   LG1   LogTransform/ForwardValuesAreLog2       — output[i] == log2(|x[i]|)
 *   LG2   LogTransform/SignBitmapPacking          — one bit per element, correct order
 *   LG3   LogTransform/RoundTripPositive          — transform+inverse is ~exact
 *   LG4   LogTransform/RoundTripMixedSign         — negatives survive the sign channel
 *   LG5   LogTransform/SpecialValuesAreOutliers   — 0/denormal/inf/NaN stored losslessly
 *   LG6   LogTransform/ThresholdEscalatesSmall    — |x| < threshold becomes an outlier
 *   LG7   LogTransform/QuantizerErrorBoundMath    — log2(1+delta), the whole point
 *   LG8   LogTransform/EstimateOutputSizes        — 4 forward ports sized correctly
 *   LG9   LogTransform/HeaderRoundTrip            — config survives serialize/deserialize
 *   LG10  LogTransform/PortNamesAndArity          — port contract
 *   LG11  LogTransform/RelativeBoundHolds         — the actual guarantee, end to end
 *   LG12  LogTransform/ZeroErrorBoundThrows       — delta <= 0 is rejected
 */

#include <gtest/gtest.h>
#include "helpers/fz_test_utils.h"
#include "transforms/log_transform/log_transform_stage.h"
#include "fzgpumodules.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

using namespace fz;
using namespace fz_test;

namespace {

/// Result of one forward pass, pulled back to the host.
struct FwdResult {
    std::vector<float>    log_values;
    std::vector<uint8_t>  signs;
    std::vector<float>    outlier_vals;
    std::vector<uint32_t> outlier_idxs;
    uint32_t              outlier_count = 0;
};

/// Run the forward transform directly on the stage (no Pipeline), so the test
/// can inspect every port including the ones a DAG would normally consume.
FwdResult run_forward(LogTransformStage<float>& stage,
                      const std::vector<float>& h_in,
                      cudaStream_t stream,
                      MemoryPool& pool)
{
    const size_t n     = h_in.size();
    const size_t bytes = n * sizeof(float);

    auto sizes = stage.estimateOutputSizes({bytes});
    EXPECT_EQ(sizes.size(), 4u);

    CudaBuffer<float>    d_in(n);
    CudaBuffer<float>    d_log(sizes[0] / sizeof(float));
    CudaBuffer<uint8_t>  d_signs(sizes[1]);
    CudaBuffer<float>    d_ovals(sizes[2] / sizeof(float) + 1);
    CudaBuffer<uint32_t> d_oidxs(sizes[3] / sizeof(uint32_t) + 1);

    d_in.upload(h_in, stream);
    cudaStreamSynchronize(stream);

    stage.onFinalize(bytes, &pool);
    stage.execute(stream, &pool,
                  {d_in.void_ptr()},
                  {d_log.void_ptr(), d_signs.void_ptr(),
                   d_ovals.void_ptr(), d_oidxs.void_ptr()},
                  {bytes});
    cudaStreamSynchronize(stream);
    stage.postStreamSync(stream);

    FwdResult r;
    r.outlier_count = stage.getOutlierCount();
    r.log_values = d_log.download_bytes(n * sizeof(float), stream);
    r.signs      = d_signs.download_bytes(sizes[1], stream);
    if (r.outlier_count > 0) {
        r.outlier_vals = d_ovals.download_bytes(
            r.outlier_count * sizeof(float), stream);
        r.outlier_idxs = d_oidxs.download_bytes(
            r.outlier_count * sizeof(uint32_t), stream);
    }
    cudaStreamSynchronize(stream);
    return r;
}

/// Forward then inverse, returning the reconstruction. Exercises the real
/// serialize/deserialize path so the outlier count travels the way it does in
/// a file, not via shared object state.
std::vector<float> round_trip(const std::vector<float>& h_in,
                              float delta,
                              float threshold,
                              cudaStream_t stream,
                              MemoryPool& pool)
{
    const size_t n     = h_in.size();
    const size_t bytes = n * sizeof(float);

    LogTransformStage<float> fwd;
    fwd.setErrorBound(delta);
    fwd.setThreshold(threshold);
    FwdResult f = run_forward(fwd, h_in, stream, pool);

    uint8_t hdr[128] = {};
    const size_t hdr_sz = fwd.serializeHeader(0, hdr, sizeof(hdr));

    LogTransformStage<float> inv;
    inv.setInverse(true);
    inv.deserializeHeader(hdr, hdr_sz);

    CudaBuffer<float>    d_log(n);
    CudaBuffer<uint8_t>  d_signs(f.signs.size());
    CudaBuffer<float>    d_ovals(f.outlier_count + 1);
    CudaBuffer<uint32_t> d_oidxs(f.outlier_count + 1);
    CudaBuffer<float>    d_out(n);

    d_log.upload(f.log_values, stream);
    cudaMemcpyAsync(d_signs.get(), f.signs.data(), f.signs.size(),
                    cudaMemcpyHostToDevice, stream);
    if (f.outlier_count > 0) {
        cudaMemcpyAsync(d_ovals.get(), f.outlier_vals.data(),
                        f.outlier_count * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_oidxs.get(), f.outlier_idxs.data(),
                        f.outlier_count * sizeof(uint32_t),
                        cudaMemcpyHostToDevice, stream);
    }
    cudaStreamSynchronize(stream);

    inv.execute(stream, &pool,
                {d_log.void_ptr(), d_signs.void_ptr(),
                 d_ovals.void_ptr(), d_oidxs.void_ptr()},
                {d_out.void_ptr()},
                {n * sizeof(float)});
    cudaStreamSynchronize(stream);

    auto h_out = d_out.download(stream);
    cudaStreamSynchronize(stream);
    return h_out;
}

} // namespace

// ─────────────────────────────────────────────────────────────────────────────

TEST(LogTransform, ForwardValuesAreLog2) {
    CudaStream cs;
    auto pool = make_test_pool(1 << 20);

    const std::vector<float> h_in = {1.0f, 2.0f, 4.0f, 0.5f, 1024.0f, 3.7f};

    LogTransformStage<float> stage;
    stage.setErrorBound(1e-3f);
    FwdResult r = run_forward(stage, h_in, cs.stream, *pool);

    EXPECT_EQ(r.outlier_count, 0u);
    for (size_t i = 0; i < h_in.size(); ++i) {
        EXPECT_NEAR(r.log_values[i], std::log2(std::fabs(h_in[i])), 1e-5f)
            << "element " << i;
    }
}

TEST(LogTransform, SignBitmapPacking) {
    CudaStream cs;
    auto pool = make_test_pool(1 << 20);

    // 10 elements => 2 sign bytes. Alternate so a misordered pack is obvious.
    std::vector<float> h_in;
    for (int i = 0; i < 10; ++i)
        h_in.push_back((i % 2 == 0) ? 1.5f : -1.5f);

    LogTransformStage<float> stage;
    stage.setErrorBound(1e-3f);
    FwdResult r = run_forward(stage, h_in, cs.stream, *pool);

    ASSERT_EQ(r.signs.size(), 2u);
    for (size_t i = 0; i < h_in.size(); ++i) {
        const bool bit = (r.signs[i >> 3] >> (i & 7)) & 1u;
        EXPECT_EQ(bit, h_in[i] < 0.0f) << "sign bit " << i;
    }
    // The magnitude stream must be sign-free: |1.5| for every element.
    for (size_t i = 0; i < h_in.size(); ++i)
        EXPECT_NEAR(r.log_values[i], std::log2(1.5f), 1e-5f);
}

TEST(LogTransform, RoundTripPositive) {
    CudaStream cs;
    auto pool = make_test_pool(1 << 22);

    std::vector<float> h_in;
    for (int i = 1; i <= 4096; ++i)
        h_in.push_back(static_cast<float>(i) * 0.125f);

    auto h_out = round_trip(h_in, 1e-3f, 0.0f, cs.stream, *pool);

    for (size_t i = 0; i < h_in.size(); ++i) {
        // The transform alone is near-lossless; only float32 log/exp2 rounding
        // separates input from output. Well inside the delta budget.
        const float rel = std::fabs(h_out[i] - h_in[i]) / std::fabs(h_in[i]);
        EXPECT_LT(rel, 1e-5f) << "element " << i;
    }
}

TEST(LogTransform, RoundTripMixedSign) {
    CudaStream cs;
    auto pool = make_test_pool(1 << 22);

    std::vector<float> h_in;
    for (int i = 1; i <= 2048; ++i) {
        const float v = static_cast<float>(i) * 0.5f;
        h_in.push_back((i % 3 == 0) ? -v : v);
    }

    auto h_out = round_trip(h_in, 1e-3f, 0.0f, cs.stream, *pool);

    for (size_t i = 0; i < h_in.size(); ++i) {
        EXPECT_EQ(std::signbit(h_out[i]), std::signbit(h_in[i]))
            << "sign lost at " << i;
        const float rel = std::fabs(h_out[i] - h_in[i]) / std::fabs(h_in[i]);
        EXPECT_LT(rel, 1e-5f) << "element " << i;
    }
}

TEST(LogTransform, SpecialValuesAreOutliers) {
    CudaStream cs;
    auto pool = make_test_pool(1 << 20);

    const float denorm = std::numeric_limits<float>::denorm_min();
    const float inf    = std::numeric_limits<float>::infinity();
    std::vector<float> h_in = {1.0f, 0.0f, 2.0f, denorm, 4.0f, inf, 8.0f, -0.0f};

    LogTransformStage<float> stage;
    stage.setErrorBound(1e-3f);
    stage.setOutlierCapacity(1.0f);
    FwdResult r = run_forward(stage, h_in, cs.stream, *pool);

    // 0.0, denorm, inf and -0.0 have no usable log2.
    EXPECT_EQ(r.outlier_count, 4u);

    // Every escalated index must be one of the special positions, and the
    // stored value must be bit-identical to the input.
    for (uint32_t k = 0; k < r.outlier_count; ++k) {
        const uint32_t idx = r.outlier_idxs[k];
        ASSERT_LT(idx, h_in.size());
        const bool is_special = (idx == 1 || idx == 3 || idx == 5 || idx == 7);
        EXPECT_TRUE(is_special) << "unexpected outlier at index " << idx;
        EXPECT_EQ(std::memcmp(&r.outlier_vals[k], &h_in[idx], sizeof(float)), 0)
            << "outlier value not bit-exact at index " << idx;
    }

    // And they must come back bit-exactly, including the sign of negative zero.
    auto h_out = round_trip(h_in, 1e-3f, 0.0f, cs.stream, *pool);
    EXPECT_EQ(h_out[1], 0.0f);
    EXPECT_EQ(h_out[3], denorm);
    EXPECT_EQ(h_out[5], inf);
    EXPECT_TRUE(std::signbit(h_out[7])) << "negative zero lost its sign";
}

TEST(LogTransform, ThresholdEscalatesSmall) {
    CudaStream cs;
    auto pool = make_test_pool(1 << 20);

    std::vector<float> h_in = {1.0f, 1e-6f, 2.0f, 1e-8f, 4.0f};

    LogTransformStage<float> stage;
    stage.setErrorBound(1e-3f);
    stage.setThreshold(1e-5f);
    stage.setOutlierCapacity(1.0f);
    FwdResult r = run_forward(stage, h_in, cs.stream, *pool);

    EXPECT_EQ(r.outlier_count, 2u);
    for (uint32_t k = 0; k < r.outlier_count; ++k) {
        const uint32_t idx = r.outlier_idxs[k];
        EXPECT_TRUE(idx == 1 || idx == 3) << "unexpected outlier index " << idx;
    }

    // Escalated positions are parked at the log floor, not left as a spike in
    // the middle of the range — that is what keeps the stream predictable.
    EXPECT_NEAR(r.log_values[1], std::log2(1e-5f), 1e-4f);

    auto h_out = round_trip(h_in, 1e-3f, 1e-5f, cs.stream, *pool);
    EXPECT_EQ(h_out[1], 1e-6f);
    EXPECT_EQ(h_out[3], 1e-8f);
}

TEST(LogTransform, QuantizerErrorBoundMath) {
    // The identity the whole stage rests on: an absolute bound of log2(1+delta)
    // in log space is a relative bound of delta in value space.
    LogTransformStage<float> stage;
    stage.setErrorBound(1e-3f);

    const float slack = LogTransformStage<float>::kLogRoundTripSlack;

    // log2(1+delta), minus the slack reserved for the transform's own rounding.
    EXPECT_FLOAT_EQ(stage.quantizerErrorBound(), std::log2(1.0f + 1e-3f) - slack);
    EXPECT_FLOAT_EQ(LogTransformStage<float>::quantizerErrorBoundFor(0.01f),
                    std::log2(1.01f) - slack);

    // The bound lives in log2 units, so it is not comparable to delta directly.
    // The meaningful check is that exponentiating it lands just *inside* the
    // intended multiplicative tolerance — inside, never outside.
    const float achieved = std::exp2(stage.quantizerErrorBound());
    EXPECT_LT(achieved, 1.0f + 1e-3f);
    EXPECT_NEAR(achieved, 1.0f + 1e-3f, 1e-5f);

    // The slack must be a rounding-scale haircut, not a meaningful loss of
    // bound: well under 1% of the budget at delta = 1e-3.
    EXPECT_LT(slack / std::log2(1.0f + 1e-3f), 0.01f);
}

TEST(LogTransform, EstimateOutputSizes) {
    LogTransformStage<float> stage;
    stage.setOutlierCapacity(0.05f);

    const size_t n = 1000;
    auto sizes = stage.estimateOutputSizes({n * sizeof(float)});
    ASSERT_EQ(sizes.size(), 4u);
    EXPECT_EQ(sizes[0], n * sizeof(float));   // log values
    EXPECT_EQ(sizes[1], (n + 7) / 8);         // sign bitmap
    EXPECT_EQ(sizes[2], 50 * sizeof(float));  // outlier vals @ 5%
    EXPECT_EQ(sizes[3], 50 * sizeof(uint32_t));

    // Non-multiple-of-8 element counts must round the bitmap up, not down.
    auto odd = stage.estimateOutputSizes({7 * sizeof(float)});
    EXPECT_EQ(odd[1], 1u);
    auto odd9 = stage.estimateOutputSizes({9 * sizeof(float)});
    EXPECT_EQ(odd9[1], 2u);

    // Small inputs get the outlier-slot floor rather than a reserve of zero,
    // capped at n (you cannot have more outliers than elements).
    EXPECT_EQ(odd[2], 7 * sizeof(float));
    EXPECT_EQ(odd9[2], 8 * sizeof(float));

    // Inverse direction: one output, one element per log value.
    LogTransformStage<float> inv;
    inv.setInverse(true);
    auto isz = inv.estimateOutputSizes({n * sizeof(float)});
    ASSERT_EQ(isz.size(), 1u);
    EXPECT_EQ(isz[0], n * sizeof(float));
}

TEST(LogTransform, HeaderRoundTrip) {
    LogTransformStage<float> src;
    src.setErrorBound(2.5e-3f);
    src.setThreshold(1e-7f);

    uint8_t buf[128] = {};
    const size_t sz = src.serializeHeader(0, buf, sizeof(buf));
    EXPECT_EQ(sz, sizeof(LogTransformConfig));

    LogTransformStage<float> dst;
    dst.setInverse(true);
    dst.deserializeHeader(buf, sz);

    EXPECT_FLOAT_EQ(dst.getErrorBound(), 2.5e-3f);
    EXPECT_FLOAT_EQ(dst.getThreshold(), 1e-7f);
}

TEST(LogTransform, PortNamesAndArity) {
    LogTransformStage<float> fwd;
    EXPECT_EQ(fwd.getNumInputs(), 1u);
    EXPECT_EQ(fwd.getNumOutputs(), 4u);
    const auto names = fwd.getOutputNames();
    ASSERT_EQ(names.size(), 4u);
    EXPECT_EQ(names[0], "output");
    EXPECT_EQ(names[1], "signs");
    EXPECT_EQ(names[2], "outlier_vals");
    EXPECT_EQ(names[3], "outlier_idxs");
    EXPECT_EQ(fwd.getStageTypeId(),
              static_cast<uint16_t>(StageType::LOG_TRANSFORM));
    EXPECT_TRUE(fwd.isGraphCompatible());

    LogTransformStage<float> inv;
    inv.setInverse(true);
    EXPECT_EQ(inv.getNumInputs(), 4u);
    EXPECT_EQ(inv.getNumOutputs(), 1u);
}

TEST(LogTransform, RelativeBoundHolds) {
    // The claim: quantizing the log stream with an ABS bound of log2(1+delta)
    // yields a *point-wise relative* bound of delta on the reconstruction —
    // uniformly, across many decades of magnitude. This is the property PREL
    // cannot provide, so test it where PREL would fail worst: values spanning
    // six orders of magnitude.
    CudaStream cs;
    auto pool = make_test_pool(1 << 22);

    const float delta = 1e-2f;
    const float abs_eb = LogTransformStage<float>::quantizerErrorBoundFor(delta);

    std::vector<float> h_in;
    for (int d = 0; d < 6; ++d)
        for (int i = 1; i <= 256; ++i)
            h_in.push_back(static_cast<float>(i) * std::pow(10.0f, -static_cast<float>(d)));

    LogTransformStage<float> fwd;
    fwd.setErrorBound(delta);
    FwdResult f = run_forward(fwd, h_in, cs.stream, *pool);
    ASSERT_EQ(f.outlier_count, 0u);

    // Emulate the downstream ABS quantizer on the host: uniform bins of width
    // 2*abs_eb, reconstructed at bin centre.
    std::vector<float> q_log(f.log_values.size());
    for (size_t i = 0; i < f.log_values.size(); ++i) {
        const float step = 2.0f * abs_eb;
        q_log[i] = std::round(f.log_values[i] / step) * step;
        ASSERT_LE(std::fabs(q_log[i] - f.log_values[i]), abs_eb * 1.0001f)
            << "host quantizer emulation violated its own ABS bound at " << i;
    }

    // Invert from the quantized log stream.
    LogTransformStage<float> inv;
    inv.setInverse(true);
    uint8_t hdr[128] = {};
    const size_t hdr_sz = fwd.serializeHeader(0, hdr, sizeof(hdr));
    inv.deserializeHeader(hdr, hdr_sz);

    const size_t n = h_in.size();
    CudaBuffer<float>    d_log(n);
    CudaBuffer<uint8_t>  d_signs(f.signs.size());
    CudaBuffer<float>    d_ovals(1);
    CudaBuffer<uint32_t> d_oidxs(1);
    CudaBuffer<float>    d_out(n);

    d_log.upload(q_log, cs.stream);
    cudaMemcpyAsync(d_signs.get(), f.signs.data(), f.signs.size(),
                    cudaMemcpyHostToDevice, cs.stream);
    cudaStreamSynchronize(cs.stream);

    inv.execute(cs.stream, pool.get(),
                {d_log.void_ptr(), d_signs.void_ptr(),
                 d_ovals.void_ptr(), d_oidxs.void_ptr()},
                {d_out.void_ptr()},
                {n * sizeof(float)});
    cudaStreamSynchronize(cs.stream);

    auto h_out = d_out.download(cs.stream);
    cudaStreamSynchronize(cs.stream);

    // A few float32 ULP of slack: the guarantee is enforced in float32 and the
    // check runs in double, same reasoning as examples/eb_mode_analysis.cpp.
    const double slack = 8.0 * static_cast<double>(
        std::numeric_limits<float>::epsilon());

    size_t violations = 0;
    for (size_t i = 0; i < n; ++i) {
        const double x   = static_cast<double>(h_in[i]);
        const double rel = std::fabs(x - static_cast<double>(h_out[i])) / std::fabs(x);
        if (rel > static_cast<double>(delta) + slack) ++violations;
    }
    EXPECT_EQ(violations, 0u)
        << violations << " / " << n
        << " elements exceeded the point-wise relative bound";
}

TEST(LogTransform, BelowFloat32FloorThrows) {
    // A delta so tight that the log-space round-trip slack alone would exceed
    // it cannot be honoured in float32. Refuse rather than emit a stream that
    // silently violates the bound.
    CudaStream cs;
    auto pool = make_test_pool(1 << 18);

    const std::vector<float> h_in = {1.0f, 2.0f, 3.0f, 4.0f};
    LogTransformStage<float> stage;
    stage.setErrorBound(LogTransformStage<float>::minimumErrorBound() * 0.5f);

    EXPECT_THROW(run_forward(stage, h_in, cs.stream, *pool), std::runtime_error);

    // Just above the floor is accepted.
    LogTransformStage<float> ok;
    ok.setErrorBound(LogTransformStage<float>::minimumErrorBound() * 10.0f);
    EXPECT_NO_THROW(run_forward(ok, h_in, cs.stream, *pool));
}

TEST(LogTransform, ZeroErrorBoundThrows) {
    CudaStream cs;
    auto pool = make_test_pool(1 << 18);

    const std::vector<float> h_in = {1.0f, 2.0f, 3.0f, 4.0f};
    LogTransformStage<float> stage;
    stage.setErrorBound(0.0f);   // delta must be positive

    EXPECT_THROW(run_forward(stage, h_in, cs.stream, *pool), std::runtime_error);
}

// ─────────────────────────────────────────────────────────────────────────────
// LG13-LG14: end-to-end DAG integration — the scheme this stage exists for.
// LogTransform -> LorenzoQuant(ABS, log2(1+delta)) gives a point-wise relative
// bound *with* prediction, which no single stage can do.
// ─────────────────────────────────────────────────────────────────────────────

TEST(LogTransform, PipelineIntegrationNoCoder) {
    const size_t nx = 64, ny = 64, n = nx * ny, bytes = n * sizeof(float);
    std::vector<float> h_in(n);
    for (size_t i = 0; i < n; ++i)
        h_in[i] = 1.0f + 0.001f * static_cast<float>(i);

    CudaBuffer<float> d_in(n);
    CudaStream cs;
    d_in.upload(h_in, cs.stream);
    cudaStreamSynchronize(cs.stream);

    const float delta = 1e-3f;

    Pipeline p(bytes, MemoryStrategy::PREALLOCATE, 10.0f);
    p.setDims(nx, ny, 1);

    auto* lg = p.addStage<LogTransformStage<float>>();
    lg->setErrorBound(delta);

    auto* lq = p.addStage<LorenzoQuantStage<float, uint16_t>>();
    lq->setErrorBound(lg->quantizerErrorBound());
    lq->setErrorBoundMode(ErrorBoundMode::ABS);
    lq->setQuantRadius(4096);
    lq->setZigzagCodes(true);
    p.connect(lq, lg, "output");
    p.finalize();


    void* d_comp = nullptr; size_t comp_sz = 0;
    p.compress(d_in.void_ptr(), bytes, &d_comp, &comp_sz, cs.stream);
    cudaStreamSynchronize(cs.stream);

    void* d_dec = nullptr; size_t dec_sz = 0;
    p.decompress(d_comp, comp_sz, &d_dec, &dec_sz, cs.stream);
    cudaStreamSynchronize(cs.stream);

    ASSERT_EQ(dec_sz, bytes);
    std::vector<float> h_out(n);
    cudaMemcpy(h_out.data(), d_dec, bytes, cudaMemcpyDeviceToHost);

    const double slack = 8.0 * static_cast<double>(
        std::numeric_limits<float>::epsilon());
    size_t violations = 0;
    for (size_t i = 0; i < n; ++i) {
        const double x = static_cast<double>(h_in[i]);
        const double rel = std::fabs(x - static_cast<double>(h_out[i])) / std::fabs(x);
        if (rel > static_cast<double>(delta) + slack) ++violations;
    }
    EXPECT_EQ(violations, 0u) << violations << " / " << n << " over the bound";
}
