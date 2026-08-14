/**
 * tests/stages/test_szx.cpp
 *
 * GPU unit tests for SZxStage<T> — ultrafast error-bounded lossy compressor with
 * per-block constant/non-constant classification and fixed-length residual
 * coding (no entropy stage). float/double input, uint8 archive output. Forward
 * (ABS) is graph-capturable; the compressed-size readback is deferred to
 * postStreamSync(). Lossy: reconstruction is checked against the error bound.
 *
 *   SZX1  SmoothRoundTrip_F32     — sine data, |x - x_hat| <= eb
 *   SZX2  RoundTrip_F64           — double round-trip within bound
 *   SZX3  ConstantInput           — all-equal input → constant blocks, exact
 *   SZX4  PartialFinalBlock       — N not a multiple of block_size, within bound
 *   SZX5  PiecewiseConstant       — constant-block path dominates, within bound
 *   SZX6  NOAModeBound            — abs_eb = eb * (max - min) honored
 *   SZX7  SerializeDeserialize    — round-trips through a header-only inverse
 *   SZX8  PortAndTypeContract     — 1 in/out; UINT8 archive / FLOAT32 input
 *   SZX9  GraphCompatFlags        — ABS fwd true, inverse false, NOA fwd false
 *   SZX10 SetBlockSizeRejects     — 0 and >4096 throw
 *   SZX11 CompressionRatio        — smooth data compresses below input
 */

#include <gtest/gtest.h>
#include "helpers/fz_test_utils.h"
#include "helpers/stage_harness.h"
#include "fzgpumodules.h"
#include "fused/szx/szx_stage.h"

#include <cmath>
#include <cstdint>
#include <vector>

using namespace fz;
using namespace fz_test;

namespace {

template<typename T>
double szx_round_trip(const std::vector<T>& in, double eb, uint32_t block,
                      SZxErrorMode mode, double* out_cr = nullptr) {
    Pipeline p(in.size() * sizeof(T), MemoryStrategy::PREALLOCATE);
    auto* s = p.addStage<SZxStage<T>>();
    s->setBlockSize(block);
    s->setErrorBound(eb);
    s->setErrorMode(mode);
    p.finalize();

    CudaStream cs;
    auto res = pipeline_round_trip<T>(p, in, cs.stream);
    if (out_cr) *out_cr = double(in.size() * sizeof(T)) / double(res.compressed_bytes);
    return res.max_error;
}

std::vector<double> to_double(const std::vector<float>& f) {
    return std::vector<double>(f.begin(), f.end());
}

}  // namespace

// ── SZX1 ──────────────────────────────────────────────────────────────────────
TEST(SZxStage, SmoothRoundTrip_F32) {
    constexpr double EB = 1e-2;
    auto in = make_sine_floats(4096, 0.05f, 10.0f);
    EXPECT_LE(szx_round_trip<float>(in, EB, 128, SZxErrorMode::ABS), EB * 1.01);
}

// ── SZX2 ──────────────────────────────────────────────────────────────────────
TEST(SZxStage, RoundTrip_F64) {
    constexpr double EB = 1e-3;
    auto in = to_double(make_sine_floats(4096, 0.03f, 5.0f));
    EXPECT_LE(szx_round_trip<double>(in, EB, 128, SZxErrorMode::ABS), EB * 1.01);
}

// ── SZX3 ──────────────────────────────────────────────────────────────────────
TEST(SZxStage, ConstantInput) {
    constexpr double EB = 1e-2;
    std::vector<float> in(2000, 3.14159f);        // (max-min)=0 -> all constant blocks
    double cr = 0;
    EXPECT_LE(szx_round_trip<float>(in, EB, 128, SZxErrorMode::ABS, &cr), EB * 1.01);
    EXPECT_GT(cr, 10.0) << "constant input should compress hard, got " << cr;
}

// ── SZX4 ──────────────────────────────────────────────────────────────────────
TEST(SZxStage, PartialFinalBlock) {
    constexpr double EB = 1e-2;
    auto in = make_sine_floats(1000 + 67, 0.05f, 8.0f);  // (N % 128) == 67
    EXPECT_LE(szx_round_trip<float>(in, EB, 128, SZxErrorMode::ABS), EB * 1.01);
}

// ── SZX5 ──────────────────────────────────────────────────────────────────────
TEST(SZxStage, PiecewiseConstant) {
    constexpr double EB = 1e-2;
    std::vector<float> in(4096);
    for (size_t i = 0; i < in.size(); ++i) in[i] = float((i / 256) % 5);  // flat regions
    double cr = 0;
    EXPECT_LE(szx_round_trip<float>(in, EB, 128, SZxErrorMode::ABS, &cr), EB * 1.01);
    EXPECT_GT(cr, 4.0) << "piecewise-constant should hit the constant-block path, cr=" << cr;
}

// ── SZX6 ──────────────────────────────────────────────────────────────────────
TEST(SZxStage, NOAModeBound) {
    constexpr double USER_EB = 1e-2;
    auto in = make_sine_floats(4096, 0.05f, 10.0f);
    float vmin = in[0], vmax = in[0];
    for (float v : in) { vmin = std::min(vmin, v); vmax = std::max(vmax, v); }
    const double abs_eb = USER_EB * (double(vmax) - double(vmin));
    EXPECT_LE(szx_round_trip<float>(in, USER_EB, 128, SZxErrorMode::NOA), abs_eb * 1.02);
}

// ── SZX7 ──────────────────────────────────────────────────────────────────────
TEST(SZxStage, SerializeDeserialize) {
    SZxStage<float> fwd;
    fwd.setBlockSize(256);
    fwd.setErrorBound(2.5e-3);
    uint8_t cfg[128] = {};
    size_t n = fwd.serializeHeader(0, cfg, sizeof(cfg));
    ASSERT_GT(n, 0u);

    SZxStage<float> inv;
    inv.setInverse(true);
    inv.deserializeHeader(cfg, n);
    EXPECT_EQ(inv.getBlockSize(), 256u);
}

// ── SZX8 ──────────────────────────────────────────────────────────────────────
TEST(SZxStage, PortAndTypeContract) {
    SZxStage<float> s;
    EXPECT_EQ(s.getNumInputs(), 1u);
    EXPECT_EQ(s.getNumOutputs(), 1u);
    EXPECT_EQ(s.getInputDataType(0),  static_cast<uint8_t>(DataType::FLOAT32));
    EXPECT_EQ(s.getOutputDataType(0), static_cast<uint8_t>(DataType::UINT8));
    s.setInverse(true);
    EXPECT_EQ(s.getInputDataType(0),  static_cast<uint8_t>(DataType::UINT8));
    EXPECT_EQ(s.getOutputDataType(0), static_cast<uint8_t>(DataType::FLOAT32));
}

// ── SZX9 ──────────────────────────────────────────────────────────────────────
TEST(SZxStage, GraphCompatFlags) {
    SZxStage<float> s;                              // ABS forward
    EXPECT_TRUE(s.isGraphCompatible());
    s.setErrorMode(SZxErrorMode::NOA);              // NOA forward: not capturable
    EXPECT_FALSE(s.isGraphCompatible());
    s.setErrorMode(SZxErrorMode::ABS);
    s.setInverse(true);                             // inverse: never capturable
    EXPECT_FALSE(s.isGraphCompatible());
}

// ── SZX10 ─────────────────────────────────────────────────────────────────────
TEST(SZxStage, SetBlockSizeRejects) {
    SZxStage<float> s;
    EXPECT_THROW(s.setBlockSize(0), std::invalid_argument);
    EXPECT_THROW(s.setBlockSize(4097), std::invalid_argument);
    EXPECT_NO_THROW(s.setBlockSize(4096));
}

// ── SZX11 ─────────────────────────────────────────────────────────────────────
TEST(SZxStage, CompressionRatio) {
    constexpr double EB = 1e-2;
    auto in = make_sine_floats(8192, 0.02f, 10.0f);
    double cr = 0;
    EXPECT_LE(szx_round_trip<float>(in, EB, 128, SZxErrorMode::ABS, &cr), EB * 1.01);
    EXPECT_GT(cr, 1.0) << "smooth data should compress below input size, cr=" << cr;
}
