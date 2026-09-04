/**
 * tests/stages/test_szp.cpp
 *
 * GPU unit tests for SZpStage<T> — SZp / fZ-light extreme-fast error-bounded
 * compressor: quantize + 1-D Lorenzo delta (block reset) + fixed-length bitpack,
 * no entropy stage. float/double input, uint8 archive output. Forward (ABS) is
 * graph-capturable. Lossy: reconstruction is checked against the error bound.
 *
 *   SZP1  SmoothRoundTrip_F32     — sine data, |x - x_hat| <= eb
 *   SZP2  RoundTrip_F64           — double round-trip within bound
 *   SZP3  ConstantInput           — all-equal input, exact within bound
 *   SZP4  PartialFinalBlock       — N not a multiple of block_size, within bound
 *   SZP5  MonotoneRamp            — block-reset 1-D delta bounds the width, within bound
 *   SZP6  NOAModeBound            — abs_eb = eb * (max - min) honored
 *   SZP7  SerializeDeserialize    — block_size survives the header
 *   SZP8  PortAndTypeContract     — 1 in/out; UINT8 archive / FLOAT32 input
 *   SZP9  GraphCompatFlags        — ABS fwd true, inverse false, NOA fwd false
 *   SZP10 SetBlockSizeRejects     — 0 and >4096 throw
 *   SZP11 CompressionRatio        — smooth data compresses below input
 */

#include <gtest/gtest.h>
#include "helpers/fz_test_utils.h"
#include "helpers/stage_harness.h"
#include "fzgpumodules.h"
// Quarantined experimental reference compressor — direct-include the header.
#include "experimental/reference_compressors/szp/szp_stage.h"

#include <cmath>
#include <cstdint>
#include <vector>

using namespace fz;
using namespace fz_test;

namespace {

template<typename T>
double szp_round_trip(const std::vector<T>& in, double eb, uint32_t block,
                      SZpErrorMode mode, double* out_cr = nullptr) {
    Pipeline p(in.size() * sizeof(T), MemoryStrategy::PREALLOCATE);
    auto* s = p.addStage<SZpStage<T>>();
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

// ── SZP1 ──────────────────────────────────────────────────────────────────────
TEST(SZpStage, SmoothRoundTrip_F32) {
    constexpr double EB = 1e-2;
    auto in = make_sine_floats(4096, 0.05f, 10.0f);
    EXPECT_LE(szp_round_trip<float>(in, EB, 128, SZpErrorMode::ABS), EB * 1.01);
}

// ── SZP2 ──────────────────────────────────────────────────────────────────────
TEST(SZpStage, RoundTrip_F64) {
    constexpr double EB = 1e-3;
    auto in = to_double(make_sine_floats(4096, 0.03f, 5.0f));
    EXPECT_LE(szp_round_trip<double>(in, EB, 128, SZpErrorMode::ABS), EB * 1.01);
}

// ── SZP3 ──────────────────────────────────────────────────────────────────────
TEST(SZpStage, ConstantInput) {
    constexpr double EB = 1e-2;
    std::vector<float> in(2000, 2.71828f);
    double cr = 0;
    EXPECT_LE(szp_round_trip<float>(in, EB, 128, SZpErrorMode::ABS, &cr), EB * 1.01);
    // SZp has NO constant-block escape (that is SZx's classification): every
    // element still pays the block bit width, so constant data lands near
    // 4*8/width, not the extreme ratios SZx reaches. ~3.5x here.
    EXPECT_GT(cr, 3.0) << "constant input SZp ratio, got " << cr;
}

// ── SZP4 ──────────────────────────────────────────────────────────────────────
TEST(SZpStage, PartialFinalBlock) {
    constexpr double EB = 1e-2;
    auto in = make_sine_floats(1000 + 67, 0.05f, 8.0f);  // (N % 128) == 67
    EXPECT_LE(szp_round_trip<float>(in, EB, 128, SZpErrorMode::ABS), EB * 1.01);
}

// ── SZP5 ──────────────────────────────────────────────────────────────────────
TEST(SZpStage, MonotoneRamp) {
    constexpr double EB = 1e-2;
    std::vector<float> in(4096);
    for (size_t i = 0; i < in.size(); ++i) in[i] = 0.5f * float(i);
    double cr = 0;
    EXPECT_LE(szp_round_trip<float>(in, EB, 128, SZpErrorMode::ABS, &cr), EB * 1.01);
    // The 1-D delta resets every block, so each block's first residual is the
    // absolute quantized level at the block head — that large value sets the
    // block width and caps the ratio. Correctness is the bound; CR stays modest.
    EXPECT_GT(cr, 1.5) << "block-reset ramp SZp ratio, got " << cr;
}

// ── SZP6 ──────────────────────────────────────────────────────────────────────
TEST(SZpStage, NOAModeBound) {
    constexpr double USER_EB = 1e-2;
    auto in = make_sine_floats(4096, 0.05f, 10.0f);
    float vmin = in[0], vmax = in[0];
    for (float v : in) { vmin = std::min(vmin, v); vmax = std::max(vmax, v); }
    const double abs_eb = USER_EB * (double(vmax) - double(vmin));
    EXPECT_LE(szp_round_trip<float>(in, USER_EB, 128, SZpErrorMode::NOA), abs_eb * 1.02);
}

// ── SZP7 ──────────────────────────────────────────────────────────────────────
TEST(SZpStage, SerializeDeserialize) {
    SZpStage<float> fwd;
    fwd.setBlockSize(256);
    fwd.setErrorBound(2.5e-3);
    uint8_t cfg[128] = {};
    size_t n = fwd.serializeHeader(0, cfg, sizeof(cfg));
    ASSERT_GT(n, 0u);

    SZpStage<float> inv;
    inv.setInverse(true);
    inv.deserializeHeader(cfg, n);
    EXPECT_EQ(inv.getBlockSize(), 256u);
}

// ── SZP8 ──────────────────────────────────────────────────────────────────────
TEST(SZpStage, PortAndTypeContract) {
    SZpStage<float> s;
    EXPECT_EQ(s.getNumInputs(), 1u);
    EXPECT_EQ(s.getNumOutputs(), 1u);
    EXPECT_EQ(s.getInputDataType(0),  static_cast<uint8_t>(DataType::FLOAT32));
    EXPECT_EQ(s.getOutputDataType(0), static_cast<uint8_t>(DataType::UINT8));
    s.setInverse(true);
    EXPECT_EQ(s.getInputDataType(0),  static_cast<uint8_t>(DataType::UINT8));
    EXPECT_EQ(s.getOutputDataType(0), static_cast<uint8_t>(DataType::FLOAT32));
}

// ── SZP9 ──────────────────────────────────────────────────────────────────────
TEST(SZpStage, GraphCompatFlags) {
    SZpStage<float> s;
    EXPECT_TRUE(s.isGraphCompatible());
    s.setErrorMode(SZpErrorMode::NOA);
    EXPECT_FALSE(s.isGraphCompatible());
    s.setErrorMode(SZpErrorMode::ABS);
    s.setInverse(true);
    EXPECT_FALSE(s.isGraphCompatible());
}

// ── SZP10 ─────────────────────────────────────────────────────────────────────
TEST(SZpStage, SetBlockSizeRejects) {
    SZpStage<float> s;
    EXPECT_THROW(s.setBlockSize(0), std::invalid_argument);
    EXPECT_THROW(s.setBlockSize(4097), std::invalid_argument);
    EXPECT_NO_THROW(s.setBlockSize(4096));
}

// ── SZP11 ─────────────────────────────────────────────────────────────────────
TEST(SZpStage, CompressionRatio) {
    constexpr double EB = 1e-2;
    auto in = make_sine_floats(8192, 0.02f, 10.0f);
    double cr = 0;
    EXPECT_LE(szp_round_trip<float>(in, EB, 128, SZpErrorMode::ABS, &cr), EB * 1.01);
    EXPECT_GT(cr, 1.0) << "smooth data should compress below input size, cr=" << cr;
}
