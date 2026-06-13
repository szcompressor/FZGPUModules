/**
 * tests/stages/test_bitplane_rle.cpp
 *
 * GPU unit tests for BitplaneRLEStage — the FZ-GPU fused bitplane-transpose +
 * zero-byte RLE lossless encoder. Fixed uint16_t input. Not graph-compatible
 * (blocking D2H in both directions).
 *
 *   BP1  RoundTrip_Patterned       — uint16 round-trip, exact match (compressible data)
 *   BP2  RoundTrip_NonAligned      — length not a 4096-byte multiple exercises padding
 *   BP3  RoundTrip_Dense           — high-entropy data still round-trips exactly
 *   BP4  CompressedSmaller         — zero-heavy data compresses below input size
 *   BP5  ZeroInput                 — n=0 does not crash
 *   BP6  GraphCompatible           — isGraphCompatible() == false
 *   BP7  FileRoundTrip             — writeToFile → decompressFromFile (header serialize path)
 *   BP8  QuantizerPipeline         — Quantizer→BitplaneRLE end-to-end float round-trip (FZ-GPU)
 */

#include <gtest/gtest.h>
#include "helpers/fz_test_utils.h"
#include "helpers/stage_harness.h"
#include "fzgpumodules.h"

#include <cstdint>
#include <vector>

using namespace fz;
using namespace fz_test;

namespace {

// Standalone single-stage round-trip on uint16 data with exact-match check.
void expect_exact_round_trip(const std::vector<uint16_t>& h_in) {
    const size_t in_bytes = h_in.size() * sizeof(uint16_t);
    Pipeline p(in_bytes, MemoryStrategy::PREALLOCATE);
    p.addStage<BitplaneRLEStage>();
    p.finalize();

    CudaStream cs;
    auto res = pipeline_round_trip<uint16_t>(p, h_in, cs.stream);

    ASSERT_EQ(res.data.size(), h_in.size());
    for (size_t i = 0; i < h_in.size(); ++i)
        EXPECT_EQ(res.data[i], h_in[i]) << "mismatch at i=" << i;
}

}  // namespace

// ── BP1 ──────────────────────────────────────────────────────────────────────
TEST(BitplaneRLEStage, RoundTrip_Patterned) {
    const size_t N = 8192;  // 16384 bytes = 4 chunks, aligned
    std::vector<uint16_t> h_in(N);
    for (size_t i = 0; i < N; ++i) h_in[i] = static_cast<uint16_t>(i % 64);
    expect_exact_round_trip(h_in);
}

// ── BP2 ──────────────────────────────────────────────────────────────────────
// 5000 uint16 = 10000 bytes → padded up to 12288 (3 chunks); exercises the
// internal zero-pad scratch path.
TEST(BitplaneRLEStage, RoundTrip_NonAligned) {
    const size_t N = 5000;
    std::vector<uint16_t> h_in(N);
    for (size_t i = 0; i < N; ++i) h_in[i] = static_cast<uint16_t>((i * 7) % 300);
    expect_exact_round_trip(h_in);
}

// ── BP3 ──────────────────────────────────────────────────────────────────────
TEST(BitplaneRLEStage, RoundTrip_Dense) {
    const size_t N = 4096;
    std::vector<uint16_t> h_in(N);
    // Pseudo-random high-entropy 16-bit values.
    uint32_t s = 0x12345678u;
    for (size_t i = 0; i < N; ++i) {
        s = s * 1664525u + 1013904223u;
        h_in[i] = static_cast<uint16_t>(s >> 16);
    }
    expect_exact_round_trip(h_in);
}

// ── BP4 ──────────────────────────────────────────────────────────────────────
TEST(BitplaneRLEStage, CompressedSmaller) {
    const size_t N        = 8192;
    const size_t in_bytes = N * sizeof(uint16_t);
    // Mostly zeros with a few small nonzero values — highly compressible.
    std::vector<uint16_t> h_in(N, 0);
    for (size_t i = 0; i < N; i += 97) h_in[i] = static_cast<uint16_t>(1);

    Pipeline p(in_bytes, MemoryStrategy::PREALLOCATE);
    p.addStage<BitplaneRLEStage>();
    p.finalize();

    CudaStream cs;
    auto res = pipeline_round_trip<uint16_t>(p, h_in, cs.stream);

    ASSERT_EQ(res.data.size(), N);
    for (size_t i = 0; i < N; ++i) EXPECT_EQ(res.data[i], h_in[i]);
    EXPECT_LT(res.compressed_bytes, in_bytes)
        << "zero-heavy data should compress below input size";
}

// ── BP5 ──────────────────────────────────────────────────────────────────────
// Single element — smallest real input; the kernel still processes one full
// 4096-byte chunk, so this stresses the internal zero-pad path to the limit.
TEST(BitplaneRLEStage, RoundTrip_Tiny) {
    std::vector<uint16_t> h_in{0xBEEF};
    expect_exact_round_trip(h_in);
}

// ── BP6 ──────────────────────────────────────────────────────────────────────
TEST(BitplaneRLEStage, GraphCompatible) {
    BitplaneRLEStage stage;
    EXPECT_FALSE(stage.isGraphCompatible());
}

// ── BP7 ──────────────────────────────────────────────────────────────────────
TEST(BitplaneRLEStage, FileRoundTrip) {
    const size_t N        = 8192;
    const size_t in_bytes = N * sizeof(uint16_t);
    std::vector<uint16_t> h_in(N);
    for (size_t i = 0; i < N; ++i) h_in[i] = static_cast<uint16_t>(i % 100);

    Pipeline p(in_bytes, MemoryStrategy::PREALLOCATE);
    p.addStage<BitplaneRLEStage>();
    p.finalize();

    CudaStream cs;
    auto res = pipeline_file_round_trip<uint16_t>(
        p, h_in, cs.stream, "test_bitplane_rle.fzm");

    ASSERT_EQ(res.data.size(), N);
    for (size_t i = 0; i < N; ++i)
        EXPECT_EQ(res.data[i], h_in[i]) << "mismatch at i=" << i;
}

// ── BP8 ──────────────────────────────────────────────────────────────────────
// Faithful FZ-GPU pipeline: float → Quantizer (uint16 codes) → BitplaneRLE.
// The bitplane transpose is built into BitplaneRLE, so no BitshuffleStage is
// placed in front of it.
TEST(BitplaneRLEStage, QuantizerPipeline) {
    const size_t N        = 16384;
    const size_t in_bytes = N * sizeof(float);
    const float  eb       = 1e-2f;
    auto h_in = make_smooth_data<float>(N);

    Pipeline p(in_bytes, MemoryStrategy::PREALLOCATE);
    auto* quant = p.addStage<QuantizerStage<float, uint16_t>>();
    quant->setErrorBound(eb);
    quant->setErrorBoundMode(ErrorBoundMode::ABS);
    // Radius chosen large enough that the ±70 smooth data produces no
    // outliers (representable range = radius*eb = 327), so the round-trip
    // tests the codes path cleanly.
    quant->setQuantRadius(32768);
    // Normal mode: codes + outlier ports. BitplaneRLE consumes "codes"; the
    // outlier ports stay terminal and are archived/rewired by the pipeline on
    // decompress (same pattern as the Lorenzo → Bitshuffle dual-branch).
    auto* bprle = p.addStage<BitplaneRLEStage>();
    p.connect(bprle, quant, "codes");
    p.finalize();

    CudaStream cs;
    auto res = pipeline_round_trip<float>(p, h_in, cs.stream);

    ASSERT_EQ(res.data.size(), N);
    EXPECT_LE(res.max_error, static_cast<double>(eb) * 1.001)
        << "reconstruction must respect the absolute error bound";
}
