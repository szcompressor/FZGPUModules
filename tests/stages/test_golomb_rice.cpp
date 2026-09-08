/**
 * tests/stages/test_golomb_rice.cpp
 *
 * GPU unit tests for GolombRiceStage<T> — chunk-local Golomb-Rice entropy
 * coder (exact per-chunk k selection, escape-bounded unary run). Lossless.
 *
 *   GR1  ForwardRoundTrip         — mixed positive/negative values round-trip exactly
 *   GR2  AllZerosRoundTrip        — all-zero input round-trips and compresses hard
 *   GR3  LargeOutlierRoundTrip    — one huge value forces the escape path, still exact
 *   GR4  PartialFinalChunk        — element count not a multiple of the chunk round-trips
 *   GR5  MultiChunkRoundTrip      — several full 16 KB chunks round-trip exactly
 *   GR6  SmoothDataCompressesBelowInput — well-predicted data actually shrinks
 *   GR7  Int16RoundTrip           — int16_t instantiation
 *   GR8  HeaderSerialization      — serializeHeader/deserializeHeader preserves config
 *   GR9  NegativeRampRoundTrip    — monotonic negative values (zigzag sign-bit exercise)
 *   GR10 EveryValueAtEscapeBoundary — values that sit exactly at 2^k*kEscapeQ, an
 *                                     off-by-one-prone boundary between coded/escaped
 *   GR11 IncompressibleRandomFallsBackRaw — uniform random int32 must not expand
 *   GR12 IntervalBoundaryPartialChunk — element counts landing exactly at, just
 *                                        before, and just after a restart-interval
 *                                        boundary within a partial final chunk
 */

#include <gtest/gtest.h>
#include <cstdint>
#include <cmath>
#include <random>
#include <vector>

#include "coders/golomb_rice/golomb_rice_stage.h"
#include "helpers/stage_harness.h"

using namespace fz;
using namespace fz_test;

namespace {

template<typename T>
RoundTripResult<T> roundTrip(const std::vector<T>& data, uint32_t chunk_size = 16384) {
    Pipeline p(data.size() * sizeof(T), MemoryStrategy::PREALLOCATE);
    auto* gr = p.addStage<GolombRiceStage<T>>();
    gr->setChunkSize(chunk_size);
    p.finalize();
    CudaStream cs;
    return pipeline_round_trip<T>(p, data, cs.stream);
}

std::vector<int32_t> make_mixed(size_t n) {
    std::vector<int32_t> v(n);
    std::mt19937 rng(42);
    std::normal_distribution<double> d(0.0, 20.0);
    for (size_t i = 0; i < n; ++i) v[i] = static_cast<int32_t>(std::lround(d(rng)));
    return v;
}

}  // namespace

TEST(GolombRiceStage, ForwardRoundTrip) {
    auto data = make_mixed(4096 * 3 + 777);   // multi-chunk + partial tail
    auto res = roundTrip<int32_t>(data);
    ASSERT_EQ(res.data.size(), data.size());
    EXPECT_EQ(res.data, data) << "Golomb-Rice must be lossless";
    EXPECT_EQ(res.max_error, 0.0f);
}

TEST(GolombRiceStage, AllZerosRoundTrip) {
    std::vector<int32_t> data(16384, 0);
    auto res = roundTrip<int32_t>(data);
    EXPECT_EQ(res.data, data);
    EXPECT_LT(res.compressed_bytes, data.size() * sizeof(int32_t) / 4)
        << "all-zero data must compress hard (k=0, every value costs 1 bit)";
}

TEST(GolombRiceStage, LargeOutlierRoundTrip) {
    // Mostly small values (favor small k) with a few huge outliers that must
    // force the escape path (q >= kEscapeQ under any reasonable k).
    std::vector<int32_t> data(4096, 1);
    data[0]    = 1'000'000'000;
    data[2000] = -2'000'000'000;
    data[4095] = 999'999'999;
    auto res = roundTrip<int32_t>(data);
    EXPECT_EQ(res.data, data) << "escape path must reconstruct outliers exactly";
}

TEST(GolombRiceStage, PartialFinalChunk) {
    for (size_t n : {1u, 7u, 4095u, 4097u, 4096u * 2 + 1u}) {
        auto data = make_mixed(n);
        auto res = roundTrip<int32_t>(data);
        ASSERT_EQ(res.data.size(), n) << "n=" << n;
        EXPECT_EQ(res.data, data) << "n=" << n;
    }
}

TEST(GolombRiceStage, MultiChunkRoundTrip) {
    auto data = make_mixed(4096 * 10);
    auto res = roundTrip<int32_t>(data);
    EXPECT_EQ(res.data, data);
}

TEST(GolombRiceStage, SmoothDataCompressesBelowInput) {
    // A gentle ramp with small noise: real Lorenzo-residual-shaped data
    // (mostly near zero, thin tail) — the actual intended input shape.
    std::vector<int32_t> data(4096 * 4);
    std::mt19937 rng(7);
    std::normal_distribution<double> noise(0.0, 3.0);
    for (size_t i = 0; i < data.size(); ++i)
        data[i] = static_cast<int32_t>(std::lround(noise(rng)));
    auto res = roundTrip<int32_t>(data);
    EXPECT_EQ(res.data, data);
    EXPECT_LT(res.compressed_bytes, data.size() * sizeof(int32_t))
        << "small-magnitude residual-shaped data must actually compress";
}

TEST(GolombRiceStage, Int16RoundTrip) {
    std::vector<int16_t> data(8192 * 2 + 55);
    std::mt19937 rng(3);
    std::normal_distribution<double> d(0.0, 50.0);
    for (auto& v : data) v = static_cast<int16_t>(std::lround(d(rng)));
    auto res = roundTrip<int16_t>(data);
    EXPECT_EQ(res.data, data);
}

TEST(GolombRiceStage, HeaderSerialization) {
    GolombRiceStage<int32_t> gr;
    gr.setChunkSize(16384);
    uint8_t buf[5] = {};
    ASSERT_EQ(gr.serializeHeader(0, buf, sizeof(buf)), 5u);
    GolombRiceStage<int32_t> gr2;
    gr2.deserializeHeader(buf, sizeof(buf));
    EXPECT_EQ(gr2.getChunkSize(), 16384u);
    EXPECT_EQ(static_cast<DataType>(buf[4]), GolombRiceStage<int32_t>::getElementDataType());
}

TEST(GolombRiceStage, NegativeRampRoundTrip) {
    std::vector<int32_t> data(4096);
    for (size_t i = 0; i < data.size(); ++i)
        data[i] = -static_cast<int32_t>(i);
    auto res = roundTrip<int32_t>(data);
    EXPECT_EQ(res.data, data);
}

TEST(GolombRiceStage, EveryValueAtEscapeBoundary) {
    // Zigzag(v) chosen so u>>k lands exactly at kEscapeQ for a plausible k,
    // the exact boundary between "coded" and "escaped" in riceCost().
    constexpr uint32_t kEscapeQ = GolombRiceStage<int32_t>::kEscapeQ;
    std::vector<int32_t> data(4096);
    for (size_t i = 0; i < data.size(); ++i) {
        const uint32_t k = static_cast<uint32_t>(i % 8);
        const uint32_t u_at_boundary = kEscapeQ << k;          // q == kEscapeQ exactly
        const int32_t v = static_cast<int32_t>(u_at_boundary >> 1)
                         ^ -static_cast<int32_t>(u_at_boundary & 1u);
        data[i] = v;
    }
    auto res = roundTrip<int32_t>(data);
    EXPECT_EQ(res.data, data) << "off-by-one at the escape boundary must not corrupt data";
}

TEST(GolombRiceStage, IntervalBoundaryPartialChunk) {
    // int32: elemsPerChunk=4096, kIntervalsPerChunk=16 -> elemsPerInterval=256.
    // One full chunk (4096) plus a partial second chunk whose length lands
    // just below / exactly at / just above a restart-interval boundary, and
    // one deep in the middle of an interval -- exercises the decode kernel's
    // iv_start>=n early-return (fully-dead intervals) and iv_n=min(...) clamp
    // (a partially-live interval) added for restart-interval decode.
    constexpr size_t kFullChunk = 4096;
    for (size_t tail : {1u, 255u, 256u, 257u, 4u * 256u + 3u}) {
        const size_t n = kFullChunk + tail;
        auto data = make_mixed(n);
        auto res = roundTrip<int32_t>(data);
        ASSERT_EQ(res.data.size(), n) << "tail=" << tail;
        EXPECT_EQ(res.data, data) << "tail=" << tail;
    }
}

TEST(GolombRiceStage, IncompressibleRandomFallsBackRaw) {
    std::vector<int32_t> data(4096 * 3);
    std::mt19937 rng(99);
    std::uniform_int_distribution<int32_t> d(INT32_MIN, INT32_MAX);
    for (auto& v : data) v = d(rng);
    auto res = roundTrip<int32_t>(data);
    EXPECT_EQ(res.data, data);
    // Raw fallback + header overhead only — must not expand meaningfully.
    EXPECT_LE(res.compressed_bytes, data.size() * sizeof(int32_t) + 4096)
        << "fully incompressible data must fall back to raw, not expand under escape overhead";
}
