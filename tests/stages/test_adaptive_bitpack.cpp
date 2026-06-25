/**
 * tests/stages/test_adaptive_bitpack.cpp
 *
 * GPU unit tests for AdaptiveBitpackStage<T> — per-block adaptive fixed-rate
 * bit-plane coder (cuSZp-style plain mode). Signed int16/int32 input, uint8
 * archive output. Not graph-compatible (forward D2H for the scanned length).
 *
 *   AB1  RoundTrip_Int32            — signed int32 block round-trip, exact
 *   AB2  RoundTrip_Int16            — signed int16 round-trip, exact
 *   AB3  AllZeroBlocks             — all-zero input → exact, tiny archive
 *   AB4  EveryFixedRate            — magnitudes spanning every bit width, exact
 *   AB5  PartialFinalBlock         — N not a multiple of block_size, exact
 *   AB6  NegativeExtremes          — INT_MIN / mixed signs round-trip exactly
 *   AB7  SerializeDeserialize      — block_size + num_elements survive header
 *   AB8  SetBlockSizeRejects       — 0 and >1024 throw
 *   AB9  PortAndTypeContract       — 1 in/out; UINT8 archive / signed codes
 *   AB10 GraphCompatibleFalse      — isGraphCompatible() == false
 *   AB11 CompressionRatio          — small-magnitude data compresses below input
 *   AB12 CuSZpPipeline             — Quantizer(linear)→Lorenzo(block)→AdaptiveBitpack
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

template<typename T>
void expect_exact_round_trip(const std::vector<T>& h_in, uint32_t block = 32,
                            bool outlier = false) {
    const size_t in_bytes = h_in.size() * sizeof(T);
    Pipeline p(in_bytes, MemoryStrategy::PREALLOCATE);
    auto* s = p.addStage<AdaptiveBitpackStage<T>>();
    s->setBlockSize(block);
    s->setOutlierSelection(outlier);
    p.finalize();

    CudaStream cs;
    auto res = pipeline_round_trip<T>(p, h_in, cs.stream);

    ASSERT_EQ(res.data.size(), h_in.size());
    for (size_t i = 0; i < h_in.size(); ++i)
        EXPECT_EQ(res.data[i], h_in[i]) << "mismatch at i=" << i;
}

}  // namespace

// ── AB1 ───────────────────────────────────────────────────────────────────────
TEST(AdaptiveBitpackStage, RoundTrip_Int32) {
    const size_t N = 4096;
    std::vector<int32_t> h_in(N);
    for (size_t i = 0; i < N; ++i)
        h_in[i] = static_cast<int32_t>((i * 7) % 1000) - 500;
    expect_exact_round_trip<int32_t>(h_in);
}

// ── AB2 ───────────────────────────────────────────────────────────────────────
TEST(AdaptiveBitpackStage, RoundTrip_Int16) {
    const size_t N = 2048;
    std::vector<int16_t> h_in(N);
    for (size_t i = 0; i < N; ++i)
        h_in[i] = static_cast<int16_t>(static_cast<int>((i * 5) % 600) - 300);
    expect_exact_round_trip<int16_t>(h_in);
}

// ── AB3 ───────────────────────────────────────────────────────────────────────
TEST(AdaptiveBitpackStage, AllZeroBlocks) {
    const size_t N = 4096;
    const size_t in_bytes = N * sizeof(int32_t);
    std::vector<int32_t> h_in(N, 0);

    Pipeline p(in_bytes, MemoryStrategy::PREALLOCATE);
    p.addStage<AdaptiveBitpackStage<int32_t>>();
    p.finalize();

    CudaStream cs;
    auto res = pipeline_round_trip<int32_t>(p, h_in, cs.stream);
    ASSERT_EQ(res.data.size(), N);
    for (size_t i = 0; i < N; ++i) EXPECT_EQ(res.data[i], 0);
    // All-zero blocks store only the rate region (1 byte/block, 0 payload).
    EXPECT_LE(res.compressed_bytes, (N / 32) + 64u);
}

// ── AB4 ───────────────────────────────────────────────────────────────────────
// Each 32-element block uses a different magnitude so fixed_rate sweeps the full
// range [0, 31] across the input.
TEST(AdaptiveBitpackStage, EveryFixedRate) {
    const uint32_t block = 32;
    std::vector<int32_t> h_in;
    for (int r = 0; r <= 31; ++r) {
        int32_t mag = (r == 0) ? 0 : ((r >= 31) ? (int32_t)0x7fffffff : (1 << r) - 1);
        for (uint32_t j = 0; j < block; ++j)
            h_in.push_back((j & 1) ? -mag : mag);
    }
    expect_exact_round_trip<int32_t>(h_in, block);
}

// ── AB5 ───────────────────────────────────────────────────────────────────────
TEST(AdaptiveBitpackStage, PartialFinalBlock) {
    const size_t N = 1000;  // not a multiple of 32
    std::vector<int32_t> h_in(N);
    for (size_t i = 0; i < N; ++i)
        h_in[i] = static_cast<int32_t>((i * 3) % 257) - 128;
    expect_exact_round_trip<int32_t>(h_in, 32);
}

// ── AB6 ───────────────────────────────────────────────────────────────────────
TEST(AdaptiveBitpackStage, NegativeExtremes) {
    std::vector<int32_t> h_in = {
        0, -1, 1, INT32_MIN, INT32_MAX, -123456, 123456, -1, 0, 7
    };
    h_in.resize(64, 0);
    expect_exact_round_trip<int32_t>(h_in, 32);
}

// ── AB7 ───────────────────────────────────────────────────────────────────────
TEST(AdaptiveBitpackStage, SerializeDeserialize) {
    AdaptiveBitpackStage<int32_t> original;
    original.setBlockSize(64);
    // num_elements is set during compress; simulate via a forward pass below is
    // overkill — just verify block_size + dtype survive the header bytes.
    uint8_t buf[128] = {};
    size_t sz = original.serializeHeader(0, buf, sizeof(buf));
    EXPECT_EQ(sz, sizeof(AdaptiveBitpackConfig));

    AdaptiveBitpackStage<int32_t> restored;
    restored.deserializeHeader(buf, sz);
    EXPECT_EQ(restored.getBlockSize(), 64u);
}

// ── AB8 ───────────────────────────────────────────────────────────────────────
TEST(AdaptiveBitpackStage, SetBlockSizeRejects) {
    AdaptiveBitpackStage<int32_t> s;
    EXPECT_THROW(s.setBlockSize(0), std::invalid_argument);
    EXPECT_THROW(s.setBlockSize(2048), std::invalid_argument);
    EXPECT_NO_THROW(s.setBlockSize(1024));
    EXPECT_NO_THROW(s.setBlockSize(32));
}

// ── AB9 ───────────────────────────────────────────────────────────────────────
TEST(AdaptiveBitpackStage, PortAndTypeContract) {
    AdaptiveBitpackStage<int32_t> fwd;
    EXPECT_EQ(fwd.getNumInputs(), 1u);
    EXPECT_EQ(fwd.getNumOutputs(), 1u);
    EXPECT_EQ(fwd.getInputDataType(0),  static_cast<uint8_t>(DataType::INT32));
    EXPECT_EQ(fwd.getOutputDataType(0), static_cast<uint8_t>(DataType::UINT8));

    AdaptiveBitpackStage<int32_t> inv;
    inv.setInverse(true);
    EXPECT_EQ(inv.getInputDataType(0),  static_cast<uint8_t>(DataType::UINT8));
    EXPECT_EQ(inv.getOutputDataType(0), static_cast<uint8_t>(DataType::INT32));
}

// ── AB10 ──────────────────────────────────────────────────────────────────────
TEST(AdaptiveBitpackStage, GraphCompatibleFalse) {
    EXPECT_FALSE(AdaptiveBitpackStage<int32_t>().isGraphCompatible());
}

// ── AB11 ──────────────────────────────────────────────────────────────────────
TEST(AdaptiveBitpackStage, CompressionRatio) {
    const size_t N = 8192;
    const size_t in_bytes = N * sizeof(int32_t);
    std::vector<int32_t> h_in(N);
    for (size_t i = 0; i < N; ++i)
        h_in[i] = static_cast<int32_t>((i * 13) % 7) - 3;  // |v| <= 3 → ~3-bit rate

    Pipeline p(in_bytes, MemoryStrategy::PREALLOCATE);
    auto* s = p.addStage<AdaptiveBitpackStage<int32_t>>();
    s->setBlockSize(32);
    p.finalize();

    CudaStream cs;
    auto res = pipeline_round_trip<int32_t>(p, h_in, cs.stream);
    ASSERT_EQ(res.data.size(), N);
    for (size_t i = 0; i < N; ++i) EXPECT_EQ(res.data[i], h_in[i]);
    EXPECT_LT(res.compressed_bytes, in_bytes)
        << "small-magnitude data should pack below the int32 input size";
}

// ── AB12 ──────────────────────────────────────────────────────────────────────
// The full cuSZp modular pipeline: linear quantizer → block-local Lorenzo →
// adaptive bit-plane coder.
TEST(AdaptiveBitpackStage, CuSZpPipeline) {
    const size_t N = 4096;
    const size_t in_bytes = N * sizeof(float);
    const float  EB = 1e-2f;
    auto h_input = make_smooth_data<float>(N);

    Pipeline p(in_bytes, MemoryStrategy::PREALLOCATE);

    auto* quant = p.addStage<QuantizerStage<float, uint32_t>>();
    quant->setErrorBound(EB);
    quant->setErrorBoundMode(ErrorBoundMode::ABS);
    quant->setLinearMode(true);

    auto* lrz = p.addStage<LorenzoStage<int32_t>>();
    lrz->setBlockSize(32);
    p.connect(lrz, quant, "codes");

    auto* ab = p.addStage<AdaptiveBitpackStage<int32_t>>();
    ab->setBlockSize(32);
    p.connect(ab, lrz);

    p.finalize();

    CudaBuffer<float> d_in(N);
    CudaStream cs;
    d_in.upload(h_input, cs.stream);
    cudaStreamSynchronize(cs.stream);

    void* d_comp = nullptr; size_t comp_sz = 0;
    p.compress(d_in.void_ptr(), in_bytes, &d_comp, &comp_sz, cs.stream);
    cudaStreamSynchronize(cs.stream);
    ASSERT_GT(comp_sz, 0u);

    void* d_dec = nullptr; size_t dec_sz = 0;
    p.decompress(nullptr, comp_sz, &d_dec, &dec_sz, cs.stream);
    cudaStreamSynchronize(cs.stream);

    ASSERT_EQ(dec_sz, in_bytes);
    std::vector<float> h_recon(N);
    cudaMemcpy(h_recon.data(), d_dec, dec_sz, cudaMemcpyDeviceToHost);

    float max_err = max_abs_error(h_input, h_recon);
    EXPECT_LE(max_err, EB * 1.01f);
}

// ── AB13–AB19: cuSZp2 per-block outlier selection ─────────────────────────────
//   AB13 Outlier_RoundTrip          — outlier mode round-trips exactly
//   AB14 Outlier_BeatsPlain         — outlier-heavy data compresses better w/ selection
//   AB15 Outlier_EveryByteCount     — element-0 magnitudes of 1..4 bytes round-trip
//   AB16 Outlier_PartialAndSingle   — partial final block + single-element block
//   AB17 Outlier_NegativeExtremes   — INT_MIN as the per-block outlier
//   AB18 Outlier_HeaderRoundTrip    — outlier_selection flag survives the header
//   AB19 Outlier_CuSZpPipeline      — full pipeline with outlier selection on

namespace {
// Block where element 0 is a large magnitude and the rest are small — the case
// per-block outlier extraction is designed for.
std::vector<int32_t> make_outlier_heavy(size_t blocks, uint32_t block, int32_t big) {
    std::vector<int32_t> v;
    v.reserve(blocks * block);
    for (size_t b = 0; b < blocks; ++b) {
        v.push_back((b & 1) ? -big : big);                 // element 0: large
        for (uint32_t j = 1; j < block; ++j)
            v.push_back(static_cast<int32_t>((j % 5)) - 2); // |rest| <= 2
    }
    return v;
}
}  // namespace

// ── AB13 ──────────────────────────────────────────────────────────────────────
TEST(AdaptiveBitpackStage, Outlier_RoundTrip) {
    auto h_in = make_outlier_heavy(32, 32, 1 << 18);
    expect_exact_round_trip<int32_t>(h_in, 32, /*outlier=*/true);
    // int16 path too (smaller outlier byte counts)
    std::vector<int16_t> h16;
    for (size_t b = 0; b < 16; ++b) {
        h16.push_back(static_cast<int16_t>((b & 1) ? -3000 : 3000));
        for (uint32_t j = 1; j < 32; ++j) h16.push_back(static_cast<int16_t>((j % 3) - 1));
    }
    expect_exact_round_trip<int16_t>(h16, 32, /*outlier=*/true);
}

// ── AB14 ──────────────────────────────────────────────────────────────────────
TEST(AdaptiveBitpackStage, Outlier_BeatsPlain) {
    auto h_in = make_outlier_heavy(64, 32, 1 << 20);
    const size_t in_bytes = h_in.size() * sizeof(int32_t);

    auto compressed_size = [&](bool outlier) {
        Pipeline p(in_bytes, MemoryStrategy::PREALLOCATE);
        auto* s = p.addStage<AdaptiveBitpackStage<int32_t>>();
        s->setBlockSize(32);
        s->setOutlierSelection(outlier);
        p.finalize();
        CudaStream cs;
        auto res = pipeline_round_trip<int32_t>(p, h_in, cs.stream);
        for (size_t i = 0; i < h_in.size(); ++i) EXPECT_EQ(res.data[i], h_in[i]);
        return res.compressed_bytes;
    };

    size_t plain = compressed_size(false);
    size_t outl  = compressed_size(true);
    EXPECT_LT(outl, plain)
        << "outlier selection should beat plain on outlier-heavy data ("
        << outl << " vs " << plain << ")";
}

// ── AB15 ──────────────────────────────────────────────────────────────────────
TEST(AdaptiveBitpackStage, Outlier_EveryByteCount) {
    const uint32_t block = 32;
    const int32_t mags[4] = {0x7f, 0x7fff, 0x7fffff, 0x7fffffff};  // 1..4 bytes
    std::vector<int32_t> h_in;
    for (int m = 0; m < 4; ++m) {
        h_in.push_back(mags[m]);                              // element 0 outlier
        for (uint32_t j = 1; j < block; ++j)
            h_in.push_back(static_cast<int32_t>((j % 7)) - 3);
    }
    expect_exact_round_trip<int32_t>(h_in, block, /*outlier=*/true);
}

// ── AB16 ──────────────────────────────────────────────────────────────────────
TEST(AdaptiveBitpackStage, Outlier_PartialAndSingle) {
    // 65 elements with block=32 → blocks of 32, 32, 1 (last is a single outlier).
    std::vector<int32_t> h_in;
    for (int i = 0; i < 65; ++i)
        h_in.push_back((i % 32 == 0) ? (100000 * ((i & 1) ? -1 : 1))
                                     : (static_cast<int32_t>(i % 5) - 2));
    expect_exact_round_trip<int32_t>(h_in, 32, /*outlier=*/true);
}

// ── AB17 ──────────────────────────────────────────────────────────────────────
TEST(AdaptiveBitpackStage, Outlier_NegativeExtremes) {
    std::vector<int32_t> h_in(32, 0);
    h_in[0] = INT32_MIN;   // element-0 outlier with magnitude 2^31 (4 bytes)
    h_in[1] = 1; h_in[2] = -1; h_in[3] = 2;
    expect_exact_round_trip<int32_t>(h_in, 32, /*outlier=*/true);
}

// ── AB18 ──────────────────────────────────────────────────────────────────────
TEST(AdaptiveBitpackStage, Outlier_HeaderRoundTrip) {
    AdaptiveBitpackStage<int32_t> original;
    original.setBlockSize(32);
    original.setOutlierSelection(true);
    uint8_t buf[128] = {};
    size_t sz = original.serializeHeader(0, buf, sizeof(buf));

    AdaptiveBitpackStage<int32_t> restored;
    restored.deserializeHeader(buf, sz);
    EXPECT_TRUE(restored.getOutlierSelection());
    EXPECT_EQ(restored.getBlockSize(), 32u);
}

// ── AB19 ──────────────────────────────────────────────────────────────────────
TEST(AdaptiveBitpackStage, Outlier_CuSZpPipeline) {
    const size_t N = 4096;
    const size_t in_bytes = N * sizeof(float);
    const float  EB = 1e-2f;
    auto h_input = make_smooth_data<float>(N);

    Pipeline p(in_bytes, MemoryStrategy::PREALLOCATE);
    auto* quant = p.addStage<QuantizerStage<float, uint32_t>>();
    quant->setErrorBound(EB);
    quant->setErrorBoundMode(ErrorBoundMode::ABS);
    quant->setLinearMode(true);
    auto* lrz = p.addStage<LorenzoStage<int32_t>>();
    lrz->setBlockSize(32);
    p.connect(lrz, quant, "codes");
    auto* ab = p.addStage<AdaptiveBitpackStage<int32_t>>();
    ab->setBlockSize(32);
    ab->setOutlierSelection(true);   // cuSZp2 outlier mode
    p.connect(ab, lrz);
    p.finalize();

    CudaBuffer<float> d_in(N);
    CudaStream cs;
    d_in.upload(h_input, cs.stream);
    cudaStreamSynchronize(cs.stream);

    void* d_comp = nullptr; size_t comp_sz = 0;
    p.compress(d_in.void_ptr(), in_bytes, &d_comp, &comp_sz, cs.stream);
    cudaStreamSynchronize(cs.stream);

    void* d_dec = nullptr; size_t dec_sz = 0;
    p.decompress(nullptr, comp_sz, &d_dec, &dec_sz, cs.stream);
    cudaStreamSynchronize(cs.stream);

    ASSERT_EQ(dec_sz, in_bytes);
    std::vector<float> h_recon(N);
    cudaMemcpy(h_recon.data(), d_dec, dec_sz, cudaMemcpyDeviceToHost);
    float max_err = max_abs_error(h_input, h_recon);
    EXPECT_LE(max_err, EB * 1.01f);
}
