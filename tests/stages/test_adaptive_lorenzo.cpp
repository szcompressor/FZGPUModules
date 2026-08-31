/**
 * tests/stages/test_adaptive_lorenzo.cpp
 *
 * Unit tests for AdaptiveLorenzoStage<T> — per-tile adaptive multi-order Lorenzo
 * with centering (FSZ prediction stage). Lossless in every mode.
 *
 *   AL1  ForwardRoundTrip        — round-trip on mixed data, all variants enabled
 *   AL2  ZeroInput               — all-zero input round-trips and stays cheap
 *   AL3  PartialFinalTile        — element count not a multiple of the tile
 *   AL4  SerializeDeserialize    — config survives the FZM header
 *   AL5  PipelineIntegration     — Quantizer -> AdaptiveLorenzo -> AdaptiveBitpack
 *   AL6  SelectsCenteringOnOffset— a large constant offset must pick centering
 *   AL7  SelectsOrder2OnRamp     — a pure linear ramp must pick LZ2
 *   AL8  SelectsPlainOnSparse    — mostly-zero data must not pay for a mean
 *   AL9  NeverWorseThanFixed     — adaptive >= each fixed variant, per dataset
 *   AL10 TileSizes               — every legal blocks_per_tile round-trips
 *   AL11 RejectsBadConfig        — coder_block_size != 32, blocks_per_tile > 32
 *   AL12 Int16RoundTrip          — int16_t instantiation
 *   AL13 MeansCompactedToCenteredTiles — no centering chosen => means port empty
 *   AL14 MeansStoredWhenEveryTileCenters — all centered => one mean per tile
 *   AL15 MixedCompactionRoundTrip — inverse rebuilds offsets from packed modes
 *   AL16 CompactionSurvivesTileCountNotMultipleOfFour — ragged mode byte
 *   AL17 BoundPlainOracleMatchesLegacy — explicit bound/fallback byte parity
 *   AL18 DownstreamOracleChangesSelection — outlier policy changes chosen modes
 *   AL19 OutlierOracleNeverWorseThanFixed — exact second-policy end-to-end gate
 */

#include <gtest/gtest.h>
#include <cstdint>
#include <cmath>
#include <vector>

#include "fused/adaptive_lorenzo/adaptive_lorenzo_stage.h"
#include "coders/adaptive_bitpack/adaptive_bitpack_stage.h"
#include "quantizers/quantizer/quantizer.h"
#include "helpers/stage_harness.h"

using namespace fz;
using namespace fz_test;

namespace {

// Mixed structure: a smooth ramp region, a constant-offset region, and a sparse
// region, so a per-tile chooser has something to actually choose between.
std::vector<int32_t> make_mixed(size_t n) {
    std::vector<int32_t> v(n);
    for (size_t i = 0; i < n; ++i) {
        const size_t region = (i / 256) % 3;
        if (region == 0)      v[i] = 1000 + 3 * static_cast<int32_t>(i % 256);
        else if (region == 1) v[i] = 100000 + static_cast<int32_t>((i * 7) % 11);
        else                  v[i] = ((i % 37) == 0) ? 5 : 0;
    }
    return v;
}

size_t compressed_size(const std::vector<int32_t>& data,
                       bool order2, bool centering, uint32_t bpt,
                       cudaStream_t stream, bool outlier = false) {
    AdaptiveLorenzoStage<int32_t>::Config c;
    c.blocks_per_tile  = bpt;
    c.enable_order2    = order2;
    c.enable_centering = centering;

    Pipeline p(data.size() * sizeof(int32_t), MemoryStrategy::PREALLOCATE);
    auto* al = p.addStage<AdaptiveLorenzoStage<int32_t>>(c);
    auto* ab = p.addStage<AdaptiveBitpackStage<int32_t>>();
    ab->setBlockSize(32);
    ab->setOutlierSelection(outlier);
    p.connect(ab, al);
    p.finalize();

    auto res = pipeline_round_trip<int32_t>(p, data, stream);
    EXPECT_EQ(res.max_error, 0.0f) << "AdaptiveLorenzo must be lossless";
    return res.compressed_bytes;
}

}  // namespace

TEST(AdaptiveLorenzoStage, ForwardRoundTrip) {
    const size_t N = 4096;
    auto h_input = make_mixed(N);

    Pipeline p(N * sizeof(int32_t), MemoryStrategy::PREALLOCATE);
    p.addStage<AdaptiveLorenzoStage<int32_t>>();
    p.finalize();

    CudaStream cs;
    auto res = pipeline_round_trip<int32_t>(p, h_input, cs.stream);
    EXPECT_EQ(res.max_error, 0.0f);
}

TEST(AdaptiveLorenzoStage, ZeroInput) {
    const size_t N = 2048;
    std::vector<int32_t> h_input(N, 0);

    Pipeline p(N * sizeof(int32_t), MemoryStrategy::PREALLOCATE);
    p.addStage<AdaptiveLorenzoStage<int32_t>>();
    p.finalize();

    CudaStream cs;
    auto res = pipeline_round_trip<int32_t>(p, h_input, cs.stream);
    EXPECT_EQ(res.max_error, 0.0f);
}

TEST(AdaptiveLorenzoStage, PartialFinalTile) {
    // 1000 is neither a multiple of the 256-element tile nor of the 32-element
    // coder block, so both the mean's live count and the tail guard matter.
    const size_t N = 1000;
    auto h_input = make_mixed(N);

    Pipeline p(N * sizeof(int32_t), MemoryStrategy::PREALLOCATE);
    p.addStage<AdaptiveLorenzoStage<int32_t>>();
    p.finalize();

    CudaStream cs;
    auto res = pipeline_round_trip<int32_t>(p, h_input, cs.stream);
    EXPECT_EQ(res.max_error, 0.0f);
}

TEST(AdaptiveLorenzoStage, SerializeDeserialize) {
    AdaptiveLorenzoStage<int32_t>::Config c;
    c.blocks_per_tile  = 16;
    c.enable_order2    = false;
    c.enable_centering = true;
    AdaptiveLorenzoStage<int32_t> original(c);

    uint8_t buf[128] = {};
    size_t written = original.serializeHeader(0, buf, sizeof(buf));
    EXPECT_EQ(written, sizeof(AdaptiveLorenzoConfig));

    AdaptiveLorenzoStage<int32_t> restored;
    restored.deserializeHeader(buf, written);
    EXPECT_EQ(restored.getTileSize(), 16u * 32u);
}

TEST(AdaptiveLorenzoStage, PipelineIntegration) {
    const size_t N = 8192;
    auto h_float = make_smooth_data<float>(N);

    Pipeline p(N * sizeof(float), MemoryStrategy::PREALLOCATE);
    auto* q = p.addStage<QuantizerStage<float, uint32_t>>();
    q->setErrorBound(1e-3f);
    q->setErrorBoundMode(ErrorBoundMode::ABS);
    q->setLinearMode(true);
    auto* al = p.addStage<AdaptiveLorenzoStage<int32_t>>();
    auto* ab = p.addStage<AdaptiveBitpackStage<int32_t>>();
    ab->setBlockSize(32);
    p.connect(al, q, "codes");
    p.connect(ab, al);
    p.finalize();

    CudaStream cs;
    auto res = pipeline_round_trip<float>(p, h_float, cs.stream);
    // 1% slack matches the equivalent cuSZp-front-end test in test_lorenzo.cpp:
    // make_smooth_data peaks near 70, where a float32 ulp is ~0.8% of a 1e-3
    // bound. The prediction stage itself is integer-lossless (see the round-trip
    // tests above, which assert exactly zero error).
    EXPECT_LE(res.max_error, 1e-3 * 1.01);
}

TEST(AdaptiveLorenzoStage, BoundPlainOracleMatchesLegacy) {
    const size_t N = 8192 + 37;  // exercise a partial coder block and tile
    const auto input = make_mixed(N);
    const size_t bytes = input.size() * sizeof(int32_t);

    int32_t* d_input = nullptr;
    ASSERT_EQ(cudaMalloc(&d_input, bytes), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice), cudaSuccess);

    auto encode = [&](bool bind) {
        Pipeline p(bytes, MemoryStrategy::PREALLOCATE);
        auto* al = p.addStage<AdaptiveLorenzoStage<int32_t>>();
        if (bind) {
            AdaptiveBitpackStage<int32_t> plain;
            plain.setBlockSize(32);
            EXPECT_TRUE(al->bindDownstreamEncodingOracle(plain.getEncodingOracle()));
        }
        p.finalize();
        EXPECT_EQ(al->hasBoundEncodingOracle(), bind);

        void* d_archive = nullptr;
        size_t archive_bytes = 0;
        p.compress(d_input, bytes, &d_archive, &archive_bytes, 0);
        EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        std::vector<uint8_t> archive(archive_bytes);
        EXPECT_EQ(cudaMemcpy(archive.data(), d_archive, archive_bytes,
                             cudaMemcpyDeviceToHost), cudaSuccess);
        return archive;
    };

    const auto legacy = encode(false);
    const auto bound  = encode(true);
    EXPECT_EQ(bound, legacy)
        << "binding the exact plain policy changed residuals, modes, or compacted means";

    EXPECT_EQ(cudaFree(d_input), cudaSuccess);
}

TEST(AdaptiveLorenzoStage, DownstreamOracleChangesSelection) {
    // Plain fixed-rate strongly rewards centering a large tile offset. The
    // outlier policy can instead store each coder block's element 0 compactly,
    // so the exact downstream policy is capable of changing the selected mode.
    const size_t N = 4096;
    std::vector<int32_t> input(N);
    for (size_t i = 0; i < N; ++i)
        input[i] = 500000 + static_cast<int32_t>((i * 13) % 17);
    const size_t bytes = input.size() * sizeof(int32_t);

    int32_t* d_input = nullptr;
    ASSERT_EQ(cudaMalloc(&d_input, bytes), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice), cudaSuccess);

    auto encode = [&](bool outlier) {
        Pipeline p(bytes, MemoryStrategy::PREALLOCATE);
        auto* al = p.addStage<AdaptiveLorenzoStage<int32_t>>();
        AdaptiveBitpackStage<int32_t> coder;
        coder.setBlockSize(32);
        coder.setOutlierSelection(outlier);
        EXPECT_TRUE(al->bindDownstreamEncodingOracle(coder.getEncodingOracle()));
        p.finalize();

        void* d_archive = nullptr;
        size_t archive_bytes = 0;
        p.compress(d_input, bytes, &d_archive, &archive_bytes, 0);
        EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        std::vector<uint8_t> archive(archive_bytes);
        EXPECT_EQ(cudaMemcpy(archive.data(), d_archive, archive_bytes,
                             cudaMemcpyDeviceToHost), cudaSuccess);
        return archive;
    };

    const auto plain    = encode(false);
    const auto adaptive = encode(true);
    EXPECT_NE(adaptive, plain)
        << "the second policy witness did not affect any selected tile mode";

    EXPECT_EQ(cudaFree(d_input), cudaSuccess);
}

TEST(AdaptiveLorenzoStage, OutlierOracleNeverWorseThanFixed) {
    const size_t N = 8192 + 37;
    auto input = make_mixed(N);
    for (auto& v : input) v += 500000;  // exercise element-0 outlier decisions

    CudaStream cs;
    const size_t all   = compressed_size(input, true,  true,  8, cs.stream, true);
    const size_t lz1c  = compressed_size(input, false, true,  8, cs.stream, true);
    const size_t lz12  = compressed_size(input, true,  false, 8, cs.stream, true);
    const size_t plain = compressed_size(input, false, false, 8, cs.stream, true);
    EXPECT_LE(all, lz1c);
    EXPECT_LE(all, lz12);
    EXPECT_LE(all, plain);
}

TEST(AdaptiveLorenzoStage, SelectsCenteringOnOffset) {
    // Every tile carries a large constant offset, so centering must win: the
    // adaptive stage should match what forcing centering on would give, and beat
    // having it unavailable.
    const size_t N = 4096;
    std::vector<int32_t> v(N);
    for (size_t i = 0; i < N; ++i) v[i] = 500000 + static_cast<int32_t>((i * 13) % 17);

    CudaStream cs;
    const size_t adaptive = compressed_size(v, true, true,  8, cs.stream);
    const size_t no_cent  = compressed_size(v, true, false, 8, cs.stream);
    EXPECT_LT(adaptive, no_cent)
        << "centering must be selected on constant-offset data";
}

TEST(AdaptiveLorenzoStage, SelectsOrder2OnRamp) {
    // A pure linear ramp: LZ2 drives every residual past the seeds to zero.
    const size_t N = 4096;
    std::vector<int32_t> v(N);
    for (size_t i = 0; i < N; ++i) v[i] = 1000 + 3 * static_cast<int32_t>(i);

    CudaStream cs;
    const size_t adaptive = compressed_size(v, true,  true, 8, cs.stream);
    const size_t no_lz2   = compressed_size(v, false, true, 8, cs.stream);
    EXPECT_LT(adaptive, no_lz2) << "LZ2 must be selected on a linear ramp";
}

TEST(AdaptiveLorenzoStage, SelectsPlainOnSparse) {
    // Mostly zeros: blocks already encode to nothing, so a stored mean is pure
    // overhead. The chooser must decline centering rather than always take it —
    // this is the case that regresses when centering is unconditional.
    const size_t N = 8192;
    std::vector<int32_t> v(N, 0);
    for (size_t i = 0; i < N; i += 512) v[i] = 3;

    CudaStream cs;
    const size_t adaptive = compressed_size(v, true, true,  8, cs.stream);
    const size_t no_cent  = compressed_size(v, true, false, 8, cs.stream);
    EXPECT_LE(adaptive, no_cent)
        << "adaptive must not pay for centering on sparse data";
}

TEST(AdaptiveLorenzoStage, NeverWorseThanFixed) {
    // The selection is by exact encoded cost, so enabling a variant can only
    // ever help: each variant is in the candidate set of the fuller one.
    const size_t N = 8192;
    auto v = make_mixed(N);

    CudaStream cs;
    const size_t all   = compressed_size(v, true,  true,  8, cs.stream);
    const size_t lz1c  = compressed_size(v, false, true,  8, cs.stream);
    const size_t lz12  = compressed_size(v, true,  false, 8, cs.stream);
    const size_t plain = compressed_size(v, false, false, 8, cs.stream);

    EXPECT_LE(all, lz1c);
    EXPECT_LE(all, lz12);
    EXPECT_LE(lz1c, plain);
    EXPECT_LE(lz12, plain);
}

TEST(AdaptiveLorenzoStage, TileSizes) {
    const size_t N = 8192;
    auto v = make_mixed(N);
    CudaStream cs;
    for (uint32_t bpt : {1u, 2u, 8u, 16u, 32u}) {
        AdaptiveLorenzoStage<int32_t>::Config c;
        c.blocks_per_tile = bpt;
        Pipeline p(N * sizeof(int32_t), MemoryStrategy::PREALLOCATE);
        p.addStage<AdaptiveLorenzoStage<int32_t>>(c);
        p.finalize();
        auto res = pipeline_round_trip<int32_t>(p, v, cs.stream);
        EXPECT_EQ(res.max_error, 0.0f) << "blocks_per_tile = " << bpt;
    }
}

TEST(AdaptiveLorenzoStage, RejectsBadConfig) {
    AdaptiveLorenzoStage<int32_t>::Config c;
    c.coder_block_size = 64;
    EXPECT_THROW(AdaptiveLorenzoStage<int32_t>{c}, std::invalid_argument);

    AdaptiveLorenzoStage<int32_t>::Config d;
    d.blocks_per_tile = 64;   // tile would exceed 1024
    EXPECT_THROW(AdaptiveLorenzoStage<int32_t>{d}, std::invalid_argument);

    AdaptiveLorenzoStage<int32_t>::Config e;
    e.blocks_per_tile = 0;
    EXPECT_THROW(AdaptiveLorenzoStage<int32_t>{e}, std::invalid_argument);
}

TEST(AdaptiveLorenzoStage, Int16RoundTrip) {
    const size_t N = 2048;
    std::vector<int16_t> v(N);
    for (size_t i = 0; i < N; ++i)
        v[i] = static_cast<int16_t>(1000 + (i % 61));

    Pipeline p(N * sizeof(int16_t), MemoryStrategy::PREALLOCATE);
    p.addStage<AdaptiveLorenzoStage<int16_t>>();
    p.finalize();

    CudaStream cs;
    auto res = pipeline_round_trip<int16_t>(p, v, cs.stream);
    EXPECT_EQ(res.max_error, 0.0f);
}

// ─────────────────────────────────────────────────────────────────────────────
// Means compaction and mode packing — AL13-AL16
// ─────────────────────────────────────────────────────────────────────────────

TEST(AdaptiveLorenzoStage, MeansCompactedToCenteredTiles) {
    // All-zero data: no tile can benefit from centering (mu is 0), so the means
    // port must compact to nothing rather than one slot per tile.
    const size_t N = 8192;
    std::vector<int32_t> v(N, 0);

    Pipeline p(N * sizeof(int32_t), MemoryStrategy::PREALLOCATE);
    auto* al = p.addStage<AdaptiveLorenzoStage<int32_t>>();
    p.finalize();

    CudaStream cs;
    auto res = pipeline_round_trip<int32_t>(p, v, cs.stream);
    EXPECT_EQ(res.max_error, 0.0f);

    auto sizes = al->getActualOutputSizesByName();
    EXPECT_EQ(sizes["means"], 0u) << "no tile centers, so no mean should be stored";
    // 2 bits per tile: 32 tiles of 256 elements -> 8 bytes, not 32.
    EXPECT_EQ(sizes["modes"], (N / 256 + 3) / 4);
}

TEST(AdaptiveLorenzoStage, MeansStoredWhenEveryTileCenters) {
    // Large constant offset: every tile should center, so means is dense again.
    const size_t N = 8192;
    std::vector<int32_t> v(N);
    for (size_t i = 0; i < N; ++i) v[i] = 500000 + static_cast<int32_t>((i * 13) % 17);

    Pipeline p(N * sizeof(int32_t), MemoryStrategy::PREALLOCATE);
    auto* al = p.addStage<AdaptiveLorenzoStage<int32_t>>();
    p.finalize();

    CudaStream cs;
    auto res = pipeline_round_trip<int32_t>(p, v, cs.stream);
    EXPECT_EQ(res.max_error, 0.0f);
    EXPECT_EQ(al->getActualOutputSizesByName()["means"],
              (N / 256) * sizeof(int32_t));
}

TEST(AdaptiveLorenzoStage, MixedCompactionRoundTrip) {
    // Half the tiles center and half do not, so the inverse must rebuild the
    // compaction offsets from the packed modes and land on the right slots.
    const size_t N = 8192;
    std::vector<int32_t> v(N);
    for (size_t i = 0; i < N; ++i) {
        const bool offset_tile = ((i / 256) % 2) == 0;
        v[i] = offset_tile ? 500000 + static_cast<int32_t>(i % 19)
                           : static_cast<int32_t>((i % 7) == 0 ? 2 : 0);
    }

    Pipeline p(N * sizeof(int32_t), MemoryStrategy::PREALLOCATE);
    auto* al = p.addStage<AdaptiveLorenzoStage<int32_t>>();
    p.finalize();

    CudaStream cs;
    auto res = pipeline_round_trip<int32_t>(p, v, cs.stream);
    EXPECT_EQ(res.max_error, 0.0f);

    const size_t means = al->getActualOutputSizesByName()["means"];
    EXPECT_GT(means, 0u);
    EXPECT_LT(means, (N / 256) * sizeof(int32_t))
        << "only some tiles center, so means must be strictly compacted";
}

TEST(AdaptiveLorenzoStage, CompactionSurvivesTileCountNotMultipleOfFour) {
    // Mode packing is 4 tiles per byte; 5 tiles exercises the ragged last byte.
    const size_t N = 5 * 256 + 100;
    auto v = make_mixed(N);

    Pipeline p(N * sizeof(int32_t), MemoryStrategy::PREALLOCATE);
    p.addStage<AdaptiveLorenzoStage<int32_t>>();
    p.finalize();

    CudaStream cs;
    auto res = pipeline_round_trip<int32_t>(p, v, cs.stream);
    EXPECT_EQ(res.max_error, 0.0f);
}
