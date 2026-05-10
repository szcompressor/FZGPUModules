/**
 * tests/stages/test_rze_stage.cpp
 *
 * GPU unit tests for RZEStage — Recursive Zero-byte Elimination.
 * Compacts non-zero bytes + bitmap (ZE), then recursively eliminates repeated
 * bitmap bytes (RE) for 3 more levels. Inverse reconstructs exactly.
 *
 *   RZ1   RZEStage/RandomBytesRoundTrip          — random bytes restore exactly
 *   RZ2   RZEStage/AllZerosRoundTrip             — all-zeros input (special fast path)
 *   RZ3   RZEStage/AllOnesRoundTrip              — all-0xFF input (incompressible, stored verbatim)
 *   RZ4   RZEStage/SparseDataRoundTrip           — ~1% non-zero bytes round-trip exactly
 *   RZ5   RZEStage/MultiChunkRoundTrip           — 2×16 KB chunks restore exactly
 *   RZ6   RZEStage/SparseDataCompressionRatio    — sparse data compresses smaller than input
 *   RZ7   RZEStage/AllZerosCompressesSmall       — all-zeros encodes to ≤17 bytes
 *   RZ8   RZEStage/HeaderSerialization           — serializeHeader/deserializeHeader preserves config
 *   RZ9   RZEStage/CheckerboardRoundTrip         — alternating 0x00/0xAA pattern round-trips
 *   RZ10  RZEStage/LongRunsRoundTrip             — long repeated-byte runs (good RE target) round-trip
 *   RZ11  RZEStage/UnsupportedChunkSizeThrows    — chunk_size≠16384 throws at execute()
 *   RZ12  RZEStage/UnsupportedLevelsThrows       — levels∉{4} throws at execute()
 *   RZ13  RZEStage/IsGraphCompatible             — forward=true, inverse=false (blocking D2H)
 *   RZ14  RZEStage/SaveRestoreStatePreservesConfig — saveState/restoreState preserves forward config
 *   RZ15  RZEStage/RepeatedRoundTripStable       — 5 repeated round-trips on same stage objects stable
 *   RZ16  RZEStage/PartialChunkRoundTrip         — input < one chunk (3000 bytes) round-trips exactly
 *   RZ17  RZEStage/FourChunksRoundTrip           — 4×16 KB (64 KB) round-trips correctly
 *   RZ18  RZEStage/MixedDensityMultiChunk        — half sparse, half dense; round-trip correct
 *   RZ19  RZEStage/PipelineIntegration           — LorenzoQuantStage→RZEStage pipeline round-trip
 *   RZ20  RZEStage/PipelineCompressionRatio      — Lorenzo→RZE compressed size < raw input
 */

#include <gtest/gtest.h>
#include "helpers/fz_test_utils.h"
#include "coders/rze/rze_stage.h"
#include "fzgpumodules.h"

#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>
#include <algorithm>
#include <random>
#include <cstring>

using namespace fz;
using namespace fz_test;

// ─────────────────────────────────────────────────────────────────────────────
// Helper: run RZEStage (encode or decode) on a byte vector.
// The stage must already have is_inverse set correctly.
// ─────────────────────────────────────────────────────────────────────────────
static std::vector<uint8_t> run_rze(
    RZEStage&                   stage,
    const std::vector<uint8_t>& h_in,
    cudaStream_t                stream,
    fz::MemoryPool&             pool)
{
    const size_t n_in = h_in.size();
    std::vector<size_t> in_sizes = {n_in};
    const size_t n_out = stage.estimateOutputSizes(in_sizes)[0];

    CudaBuffer<uint8_t> d_in(n_in);
    CudaBuffer<uint8_t> d_out(n_out);
    d_in.upload(h_in, stream);
    cudaStreamSynchronize(stream);

    std::vector<void*> inputs  = {d_in.void_ptr()};
    std::vector<void*> outputs = {d_out.void_ptr()};
    std::vector<size_t> sizes  = {n_in};

    stage.execute(stream, &pool, inputs, outputs, sizes);

    const size_t actual = stage.getActualOutputSizesByName().at("output");
    std::vector<uint8_t> h_out(actual);
    cudaMemcpy(h_out.data(), d_out.get(), actual, cudaMemcpyDeviceToHost);
    return h_out;
}

// Compress then decompress a byte vector; verify byte-exact round-trip.
static void round_trip(const std::vector<uint8_t>& original) {
    CudaStream cs;
    auto pool = make_test_pool(original.size());

    RZEStage enc;
    enc.setChunkSize(16384);
    enc.setLevels(4);

    // Encode
    const auto compressed = run_rze(enc, original, cs.stream, *pool);

    // Decode
    RZEStage dec;
    dec.setChunkSize(16384);
    dec.setLevels(4);
    dec.setInverse(true);

    // dec.estimateOutputSizes needs to know input size; we'll feed the
    // compressed byte count via overriding sizes in run_rze's internal call.
    // Since run_rze uses the vector size passed as n_in, we pass compressed.
    std::vector<size_t> in_sizes2 = {compressed.size()};
    const size_t n_dec_out = original.size() + 4096;  // conservative upper bound

    CudaBuffer<uint8_t> d_in2(compressed.size());
    CudaBuffer<uint8_t> d_out2(n_dec_out);

    cudaMemcpy(d_in2.get(), compressed.data(), compressed.size(), cudaMemcpyHostToDevice);

    std::vector<void*> ins2  = {d_in2.void_ptr()};
    std::vector<void*> outs2 = {d_out2.void_ptr()};
    std::vector<size_t> szs2 = {compressed.size()};

    dec.execute(cs.stream, pool.get(), ins2, outs2, szs2);
    const size_t actual_dec = dec.getActualOutputSizesByName().at("output");

    std::vector<uint8_t> restored(actual_dec);
    cudaMemcpy(restored.data(), d_out2.get(), actual_dec, cudaMemcpyDeviceToHost);

    ASSERT_EQ(restored.size(), original.size())
        << "Decompressed size does not match original";
    EXPECT_EQ(restored, original) << "Decompressed data does not match original";
}

// ─────────────────────────────────────────────────────────────────────────────
// RZ1: RZEStage/RandomBytesRoundTrip — random bytes restore exactly
// ─────────────────────────────────────────────────────────────────────────────
TEST(RZEStage, RandomBytesRoundTrip) {
    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> dist(0, 255);
    std::vector<uint8_t> data(16384);
    for (auto& b : data) b = static_cast<uint8_t>(dist(rng));
    round_trip(data);
}

// ─────────────────────────────────────────────────────────────────────────────
// RZ2: RZEStage/AllZerosRoundTrip — all-zeros input (special fast path)
// ─────────────────────────────────────────────────────────────────────────────
TEST(RZEStage, AllZerosRoundTrip) {
    round_trip(std::vector<uint8_t>(16384, 0));
}

// ─────────────────────────────────────────────────────────────────────────────
// RZ3: RZEStage/AllOnesRoundTrip — all-0xFF input (incompressible, stored verbatim)
// ─────────────────────────────────────────────────────────────────────────────
TEST(RZEStage, AllOnesRoundTrip) {
    round_trip(std::vector<uint8_t>(16384, 0xFF));
}

// ─────────────────────────────────────────────────────────────────────────────
// RZ4: RZEStage/SparseDataRoundTrip — ~1% non-zero bytes round-trip exactly
// ─────────────────────────────────────────────────────────────────────────────
TEST(RZEStage, SparseDataRoundTrip) {
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> idx_dist(0, 16383);
    std::uniform_int_distribution<int> val_dist(1, 255);
    std::vector<uint8_t> data(16384, 0);
    for (int i = 0; i < 164; i++)
        data[idx_dist(rng)] = static_cast<uint8_t>(val_dist(rng));
    round_trip(data);
}

// ─────────────────────────────────────────────────────────────────────────────
// RZ5: RZEStage/MultiChunkRoundTrip — 2×16 KB chunks restore exactly
// ─────────────────────────────────────────────────────────────────────────────
TEST(RZEStage, MultiChunkRoundTrip) {
    std::mt19937 rng(99);
    std::uniform_int_distribution<int> dist(0, 255);
    std::vector<uint8_t> data(32768);
    for (auto& b : data) b = static_cast<uint8_t>(dist(rng));
    round_trip(data);
}

// ─────────────────────────────────────────────────────────────────────────────
// RZ6: RZEStage/SparseDataCompressionRatio — sparse data compresses smaller than input
// ─────────────────────────────────────────────────────────────────────────────
TEST(RZEStage, SparseDataCompressionRatio) {
    // 0.5% non-zero bytes → should compress significantly
    std::vector<uint8_t> data(16384, 0);
    std::mt19937 rng(7);
    std::uniform_int_distribution<int> pos_dist(0, 16383);
    for (int i = 0; i < 80; i++)
        data[pos_dist(rng)] = static_cast<uint8_t>(i + 1);

    CudaStream cs;
    auto pool = make_test_pool(data.size());

    RZEStage enc;
    enc.setChunkSize(16384);
    enc.setLevels(4);
    const auto compressed = run_rze(enc, data, cs.stream, *pool);

    // Header alone is 4+4+4 = 12 bytes; compressed payload should be small
    EXPECT_LT(compressed.size(), data.size())
        << "Sparse input should compress below original size";
}

// ─────────────────────────────────────────────────────────────────────────────
// RZ7: RZEStage/AllZerosCompressesSmall — all-zeros encodes to ≤17 bytes
// ─────────────────────────────────────────────────────────────────────────────
TEST(RZEStage, AllZerosCompressesSmall) {
    std::vector<uint8_t> data(16384, 0);
    CudaStream cs;
    auto pool = make_test_pool(data.size());

    RZEStage enc;
    enc.setChunkSize(16384);
    enc.setLevels(4);
    const auto compressed = run_rze(enc, data, cs.stream, *pool);

    // Header (12 bytes) + 2-byte all-zeros tag = 14 bytes of real output,
    // plus up to 3 bytes of 4-byte alignment padding = 17 bytes max.
    EXPECT_LE(compressed.size(), 17u)
        << "All-zeros chunk should compress to header + 2-byte tag (≤17 with alignment)";
}

// ─────────────────────────────────────────────────────────────────────────────
// RZ8: RZEStage/HeaderSerialization — serializeHeader/deserializeHeader preserves config
// ─────────────────────────────────────────────────────────────────────────────
TEST(RZEStage, HeaderSerialization) {
    RZEStage s;
    s.setChunkSize(16384);
    s.setLevels(4);

    uint8_t buf[16] = {};
    const size_t n = s.serializeHeader(0, buf, sizeof(buf));
    EXPECT_EQ(n, 9u);  // 4 (chunk_size) + 1 (levels) + 4 (cached_orig_bytes)

    RZEStage s2;
    s2.deserializeHeader(buf, n);
    EXPECT_EQ(s2.getChunkSize(), 16384u);
    EXPECT_EQ(s2.getLevels(),    4);
    EXPECT_EQ(s2.getCachedOrigBytes(), 0u);  // not set until execute() is called
}

// ─────────────────────────────────────────────────────────────────────────────
// RZ9: RZEStage/CheckerboardRoundTrip — alternating 0x00/0xAA pattern round-trips
// ─────────────────────────────────────────────────────────────────────────────
TEST(RZEStage, CheckerboardRoundTrip) {
    std::vector<uint8_t> data(16384);
    for (size_t i = 0; i < data.size(); i++)
        data[i] = (i % 2 == 0) ? 0x00u : 0xAAu;
    round_trip(data);
}

// ─────────────────────────────────────────────────────────────────────────────
// RZ10: RZEStage/LongRunsRoundTrip — long repeated-byte runs (good RE target) round-trip
// ─────────────────────────────────────────────────────────────────────────────
TEST(RZEStage, LongRunsRoundTrip) {
    std::vector<uint8_t> data(16384, 0);
    // Fill with 128 blocks of 128 bytes, each block a distinct byte value
    for (int block = 0; block < 128; block++)
        std::fill(data.begin() + block * 128,
                  data.begin() + (block + 1) * 128,
                  static_cast<uint8_t>(block + 1));
    round_trip(data);
}

// ─────────────────────────────────────────────────────────────────────────────
// RZ11: RZEStage/UnsupportedChunkSizeThrows — chunk_size≠16384 throws at execute()
// ─────────────────────────────────────────────────────────────────────────────
TEST(RZEStage, UnsupportedChunkSizeThrows) {
    std::vector<uint8_t> data(16384, 0xAB);
    CudaStream cs;
    auto pool = make_test_pool(data.size());

    // Forward
    {
        RZEStage enc;
        enc.setChunkSize(8192);  // unsupported
        enc.setLevels(4);

        CudaBuffer<uint8_t> d_in(data.size()), d_out(data.size() * 2);
        cudaMemcpy(d_in.get(), data.data(), data.size(), cudaMemcpyHostToDevice);
        std::vector<void*> ins  = {d_in.void_ptr()};
        std::vector<void*> outs = {d_out.void_ptr()};
        std::vector<size_t> szs = {data.size()};

        EXPECT_THROW(enc.execute(cs.stream, pool.get(), ins, outs, szs),
                     std::runtime_error)
            << "chunk_size=8192 (forward) must throw";
    }

    // Inverse
    {
        RZEStage dec;
        dec.setChunkSize(8192);
        dec.setLevels(4);
        dec.setInverse(true);

        CudaBuffer<uint8_t> d_in(data.size()), d_out(data.size() * 2);
        cudaMemcpy(d_in.get(), data.data(), data.size(), cudaMemcpyHostToDevice);
        std::vector<void*> ins  = {d_in.void_ptr()};
        std::vector<void*> outs = {d_out.void_ptr()};
        std::vector<size_t> szs = {data.size()};

        EXPECT_THROW(dec.execute(cs.stream, pool.get(), ins, outs, szs),
                     std::runtime_error)
            << "chunk_size=8192 (inverse) must throw";
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// RZ12: RZEStage/UnsupportedLevelsThrows — levels∉{4} throws at execute()
// ─────────────────────────────────────────────────────────────────────────────
TEST(RZEStage, UnsupportedLevelsThrows) {
    std::vector<uint8_t> data(16384, 0x55);
    CudaStream cs;
    auto pool = make_test_pool(data.size());

    CudaBuffer<uint8_t> d_in(data.size()), d_out(data.size() * 2);
    cudaMemcpy(d_in.get(), data.data(), data.size(), cudaMemcpyHostToDevice);

    for (int bad_levels : {1, 2, 3}) {
        RZEStage enc;
        enc.setChunkSize(16384);
        enc.setLevels(bad_levels);

        std::vector<void*> ins  = {d_in.void_ptr()};
        std::vector<void*> outs = {d_out.void_ptr()};
        std::vector<size_t> szs = {data.size()};

        EXPECT_THROW(enc.execute(cs.stream, pool.get(), ins, outs, szs),
                     std::runtime_error)
            << "levels=" << bad_levels << " must throw";
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// RZ13: RZEStage/IsGraphCompatible — forward=true, inverse=false (blocking D2H)
// ─────────────────────────────────────────────────────────────────────────────
TEST(RZEStage, IsGraphCompatible) {
    RZEStage fwd;
    fwd.setChunkSize(16384);
    fwd.setLevels(4);
    EXPECT_TRUE(fwd.isGraphCompatible())
        << "RZE forward must be graph-compatible";

    RZEStage inv;
    inv.setChunkSize(16384);
    inv.setLevels(4);
    inv.setInverse(true);
    EXPECT_FALSE(inv.isGraphCompatible())
        << "RZE inverse must NOT be graph-compatible (blocking D2H in execute)";
}

// ─────────────────────────────────────────────────────────────────────────────
// RZ14: RZEStage/SaveRestoreStatePreservesConfig — saveState/restoreState preserves forward config
// ─────────────────────────────────────────────────────────────────────────────
TEST(RZEStage, SaveRestoreStatePreservesConfig) {
    std::vector<uint8_t> data(16384, 0x33);
    CudaStream cs;
    auto pool = make_test_pool(data.size() * 4);

    RZEStage stage;
    stage.setChunkSize(16384);
    stage.setLevels(4);

    // Forward compress
    const auto compressed = run_rze(stage, data, cs.stream, *pool);
    const uint32_t cached_after_compress = stage.getCachedOrigBytes();
    EXPECT_EQ(cached_after_compress, data.size())
        << "cached_orig_bytes_ must equal input size after compress";

    // Simulate what decompressMulti() does: saveState, switch to inverse,
    // execute, restoreState.
    stage.saveState();
    stage.setInverse(true);

    CudaBuffer<uint8_t> d_comp(compressed.size()), d_decomp(data.size() + 4096);
    cudaMemcpy(d_comp.get(), compressed.data(), compressed.size(), cudaMemcpyHostToDevice);
    {
        std::vector<void*> ins  = {d_comp.void_ptr()};
        std::vector<void*> outs = {d_decomp.void_ptr()};
        std::vector<size_t> szs = {compressed.size()};
        stage.execute(cs.stream, pool.get(), ins, outs, szs);
    }

    stage.setInverse(false);
    stage.restoreState();

    // Config must be restored to the forward values
    EXPECT_EQ(stage.getChunkSize(),       16384u);
    EXPECT_EQ(stage.getLevels(),          4);
    EXPECT_EQ(stage.getCachedOrigBytes(), cached_after_compress)
        << "cached_orig_bytes_ must survive saveState/restoreState";
    EXPECT_FALSE(stage.isInverse())
        << "stage must be in forward mode after restoreState";
}

// ─────────────────────────────────────────────────────────────────────────────
// RZ15: RZEStage/RepeatedRoundTripStable — 5 repeated round-trips on same stage objects stable
// ─────────────────────────────────────────────────────────────────────────────
TEST(RZEStage, RepeatedRoundTripStable) {
    std::mt19937 rng(2024);
    std::uniform_int_distribution<int> dist(0, 255);

    CudaStream cs;
    auto pool = make_test_pool(16384 * 4);

    RZEStage enc, dec;
    enc.setChunkSize(16384); enc.setLevels(4);
    dec.setChunkSize(16384); dec.setLevels(4); dec.setInverse(true);

    for (int iter = 0; iter < 5; iter++) {
        std::vector<uint8_t> data(16384);
        for (auto& b : data) b = static_cast<uint8_t>(dist(rng));

        // Compress
        const auto compressed = run_rze(enc, data, cs.stream, *pool);
        ASSERT_GT(compressed.size(), 0u) << "iter " << iter << ": empty compressed output";

        // Decompress
        CudaBuffer<uint8_t> d_in(compressed.size()), d_out(data.size() + 4096);
        cudaMemcpy(d_in.get(), compressed.data(), compressed.size(), cudaMemcpyHostToDevice);
        {
            std::vector<void*> ins  = {d_in.void_ptr()};
            std::vector<void*> outs = {d_out.void_ptr()};
            std::vector<size_t> szs = {compressed.size()};
            dec.execute(cs.stream, pool.get(), ins, outs, szs);
        }
        const size_t actual = dec.getActualOutputSizesByName().at("output");
        ASSERT_EQ(actual, data.size()) << "iter " << iter << ": wrong decompressed size";

        std::vector<uint8_t> restored(actual);
        cudaMemcpy(restored.data(), d_out.get(), actual, cudaMemcpyDeviceToHost);
        EXPECT_EQ(restored, data) << "iter " << iter << ": decompressed data mismatch";
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// RZ16: RZEStage/PartialChunkRoundTrip — input < one chunk (3000 bytes) round-trips exactly
// ─────────────────────────────────────────────────────────────────────────────
TEST(RZEStage, PartialChunkRoundTrip) {
    // Use a size that is not a power of 2 and well below 16384
    std::vector<uint8_t> data(3000, 0);
    std::mt19937 rng(555);
    std::uniform_int_distribution<int> dist(0, 255);
    for (int i = 0; i < 30; i++)
        data[rng() % 3000] = static_cast<uint8_t>(dist(rng));
    round_trip(data);
}

// ─────────────────────────────────────────────────────────────────────────────
// RZ17: RZEStage/FourChunksRoundTrip — 4×16 KB (64 KB) round-trips correctly
// ─────────────────────────────────────────────────────────────────────────────
TEST(RZEStage, FourChunksRoundTrip) {
    std::mt19937 rng(777);
    std::uniform_int_distribution<int> dist(0, 255);
    std::vector<uint8_t> data(65536);
    for (auto& b : data) b = static_cast<uint8_t>(dist(rng));
    round_trip(data);
}

// ─────────────────────────────────────────────────────────────────────────────
// RZ18: RZEStage/MixedDensityMultiChunk — half sparse, half dense; round-trip correct
// ─────────────────────────────────────────────────────────────────────────────
TEST(RZEStage, MixedDensityMultiChunkRoundTrip) {
    constexpr size_t N = 32768;  // 2 × 16 KB chunks

    std::vector<uint8_t> data(N, 0);

    // First chunk: very sparse (only 10 non-zero bytes)
    std::mt19937 rng(888);
    std::uniform_int_distribution<int> pos_dist(0, 16383);
    std::uniform_int_distribution<int> val_dist(1, 255);
    for (int i = 0; i < 10; i++)
        data[pos_dist(rng)] = static_cast<uint8_t>(val_dist(rng));

    // Second chunk: fully random (incompressible → verbatim path)
    std::uniform_int_distribution<int> byte_dist(0, 255);
    for (size_t i = 16384; i < N; i++)
        data[i] = static_cast<uint8_t>(byte_dist(rng));

    round_trip(data);
}

// ─────────────────────────────────────────────────────────────────────────────
// RZ19: RZEStage/PipelineIntegration — LorenzoQuantStage→RZEStage pipeline round-trip
// ─────────────────────────────────────────────────────────────────────────────
TEST(RZEStage, PipelineIntegration) {
    CudaStream stream;
    constexpr size_t N  = 1 << 13;  // 8 K floats
    constexpr float  EB = 1e-2f;
    const size_t in_bytes = N * sizeof(float);

    auto h_input = make_smooth_data<float>(N);

    CudaBuffer<float> d_in(N);
    d_in.upload(h_input, stream);
    stream.sync();

    // LorenzoQuantStage<float, uint16_t> → RZEStage: codes output feeds RZE.
    Pipeline pipeline(in_bytes, MemoryStrategy::MINIMAL, 5.0f);
    auto* lrz = pipeline.addStage<LorenzoQuantStage<float, uint16_t>>();
    lrz->setErrorBound(EB);
    lrz->setQuantRadius(512);
    lrz->setOutlierCapacity(0.2f);

    auto* rze = pipeline.addStage<RZEStage>();
    rze->setChunkSize(16384);
    rze->setLevels(4);
    pipeline.connect(rze, lrz, "codes");

    pipeline.setPoolManagedDecompOutput(false);
    pipeline.finalize();

    void*  d_comp  = nullptr;
    size_t comp_sz = 0;
    ASSERT_NO_THROW(
        pipeline.compress(d_in.void_ptr(), in_bytes, &d_comp, &comp_sz, stream)
    ) << "Lorenzo→RZE compress must not throw";
    stream.sync();
    ASSERT_GT(comp_sz, 0u) << "Compressed output is empty";

    void*  d_dec  = nullptr;
    size_t dec_sz = 0;
    ASSERT_NO_THROW(
        pipeline.decompress(nullptr, 0, &d_dec, &dec_sz, stream)
    ) << "Lorenzo→RZE decompress must not throw";
    stream.sync();
    ASSERT_NE(d_dec, nullptr);
    ASSERT_EQ(dec_sz, in_bytes);

    std::vector<float> h_recon(N);
    cudaMemcpy(h_recon.data(), d_dec, in_bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_dec);

    float max_err = 0.0f;
    for (size_t i = 0; i < N; i++)
        max_err = std::max(max_err, std::abs(h_recon[i] - h_input[i]));
    EXPECT_LE(max_err, EB * 1.01f)
        << "Lorenzo→RZE pipeline round-trip max_err=" << max_err;
}

// ─────────────────────────────────────────────────────────────────────────────
// RZ20: RZEStage/PipelineCompressionRatio — Lorenzo→RZE compressed size < raw input
// ─────────────────────────────────────────────────────────────────────────────
TEST(RZEStage, PipelineCompressionRatio) {
    CudaStream stream;
    constexpr size_t N  = 1 << 14;  // 16 K floats — enough for RZE to shine
    constexpr float  EB = 1e-2f;
    const size_t in_bytes = N * sizeof(float);

    auto h_input = make_smooth_data<float>(N);

    CudaBuffer<float> d_in(N);
    d_in.upload(h_input, stream);
    stream.sync();

    Pipeline pipeline(in_bytes, MemoryStrategy::MINIMAL, 5.0f);
    auto* lrz = pipeline.addStage<LorenzoQuantStage<float, uint16_t>>();
    lrz->setErrorBound(EB);
    lrz->setQuantRadius(512);
    lrz->setOutlierCapacity(0.2f);

    auto* rze = pipeline.addStage<RZEStage>();
    rze->setChunkSize(16384);
    rze->setLevels(4);
    pipeline.connect(rze, lrz, "codes");
    pipeline.setPoolManagedDecompOutput(false);
    pipeline.finalize();

    void*  d_comp  = nullptr;
    size_t comp_sz = 0;
    pipeline.compress(d_in.void_ptr(), in_bytes, &d_comp, &comp_sz, stream);
    stream.sync();
    ASSERT_GT(comp_sz, 0u);

    // The compressed stream + outlier buffers should be well under raw input.
    EXPECT_LT(comp_sz, in_bytes)
        << "Lorenzo→RZE on smooth data should compress below raw input size ("
        << in_bytes << " B); got " << comp_sz << " B";
}
