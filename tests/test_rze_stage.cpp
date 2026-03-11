/**
 * tests/test_rze_stage.cpp
 *
 * GPU unit tests for RZEStage — Recursive Zero-byte Elimination.
 *
 * RZEStage compresses a byte stream by:
 *   1. Compacting non-zero bytes and building a bitmap (zero elimination, ZE).
 *   2. Recursively compacting non-repeated bytes of the bitmap (repetition
 *      elimination, RE) for 3 more levels, leaving a tiny final bitmap that
 *      is stored raw.
 *
 * The inverse path reconstructs the original byte stream exactly.
 *
 * Properties verified:
 *   1.  Round-trip: random bytes restore exactly.
 *   2.  Round-trip: all-zeros input (special case — no bitmaps stored).
 *   3.  Round-trip: all-ones (0xFF) input (incompressible, stored as-is).
 *   4.  Round-trip: sparse data (mostly zeros, few non-zero values).
 *   5.  Round-trip: multi-chunk input (2 × 16 KB chunks).
 *   6.  Compression reduces size for highly sparse data.
 *   7.  Output is larger than input for incompressible (random) data OR
 *       within expected bounds.
 *   8.  Header serialization round-trips chunk_size and levels.
 *   9.  round-trip on a single chunk with a checkerboard byte pattern.
 *   10. round-trip on data with long runs of repeated bytes (good for RE).
 */

#include <gtest/gtest.h>
#include "helpers/fz_test_utils.h"
#include "transforms/rze/rze_stage.h"

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
// Tests
// ─────────────────────────────────────────────────────────────────────────────

// 1. Random bytes round-trip
TEST(RZEStage, RandomBytesRoundTrip) {
    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> dist(0, 255);
    std::vector<uint8_t> data(16384);
    for (auto& b : data) b = static_cast<uint8_t>(dist(rng));
    round_trip(data);
}

// 2. All-zeros input (special fast path)
TEST(RZEStage, AllZerosRoundTrip) {
    round_trip(std::vector<uint8_t>(16384, 0));
}

// 3. All-0xFF input (incompressible, stored verbatim)
TEST(RZEStage, AllOnesRoundTrip) {
    round_trip(std::vector<uint8_t>(16384, 0xFF));
}

// 4. Sparse data: 1% non-zero, rest zeros
TEST(RZEStage, SparseDataRoundTrip) {
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> idx_dist(0, 16383);
    std::uniform_int_distribution<int> val_dist(1, 255);
    std::vector<uint8_t> data(16384, 0);
    for (int i = 0; i < 164; i++)
        data[idx_dist(rng)] = static_cast<uint8_t>(val_dist(rng));
    round_trip(data);
}

// 5. Multi-chunk input (2 × 16 KB = 32 KB)
TEST(RZEStage, MultiChunkRoundTrip) {
    std::mt19937 rng(99);
    std::uniform_int_distribution<int> dist(0, 255);
    std::vector<uint8_t> data(32768);
    for (auto& b : data) b = static_cast<uint8_t>(dist(rng));
    round_trip(data);
}

// 6. Sparse data compresses to smaller than input
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

// 7. All-zeros compresses to near-minimum (header + 2-byte tag per chunk)
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

// 8. Header serialization round-trip
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

// 9. Checkerboard byte pattern (alternating 0x00 / 0xAA)
TEST(RZEStage, CheckerboardRoundTrip) {
    std::vector<uint8_t> data(16384);
    for (size_t i = 0; i < data.size(); i++)
        data[i] = (i % 2 == 0) ? 0x00u : 0xAAu;
    round_trip(data);
}

// 10. Long runs of the same byte value (excellent RE target)
TEST(RZEStage, LongRunsRoundTrip) {
    std::vector<uint8_t> data(16384, 0);
    // Fill with 128 blocks of 128 bytes, each block a distinct byte value
    for (int block = 0; block < 128; block++)
        std::fill(data.begin() + block * 128,
                  data.begin() + (block + 1) * 128,
                  static_cast<uint8_t>(block + 1));
    round_trip(data);
}
