/**
 * tests/stages/test_rre_stage.cpp
 *
 * GPU unit tests for the LC RZE zero-word bitmap reducer.
 * Compacts non-repeated words (a word differs from its predecessor) + a 1-bit
 * per-word bitmap, then recursively RE-compresses the bitmap.  Inverse
 * reconstructs exactly.  Word granularity is 1/2/4/8 bytes.
 *
 *   RZ1   RZEStage/RandomBytesRoundTrip          — random bytes restore exactly
 *   RZ2   RZEStage/AllZerosRoundTrip             — all-zeros input (fast path)
 *   RZ3   RZEStage/ConstantRunRoundTrip          — single repeated value (great RE target)
 *   RZ4   RZEStage/LongRunsRoundTrip             — long repeated-byte runs round-trip
 *   RZ5   RZEStage/MultiChunkRoundTrip           — 4×16 KB chunks restore exactly
 *   RZ6   RZEStage/PartialChunkRoundTrip         — input < one chunk round-trips exactly
 *   RZ7   RZEStage/WordSize2RoundTrip            — 2-byte word granularity round-trip
 *   RZ8   RZEStage/WordSize4RoundTrip            — 4-byte word granularity round-trip
 *   RZ9   RZEStage/WordSize8RoundTrip            — 8-byte word granularity round-trip
 *   RZ10  RZEStage/ConstantRunCompressesSmall    — constant run compresses far below input
 *   RZ11  RZEStage/HeaderSerialization           — serializeHeader/deserializeHeader preserves config
 *   RZ12  RZEStage/UnsupportedChunkSizeThrows    — chunk_size not in {4096,8192,16384} throws at execute()
 *   RZ13  RZEStage/UnsupportedWordSizeThrows     — word_size∉{1,2,4,8} throws at execute()
 *   RZ14  RZEStage/IsGraphCompatible             — forward=true, inverse=false
 *   RZ15  RZEStage/RepeatedRoundTripStable       — repeated round-trips on same objects stable
 *   RZ16  RZEStage/ChunkSize4096RandomBytesRoundTrip   — chunk_size=4096, multi-chunk random bytes
 *   RZ17  RZEStage/ChunkSize4096AllZerosRoundTrip      — chunk_size=4096, all-zeros fast path
 *   RZ18  RZEStage/ChunkSize4096ConstantRunRoundTrip   — chunk_size=4096, single repeated value
 *   RZ19  RZEStage/ChunkSize4096PartialChunkRoundTrip  — chunk_size=4096, input < one chunk
 *   RZ20  RZEStage/ChunkSize8192RandomBytesRoundTrip   — chunk_size=8192, multi-chunk random bytes
 *   RZ21  RZEStage/ChunkSize8192PartialChunkRoundTrip  — chunk_size=8192, input not a multiple of chunk_size
 *   RZ22  RZEStage/ChunkSize4096WordSize2RoundTrip     — chunk_size=4096, word_size=2 (byteshort fallback path)
 */

#include <gtest/gtest.h>
#include "helpers/fz_test_utils.h"
#include "coders/rze/rze_stage.h"
#include "fzgpumodules.h"

#include <cstdint>
#include <vector>
#include <random>
#include <cstring>

using namespace fz;
using namespace fz_test;

// Run RZEStage (encode or decode, per its is_inverse flag) on a byte vector.
static std::vector<uint8_t> run_rze(
    RZEStage& stage, const std::vector<uint8_t>& h_in,
    size_t out_cap, cudaStream_t stream, fz::MemoryPool& pool)
{
    const size_t n_in = h_in.size();
    CudaBuffer<uint8_t> d_in(n_in);
    CudaBuffer<uint8_t> d_out(out_cap);
    d_in.upload(h_in, stream);
    cudaStreamSynchronize(stream);

    std::vector<void*>  inputs  = {d_in.void_ptr()};
    std::vector<void*>  outputs = {d_out.void_ptr()};
    std::vector<size_t> sizes   = {n_in};

    stage.execute(stream, &pool, inputs, outputs, sizes);
    stage.postStreamSync(stream);
    cudaStreamSynchronize(stream);

    const size_t actual = stage.getActualOutputSizesByName().at("output");
    std::vector<uint8_t> h_out(actual);
    cudaMemcpy(h_out.data(), d_out.get(), actual, cudaMemcpyDeviceToHost);
    return h_out;
}

// Compress then decompress; verify byte-exact round-trip.
static void round_trip(const std::vector<uint8_t>& original, int word_size = 1,
                        size_t chunk_size = 16384) {
    CudaStream cs;
    auto pool = make_test_pool(original.size() + 65536);

    RZEStage enc;
    enc.setChunkSize(chunk_size);
    enc.setWordSize(word_size);
    const size_t enc_cap = enc.estimateOutputSizes({original.size()})[0];
    const auto compressed = run_rze(enc, original, enc_cap, cs.stream, *pool);

    RZEStage dec;
    dec.setChunkSize(chunk_size);
    dec.setWordSize(word_size);
    dec.setInverse(true);
    const auto restored = run_rze(dec, compressed, original.size() + 4096, cs.stream, *pool);

    ASSERT_EQ(restored.size(), original.size());
    EXPECT_EQ(restored, original) << "RZE round-trip mismatch (word_size=" << word_size
                                   << ", chunk_size=" << chunk_size << ")";
}

TEST(RZEStage, RandomBytesRoundTrip) {
    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> dist(0, 255);
    std::vector<uint8_t> data(16384);
    for (auto& b : data) b = (uint8_t)dist(rng);
    round_trip(data);
}

TEST(RZEStage, AllZerosRoundTrip) {
    round_trip(std::vector<uint8_t>(16384, 0));
}

TEST(RZEStage, ConstantRunRoundTrip) {
    round_trip(std::vector<uint8_t>(16384, 0x5A));
}

TEST(RZEStage, LongRunsRoundTrip) {
    std::vector<uint8_t> data;
    std::mt19937 rng(7);
    std::uniform_int_distribution<int> val(0, 255), len(8, 512);
    while (data.size() < 40000) {
        uint8_t v = (uint8_t)val(rng);
        int n = len(rng);
        for (int i = 0; i < n && data.size() < 40000; i++) data.push_back(v);
    }
    round_trip(data);
}

TEST(RZEStage, MultiChunkRoundTrip) {
    std::mt19937 rng(99);
    std::uniform_int_distribution<int> dist(0, 3);  // low entropy → repetitions
    std::vector<uint8_t> data(4 * 16384);
    for (auto& b : data) b = (uint8_t)dist(rng);
    round_trip(data);
}

TEST(RZEStage, PartialChunkRoundTrip) {
    std::mt19937 rng(3);
    std::uniform_int_distribution<int> dist(0, 5);
    std::vector<uint8_t> data(3000);
    for (auto& b : data) b = (uint8_t)dist(rng);
    round_trip(data);
}

TEST(RZEStage, WordSize2RoundTrip) {
    std::mt19937 rng(21);
    std::uniform_int_distribution<int> dist(0, 7);
    std::vector<uint8_t> data(2 * 16384);
    for (auto& b : data) b = (uint8_t)dist(rng);
    round_trip(data, 2);
}

TEST(RZEStage, WordSize4RoundTrip) {
    std::mt19937 rng(22);
    std::uniform_int_distribution<int> dist(0, 7);
    std::vector<uint8_t> data(2 * 16384);
    for (auto& b : data) b = (uint8_t)dist(rng);
    round_trip(data, 4);
}

TEST(RZEStage, WordSize8RoundTrip) {
    std::mt19937 rng(23);
    std::uniform_int_distribution<int> dist(0, 7);
    std::vector<uint8_t> data(2 * 16384);
    for (auto& b : data) b = (uint8_t)dist(rng);
    round_trip(data, 8);
}

TEST(RZEStage, AllZerosCompressesSmall) {
    CudaStream cs;
    std::vector<uint8_t> data(16384, 0);   // RZE eliminates zeros
    auto pool = make_test_pool(data.size() + 65536);
    RZEStage enc;
    enc.setChunkSize(16384);
    enc.setWordSize(1);
    const size_t enc_cap = enc.estimateOutputSizes({data.size()})[0];
    const auto compressed = run_rze(enc, data, enc_cap, cs.stream, *pool);
    // header (12 bytes for 1 chunk) + a couple of payload bytes
    EXPECT_LT(compressed.size(), (size_t)64) << "all-zeros did not compress";
}

TEST(RZEStage, HeaderSerialization) {
    RZEStage s;
    s.setChunkSize(16384);
    s.setWordSize(4);
    uint8_t buf[9] = {0};
    ASSERT_EQ(s.serializeHeader(0, buf, sizeof(buf)), (size_t)9);
    RZEStage s2;
    s2.deserializeHeader(buf, sizeof(buf));
    EXPECT_EQ(s2.getChunkSize(), (size_t)16384);
    EXPECT_EQ(s2.getWordSize(), 4);
}

TEST(RZEStage, UnsupportedChunkSizeThrows) {
    CudaStream cs;
    std::vector<uint8_t> data(4096, 1);
    auto pool = make_test_pool(data.size() + 65536);
    RZEStage s;
    s.setChunkSize(12345);  // not in the supported set {4096, 8192, 16384}
    CudaBuffer<uint8_t> d_in(data.size()), d_out(data.size() + 4096);
    d_in.upload(data, cs.stream);
    cudaStreamSynchronize(cs.stream);
    std::vector<void*> in = {d_in.void_ptr()}, out = {d_out.void_ptr()};
    std::vector<size_t> sz = {data.size()};
    EXPECT_THROW(s.execute(cs.stream, pool.get(), in, out, sz), std::runtime_error);
}

TEST(RZEStage, UnsupportedWordSizeThrows) {
    CudaStream cs;
    std::vector<uint8_t> data(4096, 1);
    auto pool = make_test_pool(data.size() + 65536);
    RZEStage s;
    s.setWordSize(3);
    CudaBuffer<uint8_t> d_in(data.size()), d_out(data.size() + 4096);
    d_in.upload(data, cs.stream);
    cudaStreamSynchronize(cs.stream);
    std::vector<void*> in = {d_in.void_ptr()}, out = {d_out.void_ptr()};
    std::vector<size_t> sz = {data.size()};
    EXPECT_THROW(s.execute(cs.stream, pool.get(), in, out, sz), std::runtime_error);
}

TEST(RZEStage, IsGraphCompatible) {
    RZEStage fwd;
    EXPECT_TRUE(fwd.isGraphCompatible());
    RZEStage inv;
    inv.setInverse(true);
    EXPECT_FALSE(inv.isGraphCompatible());
}

TEST(RZEStage, RepeatedRoundTripStable) {
    std::mt19937 rng(55);
    std::uniform_int_distribution<int> dist(0, 4);
    std::vector<uint8_t> data(2 * 16384);
    for (auto& b : data) b = (uint8_t)dist(rng);
    for (int i = 0; i < 5; i++) round_trip(data);
}

TEST(RZEStage, ChunkSize4096RandomBytesRoundTrip) {
    std::mt19937 rng(101);
    std::uniform_int_distribution<int> dist(0, 255);
    std::vector<uint8_t> data(3 * 4096);  // multi-chunk
    for (auto& b : data) b = (uint8_t)dist(rng);
    round_trip(data, 1, 4096);
}

TEST(RZEStage, ChunkSize4096AllZerosRoundTrip) {
    round_trip(std::vector<uint8_t>(2 * 4096, 0), 1, 4096);
}

TEST(RZEStage, ChunkSize4096ConstantRunRoundTrip) {
    round_trip(std::vector<uint8_t>(2 * 4096, 0x5A), 1, 4096);
}

TEST(RZEStage, ChunkSize4096PartialChunkRoundTrip) {
    std::mt19937 rng(102);
    std::uniform_int_distribution<int> dist(0, 5);
    std::vector<uint8_t> data(1500);  // < one 4096-byte chunk
    for (auto& b : data) b = (uint8_t)dist(rng);
    round_trip(data, 1, 4096);
}

TEST(RZEStage, ChunkSize8192RandomBytesRoundTrip) {
    std::mt19937 rng(103);
    std::uniform_int_distribution<int> dist(0, 255);
    std::vector<uint8_t> data(3 * 8192);  // multi-chunk
    for (auto& b : data) b = (uint8_t)dist(rng);
    round_trip(data, 1, 8192);
}

TEST(RZEStage, ChunkSize8192PartialChunkRoundTrip) {
    std::mt19937 rng(104);
    std::uniform_int_distribution<int> dist(0, 5);
    std::vector<uint8_t> data(2 * 8192 + 3000);  // not a multiple of chunk_size
    for (auto& b : data) b = (uint8_t)dist(rng);
    round_trip(data, 1, 8192);
}

// chunk_size=4096 + word_size=2 gives < 1 meaningful bitmap byte per thread
// in the byteshort fast path (8 threads*512 bytes/thread=4096B chunk, but
// bytesperthread/(8*sizeof(uint16_t)) = 8/16 = 0) — the dispatcher must fall
// back to the general path here instead of taking byteshort.
TEST(RZEStage, ChunkSize4096WordSize2RoundTrip) {
    std::mt19937 rng(105);
    std::uniform_int_distribution<int> dist(0, 7);
    std::vector<uint8_t> data(3 * 4096);
    for (auto& b : data) b = (uint8_t)dist(rng);
    round_trip(data, 2, 4096);
}
