/**
 * tests/stages/test_clog_stage.cpp
 *
 * GPU unit tests for CLOGStage — Compressed-Logarithm adaptive bit-width
 * coding (LC component). Splits each chunk into a fixed 32 subchunks; each
 * subchunk finds its own max value and bit-packs every element in it to the
 * minimum bit-width needed to represent that max losslessly. Unlike
 * RRE/RARE/RZE/RAZE there is no bitmap and no per-element full/dropped
 * decision — every element in a subchunk shares the same packed width. `T`
 * must be unsigned. Word granularity is 1/2/4/8 bytes.
 *
 *   CL1   CLOGStage/RandomBytesRoundTrip          — random bytes restore exactly
 *   CL2   CLOGStage/AllZerosRoundTrip             — every subchunk needs 0 bits
 *   CL3   CLOGStage/SmallValueRunCompressesSmall  — small constant compresses far below input
 *   CL4   CLOGStage/MultiChunkRoundTrip           — 4×16 KB chunks restore exactly
 *   CL5   CLOGStage/PartialChunkRoundTrip         — input < one chunk round-trips exactly
 *   CL6   CLOGStage/WordSize2RoundTrip            — 2-byte word granularity round-trip
 *   CL7   CLOGStage/WordSize4RoundTrip            — 4-byte word granularity round-trip
 *   CL8   CLOGStage/WordSize8RoundTrip            — 8-byte word granularity round-trip
 *   CL9   CLOGStage/UnevenSubchunkBoundaryRoundTrip — size not a multiple of 32 words,
 *                                                      forcing uneven beg/end subchunk splits
 *   CL10  CLOGStage/MaxWidthSubchunkRoundTrip     — a subchunk needing the full TB bits
 *                                                    mixed with all-zero subchunks
 *   CL11  CLOGStage/HeaderSerialization           — serializeHeader/deserializeHeader preserves config
 *   CL12  CLOGStage/UnsupportedChunkSizeThrows    — chunk_size not in {4096,8192,16384} throws at execute()
 *   CL13  CLOGStage/UnsupportedWordSizeThrows     — word_size∉{1,2,4,8} throws at execute()
 *   CL14  CLOGStage/IsGraphCompatible             — forward=true, inverse=false
 *   CL15  CLOGStage/RepeatedRoundTripStable       — repeated round-trips on same objects stable
 *   CL16  CLOGStage/ChunkSize4096RandomBytesRoundTrip — chunk_size=4096, multi-chunk random bytes
 *   CL17  CLOGStage/ChunkSize8192PartialChunkRoundTrip — chunk_size=8192, input not a multiple of chunk_size
 */

#include <gtest/gtest.h>
#include "helpers/fz_test_utils.h"
#include "coders/clog/clog_stage.h"
#include "fzgpumodules.h"

#include <cstdint>
#include <vector>
#include <random>
#include <cstring>

using namespace fz;
using namespace fz_test;

// Run CLOGStage (encode or decode, per its is_inverse flag) on a byte vector.
static std::vector<uint8_t> run_clog(
    CLOGStage& stage, const std::vector<uint8_t>& h_in,
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

// Compress then decompress; verify byte-exact round-trip. Returns the
// compressed size so callers can additionally check compression engaged.
static size_t round_trip(const std::vector<uint8_t>& original, int word_size = 1,
                          size_t chunk_size = 16384) {
    CudaStream cs;
    auto pool = make_test_pool(original.size() + 65536);

    CLOGStage enc;
    enc.setChunkSize(chunk_size);
    enc.setWordSize(word_size);
    const size_t enc_cap = enc.estimateOutputSizes({original.size()})[0];
    const auto compressed = run_clog(enc, original, enc_cap, cs.stream, *pool);

    CLOGStage dec;
    dec.setChunkSize(chunk_size);
    dec.setWordSize(word_size);
    dec.setInverse(true);
    const auto restored = run_clog(dec, compressed, original.size() + 4096, cs.stream, *pool);

    EXPECT_EQ(restored.size(), original.size());
    EXPECT_EQ(restored, original) << "CLOG round-trip mismatch (word_size=" << word_size
                                   << ", chunk_size=" << chunk_size << ")";
    return compressed.size();
}

TEST(CLOGStage, RandomBytesRoundTrip) {
    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> dist(0, 255);
    std::vector<uint8_t> data(16384);
    for (auto& b : data) b = (uint8_t)dist(rng);
    round_trip(data);
}

TEST(CLOGStage, AllZerosRoundTrip) {
    const size_t compressed = round_trip(std::vector<uint8_t>(16384, 0));
    EXPECT_LT(compressed, (size_t)64) << "all-zero chunk (every subchunk needs 0 bits) did not compress tiny";
}

TEST(CLOGStage, SmallValueRunCompressesSmall) {
    std::vector<uint8_t> data(16384, 0x01);  // every subchunk's max needs exactly 1 bit
    const size_t compressed = round_trip(data);
    EXPECT_LT(compressed, data.size() / 4)
        << "small-magnitude constant did not compress — per-subchunk bit-width selection "
           "likely didn't engage";
}

TEST(CLOGStage, MultiChunkRoundTrip) {
    std::mt19937 rng(99);
    std::uniform_int_distribution<int> dist(0, 3);
    std::vector<uint8_t> data(4 * 16384);
    for (auto& b : data) b = (uint8_t)dist(rng);
    round_trip(data);
}

TEST(CLOGStage, PartialChunkRoundTrip) {
    std::mt19937 rng(3);
    std::uniform_int_distribution<int> dist(0, 255);
    std::vector<uint8_t> data(3000);
    for (auto& b : data) b = (uint8_t)dist(rng);
    round_trip(data);
}

TEST(CLOGStage, WordSize2RoundTrip) {
    std::mt19937 rng(21);
    std::uniform_int_distribution<int> dist(0, 255);
    std::vector<uint8_t> data(2 * 16384);
    for (auto& b : data) b = (uint8_t)dist(rng);
    round_trip(data, 2);
}

TEST(CLOGStage, WordSize4RoundTrip) {
    std::mt19937 rng(22);
    std::uniform_int_distribution<int> dist(0, 255);
    std::vector<uint8_t> data(2 * 16384);
    for (auto& b : data) b = (uint8_t)dist(rng);
    round_trip(data, 4);
}

TEST(CLOGStage, WordSize8RoundTrip) {
    std::mt19937 rng(23);
    std::uniform_int_distribution<int> dist(0, 255);
    std::vector<uint8_t> data(2 * 16384);
    for (auto& b : data) b = (uint8_t)dist(rng);
    round_trip(data, 8);
}

// CLOG splits each chunk into exactly 32 subchunks via integer division
// (beg = i*size/32, end = (i+1)*size/32) — when `size` (words per chunk)
// isn't a multiple of 32, subchunk boundaries are uneven. Neither encode nor
// decode has any other test forcing this: most tests above use full 16 KB
// chunks where size is comfortably a multiple of 32. A partial final chunk
// with an odd word count exercises it directly.
TEST(CLOGStage, UnevenSubchunkBoundaryRoundTrip) {
    std::mt19937 rng(77);
    std::uniform_int_distribution<int> dist(0, 255);
    std::vector<uint8_t> data(2013);  // 2013 words (word_size=1), not a multiple of 32
    for (auto& b : data) b = (uint8_t)dist(rng);
    round_trip(data);
}

// One subchunk needs the full TB bits (no truncation savings at all for it —
// exercises the CB-field encoding at its upper edge, logn == TB) while the
// rest of the chunk is all zero.
TEST(CLOGStage, MaxWidthSubchunkRoundTrip) {
    std::vector<uint8_t> data(16384, 0);
    // Fill the first subchunk's element range (words [0, 16384/32) = [0,512))
    // with the max representable uint8_t value.
    for (int i = 0; i < 512; i++) data[i] = 0xFF;
    round_trip(data);
}

TEST(CLOGStage, HeaderSerialization) {
    CLOGStage s;
    s.setChunkSize(16384);
    s.setWordSize(4);
    uint8_t buf[9] = {0};
    ASSERT_EQ(s.serializeHeader(0, buf, sizeof(buf)), (size_t)9);
    CLOGStage s2;
    s2.deserializeHeader(buf, sizeof(buf));
    EXPECT_EQ(s2.getChunkSize(), (size_t)16384);
    EXPECT_EQ(s2.getWordSize(), 4);
}

TEST(CLOGStage, UnsupportedChunkSizeThrows) {
    CudaStream cs;
    std::vector<uint8_t> data(4096, 1);
    auto pool = make_test_pool(data.size() + 65536);
    CLOGStage s;
    s.setChunkSize(12345);  // not in the supported set {4096, 8192, 16384}
    CudaBuffer<uint8_t> d_in(data.size()), d_out(data.size() + 4096);
    d_in.upload(data, cs.stream);
    cudaStreamSynchronize(cs.stream);
    std::vector<void*> in = {d_in.void_ptr()}, out = {d_out.void_ptr()};
    std::vector<size_t> sz = {data.size()};
    EXPECT_THROW(s.execute(cs.stream, pool.get(), in, out, sz), std::runtime_error);
}

TEST(CLOGStage, UnsupportedWordSizeThrows) {
    CudaStream cs;
    std::vector<uint8_t> data(4096, 1);
    auto pool = make_test_pool(data.size() + 65536);
    CLOGStage s;
    s.setWordSize(3);
    CudaBuffer<uint8_t> d_in(data.size()), d_out(data.size() + 4096);
    d_in.upload(data, cs.stream);
    cudaStreamSynchronize(cs.stream);
    std::vector<void*> in = {d_in.void_ptr()}, out = {d_out.void_ptr()};
    std::vector<size_t> sz = {data.size()};
    EXPECT_THROW(s.execute(cs.stream, pool.get(), in, out, sz), std::runtime_error);
}

TEST(CLOGStage, IsGraphCompatible) {
    CLOGStage fwd;
    EXPECT_TRUE(fwd.isGraphCompatible());
    CLOGStage inv;
    inv.setInverse(true);
    EXPECT_FALSE(inv.isGraphCompatible());
}

TEST(CLOGStage, RepeatedRoundTripStable) {
    std::mt19937 rng(55);
    std::uniform_int_distribution<int> dist(0, 255);
    std::vector<uint8_t> data(2 * 16384);
    for (auto& b : data) b = (uint8_t)dist(rng);
    for (int i = 0; i < 5; i++) round_trip(data);
}

TEST(CLOGStage, ChunkSize4096RandomBytesRoundTrip) {
    std::mt19937 rng(101);
    std::uniform_int_distribution<int> dist(0, 255);
    std::vector<uint8_t> data(3 * 4096);
    for (auto& b : data) b = (uint8_t)dist(rng);
    round_trip(data, 1, 4096);
}

TEST(CLOGStage, ChunkSize8192PartialChunkRoundTrip) {
    std::mt19937 rng(104);
    std::uniform_int_distribution<int> dist(0, 255);
    std::vector<uint8_t> data(2 * 8192 + 3000);
    for (auto& b : data) b = (uint8_t)dist(rng);
    round_trip(data, 1, 8192);
}
