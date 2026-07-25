/**
 * tests/stages/test_rare_stage.cpp
 *
 * GPU unit tests for RAREStage — Repetition-Adaptive Reduction Encoding (LC
 * component). The auto-k generalization of RRE: histograms how many top bits
 * of `word ^ predecessor` match across the chunk, picks one global `keep` cut
 * that maximizes savings, then bit-packs the bottom `keep` bits of every
 * matching word (non-matching words are stored in full, as in RRE). The
 * 4-level recursive bitmap compression is identical to RRE. Word granularity
 * is 1/2/4/8 bytes.
 *
 *   RA1   RAREStage/RandomBytesRoundTrip          — random bytes restore exactly
 *   RA2   RAREStage/AllZerosRoundTrip             — all-zeros input (all-repeat fast path)
 *   RA3   RAREStage/ConstantRunRoundTrip          — single repeated value (great RE target)
 *   RA4   RAREStage/LongRunsRoundTrip             — long repeated-byte runs round-trip
 *   RA5   RAREStage/MultiChunkRoundTrip           — 4×16 KB chunks restore exactly
 *   RA6   RAREStage/PartialChunkRoundTrip         — input < one chunk round-trips exactly
 *   RA7   RAREStage/WordSize2RoundTrip            — 2-byte word granularity round-trip
 *   RA8   RAREStage/WordSize4RoundTrip            — 4-byte word granularity round-trip
 *   RA9   RAREStage/WordSize8RoundTrip            — 8-byte word granularity round-trip
 *   RA10  RAREStage/ConstantRunCompressesSmall    — constant run compresses far below input
 *   RA11  RAREStage/HeaderSerialization           — serializeHeader/deserializeHeader preserves config
 *   RA12  RAREStage/UnsupportedChunkSizeThrows    — chunk_size not in {4096,8192,16384} throws at execute()
 *   RA13  RAREStage/UnsupportedWordSizeThrows     — word_size∉{1,2,4,8} throws at execute()
 *   RA14  RAREStage/IsGraphCompatible             — forward=true, inverse=false
 *   RA15  RAREStage/RepeatedRoundTripStable       — repeated round-trips on same objects stable
 *   RA16  RAREStage/ChunkSize4096RandomBytesRoundTrip   — chunk_size=4096, multi-chunk random bytes
 *   RA17  RAREStage/ChunkSize8192PartialChunkRoundTrip  — chunk_size=8192, input not a multiple of chunk_size
 *   RA18  RAREStage/PartialKeepRoundTrip          — engineered data where the optimal keep is
 *                                                    strictly between 0 and bits, exercising the
 *                                                    bit-packed partial-value path RRE never takes
 *   RA19  RAREStage/PartialKeepCompressesSmaller  — same data: confirms the partial-keep path
 *                                                    actually engaged (output shrinks), not a raw fallback
 */

#include <gtest/gtest.h>
#include "helpers/fz_test_utils.h"
#include "coders/rare/rare_stage.h"
#include "fzgpumodules.h"

#include <cstdint>
#include <vector>
#include <random>
#include <cstring>

using namespace fz;
using namespace fz_test;

// Run RAREStage (encode or decode, per its is_inverse flag) on a byte vector.
static std::vector<uint8_t> run_rare(
    RAREStage& stage, const std::vector<uint8_t>& h_in,
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

    RAREStage enc;
    enc.setChunkSize(chunk_size);
    enc.setWordSize(word_size);
    const size_t enc_cap = enc.estimateOutputSizes({original.size()})[0];
    const auto compressed = run_rare(enc, original, enc_cap, cs.stream, *pool);

    RAREStage dec;
    dec.setChunkSize(chunk_size);
    dec.setWordSize(word_size);
    dec.setInverse(true);
    const auto restored = run_rare(dec, compressed, original.size() + 4096, cs.stream, *pool);

    EXPECT_EQ(restored.size(), original.size());
    EXPECT_EQ(restored, original) << "RARE round-trip mismatch (word_size=" << word_size
                                   << ", chunk_size=" << chunk_size << ")";
    return compressed.size();
}

TEST(RAREStage, RandomBytesRoundTrip) {
    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> dist(0, 255);
    std::vector<uint8_t> data(16384);
    for (auto& b : data) b = (uint8_t)dist(rng);
    round_trip(data);
}

TEST(RAREStage, AllZerosRoundTrip) {
    round_trip(std::vector<uint8_t>(16384, 0));
}

TEST(RAREStage, ConstantRunRoundTrip) {
    round_trip(std::vector<uint8_t>(16384, 0x5A));
}

TEST(RAREStage, LongRunsRoundTrip) {
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

TEST(RAREStage, MultiChunkRoundTrip) {
    std::mt19937 rng(99);
    std::uniform_int_distribution<int> dist(0, 3);  // low entropy → repetitions
    std::vector<uint8_t> data(4 * 16384);
    for (auto& b : data) b = (uint8_t)dist(rng);
    round_trip(data);
}

TEST(RAREStage, PartialChunkRoundTrip) {
    std::mt19937 rng(3);
    std::uniform_int_distribution<int> dist(0, 5);
    std::vector<uint8_t> data(3000);
    for (auto& b : data) b = (uint8_t)dist(rng);
    round_trip(data);
}

TEST(RAREStage, WordSize2RoundTrip) {
    std::mt19937 rng(21);
    std::uniform_int_distribution<int> dist(0, 7);
    std::vector<uint8_t> data(2 * 16384);
    for (auto& b : data) b = (uint8_t)dist(rng);
    round_trip(data, 2);
}

TEST(RAREStage, WordSize4RoundTrip) {
    std::mt19937 rng(22);
    std::uniform_int_distribution<int> dist(0, 7);
    std::vector<uint8_t> data(2 * 16384);
    for (auto& b : data) b = (uint8_t)dist(rng);
    round_trip(data, 4);
}

TEST(RAREStage, WordSize8RoundTrip) {
    std::mt19937 rng(23);
    std::uniform_int_distribution<int> dist(0, 7);
    std::vector<uint8_t> data(2 * 16384);
    for (auto& b : data) b = (uint8_t)dist(rng);
    round_trip(data, 8);
}

TEST(RAREStage, ConstantRunCompressesSmall) {
    CudaStream cs;
    std::vector<uint8_t> data(16384, 0x42);
    auto pool = make_test_pool(data.size() + 65536);
    RAREStage enc;
    enc.setChunkSize(16384);
    enc.setWordSize(1);
    const size_t enc_cap = enc.estimateOutputSizes({data.size()})[0];
    const auto compressed = run_rare(enc, data, enc_cap, cs.stream, *pool);
    // header (12 bytes for 1 chunk) + a couple of payload bytes
    EXPECT_LT(compressed.size(), (size_t)64) << "constant run did not compress";
}

TEST(RAREStage, HeaderSerialization) {
    RAREStage s;
    s.setChunkSize(16384);
    s.setWordSize(4);
    uint8_t buf[9] = {0};
    ASSERT_EQ(s.serializeHeader(0, buf, sizeof(buf)), (size_t)9);
    RAREStage s2;
    s2.deserializeHeader(buf, sizeof(buf));
    EXPECT_EQ(s2.getChunkSize(), (size_t)16384);
    EXPECT_EQ(s2.getWordSize(), 4);
}

TEST(RAREStage, UnsupportedChunkSizeThrows) {
    CudaStream cs;
    std::vector<uint8_t> data(4096, 1);
    auto pool = make_test_pool(data.size() + 65536);
    RAREStage s;
    s.setChunkSize(12345);  // not in the supported set {4096, 8192, 16384}
    CudaBuffer<uint8_t> d_in(data.size()), d_out(data.size() + 4096);
    d_in.upload(data, cs.stream);
    cudaStreamSynchronize(cs.stream);
    std::vector<void*> in = {d_in.void_ptr()}, out = {d_out.void_ptr()};
    std::vector<size_t> sz = {data.size()};
    EXPECT_THROW(s.execute(cs.stream, pool.get(), in, out, sz), std::runtime_error);
}

TEST(RAREStage, UnsupportedWordSizeThrows) {
    CudaStream cs;
    std::vector<uint8_t> data(4096, 1);
    auto pool = make_test_pool(data.size() + 65536);
    RAREStage s;
    s.setWordSize(3);
    CudaBuffer<uint8_t> d_in(data.size()), d_out(data.size() + 4096);
    d_in.upload(data, cs.stream);
    cudaStreamSynchronize(cs.stream);
    std::vector<void*> in = {d_in.void_ptr()}, out = {d_out.void_ptr()};
    std::vector<size_t> sz = {data.size()};
    EXPECT_THROW(s.execute(cs.stream, pool.get(), in, out, sz), std::runtime_error);
}

TEST(RAREStage, IsGraphCompatible) {
    RAREStage fwd;
    EXPECT_TRUE(fwd.isGraphCompatible());
    RAREStage inv;
    inv.setInverse(true);
    EXPECT_FALSE(inv.isGraphCompatible());
}

TEST(RAREStage, RepeatedRoundTripStable) {
    std::mt19937 rng(55);
    std::uniform_int_distribution<int> dist(0, 4);
    std::vector<uint8_t> data(2 * 16384);
    for (auto& b : data) b = (uint8_t)dist(rng);
    for (int i = 0; i < 5; i++) round_trip(data);
}

TEST(RAREStage, ChunkSize4096RandomBytesRoundTrip) {
    std::mt19937 rng(101);
    std::uniform_int_distribution<int> dist(0, 255);
    std::vector<uint8_t> data(3 * 4096);  // multi-chunk
    for (auto& b : data) b = (uint8_t)dist(rng);
    round_trip(data, 1, 4096);
}

TEST(RAREStage, ChunkSize8192PartialChunkRoundTrip) {
    std::mt19937 rng(104);
    std::uniform_int_distribution<int> dist(0, 5);
    std::vector<uint8_t> data(2 * 8192 + 3000);  // not a multiple of chunk_size
    for (auto& b : data) b = (uint8_t)dist(rng);
    round_trip(data, 1, 8192);
}

// Engineered so the optimal `keep` lands strictly between 0 and bits: every
// element is (constant_top_byte << 8) | random_low_byte for word_size=2, so
// consecutive elements' XOR is always confined to the low 8 bits (the top
// byte cancels) — the histogram concentrates at keep=8, which RRE's binary
// repeat-or-drop test can never represent (it only ever sees keep=0 or a
// full-word mismatch). Neither RRE nor RZE's test suites exercise this
// bit-packed partial-value code path at all.
static std::vector<uint8_t> make_partial_keep_data(size_t n_words, uint8_t const_top, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> low(0, 255);
    std::vector<uint8_t> data(n_words * 2);
    for (size_t i = 0; i < n_words; i++) {
        uint16_t v = (uint16_t)((const_top << 8) | low(rng));
        std::memcpy(&data[i * 2], &v, 2);
    }
    return data;
}

TEST(RAREStage, PartialKeepRoundTrip) {
    auto data = make_partial_keep_data(4 * 16384 / 2, 0x12, 201);
    round_trip(data, 2);
}

TEST(RAREStage, PartialKeepCompressesSmaller) {
    auto data = make_partial_keep_data(4 * 16384 / 2, 0x12, 202);
    const size_t compressed = round_trip(data, 2);
    EXPECT_LT(compressed, data.size())
        << "partial-keep data did not compress — the bit-packed partial-value "
           "path likely fell back to raw storage instead of engaging";
}
