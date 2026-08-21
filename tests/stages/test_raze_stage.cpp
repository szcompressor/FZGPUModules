/**
 * tests/stages/test_raze_stage.cpp
 *
 * GPU unit tests for the LC RAZE adaptive leading-zero-bit reducer. It
 * generalizes RZE by measuring how many top bits
 * of `word ^ predecessor` match across the chunk, picks one global `keep` cut
 * that maximizes savings, then bit-packs the bottom `keep` bits of every
 * matching word (non-matching words are stored in full, as in RZE). The
 * 4-level recursive bitmap compression is identical to RZE. Word granularity
 * is 1/2/4/8 bytes.
 *
 *   RZ1   RAZEStage/RandomBytesRoundTrip          — random bytes restore exactly
 *   RZ2   RAZEStage/AllZerosRoundTrip             — all-zeros input (all-repeat fast path)
 *   RZ3   RAZEStage/ConstantRunRoundTrip          — single repeated value (great RE target)
 *   RZ4   RAZEStage/LongRunsRoundTrip             — long repeated-byte runs round-trip
 *   RZ5   RAZEStage/MultiChunkRoundTrip           — 4×16 KB chunks restore exactly
 *   RZ6   RAZEStage/PartialChunkRoundTrip         — input < one chunk round-trips exactly
 *   RZ7   RAZEStage/WordSize2RoundTrip            — 2-byte word granularity round-trip
 *   RZ8   RAZEStage/WordSize4RoundTrip            — 4-byte word granularity round-trip
 *   RZ9   RAZEStage/WordSize8RoundTrip            — 8-byte word granularity round-trip
 *   RZ10  RAZEStage/SmallValueRunCompressesSmall  — small-magnitude constant run compresses far below input
 *   RZ11  RAZEStage/HeaderSerialization           — serializeHeader/deserializeHeader preserves config
 *   RZ12  RAZEStage/UnsupportedChunkSizeThrows    — chunk_size not in {4096,8192,16384} throws at execute()
 *   RZ13  RAZEStage/UnsupportedWordSizeThrows     — word_size∉{1,2,4,8} throws at execute()
 *   RZ14  RAZEStage/IsGraphCompatible             — forward=true, inverse=false
 *   RZ15  RAZEStage/RepeatedRoundTripStable       — repeated round-trips on same objects stable
 *   RZ16  RAZEStage/ChunkSize4096RandomBytesRoundTrip   — chunk_size=4096, multi-chunk random bytes
 *   RZ17  RAZEStage/ChunkSize8192PartialChunkRoundTrip  — chunk_size=8192, input not a multiple of chunk_size
 *   RZ18  RAZEStage/PartialKeepRoundTrip          — engineered data where the optimal keep is
 *                                                    strictly between 0 and bits, exercising the
 *                                                    bit-packed partial-value path RZE never takes
 *   RZ19  RAZEStage/PartialKeepCompressesSmaller  — same data: confirms the partial-keep path
 *                                                    actually engaged (output shrinks), not a raw fallback
 */

#include <gtest/gtest.h>
#include "helpers/fz_test_utils.h"
#include "coders/raze/raze_stage.h"
#include "fzgpumodules.h"

#include <cstdint>
#include <vector>
#include <random>
#include <cstring>

using namespace fz;
using namespace fz_test;

// Run RAZEStage (encode or decode, per its is_inverse flag) on a byte vector.
static std::vector<uint8_t> run_raze(
    RAZEStage& stage, const std::vector<uint8_t>& h_in,
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

    RAZEStage enc;
    enc.setChunkSize(chunk_size);
    enc.setWordSize(word_size);
    const size_t enc_cap = enc.estimateOutputSizes({original.size()})[0];
    const auto compressed = run_raze(enc, original, enc_cap, cs.stream, *pool);

    RAZEStage dec;
    dec.setChunkSize(chunk_size);
    dec.setWordSize(word_size);
    dec.setInverse(true);
    const auto restored = run_raze(dec, compressed, original.size() + 4096, cs.stream, *pool);

    EXPECT_EQ(restored.size(), original.size());
    EXPECT_EQ(restored, original) << "RAZE round-trip mismatch (word_size=" << word_size
                                   << ", chunk_size=" << chunk_size << ")";
    return compressed.size();
}

TEST(RAZEStage, RandomBytesRoundTrip) {
    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> dist(0, 255);
    std::vector<uint8_t> data(16384);
    for (auto& b : data) b = (uint8_t)dist(rng);
    round_trip(data);
}

TEST(RAZEStage, AllZerosRoundTrip) {
    round_trip(std::vector<uint8_t>(16384, 0));
}

TEST(RAZEStage, ConstantRunRoundTrip) {
    round_trip(std::vector<uint8_t>(16384, 0x5A));
}

TEST(RAZEStage, LongRunsRoundTrip) {
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

TEST(RAZEStage, MultiChunkRoundTrip) {
    std::mt19937 rng(99);
    std::uniform_int_distribution<int> dist(0, 3);  // low entropy → repetitions
    std::vector<uint8_t> data(4 * 16384);
    for (auto& b : data) b = (uint8_t)dist(rng);
    round_trip(data);
}

TEST(RAZEStage, PartialChunkRoundTrip) {
    std::mt19937 rng(3);
    std::uniform_int_distribution<int> dist(0, 5);
    std::vector<uint8_t> data(3000);
    for (auto& b : data) b = (uint8_t)dist(rng);
    round_trip(data);
}

TEST(RAZEStage, WordSize2RoundTrip) {
    std::mt19937 rng(21);
    std::uniform_int_distribution<int> dist(0, 7);
    std::vector<uint8_t> data(2 * 16384);
    for (auto& b : data) b = (uint8_t)dist(rng);
    round_trip(data, 2);
}

TEST(RAZEStage, WordSize4RoundTrip) {
    std::mt19937 rng(22);
    std::uniform_int_distribution<int> dist(0, 7);
    std::vector<uint8_t> data(2 * 16384);
    for (auto& b : data) b = (uint8_t)dist(rng);
    round_trip(data, 4);
}

TEST(RAZEStage, WordSize8RoundTrip) {
    std::mt19937 rng(23);
    std::uniform_int_distribution<int> dist(0, 7);
    std::vector<uint8_t> data(2 * 16384);
    for (auto& b : data) b = (uint8_t)dist(rng);
    round_trip(data, 8);
}

// Unlike RRE/RARE, RAZE has no notion of "repeats the previous element" — its
// predicate is purely each element's own leading-zero count. A repeated
// *nonzero* value (e.g. the RRE/RARE-style ConstantRunCompressesSmall test)
// doesn't hit RAZE's fast path at all; it just picks a global `keep` equal to
// that value's own bit-width and packs everything at that width (a real but
// modest win, not a near-total elimination). RAZE's actual strength case —
// what it can do that RZE cannot — is a small-but-nonzero constant value,
// where the global `keep` is tiny.
TEST(RAZEStage, SmallValueRunCompressesSmall) {
    CudaStream cs;
    std::vector<uint8_t> data(16384, 0x01);  // needs exactly 1 significant bit
    auto pool = make_test_pool(data.size() + 65536);
    RAZEStage enc;
    enc.setChunkSize(16384);
    enc.setWordSize(1);
    const size_t enc_cap = enc.estimateOutputSizes({data.size()})[0];
    const auto compressed = run_raze(enc, data, enc_cap, cs.stream, *pool);
    // keep=1 packs every element into ~1/8th its original width; the bitmap
    // (every element takes the "matched" path) compresses to near nothing.
    EXPECT_LT(compressed.size(), data.size() / 4)
        << "small-value run did not compress — the auto-k keep selection "
           "likely didn't engage RAZE's advantage over plain RZE";
}

TEST(RAZEStage, HeaderSerialization) {
    RAZEStage s;
    s.setChunkSize(16384);
    s.setWordSize(4);
    uint8_t buf[9] = {0};
    ASSERT_EQ(s.serializeHeader(0, buf, sizeof(buf)), (size_t)9);
    RAZEStage s2;
    s2.deserializeHeader(buf, sizeof(buf));
    EXPECT_EQ(s2.getChunkSize(), (size_t)16384);
    EXPECT_EQ(s2.getWordSize(), 4);
}

TEST(RAZEStage, UnsupportedChunkSizeThrows) {
    CudaStream cs;
    std::vector<uint8_t> data(4096, 1);
    auto pool = make_test_pool(data.size() + 65536);
    RAZEStage s;
    s.setChunkSize(12345);  // not in the supported set {4096, 8192, 16384}
    CudaBuffer<uint8_t> d_in(data.size()), d_out(data.size() + 4096);
    d_in.upload(data, cs.stream);
    cudaStreamSynchronize(cs.stream);
    std::vector<void*> in = {d_in.void_ptr()}, out = {d_out.void_ptr()};
    std::vector<size_t> sz = {data.size()};
    EXPECT_THROW(s.execute(cs.stream, pool.get(), in, out, sz), std::runtime_error);
}

TEST(RAZEStage, UnsupportedWordSizeThrows) {
    CudaStream cs;
    std::vector<uint8_t> data(4096, 1);
    auto pool = make_test_pool(data.size() + 65536);
    RAZEStage s;
    s.setWordSize(3);
    CudaBuffer<uint8_t> d_in(data.size()), d_out(data.size() + 4096);
    d_in.upload(data, cs.stream);
    cudaStreamSynchronize(cs.stream);
    std::vector<void*> in = {d_in.void_ptr()}, out = {d_out.void_ptr()};
    std::vector<size_t> sz = {data.size()};
    EXPECT_THROW(s.execute(cs.stream, pool.get(), in, out, sz), std::runtime_error);
}

TEST(RAZEStage, IsGraphCompatible) {
    RAZEStage fwd;
    EXPECT_TRUE(fwd.isGraphCompatible());
    RAZEStage inv;
    inv.setInverse(true);
    EXPECT_FALSE(inv.isGraphCompatible());
}

TEST(RAZEStage, RepeatedRoundTripStable) {
    std::mt19937 rng(55);
    std::uniform_int_distribution<int> dist(0, 4);
    std::vector<uint8_t> data(2 * 16384);
    for (auto& b : data) b = (uint8_t)dist(rng);
    for (int i = 0; i < 5; i++) round_trip(data);
}

TEST(RAZEStage, ChunkSize4096RandomBytesRoundTrip) {
    std::mt19937 rng(101);
    std::uniform_int_distribution<int> dist(0, 255);
    std::vector<uint8_t> data(3 * 4096);  // multi-chunk
    for (auto& b : data) b = (uint8_t)dist(rng);
    round_trip(data, 1, 4096);
}

TEST(RAZEStage, ChunkSize8192PartialChunkRoundTrip) {
    std::mt19937 rng(104);
    std::uniform_int_distribution<int> dist(0, 5);
    std::vector<uint8_t> data(2 * 8192 + 3000);  // not a multiple of chunk_size
    for (auto& b : data) b = (uint8_t)dist(rng);
    round_trip(data, 1, 8192);
}

// Engineered so the optimal `keep` lands strictly between 0 and bits: every
// element (word_size=2) is a random value in [128, 255] — bit 7 always set,
// bits 8-15 always zero — so every element has exactly 8 leading zero bits.
// The histogram concentrates at keep=8, which RZE's binary zero-or-full test
// can never represent (it only ever sees keep=0 for an exact-zero word or a
// full-word store). RZE's own test suite never exercises this bit-packed
// partial-value code path at all.
static std::vector<uint8_t> make_partial_keep_data(size_t n_words, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> val(128, 255);
    std::vector<uint8_t> data(n_words * 2);
    for (size_t i = 0; i < n_words; i++) {
        uint16_t v = (uint16_t)val(rng);
        std::memcpy(&data[i * 2], &v, 2);
    }
    return data;
}

TEST(RAZEStage, PartialKeepRoundTrip) {
    auto data = make_partial_keep_data(4 * 16384 / 2, 201);
    round_trip(data, 2);
}

TEST(RAZEStage, PartialKeepCompressesSmaller) {
    auto data = make_partial_keep_data(4 * 16384 / 2, 202);
    const size_t compressed = round_trip(data, 2);
    EXPECT_LT(compressed, data.size())
        << "partial-keep data did not compress — the bit-packed partial-value "
           "path likely fell back to raw storage instead of engaging";
}
