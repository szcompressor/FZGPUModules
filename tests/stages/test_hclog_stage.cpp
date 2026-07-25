/**
 * tests/stages/test_hclog_stage.cpp
 *
 * GPU unit tests for HCLOGStage — Compressed-Logarithm adaptive bit-width
 * coding with a per-subchunk TCMS(zigzag) fallback (LC component, the
 * auto-selecting sibling of CLOGStage). Splits each chunk into a fixed 32
 * subchunks; each subchunk finds its own max value — both the raw magnitude
 * and the TCMS-reinterpreted magnitude — and bit-packs every element in it to
 * the smaller of the two minimum bit-widths, recording the choice as one flag
 * bit per subchunk. Unlike RRE/RARE/RZE/RAZE there is no bitmap and no
 * per-element full/dropped decision — every element in a subchunk shares the
 * same packed width (and the same TCMS-or-not choice). `T` must be unsigned.
 * Word granularity is 1/2/4/8 bytes.
 *
 *   HCL1   HCLOGStage/RandomBytesRoundTrip          — random bytes restore exactly
 *   HCL2   HCLOGStage/AllZerosRoundTrip             — every subchunk needs 0 bits
 *   HCL3   HCLOGStage/SmallValueRunCompressesSmall  — small constant compresses far below input
 *   HCL4   HCLOGStage/MultiChunkRoundTrip           — 4×16 KB chunks restore exactly
 *   HCL5   HCLOGStage/PartialChunkRoundTrip         — input < one chunk round-trips exactly
 *   HCL6   HCLOGStage/WordSize2RoundTrip            — 2-byte word granularity round-trip
 *   HCL7   HCLOGStage/WordSize4RoundTrip            — 4-byte word granularity round-trip
 *   HCL8   HCLOGStage/WordSize8RoundTrip            — 8-byte word granularity round-trip
 *   HCL9   HCLOGStage/UnevenSubchunkBoundaryRoundTrip — size not a multiple of 32 words,
 *                                                      forcing uneven beg/end subchunk splits
 *   HCL10  HCLOGStage/MaxWidthSubchunkRoundTrip     — a subchunk needing the full TB bits
 *                                                    mixed with all-zero subchunks
 *   HCL11  HCLOGStage/HeaderSerialization           — serializeHeader/deserializeHeader preserves config
 *   HCL12  HCLOGStage/UnsupportedChunkSizeThrows    — chunk_size not in {4096,8192,16384} throws at execute()
 *   HCL13  HCLOGStage/UnsupportedWordSizeThrows     — word_size∉{1,2,4,8} throws at execute()
 *   HCL14  HCLOGStage/IsGraphCompatible             — forward=true, inverse=false
 *   HCL15  HCLOGStage/RepeatedRoundTripStable       — repeated round-trips on same objects stable
 *   HCL16  HCLOGStage/ChunkSize4096RandomBytesRoundTrip — chunk_size=4096, multi-chunk random bytes
 *   HCL17  HCLOGStage/ChunkSize8192PartialChunkRoundTrip — chunk_size=8192, input not a multiple of chunk_size
 *   HCL18  HCLOGStage/TCMSFallbackRoundTrip         — bipolar-looking data (values near the unsigned
 *                                                      range's top and bottom edges) round-trips exactly
 *   HCL19  HCLOGStage/TCMSFallbackCompressesSmallerThanRawWidth — same data: confirms the TCMS
 *                                                      reinterpretation actually engaged (compresses
 *                                                      far better than the raw-magnitude bit-width would)
 */

#include <gtest/gtest.h>
#include "helpers/fz_test_utils.h"
#include "coders/hclog/hclog_stage.h"
#include "fzgpumodules.h"

#include <cstdint>
#include <vector>
#include <random>
#include <cstring>

using namespace fz;
using namespace fz_test;

// Run HCLOGStage (encode or decode, per its is_inverse flag) on a byte vector.
static std::vector<uint8_t> run_hclog(
    HCLOGStage& stage, const std::vector<uint8_t>& h_in,
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

    HCLOGStage enc;
    enc.setChunkSize(chunk_size);
    enc.setWordSize(word_size);
    const size_t enc_cap = enc.estimateOutputSizes({original.size()})[0];
    const auto compressed = run_hclog(enc, original, enc_cap, cs.stream, *pool);

    HCLOGStage dec;
    dec.setChunkSize(chunk_size);
    dec.setWordSize(word_size);
    dec.setInverse(true);
    const auto restored = run_hclog(dec, compressed, original.size() + 4096, cs.stream, *pool);

    EXPECT_EQ(restored.size(), original.size());
    EXPECT_EQ(restored, original) << "HCLOG round-trip mismatch (word_size=" << word_size
                                   << ", chunk_size=" << chunk_size << ")";
    return compressed.size();
}

TEST(HCLOGStage, RandomBytesRoundTrip) {
    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> dist(0, 255);
    std::vector<uint8_t> data(16384);
    for (auto& b : data) b = (uint8_t)dist(rng);
    round_trip(data);
}

TEST(HCLOGStage, AllZerosRoundTrip) {
    const size_t compressed = round_trip(std::vector<uint8_t>(16384, 0));
    EXPECT_LT(compressed, (size_t)64) << "all-zero chunk (every subchunk needs 0 bits) did not compress tiny";
}

TEST(HCLOGStage, SmallValueRunCompressesSmall) {
    std::vector<uint8_t> data(16384, 0x01);  // every subchunk's max needs exactly 1 bit
    const size_t compressed = round_trip(data);
    EXPECT_LT(compressed, data.size() / 4)
        << "small-magnitude constant did not compress — per-subchunk bit-width selection "
           "likely didn't engage";
}

TEST(HCLOGStage, MultiChunkRoundTrip) {
    std::mt19937 rng(99);
    std::uniform_int_distribution<int> dist(0, 3);
    std::vector<uint8_t> data(4 * 16384);
    for (auto& b : data) b = (uint8_t)dist(rng);
    round_trip(data);
}

TEST(HCLOGStage, PartialChunkRoundTrip) {
    std::mt19937 rng(3);
    std::uniform_int_distribution<int> dist(0, 255);
    std::vector<uint8_t> data(3000);
    for (auto& b : data) b = (uint8_t)dist(rng);
    round_trip(data);
}

TEST(HCLOGStage, WordSize2RoundTrip) {
    std::mt19937 rng(21);
    std::uniform_int_distribution<int> dist(0, 255);
    std::vector<uint8_t> data(2 * 16384);
    for (auto& b : data) b = (uint8_t)dist(rng);
    round_trip(data, 2);
}

TEST(HCLOGStage, WordSize4RoundTrip) {
    std::mt19937 rng(22);
    std::uniform_int_distribution<int> dist(0, 255);
    std::vector<uint8_t> data(2 * 16384);
    for (auto& b : data) b = (uint8_t)dist(rng);
    round_trip(data, 4);
}

TEST(HCLOGStage, WordSize8RoundTrip) {
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
TEST(HCLOGStage, UnevenSubchunkBoundaryRoundTrip) {
    std::mt19937 rng(77);
    std::uniform_int_distribution<int> dist(0, 255);
    std::vector<uint8_t> data(2013);  // 2013 words (word_size=1), not a multiple of 32
    for (auto& b : data) b = (uint8_t)dist(rng);
    round_trip(data);
}

// One subchunk needs the full TB bits (no truncation savings at all for it —
// exercises the CB-field encoding at its upper edge, logn == TB) while the
// rest of the chunk is all zero.
TEST(HCLOGStage, MaxWidthSubchunkRoundTrip) {
    std::vector<uint8_t> data(16384, 0);
    // Fill the first subchunk's element range (words [0, 16384/32) = [0,512))
    // with the max representable uint8_t value.
    for (int i = 0; i < 512; i++) data[i] = 0xFF;
    round_trip(data);
}

TEST(HCLOGStage, HeaderSerialization) {
    HCLOGStage s;
    s.setChunkSize(16384);
    s.setWordSize(4);
    uint8_t buf[9] = {0};
    ASSERT_EQ(s.serializeHeader(0, buf, sizeof(buf)), (size_t)9);
    HCLOGStage s2;
    s2.deserializeHeader(buf, sizeof(buf));
    EXPECT_EQ(s2.getChunkSize(), (size_t)16384);
    EXPECT_EQ(s2.getWordSize(), 4);
}

TEST(HCLOGStage, UnsupportedChunkSizeThrows) {
    CudaStream cs;
    std::vector<uint8_t> data(4096, 1);
    auto pool = make_test_pool(data.size() + 65536);
    HCLOGStage s;
    s.setChunkSize(12345);  // not in the supported set {4096, 8192, 16384}
    CudaBuffer<uint8_t> d_in(data.size()), d_out(data.size() + 4096);
    d_in.upload(data, cs.stream);
    cudaStreamSynchronize(cs.stream);
    std::vector<void*> in = {d_in.void_ptr()}, out = {d_out.void_ptr()};
    std::vector<size_t> sz = {data.size()};
    EXPECT_THROW(s.execute(cs.stream, pool.get(), in, out, sz), std::runtime_error);
}

TEST(HCLOGStage, UnsupportedWordSizeThrows) {
    CudaStream cs;
    std::vector<uint8_t> data(4096, 1);
    auto pool = make_test_pool(data.size() + 65536);
    HCLOGStage s;
    s.setWordSize(3);
    CudaBuffer<uint8_t> d_in(data.size()), d_out(data.size() + 4096);
    d_in.upload(data, cs.stream);
    cudaStreamSynchronize(cs.stream);
    std::vector<void*> in = {d_in.void_ptr()}, out = {d_out.void_ptr()};
    std::vector<size_t> sz = {data.size()};
    EXPECT_THROW(s.execute(cs.stream, pool.get(), in, out, sz), std::runtime_error);
}

TEST(HCLOGStage, IsGraphCompatible) {
    HCLOGStage fwd;
    EXPECT_TRUE(fwd.isGraphCompatible());
    HCLOGStage inv;
    inv.setInverse(true);
    EXPECT_FALSE(inv.isGraphCompatible());
}

TEST(HCLOGStage, RepeatedRoundTripStable) {
    std::mt19937 rng(55);
    std::uniform_int_distribution<int> dist(0, 255);
    std::vector<uint8_t> data(2 * 16384);
    for (auto& b : data) b = (uint8_t)dist(rng);
    for (int i = 0; i < 5; i++) round_trip(data);
}

TEST(HCLOGStage, ChunkSize4096RandomBytesRoundTrip) {
    std::mt19937 rng(101);
    std::uniform_int_distribution<int> dist(0, 255);
    std::vector<uint8_t> data(3 * 4096);
    for (auto& b : data) b = (uint8_t)dist(rng);
    round_trip(data, 1, 4096);
}

TEST(HCLOGStage, ChunkSize8192PartialChunkRoundTrip) {
    std::mt19937 rng(104);
    std::uniform_int_distribution<int> dist(0, 255);
    std::vector<uint8_t> data(2 * 8192 + 3000);
    for (auto& b : data) b = (uint8_t)dist(rng);
    round_trip(data, 1, 8192);
}

// Engineered so raw-magnitude bit-packing (plain CLOG) does badly but TCMS
// (zigzag) reinterpretation does well: every byte is the two's-complement bit
// pattern of a small-magnitude *signed* int8_t in [-8, 8]. A negative value's
// raw unsigned magnitude is near 255 (e.g. -8 = 0xF8 = 248), so any subchunk
// containing a negative element (virtually certain across 512 random draws)
// needs the full 8 raw bits — no compression at all via the raw path. TCMS
// (zigzag) maps small |value| regardless of sign to a small code
// (zigzag(-8)=15), needing only 4 bits. If HCLOG's per-subchunk TCMS-or-raw
// selection is working, it picks TCMS here and the output shrinks to roughly
// half; if the flag selection or the inverse-TCMS decode path is broken, this
// either fails to round-trip or compresses no better than plain CLOG would.
static std::vector<uint8_t> make_bipolar_small_magnitude_data(size_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(-8, 8);
    std::vector<uint8_t> data(n);
    for (auto& b : data) {
        const int8_t v = (int8_t)dist(rng);
        b = (uint8_t)v;  // raw two's-complement bit pattern
    }
    return data;
}

TEST(HCLOGStage, TCMSFallbackRoundTrip) {
    auto data = make_bipolar_small_magnitude_data(16384, 301);
    round_trip(data);
}

TEST(HCLOGStage, TCMSFallbackCompressesSmallerThanRawWidth) {
    auto data = make_bipolar_small_magnitude_data(16384, 302);
    const size_t compressed = round_trip(data);
    // TCMS path: zigzag(+8) = 16 needs 5 bits/element -> ~10240 bytes + header.
    // Raw-only path: a negative value's raw unsigned magnitude is near 255
    // (e.g. -8 = 0xF8 = 248), so any subchunk containing one (virtually
    // certain across 512 random draws) needs the full 8 bits/element -> no
    // compression at all (~16384 bytes). A threshold well between the two
    // catches either a broken flag selection or a broken inverse-TCMS decode.
    EXPECT_LT(compressed, (size_t)12000)
        << "HCLOG did not compress as if the TCMS fallback engaged — expected "
           "roughly the 5-bit-per-element packed size (~10 KB, not ~16 KB)";
}
