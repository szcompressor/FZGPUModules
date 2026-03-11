/**
 * tests/test_bitshuffle_stage.cpp
 *
 * GPU unit tests for BitshuffleStage — bit-matrix transpose stage.
 *
 * BitshuffleStage transposes the bit planes of a chunk of fixed-width integers:
 *   Forward (encode): bit-matrix transpose, producing W bit-planes each of N
 *                     bits (where W = element_width*8, N = chunk elements).
 *   Inverse (decode): reverse transpose, restoring the original data.
 *
 * Properties verified:
 *   1.  encode → decode round-trip restores original data (uint32).
 *   2.  encode → decode round-trip for uint8 elements.
 *   3.  encode → decode round-trip for uint16 elements.
 *   4.  encode → decode round-trip for uint64 elements.
 *   5.  Size-preserving: output byte count equals input byte count.
 *   6.  Transform is not a no-op: non-trivial input produces different output.
 *   7.  All-zeros input round-trips correctly (and output is all-zeros).
 *   8.  All-ones (0xFF...) input round-trips correctly.
 *   9.  Multi-chunk round-trip: multiple 16 KB blocks processed correctly.
 *   10. Bit-plane ordering: the MSBit plane of all elements is stored first;
 *       verify by encoding a known pattern and checking the output uint32.
 *   11. Header serialization round-trips block_size and element_width.
 */

#include <gtest/gtest.h>
#include "helpers/fz_test_utils.h"
#include "transforms/bitshuffle/bitshuffle_stage.h"

#include <cstdint>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cstring>

using namespace fz;
using namespace fz_test;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

// Run BitshuffleStage (encode or decode) on a raw byte vector.
// The stage is assumed to already have is_inverse set appropriately.
static std::vector<uint8_t> run_bitshuffle(
    BitshuffleStage& stage,
    const std::vector<uint8_t>& h_in,
    cudaStream_t stream,
    fz::MemoryPool& pool)
{
    const size_t n_bytes = h_in.size();

    CudaBuffer<uint8_t> d_in(n_bytes);
    CudaBuffer<uint8_t> d_out(n_bytes);
    d_in.upload(h_in, stream);

    std::vector<void*> inputs  = {d_in.void_ptr()};
    std::vector<void*> outputs = {d_out.void_ptr()};
    std::vector<size_t> sizes  = {n_bytes};

    stage.execute(stream, &pool, inputs, outputs, sizes);
    cudaError_t err = cudaStreamSynchronize(stream);
    EXPECT_EQ(err, cudaSuccess) << "CUDA sync: " << cudaGetErrorString(err);

    return d_out.download(stream);
}

// Build a ramp of bytes: [0, 1, 2, ..., 255, 0, 1, ...] wrapping mod 256.
static std::vector<uint8_t> make_ramp_bytes(size_t n_bytes) {
    std::vector<uint8_t> v(n_bytes);
    for (size_t i = 0; i < n_bytes; ++i) v[i] = static_cast<uint8_t>(i & 0xFF);
    return v;
}

// Build a byte buffer filled entirely with a single repeated byte value.
static std::vector<uint8_t> make_fill_bytes(size_t n_bytes, uint8_t fill) {
    return std::vector<uint8_t>(n_bytes, fill);
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. uint32 round-trip (main PFPL use-case)
// ─────────────────────────────────────────────────────────────────────────────

TEST(BitshuffleStage, Uint32RoundTrip) {
    CudaStream stream;
    const size_t CHUNK = 16384;
    auto pool = make_test_pool(CHUNK * 2);

    auto h_in = make_ramp_bytes(CHUNK);

    BitshuffleStage enc;
    enc.setBlockSize(CHUNK);
    enc.setElementWidth(4);

    auto h_encoded = run_bitshuffle(enc, h_in, stream, *pool);
    ASSERT_EQ(h_encoded.size(), h_in.size());

    BitshuffleStage dec;
    dec.setBlockSize(CHUNK);
    dec.setElementWidth(4);
    dec.setInverse(true);

    auto h_decoded = run_bitshuffle(dec, h_encoded, stream, *pool);
    ASSERT_EQ(h_decoded.size(), h_in.size());

    EXPECT_EQ(h_in, h_decoded);
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. uint8 round-trip
// ─────────────────────────────────────────────────────────────────────────────

TEST(BitshuffleStage, Uint8RoundTrip) {
    CudaStream stream;
    const size_t CHUNK = 16384;
    auto pool = make_test_pool(CHUNK * 2);

    auto h_in = make_ramp_bytes(CHUNK);

    BitshuffleStage enc;
    enc.setBlockSize(CHUNK);
    enc.setElementWidth(1);

    auto h_encoded = run_bitshuffle(enc, h_in, stream, *pool);
    ASSERT_EQ(h_encoded.size(), h_in.size());

    BitshuffleStage dec;
    dec.setBlockSize(CHUNK);
    dec.setElementWidth(1);
    dec.setInverse(true);

    auto h_decoded = run_bitshuffle(dec, h_encoded, stream, *pool);

    EXPECT_EQ(h_in, h_decoded);
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. uint16 round-trip
// ─────────────────────────────────────────────────────────────────────────────

TEST(BitshuffleStage, Uint16RoundTrip) {
    CudaStream stream;
    const size_t CHUNK = 16384;
    auto pool = make_test_pool(CHUNK * 2);

    auto h_in = make_ramp_bytes(CHUNK);

    BitshuffleStage enc;
    enc.setBlockSize(CHUNK);
    enc.setElementWidth(2);

    auto h_encoded = run_bitshuffle(enc, h_in, stream, *pool);
    ASSERT_EQ(h_encoded.size(), h_in.size());

    BitshuffleStage dec;
    dec.setBlockSize(CHUNK);
    dec.setElementWidth(2);
    dec.setInverse(true);

    auto h_decoded = run_bitshuffle(dec, h_encoded, stream, *pool);

    EXPECT_EQ(h_in, h_decoded);
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. uint64 round-trip
// ─────────────────────────────────────────────────────────────────────────────

TEST(BitshuffleStage, Uint64RoundTrip) {
    CudaStream stream;
    const size_t CHUNK = 16384;
    auto pool = make_test_pool(CHUNK * 2);

    auto h_in = make_ramp_bytes(CHUNK);

    BitshuffleStage enc;
    enc.setBlockSize(CHUNK);
    enc.setElementWidth(8);

    auto h_encoded = run_bitshuffle(enc, h_in, stream, *pool);
    ASSERT_EQ(h_encoded.size(), h_in.size());

    BitshuffleStage dec;
    dec.setBlockSize(CHUNK);
    dec.setElementWidth(8);
    dec.setInverse(true);

    auto h_decoded = run_bitshuffle(dec, h_encoded, stream, *pool);

    EXPECT_EQ(h_in, h_decoded);
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. Size-preserving
// ─────────────────────────────────────────────────────────────────────────────

TEST(BitshuffleStage, SizePreserving) {
    BitshuffleStage stage;
    auto sizes = stage.estimateOutputSizes({16384});
    ASSERT_EQ(sizes.size(), 1u);
    EXPECT_EQ(sizes[0], 16384u);
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. Transform changes the data (non-trivial input)
// ─────────────────────────────────────────────────────────────────────────────

TEST(BitshuffleStage, TransformChangesData) {
    CudaStream stream;
    const size_t CHUNK = 16384;
    auto pool = make_test_pool(CHUNK * 2);

    // Use a ramp — guaranteed non-trivial (not all zeros, not all same bytes).
    auto h_in = make_ramp_bytes(CHUNK);

    BitshuffleStage enc;
    enc.setBlockSize(CHUNK);
    enc.setElementWidth(4);

    auto h_encoded = run_bitshuffle(enc, h_in, stream, *pool);

    EXPECT_NE(h_in, h_encoded)
        << "encode output should differ from input for non-trivial data";
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. All-zeros round-trip
// ─────────────────────────────────────────────────────────────────────────────

TEST(BitshuffleStage, AllZerosRoundTrip) {
    CudaStream stream;
    const size_t CHUNK = 16384;
    auto pool = make_test_pool(CHUNK * 2);

    auto h_in = make_fill_bytes(CHUNK, 0x00);

    BitshuffleStage enc;
    enc.setBlockSize(CHUNK);
    enc.setElementWidth(4);

    auto h_encoded = run_bitshuffle(enc, h_in, stream, *pool);

    // All-zero input → all bit-planes are zero → encoded output is also all zeros.
    EXPECT_EQ(h_in, h_encoded) << "all-zero input should encode to all-zero output";

    BitshuffleStage dec;
    dec.setBlockSize(CHUNK);
    dec.setElementWidth(4);
    dec.setInverse(true);

    auto h_decoded = run_bitshuffle(dec, h_encoded, stream, *pool);
    EXPECT_EQ(h_in, h_decoded);
}

// ─────────────────────────────────────────────────────────────────────────────
// 8. All-ones (0xFF) round-trip
// ─────────────────────────────────────────────────────────────────────────────

TEST(BitshuffleStage, AllOnesRoundTrip) {
    CudaStream stream;
    const size_t CHUNK = 16384;
    auto pool = make_test_pool(CHUNK * 2);

    auto h_in = make_fill_bytes(CHUNK, 0xFF);

    BitshuffleStage enc;
    enc.setBlockSize(CHUNK);
    enc.setElementWidth(4);

    // All-0xFF input: every bit is 1.  Every ballot_sync = 0xFFFFFFFF.
    // So encoded output should also be all-0xFF.
    auto h_encoded = run_bitshuffle(enc, h_in, stream, *pool);
    EXPECT_EQ(h_in, h_encoded) << "all-0xFF input should encode to all-0xFF";

    BitshuffleStage dec;
    dec.setBlockSize(CHUNK);
    dec.setElementWidth(4);
    dec.setInverse(true);

    auto h_decoded = run_bitshuffle(dec, h_encoded, stream, *pool);
    EXPECT_EQ(h_in, h_decoded);
}

// ─────────────────────────────────────────────────────────────────────────────
// 9. Multi-chunk round-trip (4 consecutive 16 KB chunks)
// ─────────────────────────────────────────────────────────────────────────────

TEST(BitshuffleStage, MultiChunkRoundTrip) {
    CudaStream stream;
    const size_t CHUNK = 16384;
    const size_t N_CHUNKS = 4;
    const size_t TOTAL = CHUNK * N_CHUNKS;
    auto pool = make_test_pool(TOTAL * 2);

    // Fill each chunk with a different byte pattern to distinguish them.
    std::vector<uint8_t> h_in(TOTAL);
    for (size_t i = 0; i < TOTAL; ++i)
        h_in[i] = static_cast<uint8_t>((i * 7 + 3) & 0xFF);

    BitshuffleStage enc;
    enc.setBlockSize(CHUNK);
    enc.setElementWidth(4);

    auto h_encoded = run_bitshuffle(enc, h_in, stream, *pool);
    ASSERT_EQ(h_encoded.size(), TOTAL);

    BitshuffleStage dec;
    dec.setBlockSize(CHUNK);
    dec.setElementWidth(4);
    dec.setInverse(true);

    auto h_decoded = run_bitshuffle(dec, h_encoded, stream, *pool);
    ASSERT_EQ(h_decoded.size(), TOTAL);

    EXPECT_EQ(h_in, h_decoded);
}

// ─────────────────────────────────────────────────────────────────────────────
// 10. Bit-plane ordering verification (known pattern)
//
// The 4-byte butterfly naturally places the MSBit contributions in sublane 0,
// so the physical layout is MSB-first: plane 0 = bit W-1, plane W-1 = bit 0.
//
// Input: 1024 uint32 elements (CHUNK = 4096 bytes), all = 0x80000000u
//   (only bit 31 is set).  N_chunk = 1024, n_per_plane = 32.
//   After encoding:
//     Plane 0  (bit 31, MSBit, at word offset 0) → all 32 words = 0xFFFFFFFF
//     Planes 1..31 (bits 30..0, none set)        → all words   = 0x00000000
// ─────────────────────────────────────────────────────────────────────────────

TEST(BitshuffleStage, BitPlaneOrdering_MSBOnly) {
    CudaStream stream;
    // Smallest valid chunk for element_width=4: 1024 * element_width = 4096 bytes.
    const size_t CHUNK = 4096;   // 1024 uint32 elements
    auto pool = make_test_pool(CHUNK * 2);

    // 1024 elements, each = 0x80000000u (only MSBit set).
    std::vector<uint32_t> elems(1024, 0x80000000u);
    std::vector<uint8_t> h_in(CHUNK);
    std::memcpy(h_in.data(), elems.data(), CHUNK);

    BitshuffleStage enc;
    enc.setBlockSize(CHUNK);
    enc.setElementWidth(4);

    auto h_encoded = run_bitshuffle(enc, h_in, stream, *pool);
    ASSERT_EQ(h_encoded.size(), CHUNK);

    // 1024 uint32 words: 32 planes × 32 words each.
    std::vector<uint32_t> words(1024);
    std::memcpy(words.data(), h_encoded.data(), CHUNK);

    // Plane 0 (MSBit, sublane 0) → every element had this bit set → all-1s.
    for (int w = 0; w < 32; w++) {
        EXPECT_EQ(words[0 * 32 + w], 0xFFFFFFFFu)
            << "Plane 0 (MSBit) word " << w << " should be all-1s";
    }
    // Planes 1..31 (lower bits, none set) → all-0s.
    for (int p = 1; p <= 31; p++) {
        for (int w = 0; w < 32; w++) {
            EXPECT_EQ(words[p * 32 + w], 0u)
                << "Plane " << p << " word " << w << " should be 0";
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 11. Header serialization round-trip
// ─────────────────────────────────────────────────────────────────────────────

TEST(BitshuffleStage, HeaderSerializationRoundTrip) {
    BitshuffleStage original;
    original.setBlockSize(8192);
    original.setElementWidth(2);

    uint8_t buf[8] = {};
    size_t written = original.serializeHeader(0, buf, sizeof(buf));
    ASSERT_EQ(written, 5u);

    BitshuffleStage restored;
    restored.deserializeHeader(buf, written);

    EXPECT_EQ(restored.getBlockSize(),    8192u);
    EXPECT_EQ(restored.getElementWidth(), 2u);
    EXPECT_EQ(restored.getStageTypeId(),
              static_cast<uint16_t>(StageType::BITSHUFFLE));
}
