/**
 * tests/stages/test_tupl_stage.cpp
 *
 * GPU unit tests for TUPLStage — AoS <-> SoA tuple deinterleave transpose.
 *
 * TUPLStage regroups a block of `tuples` k-field structs (LC's TUPLk) into
 * field-major (SoA) order:
 *   Forward (encode): AoS -> SoA (all field 0 words, then field 1, ...)
 *   Inverse (decode): SoA -> AoS (restores original interleaving)
 * Size-preserving (pure permutation), byte-level tail leftover (block_size
 * not a multiple of dim*word_size) copied verbatim by both directions.
 *
 *   TP1   TUPLStage/Dim2Word1RoundTrip           — byte-granular pairs (16 KB block)
 *   TP2   TUPLStage/Dim3Word2RoundTrip            — uint16 triples, non-even block/dim*word
 *   TP3   TUPLStage/Dim6Word4RoundTrip             — uint32 6-tuples
 *   TP4   TUPLStage/Dim6Word8RoundTrip             — uint64 6-tuples (LC's TUPL6_8)
 *   TP5   TUPLStage/Dim12Word1RoundTrip            — LC's max dim (TUPL12_1)
 *   TP6   TUPLStage/SizePreserving                 — estimateOutputSizes == input bytes
 *   TP7   TUPLStage/TransformChangesData           — encoded output differs for non-trivial data
 *   TP8   TUPLStage/KnownPatternSoALayout          — exact SoA layout verified by hand
 *   TP9   TUPLStage/ExtraBytesPreservedUnchanged   — intra-block leftover copied verbatim
 *   TP10  TUPLStage/MultiBlockRoundTrip             — several full blocks, distinct per-block data
 *   TP11  TUPLStage/PartialTailRoundTrip            — stream tail shorter than one block
 *   TP12  TUPLStage/DegenerateAllExtraRoundTrip     — block smaller than one tuple (pure copy)
 *   TP13  TUPLStage/HeaderSerializationRoundTrip    — block_size/word_size/dim survive header
 *   TP14  TUPLStage/InvalidConfigThrows             — bad word_size/dim/block_size throw
 *   TP15  TUPLStage/AllZerosRoundTrip                — all-zero input round-trips
 *   TP16  TUPLStage/PipelineIntegration              — LorenzoQuant -> TUPL codes round-trip
 */

#include <gtest/gtest.h>
#include "helpers/fz_test_utils.h"
#include "structural/tupl/tupl_stage.h"
#include "fzgpumodules.h"

#include <cmath>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cstring>

using namespace fz;
using namespace fz_test;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

static std::vector<uint8_t> run_tupl(
    TUPLStage& stage,
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

static std::vector<uint8_t> make_ramp_bytes(size_t n_bytes) {
    std::vector<uint8_t> v(n_bytes);
    for (size_t i = 0; i < n_bytes; ++i) v[i] = static_cast<uint8_t>(i & 0xFF);
    return v;
}

static std::vector<uint8_t> make_fill_bytes(size_t n_bytes, uint8_t fill) {
    return std::vector<uint8_t>(n_bytes, fill);
}

// Generic round-trip check: encode with (block, word, dim), decode, compare.
static void roundtrip(size_t block, size_t word, size_t dim,
                       const std::vector<uint8_t>& h_in,
                       cudaStream_t stream, fz::MemoryPool& pool) {
    TUPLStage enc;
    enc.setBlockSize(block);
    enc.setWordSize(word);
    enc.setDim(dim);
    auto h_encoded = run_tupl(enc, h_in, stream, pool);
    ASSERT_EQ(h_encoded.size(), h_in.size());

    TUPLStage dec;
    dec.setBlockSize(block);
    dec.setWordSize(word);
    dec.setDim(dim);
    dec.setInverse(true);
    auto h_decoded = run_tupl(dec, h_encoded, stream, pool);
    ASSERT_EQ(h_decoded.size(), h_in.size());

    EXPECT_EQ(h_in, h_decoded);
}

// ─────────────────────────────────────────────────────────────────────────────
// TP1: Dim2Word1RoundTrip — byte-granular pairs, default 16 KB block (LC TUPL2_1)
// ─────────────────────────────────────────────────────────────────────────────

TEST(TUPLStage, Dim2Word1RoundTrip) {
    CudaStream stream;
    const size_t CHUNK = 16384;
    auto pool = make_test_pool(CHUNK * 2);
    auto h_in = make_ramp_bytes(CHUNK);
    roundtrip(CHUNK, 1, 2, h_in, stream, *pool);
}

// ─────────────────────────────────────────────────────────────────────────────
// TP2: Dim3Word2RoundTrip — uint16 triples; 16384 doesn't divide evenly by 3*2=6
// ─────────────────────────────────────────────────────────────────────────────

TEST(TUPLStage, Dim3Word2RoundTrip) {
    CudaStream stream;
    const size_t CHUNK = 16384;
    auto pool = make_test_pool(CHUNK * 2);
    auto h_in = make_ramp_bytes(CHUNK);
    roundtrip(CHUNK, 2, 3, h_in, stream, *pool);
}

// ─────────────────────────────────────────────────────────────────────────────
// TP3: Dim6Word4RoundTrip — uint32 6-tuples (LC TUPL6_4)
// ─────────────────────────────────────────────────────────────────────────────

TEST(TUPLStage, Dim6Word4RoundTrip) {
    CudaStream stream;
    const size_t CHUNK = 16384;
    auto pool = make_test_pool(CHUNK * 2);
    auto h_in = make_ramp_bytes(CHUNK);
    roundtrip(CHUNK, 4, 6, h_in, stream, *pool);
}

// ─────────────────────────────────────────────────────────────────────────────
// TP4: Dim6Word8RoundTrip — uint64 6-tuples (LC TUPL6_8)
// ─────────────────────────────────────────────────────────────────────────────

TEST(TUPLStage, Dim6Word8RoundTrip) {
    CudaStream stream;
    const size_t CHUNK = 16384;
    auto pool = make_test_pool(CHUNK * 2);
    auto h_in = make_ramp_bytes(CHUNK);
    roundtrip(CHUNK, 8, 6, h_in, stream, *pool);
}

// ─────────────────────────────────────────────────────────────────────────────
// TP5: Dim12Word1RoundTrip — LC's max dim (TUPL12_1)
// ─────────────────────────────────────────────────────────────────────────────

TEST(TUPLStage, Dim12Word1RoundTrip) {
    CudaStream stream;
    const size_t CHUNK = 16384;
    auto pool = make_test_pool(CHUNK * 2);
    auto h_in = make_ramp_bytes(CHUNK);
    roundtrip(CHUNK, 1, 12, h_in, stream, *pool);
}

// ─────────────────────────────────────────────────────────────────────────────
// TP6: SizePreserving — estimateOutputSizes returns input byte count unchanged
// ─────────────────────────────────────────────────────────────────────────────

TEST(TUPLStage, SizePreserving) {
    TUPLStage stage;
    auto sizes = stage.estimateOutputSizes({16384});
    ASSERT_EQ(sizes.size(), 1u);
    EXPECT_EQ(sizes[0], 16384u);
}

// ─────────────────────────────────────────────────────────────────────────────
// TP7: TransformChangesData — encoded output differs from input for non-trivial data
// ─────────────────────────────────────────────────────────────────────────────

TEST(TUPLStage, TransformChangesData) {
    CudaStream stream;
    const size_t CHUNK = 16384;
    auto pool = make_test_pool(CHUNK * 2);
    auto h_in = make_ramp_bytes(CHUNK);

    TUPLStage enc;
    enc.setBlockSize(CHUNK);
    enc.setWordSize(1);
    enc.setDim(4);

    auto h_encoded = run_tupl(enc, h_in, stream, *pool);
    EXPECT_NE(h_in, h_encoded)
        << "encode output should differ from input for non-trivial data";
}

// ─────────────────────────────────────────────────────────────────────────────
// TP8: KnownPatternSoALayout — exact SoA layout verified by hand
//
// block=8, dim=2, word=1: 4 tuples of (a,b) pairs.
//   in  = [a0,b0,a1,b1,a2,b2,a3,b3] = [10,20,11,21,12,22,13,23]
//   out = [a0,a1,a2,a3, b0,b1,b2,b3] = [10,11,12,13, 20,21,22,23]
// ─────────────────────────────────────────────────────────────────────────────

TEST(TUPLStage, KnownPatternSoALayout) {
    CudaStream stream;
    auto pool = make_test_pool(64);

    std::vector<uint8_t> h_in = {10, 20, 11, 21, 12, 22, 13, 23};
    std::vector<uint8_t> expected = {10, 11, 12, 13, 20, 21, 22, 23};

    TUPLStage enc;
    enc.setBlockSize(8);
    enc.setWordSize(1);
    enc.setDim(2);

    auto h_encoded = run_tupl(enc, h_in, stream, *pool);
    EXPECT_EQ(h_encoded, expected);

    TUPLStage dec;
    dec.setBlockSize(8);
    dec.setWordSize(1);
    dec.setDim(2);
    dec.setInverse(true);

    auto h_decoded = run_tupl(dec, h_encoded, stream, *pool);
    EXPECT_EQ(h_decoded, h_in);
}

// ─────────────────────────────────────────────────────────────────────────────
// TP9: ExtraBytesPreservedUnchanged — intra-block leftover copied verbatim
//
// block=10, dim=3, word=1: tuples = 10/1/3 = 3 (9 bytes used), extra = 1 byte
// at offset 9. Both encode and decode must leave that byte untouched.
// ─────────────────────────────────────────────────────────────────────────────

TEST(TUPLStage, ExtraBytesPreservedUnchanged) {
    CudaStream stream;
    auto pool = make_test_pool(64);

    std::vector<uint8_t> h_in = {1, 2, 3, 4, 5, 6, 7, 8, 9, 0xAB};

    TUPLStage enc;
    enc.setBlockSize(10);
    enc.setWordSize(1);
    enc.setDim(3);

    auto h_encoded = run_tupl(enc, h_in, stream, *pool);
    ASSERT_EQ(h_encoded.size(), 10u);
    EXPECT_EQ(h_encoded[9], 0xAB) << "extra tail byte must pass through unchanged on encode";

    TUPLStage dec;
    dec.setBlockSize(10);
    dec.setWordSize(1);
    dec.setDim(3);
    dec.setInverse(true);

    auto h_decoded = run_tupl(dec, h_encoded, stream, *pool);
    EXPECT_EQ(h_decoded[9], 0xAB) << "extra tail byte must pass through unchanged on decode";
    EXPECT_EQ(h_decoded, h_in);
}

// ─────────────────────────────────────────────────────────────────────────────
// TP10: MultiBlockRoundTrip — several full blocks, distinct per-block data
// ─────────────────────────────────────────────────────────────────────────────

TEST(TUPLStage, MultiBlockRoundTrip) {
    CudaStream stream;
    const size_t CHUNK = 4096;
    const size_t N_CHUNKS = 5;
    const size_t TOTAL = CHUNK * N_CHUNKS;
    auto pool = make_test_pool(TOTAL * 2);

    std::vector<uint8_t> h_in(TOTAL);
    for (size_t i = 0; i < TOTAL; ++i)
        h_in[i] = static_cast<uint8_t>((i * 7 + 3) & 0xFF);

    roundtrip(CHUNK, 4, 4, h_in, stream, *pool);
}

// ─────────────────────────────────────────────────────────────────────────────
// TP11: PartialTailRoundTrip — total input shorter than a full block (raw tail copy)
// ─────────────────────────────────────────────────────────────────────────────

TEST(TUPLStage, PartialTailRoundTrip) {
    CudaStream stream;
    const size_t CHUNK = 16384;
    const size_t TOTAL = CHUNK + 500;  // one full block + a 500-byte tail
    auto pool = make_test_pool(TOTAL * 2);

    auto h_in = make_ramp_bytes(TOTAL);
    roundtrip(CHUNK, 2, 4, h_in, stream, *pool);
}

// ─────────────────────────────────────────────────────────────────────────────
// TP12: DegenerateAllExtraRoundTrip — block smaller than one tuple => pure copy
//
// block=4, dim=8, word=1: dim*word=8 > block=4, so tuples=0, the entire block
// is "extra" and both directions degrade to an identity copy.
// ─────────────────────────────────────────────────────────────────────────────

TEST(TUPLStage, DegenerateAllExtraRoundTrip) {
    CudaStream stream;
    auto pool = make_test_pool(64);

    std::vector<uint8_t> h_in = {0x11, 0x22, 0x33, 0x44};

    TUPLStage enc;
    enc.setBlockSize(4);
    enc.setWordSize(1);
    enc.setDim(8);

    auto h_encoded = run_tupl(enc, h_in, stream, *pool);
    EXPECT_EQ(h_encoded, h_in) << "block smaller than one tuple must be an identity copy";

    TUPLStage dec;
    dec.setBlockSize(4);
    dec.setWordSize(1);
    dec.setDim(8);
    dec.setInverse(true);

    auto h_decoded = run_tupl(dec, h_encoded, stream, *pool);
    EXPECT_EQ(h_decoded, h_in);
}

// ─────────────────────────────────────────────────────────────────────────────
// TP13: HeaderSerializationRoundTrip — block_size/word_size/dim survive header bytes
// ─────────────────────────────────────────────────────────────────────────────

TEST(TUPLStage, HeaderSerializationRoundTrip) {
    TUPLStage original;
    original.setBlockSize(8192);
    original.setWordSize(4);
    original.setDim(6);

    uint8_t buf[8] = {};
    size_t written = original.serializeHeader(0, buf, sizeof(buf));
    ASSERT_EQ(written, 6u);

    TUPLStage restored;
    restored.deserializeHeader(buf, written);

    EXPECT_EQ(restored.getBlockSize(), 8192u);
    EXPECT_EQ(restored.getWordSize(), 4u);
    EXPECT_EQ(restored.getDim(), 6u);
    EXPECT_EQ(restored.getStageTypeId(), static_cast<uint16_t>(StageType::TUPL));
}

// ─────────────────────────────────────────────────────────────────────────────
// TP14: InvalidConfigThrows — bad word_size/dim/block_size throw
// ─────────────────────────────────────────────────────────────────────────────

TEST(TUPLStage, InvalidConfigThrows) {
    CudaStream stream;
    const size_t N = 16384;
    auto pool = make_test_pool(N * 2);
    auto h_in = make_ramp_bytes(N);

    // Unsupported word_size: 3 is not 1, 2, 4, or 8
    {
        TUPLStage stage;
        stage.setBlockSize(N);
        stage.setWordSize(3);
        stage.setDim(2);
        EXPECT_THROW(run_tupl(stage, h_in, stream, *pool), std::invalid_argument)
            << "word_size=3 must throw";
    }

    // dim < 2 is not a tuple
    {
        TUPLStage stage;
        stage.setBlockSize(N);
        stage.setWordSize(1);
        stage.setDim(1);
        EXPECT_THROW(run_tupl(stage, h_in, stream, *pool), std::invalid_argument)
            << "dim=1 must throw";
    }

    // block_size must be a multiple of word_size
    {
        TUPLStage stage;
        stage.setBlockSize(15);
        stage.setWordSize(4);
        stage.setDim(2);
        EXPECT_THROW(run_tupl(stage, h_in, stream, *pool), std::invalid_argument)
            << "block_size=15 with word_size=4 must throw";
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TP15: AllZerosRoundTrip
// ─────────────────────────────────────────────────────────────────────────────

TEST(TUPLStage, AllZerosRoundTrip) {
    CudaStream stream;
    const size_t CHUNK = 16384;
    auto pool = make_test_pool(CHUNK * 2);
    auto h_in = make_fill_bytes(CHUNK, 0x00);

    TUPLStage enc;
    enc.setBlockSize(CHUNK);
    enc.setWordSize(4);
    enc.setDim(3);

    auto h_encoded = run_tupl(enc, h_in, stream, *pool);
    EXPECT_EQ(h_in, h_encoded) << "all-zero input should encode to all-zero output";

    TUPLStage dec;
    dec.setBlockSize(CHUNK);
    dec.setWordSize(4);
    dec.setDim(3);
    dec.setInverse(true);

    auto h_decoded = run_tupl(dec, h_encoded, stream, *pool);
    EXPECT_EQ(h_in, h_decoded);
}

// ─────────────────────────────────────────────────────────────────────────────
// TP16: PipelineIntegration — LorenzoQuant -> TUPL on the codes port, full round-trip
// ─────────────────────────────────────────────────────────────────────────────

TEST(TUPLStage, PipelineIntegration) {
    CudaStream stream;
    constexpr size_t N  = 1 << 13;  // 8 K floats
    constexpr float  EB = 1e-2f;
    const size_t in_bytes = N * sizeof(float);

    auto h_input = make_sine_floats(N, 0.01f, 50.0f);

    CudaBuffer<float> d_in(N);
    d_in.upload(h_input, stream);
    stream.sync();

    Pipeline pipeline(in_bytes, MemoryStrategy::MINIMAL, 5.0f);
    auto* lrz = pipeline.addStage<LorenzoQuantStage<float, uint16_t>>();
    lrz->setErrorBound(EB);
    lrz->setQuantRadius(512);
    lrz->setOutlierCapacity(0.2f);

    auto* tupl = pipeline.addStage<TUPLStage>();
    tupl->setBlockSize(16384);
    tupl->setWordSize(2);  // uint16_t codes
    tupl->setDim(2);
    pipeline.connect(tupl, lrz, "codes");

    pipeline.setPoolManagedDecompOutput(false);
    pipeline.finalize();

    void*  d_comp  = nullptr;
    size_t comp_sz = 0;
    ASSERT_NO_THROW(
        pipeline.compress(d_in.void_ptr(), in_bytes, &d_comp, &comp_sz, stream)
    ) << "LorenzoQuant->TUPL compress must not throw";
    stream.sync();
    ASSERT_GT(comp_sz, 0u) << "Compressed output is empty";

    void*  d_dec  = nullptr;
    size_t dec_sz = 0;
    ASSERT_NO_THROW(
        pipeline.decompress(nullptr, 0, &d_dec, &dec_sz, stream)
    ) << "LorenzoQuant->TUPL decompress must not throw";
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
        << "LorenzoQuant->TUPL pipeline round-trip max_err=" << max_err;
}
