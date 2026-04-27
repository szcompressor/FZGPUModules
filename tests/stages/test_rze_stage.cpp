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
 *   9.  Round-trip on a single chunk with a checkerboard byte pattern.
 *   10. Round-trip on data with long runs of repeated bytes (good for RE).
 *   11. Unsupported chunk_size throws at execute() time.
 *   12. Unsupported levels value throws at execute() time.
 *   13. isGraphCompatible() returns true for forward, false for inverse.
 *   14. saveState()/restoreState() preserves config across compress+decompress.
 *   15. Repeated compress+decompress on the same stage objects is stable.
 *   16. Input smaller than one chunk (partial chunk) round-trips correctly.
 *   17. Four-chunk input (4 × 16 KB) round-trips correctly.
 *   18. Mixed-density multi-chunk: half sparse, half dense; round-trip correct.
 */

#include <gtest/gtest.h>
#include "helpers/fz_test_utils.h"
#include "transforms/rze/rze_stage.h"
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

// 11. Unsupported chunk_size throws at execute()
//
// Only chunk_size == 16384 is implemented.  Any other value must throw a
// std::runtime_error so the caller gets a clear message rather than silent
// data corruption.
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

// 12. Unsupported levels value throws at execute()
//
// Levels 1, 2, 3 are not yet implemented; only levels == 4 is supported.
// Each unsupported value must throw a std::runtime_error.
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

// 13. isGraphCompatible(): true for forward, false for inverse
//
// RZE inverse contains a blocking D2H cudaMemcpy (to read per-chunk sizes
// from the stream header) and a stream sync between scatter and decode
// kernels — neither is capturable.  The forward path has no such calls.
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

// 14. saveState()/restoreState() preserves config across compress+decompress
//
// decompressMulti() wraps each stage's execute() with saveState()/restoreState()
// so that header deserialization in the inverse pass does not permanently
// overwrite the forward configuration.
// Specifically: cached_orig_bytes_ is set by the forward pass and also written
// into the serialized header; after a decompress the saved forward value must
// be intact so a subsequent compress sees the correct config.
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

// 15. Repeated compress+decompress on the same stage objects is stable
//
// Verifies that persistent scratch buffers are correctly reused (not
// double-freed or left dirty) across multiple round-trips on the same
// pair of RZEStage objects.
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

// 16. Partial chunk (input smaller than one chunk) round-trips correctly
//
// Exercises the single-chunk path where the input does not fill the
// full 16 KB chunk size.  The encoder must not read past the end of
// the input; the decoder must reproduce exactly the bytes passed in.
TEST(RZEStage, PartialChunkRoundTrip) {
    // Use a size that is not a power of 2 and well below 16384
    std::vector<uint8_t> data(3000, 0);
    std::mt19937 rng(555);
    std::uniform_int_distribution<int> dist(0, 255);
    for (int i = 0; i < 30; i++)
        data[rng() % 3000] = static_cast<uint8_t>(dist(rng));
    round_trip(data);
}

// 17. Four-chunk input (4 × 16 KB = 64 KB) round-trips correctly
//
// Tests that the prefix-sum offset computation in the packing kernel
// and the inverse scatter are correct beyond 2 chunks.
TEST(RZEStage, FourChunksRoundTrip) {
    std::mt19937 rng(777);
    std::uniform_int_distribution<int> dist(0, 255);
    std::vector<uint8_t> data(65536);
    for (auto& b : data) b = static_cast<uint8_t>(dist(rng));
    round_trip(data);
}

// 18. Mixed-density multi-chunk: first half sparse, second half dense
//
// Exercises per-chunk divergence in the encode kernel — some chunks
// will take the uncompressed-verbatim path (high-bit flag set) while
// others will be compressed.  The decoder must handle both cases in
// the same stream.
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
// 19. Pipeline integration: Lorenzo → RZEStage full round-trip
//
// Exercises RZEStage as a downstream consumer of Lorenzo's "codes" output
// through the full Pipeline API (compress → decompress).  Smooth sinusoidal
// input data produces highly compressible Lorenzo quantization codes
// (many zeros after prediction), which RZE encodes well.
// ─────────────────────────────────────────────────────────────────────────────
TEST(RZEStage, PipelineIntegration) {
    CudaStream stream;
    constexpr size_t N  = 1 << 13;  // 8 K floats
    constexpr float  EB = 1e-2f;
    const size_t in_bytes = N * sizeof(float);

    std::vector<float> h_input(N);
    for (size_t i = 0; i < N; i++)
        h_input[i] = std::sin(static_cast<float>(i) * 0.01f) * 50.0f
                   + std::cos(static_cast<float>(i) * 0.003f) * 20.0f;

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
// 20. RZE pipeline: compressed output smaller than raw for smooth data
//
// Uses the same Lorenzo→RZE pipeline as test 19 but verifies the compression
// ratio.  Smooth sinusoidal data produces mostly-zero Lorenzo codes; RZE's
// zero-elimination should give a meaningfully compressed output.
// ─────────────────────────────────────────────────────────────────────────────
TEST(RZEStage, PipelineCompressionRatio) {
    CudaStream stream;
    constexpr size_t N  = 1 << 14;  // 16 K floats — enough for RZE to shine
    constexpr float  EB = 1e-2f;
    const size_t in_bytes = N * sizeof(float);

    std::vector<float> h_input(N);
    for (size_t i = 0; i < N; i++)
        h_input[i] = std::sin(static_cast<float>(i) * 0.01f) * 50.0f
                   + std::cos(static_cast<float>(i) * 0.003f) * 20.0f;

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
