/**
 * tests/stages/test_gpulz.cpp
 *
 * GPU unit tests for GPULZStage — GPU LZSS lossless byte-stream compressor
 * (direct port of the GPULZ reference kernels). Each fixed-size chunk is
 * compressed independently by one CUDA block via a 32-word sliding-window
 * match search; inverse reconstructs exactly. Word granularity is 1/2/4/8
 * bytes; chunk size is 1024/2048/4096 bytes.
 */

#include <gtest/gtest.h>
#include "helpers/fz_test_utils.h"
#include "helpers/stage_harness.h"
#include "coders/gpulz/gpulz_stage.h"
#include "fzgpumodules.h"

#include <cstdint>
#include <vector>
#include <random>
#include <cstring>

using namespace fz;
using namespace fz_test;

// Run GPULZStage (encode or decode, per its is_inverse flag) on a byte vector.
static std::vector<uint8_t> run_gpulz(
    GPULZStage& stage, const std::vector<uint8_t>& h_in,
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

// GPULZStage requires input aligned to chunk_size (getRequiredInputAlignment());
// standalone stage tests (not routed through Pipeline::finalize()'s automatic
// padding) must pad manually. Padding bytes are zero and are stripped back off
// on decode via the recorded original byte count, so callers get back exactly
// `original` bytes even though `original.size()` need not be chunk-aligned.
static std::vector<uint8_t> pad_to_chunk(std::vector<uint8_t> data, size_t chunk_size) {
    const size_t rem = data.size() % chunk_size;
    if (rem != 0) data.resize(data.size() + (chunk_size - rem), 0);
    return data;
}

// Compress then decompress; verify byte-exact round-trip.
static void round_trip(const std::vector<uint8_t>& original, int word_size = 4,
                        size_t chunk_size = 2048, int match_level = 1) {
    CudaStream cs;
    const auto padded = pad_to_chunk(original, chunk_size);
    auto pool = make_test_pool(padded.size() + 65536);

    GPULZStage enc;
    enc.setChunkSize(chunk_size);
    enc.setWordSize(word_size);
    enc.setMatchLevel(match_level);
    const size_t enc_cap = enc.estimateOutputSizes({padded.size()})[0];
    const auto compressed = run_gpulz(enc, padded, enc_cap, cs.stream, *pool);

    GPULZStage dec;
    dec.setChunkSize(chunk_size);
    dec.setWordSize(word_size);
    dec.setInverse(true);
    const auto restored = run_gpulz(dec, compressed, padded.size() + 4096, cs.stream, *pool);

    ASSERT_EQ(restored.size(), padded.size());
    EXPECT_EQ(restored, padded) << "GPULZ round-trip mismatch (word_size=" << word_size
                                 << ", chunk_size=" << chunk_size
                                 << ", match_level=" << match_level << ")";
}

// Compressed size for `original` at a given match level.
static size_t compressed_size(const std::vector<uint8_t>& original, int match_level,
                              int word_size = 4, size_t chunk_size = 2048) {
    CudaStream cs;
    const auto padded = pad_to_chunk(original, chunk_size);
    auto pool = make_test_pool(padded.size() + 65536);

    GPULZStage enc;
    enc.setChunkSize(chunk_size);
    enc.setWordSize(word_size);
    enc.setMatchLevel(match_level);
    const size_t enc_cap = enc.estimateOutputSizes({padded.size()})[0];
    return run_gpulz(enc, padded, enc_cap, cs.stream, *pool).size();
}

// A stream with structure repeating at a stride well outside the 32-element
// near window, so only the hashed long-range matcher (match_level 1) can find
// it. word_size 4, so the stride below is 40 elements back.
static std::vector<uint8_t> long_range_pattern() {
    std::vector<uint8_t> data(8 * 2048);
    auto* w = reinterpret_cast<uint32_t*>(data.data());
    const size_t n = data.size() / 4;
    std::mt19937 rng(9001);
    std::vector<uint32_t> motif(40);
    for (auto& m : motif) m = rng() & 0xFFFF;
    for (size_t i = 0; i < n; i++) w[i] = motif[i % motif.size()];
    return data;
}

TEST(GPULZStage, MatchLevel0RoundTrip) {
    std::mt19937 rng(2024);
    std::uniform_int_distribution<int> dist(0, 6);
    std::vector<uint8_t> data(4 * 2048);
    for (auto& b : data) b = (uint8_t)dist(rng);
    for (int ws : {1, 2, 4, 8}) round_trip(data, ws, 2048, /*match_level=*/0);
}

TEST(GPULZStage, MatchLevel1RoundTrip) {
    std::mt19937 rng(2025);
    std::uniform_int_distribution<int> dist(0, 6);
    std::vector<uint8_t> data(4 * 2048);
    for (auto& b : data) b = (uint8_t)dist(rng);
    for (int ws : {1, 2, 4, 8}) round_trip(data, ws, 2048, /*match_level=*/1);
}

TEST(GPULZStage, MatchLevel1RoundTripAllChunkSizes) {
    const auto data = long_range_pattern();
    for (size_t cs : {size_t(1024), size_t(2048), size_t(4096)})
        round_trip(data, 4, cs, /*match_level=*/1);
}

// The hashed matcher exists to reach matches the 32-element near window cannot,
// so on a stream whose only redundancy is at stride 40 it must do strictly
// better than level 0.
TEST(GPULZStage, MatchLevel1BeatsLevel0OnLongRangeStructure) {
    const auto data = long_range_pattern();
    const size_t l0 = compressed_size(data, 0);
    const size_t l1 = compressed_size(data, 1);
    EXPECT_LT(l1, l0) << "match_level 1 (" << l1 << " bytes) should beat level 0 ("
                      << l0 << " bytes) on stride-40 structure";
}

// The match level is an encode-side search-effort knob only: it is not part of
// the stream format, so a level-1 stream must decode with a stage that was
// never told about the level (and vice versa).
// The hash table is filled with atomicMax rather than a plain store precisely
// so that the bucket winner -- and therefore the compressed bytes -- does not
// depend on thread scheduling.
TEST(GPULZStage, MatchLevel1OutputIsDeterministic) {
    const auto data = long_range_pattern();
    const auto padded = pad_to_chunk(data, 2048);
    CudaStream cs;
    auto pool = make_test_pool(padded.size() + 65536);

    std::vector<uint8_t> first;
    for (int rep = 0; rep < 4; rep++) {
        GPULZStage enc;
        enc.setMatchLevel(1);
        const size_t cap = enc.estimateOutputSizes({padded.size()})[0];
        auto out = run_gpulz(enc, padded, cap, cs.stream, *pool);
        if (rep == 0) first = std::move(out);
        else EXPECT_EQ(out, first) << "compressed bytes differed on repetition " << rep;
    }
}

TEST(GPULZStage, MatchLevelIsNotPartOfTheStreamFormat) {
    const auto data = long_range_pattern();
    const auto padded = pad_to_chunk(data, 2048);
    CudaStream cs;
    auto pool = make_test_pool(padded.size() + 65536);

    for (int level : {0, 1}) {
        GPULZStage enc;
        enc.setMatchLevel(level);
        const size_t cap = enc.estimateOutputSizes({padded.size()})[0];
        const auto compressed = run_gpulz(enc, padded, cap, cs.stream, *pool);

        GPULZStage dec;              // default-constructed: never told the level
        dec.setInverse(true);
        const auto restored = run_gpulz(dec, compressed, padded.size() + 4096, cs.stream, *pool);
        ASSERT_EQ(restored.size(), padded.size());
        EXPECT_EQ(restored, padded) << "level-" << level << " stream failed to decode";
    }
}

TEST(GPULZStage, RandomBytesRoundTrip) {
    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> dist(0, 255);
    std::vector<uint8_t> data(4 * 2048);
    for (auto& b : data) b = (uint8_t)dist(rng);
    round_trip(data);
}

TEST(GPULZStage, AllZerosRoundTrip) {
    round_trip(std::vector<uint8_t>(4 * 2048, 0));
}

TEST(GPULZStage, ConstantRunRoundTrip) {
    round_trip(std::vector<uint8_t>(4 * 2048, 0x5A));
}

TEST(GPULZStage, RepeatedPatternRoundTrip) {
    // Short repeating pattern -> plenty of matches within the 32-word window.
    std::vector<uint8_t> pattern = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<uint8_t> data;
    while (data.size() < 4 * 2048) data.insert(data.end(), pattern.begin(), pattern.end());
    round_trip(data);
}

TEST(GPULZStage, MixedEmptyAndNonEmptyChunksRoundTrip) {
    // Alternating all-zero chunks (exercises the encode fast-path + decode
    // zero-fill) and random-content chunks (normal LZSS path), at chunk_size=2048.
    std::mt19937 rng(77);
    std::uniform_int_distribution<int> dist(0, 255);
    std::vector<uint8_t> data;
    for (int c = 0; c < 6; c++) {
        if (c % 2 == 0) {
            data.insert(data.end(), 2048, 0);
        } else {
            for (int i = 0; i < 2048; i++) data.push_back((uint8_t)dist(rng));
        }
    }
    round_trip(data);
}

TEST(GPULZStage, AllEmptyChunksCompressToHeaderOnly) {
    CudaStream cs;
    std::vector<uint8_t> data(4 * 2048, 0);
    auto pool = make_test_pool(data.size() + 65536);
    GPULZStage enc;
    enc.setChunkSize(2048);
    enc.setWordSize(4);
    const size_t enc_cap = enc.estimateOutputSizes({data.size()})[0];
    const auto compressed = run_gpulz(enc, data, enc_cap, cs.stream, *pool);
    // header = 8 + 8*4 = 40 bytes; every chunk contributes 0 payload bytes.
    EXPECT_EQ(compressed.size(), (size_t)40);
}

TEST(GPULZStage, MultiChunkRoundTrip) {
    std::mt19937 rng(99);
    std::uniform_int_distribution<int> dist(0, 3);  // low entropy -> repetitions
    std::vector<uint8_t> data(8 * 2048);
    for (auto& b : data) b = (uint8_t)dist(rng);
    round_trip(data);
}

TEST(GPULZStage, PartialChunkRoundTrip) {
    // Not a multiple of chunk_size -- round_trip() pads internally.
    std::mt19937 rng(3);
    std::uniform_int_distribution<int> dist(0, 5);
    std::vector<uint8_t> data(3000);
    for (auto& b : data) b = (uint8_t)dist(rng);
    round_trip(data);
}

TEST(GPULZStage, WordSize1RoundTrip) {
    std::mt19937 rng(20);
    std::uniform_int_distribution<int> dist(0, 7);
    std::vector<uint8_t> data(4 * 2048);
    for (auto& b : data) b = (uint8_t)dist(rng);
    round_trip(data, 1);
}

TEST(GPULZStage, WordSize2RoundTrip) {
    std::mt19937 rng(21);
    std::uniform_int_distribution<int> dist(0, 7);
    std::vector<uint8_t> data(4 * 2048);
    for (auto& b : data) b = (uint8_t)dist(rng);
    round_trip(data, 2);
}

TEST(GPULZStage, WordSize8RoundTrip) {
    std::mt19937 rng(23);
    std::uniform_int_distribution<int> dist(0, 7);
    std::vector<uint8_t> data(4 * 2048);
    for (auto& b : data) b = (uint8_t)dist(rng);
    round_trip(data, 8);
}

TEST(GPULZStage, ConstantRunCompressesSmall) {
    CudaStream cs;
    std::vector<uint8_t> data(4 * 2048, 0x42);
    auto pool = make_test_pool(data.size() + 65536);
    GPULZStage enc;
    enc.setChunkSize(2048);
    enc.setWordSize(4);
    const size_t enc_cap = enc.estimateOutputSizes({data.size()})[0];
    const auto compressed = run_gpulz(enc, data, enc_cap, cs.stream, *pool);
    EXPECT_LT(compressed.size(), data.size() / 4) << "constant run did not compress well";
}

TEST(GPULZStage, HeaderSerialization) {
    GPULZStage s;
    s.setChunkSize(4096);
    s.setWordSize(2);
    uint8_t buf[9] = {0};
    ASSERT_EQ(s.serializeHeader(0, buf, sizeof(buf)), (size_t)9);
    GPULZStage s2;
    s2.deserializeHeader(buf, sizeof(buf));
    EXPECT_EQ(s2.getChunkSize(), (size_t)4096);
    EXPECT_EQ(s2.getWordSize(), 2);
}

TEST(GPULZStage, SaveRestoreState) {
    GPULZStage s;
    s.setChunkSize(2048);
    s.setWordSize(4);
    s.saveState();

    uint8_t other_buf[9] = {0};
    GPULZStage tmp;
    tmp.setChunkSize(4096);
    tmp.setWordSize(8);
    tmp.serializeHeader(0, other_buf, sizeof(other_buf));
    s.deserializeHeader(other_buf, sizeof(other_buf));
    EXPECT_EQ(s.getChunkSize(), (size_t)4096);

    s.restoreState();
    EXPECT_EQ(s.getChunkSize(), (size_t)2048);
    EXPECT_EQ(s.getWordSize(), 4);
}

TEST(GPULZStage, UnsupportedChunkSizeThrows) {
    CudaStream cs;
    std::vector<uint8_t> data(4096, 1);
    auto pool = make_test_pool(data.size() + 65536);
    GPULZStage s;
    s.setChunkSize(12345);  // not in the supported set {1024, 2048, 4096}
    CudaBuffer<uint8_t> d_in(data.size()), d_out(data.size() + 4096);
    d_in.upload(data, cs.stream);
    cudaStreamSynchronize(cs.stream);
    std::vector<void*> in = {d_in.void_ptr()}, out = {d_out.void_ptr()};
    std::vector<size_t> sz = {data.size()};
    EXPECT_THROW(s.execute(cs.stream, pool.get(), in, out, sz), std::runtime_error);
}

TEST(GPULZStage, UnsupportedWordSizeThrows) {
    CudaStream cs;
    std::vector<uint8_t> data(4096, 1);
    auto pool = make_test_pool(data.size() + 65536);
    GPULZStage s;
    s.setWordSize(3);
    CudaBuffer<uint8_t> d_in(data.size()), d_out(data.size() + 4096);
    d_in.upload(data, cs.stream);
    cudaStreamSynchronize(cs.stream);
    std::vector<void*> in = {d_in.void_ptr()}, out = {d_out.void_ptr()};
    std::vector<size_t> sz = {data.size()};
    EXPECT_THROW(s.execute(cs.stream, pool.get(), in, out, sz), std::runtime_error);
}

TEST(GPULZStage, IsGraphCompatible) {
    GPULZStage fwd;
    EXPECT_TRUE(fwd.isGraphCompatible());
    GPULZStage inv;
    inv.setInverse(true);
    EXPECT_FALSE(inv.isGraphCompatible());
}

TEST(GPULZStage, RepeatedRoundTripStable) {
    std::mt19937 rng(55);
    std::uniform_int_distribution<int> dist(0, 4);
    std::vector<uint8_t> data(4 * 2048);
    for (auto& b : data) b = (uint8_t)dist(rng);
    for (int i = 0; i < 5; i++) round_trip(data);
}

TEST(GPULZStage, ChunkSize1024RandomBytesRoundTrip) {
    std::mt19937 rng(101);
    std::uniform_int_distribution<int> dist(0, 255);
    std::vector<uint8_t> data(6 * 1024);  // multi-chunk
    for (auto& b : data) b = (uint8_t)dist(rng);
    round_trip(data, 4, 1024);
}

TEST(GPULZStage, ChunkSize4096RandomBytesRoundTrip) {
    std::mt19937 rng(103);
    std::uniform_int_distribution<int> dist(0, 255);
    std::vector<uint8_t> data(3 * 4096);
    for (auto& b : data) b = (uint8_t)dist(rng);
    round_trip(data, 4, 4096);
}

TEST(GPULZStage, PipelineIntegration) {
    const size_t N = 4096;  // element count -> 16384 bytes of float input
    auto h_input = make_smooth_data<float>(N);
    const size_t in_bytes = N * sizeof(float);

    Pipeline p(in_bytes, MemoryStrategy::PREALLOCATE);
    auto* gpulz = p.addStage<GPULZStage>();
    gpulz->setChunkSize(2048);
    gpulz->setWordSize(4);
    p.finalize();

    CudaStream cs;
    auto res = pipeline_round_trip<float>(p, h_input, cs.stream);
    EXPECT_LT(res.max_error, 1e-9f);  // lossless
}

TEST(GPULZStage, ZeroInput) {
    GPULZStage stage;
    CudaStream cs;
    auto pool = make_test_pool(64);

    CudaBuffer<uint8_t> d_dummy(1);
    std::vector<void*>  inputs  = {d_dummy.void_ptr()};
    std::vector<void*>  outputs = {d_dummy.void_ptr()};
    std::vector<size_t> sizes   = {0};
    EXPECT_NO_THROW(stage.execute(cs.stream, pool.get(), inputs, outputs, sizes));
    EXPECT_EQ(stage.getActualOutputSize(0), 0u);
}
