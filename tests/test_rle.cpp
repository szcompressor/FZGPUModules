/**
 * tests/test_rle.cpp
 *
 * Unit tests for RLEStage (run-length encoding / decoding).
 *
 * Forward: encodes consecutive identical values as (value, count) pairs.
 *   Format: [num_runs:u32] [val0:T, count0:u32,  val1:T, count1:u32, ...]
 * Inverse: expands pairs back to the original sequence.
 *
 * Key properties verified:
 *   1. Encoding a high-repetition sequence produces a smaller output.
 *   2. Forward followed by Inverse reconstructs the original exactly.
 *   3. Works correctly on data with no repeated values (worst case).
 *   4. Works on larger scale data.
 */

#include <gtest/gtest.h>
#include "helpers/fz_test_utils.h"
#include "encoders/RLE/rle.h"
#include "fzgpumodules.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

using namespace fz;
using namespace fz_test;

// ─────────────────────────────────────────────────────────────────────────────
// Helper: run a stage on h_input (type T), return host output.
// ─────────────────────────────────────────────────────────────────────────────
template <typename T>
static std::vector<uint8_t> run_rle_forward(RLEStage<T>&          stage,
                                             const std::vector<T>& h_input,
                                             cudaStream_t          stream,
                                             fz::MemoryPool&       pool) {
    size_t in_bytes = h_input.size() * sizeof(T);

    CudaBuffer<T> d_in(h_input.size());
    d_in.upload(h_input, stream);

    auto est = stage.estimateOutputSizes({in_bytes});
    EXPECT_FALSE(est.empty());
    size_t out_bytes = est[0];

    CudaBuffer<uint8_t> d_out(out_bytes);

    std::vector<void*> inputs  = {d_in.void_ptr()};
    std::vector<void*> outputs = {d_out.void_ptr()};
    std::vector<size_t> sizes  = {in_bytes};

    stage.execute(stream, &pool, inputs, outputs, sizes);
    cudaStreamSynchronize(stream);

    auto actual_map  = stage.getActualOutputSizesByName();
    size_t act_bytes = actual_map.count("output") ? actual_map.at("output") : out_bytes;

    auto h_out = d_out.download(stream);
    h_out.resize(act_bytes);
    return h_out;
}

template <typename T>
static std::vector<T> run_rle_inverse(RLEStage<T>&               stage,
                                      const std::vector<uint8_t>& h_encoded,
                                      size_t                       n_original,
                                      cudaStream_t                 stream,
                                      fz::MemoryPool&              pool) {
    size_t in_bytes = h_encoded.size();

    CudaBuffer<uint8_t> d_in(in_bytes);
    d_in.upload(h_encoded, stream);

    // Conservative upper bound for decompressed output
    size_t out_bytes = n_original * sizeof(T) * 2;
    CudaBuffer<T> d_out(out_bytes / sizeof(T));

    std::vector<void*> inputs  = {d_in.void_ptr()};
    std::vector<void*> outputs = {d_out.void_ptr()};
    std::vector<size_t> sizes  = {in_bytes};

    stage.execute(stream, &pool, inputs, outputs, sizes);
    cudaStreamSynchronize(stream);

    auto actual_map  = stage.getActualOutputSizesByName();
    size_t act_bytes = actual_map.count("output") ? actual_map.at("output") : n_original * sizeof(T);
    size_t act_n     = act_bytes / sizeof(T);

    auto h_out = d_out.download(stream);
    h_out.resize(act_n);
    return h_out;
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: single-element input (one run of length 1)
// ─────────────────────────────────────────────────────────────────────────────
TEST(RLEStage, SingleElement) {
    CudaStream stream;
    auto pool = make_test_pool(256);

    std::vector<uint16_t> h_input = {99};

    RLEStage<uint16_t> fwd;
    auto h_encoded = run_rle_forward(fwd, h_input, stream, *pool);

    // Encoded: [num_runs=1 : u32][val=99 : u16][count=1 : u32] = 10 bytes
    EXPECT_GE(h_encoded.size(), sizeof(uint32_t) + sizeof(uint16_t) + sizeof(uint32_t));

    RLEStage<uint16_t> inv;
    inv.setInverse(true);
    auto h_decoded = run_rle_inverse(inv, h_encoded, 1, stream, *pool);

    ASSERT_EQ(h_decoded.size(), 1u);
    EXPECT_EQ(h_decoded[0], uint16_t(99));
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: num_runs header field matches actual number of distinct runs
//
// The first 4 bytes of the encoded buffer store num_runs as a uint32_t.
// For the sequence [7,7,7, 42,42, 1] we expect exactly 3 runs.
// ─────────────────────────────────────────────────────────────────────────────
TEST(RLEStage, NumRunsHeaderField) {
    CudaStream stream;
    auto pool = make_test_pool(4096);

    // 3 distinct runs: (7×3), (42×2), (1×1)
    std::vector<uint16_t> h_input = {7, 7, 7, 42, 42, 1};

    RLEStage<uint16_t> fwd;
    auto h_encoded = run_rle_forward(fwd, h_input, stream, *pool);

    // The format starts with num_runs as a uint32_t
    ASSERT_GE(h_encoded.size(), sizeof(uint32_t)) << "Encoded buffer too small to contain num_runs";
    uint32_t num_runs = 0;
    std::memcpy(&num_runs, h_encoded.data(), sizeof(uint32_t));
    EXPECT_EQ(num_runs, 3u) << "Expected 3 runs for {7,7,7,42,42,1}";

    // Round-trip should still be perfect
    RLEStage<uint16_t> inv;
    inv.setInverse(true);
    auto h_decoded = run_rle_inverse(inv, h_encoded, h_input.size(), stream, *pool);
    ASSERT_EQ(h_decoded.size(), h_input.size());
    for (size_t i = 0; i < h_input.size(); i++)
        EXPECT_EQ(h_decoded[i], h_input[i]) << "Round-trip mismatch at " << i;
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: simple known sequence round-trip
// ─────────────────────────────────────────────────────────────────────────────
TEST(RLEStage, SimpleRoundTrip) {
    CudaStream stream;
    auto pool = make_test_pool(4096);

    // [7,7,7, 42,42, 1]  → 3 runs → should encode compactly
    std::vector<uint16_t> h_input = {7, 7, 7, 42, 42, 1};

    RLEStage<uint16_t> fwd;
    auto h_encoded = run_rle_forward(fwd, h_input, stream, *pool);

    RLEStage<uint16_t> inv;
    inv.setInverse(true);
    auto h_decoded = run_rle_inverse(inv, h_encoded, h_input.size(), stream, *pool);

    ASSERT_EQ(h_decoded.size(), h_input.size()) << "Decoded length mismatch";
    for (size_t i = 0; i < h_input.size(); i++) {
        EXPECT_EQ(h_decoded[i], h_input[i]) << "Mismatch at index " << i;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: all-same sequence — ideal case for RLE compression
// ─────────────────────────────────────────────────────────────────────────────
TEST(RLEStage, AllSameRoundTrip) {
    CudaStream stream;
    auto pool = make_test_pool(4096);

    constexpr size_t N = 1024;
    std::vector<uint16_t> h_input(N, 0);  // all zeros — 1 run

    RLEStage<uint16_t> fwd;
    auto h_encoded = run_rle_forward(fwd, h_input, stream, *pool);

    // Encoded size should be much smaller: 4 (num_runs) + 2 (val) + 4 (count) = 10 bytes
    EXPECT_LT(h_encoded.size(), h_input.size() * sizeof(uint16_t))
        << "RLE should compress constant sequences";

    RLEStage<uint16_t> inv;
    inv.setInverse(true);
    auto h_decoded = run_rle_inverse(inv, h_encoded, N, stream, *pool);

    ASSERT_EQ(h_decoded.size(), N);
    for (size_t i = 0; i < N; i++) {
        EXPECT_EQ(h_decoded[i], 0) << "Mismatch at index " << i;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: alternating values — worst case for RLE (no compression gain)
// ─────────────────────────────────────────────────────────────────────────────
TEST(RLEStage, AlternatingRoundTrip) {
    CudaStream stream;
    auto pool = make_test_pool(4096 * 8);

    constexpr size_t N = 256;
    std::vector<uint16_t> h_input(N);
    for (size_t i = 0; i < N; i++)
        h_input[i] = static_cast<uint16_t>(i % 2);  // 0,1,0,1,...

    RLEStage<uint16_t> fwd;
    auto h_encoded = run_rle_forward(fwd, h_input, stream, *pool);

    RLEStage<uint16_t> inv;
    inv.setInverse(true);
    auto h_decoded = run_rle_inverse(inv, h_encoded, N, stream, *pool);

    ASSERT_EQ(h_decoded.size(), N);
    for (size_t i = 0; i < N; i++) {
        EXPECT_EQ(h_decoded[i], h_input[i]) << "Mismatch at index " << i;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: large realistic quantization-code-like sequence (mostly zeros with
//       occasional non-zero values, mimicking Lorenzo output after difference)
// ─────────────────────────────────────────────────────────────────────────────
TEST(RLEStage, LargeSparseRoundTrip) {
    CudaStream stream;
    constexpr size_t N = 1 << 14;  // 16 K elements
    auto pool = make_test_pool(N * sizeof(uint16_t) * 10);

    // ~90 % zeros, sparse non-zeros
    std::vector<uint16_t> h_input(N, 0);
    for (size_t i = 0; i < N; i += 11)
        h_input[i] = static_cast<uint16_t>((i % 127) + 1);

    RLEStage<uint16_t> fwd;
    auto h_encoded = run_rle_forward(fwd, h_input, stream, *pool);

    RLEStage<uint16_t> inv;
    inv.setInverse(true);
    auto h_decoded = run_rle_inverse(inv, h_encoded, N, stream, *pool);

    ASSERT_EQ(h_decoded.size(), N);
    for (size_t i = 0; i < N; i++) {
        EXPECT_EQ(h_decoded[i], h_input[i]) << "Mismatch at index " << i;
    }
}
