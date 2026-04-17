/**
 * tests/test_difference.cpp
 *
 * Unit tests for DifferenceStage  (forward + inverse round-trip).
 *
 * The Difference stage is a lossless transform:
 *   Forward : output[0] = input[0];  output[i] = input[i] - input[i-1]
 *   Inverse : output[0] = input[0];  output[i] = input[i] + output[i-1]
 *
 * Key properties verified here:
 *   1. Forward transform changes values (not a no-op).
 *   2. Forward followed by Inverse reconstructs the original exactly.
 *   3. Output size equals input size (lossless / size-preserving).
 *   4. Chunked mode: diffs reset at chunk boundaries.
 *   5. Negabinary output (DifferenceStage<T, UnsignedT>): encode+decode round-trip.
 *   6. Chunked + negabinary combined.
 *   7. Header serialization preserves chunk_size across serialize/deserialize.
 */

#include <gtest/gtest.h>
#include "helpers/fz_test_utils.h"
#include "encoders/diff/diff.h"
#include "transforms/negabinary/negabinary.h"

#include <cmath>
#include <cstring>
#include <numeric>
#include <vector>

using namespace fz;
using namespace fz_test;

// ─────────────────────────────────────────────────────────────────────────────
// Helper: allocate output buffer using estimateOutputSizes, run stage, return
//         host copy of the output.
// ─────────────────────────────────────────────────────────────────────────────
template <typename T>
static std::vector<T> run_diff_stage(Stage& stage,
                                     const std::vector<T>& h_input,
                                     cudaStream_t           stream,
                                     fz::MemoryPool&        pool) {
    size_t n = h_input.size();
    size_t in_bytes = n * sizeof(T);

    // Host → device input
    CudaBuffer<T> d_in(n);
    d_in.upload(h_input, stream);

    // Estimate and allocate output
    auto est = stage.estimateOutputSizes({in_bytes});
    EXPECT_FALSE(est.empty());
    size_t out_bytes = est[0];
    size_t out_n = out_bytes / sizeof(T);

    CudaBuffer<T> d_out(out_n);

    // Execute
    std::vector<void*> inputs  = {d_in.void_ptr()};
    std::vector<void*> outputs = {d_out.void_ptr()};
    std::vector<size_t> sizes  = {in_bytes};

    stage.execute(stream, &pool, inputs, outputs, sizes);
    cudaStreamSynchronize(stream);

    // Trim to actual size
    auto actual_map = stage.getActualOutputSizesByName();
    size_t actual_bytes = actual_map.count("output") ? actual_map.at("output") : out_bytes;
    size_t actual_n = actual_bytes / sizeof(T);

    auto h_out = d_out.download(stream);
    h_out.resize(actual_n);
    return h_out;
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: float round-trip
// ─────────────────────────────────────────────────────────────────────────────
TEST(DifferenceStage, FloatRoundTrip) {
    CudaStream stream;
    auto pool = make_test_pool(1024 * sizeof(float));

    std::vector<float> h_input = {1.0f, 3.0f, 6.0f, 10.0f, 15.0f, 21.0f, 28.0f};

    // ─── Forward pass ───
    DifferenceStage<float> fwd_stage;
    auto h_diff = run_diff_stage(fwd_stage, h_input, stream, *pool);

    ASSERT_EQ(h_diff.size(), h_input.size()) << "Forward pass must preserve element count";

    // First element unchanged; rest are differences
    EXPECT_FLOAT_EQ(h_diff[0], h_input[0]);
    for (size_t i = 1; i < h_input.size(); i++) {
        EXPECT_FLOAT_EQ(h_diff[i], h_input[i] - h_input[i - 1])
            << "Mismatch at index " << i;
    }

    // ─── Inverse pass ───
    DifferenceStage<float> inv_stage;
    inv_stage.setInverse(true);
    auto h_reconstructed = run_diff_stage(inv_stage, h_diff, stream, *pool);

    ASSERT_EQ(h_reconstructed.size(), h_input.size());
    for (size_t i = 0; i < h_input.size(); i++) {
        EXPECT_FLOAT_EQ(h_reconstructed[i], h_input[i])
            << "Reconstruction mismatch at index " << i;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: uint16_t round-trip (typical after Lorenzo quantization codes)
// ─────────────────────────────────────────────────────────────────────────────
TEST(DifferenceStage, Uint16RoundTrip) {
    CudaStream stream;
    auto pool = make_test_pool(4096 * sizeof(uint16_t));

    // Build a ramp of 256 uint16_t values
    std::vector<uint16_t> h_input(256);
    for (size_t i = 0; i < h_input.size(); i++)
        h_input[i] = static_cast<uint16_t>(i * 3 + 100);

    DifferenceStage<uint16_t> fwd;
    auto h_diff = run_diff_stage(fwd, h_input, stream, *pool);
    ASSERT_EQ(h_diff.size(), h_input.size());

    DifferenceStage<uint16_t> inv;
    inv.setInverse(true);
    auto h_recon = run_diff_stage(inv, h_diff, stream, *pool);

    ASSERT_EQ(h_recon.size(), h_input.size());
    for (size_t i = 0; i < h_input.size(); i++) {
        EXPECT_EQ(h_recon[i], h_input[i]) << "Mismatch at index " << i;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: constant input → differences are all zero after first element
// ─────────────────────────────────────────────────────────────────────────────
TEST(DifferenceStage, ConstantInput) {
    CudaStream stream;
    auto pool = make_test_pool(512 * sizeof(float));

    std::vector<float> h_input(512, 3.14f);

    DifferenceStage<float> stage;
    auto h_diff = run_diff_stage(stage, h_input, stream, *pool);

    ASSERT_EQ(h_diff.size(), h_input.size());
    EXPECT_FLOAT_EQ(h_diff[0], 3.14f);
    for (size_t i = 1; i < h_diff.size(); i++) {
        EXPECT_FLOAT_EQ(h_diff[i], 0.0f) << "Expected zero diff at index " << i;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: single-element input (no neighbour → passthrough)
// ─────────────────────────────────────────────────────────────────────────────
TEST(DifferenceStage, SingleElement) {
    CudaStream stream;
    auto pool = make_test_pool(64);

    std::vector<float> h_input = {42.0f};

    DifferenceStage<float> fwd;
    auto h_diff = run_diff_stage(fwd, h_input, stream, *pool);
    ASSERT_EQ(h_diff.size(), 1u);
    EXPECT_FLOAT_EQ(h_diff[0], 42.0f);  // single element: no diff, value passes through

    DifferenceStage<float> inv;
    inv.setInverse(true);
    auto h_recon = run_diff_stage(inv, h_diff, stream, *pool);
    ASSERT_EQ(h_recon.size(), 1u);
    EXPECT_FLOAT_EQ(h_recon[0], 42.0f);
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: uint16_t alternating max / min — exercises unsigned wrap-around
//
// For uint16_t, subtraction is modular:
//   diff[0] = 0
//   diff[1] = 65535 - 0     = 65535
//   diff[2] = 0 - 65535     = 1     (wraps)
//   diff[3] = 65535 - 0     = 65535
//   ...
// The inverse (prefix-sum) must also use modular addition to recover the
// original sequence exactly.
// ─────────────────────────────────────────────────────────────────────────────
TEST(DifferenceStage, Uint16WrappingArithmetic) {
    CudaStream stream;
    auto pool = make_test_pool(128 * sizeof(uint16_t));

    constexpr size_t N = 64;
    std::vector<uint16_t> h_input(N);
    for (size_t i = 0; i < N; i++)
        h_input[i] = (i % 2 == 0) ? uint16_t(0) : std::numeric_limits<uint16_t>::max();

    DifferenceStage<uint16_t> fwd;
    auto h_diff = run_diff_stage(fwd, h_input, stream, *pool);
    ASSERT_EQ(h_diff.size(), N);

    DifferenceStage<uint16_t> inv;
    inv.setInverse(true);
    auto h_recon = run_diff_stage(inv, h_diff, stream, *pool);

    ASSERT_EQ(h_recon.size(), N);
    for (size_t i = 0; i < N; i++)
        EXPECT_EQ(h_recon[i], h_input[i]) << "Mismatch at index " << i;
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: large random-ish data round-trip
// ─────────────────────────────────────────────────────────────────────────────
TEST(DifferenceStage, LargeRoundTrip) {
    CudaStream stream;
    constexpr size_t N = 1 << 16;  // 64 K floats
    auto pool = make_test_pool(N * sizeof(float) * 4);

    auto h_input = make_sine_floats(N, 0.01f, 100.0f);

    DifferenceStage<float> fwd;
    auto h_diff = run_diff_stage(fwd, h_input, stream, *pool);

    DifferenceStage<float> inv;
    inv.setInverse(true);
    auto h_recon = run_diff_stage(inv, h_diff, stream, *pool);

    ASSERT_EQ(h_recon.size(), h_input.size());
    float max_err = 0.0f;
    for (size_t i = 0; i < N; i++) {
        float err = std::abs(h_recon[i] - h_input[i]);
        max_err = std::max(max_err, err);
    }
    // Float32 prefix-sum accumulates ~1 ULP per step.  Over 64 K elements with
    // values up to ~100 the worst-case accumulated error is O(N * eps * max_val)
    // ≈ 64K * 1.2e-7 * 100 ≈ 7.6e-4, so 1e-3 is the appropriate bound.
    EXPECT_LT(max_err, 1e-3f) << "Max reconstruction error too large: " << max_err;
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: 1 M float elements — correctness at scale, no OOM
// ─────────────────────────────────────────────────────────────────────────────
TEST(DifferenceStage, OneMillion) {
    CudaStream stream;
    constexpr size_t N = 1 << 20;  // 1 M floats
    auto pool = make_test_pool(N * sizeof(float) * 4);

    auto h_input = make_sine_floats(N, 1e-5f, 100.0f);

    DifferenceStage<float> fwd;
    auto h_diff = run_diff_stage(fwd, h_input, stream, *pool);
    ASSERT_EQ(h_diff.size(), N);

    DifferenceStage<float> inv;
    inv.setInverse(true);
    auto h_recon = run_diff_stage(inv, h_diff, stream, *pool);

    ASSERT_EQ(h_recon.size(), N);
    // Float32 prefix-sum over 1 M values of magnitude ~100 accumulates
    // O(N * eps * max_val) ≈ 1M * 1.2e-7 * 100 ≈ 0.012 maximum floating error.
    float max_err = 0.0f;
    for (size_t i = 0; i < N; i++)
        max_err = std::max(max_err, std::abs(h_recon[i] - h_input[i]));
    EXPECT_LT(max_err, 0.1f) << "1M element max reconstruction error: " << max_err;
}

// ═════════════════════════════════════════════════════════════════════════════
//  Chunked difference tests
// ═════════════════════════════════════════════════════════════════════════════

// ─────────────────────────────────────────────────────────────────────────────
// Helper: run typed DifferenceStage<TIn, TOut> forward pass; return TOut vector.
// ─────────────────────────────────────────────────────────────────────────────
template<typename TIn, typename TOut>
static std::vector<TOut> run_diff_fwd(DifferenceStage<TIn, TOut>& stage,
                                      const std::vector<TIn>& h_in,
                                      cudaStream_t stream,
                                      fz::MemoryPool& pool)
{
    const size_t n        = h_in.size();
    const size_t in_bytes = n * sizeof(TIn);

    CudaBuffer<TIn>  d_in(n);
    CudaBuffer<TOut> d_out(n);
    d_in.upload(h_in, stream);

    std::vector<void*> inputs  = {d_in.void_ptr()};
    std::vector<void*> outputs = {d_out.void_ptr()};
    std::vector<size_t> sizes  = {in_bytes};

    stage.execute(stream, &pool, inputs, outputs, sizes);
    cudaError_t e = cudaStreamSynchronize(stream);
    EXPECT_EQ(e, cudaSuccess) << cudaGetErrorString(e);
    return d_out.download(stream);
}

// Helper: run inverse pass; TOut input → TIn output.
template<typename TIn, typename TOut>
static std::vector<TIn> run_diff_inv(DifferenceStage<TIn, TOut>& stage,
                                      const std::vector<TOut>& h_in,
                                      cudaStream_t stream,
                                      fz::MemoryPool& pool)
{
    const size_t n        = h_in.size();
    const size_t in_bytes = n * sizeof(TOut);

    CudaBuffer<TOut> d_in(n);
    CudaBuffer<TIn>  d_out(n);
    d_in.upload(h_in, stream);

    std::vector<void*> inputs  = {d_in.void_ptr()};
    std::vector<void*> outputs = {d_out.void_ptr()};
    std::vector<size_t> sizes  = {in_bytes};

    stage.execute(stream, &pool, inputs, outputs, sizes);
    cudaError_t e = cudaStreamSynchronize(stream);
    EXPECT_EQ(e, cudaSuccess) << cudaGetErrorString(e);
    return d_out.download(stream);
}

// ─────────────────────────────────────────────────────────────────────────────
// Chunking: forward boundaries
//
// Verifies that the first element of each chunk is stored as-is (no diff),
// while elements within a chunk are differenced from their predecessor.
// ─────────────────────────────────────────────────────────────────────────────
TEST(DifferenceStage, ChunkBoundariesAreIndependent) {
    CudaStream stream;
    auto pool = make_test_pool(1024 * sizeof(int32_t) * 4);

    // 12 elements in 3 chunks of 4
    // Values are a simple ramp so diffs are all 1 — except at chunk starts.
    std::vector<int32_t> h_input = {10, 11, 12, 13,   // chunk 0
                                    20, 21, 22, 23,   // chunk 1
                                    30, 31, 32, 33};  // chunk 2

    DifferenceStage<int32_t> fwd;
    fwd.setChunkSize(4 * sizeof(int32_t));  // 4 elements per chunk

    auto h_diff = run_diff_fwd(fwd, h_input, stream, *pool);
    ASSERT_EQ(h_diff.size(), h_input.size());

    // Chunk boundaries: stored as-is
    EXPECT_EQ(h_diff[0], 10);   // chunk 0 start
    EXPECT_EQ(h_diff[4], 20);   // chunk 1 start
    EXPECT_EQ(h_diff[8], 30);   // chunk 2 start

    // Within each chunk: difference from predecessor
    for (int c = 0; c < 3; ++c) {
        for (int i = 1; i < 4; ++i) {
            int idx = c * 4 + i;
            EXPECT_EQ(h_diff[idx], int32_t(1))
                << "Expected diff=1 at chunk " << c << " element " << i;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Chunking: round-trip (int32)
// ─────────────────────────────────────────────────────────────────────────────
TEST(DifferenceStage, ChunkedInt32RoundTrip) {
    CudaStream stream;
    constexpr size_t N = 4096;
    constexpr size_t CHUNK_ELEMS = 512;
    auto pool = make_test_pool(N * sizeof(int32_t) * 4);

    std::vector<int32_t> h_input(N);
    for (size_t i = 0; i < N; ++i)
        h_input[i] = static_cast<int32_t>(std::sin(static_cast<float>(i) * 0.05f) * 1000.0f);

    DifferenceStage<int32_t> fwd;
    fwd.setChunkSize(CHUNK_ELEMS * sizeof(int32_t));

    auto h_diff = run_diff_fwd(fwd, h_input, stream, *pool);
    ASSERT_EQ(h_diff.size(), N);

    DifferenceStage<int32_t> inv;
    inv.setInverse(true);
    inv.setChunkSize(CHUNK_ELEMS * sizeof(int32_t));

    auto h_recon = run_diff_inv(inv, h_diff, stream, *pool);
    ASSERT_EQ(h_recon.size(), N);
    for (size_t i = 0; i < N; ++i)
        EXPECT_EQ(h_recon[i], h_input[i]) << "Mismatch at index " << i;
}

// ─────────────────────────────────────────────────────────────────────────────
// Chunking: uint16 round-trip (typical Lorenzo code path)
// ─────────────────────────────────────────────────────────────────────────────
TEST(DifferenceStage, ChunkedUint16RoundTrip) {
    CudaStream stream;
    constexpr size_t N = 8192;
    constexpr size_t CHUNK_ELEMS = 1024;
    auto pool = make_test_pool(N * sizeof(uint16_t) * 4);

    std::vector<uint16_t> h_input(N);
    for (size_t i = 0; i < N; ++i)
        h_input[i] = static_cast<uint16_t>((i * 3 + 7) % 60000);

    DifferenceStage<uint16_t> fwd;
    fwd.setChunkSize(CHUNK_ELEMS * sizeof(uint16_t));

    auto h_diff = run_diff_fwd(fwd, h_input, stream, *pool);
    ASSERT_EQ(h_diff.size(), N);

    DifferenceStage<uint16_t> inv;
    inv.setInverse(true);
    inv.setChunkSize(CHUNK_ELEMS * sizeof(uint16_t));

    auto h_recon = run_diff_inv(inv, h_diff, stream, *pool);
    ASSERT_EQ(h_recon.size(), N);
    for (size_t i = 0; i < N; ++i)
        EXPECT_EQ(h_recon[i], h_input[i]) << "Mismatch at index " << i;
}

// ─────────────────────────────────────────────────────────────────────────────
// Chunking: chunk-start elements are NOT affected by previous chunks
//
// Verifies independence by checking that the reconstruction of chunk N does
// not diverge due to carry from chunk N-1.
// ─────────────────────────────────────────────────────────────────────────────
TEST(DifferenceStage, ChunksAreIndependentContexts) {
    CudaStream stream;
    auto pool = make_test_pool(64 * sizeof(int32_t) * 4);

    // Two chunks of 8, each starting at a very different value.
    std::vector<int32_t> h_input = {1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007,
                                    -500, -499, -498, -497, -496, -495, -494, -493};

    DifferenceStage<int32_t> fwd;
    fwd.setChunkSize(8 * sizeof(int32_t));

    auto h_diff = run_diff_fwd(fwd, h_input, stream, *pool);
    ASSERT_EQ(h_diff.size(), 16u);

    // Chunk 1 start: no carry from chunk 0; stored as -500, not (−500 − 1007)
    EXPECT_EQ(h_diff[8], -500);

    DifferenceStage<int32_t> inv;
    inv.setInverse(true);
    inv.setChunkSize(8 * sizeof(int32_t));

    auto h_recon = run_diff_inv(inv, h_diff, stream, *pool);
    for (size_t i = 0; i < 16; ++i)
        EXPECT_EQ(h_recon[i], h_input[i]) << "Mismatch at index " << i;
}

// ═════════════════════════════════════════════════════════════════════════════
//  Negabinary-fused tests  (DifferenceStage<T, UnsignedT>)
// ═════════════════════════════════════════════════════════════════════════════

// ─────────────────────────────────────────────────────────────────────────────
// Negabinary opt-in: basic int32 round-trip
// ─────────────────────────────────────────────────────────────────────────────
TEST(DifferenceStage, NegabinaryFusedInt32RoundTrip) {
    CudaStream stream;
    constexpr size_t N = 1024;
    auto pool = make_test_pool(N * sizeof(int32_t) * 4);

    std::vector<int32_t> h_input(N);
    for (size_t i = 0; i < N; ++i)
        h_input[i] = static_cast<int32_t>(std::sin(static_cast<float>(i) * 0.1f) * 500.0f);

    // Forward: int32 → uint32 (with negabinary encoding of the differences)
    DifferenceStage<int32_t, uint32_t> fwd;
    auto h_encoded = run_diff_fwd(fwd, h_input, stream, *pool);
    ASSERT_EQ(h_encoded.size(), N);

    // Spot-check: verify a few encoded values match what we expect
    // diff[0] = h_input[0] (stored as-is), then negabinary-encoded
    int32_t d0 = h_input[0];
    EXPECT_EQ(h_encoded[0], fz::Negabinary<int32_t>::encode(d0));

    // Inverse: uint32 → int32 (decode negabinary then cumsum)
    DifferenceStage<int32_t, uint32_t> inv;
    inv.setInverse(true);
    auto h_recon = run_diff_inv(inv, h_encoded, stream, *pool);

    ASSERT_EQ(h_recon.size(), N);
    for (size_t i = 0; i < N; ++i)
        EXPECT_EQ(h_recon[i], h_input[i]) << "Mismatch at index " << i;
}

// ─────────────────────────────────────────────────────────────────────────────
// Negabinary opt-in: int16 round-trip
// ─────────────────────────────────────────────────────────────────────────────
TEST(DifferenceStage, NegabinaryFusedInt16RoundTrip) {
    CudaStream stream;
    constexpr size_t N = 2000;
    auto pool = make_test_pool(N * sizeof(int16_t) * 4);

    std::vector<int16_t> h_input(N);
    for (size_t i = 0; i < N; ++i)
        h_input[i] = static_cast<int16_t>((static_cast<int>(i % 500) - 250) * 3);

    DifferenceStage<int16_t, uint16_t> fwd;
    auto h_enc = run_diff_fwd(fwd, h_input, stream, *pool);
    ASSERT_EQ(h_enc.size(), N);

    DifferenceStage<int16_t, uint16_t> inv;
    inv.setInverse(true);
    auto h_recon = run_diff_inv(inv, h_enc, stream, *pool);

    ASSERT_EQ(h_recon.size(), N);
    for (size_t i = 0; i < N; ++i)
        EXPECT_EQ(h_recon[i], h_input[i]) << "Mismatch at index " << i;
}

// ─────────────────────────────────────────────────────────────────────────────
// Negabinary: constant input → all diffs are 0 → encoded as 0 → decoded back to 0
// ─────────────────────────────────────────────────────────────────────────────
TEST(DifferenceStage, NegabinaryConstantInput) {
    CudaStream stream;
    auto pool = make_test_pool(512 * sizeof(int32_t) * 2);

    std::vector<int32_t> h_input(512, 42);
    DifferenceStage<int32_t, uint32_t> fwd;
    auto h_enc = run_diff_fwd(fwd, h_input, stream, *pool);

    ASSERT_EQ(h_enc.size(), 512u);

    // First element: negabinary_encode(42)
    EXPECT_EQ(h_enc[0], fz::Negabinary<int32_t>::encode(42));
    // Remaining: diff == 0; negabinary_encode(0) == 0
    for (size_t i = 1; i < h_enc.size(); ++i)
        EXPECT_EQ(h_enc[i], uint32_t(0)) << "Expected 0 at index " << i;
}

// ─────────────────────────────────────────────────────────────────────────────
// Negabinary + chunking: combined round-trip  (the primary PFPL code path)
// ─────────────────────────────────────────────────────────────────────────────
TEST(DifferenceStage, NegabinaryChunkedRoundTrip) {
    CudaStream stream;
    // Simulate the PFPL arrangement: 16 KB chunks of int32 (4096 elements each)
    constexpr size_t CHUNK_ELEMS = 4096;
    constexpr size_t NUM_CHUNKS  = 4;
    constexpr size_t N           = CHUNK_ELEMS * NUM_CHUNKS;
    auto pool = make_test_pool(N * sizeof(int32_t) * 4);

    std::vector<int32_t> h_input(N);
    for (size_t c = 0; c < NUM_CHUNKS; ++c) {
        // Each chunk starts at a different base value
        int32_t base = static_cast<int32_t>(c) * 10000;
        for (size_t i = 0; i < CHUNK_ELEMS; ++i)
            h_input[c * CHUNK_ELEMS + i] =
                base + static_cast<int32_t>(std::sin(static_cast<float>(i) * 0.01f) * 100.0f);
    }

    DifferenceStage<int32_t, uint32_t> fwd;
    fwd.setChunkSize(CHUNK_ELEMS * sizeof(int32_t));
    auto h_enc = run_diff_fwd(fwd, h_input, stream, *pool);
    ASSERT_EQ(h_enc.size(), N);

    // Verify chunk starts are negabinary-encoded base values (no carry from prev chunk)
    for (size_t c = 0; c < NUM_CHUNKS; ++c) {
        uint32_t expected = fz::Negabinary<int32_t>::encode(h_input[c * CHUNK_ELEMS]);
        EXPECT_EQ(h_enc[c * CHUNK_ELEMS], expected)
            << "Chunk " << c << " start value mismatch";
    }

    DifferenceStage<int32_t, uint32_t> inv;
    inv.setInverse(true);
    inv.setChunkSize(CHUNK_ELEMS * sizeof(int32_t));
    auto h_recon = run_diff_inv(inv, h_enc, stream, *pool);

    ASSERT_EQ(h_recon.size(), N);
    for (size_t i = 0; i < N; ++i)
        EXPECT_EQ(h_recon[i], h_input[i]) << "Mismatch at index " << i;
}

// ─────────────────────────────────────────────────────────────────────────────
// Header serialization: chunk_size and TOut survive a serialize/deserialize
// round-trip for DifferenceStage<int32_t, uint32_t>.
// ─────────────────────────────────────────────────────────────────────────────
TEST(DifferenceStage, HeaderSerializationPreservesChunkSize) {
    using Stage = DifferenceStage<int32_t, uint32_t>;

    Stage original;
    original.setChunkSize(16384);

    uint8_t buf[16] = {};
    size_t written = original.serializeHeader(0, buf, sizeof(buf));
    ASSERT_EQ(written, 6u) << "Expected 6-byte header (TIn + TOut + chunk_size)";

    Stage restored;
    restored.deserializeHeader(buf, written);
    EXPECT_EQ(restored.getChunkSize(), size_t(16384));
}

// ─────────────────────────────────────────────────────────────────────────────
// DifferenceStage<double> round-trip
//
// Verifies that the double-precision specialisation is instantiated and correct:
//   - forward(double input) produces exact differences
//   - inverse(double diffs) reconstructs the original without any precision loss
//     (DifferenceStage is a lossless transform; no approximation tolerance needed)
// ─────────────────────────────────────────────────────────────────────────────
TEST(DifferenceStage, DoubleRoundTrip) {
    CudaStream stream;
    constexpr size_t N = 1024;
    auto pool = make_test_pool(N * sizeof(double) * 4);

    // Use integer-valued doubles: the GPU parallel prefix scan accumulates
    // additions exactly for integer arguments, so EXPECT_DOUBLE_EQ holds.
    // h_input[i] = i*3 keeps values well within the exact integer range of
    // double (< 2^53), and forward differences h_diff[i!=0] = 3.0 (exact).
    std::vector<double> h_input(N);
    for (size_t i = 0; i < N; ++i)
        h_input[i] = static_cast<double>(i) * 3.0;

    DifferenceStage<double> fwd;
    auto h_diff = run_diff_stage(fwd, h_input, stream, *pool);
    ASSERT_EQ(h_diff.size(), N);

    // Spot-check forward: first element unchanged, rest are exact differences
    EXPECT_DOUBLE_EQ(h_diff[0], h_input[0]);
    for (size_t i = 1; i < N; ++i)
        EXPECT_DOUBLE_EQ(h_diff[i], h_input[i] - h_input[i - 1])
            << "Forward mismatch at index " << i;

    DifferenceStage<double> inv;
    inv.setInverse(true);
    auto h_recon = run_diff_stage(inv, h_diff, stream, *pool);
    ASSERT_EQ(h_recon.size(), N);

    for (size_t i = 0; i < N; ++i)
        EXPECT_DOUBLE_EQ(h_recon[i], h_input[i])
            << "Reconstruction mismatch at index " << i;
}

// ─────────────────────────────────────────────────────────────────────────────
// Header serialization: vanilla DifferenceStage<float> (TOut = T) still produces
// a valid 6-byte header with chunk_size 0.
// ─────────────────────────────────────────────────────────────────────────────
TEST(DifferenceStage, LegacyFloatHeaderCompatible) {
    DifferenceStage<float> stage;

    uint8_t buf[16] = {};
    size_t written = stage.serializeHeader(0, buf, sizeof(buf));
    ASSERT_EQ(written, 6u);

    // TIn == FLOAT32, TOut == FLOAT32, chunk_size == 0
    EXPECT_EQ(buf[0], static_cast<uint8_t>(fz::DataType::FLOAT32));
    EXPECT_EQ(buf[1], static_cast<uint8_t>(fz::DataType::FLOAT32));
    uint32_t cs = 0;
    std::memcpy(&cs, buf + 2, 4);
    EXPECT_EQ(cs, uint32_t(0));
}
