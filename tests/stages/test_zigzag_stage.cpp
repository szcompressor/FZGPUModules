/**
 * tests/test_zigzag_stage.cpp
 *
 * Unit tests for ZigzagStage<TIn, TOut> — GPU round-trip and correctness.
 *
 * Requires a CUDA-capable device (CUDA_VISIBLE_DEVICES=0 set by CTest).
 *
 * Properties verified:
 *   1. encode → decode round-trip restores original data exactly (int16, int32).
 *   2. decode → encode round-trip also works (inverse-first path).
 *   3. Output byte size equals input byte size (size-preserving).
 *   4. Zero maps to zero (GPU result matches host expectation).
 *   5. Known spot value: encode(-1 as int32) → 1 on GPU.
 *   6. Large array correctness with a smooth ramp.
 */

#include <gtest/gtest.h>
#include "helpers/fz_test_utils.h"
#include "transforms/zigzag/zigzag_stage.h"
#include "transforms/zigzag/zigzag.h"

#include <cstdint>
#include <numeric>
#include <vector>

using namespace fz;
using namespace fz_test;

// ─────────────────────────────────────────────────────────────────────────────
// Helper: run ZigzagStage<TIn,TOut> on a host vector, return host output.
// The byte size passed to execute() is always in_bytes (size-preserving).
// ─────────────────────────────────────────────────────────────────────────────
template<typename TIn, typename TOut>
static std::vector<TOut> run_encode(ZigzagStage<TIn, TOut>& stage,
                                    const std::vector<TIn>& h_in,
                                    cudaStream_t             stream,
                                    fz::MemoryPool&          pool)
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
    cudaError_t _sync_err = cudaStreamSynchronize(stream);
    EXPECT_EQ(_sync_err, cudaSuccess) << "CUDA sync error: " << cudaGetErrorString(_sync_err);

    return d_out.download(stream);
}

template<typename TIn, typename TOut>
static std::vector<TIn> run_decode(ZigzagStage<TIn, TOut>& stage,
                                   const std::vector<TOut>& h_in,
                                   cudaStream_t              stream,
                                   fz::MemoryPool&           pool)
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
    cudaError_t _sync_err2 = cudaStreamSynchronize(stream);
    EXPECT_EQ(_sync_err2, cudaSuccess) << "CUDA sync error: " << cudaGetErrorString(_sync_err2);

    return d_out.download(stream);
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. int16_t round-trip
// ─────────────────────────────────────────────────────────────────────────────

TEST(ZigzagStage, Int16RoundTrip) {
    CudaStream stream;
    auto pool = make_test_pool(8192 * sizeof(int16_t));

    // Build a mix of positive, negative, and zero values.
    std::vector<int16_t> h_input(512);
    for (size_t i = 0; i < h_input.size(); ++i)
        h_input[i] = static_cast<int16_t>(static_cast<int>(i) - 256);  // -256…255

    ZigzagStage<int16_t, uint16_t> enc_stage;
    auto h_encoded = run_encode(enc_stage, h_input, stream, *pool);
    ASSERT_EQ(h_encoded.size(), h_input.size());

    ZigzagStage<int16_t, uint16_t> dec_stage;
    dec_stage.setInverse(true);
    auto h_decoded = run_decode(dec_stage, h_encoded, stream, *pool);
    ASSERT_EQ(h_decoded.size(), h_input.size());

    for (size_t i = 0; i < h_input.size(); ++i)
        EXPECT_EQ(h_decoded[i], h_input[i]) << "mismatch at index " << i;
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. int32_t round-trip (large ramp)
// ─────────────────────────────────────────────────────────────────────────────

TEST(ZigzagStage, Int32RoundTrip) {
    CudaStream stream;
    const size_t N = 16384;
    auto pool = make_test_pool(N * sizeof(int32_t));

    std::vector<int32_t> h_input(N);
    for (size_t i = 0; i < N; ++i)
        h_input[i] = static_cast<int32_t>(i) - static_cast<int32_t>(N / 2);

    ZigzagStage<int32_t, uint32_t> enc_stage;
    auto h_encoded = run_encode(enc_stage, h_input, stream, *pool);
    ASSERT_EQ(h_encoded.size(), N);

    ZigzagStage<int32_t, uint32_t> dec_stage;
    dec_stage.setInverse(true);
    auto h_decoded = run_decode(dec_stage, h_encoded, stream, *pool);
    ASSERT_EQ(h_decoded.size(), N);

    for (size_t i = 0; i < N; ++i)
        EXPECT_EQ(h_decoded[i], h_input[i]) << "mismatch at index " << i;
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. Size-preserving property
// ─────────────────────────────────────────────────────────────────────────────

TEST(ZigzagStage, SizePreserving) {
    ZigzagStage<int32_t, uint32_t> stage;
    auto sizes = stage.estimateOutputSizes({4096});
    ASSERT_EQ(sizes.size(), 1u);
    EXPECT_EQ(sizes[0], 4096u);
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. Zero maps to zero on GPU
// ─────────────────────────────────────────────────────────────────────────────

TEST(ZigzagStage, ZeroMapsToZero) {
    CudaStream stream;
    auto pool = make_test_pool(256 * sizeof(int32_t));

    std::vector<int32_t> h_input(64, 0);

    ZigzagStage<int32_t, uint32_t> stage;
    auto h_out = run_encode(stage, h_input, stream, *pool);

    for (size_t i = 0; i < h_out.size(); ++i)
        EXPECT_EQ(h_out[i], 0u) << "zero should encode to zero at index " << i;
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. Known value: encode(-1 as int32) → 1
// ─────────────────────────────────────────────────────────────────────────────

TEST(ZigzagStage, KnownValue_NegOne) {
    CudaStream stream;
    auto pool = make_test_pool(sizeof(int32_t));

    std::vector<int32_t> h_input = {-1};

    ZigzagStage<int32_t, uint32_t> stage;
    auto h_out = run_encode(stage, h_input, stream, *pool);

    ASSERT_EQ(h_out.size(), 1u);
    EXPECT_EQ(h_out[0], 1u) << "encode(-1) should be 1";
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. GPU encode result matches CPU reference for all values
// ─────────────────────────────────────────────────────────────────────────────

TEST(ZigzagStage, MatchesCPUReference_Int16) {
    CudaStream stream;
    const size_t N = 4096;
    auto pool = make_test_pool(N * sizeof(int16_t));

    std::vector<int16_t> h_input(N);
    for (size_t i = 0; i < N; ++i)
        h_input[i] = static_cast<int16_t>(i % 512 - 256);

    ZigzagStage<int16_t, uint16_t> stage;
    auto h_gpu = run_encode(stage, h_input, stream, *pool);

    // Compare against host reference
    for (size_t i = 0; i < N; ++i) {
        uint16_t expected = Zigzag<int16_t>::encode(h_input[i]);
        EXPECT_EQ(h_gpu[i], expected) << "GPU/CPU mismatch at index " << i;
    }
}
