/**
 * tests/test_negabinary_stage.cpp
 *
 * GPU unit tests for NegabinaryStage<TIn, TOut>.
 *
 * NegabinaryStage applies negabinary encoding or decoding element-wise:
 *   Forward (encode):  TIn[]  → TOut[]   (signed → unsigned, Negabinary<TIn>::encode)
 *   Inverse (decode):  TOut[] → TIn[]   (unsigned → signed, Negabinary<TIn>::decode)
 *
 * Properties verified:
 *   1. encode → decode round-trip restores original data (int16, int32).
 *   2. decode → encode round-trip also works (inverse-first path).
 *   3. Output byte size equals input byte size (size-preserving).
 *   4. Zero maps to zero: encode(0) == 0.
 *   5. Known spot values: encode(-1) == 3, encode(1) == 1, encode(-2) == 2.
 *   6. Large smooth ramp: GPU results match CPU reference for int16.
 */

#include <gtest/gtest.h>
#include "helpers/fz_test_utils.h"
#include "transforms/negabinary/negabinary_stage.h"
#include "transforms/negabinary/negabinary.h"

#include <cstdint>
#include <numeric>
#include <vector>

using namespace fz;
using namespace fz_test;

// ─────────────────────────────────────────────────────────────────────────────
// Helper: run NegabinaryStage<TIn,TOut> encode on host vector → device → host.
// ─────────────────────────────────────────────────────────────────────────────
template<typename TIn, typename TOut>
static std::vector<TOut> run_nb_encode(NegabinaryStage<TIn, TOut>& stage,
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

template<typename TIn, typename TOut>
static std::vector<TIn> run_nb_decode(NegabinaryStage<TIn, TOut>& stage,
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
// Test: int16 encode-then-decode round-trip
// ─────────────────────────────────────────────────────────────────────────────
TEST(NegabinaryStage, Int16RoundTrip) {
    CudaStream stream;
    constexpr size_t N = 2048;
    auto pool = make_test_pool(N * sizeof(int16_t) * 4);

    std::vector<int16_t> h_input(N);
    for (size_t i = 0; i < N; ++i)
        h_input[i] = static_cast<int16_t>(static_cast<int>(i % 400) - 200);

    NegabinaryStage<int16_t, uint16_t> enc;
    auto h_enc = run_nb_encode(enc, h_input, stream, *pool);
    ASSERT_EQ(h_enc.size(), N);

    NegabinaryStage<int16_t, uint16_t> dec;
    dec.setInverse(true);
    auto h_dec = run_nb_decode(dec, h_enc, stream, *pool);

    ASSERT_EQ(h_dec.size(), N);
    for (size_t i = 0; i < N; ++i)
        EXPECT_EQ(h_dec[i], h_input[i]) << "Round-trip mismatch at index " << i;
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: int32 encode-then-decode round-trip
// ─────────────────────────────────────────────────────────────────────────────
TEST(NegabinaryStage, Int32RoundTrip) {
    CudaStream stream;
    constexpr size_t N = 4096;
    auto pool = make_test_pool(N * sizeof(int32_t) * 4);

    std::vector<int32_t> h_input(N);
    for (size_t i = 0; i < N; ++i)
        h_input[i] = static_cast<int32_t>(i % 2000) - 1000;

    NegabinaryStage<int32_t, uint32_t> enc;
    auto h_enc = run_nb_encode(enc, h_input, stream, *pool);

    NegabinaryStage<int32_t, uint32_t> dec;
    dec.setInverse(true);
    auto h_dec = run_nb_decode(dec, h_enc, stream, *pool);

    ASSERT_EQ(h_dec.size(), N);
    for (size_t i = 0; i < N; ++i)
        EXPECT_EQ(h_dec[i], h_input[i]) << "Round-trip mismatch at index " << i;
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: output byte size equals input byte size
// ─────────────────────────────────────────────────────────────────────────────
TEST(NegabinaryStage, SizePreserving) {
    CudaStream stream;
    constexpr size_t N = 256;
    auto pool = make_test_pool(N * sizeof(int32_t) * 2);

    std::vector<int32_t> h_input(N, 7);
    NegabinaryStage<int32_t, uint32_t> stage;

    size_t in_bytes = N * sizeof(int32_t);
    auto est = stage.estimateOutputSizes({in_bytes});
    ASSERT_FALSE(est.empty());
    EXPECT_EQ(est[0], in_bytes);

    auto h_enc = run_nb_encode(stage, h_input, stream, *pool);
    ASSERT_EQ(h_enc.size(), N);

    auto actual = stage.getActualOutputSizesByName();
    EXPECT_EQ(actual.at("output"), in_bytes);
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: zero maps to zero (encode(0) == 0)
// ─────────────────────────────────────────────────────────────────────────────
TEST(NegabinaryStage, ZeroMapsToZero) {
    CudaStream stream;
    auto pool = make_test_pool(64 * sizeof(int32_t));

    std::vector<int32_t> h_input(32, 0);
    NegabinaryStage<int32_t, uint32_t> stage;
    auto h_enc = run_nb_encode(stage, h_input, stream, *pool);

    ASSERT_EQ(h_enc.size(), 32u);
    for (size_t i = 0; i < h_enc.size(); ++i)
        EXPECT_EQ(h_enc[i], uint32_t(0)) << "encode(0) should be 0 at index " << i;
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: known spot values
//   encode(-1) == 3   (negabinary: (-1) + 0xAAAAAAAA = 0xAAAAAAAA - 1 = 0xAAAAAAA9; XOR mask = 3)
//   encode( 1) == 1
//   encode(-2) == 2
//   encode( 2) == 6
// ─────────────────────────────────────────────────────────────────────────────
TEST(NegabinaryStage, KnownSpotValues) {
    CudaStream stream;
    auto pool = make_test_pool(16 * sizeof(int32_t));

    std::vector<int32_t> h_input = {-1, 1, -2, 2};
    NegabinaryStage<int32_t, uint32_t> stage;
    auto h_enc = run_nb_encode(stage, h_input, stream, *pool);

    ASSERT_EQ(h_enc.size(), 4u);
    EXPECT_EQ(h_enc[0], uint32_t(3)) << "encode(-1) should be 3";
    EXPECT_EQ(h_enc[1], uint32_t(1)) << "encode(1) should be 1";
    EXPECT_EQ(h_enc[2], uint32_t(2)) << "encode(-2) should be 2";
    EXPECT_EQ(h_enc[3], uint32_t(6)) << "encode(2) should be 6";
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: GPU results match host CPU reference for int16
// ─────────────────────────────────────────────────────────────────────────────
TEST(NegabinaryStage, MatchesCPUReferenceInt16) {
    CudaStream stream;
    constexpr size_t N = 512;
    auto pool = make_test_pool(N * sizeof(int16_t) * 4);

    std::vector<int16_t> h_input(N);
    for (size_t i = 0; i < N; ++i)
        h_input[i] = static_cast<int16_t>(static_cast<int>(i % 300) - 150);

    NegabinaryStage<int16_t, uint16_t> stage;
    auto h_gpu = run_nb_encode(stage, h_input, stream, *pool);

    // CPU reference
    for (size_t i = 0; i < N; ++i) {
        uint16_t expected = Negabinary<int16_t>::encode(h_input[i]);
        EXPECT_EQ(h_gpu[i], expected) << "GPU/CPU mismatch at index " << i;
    }
}
