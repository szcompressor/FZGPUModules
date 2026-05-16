/**
 * tests/stages/test_huffman.cpp
 *
 * GPU unit tests for HuffmanStage<T>.
 * HuffmanStage Huffman-encodes a flat symbol stream (PHF coarse-grained) and
 * reconstructs it exactly. Not graph-compatible (two D2H syncs in forward execute).
 *
 *   HF1   HuffmanStage/RoundTrip_U16                  — uint16_t forward+inverse exact match
 *   HF2   HuffmanStage/RoundTrip_U8                   — uint8_t forward+inverse exact match
 *   HF3   HuffmanStage/ZeroInput                      — n=0 does not crash; output size is 0
 *   HF4   HuffmanStage/CompressedSmallerThanInput      — compressed size < input for skewed data
 *   HF5   HuffmanStage/SerializeDeserialize            — serializeHeader→deserializeHeader roundtrip
 *   HF6   HuffmanStage/SaveRestoreState                — saveState+deserializeHeader+restoreState
 *   HF7   HuffmanStage/GraphCompatible                 — isGraphCompatible()==false
 *   HF8   HuffmanStage/PipelineIntegration_U16         — Pipeline round-trip with uint16_t codes
 *   HF9   HuffmanStage/LorenzoQuantPipeline            — LorenzoQuant→Huffman end-to-end float round-trip
 *   HF10  HuffmanStage/RoundTrip_U32                  — uint32_t forward+inverse exact match
 *   HF11  HuffmanStage/ReuseAfterSizeChange            — shrink reuses existing buf_; grow triggers realloc; both correct
 *   HF12  HuffmanStage/OutOfRangeSymbolThrows          — symbols >= bklen throw std::runtime_error (not silent corruption)
 */

#include <gtest/gtest.h>
#include "helpers/fz_test_utils.h"
#include "helpers/stage_harness.h"
#include "coders/huffman/huffman_stage.h"
#include "fzgpumodules.h"

#include "fused/lorenzo_quant/lorenzo_quant.h"

#include <cstdint>
#include <cstring>
#include <numeric>
#include <vector>
#include <algorithm>

using namespace fz;
using namespace fz_test;

// ─────────────────────────────────────────────────────────────────────────────
// Helper: run one HuffmanStage encode pass, return the compressed host bytes
// ─────────────────────────────────────────────────────────────────────────────

template<typename T>
static std::vector<uint8_t> huffman_encode(
    HuffmanStage<T>& stage,
    const std::vector<T>& h_in,
    cudaStream_t stream,
    fz::MemoryPool& pool)
{
    const size_t in_bytes = h_in.size() * sizeof(T);
    const size_t out_est  = stage.estimateOutputSizes({in_bytes})[0];

    CudaBuffer<T>       d_in(std::max(h_in.size(), size_t(1)));
    CudaBuffer<uint8_t> d_out(std::max(out_est, size_t(1)));
    if (!h_in.empty()) { d_in.upload(h_in, stream); cudaStreamSynchronize(stream); }

    std::vector<void*>  inputs  = {d_in.void_ptr()};
    std::vector<void*>  outputs = {d_out.void_ptr()};
    std::vector<size_t> sizes   = {in_bytes};
    stage.execute(stream, &pool, inputs, outputs, sizes);
    cudaStreamSynchronize(stream);

    const size_t actual = stage.getActualOutputSize(0);
    std::vector<uint8_t> h_out(actual);
    if (actual > 0)
        cudaMemcpy(h_out.data(), d_out.get(), actual, cudaMemcpyDeviceToHost);
    return h_out;
}

template<typename T>
static std::vector<T> huffman_decode(
    HuffmanStage<T>& stage,
    const std::vector<uint8_t>& h_encoded,
    size_t num_elements,
    cudaStream_t stream,
    fz::MemoryPool& pool)
{
    const size_t in_bytes  = h_encoded.size();
    const size_t out_bytes = num_elements * sizeof(T);

    CudaBuffer<uint8_t> d_in(std::max(in_bytes, size_t(1)));
    CudaBuffer<T>       d_out(std::max(num_elements, size_t(1)));
    if (in_bytes > 0) {
        cudaMemcpy(d_in.get(), h_encoded.data(), in_bytes, cudaMemcpyHostToDevice);
        cudaStreamSynchronize(stream);
    }

    std::vector<void*>  inputs  = {d_in.void_ptr()};
    std::vector<void*>  outputs = {d_out.void_ptr()};
    std::vector<size_t> sizes   = {in_bytes};
    stage.setInverse(true);
    stage.execute(stream, &pool, inputs, outputs, sizes);
    cudaStreamSynchronize(stream);

    std::vector<T> h_out(num_elements);
    if (out_bytes > 0)
        cudaMemcpy(h_out.data(), d_out.get(), out_bytes, cudaMemcpyDeviceToHost);
    return h_out;
}

// ─────────────────────────────────────────────────────────────────────────────
// HF1 — RoundTrip_U16
// ─────────────────────────────────────────────────────────────────────────────
TEST(HuffmanStage, RoundTrip_U16) {
    const size_t N = 4096;
    std::vector<uint16_t> h_in(N);
    for (size_t i = 0; i < N; ++i) h_in[i] = static_cast<uint16_t>(i % 128);

    CudaStream cs;
    auto pool = make_test_pool(N * sizeof(uint16_t));

    HuffmanStage<uint16_t> stage;
    stage.setBklen(1024);

    auto encoded = huffman_encode(stage, h_in, cs.stream, *pool);
    ASSERT_GT(encoded.size(), 0u);

    // original_len_ was set by the forward pass; setInverse inside huffman_decode
    stage.setInverse(false);  // reset for inverse call inside helper
    auto decoded = huffman_decode(stage, encoded, N, cs.stream, *pool);
    ASSERT_EQ(decoded.size(), N);
    for (size_t i = 0; i < N; ++i)
        EXPECT_EQ(decoded[i], h_in[i]) << "mismatch at i=" << i;
}

// ─────────────────────────────────────────────────────────────────────────────
// HF2 — RoundTrip_U8
// ─────────────────────────────────────────────────────────────────────────────
TEST(HuffmanStage, RoundTrip_U8) {
    const size_t N = 2048;
    std::vector<uint8_t> h_in(N);
    for (size_t i = 0; i < N; ++i) h_in[i] = static_cast<uint8_t>(i % 64);

    CudaStream cs;
    auto pool = make_test_pool(N * sizeof(uint8_t));

    HuffmanStage<uint8_t> stage;
    // bklen=256 is the default for uint8_t

    auto encoded = huffman_encode(stage, h_in, cs.stream, *pool);
    ASSERT_GT(encoded.size(), 0u);

    stage.setInverse(false);
    auto decoded = huffman_decode(stage, encoded, N, cs.stream, *pool);
    ASSERT_EQ(decoded.size(), N);
    for (size_t i = 0; i < N; ++i)
        EXPECT_EQ(decoded[i], h_in[i]) << "mismatch at i=" << i;
}

// ─────────────────────────────────────────────────────────────────────────────
// HF3 — ZeroInput
// ─────────────────────────────────────────────────────────────────────────────
TEST(HuffmanStage, ZeroInput) {
    HuffmanStage<uint16_t> stage;
    stage.setBklen(1024);

    CudaStream cs;
    auto pool = make_test_pool(64);

    CudaBuffer<uint16_t> d_dummy(1);
    std::vector<void*>  inputs  = {d_dummy.void_ptr()};
    std::vector<void*>  outputs = {d_dummy.void_ptr()};
    std::vector<size_t> sizes   = {0};
    EXPECT_NO_THROW(stage.execute(cs.stream, pool.get(), inputs, outputs, sizes));
    EXPECT_EQ(stage.getActualOutputSize(0), 0u);
}

// ─────────────────────────────────────────────────────────────────────────────
// HF4 — CompressedSmallerThanInput
// ─────────────────────────────────────────────────────────────────────────────
TEST(HuffmanStage, CompressedSmallerThanInput) {
    const size_t N = 8192;
    // 90% zeros, 10% ones — highly compressible
    std::vector<uint16_t> h_in(N, 0);
    for (size_t i = 0; i < N / 10; ++i) h_in[i * 10] = 1;

    CudaStream cs;
    auto pool = make_test_pool(N * sizeof(uint16_t));

    HuffmanStage<uint16_t> stage;
    stage.setBklen(1024);

    auto encoded = huffman_encode(stage, h_in, cs.stream, *pool);
    EXPECT_LT(encoded.size(), N * sizeof(uint16_t))
        << "highly skewed input should compress below raw size";
}

// ─────────────────────────────────────────────────────────────────────────────
// HF5 — SerializeDeserialize
// ─────────────────────────────────────────────────────────────────────────────
TEST(HuffmanStage, SerializeDeserialize) {
    HuffmanStage<uint16_t> original;
    original.setBklen(512);

    uint8_t buf[128] = {};
    size_t written = original.serializeHeader(0, buf, sizeof(buf));
    EXPECT_EQ(written, 11u);
    EXPECT_EQ(buf[0], static_cast<uint8_t>(DataType::UINT16));

    HuffmanStage<uint16_t> restored;
    restored.deserializeHeader(buf, written);
    EXPECT_EQ(restored.getBklen(), 512u);
}

// ─────────────────────────────────────────────────────────────────────────────
// HF6 — SaveRestoreState
// ─────────────────────────────────────────────────────────────────────────────
TEST(HuffmanStage, SaveRestoreState) {
    HuffmanStage<uint16_t> s;
    s.setBklen(512);
    s.saveState();

    // Simulate deserializeHeader overwriting bklen (as the pipeline does during decompress)
    uint8_t fake_hdr[11] = {};
    fake_hdr[0] = static_cast<uint8_t>(DataType::UINT16);
    uint16_t bk_alt = 2048;
    std::memcpy(fake_hdr + 1, &bk_alt, 2);
    s.deserializeHeader(fake_hdr, sizeof(fake_hdr));
    EXPECT_EQ(s.getBklen(), 2048u);

    s.restoreState();
    EXPECT_EQ(s.getBklen(), 512u);
}

// ─────────────────────────────────────────────────────────────────────────────
// HF7 — GraphCompatible
// ─────────────────────────────────────────────────────────────────────────────
TEST(HuffmanStage, GraphCompatible) {
    HuffmanStage<uint16_t> stage;
    EXPECT_FALSE(stage.isGraphCompatible());
}

// ─────────────────────────────────────────────────────────────────────────────
// HF8 — PipelineIntegration_U16
// Full Pipeline round-trip using pipeline_round_trip<T> harness.
// ─────────────────────────────────────────────────────────────────────────────
TEST(HuffmanStage, PipelineIntegration_U16) {
    const size_t N        = 4096;
    const size_t in_bytes = N * sizeof(uint16_t);

    // Data: ascending codes in [0, 255], fits well within bklen=1024
    std::vector<uint16_t> h_in(N);
    for (size_t i = 0; i < N; ++i) h_in[i] = static_cast<uint16_t>(i % 256);

    Pipeline p(in_bytes, MemoryStrategy::PREALLOCATE);
    auto* stage = p.addStage<HuffmanStage<uint16_t>>();
    stage->setBklen(1024);
    p.finalize();

    CudaStream cs;
    auto res = pipeline_round_trip<uint16_t>(p, h_in, cs.stream);

    ASSERT_EQ(res.data.size(), N);
    for (size_t i = 0; i < N; ++i)
        EXPECT_EQ(res.data[i], h_in[i]) << "mismatch at i=" << i;
    EXPECT_GT(res.compressed_bytes, 0u);
}

// ─────────────────────────────────────────────────────────────────────────────
// HF10 — RoundTrip_U32
// ─────────────────────────────────────────────────────────────────────────────
TEST(HuffmanStage, RoundTrip_U32) {
    const size_t N = 2048;
    std::vector<uint32_t> h_in(N);
    for (size_t i = 0; i < N; ++i) h_in[i] = static_cast<uint32_t>(i % 64);

    CudaStream cs;
    auto pool = make_test_pool(N * sizeof(uint32_t));

    HuffmanStage<uint32_t> stage;
    stage.setBklen(256);  // symbols in [0, 63] ⊂ [0, 256)

    auto encoded = huffman_encode(stage, h_in, cs.stream, *pool);
    ASSERT_GT(encoded.size(), 0u);

    stage.setInverse(false);
    auto decoded = huffman_decode(stage, encoded, N, cs.stream, *pool);
    ASSERT_EQ(decoded.size(), N);
    for (size_t i = 0; i < N; ++i)
        EXPECT_EQ(decoded[i], h_in[i]) << "mismatch at i=" << i;
}

// ─────────────────────────────────────────────────────────────────────────────
// HF11 — ReuseAfterSizeChange
// Verifies capacity-based reallocation:
//   Pass 1 (N1=8192): initial allocation; cap_inlen_ = 8192.
//   Pass 2 (N2=2048): inlen < cap_inlen_ — existing buf_ is reused (no realloc).
// Both round-trips must produce exact output.
// ─────────────────────────────────────────────────────────────────────────────
TEST(HuffmanStage, ReuseAfterSizeChange) {
    CudaStream cs;

    HuffmanStage<uint16_t> stage;
    stage.setBklen(1024);

    // ── First call: large input ───────────────────────────────────────────────
    const size_t N1 = 8192;
    std::vector<uint16_t> h_in1(N1);
    for (size_t i = 0; i < N1; ++i) h_in1[i] = static_cast<uint16_t>(i % 256);

    auto pool1 = make_test_pool(N1 * sizeof(uint16_t));
    auto encoded1 = huffman_encode(stage, h_in1, cs.stream, *pool1);
    ASSERT_GT(encoded1.size(), 0u);

    stage.setInverse(false);
    auto decoded1 = huffman_decode(stage, encoded1, N1, cs.stream, *pool1);
    ASSERT_EQ(decoded1.size(), N1);
    for (size_t i = 0; i < N1; ++i)
        EXPECT_EQ(decoded1[i], h_in1[i]) << "pass1 mismatch at i=" << i;

    // ── Second call: smaller input — reuses existing buf_ (cap_inlen_=8192) ──
    const size_t N2 = 2048;
    std::vector<uint16_t> h_in2(N2);
    for (size_t i = 0; i < N2; ++i) h_in2[i] = static_cast<uint16_t>((i * 7) % 128);

    auto pool2 = make_test_pool(N2 * sizeof(uint16_t));
    stage.setInverse(false);
    auto encoded2 = huffman_encode(stage, h_in2, cs.stream, *pool2);
    ASSERT_GT(encoded2.size(), 0u);

    stage.setInverse(false);
    auto decoded2 = huffman_decode(stage, encoded2, N2, cs.stream, *pool2);
    ASSERT_EQ(decoded2.size(), N2);
    for (size_t i = 0; i < N2; ++i)
        EXPECT_EQ(decoded2[i], h_in2[i]) << "pass2 mismatch at i=" << i;
}

// ─────────────────────────────────────────────────────────────────────────────
// HF12 — OutOfRangeSymbolThrows
// With symbol range validation enabled, any symbol >= bklen must throw
// std::runtime_error rather than silently corrupting the bitstream.
// ─────────────────────────────────────────────────────────────────────────────
TEST(HuffmanStage, OutOfRangeSymbolThrows) {
    const size_t N = 1024;
    std::vector<uint16_t> h_in(N, 0);
    h_in[42] = 100;  // out of range: bklen=64 means [0,64) is valid

    CudaStream cs;
    auto pool = make_test_pool(N * sizeof(uint16_t));

    HuffmanStage<uint16_t> stage;
    stage.setBklen(64);

    EXPECT_THROW(huffman_encode(stage, h_in, cs.stream, *pool), std::runtime_error);
}

// ─────────────────────────────────────────────────────────────────────────────
// HF9 — LorenzoQuantPipeline
// Full end-to-end: LorenzoQuantStage<float,uint16_t> → HuffmanStage<uint16_t>
// Codes port of Lorenzo feeds Huffman; outlier outputs remain unconnected.
// ─────────────────────────────────────────────────────────────────────────────
TEST(HuffmanStage, LorenzoQuantPipeline) {
    constexpr size_t N  = 1 << 14;   // 16 K floats — typical small field slice
    constexpr float  EB = 1e-2f;
    const size_t in_bytes = N * sizeof(float);

    auto h_in = make_smooth_data<float>(N);

    Pipeline p(in_bytes, MemoryStrategy::MINIMAL);

    auto* lq = p.addStage<LorenzoQuantStage<float, uint16_t>>();
    lq->setErrorBound(EB);
    lq->setQuantRadius(512);
    lq->setOutlierCapacity(0.2f);
    // Zigzag maps signed deltas → non-negative codes in [0, 2*radius−2] = [0,1022],
    // which fits within HuffmanStage's bklen=1024 symbol range.
    lq->setZigzagCodes(true);

    auto* huf = p.addStage<HuffmanStage<uint16_t>>();
    huf->setBklen(1024);
    p.connect(huf, lq, "codes");

    p.finalize();

    CudaStream cs;
    auto res = pipeline_round_trip<float>(p, h_in, cs.stream);

    ASSERT_EQ(res.data.size(), N);
    EXPECT_GT(res.compressed_bytes, 0u);
    EXPECT_LT(res.compressed_bytes, in_bytes)
        << "compressed size should be smaller than raw for smooth data";
    EXPECT_LE(res.max_error, static_cast<double>(EB) * 1.01)
        << "max reconstruction error " << res.max_error << " exceeds bound " << EB;
}
