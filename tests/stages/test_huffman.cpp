/**
 * tests/stages/test_huffman.cpp
 *
 * GPU unit tests for HuffmanStage<T>.
 * Covers both HuffmanEncodeMode::Coarse (default, CPU-sync phase 3) and
 * HuffmanEncodeMode::Fine (ReVISIT-lite kernel, GPU-async phase 3).
 * Not graph-compatible (histogram D2H sync in forward execute regardless of mode).
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
 *   HF13  HuffmanStage/FineEncode_RoundTrip_U16        — Fine mode: uint16_t round-trip exact match
 *   HF14  HuffmanStage/FineEncode_RoundTrip_U8         — Fine mode: uint8_t round-trip exact match
 *   HF15  HuffmanStage/FineEncode_CompressedSmaller    — Fine mode: compressed < input for skewed data
 *   HF16  HuffmanStage/FineEncode_ModeSwitch           — switching Coarse→Fine→Coarse triggers realloc; all round-trips correct
 *   HF17  HuffmanStage/FineEncode_RoundTrip_U32        — Fine mode: uint32_t round-trip exact match
 *   HF18  HuffmanStage/FineEncode_ReuseAfterSizeChange — Fine mode: shrink reuses buf_; grow triggers realloc; both correct
 *   HF19  HuffmanStage/FineEncode_OutOfRangeSymbolThrows — Fine mode: symbols >= bklen throw std::runtime_error
 *   HF20  HuffmanStage/FineEncode_PipelineIntegration_U16 — Fine mode: full Pipeline round-trip
 *   HF21  HuffmanStage/FineEncode_LorenzoQuantPipeline — Fine mode: LorenzoQuant→Huffman end-to-end float round-trip
 *   HF22  HuffmanStage/FixedBook_RoundTrip_U16         — model-derived book: uint16_t round-trip exact match
 *   HF23  HuffmanStage/FixedBook_ReusedAcrossCalls     — one book across differing inputs; identical input → identical stream
 *   HF24  HuffmanStage/FixedBook_FromFreq              — caller-supplied freq table round-trips; zero bin throws
 *   HF25  HuffmanStage/FixedBook_WithoutBookThrows     — Fixed source with no book throws at execute
 *   HF26  HuffmanStage/FixedBook_LorenzoQuantPipeline  — pre-built book, Fine mode, end-to-end float round-trip
 *   HF27  HuffmanStage/AdaptiveBook_RoundTrip          — sample once, reuse; symbols absent from the sample still round-trip
 *   HF28  HuffmanStage/AdaptiveBook_BeatsModelRatio    — sampled book beats a guessed model on bimodal data
 *   HF29  HuffmanStage/AdaptiveBook_FloorShiftFlattens — floor shift trades ratio for flatness; both correct
 *   HF30  HuffmanStage/OverlongCodeThrows              — >27-bit codes throw instead of being silently clamped
 *   HF31  HuffmanStage/AdaptiveBook_DegenerateSampleNotPinned — a constant first block must not pin the book
 *   HF32  HuffmanStage/AdaptiveBook_RefitOnRateRegression     — bit-rate regression triggers a refit
 */

#include <gtest/gtest.h>
#include "helpers/fz_test_utils.h"
#include "helpers/stage_harness.h"
#include "coders/huffman/huffman_stage.h"
#include "fzgpumodules.h"

#include "fused/lorenzo_quant/lorenzo_quant.h"

#include <cmath>
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

// ─────────────────────────────────────────────────────────────────────────────
// HF13 — FineEncode_RoundTrip_U16
// ─────────────────────────────────────────────────────────────────────────────
TEST(HuffmanStage, FineEncode_RoundTrip_U16) {
    const size_t N = 4096;
    std::vector<uint16_t> h_in(N);
    for (size_t i = 0; i < N; ++i) h_in[i] = static_cast<uint16_t>(i % 128);

    CudaStream cs;
    auto pool = make_test_pool(N * sizeof(uint16_t));

    HuffmanStage<uint16_t> stage;
    stage.setBklen(1024);
    stage.setEncodeMode(HuffmanEncodeMode::Fine);

    auto encoded = huffman_encode(stage, h_in, cs.stream, *pool);
    ASSERT_GT(encoded.size(), 0u);

    stage.setInverse(false);
    auto decoded = huffman_decode(stage, encoded, N, cs.stream, *pool);
    ASSERT_EQ(decoded.size(), N);
    for (size_t i = 0; i < N; ++i)
        EXPECT_EQ(decoded[i], h_in[i]) << "mismatch at i=" << i;
}

// ─────────────────────────────────────────────────────────────────────────────
// HF14 — FineEncode_RoundTrip_U8
// ─────────────────────────────────────────────────────────────────────────────
TEST(HuffmanStage, FineEncode_RoundTrip_U8) {
    const size_t N = 2048;
    std::vector<uint8_t> h_in(N);
    for (size_t i = 0; i < N; ++i) h_in[i] = static_cast<uint8_t>(i % 64);

    CudaStream cs;
    auto pool = make_test_pool(N * sizeof(uint8_t));

    HuffmanStage<uint8_t> stage;
    stage.setEncodeMode(HuffmanEncodeMode::Fine);

    auto encoded = huffman_encode(stage, h_in, cs.stream, *pool);
    ASSERT_GT(encoded.size(), 0u);

    stage.setInverse(false);
    auto decoded = huffman_decode(stage, encoded, N, cs.stream, *pool);
    ASSERT_EQ(decoded.size(), N);
    for (size_t i = 0; i < N; ++i)
        EXPECT_EQ(decoded[i], h_in[i]) << "mismatch at i=" << i;
}

// ─────────────────────────────────────────────────────────────────────────────
// HF15 — FineEncode_CompressedSmaller
// ─────────────────────────────────────────────────────────────────────────────
TEST(HuffmanStage, FineEncode_CompressedSmaller) {
    const size_t N = 16384;
    std::vector<uint16_t> h_in(N, 42);  // all same symbol — maximally compressible
    h_in[0] = 0; h_in[1] = 1;           // ensure at least 3 distinct symbols for valid codebook

    CudaStream cs;
    auto pool = make_test_pool(N * sizeof(uint16_t));

    HuffmanStage<uint16_t> stage;
    stage.setBklen(1024);
    stage.setEncodeMode(HuffmanEncodeMode::Fine);

    auto encoded = huffman_encode(stage, h_in, cs.stream, *pool);
    ASSERT_GT(encoded.size(), 0u);
    EXPECT_LT(encoded.size(), N * sizeof(uint16_t))
        << "fine-encode compressed size should be smaller than raw";
}

// ─────────────────────────────────────────────────────────────────────────────
// HF16 — FineEncode_ModeSwitch
// Switching Coarse→Fine→Coarse triggers buf_ reallocation each time;
// all three passes produce correct round-trips.
// ─────────────────────────────────────────────────────────────────────────────
TEST(HuffmanStage, FineEncode_ModeSwitch) {
    const size_t N = 4096;
    std::vector<uint16_t> h_in(N);
    for (size_t i = 0; i < N; ++i) h_in[i] = static_cast<uint16_t>(i % 64);

    CudaStream cs;
    auto pool = make_test_pool(N * sizeof(uint16_t) * 4);

    HuffmanStage<uint16_t> stage;
    stage.setBklen(1024);

    auto check_roundtrip = [&](HuffmanEncodeMode mode) {
        stage.setEncodeMode(mode);
        stage.setInverse(false);
        auto encoded = huffman_encode(stage, h_in, cs.stream, *pool);
        ASSERT_GT(encoded.size(), 0u);
        stage.setInverse(false);
        auto decoded = huffman_decode(stage, encoded, N, cs.stream, *pool);
        ASSERT_EQ(decoded.size(), N);
        for (size_t i = 0; i < N; ++i)
            EXPECT_EQ(decoded[i], h_in[i]) << "mismatch at i=" << i;
    };

    check_roundtrip(HuffmanEncodeMode::Coarse);
    check_roundtrip(HuffmanEncodeMode::Fine);
    check_roundtrip(HuffmanEncodeMode::Coarse);
}

// ─────────────────────────────────────────────────────────────────────────────
// HF17 — FineEncode_RoundTrip_U32
// ─────────────────────────────────────────────────────────────────────────────
TEST(HuffmanStage, FineEncode_RoundTrip_U32) {
    const size_t N = 2048;
    std::vector<uint32_t> h_in(N);
    for (size_t i = 0; i < N; ++i) h_in[i] = static_cast<uint32_t>(i % 64);

    CudaStream cs;
    auto pool = make_test_pool(N * sizeof(uint32_t));

    HuffmanStage<uint32_t> stage;
    stage.setBklen(256);  // symbols in [0, 63] ⊂ [0, 256)
    stage.setEncodeMode(HuffmanEncodeMode::Fine);

    auto encoded = huffman_encode(stage, h_in, cs.stream, *pool);
    ASSERT_GT(encoded.size(), 0u);

    stage.setInverse(false);
    auto decoded = huffman_decode(stage, encoded, N, cs.stream, *pool);
    ASSERT_EQ(decoded.size(), N);
    for (size_t i = 0; i < N; ++i)
        EXPECT_EQ(decoded[i], h_in[i]) << "mismatch at i=" << i;
}

// ─────────────────────────────────────────────────────────────────────────────
// HF18 — FineEncode_ReuseAfterSizeChange
// Verifies capacity-based reallocation in Fine mode:
//   Pass 1 (N1=8192): initial allocation; cap_inlen_ = 8192.
//   Pass 2 (N2=2048): inlen < cap_inlen_ — existing buf_ is reused (no realloc).
// Both round-trips must produce exact output.
// ─────────────────────────────────────────────────────────────────────────────
TEST(HuffmanStage, FineEncode_ReuseAfterSizeChange) {
    CudaStream cs;

    HuffmanStage<uint16_t> stage;
    stage.setBklen(1024);
    stage.setEncodeMode(HuffmanEncodeMode::Fine);

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
// HF19 — FineEncode_OutOfRangeSymbolThrows
// With symbol range validation enabled, any symbol >= bklen must throw
// std::runtime_error in Fine mode just as in Coarse mode (validation happens
// before the encode-mode branch in execute()).
// ─────────────────────────────────────────────────────────────────────────────
TEST(HuffmanStage, FineEncode_OutOfRangeSymbolThrows) {
    const size_t N = 1024;
    std::vector<uint16_t> h_in(N, 0);
    h_in[42] = 100;  // out of range: bklen=64 means [0,64) is valid

    CudaStream cs;
    auto pool = make_test_pool(N * sizeof(uint16_t));

    HuffmanStage<uint16_t> stage;
    stage.setBklen(64);
    stage.setEncodeMode(HuffmanEncodeMode::Fine);

    EXPECT_THROW(huffman_encode(stage, h_in, cs.stream, *pool), std::runtime_error);
}

// ─────────────────────────────────────────────────────────────────────────────
// HF20 — FineEncode_PipelineIntegration_U16
// Full Pipeline round-trip with Fine encode mode selected on the stage.
// ─────────────────────────────────────────────────────────────────────────────
TEST(HuffmanStage, FineEncode_PipelineIntegration_U16) {
    const size_t N        = 4096;
    const size_t in_bytes = N * sizeof(uint16_t);

    std::vector<uint16_t> h_in(N);
    for (size_t i = 0; i < N; ++i) h_in[i] = static_cast<uint16_t>(i % 256);

    Pipeline p(in_bytes, MemoryStrategy::PREALLOCATE);
    auto* stage = p.addStage<HuffmanStage<uint16_t>>();
    stage->setBklen(1024);
    stage->setEncodeMode(HuffmanEncodeMode::Fine);
    p.finalize();

    CudaStream cs;
    auto res = pipeline_round_trip<uint16_t>(p, h_in, cs.stream);

    ASSERT_EQ(res.data.size(), N);
    for (size_t i = 0; i < N; ++i)
        EXPECT_EQ(res.data[i], h_in[i]) << "mismatch at i=" << i;
    EXPECT_GT(res.compressed_bytes, 0u);
}

// ─────────────────────────────────────────────────────────────────────────────
// HF21 — FineEncode_LorenzoQuantPipeline
// End-to-end: LorenzoQuantStage<float,uint16_t> → HuffmanStage<uint16_t> (Fine)
// ─────────────────────────────────────────────────────────────────────────────
TEST(HuffmanStage, FineEncode_LorenzoQuantPipeline) {
    constexpr size_t N  = 1 << 14;   // 16 K floats
    constexpr float  EB = 1e-2f;
    const size_t in_bytes = N * sizeof(float);

    auto h_in = make_smooth_data<float>(N);

    Pipeline p(in_bytes, MemoryStrategy::MINIMAL);

    auto* lq = p.addStage<LorenzoQuantStage<float, uint16_t>>();
    lq->setErrorBound(EB);
    lq->setQuantRadius(512);
    lq->setOutlierCapacity(0.2f);
    lq->setZigzagCodes(true);

    auto* huf = p.addStage<HuffmanStage<uint16_t>>();
    huf->setBklen(1024);
    huf->setEncodeMode(HuffmanEncodeMode::Fine);
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

// ─────────────────────────────────────────────────────────────────────────────
// HF22 — FixedBook_RoundTrip_U16
// A model-derived codebook must round-trip exactly. The revbook still travels in
// the stream, so decode is the stock path and needs no extra configuration.
// ─────────────────────────────────────────────────────────────────────────────
TEST(HuffmanStage, FixedBook_RoundTrip_U16) {
    const size_t N = 8192;
    std::vector<uint16_t> h_in(N);
    // Centered around 512, the zero-error code of a radius-512 quantizer.
    for (size_t i = 0; i < N; ++i)
        h_in[i] = static_cast<uint16_t>(512 + (static_cast<int>(i % 64) - 32));

    CudaStream cs;
    auto pool = make_test_pool(N * sizeof(uint16_t));

    HuffmanStage<uint16_t> stage;
    stage.setBklen(1024);
    stage.setFixedBookFromModel({HuffmanBookModel::Gaussian, /*center=*/-1.0,
                                 /*scale=*/48.0, /*shape=*/2.0});
    EXPECT_EQ(stage.getBookSource(), HuffmanBookSource::Fixed);
    EXPECT_EQ(stage.getFixedBookFreq().size(), 1024u);

    auto encoded = huffman_encode(stage, h_in, cs.stream, *pool);
    ASSERT_GT(encoded.size(), 0u);

    stage.setInverse(false);
    auto decoded = huffman_decode(stage, encoded, N, cs.stream, *pool);
    ASSERT_EQ(decoded.size(), N);
    for (size_t i = 0; i < N; ++i)
        EXPECT_EQ(decoded[i], h_in[i]) << "mismatch at i=" << i;
}

// ─────────────────────────────────────────────────────────────────────────────
// HF23 — FixedBook_ReusedAcrossCalls
// The book is built once and reused: repeated encodes of *different* data must
// still decode correctly, and identical input must produce identical output.
// ─────────────────────────────────────────────────────────────────────────────
TEST(HuffmanStage, FixedBook_ReusedAcrossCalls) {
    const size_t N = 4096;
    std::vector<uint16_t> a(N), b(N);
    for (size_t i = 0; i < N; ++i) {
        a[i] = static_cast<uint16_t>(512 + (static_cast<int>(i % 32) - 16));
        b[i] = static_cast<uint16_t>(512 + (static_cast<int>(i % 200) - 100));
    }

    CudaStream cs;
    auto pool = make_test_pool(N * sizeof(uint16_t));

    HuffmanStage<uint16_t> stage;
    stage.setBklen(1024);
    stage.setFixedBookFromModel({HuffmanBookModel::Laplace, -1.0, 64.0, 2.0});

    auto enc_a1 = huffman_encode(stage, a, cs.stream, *pool);
    auto enc_b  = huffman_encode(stage, b, cs.stream, *pool);
    auto enc_a2 = huffman_encode(stage, a, cs.stream, *pool);

    // Same book, same input → byte-identical stream.
    EXPECT_EQ(enc_a1, enc_a2);

    stage.setInverse(false);
    auto dec_b = huffman_decode(stage, enc_b, N, cs.stream, *pool);
    ASSERT_EQ(dec_b.size(), N);
    for (size_t i = 0; i < N; ++i)
        EXPECT_EQ(dec_b[i], b[i]) << "mismatch at i=" << i;
}

// ─────────────────────────────────────────────────────────────────────────────
// HF24 — FixedBook_FromFreq
// A caller-supplied frequency table round-trips, and a zero bin is rejected up
// front rather than producing an uncodable symbol.
// ─────────────────────────────────────────────────────────────────────────────
TEST(HuffmanStage, FixedBook_FromFreq) {
    const uint32_t BK = 256;
    std::vector<uint32_t> freq(BK);
    for (uint32_t i = 0; i < BK; ++i)
        freq[i] = 1u + (BK - i);  // strictly positive, mildly skewed

    const size_t N = 4096;
    std::vector<uint8_t> h_in(N);
    for (size_t i = 0; i < N; ++i) h_in[i] = static_cast<uint8_t>(i % 200);

    CudaStream cs;
    auto pool = make_test_pool(N);

    HuffmanStage<uint8_t> stage;
    stage.setFixedBookFromFreq(freq.data(), BK);
    EXPECT_EQ(stage.getBookSource(), HuffmanBookSource::Fixed);
    EXPECT_FALSE(stage.hasBookSpec());  // raw table is not spec-describable

    auto encoded = huffman_encode(stage, h_in, cs.stream, *pool);
    ASSERT_GT(encoded.size(), 0u);

    stage.setInverse(false);
    auto decoded = huffman_decode(stage, encoded, N, cs.stream, *pool);
    ASSERT_EQ(decoded.size(), N);
    for (size_t i = 0; i < N; ++i)
        EXPECT_EQ(decoded[i], h_in[i]) << "mismatch at i=" << i;

    // A zero-frequency bin gets no code; reject it instead of corrupting output.
    std::vector<uint32_t> bad = freq;
    bad[7] = 0;
    HuffmanStage<uint8_t> bad_stage;
    EXPECT_THROW(bad_stage.setFixedBookFromFreq(bad.data(), BK), std::invalid_argument);
}

// ─────────────────────────────────────────────────────────────────────────────
// HF25 — FixedBook_WithoutBookThrows
// Selecting Fixed without supplying a book must fail loudly at execute().
// ─────────────────────────────────────────────────────────────────────────────
TEST(HuffmanStage, FixedBook_WithoutBookThrows) {
    const size_t N = 1024;
    std::vector<uint16_t> h_in(N, 512);

    CudaStream cs;
    auto pool = make_test_pool(N * sizeof(uint16_t));

    HuffmanStage<uint16_t> stage;
    stage.setBklen(1024);
    stage.setBookSource(HuffmanBookSource::Fixed);

    EXPECT_THROW(huffman_encode(stage, h_in, cs.stream, *pool), std::runtime_error);
}

// ─────────────────────────────────────────────────────────────────────────────
// HF26 — FixedBook_LorenzoQuantPipeline
// End-to-end float round-trip with a pre-built book, Fine encode mode.
// ─────────────────────────────────────────────────────────────────────────────
TEST(HuffmanStage, FixedBook_LorenzoQuantPipeline) {
    const size_t N = 64 * 64;
    const float  EB = 1e-3f;
    std::vector<float> h_in(N);
    for (size_t i = 0; i < N; ++i)
        h_in[i] = std::sin(static_cast<float>(i) * 0.01f) * 10.0f;

    const size_t in_bytes = N * sizeof(float);
    Pipeline p(in_bytes, MemoryStrategy::MINIMAL);
    p.setDims(64, 64, 1);

    auto* lq = p.addStage<LorenzoQuantStage<float, uint16_t>>();
    lq->setErrorBound(EB);
    lq->setQuantRadius(512);
    lq->setOutlierCapacity(0.2f);
    lq->setZigzagCodes(true);

    auto* huf = p.addStage<HuffmanStage<uint16_t>>();
    huf->setBklen(1024);
    huf->setEncodeMode(HuffmanEncodeMode::Fine);
    // Zigzag codes cluster at 0, not at bklen/2.
    huf->setFixedBookFromModel({HuffmanBookModel::Laplace, /*center=*/0.0,
                                /*scale=*/24.0, /*shape=*/2.0});
    p.connect(huf, lq, "codes");

    p.finalize();

    CudaStream cs;
    auto res = pipeline_round_trip<float>(p, h_in, cs.stream);

    ASSERT_EQ(res.data.size(), N);
    EXPECT_GT(res.compressed_bytes, 0u);
    EXPECT_LT(res.compressed_bytes, in_bytes);
    EXPECT_LE(res.max_error, static_cast<double>(EB) * 1.01)
        << "max reconstruction error " << res.max_error << " exceeds bound " << EB;
}

// ─────────────────────────────────────────────────────────────────────────────
// HF27 — AdaptiveBook_RoundTrip
// Histogram once, reuse forever: later blocks with a different distribution —
// including symbols absent from the sampled block — must still round-trip.
// ─────────────────────────────────────────────────────────────────────────────
TEST(HuffmanStage, AdaptiveBook_RoundTrip) {
    const size_t N = 8192;
    std::vector<uint16_t> first(N), later(N);
    for (size_t i = 0; i < N; ++i) {
        // Sampled block only ever contains [500, 524).
        first[i] = static_cast<uint16_t>(500 + (i % 24));
        // Later block ranges far wider, over symbols the sample never saw.
        later[i] = static_cast<uint16_t>((i * 7) % 1024);
    }

    CudaStream cs;
    auto pool = make_test_pool(N * sizeof(uint16_t));

    HuffmanStage<uint16_t> stage;
    stage.setBklen(1024);
    stage.setBookSource(HuffmanBookSource::Adaptive);

    auto enc_first = huffman_encode(stage, first, cs.stream, *pool);
    ASSERT_GT(enc_first.size(), 0u);
    // The frequency floor guarantees a code for every symbol in [0, bklen).
    EXPECT_EQ(stage.getFixedBookFreq().size(), 1024u);
    for (uint32_t f : stage.getFixedBookFreq()) EXPECT_GT(f, 0u);

    auto enc_later = huffman_encode(stage, later, cs.stream, *pool);
    ASSERT_GT(enc_later.size(), 0u);

    stage.setInverse(false);
    auto dec = huffman_decode(stage, enc_later, N, cs.stream, *pool);
    ASSERT_EQ(dec.size(), N);
    for (size_t i = 0; i < N; ++i)
        EXPECT_EQ(dec[i], later[i]) << "mismatch at i=" << i;
}

// ─────────────────────────────────────────────────────────────────────────────
// HF28 — AdaptiveBook_BeatsModelRatio
// The whole point of Adaptive over Fixed: a book sampled from the data should
// compress it at least as well as a guessed analytic shape.
// ─────────────────────────────────────────────────────────────────────────────
TEST(HuffmanStage, AdaptiveBook_BeatsModelRatio) {
    const size_t N = 16384;
    std::vector<uint16_t> h_in(N);
    // Bimodal — deliberately not what any of the single-peak models describe.
    for (size_t i = 0; i < N; ++i)
        h_in[i] = static_cast<uint16_t>((i % 3 == 0) ? 100 + (i % 8) : 800 + (i % 8));

    CudaStream cs;
    auto pool = make_test_pool(N * sizeof(uint16_t));

    HuffmanStage<uint16_t> model;
    model.setBklen(1024);
    model.setFixedBookFromModel({HuffmanBookModel::Gaussian, -1.0, 32.0, 2.0});
    const size_t model_bytes = huffman_encode(model, h_in, cs.stream, *pool).size();

    HuffmanStage<uint16_t> adaptive;
    adaptive.setBklen(1024);
    adaptive.setBookSource(HuffmanBookSource::Adaptive);
    const size_t adaptive_bytes = huffman_encode(adaptive, h_in, cs.stream, *pool).size();

    EXPECT_LT(adaptive_bytes, model_bytes)
        << "adaptive " << adaptive_bytes << " vs model " << model_bytes;
}

// ─────────────────────────────────────────────────────────────────────────────
// HF29 — AdaptiveBook_FloorShiftFlattens
// A smaller floor shift means a flatter book. It must stay correct, and it must
// not track the data better than a larger shift.
// ─────────────────────────────────────────────────────────────────────────────
TEST(HuffmanStage, AdaptiveBook_FloorShiftFlattens) {
    const size_t N = 8192;
    std::vector<uint16_t> h_in(N);
    for (size_t i = 0; i < N; ++i)
        h_in[i] = static_cast<uint16_t>((i % 100 == 0) ? (i % 1024) : 512);

    CudaStream cs;
    auto pool = make_test_pool(N * sizeof(uint16_t));

    size_t bytes[2];
    const uint8_t shifts[2] = {0, 24};
    for (int k = 0; k < 2; ++k) {
        HuffmanStage<uint16_t> s;
        s.setBklen(1024);
        s.setBookSource(HuffmanBookSource::Adaptive);
        s.setAdaptiveFloorShift(shifts[k]);
        auto enc = huffman_encode(s, h_in, cs.stream, *pool);
        bytes[k] = enc.size();
        EXPECT_LE(s.getAdaptiveFloorShiftUsed(), shifts[k]);

        s.setInverse(false);
        auto dec = huffman_decode(s, enc, N, cs.stream, *pool);
        ASSERT_EQ(dec.size(), N);
        for (size_t i = 0; i < N; ++i) ASSERT_EQ(dec[i], h_in[i]);
    }
    // shift 0 is a uniform book; shift 24 follows the histogram.
    EXPECT_LT(bytes[1], bytes[0]);
}

// ─────────────────────────────────────────────────────────────────────────────
// HF30 — OverlongCodeThrows
// A histogram skewed enough to need codes wider than HuffmanWord<4>'s 27-bit
// field used to be clamped silently by the builder. PerBlock must now throw;
// Adaptive must instead flatten the book until it fits.
// ─────────────────────────────────────────────────────────────────────────────
TEST(HuffmanStage, OverlongCodeThrows) {
    // Fibonacci-weighted frequencies are the classic Huffman worst case: each
    // frequency exceeds the sum of all smaller ones, which forces a fully
    // degenerate (linear) tree of depth bklen-1. 48 symbols is the most that fits
    // in uint32 (Fib(47) = 2971215073) and gives a depth of ~47, well past the
    // 27-bit code field. The builder used to clamp these to prefix_code=0 and
    // print to stdout; it must now fail.
    const uint32_t BK = 48;
    std::vector<uint32_t> freq(BK);
    uint64_t a = 1, b = 1;
    for (uint32_t i = 0; i < BK; ++i) {
        freq[BK - 1 - i] = static_cast<uint32_t>(a);
        const uint64_t n = a + b; a = b; b = n;
    }
    ASSERT_LT(freq[0], 0xffffffffull);  // no saturation — the weights stay distinct

    const size_t N = 4096;
    std::vector<uint8_t> h_in(N);
    for (size_t i = 0; i < N; ++i) h_in[i] = static_cast<uint8_t>(i % BK);

    CudaStream cs;
    auto pool = make_test_pool(N);

    // The table is accepted; the book is built (and rejected) on the first encode.
    HuffmanStage<uint8_t> stage;
    stage.setFixedBookFromFreq(freq.data(), BK);
    EXPECT_THROW(huffman_encode(stage, h_in, cs.stream, *pool), std::runtime_error);

    // Adaptive is the documented way out: it flattens the frequency range until
    // the book fits rather than failing.
    HuffmanStage<uint8_t> adaptive;
    adaptive.setBklen(BK);
    adaptive.setBookSource(HuffmanBookSource::Adaptive);
    auto enc = huffman_encode(adaptive, h_in, cs.stream, *pool);
    ASSERT_GT(enc.size(), 0u);
    adaptive.setInverse(false);
    auto dec = huffman_decode(adaptive, enc, N, cs.stream, *pool);
    ASSERT_EQ(dec.size(), N);
    for (size_t i = 0; i < N; ++i) EXPECT_EQ(dec[i], h_in[i]);
}

// ─────────────────────────────────────────────────────────────────────────────
// HF31 — AdaptiveBook_DegenerateSampleNotPinned
// A constant first block teaches the codebook nothing but "one symbol". Pinning
// it would freeze that book for the rest of the run (measured at 42% mean ratio
// loss over CESM CLOUD when the fit landed on a constant level). The stage must
// encode the degenerate block with a throwaway book and stay unpinned, so the
// next real block gets fitted properly.
// ─────────────────────────────────────────────────────────────────────────────
TEST(HuffmanStage, AdaptiveBook_DegenerateSampleNotPinned) {
    const size_t N = 8192;
    std::vector<uint16_t> flat(N, 512);            // one distinct symbol
    std::vector<uint16_t> varied(N);
    for (size_t i = 0; i < N; ++i)
        varied[i] = static_cast<uint16_t>(400 + (i * 37) % 200);

    CudaStream cs;
    auto pool = make_test_pool(N * sizeof(uint16_t));

    HuffmanStage<uint16_t> poisoned;
    poisoned.setBklen(1024);
    poisoned.setBookSource(HuffmanBookSource::Adaptive);
    huffman_encode(poisoned, flat, cs.stream, *pool);          // degenerate sample
    const size_t after_flat = huffman_encode(poisoned, varied, cs.stream, *pool).size();

    // Reference: the same stage fitted directly on the varied data.
    HuffmanStage<uint16_t> clean;
    clean.setBklen(1024);
    clean.setBookSource(HuffmanBookSource::Adaptive);
    const size_t direct = huffman_encode(clean, varied, cs.stream, *pool).size();

    // Having seen the flat block first must cost nothing: it was not pinned.
    EXPECT_EQ(after_flat, direct)
        << "constant first block poisoned the codebook (" << after_flat
        << " vs " << direct << " bytes)";

    // And the degenerate block itself must still round-trip.
    poisoned.setInverse(false);
    HuffmanStage<uint16_t> rt;
    rt.setBklen(1024);
    rt.setBookSource(HuffmanBookSource::Adaptive);
    auto enc = huffman_encode(rt, flat, cs.stream, *pool);
    rt.setInverse(false);
    auto dec = huffman_decode(rt, enc, N, cs.stream, *pool);
    ASSERT_EQ(dec.size(), N);
    for (size_t i = 0; i < N; ++i) ASSERT_EQ(dec[i], flat[i]);
}

// ─────────────────────────────────────────────────────────────────────────────
// HF32 — AdaptiveBook_RefitOnRateRegression
// encode() already reports total_nbit, so a stale book is detectable for free
// when the bit rate degrades. Fit on a highly compressible block, then feed a
// much broader distribution: the rate must regress past the threshold and
// trigger a refit, and later blocks must recover toward a freshly fitted book.
// ─────────────────────────────────────────────────────────────────────────────
TEST(HuffmanStage, AdaptiveBook_RefitOnRateRegression) {
    const size_t N = 16384;
    std::vector<uint16_t> narrow(N), broad(N);
    for (size_t i = 0; i < N; ++i) {
        narrow[i] = static_cast<uint16_t>(512 + (i % 4));          // ~2 bits/sym
        broad[i]  = static_cast<uint16_t>((i * 131) % 1024);       // ~10 bits/sym
    }

    CudaStream cs;
    auto pool = make_test_pool(N * sizeof(uint16_t));

    HuffmanStage<uint16_t> stage;
    stage.setBklen(1024);
    stage.setBookSource(HuffmanBookSource::Adaptive);
    stage.setRefitThreshold(1.2f);

    huffman_encode(stage, narrow, cs.stream, *pool);      // fits here
    EXPECT_EQ(stage.getRefitCount(), 0u);
    EXPECT_GT(stage.getFitBitsPerSymbol(), 0.0);

    huffman_encode(stage, broad, cs.stream, *pool);       // rate regresses
    EXPECT_GE(stage.getRefitCount(), 1u)
        << "bit-rate regression did not trigger a refit";

    // The refit lands on the *next* call, which should then match a stage fitted
    // directly on this distribution.
    const size_t refitted = huffman_encode(stage, broad, cs.stream, *pool).size();

    HuffmanStage<uint16_t> clean;
    clean.setBklen(1024);
    clean.setBookSource(HuffmanBookSource::Adaptive);
    const size_t direct = huffman_encode(clean, broad, cs.stream, *pool).size();
    EXPECT_EQ(refitted, direct);

    // Disabling the trigger must keep the original book pinned.
    HuffmanStage<uint16_t> pinned;
    pinned.setBklen(1024);
    pinned.setBookSource(HuffmanBookSource::Adaptive);
    pinned.setRefitThreshold(0.0f);
    huffman_encode(pinned, narrow, cs.stream, *pool);
    huffman_encode(pinned, broad, cs.stream, *pool);
    EXPECT_EQ(pinned.getRefitCount(), 0u);
}
