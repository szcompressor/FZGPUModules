/**
 * tests/stages/test_huffman.cpp
 *
 * GPU unit tests for HuffmanStage<T>.
 * The default host-coordinated path is not graph-compatible.  The fixed-book
 * DeviceResident forward path performs its scan and header assembly on-device
 * and is graph-compatible when Huffman is the terminal stage.
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
 *   HF22  HuffmanStage/FixedBook_RoundTrip_U16         — model-derived book: uint16_t round-trip exact match
 *   HF23  HuffmanStage/FixedBook_ReusedAcrossCalls     — one book across differing inputs; identical input → identical stream
 *   HF24  HuffmanStage/FixedBook_FromFreq              — caller-supplied freq table round-trips; zero bin throws
 *   HF25  HuffmanStage/FixedBook_WithoutBookThrows     — Fixed source with no book throws at execute
 *   HF26  HuffmanStage/FixedBook_LorenzoQuantPipeline  — pre-built book, end-to-end float round-trip
 *   HF27  HuffmanStage/AdaptiveBook_RoundTrip          — sample once, reuse; symbols absent from the sample still round-trip
 *   HF28  HuffmanStage/AdaptiveBook_BeatsModelRatio    — sampled book beats a guessed model on bimodal data
 *   HF29  HuffmanStage/AdaptiveBook_FloorShiftFlattens — floor shift trades ratio for flatness; both correct
 *   HF30  HuffmanStage/OverlongCodeThrows              — >27-bit codes throw instead of being silently clamped
 *   HF31  HuffmanStage/AdaptiveBook_DegenerateSampleNotPinned — a constant first block must not pin the book
 *   HF32  HuffmanStage/AdaptiveBook_RefitOnRateRegression     — bit-rate regression triggers a refit
 *   HF33  HuffmanStage/PinnedBookSymbolRangeGuard            — pinned books still reject symbols >= bklen
 *   HF34  HuffmanStage/DeviceResidentFixedBookRoundTrip       — device-resident uint16 round-trip
 *   HF35  HuffmanStage/DeviceResidentMatchesHostCoordinatedBytes — identical PHF stream
 *   HF36  HuffmanStage/DeviceResidentBookSourcesAndRange       — all sources + validation
 *   HF37  HuffmanStage/Uint8OutputEstimateCoversCodewordLimit — conservative archive bound
 *   HF38  HuffmanStage/DeviceResidentRoundTrip_U8             — device-resident uint8 round-trip
 *   HF39  HuffmanStage/DeviceResidentRoundTrip_U32            — device-resident uint32 round-trip
 *   HF40  HuffmanStage/DeviceResidentNonTerminalPipeline       — exact-size downstream composition
 *   HF41  HuffmanStage/DeviceResidentMatchesHostBooksForAllSources — GPU tree compatibility
 *   HF42  HuffmanStage/DeviceResidentLargeBookGlobalScratchFallback — >1024-symbol fallback
 *   HF43  HuffmanStage/DeviceResidentDecodeUsesEmbeddedPartitionGeometry — capped-grid compatibility
 */

#include <gtest/gtest.h>
#include "helpers/fz_test_utils.h"
#include "helpers/stage_harness.h"
#include "coders/huffman/huffman_stage.h"
#include "fzgpumodules.h"

#include "fused/lorenzo_quant/lorenzo_quant.h"

#include <cmath>
#include <cstdlib>
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
    stage.postStreamSync(stream);

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

    stage.setExecutionMode(HuffmanExecutionMode::DeviceResident);
    EXPECT_FALSE(stage.isGraphCompatible()) << "device mode still needs a fixed book";

    stage.setFixedBookFromModel(
        {HuffmanBookModel::Uniform, -1.0, 1.0, 2.0});
    EXPECT_TRUE(stage.isGraphCompatible());

    stage.setInverse(true);
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
// End-to-end float round-trip with a pre-built book.
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

    HuffmanStage<uint8_t> device_stage;
    device_stage.setFixedBookFromFreq(freq.data(), BK);
    device_stage.setExecutionMode(HuffmanExecutionMode::DeviceResident);
    EXPECT_THROW(
        huffman_encode(device_stage, h_in, cs.stream, *pool), std::runtime_error);

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
// HF36 — PerBlockFallsBackToAdaptiveOnOverlongCode
// The *histogram-driven* counterpart to HF30. A user-supplied frequency table
// that cannot fit the 27-bit code field is a configuration error and still
// throws; real input data that happens to histogram that way is not, and
// throwing there cost the cuSZ preset every wide-dynamic-range field in the
// corpus (HACC/vy, HACC/xx, EXAFEL/data all hard-failed). PerBlock must fall
// back to a floored Adaptive book, stay lossless, and say so.
// ─────────────────────────────────────────────────────────────────────────────
TEST(HuffmanStage, PerBlockFallsBackToAdaptiveOnOverlongCode) {
    // Fibonacci counts force a degenerate (linear) Huffman tree, so depth tracks
    // the symbol count. 30 symbols gives depth ~29, past the 27-bit field, and
    // sums to Fib(32)-1 ~ 2.18M elements — big enough to be a real histogram,
    // small enough for a unit test. (HF30 needs 48 symbols only because it hands
    // the frequencies over directly and never materializes the data.)
    // 30, not 32, and deliberately: with uint8_t symbols the reverse codebook is
    // 256 + bklen bytes, so bklen 30 puts the bitstream at offset 2 mod 4. The
    // decode kernel reads it as uint32* and silently returned wrong symbols, so
    // this doubles as the regression test for setBklen's alignment rounding.
    const uint32_t BK = 30;
    std::vector<uint32_t> freq(BK);
    uint64_t a = 1, b = 1;
    for (uint32_t i = 0; i < BK; ++i) {
        freq[BK - 1 - i] = static_cast<uint32_t>(a);
        const uint64_t n = a + b; a = b; b = n;
    }

    std::vector<uint8_t> h_in;
    for (uint32_t sym = 0; sym < BK; ++sym)
        h_in.insert(h_in.end(), freq[sym], static_cast<uint8_t>(sym));
    const size_t N = h_in.size();
    ASSERT_GT(N, 1u << 20);

    CudaStream cs;
    auto pool = make_test_pool(N * 4);

    // Control: does an EXPLICIT Adaptive book round-trip this same data? If it
    // does not, the defect is in Adaptive, not in the fallback that selects it.
    {
        HuffmanStage<uint8_t> control;
        control.setBklen(BK);
        control.setBookSource(HuffmanBookSource::Adaptive);
        auto cenc = huffman_encode(control, h_in, cs.stream, *pool);
        auto cdec = huffman_decode(control, cenc, N, cs.stream, *pool);
        ASSERT_EQ(cdec.size(), N);
        for (size_t i = 0; i < N; ++i)
            ASSERT_EQ(cdec[i], h_in[i]) << "explicit Adaptive mismatch at " << i;
    }

    HuffmanStage<uint8_t> stage;          // PerBlock — the default
    stage.setBklen(BK);
    ASSERT_EQ(stage.getBookSource(), HuffmanBookSource::PerBlock);

    std::vector<uint8_t> enc;
    ASSERT_NO_THROW(enc = huffman_encode(stage, h_in, cs.stream, *pool))
        << "PerBlock must fall back, not throw, on a histogram it cannot code";
    ASSERT_GT(enc.size(), 0u);

    EXPECT_TRUE(stage.getAdaptiveFallbackUsed())
        << "the fallback must be observable — a silent one is the bug, not the fix";
    // The configured value is reported as configured; only behaviour changed.
    EXPECT_EQ(stage.getBookSource(), HuffmanBookSource::PerBlock);

    // Falling back is only acceptable if the stream is still exactly decodable —
    // this is entropy coding, the round-trip is lossless or it is broken.
    auto dec = huffman_decode(stage, enc, N, cs.stream, *pool);
    ASSERT_EQ(dec.size(), N);
    for (size_t i = 0; i < N; ++i)
        ASSERT_EQ(dec[i], h_in[i]) << "mismatch at " << i;

    HuffmanStage<uint8_t> device_stage;
    device_stage.setBklen(BK);
    device_stage.setExecutionMode(HuffmanExecutionMode::DeviceResident);
    auto device_enc = huffman_encode(device_stage, h_in, cs.stream, *pool);
    EXPECT_TRUE(device_stage.getAdaptiveFallbackUsed());
    device_stage.setInverse(false);
    EXPECT_EQ(
        huffman_decode(device_stage, device_enc, N, cs.stream, *pool), h_in);
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

// ─────────────────────────────────────────────────────────────────────────────
// HF33 — PinnedBookSymbolRangeGuard
// PerBlock catches symbols >= bklen for free via sum(h_freq) != inlen (HF12).
// A pinned codebook skips the histogram, so that guard stops running and the
// encode kernel indexes past d_bk4 — an undecodable stream, silently. The
// device-side check must restore parity for both Adaptive and Fixed, and must
// be defeatable for callers who guarantee the range upstream.
// ─────────────────────────────────────────────────────────────────────────────
TEST(HuffmanStage, PinnedBookSymbolRangeGuard) {
    const size_t N = 8192;
    std::vector<uint16_t> ok(N), bad(N);
    for (size_t i = 0; i < N; ++i) {
        ok[i]  = static_cast<uint16_t>(400 + (i * 37) % 200);
        bad[i] = ok[i];
    }
    bad[N / 2] = 5000;   // >= bklen (1024)

    CudaStream cs;
    auto pool = make_test_pool(N * sizeof(uint16_t));

    // Adaptive: first call fits and pins, second call carries the bad symbol.
    HuffmanStage<uint16_t> adaptive;
    adaptive.setBklen(1024);
    adaptive.setBookSource(HuffmanBookSource::Adaptive);
    EXPECT_TRUE(adaptive.getValidateSymbolRange());
    huffman_encode(adaptive, ok, cs.stream, *pool);          // pins the book
    EXPECT_THROW(huffman_encode(adaptive, bad, cs.stream, *pool), std::runtime_error);

    // Fixed: never histograms at all, so the very first call must be checked.
    HuffmanStage<uint16_t> fixed;
    fixed.setBklen(1024);
    fixed.setFixedBookFromModel({HuffmanBookModel::Laplace, -1.0, 64.0, 2.0});
    EXPECT_THROW(huffman_encode(fixed, bad, cs.stream, *pool), std::runtime_error);

    // In-range data is unaffected, and still round-trips.
    HuffmanStage<uint16_t> good;
    good.setBklen(1024);
    good.setBookSource(HuffmanBookSource::Adaptive);
    huffman_encode(good, ok, cs.stream, *pool);
    auto enc = huffman_encode(good, ok, cs.stream, *pool);
    good.setInverse(false);
    auto dec = huffman_decode(good, enc, N, cs.stream, *pool);
    ASSERT_EQ(dec.size(), N);
    for (size_t i = 0; i < N; ++i) ASSERT_EQ(dec[i], ok[i]);

    // Opting out must not disturb the normal path.  Deliberately exercised with
    // in-range data only: feeding an out-of-range symbol with the check disabled
    // is undefined behaviour by construction — it reads past the codebook, and in
    // practice takes down the CUDA context, which is precisely why the check
    // defaults to on.
    HuffmanStage<uint16_t> unchecked;
    unchecked.setBklen(1024);
    unchecked.setBookSource(HuffmanBookSource::Adaptive);
    unchecked.setValidateSymbolRange(false);
    EXPECT_FALSE(unchecked.getValidateSymbolRange());
    huffman_encode(unchecked, ok, cs.stream, *pool);
    auto enc2 = huffman_encode(unchecked, ok, cs.stream, *pool);
    unchecked.setInverse(false);
    auto dec2 = huffman_decode(unchecked, enc2, N, cs.stream, *pool);
    ASSERT_EQ(dec2.size(), N);
    for (size_t i = 0; i < N; ++i) ASSERT_EQ(dec2[i], ok[i]);
}

// ─────────────────────────────────────────────────────────────────────────────
// HF34-HF40 — Device-resident assembly path
// ─────────────────────────────────────────────────────────────────────────────

TEST(HuffmanStage, DeviceResidentFixedBookRoundTrip) {
    constexpr size_t N = 32771;  // partial final coarse partition
    std::vector<uint16_t> input(N);
    for (size_t i = 0; i < N; ++i)
        input[i] = static_cast<uint16_t>((i * 29 + i / 17) % 1024);

    CudaStream cs;
    auto pool = make_test_pool(N * sizeof(uint16_t), 8.0f);

    HuffmanStage<uint16_t> stage;
    stage.setBklen(1024);
    stage.setFixedBookFromModel(
        {HuffmanBookModel::Uniform, -1.0, 1.0, 2.0});
    stage.setExecutionMode(HuffmanExecutionMode::DeviceResident);

    auto encoded = huffman_encode(stage, input, cs.stream, *pool);
    ASSERT_GT(encoded.size(), PHFHEADER_FORCED_ALIGN);
    stage.setInverse(false);
    auto decoded = huffman_decode(stage, encoded, N, cs.stream, *pool);
    EXPECT_EQ(decoded, input);
}

TEST(HuffmanStage, DeviceResidentMatchesHostCoordinatedBytes) {
    constexpr size_t N = 20003;
    std::vector<uint16_t> input(N);
    for (size_t i = 0; i < N; ++i)
        input[i] = static_cast<uint16_t>((i * 13) % 700);

    CudaStream cs;
    auto host_pool = make_test_pool(N * sizeof(uint16_t), 8.0f);
    auto device_pool = make_test_pool(N * sizeof(uint16_t), 8.0f);

    HuffmanBookSpec spec{HuffmanBookModel::Laplace, -1.0, 96.0, 2.0};
    HuffmanStage<uint16_t> host;
    host.setBklen(1024);
    host.setFixedBookFromModel(spec);

    HuffmanStage<uint16_t> device;
    device.setBklen(1024);
    device.setFixedBookFromModel(spec);
    device.setExecutionMode(HuffmanExecutionMode::DeviceResident);

    const auto host_bytes = huffman_encode(host, input, cs.stream, *host_pool);
    const auto device_bytes = huffman_encode(device, input, cs.stream, *device_pool);
    EXPECT_EQ(device_bytes, host_bytes);
}

TEST(HuffmanStage, DeviceResidentBookSourcesAndRange) {
    constexpr size_t N = 8193;
    std::vector<uint16_t> input(N);
    for (size_t i = 0; i < N; ++i)
        input[i] = static_cast<uint16_t>((i * 19 + i / 13) % 700);
    CudaStream cs;
    auto pool = make_test_pool(N * sizeof(uint16_t), 8.0f);

    HuffmanStage<uint16_t> per_block;
    per_block.setBklen(1024);
    per_block.setExecutionMode(HuffmanExecutionMode::DeviceResident);
    auto per_block_bytes = huffman_encode(per_block, input, cs.stream, *pool);
    per_block.setInverse(false);
    EXPECT_EQ(huffman_decode(per_block, per_block_bytes, N, cs.stream, *pool), input);

    HuffmanStage<uint16_t> adaptive;
    adaptive.setBklen(1024);
    adaptive.setBookSource(HuffmanBookSource::Adaptive);
    adaptive.setRefitInterval(1);
    adaptive.setExecutionMode(HuffmanExecutionMode::DeviceResident);
    auto adaptive_bytes = huffman_encode(adaptive, input, cs.stream, *pool);
    adaptive.setInverse(false);
    EXPECT_EQ(huffman_decode(adaptive, adaptive_bytes, N, cs.stream, *pool), input);

    // A second encode reuses the sampled Adaptive book while keeping the same
    // fully device-resident assembly path.
    adaptive.setInverse(false);
    std::reverse(input.begin(), input.end());
    adaptive_bytes = huffman_encode(adaptive, input, cs.stream, *pool);
    adaptive.setInverse(false);
    EXPECT_EQ(huffman_decode(adaptive, adaptive_bytes, N, cs.stream, *pool), input);
    EXPECT_EQ(adaptive.getRefitCount(), 1u)
        << "deferred device metrics must preserve Adaptive refit policy";

    HuffmanStage<uint16_t> fixed;
    fixed.setBklen(1024);
    fixed.setFixedBookFromModel(
        {HuffmanBookModel::Uniform, -1.0, 1.0, 2.0});
    fixed.setExecutionMode(HuffmanExecutionMode::DeviceResident);
    input[N / 2] = 5000;
    EXPECT_THROW(
        huffman_encode(fixed, input, cs.stream, *pool), std::runtime_error);
}

TEST(HuffmanStage, Uint8OutputEstimateCoversCodewordLimit) {
    HuffmanStage<uint8_t> stage;
    constexpr size_t N = 1 << 20;
    const size_t bound = stage.estimateOutputSizes({N})[0];
    EXPECT_GT(bound, 3 * N);
}

TEST(HuffmanStage, DeviceResidentRoundTrip_U8) {
    constexpr size_t N = 8195;
    std::vector<uint8_t> input(N);
    for (size_t i = 0; i < N; ++i)
        input[i] = static_cast<uint8_t>((i * 17 + i / 11) % 256);

    CudaStream cs;
    auto pool = make_test_pool(N, 8.0f);
    HuffmanStage<uint8_t> stage;
    stage.setBklen(256);
    stage.setFixedBookFromModel(
        {HuffmanBookModel::Uniform, -1.0, 1.0, 2.0});
    stage.setExecutionMode(HuffmanExecutionMode::DeviceResident);

    const auto encoded = huffman_encode(stage, input, cs.stream, *pool);
    stage.setInverse(false);
    EXPECT_EQ(huffman_decode(stage, encoded, N, cs.stream, *pool), input);
}

TEST(HuffmanStage, DeviceResidentRoundTrip_U32) {
    constexpr size_t N = 12291;
    std::vector<uint32_t> input(N);
    for (size_t i = 0; i < N; ++i)
        input[i] = static_cast<uint32_t>((i * 31 + i / 7) % 1024);

    CudaStream cs;
    auto pool = make_test_pool(N * sizeof(uint32_t), 8.0f);
    HuffmanStage<uint32_t> stage;
    stage.setBklen(1024);
    stage.setFixedBookFromModel(
        {HuffmanBookModel::Uniform, -1.0, 1.0, 2.0});
    stage.setExecutionMode(HuffmanExecutionMode::DeviceResident);

    const auto encoded = huffman_encode(stage, input, cs.stream, *pool);
    stage.setInverse(false);
    EXPECT_EQ(huffman_decode(stage, encoded, N, cs.stream, *pool), input);
}

TEST(HuffmanStage, DeviceResidentNonTerminalPipeline) {
    constexpr size_t N = 4096;
    std::vector<uint16_t> input(N);
    for (size_t i = 0; i < N; ++i)
        input[i] = static_cast<uint16_t>((i * 23 + i / 5) % 1024);

    Pipeline p(N * sizeof(uint16_t), MemoryStrategy::PREALLOCATE, 8.0f);
    auto* huf = p.addStage<HuffmanStage<uint16_t>>();
    huf->setBklen(1024);
    huf->setFixedBookFromModel(
        {HuffmanBookModel::Uniform, -1.0, 1.0, 2.0});
    huf->setExecutionMode(HuffmanExecutionMode::DeviceResident);
    auto* rle = p.addStage<RLEStage<uint8_t>>();
    p.connect(rle, huf);
    ASSERT_NO_THROW(p.finalize());
    EXPECT_FALSE(huf->isGraphCompatible());

    CudaStream cs;
    const auto result = pipeline_round_trip<uint16_t>(p, input, cs.stream);
    EXPECT_EQ(result.data, input);
}

TEST(HuffmanStage, DeviceResidentMatchesHostBooksForAllSources) {
    constexpr size_t N = 24017;
    std::vector<uint16_t> input(N);
    for (size_t i = 0; i < N; ++i) {
        if (i % 11 == 0) input[i] = static_cast<uint16_t>((i * 37) % 900);
        else if (i % 3 == 0) input[i] = 7;
        else input[i] = static_cast<uint16_t>((i * 5 + i / 9) % 83);
    }

    CudaStream cs;
    auto host_pool = make_test_pool(N * sizeof(uint16_t), 10.0f);
    auto device_pool = make_test_pool(N * sizeof(uint16_t), 10.0f);

    for (const auto source : {HuffmanBookSource::PerBlock,
                              HuffmanBookSource::Adaptive}) {
        HuffmanStage<uint16_t> host;
        host.setBklen(1024);
        host.setBookSource(source);

        HuffmanStage<uint16_t> device;
        device.setBklen(1024);
        device.setBookSource(source);
        device.setExecutionMode(HuffmanExecutionMode::DeviceResident);

        EXPECT_EQ(
            huffman_encode(device, input, cs.stream, *device_pool),
            huffman_encode(host, input, cs.stream, *host_pool))
            << "device-built canonical book differs for source "
            << static_cast<int>(source);
    }
}

TEST(HuffmanStage, DeviceResidentLargeBookGlobalScratchFallback) {
    constexpr size_t N = 16387;
    std::vector<uint16_t> input(N);
    for (size_t i = 0; i < N; ++i)
        input[i] = static_cast<uint16_t>((i * 29 + i / 7) % 1800);

    CudaStream cs;
    auto host_pool = make_test_pool(N * sizeof(uint16_t), 12.0f);
    auto device_pool = make_test_pool(N * sizeof(uint16_t), 12.0f);

    HuffmanStage<uint16_t> host;
    host.setBklen(2048);

    HuffmanStage<uint16_t> device;
    device.setBklen(2048);
    device.setExecutionMode(HuffmanExecutionMode::DeviceResident);

    const auto host_bytes = huffman_encode(host, input, cs.stream, *host_pool);
    const auto device_bytes = huffman_encode(device, input, cs.stream, *device_pool);
    EXPECT_EQ(device_bytes, host_bytes);

    device.setInverse(false);
    EXPECT_EQ(huffman_decode(device, device_bytes, N, cs.stream, *device_pool), input);
}

TEST(HuffmanStage, DeviceResidentDecodeUsesEmbeddedPartitionGeometry) {
    constexpr size_t N = 200003;
    std::vector<uint16_t> input(N);
    for (size_t i = 0; i < N; ++i)
        input[i] = static_cast<uint16_t>((i * 17 + i / 11) % 700);

    CudaStream cs;
    auto encode_pool = make_test_pool(N * sizeof(uint16_t), 10.0f);
    auto decode_pool = make_test_pool(N * sizeof(uint16_t), 10.0f);

    struct SublenEnvReset {
        bool armed = true;
        ~SublenEnvReset() { if (armed) unsetenv("FZ_HF_SUBLEN"); }
    } env_reset;
    setenv("FZ_HF_SUBLEN", "64", 1);

    HuffmanStage<uint16_t> encoder;
    encoder.setBklen(1024);
    const auto encoded = huffman_encode(encoder, input, cs.stream, *encode_pool);

    uint8_t stage_header[128]{};
    const size_t stage_header_bytes =
        encoder.serializeHeader(0, stage_header, sizeof(stage_header));
    unsetenv("FZ_HF_SUBLEN");
    env_reset.armed = false;

    HuffmanStage<uint16_t> decoder;
    decoder.deserializeHeader(stage_header, stage_header_bytes);
    decoder.setExecutionMode(HuffmanExecutionMode::DeviceResident);
    EXPECT_EQ(huffman_decode(decoder, encoded, N, cs.stream, *decode_pool), input);
}
