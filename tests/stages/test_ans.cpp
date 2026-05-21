/**
 * tests/stages/test_ans.cpp
 *
 * GPU unit tests for ANSStage (rANS byte-level entropy coder, dietGPU backend).
 * Forward input: uint8_t[] → ANS compressed bitstream (ANSCoalescedHeader + data).
 * Inverse input: ANS bitstream → uint8_t[].
 * Not graph-compatible (D2H sync in both directions).
 *
 *   AN1   ANSStage/RoundTrip                  — uint8_t forward+inverse exact match (4 blocks)
 *   AN2   ANSStage/ZeroInput                  — n=0 does not crash; output size is 0
 *   AN3   ANSStage/CompressedSmallerThanInput  — compressed size < input for skewed distribution
 *   AN4   ANSStage/SerializeDeserialize        — prob_bits round-trips through 12-byte header
 *   AN5   ANSStage/SaveRestoreState            — saveState+deserializeHeader+restoreState
 *   AN6   ANSStage/GraphCompatible             — isGraphCompatible()==false
 *   AN7   ANSStage/PipelineIntegration         — Pipeline round-trip with uint8_t
 *   AN8   ANSStage/ReuseAfterSizeChange        — shrink reuses cap_bytes_; both round-trips correct
 *   AN9   ANSStage/FileRoundTrip               — writeToFile → decompressFromFile via Pipeline
 *   AN10  ANSStage/LorenzoQuantPipeline        — LorenzoQuant→ANS end-to-end float round-trip
 *   AN11  ANSStage/PartialBlock                — input < kDefaultBlockSize (1000 bytes) round-trip
 *   AN12  ANSStage/UnsupportedProbBitsThrows   — setProbBits(11) then execute() throws
 */

#include <gtest/gtest.h>
#include "helpers/fz_test_utils.h"
#include "helpers/stage_harness.h"
#include "coders/ans/ans_stage.h"
#include "fzgpumodules.h"
#include "fused/lorenzo_quant/lorenzo_quant.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

using namespace fz;
using namespace fz_test;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers: single-pass encode / decode via stage API
// ─────────────────────────────────────────────────────────────────────────────

static std::vector<uint8_t> ans_encode(
    ANSStage& stage,
    const std::vector<uint8_t>& h_in,
    cudaStream_t stream,
    fz::MemoryPool& pool)
{
    const size_t in_bytes = h_in.size();
    const size_t out_est  = stage.estimateOutputSizes({in_bytes})[0];

    CudaBuffer<uint8_t> d_in(std::max(h_in.size(), size_t(1)));
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

static std::vector<uint8_t> ans_decode(
    ANSStage& stage,
    const std::vector<uint8_t>& h_encoded,
    size_t out_bytes,
    cudaStream_t stream,
    fz::MemoryPool& pool)
{
    const size_t in_bytes = h_encoded.size();

    CudaBuffer<uint8_t> d_in(std::max(in_bytes, size_t(1)));
    CudaBuffer<uint8_t> d_out(std::max(out_bytes, size_t(1)));
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

    std::vector<uint8_t> h_out(out_bytes);
    if (out_bytes > 0)
        cudaMemcpy(h_out.data(), d_out.get(), out_bytes, cudaMemcpyDeviceToHost);
    return h_out;
}

// ─────────────────────────────────────────────────────────────────────────────
// AN1 — RoundTrip
// ─────────────────────────────────────────────────────────────────────────────
TEST(ANSStage, RoundTrip) {
    const size_t N = 16384;  // 4 dietGPU blocks (4 × 4096 bytes)
    std::vector<uint8_t> h_in(N);
    for (size_t i = 0; i < N; ++i) h_in[i] = static_cast<uint8_t>(i % 256);

    CudaStream cs;
    auto pool = make_test_pool(N);

    ANSStage stage;
    auto encoded = ans_encode(stage, h_in, cs.stream, *pool);
    ASSERT_GT(encoded.size(), 0u);

    stage.setInverse(false);
    auto decoded = ans_decode(stage, encoded, N, cs.stream, *pool);
    ASSERT_EQ(decoded.size(), N);
    for (size_t i = 0; i < N; ++i)
        EXPECT_EQ(decoded[i], h_in[i]) << "mismatch at i=" << i;
}

// ─────────────────────────────────────────────────────────────────────────────
// AN2 — ZeroInput
// ─────────────────────────────────────────────────────────────────────────────
TEST(ANSStage, ZeroInput) {
    ANSStage stage;
    CudaStream cs;
    auto pool = make_test_pool(64);

    CudaBuffer<uint8_t> d_dummy(1);
    std::vector<void*>  inputs  = {d_dummy.void_ptr()};
    std::vector<void*>  outputs = {d_dummy.void_ptr()};
    std::vector<size_t> sizes   = {0};
    EXPECT_NO_THROW(stage.execute(cs.stream, pool.get(), inputs, outputs, sizes));
    EXPECT_EQ(stage.getActualOutputSize(0), 0u);
}

// ─────────────────────────────────────────────────────────────────────────────
// AN3 — CompressedSmallerThanInput
// ─────────────────────────────────────────────────────────────────────────────
TEST(ANSStage, CompressedSmallerThanInput) {
    const size_t N = 16384;
    // 90% zeros, 10% 0x01 — highly skewed distribution, compresses well under rANS
    std::vector<uint8_t> h_in(N, 0);
    for (size_t i = 0; i < N / 10; ++i) h_in[i * 10] = 1;

    CudaStream cs;
    auto pool = make_test_pool(N);

    ANSStage stage;
    auto encoded = ans_encode(stage, h_in, cs.stream, *pool);
    EXPECT_LT(encoded.size(), N)
        << "highly skewed input should compress below raw size";
}

// ─────────────────────────────────────────────────────────────────────────────
// AN4 — SerializeDeserialize
// 12-byte header layout: [0] prob_bits, [1..3] reserved=0, [4..11] original_bytes_ (LE uint64_t).
// ─────────────────────────────────────────────────────────────────────────────
TEST(ANSStage, SerializeDeserialize) {
    ANSStage original;
    // prob_bits defaults to 10

    uint8_t buf[128] = {};
    size_t written = original.serializeHeader(0, buf, sizeof(buf));
    EXPECT_EQ(written, 12u);
    EXPECT_EQ(buf[0], 10u);  // prob_bits
    EXPECT_EQ(buf[1], 0u);   // reserved
    EXPECT_EQ(buf[2], 0u);
    EXPECT_EQ(buf[3], 0u);

    ANSStage restored;
    restored.deserializeHeader(buf, written);
    EXPECT_EQ(restored.getProbBits(), 10u);
}

// ─────────────────────────────────────────────────────────────────────────────
// AN5 — SaveRestoreState
// Simulates the pipeline calling deserializeHeader() before decompress and
// restoreState() after to leave forward-pass configuration intact.
// ─────────────────────────────────────────────────────────────────────────────
TEST(ANSStage, SaveRestoreState) {
    ANSStage s;
    // prob_bits defaults to 10
    s.saveState();

    // Simulate a deserializeHeader() call from a file with different settings.
    uint8_t fake_hdr[12] = {};
    fake_hdr[0] = 11;  // different prob_bits (not executed — just testing state management)
    uint64_t fake_orig = 999999;
    std::memcpy(fake_hdr + 4, &fake_orig, sizeof(uint64_t));
    s.deserializeHeader(fake_hdr, sizeof(fake_hdr));
    EXPECT_EQ(s.getProbBits(), 11u);

    s.restoreState();
    EXPECT_EQ(s.getProbBits(), 10u);
}

// ─────────────────────────────────────────────────────────────────────────────
// AN6 — GraphCompatible
// ─────────────────────────────────────────────────────────────────────────────
TEST(ANSStage, GraphCompatible) {
    ANSStage stage;
    EXPECT_FALSE(stage.isGraphCompatible());
}

// ─────────────────────────────────────────────────────────────────────────────
// AN7 — PipelineIntegration
// Full Pipeline round-trip using pipeline_round_trip<uint8_t>.
// ─────────────────────────────────────────────────────────────────────────────
TEST(ANSStage, PipelineIntegration) {
    const size_t N        = 4096;
    const size_t in_bytes = N;  // uint8_t, so bytes == count

    std::vector<uint8_t> h_in(N);
    for (size_t i = 0; i < N; ++i) h_in[i] = static_cast<uint8_t>(i % 128);

    Pipeline p(in_bytes, MemoryStrategy::PREALLOCATE);
    p.addStage<ANSStage>();
    p.finalize();

    CudaStream cs;
    auto res = pipeline_round_trip<uint8_t>(p, h_in, cs.stream);

    ASSERT_EQ(res.data.size(), N);
    for (size_t i = 0; i < N; ++i)
        EXPECT_EQ(res.data[i], h_in[i]) << "mismatch at i=" << i;
    EXPECT_GT(res.compressed_bytes, 0u);
}

// ─────────────────────────────────────────────────────────────────────────────
// AN8 — ReuseAfterSizeChange
// Pass 1 (N1=16384): initial scratch allocation; cap_bytes_=16384.
// Pass 2 (N2=4096): inlen < cap_bytes_ — existing device buffers are reused (no realloc).
// Both round-trips must produce exact output.
// ─────────────────────────────────────────────────────────────────────────────
TEST(ANSStage, ReuseAfterSizeChange) {
    CudaStream cs;

    ANSStage stage;

    // ── First call: large input ───────────────────────────────────────────────
    const size_t N1 = 16384;
    std::vector<uint8_t> h_in1(N1);
    for (size_t i = 0; i < N1; ++i) h_in1[i] = static_cast<uint8_t>(i % 200);

    auto pool1 = make_test_pool(N1);
    auto encoded1 = ans_encode(stage, h_in1, cs.stream, *pool1);
    ASSERT_GT(encoded1.size(), 0u);

    stage.setInverse(false);
    auto decoded1 = ans_decode(stage, encoded1, N1, cs.stream, *pool1);
    ASSERT_EQ(decoded1.size(), N1);
    for (size_t i = 0; i < N1; ++i)
        EXPECT_EQ(decoded1[i], h_in1[i]) << "pass1 mismatch at i=" << i;

    // ── Second call: smaller input — reuses existing buffers (cap_bytes_=16384) ─
    const size_t N2 = 4096;
    std::vector<uint8_t> h_in2(N2);
    for (size_t i = 0; i < N2; ++i) h_in2[i] = static_cast<uint8_t>((i * 7) % 100);

    auto pool2 = make_test_pool(N2);
    stage.setInverse(false);
    auto encoded2 = ans_encode(stage, h_in2, cs.stream, *pool2);
    ASSERT_GT(encoded2.size(), 0u);

    stage.setInverse(false);
    auto decoded2 = ans_decode(stage, encoded2, N2, cs.stream, *pool2);
    ASSERT_EQ(decoded2.size(), N2);
    for (size_t i = 0; i < N2; ++i)
        EXPECT_EQ(decoded2[i], h_in2[i]) << "pass2 mismatch at i=" << i;
}

// ─────────────────────────────────────────────────────────────────────────────
// AN9 — FileRoundTrip
// Exercises the full serialization path: stage configs, FZM header, checksums,
// StageFactory (createStage with StageType::ANS and 12-byte config).
// ─────────────────────────────────────────────────────────────────────────────
TEST(ANSStage, FileRoundTrip) {
    const size_t N        = 4096;
    const size_t in_bytes = N;
    const std::string tmp = "/tmp/test_ans_stage.fzm";

    std::vector<uint8_t> h_in(N);
    for (size_t i = 0; i < N; ++i) h_in[i] = static_cast<uint8_t>(i % 128);

    Pipeline p(in_bytes, MemoryStrategy::PREALLOCATE);
    p.addStage<ANSStage>();
    p.finalize();

    CudaStream cs;
    auto res = pipeline_file_round_trip<uint8_t>(p, h_in, cs.stream, tmp);
    std::remove(tmp.c_str());

    ASSERT_EQ(res.data.size(), N);
    for (size_t i = 0; i < N; ++i)
        EXPECT_EQ(res.data[i], h_in[i]) << "mismatch at i=" << i;
}

// ─────────────────────────────────────────────────────────────────────────────
// AN10 — LorenzoQuantPipeline
// End-to-end: LorenzoQuantStage<float,uint16_t> → ANSStage.
// ANS receives the uint16_t codes stream as raw bytes (byte-transparent, UNKNOWN type).
// original_bytes_ captures the uint16_t byte count so the inverse ANS output
// is sized correctly for the inverse LorenzoQuant.
// ─────────────────────────────────────────────────────────────────────────────
TEST(ANSStage, LorenzoQuantPipeline) {
    constexpr size_t N  = 1 << 14;  // 16 K floats
    constexpr float  EB = 1e-2f;
    const size_t in_bytes = N * sizeof(float);

    auto h_in = make_smooth_data<float>(N);

    Pipeline p(in_bytes, MemoryStrategy::MINIMAL);

    auto* lq = p.addStage<LorenzoQuantStage<float, uint16_t>>();
    lq->setErrorBound(EB);
    lq->setQuantRadius(512);
    lq->setOutlierCapacity(0.2f);
    // Zigzag maps signed deltas → non-negative codes in [0, 2*radius−2] = [0, 1022],
    // keeping the symbol distribution compact for ANS.
    lq->setZigzagCodes(true);

    auto* ans = p.addStage<ANSStage>();
    p.connect(ans, lq, "codes");

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
// AN11 — PartialBlock
// Input is 1000 bytes — less than kDefaultBlockSize (4096).  Exercises the
// single-block path where max_blocks=1 and the input does not fill the block.
// ─────────────────────────────────────────────────────────────────────────────
TEST(ANSStage, PartialBlock) {
    const size_t N = 1000;  // intentionally < kDefaultBlockSize (4096)
    std::vector<uint8_t> h_in(N);
    for (size_t i = 0; i < N; ++i) h_in[i] = static_cast<uint8_t>(i % 200);

    CudaStream cs;
    auto pool = make_test_pool(N);

    ANSStage stage;
    auto encoded = ans_encode(stage, h_in, cs.stream, *pool);
    ASSERT_GT(encoded.size(), 0u);

    stage.setInverse(false);
    auto decoded = ans_decode(stage, encoded, N, cs.stream, *pool);
    ASSERT_EQ(decoded.size(), N);
    for (size_t i = 0; i < N; ++i)
        EXPECT_EQ(decoded[i], h_in[i]) << "mismatch at i=" << i;
}

// ─────────────────────────────────────────────────────────────────────────────
// AN12 — UnsupportedProbBitsThrows
// Only prob_bits=10 is supported (GpuANS.cu instantiates only that value).
// execute() must throw std::runtime_error when prob_bits != 10.
// ─────────────────────────────────────────────────────────────────────────────
TEST(ANSStage, UnsupportedProbBitsThrows) {
    const size_t N = 4096;
    std::vector<uint8_t> h_in(N, 42);

    CudaStream cs;
    auto pool = make_test_pool(N);

    ANSStage stage;
    stage.setProbBits(11);  // not currently supported

    CudaBuffer<uint8_t> d_in(N);
    CudaBuffer<uint8_t> d_out(N * 2 + 8192);
    d_in.upload(h_in, cs.stream);
    cudaStreamSynchronize(cs.stream);

    std::vector<void*>  inputs  = {d_in.void_ptr()};
    std::vector<void*>  outputs = {d_out.void_ptr()};
    std::vector<size_t> sizes   = {N};
    EXPECT_THROW(stage.execute(cs.stream, pool.get(), inputs, outputs, sizes),
                 std::runtime_error);
}
