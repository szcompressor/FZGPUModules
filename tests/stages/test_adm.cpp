/**
 * tests/stages/test_adm.cpp
 *
 * GPU unit tests for ADMStage (Adaptive Data Mapping, MANS backend).
 * Forward input: uint16_t[] or uint32_t[] → opaque ADM payload (uint8_t[]).
 * Inverse input: ADM payload → original integer array.
 * Not graph-compatible (D2H sync in both directions).
 *
 *   AD1   ADMStage/U16RoundTrip           — uint16_t forward+inverse exact match (4096 elems)
 *   AD2   ADMStage/U32RoundTrip           — uint32_t forward+inverse exact match (4096 elems)
 *   AD3   ADMStage/SmallInput             — N=200 (< 512 = one warp block) round-trip
 *   AD4   ADMStage/LargeInput             — N=600000 (gsize > 1024, uses Thrust path)
 *   AD5   ADMStage/ZeroInput              — n=0 does not crash; output size is 0
 *   AD6   ADMStage/CompressionRatio       — compressed size < input for smooth data
 *   AD7   ADMStage/SerializeDeserialize   — 12-byte header round-trips dtype + num_elements
 *   AD8   ADMStage/SaveRestoreState       — saveState + deserializeHeader + restoreState
 *   AD9   ADMStage/GraphCompatible        — isGraphCompatible() == false
 *   AD10  ADMStage/PipelineIntegration    — Pipeline round-trip with uint16_t
 *   AD11  ADMStage/FileRoundTrip          — writeToFile → decompressFromFile
 *   AD12  ADMStage/LorenzoQuantADMANS     — LorenzoQuant→ADM→ANS end-to-end float pipeline
 */

#include <gtest/gtest.h>
#include "helpers/fz_test_utils.h"
#include "helpers/stage_harness.h"
#include "transforms/adm/adm_stage.h"
#include "fzgpumodules.h"
#include "fused/lorenzo_quant/lorenzo_quant.h"
#include "coders/ans/ans_stage.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

using namespace fz;
using namespace fz_test;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers: single-pass encode / decode via stage API
// ─────────────────────────────────────────────────────────────────────────────

template <typename T>
static std::vector<uint8_t> adm_encode(
    ADMStage& stage,
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
    stage.setInverse(false);
    stage.execute(stream, &pool, inputs, outputs, sizes);
    cudaStreamSynchronize(stream);

    const size_t actual = stage.getActualOutputSize(0);
    std::vector<uint8_t> h_out(actual);
    if (actual > 0)
        cudaMemcpy(h_out.data(), d_out.get(), actual, cudaMemcpyDeviceToHost);
    return h_out;
}

template <typename T>
static std::vector<T> adm_decode(
    ADMStage& stage,
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

// Generate smooth uint16_t data clustered around center values (good for ADM).
static std::vector<uint16_t> make_u16_data(size_t n, uint16_t center = 500) {
    std::vector<uint16_t> v(n);
    for (size_t i = 0; i < n; ++i) {
        double x = static_cast<double>(center)
                 + 200.0 * std::sin(static_cast<double>(i) * 0.01)
                 + 50.0  * std::cos(static_cast<double>(i) * 0.05);
        v[i] = static_cast<uint16_t>(std::max(0.0, std::min(65535.0, x)));
    }
    return v;
}

static std::vector<uint32_t> make_u32_data(size_t n, uint32_t center = 50000) {
    std::vector<uint32_t> v(n);
    for (size_t i = 0; i < n; ++i) {
        // Amplitude kept ≤ 250 so per-thread bit_offset ≤ 4 bytes, within
        // local_bits[kChunk * kMaxSignalBytesU32 = 64 bytes].  ADM is
        // designed for bounded quantization codes, not arbitrary uint32_t.
        double x = static_cast<double>(center)
                 + 200.0 * std::sin(static_cast<double>(i) * 0.01)
                 + 50.0  * std::cos(static_cast<double>(i) * 0.05);
        v[i] = static_cast<uint32_t>(std::max(0.0, std::min(4294967295.0, x)));
    }
    return v;
}

// ─────────────────────────────────────────────────────────────────────────────
// AD1 — U16RoundTrip
// ─────────────────────────────────────────────────────────────────────────────
TEST(ADMStage, U16RoundTrip) {
    const size_t N = 4096;
    auto h_in = make_u16_data(N);

    CudaStream cs;
    auto pool = make_test_pool(N * sizeof(uint16_t), 20.0f);

    ADMStage stage;
    stage.setDtype(ADMDtype::U16);

    auto encoded = adm_encode<uint16_t>(stage, h_in, cs.stream, *pool);
    ASSERT_GT(encoded.size(), 0u);

    auto decoded = adm_decode<uint16_t>(stage, encoded, N, cs.stream, *pool);
    ASSERT_EQ(decoded.size(), N);
    for (size_t i = 0; i < N; ++i)
        EXPECT_EQ(decoded[i], h_in[i]) << "mismatch at i=" << i;
}

// ─────────────────────────────────────────────────────────────────────────────
// AD2 — U32RoundTrip
// ─────────────────────────────────────────────────────────────────────────────
TEST(ADMStage, U32RoundTrip) {
    const size_t N = 4096;
    auto h_in = make_u32_data(N);

    CudaStream cs;
    auto pool = make_test_pool(N * sizeof(uint32_t), 20.0f);

    ADMStage stage;
    stage.setDtype(ADMDtype::U32);

    auto encoded = adm_encode<uint32_t>(stage, h_in, cs.stream, *pool);
    ASSERT_GT(encoded.size(), 0u);

    auto decoded = adm_decode<uint32_t>(stage, encoded, N, cs.stream, *pool);
    ASSERT_EQ(decoded.size(), N);
    for (size_t i = 0; i < N; ++i)
        EXPECT_EQ(decoded[i], h_in[i]) << "mismatch at i=" << i;
}

// ─────────────────────────────────────────────────────────────────────────────
// AD3 — SmallInput
// N=200 is less than one ADM warp block (kBlockElems=512) — exercises the
// single-block partial-fill path.
// ─────────────────────────────────────────────────────────────────────────────
TEST(ADMStage, SmallInput) {
    const size_t N = 200;
    auto h_in = make_u16_data(N, 300);

    CudaStream cs;
    auto pool = make_test_pool(N * sizeof(uint16_t), 20.0f);

    ADMStage stage;
    stage.setDtype(ADMDtype::U16);

    auto encoded = adm_encode<uint16_t>(stage, h_in, cs.stream, *pool);
    ASSERT_GT(encoded.size(), 0u);

    auto decoded = adm_decode<uint16_t>(stage, encoded, N, cs.stream, *pool);
    ASSERT_EQ(decoded.size(), N);
    for (size_t i = 0; i < N; ++i)
        EXPECT_EQ(decoded[i], h_in[i]) << "mismatch at i=" << i;
}

// ─────────────────────────────────────────────────────────────────────────────
// AD4 — LargeInput
// N=600000 gives gsize=1172 > kDecoupledMaxGsize(1024) — forces the Thrust
// exclusive_scan fallback path in mapping_uint16.cu.
// ─────────────────────────────────────────────────────────────────────────────
TEST(ADMStage, LargeInput) {
    const size_t N = 600000;
    auto h_in = make_u16_data(N);

    CudaStream cs;
    auto pool = make_test_pool(N * sizeof(uint16_t), 20.0f);

    ADMStage stage;
    stage.setDtype(ADMDtype::U16);

    auto encoded = adm_encode<uint16_t>(stage, h_in, cs.stream, *pool);
    ASSERT_GT(encoded.size(), 0u);

    auto decoded = adm_decode<uint16_t>(stage, encoded, N, cs.stream, *pool);
    ASSERT_EQ(decoded.size(), N);
    for (size_t i = 0; i < N; ++i)
        EXPECT_EQ(decoded[i], h_in[i]) << "mismatch at i=" << i;
}

// ─────────────────────────────────────────────────────────────────────────────
// AD5 — ZeroInput
// ─────────────────────────────────────────────────────────────────────────────
TEST(ADMStage, ZeroInput) {
    ADMStage stage;
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
// AD6 — CompressionRatio
// Smooth uint16_t data should compress to fewer bytes than the raw input.
// ─────────────────────────────────────────────────────────────────────────────
TEST(ADMStage, CompressionRatio) {
    const size_t N = 8192;
    auto h_in = make_u16_data(N);
    const size_t in_bytes = N * sizeof(uint16_t);

    CudaStream cs;
    auto pool = make_test_pool(in_bytes, 20.0f);

    ADMStage stage;
    stage.setDtype(ADMDtype::U16);

    auto encoded = adm_encode<uint16_t>(stage, h_in, cs.stream, *pool);
    EXPECT_LT(encoded.size(), in_bytes)
        << "smooth data should compress below raw size; got " << encoded.size()
        << " vs " << in_bytes << " raw";
}

// ─────────────────────────────────────────────────────────────────────────────
// AD7 — SerializeDeserialize
// 12-byte header: [0] dtype, [1..3] reserved=0, [4..11] num_elements uint64_t LE.
// num_elements is set by the forward execute(); simulate that by running one.
// ─────────────────────────────────────────────────────────────────────────────
TEST(ADMStage, SerializeDeserialize) {
    const size_t N = 1024;
    auto h_in = make_u32_data(N);  // must match dtype=U32 below

    CudaStream cs;
    auto pool = make_test_pool(N * sizeof(uint32_t), 20.0f);

    ADMStage original;
    original.setDtype(ADMDtype::U32);  // non-default
    // Run forward to populate num_elements_
    auto encoded = adm_encode<uint32_t>(original, h_in, cs.stream, *pool);
    (void)encoded;

    uint8_t buf[128] = {};
    size_t written = original.serializeHeader(0, buf, sizeof(buf));
    EXPECT_EQ(written, 12u);
    EXPECT_EQ(buf[0], static_cast<uint8_t>(ADMDtype::U32));  // dtype
    EXPECT_EQ(buf[1], 0u);  // reserved
    EXPECT_EQ(buf[2], 0u);
    EXPECT_EQ(buf[3], 0u);

    uint64_t stored_n = 0;
    std::memcpy(&stored_n, buf + 4, sizeof(uint64_t));
    EXPECT_EQ(stored_n, static_cast<uint64_t>(N));

    ADMStage restored;
    restored.deserializeHeader(buf, written);
    EXPECT_EQ(restored.getDtype(), ADMDtype::U32);
}

// ─────────────────────────────────────────────────────────────────────────────
// AD8 — SaveRestoreState
// saveState + deserializeHeader (simulating decompressor) + restoreState
// should leave the stage configuration unchanged.
// ─────────────────────────────────────────────────────────────────────────────
TEST(ADMStage, SaveRestoreState) {
    ADMStage s;
    s.setDtype(ADMDtype::U16);
    s.saveState();

    // Simulate deserializeHeader with different values (as the decompressor would).
    uint8_t fake_hdr[12] = {};
    fake_hdr[0] = static_cast<uint8_t>(ADMDtype::U32);
    uint64_t fake_n = 999999;
    std::memcpy(fake_hdr + 4, &fake_n, sizeof(uint64_t));
    s.deserializeHeader(fake_hdr, sizeof(fake_hdr));
    EXPECT_EQ(s.getDtype(), ADMDtype::U32);

    s.restoreState();
    EXPECT_EQ(s.getDtype(), ADMDtype::U16);
}

// ─────────────────────────────────────────────────────────────────────────────
// AD9 — GraphCompatible
// ADMStage performs D2H copies in both forward (payload size readback) and
// inverse (output size readback) — incompatible with CUDA Graph capture.
// ─────────────────────────────────────────────────────────────────────────────
TEST(ADMStage, GraphCompatible) {
    ADMStage stage;
    EXPECT_FALSE(stage.isGraphCompatible());
}

// ─────────────────────────────────────────────────────────────────────────────
// AD10 — PipelineIntegration
// Full Pipeline round-trip: uint16_t → ADM → inverse ADM → uint16_t.
// pipeline_round_trip<uint16_t> treats the input bytes as uint16_t elements.
// ─────────────────────────────────────────────────────────────────────────────
TEST(ADMStage, PipelineIntegration) {
    const size_t N        = 4096;
    const size_t in_bytes = N * sizeof(uint16_t);

    auto h_in = make_u16_data(N);

    Pipeline p(in_bytes, MemoryStrategy::PREALLOCATE);
    auto* adm = p.addStage<ADMStage>();
    adm->setDtype(ADMDtype::U16);
    p.finalize();

    CudaStream cs;
    auto res = pipeline_round_trip<uint16_t>(p, h_in, cs.stream);

    ASSERT_EQ(res.data.size(), N);
    for (size_t i = 0; i < N; ++i)
        EXPECT_EQ(res.data[i], h_in[i]) << "mismatch at i=" << i;
    EXPECT_GT(res.compressed_bytes, 0u);
}

// ─────────────────────────────────────────────────────────────────────────────
// AD11 — FileRoundTrip
// Exercises the full serialization path: stage configs, FZM header, checksums,
// StageFactory (createStage with StageType::ADM and 12-byte config).
// ─────────────────────────────────────────────────────────────────────────────
TEST(ADMStage, FileRoundTrip) {
    const size_t N        = 4096;
    const size_t in_bytes = N * sizeof(uint16_t);
    const std::string tmp = "/tmp/test_adm_stage.fzm";

    auto h_in = make_u16_data(N);

    Pipeline p(in_bytes, MemoryStrategy::PREALLOCATE);
    auto* adm = p.addStage<ADMStage>();
    adm->setDtype(ADMDtype::U16);
    p.finalize();

    CudaStream cs;
    auto res = pipeline_file_round_trip<uint16_t>(p, h_in, cs.stream, tmp);
    std::remove(tmp.c_str());

    ASSERT_EQ(res.data.size(), N);
    for (size_t i = 0; i < N; ++i)
        EXPECT_EQ(res.data[i], h_in[i]) << "mismatch at i=" << i;
}

// ─────────────────────────────────────────────────────────────────────────────
// AD12 — LorenzoQuantADMANS
// End-to-end float pipeline: LorenzoQuantStage<float,uint16_t> → ADMStage →
// ANSStage.  ADM remaps the uint16_t code stream to a compact byte domain;
// ANS then entropy-codes those bytes.  Full round-trip must recover float data
// within the Lorenzo error bound.
// ─────────────────────────────────────────────────────────────────────────────
TEST(ADMStage, LorenzoQuantADMANS) {
#if defined(FZGMOD_BACKEND_HIP) || defined(FZGMOD_BACKEND_SYCL)
    GTEST_SKIP() << "ANSStage is not supported on the current GPU backend "
                    "(vendored dietgpu PTX lanemask assembly, no HIP/SYCL "
                    "translation)";
#endif
    constexpr size_t N  = 1 << 14;  // 16 K floats
    constexpr float  EB = 1e-2f;
    const size_t in_bytes = N * sizeof(float);

    auto h_in = make_smooth_data<float>(N);

    Pipeline p(in_bytes, MemoryStrategy::MINIMAL);

    auto* lq = p.addStage<LorenzoQuantStage<float, uint16_t>>();
    lq->setErrorBound(EB);
    lq->setQuantRadius(512);
    lq->setOutlierCapacity(0.2f);
    lq->setZigzagCodes(true);

    auto* adm = p.addStage<ADMStage>();
    adm->setDtype(ADMDtype::U16);
    p.connect(adm, lq, "codes");

    auto* ans = p.addStage<ANSStage>();
    p.connect(ans, adm);

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
