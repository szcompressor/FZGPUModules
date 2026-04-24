/**
 * tests/stages/test_bitpack.cpp
 *
 * GPU unit tests for BitpackStage<T>.
 *
 * BitpackStage packs N-bit integers into a dense byte stream (forward) and
 * restores them exactly (inverse). nbits must be a power of two.
 *
 * Properties verified:
 *   1.  Round-trip uint16_t, nbits=8  (sub-word, byte-aligned elements).
 *   2.  Round-trip uint16_t, nbits=16 (identity — full width kept).
 *   3.  Round-trip uint32_t, nbits=16 (half-width packing).
 *   4.  Round-trip uint32_t, nbits=32 (identity).
 *   5.  Round-trip uint8_t,  nbits=4  (sub-byte elements).
 *   6.  Zero-element input (n=0) does not crash; output size is 0.
 *   7.  estimateOutputSizes formula: ceil(n*nbits/8).
 *   8.  Packed output size is strictly smaller than input when nbits < 8*sizeof(T).
 *   9.  serializeHeader → deserializeHeader restores nbits.
 *   10. saveState + deserializeHeader (new nbits) + restoreState recovers original.
 *   11. Pipeline integration: Quantizer<float,uint16_t> → Bitpack<uint16_t> round-trip.
 *   12. isGraphCompatible() returns true.
 *   13. setNBits with invalid value (non-power-of-two) throws.
 *   14. setNBits with value exceeding 8*sizeof(T) throws.
 */

#include <gtest/gtest.h>
#include "helpers/fz_test_utils.h"
#include "helpers/stage_harness.h"
#include "transforms/bitpack/bitpack_stage.h"
#include "fzgpumodules.h"

#include <cstdint>
#include <cstring>
#include <numeric>
#include <vector>
#include <algorithm>
#include <random>

using namespace fz;
using namespace fz_test;

// ─────────────────────────────────────────────────────────────────────────────
// Helper: run one BitpackStage pass on a host vector of T, return host result
// ─────────────────────────────────────────────────────────────────────────────

template<typename T>
static std::vector<uint8_t> pack(
    BitpackStage<T>& stage,
    const std::vector<T>& h_in,
    cudaStream_t stream,
    fz::MemoryPool& pool)
{
    const size_t in_bytes = h_in.size() * sizeof(T);
    const size_t out_est  = stage.estimateOutputSizes({in_bytes})[0];

    CudaBuffer<T>       d_in(h_in.size());
    CudaBuffer<uint8_t> d_out(std::max(out_est, size_t(1)));
    d_in.upload(h_in, stream);
    cudaStreamSynchronize(stream);

    std::vector<void*>  inputs  = {d_in.void_ptr()};
    std::vector<void*>  outputs = {d_out.void_ptr()};
    std::vector<size_t> sizes   = {in_bytes};
    stage.execute(stream, &pool, inputs, outputs, sizes);
    cudaStreamSynchronize(stream);

    const size_t actual = stage.getActualOutputSizesByName().at("output");
    std::vector<uint8_t> h_out(actual);
    cudaMemcpy(h_out.data(), d_out.get(), actual, cudaMemcpyDeviceToHost);
    return h_out;
}

template<typename T>
static std::vector<T> unpack(
    BitpackStage<T>& stage,
    const std::vector<uint8_t>& h_packed,
    size_t n_elements,
    cudaStream_t stream,
    fz::MemoryPool& pool)
{
    const size_t out_bytes = n_elements * sizeof(T);

    CudaBuffer<uint8_t> d_in(std::max(h_packed.size(), size_t(1)));
    CudaBuffer<T>       d_out(std::max(n_elements, size_t(1)));
    cudaMemcpy(d_in.get(), h_packed.data(), h_packed.size(), cudaMemcpyHostToDevice);

    std::vector<void*>  inputs  = {d_in.void_ptr()};
    std::vector<void*>  outputs = {d_out.void_ptr()};
    std::vector<size_t> sizes   = {h_packed.size()};
    stage.execute(stream, &pool, inputs, outputs, sizes);
    cudaStreamSynchronize(stream);

    std::vector<T> h_out(n_elements);
    cudaMemcpy(h_out.data(), d_out.get(), out_bytes, cudaMemcpyDeviceToHost);
    return h_out;
}

// Round-trip helper: mask input to nbits, pack, unpack, compare.
template<typename T>
static void round_trip(uint8_t nbits, size_t n_elements) {
    std::mt19937 rng(42);
    std::uniform_int_distribution<uint64_t> dist(0, (uint64_t(1) << nbits) - 1);
    std::vector<T> original(n_elements);
    for (auto& v : original) v = static_cast<T>(dist(rng));

    CudaStream cs;
    auto pool = make_test_pool(n_elements * sizeof(T) * 4);

    BitpackStage<T> enc;
    enc.setNBits(nbits);

    const auto packed = pack<T>(enc, original, cs.stream, *pool);

    // Transfer state so the decoder knows num_elements and nbits.
    uint8_t hdr[10] = {};
    enc.serializeHeader(0, hdr, sizeof(hdr));

    BitpackStage<T> dec;
    dec.setInverse(true);
    dec.deserializeHeader(hdr, sizeof(hdr));

    const auto restored = unpack<T>(dec, packed, n_elements, cs.stream, *pool);

    ASSERT_EQ(restored.size(), original.size());
    for (size_t i = 0; i < original.size(); ++i) {
        EXPECT_EQ(restored[i], original[i])
            << "Mismatch at element " << i
            << " (nbits=" << (int)nbits << ", T=" << sizeof(T)*8 << "-bit)";
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 1-5. Round-trip tests
// ─────────────────────────────────────────────────────────────────────────────

TEST(BitpackStage, RoundTrip_U16_NBits8) {
    round_trip<uint16_t>(8, 4096);
}

TEST(BitpackStage, RoundTrip_U16_NBits16_Identity) {
    round_trip<uint16_t>(16, 4096);
}

TEST(BitpackStage, RoundTrip_U32_NBits16) {
    round_trip<uint32_t>(16, 4096);
}

TEST(BitpackStage, RoundTrip_U32_NBits32_Identity) {
    round_trip<uint32_t>(32, 4096);
}

TEST(BitpackStage, RoundTrip_U8_NBits4) {
    round_trip<uint8_t>(4, 4096);
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. Zero-element input
// ─────────────────────────────────────────────────────────────────────────────

TEST(BitpackStage, ZeroInput) {
    CudaStream cs;
    auto pool = make_test_pool(1024);

    BitpackStage<uint16_t> enc;
    enc.setNBits(8);

    std::vector<uint16_t> empty;
    const auto packed = pack<uint16_t>(enc, empty, cs.stream, *pool);
    EXPECT_EQ(packed.size(), 0u);
    EXPECT_EQ(enc.getActualOutputSizesByName().at("output"), 0u);
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. estimateOutputSizes formula
// ─────────────────────────────────────────────────────────────────────────────

TEST(BitpackStage, OutputSizeFormula) {
    BitpackStage<uint16_t> s;

    // nbits=8, n=1024 → 1024 * 8 / 8 = 1024 bytes
    s.setNBits(8);
    EXPECT_EQ(s.estimateOutputSizes({1024 * sizeof(uint16_t)})[0], 1024u);

    // nbits=4, n=1024 → 1024 * 4 / 8 = 512 bytes
    s.setNBits(4);
    EXPECT_EQ(s.estimateOutputSizes({1024 * sizeof(uint16_t)})[0], 512u);

    // nbits=1, n=1024 → 1024 * 1 / 8 = 128 bytes
    s.setNBits(1);
    EXPECT_EQ(s.estimateOutputSizes({1024 * sizeof(uint16_t)})[0], 128u);
}

// ─────────────────────────────────────────────────────────────────────────────
// 8. Packed size < input size when nbits < 8*sizeof(T)
// ─────────────────────────────────────────────────────────────────────────────

TEST(BitpackStage, PackedSizeSmallerThanInput) {
    CudaStream cs;
    auto pool = make_test_pool(64 * 1024);

    std::vector<uint16_t> data(4096, 0x00FF);  // only low 8 bits set

    BitpackStage<uint16_t> enc;
    enc.setNBits(8);

    const auto packed = pack<uint16_t>(enc, data, cs.stream, *pool);
    EXPECT_LT(packed.size(), data.size() * sizeof(uint16_t));
    EXPECT_EQ(packed.size(), 4096u);  // 4096 * 8 / 8 = 4096 bytes
}

// ─────────────────────────────────────────────────────────────────────────────
// 9. serializeHeader → deserializeHeader restores nbits
// ─────────────────────────────────────────────────────────────────────────────

TEST(BitpackStage, SerializeDeserialize) {
    BitpackStage<uint32_t> a;
    a.setNBits(16);

    uint8_t buf[10] = {};
    const size_t written = a.serializeHeader(0, buf, sizeof(buf));
    EXPECT_EQ(written, 10u);

    BitpackStage<uint32_t> b;
    b.deserializeHeader(buf, written);
    EXPECT_EQ(b.getNBits(), 16u);
}

// ─────────────────────────────────────────────────────────────────────────────
// 10. saveState + deserializeHeader + restoreState recovers original
// ─────────────────────────────────────────────────────────────────────────────

TEST(BitpackStage, SaveRestoreState) {
    BitpackStage<uint16_t> s;
    s.setNBits(8);

    s.saveState();

    // Simulate what the pipeline does before decompression: call deserializeHeader
    // with a header that has a different nbits baked in.
    uint8_t alien_hdr[10] = {};
    alien_hdr[0] = static_cast<uint8_t>(DataType::UINT16);
    alien_hdr[1] = 4;  // different nbits
    s.deserializeHeader(alien_hdr, sizeof(alien_hdr));
    EXPECT_EQ(s.getNBits(), 4u);

    s.restoreState();
    EXPECT_EQ(s.getNBits(), 8u);
}

// ─────────────────────────────────────────────────────────────────────────────
// 11. Pipeline integration: Quantizer → Bitpack round-trip
// ─────────────────────────────────────────────────────────────────────────────

TEST(BitpackStage, PipelineIntegration) {
    const size_t N = 1024;
    const float eb = 0.01f;

    auto h_input = make_smooth_data<float>(N);
    const size_t in_bytes = N * sizeof(float);

    // Single pipeline instance: compress and decompress share state
    // (inverse DAG is built from the forward pass).
    Pipeline p(in_bytes, MemoryStrategy::PREALLOCATE);
    auto* quant   = p.addStage<QuantizerStage<float, uint16_t>>();
    auto* bitpack = p.addStage<BitpackStage<uint16_t>>();
    quant->setErrorBound(eb);
    quant->setQuantRadius(32768);
    quant->setZigzagCodes(false);
    // nbits=16 is identity packing for uint16_t: verifies pipeline wiring without
    // adding truncation error on top of quantization error.
    bitpack->setNBits(16);
    p.connect(bitpack, quant, "codes");
    p.finalize();

    CudaStream cs;
    auto res = fz_test::pipeline_round_trip<float>(p, h_input, cs.stream);

    EXPECT_LE(res.max_error, eb * 2.0)
        << "Max error " << res.max_error << " exceeds bound " << eb;
    EXPECT_LT(res.compressed_bytes, in_bytes)
        << "Compressed size should be smaller than input";
}

// ─────────────────────────────────────────────────────────────────────────────
// 12. isGraphCompatible
// ─────────────────────────────────────────────────────────────────────────────

TEST(BitpackStage, GraphCompatible) {
    BitpackStage<uint16_t> fwd;
    EXPECT_TRUE(fwd.isGraphCompatible());

    BitpackStage<uint16_t> inv;
    inv.setInverse(true);
    EXPECT_TRUE(inv.isGraphCompatible());
}

// ─────────────────────────────────────────────────────────────────────────────
// 13-14. setNBits validation
// ─────────────────────────────────────────────────────────────────────────────

TEST(BitpackStage, InvalidNBits_NonPowerOfTwo) {
    BitpackStage<uint16_t> s;
    EXPECT_THROW(s.setNBits(3),  std::invalid_argument);
    EXPECT_THROW(s.setNBits(5),  std::invalid_argument);
    EXPECT_THROW(s.setNBits(12), std::invalid_argument);
}

TEST(BitpackStage, InvalidNBits_ExceedsWidth) {
    BitpackStage<uint16_t> s;
    EXPECT_THROW(s.setNBits(32), std::invalid_argument);  // max is 16 for uint16_t
    EXPECT_THROW(s.setNBits(0),  std::invalid_argument);
}
