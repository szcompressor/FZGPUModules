/**
 * tests/stages/test_lorenzo.cpp
 *
 * Unit tests for LorenzoStage<T> (lossless delta predictor) and
 * LorenzoQuantStage<TIn, TCode> (fused lossy predictor + quantizer).
 *
 *   LZ1  LorenzoStage/RoundTrip1D                — 1-D int32 ramp, exact reconstruction
 *   LZ2  LorenzoStage/RoundTrip2D                — 2-D int32 grid, exact reconstruction
 *   LZ3  LorenzoStage/RoundTrip3D                — 3-D int32 grid, exact reconstruction
 *   LZ4  LorenzoStage/SerializeDeserialize       — config round-trip via header bytes
 *   LZ5  LorenzoStage/StageTypeId               — getStageTypeId() == StageType::LORENZO
 *   LZ6  LorenzoStage/GraphCompatible            — forward stage returns isGraphCompatible()=true
 *   LZ7  LorenzoStage/Int16RoundTrip             — int16_t type instantiation, exact reconstruction
 *   LZ8  LorenzoQuantStage/DeterministicRecon    — two independent pipelines produce identical output
 *   LZ9  LorenzoStage/QuantizerLorenzoPipeline   — Quantizer→Lorenzo chained round-trip
 *
 *   Regression — the inverse scan used to launch one block per line with
 *   blockDim == the scanned extent, so any dimension above the 1024 max block
 *   size failed with `invalid configuration argument`. All axes are now scanned
 *   in fixed-size tiles with a running carry; these cover each axis past 1024
 *   and past a tile boundary:
 *   LZ10 LorenzoStage/RoundTrip2D_LargeNX         — nx = 3600 (> 1024)
 *   LZ11 LorenzoStage/RoundTrip2D_LargeNY         — ny = 1800 (> 1024)
 *   LZ12 LorenzoStage/RoundTrip2D_BothLarge       — 3600 x 1800, the real CLDHGH field
 *   LZ13 LorenzoStage/RoundTrip3D_LargeNX         — 3-D, nx = 1500 (> 1024)
 *   LZ14 LorenzoStage/RoundTrip3D_LargeNY         — 3-D, ny = 1500 (> 1024)
 *   LZ15 LorenzoStage/RoundTrip3D_LargeNZ         — 3-D, nz = 1500 (> 1024)
 *   LZ16 LorenzoStage/RoundTrip2D_TileBoundaries  — extents that are not multiples of the tile
 *
 *   Per-block mean centering (FSZ adaptive centering):
 *   LZ17 LorenzoStage/CenteringRoundTrip1D         — lossless with a large constant offset
 *   LZ18 LorenzoStage/CenteringPartialFinalBlock   — mean divides by live count, not blockDim
 *   LZ19 LorenzoStage/CenteringNonPowerOfTwoBlock  — reduction over a ragged blockDim
 *   LZ20 LorenzoStage/CenteringSerializeDeserialize— flag survives the header; legacy = off
 *   LZ21 LorenzoStage/CenteringPortCount           — adds the "means" port (out fwd, in inv)
 *   LZ22 LorenzoStage/CenteringRequiresBlockMode   — rejected without setBlockSize
 *   LZ23 LorenzoStage/CenteringMatchesUncentered   — same reconstruction either way
 *   LZ24 LorenzoQuantStage/CenteringRoundTrip1D    — fused stage, within the error bound
 *   LZ25 LorenzoQuantStage/CenteringRejectsMultiDim— fused stage throws for 2-D/3-D
 */

#include <gtest/gtest.h>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <vector>

#include "predictors/lorenzo/lorenzo_stage.h"
#include "fused/lorenzo_quant/lorenzo_quant.h"
#include "quantizers/quantizer/quantizer.h"
#include "coders/adaptive_bitpack/adaptive_bitpack_stage.h"
#include "helpers/stage_harness.h"
#include "helpers/fz_test_utils.h"

using namespace fz;
using namespace fz_test;

// ─────────────────────────────────────────────────────────────────────────────
// LZ1: RoundTrip1D — 1-D int32 ramp reconstructs exactly
// ─────────────────────────────────────────────────────────────────────────────
TEST(LorenzoStage, RoundTrip1D) {
    const size_t N = 4096;
    const size_t in_bytes = N * sizeof(int32_t);

    std::vector<int32_t> h_input(N);
    std::iota(h_input.begin(), h_input.end(), 0);

    Pipeline p(in_bytes, MemoryStrategy::PREALLOCATE);
    auto* stage = p.addStage<LorenzoStage<int32_t>>();
    stage->setDims(N);
    p.finalize();

    CudaStream cs;
    auto res = pipeline_round_trip<int32_t>(p, h_input, cs.stream);

    EXPECT_EQ(res.max_error, 0.0f);
}

// ─────────────────────────────────────────────────────────────────────────────
// LZ2: RoundTrip2D — 2-D int32 grid reconstructs exactly
// ─────────────────────────────────────────────────────────────────────────────
TEST(LorenzoStage, RoundTrip2D) {
    const size_t NX = 64, NY = 64;
    const size_t N = NX * NY;
    const size_t in_bytes = N * sizeof(int32_t);

    std::vector<int32_t> h_input(N);
    for (size_t i = 0; i < N; ++i)
        h_input[i] = static_cast<int32_t>(i % 200);

    Pipeline p(in_bytes, MemoryStrategy::PREALLOCATE);
    auto* stage = p.addStage<LorenzoStage<int32_t>>();
    stage->setDims(NX, NY);
    p.finalize();

    CudaStream cs;
    auto res = pipeline_round_trip<int32_t>(p, h_input, cs.stream);

    EXPECT_EQ(res.max_error, 0.0f);
}

// ─────────────────────────────────────────────────────────────────────────────
// LZ3: RoundTrip3D — 3-D int32 grid reconstructs exactly
// ─────────────────────────────────────────────────────────────────────────────
TEST(LorenzoStage, RoundTrip3D) {
    const size_t NX = 16, NY = 16, NZ = 16;
    const size_t N = NX * NY * NZ;
    const size_t in_bytes = N * sizeof(int32_t);

    std::vector<int32_t> h_input(N);
    for (size_t i = 0; i < N; ++i)
        h_input[i] = static_cast<int32_t>(i % 50);

    Pipeline p(in_bytes, MemoryStrategy::PREALLOCATE);
    auto* stage = p.addStage<LorenzoStage<int32_t>>();
    stage->setDims(NX, NY, NZ);
    p.finalize();

    CudaStream cs;
    auto res = pipeline_round_trip<int32_t>(p, h_input, cs.stream);

    EXPECT_EQ(res.max_error, 0.0f);
}

// ─────────────────────────────────────────────────────────────────────────────
// LZ10-LZ16: large-dimension regression for the tiled inverse scan
//
// Before the tiled rewrite these all died in cudaGetLastError() with
// "invalid configuration argument": the scan launched blockDim == extent, and
// any extent over 1024 exceeds the maximum CUDA block dimension.
// ─────────────────────────────────────────────────────────────────────────────

static void lorenzo_round_trip_2d(size_t nx, size_t ny) {
    const size_t N = nx * ny;
    std::vector<int32_t> h_input(N);
    for (size_t i = 0; i < N; ++i)
        h_input[i] = static_cast<int32_t>(i % 200);

    Pipeline p(N * sizeof(int32_t), MemoryStrategy::PREALLOCATE);
    auto* stage = p.addStage<LorenzoStage<int32_t>>();
    stage->setDims(nx, ny);
    p.finalize();

    CudaStream cs;
    auto res = pipeline_round_trip<int32_t>(p, h_input, cs.stream);
    EXPECT_EQ(res.max_error, 0.0f) << "2-D " << nx << "x" << ny;
}

static void lorenzo_round_trip_3d(size_t nx, size_t ny, size_t nz) {
    const size_t N = nx * ny * nz;
    std::vector<int32_t> h_input(N);
    for (size_t i = 0; i < N; ++i)
        h_input[i] = static_cast<int32_t>(i % 50);

    Pipeline p(N * sizeof(int32_t), MemoryStrategy::PREALLOCATE);
    auto* stage = p.addStage<LorenzoStage<int32_t>>();
    stage->setDims(nx, ny, nz);
    p.finalize();

    CudaStream cs;
    auto res = pipeline_round_trip<int32_t>(p, h_input, cs.stream);
    EXPECT_EQ(res.max_error, 0.0f)
        << "3-D " << nx << "x" << ny << "x" << nz;
}

TEST(LorenzoStage, RoundTrip2D_LargeNX) { lorenzo_round_trip_2d(3600, 64); }
TEST(LorenzoStage, RoundTrip2D_LargeNY) { lorenzo_round_trip_2d(64, 1800); }

TEST(LorenzoStage, RoundTrip2D_BothLarge) {
    // The exact geometry of data/CLDHGH.f32, which used to crash the CLI.
    lorenzo_round_trip_2d(3600, 1800);
}

TEST(LorenzoStage, RoundTrip3D_LargeNX) { lorenzo_round_trip_3d(1500, 4, 4); }
TEST(LorenzoStage, RoundTrip3D_LargeNY) { lorenzo_round_trip_3d(4, 1500, 4); }
TEST(LorenzoStage, RoundTrip3D_LargeNZ) { lorenzo_round_trip_3d(4, 4, 1500); }

TEST(LorenzoStage, RoundTrip2D_TileBoundaries) {
    // Extents on either side of the 256-element tile: the carry between tiles
    // and the zero-padded partial tail are what these exercise.
    lorenzo_round_trip_2d(255, 257);
    lorenzo_round_trip_2d(256, 256);
    lorenzo_round_trip_2d(257, 255);
    lorenzo_round_trip_2d(1000, 3);
    lorenzo_round_trip_2d(1030, 3);
}

// ─────────────────────────────────────────────────────────────────────────────
// LZ4: SerializeDeserialize — dims survive serializeHeader/deserializeHeader
// ─────────────────────────────────────────────────────────────────────────────
TEST(LorenzoStage, SerializeDeserialize) {
    LorenzoStage<int32_t> original;
    original.setDims(128, 64, 2);

    uint8_t buf[128] = {};
    size_t written = original.serializeHeader(0, buf, sizeof(buf));
    EXPECT_EQ(written, sizeof(LorenzoConfig));

    LorenzoStage<int32_t> restored;
    restored.deserializeHeader(buf, written);

    EXPECT_EQ(original.getDims(), restored.getDims());
    EXPECT_EQ(original.ndim(), restored.ndim());
}

// ─────────────────────────────────────────────────────────────────────────────
// LZ5: StageTypeId — getStageTypeId() returns StageType::LORENZO
// ─────────────────────────────────────────────────────────────────────────────
TEST(LorenzoStage, StageTypeId) {
    EXPECT_EQ(LorenzoStage<int32_t>().getStageTypeId(),
              static_cast<uint16_t>(StageType::LORENZO));
}

// ─────────────────────────────────────────────────────────────────────────────
// LZ6: GraphCompatible — forward stage returns isGraphCompatible() = true
// ─────────────────────────────────────────────────────────────────────────────
TEST(LorenzoStage, GraphCompatible) {
    EXPECT_TRUE(LorenzoStage<int32_t>().isGraphCompatible());
}

// ─────────────────────────────────────────────────────────────────────────────
// LZ7: Int16RoundTrip — int16_t instantiation, exact reconstruction
// ─────────────────────────────────────────────────────────────────────────────
TEST(LorenzoStage, Int16RoundTrip) {
    const size_t N = 1024;
    const size_t in_bytes = N * sizeof(int16_t);

    std::vector<int16_t> h_input(N);
    for (size_t i = 0; i < N; ++i)
        h_input[i] = static_cast<int16_t>(i % 100);

    Pipeline p(in_bytes, MemoryStrategy::PREALLOCATE);
    auto* stage = p.addStage<LorenzoStage<int16_t>>();
    stage->setDims(N);
    p.finalize();

    CudaStream cs;
    auto res = pipeline_round_trip<int16_t>(p, h_input, cs.stream);

    EXPECT_EQ(res.max_error, 0.0f);
}

// ─────────────────────────────────────────────────────────────────────────────
// LZ8: DeterministicReconstruction — two independent LorenzoQuantStage
//       pipelines compress the same input and produce element-wise identical
//       reconstructions, guarding against non-deterministic GPU atomics.
// ─────────────────────────────────────────────────────────────────────────────
TEST(LorenzoQuantStage, DeterministicReconstruction) {
    const size_t N  = 4096;
    const float  EB = 1e-2f;
    auto h_input = make_smooth_data<float>(N);

    auto make_pipeline = [&]() {
        auto p = std::make_unique<Pipeline>(N * sizeof(float), MemoryStrategy::PREALLOCATE);
        auto* lq = p->addStage<LorenzoQuantStage<float, uint16_t>>();
        lq->setErrorBound(EB);
        lq->setQuantRadius(512);
        lq->setOutlierCapacity(0.2f);
        lq->setDims(N);
        p->finalize();
        return p;
    };

    CudaStream cs;
    auto p1 = make_pipeline();
    auto p2 = make_pipeline();

    auto res1 = pipeline_round_trip<float>(*p1, h_input, cs.stream);
    auto res2 = pipeline_round_trip<float>(*p2, h_input, cs.stream);

    ASSERT_EQ(res1.data.size(), N);
    ASSERT_EQ(res2.data.size(), N);

    for (size_t i = 0; i < N; i++) {
        EXPECT_EQ(res1.data[i], res2.data[i])
            << "Mismatch at element " << i;
    }
    EXPECT_LE(res1.max_error, EB * 1.01);
}

// ─────────────────────────────────────────────────────────────────────────────
// LZ9: QuantizerLorenzoPipeline — Quantizer→Lorenzo chained round-trip.
//       Verifies that LorenzoStage can follow QuantizerStage in a pipeline:
//       Lorenzo delta-codes the quantizer's integer codes; the inverse path
//       undoes the delta then maps codes back to floats.
//
//       Uses decompress(nullptr, ...) because multi-output QuantizerStage
//       leaves compressed data in the pipeline's internal pool rather than
//       a single concatenated buffer — the standard pipeline_round_trip
//       pattern is not applicable for this topology.
// ─────────────────────────────────────────────────────────────────────────────
TEST(LorenzoStage, QuantizerLorenzoPipelineRoundTrip) {
    const size_t N = 4096;
    const size_t in_bytes = N * sizeof(float);
    auto h_input = make_smooth_data<float>(N);

    Pipeline p(in_bytes, MemoryStrategy::PREALLOCATE);

    auto* quant = p.addStage<QuantizerStage<float, uint32_t>>();
    quant->setErrorBound(0.01f);
    quant->setErrorBoundMode(ErrorBoundMode::ABS);
    quant->setQuantRadius(32768);
    quant->setOutlierCapacity(0.1f);
    quant->setZigzagCodes(false);

    auto* lrz = p.addStage<LorenzoStage<int32_t>>();
    lrz->setDims(N);
    p.connect(lrz, quant, "codes");

    p.finalize();

    CudaBuffer<float> d_in(N);
    CudaStream cs;
    d_in.upload(h_input, cs.stream);
    cudaStreamSynchronize(cs.stream);

    void* d_comp = nullptr; size_t comp_sz = 0;
    p.compress(d_in.void_ptr(), in_bytes, &d_comp, &comp_sz, cs.stream);
    cudaStreamSynchronize(cs.stream);

    void* d_dec = nullptr; size_t dec_sz = 0;
    p.decompress(nullptr, comp_sz, &d_dec, &dec_sz, cs.stream);
    cudaStreamSynchronize(cs.stream);

    ASSERT_EQ(dec_sz, in_bytes);
    std::vector<float> h_recon(N);
    cudaMemcpy(h_recon.data(), d_dec, dec_sz, cudaMemcpyDeviceToHost);

    float max_err = max_abs_error(h_input, h_recon);
    EXPECT_LT(max_err, 0.011f);
}

// ─────────────────────────────────────────────────────────────────────────────
// LZ10-LZ14: setBlockSize — explicit 1-D block-local reset period (cuSZp-style)
//   LZ10  LorenzoStage/BlockSizeRoundTrip1D       — block_size=32 ramp, exact
//   LZ11  LorenzoStage/BlockSizePartialFinalBlock — N not a multiple of block_size, exact
//   LZ12  LorenzoStage/BlockSizeSerializeDeserialize — block_size survives header
//   LZ13  LorenzoStage/SetBlockSizeRejectsTooLarge   — n>1024 throws
//   LZ14  LorenzoStage/CuSZpFrontEndPipeline       — Quantizer(linear)→Lorenzo(block=32)
// ─────────────────────────────────────────────────────────────────────────────

TEST(LorenzoStage, BlockSizeRoundTrip1D) {
    const size_t N = 4096;
    const size_t in_bytes = N * sizeof(int32_t);

    std::vector<int32_t> h_input(N);
    for (size_t i = 0; i < N; ++i)
        h_input[i] = static_cast<int32_t>((i * 7) % 1000) - 500;

    Pipeline p(in_bytes, MemoryStrategy::PREALLOCATE);
    auto* stage = p.addStage<LorenzoStage<int32_t>>();
    stage->setBlockSize(32);   // cuSZp block-local reset
    p.finalize();

    CudaStream cs;
    auto res = pipeline_round_trip<int32_t>(p, h_input, cs.stream);
    EXPECT_EQ(res.max_error, 0.0f);
}

TEST(LorenzoStage, BlockSizePartialFinalBlock) {
    const size_t N = 1000;   // not a multiple of 32 → ragged final block
    const size_t in_bytes = N * sizeof(int32_t);

    std::vector<int32_t> h_input(N);
    for (size_t i = 0; i < N; ++i)
        h_input[i] = static_cast<int32_t>((i * 3) % 257);

    Pipeline p(in_bytes, MemoryStrategy::PREALLOCATE);
    auto* stage = p.addStage<LorenzoStage<int32_t>>();
    stage->setBlockSize(32);
    p.finalize();

    CudaStream cs;
    auto res = pipeline_round_trip<int32_t>(p, h_input, cs.stream);
    EXPECT_EQ(res.max_error, 0.0f);
}

TEST(LorenzoStage, BlockSizeSerializeDeserialize) {
    LorenzoStage<int32_t> original;
    original.setBlockSize(32);

    uint8_t buf[128] = {};
    size_t written = original.serializeHeader(0, buf, sizeof(buf));
    EXPECT_EQ(written, sizeof(LorenzoConfig));

    LorenzoStage<int32_t> restored;
    restored.deserializeHeader(buf, written);
    EXPECT_EQ(restored.getBlockSize(), 32u);

    // Legacy 16-byte header (no block_size) → defaults to 0.
    LorenzoStage<int32_t> legacy;
    legacy.deserializeHeader(buf, 16);
    EXPECT_EQ(legacy.getBlockSize(), 0u);
}

TEST(LorenzoStage, SetBlockSizeRejectsTooLarge) {
    LorenzoStage<int32_t> stage;
    EXPECT_THROW(stage.setBlockSize(2048), std::invalid_argument);
    EXPECT_NO_THROW(stage.setBlockSize(1024));
    EXPECT_NO_THROW(stage.setBlockSize(0));   // 0 = default behavior
}

TEST(LorenzoStage, CuSZpFrontEndPipeline) {
    // The cuSZp modular front-end: linear quantizer (signed codes, no outliers)
    // feeding a block-local Lorenzo predictor.
    const size_t N = 4096;
    const size_t in_bytes = N * sizeof(float);
    auto h_input = make_smooth_data<float>(N);

    Pipeline p(in_bytes, MemoryStrategy::PREALLOCATE);

    auto* quant = p.addStage<QuantizerStage<float, uint32_t>>();
    quant->setErrorBound(0.01f);
    quant->setErrorBoundMode(ErrorBoundMode::ABS);
    quant->setLinearMode(true);           // signed INT32 codes, no outliers

    auto* lrz = p.addStage<LorenzoStage<int32_t>>();
    lrz->setBlockSize(32);                // cuSZp block-local 1-D delta
    p.connect(lrz, quant, "codes");
    p.finalize();

    CudaBuffer<float> d_in(N);
    CudaStream cs;
    d_in.upload(h_input, cs.stream);
    cudaStreamSynchronize(cs.stream);

    void* d_comp = nullptr; size_t comp_sz = 0;
    p.compress(d_in.void_ptr(), in_bytes, &d_comp, &comp_sz, cs.stream);
    cudaStreamSynchronize(cs.stream);

    void* d_dec = nullptr; size_t dec_sz = 0;
    p.decompress(nullptr, comp_sz, &d_dec, &dec_sz, cs.stream);
    cudaStreamSynchronize(cs.stream);

    ASSERT_EQ(dec_sz, in_bytes);
    std::vector<float> h_recon(N);
    cudaMemcpy(h_recon.data(), d_dec, dec_sz, cudaMemcpyDeviceToHost);

    float max_err = max_abs_error(h_input, h_recon);
    EXPECT_LE(max_err, 0.01f * 1.01f);
}

// ─────────────────────────────────────────────────────────────────────────────
// LZ10-LZ12: ErrorBoundMode REL → PREL aliasing.
//
// LorenzoQuantStage cannot honour an exact per-element relative bound, so REL
// is a deprecated alias for PREL. These lock in that the alias resolves (rather
// than silently quantizing against a mode the stage does not implement) and
// that PREL itself round-trips through the serialized header.
// ─────────────────────────────────────────────────────────────────────────────

TEST(LorenzoQuantEbMode, RelAliasesToPrelOnSetter) {
    LorenzoQuantStage<float, uint16_t> s;
    s.setErrorBoundMode(ErrorBoundMode::REL);
    EXPECT_EQ(s.getErrorBoundMode(), ErrorBoundMode::PREL)
        << "REL must not survive on a stage that cannot implement it";

    // The modes the stage does implement pass through untouched.
    s.setErrorBoundMode(ErrorBoundMode::ABS);
    EXPECT_EQ(s.getErrorBoundMode(), ErrorBoundMode::ABS);
    s.setErrorBoundMode(ErrorBoundMode::NOA);
    EXPECT_EQ(s.getErrorBoundMode(), ErrorBoundMode::NOA);
    s.setErrorBoundMode(ErrorBoundMode::PREL);
    EXPECT_EQ(s.getErrorBoundMode(), ErrorBoundMode::PREL);
}

TEST(LorenzoQuantEbMode, PrelSerializeDeserialize) {
    LorenzoQuantStage<float, uint16_t> src;
    src.setErrorBound(1e-3f);
    src.setErrorBoundMode(ErrorBoundMode::PREL);
    src.setQuantRadius(1024);

    uint8_t cfg[128] = {};
    size_t sz = src.serializeHeader(0, cfg, sizeof(cfg));
    ASSERT_GT(sz, 0u);

    LorenzoQuantStage<float, uint16_t> dst;
    dst.setInverse(true);
    dst.deserializeHeader(cfg, sz);

    EXPECT_EQ(dst.getErrorBoundMode(), ErrorBoundMode::PREL);
}

TEST(LorenzoQuantEbMode, LegacyRelHeaderReadsAsPrel) {
    // A file written before the REL/PREL split stores eb_mode == REL(1) for
    // what was always the approximate mode. Deserialization must map it, not
    // resurrect a guarantee the bytes never carried.
    LorenzoQuantStage<float, uint16_t> src;
    src.setErrorBound(1e-3f);
    src.setErrorBoundMode(ErrorBoundMode::PREL);
    src.setQuantRadius(1024);

    uint8_t cfg[128] = {};
    size_t sz = src.serializeHeader(0, cfg, sizeof(cfg));
    ASSERT_GE(sz, sizeof(LorenzoQuantConfig));

    // Rewrite the eb_mode byte in place to the legacy REL encoding.
    LorenzoQuantConfig raw;
    std::memcpy(&raw, cfg, sizeof(raw));
    raw.eb_mode = static_cast<uint8_t>(ErrorBoundMode::REL);
    std::memcpy(cfg, &raw, sizeof(raw));

    LorenzoQuantStage<float, uint16_t> dst;
    dst.setInverse(true);
    dst.deserializeHeader(cfg, sz);

    EXPECT_EQ(dst.getErrorBoundMode(), ErrorBoundMode::PREL);
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-block mean centering (FSZ adaptive centering) — LZ17-LZ25
// ─────────────────────────────────────────────────────────────────────────────

// A large constant offset is the case centering targets: without it every block's
// first residual is the raw ~100000 value.
static std::vector<int32_t> make_offset_data(size_t n, int32_t offset = 100000) {
    std::vector<int32_t> v(n);
    for (size_t i = 0; i < n; ++i)
        v[i] = offset + static_cast<int32_t>((i * 13) % 61) - 30;
    return v;
}

TEST(LorenzoStage, CenteringRoundTrip1D) {
    const size_t N = 4096;
    auto h_input = make_offset_data(N);

    Pipeline p(N * sizeof(int32_t), MemoryStrategy::PREALLOCATE);
    p.addStage<LorenzoStage<int32_t>>(/*block_size=*/256, /*centering=*/true);
    p.finalize();

    CudaStream cs;
    auto res = pipeline_round_trip<int32_t>(p, h_input, cs.stream);
    EXPECT_EQ(res.max_error, 0.0f);
}

TEST(LorenzoStage, CenteringPartialFinalBlock) {
    const size_t N = 1000;   // ragged final block: 3 full 256s + 232
    auto h_input = make_offset_data(N);

    Pipeline p(N * sizeof(int32_t), MemoryStrategy::PREALLOCATE);
    p.addStage<LorenzoStage<int32_t>>(256, true);
    p.finalize();

    CudaStream cs;
    auto res = pipeline_round_trip<int32_t>(p, h_input, cs.stream);
    EXPECT_EQ(res.max_error, 0.0f);
}

TEST(LorenzoStage, CenteringNonPowerOfTwoBlock) {
    // The mean is a block-wide reduction; a non-power-of-two blockDim exercises
    // the padded-stride path in the reduction loop.
    const size_t N = 3000;
    auto h_input = make_offset_data(N);

    Pipeline p(N * sizeof(int32_t), MemoryStrategy::PREALLOCATE);
    p.addStage<LorenzoStage<int32_t>>(100, true);
    p.finalize();

    CudaStream cs;
    auto res = pipeline_round_trip<int32_t>(p, h_input, cs.stream);
    EXPECT_EQ(res.max_error, 0.0f);
}

TEST(LorenzoStage, CenteringSerializeDeserialize) {
    LorenzoStage<int32_t> original(512, true);

    uint8_t buf[128] = {};
    size_t written = original.serializeHeader(0, buf, sizeof(buf));

    LorenzoStage<int32_t> restored;
    restored.deserializeHeader(buf, written);
    EXPECT_EQ(restored.getBlockSize(), 512u);
    EXPECT_TRUE(restored.getCentering());

    // The centering byte reuses padding that pre-centering writers zeroed, so a
    // genuine legacy header decodes as centering-off. Build one the way an old
    // writer would have: a non-centering config, truncated to 16 bytes.
    LorenzoStage<int32_t> old_writer(32, false);
    uint8_t legacy_buf[128] = {};
    old_writer.serializeHeader(0, legacy_buf, sizeof(legacy_buf));
    EXPECT_EQ(legacy_buf[2], 0u) << "centering byte must be 0 when disabled";

    LorenzoStage<int32_t> legacy;
    legacy.deserializeHeader(legacy_buf, 16);
    EXPECT_FALSE(legacy.getCentering());
    EXPECT_EQ(legacy.getBlockSize(), 0u);   // block_size lives past the legacy 16 B
}

TEST(LorenzoStage, CenteringPortCount) {
    LorenzoStage<int32_t> plain(256, false);
    EXPECT_EQ(plain.getNumOutputs(), 1u);
    EXPECT_EQ(plain.getOutputNames().size(), 1u);

    LorenzoStage<int32_t> centered(256, true);
    EXPECT_EQ(centered.getNumOutputs(), 2u);
    ASSERT_EQ(centered.getOutputNames().size(), 2u);
    EXPECT_EQ(centered.getOutputNames()[1], "means");

    // Inverse: the means port becomes a second *input*.
    centered.setInverse(true);
    EXPECT_EQ(centered.getNumInputs(), 2u);
    EXPECT_EQ(centered.getNumOutputs(), 1u);
}

TEST(LorenzoStage, CenteringRequiresBlockMode) {
    // At construction.
    EXPECT_THROW(LorenzoStage<int32_t>(0, true), std::invalid_argument);

    // And via the setter, which cannot validate eagerly — execute() rejects it.
    const size_t N = 256;
    std::vector<int32_t> h_input(N, 42);
    Pipeline p(N * sizeof(int32_t), MemoryStrategy::PREALLOCATE);
    auto* stage = p.addStage<LorenzoStage<int32_t>>();
    stage->setCentering(true);          // block_size still 0
    p.finalize();

    CudaStream cs;
    EXPECT_THROW(pipeline_round_trip<int32_t>(p, h_input, cs.stream), std::runtime_error);
}

TEST(LorenzoStage, CenteringMatchesUncentered) {
    // Centering is a pure coding choice: it changes the residuals but must not
    // change what comes back out.
    const size_t N = 2048;
    auto h_input = make_offset_data(N);
    CudaStream cs;

    Pipeline a(N * sizeof(int32_t), MemoryStrategy::PREALLOCATE);
    a.addStage<LorenzoStage<int32_t>>(256, false);
    a.finalize();
    auto res_a = pipeline_round_trip<int32_t>(a, h_input, cs.stream);

    Pipeline b(N * sizeof(int32_t), MemoryStrategy::PREALLOCATE);
    b.addStage<LorenzoStage<int32_t>>(256, true);
    b.finalize();
    auto res_b = pipeline_round_trip<int32_t>(b, h_input, cs.stream);

    EXPECT_EQ(res_a.max_error, 0.0f);
    EXPECT_EQ(res_b.max_error, 0.0f);
}

TEST(LorenzoQuantStage, CenteringRoundTrip1D) {
    const size_t N = 8192;
    std::vector<float> h_input(N);
    for (size_t i = 0; i < N; ++i)
        h_input[i] = 300.0f + 0.01f * static_cast<float>((i * 17) % 97);

    LorenzoQuantStage<float, uint16_t>::Config cfg;
    cfg.error_bound = 1e-3f;
    cfg.eb_mode     = ErrorBoundMode::ABS;
    cfg.centering   = true;

    Pipeline p(N * sizeof(float), MemoryStrategy::PREALLOCATE);
    p.addStage<LorenzoQuantStage<float, uint16_t>>(cfg);
    p.finalize();

    CudaStream cs;
    auto res = pipeline_round_trip<float>(p, h_input, cs.stream);
    EXPECT_LE(res.max_error, 1e-3f);
}

TEST(LorenzoQuantStage, CenteringRejectsMultiDim) {
    const size_t nx = 64, ny = 64, N = nx * ny;
    std::vector<float> h_input(N, 300.0f);

    LorenzoQuantStage<float, uint16_t>::Config cfg;
    cfg.error_bound = 1e-3f;
    cfg.eb_mode     = ErrorBoundMode::ABS;
    cfg.centering   = true;
    cfg.dims        = {nx, ny, 1};

    Pipeline p(N * sizeof(float), MemoryStrategy::PREALLOCATE);
    p.setDims(nx, ny);   // finalize() re-pushes pipeline dims into the stage
    p.addStage<LorenzoQuantStage<float, uint16_t>>(cfg);
    p.finalize();

    CudaStream cs;
    EXPECT_THROW(pipeline_round_trip<float>(p, h_input, cs.stream), std::runtime_error);
}

// ─────────────────────────────────────────────────────────────────────────────
// Second-order prediction (LZ2) — LZ26-LZ33
// ─────────────────────────────────────────────────────────────────────────────

// A linear ramp is the case LZ2 targets: first differences are a constant
// non-zero stride, second differences are exactly zero.
static std::vector<int32_t> make_ramp(size_t n, int32_t start = 1000, int32_t stride = 3) {
    std::vector<int32_t> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = start + stride * static_cast<int32_t>(i);
    return v;
}

TEST(LorenzoStage, Order2RoundTrip1D) {
    const size_t N = 4096;
    auto h_input = make_ramp(N);

    Pipeline p(N * sizeof(int32_t), MemoryStrategy::PREALLOCATE);
    p.addStage<LorenzoStage<int32_t>>(/*block_size=*/256, /*centering=*/false, /*order=*/2);
    p.finalize();

    CudaStream cs;
    auto res = pipeline_round_trip<int32_t>(p, h_input, cs.stream);
    EXPECT_EQ(res.max_error, 0.0f);
}

TEST(LorenzoStage, Order2PartialFinalBlock) {
    const size_t N = 1000;   // ragged final block
    auto h_input = make_ramp(N);

    Pipeline p(N * sizeof(int32_t), MemoryStrategy::PREALLOCATE);
    p.addStage<LorenzoStage<int32_t>>(256, false, 2);
    p.finalize();

    CudaStream cs;
    auto res = pipeline_round_trip<int32_t>(p, h_input, cs.stream);
    EXPECT_EQ(res.max_error, 0.0f);
}

TEST(LorenzoStage, Order2NonPowerOfTwoBlock) {
    const size_t N = 3000;
    auto h_input = make_ramp(N);

    Pipeline p(N * sizeof(int32_t), MemoryStrategy::PREALLOCATE);
    p.addStage<LorenzoStage<int32_t>>(100, false, 2);
    p.finalize();

    CudaStream cs;
    auto res = pipeline_round_trip<int32_t>(p, h_input, cs.stream);
    EXPECT_EQ(res.max_error, 0.0f);
}

TEST(LorenzoStage, Order2WithCenteringRoundTrip) {
    // Both features at once: centering subtracts mu from the whole segment and
    // the inverse re-adds it after both scans.
    const size_t N = 4096;
    auto h_input = make_ramp(N, 100000, 7);

    Pipeline p(N * sizeof(int32_t), MemoryStrategy::PREALLOCATE);
    p.addStage<LorenzoStage<int32_t>>(256, /*centering=*/true, /*order=*/2);
    p.finalize();

    CudaStream cs;
    auto res = pipeline_round_trip<int32_t>(p, h_input, cs.stream);
    EXPECT_EQ(res.max_error, 0.0f);
}

TEST(LorenzoStage, Order2BeatsOrder1OnLinearRamp) {
    // The defining property, observed end to end: on a perfect ramp LZ1 leaves a
    // constant non-zero stride in every residual while LZ2 drives them all to
    // zero, so the bit-packed stream is dramatically smaller.
    const size_t N = 8192;
    auto h_input = make_ramp(N, 1000, 3);
    CudaStream cs;

    auto run = [&](uint8_t order) {
        Pipeline p(N * sizeof(int32_t), MemoryStrategy::PREALLOCATE);
        auto* lrz = p.addStage<LorenzoStage<int32_t>>(256, false, order);
        auto* ab  = p.addStage<AdaptiveBitpackStage<int32_t>>();
        ab->setBlockSize(32);
        p.connect(ab, lrz);
        p.finalize();
        auto res = pipeline_round_trip<int32_t>(p, h_input, cs.stream);
        EXPECT_EQ(res.max_error, 0.0f) << "order " << int(order) << " must be lossless";
        return res.compressed_bytes;
    };

    const size_t lz1 = run(1);
    const size_t lz2 = run(2);
    EXPECT_LT(lz2, lz1 / 2) << "LZ2 should crush a linear ramp: lz1=" << lz1
                            << " lz2=" << lz2;
}

TEST(LorenzoStage, Order2SerializeDeserialize) {
    LorenzoStage<int32_t> original(512, false, 2);

    uint8_t buf[128] = {};
    size_t written = original.serializeHeader(0, buf, sizeof(buf));

    LorenzoStage<int32_t> restored;
    restored.deserializeHeader(buf, written);
    EXPECT_EQ(restored.getOrder(), 2u);
    EXPECT_EQ(restored.getBlockSize(), 512u);

    // Pre-LZ2 writers left this byte zero; 0 must read as first order.
    LorenzoStage<int32_t> old_writer(32, false, 1);
    uint8_t legacy_buf[128] = {};
    old_writer.serializeHeader(0, legacy_buf, sizeof(legacy_buf));
    LorenzoStage<int32_t> legacy;
    legacy.deserializeHeader(legacy_buf, written);
    EXPECT_EQ(legacy.getOrder(), 1u);
}

TEST(LorenzoStage, Order2RejectsBadValues) {
    EXPECT_THROW(LorenzoStage<int32_t>(256, false, 3), std::invalid_argument);
    EXPECT_THROW(LorenzoStage<int32_t>(256, false, 0), std::invalid_argument);
    EXPECT_THROW(LorenzoStage<int32_t>(0, false, 2), std::invalid_argument);  // needs blocks

    LorenzoStage<int32_t> stage;
    EXPECT_THROW(stage.setOrder(3), std::invalid_argument);
    EXPECT_NO_THROW(stage.setOrder(2));
}

TEST(LorenzoStage, Order2RequiresBlockMode) {
    const size_t N = 256;
    auto h_input = make_ramp(N);

    Pipeline p(N * sizeof(int32_t), MemoryStrategy::PREALLOCATE);
    auto* stage = p.addStage<LorenzoStage<int32_t>>();
    stage->setOrder(2);          // block_size still 0
    p.finalize();

    CudaStream cs;
    EXPECT_THROW(pipeline_round_trip<int32_t>(p, h_input, cs.stream), std::runtime_error);
}

// ─────────────────────────────────────────────────────────────────────────────
// Segmented inverse scan — LZ34-LZ37
//
// The block-mode inverse no longer launches blockDim == block_size; one CTA now
// owns a segment with several elements per thread. These cover the shape
// selection (Seq = 1/2/4), the non-warp-multiple fallback, and the interaction
// with both prediction orders and centering.
// ─────────────────────────────────────────────────────────────────────────────

TEST(LorenzoStage, SegmentedScanAllBlockSizes) {
    const size_t N = 16384;
    auto h_input = make_offset_data(N);
    CudaStream cs;
    // 32 keeps the warp fast path; 64..1024 route through the segmented kernel
    // with Seq = 1, 1, 1, 1, 4, 4 respectively.
    for (uint32_t bs : {32u, 64u, 128u, 256u, 512u, 1024u}) {
        Pipeline p(N * sizeof(int32_t), MemoryStrategy::PREALLOCATE);
        p.addStage<LorenzoStage<int32_t>>(bs);
        p.finalize();
        auto res = pipeline_round_trip<int32_t>(p, h_input, cs.stream);
        EXPECT_EQ(res.max_error, 0.0f) << "block_size = " << bs;
    }
}

TEST(LorenzoStage, SegmentedScanNonWarpMultipleFallback) {
    // 100 and 300 are not multiples of 32, so the warp-shuffle scan cannot be
    // used and the barrier-based fallback must take over.
    const size_t N = 9000;
    auto h_input = make_offset_data(N);
    CudaStream cs;
    for (uint32_t bs : {100u, 300u, 33u}) {
        Pipeline p(N * sizeof(int32_t), MemoryStrategy::PREALLOCATE);
        p.addStage<LorenzoStage<int32_t>>(bs);
        p.finalize();
        auto res = pipeline_round_trip<int32_t>(p, h_input, cs.stream);
        EXPECT_EQ(res.max_error, 0.0f) << "block_size = " << bs;
    }
}

TEST(LorenzoStage, SegmentedScanWithCenteringAndOrder) {
    // All four (order, centering) combinations through the shared scan, at a
    // block size that uses Seq = 4.
    const size_t N = 16384;
    auto h_input = make_offset_data(N);
    CudaStream cs;
    for (uint8_t ord : {uint8_t{1}, uint8_t{2}}) {
        for (bool cent : {false, true}) {
            Pipeline p(N * sizeof(int32_t), MemoryStrategy::PREALLOCATE);
            p.addStage<LorenzoStage<int32_t>>(1024, cent, ord);
            p.finalize();
            auto res = pipeline_round_trip<int32_t>(p, h_input, cs.stream);
            EXPECT_EQ(res.max_error, 0.0f)
                << "order = " << int(ord) << " centering = " << cent;
        }
    }
}

TEST(LorenzoStage, SegmentedScanRaggedTail) {
    // Element count not a multiple of the segment, with a segment that splits
    // unevenly across threads.
    const size_t N = 1024 * 3 + 37;
    auto h_input = make_offset_data(N);
    CudaStream cs;
    for (uint8_t ord : {uint8_t{1}, uint8_t{2}}) {
        Pipeline p(N * sizeof(int32_t), MemoryStrategy::PREALLOCATE);
        p.addStage<LorenzoStage<int32_t>>(1024, true, ord);
        p.finalize();
        auto res = pipeline_round_trip<int32_t>(p, h_input, cs.stream);
        EXPECT_EQ(res.max_error, 0.0f) << "order = " << int(ord);
    }
}
