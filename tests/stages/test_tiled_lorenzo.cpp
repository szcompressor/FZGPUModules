/**
 * tests/stages/test_tiled_lorenzo.cpp
 *
 * GPU unit tests for TiledLorenzoStage<T> — dimension-aware (tiled separable)
 * Lorenzo predictor (cuSZp3 delta). Signed int16/int32. Forward emits tile-major
 * deltas (edge tiles zero-padded); inverse un-tiles to natural row-major order.
 * Graph-compatible (pure kernels, deterministic sizes).
 *
 *   TL1  RoundTrip2D_Int32        — 8x8 default tiles, dims multiple of 8, exact
 *   TL2  RoundTrip2D_Int16        — int16 2-D round-trip, exact
 *   TL3  RoundTrip3D_Int32        — 4x4x4 default tiles, exact
 *   TL4  PartialEdgeTiles2D       — dims NOT multiples of 8, exact
 *   TL5  PartialEdgeTiles3D       — dims NOT multiples of 4, exact
 *   TL6  CustomTileShape          — setTileShape(16,16), exact
 *   TL7  ConstantField            — constant input round-trips exactly
 *   TL8  SetTileShapeRejects      — extent >255 and product >1024 throw
 *   TL9  SerializeDeserialize     — dims + tile shape survive the header
 *   TL10 PortAndTypeContract      — 1 in/out; signed type both directions; graph
 *   TL11 StageTypeId              — getStageTypeId() == TILED_LORENZO
 *   TL12 CuSZp3PlainPipeline      — Quantizer(linear)→TiledLorenzo→AdaptiveBitpack(64)
 *   TL13 CuSZp3OutlierPipeline    — same + AdaptiveBitpack outlier selection
 *   TL14 FileRoundTrip            — full pipeline through writeToFile/decompressFromFile
 *   TL15 PhasedScan_TileStampedSeeds — one-block-per-tile inverse: every tile's
 *        Z/Y/X chain seed is a distinguishable value, so a phase-ordering bug
 *        (e.g. reading shared mem before __syncthreads()) produces a visibly
 *        wrong value instead of silently matching on a smooth/low-entropy field.
 */

#include <gtest/gtest.h>
#include "helpers/fz_test_utils.h"
#include "helpers/stage_harness.h"
#include "fzgpumodules.h"

#include <cstdint>
#include <cmath>
#include <vector>

using namespace fz;
using namespace fz_test;

namespace {

// Standalone TiledLorenzo round-trip: forward (tile-major deltas) -> inverse
// (natural order). Verifies exact reconstruction for the given dims/tile.
template<typename T>
void expect_exact_tiled_round_trip(const std::vector<T>& h_in,
                                   size_t nx, size_t ny, size_t nz,
                                   uint32_t tx = 0, uint32_t ty = 0,
                                   uint32_t tz = 0) {
    const size_t in_bytes = h_in.size() * sizeof(T);
    Pipeline p(in_bytes, MemoryStrategy::PREALLOCATE);
    p.setDims(nx, ny, nz);
    auto* s = p.addStage<TiledLorenzoStage<T>>();
    if (tx || ty || tz) s->setTileShape(tx ? tx : 1, ty ? ty : 1, tz ? tz : 1);
    p.finalize();

    CudaStream cs;
    auto res = pipeline_round_trip<T>(p, h_in, cs.stream);

    ASSERT_EQ(res.data.size(), h_in.size());
    for (size_t i = 0; i < h_in.size(); ++i)
        EXPECT_EQ(res.data[i], h_in[i]) << "mismatch at i=" << i;
}

// A smooth-ish integer field with structure in every dimension.
template<typename T>
std::vector<T> make_field(size_t nx, size_t ny, size_t nz) {
    std::vector<T> v(nx * ny * nz);
    for (size_t z = 0; z < nz; ++z)
        for (size_t y = 0; y < ny; ++y)
            for (size_t x = 0; x < nx; ++x) {
                long val = (long)(x * 2 + y * 3 + z * 5) % 211 - 105;
                v[(z * ny + y) * nx + x] = static_cast<T>(val);
            }
    return v;
}

}  // namespace

// ── TL1 ─────────────────────────────────────────────────────────────────────
TEST(TiledLorenzoStage, RoundTrip2D_Int32) {
    const size_t NX = 64, NY = 48;  // multiples of 8
    auto h = make_field<int32_t>(NX, NY, 1);
    expect_exact_tiled_round_trip<int32_t>(h, NX, NY, 1);
}

// ── TL2 ─────────────────────────────────────────────────────────────────────
TEST(TiledLorenzoStage, RoundTrip2D_Int16) {
    const size_t NX = 80, NY = 24;
    auto h = make_field<int16_t>(NX, NY, 1);
    expect_exact_tiled_round_trip<int16_t>(h, NX, NY, 1);
}

// ── TL3 ─────────────────────────────────────────────────────────────────────
TEST(TiledLorenzoStage, RoundTrip3D_Int32) {
    const size_t NX = 16, NY = 12, NZ = 8;  // multiples of 4
    auto h = make_field<int32_t>(NX, NY, NZ);
    expect_exact_tiled_round_trip<int32_t>(h, NX, NY, NZ);
}

// ── TL4 ─────────────────────────────────────────────────────────────────────
TEST(TiledLorenzoStage, PartialEdgeTiles2D) {
    const size_t NX = 67, NY = 41;  // NOT multiples of 8 — exercises padding
    auto h = make_field<int32_t>(NX, NY, 1);
    expect_exact_tiled_round_trip<int32_t>(h, NX, NY, 1);
}

// ── TL5 ─────────────────────────────────────────────────────────────────────
TEST(TiledLorenzoStage, PartialEdgeTiles3D) {
    const size_t NX = 13, NY = 7, NZ = 5;  // NOT multiples of 4
    auto h = make_field<int32_t>(NX, NY, NZ);
    expect_exact_tiled_round_trip<int32_t>(h, NX, NY, NZ);
}

// ── TL6 ─────────────────────────────────────────────────────────────────────
TEST(TiledLorenzoStage, CustomTileShape) {
    const size_t NX = 50, NY = 50;
    auto h = make_field<int32_t>(NX, NY, 1);
    expect_exact_tiled_round_trip<int32_t>(h, NX, NY, 1, 16, 16, 1);
}

// ── TL7 ─────────────────────────────────────────────────────────────────────
TEST(TiledLorenzoStage, ConstantField) {
    const size_t NX = 40, NY = 33;
    std::vector<int32_t> h(NX * NY, 7);
    expect_exact_tiled_round_trip<int32_t>(h, NX, NY, 1);
}

// ── TL8 ─────────────────────────────────────────────────────────────────────
TEST(TiledLorenzoStage, SetTileShapeRejects) {
    TiledLorenzoStage<int32_t> s;
    EXPECT_THROW(s.setTileShape(256, 1, 1), std::invalid_argument);   // extent > 255
    EXPECT_THROW(s.setTileShape(64, 64, 1), std::invalid_argument);   // product 4096 > 1024
    EXPECT_NO_THROW(s.setTileShape(8, 8, 1));
    EXPECT_NO_THROW(s.setTileShape(4, 4, 4));
}

// ── TL9 ─────────────────────────────────────────────────────────────────────
TEST(TiledLorenzoStage, SerializeDeserialize) {
    TiledLorenzoStage<int32_t> original;
    original.setDims(67, 41, 1);
    original.setTileShape(16, 16, 1);

    uint8_t buf[128] = {};
    size_t n = original.serializeHeader(0, buf, sizeof(buf));
    ASSERT_GT(n, 0u);

    TiledLorenzoStage<int32_t> restored;
    restored.deserializeHeader(buf, n);
    EXPECT_EQ(restored.getDims()[0], 67u);
    EXPECT_EQ(restored.getDims()[1], 41u);
    auto t = restored.getTileShape();
    EXPECT_EQ(t[0], 16u);
    EXPECT_EQ(t[1], 16u);
    EXPECT_EQ(t[2], 1u);
}

// ── TL10 ────────────────────────────────────────────────────────────────────
TEST(TiledLorenzoStage, PortAndTypeContract) {
    TiledLorenzoStage<int32_t> s;
    EXPECT_EQ(s.getNumInputs(), 1u);
    EXPECT_EQ(s.getNumOutputs(), 1u);
    EXPECT_EQ(s.getInputDataType(0), static_cast<uint8_t>(DataType::INT32));
    EXPECT_EQ(s.getOutputDataType(0), static_cast<uint8_t>(DataType::INT32));
    EXPECT_TRUE(s.isGraphCompatible());

    TiledLorenzoStage<int16_t> s16;
    EXPECT_EQ(s16.getInputDataType(0), static_cast<uint8_t>(DataType::INT16));
}

// ── TL11 ────────────────────────────────────────────────────────────────────
TEST(TiledLorenzoStage, StageTypeId) {
    EXPECT_EQ(TiledLorenzoStage<int32_t>().getStageTypeId(),
              static_cast<uint16_t>(StageType::TILED_LORENZO));
}

// ── TL12 ────────────────────────────────────────────────────────────────────
// Faithful cuSZp3 plain pipeline on a smooth 2-D float field.
TEST(TiledLorenzoStage, CuSZp3PlainPipeline) {
    const size_t NX = 256, NY = 256, N = NX * NY;
    const size_t in_bytes = N * sizeof(float);
    const float  EB = 1e-2f;

    std::vector<float> h_input(N);
    for (size_t y = 0; y < NY; ++y)
        for (size_t x = 0; x < NX; ++x)
            h_input[y * NX + x] =
                std::sin(x * 0.05f) * std::cos(y * 0.04f) * 10.0f;

    Pipeline p(in_bytes, MemoryStrategy::PREALLOCATE);
    p.setDims(NX, NY);

    auto* quant = p.addStage<QuantizerStage<float, uint32_t>>();
    quant->setErrorBound(EB);
    quant->setErrorBoundMode(ErrorBoundMode::ABS);
    quant->setLinearMode(true);

    auto* tl = p.addStage<TiledLorenzoStage<int32_t>>();
    tl->setTileShape(8, 8);
    p.connect(tl, quant, "codes");

    auto* ab = p.addStage<AdaptiveBitpackStage<int32_t>>();
    ab->setBlockSize(64);  // = tile_elems, so one block per tile
    p.connect(ab, tl);

    p.finalize();

    CudaBuffer<float> d_in(N);
    CudaStream cs;
    d_in.upload(h_input, cs.stream);
    cudaStreamSynchronize(cs.stream);

    void* d_comp = nullptr; size_t comp_sz = 0;
    p.compress(d_in.void_ptr(), in_bytes, &d_comp, &comp_sz, cs.stream);
    cudaStreamSynchronize(cs.stream);
    ASSERT_GT(comp_sz, 0u);
    EXPECT_LT(comp_sz, in_bytes);  // smooth data must compress

    void* d_dec = nullptr; size_t dec_sz = 0;
    p.decompress(nullptr, comp_sz, &d_dec, &dec_sz, cs.stream);
    cudaStreamSynchronize(cs.stream);
    ASSERT_EQ(dec_sz, in_bytes);

    std::vector<float> h_recon(N);
    cudaMemcpy(h_recon.data(), d_dec, dec_sz, cudaMemcpyDeviceToHost);
    EXPECT_LE(max_abs_error(h_input, h_recon), EB * 1.01f);
}

// ── TL13 ────────────────────────────────────────────────────────────────────
// cuSZp3 outlier pipeline: tile-origin elements are delta-vs-0 (full magnitude),
// so per-block outlier selection should keep round-trip exact within eb.
TEST(TiledLorenzoStage, CuSZp3OutlierPipeline) {
    const size_t NX = 200, NY = 150, N = NX * NY;
    const size_t in_bytes = N * sizeof(float);
    const float  EB = 5e-3f;

    std::vector<float> h_input(N);
    for (size_t y = 0; y < NY; ++y)
        for (size_t x = 0; x < NX; ++x)
            h_input[y * NX + x] = std::sin((x + y) * 0.03f) * 20.0f + x * 0.01f;

    Pipeline p(in_bytes, MemoryStrategy::PREALLOCATE);
    p.setDims(NX, NY);

    auto* quant = p.addStage<QuantizerStage<float, uint32_t>>();
    quant->setErrorBound(EB);
    quant->setErrorBoundMode(ErrorBoundMode::ABS);
    quant->setLinearMode(true);

    auto* tl = p.addStage<TiledLorenzoStage<int32_t>>();
    tl->setTileShape(8, 8);
    p.connect(tl, quant, "codes");

    auto* ab = p.addStage<AdaptiveBitpackStage<int32_t>>();
    ab->setBlockSize(64);
    ab->setOutlierSelection(true);
    p.connect(ab, tl);

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
    EXPECT_LE(max_abs_error(h_input, h_recon), EB * 1.01f);
}

// ── TL14 ────────────────────────────────────────────────────────────────────
// Full serialization path (StageFactory + FZM header) for the standalone stage.
TEST(TiledLorenzoStage, FileRoundTrip) {
    const size_t NX = 67, NY = 41;
    auto h = make_field<int32_t>(NX, NY, 1);
    const size_t in_bytes = h.size() * sizeof(int32_t);

    Pipeline p(in_bytes, MemoryStrategy::MINIMAL);
    p.setDims(NX, NY);
    auto* s = p.addStage<TiledLorenzoStage<int32_t>>();
    s->setTileShape(8, 8);
    p.finalize();

    CudaStream cs;
    const std::string path = "/tmp/test_tiled_lorenzo.fzm";
    auto res = pipeline_file_round_trip<int32_t>(p, h, cs.stream, path);
    ASSERT_EQ(res.data.size(), h.size());
    for (size_t i = 0; i < h.size(); ++i)
        EXPECT_EQ(res.data[i], h[i]) << "mismatch at i=" << i;
    std::remove(path.c_str());
}

// ── TL15 ────────────────────────────────────────────────────────────────────
// The phased inverse kernel (one CUDA block per tile) computes Z-chain (phase
// 1) -> per-lz Y-chains (phase 2) -> per-(ly,lz) X-chains (phase 3) through
// shared memory with two __syncthreads() barriers. A field with a distinct,
// large "tile stamp" baked into every value makes any phase-ordering bug
// (reading a seed before it's written, or from the wrong tile/lane) produce an
// immediately-visible wrong value, rather than one that happens to coincide by
// chance on a smooth low-entropy field like make_field() above.
TEST(TiledLorenzoStage, PhasedScan_TileStampedSeeds) {
    const size_t NX = 16, NY = 12, NZ = 8;   // 4 x 3 x 2 tiles at 4x4x4
    const uint32_t tx = 4, ty = 4, tz = 4;
    const uint32_t ntx = static_cast<uint32_t>((NX + tx - 1) / tx);
    const uint32_t nty = static_cast<uint32_t>((NY + ty - 1) / ty);

    std::vector<int32_t> h(NX * NY * NZ);
    for (size_t z = 0; z < NZ; ++z)
        for (size_t y = 0; y < NY; ++y)
            for (size_t x = 0; x < NX; ++x) {
                const uint32_t tix = static_cast<uint32_t>(x / tx);
                const uint32_t tiy = static_cast<uint32_t>(y / ty);
                const uint32_t tiz = static_cast<uint32_t>(z / tz);
                const uint32_t tile_id = (tiz * nty + tiy) * ntx + tix;
                // Every tile gets a unique 10000-wide stamp band; within a tile,
                // the low-order digits still vary with (x,y,z) so every one of
                // the tz/ty*tz/tx-length chains sees non-constant deltas.
                const long val = static_cast<long>(tile_id) * 10000
                                + static_cast<long>(z % tz) * 100
                                + static_cast<long>(y % ty) * 10
                                + static_cast<long>(x % tx);
                h[(z * NY + y) * NX + x] = static_cast<int32_t>(val);
            }

    expect_exact_tiled_round_trip<int32_t>(h, NX, NY, NZ, tx, ty, tz);
}
