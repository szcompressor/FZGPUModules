/**
 * tests/stages/test_cuszp_block_sizes.cpp
 *
 * Verifies that the block / tile size is a freely tunable knob for the modular
 * cuSZp-family pipelines — a user can pick a value other than the cuSZp default
 * (32 for the 1-D pipelines, 8×8 / 4×4×4 tiles for cuSZp3) and still get a
 * correct round-trip. Also checks that the predictor block and the coder block
 * need not match (they are independent for correctness; matching them only
 * matters for faithful compression ratio).
 *
 *   BS1  CuSZp_PlainBlockSweep        — Lorenzo+AB block ∈ {8,16,32,64,128,256,1024}
 *   BS2  CuSZp2_OutlierBlockSweep     — same sweep with outlier selection on
 *   BS3  CuSZp_MismatchedBlocks       — Lorenzo block ≠ AdaptiveBitpack block
 *   BS4  CuSZp3_TileSweep             — TiledLorenzo tiles + AB block = tile_elems
 *   BS5  CuSZp3_CoderBlockNeNotTile   — AB block ≠ tile_elems still round-trips
 *   BS6  CuSZp3_TileSweep3D           — 3-D tile shapes
 */

#include <gtest/gtest.h>
#include "helpers/fz_test_utils.h"
#include "helpers/stage_harness.h"
#include "fzgpumodules.h"

#include <cmath>
#include <cstdint>
#include <vector>

using namespace fz;
using namespace fz_test;

namespace {

// cuSZp / cuSZp2 1-D pipeline: Quantizer(linear) → Lorenzo(block) →
// AdaptiveBitpack(block). Returns compressed bytes; asserts round-trip within eb.
size_t run_cuszp_1d(size_t N, float eb, uint32_t lrz_block, uint32_t ab_block,
                    bool outlier) {
    const size_t in_bytes = N * sizeof(float);
    auto h = make_smooth_data<float>(N);

    Pipeline p(in_bytes, MemoryStrategy::PREALLOCATE);
    auto* q = p.addStage<QuantizerStage<float, uint32_t>>();
    q->setErrorBound(eb);
    q->setErrorBoundMode(ErrorBoundMode::ABS);
    q->setLinearMode(true);

    auto* lrz = p.addStage<LorenzoStage<int32_t>>();
    lrz->setBlockSize(lrz_block);
    p.connect(lrz, q, "codes");

    auto* ab = p.addStage<AdaptiveBitpackStage<int32_t>>();
    ab->setBlockSize(ab_block);
    ab->setOutlierSelection(outlier);
    p.connect(ab, lrz);
    p.finalize();

    CudaBuffer<float> d_in(N);
    CudaStream cs;
    d_in.upload(h, cs.stream);
    cudaStreamSynchronize(cs.stream);

    void* d_comp = nullptr; size_t comp_sz = 0;
    p.compress(d_in.void_ptr(), in_bytes, &d_comp, &comp_sz, cs.stream);
    void* d_dec = nullptr; size_t dec_sz = 0;
    p.decompress(nullptr, comp_sz, &d_dec, &dec_sz, cs.stream);
    cudaStreamSynchronize(cs.stream);

    EXPECT_EQ(dec_sz, in_bytes);
    std::vector<float> rec(N);
    cudaMemcpy(rec.data(), d_dec, dec_sz, cudaMemcpyDeviceToHost);
    EXPECT_LE(max_abs_error(h, rec), eb * 1.01f);
    return comp_sz;
}

// cuSZp3 2-D/3-D pipeline: Quantizer(linear) → TiledLorenzo(tile) →
// AdaptiveBitpack(ab_block). Returns compressed bytes; asserts round-trip ≤ eb.
size_t run_cuszp3(size_t nx, size_t ny, size_t nz, float eb,
                  uint32_t tx, uint32_t ty, uint32_t tz,
                  uint32_t ab_block, bool outlier) {
    const size_t N = nx * ny * nz;
    const size_t in_bytes = N * sizeof(float);

    std::vector<float> h(N);
    for (size_t z = 0; z < nz; ++z)
        for (size_t y = 0; y < ny; ++y)
            for (size_t x = 0; x < nx; ++x)
                h[(z * ny + y) * nx + x] =
                    std::sin(x * 0.05f) * std::cos(y * 0.04f)
                    + std::sin(z * 0.03f) * 5.0f;

    Pipeline p(in_bytes, MemoryStrategy::PREALLOCATE);
    p.setDims(nx, ny, nz);

    auto* q = p.addStage<QuantizerStage<float, uint32_t>>();
    q->setErrorBound(eb);
    q->setErrorBoundMode(ErrorBoundMode::ABS);
    q->setLinearMode(true);

    auto* tl = p.addStage<TiledLorenzoStage<int32_t>>();
    tl->setTileShape(tx, ty, tz);
    p.connect(tl, q, "codes");

    auto* ab = p.addStage<AdaptiveBitpackStage<int32_t>>();
    ab->setBlockSize(ab_block);
    ab->setOutlierSelection(outlier);
    p.connect(ab, tl);
    p.finalize();

    CudaBuffer<float> d_in(N);
    CudaStream cs;
    d_in.upload(h, cs.stream);
    cudaStreamSynchronize(cs.stream);

    void* d_comp = nullptr; size_t comp_sz = 0;
    p.compress(d_in.void_ptr(), in_bytes, &d_comp, &comp_sz, cs.stream);
    void* d_dec = nullptr; size_t dec_sz = 0;
    p.decompress(nullptr, comp_sz, &d_dec, &dec_sz, cs.stream);
    cudaStreamSynchronize(cs.stream);

    EXPECT_EQ(dec_sz, in_bytes);
    std::vector<float> rec(N);
    cudaMemcpy(rec.data(), d_dec, dec_sz, cudaMemcpyDeviceToHost);
    EXPECT_LE(max_abs_error(h, rec), eb * 1.01f);
    return comp_sz;
}

}  // namespace

// ── BS1 ─────────────────────────────────────────────────────────────────────
// A user can choose any block size for the cuSZp plain pipeline, not just 32.
TEST(CuSZpBlockSizes, CuSZp_PlainBlockSweep) {
    const size_t N = 1 << 14;  // 16384
    const float  eb = 1e-2f;
    for (uint32_t block : {8u, 16u, 32u, 64u, 128u, 256u, 1024u}) {
        size_t comp = run_cuszp_1d(N, eb, block, block, /*outlier=*/false);
        EXPECT_GT(comp, 0u) << "block=" << block;
        EXPECT_LT(comp, N * sizeof(float)) << "block=" << block;  // must compress
    }
}

// ── BS2 ─────────────────────────────────────────────────────────────────────
TEST(CuSZpBlockSizes, CuSZp2_OutlierBlockSweep) {
    const size_t N = 1 << 14;
    const float  eb = 1e-2f;
    for (uint32_t block : {8u, 16u, 32u, 64u, 128u, 256u}) {
        size_t comp = run_cuszp_1d(N, eb, block, block, /*outlier=*/true);
        EXPECT_GT(comp, 0u) << "block=" << block;
    }
}

// ── BS3 ─────────────────────────────────────────────────────────────────────
// The Lorenzo (predictor) block and the AdaptiveBitpack (coder) block are
// independent for correctness — only matching them matters for faithful CR.
TEST(CuSZpBlockSizes, CuSZp_MismatchedBlocks) {
    const size_t N = 1 << 14;
    const float  eb = 1e-2f;
    EXPECT_NO_FATAL_FAILURE(run_cuszp_1d(N, eb, /*lrz=*/32, /*ab=*/64,  false));
    EXPECT_NO_FATAL_FAILURE(run_cuszp_1d(N, eb, /*lrz=*/64, /*ab=*/32,  false));
    EXPECT_NO_FATAL_FAILURE(run_cuszp_1d(N, eb, /*lrz=*/128,/*ab=*/16,  false));
}

// ── BS4 ─────────────────────────────────────────────────────────────────────
// cuSZp3 2-D: user can resize the tile, as long as AdaptiveBitpack's block
// equals tile_elems so coder blocks stay aligned with tiles.
TEST(CuSZpBlockSizes, CuSZp3_TileSweep) {
    const size_t NX = 256, NY = 192;
    const float  eb = 1e-2f;
    struct TileCase { uint32_t tx, ty; };
    for (TileCase c : {TileCase{8, 8}, TileCase{16, 16}, TileCase{4, 4},
                       TileCase{32, 8}, TileCase{8, 16}, TileCase{16, 8}}) {
        uint32_t te = c.tx * c.ty;  // = AdaptiveBitpack block size
        size_t comp = run_cuszp3(NX, NY, 1, eb, c.tx, c.ty, 1, te, false);
        EXPECT_GT(comp, 0u) << "tile=" << c.tx << "x" << c.ty;
        EXPECT_LT(comp, NX * NY * sizeof(float)) << "tile=" << c.tx << "x" << c.ty;
    }
}

// ── BS5 ─────────────────────────────────────────────────────────────────────
// Even when the coder block ≠ tile_elems the pipeline is still lossless within
// eb (it just won't reproduce cuSZp3's exact per-tile CR).
TEST(CuSZpBlockSizes, CuSZp3_CoderBlockNeedNotMatchTile) {
    const size_t NX = 256, NY = 192;
    const float  eb = 1e-2f;
    EXPECT_NO_FATAL_FAILURE(run_cuszp3(NX, NY, 1, eb, 8, 8, 1, /*ab=*/32,  false));
    EXPECT_NO_FATAL_FAILURE(run_cuszp3(NX, NY, 1, eb, 8, 8, 1, /*ab=*/128, false));
}

// ── BS6 ─────────────────────────────────────────────────────────────────────
TEST(CuSZpBlockSizes, CuSZp3_TileSweep3D) {
    const size_t NX = 64, NY = 48, NZ = 32;
    const float  eb = 1e-2f;
    struct TileCase { uint32_t tx, ty, tz; };
    for (TileCase c : {TileCase{4, 4, 4}, TileCase{8, 8, 4}, TileCase{4, 4, 8},
                       TileCase{8, 4, 2}}) {
        uint32_t te = c.tx * c.ty * c.tz;
        size_t comp = run_cuszp3(NX, NY, NZ, eb, c.tx, c.ty, c.tz, te,
                                 /*outlier=*/true);
        EXPECT_GT(comp, 0u)
            << "tile=" << c.tx << "x" << c.ty << "x" << c.tz;
    }
}
