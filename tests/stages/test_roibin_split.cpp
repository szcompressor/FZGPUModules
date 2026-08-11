/**
 * tests/stages/test_roibin_split.cpp
 *
 * GPU unit + integration tests for ROIBinSplitStage — 1→3 ROI/background split
 * (forward), 3→1 reconstruction (inverse).
 *
 *   RB1  ROIBinSplit/ExtractScatterRoundTrip   — bin=1 reconstructs the field exactly
 *   RB2  ROIBinSplit/BinnedBackgroundIsMean    — bin=2 background equals the 2x2 means
 *   RB3  ROIBinSplit/RoiSurvivesBinning        — ROI pixels are exact even when bin>1
 *   RB4  ROIBinSplit/OverlappingBoxesAreIdempotent — touching peaks still round-trip
 *   RB5  ROIBinSplit/EdgeBoxesAreClamped       — a peak in the corner does not read OOB
 *   RB6  ROIBinSplit/PortCounts                — 1→3 forward, 3→1 inverse
 *   RB7  ROIBinSplit/HeaderSerialization       — geometry survives serialize/deserialize
 *   RB8  ROIBinSplit/ExactOutputSizes          — estimate == actual, both ports
 */

#include <gtest/gtest.h>
#include "helpers/fz_test_utils.h"
#include "helpers/stage_harness.h"
#include "structural/roibin_split/roibin_split_stage.h"
#include "fzgpumodules.h"

#include <cstdint>
#include <algorithm>
#include <random>
#include <vector>

using namespace fz;
using namespace fz_test;

namespace {

struct SplitResult {
    std::vector<float> roi;
    std::vector<float> bg;
    std::vector<float> recon;
};

// Run forward then inverse through the stage and return all three streams.
SplitResult run_split(const std::vector<float>& field,
                      size_t nx, size_t ny, size_t nz,
                      const std::vector<RoiPeak>& peaks,
                      uint32_t hw, uint32_t bin)
{
    CudaStream cs;
    auto pool = make_test_pool(field.size() * sizeof(float) * 8 + (1u << 20));

    ROIBinSplitStage<float> enc;
    enc.setDims(nx, ny, nz);
    enc.setRoiHalfWidth(hw);
    enc.setBinFactor(bin);
    enc.setPeaks(peaks);

    const size_t n_roi = enc.getRoiCount();
    const size_t n_bg = enc.getBgCount();

    CudaBuffer<float> d_in(field.size());
    d_in.upload(field, cs.stream);
    CudaBuffer<float> d_roi(n_roi ? n_roi : 1);
    CudaBuffer<float> d_bg(n_bg);
    CudaBuffer<uint8_t> d_pk(peaks.size() * sizeof(RoiPeak) + 1);
    cudaStreamSynchronize(cs.stream);

    enc.onFinalize(field.size() * sizeof(float), pool.get());

    std::vector<void*> in = {d_in.void_ptr()};
    std::vector<void*> out = {d_roi.void_ptr(), d_bg.void_ptr(), d_pk.void_ptr()};
    std::vector<size_t> sz = {field.size() * sizeof(float)};
    enc.execute(cs.stream, pool.get(), in, out, sz);
    cudaStreamSynchronize(cs.stream);

    SplitResult r;
    r.roi = d_roi.download(cs.stream);
    r.bg = d_bg.download(cs.stream);
    if (n_roi == 0) r.roi.clear();

    // ── Inverse: the peak table comes back in on the third port, exactly as the
    //    DAG feeds it on the decompress path. ──
    ROIBinSplitStage<float> dec;
    dec.setDims(nx, ny, nz);
    dec.setRoiHalfWidth(hw);
    dec.setBinFactor(bin);
    dec.setPeaks(peaks);
    dec.setInverse(true);

    CudaBuffer<float> d_out(field.size());
    std::vector<void*> iin = {d_roi.void_ptr(), d_bg.void_ptr(), d_pk.void_ptr()};
    std::vector<void*> iout = {d_out.void_ptr()};
    std::vector<size_t> isz = {n_roi * sizeof(float), n_bg * sizeof(float),
                               peaks.size() * sizeof(RoiPeak)};
    dec.execute(cs.stream, pool.get(), iin, iout, isz);
    cudaStreamSynchronize(cs.stream);

    r.recon = d_out.download(cs.stream);
    return r;
}

std::vector<float> ramp_field(size_t nx, size_t ny, size_t nz) {
    std::vector<float> f(nx * ny * nz);
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> d(-100.f, 100.f);
    for (size_t i = 0; i < f.size(); i++) f[i] = static_cast<float>(i % 977) + d(rng);
    return f;
}

}  // namespace

// RB1 — with no binning the split is fully lossless: unbin is a copy and the ROI
// scatter writes back exactly what it gathered.
TEST(ROIBinSplit, ExtractScatterRoundTrip) {
    const size_t nx = 64, ny = 48, nz = 2;
    auto field = ramp_field(nx, ny, nz);
    std::vector<RoiPeak> peaks = {{0, 10, 10}, {0, 40, 30}, {1, 20, 20}};

    auto r = run_split(field, nx, ny, nz, peaks, /*hw=*/3, /*bin=*/1);
    ASSERT_EQ(r.recon.size(), field.size());
    for (size_t i = 0; i < field.size(); i++)
        ASSERT_FLOAT_EQ(r.recon[i], field[i]) << "at " << i;
}

// RB2 — the background stream really is the 2x2 block mean, not a subsample.
TEST(ROIBinSplit, BinnedBackgroundIsMean) {
    const size_t nx = 64, ny = 48, nz = 1;
    auto field = ramp_field(nx, ny, nz);
    std::vector<RoiPeak> peaks = {{0, 5, 5}};

    auto r = run_split(field, nx, ny, nz, peaks, /*hw=*/2, /*bin=*/2);
    ASSERT_EQ(r.bg.size(), (nx / 2) * (ny / 2));
    for (size_t by = 0; by < ny / 2; by++) {
        for (size_t bx = 0; bx < nx / 2; bx++) {
            double acc = 0;
            for (size_t dy = 0; dy < 2; dy++)
                for (size_t dx = 0; dx < 2; dx++)
                    acc += field[(by * 2 + dy) * nx + (bx * 2 + dx)];
            EXPECT_NEAR(r.bg[by * (nx / 2) + bx], acc / 4.0, 1e-3)
                << "block " << bx << "," << by;
        }
    }
}

// RB3 — the whole point of the design: binning degrades the background but the
// ROI pixels come back bit-exact, because they travel on their own branch.
TEST(ROIBinSplit, RoiSurvivesBinning) {
    const size_t nx = 64, ny = 48, nz = 1;
    auto field = ramp_field(nx, ny, nz);
    std::vector<RoiPeak> peaks = {{0, 20, 20}, {0, 45, 12}};
    const uint32_t hw = 3;

    auto r = run_split(field, nx, ny, nz, peaks, hw, /*bin=*/2);
    for (const auto& p : peaks) {
        for (int dy = -int(hw); dy <= int(hw); dy++) {
            for (int dx = -int(hw); dx <= int(hw); dx++) {
                const size_t x = size_t(std::clamp(int(p.x) + dx, 0, int(nx) - 1));
                const size_t y = size_t(std::clamp(int(p.y) + dy, 0, int(ny) - 1));
                ASSERT_FLOAT_EQ(r.recon[y * nx + x], field[y * nx + x])
                    << "ROI pixel " << x << "," << y;
            }
        }
    }
}

// RB4 — overlapping boxes store some pixels twice. Both copies hold the same
// value, so scatter order cannot matter; this pins that invariant.
TEST(ROIBinSplit, OverlappingBoxesAreIdempotent) {
    const size_t nx = 64, ny = 48, nz = 1;
    auto field = ramp_field(nx, ny, nz);
    // Two peaks 2 px apart with hw=3 → boxes overlap heavily.
    std::vector<RoiPeak> peaks = {{0, 30, 24}, {0, 32, 24}};

    auto r = run_split(field, nx, ny, nz, peaks, /*hw=*/3, /*bin=*/1);
    for (size_t i = 0; i < field.size(); i++)
        ASSERT_FLOAT_EQ(r.recon[i], field[i]) << "at " << i;
}

// RB5 — a corner peak's box hangs off the frame; clamping must keep every access
// in bounds and still reconstruct exactly. Run this one under compute-sanitizer.
TEST(ROIBinSplit, EdgeBoxesAreClamped) {
    const size_t nx = 32, ny = 32, nz = 1;
    auto field = ramp_field(nx, ny, nz);
    std::vector<RoiPeak> peaks = {{0, 0, 0}, {0, uint16_t(nx - 1), uint16_t(ny - 1)}};

    auto r = run_split(field, nx, ny, nz, peaks, /*hw=*/4, /*bin=*/1);
    for (size_t i = 0; i < field.size(); i++)
        ASSERT_FLOAT_EQ(r.recon[i], field[i]) << "at " << i;
}

// RB6 — port model is asymmetric and the DAG relies on it.
TEST(ROIBinSplit, PortCounts) {
    ROIBinSplitStage<float> s;
    EXPECT_EQ(s.getNumInputs(), 1u);
    EXPECT_EQ(s.getNumOutputs(), 3u);
    EXPECT_EQ(s.getOutputNames(), (std::vector<std::string>{"roi", "bg", "peaks"}));
    s.setInverse(true);
    EXPECT_EQ(s.getNumInputs(), 3u);
    EXPECT_EQ(s.getNumOutputs(), 1u);
}

// RB7 — the inverse pass gets its geometry only from this header, so a silent
// truncation here would place every ROI box on the wrong pixel.
TEST(ROIBinSplit, HeaderSerialization) {
    ROIBinSplitStage<float> enc;
    enc.setDims(1552, 1480, 7);
    enc.setRoiHalfWidth(5);
    enc.setBinFactor(2);
    enc.setPeaks({{0, 1, 2}, {3, 4, 5}});

    uint8_t buf[FZM_STAGE_CONFIG_SIZE] = {};
    const size_t n = enc.serializeHeader(0, buf, sizeof(buf));
    ASSERT_GT(n, 0u);

    ROIBinSplitStage<float> dec;
    dec.deserializeHeader(buf, n);
    EXPECT_EQ(dec.getRoiHalfWidth(), 5u);
    EXPECT_EQ(dec.getBinFactor(), 2u);
    EXPECT_EQ(dec.getNumPeaks(), 2u);
    EXPECT_EQ(dec.getBgCount(), size_t(776) * 740 * 7);

    // A later pipeline dims push must not clobber archive-restored geometry.
    dec.setDims(1, 1, 1);
    EXPECT_EQ(dec.getBgCount(), size_t(776) * 740 * 7);
}

// RB8 — PREALLOCATE depends on these being exact, not merely upper bounds.
TEST(ROIBinSplit, ExactOutputSizes) {
    ROIBinSplitStage<float> s;
    s.setDims(100, 80, 3);
    s.setRoiHalfWidth(4);
    s.setBinFactor(2);
    s.setPeaks({{0, 10, 10}, {1, 20, 20}, {2, 30, 30}});

    const auto est = s.estimateOutputSizes({100 * 80 * 3 * sizeof(float)});
    ASSERT_EQ(est.size(), 3u);
    EXPECT_EQ(est[0], 3u * 81u * sizeof(float));       // 3 peaks x 9x9
    EXPECT_EQ(est[1], 50u * 40u * 3u * sizeof(float)); // binned 2x
    EXPECT_EQ(est[2], 3u * sizeof(RoiPeak));

    const auto act = s.getActualOutputSizesByName();
    EXPECT_EQ(act.at("roi"), est[0]);
    EXPECT_EQ(act.at("bg"), est[1]);
    EXPECT_EQ(act.at("peaks"), est[2]);
}
