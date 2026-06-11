#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <vector>

#include "fzgpumodules.h"
#include "helpers/fz_test_utils.h"
#include "helpers/stage_harness.h"

using namespace fz;
using namespace fz_test;

namespace {
// Smooth 3D field — what spline interpolation handles well. Values are scaled
// to a moderate range so a 1e-2 absolute error bound exercises the quantizer
// without saturating it.
inline std::vector<float> make_smooth_3d(size_t nx, size_t ny, size_t nz,
                                          float scale = 50.0f)
{
    std::vector<float> v(nx * ny * nz);
    for (size_t z = 0; z < nz; ++z)
        for (size_t y = 0; y < ny; ++y)
            for (size_t x = 0; x < nx; ++x) {
                double fx = static_cast<double>(x) / static_cast<double>(nx);
                double fy = static_cast<double>(y) / static_cast<double>(ny);
                double fz = static_cast<double>(z) / static_cast<double>(nz);
                double val = std::sin(2.0 * M_PI * fx) *
                             std::cos(2.0 * M_PI * fy * 1.3) *
                             std::sin(2.0 * M_PI * fz * 0.7);
                v[x + nx * (y + ny * z)] = static_cast<float>(val * scale);
            }
    return v;
}
}  // namespace

// ─────────────────────────────────────────────────────────────────────────────
// GI1: RoundTripABS — 32³ smooth float volume, ABS error bound
// Forward + inverse round trip stays within the user-specified bound.
// Uses default auto-tuned radius (setQuantRadius left at 0).
// ─────────────────────────────────────────────────────────────────────────────
TEST(GInterpStage, GI1_RoundTripABS) {
    const size_t NX = 32, NY = 32, NZ = 32;
    const size_t N = NX * NY * NZ;
    const size_t in_bytes = N * sizeof(float);
    const float  eb = 1e-2f;

    auto h_input = make_smooth_3d(NX, NY, NZ);

    Pipeline p(in_bytes, MemoryStrategy::PREALLOCATE);
    p.setDims(NX, NY, NZ);
    auto* stage = p.addStage<GInterpStage<float, uint16_t>>();
    stage->setErrorBound(eb);
    stage->setErrorBoundMode(ErrorBoundMode::ABS);
    // No setQuantRadius() — exercise the auto-tune default.
    p.finalize();

    CudaStream cs;
    auto res = pipeline_round_trip<float>(p, h_input, cs.stream);

    EXPECT_LE(res.max_error, eb * 1.001)
        << "max_error=" << res.max_error << " > eb=" << eb;
}

// ─────────────────────────────────────────────────────────────────────────────
// GI2: NonCubeDims — exercises non-cube extents (48×32×16)
// ─────────────────────────────────────────────────────────────────────────────
TEST(GInterpStage, GI2_NonCubeDims) {
    const size_t NX = 48, NY = 32, NZ = 16;
    const size_t N = NX * NY * NZ;
    const size_t in_bytes = N * sizeof(float);
    const float  eb = 1e-2f;

    auto h_input = make_smooth_3d(NX, NY, NZ);

    Pipeline p(in_bytes, MemoryStrategy::PREALLOCATE);
    p.setDims(NX, NY, NZ);
    auto* stage = p.addStage<GInterpStage<float, uint16_t>>();
    stage->setErrorBound(eb);
    stage->setErrorBoundMode(ErrorBoundMode::ABS);
    p.finalize();

    CudaStream cs;
    auto res = pipeline_round_trip<float>(p, h_input, cs.stream);

    EXPECT_LE(res.max_error, eb * 1.001)
        << "max_error=" << res.max_error;
}

// ─────────────────────────────────────────────────────────────────────────────
// GI3: OutlierHandling — data with sharp spikes pushes prediction errors out
// of range; the outlier triplet must catch them and reconstruction must stay
// within the documented approximate bound.
//
// Note: spikes propagating through the multi-level interpolation pyramid can
// compound prediction error in neighbours, so the realised max-error on data
// with many outliers is up to ~2 × eb. This is documented in the class
// header ("Error bound and limitations" section).
// ─────────────────────────────────────────────────────────────────────────────
TEST(GInterpStage, GI3_OutlierHandling) {
    const size_t NX = 32, NY = 32, NZ = 32;
    const size_t N = NX * NY * NZ;
    const size_t in_bytes = N * sizeof(float);
    const float  eb = 1e-2f;

    auto h_input = make_smooth_3d(NX, NY, NZ);
    // Inject spikes that the spline cannot predict (huge residual > radius*eb).
    for (size_t i = 0; i < N; i += 73) {
        h_input[i] += 1e4f;
    }

    Pipeline p(in_bytes, MemoryStrategy::PREALLOCATE);
    p.setDims(NX, NY, NZ);
    auto* stage = p.addStage<GInterpStage<float, uint16_t>>();
    stage->setErrorBound(eb);
    stage->setErrorBoundMode(ErrorBoundMode::ABS);
    stage->setOutlierCapacity(0.20f);  // generous, ~6500 entries for N=32768
    p.finalize();

    CudaStream cs;
    auto res = pipeline_round_trip<float>(p, h_input, cs.stream);

    // Approximate bound (documented): outlier-heavy data can hit ~2 × eb due
    // to multi-level prediction error accumulation. We test 2.5 × eb to keep
    // some headroom for float-precision noise in the outlier-as-float path.
    EXPECT_LE(res.max_error, eb * 2.5)
        << "Outlier reconstruction exceeded the documented 2.5×eb operational "
           "envelope; max_error=" << res.max_error;
}

// ─────────────────────────────────────────────────────────────────────────────
// GI4: FileRoundTrip — round-trip through writeToFile + decompressFromFile,
// exercising StageFactory::createStage for G_INTERP and the FZM header.
// ─────────────────────────────────────────────────────────────────────────────
TEST(GInterpStage, GI4_FileRoundTrip) {
    const size_t NX = 32, NY = 32, NZ = 32;
    const size_t N = NX * NY * NZ;
    const size_t in_bytes = N * sizeof(float);
    const float  eb = 1e-2f;
    const std::string path = "/tmp/test_ginterp_gi4.fzm";

    auto h_input = make_smooth_3d(NX, NY, NZ);

    Pipeline p(in_bytes, MemoryStrategy::PREALLOCATE);
    p.setDims(NX, NY, NZ);
    auto* stage = p.addStage<GInterpStage<float, uint16_t>>();
    stage->setErrorBound(eb);
    p.finalize();

    CudaStream cs;
    auto res = pipeline_file_round_trip<float>(p, h_input, cs.stream, path);

    EXPECT_LE(res.max_error, eb * 1.001);
    std::remove(path.c_str());
}

// ─────────────────────────────────────────────────────────────────────────────
// GI5: SerializeDeserialize — config fields survive header round-trip.
// ─────────────────────────────────────────────────────────────────────────────
TEST(GInterpStage, GI5_SerializeDeserialize) {
    GInterpStage<float, uint16_t> original;
    original.setDims(48, 32, 16);
    original.setErrorBound(5e-3f);
    original.setQuantRadius(1024);
    original.setOutlierCapacity(0.15f);
    original.setErrorBoundMode(ErrorBoundMode::NOA);

    uint8_t buf[128] = {};
    size_t written = original.serializeHeader(0, buf, sizeof(buf));
    EXPECT_EQ(written, sizeof(GInterpConfig));

    GInterpStage<float, uint16_t> restored;
    restored.deserializeHeader(buf, written);

    EXPECT_EQ(original.getDims(),       restored.getDims());
    EXPECT_EQ(original.getQuantRadius(), restored.getQuantRadius());
    EXPECT_EQ(original.getErrorBoundMode(), restored.getErrorBoundMode());
    EXPECT_FLOAT_EQ(original.getErrorBound(), restored.getErrorBound());
}

// ─────────────────────────────────────────────────────────────────────────────
// GI6: StageTypeId + GraphCompatible (MVP graph-compat is false).
// ─────────────────────────────────────────────────────────────────────────────
TEST(GInterpStage, GI6_StageTypeId) {
    using GIU16 = GInterpStage<float, uint16_t>;
    EXPECT_EQ(GIU16().getStageTypeId(),
              static_cast<uint16_t>(StageType::G_INTERP));
}
TEST(GInterpStage, GI6b_GraphCompatibleFalseInMVP) {
    using GIU16 = GInterpStage<float, uint16_t>;
    EXPECT_FALSE(GIU16().isGraphCompatible());
}

// ─────────────────────────────────────────────────────────────────────────────
// GI7: SetDimsRejectsNon3D — 1-D / 2-D inputs must throw in MVP.
// ─────────────────────────────────────────────────────────────────────────────
TEST(GInterpStage, GI7_SetDimsRejectsNon3D) {
    GInterpStage<float, uint16_t> s;
    EXPECT_THROW(s.setDims(32, 32, 1),  std::runtime_error);  // 2-D
    EXPECT_THROW(s.setDims(1024, 1, 1), std::runtime_error);  // 1-D
    EXPECT_THROW(s.setDims(0, 32, 32),  std::runtime_error);  // any zero dim
    EXPECT_NO_THROW(s.setDims(32, 32, 32));
}

// ─────────────────────────────────────────────────────────────────────────────
// GI8: AutoRadiusDefault — leaving radius at the sentinel (0) auto-tunes to a
// non-zero, reasonable value on first execute(), and the result is preserved
// across compress/decompress.
// ─────────────────────────────────────────────────────────────────────────────
TEST(GInterpStage, GI8_AutoRadiusDefault) {
    const size_t NX = 32, NY = 32, NZ = 32;
    const size_t N = NX * NY * NZ;
    const size_t in_bytes = N * sizeof(float);
    const float  eb = 1e-2f;

    auto h_input = make_smooth_3d(NX, NY, NZ);

    Pipeline p(in_bytes, MemoryStrategy::PREALLOCATE);
    p.setDims(NX, NY, NZ);
    auto* stage = p.addStage<GInterpStage<float, uint16_t>>();
    stage->setErrorBound(eb);
    EXPECT_EQ(stage->getQuantRadius(), 0)
        << "Default radius should be 0 (auto-tune sentinel)";
    p.finalize();

    CudaStream cs;
    auto res = pipeline_round_trip<float>(p, h_input, cs.stream);

    // After execute(), the auto-tuned radius should be populated.
    int picked = stage->getQuantRadius();
    EXPECT_GT(picked, 0)
        << "Auto-tune did not populate radius after execute()";
    EXPECT_LE(picked, 32767) << "Auto-tuned radius exceeds uint16 cap";
    EXPECT_LE(res.max_error, eb * 1.001);
}

// ─────────────────────────────────────────────────────────────────────────────
// GI9: ManualRadiusSkipsScan — explicit radius > 0 keeps that value verbatim.
// This is the graph-capture / determinism path.
// ─────────────────────────────────────────────────────────────────────────────
TEST(GInterpStage, GI9_ManualRadiusSkipsScan) {
    const size_t NX = 32, NY = 32, NZ = 32;
    const size_t N = NX * NY * NZ;
    const size_t in_bytes = N * sizeof(float);

    auto h_input = make_smooth_3d(NX, NY, NZ);

    Pipeline p(in_bytes, MemoryStrategy::PREALLOCATE);
    p.setDims(NX, NY, NZ);
    auto* stage = p.addStage<GInterpStage<float, uint16_t>>();
    stage->setErrorBound(1e-2f);
    stage->setQuantRadius(512);                  // manual: climate-style
    stage->setOutlierCapacity(0.20f);
    p.finalize();

    CudaStream cs;
    auto res = pipeline_round_trip<float>(p, h_input, cs.stream);

    EXPECT_EQ(stage->getQuantRadius(), 512)
        << "Manual radius was overwritten by auto-tune (it shouldn't be)";
    EXPECT_LE(res.max_error, 1e-2f * 1.5)
        << "Manual radius=512 should still give a reasonable bound on smooth data";
}
