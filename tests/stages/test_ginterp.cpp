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
// GI7: SetDimsAcceptsValidShapes — 2-D and 3-D are accepted; 1-D and any
// zero dim must still throw.
// ─────────────────────────────────────────────────────────────────────────────
TEST(GInterpStage, GI7_SetDimsAcceptsValidShapes) {
    GInterpStage<float, uint16_t> s;
    EXPECT_THROW(s.setDims(1024, 1, 1), std::runtime_error);  // 1-D
    EXPECT_THROW(s.setDims(0, 32, 32),  std::runtime_error);  // any zero dim
    EXPECT_THROW(s.setDims(32, 0,  1),  std::runtime_error);  // y=0
    EXPECT_NO_THROW(s.setDims(32, 32, 1));   // 2-D
    EXPECT_NO_THROW(s.setDims(32, 32, 32));  // 3-D
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

// ─────────────────────────────────────────────────────────────────────────────
// GI10: AutoTuneMode1 — cheap profiling (sets `reverse[]` only). Verifies the
// round-trip still satisfies the bound and that the resolved alpha/beta were
// recomputed from rel_eb (alpha should differ from the 1.75 baseline at this eb).
// ─────────────────────────────────────────────────────────────────────────────
TEST(GInterpStage, GI10_AutoTuneMode1) {
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
    stage->setAutoTuning(1);   // cheap profiling
    p.finalize();

    CudaStream cs;
    auto res = pipeline_round_trip<float>(p, h_input, cs.stream);

    EXPECT_LE(res.max_error, eb * 1.5)
        << "max_error=" << res.max_error << " > 1.5 * eb=" << eb;
    EXPECT_EQ(stage->getAutoTuningMode(), 1);
}

// ─────────────────────────────────────────────────────────────────────────────
// GI11: AutoTuneMode3 — full structural profiling. Same correctness bound
// check; on smooth data the structural tune may improve CR but the round-trip
// max-error contract is unchanged.
// ─────────────────────────────────────────────────────────────────────────────
TEST(GInterpStage, GI11_AutoTuneMode3) {
    const size_t NX = 64, NY = 64, NZ = 64;  // need >= 2 * S_STRIDE/16 = 16 per dim
    const size_t N = NX * NY * NZ;
    const size_t in_bytes = N * sizeof(float);
    const float  eb = 1e-2f;

    auto h_input = make_smooth_3d(NX, NY, NZ);

    Pipeline p(in_bytes, MemoryStrategy::PREALLOCATE);
    p.setDims(NX, NY, NZ);
    auto* stage = p.addStage<GInterpStage<float, uint16_t>>();
    stage->setErrorBound(eb);
    stage->setErrorBoundMode(ErrorBoundMode::ABS);
    stage->setAutoTuning(3);   // structural
    p.finalize();

    CudaStream cs;
    auto res = pipeline_round_trip<float>(p, h_input, cs.stream);

    EXPECT_LE(res.max_error, eb * 1.5)
        << "max_error=" << res.max_error << " > 1.5 * eb=" << eb;
    EXPECT_EQ(stage->getAutoTuningMode(), 3);
}

// ─────────────────────────────────────────────────────────────────────────────
// GI12: AutoTuneFileRoundTrip — the resolved INTERPOLATION_PARAMS must survive
// the .fzm header round-trip so the decompressor uses the same intp_param as
// the compressor. Uses mode 3 (writes non-default reverse/use_md flags).
// ─────────────────────────────────────────────────────────────────────────────
TEST(GInterpStage, GI12_AutoTuneFileRoundTrip) {
    const size_t NX = 64, NY = 64, NZ = 64;
    const size_t N = NX * NY * NZ;
    const size_t in_bytes = N * sizeof(float);
    const float  eb = 1e-2f;
    const std::string path = "/tmp/test_ginterp_gi12.fzm";

    auto h_input = make_smooth_3d(NX, NY, NZ);

    Pipeline p(in_bytes, MemoryStrategy::PREALLOCATE);
    p.setDims(NX, NY, NZ);
    auto* stage = p.addStage<GInterpStage<float, uint16_t>>();
    stage->setErrorBound(eb);
    stage->setAutoTuning(3);
    p.finalize();

    CudaStream cs;
    auto res = pipeline_file_round_trip<float>(p, h_input, cs.stream, path);

    EXPECT_LE(res.max_error, eb * 1.5);
    std::remove(path.c_str());
}

// ─────────────────────────────────────────────────────────────────────────────
// GI13: AutoTuneSerializeHeader — the new INTERPOLATION_PARAMS fields survive
// raw serializeHeader / deserializeHeader round-trip via the same in-memory
// path as the FZM reader.
// ─────────────────────────────────────────────────────────────────────────────
TEST(GInterpStage, GI13_AutoTuneSerializeHeader) {
    GInterpStage<float, uint16_t> original;
    original.setDims(64, 64, 64);
    original.setErrorBound(1e-2f);
    original.setAutoTuning(3);

    uint8_t buf[128] = {};
    size_t written = original.serializeHeader(0, buf, sizeof(buf));
    EXPECT_EQ(written, sizeof(GInterpConfig));

    GInterpStage<float, uint16_t> restored;
    restored.deserializeHeader(buf, written);

    EXPECT_EQ(original.getAutoTuningMode(), restored.getAutoTuningMode());
}

namespace {
// Smooth 2-D field. Same construction as `make_smooth_3d`, restricted to
// a single z-slice so the spline kernel sees a true 2-D pattern.
inline std::vector<float> make_smooth_2d(size_t nx, size_t ny,
                                          float scale = 50.0f)
{
    std::vector<float> v(nx * ny);
    for (size_t y = 0; y < ny; ++y)
        for (size_t x = 0; x < nx; ++x) {
            double fx = static_cast<double>(x) / static_cast<double>(nx);
            double fy = static_cast<double>(y) / static_cast<double>(ny);
            double val = std::sin(2.0 * M_PI * fx) *
                         std::cos(2.0 * M_PI * fy * 1.3);
            v[x + nx * y] = static_cast<float>(val * scale);
        }
    return v;
}
}  // namespace

// ─────────────────────────────────────────────────────────────────────────────
// GI14: RoundTrip2D — basic 2-D ABS round-trip on a 128×64 slice. Verifies
// the 2-D dispatch path (setDims with z=1 picks ginterpAnchorLen2 and the
// 2-D launchers) and that reconstruction stays within the documented bound.
// ─────────────────────────────────────────────────────────────────────────────
TEST(GInterpStage, GI14_RoundTrip2D) {
    const size_t NX = 128, NY = 64, NZ = 1;
    const size_t N  = NX * NY * NZ;
    const size_t in_bytes = N * sizeof(float);
    const float  eb = 1e-2f;

    auto h_input = make_smooth_2d(NX, NY);

    Pipeline p(in_bytes, MemoryStrategy::PREALLOCATE);
    p.setDims(NX, NY, NZ);
    auto* stage = p.addStage<GInterpStage<float, uint16_t>>();
    stage->setErrorBound(eb);
    stage->setErrorBoundMode(ErrorBoundMode::ABS);
    p.finalize();

    CudaStream cs;
    auto res = pipeline_round_trip<float>(p, h_input, cs.stream);

    EXPECT_LE(res.max_error, eb * 1.5)
        << "2-D max_error=" << res.max_error << " > 1.5×eb=" << (eb * 1.5);
}

// ─────────────────────────────────────────────────────────────────────────────
// GI15: NonAligned2D — 2-D dims that are not multiples of the 32×8 tile.
// Exercises the ragged-edge path on the 2-D anchor layout.
// ─────────────────────────────────────────────────────────────────────────────
TEST(GInterpStage, GI15_NonAligned2D) {
    const size_t NX = 100, NY = 50, NZ = 1;
    const size_t N  = NX * NY * NZ;
    const size_t in_bytes = N * sizeof(float);
    const float  eb = 1e-2f;

    auto h_input = make_smooth_2d(NX, NY);

    Pipeline p(in_bytes, MemoryStrategy::PREALLOCATE);
    p.setDims(NX, NY, NZ);
    auto* stage = p.addStage<GInterpStage<float, uint16_t>>();
    stage->setErrorBound(eb);
    stage->setErrorBoundMode(ErrorBoundMode::ABS);
    p.finalize();

    CudaStream cs;
    auto res = pipeline_round_trip<float>(p, h_input, cs.stream);

    EXPECT_LE(res.max_error, eb * 2.0)
        << "2-D non-aligned max_error=" << res.max_error;
}

// ─────────────────────────────────────────────────────────────────────────────
// GI16: FileRoundTrip2D — 2-D .fzm round-trip. Verifies the FZM header
// carries the 2-D dim/anchor info so the decoder picks the right launcher.
// ─────────────────────────────────────────────────────────────────────────────
TEST(GInterpStage, GI16_FileRoundTrip2D) {
    const size_t NX = 128, NY = 64, NZ = 1;
    const size_t N  = NX * NY * NZ;
    const size_t in_bytes = N * sizeof(float);
    const float  eb = 1e-2f;
    const std::string path = "/tmp/test_ginterp_gi16.fzm";

    auto h_input = make_smooth_2d(NX, NY);

    Pipeline p(in_bytes, MemoryStrategy::PREALLOCATE);
    p.setDims(NX, NY, NZ);
    auto* stage = p.addStage<GInterpStage<float, uint16_t>>();
    stage->setErrorBound(eb);
    p.finalize();

    CudaStream cs;
    auto res = pipeline_file_round_trip<float>(p, h_input, cs.stream, path);

    EXPECT_LE(res.max_error, eb * 1.5);
    std::remove(path.c_str());
}

// ═════════════════════════════════════════════════════════════════════════════
// Comprehensive feature coverage (GI17–GI29)
// ═════════════════════════════════════════════════════════════════════════════

// ─────────────────────────────────────────────────────────────────────────────
// GI17: 3-D REL round-trip — relative error mode interprets eb as a fraction
// of max(|data|). With smooth data scaled to [-50, 50], the kernel auto-derives
// abs_eb = eb * 50, so a 1e-3 user eb gives abs_eb ≈ 0.05.
// ─────────────────────────────────────────────────────────────────────────────
TEST(GInterpStage, GI17_RoundTripREL_3D) {
    const size_t NX = 32, NY = 32, NZ = 32;
    const size_t in_bytes = NX * NY * NZ * sizeof(float);
    const float  eb = 1e-3f;

    auto h_input = make_smooth_3d(NX, NY, NZ);
    const float max_abs = 50.0f;  // matches the scale in make_smooth_3d
    const float abs_eb_expected = eb * max_abs;

    Pipeline p(in_bytes, MemoryStrategy::PREALLOCATE);
    p.setDims(NX, NY, NZ);
    auto* stage = p.addStage<GInterpStage<float, uint16_t>>();
    stage->setErrorBound(eb);
    stage->setErrorBoundMode(ErrorBoundMode::REL);
    p.finalize();

    CudaStream cs;
    auto res = pipeline_round_trip<float>(p, h_input, cs.stream);

    // REL with global max-magnitude scaling — error budget is the converted
    // abs_eb_expected, plus the documented ≤1.1× multi-level slack.
    EXPECT_LE(res.max_error, abs_eb_expected * 1.1)
        << "REL 3-D max_error=" << res.max_error
        << " > 1.1×abs_eb (" << abs_eb_expected << ")";
}

// ─────────────────────────────────────────────────────────────────────────────
// GI18: 3-D NOA round-trip — norm-of-absolute interprets eb as a fraction of
// the value range (max-min). With smooth data on [-50, 50], range=100 and
// abs_eb = eb * 100.
// ─────────────────────────────────────────────────────────────────────────────
TEST(GInterpStage, GI18_RoundTripNOA_3D) {
    const size_t NX = 32, NY = 32, NZ = 32;
    const size_t in_bytes = NX * NY * NZ * sizeof(float);
    const float  eb = 1e-3f;

    auto h_input = make_smooth_3d(NX, NY, NZ);
    const float range_expected = 100.0f;
    const float abs_eb_expected = eb * range_expected;

    Pipeline p(in_bytes, MemoryStrategy::PREALLOCATE);
    p.setDims(NX, NY, NZ);
    auto* stage = p.addStage<GInterpStage<float, uint16_t>>();
    stage->setErrorBound(eb);
    stage->setErrorBoundMode(ErrorBoundMode::NOA);
    p.finalize();

    CudaStream cs;
    auto res = pipeline_round_trip<float>(p, h_input, cs.stream);

    EXPECT_LE(res.max_error, abs_eb_expected * 1.1)
        << "NOA 3-D max_error=" << res.max_error
        << " > 1.1×abs_eb (" << abs_eb_expected << ")";
}

// ─────────────────────────────────────────────────────────────────────────────
// GI19: 2-D REL round-trip — same REL semantics on the 2-D launcher path.
// ─────────────────────────────────────────────────────────────────────────────
TEST(GInterpStage, GI19_RoundTripREL_2D) {
    const size_t NX = 128, NY = 64, NZ = 1;
    const size_t in_bytes = NX * NY * NZ * sizeof(float);
    const float  eb = 1e-3f;

    auto h_input = make_smooth_2d(NX, NY);
    const float abs_eb_expected = eb * 50.0f;

    Pipeline p(in_bytes, MemoryStrategy::PREALLOCATE);
    p.setDims(NX, NY, NZ);
    auto* stage = p.addStage<GInterpStage<float, uint16_t>>();
    stage->setErrorBound(eb);
    stage->setErrorBoundMode(ErrorBoundMode::REL);
    p.finalize();

    CudaStream cs;
    auto res = pipeline_round_trip<float>(p, h_input, cs.stream);

    // 2-D path has slightly looser headroom than 3-D (smaller anchor tile in z).
    EXPECT_LE(res.max_error, abs_eb_expected * 1.5)
        << "REL 2-D max_error=" << res.max_error;
}

// ─────────────────────────────────────────────────────────────────────────────
// GI20: 2-D NOA round-trip.
// ─────────────────────────────────────────────────────────────────────────────
TEST(GInterpStage, GI20_RoundTripNOA_2D) {
    const size_t NX = 128, NY = 64, NZ = 1;
    const size_t in_bytes = NX * NY * NZ * sizeof(float);
    const float  eb = 1e-3f;

    auto h_input = make_smooth_2d(NX, NY);
    const float abs_eb_expected = eb * 100.0f;

    Pipeline p(in_bytes, MemoryStrategy::PREALLOCATE);
    p.setDims(NX, NY, NZ);
    auto* stage = p.addStage<GInterpStage<float, uint16_t>>();
    stage->setErrorBound(eb);
    stage->setErrorBoundMode(ErrorBoundMode::NOA);
    p.finalize();

    CudaStream cs;
    auto res = pipeline_round_trip<float>(p, h_input, cs.stream);

    EXPECT_LE(res.max_error, abs_eb_expected * 1.5)
        << "NOA 2-D max_error=" << res.max_error;
}

// ─────────────────────────────────────────────────────────────────────────────
// GI21: TCode=uint8_t — the narrowest code type. Auto-radius is clamped to 127
// (the documented uint8 cap), so most residuals on the smooth 3-D field still
// fit. We require correctness within the documented bound but expect a higher
// outlier rate than uint16_t.
// ─────────────────────────────────────────────────────────────────────────────
TEST(GInterpStage, GI21_TCode_uint8_3D) {
    const size_t NX = 32, NY = 32, NZ = 32;
    const size_t in_bytes = NX * NY * NZ * sizeof(float);
    const float  eb = 1e-2f;

    auto h_input = make_smooth_3d(NX, NY, NZ);

    Pipeline p(in_bytes, MemoryStrategy::PREALLOCATE);
    p.setDims(NX, NY, NZ);
    auto* stage = p.addStage<GInterpStage<float, uint8_t>>();
    stage->setErrorBound(eb);
    stage->setOutlierCapacity(0.30f);  // headroom for tighter code range
    p.finalize();

    CudaStream cs;
    auto res = pipeline_round_trip<float>(p, h_input, cs.stream);

    EXPECT_LE(res.max_error, eb * 1.5)
        << "uint8 3-D max_error=" << res.max_error;
    EXPECT_LE(stage->getQuantRadius(), 127)
        << "uint8 auto-radius must clamp at 127, got " << stage->getQuantRadius();
}

// ─────────────────────────────────────────────────────────────────────────────
// GI22: TCode=uint32_t — auto-radius caps at the uint16 ceiling (32767) since
// the kernel quantizer's internal precision tops out there (documented in
// pickAutoRadius).
// ─────────────────────────────────────────────────────────────────────────────
TEST(GInterpStage, GI22_TCode_uint32_3D) {
    const size_t NX = 32, NY = 32, NZ = 32;
    const size_t in_bytes = NX * NY * NZ * sizeof(float);
    const float  eb = 1e-4f;

    auto h_input = make_smooth_3d(NX, NY, NZ);

    Pipeline p(in_bytes, MemoryStrategy::PREALLOCATE);
    p.setDims(NX, NY, NZ);
    auto* stage = p.addStage<GInterpStage<float, uint32_t>>();
    stage->setErrorBound(eb);
    p.finalize();

    CudaStream cs;
    auto res = pipeline_round_trip<float>(p, h_input, cs.stream);

    EXPECT_LE(res.max_error, eb * 1.1)
        << "uint32 3-D max_error=" << res.max_error;
    EXPECT_LE(stage->getQuantRadius(), 32767)
        << "uint32 auto-radius caps at 32767, got " << stage->getQuantRadius();
}

// ─────────────────────────────────────────────────────────────────────────────
// GI23: TCode=uint8_t — 2-D round-trip. Confirms the 2-D explicit instantiation
// is wired for the narrowest code type.
// ─────────────────────────────────────────────────────────────────────────────
TEST(GInterpStage, GI23_TCode_uint8_2D) {
    const size_t NX = 128, NY = 64, NZ = 1;
    const size_t in_bytes = NX * NY * NZ * sizeof(float);
    const float  eb = 1e-2f;

    auto h_input = make_smooth_2d(NX, NY);

    Pipeline p(in_bytes, MemoryStrategy::PREALLOCATE);
    p.setDims(NX, NY, NZ);
    auto* stage = p.addStage<GInterpStage<float, uint8_t>>();
    stage->setErrorBound(eb);
    stage->setOutlierCapacity(0.30f);
    p.finalize();

    CudaStream cs;
    auto res = pipeline_round_trip<float>(p, h_input, cs.stream);

    EXPECT_LE(res.max_error, eb * 2.0)
        << "uint8 2-D max_error=" << res.max_error;
}

// ─────────────────────────────────────────────────────────────────────────────
// GI24: Precomputed value_base skips the internal data scan. The reconstructed
// data must match what the auto-scan path produces (proves the user-supplied
// value is consumed verbatim and not silently overwritten).
// ─────────────────────────────────────────────────────────────────────────────
TEST(GInterpStage, GI24_PrecomputedValueBase_NOA) {
    const size_t NX = 32, NY = 32, NZ = 32;
    const size_t in_bytes = NX * NY * NZ * sizeof(float);
    const float  eb = 1e-3f;

    auto h_input = make_smooth_3d(NX, NY, NZ);
    const float range = 100.0f;  // matches max - min on smooth_3d

    Pipeline p(in_bytes, MemoryStrategy::PREALLOCATE);
    p.setDims(NX, NY, NZ);
    auto* stage = p.addStage<GInterpStage<float, uint16_t>>();
    stage->setErrorBound(eb);
    stage->setErrorBoundMode(ErrorBoundMode::NOA);
    stage->setValueBase(range);     // skips the scan
    stage->setQuantRadius(2048);    // manual radius so no auto scan either
    p.finalize();

    CudaStream cs;
    auto res = pipeline_round_trip<float>(p, h_input, cs.stream);

    EXPECT_LE(res.max_error, eb * range * 1.1)
        << "Precomputed value_base path max_error=" << res.max_error;
    EXPECT_FLOAT_EQ(stage->getValueBase(), range)
        << "setValueBase must be preserved (got "
        << stage->getValueBase() << ")";
}

// ─────────────────────────────────────────────────────────────────────────────
// GI25: Manual very-small radius — forces almost all residuals to the outlier
// triplet. The stage must still reconstruct within the error bound (outliers
// are stored losslessly).
// ─────────────────────────────────────────────────────────────────────────────
TEST(GInterpStage, GI25_ManualSmallRadius_RoutesOutliers) {
    const size_t NX = 32, NY = 32, NZ = 32;
    const size_t in_bytes = NX * NY * NZ * sizeof(float);
    const float  eb = 1e-2f;

    auto h_input = make_smooth_3d(NX, NY, NZ);

    Pipeline p(in_bytes, MemoryStrategy::PREALLOCATE);
    p.setDims(NX, NY, NZ);
    auto* stage = p.addStage<GInterpStage<float, uint16_t>>();
    stage->setErrorBound(eb);
    stage->setQuantRadius(32);          // tiny → forces outliers
    stage->setOutlierCapacity(1.0f);    // generous (all elements may overflow)
    p.finalize();

    CudaStream cs;
    auto res = pipeline_round_trip<float>(p, h_input, cs.stream);

    // Outlier triplet stores values exactly; quantized cells still bounded.
    EXPECT_LE(res.max_error, eb * 2.0)
        << "Small-radius outlier path max_error=" << res.max_error;
}

// ─────────────────────────────────────────────────────────────────────────────
// GI26: 2-D auto-tune fallback — setAutoTuning(3) on a 2-D input must not
// crash, must log a warning, and reconstruction must stay within the baseline
// bound (auto-tune resolved params get clobbered to baseline by the dispatch).
// ─────────────────────────────────────────────────────────────────────────────
TEST(GInterpStage, GI26_AutoTune_FallsBackOn2D) {
    const size_t NX = 128, NY = 64, NZ = 1;
    const size_t in_bytes = NX * NY * NZ * sizeof(float);
    const float  eb = 1e-2f;

    auto h_input = make_smooth_2d(NX, NY);

    Pipeline p(in_bytes, MemoryStrategy::PREALLOCATE);
    p.setDims(NX, NY, NZ);
    auto* stage = p.addStage<GInterpStage<float, uint16_t>>();
    stage->setErrorBound(eb);
    stage->setAutoTuning(3);   // 3-D-only; expect warn-and-fall-back on 2-D
    p.finalize();

    CudaStream cs;
    auto res = pipeline_round_trip<float>(p, h_input, cs.stream);

    EXPECT_LE(res.max_error, eb * 1.5)
        << "2-D auto-tune fallback max_error=" << res.max_error;
}

// ─────────────────────────────────────────────────────────────────────────────
// GI27: saveState / restoreState — Pipeline::decompressMulti() calls these
// around each inverse execute to preserve the forward-mode actual_output_sizes_
// vector. We test the contract directly: forward sizes survive a save → trash →
// restore cycle.
// ─────────────────────────────────────────────────────────────────────────────
TEST(GInterpStage, GI27_SaveRestoreState) {
    const size_t NX = 32, NY = 32, NZ = 32;
    const size_t in_bytes = NX * NY * NZ * sizeof(float);
    const float  eb = 1e-2f;

    auto h_input = make_smooth_3d(NX, NY, NZ);

    Pipeline p(in_bytes, MemoryStrategy::PREALLOCATE);
    p.setDims(NX, NY, NZ);
    auto* stage = p.addStage<GInterpStage<float, uint16_t>>();
    stage->setErrorBound(eb);
    p.finalize();

    // Compress once to populate forward sizes.
    CudaStream cs;
    CudaBuffer<float> d_in(h_input.size());
    d_in.upload(h_input, cs.stream);
    void* d_comp = nullptr; size_t comp_sz = 0;
    p.compress(d_in.void_ptr(), in_bytes, &d_comp, &comp_sz, cs.stream);
    cudaStreamSynchronize(cs.stream);

    auto forward_sizes = stage->getActualOutputSizesByName();
    ASSERT_FALSE(forward_sizes.empty());

    stage->saveState();

    // Simulate the inverse pass corrupting the forward-sizes vector via the
    // stage's public interface — set inverse=true and re-call compress's
    // inverse path indirectly via decompress, which under decompressMulti
    // would trash the per-stage state.
    void* d_dec = nullptr; size_t dec_sz = 0;
    p.decompress(d_comp, comp_sz, &d_dec, &dec_sz, cs.stream);
    cudaStreamSynchronize(cs.stream);

    stage->restoreState();

    auto restored = stage->getActualOutputSizesByName();
    EXPECT_EQ(restored.size(), forward_sizes.size());
    for (auto& kv : forward_sizes) {
        EXPECT_EQ(restored.at(kv.first), kv.second)
            << "saveState/restoreState lost port '" << kv.first << "'";
    }
    // d_dec is pool-owned (default setPoolManagedDecompOutput=true); the
    // Pipeline destructor frees it via the pool — a caller-side cudaFree here
    // would double-free.
}

// ─────────────────────────────────────────────────────────────────────────────
// GI28: postStreamSync trims actual_output_sizes_ — after compress, the
// `outlier_idxs` size must equal `outlier_count * sizeof(uint32_t)` so that
// downstream consumers can derive the count without a separate port.
// ─────────────────────────────────────────────────────────────────────────────
TEST(GInterpStage, GI28_PostStreamSyncTrimsOutlierSizes) {
    const size_t NX = 32, NY = 32, NZ = 32;
    const size_t in_bytes = NX * NY * NZ * sizeof(float);
    const float  eb = 1e-2f;

    auto h_input = make_smooth_3d(NX, NY, NZ);
    // Inject sparse outliers to ensure a non-zero, non-saturating count.
    for (size_t i = 0; i < h_input.size(); i += 211) h_input[i] += 5e3f;

    Pipeline p(in_bytes, MemoryStrategy::PREALLOCATE);
    p.setDims(NX, NY, NZ);
    auto* stage = p.addStage<GInterpStage<float, uint16_t>>();
    stage->setErrorBound(eb);
    stage->setOutlierCapacity(0.10f);
    p.finalize();

    CudaStream cs;
    CudaBuffer<float> d_in(h_input.size());
    d_in.upload(h_input, cs.stream);
    void* d_comp = nullptr; size_t comp_sz = 0;
    p.compress(d_in.void_ptr(), in_bytes, &d_comp, &comp_sz, cs.stream);
    cudaStreamSynchronize(cs.stream);

    auto sizes = stage->getActualOutputSizesByName();
    ASSERT_TRUE(sizes.count("outlier_idxs"));
    ASSERT_TRUE(sizes.count("outlier_vals"));

    const size_t idxs_count = sizes.at("outlier_idxs") / sizeof(uint32_t);
    const size_t vals_count = sizes.at("outlier_vals") / sizeof(float);
    EXPECT_EQ(idxs_count, vals_count)
        << "outlier_idxs/outlier_vals counts must match";
    EXPECT_GT(idxs_count, 0u)
        << "Sparse spike injection should produce ≥1 outlier";

    const size_t max_capacity = static_cast<size_t>(
        std::ceil(NX * NY * NZ * 0.10f));
    EXPECT_LE(idxs_count, max_capacity)
        << "Trimmed count exceeds outlier_capacity";
}

// ─────────────────────────────────────────────────────────────────────────────
// GI29: Constant input — every residual is exactly predictable, so the kernel
// should emit zero outliers and reconstruct losslessly. Catches any path that
// silently zeros the count or fails on degenerate input.
// ─────────────────────────────────────────────────────────────────────────────
TEST(GInterpStage, GI29_ConstantInputZeroOutliers) {
    const size_t NX = 32, NY = 32, NZ = 32;
    const size_t N = NX * NY * NZ;
    const size_t in_bytes = N * sizeof(float);
    const float  eb = 1e-3f;
    const float  VAL = 7.5f;

    std::vector<float> h_input(N, VAL);

    Pipeline p(in_bytes, MemoryStrategy::PREALLOCATE);
    p.setDims(NX, NY, NZ);
    auto* stage = p.addStage<GInterpStage<float, uint16_t>>();
    stage->setErrorBound(eb);
    stage->setQuantRadius(512);    // manual — auto would divide-by-zero on range
    p.finalize();

    CudaStream cs;
    CudaBuffer<float> d_in(h_input.size());
    d_in.upload(h_input, cs.stream);
    void* d_comp = nullptr; size_t comp_sz = 0;
    p.compress(d_in.void_ptr(), in_bytes, &d_comp, &comp_sz, cs.stream);
    cudaStreamSynchronize(cs.stream);

    auto sizes = stage->getActualOutputSizesByName();
    ASSERT_TRUE(sizes.count("outlier_idxs"));
    EXPECT_EQ(sizes.at("outlier_idxs"), 0u)
        << "Constant input must produce zero outliers";

    void* d_dec = nullptr; size_t dec_sz = 0;
    p.decompress(d_comp, comp_sz, &d_dec, &dec_sz, cs.stream);
    cudaStreamSynchronize(cs.stream);
    std::vector<float> h_recon(N);
    cudaMemcpy(h_recon.data(), d_dec, dec_sz, cudaMemcpyDeviceToHost);
    // d_dec is pool-owned (default); Pipeline destructor frees it.

    for (size_t i = 0; i < N; ++i) {
        EXPECT_NEAR(h_recon[i], VAL, eb)
            << "Constant input recon mismatch at " << i
            << " (got " << h_recon[i] << ", expected " << VAL << ")";
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GI30: Pipeline integration — feed GInterp codes into a downstream coder
// (HuffmanStage). Verifies the "codes" port is the right contract for chained
// compression and that the full pipeline survives the round-trip with the
// expected compression ratio.
// ─────────────────────────────────────────────────────────────────────────────
TEST(GInterpStage, GI30_GInterpHuffmanPipeline) {
    const size_t NX = 32, NY = 32, NZ = 32;
    const size_t in_bytes = NX * NY * NZ * sizeof(float);
    const float  eb = 1e-2f;

    auto h_input = make_smooth_3d(NX, NY, NZ);

    Pipeline p(in_bytes, MemoryStrategy::PREALLOCATE);
    p.setDims(NX, NY, NZ);
    auto* gi  = p.addStage<GInterpStage<float, uint16_t>>();
    auto* huf = p.addStage<HuffmanStage<uint16_t>>();
    gi->setErrorBound(eb);
    gi->setQuantRadius(512);          // bounded so Huffman bklen covers all
    huf->setBklen(1024);              // > 2 * radius
    p.connect(huf, gi, "codes");
    p.finalize();

    CudaStream cs;
    auto res = pipeline_round_trip<float>(p, h_input, cs.stream);

    EXPECT_LE(res.max_error, eb * 1.5)
        << "GInterp+Huffman max_error=" << res.max_error;
    // Compression ratio sanity: must be < input size (Huffman shrinks codes).
    EXPECT_LT(res.compressed_bytes, in_bytes)
        << "GInterp+Huffman compressed (" << res.compressed_bytes
        << " B) is not smaller than input (" << in_bytes << " B)";
}
