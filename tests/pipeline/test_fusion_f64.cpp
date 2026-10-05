// End-to-end regression coverage for f64 linear-quantizer Auto specializations.
// Kept separate from the large f32 fusion matrix to constrain NVRTC test cost.

#include "fzgpumodules.h"
#include "fused/common/nvrtc_jit.h"
#include "fused/fused_block/nvrtc_warp_fusion.h"

#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace fz;

TEST(FusionF64, ExistingCustomFloatInversePolicyNeedsNoPrecisionMember) {
    // The pre-f64 policy contract supplies undelta(), without a real_type alias.
    const std::string source = R"(
#include "fused/fused_block/warp_fusion.cuh"
using namespace fz::fused::warp;
struct LegacyFloatPredictor {
    template<int EPL>
    __device__ static void undelta(int (&)[EPL], uint32_t) {}
};
extern "C" __global__ void legacy_float_inverse(
    const unsigned char* meta, const unsigned* offset,
    const unsigned char* payload, float* out) {
    fused_unpack_body<1, PlainRateCoder, LegacyFloatPredictor>(
        32, 4, 1, 0.002f, meta, offset, payload, out);
}
)";
    EXPECT_NO_THROW(fused::nvrtcGetKernel(source, "legacy_float_inverse"));
}

namespace {

enum class Shape { Linear1D, Tiled2D, Tiled3D };

struct Case {
    const char* name;
    Shape shape;
    size_t dx, dy, dz;
    uint32_t tx, ty, tz;
    uint32_t block;
    bool predict;
    bool outlier;
    ErrorBoundMode mode;
    double eb;
};

size_t count(const Case& c) { return c.dx * c.dy * c.dz; }

void buildF64Linear(Pipeline& p, const Case& c,
                    bool high_precision = false,
                    bool use_u16_codes = false) {
    p.setDims(c.dx, c.dy, c.dz);
    if (use_u16_codes) {
        auto* q = p.addStage<QuantizerStage<double, uint16_t>>();
        q->setErrorBound(c.eb);
        q->setErrorBoundMode(c.mode);
        q->setLinearMode(true);
        auto* l = p.addStage<LorenzoStage<int16_t>>();
        l->setBlockSize(c.block);
        p.connect(l, q, "codes");
        auto* a = p.addStage<AdaptiveBitpackStage<int16_t>>();
        a->setBlockSize(c.block);
        a->setOutlierSelection(c.outlier);
        p.connect(a, l);
        return;
    }

    auto* q = p.addStage<QuantizerStage<double, uint32_t>>();
    q->setErrorBound(c.eb);
    q->setErrorBoundMode(c.mode);
    q->setLinearMode(true);
    q->setLinearHighPrecision(high_precision);

    if (c.shape == Shape::Linear1D) {
        auto* l = p.addStage<LorenzoStage<int32_t>>();
        l->setBlockSize(c.block);
        p.connect(l, q, "codes");
        auto* a = p.addStage<AdaptiveBitpackStage<int32_t>>();
        a->setBlockSize(c.block);
        a->setOutlierSelection(c.outlier);
        p.connect(a, l);
    } else {
        auto* l = p.addStage<TiledLorenzoStage<int32_t>>();
        l->setTileShape(c.tx, c.ty, c.tz);
        l->setPredict(c.predict);
        p.connect(l, q, "codes");
        auto* a = p.addStage<AdaptiveBitpackStage<int32_t>>();
        a->setBlockSize(c.tx * c.ty * c.tz);
        a->setOutlierSelection(c.outlier);
        p.connect(a, l);
    }
}

std::vector<double> makeF64Input(const Case& c) {
    std::vector<double> h(count(c));
    // At this offset, the 2e-5 increments are below float's ULP. The selected
    // ABS/NOA half-bounds also leave quantizer bins above 2^24 while fitting the
    // signed int32 code range.
    for (size_t i = 0; i < h.size(); ++i) {
        const int centered = static_cast<int>(i % 53) - 26;
        const int ripple = static_cast<int>(i % 7) - 3;
        h[i] = 1024.0 + centered * 2.0e-5 + ripple * 1.0e-7;
    }
    return h;
}

std::vector<uint8_t> compress(Pipeline& p, const std::vector<double>& h,
                              double** d_input_out = nullptr) {
    const size_t input_bytes = h.size() * sizeof(double);
    double* d_input = nullptr;
    EXPECT_EQ(cudaMalloc(&d_input, input_bytes), cudaSuccess);
    EXPECT_EQ(cudaMemcpy(d_input, h.data(), input_bytes,
                         cudaMemcpyHostToDevice), cudaSuccess);
    if (d_input_out) *d_input_out = d_input;

    void* d_archive = nullptr;
    size_t archive_bytes = 0;
    p.compress(d_input, input_bytes, &d_archive, &archive_bytes, 0);
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::vector<uint8_t> archive(archive_bytes);
    EXPECT_EQ(cudaMemcpy(archive.data(), d_archive, archive_bytes,
                         cudaMemcpyDeviceToHost), cudaSuccess);
    if (!d_input_out) cudaFree(d_input);
    return archive;
}

std::vector<double> decompress(Pipeline& p, const std::vector<uint8_t>& archive,
                               size_t input_bytes, size_t* output_bytes = nullptr) {
    void* d_archive = nullptr;
    EXPECT_EQ(cudaMalloc(&d_archive, archive.size()), cudaSuccess);
    EXPECT_EQ(cudaMemcpy(d_archive, archive.data(), archive.size(),
                         cudaMemcpyHostToDevice), cudaSuccess);
    void* d_output = nullptr;
    size_t output_size = 0;
    p.decompress(d_archive, archive.size(), &d_output, &output_size, 0);
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    if (output_bytes) *output_bytes = output_size;
    EXPECT_EQ(output_size, input_bytes);
    std::vector<double> h(output_size / sizeof(double));
    EXPECT_EQ(cudaMemcpy(h.data(), d_output, output_size,
                         cudaMemcpyDeviceToHost), cudaSuccess);
    cudaFree(d_archive);
    return h;
}

void expectBound(const std::vector<double>& input, const std::vector<double>& output,
                 double abs_bound, const std::string& tag) {
    ASSERT_EQ(input.size(), output.size()) << tag;
    double max_error = 0.0;
    for (size_t i = 0; i < input.size(); ++i)
        max_error = std::max(max_error, std::abs(input[i] - output[i]));
    EXPECT_LE(max_error, abs_bound * 1.00001) << tag << " max error=" << max_error;
}

void expectByteIdentical(const std::vector<double>& a, const std::vector<double>& b,
                         const std::string& tag) {
    ASSERT_EQ(a.size(), b.size()) << tag;
    ASSERT_FALSE(a.empty()) << tag;
    EXPECT_EQ(std::memcmp(a.data(), b.data(), a.size() * sizeof(double)), 0) << tag;
}

} // namespace

TEST(FusionF64, LinearLorenzoAndTiledChainsMatchStaged) {
    // Each case covers a distinct coder/geometry/prediction combination. The
    // non-power-of-two logical extents leave partial blocks and partial tiles.
    const std::vector<Case> cases = {
        {"1d-b32-outlier-abs", Shape::Linear1D, 139, 1, 1, 0, 0, 0,
         32, true, true, ErrorBoundMode::ABS, 1.0e-5},
        {"1d-b32-plain-noa", Shape::Linear1D, 139, 1, 1, 0, 0, 0,
         32, true, false, ErrorBoundMode::NOA, 1.0e-2},
        {"1d-b128-plain-noa", Shape::Linear1D, 267, 1, 1, 0, 0, 0,
         128, true, false, ErrorBoundMode::NOA, 1.0e-2},
        {"2d-predict-outlier-abs", Shape::Tiled2D, 19, 13, 1, 8, 8, 1,
         0, true, true, ErrorBoundMode::ABS, 1.0e-5},
        {"2d-fixed-plain-noa", Shape::Tiled2D, 19, 13, 1, 8, 8, 1,
         0, false, false, ErrorBoundMode::NOA, 1.0e-2},
        {"3d-predict-plain-abs", Shape::Tiled3D, 11, 9, 7, 4, 4, 4,
         0, true, false, ErrorBoundMode::ABS, 1.0e-5},
        {"3d-predict-outlier-abs", Shape::Tiled3D, 11, 9, 7, 4, 4, 4,
         0, true, true, ErrorBoundMode::ABS, 1.0e-5},
        {"3d-fixed-outlier-noa", Shape::Tiled3D, 11, 9, 7, 4, 4, 4,
         0, false, true, ErrorBoundMode::NOA, 1.0e-2},
    };

    for (const Case& c : cases) {
        const std::string tag = c.name;
        const std::vector<double> input = makeF64Input(c);
        const size_t input_bytes = input.size() * sizeof(double);
        Pipeline staged(input_bytes, MemoryStrategy::PREALLOCATE, 3.0f);
        staged.setFusionPolicy(FusionPolicy::Off);
        buildF64Linear(staged, c);
        staged.finalize();
        const auto staged_archive = compress(staged, input);

        Pipeline automatic(input_bytes, MemoryStrategy::PREALLOCATE, 3.0f);
        automatic.setFusionPolicy(FusionPolicy::Auto);
        buildF64Linear(automatic, c);
        automatic.finalize();
        ASSERT_EQ(automatic.getFusionInfo().installed_groups.size(), 1u) << tag;
        EXPECT_EQ(automatic.getFusionInfo().installed_groups[0].implementation,
                  "warp-register") << tag;
        const auto auto_archive = compress(automatic, input);
        ASSERT_EQ(staged_archive.size(), auto_archive.size()) << tag;
        EXPECT_EQ(staged_archive, auto_archive) << "Auto archive differs from staged: " << tag;

        size_t staged_output_bytes = 0;
        const auto staged_recon = decompress(staged, staged_archive, input_bytes,
                                             &staged_output_bytes);
        EXPECT_EQ(staged_output_bytes, input_bytes) << tag;
        size_t auto_output_bytes = 0;
        const auto auto_recon = decompress(automatic, staged_archive, input_bytes,
                                           &auto_output_bytes);
        EXPECT_EQ(auto_output_bytes, input_bytes) << tag;
        EXPECT_EQ(automatic.getFusionInfo().installed_inverse_groups.size(), 1u) << tag;
        EXPECT_EQ(automatic.getFusionInfo().installed_inverse_groups[0].implementation,
                  "warp-register-inverse") << tag;

        const auto range = std::minmax_element(input.begin(), input.end());
        const double abs_bound = c.mode == ErrorBoundMode::ABS
            ? c.eb : c.eb * (*range.second - *range.first);
        expectBound(input, staged_recon, abs_bound, tag + " staged");
        expectBound(input, auto_recon, abs_bound, tag + " Auto");
        EXPECT_EQ(staged_recon, auto_recon) << "Auto inverse differs from staged: " << tag;
        expectByteIdentical(staged_recon, auto_recon,
                            "Auto inverse bytes differ from staged: " + tag);
    }
}

TEST(FusionF64, HighBinsAndTiesToEvenRemainDoublePrecision) {
    const Case c{"half-bin", Shape::Linear1D, 97, 1, 1, 0, 0, 0,
                 32, true, false, ErrorBoundMode::ABS, 0x1p-16};
    const double step = 2.0 * c.eb;
    std::vector<double> input(count(c));
    constexpr int64_t even_bin = (1ll << 24) + 2;
    constexpr int64_t odd_bin = (1ll << 24) + 3;
    for (size_t i = 0; i < input.size(); ++i) {
        const int64_t base = (i & 1u) ? odd_bin : even_bin;
        input[i] = (static_cast<double>(base) + 0.5) * step;
    }
    const size_t input_bytes = input.size() * sizeof(double);

    Pipeline staged(input_bytes, MemoryStrategy::PREALLOCATE, 3.0f);
    staged.setFusionPolicy(FusionPolicy::Off);
    buildF64Linear(staged, c);
    staged.finalize();
    const auto staged_archive = compress(staged, input);

    Pipeline automatic(input_bytes, MemoryStrategy::PREALLOCATE, 3.0f);
    automatic.setFusionPolicy(FusionPolicy::Auto);
    buildF64Linear(automatic, c);
    automatic.finalize();
    ASSERT_EQ(automatic.getFusionInfo().installed_groups.size(), 1u);
    const auto auto_archive = compress(automatic, input);
    EXPECT_EQ(staged_archive, auto_archive);

    const auto staged_recon = decompress(staged, staged_archive, input_bytes);
    const auto auto_recon = decompress(automatic, staged_archive, input_bytes);
    EXPECT_EQ(staged_recon, auto_recon);
    expectByteIdentical(staged_recon, auto_recon, "half-bin reconstruction bytes");
    for (size_t i = 0; i < input.size(); ++i) {
        const int64_t expected_bin = (i & 1u) ? (odd_bin + 1) : even_bin;
        EXPECT_DOUBLE_EQ(staged_recon[i], static_cast<double>(expected_bin) * step)
            << "ties-to-even or f64 reconstruction changed at " << i;
    }
    expectBound(input, auto_recon, c.eb, c.name);
}

TEST(FusionF64, UnsupportedTypesPrecisionAndGeometryStayStaged) {
    const Case c{"fallback", Shape::Linear1D, 139, 1, 1, 0, 0, 0,
                 32, true, true, ErrorBoundMode::ABS, 1.0e-5};
    const size_t bytes = count(c) * sizeof(double);

    Pipeline u16(bytes, MemoryStrategy::PREALLOCATE, 3.0f);
    u16.setFusionPolicy(FusionPolicy::Auto);
    buildF64Linear(u16, c, false, true);
    u16.finalize();
    EXPECT_EQ(u16.getFusedGroupCount(), 0u) << "uint16 codes are not an f64 specialization";

    Pipeline high_precision(bytes, MemoryStrategy::PREALLOCATE, 3.0f);
    high_precision.setFusionPolicy(FusionPolicy::Auto);
    buildF64Linear(high_precision, c, true);
    high_precision.finalize();
    EXPECT_EQ(high_precision.getFusedGroupCount(), 0u)
        << "staged high-precision linear quantizer must not use the warp kernel";

    Case unsupported = c;
    unsupported.shape = Shape::Tiled2D;
    unsupported.dx = 17; unsupported.dy = 11; unsupported.dz = 1;
    unsupported.tx = 4; unsupported.ty = 8; unsupported.tz = 1;
    Pipeline geometry(unsupported.dx * unsupported.dy * sizeof(double),
                      MemoryStrategy::PREALLOCATE, 3.0f);
    geometry.setFusionPolicy(FusionPolicy::Auto);
    buildF64Linear(geometry, unsupported);
    geometry.finalize();
    EXPECT_EQ(geometry.getFusedGroupCount(), 0u)
        << "unregistered tiled geometry must remain staged";
}

TEST(FusionF64, LinearQuantizerRejectsOverflowAndNonfiniteValues) {
    Case c{"overflow", Shape::Linear1D, 65, 1, 1, 0, 0, 0,
           32, true, false, ErrorBoundMode::ABS, 1.0e-7};
    std::vector<double> input(count(c), 1.0e6);
    const size_t bytes = input.size() * sizeof(double);
    double* d_input = nullptr;
    ASSERT_EQ(cudaMalloc(&d_input, bytes), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice), cudaSuccess);

    auto expectRejects = [&](const std::vector<double>& values, const std::string& label) {
        ASSERT_EQ(values.size(), input.size());
        ASSERT_EQ(cudaMemcpy(d_input, values.data(), bytes, cudaMemcpyHostToDevice), cudaSuccess);
        for (FusionPolicy policy : {FusionPolicy::Off, FusionPolicy::Auto}) {
            Pipeline p(bytes, MemoryStrategy::PREALLOCATE, 3.0f);
            p.setFusionPolicy(policy);
            buildF64Linear(p, c);
            p.finalize();
            EXPECT_EQ(p.getFusedGroupCount(), policy == FusionPolicy::Auto ? 1u : 0u)
                << label;
            void* d_archive = nullptr;
            size_t archive_bytes = 0;
            EXPECT_THROW(p.compress(d_input, bytes, &d_archive, &archive_bytes, 0),
                         std::runtime_error)
                << (policy == FusionPolicy::Auto ? "Auto" : "staged")
                << " must reject " << label;
        }
    };

    expectRejects(input, "an unrepresentable int32 quantized coordinate");
    for (double bad_value : {std::numeric_limits<double>::infinity(),
                             -std::numeric_limits<double>::infinity(),
                             std::numeric_limits<double>::quiet_NaN()}) {
        std::vector<double> nonfinite(count(c), 1.0);
        nonfinite[17] = bad_value;  // one offending value in each input vector
        expectRejects(nonfinite, "a nonfinite ABS input value");
    }
    cudaDeviceSynchronize();
    cudaFree(d_input);
}

TEST(FusionF64, AutoArchiveDecodesFromFileThroughStagedFactory) {
    const Case c{"file", Shape::Linear1D, 71, 1, 1, 0, 0, 0,
                 32, true, false, ErrorBoundMode::ABS, 1.0e-5};
    const auto input = makeF64Input(c);
    const size_t bytes = input.size() * sizeof(double);
    Pipeline automatic(bytes, MemoryStrategy::PREALLOCATE, 3.0f);
    automatic.setFusionPolicy(FusionPolicy::Auto);
    buildF64Linear(automatic, c);
    automatic.finalize();
    const auto archive = compress(automatic, input);
    ASSERT_GT(archive.size(), 0u);
    const auto auto_memory = decompress(automatic, archive, bytes);
    Pipeline staged(bytes, MemoryStrategy::PREALLOCATE, 3.0f);
    staged.setFusionPolicy(FusionPolicy::Off);
    buildF64Linear(staged, c);
    staged.finalize();
    const auto staged_archive = compress(staged, input);
    EXPECT_EQ(staged_archive, archive);
    const auto staged_memory = decompress(staged, archive, bytes);
    expectByteIdentical(staged_memory, auto_memory, "in-memory file-test outputs");
    const std::string path = "/tmp/fzgmod_f64_auto_specialization.fzm";
    automatic.writeToFile(path, 0);

    void* d_output = nullptr;
    size_t output_bytes = 0;
    Pipeline::decompressFromFile(path, &d_output, &output_bytes, 0);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    EXPECT_EQ(output_bytes, bytes);
    std::vector<double> decoded(input.size());
    ASSERT_EQ(cudaMemcpy(decoded.data(), d_output, bytes, cudaMemcpyDeviceToHost), cudaSuccess);
    expectBound(input, decoded, c.eb, c.name);
    expectByteIdentical(decoded, staged_memory, "file decode vs staged in-memory decode");
    expectByteIdentical(decoded, auto_memory, "file decode vs Auto in-memory decode");
    cudaFree(d_output);
    std::remove(path.c_str());
}

TEST(FusionF64, WarpCodegenSelectsDoublePolicyAndCoordinates) {
    fused::WarpFusionSpec f32;
    const std::string src_f32 = fused::generateWarpFusionSource(f32);
    fused::WarpFusionSpec f64;
    f64.use_double = true;
    const std::string src_f64 = fused::generateWarpFusionSource(f64);

    EXPECT_NE(src_f32.find("// precision=f32"), std::string::npos);
    EXPECT_NE(src_f32.find("const float* in"), std::string::npos);
    EXPECT_EQ(src_f32.find("Lorenzo1DPredictorF64"), std::string::npos);
    EXPECT_NE(src_f64.find("// precision=f64"), std::string::npos);
    EXPECT_NE(src_f64.find("const double* in"), std::string::npos);
    EXPECT_NE(src_f64.find("Lorenzo1DPredictorF64 pred"), std::string::npos);
    EXPECT_NE(src_f64.find("double inv2eb"), std::string::npos);
}
