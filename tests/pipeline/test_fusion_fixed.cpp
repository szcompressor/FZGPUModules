// End-to-end coverage for predictor-free Quantizer(linear) -> AdaptiveBitpack.
// This is the cuSZp3 fixed 1-D graph; the pipeline intentionally has exactly two
// stages so the test will fail if a synthetic predictor is introduced.

#include "fzgpumodules.h"

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
#include <type_traits>
#include <vector>

using namespace fz;

namespace {

struct FixedCase {
    const char* name;
    size_t n;
    uint32_t block;
    bool outlier;
    ErrorBoundMode mode;
    double eb;
    bool half_bins = false;
};

template<typename Real>
std::vector<Real> makeInput(const FixedCase& c) {
    std::vector<Real> h(c.n);
    if constexpr (std::is_same_v<Real, double>) {
        if (c.half_bins) {
            const double step = 2.0 * c.eb;
            constexpr int64_t even_bin = (1ll << 24) + 2;
            constexpr int64_t odd_bin = (1ll << 24) + 3;
            for (size_t i = 0; i < c.n; ++i) {
                const int64_t base = (i & 1u) ? odd_bin : even_bin;
                h[i] = (static_cast<double>(base) + 0.5) * step;
            }
            return h;
        }
    }

    for (size_t i = 0; i < c.n; ++i) {
        if constexpr (std::is_same_v<Real, double>) {
            // Many distinct values sit between adjacent float values, while
            // the absolute quantizer coordinate remains inside signed int32.
            const int centered = static_cast<int>(i % 53) - 26;
            const int ripple = static_cast<int>(i % 7) - 3;
            h[i] = 1024.0 + centered * 2.0e-5 + ripple * 1.0e-7;
        } else {
            h[i] = static_cast<Real>(0.5 * std::sin(i * 0.003) +
                                     0.2 * std::cos(i * 0.017));
        }
    }
    return h;
}

template<typename Real>
void buildFixed(Pipeline& p, const FixedCase& c, bool high_precision = false,
                bool use_u16 = false) {
    p.setDims(c.n, 1, 1);
    if (use_u16) {
        auto* q = p.addStage<QuantizerStage<Real, uint16_t>>();
        q->setErrorBound(c.eb);
        q->setErrorBoundMode(c.mode);
        q->setLinearMode(true);
        auto* coder = p.addStage<AdaptiveBitpackStage<int16_t>>();
        coder->setBlockSize(c.block);
        coder->setOutlierSelection(c.outlier);
        p.connect(coder, q, "codes");
        return;
    }
    auto* q = p.addStage<QuantizerStage<Real, uint32_t>>();
    q->setErrorBound(c.eb);
    q->setErrorBoundMode(c.mode);
    q->setLinearMode(true);
    q->setLinearHighPrecision(high_precision);
    auto* coder = p.addStage<AdaptiveBitpackStage<int32_t>>();
    coder->setBlockSize(c.block);
    coder->setOutlierSelection(c.outlier);
    p.connect(coder, q, "codes");
}

template<typename Real>
std::vector<uint8_t> compress(Pipeline& p, const std::vector<Real>& input) {
    const size_t input_bytes = input.size() * sizeof(Real);
    Real* d_input = nullptr;
    EXPECT_EQ(cudaMalloc(&d_input, input_bytes), cudaSuccess);
    EXPECT_EQ(cudaMemcpy(d_input, input.data(), input_bytes,
                         cudaMemcpyHostToDevice), cudaSuccess);
    void* d_payload = nullptr;
    size_t payload_bytes = 0;
    p.compress(d_input, input_bytes, &d_payload, &payload_bytes, 0);
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::vector<uint8_t> payload(payload_bytes);
    EXPECT_EQ(cudaMemcpy(payload.data(), d_payload, payload_bytes,
                         cudaMemcpyDeviceToHost), cudaSuccess);
    cudaFree(d_input);
    return payload;
}

template<typename Real>
std::vector<Real> decompress(Pipeline& p, const std::vector<uint8_t>& payload,
                             size_t expected_bytes, size_t* output_bytes = nullptr) {
    void* d_payload = nullptr;
    EXPECT_EQ(cudaMalloc(&d_payload, payload.size()), cudaSuccess);
    EXPECT_EQ(cudaMemcpy(d_payload, payload.data(), payload.size(),
                         cudaMemcpyHostToDevice), cudaSuccess);
    void* d_output = nullptr;
    size_t bytes = 0;
    p.decompress(d_payload, payload.size(), &d_output, &bytes, 0);
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    if (output_bytes) *output_bytes = bytes;
    EXPECT_EQ(bytes, expected_bytes);
    std::vector<Real> output(bytes / sizeof(Real));
    EXPECT_EQ(cudaMemcpy(output.data(), d_output, bytes,
                         cudaMemcpyDeviceToHost), cudaSuccess);
    cudaFree(d_payload);
    return output;
}

template<typename Real>
void expectSameBytes(const std::vector<Real>& a, const std::vector<Real>& b,
                     const std::string& context) {
    ASSERT_EQ(a.size(), b.size()) << context;
    ASSERT_FALSE(a.empty()) << context;
    EXPECT_EQ(std::memcmp(a.data(), b.data(), a.size() * sizeof(Real)), 0) << context;
}

template<typename Real>
void expectBound(const FixedCase& c, const std::vector<Real>& input,
                 const std::vector<Real>& output, const std::string& context) {
    ASSERT_EQ(input.size(), output.size()) << context;
    double max_error = 0.0;
    for (size_t i = 0; i < input.size(); ++i)
        max_error = std::max(max_error,
                             std::abs(static_cast<double>(input[i]) - output[i]));
    double absolute_bound = c.eb;
    if (c.mode == ErrorBoundMode::NOA) {
        const auto mm = std::minmax_element(input.begin(), input.end());
        absolute_bound *= static_cast<double>(*mm.second) - *mm.first;
    }
    EXPECT_LE(max_error, absolute_bound * 1.00001)
        << context << " max error=" << max_error << " bound=" << absolute_bound;
}

template<typename Real>
void runCase(const FixedCase& c) {
    const auto input = makeInput<Real>(c);
    const size_t bytes = input.size() * sizeof(Real);
    Pipeline staged(bytes, MemoryStrategy::PREALLOCATE, 3.0f);
    staged.setFusionPolicy(FusionPolicy::Off);
    buildFixed<Real>(staged, c);
    staged.finalize();
    ASSERT_EQ(staged.getFusedGroupCount(), 0u) << c.name;
    const auto staged_payload = compress(staged, input);

    Pipeline automatic(bytes, MemoryStrategy::PREALLOCATE, 3.0f);
    automatic.setFusionPolicy(FusionPolicy::Auto);
    buildFixed<Real>(automatic, c);
    automatic.finalize();
    ASSERT_EQ(automatic.getFusionInfo().installed_groups.size(), 1u) << c.name;
    const auto& forward = automatic.getFusionInfo().installed_groups[0];
    EXPECT_EQ(forward.stages.size(), 2u) << c.name;
    EXPECT_EQ(forward.implementation, "warp-linear-quant-coder") << c.name;
    const auto auto_payload = compress(automatic, input);
    EXPECT_EQ(auto_payload, staged_payload) << c.name;

    size_t staged_bytes = 0;
    const auto staged_output = decompress<Real>(staged, staged_payload, bytes, &staged_bytes);
    EXPECT_EQ(staged_bytes, bytes) << c.name;
    size_t auto_bytes = 0;
    const auto auto_output = decompress<Real>(automatic, staged_payload, bytes, &auto_bytes);
    EXPECT_EQ(auto_bytes, bytes) << c.name;
    ASSERT_EQ(automatic.getFusionInfo().installed_inverse_groups.size(), 1u) << c.name;
    const auto& inverse = automatic.getFusionInfo().installed_inverse_groups[0];
    EXPECT_EQ(inverse.stages.size(), 2u) << c.name;
    EXPECT_EQ(inverse.implementation, "warp-linear-quant-coder-inverse") << c.name;

    expectBound(c, input, staged_output, std::string(c.name) + " staged");
    expectBound(c, input, auto_output, std::string(c.name) + " Auto");
    expectSameBytes(staged_output, auto_output, std::string(c.name) + " roundtrip");

    if (c.half_bins) {
        const double step = 2.0 * c.eb;
        constexpr int64_t even_bin = (1ll << 24) + 2;
        constexpr int64_t odd_bin = (1ll << 24) + 3;
        for (size_t i = 0; i < input.size(); ++i) {
            const int64_t expected_bin = (i & 1u) ? odd_bin + 1 : even_bin;
            EXPECT_DOUBLE_EQ(staged_output[i], static_cast<double>(expected_bin) * step)
                << c.name << " ties-to-even mismatch at element " << i;
        }
    }
}

template<typename Real>
void expectFixedInputRejected(const FixedCase& c, const std::vector<Real>& input,
                              const std::string& reason) {
    const size_t bytes = input.size() * sizeof(Real);
    Real* d_input = nullptr;
    ASSERT_EQ(cudaMalloc(&d_input, bytes), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice), cudaSuccess);
    for (FusionPolicy policy : {FusionPolicy::Off, FusionPolicy::Auto}) {
        Pipeline p(bytes, MemoryStrategy::PREALLOCATE, 3.0f);
        p.setFusionPolicy(policy);
        buildFixed<Real>(p, c);
        p.finalize();
        EXPECT_EQ(p.getFusedGroupCount(), policy == FusionPolicy::Auto ? 1u : 0u)
            << reason;
        void* d_payload = nullptr;
        size_t payload_bytes = 0;
        EXPECT_THROW(p.compress(d_input, bytes, &d_payload, &payload_bytes, 0),
                     std::runtime_error)
            << (policy == FusionPolicy::Auto ? "Auto" : "staged")
            << " must reject " << reason;
    }
    cudaFree(d_input);
}

} // namespace

TEST(FusionFixed, PredictorFreeFixed1DMatchesStaged) {
    // Include the paper's block-32 fixed preset, both coder modes, both input
    // precisions, NOA and ABS, and the supported EPL=2/4 blocks. Every length
    // has a partial final block; payloads from each Auto arm are compared with
    // the staged archive before their inverse paths are checked.
    runCase<float>({"f32-b32-plain-abs", 1037, 32, false,
                    ErrorBoundMode::ABS, 1.0e-3});
    runCase<float>({"f32-b32-outlier-noa", 1037, 32, true,
                    ErrorBoundMode::NOA, 1.0e-3});
    runCase<float>({"f32-b128-plain-abs", 1165, 128, false,
                    ErrorBoundMode::ABS, 1.0e-3});

    runCase<double>({"f64-b32-plain-halfbins", 97, 32, false,
                     ErrorBoundMode::ABS, 0x1p-16, true});
    runCase<double>({"f64-b32-outlier-noa", 1037, 32, true,
                     ErrorBoundMode::NOA, 1.0e-2});
    runCase<double>({"f64-b64-plain-noa", 1091, 64, false,
                     ErrorBoundMode::NOA, 1.0e-2});
    runCase<double>({"f64-b128-plain-abs", 1165, 128, false,
                     ErrorBoundMode::ABS, 1.0e-5});
}

TEST(FusionFixed, FileDecodeMatchesBothInMemoryPolicies) {
    const FixedCase c{"fixed-file", 131, 32, true, ErrorBoundMode::ABS, 1.0e-5};
    const auto input = makeInput<double>(c);
    const size_t bytes = input.size() * sizeof(double);

    Pipeline automatic(bytes, MemoryStrategy::PREALLOCATE, 3.0f);
    automatic.setFusionPolicy(FusionPolicy::Auto);
    buildFixed<double>(automatic, c);
    automatic.finalize();
    const auto auto_payload = compress(automatic, input);
    const auto auto_memory = decompress<double>(automatic, auto_payload, bytes);

    Pipeline staged(bytes, MemoryStrategy::PREALLOCATE, 3.0f);
    staged.setFusionPolicy(FusionPolicy::Off);
    buildFixed<double>(staged, c);
    staged.finalize();
    const auto staged_payload = compress(staged, input);
    EXPECT_EQ(auto_payload, staged_payload);
    const auto staged_memory = decompress<double>(staged, auto_payload, bytes);
    expectSameBytes(staged_memory, auto_memory, "fixed file in-memory decode");

    const std::string path = "/tmp/fzgmod_fixed_specialization.fzm";
    automatic.writeToFile(path, 0);
    void* d_output = nullptr;
    size_t output_bytes = 0;
    Pipeline::decompressFromFile(path, &d_output, &output_bytes, 0);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(output_bytes, bytes);
    std::vector<double> file_output(input.size());
    ASSERT_EQ(cudaMemcpy(file_output.data(), d_output, bytes,
                         cudaMemcpyDeviceToHost), cudaSuccess);
    expectBound(c, input, file_output, "fixed file decode");
    expectSameBytes(file_output, staged_memory, "fixed file vs staged decode");
    expectSameBytes(file_output, auto_memory, "fixed file vs Auto decode");
    cudaFree(d_output);
    std::remove(path.c_str());
}

TEST(FusionFixed, UnsupportedModesAndBlocksStayStaged) {
    const FixedCase c{"fixed-fallback", 139, 32, false,
                      ErrorBoundMode::ABS, 1.0e-5};
    const size_t bytes = c.n * sizeof(double);

    Pipeline high_precision(bytes, MemoryStrategy::PREALLOCATE, 3.0f);
    high_precision.setFusionPolicy(FusionPolicy::Auto);
    buildFixed<double>(high_precision, c, true);
    high_precision.finalize();
    EXPECT_EQ(high_precision.getFusedGroupCount(), 0u)
        << "strict high precision remains staged";

    Pipeline u16(bytes, MemoryStrategy::PREALLOCATE, 3.0f);
    u16.setFusionPolicy(FusionPolicy::Auto);
    buildFixed<double>(u16, c, false, true);
    u16.finalize();
    EXPECT_EQ(u16.getFusedGroupCount(), 0u) << "uint16 codes remain staged";

    FixedCase unsupported = c;
    unsupported.block = 16;
    Pipeline block(bytes, MemoryStrategy::PREALLOCATE, 3.0f);
    block.setFusionPolicy(FusionPolicy::Auto);
    buildFixed<double>(block, unsupported);
    block.finalize();
    EXPECT_EQ(block.getFusedGroupCount(), 0u)
        << "block sizes below the warp-register minimum remain staged";
}

TEST(FusionFixed, InvalidBinsAndNonfiniteValuesRejectInStagedAndAuto) {
    const FixedCase c{"fixed-invalid", 65, 32, false,
                      ErrorBoundMode::ABS, 1.0e-7};
    auto exercisePrecision = [&](auto real_tag) {
        using Real = decltype(real_tag);
        std::vector<Real> overflow(c.n, static_cast<Real>(1.0e6));
        expectFixedInputRejected(c, overflow, "an unrepresentable int32 coordinate");
        for (Real bad : {std::numeric_limits<Real>::infinity(),
                         -std::numeric_limits<Real>::infinity(),
                         std::numeric_limits<Real>::quiet_NaN()}) {
            std::vector<Real> nonfinite(c.n, Real{1});
            nonfinite[17] = bad;
            expectFixedInputRejected(c, nonfinite, "a nonfinite ABS input value");
        }
    };
    exercisePrecision(float{});
    exercisePrecision(double{});
}
