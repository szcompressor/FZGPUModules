/**
 * Simple API usage example:
 *
 *   Lorenzo(+fused quantization, zigzag codes) -> bitshuffle -> RZE
 *
 * Usage:
 *   ./build/bin/simple_api_lorenzo_dual_branch [error_bound]
 */

#include "cuda_check.h"
#include "fzgpumodules.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace fz;

static constexpr size_t DIM_X = 4096;
static constexpr size_t DIM_Y = 2048;
static constexpr size_t N = DIM_X * DIM_Y;
static constexpr size_t CHUNK = 16384;
static constexpr size_t MIN_TEST_BYTES = 20ull * 1024ull * 1024ull;
static_assert(N * sizeof(float) >= MIN_TEST_BYTES,
              "simple_api_lorenzo_dual_branch input must be >= 20 MiB");

static std::vector<float> make_demo_data() {
    std::vector<float> h(N);
    for (size_t y = 0; y < DIM_Y; ++y) {
        for (size_t x = 0; x < DIM_X; ++x) {
            const size_t i = y * DIM_X + x;
            float v = 1.5f * std::sin(0.01f * static_cast<float>(x))
                    + 0.75f * std::cos(0.02f * static_cast<float>(y));
            if ((x + y) % 257 == 0) v += 6.0f;
            h[i] = v;
        }
    }
    return h;
}

int main(int argc, char** argv) {
    float eb = 1e-3f;
    if (argc > 1) {
        eb = std::stof(argv[1]);
        if (eb <= 0.0f) {
            std::cerr << "error_bound must be > 0\n";
            return 1;
        }
    }

    const size_t input_bytes = N * sizeof(float);
    const auto h_input = make_demo_data();

    float* d_input = nullptr;
    FZ_CUDA_CHECK(cudaMalloc(&d_input, input_bytes));
    FZ_CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), input_bytes, cudaMemcpyHostToDevice));

    Pipeline p(input_bytes, MemoryStrategy::PREALLOCATE, 4.0f);
    p.enableProfiling(true);
    p.setDims(DIM_X, DIM_Y, 1);

    auto* lorenzo = p.addStage<LorenzoStage<float, uint16_t>>();
    lorenzo->setErrorBound(eb);
    lorenzo->setErrorBoundMode(ErrorBoundMode::ABS);
    lorenzo->setQuantRadius(32768);
    lorenzo->setOutlierCapacity(0.10f);
    lorenzo->setZigzagCodes(true);

    auto* bshuf_codes = p.addStage<BitshuffleStage>();
    bshuf_codes->setBlockSize(CHUNK);
    bshuf_codes->setElementWidth(sizeof(uint16_t));
    p.connect(bshuf_codes, lorenzo, "codes");

    auto* rze_codes = p.addStage<RZEStage>();
    rze_codes->setChunkSize(CHUNK);
    rze_codes->setLevels(4);
    p.connect(rze_codes, bshuf_codes);

    p.finalize();

    void* d_compressed = nullptr;
    size_t compressed_size = 0;
    p.compress(d_input, input_bytes, &d_compressed, &compressed_size, 0);
    FZ_CUDA_CHECK(cudaDeviceSynchronize());

    const auto lrz_sizes = lorenzo->getActualOutputSizesByName();
    const size_t outlier_indices_bytes =
        (lrz_sizes.count("outlier_indices") > 0) ? lrz_sizes.at("outlier_indices") : 0;
    const size_t outlier_count = outlier_indices_bytes / sizeof(uint32_t);

    std::cout << "\n-- Compress profiling --\n";
    p.getLastPerfResult().print(std::cout);

    void* d_reconstructed = nullptr;
    size_t reconstructed_size = 0;
    p.decompress(d_compressed, compressed_size, &d_reconstructed, &reconstructed_size, 0);
    FZ_CUDA_CHECK(cudaDeviceSynchronize());

    std::cout << "\n-- Decompress profiling --\n";
    p.getLastPerfResult().print(std::cout);

    if (reconstructed_size != input_bytes) {
        std::cerr << "unexpected reconstructed size: " << reconstructed_size
                  << " (expected " << input_bytes << ")\n";
        if (d_reconstructed) FZ_CUDA_CHECK(cudaFree(d_reconstructed));
        FZ_CUDA_CHECK(cudaFree(d_input));
        return 1;
    }

    std::vector<float> h_reconstructed(N);
    FZ_CUDA_CHECK(cudaMemcpy(
        h_reconstructed.data(),
        d_reconstructed,
        reconstructed_size,
        cudaMemcpyDeviceToHost));

    float max_abs_error = 0.0f;
    double sum_abs_error = 0.0;
    double sum_sq_error = 0.0;
    size_t within_eb_count = 0;
    for (size_t i = 0; i < N; ++i) {
        const float abs_err = std::abs(h_reconstructed[i] - h_input[i]);
        max_abs_error = std::max(max_abs_error, abs_err);
        sum_abs_error += static_cast<double>(abs_err);
        sum_sq_error += static_cast<double>(abs_err) * static_cast<double>(abs_err);
        if (abs_err <= eb) {
            ++within_eb_count;
        }
    }
    const double mae = sum_abs_error / static_cast<double>(N);
    const double rmse = std::sqrt(sum_sq_error / static_cast<double>(N));
    const double within_eb_pct = 100.0 * static_cast<double>(within_eb_count) / static_cast<double>(N);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Simple PFPL-style Lorenzo API example\n";
    std::cout << "  data grid:     " << DIM_X << " x " << DIM_Y << "\n";
    std::cout << "  lorenzo mode:  2D\n";
    std::cout << "  error bound:   " << eb << " (ABS)\n";
    std::cout << "  input bytes:   " << input_bytes << "\n";
    std::cout << "  input MiB:     "
              << (static_cast<double>(input_bytes) / (1024.0 * 1024.0))
              << "\n";
    std::cout << "  compressed:    " << compressed_size << "\n";
    std::cout << "  outliers:      " << outlier_count
              << (outlier_count > 0 ? " (present)" : " (none)") << "\n";
    std::cout << "  ratio:         "
              << (compressed_size > 0
                  ? static_cast<double>(input_bytes) / compressed_size
                  : 0.0)
              << "x\n";
    std::cout << "  max abs error: " << max_abs_error << "\n";
    std::cout << "  mae:           " << mae << "\n";
    std::cout << "  rmse:          " << rmse << "\n";
    std::cout << "  <= eb:         " << within_eb_pct << "%\n";

    FZ_CUDA_CHECK(cudaFree(d_reconstructed));
    FZ_CUDA_CHECK(cudaFree(d_input));
    return 0;
}