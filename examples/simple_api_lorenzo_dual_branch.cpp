/**
 * Simple API usage example: Lorenzo → Bitshuffle → RZE pipeline on a real input file.
 *
 * Demonstrates:
 *   - Building and running a Pipeline on a binary float32 input file
 *   - 2D Lorenzo with zigzag codes on the "codes" output port
 *   - Accessing per-stage actual output sizes after compress()
 *   - enableProfiling() + getLastPerfResult().print() for timing
 *   - Computing compression error statistics on the host
 *
 * Usage:
 *   ./build/bin/examples/simple_api_lorenzo_dual_branch <input_file> [dim_x=3600] [dim_y=1800] [error_bound=1e-3]
 *   Positional and key=value args are both accepted (e.g. dim_x=3600 or just 3600).
 */

#include "fzgpumodules.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace fz;

#include <fstream>

static constexpr size_t CHUNK = 16384;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " <input_file> [dim_x=3600] [dim_y=1800] [error_bound=1e-3]\n"
                  << "  Positional:  <file> <dim_x> <dim_y> <error_bound>\n"
                  << "  Named:       dim_x=N  dim_y=N  error_bound=F  (case-insensitive keys)\n";
        return 1;
    }

    std::string input_file = argv[1];
    size_t dim_x = 3600;
    size_t dim_y = 1800;
    float eb = 1e-3f;

    // Parse remaining args as either positional or key=value (case-insensitive key).
    int positional = 0;
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        const auto eq = arg.find('=');
        if (eq != std::string::npos) {
            std::string key = arg.substr(0, eq);
            const std::string val = arg.substr(eq + 1);
            for (auto& c : key) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            if (key == "dim_x" || key == "dimx")          dim_x = std::stoull(val);
            else if (key == "dim_y" || key == "dimy")     dim_y = std::stoull(val);
            else if (key == "error_bound" || key == "eb") eb    = std::stof(val);
            else { std::cerr << "Unknown argument: " << arg << "\n"; return 1; }
        } else {
            if      (positional == 0) dim_x = std::stoull(arg);
            else if (positional == 1) dim_y = std::stoull(arg);
            else if (positional == 2) eb    = std::stof(arg);
            ++positional;
        }
    }

    if (eb <= 0.0f) {
        std::cerr << "error_bound must be > 0\n";
        return 1;
    }

    const size_t N = dim_x * dim_y;
    const size_t input_bytes = N * sizeof(float);

    std::vector<float> h_input(N);
    std::ifstream infile(input_file, std::ios::binary);
    if (!infile) {
        std::cerr << "Error opening file: " << input_file << "\n";
        return 1;
    }
    infile.read(reinterpret_cast<char*>(h_input.data()), input_bytes);
    if (infile.gcount() != static_cast<std::streamsize>(input_bytes)) {
        std::cerr << "Read mismatch: expected " << input_bytes << " bytes, got " << infile.gcount() << "\n";
        return 1;
    }
    infile.close();

    float* d_input = nullptr;
    cudaMalloc(&d_input, input_bytes);
    cudaMemcpy(d_input, h_input.data(), input_bytes, cudaMemcpyHostToDevice);

    Pipeline p(input_bytes, MemoryStrategy::PREALLOCATE, 4.0f);
    p.enableProfiling(true);
    p.setDims(dim_x, dim_y, 1);

    auto* lorenzo = p.addStage<LorenzoQuantizerStage<float, uint16_t>>();
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
    cudaDeviceSynchronize();

    const auto lrz_sizes = lorenzo->getActualOutputSizesByName();
    const size_t outlier_indices_bytes =
        (lrz_sizes.count("outlier_indices") > 0) ? lrz_sizes.at("outlier_indices") : 0;
    const size_t outlier_count = outlier_indices_bytes / sizeof(uint32_t);

    std::cout << "\n-- Compress profiling --\n";
    p.getLastPerfResult().print(std::cout);

    void* d_reconstructed = nullptr;
    size_t reconstructed_size = 0;
    p.decompress(d_compressed, compressed_size, &d_reconstructed, &reconstructed_size, 0);
    cudaDeviceSynchronize();

    std::cout << "\n-- Decompress profiling --\n";
    p.getLastPerfResult().print(std::cout);

    if (reconstructed_size != input_bytes) {
        std::cerr << "unexpected reconstructed size: " << reconstructed_size
                  << " (expected " << input_bytes << ")\n";
        // d_reconstructed is pool-owned — do NOT cudaFree.
        cudaFree(d_input);
        return 1;
    }

    std::vector<float> h_reconstructed(N);
    cudaMemcpy(
        h_reconstructed.data(),
        d_reconstructed,
        reconstructed_size,
        cudaMemcpyDeviceToHost);

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
    std::cout << "  data grid:     " << dim_x << " x " << dim_y << "\n";
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

    // d_reconstructed is pool-owned (default) — do NOT cudaFree.
    cudaFree(d_input);
    return 0;
}