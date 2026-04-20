/**
 * Simple caller-allocated output example:
 *
 *   - compressInto():    writes compressed bytes into a user-owned device buffer
 *   - decompressInto():  writes decompressed bytes into a user-owned device buffer
 *
 * Build:
 *   cmake -S . -B build -DBUILD_EXAMPLES=ON
 *   cmake --build build -j --target caller_allocated_output
 *
 * Run:
 *   ./build/bin/caller_allocated_output
 */

#include "cuda_check.h"
#include "fzgpumodules.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace fz;

static std::vector<float> make_smooth_data(size_t n) {
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) {
        v[i] = std::sin(static_cast<float>(i) * 0.01f) * 50.0f
             + std::cos(static_cast<float>(i) * 0.003f) * 20.0f;
    }
    return v;
}

int main() {
    constexpr size_t N = 1 << 16;
    constexpr float EB = 1e-2f;
    const size_t input_bytes = N * sizeof(float);

    auto h_input = make_smooth_data(N);

    float* d_input = nullptr;
    FZ_CUDA_CHECK(cudaMalloc(&d_input, input_bytes));
    FZ_CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), input_bytes, cudaMemcpyHostToDevice));

    Pipeline pipeline(input_bytes, MemoryStrategy::MINIMAL);
    auto* lorenzo = pipeline.addStage<LorenzoStage<float, uint16_t>>();
    lorenzo->setErrorBound(EB);
    lorenzo->setQuantRadius(512);
    pipeline.finalize();

    // Ask the pipeline for a max compressed output size before allocating.
    const size_t compressed_capacity = pipeline.getMaxCompressedOutputSize();

    void* d_compressed_user = nullptr;
    FZ_CUDA_CHECK(cudaMalloc(&d_compressed_user, compressed_capacity));

    size_t compressed_size = 0;
    pipeline.compressInto(
        d_input,
        input_bytes,
        d_compressed_user,
        compressed_capacity,
        &compressed_size,
        0);
    FZ_CUDA_CHECK(cudaDeviceSynchronize());

    // Ask the pipeline for a safe upper bound before allocating output.
    const size_t decompressed_capacity = pipeline.getMaxDecompressedOutputSize();

    // User-owned decompressed output buffer.
    void* d_decompressed_user = nullptr;
    FZ_CUDA_CHECK(cudaMalloc(&d_decompressed_user, decompressed_capacity));

    size_t decompressed_size = 0;
    pipeline.decompressInto(
        d_compressed_user,
        compressed_size,
        d_decompressed_user,
        decompressed_capacity,
        &decompressed_size,
        0);
    FZ_CUDA_CHECK(cudaDeviceSynchronize());

    if (decompressed_size != input_bytes) {
        std::cerr << "Unexpected decompressed size: " << decompressed_size
                  << " (expected " << input_bytes << ")\n";
        cudaFree(d_decompressed_user);
        cudaFree(d_compressed_user);
        cudaFree(d_input);
        return 1;
    }

    std::vector<float> h_recon(N);
    FZ_CUDA_CHECK(cudaMemcpy(
        h_recon.data(), d_decompressed_user, input_bytes, cudaMemcpyDeviceToHost));

    float max_abs_error = 0.0f;
    for (size_t i = 0; i < N; ++i) {
        max_abs_error = std::max(max_abs_error, std::abs(h_recon[i] - h_input[i]));
    }

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Caller-allocated output API example\n";
    std::cout << "  input bytes:        " << input_bytes << "\n";
    std::cout << "  compressed size:    " << compressed_size << "\n";
    std::cout << "  compressed cap:     " << compressed_capacity << "\n";
    std::cout << "  decompressed cap:   " << decompressed_capacity << "\n";
    std::cout << "  decompressed size:  " << decompressed_size << "\n";
    std::cout << "  max abs error:      " << max_abs_error << "\n";

    cudaFree(d_decompressed_user);
    cudaFree(d_compressed_user);
    cudaFree(d_input);
    return 0;
}
