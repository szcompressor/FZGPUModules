/**
 * Direct benchmark for QuantizerStage<float,uint32_t> linear mode.
 *
 * All variants use NOA so the min/max scan cost is identical. This isolates
 * the coordinate kernel change while still reporting full stage wall time,
 * including the overflow-status copy performed by postStreamSync().
 *
 * Usage: fzgmod-profile-quantizer-linear [elements] [runs] [relative_eb]
 */

#include "fzgpumodules.h"
#include "mem/mempool.h"
#include "quantizers/quantizer/quantizer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

using namespace fz;
using Clock = std::chrono::steady_clock;

struct Variant {
    const char* name;
    bool high_precision;
    bool power_of_two;
};

static double percentile50(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const size_t mid = values.size() / 2;
    return values.size() % 2 ? values[mid] : (values[mid - 1] + values[mid]) * 0.5;
}

static void run_variant(const Variant& v, const float* d_input,
                        const std::vector<float>& input, int runs, float user_eb) {
    const size_t n = input.size();
    const size_t bytes = n * sizeof(float);
    MemoryPool pool(MemoryPoolConfig(bytes, 2.5f));
    QuantizerStage<float, uint32_t> stage;
    stage.setErrorBound(user_eb);
    stage.setErrorBoundMode(ErrorBoundMode::NOA);
    stage.setLinearMode(true);
    stage.setLinearHighPrecision(v.high_precision);
    stage.setPowerOfTwoBound(v.power_of_two);
    stage.onFinalize(bytes, &pool);

    uint32_t* d_codes = nullptr;
    cudaMalloc(&d_codes, n * sizeof(uint32_t));
    std::vector<void*> inputs = {const_cast<float*>(d_input)};
    std::vector<void*> outputs = {d_codes};
    std::vector<size_t> sizes = {bytes};

    // Warm the allocator, reduction, kernel, and overflow-status path.
    stage.execute(0, &pool, inputs, outputs, sizes);
    stage.postStreamSync(0);

    std::vector<double> timings;
    timings.reserve(runs);
    for (int i = 0; i < runs; ++i) {
        auto begin = Clock::now();
        stage.execute(0, &pool, inputs, outputs, sizes);
        stage.postStreamSync(0);
        auto end = Clock::now();
        timings.push_back(std::chrono::duration<double, std::milli>(end - begin).count());
    }

    QuantizerStage<float, uint32_t> inverse;
    uint8_t header[128] = {};
    inverse.deserializeHeader(header, stage.serializeHeader(0, header, sizeof(header)));
    inverse.setInverse(true);
    float* d_recon = nullptr;
    cudaMalloc(&d_recon, bytes);
    std::vector<void*> inv_inputs = {d_codes};
    std::vector<void*> inv_outputs = {d_recon};
    std::vector<size_t> inv_sizes = {n * sizeof(uint32_t)};
    inverse.execute(0, &pool, inv_inputs, inv_outputs, inv_sizes);
    cudaDeviceSynchronize();
    std::vector<float> recon(n);
    cudaMemcpy(recon.data(), d_recon, bytes, cudaMemcpyDeviceToHost);

    float max_error = 0.0f;
    for (size_t i = 0; i < n; ++i)
        max_error = std::max(max_error, std::abs(input[i] - recon[i]));
    const float requested_abs_eb = user_eb *
        (*std::max_element(input.begin(), input.end()) -
         *std::min_element(input.begin(), input.end()));
    const double median_ms = percentile50(timings);
    const double gib_s = static_cast<double>(bytes) / median_ms / 1.0e6;

    std::cout << std::left << std::setw(24) << v.name << std::right
              << " median=" << std::fixed << std::setprecision(4) << median_ms << " ms"
              << "  throughput=" << std::setprecision(2) << gib_s << " GB/s"
              << "  effective_eb=" << std::scientific << stage.getComputedAbsEb()
              << "  max/requested=" << std::fixed << std::setprecision(6)
              << (max_error / requested_abs_eb) << "\n";

    cudaFree(d_recon);
    cudaFree(d_codes);
}

int main(int argc, char** argv) {
    const size_t n = argc > 1 ? std::stoull(argv[1]) : (size_t{1} << 24);
    const int runs = argc > 2 ? std::stoi(argv[2]) : 30;
    const float user_eb = argc > 3 ? std::stof(argv[3]) : 1.0e-6f;
    if (n == 0 || runs <= 0 || !(user_eb > 0.0f)) return 2;

    std::vector<float> input(n);
    for (size_t i = 0; i < n; ++i) {
        const float x = static_cast<float>(i % 1048573) * 0.000191f;
        input[i] = 100.0f * std::sin(x) + 0.01f * std::cos(17.0f * x);
    }
    float* d_input = nullptr;
    cudaMalloc(&d_input, n * sizeof(float));
    cudaMemcpy(d_input, input.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    std::cout << "Linear Quantizer NOA benchmark: n=" << n
              << " runs=" << runs << " user_eb=" << std::scientific << user_eb << "\n";
    const Variant variants[] = {
        {"float", false, false},
        {"double", true, false},
        {"float+power2", false, true},
        {"double+power2", true, true},
    };
    for (const auto& v : variants)
        run_variant(v, d_input, input, runs, user_eb);

    cudaFree(d_input);
    return 0;
}
