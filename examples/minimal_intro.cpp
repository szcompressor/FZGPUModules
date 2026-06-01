/**
 * examples/minimal_intro.cpp
 *
 * Hello-world introduction to FZGPUModules.
 *
 * The shortest complete example: compress a float32 array, decompress it,
 * and verify the reconstruction error and compression ratio.
 *
 * Pipeline: LorenzoQuantStage<float,uint16_t> → HuffmanStage<uint16_t>
 *   (the canonical cuSZ pipeline — good CR on smooth scientific data)
 *
 * Demonstrates:
 *   - Pipeline construction: addStage, connect, finalize
 *   - compress() and decompress()
 *   - Pool-ownership rules (what NOT to cudaFree)
 *   - Computing max absolute error and compression ratio on the host
 *
 * No external data files required — uses synthetic float data.
 *
 * Usage:
 *   ./build/bin/examples/minimal_intro
 *
 * Build:
 *   cmake --preset release -DBUILD_EXAMPLES=ON && cmake --build build/release -j$(nproc)
 *   Binary: build/release/bin/examples/minimal_intro
 */

#include "fzgpumodules.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace fz;

int main() {
    // ── 1. Prepare data ───────────────────────────────────────────────────────
    static constexpr size_t N  = 1 << 20;  // 1 M floats = 4 MB
    static constexpr float  EB = 1e-4f;

    std::vector<float> h_input(N);
    for (size_t i = 0; i < N; ++i) {
        const float t = static_cast<float>(i) / static_cast<float>(N);
        h_input[i] = std::sin(2.0f * 3.14159265f * t)
                   + 0.5f * std::cos(6.0f * 3.14159265f * t);
    }
    const size_t input_bytes = N * sizeof(float);

    float* d_input = nullptr;
    cudaMalloc(&d_input, input_bytes);
    cudaMemcpy(d_input, h_input.data(), input_bytes, cudaMemcpyHostToDevice);

    // ── 2. Build pipeline ─────────────────────────────────────────────────────
    Pipeline p(input_bytes);

    auto* lq = p.addStage<LorenzoQuantStage<float, uint16_t>>();
    lq->setErrorBound(EB);
    lq->setErrorBoundMode(ErrorBoundMode::ABS);
    lq->setQuantRadius(512);
    lq->setZigzagCodes(true);

    // Zigzag maps [-511, 511] → [0, 1022]; bklen=1024 covers the full range.
    auto* huf = p.addStage<HuffmanStage<uint16_t>>();
    huf->setBklen(1024);
    p.connect(huf, lq, "codes");

    p.finalize();

    // ── 3. Compress ───────────────────────────────────────────────────────────
    void*  d_comp  = nullptr;
    size_t comp_sz = 0;
    p.compress(d_input, input_bytes, &d_comp, &comp_sz);
    cudaDeviceSynchronize();
    // d_comp is pool-owned — do NOT cudaFree.

    // ── 4. Decompress ─────────────────────────────────────────────────────────
    void*  d_decomp  = nullptr;
    size_t decomp_sz = 0;
    p.decompress(d_comp, comp_sz, &d_decomp, &decomp_sz);
    cudaDeviceSynchronize();
    // d_decomp is pool-owned — do NOT cudaFree.

    // ── 5. Verify ─────────────────────────────────────────────────────────────
    std::vector<float> h_out(N);
    cudaMemcpy(h_out.data(), d_decomp, decomp_sz, cudaMemcpyDeviceToHost);

    float max_err = 0.0f;
    for (size_t i = 0; i < N; ++i)
        max_err = std::max(max_err, std::abs(h_out[i] - h_input[i]));

    std::printf("input:     %zu bytes (%.2f MB)\n", input_bytes, input_bytes / 1048576.0);
    std::printf("compressed:%zu bytes (%.2f MB)\n", comp_sz, comp_sz / 1048576.0);
    std::printf("ratio:     %.2fx\n", static_cast<double>(input_bytes) / comp_sz);
    std::printf("max_err:   %.2e  (eb = %.2e)\n", max_err, static_cast<double>(EB));

    cudaFree(d_input);  // d_input is caller-owned — must cudaFree
    return 0;
}
