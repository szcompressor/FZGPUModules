/**
 * examples/cusz_standalone.cpp
 *
 * LorenzoQuantStage → HuffmanStage driven manually without a Pipeline.
 *
 * Use this pattern when you need precise control over execution order,
 * intermediate buffer placement, or per-stage stream assignment — for
 * instance, to interleave other GPU work between compression stages,
 * or to run stages asynchronously on multiple streams.
 *
 * Demonstrates:
 *   - Constructing a caller-managed MemoryPool for stage scratch
 *   - HuffmanStage::onFinalize() and LorenzoQuantStage::onFinalize() to
 *     pre-allocate persistent scratch (PHF buffers + the 4-byte outlier-count
 *     scratch) so no allocation happens inside the timed compress loop
 *   - LorenzoQuantStage::execute() — 1 input, 3 outputs
 *       (codes, outlier_errors, outlier_indices)
 *   - LorenzoQuantStage::postStreamSync() to read the actual outlier count
 *     from the stage-private device scratch (no DAG count port)
 *   - HuffmanStage::execute() in compress and decompress direction
 *   - Stage::setInverse(true) to switch to the decompression pass
 *   - LorenzoQuantStage::execute() in decompress mode — 3 inputs, 1 output;
 *     the inverse path reads the outlier count from the stage's internal
 *     state (populated by postStreamSync on the forward pass; populated by
 *     deserializeHeader when going via .fzm files)
 *   - Verifying the full compress → decompress round-trip
 *
 * Contrast with the Pipeline DAG path:
 *   - Pipeline manages the pool, allocates intermediate buffers, and calls
 *     postStreamSync() automatically — you write connect() instead of buffer ptrs.
 *   - The standalone path gives you raw pointers but requires manual management
 *     of everything the Pipeline normally hides.
 *
 * No external data files required — uses synthetic float data.
 *
 * Usage:
 *   ./build/release/bin/examples/cusz_standalone
 *
 * Build:
 *   cmake --preset release -DBUILD_EXAMPLES=ON && cmake --build build/release -j$(nproc)
 *   Binary: build/release/bin/examples/cusz_standalone
 */

#include "fzgpumodules.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace fz;

int main() {
    // ── 1. Input data ─────────────────────────────────────────────────────────
    static constexpr size_t N  = 1 << 18;  // 256 K floats = 1 MB
    static constexpr float  EB = 1e-4f;
    static constexpr float  OUTLIER_CAPACITY = 0.10f;
    static constexpr uint16_t QUANT_RADIUS   = 512;
    // Zigzag maps [-511, 511] → [0, 1022]; bklen=1024 covers the full range.
    static constexpr uint16_t BKLEN          = 2 * QUANT_RADIUS;

    std::vector<float> h_input(N);
    for (size_t i = 0; i < N; ++i) {
        const float t = static_cast<float>(i) / static_cast<float>(N);
        h_input[i] = std::sin(2.0f * 3.14159265f * t)
                   + 0.5f * std::cos(6.0f * 3.14159265f * t);
    }
    const size_t input_bytes = N * sizeof(float);
    const size_t codes_bytes = N * sizeof(uint16_t);
    const size_t max_outliers = static_cast<size_t>(N * OUTLIER_CAPACITY) + 1;
    const size_t huf_out_cap  = codes_bytes * 2 + 4096;  // generous upper bound

    float* d_input = nullptr;
    cudaMalloc(&d_input, input_bytes);
    cudaMemcpy(d_input, h_input.data(), input_bytes, cudaMemcpyHostToDevice);

    // ── 2. Stage setup ────────────────────────────────────────────────────────
    LorenzoQuantStage<float, uint16_t> lq;
    lq.setErrorBound(EB);
    lq.setErrorBoundMode(ErrorBoundMode::ABS);
    lq.setQuantRadius(QUANT_RADIUS);
    lq.setOutlierCapacity(OUTLIER_CAPACITY);
    lq.setZigzagCodes(true);

    HuffmanStage<uint16_t> huf;
    huf.setBklen(BKLEN);

    // ── 3. Pool and persistent scratch ───────────────────────────────────────
    //
    // The pool serves two roles:
    //   a) Huffman persistent scratch: pre-allocated here via onFinalize()
    //      so that no allocation happens inside the timed compress() loop.
    //   b) LorenzoQuant transient scratch: tiny workspace for the NOA/REL
    //      value-range scan (not needed here with ABS mode, but the pool
    //      pointer is required by execute() regardless).
    //
    // Size the pool to cover Huffman's full device footprint plus padding.
    MemoryPool pool(MemoryPoolConfig(input_bytes, 4.0f));
    huf.onFinalize(codes_bytes, &pool);  // pre-allocates PHF scratch buffers
    lq.onFinalize(input_bytes, &pool);   // pre-allocates the 4-byte outlier-count scratch

    // ── 4. Intermediate device buffers ────────────────────────────────────────
    //
    // Caller owns all of these — cudaFree every one on exit.
    // These are distinct from pool memory (which is internal scratch).
    void* d_codes            = nullptr;
    void* d_outlier_errors   = nullptr;
    void* d_outlier_indices  = nullptr;
    void* d_huf_output       = nullptr;
    void* d_reconstructed    = nullptr;

    cudaMalloc(&d_codes,           codes_bytes);
    cudaMalloc(&d_outlier_errors,  max_outliers * sizeof(float));
    cudaMalloc(&d_outlier_indices, max_outliers * sizeof(uint32_t));
    cudaMalloc(&d_huf_output,      huf_out_cap);
    cudaMalloc(&d_reconstructed,   input_bytes);
    cudaDeviceSynchronize();

    // ── 5. Compress: LorenzoQuant → Huffman ───────────────────────────────────
    //
    // LorenzoQuant compress: 1 input → 3 outputs (codes, outlier_errors,
    // outlier_indices). The outlier count is held in a stage-private 4-byte
    // device scratch (allocated by onFinalize() above) and read back by
    // postStreamSync() — there is no count output port.
    lq.execute(0, &pool,
        {d_input},
        {d_codes, d_outlier_errors, d_outlier_indices},
        {input_bytes});
    cudaDeviceSynchronize();

    // postStreamSync() reads the actual outlier count back to host and trims
    // actual_output_sizes_ to the real (not max-capacity) sizes.
    // Must be called after the stream is fully synchronized.
    lq.postStreamSync(0);

    const size_t actual_outlier_errors_bytes  = lq.getActualOutputSize(1);
    const size_t actual_outlier_indices_bytes = lq.getActualOutputSize(2);
    const size_t n_outliers = actual_outlier_errors_bytes / sizeof(float);
    std::printf("LorenzoQuant compress: %zu elements, %zu outliers (%.2f%%)\n",
                N, n_outliers, 100.0 * n_outliers / N);

    // Huffman compress: 1 input (codes) → 1 output (compressed bitstream).
    huf.execute(0, &pool,
        {d_codes},
        {d_huf_output},
        {codes_bytes});
    cudaDeviceSynchronize();

    const size_t compressed_bytes = huf.getActualOutputSize(0);
    std::printf("Huffman compress:      %zu bytes → %zu bytes (%.2fx for codes)\n",
                codes_bytes, compressed_bytes,
                static_cast<double>(codes_bytes) / compressed_bytes);
    std::printf("Full ratio (input/compressed_codes): %.2fx\n",
                static_cast<double>(input_bytes) / compressed_bytes);

    // ── 6. Decompress: Huffman inverse → LorenzoQuant inverse ─────────────────
    //
    // Both stages reuse the same objects with setInverse(true).
    // The Huffman header embedded in d_huf_output tells the decoder the
    // original length — no external metadata needed for the codes buffer.

    // Huffman decompress: compressed bitstream → codes.
    huf.setInverse(true);
    huf.execute(0, &pool,
        {d_huf_output},
        {d_codes},       // reuse d_codes as output
        {compressed_bytes});
    cudaDeviceSynchronize();

    // LorenzoQuant decompress: 3 inputs → 1 output (reconstructed floats).
    // The outlier count comes from the stage's internal `actual_outlier_count_`
    // (populated by postStreamSync() above on the forward pass; populated by
    // deserializeHeader() when going via .fzm files) — not from a port.
    lq.setInverse(true);
    lq.execute(0, &pool,
        {d_codes, d_outlier_errors, d_outlier_indices},
        {d_reconstructed},
        {codes_bytes,
         actual_outlier_errors_bytes,
         actual_outlier_indices_bytes});
    cudaDeviceSynchronize();

    // ── 7. Verify ─────────────────────────────────────────────────────────────
    std::vector<float> h_out(N);
    cudaMemcpy(h_out.data(), d_reconstructed, input_bytes, cudaMemcpyDeviceToHost);

    float max_err = 0.0f;
    for (size_t i = 0; i < N; ++i)
        max_err = std::max(max_err, std::abs(h_out[i] - h_input[i]));

    std::printf("max_abs_error: %.2e  (eb = %.2e)\n",
                max_err, static_cast<double>(EB));

    // ── 8. Cleanup ────────────────────────────────────────────────────────────
    cudaFree(d_input);
    cudaFree(d_codes);
    cudaFree(d_outlier_errors);
    cudaFree(d_outlier_indices);
    cudaFree(d_huf_output);
    cudaFree(d_reconstructed);
    // pool is stack-allocated — its destructor frees Huffman's persistent scratch
    // (and the 4-byte outlier-count scratch owned by LorenzoQuantStage).
    return 0;
}
