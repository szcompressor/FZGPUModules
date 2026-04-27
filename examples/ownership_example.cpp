/**
 * ownership_example.cpp — compress and decompress output ownership semantics.
 *
 * The pipeline has two independently controllable ownership modes.
 *
 * Compress output (choose one per call site):
 *   pool-owned (default)   compress() returns a pool-owned pointer.
 *                          Do NOT cudaFree. Valid until the next compress(),
 *                          reset(), or Pipeline destruction — whichever comes first.
 *
 *   caller-provided        Use compress(d_input, input_sz, d_buf, buf_cap, &actual_sz).
 *                          Caller allocates d_buf (cudaMalloc / static buffer / etc.)
 *                          and owns it completely — the pipeline never touches it after
 *                          the call returns. Use getMaxCompressedSize() for a safe cap.
 *
 * Decompress output (choose once per Pipeline via setPoolManagedDecompOutput):
 *   pool-owned (default)   setPoolManagedDecompOutput(true)
 *                          Do NOT cudaFree. Valid until the next decompress() call
 *                          or Pipeline destruction. Survives reset() — the compress
 *                          output is freed by reset() but the decompress output is not.
 *
 *   caller-owned           setPoolManagedDecompOutput(false)
 *                          decompress() returns a fresh cudaMalloc'd pointer.
 *                          Caller MUST cudaFree when done.
 *
 * No external data files required.
 *
 * Usage:
 *   ./build/bin/ownership_example
 */

#include "fzgpumodules.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

using namespace fz;

static constexpr size_t N           = 1 << 18;  // 256 K floats = 1 MB
static constexpr float  ERROR_BOUND = 1e-3f;

static std::vector<float> make_data(size_t n) {
    std::vector<float> h(n);
    for (size_t i = 0; i < n; ++i) {
        const float t = static_cast<float>(i) / static_cast<float>(n);
        h[i] = std::sin(2.0f * 3.14159265f * t)
             + 0.5f * std::cos(6.0f * 3.14159265f * t);
    }
    return h;
}

static void build_pipeline(Pipeline& p) {
    auto* lorenzo = p.addStage<LorenzoQuantStage<float, uint16_t>>();
    lorenzo->setErrorBound(ERROR_BOUND);
    lorenzo->setErrorBoundMode(ErrorBoundMode::REL);
    auto* rle = p.addStage<RLEStage<uint16_t>>();
    p.connect(rle, lorenzo, "codes");
    p.finalize();
}

// Copies d_rec back to host and prints the max absolute error.
static void print_error(const std::vector<float>& h_orig, const void* d_rec, size_t bytes) {
    std::vector<float> h_rec(bytes / sizeof(float));
    cudaMemcpy(h_rec.data(), d_rec, bytes, cudaMemcpyDeviceToHost);
    float max_err = 0.0f;
    for (size_t i = 0; i < h_rec.size(); ++i)
        max_err = std::max(max_err, std::abs(h_rec[i] - h_orig[i]));
    std::printf("  max_abs_error = %.6f\n", max_err);
}

// ── Main ──────────────────────────────────────────────────────────────────────

int main() {
    const std::vector<float> h_data = make_data(N);
    const size_t data_bytes = N * sizeof(float);

    float* d_input = nullptr;
    cudaMalloc(&d_input, data_bytes);
    cudaMemcpy(d_input, h_data.data(), data_bytes, cudaMemcpyHostToDevice);

    // ── Section 1: Pool-owned compress output ─────────────────────────────────
    //
    // compress() returns a pointer into the internal pool.
    // The pipeline owns this buffer — do NOT cudaFree it.
    //
    // The pointer is valid until the NEXT event, whichever comes first:
    //   - another compress() call
    //   - reset()
    //   - Pipeline destruction
    //
    // Pattern: use the compressed data (write to file, send over network, etc.)
    // BEFORE calling compress() again on the same pipeline.
    std::printf("── Section 1: pool-owned compress output ─────────────────────────────\n");
    {
        Pipeline p(data_bytes);
        build_pipeline(p);

        void*  d_comp1  = nullptr;
        size_t comp_sz1 = 0;
        p.compress(d_input, data_bytes, &d_comp1, &comp_sz1);
        cudaDeviceSynchronize();
        std::printf("  compress #1: %zu bytes at %p\n", comp_sz1, d_comp1);
        // Use d_comp1 here (e.g., write to file, send over network).
        // Do NOT cudaFree(d_comp1).

        // Calling compress() again invalidates d_comp1.
        void*  d_comp2  = nullptr;
        size_t comp_sz2 = 0;
        p.compress(d_input, data_bytes, &d_comp2, &comp_sz2);
        cudaDeviceSynchronize();
        std::printf("  compress #2: %zu bytes at %p\n", comp_sz2, d_comp2);
        // d_comp1 is now invalid — the pool reclaimed it. d_comp2 is the live pointer.

        // Decompress from the live pool buffer to verify.
        void*  d_dec  = nullptr;
        size_t dec_sz = 0;
        p.decompress(d_comp2, comp_sz2, &d_dec, &dec_sz);
        cudaDeviceSynchronize();
        print_error(h_data, d_dec, dec_sz);
        // d_dec is pool-owned (default) — do NOT cudaFree.
    }

    // ── Section 2: Caller-provided compress buffer ────────────────────────────
    //
    // Use this when you need the compressed data to outlive a subsequent
    // compress() or reset(), or when you want to manage the buffer yourself
    // (e.g., a pre-allocated staging area, a pinned host mirror, etc.).
    //
    // Steps:
    //   1. getMaxCompressedSize(input_bytes) — tight worst-case upper bound.
    //   2. cudaMalloc your own buffer.
    //   3. Call the caller-provided compress() overload.
    //   4. cudaFree when you are done with the compressed data.
    std::printf("\n── Section 2: caller-provided compress buffer ─────────────────────────\n");
    {
        Pipeline p(data_bytes);
        build_pipeline(p);

        const size_t max_comp = p.getMaxCompressedSize(data_bytes);
        std::printf("  getMaxCompressedSize: %zu bytes\n", max_comp);

        // Caller allocates and owns this buffer for its entire lifetime.
        void* d_comp_buf = nullptr;
        cudaMalloc(&d_comp_buf, max_comp);

        size_t actual_sz = 0;
        p.compress(d_input, data_bytes, d_comp_buf, max_comp, &actual_sz);
        cudaDeviceSynchronize();
        std::printf("  actual compressed:    %zu bytes (%.2fx ratio)\n",
                    actual_sz, static_cast<double>(data_bytes) / actual_sz);

        // d_comp_buf holds the compressed data and survives any number of
        // subsequent compress() calls or reset() calls on p.

        // Compress again into a second independent buffer — d_comp_buf is unaffected.
        void* d_comp_buf2 = nullptr;
        cudaMalloc(&d_comp_buf2, max_comp);
        size_t actual_sz2 = 0;
        p.compress(d_input, data_bytes, d_comp_buf2, max_comp, &actual_sz2);
        cudaDeviceSynchronize();
        std::printf("  second compress:      %zu bytes  (d_comp_buf still valid)\n", actual_sz2);

        // Decompress the first buffer — it is still intact.
        void*  d_dec  = nullptr;
        size_t dec_sz = 0;
        p.decompress(d_comp_buf, actual_sz, &d_dec, &dec_sz);
        cudaDeviceSynchronize();
        std::printf("  decompress from first buffer:\n");
        print_error(h_data, d_dec, dec_sz);

        cudaFree(d_comp_buf);   // required: caller-provided buffers must be freed
        cudaFree(d_comp_buf2);
    }

    // ── Section 3: Pool-owned decompress output (default) ────────────────────
    //
    // setPoolManagedDecompOutput(true) is the default.
    // decompress() returns a pool-owned pointer — do NOT cudaFree.
    //
    // Lifetime rules:
    //   - Valid until the NEXT decompress() call — a second call invalidates the first.
    //   - Valid across reset() — reset() frees the compress output but NOT the
    //     decompress output. This is intentional: you may want to inspect the
    //     decompressed data after freeing forward-DAG memory.
    //   - Freed on Pipeline destruction.
    std::printf("\n── Section 3: pool-owned decompress output (default) ─────────────────\n");
    {
        Pipeline p(data_bytes);
        build_pipeline(p);

        // setPoolManagedDecompOutput(true) is the default; shown here for clarity.
        p.setPoolManagedDecompOutput(true);

        void*  d_comp  = nullptr;
        size_t comp_sz = 0;
        p.compress(d_input, data_bytes, &d_comp, &comp_sz);
        cudaDeviceSynchronize();

        void*  d_dec1  = nullptr;
        size_t dec_sz1 = 0;
        p.decompress(d_comp, comp_sz, &d_dec1, &dec_sz1);
        cudaDeviceSynchronize();
        std::printf("  decompress #1: %zu bytes at %p\n", dec_sz1, d_dec1);

        // reset() frees the compress output — d_comp is now invalid.
        // But d_dec1 is still valid.
        p.reset();
        std::printf("  after reset(): d_comp invalid, d_dec1 still valid\n");

        // Re-verify that d_dec1 is still readable after reset().
        print_error(h_data, d_dec1, dec_sz1);

        // A second decompress() invalidates d_dec1 (pool reclaims it).
        p.compress(d_input, data_bytes, &d_comp, &comp_sz);
        cudaDeviceSynchronize();
        void*  d_dec2  = nullptr;
        size_t dec_sz2 = 0;
        p.decompress(d_comp, comp_sz, &d_dec2, &dec_sz2);
        cudaDeviceSynchronize();
        std::printf("  decompress #2: %zu bytes at %p  (d_dec1 now invalid)\n",
                    dec_sz2, d_dec2);
        // Do NOT cudaFree d_dec1 or d_dec2 — pool-owned.
    }

    // ── Section 4: Caller-owned decompress output ─────────────────────────────
    //
    // setPoolManagedDecompOutput(false) makes decompress() return a fresh
    // cudaMalloc'd pointer on every call. Use this when you need the decompressed
    // data to outlive the pipeline, or when you want explicit ownership.
    // Caller MUST cudaFree every returned pointer.
    std::printf("\n── Section 4: caller-owned decompress output ─────────────────────────\n");
    {
        Pipeline p(data_bytes);
        build_pipeline(p);
        p.setPoolManagedDecompOutput(false);

        void*  d_comp  = nullptr;
        size_t comp_sz = 0;
        p.compress(d_input, data_bytes, &d_comp, &comp_sz);
        cudaDeviceSynchronize();

        void*  d_dec  = nullptr;
        size_t dec_sz = 0;
        p.decompress(d_comp, comp_sz, &d_dec, &dec_sz);
        cudaDeviceSynchronize();
        std::printf("  decompress: %zu bytes at %p\n", dec_sz, d_dec);
        print_error(h_data, d_dec, dec_sz);

        // Caller owns d_dec — must cudaFree when done.
        cudaFree(d_dec);
        std::printf("  cudaFree'd successfully\n");

        // Each subsequent decompress() returns a new independent pointer.
        void*  d_dec2  = nullptr;
        size_t dec_sz2 = 0;
        p.decompress(d_comp, comp_sz, &d_dec2, &dec_sz2);
        cudaDeviceSynchronize();
        cudaFree(d_dec2);  // each call must be individually freed
        std::printf("  second decompress cudaFree'd successfully\n");
    }

    cudaFree(d_input);
    std::printf("\nDone.\n");
    return 0;
}
