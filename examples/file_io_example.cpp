/**
 * file_io_example.cpp — compress to .fzm file and decompress from file.
 *
 * Demonstrates the three decompression paths for FZM files:
 *
 *   1. Pipeline::decompressFromFile()    (static)
 *      One-shot. Reconstructs the pipeline from the file header automatically.
 *      Output is always caller-owned — you must cudaFree(*d_output).
 *
 *   2. Pipeline::readHeader()
 *      Parse the file header without running decompression. Use this to inspect
 *      metadata (compressed size, stage count, etc.) before deciding whether to
 *      decompress, or to compute a custom pool size via pool_override_bytes.
 *
 *   3. pipeline.decompressFromFileInstance()   (instance method)
 *      Same reconstruction logic as the static overload, but output ownership
 *      follows setPoolManagedDecompOutput() on the instance:
 *        true (default) → pool-owned, valid until reset()/destruction, do NOT cudaFree
 *        false          → caller-owned, must cudaFree
 *
 * Ownership quick-reference:
 *   compress()                    pool-owned; valid until next compress()/reset()/destroy
 *   decompress()                  pool-owned by default; caller-owned if
 *                                 setPoolManagedDecompOutput(false)
 *   decompressFromFile()          always caller-owned → cudaFree required
 *   decompressFromFileInstance()  respects setPoolManagedDecompOutput() → pool-owned by default
 *
 * No external data files required — uses synthetic float data.
 *
 * Usage:
 *   ./build/bin/file_io_example [output.fzm]
 *   Default output path: /tmp/fzgmod_file_io_example.fzm
 */

#include "fzgpumodules.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace fz;

static constexpr size_t N            = 1 << 20;  // 1 M floats = 4 MB
static constexpr float  ERROR_BOUND  = 1e-3f;

// ── Helpers ───────────────────────────────────────────────────────────────────

// Two-component sinusoid with values in [-1.5, 1.5] — good compressibility for Lorenzo.
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
    auto* lorenzo = p.addStage<LorenzoStage<float, uint16_t>>();
    lorenzo->setErrorBound(ERROR_BOUND);
    lorenzo->setErrorBoundMode(ErrorBoundMode::REL);

    auto* rle = p.addStage<RLEStage<uint16_t>>();
    p.connect(rle, lorenzo, "codes");

    p.finalize();
}

// Copies d_rec back to host and checks the max absolute error against the input.
static bool verify(const std::vector<float>& h_orig, const void* d_rec, size_t bytes) {
    const size_t n = bytes / sizeof(float);
    std::vector<float> h_rec(n);
    cudaMemcpy(h_rec.data(), d_rec, bytes, cudaMemcpyDeviceToHost);

    float max_err = 0.0f;
    for (size_t i = 0; i < n; ++i)
        max_err = std::max(max_err, std::abs(h_rec[i] - h_orig[i]));

    // For REL error bound 1e-3 on data with range ~3.0, expect max error ~3e-3.
    const bool ok = max_err < 0.05f;
    std::printf("  max_abs_error = %.6f  %s\n", max_err, ok ? "PASS" : "FAIL");
    return ok;
}

// ── Main ──────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    const std::string fzm_path = argc > 1 ? argv[1] : "/tmp/fzgmod_file_io_example.fzm";

    // Upload synthetic data to device.
    const std::vector<float> h_orig = make_data(N);
    const size_t data_bytes = N * sizeof(float);

    float* d_input = nullptr;
    cudaMalloc(&d_input, data_bytes);
    cudaMemcpy(d_input, h_orig.data(), data_bytes, cudaMemcpyHostToDevice);

    // ── Step 1: Compress and write to file ────────────────────────────────────
    //
    // compress() output is pool-owned — do NOT cudaFree it.
    // writeToFile() reads from the live pool buffer and serialises the pipeline
    // config into the header. Calling writeToFile() before compress() throws.
    std::printf("── Step 1: compress() + writeToFile() ────────────────────────────────\n");
    {
        Pipeline comp(data_bytes);
        build_pipeline(comp);

        void*  d_compressed  = nullptr;
        size_t compressed_sz = 0;
        comp.compress(d_input, data_bytes, &d_compressed, &compressed_sz);
        cudaDeviceSynchronize();

        comp.writeToFile(fzm_path);

        std::printf("  Wrote: %s\n", fzm_path.c_str());
        std::printf("  Input:      %zu bytes (%.2f MB)\n",
                    data_bytes, data_bytes / 1048576.0);
        std::printf("  Compressed: %zu bytes (%.2f MB)  ratio = %.2fx\n",
                    compressed_sz, compressed_sz / 1048576.0,
                    static_cast<double>(data_bytes) / static_cast<double>(compressed_sz));
    }
    // Pipeline goes out of scope here — pool freed, d_compressed invalidated.
    // The .fzm file on disk is still valid.

    // ── Step 2: Static decompressFromFile (caller-owned) ─────────────────────
    //
    // The static overload reconstructs the pipeline from the file header, allocates
    // a fresh cudaMalloc'd buffer for the output, and returns it.
    // Ownership is ALWAYS caller — you must cudaFree(*d_output) when done.
    std::printf("\n── Step 2: Pipeline::decompressFromFile()  [static, caller-owned] ───\n");
    {
        void*  d_out  = nullptr;
        size_t out_sz = 0;

        PipelinePerfResult perf;
        Pipeline::decompressFromFile(fzm_path, &d_out, &out_sz, 0, &perf);
        cudaDeviceSynchronize();

        std::printf("  Output: %zu bytes  DAG: %.3f ms\n", out_sz, perf.dag_elapsed_ms);
        const bool ok = verify(h_orig, d_out, out_sz);

        cudaFree(d_out);  // required: static overload always returns caller-owned memory
        if (!ok) { cudaFree(d_input); return 1; }
    }

    // ── Step 3: Inspect header without decompressing ─────────────────────────
    //
    // readHeader() is purely a file read — no GPU work, no pipeline construction.
    // Use it to check version, sizes, and stage configuration before committing
    // to a decompression, or to pre-compute a pool size via pool_override_bytes.
    std::printf("\n── Step 3: Pipeline::readHeader()  [inspect without decompressing] ──\n");
    {
        const auto hdr = Pipeline::readHeader(fzm_path);

        const uint8_t ver_major = (hdr.core.version >> 8) & 0xFF;
        const uint8_t ver_minor =  hdr.core.version       & 0xFF;

        std::printf("  FZM version:       %u.%u\n", ver_major, ver_minor);
        std::printf("  Uncompressed size: %zu bytes (%.2f MB)\n",
                    hdr.core.uncompressed_size,
                    hdr.core.uncompressed_size / 1048576.0);
        std::printf("  Compressed size:   %zu bytes (%.2f MB)\n",
                    hdr.core.compressed_size,
                    hdr.core.compressed_size / 1048576.0);
        std::printf("  Stages:            %u\n", hdr.core.num_stages);
        std::printf("  Intermediate bufs: %u\n", hdr.core.num_buffers);

        for (size_t i = 0; i < hdr.stages.size(); ++i) {
            std::printf("    stage[%zu]: type_id=%-4u  config_bytes=%u\n",
                        i, static_cast<uint16_t>(hdr.stages[i].stage_type),
                        hdr.stages[i].config_size);
        }

        // The pool_override_bytes formula is: C + 2.5 * max_stage_uncompressed + 32 MiB.
        // Callers with unusual pipelines (many stages, large fanout) can use this to
        // pass a custom pool size to decompressFromFile()'s pool_override_bytes param.
        const size_t C            = hdr.core.compressed_size;
        const size_t max_U        = hdr.core.uncompressed_size;
        const size_t pool_formula = C + static_cast<size_t>(2.5 * max_U) + (32u << 20);
        std::printf("  Auto pool formula: %zu bytes (C + 2.5*U + 32 MiB)\n", pool_formula);
    }

    // ── Step 4: Instance overload — pool-owned output ─────────────────────────
    //
    // decompressFromFileInstance() performs the same file reconstruction as the
    // static overload, but the output is pool-owned by the Pipeline instance.
    //
    // Default: setPoolManagedDecompOutput(true) — do NOT cudaFree the output.
    //          The pointer stays valid until the next reset() or Pipeline destruction.
    //
    // To get caller-owned output instead: call setPoolManagedDecompOutput(false).
    // In that case the behaviour is identical to the static overload.
    std::printf("\n── Step 4: pipeline.decompressFromFileInstance()  [pool-owned] ──────\n");
    {
        // The instance just needs a pool large enough to hold the decompressed output.
        // It does not need to be the same pipeline or have the same topology —
        // decompressFromFileInstance() reconstructs the pipeline from the file header.
        Pipeline inst(data_bytes);

        // setPoolManagedDecompOutput(true) is the default; shown here for clarity.
        inst.setPoolManagedDecompOutput(true);

        void*  d_out  = nullptr;
        size_t out_sz = 0;
        inst.decompressFromFileInstance(fzm_path, &d_out, &out_sz);
        cudaDeviceSynchronize();

        std::printf("  Output: %zu bytes\n", out_sz);
        verify(h_orig, d_out, out_sz);

        // d_out is valid here — pool-owned by inst.
        // No cudaFree: inst's destructor frees the pool (and with it, d_out).
    }

    cudaFree(d_input);
    std::printf("\nDone.\n");
    return 0;
}
