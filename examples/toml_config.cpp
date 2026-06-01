/**
 * examples/toml_config.cpp
 *
 * Pipeline configuration via TOML files.
 *
 * Demonstrates:
 *   - Pipeline::loadConfig(): load a preset TOML, wire stages, finalize
 *   - Pipeline::saveConfig(): serialize a finalized pipeline to TOML
 *   - Round-trip: save → load → compress → identical results
 *   - Pipeline::readHeader(): inspect a .fzm file without decompressing
 *
 * The recommended pattern for using a preset:
 *
 *     Pipeline p(data_bytes);   // sets pool size for PREALLOCATE strategy
 *     p.loadConfig("examples/presets/cusz.toml");
 *     p.compress(...);
 *
 * Do NOT use the single-argument Pipeline(path) constructor with presets that
 * specify memory_strategy = "PREALLOCATE" — it passes input_size = 0 and the
 * pool will be empty. Use Pipeline(data_bytes) + loadConfig() instead.
 *
 * Preset files live in examples/presets/:
 *   cusz.toml                     LorenzoQuant → Huffman (cuSZ-style)
 *   pfpl.toml                     Quantizer → Difference → Bitshuffle → RZE
 *   speed.toml                    Quantizer → Bitshuffle → RZE
 *   lorenzo_bitpack.toml          LorenzoQuant → Bitpack
 *   quantizer_lorenzo_bitpack.toml  Quantizer + Lorenzo → Bitpack
 *
 * No external data files required — uses synthetic float data.
 *
 * Usage:
 *   ./build/release/bin/examples/toml_config [preset_dir]
 *   Default preset_dir: examples/presets
 *
 * Build:
 *   cmake --preset release -DBUILD_EXAMPLES=ON && cmake --build build/release -j$(nproc)
 *   Binary: build/release/bin/examples/toml_config
 */

#include "fzgpumodules.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace fz;

static constexpr size_t N = 1 << 20;  // 1 M floats = 4 MB

static std::vector<float> make_data(size_t n) {
    std::vector<float> h(n);
    for (size_t i = 0; i < n; ++i) {
        const float t = static_cast<float>(i) / static_cast<float>(n);
        h[i] = std::sin(2.0f * 3.14159265f * t)
             + 0.5f * std::cos(6.0f * 3.14159265f * t);
    }
    return h;
}

static float max_abs_error(const std::vector<float>& orig, const void* d_rec, size_t bytes) {
    std::vector<float> h_rec(bytes / sizeof(float));
    cudaMemcpy(h_rec.data(), d_rec, bytes, cudaMemcpyDeviceToHost);
    float err = 0.0f;
    for (size_t i = 0; i < h_rec.size(); ++i)
        err = std::max(err, std::abs(h_rec[i] - orig[i]));
    return err;
}

int main(int argc, char* argv[]) {
    const std::string preset_dir = (argc > 1) ? argv[1] : "examples/presets";
    const std::string cusz_toml  = preset_dir + "/cusz.toml";
    const std::string saved_toml = "/tmp/fzgmod_toml_example.toml";
    const std::string saved_fzm  = "/tmp/fzgmod_toml_example.fzm";

    const std::vector<float> h_data = make_data(N);
    const size_t data_bytes = N * sizeof(float);

    float* d_input = nullptr;
    cudaMalloc(&d_input, data_bytes);
    cudaMemcpy(d_input, h_data.data(), data_bytes, cudaMemcpyHostToDevice);

    // ── Section 1: Load a preset file ────────────────────────────────────────
    //
    // loadConfig() reads the TOML, adds stages with the saved config, wires them,
    // and calls finalize() internally.
    //
    // Pass data_bytes to the constructor so the pool is correctly sized for
    // PREALLOCATE strategy.  loadConfig() then applies the TOML's strategy and
    // stage topology on top.
    //
    // NOA error-bound mode (used in cusz.toml) scans the input data range
    // automatically during compress() — no extra setup needed.
    std::printf("── Section 1: loadConfig() from preset  ─────────────────────────────\n");
    {
        Pipeline p(data_bytes);
        p.loadConfig(cusz_toml);

        void*  d_comp  = nullptr;
        size_t comp_sz = 0;
        p.compress(d_input, data_bytes, &d_comp, &comp_sz);
        cudaDeviceSynchronize();

        void*  d_dec  = nullptr;
        size_t dec_sz = 0;
        p.decompress(d_comp, comp_sz, &d_dec, &dec_sz);
        cudaDeviceSynchronize();

        std::printf("  preset:  %s\n", cusz_toml.c_str());
        std::printf("  ratio:   %.2fx\n",
                    static_cast<double>(data_bytes) / static_cast<double>(comp_sz));
        std::printf("  max_err: %.2e\n", max_abs_error(h_data, d_dec, dec_sz));

        // Save the config and compressed data for later sections.
        p.saveConfig(saved_toml);
        p.writeToFile(saved_fzm);
        std::printf("  config saved → %s\n", saved_toml.c_str());
    }

    // ── Section 2: Inspect the saved TOML ────────────────────────────────────
    //
    // saveConfig() serializes the full pipeline topology: stage types, config
    // parameters (error bound, quant radius, bklen, …), and the wiring between
    // them.  The file can be passed back to loadConfig() to reconstruct the
    // pipeline exactly.
    std::printf("\n── Section 2: inspect saved TOML  ───────────────────────────────────\n");
    {
        std::ifstream f(saved_toml);
        std::ostringstream buf;
        buf << f.rdbuf();
        std::printf("%s\n", buf.str().c_str());
    }

    // ── Section 3: Round-trip — load the saved config and compress again ──────
    //
    // The reconstructed pipeline should produce identical compressed size and
    // error statistics, confirming that saveConfig → loadConfig is lossless.
    //
    // Pipeline-level options that do not affect finalization (profiling, bounds
    // check) should be set AFTER loadConfig(), since loadConfig() runs an
    // internal warmup pass that would overwrite the profiling result otherwise.
    // Options that must precede finalize (setDims) go before loadConfig().
    std::printf("── Section 3: loadConfig() round-trip  ──────────────────────────────\n");
    {
        Pipeline p2(data_bytes);
        p2.loadConfig(saved_toml);  // finalizes internally
        p2.enableProfiling(true);

        void*  d_comp  = nullptr;
        size_t comp_sz = 0;
        p2.compress(d_input, data_bytes, &d_comp, &comp_sz);
        cudaDeviceSynchronize();
        const auto comp_perf = p2.getLastPerfResult();

        void*  d_dec  = nullptr;
        size_t dec_sz = 0;
        p2.decompress(d_comp, comp_sz, &d_dec, &dec_sz);
        cudaDeviceSynchronize();

        std::printf("  loaded:  %s\n", saved_toml.c_str());
        std::printf("  ratio:   %.2fx  (should match Section 1)\n",
                    static_cast<double>(data_bytes) / static_cast<double>(comp_sz));
        std::printf("  max_err: %.2e\n", max_abs_error(h_data, d_dec, dec_sz));

        std::printf("\n[compress profiling]\n");
        comp_perf.print(std::cout);
    }

    // ── Section 4: Inspect a .fzm file header ─────────────────────────────────
    //
    // Pipeline::readHeader() parses the binary .fzm file without running
    // decompression.  Use it to check metadata — version, sizes, stage count —
    // before deciding whether to decompress.
    std::printf("\n── Section 4: Pipeline::readHeader()  ───────────────────────────────\n");
    {
        const auto hdr = Pipeline::readHeader(saved_fzm);
        const uint8_t ver_major = (hdr.core.version >> 8) & 0xFF;
        const uint8_t ver_minor =  hdr.core.version       & 0xFF;
        std::printf("  FZM version:   %u.%u\n", ver_major, ver_minor);
        std::printf("  Uncompressed:  %zu bytes (%.2f MB)\n",
                    hdr.core.uncompressed_size, hdr.core.uncompressed_size / 1048576.0);
        std::printf("  Compressed:    %zu bytes (%.2f MB)\n",
                    hdr.core.compressed_size, hdr.core.compressed_size / 1048576.0);
        std::printf("  Stages:        %u\n", hdr.core.num_stages);
        for (size_t i = 0; i < hdr.stages.size(); ++i)
            std::printf("    stage[%zu]: type_id=%-4u  config_bytes=%u\n",
                        i, static_cast<uint16_t>(hdr.stages[i].stage_type),
                        hdr.stages[i].config_size);
    }

    cudaFree(d_input);
    std::printf("\nDone.\n");
    return 0;
}
