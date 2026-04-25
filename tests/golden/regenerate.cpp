/**
 * tests/golden/regenerate.cpp
 *
 * Standalone program that generates the committed golden reference FZM files
 * in the tests/golden/ directory.
 *
 * Build + run via the CMake target:
 *
 *   cmake --preset release
 *   cmake --build --preset release --target generate_golden_files
 *
 * The program overwrites any existing golden files.  Run this whenever
 * FZM_VERSION is bumped or a stage's serialization format changes, then
 * commit the updated files alongside the code change.
 *
 * Input data is fixed (deterministic sinusoid seed) so the same golden file
 * is produced on any machine.
 */

#include "fzgpumodules.h"
#include "fzm_format.h"
#include "predictors/lorenzo/lorenzo.h"
#include "transforms/bitshuffle/bitshuffle_stage.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace fz;

// ─────────────────────────────────────────────────────────────────────────────
// Parameters — must match test_golden.cpp exactly.
// ─────────────────────────────────────────────────────────────────────────────
static constexpr size_t N          = 1 << 12;   // 4096 floats
static constexpr float  EB         = 1e-2f;
static constexpr int    QRAD       = 512;
static constexpr float  OUTLIER_CAP= 0.2f;

// ─────────────────────────────────────────────────────────────────────────────
// Generate the same deterministic smooth input used by test_golden.cpp.
// Two-component sinusoid identical to make_smooth_data<float>(N).
// ─────────────────────────────────────────────────────────────────────────────
static std::vector<float> make_input() {
    std::vector<float> v(N);
    for (size_t i = 0; i < N; i++) {
        v[i] = static_cast<float>(
            50.0 * std::sin(static_cast<double>(i) * 0.01)
          + 20.0 * std::cos(static_cast<double>(i) * 0.003));
    }
    return v;
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    // Accept an optional output directory override; default = tests/golden/ relative
    // to the working directory.  CMake target sets this to the source-tree path.
    const std::string out_dir = (argc > 1) ? argv[1] : "tests/golden";
    const std::string out_path = out_dir + "/ref_lorenzo_bitshuffle.fzm";

    cudaStream_t stream = 0;

    // ── Build input ─────────────────────────────────────────────────────────
    auto h_input = make_input();
    const size_t in_bytes = N * sizeof(float);

    float* d_in = nullptr;
    cudaMalloc(&d_in, in_bytes);
    cudaMemcpy(d_in, h_input.data(), in_bytes, cudaMemcpyHostToDevice);

    // ── Build pipeline: Lorenzo<float,uint16> → Bitshuffle ──────────────────
    Pipeline pipeline(in_bytes, MemoryStrategy::MINIMAL);
    auto* lrz = pipeline.addStage<LorenzoQuantizerStage<float, uint16_t>>();
    lrz->setErrorBound(EB);
    lrz->setQuantRadius(QRAD);
    lrz->setOutlierCapacity(OUTLIER_CAP);

    auto* bs = pipeline.addStage<BitshuffleStage>();
    pipeline.connect(bs, lrz, "codes");

    pipeline.setPoolManagedDecompOutput(false);
    pipeline.finalize();

    // ── Compress ─────────────────────────────────────────────────────────────
    void*  d_comp  = nullptr;
    size_t comp_sz = 0;
    pipeline.compress(d_in, in_bytes, &d_comp, &comp_sz, stream);
    cudaStreamSynchronize(stream);

    // ── Write golden file ────────────────────────────────────────────────────
    pipeline.writeToFile(out_path, stream);
    cudaStreamSynchronize(stream);

    // ── Print summary ────────────────────────────────────────────────────────
    auto hdr = Pipeline::readHeader(out_path);
    std::printf("Generated: %s\n", out_path.c_str());
    std::printf("  FZM version:    %u.%u\n",
        (unsigned)(hdr.core.version >> 8), (unsigned)(hdr.core.version & 0xFF));
    std::printf("  Uncompressed:   %zu bytes (%.1f KB)\n",
        (size_t)hdr.core.uncompressed_size,
        hdr.core.uncompressed_size / 1024.0);
    std::printf("  Compressed:     %zu bytes (%.1f KB)  ratio=%.2fx\n",
        (size_t)hdr.core.compressed_size,
        hdr.core.compressed_size / 1024.0,
        static_cast<double>(hdr.core.uncompressed_size) / hdr.core.compressed_size);
    std::printf("  Stages:         %u\n",  (unsigned)hdr.core.num_stages);
    std::printf("  Buffers:        %u\n",  (unsigned)hdr.core.num_buffers);
    std::printf("  Checksums:      data=0x%08X  header=0x%08X\n",
        hdr.core.data_checksum, hdr.core.header_checksum);

    cudaFree(d_in);
    return 0;
}
