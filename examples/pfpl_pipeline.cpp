/**
 * PFPL Pipeline Example
 *
 * Full replication of the PFPL (Parallelized Floating-Point Lossy) compression
 * pipeline on synthetic smooth floating-point data.
 *
 * Pipeline chain (16 KB chunks throughout):
 *
 *   QuantizerStage<float, uint32_t>      — REL mode log2-space quantization
 *     → DifferenceStage<int32_t, uint32_t> — chunked first-difference, negabinary output
 *       → BitshuffleStage                  — 32-bit LC butterfly bit-plane transpose
 *         → RZEStage                       — 4-level recursive zero-byte elimination
 *
 * Rationale for each stage transition:
 *
 *   Quantizer (REL):
 *     Converts each float to a log2-domain bin index (uint32_t, bit-packed with sign
 *     and log-bin magnitude). Smooth data → slowly varying bin indices.
 *
 *   Difference (negabinary fused):
 *     Consecutive bins in smooth data differ by small signed integers. Differencing
 *     concentrates these small values near zero. Negabinary encoding (base -2)
 *     distributes the sign bit across low bit planes; after bitshuffle the high
 *     bit planes are nearly all-zero, which RZE eliminates very efficiently.
 *
 *   Bitshuffle:
 *     Transposes the NxW bit matrix (N elements, W=32 bits each) so that bit-plane k
 *     contains the k-th bit of all N elements. Concentrates zero bytes/words into
 *     contiguous regions, maximizing RZE's zero-byte elimination rate.
 *
 *   RZE:
 *     4 rounds of "emit non-zero bytes + bitmap"; the bitmap is then compressed
 *     recursively. All-zero or near-zero chunks reduce to a tiny bitmap + header.
 *
 * Usage:
 *   ./build/bin/pfpl_example [error_bound]
 *   ./build/bin/pfpl_example 1e-3
 */

#include "fzgpumodules.h"
#include "pipeline/stat.h"

#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

using namespace fz;

// ── Configuration ─────────────────────────────────────────────────────────────

// CLDHGH: CESM ATM 1800×3600 cloud-top high-level fraction field (≈24.75 MB)
static const char*      CLDHGH_PATH    = "/home/skyler/data/SDRB/CESM_ATM_1800x3600/CLDHGH.f32";
static constexpr size_t CLDHGH_DIM_X  = 3600;
static constexpr size_t CLDHGH_DIM_Y  = 1800;

static constexpr float  DEFAULT_EB = 1e-3f;   // default relative error bound
static constexpr size_t CHUNK      = 16384;    // 16 KB = 4096 × float32

// ── Data loading ──────────────────────────────────────────────────────────────

// Try to load CLDHGH.f32 from disk.  Returns {nullptr, 0} if the file is missing.
static std::pair<float*, size_t> load_cldhgh() {
    const size_t n = CLDHGH_DIM_X * CLDHGH_DIM_Y;
    std::vector<float> h(n);
    std::FILE* fp = std::fopen(CLDHGH_PATH, "rb");
    if (!fp) return {nullptr, 0};
    const size_t read_n = std::fread(h.data(), sizeof(float), n, fp);
    std::fclose(fp);
    if (read_n != n) return {nullptr, 0};
    float* d = nullptr;
    cudaMalloc(&d, n * sizeof(float));
    cudaMemcpy(d, h.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    return {d, n};
}

// Fallback: 2D smooth cosine–sine field mimicking a climate variable.
static float* make_smooth_data(size_t n) {
    const size_t NX = static_cast<size_t>(std::sqrt(static_cast<double>(n)));
    const size_t NY = (n + NX - 1) / NX;
    std::vector<float> h(n);
    for (size_t iy = 0; iy < NY && iy * NX < n; ++iy) {
        const float y = static_cast<float>(iy) / NY;
        for (size_t ix = 0; ix < NX && iy * NX + ix < n; ++ix) {
            const float x = static_cast<float>(ix) / NX;
            h[iy * NX + ix] =
                std::cos(2.0f * static_cast<float>(M_PI) * 3.0f * x) *
                std::sin(2.0f * static_cast<float>(M_PI) * 2.0f * y) * 80.0f
                + std::cos(2.0f * static_cast<float>(M_PI) * 7.0f * x +
                           2.0f * static_cast<float>(M_PI) * 5.0f * y) * 20.0f
                + 200.0f;
        }
    }
    float* d = nullptr;
    cudaMalloc(&d, n * sizeof(float));
    cudaMemcpy(d, h.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    return d;
}

// ── Reporting helpers ─────────────────────────────────────────────────────────

static void print_compression_stats(size_t raw_bytes, size_t comp_bytes) {
    std::cout << std::fixed;
    std::cout << "  Raw bytes:    " << std::setw(12) << raw_bytes
              << "  (" << std::setprecision(2) << raw_bytes  / (1024.0 * 1024.0) << " MB)\n";
    std::cout << "  Compressed:   " << std::setw(12) << comp_bytes
              << "  (" << std::setprecision(2) << comp_bytes / (1024.0 * 1024.0) << " MB)\n";
    std::cout << "  Ratio:        "
              << std::setprecision(2) << static_cast<double>(raw_bytes) / comp_bytes << "x\n";
}

static void print_quality(const ReconstructionStats& s, float eb) {
    const int W = 12;
    std::cout << std::fixed
              << "  PSNR:         " << std::setw(W) << std::setprecision(2) << s.psnr    << " dB\n"
              << "  Max |error|:  " << std::setw(W) << std::setprecision(6) << s.max_error
              << "  (EB=" << eb << ")\n"
              << "  NRMSE:        " << std::setw(W) << std::setprecision(6) << s.nrmse   << "\n"
              << "  Value range:  " << std::setw(W) << std::setprecision(3) << s.value_range << "\n";
}

// ── PFPL pipeline construction ────────────────────────────────────────────────

static void build_pfpl_pipeline(Pipeline& p, float eb, ErrorBoundMode mode = ErrorBoundMode::REL,
                                 float threshold = std::numeric_limits<float>::infinity()) {
    // ── Stage 1: Quantizer ────────────────────────────────────────────────
    // Three quantization modes are supported:
    //
    //   REL — log2-space quantization: bin = round(log2|x| / log2(1+eb)).
    //         The code range is bounded by the number of float32 exponent
    //         values (~250 for eb=1e-3), entirely independent of data range.
    //         quant_radius=32768 has ample headroom for any float32 data.
    //
    //   ABS — uniform quantization: bin = round(x / (2*eb)).
    //         Max bin = max|x| / (2*eb). For data in [100,300] at eb=1e-3,
    //         max bin ≈ 150,000.  quant_radius must be larger; we use 1<<22
    //         (4,194,304) matching the PFPL reference implementation.
    //
    //   NOA — norm-of-absolute: abs_eb = user_eb × (max−min), computed via
    //         a GPU thrust scan before the forward kernel.  After scaling,
    //         max bin = max|x| / (2*abs_eb) ≈ max|x| / (2*eb*(max−min)) ≤ 0.5/eb.
    //         For eb=1e-3: max bin ≤ 500, well within quant_radius=32768.
    //
    // setZigzagCodes(true) applies TCMS (two's-complement → magnitude+sign,
    // i.e. zigzag encoding) to the bin index: signed q → 2q if q≥0, −2q−1 if q<0.
    // This maps small positive and negative bins to small unsigned integers,
    // so consecutive differences stay small and the Diff→Bitshuffle→RZE chain
    // compresses well.  The PFPL reference does the same ((bin<<1)^(bin>>31)).
    auto* quant = p.addStage<QuantizerStage<float, uint32_t>>();
    quant->setErrorBound(eb);
    quant->setErrorBoundMode(mode);
    // quant_radius: REL/NOA use 32768 (log-space / range-normalized bins are small);
    // ABS needs 1<<22 because raw bins can be as large as max|x|/(2*eb).
    quant->setQuantRadius(mode == ErrorBoundMode::ABS ? (1 << 22) : 32768);
    quant->setOutlierCapacity(0.05f);  // used only when inplace_outliers=false
    quant->setZigzagCodes(true);   // TCMS: map signed bins to small unsigned values
    if (std::isfinite(threshold))
        quant->setOutlierThreshold(threshold);  // |x| >= threshold → forced outlier (all modes)
    if (mode != ErrorBoundMode::REL) {
        // In-place outlier encoding (LC-reference style): raw float bits written
        // directly into the codes array; decompressor uses (code>>1)>=radius check.
        // Eliminates the separate outlier_vals/idxs/count buffers and scatter pass.
        quant->setInplaceOutliers(true);
    }

    // ── Stage 2: Difference (chunked, negabinary-fused) ───────────────────
    // DifferenceStage<int32_t, uint32_t>:
    //   forward:  diff[i] = codes[i] - codes[i-1]  (as int32)
    //             output[i] = Negabinary::encode(diff[i])  (uint32)
    //   inverse:  diff[i] = Negabinary::decode(output[i])
    //             codes[i] = cumsum(diff)
    //
    // Chunk size = 16 KB: each chunk is an independent difference context.
    // codes[] is uint32_t from the quantizer; DifferenceStage reinterprets it
    // as int32_t (same bit width) for the signed difference.  This is safe and
    // matches the PFPL reference implementation.
    auto* diff = p.addStage<DifferenceStage<int32_t, uint32_t>>();
    diff->setChunkSize(CHUNK);
    p.connect(diff, quant, "codes");

    // ── Stage 3: Bitshuffle (32-bit LC butterfly) ─────────────────────────
    // Transposes the 4096×32 bit matrix for each 16 KB chunk.
    // After differencing + negabinary, the high bit planes of smooth data are
    // nearly all-zero.  Bitshuffle gathers these zero bits into contiguous
    // zero bytes that RZE can eliminate in bulk.
    auto* bitshuffle = p.addStage<BitshuffleStage>();
    bitshuffle->setBlockSize(CHUNK);
    bitshuffle->setElementWidth(4);  // 4-byte (uint32_t) elements
    p.connect(bitshuffle, diff);

    // ── Stage 4: RZE (4-level recursive zero-byte elimination) ────────────
    // Each 16 KB chunk is processed independently by one CUDA block.
    // Round 1: build 2048-byte bitmap of non-zero bytes, emit non-zero bytes.
    // Round 2: apply same to the 2048-byte bitmap → 256-byte bitmap + non-zeros.
    // Round 3: 256 → 32-byte bitmap.
    // Round 4: 32 → 4-byte bitmap (stored raw, no further recursion).
    // All-zero chunks produce a 2-byte sentinel (very high compression).
    auto* rze = p.addStage<RZEStage>();
    rze->setChunkSize(CHUNK);
    rze->setLevels(4);
    p.connect(rze, bitshuffle);

    p.finalize();
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    float eb = DEFAULT_EB;
    ErrorBoundMode mode = ErrorBoundMode::REL;
    std::string mode_str = "rel";

    if (argc > 1) {
        eb = std::stof(argv[1]);
        if (eb <= 0.0f || eb >= 1.0f) {
            std::cerr << "Usage: pfpl_example [error_bound [mode]]\n"
                      << "  error_bound: 0 < eb < 1  (e.g. 1e-3)\n"
                      << "  mode:        rel | abs | noa  (default: rel)\n";
            return 1;
        }
    }
    if (argc > 2) {
        mode_str = argv[2];
        if      (mode_str == "abs") mode = ErrorBoundMode::ABS;
        else if (mode_str == "noa") mode = ErrorBoundMode::NOA;
        else if (mode_str == "rel") mode = ErrorBoundMode::REL;
        else {
            std::cerr << "Unknown mode '" << mode_str << "'. Use: rel | abs | noa\n";
            return 1;
        }
    }
    float threshold = std::numeric_limits<float>::infinity();
    if (argc > 3) {
        threshold = std::stof(argv[3]);
        if (threshold <= 0.0f) {
            std::cerr << "threshold must be a positive float (e.g. 1e6)\n";
            return 1;
        }
    }

    // ── Load data (CLDHGH real dataset or synthetic fallback) ─────────────
    float*      d_input    = nullptr;
    size_t      n          = 0;       // original element count (for quality/ratio)
    std::string data_label;

    auto [d_cldhgh, cldhgh_n] = load_cldhgh();
    if (d_cldhgh) {
        d_input    = d_cldhgh;
        n          = cldhgh_n;
        data_label = "CESM ATM CLDHGH (1800×3600)";
    } else {
        std::cout << "  [note] CLDHGH not found at " << CLDHGH_PATH
                  << "; using synthetic 1M-element fallback.\n";
        n          = 1024 * 1024;
        d_input    = make_smooth_data(n);
        data_label = "Synthetic cosine-sine field";
    }

    const size_t data_bytes = n * sizeof(float);

    std::cout << "=== PFPL Pipeline Example ===\n\n"
              << "  Dataset:     " << data_label << "\n"
              << "  Elements:    " << n << "\n"
              << "  Raw size:    " << std::fixed << std::setprecision(2)
              << data_bytes / (1024.0 * 1024.0) << " MB\n"
              << "  Error bound: " << std::scientific << std::setprecision(1) << eb
              << " (" << mode_str << ")\n";
    if (std::isfinite(threshold))
        std::cout << "  Threshold:   " << threshold
                  << "  (|x|>=threshold → lossless outlier)\n";
    std::cout << "  Chunk size:  " << CHUNK << " bytes\n\n"
              << std::fixed;

    // ── Construct the PFPL pipeline ───────────────────────────────────────
    Pipeline comp(data_bytes, MemoryStrategy::PREALLOCATE, 3.0f);
    build_pfpl_pipeline(comp, eb, mode, threshold);
    comp.enableProfiling(true);

    // ── Compress ──────────────────────────────────────────────────────────
    std::cout << "── Compress ──────────────────────────────────────────────\n";
    void*  d_compressed  = nullptr;
    size_t compressed_sz = 0;
    comp.compress(d_input, data_bytes, &d_compressed, &compressed_sz, /*stream=*/0);
    cudaDeviceSynchronize();
    print_compression_stats(data_bytes, compressed_sz);
    std::cout << "\n";
    comp.getLastPerfResult().print(std::cout);

    // ── Decompress ────────────────────────────────────────────────────────
    std::cout << "\n── Decompress ────────────────────────────────────────────\n";
    void*  d_reconstructed  = nullptr;
    size_t reconstructed_sz = 0;
    comp.decompress(d_compressed, compressed_sz,
                    &d_reconstructed, &reconstructed_sz, /*stream=*/0);
    cudaDeviceSynchronize();
    std::cout << "  Reconstructed: " << std::fixed << std::setprecision(2)
              << reconstructed_sz / (1024.0 * 1024.0) << " MB\n\n";
    comp.getLastPerfResult().print(std::cout);

    // ── Verify quality ────────────────────────────────────────────────────
    std::cout << "\n── Quality ───────────────────────────────────────────────\n";
    if (!d_reconstructed || reconstructed_sz != data_bytes) {
        std::cerr << "  [FAIL] Size mismatch — expected " << data_bytes
                  << " bytes, got " << reconstructed_sz << "\n";
        cudaFree(d_input);
        return 1;
    }

    auto stats = calculateStatistics<float>(
        d_input,
        static_cast<const float*>(d_reconstructed),
        n
    );
    print_quality(stats, eb);

    // REL: per-element |x̂-x|/|x| ≤ eb.  ABS: |x̂-x| ≤ eb.  NOA: |x̂-x| ≤ eb×range.
    // All modes are enforced by the quantizer; we just sanity-check PSNR here.
    const bool ok = (stats.psnr > 20.0);
    std::cout << "\n  Status: " << (ok ? "PASS" : "WARN (low PSNR)") << "\n";

    // ── Cleanup ───────────────────────────────────────────────────────────
    if (d_reconstructed) cudaFree(d_reconstructed);
    cudaFree(d_input);

    std::cout << "\nDone.\n";
    return ok ? 0 : 1;
}
