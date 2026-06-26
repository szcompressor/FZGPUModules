/**
 * cuSZp variants (1 / 2 / 3) — performance with and without CUDA Graph mode.
 *
 * Builds the three modular cuSZp emulation pipelines and benchmarks each one,
 * reporting compression ratio plus compress/decompress throughput:
 *
 *   cuSZp  (SC'23, plain)   : Quantizer(linear) → Lorenzo(block=32)   → AdaptiveBitpack(32)
 *   cuSZp2 (SC'24, outlier) : Quantizer(linear) → Lorenzo(block=32)   → AdaptiveBitpack(32, outlier)
 *   cuSZp3 (SC'25, plain 2D): Quantizer(linear) → TiledLorenzo(8x8)   → AdaptiveBitpack(64)
 *
 * For each variant we measure two strategies:
 *   PREALLOCATE  — all buffers fixed at finalize() (the fastest non-graph path)
 *   GRAPH        — PREALLOCATE + CUDA Graph capture (attempted)
 *
 * IMPORTANT — graph mode is NOT available for any cuSZp variant.  All three end
 * in AdaptiveBitpackStage, whose isGraphCompatible() is false: its forward pass
 * does one blocking D2H copy to read the variable compressed size (exactly the
 * single per-compress readback the real cuSZp performs for cmpSize).  A CUDA
 * graph cannot contain that host-synchronizing copy, so captureGraph() throws.
 * This example attempts capture anyway and reports the failure verbatim, so the
 * "fastest performance" for these pipelines is the PREALLOCATE column.
 *
 * Usage:
 *   ./build/release/bin/examples/cuszp_variants <file> [dim_x [dim_y [eb [runs]]]]
 *
 * Examples:
 *   ./build/release/bin/examples/cuszp_variants data/CLDHGH.f32 3600 1800
 *   ./build/release/bin/examples/cuszp_variants data/CLDHGH.f32 3600 1800 1e-3 50
 *
 * Build:
 *   cmake --preset release -DBUILD_EXAMPLES=ON && cmake --build build/release -j$(nproc)
 *   Binary: build/release/bin/examples/cuszp_variants
 */

#include "fzgpumodules.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

using namespace fz;

static constexpr float DEFAULT_EB = 1e-3f;
static constexpr int   DEFAULT_RUNS = 50;
static constexpr float POOL_MULT = 3.0f;

// ── Variant description ────────────────────────────────────────────────────────
enum class Variant { CUSZP1, CUSZP2, CUSZP3 };

static const char* variant_name(Variant v) {
    switch (v) {
        case Variant::CUSZP1: return "cuSZp  (SC'23, plain)";
        case Variant::CUSZP2: return "cuSZp2 (SC'24, outlier)";
        case Variant::CUSZP3: return "cuSZp3 (SC'25, plain 2D)";
    }
    return "?";
}

// Build the stage graph for `v` onto `p`.  Does NOT finalize so the caller can
// toggle graph mode first.  All variants are ABS error-bound, so no value_base
// injection is needed (the only graph blocker is AdaptiveBitpack's D2H).
static void build_variant(Pipeline& p, Variant v, float eb,
                          size_t dim_x, size_t dim_y) {
    p.setDims(dim_x, dim_y, 1);

    auto* quant = p.addStage<QuantizerStage<float, uint32_t>>();
    quant->setErrorBound(eb);
    quant->setErrorBoundMode(ErrorBoundMode::ABS);
    quant->setLinearMode(true);   // q = round(x/2eb), signed codes, no outliers

    if (v == Variant::CUSZP3) {
        // Dimension-aware separable delta on 8x8 tiles, tile-major output.
        auto* tl = p.addStage<TiledLorenzoStage<int32_t>>();
        tl->setTileShape(8, 8);
        p.connect(tl, quant, "codes");

        auto* ab = p.addStage<AdaptiveBitpackStage<int32_t>>();
        ab->setBlockSize(64);     // = tile_x * tile_y (one block per tile)
        p.connect(ab, tl);
    } else {
        // 1-D block-local Lorenzo, resets every 32 elements.
        auto* lrz = p.addStage<LorenzoStage<int32_t>>();
        lrz->setBlockSize(32);
        p.connect(lrz, quant, "codes");

        auto* ab = p.addStage<AdaptiveBitpackStage<int32_t>>();
        ab->setBlockSize(32);
        if (v == Variant::CUSZP2) ab->setOutlierSelection(true);  // cuSZp2 mode
        p.connect(ab, lrz);
    }
}

// ── Timing helpers ─────────────────────────────────────────────────────────────
struct Stats {
    double mean = 0, min = 0, max = 0;
};

static Stats summarize(const std::vector<double>& v) {
    Stats s;
    s.mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    s.min  = *std::min_element(v.begin(), v.end());
    s.max  = *std::max_element(v.begin(), v.end());
    return s;
}

static double tput_gbs(size_t bytes, double ms) {
    return static_cast<double>(bytes) / (ms * 1e-3) / 1e9;
}

// ── Per-variant benchmark result ───────────────────────────────────────────────
struct VariantResult {
    Variant     variant;
    size_t      compressed_size = 0;
    double      max_abs_error   = 0;
    Stats       comp_ms;          // PREALLOCATE compress
    Stats       decomp_ms;        // PREALLOCATE decompress
    bool        graph_ok = false; // did graph capture succeed?
    std::string graph_msg;        // failure reason if not
};

// Run the PREALLOCATE compress+decompress benchmark for one variant, validate
// the round-trip, then attempt CUDA graph capture and record the outcome.
static VariantResult run_variant(Variant v,
                                 const std::vector<float>& h_input,
                                 float* d_input,
                                 size_t data_bytes,
                                 size_t dim_x,
                                 size_t dim_y,
                                 float eb,
                                 int runs) {
    VariantResult R;
    R.variant = v;
    const size_t n = h_input.size();

    std::cout << "\n══════════════════════════════════════════════════════════════\n"
              << " " << variant_name(v) << "\n"
              << "══════════════════════════════════════════════════════════════\n";

    // ── PREALLOCATE: the fastest non-graph path ──────────────────────────────
    Pipeline comp(data_bytes, MemoryStrategy::PREALLOCATE, POOL_MULT);
    build_variant(comp, v, eb, dim_x, dim_y);
    comp.finalize();
    comp.enableProfiling(true);

    void*  d_comp   = nullptr; size_t comp_sz   = 0;
    void*  d_decomp = nullptr; size_t decomp_sz = 0;

    // Warmup (JIT) — compress then decompress once, untimed.
    comp.compress(d_input, data_bytes, &d_comp, &comp_sz, 0);
    comp.decompress(d_comp, comp_sz, &d_decomp, &decomp_sz, 0);
    cudaDeviceSynchronize();
    R.compressed_size = comp_sz;

    // Validate round-trip on the warmup output.
    std::vector<float> h_decomp(n);
    cudaMemcpy(h_decomp.data(), d_decomp, n * sizeof(float), cudaMemcpyDeviceToHost);
    double max_err = 0.0;
    for (size_t i = 0; i < n; ++i)
        max_err = std::max(max_err, std::abs(static_cast<double>(h_input[i]) - h_decomp[i]));
    R.max_abs_error = max_err;

    // Timed compress loop.
    std::vector<double> comp_v, decomp_v;
    comp_v.reserve(runs);
    decomp_v.reserve(runs);
    for (int i = 0; i < runs; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        comp.compress(d_input, data_bytes, &d_comp, &comp_sz, 0);
        cudaDeviceSynchronize();
        auto t1 = std::chrono::high_resolution_clock::now();
        comp_v.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());

        auto t2 = std::chrono::high_resolution_clock::now();
        comp.decompress(d_comp, comp_sz, &d_decomp, &decomp_sz, 0);
        cudaDeviceSynchronize();
        auto t3 = std::chrono::high_resolution_clock::now();
        decomp_v.push_back(std::chrono::duration<double, std::milli>(t3 - t2).count());
    }
    R.comp_ms   = summarize(comp_v);
    R.decomp_ms = summarize(decomp_v);

    const double cr = static_cast<double>(data_bytes) / R.compressed_size;
    std::cout << std::fixed
              << "  Compression ratio : " << std::setprecision(3) << cr << "x\n"
              << "  Max abs error      : " << std::scientific << std::setprecision(3)
              << R.max_abs_error << "  (eb = " << eb << ")"
              // cuSZp reconstructs as q*2eb in float; the worst-case error sits
              // right at eb and can exceed it by a rounding ulp. Allow a tiny
              // tolerance so an at-bound round-trip isn't flagged as a failure.
              << (R.max_abs_error <= eb * 1.0001 ? "  [within bound]" : "  [OVER BOUND]") << "\n"
              << std::fixed << std::setprecision(3)
              << "  Compress  (PREALLOCATE): mean " << R.comp_ms.mean << " ms  min "
              << R.comp_ms.min << " ms  → " << std::setprecision(2)
              << tput_gbs(data_bytes, R.comp_ms.min) << " GB/s (peak)\n"
              << std::setprecision(3)
              << "  Decompress(PREALLOCATE): mean " << R.decomp_ms.mean << " ms  min "
              << R.decomp_ms.min << " ms  → " << std::setprecision(2)
              << tput_gbs(data_bytes, R.decomp_ms.min) << " GB/s (peak)\n";

    // ── GRAPH: attempt capture, expect it to fail on AdaptiveBitpack ──────────
    std::cout << "\n  ── CUDA Graph capture attempt ──\n";
    try {
        Pipeline gcomp(data_bytes, MemoryStrategy::PREALLOCATE, POOL_MULT);
        build_variant(gcomp, v, eb, dim_x, dim_y);
        gcomp.enableGraphMode(true);
        gcomp.finalize();

        cudaStream_t s = nullptr;
        cudaStreamCreate(&s);
        gcomp.warmup(s);
        gcomp.captureGraph(s);   // throws: AdaptiveBitpack is not graph-compatible
        cudaStreamSynchronize(s);
        cudaStreamDestroy(s);

        R.graph_ok = true;
        std::cout << "  Graph captured (unexpected for a cuSZp pipeline).\n";
    } catch (const std::exception& e) {
        R.graph_ok = false;
        R.graph_msg = e.what();
        std::cout << "  Graph mode UNAVAILABLE for this pipeline:\n    " << e.what() << "\n"
                  << "  → cuSZp's per-compress size readback (D2H) cannot live inside a\n"
                  << "    CUDA graph; PREALLOCATE above is the fastest capturable path.\n";
    }

    return R;
}

// ── Usage ───────────────────────────────────────────────────────────────────────
static void print_usage() {
    std::cerr << "Usage: cuszp_variants <file> [dim_x [dim_y [error_bound [runs]]]]\n"
              << "  file:        float32 binary input (required)\n"
              << "  dim_x:       X dimension (default 3600)\n"
              << "  dim_y:       Y dimension (default 1800)\n"
              << "  error_bound: 0 < eb < 1, ABS (default 1e-3)\n"
              << "  runs:        integer > 0 (default 50)\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) { print_usage(); return 1; }

    const char* input_file = argv[1];
    size_t dim_x = 3600, dim_y = 1800;
    float  eb    = DEFAULT_EB;
    int    runs  = DEFAULT_RUNS;

    if (argc > 2) dim_x = std::stoull(argv[2]);
    if (argc > 3) dim_y = std::stoull(argv[3]);
    if (argc > 4) { eb = std::stof(argv[4]); if (eb <= 0.0f) { print_usage(); return 1; } }
    if (argc > 5) { runs = std::stoi(argv[5]); if (runs <= 0) { print_usage(); return 1; } }

    const size_t n = dim_x * dim_y;
    std::vector<float> h_input(n);
    {
        std::FILE* fp = std::fopen(input_file, "rb");
        if (!fp) { std::cerr << "Cannot open: " << input_file << "\n"; return 1; }
        const size_t r = std::fread(h_input.data(), sizeof(float), n, fp);
        std::fclose(fp);
        if (r != n) {
            std::cerr << "Short read: expected " << n << " floats, got " << r
                      << ". Check dim_x/dim_y.\n";
            return 1;
        }
    }
    const size_t data_bytes = n * sizeof(float);

    float* d_input = nullptr;
    cudaMalloc(&d_input, data_bytes);
    cudaMemcpy(d_input, h_input.data(), data_bytes, cudaMemcpyHostToDevice);

    std::cout << "=== cuSZp 1/2/3 — performance (PREALLOCATE) + graph-mode check ===\n"
              << "  Dataset    : " << input_file << " (" << dim_x << "x" << dim_y << ")\n"
              << "  Elements   : " << n << "\n"
              << "  Raw size   : " << std::fixed << std::setprecision(2)
              << data_bytes / (1024.0 * 1024.0) << " MB\n"
              << "  Error bound: " << std::scientific << std::setprecision(1) << eb << " (ABS)\n"
              << "  Runs       : " << runs << " (+ 1 warmup)\n"
              << "  Pool mult  : " << std::fixed << std::setprecision(1) << POOL_MULT << "x\n";

    std::vector<VariantResult> results;
    for (Variant v : {Variant::CUSZP1, Variant::CUSZP2, Variant::CUSZP3})
        results.push_back(
            run_variant(v, h_input, d_input, data_bytes, dim_x, dim_y, eb, runs));

    // ── Summary table ─────────────────────────────────────────────────────────
    std::cout << "\n══ Summary (PREALLOCATE = fastest path; GRAPH unavailable) ═══════════════════\n"
              << std::left << std::setw(26) << "Variant"
              << std::right << std::setw(8)  << "CR"
              << std::setw(14) << "Comp GB/s"
              << std::setw(14) << "Decomp GB/s"
              << std::setw(10) << "Graph" << "\n"
              << std::string(72, '-') << "\n";
    for (const auto& R : results) {
        const double cr = static_cast<double>(data_bytes) / R.compressed_size;
        std::cout << std::left << std::setw(26) << variant_name(R.variant)
                  << std::right << std::fixed << std::setprecision(2)
                  << std::setw(7) << cr << "x"
                  << std::setw(14) << tput_gbs(data_bytes, R.comp_ms.min)
                  << std::setw(14) << tput_gbs(data_bytes, R.decomp_ms.min)
                  << std::setw(10) << (R.graph_ok ? "yes" : "no") << "\n";
    }
    std::cout << std::string(72, '-') << "\n"
              << "  Throughput = peak (min-latency run).  Graph 'no' = AdaptiveBitpack D2H\n"
              << "  size readback cannot be captured; PREALLOCATE is the fastest path.\n";

    cudaFree(d_input);
    std::cout << "\nDone.\n";
    return 0;
}
