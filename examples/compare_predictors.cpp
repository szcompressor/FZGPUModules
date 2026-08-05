/**
 * examples/compare_predictors.cpp
 *
 * Head-to-head comparison of the two fused predictor+quantizer stages on the
 * same input, downstream Huffman pipeline held fixed:
 *
 *   Pipeline L: LorenzoQuantStage<float, uint16_t> → HuffmanStage<uint16_t>
 *   Pipeline G: GInterpStage<float, uint16_t>      → HuffmanStage<uint16_t>
 *
 * Same error bound, same quantizer radius, same Huffman bklen — the only
 * variable is the prediction model. Lorenzo predicts each value from its
 * neighbours' decoded values; G-Interp predicts samples from a multi-level
 * spline interpolation pyramid (cuSZ-Hi port). On smooth scientific data
 * (climate fields, simulation snapshots) G-Interp typically wins on CR by
 * 1.5–3× but pays a higher per-element kernel cost; Lorenzo wins on
 * throughput. This binary reports the trade.
 *
 * Metrics per variant:
 *   - Compressed size and compression ratio
 *   - Compress / decompress throughput  (host-wall and DAG-timer, mean/best)
 *   - Reconstruction error (max-abs, MAE, RMSE)
 *   - PSNR vs the input data range
 *
 * Shared settings:
 *   - ErrorBoundMode::PREL (abs_eb = eb x max(|data|); the predictor-fused
 *     stages have no exact point-wise relative mode — see QuantizerStage REL)
 *   - quant_radius = 1024  (zigzag-compatible; fits Huffman bklen)
 *   - outlier_capacity = 10% of N
 *   - LorenzoQuant uses zigzag (cuSZ-style); G-Interp does not zigzag
 *   - Huffman bklen = 2048 (covers zigzag Lorenzo codes in [0, 2046]; G-Interp
 *     codes are centred at radius so range is [0, 2*radius-1 = 2047])
 *   - GInterp auto-tune mode = 3 (full structural — the cuSZ-Hi paper mode)
 *
 * Usage:
 *   ./build/bin/examples/compare_predictors <input.f32> [dim_x] [dim_y] [error_bound] [runs]
 *
 * Examples (CLDHGH is 2-D 3600x1800):
 *   ./build/bin/examples/compare_predictors data/CLDHGH.f32
 *   ./build/bin/examples/compare_predictors data/CLDHGH.f32 3600 1800 1e-4 20
 *
 * Build:
 *   cmake --preset release -DBUILD_EXAMPLES=ON
 *   cmake --build build/release -j$(nproc)
 *   Binary: build/release/bin/examples/compare_predictors
 */

#include "fzgpumodules.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

using namespace fz;

static constexpr float    DEFAULT_EB    = 1e-4f;
static constexpr int      DEFAULT_RUNS  = 10;
static constexpr float    POOL_MULT     = 4.0f;
static constexpr uint16_t QUANT_RADIUS  = 4096;   // bklen=8192 — wider window
                                                  // so both predictors can keep
                                                  // most residuals out of the
                                                  // outlier triplet at strict eb.
static constexpr uint16_t HUF_BKLEN     = 8192;
static constexpr float    OUTLIER_CAP   = 0.10f;
static constexpr uint8_t  GINTERP_TUNE  = 3;      // full structural auto-tune

// ── Data loading ──────────────────────────────────────────────────────────────

static bool load_data(
    const char* path, size_t dim_x, size_t dim_y,
    std::vector<float>& h_out, float** d_out)
{
    const size_t n = dim_x * dim_y;
    h_out.resize(n);

    std::ifstream f(path, std::ios::binary);
    if (!f) { std::cerr << "Cannot open: " << path << "\n"; return false; }
    f.read(reinterpret_cast<char*>(h_out.data()), n * sizeof(float));
    if (f.gcount() != static_cast<std::streamsize>(n * sizeof(float))) {
        std::cerr << "Read mismatch: expected " << n * sizeof(float)
                  << " bytes, got " << f.gcount() << "\n";
        return false;
    }

    cudaMalloc(d_out, n * sizeof(float));
    cudaMemcpy(*d_out, h_out.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    return true;
}

// ── Pipeline builders ─────────────────────────────────────────────────────────

static void build_lorenzo_pipeline(Pipeline& p, float eb, size_t dim_x, size_t dim_y)
{
    p.setDims(dim_x, dim_y, 1);

    auto* lq = p.addStage<LorenzoQuantStage<float, uint16_t>>();
    lq->setErrorBound(eb);
    lq->setErrorBoundMode(ErrorBoundMode::PREL);
    lq->setQuantRadius(QUANT_RADIUS);
    lq->setOutlierCapacity(OUTLIER_CAP);
    lq->setZigzagCodes(true);   // [-radius, radius-1] → [0, 2*radius-2]

    auto* huf = p.addStage<HuffmanStage<uint16_t>>();
    huf->setBklen(HUF_BKLEN);
    p.connect(huf, lq, "codes");

    p.finalize();
}

static void build_ginterp_pipeline(Pipeline& p, float eb, size_t dim_x, size_t dim_y)
{
    p.setDims(dim_x, dim_y, 1);

    auto* gi = p.addStage<GInterpStage<float, uint16_t>>();
    gi->setErrorBound(eb);
    gi->setErrorBoundMode(ErrorBoundMode::PREL);
    gi->setQuantRadius(QUANT_RADIUS);
    gi->setOutlierCapacity(OUTLIER_CAP);
    gi->setAutoTuning(GINTERP_TUNE);

    auto* huf = p.addStage<HuffmanStage<uint16_t>>();
    huf->setBklen(HUF_BKLEN);
    p.connect(huf, gi, "codes");

    p.finalize();
}

// ── Per-variant timing result ─────────────────────────────────────────────────

struct VariantResult {
    std::string name;
    size_t      compressed_size = 0;
    size_t      peak_memory     = 0;
    double comp_mean_host_ms = 0.0;
    float  comp_mean_dag_ms  = 0.0f;
    double comp_min_host_ms  = 0.0;
    float  comp_min_dag_ms   = 0.0f;
    double decomp_mean_host_ms = 0.0;
    float  decomp_mean_dag_ms  = 0.0f;
    double decomp_min_host_ms  = 0.0;
    float  decomp_min_dag_ms   = 0.0f;
    float  max_abs_error = 0.0f;
    double mae           = 0.0;
    double rmse          = 0.0;
    double psnr          = 0.0;   // vs data range
};

// ── Main benchmark loop ───────────────────────────────────────────────────────

static VariantResult run_variant(
    Pipeline& p, const std::string& name,
    float* d_input, size_t input_bytes,
    const std::vector<float>& h_input, double data_range, int runs)
{
    VariantResult res;
    res.name = name;
    const size_t N = h_input.size();

    std::cout << "\n══ " << name << " ══════════════════════════════════════════\n";

    void*  d_comp  = nullptr;
    size_t comp_sz = 0;
    p.compress(d_input, input_bytes, &d_comp, &comp_sz, 0);
    cudaDeviceSynchronize();

    res.compressed_size = comp_sz;
    res.peak_memory     = p.getPeakMemoryUsage();

    // ── Compress benchmark ────────────────────────────────────────────────────
    std::cout << "\n  -- Compress --\n";
    std::vector<double> comp_host;  comp_host.reserve(runs);
    std::vector<float>  comp_dag;   comp_dag.reserve(runs);
    for (int i = 0; i < runs; ++i) {
        const auto t0 = std::chrono::high_resolution_clock::now();
        p.compress(d_input, input_bytes, &d_comp, &comp_sz, 0);
        cudaDeviceSynchronize();
        const auto t1 = std::chrono::high_resolution_clock::now();

        const double hms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        const float  dms = p.getLastPerfResult().dag_elapsed_ms;
        comp_host.push_back(hms);
        comp_dag.push_back(dms);

        const float h_gbs = static_cast<float>(input_bytes) / static_cast<float>(hms * 1e-3) / 1e9f;
        const float d_gbs = static_cast<float>(input_bytes) / static_cast<float>(dms * 1e-3f) / 1e9f;
        std::cout << "  run " << std::setw(2) << (i + 1) << ":  "
                  << "host " << std::setw(8) << std::fixed << std::setprecision(3) << hms << " ms  "
                  << std::setw(7) << std::setprecision(2) << h_gbs << " GB/s   "
                  << "dag "  << std::setw(8) << std::setprecision(3) << dms << " ms  "
                  << std::setw(7) << std::setprecision(2) << d_gbs << " GB/s\n";
    }
    res.comp_mean_host_ms = std::accumulate(comp_host.begin(), comp_host.end(), 0.0) / runs;
    res.comp_mean_dag_ms  = std::accumulate(comp_dag.begin(),  comp_dag.end(),  0.0f) / runs;
    res.comp_min_host_ms  = *std::min_element(comp_host.begin(), comp_host.end());
    res.comp_min_dag_ms   = *std::min_element(comp_dag.begin(),  comp_dag.end());

    // ── Decompress benchmark ──────────────────────────────────────────────────
    std::cout << "\n  -- Decompress --\n";
    std::vector<double> decomp_host;  decomp_host.reserve(runs);
    std::vector<float>  decomp_dag;   decomp_dag.reserve(runs);
    void*  d_rec  = nullptr;
    size_t rec_sz = 0;
    for (int i = 0; i < runs; ++i) {
        const auto t0 = std::chrono::high_resolution_clock::now();
        p.decompress(d_comp, comp_sz, &d_rec, &rec_sz, 0);
        cudaDeviceSynchronize();
        const auto t1 = std::chrono::high_resolution_clock::now();

        const double hms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        const float  dms = p.getLastPerfResult().dag_elapsed_ms;
        decomp_host.push_back(hms);
        decomp_dag.push_back(dms);

        const float h_gbs = static_cast<float>(input_bytes) / static_cast<float>(hms * 1e-3) / 1e9f;
        const float d_gbs = static_cast<float>(input_bytes) / static_cast<float>(dms * 1e-3f) / 1e9f;
        std::cout << "  run " << std::setw(2) << (i + 1) << ":  "
                  << "host " << std::setw(8) << std::fixed << std::setprecision(3) << hms << " ms  "
                  << std::setw(7) << std::setprecision(2) << h_gbs << " GB/s   "
                  << "dag "  << std::setw(8) << std::setprecision(3) << dms << " ms  "
                  << std::setw(7) << std::setprecision(2) << d_gbs << " GB/s\n";
    }
    res.decomp_mean_host_ms = std::accumulate(decomp_host.begin(), decomp_host.end(), 0.0) / runs;
    res.decomp_mean_dag_ms  = std::accumulate(decomp_dag.begin(),  decomp_dag.end(),  0.0f) / runs;
    res.decomp_min_host_ms  = *std::min_element(decomp_host.begin(), decomp_host.end());
    res.decomp_min_dag_ms   = *std::min_element(decomp_dag.begin(),  decomp_dag.end());

    // ── Error statistics ──────────────────────────────────────────────────────
    if (rec_sz == input_bytes) {
        std::vector<float> h_rec(N);
        cudaMemcpy(h_rec.data(), d_rec, rec_sz, cudaMemcpyDeviceToHost);
        double sum_abs = 0.0, sum_sq = 0.0;
        float  max_abs = 0.0f;
        for (size_t i = 0; i < N; ++i) {
            const float e = std::abs(h_rec[i] - h_input[i]);
            max_abs  = std::max(max_abs, e);
            sum_abs += static_cast<double>(e);
            sum_sq  += static_cast<double>(e) * static_cast<double>(e);
        }
        res.max_abs_error = max_abs;
        res.mae           = sum_abs / static_cast<double>(N);
        res.rmse          = std::sqrt(sum_sq / static_cast<double>(N));
        // PSNR vs data range (the standard SZ-paper definition).
        res.psnr = (res.rmse > 0.0)
            ? 20.0 * std::log10(data_range) - 20.0 * std::log10(res.rmse)
            : std::numeric_limits<double>::infinity();
    } else {
        std::cerr << "  WARNING: decompressed size " << rec_sz
                  << " != input size " << input_bytes << " — skipping error stats\n";
    }
    return res;
}

// ── Usage ─────────────────────────────────────────────────────────────────────

static void print_usage(const char* prog) {
    std::cerr
        << "Usage: " << prog << " <input.f32> [dim_x] [dim_y] [error_bound] [runs]\n"
        << "  input.f32:    path to float32 binary input file (required)\n"
        << "  dim_x:        X dimension (default: 3600)\n"
        << "  dim_y:        Y dimension (default: 1800)\n"
        << "  error_bound:  > 0  REL (default: 1e-4)\n"
        << "  runs:         integer > 0 (default: " << DEFAULT_RUNS << ")\n"
        << "\nExamples:\n"
        << "  " << prog << " data/CLDHGH.f32\n"
        << "  " << prog << " data/CLDHGH.f32 3600 1800 1e-4 20\n";
}

int main(int argc, char* argv[])
{
    if (argc < 2) { print_usage(argv[0]); return 1; }

    const char* input_file = argv[1];
    size_t dim_x = 3600, dim_y = 1800;
    float  eb    = DEFAULT_EB;
    int    runs  = DEFAULT_RUNS;

    if (argc > 2) dim_x = std::stoull(argv[2]);
    if (argc > 3) dim_y = std::stoull(argv[3]);
    if (argc > 4) { eb = std::stof(argv[4]); if (eb <= 0.0f) { print_usage(argv[0]); return 1; } }
    if (argc > 5) { runs = std::stoi(argv[5]); if (runs <= 0) { std::cerr << "runs must be > 0\n"; return 1; } }

    const size_t N           = dim_x * dim_y;
    const size_t input_bytes = N * sizeof(float);

    std::vector<float> h_input;
    float* d_input = nullptr;
    if (!load_data(input_file, dim_x, dim_y, h_input, &d_input)) return 1;

    // Data range — used for the PSNR denominator.
    float mn = h_input[0], mx = h_input[0];
    for (float v : h_input) { if (v < mn) mn = v; if (v > mx) mx = v; }
    const double data_range = double(mx) - double(mn);

    std::cout << "=== Predictor Comparison: LorenzoQuant vs GInterp ===\n"
              << "  Dataset:        " << input_file << " (" << dim_x << " x " << dim_y << ")\n"
              << "  Elements:       " << N << "\n"
              << "  Raw size:       " << std::fixed << std::setprecision(2)
              << input_bytes / (1024.0 * 1024.0) << " MB\n"
              << "  Data range:     [" << mn << ", " << mx << "] rng=" << data_range << "\n"
              << "  Error bound:    " << std::scientific << std::setprecision(1) << eb << " (REL)\n"
              << "  Quant radius:   " << QUANT_RADIUS << "\n"
              << "  Outlier cap:    " << std::fixed << std::setprecision(0)
              << OUTLIER_CAP * 100.0f << "% of N\n"
              << "  Huffman bklen:  " << HUF_BKLEN << "\n"
              << "  GInterp tune:   mode " << (int)GINTERP_TUNE
              << " (full structural)\n"
              << "  Runs:           " << runs << " (+1 warmup each)\n"
              << "  Pool mult:      " << POOL_MULT << "x\n";

    VariantResult lrz_res, gi_res;

    {
        Pipeline p(input_bytes, MemoryStrategy::PREALLOCATE, POOL_MULT);
        p.enableProfiling(true);
        build_lorenzo_pipeline(p, eb, dim_x, dim_y);
        lrz_res = run_variant(p, "LorenzoQuant + Huffman",
                              d_input, input_bytes, h_input, data_range, runs);
    }
    {
        Pipeline p(input_bytes, MemoryStrategy::PREALLOCATE, POOL_MULT);
        p.enableProfiling(true);
        build_ginterp_pipeline(p, eb, dim_x, dim_y);
        gi_res  = run_variant(p, "GInterp + Huffman",
                              d_input, input_bytes, h_input, data_range, runs);
    }

    // ── Side-by-side summary ──────────────────────────────────────────────────
    const auto tput = [&](double ms) -> double {
        return double(input_bytes) / (ms * 1e-3) / 1e9;
    };
    const double cr_lrz = double(input_bytes) / double(lrz_res.compressed_size);
    const double cr_gi  = double(input_bytes) / double(gi_res.compressed_size);
    const auto fmt_delta = [](double v_new, double v_base, bool higher_is_better) -> std::string {
        const double delta = (v_new - v_base) / v_base * 100.0;
        if (std::abs(delta) < 0.05) return "~same";
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(1) << std::abs(delta) << "%";
        return ((delta > 0) == higher_is_better ? "+" : "-") + oss.str();
    };

    std::cout << "\n\n══ Summary ══════════════════════════════════════════════════════════════\n";
    std::cout << std::left  << std::setw(28) << "Metric"
              << std::right << std::setw(18) << "LorenzoQuant"
              << std::setw(18) << "GInterp"
              << std::setw(16) << "GInterp/Lrz"
              << "\n" << std::string(80, '-') << "\n";

    const auto row = [&](const std::string& label,
                         double v_lrz, double v_gi,
                         const std::string& unit, bool higher_is_better)
    {
        std::cout << std::left  << std::setw(28) << label
                  << std::right << std::setw(15) << std::fixed << std::setprecision(3) << v_lrz << unit
                  << std::setw(15) << v_gi << unit
                  << "  " << std::setw(12) << fmt_delta(v_gi, v_lrz, higher_is_better)
                  << "\n";
    };

    row("Compressed size",
        lrz_res.compressed_size / (1024.0 * 1024.0),
        gi_res.compressed_size  / (1024.0 * 1024.0), " MB", false);
    row("Compression ratio",   cr_lrz,  cr_gi,  "x  ", true);
    row("Comp tput dag mean",  tput(lrz_res.comp_mean_dag_ms),  tput(gi_res.comp_mean_dag_ms),  " GB/s", true);
    row("Comp tput dag best",  tput(lrz_res.comp_min_dag_ms),   tput(gi_res.comp_min_dag_ms),   " GB/s", true);
    row("Comp tput host mean", tput(lrz_res.comp_mean_host_ms), tput(gi_res.comp_mean_host_ms), " GB/s", true);
    row("Decomp tput dag mean", tput(lrz_res.decomp_mean_dag_ms), tput(gi_res.decomp_mean_dag_ms),  " GB/s", true);
    row("Decomp tput dag best", tput(lrz_res.decomp_min_dag_ms),  tput(gi_res.decomp_min_dag_ms),   " GB/s", true);
    row("Decomp tput host mean", tput(lrz_res.decomp_mean_host_ms), tput(gi_res.decomp_mean_host_ms), " GB/s", true);
    row("Max abs error",       lrz_res.max_abs_error,        gi_res.max_abs_error,        "", false);
    row("MAE",                 lrz_res.mae,                  gi_res.mae,                  "", false);
    row("RMSE",                lrz_res.rmse,                 gi_res.rmse,                 "", false);
    row("PSNR (vs data range)", lrz_res.psnr,                gi_res.psnr,                 " dB", true);
    row("Peak memory",
        lrz_res.peak_memory / (1024.0 * 1024.0),
        gi_res.peak_memory  / (1024.0 * 1024.0), " MB", false);

    std::cout << std::string(80, '-') << "\n"
              << "Note: larger CR / PSNR / throughput is better; smaller MAE / RMSE / memory is better.\n";

    cudaFree(d_input);
    return 0;
}
