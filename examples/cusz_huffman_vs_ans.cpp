/**
 * examples/cusz_huffman_vs_ans.cpp
 *
 * Throughput and compression-ratio comparison:
 *   Pipeline A (Huffman):  LorenzoQuantStage<float,uint16_t> → HuffmanStage<uint16_t>
 *   Pipeline B (ANS):      LorenzoQuantStage<float,uint16_t> → ANSStage
 *   Pipeline C (ADM+ANS):  LorenzoQuantStage<float,uint16_t> → ADMStage → ANSStage
 *
 * Pipelines A and B isolate the entropy-coder variable.  Pipeline C adds ADM
 * between LorenzoQuant and ANS to show the additional compression-ratio gain
 * from remapping uint16 codes into a compact 8-bit symbol domain before ANS.
 * All three pipelines share identical LorenzoQuant settings.  Metrics reported:
 *   - Compressed size and compression ratio
 *   - Compress throughput  (host-wall and DAG-timer, mean / best)
 *   - Decompress throughput (host-wall and DAG-timer, mean / best)
 *   - Peak device memory usage
 *   - Reconstruction error (max absolute error, MAE, RMSE)
 *
 * LorenzoQuant settings:
 *   - ErrorBoundMode::ABS, quant_radius=512, zigzag=true
 *   - Zigzag maps [-511, 511] → [0, 1022], so Huffman bklen=1024 is sufficient
 *     and no out-of-range symbols are produced
 *   - outlier_capacity=10% of N
 *
 * Usage:
 *   ./build/bin/examples/cusz_huffman_vs_ans <input.f32> [dim_x] [dim_y] [error_bound] [runs]
 *
 * Examples:
 *   ./build/bin/examples/cusz_huffman_vs_ans data/CLDHGH.f32 3600 1800
 *   ./build/bin/examples/cusz_huffman_vs_ans data/CLDHGH.f32 3600 1800 1e-3 20
 *
 * Delta columns in the summary:
 *   ANS/Huf      — ANS relative to Huffman (isolates entropy-coder change)
 *   ADM+ANS/ANS  — ADM+ANS relative to ANS (isolates ADM's contribution)
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
#include <limits>
#include <numeric>
#include <string>
#include <vector>

using namespace fz;

static constexpr float  DEFAULT_EB   = 1e-3f;
static constexpr int    DEFAULT_RUNS = 10;
static constexpr float  POOL_MULT    = 4.0f;

// Shared LorenzoQuant settings — identical for both pipelines.
static constexpr uint16_t QUANT_RADIUS = 512;   // zigzag → [0, 1022], fits Huffman bklen=1024
static constexpr float    OUTLIER_CAP  = 0.10f;

// ── Data loading ──────────────────────────────────────────────────────────────

static bool load_data(
    const char*    path,
    size_t         dim_x,
    size_t         dim_y,
    std::vector<float>& h_out,
    float**        d_out)
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

static void build_huffman_pipeline(Pipeline& p, float eb, size_t dim_x, size_t dim_y)
{
    p.setDims(dim_x, dim_y, 1);

    auto* lq = p.addStage<LorenzoQuantStage<float, uint16_t>>();
    lq->setErrorBound(eb);
    lq->setErrorBoundMode(ErrorBoundMode::ABS);
    lq->setQuantRadius(QUANT_RADIUS);
    lq->setOutlierCapacity(OUTLIER_CAP);
    lq->setZigzagCodes(true);

    // bklen=1024 covers all zigzag codes in [0, 2*512-2 = 1022].
    auto* huf = p.addStage<HuffmanStage<uint16_t>>();
    huf->setBklen(1024);
    p.connect(huf, lq, "codes");

    p.finalize();
}

static void build_ans_pipeline(Pipeline& p, float eb, size_t dim_x, size_t dim_y)
{
    p.setDims(dim_x, dim_y, 1);

    auto* lq = p.addStage<LorenzoQuantStage<float, uint16_t>>();
    lq->setErrorBound(eb);
    lq->setErrorBoundMode(ErrorBoundMode::ABS);
    lq->setQuantRadius(QUANT_RADIUS);
    lq->setOutlierCapacity(OUTLIER_CAP);
    lq->setZigzagCodes(true);

    auto* ans = p.addStage<ANSStage>();
    (void)ans;  // default prob_bits=10
    p.connect(ans, lq, "codes");

    p.finalize();
}

static void build_adm_ans_pipeline(Pipeline& p, float eb, size_t dim_x, size_t dim_y)
{
    p.setDims(dim_x, dim_y, 1);

    auto* lq = p.addStage<LorenzoQuantStage<float, uint16_t>>();
    lq->setErrorBound(eb);
    lq->setErrorBoundMode(ErrorBoundMode::ABS);
    lq->setQuantRadius(QUANT_RADIUS);
    lq->setOutlierCapacity(OUTLIER_CAP);
    lq->setZigzagCodes(true);

    // ADM remaps uint16_t codes into a compact 8-bit domain before ANS encoding.
    auto* adm = p.addStage<ADMStage>();  // default dtype=U16 matches LorenzoQuant output
    p.connect(adm, lq, "codes");

    auto* ans = p.addStage<ANSStage>();
    p.connect(ans, adm);

    p.finalize();
}

// ── Per-variant timing result ─────────────────────────────────────────────────

struct VariantResult {
    std::string name;
    size_t      compressed_size = 0;
    size_t      peak_memory     = 0;
    // compress
    double comp_mean_host_ms = 0.0;
    float  comp_mean_dag_ms  = 0.0f;
    double comp_min_host_ms  = 0.0;
    float  comp_min_dag_ms   = 0.0f;
    // decompress
    double decomp_mean_host_ms = 0.0;
    float  decomp_mean_dag_ms  = 0.0f;
    double decomp_min_host_ms  = 0.0;
    float  decomp_min_dag_ms   = 0.0f;
    // error
    float  max_abs_error = 0.0f;
    double mae           = 0.0;
    double rmse          = 0.0;
};

// ── Main benchmark loop ───────────────────────────────────────────────────────

static VariantResult run_variant(
    Pipeline&                  p,
    const std::string&         name,
    float*                     d_input,
    size_t                     input_bytes,
    const std::vector<float>&  h_input,
    int                        runs)
{
    VariantResult res;
    res.name = name;
    const size_t N = h_input.size();

    std::cout << "\n══ " << name << " ══════════════════════════════════════════\n";

    // ── Warmup compress ───────────────────────────────────────────────────────
    fz::BorrowedDeviceBuffer comp = p.compress({d_input, input_bytes}, 0);
    cudaDeviceSynchronize();

    res.compressed_size = comp.bytes();
    res.peak_memory     = p.getPeakMemoryUsage();

    // ── Compress benchmark ────────────────────────────────────────────────────
    std::cout << "\n  -- Compress --\n";
    std::vector<double> comp_host;  comp_host.reserve(static_cast<size_t>(runs));
    std::vector<float>  comp_dag;   comp_dag.reserve(static_cast<size_t>(runs));

    bool printed_comp = false;
    for (int i = 0; i < runs; ++i) {
        const auto t0 = std::chrono::high_resolution_clock::now();
        comp = p.compress({d_input, input_bytes}, 0);
        cudaDeviceSynchronize();
        const auto t1 = std::chrono::high_resolution_clock::now();

        const double hms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        const float  dms = p.getLastPerfResult().dag_elapsed_ms;

        if (!printed_comp) {
            std::cout << "\n    Library report (run 1, post-warmup):\n";
            p.getLastPerfResult().print(std::cout);
            printed_comp = true;
        }

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
    std::vector<double> decomp_host;  decomp_host.reserve(static_cast<size_t>(runs));
    std::vector<float>  decomp_dag;   decomp_dag.reserve(static_cast<size_t>(runs));

    fz::BorrowedDeviceBuffer rec;

    bool printed_decomp = false;
    for (int i = 0; i < runs; ++i) {
        const auto t0 = std::chrono::high_resolution_clock::now();
        rec = p.decompressBorrowed(comp.cspan(), 0);
        cudaDeviceSynchronize();
        const auto t1 = std::chrono::high_resolution_clock::now();

        const double hms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        const float  dms = p.getLastPerfResult().dag_elapsed_ms;

        if (!printed_decomp) {
            std::cout << "\n    Library report (run 1):\n";
            p.getLastPerfResult().print(std::cout);
            printed_decomp = true;
        }

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
    if (rec.bytes() == input_bytes) {
        std::vector<float> h_rec(N);
        cudaMemcpy(h_rec.data(), rec.data(), rec.bytes(), cudaMemcpyDeviceToHost);
        // rec is a BorrowedDeviceBuffer (pool-owned) — do NOT free.

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
    } else {
        std::cerr << "  WARNING: decompressed size " << rec.bytes()
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
        << "  error_bound:  > 0 (default: 1e-3)\n"
        << "  runs:         integer > 0 (default: " << DEFAULT_RUNS << ")\n"
        << "\nExamples:\n"
        << "  " << prog << " data/CLDHGH.f32 3600 1800\n"
        << "  " << prog << " data/CLDHGH.f32 3600 1800 1e-3 20\n";
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[])
{
    if (argc < 2) { print_usage(argv[0]); return 1; }

    const char* input_file = argv[1];
    size_t      dim_x      = 3600;
    size_t      dim_y      = 1800;
    float       eb         = DEFAULT_EB;
    int         runs       = DEFAULT_RUNS;

    if (argc > 2) dim_x = std::stoull(argv[2]);
    if (argc > 3) dim_y = std::stoull(argv[3]);
    if (argc > 4) { eb = std::stof(argv[4]); if (eb <= 0.0f) { print_usage(argv[0]); return 1; } }
    if (argc > 5) { runs = std::stoi(argv[5]); if (runs <= 0) { std::cerr << "runs must be > 0\n"; return 1; } }

    const size_t N          = dim_x * dim_y;
    const size_t input_bytes = N * sizeof(float);

    // ── Load data ─────────────────────────────────────────────────────────────
    std::vector<float> h_input;
    float* d_input = nullptr;
    if (!load_data(input_file, dim_x, dim_y, h_input, &d_input)) return 1;

    std::cout << "=== cuSZ: Huffman vs ANS vs ADM+ANS Comparison ===\n"
              << "  Dataset:        " << input_file << " (" << dim_x << " x " << dim_y << ")\n"
              << "  Elements:       " << N << "\n"
              << "  Raw size:       " << std::fixed << std::setprecision(2)
              << input_bytes / (1024.0 * 1024.0) << " MB\n"
              << "  Error bound:    " << std::scientific << std::setprecision(1) << eb << " (ABS)\n"
              << "  Quant radius:   " << QUANT_RADIUS << "  (zigzag → [0, "
              << 2 * QUANT_RADIUS - 2 << "])\n"
              << "  Outlier cap:    " << std::fixed << std::setprecision(0)
              << OUTLIER_CAP * 100.0f << "% of N\n"
              << "  Huffman bklen:  1024\n"
              << "  ANS prob_bits:  10 (both ANS and ADM+ANS)\n"
              << "  ADM dtype:      U16 (matches LorenzoQuant uint16_t codes)\n"
              << "  Runs:           " << runs << " (+ 1 warmup each)\n"
              << "  Pool mult:      " << POOL_MULT << "x\n";

    // ── Huffman variant ───────────────────────────────────────────────────────
    VariantResult huf_res, ans_res, adm_ans_res;

    {
        Pipeline p(input_bytes, MemoryStrategy::PREALLOCATE, POOL_MULT);
        p.enableProfiling(true);
        build_huffman_pipeline(p, eb, dim_x, dim_y);
        huf_res = run_variant(p, "Huffman (cuSZ-style)", d_input, input_bytes, h_input, runs);
    }

    // ── ANS variant ───────────────────────────────────────────────────────────
    {
        Pipeline p(input_bytes, MemoryStrategy::PREALLOCATE, POOL_MULT);
        p.enableProfiling(true);
        build_ans_pipeline(p, eb, dim_x, dim_y);
        ans_res = run_variant(p, "ANS (rANS, dietGPU)", d_input, input_bytes, h_input, runs);
    }

    // ── ADM+ANS variant ───────────────────────────────────────────────────────
    {
        Pipeline p(input_bytes, MemoryStrategy::PREALLOCATE, POOL_MULT);
        p.enableProfiling(true);
        build_adm_ans_pipeline(p, eb, dim_x, dim_y);
        adm_ans_res = run_variant(p, "ADM+ANS (MANS-style)", d_input, input_bytes, h_input, runs);
    }

    // ── Side-by-side comparison ───────────────────────────────────────────────
    const auto tput = [&](double ms) -> float {
        return static_cast<float>(input_bytes) / static_cast<float>(ms * 1e-3) / 1e9f;
    };

    const double cr_huf     = static_cast<double>(input_bytes) / huf_res.compressed_size;
    const double cr_ans     = static_cast<double>(input_bytes) / ans_res.compressed_size;
    const double cr_adm_ans = static_cast<double>(input_bytes) / adm_ans_res.compressed_size;

    // Returns "+X.X%" / "-X.X%" / "~same" for a relative delta.
    // "better" direction follows higher_is_better.
    const auto fmt_delta = [](double v_new, double v_base, bool higher_is_better) -> std::string {
        const double delta = (v_new - v_base) / v_base * 100.0;
        if (std::abs(delta) < 0.05) return "~same";
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(1) << std::abs(delta) << "%";
        return ((delta > 0) == higher_is_better ? "+" : "-") + oss.str();
    };

    std::cout << "\n\n══ Summary ═══════════════════════════════════════════════════════════════════════════════\n";
    std::cout << std::left  << std::setw(28) << "Metric"
              << std::right << std::setw(16) << "Huffman"
              << std::setw(16) << "ANS"
              << std::setw(16) << "ADM+ANS"
              << std::setw(14) << "ANS/Huf"
              << std::setw(16) << "ADM+ANS/ANS"
              << "\n" << std::string(90, '-') << "\n";

    const auto row = [&](const std::string& label,
                         double v_huf, double v_ans, double v_adm,
                         const std::string& unit,
                         bool higher_is_better = true)
    {
        std::cout << std::left  << std::setw(28) << label
                  << std::right << std::setw(13) << std::fixed << std::setprecision(2) << v_huf << unit
                  << std::setw(13) << v_ans << unit
                  << std::setw(13) << v_adm << unit
                  << "  " << std::setw(10) << fmt_delta(v_ans, v_huf, higher_is_better)
                  << "  " << std::setw(12) << fmt_delta(v_adm, v_ans, higher_is_better)
                  << "\n";
    };

    row("Compressed size",
        huf_res.compressed_size / (1024.0 * 1024.0),
        ans_res.compressed_size / (1024.0 * 1024.0),
        adm_ans_res.compressed_size / (1024.0 * 1024.0), " MB", false);
    row("Compression ratio",   cr_huf,  cr_ans,  cr_adm_ans,  "x  ", true);
    row("Comp tput dag mean",
        tput(huf_res.comp_mean_dag_ms),
        tput(ans_res.comp_mean_dag_ms),
        tput(adm_ans_res.comp_mean_dag_ms),  " GB/s", true);
    row("Comp tput dag best",
        tput(huf_res.comp_min_dag_ms),
        tput(ans_res.comp_min_dag_ms),
        tput(adm_ans_res.comp_min_dag_ms),   " GB/s", true);
    row("Comp tput host mean",
        tput(huf_res.comp_mean_host_ms),
        tput(ans_res.comp_mean_host_ms),
        tput(adm_ans_res.comp_mean_host_ms), " GB/s", true);
    row("Decomp tput dag mean",
        tput(huf_res.decomp_mean_dag_ms),
        tput(ans_res.decomp_mean_dag_ms),
        tput(adm_ans_res.decomp_mean_dag_ms),  " GB/s", true);
    row("Decomp tput dag best",
        tput(huf_res.decomp_min_dag_ms),
        tput(ans_res.decomp_min_dag_ms),
        tput(adm_ans_res.decomp_min_dag_ms),   " GB/s", true);
    row("Decomp tput host mean",
        tput(huf_res.decomp_mean_host_ms),
        tput(ans_res.decomp_mean_host_ms),
        tput(adm_ans_res.decomp_mean_host_ms), " GB/s", true);
    row("Peak device memory",
        huf_res.peak_memory / (1024.0 * 1024.0),
        ans_res.peak_memory / (1024.0 * 1024.0),
        adm_ans_res.peak_memory / (1024.0 * 1024.0), " MB", false);
    row("Max abs error",
        static_cast<double>(huf_res.max_abs_error),
        static_cast<double>(ans_res.max_abs_error),
        static_cast<double>(adm_ans_res.max_abs_error), "   ", false);
    row("MAE",  huf_res.mae,  ans_res.mae,  adm_ans_res.mae,  "   ", false);
    row("RMSE", huf_res.rmse, ans_res.rmse, adm_ans_res.rmse, "   ", false);

    std::cout << std::string(90, '-') << "\n"
              << "  Input bytes: " << input_bytes << "  ("
              << std::setprecision(2) << input_bytes / (1024.0 * 1024.0) << " MB)\n"
              << "  Error bound: " << std::scientific << std::setprecision(1) << eb << " ABS\n";

    cudaFree(d_input);
    std::cout << "\nDone.\n";
    return 0;
}
