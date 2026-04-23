/**
 * PFPL profiling example (minimal).
 *
 * Runs the same pipeline under both PREALLOCATE and MINIMAL memory strategies
 * and prints a side-by-side memory and throughput comparison.
 *
 * Profiles one phase at a time:
 *   - compress
 *   - decompress
 *
 * Usage:
 *   ./build/bin/examples/pfpl_example [file [dim_x [dim_y [error_bound [mode [phase [runs [threshold]]]]]]]]
 *
 * Examples:
 *   ./build/bin/examples/pfpl_example data/CLDHGH.f32 3600 1800
 *   ./build/bin/examples/pfpl_example data/CLDHGH.f32 3600 1800 1e-3 rel compress 20
 *   ./build/bin/examples/pfpl_example data/CLDHGH.f32 3600 1800 1e-3 abs decompress 20
 */

#include "fzgpumodules.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

using namespace fz;

static constexpr float  DEFAULT_EB   = 1e-3f;
static constexpr size_t CHUNK        = 16384;
static constexpr int    DEFAULT_RUNS = 20;
// Initial pool construction multiplier (applied to input_size before topology
// is known).  After finalize() the threshold is tightened to topo_base × 1.1.
static constexpr float  POOL_MULT    = 3.0f;

enum class ProfilePhase { Compress, Decompress };

static std::pair<float*, size_t> load_data(const char* path, size_t dim_x, size_t dim_y) {
    const size_t n = dim_x * dim_y;
    std::vector<float> h(n);

    std::FILE* fp = std::fopen(path, "rb");
    if (!fp) return {nullptr, 0};

    const size_t read_n = std::fread(h.data(), sizeof(float), n, fp);
    std::fclose(fp);
    if (read_n != n) return {nullptr, 0};

    float* d = nullptr;
    cudaMalloc(&d, n * sizeof(float));
    cudaMemcpy(d, h.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    return {d, n};
}

static void build_pfpl_pipeline(
    Pipeline& p,
    float eb,
    ErrorBoundMode mode = ErrorBoundMode::REL,
    float threshold = std::numeric_limits<float>::infinity()) {

    auto* quant = p.addStage<QuantizerStage<float, uint32_t>>();
    quant->setErrorBound(eb);
    quant->setErrorBoundMode(mode);
    quant->setQuantRadius(mode == ErrorBoundMode::ABS ? (1 << 22) : 32768);
    quant->setOutlierCapacity(0.05f);
    quant->setZigzagCodes(true);
    if (std::isfinite(threshold)) quant->setOutlierThreshold(threshold);
    if (mode != ErrorBoundMode::REL) quant->setInplaceOutliers(true);

    auto* diff = p.addStage<DifferenceStage<int32_t, uint32_t>>();
    diff->setChunkSize(CHUNK);
    p.connect(diff, quant, "codes");

    auto* bitshuffle = p.addStage<BitshuffleStage>();
    bitshuffle->setBlockSize(CHUNK);
    bitshuffle->setElementWidth(4);
    p.connect(bitshuffle, diff);

    auto* rze = p.addStage<RZEStage>();
    rze->setChunkSize(CHUNK);
    rze->setLevels(4);
    p.connect(rze, bitshuffle);

    p.finalize();
}

static void print_usage() {
    std::cerr
        << "Usage: pfpl_example [file [dim_x [dim_y [error_bound [mode [phase [runs [threshold]]]]]]]]\n"
        << "  file:        path to float32 binary input file (required)\n"
        << "  dim_x:       X dimension (default: 3600)\n"
        << "  dim_y:       Y dimension (default: 1800)\n"
        << "  error_bound: 0 < eb < 1 (default: 1e-3)\n"
        << "  mode:        rel | abs | noa (default: rel)\n"
        << "  phase:       compress | decompress (default: compress)\n"
        << "  runs:        integer > 0 (default: 20)\n"
        << "  threshold:   positive float (optional; default: inf)\n";
}

// ── Per-strategy result ───────────────────────────────────────────────────────
struct StrategyResult {
    std::string strat_name;
    size_t      pool_threshold;   // topology-aware threshold set at finalize()
    size_t      peak_memory;      // actual CUDA pool high-water mark after warmup
    size_t      compressed_size;
    double      mean_host_ms;
    float       mean_dag_ms;
    double      min_host_ms;
    float       min_dag_ms;
    double      max_host_ms;
    float       max_dag_ms;
};

// ── Run one strategy through the benchmark loop ───────────────────────────────
static StrategyResult run_strategy(
    MemoryStrategy    strat,
    const std::string& strat_name,
    ProfilePhase      phase,
    const std::string& phase_str,
    float*            d_input,
    size_t            data_bytes,
    float             eb,
    ErrorBoundMode    mode,
    float             threshold,
    int               runs)
{
    std::cout << "\n══ Strategy: " << strat_name << " ══════════════════════\n";

    Pipeline comp(data_bytes, strat, POOL_MULT);
    build_pfpl_pipeline(comp, eb, mode, threshold);
    comp.enableProfiling(true);

    // finalize() (called inside build_pfpl_pipeline) tightens the pool threshold
    // to topo_base × 1.1.  Read back the actual value that was set.
    const size_t pool_threshold = comp.getPoolThreshold();

    void*  d_compressed  = nullptr;
    size_t compressed_sz = 0;

    // Warmup — also establishes the pipeline output buffer for decompress runs.
    comp.compress(d_input, data_bytes, &d_compressed, &compressed_sz, 0);
    cudaDeviceSynchronize();

    // Peak memory is measured after warmup so both preallocate (which allocates
    // at finalize) and minimal (which allocates during compress) are captured at
    // their true high-water mark.
    const size_t peak_mem = comp.getPeakMemoryUsage();

    std::vector<double> host_ms_v;
    std::vector<float>  dag_ms_v;
    host_ms_v.reserve(static_cast<size_t>(runs));
    dag_ms_v.reserve(static_cast<size_t>(runs));
    bool printed_first = false;

    for (int i = 0; i < runs; ++i) {
        const auto t0 = std::chrono::high_resolution_clock::now();

        if (phase == ProfilePhase::Compress) {
            comp.compress(d_input, data_bytes, &d_compressed, &compressed_sz, 0);
            cudaDeviceSynchronize();
        } else {
            void*  d_rec   = nullptr;
            size_t rec_sz  = 0;
            comp.decompress(d_compressed, compressed_sz, &d_rec, &rec_sz, 0);
            cudaDeviceSynchronize();
            // d_rec is pool-owned (default) — do NOT cudaFree.
        }

        const auto t1 = std::chrono::high_resolution_clock::now();
        const double hms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        const float  dms = comp.getLastPerfResult().dag_elapsed_ms;

        if (!printed_first) {
            std::cout << "\n── Library report (run 1, post-warmup) ──────────────────────\n";
            comp.getLastPerfResult().print(std::cout);
            printed_first = true;
        }

        host_ms_v.push_back(hms);
        dag_ms_v.push_back(dms);

        const float host_gbs = static_cast<float>(data_bytes) / (hms * 1e-3) / 1e9f;
        const float dag_gbs  = static_cast<float>(data_bytes) / (dms * 1e-3f) / 1e9f;

        std::cout << "  run " << std::setw(2) << (i + 1) << ":  "
                  << "host " << std::setw(8) << std::fixed << std::setprecision(3)
                  << hms << " ms  "
                  << std::setw(7) << std::setprecision(2) << host_gbs << " GB/s   "
                  << "dag "  << std::setw(8) << std::setprecision(3) << dms << " ms  "
                  << std::setw(7) << std::setprecision(2) << dag_gbs  << " GB/s\n";
    }

    const double mean_h = std::accumulate(host_ms_v.begin(), host_ms_v.end(), 0.0) / runs;
    const float  mean_d = std::accumulate(dag_ms_v.begin(),  dag_ms_v.end(),  0.0f) / runs;
    const double min_h  = *std::min_element(host_ms_v.begin(), host_ms_v.end());
    const float  min_d  = *std::min_element(dag_ms_v.begin(),  dag_ms_v.end());
    const double max_h  = *std::max_element(host_ms_v.begin(), host_ms_v.end());
    const float  max_d  = *std::max_element(dag_ms_v.begin(),  dag_ms_v.end());

    const auto tput = [&](double ms) {
        return static_cast<float>(data_bytes) / (ms * 1e-3) / 1e9f;
    };

    std::cout << "\n── " << strat_name << " " << phase_str << " summary ────────────────────────\n"
              << "  host   mean=" << std::setw(8) << std::setprecision(3) << mean_h << " ms  "
              << "min=" << std::setw(8) << min_h << " ms  "
              << "max=" << std::setw(8) << max_h << " ms\n"
              << "  dag    mean=" << std::setw(8) << mean_d << " ms  "
              << "min=" << std::setw(8) << min_d << " ms  "
              << "max=" << std::setw(8) << max_d << " ms\n"
              << "  Throughput (host mean): " << std::setw(6) << std::setprecision(2)
              << tput(mean_h) << " GB/s\n"
              << "  Throughput (dag  mean): " << std::setw(6)
              << tput(static_cast<double>(mean_d)) << " GB/s\n";

    return {strat_name, pool_threshold, peak_mem, compressed_sz,
            mean_h, mean_d, min_h, min_d, max_h, max_d};
}

int main(int argc, char* argv[]) {
    if (argc < 2) { print_usage(); return 1; }

    const char* input_file = argv[1];
    size_t dim_x = 3600;
    size_t dim_y = 1800;
    float eb = DEFAULT_EB;
    ErrorBoundMode mode = ErrorBoundMode::REL;
    std::string mode_str = "rel";
    ProfilePhase phase = ProfilePhase::Compress;
    std::string phase_str = "compress";
    int runs = DEFAULT_RUNS;
    float threshold = std::numeric_limits<float>::infinity();

    if (argc > 2) dim_x = std::stoull(argv[2]);
    if (argc > 3) dim_y = std::stoull(argv[3]);
    if (argc > 4) {
        eb = std::stof(argv[4]);
        if (eb <= 0.0f) { print_usage(); return 1; }
    }
    if (argc > 5) {
        mode_str = argv[5];
        if      (mode_str == "abs") mode = ErrorBoundMode::ABS;
        else if (mode_str == "noa") mode = ErrorBoundMode::NOA;
        else if (mode_str == "rel") mode = ErrorBoundMode::REL;
        else { std::cerr << "Unknown mode '" << mode_str << "'.\n"; print_usage(); return 1; }
    }
    if (argc > 6) {
        phase_str = argv[6];
        if      (phase_str == "compress")   phase = ProfilePhase::Compress;
        else if (phase_str == "decompress") phase = ProfilePhase::Decompress;
        else { std::cerr << "Unknown phase '" << phase_str << "'.\n"; print_usage(); return 1; }
    }
    if (argc > 7) {
        runs = std::stoi(argv[7]);
        if (runs <= 0) { std::cerr << "runs must be > 0\n"; return 1; }
    }
    if (argc > 8) {
        threshold = std::stof(argv[8]);
        if (threshold <= 0.0f) { std::cerr << "threshold must be positive\n"; return 1; }
    }

    auto [d_input, n] = load_data(input_file, dim_x, dim_y);
    if (!d_input || n == 0) {
        std::cerr << "Dataset not found or unreadable: " << input_file << "\n";
        return 1;
    }

    const size_t data_bytes = n * sizeof(float);

    std::cout << "=== PFPL Memory Strategy Comparison ===\n"
              << "  Dataset:     " << input_file << " (" << dim_x << "x" << dim_y << ")\n"
              << "  Elements:    " << n << "\n"
              << "  Raw size:    " << std::fixed << std::setprecision(2)
              << data_bytes / (1024.0 * 1024.0) << " MB\n"
              << "  Error bound: " << std::scientific << std::setprecision(1)
              << eb << " (" << mode_str << ")\n"
              << "  Phase:       " << phase_str << "\n"
              << "  Runs:        " << runs << "\n"
              << "  Chunk size:  " << CHUNK << " bytes\n"
              << "  Pool mult:   " << std::fixed << POOL_MULT << "x (initial), 1.1x (post-finalize topo)\n";
    if (std::isfinite(threshold))
        std::cout << "  Threshold:   " << threshold << "\n";
    std::cout << "\n";

    const StrategyResult pre = run_strategy(
        MemoryStrategy::PREALLOCATE, "PREALLOCATE",
        phase, phase_str, d_input, data_bytes, eb, mode, threshold, runs);

    const StrategyResult min = run_strategy(
        MemoryStrategy::MINIMAL, "MINIMAL",
        phase, phase_str, d_input, data_bytes, eb, mode, threshold, runs);

    // ── Memory + throughput comparison ───────────────────────────────────────
    const auto tput = [&](float ms) {
        return static_cast<float>(data_bytes) / (ms * 1e-3f) / 1e9f;
    };
    const double mem_savings_pct =
        100.0 * (1.0 - static_cast<double>(min.peak_memory)
                     / static_cast<double>(pre.peak_memory));

    std::cout << "\n══ Memory + Throughput Comparison ══════════════════════════════\n"
              << std::left  << std::setw(20) << "Metric"
              << std::right << std::setw(16) << "PREALLOCATE"
              << std::setw(16) << "MINIMAL"
              << "\n" << std::string(52, '-') << "\n";

    const auto row = [&](const std::string& label, double v_pre, double v_min,
                         const std::string& unit) {
        std::cout << std::left  << std::setw(20) << label
                  << std::right << std::setw(13) << std::fixed << std::setprecision(2)
                  << v_pre << unit
                  << std::setw(13) << v_min << unit << "\n";
    };

    row("Pool threshold",
        pre.pool_threshold / (1024.0 * 1024.0),
        min.pool_threshold / (1024.0 * 1024.0), " MB");
    row("Peak memory used",
        pre.peak_memory    / (1024.0 * 1024.0),
        min.peak_memory    / (1024.0 * 1024.0), " MB");
    row("Input size",
        data_bytes         / (1024.0 * 1024.0),
        data_bytes         / (1024.0 * 1024.0), " MB");
    row("Compressed size",
        pre.compressed_size / (1024.0 * 1024.0),
        min.compressed_size / (1024.0 * 1024.0), " MB");
    row("Compression ratio",
        static_cast<double>(data_bytes) / pre.compressed_size,
        static_cast<double>(data_bytes) / min.compressed_size, "x  ");
    row("Throughput dag mean",
        tput(pre.mean_dag_ms),
        tput(min.mean_dag_ms), " GB/s");
    row("Throughput dag min",
        tput(pre.min_dag_ms),
        tput(min.min_dag_ms), " GB/s");

    std::cout << std::string(52, '-') << "\n"
              << "  MINIMAL peak memory is " << std::setprecision(1)
              << mem_savings_pct << "% lower than PREALLOCATE\n"
              << "  (pool threshold = topo base × 1.1 safety margin after finalize)\n";

    cudaFree(d_input);
    std::cout << "\nDone.\n";
    return 0;
}
