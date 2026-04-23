/**
 * PFPL CUDA Graph Capture profiling example.
 *
 * Runs the same PFPL compression pipeline (Quantizer → Difference → Bitshuffle
 * → RZE) under three strategies and prints a side-by-side comparison:
 *
 *   MINIMAL      — on-demand alloc/free per call (lowest memory, highest overhead)
 *   PREALLOCATE  — all buffers fixed at finalize() (fast, standard path)
 *   GRAPH        — PREALLOCATE + CUDA Graph capture (lowest dispatch overhead)
 *
 * GRAPH mode records the entire dag_->execute() body as a CUDA Graph during
 * setup.  Each compress() call then does:
 *   1. D2D copy: user input → fixed graph input slot
 *   2. cudaGraphLaunch() — replays all kernels with zero CPU dispatch
 *   3. postStreamSync + output collection (unchanged CPU-side work)
 *
 * Usage:
 *   ./build/bin/examples/pfpl_graph_capture <file> [dim_x [dim_y [error_bound [mode [runs]]]]]
 *
 * Examples:
 *   ./build/bin/examples/pfpl_graph_capture data/CLDHGH.f32 3600 1800
 *   ./build/bin/examples/pfpl_graph_capture data/CLDHGH.f32 3600 1800 1e-3 rel 30
 *   ./build/bin/examples/pfpl_graph_capture data/CLDHGH.f32 3600 1800 1e-4 abs 20
 */

#include "fzgpumodules.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

using namespace fz;

static constexpr float  DEFAULT_EB   = 1e-3f;
static constexpr size_t CHUNK        = 16384;
static constexpr int    DEFAULT_RUNS = 30;
static constexpr float  POOL_MULT    = 3.0f;

// ── Helpers ───────────────────────────────────────────────────────────────────
static std::pair<float*, size_t> load_data(const char* path, size_t dim_x, size_t dim_y) {
    const size_t n = dim_x * dim_y;
    std::vector<float> h(n);
    std::FILE* fp = std::fopen(path, "rb");
    if (!fp) return {nullptr, 0};
    const size_t r = std::fread(h.data(), sizeof(float), n, fp);
    std::fclose(fp);
    if (r != n) return {nullptr, 0};
    float* d = nullptr;
    cudaMalloc(&d, n * sizeof(float));
    cudaMemcpy(d, h.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    return {d, n};
}

// Build the PFPL stage graph onto `p`.  Does NOT call finalize() so the
// caller can set graph mode flags before finalizing.
static void build_pfpl_stages(
    Pipeline& p,
    float eb,
    ErrorBoundMode mode = ErrorBoundMode::REL)
{
    auto* quant = p.addStage<QuantizerStage<float, uint32_t>>();
    quant->setErrorBound(eb);
    quant->setErrorBoundMode(mode);
    quant->setQuantRadius(mode == ErrorBoundMode::ABS ? (1 << 22) : 32768);
    quant->setOutlierCapacity(0.05f);
    quant->setZigzagCodes(true);
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
}

// ── Per-strategy result ───────────────────────────────────────────────────────
struct StrategyResult {
    std::string strat_name;
    size_t      pool_threshold;
    size_t      peak_memory;
    size_t      compressed_size;
    double      mean_host_ms;
    float       mean_dag_ms;
    double      min_host_ms;
    float       min_dag_ms;
    double      max_host_ms;
    float       max_dag_ms;
};

// ── Normal path (MINIMAL or PREALLOCATE) ─────────────────────────────────────
static StrategyResult run_normal(
    MemoryStrategy     strat,
    const std::string& strat_name,
    float*             d_input,
    size_t             data_bytes,
    float              eb,
    ErrorBoundMode     mode,
    int                runs)
{
    std::cout << "\n══ Strategy: " << strat_name << " ══════════════════════\n";

    Pipeline comp(data_bytes, strat, POOL_MULT);
    build_pfpl_stages(comp, eb, mode);
    comp.finalize();
    comp.enableProfiling(true);

    const size_t pool_threshold = comp.getPoolThreshold();

    void*  d_out  = nullptr;
    size_t out_sz = 0;

    // Warmup — JIT compiles all kernels.
    comp.compress(d_input, data_bytes, &d_out, &out_sz, 0);
    cudaDeviceSynchronize();
    const size_t peak_mem = comp.getPeakMemoryUsage();

    std::vector<double> host_ms_v;
    std::vector<float>  dag_ms_v;
    host_ms_v.reserve(static_cast<size_t>(runs));
    dag_ms_v.reserve(static_cast<size_t>(runs));
    bool printed_first = false;

    for (int i = 0; i < runs; ++i) {
        const auto t0 = std::chrono::high_resolution_clock::now();
        comp.compress(d_input, data_bytes, &d_out, &out_sz, 0);
        cudaDeviceSynchronize();
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

        const float hgbs = static_cast<float>(data_bytes) / (hms * 1e-3) / 1e9f;
        const float dgbs = static_cast<float>(data_bytes) / (dms * 1e-3f) / 1e9f;
        std::cout << "  run " << std::setw(2) << (i + 1)
                  << ":  host " << std::setw(8) << std::fixed << std::setprecision(3) << hms << " ms"
                  << "  " << std::setw(7) << std::setprecision(2) << hgbs << " GB/s"
                  << "   dag "  << std::setw(8) << std::setprecision(3) << dms << " ms"
                  << "  " << std::setw(7) << std::setprecision(2) << dgbs << " GB/s\n";
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
    std::cout << "\n── " << strat_name << " compress summary ──────────────────────────\n"
              << "  host   mean=" << std::setw(8) << std::setprecision(3) << mean_h << " ms"
              << "  min=" << std::setw(8) << min_h << " ms"
              << "  max=" << std::setw(8) << max_h << " ms\n"
              << "  dag    mean=" << std::setw(8) << mean_d << " ms"
              << "  min=" << std::setw(8) << min_d << " ms"
              << "  max=" << std::setw(8) << max_d << " ms\n"
              << "  Throughput (host mean): " << std::setw(6) << std::setprecision(2)
              << tput(mean_h) << " GB/s\n"
              << "  Throughput (dag  mean): " << std::setw(6) << tput(mean_d) << " GB/s\n";

    return {strat_name, pool_threshold, peak_mem, out_sz,
            mean_h, mean_d, min_h, min_d, max_h, max_d};
}

// ── Graph capture path ────────────────────────────────────────────────────────
static StrategyResult run_graph(
    float*             d_input,
    size_t             data_bytes,
    float              eb,
    ErrorBoundMode     mode,
    int                runs)
{
    const std::string strat_name = "GRAPH";
    std::cout << "\n══ Strategy: " << strat_name << " ══════════════════════\n";

    // Graph mode requires PREALLOCATE and a non-zero input size hint.
    Pipeline comp(data_bytes, MemoryStrategy::PREALLOCATE, POOL_MULT);
    build_pfpl_stages(comp, eb, mode);

    // enableGraphMode() must be called before finalize().
    comp.enableGraphMode(true);
    comp.finalize();
    comp.enableProfiling(true);

    const size_t pool_threshold = comp.getPoolThreshold();

    // CUDA Graph capture requires a non-default stream.  The legacy default
    // stream (stream 0) cannot be captured; cudaStreamBeginCapture would
    // return cudaErrorStreamCaptureUnsupported.  All subsequent compress()
    // calls in graph mode must use the same stream so the graph launches on
    // the correct stream where the D2D input copy was enqueued.
    cudaStream_t graph_stream = nullptr;
    cudaStreamCreate(&graph_stream);

    // Warmup: JIT-compile all kernels with a dummy compress() pass.
    // warmup() resets the DAG back to its post-finalize state afterward.
    // Use the graph stream for warmup so the JIT-compiled kernels are
    // resident for the capture pass.
    comp.warmup(graph_stream);

    // Capture: record the entire execute() body as a CUDA Graph.
    // Must happen after warmup() so the graph bakes in the compiled SASS,
    // and before any real compress() call.
    comp.captureGraph(graph_stream);
    cudaStreamSynchronize(graph_stream);
    std::cout << "  Graph captured successfully.\n";

    // Measure peak memory after both warmup and capture.
    const size_t peak_mem = comp.getPeakMemoryUsage();

    void*  d_out  = nullptr;
    size_t out_sz = 0;

    // One additional un-timed launch to populate d_out / out_sz and establish
    // baseline state before the benchmark loop.
    comp.compress(d_input, data_bytes, &d_out, &out_sz, graph_stream);
    cudaStreamSynchronize(graph_stream);

    std::vector<double> host_ms_v;
    std::vector<float>  dag_ms_v;
    host_ms_v.reserve(static_cast<size_t>(runs));
    dag_ms_v.reserve(static_cast<size_t>(runs));
    bool printed_first = false;

    for (int i = 0; i < runs; ++i) {
        const auto t0 = std::chrono::high_resolution_clock::now();
        comp.compress(d_input, data_bytes, &d_out, &out_sz, graph_stream);
        cudaStreamSynchronize(graph_stream);
        const auto t1 = std::chrono::high_resolution_clock::now();

        const double hms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        const float  dms = comp.getLastPerfResult().dag_elapsed_ms;

        if (!printed_first) {
            // dag_elapsed_ms = host chrono around cudaGraphLaunch (not GPU compute).
            // Stage CUDA event timings show -1: events are baked into the graph
            // node list and their relative timestamps are not host-queryable after
            // replay via cudaEventElapsedTime.  Use Nsight Systems for per-stage
            // graph breakdown.
            std::cout << "\n── Library report (run 1, post-capture) ─────────────────────\n";
            std::cout << "  Note: dag_elapsed_ms = cudaGraphLaunch dispatch latency only;\n"
                      << "        stage timings (-1) require Nsight for graph-mode breakdown.\n";
            comp.getLastPerfResult().print(std::cout);
            printed_first = true;
        }

        host_ms_v.push_back(hms);
        dag_ms_v.push_back(dms);

        const float hgbs = static_cast<float>(data_bytes) / (hms * 1e-3) / 1e9f;
        const float dgbs = static_cast<float>(data_bytes) / (dms * 1e-3f) / 1e9f;
        std::cout << "  run " << std::setw(2) << (i + 1)
                  << ":  host " << std::setw(8) << std::fixed << std::setprecision(3) << hms << " ms"
                  << "  " << std::setw(7) << std::setprecision(2) << hgbs << " GB/s"
                  << "   dag "  << std::setw(8) << std::setprecision(3) << dms << " ms"
                  << "  " << std::setw(7) << std::setprecision(2) << dgbs << " GB/s\n";
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
    std::cout << "\n── " << strat_name << " compress summary ──────────────────────────\n"
              << "  host   mean=" << std::setw(8) << std::setprecision(3) << mean_h << " ms"
              << "  min=" << std::setw(8) << min_h << " ms"
              << "  max=" << std::setw(8) << max_h << " ms\n"
              << "  dag    mean=" << std::setw(8) << mean_d << " ms"
              << "  min=" << std::setw(8) << min_d << " ms"
              << "  max=" << std::setw(8) << max_d << " ms\n"
              << "  Throughput (host mean): " << std::setw(6) << std::setprecision(2)
              << tput(mean_h) << " GB/s\n"
              << "  Throughput (dag  mean): " << std::setw(6) << tput(mean_d) << " GB/s\n";

    cudaStreamDestroy(graph_stream);

    return {strat_name, pool_threshold, peak_mem, out_sz,
            mean_h, mean_d, min_h, min_d, max_h, max_d};
}

// ── Usage ─────────────────────────────────────────────────────────────────────
static void print_usage() {
    std::cerr
        << "Usage: pfpl_graph_capture <file> [dim_x [dim_y [error_bound [mode [runs]]]]]\n"
        << "  file:        path to float32 binary input file (required)\n"
        << "  dim_x:       X dimension (default: 3600)\n"
        << "  dim_y:       Y dimension (default: 1800)\n"
        << "  error_bound: 0 < eb < 1 (default: 1e-3)\n"
        << "  mode:        rel | abs | noa (default: rel)\n"
        << "  runs:        integer > 0 (default: 30)\n";
}

// ── main ──────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    if (argc < 2) { print_usage(); return 1; }

    const char* input_file = argv[1];
    size_t dim_x = 3600;
    size_t dim_y = 1800;
    float eb = DEFAULT_EB;
    ErrorBoundMode mode = ErrorBoundMode::REL;
    std::string mode_str = "rel";
    int runs = DEFAULT_RUNS;

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
        runs = std::stoi(argv[6]);
        if (runs <= 0) { std::cerr << "runs must be > 0\n"; return 1; }
    }

    auto [d_input, n] = load_data(input_file, dim_x, dim_y);
    if (!d_input || n == 0) {
        std::cerr << "Dataset not found or unreadable: " << input_file << "\n";
        return 1;
    }
    const size_t data_bytes = n * sizeof(float);

    std::cout << "=== PFPL Graph Capture Comparison ===\n"
              << "  Dataset:     " << input_file << " (" << dim_x << "x" << dim_y << ")\n"
              << "  Elements:    " << n << "\n"
              << "  Raw size:    " << std::fixed << std::setprecision(2)
              << data_bytes / (1024.0 * 1024.0) << " MB\n"
              << "  Error bound: " << std::scientific << std::setprecision(1)
              << eb << " (" << mode_str << ")\n"
              << "  Runs:        " << runs << " (+ 1 warmup + 1 untimed pre-loop)\n"
              << "  Strategies:  MINIMAL | PREALLOCATE | GRAPH\n"
              << "  Chunk size:  " << CHUNK << " bytes\n"
              << "  Pool mult:   " << std::fixed << POOL_MULT
              << "x (initial), 1.1x (post-finalize topo)\n\n";

    const StrategyResult r_min  = run_normal(MemoryStrategy::MINIMAL,      "MINIMAL",
                                             d_input, data_bytes, eb, mode, runs);
    const StrategyResult r_pre  = run_normal(MemoryStrategy::PREALLOCATE,   "PREALLOCATE",
                                             d_input, data_bytes, eb, mode, runs);
    const StrategyResult r_graph = run_graph(d_input, data_bytes, eb, mode, runs);

    // ── 3-column comparison table ─────────────────────────────────────────────
    const auto tput = [&](float ms) {
        return static_cast<float>(data_bytes) / (ms * 1e-3f) / 1e9f;
    };

    std::cout << "\n══ Comparison: MINIMAL vs PREALLOCATE vs GRAPH ══════════════════════════════\n"
              << std::left  << std::setw(22) << "Metric"
              << std::right << std::setw(14) << "MINIMAL"
              << std::setw(14) << "PREALLOCATE"
              << std::setw(14) << "GRAPH"
              << "\n" << std::string(64, '-') << "\n";

    const auto row3 = [&](const std::string& label,
                          double v_min, double v_pre, double v_graph,
                          const std::string& unit, int precision = 2) {
        std::cout << std::left  << std::setw(22) << label
                  << std::right << std::fixed << std::setprecision(precision)
                  << std::setw(11) << v_min  << unit
                  << std::setw(11) << v_pre  << unit
                  << std::setw(11) << v_graph << unit << "\n";
    };

    row3("Pool threshold (MB)",
         r_min.pool_threshold  / (1024.0 * 1024.0),
         r_pre.pool_threshold  / (1024.0 * 1024.0),
         r_graph.pool_threshold / (1024.0 * 1024.0), " MB");
    row3("Peak memory (MB)",
         r_min.peak_memory  / (1024.0 * 1024.0),
         r_pre.peak_memory  / (1024.0 * 1024.0),
         r_graph.peak_memory / (1024.0 * 1024.0), " MB");
    row3("Compressed size (MB)",
         r_min.compressed_size  / (1024.0 * 1024.0),
         r_pre.compressed_size  / (1024.0 * 1024.0),
         r_graph.compressed_size / (1024.0 * 1024.0), " MB");
    row3("Compression ratio",
         static_cast<double>(data_bytes) / r_min.compressed_size,
         static_cast<double>(data_bytes) / r_pre.compressed_size,
         static_cast<double>(data_bytes) / r_graph.compressed_size, "x  ");

    std::cout << std::string(64, '-') << "\n";

    row3("Host mean (ms)",  r_min.mean_host_ms, r_pre.mean_host_ms, r_graph.mean_host_ms, " ms", 3);
    row3("Host min  (ms)",  r_min.min_host_ms,  r_pre.min_host_ms,  r_graph.min_host_ms,  " ms", 3);
    row3("DAG  mean (ms)",
         static_cast<double>(r_min.mean_dag_ms),
         static_cast<double>(r_pre.mean_dag_ms),
         static_cast<double>(r_graph.mean_dag_ms), " ms", 3);
    row3("DAG  min  (ms)",
         static_cast<double>(r_min.min_dag_ms),
         static_cast<double>(r_pre.min_dag_ms),
         static_cast<double>(r_graph.min_dag_ms),  " ms", 3);

    std::cout << std::string(64, '-') << "\n";

    row3("Throughput host mean",
         static_cast<float>(data_bytes) / (r_min.mean_host_ms  * 1e-3) / 1e9f,
         static_cast<float>(data_bytes) / (r_pre.mean_host_ms  * 1e-3) / 1e9f,
         static_cast<float>(data_bytes) / (r_graph.mean_host_ms * 1e-3) / 1e9f, " GB/s");
    row3("Throughput dag  mean",
         tput(r_min.mean_dag_ms),
         tput(r_pre.mean_dag_ms),
         tput(r_graph.mean_dag_ms), " GB/s");
    row3("Throughput dag  min",
         tput(r_min.min_dag_ms),
         tput(r_pre.min_dag_ms),
         tput(r_graph.min_dag_ms),  " GB/s");

    std::cout << std::string(64, '-') << "\n";

    // Speedup of GRAPH over PREALLOCATE
    const double spdup_host = r_pre.mean_host_ms / r_graph.mean_host_ms;
    const double spdup_dag  = r_pre.mean_dag_ms  / r_graph.mean_dag_ms;
    std::cout << std::fixed << std::setprecision(2)
              << "  GRAPH vs PREALLOCATE:  host " << spdup_host << "x faster"
              << "   dag " << spdup_dag << "x faster\n"
              << "  (DAG speedup reflects reduced CPU dispatch; GPU compute is identical)\n";

    cudaFree(d_input);
    std::cout << "\nDone.\n";
    return 0;
}
