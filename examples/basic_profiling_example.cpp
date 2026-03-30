/**
 * basic_profiling_example — optimization target benchmark
 *
 * Runs the PFPL pipeline (Quantizer → Diff → Bitshuffle → RZE) on CLDHGH and
 * measures the specific overheads identified in docs/performance_improvements.md:
 *
 *   1. Cold-call overhead    — first compress/decompress with no prior warmup
 *   2. Steady-state          — N subsequent calls after warmup
 *   3. Host vs DAG split     — host_elapsed_ms - dag_elapsed_ms = overhead budget
 *   4. Decompress cold/warm  — separate from compress since inv-DAG rebuild happens there
 *   5. Quality verification  — ensures eb is respected at 1e-4 ABS
 *
 * Usage:
 *   ./basic_profiling_example [strategy] [runs] [eb] [warmup]
 *
 *   strategy : preallocate | minimal          (default: preallocate)
 *   runs     : integer > 0                    (default: 10)
 *   eb       : positive float                 (default: 1e-4)
 *   warmup   : none | auto | manual           (default: none)
 *                none   — no warmup; cold call shows full JIT cost
 *                auto   — setWarmupOnFinalize(true); finalize() warms all kernels
 *                manual — explicit p.warmup() call after finalize()
 *
 * Examples:
 *   ./basic_profiling_example preallocate 10 1e-4 none
 *   ./basic_profiling_example preallocate 10 1e-4 auto
 *   ./basic_profiling_example minimal 20 1e-4 manual
 *
 * Nsys capture (GPU+host overhead visible):
 *   nsys profile --trace=cuda,nvtx --capture-range=cudaProfilerApi \
 *        -o cldhgh_opt ./build/bin/basic_profiling_example preallocate 10
 */

#include "fzgpumodules.h"
#include "pipeline/stat.h"
#ifdef FZ_PROFILING_ENABLED
#include <cuda_profiler_api.h>
#include <nvtx3/nvtx3.hpp>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

using namespace fz;

// ── Dataset ───────────────────────────────────────────────────────────────────

static const char*      DATA_PATH  = "/home/skyler/data/SDRB/CESM_ATM_1800x3600/CLDHGH.f32";
static constexpr size_t DIM_X      = 3600;
static constexpr size_t DIM_Y      = 1800;
static constexpr size_t N_ELEMS    = DIM_X * DIM_Y;
static constexpr size_t CHUNK      = 16384;
static constexpr float  POOL_MULT  = 3.0f;

// ── Defaults (overridden by argv) ─────────────────────────────────────────────

static constexpr float DEFAULT_EB   = 1e-4f;
static constexpr int   DEFAULT_RUNS = 10;

// ─────────────────────────────────────────────────────────────────────────────

static float* load_to_device() {
    std::vector<float> h(N_ELEMS);
    std::FILE* fp = std::fopen(DATA_PATH, "rb");
    if (!fp) {
        std::cerr << "[ERROR] cannot open: " << DATA_PATH << "\n";
        std::exit(1);
    }
    size_t got = std::fread(h.data(), sizeof(float), N_ELEMS, fp);
    std::fclose(fp);
    if (got != N_ELEMS) {
        std::cerr << "[ERROR] expected " << N_ELEMS << " floats, got " << got << "\n";
        std::exit(1);
    }
    float* d = nullptr;
    cudaMalloc(&d, N_ELEMS * sizeof(float));
    cudaMemcpy(d, h.data(), N_ELEMS * sizeof(float), cudaMemcpyHostToDevice);
    return d;
}

enum class WarmupMode { NONE, AUTO, MANUAL };

// Build the PFPL pipeline: Quantizer → Diff → Bitshuffle → RZE
// ABS error bound — quant_radius sized for fine quantization at 1e-4
//
// warmup_mode:
//   NONE   — plain finalize(), cold first compress sees full JIT cost
//   AUTO   — setWarmupOnFinalize(true) before finalize(); finalize() warms all kernels
//   MANUAL — finalize() then explicit warmup() call
static void build_pipeline(Pipeline& p, float eb, WarmupMode warmup_mode) {
    auto* quant = p.addStage<QuantizerStage<float, uint32_t>>();
    quant->setErrorBound(eb);
    quant->setErrorBoundMode(ErrorBoundMode::ABS);
    quant->setQuantRadius(1 << 22);  // 4M bins — appropriate for ABS at 1e-4
    quant->setOutlierCapacity(0.05f);
    quant->setZigzagCodes(true);
    quant->setInplaceOutliers(true);

    auto* diff = p.addStage<DifferenceStage<int32_t, uint32_t>>();
    diff->setChunkSize(CHUNK);
    p.connect(diff, quant, "codes");

    auto* bshuf = p.addStage<BitshuffleStage>();
    bshuf->setBlockSize(CHUNK);
    bshuf->setElementWidth(4);
    p.connect(bshuf, diff);

    auto* rze = p.addStage<RZEStage>();
    rze->setChunkSize(CHUNK);
    rze->setLevels(4);
    p.connect(rze, bshuf);

    if (warmup_mode == WarmupMode::AUTO) {
        p.setWarmupOnFinalize(true);
    }
    p.finalize();
    if (warmup_mode == WarmupMode::MANUAL) {
        std::cout << "  Running explicit warmup()...\n";
        const auto tw0 = std::chrono::high_resolution_clock::now();
        p.warmup(/*stream=*/0);
        cudaDeviceSynchronize();
        const auto tw1 = std::chrono::high_resolution_clock::now();
        std::cout << "  Warmup done in "
                  << std::fixed << std::setprecision(1)
                  << std::chrono::duration<double, std::milli>(tw1 - tw0).count()
                  << " ms\n\n";
    }
}

// ── Timing helpers ────────────────────────────────────────────────────────────

using Clock     = std::chrono::high_resolution_clock;
using TimePoint = std::chrono::time_point<Clock>;

static double elapsed_ms(TimePoint t0, TimePoint t1) {
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

static float tput_gbs(size_t bytes, double ms) {
    return static_cast<float>(bytes) / (ms * 1e-3) / 1e9f;
}

// ── Pretty-print helpers ──────────────────────────────────────────────────────

static void print_separator(char c = '-', int width = 66) {
    std::cout << std::string(width, c) << "\n";
}

static void print_header(const std::string& title) {
    print_separator('=');
    std::cout << "  " << title << "\n";
    print_separator('=');
}

static void print_call_row(int run_idx, const char* label,
                           double host_ms, float dag_ms, size_t uncompressed_bytes)
{
    const float host_gbs = tput_gbs(uncompressed_bytes, host_ms);
    const float dag_gbs  = tput_gbs(uncompressed_bytes, static_cast<double>(dag_ms));
    const float overhead_ms = static_cast<float>(host_ms) - dag_ms;

    std::cout << std::fixed
              << "  " << std::setw(3) << std::left << label
              << std::right
              << "  host " << std::setw(8) << std::setprecision(3) << host_ms << " ms"
              << " (" << std::setw(6) << std::setprecision(2) << host_gbs << " GB/s)"
              << "  dag "  << std::setw(7) << std::setprecision(3) << dag_ms  << " ms"
              << " (" << std::setw(6) << dag_gbs  << " GB/s)"
              << "  overhead " << std::setw(6) << std::setprecision(3) << overhead_ms << " ms"
              << "\n";
    (void)run_idx;
}

static void print_summary(const std::string& label,
                           const std::vector<double>& host_ms_v,
                           const std::vector<float>&  dag_ms_v,
                           size_t uncompressed_bytes)
{
    const int n = static_cast<int>(host_ms_v.size());
    if (n == 0) return;

    const double mean_h = std::accumulate(host_ms_v.begin(), host_ms_v.end(), 0.0) / n;
    const float  mean_d = std::accumulate(dag_ms_v.begin(),  dag_ms_v.end(),  0.0f) / n;
    const double min_h  = *std::min_element(host_ms_v.begin(), host_ms_v.end());
    const float  min_d  = *std::min_element(dag_ms_v.begin(),  dag_ms_v.end());
    const double max_h  = *std::max_element(host_ms_v.begin(), host_ms_v.end());
    const float  max_d  = *std::max_element(dag_ms_v.begin(),  dag_ms_v.end());

    const float mean_overhead = static_cast<float>(mean_h) - mean_d;
    const float overhead_pct  = (mean_d > 0.0f)
        ? 100.0f * mean_overhead / static_cast<float>(mean_h) : 0.0f;

    std::cout << "\n  " << label << " summary (" << n << " runs):\n";
    print_separator();
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "  Host (ms)  mean=" << std::setw(8) << mean_h
              << "  min=" << std::setw(8) << min_h
              << "  max=" << std::setw(8) << max_h << "\n";
    std::cout << "  DAG  (ms)  mean=" << std::setw(8) << mean_d
              << "  min=" << std::setw(8) << min_d
              << "  max=" << std::setw(8) << max_d << "\n";
    std::cout << "  Host overhead (mean): " << std::setw(7) << std::setprecision(3)
              << mean_overhead << " ms  (" << std::setprecision(1)
              << overhead_pct << "% of host time)\n";
    std::cout << "  Throughput  host-mean=" << std::setw(6) << std::setprecision(2)
              << tput_gbs(uncompressed_bytes, mean_h)
              << " GB/s   dag-mean=" << std::setw(6)
              << tput_gbs(uncompressed_bytes, static_cast<double>(mean_d)) << " GB/s\n";
    std::cout << "  Throughput  host-best=" << std::setw(6)
              << tput_gbs(uncompressed_bytes, min_h)
              << " GB/s   dag-best=" << std::setw(6)
              << tput_gbs(uncompressed_bytes, static_cast<double>(min_d)) << " GB/s\n";
}

// ── Quality check ─────────────────────────────────────────────────────────────

static void print_quality(const float* d_orig, const float* d_recon,
                           size_t n, float eb)
{
    auto stats = calculateStatistics<float>(d_orig, d_recon, n);
    const int W = 12;
    std::cout << std::fixed;
    std::cout << "  ┌─ Reconstruction quality ────────────────────────\n";
    std::cout << "  │  PSNR        " << std::setw(W) << std::setprecision(4)
              << stats.psnr << "  dB\n";
    std::cout << "  │  Max error   " << std::setw(W) << std::setprecision(2)
              << std::scientific << stats.max_error
              << "  (bound=" << eb << ", ratio="
              << std::fixed << std::setprecision(3) << stats.max_error / eb << "x)\n";
    std::cout << "  │  NRMSE       " << std::setw(W) << std::setprecision(6)
              << std::fixed << stats.nrmse << "\n";
    std::cout << "  │  Value range " << std::setw(W) << stats.value_range << "\n";
    std::cout << "  └────────────────────────────────────────────────\n";
}

// ── Argument parsing ──────────────────────────────────────────────────────────

static void print_usage(const char* argv0) {
    std::cerr << "Usage: " << argv0 << " [strategy] [runs] [eb] [warmup]\n"
              << "  strategy : preallocate | minimal  (default: preallocate)\n"
              << "  runs     : integer > 0             (default: " << DEFAULT_RUNS << ")\n"
              << "  eb       : positive float           (default: " << DEFAULT_EB << ")\n"
              << "  warmup   : none | auto | manual    (default: none)\n";
}

// =============================================================================
//  main
// =============================================================================

int main(int argc, char* argv[]) {
    MemoryStrategy strategy    = MemoryStrategy::PREALLOCATE;
    std::string    strat_str   = "preallocate";
    int            runs        = DEFAULT_RUNS;
    float          eb          = DEFAULT_EB;
    WarmupMode     warmup_mode = WarmupMode::NONE;
    std::string    warmup_str  = "none";

    if (argc > 1) {
        strat_str = argv[1];
        if      (strat_str == "preallocate") strategy = MemoryStrategy::PREALLOCATE;
        else if (strat_str == "minimal")     strategy = MemoryStrategy::MINIMAL;
        else { print_usage(argv[0]); return 1; }
    }
    if (argc > 2) {
        runs = std::stoi(argv[2]);
        if (runs <= 0) { std::cerr << "runs must be > 0\n"; return 1; }
    }
    if (argc > 3) {
        eb = std::stof(argv[3]);
        if (eb <= 0.0f) { std::cerr << "eb must be positive\n"; return 1; }
    }
    if (argc > 4) {
        warmup_str = argv[4];
        if      (warmup_str == "none")   warmup_mode = WarmupMode::NONE;
        else if (warmup_str == "auto")   warmup_mode = WarmupMode::AUTO;
        else if (warmup_str == "manual") warmup_mode = WarmupMode::MANUAL;
        else { print_usage(argv[0]); return 1; }
    }

    const size_t data_bytes = N_ELEMS * sizeof(float);

    // ── Banner ──────────────────────────────────────────────────────────────
    print_header("FZGPUModules optimization benchmark — PFPL on CLDHGH");
    std::cout << "  Dataset  : CESM ATM CLDHGH " << DIM_X << "x" << DIM_Y
              << "  (" << std::fixed << std::setprecision(2)
              << data_bytes / (1024.0 * 1024.0) << " MB)\n"
              << "  Pipeline : Quantizer(ABS) -> Diff -> Bitshuffle -> RZE\n"
              << "  Strategy : " << strat_str << "\n"
              << "  EB       : " << std::scientific << std::setprecision(1) << eb << " (ABS)\n"
              << "  Runs     : " << runs << " (post-cold-call)\n"
              << "  Warmup   : " << warmup_str << "\n"
              << "  Chunk    : " << CHUNK << " bytes\n\n";

    // ── Load data ────────────────────────────────────────────────────────────
    std::cout << "Loading CLDHGH.f32...\n";
    float* d_input = load_to_device();
    std::cout << "  Loaded " << std::fixed << std::setprecision(2)
              << data_bytes / (1024.0 * 1024.0) << " MB to device\n\n";

    // ── Build pipeline ───────────────────────────────────────────────────────
    Pipeline p(data_bytes, strategy, POOL_MULT);
    p.setDims(DIM_X, DIM_Y, 1);
    p.enableProfiling(true);
    build_pipeline(p, eb, warmup_mode);

    std::cout << "  Pool threshold after finalize: "
              << std::setprecision(2) << p.getPoolThreshold() / (1024.0 * 1024.0) << " MB\n\n";

    void*  d_compressed  = nullptr;
    size_t compressed_sz = 0;

#ifdef FZ_PROFILING_ENABLED
    cudaProfilerStart();
#endif

    // =========================================================================
    //  COMPRESSION
    // =========================================================================
    print_header("COMPRESS");

    // ── Cold call (or first call after warmup) ────────────────────────────
    std::cout << "  First call ("
              << (warmup_mode == WarmupMode::NONE
                      ? "cold — includes kernel JIT"
                      : "warm — kernels already JIT-compiled by warmup()")
              << "):\n";
    {
#ifdef FZ_PROFILING_ENABLED
        nvtx3::scoped_range r{"compress::cold"};
#endif
        const auto t0 = Clock::now();
        p.compress(d_input, data_bytes, &d_compressed, &compressed_sz, 0);
        cudaDeviceSynchronize();
        const auto t1 = Clock::now();

        const double hms = elapsed_ms(t0, t1);
        const float  dms = p.getLastPerfResult().dag_elapsed_ms;
        print_call_row(0, "cold", hms, dms, data_bytes);

        std::cout << "\n  Stage breakdown (first call):\n";
        p.getLastPerfResult().print(std::cout);
    }

    std::cout << "\n  Compressed: " << std::fixed << std::setprecision(2)
              << compressed_sz / (1024.0 * 1024.0) << " MB  (ratio "
              << std::setprecision(2)
              << static_cast<double>(data_bytes) / compressed_sz << "x)\n\n";

    // ── Subsequent calls (steady-state) ───────────────────────────────────
    std::cout << "  Subsequent compress calls (steady-state):\n";
    {
        std::vector<double> host_ms_v;
        std::vector<float>  dag_ms_v;
        host_ms_v.reserve(static_cast<size_t>(runs));
        dag_ms_v.reserve(static_cast<size_t>(runs));

        for (int i = 0; i < runs; ++i) {
#ifdef FZ_PROFILING_ENABLED
            const std::string rname = "compress::run" + std::to_string(i + 1);
            nvtx3::scoped_range r{rname.c_str()};
#endif
            const auto t0 = Clock::now();
            p.compress(d_input, data_bytes, &d_compressed, &compressed_sz, 0);
            cudaDeviceSynchronize();
            const auto t1 = Clock::now();

            const double hms = elapsed_ms(t0, t1);
            const float  dms = p.getLastPerfResult().dag_elapsed_ms;
            host_ms_v.push_back(hms);
            dag_ms_v.push_back(dms);

            char label[16];
            std::snprintf(label, sizeof(label), "%d", i + 1);
            print_call_row(i + 1, label, hms, dms, data_bytes);

            if (i == 0) {
                std::cout << "\n  Stage breakdown (run 1 steady-state):\n";
                p.getLastPerfResult().print(std::cout);
                std::cout << "\n";
            }
        }

        print_summary("compress", host_ms_v, dag_ms_v, data_bytes);
    }

    // =========================================================================
    //  DECOMPRESSION
    // =========================================================================
    print_header("DECOMPRESS");

    // ── First decompress call ─────────────────────────────────────────────
    std::cout << "  First call (inv-DAG build always happens on first decompress):\n";
    void* d_recon_check = nullptr;
    size_t recon_sz     = 0;
    {
#ifdef FZ_PROFILING_ENABLED
        nvtx3::scoped_range r{"decompress::cold"};
#endif
        const auto t0 = Clock::now();
        p.decompress(d_compressed, compressed_sz, &d_recon_check, &recon_sz, 0);
        cudaDeviceSynchronize();
        const auto t1 = Clock::now();

        const double hms = elapsed_ms(t0, t1);
        const float  dms = p.getLastPerfResult().dag_elapsed_ms;
        print_call_row(0, "cold", hms, dms, recon_sz);

        std::cout << "\n  Stage breakdown (first decompress):\n";
        p.getLastPerfResult().print(std::cout);
    }
    std::cout << "\n";

    // ── Subsequent decompress calls ───────────────────────────────────────
    std::cout << "  Subsequent decompress calls (steady-state):\n";
    {
        std::vector<double> host_ms_v;
        std::vector<float>  dag_ms_v;
        host_ms_v.reserve(static_cast<size_t>(runs));
        dag_ms_v.reserve(static_cast<size_t>(runs));

        for (int i = 0; i < runs; ++i) {
#ifdef FZ_PROFILING_ENABLED
            const std::string rname = "decompress::run" + std::to_string(i + 1);
            nvtx3::scoped_range r{rname.c_str()};
#endif
            void*  d_rec   = nullptr;
            size_t rec_sz  = 0;
            const auto t0 = Clock::now();
            p.decompress(d_compressed, compressed_sz, &d_rec, &rec_sz, 0);
            cudaDeviceSynchronize();
            const auto t1 = Clock::now();

            const double hms = elapsed_ms(t0, t1);
            const float  dms = p.getLastPerfResult().dag_elapsed_ms;
            host_ms_v.push_back(hms);
            dag_ms_v.push_back(dms);

            char label[16];
            std::snprintf(label, sizeof(label), "%d", i + 1);
            print_call_row(i + 1, label, hms, dms, rec_sz);

            if (i == 0) {
                std::cout << "\n  Stage breakdown (run 1 steady-state):\n";
                p.getLastPerfResult().print(std::cout);
                std::cout << "\n";
            }

            if (d_rec) cudaFree(d_rec);
        }

        print_summary("decompress", host_ms_v, dag_ms_v, recon_sz);
    }

#ifdef FZ_PROFILING_ENABLED
    cudaProfilerStop();
#endif

    // =========================================================================
    //  QUALITY
    // =========================================================================
    print_header("QUALITY");
    if (d_recon_check && recon_sz == data_bytes) {
        print_quality(d_input, static_cast<const float*>(d_recon_check), N_ELEMS, eb);
    } else {
        std::cout << "  [WARN] size mismatch — skipping quality check\n"
                  << "         got " << recon_sz << " bytes, expected " << data_bytes << "\n";
    }

    // =========================================================================
    //  OVERHEAD SUMMARY
    // =========================================================================
    // Print a concise table comparing cold vs mean-steady on both phases so
    // the targets from performance_improvements.md are easy to track.
    print_header("OVERHEAD BUDGET (targets from performance_improvements.md)");
    std::cout << "  These numbers are what the optimizations aim to reduce:\n\n";
    std::cout << "  ┌─ What to look at ─────────────────────────────────────────────\n"
              << "  │  cold - mean(steady):          module JIT overhead  (target §1, §7)\n"
              << "  │  host_ms - dag_ms (compress):  concat overhead      (target §2)\n"
              << "  │  cold_decomp - mean(decomp):   inv-DAG rebuild      (target §3)\n"
              << "  │  host_ms - dag_ms (decomp):    inv-DAG build+copy   (target §3, §4)\n"
              << "  └───────────────────────────────────────────────────────────────\n";

    // ── Cleanup ──────────────────────────────────────────────────────────────
    if (d_recon_check) cudaFree(d_recon_check);
    cudaFree(d_input);

    std::cout << "\nDone.\n";
    return 0;
}
