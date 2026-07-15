/**
 * pfpl_profile — PFPL pipeline benchmark
 *
 * Pipeline (from examples/presets/pfpl.toml):
 *   Quantizer(NOA, zigzag, r=32768, cap=10%) → Difference(chunk=16384)
 *   → Bitshuffle(ew=4, block=16384) → RZE(w=1)
 *
 * The SZ-style predictor-based workhorse (aka SZ3's default pipeline).
 * Adds a delta predictor (Difference) over the raw speed preset and uses a
 * larger quantizer radius.  Bitshuffle + RZE form the primary lossless
 * back-end bottleneck worth measuring with ncu.
 *
 * Phases:
 *   Phase A — PREALLOCATE: cold + N steady-state compress and decompress
 *   Phase B — CUDA Graph:  capture + N replay compress (NOA value_base injected)
 *
 * Usage:
 *   ./fzgmod-profile-pfpl [runs] [eb]
 *   runs : integer > 0       (default: 10)
 *   eb   : positive float    (default: 1e-4, interpreted as NOA)
 *
 * Nsys:
 *   nsys profile --trace=cuda,nvtx --capture-range=cudaProfilerApi \
 *        -o pfpl ./build_profiling/bin/profiling/fzgmod-profile-pfpl
 *   nsys-ui pfpl.nsys-rep
 */

#include "fzgpumodules.h"
#include "pipeline/perf.h"
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
#include <numeric>
#include <string>
#include <vector>

using namespace fz;
using Clock = std::chrono::high_resolution_clock;

static const char*      DATA_PATH = "/home/skyler/data/SDRB/CESM_ATM_1800x3600/CLDHGH.f32";
static constexpr size_t DIM_X     = 3600;
static constexpr size_t DIM_Y     = 1800;
static constexpr size_t N_ELEMS   = DIM_X * DIM_Y;
static constexpr size_t CHUNK     = 16384;

static constexpr float DEFAULT_EB   = 1e-4f;
static constexpr int   DEFAULT_RUNS = 10;
static constexpr float POOL_MULT    = 4.0f;

struct LoadResult { float* d_ptr; float data_range; float max_abs; };

static LoadResult load_to_device(size_t data_bytes) {
    std::vector<float> h(N_ELEMS);
    std::FILE* fp = std::fopen(DATA_PATH, "rb");
    if (!fp) { std::cerr << "[ERROR] cannot open: " << DATA_PATH << "\n"; std::exit(1); }
    const size_t got = std::fread(h.data(), sizeof(float), N_ELEMS, fp);
    std::fclose(fp);
    if (got != N_ELEMS) { std::cerr << "[ERROR] expected " << N_ELEMS << " floats, got " << got << "\n"; std::exit(1); }
    auto [it_min, it_max] = std::minmax_element(h.begin(), h.end());
    const float vmin = *it_min, vmax = *it_max;
    float* d = nullptr;
    cudaMalloc(&d, data_bytes);
    cudaMemcpy(d, h.data(), data_bytes, cudaMemcpyHostToDevice);
    return {d, vmax - vmin, std::max(std::abs(vmin), std::abs(vmax))};
}

// Returns the quant stage pointer so the caller can inject value_base for graph capture.
static QuantizerStage<float, uint32_t>* build_pfpl(Pipeline& p, float eb) {
    auto* quant = p.addStage<QuantizerStage<float, uint32_t>>();
    quant->setErrorBound(eb);
    quant->setErrorBoundMode(ErrorBoundMode::NOA);
    quant->setQuantRadius(32768);
    quant->setOutlierCapacity(0.1f);
    quant->setZigzagCodes(true);

    auto* diff = p.addStage<DifferenceStage<int32_t, uint32_t>>();
    diff->setChunkSize(CHUNK);
    p.connect(diff, quant, "codes");

    auto* bshuf = p.addStage<BitshuffleStage>();
    bshuf->setElementWidth(4);
    bshuf->setBlockSize(CHUNK);
    p.connect(bshuf, diff);

    auto* rze = p.addStage<RZEStage>();
    rze->setWordSize(1);
    p.connect(rze, bshuf);

    return quant;
}

static double elapsed_ms(std::chrono::time_point<Clock> t0,
                         std::chrono::time_point<Clock> t1) {
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

static double tput_gbs(size_t bytes, double ms) {
    return static_cast<double>(bytes) / (ms * 1e-3) / 1e9;
}

static void print_sep(char c = '-', int w = 68) { std::cout << std::string(w, c) << "\n"; }

static void print_header(const std::string& s) {
    print_sep('=');
    std::cout << "  " << s << "\n";
    print_sep('=');
}

static void print_row(const char* label, double host_ms, float dag_ms, size_t bytes) {
    std::cout << std::fixed
              << "  " << std::setw(4) << std::left << label << std::right
              << "  host " << std::setw(8) << std::setprecision(3) << host_ms << " ms"
              << " (" << std::setw(6) << std::setprecision(2) << tput_gbs(bytes, host_ms) << " GB/s)"
              << "  dag "  << std::setw(7) << std::setprecision(3) << dag_ms  << " ms"
              << " (" << std::setw(6) << tput_gbs(bytes, dag_ms)  << " GB/s)"
              << "  ovhd "  << std::setw(6) << std::setprecision(3)
              << static_cast<float>(host_ms) - dag_ms << " ms\n";
}

static void print_summary(const char* label,
                          const std::vector<double>& hv,
                          const std::vector<float>&  dv,
                          size_t bytes) {
    const int n = static_cast<int>(hv.size());
    if (n == 0) return;
    const double mh = std::accumulate(hv.begin(), hv.end(), 0.0) / n;
    const float  md = std::accumulate(dv.begin(),  dv.end(), 0.0f) / n;
    const double bh = *std::min_element(hv.begin(), hv.end());
    const float  bd = *std::min_element(dv.begin(),  dv.end());
    std::cout << "\n  " << label << " (" << n << " runs):\n";
    print_sep();
    std::cout << std::fixed << std::setprecision(3)
              << "  host mean=" << std::setw(8) << mh
              << " ms   best=" << std::setw(8) << bh << " ms"
              << "   → " << std::setprecision(2) << tput_gbs(bytes, bh) << " GB/s (peak)\n"
              << std::setprecision(3)
              << "  dag  mean=" << std::setw(8) << md
              << " ms   best=" << std::setw(8) << bd << " ms"
              << "   → " << std::setprecision(2) << tput_gbs(bytes, static_cast<double>(bd)) << " GB/s (peak)\n"
              << "  host overhead mean=" << std::setw(7) << std::setprecision(3)
              << static_cast<float>(mh) - md << " ms\n";
}

int main(int argc, char* argv[]) {
    int   runs = DEFAULT_RUNS;
    float eb   = DEFAULT_EB;

    if (argc > 1) { runs = std::stoi(argv[1]); if (runs <= 0) { std::cerr << "runs > 0\n"; return 1; } }
    if (argc > 2) { eb   = std::stof(argv[2]); if (eb   <= 0) { std::cerr << "eb > 0\n";   return 1; } }

    const size_t data_bytes = N_ELEMS * sizeof(float);

    print_header("PFPL profiling benchmark — CLDHGH " + std::to_string(DIM_X) + "x" + std::to_string(DIM_Y));
    std::cout << "  Pipeline : Quantizer(NOA,zigzag,r=32768,cap=10%) -> Difference -> Bitshuffle(ew=4) -> RZE(w=1)\n"
              << "  Dataset  : " << DATA_PATH << "\n"
              << "  EB       : " << std::scientific << std::setprecision(1) << eb << " (NOA)\n"
              << "  Runs     : " << runs << " steady-state after cold call\n"
              << "  Pool     : " << std::fixed << std::setprecision(1) << POOL_MULT << "x\n\n";

    const auto ld = load_to_device(data_bytes);
    float* d_input = ld.d_ptr;

    std::cout << "  Loaded " << std::fixed << std::setprecision(2)
              << data_bytes / (1024.0 * 1024.0) << " MB to device\n"
              << "  Data range: " << std::fixed << std::setprecision(4) << ld.data_range
              << "  (NOA value_base for graph capture)\n\n";

    // =========================================================================
    //  Phase A — PREALLOCATE
    // =========================================================================

    Pipeline p(data_bytes, MemoryStrategy::PREALLOCATE, POOL_MULT);
    build_pfpl(p, eb);
    p.enableProfiling(true);
    p.finalize();

    void*  d_comp  = nullptr;
    size_t comp_sz = 0;

#ifdef FZ_PROFILING_ENABLED
    cudaProfilerStart();
#endif

    // ── A-1: compress ─────────────────────────────────────────────────────────
    print_header("Phase A — PREALLOCATE compress");

    std::cout << "  Cold call:\n";
    {
#ifdef FZ_PROFILING_ENABLED
        nvtx3::scoped_range r{"A:compress:cold"};
#endif
        auto t0 = Clock::now();
        p.compress(d_input, data_bytes, &d_comp, &comp_sz, 0);
        cudaDeviceSynchronize();
        auto t1 = Clock::now();
        print_row("cold", elapsed_ms(t0, t1), p.getLastPerfResult().dag_elapsed_ms, data_bytes);
        std::cout << "\n  Stage breakdown (cold):\n";
        p.getLastPerfResult().print(std::cout);
    }

    std::cout << "\n  Compressed: " << std::fixed << std::setprecision(2)
              << comp_sz / (1024.0 * 1024.0) << " MB  (ratio "
              << std::setprecision(2) << static_cast<double>(data_bytes) / comp_sz << "x)\n\n";

    std::cout << "  Steady-state:\n";
    {
        std::vector<double> hv;
        std::vector<float>  dv;
        hv.reserve(runs); dv.reserve(runs);
        for (int i = 0; i < runs; ++i) {
#ifdef FZ_PROFILING_ENABLED
            const std::string rn = "A:compress:" + std::to_string(i + 1);
            nvtx3::scoped_range r{rn.c_str()};
#endif
            auto t0 = Clock::now();
            p.compress(d_input, data_bytes, &d_comp, &comp_sz, 0);
            cudaDeviceSynchronize();
            auto t1 = Clock::now();
            hv.push_back(elapsed_ms(t0, t1));
            dv.push_back(p.getLastPerfResult().dag_elapsed_ms);
            char label[16]; std::snprintf(label, sizeof(label), "%d", i + 1);
            print_row(label, hv.back(), dv.back(), data_bytes);
            if (i == 0) {
                std::cout << "\n  Stage breakdown (run 1):\n";
                p.getLastPerfResult().print(std::cout);
                std::cout << "\n";
            }
        }
        print_summary("compress steady-state", hv, dv, data_bytes);
    }

    // ── A-2: decompress ───────────────────────────────────────────────────────
    print_header("Phase A — PREALLOCATE decompress");

    void*  d_recon    = nullptr;
    size_t d_recon_sz = 0;
    std::cout << "  Cold call (inv-DAG built here):\n";
    {
#ifdef FZ_PROFILING_ENABLED
        nvtx3::scoped_range r{"A:decompress:cold"};
#endif
        auto t0 = Clock::now();
        p.decompress(d_comp, comp_sz, &d_recon, &d_recon_sz, 0);
        cudaDeviceSynchronize();
        auto t1 = Clock::now();
        print_row("cold", elapsed_ms(t0, t1), p.getLastPerfResult().dag_elapsed_ms, d_recon_sz);
        std::cout << "\n  Stage breakdown (cold):\n";
        p.getLastPerfResult().print(std::cout);
    }
    std::cout << "\n";

    std::cout << "  Steady-state:\n";
    {
        std::vector<double> hv;
        std::vector<float>  dv;
        hv.reserve(runs); dv.reserve(runs);
        for (int i = 0; i < runs; ++i) {
#ifdef FZ_PROFILING_ENABLED
            const std::string rn = "A:decompress:" + std::to_string(i + 1);
            nvtx3::scoped_range r{rn.c_str()};
#endif
            void*  d_rec  = nullptr;
            size_t rec_sz = 0;
            auto t0 = Clock::now();
            p.decompress(d_comp, comp_sz, &d_rec, &rec_sz, 0);
            cudaDeviceSynchronize();
            auto t1 = Clock::now();
            hv.push_back(elapsed_ms(t0, t1));
            dv.push_back(p.getLastPerfResult().dag_elapsed_ms);
            char label[16]; std::snprintf(label, sizeof(label), "%d", i + 1);
            print_row(label, hv.back(), dv.back(), rec_sz);
            if (i == 0) {
                std::cout << "\n  Stage breakdown (run 1):\n";
                p.getLastPerfResult().print(std::cout);
                std::cout << "\n";
            }
        }
        print_summary("decompress steady-state", hv, dv, d_recon_sz);
    }

    // =========================================================================
    //  Phase B — CUDA Graph (compress only)
    //  NOA mode: inject data_range as value_base so execute() skips the D2H scan.
    // =========================================================================
    print_header("Phase B — CUDA Graph compress");

    try {
        Pipeline gp(data_bytes, MemoryStrategy::PREALLOCATE, POOL_MULT);
        auto* gquant = build_pfpl(gp, eb);
        gquant->setValueBase(ld.data_range);   // skip NOA scan during capture
        gp.enableGraphMode(true);
        gp.enableProfiling(true);
        gp.finalize();

        cudaStream_t s = nullptr;
        cudaStreamCreate(&s);

        gp.warmup(s);
        gp.captureGraph(s);
        cudaStreamSynchronize(s);
        std::cout << "  Graph captured.\n\n";

        void*  g_comp    = nullptr;
        size_t g_comp_sz = 0;

        gp.compress(d_input, data_bytes, &g_comp, &g_comp_sz, s);
        cudaStreamSynchronize(s);

        const bool size_ok = (g_comp_sz == comp_sz);
        std::cout << "  Graph output size " << g_comp_sz << " B vs PREALLOCATE "
                  << comp_sz << " B — " << (size_ok ? "match" : "MISMATCH") << "\n\n";

        std::cout << "  Graph replay:\n";
        std::vector<double> hv;
        hv.reserve(runs);
        for (int i = 0; i < runs; ++i) {
#ifdef FZ_PROFILING_ENABLED
            const std::string rn = "B:graph:" + std::to_string(i + 1);
            nvtx3::scoped_range r{rn.c_str()};
#endif
            auto t0 = Clock::now();
            gp.compress(d_input, data_bytes, &g_comp, &g_comp_sz, s);
            cudaStreamSynchronize(s);
            auto t1 = Clock::now();
            hv.push_back(elapsed_ms(t0, t1));
            char label[16]; std::snprintf(label, sizeof(label), "%d", i + 1);
            std::cout << std::fixed
                      << "  " << std::setw(4) << std::left << label << std::right
                      << "  host " << std::setw(8) << std::setprecision(3) << hv.back() << " ms"
                      << " (" << std::setw(6) << std::setprecision(2)
                      << tput_gbs(data_bytes, hv.back()) << " GB/s)\n";
        }

        const double best_g = *std::min_element(hv.begin(), hv.end());
        const double mean_g = std::accumulate(hv.begin(), hv.end(), 0.0) / runs;
        std::cout << "\n  Graph compress (" << runs << " runs):\n";
        print_sep();
        std::cout << std::fixed << std::setprecision(3)
                  << "  mean=" << std::setw(8) << mean_g
                  << " ms   best=" << std::setw(8) << best_g << " ms"
                  << "   → " << std::setprecision(2) << tput_gbs(data_bytes, best_g) << " GB/s (peak)\n";

        cudaStreamDestroy(s);

    } catch (const std::exception& e) {
        std::cout << "  Graph capture failed: " << e.what() << "\n";
    }

#ifdef FZ_PROFILING_ENABLED
    cudaProfilerStop();
#endif

    // =========================================================================
    //  Quality (PREALLOCATE round-trip)
    // =========================================================================
    print_header("Quality — PREALLOCATE round-trip");
    if (d_recon && d_recon_sz == data_bytes) {
        auto stats = calculateStatistics<float>(
            d_input, static_cast<const float*>(d_recon), N_ELEMS);
        const double abs_bound = static_cast<double>(eb) * static_cast<double>(ld.data_range);
        const bool within = stats.max_error <= abs_bound * 1.005;   // 0.5% float tolerance for NOA
        std::cout << std::fixed
                  << "  PSNR       : " << std::setprecision(4) << stats.psnr << " dB\n"
                  << "  Max error  : " << std::scientific << std::setprecision(3)
                  << stats.max_error << "  (NOA abs bound=" << abs_bound << ")  "
                  << (within ? "[within bound]" : "[OVER BOUND]") << "\n"
                  << "  NRMSE      : " << std::fixed << std::setprecision(6) << stats.nrmse << "\n"
                  << "  Value range: " << stats.value_range << "\n";
    } else {
        std::cout << "  [WARN] size mismatch — got " << d_recon_sz
                  << " B, expected " << data_bytes << " B\n";
    }

    cudaFree(d_input);
    std::cout << "\nDone.\n";
    return 0;
}
