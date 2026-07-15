/**
 * cuszhi_tp_profile — cuSZ-Hi throughput-mode pipeline benchmark
 *
 * Pipeline (from examples/presets/cusz_hi_tp.toml):
 *   GInterp(NOA, uint8, r=128, cap=20%) ─┬─ Zigzag(int8) → Bitshuffle(ew=1) → RRE(w=1)
 *                                         └─ Merge(anchor|outlier_vals|outlier_idxs)
 *                                              → Bitshuffle(ew=4) → RRE(w=2) → RZE(w=1)
 *
 * This is the library's most complex pipeline: a multi-branch DAG with two
 * parallel lossless chains (codes chain TCMS1→BIT1→RRE1 and outlier chain
 * BITR).  Key profiling targets:
 *   - GInterp kernel cost (spline interpolation, multiscale)
 *   - DAG parallelism: do the two branches overlap on-device?
 *   - MINIMAL vs PREALLOCATE overhead (strategy is MINIMAL in the preset
 *     because outlier buffer size varies per-call)
 *   - Cold decompress (inv-DAG rebuilt on first call)
 *
 * No CUDA Graph capture: GInterp performs a D2H autotune scan during execute().
 *
 * Usage:
 *   ./fzgmod-profile-cuszhi-tp [runs] [eb]
 *   runs : integer > 0       (default: 10)
 *   eb   : positive float    (default: 1e-3, interpreted as NOA)
 *
 * Nsys:
 *   nsys profile --trace=cuda,nvtx --capture-range=cudaProfilerApi \
 *        -o cuszhi_tp ./build_profiling/bin/profiling/fzgmod-profile-cuszhi-tp
 *   nsys-ui cuszhi_tp.nsys-rep
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

static constexpr float DEFAULT_EB   = 1e-3f;
static constexpr int   DEFAULT_RUNS = 10;
static constexpr float POOL_MULT    = 4.0f;

struct LoadResult { float* d_ptr; float data_range; };

static LoadResult load_to_device(size_t data_bytes) {
    std::vector<float> h(N_ELEMS);
    std::FILE* fp = std::fopen(DATA_PATH, "rb");
    if (!fp) { std::cerr << "[ERROR] cannot open: " << DATA_PATH << "\n"; std::exit(1); }
    const size_t got = std::fread(h.data(), sizeof(float), N_ELEMS, fp);
    std::fclose(fp);
    if (got != N_ELEMS) { std::cerr << "[ERROR] expected " << N_ELEMS << " floats, got " << got << "\n"; std::exit(1); }
    auto [it_min, it_max] = std::minmax_element(h.begin(), h.end());
    float* d = nullptr;
    cudaMalloc(&d, data_bytes);
    cudaMemcpy(d, h.data(), data_bytes, cudaMemcpyHostToDevice);
    return {d, *it_max - *it_min};
}

// Build the cuSZ-Hi TP pipeline matching examples/presets/cusz_hi_tp.toml.
// MINIMAL strategy; setDims must be called before GInterpStage is added.
static void build_cuszhi_tp(Pipeline& p, float eb, size_t dim_x, size_t dim_y) {
    p.setDims(dim_x, dim_y, 1);

    auto* gi = p.addStage<GInterpStage<float, uint8_t>>();
    gi->setErrorBound(eb);
    gi->setErrorBoundMode(ErrorBoundMode::NOA);
    gi->setQuantRadius(128);
    gi->setOutlierCapacity(0.2f);
    gi->setAutoTuning(3);   // full structural auto-tune (cuSZ-Hi paper mode)

    // ── codes chain: Zigzag(int8) → Bitshuffle(ew=1) → RRE(w=1) ──
    auto* tcms = p.addStage<ZigzagStage<int8_t>>();
    tcms->setByteTransparent(true);
    p.connect(tcms, gi, "codes");

    auto* bit1 = p.addStage<BitshuffleStage>();
    bit1->setElementWidth(1);
    bit1->setBlockSize(CHUNK);
    p.connect(bit1, tcms);

    auto* rre1 = p.addStage<RREStage>();
    rre1->setWordSize(1);
    p.connect(rre1, bit1);

    // ── outlier chain (BITR): Merge(anchor|outlier_vals|outlier_idxs) → Bitshuffle(ew=4) → RRE(w=2) → RZE(w=1) ──
    auto* merge = p.addStage<MergeStage>();
    merge->setSegmentNames({"anchor", "outlier_vals", "outlier_idxs"});
    p.connect(merge, gi, "anchor");
    p.connect(merge, gi, "outlier_vals");
    p.connect(merge, gi, "outlier_idxs");

    auto* bit4 = p.addStage<BitshuffleStage>();
    bit4->setElementWidth(4);
    bit4->setBlockSize(CHUNK);
    p.connect(bit4, merge);

    auto* rre2 = p.addStage<RREStage>();
    rre2->setWordSize(2);
    p.connect(rre2, bit4);

    auto* rze = p.addStage<RZEStage>();
    rze->setWordSize(1);
    p.connect(rze, rre2);
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

    print_header("cuSZ-Hi TP profiling benchmark — CLDHGH " + std::to_string(DIM_X) + "x" + std::to_string(DIM_Y));
    std::cout << "  Pipeline : GInterp(NOA,uint8,r=128,cap=20%) -> {codes: Zigzag->BIT1->RRE1}\n"
              << "                                               -> {outlier: Merge->BIT4->RRE2->RZE}\n"
              << "  Dataset  : " << DATA_PATH << "\n"
              << "  EB       : " << std::scientific << std::setprecision(1) << eb << " (NOA)\n"
              << "  Strategy : MINIMAL (no graph capture — GInterp D2H autotune)\n"
              << "  Runs     : " << runs << " steady-state after cold call\n"
              << "  Pool     : " << std::fixed << std::setprecision(1) << POOL_MULT << "x\n\n";

    const auto ld = load_to_device(data_bytes);
    float* d_input = ld.d_ptr;

    std::cout << "  Loaded " << std::fixed << std::setprecision(2)
              << data_bytes / (1024.0 * 1024.0) << " MB to device\n\n";

    // =========================================================================
    //  Phase A — MINIMAL compress
    //  MINIMAL allocates buffers on each call; cold call includes JIT + inv-DAG.
    // =========================================================================

    Pipeline p(data_bytes, MemoryStrategy::MINIMAL, POOL_MULT);
    build_cuszhi_tp(p, eb, DIM_X, DIM_Y);
    p.enableProfiling(true);
    p.finalize();

    void*  d_comp  = nullptr;
    size_t comp_sz = 0;

#ifdef FZ_PROFILING_ENABLED
    cudaProfilerStart();
#endif

    print_header("Phase A — MINIMAL compress");

    std::cout << "  Cold call (JIT + autotune here):\n";
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

    // =========================================================================
    //  Phase B — MINIMAL decompress
    // =========================================================================
    print_header("Phase B — MINIMAL decompress");

    void*  d_recon    = nullptr;
    size_t d_recon_sz = 0;
    std::cout << "  Cold call (inv-DAG built here):\n";
    {
#ifdef FZ_PROFILING_ENABLED
        nvtx3::scoped_range r{"B:decompress:cold"};
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
            const std::string rn = "B:decompress:" + std::to_string(i + 1);
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

#ifdef FZ_PROFILING_ENABLED
    cudaProfilerStop();
#endif

    // =========================================================================
    //  Quality (round-trip)
    // =========================================================================
    print_header("Quality — MINIMAL round-trip");
    if (d_recon && d_recon_sz == data_bytes) {
        auto stats = calculateStatistics<float>(
            d_input, static_cast<const float*>(d_recon), N_ELEMS);
        const double abs_bound = static_cast<double>(eb) * static_cast<double>(ld.data_range);
        const bool within = stats.max_error <= abs_bound * 1.0001;
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
