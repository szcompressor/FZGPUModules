/**
 * Repeated Compression Benchmark
 *
 * Demonstrates the throughput benefit of reusing a single preallocated Pipeline
 * versus constructing a fresh pipeline for every compress() call.
 *
 * Two experimental arms are timed over N_RUNS iterations each:
 *
 *   ARM A — PREALLOCATE, reused pipeline
 *     A single Pipeline with MemoryStrategy::PREALLOCATE is built once. All GPU
 *     buffers are allocated upfront during finalize(). Subsequent compress() calls
 *     skip all allocation; only kernels and stream-sync run.
 *
 *   ARM B — MINIMAL, fresh pipeline per call
 *     A brand-new Pipeline is constructed and finalize()'d for every iteration.
 *     Buffers are allocated on-demand and freed as soon as each stage consumer is
 *     done. Both host-side setup and cudaMalloc/cudaFree overhead are paid every
 *     call.
 *
 * Usage (plain):
 *   ./build/bin/fzgmod-profile-repeat
 *
 * Usage (nsys profiling to see the two arms as distinct NVTX regions):
 *   nsys profile --trace=cuda,nvtx --capture-range=cudaProfilerApi \
 *        -o repeat_profile ./build/bin/fzgmod-profile-repeat
 *   nsys-ui repeat_profile.nsys-rep
 */

#include "fzgpumodules.h"
#include "pipeline/perf.h"
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
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

using namespace fz;
using Clock = std::chrono::high_resolution_clock;

// ─────────────────────────────────────────────────────────────
//  Dataset constants
// ─────────────────────────────────────────────────────────────

static constexpr const char* DATA_PATH =
    "/home/skyler/data/SDRB/CESM_ATM_1800x3600/CLDHGH.f32";

static constexpr size_t DIM_X = 3600;
static constexpr size_t DIM_Y = 1800;
static constexpr size_t N     = DIM_X * DIM_Y;
static constexpr float  EB    = 1e-3f;

// Number of compress() calls timed per arm (excludes the mandatory warm-up).
static constexpr int N_RUNS = 20;

// ─────────────────────────────────────────────────────────────
//  Load data from disk → device
// ─────────────────────────────────────────────────────────────

static float* load_to_device() {
    std::vector<float> h(N);

    std::FILE* fp = std::fopen(DATA_PATH, "rb");
    if (!fp) {
        std::cerr << "[repeat] ERROR: cannot open " << DATA_PATH << "\n";
        std::exit(1);
    }
    const size_t got = std::fread(h.data(), sizeof(float), N, fp);
    std::fclose(fp);

    if (got != N) {
        std::cerr << "[repeat] ERROR: expected " << N << " floats, got " << got << "\n";
        std::exit(1);
    }

    float* d = nullptr;
    cudaMalloc(&d, N * sizeof(float));
    cudaMemcpy(d, h.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    return d;
}

// ─────────────────────────────────────────────────────────────
//  Pipeline factory
// ─────────────────────────────────────────────────────────────

static const size_t DATA_BYTES = N * sizeof(float);

static void setup_pipeline(Pipeline& p) {
    auto* quantizer = p.addStage<QuantizerStage<float, uint16_t>>();
    quantizer->setErrorBound(EB);
    quantizer->setErrorBoundMode(ErrorBoundMode::ABS);
    quantizer->setQuantRadius(32768);
    quantizer->setOutlierCapacity(0.1f);

    auto* diff = p.addStage<DifferenceStage<uint16_t>>();
    p.connect(diff, quantizer, "codes");

    p.finalize();
}

// ─────────────────────────────────────────────────────────────
//  Stat helpers
// ─────────────────────────────────────────────────────────────

struct RunStats {
    std::vector<double> host_ms; // wall-clock time per compress() call (ms)
    std::vector<float>  dag_ms;  // GPU-only dag->execute() time (ms)

    void reserve(int n) { host_ms.reserve(n); dag_ms.reserve(n); }

    void push(double h, float d) { host_ms.push_back(h); dag_ms.push_back(d); }

    double mean_host() const {
        return std::accumulate(host_ms.begin(), host_ms.end(), 0.0) / host_ms.size();
    }
    float mean_dag() const {
        return std::accumulate(dag_ms.begin(), dag_ms.end(), 0.0f) / dag_ms.size();
    }
    double min_host() const { return *std::min_element(host_ms.begin(), host_ms.end()); }
    float  min_dag()  const { return *std::min_element(dag_ms.begin(), dag_ms.end()); }
    double max_host() const { return *std::max_element(host_ms.begin(), host_ms.end()); }
    float  max_dag()  const { return *std::max_element(dag_ms.begin(), dag_ms.end()); }

    float throughput_gbs(float elapsed_ms) const {
        return static_cast<float>(DATA_BYTES) / (elapsed_ms * 1e-3f) / 1e9f;
    }

    void print(const char* label) const {
        const int W = 10;
        std::cout << std::fixed;
        std::cout << "  ┌─ " << label << " (" << host_ms.size() << " runs)\n";
        std::cout << "  │  Host wall time  mean=" << std::setw(W) << std::setprecision(3)
                  << mean_host() << " ms   "
                  << "min=" << std::setw(W) << min_host() << " ms   "
                  << "max=" << std::setw(W) << max_host() << " ms\n";
        std::cout << "  │  DAG only        mean=" << std::setw(W) << std::setprecision(3)
                  << mean_dag() << " ms   "
                  << "min=" << std::setw(W) << min_dag() << " ms   "
                  << "max=" << std::setw(W) << max_dag() << " ms\n";
        std::cout << "  │  Throughput (host)  " << std::setw(W) << std::setprecision(2)
                  << throughput_gbs(static_cast<float>(mean_host())) << " GB/s  (mean)\n";
        std::cout << "  │  Throughput (DAG)   " << std::setw(W) << std::setprecision(2)
                  << throughput_gbs(mean_dag()) << " GB/s  (mean)\n";
        std::cout << "  └──────────────────────────────────────────────────────\n";
    }
};

// ─────────────────────────────────────────────────────────────
//  main
// ─────────────────────────────────────────────────────────────

int main() {
    std::cout << "[repeat] Loading CLDHGH.f32 (" << DIM_X << "x" << DIM_Y
              << ", " << DATA_BYTES / 1024.0 / 1024.0 << " MB)\n";

    float* d_input = load_to_device();

    // ── Warm-up: exercise the CUDA driver / caches once before measuring.
    // Build a PREALLOCATE pipeline, compress once, discard result.
    {
        std::cout << "[repeat] Warm-up pass...\n";
        Pipeline warmup(DATA_BYTES, MemoryStrategy::PREALLOCATE, 3.0f);
        setup_pipeline(warmup);
        void* d_out = nullptr;
        size_t out_sz = 0;
        warmup.compress(d_input, DATA_BYTES, &d_out, &out_sz, 0);
        cudaDeviceSynchronize();
        // d_out points into the DAG's internal pool — do NOT cudaFree it.
    }

#ifdef FZ_PROFILING_ENABLED
    cudaProfilerStart();
#endif

    // ══════════════════════════════════════════════════════════
    //  ARM A: PREALLOCATE — single pipeline reused N_RUNS times
    // ══════════════════════════════════════════════════════════
    {
#ifdef FZ_PROFILING_ENABLED
        nvtx3::scoped_range arm_a{"arm_a::preallocate_reused"};
#endif
        std::cout << "\n[repeat] ARM A — PREALLOCATE, reused pipeline (" << N_RUNS << " runs)\n";

        Pipeline p(DATA_BYTES, MemoryStrategy::PREALLOCATE, 3.0f);
        setup_pipeline(p);
        p.enableProfiling(true);

        RunStats stats;
        stats.reserve(N_RUNS);

        for (int i = 0; i < N_RUNS; ++i) {
            void*  d_out  = nullptr;
            size_t out_sz = 0;

#ifdef FZ_PROFILING_ENABLED
            nvtx3::scoped_range iter{std::string("arm_a::iter_" + std::to_string(i)).c_str()};
#endif
            auto t0 = Clock::now();
            p.compress(d_input, DATA_BYTES, &d_out, &out_sz, 0);
            cudaDeviceSynchronize();
            auto t1 = Clock::now();

            const double host_ms =
                std::chrono::duration<double, std::milli>(t1 - t0).count();
            const float dag_ms = p.getLastPerfResult().dag_elapsed_ms;

            stats.push(host_ms, dag_ms);
            // d_out is an internal DAG buffer — do NOT cudaFree it.
        }

        stats.print("PREALLOCATE reused");
    }

    // ══════════════════════════════════════════════════════════
    //  ARM B: MINIMAL — fresh pipeline constructed every call
    // ══════════════════════════════════════════════════════════
    {
#ifdef FZ_PROFILING_ENABLED
        nvtx3::scoped_range arm_b{"arm_b::minimal_fresh"};
#endif
        std::cout << "\n[repeat] ARM B — MINIMAL, fresh pipeline per call (" << N_RUNS << " runs)\n";

        RunStats stats;
        stats.reserve(N_RUNS);

        for (int i = 0; i < N_RUNS; ++i) {
            void*  d_out  = nullptr;
            size_t out_sz = 0;

#ifdef FZ_PROFILING_ENABLED
            nvtx3::scoped_range iter{std::string("arm_b::iter_" + std::to_string(i)).c_str()};
#endif
            auto t0 = Clock::now();

            // Pipeline construction, finalize, and compress all measured together.
            Pipeline p(DATA_BYTES, MemoryStrategy::MINIMAL, 3.0f);
            setup_pipeline(p);
            p.enableProfiling(true);
            p.compress(d_input, DATA_BYTES, &d_out, &out_sz, 0);
            cudaDeviceSynchronize();

            auto t1 = Clock::now();

            const double host_ms =
                std::chrono::duration<double, std::milli>(t1 - t0).count();
            const float dag_ms = p.getLastPerfResult().dag_elapsed_ms;

            stats.push(host_ms, dag_ms);
            // d_out is an internal DAG buffer — owned by p, freed on scope exit.
        }

        stats.print("MINIMAL fresh-per-call");
    }

    // ══════════════════════════════════════════════════════════
    //  ARM C: PREALLOCATE — first call only (cold allocation cost)
    // ══════════════════════════════════════════════════════════
    //
    // Isolates the first-call overhead of PREALLOCATE (cudaMalloc for all
    // buffers up-front) so it can be compared against ARM A's steady-state.
    {
#ifdef FZ_PROFILING_ENABLED
        nvtx3::scoped_range arm_c{"arm_c::preallocate_cold"};
#endif
        std::cout << "\n[repeat] ARM C — PREALLOCATE first-call overhead (" << N_RUNS << " cold pipelines)\n";

        RunStats stats;
        stats.reserve(N_RUNS);

        for (int i = 0; i < N_RUNS; ++i) {
            void*  d_out  = nullptr;
            size_t out_sz = 0;

#ifdef FZ_PROFILING_ENABLED
            nvtx3::scoped_range iter{std::string("arm_c::iter_" + std::to_string(i)).c_str()};
#endif
            auto t0 = Clock::now();

            Pipeline p(DATA_BYTES, MemoryStrategy::PREALLOCATE, 3.0f);
            setup_pipeline(p);
            p.enableProfiling(true);
            p.compress(d_input, DATA_BYTES, &d_out, &out_sz, 0);
            cudaDeviceSynchronize();

            auto t1 = Clock::now();

            const double host_ms =
                std::chrono::duration<double, std::milli>(t1 - t0).count();
            const float dag_ms = p.getLastPerfResult().dag_elapsed_ms;

            stats.push(host_ms, dag_ms);
            // d_out is an internal DAG buffer — owned by p, freed on scope exit.
        }

        stats.print("PREALLOCATE cold (first call)");
    }

#ifdef FZ_PROFILING_ENABLED
    cudaProfilerStop();
#endif

    cudaFree(d_input);
    std::cout << "\n[repeat] Done.\n";
    return 0;
}
