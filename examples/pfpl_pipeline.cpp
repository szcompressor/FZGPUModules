/**
 * PFPL profiling example (minimal).
 *
 * Profiles one phase at a time:
 *   - compress
 *   - decompress
 *
 * Usage:
 *   ./build/bin/pfpl_example [error_bound [mode [phase [runs [threshold]]]]]
 *
 * Examples:
 *   ./build/bin/pfpl_example 1e-3 rel compress 20
 *   ./build/bin/pfpl_example 1e-3 abs decompress 20
 */

#include "fzgpumodules.h"
#ifdef FZ_PROFILING_ENABLED
#include <cuda_profiler_api.h>
#include <nvtx3/nvtx3.hpp>
#endif

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

// CLDHGH: CESM ATM 1800×3600 cloud-top high-level fraction field (≈24.75 MB)
static const char*      CLDHGH_PATH   = "/home/skyler/data/SDRB/CESM_ATM_1800x3600/CLDHGH.f32";
static constexpr size_t CLDHGH_DIM_X  = 3600;
static constexpr size_t CLDHGH_DIM_Y  = 1800;

static constexpr float  DEFAULT_EB = 1e-3f;
static constexpr size_t CHUNK      = 16384;
static constexpr int    DEFAULT_RUNS = 20;

enum class ProfilePhase {
    Compress,
    Decompress
};

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
        << "Usage: pfpl_example [error_bound [mode [phase [runs [threshold]]]]]\n"
        << "  error_bound: 0 < eb < 1 (default: 1e-3)\n"
        << "  mode:        rel | abs | noa (default: rel)\n"
        << "  phase:       compress | decompress (default: compress)\n"
        << "  runs:        integer > 0 (default: 20)\n"
        << "  threshold:   positive float (optional; default: inf)\n";
}

int main(int argc, char* argv[]) {
    float eb = DEFAULT_EB;
    ErrorBoundMode mode = ErrorBoundMode::REL;
    std::string mode_str = "rel";
    ProfilePhase phase = ProfilePhase::Compress;
    std::string phase_str = "compress";
    int runs = DEFAULT_RUNS;
    float threshold = std::numeric_limits<float>::infinity();

    if (argc > 1) {
        eb = std::stof(argv[1]);
        if (eb <= 0.0f || eb >= 1.0f) {
            print_usage();
            return 1;
        }
    }

    if (argc > 2) {
        mode_str = argv[2];
        if (mode_str == "abs") mode = ErrorBoundMode::ABS;
        else if (mode_str == "noa") mode = ErrorBoundMode::NOA;
        else if (mode_str == "rel") mode = ErrorBoundMode::REL;
        else {
            std::cerr << "Unknown mode '" << mode_str << "'.\n";
            print_usage();
            return 1;
        }
    }

    if (argc > 3) {
        phase_str = argv[3];
        if (phase_str == "compress") phase = ProfilePhase::Compress;
        else if (phase_str == "decompress") phase = ProfilePhase::Decompress;
        else {
            std::cerr << "Unknown phase '" << phase_str << "'.\n";
            print_usage();
            return 1;
        }
    }

    if (argc > 4) {
        runs = std::stoi(argv[4]);
        if (runs <= 0) {
            std::cerr << "runs must be > 0\n";
            return 1;
        }
    }

    if (argc > 5) {
        threshold = std::stof(argv[5]);
        if (threshold <= 0.0f) {
            std::cerr << "threshold must be positive\n";
            return 1;
        }
    }

    auto [d_input, n] = load_cldhgh();
    if (!d_input || n == 0) {
        std::cerr << "Dataset not found or unreadable: " << CLDHGH_PATH << "\n";
        return 1;
    }

    const size_t data_bytes = n * sizeof(float);

    std::cout << "=== PFPL Profiling Example (minimal) ===\n"
              << "  Dataset:     CESM ATM CLDHGH (1800x3600)\n"
              << "  Elements:    " << n << "\n"
              << "  Raw size:    " << std::fixed << std::setprecision(2)
              << data_bytes / (1024.0 * 1024.0) << " MB\n"
              << "  Error bound: " << std::scientific << std::setprecision(1)
              << eb << " (" << mode_str << ")\n"
              << "  Phase:       " << phase_str << "\n"
              << "  Runs:        " << runs << "\n"
              << "  Chunk size:  " << CHUNK << " bytes\n";
    if (std::isfinite(threshold)) {
        std::cout << "  Threshold:   " << threshold << "\n";
    }
    std::cout << std::fixed << "\n";

    Pipeline comp(data_bytes, MemoryStrategy::PREALLOCATE, 5.0f);
    build_pfpl_pipeline(comp, eb, mode, threshold);
    comp.enableProfiling(true);

    void*  d_compressed  = nullptr;
    size_t compressed_sz = 0;

#ifdef FZ_PROFILING_ENABLED
    cudaProfilerStart();
#endif

    comp.compress(d_input, data_bytes, &d_compressed, &compressed_sz, 0);
    cudaDeviceSynchronize();

    std::vector<double> host_ms_v;
    std::vector<float> dag_ms_v;
    host_ms_v.reserve(static_cast<size_t>(runs));
    dag_ms_v.reserve(static_cast<size_t>(runs));
    bool printed_first_run_library_report = false;

    for (int i = 0; i < runs; ++i) {
#ifdef FZ_PROFILING_ENABLED
        const std::string range_name = "bench::" + phase_str + "::" + std::to_string(i + 1);
        nvtx3::scoped_range bench_range{range_name.c_str()};
#endif
        const auto t0 = std::chrono::high_resolution_clock::now();

        if (phase == ProfilePhase::Compress) {
            comp.compress(d_input, data_bytes, &d_compressed, &compressed_sz, 0);
            cudaDeviceSynchronize();
        } else {
            void*  d_reconstructed = nullptr;
            size_t reconstructed_sz = 0;
            comp.decompress(d_compressed, compressed_sz, &d_reconstructed, &reconstructed_sz, 0);
            cudaDeviceSynchronize();
            if (d_reconstructed) cudaFree(d_reconstructed);
        }

        const auto t1 = std::chrono::high_resolution_clock::now();

        const double hms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        const float  dms = comp.getLastPerfResult().dag_elapsed_ms;

        if (!printed_first_run_library_report) {
            std::cout << "\n── Library profiling report (run 1, post-warmup) ───────────────\n";
            comp.getLastPerfResult().print(std::cout);
            printed_first_run_library_report = true;
        }

        host_ms_v.push_back(hms);
        dag_ms_v.push_back(dms);

        const float host_gbs = static_cast<float>(data_bytes) / (hms * 1e-3) / 1e9f;
        const float dag_gbs  = static_cast<float>(data_bytes) / (dms * 1e-3f) / 1e9f;

        std::cout << "  run " << std::setw(2) << (i + 1) << ":  "
                  << "host " << std::setw(8) << std::setprecision(3) << hms << " ms  "
                  << std::setw(7) << std::setprecision(2) << host_gbs << " GB/s   "
                  << "dag " << std::setw(8) << std::setprecision(3) << dms << " ms  "
                  << std::setw(7) << std::setprecision(2) << dag_gbs << " GB/s\n";
    }

#ifdef FZ_PROFILING_ENABLED
    cudaProfilerStop();
#endif

    const double mean_h = std::accumulate(host_ms_v.begin(), host_ms_v.end(), 0.0) / runs;
    const float  mean_d = std::accumulate(dag_ms_v.begin(), dag_ms_v.end(), 0.0f) / runs;
    const double min_h  = *std::min_element(host_ms_v.begin(), host_ms_v.end());
    const float  min_d  = *std::min_element(dag_ms_v.begin(), dag_ms_v.end());
    const double max_h  = *std::max_element(host_ms_v.begin(), host_ms_v.end());
    const float  max_d  = *std::max_element(dag_ms_v.begin(), dag_ms_v.end());

    const auto tput = [&](double ms) {
        return static_cast<float>(data_bytes) / (ms * 1e-3) / 1e9f;
    };

    std::cout << "\n── " << phase_str << " summary ─────────────────────────────────────────\n"
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

    cudaFree(d_input);
    std::cout << "\nDone.\n";
    return 0;
}
