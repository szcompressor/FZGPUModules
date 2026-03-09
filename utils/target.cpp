/**
 * CLDHGH Profiling Target
 *
 * A minimal, focused binary for profiling the Lorenzo + Diff pipeline
 * on the CESM ATM CLDHGH dataset (1800x3600 float32).
 * All compression and decompression is done in-memory (no file I/O).
 *
 * Usage:
 *   nsys profile --trace=cuda,nvtx --capture-range=cudaProfilerApi -o cldhgh_profile ./build/fzgmod-profile
 *   nsys-ui cldhgh_profile.nsys-rep
 */

#include "fzgpumodules.h"
#include "pipeline/stat.h"
#ifdef FZ_PROFILING_ENABLED
#include <cuda_profiler_api.h>
#include <nvtx3/nvtx3.hpp>
#endif

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

using namespace fz;

static constexpr const char* DATA_PATH =
    "/home/skyler/data/SDRB/CESM_ATM_1800x3600/CLDHGH.f32";

static constexpr size_t DIM_X = 3600;
static constexpr size_t DIM_Y = 1800;
static constexpr size_t N     = DIM_X * DIM_Y;
static constexpr float  EB    = 1e-3f;

// ─────────────────────────────────────────────────────────────
//  Load data from disk → device
// ─────────────────────────────────────────────────────────────

static float* load_to_device() {
    std::vector<float> h(N);

    std::FILE* fp = std::fopen(DATA_PATH, "rb");
    if (!fp) {
        std::cerr << "[profile] ERROR: cannot open " << DATA_PATH << "\n";
        std::exit(1);
    }
    size_t got = std::fread(h.data(), sizeof(float), N, fp);
    std::fclose(fp);

    if (got != N) {
        std::cerr << "[profile] ERROR: expected " << N
                  << " floats, got " << got << "\n";
        std::exit(1);
    }

    float* d = nullptr;
    cudaMalloc(&d, N * sizeof(float));
    cudaMemcpy(d, h.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    return d;
}

// ─────────────────────────────────────────────────────────────
//  Helpers
// ─────────────────────────────────────────────────────────────

// Configure a fresh compression pipeline in-place.
static void setup_pipeline(Pipeline& p) {
    auto* lorenzo = p.addStage<LorenzoStage<float, uint16_t>>();
    lorenzo->setErrorBound(EB);
    lorenzo->setQuantRadius(512);
    lorenzo->setOutlierCapacity(0.15f);

    auto* diff = p.addStage<DifferenceStage<uint16_t>>();
    p.connect(diff, lorenzo, "codes");

    p.finalize();
}

// Print reconstruction quality metrics.
static void print_stats(const ReconstructionStats& s, size_t n, float eb) {
    const int W = 14;
    std::cout << std::fixed;
    std::cout << "  ┌─ Reconstruction quality ──────────────────────\n";
    std::cout << "  │  PSNR           " << std::setw(W) << std::setprecision(4)
              << s.psnr    << "  dB\n";
    std::cout << "  │  MSE            " << std::setw(W) << std::setprecision(6)
              << s.mse     << "\n";
    std::cout << "  │  NRMSE          " << std::setw(W) << std::setprecision(6)
              << s.nrmse   << "\n";
    std::cout << "  │  Max error      " << std::setw(W) << std::setprecision(6)
              << s.max_error
              << "  (bound=" << eb << ", ratio="
              << std::setprecision(3) << s.max_error / eb << "x)\n";
    std::cout << "  │  Value range    " << std::setw(W) << std::setprecision(6)
              << s.value_range << "\n";
    std::cout << "  │  Elements       " << std::setw(W) << n << "\n";
    std::cout << "  └───────────────────────────────────────────────\n";
}

// ─────────────────────────────────────────────────────────────
//  main
// ─────────────────────────────────────────────────────────────

int main() {
    const size_t data_bytes = N * sizeof(float);

    std::cout << "[profile] Loading CLDHGH.f32 (" << DIM_X << "x" << DIM_Y
              << ", " << data_bytes / 1024.0 / 1024.0 << " MB)\n";

    float* d_input = load_to_device();

    // ── Warm-up pass (excluded from profiler timeline — cudaProfilerStart not called yet)
    {
#ifdef FZ_PROFILING_ENABLED
        nvtx3::scoped_range warmup{"warmup"};
#endif
        std::cout << "[profile] Warm-up pass...\n";

        Pipeline p(data_bytes, MemoryStrategy::PREALLOCATE, 3.0f);
        setup_pipeline(p);

        void*  d_comp  = nullptr;
        size_t comp_sz = 0;
        {
#ifdef FZ_PROFILING_ENABLED
            nvtx3::scoped_range r{"warmup::compress"};
#endif
            p.compress(d_input, data_bytes, &d_comp, &comp_sz, 0);
            cudaDeviceSynchronize();
        }

        void*  d_decomp  = nullptr;
        size_t decomp_sz = 0;
        {
#ifdef FZ_PROFILING_ENABLED
            nvtx3::scoped_range r{"warmup::decompress"};
#endif
            p.decompress(nullptr, 0, &d_decomp, &decomp_sz, 0);
            cudaDeviceSynchronize();
        }

        if (d_decomp) cudaFree(d_decomp);
    }

    // ── Profiled pass  (nsys captures only this region)
#ifdef FZ_PROFILING_ENABLED
    cudaProfilerStart();
#endif
    {
        std::cout << "[profile] Profiled pass...\n";

        Pipeline comp(data_bytes, MemoryStrategy::PREALLOCATE, 3.0f);
        comp.enableProfiling(true);
        setup_pipeline(comp);

        // ── Compress ────────────────────────────────────────────
        void*  d_compressed  = nullptr;
        size_t compressed_sz = 0;
        {
#ifdef FZ_PROFILING_ENABLED
            nvtx3::scoped_range r{"compress"};
#endif
            comp.compress(d_input, data_bytes, &d_compressed, &compressed_sz, 0);
            cudaDeviceSynchronize();
        }

        std::cout << "\n[profile] ── Compress ──────────────────────────────\n";
        std::cout << "  Original:   " << data_bytes / 1024.0 / 1024.0 << " MB\n";
        std::cout << "  Compressed: " << compressed_sz / 1024.0 / 1024.0
                  << " MB  (ratio " << static_cast<double>(data_bytes) / compressed_sz << "x)\n";
        comp.getLastPerfResult().print(std::cout);

        // ── Decompress (in-memory) ───────────────────────────────
        void*  d_decompressed  = nullptr;
        size_t decompressed_sz = 0;
        {
#ifdef FZ_PROFILING_ENABLED
            nvtx3::scoped_range r{"decompress"};
#endif
            // Pass the compressed pointer so the inverse path uses our buffer
            // rather than reading directly from the forward DAG's memory.
            comp.decompress(d_compressed, compressed_sz, &d_decompressed, &decompressed_sz, 0);
            cudaDeviceSynchronize();
        }

        std::cout << "\n[profile] ── Decompress ────────────────────────────\n";
        std::cout << "  Reconstructed: " << decompressed_sz / 1024.0 / 1024.0 << " MB\n";
        comp.getLastPerfResult().print(std::cout);

        // ── Reconstruction quality ───────────────────────────────
        std::cout << "\n[profile] ── Quality ───────────────────────────────\n";
        if (d_decompressed && decompressed_sz == data_bytes) {
            auto stats = calculateStatistics<float>(
                d_input,
                static_cast<const float*>(d_decompressed),
                N
            );
            print_stats(stats, N, EB);
        } else {
            std::cout << "  [WARN] Size mismatch — skipping stats "
                      << "(got " << decompressed_sz << " vs " << data_bytes << ")\n";
        }

        if (d_decompressed) cudaFree(d_decompressed);
    }
#ifdef FZ_PROFILING_ENABLED
    cudaProfilerStop();
#endif

    cudaFree(d_input);
    std::cout << "\n[profile] Done.\n";
    return 0;
}
