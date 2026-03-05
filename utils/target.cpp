/**
 * CLDHGH Profiling Target
 *
 * A minimal, focused binary for profiling the Lorenzo + Diff pipeline
 * on the CESM ATM CLDHGH dataset (1800x3600 float32).
 *
 * Usage:
 *   nsys profile --trace=cuda,nvtx --capture-range=cudaProfilerApi -o cldhgh_profile ./build/fzmod-profile
 *   nsys-ui cldhgh_profile.nsys-rep
 */

#include "fzmodules.h"
#include <cuda_profiler_api.h>
#include <nvtx3/nvtx3.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
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
//  main
// ─────────────────────────────────────────────────────────────

// Helper: configure a fresh compression pipeline in-place.
static void setup_pipeline(Pipeline& p) {
    auto* lorenzo = p.addStage<LorenzoStage<float, uint16_t>>();
    lorenzo->setErrorBound(EB);
    lorenzo->setQuantRadius(512);
    lorenzo->setOutlierCapacity(0.15f);

    auto* diff = p.addStage<DifferenceStage<uint16_t>>();
    p.connect(diff, lorenzo, "codes");

    p.finalize();
}

int main() {
    const size_t data_bytes = N * sizeof(float);

    std::cout << "[profile] Loading CLDHGH.f32 (" << DIM_X << "x" << DIM_Y
              << ", " << data_bytes / 1024.0 / 1024.0 << " MB)\n";

    float* d_input = load_to_device();

    // ── Warm-up pass (excluded from profiler timeline — cudaProfilerStart not called yet)
    {
        nvtx3::scoped_range warmup{"warmup"};
        std::cout << "[profile] Warm-up pass...\n";

        Pipeline p(data_bytes, MemoryStrategy::PREALLOCATE, 3.0f);
        setup_pipeline(p);

        void*  d_comp  = nullptr;
        size_t comp_sz = 0;
        {
            nvtx3::scoped_range r{"warmup::compress"};
            p.compress(d_input, data_bytes, &d_comp, &comp_sz, 0);
            cudaDeviceSynchronize();
        }

        static const std::string warmup_file = "cldhgh_warmup.fzm";
        p.writeToFile(warmup_file, 0);

        void*  d_decomp  = nullptr;
        size_t decomp_sz = 0;
        {
            nvtx3::scoped_range r{"warmup::decompress"};
            Pipeline::decompressFromFile(warmup_file, &d_decomp, &decomp_sz, 0, nullptr);
            cudaDeviceSynchronize();
        }

        if (d_decomp) cudaFree(d_decomp);
        std::remove(warmup_file.c_str());
    }

    // ── Profiled pass  (nsys captures only this region)
    cudaProfilerStart();
    {
        std::cout << "[profile] Profiled pass...\n";

        static const std::string out_file = "cldhgh_profile.fzm";

        // Compress
        void*  d_compressed  = nullptr;
        size_t compressed_sz = 0;
        Pipeline comp(data_bytes, MemoryStrategy::PREALLOCATE, 3.0f);
        comp.enableProfiling(true);
        setup_pipeline(comp);
        {
            nvtx3::scoped_range r{"compress"};
            comp.compress(d_input, data_bytes, &d_compressed, &compressed_sz, 0);
            cudaDeviceSynchronize();
        }

        std::cout << "\n[profile] ── Compress ──────────────────────────────\n";
        std::cout << "  Original:   " << data_bytes / 1024.0 / 1024.0 << " MB\n";
        std::cout << "  Compressed: " << compressed_sz / 1024.0 / 1024.0
                  << " MB  (ratio " << static_cast<double>(data_bytes) / compressed_sz << "x)\n";
        comp.getLastPerfResult().print(std::cout);

        comp.writeToFile(out_file, 0);

        // Decompress
        void*            d_decompressed = nullptr;
        size_t           decompressed_sz = 0;
        PipelinePerfResult decomp_perf;
        {
            nvtx3::scoped_range r{"decompress"};
            Pipeline::decompressFromFile(
                out_file, &d_decompressed, &decompressed_sz, 0, &decomp_perf);
            cudaDeviceSynchronize();
        }

        std::cout << "\n[profile] ── Decompress ────────────────────────────\n";
        decomp_perf.print(std::cout);

        if (d_decompressed) cudaFree(d_decompressed);
        std::remove(out_file.c_str());
    }
    cudaProfilerStop();

    cudaFree(d_input);
    std::cout << "\n[profile] Done.\n";
    return 0;
}
