// L2-blocked tiling prototype (compress).
//
// Hypothesis: FZGM's modular compress materialises the quantization codes to
// DRAM and then re-reads them twice (AdaptiveBitpack rate + pack). When the
// codes exceed the GPU L2 those re-reads are DRAM traffic; if instead we process
// the field in tiles whose codes fit in L2, the re-reads become L2 hits and the
// effective DRAM traffic drops toward the single-pass floor (read input + write
// output). This harness measures whole-field compress vs sequential tiled
// compress (one reused pipeline, so the intermediate buffers keep the same
// physical addresses and stay L2-resident) across a tile-size sweep.
//
// It is a throughput/locality probe, not a container writer: the tiled path
// compresses each chunk into the pool output and times the aggregate. A chunked
// archive is a legitimate format (it is what cuSZp does internally); assembling
// the final blob is a separate concern and excluded from the timed region.
//
// Usage: fzgmod-profile-l2tile [total_MB] [eb] [runs] [file]
//   total_MB : target field size, CLDHGH replicated up to it (default 256)
//   eb       : abs error bound (default 1e-3)
//   runs     : timed repetitions, best-of reported (default 10)
//   file     : optional real .f32 field to load instead of replicated CLDHGH
//              (first total_MB of it; if smaller, the whole file)
//
// Reports both wall-clock throughput and device-only throughput (summed
// dag_elapsed_ms via CUDA events) so per-tile host launch/sync overhead is
// separated from the GPU/L2 effect the tiling is meant to probe.

#include "fzgpumodules.h"
#include "pipeline/perf.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace fz;
using Clock = std::chrono::high_resolution_clock;

static const char*      DATA_PATH = FZ_DATA_DIR "/CLDHGH.f32";
static constexpr size_t SRC_ELEMS = 3600 * 1800;   // CLDHGH

static double ms_between(std::chrono::time_point<Clock> a,
                         std::chrono::time_point<Clock> b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}
static double gbs(size_t bytes, double ms) { return double(bytes) / (ms * 1e-3) / 1e9; }

// Build the cuSZp2 pipeline (matches examples/presets/cuszp2.toml), 1-D dims.
static void build_cuszp2(Pipeline& p, float eb, size_t n_elems) {
    p.setDims(n_elems, 1, 1);
    auto* quant = p.addStage<QuantizerStage<float, uint32_t>>();
    quant->setErrorBound(eb);
    quant->setErrorBoundMode(ErrorBoundMode::ABS);
    quant->setLinearMode(true);
    auto* lrz = p.addStage<LorenzoStage<int32_t>>();
    lrz->setBlockSize(32);
    p.connect(lrz, quant, "codes");
    auto* ab = p.addStage<AdaptiveBitpackStage<int32_t>>();
    ab->setBlockSize(32);
    ab->setOutlierSelection(true);
    p.connect(ab, lrz);
}

int main(int argc, char** argv) {
    size_t total_mb = (argc > 1) ? std::strtoul(argv[1], nullptr, 10) : 256;
    float  eb       = (argc > 2) ? std::strtof(argv[2], nullptr)      : 1e-3f;
    int    runs     = (argc > 3) ? std::atoi(argv[3])                 : 10;
    const char* file = (argc > 4) ? argv[4] : nullptr;

    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    const size_t l2 = prop.l2CacheSize;
    printf("GPU: %s   SMs=%d   L2=%.1f MB\n", prop.name, prop.multiProcessorCount,
           l2 / (1024.0 * 1024.0));

    // ── Build the field on device (real file, or replicated CLDHGH) ─────────
    size_t total_elems = (total_mb * 1024ull * 1024ull) / sizeof(float);
    std::vector<float> src;
    const char* src_path = DATA_PATH;
    if (file) {
        FILE* f = std::fopen(file, "rb");
        if (!f) { std::fprintf(stderr, "cannot open %s\n", file); return 1; }
        std::fseek(f, 0, SEEK_END); size_t fbytes = std::ftell(f); std::fseek(f, 0, SEEK_SET);
        size_t felems = fbytes / sizeof(float);
        total_elems = std::min(total_elems, felems);       // cap at requested size
        src.resize(total_elems);
        if (std::fread(src.data(), sizeof(float), total_elems, f) != total_elems) {
            std::fprintf(stderr, "short read %s\n", file); return 1;
        }
        std::fclose(f);
        src_path = file;
    } else {
        src.resize(SRC_ELEMS);
        FILE* f = std::fopen(DATA_PATH, "rb");
        if (!f || std::fread(src.data(), sizeof(float), SRC_ELEMS, f) != SRC_ELEMS) {
            std::fprintf(stderr, "cannot read %s\n", DATA_PATH); return 1;
        }
        std::fclose(f);
    }

    const size_t in_bytes = total_elems * sizeof(float);
    float* d_input = nullptr;
    if (cudaMalloc(&d_input, in_bytes) != cudaSuccess) {
        std::fprintf(stderr, "cudaMalloc %zu MB failed\n", in_bytes >> 20); return 1;
    }
    const size_t src_elems = src.size();
    for (size_t off = 0; off < total_elems; off += src_elems) {
        size_t chunk = std::min(src_elems, total_elems - off);
        cudaMemcpy(d_input + off, src.data(), chunk * sizeof(float), cudaMemcpyHostToDevice);
    }
    cudaDeviceSynchronize();
    printf("Source: %s\n", src_path);
    printf("Field: %.1f MB (%zu elems), codes @int32 = %.1f MB  (L2 = %.1f MB)\n",
           in_bytes / (1024.0 * 1024.0), total_elems,
           total_elems * 4.0 / (1024.0 * 1024.0), l2 / (1024.0 * 1024.0));
    printf("eb=%.0e  runs=%d\n\n", eb, runs);

    // ── Baseline: whole-field single compress ───────────────────────────────
    // Returns {best wall ms, summed device ms on that best run}.
    auto time_pipeline = [&](Pipeline& p, size_t tile_elems) -> std::pair<double,double> {
        const size_t n_tiles = (total_elems + tile_elems - 1) / tile_elems;
        void* d_comp = nullptr; size_t comp_sz = 0;
        double best = 1e30, best_dev = 0;
        for (int r = 0; r < runs + 1; ++r) {          // +1 warmup
            cudaDeviceSynchronize();
            double dev = 0;
            auto t0 = Clock::now();
            for (size_t t = 0; t < n_tiles; ++t) {
                size_t off = t * tile_elems;
                size_t e   = std::min(tile_elems, total_elems - off);
                p.compress(d_input + off, e * sizeof(float), &d_comp, &comp_sz, 0);
                dev += p.getLastPerfResult().dag_elapsed_ms;
            }
            cudaDeviceSynchronize();
            auto t1 = Clock::now();
            double wall = ms_between(t0, t1);
            if (r > 0 && wall < best) { best = wall; best_dev = dev; }
        }
        return {best, best_dev};
    };

    Pipeline base(in_bytes, MemoryStrategy::PREALLOCATE, 2.0f);
    build_cuszp2(base, eb, total_elems);
    base.enableProfiling(true);
    base.finalize();
    auto [base_ms, base_dev] = time_pipeline(base, total_elems);
    printf("%-13s %-20s wall %7.3f ms %7.2f GB/s | dev %7.3f ms %7.2f GB/s  [baseline]\n",
           "whole-field", "", base_ms, gbs(in_bytes, base_ms), base_dev, gbs(in_bytes, base_dev));

    // ── Tiled sweep: tile codes at fractions of L2 ──────────────────────────
    printf("\n%-13s %-20s %-24s %-24s %s\n", "mode", "tile", "wall", "device (GPU only)", "dev vs base");
    printf("%s\n", std::string(96, '-').c_str());
    for (double frac : {2.0, 1.0, 0.5, 0.25, 0.125}) {
        size_t code_budget = size_t(l2 * frac);
        size_t tile_elems  = code_budget / 4;                 // int32 codes
        tile_elems = (tile_elems / 32) * 32;                  // block-align
        if (tile_elems == 0 || tile_elems >= total_elems) continue;
        size_t tile_bytes = tile_elems * sizeof(float);

        Pipeline tp(tile_bytes, MemoryStrategy::PREALLOCATE, 2.5f);
        build_cuszp2(tp, eb, tile_elems);
        tp.enableProfiling(true);
        tp.finalize();
        auto [ms, dev] = time_pipeline(tp, tile_elems);
        char lbl[48];
        std::snprintf(lbl, sizeof(lbl), "%.3gxL2 %.1fMB", frac, tile_elems * 4.0 / (1024.0*1024.0));
        printf("%-13s %-20s %7.3f ms %7.2f GB/s   %7.3f ms %7.2f GB/s    %6.2fx\n",
               "tiled", lbl, ms, gbs(in_bytes, ms), dev, gbs(in_bytes, dev), base_dev / dev);
    }

    cudaFree(d_input);
    return 0;
}
