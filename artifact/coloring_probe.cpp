// A4 probe: does PREALLOCATE-without-coloring measure "worst-case reservation"?
//
// Reports, per preset, the peak device memory a PREALLOCATE pipeline holds with
// liveness coloring on vs. off. Coloring-off = every buffer at its own worst-case
// capacity = exactly what a static-shape task framework must reserve.
//
// Usage: coloring_probe <preset.toml> <nx> [ny] [nz]

#include "fzgpumodules.h"
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

using namespace fz;

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: %s <preset.toml> <nx> [ny] [nz]\n", argv[0]); return 2; }
    const std::string cfg = argv[1];
    size_t nx = std::strtoul(argv[2], nullptr, 10);
    size_t ny = argc > 3 ? std::strtoul(argv[3], nullptr, 10) : 1;
    size_t nz = argc > 4 ? std::strtoul(argv[4], nullptr, 10) : 1;
    const size_t n = nx * ny * nz;
    const size_t nbytes = n * sizeof(float);

    std::vector<float> host(n);
    for (size_t i = 0; i < n; i++) host[i] = 0.5f * std::sin(0.001 * double(i)) + 1.0f;

    float* d_in = nullptr;
    cudaMalloc(&d_in, nbytes);
    cudaMemcpy(d_in, host.data(), nbytes, cudaMemcpyHostToDevice);

    size_t peak[2] = {0, 0};
    size_t regions = 0;
    size_t csize[2] = {0, 0};

    for (int mode = 0; mode < 2; mode++) {           // 0 = colored, 1 = uncolored
        // loadConfig() finalizes the pipeline, so every toggle must be set first.
        // The pool-sizing hint is ctor-only; the TOML supplies the strategy.
        Pipeline p(nbytes, MemoryStrategy::PREALLOCATE, 4.0f);
        p.setDims(nx, ny, nz);
        if (mode == 1) p.setColoringEnabled(false);
        p.loadConfig(cfg);

        void*  d_out = nullptr;
        size_t out_sz = 0;
        p.compress(d_in, nbytes, &d_out, &out_sz, 0);
        cudaStreamSynchronize(0);

        peak[mode]  = p.getPeakMemoryUsage();
        csize[mode] = out_sz;
        if (mode == 0) regions = p.getColorRegionCount();
    }

    cudaFree(d_in);

    const double mb = 1024.0 * 1024.0;
    std::printf("%-28s colored=%8.1f MB  uncolored=%8.1f MB  saved=%6.1f MB (%5.1f%%)  regions=%zu  cmp_equal=%s\n",
                cfg.substr(cfg.find_last_of('/') + 1).c_str(),
                peak[0] / mb, peak[1] / mb,
                (double(peak[1]) - double(peak[0])) / mb,
                peak[1] ? 100.0 * (double(peak[1]) - double(peak[0])) / double(peak[1]) : 0.0,
                regions,
                csize[0] == csize[1] ? "yes" : "NO");
    if (csize[0] != csize[1])
        std::printf("    cmp sizes differ: %zu vs %zu (delta %+lld bytes)\n",
                    csize[0], csize[1], (long long)csize[1] - (long long)csize[0]);
    return 0;
}
