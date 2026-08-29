/**
 * @file sperr_gpu_dag_bounded.cu
 * @brief DAG-integrated version of the bound-guarantee mechanism: builds the
 *        real `fz::Pipeline` topology (Cdf97Stage -> QuantizerStage ->
 *        Cdf97OutlierCorrectStage -> Speck2DStage) and calls
 *        `Pipeline::compress()`/`decompress()`, instead of calling Stage
 *        classes directly (`examples/sperr_gpu_bounded.cu`, the earlier
 *        validated prototype). See `Cdf97OutlierCorrectStage`'s header
 *        (`modules/coders/outlier_correct/outlier_correct_stage.h`) for the
 *        port shape and `Pipeline::bindExternalInput()`'s doc comment for
 *        why no fan-out stage is needed: both `Cdf97Stage` and
 *        `Cdf97OutlierCorrectStage` bind the pipeline's raw input directly.
 */

#include "fzgpumodules.h"
#include <cstdio>
#include <cstdint>
#include <vector>
#include <fstream>
#include <cmath>
#include <algorithm>

using namespace fz;

static void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) { fprintf(stderr, "CUDA %s: %s\n", w, cudaGetErrorString(e)); exit(1); }
}

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "data/CLDHGH.f32";
    const int nx = argc > 2 ? atoi(argv[2]) : 3600;
    const int ny = argc > 3 ? atoi(argv[3]) : 1800;
    const size_t n = (size_t)nx * ny;
    const size_t bytes = n * sizeof(float);

    std::vector<float> field(n);
    { std::ifstream f(path, std::ios::binary);
      if (!f) { fprintf(stderr, "cannot open %s\n", path); return 1; }
      f.read((char*)field.data(), bytes); }

    void* d_input = nullptr;
    ck(cudaMalloc(&d_input, bytes), "m");
    ck(cudaMemcpy(d_input, field.data(), bytes, cudaMemcpyHostToDevice), "h2d");

    int fails = 0;
    printf("%-10s %14s %10s   %s\n", "bound", "compressed", "err_after", "guaranteed");
    printf("%s\n", std::string(56, '-').c_str());

    for (float bound : {1e-2f, 1e-3f, 1e-4f, 1e-5f}) {
        Pipeline p(bytes, MemoryStrategy::PREALLOCATE, /*pool_mult=*/8.0f);
        p.setDims(nx, ny, 1);

        auto* dwt = p.addStage<Cdf97Stage<float>>();   // pure source (no other inputs):
                                                        // auto-discovered, no bindExternalInput() needed

        auto* quant = p.addStage<QuantizerStage<float, uint32_t>>();
        quant->setErrorBound(bound);
        quant->setErrorBoundMode(ErrorBoundMode::ABS);
        quant->setLinearMode(true);

        auto* corr = p.addStage<Cdf97OutlierCorrectStage>();
        corr->setErrorBound(bound);   // MUST match quant's bound
        p.bindExternalInput(corr);    // corr.input[0] = raw field, bound BEFORE the
                                      // connect() below so it lands on port 0.

        auto* speck = p.addStage<Speck2DStage>();

        p.connect(quant, dwt);
        p.connect(corr, quant, "codes");     // corr.input[1] = codes
        p.connect(speck, corr, "codes");     // Speck2D consumes corr's codes passthrough

        p.setPrimarySource(corr);            // decompress() returns corr's corrected field
        p.finalize();

        void* d_compressed = nullptr; size_t compressed_size = 0;
        p.compress(d_input, bytes, &d_compressed, &compressed_size, /*stream=*/0);
        ck(cudaDeviceSynchronize(), "sync");

        void* d_recon = nullptr; size_t recon_size = 0;
        p.decompress(d_compressed, compressed_size, &d_recon, &recon_size, /*stream=*/0);
        ck(cudaDeviceSynchronize(), "sync");

        std::vector<float> recon(n);
        ck(cudaMemcpy(recon.data(), d_recon, std::min(recon_size, bytes), cudaMemcpyDeviceToHost), "d2h");

        double err_after = 0;
        for (size_t i = 0; i < n; ++i) err_after = std::max(err_after, (double)std::fabs(field[i] - recon[i]));
        bool guaranteed = (recon_size == bytes) && (err_after <= (double)bound + 1e-9);

        printf("%-10.3g %14zu %10.6f   %s\n", bound, compressed_size, err_after, guaranteed ? "YES" : "** NO **");
        if (!guaranteed) fails++;
    }

    printf(fails ? "\nFAIL (%d bound(s) not guaranteed)\n" : "\nALL BOUNDS GUARANTEED (via real fz::Pipeline DAG)\n", fails);
    return fails ? 1 : 0;
}
