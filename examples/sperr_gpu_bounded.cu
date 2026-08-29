/**
 * @file sperr_gpu_bounded.cpp
 * @brief Prototype: a TRUE pointwise error-bound guarantee on top of the GPU
 *        SPERR pipeline (CDF97 -> Quantizer -> SPECK2D), via sparse outlier
 *        correction -- the same mechanism native SPERR's own Outlier_Coder
 *        uses, and the fix for the gap found in memory/speck_gpu_design.md
 *        sec.9 / compression_benchmarking's RUN_LEDGER.md sec.3.5.
 *
 * Why NOT per-level/subband quantization-step scaling (the other candidate
 * fix): measured on real CLDHGH data (scratchpad/level_quant_check.cpp) that
 * scaling each coefficient's OWN quantization step by its level's synthesis-
 * filter gain (computed via scratchpad/gain_calib.cpp, a real, correct
 * per-level L-inf impulse-response measurement) makes max reconstruction
 * error WORSE, not better -- because many coefficients across levels jointly
 * influence any given pixel, so bounding each one's OWN worst-case
 * contribution in isolation does not bound their SUM. Outlier correction
 * sidesteps this entirely: it operates on the ACTUAL final reconstruction
 * error, whatever the accumulated cause, and is essentially free (measured:
 * 0.0002%-0.02% of pixels need correction on CLDHGH, costing under 11 KB
 * against a multi-MB archive).
 *
 * Mechanism:
 *   COMPRESS: (1) normal pipeline: DWT -> Quantizer(linear/ABS) -> SPECK2D.
 *     (2) in parallel, dequantize the SAME codes and inverse-DWT a COPY to
 *     get a "trial reconstruction" -- SPECK2D is lossless w.r.t. the codes
 *     (proven in speck2d's own tests), so this trial reconstruction is
 *     EXACTLY what decompress will produce before correction; no need to
 *     actually run SPECK2D encode/decode to know it.
 *     (3) compare trial reconstruction to the ORIGINAL field; every pixel
 *     whose error exceeds the bound gets an EXACT correction value
 *     (original - trial) recorded in a sparse (index, value) list.
 *   DECOMPRESS: normal pipeline inverse, then scatter-apply the correction
 *     list. Result is guaranteed within bound for every corrected pixel
 *     (exact) and was already within bound for every other pixel (by
 *     construction of the outlier test) -- so the WHOLE field is guaranteed.
 *
 * STATUS: validated standalone (this program), calling the Stage classes
 * directly rather than through fz::Pipeline/CompressionDAG -- proves the
 * mechanism and gives real numbers before committing to a DAG-integrated
 * Stage (which needs a "Tee" of the pipeline's raw input to two consumers;
 * see memory/speck_gpu_design.md sec.9 addendum for the follow-up plan).
 */

#include "fzgpumodules.h"
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <fstream>
#include <string>
#include <cmath>

using namespace fz;

static void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) { fprintf(stderr, "CUDA %s: %s\n", w, cudaGetErrorString(e)); exit(1); }
}

// Outlier flag (widened to int for CUB) + exact correction value (original - trial).
__global__ void k_outlier_flag(const float* original, const float* trial, int n, float bound,
                               int* flag, float* corr_val) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    float err = original[i] - trial[i];
    bool out = fabsf(err) > bound;
    flag[i] = out ? 1 : 0;
    corr_val[i] = out ? err : 0.0f;
}
__global__ void k_pack_corrections(const int* flag, const int* rank, const float* corr_val,
                                   int n, int* out_idx, float* out_val) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    if (flag[i]) { int r = rank[i]; out_idx[r] = i; out_val[r] = corr_val[i]; }
}
__global__ void k_apply_corrections(const int* idx, const float* val, int count, float* field) {
    int j = blockIdx.x * blockDim.x + threadIdx.x; if (j >= count) return;
    field[idx[j]] += val[j];
}

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "data/CLDHGH.f32";
    const int nx = argc > 2 ? atoi(argv[2]) : 3600;
    const int ny = argc > 3 ? atoi(argv[3]) : 1800;
    const size_t n = (size_t)nx * ny;

    std::vector<float> field(n);
    { std::ifstream f(path, std::ios::binary);
      if (!f) { fprintf(stderr, "cannot open %s\n", path); return 1; }
      f.read((char*)field.data(), n * 4); }

    cudaStream_t s; ck(cudaStreamCreate(&s), "stream");
    auto g = [&](int cnt) { return dim3((unsigned)((cnt + 255) / 256)); };

    float *d_field, *d_coeff, *d_trial;
    ck(cudaMalloc(&d_field, n * 4), "m"); ck(cudaMalloc(&d_coeff, n * 4), "m"); ck(cudaMalloc(&d_trial, n * 4), "m");
    ck(cudaMemcpyAsync(d_field, field.data(), n * 4, cudaMemcpyHostToDevice, s), "h2d");
    int32_t* d_codes; ck(cudaMalloc(&d_codes, n * 4), "m");

    int *d_flag, *d_rank; float* d_corrval;
    ck(cudaMalloc(&d_flag, n * 4), "m"); ck(cudaMalloc(&d_rank, n * 4), "m"); ck(cudaMalloc(&d_corrval, n * 4), "m");
    void* d_tmp = nullptr; size_t tmpb = 0;
    cub::DeviceScan::ExclusiveSum(d_tmp, tmpb, d_flag, d_rank, (int)n); ck(cudaMalloc(&d_tmp, tmpb), "m");

    int fails = 0;
    printf("%-10s %12s %12s  %10s %8s %12s %12s  %s\n",
           "bound", "main_bytes", "corr_bytes", "#outlier", "%out", "err_before", "err_after", "guaranteed");
    printf("%s\n", std::string(96, '-').c_str());

    for (float bound : {1e-2f, 1e-3f, 1e-4f, 1e-5f}) {
        // ---- forward: DWT -> Quantizer(linear/ABS) ----
        ck(cudaMemcpyAsync(d_coeff, d_field, n * 4, cudaMemcpyDeviceToDevice, s), "d2d");
        Cdf97Stage<float> dwt; dwt.setDims(nx, ny, 1);
        dwt.execute(s, nullptr, {d_coeff}, {d_coeff}, {n * 4});

        QuantizerStage<float, uint32_t> quant;
        quant.setDims({(size_t)nx, (size_t)ny, 1});
        quant.setErrorBound(bound);
        quant.setErrorBoundMode(ErrorBoundMode::ABS);
        quant.setLinearMode(true);
        quant.execute(s, nullptr, {d_coeff}, {d_codes}, {n * 4});

        // ---- main archive: SPECK2D ----
        Speck2DStage speck; speck.setDims(nx, ny);
        auto est = speck.estimateOutputSizes({n * 4});
        uint8_t* d_arc; ck(cudaMalloc(&d_arc, est[0]), "m");
        speck.execute(s, nullptr, {d_codes}, {d_arc}, {n * 4});
        ck(cudaStreamSynchronize(s), "sync");
        speck.postStreamSync(s);
        size_t arc_bytes = speck.getActualOutputSize(0);

        // ---- correction pass: dequantize the SAME codes via the REAL
        //      QuantizerStage inverse (not a hand-rolled formula -- must be
        //      bit-for-bit what decode will actually do, or the computed
        //      correction is for the wrong baseline), inverse-DWT a copy,
        //      compare to original. SPECK2D is lossless w.r.t. codes (proven
        //      elsewhere), so this trial equals what decompress produces
        //      before correction -- no need to actually run SPECK2D decode
        //      here to know it. ----
        // NOTE: an inverse-direction QuantizerStage must get computed_abs_eb_
        // from the FORWARD object's serialized header (deserializeHeader),
        // not from setErrorBound() on a fresh object -- setErrorBound() only
        // updates config_.error_bound; the inverse path reads computed_abs_eb_
        // directly, which forward's execute() computes and only
        // serializeHeader()/deserializeHeader() round-trips. (Same bug class
        // as the Speck2DStage cached_n_ fix earlier this session -- forgetting
        // this the first time produced 100% "outliers" here, not the
        // documented tiny fraction: the trial reconstruction was dequantizing
        // with the default 1e-4 bound regardless of what `bound` actually was.)
        uint8_t qhdr[FZM_STAGE_CONFIG_SIZE] = {};
        size_t qhlen = quant.serializeHeader(0, qhdr, sizeof(qhdr));
        QuantizerStage<float, uint32_t> quant_trial;
        quant_trial.deserializeHeader(qhdr, qhlen);
        quant_trial.setDims({(size_t)nx, (size_t)ny, 1});
        quant_trial.setInverse(true);
        quant_trial.execute(s, nullptr, {d_codes}, {d_trial}, {n * 4});
        Cdf97Stage<float> dwt_inv; dwt_inv.setDims(nx, ny, 1); dwt_inv.setInverse(true);
        dwt_inv.execute(s, nullptr, {d_trial}, {d_trial}, {n * 4});

        k_outlier_flag<<<g((int)n), 256, 0, s>>>(d_field, d_trial, (int)n, bound, d_flag, d_corrval);
        cub::DeviceScan::ExclusiveSum(d_tmp, tmpb, d_flag, d_rank, (int)n, s);
        int last_flag, last_rank;
        ck(cudaMemcpyAsync(&last_flag, d_flag + n - 1, 4, cudaMemcpyDeviceToHost, s), "d");
        ck(cudaMemcpyAsync(&last_rank, d_rank + n - 1, 4, cudaMemcpyDeviceToHost, s), "d");
        ck(cudaStreamSynchronize(s), "sync");
        int num_out = last_rank + last_flag;

        int* d_out_idx; float* d_out_val;
        ck(cudaMalloc(&d_out_idx, std::max(1, num_out) * 4), "m");
        ck(cudaMalloc(&d_out_val, std::max(1, num_out) * 4), "m");
        k_pack_corrections<<<g((int)n), 256, 0, s>>>(d_flag, d_rank, d_corrval, (int)n, d_out_idx, d_out_val);

        // measure pre-correction max error (== what decompress would give without the fix)
        std::vector<float> trial_h(n), field_h(n);
        ck(cudaMemcpyAsync(trial_h.data(), d_trial, n * 4, cudaMemcpyDeviceToHost, s), "d");
        ck(cudaStreamSynchronize(s), "sync");
        double err_before = 0; for (size_t i = 0; i < n; ++i) err_before = std::max(err_before, (double)std::fabs(field[i] - trial_h[i]));

        // ---- DECODE + apply correction (this is what a real decompress does) ----
        int32_t* d_codes2; ck(cudaMalloc(&d_codes2, n * 4), "m");
        Speck2DStage speck_inv;
        uint8_t hdr[FZM_STAGE_CONFIG_SIZE] = {};
        size_t hlen = speck.serializeHeader(0, hdr, sizeof(hdr));
        speck_inv.deserializeHeader(hdr, hlen);
        speck_inv.setInverse(true);
        speck_inv.execute(s, nullptr, {d_arc}, {d_codes2}, {arc_bytes});

        float* d_coeff2; ck(cudaMalloc(&d_coeff2, n * 4), "m");
        QuantizerStage<float, uint32_t> quant_inv;
        quant_inv.deserializeHeader(qhdr, qhlen);   // see quant_trial's comment above
        quant_inv.setDims({(size_t)nx, (size_t)ny, 1});
        quant_inv.setInverse(true);
        quant_inv.execute(s, nullptr, {d_codes2}, {d_coeff2}, {n * 4});

        Cdf97Stage<float> dwt_inv2; dwt_inv2.setDims(nx, ny, 1); dwt_inv2.setInverse(true);
        dwt_inv2.execute(s, nullptr, {d_coeff2}, {d_coeff2}, {n * 4});

        k_apply_corrections<<<g(std::max(1, num_out)), 256, 0, s>>>(d_out_idx, d_out_val, num_out, d_coeff2);
        ck(cudaStreamSynchronize(s), "sync");

        std::vector<float> recon(n);
        ck(cudaMemcpy(recon.data(), d_coeff2, n * 4, cudaMemcpyDeviceToHost), "d");
        double err_after = 0; for (size_t i = 0; i < n; ++i) err_after = std::max(err_after, (double)std::fabs(field[i] - recon[i]));
        bool guaranteed = err_after <= (double)bound + 1e-9;

        size_t corr_bytes = 4 + (size_t)num_out * 8;   // count:u32 + (idx:u32,val:f32) per outlier
        printf("%-10.3g %12zu %12zu  %10d %7.4f%% %12.6f %12.6f  %s\n",
               bound, arc_bytes, corr_bytes, num_out, 100.0 * num_out / n, err_before, err_after,
               guaranteed ? "YES" : "** NO **");
        if (!guaranteed) fails++;

        cudaFree(d_out_idx); cudaFree(d_out_val); cudaFree(d_codes2); cudaFree(d_coeff2); cudaFree(d_arc);
    }

    printf(fails ? "\nFAIL (%d bound(s) not guaranteed)\n" : "\nALL BOUNDS GUARANTEED\n", fails);
    return fails ? 1 : 0;
}
