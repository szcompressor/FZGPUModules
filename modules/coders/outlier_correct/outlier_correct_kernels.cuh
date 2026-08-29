#pragma once

/**
 * @file outlier_correct_kernels.cuh
 * @brief Generic per-element kernels shared by every `OutlierCorrectStage<Reconstructor>`
 *        instantiation. None of these reference any particular transform --
 *        transform-specific behavior lives entirely in the `Reconstructor`
 *        policy passed to the stage template (see `outlier_correct_stage.h`).
 *
 * `static` linkage: this header is included by more than one translation
 * unit (once per `Reconstructor` instantiation's .cu file), so these must
 * stay internal-linkage to avoid multiple-definition link errors -- each
 * TU gets its own private copy, matching the existing convention this file
 * was factored out of.
 */

#include <cstdint>

namespace fz {

__global__ static void oc_k_dequant_linear(const int32_t* code, int n, float ebx2, float* coeff) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    coeff[i] = (float)code[i] * ebx2;
}
__global__ static void oc_k_outlier_flag(const float* original, const float* trial, int n, float bound,
                                         int* flag, float* corr_val) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    float err = original[i] - trial[i];
    bool out = fabsf(err) > bound;
    flag[i] = out ? 1 : 0;
    corr_val[i] = out ? err : 0.0f;
}
__global__ static void oc_k_pack(const int* flag, const int* rank, const float* corr_val, int n,
                                 uint32_t* out_idx, float* out_val) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    if (flag[i]) { int r = rank[i]; out_idx[r] = (uint32_t)i; out_val[r] = corr_val[i]; }
}
__global__ static void oc_k_apply(const uint32_t* idx, const float* val, int count, float* field) {
    int j = blockIdx.x * blockDim.x + threadIdx.x; if (j >= count) return;
    field[idx[j]] += val[j];
}

} // namespace fz
