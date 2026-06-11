// Algorithm adapted from cuSZ-Hi (Indiana University, Argonne National Laboratory,
// https://github.com/shixun404/cuSZ-Hi), BSD-3-Clause. See THIRD_PARTY.md.

/**
 * @file ginterp_kernels.cu
 * @brief Host-side launchers for the G-Interp encode/decode kernels and the
 *        small outlier-scatter helper used on the inverse path.
 *
 * The 3D-only MVP wires the LEVEL=4, AnchorBlockSize=16³, numAnchorBlock=1³
 * configuration that matches cuSZ-Hi's deterministic baseline (the
 * `if (l3.z != 1)` branch of upstream `spline3.cu` line 530).
 * `INTERPOLATION_PARAMS` is default-constructed so `auto_tuning=0` semantics
 * apply (alpha=1.75, beta=4.0, level toggles all defaulted). The four
 * profiling-pass kernels in `ginterp_md.inl` are not invoked here — they are
 * wired up in phase 2 with `setAutoTuning(uint8_t > 0)`.
 */

#include "predictors/ginterp/ginterp_kernels.h"
#include "predictors/ginterp/ginterp_md.inl"

#include <cuda_runtime.h>
#include <cstdint>

namespace fz {
namespace ginterp {

// MVP tile / level configuration (3D path, matches cuSZ-Hi baseline)
static constexpr int kLevel               = 4;
static constexpr int kSplineDim           = 3;
static constexpr int kAnchorBlockSizeX    = 16;
static constexpr int kAnchorBlockSizeY    = 16;
static constexpr int kAnchorBlockSizeZ    = 16;
static constexpr int kNumAnchorBlockX     = 1;
static constexpr int kNumAnchorBlockY     = 1;
static constexpr int kNumAnchorBlockZ     = 1;
static constexpr int kLinearBlockSize     = 384;

// Anchor stride per spatial axis (must match the encode kernel's anchor layout).
constexpr int kAnchorStride =
    kAnchorBlockSizeX;  // == AnchorBlockSizeY == AnchorBlockSizeZ in 3D

dim3 ginterpAnchorLen3(size_t nx, size_t ny, size_t nz) {
    auto div_up = [](size_t a, size_t b) -> unsigned int {
        return static_cast<unsigned int>((a + b - 1) / b);
    };
    return dim3(div_up(nx, kAnchorStride),
                div_up(ny, kAnchorStride),
                div_up(nz, kAnchorStride));
}

dim3 stride3FromLen3(dim3 len3) {
    return dim3(1u, len3.x, len3.x * len3.y);
}

// ─── outlier scatter (inverse path) ───────────────────────────────────────────
//
// Writes `outlier_vals[i]` to `outlier_tmp[outlier_idxs[i]]` for i in
// [0, *outlier_count). `outlier_tmp` is pre-zeroed by the caller.
template <typename TInput>
__global__ void scatterOutliersKernel(
    const TInput* __restrict__ outlier_vals,
    const uint32_t* __restrict__ outlier_idxs,
    const uint32_t* __restrict__ outlier_count,
    TInput* __restrict__ outlier_tmp)
{
    uint32_t n = *outlier_count;
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += gridDim.x * blockDim.x)
    {
        outlier_tmp[outlier_idxs[i]] = outlier_vals[i];
    }
}

template <typename TInput>
void launchScatterOutliers(
    const TInput* d_outlier_vals,
    const uint32_t* d_outlier_idxs,
    const uint32_t* d_outlier_count,
    TInput* d_outlier_tmp,
    size_t /*max_outliers*/,
    cudaStream_t stream)
{
    // 256 threads per block; grid sized for ~1M-element worst case at outlier
    // capacity 0.10 → up to 100k outliers, so ~400 blocks. Cap at 1024.
    int grid = 1024;
    int block = 256;
    scatterOutliersKernel<TInput><<<grid, block, 0, stream>>>(
        d_outlier_vals, d_outlier_idxs, d_outlier_count, d_outlier_tmp);
}

// ─── forward (compress) launcher ──────────────────────────────────────────────

template <typename TInput, typename TCode>
void launchGInterpForward3D(
    const TInput* d_data, dim3 data_len3,
    TCode* d_ectrl,
    TInput* d_anchor, dim3 anchor_len3,
    TInput* d_outlier_vals, uint32_t* d_outlier_idxs, uint32_t* d_outlier_count,
    float eb_r, float ebx2, int radius,
    cudaStream_t stream)
{
    auto div_up = [](unsigned int a, unsigned int b) -> unsigned int {
        return (a + b - 1) / b;
    };

    dim3 grid_dim(
        div_up(data_len3.x, kAnchorBlockSizeX * kNumAnchorBlockX),
        div_up(data_len3.y, kAnchorBlockSizeY * kNumAnchorBlockY),
        div_up(data_len3.z, kAnchorBlockSizeZ * kNumAnchorBlockZ));

    dim3 data_st3   = stride3FromLen3(data_len3);
    dim3 ectrl_st3  = data_st3;       // ectrl_len3 == data_len3 in this kernel
    dim3 anchor_st3 = stride3FromLen3(anchor_len3);
    dim3 ectrl_len3 = data_len3;

    INTERPOLATION_PARAMS intp_param;  // default-constructed: cuSZ-Hi baseline

    fz::ginterp::c_spline_infprecis_data<
        TInput*, TCode*, float,
        kLevel, kSplineDim,
        kAnchorBlockSizeX, kAnchorBlockSizeY, kAnchorBlockSizeZ,
        kNumAnchorBlockX, kNumAnchorBlockY, kNumAnchorBlockZ,
        kLinearBlockSize>
        <<<grid_dim, dim3(kLinearBlockSize, 1, 1), 0, stream>>>(
            const_cast<TInput*>(d_data), data_len3, data_st3,
            d_ectrl, ectrl_len3, ectrl_st3,
            d_anchor, anchor_st3,
            d_outlier_vals, d_outlier_idxs, d_outlier_count,
            eb_r, ebx2, radius,
            intp_param);
}

// ─── inverse (decompress) launcher ────────────────────────────────────────────

template <typename TInput, typename TCode>
void launchGInterpInverse3D(
    const TCode* d_ectrl, dim3 data_len3,
    const TInput* d_anchor, dim3 anchor_len3,
    TInput* d_outlier_tmp,        // pre-scattered outlier buffer (full N)
    TInput* d_out,                // reconstructed data
    float eb_r, float ebx2, int radius,
    cudaStream_t stream)
{
    auto div_up = [](unsigned int a, unsigned int b) -> unsigned int {
        return (a + b - 1) / b;
    };

    dim3 grid_dim(
        div_up(data_len3.x, kAnchorBlockSizeX * kNumAnchorBlockX),
        div_up(data_len3.y, kAnchorBlockSizeY * kNumAnchorBlockY),
        div_up(data_len3.z, kAnchorBlockSizeZ * kNumAnchorBlockZ));

    dim3 data_st3   = stride3FromLen3(data_len3);
    dim3 ectrl_len3 = data_len3;
    dim3 ectrl_st3  = data_st3;
    dim3 anchor_st3 = stride3FromLen3(anchor_len3);

    INTERPOLATION_PARAMS intp_param;  // must match the encode-time params

    fz::ginterp::x_spline_infprecis_data<
        TCode*, TInput*, float,
        kLevel, kSplineDim,
        kAnchorBlockSizeX, kAnchorBlockSizeY, kAnchorBlockSizeZ,
        kNumAnchorBlockX, kNumAnchorBlockY, kNumAnchorBlockZ,
        kLinearBlockSize>
        <<<grid_dim, dim3(kLinearBlockSize, 1, 1), 0, stream>>>(
            const_cast<TCode*>(d_ectrl), ectrl_len3, ectrl_st3,
            const_cast<TInput*>(d_anchor), anchor_len3, anchor_st3,
            d_out, data_len3, data_st3,
            d_outlier_tmp,
            eb_r, ebx2, radius,
            intp_param);
}

// ─── Explicit instantiations ──────────────────────────────────────────────────

template void launchGInterpForward3D<float, uint8_t>(
    const float*, dim3, uint8_t*, float*, dim3,
    float*, uint32_t*, uint32_t*, float, float, int, cudaStream_t);
template void launchGInterpForward3D<float, uint16_t>(
    const float*, dim3, uint16_t*, float*, dim3,
    float*, uint32_t*, uint32_t*, float, float, int, cudaStream_t);
template void launchGInterpForward3D<float, uint32_t>(
    const float*, dim3, uint32_t*, float*, dim3,
    float*, uint32_t*, uint32_t*, float, float, int, cudaStream_t);

template void launchGInterpInverse3D<float, uint8_t>(
    const uint8_t*, dim3, const float*, dim3, float*, float*,
    float, float, int, cudaStream_t);
template void launchGInterpInverse3D<float, uint16_t>(
    const uint16_t*, dim3, const float*, dim3, float*, float*,
    float, float, int, cudaStream_t);
template void launchGInterpInverse3D<float, uint32_t>(
    const uint32_t*, dim3, const float*, dim3, float*, float*,
    float, float, int, cudaStream_t);

template void launchScatterOutliers<float>(
    const float*, const uint32_t*, const uint32_t*, float*, size_t, cudaStream_t);

} // namespace ginterp
} // namespace fz
