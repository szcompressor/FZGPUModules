// Algorithm adapted from cuSZ-Hi (Indiana University, Argonne National Laboratory,
// https://github.com/shixun404/cuSZ-Hi), BSD-3-Clause. See THIRD_PARTY.md.

/**
 * @file ginterp_kernels.cu
 * @brief Host-side launchers for the G-Interp encode/decode kernels, the
 *        cuSZ-Hi profiling kernels used by phase-2 auto-tuning, and the
 *        small outlier-scatter helper used on the inverse path.
 *
 * The 3D-only MVP wires the LEVEL=4, AnchorBlockSize=16³, numAnchorBlock=1³
 * configuration that matches cuSZ-Hi's `if (l3.z != 1)` branch in upstream
 * `spline3.cu` line 530. `INTERPOLATION_PARAMS` is passed by reference from
 * the host wrapper — phase 1 callers pass a default-constructed struct
 * (deterministic baseline: alpha=1.75, beta=4.0); phase-2 callers pass the
 * auto-tuned result of `c_spline_profiling_data` (mode 1) or
 * `pa_spline_infprecis_data` (mode 3) read back via D2H.
 *
 * Mode-2 profiling (`c_spline_profiling_data_2`), mode-4 (alpha/beta sweep
 * inside `pa_spline_infprecis_data` with workflow=false), and the 2-D path
 * are not wired here — they're follow-up scope.
 */

#include "fused/ginterp/ginterp_kernels.h"
#include "fused/ginterp/ginterp_md.inl"

#include <cuda_runtime.h>
#include <algorithm>
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
// Writes `outlier_vals[i]` to `outlier_tmp[outlier_idxs[i]]` for i in [0, n).
// `outlier_tmp` is pre-zeroed by the caller. `n` is passed by value (register
// arg) — no on-device count buffer dereference.
template <typename TInput>
__global__ void scatterOutliersKernel(
    const TInput* __restrict__ outlier_vals,
    const uint32_t* __restrict__ outlier_idxs,
    uint32_t n,
    TInput* __restrict__ outlier_tmp)
{
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
    uint32_t n,
    TInput* d_outlier_tmp,
    cudaStream_t stream)
{
    if (n == 0) return;  // fast no-op: no outliers, scatter has nothing to do
    // 256 threads per block; cap grid at 1024 so very large outlier counts
    // still see good occupancy without exploding launch overhead.
    int block = 256;
    int grid = static_cast<int>(std::min<uint32_t>(
        (n + block - 1) / block, 1024u));
    if (grid < 1) grid = 1;
    scatterOutliersKernel<TInput><<<grid, block, 0, stream>>>(
        d_outlier_vals, d_outlier_idxs, n, d_outlier_tmp);
}

// ─── forward (compress) launcher ──────────────────────────────────────────────

template <typename TInput, typename TCode>
void launchGInterpForward3D(
    const TInput* d_data, dim3 data_len3,
    TCode* d_ectrl,
    TInput* d_anchor, dim3 anchor_len3,
    TInput* d_outlier_vals, uint32_t* d_outlier_idxs,
    uint32_t* d_outlier_count_scratch,
    float eb_r, float ebx2, int radius,
    const INTERPOLATION_PARAMS& intp_param,
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
            d_outlier_vals, d_outlier_idxs, d_outlier_count_scratch,
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
    const INTERPOLATION_PARAMS& intp_param,
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

// ─── reset_errors launcher ────────────────────────────────────────────────────

void launchGInterpResetErrors(float* d_errors, cudaStream_t stream)
{
    // Matches cuSZ-Hi's launch:
    //   reset_errors<<<dim3(1,1,1), dim3(DEFAULT_BLOCK_SIZE,1,1), 0, stream>>>(errors)
    // The kernel zeros the 36-float scratch.
    fz::ginterp::reset_errors<float*>
        <<<dim3(1, 1, 1), dim3(kLinearBlockSize, 1, 1), 0, stream>>>(d_errors);
}

// ─── profiling mode 1 (cheap reverse-only) ───────────────────────────────────

// cuSZ-Hi `c_spline_profiling_data` uses these PROFILE_* constants.
static constexpr int kProfileBlockSizeX = 4;
static constexpr int kProfileBlockSizeY = 4;
static constexpr int kProfileBlockSizeZ = 4;
static constexpr int kProfileNumBlockX  = 4;
static constexpr int kProfileNumBlockY  = 4;
static constexpr int kProfileNumBlockZ  = 4;

template <typename TInput>
void launchGInterpProfileMode1(
    const TInput* d_data, dim3 data_len3,
    float* d_errors,
    cudaStream_t stream)
{
    dim3 grid(1, 1, 1);
    dim3 block(kLinearBlockSize, 1, 1);
    dim3 data_st3 = stride3FromLen3(data_len3);

    fz::ginterp::c_spline_profiling_data<
        TInput*, kSplineDim,
        kProfileBlockSizeX, kProfileBlockSizeY, kProfileBlockSizeZ,
        kProfileNumBlockX, kProfileNumBlockY, kProfileNumBlockZ,
        kLinearBlockSize>
        <<<grid, block, 0, stream>>>(
            const_cast<TInput*>(d_data), data_len3, data_st3,
            d_errors);
}

// ─── profiling mode 3 (full structural) ──────────────────────────────────────

template <typename TInput>
void launchGInterpProfileMode3(
    const TInput* d_data, dim3 data_len3,
    dim3 sample_starts, dim3 sample_block_grid_sizes, dim3 sample_strides,
    float eb_r, float ebx2,
    const INTERPOLATION_PARAMS& intp_param,
    float* d_errors,
    cudaStream_t stream)
{
    // Match cuSZ-Hi 3-D launch (spline3.cu line 217):
    //   grid = (block_num, 9, 1)   -- workflow=true → 9 variant probes per level
    // Each grid.y slot writes a distinct error slot in `d_errors`.
    unsigned block_num = sample_block_grid_sizes.x
                        * sample_block_grid_sizes.y
                        * sample_block_grid_sizes.z;
    dim3 grid(block_num, 9, 1);
    dim3 block(kLinearBlockSize, 1, 1);
    dim3 data_st3 = stride3FromLen3(data_len3);

    // pa_spline_infprecis_data is templated on TITER, FP, LEVEL=4, SPLINE_DIM=3,
    // block sizes 16,16,16, numAnchorBlock 1,1,1, LINEAR_BLOCK_SIZE.
    fz::ginterp::pa_spline_infprecis_data<
        TInput*, float,
        kLevel, kSplineDim,
        kAnchorBlockSizeX, kAnchorBlockSizeY, kAnchorBlockSizeZ,
        kNumAnchorBlockX, kNumAnchorBlockY, kNumAnchorBlockZ,
        kLinearBlockSize>
        <<<grid, block, 0, stream>>>(
            const_cast<TInput*>(d_data), data_len3, data_st3,
            sample_starts, sample_block_grid_sizes, sample_strides,
            eb_r, ebx2, intp_param,
            d_errors,
            /*workflow=*/true);
}

// ─── Explicit instantiations ──────────────────────────────────────────────────

template void launchGInterpForward3D<float, uint8_t>(
    const float*, dim3, uint8_t*, float*, dim3,
    float*, uint32_t*, uint32_t*, float, float, int,
    const INTERPOLATION_PARAMS&, cudaStream_t);
template void launchGInterpForward3D<float, uint16_t>(
    const float*, dim3, uint16_t*, float*, dim3,
    float*, uint32_t*, uint32_t*, float, float, int,
    const INTERPOLATION_PARAMS&, cudaStream_t);
template void launchGInterpForward3D<float, uint32_t>(
    const float*, dim3, uint32_t*, float*, dim3,
    float*, uint32_t*, uint32_t*, float, float, int,
    const INTERPOLATION_PARAMS&, cudaStream_t);

template void launchGInterpInverse3D<float, uint8_t>(
    const uint8_t*, dim3, const float*, dim3, float*, float*,
    float, float, int, const INTERPOLATION_PARAMS&, cudaStream_t);
template void launchGInterpInverse3D<float, uint16_t>(
    const uint16_t*, dim3, const float*, dim3, float*, float*,
    float, float, int, const INTERPOLATION_PARAMS&, cudaStream_t);
template void launchGInterpInverse3D<float, uint32_t>(
    const uint32_t*, dim3, const float*, dim3, float*, float*,
    float, float, int, const INTERPOLATION_PARAMS&, cudaStream_t);

template void launchScatterOutliers<float>(
    const float*, const uint32_t*, uint32_t, float*, cudaStream_t);

template void launchGInterpProfileMode1<float>(
    const float*, dim3, float*, cudaStream_t);

template void launchGInterpProfileMode3<float>(
    const float*, dim3, dim3, dim3, dim3, float, float,
    const INTERPOLATION_PARAMS&, float*, cudaStream_t);

} // namespace ginterp
} // namespace fz
