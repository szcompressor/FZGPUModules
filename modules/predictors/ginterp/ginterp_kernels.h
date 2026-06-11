#pragma once

/**
 * @file ginterp_kernels.h
 * @brief Host-callable launchers for the G-Interp encode/decode kernels.
 *
 * This is an internal interface — only `ginterp_stage.cu` should include it.
 * The actual template instantiations live in `ginterp_kernels.cu`, which
 * includes the 3071-line `ginterp_md.inl` privately so callers do not pay the
 * compile-time cost.
 */

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace fz {
namespace ginterp {

/**
 * Compute the anchor grid extent for an input volume of size
 * `(nx, ny, nz)`. The 3D-MVP kernel uses a 16³ anchor stride, so the anchor
 * volume is roughly 1/4096 of the input.
 */
dim3 ginterpAnchorLen3(size_t nx, size_t ny, size_t nz);

/**
 * Forward (compress) launcher — predicts via spline interpolation, quantizes
 * residuals into `d_ectrl`, writes anchor corners to `d_anchor`, and routes
 * out-of-range residuals into the outlier triplet (`d_outlier_vals`,
 * `d_outlier_idxs`, `d_outlier_count` — pre-zeroed by caller).
 *
 * Pre-conditions:
 *   - `d_ectrl` is sized `nx * ny * nz * sizeof(TCode)`
 *   - `d_anchor` is sized `prod(ginterpAnchorLen3(nx,ny,nz)) * sizeof(TInput)`
 *   - `d_outlier_count` has been `cudaMemsetAsync(0, …)` on the same stream
 *   - `eb_r = 1 / (2 * abs_eb)`, `ebx2 = 2 * abs_eb`
 *   - `data_len3.z >= 2` (3D path only in MVP)
 */
template <typename TInput, typename TCode>
void launchGInterpForward3D(
    const TInput* d_data, dim3 data_len3,
    TCode* d_ectrl,
    TInput* d_anchor, dim3 anchor_len3,
    TInput* d_outlier_vals, uint32_t* d_outlier_idxs, uint32_t* d_outlier_count,
    float eb_r, float ebx2, int radius,
    cudaStream_t stream);

/**
 * Inverse (decompress) launcher — reads ectrl + anchor + scattered outliers
 * (pre-merged into `d_outlier_tmp` by `launchScatterOutliers`) and produces
 * the reconstructed volume in `d_out`.
 *
 * `d_outlier_tmp` must be a full-N buffer with outlier values written at
 * outlier indices and zero elsewhere — the kernel reads it via
 * `global2shmem_fuse` during shmem load.
 */
template <typename TInput, typename TCode>
void launchGInterpInverse3D(
    const TCode* d_ectrl, dim3 data_len3,
    const TInput* d_anchor, dim3 anchor_len3,
    TInput* d_outlier_tmp,
    TInput* d_out,
    float eb_r, float ebx2, int radius,
    cudaStream_t stream);

/**
 * Scatter outlier-triplet entries into a full-N temp buffer. Reads
 * `*d_outlier_count` on the device to determine how many to write — never
 * triggers a D2H, so safe inside CUDA graph capture.
 *
 * Caller must `cudaMemsetAsync(d_outlier_tmp, 0, N*sizeof(TInput), stream)`
 * before invoking.
 */
template <typename TInput>
void launchScatterOutliers(
    const TInput* d_outlier_vals,
    const uint32_t* d_outlier_idxs,
    const uint32_t* d_outlier_count,
    TInput* d_outlier_tmp,
    size_t max_outliers,
    cudaStream_t stream);

} // namespace ginterp
} // namespace fz
