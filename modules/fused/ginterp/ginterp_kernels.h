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

// Forward-declared; full definition lives in `cusz_type_subset.h` (included
// only by ginterp_kernels.cu and ginterp_stage.cu, the two consumers).
struct INTERPOLATION_PARAMS;

namespace fz {
namespace ginterp {

/**
 * Compute the anchor grid extent for an input volume of size
 * `(nx, ny, nz)`. The 3D kernel uses a 16³ anchor stride, so the anchor
 * volume is roughly 1/4096 of the input.
 */
dim3 ginterpAnchorLen3(size_t nx, size_t ny, size_t nz);

/**
 * Compute the anchor grid extent for a 2-D input of size `(nx, ny)`. The 2-D
 * tile configuration mirrors the 3-D path with the z axis flattened:
 * `AnchorBlockSize{X,Y,Z}={16,16,1}` × `numAnchorBlock{X,Y,Z}={1,1,1}`. Each
 * grid block covers `16×16` input elements and emits one corner anchor, so
 * the anchor extent is `(ceil(nx/16), ceil(ny/16), 1)` — roughly 1/256 of
 * the input.
 */
dim3 ginterpAnchorLen2(size_t nx, size_t ny);

/**
 * Forward (compress) launcher — predicts via spline interpolation, quantizes
 * residuals into `d_ectrl`, writes anchor corners to `d_anchor`, and routes
 * out-of-range residuals into the outlier pair (`d_outlier_vals`,
 * `d_outlier_idxs`). `d_outlier_count_scratch` is a stage-private 4-byte
 * device pointer the kernel atomically increments — it is **not** a DAG
 * output port. Caller D2H's it during `postStreamSync()` and stores the
 * result in the FZM stage header.
 *
 * Pre-conditions:
 *   - `d_ectrl` is sized `nx * ny * nz * sizeof(TCode)`
 *   - `d_anchor` is sized `prod(ginterpAnchorLen3(nx,ny,nz)) * sizeof(TInput)`
 *   - `d_outlier_count_scratch` has been `cudaMemsetAsync(0, …)` on the same stream
 *   - `eb_r = 1 / (2 * abs_eb)`, `ebx2 = 2 * abs_eb`
 *   - `data_len3.z >= 2` (3D path only in MVP)
 *   - `intp_param` is the resolved cuSZ-Hi interpolation bundle. For phase-1
 *     callers pass a default-constructed struct (deterministic baseline);
 *     phase-2 callers pass the auto-tuned result.
 */
template <typename TInput, typename TCode>
void launchGInterpForward3D(
    const TInput* d_data, dim3 data_len3,
    TCode* d_ectrl,
    TInput* d_anchor, dim3 anchor_len3,
    TInput* d_outlier_vals, uint32_t* d_outlier_idxs,
    uint32_t* d_outlier_count_scratch,
    double eb_r, double ebx2, int radius,
    const INTERPOLATION_PARAMS& intp_param,
    cudaStream_t stream);

/**
 * Inverse (decompress) launcher — reads ectrl + anchor + scattered outliers
 * (pre-merged into `d_outlier_tmp` by `launchScatterOutliers`) and produces
 * the reconstructed volume in `d_out`. `intp_param` MUST match the value used
 * during compression — both encoder and decoder kernels are parameterised by it.
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
    double eb_r, double ebx2, int radius,
    const INTERPOLATION_PARAMS& intp_param,
    cudaStream_t stream);

/**
 * Forward (compress) launcher for 2-D input. Identical contract to the 3-D
 * variant — `data_len3.z` is assumed to be `1`. Internally instantiates the
 * spline kernels with `SPLINE_DIM=2`, `AnchorBlockSize={16,16,1}`,
 * `numAnchorBlock={1,1,1}` (3-D-like tile, z flattened).
 *
 * Pre-conditions:
 *   - `data_len3.z == 1`
 *   - `d_anchor` sized `prod(ginterpAnchorLen2(nx,ny)) * sizeof(TInput)`
 *   - other preconditions identical to the 3-D launcher
 */
template <typename TInput, typename TCode>
void launchGInterpForward2D(
    const TInput* d_data, dim3 data_len3,
    TCode* d_ectrl,
    TInput* d_anchor, dim3 anchor_len3,
    TInput* d_outlier_vals, uint32_t* d_outlier_idxs,
    uint32_t* d_outlier_count_scratch,
    double eb_r, double ebx2, int radius,
    const INTERPOLATION_PARAMS& intp_param,
    cudaStream_t stream);

/**
 * Inverse (decompress) launcher for 2-D input. Mirrors the 3-D variant; the
 * caller must have pre-scattered outliers into `d_outlier_tmp` already.
 * `intp_param` must match the value used during compression.
 */
template <typename TInput, typename TCode>
void launchGInterpInverse2D(
    const TCode* d_ectrl, dim3 data_len3,
    const TInput* d_anchor, dim3 anchor_len3,
    TInput* d_outlier_tmp,
    TInput* d_out,
    double eb_r, double ebx2, int radius,
    const INTERPOLATION_PARAMS& intp_param,
    cudaStream_t stream);

/**
 * Reset the 36-float profiling-errors scratch to zero. One-block, one-thread
 * kernel — used between profiling passes when reusing the same scratch buffer.
 */
void launchGInterpResetErrors(float* d_errors, cudaStream_t stream);

/**
 * Profiling mode 1 — runs the cheap `c_spline_profiling_data` kernel that
 * estimates per-axis residual variance from a tiny shared-mem sample. Writes
 * 2 floats: `errors[0]` (forward order), `errors[1]` (reverse order). Used to
 * pick `intp_param.reverse[0..3]` (single global bool replicated to all levels).
 *
 * Single-block launch — `auto_tuning_grid_dim = dim3(1,1,1)`.
 */
template <typename TInput>
void launchGInterpProfileMode1(
    const TInput* d_data, dim3 data_len3,
    float* d_errors,
    cudaStream_t stream);

/**
 * Profiling mode 2 — runs the alternate cheap `c_spline_profiling_data_2`
 * kernel. Writes 6 floats to `d_errors[0..5]` covering forward/reverse × cubic
 * and natural splines on a tiny shared-mem sample. Used to pick a single
 * `use_natural` × `reverse` pair replicated across all levels (and clears
 * `use_md`). Cheaper than mode 3 and works on both 3-D and 2-D inputs.
 *
 * `dim` is 3 for 3-D inputs and 2 for 2-D inputs (`data_len3.z == 1`).
 * Single-block launch — `auto_tuning_grid_dim = dim3(1,1,1)`.
 */
template <typename TInput>
void launchGInterpProfileMode2(
    const TInput* d_data, dim3 data_len3,
    int dim,
    float* d_errors,
    cudaStream_t stream);

/**
 * Profiling mode 3 — runs the structural `pa_spline_infprecis_data` kernel
 * (cuSZ-Hi `auto_tuning >= 3`) that probes a grid of sample blocks. Caller
 * must `launchGInterpResetErrors` first.
 *
 * `dim` selects the spline-kernel branch (3 → SPLINE_DIM=3, 2 → SPLINE_DIM=2).
 *
 * Outputs depend on `dim`:
 *   - 3-D, LEVEL=4, errors[0..17] (workflow=true):
 *       errors[0..2]   level 3 variants (reverse off, reverse on, use_md)
 *       errors[3..5]   level 2 variants (same triad)
 *       errors[6..11]  level 1 (6 variants: rev×{off,on}, use_md×{0,1}, use_nat×{0,1})
 *       errors[12..17] level 0 (same 6 variants)
 *   - 2-D, LEVEL=4, errors[6..26] (workflow=true; [0..5] are degenerate at
 *     this tile size and ignored by the host analysis):
 *       errors[6..8]    coarsest probed level (kernel level=3) — 3 variants
 *       errors[9..14]   kernel level=2 (intp_param[2]) — 6 variants
 *       errors[15..20]  kernel level=1 (intp_param[1]) — 6 variants
 *       errors[21..26]  kernel level=0 (intp_param[0]) — 6 variants
 *     The level=0 atomic offset is locally patched from `errors+15+BIY` to
 *     `errors+16+BIY` (see adapter-changes block at top of ginterp_md.inl).
 *
 * `sample_starts`, `sample_block_grid_sizes`, `sample_strides` are derived
 * from `data_len3` (see cuSZ-Hi `spline3.cu` `calc_start_size` for the recipe;
 * `S_STRIDE = 8 * 16` in 3-D, `20 * AnchorBlockSize` in 2-D).
 *
 * `workflow` selects the probe family:
 *   - `true`  → structural (mode 3): grid.y=9 (3-D) / 11 (2-D)
 *   - `false` → alpha/beta sweep (mode 4): grid.y=11, errors[0..10] one per
 *     `(alpha, beta)` combo enumerated by pre_compute_att (SPLINE3_AB_ATT).
 */
template <typename TInput>
void launchGInterpProfileMode3(
    const TInput* d_data, dim3 data_len3,
    int dim,
    dim3 sample_starts, dim3 sample_block_grid_sizes, dim3 sample_strides,
    float eb_r, float ebx2,
    const INTERPOLATION_PARAMS& intp_param,
    float* d_errors,
    bool workflow,
    cudaStream_t stream);

/**
 * Scatter outlier-pair entries into a full-N temp buffer. The count `n` is
 * supplied by the host (read from the deserialized FZM header) and passed
 * as a register-resident kernel argument — the kernel never has to load it
 * from device memory.
 *
 * Caller must `cudaMemsetAsync(d_outlier_tmp, 0, N*sizeof(TInput), stream)`
 * before invoking. `n == 0` is a fast no-op.
 */
template <typename TInput>
void launchScatterOutliers(
    const TInput* d_outlier_vals,
    const uint32_t* d_outlier_idxs,
    uint32_t n,
    TInput* d_outlier_tmp,
    cudaStream_t stream);

} // namespace ginterp
} // namespace fz
