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
 * Mode-2 profiling (`c_spline_profiling_data_2` — `auto_tuning==2`), mode-4
 * (alpha/beta sweep inside `pa_spline_infprecis_data` with workflow=false),
 * and the 2-D path are all wired here.
 */

#include "fused/ginterp/ginterp_kernels.h"
#include "fused/ginterp/ginterp_md.inl"

#include "backend/api.h"
#include <algorithm>
#include <cstdint>
#include <sstream>
#include <stdexcept>

namespace fz {
namespace ginterp {

// Shared level / block-size configuration.
static constexpr int kLevel               = 4;
static constexpr int kLinearBlockSize     = 384;

// 3-D path: matches cuSZ-Hi baseline (anchor tile 16³, one anchor per grid block).
static constexpr int kSplineDim3          = 3;
static constexpr int kAnchorBlockSizeX3   = 16;
static constexpr int kAnchorBlockSizeY3   = 16;
static constexpr int kAnchorBlockSizeZ3   = 16;
static constexpr int kNumAnchorBlockX3    = 1;
static constexpr int kNumAnchorBlockY3    = 1;
static constexpr int kNumAnchorBlockZ3    = 1;

// 2-D path: structurally mirrors the 3-D layout (anchor tile 16×16, one anchor
// per grid block) with the z axis flattened. The upstream cuSZ-Hi default of
// `AnchorBlockSize={8,8,1}, numAnchorBlock={4,1,1}` would make
// `c_gather_anchor` write 4× too few anchors per block (it emits one anchor
// per grid block regardless of numAnchorBlockX; see the
// `// 2d bug may be here!` comment in ginterp_md.inl c_gather_anchor). Our
// {16,16,1}/{1,1,1} variant sidesteps that — every grid block gathers exactly
// the corner anchor that `x_reset_scratch_data` will look for in the inverse
// path, matching the 3-D contract.
static constexpr int kSplineDim2          = 2;
static constexpr int kAnchorBlockSizeX2   = 16;
static constexpr int kAnchorBlockSizeY2   = 16;
static constexpr int kAnchorBlockSizeZ2   = 1;
static constexpr int kNumAnchorBlockX2    = 1;
static constexpr int kNumAnchorBlockY2    = 1;
static constexpr int kNumAnchorBlockZ2    = 1;

// Anchor stride per spatial axis for the 3-D path.
constexpr int kAnchorStride3 =
    kAnchorBlockSizeX3;  // == AnchorBlockSizeY3 == AnchorBlockSizeZ3 in 3D

dim3 ginterpAnchorLen3(size_t nx, size_t ny, size_t nz) {
    auto div_up = [](size_t a, size_t b) -> unsigned int {
        return static_cast<unsigned int>((a + b - 1) / b);
    };
    return dim3(div_up(nx, kAnchorStride3),
                div_up(ny, kAnchorStride3),
                div_up(nz, kAnchorStride3));
}

dim3 ginterpAnchorLen2(size_t nx, size_t ny) {
    auto div_up = [](size_t a, size_t b) -> unsigned int {
        return static_cast<unsigned int>((a + b - 1) / b);
    };
    // 2-D tile = AnchorBlockSize × numAnchorBlock = 16×16, one anchor per
    // grid block. Anchor extent matches the grid extent (same shape as the
    // 3-D path, just with z=1).
    return dim3(
        div_up(nx, kAnchorBlockSizeX2 * kNumAnchorBlockX2),  // 16
        div_up(ny, kAnchorBlockSizeY2 * kNumAnchorBlockY2),  // 16
        1u);
}

dim3 stride3FromLen3(dim3 len3) {
    return dim3(1u, len3.x, len3.x * len3.y);
}

// ─── dynamic shared-memory sizing ─────────────────────────────────────────────
//
// The main encode/decode kernels carve two T tiles (data + ectrl) out of
// dynamic shared memory. A 16^3 anchor tile + 1-element halo per spatial axis
// is (16+1)^3 = 4913 elements; two of those in double are ~77 KB, well over the
// 48 KB static __shared__ limit, which is why those kernels use dynamic shmem.
template <typename T>
static inline size_t ginterpSplineSmemBytes(
    int spline_dim,
    int abx, int aby, int abz,
    int nbx, int nby, int nbz)
{
    const int tx = abx * nbx + (spline_dim >= 1 ? 1 : 0);
    const int ty = aby * nby + (spline_dim >= 2 ? 1 : 0);
    const int tz = abz * nbz + (spline_dim >= 3 ? 1 : 0);
    // Two tiles: shmem_data + shmem_ectrl.
    return static_cast<size_t>(tx) * ty * tz * sizeof(T) * 2;
}

// Opt into the >48 KB dynamic shared-memory region for the given kernel when
// needed. Below the static cap this is a no-op (the default 48 KB is always
// available). Above it, the device has a hardware ceiling on how far it can
// be raised (`cudaDevAttrMaxSharedMemoryPerBlockOptin`) — on NVIDIA Volta+
// that ceiling scales with the GPU (up to ~227 KB on Hopper), but on AMD
// CDNA (MI100/MI200/MI300, tested: 64 KB) it is fixed at the base LDS size,
// i.e. there is no opt-in region at all. A double-precision 3-D spline tile
// (17^3 elements x 2 tiles x 8 bytes ~= 76.8 KB) fits on the former and
// cannot fit on the latter no matter what — cudaFuncSetAttribute would
// silently fail and the kernel launch itself would later fail with an
// opaque "invalid argument" from cudaGetLastError(). Check against the
// hardware ceiling up front and fail with a clear, actionable message
// instead.
template <typename KernelPtr>
static inline void ginterpRaiseSmemIfNeeded(KernelPtr kernel, size_t smem_bytes)
{
    constexpr size_t kStaticSmemCap = 48u * 1024u;
    if (smem_bytes <= kStaticSmemCap) return;

    int max_optin = 0;
    cudaDeviceGetAttribute(&max_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
    if (max_optin <= 0 || smem_bytes > static_cast<size_t>(max_optin)) {
        std::ostringstream oss;
        oss << "GInterp: this kernel configuration requires " << smem_bytes
            << " bytes of dynamic shared memory per block, which exceeds "
               "this GPU's hardware shared-memory ceiling ("
            << max_optin
            << " bytes, cudaDevAttrMaxSharedMemoryPerBlockOptin). This is a "
               "hardware limit, not a software cap raisable via "
               "cudaFuncSetAttribute — typically hit by double-precision "
               "3-D GInterp (16^3 anchor tile) on GPUs with a fixed 64 KB "
               "LDS/shared-memory size (e.g. AMD CDNA MI100/MI200/MI300). "
               "Reduce the anchor tile size for this precision/dimension, "
               "or use this stage on a GPU with a larger shared-memory "
               "opt-in ceiling.";
        throw std::runtime_error(oss.str());
    }

    cudaFuncSetAttribute(
        reinterpret_cast<const void*>(kernel),
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_bytes));
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
    double eb_r, double ebx2, int radius,
    const INTERPOLATION_PARAMS& intp_param,
    cudaStream_t stream)
{
    auto div_up = [](unsigned int a, unsigned int b) -> unsigned int {
        return (a + b - 1) / b;
    };

    dim3 grid_dim(
        div_up(data_len3.x, kAnchorBlockSizeX3 * kNumAnchorBlockX3),
        div_up(data_len3.y, kAnchorBlockSizeY3 * kNumAnchorBlockY3),
        div_up(data_len3.z, kAnchorBlockSizeZ3 * kNumAnchorBlockZ3));

    dim3 data_st3   = stride3FromLen3(data_len3);
    dim3 ectrl_st3  = data_st3;       // ectrl_len3 == data_len3 in this kernel
    dim3 anchor_st3 = stride3FromLen3(anchor_len3);
    dim3 ectrl_len3 = data_len3;

    auto kernel = &fz::ginterp::c_spline_infprecis_data<
        TInput*, TCode*, TInput,
        kLevel, kSplineDim3,
        kAnchorBlockSizeX3, kAnchorBlockSizeY3, kAnchorBlockSizeZ3,
        kNumAnchorBlockX3, kNumAnchorBlockY3, kNumAnchorBlockZ3,
        kLinearBlockSize>;
    size_t smem = ginterpSplineSmemBytes<TInput>(
        kSplineDim3, kAnchorBlockSizeX3, kAnchorBlockSizeY3, kAnchorBlockSizeZ3,
        kNumAnchorBlockX3, kNumAnchorBlockY3, kNumAnchorBlockZ3);
    ginterpRaiseSmemIfNeeded(kernel, smem);

    kernel<<<grid_dim, dim3(kLinearBlockSize, 1, 1), smem, stream>>>(
            const_cast<TInput*>(d_data), data_len3, data_st3,
            d_ectrl, ectrl_len3, ectrl_st3,
            d_anchor, anchor_st3,
            d_outlier_vals, d_outlier_idxs, d_outlier_count_scratch,
            static_cast<TInput>(eb_r), static_cast<TInput>(ebx2), radius,
            intp_param);
}

// ─── inverse (decompress) launcher ────────────────────────────────────────────

template <typename TInput, typename TCode>
void launchGInterpInverse3D(
    const TCode* d_ectrl, dim3 data_len3,
    const TInput* d_anchor, dim3 anchor_len3,
    TInput* d_outlier_tmp,        // pre-scattered outlier buffer (full N)
    TInput* d_out,                // reconstructed data
    double eb_r, double ebx2, int radius,
    const INTERPOLATION_PARAMS& intp_param,
    cudaStream_t stream)
{
    auto div_up = [](unsigned int a, unsigned int b) -> unsigned int {
        return (a + b - 1) / b;
    };

    dim3 grid_dim(
        div_up(data_len3.x, kAnchorBlockSizeX3 * kNumAnchorBlockX3),
        div_up(data_len3.y, kAnchorBlockSizeY3 * kNumAnchorBlockY3),
        div_up(data_len3.z, kAnchorBlockSizeZ3 * kNumAnchorBlockZ3));

    dim3 data_st3   = stride3FromLen3(data_len3);
    dim3 ectrl_len3 = data_len3;
    dim3 ectrl_st3  = data_st3;
    dim3 anchor_st3 = stride3FromLen3(anchor_len3);

    auto kernel = &fz::ginterp::x_spline_infprecis_data<
        TCode*, TInput*, TInput,
        kLevel, kSplineDim3,
        kAnchorBlockSizeX3, kAnchorBlockSizeY3, kAnchorBlockSizeZ3,
        kNumAnchorBlockX3, kNumAnchorBlockY3, kNumAnchorBlockZ3,
        kLinearBlockSize>;
    size_t smem = ginterpSplineSmemBytes<TInput>(
        kSplineDim3, kAnchorBlockSizeX3, kAnchorBlockSizeY3, kAnchorBlockSizeZ3,
        kNumAnchorBlockX3, kNumAnchorBlockY3, kNumAnchorBlockZ3);
    ginterpRaiseSmemIfNeeded(kernel, smem);

    kernel<<<grid_dim, dim3(kLinearBlockSize, 1, 1), smem, stream>>>(
            const_cast<TCode*>(d_ectrl), ectrl_len3, ectrl_st3,
            const_cast<TInput*>(d_anchor), anchor_len3, anchor_st3,
            d_out, data_len3, data_st3,
            d_outlier_tmp,
            static_cast<TInput>(eb_r), static_cast<TInput>(ebx2), radius,
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
        TInput*, kSplineDim3,
        kProfileBlockSizeX, kProfileBlockSizeY, kProfileBlockSizeZ,
        kProfileNumBlockX, kProfileNumBlockY, kProfileNumBlockZ,
        kLinearBlockSize>
        <<<grid, block, 0, stream>>>(
            const_cast<TInput*>(d_data), data_len3, data_st3,
            d_errors);
}

// ─── profiling mode 2 (alternate cheap probe) ────────────────────────────────
//
// Runs `c_spline_profiling_data_2`. Single-block launch, writes 6 floats:
// forward/reverse × {cubic, natural} on a tiny shared-mem sample. Works for
// both 3-D and 2-D inputs (the kernel is templated on SPLINE_DIM).

template <typename TInput>
void launchGInterpProfileMode2(
    const TInput* d_data, dim3 data_len3,
    int dim,
    float* d_errors,
    cudaStream_t stream)
{
    dim3 grid(1, 1, 1);
    dim3 block(kLinearBlockSize, 1, 1);
    dim3 data_st3 = stride3FromLen3(data_len3);

    if (dim == 3) {
        fz::ginterp::c_spline_profiling_data_2<
            TInput*, kSplineDim3,
            kProfileNumBlockX, kProfileNumBlockY, kProfileNumBlockZ,
            kLinearBlockSize>
            <<<grid, block, 0, stream>>>(
                const_cast<TInput*>(d_data), data_len3, data_st3,
                d_errors);
    } else {
        // 2-D path: PROFILE_NUM_BLOCK_Z = 1 (z axis flattened).
        fz::ginterp::c_spline_profiling_data_2<
            TInput*, kSplineDim2,
            kProfileNumBlockX, kProfileNumBlockY, 1,
            kLinearBlockSize>
            <<<grid, block, 0, stream>>>(
                const_cast<TInput*>(d_data), data_len3, data_st3,
                d_errors);
    }
}

// ─── profiling mode 3 (full structural) / mode 4 (alpha/beta sweep) ──────────
//
// The cuSZ-Hi `pa_spline_infprecis_data` kernel handles both probe families:
//   - `workflow=true`  → mode-3 structural probe: grid.y=9, errors 0..17 (3-D)
//   - `workflow=false` → mode-4 alpha/beta sweep: grid.y=11, errors 0..10
// The kernel's internal `pre_compute_att` reads BIY to dispatch the variant
// for that grid block (see ginterp_md.inl pre_compute_att SPLINE_DIM==3 block).

template <typename TInput>
void launchGInterpProfileMode3(
    const TInput* d_data, dim3 data_len3,
    int dim,
    dim3 sample_starts, dim3 sample_block_grid_sizes, dim3 sample_strides,
    float eb_r, float ebx2,
    const INTERPOLATION_PARAMS& intp_param,
    float* d_errors,
    bool workflow,                // true: mode-3 (structural); false: mode-4 (a/b)
    cudaStream_t stream)
{
    // grid.y: workflow=true mode 3 — 9 for 3-D (cuSZ-Hi spline3.cu line 217),
    // 11 for 2-D (cuSZ-Hi line 294). workflow=false mode 4 — 11 in both
    // (cuSZ-Hi line 409/421). Internal pre_compute_att reads BIY to dispatch
    // each variant; out-of-range BIY values are degenerate (level=5,4 at
    // LEVEL=4 read zero).
    unsigned block_num = sample_block_grid_sizes.x
                        * sample_block_grid_sizes.y
                        * sample_block_grid_sizes.z;
    unsigned grid_y;
    if (workflow) grid_y = (dim == 3) ? 9u : 11u;
    else          grid_y = 11u;
    dim3 grid(block_num, grid_y, 1);
    dim3 block(kLinearBlockSize, 1, 1);
    dim3 data_st3 = stride3FromLen3(data_len3);

    if (dim == 3) {
        // 3-D: LEVEL=4, AnchorBlockSize=16³, numAnchorBlock=1³.
        fz::ginterp::pa_spline_infprecis_data<
            TInput*, float,
            kLevel, kSplineDim3,
            kAnchorBlockSizeX3, kAnchorBlockSizeY3, kAnchorBlockSizeZ3,
            kNumAnchorBlockX3, kNumAnchorBlockY3, kNumAnchorBlockZ3,
            kLinearBlockSize>
            <<<grid, block, 0, stream>>>(
                const_cast<TInput*>(d_data), data_len3, data_st3,
                sample_starts, sample_block_grid_sizes, sample_strides,
                eb_r, ebx2, intp_param,
                d_errors,
                workflow);
    } else {
        // 2-D: LEVEL=4, AnchorBlockSize=16×16×1, numAnchorBlock=1×1×1.
        // Same LEVEL as 3-D so we can reuse the 16×16 anchor tile choice
        // (matches our 2-D forward/inverse path). BIY=0 in pa_spline_infprecis
        // probes kernel-levels {5,4,3} — at LEVEL=4 the first two are
        // degenerate (unit > tile size). Host analysis only consumes
        // errors[6..26].
        fz::ginterp::pa_spline_infprecis_data<
            TInput*, float,
            kLevel, kSplineDim2,
            kAnchorBlockSizeX2, kAnchorBlockSizeY2, kAnchorBlockSizeZ2,
            kNumAnchorBlockX2, kNumAnchorBlockY2, kNumAnchorBlockZ2,
            kLinearBlockSize>
            <<<grid, block, 0, stream>>>(
                const_cast<TInput*>(d_data), data_len3, data_st3,
                sample_starts, sample_block_grid_sizes, sample_strides,
                eb_r, ebx2, intp_param,
                d_errors,
                workflow);
    }
}

// ─── forward (compress) launcher — 2-D ───────────────────────────────────────

template <typename TInput, typename TCode>
void launchGInterpForward2D(
    const TInput* d_data, dim3 data_len3,
    TCode* d_ectrl,
    TInput* d_anchor, dim3 anchor_len3,
    TInput* d_outlier_vals, uint32_t* d_outlier_idxs,
    uint32_t* d_outlier_count_scratch,
    double eb_r, double ebx2, int radius,
    const INTERPOLATION_PARAMS& intp_param,
    cudaStream_t stream)
{
    auto div_up = [](unsigned int a, unsigned int b) -> unsigned int {
        return (a + b - 1) / b;
    };

    // 2-D grid: each block covers AnchorBlockSize*numAnchorBlock = 32×8 input
    // elements; z dimension is exactly 1.
    dim3 grid_dim(
        div_up(data_len3.x, kAnchorBlockSizeX2 * kNumAnchorBlockX2),
        div_up(data_len3.y, kAnchorBlockSizeY2 * kNumAnchorBlockY2),
        1u);

    dim3 data_st3   = stride3FromLen3(data_len3);
    dim3 ectrl_st3  = data_st3;
    dim3 anchor_st3 = stride3FromLen3(anchor_len3);
    dim3 ectrl_len3 = data_len3;

    auto kernel = &fz::ginterp::c_spline_infprecis_data<
        TInput*, TCode*, TInput,
        kLevel, kSplineDim2,
        kAnchorBlockSizeX2, kAnchorBlockSizeY2, kAnchorBlockSizeZ2,
        kNumAnchorBlockX2, kNumAnchorBlockY2, kNumAnchorBlockZ2,
        kLinearBlockSize>;
    size_t smem = ginterpSplineSmemBytes<TInput>(
        kSplineDim2, kAnchorBlockSizeX2, kAnchorBlockSizeY2, kAnchorBlockSizeZ2,
        kNumAnchorBlockX2, kNumAnchorBlockY2, kNumAnchorBlockZ2);
    ginterpRaiseSmemIfNeeded(kernel, smem);

    kernel<<<grid_dim, dim3(kLinearBlockSize, 1, 1), smem, stream>>>(
            const_cast<TInput*>(d_data), data_len3, data_st3,
            d_ectrl, ectrl_len3, ectrl_st3,
            d_anchor, anchor_st3,
            d_outlier_vals, d_outlier_idxs, d_outlier_count_scratch,
            static_cast<TInput>(eb_r), static_cast<TInput>(ebx2), radius,
            intp_param);
}

// ─── inverse (decompress) launcher — 2-D ─────────────────────────────────────

template <typename TInput, typename TCode>
void launchGInterpInverse2D(
    const TCode* d_ectrl, dim3 data_len3,
    const TInput* d_anchor, dim3 anchor_len3,
    TInput* d_outlier_tmp,
    TInput* d_out,
    double eb_r, double ebx2, int radius,
    const INTERPOLATION_PARAMS& intp_param,
    cudaStream_t stream)
{
    auto div_up = [](unsigned int a, unsigned int b) -> unsigned int {
        return (a + b - 1) / b;
    };

    dim3 grid_dim(
        div_up(data_len3.x, kAnchorBlockSizeX2 * kNumAnchorBlockX2),
        div_up(data_len3.y, kAnchorBlockSizeY2 * kNumAnchorBlockY2),
        1u);

    dim3 data_st3   = stride3FromLen3(data_len3);
    dim3 ectrl_len3 = data_len3;
    dim3 ectrl_st3  = data_st3;
    dim3 anchor_st3 = stride3FromLen3(anchor_len3);

    auto kernel = &fz::ginterp::x_spline_infprecis_data<
        TCode*, TInput*, TInput,
        kLevel, kSplineDim2,
        kAnchorBlockSizeX2, kAnchorBlockSizeY2, kAnchorBlockSizeZ2,
        kNumAnchorBlockX2, kNumAnchorBlockY2, kNumAnchorBlockZ2,
        kLinearBlockSize>;
    size_t smem = ginterpSplineSmemBytes<TInput>(
        kSplineDim2, kAnchorBlockSizeX2, kAnchorBlockSizeY2, kAnchorBlockSizeZ2,
        kNumAnchorBlockX2, kNumAnchorBlockY2, kNumAnchorBlockZ2);
    ginterpRaiseSmemIfNeeded(kernel, smem);

    kernel<<<grid_dim, dim3(kLinearBlockSize, 1, 1), smem, stream>>>(
            const_cast<TCode*>(d_ectrl), ectrl_len3, ectrl_st3,
            const_cast<TInput*>(d_anchor), anchor_len3, anchor_st3,
            d_out, data_len3, data_st3,
            d_outlier_tmp,
            static_cast<TInput>(eb_r), static_cast<TInput>(ebx2), radius,
            intp_param);
}

// ─── Explicit instantiations ──────────────────────────────────────────────────

#define FZ_GINTERP_INSTANTIATE_FORWARD(TIN, TCODE)                       \
    template void launchGInterpForward3D<TIN, TCODE>(                     \
        const TIN*, dim3, TCODE*, TIN*, dim3,                            \
        TIN*, uint32_t*, uint32_t*, double, double, int,                 \
        const INTERPOLATION_PARAMS&, cudaStream_t);                      \
    template void launchGInterpForward2D<TIN, TCODE>(                     \
        const TIN*, dim3, TCODE*, TIN*, dim3,                            \
        TIN*, uint32_t*, uint32_t*, double, double, int,                 \
        const INTERPOLATION_PARAMS&, cudaStream_t);

#define FZ_GINTERP_INSTANTIATE_INVERSE(TIN, TCODE)                       \
    template void launchGInterpInverse3D<TIN, TCODE>(                     \
        const TCODE*, dim3, const TIN*, dim3, TIN*, TIN*,                \
        double, double, int, const INTERPOLATION_PARAMS&, cudaStream_t); \
    template void launchGInterpInverse2D<TIN, TCODE>(                     \
        const TCODE*, dim3, const TIN*, dim3, TIN*, TIN*,                \
        double, double, int, const INTERPOLATION_PARAMS&, cudaStream_t);

// Core compress/decompress + outlier scatter — instantiated for float AND
// double. These are the data path.
#define FZ_GINTERP_INSTANTIATE_CORE(TIN)                                 \
    FZ_GINTERP_INSTANTIATE_FORWARD(TIN, uint8_t)                         \
    FZ_GINTERP_INSTANTIATE_FORWARD(TIN, uint16_t)                        \
    FZ_GINTERP_INSTANTIATE_FORWARD(TIN, uint32_t)                        \
    FZ_GINTERP_INSTANTIATE_INVERSE(TIN, uint8_t)                         \
    FZ_GINTERP_INSTANTIATE_INVERSE(TIN, uint16_t)                        \
    FZ_GINTERP_INSTANTIATE_INVERSE(TIN, uint32_t)                        \
    template void launchScatterOutliers<TIN>(                            \
        const TIN*, const uint32_t*, uint32_t, TIN*, cudaStream_t);

// Auto-tuning profiling kernels — float only. The cuSZ-Hi profiling kernels
// write their error metrics into a buffer typed by the data iterator (TITER),
// while the host-side analysis and the scratch buffers are float. Rather than
// thread double through all of that for an off-by-default heuristic, double
// inputs skip auto-tuning and use the deterministic baseline (see
// GInterpStage::execute, the `if constexpr (float)` guard).
#define FZ_GINTERP_INSTANTIATE_PROFILING(TIN)                            \
    template void launchGInterpProfileMode1<TIN>(                        \
        const TIN*, dim3, float*, cudaStream_t);                         \
    template void launchGInterpProfileMode2<TIN>(                        \
        const TIN*, dim3, int, float*, cudaStream_t);                    \
    template void launchGInterpProfileMode3<TIN>(                        \
        const TIN*, dim3, int, dim3, dim3, dim3, float, float,          \
        const INTERPOLATION_PARAMS&, float*, bool, cudaStream_t);

FZ_GINTERP_INSTANTIATE_CORE(float)
FZ_GINTERP_INSTANTIATE_CORE(double)
FZ_GINTERP_INSTANTIATE_PROFILING(float)

#undef FZ_GINTERP_INSTANTIATE_PROFILING
#undef FZ_GINTERP_INSTANTIATE_CORE
#undef FZ_GINTERP_INSTANTIATE_INVERSE
#undef FZ_GINTERP_INSTANTIATE_FORWARD

} // namespace ginterp
} // namespace fz
