// Algorithm adapted from cuSZ-Hi (Indiana University, Argonne National Laboratory,
// https://github.com/shixun404/cuSZ-Hi), BSD-3-Clause. See THIRD_PARTY.md.

#include "fused/ginterp/ginterp_stage.h"
#include "fused/ginterp/ginterp_kernels.h"
#include "fused/ginterp/cusz_type_subset.h"  // INTERPOLATION_PARAMS full def
#include "predictors/predictor_utils.cuh"
#include "mem/mempool.h"
#include "cuda_check.h"
#include "log.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace fz {

// ─── buildIntpParam ──────────────────────────────────────────────────────────
// Pack the resolved POD fields (alpha, beta, three uint8[6] arrays) into a
// cuSZ-Hi `INTERPOLATION_PARAMS` for kernel consumption. The bool arrays in
// the upstream struct are layout-equivalent to our uint8 arrays since C++
// `bool` is a one-byte type on every supported platform; we copy element-wise
// rather than memcpy to be portable.
template <typename TInput, typename TCode>
INTERPOLATION_PARAMS GInterpStage<TInput, TCode>::buildIntpParam() const
{
    INTERPOLATION_PARAMS p;  // default-constructed = baseline; we overwrite
    p.alpha = resolved_alpha_;
    p.beta  = resolved_beta_;
    for (int i = 0; i < 6; ++i) {
        p.use_md[i]      = (resolved_use_md_[i]      != 0);
        p.use_natural[i] = (resolved_use_natural_[i] != 0);
        p.reverse[i]     = (resolved_reverse_[i]     != 0);
    }
    p.auto_tuning = 0;  // Kernel itself never re-tunes; host has already.
    return p;
}

// ─── pickAutoRadius ───────────────────────────────────────────────────────────
// Choose a quantizer radius based on observed data range. The constraint is
// that residuals at the coarsest interpolation level (where `cur_ebx2 = ebx2/beta`
// with default beta=4) must satisfy `|err| < radius * cur_ebx2` to be
// quantizable; anything beyond goes to the outlier triplet.
//
// We size `radius = ceil(data_range / cur_ebx2_min)` so that even a residual
// equal to the full data range stays quantizable, then clamp to the TCode
// bit-width's maximum (codes are stored as TCode). The minimum of 32 keeps
// constant or near-constant data from degenerating to radius=0.
template <typename TCode>
static int pickAutoRadius(float data_range, float ebx2)
{
    constexpr int kMaxForCode =
        (sizeof(TCode) == 1) ?    127 :   // uint8: codes in [0, 255]
        (sizeof(TCode) == 2) ?  32767 :   // uint16: codes in [0, 65535]
                                32767;    // uint32: cap at uint16-max — past
                                          // this, float precision in the
                                          // outlier-as-float path dominates.
    constexpr int kMinRadius = 32;
    if (data_range <= 0.0f || ebx2 <= 0.0f) return kMinRadius;
    const float cur_ebx2_min = ebx2 / 4.0f;  // beta=4 default
    const double needed =
        std::ceil(static_cast<double>(data_range) / cur_ebx2_min);
    if (needed >= static_cast<double>(kMaxForCode)) return kMaxForCode;
    return std::max(kMinRadius, static_cast<int>(needed));
}

// ─── destructor ──────────────────────────────────────────────────────────────
template <typename TInput, typename TCode>
GInterpStage<TInput, TCode>::~GInterpStage()
{
    if (persistent_pool_ != nullptr) {
        if (d_profiling_errors_)      persistent_pool_->freePersistentDevice(d_profiling_errors_);
        if (h_profiling_errors_)      persistent_pool_->freePersistentPinned(h_profiling_errors_);
        if (d_outlier_count_scratch_) persistent_pool_->freePersistentDevice(d_outlier_count_scratch_);
    }
    d_profiling_errors_      = nullptr;
    h_profiling_errors_      = nullptr;
    d_outlier_count_scratch_ = nullptr;
    persistent_pool_         = nullptr;
}

// ─── initOutlierCountScratch ─────────────────────────────────────────────────
// Allocate the 4-byte outlier-count device scratch. Idempotent. Called from
// onFinalize() in PREALLOCATE mode and from the first compress execute() in
// MINIMAL mode. The counter lives for the stage's lifetime; freed in dtor.
template <typename TInput, typename TCode>
void GInterpStage<TInput, TCode>::initOutlierCountScratch(MemoryPool* pool)
{
    if (d_outlier_count_scratch_ != nullptr) return;
    if (pool == nullptr) {
        throw std::runtime_error(
            "GInterpStage: outlier-count scratch requires a MemoryPool");
    }
    persistent_pool_ = pool;  // safe to overwrite — same pool either way
    d_outlier_count_scratch_ = static_cast<uint32_t*>(
        pool->allocatePersistentDevice(sizeof(uint32_t), "ginterp_outlier_count"));
}

// ─── initProfilingScratch ────────────────────────────────────────────────────
// Allocate the 36-float profiling-errors scratch (device + pinned host) via
// the pool's persistent allocators. Idempotent: subsequent calls are no-ops.
// Called from both onFinalize() (PREALLOCATE path) and the first execute()
// (MINIMAL path) so the stage works in either memory strategy.
template <typename TInput, typename TCode>
void GInterpStage<TInput, TCode>::initProfilingScratch(MemoryPool* pool)
{
    if (!needsProfilingScratch()) return;
    if (d_profiling_errors_ != nullptr) return;
    if (pool == nullptr) {
        throw std::runtime_error(
            "GInterpStage: auto-tuning requires a MemoryPool");
    }
    persistent_pool_    = pool;
    d_profiling_errors_ = static_cast<float*>(
        pool->allocatePersistentDevice(
            kProfilingErrCount * sizeof(float), "ginterp_profiling_d"));
    h_profiling_errors_ = static_cast<float*>(
        pool->allocatePersistentPinned(
            kProfilingErrCount * sizeof(float), "ginterp_profiling_h"));
}

// ─── onFinalize ──────────────────────────────────────────────────────────────
template <typename TInput, typename TCode>
void GInterpStage<TInput, TCode>::onFinalize(
    size_t /*estimated_inlen*/, MemoryPool* pool)
{
    initProfilingScratch(pool);
    initOutlierCountScratch(pool);
}

// ─── pickAlphaFromRelEb ──────────────────────────────────────────────────────
// Reproduce the cuSZ-Hi piecewise-linear interpolation that picks alpha from
// rel_eb (see spline3.cu lines 80-103). beta is always 4.0 in this branch.
static double pickAlphaFromRelEb(double rel_eb)
{
    constexpr double a1 = 2.0, a2 = 1.75, a3 = 1.5, a4 = 1.25, a5 = 1.0;
    constexpr double e1 = 1e-1, e2 = 1e-2, e3 = 1e-3, e4 = 1e-4, e5 = 1e-5;
    if (rel_eb >= e1) return a1;
    if (rel_eb >= e2) return a2 + (a1 - a2) * (rel_eb - e2) / (e1 - e2);
    if (rel_eb >= e3) return a3 + (a2 - a3) * (rel_eb - e3) / (e2 - e3);
    if (rel_eb >= e4) return a4 + (a3 - a4) * (rel_eb - e4) / (e3 - e4);
    if (rel_eb >= e5) return a5 + (a4 - a5) * (rel_eb - e5) / (e4 - e5);
    return a5;
}

// ─── runAutoTuneMode1: cheap reverse-only ────────────────────────────────────
template <typename TInput, typename TCode>
static void runAutoTuneMode1(
    const TInput* d_data,
    dim3 data_len3,
    float* d_errors,
    float* h_errors,
    cudaStream_t stream,
    uint8_t (&out_reverse)[6])
{
    fz::ginterp::launchGInterpResetErrors(d_errors, stream);
    fz::ginterp::launchGInterpProfileMode1<TInput>(d_data, data_len3, d_errors, stream);
    FZ_CUDA_CHECK(cudaGetLastError());
    FZ_CUDA_CHECK(cudaMemcpyAsync(h_errors, d_errors, 2 * sizeof(float),
                                   cudaMemcpyDeviceToHost, stream));
    FZ_CUDA_CHECK(cudaStreamSynchronize(stream));
    bool do_reverse = (h_errors[1] > 3.0f * h_errors[0]);
    for (int i = 0; i < 4; ++i) out_reverse[i] = do_reverse ? 1 : 0;
    FZ_LOG(DEBUG,
        "GInterp auto-tune mode 1: errs[0]=%.3e errs[1]=%.3e -> reverse=%d",
        h_errors[0], h_errors[1], (int)do_reverse);
}

// ─── runAutoTuneMode2_3D: alternate cheap probe (3-D path) ───────────────────
//
// Implements cuSZ-Hi spline3.cu lines 127-148 (auto_tuning==2, l3.z != 1).
// `c_spline_profiling_data_2` writes 6 floats per axis. We sum opposing slots
// to pick a single `use_natural` value, then compare reverse-on vs reverse-off
// with a 3× margin to pick the global `reverse` flag. Replicates both decisions
// across all 4 active levels and forces `use_md` off.
template <typename TInput>
static void runAutoTuneMode2_3D(
    const TInput* d_data,
    dim3 data_len3,
    float* d_errors,
    float* h_errors,
    cudaStream_t stream,
    uint8_t (&out_use_md)[6],
    uint8_t (&out_use_natural)[6],
    uint8_t (&out_reverse)[6])
{
    fz::ginterp::launchGInterpProfileMode2<TInput>(
        d_data, data_len3, /*dim=*/3, d_errors, stream);
    FZ_CUDA_CHECK(cudaGetLastError());
    FZ_CUDA_CHECK(cudaMemcpyAsync(h_errors, d_errors, 6 * sizeof(float),
                                   cudaMemcpyDeviceToHost, stream));
    FZ_CUDA_CHECK(cudaStreamSynchronize(stream));

    const bool do_nat =
        (h_errors[0] + h_errors[2] + h_errors[4]) >
        (h_errors[1] + h_errors[3] + h_errors[5]);
    const int  nat_off    = do_nat ? 1 : 0;
    const bool do_reverse = (h_errors[4 + nat_off] > 3.0f * h_errors[nat_off]);

    for (int i = 0; i < 4; ++i) {
        out_use_natural[i] = do_nat     ? 1 : 0;
        out_reverse[i]     = do_reverse ? 1 : 0;
    }
    for (int i = 0; i < 6; ++i) out_use_md[i] = 0;

    FZ_LOG(DEBUG,
        "GInterp auto-tune mode 2 (3-D): errs={%.3e,%.3e,%.3e,%.3e,%.3e,%.3e} "
        "-> use_natural=%d reverse=%d (use_md=0 for all levels)",
        h_errors[0], h_errors[1], h_errors[2], h_errors[3], h_errors[4], h_errors[5],
        (int)do_nat, (int)do_reverse);
}

// ─── runAutoTuneMode2_2D: alternate cheap probe (2-D path) ───────────────────
//
// cuSZ-Hi spline3.cu lines 149-170 (auto_tuning==2, l3.z == 1). Slightly
// different decision rule: nat is summed over 4 slots (not 6), the reverse
// margin is 2× (not 3×), and the flags are written to all 6 level slots
// (LEVEL=6 in the 2-D path vs LEVEL=4 in 3-D).
template <typename TInput>
static void runAutoTuneMode2_2D(
    const TInput* d_data,
    dim3 data_len3,
    float* d_errors,
    float* h_errors,
    cudaStream_t stream,
    uint8_t (&out_use_md)[6],
    uint8_t (&out_use_natural)[6],
    uint8_t (&out_reverse)[6])
{
    fz::ginterp::launchGInterpProfileMode2<TInput>(
        d_data, data_len3, /*dim=*/2, d_errors, stream);
    FZ_CUDA_CHECK(cudaGetLastError());
    FZ_CUDA_CHECK(cudaMemcpyAsync(h_errors, d_errors, 6 * sizeof(float),
                                   cudaMemcpyDeviceToHost, stream));
    FZ_CUDA_CHECK(cudaStreamSynchronize(stream));

    const bool do_nat =
        (h_errors[0] + h_errors[2]) > (h_errors[1] + h_errors[3]);
    const int  nat_off    = do_nat ? 1 : 0;
    const bool do_reverse = (h_errors[2 + nat_off] > 2.0f * h_errors[nat_off]);

    for (int i = 0; i < 6; ++i) {
        out_use_natural[i] = do_nat     ? 1 : 0;
        out_reverse[i]     = do_reverse ? 1 : 0;
        out_use_md[i]      = 0;
    }

    FZ_LOG(DEBUG,
        "GInterp auto-tune mode 2 (2-D): errs={%.3e,%.3e,%.3e,%.3e,%.3e,%.3e} "
        "-> use_natural=%d reverse=%d (use_md=0 for all levels)",
        h_errors[0], h_errors[1], h_errors[2], h_errors[3], h_errors[4], h_errors[5],
        (int)do_nat, (int)do_reverse);
}

// ─── runAutoTuneMode3: full structural (cuSZ-Hi paper mode) ──────────────────
// Implements the analysis from spline3.cu lines 215-288 (3-D branch).
template <typename TInput, typename TCode>
static void runAutoTuneMode3(
    const TInput* d_data,
    dim3 data_len3,
    float eb_r, float ebx2,
    INTERPOLATION_PARAMS intp_param,
    float* d_errors,
    float* h_errors,
    cudaStream_t stream,
    uint8_t (&out_use_md)[6],
    uint8_t (&out_use_natural)[6],
    uint8_t (&out_reverse)[6])
{
    // S_STRIDE = 8 * BLOCK16 (cuSZ-Hi line 175).
    constexpr int kBlock16 = 16;
    const int S_STRIDE = 8 * kBlock16;

    auto calc_start_size = [&](int dim, int& s_start, int& s_size, int block_sz) {
        int mid = dim / 2;
        int k = (mid - block_sz / 2) / S_STRIDE;
        int t = (dim - block_sz / 2 - 1 - mid) / S_STRIDE;
        s_start = mid - k * S_STRIDE;
        s_size  = k + t + 1;
        if (s_size < 1) s_size = 1;
        if (s_start < 0) s_start = 0;
    };
    int sx_start, sy_start, sz_start;
    int sx_size,  sy_size,  sz_size;
    calc_start_size((int)data_len3.x, sx_start, sx_size, kBlock16);
    calc_start_size((int)data_len3.y, sy_start, sy_size, kBlock16);
    calc_start_size((int)data_len3.z, sz_start, sz_size, kBlock16);

    fz::ginterp::launchGInterpResetErrors(d_errors, stream);
    fz::ginterp::launchGInterpProfileMode3<TInput>(
        d_data, data_len3, /*dim=*/3,
        dim3((unsigned)sx_start, (unsigned)sy_start, (unsigned)sz_start),
        dim3((unsigned)sx_size,  (unsigned)sy_size,  (unsigned)sz_size),
        dim3((unsigned)S_STRIDE, (unsigned)S_STRIDE, (unsigned)S_STRIDE),
        eb_r, ebx2, intp_param,
        d_errors, /*workflow=*/true, stream);
    FZ_CUDA_CHECK(cudaGetLastError());
    FZ_CUDA_CHECK(cudaMemcpyAsync(h_errors, d_errors, 18 * sizeof(float),
                                   cudaMemcpyDeviceToHost, stream));
    FZ_CUDA_CHECK(cudaStreamSynchronize(stream));

    // Level 3 (errors[0..2]): pick best of {reverse off, reverse on, use_md}.
    float best;
    if (h_errors[0] > h_errors[1]) { best = h_errors[1]; out_reverse[3] = 1; }
    else                            { best = h_errors[0]; out_reverse[3] = 0; }
    out_use_md[3] = (h_errors[2] < best) ? 1 : 0;
    if (h_errors[2] < best) best = h_errors[2];

    // Level 2 (errors[3..5]).
    if (h_errors[3] > h_errors[4]) { best = h_errors[4]; out_reverse[2] = 1; }
    else                            { best = h_errors[3]; out_reverse[2] = 0; }
    out_use_md[2] = (h_errors[5] < best) ? 1 : 0;

    // Level 1 (errors[6..11], 6 variants): use_natural × reverse × use_md.
    int best_idx = 6;
    best = h_errors[6];
    for (int i = 7; i < 12; ++i) {
        if (h_errors[i] < best) { best = h_errors[i]; best_idx = i; }
    }
    out_use_natural[1] = (best_idx > 8) ? 1 : 0;
    out_use_md[1]      = (best_idx == 8 || best_idx == 11) ? 1 : 0;
    out_reverse[1]     = static_cast<uint8_t>(best_idx % 3);

    // Level 0 (errors[12..17]).
    best_idx = 12;
    best = h_errors[12];
    for (int i = 13; i < 18; ++i) {
        if (h_errors[i] < best) { best = h_errors[i]; best_idx = i; }
    }
    out_use_natural[0] = (best_idx > 14) ? 1 : 0;
    out_use_md[0]      = (best_idx == 14 || best_idx == 17) ? 1 : 0;
    out_reverse[0]     = static_cast<uint8_t>((best_idx - 12) % 3);

    FZ_LOG(DEBUG,
        "GInterp auto-tune mode 3: lvl{3,2,1,0} reverse={%d,%d,%d,%d} use_md={%d,%d,%d,%d} use_nat={%d,%d,%d,%d}",
        (int)out_reverse[3], (int)out_reverse[2], (int)out_reverse[1], (int)out_reverse[0],
        (int)out_use_md[3],  (int)out_use_md[2],  (int)out_use_md[1],  (int)out_use_md[0],
        (int)out_use_natural[3], (int)out_use_natural[2],
        (int)out_use_natural[1], (int)out_use_natural[0]);
}

// ─── runAutoTuneMode4: alpha/beta sweep (on top of mode-3 structural) ────────
//
// Runs `pa_spline_infprecis_data` with workflow=false. The kernel's
// `pre_compute_att` (SPLINE_DIM==3 SPLINE3_AB_ATT block) enumerates 11
// (alpha, beta) combos across BIY=0..10 and writes one error per combo:
//   BIY=0 → (1.0, 2.0)
//   BIY=1 → (1.25, 2.0)
//   BIY=2 → (1.5,  2.0)    BIY=3 → (1.5,  3.0)    BIY=4 → (1.5,  4.0)
//   BIY=5 → (1.75, 2.0)    BIY=6 → (1.75, 3.0)    BIY=7 → (1.75, 4.0)
//   BIY=8 → (2.0,  2.0)    BIY=9 → (2.0,  3.0)    BIY=10→ (2.0,  4.0)
// We pick the lowest-error combo as the resolved alpha/beta. Matches the
// 11-row alpha/beta table at the bottom of cuSZ-Hi `spline3.cu`
// (`auto_tuning==4` and `>=5` blocks).
template <typename TInput, typename TCode>
static void runAutoTuneMode4(
    const TInput* d_data,
    dim3 data_len3,
    float eb_r, float ebx2,
    INTERPOLATION_PARAMS intp_param,   // carries the mode-3-resolved flags
    float* d_errors,
    float* h_errors,
    cudaStream_t stream,
    double& out_alpha,
    double& out_beta)
{
    constexpr int kBlock16 = 16;
    const int S_STRIDE = 8 * kBlock16;

    auto calc_start_size = [&](int dim, int& s_start, int& s_size, int block_sz) {
        int mid = dim / 2;
        int k = (mid - block_sz / 2) / S_STRIDE;
        int t = (dim - block_sz / 2 - 1 - mid) / S_STRIDE;
        s_start = mid - k * S_STRIDE;
        s_size  = k + t + 1;
        if (s_size < 1) s_size = 1;
        if (s_start < 0) s_start = 0;
    };
    int sx_start, sy_start, sz_start;
    int sx_size,  sy_size,  sz_size;
    calc_start_size((int)data_len3.x, sx_start, sx_size, kBlock16);
    calc_start_size((int)data_len3.y, sy_start, sy_size, kBlock16);
    calc_start_size((int)data_len3.z, sz_start, sz_size, kBlock16);

    fz::ginterp::launchGInterpResetErrors(d_errors, stream);
    fz::ginterp::launchGInterpProfileMode3<TInput>(
        d_data, data_len3, /*dim=*/3,
        dim3((unsigned)sx_start, (unsigned)sy_start, (unsigned)sz_start),
        dim3((unsigned)sx_size,  (unsigned)sy_size,  (unsigned)sz_size),
        dim3((unsigned)S_STRIDE, (unsigned)S_STRIDE, (unsigned)S_STRIDE),
        eb_r, ebx2, intp_param,
        d_errors, /*workflow=*/false, stream);
    FZ_CUDA_CHECK(cudaGetLastError());
    FZ_CUDA_CHECK(cudaMemcpyAsync(h_errors, d_errors, 11 * sizeof(float),
                                   cudaMemcpyDeviceToHost, stream));
    FZ_CUDA_CHECK(cudaStreamSynchronize(stream));

    // (alpha, beta) table — must match pre_compute_att SPLINE3_AB_ATT
    // (11 combos: cuSZ-Hi spline3.cu lines 455-466 closed-form mapping).
    static constexpr double kAlphaBeta[11][2] = {
        {1.0,  2.0}, {1.25, 2.0},
        {1.5,  2.0}, {1.5,  3.0}, {1.5,  4.0},
        {1.75, 2.0}, {1.75, 3.0}, {1.75, 4.0},
        {2.0,  2.0}, {2.0,  3.0}, {2.0,  4.0},
    };

    int best_idx = 0;
    float best   = h_errors[0];
    for (int i = 1; i < 11; ++i) {
        if (h_errors[i] < best) { best = h_errors[i]; best_idx = i; }
    }
    out_alpha = kAlphaBeta[best_idx][0];
    out_beta  = kAlphaBeta[best_idx][1];

    FZ_LOG(DEBUG,
        "GInterp auto-tune mode 4: best combo idx=%d → alpha=%.3f beta=%.3f "
        "(errs=%.3e/%.3e/%.3e/%.3e/%.3e/%.3e/%.3e/%.3e/%.3e/%.3e/%.3e)",
        best_idx, out_alpha, out_beta,
        h_errors[0], h_errors[1], h_errors[2],  h_errors[3],
        h_errors[4], h_errors[5], h_errors[6],  h_errors[7],
        h_errors[8], h_errors[9], h_errors[10]);
}

// ─── runAutoTuneMode3_2D: full structural (2-D path) ─────────────────────────
//
// Mirrors `runAutoTuneMode3` but for 2-D inputs. The 2-D `pa_spline_infprecis`
// kernel writes errors[0..26] with a different per-BIY layout (see kernel
// launcher doc). For our LEVEL=4 / 16×16 anchor tile the BIY=0 errors[0..5]
// probe kernel-levels 5,4 which are degenerate at this tile size and are
// ignored. We consume:
//   errors[6..8]    coarsest probed (kernel level=3) — 3 variants
//                   (rev off, rev on, use_md) → resolves intp_param[3]
//   errors[9..14]   kernel level=2 — 6 variants → resolves intp_param[2]
//   errors[15..20]  kernel level=1 — 6 variants → resolves intp_param[1]
//   errors[21..26]  kernel level=0 — 6 variants → resolves intp_param[0]
// Variant decoding follows cuSZ-Hi spline3.cu's 2-D analysis loop
// (line 344-362): best_idx in [base..base+5] →
//   use_natural = ((best_idx + 3) % 6) > 2
//   use_md      = ((best_idx + 3) % 6) == 2 or == 5
//   reverse     = (best_idx + 3) % 3
// `S_STRIDE = 20 * AnchorBlockSizeX = 320` matches cuSZ-Hi line 174.
template <typename TInput, typename TCode>
static void runAutoTuneMode3_2D(
    const TInput* d_data,
    dim3 data_len3,
    float eb_r, float ebx2,
    INTERPOLATION_PARAMS intp_param,
    float* d_errors,
    float* h_errors,
    cudaStream_t stream,
    uint8_t (&out_use_md)[6],
    uint8_t (&out_use_natural)[6],
    uint8_t (&out_reverse)[6])
{
    constexpr int kBlock16  = 16;
    const int     S_STRIDE  = 20 * kBlock16;

    auto calc_start_size = [&](int dim, int& s_start, int& s_size, int block_sz) {
        int mid = dim / 2;
        int k = (mid - block_sz / 2) / S_STRIDE;
        int t = (dim - block_sz / 2 - 1 - mid) / S_STRIDE;
        s_start = mid - k * S_STRIDE;
        s_size  = k + t + 1;
        if (s_size < 1) s_size = 1;
        if (s_start < 0) s_start = 0;
    };
    int sx_start, sy_start, sz_start;
    int sx_size,  sy_size,  sz_size;
    calc_start_size((int)data_len3.x, sx_start, sx_size, kBlock16);
    calc_start_size((int)data_len3.y, sy_start, sy_size, kBlock16);
    // z fixed at 1 for 2-D — calc_start_size with size=1 produces s_start=0, s_size=1.
    calc_start_size((int)data_len3.z, sz_start, sz_size, kBlock16);

    fz::ginterp::launchGInterpResetErrors(d_errors, stream);
    fz::ginterp::launchGInterpProfileMode3<TInput>(
        d_data, data_len3, /*dim=*/2,
        dim3((unsigned)sx_start, (unsigned)sy_start, (unsigned)sz_start),
        dim3((unsigned)sx_size,  (unsigned)sy_size,  (unsigned)sz_size),
        dim3((unsigned)S_STRIDE, (unsigned)S_STRIDE, (unsigned)S_STRIDE),
        eb_r, ebx2, intp_param,
        d_errors, /*workflow=*/true, stream);
    FZ_CUDA_CHECK(cudaGetLastError());
    FZ_CUDA_CHECK(cudaMemcpyAsync(h_errors, d_errors, 27 * sizeof(float),
                                   cudaMemcpyDeviceToHost, stream));
    FZ_CUDA_CHECK(cudaStreamSynchronize(stream));

    // Level 3 (kernel coarsest, errors[6..8]): 3 variants — rev off, rev on, use_md.
    float best;
    if (h_errors[6] > h_errors[7]) { best = h_errors[7]; out_reverse[3] = 1; }
    else                            { best = h_errors[6]; out_reverse[3] = 0; }
    out_use_md[3] = (h_errors[8] < best) ? 1 : 0;

    // Helper for the 6-variant decode used by levels 2, 1, 0 (cuSZ-Hi 2-D loop).
    auto decode_6 = [](const float* e, uint8_t& um, uint8_t& un, uint8_t& rv) {
        int best_idx = 0;
        float bv = e[0];
        for (int i = 1; i < 6; ++i) if (e[i] < bv) { bv = e[i]; best_idx = i; }
        const int adj = (best_idx + 3) % 6;
        un = (adj > 2) ? 1 : 0;
        um = (adj == 2 || adj == 5) ? 1 : 0;
        rv = static_cast<uint8_t>((best_idx + 3) % 3);
    };

    decode_6(&h_errors[9],  out_use_md[2], out_use_natural[2], out_reverse[2]);
    decode_6(&h_errors[15], out_use_md[1], out_use_natural[1], out_reverse[1]);
    decode_6(&h_errors[21], out_use_md[0], out_use_natural[0], out_reverse[0]);

    FZ_LOG(DEBUG,
        "GInterp auto-tune mode 3 (2-D): lvl{3,2,1,0} reverse={%d,%d,%d,%d} "
        "use_md={%d,%d,%d,%d} use_nat={%d,%d,%d,%d}",
        (int)out_reverse[3], (int)out_reverse[2], (int)out_reverse[1], (int)out_reverse[0],
        (int)out_use_md[3],  (int)out_use_md[2],  (int)out_use_md[1],  (int)out_use_md[0],
        (int)out_use_natural[3], (int)out_use_natural[2],
        (int)out_use_natural[1], (int)out_use_natural[0]);
}

// ─── runAutoTuneMode4_2D: alpha/beta sweep (2-D path) ────────────────────────
//
// Same 11-(α,β)-combo sweep as `runAutoTuneMode4` but for 2-D. The kernel's
// SPLINE_DIM==2 SPLINE3_AB_ATT pre_compute_att enumerates the same BIY=0..10
// alpha/beta mapping (cuSZ-Hi spline3.cu line 2063-2076 in ginterp_md.inl).
template <typename TInput, typename TCode>
static void runAutoTuneMode4_2D(
    const TInput* d_data,
    dim3 data_len3,
    float eb_r, float ebx2,
    INTERPOLATION_PARAMS intp_param,
    float* d_errors,
    float* h_errors,
    cudaStream_t stream,
    double& out_alpha,
    double& out_beta)
{
    constexpr int kBlock16 = 16;
    const int     S_STRIDE = 20 * kBlock16;

    auto calc_start_size = [&](int dim, int& s_start, int& s_size, int block_sz) {
        int mid = dim / 2;
        int k = (mid - block_sz / 2) / S_STRIDE;
        int t = (dim - block_sz / 2 - 1 - mid) / S_STRIDE;
        s_start = mid - k * S_STRIDE;
        s_size  = k + t + 1;
        if (s_size < 1) s_size = 1;
        if (s_start < 0) s_start = 0;
    };
    int sx_start, sy_start, sz_start;
    int sx_size,  sy_size,  sz_size;
    calc_start_size((int)data_len3.x, sx_start, sx_size, kBlock16);
    calc_start_size((int)data_len3.y, sy_start, sy_size, kBlock16);
    calc_start_size((int)data_len3.z, sz_start, sz_size, kBlock16);

    fz::ginterp::launchGInterpResetErrors(d_errors, stream);
    fz::ginterp::launchGInterpProfileMode3<TInput>(
        d_data, data_len3, /*dim=*/2,
        dim3((unsigned)sx_start, (unsigned)sy_start, (unsigned)sz_start),
        dim3((unsigned)sx_size,  (unsigned)sy_size,  (unsigned)sz_size),
        dim3((unsigned)S_STRIDE, (unsigned)S_STRIDE, (unsigned)S_STRIDE),
        eb_r, ebx2, intp_param,
        d_errors, /*workflow=*/false, stream);
    FZ_CUDA_CHECK(cudaGetLastError());
    FZ_CUDA_CHECK(cudaMemcpyAsync(h_errors, d_errors, 11 * sizeof(float),
                                   cudaMemcpyDeviceToHost, stream));
    FZ_CUDA_CHECK(cudaStreamSynchronize(stream));

    static constexpr double kAlphaBeta[11][2] = {
        {1.0,  2.0}, {1.25, 2.0},
        {1.5,  2.0}, {1.5,  3.0}, {1.5,  4.0},
        {1.75, 2.0}, {1.75, 3.0}, {1.75, 4.0},
        {2.0,  2.0}, {2.0,  3.0}, {2.0,  4.0},
    };
    int best_idx = 0;
    float best   = h_errors[0];
    for (int i = 1; i < 11; ++i) {
        if (h_errors[i] < best) { best = h_errors[i]; best_idx = i; }
    }
    out_alpha = kAlphaBeta[best_idx][0];
    out_beta  = kAlphaBeta[best_idx][1];

    FZ_LOG(DEBUG,
        "GInterp auto-tune mode 4 (2-D): best combo idx=%d → alpha=%.3f beta=%.3f",
        best_idx, out_alpha, out_beta);
}

// ─── setDims ─────────────────────────────────────────────────────────────────
template <typename TInput, typename TCode>
void GInterpStage<TInput, TCode>::setDims(const std::array<size_t, 3>& dims) {
    if (dims[0] == 0 || dims[1] == 0 || dims[2] == 0) {
        throw std::runtime_error(
            "GInterpStage::setDims: all three dimensions must be > 0 "
            "(2-D inputs must still pass dims[2]=1, not 0)");
    }
    // 1-D rejection: cuSZ-Hi has no 1-D spline path, and a 2-D kernel with
    // ny=1 degenerates badly (only one row of anchors, no boundary samples).
    if (dims[1] <= 1) {
        throw std::runtime_error(
            "GInterpStage::setDims: dims[1] must be > 1; 1-D input is not "
            "supported (got y=" + std::to_string(dims[1]) + ")");
    }
    config_.dims = dims;
    dim3 anchor = (dims[2] > 1)
        ? ginterp::ginterpAnchorLen3(dims[0], dims[1], dims[2])
        : ginterp::ginterpAnchorLen2(dims[0], dims[1]);
    anchor_dims_ = {anchor.x, anchor.y, anchor.z};
}

// ─── estimateOutputSizes ─────────────────────────────────────────────────────
template <typename TInput, typename TCode>
std::vector<size_t> GInterpStage<TInput, TCode>::estimateOutputSizes(
    const std::vector<size_t>& input_sizes) const
{
    if (input_sizes.empty()) return {0, 0, 0, 0};
    size_t input_bytes = input_sizes[0];
    size_t N           = input_bytes / sizeof(TInput);
    size_t max_outliers = getMaxOutlierCount(N);

    // Anchor count from cached dims (set by setDims). If setDims wasn't
    // called (e.g. estimate is being called before finalize from a pre-finalize
    // path), fall back to a conservative 1/64 estimate (worst case for the
    // smallest legal volume: nx=ny=nz=2 → anchor 1×1×1, but we use a
    // generous upper bound here since the DAG only sizes once).
    size_t anchor_N = (anchor_dims_[0] && anchor_dims_[1] && anchor_dims_[2])
        ? (anchor_dims_[0] * anchor_dims_[1] * anchor_dims_[2])
        : ((N + 4095) / 4096);  // ≤ 1/4096 of N for any valid setDims()

    return {
        N           * sizeof(TCode),    // codes
        anchor_N    * sizeof(TInput),   // anchor
        max_outliers * sizeof(TInput),  // outlier_vals
        max_outliers * sizeof(uint32_t),// outlier_idxs
    };
}

// ─── execute ─────────────────────────────────────────────────────────────────
template <typename TInput, typename TCode>
void GInterpStage<TInput, TCode>::execute(
    cudaStream_t stream,
    MemoryPool* pool,
    const std::vector<void*>& inputs,
    const std::vector<void*>& outputs,
    const std::vector<size_t>& sizes)
{
    if (is_inverse_) {
        // ───── DECOMPRESS: 4 inputs → 1 output ────────────────────────────
        if (inputs.size() < 4 || outputs.empty() || sizes.size() < 4) {
            throw std::runtime_error(
                "GInterpStage (inverse): requires 4 inputs and 1 output");
        }
        if (config_.dims[0] == 0 || config_.dims[1] <= 1) {
            throw std::runtime_error(
                "GInterpStage (inverse): dims not set — deserializeHeader() "
                "should have populated them");
        }

        size_t nx = config_.dims[0];
        size_t ny = config_.dims[1];
        size_t nz = config_.dims[2];
        size_t N  = nx * ny * nz;
        const int dim = ndim();
        if (N == 0) {
            actual_output_sizes_.assign(1, 0);
            return;
        }

        // Conversion factors (computed_abs_eb_ set by deserializeHeader).
        // cuSZ-Hi convention: eb_r = 1/eb (NOT 1/(2*eb)). The kernel's
        // quantizer at ginterp_md.inl:865 does `int(code/2)`, which combined
        // with `eb_r = 1/eb` gives the canonical `round(err / (2*eb))` mapping.
        TInput abs_eb = computed_abs_eb_;
        float  ebx2   = static_cast<float>(2.0 * abs_eb);
        float  eb_r   = (abs_eb > TInput(0)) ? (1.0f / static_cast<float>(abs_eb)) : 0.0f;

        // Outlier count comes from the deserialized FZM header (set by
        // deserializeHeader() into actual_outlier_count_), not from a port.
        uint32_t outlier_n = actual_outlier_count_;

        // 1. Allocate full-N outlier_tmp from the pool (stream-ordered).
        TInput* d_outlier_tmp = static_cast<TInput*>(
            pool->allocate(N * sizeof(TInput), stream,
                           "ginterp_outlier_tmp", /*persistent=*/false));
        FZ_CUDA_CHECK(cudaMemsetAsync(d_outlier_tmp, 0,
                                       N * sizeof(TInput), stream));

        // 2. Scatter outliers into the temp buffer using the host-side count
        // (passed as a kernel arg — no D2H to find the loop bound).
        ginterp::launchScatterOutliers<TInput>(
            static_cast<const TInput*>(inputs[2]),
            static_cast<const uint32_t*>(inputs[3]),
            outlier_n,
            d_outlier_tmp,
            stream);
        FZ_CUDA_CHECK(cudaGetLastError());

        // 3. Launch the decode kernel.
        dim3 data_len3(static_cast<unsigned>(nx),
                       static_cast<unsigned>(ny),
                       static_cast<unsigned>(nz));
        dim3 anchor_len3(static_cast<unsigned>(anchor_dims_[0]),
                         static_cast<unsigned>(anchor_dims_[1]),
                         static_cast<unsigned>(anchor_dims_[2]));

        INTERPOLATION_PARAMS intp_param = buildIntpParam();
        if (dim == 3) {
            ginterp::launchGInterpInverse3D<TInput, TCode>(
                static_cast<const TCode*>(inputs[0]),  data_len3,
                static_cast<const TInput*>(inputs[1]), anchor_len3,
                d_outlier_tmp,
                static_cast<TInput*>(outputs[0]),
                eb_r, ebx2, static_cast<int>(config_.quant_radius),
                intp_param,
                stream);
        } else {
            ginterp::launchGInterpInverse2D<TInput, TCode>(
                static_cast<const TCode*>(inputs[0]),  data_len3,
                static_cast<const TInput*>(inputs[1]), anchor_len3,
                d_outlier_tmp,
                static_cast<TInput*>(outputs[0]),
                eb_r, ebx2, static_cast<int>(config_.quant_radius),
                intp_param,
                stream);
        }
        FZ_CUDA_CHECK(cudaGetLastError());

        // 4. Return outlier_tmp to the pool (stream-ordered free).
        pool->free(d_outlier_tmp, stream);

        actual_output_sizes_.assign(1, N * sizeof(TInput));
        return;
    }

    // ───── COMPRESS: 1 input → 4 outputs ─────────────────────────────────────
    if (inputs.empty() || outputs.size() < 4 || sizes.empty()) {
        throw std::runtime_error(
            "GInterpStage: requires 1 input and 4 outputs");
    }
    if (config_.dims[0] == 0 || config_.dims[1] <= 1) {
        throw std::runtime_error(
            "GInterpStage: dims not set — call Pipeline::setDims() "
            "before addStage() (1-D not supported)");
    }
    const int dim = ndim();

    size_t input_bytes  = sizes[0];
    size_t N            = input_bytes / sizeof(TInput);
    size_t max_outliers = getMaxOutlierCount(N);
    num_elements_       = N;

    if (N == 0) {
        for (size_t i = 0; i < 4; i++) actual_output_sizes_[i] = 0;
        actual_outlier_count_ = 0;
        return;
    }

    // Zero the stage-private outlier-count scratch (kernel uses atomicAdd).
    // Lazy-allocate it here for MINIMAL mode (PREALLOCATE allocated it in onFinalize).
    initOutlierCountScratch(pool);
    FZ_CUDA_CHECK(cudaMemsetAsync(d_outlier_count_scratch_, 0,
                                   sizeof(uint32_t), stream));

    // Resolve absolute error bound (ABS direct; REL/NOA via scan unless caller
    // pre-computed value_base for graph-safe operation).
    if (config_.eb_mode == ErrorBoundMode::ABS) {
        computed_abs_eb_     = static_cast<TInput>(config_.error_bound);
        computed_value_base_ = 0.0f;
    } else {
        float value_base = config_.precomputed_value_base;
        if (value_base <= 0.0f) {
            value_base = computeValueBase<TInput>(
                static_cast<const TInput*>(inputs[0]),
                N, config_.eb_mode, stream, pool);
        }
        computed_value_base_ = value_base;
        if (value_base <= 0.0f) {
            FZ_LOG(WARN,
                "GInterpStage: value_base is zero for %s mode "
                "(constant or empty data?); falling back to ABS",
                config_.eb_mode == ErrorBoundMode::NOA ? "NOA" : "REL");
            computed_abs_eb_ = static_cast<TInput>(config_.error_bound);
        } else {
            computed_abs_eb_ = static_cast<TInput>(config_.error_bound)
                                * static_cast<TInput>(value_base);
        }
    }

    // cuSZ-Hi convention: eb_r = 1/eb (see inverse path comment above).
    float ebx2 = static_cast<float>(2.0 * computed_abs_eb_);
    float eb_r = (computed_abs_eb_ > TInput(0))
                 ? (1.0f / static_cast<float>(computed_abs_eb_)) : 0.0f;

    // ── Auto-tune radius if user left it at the sentinel (0) ──
    // Manual override (any radius > 0) skips the scan — required for CUDA
    // graph capture, since the scan does a D2H sync inside computeValueBase.
    if (config_.quant_radius == 0 && ebx2 > 0.0f) {
        float data_range = 0.0f;
        if (config_.eb_mode == ErrorBoundMode::NOA && computed_value_base_ > 0.0f) {
            // NOA's value_base is already (max - min); reuse it.
            data_range = computed_value_base_;
        } else if (config_.eb_mode == ErrorBoundMode::REL && computed_value_base_ > 0.0f) {
            // REL's value_base is max(|data|); the full range is at most 2x.
            data_range = 2.0f * computed_value_base_;
        } else {
            // ABS, or NOA/REL without pre-computed value_base: do the scan.
            // We always invoke NOA mode here because we want the actual data
            // range (max - min), not a magnitude.
            data_range = computeValueBase<TInput>(
                static_cast<const TInput*>(inputs[0]),
                N, ErrorBoundMode::NOA, stream, pool);
        }
        int auto_r = pickAutoRadius<TCode>(data_range, ebx2);
        config_.quant_radius = auto_r;
        FZ_LOG(DEBUG,
               "GInterpStage: auto-tuned radius=%d (data_range=%.6g, ebx2=%.6g)",
               auto_r, static_cast<double>(data_range),
               static_cast<double>(ebx2));
    }

    dim3 data_len3(static_cast<unsigned>(config_.dims[0]),
                   static_cast<unsigned>(config_.dims[1]),
                   static_cast<unsigned>(config_.dims[2]));
    dim3 anchor_len3(static_cast<unsigned>(anchor_dims_[0]),
                     static_cast<unsigned>(anchor_dims_[1]),
                     static_cast<unsigned>(anchor_dims_[2]));

    // ── INTERPOLATION_PARAMS auto-tune dispatch ──
    // Modes:
    //   0 : off (deterministic baseline) — also falls through here
    //   1 : cheap reverse-only probe (3-D only; D2H sync)
    //   2 : alternate cheap probe — use_natural × reverse (dim-agnostic; D2H sync)
    //   3 : full structural probe (dim-agnostic; D2H sync)
    //   4 : mode 3 + alpha/beta sweep (dim-agnostic; 2 D2H syncs)
    //   5 : manual alpha/beta override, no profiling (dim-agnostic, graph-safe)
    // Mode 1 is 3-D only (`c_spline_profiling_data` has no SPLINE_DIM==2
    // branch). Modes 2/3/4/5 work on both dims. On 2-D mode 1 we warn and
    // fall back to baseline.
    const bool needs_profiling =
        (config_.auto_tuning_mode == 1 || config_.auto_tuning_mode == 3 ||
         config_.auto_tuning_mode == 4);
    if (config_.auto_tuning_mode == 1 && dim == 2) {
        FZ_LOG(WARN,
            "GInterp: setAutoTuning(1) is wired for the 3-D path only — "
            "falling back to deterministic baseline for this 2-D input "
            "(try mode 2, 3, 4, or 5 for 2-D auto-tuning)");
    }

    // Compute rel_eb (needed for the piecewise-linear alpha schedule) for
    // any non-zero mode.
    auto compute_rel_eb = [&]() -> double {
        if (config_.eb_mode == ErrorBoundMode::ABS) {
            float range = computed_value_base_;
            if (range <= 0.0f) {
                range = computeValueBase<TInput>(
                    static_cast<const TInput*>(inputs[0]),
                    N, ErrorBoundMode::NOA, stream, pool);
                computed_value_base_ = range;
            }
            return (range > 0.0f)
                ? static_cast<double>(config_.error_bound) / range : 0.0;
        }
        return static_cast<double>(config_.error_bound);
    };

    if (config_.auto_tuning_mode == 5) {
        // Manual override path — no profiling, dim-agnostic, graph-safe.
        // Reset structural flags to baseline (mode 5 only tunes alpha/beta).
        std::memset(resolved_use_md_,      0, sizeof(resolved_use_md_));
        std::memset(resolved_use_natural_, 0, sizeof(resolved_use_natural_));
        std::memset(resolved_reverse_,     0, sizeof(resolved_reverse_));
        resolved_use_md_[0] = resolved_use_md_[1] = 1;

        // Defer compute_rel_eb() until we know it's needed — the rel_eb scan
        // would break CUDA Graph capture, but with user-supplied alpha+beta
        // we never look at it.
        const bool need_rel_eb_for_alpha = (config_.manual_alpha <= 0.0);
        const double rel_eb = need_rel_eb_for_alpha ? compute_rel_eb() : 0.0;
        resolved_alpha_ = (config_.manual_alpha > 0.0)
            ? config_.manual_alpha
            : pickAlphaFromRelEb(rel_eb);
        resolved_beta_  = (config_.manual_beta > 0.0)
            ? config_.manual_beta
            : 4.0;
        FZ_LOG(DEBUG,
            "GInterp auto-tune mode 5: manual override alpha=%.3f beta=%.3f "
            "(user manual=%g/%g, rel_eb=%g)",
            resolved_alpha_, resolved_beta_,
            config_.manual_alpha, config_.manual_beta, rel_eb);
    } else if (config_.auto_tuning_mode == 2) {
        // Mode 2: alternate cheap probe. Dim-agnostic — works on both 3-D
        // and 2-D (cuSZ-Hi spline3.cu branches on l3.z != 1).
        initProfilingScratch(pool);

        const double rel_eb = compute_rel_eb();
        resolved_alpha_ = pickAlphaFromRelEb(rel_eb);
        resolved_beta_  = 4.0;

        std::memset(resolved_use_md_,      0, sizeof(resolved_use_md_));
        std::memset(resolved_use_natural_, 0, sizeof(resolved_use_natural_));
        std::memset(resolved_reverse_,     0, sizeof(resolved_reverse_));
        resolved_use_md_[0] = resolved_use_md_[1] = 1;

        if (dim == 3) {
            runAutoTuneMode2_3D<TInput>(
                static_cast<const TInput*>(inputs[0]), data_len3,
                d_profiling_errors_, h_profiling_errors_, stream,
                resolved_use_md_, resolved_use_natural_, resolved_reverse_);
        } else {
            runAutoTuneMode2_2D<TInput>(
                static_cast<const TInput*>(inputs[0]), data_len3,
                d_profiling_errors_, h_profiling_errors_, stream,
                resolved_use_md_, resolved_use_natural_, resolved_reverse_);
        }
    } else if (needs_profiling && (dim == 3 ||
                                   (dim == 2 && config_.auto_tuning_mode != 1))) {
        initProfilingScratch(pool);

        const double rel_eb = compute_rel_eb();
        resolved_alpha_ = pickAlphaFromRelEb(rel_eb);
        resolved_beta_  = 4.0;

        // Reset resolved flags to baseline before each tune.
        std::memset(resolved_use_md_,      0, sizeof(resolved_use_md_));
        std::memset(resolved_use_natural_, 0, sizeof(resolved_use_natural_));
        std::memset(resolved_reverse_,     0, sizeof(resolved_reverse_));
        resolved_use_md_[0] = resolved_use_md_[1] = 1;

        if (config_.auto_tuning_mode == 1) {
            // 3-D only (caught by the 2-D fallback warning above).
            runAutoTuneMode1<TInput, TCode>(
                static_cast<const TInput*>(inputs[0]), data_len3,
                d_profiling_errors_, h_profiling_errors_, stream,
                resolved_reverse_);
        } else if (config_.auto_tuning_mode == 3 ||
                   config_.auto_tuning_mode == 4) {
            INTERPOLATION_PARAMS profile_param = buildIntpParam();
            profile_param.alpha = resolved_alpha_;
            profile_param.beta  = resolved_beta_;
            if (dim == 3) {
                runAutoTuneMode3<TInput, TCode>(
                    static_cast<const TInput*>(inputs[0]), data_len3,
                    eb_r, ebx2, profile_param,
                    d_profiling_errors_, h_profiling_errors_, stream,
                    resolved_use_md_, resolved_use_natural_, resolved_reverse_);
            } else {
                runAutoTuneMode3_2D<TInput, TCode>(
                    static_cast<const TInput*>(inputs[0]), data_len3,
                    eb_r, ebx2, profile_param,
                    d_profiling_errors_, h_profiling_errors_, stream,
                    resolved_use_md_, resolved_use_natural_, resolved_reverse_);
            }
            if (config_.auto_tuning_mode == 4) {
                // Mode 4: take the mode-3-resolved flags and sweep alpha/beta
                // on top of them. profile_param needs the structural flags
                // updated (alpha/beta will be overwritten per-block by the
                // SPLINE3_AB_ATT pre_compute_att path).
                INTERPOLATION_PARAMS sweep_param = buildIntpParam();
                sweep_param.alpha = resolved_alpha_;
                sweep_param.beta  = resolved_beta_;
                if (dim == 3) {
                    runAutoTuneMode4<TInput, TCode>(
                        static_cast<const TInput*>(inputs[0]), data_len3,
                        eb_r, ebx2, sweep_param,
                        d_profiling_errors_, h_profiling_errors_, stream,
                        resolved_alpha_, resolved_beta_);
                } else {
                    runAutoTuneMode4_2D<TInput, TCode>(
                        static_cast<const TInput*>(inputs[0]), data_len3,
                        eb_r, ebx2, sweep_param,
                        d_profiling_errors_, h_profiling_errors_, stream,
                        resolved_alpha_, resolved_beta_);
                }
            }
        }
    } else if (config_.auto_tuning_mode > 0 &&
               config_.auto_tuning_mode != 5 && dim == 3) {
        // Unknown mode on 3-D — log once and fall back.
        FZ_LOG(WARN,
            "GInterp: unsupported auto_tuning_mode=%u (only 0/1/2/3/4/5 wired); "
            "falling back to deterministic baseline",
            config_.auto_tuning_mode);
    }

    INTERPOLATION_PARAMS intp_param = buildIntpParam();
    if (dim == 3) {
        ginterp::launchGInterpForward3D<TInput, TCode>(
            static_cast<const TInput*>(inputs[0]), data_len3,
            static_cast<TCode*>(outputs[0]),
            static_cast<TInput*>(outputs[1]), anchor_len3,
            static_cast<TInput*>(outputs[2]),
            static_cast<uint32_t*>(outputs[3]),
            d_outlier_count_scratch_,
            eb_r, ebx2, static_cast<int>(config_.quant_radius),
            intp_param,
            stream);
    } else {
        ginterp::launchGInterpForward2D<TInput, TCode>(
            static_cast<const TInput*>(inputs[0]), data_len3,
            static_cast<TCode*>(outputs[0]),
            static_cast<TInput*>(outputs[1]), anchor_len3,
            static_cast<TInput*>(outputs[2]),
            static_cast<uint32_t*>(outputs[3]),
            d_outlier_count_scratch_,
            eb_r, ebx2, static_cast<int>(config_.quant_radius),
            intp_param,
            stream);
    }
    FZ_CUDA_CHECK(cudaGetLastError());

    // Max-capacity placeholders; postStreamSync() trims (and refreshes
    // actual_outlier_count_). We do NOT zero actual_outlier_count_ here:
    // this method may run inside cudaStreamBeginCapture/EndCapture (graph
    // recording), where postStreamSync isn't called — clobbering the count
    // would break the subsequent inverse path.
    size_t anchor_N             = anchor_dims_[0] * anchor_dims_[1] * anchor_dims_[2];
    actual_output_sizes_[0]     = N           * sizeof(TCode);
    actual_output_sizes_[1]     = anchor_N    * sizeof(TInput);
    actual_output_sizes_[2]     = max_outliers * sizeof(TInput);
    actual_output_sizes_[3]     = max_outliers * sizeof(uint32_t);
}

// ─── postStreamSync ──────────────────────────────────────────────────────────
template <typename TInput, typename TCode>
void GInterpStage<TInput, TCode>::postStreamSync(cudaStream_t /*stream*/) {
    if (is_inverse_ || d_outlier_count_scratch_ == nullptr) return;

    uint32_t h_outlier_count = 0;
    FZ_CUDA_CHECK(cudaMemcpy(&h_outlier_count, d_outlier_count_scratch_,
                              sizeof(uint32_t), cudaMemcpyDeviceToHost));

    size_t max_outliers = getMaxOutlierCount(num_elements_);
    if (h_outlier_count > max_outliers) {
        float actual_pct   = 100.0f * h_outlier_count
                              / static_cast<float>(num_elements_);
        float capacity_pct = 100.0f * max_outliers
                              / static_cast<float>(num_elements_);
        FZ_LOG(WARN,
               "GInterp outlier overflow! Detected %u (%.1f%%) outliers but "
               "only %.1f%% capacity allocated. Outliers beyond capacity were "
               "DROPPED — data will be corrupted for those elements. "
               "Increase outlier_capacity to at least %.1f%%.",
               h_outlier_count, actual_pct, capacity_pct, actual_pct * 1.1f);
        h_outlier_count = static_cast<uint32_t>(max_outliers);
    }

    actual_outlier_count_   = h_outlier_count;
    actual_output_sizes_[2] = h_outlier_count * sizeof(TInput);
    actual_output_sizes_[3] = h_outlier_count * sizeof(uint32_t);
}

// ─── serialization ───────────────────────────────────────────────────────────
template <typename TInput, typename TCode>
size_t GInterpStage<TInput, TCode>::serializeHeader(
    size_t /*output_index*/, uint8_t* buf, size_t max_size) const
{
    if (max_size < sizeof(GInterpConfig)) {
        throw std::runtime_error(
            "GInterpStage::serializeHeader: insufficient buffer");
    }
    GInterpConfig c;
    c.error_bound   = static_cast<float>(computed_abs_eb_);
    c.quant_radius  = static_cast<uint32_t>(config_.quant_radius);
    c.num_elements  = static_cast<uint32_t>(num_elements_);
    c.outlier_count = actual_outlier_count_;
    c.input_type    = inputDataType();
    c.code_type     = codeDataType();
    c.ndim          = static_cast<uint8_t>(ndim());
    c.eb_mode       = static_cast<uint8_t>(config_.eb_mode);
    c.dim_x         = static_cast<uint32_t>(config_.dims[0]);
    c.dim_y         = static_cast<uint32_t>(config_.dims[1]);
    c.dim_z         = static_cast<uint32_t>(config_.dims[2]);
    c.anchor_dim_x  = static_cast<uint32_t>(anchor_dims_[0]);
    c.anchor_dim_y  = static_cast<uint32_t>(anchor_dims_[1]);
    c.anchor_dim_z  = static_cast<uint32_t>(anchor_dims_[2]);
    c.user_eb       = static_cast<float>(config_.error_bound);
    c.value_base    = computed_value_base_;
    c.intp_alpha    = resolved_alpha_;
    c.intp_beta     = resolved_beta_;
    for (int i = 0; i < 6; ++i) {
        c.intp_use_md[i]      = resolved_use_md_[i];
        c.intp_use_natural[i] = resolved_use_natural_[i];
        c.intp_reverse[i]     = resolved_reverse_[i];
    }
    c.auto_tuning_mode = config_.auto_tuning_mode;
    std::memcpy(buf, &c, sizeof(c));
    return sizeof(c);
}

template <typename TInput, typename TCode>
void GInterpStage<TInput, TCode>::deserializeHeader(
    const uint8_t* buf, size_t size)
{
    if (size < sizeof(GInterpConfig)) {
        throw std::runtime_error(
            "GInterpStage::deserializeHeader: invalid config size");
    }
    GInterpConfig c;
    std::memcpy(&c, buf, sizeof(c));

    computed_abs_eb_       = static_cast<TInput>(c.error_bound);
    config_.error_bound    = c.user_eb;
    config_.quant_radius   = static_cast<int>(c.quant_radius);
    config_.eb_mode        = static_cast<ErrorBoundMode>(c.eb_mode);
    config_.precomputed_value_base = c.value_base;
    computed_value_base_   = c.value_base;
    num_elements_          = c.num_elements;
    actual_outlier_count_  = c.outlier_count;

    config_.dims[0] = c.dim_x;
    config_.dims[1] = c.dim_y;
    config_.dims[2] = c.dim_z;
    anchor_dims_[0] = c.anchor_dim_x;
    anchor_dims_[1] = c.anchor_dim_y;
    anchor_dims_[2] = c.anchor_dim_z;

    // Restore the resolved INTERPOLATION_PARAMS so the inverse kernel uses
    // the same configuration the encoder did. `auto_tuning_mode` is restored
    // on the config so save/load TOML round-trips also work.
    resolved_alpha_ = c.intp_alpha;
    resolved_beta_  = c.intp_beta;
    for (int i = 0; i < 6; ++i) {
        resolved_use_md_[i]      = c.intp_use_md[i];
        resolved_use_natural_[i] = c.intp_use_natural[i];
        resolved_reverse_[i]     = c.intp_reverse[i];
    }
    config_.auto_tuning_mode = c.auto_tuning_mode;
}

// ─── Explicit instantiations ─────────────────────────────────────────────────
template class GInterpStage<float, uint8_t>;
template class GInterpStage<float, uint16_t>;
template class GInterpStage<float, uint32_t>;

} // namespace fz
