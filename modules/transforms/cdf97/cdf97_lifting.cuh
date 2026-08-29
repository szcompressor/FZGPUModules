#pragma once

// Lifting constants (ALPHA/BETA/GAMMA/DELTA/EPSILON, computed from the filter
// bank coefficients, not the commented QccPack literals) and boundary
// handling ported from the SPERR project (Li, Lindstrom, Clyne — NCAR),
// Apache License 2.0 — see THIRD_PARTY.md. Original: include/CDF97.h,
// src/CDF97.cpp in https://github.com/NCAR/SPERR.

/**
 * @file cdf97_lifting.cuh
 * @brief CDF 9/7 biorthogonal wavelet — single-line lifting primitive.
 *
 * This header holds *only* the 1-D lifting kernel that every dimension of the
 * separable multi-level DWT is built from. It is deliberately isolated from the
 * pipeline/stage machinery so the numerically sensitive part can be validated
 * bit-for-bit against a reference before it is wired into a `Stage`.
 *
 * ### Provenance (must stay bit-exact)
 *
 * The lifting constants and the forward analysis sweep are a 1:1 port of
 * `sperr::CDF97::QccWAVCDF97AnalysisSymmetric` in NCAR/SPERR
 * (https://github.com/NCAR/SPERR, `src/CDF97.cpp` / `include/CDF97.h`). We port
 * the *computed* constant path (derived from the Cohen–Daubechies–Feauveau 9/7
 * filter bank `h[]`), NOT the commented-out QccPack literals — SPERR uses the
 * former, so only the former reproduces its coefficients. Matching SPERR
 * exactly matters because the downstream SPECK bit-plane coder is defined on
 * *these* coefficients; any drift here is invisible until SPECK fails to
 * validate against reference SPERR.
 *
 * The GPU parallelization (shared-memory line, clamped-index boundaries) follows
 * the JPEG2000 lifting-DWT literature:
 *   - J. Matela, "GPU-Based DWT Acceleration for JPEG2000" (2009).
 *   - W. J. van der Laan et al., "Accelerating wavelet-based video coding on
 *     GPU using CUDA."
 *
 * ### Precision policy
 *
 * `T = double` is the bit-exact path (SPERR does the whole transform in
 * `double`). `T = float` is our own faster/lossier variant: constants are still
 * derived in `double` and cast to `float`, but the sweep arithmetic is `float`,
 * so it is deliberately NOT bit-exact with SPERR. See docs — cf. the f64->f32
 * narrowing bug class we have hit before.
 */

#include <cstddef>
#ifndef __CUDA_ARCH__
  #include <cmath>
  #include <algorithm>
#endif

namespace fz {
namespace cdf97 {

/// The six CDF 9/7 lifting constants, derived exactly as SPERR derives them.
struct Constants {
    double alpha, beta, gamma, delta, epsilon, inv_epsilon;
};

/**
 * Compute the lifting constants in `double`, matching SPERR's `include/CDF97.h`
 * expression-for-expression (same operation order => same rounding). The
 * compiler folds this to constants at the call site; do not "optimize" the
 * arithmetic ordering or it stops matching SPERR.
 */
__host__ __device__ inline Constants constants()
{
    // CDF 9/7 filter-bank coefficients (SPERR: `const std::array<double,5> h`).
    const double h0 =  0.602949018236;
    const double h1 =  0.266864118443;
    const double h2 = -0.078223266529;
    const double h3 = -0.016864118443;
    const double h4 =  0.026748757411;

    const double r0 = h0 - 2.0 * h4 * h1 / h3;
    const double r1 = h2 - h4 - h4 * h1 / h3;
    const double s0 = h1 - h3 - h3 * r0 / r1;
    const double t0 = h0 - 2.0 * (h2 - h4);

    Constants c;
    c.alpha = h4 / h3;
    c.beta  = h3 / r1;
    c.gamma = r1 / s0;
    c.delta = s0 / t0;
#ifdef __CUDA_ARCH__
    c.epsilon = sqrt(2.0) * t0;          // IEEE-correct double sqrt on device
#else
    c.epsilon = std::sqrt(2.0) * t0;
#endif
    c.inv_epsilon = 1.0 / c.epsilon;
    return c;
}

// ── Index clamps ───────────────────────────────────────────────────────────
// SPERR special-cases the first/last element of every lifting step across four
// length-parity variants (even/odd length x even/odd step). Each of those cases
// is algebraically identical to the interior formula once the out-of-range
// neighbour is *clamped* to the boundary (whole-point symmetric reflection).
// Verified equivalent for all parities of `len`. Clamping makes every thread run
// one branch-free expression — the whole reason this maps cleanly to a warp.
__host__ __device__ inline int clamp_hi(int i, int hi) { return i < hi ? i : hi; }
__host__ __device__ inline int clamp_lo(int i)         { return i > 0  ? i : 0;  }

/**
 * Cooperative forward CDF 9/7 analysis of ONE line, in place.
 *
 * @param s   Length-`len` line, already **de-interleaved** into
 *            `[ even_len | odd_len ]` layout (the gather is done by the caller
 *            during the shared-memory load — see cdf97_kernels). On return the
 *            same buffer holds `[ low-subband | high-subband ]`, i.e. it is
 *            already in dyadic subband order for the next level's recursion.
 * @param len Line length (>= 2; `len < 2` is a no-op, matching SPERR's level cap
 *            which never recurses below length 2).
 * @param t   Cooperative thread index for this line (usually `threadIdx.x`, but
 *            not always — a caller processing several lines per block with a
 *            2-D thread shape passes a computed flat index; see
 *            `cdf97_axis_kernel_tiled` in cdf97_kernels.cuh).
 * @param nt  Number of threads cooperating (usually `blockDim.x`, same caveat).
 *
 * All `nt` threads stride over the line; the `__syncthreads()` between lifting
 * steps are the only barriers required (each step's writes are to a set
 * disjoint from its reads). Every thread passing the same `(len, nt)` must call
 * this together — `__syncthreads()` is block-scoped, so `t`/`nt` may safely
 * differ from raw `threadIdx.x`/`blockDim.x` as long as the whole block agrees
 * on `nt` and every value in `[0,nt)` is covered by some thread's `t`.
 */
template <typename T>
__device__ inline void analysis_line(T* s, int len, int t, int nt)
{
    if (len < 2) return;
    const Constants c = constants();
    const T ALPHA = (T)c.alpha, BETA  = (T)c.beta,  GAMMA = (T)c.gamma;
    const T DELTA = (T)c.delta, EPS   = (T)c.epsilon, IEPS = (T)c.inv_epsilon;

    const int even_len = len - len / 2;
    const int odd_len  = len / 2;
    T* even = s;
    T* odd  = s + even_len;

    // 1) predict (ALPHA):  odd += ALPHA * (even[i] + even[i+1])
    for (int i = t; i < odd_len; i += nt)
        odd[i] += ALPHA * (even[i] + even[clamp_hi(i + 1, even_len - 1)]);
    __syncthreads();
    // 2) update (BETA):    even += BETA * (odd[j-1] + odd[j])
    for (int j = t; j < even_len; j += nt)
        even[j] += BETA * (odd[clamp_lo(j - 1)] + odd[clamp_hi(j, odd_len - 1)]);
    __syncthreads();
    // 3) predict (GAMMA)
    for (int i = t; i < odd_len; i += nt)
        odd[i] += GAMMA * (even[i] + even[clamp_hi(i + 1, even_len - 1)]);
    __syncthreads();
    // 4) update + low-band scale (DELTA, EPSILON)
    for (int j = t; j < even_len; j += nt)
        even[j] = EPS * (even[j] + DELTA * (odd[clamp_lo(j - 1)] + odd[clamp_hi(j, odd_len - 1)]));
    __syncthreads();
    // 5) high-band scale
    for (int i = t; i < odd_len; i += nt)
        odd[i] *= -IEPS;
    __syncthreads();
}

/**
 * Cooperative inverse (synthesis) of ONE line, in place — exact algebraic
 * mirror of analysis_line(). Input: `[ low | high ]` subband order; output:
 * `[ even_len | odd_len ]` de-interleaved sample values (the caller scatters
 * back to natural order on write-out).
 *
 * `t`/`nt`: see analysis_line() — usually `threadIdx.x`/`blockDim.x`.
 */
template <typename T>
__device__ inline void synthesis_line(T* s, int len, int t, int nt)
{
    if (len < 2) return;
    const Constants c = constants();
    const T ALPHA = (T)c.alpha, BETA  = (T)c.beta,  GAMMA = (T)c.gamma;
    const T DELTA = (T)c.delta, EPS   = (T)c.epsilon, IEPS = (T)c.inv_epsilon;

    const int even_len = len - len / 2;
    const int odd_len  = len / 2;
    T* even = s;
    T* odd  = s + even_len;

    // undo 5) high-band scale
    for (int i = t; i < odd_len; i += nt)
        odd[i] *= -EPS;
    __syncthreads();
    // undo 4) update + low-band scale
    for (int j = t; j < even_len; j += nt)
        even[j] = IEPS * even[j] - DELTA * (odd[clamp_lo(j - 1)] + odd[clamp_hi(j, odd_len - 1)]);
    __syncthreads();
    // undo 3) predict (GAMMA)
    for (int i = t; i < odd_len; i += nt)
        odd[i] -= GAMMA * (even[i] + even[clamp_hi(i + 1, even_len - 1)]);
    __syncthreads();
    // undo 2) update (BETA)
    for (int j = t; j < even_len; j += nt)
        even[j] -= BETA * (odd[clamp_lo(j - 1)] + odd[clamp_hi(j, odd_len - 1)]);
    __syncthreads();
    // undo 1) predict (ALPHA)
    for (int i = t; i < odd_len; i += nt)
        odd[i] -= ALPHA * (even[i] + even[clamp_hi(i + 1, even_len - 1)]);
    __syncthreads();
}

#ifndef __CUDA_ARCH__
// ── Host reference (serial) ────────────────────────────────────────────────
// Faithful serial transcription of SPERR's QccWAVCDF97AnalysisSymmetric, kept
// in the *literal* boundary-special-cased form (not the clamped form) so it is
// an independent oracle: a unit test that runs this and the device kernel on
// the same input and diffs the results proves the clamp rewrite is correct.
// Operates on de-interleaved `[ even_len | odd_len ]` layout, like the device fn.
template <typename T>
inline void host_analysis_line(T* s, size_t len)
{
    if (len < 2) return;
    const Constants c = constants();
    const size_t even_len = len - len / 2;
    const size_t odd_len  = len / 2;
    T* even = s;
    T* odd  = s + even_len;

    for (size_t i = 0; i + 1 < odd_len; ++i)
        odd[i] += (T)c.alpha * (even[i] + even[i + 1]);
    odd[odd_len - 1] += (T)c.alpha * (even[odd_len - 1] + even[even_len - 1]);

    even[0] += (T)2.0 * (T)c.beta * odd[0];
    for (size_t i = 1; i + 1 < even_len; ++i)
        even[i] += (T)c.beta * (odd[i - 1] + odd[i]);
    even[even_len - 1] += (T)c.beta * (odd[even_len - 2] + odd[odd_len - 1]);

    for (size_t i = 0; i + 1 < odd_len; ++i)
        odd[i] += (T)c.gamma * (even[i] + even[i + 1]);
    odd[odd_len - 1] += (T)c.gamma * (even[odd_len - 1] + even[even_len - 1]);

    even[0] = (T)c.epsilon * (even[0] + (T)2.0 * (T)c.delta * odd[0]);
    for (size_t i = 1; i + 1 < even_len; ++i)
        even[i] = (T)c.epsilon * (even[i] + (T)c.delta * (odd[i - 1] + odd[i]));
    even[even_len - 1] =
        (T)c.epsilon * (even[even_len - 1] + (T)c.delta * (odd[even_len - 2] + odd[odd_len - 1]));

    for (size_t i = 0; i < odd_len; ++i)
        odd[i] *= -(T)c.inv_epsilon;
}
#endif  // !__CUDA_ARCH__

}  // namespace cdf97
}  // namespace fz
