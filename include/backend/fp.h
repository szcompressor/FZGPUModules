#pragma once

/**
 * @file include/backend/fp.h
 * @brief Floating-point operations whose rounding must not depend on the
 *        compiler's contraction policy.
 *
 * Dependency-free (no runtime headers): also compiled by NVRTC.
 */

namespace fz {
namespace backend {

/**
 * a * b rounded to float on its own, never fused into the operation that
 * consumes it. An error-bound check `|x - q*step| <= eb` must test the product
 * the decoder will store, which is the rounded one. nvcc honors __fmul_rn;
 * HIP's __fmul_rn is a plain multiply that clang's default
 * -ffp-contract=fast-honor-pragmas fuses with a following subtraction into one
 * FMA (gfx90a: v_fma_f32), which tests the exact product and passes values
 * whose stored reconstruction misses the bound by up to half an ulp.
 */
__device__ __forceinline__ float fmulRounded(float a, float b) {
#if defined(FZGMOD_BACKEND_HIP)
    #pragma clang fp contract(off)
    return a * b;
#else
    return __fmul_rn(a, b);
#endif
}

/** The same for a double: the product the decoder stores, never fused into the operation that consumes it. */
__device__ __forceinline__ double fmulRounded(double a, double b) {
#if defined(FZGMOD_BACKEND_HIP)
    #pragma clang fp contract(off)
    return a * b;
#else
    return __dmul_rn(a, b);
#endif
}

} // namespace backend
} // namespace fz
