#pragma once

/**
 * @file include/backend/atomics.h
 * @brief Backend-neutral block-scoped atomics.
 *
 * `atomicAdd_block`/`atomicOr_block` (CUDA's block-memory-scope atomic family,
 * as opposed to the default device-scope `atomicAdd`/`atomicOr`) are used by
 * the RARE/RAZE device code (`modules/coders/lc_common/lc_chunk_components.cuh`)
 * for a per-block histogram accumulation and for OR-ing a straddling partial
 * word into shared memory — both cases where every participating thread is
 * known to be in the same block, so scoping the atomic down from "device"
 * saves the extra memory-fence cost of a full device-wide atomic.
 *
 * The `_block` suffix family is CUDA-only spelling. Contrary to this header's
 * original expectation ("ROCm declares them with identical names"), ROCm 6.4.1
 * declares neither `atomicAdd_block` nor `atomicOr_block` anywhere under
 * `hip/` — verified by grep against the installed toolchain, and by the
 * `use of undeclared identifier 'atomicAdd_block'` error the first HIP build
 * of the RARE/RAZE stages produced. This is exactly the divergence the header
 * was created to absorb, so the patch lands here and the call sites are
 * untouched.
 *
 * HIP's equivalent is the scoped-atomic builtin family
 * (`__hip_atomic_fetch_add`/`_or` + `__HIP_MEMORY_SCOPE_WORKGROUP`), where
 * "workgroup" is AMD's name for a CUDA block. Memory ordering is
 * `__ATOMIC_RELAXED` to match CUDA's `atomic*_block`, which carry no ordering
 * guarantees beyond the atomicity of the read-modify-write itself — the RARE/
 * RAZE call sites (per-block histogram accumulation; OR-ing a straddling
 * partial word into shared memory) separately `__syncthreads()` before reading
 * the accumulated result, so they rely on that barrier for ordering, not on
 * the atomic.
 *
 * Unlike the warp shuffle/ballot family (see warp.h), a block-scoped atomic has
 * no wavefront-width dependency, so there is no 32-vs-64-lane subtlety here.
 */

#include "backend/api.h"

namespace fz {
namespace backend {

#if defined(FZGMOD_BACKEND_HIP)

/** Block-scoped atomic add. `T` must be an arithmetic type the HIP scoped-atomic builtin accepts (int, unsigned, unsigned long long, float, double). */
template <typename T>
__device__ inline T atomicAddBlock(T* addr, T val) {
    return __hip_atomic_fetch_add(addr, val, __ATOMIC_RELAXED,
                                  __HIP_MEMORY_SCOPE_WORKGROUP);
}

/** Block-scoped atomic bitwise-OR. `T` must be an integral type (int, unsigned, unsigned long long). */
template <typename T>
__device__ inline T atomicOrBlock(T* addr, T val) {
    return __hip_atomic_fetch_or(addr, val, __ATOMIC_RELAXED,
                                 __HIP_MEMORY_SCOPE_WORKGROUP);
}

/** Block-scoped atomic max. `T` must be an integral type (int, unsigned, unsigned long long). */
template <typename T>
__device__ inline T atomicMaxBlock(T* addr, T val) {
    return __hip_atomic_fetch_max(addr, val, __ATOMIC_RELAXED,
                                  __HIP_MEMORY_SCOPE_WORKGROUP);
}

#else // CUDA

/** Block-scoped atomic add. `T` must be one of the types CUDA's `atomicAdd_block` overloads on (int, unsigned, unsigned long long, float, double). */
template <typename T>
__device__ inline T atomicAddBlock(T* addr, T val) {
    return atomicAdd_block(addr, val);
}

/** Block-scoped atomic bitwise-OR. `T` must be one of the types CUDA's `atomicOr_block` overloads on (int, unsigned, unsigned long long). */
template <typename T>
__device__ inline T atomicOrBlock(T* addr, T val) {
    return atomicOr_block(addr, val);
}

/** Block-scoped atomic max. `T` must be one of the types CUDA's `atomicMax_block` overloads on (int, unsigned, unsigned long long). */
template <typename T>
__device__ inline T atomicMaxBlock(T* addr, T val) {
    return atomicMax_block(addr, val);
}

#endif

} // namespace backend
} // namespace fz
