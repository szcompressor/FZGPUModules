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
 * ROCm's HIP declares `atomicAdd_block`/`atomicOr_block` with identical names
 * and signatures to CUDA's (`hip/amd_detail/amd_hip_atomic.h`) — unlike the
 * warp shuffle/ballot family (see warp.h), there is no warp/wavefront-width
 * dependency in a block-scope atomic, so naively this should "just hipify".
 * That said, this codebase has been burned before by CUDA intrinsics that
 * looked source-portable but silently diverged under HIP (see warp.h's
 * `__shfl_down_sync` width story), and — unlike every other intrinsic family
 * touched during the HIP port — these have **not yet been exercised or
 * verified on real AMD hardware** (no existing call site in this codebase
 * uses any block-scoped atomic). This header exists so there is exactly one
 * place to patch if that verification turns up a divergence, rather than
 * scattering raw `atomicAdd_block`/`atomicOr_block` calls through the RARE/RAZE
 * kernels. Treat the HIP branch below as "expected to work, not yet confirmed
 * on hardware" until it has been built and run on the project's MI100 target.
 */

#include "backend/api.h"

namespace fz {
namespace backend {

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

} // namespace backend
} // namespace fz
