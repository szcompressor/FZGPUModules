#pragma once

/**
 * @file include/backend/types.h
 * @brief Backend-neutral GPU type aliases.
 *
 * Every stage/pipeline/pool header spells the GPU stream/event/pool handle
 * types as `fz::stream_t`/`fz::event_t`/`fz::mempool_t` instead of naming
 * `cudaStream_t` etc. directly. Exactly one backend is compiled into any
 * given build (selected at configure time via the `FZGMOD_BACKEND` CMake
 * option, which defines one of `FZGMOD_BACKEND_CUDA`/`_HIP`/`_SYCL`) — this
 * is a compile-time alias, not a runtime abstraction, so it costs nothing.
 *
 * Only the CUDA branch is implemented today; HIP/SYCL branches are added
 * when those backends land.
 */

#if defined(FZGMOD_BACKEND_HIP)

#error "FZGMOD_BACKEND=HIP is not implemented yet"

#elif defined(FZGMOD_BACKEND_SYCL)

#error "FZGMOD_BACKEND=SYCL is not implemented yet"

#else // FZGMOD_BACKEND_CUDA (default)

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace fz {

using stream_t     = cudaStream_t;
using event_t      = cudaEvent_t;
using mempool_t    = cudaMemPool_t;
using error_t      = cudaError_t;
using graph_t      = cudaGraph_t;
using graph_exec_t = cudaGraphExec_t;

inline constexpr error_t kBackendSuccess = cudaSuccess;

/** True for backends with a mature CUDA-Graph-equivalent capture API (CUDA, HIP). */
inline constexpr bool kBackendSupportsGraphCapture = true;

/** Returns a human-readable description of a backend error code. */
inline const char* getBackendErrorString(error_t err) {
    return cudaGetErrorString(err);
}

/** Current live bytes in `pool` (0 if `pool` is null). */
inline size_t getPoolUsedMemCurrent(mempool_t pool) {
    if (!pool) return 0;
    uint64_t used = 0;
    cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemCurrent, &used);
    return static_cast<size_t>(used);
}

/** Peak live bytes in `pool` since the attribute was last reset (0 if `pool` is null). */
inline size_t getPoolUsedMemHigh(mempool_t pool) {
    if (!pool) return 0;
    uint64_t high = 0;
    cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemHigh, &high);
    return static_cast<size_t>(high);
}

} // namespace fz

#endif
