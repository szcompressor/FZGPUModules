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
 * CUDA and HIP are implemented; the SYCL branch is added when that backend
 * lands.
 */

// Pulls in the active backend's runtime headers, and (under HIP) re-points the
// CUDA-spelled host API onto its HIP equivalents.
#include "backend/api.h"

#if defined(FZGMOD_BACKEND_HIP)

#include <cstddef>
#include <cstdint>

namespace fz {

using stream_t     = hipStream_t;
using event_t      = hipEvent_t;
using mempool_t    = hipMemPool_t;
using error_t      = hipError_t;
using graph_t      = hipGraph_t;
using graph_exec_t = hipGraphExec_t;

inline constexpr error_t kBackendSuccess = hipSuccess;

/** True for backends with a mature CUDA-Graph-equivalent capture API (CUDA, HIP). */
inline constexpr bool kBackendSupportsGraphCapture = true;

/** Returns a human-readable description of a backend error code. */
inline const char* getBackendErrorString(error_t err) {
    return hipGetErrorString(err);
}

/** Current live bytes in `pool` (0 if `pool` is null). */
inline size_t getPoolUsedMemCurrent(mempool_t pool) {
    if (!pool) return 0;
    uint64_t used = 0;
    hipMemPoolGetAttribute(pool, hipMemPoolAttrUsedMemCurrent, &used);
    return static_cast<size_t>(used);
}

/** Peak live bytes in `pool` since the attribute was last reset (0 if `pool` is null). */
inline size_t getPoolUsedMemHigh(mempool_t pool) {
    if (!pool) return 0;
    uint64_t high = 0;
    hipMemPoolGetAttribute(pool, hipMemPoolAttrUsedMemHigh, &high);
    return static_cast<size_t>(high);
}

} // namespace fz

#elif defined(FZGMOD_BACKEND_SYCL)

#error "FZGMOD_BACKEND=SYCL is not implemented yet"

#else // FZGMOD_BACKEND_CUDA (default)

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
