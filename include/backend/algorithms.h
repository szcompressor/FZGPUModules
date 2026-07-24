#pragma once

/**
 * @file include/backend/algorithms.h
 * @brief Backend-neutral wrappers for device-wide scan/reduce/sort algorithms.
 *
 * FZGPUModules stages lean on cub's Device-level primitives (DeviceScan,
 * DeviceReduce, DeviceRadixSort), all of which share the same two-call shape:
 * query the required scratch bytes, allocate scratch (preferring the
 * pipeline's MemoryPool, falling back to a raw device allocation), then run
 * the real call. `withTempStorage()` below collapses that repeated
 * query/allocate/fallback dance into one call; the actual algorithm
 * invocation (still `cub::Device*` under the CUDA backend) stays at the call
 * site via a lambda, so this header does not need to know about cub itself.
 *
 * `exclusiveScan()` wraps the two ADM-stage call sites that use
 * `thrust::exclusive_scan` directly (no scratch-buffer dance — thrust manages
 * its own temporary storage internally).
 *
 * CUDA and HIP are implemented. The two share one implementation: the
 * allocation calls below are spelled with their CUDA names and re-pointed at
 * HIP by backend/api.h, and hipCUB mirrors cub's API closely enough that the
 * `cub::Device*` calls left at the call sites are a namespace swap. The only
 * genuine divergence is thrust's execution policy (`thrust::cuda::par` vs
 * rocThrust's `thrust::hip::par`), isolated in `detail::parOn()` below.
 *
 * SYCL is added when that backend lands (oneDPL for exclusiveScan(), since
 * oneDPL has no direct cub::Device* analogue).
 */

#include "backend/api.h"
#include "backend/types.h"
#include "cuda_check.h"
#include "mem/mempool.h"

#if defined(FZGMOD_BACKEND_SYCL)

#error "FZGMOD_BACKEND=SYCL is not implemented yet"

#endif

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#include <cstddef>

namespace fz {
namespace backend {

namespace detail {

/** The active backend's stream-bound thrust execution policy. */
inline auto parOn(fz::stream_t stream) {
#if defined(FZGMOD_BACKEND_HIP)
    return thrust::hip::par.on(stream);
#else
    return thrust::cuda::par.on(stream);
#endif
}

} // namespace detail

/**
 * Scratch pointer returned by withTempStorage(), tagged with where it came
 * from so freeTempStorage() routes to the matching free — tracking this
 * explicitly (rather than inferring it from `pool != nullptr` at free time)
 * matters because `pool` can be non-null yet still have fallen back to a raw
 * allocation, if `pool->allocate()` itself returned null (e.g. pool
 * exhaustion) rather than the pool being absent altogether.
 */
struct TempStorage {
    void* ptr       = nullptr;
    bool  from_pool = false;
};

/**
 * Runs a cub-shaped two-call device algorithm: `fn(nullptr, bytes)` sizes the
 * scratch buffer, `bytes` of scratch is allocated (via `pool` if non-null and
 * it succeeds, else a raw device allocation), then `fn(d_temp, bytes)` runs
 * the real call.
 *
 * Returns the scratch pointer plus its provenance — freeing it via
 * freeTempStorage() is the caller's responsibility, since call sites differ
 * on when it's safe to free (some free immediately after the call, some
 * defer until after a downstream kernel that also reads scratch-adjacent
 * output).
 */
template<typename Fn>
TempStorage withTempStorage(MemoryPool* pool, fz::stream_t stream, const char* tag, Fn&& fn) {
    size_t bytes = 0;
    fn(nullptr, bytes);
    void* d_temp    = pool ? pool->allocate(bytes, stream, tag) : nullptr;
    bool  from_pool = (d_temp != nullptr);
    if (!d_temp && bytes > 0) {
        FZ_CUDA_CHECK(cudaMalloc(&d_temp, bytes));
        from_pool = false;
    }
    fn(d_temp, bytes);
    return TempStorage{d_temp, from_pool};
}

/** Frees storage returned by withTempStorage(), routing to the pool or a raw free as appropriate. */
inline void freeTempStorage(MemoryPool* pool, const TempStorage& temp, fz::stream_t stream) {
    if (!temp.ptr) return;
    if (temp.from_pool && pool) {
        pool->free(temp.ptr, stream);
    } else {
        FZ_CUDA_CHECK_WARN(cudaStreamSynchronize(stream));
        FZ_CUDA_CHECK_WARN(cudaFree(temp.ptr));
    }
}

/** Exclusive prefix sum over `n` elements. No scratch-buffer management needed. */
template<typename T>
void exclusiveScan(fz::stream_t stream, T* d_in, T* d_out, size_t n) {
    thrust::device_ptr<T> in_ptr(d_in);
    thrust::device_ptr<T> out_ptr(d_out);
    thrust::exclusive_scan(detail::parOn(stream), in_ptr, in_ptr + n, out_ptr);
}

} // namespace backend
} // namespace fz
