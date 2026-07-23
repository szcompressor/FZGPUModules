#pragma once

#include <stdexcept>
#include <string>

#include "backend/types.h"
#include "log.h"

/**
 * @file include/cuda_check.h
 * @brief GPU backend API error-checking macros.
 *
 * Despite the "CUDA" name (kept for call-site stability — this macro is used
 * at ~30 call sites across the codebase), the implementation is backend-neutral:
 * it checks against `fz::error_t`/`fz::kBackendSuccess`, which alias to the
 * active backend's native error type (see include/backend/types.h).
 *
 * FZ_CUDA_CHECK(call)
 *   Evaluates a CUDA API call and throws std::runtime_error on failure.
 *   Use on every CUDA API call in normal code paths — the overhead is a
 *   single integer comparison, which the CPU branch predictor predicts
 *   correctly 100% of the time in the success case.
 *
 *   Example:
 *     FZ_CUDA_CHECK(cudaMalloc(&ptr, size));
 *     FZ_CUDA_CHECK(cudaStreamSynchronize(stream));
 *
 * FZ_CUDA_CHECK_WARN(call)
 *   Evaluates a CUDA API call and emits a FZ_LOG(WARN) on failure.
 *   Does NOT throw. Use in destructors, cleanup/teardown paths, and
 *   resource-release calls where throwing would cause additional harm
 *   (e.g. cudaEventDestroy, cudaStreamDestroy, cudaFreeAsync in reset()).
 *
 *   Example:
 *     FZ_CUDA_CHECK_WARN(cudaEventDestroy(event));
 *     FZ_CUDA_CHECK_WARN(cudaFreeAsync(ptr, stream));
 */

#define FZ_CUDA_CHECK(call)                                                      \
    do {                                                                         \
        fz::error_t _fz_cuda_err_ = (call);                                      \
        if (_fz_cuda_err_ != fz::kBackendSuccess) {                              \
            throw std::runtime_error(                                            \
                std::string("[fzgmod] CUDA error at " __FILE__ ":") +             \
                std::to_string(__LINE__) +                                       \
                " — " #call " → " +                                              \
                fz::getBackendErrorString(_fz_cuda_err_));                       \
        }                                                                        \
    } while (0)

#define FZ_CUDA_CHECK_WARN(call)                                                 \
    do {                                                                         \
        fz::error_t _fz_cuda_err_ = (call);                                      \
        if (_fz_cuda_err_ != fz::kBackendSuccess) {                              \
            FZ_LOG(WARN,                                                         \
                   "CUDA error at %s:%d — " #call " → %s",                      \
                   __FILE__, __LINE__,                                           \
                   fz::getBackendErrorString(_fz_cuda_err_));                    \
        }                                                                        \
    } while (0)
