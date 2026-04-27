#pragma once

/**
 * Shared utilities for predictor stages (Lorenzo, Quantizer, etc.)
 *
 * Included by .cu files only — not a public API header.
 *
 * Contents:
 *  - minmax_partial_kernel   : two-level device min/max reduction
 *  - computeValueBase        : host-side launcher → returns value_range (NOA)
 *                              or max(|data|) (REL)
 *  - scatter_add_kernel      : scatter (outlier_value += existing) for Lorenzo
 *  - scatter_assign_kernel   : scatter (outlier_value = original) for Quantizer
 */

#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include "predictors/lorenzo_quant/lorenzo_quant.h"  // for ErrorBoundMode
#include "mem/mempool.h"
#include "cuda_check.h"

namespace fz {

// ===== Min/Max Two-Level Reduction =====

template<typename T>
__global__ void minmax_partial_kernel(
    const T* __restrict__ data, size_t n,
    T* __restrict__ partial_min, T* __restrict__ partial_max
) {
    extern __shared__ char smem[];
    T* s_min = reinterpret_cast<T*>(smem);
    T* s_max = s_min + blockDim.x;

    int tid    = static_cast<int>(threadIdx.x);
    int stride = static_cast<int>(blockDim.x) * static_cast<int>(gridDim.x);
    int gid    = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + tid;

    T tmin = (gid < static_cast<int>(n)) ? data[gid] : data[0];
    T tmax = tmin;
    for (int i = gid + stride; i < static_cast<int>(n); i += stride) {
        T v = data[i];
        if (v < tmin) tmin = v;
        if (v > tmax) tmax = v;
    }
    s_min[tid] = tmin;
    s_max[tid] = tmax;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_min[tid + s] < s_min[tid]) s_min[tid] = s_min[tid + s];
            if (s_max[tid + s] > s_max[tid]) s_max[tid] = s_max[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        partial_min[blockIdx.x] = s_min[0];
        partial_max[blockIdx.x] = s_max[0];
    }
}

/**
 * Compute the scaling denominator for NOA or REL error bound conversion.
 *
 *   NOA:  returns  max(data) - min(data)
 *   REL:  returns  max(|data|)          (only approximates per-element REL for
 *                                        Lorenzo; QuantizerStage REL is exact)
 *
 * Synchronises the stream and performs a small D2H copy.  Latency is
 * proportional to min(n, 1024) blocks, typically a few µs.
 */
template<typename TInput>
inline float computeValueBase(
    const TInput* d_data, size_t n,
    ErrorBoundMode mode,
    cudaStream_t stream, MemoryPool* pool
) {
    constexpr int kBlockSize = 256;
    int num_blocks = static_cast<int>(std::min(
        (n + kBlockSize - 1) / kBlockSize, static_cast<size_t>(1024)));

    TInput* d_pmin = static_cast<TInput*>(
        pool->allocate(num_blocks * sizeof(TInput), stream, "eb_scan_min", false));
    TInput* d_pmax = static_cast<TInput*>(
        pool->allocate(num_blocks * sizeof(TInput), stream, "eb_scan_max", false));

    size_t smem = 2 * kBlockSize * sizeof(TInput);
    minmax_partial_kernel<TInput><<<num_blocks, kBlockSize, smem, stream>>>(
        d_data, n, d_pmin, d_pmax);

    FZ_CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<TInput> h_pmin(num_blocks), h_pmax(num_blocks);
    FZ_CUDA_CHECK(cudaMemcpy(h_pmin.data(), d_pmin,
                             num_blocks * sizeof(TInput), cudaMemcpyDeviceToHost));
    FZ_CUDA_CHECK(cudaMemcpy(h_pmax.data(), d_pmax,
                             num_blocks * sizeof(TInput), cudaMemcpyDeviceToHost));

    pool->free(d_pmin, 0);
    pool->free(d_pmax, 0);

    TInput gmin = h_pmin[0], gmax = h_pmax[0];
    for (int i = 1; i < num_blocks; i++) {
        if (h_pmin[i] < gmin) gmin = h_pmin[i];
        if (h_pmax[i] > gmax) gmax = h_pmax[i];
    }

    if (mode == ErrorBoundMode::NOA) {
        return static_cast<float>(gmax - gmin);
    } else {
        TInput abs_min = (gmin < TInput(0)) ? -gmin : gmin;
        TInput abs_max = (gmax < TInput(0)) ? -gmax : gmax;
        return static_cast<float>(abs_min > abs_max ? abs_min : abs_max);
    }
}

// ===== Scatter Kernels =====

/**
 * scatter_add_kernel — Lorenzo-style scatter.
 *
 * Adds each outlier's stored prediction error onto whatever the prefix-sum
 * reconstruction left at that position.  Reads the actual count from a device
 * pointer so no D2H sync is needed at launch time.
 */
template<typename T>
__global__ void scatter_add_kernel(
    const T* __restrict__        outlier_values,
    const uint32_t* __restrict__ outlier_indices,
    const uint32_t* __restrict__ outlier_count_ptr,
    T* __restrict__              output
) {
    uint32_t outlier_count = *outlier_count_ptr;
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < outlier_count) {
        output[outlier_indices[tid]] += outlier_values[tid];
    }
}

/**
 * scatter_assign_kernel — Quantizer-style scatter.
 *
 * Writes each outlier's original value directly to its position (replaces the
 * placeholder 0 left by the dequantization kernel).
 */
template<typename T>
__global__ void scatter_assign_kernel(
    const T* __restrict__        outlier_values,
    const uint32_t* __restrict__ outlier_indices,
    const uint32_t* __restrict__ outlier_count_ptr,
    T* __restrict__              output
) {
    uint32_t outlier_count = *outlier_count_ptr;
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < outlier_count) {
        output[outlier_indices[tid]] = outlier_values[tid];
    }
}

} // namespace fz
