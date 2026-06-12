#pragma once
/**
 * fused/lorenzo_quant/lorenzo_quant_kernels.cuh
 *
 * Internal CUDA device/global kernel primitives shared between
 * lorenzo.cu (1-D) and lorenzo_quant_nd.cu (2-D / 3-D).
 *
 * Not part of the public API — include only from .cu files in this directory.
 */

#include <cuda_runtime.h>
#include <cstdint>

namespace fz {

/**
 * Scatter outlier prediction errors back to their original positions.
 *
 * Used by all inverse Lorenzo launchers (1-D, 2-D, 3-D) before the
 * prefix-sum / dequantize kernel runs.  The count `n` is supplied by the
 * host (read from the deserialized FZM stage header) and passed as a
 * register-resident kernel argument — the kernel never loads it from
 * device memory.
 */
template<typename T>
__global__ void scatter_outliers_kernel(
    const T* __restrict__ outlier_values,
    const uint32_t* __restrict__ outlier_indices,
    uint32_t n,
    T* __restrict__ output
) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n) {
        uint32_t idx = outlier_indices[tid];
        // Write the outlier prediction error in quantized units
        // The prefix sum kernel will read this and incorporate it into the sum
        output[idx] = outlier_values[tid];
    }
}

} // namespace fz
