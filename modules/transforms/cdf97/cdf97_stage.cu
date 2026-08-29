// Adapted from the SPERR project (Li, Lindstrom, Clyne — NCAR), Apache
// License 2.0 — see THIRD_PARTY.md.
// Original: include/CDF97.h, src/CDF97.cpp in https://github.com/NCAR/SPERR

/**
 * @file cdf97_stage.cu
 * @brief Cdf97Stage::execute() — drives the CDF 9/7 DWT kernels.
 */

#include "transforms/cdf97/cdf97_stage.h"
#include "transforms/cdf97/cdf97_kernels.cuh"
#include "mem/mempool.h"
#include "cuda_check.h"
#include "backend/api.h"
#include <algorithm>
#include <stdexcept>
#include <string>

namespace fz {

template <typename TInput>
void Cdf97Stage<TInput>::execute(cudaStream_t stream, MemoryPool* /*pool*/,
                                 const std::vector<void*>& inputs,
                                 const std::vector<void*>& outputs,
                                 const std::vector<size_t>& sizes)
{
    if (inputs.empty() || outputs.empty() || sizes.empty())
        throw std::runtime_error("Cdf97Stage: inputs, outputs, and sizes must be non-empty");

    const size_t bytes = sizes[0];
    const size_t n     = bytes / sizeof(TInput);
    if (n == 0) { actual_output_size_ = 0; return; }

    if (dims_[0] == 0)
        throw std::runtime_error("Cdf97Stage: dimensions not set — call setDims() before compress");

    const size_t nx = dims_[0], ny = dims_[1], nz = dims_[2];
    if (nx * ny * nz != n)
        throw std::runtime_error(
            "Cdf97Stage: dims (" + std::to_string(nx) + "x" + std::to_string(ny) + "x" +
            std::to_string(nz) + ") do not match element count " + std::to_string(n));

    const size_t maxdim = std::max({nx, ny, nz});
    if (maxdim > kMaxLineElems)
        throw std::runtime_error(
            "Cdf97Stage: largest dimension " + std::to_string(maxdim) +
            " exceeds the shared-memory line limit (" + std::to_string(kMaxLineElems) +
            " elements); long-line fallback not yet implemented");

    // The lifting sweep is in place; run it on the output buffer.
    if (outputs[0] != inputs[0])
        FZ_CUDA_CHECK(cudaMemcpyAsync(outputs[0], inputs[0], bytes,
                                      cudaMemcpyDeviceToDevice, stream));

    TInput* d = static_cast<TInput*>(outputs[0]);
    switch (ndim()) {
        case 1: cdf97::dwt1d<TInput>(d, (int)nx, is_inverse_, stream); break;
        case 2: cdf97::dwt2d<TInput>(d, (int)nx, (int)ny, is_inverse_, stream); break;
        default: cdf97::dwt3d<TInput>(d, (int)nx, (int)ny, (int)nz, is_inverse_, stream); break;
    }
    FZ_CUDA_CHECK(cudaGetLastError());

    actual_output_size_ = bytes;
}

template class Cdf97Stage<float>;
template class Cdf97Stage<double>;

} // namespace fz
