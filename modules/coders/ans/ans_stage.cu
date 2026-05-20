#include "coders/ans/ans_stage.h"
#include "mem/mempool.h"
#include "cuda_check.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace fz {

// ── Device kernels ─────────────────────────────────────────────────────────────
// TODO: add __global__ kernel(s) here.
// All kernels must be launched on the provided stream — never call
// cudaDeviceSynchronize() inside execute().

// ── ANSStage::execute ──────────────────────────────────────────────────────
void ANSStage::execute(
    cudaStream_t stream,
    MemoryPool* pool,
    const std::vector<void*>& inputs,
    const std::vector<void*>& outputs,
    const std::vector<size_t>& sizes)
{
    (void)pool; // remove if you use pool->allocate()

    if (inputs.empty() || outputs.empty() || sizes.empty())
        throw std::runtime_error(
            "ANSStage: inputs, outputs, and sizes must be non-empty");

    const size_t byte_size = sizes[0];
    if (byte_size == 0) {
        actual_output_size_ = 0;
        return;
    }

    // TODO: launch kernel(s). Example pattern:
    //   constexpr int kBlock = 256;
    //   const int grid = static_cast<int>((n + kBlock - 1) / kBlock);
    //   myKernel<<<grid, kBlock, 0, stream>>>(...);
    //   FZ_CUDA_CHECK(cudaGetLastError());

    actual_output_size_ = byte_size; // TODO: set to actual output bytes written
}

} // namespace fz
