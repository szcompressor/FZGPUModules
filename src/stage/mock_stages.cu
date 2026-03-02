#include "stage/mock_stages.h"
#include <cuda_runtime.h>

namespace fz {

// CUDA kernel to scale float data by 2 (forward) or divide by 2 (inverse)
__global__ void scaleKernel(const float* input, float* output, size_t n, bool is_inverse) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = is_inverse ? (input[idx] * 0.5f) : (input[idx] * 2.0f);
    }
}

void ScaleStage::execute(
    cudaStream_t stream,
    MemoryPool* pool,
    const std::vector<void*>& inputs,
    const std::vector<void*>& outputs,
    const std::vector<size_t>& sizes
) {
    (void)pool;
    size_t n = sizes[0] / sizeof(float);
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    
    scaleKernel<<<grid_size, block_size, 0, stream>>>(
        static_cast<const float*>(inputs[0]),
        static_cast<float*>(outputs[0]),
        n,
        is_inverse_
    );
    
    actual_output_size_ = sizes[0];
}

} // namespace fz
