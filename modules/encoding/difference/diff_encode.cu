#include "diff_encode.h"
#include <cstdio>

namespace fz {

/**
 * Simple difference encoding kernel
 */
template<typename T>
__global__ void difference_encode_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    size_t num_elements
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx == 0 && num_elements > 0) {
        // First element stays as-is
        output[0] = input[0];
    } else if (idx < num_elements) {
        // Subsequent elements are differences
        output[idx] = input[idx] - input[idx - 1];
    }
}

// ========== DifferenceStage Implementation ==========

template<typename T>
DifferenceStage<T>::DifferenceStage(const DifferenceConfig<T>& config)
    : Stage("Diff_" + std::string(
          sizeof(T) == 1 ? "u8" : 
          sizeof(T) == 2 ? "u16" : 
          sizeof(T) == 4 ? "u32" : "u64")),
      config_(config) {
}

template<typename T>
DifferenceStage<T>::~DifferenceStage() {
}

template<typename T>
StageMemoryRequirements DifferenceStage<T>::getMemoryRequirements(size_t input_size) const {
    StageMemoryRequirements req;
    req.output_size = input_size;  // Same size as input
    req.aux_output_size = 0;
    req.temp_size = 0;
    
    return req;
}

template<typename T>
size_t DifferenceStage<T>::getMaxOutputSize(size_t input_size) const {
    return input_size;  // Same size as input
}

template<typename T>
size_t DifferenceStage<T>::getAverageOutputSize(size_t input_size) const {
    return input_size;  // Always same size as input
}

template<typename T>
int DifferenceStage<T>::execute(void* input, size_t input_size,
                                void* output, cudaStream_t stream) {
    const T* d_input = static_cast<const T*>(input);
    T* d_output = static_cast<T*>(output);
    size_t num_elements = input_size / sizeof(T);
    
    if (num_elements == 0) {
        return 0;
    }
    
    recordProfilingStart(stream);
    
    // Launch kernel with enough threads
    int block_size = 256;
    int grid_size = (num_elements + block_size - 1) / block_size;
    
    difference_encode_kernel<T><<<grid_size, block_size, 0, stream>>>(
        d_input, d_output, num_elements
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Difference encoding kernel failed: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    recordProfilingEnd(stream);
    cudaStreamSynchronize(stream);
    updateProfilingStats();
    
    return input_size;  // Output size same as input
}

template<typename T>
cudaGraphNode_t DifferenceStage<T>::addToGraph(cudaGraph_t graph,
                                               cudaGraphNode_t* dependencies,
                                               size_t num_deps,
                                               void* input, size_t input_size,
                                               void* output,
                                               const std::vector<void*>& aux_buffers,
                                               cudaStream_t stream) {
    const T* d_input = static_cast<const T*>(input);
    T* d_output = static_cast<T*>(output);
    size_t num_elements = input_size / sizeof(T);
    
    if (num_elements == 0) {
        return nullptr;
    }
    
    int block_size = 256;
    int grid_size = (num_elements + block_size - 1) / block_size;
    
    void* kernel_args[] = {
        (void*)&d_input,
        (void*)&d_output,
        (void*)&num_elements
    };
    
    cudaKernelNodeParams kernel_params = {};
    kernel_params.func = (void*)difference_encode_kernel<T>;
    kernel_params.gridDim = dim3(grid_size);
    kernel_params.blockDim = dim3(block_size);
    kernel_params.sharedMemBytes = 0;
    kernel_params.kernelParams = kernel_args;
    kernel_params.extra = nullptr;
    
    cudaGraphNode_t kernel_node;
    cudaError_t err = cudaGraphAddKernelNode(&kernel_node, graph, dependencies, num_deps, &kernel_params);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to add difference encoding kernel node: %s\n", cudaGetErrorString(err));
        return nullptr;
    }
    
    return kernel_node;
}

// Explicit template instantiations
template class DifferenceStage<uint8_t>;
template class DifferenceStage<uint16_t>;
template class DifferenceStage<uint32_t>;
template class DifferenceStage<uint64_t>;

} // namespace fz
