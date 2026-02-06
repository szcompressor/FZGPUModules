#include "concat.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>

namespace fz {

/**
 * Kernel to write header information (num_inputs and sizes)
 */
__global__ void write_concat_header_kernel(
    uint8_t* output,
    uint32_t num_inputs,
    const uint32_t* sizes,  // Device array of sizes
    size_t num_sizes
) {
    int idx = threadIdx.x;
    
    if (idx == 0) {
        // Write number of inputs
        *reinterpret_cast<uint32_t*>(output) = num_inputs;
    }
    
    if (idx < num_sizes) {
        // Write size at appropriate offset
        size_t offset = sizeof(uint32_t) + idx * sizeof(uint32_t);
        *reinterpret_cast<uint32_t*>(output + offset) = sizes[idx];
    }
}

ConcatenationStage::ConcatenationStage() 
    : Stage("Concatenate") {
}

int ConcatenationStage::execute(void* input, size_t input_size,
                                void* output, cudaStream_t stream) {
    uint8_t* d_output = static_cast<uint8_t*>(output);
    
    if (dependency_outputs_.empty()) {
        fprintf(stderr, "ConcatenationStage: No dependencies set!\n");
        return -1;
    }
    
    recordProfilingStart(stream);
    
    uint32_t num_inputs = static_cast<uint32_t>(dependency_outputs_.size());
    size_t offset = 0;
    
    // Write number of inputs
    cudaMemcpyAsync(d_output + offset, &num_inputs, sizeof(uint32_t),
                    cudaMemcpyHostToDevice, stream);
    offset += sizeof(uint32_t);
    
    // Write all sizes
    for (size_t i = 0; i < dependency_sizes_.size(); i++) {
        uint32_t size = static_cast<uint32_t>(dependency_sizes_[i]);
        cudaMemcpyAsync(d_output + offset, &size, sizeof(uint32_t),
                        cudaMemcpyHostToDevice, stream);
        offset += sizeof(uint32_t);
    }
    
    // Copy all data buffers
    for (size_t i = 0; i < dependency_outputs_.size(); i++) {
        if (dependency_sizes_[i] > 0 && dependency_outputs_[i]) {
            cudaMemcpyAsync(d_output + offset, dependency_outputs_[i],
                           dependency_sizes_[i], cudaMemcpyDeviceToDevice, stream);
            offset += dependency_sizes_[i];
        }
    }
    
    cudaStreamSynchronize(stream);
    
    recordProfilingEnd(stream);
    updateProfilingStats();
    
    return static_cast<int>(offset);
}

cudaGraphNode_t ConcatenationStage::addToGraph(
    cudaGraph_t graph,
    cudaGraphNode_t* dependencies,
    size_t num_deps,
    void* input, size_t input_size,
    void* output,
    const std::vector<void*>& aux_buffers,
    cudaStream_t stream) {
    
    uint8_t* d_output = static_cast<uint8_t*>(output);
    
    if (dependency_outputs_.empty()) {
        fprintf(stderr, "ConcatenationStage: No dependencies set!\n");
        return nullptr;
    }
    
    printf("ConcatenationStage::addToGraph: %zu dependencies\n", dependency_outputs_.size());
    for (size_t i = 0; i < dependency_outputs_.size(); i++) {
        printf("  Dep %zu: output=%p, size=%zu\n", i, dependency_outputs_[i], dependency_sizes_[i]);
    }
    
    // For graph mode, we'll use a simpler approach:
    // Just copy data buffers sequentially without headers
    // (Headers would require H2D copies which don't work well in graphs)
    
    cudaGraphNode_t last_node = nullptr;
    std::vector<cudaGraphNode_t> local_deps;
    
    // Start with the provided dependencies
    if (dependencies && num_deps > 0) {
        local_deps.assign(dependencies, dependencies + num_deps);
    }
    
    size_t offset = 0;
    
    // Copy all data buffers in sequence
    for (size_t i = 0; i < dependency_outputs_.size(); i++) {
        if (dependency_sizes_[i] > 0 && dependency_outputs_[i]) {
            printf("  Adding memcpy node %zu: %zu bytes from %p to %p+%zu\n",
                   i, dependency_sizes_[i], dependency_outputs_[i], d_output, offset);
            
            cudaMemcpy3DParms data_params = {};
            data_params.srcPtr = make_cudaPitchedPtr(dependency_outputs_[i], dependency_sizes_[i], dependency_sizes_[i], 1);
            data_params.dstPtr = make_cudaPitchedPtr(d_output + offset, dependency_sizes_[i], dependency_sizes_[i], 1);
            data_params.extent = make_cudaExtent(dependency_sizes_[i], 1, 1);
            data_params.kind = cudaMemcpyDeviceToDevice;
            
            cudaGraphNode_t data_node;
            cudaError_t err = cudaGraphAddMemcpyNode(&data_node, graph,
                                         local_deps.data(), local_deps.size(),
                                         &data_params);
            if (err != cudaSuccess) {
                fprintf(stderr, "Failed to add data memcpy node %zu: %s\n", i, cudaGetErrorString(err));
                return nullptr;
            }
            
            last_node = data_node;
            local_deps.clear();
            local_deps.push_back(last_node);
            offset += dependency_sizes_[i];
        }
    }
    
    printf("ConcatenationStage::addToGraph: Complete, last_node=%p\n", last_node);
    return last_node;
}

StageMemoryRequirements ConcatenationStage::getMemoryRequirements(size_t input_size) const {
    StageMemoryRequirements req;
    req.output_size = getMaxOutputSize(input_size);
    req.temp_size = 0;
    req.aux_output_size = 0;
    return req;
}

size_t ConcatenationStage::getMaxOutputSize(size_t input_size) const {
    // Header: num_inputs + sizes for each input
    // Assume 3 inputs for now (codes, indices, values)
    size_t header_size = sizeof(uint32_t) * 4;  // num_inputs + 3 sizes
    
    // Assume worst case: input_size * 3 (if all dependencies produce full-size outputs)
    return header_size + input_size * 3;
}

size_t ConcatenationStage::getAverageOutputSize(size_t input_size) const {
    // More realistic: codes compressed 2x, indices compressed 4x, values 20%
    size_t header_size = sizeof(uint32_t) * 4;
    size_t codes_size = input_size / 2;  // Lorenzo codes
    size_t indices_size = input_size / 5;  // 20% outliers, compressed 4x
    size_t values_size = input_size / 5;  // 20% outliers as floats
    return header_size + codes_size + indices_size + values_size;
}

void ConcatenationStage::setDependencyInfo(const std::vector<void*>& dep_outputs,
                                          const std::vector<size_t>& dep_sizes) {
    dependency_outputs_ = dep_outputs;
    dependency_sizes_ = dep_sizes;
}

} // namespace fz

