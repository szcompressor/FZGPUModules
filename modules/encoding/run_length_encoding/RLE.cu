#include "RLE.h"
#include <cub/cub.cuh>
#include <iostream>

namespace fz {

// ========== CUDA Kernels ==========

/**
 * Kernel 1: Identify run boundaries
 * Each thread compares its element with the next, marks where value changes
 */
template<typename TInput>
__global__ void rle_identify_boundaries_kernel(
    const TInput* __restrict__ input,
    uint8_t* __restrict__ flags,  // 1 where run ends, 0 otherwise
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n - 1) {
        // Mark boundary if current != next
        flags[idx] = (input[idx] != input[idx + 1]) ? 1 : 0;
    } else if (idx == n - 1) {
        // Last element always ends a run
        flags[idx] = 1;
    }
}

/**
 * Kernel 2: Encode runs from boundary positions
 * Each thread handles one run
 */
template<typename TInput>
__global__ void rle_encode_runs_kernel(
    const TInput* __restrict__ input,
    const uint8_t* __restrict__ flags,
    const uint32_t* __restrict__ scan,    // Prefix sum of flags = run index
    TInput* __restrict__ out_values,
    uint32_t* __restrict__ out_counts,
    uint32_t num_runs,
    size_t n
) {
    uint32_t run_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (run_idx >= num_runs) return;
    
    // Find start of this run by scanning backwards for the previous boundary
    size_t run_start = 0;
    
    if (run_idx > 0) {
        // Binary search for where scan[i] == run_idx - 1
        // This gives us the last position of the previous run
        for (size_t i = 0; i < n; i++) {
            if (scan[i] == run_idx) {
                run_start = i;
                break;
            }
        }
    }
    
    // Find end of this run (where scan[i] == run_idx and flag[i] == 1)
    size_t run_end = run_start;
    for (size_t i = run_start; i < n; i++) {
        if (scan[i] == run_idx && flags[i] == 1) {
            run_end = i;
            break;
        }
    }
    
    // Encode run
    TInput value = input[run_start];
    uint32_t count = (run_end - run_start) + 1;
    
    out_values[run_idx] = value;
    out_counts[run_idx] = count;
}

/**
 * Test kernel
 */
__global__ void test_kernel(uint32_t* counter) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        atomicAdd(counter, 1);
        printf("TEST KERNEL RAN!\n");
    }
}

/**
 * Simple kernel to zero a uint32_t counter (graph-compatible)
 */
__global__ void zero_uint32_rle_kernel(uint32_t* ptr) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *ptr = 0;
    }
}

/**
 * Sequential RLE kernel - single thread for correctness
 * Not optimal for performance, but guarantees correct encoding
 */
__global__ void rle_encode_sequential_uint16_kernel(
    const uint16_t* __restrict__ input,
    uint8_t* __restrict__ output,
    uint32_t* __restrict__ num_runs_out,
    size_t n
) {
    // Only thread 0 does the work
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    
    if (n == 0) {
        *num_runs_out = 0;
        return;
    }
    
    uint32_t num_runs = 0;
    size_t i = 0;
    
    while (i < n) {
        uint16_t current_value = input[i];
        uint32_t count = 1;
        
        // Count consecutive equal elements
        while (i + count < n && input[i + count] == current_value) {
            count++;
        }
        
        // Write this run to output
        size_t base = 4 + num_runs * 6;  // header + num_runs * (2 bytes value + 4 bytes count)
        
        // Write value (2 bytes)
        output[base + 0] = (current_value >> 0) & 0xFF;
        output[base + 1] = (current_value >> 8) & 0xFF;
        
        // Write count (4 bytes)
        output[base + 2] = (count >> 0) & 0xFF;
        output[base + 3] = (count >> 8) & 0xFF;
        output[base + 4] = (count >> 16) & 0xFF;
        output[base + 5] = (count >> 24) & 0xFF;
        
        num_runs++;
        i += count;
    }
    
    *num_runs_out = num_runs;
}

/**
 * Kernel to write num_runs header after encoding completes
 */
__global__ void rle_finalize_header_kernel(
    const uint32_t* __restrict__ num_runs_ptr,
    uint8_t* __restrict__ output
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        uint32_t num_runs = *num_runs_ptr;
        // Write header as 4 individual bytes to avoid alignment issues
        output[0] = (num_runs >> 0) & 0xFF;
        output[1] = (num_runs >> 8) & 0xFF;
        output[2] = (num_runs >> 16) & 0xFF;
        output[3] = (num_runs >> 24) & 0xFF;
    }
}

// ========== RLEStage Implementation ==========

template<typename TInput>
RLEStage<TInput>::RLEStage(const RLEConfig<TInput>& config)
    : Stage("RLE_" + std::string(
          sizeof(TInput) == 1 ? "u8" : 
          sizeof(TInput) == 2 ? "u16" : "u32")),
      config_(config),
      last_num_runs_(0),
      last_compression_ratio_(1.0f),
      graph_d_input_(nullptr),
      graph_d_output_(nullptr),
      graph_d_run_counter_(nullptr),
      graph_num_elements_(0) {
}

template<typename TInput>
RLEStage<TInput>::~RLEStage() {
}

template<typename TInput>
StageMemoryRequirements RLEStage<TInput>::getMemoryRequirements(size_t input_size) const {
    // Output: num_runs header + worst-case all elements are runs
    size_t output_size = getMaxOutputSize(input_size);
    
    // No temp buffers needed (allocated dynamically in execute())
    size_t temp_size = 0;
    
    // Auxiliary buffer (for graph mode): just the run counter
    size_t aux_size = sizeof(uint32_t);  // run_counter only
    
    return StageMemoryRequirements(output_size, temp_size, aux_size);
}

template<typename TInput>
int RLEStage<TInput>::execute(void* input, size_t input_size,
                              void* output, cudaStream_t stream) {
    if (!input || !output || input_size == 0) {
        return -1;
    }
    
    recordProfilingStart(stream);
    
    const TInput* d_input = static_cast<const TInput*>(input);
    uint8_t* d_output = static_cast<uint8_t*>(output);
    size_t num_elements = input_size / sizeof(TInput);
    
    // Clear any previous CUDA errors
    cudaGetLastError();
    
    // Allocate counter on device (initialized to 0)
    uint32_t* d_run_counter;
    cudaError_t alloc_err = cudaMallocAsync(&d_run_counter, sizeof(uint32_t), stream);
    if (alloc_err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate run counter: %s\n", cudaGetErrorString(alloc_err));
        return -1;
    }
    
    cudaMemsetAsync(d_run_counter, 0, sizeof(uint32_t), stream);
    
    // Cast input to uint16_t* for non-template kernel
    const uint16_t* d_input_u16 = static_cast<const uint16_t*>((const void*)d_input);
    
    rle_encode_sequential_uint16_kernel<<<1, 1, 0, stream>>>(
        d_input_u16, d_output, d_run_counter, num_elements
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "RLE encode kernel launch failed: %s\n", cudaGetErrorString(err));
    }
    
    // Finalize header
    rle_finalize_header_kernel<<<1, 1, 0, stream>>>(
        d_run_counter, d_output
    );
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "RLE header kernel launch failed: %s\n", cudaGetErrorString(err));
    }
    
    // Synchronize before reading result
    cudaStreamSynchronize(stream);
    
    // Get num_runs to calculate output size
    uint32_t num_runs;
    cudaMemcpyAsync(&num_runs, d_run_counter, sizeof(uint32_t),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    // printf("RLE: %u runs encoded\n", num_runs);
    
    // Cleanup
    cudaFreeAsync(d_run_counter, stream);
    
    // Calculate output size
    size_t output_size = sizeof(uint32_t) + 
                        num_runs * (sizeof(TInput) + sizeof(uint32_t));
    
    // Update stats
    last_num_runs_ = num_runs;
    last_compression_ratio_ = static_cast<float>(input_size) / output_size;
    
    recordProfilingEnd(stream);
    updateProfilingStats();
    
    return static_cast<int>(output_size);
}

template<typename TInput>
cudaGraphNode_t RLEStage<TInput>::addToGraph(
    cudaGraph_t graph,
    cudaGraphNode_t* dependencies,
    size_t num_deps,
    void* input, size_t input_size,
    void* output,
    const std::vector<void*>& aux_buffers,
    cudaStream_t stream) {
    
    // Simplified RLE for graph mode: only needs counter buffer
    // aux_buffers[0] = run_counter
    if (aux_buffers.empty()) {
        fprintf(stderr, "RLE stage requires at least 1 auxiliary buffer (counter)\n");
        return nullptr;
    }
    
    // Store parameters in member variables so they persist beyond this function
    graph_d_input_ = static_cast<const TInput*>(input);
    graph_d_output_ = static_cast<uint8_t*>(output);
    graph_num_elements_ = input_size / sizeof(TInput);
    graph_d_run_counter_ = static_cast<uint32_t*>(aux_buffers[0]);
    
    // Node 1: Zero-init run counter (kernel is more graph-compatible than memset)
    cudaGraphNode_t zero_node;
    void* zero_args[] = {(void*)&graph_d_run_counter_};
    
    cudaKernelNodeParams zero_params = {};
    zero_params.func = (void*)zero_uint32_rle_kernel;
    zero_params.gridDim = dim3(1, 1, 1);
    zero_params.blockDim = dim3(1, 1, 1);
    zero_params.sharedMemBytes = 0;
    zero_params.kernelParams = zero_args;
    zero_params.extra = nullptr;
    
    cudaError_t err = cudaGraphAddKernelNode(&zero_node, graph, dependencies, num_deps, &zero_params);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to add zero-init kernel for RLE: %s\n", cudaGetErrorString(err));
        return nullptr;
    }
    
    // Node 2: Sequential RLE encode kernel (single thread for correctness)
    // Use pointers to member variables so they remain valid during graph execution
    void* encode_args[] = {
        (void*)&graph_d_input_,
        (void*)&graph_d_output_,
        (void*)&graph_d_run_counter_,
        (void*)&graph_num_elements_
    };
    
    cudaKernelNodeParams encode_params = {};
    encode_params.func = (void*)rle_encode_sequential_uint16_kernel;
    encode_params.gridDim = dim3(1);
    encode_params.blockDim = dim3(1);
    encode_params.sharedMemBytes = 0;
    encode_params.kernelParams = encode_args;
    encode_params.extra = nullptr;
    
    cudaGraphNode_t encode_node;
    err = cudaGraphAddKernelNode(&encode_node, graph, &zero_node, 1, &encode_params);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to add RLE encode kernel node: %s\n", cudaGetErrorString(err));
        return nullptr;
    }
    
    // Node 3: Finalize header
    void* header_args[] = {
        (void*)&graph_d_run_counter_,
        (void*)&graph_d_output_
    };
    
    cudaKernelNodeParams header_params = {};
    header_params.func = (void*)rle_finalize_header_kernel;
    header_params.gridDim = dim3(1);
    header_params.blockDim = dim3(1);
    header_params.sharedMemBytes = 0;
    header_params.kernelParams = header_args;
    header_params.extra = nullptr;
    
    cudaGraphNode_t header_node;
    err = cudaGraphAddKernelNode(&header_node, graph, &encode_node, 1, &header_params);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to add RLE header kernel node: %s\n", cudaGetErrorString(err));
        return nullptr;
    }
    
    return header_node;
}

// Explicit template instantiations
template class RLEStage<uint8_t>;
template class RLEStage<uint16_t>;
template class RLEStage<uint32_t>;

} // namespace fz
