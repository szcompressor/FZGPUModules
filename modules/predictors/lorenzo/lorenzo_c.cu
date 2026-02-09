#include "lorenzo.h"
#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <cmath>

namespace fz {

// ========== CUDA Kernel Implementation ==========

/**
 * Optimized Lorenzo 1D prediction + quantization kernel
 * 
 * Performance optimizations:
 * 1. Pre-quantization: Multiply by ebx2_r first to simplify all subsequent math
 * 2. Shared memory staging: Better memory coalescing for global loads/stores
 * 3. Sequential processing: Each thread processes Seq elements for better occupancy
 * 4. Proper thread boundaries: Uses shared memory to get previous element correctly
 * 5. Direct radius check: Simpler outlier detection (fabs(delta) < radius)
 * 6. Store prediction errors: More compressible than original values
 * 
 * Memory layout:
 * - TileDim = 1024 elements per block
 * - Seq = 4 elements per thread
 * - NumThreads = 256 (TileDim / Seq)
 */
template<typename TInput, typename TCode, int TileDim, int Seq>
__global__ void lorenzo_quantize_1d_kernel(
    const TInput* __restrict__ input,
    const size_t n,
    const TInput ebx2_r,
    const TCode quant_radius,
    TCode* __restrict__ quant_codes,
    TInput* __restrict__ outlier_errors,
    uint32_t* __restrict__ outlier_indices,
    uint32_t* __restrict__ outlier_count,
    const size_t max_outliers
) {
    constexpr int NumThreads = TileDim / Seq;
    
    // Shared memory for staging and better memory access patterns
    __shared__ TInput s_data[TileDim];
    __shared__ TCode s_codes[TileDim];
    
    // Private thread data: current Seq elements + 1 previous element
    TInput thp_data[Seq + 1];
    
    const size_t block_offset = blockIdx.x * TileDim;
    
    // ========== Stage 1: Load input data to shared memory with pre-quantization ==========
    // Coalesced loads: each thread loads Seq elements
    #pragma unroll
    for (int i = 0; i < Seq; i++) {
        size_t global_idx = block_offset + threadIdx.x + i * NumThreads;
        if (global_idx < n) {
            // Pre-quantize: multiply by 1/(2*eb) to normalize to quantization units
            s_data[threadIdx.x + i * NumThreads] = round(input[global_idx] * ebx2_r);
        }
    }
    __syncthreads();
    
    // ========== Stage 2: Load to private registers ==========
    // Each thread loads its Seq elements
    #pragma unroll
    for (int i = 0; i < Seq; i++) {
        thp_data[i + 1] = s_data[threadIdx.x * Seq + i];
    }
    
    // Load previous element for Lorenzo prediction
    // Critical fix: properly handle thread boundaries within block
    if (threadIdx.x > 0) {
        // Previous element is in shared memory from previous thread
        thp_data[0] = s_data[threadIdx.x * Seq - 1];
    } else if (block_offset > 0) {
        // First thread of block: need to load from previous block in global memory
        thp_data[0] = round(input[block_offset - 1] * ebx2_r);
    } else {
        // Very first element: no prediction
        thp_data[0] = static_cast<TInput>(0);
    }
    
    // ========== Stage 3: Lorenzo prediction + quantization ==========
    #pragma unroll
    for (int i = 0; i < Seq; i++) {
        size_t global_idx = block_offset + threadIdx.x * Seq + i;
        
        if (global_idx < n) {
            // Lorenzo 1D: predict from previous element
            TInput delta = thp_data[i + 1] - thp_data[i];
            
            // Check if quantizable (directly using radius)
            bool quantizable = fabs(delta) < static_cast<TInput>(quant_radius);
            
            if (quantizable) {
                // Quantize: delta is already in quantization units, just offset to positive range
                // delta ∈ (-radius, radius) → code ∈ (0, 2*radius)
                TInput candidate = delta + static_cast<TInput>(quant_radius);
                s_codes[threadIdx.x * Seq + i] = static_cast<TCode>(candidate);
            } else {
                // Outlier: write 0 (outlier_indices tells us where to reconstruct)
                s_codes[threadIdx.x * Seq + i] = 0;
                
                // Store outlier prediction error (not original value!)
                uint32_t outlier_idx = atomicAdd(outlier_count, 1);
                if (outlier_idx < max_outliers) {
                    outlier_errors[outlier_idx] = delta;  // Store prediction error
                    outlier_indices[outlier_idx] = static_cast<uint32_t>(global_idx);
                }
            }
        }
    }
    __syncthreads();
    
    // ========== Stage 4: Write quantization codes to global memory ==========
    // Coalesced stores
    #pragma unroll
    for (int i = 0; i < Seq; i++) {
        size_t global_idx = block_offset + threadIdx.x + i * NumThreads;
        if (global_idx < n) {
            quant_codes[global_idx] = s_codes[threadIdx.x + i * NumThreads];
        }
    }
}

// ========== LorenzoStage Implementation ==========

template<typename TInput, typename TCode>
LorenzoStage<TInput, TCode>::LorenzoStage(const LorenzoConfig<TInput, TCode>& config)
    : MultiOutputStage("Lorenzo1D"),
      config_(config) {
}

template<typename TInput, typename TCode>
int LorenzoStage<TInput, TCode>::execute(void* input, size_t input_size,
                                         void* output, cudaStream_t stream) {
    // Simple version without outlier handling
    size_t num_elements = input_size / sizeof(TInput);
    
    // Allocate temporary outlier buffers (will be discarded)
    size_t max_outliers = getMaxOutlierCount(num_elements);
    
    TInput* d_outlier_values;
    uint32_t* d_outlier_indices;
    uint32_t* d_outlier_count;
    
    cudaMallocAsync(&d_outlier_values, max_outliers * sizeof(TInput), stream);
    cudaMallocAsync(&d_outlier_indices, max_outliers * sizeof(uint32_t), stream);
    cudaMallocAsync(&d_outlier_count, sizeof(uint32_t), stream);
    cudaMemsetAsync(d_outlier_count, 0, sizeof(uint32_t), stream);
    
    // Launch kernel
    dim3 block_size, grid_size;
    getOptimalLaunchConfig(input_size, block_size, grid_size);
    
    recordProfilingStart(stream);
    
    // Calculate ebx2_r = 1 / (2 * error_bound) for pre-quantization
    TInput ebx2_r = static_cast<TInput>(1) / (static_cast<TInput>(2) * config_.error_bound);
    
    lorenzo_quantize_1d_kernel<TInput, TCode, 1024, 4><<<grid_size, block_size, 0, stream>>>(
        static_cast<const TInput*>(input),
        num_elements,
        ebx2_r,
        config_.quant_radius,
        static_cast<TCode*>(output),
        d_outlier_values,
        d_outlier_indices,
        d_outlier_count,
        max_outliers
    );
    
    recordProfilingEnd(stream);
    
    // Cleanup temporary buffers
    cudaFreeAsync(d_outlier_values, stream);
    cudaFreeAsync(d_outlier_indices, stream);
    cudaFreeAsync(d_outlier_count, stream);
    
    updateProfilingStats();
    
    return num_elements * sizeof(TCode);
}

template<typename TInput, typename TCode>
int LorenzoStage<TInput, TCode>::executeMulti(void* input, size_t input_size,
                                              void* primary_output,
                                              std::vector<void*>& aux_outputs,
                                              std::vector<size_t>& aux_sizes,
                                              cudaStream_t stream) {
    if (aux_outputs.size() < 3) {
        fprintf(stderr, "LorenzoStage: Need 3 auxiliary outputs\n");
        return -1;
    }
    
    size_t num_elements = input_size / sizeof(TInput);
    size_t max_outliers = getMaxOutlierCount(num_elements);
    
    // Initialize outlier count to 0
    cudaMemsetAsync(aux_outputs[2], 0, sizeof(uint32_t), stream);
    
    // Launch kernel
    dim3 block_size, grid_size;
    getOptimalLaunchConfig(input_size, block_size, grid_size);
    
    recordProfilingStart(stream);
    
    // Calculate ebx2_r = 1 / (2 * error_bound) for pre-quantization
    TInput ebx2_r = static_cast<TInput>(1) / (static_cast<TInput>(2) * config_.error_bound);
    
    lorenzo_quantize_1d_kernel<TInput, TCode, 1024, 4><<<grid_size, block_size, 0, stream>>>(
        static_cast<const TInput*>(input),
        num_elements,
        ebx2_r,
        config_.quant_radius,
        static_cast<TCode*>(primary_output),
        static_cast<TInput*>(aux_outputs[0]),
        static_cast<uint32_t*>(aux_outputs[1]),
        static_cast<uint32_t*>(aux_outputs[2]),
        max_outliers
    );
    
    recordProfilingEnd(stream);
    
    // Read back outlier count to determine actual sizes
    uint32_t h_outlier_count;
    cudaMemcpyAsync(&h_outlier_count, aux_outputs[2], sizeof(uint32_t),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    // Check if outlier count exceeded allocated capacity
    if (h_outlier_count > max_outliers) {
        float actual_percent = 100.0f * h_outlier_count / num_elements;
        float capacity_percent = 100.0f * max_outliers / num_elements;
        
        fprintf(stderr, "\n⚠️  ERROR: Lorenzo outlier overflow!\n");
        fprintf(stderr, "   Detected outliers: %u (%.1f%% of data)\n", 
                h_outlier_count, actual_percent);
        fprintf(stderr, "   Allocated capacity: %zu (%.1f%% of data)\n", 
                max_outliers, capacity_percent);
        fprintf(stderr, "   Outliers beyond capacity were DROPPED - data will be corrupted!\n");
        fprintf(stderr, "   Solution: Increase outlier_capacity to at least %.1f%%\n\n", 
                actual_percent * 1.1f);  // Add 10% margin
        
        // Clamp to allocated size to prevent buffer overflow
        h_outlier_count = max_outliers;
    }
    
    // Set actual auxiliary output sizes
    aux_sizes.resize(3);
    aux_sizes[0] = h_outlier_count * sizeof(TInput);      // outlier values
    aux_sizes[1] = h_outlier_count * sizeof(uint32_t);   // outlier indices
    aux_sizes[2] = sizeof(uint32_t);                      // outlier count
    
    updateProfilingStats();
    
    return num_elements * sizeof(TCode);
}

template<typename TInput, typename TCode>
cudaGraphNode_t LorenzoStage<TInput, TCode>::addToGraph(cudaGraph_t graph,
                                                         cudaGraphNode_t* dependencies,
                                                         size_t num_deps,
                                                         void* input, size_t input_size,
                                                         void* output,
                                                         const std::vector<void*>& aux_buffers,
                                                         cudaStream_t stream) {
    // Use pre-allocated auxiliary buffers from mempool
    size_t num_elements = input_size / sizeof(TInput);
    size_t max_outliers = getMaxOutlierCount(num_elements);
    
    // Calculate kernel parameters
    dim3 block_size, grid_size;
    getOptimalLaunchConfig(input_size, block_size, grid_size);
    
    TInput ebx2_r = static_cast<TInput>(1) / (static_cast<TInput>(2) * config_.error_bound);
    
    // Use pre-allocated buffers from mempool (allocated in DAG::allocateBuffers)
    // Expected aux_buffers: [0] = outlier_values, [1] = outlier_indices, [2] = outlier_count
    if (aux_buffers.size() < 3) {
        fprintf(stderr, "Lorenzo stage requires 3 auxiliary buffers, got %zu\n", aux_buffers.size());
        return nullptr;
    }
    
    TInput* d_outlier_values = static_cast<TInput*>(aux_buffers[0]);
    uint32_t* d_outlier_indices = static_cast<uint32_t*>(aux_buffers[1]);
    uint32_t* d_outlier_count = static_cast<uint32_t*>(aux_buffers[2]);
    
    // Add memset node to zero the outlier count
    cudaGraphNode_t memset_node;
    cudaMemsetParams memset_params = {};
    memset_params.dst = d_outlier_count;
    memset_params.value = 0;
    memset_params.pitch = 0;
    memset_params.elementSize = 1;  // Byte-level granularity
    memset_params.width = sizeof(uint32_t);  // 4 bytes for uint32_t
    memset_params.height = 1;
    
    cudaError_t err = cudaGraphAddMemsetNode(&memset_node, graph, dependencies, num_deps, &memset_params);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to add memset node: %s\n", cudaGetErrorString(err));
        return nullptr;
    }
    
    // Cast input/output pointers to correct types
    const TInput* typed_input = static_cast<const TInput*>(input);
    TCode* typed_output = static_cast<TCode*>(output);
    
    // Set up kernel parameters - must match kernel signature exactly
    void* kernel_args[] = {
        (void*)&typed_input,      // const TInput* input
        (void*)&num_elements,     // size_t n
        (void*)&ebx2_r,          // TInput ebx2_r
        (void*)&config_.quant_radius,  // TCode quant_radius
        (void*)&typed_output,    // TCode* quant_codes
        (void*)&d_outlier_values,     // TInput* outlier_errors
        (void*)&d_outlier_indices,    // uint32_t* outlier_indices
        (void*)&d_outlier_count,      // uint32_t* outlier_count
        (void*)&max_outliers     // size_t max_outliers
    };
    
    // Create kernel node parameters
    cudaKernelNodeParams kernel_params = {};
    kernel_params.func = (void*)lorenzo_quantize_1d_kernel<TInput, TCode, 1024, 4>;
    kernel_params.gridDim = grid_size;
    kernel_params.blockDim = block_size;
    kernel_params.sharedMemBytes = 0;
    kernel_params.kernelParams = kernel_args;
    kernel_params.extra = nullptr;
    
    // Add kernel node to graph - depends on memset node
    cudaGraphNode_t kernel_node;
    err = cudaGraphAddKernelNode(&kernel_node, graph, &memset_node, 1, &kernel_params);
    
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to add Lorenzo kernel node: %s\n", cudaGetErrorString(err));
        return nullptr;
    }
    
    return kernel_node;
}

template<typename TInput, typename TCode>
StageMemoryRequirements LorenzoStage<TInput, TCode>::getMemoryRequirements(size_t input_size) const {
    size_t num_elements = input_size / sizeof(TInput);
    size_t max_outliers = getMaxOutlierCount(num_elements);
    
    StageMemoryRequirements req;
    
    // Primary output: quantization codes
    req.output_size = num_elements * sizeof(TCode);
    
    // Temporary memory: minimal (none needed for this implementation)
    req.temp_size = 0;
    
    // Auxiliary outputs: outlier data
    // - outlier values: max_outliers * sizeof(TInput)
    // - outlier indices: max_outliers * sizeof(uint32_t)
    // - outlier count: sizeof(uint32_t)
    req.aux_output_size = max_outliers * sizeof(TInput) +
                          max_outliers * sizeof(uint32_t) +
                          sizeof(uint32_t);
    
    return req;
}

template<typename TInput, typename TCode>
void LorenzoStage<TInput, TCode>::setErrorBound(TInput error_bound) {
    config_.error_bound = error_bound;
}

template<typename TInput, typename TCode>
StageMetadata LorenzoStage<TInput, TCode>::getMetadata() const {
    StageMetadata meta("Lorenzo1D");
    meta.register_pressure = 16;        // Low register usage
    
    // Shared memory usage: s_data[1024] + s_codes[1024]
    constexpr int TileDim = 1024;
    meta.shared_memory_bytes = (TileDim * sizeof(TInput)) + (TileDim * sizeof(TCode));
    
    meta.is_memory_bound = true;        // Memory bandwidth limited
    meta.produces_variable_output = true; // Outlier count varies
    return meta;
}

template<typename TInput, typename TCode>
void LorenzoStage<TInput, TCode>::getOptimalLaunchConfig(size_t input_size,
                                                          dim3& block_size,
                                                          dim3& grid_size) const {
    // Optimized launch config: 256 threads, each processing 4 sequential elements
    // Total: 1024 elements per block
    constexpr int TileDim = 1024;
    constexpr int Seq = 4;
    constexpr int NumThreads = TileDim / Seq;  // 256 threads
    
    block_size = dim3(NumThreads, 1, 1);
    
    size_t num_elements = input_size / sizeof(TInput);
    size_t num_blocks = (num_elements + TileDim - 1) / TileDim;
    
    grid_size = dim3(num_blocks, 1, 1);
}

// ========== Explicit Template Instantiations ==========
// Instantiate common type combinations
template class LorenzoStage<float, uint16_t>;
template class LorenzoStage<float, uint8_t>;
template class LorenzoStage<double, uint16_t>;
template class LorenzoStage<double, uint32_t>;

// Instantiate kernels with optimized tile configuration (1024 elements, 4 seq)
template __global__ void lorenzo_quantize_1d_kernel<float, uint16_t, 1024, 4>(
    const float*, size_t, float, uint16_t,
    uint16_t*, float*, uint32_t*, uint32_t*, size_t);
template __global__ void lorenzo_quantize_1d_kernel<float, uint8_t, 1024, 4>(
    const float*, size_t, float, uint8_t,
    uint8_t*, float*, uint32_t*, uint32_t*, size_t);
template __global__ void lorenzo_quantize_1d_kernel<double, uint16_t, 1024, 4>(
    const double*, size_t, double, uint16_t,
    uint16_t*, double*, uint32_t*, uint32_t*, size_t);
template __global__ void lorenzo_quantize_1d_kernel<double, uint32_t, 1024, 4>(
    const double*, size_t, double, uint32_t,
    uint32_t*, double*, uint32_t*, uint32_t*, size_t);

} // namespace fz
