#include "predictors/lorenzo/lorenzo.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

namespace fz {

// ===== Warp-level scan primitives for parallel prefix sum =====

template<typename T, int Seq>
__device__ void warp_inclusive_scan(T* thp_data) {
    // Inclusive scan within a warp (32 threads)
    // Each thread has Seq elements in registers
    
    // First, do sequential scan within thread's Seq elements
    for (int i = 1; i < Seq; i++) {
        thp_data[i] += thp_data[i - 1];
    }
    
    // Now scan across threads in warp
    T thread_sum = thp_data[Seq - 1];  // Last element of this thread
    
    // Get lane ID within warp (0-31)
    int lane_id = threadIdx.x % 32;
    
    // Warp-level scan using shuffle operations
    #pragma unroll
    for (int offset = 1; offset < 32; offset *= 2) {
        T val = __shfl_up_sync(0xffffffff, thread_sum, offset);
        if (lane_id >= offset) {
            thread_sum += val;
        }
    }
    
    // Get prefix sum from previous thread in warp
    T prefix = __shfl_up_sync(0xffffffff, thread_sum, 1);
    if (lane_id == 0) prefix = 0;
    
    // Add prefix to all elements in this thread
    for (int i = 0; i < Seq; i++) {
        thp_data[i] += prefix;
    }
}

template<typename T, int Seq, int NumThreads>
__device__ void block_exclusive_scan(T* thp_data, T* exch_in, T* exch_out) {
    constexpr int NumWarps = NumThreads / 32;
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    
    // Each warp writes its total to shared memory
    if (lane_id == 31) {
        exch_in[warp_id] = thp_data[Seq - 1];
    }
    __syncthreads();
    
    // First warp scans the per-warp totals
    if (warp_id == 0 && lane_id < NumWarps) {
        T val = exch_in[lane_id];
        
        // Create mask for only the participating threads
        unsigned mask = (1u << NumWarps) - 1;  // e.g., 0xFF for 8 warps
        
        #pragma unroll
        for (int offset = 1; offset < NumWarps; offset *= 2) {
            T tmp = __shfl_up_sync(mask, val, offset);
            if (lane_id >= offset) val += tmp;
        }
        exch_out[lane_id] = val;
    }
    __syncthreads();
    
    // Add the prefix from previous warps to all threads
    if (warp_id > 0) {
        T warp_prefix = exch_out[warp_id - 1];
        for (int i = 0; i < Seq; i++) {
            thp_data[i] += warp_prefix;
        }
    }
}

/**
 * Scatter kernel to restore outliers to their original positions
 * 
 * Takes sparse (value, index) pairs and ADDS outlier prediction errors
 * to the already-reconstructed values from quantized codes.
 * 
 * During compression, outlier positions store code=radius, 
 * so after scan they have just the cumulative sum up to that point.
 * This kernel scales the outlier errors (stored in quantized units) by ebx2
 * and adds them to the reconstructed values.
 */
template<typename T>
__global__ void scatter_outliers_kernel(
    const T* __restrict__ outlier_values,
    const uint32_t* __restrict__ outlier_indices,
    const uint32_t outlier_count,
    T* __restrict__ output
) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < outlier_count) {
        uint32_t idx = outlier_indices[tid];
        // Write the outlier prediction error in quantized units
        // The prefix sum kernel will read this and incorporate it into the sum
        output[idx] = outlier_values[tid];
    }
}

/**
 * Helper kernel to propagate block sums across blocks
 * After each block computes its local prefix sum, we need to add the total
 * from all previous blocks to maintain global cumulative sum.
 */
template<typename T>
__global__ void propagate_block_sums_kernel(
    T* __restrict__ data,
    T* __restrict__ block_sums,
    const size_t n,
    const int tile_dim
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Determine which original tile (dequantize block) this element belongs to
        int original_block = static_cast<int>(idx / tile_dim);
        
        // Add cumulative sum from all previous original blocks
        if (original_block > 0) {
            T prefix = 0;
            for (int b = 0; b < original_block; b++) {
                prefix += block_sums[b];
            }
            data[idx] += prefix;
        }
    }
}

/**
 * Helper kernel to extract last value from each block
 */
template<typename T>
__global__ void extract_block_sums_kernel(
    const T* __restrict__ data,
    T* __restrict__ block_sums,
    const size_t n,
    const int tile_dim
) {
    int block_id = blockIdx.x;
    
    if (threadIdx.x == 0) {
        // Get last valid element in this block
        size_t last_idx = min(static_cast<size_t>((block_id + 1) * tile_dim - 1), n - 1);
        block_sums[block_id] = data[last_idx];
    }
}

/**
 * Optimized Lorenzo 1D decompression kernel with parallel prefix sum
 * 
 * Based on cuSZ-style block-level scan algorithm:
 * 1. Load and dequantize: codes → prediction errors
 * 2. Parallel prefix sum: warp-level + block-level scan (inverse Lorenzo)
 * 3. Scale by 2*eb and write output
 * 
 * Note: This kernel only handles quantized codes. Outliers are scattered
 * separately using scatter_outliers_kernel.
 * 
 * Performance optimizations:
 * - Shared memory staging for coalesced access
 * - Warp shuffle for efficient intra-warp scan
 * - Block-level coordination for inter-warp scan
 * - Sequential processing per thread (Seq=4) for better occupancy
 */
template<typename TInput, typename TCode, int TileDim, int Seq>
__global__ void lorenzo_dequantize_1d_kernel(
    const TCode* __restrict__ quant_codes,
    const size_t n,
    const TInput ebx2,
    const TCode quant_radius,
    TInput* __restrict__ output
) {
    constexpr int NumThreads = TileDim / Seq;
    
    __shared__ TInput scratch[TileDim];
    __shared__ TCode s_codes[TileDim];
    __shared__ TInput exch_in[NumThreads / 32];
    __shared__ TInput exch_out[NumThreads / 32];
    
    TInput thp_data[Seq];  // Private registers for each thread
    
    const size_t block_offset = blockIdx.x * TileDim;
    
    // ===== Stage 1: Load codes to shared memory (sequential per thread) =====
    #pragma unroll
    for (int i = 0; i < Seq; i++) {
        size_t local_id = threadIdx.x * Seq + i;  // Sequential: thread 0 reads 0,1,2,3
        size_t global_id = block_offset + local_id;
        if (global_id < n && local_id < TileDim) {
            s_codes[local_id] = quant_codes[global_id];
        }
    }
    __syncthreads();
    
    // ===== Stage 2: Dequantize codes to prediction errors =====
    // Write in sequential order for each thread's Seq elements
    #pragma unroll
    for (int i = 0; i < Seq; i++) {
        size_t local_id = threadIdx.x * Seq + i;  // Sequential: 0,1,2,3 for thread 0
        size_t global_id = block_offset + local_id;
        
        if (global_id < n && local_id < TileDim) {
            // Dequantize: code - radius → prediction error
            TInput delta = static_cast<TInput>(s_codes[local_id]) - static_cast<TInput>(quant_radius);
            
            // Inject any scattered outlier prediction errors BEFORE prefix sum
            delta += output[global_id];
            
            scratch[local_id] = delta;
        } else if (local_id < TileDim) {
            scratch[local_id] = 0;
        }
    }
    __syncthreads();
    
    // ===== Stage 3: Load to private registers =====
    #pragma unroll
    for (int i = 0; i < Seq; i++) {
        thp_data[i] = scratch[threadIdx.x * Seq + i];
    }
    __syncthreads();
    
    // ===== Stage 4: Parallel prefix sum (cumulative sum) =====
    // Intra-warp inclusive scan
    warp_inclusive_scan<TInput, Seq>(thp_data);
    
    // Inter-warp exclusive scan (coordinate across warps in block)
    block_exclusive_scan<TInput, Seq, NumThreads>(thp_data, exch_in, exch_out);
    
    // ===== Stage 5: Scale by 2*eb and write back to shared memory =====
    #pragma unroll
    for (int i = 0; i < Seq; i++) {
        scratch[threadIdx.x * Seq + i] = thp_data[i] * ebx2;
    }
    __syncthreads();
    
    // ===== Stage 6: Write output to global memory (sequential per thread) =====
    #pragma unroll
    for (int i = 0; i < Seq; i++) {
        size_t local_id = threadIdx.x * Seq + i;  // Sequential: thread 0 writes 0,1,2,3
        size_t global_id = block_offset + local_id;
        if (global_id < n && local_id < TileDim) {
            output[global_id] = scratch[local_id];
        }
    }
}

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
                // Outlier: write radius (dequantizes to 0 = neutral for prefix sum)
                s_codes[threadIdx.x * Seq + i] = quant_radius;
                
                // Store outlier prediction error (not original value!)
                uint32_t outlier_idx = atomicAdd(outlier_count, 1);
                if (outlier_idx < max_outliers) {
                    outlier_errors[outlier_idx] = delta;  // Store prediction error
                    outlier_indices[outlier_idx] = static_cast<uint32_t>(global_idx);
                }
                
                // Do NOT rewrite thp_data[i + 1]. We want the next element to 
                // predict from this element's true value, not the predicted value.
                // The decompression prefix sum will correctly reconstruct this.
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



// Kernel launcher wrappers (force kernel instantiation)
template<typename TInput, typename TCode>
void launchLorenzoKernel(
    const TInput* input,
    size_t n,
    TInput ebx2_r,
    TCode quant_radius,
    TCode* quant_codes,
    TInput* outlier_errors,
    uint32_t* outlier_indices,
    uint32_t* outlier_count,
    size_t max_outliers,
    int grid_size,
    cudaStream_t stream
) {
    constexpr int TileDim = 1024;
    constexpr int Seq = 4;
    constexpr int NumThreads = TileDim / Seq;  // 256
    
    lorenzo_quantize_1d_kernel<TInput, TCode, TileDim, Seq>
        <<<grid_size, NumThreads, 0, stream>>>(
            input, n, ebx2_r, quant_radius,
            quant_codes, outlier_errors, outlier_indices, outlier_count,
            max_outliers
        );
}

template<typename TInput, typename TCode>
void launchLorenzoInverseKernel(
    const TCode* quant_codes,
    const TInput* outlier_errors,
    const uint32_t* outlier_indices,
    const uint32_t* outlier_count_ptr,
    size_t n,
    TInput ebx2,
    TCode quant_radius,
    TInput* output,
    cudaStream_t stream
) {
    constexpr int TileDim = 1024;
    constexpr int Seq = 4;
    constexpr int NumThreads = TileDim / Seq;  // 256
    
    const int grid_size = (n + TileDim - 1) / TileDim;
    
    // Step 0: Initialize output array to 0 because we will scatter outliers into it
    cudaMemsetAsync(output, 0, n * sizeof(TInput), stream);
    
    // Step 1: Read outlier count and scatter outliers BEFORE prefix sum
    // This allows the prefix sum to correctly propagate outlier values
    uint32_t h_outlier_count = 0;
    if (outlier_count_ptr != nullptr) {
        cudaMemcpyAsync(&h_outlier_count, outlier_count_ptr, sizeof(uint32_t),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
    }
    
    if (h_outlier_count > 0) {
        int scatter_block_size = 256;
        int scatter_grid_size = (h_outlier_count + scatter_block_size - 1) / scatter_block_size;
        
        scatter_outliers_kernel<TInput>
            <<<scatter_grid_size, scatter_block_size, 0, stream>>>(
                outlier_errors, outlier_indices, h_outlier_count, output
            );
            
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("scatter_outliers_kernel launch failed: ") + 
                cudaGetErrorString(err)
            );
        }
    }
    
    // Step 2: Launch decompression kernel (dequantize + prefix sum)
    // This processes the quantized codes and adds the pre-scattered outliers
    lorenzo_dequantize_1d_kernel<TInput, TCode, TileDim, Seq>
        <<<grid_size, NumThreads, 0, stream>>>(
            quant_codes, n, ebx2, quant_radius, output
        );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("lorenzo_dequantize_1d_kernel launch failed: ") + 
            cudaGetErrorString(err)
        );
    }
    
    cudaStreamSynchronize(stream);
    
    // Allocate temporary buffer for block sums (only if multi-block)
    TInput* d_block_sums = nullptr;
    if (grid_size > 1) {
        cudaMallocAsync(&d_block_sums, grid_size * sizeof(TInput), stream);
    }
    
    // Step 3: Extract block sums (last element of each block)
    if (grid_size > 1) {
        extract_block_sums_kernel<TInput>
            <<<grid_size, 1, 0, stream>>>(
                output, d_block_sums, n, TileDim
            );
        
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            if (d_block_sums) cudaFreeAsync(d_block_sums, stream);
            throw std::runtime_error(
                std::string("extract_block_sums_kernel launch failed: ") + 
                cudaGetErrorString(err)
            );
        }
        
        // Step 4: Propagate block sums to maintain global cumulative sum
        int prop_grid_size = (n + NumThreads - 1) / NumThreads;
        propagate_block_sums_kernel<TInput>
            <<<prop_grid_size, NumThreads, 0, stream>>>(
                output, d_block_sums, n, TileDim
            );
        
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            cudaFreeAsync(d_block_sums, stream);
            throw std::runtime_error(
                std::string("propagate_block_sums_kernel launch failed: ") + 
                cudaGetErrorString(err)
            );
        }
    }
    
    // Cleanup
    if (d_block_sums) {
        cudaFreeAsync(d_block_sums, stream);
    }
}

// ========== LorenzoStage Implementation ==========

template<typename TInput, typename TCode>
LorenzoStage<TInput, TCode>::LorenzoStage(const Config& config)
    : config_(config) {
    actual_output_sizes_.resize(4, 0);
}

template<typename TInput, typename TCode>
void LorenzoStage<TInput, TCode>::execute(
    cudaStream_t stream,
    const std::vector<void*>& inputs,
    const std::vector<void*>& outputs,
    const std::vector<size_t>& sizes
) {
    if (is_inverse_) {
        // ===== DECOMPRESSION MODE: 4 inputs → 1 output =====
        if (inputs.size() < 4 || outputs.empty() || sizes.size() < 4) {
            throw std::runtime_error("LorenzoStage (inverse): Requires 4 inputs and 1 output");
        }
        
        size_t codes_size = sizes[0];
        size_t num_elements = codes_size / sizeof(TCode);
        
        if (num_elements == 0) {
            actual_output_sizes_.resize(1);
            actual_output_sizes_[0] = 0;
            return;
        }
        
        // Calculate ebx2 = 2 * error_bound for dequantization
        TInput ebx2 = static_cast<TInput>(2) * config_.error_bound;
        
        // Launch inverse kernel
        launchLorenzoInverseKernel<TInput, TCode>(
            static_cast<const TCode*>(inputs[0]),      // codes
            static_cast<const TInput*>(inputs[1]),     // outlier_errors
            static_cast<const uint32_t*>(inputs[2]),   // outlier_indices
            static_cast<const uint32_t*>(inputs[3]),   // outlier_count
            num_elements,
            ebx2,
            config_.quant_radius,
            static_cast<TInput*>(outputs[0]),          // reconstructed data
            stream
        );
        
        // Check for launch errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("LorenzoStage (inverse) kernel launch failed: ") + 
                cudaGetErrorString(err)
            );
        }
        
        // Set output size
        actual_output_sizes_.resize(1);
        actual_output_sizes_[0] = num_elements * sizeof(TInput);
        
    } else {
        // ===== COMPRESSION MODE: 1 input → 4 outputs =====
        if (inputs.empty() || outputs.size() < 4 || sizes.empty()) {
            throw std::runtime_error("LorenzoStage: Requires 1 input and 4 outputs");
        }
        
        size_t input_size = sizes[0];
        size_t num_elements = input_size / sizeof(TInput);
        size_t max_outliers = getMaxOutlierCount(num_elements);
        
        // Store for header generation
        num_elements_ = num_elements;
        
        if (num_elements == 0) {
            // Empty input
            for (size_t i = 0; i < 4; i++) {
                actual_output_sizes_[i] = 0;
            }
            actual_outlier_count_ = 0;
            return;
        }
        
        // Initialize outlier count to 0
        cudaMemsetAsync(outputs[3], 0, sizeof(uint32_t), stream);
        
        // Calculate ebx2_r = 1 / (2 * error_bound) for pre-quantization
        TInput ebx2_r = static_cast<TInput>(1) / (static_cast<TInput>(2) * config_.error_bound);
        
        // Launch kernel
        constexpr int TileDim = 1024;
        const int grid_size = (num_elements + TileDim - 1) / TileDim;
        
        launchLorenzoKernel<TInput, TCode>(
            static_cast<const TInput*>(inputs[0]),
            num_elements,
            ebx2_r,
            config_.quant_radius,
            static_cast<TCode*>(outputs[0]),
            static_cast<TInput*>(outputs[1]),
            static_cast<uint32_t*>(outputs[2]),
            static_cast<uint32_t*>(outputs[3]),
            max_outliers,
            grid_size,
            stream
        );
        
        // Check for launch errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("LorenzoStage kernel launch failed: ") + 
                cudaGetErrorString(err)
            );
        }
        
        // Read back outlier count to determine actual sizes
        uint32_t h_outlier_count;
        cudaMemcpyAsync(&h_outlier_count, outputs[3], sizeof(uint32_t),
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
        
        // Store for header generation
        actual_outlier_count_ = h_outlier_count;
        
        // Set actual output sizes
        actual_output_sizes_[0] = num_elements * sizeof(TCode);           // codes (fixed)
        actual_output_sizes_[1] = h_outlier_count * sizeof(TInput);       // outlier_errors (variable)
        actual_output_sizes_[2] = h_outlier_count * sizeof(uint32_t);     // outlier_indices (variable)
        actual_output_sizes_[3] = sizeof(uint32_t);                        // outlier_count (fixed)
    }
}

template<typename TInput, typename TCode>
std::vector<size_t> LorenzoStage<TInput, TCode>::estimateOutputSizes(
    const std::vector<size_t>& input_sizes
) const {
    size_t input_size = input_sizes[0];
    size_t num_elements = input_size / sizeof(TInput);
    size_t max_outliers = getMaxOutlierCount(num_elements);
    
    return {
        num_elements * sizeof(TCode),        // codes (fixed size)
        max_outliers * sizeof(TInput),       // outlier_errors (max capacity)
        max_outliers * sizeof(uint32_t),     // outlier_indices (max capacity)
        sizeof(uint32_t)                      // outlier_count (fixed)
    };
}

// ========== Explicit Template Instantiations ==========

// Instantiate classes
template class LorenzoStage<float, uint16_t>;
template class LorenzoStage<float, uint8_t>;
template class LorenzoStage<double, uint16_t>;
template class LorenzoStage<double, uint32_t>;

// Instantiate kernel launchers
template void launchLorenzoKernel<float, uint16_t>(
    const float*, size_t, float, uint16_t,
    uint16_t*, float*, uint32_t*, uint32_t*, size_t, int, cudaStream_t);
template void launchLorenzoKernel<float, uint8_t>(
    const float*, size_t, float, uint8_t,
    uint8_t*, float*, uint32_t*, uint32_t*, size_t, int, cudaStream_t);
template void launchLorenzoKernel<double, uint16_t>(
    const double*, size_t, double, uint16_t,
    uint16_t*, double*, uint32_t*, uint32_t*, size_t, int, cudaStream_t);
template void launchLorenzoKernel<double, uint32_t>(
    const double*, size_t, double, uint32_t,
    uint32_t*, double*, uint32_t*, uint32_t*, size_t, int, cudaStream_t);

template void launchLorenzoInverseKernel<float, uint16_t>(
    const uint16_t*, const float*, const uint32_t*, const uint32_t*,
    size_t, float, uint16_t, float*, cudaStream_t);
template void launchLorenzoInverseKernel<float, uint8_t>(
    const uint8_t*, const float*, const uint32_t*, const uint32_t*,
    size_t, float, uint8_t, float*, cudaStream_t);
template void launchLorenzoInverseKernel<double, uint16_t>(
    const uint16_t*, const double*, const uint32_t*, const uint32_t*,
    size_t, double, uint16_t, double*, cudaStream_t);
template void launchLorenzoInverseKernel<double, uint32_t>(
    const uint32_t*, const double*, const uint32_t*, const uint32_t*,
    size_t, double, uint32_t, double*, cudaStream_t);

} // namespace fz
