#include "encoders/RLE/rle.h"
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include "mem/mempool.h"
#include "cuda_check.h"

namespace fz {

/**
 * RLE Decompression Kernel (Inverse)
 * 
 * Expands (value, run_length) pairs back to original sequence.
 * Each thread handles one run, writing multiple output values.
 * 
 * Input format: [num_runs] [value1, count1, value2, count2, ...]
 * Output: Expanded sequence [value1×count1, value2×count2, ...]
 */
template<typename T>
__global__ void rle_decompress_kernel(
    const T* __restrict__ compressed_values,     // [num_runs]
    const uint32_t* __restrict__ run_lengths,    // [num_runs]
    const uint32_t* __restrict__ run_offsets,    // [num_runs] prefix sum of run_lengths
    T* __restrict__ output,
    const uint32_t num_runs
) {
    uint32_t run_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (run_idx < num_runs) {
        T value = compressed_values[run_idx];
        uint32_t start = (run_idx == 0) ? 0 : run_offsets[run_idx - 1];
        uint32_t end = run_offsets[run_idx];
        
        // Write this value 'count' times
        for (uint32_t i = start; i < end; i++) {
            output[i] = value;
        }
    }
}

/**
 * RLE Compression Kernel (Forward) - Two-phase approach
 * 
 * Phase 1: Identify run boundaries (mark positions where value changes)
 * Phase 2: Compact runs and compute lengths
 */
template<typename T>
__global__ void rle_mark_boundaries_kernel(
    const T* __restrict__ input,
    uint8_t* __restrict__ is_boundary,  // 1 if start of new run, 0 otherwise
    const size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        if (idx == 0) {
            is_boundary[idx] = 1;  // First element always starts a run
        } else {
            is_boundary[idx] = (input[idx] != input[idx - 1]) ? 1 : 0;
        }
    }
}

/**
 * Scatter boundary positions to create a compact array
 * Each thread checks if its position is a boundary, and if so,
 * writes its position to the output array at the index given by boundary_scan
 */
__global__ void scatter_boundary_positions_kernel(
    const uint8_t* __restrict__ is_boundary,
    const uint32_t* __restrict__ boundary_scan,  // Prefix sum of is_boundary
    uint32_t* __restrict__ boundary_positions,   // Output: scattered positions
    const size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n && is_boundary[idx]) {
        uint32_t run_id = boundary_scan[idx] - 1;  // Convert to 0-based run index
        boundary_positions[run_id] = static_cast<uint32_t>(idx);
    }
}

/**
 * Extract run information after boundary detection
 * Each thread processes one detected run boundary
 * 
 * Strategy: Use boundary positions to compute run lengths directly
 * boundary_scan[i] = cumulative count of boundaries up to position i
 * If is_boundary[i] == 1, then run_id = boundary_scan[i] - 1
 */
template<typename T>
__global__ void rle_extract_runs_kernel(
    const T* __restrict__ input,
    const uint8_t* __restrict__ is_boundary,
    const uint32_t* __restrict__ boundary_positions,  // Scattered positions of boundaries
    const uint32_t num_runs,
    T* __restrict__ compressed_values,
    uint32_t* __restrict__ run_lengths,
    const size_t n
) {
    uint32_t run_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (run_id < num_runs) {
        uint32_t start_pos = boundary_positions[run_id];
        uint32_t end_pos = (run_id + 1 < num_runs) ? boundary_positions[run_id + 1] : n;
        
        // Store the value for this run
        compressed_values[run_id] = input[start_pos];
        
        // Compute run length
        run_lengths[run_id] = end_pos - start_pos;
    }
}

// Kernel launchers
template<typename T>
void launchRLEDecompressKernel(
    const T* compressed_values,
    const uint32_t* run_lengths,
    const uint32_t* run_offsets,
    T* output,
    uint32_t num_runs,
    cudaStream_t stream
) {
    const int block_size = 256;
    const int grid_size = (num_runs + block_size - 1) / block_size;
    
    rle_decompress_kernel<T><<<grid_size, block_size, 0, stream>>>(
        compressed_values, run_lengths, run_offsets, output, num_runs
    );
}

template<typename T>
void RLEStage<T>::execute(
    cudaStream_t stream,
    MemoryPool* pool,
    const std::vector<void*>& inputs,
    const std::vector<void*>& outputs,
    const std::vector<size_t>& sizes
) {
    if (inputs.empty() || outputs.empty() || sizes.empty()) {
        throw std::runtime_error("RLEStage: Invalid inputs/outputs");
    }
    
    if (is_inverse_) {
        // ===== DECOMPRESSION: Expand runs =====
        
        // Read the number of runs from the first uint32_t
        uint32_t num_runs;
        FZ_CUDA_CHECK(cudaMemcpyAsync(&num_runs, inputs[0], sizeof(uint32_t), 
                       cudaMemcpyDeviceToHost, stream));
        FZ_CUDA_CHECK(cudaStreamSynchronize(stream));
        
        if (num_runs == 0) {
            actual_output_sizes_ = {0};
            return;
        }
        
        // Input layout: [num_runs (4B)] [values: num_runs*T (aligned)] [counts: num_runs*4B]
        const uint8_t* input_base = static_cast<const uint8_t*>(inputs[0]);
        const T* compressed_values = reinterpret_cast<const T*>(input_base + sizeof(uint32_t));
        
        // Calculate aligned offset for run_lengths
        size_t values_size = num_runs * sizeof(T);
        size_t values_aligned = (values_size + 3) & ~3;  // Round up to multiple of 4
        const uint32_t* run_lengths = reinterpret_cast<const uint32_t*>(
            input_base + sizeof(uint32_t) + values_aligned
        );
        
        // Allocate temporary buffer for prefix sum of run_lengths
        uint32_t* d_run_offsets = nullptr;
        if (pool) {
            d_run_offsets = static_cast<uint32_t*>(pool->allocate(num_runs * sizeof(uint32_t), stream, "rle_run_offsets"));
        } else {
            FZ_CUDA_CHECK(cudaMallocAsync(&d_run_offsets, num_runs * sizeof(uint32_t), stream));
        }
        
        // Compute prefix sum of run_lengths using CUB
        void* d_temp_storage = nullptr;
        size_t temp_storage_bytes = 0;
        cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes,
                                      run_lengths, d_run_offsets, num_runs, stream);
        d_temp_storage = pool
            ? pool->allocate(temp_storage_bytes, stream, "rle_cub_decomp_temp")
            : nullptr;
        if (!pool) FZ_CUDA_CHECK(cudaMallocAsync(&d_temp_storage, temp_storage_bytes, stream));
        cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes,
                                      run_lengths, d_run_offsets, num_runs, stream);
        
        // Get total output size (last element of prefix sum)
        uint32_t total_output_size;
        FZ_CUDA_CHECK(cudaMemcpyAsync(&total_output_size, d_run_offsets + num_runs - 1, 
                       sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
        FZ_CUDA_CHECK(cudaStreamSynchronize(stream));
        
        // Launch decompression kernel
        launchRLEDecompressKernel<T>(
            compressed_values,
            run_lengths,
            d_run_offsets,
            static_cast<T*>(outputs[0]),
            num_runs,
            stream
        );
        
        // Cleanup
        if (pool) { pool->free(d_run_offsets, stream); pool->free(d_temp_storage, stream); }
        else { FZ_CUDA_CHECK_WARN(cudaFreeAsync(d_run_offsets, stream)); FZ_CUDA_CHECK_WARN(cudaFreeAsync(d_temp_storage, stream)); }
        
        actual_output_sizes_ = {total_output_size * sizeof(T)};
        
    } else {
        // ===== COMPRESSION: Encode runs =====
        
        size_t byte_size = sizes[0];
        size_t n = byte_size / sizeof(T);
        
        if (n == 0) {
            // Write num_runs = 0
            uint32_t zero = 0;
            FZ_CUDA_CHECK(cudaMemcpyAsync(outputs[0], &zero, sizeof(uint32_t), 
                           cudaMemcpyHostToDevice, stream));
            actual_output_sizes_ = {sizeof(uint32_t)};
            return;
        }
        
        const T* input = static_cast<const T*>(inputs[0]);
        const int block_size = 256;
        const int grid_size = (n + block_size - 1) / block_size;
        
        // Allocate temporary buffers
        uint8_t* d_is_boundary = nullptr;
        uint32_t* d_boundary_scan = nullptr;
        if (pool) {
            d_is_boundary = static_cast<uint8_t*>(pool->allocate(n * sizeof(uint8_t), stream, "rle_is_boundary"));
            d_boundary_scan = static_cast<uint32_t*>(pool->allocate(n * sizeof(uint32_t), stream, "rle_boundary_scan"));
        } else {
            FZ_CUDA_CHECK(cudaMallocAsync(&d_is_boundary, n * sizeof(uint8_t), stream));
            FZ_CUDA_CHECK(cudaMallocAsync(&d_boundary_scan, n * sizeof(uint32_t), stream));
        }
        uint32_t* d_boundary_positions = nullptr;  // allocated after counting runs
        
        // Phase 1: Mark run boundaries
        rle_mark_boundaries_kernel<T><<<grid_size, block_size, 0, stream>>>(
            input, d_is_boundary, n
        );
        cudaError_t err1 = cudaGetLastError();
        if (err1 != cudaSuccess) {
            throw std::runtime_error(std::string("rle_mark_boundaries_kernel failed: ") + cudaGetErrorString(err1));
        }
        FZ_CUDA_CHECK(cudaStreamSynchronize(stream));
        
        // Compute prefix sum to count runs and get run IDs
        void* d_temp_storage = nullptr;
        size_t temp_storage_bytes = 0;
        cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes,
                                      d_is_boundary, d_boundary_scan, n, stream);
        d_temp_storage = pool
            ? pool->allocate(temp_storage_bytes, stream, "rle_cub_scan_temp")
            : nullptr;
        if (!pool) FZ_CUDA_CHECK(cudaMallocAsync(&d_temp_storage, temp_storage_bytes, stream));
        cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes,
                                      d_is_boundary, d_boundary_scan, n, stream);
        
        // Get total number of runs
        uint32_t num_runs;
        FZ_CUDA_CHECK(cudaMemcpyAsync(&num_runs, d_boundary_scan + n - 1, sizeof(uint32_t),
                       cudaMemcpyDeviceToHost, stream));
        FZ_CUDA_CHECK(cudaStreamSynchronize(stream));
        
        // Allocate array for boundary positions
        if (pool) {
            d_boundary_positions = static_cast<uint32_t*>(pool->allocate(num_runs * sizeof(uint32_t), stream, "rle_boundary_pos"));
        } else {
            FZ_CUDA_CHECK(cudaMallocAsync(&d_boundary_positions, num_runs * sizeof(uint32_t), stream));
        }
        
        // Phase 2: Scatter boundary positions to compact array
        scatter_boundary_positions_kernel<<<grid_size, block_size, 0, stream>>>(
            d_is_boundary, d_boundary_scan, d_boundary_positions, n
        );
        cudaError_t err2 = cudaGetLastError();
        if (err2 != cudaSuccess) {
            throw std::runtime_error(std::string("scatter_boundary_positions_kernel failed: ") + cudaGetErrorString(err2));
        }
        FZ_CUDA_CHECK(cudaStreamSynchronize(stream));
        
        // Allocate output buffer for compressed data
        // Layout: [num_runs] [values] [PAD] [run_lengths]
        // Need to align run_lengths to 4-byte boundary
        size_t values_size = num_runs * sizeof(T);
        size_t values_aligned = (values_size + 3) & ~3;  // Round up to multiple of 4
        size_t compressed_size = sizeof(uint32_t) +      // num_runs
                                values_aligned +          // values (aligned)
                                num_runs * sizeof(uint32_t);  // run_lengths
        
        uint8_t* output_base = static_cast<uint8_t*>(outputs[0]);
        
        // Write num_runs
        FZ_CUDA_CHECK(cudaMemcpyAsync(output_base, &num_runs, sizeof(uint32_t),
                       cudaMemcpyHostToDevice, stream));
        
        T* d_compressed_values = reinterpret_cast<T*>(output_base + sizeof(uint32_t));
        uint32_t* d_run_lengths = reinterpret_cast<uint32_t*>(
            output_base + sizeof(uint32_t) + values_aligned  // Use aligned offset
        );
        
        // Phase 3: Extract run values and lengths using boundary positions
        const int extract_grid_size = (num_runs + block_size - 1) / block_size;
        rle_extract_runs_kernel<T><<<extract_grid_size, block_size, 0, stream>>>(
            input, d_is_boundary, d_boundary_positions, num_runs,
            d_compressed_values, d_run_lengths, n
        );
        cudaError_t err3 = cudaGetLastError();
        if (err3 != cudaSuccess) {
            throw std::runtime_error(std::string("rle_extract_runs_kernel failed: ") + cudaGetErrorString(err3));
        }
        FZ_CUDA_CHECK(cudaStreamSynchronize(stream));
        
        // Cleanup
        if (pool) {
            pool->free(d_is_boundary, stream);
            pool->free(d_boundary_scan, stream);
            pool->free(d_boundary_positions, stream);
            pool->free(d_temp_storage, stream);
        } else {
            FZ_CUDA_CHECK_WARN(cudaFreeAsync(d_is_boundary, stream));
            FZ_CUDA_CHECK_WARN(cudaFreeAsync(d_boundary_scan, stream));
            FZ_CUDA_CHECK_WARN(cudaFreeAsync(d_boundary_positions, stream));
            FZ_CUDA_CHECK_WARN(cudaFreeAsync(d_temp_storage, stream));
        }
        
        actual_output_sizes_ = {compressed_size};
    }
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("RLEStage kernel launch failed: ") + 
            cudaGetErrorString(err)
        );
    }
}

// Explicit template instantiations
template class RLEStage<uint8_t>;
template class RLEStage<uint16_t>;
template class RLEStage<uint32_t>;
template class RLEStage<int32_t>;

// Explicitly instantiate kernel launchers
template void launchRLEDecompressKernel<uint8_t>(const uint8_t*, const uint32_t*, const uint32_t*, uint8_t*, uint32_t, cudaStream_t);
template void launchRLEDecompressKernel<uint16_t>(const uint16_t*, const uint32_t*, const uint32_t*, uint16_t*, uint32_t, cudaStream_t);
template void launchRLEDecompressKernel<uint32_t>(const uint32_t*, const uint32_t*, const uint32_t*, uint32_t*, uint32_t, cudaStream_t);
template void launchRLEDecompressKernel<int32_t>(const int32_t*, const uint32_t*, const uint32_t*, int32_t*, uint32_t, cudaStream_t);

} // namespace fz
