#include "encoders/diff/diff.h"
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include "mem/mempool.h"
#include "cuda_check.h"

namespace fz {

// Forward: Difference coding (compression)
template<typename T>
__global__ void differenceKernel(const T* input, T* output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx == 0 && n > 0) {
        // First element unchanged
        output[0] = input[0];
    } else if (idx < n) {
        // Compute difference from previous element
        output[idx] = input[idx] - input[idx - 1];
    }
}

// Explicit kernel launcher wrappers (these force kernel instantiation)
template<typename T>
void launchDifferenceKernel(const T* input, T* output, size_t n, int grid_size, int block_size, cudaStream_t stream) {
    differenceKernel<T><<<grid_size, block_size, 0, stream>>>(input, output, n);
}

template<typename T>
void launchUndoDifferenceKernel(const T* input, T* output, size_t n, cudaStream_t stream, MemoryPool* pool) {
    // Parallel inclusive prefix sum via CUB DeviceScan.
    // All allocations and launches are stream-ordered — no host sync required.
    int n_int = static_cast<int>(n);

    // Step 1: query required temp storage (d_temp = nullptr signals a size query)
    size_t temp_bytes = 0;
    cub::DeviceScan::InclusiveSum(nullptr, temp_bytes, input, output, n_int, stream);

    // Step 2: allocate temp storage (freed before function returns, same stream)
    void* d_temp = nullptr;
    if (pool) {
        d_temp = pool->allocate(temp_bytes, stream, "diff_cub_temp");
    } else {
        FZ_CUDA_CHECK(cudaMallocAsync(&d_temp, temp_bytes, stream));
    }

    // Step 3: run the scan
    cub::DeviceScan::InclusiveSum(d_temp, temp_bytes, input, output, n_int, stream);

    // Step 4: free temp storage (stream-ordered — safe to issue immediately)
    if (pool) { pool->free(d_temp, stream); } else { FZ_CUDA_CHECK_WARN(cudaFreeAsync(d_temp, stream)); }
}

template<typename T>
void DifferenceStage<T>::execute(
    cudaStream_t stream,
    MemoryPool* pool,
    const std::vector<void*>& inputs,
    const std::vector<void*>& outputs,
    const std::vector<size_t>& sizes
) {
    if (inputs.empty() || outputs.empty() || sizes.empty()) {
        throw std::runtime_error("DifferenceStage: Invalid inputs/outputs");
    }
    
    size_t byte_size = sizes[0];
    size_t n = byte_size / sizeof(T);
    
    if (n == 0) {
        actual_output_size_ = 0;
        return;
    }
    
    if (is_inverse_) {
        // Decompression: cumulative sum
        launchUndoDifferenceKernel<T>(
            static_cast<const T*>(inputs[0]),
            static_cast<T*>(outputs[0]),
            n,
            stream,
            pool
        );
    } else {
        // Compression: difference coding
        const int block_size = 256;
        const int grid_size = (n + block_size - 1) / block_size;
        
        launchDifferenceKernel<T>(
            static_cast<const T*>(inputs[0]),
            static_cast<T*>(outputs[0]),
            n,
            grid_size,
            block_size,
            stream
        );
    }
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("DifferenceStage kernel launch failed: ") + 
            cudaGetErrorString(err)
        );
    }
    
    actual_output_size_ = byte_size;
}

// Explicit template instantiations for both class and kernel launcher
template class DifferenceStage<float>;
template class DifferenceStage<double>;
template class DifferenceStage<int32_t>;
template class DifferenceStage<int64_t>;
template class DifferenceStage<uint16_t>;
template class DifferenceStage<uint8_t>;
template class DifferenceStage<uint32_t>;

// Explicitly instantiate kernel launchers
template void launchDifferenceKernel<float>(const float*, float*, size_t, int, int, cudaStream_t);
template void launchDifferenceKernel<double>(const double*, double*, size_t, int, int, cudaStream_t);
template void launchDifferenceKernel<int32_t>(const int32_t*, int32_t*, size_t, int, int, cudaStream_t);
template void launchDifferenceKernel<int64_t>(const int64_t*, int64_t*, size_t, int, int, cudaStream_t);
template void launchDifferenceKernel<uint16_t>(const uint16_t*, uint16_t*, size_t, int, int, cudaStream_t);
template void launchDifferenceKernel<uint8_t>(const uint8_t*, uint8_t*, size_t, int, int, cudaStream_t);

template void launchUndoDifferenceKernel<float>(const float*, float*, size_t, cudaStream_t, MemoryPool*);
template void launchUndoDifferenceKernel<double>(const double*, double*, size_t, cudaStream_t, MemoryPool*);
template void launchUndoDifferenceKernel<int32_t>(const int32_t*, int32_t*, size_t, cudaStream_t, MemoryPool*);
template void launchUndoDifferenceKernel<int64_t>(const int64_t*, int64_t*, size_t, cudaStream_t, MemoryPool*);
template void launchUndoDifferenceKernel<uint16_t>(const uint16_t*, uint16_t*, size_t, cudaStream_t, MemoryPool*);
template void launchUndoDifferenceKernel<uint8_t>(const uint8_t*, uint8_t*, size_t, cudaStream_t, MemoryPool*);
template void launchUndoDifferenceKernel<uint32_t>(const uint32_t*, uint32_t*, size_t, cudaStream_t, MemoryPool*);

} // namespace fz
