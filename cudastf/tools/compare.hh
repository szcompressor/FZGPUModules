#ifndef CUDASTF_COMPARE_HH
#define CUDASTF_COMPARE_HH

#include <cuda_runtime.h>
#include <math.h>

namespace fz {
namespace module {

// Calculate min, max, avg
template <typename T>
__global__ void extrema_kernel(T* data, size_t len, T* result) {
    __shared__ T s_min[256];
    __shared__ T s_max[256];
    __shared__ T s_sum[256];

    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    T thread_min = std::numeric_limits<T>::max();
    T thread_max = std::numeric_limits<T>::lowest();
    T thread_sum = 0;

    for (size_t i = tid; i < len; i += stride) {
        thread_min = min(thread_min, data[i]);
        thread_max = max(thread_max, data[i]);
        thread_sum += data[i];
    }

    s_min[threadIdx.x] = thread_min;
    s_max[threadIdx.x] = thread_max;
    s_sum[threadIdx.x] = thread_sum;
    __syncthreads();

    // Reduction
    for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_min[threadIdx.x] = min(s_min[threadIdx.x], s_min[threadIdx.x + s]);
            s_max[threadIdx.x] = max(s_max[threadIdx.x], s_max[threadIdx.x + s]);
            s_sum[threadIdx.x] += s_sum[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicMin((int*)&result[0], __float_as_int(s_min[0]));
        atomicMax((int*)&result[1], __float_as_int(s_max[0]));
        atomicAdd(&result[2], s_sum[0]);
    }
}

template <typename T>
void GPU_extrema(T* d_data, size_t len, T* h_result, cudaStream_t stream = 0) {
    // Initialize result with appropriate values
    T init_values[4] = {
        std::numeric_limits<T>::max(),     // min
        std::numeric_limits<T>::lowest(),  // max
        0,                                 // sum
        0                                  // unused
    };
    
    T* d_result;
    cudaMalloc(&d_result, 4 * sizeof(T));
    cudaMemcpy(d_result, init_values, 4 * sizeof(T), cudaMemcpyHostToDevice);
    
    int blockSize = 256;
    int gridSize = (len + blockSize - 1) / blockSize;
    gridSize = min(gridSize, 1024); // Limit grid size
    
    extrema_kernel<<<gridSize, blockSize, 0, stream>>>(d_data, len, d_result);
    
    cudaMemcpy(h_result, d_result, 4 * sizeof(T), cudaMemcpyDeviceToHost);
    h_result[2] /= len; // Calculate average
    
    cudaFree(d_result);
}

// Calculate errors between original and decompressed data
template <typename T>
__global__ void calculate_errors_kernel(
    T* orig_data, T orig_mean, 
    T* decomp_data, T decomp_mean, 
    size_t len, T* result) {
    
    __shared__ T s_corr[256];
    __shared__ T s_err_sq[256];
    __shared__ T s_var_orig[256];
    __shared__ T s_var_decomp[256];
    
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    T thread_corr = 0;
    T thread_err_sq = 0;
    T thread_var_orig = 0;
    T thread_var_decomp = 0;
    
    for (size_t i = tid; i < len; i += stride) {
        T o_val = orig_data[i];
        T d_val = decomp_data[i];
        T o_diff = o_val - orig_mean;
        T d_diff = d_val - decomp_mean;
        
        thread_corr += o_diff * d_diff;
        thread_err_sq += (o_val - d_val) * (o_val - d_val);
        thread_var_orig += o_diff * o_diff;
        thread_var_decomp += d_diff * d_diff;
    }
    
    s_corr[threadIdx.x] = thread_corr;
    s_err_sq[threadIdx.x] = thread_err_sq;
    s_var_orig[threadIdx.x] = thread_var_orig;
    s_var_decomp[threadIdx.x] = thread_var_decomp;
    __syncthreads();
    
    // Reduction
    for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_corr[threadIdx.x] += s_corr[threadIdx.x + s];
            s_err_sq[threadIdx.x] += s_err_sq[threadIdx.x + s];
            s_var_orig[threadIdx.x] += s_var_orig[threadIdx.x + s];
            s_var_decomp[threadIdx.x] += s_var_decomp[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        atomicAdd(&result[0], s_corr[0]);
        atomicAdd(&result[1], s_err_sq[0]);
        atomicAdd(&result[2], s_var_orig[0]);
        atomicAdd(&result[3], s_var_decomp[0]);
    }
}

template <typename T>
void GPU_calculate_errors(
    T* d_orig_data, T orig_mean, 
    T* d_decomp_data, T decomp_mean, 
    size_t len, T* h_result, 
    cudaStream_t stream = 0) {
    
    T* d_result;
    cudaMalloc(&d_result, 4 * sizeof(T));
    cudaMemset(d_result, 0, 4 * sizeof(T));
    
    int blockSize = 256;
    int gridSize = (len + blockSize - 1) / blockSize;
    gridSize = min(gridSize, 1024); // Limit grid size
    
    calculate_errors_kernel<<<gridSize, blockSize, 0, stream>>>(
        d_orig_data, orig_mean, d_decomp_data, decomp_mean, len, d_result);
    
    cudaMemcpy(h_result, d_result, 4 * sizeof(T), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
}

// Find maximum error and its location
template <typename T>
__global__ void find_max_error_kernel(
    T* decomp_data, T* orig_data, size_t len,
    T* max_err, size_t* max_idx) {
    
    __shared__ T s_max_err[256];
    __shared__ size_t s_max_idx[256];
    
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    T thread_max_err = 0;
    size_t thread_max_idx = 0;
    
    for (size_t i = tid; i < len; i += stride) {
        T err = fabs(decomp_data[i] - orig_data[i]);
        if (err > thread_max_err) {
            thread_max_err = err;
            thread_max_idx = i;
        }
    }
    
    s_max_err[threadIdx.x] = thread_max_err;
    s_max_idx[threadIdx.x] = thread_max_idx;
    __syncthreads();
    
    // Reduction to find max error
    for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (s_max_err[threadIdx.x] < s_max_err[threadIdx.x + s]) {
                s_max_err[threadIdx.x] = s_max_err[threadIdx.x + s];
                s_max_idx[threadIdx.x] = s_max_idx[threadIdx.x + s];
            }
        }
        __syncthreads();
    }
    
    // Update global max if needed
    if (threadIdx.x == 0) {
        float old_max;
        old_max = atomicMax(max_err, s_max_err[0]);
        if (s_max_err[0] > old_max) {
            // This is a bit of a race condition, but for our purposes it's fine
            // if we don't always get the exact index of the absolute maximum
            *max_idx = s_max_idx[0];
        }
    }
}

template <typename T>
void GPU_find_max_error(
    T* d_decomp_data, T* d_orig_data, size_t len,
    T& h_max_err, size_t& h_max_idx,
    cudaStream_t stream = 0) {
    
    T* d_max_err;
    size_t* d_max_idx;
    cudaMalloc(&d_max_err, sizeof(T));
    cudaMalloc(&d_max_idx, sizeof(size_t));
    
    cudaMemset(d_max_err, 0, sizeof(T));
    cudaMemset(d_max_idx, 0, sizeof(size_t));
    
    int blockSize = 256;
    int gridSize = (len + blockSize - 1) / blockSize;
    gridSize = min(gridSize, 1024); // Limit grid size
    
    find_max_error_kernel<<<gridSize, blockSize, 0, stream>>>(
        d_decomp_data, d_orig_data, len, d_max_err, d_max_idx);
    
    cudaMemcpy(&h_max_err, d_max_err, sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_max_idx, d_max_idx, sizeof(size_t), cudaMemcpyDeviceToHost);
    
    cudaFree(d_max_err);
    cudaFree(d_max_idx);
}

} // namespace module
} // namespace fz

#endif // CUDASTF_COMPARE_HH
