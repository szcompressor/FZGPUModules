#include <cuda_runtime.h>
#include <limits>
#include <cstddef>
#include <stdexcept>

using namespace cuda::experimental::stf;

#define MIN(a, b) ((a) < (b)) ? (a) : (b)
const static unsigned int WARP_SIZE = 32;

__global__ void kernel_hist_naive(slice<const uint16_t> in_data,
                                  size_t const data_len, 
                                  slice<uint32_t> hist,
                                  uint16_t const bins_len,
                                  uint16_t const repeat) {
  auto i = blockDim.x * blockIdx.x + threadIdx.x;
  auto j = 0u;
  if (i * repeat < data_len) {  // if there is a symbol to count,
    for (j = i * repeat; j < (i + 1) * repeat; j++) {
      if (j < data_len) {
        auto item = in_data(j);         // symbol to count
        atomicAdd(&hist.data_handle()[item], 1);  // update bin count by 1
      }
    }
  }
} 

__global__ void kernel_hist_generic(slice<const uint16_t> quant_codes,
                                    size_t const data_len, 
                                    slice<uint32_t> hist,
                                    uint16_t const hist_len,
                                    int const r_per_block) {
  extern __shared__ int Hs[];

  const unsigned int warp_id = (int)(threadIdx.x / WARP_SIZE);
  const unsigned int lane = threadIdx.x % WARP_SIZE;
  const unsigned int warps_block = blockDim.x / WARP_SIZE;
  const unsigned int off_rep = (hist_len + 1) * (threadIdx.x % r_per_block);
  const unsigned int begin =
      (data_len / warps_block) * warp_id + WARP_SIZE * blockIdx.x + lane;
  unsigned int end = (data_len / warps_block) * (warp_id + 1);
  const unsigned int step = WARP_SIZE * gridDim.x;

  if (warp_id >= warps_block - 1) end = data_len;

  for (unsigned int pos = threadIdx.x; pos < (hist_len + 1) * r_per_block; pos += blockDim.x) {
    Hs[pos] = 0;
  }
  __syncthreads();

  for (unsigned int i = begin; i < end; i += step) {
    int d = quant_codes(i);
    d = d <= 0 and d >= hist_len ? hist_len / 2 : d;
    atomicAdd(&Hs[off_rep + d], 1);
  }
  __syncthreads();

  for (unsigned int pos = threadIdx.x; pos < hist_len; pos += blockDim.x) {
    int sum = 0;
    for (int base = 0; base < (hist_len + 1) * r_per_block; base += hist_len + 1) {
      sum += Hs[base + pos];
    }
    atomicAdd(hist.data_handle() + pos, sum);
  }
}

void histogram_optimizer(size_t const data_len, 
                         uint16_t const hist_len,
                         int& grid_dim, 
                         int& block_dim, 
                         int& shmem_use,
                         int& r_per_block) 
{
  int device_id, max_bytes, num_SMs;
  int items_per_thread;

  cudaGetDevice(&device_id);
  cudaDeviceGetAttribute(&num_SMs, cudaDevAttrMultiProcessorCount, device_id);

  int max_bytes_opt_in;
  cudaDeviceGetAttribute(&max_bytes, cudaDevAttrMaxSharedMemoryPerBlock,
                         device_id);

  // account for opt-in extra shared mem on some architectures
  cudaDeviceGetAttribute(&max_bytes_opt_in, cudaDevAttrMaxSharedMemoryPerBlockOptin,device_id);
  max_bytes = std::max(max_bytes, max_bytes_opt_in);

  cudaFuncSetAttribute((void*)kernel_hist_generic, (cudaFuncAttribute)cudaFuncAttributeMaxDynamicSharedMemorySize, max_bytes);

  // optimize launch
  items_per_thread = 1;
  r_per_block = (max_bytes / sizeof(int) / (hist_len + 1));
  grid_dim = num_SMs;

  // fits to size
  block_dim = ((((data_len / (grid_dim * items_per_thread)) + 1) / 64) + 1) * 64;
  while (block_dim > 1024) {
    if (r_per_block <= 1) { block_dim = 1024; }
    else {
      r_per_block /= 2;
      grid_dim *= 2;
      block_dim = ((((data_len / (grid_dim * items_per_thread)) + 1) / 64) + 1) * 64;
    }
  }
  shmem_use = ((hist_len + 1) * r_per_block) * sizeof(int);
}