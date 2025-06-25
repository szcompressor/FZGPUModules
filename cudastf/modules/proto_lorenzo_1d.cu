#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <stdexcept>

using namespace cuda::experimental::stf;

void lorenzo_optimizer(dim3& grid1D, dim3& block1D, size_t data_len) {

  auto divide_3 = [](dim3 len, dim3 sublen) {
    return dim3(
        (len.x - 1) / sublen.x + 1,
        (len.y - 1) / sublen.y + 1,
        (len.z - 1) / sublen.z + 1);
  };

  block1D = dim3(256, 1, 1);
  grid1D = divide_3(dim3(data_len, 1, 1), block1D);
}

__global__ void kernel_proto_lorenzo_1d(slice<float> input,
                                  size_t input_size,
                                  slice<uint16_t> quant_codes,
                                  slice<float> outlier_vals,
                                  slice<uint32_t> outlier_idxs,
                                  slice<uint32_t> outlier_num,
                                  double ebx2,
                                  double ebx2_r,
                                  size_t radius = 512) 
{
  __shared__ float buf[256];

  auto id = threadIdx.x + blockIdx.x * 256;
  auto data = [&](auto dx) -> float& { return buf[threadIdx.x + dx]; };

  // prequant
  if (id < input_size) { data(0) = round(input(id) * ebx2_r); }
  __syncthreads();

  // quantization
  float delta = data(0) - (threadIdx.x == 0 ? 0 : data(-1));
  bool quantizable = fabs(delta) < radius;
  float candidate = delta + radius;
  if (id < input_size) {
    quant_codes[id] = quantizable * static_cast<uint16_t>(candidate);
    if (not quantizable) {
      auto curr_idx = atomicAdd(outlier_num.data_handle(), 1);
      outlier_idxs(curr_idx) = id;
      outlier_vals(curr_idx) = candidate;
    }
  }
}