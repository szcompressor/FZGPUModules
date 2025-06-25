#pragma once

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <cuda_runtime.h>

using namespace cuda::experimental::stf;

__global__ void kernel_hist_sparse(
    slice<const uint16_t> quant_codes, size_t const data_len,
    slice<uint32_t> hist, uint16_t const hist_len, int chunk, int offset = 0) {

  constexpr int K = 5;
  constexpr auto R = (K - 1) / 2;

  extern __shared__ uint32_t s_hist[];

  uint32_t p_hist[K] = {0};

  auto global_id = [&](auto i) { return blockIdx.x * chunk + i; };
  // auto nworker = [&]() { return blockDim.x; };

  for (auto i = threadIdx.x; i < hist_len; i += blockDim.x) s_hist[i] = 0;
  __syncthreads();

  for (auto i = threadIdx.x; i < chunk; i += blockDim.x) {
    auto gid = global_id(i);
    if (gid < data_len) {
      auto ori = (int)quant_codes(gid);
      auto sym = ori - offset;

      if (2 * abs(sym) < K) {
        p_hist[sym + R] += 1;
      } else {
        atomicAdd(&s_hist[ori], 1);
      }
    }
    __syncthreads();
  }
  __syncthreads();

  for (auto& sum : p_hist) {
    for (auto d = 1; d < 32; d *= 2) {
      auto n = __shfl_up_sync(0xffffffff, sum, d);
      if (threadIdx.x % 32 >= d) sum += n;
    }
  }

  for (auto i = 0; i < K; i++) {
    if (threadIdx.x % 32 == 31) atomicAdd(&s_hist[(int)offset + i - R], p_hist[i]);
  }
  __syncthreads();

  for (auto i = threadIdx.x; i < hist_len; i += blockDim.x) atomicAdd(hist.data_handle() + i, s_hist[i]);
  __syncthreads();
}


