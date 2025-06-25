#pragma once

#include <cuda_runtime.h>

using namespace cuda::experimental::stf;

template <typename T>
__global__ void spvn_scatter(
    slice<const T> val, slice<const uint32_t> idx, int const nnz, slice<T> out) {
  auto tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < nnz) {
    int dst_idx = idx(tid);
    out(dst_idx) = val(tid);
  }
}