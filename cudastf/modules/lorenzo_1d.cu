#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <stdexcept>

#include "detail/wave32.cu"

using namespace cuda::experimental::stf;

#define COUNT_LOCAL_STAT(DELTA, IS_VALID_RANGE)           \
  int is_zero = IS_VALID_RANGE ? (DELTA == 0) : 0;        \
  unsigned int mask = __ballot_sync(0xffffffff, is_zero); \
  if (threadIdx.x % 32 == 0) thp_top1_count += __popc(mask);


template <int dim, int X = 0, int Y = 0>
struct c_lorenzo;

template <>
struct c_lorenzo<1> {
  static constexpr dim3 tile = dim3(1024, 1, 1);
  static constexpr dim3 sequentiality = dim3(4, 1, 1);  // x-sequentiality == 4
  static constexpr dim3 thread_block = dim3(1024 / 4, 1, 1);
  static dim3 thread_grid(dim3 len3) { return div3(len3, tile); };
};

template <int dim, int X = 0, int Y = 0>
struct x_lorenzo;

template <>
struct x_lorenzo<1> {
  static constexpr dim3 tile = dim3(1024, 1, 1);
  static constexpr dim3 sequentiality = dim3(4, 1, 1);  // x-sequentiality == 8
  static constexpr dim3 thread_block = dim3(1024 / 4, 1, 1);
  static dim3 thread_grid(dim3 len3) { return div3(len3, tile); };
};

template <typename T, int TileDim, int Seq, bool UseLocalStat = true, int UseGlobalStat = true>
__global__ void kernel_lorenzo_1d(
    slice<T> input,
    size_t input_size,
    slice<uint16_t> quant_codes,
    slice<T> outlier_vals,
    slice<uint32_t> outlier_idxs,
    slice<uint32_t> outlier_num,
    slice<uint32_t> out_top1,
    double ebx2,
    double ebx2_r,
    size_t radius) 
{
  constexpr auto NumThreads = TileDim / Seq;

  __shared__ uint32_t s_top1_counts[1];
  if (threadIdx.x == 0) s_top1_counts[0] = 0;

  __shared__ T s_data[TileDim];
  __shared__ uint16_t s_eq_uint[TileDim];

  T _thp_data[Seq + 1] = {0};
  auto prev = [&]() -> T& { return _thp_data[0]; };
  auto thp_data = [&](auto i) -> T& { return _thp_data[i + 1]; };

  auto const id_base = blockIdx.x * TileDim;

  #pragma unroll
  for (auto ix = 0; ix < Seq; ix++) {
    auto id = id_base + threadIdx.x + ix * NumThreads;
    if (id < input_size) s_data[threadIdx.x + ix * NumThreads] = round(input(id) * ebx2_r);
  }

  __syncthreads();

  #pragma unroll
  for (auto ix = 0; ix < Seq; ix++) thp_data(ix) = s_data[threadIdx.x * Seq + ix];
  if (threadIdx.x > 0) prev() = s_data[threadIdx.x * Seq - 1];
  __syncthreads();

  uint32_t thp_top1_count{0};

  #pragma unroll
  for (auto ix = 0; ix < Seq; ix++) {
    T delta = thp_data(ix) - thp_data(ix - 1);
    bool quantizable = fabs(delta) < radius;

    if constexpr(UseLocalStat) {
      bool is_valid_range = id_base + threadIdx.x * Seq + ix < input_size;
      COUNT_LOCAL_STAT(delta, is_valid_range);
    }

    T candidate;
    candidate = delta + radius;
    s_eq_uint[threadIdx.x * Seq + ix] = quantizable * (uint16_t)candidate;

    if (not quantizable) {
      auto global_idx = id_base + threadIdx.x * Seq + ix;
      if (global_idx < input_size) {  // Add this check
        auto cur_idx = atomicAdd(outlier_num.data_handle(), 1);
        outlier_idxs(cur_idx) = global_idx;
        outlier_vals(cur_idx) = candidate;
      }
    }
  }
  __syncthreads();

  if constexpr(UseLocalStat) {
    if (threadIdx.x % 32 == 0) atomicAdd(s_top1_counts, thp_top1_count);
    __syncthreads();

    if constexpr(UseGlobalStat) {
      if (threadIdx.x == 0) atomicAdd(out_top1.data_handle(), s_top1_counts[0]);
    }
  }

  #pragma unroll
  for (auto ix = 0; ix < Seq; ix++) {
    auto id = id_base + threadIdx.x + ix * NumThreads;
    if (id < input_size) quant_codes(id) = s_eq_uint[threadIdx.x + ix * NumThreads];
  }
}

template <typename T, int TileDim, int Seq>
__global__ void kernel_decomp_lorenzo_1d(
  slice<uint16_t> in_eq,
  slice<T> out_data,
  size_t input_size,
  double ebx2,
  double ebx2_r,
  uint16_t radius
) {
  constexpr auto NTHREAD = TileDim / Seq;

  __shared__ T scratch[TileDim];
  __shared__ T exch_in[NTHREAD / 32];
  __shared__ T exch_out[NTHREAD / 32];

  T thp_data[Seq];

  auto id_base = blockIdx.x * TileDim;

  auto load_fuse_1d = [&]() {
    #pragma unroll
    for (auto i = 0; i < Seq; i++) {
      auto local_id = threadIdx.x + i * NTHREAD;
      auto id = id_base + local_id;
      if (id < input_size) {
        scratch[local_id] = out_data(id) + static_cast<T>(in_eq(id)) - radius;
      }
    }
    __syncthreads();

    #pragma unroll
    for (auto i = 0; i < Seq; i++) thp_data[i] = scratch[threadIdx.x * Seq + i];
    __syncthreads();
  };

  auto block_scan_1d = [&]() {
    fz::SUBR_CUHIP_WAVE32_intrawarp_inclscan_1d<T, Seq>(thp_data);
    fz::SUBR_CUHIP_WAVE32_intrablock_exclscan_1d<T, Seq, NTHREAD>(thp_data, exch_in, exch_out);

    #pragma unroll
    for (auto i = 0; i < Seq; i++) scratch[threadIdx.x * Seq + i] = thp_data[i] * ebx2;
    __syncthreads();
  };

  auto write_1d = [&]() {
    #pragma unroll
    for (auto i = 0; i < Seq; i++) {
      auto local_id = threadIdx.x + i * NTHREAD;
      auto id = id_base + local_id;
      if (id < input_size) out_data(id) = scratch[local_id];
    }
  };

  load_fuse_1d();
  block_scan_1d();
  write_1d();
}

template <typename T>
void lorenzo_1d(
    slice<T> input,
    size_t input_size,
    slice<uint16_t> quant_codes,
    slice<T> outlier_vals,
    slice<uint32_t> outlier_idxs,
    slice<uint32_t> outlier_num,
    slice<uint32_t> out_top1,
    double ebx2,
    double ebx2_r,
    size_t radius,
    cudaStream_t stream) 
{
  kernel_lorenzo_1d<T, c_lorenzo<1>::tile.x, c_lorenzo<1>::sequentiality.x>
      <<<c_lorenzo<1>::thread_grid(input_size), c_lorenzo<1>::thread_block, 0, stream>>>(input, input_size, quant_codes, outlier_vals, outlier_idxs, outlier_num, out_top1, ebx2, (T)ebx2_r, radius);
}

template <typename T>
void lorenzo_decomp_1d(
  slice<uint16_t> in_eq,
  slice<T> out_data,
  size_t input_size,
  double ebx2,
  double ebx2_r,
  uint16_t radius,
  cudaStream_t stream
) {
  kernel_decomp_lorenzo_1d<T, x_lorenzo<1>::tile.x, x_lorenzo<1>::sequentiality.x>
    <<<x_lorenzo<1>::thread_grid(input_size), x_lorenzo<1>::thread_block, 0, stream>>>(
      in_eq, out_data, input_size, ebx2, ebx2_r, radius
    );
}