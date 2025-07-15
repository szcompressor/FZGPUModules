#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <stdexcept>

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#include "detail/wave32.cu"
#include "detail/zigzag.hh"
#include "lrz.gpu_config.hh"

using namespace cuda::experimental::stf;

#define COUNT_LOCAL_STAT(DELTA, IS_VALID_RANGE)           \
  int is_zero = IS_VALID_RANGE ? (DELTA == 0) : 0;        \
  unsigned int mask = __ballot_sync(0xffffffff, is_zero); \
  if (threadIdx.x % 32 == 0) thp_top1_count += __popc(mask);

#define SETUP_ZIGZAG                    \
using ZigZag = fz::ZigZag<uint16_t>;       \
using EqUInt = typename ZigZag::UInt; \
using EqSInt = typename ZigZag::SInt;

#define Z(LEN3) LEN3[2]
#define Y(LEN3) LEN3[1]
#define X(LEN3) LEN3[0]
#define TO_DIM3(LEN3) dim3(X(LEN3), Y(LEN3), Z(LEN3))

template <typename T, bool UseZigZag, class Perf, 
  bool UseLocalStat = true, int UseGlobalStat = true>
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
  SETUP_ZIGZAG;

  constexpr auto TileDim = Perf::TileDim;
  constexpr auto Seq = Perf::Seq;
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

    if constexpr(UseZigZag) {
      candidate = delta;
      s_eq_uint[threadIdx.x * Seq + ix] = 
        ZigZag::encode(static_cast<EqSInt>(quantizable * candidate));
    } else {
      candidate = delta + radius;
      s_eq_uint[threadIdx.x * Seq + ix] = quantizable * static_cast<EqUInt>(candidate);
    }

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

template <typename T, bool UseZigZag, class Perf,
  bool UseLocalStat = true, int UseGlobalStat = true>
__global__ void kernel_lorenzo_2d(
    slice<T> input,
    size_t x_len,
    size_t y_len,
    dim3 const leap3,
    slice<uint16_t> quant_codes,
    slice<T> outlier_vals,
    slice<uint32_t> outlier_idxs,
    slice<uint32_t> outlier_num,
    slice<uint32_t> out_top1,
    double ebx2,
    double ebx2_r,
    size_t radius) 
{
  SETUP_ZIGZAG;
  
  constexpr auto TileDim = Perf::TileDim;
  constexpr auto Yseq = Perf::SeqY;
  constexpr auto NumWarps = 4;
  static_assert(NumWarps == TileDim * TileDim / Yseq / 32, "wrong TileDim");

  __shared__ uint32_t s_top1_counts[1];
  if (cg::this_thread_block().thread_rank() == 0) s_top1_counts[0] = 0;

  __shared__ T exchange[NumWarps - 1][TileDim + 1];

  T center[Yseq + 1] = {0};

  auto gix = blockIdx.x * TileDim + threadIdx.x;
  auto giy_base = blockIdx.y * TileDim + threadIdx.y * Yseq;
  auto g_id = [&](auto i) { return (giy_base + i) * leap3.y + gix; };

  #pragma unroll
  for (auto iy = 0; iy < Yseq; iy++) {
    if (gix < x_len and giy_base + iy < y_len) {
      center[iy + 1] = round(input(g_id(iy)) * ebx2_r);
    }
  }
  if (threadIdx.y < NumWarps - 1) exchange[threadIdx.y][threadIdx.x] = center[Yseq];
  __syncthreads();
  if (threadIdx.y > 0) center[0] = exchange[threadIdx.y - 1][threadIdx.x];
  __syncthreads();

  uint32_t thp_top1_count{0};

  #pragma unroll
  for (auto i = Yseq; i > 0; i--) {
    // 1) prediction (apply lorenzo filter)
    center[i] -= center[i - 1];
    auto west = __shfl_up_sync(0xffffffff, center[i], 1, 32);
    if (threadIdx.x > 0) center[i] -= west;

    // 2) store quant codes
    auto gid = g_id(i - 1);

    bool quantizable = fabs(center[i]) < radius;

    bool is_valid_range = (gix < x_len and (giy_base + i - 1) < y_len);

    if constexpr (UseLocalStat) { COUNT_LOCAL_STAT(center[i], is_valid_range); }

    T candidate;

    if constexpr (UseZigZag) {
      candidate = center[i];
      if (is_valid_range)
        quant_codes(gid) = ZigZag::encode(static_cast<EqSInt>(quantizable * candidate));
    } else {
      candidate = center[i] + radius;
      if (is_valid_range) quant_codes(gid) = quantizable * (EqUInt)candidate;
    }

    if (not quantizable) {
      if (gix < x_len and (giy_base + i - 1) < y_len) {
        auto cur_idx = atomicAdd(outlier_num.data_handle(), 1);
        outlier_idxs(cur_idx) = gid;
        outlier_vals(cur_idx) = candidate;
      }
    }
  }

  if constexpr (UseLocalStat) {
    if (cg::this_thread_block().thread_rank() % 32 == 0) atomicAdd(s_top1_counts, thp_top1_count);
    __syncthreads();

    if constexpr (UseGlobalStat) {
      if (cg::this_thread_block().thread_rank() == 0) {
        atomicAdd(out_top1.data_handle(), s_top1_counts[0]);
      }
    }
  }

} // kernel_lorenzo_2d

template <typename T, bool UseZigZag, class Perf,
  bool UseLocalStat = true, int UseGlobalStat = true>
__global__ void kernel_lorenzo_3d(
    slice<T> input,
    size_t x_len,
    size_t y_len,
    size_t z_len,
    dim3 const leap3,
    slice<uint16_t> quant_codes,
    slice<T> outlier_vals,
    slice<uint32_t> outlier_idxs,
    slice<uint32_t> outlier_num,
    slice<uint32_t> out_top1,
    double ebx2,
    double ebx2_r,
    size_t radius) 
{

  constexpr auto TileDim = Perf::TileDim;
  SETUP_ZIGZAG;

  __shared__ uint32_t s_top1_counts[1];
  if (cg::this_thread_block().thread_rank() == 0) s_top1_counts[0] = 0;

  __shared__ T s[9][33];

  T delta[TileDim + 1] = {0};

  const auto gix = blockIdx.x * (TileDim * 4) + threadIdx.x;
  const auto giy = blockIdx.y * TileDim + threadIdx.y;
  const auto giz_base = blockIdx.z * TileDim;
  const auto base_id = gix + giy * leap3.y + giz_base * leap3.z;

  auto giz = [&](auto z) { return giz_base + z; };
  auto gid = [&](auto z) { return base_id + z * leap3.z; };

  auto load_prequant_3d = [&]() {
    if (gix < x_len and giy < y_len) {
      for (auto z = 0; z < TileDim; z++) {
        if (giz(z) < z_len) {
          delta[z + 1] = round(input(gid(z)) * ebx2_r);
        }
      }
    }
    __syncthreads();
  };

  auto quantize_compact_write = [&](T delta, auto x, auto y, auto z, auto gid) {
    bool quantizable = fabs(delta) < radius;

    if (x < x_len and y < y_len and z < z_len) {
      T candidate;

      if constexpr (UseZigZag) {
        candidate = delta;
        quant_codes(gid) = ZigZag::encode(static_cast<EqSInt>(quantizable * candidate));
      } else {
        candidate = delta + radius;
        quant_codes(gid) = quantizable * (EqUInt)candidate;
      }

      if (not quantizable) {
        auto cur_idx = atomicAdd(outlier_num.data_handle(), 1);
        outlier_idxs(cur_idx) = gid;
        outlier_vals(cur_idx) = candidate;
      }
    }
  };

  load_prequant_3d();

  uint32_t thp_top1_count{0};

  for (auto z = TileDim; z > 0; z--) {
    // z-direction
    delta[z] -= delta[z - 1];

    // x-direction
    auto prev_x = __shfl_up_sync(0xffffffff, delta[z], 1, 8);
    if (threadIdx.x % TileDim > 0) delta[z] -= prev_x;

    // y-direction
    s[threadIdx.y + 1][threadIdx.x] = delta[z];
    __syncthreads();

    delta[z] -= (threadIdx.y > 0) * s[threadIdx.y][threadIdx.x];

    if constexpr (UseLocalStat) {
      auto if_valid_range = (gix < x_len and giy < y_len and giz(z - 1) < z_len);
      COUNT_LOCAL_STAT(delta[z], if_valid_range);
    }

    quantize_compact_write(delta[z], gix, giy, giz(z - 1), gid(z - 1));
    __syncthreads();
  }

  if constexpr (UseLocalStat) {
    if (cg::this_thread_block().thread_rank() % 32 == 0) atomicAdd(s_top1_counts, thp_top1_count);
    __syncthreads();

    if constexpr (UseGlobalStat) {
      if (cg::this_thread_block().thread_rank() == 0) {
        atomicAdd(out_top1.data_handle(), s_top1_counts[0]);
      }
    }
  }

} // kernel_lorenzo_3d

// ###########################################

template <typename T, bool UseZigZag, class Perf>
__global__ void kernel_decomp_lorenzo_1d(
  slice<uint16_t> in_eq,
  slice<T> out_data,
  size_t input_size,
  double ebx2,
  double ebx2_r,
  uint16_t radius
) {
  constexpr auto TileDim = Perf::TileDim;
  constexpr auto Seq = Perf::Seq;
  constexpr auto NTHREAD = TileDim / Seq;  // equiv. to blockDim.x

  SETUP_ZIGZAG

  __shared__ T scratch[TileDim];
  // __shared__ uint16_t s_eq[TileDim];
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
        if constexpr (UseZigZag) {
          auto e = in_eq(id);
          scratch[local_id] = out_data(id) + 
            static_cast<T>(ZigZag::decode(static_cast<EqUInt>(e)));
        } else {
          scratch[local_id] = out_data(id) + static_cast<T>(in_eq(id)) - radius;
        }
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


template <typename T, bool UseZigZag, class Perf>
__global__ void kernel_decomp_lorenzo_2d(
  slice<uint16_t> in_eq,
  slice<T> out_data,
  size_t len_x,
  size_t len_y,
  dim3 const leap3,
  double ebx2,
  double ebx2_r,
  uint16_t radius
) {
  SETUP_ZIGZAG
  constexpr auto TileDim = Perf::TileDim;
  constexpr auto NumWarps = 4;
  constexpr auto Yseq = TileDim / NumWarps;

  static_assert(Perf::SeqY == Yseq, "wrong SeqY");

  __shared__ T scratch[NumWarps - 1][TileDim + 1];
  T thp_data[Yseq] = {0};

  auto gix = blockIdx.x * TileDim + threadIdx.x;
  auto giy_base = blockIdx.y * TileDim + threadIdx.y * Yseq;
  auto get_gid = [&](auto i) { return (giy_base + i) * leap3.y + gix; };

  auto load_fuse_2d = [&]() {
    #pragma unroll
    for (auto i = 0; i < Yseq; i++) {
      auto gid = get_gid(i);
      if (gix < len_x and (giy_base + i) < len_y) {
        if constexpr (UseZigZag) {
          auto e = in_eq(gid);
          thp_data[i] = out_data(gid) + static_cast<T>(ZigZag::decode(static_cast<EqUInt>(e)));
        } else {
          thp_data[i] = out_data(gid) + static_cast<T>(in_eq(gid)) - radius;
        }
      }
    }
  };

  auto block_scan_2d = [&]() {
    for (auto i = 1; i < Yseq; i++) thp_data[i] += thp_data[i - 1];

    if (threadIdx.y < NumWarps - 1) scratch[threadIdx.y][threadIdx.x] = thp_data[Yseq - 1];
    __syncthreads();

    if (threadIdx.y == 0) {
      T warp_accum[NumWarps - 1];
      #pragma unroll
      for (auto i = 0; i < NumWarps - 1; i++) {
        warp_accum[i] = scratch[i][threadIdx.x];
      }
      #pragma unroll
      for (auto i =1; i < NumWarps; i++) {
        warp_accum[i] += warp_accum[i - 1];
      }
      #pragma unroll
      for (auto i = 1; i < NumWarps; i++) {
        scratch[i][threadIdx.x] = warp_accum[i];
      }
    }
    __syncthreads();

    if (threadIdx.y > 0) {
      auto addend = scratch[threadIdx.y - 1][threadIdx.x];
      #pragma unroll
      for (auto i = 0; i < Yseq; i++) thp_data[i] += addend;
    }
    __syncthreads();

    #pragma unroll
    for (auto i = 0; i < Yseq; i++) {
      for (auto d = 1; d < TileDim; d *= 2) {
        T n = __shfl_up_sync(0xffffffff, thp_data[i], d, 32);
        if (threadIdx.x >= d) thp_data[i] += n;
      }
      thp_data[i] *= ebx2;
    }
  };

  auto decomp_write_2d = [&]() {
    #pragma unroll
    for (auto i = 0; i < Yseq; i++) {
      auto gid = get_gid(i);
      if (gix < len_x and (giy_base + i) < len_y) {
        out_data(gid) = thp_data[i];
      }
    }
  };

  load_fuse_2d();
  block_scan_2d();
  decomp_write_2d();
}

template <typename T, bool UseZigZag, class Perf>
__global__ void kernel_decomp_lorenzo_3d(
  slice<uint16_t> in_eq,
  slice<T> out_data,
  size_t len_x,
  size_t len_y,
  size_t len_z,
  dim3 const leap3,
  double ebx2,
  double ebx2_r,
  uint16_t radius
) {
  SETUP_ZIGZAG
  constexpr auto TileDim = 8;
  constexpr auto YSEQ = TileDim;

  __shared__ T scratch[TileDim][4][8];
  T thread_private[YSEQ] = {0};

  auto seg_id = threadIdx.x / 8;
  auto seg_tix = threadIdx.x % 8;

  auto gix = blockIdx.x * (4 * TileDim) + threadIdx.x;
  auto giy_base = blockIdx.y * TileDim;
  auto giy = [&](auto y) { return giy_base + y; };
  auto giz = blockIdx.z * TileDim + threadIdx.z;
  auto gid = [&](auto y) { return giz * leap3.z + (giy_base + y) * leap3.y + gix; };

  auto load_fuse_3d = [&]() {
    #pragma unroll
    for (auto y = 0; y < YSEQ; y++) {
      if (gix < len_x and giy_base + y < len_y and giz < len_z) {
        if constexpr (UseZigZag) {
          auto e = in_eq(gid(y));
          thread_private[y] = out_data(gid(y)) + 
            static_cast<T>(ZigZag::decode(static_cast<EqUInt>(e)));
        } else {
          thread_private[y] = out_data(gid(y)) + static_cast<T>(in_eq(gid(y))) - radius;
        }
      }
    }
  };

  auto block_scan_3d = [&]() {
    for (auto y = 1; y < YSEQ; y++) thread_private[y] += thread_private[y - 1];

    #pragma unroll
    for (auto i = 0; i < TileDim; i++) {
      T val = thread_private[i];
      for (auto dist = 1; dist < TileDim; dist *= 2) {
        auto addend = __shfl_up_sync(0xffffffff, val, dist, 8);
        if (seg_tix >= dist) val += addend;
      }

      scratch[threadIdx.z][seg_id][seg_tix] = val;
      __syncthreads();
      val = scratch[seg_tix][seg_id][threadIdx.z];
      __syncthreads();

      for (auto dist = 1; dist < TileDim; dist *= 2) {
        auto addend = __shfl_up_sync(0xffffffff, val, dist, 8);
        if (threadIdx.x >= dist) val += addend;
      }

      scratch[threadIdx.z][seg_id][seg_tix] = val;
      __syncthreads();
      val = scratch[seg_tix][seg_id][threadIdx.z];
      __syncthreads();

      thread_private[i] = val;
    }
  };

  auto decomp_write_3d = [&]() {
    #pragma unroll
    for (auto y = 0; y < YSEQ; y++) {
      if (gix < len_x and giy(y) < len_y and giz < len_z) {
        out_data(gid(y)) = thread_private[y] * ebx2;
      }
    }
  };

  load_fuse_3d();
  block_scan_3d();
  decomp_write_3d();
}

// ###########################################

template <typename T, bool UseZigZag>
void lorenzo_1d(
    slice<T> input,
    size_t x,
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
  using lrz1 = fz::kernelconfig::c_lorenzo<1>;

  kernel_lorenzo_1d<T, UseZigZag, lrz1::Perf>
      <<<lrz1::thread_grid(dim3(x, 1, 1)), lrz1::thread_block, 0, stream>>>
      (input, x, quant_codes, outlier_vals, outlier_idxs, outlier_num, 
      out_top1, ebx2, (T)ebx2_r, radius);
}

template <typename T, bool UseZigZag>
void lorenzo_2d(
    slice<T> input,
    size_t x,
    size_t y,
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
  std::array<size_t, 3> stdlen3 = {x, y, 1};
  auto dims = TO_DIM3(stdlen3);
  auto leap3 = dim3(1, dims.x, dims.x * dims.y);
  using lrz2 = fz::kernelconfig::c_lorenzo<2, 32, 32>;

  kernel_lorenzo_2d<T, UseZigZag, lrz2::Perf>
      <<<lrz2::thread_grid(dim3(x, y, 1)), lrz2::thread_block, 0, stream>>>(
      input, x, y, leap3, quant_codes, outlier_vals, outlier_idxs, outlier_num, 
      out_top1, ebx2, (T)ebx2_r, radius);
}

template <typename T, bool UseZigZag>
void lorenzo_3d(
    slice<T> input,
    size_t x,
    size_t y,
    size_t z,
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
  std::array<size_t, 3> stdlen3 = {x, y, z};
  auto dims = TO_DIM3(stdlen3);
  auto leap3 = dim3(1, dims.x, dims.x * dims.y);
  using lrz3 = fz::kernelconfig::c_lorenzo<3>;

  kernel_lorenzo_3d<T, UseZigZag, lrz3::Perf><<<lrz3::thread_grid(dims), 
    lrz3::thread_block, 0, stream>>>(input, x, y, z, leap3, quant_codes, 
    outlier_vals, outlier_idxs, outlier_num, out_top1, ebx2, (T)ebx2_r, radius);
}

template <typename T, bool UseZigZag>
void lorenzo_decomp_1d(
  slice<uint16_t> in_eq,
  slice<T> out_data,
  size_t input_size,
  double ebx2,
  double ebx2_r,
  uint16_t radius,
  cudaStream_t stream
) {
  using lrx1 = fz::kernelconfig::x_lorenzo<1>;
  kernel_decomp_lorenzo_1d<T, UseZigZag, lrx1::Perf>
    <<<lrx1::thread_grid(dim3(input_size, 1, 1)), lrx1::thread_block, 0, stream>>>(
      in_eq, out_data, input_size, ebx2, ebx2_r, radius
    );
}

template <typename T, bool UseZigZag>
void lorenzo_decomp_2d(
  slice<uint16_t> in_eq,
  slice<T> out_data,
  size_t len_x,
  size_t len_y,
  double ebx2,
  double ebx2_r,
  uint16_t radius,
  cudaStream_t stream
) {
  using lrx2 = fz::kernelconfig::x_lorenzo<2, 32>;
  std::array<size_t, 3> stdlen3 = {len_x, len_y, 1};
  auto dims = TO_DIM3(stdlen3);
  auto leap3 = dim3(1, dims.x, dims.x * dims.y);

  kernel_decomp_lorenzo_2d<T, UseZigZag, lrx2::Perf>
    <<<lrx2::thread_grid(dims), lrx2::thread_block, 0, stream>>>(
      in_eq, out_data, len_x, len_y, leap3, ebx2, ebx2_r, radius
    );
}

template <typename T, bool UseZigZag>
void lorenzo_decomp_3d(
  slice<uint16_t> in_eq,
  slice<T> out_data,
  size_t len_x,
  size_t len_y,
  size_t len_z,
  double ebx2,
  double ebx2_r,
  uint16_t radius,
  cudaStream_t stream
) {
  using lrx3 = fz::kernelconfig::x_lorenzo<3>;
  std::array<size_t, 3> stdlen3 = {len_x, len_y, len_z};
  auto dims = TO_DIM3(stdlen3);
  auto leap3 = dim3(1, dims.x, dims.x * dims.y);

  kernel_decomp_lorenzo_3d<T, UseZigZag, lrx3::Perf>
    <<<lrx3::thread_grid(dims), lrx3::thread_block, 0, stream>>>(
      in_eq, out_data, len_x, len_y, len_z, leap3, ebx2, ebx2_r, radius
    );
}