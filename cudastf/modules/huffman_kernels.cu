#pragma once

#include <cuda_runtime.h>
#include <limits>
#include <cstddef>
#include <stdexcept>

#include "hf_impl.hh"

using namespace cuda::experimental::stf;

#define TIX threadIdx.x
#define BIX blockIdx.x
#define BDX blockDim.x

extern __shared__ char __codec_raw[];

namespace {
  struct helper {
    __device__ __forceinline__ static unsigned int local_tid_1() {
      return threadIdx.x;
    }
    __device__ __forceinline__ static unsigned int global_tid_1() {
      return blockIdx.x * blockDim.x + threadIdx.x;
    }
    __device__ __forceinline__ static unsigned int block_stride_1() {
      return blockDim.x;
    }
    __device__ __forceinline__ static unsigned int grid_stride_1() {
      return blockDim.x * gridDim.x;
    }
    template <int SEQ>
    __device__ __forceinline__ static unsigned int global_tid() {
      return blockIdx.x * blockDim.x * SEQ + threadIdx.x;
    }
    template <int SEQ>
    __device__ __forceinline__ static unsigned int grid_stride() {
      return blockDim.x * gridDim.x * SEQ;
    }
  };
}

__global__ void hf_encode_phase1_fill(
  slice<const uint16_t> in,
  size_t const in_len,
  slice<const uint32_t> in_bk,
  int const in_bklen,
  slice<uint32_t> out) 
{
  auto s_bk = reinterpret_cast<uint32_t*>(__codec_raw);
  for (auto idx = helper::local_tid_1(); 
       idx < in_bklen; 
       idx += helper::block_stride_1()) {
    s_bk[idx] = in_bk(idx);
  }
  __syncthreads();

  for (auto idx = helper::global_tid_1(); 
       idx < in_len; 
       idx += helper::grid_stride_1()) {
    out(idx) = s_bk[(int)in(idx)];
  }
}

__global__ void hf_encode_phase2_deflate(
  slice<uint32_t> inout_inplace,
  size_t const len,
  slice<uint32_t> par_nbit,
  slice<uint32_t> par_ncell,
  int const sublen,
  int const pardeg)
{
  constexpr int CELL_BITWIDTH = sizeof(uint32_t) * 8;
  auto tid = BIX * BDX + TIX;

  if (tid * sublen < len) {
    int residue_bits = CELL_BITWIDTH;
    int total_bits = 0;
    uint32_t* ptr = inout_inplace.data_handle() + tid * sublen;
    uint32_t bufr;
    uint8_t word_width;

    auto did = tid * sublen;
    for (auto i = 0; i < sublen; i++, did++) {
      if (did == len) break;

      uint32_t packed_word = inout_inplace(tid * sublen + i);
      auto word_ptr = reinterpret_cast<
        struct HuffmanWord<sizeof(uint32_t)>*>(&packed_word);
      word_width = word_ptr->bitcount;
      word_ptr->bitcount = (uint8_t)0x0;

      if (residue_bits == CELL_BITWIDTH) {
        bufr = 0x0;
      }

      if (word_width <= residue_bits) {
        residue_bits -= word_width;
        bufr |= packed_word << residue_bits;

        if (residue_bits == 0) {
          residue_bits = CELL_BITWIDTH;
          *(ptr++) = bufr;
        }
      } else {
        auto l_bits = word_width - residue_bits;
        auto r_bits = CELL_BITWIDTH - l_bits;

        bufr |= packed_word >> l_bits;
        *(ptr++) = bufr;
        bufr = packed_word << r_bits;

        residue_bits = r_bits;
      }
      total_bits += word_width;
    }
    *ptr = bufr;

    par_nbit(tid) = total_bits;
    par_ncell(tid) = (total_bits + CELL_BITWIDTH - 1) / CELL_BITWIDTH;
  }
}

__global__ void hf_encode_phase4_concatenate(
  slice<const uint32_t> gapped,
  slice<const uint32_t> par_entry,
  slice<const uint32_t> par_ncell,
  int const cfg_sublen,
  slice<uint32_t> non_gapped) 
{
  auto n = par_ncell(blockIdx.x);
  auto src = gapped.data_handle() + cfg_sublen * blockIdx.x;
  auto dst = non_gapped.data_handle() + par_entry(blockIdx.x);

  for (auto i = threadIdx.x; i < n; i += blockDim.x) {
    dst[i] = src[i];
  }
}

__global__ void hf_kernel_decode(
  uint32_t* in,
  uint8_t* revbook,
  uint32_t* par_nbit,
  uint32_t* par_entry,
  int revbook_nbyte,
  int sublen,
  int pardeg,
  slice<uint16_t> out) 
{
  constexpr int CELL_BITWIDTH = sizeof(uint32_t) * 8;
  extern __shared__ uint8_t s_revbook[];
  constexpr auto block_dim = 256;

  auto single_thread_inflate = [&](const uint32_t* input, 
                                   uint16_t* out, 
                                   const uint32_t total_bw) {
    int next_bit;
    auto idx_bit = 0, idx_byte = 0, idx_out = 0;
    uint32_t bufr = input[idx_byte];
    auto first = (uint32_t*)(s_revbook);
    auto entry = first + CELL_BITWIDTH;
    auto keys = (uint16_t*)
      (s_revbook + sizeof(uint32_t) * (2 * CELL_BITWIDTH));
    uint32_t v = (bufr >> (CELL_BITWIDTH - 1)) & 0x1; // first bit
    auto l = 1, i = 0;

    while (i < total_bw) {
      while (v < first[l]) {
        ++i;
        idx_byte = i / CELL_BITWIDTH;
        idx_bit = i % CELL_BITWIDTH;
        if (idx_bit == 0) {
          bufr = input[idx_byte];
        }

        next_bit = ((bufr >> (CELL_BITWIDTH - 1 - idx_bit)) & 0x1);
        v = (v << 1) | next_bit;
        ++l;
      }
      out[idx_out++] = keys[entry[l] + v - first[l]];
      {
        ++i;
        idx_byte = i / CELL_BITWIDTH;
        idx_bit = i % CELL_BITWIDTH;
        if (idx_bit == 0) {
          bufr = input[idx_byte];
        }
        next_bit = ((bufr >> (CELL_BITWIDTH - 1 - idx_bit)) & 0x1);
        v = 0x0 | next_bit;
      }
      l = 1;
    }
  };

  auto R = (revbook_nbyte - 1 + block_dim) / block_dim;
  for (auto i = 0; i < R; i++) {
    if (TIX + i * block_dim < revbook_nbyte) {
      s_revbook[TIX + i * block_dim] = revbook[TIX + i * block_dim];
    }
  }
  __syncthreads();

  auto gid = BIX * BDX + TIX;
  if (gid < pardeg) {
    single_thread_inflate(in + par_entry[gid],
                          out.data_handle() + sublen * gid,
                          par_nbit[gid]);
    __syncthreads();
  }
}