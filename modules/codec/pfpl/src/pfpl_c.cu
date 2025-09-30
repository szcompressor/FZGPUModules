/*
This file is part of the LC framework for synthesizing high-speed parallel lossless and error-bounded lossy data compression and decompression algorithms for CPUs and GPUs.

BSD 3-Clause License

Copyright (c) 2021-2025, Noushin Azami, Alex Fallin, Brandon Burtchell, Andrew Rodriguez, Benila Jerald, Yiqian Liu, and Martin Burtscher
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

URL: The latest version of this code is available at https://github.com/burtscher/LC-framework.

Sponsor: This code is based upon work supported by the U.S. Department of Energy, Office of Science, Office of Advanced Scientific Research (ASCR), under contract DE-SC0022223.
*/

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cassert>
#include <cuda_runtime.h>


#include "prefix_sum.h"
#include "zero_elimination.h"
#include "repetition_elimination.h"
#include "mem/cxx_smart_ptr.h"

namespace pfpl {

namespace detail {

static __device__ int g_chunk_counter;

static __global__ void d_reset()
{
  g_chunk_counter = 0;
}

template <typename T>
static __device__ inline bool diff_coding_nb(int& csize, uint8_t in [1024*16], uint8_t out [1024*16], uint8_t temp [1024*16]) 
{
  const T s = (T)0xAAAAAAAAAAAAAAAAULL;
  T* const in_t = (T*)in;
  T* const out_t = (T*)out;
  const int size = csize / sizeof(T);
  const int tid = threadIdx.x;

  for (int i = tid; i < size; i += 512) {
    const T prev = (i == 0) ? 0 : in_t[i - 1];
    const T val = in_t[i];
    const T data = val - prev;
    out_t[i] = (data + s) ^ s;
  }

  // copy leftover bytes
  if constexpr(sizeof(T) > 1) {
    const int extra = csize % sizeof(T);
    if (tid < extra) out[csize - extra + tid] = in[csize - extra + tid];
  }
  return true;
}  

#define swp(x, y, s, m) t = ((x) ^ ((y) >> (s))) & (m);  (x) ^= t;  (y) ^= t << (s);

static __device__ inline bool bitshuffle_2byte(int& csize, uint8_t in [1024*16], uint8_t out [1024*16], uint8_t temp [1024*16])
{
  const int extra = csize % (16 * 16 / 8);
  const int size = (csize - extra) / 2;
  const int tid = threadIdx.x;
  unsigned long long* const in_l = (unsigned long long*)in;
  unsigned short* const out_s = (unsigned short*)out;

  for (int pos = 16 * tid; pos < size; pos += 16 * 512) {
    unsigned long long t, *a = &in_l[pos / 4];  // process 4 shorts in 1 long long

    for (int i = 0; i < 2; i++) {
      swp(a[i], a[i + 2], 8, 0x00FF00FF00FF00FFULL);
    }

    for (int j = 0; j < 4; j += 2) {
      swp(a[j], a[j + 1], 4, 0x0F0F0F0F0F0F0F0FULL);
    }

    for (int j = 0; j < 4; j++) {
      const unsigned long long m = 0x33333333CCCCCCCCULL;
      const unsigned long long vnm = a[j] & ~m;
      a[j] = (a[j] & m) | (vnm >> 34) | (vnm << 34);
    }

    for (int j = 0; j < 4; j++) {
      const unsigned long long m = 0x5555AAAA5555AAAAULL;
      const unsigned long long m1 = 0xFFFF0000FFFF0000ULL;
      const unsigned long long vnm = a[j] & ~m;
      const unsigned long long res = (a[j] & m) | ((vnm & m1) >> 17) | ((vnm & ~m1) << 17);
      out_s[pos / 16 + (j * 4) * (size / 16)] = res;
      out_s[pos / 16 + (j * 4 + 1) * (size / 16)] = res >> 16;
      out_s[pos / 16 + (j * 4 + 2) * (size / 16)] = res >> 32;
      out_s[pos / 16 + (j * 4 + 3) * (size / 16)] = res >> 48;
    }
  }

  // copy leftover bytes
  if (tid < extra) out[csize - extra + tid] = in[csize - extra + tid];
  return true;
}

template <typename T>
static __device__ inline bool zero_removal(int& csize, uint8_t in [1024*16], uint8_t out [1024*16], uint8_t temp [1024*16])
{
  const int tid = threadIdx.x;
  const int size = csize / sizeof(T);
  const int extra = csize % sizeof(T);
  const int avail = (1024 * 16) - 2 - extra;
  const int bits = 8 * sizeof(T);

  // zero out end of bitmap
  int* const temp_w = (int*)temp;
  uint8_t* const bitmap = (uint8_t*)&temp_w[32 + 1];
  if (csize < (1024 * 16)) {
    for (int i = csize / bits + tid; i < (1024*16) / bits; i += 512) {
      bitmap[i] = 0;
    }
    __syncthreads();
  }

  // copy nonzero values and create bitmap
  int wpos = 0;
  if (size > 0) d_ZEencode((T*)in, size, (T*)out, wpos, (T*)bitmap, temp_w);
  wpos *= sizeof(T);
  if (wpos >= avail) return false;
  __syncthreads();

  // check if not all zeros
  if (wpos != 0) {
    // iteratively compress bitmap
    int base = 0 / sizeof(T);
    int range = 2048 / sizeof(T);
    int cnt = avail - wpos;
    if (!d_REencode<uint8_t, 2048 / sizeof(T), true>(&bitmap[base], range, &out[wpos], cnt, &bitmap[base + range], temp_w)) return false;
    wpos += cnt;
    __syncthreads();

    base = 2048 / sizeof(T);
    range = 256 / sizeof(T);
    cnt = avail - wpos;
    if (!d_REencode<uint8_t, 256 / sizeof(T), true>(&bitmap[base], range, &out[wpos], cnt, &bitmap[base + range], temp_w)) return false;
    wpos += cnt;
    __syncthreads();

    base = (2048 + 256) / sizeof(T);
    range = 32 / sizeof(T);
    if constexpr (sizeof(T) < 8) {
      cnt = avail - wpos;
      if (!d_REencode<uint8_t, 32 / sizeof(T), true>(&bitmap[base], range, &out[wpos], cnt, &bitmap[base + range], temp_w)) return false;
      wpos += cnt;

      base = (2048 + 256 + 32) / sizeof(T);
      range = 4 / sizeof(T);
    }

    // output last level of bitmap
    if (wpos >= avail - range) return false;
    if (tid < range) {  // 4 / sizeof(T)
      out[wpos + tid] = bitmap[base + tid];
    }
    wpos += range;
  }

  // copy leftover bytes
  if constexpr (sizeof(T) > 1) {
    if (tid < extra) out[wpos + tid] = in[csize - extra + tid];
  }

  // output old csize and update csize
  const int new_size = wpos + 2 + extra;
  if (tid == 0) {
    out[new_size - 2] = csize;  // bottom byte
    out[new_size - 1] = csize >> 8;  // second byte
  }
  csize = new_size;
  return true;
}

static inline __device__ void propagate_carry(const int value, const int chunkID, 
  volatile int* const __restrict__ fullcarry, int* const __restrict__ s_fullc)
{
  if (threadIdx.x == 512 - 1) {  // last thread
    fullcarry[chunkID] = (chunkID == 0) ? value : -value;
  }

  if (chunkID != 0) {
    if (threadIdx.x + 32 >= 512) {  // last warp
      const int lane = threadIdx.x % 32;
      const int cidm1ml = chunkID - 1 - lane;
      int val = -1;
      __syncwarp();  // not optional
      do {
        if (cidm1ml >= 0) {
          val = fullcarry[cidm1ml];
        }
      } while ((__any_sync(~0, val == 0)) || (__all_sync(~0, val <= 0)));
      const int mask = __ballot_sync(~0, val > 0);
      const int pos = __ffs(mask) - 1;
      int partc = (lane < pos) ? -val : 0;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
      partc = __reduce_add_sync(~0, partc);
#else
      partc += __shfl_xor_sync(~0, partc, 1);
      partc += __shfl_xor_sync(~0, partc, 2);
      partc += __shfl_xor_sync(~0, partc, 4);
      partc += __shfl_xor_sync(~0, partc, 8);
      partc += __shfl_xor_sync(~0, partc, 16);
#endif
      if (lane == pos) {
        const int fullc = partc + val;
        fullcarry[chunkID] = fullc + value;
        *s_fullc = fullc;
      }
    }
  }
}

// copy (len) bytes from shared memory (source) to global memory (destination)
// source must we word aligned
static inline __device__ void s2g(void* const __restrict__ destination, 
  const void* const __restrict__ source, const int len)
{
  const int tid = threadIdx.x;
  const uint8_t* const __restrict__ input = (uint8_t*)source;
  uint8_t* const __restrict__ output = (uint8_t*)destination;
  if (len < 128) {
    if (tid < len) output[tid] = input[tid];
  } else {
    const int nonaligned = (int)(size_t)output;
    const int wordaligned = (nonaligned + 3) & ~3;
    const int linealigned = (nonaligned + 127) & ~127;
    const int bcnt = wordaligned - nonaligned;
    const int wcnt = (linealigned - wordaligned) / 4;
    const int* const __restrict__ in_w = (int*)input;
    if (bcnt == 0) {
      int* const __restrict__ out_w = (int*)output;
      if (tid < wcnt) out_w[tid] = in_w[tid];
      for (int i = tid + wcnt; i < len / 4; i += 512) {
        out_w[i] = in_w[i];
      }
      if (tid < (len & 3)) {
        const int i = len - 1 - tid;
        output[i] = input[i];
      }
    } else {
      const int shift = bcnt * 8;
      const int rlen = len - bcnt;
      int* const __restrict__ out_w = (int*)&output[bcnt];
      if (tid < bcnt) output[tid] = input[tid];
      if (tid < wcnt) out_w[tid] = __funnelshift_r(in_w[tid], in_w[tid + 1], shift);
      for (int i = tid + wcnt; i < rlen / 4; i += 512) {
        out_w[i] = __funnelshift_r(in_w[i], in_w[i + 1], shift);
      }
      if (tid < (rlen & 3)) {
        const int i = len - 1 - tid;
        output[i] = input[i];
      }
    }
  }
}

}  // namespace detail


template <typename T>
__global__ void pfpl_encode_kernel(
    const T* __restrict__ d_input, 
    size_t num_codes,
    uint8_t* __restrict__ d_archive,
    size_t* const __restrict__ d_archive_len,
    int* const __restrict__ fullcarry)
{
  const int CHUNK_SIZE = 1024 * 16;
  
  // shared mem buffer
  __shared__ long long chunk [3 * (CHUNK_SIZE / sizeof(long long))];

  // split into 3 buffers
  uint8_t* in = (uint8_t*)&chunk[0];
  uint8_t* out = (uint8_t*)&chunk[CHUNK_SIZE / sizeof(long long)];
  uint8_t* const temp = (uint8_t*)&chunk[2 * (CHUNK_SIZE / sizeof(long long))];

  // init
  const int tid = threadIdx.x;
  const int last = 3 * (CHUNK_SIZE / sizeof(long long)) - 2 - 32;
  const int chunks = (num_codes + CHUNK_SIZE - 1) / CHUNK_SIZE;
  long long* const head_out = (long long*)d_archive;
  unsigned short* const size_out = (unsigned short*)&head_out[2];
  uint8_t* const data_out = (uint8_t*)&size_out[chunks];
  
  do {

    // assign work
    if (tid == 0) chunk[last] = atomicAdd(&detail::g_chunk_counter, 1);
    __syncthreads();

    // check if done
    const int chunkID = chunk[last];
    const int base = chunkID * CHUNK_SIZE;
    if (base >= num_codes) break;

    // load input chunk
    const int osize = min(CHUNK_SIZE, static_cast<int>(num_codes - base));
    long long* const input_l = (long long*)&d_input[base];
    long long* const out_l = (long long*)out;
    for (int i = tid; i < osize / 8; i += 512) {
      out_l[i] = input_l[i];
    }
    const int extra = osize % 8;
    if (tid < extra) out[osize - extra + tid] = ((uint8_t*)d_input)[base + osize - extra + tid];
    __syncthreads();

    // encode chunk
    int csize = osize;
    
    { // difference coding -> negabinary
      uint8_t* tmp = in; in = out; out = tmp;
      detail::diff_coding_nb<T>(csize, in, out, temp);
      __syncthreads();
    }

    { // bitshuffle
      uint8_t* tmp = in; in = out; out = tmp;
      detail::bitshuffle_2byte(csize, in, out, temp);
      __syncthreads();
    }

    { // zero removal
      uint8_t* tmp = in; in = out; out = tmp;
      detail::zero_removal<T>(csize, in, out, temp);
      __syncthreads();
    }

    // handle carry
    if (csize >= osize) csize = osize;
    detail::propagate_carry(csize, chunkID, fullcarry, (int*)temp);

    if (tid == 0) size_out[chunkID] = csize;

    if (csize == osize) {
      long long* const out_l = (long long*)out;
      for (int i = tid; i < osize / 8; i += 512) {
        out_l[i] = input_l[i];
      }
      const int extra = osize % 8;
      if (tid < extra) out[osize - extra + tid] = ((uint8_t*)d_input)[base + osize - extra + tid];
    }
    __syncthreads();

    // store chunk
    const int offs = (chunkID == 0) ? 0 : *((int*)temp);
    detail::s2g(&data_out[offs], out, csize);

    // finalize if last chunk
    if ((tid == 0) && (base + (CHUNK_SIZE) >= num_codes)) {
      head_out[0] = (long long)num_codes;
      *d_archive_len = &data_out[fullcarry[chunkID]] - d_archive;
    }

  } while (true);
 
}

// #############################

template <typename T>
void GPU_PFPL_encode(const T* d_input, size_t num_codes,
                     uint8_t* d_archive, size_t* d_archive_len,
                     int* d_fullcarry, int blocks, cudaStream_t stream)
{
  if (!d_input || !d_archive || !d_archive_len || num_codes == 0) {
    if (d_archive_len) {
      cudaMemsetAsync(d_archive_len, 0, sizeof(size_t), stream);
    }
    printf("PFPL Encoding: No data to encode.\n");
    return;
  }

  detail::d_reset<<<1, 1, 0, stream>>>();

  pfpl_encode_kernel<T><<<blocks, 512, 0, stream>>>(
      d_input, num_codes, d_archive, d_archive_len, d_fullcarry);

  cudaStreamSynchronize(stream);

  // DEBUG PRINTS
  printf("PFPL num blocks: %d\n", blocks);
  

  size_t compressed_size;
  cudaMemcpyAsync(&compressed_size, d_archive_len, sizeof(size_t), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  printf("PFPL Encoding completed: %zu input elements, %zu bytes compressed.\n", num_codes, compressed_size);
  if (compressed_size > 0) {
      auto h_test_data = MAKE_UNIQUE_HOST(uint8_t, compressed_size);
      cudaMemcpyAsync(h_test_data.get(), d_archive, compressed_size, cudaMemcpyDeviceToHost, stream);
      cudaStreamSynchronize(stream);
      
      // Print header information
      long long* const header = (long long*)h_test_data.get();
      printf("Header - Element count: %lld\n", header[0]);
      printf("Header - Second value: %016llx\n", header[1]);
      
      // Print chunk sizes
      int chunks = (header[0] + 16384 - 1) / 16384;  // CHUNK_SIZE = 1024*16 = 16384
      unsigned short* const chunk_sizes = (unsigned short*)&header[2];
      printf("First %d chunk sizes: ", std::min(chunks, 10));
      for (int i = 0; i < std::min(chunks, 10); i++) {
          printf("%u ", chunk_sizes[i]);
      }
      printf("\n");
  }

}

template void GPU_PFPL_encode<uint8_t>(const uint8_t*, size_t, uint8_t*, size_t*, int*, int, cudaStream_t);
template void GPU_PFPL_encode<uint16_t>(const uint16_t*, size_t, uint8_t*, size_t*, int*, int, cudaStream_t);
template void GPU_PFPL_encode<uint32_t>(const uint32_t*, size_t, uint8_t*, size_t*, int*, int, cudaStream_t);

}  // namespace pfpl