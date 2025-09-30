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

#include "mem/cxx_smart_ptr.h"
#include "sum_reductions.h"
#include "prefix_sum.h"
#include "zero_elimination.h"
#include "repetition_elimination.h"

namespace pfpl {

namespace detail {

static __device__ int g_chunk_counter;

static __global__ void d_reset()
{
  g_chunk_counter = 0;
}

template <typename T>
static __device__ inline void zero_elimination_x(int& csize, uint8_t in [1024*16], uint8_t out [1024*16], uint8_t temp [1024*16])
{
  const int tid = threadIdx.x;
  int rpos = csize;
  csize = (int)in[--rpos] << 8;  // second byte
  csize |= in[--rpos];  // bottom byte
  const int size = csize / sizeof(T);  // words in chunk (rounded down)
  // assert(CS == 16384);
  // assert(TPB >= 256);

  // copy leftover byte
  if constexpr (sizeof(T) > 1) {
    const int extra = csize % sizeof(T);
    if (tid < extra) out[csize - extra + tid] = in[rpos - extra + tid];
    rpos -= extra;
  }

  if (rpos == 0) {
    // all zeros
    T* const out_t = (T*)out;
    for (int i = tid; i < size; i += 512) {
      out_t[i] = 0;
    }
  } else {
    int* const temp_w = (int*)temp;
    uint8_t* const bitmap = (uint8_t*)&temp_w[32];

    // iteratively decompress bitmaps
    int base, range;
    if constexpr (sizeof(T) == 8) {
      base = (2048 + 256) / sizeof(T);
      range = 32 / sizeof(T);
      // read in last level of bitmap
      rpos -= range;
      if (tid < range) bitmap[base + tid] = in[rpos + tid];
    } else {
      base = (2048 + 256 + 32) / sizeof(T);
      range = 4 / sizeof(T);
      // read in last level of bitmap
      rpos -= range;
      if (tid < range) bitmap[base + tid] = in[rpos + tid];

      rpos -= __syncthreads_count((tid < range * 8) && ((in[rpos + tid / 8] >> (tid % 8)) & 1));
      base = (2048 + 256) / sizeof(T);
      range = 32 / sizeof(T);
      d_REdecode<uint8_t, 32 / sizeof(T)>(range, &in[rpos], &bitmap[base + range], &bitmap[base], temp_w);
    }
    __syncthreads();

    rpos -= __syncthreads_count((tid < range * 8) && ((bitmap[base + tid / 8] >> (tid % 8)) & 1));
    base = 2048 / sizeof(T);
    range = 256 / sizeof(T);
    d_REdecode<uint8_t, 256 / sizeof(T)>(range, &in[rpos], &bitmap[base + range], &bitmap[base], temp_w);
    __syncthreads();

    if constexpr (sizeof(T) >= 4) {
      rpos -= __syncthreads_count((tid < range * 8) && ((bitmap[base + tid / 8] >> (tid % 8)) & 1));
    }
    if constexpr (sizeof(T) == 2) {
      int sum = __syncthreads_count((tid < range * 8) && ((bitmap[base + tid / 8] >> (tid % 8)) & 1));
      sum += __syncthreads_count((tid + 512 < range * 8) && ((bitmap[base + (tid + 512) / 8] >> (tid % 8)) & 1));
      rpos -= sum;
    }
    if constexpr (sizeof(T) == 1) {
      int sum = 0;
      for (int i = 0; i < 512 * 4; i += 512) {
        sum += __syncthreads_count((tid + i < range * 8) && ((bitmap[base + (tid + i) / 8] >> (tid % 8)) & 1));
      }
      rpos -= sum;
    }
    base = 0 / sizeof(T);
    range = 2048 / sizeof(T);
    d_REdecode<uint8_t, 2048 / sizeof(T)>(range, &in[rpos], &bitmap[base + range], &bitmap[base], temp_w);
    __syncthreads();

    // copy non-zero values based on bitmap
    if (size > 0) d_ZEdecode(size, (T*)in, (T*)bitmap, (T*)out, temp_w);
  }
}

static __device__ inline void bitshuffle_x(int& csize, uint8_t in [1024*16], uint8_t out [1024*16], uint8_t temp [1024*16])
{
  const int extra = csize % (16 * 16 / 8);
  const int size = (csize - extra) / 2;
  const int tid = threadIdx.x;
  unsigned short* const in_s = (unsigned short*)in;
  unsigned long long* const out_l = (unsigned long long*)out;

  for (int pos = 16 * tid; pos < size; pos += 16 * 512) {
    unsigned long long t, *a = &out_l[pos / 4];  // process 4 shorts in 1 long long

    for (int i = 0; i < 4; i++) {
      a[i] = in_s[pos / 16 + (i * 4) * (size / 16)] | ((unsigned long long)in_s[pos / 16 + (i * 4 + 1) * (size / 16)] << 16) | ((unsigned long long)in_s[pos / 16 + (i * 4 + 2) * (size / 16)] << 32) | ((unsigned long long)in_s[pos / 16 + (i * 4 + 3) * (size / 16)] << 48);
    }

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
      a[j] = (a[j] & m) | ((vnm & m1) >> 17) | ((vnm & ~m1) << 17);
    }
  }

  // copy leftover bytes
  if (tid < extra) out[csize - extra + tid] = in[csize - extra + tid];
}

template <typename T>
static __device__ inline void difference_coding_x(int& csize, uint8_t in [CS], uint8_t out [CS], uint8_t temp [CS])
{
  const T s = (T)0xAAAAAAAAAAAAAAAAULL;
  T* const in_t = (T*)in;
  T* const out_t = (T*)out;
  const int size = csize / sizeof(T);  // words in chunk (rounded down)
  const int tid = threadIdx.x;
  const int beg = tid * size / 512;
  const int end = (tid + 1) * size / 512;

  // compute local sums
  T sum = 0;
  for (int i = beg; i < end; i++) {
    const T data = in_t[i];
    const T val = (data ^ s) - s;
    sum += val;
    in_t[i] = val;
  }

  // compute prefix sum
  sum = block_prefix_sum(sum, temp);  // includes barrier

  // compute intermediate values
  for (int i = end - 1; i >= beg; i--) {
    out_t[i] = sum;
    sum -= in_t[i];
  }

  // copy leftover bytes at end
  if constexpr (sizeof(T) > 1) {
    const int extra = csize % sizeof(T);
    if (tid < extra) out[csize - extra + tid] = in[csize - extra + tid];
  }
}


// copy (len) bytes from global memory (source) to shared memory (destination) using separate shared memory buffer (temp)
// destination and temp must we word aligned, accesses up to CS + 3 bytes in temp
static inline __device__ void g2s(void* const __restrict__ destination, 
  const void* const __restrict__ source, const int len, void* const __restrict__ temp)
{
  const int tid = threadIdx.x;
  const uint8_t* const __restrict__ input = (uint8_t*)source;
  if (len < 128) {
    uint8_t* const __restrict__ output = (uint8_t*)destination;
    if (tid < len) output[tid] = input[tid];
  } else {
    const int nonaligned = (int)(size_t)input;
    const int wordaligned = (nonaligned + 3) & ~3;
    const int linealigned = (nonaligned + 127) & ~127;
    const int bcnt = wordaligned - nonaligned;
    const int wcnt = (linealigned - wordaligned) / 4;
    int* const __restrict__ out_w = (int*)destination;
    if (bcnt == 0) {
      const int* const __restrict__ in_w = (int*)input;
      uint8_t* const __restrict__ out = (uint8_t*)destination;
      if (tid < wcnt) out_w[tid] = in_w[tid];
      for (int i = tid + wcnt; i < len / 4; i += 512) {
        out_w[i] = in_w[i];
      }
      if (tid < (len & 3)) {
        const int i = len - 1 - tid;
        out[i] = input[i];
      }
    } else {
      const int offs = 4 - bcnt;  //(4 - bcnt) & 3;
      const int shift = offs * 8;
      const int rlen = len - bcnt;
      const int* const __restrict__ in_w = (int*)&input[bcnt];
      uint8_t* const __restrict__ buffer = (uint8_t*)temp;
      uint8_t* const __restrict__ buf = (uint8_t*)&buffer[offs];
      int* __restrict__ buf_w = (int*)&buffer[4];  //(int*)&buffer[(bcnt + 3) & 4];
      if (tid < bcnt) buf[tid] = input[tid];
      if (tid < wcnt) buf_w[tid] = in_w[tid];
      for (int i = tid + wcnt; i < rlen / 4; i += 512) {
        buf_w[i] = in_w[i];
      }
      if (tid < (rlen & 3)) {
        const int i = len - 1 - tid;
        buf[i] = input[i];
      }
      __syncthreads();
      buf_w = (int*)buffer;
      for (int i = tid; i < (len + 3) / 4; i += 512) {
        out_w[i] = __funnelshift_r(buf_w[i], buf_w[i + 1], shift);
      }
    }
  }
}

} // namespace detail

// ##########################

template <typename T>
__global__ void pfpl_decode_kernel(const uint8_t* const __restrict__ d_input,
                                   T* const __restrict__ d_output,
                                   int* const __restrict__ d_archive_len)
{
  constexpr int chunk_size = 1024 * 16;
  __shared__ long long chunk [3 * (chunk_size / sizeof(long long))];
  const int last = 3 * (chunk_size / sizeof(long long)) - 2 - 32;

  // input header
  long long* const head_in = (long long*)d_input;
  const int outsize = (int)head_in[0];
  
  // init
  const int chunks = (outsize + chunk_size - 1) / chunk_size;
  unsigned short* const size_in = (unsigned short*)&head_in[2];
  uint8_t* const data_in = (uint8_t*)&size_in[chunks];

  // loop over chunks
  const int tid = threadIdx.x;
  int prevChunkID = 0;
  int prevOffset = 0;
  do {
    
    // assign work dynamically
    if (tid == 0) chunk[last] = atomicAdd(&detail::g_chunk_counter, 1);
    __syncthreads();

    // terminate if done
    const int chunkID = (int)chunk[last];
    const int base = chunkID * chunk_size;
    if (base >= outsize) break;

    // compute sum of all prior csizes (start where left off in previous iteration)
    int sum = 0;
    for (int i = prevChunkID + tid; i < chunkID; i += 512) {
      sum += (int)size_in[i];
    }
    int csize = (int)size_in[chunkID];
    const int offs = prevOffset + block_sum_reduction(sum, (int*)&chunk[last + 1]);
    prevChunkID = chunkID;
    prevOffset = offs;

    // create 3 shared memory buffers
    uint8_t* in = (uint8_t*)&chunk[0];
    uint8_t* out = (uint8_t*)&chunk[chunk_size / sizeof(long long)];
    uint8_t* temp = (uint8_t*)&chunk[2 * (chunk_size / sizeof(long long))];

    // load chunk
    detail::g2s(in, &data_in[offs], csize, out);
    uint8_t* tmp = in; in = out; out = tmp;
    __syncthreads();

    // decode
    const int osize = min(chunk_size, outsize - base);
    // if (csize < osize) {
      tmp = in; in = out; out = tmp;
      detail::zero_elimination_x<T>(csize, in, out, temp);
      __syncthreads();
      tmp = in; in = out; out = tmp;
      detail::bitshuffle_x(csize, in, out, temp);
      __syncthreads();
      tmp = in; in = out; out = tmp;
      detail::difference_coding_x<T>(csize, in, out, temp);
      __syncthreads();
    // }

    long long* const output_l = (long long*)&d_output[base];
    long long* const out_l = (long long*)out;
    for (int i = tid; i < osize / 8; i += 512) {
      output_l[i] = out_l[i];
    }
    const int extra = osize % 8;
    if (tid < extra) d_output[base + osize - extra + tid] = out[osize - extra + tid];
  } while (true);

  if (blockIdx.x == 0 && tid == 0) {
    *d_archive_len = outsize;
  }
}

// #########################

template <typename T>
void GPU_PFPL_decode(const uint8_t* d_input, size_t data_len,
                     T* d_output, int* d_archive_len, int blocks, cudaStream_t stream)
{

  // DEBUG PRINT
  printf("GPU_PFPL_decode<%zu> launched with %d blocks on stream %p\n", sizeof(T), blocks, (void*)stream);
  printf("Input data length: %zu\n", data_len);
  printf("Output data pointer: %p\n", (void*)d_output);
  printf("Archive length pointer: %p\n", (void*)d_archive_len);

  auto h_test_data = MAKE_UNIQUE_HOST(uint8_t, data_len);
  cudaMemcpyAsync(h_test_data.get(), d_input, data_len, cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
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

  size_t shared_mem_size = 3 * (1024 * 16);

  detail::d_reset<<<1, 1, 0, stream>>>();
  pfpl_decode_kernel<T><<<blocks, 512, shared_mem_size, stream>>>(d_input, d_output, d_archive_len);
}

template void GPU_PFPL_decode<uint8_t>(const uint8_t* d_input, size_t data_len,
                                      uint8_t* d_output, int* d_archive_len, int blocks, cudaStream_t stream);
template void GPU_PFPL_decode<uint16_t>(const uint8_t* d_input, size_t data_len,
                                       uint16_t* d_output, int* d_archive_len, int blocks, cudaStream_t stream);
template void GPU_PFPL_decode<uint32_t>(const uint8_t* d_input, size_t data_len,
                                       uint32_t* d_output, int* d_archive_len, int blocks, cudaStream_t stream);

} // namespace pfpl