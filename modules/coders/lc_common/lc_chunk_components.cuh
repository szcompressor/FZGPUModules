#pragma once

/**
 * @file modules/coders/lc_common/lc_chunk_components.cuh
 * @brief Vendored LC-framework single-chunk device codecs (shared by RRE + RZE).
 *
 * Faithful port of the GPU device functions for the LC framework's `RRE` and
 * `RZE` lossless components (Burtscher et al., BSD-3-Clause).  It bundles:
 *   - `block_prefix_sum`            (from `lc/prefix_sum.h`)
 *   - `d_REencode*` / `d_REdecode*`  (from `lc/components/include/d_repetition_elimination.h`)
 *   - `d_ZEencode*` / `d_ZEdecode*`  (from `lc/components/include/d_zero_elimination.h`)
 *   - `d_RRE<T>` / `d_iRRE<T>`        (from `lc/components/include/d_RRE.h`)
 *   - `d_RZE<T>` / `d_iRZE<T>`        (from `lc/components/include/d_RZE.h`)
 *
 * The functions operate on a single CS-byte chunk held in shared memory and are
 * invoked, one block per chunk, by `rre_stage.cu` and `rze_stage.cu`.  They are
 * wrapped in `fz::lc_detail` and declared `static __device__` (internal linkage).
 * The `_N` word-size variants (T = uint8/16/32/64) reproduce LC's RRE_1/2/4/8 and
 * RZE_1/2/4/8 — the `_N` suffix is the word size, not a recursion-level count.
 *
 * Upstream: https://github.com/burtscher/LC-framework — see THIRD_PARTY.md.
 *
 * The chunk geometry (CS = 16384, TPB = 512, WS = 32) matches the LC defaults
 * the component was tuned and validated against; do not change them without
 * re-checking the word-layout assertions inside d_REencode.
 */

#include <cuda_runtime.h>
#include <cstdint>
#include <cassert>
#include <type_traits>

// LC component geometry.  These tokens are required verbatim by the vendored
// device code below; they are #undef'd at the end of this header so they do
// not leak into the rest of the translation unit.
#define CS  (1024 * 16)   // chunk size in bytes (must be a multiple of 8)
#define TPB 512           // threads per block (power of two, >= 128)
#define WS  32            // warp size

namespace fz {
namespace lc_detail {

using byte = unsigned char;

// ─────────────────────────────────────────────────────────────────────────
// block_prefix_sum  (lc/prefix_sum.h) — block-wide inclusive prefix sum.
// ─────────────────────────────────────────────────────────────────────────
template <typename T>
static __device__ inline T block_prefix_sum(T val, void* buffer)  // returns inclusive prefix sum
{
  const int lane = threadIdx.x % WS;
  const int warp = threadIdx.x / WS;
  const int warps = TPB / WS;
  T* const carry = (T*)buffer;
  assert(WS >= warps);

  T tmp = __shfl_up_sync(~0, val, 1);
  if (lane >= 1) val += tmp;
  tmp = __shfl_up_sync(~0, val, 2);
  if (lane >= 2) val += tmp;
  tmp = __shfl_up_sync(~0, val, 4);
  if (lane >= 4) val += tmp;
  tmp = __shfl_up_sync(~0, val, 8);
  if (lane >= 8) val += tmp;
  tmp = __shfl_up_sync(~0, val, 16);
  if (lane >= 16) val += tmp;

  if (lane == WS - 1) carry[warp] = val;
  __syncthreads();  // carry written

  if constexpr (warps > 1) {
    if (warp == 0) {
      T sum = carry[lane];
      T tmp2 = __shfl_up_sync(~0, sum, 1);
      if (lane >= 1) sum += tmp2;
      if constexpr (warps > 2) {
        tmp2 = __shfl_up_sync(~0, sum, 2);
        if (lane >= 2) sum += tmp2;
        if constexpr (warps > 4) {
          tmp2 = __shfl_up_sync(~0, sum, 4);
          if (lane >= 4) sum += tmp2;
          if constexpr (warps > 8) {
            tmp2 = __shfl_up_sync(~0, sum, 8);
            if (lane >= 8) sum += tmp2;
            if constexpr (warps > 16) {
              tmp2 = __shfl_up_sync(~0, sum, 16);
              if (lane >= 16) sum += tmp2;
            }
          }
        }
      }
      carry[lane] = sum;
    }
    __syncthreads();  // carry updated

    if (warp > 0) val += carry[warp - 1];
  }

  return val;
}

// ─────────────────────────────────────────────────────────────────────────
// d_repetition_elimination.h — RE encode/decode primitives.
// ─────────────────────────────────────────────────────────────────────────

//special case for byte and short data
template <typename T, bool check = false>
static __device__ inline bool d_REencodebyteshort(const T* const in, const int insize, T* const dataout, int& datasize, T* const bmout, int* const temp_w)  // all sizes in number of words
{
  using type = T;
  using ull = unsigned long long;
  const int bitsperword = 8 * sizeof(type);
  const int bitsperlong = 8 * sizeof(ull);
  const int wordsperlong = bitsperlong / bitsperword;
  const int bytesperthread = CS / TPB;
  const ull* const in_l = (ull*)in;
  const int csize = insize * sizeof(T);
  assert(bytesperthread % sizeof(ull) == 0);
  assert(bytesperthread / sizeof(type) <= sizeof(int) * 8);
  assert(bytesperthread / sizeof(ull) * wordsperlong >= 8);
  assert(std::is_unsigned<type>::value);

  // output bitmaps and count non-repeating values
  const int tid = threadIdx.x;
  int bmp = 0, cnt = 0;
  if (tid * bytesperthread < csize) {
    type prev = (tid == 0) ? 0 : in[tid * (bytesperthread / sizeof(type)) - 1];
    for (int i = 0; i < bytesperthread / sizeof(ull); i++) {
      const ull lval = in_l[tid * (bytesperthread / sizeof(ull)) + i];
      const ull pval = (bitsperword < bitsperlong) ? ((lval << bitsperword) | prev) : prev;
      int bm = 0;
      for (int j = 0; j < wordsperlong; j++) {
        const type val = lval >> (j * bitsperword);
        const type prv = pval >> (j * bitsperword);
        bm |= (val != prv) << j;
      }
      prev = lval >> (bitsperlong - bitsperword);
      bmp |= bm << (i * wordsperlong);
    }
    if (tid * bytesperthread - (csize - bytesperthread) > 0) {
      bmp &= ~(-1 << ((csize % bytesperthread + sizeof(type) - 1) / sizeof(type)));
    }
    if constexpr (sizeof(type) == 1) {
      bmout[tid * 4] = bmp;
      bmout[tid * 4 + 1] = bmp >> 8;
      bmout[tid * 4 + 2] = bmp >> 16;
      bmout[tid * 4 + 3] = bmp >> 24;
    }
    if constexpr (sizeof(type) == 2) bmout[tid] = bmp;
    if constexpr (sizeof(type) == 4) ((byte*)bmout)[tid] = bmp;
    cnt = __popc(bmp);
  }

  int pos = block_prefix_sum(cnt, temp_w);
  if (tid == TPB - 1) temp_w[WS] = pos;
  if constexpr (check) {
    if (__syncthreads_or(pos > datasize)) return false;
  } else {
    __syncthreads();
  }
  pos -= cnt;

  // output non-repeating values
  if (bmp != 0) {
    for (int i = 0; i < bytesperthread / sizeof(ull); i++) {
      const ull lval = in_l[tid * (bytesperthread / sizeof(ull)) + i];
      const int bm = bmp >> (i * wordsperlong);
      for (int j = 0; j < wordsperlong; j++) {
        if ((bm >> j) & 1) {
          dataout[pos++] = lval >> (j * bitsperword);
        }
      }
    }
  }

  datasize = temp_w[WS];
  return true;
}


//warp-based one word per thread
template <typename T, bool check = false>
static __device__ inline bool d_REencode1wordperthread(const T* const in, const int insize, T* const dataout, int& datasize, T* const bmout, int* const temp_w)  // all sizes in number of words
{
  byte* const bmout_b = (byte*)bmout;
  const int tid = threadIdx.x;
  const int warp = tid / WS;
  const int lane = tid % WS;

  // count non-repeating values and output bitmaps
  const bool active = (tid < insize);
  const T prev = !active ? 0 : ((tid == 0) ? 0 : in[tid - 1]);
  const T val = active ? in[tid] : 0;
  const bool havenonrepval = (active && (val != prev));
  const int bm = __ballot_sync(~0, havenonrepval);
  const int cnt = __popc(bm);
  const int subwarps = TPB / 32;
  const int sublane = lane;
  const int subwarp = warp;
  if (active && (lane % 8 == 0)) bmout_b[tid / 8] = bm >> lane;
  if constexpr (sizeof(T) > 1) {
    if (warp == 0) {
      const int base = (insize + 7) / 8;
      const int top = (insize + (sizeof(T) * 8 - 1)) / 8;
      if (base + tid < top) bmout_b[base + tid] = 0;
    }
  }
  if (sublane == 0) temp_w[subwarp] = cnt;
  __syncthreads();

  // compute prefix sum
  int sum = 0;
  if (warp == 0) {
    if (lane < subwarps) sum = temp_w[lane];
    for (int i = 1; i < subwarps; i *= 2) {
      const int tmp = __shfl_up_sync(~0, sum, i);
      if (lane >= i) sum += tmp;
    }
    temp_w[lane] = sum;
  }
  if constexpr (check) {
    if (__syncthreads_or(sum > datasize)) return false;
  } else {
    __syncthreads();
  }

  // output non-repeating values
  if (havenonrepval) {
    const int loc = temp_w[subwarp] - cnt + __popc(bm & ((1 << sublane) - 1));
    dataout[loc] = val;
  }

  datasize = temp_w[subwarps - 1];
  return true;
}


//warp-based two words per thread
template <typename T, bool check = false>
static __device__ inline bool d_REencode2wordsperthread(const T* const in, const int insize, T* const dataout, int& datasize, T* const bmout, int* const temp_w)  // all sizes in number of words
{
  byte* const bmout_b = (byte*)bmout;
  const int tid = threadIdx.x;
  const int warp = tid / WS;
  const int lane = tid % WS;

  // count non-repeating values and output bitmaps
  const int tid1 = tid * 2;
  const int tid2 = tid1 + 1;
  const bool active1 = (tid1 < insize);
  const bool active2 = (tid2 < insize);
  const T prev = (!active1) ? 0 : ((tid1 == 0) ? 0 : in[tid1 - 1]);
  const T val1 = active1 ? in[tid1] : 0;
  const T val2 = active2 ? in[tid2] : 0;
  const bool havenonrepval1 = (active1 && (val1 != prev));
  const bool havenonrepval2 = (active2 && (val2 != val1));
  const int bm1 = __ballot_sync(~0, havenonrepval1);
  const int bm2 = __ballot_sync(~0, havenonrepval2);
  const int cnt = __popc(bm1) + __popc(bm2);
  const int comb = havenonrepval1 + havenonrepval2 * 2;
  const int sublane = lane;
  const int tmp1 = __shfl_sync(~0, comb, lane / 2) >> (lane % 2);
  const int bmlo = __ballot_sync(~0, tmp1 & 1);
  const int tmp2 = __shfl_sync(~0, comb, 16 + lane / 2) >> (lane % 2);
  const int bmhi = __ballot_sync(~0, tmp2 & 1);
  const int subwarps = TPB / 32;
  const int subwarp = warp;
  if (__any_sync(~0, active1) && (lane % 8 == 0)) bmout_b[warp * 8 + lane / 8] = bmlo >> lane;
  if (__any_sync(~0, active2) && (lane % 8 == 0)) bmout_b[warp * 8 + lane / 8 + 4] = bmhi >> lane;
  if constexpr (sizeof(T) > 1) {
    if (warp == 0) {
      const int base = (insize + 7) / 8;
      const int top = (insize + (sizeof(T) * 8 - 1)) / 8;
      if (base + tid < top) bmout_b[base + tid] = 0;
    }
  }
  if (sublane == 0) temp_w[subwarp] = cnt;
  __syncthreads();

  // compute prefix sum
  int sum = 0;
  if (warp == 0) {
    if (lane < subwarps) sum = temp_w[lane];
    for (int i = 1; i < subwarps; i *= 2) {
      const int tmp = __shfl_up_sync(~0, sum, i);
      if (lane >= i) sum += tmp;
    }
    temp_w[lane] = sum;
  }
  if constexpr (check) {
    if (__syncthreads_or(sum > datasize)) return false;
  } else {
    __syncthreads();
  }

  // output non-repeating values
  int loc = temp_w[subwarp] - cnt + __popc(bm1 & ((1 << sublane) - 1)) + __popc(bm2 & ((1 << sublane) - 1));
  if (havenonrepval1) dataout[loc++] = val1;
  if (havenonrepval2) dataout[loc] = val2;

  datasize = temp_w[subwarps - 1];
  return true;
}


//warp-based four words per thread
template <typename T, bool check = false>
static __device__ inline bool d_REencode4wordsperthread(const T* const in, const int insize, T* const dataout, int& datasize, T* const bmout, int* const temp_w)  // all sizes in number of words
{
  byte* const bmout_b = (byte*)bmout;
  const int tid = threadIdx.x;
  const int warp = tid / WS;
  const int lane = tid % WS;

  // count non-repeating values and output bitmaps
  const int tid1 = tid * 4;
  const int tid2 = tid1 + 1;
  const int tid3 = tid2 + 1;
  const int tid4 = tid3 + 1;
  const bool active1 = (tid1 < insize);
  const bool active2 = (tid2 < insize);
  const bool active3 = (tid3 < insize);
  const bool active4 = (tid4 < insize);
  const T prev = !active1 ? 0 : ((tid1 == 0) ? 0 : in[tid1 - 1]);
  const T val1 = active1 ? in[tid1] : 0;
  const T val2 = active2 ? in[tid2] : 0;
  const T val3 = active3 ? in[tid3] : 0;
  const T val4 = active4 ? in[tid4] : 0;
  const bool havenonrepval1 = (active1 && (val1 != prev));
  const bool havenonrepval2 = (active2 && (val2 != val1));
  const bool havenonrepval3 = (active3 && (val3 != val2));
  const bool havenonrepval4 = (active4 && (val4 != val3));
  const int bm1 = __ballot_sync(~0, havenonrepval1);
  const int bm2 = __ballot_sync(~0, havenonrepval2);
  const int bm3 = __ballot_sync(~0, havenonrepval3);
  const int bm4 = __ballot_sync(~0, havenonrepval4);
  const int cnt = __popc(bm1) + __popc(bm2) + __popc(bm3) + __popc(bm4);
  const int comb = havenonrepval1 + havenonrepval2 * 2 + havenonrepval3 * 4 + havenonrepval4 * 8;
  const int sublane = lane;
  const int tmp1 = __shfl_sync(~0, comb, lane / 4) >> (lane % 4);
  const int bmA = __ballot_sync(~0, tmp1 & 1);
  const int tmp2 = __shfl_sync(~0, comb, 8 + lane / 4) >> (lane % 4);
  const int bmB = __ballot_sync(~0, tmp2 & 1);
  const int tmp3 = __shfl_sync(~0, comb, 16 + lane / 4) >> (lane % 4);
  const int bmC = __ballot_sync(~0, tmp3 & 1);
  const int tmp4 = __shfl_sync(~0, comb, 24 + lane / 4) >> (lane % 4);
  const int bmD = __ballot_sync(~0, tmp4 & 1);
  const int subwarps = TPB / 32;
  const int subwarp = warp;
  if (__any_sync(~0, active1) && (lane % 8 == 0)) bmout_b[warp * 16 + lane / 8] = bmA >> lane;
  if (__any_sync(~0, active2) && (lane % 8 == 0)) bmout_b[warp * 16 + lane / 8 + 4] = bmB >> lane;
  if (__any_sync(~0, active3) && (lane % 8 == 0)) bmout_b[warp * 16 + lane / 8 + 8] = bmC >> lane;
  if (__any_sync(~0, active4) && (lane % 8 == 0)) bmout_b[warp * 16 + lane / 8 + 12] = bmD >> lane;
  if constexpr (sizeof(T) > 1) {
    if (warp == 0) {
      const int base = (insize + 7) / 8;
      const int top = (insize + (sizeof(T) * 8 - 1)) / 8;
      if (base + tid < top) bmout_b[base + tid] = 0;
    }
  }
  if (sublane == 0) temp_w[subwarp] = cnt;
  __syncthreads();

  // compute prefix sum
  int sum = 0;
  if (warp == 0) {
    if (lane < subwarps) sum = temp_w[lane];
    for (int i = 1; i < subwarps; i *= 2) {
      const int tmp = __shfl_up_sync(~0, sum, i);
      if (lane >= i) sum += tmp;
    }
    temp_w[lane] = sum;
  }
  if constexpr (check) {
    if (__syncthreads_or(sum > datasize)) return false;
  } else {
    __syncthreads();
  }

  // output non-repeating values
  int loc = temp_w[subwarp] - cnt + __popc(bm1 & ((1 << sublane) - 1)) + __popc(bm2 & ((1 << sublane) - 1)) + __popc(bm3 & ((1 << sublane) - 1)) + __popc(bm4 & ((1 << sublane) - 1));
  if (havenonrepval1) dataout[loc++] = val1;
  if (havenonrepval2) dataout[loc++] = val2;
  if (havenonrepval3) dataout[loc++] = val3;
  if (havenonrepval4) dataout[loc] = val4;

  datasize = temp_w[subwarps - 1];
  return true;
}


//thread-based X words per thread, X must be 8, 16, or 32
template <int X, typename T, bool check = false>
static __device__ inline bool d_REencodeXwordsperthread(const T* const in, const int insize, T* const dataout, int& datasize, T* const bmout, int* const temp_w)  // all sizes in number of words
{
  assert((X == 8) || (X == 16) || (X == 32));

  // count non-repeating values and output bitmaps
  const int WPT = X;  // words per thread
  const int tid = threadIdx.x;
  int bmp = 0, cnt = 0;
  if (tid * WPT < insize) {
    T prev = (tid == 0) ? 0 : in[tid * WPT - 1];
    for (int i = 0; i < WPT; i++) {
      const T val = in[tid * WPT + i];
      bmp |= (val != prev) << i;
      prev = val;
    }
    if (tid * WPT - (insize - WPT) > 0) {
      bmp &= ~(-1 << (insize % WPT));
    }
    if constexpr (X == 8) ((byte*)bmout)[tid] = bmp;
    if constexpr (X == 16) ((short*)bmout)[tid] = bmp;
    if constexpr (X == 32) ((int*)bmout)[tid] = bmp;
    cnt = __popc(bmp);
  }

  // pad with zeros if necessary to alignment point
  if constexpr (sizeof(T) * 8 > X) {
    if (tid < WS) {
      const int base = (insize + (X - 1)) / 8;
      const int top = (insize + (sizeof(T) * 8 - 1)) / 8;
      if (base + tid < top) ((byte*)bmout)[base + tid] = 0;
    }
  }

  // compute prefix sum
  int pos = block_prefix_sum(cnt, temp_w);
  if (tid == TPB - 1) temp_w[WS] = pos;
  if constexpr (check) {
    if (__syncthreads_or(pos > datasize)) return false;
  } else {
    __syncthreads();
  }
  pos -= cnt;

  // output non-repeating values
  if (bmp != 0) {
    for (int i = 0; i < WPT; i++) {
      if ((bmp >> i) & 1) {
        const T val = in[tid * WPT + i];
        dataout[pos++] = val;
      }
    }
  }

  datasize = temp_w[WS];
  return true;
}


template <typename T, int maxsize = CS, bool check = false>  // maxsize in bytes
static __device__ inline bool d_REencode(const T* const in, const int insize, T* const dataout, int& datasize, T* const bmout, int* const temp_w)  // all sizes in number of words
{
  assert((TPB & (TPB - 1)) == 0);
  assert((maxsize & (maxsize - 1)) == 0);
  assert(maxsize % sizeof(T) == 0);
  assert((maxsize / sizeof(T) % TPB == 0) || (maxsize / sizeof(T) < TPB));
  const int wordsperthread = maxsize / sizeof(T) / TPB;
  if constexpr ((sizeof(T) <= 2) && (maxsize > 2048)) {
    // special case for byte and short data
    return d_REencodebyteshort<T, check>(in, insize, dataout, datasize, bmout, temp_w);
  } else if constexpr (wordsperthread <= 1) {
    return d_REencode1wordperthread<T, check>(in, insize, dataout, datasize, bmout, temp_w);
  } else if constexpr (wordsperthread == 2) {
    return d_REencode2wordsperthread<T, check>(in, insize, dataout, datasize, bmout, temp_w);
  } else if constexpr (wordsperthread == 4) {
    return d_REencode4wordsperthread<T, check>(in, insize, dataout, datasize, bmout, temp_w);
  } else if constexpr (wordsperthread == 8) {
    return d_REencodeXwordsperthread<8, T, check>(in, insize, dataout, datasize, bmout, temp_w);
  } else if constexpr (wordsperthread == 16) {
    return d_REencodeXwordsperthread<16, T, check>(in, insize, dataout, datasize, bmout, temp_w);
  } else if constexpr (wordsperthread == 32) {
    return d_REencodeXwordsperthread<32, T, check>(in, insize, dataout, datasize, bmout, temp_w);
  } else {
    __trap();
    return false;
  }
}


template <typename T, typename U>  // U must be int or smaller; if smaller, it must be unsigned
static __device__ inline void d_REdecode_specialized(const int decsize, const T* const datain, const U* const bmin_t, T* const out, int* const temp_w)  // all sizes in number of words
{
  const int subWS = 32;
  const int tid = threadIdx.x;
  const int subwarp = tid / subWS;
  const int subwarps = TPB / subWS;
  const int sublane = tid % subWS;
  int num = (decsize + subWS - 1) / subWS;  // number of subchunks (rounded up)
  if constexpr (sizeof(T) == 8) num += num & 1;  // next higher even value

  // count non-repeating values
  const int beg = subwarp * num / subwarps;
  const int end = (subwarp + 1) * num / subwarps;
  int cnt = 0;

  for (int i = beg * (4 / sizeof(U)) + sublane; i < end * (4 / sizeof(U)); i += subWS) {
    const int bm = bmin_t[i];
    cnt += __popc(bm);
  }

  for (int i = 1; i < subWS; i *= 2) {
    cnt += __shfl_xor_sync(~0, cnt, i, subWS);
  }
  if (sublane == 0) temp_w[subwarp] = cnt;
  __syncthreads();

  if (tid < WS) {
    const int lane = tid % WS;
    int sum = temp_w[lane];
    for (int i = 1; i < subwarps; i *= 2) {
      const int tmp = __shfl_up_sync(~0, sum, i);
      if (lane >= i) sum += tmp;
    }
    temp_w[lane] = sum;
  }
  __syncthreads();

  // output non-repeating values based on bitmap
  int pos = temp_w[subwarp] - cnt;
  for (int i = beg; i < end; i++) {
    int bm;
    if constexpr (sizeof(U) == 1) {
      bm = (int)bmin_t[i * 4 + sublane / 8] << (sublane & ~7);
      bm |= __shfl_xor_sync(~0, bm, 8, subWS);
      bm |= __shfl_xor_sync(~0, bm, 16, subWS);
    }
    if constexpr (sizeof(U) == 2) {
      bm = (int)bmin_t[i * 2 + sublane / 16] << (sublane & ~15);
      bm |= __shfl_xor_sync(~0, bm, 16, subWS);
    }
    if constexpr (sizeof(U) == 4) {
      bm = bmin_t[i];
    }

    const int offs = __popc(bm & ((1 << sublane) - 1)) - (((bm >> sublane) & 1) ^ 1);
    const T val = (pos + offs < 0) ? 0 : datain[pos + offs];
    const int loc = i * subWS + sublane;
    if (loc < decsize) out[loc] = val;
    pos += __popc(bm);
  }
}


//warp-based one word per thread
template <typename T>
static __device__ inline void d_REdecode1wordperthread(const int decsize, const T* const datain, const T* const bmin, T* const out, int* const temp_w)  // all sizes in number of words
{
  const byte* const bmin_b = (byte*)bmin;
  const int tid = threadIdx.x;
  const int subWS = 32;
  const int subwarps = TPB / subWS;
  const int subwarp = tid / subWS;
  const int sublane = tid % subWS;
  const int warp = tid / WS;
  const int lane = tid % WS;

  // read bitmap and count non-repeating values
  const bool active = (tid < decsize);
  const bool havenonrepval = (active && ((bmin_b[tid / 8] >> (tid % 8)) & 1));
  const int bm = __ballot_sync(~0, havenonrepval);
  const int cnt = __popc(bm);
  if (sublane == 0) temp_w[subwarp] = cnt;
  __syncthreads();

  // compute prefix sum
  int sum = 0;
  if (warp == 0) {
    if (lane < subwarps) sum = temp_w[lane];
    for (int i = 1; i < subwarps; i *= 2) {
      const int tmp = __shfl_up_sync(~0, sum, i);
      if (lane >= i) sum += tmp;
    }
    temp_w[lane] = sum;
  }
  __syncthreads();

  // output values
  if (active) {
    const int loc = temp_w[subwarp] - cnt + __popc(bm & ((1 << sublane) - 1)) - (havenonrepval ^ 1);
    out[tid] = (loc < 0) ? 0 : datain[loc];
  }
}


//warp-based two words per thread
template <typename T>
static __device__ inline void d_REdecode2wordsperthread(const int decsize, const T* const datain, const T* const bmin, T* const out, int* const temp_w)  // all sizes in number of words
{
  const byte* const bmin_b = (byte*)bmin;
  const int tid = threadIdx.x;
  const int subWS = 32;
  const int subwarps = TPB / subWS;
  const int subwarp = tid / subWS;
  const int sublane = tid % subWS;
  const int warp = tid / WS;
  const int lane = tid % WS;

  // read bitmap and count non-repeating values
  const int tid1 = tid * 2;
  const int tid2 = tid1 + 1;
  const bool active1 = (tid1 < decsize);
  const bool active2 = (tid2 < decsize);
  const byte b = active1 ? (bmin_b[tid1 / 8] >> (tid1 % 8)) : 0;
  const bool havenonrepval1 = (active1 && (b & 1));
  const bool havenonrepval2 = (active2 && (b & 2));
  const int bm1 = __ballot_sync(~0, havenonrepval1);
  const int bm2 = __ballot_sync(~0, havenonrepval2);
  const int cnt = __popc(bm1) + __popc(bm2);
  if (sublane == 0) temp_w[subwarp] = cnt;
  __syncthreads();

  // compute prefix sum
  int sum = 0;
  if (warp == 0) {
    if (lane < subwarps) sum = temp_w[lane];
    for (int i = 1; i < subwarps; i *= 2) {
      const int tmp = __shfl_up_sync(~0, sum, i);
      if (lane >= i) sum += tmp;
    }
    temp_w[lane] = sum;
  }
  __syncthreads();

  // output values
  const int common = temp_w[subwarp] - cnt + __popc(bm1 & ((1 << sublane) - 1)) + __popc(bm2 & ((1 << sublane) - 1));
  const int loc1 = common - (havenonrepval1 ^ 1);
  const int loc2 = common + havenonrepval1 - (havenonrepval2 ^ 1);
  if (active1) out[tid1] = (loc1 < 0) ? 0 : datain[loc1];
  if (active2) out[tid2] = (loc2 < 0) ? 0 : datain[loc2];
}


//warp-based four words per thread
template <typename T>
static __device__ inline void d_REdecode4wordsperthread(const int decsize, const T* const datain, const T* const bmin, T* const out, int* const temp_w)  // all sizes in number of words
{
  const byte* const bmin_b = (byte*)bmin;
  const int tid = threadIdx.x;
  const int subWS = 32;
  const int subwarps = TPB / subWS;
  const int subwarp = tid / subWS;
  const int sublane = tid % subWS;
  const int warp = tid / WS;
  const int lane = tid % WS;

  // read bitmap and count non-repeating values
  const int tid1 = tid * 4;
  const int tid2 = tid1 + 1;
  const int tid3 = tid2 + 1;
  const int tid4 = tid3 + 1;
  const bool active1 = (tid1 < decsize);
  const bool active2 = (tid2 < decsize);
  const bool active3 = (tid3 < decsize);
  const bool active4 = (tid4 < decsize);
  const byte b = active1 ? (bmin_b[tid1 / 8] >> (tid1 % 8)) : 0;
  const bool havenonrepval1 = (active1 && (b & 1));
  const bool havenonrepval2 = (active2 && (b & 2));
  const bool havenonrepval3 = (active3 && (b & 4));
  const bool havenonrepval4 = (active4 && (b & 8));
  const int bm1 = __ballot_sync(~0, havenonrepval1);
  const int bm2 = __ballot_sync(~0, havenonrepval2);
  const int bm3 = __ballot_sync(~0, havenonrepval3);
  const int bm4 = __ballot_sync(~0, havenonrepval4);
  const int cnt = __popc(bm1) + __popc(bm2) + __popc(bm3) + __popc(bm4);
  if (sublane == 0) temp_w[subwarp] = cnt;
  __syncthreads();

  // compute prefix sum
  int sum = 0;
  if (warp == 0) {
    if (lane < subwarps) sum = temp_w[lane];
    for (int i = 1; i < subwarps; i *= 2) {
      const int tmp = __shfl_up_sync(~0, sum, i);
      if (lane >= i) sum += tmp;
    }
    temp_w[lane] = sum;
  }
  __syncthreads();

  // output values
  const int common = temp_w[subwarp] - cnt + __popc(bm1 & ((1 << sublane) - 1)) + __popc(bm2 & ((1 << sublane) - 1)) + __popc(bm3 & ((1 << sublane) - 1)) + __popc(bm4 & ((1 << sublane) - 1));
  const int loc1 = common - (havenonrepval1 ^ 1);
  const int loc2 = common + havenonrepval1 - (havenonrepval2 ^ 1);
  const int loc3 = common + havenonrepval1 + havenonrepval2 - (havenonrepval3 ^ 1);
  const int loc4 = common + havenonrepval1 + havenonrepval2 + havenonrepval3 - (havenonrepval4 ^ 1);
  if (active1) out[tid1] = (loc1 < 0) ? 0 : datain[loc1];
  if (active2) out[tid2] = (loc2 < 0) ? 0 : datain[loc2];
  if (active3) out[tid3] = (loc3 < 0) ? 0 : datain[loc3];
  if (active4) out[tid4] = (loc4 < 0) ? 0 : datain[loc4];
}


//thread-based X words per thread, X must be 8, 16, or 32
template <int X, typename T>
static __device__ inline void d_REdecodeXwordsperthread(const int decsize, const T* const datain, const T* const bmin, T* const out, int* const temp_w)  // all sizes in number of words
{
  assert((X == 8) || (X == 16) || (X == 32));

  // read bitmap and count non-repeating values
  const int WPT = X;  // words per thread
  const int tid = threadIdx.x;
  int bmp, cnt = 0;
  if (tid * WPT < decsize) {
    if constexpr (X == 8) bmp = ((byte*)bmin)[tid];
    if constexpr (X == 16) bmp = ((unsigned short*)bmin)[tid];
    if constexpr (X == 32) bmp = ((int*)bmin)[tid];
    cnt = __popc(bmp);
  }

  // compute prefix sum
  int pos = block_prefix_sum(cnt, temp_w) - cnt;

  // output values
  if (tid * WPT < decsize) {
    T val = (bmp & 1) ? 0 : ((pos > 0) ? datain[pos - 1] : 0);
    if ((tid | 31) * WPT + (WPT - 1) < decsize) {
      for (int i = 0; i < WPT; i++) {
        if ((bmp >> i) & 1) val = datain[pos++];
        out[tid * WPT + i] = val;
      }
    } else {
      for (int i = 0; i < WPT; i++) {
        if (tid * WPT + i >= decsize) break;
        if ((bmp >> i) & 1) val = datain[pos++];
        out[tid * WPT + i] = val;
      }
    }
  }
}


template <typename T, int maxsize = CS>  // maxsize in bytes
static __device__ inline void d_REdecode_small(const int decsize, const T* const datain, const T* const bmin, T* const out, int* const temp_w)  // all sizes in number of words
{
  assert((TPB & (TPB - 1)) == 0);
  assert((maxsize & (maxsize - 1)) == 0);
  assert(maxsize % sizeof(T) == 0);
  assert((maxsize / sizeof(T) % TPB == 0) || (maxsize / sizeof(T) < TPB));
  const int wordsperthread = maxsize / sizeof(T) / TPB;
  if constexpr (wordsperthread <= 1) {
    d_REdecode1wordperthread<T>(decsize, datain, bmin, out, temp_w);
  } else if constexpr (wordsperthread == 2) {
    d_REdecode2wordsperthread<T>(decsize, datain, bmin, out, temp_w);
  } else if constexpr (wordsperthread == 4) {
    d_REdecode4wordsperthread<T>(decsize, datain, bmin, out, temp_w);
  } else if constexpr (wordsperthread == 8) {
    d_REdecodeXwordsperthread<8, T>(decsize, datain, bmin, out, temp_w);
  } else if constexpr (wordsperthread == 16) {
    d_REdecodeXwordsperthread<16, T>(decsize, datain, bmin, out, temp_w);
  } else if constexpr (wordsperthread == 32) {
    d_REdecodeXwordsperthread<32, T>(decsize, datain, bmin, out, temp_w);
  } else {
    __trap();
  }
}


template <typename T, int maxsize = CS>
static __device__ inline void d_REdecode(const int decsize, const T* const datain, const T* const bmin, T* const out, int* const temp_w)  // all sizes in number of words
{
  if constexpr (maxsize <= 2048) {
    d_REdecode_small<T, maxsize>(decsize, datain, bmin, out, temp_w);
  } else if ((sizeof(T) >= 4)) {  // at least int aligned
    d_REdecode_specialized(decsize, datain, (int*)bmin, out, temp_w);
  } else if constexpr (sizeof(T) == 2) {  // short aligned
    const int tid = threadIdx.x;
    const int num = (decsize + 15) / 16;  // number of subchunks (rounded up)

    // count non-repeating values
    const int beg = tid * num / TPB;
    const int end = (tid + 1) * num / TPB;
    int cnt = 0;
    for (int i = beg; i < end; i++) {
      const unsigned short bm = bmin[i];
      cnt += __popc((int)bm);
    }
    int pos = block_prefix_sum(cnt, temp_w) - cnt;

    // output non-repeating values based on bitmap
    short val = (pos > 0) ? datain[pos - 1] : 0;
    for (int i = beg; i < end; i++) {
      const unsigned short bm = bmin[i];
      for (int j = 0; j < 16; j++) {
        if ((bm >> j) & 1) val = datain[pos++];
        if (i * 16 + j < decsize) out[i * 16 + j] = val;
      }
    }
  } else {  // byte aligned
    const int tid = threadIdx.x;
    const int num = (decsize + 7) / 8;  // number of subchunks (rounded up)
    long long* const out_l = (long long*)out;
    assert(num <= TPB * 4);

    // count non-zeros
    const int beg = tid * num / TPB;
    const int end = (tid + 1) * num / TPB;
    int bmp = 0;
    for (int i = beg; i < end; i++) {
      bmp |= (int)bmin[i] << (8 * (i - beg));
    }
    const int cnt = __popc(bmp);
    int pos = block_prefix_sum(cnt, temp_w) - cnt;

    // output non-repeating values based on bitmap
    long long val = (pos > 0) ? datain[pos - 1] : 0;
    for (int i = beg; i < end; i++) {
      const byte bm = bmp >> (8 * (i - beg));
      long long lval = 0;
      for (int j = 0; j < 8; j++) {
        if ((bm >> j) & 1) val = datain[pos++];
        lval |= val << (j * 8);
      }
      out_l[i] = lval;
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────
// d_zero_elimination.h — ZE encode/decode primitives.
// ─────────────────────────────────────────────────────────────────────────

//special case for byte and short data
template <typename T, bool check = false>
static __device__ inline bool d_ZEencodebyteshort(const T* const in, const int insize, T* const dataout, int& datasize, T* const bmout, int* const temp_w)
{
  using type = T;
  using ull = unsigned long long;
  const int bitsperword = 8 * sizeof(type);
  const int bitsperlong = 8 * sizeof(ull);
  const int wordsperlong = bitsperlong / bitsperword;
  const int bytesperthread = CS / TPB;
  const ull* const in_l = (ull*)in;
  const int csize = insize * sizeof(T);
  assert(std::is_unsigned<type>::value);

  const int tid = threadIdx.x;
  int bmp = 0, cnt = 0;
  if (tid * bytesperthread < csize) {
    for (int i = 0; i < bytesperthread / sizeof(ull); i++) {
      const ull lval = in_l[tid * (bytesperthread / sizeof(ull)) + i];
      int bm = 0;
      for (int j = 0; j < wordsperlong; j++) {
        const type val = lval >> (j * bitsperword);
        bm |= (val != 0) << j;
      }
      bmp |= bm << (i * wordsperlong);
    }
    if (tid * bytesperthread - (csize - bytesperthread) > 0) {
      bmp &= ~(-1 << ((csize % bytesperthread + sizeof(type) - 1) / sizeof(type)));
    }
    if constexpr (sizeof(type) == 1) {
      bmout[tid * 4] = bmp;
      bmout[tid * 4 + 1] = bmp >> 8;
      bmout[tid * 4 + 2] = bmp >> 16;
      bmout[tid * 4 + 3] = bmp >> 24;
    }
    if constexpr (sizeof(type) == 2) bmout[tid] = bmp;
    if constexpr (sizeof(type) == 4) ((byte*)bmout)[tid] = bmp;
    cnt = __popc(bmp);
  }

  int pos = block_prefix_sum(cnt, temp_w);
  if (tid == TPB - 1) temp_w[WS] = pos;
  if constexpr (check) {
    if (__syncthreads_or(pos > datasize)) return false;
  } else {
    __syncthreads();
  }
  pos -= cnt;

  if (bmp != 0) {
    for (int i = 0; i < bytesperthread / sizeof(ull); i++) {
      const ull lval = in_l[tid * (bytesperthread / sizeof(ull)) + i];
      const int bm = bmp >> (i * wordsperlong);
      for (int j = 0; j < wordsperlong; j++) {
        if ((bm >> j) & 1) dataout[pos++] = lval >> (j * bitsperword);
      }
    }
  }

  datasize = temp_w[WS];
  return true;
}


//warp-based one word per thread
template <typename T, bool check = false>
static __device__ inline bool d_ZEencode1wordperthread(const T* const in, const int insize, T* const dataout, int& datasize, T* const bmout, int* const temp_w)
{
  byte* const bmout_b = (byte*)bmout;
  const int tid = threadIdx.x;
  const int warp = tid / WS;
  const int lane = tid % WS;

  const bool active = (tid < insize);
  const T val = active ? in[tid] : 0;
  const bool havenonzeroval = (active && (val != 0));
  const int bm = __ballot_sync(~0, havenonzeroval);
  const int cnt = __popc(bm);
  const int subwarps = TPB / 32;
  const int sublane = lane;
  const int subwarp = warp;
  if (active && (lane % 8 == 0)) bmout_b[tid / 8] = bm >> lane;
  if constexpr (sizeof(T) > 1) {
    if (warp == 0) {
      const int base = (insize + 7) / 8;
      const int top = (insize + (sizeof(T) * 8 - 1)) / 8;
      if (base + tid < top) bmout_b[base + tid] = 0;
    }
  }
  if (sublane == 0) temp_w[subwarp] = cnt;
  __syncthreads();

  int sum = 0;
  if (warp == 0) {
    if (lane < subwarps) sum = temp_w[lane];
    for (int i = 1; i < subwarps; i *= 2) {
      const int tmp = __shfl_up_sync(~0, sum, i);
      if (lane >= i) sum += tmp;
    }
    temp_w[lane] = sum;
  }
  if constexpr (check) {
    if (__syncthreads_or(sum > datasize)) return false;
  } else {
    __syncthreads();
  }

  if (havenonzeroval) {
    const int loc = temp_w[subwarp] - cnt + __popc(bm & ((1 << sublane) - 1));
    dataout[loc] = val;
  }

  datasize = temp_w[subwarps - 1];
  return true;
}


//warp-based two words per thread
template <typename T, bool check = false>
static __device__ inline bool d_ZEencode2wordsperthread(const T* const in, const int insize, T* const dataout, int& datasize, T* const bmout, int* const temp_w)
{
  byte* const bmout_b = (byte*)bmout;
  const int tid = threadIdx.x;
  const int warp = tid / WS;
  const int lane = tid % WS;

  const int tid1 = tid * 2;
  const int tid2 = tid1 + 1;
  const bool active1 = (tid1 < insize);
  const bool active2 = (tid2 < insize);
  const T val1 = active1 ? in[tid1] : 0;
  const T val2 = active2 ? in[tid2] : 0;
  const bool havenonzeroval1 = (active1 && (val1 != 0));
  const bool havenonzeroval2 = (active2 && (val2 != 0));
  const int bm1 = __ballot_sync(~0, havenonzeroval1);
  const int bm2 = __ballot_sync(~0, havenonzeroval2);
  const int cnt = __popc(bm1) + __popc(bm2);
  const int comb = havenonzeroval1 + havenonzeroval2 * 2;
  const int sublane = lane;
  const int tmp1 = __shfl_sync(~0, comb, lane / 2) >> (lane % 2);
  const int bmlo = __ballot_sync(~0, tmp1 & 1);
  const int tmp2 = __shfl_sync(~0, comb, 16 + lane / 2) >> (lane % 2);
  const int bmhi = __ballot_sync(~0, tmp2 & 1);
  const int subwarps = TPB / 32;
  const int subwarp = warp;
  if (__any_sync(~0, active1) && (lane % 8 == 0)) bmout_b[warp * 8 + lane / 8] = bmlo >> lane;
  if (__any_sync(~0, active2) && (lane % 8 == 0)) bmout_b[warp * 8 + lane / 8 + 4] = bmhi >> lane;
  if constexpr (sizeof(T) > 1) {
    if (warp == 0) {
      const int base = (insize + 7) / 8;
      const int top = (insize + (sizeof(T) * 8 - 1)) / 8;
      if (base + tid < top) bmout_b[base + tid] = 0;
    }
  }
  if (sublane == 0) temp_w[subwarp] = cnt;
  __syncthreads();

  int sum = 0;
  if (warp == 0) {
    if (lane < subwarps) sum = temp_w[lane];
    for (int i = 1; i < subwarps; i *= 2) {
      const int tmp = __shfl_up_sync(~0, sum, i);
      if (lane >= i) sum += tmp;
    }
    temp_w[lane] = sum;
  }
  if constexpr (check) {
    if (__syncthreads_or(sum > datasize)) return false;
  } else {
    __syncthreads();
  }

  int loc = temp_w[subwarp] - cnt + __popc(bm1 & ((1 << sublane) - 1)) + __popc(bm2 & ((1 << sublane) - 1));
  if (havenonzeroval1) dataout[loc++] = val1;
  if (havenonzeroval2) dataout[loc] = val2;

  datasize = temp_w[subwarps - 1];
  return true;
}


//warp-based four words per thread
template <typename T, bool check = false>
static __device__ inline bool d_ZEencode4wordsperthread(const T* const in, const int insize, T* const dataout, int& datasize, T* const bmout, int* const temp_w)
{
  byte* const bmout_b = (byte*)bmout;
  const int tid = threadIdx.x;
  const int warp = tid / WS;
  const int lane = tid % WS;

  const int tid1 = tid * 4;
  const int tid2 = tid1 + 1;
  const int tid3 = tid2 + 1;
  const int tid4 = tid3 + 1;
  const bool active1 = (tid1 < insize);
  const bool active2 = (tid2 < insize);
  const bool active3 = (tid3 < insize);
  const bool active4 = (tid4 < insize);
  const T val1 = active1 ? in[tid1] : 0;
  const T val2 = active2 ? in[tid2] : 0;
  const T val3 = active3 ? in[tid3] : 0;
  const T val4 = active4 ? in[tid4] : 0;
  const bool havenonzeroval1 = (active1 && (val1 != 0));
  const bool havenonzeroval2 = (active2 && (val2 != 0));
  const bool havenonzeroval3 = (active3 && (val3 != 0));
  const bool havenonzeroval4 = (active4 && (val4 != 0));
  const int bm1 = __ballot_sync(~0, havenonzeroval1);
  const int bm2 = __ballot_sync(~0, havenonzeroval2);
  const int bm3 = __ballot_sync(~0, havenonzeroval3);
  const int bm4 = __ballot_sync(~0, havenonzeroval4);
  const int cnt = __popc(bm1) + __popc(bm2) + __popc(bm3) + __popc(bm4);
  const int comb = havenonzeroval1 + havenonzeroval2 * 2 + havenonzeroval3 * 4 + havenonzeroval4 * 8;
  const int sublane = lane;
  const int tmp1 = __shfl_sync(~0, comb, lane / 4) >> (lane % 4);
  const int bmA = __ballot_sync(~0, tmp1 & 1);
  const int tmp2 = __shfl_sync(~0, comb, 8 + lane / 4) >> (lane % 4);
  const int bmB = __ballot_sync(~0, tmp2 & 1);
  const int tmp3 = __shfl_sync(~0, comb, 16 + lane / 4) >> (lane % 4);
  const int bmC = __ballot_sync(~0, tmp3 & 1);
  const int tmp4 = __shfl_sync(~0, comb, 24 + lane / 4) >> (lane % 4);
  const int bmD = __ballot_sync(~0, tmp4 & 1);
  const int subwarps = TPB / 32;
  const int subwarp = warp;
  if (__any_sync(~0, active1) && (lane % 8 == 0)) bmout_b[warp * 16 + lane / 8] = bmA >> lane;
  if (__any_sync(~0, active2) && (lane % 8 == 0)) bmout_b[warp * 16 + lane / 8 + 4] = bmB >> lane;
  if (__any_sync(~0, active3) && (lane % 8 == 0)) bmout_b[warp * 16 + lane / 8 + 8] = bmC >> lane;
  if (__any_sync(~0, active4) && (lane % 8 == 0)) bmout_b[warp * 16 + lane / 8 + 12] = bmD >> lane;
  if constexpr (sizeof(T) > 1) {
    if (warp == 0) {
      const int base = (insize + 7) / 8;
      const int top = (insize + (sizeof(T) * 8 - 1)) / 8;
      if (base + tid < top) bmout_b[base + tid] = 0;
    }
  }
  if (sublane == 0) temp_w[subwarp] = cnt;
  __syncthreads();

  int sum = 0;
  if (warp == 0) {
    if (lane < subwarps) sum = temp_w[lane];
    for (int i = 1; i < subwarps; i *= 2) {
      const int tmp = __shfl_up_sync(~0, sum, i);
      if (lane >= i) sum += tmp;
    }
    temp_w[lane] = sum;
  }
  if constexpr (check) {
    if (__syncthreads_or(sum > datasize)) return false;
  } else {
    __syncthreads();
  }

  int loc = temp_w[subwarp] - cnt + __popc(bm1 & ((1 << sublane) - 1)) + __popc(bm2 & ((1 << sublane) - 1)) + __popc(bm3 & ((1 << sublane) - 1)) + __popc(bm4 & ((1 << sublane) - 1));
  if (havenonzeroval1) dataout[loc++] = val1;
  if (havenonzeroval2) dataout[loc++] = val2;
  if (havenonzeroval3) dataout[loc++] = val3;
  if (havenonzeroval4) dataout[loc] = val4;

  datasize = temp_w[subwarps - 1];
  return true;
}


//thread-based X words per thread, X must be 8, 16, or 32
template <int X, typename T, bool check = false>
static __device__ inline bool d_ZEencodeXwordsperthread(const T* const in, const int insize, T* const dataout, int& datasize, T* const bmout, int* const temp_w)
{
  assert((X == 8) || (X == 16) || (X == 32));
  const int WPT = X;
  const int tid = threadIdx.x;
  int bmp = 0, cnt = 0;
  if (tid * WPT < insize) {
    for (int i = 0; i < WPT; i++) {
      const T val = in[tid * WPT + i];
      bmp |= (val != 0) << i;
    }
    if (tid * WPT - (insize - WPT) > 0) {
      bmp &= ~(-1 << (insize % WPT));
    }
    if constexpr (X == 8) ((byte*)bmout)[tid] = bmp;
    if constexpr (X == 16) ((short*)bmout)[tid] = bmp;
    if constexpr (X == 32) ((int*)bmout)[tid] = bmp;
    cnt = __popc(bmp);
  }

  if constexpr (sizeof(T) * 8 > X) {
    if (tid < WS) {
      const int base = (insize + (X - 1)) / 8;
      const int top = (insize + (sizeof(T) * 8 - 1)) / 8;
      if (base + tid < top) ((byte*)bmout)[base + tid] = 0;
    }
  }

  int pos = block_prefix_sum(cnt, temp_w);
  if (tid == TPB - 1) temp_w[WS] = pos;
  if constexpr (check) {
    if (__syncthreads_or(pos > datasize)) return false;
  } else {
    __syncthreads();
  }
  pos -= cnt;

  if (bmp != 0) {
    for (int i = 0; i < WPT; i++) {
      if ((bmp >> i) & 1) dataout[pos++] = in[tid * WPT + i];
    }
  }

  datasize = temp_w[WS];
  return true;
}


template <typename T, int maxsize = CS, bool check = false>
static __device__ inline bool d_ZEencode(const T* const in, const int insize, T* const dataout, int& datasize, T* const bmout, int* const temp_w)
{
  const int wordsperthread = maxsize / sizeof(T) / TPB;
  if constexpr ((sizeof(T) <= 2) && (maxsize > 2048)) {
    return d_ZEencodebyteshort<T, check>(in, insize, dataout, datasize, bmout, temp_w);
  } else if constexpr (wordsperthread <= 1) {
    return d_ZEencode1wordperthread<T, check>(in, insize, dataout, datasize, bmout, temp_w);
  } else if constexpr (wordsperthread == 2) {
    return d_ZEencode2wordsperthread<T, check>(in, insize, dataout, datasize, bmout, temp_w);
  } else if constexpr (wordsperthread == 4) {
    return d_ZEencode4wordsperthread<T, check>(in, insize, dataout, datasize, bmout, temp_w);
  } else if constexpr (wordsperthread == 8) {
    return d_ZEencodeXwordsperthread<8, T, check>(in, insize, dataout, datasize, bmout, temp_w);
  } else if constexpr (wordsperthread == 16) {
    return d_ZEencodeXwordsperthread<16, T, check>(in, insize, dataout, datasize, bmout, temp_w);
  } else if constexpr (wordsperthread == 32) {
    return d_ZEencodeXwordsperthread<32, T, check>(in, insize, dataout, datasize, bmout, temp_w);
  } else {
    __trap();
    return false;
  }
}


template <typename T, typename U>
static __device__ inline void d_ZEdecode_specialized(const int decsize, const T* const datain, const U* const bmin_t, T* const out, int* const temp_w)
{
  const int subWS = 32;
  const int tid = threadIdx.x;
  const int subwarp = tid / subWS;
  const int subwarps = TPB / subWS;
  const int sublane = tid % subWS;
  int num = (decsize + subWS - 1) / subWS;
  if constexpr (sizeof(T) == 8) num += num & 1;

  const int beg = subwarp * num / subwarps;
  const int end = (subwarp + 1) * num / subwarps;
  int cnt = 0;

  for (int i = beg * (4 / sizeof(U)) + sublane; i < end * (4 / sizeof(U)); i += subWS) {
    const int bm = bmin_t[i];
    cnt += __popc(bm);
  }

  for (int i = 1; i < subWS; i *= 2) {
    cnt += __shfl_xor_sync(~0, cnt, i, subWS);
  }
  if (sublane == 0) temp_w[subwarp] = cnt;
  __syncthreads();

  if (tid < WS) {
    const int lane = tid % WS;
    int sum = temp_w[lane];
    for (int i = 1; i < subwarps; i *= 2) {
      const int tmp = __shfl_up_sync(~0, sum, i);
      if (lane >= i) sum += tmp;
    }
    temp_w[lane] = sum;
  }
  __syncthreads();

  int pos = temp_w[subwarp] - cnt;
  for (int i = beg; i < end; i++) {
    int bm;
    if constexpr (sizeof(U) == 1) {
      bm = (int)bmin_t[i * 4 + sublane / 8] << (sublane & ~7);
      bm |= __shfl_xor_sync(~0, bm, 8, subWS);
      bm |= __shfl_xor_sync(~0, bm, 16, subWS);
    }
    if constexpr (sizeof(U) == 2) {
      bm = (int)bmin_t[i * 2 + sublane / 16] << (sublane & ~15);
      bm |= __shfl_xor_sync(~0, bm, 16, subWS);
    }
    if constexpr (sizeof(U) == 4) {
      bm = bmin_t[i];
    }
    const int offs = __popc(bm & ((1 << sublane) - 1)) - (((bm >> sublane) & 1) ^ 1);
    const int loc = i * subWS + sublane;
    if (loc < decsize) out[loc] = ((bm >> sublane) & 1) ? datain[pos + offs] : (T)0;
    pos += __popc(bm);
  }
}


//warp-based one word per thread
template <typename T>
static __device__ inline void d_ZEdecode1wordperthread(const int decsize, const T* const datain, const T* const bmin, T* const out, int* const temp_w)
{
  const byte* const bmin_b = (byte*)bmin;
  const int tid = threadIdx.x;
  const int subWS = 32;
  const int subwarps = TPB / subWS;
  const int subwarp = tid / subWS;
  const int sublane = tid % subWS;
  const int warp = tid / WS;
  const int lane = tid % WS;

  const bool active = (tid < decsize);
  const bool havenonzeroval = (active && ((bmin_b[tid / 8] >> (tid % 8)) & 1));
  const int bm = __ballot_sync(~0, havenonzeroval);
  const int cnt = __popc(bm);
  if (sublane == 0) temp_w[subwarp] = cnt;
  __syncthreads();

  int sum = 0;
  if (warp == 0) {
    if (lane < subwarps) sum = temp_w[lane];
    for (int i = 1; i < subwarps; i *= 2) {
      const int tmp = __shfl_up_sync(~0, sum, i);
      if (lane >= i) sum += tmp;
    }
    temp_w[lane] = sum;
  }
  __syncthreads();

  if (active) {
    const int loc = temp_w[subwarp] - cnt + __popc(bm & ((1 << sublane) - 1)) - (havenonzeroval ^ 1);
    out[tid] = havenonzeroval ? datain[loc] : (T)0;
  }
}


//warp-based two words per thread
template <typename T>
static __device__ inline void d_ZEdecode2wordsperthread(const int decsize, const T* const datain, const T* const bmin, T* const out, int* const temp_w)
{
  const byte* const bmin_b = (byte*)bmin;
  const int tid = threadIdx.x;
  const int subWS = 32;
  const int subwarps = TPB / subWS;
  const int subwarp = tid / subWS;
  const int sublane = tid % subWS;
  const int warp = tid / WS;
  const int lane = tid % WS;

  const int tid1 = tid * 2;
  const int tid2 = tid1 + 1;
  const bool active1 = (tid1 < decsize);
  const bool active2 = (tid2 < decsize);
  const byte b = active1 ? (bmin_b[tid1 / 8] >> (tid1 % 8)) : 0;
  const bool havenonzeroval1 = (active1 && (b & 1));
  const bool havenonzeroval2 = (active2 && (b & 2));
  const int bm1 = __ballot_sync(~0, havenonzeroval1);
  const int bm2 = __ballot_sync(~0, havenonzeroval2);
  const int cnt = __popc(bm1) + __popc(bm2);
  if (sublane == 0) temp_w[subwarp] = cnt;
  __syncthreads();

  int sum = 0;
  if (warp == 0) {
    if (lane < subwarps) sum = temp_w[lane];
    for (int i = 1; i < subwarps; i *= 2) {
      const int tmp = __shfl_up_sync(~0, sum, i);
      if (lane >= i) sum += tmp;
    }
    temp_w[lane] = sum;
  }
  __syncthreads();

  const int common = temp_w[subwarp] - cnt + __popc(bm1 & ((1 << sublane) - 1)) + __popc(bm2 & ((1 << sublane) - 1));
  const int loc1 = common - (havenonzeroval1 ^ 1);
  const int loc2 = common + havenonzeroval1 - (havenonzeroval2 ^ 1);
  if (active1) out[tid1] = havenonzeroval1 ? datain[loc1] : (T)0;
  if (active2) out[tid2] = havenonzeroval2 ? datain[loc2] : (T)0;
}


//warp-based four words per thread
template <typename T>
static __device__ inline void d_ZEdecode4wordsperthread(const int decsize, const T* const datain, const T* const bmin, T* const out, int* const temp_w)
{
  const byte* const bmin_b = (byte*)bmin;
  const int tid = threadIdx.x;
  const int subWS = 32;
  const int subwarps = TPB / subWS;
  const int subwarp = tid / subWS;
  const int sublane = tid % subWS;
  const int warp = tid / WS;
  const int lane = tid % WS;

  const int tid1 = tid * 4;
  const int tid2 = tid1 + 1;
  const int tid3 = tid2 + 1;
  const int tid4 = tid3 + 1;
  const bool active1 = (tid1 < decsize);
  const bool active2 = (tid2 < decsize);
  const bool active3 = (tid3 < decsize);
  const bool active4 = (tid4 < decsize);
  const byte b = active1 ? (bmin_b[tid1 / 8] >> (tid1 % 8)) : 0;
  const bool havenonzeroval1 = (active1 && (b & 1));
  const bool havenonzeroval2 = (active2 && (b & 2));
  const bool havenonzeroval3 = (active3 && (b & 4));
  const bool havenonzeroval4 = (active4 && (b & 8));
  const int bm1 = __ballot_sync(~0, havenonzeroval1);
  const int bm2 = __ballot_sync(~0, havenonzeroval2);
  const int bm3 = __ballot_sync(~0, havenonzeroval3);
  const int bm4 = __ballot_sync(~0, havenonzeroval4);
  const int cnt = __popc(bm1) + __popc(bm2) + __popc(bm3) + __popc(bm4);
  if (sublane == 0) temp_w[subwarp] = cnt;
  __syncthreads();

  int sum = 0;
  if (warp == 0) {
    if (lane < subwarps) sum = temp_w[lane];
    for (int i = 1; i < subwarps; i *= 2) {
      const int tmp = __shfl_up_sync(~0, sum, i);
      if (lane >= i) sum += tmp;
    }
    temp_w[lane] = sum;
  }
  __syncthreads();

  const int common = temp_w[subwarp] - cnt + __popc(bm1 & ((1 << sublane) - 1)) + __popc(bm2 & ((1 << sublane) - 1)) + __popc(bm3 & ((1 << sublane) - 1)) + __popc(bm4 & ((1 << sublane) - 1));
  const int loc1 = common - (havenonzeroval1 ^ 1);
  const int loc2 = common + havenonzeroval1 - (havenonzeroval2 ^ 1);
  const int loc3 = common + havenonzeroval1 + havenonzeroval2 - (havenonzeroval3 ^ 1);
  const int loc4 = common + havenonzeroval1 + havenonzeroval2 + havenonzeroval3 - (havenonzeroval4 ^ 1);
  if (active1) out[tid1] = havenonzeroval1 ? datain[loc1] : (T)0;
  if (active2) out[tid2] = havenonzeroval2 ? datain[loc2] : (T)0;
  if (active3) out[tid3] = havenonzeroval3 ? datain[loc3] : (T)0;
  if (active4) out[tid4] = havenonzeroval4 ? datain[loc4] : (T)0;
}


//thread-based X words per thread, X must be 8, 16, or 32
template <int X, typename T>
static __device__ inline void d_ZEdecodeXwordsperthread(const int decsize, const T* const datain, const T* const bmin, T* const out, int* const temp_w)
{
  assert((X == 8) || (X == 16) || (X == 32));
  const int WPT = X;
  const int tid = threadIdx.x;
  int bmp, cnt = 0;
  if (tid * WPT < decsize) {
    if constexpr (X == 8) bmp = ((byte*)bmin)[tid];
    if constexpr (X == 16) bmp = ((unsigned short*)bmin)[tid];
    if constexpr (X == 32) bmp = ((int*)bmin)[tid];
    cnt = __popc(bmp);
  }

  int pos = block_prefix_sum(cnt, temp_w) - cnt;

  if (tid * WPT < decsize) {
    if ((tid | 31) * WPT + (WPT - 1) < decsize) {
      for (int i = 0; i < WPT; i++) {
        T val = 0;
        if ((bmp >> i) & 1) val = datain[pos++];
        out[tid * WPT + i] = val;
      }
    } else {
      for (int i = 0; i < WPT; i++) {
        if (tid * WPT + i >= decsize) break;
        T val = 0;
        if ((bmp >> i) & 1) val = datain[pos++];
        out[tid * WPT + i] = val;
      }
    }
  }
}


template <typename T, int maxsize = CS>
static __device__ inline void d_ZEdecode_small(const int decsize, const T* const datain, const T* const bmin, T* const out, int* const temp_w)
{
  const int wordsperthread = maxsize / sizeof(T) / TPB;
  if constexpr (wordsperthread <= 1) {
    d_ZEdecode1wordperthread<T>(decsize, datain, bmin, out, temp_w);
  } else if constexpr (wordsperthread == 2) {
    d_ZEdecode2wordsperthread<T>(decsize, datain, bmin, out, temp_w);
  } else if constexpr (wordsperthread == 4) {
    d_ZEdecode4wordsperthread<T>(decsize, datain, bmin, out, temp_w);
  } else if constexpr (wordsperthread == 8) {
    d_ZEdecodeXwordsperthread<8, T>(decsize, datain, bmin, out, temp_w);
  } else if constexpr (wordsperthread == 16) {
    d_ZEdecodeXwordsperthread<16, T>(decsize, datain, bmin, out, temp_w);
  } else if constexpr (wordsperthread == 32) {
    d_ZEdecodeXwordsperthread<32, T>(decsize, datain, bmin, out, temp_w);
  } else {
    __trap();
  }
}


template <typename T, int maxsize = CS>
static __device__ inline void d_ZEdecode(const int decsize, const T* const datain, const T* const bmin, T* const out, int* const temp_w)
{
  if constexpr (maxsize <= 2048) {
    d_ZEdecode_small<T, maxsize>(decsize, datain, bmin, out, temp_w);
  } else if ((sizeof(T) >= 4)) {
    d_ZEdecode_specialized(decsize, datain, (int*)bmin, out, temp_w);
  } else if constexpr (sizeof(T) == 2) {
    const int tid = threadIdx.x;
    const int num = (decsize + 15) / 16;
    const int beg = tid * num / TPB;
    const int end = (tid + 1) * num / TPB;
    int cnt = 0;
    for (int i = beg; i < end; i++) cnt += __popc((int)(unsigned short)bmin[i]);
    int pos = block_prefix_sum(cnt, temp_w) - cnt;
    for (int i = beg; i < end; i++) {
      const unsigned short bm = bmin[i];
      for (int j = 0; j < 16; j++) {
        short val = 0;
        if ((bm >> j) & 1) val = datain[pos++];
        if (i * 16 + j < decsize) out[i * 16 + j] = val;
      }
    }
  } else {
    const int tid = threadIdx.x;
    const int num = (decsize + 7) / 8;
    long long* const out_l = (long long*)out;
    const int beg = tid * num / TPB;
    const int end = (tid + 1) * num / TPB;
    int bmp = 0;
    for (int i = beg; i < end; i++) bmp |= (int)bmin[i] << (8 * (i - beg));
    const int cnt = __popc(bmp);
    int pos = block_prefix_sum(cnt, temp_w) - cnt;
    for (int i = beg; i < end; i++) {
      const byte bm = bmp >> (8 * (i - beg));
      long long lval = 0;
      for (int j = 0; j < 8; j++) {
        long long val = 0;
        if ((bm >> j) & 1) val = datain[pos++];
        lval |= val << (j * 8);
      }
      out_l[i] = lval;
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────
// d_RRE.h — single-chunk Repetition-Reduction Encode / Decode.
// ─────────────────────────────────────────────────────────────────────────
template <typename T>
static __device__ inline bool d_RRE(int& csize, byte in [CS], byte out [CS], byte temp [CS])
{
  const int tid = threadIdx.x;
  const int size = csize / sizeof(T);  // words in chunk (rounded down)
  const int extra = csize % sizeof(T);
  const int avail = CS - 2 - extra;
  const int bits = 8 * sizeof(T);
  assert(CS == 16384);

  // zero out end of bitmap
  int* const temp_w = (int*)temp;
  byte* const bitmap = (byte*)&temp_w[WS + 1];
  if (csize < CS) {
    for (int i = csize / bits + tid; i < CS / bits; i += TPB) {
      bitmap[i] = 0;
    }
    __syncthreads();
  }

  // copy non-repeating values and generate bitmap
  int wpos = 0;
  if (size > 0) d_REencode((T*)in, size, (T*)out, wpos, (T*)bitmap, temp_w);
  wpos *= sizeof(T);
  if (wpos >= avail) return false;
  __syncthreads();

  // check if not all zeros
  if (wpos != 0) {
    // iteratively compress bitmaps
    int base = 0 / sizeof(T);
    int range = 2048 / sizeof(T);
    int cnt = avail - wpos;
    if (!d_REencode<byte, 2048 / sizeof(T), true>(&bitmap[base], range, &out[wpos], cnt, &bitmap[base + range], temp_w)) return false;
    wpos += cnt;
    __syncthreads();

    base = 2048 / sizeof(T);
    range = 256 / sizeof(T);
    cnt = avail - wpos;
    if (!d_REencode<byte, 256 / sizeof(T), true>(&bitmap[base], range, &out[wpos], cnt, &bitmap[base + range], temp_w)) return false;
    wpos += cnt;
    __syncthreads();

    base = (2048 + 256) / sizeof(T);
    range = 32 / sizeof(T);
    if constexpr (sizeof(T) < 8) {
      cnt = avail - wpos;
      if (!d_REencode<byte, 32 / sizeof(T), true>(&bitmap[base], range, &out[wpos], cnt, &bitmap[base + range], temp_w)) return false;
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


template <typename T>
static __device__ inline void d_iRRE(int& csize, byte in [CS], byte out [CS], byte temp [CS])
{
  const int tid = threadIdx.x;
  int rpos = csize;
  csize = (int)in[--rpos] << 8;  // second byte
  csize |= in[--rpos];  // bottom byte
  const int size = csize / sizeof(T);  // words in chunk (rounded down)
  assert(CS == 16384);
  assert(TPB >= 256);

  // copy leftover byte
  if constexpr (sizeof(T) > 1) {
    const int extra = csize % sizeof(T);
    if (tid < extra) out[csize - extra + tid] = in[rpos - extra + tid];
    rpos -= extra;
  }

  if (rpos == 0) {
    // all zeros
    T* const out_t = (T*)out;
    for (int i = tid; i < size; i += TPB) {
      out_t[i] = 0;
    }
  } else {
    int* const temp_w = (int*)temp;
    byte* const bitmap = (byte*)&temp_w[WS];

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
      d_REdecode<byte, 32 / sizeof(T)>(range, &in[rpos], &bitmap[base + range], &bitmap[base], temp_w);
    }
    __syncthreads();

    rpos -= __syncthreads_count((tid < range * 8) && ((bitmap[base + tid / 8] >> (tid % 8)) & 1));
    base = 2048 / sizeof(T);
    range = 256 / sizeof(T);
    d_REdecode<byte, 256 / sizeof(T)>(range, &in[rpos], &bitmap[base + range], &bitmap[base], temp_w);
    __syncthreads();

    if constexpr (sizeof(T) >= 4) {
      rpos -= __syncthreads_count((tid < range * 8) && ((bitmap[base + tid / 8] >> (tid % 8)) & 1));
    }
    if constexpr (sizeof(T) == 2) {
      int sum = __syncthreads_count((tid < range * 8) && ((bitmap[base + tid / 8] >> (tid % 8)) & 1));
      sum += __syncthreads_count((tid + TPB < range * 8) && ((bitmap[base + (tid + TPB) / 8] >> (tid % 8)) & 1));
      rpos -= sum;
    }
    if constexpr (sizeof(T) == 1) {
      int sum = 0;
      for (int i = 0; i < TPB * 4; i += TPB) {
        sum += __syncthreads_count((tid + i < range * 8) && ((bitmap[base + (tid + i) / 8] >> (tid % 8)) & 1));
      }
      rpos -= sum;
    }
    base = 0 / sizeof(T);
    range = 2048 / sizeof(T);
    d_REdecode<byte, 2048 / sizeof(T)>(range, &in[rpos], &bitmap[base + range], &bitmap[base], temp_w);
    __syncthreads();

    // copy non-repeating values based on bitmap
    if (size > 0) d_REdecode(size, (T*)in, (T*)bitmap, (T*)out, temp_w);
  }
}

// ─────────────────────────────────────────────────────────────────────────
// d_RZE.h — single-chunk Zero-Elimination Encode / Decode (ZE at T-byte words
// + the same recursive bitmap compression as d_RRE).
// ─────────────────────────────────────────────────────────────────────────
template <typename T>
static __device__ inline bool d_RZE(int& csize, byte in [CS], byte out [CS], byte temp [CS])
{
  const int tid = threadIdx.x;
  const int size = csize / sizeof(T);
  const int extra = csize % sizeof(T);
  const int avail = CS - 2 - extra;
  const int bits = 8 * sizeof(T);
  assert(CS == 16384);

  int* const temp_w = (int*)temp;
  byte* const bitmap = (byte*)&temp_w[WS + 1];
  if (csize < CS) {
    for (int i = csize / bits + tid; i < CS / bits; i += TPB) bitmap[i] = 0;
    __syncthreads();
  }

  int wpos = 0;
  if (size > 0) d_ZEencode((T*)in, size, (T*)out, wpos, (T*)bitmap, temp_w);
  wpos *= sizeof(T);
  if (wpos >= avail) return false;
  __syncthreads();

  if (wpos != 0) {
    int base = 0 / sizeof(T);
    int range = 2048 / sizeof(T);
    int cnt = avail - wpos;
    if (!d_REencode<byte, 2048 / sizeof(T), true>(&bitmap[base], range, &out[wpos], cnt, &bitmap[base + range], temp_w)) return false;
    wpos += cnt;
    __syncthreads();

    base = 2048 / sizeof(T);
    range = 256 / sizeof(T);
    cnt = avail - wpos;
    if (!d_REencode<byte, 256 / sizeof(T), true>(&bitmap[base], range, &out[wpos], cnt, &bitmap[base + range], temp_w)) return false;
    wpos += cnt;
    __syncthreads();

    base = (2048 + 256) / sizeof(T);
    range = 32 / sizeof(T);
    if constexpr (sizeof(T) < 8) {
      cnt = avail - wpos;
      if (!d_REencode<byte, 32 / sizeof(T), true>(&bitmap[base], range, &out[wpos], cnt, &bitmap[base + range], temp_w)) return false;
      wpos += cnt;

      base = (2048 + 256 + 32) / sizeof(T);
      range = 4 / sizeof(T);
    }

    if (wpos >= avail - range) return false;
    if (tid < range) out[wpos + tid] = bitmap[base + tid];
    wpos += range;
  }

  if constexpr (sizeof(T) > 1) {
    if (tid < extra) out[wpos + tid] = in[csize - extra + tid];
  }

  const int new_size = wpos + 2 + extra;
  if (tid == 0) {
    out[new_size - 2] = csize;
    out[new_size - 1] = csize >> 8;
  }
  csize = new_size;
  return true;
}


template <typename T>
static __device__ inline void d_iRZE(int& csize, byte in [CS], byte out [CS], byte temp [CS])
{
  const int tid = threadIdx.x;
  int rpos = csize;
  csize = (int)in[--rpos] << 8;
  csize |= in[--rpos];
  const int size = csize / sizeof(T);
  assert(CS == 16384);
  assert(TPB >= 256);

  if constexpr (sizeof(T) > 1) {
    const int extra = csize % sizeof(T);
    if (tid < extra) out[csize - extra + tid] = in[rpos - extra + tid];
    rpos -= extra;
  }

  if (rpos == 0) {
    T* const out_t = (T*)out;
    for (int i = tid; i < size; i += TPB) out_t[i] = 0;
  } else {
    int* const temp_w = (int*)temp;
    byte* const bitmap = (byte*)&temp_w[WS];

    int base, range;
    if constexpr (sizeof(T) == 8) {
      base = (2048 + 256) / sizeof(T);
      range = 32 / sizeof(T);
      rpos -= range;
      if (tid < range) bitmap[base + tid] = in[rpos + tid];
    } else {
      base = (2048 + 256 + 32) / sizeof(T);
      range = 4 / sizeof(T);
      rpos -= range;
      if (tid < range) bitmap[base + tid] = in[rpos + tid];

      rpos -= __syncthreads_count((tid < range * 8) && ((in[rpos + tid / 8] >> (tid % 8)) & 1));
      base = (2048 + 256) / sizeof(T);
      range = 32 / sizeof(T);
      d_REdecode<byte, 32 / sizeof(T)>(range, &in[rpos], &bitmap[base + range], &bitmap[base], temp_w);
    }
    __syncthreads();

    rpos -= __syncthreads_count((tid < range * 8) && ((bitmap[base + tid / 8] >> (tid % 8)) & 1));
    base = 2048 / sizeof(T);
    range = 256 / sizeof(T);
    d_REdecode<byte, 256 / sizeof(T)>(range, &in[rpos], &bitmap[base + range], &bitmap[base], temp_w);
    __syncthreads();

    if constexpr (sizeof(T) >= 4) {
      rpos -= __syncthreads_count((tid < range * 8) && ((bitmap[base + tid / 8] >> (tid % 8)) & 1));
    }
    if constexpr (sizeof(T) == 2) {
      int sum = __syncthreads_count((tid < range * 8) && ((bitmap[base + tid / 8] >> (tid % 8)) & 1));
      sum += __syncthreads_count((tid + TPB < range * 8) && ((bitmap[base + (tid + TPB) / 8] >> (tid % 8)) & 1));
      rpos -= sum;
    }
    if constexpr (sizeof(T) == 1) {
      int sum = 0;
      for (int i = 0; i < TPB * 4; i += TPB) {
        sum += __syncthreads_count((tid + i < range * 8) && ((bitmap[base + (tid + i) / 8] >> (tid % 8)) & 1));
      }
      rpos -= sum;
    }
    base = 0 / sizeof(T);
    range = 2048 / sizeof(T);
    d_REdecode<byte, 2048 / sizeof(T)>(range, &in[rpos], &bitmap[base + range], &bitmap[base], temp_w);
    __syncthreads();

    // copy non-zero values based on bitmap
    if (size > 0) d_ZEdecode(size, (T*)in, (T*)bitmap, (T*)out, temp_w);
  }
}

}  // namespace lc_detail
}  // namespace fz

#undef CS
#undef TPB
#undef WS
