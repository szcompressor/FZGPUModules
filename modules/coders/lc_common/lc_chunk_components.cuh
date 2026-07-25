#pragma once

/**
 * @file modules/coders/lc_common/lc_chunk_components.cuh
 * @brief Vendored LC-framework single-chunk device codecs (shared by RRE, RZE,
 *        RARE, RAZE).
 *
 * Faithful port of the GPU device functions for the LC framework's `RRE`,
 * `RZE`, `RARE`, and `RAZE` lossless components (Burtscher et al.,
 * BSD-3-Clause).  It bundles:
 *   - `block_prefix_sum`            (from `lc/prefix_sum.h`)
 *   - `d_REencode*` / `d_REdecode*`  (from `lc/components/include/d_repetition_elimination.h`)
 *   - `d_ZEencode*` / `d_ZEdecode*`  (from `lc/components/include/d_zero_elimination.h`)
 *   - `d_RRE<T>` / `d_iRRE<T>`        (from `lc/components/include/d_RRE.h`)
 *   - `d_RZE<T>` / `d_iRZE<T>`        (from `lc/components/include/d_RZE.h`)
 *   - `d_RARE<T>` / `d_iRARE<T>`      (from `lc/components/include/d_RARE.h`)
 *   - `d_RAZE<T>` / `d_iRAZE<T>`      (from `lc/components/include/d_RAZE.h`)
 *
 * The functions operate on a single CS-byte chunk held in shared memory and are
 * invoked, one block per chunk, by `rre_stage.cu`, `rze_stage.cu`,
 * `rare_stage.cu`, and `raze_stage.cu`.  They are wrapped in `fz::lc_detail`
 * and declared `static __device__` (internal linkage). The `_N` word-size
 * variants (T = uint8/16/32/64) reproduce LC's RRE_1/2/4/8, RZE_1/2/4/8,
 * RARE_1/2/4/8, and RAZE_1/2/4/8 — the `_N` suffix is the word size, not a
 * recursion-level count. RARE/RAZE are implemented as one shared
 * `d_PRencode`/`d_PRdecode<T, PartialReduceMode>` template (the auto-k
 * generalization of RRE/RZE — see the comment above that section) with
 * `d_RARE`/`d_RAZE` as thin named aliases, mirroring how RRE/RZE already
 * share `d_REencode`/`d_REdecode` for their bitmap recursion.
 *
 * Upstream: https://github.com/burtscher/LC-framework — see THIRD_PARTY.md.
 *
 * The chunk geometry (CS = 16384, TPB = 512, WS = 32) matches the LC defaults
 * the component was tuned and validated against; do not change them without
 * re-checking the word-layout assertions inside d_REencode.
 */

#include "backend/api.h"
#include "backend/atomics.h"
#include "backend/warp.h"
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

  T tmp = fz::backend::shflUp(val, 1, 32);
  if (lane >= 1) val += tmp;
  tmp = fz::backend::shflUp(val, 2, 32);
  if (lane >= 2) val += tmp;
  tmp = fz::backend::shflUp(val, 4, 32);
  if (lane >= 4) val += tmp;
  tmp = fz::backend::shflUp(val, 8, 32);
  if (lane >= 8) val += tmp;
  tmp = fz::backend::shflUp(val, 16, 32);
  if (lane >= 16) val += tmp;

  if (lane == WS - 1) carry[warp] = val;
  __syncthreads();  // carry written

  if constexpr (warps > 1) {
    if (warp == 0) {
      T sum = carry[lane];
      T tmp2 = fz::backend::shflUp(sum, 1, 32);
      if (lane >= 1) sum += tmp2;
      if constexpr (warps > 2) {
        tmp2 = fz::backend::shflUp(sum, 2, 32);
        if (lane >= 2) sum += tmp2;
        if constexpr (warps > 4) {
          tmp2 = fz::backend::shflUp(sum, 4, 32);
          if (lane >= 4) sum += tmp2;
          if constexpr (warps > 8) {
            tmp2 = fz::backend::shflUp(sum, 8, 32);
            if (lane >= 8) sum += tmp2;
            if constexpr (warps > 16) {
              tmp2 = fz::backend::shflUp(sum, 16, 32);
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
template <typename T, int maxsize = CS, bool check = false>
static __device__ inline bool d_REencodebyteshort(const T* const in, const int insize, T* const dataout, int& datasize, T* const bmout, int* const temp_w)  // all sizes in number of words
{
  using type = T;
  using ull = unsigned long long;
  const int bitsperword = 8 * sizeof(type);
  const int bitsperlong = 8 * sizeof(ull);
  const int wordsperlong = bitsperlong / bitsperword;
  const int bytesperthread = maxsize / TPB;
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
    // Write bmp as a *tightly packed* bitmap (no per-thread padding): the
    // consumer (recursive d_REencode/d_REdecode calls, or the caller's own
    // bitmap recursion) expects ceil(insize/8) contiguous bytes, one bit per
    // input word. Each thread contributes exactly
    // bytesperthread/(8*sizeof(type)) meaningful bytes (>= 1 whenever this
    // function is actually selected — see the dispatch guard in d_REencode);
    // the old code hardcoded a 4-byte (or sizeof(type)-byte) stride that only
    // happened to match this value for the original CS=16384/TPB=512 config.
    byte* const bmout_b = (byte*)bmout;
    const int bitmap_bytes_per_thread = bytesperthread / (8 * (int)sizeof(type));
    for (int k = 0; k < bitmap_bytes_per_thread; k++)
      bmout_b[tid * bitmap_bytes_per_thread + k] = (byte)(bmp >> (8 * k));
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
  const int bm = fz::backend::ballotSync32(havenonrepval);
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
      const int tmp = fz::backend::shflUp(sum, i, 32);
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
  const int bm1 = fz::backend::ballotSync32(havenonrepval1);
  const int bm2 = fz::backend::ballotSync32(havenonrepval2);
  const int cnt = __popc(bm1) + __popc(bm2);
  const int comb = havenonrepval1 + havenonrepval2 * 2;
  const int sublane = lane;
  const int tmp1 = fz::backend::shfl(comb, lane / 2, 32) >> (lane % 2);
  const int bmlo = fz::backend::ballotSync32(tmp1 & 1);
  const int tmp2 = fz::backend::shfl(comb, 16 + lane / 2, 32) >> (lane % 2);
  const int bmhi = fz::backend::ballotSync32(tmp2 & 1);
  const int subwarps = TPB / 32;
  const int subwarp = warp;
  if (fz::backend::anySync32(active1) && (lane % 8 == 0)) bmout_b[warp * 8 + lane / 8] = bmlo >> lane;
  if (fz::backend::anySync32(active2) && (lane % 8 == 0)) bmout_b[warp * 8 + lane / 8 + 4] = bmhi >> lane;
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
      const int tmp = fz::backend::shflUp(sum, i, 32);
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
  const int bm1 = fz::backend::ballotSync32(havenonrepval1);
  const int bm2 = fz::backend::ballotSync32(havenonrepval2);
  const int bm3 = fz::backend::ballotSync32(havenonrepval3);
  const int bm4 = fz::backend::ballotSync32(havenonrepval4);
  const int cnt = __popc(bm1) + __popc(bm2) + __popc(bm3) + __popc(bm4);
  const int comb = havenonrepval1 + havenonrepval2 * 2 + havenonrepval3 * 4 + havenonrepval4 * 8;
  const int sublane = lane;
  const int tmp1 = fz::backend::shfl(comb, lane / 4, 32) >> (lane % 4);
  const int bmA = fz::backend::ballotSync32(tmp1 & 1);
  const int tmp2 = fz::backend::shfl(comb, 8 + lane / 4, 32) >> (lane % 4);
  const int bmB = fz::backend::ballotSync32(tmp2 & 1);
  const int tmp3 = fz::backend::shfl(comb, 16 + lane / 4, 32) >> (lane % 4);
  const int bmC = fz::backend::ballotSync32(tmp3 & 1);
  const int tmp4 = fz::backend::shfl(comb, 24 + lane / 4, 32) >> (lane % 4);
  const int bmD = fz::backend::ballotSync32(tmp4 & 1);
  const int subwarps = TPB / 32;
  const int subwarp = warp;
  if (fz::backend::anySync32(active1) && (lane % 8 == 0)) bmout_b[warp * 16 + lane / 8] = bmA >> lane;
  if (fz::backend::anySync32(active2) && (lane % 8 == 0)) bmout_b[warp * 16 + lane / 8 + 4] = bmB >> lane;
  if (fz::backend::anySync32(active3) && (lane % 8 == 0)) bmout_b[warp * 16 + lane / 8 + 8] = bmC >> lane;
  if (fz::backend::anySync32(active4) && (lane % 8 == 0)) bmout_b[warp * 16 + lane / 8 + 12] = bmD >> lane;
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
      const int tmp = fz::backend::shflUp(sum, i, 32);
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
  // The byteshort fast path needs >= 1 full bitmap byte per thread (8 words
  // worth of bits); below that (wordsperthread < 8) its per-thread write
  // stride degenerates (see the comment in d_REencodebyteshort), so fall
  // through to the general path instead, which is correct for any maxsize.
  if constexpr ((sizeof(T) <= 2) && (maxsize > 2048) && (wordsperthread >= 8)) {
    // special case for byte and short data
    return d_REencodebyteshort<T, maxsize, check>(in, insize, dataout, datasize, bmout, temp_w);
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
    cnt += fz::backend::shflXor(cnt, i, 32);
  }
  if (sublane == 0) temp_w[subwarp] = cnt;
  __syncthreads();

  if (tid < WS) {
    const int lane = tid % WS;
    int sum = temp_w[lane];
    for (int i = 1; i < subwarps; i *= 2) {
      const int tmp = fz::backend::shflUp(sum, i, 32);
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
      bm |= fz::backend::shflXor(bm, 8, 32);
      bm |= fz::backend::shflXor(bm, 16, 32);
    }
    if constexpr (sizeof(U) == 2) {
      bm = (int)bmin_t[i * 2 + sublane / 16] << (sublane & ~15);
      bm |= fz::backend::shflXor(bm, 16, 32);
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
  const int bm = fz::backend::ballotSync32(havenonrepval);
  const int cnt = __popc(bm);
  if (sublane == 0) temp_w[subwarp] = cnt;
  __syncthreads();

  // compute prefix sum
  int sum = 0;
  if (warp == 0) {
    if (lane < subwarps) sum = temp_w[lane];
    for (int i = 1; i < subwarps; i *= 2) {
      const int tmp = fz::backend::shflUp(sum, i, 32);
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
  const int bm1 = fz::backend::ballotSync32(havenonrepval1);
  const int bm2 = fz::backend::ballotSync32(havenonrepval2);
  const int cnt = __popc(bm1) + __popc(bm2);
  if (sublane == 0) temp_w[subwarp] = cnt;
  __syncthreads();

  // compute prefix sum
  int sum = 0;
  if (warp == 0) {
    if (lane < subwarps) sum = temp_w[lane];
    for (int i = 1; i < subwarps; i *= 2) {
      const int tmp = fz::backend::shflUp(sum, i, 32);
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
  const int bm1 = fz::backend::ballotSync32(havenonrepval1);
  const int bm2 = fz::backend::ballotSync32(havenonrepval2);
  const int bm3 = fz::backend::ballotSync32(havenonrepval3);
  const int bm4 = fz::backend::ballotSync32(havenonrepval4);
  const int cnt = __popc(bm1) + __popc(bm2) + __popc(bm3) + __popc(bm4);
  if (sublane == 0) temp_w[subwarp] = cnt;
  __syncthreads();

  // compute prefix sum
  int sum = 0;
  if (warp == 0) {
    if (lane < subwarps) sum = temp_w[lane];
    for (int i = 1; i < subwarps; i *= 2) {
      const int tmp = fz::backend::shflUp(sum, i, 32);
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
template <typename T, int maxsize = CS, bool check = false>
static __device__ inline bool d_ZEencodebyteshort(const T* const in, const int insize, T* const dataout, int& datasize, T* const bmout, int* const temp_w)
{
  using type = T;
  using ull = unsigned long long;
  const int bitsperword = 8 * sizeof(type);
  const int bitsperlong = 8 * sizeof(ull);
  const int wordsperlong = bitsperlong / bitsperword;
  const int bytesperthread = maxsize / TPB;
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
    // Tightly packed bitmap write — see the comment in d_REencodebyteshort
    // (same fix, same rationale).
    byte* const bmout_b = (byte*)bmout;
    const int bitmap_bytes_per_thread = bytesperthread / (8 * (int)sizeof(type));
    for (int k = 0; k < bitmap_bytes_per_thread; k++)
      bmout_b[tid * bitmap_bytes_per_thread + k] = (byte)(bmp >> (8 * k));
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
  const int bm = fz::backend::ballotSync32(havenonzeroval);
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
      const int tmp = fz::backend::shflUp(sum, i, 32);
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
  const int bm1 = fz::backend::ballotSync32(havenonzeroval1);
  const int bm2 = fz::backend::ballotSync32(havenonzeroval2);
  const int cnt = __popc(bm1) + __popc(bm2);
  const int comb = havenonzeroval1 + havenonzeroval2 * 2;
  const int sublane = lane;
  const int tmp1 = fz::backend::shfl(comb, lane / 2, 32) >> (lane % 2);
  const int bmlo = fz::backend::ballotSync32(tmp1 & 1);
  const int tmp2 = fz::backend::shfl(comb, 16 + lane / 2, 32) >> (lane % 2);
  const int bmhi = fz::backend::ballotSync32(tmp2 & 1);
  const int subwarps = TPB / 32;
  const int subwarp = warp;
  if (fz::backend::anySync32(active1) && (lane % 8 == 0)) bmout_b[warp * 8 + lane / 8] = bmlo >> lane;
  if (fz::backend::anySync32(active2) && (lane % 8 == 0)) bmout_b[warp * 8 + lane / 8 + 4] = bmhi >> lane;
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
      const int tmp = fz::backend::shflUp(sum, i, 32);
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
  const int bm1 = fz::backend::ballotSync32(havenonzeroval1);
  const int bm2 = fz::backend::ballotSync32(havenonzeroval2);
  const int bm3 = fz::backend::ballotSync32(havenonzeroval3);
  const int bm4 = fz::backend::ballotSync32(havenonzeroval4);
  const int cnt = __popc(bm1) + __popc(bm2) + __popc(bm3) + __popc(bm4);
  const int comb = havenonzeroval1 + havenonzeroval2 * 2 + havenonzeroval3 * 4 + havenonzeroval4 * 8;
  const int sublane = lane;
  const int tmp1 = fz::backend::shfl(comb, lane / 4, 32) >> (lane % 4);
  const int bmA = fz::backend::ballotSync32(tmp1 & 1);
  const int tmp2 = fz::backend::shfl(comb, 8 + lane / 4, 32) >> (lane % 4);
  const int bmB = fz::backend::ballotSync32(tmp2 & 1);
  const int tmp3 = fz::backend::shfl(comb, 16 + lane / 4, 32) >> (lane % 4);
  const int bmC = fz::backend::ballotSync32(tmp3 & 1);
  const int tmp4 = fz::backend::shfl(comb, 24 + lane / 4, 32) >> (lane % 4);
  const int bmD = fz::backend::ballotSync32(tmp4 & 1);
  const int subwarps = TPB / 32;
  const int subwarp = warp;
  if (fz::backend::anySync32(active1) && (lane % 8 == 0)) bmout_b[warp * 16 + lane / 8] = bmA >> lane;
  if (fz::backend::anySync32(active2) && (lane % 8 == 0)) bmout_b[warp * 16 + lane / 8 + 4] = bmB >> lane;
  if (fz::backend::anySync32(active3) && (lane % 8 == 0)) bmout_b[warp * 16 + lane / 8 + 8] = bmC >> lane;
  if (fz::backend::anySync32(active4) && (lane % 8 == 0)) bmout_b[warp * 16 + lane / 8 + 12] = bmD >> lane;
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
      const int tmp = fz::backend::shflUp(sum, i, 32);
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
  // See d_REencode for why wordsperthread >= 8 gates the byteshort fast path.
  if constexpr ((sizeof(T) <= 2) && (maxsize > 2048) && (wordsperthread >= 8)) {
    return d_ZEencodebyteshort<T, maxsize, check>(in, insize, dataout, datasize, bmout, temp_w);
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
    cnt += fz::backend::shflXor(cnt, i, 32);
  }
  if (sublane == 0) temp_w[subwarp] = cnt;
  __syncthreads();

  if (tid < WS) {
    const int lane = tid % WS;
    int sum = temp_w[lane];
    for (int i = 1; i < subwarps; i *= 2) {
      const int tmp = fz::backend::shflUp(sum, i, 32);
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
      bm |= fz::backend::shflXor(bm, 8, 32);
      bm |= fz::backend::shflXor(bm, 16, 32);
    }
    if constexpr (sizeof(U) == 2) {
      bm = (int)bmin_t[i * 2 + sublane / 16] << (sublane & ~15);
      bm |= fz::backend::shflXor(bm, 16, 32);
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
  const int bm = fz::backend::ballotSync32(havenonzeroval);
  const int cnt = __popc(bm);
  if (sublane == 0) temp_w[subwarp] = cnt;
  __syncthreads();

  int sum = 0;
  if (warp == 0) {
    if (lane < subwarps) sum = temp_w[lane];
    for (int i = 1; i < subwarps; i *= 2) {
      const int tmp = fz::backend::shflUp(sum, i, 32);
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
  const int bm1 = fz::backend::ballotSync32(havenonzeroval1);
  const int bm2 = fz::backend::ballotSync32(havenonzeroval2);
  const int cnt = __popc(bm1) + __popc(bm2);
  if (sublane == 0) temp_w[subwarp] = cnt;
  __syncthreads();

  int sum = 0;
  if (warp == 0) {
    if (lane < subwarps) sum = temp_w[lane];
    for (int i = 1; i < subwarps; i *= 2) {
      const int tmp = fz::backend::shflUp(sum, i, 32);
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
  const int bm1 = fz::backend::ballotSync32(havenonzeroval1);
  const int bm2 = fz::backend::ballotSync32(havenonzeroval2);
  const int bm3 = fz::backend::ballotSync32(havenonzeroval3);
  const int bm4 = fz::backend::ballotSync32(havenonzeroval4);
  const int cnt = __popc(bm1) + __popc(bm2) + __popc(bm3) + __popc(bm4);
  if (sublane == 0) temp_w[subwarp] = cnt;
  __syncthreads();

  int sum = 0;
  if (warp == 0) {
    if (lane < subwarps) sum = temp_w[lane];
    for (int i = 1; i < subwarps; i *= 2) {
      const int tmp = fz::backend::shflUp(sum, i, 32);
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
template <typename T, int ChunkBytes = CS>
static __device__ inline bool d_RRE(int& csize, byte in [ChunkBytes], byte out [ChunkBytes], byte temp [ChunkBytes])
{
  // The bitmap recursion below is a fixed 4-level hierarchy (L1/L2/L3/L4 =
  // ChunkBytes/8/64/512/4096 bytes); ChunkBytes >= 4096 keeps L4 >= 1 byte.
  static_assert((ChunkBytes & (ChunkBytes - 1)) == 0 && ChunkBytes >= 4096,
                "ChunkBytes must be a power of two >= 4096");
  constexpr int L1 = ChunkBytes / 8;
  constexpr int L2 = ChunkBytes / 64;
  constexpr int L3 = ChunkBytes / 512;
  constexpr int L4 = ChunkBytes / 4096;

  const int tid = threadIdx.x;
  const int size = csize / sizeof(T);  // words in chunk (rounded down)
  const int extra = csize % sizeof(T);
  const int avail = ChunkBytes - 2 - extra;
  const int bits = 8 * sizeof(T);

  // zero out end of bitmap
  int* const temp_w = (int*)temp;
  byte* const bitmap = (byte*)&temp_w[WS + 1];
  if (csize < ChunkBytes) {
    for (int i = csize / bits + tid; i < ChunkBytes / bits; i += TPB) {
      bitmap[i] = 0;
    }
    __syncthreads();
  }

  // copy non-repeating values and generate bitmap
  int wpos = 0;
  if (size > 0) d_REencode<T, ChunkBytes>((T*)in, size, (T*)out, wpos, (T*)bitmap, temp_w);
  wpos *= sizeof(T);
  if (wpos >= avail) return false;
  __syncthreads();

  // check if not all zeros
  if (wpos != 0) {
    // iteratively compress bitmaps
    int base = 0;
    int range = L1 / sizeof(T);
    int cnt = avail - wpos;
    if (!d_REencode<byte, L1 / sizeof(T), true>(&bitmap[base], range, &out[wpos], cnt, &bitmap[base + range], temp_w)) return false;
    wpos += cnt;
    __syncthreads();

    base = L1 / sizeof(T);
    range = L2 / sizeof(T);
    cnt = avail - wpos;
    if (!d_REencode<byte, L2 / sizeof(T), true>(&bitmap[base], range, &out[wpos], cnt, &bitmap[base + range], temp_w)) return false;
    wpos += cnt;
    __syncthreads();

    base = (L1 + L2) / sizeof(T);
    range = L3 / sizeof(T);
    if constexpr (sizeof(T) < 8) {
      cnt = avail - wpos;
      if (!d_REencode<byte, L3 / sizeof(T), true>(&bitmap[base], range, &out[wpos], cnt, &bitmap[base + range], temp_w)) return false;
      wpos += cnt;

      base = (L1 + L2 + L3) / sizeof(T);
      range = L4 / sizeof(T);
    }

    // output last level of bitmap
    if (wpos >= avail - range) return false;
    if (tid < range) {  // L4 / sizeof(T)
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


template <typename T, int ChunkBytes = CS>
static __device__ inline void d_iRRE(int& csize, byte in [ChunkBytes], byte out [ChunkBytes], byte temp [ChunkBytes])
{
  static_assert((ChunkBytes & (ChunkBytes - 1)) == 0 && ChunkBytes >= 4096,
                "ChunkBytes must be a power of two >= 4096");
  constexpr int L1 = ChunkBytes / 8;
  constexpr int L2 = ChunkBytes / 64;
  constexpr int L3 = ChunkBytes / 512;
  constexpr int L4 = ChunkBytes / 4096;

  const int tid = threadIdx.x;
  int rpos = csize;
  csize = (int)in[--rpos] << 8;  // second byte
  csize |= in[--rpos];  // bottom byte
  const int size = csize / sizeof(T);  // words in chunk (rounded down)
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
      base = (L1 + L2) / sizeof(T);
      range = L3 / sizeof(T);
      // read in last level of bitmap
      rpos -= range;
      if (tid < range) bitmap[base + tid] = in[rpos + tid];
    } else {
      base = (L1 + L2 + L3) / sizeof(T);
      range = L4 / sizeof(T);
      // read in last level of bitmap
      rpos -= range;
      if (tid < range) bitmap[base + tid] = in[rpos + tid];

      rpos -= __syncthreads_count((tid < range * 8) && ((in[rpos + tid / 8] >> (tid % 8)) & 1));
      base = (L1 + L2) / sizeof(T);
      range = L3 / sizeof(T);
      d_REdecode<byte, L3 / sizeof(T)>(range, &in[rpos], &bitmap[base + range], &bitmap[base], temp_w);
    }
    __syncthreads();

    rpos -= __syncthreads_count((tid < range * 8) && ((bitmap[base + tid / 8] >> (tid % 8)) & 1));
    base = L1 / sizeof(T);
    range = L2 / sizeof(T);
    d_REdecode<byte, L2 / sizeof(T)>(range, &in[rpos], &bitmap[base + range], &bitmap[base], temp_w);
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
    base = 0;
    range = L1 / sizeof(T);
    d_REdecode<byte, L1 / sizeof(T)>(range, &in[rpos], &bitmap[base + range], &bitmap[base], temp_w);
    __syncthreads();

    // copy non-repeating values based on bitmap
    if (size > 0) d_REdecode<T, ChunkBytes>(size, (T*)in, (T*)bitmap, (T*)out, temp_w);
  }
}

// ─────────────────────────────────────────────────────────────────────────
// d_RZE.h — single-chunk Zero-Elimination Encode / Decode (ZE at T-byte words
// + the same recursive bitmap compression as d_RRE).
// ─────────────────────────────────────────────────────────────────────────
template <typename T, int ChunkBytes = CS>
static __device__ inline bool d_RZE(int& csize, byte in [ChunkBytes], byte out [ChunkBytes], byte temp [ChunkBytes])
{
  static_assert((ChunkBytes & (ChunkBytes - 1)) == 0 && ChunkBytes >= 4096,
                "ChunkBytes must be a power of two >= 4096");
  constexpr int L1 = ChunkBytes / 8;
  constexpr int L2 = ChunkBytes / 64;
  constexpr int L3 = ChunkBytes / 512;
  constexpr int L4 = ChunkBytes / 4096;

  const int tid = threadIdx.x;
  const int size = csize / sizeof(T);
  const int extra = csize % sizeof(T);
  const int avail = ChunkBytes - 2 - extra;
  const int bits = 8 * sizeof(T);

  int* const temp_w = (int*)temp;
  byte* const bitmap = (byte*)&temp_w[WS + 1];
  if (csize < ChunkBytes) {
    for (int i = csize / bits + tid; i < ChunkBytes / bits; i += TPB) bitmap[i] = 0;
    __syncthreads();
  }

  int wpos = 0;
  if (size > 0) d_ZEencode<T, ChunkBytes>((T*)in, size, (T*)out, wpos, (T*)bitmap, temp_w);
  wpos *= sizeof(T);
  if (wpos >= avail) return false;
  __syncthreads();

  if (wpos != 0) {
    int base = 0;
    int range = L1 / sizeof(T);
    int cnt = avail - wpos;
    if (!d_REencode<byte, L1 / sizeof(T), true>(&bitmap[base], range, &out[wpos], cnt, &bitmap[base + range], temp_w)) return false;
    wpos += cnt;
    __syncthreads();

    base = L1 / sizeof(T);
    range = L2 / sizeof(T);
    cnt = avail - wpos;
    if (!d_REencode<byte, L2 / sizeof(T), true>(&bitmap[base], range, &out[wpos], cnt, &bitmap[base + range], temp_w)) return false;
    wpos += cnt;
    __syncthreads();

    base = (L1 + L2) / sizeof(T);
    range = L3 / sizeof(T);
    if constexpr (sizeof(T) < 8) {
      cnt = avail - wpos;
      if (!d_REencode<byte, L3 / sizeof(T), true>(&bitmap[base], range, &out[wpos], cnt, &bitmap[base + range], temp_w)) return false;
      wpos += cnt;

      base = (L1 + L2 + L3) / sizeof(T);
      range = L4 / sizeof(T);
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


template <typename T, int ChunkBytes = CS>
static __device__ inline void d_iRZE(int& csize, byte in [ChunkBytes], byte out [ChunkBytes], byte temp [ChunkBytes])
{
  static_assert((ChunkBytes & (ChunkBytes - 1)) == 0 && ChunkBytes >= 4096,
                "ChunkBytes must be a power of two >= 4096");
  constexpr int L1 = ChunkBytes / 8;
  constexpr int L2 = ChunkBytes / 64;
  constexpr int L3 = ChunkBytes / 512;
  constexpr int L4 = ChunkBytes / 4096;

  const int tid = threadIdx.x;
  int rpos = csize;
  csize = (int)in[--rpos] << 8;
  csize |= in[--rpos];
  const int size = csize / sizeof(T);
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
      base = (L1 + L2) / sizeof(T);
      range = L3 / sizeof(T);
      rpos -= range;
      if (tid < range) bitmap[base + tid] = in[rpos + tid];
    } else {
      base = (L1 + L2 + L3) / sizeof(T);
      range = L4 / sizeof(T);
      rpos -= range;
      if (tid < range) bitmap[base + tid] = in[rpos + tid];

      rpos -= __syncthreads_count((tid < range * 8) && ((in[rpos + tid / 8] >> (tid % 8)) & 1));
      base = (L1 + L2) / sizeof(T);
      range = L3 / sizeof(T);
      d_REdecode<byte, L3 / sizeof(T)>(range, &in[rpos], &bitmap[base + range], &bitmap[base], temp_w);
    }
    __syncthreads();

    rpos -= __syncthreads_count((tid < range * 8) && ((bitmap[base + tid / 8] >> (tid % 8)) & 1));
    base = L1 / sizeof(T);
    range = L2 / sizeof(T);
    d_REdecode<byte, L2 / sizeof(T)>(range, &in[rpos], &bitmap[base + range], &bitmap[base], temp_w);
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
    base = 0;
    range = L1 / sizeof(T);
    d_REdecode<byte, L1 / sizeof(T)>(range, &in[rpos], &bitmap[base + range], &bitmap[base], temp_w);
    __syncthreads();

    // copy non-zero values based on bitmap
    if (size > 0) d_ZEdecode<T, ChunkBytes>(size, (T*)in, (T*)bitmap, (T*)out, temp_w);
  }
}

// ─────────────────────────────────────────────────────────────────────────
// d_RARE.h / d_RAZE.h — single-chunk partial-bit-reduction encode/decode.
//
// RARE (Mode::REPEAT) generalizes d_RRE: instead of a binary "matches prev
// in full, or is dropped entirely" test, it histograms how many top bits of
// `val ^ prev` are zero across the whole chunk, picks one global cut `keep`
// (0 <= keep < bits) that maximizes total bit savings via a warp-shuffle
// max-reduction, then every element whose top `bits-keep` bits match prev
// stores only its bottom `keep` bits — bit-packed, not byte-aligned, via
// shift/mask accumulation across word boundaries (`d_PRencode`'s "encode
// values and generate bitmap" loop below) plus a block-scoped atomic OR for
// each thread's own trailing partial word. RAZE (Mode::ZERO) is the same
// algorithm with the per-element predicate replaced by the value's own
// leading-zero count instead of a match against the previous element — the
// two are otherwise textually identical, exactly like the RRE/RZE split
// above (d_repetition_elimination.h vs d_zero_elimination.h).
//
// The 4-level recursive bitmap compression (L1/L2/L3/L4 hierarchy) is
// byte-for-byte identical to d_RRE/d_RZE and reuses d_REencode/d_REdecode
// unchanged — no new code there. What's new is everything about picking
// `keep` and the partial-bit-pack encode/decode, neither of which has any
// precedent in d_RRE/d_RZE (which only ever emit whole words).
//
// Backend note: the warp reduction below reuses the same facade functions
// (`fz::backend::shflUp/shflXor`, `ballotSync32`) that block_prefix_sum and
// d_REencode already use, so no new warp.h work was needed. The histogram
// accumulation and trailing partial-word write use block-scoped atomics
// (`fz::backend::atomicAddBlock/atomicOrBlock`, include/backend/atomics.h)
// that are new to this codebase and — unlike shuffle/ballot — have not yet
// been verified on HIP hardware; see that header's doc comment.
// ─────────────────────────────────────────────────────────────────────────

enum class PartialReduceMode { REPEAT, ZERO };

template <typename T, PartialReduceMode Mode, int ChunkBytes = CS>
static __device__ inline bool d_PRencode(int& csize, byte in [ChunkBytes], byte out [ChunkBytes], byte temp [ChunkBytes])
{
  static_assert((ChunkBytes & (ChunkBytes - 1)) == 0 && ChunkBytes >= 4096,
                "ChunkBytes must be a power of two >= 4096");
  constexpr int L1 = ChunkBytes / 8;
  constexpr int L2 = ChunkBytes / 64;
  constexpr int L3 = ChunkBytes / 512;
  constexpr int L4 = ChunkBytes / 4096;

  const int tid = threadIdx.x;
  const int size = csize / sizeof(T);  // words in chunk (rounded down)
  const int extra = csize % sizeof(T);
  const int bits = 8 * sizeof(T);
  T* const in_t = (T*)in;
  T* const out_t = (T*)out;

  // histogram how many top bits repeat (REPEAT) / are zero (ZERO)
  int* const count = (int*)temp;  // 66 ints reserved regardless of `bits`,
                                   // so the bitmap pointer below (&count[66])
                                   // sits at a fixed offset for every T
  if (tid < bits) count[tid] = 0;
  __syncthreads();

  bool allmatch = true;
  for (int i = tid; i < size; i += TPB) {
    T predicate;
    if constexpr (Mode == PartialReduceMode::REPEAT) {
      const T prev = (i > 0) ? in_t[i - 1] : 0;
      predicate = in_t[i] ^ prev;
    } else {
      predicate = in_t[i];
    }
    if (predicate != 0) allmatch = false;
    int keep;
    if constexpr (sizeof(T) == 8) {
      keep = (predicate == 0) ? 0 : (64 - __builtin_clzll((unsigned long long)predicate));
    } else {
      keep = (predicate == 0) ? 0 : (32 - __builtin_clz((unsigned int)predicate));
    }
    fz::backend::atomicAddBlock(&count[keep], 1);
  }
  allmatch = __syncthreads_and(allmatch);

  // special case: every element matches in full (all-repeat / all-zero)
  if (allmatch) {
    if constexpr (sizeof(T) > 1) {
      if (tid < extra) out[tid] = in[csize - extra + tid];
    }
    if (tid == WS) {
      out[extra] = bits + 1;  // special "keep" value
      out[extra + 1] = csize;
      out[extra + 2] = csize >> 8;
    }
    csize = extra + 3;
    return true;
  }

  // prefix sum counts and find the keep value with maximum savings
  if constexpr (bits <= WS) {
    if (tid < WS) {  // first warp only
      const int lane = tid;
      int pfs = count[lane];
      int tmp = fz::backend::shflUp(pfs, 1, 32);
      if (lane >= 1) pfs += tmp;
      tmp = fz::backend::shflUp(pfs, 2, 32);
      if (lane >= 2) pfs += tmp;
      tmp = fz::backend::shflUp(pfs, 4, 32);
      if (lane >= 4) pfs += tmp;
      if constexpr (bits > 8) {
        tmp = fz::backend::shflUp(pfs, 8, 32);
        if (lane >= 8) pfs += tmp;
        if constexpr (bits > 16) {
          tmp = fz::backend::shflUp(pfs, 16, 32);
          if (lane >= 16) pfs += tmp;
        }
      }
      count[lane] = pfs;

      // determine maximum savings
      const int sav = (bits <= lane) ? -1 : ((bits - lane) * pfs);
      int val = sav;
      val = max(val, fz::backend::shflXor(val, 1, 32));
      val = max(val, fz::backend::shflXor(val, 2, 32));
      val = max(val, fz::backend::shflXor(val, 4, 32));
      val = max(val, fz::backend::shflXor(val, 8, 32));
      val = max(val, fz::backend::shflXor(val, 16, 32));
      const uint32_t bal = fz::backend::ballotSync32(val == sav);
      const int who = __ffs((int)bal) - 1;
      if (lane == 0) count[64] = val;  // saved
      if (lane == 0) count[65] = who;  // keep
    }
  } else {
    static_assert(bits == WS * 2, "unsupported word size for partial-reduce keep selection");
    if (tid < WS) {  // first warp only
      const int l0 = tid * 2;
      const int l1 = l0 + 1;
      const int lane = tid;
      const int c1 = count[l1];
      int pfs = count[l0] + c1;
      int tmp = fz::backend::shflUp(pfs, 1, 32);
      if (lane >= 1) pfs += tmp;
      tmp = fz::backend::shflUp(pfs, 2, 32);
      if (lane >= 2) pfs += tmp;
      tmp = fz::backend::shflUp(pfs, 4, 32);
      if (lane >= 4) pfs += tmp;
      tmp = fz::backend::shflUp(pfs, 8, 32);
      if (lane >= 8) pfs += tmp;
      tmp = fz::backend::shflUp(pfs, 16, 32);
      if (lane >= 16) pfs += tmp;
      count[l1] = pfs;
      count[l0] = pfs - c1;

      // determine maximum savings
      const int sav1 = (bits - l1) * pfs;
      const int sav0 = (bits - l0) * (pfs - c1);
      int val = max(sav0, sav1);
      val = max(val, fz::backend::shflXor(val, 1, 32));
      val = max(val, fz::backend::shflXor(val, 2, 32));
      val = max(val, fz::backend::shflXor(val, 4, 32));
      val = max(val, fz::backend::shflXor(val, 8, 32));
      val = max(val, fz::backend::shflXor(val, 16, 32));
      const uint32_t bal = fz::backend::ballotSync32((val == sav0) || (val == sav1));
      const int who = __ffs((int)bal) - 1;
      if (lane == who) {
        count[64] = val;  // saved
        count[65] = (val == sav0) ? l0 : l1;  // keep
      }
    }
  }
  __syncthreads();

  const int saved = count[64];
  const int keep = count[65];
  const int countk = count[keep];

  // special case: no savings possible, keep every bit of every word
  if (saved == 0) {
    if (csize + 3 >= ChunkBytes) return false;
    for (int i = tid; i < size; i += TPB) {
      out_t[i] = in_t[i];
    }
    if constexpr (sizeof(T) > 1) {
      if (tid < extra) out[csize - extra + tid] = in[csize - extra + tid];
    }
    if (tid == 0) {
      out[csize] = bits;  // special "keep" value
      out[csize + 1] = csize;
      out[csize + 2] = csize >> 8;
    }
    csize += 3;
    return true;
  }

  // keep some bits from each matching value (0 <= keep < bits)

  // zero out for atomic OR (trailing partial words get OR'd into this below)
  for (int i = tid + size - countk; i < size - countk + ((countk * keep + bits - 1) / bits); i += TPB) {
    out_t[i] = 0;
  }
  __syncthreads();

  byte* const bitmap = (byte*)&count[66];

  const T tmask = ~(T)0 << keep;  // 111...00
  const T bmask = ~tmask;  // 000...11

  // determine wpos1 (exclusive prefix sum of per-thread full-value counts)
  const int ept = (((size + TPB - 1) / TPB + 7) / 8) * 8;  // elements per thread (multiple of 8)
  int cnt = 0;
  T prevMask = 0;
  if constexpr (Mode == PartialReduceMode::REPEAT) {
    prevMask = ((tid * ept == 0) || (tid * ept >= size)) ? 0 : (in_t[tid * ept - 1] & tmask);
    for (int i = tid * ept; i < min((tid + 1) * ept, size); i++) {
      const T val = in_t[i];
      if (prevMask != (val & tmask)) {
        prevMask = val & tmask;
        cnt++;
      }
    }
  } else {
    for (int i = tid * ept; i < min((tid + 1) * ept, size); i++) {
      if (0 != (in_t[i] & tmask)) cnt++;
    }
  }
  int wpos1 = block_prefix_sum(cnt, temp) - cnt;
  int wloc2 = bits * (size - countk) + (tid * ept - wpos1) * keep;
  int wpos2 = wloc2 / bits;

  // encode values and generate bitmap
  T oval = 0;
  byte bmp = 0;
  if constexpr (Mode == PartialReduceMode::REPEAT) {
    prevMask = ((tid * ept == 0) || (tid * ept >= size)) ? 0 : (in_t[tid * ept - 1] & tmask);
  }
  for (int i = tid * ept; i < min((tid + 1) * ept, size); i++) {
    const T val = in_t[i];
    bool full;
    if constexpr (Mode == PartialReduceMode::REPEAT) {
      full = (prevMask != (val & tmask));
      if (full) prevMask = val & tmask;
    } else {
      full = (0 != (val & tmask));
    }
    if (full) {
      bmp |= 1 << (i % 8);
      out_t[wpos1++] = val;  // output all bits
    } else {
      if (keep != 0) {
        // output bottom bits only
        const T bval = val & bmask;
        const int shift = wloc2 % bits;
        const int bms = bits - shift;
        oval |= bval << shift;
        if (bms <= keep) {
          out_t[wpos2++] = oval;
          oval = bval >> bms;
        }
        wloc2 += keep;
      }
    }
    if ((i % 8) == 7) {
      bitmap[i / 8] = bmp;
      bmp = 0;
    }
  }
  if ((tid * ept < size) && ((tid + 1) * ept > size)) {
    bitmap[size / 8] = bmp;
  }

  // zero out rest of bitmap
  for (int i = tid + (size + 7) / 8; i < ChunkBytes / bits; i += TPB) {
    bitmap[i] = 0;
  }
  __syncthreads();

  // output last partial word. Guarded by `tid * ept < size` — threads whose
  // entire ept-range lies past `size` ("empty" threads, unavoidable whenever
  // ept*TPB overshoots size, e.g. size=2048/ept=8/TPB=512 leaves 256 empty
  // trailing threads) never entered the loop above, so wloc2 still holds its
  // pre-loop seed value `bits*(size-countk) + (tid*ept - wpos1)*keep`. Every
  // active thread's wpos1 collapses to the same block-wide total (size-countk)
  // once all real contributions are in, so for an empty thread this reduces
  // to `bits*(size-countk) + (tid*ept - (size-countk))*keep` — spuriously
  // nonzero whenever keep > 0, growing without bound as tid increases past
  // the last active thread. Without this guard that garbage wpos2 drives an
  // out-of-bounds atomicOrBlock (silent corruption for small `bits`, a
  // shared-memory fault for bits=64 where the overshoot is largest). The
  // upstream LC source has this exact same gap — reproduced independently
  // during RARE/RAZE bring-up via WordSize8RoundTrip (compute-sanitizer:
  // invalid __shared__ access from atomicOrBlock) and ConstantRunRoundTrip
  // (silent wrong output, the in-bounds-but-wrong-address case for bits=8).
  if ((tid * ept < size) && (wloc2 % bits) != 0) {
    if constexpr (bits == 8) {
      fz::backend::atomicOrBlock((int*)&out_t[wpos2 & ~3], (int)oval << (8 * (wpos2 & 3)));
    } else if constexpr (bits == 16) {
      fz::backend::atomicOrBlock((int*)&out_t[wpos2 & ~1], (int)oval << (16 * (wpos2 & 1)));
    } else if constexpr (bits == 32) {
      fz::backend::atomicOrBlock((int*)&out_t[wpos2], (int)oval);
    } else {
      fz::backend::atomicOrBlock((unsigned long long*)&out_t[wpos2], (unsigned long long)oval);
    }
  }

  // iteratively compress bitmaps (identical to d_RRE/d_RZE)
  const int avail = ChunkBytes - 3 - extra;
  int wpos = (bits * (size - countk) + keep * countk + 7) / 8;
  int base = 0;
  int range = L1 / sizeof(T);
  cnt = avail - wpos;
  if (!d_REencode<byte, L1 / sizeof(T), true>(&bitmap[base], range, &out[wpos], cnt, &bitmap[base + range], (int*)temp)) return false;
  wpos += cnt;
  __syncthreads();

  base = L1 / sizeof(T);
  range = L2 / sizeof(T);
  cnt = avail - wpos;
  if (!d_REencode<byte, L2 / sizeof(T), true>(&bitmap[base], range, &out[wpos], cnt, &bitmap[base + range], (int*)temp)) return false;
  wpos += cnt;
  __syncthreads();

  base = (L1 + L2) / sizeof(T);
  range = L3 / sizeof(T);
  if constexpr (sizeof(T) < 8) {
    cnt = avail - wpos;
    if (!d_REencode<byte, L3 / sizeof(T), true>(&bitmap[base], range, &out[wpos], cnt, &bitmap[base + range], (int*)temp)) return false;
    wpos += cnt;

    base = (L1 + L2 + L3) / sizeof(T);
    range = L4 / sizeof(T);
  }

  if (wpos >= avail - range) return false;
  if (tid < range) {
    out[wpos + tid] = bitmap[base + tid];
  }
  wpos += range;

  // copy leftover bytes
  if constexpr (sizeof(T) > 1) {
    if (tid < extra) out[wpos + tid] = in[csize - extra + tid];
  }

  // output old csize and update csize
  const int new_size = wpos + 3 + extra;
  if (tid == 0) {
    out[new_size - 3] = keep;  // "keep" value
    out[new_size - 2] = csize;
    out[new_size - 1] = csize >> 8;
  }
  csize = new_size;
  return true;
}


template <typename T, PartialReduceMode Mode, int ChunkBytes = CS>
static __device__ inline void d_PRdecode(int& csize, byte in [ChunkBytes], byte out [ChunkBytes], byte temp [ChunkBytes])
{
  static_assert((ChunkBytes & (ChunkBytes - 1)) == 0 && ChunkBytes >= 4096,
                "ChunkBytes must be a power of two >= 4096");
  constexpr int L1 = ChunkBytes / 8;
  constexpr int L2 = ChunkBytes / 64;
  constexpr int L3 = ChunkBytes / 512;
  constexpr int L4 = ChunkBytes / 4096;

  // read in original csize and keep
  const int oldsize = in[csize - 2] + ((int)in[csize - 1] << 8);
  const int keep = in[csize - 3];

  const int tid = threadIdx.x;
  const int bits = 8 * sizeof(T);
  const int size = oldsize / sizeof(T);
  const int extra = oldsize % sizeof(T);
  T* const in_t = (T*)in;
  T* const out_t = (T*)out;
  assert(TPB >= 256);

  if (keep == bits + 1) {
    // special case: all values (other than extra) are zero — an all-repeat
    // chunk (Mode::REPEAT) with an implicit-0 predecessor is exactly an
    // all-zero chunk, same reconstruction as Mode::ZERO
    for (int i = tid; i < size; i += TPB) {
      out_t[i] = 0;
    }
  } else if (keep == bits) {  // keep all bits
    for (int i = tid; i < size; i += TPB) {
      out_t[i] = in_t[i];
    }
  } else {  // keep some bits from each value (0 <= keep < bits)
    int rpos = csize - 3 - extra;
    int* const temp_w = (int*)temp;
    byte* const bitmap = (byte*)&temp_w[WS];

    // iteratively decompress bitmaps (identical to d_RRE/d_RZE)
    int base, range;
    if constexpr (sizeof(T) == 8) {
      base = (L1 + L2) / sizeof(T);
      range = L3 / sizeof(T);
      rpos -= range;
      if (tid < range) bitmap[base + tid] = in[rpos + tid];
    } else {
      base = (L1 + L2 + L3) / sizeof(T);
      range = L4 / sizeof(T);
      rpos -= range;
      if (tid < range) bitmap[base + tid] = in[rpos + tid];

      rpos -= __syncthreads_count((tid < range * 8) && ((in[rpos + tid / 8] >> (tid % 8)) & 1));
      base = (L1 + L2) / sizeof(T);
      range = L3 / sizeof(T);
      d_REdecode<byte, L3 / sizeof(T)>(range, &in[rpos], &bitmap[base + range], &bitmap[base], temp_w);
    }
    __syncthreads();

    rpos -= __syncthreads_count((tid < range * 8) && ((bitmap[base + tid / 8] >> (tid % 8)) & 1));
    base = L1 / sizeof(T);
    range = L2 / sizeof(T);
    d_REdecode<byte, L2 / sizeof(T)>(range, &in[rpos], &bitmap[base + range], &bitmap[base], temp_w);
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
    base = 0;
    range = L1 / sizeof(T);
    d_REdecode<byte, L1 / sizeof(T)>(range, &in[rpos], &bitmap[base + range], &bitmap[base], temp_w);
    __syncthreads();

    // determine rpos1 etc.
    const int ept = (((size + TPB - 1) / TPB + 7) / 8) * 8;
    int cnt = 0;
    for (int i = tid * ept; i < min((tid + 1) * ept, size); i += 8) {
      cnt += __builtin_popcount((int)bitmap[i / 8]);
    }
    int rpos1 = block_prefix_sum(cnt, temp_w) - cnt;
    const int count = temp_w[TPB / WS - 1];
    int rloc2 = bits * count + (tid * ept - rpos1) * keep;
    int rpos2 = rloc2 / bits;

    // decode values
    const T tmask = ~(T)0 << keep;
    const T bmask = ~tmask;
    T ival = in_t[rpos2++];
    byte bmp;
    T prev = 0;
    if constexpr (Mode == PartialReduceMode::REPEAT) {
      const T seed = (rpos1 > 0) ? in_t[rpos1 - 1] : 0;
      prev = seed & tmask;
    }
    for (int i = tid * ept; i < min((tid + 1) * ept, size); i++) {
      if ((i % 8) == 0) bmp = bitmap[i / 8];
      T val = 0;
      if ((bmp >> (i % 8)) & 1) {
        val = in_t[rpos1++];  // read all bits
        if constexpr (Mode == PartialReduceMode::REPEAT) prev = val & tmask;
      } else {
        // Baseline for a "matched" element: REPEAT reconstructs to `prev`
        // (matters when keep==0, where bmask==0 so there are no bottom bits
        // to OR in at all — the original LC decode gets this for free by
        // declaring `val` outside the loop so it persists from the previous
        // iteration; here it's made explicit instead of relying on that).
        // ZERO's baseline is 0, matching a value whose top bits are all zero
        // and keep==0 leaves no bottom bits either.
        if constexpr (Mode == PartialReduceMode::REPEAT) val = prev;
        if (keep != 0) {
          // read only bottom bits
          const int shift = rloc2 % bits;
          const int bms = bits - shift;
          T res = ival >> shift;
          if (bms <= keep) {
            ival = in_t[rpos2++];
            res |= ival << bms;
          }
          rloc2 += keep;
          const T bot = res & bmask;
          if constexpr (Mode == PartialReduceMode::REPEAT) {
            val = prev | bot;
          } else {
            val = bot;
          }
        }
      }
      out_t[i] = val;
    }
  }

  // copy leftover bytes
  if constexpr (sizeof(T) > 1) {
    if (tid < extra) {
      out[oldsize - extra + tid] = in[csize - 3 - extra + tid];
    }
  }

  csize = oldsize;
}


// Named aliases matching the d_RRE/d_RZE call-site convention.
template <typename T, int ChunkBytes = CS>
static __device__ inline bool d_RARE(int& csize, byte in [ChunkBytes], byte out [ChunkBytes], byte temp [ChunkBytes])
{
  return d_PRencode<T, PartialReduceMode::REPEAT, ChunkBytes>(csize, in, out, temp);
}

template <typename T, int ChunkBytes = CS>
static __device__ inline void d_iRARE(int& csize, byte in [ChunkBytes], byte out [ChunkBytes], byte temp [ChunkBytes])
{
  d_PRdecode<T, PartialReduceMode::REPEAT, ChunkBytes>(csize, in, out, temp);
}

template <typename T, int ChunkBytes = CS>
static __device__ inline bool d_RAZE(int& csize, byte in [ChunkBytes], byte out [ChunkBytes], byte temp [ChunkBytes])
{
  return d_PRencode<T, PartialReduceMode::ZERO, ChunkBytes>(csize, in, out, temp);
}

template <typename T, int ChunkBytes = CS>
static __device__ inline void d_iRAZE(int& csize, byte in [ChunkBytes], byte out [ChunkBytes], byte temp [ChunkBytes])
{
  d_PRdecode<T, PartialReduceMode::ZERO, ChunkBytes>(csize, in, out, temp);
}

}  // namespace lc_detail
}  // namespace fz

#undef CS
#undef TPB
#undef WS
