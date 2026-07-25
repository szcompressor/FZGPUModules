#pragma once

/**
 * @file modules/coders/lc_common/lc_clog_components.cuh
 * @brief Vendored LC-framework single-chunk device codecs for CLOG + HCLOG.
 *
 * Faithful port of the GPU device functions for the LC framework's `CLOG` and
 * `HCLOG` lossless components (Burtscher et al., BSD-3-Clause). Unlike
 * RRE/RZE/RARE/RAZE (`lc_chunk_components.cuh`), CLOG/HCLOG share none of
 * that machinery — no bitmap, no `block_prefix_sum`, no `d_REencode` — so
 * they're vendored in this separate header:
 *   - `d_CLOG<T>` / `d_iCLOG<T>`   (from `lc/components/include/d_CLOG.h`)
 *   - `d_HCLOG<T>` / `d_iHCLOG<T>` (from `lc/components/include/d_HCLOG.h`)
 *
 * **Algorithm.** The chunk is split into a fixed **32 subchunks** (`SC = 32`,
 * tied to the 32-lane warp used to parallelize the per-subchunk reduction —
 * unlike `ChunkBytes`, this is not a tunable). Each subchunk independently
 * finds its own max value and computes the minimum bit-width `logn` needed to
 * represent it — then every element in that subchunk is truncated to exactly
 * `logn` bits. This is lossless: no element in the subchunk exceeds the max,
 * so none needs more than `logn` bits. `T` must be unsigned (`uint8/16/32/64`
 * only — no signed word-size variants exist upstream). No per-element
 * full/dropped decision as in RRE/RAZE — every element in a subchunk is
 * always truncated to the same width, and no auxiliary bitmap is needed.
 *
 * HCLOG adds one wrinkle: for each subchunk, it *also* computes the max
 * bit-width after reinterpreting every value via TCMS (the same
 * two's-complement -> sign-magnitude / zigzag transform `ZigzagStage` uses)
 * and picks whichever of the two needs fewer bits, recording the choice as
 * one flag bit per subchunk (32 bits total, stored as a single `int` at the
 * very front of the chunk). This lets HCLOG do well on bipolar-looking data
 * (e.g. already-negabinary/zigzag-shaped residuals) that would otherwise
 * bit-pack poorly as raw unsigned magnitudes.
 *
 * Implemented as a single shared `d_CLOGencode`/`d_CLOGdecode<T, Mode,
 * ChunkBytes>` template (`Mode` = `PLAIN` for CLOG, `WITH_TCMS_FALLBACK` for
 * HCLOG) — the two upstream LC headers are identical apart from the
 * flag-selection/TCMS logic, mirroring how `d_PRencode`/`d_PRdecode` unify
 * RARE/RAZE in the sibling header. `d_CLOG`/`d_iCLOG`/`d_HCLOG`/`d_iHCLOG`
 * are thin named aliases matching the upstream call-site convention.
 *
 * Upstream: https://github.com/burtscher/LC-framework — see THIRD_PARTY.md.
 *
 * The chunk geometry (TPB = 512, WS = 32) matches `lc_chunk_components.cuh`;
 * `ChunkBytes` (LC's `CS`) is a template parameter, not hardcoded, following
 * the same generalization RRE/RZE/RARE/RAZE already went through.
 */

#include "backend/api.h"
#include "backend/atomics.h"
#include "backend/warp.h"
#include <cstdint>
#include <cassert>
#include <type_traits>

#define TPB 512  // threads per block (power of two, >= 128; must equal SC * (TPB/32))
#define WS  32    // warp size

namespace fz {
namespace lc_detail {

using byte = unsigned char;

enum class CLogMode { PLAIN, WITH_TCMS_FALLBACK };

// ─────────────────────────────────────────────────────────────────────────
// d_CLOG.h / d_HCLOG.h — single-chunk 32-subchunk adaptive bit-width packing.
// ─────────────────────────────────────────────────────────────────────────
template <typename T, CLogMode Mode, int ChunkBytes>
static __device__ inline bool d_CLOGencode(int& csize, byte in [ChunkBytes], byte out [ChunkBytes], byte temp [ChunkBytes])
{
  static_assert(std::is_unsigned<T>::value, "CLOG/HCLOG require an unsigned word type");
  static_assert((ChunkBytes & (ChunkBytes - 1)) == 0, "ChunkBytes must be a power of two");

  const int tid = threadIdx.x;
  const int lane = threadIdx.x % WS;
  const int warp = threadIdx.x / WS;
  const int warps = TPB / WS;

  constexpr int TB = sizeof(T) * 8;  // bits per T word
  constexpr int SC = 32;             // subchunks (fixed — matches warp width)
  // Counter bits: smallest CB such that a subchunk's bit-count (0..TB) fits.
  constexpr int CB = (sizeof(T) == 1) ? 4 : (sizeof(T) == 2) ? 5 : (sizeof(T) == 4) ? 6 : 7;
  static_assert((1 << CB) > TB && (1 << (CB - 1)) <= TB, "CB miscalculated for this T");
  static_assert(WS >= SC, "warp must cover all subchunks");
  constexpr int flagBits = (Mode == CLogMode::WITH_TCMS_FALLBACK) ? SC : 0;

  T* const in_t = (T*)in;
  int* const out_i = (int*)out;
  constexpr int TB_i = 32;
  const int size = csize / sizeof(T);

  byte* const ln = temp;
  int* const bits = (int*)&temp[SC];
  int* const total_bits = (int*)&bits[WS];

  if constexpr (Mode == CLogMode::WITH_TCMS_FALLBACK) {
    if (tid == 0) out_i[0] = 0;  // flags = 0
    __syncthreads();
  }

  // determine bits needed for each subchunk
  for (int i = warp; i < SC; i += warps) {
    const int beg = i * size / SC;
    const int end = (i + 1) * size / SC;

    T max_val1 = 0;
    T max_val2 = 0;
    for (int j = beg + lane; j < end; j += WS) {
      const T val1 = in_t[j];
      max_val1 = max(max_val1, val1);
      if constexpr (Mode == CLogMode::WITH_TCMS_FALLBACK) {
        const T val2 = (val1 << 1) ^ (T)(((typename std::make_signed<T>::type)val1) >> (TB - 1));  // TCMS
        max_val2 = max(max_val2, val2);
      }
    }

    max_val1 = max(max_val1, fz::backend::shflXor(max_val1, 1, 32));
    max_val1 = max(max_val1, fz::backend::shflXor(max_val1, 2, 32));
    max_val1 = max(max_val1, fz::backend::shflXor(max_val1, 4, 32));
    max_val1 = max(max_val1, fz::backend::shflXor(max_val1, 8, 32));
    max_val1 = max(max_val1, fz::backend::shflXor(max_val1, 16, 32));
    if constexpr (Mode == CLogMode::WITH_TCMS_FALLBACK) {
      max_val2 = max(max_val2, fz::backend::shflXor(max_val2, 1, 32));
      max_val2 = max(max_val2, fz::backend::shflXor(max_val2, 2, 32));
      max_val2 = max(max_val2, fz::backend::shflXor(max_val2, 4, 32));
      max_val2 = max(max_val2, fz::backend::shflXor(max_val2, 8, 32));
      max_val2 = max(max_val2, fz::backend::shflXor(max_val2, 16, 32));
    }

    if (lane == 0) {
      T max_val = max_val1;
      if constexpr (Mode == CLogMode::WITH_TCMS_FALLBACK) {
        max_val = min(max_val1, max_val2);
        if (max_val2 < max_val1) fz::backend::atomicOrBlock(out_i, 1 << i);
      }
      int cnt = 0;
      if (max_val != 0) {
        cnt = (sizeof(T) == 8) ? (64 - __clzll((unsigned long long)max_val))
                                : (32 - __clz((unsigned int)max_val));
      }
      bits[i] = cnt * (end - beg);
      ln[i] = (byte)cnt;
    }
  }
  __syncthreads();

  // warp prefix sum over bits (exclusive, plus grand total)
  if (warp == 0) {
    const int org = bits[lane];
    int val = org;
    int tmp = fz::backend::shflUp(val, 1, 32);
    if (lane >= 1) val += tmp;
    tmp = fz::backend::shflUp(val, 2, 32);
    if (lane >= 2) val += tmp;
    tmp = fz::backend::shflUp(val, 4, 32);
    if (lane >= 4) val += tmp;
    tmp = fz::backend::shflUp(val, 8, 32);
    if (lane >= 8) val += tmp;
    tmp = fz::backend::shflUp(val, 16, 32);
    if (lane >= 16) val += tmp;
    bits[lane] = val - org;
    if (lane == SC - 1) *total_bits = val;
  }
  __syncthreads();

  // check if encoded data fits
  const int extra = csize % sizeof(T);
  const int newsize = (flagBits + 16 + CB * SC + *total_bits + 7) / 8;
  if (newsize + extra >= ChunkBytes) return false;

  // clear out buffer (HCLOG: preserve the already-written flags word at out_i[0])
  constexpr int clearStart = (Mode == CLogMode::WITH_TCMS_FALLBACK) ? 1 : 0;
  for (int i = tid + clearStart; i < (newsize + (int)sizeof(int) - 1) / (int)sizeof(int); i += TPB) out_i[i] = 0;
  __syncthreads();

  // encode logn values
  if (lane < SC) {
    const int val = ln[lane];
    const int loc = flagBits + 16 + (CB * lane);
    const int pos = loc / TB_i;
    const int shift = loc % TB_i;
    fz::backend::atomicOrBlock(&out_i[pos], val << shift);
    if (TB_i - CB < shift) {
      fz::backend::atomicOrBlock(&out_i[pos + 1], val >> (TB_i - shift));
    }
  }

  // encode data values
  const int flags = (Mode == CLogMode::WITH_TCMS_FALLBACK) ? out_i[0] : 0;
  for (int i = warp; i < SC; i += warps) {
    const int logn = ln[i];
    const int beg = i * size / SC;
    const int end = (i + 1) * size / SC;
    const int offs = flagBits + 16 + CB * SC + bits[i];
    const bool flag = (Mode == CLogMode::WITH_TCMS_FALLBACK) && ((flags >> i) & 1);
    for (int j = beg + lane; j < end; j += WS) {
      T val = in_t[j];
      if constexpr (Mode == CLogMode::WITH_TCMS_FALLBACK) {
        if (flag) val = (val << 1) ^ (T)(((typename std::make_signed<T>::type)val) >> (TB - 1));  // TCMS
      }
      const int loc = offs + (j - beg) * logn;
      if constexpr (sizeof(T) < 8) {
        const int pos = loc / TB_i;
        const int shift = loc % TB_i;
        fz::backend::atomicOrBlock(&out_i[pos], (int)((unsigned int)val << shift));
        if (TB_i - logn < shift) {
          fz::backend::atomicOrBlock(&out_i[pos + 1], (int)((unsigned int)val >> (TB_i - shift)));
        }
      } else {
        long long* const out_l = (long long*)out;
        const int pos = loc / TB;
        const int shift = loc % TB;
        fz::backend::atomicOrBlock((unsigned long long*)&out_l[pos], (unsigned long long)val << shift);
        if (TB - logn < shift) {
          fz::backend::atomicOrBlock((unsigned long long*)&out_l[pos + 1], (unsigned long long)val >> (TB - shift));
        }
      }
    }
  }
  __syncthreads();

  // copy leftover bytes
  if constexpr (sizeof(T) > 1) {
    if (tid < extra) out[newsize + tid] = in[csize - extra + tid];
  }

  // record old csize (short tag right after the flags word, if any)
  if (tid == 0) {
    *((short*)&out[flagBits / 8]) = (short)csize;
  }
  csize = newsize + extra;
  return true;
}


template <typename T, CLogMode Mode, int ChunkBytes>
static __device__ inline void d_CLOGdecode(int& csize, byte in [ChunkBytes], byte out [ChunkBytes], byte temp [ChunkBytes])
{
  static_assert(std::is_unsigned<T>::value, "CLOG/HCLOG require an unsigned word type");

  const int tid = threadIdx.x;
  const int lane = threadIdx.x % WS;
  const int warp = threadIdx.x / WS;

  constexpr int TB = sizeof(T) * 8;
  constexpr int SC = 32;
  constexpr int CB = (sizeof(T) == 1) ? 4 : (sizeof(T) == 2) ? 5 : (sizeof(T) == 4) ? 6 : 7;
  static_assert((1 << CB) > TB && (1 << (CB - 1)) <= TB, "CB miscalculated for this T");
  static_assert(WS >= SC, "warp must cover all subchunks");
  constexpr int flagBits = (Mode == CLogMode::WITH_TCMS_FALLBACK) ? SC : 0;

  T* const in_t = (T*)in;
  T* const out_t = (T*)out;
  byte* const ln = (byte*)temp;
  int* const bits = (int*)&temp[SC];
  const int orig_csize = *((short*)&in[flagBits / 8]);
  const int size = orig_csize / sizeof(T);
  const int flags = (Mode == CLogMode::WITH_TCMS_FALLBACK) ? *((int*)in) : 0;

  // decode logn values + per-subchunk bit offsets
  const T mask = ((1 << CB) - 1);
  if (warp == 0) {
    T res = 0;
    if (lane < SC) {
      const int loc = flagBits + 16 + (lane * CB);
      const int pos = loc / TB;
      const int shift = loc % TB;
      res = in_t[pos] >> shift;
      if (TB - CB < shift) {
        res |= in_t[pos + 1] << (TB - shift);
      }
      res &= mask;
      ln[lane] = (byte)res;
    }

    const int beg = lane * size / SC;
    const int end = (lane + 1) * size / SC;
    const int org = res * (end - beg);
    int val = org;
    int tmp = fz::backend::shflUp(val, 1, 32);
    if (lane >= 1) val += tmp;
    tmp = fz::backend::shflUp(val, 2, 32);
    if (lane >= 2) val += tmp;
    tmp = fz::backend::shflUp(val, 4, 32);
    if (lane >= 4) val += tmp;
    tmp = fz::backend::shflUp(val, 8, 32);
    if (lane >= 8) val += tmp;
    tmp = fz::backend::shflUp(val, 16, 32);
    if (lane >= 16) val += tmp;
    bits[lane] = val - org;
  }
  __syncthreads();

  // decode data values
  for (int i = warp; i < SC; i += TPB / WS) {
    const int logn = ln[i];
    const int beg = i * size / SC;
    const int end = (i + 1) * size / SC;
    const T dmask = (sizeof(T) < 8) ? (T)((1ULL << logn) - 1)
                                     : ((logn == 64) ? (T)(~0ULL) : (T)((1ULL << logn) - 1));
    const int offs = flagBits + 16 + SC * CB + bits[i];
    const bool flag = (Mode == CLogMode::WITH_TCMS_FALLBACK) && ((flags >> i) & 1);
    for (int j = beg + lane; j < end; j += WS) {
      const int loc = offs + (j - beg) * logn;
      const int pos = loc / TB;
      const int shift = loc % TB;
      T res = in_t[pos] >> shift;
      if (TB - logn < shift) {
        res |= in_t[pos + 1] << (TB - shift);
      }
      T val = res & dmask;
      if constexpr (Mode == CLogMode::WITH_TCMS_FALLBACK) {
        if (flag) {
          val = (T)((val >> 1) ^ (T)(((typename std::make_signed<T>::type)(val << (TB - 1))) >> (TB - 1)));  // inverse TCMS
        }
      }
      out_t[j] = val;
    }
  }

  // copy leftover bytes
  if constexpr (sizeof(T) > 1) {
    const int extra = orig_csize % sizeof(T);
    if (tid < extra) out[orig_csize - extra + tid] = in[csize - extra + tid];
  }
  csize = orig_csize;
}


// Named aliases matching the upstream d_CLOG/d_HCLOG call-site convention.
template <typename T, int ChunkBytes>
static __device__ inline bool d_CLOG(int& csize, byte in [ChunkBytes], byte out [ChunkBytes], byte temp [ChunkBytes])
{
  return d_CLOGencode<T, CLogMode::PLAIN, ChunkBytes>(csize, in, out, temp);
}

template <typename T, int ChunkBytes>
static __device__ inline void d_iCLOG(int& csize, byte in [ChunkBytes], byte out [ChunkBytes], byte temp [ChunkBytes])
{
  d_CLOGdecode<T, CLogMode::PLAIN, ChunkBytes>(csize, in, out, temp);
}

template <typename T, int ChunkBytes>
static __device__ inline bool d_HCLOG(int& csize, byte in [ChunkBytes], byte out [ChunkBytes], byte temp [ChunkBytes])
{
  return d_CLOGencode<T, CLogMode::WITH_TCMS_FALLBACK, ChunkBytes>(csize, in, out, temp);
}

template <typename T, int ChunkBytes>
static __device__ inline void d_iHCLOG(int& csize, byte in [ChunkBytes], byte out [ChunkBytes], byte temp [ChunkBytes])
{
  d_CLOGdecode<T, CLogMode::WITH_TCMS_FALLBACK, ChunkBytes>(csize, in, out, temp);
}

}  // namespace lc_detail
}  // namespace fz

#undef TPB
#undef WS
