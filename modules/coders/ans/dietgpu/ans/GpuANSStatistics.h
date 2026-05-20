/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Adapted for FZGPUModules: namespace renamed from multibyte_ans to fz::ans;
 * stripped histogramSingle, histogramBatch, ansHistogramBatch (replaced by
 * fz::module::GPU_histogram_generic in ans_stage.cu).
 */
#ifndef FZ_ANS_DIETGPU_ANS_GPUANSSTATISTICS_H
#define FZ_ANS_DIETGPU_ANS_GPUANSSTATISTICS_H

#pragma once

#include "GpuANSCodec.h"
#include "utils/DeviceUtils.h"
#include "utils/PtxUtils.h"
#include "utils/StaticUtils.h"

#include <cmath>
#include <cub/cub.cuh>
#include <memory>

namespace fz { namespace ans {

// sum that allows passing in smem for usage, so as to avoid a trailing
// syncthreads and associated latency
template <int Threads>
__device__ inline int
blockSum(int warpId, int laneId, int valForSum, int* smem) {
  static_assert(isEvenDivisor(Threads, kWarpSize), "");
  constexpr int kWarps = Threads / kWarpSize;

  auto allSum = warpReduceAllSum(valForSum);

  if (laneId == 0) {
    smem[warpId] = allSum;
  }
  __syncthreads();

  if (warpId == 0) {
    int v = laneId < kWarps ? smem[laneId] : 0;
    v = warpReduceAllSum(v);

    if (laneId == 0) {
      smem[0] = v;
    }
  }

  __syncthreads();

  // trailing syncthreads is elsewhere
  return smem[0];
}

// Function that allows normalization of symbol probabilities with a varying
// (statically known) number of threads, to allow for kernel fusion as needed
// Stand-alone normalization will use Threads == kNumSymbols (256)
template <int Threads>
__device__ void normalizeProbabilitiesFromHistogram(
    // Size 256 histogram in gmem
    const uint32_t* __restrict__ counts,
    uint32_t totalNum,
    int probBits,
    uint4* __restrict__ table) {

  static_assert(
      kNumSymbols == Threads || isEvenDivisor(kNumSymbols, uint32_t(Threads)),
      "");

  constexpr int kNumSymPerThread =
      kNumSymbols == Threads ? 1 : (kNumSymbols / Threads);

  // There's nothing to do if the input array in the batch was of zero size
  if (totalNum == 0) {
    return;
  }

  constexpr int kWarps = Threads / kWarpSize;
  uint32_t kProbWeight = 1 << probBits;
  int tid = threadIdx.x;
  int warpId = tid / kWarpSize;
  int laneId = getLaneId();

  // Load the current count and compute the min/max non-zero values, then
  // perform an approximate quantization
  uint32_t qProb[kNumSymPerThread];

  int qProbSum = 0;

#pragma unroll
  for (int i = 0; i < kNumSymPerThread; ++i) {
    int curSym = i * Threads + tid;
    uint32_t count = counts[curSym];

    // Rough initial quantization
    qProb[i] = kProbWeight * ((float)count / (float)totalNum);

    // All weights for symbols present must be > 0
    qProb[i] = (count > 0 && qProb[i] == 0) ? 1 : qProb[i];

    qProbSum += qProb[i];
  }

  // Sum qProbSym across all threads
  __shared__ int smemSum[kWarps];
  qProbSum = blockSum<Threads>(warpId, laneId, qProbSum, smemSum);

  // In order to use radix sorting, and also in order to only sort a single
  // word, pack both the weight and index into a single integer
  uint32_t sortedPair[kNumSymPerThread];

#pragma unroll
  for (int i = 0; i < kNumSymPerThread; ++i) {
    int curSym = i * Threads + tid;
    sortedPair[i] = (qProb[i] << 16) | curSym;
  }

  using Sort = cub::BlockRadixSort<uint32_t, Threads, kNumSymPerThread>;
  __shared__ typename Sort::TempStorage smemSort;
  Sort(smemSort).SortDescending(sortedPair);

  uint32_t tidSymbol[kNumSymPerThread];

#pragma unroll
  for (int i = 0; i < kNumSymPerThread; ++i) {
    tidSymbol[i] = sortedPair[i] & 0xffffU;
    qProb[i] = sortedPair[i] >> 16;
  }

  // How far below (positive) or above (negative) our current first-pass
  // quantization is from our target sum 2^probBits
  int diff = (int)kProbWeight - (int)qProbSum;

  if (diff > 0) {
    while (diff > 0) {
      int iterToApply = diff < kNumSymbols ? diff : kNumSymbols;

#pragma unroll
      for (int i = 0; i < kNumSymPerThread; ++i) {
        int curSym = tidSymbol[i];
        if (curSym < iterToApply) {
          qProb[i] += 1;
        }
      }

      diff -= iterToApply;
    }
  } else if (diff < 0) {
    diff = -diff;

    while (diff > 0) {
      int qNumGt1s = 0;

#pragma unroll
      for (int i = 0; i < kNumSymPerThread; ++i) {
        qNumGt1s += (int)(qProb[i] > 1);
      }

      qNumGt1s = blockSum<Threads>(warpId, laneId, qNumGt1s, smemSum);
      __syncthreads();

      int iterToApply = diff < qNumGt1s ? diff : qNumGt1s;
      assert(iterToApply > 0);
      int startIndex = qNumGt1s - iterToApply;

#pragma unroll
      for (int i = 0; i < kNumSymPerThread; ++i) {
        int curSym = tid * kNumSymPerThread + i;
        if (curSym >= startIndex && curSym < qNumGt1s) {
          qProb[i] -= 1;
        }
      }

      diff -= iterToApply;

      __syncthreads();
    }
  }

  __shared__ uint32_t smemPdf[kNumSymbols];

#pragma unroll
  for (int i = 0; i < kNumSymPerThread; ++i) {
    smemPdf[tidSymbol[i]] = qProb[i];
  }

  __syncthreads();

  uint32_t symPdf[kNumSymPerThread];
#pragma unroll
  for (int i = 0; i < kNumSymPerThread; ++i) {
    int curSym = tid * kNumSymPerThread + i;
    symPdf[i] = smemPdf[curSym];
  }

  using Scan = cub::BlockScan<uint32_t, Threads>;
  __shared__ typename Scan::TempStorage smemScan;

  uint32_t symCdf[kNumSymPerThread];
  Scan(smemScan).ExclusiveSum(symPdf, symCdf);

  // Compute divisor information (constant division via integer
  // multiplication + shift)
  uint32_t shift[kNumSymPerThread];
  uint32_t magic[kNumSymPerThread];

#pragma unroll
  for (int i = 0; i < kNumSymPerThread; ++i) {
    shift[i] = 32 - __clz(symPdf[i] - 1);
    constexpr uint64_t one = 1;
    uint64_t magic64 = 0;
    if (symPdf[i] != 0)
      magic64 =
          ((one << 32) * ((one << shift[i]) - symPdf[i])) / symPdf[i] + 1;
    magic[i] = (uint32_t)magic64;
  }

#pragma unroll
  for (int i = 0; i < kNumSymPerThread; ++i) {
    int curSym = tid * kNumSymPerThread + i;
    table[curSym] = uint4{symPdf[i], symCdf[i], magic[i], shift[i]};
  }
}

template <int Threads>
__global__ void quantizeWeights(
    const uint32_t* __restrict__ counts,
    uint32_t inSize_dev,
    int probBits,
    uint4* __restrict__ table) {
  normalizeProbabilitiesFromHistogram<Threads>(
      counts,
      inSize_dev,
      probBits,
      table);
}

inline void ansCalcWeights(
    int probBits,
    uint32_t inSize_dev,
    const uint32_t* histogram_dev,
    uint4* table_dev,
    cudaStream_t stream) {
  constexpr int kThreads = kNumSymbols;
  quantizeWeights<kThreads><<<1, kThreads, 0, stream>>>(
      histogram_dev, inSize_dev, probBits, table_dev);
}

}} // namespace fz::ans

#endif // FZ_ANS_DIETGPU_ANS_GPUANSSTATISTICS_H
