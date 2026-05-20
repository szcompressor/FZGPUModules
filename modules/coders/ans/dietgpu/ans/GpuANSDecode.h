/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Adapted for FZGPUModules: namespace renamed from multibyte_ans to fz::ans;
 * stripped ansDecode() host function (called directly from ans_stage.cu instead).
 */
#ifndef FZ_ANS_DIETGPU_ANS_GPUANSDECODE_H
#define FZ_ANS_DIETGPU_ANS_GPUANSDECODE_H

#pragma once

#include "GpuANSCodec.h"
#include "utils/DeviceUtils.h"
#include "utils/PtxUtils.h"
#include "utils/StaticUtils.h"
#include <cmath>
#include <cub/block/block_scan.cuh>
#include <memory>

namespace fz { namespace ans {

// We are limited to 11 bits of probability resolution
// (worst case, prec = 12, pdf == 2^12, single symbol. 2^12 cannot be
// represented in 12 bits)
inline __device__ uint32_t
packDecodeLookup(uint32_t sym, uint32_t pdf, uint32_t cdf) {
  static_assert(sizeof(ANSDecodedT) == 1, "");
  // [31:20] cdf
  // [19:8] pdf
  // [7:0] symbol
  return (cdf << 20) | (pdf << 8) | sym;
}

inline __device__ void
unpackDecodeLookup(uint32_t v, uint32_t& sym, uint32_t& pdf, uint32_t& cdf) {
  // [31:20] cdf
  // [19:8] pdf
  // [7:0] symbol
  sym = v & 0xffU;
  v >>= 8;
  pdf = v & 0xfffU;
  v >>= 12;
  cdf = v;
}

template <int ProbBits>
__device__ void decodeOneWarp(
    ANSStateT& state,
    uint32_t compressedOffset,
    const ANSEncodedT* __restrict__ in,
    const uint32_t* lookup,
    uint32_t& outNumRead,
    ANSDecodedT& outSym) {
  constexpr ANSStateT StateMask = (ANSStateT(1) << ProbBits) - ANSStateT(1);

  auto s_bar = state & StateMask;

  uint32_t sym;
  uint32_t pdf;
  uint32_t sMinusCdf;
  unpackDecodeLookup(lookup[s_bar], sym, pdf, sMinusCdf);

  outSym = sym;
  state = pdf * (state >> ProbBits) + ANSStateT(sMinusCdf);

  bool read = state < kANSMinState;
  auto vote = __ballot_sync(0xffffffff, read);
  auto prefix = __popc(vote & getLaneMaskGe());

  if (read) {
    auto v = in[-prefix];
    state = (state << kANSEncodedBits) + ANSStateT(v);
  }

  outNumRead = __popc(vote);
}

template <int ProbBits>
__device__ void decodeOnePartialWarp(
    bool valid,
    ANSStateT& state,
    uint32_t compressedOffset,
    const ANSEncodedT* __restrict__ in,
    const uint32_t* lookup,
    uint32_t& outNumRead,
    ANSDecodedT& outSym) {
  constexpr ANSStateT StateMask = (ANSStateT(1) << ProbBits) - ANSStateT(1);

  auto s_bar = state & StateMask;

  uint32_t sym;
  uint32_t pdf;
  uint32_t sMinusCdf;
  unpackDecodeLookup(lookup[s_bar], sym, pdf, sMinusCdf);

  if (valid) {
    outSym = sym;
    state = pdf * (state >> ProbBits) + ANSStateT(sMinusCdf);
  }

  bool read = valid && (state < kANSMinState);
  auto vote = __ballot_sync(0xffffffff, read);
  auto prefix = __popc(vote & getLaneMaskGe());

  if (read) {
    auto v = in[-prefix];
    state = (state << kANSEncodedBits) + ANSStateT(v);
  }

  outNumRead = __popc(vote);
}

template <int ProbBits>
__device__ void ansDecodeWarpBlock(
    int laneId,
    ANSStateT state,
    uint32_t uncompressedWords,
    uint32_t compressedWords,
    const ANSEncodedT* __restrict__ in,
    BatchWriter& writer,
    const uint32_t* __restrict__ table) {
  uint32_t remainder = uncompressedWords % kWarpSize;

  int uncompressedOffset = uncompressedWords - remainder;

  uint32_t compressedOffset = compressedWords;

  in += compressedOffset;

  if (remainder) {
    bool valid = laneId < remainder;

    uint32_t numCompressedRead;
    ANSDecodedT sym;

    decodeOnePartialWarp<ProbBits>(
        valid, state, compressedOffset, in, table, numCompressedRead, sym);

    if (valid) {
      writer.write(uncompressedOffset + laneId, sym);
    }

    in -= numCompressedRead;
  }

  while (uncompressedOffset > 0) {
    uncompressedOffset -= kWarpSize;

    uint32_t numCompressedRead;
    ANSDecodedT sym;

    decodeOneWarp<ProbBits>(
        state, compressedOffset, in, table, numCompressedRead, sym);

    writer.write(uncompressedOffset + laneId, sym);

    in -= numCompressedRead;
  }
}

template <
    int Threads,
    int ProbBits,
    int BlockSize>
__global__ __launch_bounds__(128) void ansDecodeKernel(
    void* in,
    uint32_t* table,
    void* out) {
  int tid = threadIdx.x;

  auto headerIn = (const ANSCoalescedHeader*)in;
  headerIn->checkMagicAndVersion();

  auto header = *headerIn;
  auto numBlocks = header.getNumBlocks();
  auto totalUncompressedWords = header.getTotalUncompressedWords();

  assert(ProbBits == header.getProbBits());

  constexpr int kBuckets = 1 << ProbBits;
  __shared__ uint32_t lookup[kBuckets];

  {
    uint4* lookup4 = (uint4*)lookup;
    const uint4* table4 = (const uint4*)table;

    static_assert(isEvenDivisor(kBuckets, Threads * 4), "");
    for (int j = 0;
         j < kBuckets / (Threads * (sizeof(uint4) / sizeof(uint32_t)));
         ++j) {
      lookup4[j * Threads + tid] = table4[j * Threads + tid];
    }
  }

  __syncthreads();

  auto writer = BatchWriter(out);

  int globalWarpId =
      __shfl_sync(0xffffffff, (blockIdx.x * blockDim.x + tid) / kWarpSize, 0);

  int warpsPerGrid = gridDim.x * Threads / kWarpSize;
  int laneId = getLaneId();

  for (int block = globalWarpId; block < numBlocks; block += warpsPerGrid) {
    ANSStateT state = headerIn->getWarpStates()[block].warpState[laneId];

    auto blockWords = headerIn->getBlockWords(numBlocks)[block];
    uint32_t uncompressedWords = (blockWords.x >> 16);
    uint32_t compressedWords = (blockWords.x & 0xffff);
    uint32_t blockCompressedWordStart = blockWords.y;

    auto blockDataIn =
        headerIn->getBlockDataStart(numBlocks) + blockCompressedWordStart;

    writer.setBlock(block);

    if (uncompressedWords == BlockSize) {
      blockDataIn += compressedWords;

      for (int i = BlockSize - kWarpSize + laneId; i >= 0; i -= kWarpSize) {
        ANSDecodedT sym;
        uint32_t numCompressedRead;

        decodeOneWarp<ProbBits>(
            state, compressedWords, blockDataIn, lookup, numCompressedRead, sym);

        blockDataIn -= numCompressedRead;

        writer.write(i, sym);
      }
    } else {
      ansDecodeWarpBlock<ProbBits>(
          laneId,
          state,
          uncompressedWords,
          compressedWords,
          blockDataIn,
          writer,
          lookup);
    }
  }
}

template <int Threads>
__global__ void ansDecodeTable(
    void* in,
    uint32_t probBits,
    uint32_t* __restrict__ table) {
  int tid = threadIdx.x;
  int warpId = tid / kWarpSize;
  int laneId = getLaneId();

  auto headerIn = (const ANSCoalescedHeader*)in;

  auto header = *headerIn;

  if (header.getTotalUncompressedWords() == 0) {
    return;
  }

  auto probs = headerIn->getSymbolProbs();

  uint32_t pdf = tid < kNumSymbols ? probs[tid] : 0;
  uint32_t cdf = 0;

  using BlockScan = cub::BlockScan<uint32_t, Threads>;
  __shared__ typename BlockScan::TempStorage tempStorage;

  uint32_t total = 0;
  BlockScan(tempStorage).ExclusiveSum(pdf, cdf, total);

  __shared__ uint2 smemPdfCdf[kNumSymbols];

  if (tid < kNumSymbols) {
    smemPdfCdf[tid] = uint2{pdf, cdf};
  }

  __syncthreads();

  constexpr int kWarpsPerBlock = Threads / kWarpSize;

  for (int i = warpId; i < kNumSymbols; i += kWarpsPerBlock) {
    auto v = smemPdfCdf[i];

    auto pdf = v.x;
    auto begin = v.y;
    auto end = begin + pdf;

    for (int j = begin + laneId; j < end; j += kWarpSize) {
      table[j] = packDecodeLookup(
          i,       // symbol
          pdf,     // bucket pdf
          j - begin); // within-bucket cdf
    }
  }
}

}} // namespace fz::ans

#endif // FZ_ANS_DIETGPU_ANS_GPUANSDECODE_H
