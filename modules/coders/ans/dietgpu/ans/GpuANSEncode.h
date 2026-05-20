/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Adapted for FZGPUModules: namespace renamed from multibyte_ans to fz::ans;
 * stripped ansEncode() host function (called directly from ans_stage.cu instead).
 */
#ifndef FZ_ANS_DIETGPU_ANS_GPUANSENCODE_H
#define FZ_ANS_DIETGPU_ANS_GPUANSENCODE_H

#pragma once

#include "BatchPrefixSum.h"
#include "GpuANSCodec.h"
#include "GpuANSStatistics.h"
#include <cmath>

namespace fz { namespace ans {

template <int ProbBits>
__device__ __forceinline__ uint32_t encodeOne(
    ANSStateT& state,
    ANSDecodedT sym,
    uint32_t outOffset,
    ANSEncodedT* __restrict__ outWords,
    const uint4* __restrict__ table) {
  auto lookup = table[sym];

  uint32_t pdf = lookup.x;
  uint32_t cdf = lookup.y;
  uint32_t div_m1 = lookup.z;
  uint32_t div_shift = lookup.w;

  constexpr ANSStateT kStateCheckMul = 1 << (kANSStateBits - ProbBits);

  ANSStateT maxStateCheck = pdf * kStateCheckMul;
  bool write = state >= maxStateCheck;

  auto vote = __ballot_sync(0xffffffff, write);
  auto prefix = __popc(vote & getLaneMaskLt());

  if (write) {
    outWords[outOffset + prefix] = state & kANSEncodedMask;
    state >>= kANSEncodedBits;
  }

  uint32_t t = __umulhi(state, div_m1);
  uint32_t div = (t + state) >> div_shift;
  auto mod = state - (div * pdf);

  constexpr uint32_t kProbBitsMul = 1 << ProbBits;
  state = div * kProbBitsMul + mod + cdf;

  return __popc(vote);
}

template <int ProbBits>
__device__ __forceinline__ uint32_t encodeOnePartial(
    bool valid,
    ANSStateT& state,
    ANSDecodedT sym,
    uint32_t outOffset,
    ANSEncodedT* __restrict__ outWords,
    const uint4* __restrict__ table) {
  if (!valid) return 0;
  auto lookup = table[sym];

  uint32_t pdf = lookup.x;
  uint32_t cdf = lookup.y;
  uint32_t div_m1 = lookup.z;
  uint32_t div_shift = lookup.w;

  constexpr ANSStateT kStateCheckMul = 1 << (kANSStateBits - ProbBits);

  ANSStateT maxStateCheck = pdf * kStateCheckMul;
  bool write = (state >= maxStateCheck);

  auto vote = __ballot_sync(0xffffffff, write);
  auto prefix = __popc(vote & getLaneMaskLt());

  if (write) {
    outWords[outOffset + prefix] = state & kANSEncodedMask;
    state >>= kANSEncodedBits;
  }

  uint32_t t = __umulhi(state, div_m1);
  uint32_t div = (t + state) >> div_shift;
  auto mod = state - (div * pdf);

  constexpr uint32_t kProbBitsMul = 1 << ProbBits;
  state = div * kProbBitsMul + mod + cdf;

  return __popc(vote);
}

template <int ProbBits, int BlockSize>
__global__ void ansEncodeBatch(
    uint8_t* in_dev,
    int inSize_dev,
    uint32_t maxNumCompressedBlocks,
    uint32_t uncoalescedBlockStride,
    uint8_t* compressedBlocks_dev,
    uint32_t* compressedWords_dev,
    const uint4* table_dev) {
  uint32_t numBlocks = (inSize_dev + BlockSize - 1) / BlockSize;
  int tid = threadIdx.x;
  int grim_warp_numid =
      __shfl_sync(0xffffffff, (blockIdx.x * blockDim.x + tid) / kWarpSize, 0);
  int laneId = getLaneId();

  __shared__ uint4 smemLookup[kNumSymbols];

  // we always have at least 256 threads
  if (tid < kNumSymbols) {
    smemLookup[tid] = table_dev[tid];
  }
  __syncthreads();

  uint32_t start = grim_warp_numid * BlockSize;
  if (start >= inSize_dev) {
    return;
  }

  uint32_t blockSize = min(start + BlockSize, inSize_dev) - start;

  if (grim_warp_numid >= numBlocks)
    return;

  auto inBlock = in_dev + start;
  auto outBlock = (ANSWarpState*)(compressedBlocks_dev
            + grim_warp_numid * uncoalescedBlockStride);

  assert(isPointerAligned(inBlock, kANSRequiredAlignment));

  ANSEncodedT* outWords = (ANSEncodedT*)(outBlock + 1);

  ANSStateT state = kANSStartState;

  uint32_t inOffset = laneId;
  uint32_t outOffset = 0;

  constexpr int kUnroll = 8;

  uint32_t limit = roundDown(blockSize, kWarpSize * kUnroll);

  {
    for (; inOffset < limit; inOffset += kWarpSize * kUnroll) {
#pragma unroll
      for (int j = 0; j < kUnroll; ++j) {
        outOffset +=
            encodeOne<ProbBits>(state, inBlock[inOffset + j * kWarpSize], outOffset, outWords, smemLookup);
      }
    }
  }

  if (limit != blockSize) {
    limit = roundDown(blockSize, kWarpSize);

    for (; inOffset < limit; inOffset += kWarpSize) {
      outOffset +=
          encodeOne<ProbBits>(state, inBlock[inOffset], outOffset, outWords, smemLookup);
    }
    if (limit != blockSize) {
      bool valid = inOffset < blockSize;
      ANSDecodedT sym = valid ? inBlock[inOffset] : ANSDecodedT(0);
      outOffset += encodeOnePartial<ProbBits>(
          valid, state, sym, outOffset, outWords, smemLookup);
    }
  }
  // Write final state at the beginning (aligned addresses)
  outBlock->warpState[laneId] = state;

  if (laneId == 0) {
    compressedWords_dev[grim_warp_numid] = outOffset;
  }
}

template <typename A, int B>
struct Align {
  typedef uint32_t argument_type;
  typedef uint32_t result_type;

  template <typename T>
  __host__ __device__ uint32_t operator()(T x) const {
    constexpr int kDiv = B / sizeof(A);
    constexpr int kSize = kDiv < 1 ? 1 : kDiv;

    return roundUp(x, T(kSize));
  }
};

template <int Threads>
__global__ void ansEncodeCoalesceBatch(
    const uint8_t* __restrict__ compressedBlocks_dev,
    int uncompressedWords,
    uint32_t maxNumCompressedBlocks,
    uint32_t uncoalescedBlockStride,
    const uint32_t* __restrict__ compressedWords_dev,
    const uint32_t* __restrict__ compressedWordsPrefix_dev,
    const uint4* __restrict__ table_dev,
    uint32_t config_probBits,
    uint8_t* out_dev,
    uint32_t* outSize_dev) {

  auto numBlocks = divUp(uncompressedWords, kDefaultBlockSize);

  int block = blockIdx.x;
  int tid = threadIdx.x;

  ANSCoalescedHeader* headerOut = (ANSCoalescedHeader*)out_dev;

  // The first block will be responsible for the coalesced header
  if (block == 0) {
    if (tid == 0) {
      uint32_t totalCompressedWords = 0;

      if (numBlocks > 0) {
        totalCompressedWords =
            compressedWordsPrefix_dev[numBlocks - 1] +
            roundUp(
                compressedWords_dev[numBlocks - 1],
                kBlockAlignment / sizeof(ANSEncodedT));
      }

      ANSCoalescedHeader header;
      header.setMagicAndVersion();
      header.setNumBlocks(numBlocks);
      header.setTotalUncompressedWords(uncompressedWords);
      header.setTotalCompressedWords(totalCompressedWords);
      header.setProbBits(config_probBits);

      if (outSize_dev) {
        *outSize_dev = header.getTotalCompressedSize();
      }

      *headerOut = header;
    }

    auto probsOut = headerOut->getSymbolProbs();

    #pragma unroll
    for (int i = tid; i < kNumSymbols; i += Threads) {
      probsOut[i] = table_dev[i].x;
    }
  }

  if (block >= numBlocks) {
    return;
  }

  // where our per-warp data lies
  auto uncoalescedBlock = compressedBlocks_dev +
                      block * uncoalescedBlockStride;

  // Write per-block warp state
  if (tid < kWarpSize) {
    auto warpStateIn = (ANSWarpState*)uncoalescedBlock;

    headerOut->getWarpStates()[block].warpState[tid] =
        warpStateIn->warpState[tid];
  }

  auto blockWordsOut = headerOut->getBlockWords(numBlocks);

  // Write out per-block word length
  for (int i = blockIdx.x * Threads + tid; i < numBlocks;
       i += gridDim.x * Threads) {
    uint32_t lastBlockWords = uncompressedWords % kDefaultBlockSize;
    lastBlockWords = lastBlockWords == 0 ? kDefaultBlockSize : lastBlockWords;

    uint32_t blockWords =
        (i == numBlocks - 1) ? lastBlockWords : kDefaultBlockSize;

    blockWordsOut[i] = uint2{
        (blockWords << 16) | compressedWords_dev[i], compressedWordsPrefix_dev[i]};
  }

  // Number of compressed words in this block
  uint32_t numWords = compressedWords_dev[block];

  using LoadT = uint4;

  uint32_t limitEnd = divUp(numWords, kBlockAlignment / sizeof(ANSEncodedT));

  auto inT = (const LoadT*)(uncoalescedBlock + sizeof(ANSWarpState));
  auto outT =
      (LoadT*)(headerOut->getBlockDataStart(numBlocks) + compressedWordsPrefix_dev[block]);

  for (uint32_t i = tid; i < limitEnd; i += Threads) {
    outT[i] = inT[i];
  }
}

}} // namespace fz::ans

#endif // FZ_ANS_DIETGPU_ANS_GPUANSENCODE_H
