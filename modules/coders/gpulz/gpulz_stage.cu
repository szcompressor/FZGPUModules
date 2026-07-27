// GPU kernels are a direct port of compressKernelI / decompressKernel from
// GPULZ (Zhang, Tian, Di, Yu, Swany, Tao, Cappello, ICS '23).
// Upstream: https://github.com/hpdps-group/ICS23-GPULZ — see THIRD_PARTY.md.
// The per-chunk container format, raw-fallback flag, CUB exclusive scan for
// packing offsets, and deferred tail-size readback are FZGM's own, mirroring
// RREStage/RZEStage.

#include "coders/gpulz/gpulz_stage.h"
#include "backend/algorithms.h"
#include "mem/mempool.h"
#include "cuda_check.h"

#include "backend/api.h"
#include "backend/cub.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace fz {

static constexpr int GPULZ_THREAD_SIZE = 128; // threads per block == symbols processed per iteration
static constexpr int GPULZ_WINDOW_SIZE = 32;  // sliding-window / max match length (upstream default)

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Encode kernel — one CUDA block per chunk. Direct port of GPULZ's
// compressKernelI: builds a per-element (length, offset) match table in
// shared memory, derives a literal/match flag bit per encode item, prefix-
// sums the per-item byte sizes (Blelloch scan), then writes the flag bits
// and packed literal/match bytes to per-chunk uniform scratch buffers.
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
template <typename T, int CS>
static __global__ void __launch_bounds__(GPULZ_THREAD_SIZE)
gpulzEncodeKernel(
    const T*    __restrict__ d_in,
    uint8_t*    __restrict__ d_flag_scratch,
    uint8_t*    __restrict__ d_data_scratch,
    uint32_t*   __restrict__ d_flag_size,
    uint32_t*   __restrict__ d_data_size,
    int minEncodeLength)
{
    constexpr int blockSize  = CS / sizeof(T);  // chunk size in elements
    constexpr int threadSize = GPULZ_THREAD_SIZE;
    constexpr int flagBytes  = (blockSize + 7) / 8;

    __shared__ T        buffer[blockSize];
    __shared__ uint8_t  lengthBuffer[blockSize];
    __shared__ uint8_t  offsetBuffer[blockSize];
    __shared__ uint32_t prefixBuffer[blockSize + 1];
    __shared__ uint8_t  byteFlagArr[flagBytes];

    const T* chunkIn = d_in + (size_t)blockIdx.x * blockSize;

    for (int i = 0; i < blockSize / threadSize; i++)
        buffer[threadIdx.x + threadSize * i] = chunkIn[threadIdx.x + threadSize * i];
    __syncthreads();

    // find longest match for every element
    for (int iteration = 0; iteration < blockSize / threadSize; iteration++) {
        int tid          = threadIdx.x + iteration * threadSize;
        int bufferStart  = tid;
        int bufferPointer = bufferStart;
        int windowStart  = bufferStart - GPULZ_WINDOW_SIZE < 0 ? 0 : bufferStart - GPULZ_WINDOW_SIZE;
        int windowPointer = windowStart;

        uint8_t maxLen = 0, maxOffset = 0, len = 0, offset = 0;

        while (windowPointer < bufferStart && bufferPointer < blockSize) {
            if (buffer[bufferPointer] == buffer[windowPointer]) {
                if (offset == 0) offset = bufferPointer - windowPointer;
                len++;
                bufferPointer++;
            } else {
                if (len > maxLen) { maxLen = len; maxOffset = offset; }
                len = 0; offset = 0;
                bufferPointer = bufferStart;
            }
            windowPointer++;
        }
        if (len > maxLen) { maxLen = len; maxOffset = offset; }

        lengthBuffer[tid] = maxLen;
        offsetBuffer[tid] = maxOffset;
        prefixBuffer[tid] = 0;
    }
    __syncthreads();

    // build the literal/match flag bitmap and per-item byte-size prefix seeds
    // (inherently sequential — matches upstream, only thread 0 performs it)
    uint32_t flagCount = 0;
    if (threadIdx.x == 0) {
        uint8_t flagPosition = 0x01;
        uint8_t byteFlag     = 0;
        int     encodeIndex  = 0;

        while (encodeIndex < blockSize) {
            if (lengthBuffer[encodeIndex] < minEncodeLength) {
                prefixBuffer[encodeIndex] = sizeof(T);
                encodeIndex++;
            } else {
                prefixBuffer[encodeIndex] = 2;
                encodeIndex += lengthBuffer[encodeIndex];
                byteFlag |= flagPosition;
            }
            if (flagPosition == 0x80) {
                byteFlagArr[flagCount] = byteFlag;
                flagCount++;
                flagPosition = 0x01;
                byteFlag     = 0;
                continue;
            }
            flagPosition <<= 1;
        }
        if (flagPosition != 0x01) {
            byteFlagArr[flagCount] = byteFlag;
            flagCount++;
        }
    }
    __syncthreads();

    // Blelloch prefix sum (up-sweep then down-sweep) over per-item byte sizes
    int prefixSumOffset = 1;
    for (uint32_t d = (uint32_t)blockSize >> 1; d > 0; d >>= 1) {
        for (int iteration = 0; iteration < blockSize / threadSize; iteration++) {
            int tid = threadIdx.x + iteration * threadSize;
            if (tid < (int)d) {
                int ai = prefixSumOffset * (2 * tid + 1) - 1;
                int bi = prefixSumOffset * (2 * tid + 2) - 1;
                prefixBuffer[bi] += prefixBuffer[ai];
            }
            __syncthreads();
        }
        prefixSumOffset *= 2;
    }

    if (threadIdx.x == 0) {
        d_data_size[blockIdx.x] = prefixBuffer[blockSize - 1];
        d_flag_size[blockIdx.x] = flagCount;
        prefixBuffer[blockSize] = prefixBuffer[blockSize - 1];
        prefixBuffer[blockSize - 1] = 0;
    }
    __syncthreads();

    for (int d = 1; d < blockSize; d *= 2) {
        prefixSumOffset >>= 1;
        for (int iteration = 0; iteration < blockSize / threadSize; iteration++) {
            int tid = threadIdx.x + iteration * threadSize;
            if (tid < d) {
                int ai = prefixSumOffset * (2 * tid + 1) - 1;
                int bi = prefixSumOffset * (2 * tid + 2) - 1;
                uint32_t t = prefixBuffer[ai];
                prefixBuffer[ai] = prefixBuffer[bi];
                prefixBuffer[bi] += t;
            }
            __syncthreads();
        }
    }

    // write literal/match bytes into this chunk's data scratch
    uint8_t* chunkData = d_data_scratch + (size_t)blockIdx.x * CS;
    for (int iteration = 0; iteration < blockSize / threadSize; iteration++) {
        int tid = threadIdx.x + iteration * threadSize;
        if (prefixBuffer[tid + 1] != prefixBuffer[tid]) {
            uint32_t off = prefixBuffer[tid];
            if (lengthBuffer[tid] < minEncodeLength) {
                const uint8_t* bytePtr = (const uint8_t*)&buffer[tid];
                for (unsigned tmpIndex = 0; tmpIndex < sizeof(T); tmpIndex++)
                    chunkData[off + tmpIndex] = bytePtr[tmpIndex];
            } else {
                chunkData[off]     = lengthBuffer[tid];
                chunkData[off + 1] = offsetBuffer[tid];
            }
        }
    }

    // copy the flag bitmap out to this chunk's flag scratch
    if (threadIdx.x == 0) {
        uint8_t* chunkFlag = d_flag_scratch + (size_t)blockIdx.x * flagBytes;
        for (uint32_t i = 0; i < flagCount; i++)
            chunkFlag[i] = byteFlagArr[i];
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Decode kernel — one thread per chunk. Direct port of GPULZ's
// decompressKernel: walks the flag bitmap, copying literals verbatim or
// expanding (length, offset) back-references within the already-decoded
// prefix of the chunk.
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
template <typename T, int CS>
static __global__ void gpulzDecodeKernel(
    T*              __restrict__ d_out,
    const uint8_t*  __restrict__ d_in,          // packed payload base
    const uint32_t* __restrict__ d_in_offsets,  // per-chunk offset into payload
    const uint32_t* __restrict__ d_flag_size,
    const uint32_t* __restrict__ d_data_size,
    uint32_t num_chunks)
{
    constexpr int blockSize = CS / sizeof(T);

    uint32_t cid = blockIdx.x * blockDim.x + threadIdx.x;
    if (cid >= num_chunks) return;

    const uint8_t* chunkIn = d_in + d_in_offsets[cid];
    T* out = d_out + (size_t)cid * blockSize;

    const uint32_t flag_sz = d_flag_size[cid];
    if (flag_sz == 0) return; // raw chunk — already copied through by the host

    const uint8_t* flagArr  = chunkIn;
    const uint8_t* compData = chunkIn + flag_sz;

    uint32_t dataPointsIndex     = 0;
    uint32_t compressedDataIndex = 0;

    for (uint32_t flagArrayIndex = 0; flagArrayIndex < flag_sz; flagArrayIndex++) {
        uint8_t byteFlag = flagArr[flagArrayIndex];

        for (int bitCount = 0; bitCount < 8; bitCount++) {
            int matchFlag = (byteFlag >> bitCount) & 0x1;
            if (matchFlag == 1) {
                int length = compData[compressedDataIndex];
                int offset = compData[compressedDataIndex + 1];
                compressedDataIndex += 2;
                uint32_t dataPointsStart = dataPointsIndex;
                for (int i = 0; i < length; i++) {
                    out[dataPointsIndex] = out[dataPointsStart - offset + i];
                    dataPointsIndex++;
                }
            } else {
                uint8_t* tmpPtr = (uint8_t*)&out[dataPointsIndex];
                for (unsigned i = 0; i < sizeof(T); i++)
                    tmpPtr[i] = compData[compressedDataIndex + i];
                compressedDataIndex += sizeof(T);
                dataPointsIndex++;
            }
            if (dataPointsIndex >= (uint32_t)blockSize) return;
        }
    }
}

// ━━━━ helper kernels (word-size independent) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

// Decide, per chunk, whether the LZSS-encoded form (flag_size + data_size)
// is smaller than the raw chunk; if not, fall back to raw storage. Writes
// `d_clean` (the scan input: final packed size per chunk) and rewrites
// `d_flag_size`/`d_data_size` in place so downstream pack/decode agree.
static __global__ void gpulzFinalizeSizesKernel(
    uint32_t* __restrict__ d_flag_size,
    uint32_t* __restrict__ d_data_size,
    uint32_t* __restrict__ d_clean,
    uint32_t n_chunks,
    uint32_t chunk_bytes)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_chunks) return;

    uint32_t total = d_flag_size[i] + d_data_size[i];
    if (total >= chunk_bytes) {
        // Raw fallback: flag_size's high bit marks "stored raw"; data_size
        // becomes the raw chunk byte count.
        d_flag_size[i] = 0x80000000u;
        d_data_size[i] = chunk_bytes;
        d_clean[i]     = chunk_bytes;
    } else {
        d_clean[i] = total;
    }
}

static __global__ void gpulzAddOffsetKernel(
    uint32_t* __restrict__ arr, uint32_t n, uint32_t offset)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) arr[i] += offset;
}

// Interleaves the per-chunk (flag_size, data_size) arrays into the header's
// packed entry layout (avoids cudaMemcpy2DAsync, which has no HIP facade).
static __global__ void gpulzInterleaveHeaderKernel(
    const uint32_t* __restrict__ d_flag_size,
    const uint32_t* __restrict__ d_data_size,
    uint32_t*       __restrict__ d_hdr_entries,
    uint32_t n_chunks)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_chunks) return;
    d_hdr_entries[2 * i]     = d_flag_size[i];
    d_hdr_entries[2 * i + 1] = d_data_size[i];
}

// Packs each chunk's [flag bytes][data bytes] (or raw chunk bytes, if
// flagged) from uniform scratch into the final compact output at its
// scanned destination offset.
static __global__ void gpulzPackKernel(
    const uint8_t*  __restrict__ d_flag_scratch,
    const uint8_t*  __restrict__ d_data_scratch,
    const uint8_t*  __restrict__ d_in_raw,      // original input, for raw fallback
    uint8_t*        __restrict__ d_out,
    const uint32_t* __restrict__ d_dst_offsets,
    const uint32_t* __restrict__ d_flag_size,
    const uint32_t* __restrict__ d_data_size,
    uint32_t flag_scratch_stride,
    uint32_t chunk_bytes)
{
    uint32_t cid     = blockIdx.x;
    uint32_t dst_off = d_dst_offsets[cid];
    uint32_t fsize   = d_flag_size[cid];
    bool     raw     = (fsize & 0x80000000u) != 0;
    uint32_t dsize   = d_data_size[cid];

    uint8_t* dst = d_out + dst_off;
    if (raw) {
        const uint8_t* src = d_in_raw + (size_t)cid * chunk_bytes;
        for (uint32_t i = threadIdx.x; i < chunk_bytes; i += blockDim.x)
            dst[i] = src[i];
    } else {
        const uint8_t* fsrc = d_flag_scratch + (size_t)cid * flag_scratch_stride;
        const uint8_t* dsrc = d_data_scratch + (size_t)cid * chunk_bytes;
        for (uint32_t i = threadIdx.x; i < fsize; i += blockDim.x)
            dst[i] = fsrc[i];
        for (uint32_t i = threadIdx.x; i < dsize; i += blockDim.x)
            dst[fsize + i] = dsrc[i];
    }
}

// ━━━━ word-size × chunk-size dispatch helpers ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// chunk_size selects the compile-time CS template argument; word_size
// selects T. Supported chunk sizes: 1024, 2048, 4096 (see GPULZStage::execute()).
static void launchEncode(uint8_t word_size, uint32_t chunk_size, int n_chunks, cudaStream_t stream,
                          const uint8_t* d_in, uint8_t* d_flag_scratch, uint8_t* d_data_scratch,
                          uint32_t* d_flag_size, uint32_t* d_data_size)
{
    const int minEncodeLength = (word_size == 1) ? 2 : 1;
#define FZ_GPULZ_ENCODE_CASE(CS_VAL)                                                                                     \
    case CS_VAL:                                                                                                          \
        switch (word_size) {                                                                                              \
            case 1: gpulzEncodeKernel<uint8_t,  CS_VAL><<<n_chunks, GPULZ_THREAD_SIZE, 0, stream>>>((const uint8_t*)d_in,  d_flag_scratch, d_data_scratch, d_flag_size, d_data_size, minEncodeLength); break; \
            case 2: gpulzEncodeKernel<uint16_t, CS_VAL><<<n_chunks, GPULZ_THREAD_SIZE, 0, stream>>>((const uint16_t*)d_in, d_flag_scratch, d_data_scratch, d_flag_size, d_data_size, minEncodeLength); break; \
            case 4: gpulzEncodeKernel<uint32_t, CS_VAL><<<n_chunks, GPULZ_THREAD_SIZE, 0, stream>>>((const uint32_t*)d_in, d_flag_scratch, d_data_scratch, d_flag_size, d_data_size, minEncodeLength); break; \
            case 8: gpulzEncodeKernel<uint64_t, CS_VAL><<<n_chunks, GPULZ_THREAD_SIZE, 0, stream>>>((const uint64_t*)d_in, d_flag_scratch, d_data_scratch, d_flag_size, d_data_size, minEncodeLength); break; \
            default: throw std::runtime_error("GPULZStage: word_size must be 1, 2, 4, or 8");                             \
        }                                                                                                                  \
        break;
    switch (chunk_size) {
        FZ_GPULZ_ENCODE_CASE(1024)
        FZ_GPULZ_ENCODE_CASE(2048)
        FZ_GPULZ_ENCODE_CASE(4096)
        default: throw std::runtime_error("GPULZStage: chunk_size must be 1024, 2048, or 4096");
    }
#undef FZ_GPULZ_ENCODE_CASE
}

static void launchDecode(uint8_t word_size, uint32_t chunk_size, int n_chunks, cudaStream_t stream,
                          uint8_t* d_out, const uint8_t* d_in, const uint32_t* d_in_off,
                          const uint32_t* d_flag_size, const uint32_t* d_data_size)
{
    constexpr int kDecodeTPB = 32;
    const int grid = (n_chunks + kDecodeTPB - 1) / kDecodeTPB;
#define FZ_GPULZ_DECODE_CASE(CS_VAL)                                                                                      \
    case CS_VAL:                                                                                                          \
        switch (word_size) {                                                                                              \
            case 1: gpulzDecodeKernel<uint8_t,  CS_VAL><<<grid, kDecodeTPB, 0, stream>>>((uint8_t*)d_out,  d_in, d_in_off, d_flag_size, d_data_size, (uint32_t)n_chunks); break; \
            case 2: gpulzDecodeKernel<uint16_t, CS_VAL><<<grid, kDecodeTPB, 0, stream>>>((uint16_t*)d_out, d_in, d_in_off, d_flag_size, d_data_size, (uint32_t)n_chunks); break; \
            case 4: gpulzDecodeKernel<uint32_t, CS_VAL><<<grid, kDecodeTPB, 0, stream>>>((uint32_t*)d_out, d_in, d_in_off, d_flag_size, d_data_size, (uint32_t)n_chunks); break; \
            case 8: gpulzDecodeKernel<uint64_t, CS_VAL><<<grid, kDecodeTPB, 0, stream>>>((uint64_t*)d_out, d_in, d_in_off, d_flag_size, d_data_size, (uint32_t)n_chunks); break; \
            default: throw std::runtime_error("GPULZStage: word_size must be 1, 2, 4, or 8");                             \
        }                                                                                                                  \
        break;
    switch (chunk_size) {
        FZ_GPULZ_DECODE_CASE(1024)
        FZ_GPULZ_DECODE_CASE(2048)
        FZ_GPULZ_DECODE_CASE(4096)
        default: throw std::runtime_error("GPULZStage: chunk_size must be 1024, 2048, or 4096");
    }
#undef FZ_GPULZ_DECODE_CASE
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GPULZStage::~GPULZStage() {
    auto fwd_free = [&](void* p) {
        if (!p) return;
        if (scratch_from_pool_ && scratch_pool_owner_) scratch_pool_owner_->free(p, 0);
        else cudaFree(p);
    };
    fwd_free(d_data_scratch_);
    fwd_free(d_flag_scratch_);
    fwd_free(d_flag_size_);
    fwd_free(d_data_size_);
    fwd_free(d_clean_dev_);
    fwd_free(d_dst_off_dev_);
}

void GPULZStage::postStreamSync(cudaStream_t stream) {
    if (!tail_readback_pending_) return;

    uint32_t tail_off = 0, tail_sz = 0;
    FZ_CUDA_CHECK(cudaMemcpyAsync(&tail_off, d_dst_off_dev_ + tail_last_index_,
                                  sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
    FZ_CUDA_CHECK(cudaMemcpyAsync(&tail_sz, d_clean_dev_ + tail_last_index_,
                                  sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
    FZ_CUDA_CHECK(cudaStreamSynchronize(stream));
    const size_t total_out = (size_t)tail_off + (size_t)tail_sz;
    actual_output_size_ = (total_out + 3) & ~size_t(3);
    if (tail_output_ptr_ && actual_output_size_ > total_out) {
        FZ_CUDA_CHECK(cudaMemsetAsync(tail_output_ptr_ + total_out, 0,
                                      actual_output_size_ - total_out, stream));
        FZ_CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    tail_output_ptr_       = nullptr;
    tail_readback_pending_ = false;
    tail_readback_stream_  = nullptr;

    const float ratio = cached_orig_bytes_ > 0
        ? (float)cached_orig_bytes_ / (float)actual_output_size_ : 0.0f;
    FZ_LOG(DEBUG, "GPULZ encode done: %.1f KB -> %.1f KB  ratio %.2fx",
           cached_orig_bytes_ / 1024.0f, actual_output_size_ / 1024.0f, ratio);
}

std::unordered_map<std::string, size_t>
GPULZStage::getActualOutputSizesByName() const {
    if (tail_readback_pending_) {
        uint32_t tail_off = 0, tail_sz = 0;
        FZ_CUDA_CHECK(cudaMemcpyAsync(&tail_off, d_dst_off_dev_ + tail_last_index_,
                                      sizeof(uint32_t), cudaMemcpyDeviceToHost, tail_readback_stream_));
        FZ_CUDA_CHECK(cudaMemcpyAsync(&tail_sz, d_clean_dev_ + tail_last_index_,
                                      sizeof(uint32_t), cudaMemcpyDeviceToHost, tail_readback_stream_));
        FZ_CUDA_CHECK(cudaStreamSynchronize(tail_readback_stream_));
        const size_t total_out = (size_t)tail_off + (size_t)tail_sz;
        const_cast<GPULZStage*>(this)->actual_output_size_ = (total_out + 3) & ~size_t(3);
        if (tail_output_ptr_ && actual_output_size_ > total_out) {
            FZ_CUDA_CHECK(cudaMemsetAsync(tail_output_ptr_ + total_out, 0,
                                          actual_output_size_ - total_out, tail_readback_stream_));
            FZ_CUDA_CHECK(cudaStreamSynchronize(tail_readback_stream_));
        }
        const_cast<GPULZStage*>(this)->tail_output_ptr_  = nullptr;
        tail_readback_pending_ = false;
        tail_readback_stream_  = nullptr;
    }
    return {{"output", actual_output_size_}};
}

size_t GPULZStage::getActualOutputSize(int index) const {
    if (index != 0) return 0;
    getActualOutputSizesByName();
    return actual_output_size_;
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
void GPULZStage::execute(
    cudaStream_t stream,
    MemoryPool* pool,
    const std::vector<void*>& inputs,
    const std::vector<void*>& outputs,
    const std::vector<size_t>& sizes)
{
    if (inputs.empty() || outputs.empty() || sizes.empty())
        throw std::runtime_error("GPULZStage: invalid inputs/outputs");

    tail_readback_pending_ = false;
    tail_readback_stream_  = nullptr;
    tail_last_index_       = 0;

    const size_t in_bytes = sizes[0];
    if (in_bytes == 0) { actual_output_size_ = 0; return; }

    if (chunk_size_ != 1024 && chunk_size_ != 2048 && chunk_size_ != 4096)
        throw std::runtime_error(
            "GPULZStage: chunk_size must be 1024, 2048, or 4096; got "
            + std::to_string(chunk_size_));
    if (word_size_ != 1 && word_size_ != 2 && word_size_ != 4 && word_size_ != 8)
        throw std::runtime_error(
            "GPULZStage: word_size must be 1, 2, 4, or 8; got "
            + std::to_string((int)word_size_));
    // ── Forward (compress) ─────────────────────────────────────────────
    if (!is_inverse_) {
        if (in_bytes % chunk_size_ != 0)
            throw std::runtime_error("GPULZStage: input size must be a multiple of chunk_size (see getRequiredInputAlignment())");

        const size_t   n_chunks     = in_bytes / chunk_size_;
        const uint32_t n_chunks_u   = (uint32_t)n_chunks;
        const uint32_t in_bytes_u   = (uint32_t)in_bytes;
        const uint32_t grid256      = (n_chunks_u + 255u) / 256u;
        const uint32_t block_elems  = chunk_size_ / word_size_;
        const uint32_t flag_stride  = (block_elems + 7) / 8;

        cached_orig_bytes_ = in_bytes_u;

        if (n_chunks > scratch_capacity_) {
            auto fwd_free = [&](void* p) {
                if (!p) return;
                if (scratch_from_pool_ && scratch_pool_owner_)
                    scratch_pool_owner_->free(p, stream);
                else { FZ_CUDA_CHECK_WARN(cudaStreamSynchronize(stream)); cudaFree(p); }
            };
            fwd_free(d_data_scratch_); d_data_scratch_ = nullptr;
            fwd_free(d_flag_scratch_); d_flag_scratch_ = nullptr;
            fwd_free(d_flag_size_);    d_flag_size_    = nullptr;
            fwd_free(d_data_size_);    d_data_size_    = nullptr;
            fwd_free(d_clean_dev_);    d_clean_dev_    = nullptr;
            fwd_free(d_dst_off_dev_);  d_dst_off_dev_  = nullptr;

            if (pool) {
                d_data_scratch_ = (uint8_t*) pool->allocate(n_chunks * (size_t)chunk_size_, stream, "gpulz_data_scratch", true);
                d_flag_scratch_ = (uint8_t*) pool->allocate(n_chunks * (size_t)flag_stride,  stream, "gpulz_flag_scratch", true);
                d_flag_size_    = (uint32_t*)pool->allocate(n_chunks * sizeof(uint32_t),     stream, "gpulz_flag_size",    true);
                d_data_size_    = (uint32_t*)pool->allocate(n_chunks * sizeof(uint32_t),     stream, "gpulz_data_size",    true);
                d_clean_dev_    = (uint32_t*)pool->allocate(n_chunks * sizeof(uint32_t),     stream, "gpulz_clean",        true);
                d_dst_off_dev_  = (uint32_t*)pool->allocate(n_chunks * sizeof(uint32_t),     stream, "gpulz_offsets",      true);
                if (!d_data_scratch_ || !d_flag_scratch_ || !d_flag_size_ || !d_data_size_ || !d_clean_dev_ || !d_dst_off_dev_)
                    throw std::runtime_error("GPULZStage: failed to allocate persistent forward scratch from MemoryPool");
                scratch_pool_owner_ = pool;
                scratch_from_pool_  = true;
            } else {
                FZ_CUDA_CHECK(cudaMalloc(&d_data_scratch_, n_chunks * (size_t)chunk_size_));
                FZ_CUDA_CHECK(cudaMalloc(&d_flag_scratch_, n_chunks * (size_t)flag_stride));
                FZ_CUDA_CHECK(cudaMalloc(&d_flag_size_,    n_chunks * sizeof(uint32_t)));
                FZ_CUDA_CHECK(cudaMalloc(&d_data_size_,    n_chunks * sizeof(uint32_t)));
                FZ_CUDA_CHECK(cudaMalloc(&d_clean_dev_,    n_chunks * sizeof(uint32_t)));
                FZ_CUDA_CHECK(cudaMalloc(&d_dst_off_dev_,  n_chunks * sizeof(uint32_t)));
                scratch_pool_owner_ = nullptr;
                scratch_from_pool_  = false;
            }
            scratch_capacity_ = n_chunks;
        }

        FZ_LOG(TRACE, "GPULZ encode: %.1f KB in, %u chunks, word_size %d",
               in_bytes / 1024.0, n_chunks_u, (int)word_size_);

        const size_t header_size = 4 + 4 + 8 * n_chunks;

        // (1) Encode each chunk into uniform scratch, recording flag/data sizes.
        launchEncode(word_size_, chunk_size_, (int)n_chunks, stream,
                     (const uint8_t*)inputs[0], d_flag_scratch_, d_data_scratch_,
                     d_flag_size_, d_data_size_);
        FZ_CUDA_CHECK(cudaGetLastError());

        // (2) Raw-fallback decision + scan-input sizes.
        gpulzFinalizeSizesKernel<<<grid256, 256, 0, stream>>>(
            d_flag_size_, d_data_size_, d_clean_dev_, n_chunks_u, chunk_size_);
        FZ_CUDA_CHECK(cudaGetLastError());

        // (3) Write the stream header (orig_total, num_chunks, per-chunk entries).
        uint8_t* d_out = (uint8_t*)outputs[0];
        const uint32_t h_hdr[2] = {in_bytes_u, n_chunks_u};
        FZ_CUDA_CHECK(cudaMemcpyAsync(d_out, h_hdr, 8, cudaMemcpyHostToDevice, stream));
        // Interleave (flag_size, data_size) per chunk directly from the two
        // device arrays into the header's packed entry layout.
        gpulzInterleaveHeaderKernel<<<grid256, 256, 0, stream>>>(
            d_flag_size_, d_data_size_, (uint32_t*)(d_out + 8), n_chunks_u);
        FZ_CUDA_CHECK(cudaGetLastError());

        // (4) Exclusive prefix sum of clean sizes -> payload-relative offsets.
        {
            auto scan_tmp = fz::backend::withTempStorage(pool, stream, "gpulz_cub_scan_tmp",
                [&](void* tmp, size_t& bytes) {
                    cub::DeviceScan::ExclusiveSum(tmp, bytes,
                                                  d_clean_dev_, d_dst_off_dev_,
                                                  (int)n_chunks, stream);
                });
            fz::backend::freeTempStorage(pool, scan_tmp, stream);
        }

        // (5) Convert to absolute output offsets (add header size).
        gpulzAddOffsetKernel<<<grid256, 256, 0, stream>>>(d_dst_off_dev_, n_chunks_u, (uint32_t)header_size);

        // (6) Defer final output-size readback to postStreamSync.
        tail_last_index_       = n_chunks_u - 1;
        tail_output_ptr_       = (uint8_t*)outputs[0];
        tail_readback_pending_ = true;
        tail_readback_stream_  = stream;

        // (7) Pack compressed chunks from uniform scratch (or raw input, on
        // fallback) into the final packed output.
        gpulzPackKernel<<<(int)n_chunks, 256, 0, stream>>>(
            d_flag_scratch_, d_data_scratch_, (const uint8_t*)inputs[0], d_out,
            d_dst_off_dev_, d_flag_size_, d_data_size_, flag_stride, chunk_size_);
        FZ_CUDA_CHECK(cudaGetLastError());

    // ── Inverse (decompress) ───────────────────────────────────────────
    } else {
        const uint8_t* d_in  = (const uint8_t*)inputs[0];
        uint8_t*       d_out = (uint8_t*)outputs[0];

        uint8_t h_hdr_raw[8];
        FZ_CUDA_CHECK(cudaMemcpyAsync(h_hdr_raw, d_in, 8, cudaMemcpyDeviceToHost, stream));
        FZ_CUDA_CHECK(cudaStreamSynchronize(stream));
        uint32_t orig_total, num_chunks;
        std::memcpy(&orig_total, h_hdr_raw + 0, sizeof(uint32_t));
        std::memcpy(&num_chunks, h_hdr_raw + 4, sizeof(uint32_t));
        cached_orig_bytes_ = orig_total;

        if (num_chunks == 0 || orig_total == 0) { actual_output_size_ = 0; return; }

        std::vector<uint32_t> h_entries(2 * num_chunks);
        FZ_CUDA_CHECK(cudaMemcpyAsync(h_entries.data(), d_in + 8,
                                      2 * num_chunks * sizeof(uint32_t),
                                      cudaMemcpyDeviceToHost, stream));
        FZ_CUDA_CHECK(cudaStreamSynchronize(stream));

        const size_t header_bytes = 4 + 4 + 8 * num_chunks;

        std::vector<uint32_t> h_in_off(num_chunks), h_flag_sz(num_chunks), h_data_sz(num_chunks);
        std::vector<bool>     h_is_raw(num_chunks);

        uint32_t in_cursor = (uint32_t)header_bytes;
        for (uint32_t i = 0; i < num_chunks; i++) {
            const uint32_t flag_entry = h_entries[2 * i];
            const uint32_t data_sz    = h_entries[2 * i + 1];
            const bool     raw        = (flag_entry & 0x80000000u) != 0;
            const uint32_t flag_sz    = raw ? 0u : flag_entry;

            h_in_off[i]  = in_cursor;
            h_flag_sz[i] = flag_entry; // keep raw-flag bit for the decode kernel
            h_data_sz[i] = data_sz;
            h_is_raw[i]  = raw;

            in_cursor += raw ? data_sz : (flag_sz + data_sz);
        }

        auto alloc_u32 = [&](const char* tag) -> uint32_t* {
            if (pool) return (uint32_t*)pool->allocate(num_chunks * sizeof(uint32_t), stream, tag);
            uint32_t* p = nullptr; FZ_CUDA_CHECK(cudaMalloc(&p, num_chunks * sizeof(uint32_t))); return p;
        };
        uint32_t* d_in_off_  = alloc_u32("gpulz_inv_in_off");
        uint32_t* d_flag_sz_ = alloc_u32("gpulz_inv_flag_sz");
        uint32_t* d_data_sz_ = alloc_u32("gpulz_inv_data_sz");

        FZ_CUDA_CHECK(cudaMemcpyAsync(d_in_off_,  h_in_off.data(),  num_chunks * sizeof(uint32_t), cudaMemcpyHostToDevice, stream));

        // Raw-flagged chunks decode to flag_size=0/data_size=0 (skip the
        // flag-bitmap loop entirely); the host copies them through directly.
        std::vector<uint32_t> h_flag_sz_clean(num_chunks), h_data_sz_clean(num_chunks);
        for (uint32_t i = 0; i < num_chunks; i++) {
            h_flag_sz_clean[i] = h_is_raw[i] ? 0u : h_flag_sz[i];
            h_data_sz_clean[i] = h_is_raw[i] ? 0u : h_data_sz[i];
        }
        FZ_CUDA_CHECK(cudaMemcpyAsync(d_flag_sz_, h_flag_sz_clean.data(), num_chunks * sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
        FZ_CUDA_CHECK(cudaMemcpyAsync(d_data_sz_, h_data_sz_clean.data(), num_chunks * sizeof(uint32_t), cudaMemcpyHostToDevice, stream));

        for (uint32_t i = 0; i < num_chunks; i++) {
            if (h_is_raw[i]) {
                FZ_CUDA_CHECK(cudaMemcpyAsync(d_out + (size_t)i * chunk_size_,
                                              d_in + h_in_off[i], h_data_sz[i],
                                              cudaMemcpyDeviceToDevice, stream));
            }
        }

        launchDecode(word_size_, chunk_size_, (int)num_chunks, stream,
                     d_out, d_in, d_in_off_, d_flag_sz_, d_data_sz_);
        FZ_CUDA_CHECK(cudaGetLastError());
        FZ_CUDA_CHECK(cudaStreamSynchronize(stream));

        if (pool) {
            pool->free(d_in_off_, stream); pool->free(d_flag_sz_, stream); pool->free(d_data_sz_, stream);
        } else {
            cudaFree(d_in_off_); cudaFree(d_flag_sz_); cudaFree(d_data_sz_);
        }

        actual_output_size_ = (size_t)orig_total;
        FZ_LOG(DEBUG, "GPULZ decode done: %.1f KB -> %.1f KB",
               in_bytes / 1024.0, actual_output_size_ / 1024.0);
    }
}

} // namespace fz
