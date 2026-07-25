/**
 * modules/shufflers/tupl/tupl_stage.cu
 *
 * GPU implementation of TUPLStage — a tuple deinterleave (AoS -> SoA)
 * transpose, ported from `d_TUPL` / `d_iTUPL` in the LC framework
 * (Burtscher et al., BSD-3 licensed).
 *
 * Algorithm overview
 * ------------------
 * Each block of `block_size` bytes is viewed as `tuples` structs of `dim`
 * fields, each field `word_size` bytes wide (`tuples = (block_size /
 * word_size) / dim`, rounded down). Element `i` of the flattened
 * `tuples * dim` word stream belongs to tuple `i / dim`, field `i % dim`.
 *
 *   Forward (AoS -> SoA): out[(i/dim) + (i%dim)*tuples] = in[i]
 *   Inverse (SoA -> AoS): out[i] = in[(i/dim) + (i%dim)*tuples]
 *
 * Any leftover bytes at the tail of a block that don't form a complete tuple
 * (block_size not evenly divisible by dim * word_size — the common case for
 * dim in {3, 6, 12}, since LC's fixed 16 KB chunk doesn't divide evenly by
 * those) are copied byte-for-byte, unchanged by either direction.
 *
 * Block mapping: one CUDA block per `block_size`-byte chunk of the input.
 * A final partial chunk (input size not a multiple of block_size) is copied
 * raw via a plain D2D memcpy, mirroring BitshuffleStage's tail handling.
 */

#include "shufflers/tupl/tupl_stage.h"
#include "mem/mempool.h"
#include "cuda_check.h"
#include "log.h"
#include <stdexcept>
#include <string>
#include <algorithm>

namespace fz {

// ─────────────────────────────────────────────────────────────────────────────
// Kernels — one CUDA block per `block_words`-word chunk.
// ─────────────────────────────────────────────────────────────────────────────

template <typename T>
__global__ void tuplEncodeKernel(
    const T* __restrict__ in,
    T*       __restrict__ out,
    uint32_t block_words,
    uint32_t dim,
    uint32_t tuples)
{
    const uint32_t size       = tuples * dim;
    const uint32_t chunk_base = blockIdx.x * block_words;

    for (uint32_t i = threadIdx.x; i < size; i += blockDim.x) {
        out[chunk_base + (i / dim) + (i % dim) * tuples] = in[chunk_base + i];
    }

    const uint32_t extra = block_words - size;
    if (extra > 0) {
        const uint8_t* in_bytes  = reinterpret_cast<const uint8_t*>(in  + chunk_base) + (size_t)size * sizeof(T);
        uint8_t*       out_bytes = reinterpret_cast<uint8_t*>(out + chunk_base) + (size_t)size * sizeof(T);
        const uint32_t extra_bytes = extra * (uint32_t)sizeof(T);
        for (uint32_t e = threadIdx.x; e < extra_bytes; e += blockDim.x) {
            out_bytes[e] = in_bytes[e];
        }
    }
}

template <typename T>
__global__ void tuplDecodeKernel(
    const T* __restrict__ in,
    T*       __restrict__ out,
    uint32_t block_words,
    uint32_t dim,
    uint32_t tuples)
{
    const uint32_t size       = tuples * dim;
    const uint32_t chunk_base = blockIdx.x * block_words;

    for (uint32_t i = threadIdx.x; i < size; i += blockDim.x) {
        out[chunk_base + i] = in[chunk_base + (i / dim) + (i % dim) * tuples];
    }

    const uint32_t extra = block_words - size;
    if (extra > 0) {
        const uint8_t* in_bytes  = reinterpret_cast<const uint8_t*>(in  + chunk_base) + (size_t)size * sizeof(T);
        uint8_t*       out_bytes = reinterpret_cast<uint8_t*>(out + chunk_base) + (size_t)size * sizeof(T);
        const uint32_t extra_bytes = extra * (uint32_t)sizeof(T);
        for (uint32_t e = threadIdx.x; e < extra_bytes; e += blockDim.x) {
            out_bytes[e] = in_bytes[e];
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TUPLStage::execute
// ─────────────────────────────────────────────────────────────────────────────

namespace {

template <typename T>
void launchTupl(
    bool is_inverse, int grid, int bdim, fz::stream_t stream,
    const void* in, void* out,
    uint32_t block_words, uint32_t dim, uint32_t tuples)
{
    if (!is_inverse) {
        tuplEncodeKernel<T><<<grid, bdim, 0, stream>>>(
            static_cast<const T*>(in), static_cast<T*>(out), block_words, dim, tuples);
    } else {
        tuplDecodeKernel<T><<<grid, bdim, 0, stream>>>(
            static_cast<const T*>(in), static_cast<T*>(out), block_words, dim, tuples);
    }
}

} // namespace

void TUPLStage::execute(
    fz::stream_t stream,
    MemoryPool* pool,
    const std::vector<void*>& inputs,
    const std::vector<void*>& outputs,
    const std::vector<size_t>& sizes)
{
    (void)pool;

    if (inputs.empty() || outputs.empty() || sizes.empty())
        throw std::runtime_error("TUPLStage: invalid inputs/outputs");

    const size_t in_bytes = sizes[0];
    if (in_bytes == 0) { actual_output_size_ = 0; return; }

    validateConfig();

    const size_t full_bytes = (in_bytes / block_size_) * block_size_;
    const size_t tail_bytes = in_bytes - full_bytes;
    const int grid = static_cast<int>(full_bytes / block_size_);

    const uint32_t block_words = static_cast<uint32_t>(block_size_ / word_size_);
    const uint32_t tuples      = block_words / static_cast<uint32_t>(dim_);

    if (grid > 0) {
        const int bdim = static_cast<int>(
            std::min<size_t>(std::max<size_t>((size_t)tuples * dim_, size_t(32)), size_t(1024)));

        switch (word_size_) {
            case 1:
                launchTupl<uint8_t>(is_inverse_, grid, bdim, stream,
                    inputs[0], outputs[0], block_words, dim_, tuples);
                break;
            case 2:
                launchTupl<uint16_t>(is_inverse_, grid, bdim, stream,
                    inputs[0], outputs[0], block_words, dim_, tuples);
                break;
            case 4:
                launchTupl<uint32_t>(is_inverse_, grid, bdim, stream,
                    inputs[0], outputs[0], block_words, dim_, tuples);
                break;
            case 8:
                launchTupl<uint64_t>(is_inverse_, grid, bdim, stream,
                    inputs[0], outputs[0], block_words, dim_, tuples);
                break;
            default:
                throw std::runtime_error("TUPLStage: unsupported word_size");
        }
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(
            std::string("TUPLStage kernel launch failed: ") +
            cudaGetErrorString(err));

    if (tail_bytes > 0) {
        const auto* in_tail = static_cast<const uint8_t*>(inputs[0]) + full_bytes;
        auto* out_tail = static_cast<uint8_t*>(outputs[0]) + full_bytes;
        FZ_CUDA_CHECK(cudaMemcpyAsync(
            out_tail,
            in_tail,
            tail_bytes,
            cudaMemcpyDeviceToDevice,
            stream));
    }

    actual_output_size_ = in_bytes;
    FZ_LOG(TRACE, "TUPL %s: %.1f KB, block=%zu dim=%d word=%d",
           is_inverse_ ? "decode" : "encode",
           in_bytes / 1024.0, block_size_, static_cast<int>(dim_), static_cast<int>(word_size_));
}

} // namespace fz
