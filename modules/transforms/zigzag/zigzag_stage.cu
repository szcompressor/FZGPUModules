#include "transforms/zigzag/zigzag_stage.h"
#include "transforms/zigzag/zigzag.h"
#include "mem/mempool.h"
#include "cuda_check.h"
#include "backend/api.h"
#include <stdexcept>
#include <string>

namespace fz {

// ─────────────────────────────────────────────────────────────────────────────
// Device kernels
// ─────────────────────────────────────────────────────────────────────────────

template<typename TIn, typename TOut>
__global__ void zigzagEncodeKernel(const TIn* __restrict__ in,
                                   TOut*      __restrict__ out,
                                   size_t n)
{
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = Zigzag<TIn>::encode(in[idx]);
}

template<typename TIn, typename TOut>
__global__ void zigzagDecodeKernel(const TOut* __restrict__ in,
                                   TIn*        __restrict__ out,
                                   size_t n)
{
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = Zigzag<TIn>::decode(in[idx]);
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel launcher helpers (one level of indirection forces instantiation)
// ─────────────────────────────────────────────────────────────────────────────

template<typename TIn, typename TOut>
static void launchEncode(const TIn* in, TOut* out, size_t n, cudaStream_t stream)
{
    if (n == 0) return;
    constexpr int kBlock = 256;
    const int grid = static_cast<int>((n + kBlock - 1) / kBlock);
    zigzagEncodeKernel<TIn, TOut><<<grid, kBlock, 0, stream>>>(in, out, n);
}

template<typename TIn, typename TOut>
static void launchDecode(const TOut* in, TIn* out, size_t n, cudaStream_t stream)
{
    if (n == 0) return;
    constexpr int kBlock = 256;
    const int grid = static_cast<int>((n + kBlock - 1) / kBlock);
    zigzagDecodeKernel<TIn, TOut><<<grid, kBlock, 0, stream>>>(in, out, n);
}

// ─────────────────────────────────────────────────────────────────────────────
// ZigzagStage::execute
// ─────────────────────────────────────────────────────────────────────────────

template<typename TIn, typename TOut>
void ZigzagStage<TIn, TOut>::execute(
    cudaStream_t stream,
    MemoryPool* pool,
    const std::vector<void*>& inputs,
    const std::vector<void*>& outputs,
    const std::vector<size_t>& sizes)
{
    (void)pool;

    if (inputs.empty() || outputs.empty() || sizes.empty())
        throw std::runtime_error("ZigzagStage: invalid inputs/outputs");

    const size_t byte_size = sizes[0];
    if (byte_size == 0) {
        actual_output_size_ = 0;
        return;
    }

    static_assert(sizeof(TIn) == sizeof(TOut),
                  "ZigzagStage: word size must match between TIn and TOut; the "
                  "unaligned-tail passthrough below assumes one element width.");
    constexpr size_t kWord = sizeof(TIn);

    // A byte-transparent Zigzag is routinely fed a compressed blob from an
    // upstream coder, whose length has no reason to be a multiple of kWord.
    // The kernels transform whole words only, so the trailing
    // `byte_size % kWord` bytes are outside their range.
    const size_t n    = byte_size / kWord;
    const size_t tail = byte_size - n * kWord;

    if (!is_inverse_) {
        // Forward: TIn[] → TOut[]
        launchEncode<TIn, TOut>(
            static_cast<const TIn*>(inputs[0]),
            static_cast<TOut*>(outputs[0]),
            n, stream);
    } else {
        // Inverse: TOut[] → TIn[]
        launchDecode<TIn, TOut>(
            static_cast<const TOut*>(inputs[0]),
            static_cast<TIn*>(outputs[0]),
            n, stream);
    }

    // Copy the partial trailing word through unchanged.  Zigzag is a per-word
    // bijection, so a fragment of a word cannot be transformed — but it must
    // still be *written*, on both paths symmetrically, or the stage silently
    // emits whatever the pooled output buffer happened to contain while
    // reporting actual_output_size_ == byte_size.  That produced corruption far
    // downstream (a garbled coder stream, decoded out of bounds) whose
    // appearance depended on stale pool contents, so it surfaced only for
    // inputs whose compressed length was not a multiple of kWord.
    if (tail != 0) {
        FZ_CUDA_CHECK(cudaMemcpyAsync(
            static_cast<uint8_t*>(outputs[0]) + n * kWord,
            static_cast<const uint8_t*>(inputs[0]) + n * kWord,
            tail, cudaMemcpyDeviceToDevice, stream));
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(
            std::string("ZigzagStage kernel launch failed: ") +
            cudaGetErrorString(err));

    actual_output_size_ = byte_size;
}

// ─────────────────────────────────────────────────────────────────────────────
// Explicit template instantiations
// ─────────────────────────────────────────────────────────────────────────────

template class ZigzagStage<int8_t,  uint8_t>;
template class ZigzagStage<int16_t, uint16_t>;
template class ZigzagStage<int32_t, uint32_t>;
template class ZigzagStage<int64_t, uint64_t>;

// Explicit kernel instantiations (required to ensure device code is compiled)
template __global__ void zigzagEncodeKernel<int8_t,  uint8_t> (const  int8_t*, uint8_t*,  size_t);
template __global__ void zigzagEncodeKernel<int16_t, uint16_t>(const int16_t*, uint16_t*, size_t);
template __global__ void zigzagEncodeKernel<int32_t, uint32_t>(const int32_t*, uint32_t*, size_t);
template __global__ void zigzagEncodeKernel<int64_t, uint64_t>(const int64_t*, uint64_t*, size_t);

template __global__ void zigzagDecodeKernel<int8_t,  uint8_t> (const  uint8_t*,  int8_t*, size_t);
template __global__ void zigzagDecodeKernel<int16_t, uint16_t>(const uint16_t*, int16_t*, size_t);
template __global__ void zigzagDecodeKernel<int32_t, uint32_t>(const uint32_t*, int32_t*, size_t);
template __global__ void zigzagDecodeKernel<int64_t, uint64_t>(const uint64_t*, int64_t*, size_t);

} // namespace fz
