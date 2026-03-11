#include "transforms/negabinary/negabinary_stage.h"
#include "transforms/negabinary/negabinary.h"
#include "mem/mempool.h"
#include "cuda_check.h"
#include <cuda_runtime.h>
#include <stdexcept>

namespace fz {

// ─── Encode kernel: TIn[] → TOut[] ───────────────────────────────────────────
template<typename TIn, typename TOut>
__global__ void negabinaryEncodeKernel(const TIn* __restrict__ in,
                                        TOut* __restrict__ out,
                                        size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = Negabinary<TIn>::encode(in[idx]);
}

// ─── Decode kernel: TOut[] → TIn[] ───────────────────────────────────────────
template<typename TIn, typename TOut>
__global__ void negabinaryDecodeKernel(const TOut* __restrict__ in,
                                        TIn* __restrict__ out,
                                        size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = Negabinary<TIn>::decode(in[idx]);
}

// ─── execute() ───────────────────────────────────────────────────────────────
template<typename TIn, typename TOut>
void NegabinaryStage<TIn, TOut>::execute(
    cudaStream_t stream,
    MemoryPool*  pool,
    const std::vector<void*>& inputs,
    const std::vector<void*>& outputs,
    const std::vector<size_t>& sizes
) {
    (void)pool;
    if (inputs.empty() || outputs.empty() || sizes.empty())
        throw std::runtime_error("NegabinaryStage: invalid inputs/outputs");

    size_t in_bytes = sizes[0];
    size_t n        = in_bytes / sizeof(TIn);

    if (n == 0) { actual_output_size_ = 0; return; }

    constexpr int kBlock = 256;
    int grid = static_cast<int>((n + kBlock - 1) / kBlock);

    if (!is_inverse_) {
        negabinaryEncodeKernel<TIn, TOut><<<grid, kBlock, 0, stream>>>(
            static_cast<const TIn*>(inputs[0]),
            static_cast<TOut*>(outputs[0]),
            n);
    } else {
        negabinaryDecodeKernel<TIn, TOut><<<grid, kBlock, 0, stream>>>(
            static_cast<const TOut*>(inputs[0]),
            static_cast<TIn*>(outputs[0]),
            n);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(
            std::string("NegabinaryStage kernel launch failed: ")
            + cudaGetErrorString(err));

    actual_output_size_ = in_bytes;   // size-preserving
}

// ─── Explicit instantiations ──────────────────────────────────────────────────
template class NegabinaryStage<int8_t,  uint8_t>;
template class NegabinaryStage<int16_t, uint16_t>;
template class NegabinaryStage<int32_t, uint32_t>;
template class NegabinaryStage<int64_t, uint64_t>;

template __global__ void negabinaryEncodeKernel<int8_t,  uint8_t> (const  int8_t*, uint8_t*,  size_t);
template __global__ void negabinaryEncodeKernel<int16_t, uint16_t>(const int16_t*, uint16_t*, size_t);
template __global__ void negabinaryEncodeKernel<int32_t, uint32_t>(const int32_t*, uint32_t*, size_t);
template __global__ void negabinaryEncodeKernel<int64_t, uint64_t>(const int64_t*, uint64_t*, size_t);

template __global__ void negabinaryDecodeKernel<int8_t,  uint8_t> (const  uint8_t*,  int8_t*, size_t);
template __global__ void negabinaryDecodeKernel<int16_t, uint16_t>(const uint16_t*, int16_t*, size_t);
template __global__ void negabinaryDecodeKernel<int32_t, uint32_t>(const uint32_t*, int32_t*, size_t);
template __global__ void negabinaryDecodeKernel<int64_t, uint64_t>(const uint64_t*, int64_t*, size_t);

} // namespace fz
