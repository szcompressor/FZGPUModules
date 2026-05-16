/**
 * Adapted from PHF reference (hist.cuhip.inl)
 * Original authors: Cody Rivera (cjrivera1@crimson.ua.edu),
 *                   Megan Hickman Fulp (mlhickm@g.clemson.edu)
 * Based on: J. Gómez-Luna et al., "In-place data sliding algorithms for
 *            many-core architectures," 2013.
 * Changes:
 *   - Removed #include "timer.hh"
 *   - Changed #include "hist.hh" → "histogram.h"
 *   - Removed SEQ_histogram_generic, SEQ_histogram_Cauchy_v2, GPU_histogram_Cauchy,
 *     and KERNEL_CUHIP_histogram_naive (none used by HuffmanStage)
 *   - Replaced C++ alternative token `and` with `&&`
 *   - Qualified kernel as fz::KERNEL_CUHIP_p2013Histogram in optimizer and launcher
 */

#include <algorithm>
#include <cstdint>

#include "histogram.h"

static const unsigned int WARP_SIZE = 32;

namespace fz {

template <typename T, typename FREQ>
__global__ void KERNEL_CUHIP_p2013Histogram(
    T* in_data, size_t const data_len, FREQ* out_bins,
    uint16_t const bins_len, uint16_t const repeat)
{
    extern __shared__ int Hs[];

    const unsigned int warp_id     = (int)(threadIdx.x / WARP_SIZE);
    const unsigned int lane        = threadIdx.x % WARP_SIZE;
    const unsigned int warps_block = blockDim.x / WARP_SIZE;
    const unsigned int off_rep     = (bins_len + 1) * (threadIdx.x % repeat);
    const unsigned int begin       = (data_len / warps_block) * warp_id
                                     + WARP_SIZE * blockIdx.x + lane;
    unsigned int       end         = (data_len / warps_block) * (warp_id + 1);
    const unsigned int step        = WARP_SIZE * gridDim.x;

    if (warp_id >= warps_block - 1) end = data_len;

    for (unsigned int pos = threadIdx.x; pos < (bins_len + 1) * repeat; pos += blockDim.x)
        Hs[pos] = 0;
    __syncthreads();

    for (unsigned int i = begin; i < end; i += step) {
        auto sym = static_cast<uint32_t>(in_data[i]);
        if (sym >= static_cast<uint32_t>(bins_len)) continue;
        atomicAdd(&Hs[off_rep + static_cast<int>(sym)], 1);
    }
    __syncthreads();

    for (unsigned int pos = threadIdx.x; pos < bins_len; pos += blockDim.x) {
        int sum = 0;
        for (int base = 0; base < (bins_len + 1) * repeat; base += bins_len + 1)
            sum += Hs[base + pos];
        atomicAdd(out_bins + pos, sum);
    }
}

}  // namespace fz

namespace fz::module {

template <typename T>
void GPU_histogram_generic_optimizer_on_initialization(
    size_t const data_len, uint16_t const hist_len,
    int& grid_dim, int& block_dim, int& shmem_use, int& r_per_block)
{
    int device_id, max_bytes, num_SMs;

    cudaGetDevice(&device_id);
    cudaDeviceGetAttribute(&num_SMs, cudaDevAttrMultiProcessorCount, device_id);

    int max_bytes_opt_in;
    cudaDeviceGetAttribute(&max_bytes,         cudaDevAttrMaxSharedMemoryPerBlock,       device_id);
    cudaDeviceGetAttribute(&max_bytes_opt_in,  cudaDevAttrMaxSharedMemoryPerBlockOptin,  device_id);
    max_bytes = std::max(max_bytes, max_bytes_opt_in);

    cudaFuncSetAttribute(
        (void*)fz::KERNEL_CUHIP_p2013Histogram<T, uint32_t>,
        (cudaFuncAttribute)cudaFuncAttributeMaxDynamicSharedMemorySize,
        max_bytes);

    r_per_block = (max_bytes / sizeof(int)) / (hist_len + 1);
    grid_dim    = num_SMs;
    block_dim   = ((((data_len / (grid_dim * 1)) + 1) / 64) + 1) * 64;
    while (block_dim > 1024) {
        if (r_per_block <= 1) {
            block_dim = 1024;
        } else {
            r_per_block /= 2;
            grid_dim    *= 2;
            block_dim    = ((((data_len / (grid_dim * 1)) + 1) / 64) + 1) * 64;
        }
    }
    shmem_use = ((hist_len + 1) * r_per_block) * sizeof(int);
}

template <typename T>
int GPU_histogram_generic(
    T* in_data, size_t const data_len, uint32_t* out_hist, uint16_t const hist_len,
    int const grid_dim, int const block_dim, int const shmem_use, int const r_per_block,
    void* stream)
{
    fz::KERNEL_CUHIP_p2013Histogram<<<grid_dim, block_dim, shmem_use, (cudaStream_t)stream>>>(
        in_data, data_len, out_hist, hist_len, (uint16_t)r_per_block);
    return 0;
}

}  // namespace fz::module

#define INIT_HIST(E)                                                                         \
    template void fz::module::GPU_histogram_generic_optimizer_on_initialization<E>(          \
        size_t const, uint16_t const, int&, int&, int&, int&);                               \
    template int fz::module::GPU_histogram_generic<E>(                                       \
        E*, size_t const, uint32_t*, uint16_t const,                                         \
        int const, int const, int const, int const, void*);

INIT_HIST(uint8_t)
INIT_HIST(uint16_t)
INIT_HIST(uint32_t)

#undef INIT_HIST
