/**
 * predictors/lorenzo/lorenzo_nd.cu
 *
 * 2-D and 3-D Lorenzo prediction kernels and their launchers.
 * Split from lorenzo.cu to keep per-file size manageable.
 * The LorenzoStage class implementation remains in lorenzo.cu.
 */

#include "predictors/lorenzo/lorenzo.h"
#include "predictors/lorenzo/lorenzo_kernels.cuh"
#include "transforms/zigzag/zigzag.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <type_traits>
#include <stdexcept>
#include <string>
#include "mem/mempool.h"
#include "cuda_check.h"

namespace fz {

// ========== 2-D Compression Kernel ==========
// Adapted from cuSZ KERNEL_CUHIP_c_lorenzo_2d__32x32.
// TileDim=32(x), Yseq=8, NumWarps=4(y).
// blockDim=(32,4,1), gridDim=(ceil(nx/32), ceil(ny/32), 1).
// Outlier format: separate errors/indices/count arrays (no ZigZag).

template<typename TInput, typename TCode, bool ZigzagCodes = false>
__global__ void lorenzo_quantize_2d_kernel(
    const TInput* __restrict__ in_data,
    uint32_t data_lenx, uint32_t data_leny, uint32_t data_leapy,
    TInput ebx2_r, TCode quant_radius,
    TCode*    __restrict__ out_codes,
    TInput*   __restrict__ out_outlier_errors,
    uint32_t* __restrict__ out_outlier_indices,
    uint32_t* __restrict__ out_outlier_count,
    size_t max_outliers
) {
    constexpr int TileDim  = 32;
    constexpr int Yseq     = 8;
    constexpr int NumWarps = 4;   // BDY = 4, each warp covers Yseq rows

    // Inter-warp y-boundary exchange: warp w provides its last row to warp w+1.
    __shared__ TInput exchange[NumWarps - 1][TileDim + 1];

    // center[0] = value from preceding y-thread (or 0 for warp 0)
    // center[1..Yseq] = this thread's Yseq rows
    TInput center[Yseq + 1] = {0};

    const auto gix      = blockIdx.x * TileDim + threadIdx.x;
    const auto giy_base = blockIdx.y * TileDim + threadIdx.y * Yseq;
    auto g_id = [&](int i) -> uint32_t { return (giy_base + i) * data_leapy + gix; };

    // Load + pre-quantize
#pragma unroll
    for (int iy = 0; iy < Yseq; iy++) {
        if (gix < data_lenx && (giy_base + iy) < data_leny)
            center[iy + 1] = round(in_data[g_id(iy)] * ebx2_r);
    }

    // Cross-warp boundary: pass last row down to next warp
    if (threadIdx.y < NumWarps - 1)
        exchange[threadIdx.y][threadIdx.x] = center[Yseq];
    __syncthreads();
    if (threadIdx.y > 0)
        center[0] = exchange[threadIdx.y - 1][threadIdx.x];
    __syncthreads();

    // Lorenzo 2D prediction + quantize (iterate y in reverse so center[i-1] is intact)
#pragma unroll
    for (int i = Yseq; i > 0; i--) {
        // y-direction: subtract previous row
        center[i] -= center[i - 1];
        // x-direction: subtract west neighbour via full-warp shuffle
        TInput west = __shfl_up_sync(0xffffffff, center[i], 1, 32);
        if (threadIdx.x > 0) center[i] -= west;

        auto gid       = g_id(i - 1);
        bool is_valid  = (gix < data_lenx && (giy_base + i - 1) < data_leny);

        if (is_valid) {
            bool quantizable = fabsf(center[i]) < static_cast<TInput>(quant_radius);
            if (quantizable) {
                if constexpr (ZigzagCodes) {
                    using SCode = typename std::make_signed<TCode>::type;
                    out_codes[gid] = static_cast<TCode>(
                        Zigzag<SCode>::encode(static_cast<SCode>(static_cast<int>(center[i]))));
                } else {
                    out_codes[gid] = static_cast<TCode>(static_cast<int>(center[i]));
                }
            } else {
                out_codes[gid] = static_cast<TCode>(0);   // 0 contributes 0 to prefix sum
            }

            if (!quantizable) {
                uint32_t cur_idx = atomicAdd(out_outlier_count, 1u);
                if (cur_idx < max_outliers) {
                    out_outlier_errors[cur_idx]  = center[i];   // pre-quantized delta
                    out_outlier_indices[cur_idx] = gid;
                }
            }
        }
    }
}

// ========== 2-D Decompression Kernel ==========
// Adapted from cuSZ KERNEL_CUHIP_x_lorenzo_2d__32x32.
// Outliers are pre-scattered into inout_data before this kernel is called.
// blockDim=(32,4,1), gridDim=(ceil(nx/32), ceil(ny/32), 1).

template<typename TInput, typename TCode, bool ZigzagCodes = false>
__global__ void lorenzo_dequantize_2d_kernel(
    const TCode* __restrict__ in_codes,
    TInput*      __restrict__ inout_data,   // holds scattered outlier deltas on entry, output on exit
    uint32_t data_lenx, uint32_t data_leny, uint32_t data_leapy,
    TCode quant_radius, TInput ebx2
) {
    constexpr int TileDim  = 32;
    constexpr int NumWarps = 4;
    constexpr int Yseq     = TileDim / NumWarps;   // 8

    __shared__ TInput scratch[NumWarps - 1][TileDim + 1];

    TInput thp_data[Yseq] = {0};

    const auto gix      = blockIdx.x * TileDim + threadIdx.x;
    const auto giy_base = blockIdx.y * TileDim + threadIdx.y * Yseq;
    auto get_gid = [&](int i) -> uint32_t { return (giy_base + i) * data_leapy + gix; };

    // Load: fuse scattered outlier + (code - radius)
#pragma unroll
    for (int i = 0; i < Yseq; i++) {
        auto gid = get_gid(i);
        if (gix < data_lenx && (giy_base + i) < data_leny) {
            using SCode = typename std::make_signed<TCode>::type;
            TInput decoded;
            if constexpr (ZigzagCodes) {
                decoded = static_cast<TInput>(Zigzag<SCode>::decode(in_codes[gid]));
            } else {
                decoded = static_cast<TInput>(static_cast<SCode>(in_codes[gid]));
            }
            thp_data[i] = inout_data[gid] + decoded;
        }
    }

    // Partial-sum along y sequentially (restores y-Lorenzo)
    for (int i = 1; i < Yseq; i++) thp_data[i] += thp_data[i - 1];

    // Cross-warp y scan (exclusive prefix across the 4 warps)
    if (threadIdx.y < NumWarps - 1)
        scratch[threadIdx.y][threadIdx.x] = thp_data[Yseq - 1];
    __syncthreads();

    if (threadIdx.y == 0) {
        TInput warp_accum[NumWarps - 1];
#pragma unroll
        for (int i = 0; i < NumWarps - 1; i++) warp_accum[i] = scratch[i][threadIdx.x];
#pragma unroll
        for (int i = 1; i < NumWarps - 1; i++) warp_accum[i] += warp_accum[i - 1];
#pragma unroll
        for (int i = 1; i < NumWarps - 1; i++) scratch[i][threadIdx.x] = warp_accum[i];
    }
    __syncthreads();

    if (threadIdx.y > 0) {
        TInput addend = scratch[threadIdx.y - 1][threadIdx.x];
#pragma unroll
        for (int i = 0; i < Yseq; i++) thp_data[i] += addend;
    }
    __syncthreads();

    // x-axis partial sum via full-warp shuffle, then scale by ebx2
#pragma unroll
    for (int i = 0; i < Yseq; i++) {
        for (int d = 1; d < TileDim; d *= 2) {
            TInput n = __shfl_up_sync(0xffffffff, thp_data[i], d, 32);
            if (threadIdx.x >= d) thp_data[i] += n;
        }
        thp_data[i] *= ebx2;
    }

    // Write output
#pragma unroll
    for (int i = 0; i < Yseq; i++) {
        auto gid = get_gid(i);
        if (gix < data_lenx && (giy_base + i) < data_leny)
            inout_data[gid] = thp_data[i];
    }
}

// ========== 3-D Compression Kernel ==========
// Adapted from cuSZ KERNEL_CUHIP_c_lorenzo_3d.
// blockDim=(32,8,1), gridDim=(ceil(nx/(4*8)), ceil(ny/8), ceil(nz/8)).
// threadIdx.x ∈ [0,32): covers 4 segments of 8 in the x-dimension.
// threadIdx.y ∈ [0,8): covers one TileDim slice in y.
// z is iterated sequentially per thread (TileDim=8 z-slices per block).

template<typename TInput, typename TCode, bool ZigzagCodes = false>
__global__ void lorenzo_quantize_3d_kernel(
    const TInput* __restrict__ in_data,
    uint32_t data_lenx, uint32_t data_leny, uint32_t data_leapy,
    uint32_t data_lenz, uint32_t data_leapz,
    TInput ebx2_r, TCode quant_radius,
    TCode*    __restrict__ out_codes,
    TInput*   __restrict__ out_outlier_errors,
    uint32_t* __restrict__ out_outlier_indices,
    uint32_t* __restrict__ out_outlier_count,
    size_t max_outliers
) {
    constexpr int TileDim = 8;

    // s[threadIdx.y + 1][threadIdx.x]: ghost row at index 0 for y-direction diff
    __shared__ TInput s[9][33];

    TInput delta[TileDim + 1] = {0};   // delta[0] = ghost (0), delta[1..TileDim] = slices

    const auto gix      = blockIdx.x * (TileDim * 4) + threadIdx.x;
    const auto giy      = blockIdx.y * TileDim + threadIdx.y;
    const auto giz_base = blockIdx.z * TileDim;
    const auto base_id  = gix + giy * data_leapy + giz_base * data_leapz;

    auto giz = [&](int z) { return giz_base + z; };
    auto gid = [&](int z) -> uint32_t { return base_id + z * data_leapz; };

    // Load + pre-quantize along z
    if (gix < data_lenx && giy < data_leny) {
        for (int z = 0; z < TileDim; z++)
            if ((uint32_t)giz(z) < data_lenz)
                delta[z + 1] = round(in_data[gid(z)] * ebx2_r);
    }
    __syncthreads();

    for (int z = TileDim; z > 0; z--) {
        // z-direction: subtract previous z-slice
        delta[z] -= delta[z - 1];

        // x-direction: subtract west neighbour within 8-thread segment
        auto seg_tix = threadIdx.x % TileDim;
        TInput prev_x = __shfl_up_sync(0xffffffff, delta[z], 1, 8);
        if (seg_tix > 0) delta[z] -= prev_x;

        // y-direction: exchange via shared memory (ghost row at index 0)
        s[threadIdx.y + 1][threadIdx.x] = delta[z];
        __syncthreads();
        delta[z] -= (threadIdx.y > 0) * s[threadIdx.y][threadIdx.x];

        // Quantize and write
        bool is_valid = (gix < data_lenx && giy < data_leny && (uint32_t)giz(z - 1) < data_lenz);
        if (is_valid) {
            bool quantizable = fabsf(delta[z]) < static_cast<TInput>(quant_radius);
            if (quantizable) {
                if constexpr (ZigzagCodes) {
                    using SCode = typename std::make_signed<TCode>::type;
                    out_codes[gid(z - 1)] = static_cast<TCode>(
                        Zigzag<SCode>::encode(static_cast<SCode>(static_cast<int>(delta[z]))));
                } else {
                    out_codes[gid(z - 1)] = static_cast<TCode>(static_cast<int>(delta[z]));
                }
            } else {
                out_codes[gid(z - 1)] = static_cast<TCode>(0);   // 0 contributes 0 to prefix sum
            }

            if (!quantizable) {
                uint32_t cur_idx = atomicAdd(out_outlier_count, 1u);
                if (cur_idx < max_outliers) {
                    out_outlier_errors[cur_idx]  = delta[z];   // pre-quantized delta
                    out_outlier_indices[cur_idx] = gid(z - 1);
                }
            }
        }
        __syncthreads();
    }
}

// ========== 3-D Decompression Kernel ==========
// Adapted from cuSZ KERNEL_CUHIP_x_lorenzo_3d.
// Outliers are pre-scattered into inout_data before this kernel is called.
// blockDim=(32,1,8), gridDim=(ceil(nx/(4*8)), ceil(ny/8), ceil(nz/8)).

template<typename TInput, typename TCode, bool ZigzagCodes = false>
__global__ void lorenzo_dequantize_3d_kernel(
    const TCode* __restrict__ in_codes,
    TInput*      __restrict__ inout_data,
    uint32_t data_lenx, uint32_t data_leny, uint32_t data_leapy,
    uint32_t data_lenz, uint32_t data_leapz,
    TCode quant_radius, TInput ebx2
) {
    constexpr int TileDim = 8;
    constexpr int Yseq    = TileDim;

    // scratch[TileDim][4][8] — used for x-z transpose during 3D partial sum
    __shared__ TInput scratch[TileDim][4][8];

    TInput thread_private[Yseq] = {0};

    const auto seg_id  = threadIdx.x / 8;
    const auto seg_tix = threadIdx.x % 8;

    const auto gix      = blockIdx.x * (4 * TileDim) + threadIdx.x;
    const auto giy_base = blockIdx.y * TileDim;
    const auto giz      = blockIdx.z * TileDim + threadIdx.z;

    auto giy = [&](int y) { return giy_base + y; };
    auto gid = [&](int y) -> uint32_t {
        return giz * data_leapz + (giy_base + y) * data_leapy + gix;
    };

    // Load: fuse scattered outlier + (code - radius)
#pragma unroll
    for (int y = 0; y < Yseq; y++) {
        if (gix < data_lenx && (uint32_t)giy(y) < data_leny && giz < data_lenz) {
            using SCode = typename std::make_signed<TCode>::type;
            TInput decoded;
            if constexpr (ZigzagCodes) {
                decoded = static_cast<TInput>(Zigzag<SCode>::decode(in_codes[gid(y)]));
            } else {
                decoded = static_cast<TInput>(static_cast<SCode>(in_codes[gid(y)]));
            }
            thread_private[y] = inout_data[gid(y)] + decoded;
        }
    }

    // Partial-sum along y sequentially (restores y-Lorenzo)
    for (int y = 1; y < Yseq; y++) thread_private[y] += thread_private[y - 1];

    // x and z partial sums via warp shuffle + shared-memory transpose
#pragma unroll
    for (int i = 0; i < TileDim; i++) {
        TInput val = thread_private[i];

        // x partial sum within 8-thread segment
        for (int dist = 1; dist < TileDim; dist *= 2) {
            TInput addend = __shfl_up_sync(0xffffffff, val, dist, 8);
            if (seg_tix >= dist) val += addend;
        }

        // Transpose x <-> z through shared memory
        scratch[threadIdx.z][seg_id][seg_tix] = val;
        __syncthreads();
        val = scratch[seg_tix][seg_id][threadIdx.z];
        __syncthreads();

        // z partial sum within 8-thread segment
        for (int dist = 1; dist < TileDim; dist *= 2) {
            TInput addend = __shfl_up_sync(0xffffffff, val, dist, 8);
            if (seg_tix >= dist) val += addend;
        }

        // Transpose back z <-> x
        scratch[threadIdx.z][seg_id][seg_tix] = val;
        __syncthreads();
        val = scratch[seg_tix][seg_id][threadIdx.z];
        __syncthreads();

        thread_private[i] = val;
    }

    // Write: scale by ebx2 and store
#pragma unroll
    for (int y = 0; y < Yseq; y++)
        if (gix < data_lenx && (uint32_t)giy(y) < data_leny && giz < data_lenz)
            inout_data[gid(y)] = thread_private[y] * ebx2;
}

// ========== 2-D Kernel Launchers ==========

template<typename TInput, typename TCode>
void launchLorenzoKernel2D(
    const TInput* d_input, size_t nx, size_t ny,
    TInput ebx2_r, TCode quant_radius,
    TCode* d_codes, TInput* d_outlier_errors,
    uint32_t* d_outlier_indices, uint32_t* d_outlier_count,
    size_t max_outliers,
    bool zigzag_codes,
    cudaStream_t stream
) {
    constexpr int TileDim = 32;
    constexpr int BDY     = 4;    // NumWarps in y

    dim3 block(TileDim, BDY, 1);
    dim3 grid(
        (static_cast<uint32_t>(nx) + TileDim - 1) / TileDim,
        (static_cast<uint32_t>(ny) + TileDim - 1) / TileDim,
        1
    );
    uint32_t leapy = static_cast<uint32_t>(nx);

    if (zigzag_codes) {
        lorenzo_quantize_2d_kernel<TInput, TCode, true><<<grid, block, 0, stream>>>(
            d_input,
            static_cast<uint32_t>(nx), static_cast<uint32_t>(ny), leapy,
            ebx2_r, quant_radius,
            d_codes, d_outlier_errors, d_outlier_indices, d_outlier_count,
            max_outliers
        );
    } else {
        lorenzo_quantize_2d_kernel<TInput, TCode, false><<<grid, block, 0, stream>>>(
            d_input,
            static_cast<uint32_t>(nx), static_cast<uint32_t>(ny), leapy,
            ebx2_r, quant_radius,
            d_codes, d_outlier_errors, d_outlier_indices, d_outlier_count,
            max_outliers
        );
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(
            std::string("lorenzo_quantize_2d_kernel launch failed: ") + cudaGetErrorString(err));
}

template<typename TInput, typename TCode>
void launchLorenzoInverseKernel2D(
    const TCode* d_codes,
    const TInput* d_outlier_errors, const uint32_t* d_outlier_indices,
    const uint32_t* d_outlier_count,
    size_t nx, size_t ny, size_t max_outliers,
    TInput ebx2, TCode quant_radius,
    TInput* d_output,
    bool zigzag_codes,
    cudaStream_t stream, MemoryPool* pool
) {
    (void)pool;

    const size_t total = nx * ny;
    constexpr int TileDim = 32;
    constexpr int BDY     = 4;

    // Step 0: zero output (scatter target)
    FZ_CUDA_CHECK(cudaMemsetAsync(d_output, 0, total * sizeof(TInput), stream));

    // Step 1: scatter outlier prediction errors into output
    if (d_outlier_count != nullptr && max_outliers > 0) {
        int sblk  = 256;
        int sgrid = (static_cast<int>(max_outliers) + sblk - 1) / sblk;
        scatter_outliers_kernel<TInput><<<sgrid, sblk, 0, stream>>>(
            d_outlier_errors, d_outlier_indices, d_outlier_count, d_output);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            throw std::runtime_error(
                std::string("scatter_outliers_kernel (2D) launch failed: ") +
                cudaGetErrorString(err));
    }

    // Step 2: 2D inverse Lorenzo (dequantize + 2D partial sum)
    dim3 block(TileDim, BDY, 1);
    dim3 grid(
        (static_cast<uint32_t>(nx) + TileDim - 1) / TileDim,
        (static_cast<uint32_t>(ny) + TileDim - 1) / TileDim,
        1
    );
    uint32_t leapy = static_cast<uint32_t>(nx);

    if (zigzag_codes) {
        lorenzo_dequantize_2d_kernel<TInput, TCode, true><<<grid, block, 0, stream>>>(
            d_codes, d_output,
            static_cast<uint32_t>(nx), static_cast<uint32_t>(ny), leapy,
            quant_radius, ebx2
        );
    } else {
        lorenzo_dequantize_2d_kernel<TInput, TCode, false><<<grid, block, 0, stream>>>(
            d_codes, d_output,
            static_cast<uint32_t>(nx), static_cast<uint32_t>(ny), leapy,
            quant_radius, ebx2
        );
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(
            std::string("lorenzo_dequantize_2d_kernel launch failed: ") + cudaGetErrorString(err));
}

// ========== 3-D Kernel Launchers ==========

template<typename TInput, typename TCode>
void launchLorenzoKernel3D(
    const TInput* d_input, size_t nx, size_t ny, size_t nz,
    TInput ebx2_r, TCode quant_radius,
    TCode* d_codes, TInput* d_outlier_errors,
    uint32_t* d_outlier_indices, uint32_t* d_outlier_count,
    size_t max_outliers,
    bool zigzag_codes,
    cudaStream_t stream
) {
    constexpr int TileDim = 8;

    dim3 block(TileDim * 4, TileDim, 1);
    dim3 grid(
        (static_cast<uint32_t>(nx) + (TileDim * 4) - 1) / (TileDim * 4),
        (static_cast<uint32_t>(ny) + TileDim - 1) / TileDim,
        (static_cast<uint32_t>(nz) + TileDim - 1) / TileDim
    );
    uint32_t leapy = static_cast<uint32_t>(nx);
    uint32_t leapz = static_cast<uint32_t>(nx) * static_cast<uint32_t>(ny);

    if (zigzag_codes) {
        lorenzo_quantize_3d_kernel<TInput, TCode, true><<<grid, block, 0, stream>>>(
            d_input,
            static_cast<uint32_t>(nx), static_cast<uint32_t>(ny), leapy,
            static_cast<uint32_t>(nz), leapz,
            ebx2_r, quant_radius,
            d_codes, d_outlier_errors, d_outlier_indices, d_outlier_count,
            max_outliers
        );
    } else {
        lorenzo_quantize_3d_kernel<TInput, TCode, false><<<grid, block, 0, stream>>>(
            d_input,
            static_cast<uint32_t>(nx), static_cast<uint32_t>(ny), leapy,
            static_cast<uint32_t>(nz), leapz,
            ebx2_r, quant_radius,
            d_codes, d_outlier_errors, d_outlier_indices, d_outlier_count,
            max_outliers
        );
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(
            std::string("lorenzo_quantize_3d_kernel launch failed: ") + cudaGetErrorString(err));
}

template<typename TInput, typename TCode>
void launchLorenzoInverseKernel3D(
    const TCode* d_codes,
    const TInput* d_outlier_errors, const uint32_t* d_outlier_indices,
    const uint32_t* d_outlier_count,
    size_t nx, size_t ny, size_t nz, size_t max_outliers,
    TInput ebx2, TCode quant_radius,
    TInput* d_output,
    bool zigzag_codes,
    cudaStream_t stream, MemoryPool* pool
) {
    (void)pool;

    const size_t total = nx * ny * nz;
    constexpr int TileDim = 8;

    // Step 0: zero output (scatter target)
    FZ_CUDA_CHECK(cudaMemsetAsync(d_output, 0, total * sizeof(TInput), stream));

    // Step 1: scatter outlier prediction errors into output
    if (d_outlier_count != nullptr && max_outliers > 0) {
        int sblk  = 256;
        int sgrid = (static_cast<int>(max_outliers) + sblk - 1) / sblk;
        scatter_outliers_kernel<TInput><<<sgrid, sblk, 0, stream>>>(
            d_outlier_errors, d_outlier_indices, d_outlier_count, d_output);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            throw std::runtime_error(
                std::string("scatter_outliers_kernel (3D) launch failed: ") +
                cudaGetErrorString(err));
    }

    // Step 2: 3D inverse Lorenzo (dequantize + 3D partial sum)
    // blockDim=(32,1,8): 32 x-threads covering 4 x-segments, 8 z-threads for z scan
    dim3 block(32, 1, TileDim);
    dim3 grid(
        (static_cast<uint32_t>(nx) + (4 * TileDim) - 1) / (4 * TileDim),
        (static_cast<uint32_t>(ny) + TileDim - 1) / TileDim,
        (static_cast<uint32_t>(nz) + TileDim - 1) / TileDim
    );
    uint32_t leapy = static_cast<uint32_t>(nx);
    uint32_t leapz = static_cast<uint32_t>(nx) * static_cast<uint32_t>(ny);

    if (zigzag_codes) {
        lorenzo_dequantize_3d_kernel<TInput, TCode, true><<<grid, block, 0, stream>>>(
            d_codes, d_output,
            static_cast<uint32_t>(nx), static_cast<uint32_t>(ny), leapy,
            static_cast<uint32_t>(nz), leapz,
            quant_radius, ebx2
        );
    } else {
        lorenzo_dequantize_3d_kernel<TInput, TCode, false><<<grid, block, 0, stream>>>(
            d_codes, d_output,
            static_cast<uint32_t>(nx), static_cast<uint32_t>(ny), leapy,
            static_cast<uint32_t>(nz), leapz,
            quant_radius, ebx2
        );
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(
            std::string("lorenzo_dequantize_3d_kernel launch failed: ") + cudaGetErrorString(err));
}

// Explicit instantiations for 2-D / 3-D launchers (same type combos as 1-D)
#define INSTANTIATE_LORENZO_ND(TInput, TCode) \
    template void launchLorenzoKernel2D<TInput, TCode>( \
        const TInput*, size_t, size_t, TInput, TCode, \
        TCode*, TInput*, uint32_t*, uint32_t*, size_t, bool, cudaStream_t); \
    template void launchLorenzoInverseKernel2D<TInput, TCode>( \
        const TCode*, const TInput*, const uint32_t*, const uint32_t*, \
        size_t, size_t, size_t, TInput, TCode, TInput*, bool, cudaStream_t, MemoryPool*); \
    template void launchLorenzoKernel3D<TInput, TCode>( \
        const TInput*, size_t, size_t, size_t, TInput, TCode, \
        TCode*, TInput*, uint32_t*, uint32_t*, size_t, bool, cudaStream_t); \
    template void launchLorenzoInverseKernel3D<TInput, TCode>( \
        const TCode*, const TInput*, const uint32_t*, const uint32_t*, \
        size_t, size_t, size_t, size_t, TInput, TCode, TInput*, bool, cudaStream_t, MemoryPool*);

INSTANTIATE_LORENZO_ND(float,  uint16_t)
INSTANTIATE_LORENZO_ND(float,  uint8_t)
INSTANTIATE_LORENZO_ND(double, uint16_t)
INSTANTIATE_LORENZO_ND(double, uint32_t)

#undef INSTANTIATE_LORENZO_ND

} // namespace fz
