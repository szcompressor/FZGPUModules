// Adapted from the MANS project (Huang et al.), BSD-3-Clause License — see THIRD_PARTY.md.
// Original: nv/adm/mapping_uint32.cu in https://github.com/hpdps-group/MANS
//
// Changes from original: identical to mapping_uint16.cu changes (see that file).
// Structural differences vs u16: uint32_t input/centers, max_signal_bytes=4,
// decompress result vectors use int4 (4 bytes) rather than int4 (2-byte packed u16).

#include "transforms/adm/adm_kernels.h"
#include "backend/algorithms.h"
#include "cuda_check.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace fz {
namespace adm {

// ── Kernel implementations ────────────────────────────────────────────────────

__global__ static void adm_decompress_u32(
    uint32_t* decmp_data, uint32_t* centers, uint8_t* codes,
    int* d_output_lengths, uint8_t* d_concatenated_signals,
    int data_size, int last_length, int gsize, int shift)
{
    const int idx  = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane = idx & 0x1f;
    const int warp = idx >> 5;

    if (idx * kDecmpChunk >= data_size) return;
    int end = (d_output_lengths[gsize - 1] + last_length) * kBlockThreads;

    int w = lane < 16 ? warp * 2 : warp * 2 + 1;
    int length = (w == (gsize - 1)) ? last_length
                                    : d_output_lengths[w + 1] - d_output_lengths[w];

    int src_start = d_output_lengths[w] * kBlockThreads;
    src_start = lane < 16 ? src_start + lane * length * 2
                           : src_start + (lane - 16) * length * 2;
    int dst_start = idx * kDecmpChunk;

    uint8_t bit_buffer = 0;
    int signal_idx = -1, offset_byte = 0;
    bool bit = 0;
    uint8_t local_signal[kDecmpChunk] = {0};

    #pragma unroll
    for (; offset_byte < length && signal_idx < kChunk; offset_byte++) {
        bit_buffer = d_concatenated_signals[src_start + offset_byte];
        #pragma unroll
        for (int i = 7; i >= 0 && signal_idx < kChunk; i--) {
            bit = (bit_buffer >> i) & 1;
            if (bit) signal_idx++;
            else     local_signal[signal_idx]++;
        }
    }

    offset_byte = 0; signal_idx = 15;
    #pragma unroll
    for (; offset_byte < length && signal_idx < kDecmpChunk; offset_byte++) {
        if (src_start + offset_byte + length > end) return;
        bit_buffer = d_concatenated_signals[src_start + offset_byte + length];
        #pragma unroll
        for (int i = 7; i >= 0 && signal_idx < kDecmpChunk; i--) {
            bit = (bit_buffer >> i) & 1;
            if (bit) signal_idx++;
            else     local_signal[signal_idx]++;
        }
    }

    uint8_t local_codes[kDecmpChunk];
    int4* local_codes_v = reinterpret_cast<int4*>(local_codes);
    int4* codes_v       = reinterpret_cast<int4*>(codes + dst_start);
    #pragma unroll
    for (int i = 0; i < kDecmpChunk / 16; ++i) local_codes_v[i] = codes_v[i];

    uint32_t local_result[kDecmpChunk];
    int center = lane < 16 ? static_cast<int>(centers[blockIdx.x * 2])
                           : static_cast<int>(centers[blockIdx.x * 2 + 1]);

    for (int i = 0; i < kDecmpChunk; i++) {
        uint8_t code   = local_codes[i];
        uint8_t signal = local_signal[i];
        int diff = (code % 2 == 1) ? (code - 1) / 2 : code / 2;
        diff += signal * 126;
        local_result[i] = static_cast<uint32_t>(
            (code % 2 == 1) ? center - diff : center + diff);
    }

    // Vectorised int4 stores cover kDecmpChunk uint32 elements (128 B / thread).
    // Guard the tail: see mapping_uint16.cu for the same fix.
    if (dst_start + kDecmpChunk <= data_size) {
        int4* result_v = reinterpret_cast<int4*>(local_result);
        int4* out_v    = reinterpret_cast<int4*>(decmp_data + dst_start);
        #pragma unroll
        for (int i = 0; i < kDecmpChunk / 4; ++i) out_v[i] = result_v[i];
    } else {
        int tail_end = min(dst_start + kDecmpChunk, data_size);
        for (int i = dst_start; i < tail_end; ++i)
            decmp_data[i] = local_result[i - dst_start];
    }
}


__global__ static void adm_map_thrust_u32(
    const uint32_t* data, uint8_t* code, uint8_t* bit_signal,
    uint32_t* centers, int* signal_length, uint32_t* block_flags,
    int data_size, int shift, unsigned int* d_overflow_flag)
{
    (void)block_flags;
    const int idx  = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane = idx & 0x1f;
    const int warp = idx >> 5;

    int base_block = warp * kChunk * kBlockThreads;
    int base_th    = base_block + lane * kChunk;
    int base_th_end = base_th + kChunk;

    uint32_t local_sum = 0, local_count = 0;
    int end = min(base_th_end, data_size);
    for (int i = base_th; i < end; i++) { local_sum += __ldg(&data[i]); local_count++; }
    #pragma unroll
    for (int off = 16; off > 0; off /= 2) {
        local_sum   += __shfl_down_sync(0xffffffff, local_sum,   off);
        local_count += __shfl_down_sync(0xffffffff, local_count, off);
    }
    int center = (local_count > 0) ? static_cast<int>(local_sum / local_count) : 0;
    center = __shfl_sync(0xffffffff, center, 0);

    uint8_t local_code[kChunk] = {0};
    uint8_t local_bits[kChunk * kMaxSignalBytesU32] = {0};
    int diff = 0, output_idx = 0, local_idx = 0, bit_offset = 0;
    uint32_t curr = 0;

    if (base_th <= data_size) {
        for (int i = base_th; i < end; i += 4) {
            uint4 tmp = reinterpret_cast<const uint4*>(data)[i / 4];
            #define ADM_ENCODE_U32(VAL) do {                                    \
                curr = static_cast<uint32_t>(VAL);                              \
                bool ic = (curr == static_cast<uint32_t>(center));              \
                diff = (curr > static_cast<uint32_t>(center))                   \
                       ? static_cast<int>(curr - center)                        \
                       : static_cast<int>(center - curr);                       \
                uint32_t rem = static_cast<uint32_t>(diff % 126);               \
                output_idx = ic ? 1                                             \
                           : (rem == 0) ? diff / 126 : diff / 126 + 1;         \
                uint8_t res = ic ? static_cast<uint8_t>(shift)                  \
                    : static_cast<uint8_t>(                                     \
                        (curr > static_cast<uint32_t>(center))                  \
                        ? (diff + 126 - output_idx * 126) * 2 - 1 + shift      \
                        : (diff + 126 - output_idx * 126) * 2 + shift);         \
                local_code[local_idx] = res;                                    \
                local_bits[bit_offset / 8] |= static_cast<uint8_t>(1 << (7 - (bit_offset % 8))); \
                local_idx++;                                                     \
                bit_offset += output_idx;                                       \
            } while(0)
            ADM_ENCODE_U32(tmp.x); ADM_ENCODE_U32(tmp.y);
            ADM_ENCODE_U32(tmp.z); ADM_ENCODE_U32(tmp.w);
            #undef ADM_ENCODE_U32
        }
    }

#ifndef NDEBUG
    if (bit_offset > kChunk * kMaxSignalBytesU32 * 8)
        atomicOr(d_overflow_flag, 1u);
#endif

    uint16_t max_bits = static_cast<uint16_t>(bit_offset);
    #pragma unroll
    for (int off = 16; off > 0; off /= 2)
        max_bits = max(max_bits, __shfl_down_sync(0xFFFFFFFF, max_bits, off));
    max_bits = static_cast<uint16_t>((max_bits + 7) / 8);
    max_bits = __shfl_sync(0xffffffff, max_bits, 0);

    if (lane == 0) { signal_length[blockIdx.x] = max_bits; centers[blockIdx.x] = static_cast<uint32_t>(center); }

    int total_bits = max_bits * 8;
    int cur_byte   = bit_offset / 8;
    uint8_t mask   = static_cast<uint8_t>(0xFF >> (bit_offset % 8));
    if (bit_offset < total_bits)
        local_bits[cur_byte] = (bit_offset % 8 == 0) ? 0xFF
                                                      : (local_bits[cur_byte] | mask);

    int write_pos = idx * kChunk * kMaxSignalBytesU32;
    int valid = (max_bits + 7) / 8;
    for (int i = 0; i < valid; i++)
        bit_signal[write_pos + i] = local_bits[i];

    int4* code_v = reinterpret_cast<int4*>(code + base_th);
    int4* lc_v   = reinterpret_cast<int4*>(local_code);
    #pragma unroll
    for (int i = 0; i < kChunk / 16; ++i) code_v[i] = lc_v[i];
}


__global__ static void adm_concat_u32(
    const uint8_t* d_bit_signals, const int* d_signal_length,
    const int* d_output_lengths, uint8_t* d_out, int gsize, int num_elements)
{
    int idx  = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = idx & 0x1f;
    int warp = idx >> 5;
    if (idx * kChunk >= num_elements) return;
    int length       = d_signal_length[warp];
    int bit_start    = idx * kChunk * kMaxSignalBytesU32;
    int concat_start = d_output_lengths[warp] * kBlockThreads + lane * length;
    for (int i = 0; i < length; i++)
        d_out[concat_start + i] = d_bit_signals[bit_start + i];
}


__global__ static void adm_map_decoupled_u32(
    const uint32_t* data, uint8_t* code, uint8_t* bit_signal,
    uint32_t* centers, int* signal_length, uint32_t* block_flags,
    volatile int* __restrict__ locOffset, volatile int* __restrict__ cmpOffset,
    volatile int* __restrict__ prefix_state, volatile int* __restrict__ blockResolved,
    int data_size, int shift, unsigned int* d_overflow_flag)
{
    (void)block_flags;
    // Two-level scan: each warp's total goes in s_warp_totals, the block leader
    // (warp_in_block==0, lane 0) does one decoupled look-back per BLOCK (not per
    // warp) into s_block_excl, then every warp adds its own intra-block exclusive
    // offset. See kWarpsPerBlock in adm_kernels.h for why.
    __shared__ int s_warp_totals[kWarpsPerBlock];
    __shared__ int s_warp_excl[kWarpsPerBlock];
    __shared__ int s_block_excl;

    const int bid  = blockIdx.x;
    const int idx  = bid * blockDim.x + threadIdx.x;
    const int lane = idx & 0x1f;
    const int warp = idx >> 5;             // global warp index -- still used for data slicing
    const int warp_in_block = threadIdx.x >> 5;
    const int block_num = kChunk >> 4;

    int base_block = warp * kChunk * kBlockThreads;
    uint8_t local_code[kChunk] = {0};
    uint8_t local_bits[kChunk * kMaxSignalBytesU32] = {0};
    int diff = 0, output_idx = 0, local_idx = 0, bit_offset = 0;
    uint32_t curr = 0;

    uint32_t local_sum = 0, local_count = 0;
    int base_th = base_block + lane * kChunk;
    int end = min(base_th + kChunk, data_size);
    for (int i = base_th; i < end; i++) { local_sum += __ldg(&data[i]); local_count++; }
    #pragma unroll
    for (int off = 16; off > 0; off /= 2) {
        local_sum   += __shfl_down_sync(0xffffffff, local_sum,   off);
        local_count += __shfl_down_sync(0xffffffff, local_count, off);
    }
    int center = (local_count > 0) ? static_cast<int>(local_sum / local_count) : 0;
    center = __shfl_sync(0xffffffff, center, 0);

    for (int j = 0; j < block_num; j++) {
        int bth = base_block + lane * kChunk + j * 16;
        end = min(bth + 16, data_size);
        if (bth > data_size) break;

        for (int i = bth; i < end; i += 4) {
            uint4 tmp = reinterpret_cast<const uint4*>(data)[i / 4];
            #define ADM_ENCODE_U32D(VAL) do {                                   \
                curr = static_cast<uint32_t>(VAL);                              \
                bool ic = (curr == static_cast<uint32_t>(center));              \
                diff = (curr > static_cast<uint32_t>(center))                   \
                       ? static_cast<int>(curr - center)                        \
                       : static_cast<int>(center - curr);                       \
                uint32_t rem = static_cast<uint32_t>(diff % 126);               \
                output_idx = ic ? 1                                             \
                           : (rem == 0) ? diff / 126 : diff / 126 + 1;         \
                uint8_t res = ic ? static_cast<uint8_t>(shift)                  \
                    : static_cast<uint8_t>(                                     \
                        (curr > static_cast<uint32_t>(center))                  \
                        ? (diff + 126 - output_idx * 126) * 2 - 1 + shift      \
                        : (diff + 126 - output_idx * 126) * 2 + shift);         \
                local_code[local_idx] = res;                                    \
                local_bits[bit_offset / 8] |= static_cast<uint8_t>(1 << (7 - (bit_offset % 8))); \
                local_idx++;                                                     \
                bit_offset += output_idx;                                       \
            } while(0)
            ADM_ENCODE_U32D(tmp.x); ADM_ENCODE_U32D(tmp.y);
            ADM_ENCODE_U32D(tmp.z); ADM_ENCODE_U32D(tmp.w);
            #undef ADM_ENCODE_U32D
        }
    }

#ifndef NDEBUG
    if (bit_offset > kChunk * kMaxSignalBytesU32 * 8)
        atomicOr(d_overflow_flag, 1u);
#endif

    uint16_t max_bits = static_cast<uint16_t>(bit_offset);
    #pragma unroll
    for (int off = 16; off > 0; off /= 2)
        max_bits = max(max_bits, __shfl_down_sync(0xFFFFFFFF, max_bits, off));
    max_bits = static_cast<uint16_t>((max_bits + 7) / 8);
    max_bits = __shfl_sync(0xffffffff, max_bits, 0);

    int total_bits = max_bits * 8;
    int cur_byte   = bit_offset / 8;
    uint8_t mask   = static_cast<uint8_t>(0xFF >> (bit_offset % 8));
    if (bit_offset < total_bits)
        local_bits[cur_byte] = (bit_offset % 8 == 0) ? 0xFF
                                                      : (local_bits[cur_byte] | mask);

    // signal_length/centers are read back per-warp on the host (and signal_length
    // is part of the archive's byte-accounting), so they stay warp-indexed even
    // though bid no longer equals warp.
    if (lane == 31) {
        signal_length[warp] = max_bits;
        centers[warp]       = static_cast<uint32_t>(center);
        s_warp_totals[warp_in_block] = static_cast<int>(max_bits);
    }
    __syncthreads();

    if (warp_in_block == 0 && lane == 0) {
        int running = 0;
        #pragma unroll
        for (int w = 0; w < kWarpsPerBlock; w++) {
            s_warp_excl[w] = running;
            running += s_warp_totals[w];
        }
        int block_total = running;

        if (bid == 0) {
            locOffset[1] = block_total;
            __threadfence();
            prefix_state[0] = 2;
            __threadfence();
            prefix_state[1] = 1;
            __threadfence();
            s_block_excl = 0;
        } else {
            locOffset[bid + 1] = block_total;
            __threadfence();
            prefix_state[bid + 1] = 1;
            __threadfence();

            int lookback = bid, loc_excl = 0;
            while (lookback > 0) {
                int status;
                do { status = prefix_state[lookback]; } while (status == 0);
                __threadfence();
                if (status == 2) { loc_excl += blockResolved[lookback]; __threadfence(); break; }
                if (status == 1)   loc_excl += locOffset[lookback];
                lookback--;
                __threadfence();
            }
            s_block_excl = loc_excl;

            blockResolved[bid] = s_block_excl;
            __threadfence();
            prefix_state[bid] = 2;
            __threadfence();
        }
    }
    __syncthreads();

    // cmpOffset (= d_output_lengths, part of the serialized archive) still needs
    // one entry per warp -- every warp writes its own global exclusive offset.
    int warp_excl_global = s_block_excl + s_warp_excl[warp_in_block];
    if (!lane) cmpOffset[warp] = warp_excl_global;

    int write_pos = warp_excl_global * kBlockThreads + lane * max_bits;
    for (int i = 0; i < max_bits; i++)
        bit_signal[write_pos + i] = local_bits[i];

    int4* code_v = reinterpret_cast<int4*>(code + base_block + lane * kChunk);
    int4* lc_v   = reinterpret_cast<int4*>(local_code);
    #pragma unroll
    for (int i = 0; i < kChunk / 16; ++i) code_v[i] = lc_v[i];
}


// ── Host wrappers ─────────────────────────────────────────────────────────────

size_t get_max_u32_payload_bytes(size_t n) {
    const size_t g = adm_gsize(n);
    return g * sizeof(int)
         + g * sizeof(uint32_t)
         + adm_flags_bytes(g)
         + n
         + n * kMaxSignalBytesU32;
}

void compress_u32(
    const uint32_t* d_input, size_t num_elements,
    uint8_t* d_output, size_t& output_size,
    const AdmScratch& s, cudaStream_t stream)
{
    if (num_elements == 0) { output_size = 0; return; }

    const int bsize    = kBlockThreads;
    const int gsize    = static_cast<int>(adm_gsize(num_elements));
    const bool use_dec = (gsize <= kDecoupledMaxGsize);

    FZ_CUDA_CHECK(cudaMemsetAsync(s.d_signal_length,  0,
        static_cast<size_t>(gsize) * sizeof(int), stream));
    FZ_CUDA_CHECK(cudaMemsetAsync(s.d_output_lengths, 0,
        static_cast<size_t>(gsize + 1) * sizeof(int), stream));
    FZ_CUDA_CHECK(cudaMemsetAsync(s.d_block_flags, 0xFF,
        adm_flags_words(gsize) * sizeof(uint32_t), stream));
    FZ_CUDA_CHECK(cudaMemsetAsync(s.d_concat_signals, 0xFF,
        static_cast<size_t>(num_elements) * kMaxSignalBytesU32, stream));

    if (use_dec) {
        const int num_blocks = static_cast<int>(adm_num_blocks(static_cast<size_t>(gsize)));

        FZ_CUDA_CHECK(cudaMemsetAsync(s.d_loc_offset,   0,
            static_cast<size_t>(gsize + 1) * sizeof(int), stream));
        FZ_CUDA_CHECK(cudaMemsetAsync(s.d_prefix_state, 0,
            static_cast<size_t>(gsize + 1) * sizeof(int), stream));
        FZ_CUDA_CHECK(cudaMemsetAsync(s.d_block_resolved, 0,
            static_cast<size_t>(num_blocks + 1) * sizeof(int), stream));
#ifndef NDEBUG
        FZ_CUDA_CHECK(cudaMemsetAsync(s.d_overflow_flag, 0, sizeof(unsigned int), stream));
#endif
        adm_map_decoupled_u32<<<num_blocks, kWarpsPerBlock * bsize, sizeof(unsigned int) * 2, stream>>>(
            d_input, s.d_codes, s.d_concat_signals,
            static_cast<uint32_t*>(s.d_centers),
            s.d_signal_length, s.d_block_flags,
            s.d_loc_offset, s.d_output_lengths, s.d_prefix_state, s.d_block_resolved,
            static_cast<int>(num_elements), kShift, s.d_overflow_flag);
        FZ_CUDA_CHECK(cudaGetLastError());
        FZ_CUDA_CHECK(cudaStreamSynchronize(stream));
#ifndef NDEBUG
        {
            unsigned int flag = 0;
            FZ_CUDA_CHECK(cudaMemcpy(&flag, s.d_overflow_flag,
                sizeof(unsigned int), cudaMemcpyDeviceToHost));
            if (flag)
                throw std::runtime_error(
                    "ADMStage (U32): local_bits overflow — input diffs exceed algorithm "
                    "capacity; ADM is designed for bounded quantization codes");
        }
#endif
    } else {
        FZ_CUDA_CHECK(cudaMemsetAsync(s.d_bit_signals, 0,
            static_cast<size_t>(num_elements) * kMaxSignalBytesU32, stream));
#ifndef NDEBUG
        FZ_CUDA_CHECK(cudaMemsetAsync(s.d_overflow_flag, 0, sizeof(unsigned int), stream));
#endif
        adm_map_thrust_u32<<<gsize, bsize, 0, stream>>>(
            d_input, s.d_codes, s.d_bit_signals,
            static_cast<uint32_t*>(s.d_centers),
            s.d_signal_length, s.d_block_flags,
            static_cast<int>(num_elements), kShift, s.d_overflow_flag);
        FZ_CUDA_CHECK(cudaGetLastError());
        FZ_CUDA_CHECK(cudaStreamSynchronize(stream));
#ifndef NDEBUG
        {
            unsigned int flag = 0;
            FZ_CUDA_CHECK(cudaMemcpy(&flag, s.d_overflow_flag,
                sizeof(unsigned int), cudaMemcpyDeviceToHost));
            if (flag)
                throw std::runtime_error(
                    "ADMStage (U32): local_bits overflow — input diffs exceed algorithm "
                    "capacity; ADM is designed for bounded quantization codes");
        }
#endif

        fz::backend::exclusiveScan(stream, s.d_signal_length, s.d_output_lengths,
                                    static_cast<size_t>(gsize));
        FZ_CUDA_CHECK(cudaGetLastError());
        FZ_CUDA_CHECK(cudaStreamSynchronize(stream));

        adm_concat_u32<<<gsize, bsize, 0, stream>>>(
            s.d_bit_signals, s.d_signal_length, s.d_output_lengths,
            s.d_concat_signals, gsize, static_cast<int>(num_elements));
        FZ_CUDA_CHECK(cudaGetLastError());
        FZ_CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    std::vector<int> signal_lengths(gsize), output_lengths_h(gsize, 0);
    FZ_CUDA_CHECK(cudaMemcpyAsync(signal_lengths.data(), s.d_signal_length,
        static_cast<size_t>(gsize) * sizeof(int), cudaMemcpyDeviceToHost, stream));
    FZ_CUDA_CHECK(cudaMemcpyAsync(output_lengths_h.data(), s.d_output_lengths,
        static_cast<size_t>(gsize) * sizeof(int), cudaMemcpyDeviceToHost, stream));
    FZ_CUDA_CHECK(cudaStreamSynchronize(stream));

    const size_t bit_signals_size =
        static_cast<size_t>(signal_lengths.back() + output_lengths_h.back()) * bsize;
    const size_t ofl_size   = static_cast<size_t>(gsize) * sizeof(int);
    const size_t ctr_size   = static_cast<size_t>(gsize) * sizeof(uint32_t);
    const size_t flags_size = adm_flags_bytes(gsize);
    const size_t codes_size = num_elements;
    output_size = ofl_size + ctr_size + flags_size + codes_size + bit_signals_size;

    FZ_CUDA_CHECK(cudaMemcpyAsync(d_output,
        s.d_output_lengths, ofl_size, cudaMemcpyDeviceToDevice, stream));
    FZ_CUDA_CHECK(cudaMemcpyAsync(d_output + ofl_size,
        s.d_centers, ctr_size, cudaMemcpyDeviceToDevice, stream));
    FZ_CUDA_CHECK(cudaMemcpyAsync(d_output + ofl_size + ctr_size,
        s.d_block_flags, flags_size, cudaMemcpyDeviceToDevice, stream));
    FZ_CUDA_CHECK(cudaMemcpyAsync(d_output + ofl_size + ctr_size + flags_size,
        s.d_codes, codes_size, cudaMemcpyDeviceToDevice, stream));
    FZ_CUDA_CHECK(cudaMemcpyAsync(
        d_output + ofl_size + ctr_size + flags_size + codes_size,
        s.d_concat_signals, bit_signals_size, cudaMemcpyDeviceToDevice, stream));
    FZ_CUDA_CHECK(cudaStreamSynchronize(stream));
}

void decompress_u32(
    const uint8_t* d_input, size_t input_size,
    uint32_t* d_output, size_t num_elements,
    const AdmScratch& s, cudaStream_t stream)
{
    if (num_elements == 0) return;

    const int bsize   = kBlockThreads;
    const int gsize   = static_cast<int>(adm_gsize(num_elements));
    const size_t ofl_size   = static_cast<size_t>(gsize) * sizeof(int);
    const size_t ctr_size   = static_cast<size_t>(gsize) * sizeof(uint32_t);
    const size_t flags_size = adm_flags_bytes(gsize);
    const size_t codes_size = num_elements;
    const size_t fixed_size = ofl_size + ctr_size + flags_size + codes_size;

    if (input_size < fixed_size)
        throw std::runtime_error("ADMStage: truncated payload in decompress_u32");

    std::vector<int> ofl_h(gsize);
    FZ_CUDA_CHECK(cudaMemcpyAsync(ofl_h.data(), d_input,
        ofl_size, cudaMemcpyDeviceToHost, stream));
    FZ_CUDA_CHECK(cudaStreamSynchronize(stream));

    const size_t bit_signals_size = input_size - fixed_size;
    if (bit_signals_size % bsize != 0)
        throw std::runtime_error("ADMStage: invalid bit-signal size in decompress_u32");
    const int last_length = static_cast<int>(bit_signals_size / bsize) - ofl_h.back();
    if (last_length < 0)
        throw std::runtime_error("ADMStage: negative last_length in decompress_u32");

    const uint8_t* ctr_ptr   = d_input + ofl_size;
    const uint8_t* codes_ptr = ctr_ptr + ctr_size + flags_size;
    const uint8_t* bits_ptr  = codes_ptr + codes_size;

    FZ_CUDA_CHECK(cudaMemcpyAsync(s.d_output_lengths, d_input,
        ofl_size, cudaMemcpyDeviceToDevice, stream));
    FZ_CUDA_CHECK(cudaMemcpyAsync(s.d_centers, ctr_ptr,
        ctr_size, cudaMemcpyDeviceToDevice, stream));
    FZ_CUDA_CHECK(cudaMemcpyAsync(s.d_codes, codes_ptr,
        codes_size, cudaMemcpyDeviceToDevice, stream));
    FZ_CUDA_CHECK(cudaMemcpyAsync(s.d_concat_signals, bits_ptr,
        bit_signals_size, cudaMemcpyDeviceToDevice, stream));

    const int decmp_g = static_cast<int>(
        (num_elements + bsize * kDecmpChunk - 1) / (bsize * kDecmpChunk));

    adm_decompress_u32<<<decmp_g, bsize, 0, stream>>>(
        d_output,
        static_cast<uint32_t*>(s.d_centers),
        s.d_codes,
        s.d_output_lengths,
        s.d_concat_signals,
        static_cast<int>(num_elements),
        last_length,
        gsize,
        kShift);
    FZ_CUDA_CHECK(cudaGetLastError());
    FZ_CUDA_CHECK(cudaStreamSynchronize(stream));
}

} // namespace adm
} // namespace fz
