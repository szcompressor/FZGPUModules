// Adapted from the MANS project (Huang et al.), BSD-3-Clause License — see THIRD_PARTY.md.
// Original: nv/adm/mapping_uint16.cu in https://github.com/hpdps-group/MANS
//
// Changes from original:
//   - Removed MansParams parameter (was unused in all functions).
//   - Replaced per-call cudaMalloc/cudaFree with pool-allocated scratch (AdmScratch).
//   - Replaced check_cuda() with FZ_CUDA_CHECK.
//   - Renamed namespace from mans::nv::adm to fz::adm.
//   - Renamed kernels with _u16 suffix to avoid TU-level naming conflicts.
//   - Removed unused d_signals parameter from decompress kernel.
//   - Translated inline Chinese comments to English.
//   - Kernel logic is unchanged from the original.

#include "transforms/adm/adm_kernels.h"
#include "cuda_check.h"
#include <cuda_runtime.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace fz {
namespace adm {

// ── Kernel implementations ────────────────────────────────────────────────────
// These are TU-private — only the host wrappers at the bottom are exported.

__global__ static void adm_decompress_u16(
    uint16_t* decmp_data, uint16_t* centers, uint8_t* codes,
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

    // Decode first-half signals (elements 0..kChunk-1).
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

    // Decode second-half signals (elements kChunk..kDecmpChunk-1).
    offset_byte = 0;
    signal_idx  = 15;
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
    int4*  local_codes_v = reinterpret_cast<int4*>(local_codes);
    int4*  codes_v       = reinterpret_cast<int4*>(codes + dst_start);
    #pragma unroll
    for (int i = 0; i < kDecmpChunk / 16; ++i)
        local_codes_v[i] = codes_v[i];

    uint16_t local_result[kDecmpChunk];
    int center = lane < 16 ? centers[blockIdx.x * 2] : centers[blockIdx.x * 2 + 1];

    for (int i = 0; i < kDecmpChunk; i++) {
        uint8_t code   = local_codes[i];
        uint8_t signal = local_signal[i];
        int diff = (code % 2 == 1) ? (code - 1) / 2 : code / 2;
        diff += signal * 126;
        local_result[i] = static_cast<uint16_t>(
            (code % 2 == 1) ? center - diff : center + diff);
    }

    // Vectorised int4 stores cover kDecmpChunk uint16 elements (64 B / thread).
    // The last thread on an unaligned input has `dst_start + kDecmpChunk >
    // data_size` — its vector store would write past the user output buffer.
    // Fall back to a scalar store for that partial chunk; the fast path covers
    // the common aligned case.
    if (dst_start + kDecmpChunk <= data_size) {
        int4* result_v = reinterpret_cast<int4*>(local_result);
        int4* out_v    = reinterpret_cast<int4*>(decmp_data + dst_start);
        #pragma unroll
        for (int i = 0; i < kDecmpChunk / 8; ++i)
            out_v[i] = result_v[i];
    } else {
        int tail_end = min(dst_start + kDecmpChunk, data_size);
        for (int i = dst_start; i < tail_end; ++i)
            decmp_data[i] = local_result[i - dst_start];
    }
}


__global__ static void adm_map_thrust_u16(
    const uint16_t* data, uint8_t* code, uint8_t* bit_signal,
    uint16_t* centers, int* signal_length, uint32_t* block_flags,
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
    for (int i = base_th; i < end; i++) {
        local_sum += __ldg(&data[i]);
        local_count++;
    }
    #pragma unroll
    for (int off = 16; off > 0; off /= 2) {
        local_sum   += __shfl_down_sync(0xffffffff, local_sum,   off);
        local_count += __shfl_down_sync(0xffffffff, local_count, off);
    }
    int center = (local_count > 0) ? static_cast<int>(local_sum / local_count) : 0;
    center = __shfl_sync(0xffffffff, center, 0);

    uint8_t local_code[kChunk] = {0};
    uint8_t local_bits[kChunk * kMaxSignalBytesU16] = {0};
    int diff = 0, output_idx = 0, local_idx = 0, bit_offset = 0;
    uint16_t curr = 0;
    uint4 tmp;

    #pragma unroll
    for (int i = base_th; i < end; i += 8) {
        tmp = reinterpret_cast<const uint4*>(data)[i / 8];
        #define ADM_ENCODE_U16(VAL) do {                                    \
            curr = static_cast<uint16_t>(VAL);                              \
            bool ic = (curr == static_cast<uint16_t>(center));              \
            diff = (curr > center) ? curr - center : center - curr;         \
            uint16_t rem = static_cast<uint16_t>(diff % 126);               \
            output_idx = ic ? 1                                             \
                       : (rem == 0) ? diff / 126 : diff / 126 + 1;         \
            uint8_t res = ic ? static_cast<uint8_t>(shift)                  \
                : static_cast<uint8_t>(                                     \
                    (curr > center)                                         \
                    ? (diff + 126 - output_idx * 126) * 2 - 1 + shift      \
                    : (diff + 126 - output_idx * 126) * 2 + shift);         \
            local_code[local_idx] = res;                                    \
            local_bits[bit_offset / 8] |= static_cast<uint8_t>(1 << (7 - (bit_offset % 8))); \
            local_idx++;                                                     \
            bit_offset += output_idx;                                       \
        } while(0)

        ADM_ENCODE_U16(tmp.x & 0xFFFF);
        ADM_ENCODE_U16((tmp.x >> 16) & 0xFFFF);
        ADM_ENCODE_U16(tmp.y & 0xFFFF);
        ADM_ENCODE_U16((tmp.y >> 16) & 0xFFFF);
        ADM_ENCODE_U16(tmp.z & 0xFFFF);
        ADM_ENCODE_U16((tmp.z >> 16) & 0xFFFF);
        ADM_ENCODE_U16(tmp.w & 0xFFFF);
        ADM_ENCODE_U16((tmp.w >> 16) & 0xFFFF);
        #undef ADM_ENCODE_U16
    }

#ifndef NDEBUG
    if (bit_offset > kChunk * kMaxSignalBytesU16 * 8)
        atomicOr(d_overflow_flag, 1u);
#endif

    uint16_t max_bits = static_cast<uint16_t>(bit_offset);
    #pragma unroll
    for (int off = 16; off > 0; off /= 2)
        max_bits = max(max_bits, __shfl_down_sync(0xFFFFFFFF, max_bits, off));
    max_bits = static_cast<uint16_t>((max_bits + 7) / 8);
    max_bits = __shfl_sync(0xffffffff, max_bits, 0);

    if (lane == 0) { signal_length[blockIdx.x] = max_bits; centers[blockIdx.x] = static_cast<uint16_t>(center); }

    int total_bits = max_bits * 8;
    int cur_byte   = bit_offset / 8;
    uint8_t mask   = static_cast<uint8_t>(0xFF >> (bit_offset % 8));
    if (bit_offset < total_bits)
        local_bits[cur_byte] = (bit_offset % 8 == 0) ? 0xFF
                                                      : (local_bits[cur_byte] | mask);

    int write_pos = idx * kChunk * kMaxSignalBytesU16;
    int2* dst_v = reinterpret_cast<int2*>(bit_signal + write_pos);
    int2* src_v = reinterpret_cast<int2*>(local_bits);
    int valid = (max_bits + 7) / 8;
    #pragma unroll
    for (int i = 0; i < valid; i++) dst_v[i] = src_v[i];

    int4* code_v = reinterpret_cast<int4*>(code + base_th);
    int4* lc_v   = reinterpret_cast<int4*>(local_code);
    #pragma unroll
    for (int i = 0; i < kChunk / 16; ++i) code_v[i] = lc_v[i];
}


__global__ static void adm_concat_u16(
    const uint8_t* d_bit_signals, const int* d_signal_length,
    const int* d_output_lengths, uint8_t* d_out, int gsize, int num_elements)
{
    int idx  = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = idx & 0x1f;
    int warp = idx >> 5;
    if (idx * kChunk >= num_elements) return;
    int length     = d_signal_length[warp];
    int bit_start  = idx * kChunk * kMaxSignalBytesU16;
    int concat_start = d_output_lengths[warp] * kBlockThreads + lane * length;
    for (int i = 0; i < length; i++)
        d_out[concat_start + i] = d_bit_signals[bit_start + i];
}


__global__ static void adm_map_decoupled_u16(
    const uint16_t* data, uint8_t* code, uint8_t* bit_signal,
    uint16_t* centers, int* signal_length, uint32_t* block_flags,
    volatile int* __restrict__ locOffset, volatile int* __restrict__ cmpOffset,
    volatile int* __restrict__ prefix_state,
    int data_size, int shift, unsigned int* d_overflow_flag)
{
    (void)block_flags;
    __shared__ unsigned int excl_sum;
    // warp=0 blocks never enter the lookback path, so initialize excl_sum=0
    // (correct exclusive prefix sum for the first warp block).
    if (!threadIdx.x) excl_sum = 0u;
    __syncthreads();

    const int tid  = threadIdx.x;
    const int bid  = blockIdx.x;
    const int idx  = bid * blockDim.x + tid;
    const int lane = idx & 0x1f;
    const int warp = idx >> 5;
    const int block_num = kChunk >> 4;  // kChunk / k = 16 / 4 = 1

    int base_block = warp * kChunk * kBlockThreads;
    uint4 tmp;

    uint8_t local_code[kChunk] = {0};
    uint8_t local_bits[kChunk * kMaxSignalBytesU16] = {0};
    int diff = 0, output_idx = 0, local_idx = 0, bit_offset = 0;
    uint16_t curr = 0;

    // Compute warp center.
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

    // Encode elements in kChunk / k sub-blocks.
    for (int j = 0; j < block_num; j++) {
        int bth = base_block + lane * kChunk + j * 16;
        end = min(bth + 16, data_size);
        if (bth > data_size) break;

        #pragma unroll
        for (int i = bth; i < end; i += 8) {
            tmp = reinterpret_cast<const uint4*>(data)[i / 8];
            #define ADM_ENCODE_U16D(VAL) do {                                   \
                curr = static_cast<uint16_t>(VAL);                              \
                bool ic = (curr == static_cast<uint16_t>(center));              \
                diff = (curr > center) ? curr - center : center - curr;         \
                uint16_t rem = static_cast<uint16_t>(diff % 126);               \
                output_idx = ic ? 1                                             \
                           : (rem == 0) ? diff / 126 : diff / 126 + 1;         \
                uint8_t res = ic ? static_cast<uint8_t>(shift)                  \
                    : static_cast<uint8_t>(                                     \
                        (curr > center)                                         \
                        ? (diff + 126 - output_idx * 126) * 2 - 1 + shift      \
                        : (diff + 126 - output_idx * 126) * 2 + shift);         \
                local_code[local_idx] = res;                                    \
                local_bits[bit_offset / 8] |= static_cast<uint8_t>(1 << (7 - (bit_offset % 8))); \
                local_idx++;                                                     \
                bit_offset += output_idx;                                       \
            } while(0)

            ADM_ENCODE_U16D(tmp.x & 0xFFFF);
            ADM_ENCODE_U16D((tmp.x >> 16) & 0xFFFF);
            ADM_ENCODE_U16D(tmp.y & 0xFFFF);
            ADM_ENCODE_U16D((tmp.y >> 16) & 0xFFFF);
            ADM_ENCODE_U16D(tmp.z & 0xFFFF);
            ADM_ENCODE_U16D((tmp.z >> 16) & 0xFFFF);
            ADM_ENCODE_U16D(tmp.w & 0xFFFF);
            ADM_ENCODE_U16D((tmp.w >> 16) & 0xFFFF);
            #undef ADM_ENCODE_U16D
        }
    }

#ifndef NDEBUG
    if (bit_offset > kChunk * kMaxSignalBytesU16 * 8)
        atomicOr(d_overflow_flag, 1u);
#endif

    // Compute warp-level max signal length.
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

    // Decoupled look-back prefix sum (writes signals directly to their final position).
    if (lane == 31) {
        signal_length[bid] = max_bits;
        centers[bid]       = static_cast<uint16_t>(center);
        locOffset[warp + 1] = max_bits;
        __threadfence();
        if (warp == 0) { prefix_state[0] = 2; __threadfence(); prefix_state[1] = 1; __threadfence(); }
        else           { prefix_state[warp + 1] = 1; __threadfence(); }
    }
    __syncthreads();

    if (warp > 0) {
        if (!lane) {
            int lookback = warp, loc_excl = 0;
            while (lookback > 0) {
                int status;
                do { status = prefix_state[lookback]; __threadfence(); } while (status == 0);
                if (status == 2) { loc_excl += cmpOffset[lookback]; __threadfence(); break; }
                if (status == 1)   loc_excl += locOffset[lookback];
                lookback--;
                __threadfence();
            }
            excl_sum = static_cast<unsigned int>(loc_excl);
        }
        __syncthreads();
    }

    if (warp > 0) {
        if (!lane) cmpOffset[warp] = static_cast<int>(excl_sum);
        __threadfence();
        if (!lane) prefix_state[warp] = 2;
        __threadfence();
    }
    __syncthreads();

    int write_pos = static_cast<int>(excl_sum) * kBlockThreads + lane * max_bits;
    #pragma unroll
    for (int i = 0; i < max_bits; i++)
        bit_signal[write_pos + i] = local_bits[i];

    int4* code_v = reinterpret_cast<int4*>(code + base_block + lane * kChunk);
    int4* lc_v   = reinterpret_cast<int4*>(local_code);
    #pragma unroll
    for (int i = 0; i < kChunk / 16; ++i) code_v[i] = lc_v[i];
}


// ── Host wrappers ─────────────────────────────────────────────────────────────

size_t get_max_u16_payload_bytes(size_t n) {
    const size_t g = adm_gsize(n);
    return g * sizeof(int)              // output_lengths
         + g * sizeof(uint16_t)         // centers
         + adm_flags_bytes(g)           // block_flags
         + n                            // codes (1 byte each)
         + n * kMaxSignalBytesU16;      // max signal bytes
}

void compress_u16(
    const uint16_t* d_input, size_t num_elements,
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
        static_cast<size_t>(num_elements) * kMaxSignalBytesU16, stream));

    if (use_dec) {
        FZ_CUDA_CHECK(cudaMemsetAsync(s.d_loc_offset,   0,
            static_cast<size_t>(gsize + 1) * sizeof(int), stream));
        FZ_CUDA_CHECK(cudaMemsetAsync(s.d_prefix_state, 0,
            static_cast<size_t>(gsize + 1) * sizeof(int), stream));
#ifndef NDEBUG
        FZ_CUDA_CHECK(cudaMemsetAsync(s.d_overflow_flag, 0, sizeof(unsigned int), stream));
#endif
        adm_map_decoupled_u16<<<gsize, bsize, sizeof(unsigned int) * 2, stream>>>(
            d_input, s.d_codes, s.d_concat_signals,
            static_cast<uint16_t*>(s.d_centers),
            s.d_signal_length, s.d_block_flags,
            s.d_loc_offset, s.d_output_lengths, s.d_prefix_state,
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
                    "ADMStage (U16): local_bits overflow — input diffs exceed algorithm "
                    "capacity; ADM is designed for bounded quantization codes");
        }
#endif
    } else {
        FZ_CUDA_CHECK(cudaMemsetAsync(s.d_bit_signals, 0,
            static_cast<size_t>(num_elements) * kMaxSignalBytesU16, stream));
#ifndef NDEBUG
        FZ_CUDA_CHECK(cudaMemsetAsync(s.d_overflow_flag, 0, sizeof(unsigned int), stream));
#endif
        adm_map_thrust_u16<<<gsize, bsize, 0, stream>>>(
            d_input, s.d_codes, s.d_bit_signals,
            static_cast<uint16_t*>(s.d_centers),
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
                    "ADMStage (U16): local_bits overflow — input diffs exceed algorithm "
                    "capacity; ADM is designed for bounded quantization codes");
        }
#endif

        thrust::device_ptr<int> dev_sig(s.d_signal_length);
        thrust::device_ptr<int> dev_out(s.d_output_lengths);
        thrust::exclusive_scan(thrust::cuda::par.on(stream),
            dev_sig, dev_sig + gsize, dev_out);
        FZ_CUDA_CHECK(cudaGetLastError());
        FZ_CUDA_CHECK(cudaStreamSynchronize(stream));

        adm_concat_u16<<<gsize, bsize, 0, stream>>>(
            s.d_bit_signals, s.d_signal_length, s.d_output_lengths,
            s.d_concat_signals, gsize, static_cast<int>(num_elements));
        FZ_CUDA_CHECK(cudaGetLastError());
        FZ_CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    // D2H: read signal_lengths and output_lengths to compute total payload size.
    std::vector<int> signal_lengths(gsize), output_lengths_h(gsize, 0);
    FZ_CUDA_CHECK(cudaMemcpyAsync(signal_lengths.data(), s.d_signal_length,
        static_cast<size_t>(gsize) * sizeof(int), cudaMemcpyDeviceToHost, stream));
    FZ_CUDA_CHECK(cudaMemcpyAsync(output_lengths_h.data(), s.d_output_lengths,
        static_cast<size_t>(gsize) * sizeof(int), cudaMemcpyDeviceToHost, stream));
    FZ_CUDA_CHECK(cudaStreamSynchronize(stream));

    const size_t bit_signals_size =
        static_cast<size_t>(signal_lengths.back() + output_lengths_h.back()) * bsize;
    const size_t ofl_size    = static_cast<size_t>(gsize) * sizeof(int);
    const size_t ctr_size    = static_cast<size_t>(gsize) * sizeof(uint16_t);
    const size_t flags_size  = adm_flags_bytes(gsize);
    const size_t codes_size  = num_elements;
    output_size = ofl_size + ctr_size + flags_size + codes_size + bit_signals_size;

    // Pack the output payload (all D2D copies).
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

void decompress_u16(
    const uint8_t* d_input, size_t input_size,
    uint16_t* d_output, size_t num_elements,
    const AdmScratch& s, cudaStream_t stream)
{
    if (num_elements == 0) return;

    const int bsize   = kBlockThreads;
    const int gsize   = static_cast<int>(adm_gsize(num_elements));
    const size_t ofl_size   = static_cast<size_t>(gsize) * sizeof(int);
    const size_t ctr_size   = static_cast<size_t>(gsize) * sizeof(uint16_t);
    const size_t flags_size = adm_flags_bytes(gsize);
    const size_t codes_size = num_elements;
    const size_t fixed_size = ofl_size + ctr_size + flags_size + codes_size;

    if (input_size < fixed_size)
        throw std::runtime_error("ADMStage: truncated payload in decompress_u16");

    std::vector<int> ofl_h(gsize);
    FZ_CUDA_CHECK(cudaMemcpyAsync(ofl_h.data(), d_input,
        ofl_size, cudaMemcpyDeviceToHost, stream));
    FZ_CUDA_CHECK(cudaStreamSynchronize(stream));

    const size_t bit_signals_size = input_size - fixed_size;
    if (bit_signals_size % bsize != 0)
        throw std::runtime_error("ADMStage: invalid bit-signal size in decompress_u16");
    const int last_length = static_cast<int>(bit_signals_size / bsize) - ofl_h.back();
    if (last_length < 0)
        throw std::runtime_error("ADMStage: negative last_length in decompress_u16");

    const uint8_t* ctr_ptr = d_input + ofl_size;
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

    adm_decompress_u16<<<decmp_g, bsize, 0, stream>>>(
        d_output,
        static_cast<uint16_t*>(s.d_centers),
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
