/**
 * modules/coders/clog/clog_stage.cu
 *
 * GPU implementation of CLOGStage — per-subchunk leading-zero compression
 * and adaptive bit-width packing.
 *
 * One CUDA block per chunk, blockDim = 512 threads (CLOG_TPB). Each block holds
 * the chunk in shared memory (in / out / temp) and runs the vendored LC
 * `d_CLOG<T>` / `d_iCLOG<T>` device functions on it. Inter-chunk packing offsets
 * are computed with a CUB exclusive scan, identical to RREStage — the host-side
 * container format doesn't know or care about CLOG's internal header layout
 * (a `short` original-size tag followed by the 32-entry bit-width table, all
 * inside the chunk's own compressed bytes, accounted for by `csize`).
 *
 * Algorithm reference: d_CLOG.h from the LC framework (Burtscher et al.,
 * BSD-3-Clause). See lc_clog_components.cuh and THIRD_PARTY.md.
 */

#include "coders/clog/clog_stage.h"
#include "stage/stage_registry.h"
#include "coders/lc_common/lc_clog_components.cuh"
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

static constexpr int CLOG_TPB  = 512;    // threads per block (must match lc_clog_components.cuh TPB)
static constexpr int CLOG_TEMP = 256;    // shared bytes for the 32-subchunk ln[]/bits[]/total_bits scratch
                                          // (32 + 32*4 + 4 = 164 bytes needed; padded for alignment)

using fz::lc_detail::byte;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Encode kernel — one block per chunk. Runs d_CLOG<T,CS> on the chunk in shared
// memory; falls back to verbatim storage (high-bit flag) when CLOG fails or
// fails to shrink the chunk. CS is compile-time; see launchEncode for the
// supported instantiations.
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
template <typename T, int CS>
static __global__ void __launch_bounds__(CLOG_TPB)
clogEncodeKernel(
    const byte* __restrict__ d_in,
    byte*       __restrict__ d_scratch,
    uint32_t*   __restrict__ d_sizes,
    uint32_t total_in_bytes)
{
    __shared__ __align__(16) byte s_in[CS];
    __shared__ __align__(16) byte s_out[CS];
    __shared__ __align__(16) byte s_temp[CLOG_TEMP];

    const uint32_t cid     = blockIdx.x;
    const uint32_t in_off  = cid * (uint32_t)CS;
    const int      in_size = static_cast<int>(
        min((uint32_t)CS, total_in_bytes - in_off));

    for (int i = threadIdx.x; i < CS; i += CLOG_TPB)
        s_in[i] = (i < in_size) ? d_in[in_off + i] : (byte)0;
    __syncthreads();

    int  csize = in_size;
    bool good  = lc_detail::d_CLOG<T, CS>(csize, s_in, s_out, s_temp);
    __syncthreads();

    byte* out = d_scratch + (size_t)cid * (uint32_t)CS;

    if (good && csize < in_size) {
        for (int i = threadIdx.x; i < csize; i += CLOG_TPB)
            out[i] = s_out[i];
        if (threadIdx.x == 0)
            d_sizes[cid] = (uint32_t)csize;
    } else {
        // Incompressible: store the original chunk verbatim (s_in is untouched
        // by d_CLOG, which only writes s_out / s_temp).
        for (int i = threadIdx.x; i < in_size; i += CLOG_TPB)
            out[i] = s_in[i];
        if (threadIdx.x == 0)
            d_sizes[cid] = (1u << 31) | (uint32_t)in_size;
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Decode kernel — one block per (compressed) chunk. Raw chunks (comp_size == 0)
// are copied by the host and skipped here.
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
template <typename T, int CS>
static __global__ void __launch_bounds__(CLOG_TPB)
clogDecodeKernel(
    const byte*     __restrict__ d_in,
    byte*           __restrict__ d_out,
    const uint32_t* __restrict__ d_in_offsets,
    const uint32_t* __restrict__ d_comp_sizes,
    const uint32_t* __restrict__ d_out_offsets,
    const uint32_t* __restrict__ d_orig_sizes)
{
    __shared__ __align__(16) byte s_in[CS];
    __shared__ __align__(16) byte s_out[CS];
    __shared__ __align__(16) byte s_temp[CLOG_TEMP];

    const uint32_t cid       = blockIdx.x;
    const uint32_t comp_size = d_comp_sizes[cid];
    const uint32_t orig_size = d_orig_sizes[cid];

    if (comp_size == 0) return;  // raw passthrough; copied by host

    const byte* in  = d_in  + d_in_offsets[cid];
    byte*       out = d_out + d_out_offsets[cid];

    for (int i = threadIdx.x; i < (int)comp_size; i += CLOG_TPB)
        s_in[i] = in[i];
    __syncthreads();

    int csize = (int)comp_size;
    lc_detail::d_iCLOG<T, CS>(csize, s_in, s_out, s_temp);  // sets csize = orig_size
    __syncthreads();

    for (int i = threadIdx.x; i < (int)orig_size; i += CLOG_TPB)
        out[i] = s_out[i];
}

// ━━━━ helper kernels (word-size independent; mirror RRE/RZE/RARE/RAZEStage) ━━━━
static __global__ void clogStripFlagKernel(
    const uint32_t* __restrict__ d_sizes_with_flag,
    uint32_t*       __restrict__ d_clean_sizes,
    uint32_t n_chunks)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_chunks)
        d_clean_sizes[i] = d_sizes_with_flag[i] & 0x7FFFFFFFu;
}

static __global__ void clogAddOffsetKernel(
    uint32_t* __restrict__ arr, uint32_t n, uint32_t offset)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) arr[i] += offset;
}

static __global__ void clogPackKernel(
    const byte*     __restrict__ d_scratch,
    byte*           __restrict__ d_out,
    const uint32_t* __restrict__ d_dst_offsets,
    const uint32_t* __restrict__ d_sizes,
    uint32_t CS)
{
    const uint32_t cid     = blockIdx.x;
    const uint32_t src_off = cid * CS;
    const uint32_t dst_off = d_dst_offsets[cid];
    const uint32_t sz      = d_sizes[cid];
    const byte* src = d_scratch + src_off;
    byte*       dst = d_out     + dst_off;
    for (uint32_t i = threadIdx.x; i < sz; i += blockDim.x)
        dst[i] = src[i];
}

// ━━━━ word-size × chunk-size dispatch helpers ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// chunk_size selects the compile-time CS template argument; word_size selects
// T. Supported chunk sizes: 4096, 8192, 16384 (see CLOGStage::execute()).
static void launchEncode(uint8_t word_size, uint32_t chunk_size, int n_chunks, cudaStream_t stream,
                         const byte* d_in, byte* d_scratch,
                         uint32_t* d_sizes, uint32_t in_bytes)
{
#define FZ_CLOG_ENCODE_CASE(CS_VAL)                                                                          \
    case CS_VAL:                                                                                             \
        switch (word_size) {                                                                                 \
            case 1: clogEncodeKernel<uint8_t,  CS_VAL><<<n_chunks, CLOG_TPB, 0, stream>>>(d_in, d_scratch, d_sizes, in_bytes); break; \
            case 2: clogEncodeKernel<uint16_t, CS_VAL><<<n_chunks, CLOG_TPB, 0, stream>>>(d_in, d_scratch, d_sizes, in_bytes); break; \
            case 4: clogEncodeKernel<uint32_t, CS_VAL><<<n_chunks, CLOG_TPB, 0, stream>>>(d_in, d_scratch, d_sizes, in_bytes); break; \
            case 8: clogEncodeKernel<uint64_t, CS_VAL><<<n_chunks, CLOG_TPB, 0, stream>>>(d_in, d_scratch, d_sizes, in_bytes); break; \
            default: throw std::runtime_error("CLOGStage: word_size must be 1, 2, 4, or 8");                 \
        }                                                                                                     \
        break;
    switch (chunk_size) {
        FZ_CLOG_ENCODE_CASE(4096)
        FZ_CLOG_ENCODE_CASE(8192)
        FZ_CLOG_ENCODE_CASE(16384)
        default: throw std::runtime_error("CLOGStage: chunk_size must be 4096, 8192, or 16384");
    }
#undef FZ_CLOG_ENCODE_CASE
}

static void launchDecode(uint8_t word_size, uint32_t chunk_size, int n_chunks, cudaStream_t stream,
                         const byte* d_in, byte* d_out,
                         const uint32_t* in_off, const uint32_t* comp_sz,
                         const uint32_t* out_off, const uint32_t* orig_sz)
{
#define FZ_CLOG_DECODE_CASE(CS_VAL)                                                                          \
    case CS_VAL:                                                                                             \
        switch (word_size) {                                                                                 \
            case 1: clogDecodeKernel<uint8_t,  CS_VAL><<<n_chunks, CLOG_TPB, 0, stream>>>(d_in, d_out, in_off, comp_sz, out_off, orig_sz); break; \
            case 2: clogDecodeKernel<uint16_t, CS_VAL><<<n_chunks, CLOG_TPB, 0, stream>>>(d_in, d_out, in_off, comp_sz, out_off, orig_sz); break; \
            case 4: clogDecodeKernel<uint32_t, CS_VAL><<<n_chunks, CLOG_TPB, 0, stream>>>(d_in, d_out, in_off, comp_sz, out_off, orig_sz); break; \
            case 8: clogDecodeKernel<uint64_t, CS_VAL><<<n_chunks, CLOG_TPB, 0, stream>>>(d_in, d_out, in_off, comp_sz, out_off, orig_sz); break; \
            default: throw std::runtime_error("CLOGStage: word_size must be 1, 2, 4, or 8");                 \
        }                                                                                                     \
        break;
    switch (chunk_size) {
        FZ_CLOG_DECODE_CASE(4096)
        FZ_CLOG_DECODE_CASE(8192)
        FZ_CLOG_DECODE_CASE(16384)
        default: throw std::runtime_error("CLOGStage: chunk_size must be 4096, 8192, or 16384");
    }
#undef FZ_CLOG_DECODE_CASE
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CLOGStage::~CLOGStage() {
    auto fwd_free = [&](void* p) {
        if (!p) return;
        if (scratch_from_pool_ && scratch_pool_owner_ && !scratch_alive_.expired())
            scratch_pool_owner_->free(p, 0);
        else if (!scratch_from_pool_) cudaFree(p);
    };
    fwd_free(d_scratch_);
    fwd_free(d_sizes_dev_);
    fwd_free(d_clean_dev_);
    fwd_free(d_dst_off_dev_);
}

void CLOGStage::postStreamSync(cudaStream_t stream) {
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
    FZ_LOG(DEBUG, "CLOG encode done: %.1f KB -> %.1f KB  ratio %.2fx",
           cached_orig_bytes_ / 1024.0f, actual_output_size_ / 1024.0f, ratio);
}

std::unordered_map<std::string, size_t>
CLOGStage::getActualOutputSizesByName() const {
    if (tail_readback_pending_) {
        uint32_t tail_off = 0, tail_sz = 0;
        FZ_CUDA_CHECK(cudaMemcpyAsync(&tail_off, d_dst_off_dev_ + tail_last_index_,
                                      sizeof(uint32_t), cudaMemcpyDeviceToHost, tail_readback_stream_));
        FZ_CUDA_CHECK(cudaMemcpyAsync(&tail_sz, d_clean_dev_ + tail_last_index_,
                                      sizeof(uint32_t), cudaMemcpyDeviceToHost, tail_readback_stream_));
        FZ_CUDA_CHECK(cudaStreamSynchronize(tail_readback_stream_));
        const size_t total_out = (size_t)tail_off + (size_t)tail_sz;
        const_cast<CLOGStage*>(this)->actual_output_size_ = (total_out + 3) & ~size_t(3);
        if (tail_output_ptr_ && actual_output_size_ > total_out) {
            FZ_CUDA_CHECK(cudaMemsetAsync(tail_output_ptr_ + total_out, 0,
                                          actual_output_size_ - total_out, tail_readback_stream_));
            FZ_CUDA_CHECK(cudaStreamSynchronize(tail_readback_stream_));
        }
        const_cast<CLOGStage*>(this)->tail_output_ptr_  = nullptr;
        tail_readback_pending_ = false;
        tail_readback_stream_  = nullptr;
    }
    return {{"output", actual_output_size_}};
}

size_t CLOGStage::getActualOutputSize(int index) const {
    if (index != 0) return 0;
    getActualOutputSizesByName();
    return actual_output_size_;
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
void CLOGStage::execute(
    cudaStream_t stream,
    MemoryPool* pool,
    const std::vector<void*>& inputs,
    const std::vector<void*>& outputs,
    const std::vector<size_t>& sizes)
{
    if (inputs.empty() || outputs.empty() || sizes.empty())
        throw std::runtime_error("CLOGStage: invalid inputs/outputs");

    tail_readback_pending_ = false;
    tail_readback_stream_  = nullptr;
    tail_last_index_       = 0;

    const size_t in_bytes = sizes[0];
    if (in_bytes == 0) { actual_output_size_ = 0; return; }

    if (chunk_size_ != 4096 && chunk_size_ != 8192 && chunk_size_ != 16384)
        throw std::runtime_error(
            "CLOGStage: chunk_size must be 4096, 8192, or 16384; got "
            + std::to_string(chunk_size_));
    if (word_size_ != 1 && word_size_ != 2 && word_size_ != 4 && word_size_ != 8)
        throw std::runtime_error(
            "CLOGStage: word_size must be 1, 2, 4, or 8; got "
            + std::to_string((int)word_size_));

    const size_t   n_chunks   = (in_bytes + chunk_size_ - 1) / chunk_size_;
    const uint32_t n_chunks_u = (uint32_t)n_chunks;
    const uint32_t in_bytes_u = (uint32_t)in_bytes;
    const uint32_t grid256    = (n_chunks_u + 255u) / 256u;

    // ── Forward (compress) ───────────────────────────────────────────────
    if (!is_inverse_) {
        cached_orig_bytes_ = in_bytes_u;

        // Grow persistent forward scratch if needed.
        if (n_chunks > scratch_capacity_) {
            auto fwd_free = [&](void* p) {
                if (!p) return;
                if (scratch_from_pool_ && scratch_pool_owner_ && !scratch_alive_.expired())
                    scratch_pool_owner_->free(p, stream);
                else if (!scratch_from_pool_) { FZ_CUDA_CHECK_WARN(cudaStreamSynchronize(stream)); cudaFree(p); }
            };
            fwd_free(d_scratch_);    d_scratch_    = nullptr;
            fwd_free(d_sizes_dev_);  d_sizes_dev_  = nullptr;
            fwd_free(d_clean_dev_);  d_clean_dev_  = nullptr;
            fwd_free(d_dst_off_dev_);d_dst_off_dev_= nullptr;

            if (pool) {
                d_scratch_    = (uint8_t*) pool->allocate(n_chunks * (size_t)chunk_size_, stream, "clog_scratch", true);
                d_sizes_dev_  = (uint32_t*)pool->allocate(n_chunks * sizeof(uint32_t),    stream, "clog_sizes",   true);
                d_clean_dev_  = (uint32_t*)pool->allocate(n_chunks * sizeof(uint32_t),    stream, "clog_clean",   true);
                d_dst_off_dev_= (uint32_t*)pool->allocate(n_chunks * sizeof(uint32_t),    stream, "clog_offsets", true);
                if (!d_scratch_ || !d_sizes_dev_ || !d_clean_dev_ || !d_dst_off_dev_)
                    throw std::runtime_error("CLOGStage: failed to allocate persistent forward scratch from MemoryPool");
                scratch_pool_owner_ = pool;
                scratch_from_pool_  = true;
                scratch_alive_      = pool->lifetimeToken();
            } else {
                FZ_CUDA_CHECK(cudaMalloc(&d_scratch_,     n_chunks * (size_t)chunk_size_));
                FZ_CUDA_CHECK(cudaMalloc(&d_sizes_dev_,   n_chunks * sizeof(uint32_t)));
                FZ_CUDA_CHECK(cudaMalloc(&d_clean_dev_,   n_chunks * sizeof(uint32_t)));
                FZ_CUDA_CHECK(cudaMalloc(&d_dst_off_dev_, n_chunks * sizeof(uint32_t)));
                scratch_pool_owner_ = nullptr;
                scratch_from_pool_  = false;
            }
            scratch_capacity_ = n_chunks;
        }

        FZ_LOG(TRACE, "CLOG encode: %.1f KB in, %u chunks, word_size %d",
               in_bytes / 1024.0, n_chunks_u, (int)word_size_);

        const size_t header_size = 4 + 4 + 4 * n_chunks;

        // (1) Encode each chunk into uniform scratch, recording flagged sizes.
        launchEncode(word_size_, (uint32_t)chunk_size_, (int)n_chunks, stream,
                     (const byte*)inputs[0], d_scratch_, d_sizes_dev_, in_bytes_u);
        FZ_CUDA_CHECK(cudaGetLastError());

        // (2) Write the stream header (orig_total, num_chunks, per-chunk sizes).
        uint8_t* d_out = (uint8_t*)outputs[0];
        const uint32_t h_hdr[2] = {in_bytes_u, n_chunks_u};
        FZ_CUDA_CHECK(cudaMemcpyAsync(d_out, h_hdr, 8, cudaMemcpyHostToDevice, stream));
        FZ_CUDA_CHECK(cudaMemcpyAsync(d_out + 8, d_sizes_dev_,
                                      n_chunks * sizeof(uint32_t),
                                      cudaMemcpyDeviceToDevice, stream));

        // (3) Strip flag bits → clean sizes.
        clogStripFlagKernel<<<grid256, 256, 0, stream>>>(d_sizes_dev_, d_clean_dev_, n_chunks_u);

        // (4) Exclusive prefix sum of clean sizes → payload-relative offsets.
        {
            auto scan_tmp = fz::backend::withTempStorage(pool, stream, "clog_cub_scan_tmp",
                [&](void* tmp, size_t& bytes) {
                    cub::DeviceScan::ExclusiveSum(tmp, bytes,
                                                  d_clean_dev_, d_dst_off_dev_,
                                                  (int)n_chunks, stream);
                });
            fz::backend::freeTempStorage(pool, scan_tmp, stream);
        }

        // (5) Convert to absolute output offsets (add header size).
        clogAddOffsetKernel<<<grid256, 256, 0, stream>>>(d_dst_off_dev_, n_chunks_u, (uint32_t)header_size);

        // (6) Defer final output-size readback to postStreamSync.
        tail_last_index_       = n_chunks_u - 1;
        tail_output_ptr_       = (uint8_t*)outputs[0];
        tail_readback_pending_ = true;
        tail_readback_stream_  = stream;

        // (7) Pack compressed chunks from uniform scratch to packed output.
        clogPackKernel<<<(int)n_chunks, 512, 0, stream>>>(
            d_scratch_, d_out, d_dst_off_dev_, d_clean_dev_, (uint32_t)chunk_size_);
        FZ_CUDA_CHECK(cudaGetLastError());

    // ── Inverse (decompress) ─────────────────────────────────────────────
    } else {
        const byte* d_in  = (const byte*)inputs[0];
        byte*       d_out = (byte*)outputs[0];

        uint8_t h_hdr_raw[8];
        FZ_CUDA_CHECK(cudaMemcpyAsync(h_hdr_raw, d_in, 8, cudaMemcpyDeviceToHost, stream));
        FZ_CUDA_CHECK(cudaStreamSynchronize(stream));
        uint32_t orig_total, num_chunks;
        std::memcpy(&orig_total, h_hdr_raw + 0, sizeof(uint32_t));
        std::memcpy(&num_chunks, h_hdr_raw + 4, sizeof(uint32_t));
        cached_orig_bytes_ = orig_total;

        if (num_chunks == 0 || orig_total == 0) { actual_output_size_ = 0; return; }

        std::vector<uint32_t> h_chunk_entries(num_chunks);
        FZ_CUDA_CHECK(cudaMemcpyAsync(h_chunk_entries.data(), d_in + 8,
                                      num_chunks * sizeof(uint32_t),
                                      cudaMemcpyDeviceToHost, stream));
        FZ_CUDA_CHECK(cudaStreamSynchronize(stream));

        const size_t header_bytes = 4 + 4 + 4 * num_chunks;

        std::vector<uint32_t> h_in_off(num_chunks), h_comp_sz(num_chunks),
                              h_out_off(num_chunks), h_orig_sz(num_chunks);
        std::vector<bool>     h_is_uncmp(num_chunks);

        uint32_t in_cursor = (uint32_t)header_bytes, out_cursor = 0;
        for (uint32_t i = 0; i < num_chunks; i++) {
            const uint32_t entry  = h_chunk_entries[i];
            const bool     uncmp  = (entry & 0x80000000u) != 0;
            const uint32_t stored = entry & 0x7FFFFFFFu;
            h_in_off[i]   = in_cursor;
            h_comp_sz[i]  = uncmp ? 0u : stored;
            h_out_off[i]  = out_cursor;
            h_is_uncmp[i] = uncmp;
            const uint32_t chunk_orig = (i + 1 < num_chunks)
                ? (uint32_t)chunk_size_ : (uint32_t)(orig_total - out_cursor);
            h_orig_sz[i]  = chunk_orig;
            in_cursor  += stored;
            out_cursor += chunk_orig;
        }

        // Transient decode tables (inverse path is not graph-captured).
        auto alloc_u32 = [&](const char* tag) -> uint32_t* {
            if (pool) return (uint32_t*)pool->allocate(num_chunks * sizeof(uint32_t), stream, tag);
            uint32_t* p = nullptr; FZ_CUDA_CHECK(cudaMalloc(&p, num_chunks * sizeof(uint32_t))); return p;
        };
        uint32_t* d_in_off  = alloc_u32("clog_inv_in_off");
        uint32_t* d_comp_sz = alloc_u32("clog_inv_comp_sz");
        uint32_t* d_out_off = alloc_u32("clog_inv_out_off");
        uint32_t* d_orig_sz = alloc_u32("clog_inv_orig_sz");

        FZ_CUDA_CHECK(cudaMemcpyAsync(d_in_off,  h_in_off.data(),  num_chunks * sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
        FZ_CUDA_CHECK(cudaMemcpyAsync(d_comp_sz, h_comp_sz.data(), num_chunks * sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
        FZ_CUDA_CHECK(cudaMemcpyAsync(d_out_off, h_out_off.data(), num_chunks * sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
        FZ_CUDA_CHECK(cudaMemcpyAsync(d_orig_sz, h_orig_sz.data(), num_chunks * sizeof(uint32_t), cudaMemcpyHostToDevice, stream));

        launchDecode(word_size_, (uint32_t)chunk_size_, (int)num_chunks, stream,
                     d_in, d_out, d_in_off, d_comp_sz, d_out_off, d_orig_sz);
        FZ_CUDA_CHECK(cudaGetLastError());

        // Copy verbatim (raw) chunks the decoder skipped.
        for (uint32_t i = 0; i < num_chunks; i++) {
            if (h_is_uncmp[i])
                FZ_CUDA_CHECK(cudaMemcpyAsync(d_out + h_out_off[i], d_in + h_in_off[i],
                                              h_orig_sz[i], cudaMemcpyDeviceToDevice, stream));
        }

        FZ_CUDA_CHECK(cudaStreamSynchronize(stream));

        if (pool) {
            pool->free(d_in_off, stream);  pool->free(d_comp_sz, stream);
            pool->free(d_out_off, stream); pool->free(d_orig_sz, stream);
        } else {
            cudaFree(d_in_off);  cudaFree(d_comp_sz);
            cudaFree(d_out_off); cudaFree(d_orig_sz);
        }

        actual_output_size_ = (size_t)orig_total;
        FZ_LOG(DEBUG, "CLOG decode done: %.1f KB -> %.1f KB",
               in_bytes / 1024.0, actual_output_size_ / 1024.0);
    }
}

} // namespace fz

// Self-registration for FZM-header reconstruction (see stage_registry.h).
FZ_REGISTER_SIMPLE_STAGE(fz::StageType::CLOG, fz::CLOGStage);
