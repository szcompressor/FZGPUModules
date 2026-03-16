/**
 * modules/transforms/rze/rze_stage.cu
 *
 * GPU implementation of RZEStage — Recursive Zero-byte Elimination.
 *
 * One CUDA block per chunk.  blockDim = 1024 threads.
 * Shared-memory workspace: ~2.5 KB per block (level bitmaps + prefix-sum work).
 *
 * Only chunk_size == 16384 is implemented in this version (runtime check).
 *
 * Algorithm references:
 *   zero_elim.h / repeated_elim.h from the LC framework (Burtscher et al.)
 *   rze.h             — reference composite encode/decode wrapper
 */

#include "transforms/rze/rze_stage.h"
#include "mem/mempool.h"
#include "cuda_check.h"

#include <cuda_runtime.h>
#include <cub/device/device_scan.cuh>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include <iostream>

namespace fz {

RZEStage::~RZEStage() {
    if (d_scratch_) {
        if (scratch_from_pool_ && scratch_pool_owner_) scratch_pool_owner_->free(d_scratch_, 0);
        else cudaFree(d_scratch_);
    }
    if (d_sizes_dev_) {
        if (scratch_from_pool_ && scratch_pool_owner_) scratch_pool_owner_->free(d_sizes_dev_, 0);
        else cudaFree(d_sizes_dev_);
    }
    if (d_clean_dev_) {
        if (scratch_from_pool_ && scratch_pool_owner_) scratch_pool_owner_->free(d_clean_dev_, 0);
        else cudaFree(d_clean_dev_);
    }
    if (d_dst_off_dev_) {
        if (scratch_from_pool_ && scratch_pool_owner_) scratch_pool_owner_->free(d_dst_off_dev_, 0);
        else cudaFree(d_dst_off_dev_);
    }

    if (d_inv_in_off_) {
        if (inv_from_pool_ && inv_pool_owner_) inv_pool_owner_->free(d_inv_in_off_, 0);
        else cudaFree(d_inv_in_off_);
    }
    if (d_inv_comp_sz_) {
        if (inv_from_pool_ && inv_pool_owner_) inv_pool_owner_->free(d_inv_comp_sz_, 0);
        else cudaFree(d_inv_comp_sz_);
    }
    if (d_inv_out_off_) {
        if (inv_from_pool_ && inv_pool_owner_) inv_pool_owner_->free(d_inv_out_off_, 0);
        else cudaFree(d_inv_out_off_);
    }
    if (d_inv_orig_sz_) {
        if (inv_from_pool_ && inv_pool_owner_) inv_pool_owner_->free(d_inv_orig_sz_, 0);
        else cudaFree(d_inv_orig_sz_);
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Compile-time constants for CS = 16384, T = uint8_t
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

static constexpr int RZE_CS     = 16384;  // chunk size in bytes (must match stage config)
static constexpr int RZE_TPB    = 1024;   // threads per block
static constexpr int RZE_NWARPS = RZE_TPB / 32;  // = 32

// Bitmap sizes (for T = uint8_t, one bit per input byte)
static constexpr int RZE_BM1 = RZE_CS / 8;     // 2048  (level-1: 1 bit per data byte)
static constexpr int RZE_BM2 = RZE_CS / 64;    // 256   (level-2: 1 bit per bm1 byte)
static constexpr int RZE_BM3 = RZE_CS / 512;   // 32    (level-3: 1 bit per bm2 byte)
static constexpr int RZE_BM4 = RZE_CS / 4096;  // 4     (level-4: 1 bit per bm3 byte, stored raw)

// Shared-memory layout offsets inside each block's extern smem[]
static constexpr int RZE_SMEM_PS    = 0;                          // 128 bytes: 32-int ps_work
static constexpr int RZE_SMEM_BM1   = RZE_SMEM_PS  + 128;        // 2048 bytes
static constexpr int RZE_SMEM_BM2   = RZE_SMEM_BM1 + RZE_BM1;   // 256  bytes
static constexpr int RZE_SMEM_BM3   = RZE_SMEM_BM2 + RZE_BM2;   // 32   bytes
static constexpr int RZE_SMEM_BM4   = RZE_SMEM_BM3 + RZE_BM3;   // 4    bytes
static constexpr int RZE_SMEM_TOTAL = RZE_SMEM_BM4 + RZE_BM4;   // 2468 bytes

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Block-wide inclusive prefix sum
// Produces correct results when called by the entire block.
// ps_work: RZE_NWARPS integers allocated before the call.
// Returns the per-thread inclusive prefix sum.
// After the call (before next syncthreads), ps_work[RZE_NWARPS-1] == total.
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
static __device__ __forceinline__ int rze_block_prefix_sum(int val, int* ps_work) {
    const int lane  = threadIdx.x & 31;
    const int warp  = threadIdx.x >> 5;

    // Warp-level inclusive prefix
    for (int d = 1; d <= 16; d <<= 1) {
        int n = __shfl_up_sync(0xFFFFFFFFu, val, d);
        if (lane >= d) val += n;
    }

    // Store warp totals (lane 31 == inclusive warp sum)
    if (lane == 31) ps_work[warp] = val;
    __syncthreads();

    // Prefix sum of warp totals (warp 0 does this)
    if (warp == 0 && lane < RZE_NWARPS) {
        int s = ps_work[lane];
        for (int d = 1; d < RZE_NWARPS; d <<= 1) {
            int t = __shfl_up_sync(0xFFFFFFFFu, s, d);
            if (lane >= d) s += t;
        }
        ps_work[lane] = s;
    }
    __syncthreads();

    if (warp > 0) val += ps_work[warp - 1];
    return val;  // inclusive prefix sum for this thread
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// ZE encode: compact non-zero bytes; build bitmap (1 bit per byte).
// Called by the whole block.
//   in[0..N-1]         – input byte array (global memory)
//   out[0..]           – output: non-zero bytes written here
//   bm[0..N/8-1]       – output bitmap (1 = non-zero, 0 = zero); must be smem
//   ps_work[NWARPS]    – smem workspace for prefix sum
// Returns total non-zero bytes written.
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
static __device__ int rze_ze_encode(
    const uint8_t* __restrict__ in, int N,
    uint8_t* __restrict__ out,
    uint8_t* __restrict__ bm,
    int*     __restrict__ ps_work)
{
    const int tid    = threadIdx.x;
    const int lane   = tid & 31;
    const int warp   = tid >> 5;
    int total = 0;

    for (int base = 0; base < N; base += RZE_TPB) {
        const int  gi     = base + tid;
        const bool active = gi < N;
        const uint8_t v   = active ? in[gi] : 0;
        const bool    nz  = active && (v != 0);

        // One 32-bit ballot per warp → 4 bytes of bitmap per warp per iteration
        const uint32_t mask = __ballot_sync(0xFFFFFFFFu, (unsigned)nz);
        if (lane == 0 && (base + warp * 32) < N)
            ((uint32_t*)bm)[base / 32 + warp] = mask;

        // Warp output positions via block prefix sum
        if (lane == 0) ps_work[warp] = (int)__popc(mask);
        __syncthreads();
        if (warp == 0 && lane < RZE_NWARPS) {
            int s = ps_work[lane];
            for (int d = 1; d < RZE_NWARPS; d <<= 1) {
                int t = __shfl_up_sync(~0u, s, d); if (lane >= d) s += t;
            }
            ps_work[lane] = s;
        }
        __syncthreads();

        const int warp_exc = (warp > 0) ? ps_work[warp - 1] : 0;
        if (nz) out[total + warp_exc + (int)__popc(mask & ((1u << lane) - 1u))] = v;

        total += ps_work[RZE_NWARPS - 1];
        __syncthreads();
    }
    return total;
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// RE encode: compact non-repeated bytes; build bitmap (1 bit per byte).
// A byte at position i is "non-repeated" if it differs from byte i-1.
// Position 0 is compared against 0 (implicit predecessor).
// Called by the whole block.
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
static __device__ int rze_re_encode(
    const uint8_t* __restrict__ in, int N,
    uint8_t* __restrict__ out,
    uint8_t* __restrict__ bm,
    int*     __restrict__ ps_work)
{
    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    int total = 0;

    for (int base = 0; base < N; base += RZE_TPB) {
        const int  gi     = base + tid;
        const bool active = gi < N;
        const uint8_t v   = active ? in[gi]       : 0;
        const uint8_t p   = (gi > 0 && active) ? in[gi - 1] : 0;
        const bool    nr  = active && (v != p);

        const uint32_t mask = __ballot_sync(0xFFFFFFFFu, (unsigned)nr);
        if (lane == 0 && (base + warp * 32) < N)
            ((uint32_t*)bm)[base / 32 + warp] = mask;

        if (lane == 0) ps_work[warp] = (int)__popc(mask);
        __syncthreads();
        if (warp == 0 && lane < RZE_NWARPS) {
            int s = ps_work[lane];
            for (int d = 1; d < RZE_NWARPS; d <<= 1) {
                int t = __shfl_up_sync(~0u, s, d); if (lane >= d) s += t;
            }
            ps_work[lane] = s;
        }
        __syncthreads();

        const int warp_exc = (warp > 0) ? ps_work[warp - 1] : 0;
        if (nr) out[total + warp_exc + (int)__popc(mask & ((1u << lane) - 1u))] = v;

        total += ps_work[RZE_NWARPS - 1];
        __syncthreads();
    }
    return total;
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// ZE decode: scatter datain bytes back to out using bm.
// Parallel – called by the full block.
//   decsize               – number of output bytes to reconstruct
//   datain[0..data_nz-1]  – non-zero bytes (global memory or scratch)
//   bm[0..decsize/8-1]    – bitmap (smem)
//   out[0..decsize-1]     – output
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
static __device__ void rze_ze_decode(
    int decsize,
    const uint8_t* __restrict__ datain,
    const uint8_t* __restrict__ bm,
    uint8_t* __restrict__ out,
    int*     __restrict__ ps_work)
{
    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    int data_offset = 0;

    for (int base = 0; base < decsize; base += RZE_TPB) {
        const int  gi     = base + tid;
        const bool active = gi < decsize;
        const bool nz = active && ((bm[gi >> 3] >> (gi & 7)) & 1);

        const uint32_t mask = __ballot_sync(0xFFFFFFFFu, (unsigned)nz);

        if (lane == 0) ps_work[warp] = (int)__popc(mask);
        __syncthreads();
        if (warp == 0 && lane < RZE_NWARPS) {
            int s = ps_work[lane];
            for (int d = 1; d < RZE_NWARPS; d <<= 1) {
                int t = __shfl_up_sync(~0u, s, d); if (lane >= d) s += t;
            }
            ps_work[lane] = s;
        }
        __syncthreads();

        const int warp_exc = (warp > 0) ? ps_work[warp - 1] : 0;
        if (active) {
            if (nz) out[gi] = datain[data_offset + warp_exc + (int)__popc(mask & ((1u << lane) - 1u))];
            else    out[gi] = 0;
        }
        data_offset += ps_work[RZE_NWARPS - 1];
        __syncthreads();
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// RE decode (sequential, thread 0 only): expand non-repeated bytes back to
// decsize bytes.  Suitable for small decsize (≤ 2048 for the bitmap levels).
//   decsize               – output size
//   datain[0..nr_count-1] – non-repeated bytes
//   bm[0..decsize/8-1]    – bitmap
//   out[0..decsize-1]     – output (smem)
// Called inside a syncthreads() boundary; subsequent code must syncthreads.
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
static __device__ void rze_re_decode_seq(
    int decsize,
    const uint8_t* __restrict__ datain,
    const uint8_t* __restrict__ bm,
    uint8_t* __restrict__ out)
{
    if (threadIdx.x == 0) {
        int pos = 0;
        uint8_t cur = 0;
        for (int i = 0; i < decsize; i++) {
            if ((bm[i >> 3] >> (i & 7)) & 1) cur = datain[pos++];
            out[i] = cur;
        }
    }
    __syncthreads();
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Block-wide popcount for a byte array in smem or gmem.
// Returns the total number of set bits across arr[0..n_bytes-1].
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
static __device__ int rze_block_popcount(const uint8_t* arr, int n_bytes, int* ps_work) {
    const int tid = threadIdx.x;
    int cnt = 0;
    for (int i = tid; i < n_bytes; i += RZE_TPB)
        cnt += __popc((unsigned)arr[i]);
    cnt = rze_block_prefix_sum(cnt, ps_work);
    (void)cnt;  // we only care about the total
    // After rze_block_prefix_sum, ps_work[RZE_NWARPS-1] == total
    __syncthreads();
    return ps_work[RZE_NWARPS - 1];
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Main encode kernel (one block per chunk, blockDim = 1024, levels = 4)
//
//  d_in[0..total_in_bytes-1]             – all input chunks concatenated
//  d_scratch[chunk_id*CS..(+1)*CS-1]     – per-chunk output scratch (CS bytes each)
//  d_sizes[chunk_id]                     – actual compressed bytes written
//    (high bit set when chunk was stored uncompressed)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
static __global__ void __launch_bounds__(RZE_TPB)
rzeEncodeKernel(
    const uint8_t* __restrict__ d_in,
    uint8_t*       __restrict__ d_scratch,
    uint32_t*      __restrict__ d_sizes,
    uint32_t total_in_bytes)
{
    extern __shared__ uint8_t smem[];

    int* __restrict__    ps_work = reinterpret_cast<int*>(smem + RZE_SMEM_PS);
    uint8_t* __restrict__ bm1   = smem + RZE_SMEM_BM1;
    uint8_t* __restrict__ bm2   = smem + RZE_SMEM_BM2;
    uint8_t* __restrict__ bm3   = smem + RZE_SMEM_BM3;
    uint8_t* __restrict__ bm4   = smem + RZE_SMEM_BM4;

    const uint32_t cid      = blockIdx.x;
    const uint32_t in_off   = cid * (uint32_t)RZE_CS;
    const int      in_size  = static_cast<int>(
        min((uint32_t)RZE_CS, total_in_bytes - in_off));

    const uint8_t* in  = d_in + in_off;
    uint8_t*       out = d_scratch + cid * (uint32_t)RZE_CS;

    // Zero bm1..bm4 for the padded region (if in_size < RZE_CS)
    if (in_size < RZE_CS) {
        const int bm1_full = (in_size + 7) / 8;
        for (int i = threadIdx.x; i < RZE_BM1; i += RZE_TPB)
            if (i >= bm1_full) bm1[i] = 0;
        __syncthreads();
    }

    // ── Level 1: ZE encode ------------------------------------------------
    const int data_nz = rze_ze_encode(in, in_size, out, bm1, ps_work);
    int wpos = data_nz;

    if (wpos == 0) {
        // All zeros — just write the 2-byte size tag; no bitmaps needed
        if (threadIdx.x == 0) {
            out[0] = (uint8_t)(in_size & 0xFF);
            out[1] = (uint8_t)((in_size >> 8) & 0xFF);
            d_sizes[cid] = 2;
        }
        return;
    }

    const int avail = RZE_CS - 2;  // 2 bytes for size tag

    // ── Level 2: RE encode of bm1 ----------------------------------------
    // bm1 output buffer for level-2 RE goes AFTER wpos in out[]; bm2 into smem
    const int bm1_bytes = (in_size + 7) / 8;  // actual bm1 size
    if (wpos >= avail || wpos > avail - bm1_bytes) {
        for (int i = threadIdx.x; i < in_size; i += RZE_TPB)
            out[i] = in[i];
        if (threadIdx.x == 0)
            d_sizes[cid] = (1u << 31) | (uint32_t)in_size;
        return;
    }
    const int bm2_nr = rze_re_encode(bm1, bm1_bytes, out + wpos, bm2, ps_work);
    wpos += bm2_nr;

    if (wpos >= avail) {
        // Incompressible — store original data verbatim
        for (int i = threadIdx.x; i < in_size; i += RZE_TPB)
            out[i] = in[i];
        if (threadIdx.x == 0)
            d_sizes[cid] = (1u << 31) | (uint32_t)in_size;
        return;
    }

    // ── Level 3: RE encode of bm2 ----------------------------------------
    const int bm2_bytes = (bm1_bytes + 7) / 8;
    if (wpos >= avail || wpos > avail - bm2_bytes) {
        for (int i = threadIdx.x; i < in_size; i += RZE_TPB)
            out[i] = in[i];
        if (threadIdx.x == 0)
            d_sizes[cid] = (1u << 31) | (uint32_t)in_size;
        return;
    }
    const int bm3_nr = rze_re_encode(bm2, bm2_bytes, out + wpos, bm3, ps_work);
    wpos += bm3_nr;

    if (wpos >= avail) {
        for (int i = threadIdx.x; i < in_size; i += RZE_TPB)
            out[i] = in[i];
        if (threadIdx.x == 0)
            d_sizes[cid] = (1u << 31) | (uint32_t)in_size;
        return;
    }

    // ── Level 4: RE encode of bm3 ----------------------------------------
    const int bm3_bytes = (bm2_bytes + 7) / 8;
    if (wpos >= avail - (int)RZE_BM4 || wpos > (avail - (int)RZE_BM4) - bm3_bytes) {
        for (int i = threadIdx.x; i < in_size; i += RZE_TPB)
            out[i] = in[i];
        if (threadIdx.x == 0)
            d_sizes[cid] = (1u << 31) | (uint32_t)in_size;
        return;
    }
    const int bm4_nr = rze_re_encode(bm3, bm3_bytes, out + wpos, bm4, ps_work);
    wpos += bm4_nr;

    if (wpos >= avail - (int)RZE_BM4) {
        for (int i = threadIdx.x; i < in_size; i += RZE_TPB)
            out[i] = in[i];
        if (threadIdx.x == 0)
            d_sizes[cid] = (1u << 31) | (uint32_t)in_size;
        return;
    }

    // ── Write raw bm4 (4 bytes) + 2-byte size tag -------------------------
    if (threadIdx.x < RZE_BM4)
        out[wpos + threadIdx.x] = bm4[threadIdx.x];
    wpos += RZE_BM4;

    __syncthreads();

    if (threadIdx.x == 0) {
        out[wpos]     = (uint8_t)(in_size & 0xFF);
        out[wpos + 1] = (uint8_t)((in_size >> 8) & 0xFF);
        d_sizes[cid]  = (uint32_t)(wpos + 2);
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Main decode kernel (one block per chunk, blockDim = 1024, levels = 4)
//
//  d_in[chunk_in_off..chunk_in_off+comp_size-1]  – compressed chunk bytes
//  d_out[chunk_out_off..+orig_size-1]            – decompressed output
//  d_in_offsets[chunk_id]                        – start of this chunk in d_in
//  d_comp_sizes[chunk_id]                        – compressed size (without high-bit flag)
//  d_out_offsets[chunk_id]                       – start of output chunk in d_out
//  d_orig_sizes[chunk_id]                        – expected original size (for zeros case)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
static __global__ void __launch_bounds__(RZE_TPB)
rzeDecodeKernel(
    const uint8_t* __restrict__ d_in,
    uint8_t*       __restrict__ d_out,
    const uint32_t* __restrict__ d_in_offsets,
    const uint32_t* __restrict__ d_comp_sizes,
    const uint32_t* __restrict__ d_out_offsets,
    const uint32_t* __restrict__ d_orig_sizes)
{
    extern __shared__ uint8_t smem[];

    int* __restrict__    ps_work = reinterpret_cast<int*>(smem + RZE_SMEM_PS);
    uint8_t* __restrict__ bm1   = smem + RZE_SMEM_BM1;
    uint8_t* __restrict__ bm2   = smem + RZE_SMEM_BM2;
    uint8_t* __restrict__ bm3   = smem + RZE_SMEM_BM3;
    uint8_t* __restrict__ bm4   = smem + RZE_SMEM_BM4;

    const uint32_t cid       = blockIdx.x;
    const uint32_t in_off    = d_in_offsets[cid];
    const uint32_t comp_size = d_comp_sizes[cid];
    const uint32_t out_off   = d_out_offsets[cid];
    const uint32_t orig_size = d_orig_sizes[cid];

    const uint8_t* in  = d_in  + in_off;
    uint8_t*       out = d_out + out_off;

    // Handle uncompressed passthrough (marked by caller before launching)
    // comp_sizes/flags resolved by host; this kernel only sees clean data.

    if (comp_size == 0) {
        // Shouldn't normally happen; treat as all zeros
        for (int i = threadIdx.x; i < (int)orig_size; i += RZE_TPB)
            out[i] = 0;
        return;
    }

    // Read original chunk size from last 2 bytes of compressed data
    // (Should match orig_size passed from header, but we use the embedded tag
    // to determine whether this was the all-zeros case below.)
    const int cs_tag = (int)in[comp_size - 2]
                     | ((int)in[comp_size - 1] << 8);
    (void)cs_tag;  // matches orig_size

    // All-zeros case: only the 2-byte tag was stored
    if (comp_size == 2) {
        for (int i = threadIdx.x; i < (int)orig_size; i += RZE_TPB)
            out[i] = 0;
        return;
    }

    // ── Read bm4 raw (4 bytes immediately before the 2-byte tag) ---------
    const int bm4_start = (int)comp_size - 2 - RZE_BM4;
    if (threadIdx.x < RZE_BM4)
        bm4[threadIdx.x] = in[bm4_start + threadIdx.x];
    __syncthreads();

    // Count set bits in bm4 → number of non-repeated bm3 bytes stored
    const int bm4_nr = rze_block_popcount(bm4, RZE_BM4, ps_work);

    // bm4_nr non-repeated bm3 bytes are just before bm4
    const int bm3_nr_start = bm4_start - bm4_nr;

    // ── RE decode bm3 (bm4_nr bytes → bm3_bytes decoded into smem) -------
    const int bm2_bytes = (RZE_BM1 + 7) / 8;           // = RZE_BM2 = 256
    const int bm3_bytes = (bm2_bytes + 7) / 8;          // = RZE_BM3 = 32
    rze_re_decode_seq(bm3_bytes, in + bm3_nr_start, bm4, bm3);
    // (rze_re_decode_seq ends with __syncthreads)

    // Count set bits in bm3 → number of non-repeated bm2 bytes stored
    const int bm3_nr = rze_block_popcount(bm3, bm3_bytes, ps_work);

    const int bm2_nr_start = bm3_nr_start - bm3_nr;

    // ── RE decode bm2 (bm3_nr bytes → bm2_bytes decoded into smem) -------
    rze_re_decode_seq(bm2_bytes, in + bm2_nr_start, bm3, bm2);

    // Count set bits in bm2 → number of non-repeated bm1 bytes stored
    const int bm2_nr = rze_block_popcount(bm2, bm2_bytes, ps_work);

    const int bm1_nr_start = bm2_nr_start - bm2_nr;

    // ── RE decode bm1 (bm2_nr bytes → bm1_bytes decoded into smem) -------
    const int bm1_bytes = (RZE_CS + 7) / 8;  // = RZE_BM1 = 2048
    rze_re_decode_seq(bm1_bytes, in + bm1_nr_start, bm2, bm1);

    // ── ZE decode data (data_nz bytes → orig_size bytes) -----------------
    // data_nz = set bits in bm1 = bm1_nr_start  (non-zero bytes start at in[0])
    rze_ze_decode((int)orig_size, in, bm1, out, ps_work);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// stripFlagKernel: clear the high-bit "incompressible" flag from each chunk's
// stored size so that rzePackKernel copies exactly the right byte count.
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
static __global__ void stripFlagKernel(
    const uint32_t* __restrict__ d_sizes_with_flag,
    uint32_t*       __restrict__ d_clean_sizes,
    uint32_t n_chunks)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_chunks)
        d_clean_sizes[i] = d_sizes_with_flag[i] & 0x7FFFFFFFu;
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// addOffsetKernel: shift every element of an uint32 array by a constant.
// Used to convert the CUB exclusive-scan result (offsets relative to payload
// start) into absolute offsets within the output buffer (by adding the header
// size).
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
static __global__ void addOffsetKernel(
    uint32_t* __restrict__ arr,
    uint32_t n,
    uint32_t offset)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) arr[i] += offset;
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Pack kernel: copy compressed chunks from a uniform scratch layout
// (each chunk at scratch[cid * CS]) to a compact output buffer.
//
//  d_scratch[cid * CS .. cid*CS + sizes[cid]-1]  – compressed chunk data
//  d_dst_offsets[cid]                             – destination offset in d_out
//  d_sizes[cid]                                   – compressed chunk size (clean; no flags)
//  CS                                             – scratch stride per chunk
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
static __global__ void rzePackKernel(
    const uint8_t*  __restrict__ d_scratch,
    uint8_t*        __restrict__ d_out,
    const uint32_t* __restrict__ d_dst_offsets,
    const uint32_t* __restrict__ d_sizes,
    uint32_t CS)
{
    const uint32_t cid      = blockIdx.x;
    const uint32_t src_off  = cid * CS;
    const uint32_t dst_off  = d_dst_offsets[cid];
    const uint32_t sz       = d_sizes[cid];
    const uint8_t* src = d_scratch + src_off;
    uint8_t*       dst = d_out     + dst_off;
    for (uint32_t i = threadIdx.x; i < sz; i += blockDim.x)
        dst[i] = src[i];
}

void RZEStage::postStreamSync(cudaStream_t /*stream*/) {
    if (!tail_readback_pending_) return;

    uint32_t tail_off = 0;
    uint32_t tail_sz  = 0;
    FZ_CUDA_CHECK(cudaMemcpy(&tail_off, d_dst_off_dev_ + tail_last_index_,
                             sizeof(uint32_t), cudaMemcpyDeviceToHost));
    FZ_CUDA_CHECK(cudaMemcpy(&tail_sz, d_clean_dev_ + tail_last_index_,
                             sizeof(uint32_t), cudaMemcpyDeviceToHost));
    const size_t total_out = static_cast<size_t>(tail_off)
                           + static_cast<size_t>(tail_sz);
    actual_output_size_ = (total_out + 3) & ~size_t(3);
    tail_readback_pending_ = false;
    tail_readback_stream_ = nullptr;
}

std::unordered_map<std::string, size_t>
RZEStage::getActualOutputSizesByName() const {
    if (tail_readback_pending_) {
        FZ_CUDA_CHECK(cudaStreamSynchronize(tail_readback_stream_));
        uint32_t tail_off = 0;
        uint32_t tail_sz  = 0;
        FZ_CUDA_CHECK(cudaMemcpy(&tail_off, d_dst_off_dev_ + tail_last_index_,
                                 sizeof(uint32_t), cudaMemcpyDeviceToHost));
        FZ_CUDA_CHECK(cudaMemcpy(&tail_sz, d_clean_dev_ + tail_last_index_,
                                 sizeof(uint32_t), cudaMemcpyDeviceToHost));
        const size_t total_out = static_cast<size_t>(tail_off)
                               + static_cast<size_t>(tail_sz);
        const_cast<RZEStage*>(this)->actual_output_size_ = (total_out + 3) & ~size_t(3);
        tail_readback_pending_ = false;
        tail_readback_stream_ = nullptr;
    }
    return {{"output", actual_output_size_}};
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// RZEStage::execute
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
void RZEStage::execute(
    cudaStream_t stream,
    MemoryPool* pool,
    const std::vector<void*>& inputs,
    const std::vector<void*>& outputs,
    const std::vector<size_t>& sizes)
{
    if (inputs.empty() || outputs.empty() || sizes.empty())
        throw std::runtime_error("RZEStage: invalid inputs/outputs");

    tail_readback_pending_ = false;
    tail_readback_stream_ = nullptr;
    tail_last_index_ = 0;

    const size_t in_bytes = sizes[0];
    if (in_bytes == 0) { actual_output_size_ = 0; return; }

    // Only CS = 16384 is implemented
    if (chunk_size_ != 16384)
        throw std::runtime_error(
            "RZEStage: only chunk_size == 16384 is currently supported; got "
            + std::to_string(chunk_size_));

    if (levels_ != 4)
        throw std::runtime_error(
            "RZEStage: only levels == 4 is currently supported; got "
            + std::to_string(static_cast<int>(levels_)));

    const size_t   n_chunks    = (in_bytes + chunk_size_ - 1) / chunk_size_;
    const uint32_t n_chunks_u  = static_cast<uint32_t>(n_chunks);
    const uint32_t in_bytes_u  = static_cast<uint32_t>(in_bytes);
    const uint32_t grid256     = (n_chunks_u + 255u) / 256u;

    // ── Grow persistent scratch if the current dataset is larger ─────────
    // These allocations happen only on the very first call (or if a larger
    // dataset is seen), not on every compress() invocation.  cudaMalloc is
    // synchronous but rare; for steady-state repeated calls the branches are
    // not taken and the GPU pipeline is fully asynchronous.
    if (n_chunks > scratch_capacity_) {
        if (d_scratch_) {
            if (scratch_from_pool_ && scratch_pool_owner_) scratch_pool_owner_->free(d_scratch_, stream);
            else cudaFree(d_scratch_);
            d_scratch_ = nullptr;
        }
        if (d_sizes_dev_) {
            if (scratch_from_pool_ && scratch_pool_owner_) scratch_pool_owner_->free(d_sizes_dev_, stream);
            else cudaFree(d_sizes_dev_);
            d_sizes_dev_ = nullptr;
        }
        if (d_clean_dev_) {
            if (scratch_from_pool_ && scratch_pool_owner_) scratch_pool_owner_->free(d_clean_dev_, stream);
            else cudaFree(d_clean_dev_);
            d_clean_dev_ = nullptr;
        }
        if (d_dst_off_dev_) {
            if (scratch_from_pool_ && scratch_pool_owner_) scratch_pool_owner_->free(d_dst_off_dev_, stream);
            else cudaFree(d_dst_off_dev_);
            d_dst_off_dev_ = nullptr;
        }

        if (pool) {
            d_scratch_ = static_cast<uint8_t*>(pool->allocate(
                n_chunks * static_cast<size_t>(chunk_size_), stream,
                "rze_persistent_scratch", /*persistent=*/true));
            d_sizes_dev_ = static_cast<uint32_t*>(pool->allocate(
                n_chunks * sizeof(uint32_t), stream,
                "rze_persistent_sizes", /*persistent=*/true));
            d_clean_dev_ = static_cast<uint32_t*>(pool->allocate(
                n_chunks * sizeof(uint32_t), stream,
                "rze_persistent_clean", /*persistent=*/true));
            d_dst_off_dev_ = static_cast<uint32_t*>(pool->allocate(
                n_chunks * sizeof(uint32_t), stream,
                "rze_persistent_offsets", /*persistent=*/true));
            if (!d_scratch_ || !d_sizes_dev_ || !d_clean_dev_ || !d_dst_off_dev_) {
                throw std::runtime_error("RZEStage: failed to allocate persistent forward scratch from MemoryPool");
            }
            scratch_pool_owner_ = pool;
            scratch_from_pool_ = true;
        } else {
            FZ_CUDA_CHECK(cudaMalloc(&d_scratch_,     n_chunks * static_cast<size_t>(chunk_size_)));
            FZ_CUDA_CHECK(cudaMalloc(&d_sizes_dev_,   n_chunks * sizeof(uint32_t)));
            FZ_CUDA_CHECK(cudaMalloc(&d_clean_dev_,   n_chunks * sizeof(uint32_t)));
            FZ_CUDA_CHECK(cudaMalloc(&d_dst_off_dev_, n_chunks * sizeof(uint32_t)));
            scratch_pool_owner_ = nullptr;
            scratch_from_pool_ = false;
        }
        scratch_capacity_ = n_chunks;
    }

    // ── Forward (compress) ───────────────────────────────────────────────
    if (!is_inverse_) {
        cached_orig_bytes_ = in_bytes_u;

        const size_t header_size = 4 + 4 + 4 * n_chunks;

        // ── (1) Encode: each block writes its compressed chunk to d_scratch_
        //               and records the (possibly-flagged) size in d_sizes_dev_.
        rzeEncodeKernel<<<static_cast<int>(n_chunks), RZE_TPB, RZE_SMEM_TOTAL, stream>>>(
            static_cast<const uint8_t*>(inputs[0]),
            d_scratch_,
            d_sizes_dev_,
            in_bytes_u);
        FZ_CUDA_CHECK(cudaGetLastError());

        // ── (2) Write header directly to the output buffer (fully async).
        //    First 8 bytes come from host (2 × uint32); per-chunk sizes are
        //    D→D copied straight from d_sizes_dev_ (flag bits preserved so
        //    the decoder can detect uncompressed chunks).
        uint8_t* d_out = static_cast<uint8_t*>(outputs[0]);
        const uint32_t h_hdr[2] = {in_bytes_u, n_chunks_u};
        FZ_CUDA_CHECK(cudaMemcpyAsync(d_out, h_hdr, 8, cudaMemcpyHostToDevice, stream));
        FZ_CUDA_CHECK(cudaMemcpyAsync(d_out + 8, d_sizes_dev_,
                                     n_chunks * sizeof(uint32_t),
                                     cudaMemcpyDeviceToDevice, stream));

        // ── (3) Strip flag bits → d_clean_dev_ (clean compressed sizes).
        stripFlagKernel<<<grid256, 256, 0, stream>>>(d_sizes_dev_, d_clean_dev_, n_chunks_u);

        // ── (4) Exclusive prefix sum of clean sizes → d_dst_off_dev_.
        //    After this, d_dst_off_dev_[i] = sum of clean_sizes[0..i-1] (relative
        //    to the start of the payload region).  We add header_size below so
        //    that rzePackKernel can write directly to absolute output offsets.
        {
            size_t scan_tmp_bytes = 0;
            cub::DeviceScan::ExclusiveSum(nullptr, scan_tmp_bytes,
                                          d_clean_dev_, d_dst_off_dev_,
                                          static_cast<int>(n_chunks), stream);
            void* d_scan_tmp = pool
                ? pool->allocate(scan_tmp_bytes, stream, "rze_cub_scan_tmp")
                : nullptr;
            if (!d_scan_tmp && scan_tmp_bytes > 0)
                FZ_CUDA_CHECK(cudaMallocAsync(&d_scan_tmp, scan_tmp_bytes, stream));
            cub::DeviceScan::ExclusiveSum(d_scan_tmp, scan_tmp_bytes,
                                          d_clean_dev_, d_dst_off_dev_,
                                          static_cast<int>(n_chunks), stream);
            if (pool && d_scan_tmp)
                pool->free(d_scan_tmp, stream);
            else if (d_scan_tmp)
                FZ_CUDA_CHECK_WARN(cudaFreeAsync(d_scan_tmp, stream));
        }

        // ── (5) Convert payload-relative offsets to absolute output offsets.
        addOffsetKernel<<<grid256, 256, 0, stream>>>(
            d_dst_off_dev_, n_chunks_u, static_cast<uint32_t>(header_size));

        // ── (6) Mark that output size should be finalized after stream sync.
        //    total = d_dst_off_dev_[last] + d_clean_dev_[last]
        tail_last_index_ = n_chunks_u - 1;
        tail_readback_pending_ = true;
        tail_readback_stream_ = stream;

        // ── (7) Pack: scatter compressed chunks from uniform scratch to packed output.
        rzePackKernel<<<static_cast<int>(n_chunks), 512, 0, stream>>>(
            d_scratch_, d_out,
            d_dst_off_dev_, d_clean_dev_,
            static_cast<uint32_t>(chunk_size_));
        FZ_CUDA_CHECK(cudaGetLastError());

    // ── Inverse (decompress) ─────────────────────────────────────────────
    } else {

        const uint8_t* d_in = static_cast<const uint8_t*>(inputs[0]);
        uint8_t*       d_out = static_cast<uint8_t*>(outputs[0]);

        // Read 8-byte header prefix to learn orig_total and num_chunks.
        uint8_t h_hdr_raw[8];
        FZ_CUDA_CHECK(cudaMemcpy(h_hdr_raw, d_in, 8, cudaMemcpyDeviceToHost));

        uint32_t orig_total, num_chunks;
        std::memcpy(&orig_total,  h_hdr_raw + 0, sizeof(uint32_t));
        std::memcpy(&num_chunks,  h_hdr_raw + 4, sizeof(uint32_t));

        cached_orig_bytes_ = orig_total;

        if (num_chunks == 0 || orig_total == 0) {
            actual_output_size_ = 0;
            return;
        }

        // Read per-chunk size entries from header (small, host-side only).
        std::vector<uint32_t> h_chunk_entries(num_chunks);
        FZ_CUDA_CHECK(cudaMemcpy(h_chunk_entries.data(), d_in + 8,
                              num_chunks * sizeof(uint32_t), cudaMemcpyDeviceToHost));

        const size_t header_bytes = 4 + 4 + 4 * num_chunks;

        // Compute per-chunk decode tables on host (fast integer work).
        std::vector<uint32_t> h_in_off(num_chunks);
        std::vector<uint32_t> h_comp_sz(num_chunks);
        std::vector<uint32_t> h_out_off(num_chunks);
        std::vector<uint32_t> h_orig_sz(num_chunks);
        std::vector<bool>     h_is_uncmp(num_chunks);

        uint32_t in_cursor  = static_cast<uint32_t>(header_bytes);
        uint32_t out_cursor = 0;

        for (uint32_t i = 0; i < num_chunks; i++) {
            const uint32_t entry  = h_chunk_entries[i];
            const bool     uncmp  = (entry & 0x80000000u) != 0;
            const uint32_t stored = entry & 0x7FFFFFFFu;

            h_in_off[i]  = in_cursor;
            h_comp_sz[i] = uncmp ? 0u : stored;
            h_out_off[i] = out_cursor;
            h_is_uncmp[i] = uncmp;

            const uint32_t chunk_orig = (i + 1 < num_chunks)
                ? static_cast<uint32_t>(chunk_size_)
                : static_cast<uint32_t>(orig_total - out_cursor);
            h_orig_sz[i] = chunk_orig;

            in_cursor  += stored;
            out_cursor += chunk_orig;
        }

        // ── Grow inverse decode-table scratch if needed ───────────────────
        if (num_chunks > inv_capacity_) {
            if (d_inv_in_off_) {
                if (inv_from_pool_ && inv_pool_owner_) inv_pool_owner_->free(d_inv_in_off_, stream);
                else cudaFree(d_inv_in_off_);
                d_inv_in_off_ = nullptr;
            }
            if (d_inv_comp_sz_) {
                if (inv_from_pool_ && inv_pool_owner_) inv_pool_owner_->free(d_inv_comp_sz_, stream);
                else cudaFree(d_inv_comp_sz_);
                d_inv_comp_sz_ = nullptr;
            }
            if (d_inv_out_off_) {
                if (inv_from_pool_ && inv_pool_owner_) inv_pool_owner_->free(d_inv_out_off_, stream);
                else cudaFree(d_inv_out_off_);
                d_inv_out_off_ = nullptr;
            }
            if (d_inv_orig_sz_) {
                if (inv_from_pool_ && inv_pool_owner_) inv_pool_owner_->free(d_inv_orig_sz_, stream);
                else cudaFree(d_inv_orig_sz_);
                d_inv_orig_sz_ = nullptr;
            }

            if (pool) {
                d_inv_in_off_ = static_cast<uint32_t*>(pool->allocate(
                    num_chunks * sizeof(uint32_t), stream,
                    "rze_persistent_inv_in_off", /*persistent=*/true));
                d_inv_comp_sz_ = static_cast<uint32_t*>(pool->allocate(
                    num_chunks * sizeof(uint32_t), stream,
                    "rze_persistent_inv_comp_sz", /*persistent=*/true));
                d_inv_out_off_ = static_cast<uint32_t*>(pool->allocate(
                    num_chunks * sizeof(uint32_t), stream,
                    "rze_persistent_inv_out_off", /*persistent=*/true));
                d_inv_orig_sz_ = static_cast<uint32_t*>(pool->allocate(
                    num_chunks * sizeof(uint32_t), stream,
                    "rze_persistent_inv_orig_sz", /*persistent=*/true));
                if (!d_inv_in_off_ || !d_inv_comp_sz_ || !d_inv_out_off_ || !d_inv_orig_sz_) {
                    throw std::runtime_error("RZEStage: failed to allocate persistent inverse scratch from MemoryPool");
                }
                inv_pool_owner_ = pool;
                inv_from_pool_ = true;
            } else {
                FZ_CUDA_CHECK(cudaMalloc(&d_inv_in_off_,  num_chunks * sizeof(uint32_t)));
                FZ_CUDA_CHECK(cudaMalloc(&d_inv_comp_sz_, num_chunks * sizeof(uint32_t)));
                FZ_CUDA_CHECK(cudaMalloc(&d_inv_out_off_, num_chunks * sizeof(uint32_t)));
                FZ_CUDA_CHECK(cudaMalloc(&d_inv_orig_sz_, num_chunks * sizeof(uint32_t)));
                inv_pool_owner_ = nullptr;
                inv_from_pool_ = false;
            }
            inv_capacity_ = num_chunks;
        }

        // Upload decode tables asynchronously.
        FZ_CUDA_CHECK(cudaMemcpyAsync(d_inv_in_off_,  h_in_off.data(),
                                     num_chunks * sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
        FZ_CUDA_CHECK(cudaMemcpyAsync(d_inv_comp_sz_, h_comp_sz.data(),
                                     num_chunks * sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
        FZ_CUDA_CHECK(cudaMemcpyAsync(d_inv_out_off_, h_out_off.data(),
                                     num_chunks * sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
        FZ_CUDA_CHECK(cudaMemcpyAsync(d_inv_orig_sz_, h_orig_sz.data(),
                                     num_chunks * sizeof(uint32_t), cudaMemcpyHostToDevice, stream));

        // Launch decode kernel for all chunks (uncompressed chunks are handled
        // inside the kernel via passthrough, so no separate branch is needed here).
        rzeDecodeKernel<<<static_cast<int>(num_chunks), RZE_TPB, RZE_SMEM_TOTAL, stream>>>(
            d_in, d_out,
            d_inv_in_off_, d_inv_comp_sz_, d_inv_out_off_, d_inv_orig_sz_);
        FZ_CUDA_CHECK(cudaGetLastError());

        // Copy uncompressed chunks (passthrough — decoder doesn't handle these).
        for (uint32_t i = 0; i < num_chunks; i++) {
            if (h_is_uncmp[i]) {
                FZ_CUDA_CHECK(cudaMemcpyAsync(
                    d_out + h_out_off[i],
                    d_in  + h_in_off[i],
                    h_orig_sz[i],
                    cudaMemcpyDeviceToDevice, stream));
            }
        }

        FZ_CUDA_CHECK(cudaStreamSynchronize(stream));

        actual_output_size_ = static_cast<size_t>(orig_total);
    }
}

} // namespace fz
