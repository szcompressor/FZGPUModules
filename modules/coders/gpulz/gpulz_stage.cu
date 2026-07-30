// The per-chunk stream format (flag bitmap + literal/match tokens) and the
// sequential literal/match parse follow compressKernelI from GPULZ (Zhang,
// Tian, Di, Yu, Swany, Tao, Cappello, ICS '23).
// Upstream: https://github.com/hpdps-group/ICS23-GPULZ — see THIRD_PARTY.md.
// The per-chunk container format, raw-fallback flag, CUB exclusive scan for
// packing offsets, and deferred tail-size readback are FZGM's own, mirroring
// RREStage/RZEStage. So are the exact longest-match search, the BlockScan
// prefix sum and staged data writes in the encode kernel, and the whole
// block-parallel decode kernel (upstream decodes a chunk on one thread).
//
// The all-zero-chunk fast path in gpulzEncodeKernel (the `notEmptyFlag`
// warp-vote skip) is adapted from the "sparse" GPULZ variant in
// boyuanzhang62/AIZ_VLDB26 (test/gpulz.cuh), which applies the same GPULZ
// kernels to the sparse quantized latents produced by a neural compressor.
// Upstream: https://github.com/boyuanzhang62/AIZ_VLDB26 — see THIRD_PARTY.md.

#include "coders/gpulz/gpulz_stage.h"
#include "backend/algorithms.h"
#include "mem/mempool.h"
#include "cuda_check.h"

#include "backend/api.h"
#include "backend/cub.h"
#include "backend/warp.h"
#include "backend/atomics.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace fz {

static constexpr int GPULZ_THREAD_SIZE = 128; // threads per block == symbols processed per iteration
static constexpr int GPULZ_WINDOW_SIZE = 32;  // sliding-window size (upstream default)
// Longest match a single token may encode. Upstream's greedy search could not
// produce a run longer than the window; the exact longest-match search can, and
// a longer run is strictly better (still a 2-byte token). Bounded by the
// token's 1-byte length field.
static constexpr int GPULZ_MAX_MATCH = 255;
// Hashed long-range matcher (match_level 1). The near window is exhaustive and
// exact but costs one comparison per element per offset, so it cannot be widened
// far; a hash lookup finds a candidate in O(1) no matter how far back it is.
static constexpr int GPULZ_HASH_BITS   = 10;  // 1024-entry shared table (2 KB)
static constexpr int GPULZ_HASH_ROUNDS = 8;   // sub-blocks the chunk is walked in
static constexpr int GPULZ_HASH_EARLY  = 16;  // skip lookup if the near window already found this much
static constexpr int GPULZ_MAX_OFFSET  = 255; // limit of the token's 1-byte offset field

// Hash of the two words at `i`. Two words rather than one matters: on quantized
// data most single values are small integers that recur constantly, so a
// one-word table just points at a nearby duplicate the near window already
// covers. Requiring two words makes a hit imply a >=2-word match worth
// extending (measured: ratio 4.70x -> 5.01x at identical throughput).
template <typename T>
static __device__ __forceinline__ uint32_t
gpulzHash(const T* __restrict__ buf, int i, int n)
{
    uint64_t x = (uint64_t)buf[i] * 0x9E3779B97F4A7C15ull;
    if (i + 1 < n)
        x ^= ((uint64_t)buf[i + 1] + 0x165667B19E3779F9ull) * 0xC2B2AE3D27D4EB4Full;
    return (uint32_t)((x * 0xFF51AFD7ED558CCDull) >> (64 - GPULZ_HASH_BITS));
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Encode kernel — one CUDA block per chunk. Same per-chunk stream format as
// upstream GPULZ's compressKernelI, but the match search, the prefix sum and
// the data writes are FZGM's own:
//
//  * Match search. Upstream walks the sliding window one position at a time,
//    advancing the window pointer on every step, so a match it finds consumes
//    the candidate offsets behind it -- it is a greedy approximation, not the
//    longest match. Here an offset-indexed equality mask is built instead
//    (bit o-1 of omask[i] set iff buffer[i] == buffer[i-o]); AND-ing
//    successive masks drops exactly the offsets that just stopped matching,
//    so the extension loop runs (longest_match + 1) times for *all* offsets
//    at once rather than once per candidate offset. That is both cheaper and
//    an exact longest-match search, so it also compresses better.
//  * Prefix sum. cub::BlockScan replaces the hand-rolled Blelloch scan (whose
//    inner loop carried a __syncthreads per level per iteration).
//  * Data writes. Items are emitted into a shared staging buffer at their
//    scanned offsets and then copied out in coalesced 32-bit stores, instead
//    of scattering individual bytes straight to global scratch.
//
// The literal/match parse (the flag bitmap) remains the upstream sequential
// walk on thread 0 -- it is now the largest single cost in this kernel.
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
template <typename T, int CS, int MATCH_LEVEL>
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
    constexpr int IPT        = blockSize / threadSize;

    using BlockScan = cub::BlockScan<uint32_t, threadSize>;

    // omask (match search), prefixBuffer (parse output) and the scan's temp
    // storage are live in disjoint phases, so they share one allocation --
    // without this the largest configuration (chunk_size 4096, word_size 1)
    // would exceed the 48 KB static shared-memory limit.
    __shared__ union {
        uint32_t omask[blockSize];
        uint32_t prefixBuffer[blockSize + 1];
        typename BlockScan::TempStorage scan;
    } shr;
    __shared__ T        buffer[blockSize];
    __shared__ uint8_t  lengthBuffer[blockSize];
    __shared__ uint8_t  offsetBuffer[blockSize];
    __shared__ uint8_t  byteFlagArr[flagBytes];
    __shared__ __align__(16) uint8_t stage[CS]; // coalesced-write staging
    __shared__ int      notEmptyFlag;
    __shared__ uint32_t s_flagCount;
    // Buckets hold position+1 so that 0 means empty and atomicMax keeps the
    // most recent position -- the nearest candidate, and a deterministic one.
    // (A plain store would be correct too, since any bucket occupant is a legal
    // candidate, but the winner would vary run to run and so would the
    // compressed bytes.)
    __shared__ uint32_t ht[MATCH_LEVEL >= 1 ? (1 << GPULZ_HASH_BITS) : 1];

    const T* chunkIn = d_in + (size_t)blockIdx.x * blockSize;

    if (threadIdx.x == 0) notEmptyFlag = 0;
    __syncthreads();

    bool localNonzero = false;
    for (int i = 0; i < IPT; i++) {
        T v = chunkIn[threadIdx.x + threadSize * i];
        buffer[threadIdx.x + threadSize * i] = v;
        localNonzero = localNonzero || (v != T(0));
    }
    // All-zero-chunk fast path (mirrors GPULZ's own "sparse" variant): skip
    // the match search and flag/data encode entirely for chunks that are
    // entirely zero (common in sparse quantized latents) -- flag_size=0 with
    // data_size=0 (no raw-fallback high bit) is the sentinel; the host side
    // memsets the corresponding output span to zero on decode.
    if (fz::backend::anySync32((int)localNonzero) && (threadIdx.x % 32) == 0)
        notEmptyFlag = 1;
    __syncthreads();
    if (notEmptyFlag == 0) {
        if (threadIdx.x == 0) {
            d_flag_size[blockIdx.x] = 0;
            d_data_size[blockIdx.x] = 0;
        }
        return;
    }

    // ── longest match for every element, via the offset-indexed mask ──────
    for (int it = 0; it < IPT; it++) {
        const int i = threadIdx.x + it * threadSize;
        const T   v = buffer[i];
        uint32_t m = 0;
        for (int o = 1; o <= GPULZ_WINDOW_SIZE; o++)
            if (i >= o && buffer[i - o] == v) m |= 1u << (o - 1);
        shr.omask[i] = m;
    }
    __syncthreads();
    for (int it = 0; it < IPT; it++) {
        const int i = threadIdx.x + it * threadSize;
        // Cap at the chunk end so a match can never decode past its chunk,
        // and at 255 so the length still fits the token's length byte.
        int cap = blockSize - i;
        if (cap > GPULZ_MAX_MATCH) cap = GPULZ_MAX_MATCH;
        uint32_t acc = shr.omask[i], prev = 0;
        int len = 0;
        while (acc && len < cap) {
            prev = acc;
            len++;
            if (len < cap) acc &= shr.omask[i + len];
        }
        lengthBuffer[i] = (uint8_t)len;
        offsetBuffer[i] = len ? (uint8_t)__ffs(prev) : 0;
    }

    // ── match_level 1: hashed long-range candidates ───────────────────────
    // The chunk is walked in GPULZ_HASH_ROUNDS sub-blocks. The table only ever
    // holds positions from earlier sub-blocks, so a hit is always a legal
    // back-reference (strictly before the current position) and needs no
    // ordering fixup -- the lookup and the insert for a sub-block are simply
    // separated by a barrier.
    if (MATCH_LEVEL >= 1) {
        constexpr int RND = blockSize / GPULZ_HASH_ROUNDS;
        for (int i = threadIdx.x; i < (1 << GPULZ_HASH_BITS); i += threadSize)
            ht[i] = 0u;
        __syncthreads();

        for (int r = 0; r < GPULZ_HASH_ROUNDS; r++) {
            for (int p = threadIdx.x; p < RND; p += threadSize) {
                const int i = r * RND + p;
                // If the near window already found a long match here, a hash
                // candidate can add little -- skip the verify/extend entirely.
                if ((int)lengthBuffer[i] >= GPULZ_HASH_EARLY) continue;
                const uint32_t slot = ht[gpulzHash(buffer, i, blockSize)];
                if (slot == 0u) continue;
                const int c   = (int)slot - 1;
                const int off = i - c;
                if (off <= 0 || off > GPULZ_MAX_OFFSET) continue;
                int cap = blockSize - i;
                if (cap > GPULZ_MAX_MATCH) cap = GPULZ_MAX_MATCH;
                int k = 0;
                while (k < cap && buffer[i + k] == buffer[c + k]) k++;
                if (k > (int)lengthBuffer[i]) {
                    lengthBuffer[i] = (uint8_t)k;
                    offsetBuffer[i] = (uint8_t)off;
                }
            }
            __syncthreads();
            for (int p = threadIdx.x; p < RND; p += threadSize) {
                const int i = r * RND + p;
                fz::backend::atomicMaxBlock(&ht[gpulzHash(buffer, i, blockSize)],
                                            (uint32_t)(i + 1));
            }
            __syncthreads();
        }
    }
    __syncthreads();

    // Elements skipped by a match contribute no bytes; seed every slot to 0
    // so the parse below only has to fill in the items it actually emits.
    for (int it = 0; it < IPT; it++) shr.prefixBuffer[threadIdx.x + it * threadSize] = 0;
    __syncthreads();

    // build the literal/match flag bitmap and per-item byte sizes
    // (inherently sequential — matches upstream, only thread 0 performs it)
    if (threadIdx.x == 0) {
        uint32_t flagCount   = 0;
        uint8_t  flagPosition = 0x01;
        uint8_t  byteFlag     = 0;
        int      encodeIndex  = 0;

        while (encodeIndex < blockSize) {
            if (lengthBuffer[encodeIndex] < minEncodeLength) {
                shr.prefixBuffer[encodeIndex] = sizeof(T);
                encodeIndex++;
            } else {
                shr.prefixBuffer[encodeIndex] = 2;
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
        s_flagCount = flagCount;
    }
    __syncthreads();

    // ── exclusive block scan of per-item byte sizes -> packing offsets ────
    uint32_t sz[IPT], off[IPT];
    for (int it = 0; it < IPT; it++) sz[it] = shr.prefixBuffer[threadIdx.x * IPT + it];
    const uint32_t flagCount = s_flagCount;
    uint32_t total = 0;
    __syncthreads();   // prefixBuffer is dead now; the scan reuses its storage
    BlockScan(shr.scan).ExclusiveSum(sz, off, total);
    if (threadIdx.x == 0) {
        d_data_size[blockIdx.x] = total;
        d_flag_size[blockIdx.x] = flagCount;
    }

    // ── emit into shared staging, then copy out in coalesced words ────────
    for (int it = 0; it < IPT; it++) {
        if (sz[it] == 0) continue;
        const int      i = threadIdx.x * IPT + it;
        const uint32_t o = off[it];
        if (lengthBuffer[i] < minEncodeLength) {
            const uint8_t* bytePtr = (const uint8_t*)&buffer[i];
            for (unsigned b = 0; b < sizeof(T); b++) stage[o + b] = bytePtr[b];
        } else {
            stage[o]     = lengthBuffer[i];
            stage[o + 1] = offsetBuffer[i];
        }
    }
    __syncthreads();

    uint8_t* chunkData = d_data_scratch + (size_t)blockIdx.x * CS;
    const uint32_t words = (total + 3) >> 2;
    for (uint32_t w = threadIdx.x; w < words; w += threadSize)
        ((uint32_t*)chunkData)[w] = ((const uint32_t*)stage)[w];

    uint8_t* chunkFlag = d_flag_scratch + (size_t)blockIdx.x * flagBytes;
    for (uint32_t i = threadIdx.x; i < flagCount; i += threadSize)
        chunkFlag[i] = byteFlagArr[i];
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Decode kernel — one CUDA *block* per chunk (FZGM's own; the upstream GPULZ
// decompressKernel decodes a whole chunk on a single thread).
//
// The serial walk exists only because each item's position in the stream
// appears to depend on every item before it. It does not: the flag bitmap
// alone fixes every item's *input* size (2 bytes for a match token,
// sizeof(T) for a literal), so a block scan recovers all input offsets in
// one shot; reading each match's length byte at its now-known input offset
// gives every item's *output* length, and a second block scan recovers all
// output positions. The only remaining serial structure is back-references
// chaining onto other back-references, which is resolved by pointer doubling
// over a per-element source-index array (log2(blockSize) rounds, all
// parallel) rather than by replaying the stream in order.
//
//   1. scan item input sizes   -> input offset of every item
//   2. read match length bytes -> output length of every item
//   3. scan output lengths     -> output position of every item
//   4. scatter a source index per output element:
//        literal element -> itself (terminal, value already materialized)
//        match element q -> q - offset  (always < q, so chains terminate)
//   5. pointer-double src[q] = src[src[q]] until every chain lands on a
//      literal (early-exit once no chain moved)
//   6. gather out[q] = literal_value[src[q]]
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
static constexpr int GPULZ_DECODE_TPB = 128;

template <typename T, int CS>
static __global__ void __launch_bounds__(GPULZ_DECODE_TPB)
gpulzDecodeKernel(
    T*              __restrict__ d_out,
    const uint8_t*  __restrict__ d_in,          // packed payload base
    const uint32_t* __restrict__ d_in_offsets,  // per-chunk offset into payload
    const uint32_t* __restrict__ d_flag_size,
    const uint32_t* __restrict__ d_data_size,
    uint32_t num_chunks)
{
    constexpr int N   = CS / sizeof(T);           // elements per chunk
    constexpr int TPB = GPULZ_DECODE_TPB;
    constexpr int IPT = N / TPB;                  // items (and elements) per thread

    const uint32_t cid = blockIdx.x;
    if (cid >= num_chunks) return;

    const uint32_t flag_sz = d_flag_size[cid];
    // flag_size 0 marks a raw or all-zero chunk; gpulzDecodePassthroughKernel
    // has already produced its output.
    if (flag_sz == 0) return;

    const uint32_t data_sz = d_data_size[cid];
    const uint8_t* chunkIn  = d_in + d_in_offsets[cid];
    const uint8_t* flagArr  = chunkIn;

    using BlockScan = cub::BlockScan<uint32_t, TPB>;
    __shared__ typename BlockScan::TempStorage scan_tmp;
    __shared__ uint8_t  cdata[CS + 8];    // this chunk's literal/match byte stream (+pad)
    __shared__ T        litval[N];        // materialized literal values
    __shared__ uint16_t srcA[N], srcB[N]; // pointer-doubling source indices
    __shared__ int      s_moved;

    // Stage the byte stream into shared: the per-item reads below are random
    // access, and every item's bytes are re-read during the scatter.
    {
        const uint8_t* compData = chunkIn + flag_sz;
        for (uint32_t i = threadIdx.x; i < data_sz; i += TPB) cdata[i] = compData[i];
    }
    // A well-formed stream covers every output element exactly once; seed the
    // arrays anyway so a corrupt stream yields zeros rather than out-of-range
    // shared reads during the pointer doubling below.
    for (int q = threadIdx.x; q < N; q += TPB) { srcA[q] = (uint16_t)q; litval[q] = T(0); }
    __syncthreads();

    // ── 1. item input sizes -> input offsets ──────────────────────────────
    // Item j is bit j of the flag bitmap. The bitmap's last byte is zero-padded,
    // so trailing "phantom" items may appear; they sort after every real item
    // and are dropped in step 4 by the out_pos >= N test.
    const uint32_t n_items = (flag_sz * 8u < (uint32_t)N) ? flag_sz * 8u : (uint32_t)N;
    uint32_t in_sz[IPT], in_off[IPT], out_len[IPT], out_pos[IPT];
    bool     is_match[IPT];

    for (int t = 0; t < IPT; t++) {
        const uint32_t j = threadIdx.x * IPT + t;
        is_match[t] = (j < n_items) && ((flagArr[j >> 3] >> (j & 7)) & 1u);
        in_sz[t]    = (j < n_items) ? (is_match[t] ? 2u : (uint32_t)sizeof(T)) : 0u;
    }
    BlockScan(scan_tmp).ExclusiveSum(in_sz, in_off);
    __syncthreads();

    // ── 2/3. output lengths -> output positions ───────────────────────────
    for (int t = 0; t < IPT; t++) {
        if (in_sz[t] == 0 || in_off[t] >= data_sz) { out_len[t] = 0; continue; }
        out_len[t] = is_match[t] ? (uint32_t)cdata[in_off[t]] : 1u;
    }
    BlockScan(scan_tmp).ExclusiveSum(out_len, out_pos);
    __syncthreads();

    // ── 4. scatter source indices / materialize literals ──────────────────
    for (int t = 0; t < IPT; t++) {
        if (out_len[t] == 0 || out_pos[t] >= (uint32_t)N) continue;
        const uint32_t p = out_pos[t];
        if (is_match[t]) {
            const uint32_t off = cdata[in_off[t] + 1];
            uint32_t len = out_len[t];
            if (p + len > (uint32_t)N) len = (uint32_t)N - p;   // defensive clamp
            for (uint32_t k = 0; k < len; k++) srcA[p + k] = (uint16_t)(p + k - off);
        } else {
            T v;
            #pragma unroll
            for (unsigned b = 0; b < sizeof(T); b++)
                ((uint8_t*)&v)[b] = cdata[in_off[t] + b];
            litval[p] = v;
            srcA[p]   = (uint16_t)p;    // terminal: a literal is its own source
        }
    }
    __syncthreads();

    // ── 5. pointer doubling until every chain lands on a literal ──────────
    uint16_t* cur = srcA;
    uint16_t* nxt = srcB;
    for (int round = 0; round < 32; round++) {
        if (threadIdx.x == 0) s_moved = 0;
        __syncthreads();
        int moved = 0;
        for (int t = 0; t < IPT; t++) {
            const int q = threadIdx.x * IPT + t;
            const uint16_t s = cur[q];
            const uint16_t s2 = cur[s];
            nxt[q] = s2;
            moved |= (s2 != s);
        }
        if (fz::backend::anySync32(moved) && (threadIdx.x & 31) == 0) s_moved = 1;
        __syncthreads();
        uint16_t* tmp = cur; cur = nxt; nxt = tmp;
        if (!s_moved) break;
        __syncthreads();
    }

    // ── 6. gather ─────────────────────────────────────────────────────────
    T* out = d_out + (size_t)cid * N;
    for (int t = 0; t < IPT; t++) {
        const int q = threadIdx.x * IPT + t;
        out[q] = litval[cur[q]];
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

// Handles the two decode-side passthrough cases (raw-fallback copy, and
// all-zero-chunk fill) for every chunk in one launch, keyed on a per-chunk
// mode byte (0=normal chunk, handled separately by the decode kernel;
// 1=raw, copy chunk_bytes verbatim from the input offset; 2=empty, zero-fill
// chunk_bytes). Batching this into a single kernel (instead of a host loop
// issuing one cudaMemcpyAsync/cudaMemsetAsync per matching chunk) matters
// because empty chunks are the common case for sparse inputs -- a host loop
// over thousands of chunks turns into thousands of tiny async launches and
// dominates decode time (observed: >10x decode slowdown on Lorenzo-residual
// data, where whole chunks are frequently exactly zero).
static __global__ void gpulzDecodePassthroughKernel(
    const uint8_t*  __restrict__ d_in,
    uint8_t*        __restrict__ d_out,
    const uint32_t* __restrict__ d_in_offsets,
    const uint8_t*  __restrict__ d_mode,
    uint32_t chunk_bytes)
{
    const uint32_t cid  = blockIdx.x;
    const uint8_t  mode = d_mode[cid];
    if (mode == 0) return;

    uint8_t* out = d_out + (size_t)cid * chunk_bytes;
    if (mode == 1) {
        const uint8_t* in = d_in + d_in_offsets[cid];
        for (uint32_t i = threadIdx.x; i < chunk_bytes; i += blockDim.x)
            out[i] = in[i];
    } else { // mode == 2: all-zero chunk
        for (uint32_t i = threadIdx.x; i < chunk_bytes; i += blockDim.x)
            out[i] = 0;
    }
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

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Split mode (Zstd-style literals/sequences separation)
//
// The encode kernel is untouched: it still produces a per-chunk flag bitmap
// plus one interleaved literal/match byte stream. Split mode replaces only the
// final pack step, scattering those bytes into four independent streams so
// each can be entropy coded against its own symbol distribution.
//
// Per-chunk destination offsets come from three exclusive scans over per-chunk
// counts. The counts are derived without walking the item list: a chunk's
// match count is exactly the popcount of its flag bitmap (bitmap padding bits
// are zero, so they never inflate it), and its literal byte count then follows
// from `data_size = n_literals*word_size + n_matches*2`.
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

// meta stream holds one flag bitmap per *encoded* chunk; raw and all-zero
// chunks contribute none. Derivable from the size table alone, which is why it
// is a separate pass from the popcount below (the popcount needs the bitmap,
// whose location needs this scan).
static __global__ void gpulzSplitMetaCountKernel(
    const uint32_t* __restrict__ d_flag_size,
    const uint32_t* __restrict__ d_data_size,
    uint32_t*       __restrict__ d_meta_cnt,
    uint32_t n_chunks)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_chunks) return;
    const uint32_t fe = d_flag_size[i];
    const bool raw = (fe & 0x80000000u) != 0;
    d_meta_cnt[i] = (raw || (fe == 0 && d_data_size[i] == 0)) ? 0u : fe;
}

// Per-chunk literal-byte and match-token counts. `d_flag_at` gives each chunk's
// bitmap position: null means the uniform forward scratch stride, non-null the
// packed meta-stream offsets used on the inverse path.
static __global__ void gpulzSplitCountKernel(
    const uint8_t*  __restrict__ d_flags_base,
    const uint32_t* __restrict__ d_flag_at,     // may be null
    uint32_t flag_stride,
    const uint32_t* __restrict__ d_flag_size,
    const uint32_t* __restrict__ d_data_size,
    uint32_t*       __restrict__ d_lit_cnt,
    uint32_t*       __restrict__ d_tok_cnt,
    uint32_t n_chunks, uint32_t chunk_bytes)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_chunks) return;

    const uint32_t fe = d_flag_size[i];
    const uint32_t ds = d_data_size[i];

    if ((fe & 0x80000000u) != 0) {          // raw fallback: all literal
        d_lit_cnt[i] = chunk_bytes;
        d_tok_cnt[i] = 0;
        return;
    }
    if (fe == 0 && ds == 0) {               // all-zero chunk: contributes nothing
        d_lit_cnt[i] = 0;
        d_tok_cnt[i] = 0;
        return;
    }

    const uint8_t* flags = d_flags_base
        + (d_flag_at ? d_flag_at[i] : (size_t)i * flag_stride);
    uint32_t n_match = 0;
    for (uint32_t b = 0; b < fe; b++) n_match += (uint32_t)__popc((unsigned)flags[b]);

    d_tok_cnt[i] = n_match;
    d_lit_cnt[i] = ds - 2u * n_match;
}

static __global__ void gpulzSplitTotalsKernel(
    const uint32_t* __restrict__ lit_off, const uint32_t* __restrict__ lit_cnt,
    const uint32_t* __restrict__ tok_off, const uint32_t* __restrict__ tok_cnt,
    const uint32_t* __restrict__ meta_off, const uint32_t* __restrict__ meta_cnt,
    uint32_t* __restrict__ totals, uint32_t last)
{
    if (blockIdx.x || threadIdx.x) return;
    totals[0] = lit_off[last]  + lit_cnt[last];
    totals[1] = tok_off[last]  + tok_cnt[last];
    totals[2] = totals[1];
    totals[3] = meta_off[last] + meta_cnt[last];
}

// Scatter one chunk's items into the four streams. Item input offsets come
// from the same block scan the parallel decoder uses; two further scans give
// each item its slot in the literals and token streams.
template <typename T, int CS>
static __global__ void __launch_bounds__(GPULZ_DECODE_TPB)
gpulzDestripeKernel(
    const uint8_t*  __restrict__ d_flag_scratch,
    const uint8_t*  __restrict__ d_data_scratch,
    const uint8_t*  __restrict__ d_in_raw,
    const uint32_t* __restrict__ d_flag_size,
    const uint32_t* __restrict__ d_data_size,
    const uint32_t* __restrict__ d_lit_off,
    const uint32_t* __restrict__ d_tok_off,
    const uint32_t* __restrict__ d_meta_off,
    uint8_t* __restrict__ d_lit, uint8_t* __restrict__ d_len,
    uint8_t* __restrict__ d_off, uint8_t* __restrict__ d_meta,
    uint32_t flag_stride, uint32_t n_chunks)
{
    constexpr int N   = CS / sizeof(T);
    constexpr int TPB = GPULZ_DECODE_TPB;
    constexpr int IPT = N / TPB;

    const uint32_t cid = blockIdx.x;
    if (cid >= n_chunks) return;

    const uint32_t fe = d_flag_size[cid];
    const uint32_t ds = d_data_size[cid];

    if ((fe & 0x80000000u) != 0) {   // raw chunk -> literals stream verbatim
        const uint8_t* src = d_in_raw + (size_t)cid * CS;
        uint8_t*       dst = d_lit + d_lit_off[cid];
        for (uint32_t i = threadIdx.x; i < (uint32_t)CS; i += TPB) dst[i] = src[i];
        return;
    }
    if (fe == 0 && ds == 0) return;  // all-zero chunk

    const uint8_t* flags = d_flag_scratch + (size_t)cid * flag_stride;
    const uint8_t* data  = d_data_scratch + (size_t)cid * CS;

    for (uint32_t i = threadIdx.x; i < fe; i += TPB)
        d_meta[d_meta_off[cid] + i] = flags[i];

    using BlockScan = cub::BlockScan<uint32_t, TPB>;
    __shared__ typename BlockScan::TempStorage scan_tmp;

    // Trailing bits of the last bitmap byte are zero padding; they look like
    // literals but land at in_off >= data_size and are dropped below, which
    // keeps this consistent with the popcount-derived counts.
    const uint32_t n_items = (fe * 8u < (uint32_t)N) ? fe * 8u : (uint32_t)N;
    uint32_t in_sz[IPT], in_off[IPT], litw[IPT], litpos[IPT], tok[IPT], tokpos[IPT];
    bool is_match[IPT];

    for (int t = 0; t < IPT; t++) {
        const uint32_t j = threadIdx.x * IPT + t;
        is_match[t] = (j < n_items) && ((flags[j >> 3] >> (j & 7)) & 1u);
        in_sz[t]    = (j < n_items) ? (is_match[t] ? 2u : (uint32_t)sizeof(T)) : 0u;
    }
    BlockScan(scan_tmp).ExclusiveSum(in_sz, in_off);
    __syncthreads();

    for (int t = 0; t < IPT; t++) {
        const bool valid = (in_sz[t] != 0) && (in_off[t] < ds);
        litw[t] = (valid && !is_match[t]) ? (uint32_t)sizeof(T) : 0u;
        tok[t]  = (valid &&  is_match[t]) ? 1u : 0u;
    }
    BlockScan(scan_tmp).ExclusiveSum(litw, litpos);
    __syncthreads();
    BlockScan(scan_tmp).ExclusiveSum(tok, tokpos);
    __syncthreads();

    uint8_t* lit = d_lit + d_lit_off[cid];
    uint8_t* len = d_len + d_tok_off[cid];
    uint8_t* off = d_off + d_tok_off[cid];

    for (int t = 0; t < IPT; t++) {
        if (in_sz[t] == 0 || in_off[t] >= ds) continue;
        if (is_match[t]) {
            len[tokpos[t]] = data[in_off[t]];
            off[tokpos[t]] = data[in_off[t] + 1];
        } else {
            #pragma unroll
            for (unsigned b = 0; b < sizeof(T); b++)
                lit[litpos[t] + b] = data[in_off[t] + b];
        }
    }
}

// Inverse of the above: rebuild the packed single-stream form (per chunk:
// flag bitmap followed by the interleaved literal/match bytes) so the existing
// decode path can run unchanged.
template <typename T, int CS>
static __global__ void __launch_bounds__(GPULZ_DECODE_TPB)
gpulzRestripeKernel(
    const uint8_t*  __restrict__ d_lit, const uint8_t* __restrict__ d_len,
    const uint8_t*  __restrict__ d_off, const uint8_t* __restrict__ d_meta,
    const uint32_t* __restrict__ d_flag_size,
    const uint32_t* __restrict__ d_data_size,
    const uint32_t* __restrict__ d_lit_off,
    const uint32_t* __restrict__ d_tok_off,
    const uint32_t* __restrict__ d_meta_off,
    const uint32_t* __restrict__ d_dst_off,
    uint8_t* __restrict__ d_out, uint32_t n_chunks)
{
    constexpr int N   = CS / sizeof(T);
    constexpr int TPB = GPULZ_DECODE_TPB;
    constexpr int IPT = N / TPB;

    const uint32_t cid = blockIdx.x;
    if (cid >= n_chunks) return;

    const uint32_t fe = d_flag_size[cid];
    const uint32_t ds = d_data_size[cid];
    uint8_t* dst = d_out + d_dst_off[cid];

    if ((fe & 0x80000000u) != 0) {   // raw chunk: literals stream holds it
        const uint8_t* src = d_lit + d_lit_off[cid];
        for (uint32_t i = threadIdx.x; i < (uint32_t)CS; i += TPB) dst[i] = src[i];
        return;
    }
    if (fe == 0 && ds == 0) return;

    const uint8_t* flags = d_meta + d_meta_off[cid];
    for (uint32_t i = threadIdx.x; i < fe; i += TPB) dst[i] = flags[i];

    using BlockScan = cub::BlockScan<uint32_t, TPB>;
    __shared__ typename BlockScan::TempStorage scan_tmp;

    const uint32_t n_items = (fe * 8u < (uint32_t)N) ? fe * 8u : (uint32_t)N;
    uint32_t in_sz[IPT], in_off[IPT], litw[IPT], litpos[IPT], tok[IPT], tokpos[IPT];
    bool is_match[IPT];

    for (int t = 0; t < IPT; t++) {
        const uint32_t j = threadIdx.x * IPT + t;
        is_match[t] = (j < n_items) && ((flags[j >> 3] >> (j & 7)) & 1u);
        in_sz[t]    = (j < n_items) ? (is_match[t] ? 2u : (uint32_t)sizeof(T)) : 0u;
    }
    BlockScan(scan_tmp).ExclusiveSum(in_sz, in_off);
    __syncthreads();

    for (int t = 0; t < IPT; t++) {
        const bool valid = (in_sz[t] != 0) && (in_off[t] < ds);
        litw[t] = (valid && !is_match[t]) ? (uint32_t)sizeof(T) : 0u;
        tok[t]  = (valid &&  is_match[t]) ? 1u : 0u;
    }
    BlockScan(scan_tmp).ExclusiveSum(litw, litpos);
    __syncthreads();
    BlockScan(scan_tmp).ExclusiveSum(tok, tokpos);
    __syncthreads();

    const uint8_t* lit = d_lit + d_lit_off[cid];
    const uint8_t* len = d_len + d_tok_off[cid];
    const uint8_t* off = d_off + d_tok_off[cid];
    uint8_t* data = dst + fe;

    for (int t = 0; t < IPT; t++) {
        if (in_sz[t] == 0 || in_off[t] >= ds) continue;
        if (is_match[t]) {
            data[in_off[t]]     = len[tokpos[t]];
            data[in_off[t] + 1] = off[tokpos[t]];
        } else {
            #pragma unroll
            for (unsigned b = 0; b < sizeof(T); b++)
                data[in_off[t] + b] = lit[litpos[t] + b];
        }
    }
}

// Packed-stream size of each chunk, for the inverse path. Unlike
// gpulzFinalizeSizesKernel this only *reads* the raw-fallback flag (the
// decision was already made and recorded at encode time).
static __global__ void gpulzCleanSizesKernel(
    const uint32_t* __restrict__ d_flag_size,
    const uint32_t* __restrict__ d_data_size,
    uint32_t*       __restrict__ d_clean,
    uint32_t n_chunks, uint32_t chunk_bytes)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_chunks) return;
    const uint32_t fe = d_flag_size[i];
    if ((fe & 0x80000000u) != 0)          d_clean[i] = chunk_bytes;
    else if (fe == 0 && d_data_size[i] == 0) d_clean[i] = 0;
    else                                  d_clean[i] = fe + d_data_size[i];
}

// Fused split decode: decode straight out of the four ports, with no
// intermediate packed stream.
//
// The restripe-then-decode pair it replaces did the same block scans twice --
// once to rebuild the packed per-chunk byte stream, once to re-parse it -- and
// moved the whole uncompressed-size intermediate through global memory both
// ways. Reading the ports directly also removes the shared-memory staging of
// that byte stream, since literals and tokens are now fetched from their own
// arrays at scan-derived indices.
template <typename T, int CS>
static __global__ void __launch_bounds__(GPULZ_DECODE_TPB)
gpulzSplitDecodeKernel(
    const uint8_t*  __restrict__ d_lit, const uint8_t* __restrict__ d_len,
    const uint8_t*  __restrict__ d_off, const uint8_t* __restrict__ d_meta,
    const uint32_t* __restrict__ d_flag_size,
    const uint32_t* __restrict__ d_data_size,
    const uint32_t* __restrict__ d_lit_off,
    const uint32_t* __restrict__ d_tok_off,
    const uint32_t* __restrict__ d_meta_off,
    T* __restrict__ d_out, uint32_t n_chunks)
{
    constexpr int N   = CS / sizeof(T);
    constexpr int TPB = GPULZ_DECODE_TPB;
    constexpr int IPT = N / TPB;

    const uint32_t cid = blockIdx.x;
    if (cid >= n_chunks) return;

    const uint32_t fe = d_flag_size[cid];
    const uint32_t ds = d_data_size[cid];
    T* out = d_out + (size_t)cid * N;

    if ((fe & 0x80000000u) != 0) {   // raw chunk: verbatim from the literals port
        const uint8_t* src = d_lit + d_lit_off[cid];
        uint8_t*       dst = (uint8_t*)out;
        for (uint32_t i = threadIdx.x; i < (uint32_t)CS; i += TPB) dst[i] = src[i];
        return;
    }
    if (fe == 0 && ds == 0) {        // all-zero chunk
        for (int q = threadIdx.x; q < N; q += TPB) out[q] = T(0);
        return;
    }

    const uint8_t* flags = d_meta + d_meta_off[cid];
    const uint8_t* lit   = d_lit  + d_lit_off[cid];
    const uint8_t* len   = d_len  + d_tok_off[cid];
    const uint8_t* offs  = d_off  + d_tok_off[cid];

    using BlockScan = cub::BlockScan<uint32_t, TPB>;
    __shared__ typename BlockScan::TempStorage scan_tmp;
    __shared__ T        litval[N];
    __shared__ uint16_t srcA[N], srcB[N];
    __shared__ int      s_moved;

    for (int q = threadIdx.x; q < N; q += TPB) { srcA[q] = (uint16_t)q; litval[q] = T(0); }
    __syncthreads();

    const uint32_t n_items = (fe * 8u < (uint32_t)N) ? fe * 8u : (uint32_t)N;
    uint32_t in_sz[IPT], in_off[IPT], litw[IPT], litpos[IPT];
    uint32_t tok[IPT], tokpos[IPT], out_len[IPT], out_pos[IPT];
    bool is_match[IPT];

    for (int t = 0; t < IPT; t++) {
        const uint32_t j = threadIdx.x * IPT + t;
        is_match[t] = (j < n_items) && ((flags[j >> 3] >> (j & 7)) & 1u);
        in_sz[t]    = (j < n_items) ? (is_match[t] ? 2u : (uint32_t)sizeof(T)) : 0u;
    }
    // in_off is still needed: it is what identifies the zero-padding bits in the
    // bitmap's last byte, which look like literals but address past data_size.
    BlockScan(scan_tmp).ExclusiveSum(in_sz, in_off);
    __syncthreads();

    for (int t = 0; t < IPT; t++) {
        const bool valid = (in_sz[t] != 0) && (in_off[t] < ds);
        litw[t] = (valid && !is_match[t]) ? (uint32_t)sizeof(T) : 0u;
        tok[t]  = (valid &&  is_match[t]) ? 1u : 0u;
    }
    BlockScan(scan_tmp).ExclusiveSum(litw, litpos);
    __syncthreads();
    BlockScan(scan_tmp).ExclusiveSum(tok, tokpos);
    __syncthreads();

    for (int t = 0; t < IPT; t++) {
        if (in_sz[t] == 0 || in_off[t] >= ds) { out_len[t] = 0; continue; }
        out_len[t] = is_match[t] ? (uint32_t)len[tokpos[t]] : 1u;
    }
    BlockScan(scan_tmp).ExclusiveSum(out_len, out_pos);
    __syncthreads();

    for (int t = 0; t < IPT; t++) {
        if (out_len[t] == 0 || out_pos[t] >= (uint32_t)N) continue;
        const uint32_t p = out_pos[t];
        if (is_match[t]) {
            const uint32_t o = offs[tokpos[t]];
            uint32_t l = out_len[t];
            if (p + l > (uint32_t)N) l = (uint32_t)N - p;   // defensive clamp
            for (uint32_t k = 0; k < l; k++) srcA[p + k] = (uint16_t)(p + k - o);
        } else {
            T v;
            #pragma unroll
            for (unsigned b = 0; b < sizeof(T); b++)
                ((uint8_t*)&v)[b] = lit[litpos[t] + b];
            litval[p] = v;
            srcA[p]   = (uint16_t)p;
        }
    }
    __syncthreads();

    uint16_t* cur = srcA;
    uint16_t* nxt = srcB;
    for (int round = 0; round < 32; round++) {
        if (threadIdx.x == 0) s_moved = 0;
        __syncthreads();
        int moved = 0;
        for (int t = 0; t < IPT; t++) {
            const int q = threadIdx.x * IPT + t;
            const uint16_t sv  = cur[q];
            const uint16_t sv2 = cur[sv];
            nxt[q] = sv2;
            moved |= (sv2 != sv);
        }
        if (fz::backend::anySync32(moved) && (threadIdx.x & 31) == 0) s_moved = 1;
        __syncthreads();
        uint16_t* tmp = cur; cur = nxt; nxt = tmp;
        if (!s_moved) break;
        __syncthreads();
    }

    for (int t = 0; t < IPT; t++) {
        const int q = threadIdx.x * IPT + t;
        out[q] = litval[cur[q]];
    }
}

// Rebuilds the per-chunk size arrays from the meta stream's entry table.
static __global__ void gpulzDeinterleaveHeaderKernel(
    const uint32_t* __restrict__ d_hdr_entries,
    uint32_t* __restrict__ d_flag_size,
    uint32_t* __restrict__ d_data_size,
    uint32_t n_chunks)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_chunks) return;
    d_flag_size[i] = d_hdr_entries[2 * i];
    d_data_size[i] = d_hdr_entries[2 * i + 1];
}

// ━━━━ word-size × chunk-size dispatch helpers ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// chunk_size selects the compile-time CS template argument; word_size
// selects T. Supported chunk sizes: 1024, 2048, 4096 (see GPULZStage::execute()).
static void launchEncode(uint8_t word_size, uint32_t chunk_size, int n_chunks, cudaStream_t stream,
                          const uint8_t* d_in, uint8_t* d_flag_scratch, uint8_t* d_data_scratch,
                          uint32_t* d_flag_size, uint32_t* d_data_size, uint8_t match_level)
{
    const int minEncodeLength = (word_size == 1) ? 2 : 1;
#define FZ_GPULZ_ENCODE_LAUNCH(T_VAL, CS_VAL, ML_VAL)                                                                     \
    gpulzEncodeKernel<T_VAL, CS_VAL, ML_VAL><<<n_chunks, GPULZ_THREAD_SIZE, 0, stream>>>(                                  \
        (const T_VAL*)d_in, d_flag_scratch, d_data_scratch, d_flag_size, d_data_size, minEncodeLength)
#define FZ_GPULZ_ENCODE_WS(CS_VAL, ML_VAL)                                                                                \
    switch (word_size) {                                                                                                  \
        case 1: FZ_GPULZ_ENCODE_LAUNCH(uint8_t,  CS_VAL, ML_VAL); break;                                                  \
        case 2: FZ_GPULZ_ENCODE_LAUNCH(uint16_t, CS_VAL, ML_VAL); break;                                                  \
        case 4: FZ_GPULZ_ENCODE_LAUNCH(uint32_t, CS_VAL, ML_VAL); break;                                                  \
        case 8: FZ_GPULZ_ENCODE_LAUNCH(uint64_t, CS_VAL, ML_VAL); break;                                                  \
        default: throw std::runtime_error("GPULZStage: word_size must be 1, 2, 4, or 8");                                  \
    }
#define FZ_GPULZ_ENCODE_CASE(CS_VAL)                                                                                      \
    case CS_VAL:                                                                                                          \
        if (match_level == 0) { FZ_GPULZ_ENCODE_WS(CS_VAL, 0) } else { FZ_GPULZ_ENCODE_WS(CS_VAL, 1) }                     \
        break;
    switch (chunk_size) {
        FZ_GPULZ_ENCODE_CASE(1024)
        FZ_GPULZ_ENCODE_CASE(2048)
        FZ_GPULZ_ENCODE_CASE(4096)
        default: throw std::runtime_error("GPULZStage: chunk_size must be 1024, 2048, or 4096");
    }
#undef FZ_GPULZ_ENCODE_CASE
#undef FZ_GPULZ_ENCODE_WS
#undef FZ_GPULZ_ENCODE_LAUNCH
}

static void launchDecode(uint8_t word_size, uint32_t chunk_size, int n_chunks, cudaStream_t stream,
                          uint8_t* d_out, const uint8_t* d_in, const uint32_t* d_in_off,
                          const uint32_t* d_flag_size, const uint32_t* d_data_size)
{
    // One block per chunk: gpulzDecodeKernel decodes a chunk cooperatively.
    constexpr int kDecodeTPB = GPULZ_DECODE_TPB;
    const int grid = n_chunks;
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

struct GpulzSplitPtrs {
    const uint8_t*  flag_scratch; const uint8_t* data_scratch; const uint8_t* in_raw;
    const uint32_t* flag_size;    const uint32_t* data_size;
    const uint32_t* lit_off;      const uint32_t* tok_off;     const uint32_t* meta_off;
    uint8_t* lit; uint8_t* len; uint8_t* off; uint8_t* meta;
    uint32_t flag_stride;
};

static void launchDestripe(uint8_t word_size, uint32_t chunk_size, int n_chunks,
                           cudaStream_t stream, const GpulzSplitPtrs& p)
{
#define FZ_GPULZ_DESTRIPE_LAUNCH(T_VAL, CS_VAL)                                             \
    gpulzDestripeKernel<T_VAL, CS_VAL><<<n_chunks, GPULZ_DECODE_TPB, 0, stream>>>(          \
        p.flag_scratch, p.data_scratch, p.in_raw, p.flag_size, p.data_size,                 \
        p.lit_off, p.tok_off, p.meta_off, p.lit, p.len, p.off, p.meta,                      \
        p.flag_stride, (uint32_t)n_chunks)
#define FZ_GPULZ_DESTRIPE_CASE(CS_VAL)                                                      \
    case CS_VAL:                                                                            \
        switch (word_size) {                                                                \
            case 1: FZ_GPULZ_DESTRIPE_LAUNCH(uint8_t,  CS_VAL); break;                      \
            case 2: FZ_GPULZ_DESTRIPE_LAUNCH(uint16_t, CS_VAL); break;                      \
            case 4: FZ_GPULZ_DESTRIPE_LAUNCH(uint32_t, CS_VAL); break;                      \
            case 8: FZ_GPULZ_DESTRIPE_LAUNCH(uint64_t, CS_VAL); break;                      \
            default: throw std::runtime_error("GPULZStage: word_size must be 1, 2, 4, or 8"); \
        }                                                                                    \
        break;
    switch (chunk_size) {
        FZ_GPULZ_DESTRIPE_CASE(1024)
        FZ_GPULZ_DESTRIPE_CASE(2048)
        FZ_GPULZ_DESTRIPE_CASE(4096)
        default: throw std::runtime_error("GPULZStage: chunk_size must be 1024, 2048, or 4096");
    }
#undef FZ_GPULZ_DESTRIPE_CASE
#undef FZ_GPULZ_DESTRIPE_LAUNCH
}

struct GpulzSplitDecodePtrs {
    const uint8_t*  lit; const uint8_t* len; const uint8_t* off; const uint8_t* meta;
    const uint32_t* flag_size; const uint32_t* data_size;
    const uint32_t* lit_off;   const uint32_t* tok_off; const uint32_t* meta_off;
    uint8_t* out;
};

static void launchSplitDecode(uint8_t word_size, uint32_t chunk_size, int n_chunks,
                              cudaStream_t stream, const GpulzSplitDecodePtrs& p)
{
#define FZ_GPULZ_SPLITDEC_LAUNCH(T_VAL, CS_VAL)                                             \
    gpulzSplitDecodeKernel<T_VAL, CS_VAL><<<n_chunks, GPULZ_DECODE_TPB, 0, stream>>>(       \
        p.lit, p.len, p.off, p.meta, p.flag_size, p.data_size,                              \
        p.lit_off, p.tok_off, p.meta_off, (T_VAL*)p.out, (uint32_t)n_chunks)
#define FZ_GPULZ_SPLITDEC_CASE(CS_VAL)                                                      \
    case CS_VAL:                                                                            \
        switch (word_size) {                                                                \
            case 1: FZ_GPULZ_SPLITDEC_LAUNCH(uint8_t,  CS_VAL); break;                      \
            case 2: FZ_GPULZ_SPLITDEC_LAUNCH(uint16_t, CS_VAL); break;                      \
            case 4: FZ_GPULZ_SPLITDEC_LAUNCH(uint32_t, CS_VAL); break;                      \
            case 8: FZ_GPULZ_SPLITDEC_LAUNCH(uint64_t, CS_VAL); break;                      \
            default: throw std::runtime_error("GPULZStage: word_size must be 1, 2, 4, or 8"); \
        }                                                                                    \
        break;
    switch (chunk_size) {
        FZ_GPULZ_SPLITDEC_CASE(1024)
        FZ_GPULZ_SPLITDEC_CASE(2048)
        FZ_GPULZ_SPLITDEC_CASE(4096)
        default: throw std::runtime_error("GPULZStage: chunk_size must be 1024, 2048, or 4096");
    }
#undef FZ_GPULZ_SPLITDEC_CASE
#undef FZ_GPULZ_SPLITDEC_LAUNCH
}

struct GpulzRestripePtrs {
    const uint8_t*  lit; const uint8_t* len; const uint8_t* off; const uint8_t* meta;
    const uint32_t* flag_size; const uint32_t* data_size;
    const uint32_t* lit_off;   const uint32_t* tok_off;
    const uint32_t* meta_off;  const uint32_t* dst_off;
    uint8_t* out;
};

static void launchRestripe(uint8_t word_size, uint32_t chunk_size, int n_chunks,
                           cudaStream_t stream, const GpulzRestripePtrs& p)
{
#define FZ_GPULZ_RESTRIPE_LAUNCH(T_VAL, CS_VAL)                                             \
    gpulzRestripeKernel<T_VAL, CS_VAL><<<n_chunks, GPULZ_DECODE_TPB, 0, stream>>>(          \
        p.lit, p.len, p.off, p.meta, p.flag_size, p.data_size,                              \
        p.lit_off, p.tok_off, p.meta_off, p.dst_off, p.out, (uint32_t)n_chunks)
#define FZ_GPULZ_RESTRIPE_CASE(CS_VAL)                                                      \
    case CS_VAL:                                                                            \
        switch (word_size) {                                                                \
            case 1: FZ_GPULZ_RESTRIPE_LAUNCH(uint8_t,  CS_VAL); break;                      \
            case 2: FZ_GPULZ_RESTRIPE_LAUNCH(uint16_t, CS_VAL); break;                      \
            case 4: FZ_GPULZ_RESTRIPE_LAUNCH(uint32_t, CS_VAL); break;                      \
            case 8: FZ_GPULZ_RESTRIPE_LAUNCH(uint64_t, CS_VAL); break;                      \
            default: throw std::runtime_error("GPULZStage: word_size must be 1, 2, 4, or 8"); \
        }                                                                                    \
        break;
    switch (chunk_size) {
        FZ_GPULZ_RESTRIPE_CASE(1024)
        FZ_GPULZ_RESTRIPE_CASE(2048)
        FZ_GPULZ_RESTRIPE_CASE(4096)
        default: throw std::runtime_error("GPULZStage: chunk_size must be 1024, 2048, or 4096");
    }
#undef FZ_GPULZ_RESTRIPE_CASE
#undef FZ_GPULZ_RESTRIPE_LAUNCH
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
    fwd_free(d_lit_off_dev_);
    fwd_free(d_tok_off_dev_);
    fwd_free(d_meta_off_dev_);
    fwd_free(d_lit_cnt_dev_);
    fwd_free(d_tok_cnt_dev_);
    fwd_free(d_totals_dev_);
}

// Completes the deferred 4-entry totals readback for split mode.
void GPULZStage::finishSplitReadback(cudaStream_t stream) const {
    if (!split_readback_pending_) return;
    uint32_t h[4] = {0, 0, 0, 0};
    FZ_CUDA_CHECK(cudaMemcpyAsync(h, d_totals_dev_, sizeof(h),
                                  cudaMemcpyDeviceToHost, stream));
    FZ_CUDA_CHECK(cudaStreamSynchronize(stream));
    auto* self = const_cast<GPULZStage*>(this);

    // Report each port rounded up to 4 bytes, zero-filling the pad. Downstream
    // consumers and the archive layout rely on stage outputs being 4-byte
    // aligned (same convention as RREStage): the Huffman and ANS decoders read
    // their bitstreams as 32-bit words, so a port ending on an odd byte pushes
    // the *next* archive buffer off alignment and faults their decode kernels.
    for (int i = 0; i < 4; i++) {
        const size_t exact  = h[i];
        const size_t padded = (exact + 3) & ~size_t(3);
        if (split_out_ptr_[i] && padded > exact)
            FZ_CUDA_CHECK(cudaMemsetAsync(split_out_ptr_[i] + exact, 0,
                                          padded - exact, stream));
        self->actual_split_sizes_[i] = padded;
    }
    FZ_CUDA_CHECK(cudaStreamSynchronize(stream));
    self->actual_output_size_ = self->actual_split_sizes_[0];
    for (int i = 0; i < 4; i++) self->split_out_ptr_[i] = nullptr;
    split_readback_pending_ = false;
    tail_readback_stream_   = nullptr;
    FZ_LOG(DEBUG, "GPULZ split encode: literals %u B, lengths %u B, offsets %u B, meta %u B",
           h[0], h[1], h[2], h[3]);
}

void GPULZStage::postStreamSync(cudaStream_t stream) {
    if (split_readback_pending_) { finishSplitReadback(stream); return; }
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
    if (split_mode_ && !is_inverse_) {
        if (split_readback_pending_) finishSplitReadback(tail_readback_stream_);
        return {{"literals", actual_split_sizes_[0]},
                {"lengths",  actual_split_sizes_[1]},
                {"offsets",  actual_split_sizes_[2]},
                {"meta",     actual_split_sizes_[3]}};
    }
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
    if (split_mode_ && !is_inverse_) {
        if (index < 0 || index > 3) return 0;
        if (split_readback_pending_) finishSplitReadback(tail_readback_stream_);
        return actual_split_sizes_[index];
    }
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
        // getRequiredInputAlignment() makes Pipeline pad the *pipeline* input to
        // a chunk multiple, but that guarantee does not survive an upstream
        // stage that changes bytes per element: LorenzoQuant turns float32 into
        // uint16 codes, so a padded 4 KB-aligned input reaches this stage as
        // half as many bytes and need not be chunk-aligned any more. Rather
        // than reject that (a very common wiring), zero-pad the tail chunk
        // here. The padding decodes back as trailing zeros, which the
        // consumer's own element count ignores.
        const uint8_t* d_src = (const uint8_t*)inputs[0];
        size_t         eff_bytes = in_bytes;
        void*          pad_buf = nullptr;
        if (in_bytes % chunk_size_ != 0) {
            eff_bytes = ((in_bytes + chunk_size_ - 1) / chunk_size_) * chunk_size_;
            pad_buf = pool ? pool->allocate(eff_bytes, stream, "gpulz_pad") : nullptr;
            if (!pad_buf) FZ_CUDA_CHECK(cudaMalloc(&pad_buf, eff_bytes));
            FZ_CUDA_CHECK(cudaMemcpyAsync(pad_buf, d_src, in_bytes,
                                          cudaMemcpyDeviceToDevice, stream));
            FZ_CUDA_CHECK(cudaMemsetAsync((uint8_t*)pad_buf + in_bytes, 0,
                                          eff_bytes - in_bytes, stream));
            d_src = (const uint8_t*)pad_buf;
            FZ_LOG(DEBUG, "GPULZ: padded %zu -> %zu bytes for chunk alignment",
                   in_bytes, eff_bytes);
        }
        struct PadGuard {
            MemoryPool* pool; void* p; cudaStream_t s;
            ~PadGuard() { if (!p) return; if (pool) pool->free(p, s); else cudaFree(p); }
        } pad_guard{pool, pad_buf, stream};

        const size_t   n_chunks     = eff_bytes / chunk_size_;
        const uint32_t n_chunks_u   = (uint32_t)n_chunks;
        const uint32_t in_bytes_u   = (uint32_t)eff_bytes;
        const uint32_t grid256      = (n_chunks_u + 255u) / 256u;
        const uint32_t block_elems  = chunk_size_ / word_size_;
        const uint32_t flag_stride  = (block_elems + 7) / 8;

        cached_orig_bytes_   = in_bytes_u;         // padded extent
        orig_unpadded_bytes_ = (uint32_t)in_bytes; // what the inverse reports

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
            fwd_free(d_lit_off_dev_);  d_lit_off_dev_  = nullptr;
            fwd_free(d_tok_off_dev_);  d_tok_off_dev_  = nullptr;
            fwd_free(d_meta_off_dev_); d_meta_off_dev_ = nullptr;
            fwd_free(d_lit_cnt_dev_);  d_lit_cnt_dev_  = nullptr;
            fwd_free(d_tok_cnt_dev_);  d_tok_cnt_dev_  = nullptr;
            fwd_free(d_totals_dev_);   d_totals_dev_   = nullptr;

            if (pool) {
                d_data_scratch_ = (uint8_t*) pool->allocate(n_chunks * (size_t)chunk_size_, stream, "gpulz_data_scratch", true);
                d_flag_scratch_ = (uint8_t*) pool->allocate(n_chunks * (size_t)flag_stride,  stream, "gpulz_flag_scratch", true);
                d_flag_size_    = (uint32_t*)pool->allocate(n_chunks * sizeof(uint32_t),     stream, "gpulz_flag_size",    true);
                d_data_size_    = (uint32_t*)pool->allocate(n_chunks * sizeof(uint32_t),     stream, "gpulz_data_size",    true);
                d_clean_dev_    = (uint32_t*)pool->allocate(n_chunks * sizeof(uint32_t),     stream, "gpulz_clean",        true);
                d_dst_off_dev_  = (uint32_t*)pool->allocate(n_chunks * sizeof(uint32_t),     stream, "gpulz_offsets",      true);
                if (split_mode_) {
                    d_lit_off_dev_  = (uint32_t*)pool->allocate(n_chunks * sizeof(uint32_t), stream, "gpulz_lit_off",  true);
                    d_tok_off_dev_  = (uint32_t*)pool->allocate(n_chunks * sizeof(uint32_t), stream, "gpulz_tok_off",  true);
                    d_meta_off_dev_ = (uint32_t*)pool->allocate(n_chunks * sizeof(uint32_t), stream, "gpulz_meta_off", true);
                    d_lit_cnt_dev_  = (uint32_t*)pool->allocate(n_chunks * sizeof(uint32_t), stream, "gpulz_lit_cnt",  true);
                    d_tok_cnt_dev_  = (uint32_t*)pool->allocate(n_chunks * sizeof(uint32_t), stream, "gpulz_tok_cnt",  true);
                    d_totals_dev_   = (uint32_t*)pool->allocate(4 * sizeof(uint32_t),        stream, "gpulz_totals",   true);
                    if (!d_lit_off_dev_ || !d_tok_off_dev_ || !d_meta_off_dev_
                        || !d_lit_cnt_dev_ || !d_tok_cnt_dev_ || !d_totals_dev_)
                        throw std::runtime_error("GPULZStage: failed to allocate split scratch from MemoryPool");
                }
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
                if (split_mode_) {
                    FZ_CUDA_CHECK(cudaMalloc(&d_lit_off_dev_,  n_chunks * sizeof(uint32_t)));
                    FZ_CUDA_CHECK(cudaMalloc(&d_tok_off_dev_,  n_chunks * sizeof(uint32_t)));
                    FZ_CUDA_CHECK(cudaMalloc(&d_meta_off_dev_, n_chunks * sizeof(uint32_t)));
                    FZ_CUDA_CHECK(cudaMalloc(&d_lit_cnt_dev_,  n_chunks * sizeof(uint32_t)));
                    FZ_CUDA_CHECK(cudaMalloc(&d_tok_cnt_dev_,  n_chunks * sizeof(uint32_t)));
                    FZ_CUDA_CHECK(cudaMalloc(&d_totals_dev_,   4 * sizeof(uint32_t)));
                }
                scratch_pool_owner_ = nullptr;
                scratch_from_pool_  = false;
            }
            scratch_capacity_ = n_chunks;
        }

        FZ_LOG(TRACE, "GPULZ encode: %.1f KB in, %u chunks, word_size %d",
               eff_bytes / 1024.0, n_chunks_u, (int)word_size_);

        const size_t header_size = 4 + 4 + 8 * n_chunks;

        // (1) Encode each chunk into uniform scratch, recording flag/data sizes.
        launchEncode(word_size_, chunk_size_, (int)n_chunks, stream,
                     d_src, d_flag_scratch_, d_data_scratch_,
                     d_flag_size_, d_data_size_, match_level_);
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

        // ── Split mode: scatter into four independently codable streams ───
        if (split_mode_) {
            if (outputs.size() < 4)
                throw std::runtime_error("GPULZStage: split mode needs 4 output buffers");

            uint8_t* d_lit  = (uint8_t*)outputs[0];
            uint8_t* d_len  = (uint8_t*)outputs[1];
            uint8_t* d_offs = (uint8_t*)outputs[2];
            uint8_t* d_meta = (uint8_t*)outputs[3];

            // meta carries the stream header + entry table ahead of the bitmaps.
            FZ_CUDA_CHECK(cudaMemcpyAsync(d_meta, h_hdr, 8, cudaMemcpyHostToDevice, stream));
            gpulzInterleaveHeaderKernel<<<grid256, 256, 0, stream>>>(
                d_flag_size_, d_data_size_, (uint32_t*)(d_meta + 8), n_chunks_u);
            FZ_CUDA_CHECK(cudaGetLastError());

            // meta bitmap offsets: scan of per-chunk flag sizes, then shifted
            // past the header. Must precede the popcount pass, which reads the
            // bitmaps through these offsets on the inverse path.
            gpulzSplitMetaCountKernel<<<grid256, 256, 0, stream>>>(
                d_flag_size_, d_data_size_, d_lit_cnt_dev_, n_chunks_u);
            {
                auto tmp = fz::backend::withTempStorage(pool, stream, "gpulz_split_scan_meta",
                    [&](void* t, size_t& b) {
                        cub::DeviceScan::ExclusiveSum(t, b, d_lit_cnt_dev_,
                                                      d_meta_off_dev_, (int)n_chunks, stream);
                    });
                fz::backend::freeTempStorage(pool, tmp, stream);
            }
            // totals[3] is read from meta_off + meta_cnt, so shift both alike.
            gpulzAddOffsetKernel<<<grid256, 256, 0, stream>>>(
                d_meta_off_dev_, n_chunks_u, (uint32_t)header_size);
            // d_lit_cnt_dev_ still holds meta counts here; stash for the totals
            // kernel before it is overwritten by the literal counts below.
            FZ_CUDA_CHECK(cudaMemcpyAsync(d_tok_cnt_dev_, d_lit_cnt_dev_,
                                          n_chunks * sizeof(uint32_t),
                                          cudaMemcpyDeviceToDevice, stream));

            gpulzSplitCountKernel<<<grid256, 256, 0, stream>>>(
                d_flag_scratch_, /*d_flag_at=*/nullptr, flag_stride,
                d_flag_size_, d_data_size_, d_lit_cnt_dev_, d_clean_dev_,
                n_chunks_u, chunk_size_);
            FZ_CUDA_CHECK(cudaGetLastError());
            {
                auto t1 = fz::backend::withTempStorage(pool, stream, "gpulz_split_scan_lit",
                    [&](void* t, size_t& b) {
                        cub::DeviceScan::ExclusiveSum(t, b, d_lit_cnt_dev_,
                                                      d_lit_off_dev_, (int)n_chunks, stream);
                    });
                fz::backend::freeTempStorage(pool, t1, stream);
                auto t2 = fz::backend::withTempStorage(pool, stream, "gpulz_split_scan_tok",
                    [&](void* t, size_t& b) {
                        cub::DeviceScan::ExclusiveSum(t, b, d_clean_dev_,
                                                      d_tok_off_dev_, (int)n_chunks, stream);
                    });
                fz::backend::freeTempStorage(pool, t2, stream);
            }

            gpulzSplitTotalsKernel<<<1, 1, 0, stream>>>(
                d_lit_off_dev_, d_lit_cnt_dev_, d_tok_off_dev_, d_clean_dev_,
                d_meta_off_dev_, d_tok_cnt_dev_, d_totals_dev_, n_chunks_u - 1);
            FZ_CUDA_CHECK(cudaGetLastError());

            GpulzSplitPtrs sp{d_flag_scratch_, d_data_scratch_, d_src,
                              d_flag_size_, d_data_size_,
                              d_lit_off_dev_, d_tok_off_dev_, d_meta_off_dev_,
                              d_lit, d_len, d_offs, d_meta, flag_stride};
            launchDestripe(word_size_, chunk_size_, (int)n_chunks, stream, sp);
            FZ_CUDA_CHECK(cudaGetLastError());

            split_out_ptr_[0] = d_lit;  split_out_ptr_[1] = d_len;
            split_out_ptr_[2] = d_offs; split_out_ptr_[3] = d_meta;
            split_readback_pending_ = true;
            tail_readback_stream_   = stream;
            return;
        }

        // (6) Defer final output-size readback to postStreamSync.
        tail_last_index_       = n_chunks_u - 1;
        tail_output_ptr_       = (uint8_t*)outputs[0];
        tail_readback_pending_ = true;
        tail_readback_stream_  = stream;

        // (7) Pack compressed chunks from uniform scratch (or raw input, on
        // fallback) into the final packed output.
        gpulzPackKernel<<<(int)n_chunks, 256, 0, stream>>>(
            d_flag_scratch_, d_data_scratch_, d_src, d_out,
            d_dst_off_dev_, d_flag_size_, d_data_size_, flag_stride, chunk_size_);
        FZ_CUDA_CHECK(cudaGetLastError());

    // ── Inverse (decompress) ───────────────────────────────────────────
    } else {
        const uint8_t* d_in  = (const uint8_t*)inputs[0];
        uint8_t*       d_out = (uint8_t*)outputs[0];

        // Split inverse: rebuild the packed single-stream form from the four
        // ports, then fall through to the ordinary decode path unchanged. The
        // restripe is the exact inverse of the destripe, so everything below
        // this block stays oblivious to split mode.
        std::vector<void*> split_tmp;
        auto split_free = [&]() {
            for (void* p : split_tmp) { if (pool) pool->free(p, stream); else cudaFree(p); }
            split_tmp.clear();
        };
        if (split_mode_) {
            if (inputs.size() < 4)
                throw std::runtime_error("GPULZStage: split inverse needs 4 input buffers");
            const uint8_t* d_lit  = (const uint8_t*)inputs[0];
            const uint8_t* d_len  = (const uint8_t*)inputs[1];
            const uint8_t* d_offs = (const uint8_t*)inputs[2];
            const uint8_t* d_meta = (const uint8_t*)inputs[3];

            uint8_t h_meta_hdr[8];
            FZ_CUDA_CHECK(cudaMemcpyAsync(h_meta_hdr, d_meta, 8, cudaMemcpyDeviceToHost, stream));
            FZ_CUDA_CHECK(cudaStreamSynchronize(stream));
            uint32_t s_orig = 0, s_nchunks = 0;
            std::memcpy(&s_orig,    h_meta_hdr + 0, sizeof(uint32_t));
            std::memcpy(&s_nchunks, h_meta_hdr + 4, sizeof(uint32_t));
            cached_orig_bytes_ = s_orig;
            if (s_nchunks == 0 || s_orig == 0) { actual_output_size_ = 0; return; }

            const size_t s_hdr_bytes  = 8 + 8ull * s_nchunks;
            const size_t s_blk_elems  = chunk_size_ / word_size_;
            const size_t s_flag_strd  = (s_blk_elems + 7) / 8;
            const uint32_t s_grid256  = ((uint32_t)s_nchunks + 255u) / 256u;

            auto alloc = [&](size_t bytes, const char* tag) -> void* {
                void* p = pool ? pool->allocate(bytes, stream, tag) : nullptr;
                if (!p) FZ_CUDA_CHECK(cudaMalloc(&p, bytes));
                split_tmp.push_back(p);
                return p;
            };
            uint32_t* s_fsz    = (uint32_t*)alloc(s_nchunks * 4, "gpulz_s_fsz");
            uint32_t* s_dsz    = (uint32_t*)alloc(s_nchunks * 4, "gpulz_s_dsz");
            uint32_t* s_mcnt   = (uint32_t*)alloc(s_nchunks * 4, "gpulz_s_mcnt");
            uint32_t* s_moff   = (uint32_t*)alloc(s_nchunks * 4, "gpulz_s_moff");
            uint32_t* s_lcnt   = (uint32_t*)alloc(s_nchunks * 4, "gpulz_s_lcnt");
            uint32_t* s_loff   = (uint32_t*)alloc(s_nchunks * 4, "gpulz_s_loff");
            uint32_t* s_tcnt   = (uint32_t*)alloc(s_nchunks * 4, "gpulz_s_tcnt");
            uint32_t* s_toff   = (uint32_t*)alloc(s_nchunks * 4, "gpulz_s_toff");

            gpulzDeinterleaveHeaderKernel<<<s_grid256, 256, 0, stream>>>(
                (const uint32_t*)(d_meta + 8), s_fsz, s_dsz, (uint32_t)s_nchunks);

            auto scan = [&](const uint32_t* in, uint32_t* out, const char* tag) {
                auto t = fz::backend::withTempStorage(pool, stream, tag,
                    [&](void* p, size_t& b) {
                        cub::DeviceScan::ExclusiveSum(p, b, in, out, (int)s_nchunks, stream);
                    });
                fz::backend::freeTempStorage(pool, t, stream);
            };
            gpulzSplitMetaCountKernel<<<s_grid256, 256, 0, stream>>>(
                s_fsz, s_dsz, s_mcnt, (uint32_t)s_nchunks);
            scan(s_mcnt, s_moff, "gpulz_inv_scan_meta");
            gpulzAddOffsetKernel<<<s_grid256, 256, 0, stream>>>(
                s_moff, (uint32_t)s_nchunks, (uint32_t)s_hdr_bytes);

            gpulzSplitCountKernel<<<s_grid256, 256, 0, stream>>>(
                d_meta, s_moff, (uint32_t)s_flag_strd, s_fsz, s_dsz,
                s_lcnt, s_tcnt, (uint32_t)s_nchunks, chunk_size_);
            scan(s_lcnt, s_loff, "gpulz_inv_scan_lit");
            scan(s_tcnt, s_toff, "gpulz_inv_scan_tok");

            GpulzSplitDecodePtrs dp{d_lit, d_len, d_offs, d_meta, s_fsz, s_dsz,
                                    s_loff, s_toff, s_moff, d_out};
            launchSplitDecode(word_size_, chunk_size_, (int)s_nchunks, stream, dp);
            FZ_CUDA_CHECK(cudaGetLastError());
            FZ_CUDA_CHECK(cudaStreamSynchronize(stream));
            split_free();

            actual_output_size_ = (orig_unpadded_bytes_ > 0 &&
                                   orig_unpadded_bytes_ <= s_orig)
                                ? (size_t)orig_unpadded_bytes_ : (size_t)s_orig;
            FZ_LOG(DEBUG, "GPULZ split decode done -> %.1f KB",
                   actual_output_size_ / 1024.0);
            return;
        }

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
        uint8_t*  d_mode_    = pool ? (uint8_t*)pool->allocate(num_chunks, stream, "gpulz_inv_mode")
                                     : [&]{ uint8_t* p = nullptr; FZ_CUDA_CHECK(cudaMalloc(&p, num_chunks)); return p; }();

        FZ_CUDA_CHECK(cudaMemcpyAsync(d_in_off_,  h_in_off.data(),  num_chunks * sizeof(uint32_t), cudaMemcpyHostToDevice, stream));

        // Raw-flagged and all-zero (empty) chunks both decode to
        // flag_size=0/data_size=0 (skip the flag-bitmap loop in the decode
        // kernel entirely); a single passthrough kernel below handles the
        // raw copy / zero-fill for every such chunk in one launch (a host
        // loop issuing one async call per chunk is a severe anti-pattern
        // here -- empty chunks are the *common* case for sparse inputs, so a
        // per-chunk host loop turns into thousands of tiny launches that
        // dominate decode time).
        std::vector<uint32_t> h_flag_sz_clean(num_chunks), h_data_sz_clean(num_chunks);
        std::vector<uint8_t>  h_mode(num_chunks);
        for (uint32_t i = 0; i < num_chunks; i++) {
            h_flag_sz_clean[i] = h_is_raw[i] ? 0u : h_flag_sz[i];
            h_data_sz_clean[i] = h_is_raw[i] ? 0u : h_data_sz[i];
            h_mode[i] = h_is_raw[i] ? 1u
                      : (h_flag_sz[i] == 0 && h_data_sz[i] == 0) ? 2u
                      : 0u;
        }
        FZ_CUDA_CHECK(cudaMemcpyAsync(d_flag_sz_, h_flag_sz_clean.data(), num_chunks * sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
        FZ_CUDA_CHECK(cudaMemcpyAsync(d_data_sz_, h_data_sz_clean.data(), num_chunks * sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
        FZ_CUDA_CHECK(cudaMemcpyAsync(d_mode_, h_mode.data(), num_chunks, cudaMemcpyHostToDevice, stream));

        gpulzDecodePassthroughKernel<<<(int)num_chunks, 256, 0, stream>>>(
            d_in, d_out, d_in_off_, d_mode_, chunk_size_);
        FZ_CUDA_CHECK(cudaGetLastError());

        launchDecode(word_size_, chunk_size_, (int)num_chunks, stream,
                     d_out, d_in, d_in_off_, d_flag_sz_, d_data_sz_);
        FZ_CUDA_CHECK(cudaGetLastError());
        FZ_CUDA_CHECK(cudaStreamSynchronize(stream));

        split_free();  // restripe scratch, if split mode allocated any

        if (pool) {
            pool->free(d_in_off_, stream); pool->free(d_flag_sz_, stream);
            pool->free(d_data_sz_, stream); pool->free(d_mode_, stream);
        } else {
            cudaFree(d_in_off_); cudaFree(d_flag_sz_);
            cudaFree(d_data_sz_); cudaFree(d_mode_);
        }

        // Decode wrote `orig_total` bytes (the padded extent), but report the
        // pre-padding size so a downstream stage deriving an element count from
        // it does not overrun its own output buffer.
        actual_output_size_ = (orig_unpadded_bytes_ > 0 &&
                               orig_unpadded_bytes_ <= (uint32_t)orig_total)
                            ? (size_t)orig_unpadded_bytes_ : (size_t)orig_total;
        FZ_LOG(DEBUG, "GPULZ decode done: %.1f KB -> %.1f KB",
               in_bytes / 1024.0, actual_output_size_ / 1024.0);
    }
}

} // namespace fz
