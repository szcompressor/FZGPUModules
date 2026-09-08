/**
 * modules/coders/golomb_rice/golomb_rice_stage.cu
 *
 * GPU implementation of the chunk-local Golomb-Rice coder. One CUDA block per
 * 16 KB chunk, GR_TPB=512 threads (16 warps), chunk held in shared memory.
 * Three phases per chunk, all cooperative:
 *   1. Exact cost search over k in [0, kMaxCandidate()] -> pick the cheapest,
 *      using the WHOLE chunk (one k per chunk, no ratio cost from splitting).
 *   2. Per-element bit length -> WARP-LOCAL exclusive prefix sum (each of the
 *      16 warps IS a restart interval), each warp's bit total independently
 *      rounded up to a byte boundary, then a 16-entry scan over those
 *      byte lengths gives each interval's byte offset in the payload.
 *   3. atomicOr-scatter each element's variable-length codeword into a
 *      zeroed, word-aligned output buffer at its computed bit offset.
 * Falls back to storing the chunk verbatim when packing doesn't shrink it
 * (same convention as RZEStage).
 *
 * Decode is one WARP per chunk, one LANE per restart interval: within a
 * single interval a Rice codeword's length isn't known until it's decoded,
 * so that part is inherently sequential, but the 16 restart intervals per
 * chunk now decode concurrently instead of one thread serially decoding the
 * whole 16 KB chunk -- see golombRiceDecodeKernel.
 */

#include "coders/golomb_rice/golomb_rice_stage.h"
#include "stage/stage_registry.h"
#include "backend/api.h"
#include "backend/warp.h"
#include "backend/algorithms.h"
#include "backend/cub.h"
#include "mem/mempool.h"
#include "cuda_check.h"

#include <thrust/iterator/transform_iterator.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace fz {

// 16 warps -> kIntervalsPerChunk=16 restart intervals/chunk (half the decode
// warp's 32 lanes active). See GolombRiceStage<T>::kIntervalsPerChunk in the
// header for why 32 (GR_TPB=1024) was tried and reverted.
static constexpr int GR_TPB = 512;

namespace {

template<typename T>
__device__ __forceinline__ uint32_t zigzagEncode(T v) {
    using U = typename std::make_unsigned<T>::type;
    const U uv = static_cast<U>(v);
    // Matches the (q<<1)^(q>>31) convention already used in quantizer.cu's
    // TCMS zigzag encode, generalized to T's own width. The extra
    // static_cast<U> around (uv<<1) is NOT redundant: integer promotion widens
    // a sub-int operand (uint16_t for T=int16_t) to `int` before the shift, so
    // without it the result doesn't wrap at 16 bits and XORs against a
    // 32-bit-wide value — invisible for T=int32_t (already >= int's rank) but
    // silently wrong for T=int16_t. Caught by GolombRiceStage's Int16RoundTrip
    // test.
    constexpr int kBits = 8 * sizeof(T);
    return static_cast<uint32_t>(static_cast<U>(uv << 1) ^ static_cast<U>(v >> (kBits - 1)));
}

template<typename T>
__device__ __forceinline__ T zigzagDecode(uint32_t u) {
    using U = typename std::make_unsigned<T>::type;
    const U lo = static_cast<U>(u >> 1);
    const U sign_mask = static_cast<U>(-(static_cast<T>(u & 1u)));
    return static_cast<T>(lo ^ sign_mask);
}

// Scatter the low `nbits` of `value` into the bit stream starting at `bit_off`,
// treating `words` as a flat little-endian bit array (bit i lives in
// words[i/32] at bit position i%32). Every element's bit range is disjoint by
// construction (exclusive prefix sum of lengths), but two elements can still
// share one 32-bit WORD, so the read-modify-write must be atomic.
__device__ __forceinline__ void orBits(uint32_t* words, uint64_t bit_off,
                                       uint64_t value, uint32_t nbits) {
    while (nbits > 0) {
        const uint32_t widx = static_cast<uint32_t>(bit_off >> 5);
        const uint32_t boff = static_cast<uint32_t>(bit_off & 31u);
        const uint32_t take = min(nbits, 32u - boff);
        const uint32_t mask = (take == 32u) ? 0xFFFFFFFFu : ((1u << take) - 1u);
        const uint32_t chunk = static_cast<uint32_t>(value & mask);
        atomicOr(&words[widx], chunk << boff);
        value  >>= take;
        bit_off += take;
        nbits   -= take;
    }
}

// Exact cost (bits) of one value under Rice parameter k, WITH the escape cap:
// q=u>>k ones-then-stop (q+1+k bits) unless q reaches kEscapeQ, in which case
// it's the escape form (kEscapeQ + bitWidth<T> bits, no stop bit needed).
template<uint32_t kEscapeQ, uint32_t kRawBits>
__device__ __forceinline__ uint32_t riceCost(uint32_t u, uint32_t k) {
    const uint32_t q = u >> k;
    return (q < kEscapeQ) ? (q + 1u + k) : (kEscapeQ + kRawBits);
}

template<typename T, int CS>
__global__ void __launch_bounds__(GR_TPB)
golombRiceEncodeKernel(
    const T* __restrict__ in,
    uint8_t* __restrict__ scratch,
    uint32_t* __restrict__ sizes,
    uint32_t total_elems,
    uint32_t scratch_stride)
{
    constexpr int ELEMS = CS / static_cast<int>(sizeof(T));
    constexpr int LOCAL = ELEMS / GR_TPB;
    constexpr uint32_t kEscapeQ = GolombRiceStage<T>::kEscapeQ;
    constexpr uint32_t kRawBits = GolombRiceStage<T>::bitWidth();
    constexpr uint32_t kMaxK    = GolombRiceStage<T>::kMaxCandidate();

    // No separate s_v[] for the original T values: the raw-fallback path
    // reconstructs them from s_u[] via zigzagDecode (zigzag is bijective), so
    // only ONE ELEMS-sized shared array is needed. This matters for T=int16_t
    // (ELEMS=8192): storing both would need 8192*(2+4)=49152B, over the 48KB
    // static shared-memory limit (ptxas: "uses too much shared data").
    constexpr uint32_t kIntervalsPerChunk = GolombRiceStage<T>::kIntervalsPerChunk;
    static_assert(GR_TPB == 32u * kIntervalsPerChunk,
                  "GR_TPB must equal kIntervalsPerChunk*32 -- one encode warp per restart interval");

    __shared__ uint32_t s_u[ELEMS];
    __shared__ uint32_t s_warp_cost[kIntervalsPerChunk][kMaxK + 1];
    __shared__ uint32_t s_total_cost[kMaxK + 1];
    __shared__ uint32_t s_k;
    // Restart-interval (= one per encode warp) byte lengths/bases, computed in
    // Phase 2 below. Kept separate from s_warp_cost (whose meaning is "bit
    // cost per candidate k", unrelated) for clarity.
    __shared__ uint32_t s_warp_bytelen[kIntervalsPerChunk];
    __shared__ uint32_t s_warp_bytebase[kIntervalsPerChunk];
    __shared__ uint32_t s_total_payload_bytes;

    const uint32_t cid      = blockIdx.x;
    const uint32_t base     = cid * static_cast<uint32_t>(ELEMS);
    const int      tid      = threadIdx.x;
    const int      warp     = tid >> 5, lane = tid & 31;
    const int      nwarps   = GR_TPB >> 5;
    const uint32_t live     = min(static_cast<uint32_t>(ELEMS), total_elems - base);

    for (int j = 0; j < LOCAL; ++j) {
        const int i = tid * LOCAL + j;
        const uint32_t g = base + static_cast<uint32_t>(i);
        const T v = (static_cast<uint32_t>(i) < live) ? in[g] : static_cast<T>(0);
        s_u[i] = zigzagEncode<T>(v);
    }
    __syncthreads();

    // ---- Phase 1: exact cost search over k in [0, kMaxK] ----
    uint32_t local_cost[kMaxK + 1];
    #pragma unroll
    for (uint32_t k = 0; k <= kMaxK; ++k) local_cost[k] = 0u;
    for (int j = 0; j < LOCAL; ++j) {
        const int i = tid * LOCAL + j;
        if (static_cast<uint32_t>(i) >= live) continue;
        const uint32_t u = s_u[i];
        #pragma unroll
        for (uint32_t k = 0; k <= kMaxK; ++k)
            local_cost[k] += riceCost<kEscapeQ, kRawBits>(u, k);
    }
    #pragma unroll
    for (uint32_t k = 0; k <= kMaxK; ++k) {
        uint32_t c = local_cost[k];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            c += fz::backend::shflDown(c, off, 32);
        if (lane == 0) s_warp_cost[warp][k] = c;
    }
    __syncthreads();
    if (warp == 0) {
        #pragma unroll
        for (uint32_t k = 0; k <= kMaxK; ++k) {
            uint32_t c = (lane < nwarps) ? s_warp_cost[lane][k] : 0u;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                c += fz::backend::shflDown(c, off, 32);
            if (lane == 0) s_total_cost[k] = c;
        }
        if (lane == 0) {
            uint32_t best = 0;
            for (uint32_t k = 1; k <= kMaxK; ++k)
                if (s_total_cost[k] < s_total_cost[best]) best = k;
            s_k = best;
        }
    }
    __syncthreads();
    const uint32_t k_opt = s_k;

    // ---- Phase 2: per-element bit length -> WARP-LOCAL exclusive scan ----
    // Each warp's scan stays warp-local (not chained into a single flat block
    // scan) because each warp IS a restart interval: its bitstream will be
    // independently byte-aligned so a decode-side warp can start any lane's
    // interval without decoding everything before it (see file header).
    uint32_t local_len[LOCAL];
    uint32_t thread_total = 0;
    for (int j = 0; j < LOCAL; ++j) {
        const int i = tid * LOCAL + j;
        const uint32_t len = (static_cast<uint32_t>(i) < live)
            ? riceCost<kEscapeQ, kRawBits>(s_u[i], k_opt) : 0u;
        local_len[j] = thread_total;   // exclusive within this thread, pre-update
        thread_total += len;
    }
    uint32_t warp_excl = thread_total;
    #pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
        const uint32_t up = fz::backend::shflUp(warp_excl, d, 32);
        if (lane >= d) warp_excl += up;
    }
    const uint32_t warp_bit_total = fz::backend::shfl(warp_excl, 31, 32);
    warp_excl -= thread_total;   // convert inclusive scan -> exclusive (bit offset WITHIN this warp's interval)

    // Round this interval's bit total up to a byte boundary -- that rounded
    // byte length IS this restart interval's payload length.
    const uint32_t warp_byte_len = (warp_bit_total + 7u) >> 3;
    if (lane == 0) s_warp_bytelen[warp] = warp_byte_len;
    __syncthreads();

    // Small (<=16-entry) exclusive scan over per-interval byte lengths, done
    // entirely by warp 0, to get each interval's byte offset within the
    // packed payload.
    if (warp == 0) {
        uint32_t v = (lane < nwarps) ? s_warp_bytelen[lane] : 0u;
        uint32_t excl = v;
        #pragma unroll
        for (int d = 1; d < 32; d <<= 1) {
            const uint32_t up = fz::backend::shflUp(excl, d, 32);
            if (lane >= d) excl += up;
        }
        const uint32_t grand_total = fz::backend::shfl(excl, nwarps - 1, 32);
        excl -= v;
        if (lane < nwarps) s_warp_bytebase[lane] = excl;
        if (lane == 0) s_total_payload_bytes = grand_total;
    }
    __syncthreads();
    const uint64_t thread_base =
        static_cast<uint64_t>(s_warp_bytebase[warp]) * 8ull + warp_excl;
    const uint32_t total_payload_bytes = s_total_payload_bytes;

    // ---- Fallback check: does packing actually shrink this chunk? ----
    // Header is now a 4-byte k word plus a kIntervalsPerChunk-entry byte
    // offset table (also 4-byte words) so the payload that follows stays
    // 4-byte aligned in `scratch` — atomicOr on a uint32_t* REQUIRES aligned
    // addresses (unlike a plain load, this faults, not just slows down, on a
    // misaligned address). `out` itself is always 4-aligned since
    // scratch_stride is a multiple of 4 (see chunkScratchStride()).
    constexpr uint32_t kHeaderBytes = 4u + 4u * kIntervalsPerChunk;
    const uint32_t packed_bytes = kHeaderBytes + total_payload_bytes;
    const uint32_t orig_bytes   = live * static_cast<uint32_t>(sizeof(T));
    uint8_t* out = scratch + static_cast<size_t>(cid) * scratch_stride;

    if (packed_bytes < orig_bytes) {
        // Zero the payload words this chunk will atomicOr into (round up to a
        // 4-byte word boundary; the stride already reserves that + tail pad).
        const uint32_t payload_words = (total_payload_bytes + 3) / 4 + 1;  // +1 word tail safety
        uint32_t* words = reinterpret_cast<uint32_t*>(out + kHeaderBytes);
        for (uint32_t w = tid; w < payload_words; w += GR_TPB) words[w] = 0u;
        if (tid == 0) *reinterpret_cast<uint32_t*>(out) = k_opt;
        // One write per warp (lane 0): this interval's byte offset.
        if (lane == 0)
            reinterpret_cast<uint32_t*>(out + 4)[warp] = s_warp_bytebase[warp];
        __syncthreads();

        for (int j = 0; j < LOCAL; ++j) {
            const int i = tid * LOCAL + j;
            if (static_cast<uint32_t>(i) >= live) continue;
            const uint32_t u   = s_u[i];
            const uint32_t q   = u >> k_opt;
            const uint64_t off = thread_base + local_len[j];
            if (q < kEscapeQ) {
                const uint32_t r = u & ((k_opt == 32u) ? 0xFFFFFFFFu : ((1u << k_opt) - 1u));
                // Unary run FIRST (bits [0,q)), then the implicit 0 stop bit at
                // bit q, then r starting at bit q+1 — this is the order orBits
                // writes to the stream (its low bits go to the earliest stream
                // position), and the order the decoder reads back (unary run,
                // then k more bits for r). r's shift is per-element (q varies).
                const uint64_t codeword = ((1ull << q) - 1ull)
                    | (static_cast<uint64_t>(r) << (q + 1u));
                orBits(words, off, codeword, q + 1u + k_opt);
            } else {
                const uint64_t codeword = ((1ull << kEscapeQ) - 1ull)
                    | (static_cast<uint64_t>(u) << kEscapeQ);
                orBits(words, off, codeword, kEscapeQ + kRawBits);
            }
        }
        if (tid == 0) sizes[cid] = packed_bytes;
    } else {
        // Store verbatim as raw T bytes, reconstructed from the zigzag values
        // (zigzag is bijective, so this is exact — see the s_u[]-only note above).
        T* raw = reinterpret_cast<T*>(out);
        for (uint32_t i = tid; i < live; i += GR_TPB) raw[i] = zigzagDecode<T>(s_u[i]);
        if (tid == 0) sizes[cid] = (1u << 31) | orig_bytes;
    }
}

// Persistent register-resident bit reader for ONE restart interval's decode
// loop. Replaces the original design's per-element peek32() (5 fresh global
// byte-loads EVERY element, even though consecutive Rice codewords' windows
// overlap heavily -- bitpos advances by only k+1..k+kEscapeQ+kRawBits bits
// per element out of the 32-40 bits reloaded each time). Instead this keeps
// up to 64 bits in a register (`buf`, low `nbits` valid, consumed from the
// LSB end) and only touches global memory when fewer than 32 bits remain --
// comfortably enough headroom for the largest single read this coder ever
// does (kRawBits=32 for an escape payload, or kMaxCandidate<=24 for a
// remainder). Byte-wise loads, not a uint32_t* reinterpret, for the same
// reason as the original design: a restart interval's byte offset (from the
// per-chunk offset table) has no alignment guarantee.
template<typename T>
struct GrBitReader {
    const uint8_t* base;
    uint64_t buf;
    uint32_t nbits;
    uint64_t byte_pos;

    __device__ __forceinline__ void refill() {
        while (nbits <= 32u) {
            const uint32_t w = static_cast<uint32_t>(base[byte_pos])
                | (static_cast<uint32_t>(base[byte_pos + 1]) << 8)
                | (static_cast<uint32_t>(base[byte_pos + 2]) << 16)
                | (static_cast<uint32_t>(base[byte_pos + 3]) << 24);
            buf   |= static_cast<uint64_t>(w) << nbits;
            nbits += 32u;
            byte_pos += 4;
        }
    }
    __device__ __forceinline__ void init(const uint8_t* p, uint64_t start_bit) {
        base = p;
        byte_pos = start_bit >> 3;
        const uint32_t skip = static_cast<uint32_t>(start_bit & 7u);
        buf = 0; nbits = 0;
        refill();          // two words -> nbits=64 from a cold start
        buf   >>= skip;
        nbits  -= skip;    // nbits now in [57,64], still comfortably > 32
    }
    __device__ __forceinline__ uint32_t peek32() const { return static_cast<uint32_t>(buf); }
    __device__ __forceinline__ void consume(uint32_t n) {
        buf   >>= n;
        nbits  -= n;
        refill();          // keeps nbits > 32 for the NEXT peek32()/consume()
    }
};

// One WARP per chunk, one LANE per restart interval (kIntervalsPerChunk=16 of
// the warp's 32 lanes are active). Each lane decodes its own interval's
// elements sequentially — a Rice codeword's length isn't known until it's
// decoded, so that part is unavoidable — but kIntervalsPerChunk intervals now
// run concurrently instead of one thread serially decoding the whole 16 KB
// chunk. blockDim.x is a multiple of 32; multiple warps per block decode
// multiple chunks concurrently (warp_in_block picks the chunk).
template<typename T>
__global__ void golombRiceDecodeKernel(
    const uint8_t* __restrict__ d_in,
    T*             __restrict__ d_out,
    const uint32_t* __restrict__ d_in_offsets,
    const uint32_t* __restrict__ d_comp_sizes,
    const uint32_t* __restrict__ d_out_offsets,
    const uint32_t* __restrict__ d_orig_sizes,
    uint32_t num_chunks,
    uint32_t elems_per_interval)
{
    constexpr uint32_t kEscapeQ = GolombRiceStage<T>::kEscapeQ;
    constexpr uint32_t kRawBits = GolombRiceStage<T>::bitWidth();
    constexpr uint32_t kEscMask = (1u << kEscapeQ) - 1u;
    constexpr uint32_t kIntervalsPerChunk = GolombRiceStage<T>::kIntervalsPerChunk;

    const uint32_t lane            = threadIdx.x & 31u;
    const uint32_t warp_in_block    = threadIdx.x >> 5;
    const uint32_t warps_per_block = blockDim.x >> 5;
    const uint32_t cid = blockIdx.x * warps_per_block + warp_in_block;
    if (cid >= num_chunks) return;
    if (lane >= kIntervalsPerChunk) return;   // only kIntervalsPerChunk of 32 lanes do work

    const uint32_t comp_size = d_comp_sizes[cid];
    if (comp_size == 0) return;   // raw passthrough, copied by host

    const uint8_t* chunk = d_in + d_in_offsets[cid];
    // k is a 4-byte little-endian word; assembled byte-wise since `chunk` has
    // no alignment guarantee in the packed archive.
    const uint32_t k = static_cast<uint32_t>(chunk[0])
        | (static_cast<uint32_t>(chunk[1]) << 8)
        | (static_cast<uint32_t>(chunk[2]) << 16)
        | (static_cast<uint32_t>(chunk[3]) << 24);

    // This lane's restart interval: its byte offset from the per-chunk offset
    // table (kIntervalsPerChunk uint32 words right after k), same byte-wise
    // assembly for the same alignment reason.
    const uint8_t* offtab = chunk + 4;
    const uint8_t* ent = offtab + static_cast<size_t>(lane) * 4;
    const uint32_t iv_byte_off = static_cast<uint32_t>(ent[0])
        | (static_cast<uint32_t>(ent[1]) << 8)
        | (static_cast<uint32_t>(ent[2]) << 16)
        | (static_cast<uint32_t>(ent[3]) << 24);
    const uint8_t* payload = offtab + static_cast<size_t>(kIntervalsPerChunk) * 4;

    // d_out_offsets is in BYTES (matches d_orig_sizes / the host's out_cursor
    // bookkeeping), so advance a byte view of d_out, not a T* (which would
    // overshoot by sizeof(T)x for every chunk after the first).
    const uint32_t n = d_orig_sizes[cid] / static_cast<uint32_t>(sizeof(T));
    const uint32_t iv_start = lane * elems_per_interval;
    if (iv_start >= n) return;   // this interval is entirely past the chunk's live tail
    const uint32_t iv_n = min(elems_per_interval, n - iv_start);
    T* out = reinterpret_cast<T*>(reinterpret_cast<uint8_t*>(d_out) + d_out_offsets[cid]) + iv_start;

    GrBitReader<T> br;
    br.init(payload, static_cast<uint64_t>(iv_byte_off) * 8ull);
    for (uint32_t i = 0; i < iv_n; ++i) {
        const uint32_t window = br.peek32();
        const uint32_t inv24  = (~window) & kEscMask;
        uint32_t u;
        if (inv24 == 0u) {
            // Escape: kEscapeQ ones consumed, next kRawBits bits are raw u.
            br.consume(kEscapeQ);
            u = br.peek32() & ((kRawBits == 32u) ? 0xFFFFFFFFu : ((1u << kRawBits) - 1u));
            br.consume(kRawBits);
        } else {
            const uint32_t q = static_cast<uint32_t>(__ffs(static_cast<int>(inv24)) - 1);
            br.consume(q + 1u);   // q ones + the stop bit
            const uint32_t r = (k == 0u) ? 0u
                : (br.peek32() & ((k == 32u) ? 0xFFFFFFFFu : ((1u << k) - 1u)));
            br.consume(k);
            u = (q << k) | r;
        }
        out[i] = zigzagDecode<T>(u);
    }
}

// Packs each chunk's compressed bytes from uniform scratch to the packed
// output (same shape as RZE's rzePackKernel), stripping the raw-flag bit and
// folding in the header offset.
static __global__ void golombRicePackKernel(
    const uint8_t* __restrict__ d_scratch,
    uint8_t*       __restrict__ d_out,
    const uint32_t* __restrict__ d_dst_offsets,
    const uint32_t* __restrict__ d_sizes,
    uint32_t scratch_stride, uint32_t header_off)
{
    const uint32_t cid     = blockIdx.x;
    const uint32_t src_off = cid * scratch_stride;
    const uint32_t dst_off = header_off + d_dst_offsets[cid];
    const uint32_t sz      = d_sizes[cid] & 0x7FFFFFFFu;
    const uint8_t* src = d_scratch + src_off;
    uint8_t*       dst = d_out     + dst_off;
    for (uint32_t i = threadIdx.x; i < sz; i += blockDim.x) dst[i] = src[i];
}

struct GrStripFlagOp {
    __host__ __device__ __forceinline__ uint32_t operator()(uint32_t x) const {
        return x & 0x7FFFFFFFu;
    }
};

} // namespace

template<typename T>
GolombRiceStage<T>::~GolombRiceStage() {
    auto fwd_free = [&](void* p) {
        if (!p) return;
        if (scratch_from_pool_ && scratch_pool_owner_ && !scratch_alive_.expired())
            scratch_pool_owner_->free(p, 0);
        else if (!scratch_from_pool_) cudaFree(p);
    };
    fwd_free(d_scratch_);
    fwd_free(d_sizes_dev_);
    fwd_free(d_dst_off_dev_);
}

template<typename T>
void GolombRiceStage<T>::execute(
    fz::stream_t stream,
    MemoryPool* pool,
    const std::vector<void*>& inputs,
    const std::vector<void*>& outputs,
    const std::vector<size_t>& sizes)
{
    if (inputs.empty() || outputs.empty() || sizes.empty())
        throw std::runtime_error("GolombRiceStage: invalid inputs/outputs");

    tail_readback_pending_ = false;
    const size_t in_bytes = sizes[0];
    if (in_bytes == 0) { actual_output_size_ = 0; return; }

    if (chunk_size_ != 16384u)
        throw std::runtime_error(
            "GolombRiceStage: chunk_size must be 16384 (V1), got "
            + std::to_string(chunk_size_));

    const size_t   n_chunks   = (in_bytes + chunk_size_ - 1) / chunk_size_;
    const uint32_t n_chunks_u = static_cast<uint32_t>(n_chunks);
    const uint32_t in_bytes_u = static_cast<uint32_t>(in_bytes);
    const size_t   stride     = chunkScratchStride();

    if (!is_inverse_) {
        cached_orig_bytes_ = in_bytes_u;

        if (n_chunks > scratch_capacity_) {
            auto fwd_free = [&](void* p) {
                if (!p) return;
                if (scratch_from_pool_ && scratch_pool_owner_ && !scratch_alive_.expired())
                    scratch_pool_owner_->free(p, stream);
                else if (!scratch_from_pool_) { FZ_CUDA_CHECK_WARN(cudaStreamSynchronize(stream)); cudaFree(p); }
            };
            fwd_free(d_scratch_); d_scratch_ = nullptr;
            fwd_free(d_sizes_dev_); d_sizes_dev_ = nullptr;
            fwd_free(d_dst_off_dev_); d_dst_off_dev_ = nullptr;

            if (pool) {
                d_scratch_     = static_cast<uint8_t*>(pool->allocate(n_chunks * stride, stream, "gr_scratch", true));
                d_sizes_dev_   = static_cast<uint32_t*>(pool->allocate(n_chunks * sizeof(uint32_t), stream, "gr_sizes", true));
                d_dst_off_dev_ = static_cast<uint32_t*>(pool->allocate(n_chunks * sizeof(uint32_t), stream, "gr_offsets", true));
                if (!d_scratch_ || !d_sizes_dev_ || !d_dst_off_dev_)
                    throw std::runtime_error("GolombRiceStage: failed to allocate persistent forward scratch");
                scratch_pool_owner_ = pool;
                scratch_from_pool_  = true;
                scratch_alive_      = pool->lifetimeToken();
            } else {
                FZ_CUDA_CHECK(cudaMalloc(&d_scratch_, n_chunks * stride));
                FZ_CUDA_CHECK(cudaMalloc(&d_sizes_dev_, n_chunks * sizeof(uint32_t)));
                FZ_CUDA_CHECK(cudaMalloc(&d_dst_off_dev_, n_chunks * sizeof(uint32_t)));
                scratch_pool_owner_ = nullptr;
                scratch_from_pool_  = false;
            }
            scratch_capacity_ = n_chunks;
        }

        const size_t header_size = 4 + 4 + 4 * n_chunks;
        const uint32_t total_elems = static_cast<uint32_t>(in_bytes / sizeof(T));

        golombRiceEncodeKernel<T, 16384><<<n_chunks_u, GR_TPB, 0, (cudaStream_t)stream>>>(
            static_cast<const T*>(inputs[0]), d_scratch_, d_sizes_dev_,
            total_elems, static_cast<uint32_t>(stride));
        FZ_CUDA_CHECK(cudaGetLastError());

        uint8_t* d_out = static_cast<uint8_t*>(outputs[0]);
        const uint32_t h_hdr[2] = {in_bytes_u, n_chunks_u};
        FZ_CUDA_CHECK(cudaMemcpyAsync(d_out, h_hdr, 8, cudaMemcpyHostToDevice, (cudaStream_t)stream));
        FZ_CUDA_CHECK(cudaMemcpyAsync(d_out + 8, d_sizes_dev_, n_chunks * sizeof(uint32_t),
                                      cudaMemcpyDeviceToDevice, (cudaStream_t)stream));

        {
            auto clean_it = thrust::make_transform_iterator(
                static_cast<const uint32_t*>(d_sizes_dev_), GrStripFlagOp{});
            auto scan_tmp = fz::backend::withTempStorage(pool, stream, "gr_cub_scan_tmp",
                [&](void* tmp, size_t& bytes) {
                    cub::DeviceScan::ExclusiveSum(tmp, bytes, clean_it, d_dst_off_dev_,
                                                  static_cast<int>(n_chunks), (cudaStream_t)stream);
                });
            fz::backend::freeTempStorage(pool, scan_tmp, stream);
        }

        tail_last_index_       = n_chunks_u - 1;
        tail_header_size_      = static_cast<uint32_t>(header_size);
        tail_output_ptr_       = d_out;
        tail_readback_pending_ = true;

        golombRicePackKernel<<<n_chunks_u, 512, 0, (cudaStream_t)stream>>>(
            d_scratch_, d_out, d_dst_off_dev_, d_sizes_dev_,
            static_cast<uint32_t>(stride), static_cast<uint32_t>(header_size));
        FZ_CUDA_CHECK(cudaGetLastError());

    } else {
        const uint8_t* d_in  = static_cast<const uint8_t*>(inputs[0]);
        T*             d_out = static_cast<T*>(outputs[0]);

        uint8_t h_hdr_raw[8];
        FZ_CUDA_CHECK(cudaMemcpyAsync(h_hdr_raw, d_in, 8, cudaMemcpyDeviceToHost, (cudaStream_t)stream));
        FZ_CUDA_CHECK(cudaStreamSynchronize((cudaStream_t)stream));
        uint32_t orig_total, num_chunks;
        std::memcpy(&orig_total, h_hdr_raw + 0, sizeof(uint32_t));
        std::memcpy(&num_chunks, h_hdr_raw + 4, sizeof(uint32_t));
        cached_orig_bytes_ = orig_total;
        if (num_chunks == 0 || orig_total == 0) { actual_output_size_ = 0; return; }

        std::vector<uint32_t> h_entries(num_chunks);
        FZ_CUDA_CHECK(cudaMemcpyAsync(h_entries.data(), d_in + 8, num_chunks * sizeof(uint32_t),
                                      cudaMemcpyDeviceToHost, (cudaStream_t)stream));
        FZ_CUDA_CHECK(cudaStreamSynchronize((cudaStream_t)stream));

        const size_t header_bytes = 4 + 4 + 4 * num_chunks;
        std::vector<uint32_t> h_in_off(num_chunks), h_comp_sz(num_chunks),
                              h_out_off(num_chunks), h_orig_sz(num_chunks);
        std::vector<bool> h_is_raw(num_chunks);

        uint32_t in_cursor = static_cast<uint32_t>(header_bytes), out_cursor = 0;
        for (uint32_t i = 0; i < num_chunks; ++i) {
            const uint32_t entry  = h_entries[i];
            const bool     raw    = (entry & 0x80000000u) != 0;
            const uint32_t stored = entry & 0x7FFFFFFFu;
            h_in_off[i]  = in_cursor;
            h_comp_sz[i] = raw ? 0u : stored;
            h_out_off[i] = out_cursor;
            h_is_raw[i]  = raw;
            const uint32_t chunk_orig = (i + 1 < num_chunks)
                ? chunk_size_ : (orig_total - out_cursor);
            h_orig_sz[i] = chunk_orig;
            in_cursor  += stored;
            out_cursor += chunk_orig;
        }

        auto alloc_u32 = [&](const char* tag) -> uint32_t* {
            if (pool) return static_cast<uint32_t*>(pool->allocate(num_chunks * sizeof(uint32_t), stream, tag));
            uint32_t* p = nullptr; FZ_CUDA_CHECK(cudaMalloc(&p, num_chunks * sizeof(uint32_t))); return p;
        };
        uint32_t* d_in_off  = alloc_u32("gr_inv_in_off");
        uint32_t* d_comp_sz = alloc_u32("gr_inv_comp_sz");
        uint32_t* d_out_off = alloc_u32("gr_inv_out_off");
        uint32_t* d_orig_sz = alloc_u32("gr_inv_orig_sz");
        FZ_CUDA_CHECK(cudaMemcpyAsync(d_in_off,  h_in_off.data(),  num_chunks * 4, cudaMemcpyHostToDevice, (cudaStream_t)stream));
        FZ_CUDA_CHECK(cudaMemcpyAsync(d_comp_sz, h_comp_sz.data(), num_chunks * 4, cudaMemcpyHostToDevice, (cudaStream_t)stream));
        FZ_CUDA_CHECK(cudaMemcpyAsync(d_out_off, h_out_off.data(), num_chunks * 4, cudaMemcpyHostToDevice, (cudaStream_t)stream));
        FZ_CUDA_CHECK(cudaMemcpyAsync(d_orig_sz, h_orig_sz.data(), num_chunks * 4, cudaMemcpyHostToDevice, (cudaStream_t)stream));

        // One warp per chunk (kIntervalsPerChunk of its 32 lanes decode one
        // restart interval each); DEC_TPB/32 warps, hence chunks, per block.
        constexpr int DEC_TPB = 128;
        constexpr int kWarpsPerBlock = DEC_TPB / 32;
        const uint32_t grid = (num_chunks + kWarpsPerBlock - 1) / kWarpsPerBlock;
        const uint32_t elems_per_interval =
            chunk_size_ / static_cast<uint32_t>(sizeof(T)) / GolombRiceStage<T>::kIntervalsPerChunk;
        golombRiceDecodeKernel<T><<<grid, DEC_TPB, 0, (cudaStream_t)stream>>>(
            d_in, d_out, d_in_off, d_comp_sz, d_out_off, d_orig_sz, num_chunks,
            elems_per_interval);
        FZ_CUDA_CHECK(cudaGetLastError());

        for (uint32_t i = 0; i < num_chunks; ++i) {
            if (h_is_raw[i])
                FZ_CUDA_CHECK(cudaMemcpyAsync(
                    reinterpret_cast<uint8_t*>(d_out) + h_out_off[i], d_in + h_in_off[i],
                    h_orig_sz[i], cudaMemcpyDeviceToDevice, (cudaStream_t)stream));
        }
        FZ_CUDA_CHECK(cudaStreamSynchronize((cudaStream_t)stream));

        if (pool) {
            pool->free(d_in_off, stream); pool->free(d_comp_sz, stream);
            pool->free(d_out_off, stream); pool->free(d_orig_sz, stream);
        } else {
            cudaFree(d_in_off); cudaFree(d_comp_sz); cudaFree(d_out_off); cudaFree(d_orig_sz);
        }
        actual_output_size_ = static_cast<size_t>(orig_total);
    }
}

template<typename T>
void GolombRiceStage<T>::postStreamSync(fz::stream_t stream) {
    if (!tail_readback_pending_) return;
    uint32_t tail_off = 0, tail_sz = 0;
    FZ_CUDA_CHECK(cudaMemcpyAsync(&tail_off, d_dst_off_dev_ + tail_last_index_, 4,
                                  cudaMemcpyDeviceToHost, (cudaStream_t)stream));
    FZ_CUDA_CHECK(cudaMemcpyAsync(&tail_sz, d_sizes_dev_ + tail_last_index_, 4,
                                  cudaMemcpyDeviceToHost, (cudaStream_t)stream));
    FZ_CUDA_CHECK(cudaStreamSynchronize((cudaStream_t)stream));
    const size_t total_out = static_cast<size_t>(tail_header_size_) + tail_off
                           + (tail_sz & 0x7FFFFFFFu);
    actual_output_size_ = (total_out + 3) & ~size_t(3);
    if (tail_output_ptr_ && actual_output_size_ > total_out) {
        FZ_CUDA_CHECK(cudaMemsetAsync(tail_output_ptr_ + total_out, 0,
                                      actual_output_size_ - total_out, (cudaStream_t)stream));
        FZ_CUDA_CHECK(cudaStreamSynchronize((cudaStream_t)stream));
    }
    tail_output_ptr_ = nullptr;
    tail_readback_pending_ = false;
}

template class GolombRiceStage<int16_t>;
template class GolombRiceStage<int32_t>;

} // namespace fz

// ── FZM-header reconstruction (self-registered; see stage_registry.h) ─────────
namespace {
fz::Stage* GolombRice_fromHeader(const uint8_t* config, size_t config_size) {
    using fz::DataType;
    DataType dt = (config_size > 4) ? static_cast<DataType>(config[4]) : DataType::INT32;
    if (dt == DataType::INT16) {
        auto* s = new fz::GolombRiceStage<int16_t>(); s->deserializeHeader(config, config_size); return s;
    }
    auto* s = new fz::GolombRiceStage<int32_t>(); s->deserializeHeader(config, config_size); return s;
}
}  // namespace
FZ_REGISTER_STAGE_FACTORY(fz::StageType::GOLOMB_RICE, GolombRice_fromHeader);
