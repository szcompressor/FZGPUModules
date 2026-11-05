/**
 * @file modules/fused/chunk_fusion/pfpl_stream.cu
 * @brief Two-array PFPL stream (pfpl_stream.h): the chunk-cooperative harness
 *        (chunk_fusion.cuh) driven by one encode and one decode kernel that each
 *        place their chunk in the archive by decoupled look-back.
 */

#include "fused/chunk_fusion/pfpl_stream.h"
#include "fused/chunk_fusion/chunk_fusion.cuh"
#include "backend/api.h"
#include "backend/warp.h"
#include "cuda_check.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace fz {
namespace pfpl {
namespace {

using fused::chunk::byte;
using fused::chunk::TPB;
using fused::chunk::Geom;
using fused::chunk::ChunkEncoded;
using fused::chunk::ChunkSideCtx;
using fused::chunk::QuantInplaceZigzag;
using fused::chunk::QuantInplaceZigzagParams;
using fused::chunk::DiffNegabinary;
using fused::chunk::Bitshuffle32;
using fused::chunk::RZECoder;

constexpr uint32_t kElems        = Geom<kChunkBytes>::NELEM;   // floats per chunk
constexpr uint32_t kElems64      = kChunkBytes / 8;            // doubles per chunk
constexpr uint32_t kNPP64        = kElems64 / 32;              // 32-bit words per bit-plane of a double chunk
constexpr uint32_t kLcHeader     = 8;                          // LC archive: [u32 coded bytes][u32 chunks]
constexpr uint32_t kStreamHeader = sizeof(StreamHeader);
constexpr int      kWave         = fz::backend::kWaveSize;

// The double quantizer's parameters: the float ones' rule, with no threshold (the radius check also turns away
// non-finite values).
struct QuantParams64 {
    double ebx2_r;
    double ebx2;
    double abs_eb;
};

template <typename Real> struct Elem;
template <> struct Elem<float> {
    using Word   = uint32_t;
    using SWord  = int32_t;
    using Params = QuantInplaceZigzagParams;
    static constexpr uint32_t kMagic = kStreamMagic;
};
template <> struct Elem<double> {
    using Word   = uint64_t;
    using SWord  = int64_t;
    using Params = QuantParams64;
    static constexpr uint32_t kMagic = kStreamMagic64;
};

__device__ __forceinline__ uint64_t doubleBits(double x) {
    union { double d; uint64_t u; } b;
    b.d = x;
    return b.u;
}

__device__ __forceinline__ double bitsDouble(uint64_t u) {
    union { double d; uint64_t u; } b;
    b.u = u;
    return b.d;
}

// x's code: its zigzagged quantization index, or its raw bits when that index would not reconstruct within the bound.
// Raw bits read as raw because a raw value is never subnormal (one that small quantizes to index 0 at any allowed
// bound), so they lie at or above 2 * kQuantRadius64. The range check comes first: converting an index out of
// long long's range, or a non-finite one, is undefined.
__device__ __forceinline__ uint64_t quantCode64(double x, const QuantParams64& p) {
    const double r = rint(x * p.ebx2_r);
    const bool repr = fabs(r) < static_cast<double>(kQuantRadius64) &&
                      fabs(x - fz::backend::fmulRounded(r, p.ebx2)) <= p.abs_eb;
    const long long q = repr ? static_cast<long long>(r) : 0ll;
    return repr ? (static_cast<uint64_t>(q) << 1) ^ static_cast<uint64_t>(q >> 63) : doubleBits(x);
}

__device__ __forceinline__ double valueOfCode64(uint64_t code, double ebx2) {
    if ((code >> 1) >= kQuantRadius64) return bitsDouble(code);
    const long long q = static_cast<long long>((code >> 1) ^ (0ull - (code & 1ull)));
    return static_cast<double>(q) * ebx2;
}

// Look-back word: [63:62] kind (0 empty, 1 the chunk's own size, 2 its inclusive
// prefix) | [61:32] epoch of the call that wrote it | [31:0] bytes. A word left by
// another call reads as empty, so the state is never cleared between calls.
constexpr unsigned long long kKindMask  = 3ull << 62;
constexpr unsigned long long kAggregate = 1ull << 62;
constexpr unsigned long long kInclusive = 2ull << 62;
constexpr uint32_t           kEpochMask = 0x3fffffffu;

__device__ __forceinline__ unsigned long long lbWord(unsigned long long kind, uint32_t epoch, uint32_t bytes) {
    return kind | (static_cast<unsigned long long>(epoch) << 32) | bytes;
}

__device__ __forceinline__ bool lbCurrent(unsigned long long w, uint32_t epoch) {
    return (w & kKindMask) != 0ull && static_cast<uint32_t>((w >> 32) & kEpochMask) == epoch;
}

__device__ __forceinline__ void lbPause() {
#if defined(FZGMOD_BACKEND_HIP)
    __builtin_amdgcn_s_sleep(1);
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    __nanosleep(32);
#endif
}

__device__ __forceinline__ void streamTrap() {
#if defined(FZGMOD_BACKEND_HIP)
    __builtin_trap();
#else
    __trap();
#endif
}

// Exclusive payload offset of chunk `cid`, found by one whole wavefront (every
// lane gets it): kWave predecessors at a time (lane k looks at chunk cid-1-k),
// stopping at the nearest inclusive prefix. Encode (`table` null) waits for a
// predecessor that has not published -- chunks are ticketed in start order, so
// it is running. Decode reads an unpublished predecessor's size from the
// archive's size table instead, so it never waits and needs no ticket.
__device__ uint32_t lookbackWave(uint32_t cid, uint32_t own, unsigned long long* __restrict__ state,
                                 uint32_t epoch, const uint32_t* __restrict__ table) {
    const int lane = static_cast<int>(threadIdx.x) % kWave;
    if (lane == 0 && table == nullptr)
        atomicExch(&state[cid], lbWord(cid == 0 ? kInclusive : kAggregate, epoch, own));
    unsigned long long prefix = 0;
    for (long long window = static_cast<long long>(cid) - 1; window >= 0; window -= kWave) {
        const long long j = window - lane;
        bool incl = true;                 // before chunk 0: an inclusive prefix of 0
        unsigned long long bytes = 0;
        if (j >= 0) {
            unsigned long long w = fused::chunk::lookbackLoad(&state[j]);
            if (table == nullptr) {
                while (!lbCurrent(w, epoch)) {
                    lbPause();
                    w = fused::chunk::lookbackLoad(&state[j]);
                }
            } else if (!lbCurrent(w, epoch) || (w & kKindMask) != kInclusive) {
                w = lbWord(kAggregate, epoch, table[j] & 0x7fffffffu);
            }
            incl  = (w & kKindMask) == kInclusive;
            bytes = w & 0xffffffffull;
        }
        const fz::backend::wave_mask_t inclMask = fz::backend::ballotWave(incl);
        const int upto = inclMask ? fz::backend::ffsWave(inclMask) - 1 : kWave - 1;
        prefix += fz::backend::waveSum(lane <= upto ? bytes : 0ull);
        if (inclMask) break;
    }
    if (lane == 0 && (cid != 0 || table != nullptr))
        atomicExch(&state[cid], lbWord(kInclusive, epoch, static_cast<uint32_t>(prefix) + own));
    return static_cast<uint32_t>(prefix);
}

// lookbackWave() run by the block's first wavefront, its result shared with the block.
__device__ uint32_t lookbackOffset(uint32_t cid, uint32_t own, unsigned long long* __restrict__ state,
                                   uint32_t epoch, const uint32_t* __restrict__ table) {
    __shared__ uint32_t s_offset;
    if (threadIdx.x < kWave) {
        const uint32_t offset = lookbackWave(cid, own, state, epoch, table);
        if (threadIdx.x == 0) s_offset = offset;
    }
    __syncthreads();
    return s_offset;
}

// A coded chunk's size-table entry and payload go into the stream; the last chunk adds the headers and the length.
// Layout: [StreamHeader][LC header][u32 sizes x chunks][payloads].
__device__ __forceinline__ void storeEncoded(const ChunkEncoded& e, uint32_t cid, uint32_t offset, uint32_t magic,
                                             float abs_eb, uint32_t n, byte* __restrict__ out,
                                             unsigned long long capacity, uint32_t chunks,
                                             size_t* __restrict__ d_stream_bytes) {
    const unsigned long long table = kStreamHeader + kLcHeader;
    const unsigned long long data  = table + 4ull * chunks;
    if (threadIdx.x == 0 && table + 4ull * (cid + 1u) <= capacity)
        reinterpret_cast<uint32_t*>(out + table)[cid] = e.size_word;
    if (data + offset + e.nbytes <= capacity)
        fused::chunk::storeChunkBytes(out + data + offset, e.src, e.nbytes);

    if (cid == chunks - 1u && threadIdx.x == 0) {
        const unsigned long long archive = kLcHeader + 4ull * chunks + offset + e.nbytes;
        const unsigned long long padded  = (archive + 3ull) & ~3ull;
        const unsigned long long total   = kStreamHeader + padded;
        uint32_t* w = reinterpret_cast<uint32_t*>(out);
        if (total <= capacity) {
            for (unsigned long long b = kStreamHeader + archive; b < total; ++b) out[b] = 0;
            w[4] = chunks * static_cast<uint32_t>(kChunkBytes);
            w[5] = chunks;
        }
        // The stream header goes in whenever it fits, so a stream too long for the
        // buffer still records the length it needed.
        if (kStreamHeader <= capacity) {
            w[0] = magic;
            w[1] = static_cast<uint32_t>(padded);
            w[2] = __float_as_uint(abs_eb);
            w[3] = n;
        }
        if (d_stream_bytes != nullptr) *d_stream_bytes = static_cast<size_t>(total);
    }
}

// One block per chunk: code it in shared memory, find its offset, store it.
__global__ void __launch_bounds__(TPB)
encodeKernel(const float* __restrict__ in, uint32_t n, uint32_t coded_n, QuantInplaceZigzagParams qp,
             byte* __restrict__ out, unsigned long long capacity, uint32_t chunks,
             unsigned long long* __restrict__ state, unsigned long long* __restrict__ ticket,
             unsigned long long ticket_base, uint32_t epoch, size_t* __restrict__ d_stream_bytes) {
    __shared__ uint32_t s_cid;
    __shared__ QuantInplaceZigzagParams s_qp;
    if (threadIdx.x == 0) {
        s_cid = static_cast<uint32_t>(atomicAdd(ticket, 1ull) - ticket_base);
        s_qp  = qp;
    }
    __syncthreads();
    const uint32_t cid = s_cid;

    const ChunkEncoded e =
        fused::chunk::chunk_encode_at<kChunkBytes, QuantInplaceZigzag, RZECoder<kChunkBytes>,
                                      DiffNegabinary, Bitshuffle32<kChunkBytes>>(
            cid, in, coded_n, reinterpret_cast<const byte*>(&s_qp), ChunkSideCtx{}, n);
    const uint32_t offset = lookbackOffset(cid, e.nbytes, state, epoch, nullptr);
    storeEncoded(e, cid, offset, kStreamMagic, qp.abs_eb, n, out, capacity, chunks, d_stream_bytes);
}

// One double chunk in the block's shared memory: codes, difference and negabinary in 64 bits, then the bitshuffle and
// RZE. A 64-plane bitshuffle is two 32-plane ones: each 32-element group's high words give planes 0..31 (MSB first, as
// the float planes) and its low words planes 32..63. Each plane's 64-bit words hold an even group's bits above the next
// odd group's, so in memory the odd group's word comes first (group ^ 1), as in the staged 8-byte bitshuffle.
__device__ __forceinline__ ChunkEncoded encodeChunk64(uint32_t cid, const double* __restrict__ in, uint32_t n,
                                                      const QuantParams64& qp) {
    __shared__ __align__(16) uint64_t sA[kElems64];
    __shared__ __align__(16) uint64_t sB[kElems64];
    __shared__ __align__(16) byte     sTemp[fused::chunk::TEMP_BYTES];

    const size_t base = static_cast<size_t>(cid) * kElems64;
    for (uint32_t i = threadIdx.x; i < kElems64; i += TPB) {
        const size_t e = base + i;
        sA[i] = quantCode64(e < n ? in[e] : 0.0, qp);
    }
    __syncthreads();

    for (uint32_t i = threadIdx.x; i < kElems64; i += TPB) {
        const uint64_t d = i == 0 ? sA[i] : sA[i] - sA[i - 1];
        sB[i] = Negabinary<int64_t>::encode(static_cast<int64_t>(d));
    }
    __syncthreads();

    uint32_t* planes = reinterpret_cast<uint32_t*>(sA);
    const int lane = threadIdx.x & 31;
    for (uint32_t i = threadIdx.x; i < kElems64; i += TPB) {
        const uint32_t g = (i / 32) ^ 1u;
        planes[g + lane * kNPP64] = fused::chunk::butterfly32(static_cast<uint32_t>(sB[i] >> 32), lane);
        planes[g + (32 + lane) * kNPP64] = fused::chunk::butterfly32(static_cast<uint32_t>(sB[i]), lane);
    }
    __syncthreads();

    int csize = kChunkBytes;
    const bool good = RZECoder<kChunkBytes>::encode(csize, reinterpret_cast<byte*>(sA), reinterpret_cast<byte*>(sB),
                                                    sTemp, nullptr);
    __syncthreads();
    if (good && csize < kChunkBytes)
        return ChunkEncoded{reinterpret_cast<const byte*>(sB), static_cast<uint32_t>(csize), static_cast<uint32_t>(csize)};
    return ChunkEncoded{reinterpret_cast<const byte*>(sA), static_cast<uint32_t>(kChunkBytes),
                        (1u << 31) | static_cast<uint32_t>(kChunkBytes)};
}

__global__ void __launch_bounds__(TPB)
encodeKernel64(const double* __restrict__ in, uint32_t n, QuantParams64 qp, float abs_eb,
               byte* __restrict__ out, unsigned long long capacity, uint32_t chunks,
               unsigned long long* __restrict__ state, unsigned long long* __restrict__ ticket,
               unsigned long long ticket_base, uint32_t epoch, size_t* __restrict__ d_stream_bytes) {
    __shared__ uint32_t s_cid;
    if (threadIdx.x == 0) s_cid = static_cast<uint32_t>(atomicAdd(ticket, 1ull) - ticket_base);
    __syncthreads();
    const uint32_t cid = s_cid;
    const ChunkEncoded e = encodeChunk64(cid, in, n, qp);
    const uint32_t offset = lookbackOffset(cid, e.nbytes, state, epoch, nullptr);
    storeEncoded(e, cid, offset, kStreamMagic64, abs_eb, n, out, capacity, chunks, d_stream_bytes);
}

// A field of no elements: the headers alone.
__global__ void emptyKernel(byte* __restrict__ out, unsigned long long capacity, uint32_t magic, float abs_eb,
                            size_t* __restrict__ d_stream_bytes) {
    const unsigned long long total = kStreamHeader + kLcHeader;
    uint32_t* w = reinterpret_cast<uint32_t*>(out);
    if (kStreamHeader <= capacity) {
        w[0] = magic;
        w[1] = kLcHeader;
        w[2] = __float_as_uint(abs_eb);
        w[3] = 0u;
    }
    if (total <= capacity) {
        w[4] = 0u;
        w[5] = 0u;
    }
    if (d_stream_bytes != nullptr) *d_stream_bytes = static_cast<size_t>(total);
}

// The headers must describe this field. A stream that does not is a caller's error: stop rather than decode garbage.
__device__ __forceinline__ void checkStream(const uint32_t* w, uint32_t magic, uint32_t n, uint32_t chunks) {
    if (w[0] != magic || w[3] != n || w[4] != chunks * static_cast<uint32_t>(kChunkBytes) || w[5] != chunks ||
        w[1] < kLcHeader + 4u * chunks || !(__uint_as_float(w[2]) > 0.0f)) {
        streamTrap();
    }
}

// One block per chunk (gridDim.x == chunks): find the chunk's payload, decode it.
__global__ void __launch_bounds__(TPB)
decodeKernel(const byte* __restrict__ stream, float* __restrict__ out, uint32_t n,
             unsigned long long* __restrict__ state, uint32_t epoch) {
    const uint32_t* w = reinterpret_cast<const uint32_t*>(stream);
    const uint32_t chunks = gridDim.x;
    checkStream(w, kStreamMagic, n, chunks);
    const uint32_t cid   = blockIdx.x;
    const uint32_t* table = w + (kStreamHeader + kLcHeader) / 4u;
    const uint32_t entry  = table[cid];
    const uint32_t offset = lookbackOffset(cid, entry & 0x7fffffffu, state, epoch, table);
    const byte* payload = stream + kStreamHeader + kLcHeader + 4ull * chunks + offset;
    const uint32_t base = cid * kElems;
    fused::chunk::chunk_inverse_pfpl_at<kChunkBytes, RZECoder<kChunkBytes>>(
        payload, entry, /*coded_full=*/true, static_cast<int>(min(kElems, n - base)),
        2.0f * __uint_as_float(w[2]), /*inplace_outliers=*/true, kQuantRadius, out + base);
}

// One double chunk from the archive into out[0, cnt): RZE, the inverse bitshuffle, then the inverse negabinary and a
// chunk-local inclusive scan, as chunk_inverse_pfpl_at does for floats: each 512-value slab is scanned warp-wise, with
// 16 warp totals and one carry in the coder scratch.
__device__ __forceinline__ void decodeChunk64(const byte* __restrict__ payload, uint32_t entry, int cnt, double ebx2,
                                              double* __restrict__ out) {
    __shared__ __align__(16) uint64_t sA[kElems64];
    __shared__ __align__(16) uint64_t sB[kElems64];
    __shared__ __align__(16) byte     sTemp[fused::chunk::TEMP_BYTES];

    const uint32_t stored = entry & 0x7fffffffu;
    const bool raw = (entry & 0x80000000u) != 0;
    fused::chunk::loadChunkBytes(reinterpret_cast<byte*>(sA), payload, stored);
    __syncthreads();

    uint64_t* cur = sA;
    uint64_t* alt = sB;
    if (!raw) {
        int csize = static_cast<int>(stored);
        RZECoder<kChunkBytes>::decode(csize, reinterpret_cast<byte*>(sA), reinterpret_cast<byte*>(sB), sTemp);
        __syncthreads();
        cur = sB;
        alt = sA;
    }

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const uint32_t* planes = reinterpret_cast<const uint32_t*>(cur);
    for (uint32_t i = threadIdx.x; i < kElems64; i += TPB) {
        const uint32_t g  = (i / 32) ^ 1u;
        const uint32_t hi = fused::chunk::butterfly32(planes[g + lane * kNPP64], lane);
        const uint32_t lo = fused::chunk::butterfly32(planes[g + (32 + lane) * kNPP64], lane);
        alt[i] = (static_cast<uint64_t>(hi) << 32) | lo;
    }
    __syncthreads();

    const int64_t* words = reinterpret_cast<const int64_t*>(alt);
    int64_t* qout = reinterpret_cast<int64_t*>(cur);   // the bitshuffled bytes are done with
    int64_t* scan  = reinterpret_cast<int64_t*>(sTemp);
    if (threadIdx.x == 0) scan[16] = 0;
    __syncthreads();
    for (int slab = 0; slab < cnt; slab += TPB) {
        const int idx = slab + static_cast<int>(threadIdx.x);
        const bool valid = idx < cnt;
        int64_t v = valid ? Negabinary<int64_t>::decode(static_cast<uint64_t>(words[idx])) : 0;
        for (int delta = 1; delta < 32; delta <<= 1) {
            const int64_t up = fz::backend::shflUp(v, delta, 32);
            if (lane >= delta) v += up;
        }
        if (lane == 31) scan[warp] = v;
        __syncthreads();
        if (warp == 0) {
            int64_t t = lane < 16 ? scan[lane] : 0;
            for (int delta = 1; delta < 32; delta <<= 1) {
                const int64_t up = fz::backend::shflUp(t, delta, 32);
                if (lane >= delta) t += up;
            }
            if (lane < 16) scan[lane] = t;
        }
        __syncthreads();
        const int64_t q = v + (warp ? scan[warp - 1] : 0) + scan[16];
        if (valid) qout[idx] = q;
        __syncthreads();
        if (threadIdx.x == 0) scan[16] += scan[15];
        __syncthreads();
    }
    for (int i = threadIdx.x; i < cnt; i += TPB)
        out[i] = valueOfCode64(static_cast<uint64_t>(qout[i]), ebx2);
}

__global__ void __launch_bounds__(TPB)
decodeKernel64(const byte* __restrict__ stream, double* __restrict__ out, uint32_t n,
               unsigned long long* __restrict__ state, uint32_t epoch) {
    const uint32_t* w = reinterpret_cast<const uint32_t*>(stream);
    const uint32_t chunks = gridDim.x;
    checkStream(w, kStreamMagic64, n, chunks);
    const uint32_t cid   = blockIdx.x;
    const uint32_t* table = w + (kStreamHeader + kLcHeader) / 4u;
    const uint32_t entry  = table[cid];
    const uint32_t offset = lookbackOffset(cid, entry & 0x7fffffffu, state, epoch, table);
    const byte* payload = stream + kStreamHeader + kLcHeader + 4ull * chunks + offset;
    const uint32_t base = cid * kElems64;
    decodeChunk64(payload, entry, static_cast<int>(min(kElems64, n - base)),
                  2.0 * static_cast<double>(__uint_as_float(w[2])), out + base);
}

#if defined(FZGMOD_BACKEND_HIP)
#include "fused/chunk_fusion/pfpl_stream_amd.inl"

// FZ_PFPL_HARNESS=1 runs the generic harness kernels instead of the AMD ones (same stream; for comparison).
bool useHarness() {
    static const bool harness = [] {
        const char* e = std::getenv("FZ_PFPL_HARNESS");
        return e != nullptr && e[0] == '1';
    }();
    return harness;
}
#endif

uint32_t chunksOfBytes(size_t bytes) { return static_cast<uint32_t>((bytes + kChunkBytes - 1) / kChunkBytes); }

template <typename Real>
uint32_t chunksOf(size_t n) { return chunksOfBytes(n * sizeof(Real)); }

// The archive's size fields are 32-bit: even an all-raw stream must stay below 4 GiB.
template <typename Real>
bool fitsArchive(size_t n) {
    return n < (static_cast<size_t>(1) << 32) / sizeof(Real) &&
           kStreamHeader + kLcHeader + static_cast<size_t>(chunksOf<Real>(n)) * (4u + kChunkBytes)
         < (static_cast<size_t>(1) << 32);
}

void requireAligned(const void* p, size_t alignment, const char* what) {
    if (reinterpret_cast<uintptr_t>(p) % alignment != 0)
        throw std::invalid_argument(std::string("fz::pfpl: ") + what + " must be " + std::to_string(alignment) +
                                    "-byte aligned");
}

} // namespace

template <typename T>
size_t maxStreamBytes(size_t n) {
    return kStreamHeader + kLcHeader + chunksOf<T>(n) * (4u + static_cast<size_t>(kChunkBytes));
}
template size_t maxStreamBytes<float>(size_t);
template size_t maxStreamBytes<double>(size_t);

size_t streamBytes(const void* h_stream, size_t capacity) {
    if (h_stream == nullptr || capacity < kStreamHeader) return 0;
    StreamHeader h;
    std::memcpy(&h, h_stream, sizeof(h));
    if ((h.magic != kStreamMagic && h.magic != kStreamMagic64) || (h.archive_bytes & 3u) != 0)
        return 0;
    const uint32_t chunks = h.magic == kStreamMagic ? chunksOf<float>(h.n) : chunksOf<double>(h.n);
    if (h.archive_bytes < kLcHeader + 4ull * chunks)
        return 0;
    const size_t total = kStreamHeader + static_cast<size_t>(h.archive_bytes);
    return total <= capacity ? total : 0;
}

Plan::~Plan() {
    // Teardown: nothing useful to do with an error here.
    if (state_ != nullptr) (void)cudaFree(state_);
    if (ticket_ != nullptr) (void)cudaFree(ticket_);
}

void Plan::reserve(size_t n) { reserveBytes(n * sizeof(float)); }

void Plan::reserveBytes(size_t fieldBytes) {
    const size_t chunks = std::max<size_t>(chunksOfBytes(fieldBytes), 1);
    if (ticket_ == nullptr) {
        FZ_CUDA_CHECK(cudaMalloc(&ticket_, sizeof(unsigned long long)));
        clear_ticket_ = true;
        tickets_ = 0;
    }
    if (chunks <= chunks_) return;
    if (state_ != nullptr) {   // calls still in flight may read the old state
        FZ_CUDA_CHECK(cudaDeviceSynchronize());
        FZ_CUDA_CHECK(cudaFree(state_));
        state_ = nullptr;
    }
    FZ_CUDA_CHECK(cudaMalloc(&state_, chunks * sizeof(unsigned long long)));
    clear_state_ = true;
    chunks_ = chunks;
    epoch_  = 0;   // a cleared state: every epoch is new again
}

uint32_t Plan::nextEpoch(fz::stream_t stream) {
    // New memory can hold earlier bytes: a stale ticket is a wild chunk id, and stale look-back words are a wrong
    // offset. Clear it on the calling stream, because a stream-less cudaMemset is not ordered with a non-blocking
    // stream's kernels. See CN-PFPLSTREAM-2.
    if (clear_ticket_) {
        FZ_CUDA_CHECK(cudaMemsetAsync(ticket_, 0, sizeof(unsigned long long), stream));
        clear_ticket_ = false;
    }
    if (clear_state_) {
        FZ_CUDA_CHECK(cudaMemsetAsync(state_, 0, chunks_ * sizeof(unsigned long long), stream));
        clear_state_ = false;
    }
    if (++epoch_ > kEpochMask) {   // after 2^30 calls, start over on a cleared state
        FZ_CUDA_CHECK(cudaMemsetAsync(state_, 0, chunks_ * sizeof(unsigned long long), stream));
        epoch_ = 1;
    }
    return epoch_;
}

struct PlanAccess {
    template <typename Real>
    static void encode(Plan& plan, const Real* d_in, size_t n, const typename Elem<Real>::Params& qp, float abs_eb,
                       void* d_out, size_t capacity, size_t* d_stream_bytes, fz::stream_t stream) {
        requireAligned(d_in, sizeof(Real), "the input");
        requireAligned(d_out, 4, "the stream buffer");
        plan.reserveBytes(n * sizeof(Real));
        const uint32_t epoch = plan.nextEpoch(stream);
        const uint32_t chunks = chunksOf<Real>(n);
        byte* out = static_cast<byte*>(d_out);
        if (chunks == 0) {
            emptyKernel<<<1, 1, 0, stream>>>(out, capacity, Elem<Real>::kMagic, abs_eb, d_stream_bytes);
            FZ_CUDA_CHECK(cudaGetLastError());
            return;
        }
#if defined(FZGMOD_BACKEND_HIP)
        if (!useHarness())
            amd::encodeKernel<Real><<<chunks, amd::kTPB, 0, stream>>>(d_in, static_cast<uint32_t>(n), qp, abs_eb, out,
                                                                      capacity, chunks, plan.state_, plan.ticket_,
                                                                      plan.tickets_, epoch, d_stream_bytes);
        else
#endif
        if constexpr (sizeof(Real) == 4)
            encodeKernel<<<chunks, TPB, 0, stream>>>(d_in, static_cast<uint32_t>(n), chunks * kElems, qp, out, capacity,
                                                     chunks, plan.state_, plan.ticket_, plan.tickets_, epoch,
                                                     d_stream_bytes);
        else
            encodeKernel64<<<chunks, TPB, 0, stream>>>(d_in, static_cast<uint32_t>(n), qp, abs_eb, out, capacity, chunks,
                                                       plan.state_, plan.ticket_, plan.tickets_, epoch, d_stream_bytes);
        FZ_CUDA_CHECK(cudaGetLastError());
        plan.tickets_ += chunks;
    }

    template <typename Real>
    static void decode(Plan& plan, const void* d_stream, Real* d_out, size_t n, fz::stream_t stream) {
        if (n == 0) return;
        if (!fitsArchive<Real>(n))
            throw std::invalid_argument("fz::pfpl::decompress: no stream holds a field of 4 GiB or more");
        requireAligned(d_stream, 4, "the stream");
        requireAligned(d_out, sizeof(Real), "the output");
        plan.reserveBytes(n * sizeof(Real));
        const uint32_t epoch = plan.nextEpoch(stream);
        const uint32_t chunks = chunksOf<Real>(n);
        const byte* in = static_cast<const byte*>(d_stream);
#if defined(FZGMOD_BACKEND_HIP)
        if (!useHarness())
            amd::decodeKernel<Real><<<chunks, amd::kTPB, 0, stream>>>(in, d_out, static_cast<uint32_t>(n), plan.state_,
                                                                      epoch);
        else
#endif
        if constexpr (sizeof(Real) == 4)
            decodeKernel<<<chunks, TPB, 0, stream>>>(in, d_out, static_cast<uint32_t>(n), plan.state_, epoch);
        else
            decodeKernel64<<<chunks, TPB, 0, stream>>>(in, d_out, static_cast<uint32_t>(n), plan.state_, epoch);
        FZ_CUDA_CHECK(cudaGetLastError());
    }
};

void compress(Plan& plan, const float* d_in, size_t n, float abs_eb,
              void* d_out, size_t capacity, size_t* d_stream_bytes, fz::stream_t stream) {
    if (!(abs_eb > 0.0f) || !std::isfinite(abs_eb) || !std::isfinite(0.5f / abs_eb))
        throw std::invalid_argument("fz::pfpl::compress: the error bound must be positive and finite, "
                                    "with a finite quantizer scale 0.5/abs_eb");
    if (!fitsArchive<float>(n))
        throw std::invalid_argument("fz::pfpl::compress: a field of 4 GiB or more (in whole chunks) exceeds "
                                    "the LC archive's 32-bit sizes");

    // examples/presets/pfpl_specialization.toml in ABS mode, as QuantizerStage::getFusedOp() packs it.
    QuantInplaceZigzagParams qp{};
    qp.ebx2_r                = 0.5f / abs_eb;   // verify_reconstruction's operation order
    qp.ebx2                  = 2.0f * abs_eb;
    qp.abs_eb                = abs_eb;
    qp.radius                = kQuantRadius;
    qp.threshold             = std::numeric_limits<float>::infinity();
    qp.verify_reconstruction = 1;
    PlanAccess::encode<float>(plan, d_in, n, qp, abs_eb, d_out, capacity, d_stream_bytes, stream);
}

void compress(Plan& plan, const double* d_in, size_t n, double abs_eb,
              void* d_out, size_t capacity, size_t* d_stream_bytes, fz::stream_t stream) {
    if (!(abs_eb > 0.0) || !std::isfinite(abs_eb))
        throw std::invalid_argument("fz::pfpl::compress: the error bound must be positive and finite");
    // The bound the stream records: never above the one asked for.
    float recorded = static_cast<float>(abs_eb);
    if (static_cast<double>(recorded) > abs_eb) recorded = std::nextafter(recorded, 0.0f);
    if (recorded < std::numeric_limits<float>::min())
        throw std::invalid_argument("fz::pfpl::compress: a double stream records its bound as a float, so the bound "
                                    "must be at least the smallest normal float");
    if (!fitsArchive<double>(n))
        throw std::invalid_argument("fz::pfpl::compress: a field of 4 GiB or more (in whole chunks) exceeds "
                                    "the LC archive's 32-bit sizes");
    const double eb = static_cast<double>(recorded);
    QuantParams64 qp{};
    qp.ebx2_r = 0.5 / eb;
    qp.ebx2   = 2.0 * eb;
    qp.abs_eb = eb;
    PlanAccess::encode<double>(plan, d_in, n, qp, recorded, d_out, capacity, d_stream_bytes, stream);
}

void decompress(Plan& plan, const void* d_stream, float* d_out, size_t n, fz::stream_t stream) {
    PlanAccess::decode<float>(plan, d_stream, d_out, n, stream);
}

void decompress(Plan& plan, const void* d_stream, double* d_out, size_t n, fz::stream_t stream) {
    PlanAccess::decode<double>(plan, d_stream, d_out, n, stream);
}

} // namespace pfpl
} // namespace fz
