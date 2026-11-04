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
constexpr uint32_t kLcHeader     = 8;                          // LC archive: [u32 coded bytes][u32 chunks]
constexpr uint32_t kStreamHeader = sizeof(StreamHeader);
constexpr int      kWave         = fz::backend::kWaveSize;

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

// One block per chunk: code it in shared memory, find its offset, write its
// size-table entry and payload into the stream; the last chunk adds the headers
// and the length. Layout: [StreamHeader][LC header][u32 sizes x chunks][payloads].
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
            w[4] = coded_n * 4u;
            w[5] = chunks;
        }
        // The stream header goes in whenever it fits, so a stream too long for the
        // buffer still records the length it needed.
        if (kStreamHeader <= capacity) {
            w[0] = kStreamMagic;
            w[1] = static_cast<uint32_t>(padded);
            w[2] = __float_as_uint(qp.abs_eb);
            w[3] = n;
        }
        if (d_stream_bytes != nullptr) *d_stream_bytes = static_cast<size_t>(total);
    }
}

// A field of no elements: the headers alone.
__global__ void emptyKernel(byte* __restrict__ out, unsigned long long capacity, float abs_eb,
                            size_t* __restrict__ d_stream_bytes) {
    const unsigned long long total = kStreamHeader + kLcHeader;
    uint32_t* w = reinterpret_cast<uint32_t*>(out);
    if (kStreamHeader <= capacity) {
        w[0] = kStreamMagic;
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

// One block per chunk (gridDim.x == chunks): find the chunk's payload, decode it.
__global__ void __launch_bounds__(TPB)
decodeKernel(const byte* __restrict__ stream, float* __restrict__ out, uint32_t n,
             unsigned long long* __restrict__ state, uint32_t epoch) {
    const uint32_t* w = reinterpret_cast<const uint32_t*>(stream);
    const uint32_t chunks = gridDim.x;
    if (w[0] != kStreamMagic || w[3] != n || w[4] != chunks * kElems * 4u || w[5] != chunks ||
        w[1] < kLcHeader + 4u * chunks || !(__uint_as_float(w[2]) > 0.0f)) {
        streamTrap();
    }
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

#if defined(FZGMOD_BACKEND_HIP)
#include "fused/chunk_fusion/pfpl_stream_amd.inl"
#endif

// FZ_PFPL_HARNESS=1 runs the generic harness kernels instead of the AMD ones (same stream; for comparison).
bool useHarness() {
#if defined(FZGMOD_BACKEND_HIP)
    static const bool harness = [] {
        const char* e = std::getenv("FZ_PFPL_HARNESS");
        return e != nullptr && e[0] == '1';
    }();
    return harness;
#else
    return true;
#endif
}

uint32_t chunksOf(size_t n) { return static_cast<uint32_t>((n + kElems - 1) / kElems); }

// The archive's size fields are 32-bit: even an all-raw stream must stay below 4 GiB.
bool fitsArchive(size_t n) {
    return kStreamHeader + kLcHeader + static_cast<size_t>(chunksOf(n)) * (4u + kChunkBytes)
         < (static_cast<size_t>(1) << 32);
}

void requireAligned(const void* p, const char* what) {
    if (reinterpret_cast<uintptr_t>(p) % 4u != 0)
        throw std::invalid_argument(std::string("fz::pfpl: ") + what + " must be 4-byte aligned");
}

} // namespace

size_t maxStreamBytes(size_t n) {
    const size_t chunks = chunksOf(n);
    return kStreamHeader + kLcHeader + chunks * (4u + static_cast<size_t>(kChunkBytes));
}

size_t streamBytes(const void* h_stream, size_t capacity) {
    if (h_stream == nullptr || capacity < kStreamHeader) return 0;
    StreamHeader h;
    std::memcpy(&h, h_stream, sizeof(h));
    if (h.magic != kStreamMagic || (h.archive_bytes & 3u) != 0 ||
        h.archive_bytes < kLcHeader + 4ull * chunksOf(h.n))
        return 0;
    const size_t total = kStreamHeader + static_cast<size_t>(h.archive_bytes);
    return total <= capacity ? total : 0;
}

Plan::~Plan() {
    // Teardown: nothing useful to do with an error here.
    if (state_ != nullptr) (void)cudaFree(state_);
    if (ticket_ != nullptr) (void)cudaFree(ticket_);
}

void Plan::reserve(size_t n) {
    const size_t chunks = std::max<size_t>(chunksOf(n), 1);
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

void compress(Plan& plan, const float* d_in, size_t n, float abs_eb,
              void* d_out, size_t capacity, size_t* d_stream_bytes, fz::stream_t stream) {
    if (!(abs_eb > 0.0f) || !std::isfinite(abs_eb) || !std::isfinite(0.5f / abs_eb))
        throw std::invalid_argument("fz::pfpl::compress: the error bound must be positive and finite, "
                                    "with a finite quantizer scale 0.5/abs_eb");
    if (!fitsArchive(n))
        throw std::invalid_argument("fz::pfpl::compress: a field of 4 GiB or more (in whole chunks) exceeds "
                                    "the LC archive's 32-bit sizes");
    requireAligned(d_out, "the stream buffer");
    plan.reserve(n);
    const uint32_t epoch = plan.nextEpoch(stream);

    // examples/presets/pfpl_specialization.toml in ABS mode, as QuantizerStage::getFusedOp() packs it.
    QuantInplaceZigzagParams qp{};
    qp.ebx2_r                = 0.5f / abs_eb;   // verify_reconstruction's operation order
    qp.ebx2                  = 2.0f * abs_eb;
    qp.abs_eb                = abs_eb;
    qp.radius                = kQuantRadius;
    qp.threshold             = std::numeric_limits<float>::infinity();
    qp.verify_reconstruction = 1;

    const uint32_t chunks = chunksOf(n);
    if (chunks == 0) {
        emptyKernel<<<1, 1, 0, stream>>>(static_cast<byte*>(d_out), capacity, abs_eb, d_stream_bytes);
        FZ_CUDA_CHECK(cudaGetLastError());
        return;
    }
#if defined(FZGMOD_BACKEND_HIP)
    if (!useHarness())
        amd::encodeKernel<<<chunks, amd::kTPB, 0, stream>>>(d_in, static_cast<uint32_t>(n), qp,
                                                            static_cast<byte*>(d_out), capacity, chunks,
                                                            plan.state_, plan.ticket_, plan.tickets_, epoch,
                                                            d_stream_bytes);
    else
#endif
    encodeKernel<<<chunks, TPB, 0, stream>>>(d_in, static_cast<uint32_t>(n), chunks * kElems, qp,
                                             static_cast<byte*>(d_out), capacity, chunks, plan.state_,
                                             plan.ticket_, plan.tickets_, epoch, d_stream_bytes);
    FZ_CUDA_CHECK(cudaGetLastError());
    plan.tickets_ += chunks;
}

void decompress(Plan& plan, const void* d_stream, float* d_out, size_t n, fz::stream_t stream) {
    if (n == 0) return;
    if (!fitsArchive(n))
        throw std::invalid_argument("fz::pfpl::decompress: no stream holds a field of 4 GiB or more");
    requireAligned(d_stream, "the stream");
    requireAligned(d_out, "the output");
    plan.reserve(n);
    const uint32_t epoch = plan.nextEpoch(stream);
#if defined(FZGMOD_BACKEND_HIP)
    if (!useHarness())
        amd::decodeKernel<<<chunksOf(n), amd::kTPB, 0, stream>>>(static_cast<const byte*>(d_stream), d_out,
                                                                 static_cast<uint32_t>(n), plan.state_, epoch);
    else
#endif
    decodeKernel<<<chunksOf(n), TPB, 0, stream>>>(static_cast<const byte*>(d_stream), d_out,
                                                  static_cast<uint32_t>(n), plan.state_, epoch);
    FZ_CUDA_CHECK(cudaGetLastError());
}

} // namespace pfpl
} // namespace fz
