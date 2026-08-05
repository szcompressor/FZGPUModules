/**
 * @file adaptive_lorenzo_stage.cu
 * @brief Per-tile adaptive multi-order Lorenzo predictor with centering.
 *
 * Prior work: the cross-block prediction state, four-variant adaptive selection,
 * and the finite-difference cancellation that collapses the evaluation into a
 * single data read are the design of FSZ (Jiajun Huang, "FSZ: Breaking the
 * Prediction-Throughput Trade-off in GPU Lossy Compression", SC'26,
 * arXiv:2607.15413). This is an independent reimplementation as a modular DAG
 * stage; no FSZ source was used. See THIRD_PARTY.md.
 *
 * Residual algebra, with a = q - mu when centering is on (mu = 0 otherwise),
 * d1 = delta(a) and d2 = delta(d1), all zero-padded at the tile start:
 *
 *   LZ1        out_0 = q_0,        out_i = q_i - q_{i-1}
 *   LZ1 + cen  out_0 = q_0 - mu,   out_i = q_i - q_{i-1}
 *   LZ2        out_0 = q_0,        out_1 = d1_1 - q_0,          out_i = d2_i
 *   LZ2 + cen  out_0 = q_0 - mu,   out_1 = d1_1 - (q_0 - mu),   out_i = d2_i
 *
 * so centering perturbs only the first one (LZ1) or two (LZ2) residuals, all of
 * which live in coder block 0 — the whole basis for costing four variants from
 * one read. The inverse is one scan (LZ1) or two (LZ2), then a uniform `+ mu`.
 */

#include "fused/adaptive_lorenzo/adaptive_lorenzo_stage.h"
#include "mem/mempool.h"
#include "cuda_check.h"
#include "backend/api.h"
#include "backend/warp.h"
#include "backend/algorithms.h"
#include "backend/cub.h"
#include <stdexcept>
#include <string>

namespace fz {

namespace {

constexpr uint32_t kMaxBlocksPerTile = 32;
constexpr uint8_t  kModeOrder2       = 0x1;
constexpr uint8_t  kModeCentering    = 0x2;
constexpr uint32_t kNoVariant        = 0xFFFFFFFFu;

// Two's-complement magnitude (well-defined for INT_MIN), matching the
// AdaptiveBitpack encoder's own convention.
template<typename T>
__device__ __forceinline__ typename std::make_unsigned<T>::type absU(T v) {
    using U = typename std::make_unsigned<T>::type;
    U uv = static_cast<U>(v);
    return (v < 0) ? static_cast<U>(~uv + static_cast<U>(1)) : uv;
}

__device__ __forceinline__ int bitWidth32(uint32_t x) {
    return x ? (32 - __clz(x)) : 0;
}

// AdaptiveBitpackStage's per-block encoded size: nothing when every residual is
// zero, else a sign bitmap plus `r` bit-planes, one 32-bit word each.
__device__ __forceinline__ uint32_t blockCost(int r) {
    return (r > 0) ? 4u * (static_cast<uint32_t>(r) + 1u) : 0u;
}

// Modes are packed 2 bits per tile (4 tiles per byte): at one byte per tile the
// mode map alone costs 0.031 bits/element at a 256-element tile, which is a
// visible fraction of the output on very sparse fields.
__device__ __host__ __forceinline__ uint8_t unpackMode(const uint8_t* modes, size_t tile) {
    return static_cast<uint8_t>((modes[tile >> 2] >> ((tile & 3u) * 2u)) & 0x3u);
}

// Compact the dense per-tile means down to only the tiles that chose centering,
// and pack the mode bytes 4-to-1. One thread per output mode byte.
template<typename T>
__global__ void adaptive_lorenzo_compact_kernel(
    const uint8_t*  __restrict__ modes_dense,
    const T*        __restrict__ means_dense,
    const uint32_t* __restrict__ offsets,
    uint8_t*        __restrict__ modes_packed,
    T*              __restrict__ means_compact,
    size_t num_tiles)
{
    const size_t byte_idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t num_bytes = (num_tiles + 3) / 4;
    if (byte_idx >= num_bytes) return;

    uint8_t packed = 0;
    for (unsigned k = 0; k < 4u; ++k) {
        const size_t t = byte_idx * 4 + k;
        if (t >= num_tiles) break;
        const uint8_t m = modes_dense[t] & 0x3u;
        packed = static_cast<uint8_t>(packed | (m << (k * 2u)));
        if (m & kModeCentering) means_compact[offsets[t]] = means_dense[t];
    }
    modes_packed[byte_idx] = packed;
}

// Inverse-side counterpart: rebuild the centering flags from the packed modes so
// the same exclusive scan reproduces the compaction offsets. Nothing extra has
// to be stored for this.
__global__ void adaptive_lorenzo_flags_kernel(
    const uint8_t* __restrict__ modes_packed,
    uint32_t*      __restrict__ flags,
    size_t num_tiles)
{
    const size_t t = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t > num_tiles) return;
    flags[t] = (t < num_tiles && (unpackMode(modes_packed, t) & kModeCentering)) ? 1u : 0u;
}

// OR-reduce across a warp: the OR shares its top set bit with the max, which is
// all the bit width depends on.
__device__ __forceinline__ uint32_t warpOr(uint32_t v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v |= fz::backend::shflXor(v, off, 32);
    return v;
}

template<typename T>
__global__ void adaptive_lorenzo_forward_kernel(
    const T* __restrict__ in,
    T*       __restrict__ residuals,
    uint8_t*  __restrict__ modes,   // dense scratch, 1 byte per tile
    T*        __restrict__ means,   // dense scratch, 1 mean per tile
    uint32_t* __restrict__ flags,   // 1 if this tile chose centering
    size_t n,
    uint32_t tile_size,
    bool enable_order2,
    bool enable_centering)
{
    extern __shared__ char smem[];
    T*         s   = reinterpret_cast<T*>(smem);
    long long* red = reinterpret_cast<long long*>(smem + tile_size * sizeof(T));

    __shared__ uint32_t acc1[kMaxBlocksPerTile];  // per coder block, LZ1
    __shared__ uint32_t acc2[kMaxBlocksPerTile];  // per coder block, LZ2
    __shared__ T        s_mu;
    __shared__ T        s_q0;                     // tile's first value, kept live
    __shared__ uint8_t  s_mode;

    const size_t   base   = static_cast<size_t>(blockIdx.x) * tile_size;
    const unsigned tid    = threadIdx.x;
    const size_t   gid    = base + tid;
    const bool     live   = (gid < n);
    const unsigned warp   = tid >> 5;
    const unsigned lane   = tid & 31u;
    const unsigned nwarps = tile_size >> 5;

    const T v = live ? in[gid] : static_cast<T>(0);
    if (tid == 0) s_q0 = v;

    // ---- Tile mean ----
    if (enable_centering) {
        red[tid] = live ? static_cast<long long>(v) : 0LL;
        __syncthreads();
        unsigned p = 1u;
        while (p < tile_size) p <<= 1;
        for (unsigned stride = p >> 1; stride > 0u; stride >>= 1) {
            if (tid < stride && tid + stride < tile_size) red[tid] += red[tid + stride];
            __syncthreads();
        }
        if (tid == 0) {
            // Trailing partial tile: divide by the live count, not tile_size.
            const long long count =
                static_cast<long long>(min(static_cast<size_t>(tile_size), n - base));
            const long long tot = red[0];
            s_mu = static_cast<T>((tot >= 0) ? (tot + count / 2) / count
                                             : (tot - count / 2) / count);
        }
    } else if (tid == 0) {
        s_mu = static_cast<T>(0);
    }
    __syncthreads();

    // ---- First and second differences across the whole tile ----
    s[tid] = v;
    __syncthreads();
    const T d1 = static_cast<T>(s[tid] - ((tid > 0u) ? s[tid - 1] : static_cast<T>(0)));
    __syncthreads();
    s[tid] = d1;
    __syncthreads();
    const T d2 = static_cast<T>(s[tid] - ((tid > 0u) ? s[tid - 1] : static_cast<T>(0)));
    __syncthreads();

    const T mu = s_mu;
    const T q0 = s_q0;

    // ---- Per-coder-block magnitudes, uncentered ----
    const uint32_t o1 = warpOr(live ? static_cast<uint32_t>(absU<T>(d1)) : 0u);
    const uint32_t o2 = warpOr(live ? static_cast<uint32_t>(absU<T>(d2)) : 0u);
    if (lane == 0) { acc1[warp] = o1; acc2[warp] = o2; }

    // ---- Centered variants: only coder block 0 can differ ----
    uint32_t acc1c0 = 0u, acc2c0 = 0u;
    if (enable_centering && warp == 0u) {
        const T c0 = static_cast<T>(q0 - mu);
        const T r1 = (tid == 0u) ? c0 : d1;
        T       r2 = d2;
        if      (tid == 0u) r2 = c0;
        else if (tid == 1u) r2 = static_cast<T>(d1 - c0);
        acc1c0 = warpOr(live ? static_cast<uint32_t>(absU<T>(r1)) : 0u);
        acc2c0 = warpOr(live ? static_cast<uint32_t>(absU<T>(r2)) : 0u);
    }
    __syncthreads();

    // ---- Cost each variant, pick the cheapest ----
    if (tid == 0) {
        uint32_t c_lz1 = 0, c_lz2 = 0;
        for (unsigned w = 0; w < nwarps; ++w) {
            c_lz1 += blockCost(bitWidth32(acc1[w]));
            c_lz2 += blockCost(bitWidth32(acc2[w]));
        }
        uint32_t costs[4];
        costs[0] = c_lz1;
        costs[1] = enable_order2 ? c_lz2 : kNoVariant;
        costs[2] = kNoVariant;
        costs[3] = kNoVariant;
        if (enable_centering) {
            // Swap coder block 0's contribution for its centered rate, and pay
            // sizeof(T) for the mean. That charge is real: means are compacted
            // to only the tiles that chose centering, so declining centering
            // genuinely saves those bytes. (The 2-bit mode is charged to nobody
            // because every tile pays it regardless of what it picks.)
            const uint32_t mean_cost = static_cast<uint32_t>(sizeof(T));
            costs[2] = c_lz1 - blockCost(bitWidth32(acc1[0]))
                             + blockCost(bitWidth32(acc1c0)) + mean_cost;
            if (enable_order2)
                costs[3] = c_lz2 - blockCost(bitWidth32(acc2[0]))
                                 + blockCost(bitWidth32(acc2c0)) + mean_cost;
        }
        uint32_t best = 0;
        for (uint32_t i = 1; i < 4; ++i)
            if (costs[i] < costs[best]) best = i;

        s_mode = static_cast<uint8_t>(((best & 1u) ? kModeOrder2    : 0u)
                                    | ((best & 2u) ? kModeCentering : 0u));
        modes[blockIdx.x] = s_mode;
        means[blockIdx.x] = mu;
        flags[blockIdx.x] = (s_mode & kModeCentering) ? 1u : 0u;
    }
    __syncthreads();

    // ---- Emit the winner's residuals ----
    if (!live) return;
    const uint8_t mode = s_mode;
    const bool    ord2 = (mode & kModeOrder2) != 0;
    const bool    cent = (mode & kModeCentering) != 0;
    const T       c0   = static_cast<T>(q0 - mu);

    T out;
    if (!ord2) {
        out = (cent && tid == 0u) ? c0 : d1;
    } else if (!cent) {
        out = d2;
    } else {
        if      (tid == 0u) out = c0;
        else if (tid == 1u) out = static_cast<T>(d1 - c0);
        else                out = d2;
    }
    residuals[gid] = out;
}

template<typename T>
__global__ void adaptive_lorenzo_inverse_kernel(
    const T*        __restrict__ residuals,
    const uint8_t*  __restrict__ modes,    // packed, 2 bits per tile
    const T*        __restrict__ means,    // compacted to centered tiles only
    const uint32_t* __restrict__ offsets,  // exclusive scan of the centering bits
    T*              __restrict__ out,
    size_t n,
    uint32_t tile_size)
{
    extern __shared__ char smem[];
    T* s = reinterpret_cast<T*>(smem);

    const size_t gid = static_cast<size_t>(blockIdx.x) * tile_size + threadIdx.x;
    const int    tid = static_cast<int>(threadIdx.x);

    const uint8_t mode = unpackMode(modes, blockIdx.x);
    const bool    ord2 = (mode & kModeOrder2) != 0;
    const bool    cent = (mode & kModeCentering) != 0;

    s[tid] = (gid < n) ? residuals[gid] : static_cast<T>(0);
    __syncthreads();

    // One inclusive scan undoes one difference; LZ2 needs two, in the same order.
    const int passes = ord2 ? 2 : 1;
    for (int pass = 0; pass < passes; ++pass) {
        for (int stride = 1; stride < static_cast<int>(tile_size); stride <<= 1) {
            T val = (tid >= stride) ? s[tid - stride] : static_cast<T>(0);
            __syncthreads();
            s[tid] += val;
            __syncthreads();
        }
    }

    T q = s[tid];
    if (cent) q = static_cast<T>(q + means[offsets[blockIdx.x]]);
    if (gid < n) out[gid] = q;
}

}  // namespace

template<typename T>
void launchAdaptiveLorenzoForward(
    const T* d_input, T* d_residuals, uint8_t* d_modes_dense, T* d_means_dense,
    uint32_t* d_flags, size_t n, uint32_t tile_size,
    bool enable_order2, bool enable_centering, cudaStream_t stream)
{
    if (n == 0) return;
    const int    grid  = static_cast<int>((n + tile_size - 1) / tile_size);
    const size_t shmem = tile_size * (sizeof(T) + sizeof(long long));
    adaptive_lorenzo_forward_kernel<T><<<grid, tile_size, shmem, stream>>>(
        d_input, d_residuals, d_modes_dense, d_means_dense, d_flags, n, tile_size,
        enable_order2, enable_centering);
    FZ_CUDA_CHECK(cudaGetLastError());
}

template<typename T>
void launchAdaptiveLorenzoCompact(
    const uint8_t* d_modes_dense, const T* d_means_dense, const uint32_t* d_offsets,
    uint8_t* d_modes_packed, T* d_means_compact, size_t num_tiles,
    cudaStream_t stream)
{
    if (num_tiles == 0) return;
    const size_t num_bytes = (num_tiles + 3) / 4;
    const int    kBlk = 256;
    const int    grid = static_cast<int>((num_bytes + kBlk - 1) / kBlk);
    adaptive_lorenzo_compact_kernel<T><<<grid, kBlk, 0, stream>>>(
        d_modes_dense, d_means_dense, d_offsets, d_modes_packed, d_means_compact,
        num_tiles);
    FZ_CUDA_CHECK(cudaGetLastError());
}

void launchAdaptiveLorenzoFlags(
    const uint8_t* d_modes_packed, uint32_t* d_flags, size_t num_tiles,
    cudaStream_t stream)
{
    const int kBlk = 256;
    const int grid = static_cast<int>((num_tiles + 1 + kBlk - 1) / kBlk);
    adaptive_lorenzo_flags_kernel<<<grid, kBlk, 0, stream>>>(
        d_modes_packed, d_flags, num_tiles);
    FZ_CUDA_CHECK(cudaGetLastError());
}

template<typename T>
void launchAdaptiveLorenzoInverse(
    const T* d_residuals, const uint8_t* d_modes, const T* d_means,
    const uint32_t* d_offsets, T* d_output,
    size_t n, uint32_t tile_size, cudaStream_t stream)
{
    if (n == 0) return;
    const int grid = static_cast<int>((n + tile_size - 1) / tile_size);
    adaptive_lorenzo_inverse_kernel<T><<<grid, tile_size, tile_size * sizeof(T), stream>>>(
        d_residuals, d_modes, d_means, d_offsets, d_output, n, tile_size);
    FZ_CUDA_CHECK(cudaGetLastError());
}

template<typename T>
size_t AdaptiveLorenzoStage<T>::estimateScratchBytes(
    const std::vector<size_t>& input_sizes) const {
    const size_t n = (is_inverse_ || input_sizes.empty())
        ? num_elements_ : input_sizes[0] / sizeof(T);
    const size_t tiles = numTiles(n);
    if (tiles == 0) return 0;
    size_t cub_tmp = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, cub_tmp,
                                  static_cast<uint32_t*>(nullptr),
                                  static_cast<uint32_t*>(nullptr), tiles + 1);
    return tiles * (1 + sizeof(T)) + 2 * (tiles + 1) * sizeof(uint32_t) + cub_tmp;
}

template<typename T>
void AdaptiveLorenzoStage<T>::releaseScratch() {
    if (!scratch_pool_) return;
    if (d_modes_dense_) scratch_pool_->free(d_modes_dense_, 0);
    if (d_means_dense_) scratch_pool_->free(d_means_dense_, 0);
    if (d_flags_)       scratch_pool_->free(d_flags_, 0);
    if (d_offsets_)     scratch_pool_->free(d_offsets_, 0);
    d_modes_dense_ = nullptr; d_means_dense_ = nullptr;
    d_flags_ = nullptr; d_offsets_ = nullptr;
    scratch_tiles_ = 0; scratch_pool_ = nullptr;
}

template<typename T>
size_t AdaptiveLorenzoStage<T>::ensureScratch(size_t num_tiles, MemoryPool* pool,
                                              cudaStream_t stream) {
    if (num_tiles <= scratch_tiles_) return num_tiles;
    if (scratch_pool_) {
        if (d_modes_dense_) scratch_pool_->free(d_modes_dense_, stream);
        if (d_means_dense_) scratch_pool_->free(d_means_dense_, stream);
        if (d_flags_)       scratch_pool_->free(d_flags_, stream);
        if (d_offsets_)     scratch_pool_->free(d_offsets_, stream);
    }
    d_modes_dense_ = static_cast<uint8_t*>(pool->allocate(
        num_tiles, stream, "alrz_modes", /*persistent=*/true));
    d_means_dense_ = static_cast<T*>(pool->allocate(
        num_tiles * sizeof(T), stream, "alrz_means", true));
    // One extra slot so the exclusive scan's last entry is the total count.
    d_flags_ = static_cast<uint32_t*>(pool->allocate(
        (num_tiles + 1) * sizeof(uint32_t), stream, "alrz_flags", true));
    d_offsets_ = static_cast<uint32_t*>(pool->allocate(
        (num_tiles + 1) * sizeof(uint32_t), stream, "alrz_offsets", true));
    if (!d_modes_dense_ || !d_means_dense_ || !d_flags_ || !d_offsets_)
        throw std::runtime_error("AdaptiveLorenzoStage: failed to allocate scratch");
    scratch_tiles_ = num_tiles;
    scratch_pool_  = pool;
    return num_tiles;
}

template<typename T>
void AdaptiveLorenzoStage<T>::execute(
    cudaStream_t stream,
    MemoryPool* pool,
    const std::vector<void*>& inputs,
    const std::vector<void*>& outputs,
    const std::vector<size_t>& sizes)
{
    if (inputs.empty() || outputs.empty() || sizes.empty())
        throw std::runtime_error(
            "AdaptiveLorenzoStage: inputs, outputs, and sizes must be non-empty");

    const size_t byte_size = sizes[0];
    if (byte_size == 0) {
        actual_output_sizes_.assign(is_inverse_ ? 1 : 3, 0);
        return;
    }

    const uint32_t tile = getTileSize();

    if (!is_inverse_) {
        const size_t n     = byte_size / sizeof(T);
        num_elements_      = n;
        const size_t tiles = numTiles(n);
        if (outputs.size() < 3 || outputs[1] == nullptr || outputs[2] == nullptr)
            throw std::runtime_error(
                "AdaptiveLorenzoStage: the 'modes' and 'means' output ports must be "
                "connected or left as pipeline outputs");

        ensureScratch(tiles, pool, stream);

        launchAdaptiveLorenzoForward<T>(
            static_cast<const T*>(inputs[0]), static_cast<T*>(outputs[0]),
            d_modes_dense_, d_means_dense_, d_flags_,
            n, tile, config_.enable_order2, config_.enable_centering, stream);

        // flags[tiles] = 0 so offsets[tiles] lands on the total centered count.
        FZ_CUDA_CHECK(cudaMemsetAsync(d_flags_ + tiles, 0, sizeof(uint32_t), stream));

        auto d_tmp = fz::backend::withTempStorage(pool, stream, "alrz_cub_tmp",
            [&](void* tmp, size_t& bytes) {
                cub::DeviceScan::ExclusiveSum(tmp, bytes, d_flags_, d_offsets_,
                                              tiles + 1, stream);
            });

        launchAdaptiveLorenzoCompact<T>(
            d_modes_dense_, d_means_dense_, d_offsets_,
            static_cast<uint8_t*>(outputs[1]), static_cast<T*>(outputs[2]),
            tiles, stream);

        fz::backend::freeTempStorage(pool, d_tmp, stream);

        actual_output_sizes_.resize(3);
        actual_output_sizes_[0] = byte_size;
        actual_output_sizes_[1] = (tiles + 3) / 4;      // 2 bits per tile
        // The means length is data-dependent; postStreamSync() reads the scanned
        // total once the stream is idle. Assume the worst until then.
        actual_output_sizes_[2] = tiles * sizeof(T);
        pending_tiles_ = tiles;
        return;
    }

    if (inputs.size() < 3 || inputs[1] == nullptr || inputs[2] == nullptr)
        throw std::runtime_error(
            "AdaptiveLorenzoStage: inverse requires the 'modes' and 'means' inputs");

    const size_t n     = byte_size / sizeof(T);
    const size_t tiles = numTiles(n);
    ensureScratch(tiles, pool, stream);

    // Rebuild the compaction offsets from the packed modes — the same scan the
    // forward ran, so no slot table has to be stored.
    launchAdaptiveLorenzoFlags(
        static_cast<const uint8_t*>(inputs[1]), d_flags_, tiles, stream);

    auto d_tmp = fz::backend::withTempStorage(pool, stream, "alrz_cub_tmp",
        [&](void* tmp, size_t& bytes) {
            cub::DeviceScan::ExclusiveSum(tmp, bytes, d_flags_, d_offsets_,
                                          tiles + 1, stream);
        });

    launchAdaptiveLorenzoInverse<T>(
        static_cast<const T*>(inputs[0]),
        static_cast<const uint8_t*>(inputs[1]),
        static_cast<const T*>(inputs[2]),
        d_offsets_,
        static_cast<T*>(outputs[0]), n, tile, stream);

    fz::backend::freeTempStorage(pool, d_tmp, stream);
    actual_output_sizes_.assign(1, byte_size);
}

template<typename T>
void AdaptiveLorenzoStage<T>::postStreamSync(cudaStream_t stream) {
    if (is_inverse_ || pending_tiles_ == 0 || d_offsets_ == nullptr) return;
    uint32_t centered = 0;
    FZ_CUDA_CHECK(cudaMemcpyAsync(&centered, d_offsets_ + pending_tiles_,
                                  sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
    FZ_CUDA_CHECK(cudaStreamSynchronize(stream));
    if (actual_output_sizes_.size() >= 3)
        actual_output_sizes_[2] = static_cast<size_t>(centered) * sizeof(T);
    pending_tiles_ = 0;
}

template class AdaptiveLorenzoStage<int16_t>;
template class AdaptiveLorenzoStage<int32_t>;

template void launchAdaptiveLorenzoForward<int16_t>(
    const int16_t*, int16_t*, uint8_t*, int16_t*, uint32_t*, size_t, uint32_t, bool, bool, cudaStream_t);
template void launchAdaptiveLorenzoForward<int32_t>(
    const int32_t*, int32_t*, uint8_t*, int32_t*, uint32_t*, size_t, uint32_t, bool, bool, cudaStream_t);

template void launchAdaptiveLorenzoCompact<int16_t>(
    const uint8_t*, const int16_t*, const uint32_t*, uint8_t*, int16_t*, size_t, cudaStream_t);
template void launchAdaptiveLorenzoCompact<int32_t>(
    const uint8_t*, const int32_t*, const uint32_t*, uint8_t*, int32_t*, size_t, cudaStream_t);

template void launchAdaptiveLorenzoInverse<int16_t>(
    const int16_t*, const uint8_t*, const int16_t*, const uint32_t*, int16_t*, size_t, uint32_t, cudaStream_t);
template void launchAdaptiveLorenzoInverse<int32_t>(
    const int32_t*, const uint8_t*, const int32_t*, const uint32_t*, int32_t*, size_t, uint32_t, cudaStream_t);

}  // namespace fz
