// SZx — ultrafast error-bounded lossy compressor, as a fused stage.
//
// From-the-paper reimplementation (no SZx source vendored). The distinguishing
// feature vs. the cuSZ-style chain is per-block constant/non-constant
// classification with a fixed-length residual coder and NO entropy stage.
//
// Archive layout (outputs[0]; the SZxConfig header lives in the FZM stage-config
// slot, not here):
//   [ meta region : 2 bytes/block = {type, width} ] [ payload region ]
//   payload per block, at the scanned byte offset:
//     constant     : reference value (sizeof(T) bytes), broadcast on decode
//     non-constant : reference value + block_len residual codes, width bits each
// The layout round-trips against itself; it is NOT byte-compatible with the
// reference SZx container (documented in the stage header).

#include "fused/szx/szx_stage.h"
#include "backend/algorithms.h"
#include "mem/mempool.h"
#include "cuda_check.h"
#include "log.h"
#include "backend/cub.h"
#include "backend/api.h"
#include <cstdio>
#include <stdexcept>

namespace fz {

namespace szx_detail {

constexpr int kMetaBytesPerBlock = 2;  // {type, width}

__host__ __device__ inline size_t numBlocks(size_t n, uint32_t bs) {
    return (bs == 0) ? 0 : (n + bs - 1) / bs;
}

// Fixed-length bit I/O. Each block owns a byte-aligned, disjoint payload slice,
// so the read-modify-write in putBits() never races another thread.
__device__ __forceinline__ void putBits(uint8_t* buf, unsigned long long& bitpos,
                                         unsigned long long u, int width) {
    for (int b = 0; b < width; ++b) {
        const unsigned long long byte_i = bitpos >> 3;
        const uint8_t mask = static_cast<uint8_t>(1u << (bitpos & 7));
        if ((u >> b) & 1ull) buf[byte_i] |= mask;
        else                 buf[byte_i] &= static_cast<uint8_t>(~mask);
        ++bitpos;
    }
}
__device__ __forceinline__ unsigned long long getBits(const uint8_t* buf,
                                                      unsigned long long& bitpos,
                                                      int width) {
    unsigned long long u = 0;
    for (int b = 0; b < width; ++b) {
        const uint8_t bit = (buf[bitpos >> 3] >> (bitpos & 7)) & 1u;
        u |= static_cast<unsigned long long>(bit) << b;
        ++bitpos;
    }
    return u;
}
__device__ __forceinline__ unsigned long long zigzag(long long q) {
    return (static_cast<unsigned long long>(q) << 1) ^
           static_cast<unsigned long long>(q >> 63);
}
__device__ __forceinline__ long long unzigzag(unsigned long long u) {
    return static_cast<long long>((u >> 1) ^ (0ull - (u & 1ull)));
}
__device__ __forceinline__ int bitWidth(unsigned long long maxu) {
    return (maxu == 0) ? 0 : (64 - __clzll(maxu));
}

template<typename T>
__device__ __forceinline__ void storeRef(uint8_t* p, T ref) {
    const uint8_t* r = reinterpret_cast<const uint8_t*>(&ref);
    for (size_t b = 0; b < sizeof(T); ++b) p[b] = r[b];
}
template<typename T>
__device__ __forceinline__ T loadRef(const uint8_t* p) {
    T ref; uint8_t* r = reinterpret_cast<uint8_t*>(&ref);
    for (size_t b = 0; b < sizeof(T); ++b) r[b] = p[b];
    return ref;
}

// Pass 1 — classify + cost. One thread per block.
template<typename T>
__global__ void classifyKernel(const T* __restrict__ in, size_t n, uint32_t bs,
                               double eb, uint8_t* __restrict__ meta,
                               uint32_t* __restrict__ cost, T* __restrict__ refs) {
    const size_t nb = numBlocks(n, bs);
    for (size_t blk = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
         blk < nb; blk += (size_t)gridDim.x * blockDim.x) {
        const size_t start = blk * (size_t)bs;
        const int len = (int)min((size_t)bs, n - start);

        double mn = (double)in[start], mx = mn;
        for (int i = 1; i < len; ++i) {
            const double v = (double)in[start + i];
            mn = fmin(mn, v); mx = fmax(mx, v);
        }
        const T   refv = (T)(0.5 * (mn + mx));  // midpoint, stored at T precision
        const double ref = (double)refv;
        refs[blk] = refv;

        if ((mx - mn) <= 2.0 * eb) {            // constant block
            meta[kMetaBytesPerBlock * blk + 0] = 0;
            meta[kMetaBytesPerBlock * blk + 1] = 0;
            cost[blk] = (uint32_t)sizeof(T);
        } else {                                // non-constant block
            unsigned long long maxu = 0;
            for (int i = 0; i < len; ++i) {
                const long long q = llround(((double)in[start + i] - ref) / (2.0 * eb));
                const unsigned long long u = zigzag(q);
                if (u > maxu) maxu = u;
            }
            const int w = bitWidth(maxu);
            meta[kMetaBytesPerBlock * blk + 0] = 1;
            meta[kMetaBytesPerBlock * blk + 1] = (uint8_t)w;
            const size_t packed = ((size_t)len * w + 7) / 8;
            cost[blk] = (uint32_t)(sizeof(T) + packed);
        }
    }
}

// Pass 2 — encode into payload at the scanned per-block byte offsets.
template<typename T>
__global__ void encodeKernel(const T* __restrict__ in, size_t n, uint32_t bs,
                             double eb, const uint8_t* __restrict__ meta,
                             const uint32_t* __restrict__ off,
                             const T* __restrict__ refs, uint8_t* __restrict__ payload) {
    const size_t nb = numBlocks(n, bs);
    for (size_t blk = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
         blk < nb; blk += (size_t)gridDim.x * blockDim.x) {
        const size_t start = blk * (size_t)bs;
        const int len = (int)min((size_t)bs, n - start);
        uint8_t* p = payload + off[blk];
        const T refv = refs[blk];
        storeRef<T>(p, refv);
        if (meta[kMetaBytesPerBlock * blk + 0] == 1) {
            const int w = meta[kMetaBytesPerBlock * blk + 1];
            const double ref = (double)refv;
            uint8_t* codes = p + sizeof(T);
            unsigned long long bitpos = 0;
            for (int i = 0; i < len; ++i) {
                const long long q = llround(((double)in[start + i] - ref) / (2.0 * eb));
                putBits(codes, bitpos, zigzag(q), w);
            }
        }
    }
}

// Inverse helpers: recompute per-block cost from meta, then decode.
__global__ void costFromMetaKernel(const uint8_t* __restrict__ meta, size_t n,
                                   uint32_t bs, uint32_t elem_size,
                                   uint32_t* __restrict__ cost) {
    const size_t nb = numBlocks(n, bs);
    for (size_t blk = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
         blk < nb; blk += (size_t)gridDim.x * blockDim.x) {
        const size_t start = blk * (size_t)bs;
        const int len = (int)min((size_t)bs, n - start);
        if (meta[kMetaBytesPerBlock * blk + 0] == 0) {
            cost[blk] = elem_size;
        } else {
            const int w = meta[kMetaBytesPerBlock * blk + 1];
            cost[blk] = elem_size + (uint32_t)(((size_t)len * w + 7) / 8);
        }
    }
}

template<typename T>
__global__ void decodeKernel(const uint8_t* __restrict__ meta,
                             const uint8_t* __restrict__ payload,
                             const uint32_t* __restrict__ off, size_t n,
                             uint32_t bs, double eb, T* __restrict__ out) {
    const size_t nb = numBlocks(n, bs);
    for (size_t blk = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
         blk < nb; blk += (size_t)gridDim.x * blockDim.x) {
        const size_t start = blk * (size_t)bs;
        const int len = (int)min((size_t)bs, n - start);
        const uint8_t* p = payload + off[blk];
        const T refv = loadRef<T>(p);
        if (meta[kMetaBytesPerBlock * blk + 0] == 0) {
            for (int i = 0; i < len; ++i) out[start + i] = refv;
        } else {
            const int w = meta[kMetaBytesPerBlock * blk + 1];
            const double ref = (double)refv;
            const uint8_t* codes = p + sizeof(T);
            unsigned long long bitpos = 0;
            for (int i = 0; i < len; ++i) {
                const long long q = unzigzag(getBits(codes, bitpos, w));
                out[start + i] = (T)(ref + (double)q * 2.0 * eb);
            }
        }
    }
}

} // namespace szx_detail

// ─────────────────────────────────────────────────────────────────────────────

template<typename T>
std::vector<size_t> SZxStage<T>::estimateOutputSizes(
    const std::vector<size_t>& input_sizes) const {
    if (is_inverse_) return {num_elements_ * sizeof(T)};
    const size_t in_bytes = input_sizes.empty() ? 0 : input_sizes[0];
    const size_t n  = in_bytes / sizeof(T);
    const size_t nb = szx_detail::numBlocks(n, block_size_);
    // Worst case: every block non-constant, one ref + up-to-64-bit codes each.
    const size_t meta    = szx_detail::kMetaBytesPerBlock * nb;
    const size_t payload = nb * sizeof(T) + n * 8u;
    return {meta + payload + 64u};
}

template<typename T>
size_t SZxStage<T>::estimateScratchBytes(
    const std::vector<size_t>& input_sizes) const {
    const size_t n = is_inverse_ ? num_elements_
        : (input_sizes.empty() ? 0 : input_sizes[0] / sizeof(T));
    const size_t nb = szx_detail::numBlocks(n, block_size_);
    if (nb == 0) return 0;
    size_t cub_tmp = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, cub_tmp,
                                  static_cast<uint32_t*>(nullptr),
                                  static_cast<uint32_t*>(nullptr), nb);
    // cost + offset (uint32) + refs (T); cub temp; (+ decode reuses these).
    return nb * (2u * sizeof(uint32_t) + sizeof(T)) + cub_tmp;
}

// Resolve the absolute bound. ABS is graph-clean; NOA needs a range reduce +
// host read, so the forward path reports itself non-graph-compatible under NOA.
template<typename T>
double SZxStage<T>::resolveAbsEb(fz::stream_t stream, MemoryPool* pool,
                                 const T* d_in, size_t n) {
    if (eb_mode_ != SZxErrorMode::NOA) { value_base_ = 0.0; return user_eb_; }
    T *d_min = static_cast<T*>(pool->allocate(sizeof(T), stream, "szx_min"));
    T *d_max = static_cast<T*>(pool->allocate(sizeof(T), stream, "szx_max"));
    auto t1 = fz::backend::withTempStorage(pool, stream, "szx_red_min",
        [&](void* tmp, size_t& b){ cub::DeviceReduce::Min(tmp, b, d_in, d_min, (int)n, stream); });
    auto t2 = fz::backend::withTempStorage(pool, stream, "szx_red_max",
        [&](void* tmp, size_t& b){ cub::DeviceReduce::Max(tmp, b, d_in, d_max, (int)n, stream); });
    T h_min = 0, h_max = 0;
    FZ_CUDA_CHECK(cudaMemcpyAsync(&h_min, d_min, sizeof(T), cudaMemcpyDeviceToHost, stream));
    FZ_CUDA_CHECK(cudaMemcpyAsync(&h_max, d_max, sizeof(T), cudaMemcpyDeviceToHost, stream));
    FZ_CUDA_CHECK(cudaStreamSynchronize(stream));
    fz::backend::freeTempStorage(pool, t1, stream);
    fz::backend::freeTempStorage(pool, t2, stream);
    pool->free(d_max, stream);
    pool->free(d_min, stream);
    value_base_ = (double)h_max - (double)h_min;
    return user_eb_ * value_base_;
}

template<typename T>
void SZxStage<T>::execute(fz::stream_t stream, MemoryPool* pool,
                          const std::vector<void*>& inputs,
                          const std::vector<void*>& outputs,
                          const std::vector<size_t>& sizes) {
    if (inputs.empty() || outputs.empty() || sizes.empty())
        throw std::runtime_error("SZxStage: inputs/outputs/sizes must be non-empty");
    if (pool == nullptr) throw std::runtime_error("SZxStage: requires a MemoryPool");
    namespace d = szx_detail;

    constexpr int kTpb = 256;

    if (is_inverse_) {
        const size_t n = num_elements_;
        if (n == 0) { actual_output_size_ = 0; return; }
        const size_t nb = d::numBlocks(n, block_size_);
        const auto* archive = static_cast<const uint8_t*>(inputs[0]);
        const uint8_t* d_meta    = archive;
        const uint8_t* d_payload = archive + d::kMetaBytesPerBlock * nb;

        auto* d_cost   = static_cast<uint32_t*>(
            pool->allocate(sizeof(uint32_t) * nb, stream, "szx_cost"));
        auto* d_offset = static_cast<uint32_t*>(
            pool->allocate(sizeof(uint32_t) * nb, stream, "szx_offset"));

        const int blocks = (int)((nb + kTpb - 1) / kTpb);
        d::costFromMetaKernel<<<blocks, kTpb, 0, stream>>>(
            d_meta, n, block_size_, (uint32_t)sizeof(T), d_cost);
        auto d_tmp = fz::backend::withTempStorage(pool, stream, "szx_cub_tmp",
            [&](void* tmp, size_t& b){
                cub::DeviceScan::ExclusiveSum(tmp, b, d_cost, d_offset, nb, stream); });
        d::decodeKernel<T><<<blocks, kTpb, 0, stream>>>(
            d_meta, d_payload, d_offset, n, block_size_, abs_eb_,
            static_cast<T*>(outputs[0]));

        actual_output_size_ = n * sizeof(T);
        fz::backend::freeTempStorage(pool, d_tmp, stream);
        pool->free(d_offset, stream);
        pool->free(d_cost, stream);
        return;
    }

    // ── Forward ──────────────────────────────────────────────────────────────
    const size_t n = sizes[0] / sizeof(T);
    num_elements_ = n;
    if (n == 0) { actual_output_size_ = 0; return; }
    const T* d_in = static_cast<const T*>(inputs[0]);

    abs_eb_ = resolveAbsEb(stream, pool, d_in, n);
    if (!(abs_eb_ > 0.0))
        throw std::runtime_error(
            "SZxStage: SZx is lossy and requires error_bound > 0 (resolved abs_eb <= 0)");

    const size_t nb = d::numBlocks(n, block_size_);
    auto* archive = static_cast<uint8_t*>(outputs[0]);
    uint8_t* d_meta    = archive;
    const size_t meta_region = d::kMetaBytesPerBlock * nb;
    uint8_t* d_payload = archive + meta_region;

    // Persistent cost/offset (for the deferred size readback + graph replay),
    // grown only when a larger input is seen — mirrors AdaptiveBitpackStage.
    if (nb > scratch_blocks_) {
        if (scratch_pool_ && d_block_cost_)   scratch_pool_->free(d_block_cost_, stream);
        if (scratch_pool_ && d_block_offset_) scratch_pool_->free(d_block_offset_, stream);
        d_block_cost_ = static_cast<uint32_t*>(pool->allocate(
            sizeof(uint32_t) * nb, stream, "szx_cost", /*persistent=*/true));
        d_block_offset_ = static_cast<uint32_t*>(pool->allocate(
            sizeof(uint32_t) * nb, stream, "szx_offset", /*persistent=*/true));
        if (!d_block_cost_ || !d_block_offset_)
            throw std::runtime_error("SZxStage: failed to allocate forward scratch");
        scratch_blocks_ = nb;
        scratch_pool_   = pool;
    }
    // Per-block reference values: transient (only needed between the two passes).
    T* d_refs = static_cast<T*>(pool->allocate(sizeof(T) * nb, stream, "szx_refs"));

    const int blocks = (int)((nb + kTpb - 1) / kTpb);
    d::classifyKernel<T><<<blocks, kTpb, 0, stream>>>(
        d_in, n, block_size_, abs_eb_, d_meta, d_block_cost_, d_refs);
    auto d_tmp = fz::backend::withTempStorage(pool, stream, "szx_cub_tmp",
        [&](void* tmp, size_t& b){
            cub::DeviceScan::ExclusiveSum(tmp, b, d_block_cost_, d_block_offset_, nb, stream); });
    d::encodeKernel<T><<<blocks, kTpb, 0, stream>>>(
        d_in, n, block_size_, abs_eb_, d_meta, d_block_offset_, d_refs, d_payload);

    fz::backend::freeTempStorage(pool, d_tmp, stream);
    pool->free(d_refs, stream);

    fwd_num_blocks_     = nb;
    fwd_meta_bytes_     = meta_region;
    // Provisional (valid during graph recording); refined in postStreamSync().
    actual_output_size_ = meta_region + nb * sizeof(T) + n * 8u;
}

template<typename T>
void SZxStage<T>::postStreamSync(fz::stream_t stream) {
    if (is_inverse_ || d_block_offset_ == nullptr || fwd_num_blocks_ == 0) return;
    const size_t last = fwd_num_blocks_ - 1;
    uint32_t h_off = 0, h_cost = 0;
    FZ_CUDA_CHECK(cudaMemcpyAsync(&h_off, d_block_offset_ + last,
                                  sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
    FZ_CUDA_CHECK(cudaMemcpyAsync(&h_cost, d_block_cost_ + last,
                                  sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
    FZ_CUDA_CHECK(cudaStreamSynchronize(stream));
    actual_output_size_ = fwd_meta_bytes_ + (size_t)h_off + h_cost;
}

template<typename T>
std::vector<std::string> SZxStage<T>::getRunNotes() const {
    std::vector<std::string> notes;
    if (!is_inverse_ && const_block_frac_ > 0.0) {
        char buf[96];
        std::snprintf(buf, sizeof(buf),
            "SZx: %.1f%% of blocks classified constant", 100.0 * const_block_frac_);
        notes.emplace_back(buf);
    }
    return notes;
}

template<typename T>
SZxStage<T>::~SZxStage() {
    if (scratch_pool_) {
        if (d_block_cost_)   scratch_pool_->free(d_block_cost_, 0);
        if (d_block_offset_) scratch_pool_->free(d_block_offset_, 0);
    }
}

template class SZxStage<float>;
template class SZxStage<double>;

} // namespace fz
