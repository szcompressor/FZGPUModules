// SZp / fZ-light — extreme-fast error-bounded compressor, as a fused stage.
//
// Independent GPU reimplementation of the SZp forward/inverse. The upstream
// CPU/OpenMP reference is https://github.com/szcompressor/SZp (MIT); no upstream
// source is vendored or copied. Inner loop, per block of block_size elements:
//   quantize  q_i   = round(x_i / (2*eb))            (linear, signed)
//   predict   d_i   = q_i - q_{i-1}   (d_0 = q_0)    (1-D Lorenzo, block reset)
//   pack      zigzag(d_i) at the block's fixed bit width
// No entropy coder. Archive layout (outputs[0]; SZpConfig lives in the FZM
// stage-config slot):
//   [ meta region : 1 byte/block = width ] [ payload : packed deltas per block ]
// Round-trips against itself; NOT byte-compatible with the reference SZp
// container (see the stage header). hZCCL's compressed-domain arithmetic is out
// of scope — that is a separate HomomorphicOp interface, not a Stage.

#include "fused/szp/szp_stage.h"
#include "stage/stage_registry.h"
#include <cstring>
#include <algorithm>
#include <stdexcept>
#include <string>
#include "backend/algorithms.h"
#include "mem/mempool.h"
#include "cuda_check.h"
#include "log.h"
#include "backend/cub.h"
#include "backend/api.h"
#include <stdexcept>

namespace fz {

namespace szp_detail {

__host__ __device__ inline size_t numBlocks(size_t n, uint32_t bs) {
    return (bs == 0) ? 0 : (n + bs - 1) / bs;
}
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
                                                      unsigned long long& bitpos, int width) {
    unsigned long long u = 0;
    for (int b = 0; b < width; ++b) {
        const uint8_t bit = (buf[bitpos >> 3] >> (bitpos & 7)) & 1u;
        u |= static_cast<unsigned long long>(bit) << b;
        ++bitpos;
    }
    return u;
}
__device__ __forceinline__ unsigned long long zigzag(long long q) {
    return (static_cast<unsigned long long>(q) << 1) ^ static_cast<unsigned long long>(q >> 63);
}
__device__ __forceinline__ long long unzigzag(unsigned long long u) {
    return static_cast<long long>((u >> 1) ^ (0ull - (u & 1ull)));
}
__device__ __forceinline__ int bitWidth(unsigned long long maxu) {
    return (maxu == 0) ? 0 : (64 - __clzll(maxu));
}
template<typename T>
__device__ __forceinline__ long long quant(T x, double eb) {
    return llround((double)x / (2.0 * eb));
}

// Pass 1 — quantize + delta + width + cost. One thread per block.
template<typename T>
__global__ void predictCostKernel(const T* __restrict__ in, size_t n, uint32_t bs,
                                  double eb, uint8_t* __restrict__ width_meta,
                                  uint32_t* __restrict__ cost) {
    const size_t nb = numBlocks(n, bs);
    for (size_t blk = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
         blk < nb; blk += (size_t)gridDim.x * blockDim.x) {
        const size_t start = blk * (size_t)bs;
        const int len = (int)min((size_t)bs, n - start);
        long long prev = 0;
        unsigned long long maxu = 0;
        for (int i = 0; i < len; ++i) {
            const long long q = quant<T>(in[start + i], eb);
            const unsigned long long u = zigzag(q - prev);
            if (u > maxu) maxu = u;
            prev = q;
        }
        const int w = bitWidth(maxu);
        width_meta[blk] = (uint8_t)w;
        cost[blk] = (uint32_t)(((size_t)len * w + 7) / 8);
    }
}

// Pass 2 — encode packed deltas at the scanned per-block byte offsets.
template<typename T>
__global__ void encodeKernel(const T* __restrict__ in, size_t n, uint32_t bs,
                             double eb, const uint8_t* __restrict__ width_meta,
                             const uint32_t* __restrict__ off, uint8_t* __restrict__ payload) {
    const size_t nb = numBlocks(n, bs);
    for (size_t blk = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
         blk < nb; blk += (size_t)gridDim.x * blockDim.x) {
        const size_t start = blk * (size_t)bs;
        const int len = (int)min((size_t)bs, n - start);
        const int w = width_meta[blk];
        uint8_t* codes = payload + off[blk];
        unsigned long long bitpos = 0;
        long long prev = 0;
        for (int i = 0; i < len; ++i) {
            const long long q = quant<T>(in[start + i], eb);
            putBits(codes, bitpos, zigzag(q - prev), w);
            prev = q;
        }
    }
}

__global__ void costFromMetaKernel(const uint8_t* __restrict__ width_meta, size_t n,
                                   uint32_t bs, uint32_t* __restrict__ cost) {
    const size_t nb = numBlocks(n, bs);
    for (size_t blk = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
         blk < nb; blk += (size_t)gridDim.x * blockDim.x) {
        const size_t start = blk * (size_t)bs;
        const int len = (int)min((size_t)bs, n - start);
        cost[blk] = (uint32_t)(((size_t)len * width_meta[blk] + 7) / 8);
    }
}

template<typename T>
__global__ void decodeKernel(const uint8_t* __restrict__ width_meta,
                             const uint8_t* __restrict__ payload,
                             const uint32_t* __restrict__ off, size_t n,
                             uint32_t bs, double eb, T* __restrict__ out) {
    const size_t nb = numBlocks(n, bs);
    for (size_t blk = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
         blk < nb; blk += (size_t)gridDim.x * blockDim.x) {
        const size_t start = blk * (size_t)bs;
        const int len = (int)min((size_t)bs, n - start);
        const int w = width_meta[blk];
        const uint8_t* codes = payload + off[blk];
        unsigned long long bitpos = 0;
        long long q = 0;
        for (int i = 0; i < len; ++i) {
            q += unzigzag(getBits(codes, bitpos, w));   // prefix-sum the deltas
            out[start + i] = (T)((double)q * 2.0 * eb);
        }
    }
}

} // namespace szp_detail

// ─────────────────────────────────────────────────────────────────────────────

template<typename T>
std::vector<size_t> SZpStage<T>::estimateOutputSizes(
    const std::vector<size_t>& input_sizes) const {
    if (is_inverse_) return {num_elements_ * sizeof(T)};
    const size_t in_bytes = input_sizes.empty() ? 0 : input_sizes[0];
    const size_t n  = in_bytes / sizeof(T);
    const size_t nb = szp_detail::numBlocks(n, block_size_);
    // Worst case: 1 width byte/block + up-to-64-bit deltas.
    return {nb + n * 8u + 64u};
}

template<typename T>
size_t SZpStage<T>::estimateScratchBytes(
    const std::vector<size_t>& input_sizes) const {
    const size_t n = is_inverse_ ? num_elements_
        : (input_sizes.empty() ? 0 : input_sizes[0] / sizeof(T));
    const size_t nb = szp_detail::numBlocks(n, block_size_);
    if (nb == 0) return 0;
    size_t cub_tmp = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, cub_tmp,
                                  static_cast<uint32_t*>(nullptr),
                                  static_cast<uint32_t*>(nullptr), nb);
    return nb * (2u * sizeof(uint32_t)) + cub_tmp;
}

template<typename T>
double SZpStage<T>::resolveAbsEb(fz::stream_t stream, MemoryPool* pool,
                                 const T* d_in, size_t n) {
    if (eb_mode_ != SZpErrorMode::NOA) { value_base_ = 0.0; return user_eb_; }
    T *d_min = static_cast<T*>(pool->allocate(sizeof(T), stream, "szp_min"));
    T *d_max = static_cast<T*>(pool->allocate(sizeof(T), stream, "szp_max"));
    auto t1 = fz::backend::withTempStorage(pool, stream, "szp_red_min",
        [&](void* tmp, size_t& b){ cub::DeviceReduce::Min(tmp, b, d_in, d_min, (int)n, stream); });
    auto t2 = fz::backend::withTempStorage(pool, stream, "szp_red_max",
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
void SZpStage<T>::execute(fz::stream_t stream, MemoryPool* pool,
                          const std::vector<void*>& inputs,
                          const std::vector<void*>& outputs,
                          const std::vector<size_t>& sizes) {
    if (inputs.empty() || outputs.empty() || sizes.empty())
        throw std::runtime_error("SZpStage: inputs/outputs/sizes must be non-empty");
    if (pool == nullptr) throw std::runtime_error("SZpStage: requires a MemoryPool");
    namespace d = szp_detail;
    constexpr int kTpb = 256;

    if (is_inverse_) {
        const size_t n = num_elements_;
        if (n == 0) { actual_output_size_ = 0; return; }
        const size_t nb = d::numBlocks(n, block_size_);
        const auto* archive = static_cast<const uint8_t*>(inputs[0]);
        const uint8_t* d_meta    = archive;
        const uint8_t* d_payload = archive + nb;   // 1 width byte per block

        auto* d_cost   = static_cast<uint32_t*>(pool->allocate(sizeof(uint32_t)*nb, stream, "szp_cost"));
        auto* d_offset = static_cast<uint32_t*>(pool->allocate(sizeof(uint32_t)*nb, stream, "szp_offset"));
        const int blocks = (int)((nb + kTpb - 1) / kTpb);
        d::costFromMetaKernel<<<blocks, kTpb, 0, stream>>>(d_meta, n, block_size_, d_cost);
        auto d_tmp = fz::backend::withTempStorage(pool, stream, "szp_cub_tmp",
            [&](void* tmp, size_t& b){
                cub::DeviceScan::ExclusiveSum(tmp, b, d_cost, d_offset, nb, stream); });
        d::decodeKernel<T><<<blocks, kTpb, 0, stream>>>(
            d_meta, d_payload, d_offset, n, block_size_, abs_eb_, static_cast<T*>(outputs[0]));
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
            "SZpStage: SZp is lossy and requires error_bound > 0 (resolved abs_eb <= 0)");

    const size_t nb = d::numBlocks(n, block_size_);
    auto* archive = static_cast<uint8_t*>(outputs[0]);
    uint8_t* d_meta    = archive;
    const size_t meta_region = nb;
    uint8_t* d_payload = archive + meta_region;

    if (nb > scratch_blocks_) {
        if (scratch_pool_ && d_block_cost_)   scratch_pool_->free(d_block_cost_, stream);
        if (scratch_pool_ && d_block_offset_) scratch_pool_->free(d_block_offset_, stream);
        d_block_cost_ = static_cast<uint32_t*>(pool->allocate(
            sizeof(uint32_t) * nb, stream, "szp_cost", /*persistent=*/true));
        d_block_offset_ = static_cast<uint32_t*>(pool->allocate(
            sizeof(uint32_t) * nb, stream, "szp_offset", /*persistent=*/true));
        if (!d_block_cost_ || !d_block_offset_)
            throw std::runtime_error("SZpStage: failed to allocate forward scratch");
        scratch_blocks_ = nb;
        scratch_pool_   = pool;
    }

    const int blocks = (int)((nb + kTpb - 1) / kTpb);
    d::predictCostKernel<T><<<blocks, kTpb, 0, stream>>>(
        d_in, n, block_size_, abs_eb_, d_meta, d_block_cost_);
    auto d_tmp = fz::backend::withTempStorage(pool, stream, "szp_cub_tmp",
        [&](void* tmp, size_t& b){
            cub::DeviceScan::ExclusiveSum(tmp, b, d_block_cost_, d_block_offset_, nb, stream); });
    d::encodeKernel<T><<<blocks, kTpb, 0, stream>>>(
        d_in, n, block_size_, abs_eb_, d_meta, d_block_offset_, d_payload);

    fz::backend::freeTempStorage(pool, d_tmp, stream);

    fwd_num_blocks_     = nb;
    fwd_meta_bytes_     = meta_region;
    actual_output_size_ = meta_region + n * 8u;  // provisional; refined post-sync
}

template<typename T>
void SZpStage<T>::postStreamSync(fz::stream_t stream) {
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
SZpStage<T>::~SZpStage() {
    if (scratch_pool_) {
        if (d_block_cost_)   scratch_pool_->free(d_block_cost_, 0);
        if (d_block_offset_) scratch_pool_->free(d_block_offset_, 0);
    }
}

template class SZpStage<float>;
template class SZpStage<double>;

} // namespace fz

// ── FZM-header reconstruction (self-registered; see stage_registry.h) ─────────
namespace {
fz::Stage* SZp_fromHeader(const uint8_t* config, size_t config_size) {
    using fz::DataType; using fz::SZpStage; using fz::Stage;
    DataType dt = (config_size > 0) ? static_cast<DataType>(config[0]) : DataType::FLOAT32;
    Stage* stage = nullptr;
    if      (dt == DataType::FLOAT32) stage = new SZpStage<float>();
    else if (dt == DataType::FLOAT64) stage = new SZpStage<double>();
    else throw std::runtime_error("Unsupported SZpStage DataType: "
            + std::to_string(static_cast<int>(dt)));
    stage->deserializeHeader(config, config_size);
    return stage;
}
}  // namespace
FZ_REGISTER_STAGE_FACTORY(fz::StageType::SZP, SZp_fromHeader);
