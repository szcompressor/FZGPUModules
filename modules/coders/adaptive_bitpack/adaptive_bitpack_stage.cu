// Per-block adaptive fixed-rate bit-plane coder (cuSZp-style plain mode).
//
// Prior work: the per-block fixed-rate bit-plane scheme is the cuSZp lossless
// back-end (Yafan Huang et al., SC'23/SC'24, BSD-3-Clause). The stage wrapper,
// byte-granular layout, and CUB offset scan are FZGPUModules code. The cuSZp
// decoupled look-back scan is intentionally replaced by a plain device-wide
// exclusive scan — fusion is a downstream-compiler concern. See THIRD_PARTY.md.

#include "coders/adaptive_bitpack/adaptive_bitpack_stage.h"
#include "stage/stage_registry.h"
#include <cstring>
#include <algorithm>
#include <stdexcept>
#include <string>
#include "coders/adaptive_bitpack/adaptive_bitpack_kernels.h"
#include "backend/algorithms.h"
#include "mem/mempool.h"
#include "cuda_check.h"
#include "log.h"

#include "backend/cub.h"
#include "backend/api.h"
#include <stdexcept>

namespace fz {

namespace ab = adaptive_bitpack;

template<typename T>
std::vector<size_t> AdaptiveBitpackStage<T>::estimateOutputSizes(
    const std::vector<size_t>& input_sizes) const {
    if (is_inverse_) {
        // Inverse output size comes from the element count in the FZM header.
        return {num_elements_ * sizeof(T)};
    }
    const size_t in_bytes = input_sizes.empty() ? 0 : input_sizes[0];
    const size_t n = in_bytes / sizeof(T);
    ab::Config cfg = ab::configure(n, block_size_, outlier_selection_);
    return {ab::maxArchiveBytes(cfg, 8u * sizeof(T))};
}

template<typename T>
size_t AdaptiveBitpackStage<T>::estimateScratchBytes(
    const std::vector<size_t>& input_sizes) const {
    // Per-block cost + offset arrays and the CUB scan temp are allocated from the
    // pool inside execute() (both forward and inverse). Forward derives the block
    // count from the code input bytes; inverse from the header element count.
    size_t num_blocks;
    if (is_inverse_) {
        num_blocks = (block_size_ == 0)
            ? 0 : (num_elements_ + block_size_ - 1) / block_size_;
    } else {
        const size_t in_bytes = input_sizes.empty() ? 0 : input_sizes[0];
        const size_t n = in_bytes / sizeof(T);
        num_blocks = (block_size_ == 0) ? 0 : (n + block_size_ - 1) / block_size_;
    }
    if (num_blocks == 0) return 0;

    size_t cub_tmp = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, cub_tmp,
                                  static_cast<uint32_t*>(nullptr),
                                  static_cast<uint32_t*>(nullptr),
                                  num_blocks);
    return 2u * num_blocks * sizeof(uint32_t) + cub_tmp;
}

template<typename T>
void AdaptiveBitpackStage<T>::execute(
    cudaStream_t stream,
    MemoryPool* pool,
    const std::vector<void*>& inputs,
    const std::vector<void*>& outputs,
    const std::vector<size_t>& sizes) {

    if (inputs.empty() || outputs.empty() || sizes.empty())
        throw std::runtime_error(
            "AdaptiveBitpackStage: inputs, outputs, and sizes must be non-empty");
    if (pool == nullptr)
        throw std::runtime_error("AdaptiveBitpackStage: requires a MemoryPool");

    // ── Decode ───────────────────────────────────────────────────────────────
    if (is_inverse_) {
        const size_t n = num_elements_;
        if (n == 0) { actual_output_size_ = 0; return; }
        ab::Config cfg = ab::configure(n, block_size_, outlier_selection_);

        const auto* archive = static_cast<const uint8_t*>(inputs[0]);
        const uint8_t* d_meta    = archive;  // metadata region (meta_bytes per block)
        const uint8_t* d_payload = archive + cfg.meta_bytes * cfg.num_blocks;

        auto* d_cost   = static_cast<uint32_t*>(
            pool->allocate(sizeof(uint32_t) * cfg.num_blocks, stream, "ab_cost"));
        auto* d_offset = static_cast<uint32_t*>(
            pool->allocate(sizeof(uint32_t) * cfg.num_blocks, stream, "ab_offset"));

        if (outlier_selection_) ab::launchDecodeCostOutlier(d_meta, cfg, d_cost, stream);
        else                    ab::launchDecodeCost(d_meta, cfg, d_cost, stream);

        auto d_tmp = fz::backend::withTempStorage(pool, stream, "ab_cub_tmp",
            [&](void* tmp, size_t& bytes) {
                cub::DeviceScan::ExclusiveSum(tmp, bytes, d_cost, d_offset,
                                              cfg.num_blocks, stream);
            });

        if (outlier_selection_)
            ab::launchDecodeUnpackOutlier<T>(d_meta, d_offset, d_payload, cfg,
                                             static_cast<T*>(outputs[0]), stream);
        else
            ab::launchDecodeUnpack<T>(d_meta, d_offset, d_payload, cfg,
                                      static_cast<T*>(outputs[0]), stream);

        actual_output_size_ = n * sizeof(T);
        fz::backend::freeTempStorage(pool, d_tmp, stream);
        pool->free(d_offset, stream);
        pool->free(d_cost, stream);
        return;
    }

    // ── Encode ─────────────────────────────────────────────────────────────
    const size_t in_bytes = sizes[0];
    const size_t n = in_bytes / sizeof(T);
    num_elements_ = n;
    if (n == 0) { actual_output_size_ = 0; return; }

    ab::Config cfg = ab::configure(n, block_size_, outlier_selection_);
    const T* d_in   = static_cast<const T*>(inputs[0]);
    auto*    archive = static_cast<uint8_t*>(outputs[0]);
    uint8_t* d_meta    = archive;  // metadata region (meta_bytes per block)
    const size_t meta_region = cfg.meta_bytes * cfg.num_blocks;
    uint8_t* d_payload = archive + meta_region;

    // Persistent per-block cost/offset scratch, grown only when a larger input is
    // seen. Keeping it across calls (a) lets postStreamSync() read the scanned
    // length after the stream is idle instead of a host-blocking D2H here, and
    // (b) avoids any allocation inside a captured graph replay. This is what
    // makes the forward path graph-compatible (mirrors RZEStage's forward).
    if (cfg.num_blocks > scratch_blocks_) {
        if (scratch_pool_ && d_cost_)   scratch_pool_->free(d_cost_, stream);
        if (scratch_pool_ && d_offset_) scratch_pool_->free(d_offset_, stream);
        d_cost_ = static_cast<uint32_t*>(pool->allocate(
            sizeof(uint32_t) * cfg.num_blocks, stream, "ab_cost", /*persistent=*/true));
        d_offset_ = static_cast<uint32_t*>(pool->allocate(
            sizeof(uint32_t) * cfg.num_blocks, stream, "ab_offset", /*persistent=*/true));
        if (!d_cost_ || !d_offset_)
            throw std::runtime_error(
                "AdaptiveBitpackStage: failed to allocate persistent forward scratch");
        scratch_blocks_ = cfg.num_blocks;
        scratch_pool_   = pool;
    }

    if (outlier_selection_) ab::launchEncodeRateOutlier<T>(d_in, cfg, d_meta, d_cost_, stream);
    else                    ab::launchEncodeRate<T>(d_in, cfg, d_meta, d_cost_, stream);

    auto d_tmp = fz::backend::withTempStorage(pool, stream, "ab_cub_tmp",
        [&](void* tmp, size_t& bytes) {
            cub::DeviceScan::ExclusiveSum(tmp, bytes, d_cost_, d_offset_,
                                          cfg.num_blocks, stream);
        });

    if (outlier_selection_)
        ab::launchEncodePackOutlier<T>(d_in, cfg, d_meta, d_offset_, d_payload, stream);
    else
        ab::launchEncodePack<T>(d_in, cfg, d_meta, d_offset_, d_payload, stream);

    fz::backend::freeTempStorage(pool, d_tmp, stream);

    // The real archive length (meta_region + total payload) needs the scanned
    // tail, which we read in postStreamSync() once the stream is idle — doing a
    // D2H here would forbid CUDA graph capture. Record what postStreamSync needs
    // and set a worst-case provisional size (postStreamSync is not invoked during
    // graph *recording*, so actual_output_size_ must be valid right now too).
    fwd_num_blocks_     = cfg.num_blocks;
    fwd_meta_region_    = meta_region;
    actual_output_size_ = ab::maxArchiveBytes(cfg, 8u * sizeof(T));
}

template<typename T>
void AdaptiveBitpackStage<T>::postStreamSync(cudaStream_t stream) {
    if (is_inverse_ || fwd_num_blocks_ == 0) return;
    if (d_offset_ == nullptr) return;
    // Batch both tail reads on the caller's stream; one stream-scoped sync covers both.
    const size_t last = fwd_num_blocks_ - 1;
    uint32_t h_last_off = 0, h_last_cost = 0;
    FZ_CUDA_CHECK(cudaMemcpyAsync(&h_last_off, d_offset_ + last,
                                  sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
    FZ_CUDA_CHECK(cudaMemcpyAsync(&h_last_cost, d_cost_ + last,
                                  sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
    FZ_CUDA_CHECK(cudaStreamSynchronize(stream));
    actual_output_size_ =
        fwd_meta_region_ + static_cast<size_t>(h_last_off) + h_last_cost;
}

template<typename T>
AdaptiveBitpackStage<T>::~AdaptiveBitpackStage() {
    if (scratch_pool_) {
        if (d_cost_)   scratch_pool_->free(d_cost_, 0);
        if (d_offset_) scratch_pool_->free(d_offset_, 0);
    }
}

// ── Explicit instantiations ─────────────────────────────────────────────────
template class AdaptiveBitpackStage<int16_t>;
template class AdaptiveBitpackStage<int32_t>;

} // namespace fz

// ── FZM-header reconstruction (self-registered; see stage_registry.h) ─────────
namespace {
fz::Stage* AdaptiveBitpack_fromHeader(const uint8_t* config, size_t config_size) {
    using fz::DataType; using fz::AdaptiveBitpackStage; using fz::Stage;
    DataType dt = (config_size > 0) ? static_cast<DataType>(config[0]) : DataType::INT32;
    Stage* stage = nullptr;
    if      (dt == DataType::INT16) stage = new AdaptiveBitpackStage<int16_t>();
    else if (dt == DataType::INT32) stage = new AdaptiveBitpackStage<int32_t>();
    else throw std::runtime_error("Unsupported AdaptiveBitpackStage DataType: "
            + std::to_string(static_cast<int>(dt)));
    stage->deserializeHeader(config, config_size);
    return stage;
}
}  // namespace
FZ_REGISTER_STAGE_FACTORY(fz::StageType::ADAPTIVE_BITPACK, AdaptiveBitpack_fromHeader);
