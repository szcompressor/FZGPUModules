// Per-block adaptive fixed-rate bit-plane coder (cuSZp-style plain mode).
//
// Prior work: the per-block fixed-rate bit-plane scheme is the cuSZp lossless
// back-end (Yafan Huang et al., SC'23/SC'24, BSD-3-Clause). The stage wrapper,
// byte-granular layout, and CUB offset scan are FZGPUModules code. The cuSZp
// decoupled look-back scan is intentionally replaced by a plain device-wide
// exclusive scan — fusion is a downstream-compiler concern. See THIRD_PARTY.md
// and memory/cuszp_stages.md.

#include "coders/adaptive_bitpack/adaptive_bitpack_stage.h"
#include "coders/adaptive_bitpack/adaptive_bitpack_kernels.h"
#include "mem/mempool.h"
#include "cuda_check.h"
#include "log.h"

#include <cub/cub.cuh>
#include <cuda_runtime.h>
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

        size_t tmp_bytes = 0;
        cub::DeviceScan::ExclusiveSum(nullptr, tmp_bytes, d_cost, d_offset,
                                      cfg.num_blocks, stream);
        auto* d_tmp = pool->allocate(tmp_bytes, stream, "ab_cub_tmp");
        cub::DeviceScan::ExclusiveSum(d_tmp, tmp_bytes, d_cost, d_offset,
                                      cfg.num_blocks, stream);

        if (outlier_selection_)
            ab::launchDecodeUnpackOutlier<T>(d_meta, d_offset, d_payload, cfg,
                                             static_cast<T*>(outputs[0]), stream);
        else
            ab::launchDecodeUnpack<T>(d_meta, d_offset, d_payload, cfg,
                                      static_cast<T*>(outputs[0]), stream);

        actual_output_size_ = n * sizeof(T);
        pool->free(d_tmp, stream);
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

    auto* d_cost   = static_cast<uint32_t*>(
        pool->allocate(sizeof(uint32_t) * cfg.num_blocks, stream, "ab_cost"));
    auto* d_offset = static_cast<uint32_t*>(
        pool->allocate(sizeof(uint32_t) * cfg.num_blocks, stream, "ab_offset"));

    if (outlier_selection_) ab::launchEncodeRateOutlier<T>(d_in, cfg, d_meta, d_cost, stream);
    else                    ab::launchEncodeRate<T>(d_in, cfg, d_meta, d_cost, stream);

    size_t tmp_bytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, tmp_bytes, d_cost, d_offset,
                                  cfg.num_blocks, stream);
    auto* d_tmp = pool->allocate(tmp_bytes, stream, "ab_cub_tmp");
    cub::DeviceScan::ExclusiveSum(d_tmp, tmp_bytes, d_cost, d_offset,
                                  cfg.num_blocks, stream);

    // Total payload = exclusive_offset[last] + cost[last].
    uint32_t h_last_off = 0, h_last_cost = 0;
    FZ_CUDA_CHECK(cudaMemcpyAsync(&h_last_off, d_offset + (cfg.num_blocks - 1),
                                  sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
    FZ_CUDA_CHECK(cudaMemcpyAsync(&h_last_cost, d_cost + (cfg.num_blocks - 1),
                                  sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
    FZ_CUDA_CHECK(cudaStreamSynchronize(stream));
    const size_t total_payload = static_cast<size_t>(h_last_off) + h_last_cost;

    if (outlier_selection_)
        ab::launchEncodePackOutlier<T>(d_in, cfg, d_meta, d_offset, d_payload, stream);
    else
        ab::launchEncodePack<T>(d_in, cfg, d_meta, d_offset, d_payload, stream);

    actual_output_size_ = meta_region + total_payload;

    pool->free(d_tmp, stream);
    pool->free(d_offset, stream);
    pool->free(d_cost, stream);
}

// ── Explicit instantiations ─────────────────────────────────────────────────
template class AdaptiveBitpackStage<int16_t>;
template class AdaptiveBitpackStage<int32_t>;

} // namespace fz
