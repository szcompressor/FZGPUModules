#pragma once

/**
 * @file rle.h
 * @brief Run-Length Encoding stage (lossless, stream-ordered).
 */

#include "stage/stage.h"
#include "fzm_format.h"
#include "log.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace fz {

/**
 * Run-Length Encoding stage. Lossless; effective when data has long runs of
 * identical values (e.g. quantized codes).
 *
 * Forward wire format: `[num_runs:u32][values:T×n (4B-aligned)][lengths:u32×n]`
 *
 * Worst-case output is 2× input + 4 bytes (no repeated values), so RLE should
 * follow a predictor/quantizer stage that creates repetition.
 *
 * @tparam T  Element type (`uint8_t`, `uint16_t`, `uint32_t`, …). Run counts
 *            are always `uint32_t`.
 */
template<typename T = uint16_t>
class RLEStage : public Stage {
public:
    RLEStage() : is_inverse_(false) {}
    ~RLEStage() override;

    void setInverse(bool inverse) override { is_inverse_ = inverse; }
    bool isInverse() const override { return is_inverse_; }

    void execute(
        cudaStream_t stream,
        MemoryPool* pool,
        const std::vector<void*>& inputs,
        const std::vector<void*>& outputs,
        const std::vector<size_t>& sizes
    ) override;

    /**
     * Completes the async D2H readback of num_runs started during forward
     * execute() and sets actual_output_sizes_.  Must be called after the
     * stream passed to execute() has been synchronized.
     */
    void postStreamSync(cudaStream_t stream) override;
    
    std::string getName() const override { return "RLE"; }
    size_t getNumInputs() const override { return 1; }
    size_t getNumOutputs() const override { return 1; }

    /**
     * Persistent forward-path scratch:
     *   d_is_boundary_     : n bytes
     *   d_boundary_scan_   : n × u32
     *   d_boundary_positions_: n × u32  (worst-case, avoids D2H for num_runs)
     *   d_values_scratch_  : n × T
     *   d_lengths_scratch_ : n × u32
     * All five arrays are sized to the largest n seen so far and reused
     * across calls, eliminating per-call cudaMallocAsync overhead.
     */
    size_t estimateScratchBytes(
        const std::vector<size_t>& input_sizes
    ) const override {
        if (is_inverse_ || input_sizes.empty()) return 0;
        const size_t n = input_sizes[0] / sizeof(T);
        // is_boundary(1B) + boundary_scan(4B) + boundary_positions(4B)
        //   + values_scratch(sizeof(T)) + lengths_scratch(4B)
        return n * (1 + 4 + 4 + sizeof(T) + 4);
    }
    
    std::vector<size_t> estimateOutputSizes(
        const std::vector<size_t>& input_sizes
    ) const override {
        if (is_inverse_) {
            // Use the element count cached from the forward pass (or deserialized
            // from the file header) for an exact estimate.  Falls back to a
            // conservative 2× bound only when no prior forward pass has run.
            if (cached_num_elements_ > 0)
                return {static_cast<size_t>(cached_num_elements_) * sizeof(T)};
            return {input_sizes[0] * 2};
        } else {
            // Compression: worst case is every element is unique.
            // Wire format: [num_runs:u32][values:T×n, 4B-aligned][lengths:u32×n]
            // The values section is padded to a 4-byte boundary (matching
            // rle_pack_kernel), so the estimate must include that padding or
            // the allocated buffer will be too small and the lengths write OOBs.
            size_t n = input_sizes[0] / sizeof(T);
            size_t values_bytes   = n * sizeof(T);
            size_t values_aligned = (values_bytes + 3u) & ~3u;
            return {sizeof(uint32_t) + values_aligned + n * sizeof(uint32_t)};
        }
    }
    
    std::unordered_map<std::string, size_t> getActualOutputSizesByName() const override {
        completePendingSync();
        return {{"output", actual_output_sizes_.empty() ? 0 : actual_output_sizes_[0]}};
    }
    size_t getActualOutputSize(int index) const override {
        completePendingSync();
        return (index == 0 && !actual_output_sizes_.empty()) ? actual_output_sizes_[0] : 0;
    }

    uint16_t getStageTypeId() const override {
        return static_cast<uint16_t>(StageType::RLE);
    }
    
    uint8_t getOutputDataType(size_t output_index) const override {
        (void)output_index;
        return static_cast<uint8_t>(getDataTypeEnum());
    }

    uint8_t getInputDataType(size_t /*input_index*/) const override {
        return static_cast<uint8_t>(getDataTypeEnum());
    }
    
    size_t serializeHeader(size_t output_index, uint8_t* header_buffer, size_t max_size) const override {
        (void)output_index;
        const size_t needed = sizeof(DataType) + sizeof(uint32_t);
        if (max_size < needed) return 0;
        DataType dt = getDataTypeEnum();
        std::memcpy(header_buffer, &dt, sizeof(DataType));
        std::memcpy(header_buffer + sizeof(DataType), &cached_num_elements_, sizeof(uint32_t));
        return needed;
    }

    void deserializeHeader(const uint8_t* header_buffer, size_t size) override {
        if (size >= sizeof(DataType) + sizeof(uint32_t))
            std::memcpy(&cached_num_elements_, header_buffer + sizeof(DataType), sizeof(uint32_t));
    }

    size_t getMaxHeaderSize(size_t output_index) const override {
        (void)output_index;
        return sizeof(DataType) + sizeof(uint32_t);
    }
    
private:
    bool is_inverse_;

    /// Cached original element count from the most recent forward pass.
    /// Persisted in the serialized header so inverse `estimateOutputSizes()`
    /// returns an exact bound even for cold decompression from file.
    uint32_t cached_num_elements_ = 0;

    // ── Persistent forward-path scratch ──────────────────────────────────────
    // Allocated lazily on the first forward execute(); grown if n increases.
    uint8_t*    d_is_boundary_        = nullptr;
    uint32_t*   d_boundary_scan_      = nullptr;
    uint32_t*   d_boundary_positions_ = nullptr; ///< Worst-case n elements.
    T*          d_values_scratch_     = nullptr;
    uint32_t*   d_lengths_scratch_    = nullptr;
    size_t      fwd_scratch_n_        = 0;        ///< Current scratch capacity (elements).
    MemoryPool* fwd_scratch_pool_     = nullptr;
    bool        fwd_from_pool_        = false;

    // Pinned host buffer for async D2H of num_runs.
    // mutable so getActualOutputSizesByName() can complete the pending
    // readback even when called on a const Stage reference.
    mutable uint32_t*           h_num_runs_          = nullptr;
    mutable bool                fwd_sync_pending_    = false;
    mutable cudaStream_t        fwd_last_stream_     = nullptr;
    mutable std::vector<size_t> actual_output_sizes_;

    // Complete a pending forward-path readback (if any) by syncing the stream
    // that was used and computing actual_output_sizes_.  Safe to call from
    // const methods; all state it touches is mutable.
    void completePendingSync() const {
        if (!fwd_sync_pending_) return;
        cudaStreamSynchronize(fwd_last_stream_);
        const uint32_t num_runs      = *h_num_runs_;
        const size_t   values_bytes  = num_runs * sizeof(T);
        const size_t   values_aligned = (values_bytes + 3) & ~3;
        actual_output_sizes_ = {
            sizeof(uint32_t) + values_aligned + num_runs * sizeof(uint32_t)
        };
        fwd_sync_pending_ = false;
        // Log run count and effective compression ratio.
        const size_t in_bytes  = static_cast<size_t>(cached_num_elements_) * sizeof(T);
        const size_t out_bytes = actual_output_sizes_[0];
        const float  ratio     = in_bytes > 0
            ? static_cast<float>(in_bytes) / static_cast<float>(out_bytes) : 0.0f;
        FZ_LOG(DEBUG, "RLE encode: %u runs / %u elems  %.1f KB -> %.1f KB  ratio %.2fx",
               num_runs, cached_num_elements_,
               in_bytes / 1024.0f, out_bytes / 1024.0f, ratio);
    }

    // Helper to map template type T to DataType enum
    DataType getDataTypeEnum() const {
        if (std::is_same<T, uint8_t>::value) return DataType::UINT8;
        if (std::is_same<T, uint16_t>::value) return DataType::UINT16;
        if (std::is_same<T, uint32_t>::value) return DataType::UINT32;
        if (std::is_same<T, uint64_t>::value) return DataType::UINT64;
        if (std::is_same<T, int8_t>::value) return DataType::INT8;
        if (std::is_same<T, int16_t>::value) return DataType::INT16;
        if (std::is_same<T, int32_t>::value) return DataType::INT32;
        if (std::is_same<T, int64_t>::value) return DataType::INT64;
        if (std::is_same<T, float>::value) return DataType::FLOAT32;
        if (std::is_same<T, double>::value) return DataType::FLOAT64;
        return DataType::UINT8;  // Fallback
    }
};

extern template class RLEStage<uint8_t>;
extern template class RLEStage<uint16_t>;
extern template class RLEStage<uint32_t>;
extern template class RLEStage<int32_t>;

} // namespace fz
