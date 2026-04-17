#pragma once

#include "stage/stage.h"
#include "fzm_format.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace fz {

/**
 * Run-Length Encoding (RLE) stage
 * 
 * Forward (compression): Encodes consecutive identical values as (value, run_length) pairs
 *   Format: [num_runs (uint32_t)] [value1, count1, value2, count2, ...]
 *   Example: [1,1,1,2,2,3] → [3] [1,3, 2,2, 3,1]
 * 
 * Inverse (decompression): Expands (value, run_length) pairs back to original sequence
 *   Example: [3] [1,3, 2,2, 3,1] → [1,1,1,2,2,3]
 * 
 * This is a lossless encoding particularly effective for data with long runs
 * of identical values (e.g., quantized codes with many repeated values).
 * 
 * Template parameter T determines value data type (uint16_t, uint32_t, etc.)
 * Run counts are always stored as uint32_t.
 * 
 * Note: For data with no repetition, RLE can increase size (worst case: 2x + 4 bytes)
 */
template<typename T = uint16_t>
class RLEStage : public Stage {
public:
    RLEStage() : is_inverse_(false) {}
    ~RLEStage() override;

    /**
     * Set inverse mode for decompression
     * When true, performs run expansion instead of run encoding
     */
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
            // Compression: worst case is every element is unique
            // Format: sizeof(uint32_t) + n * (sizeof(T) + sizeof(uint32_t))
            size_t n = input_sizes[0] / sizeof(T);
            return {sizeof(uint32_t) + n * (sizeof(T) + sizeof(uint32_t))};
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

    // Cached original element count from the most recent forward pass.
    // Set during forward execute(); persisted in the serialized header so that
    // inverse estimateOutputSizes() returns an exact bound even for cold
    // decompression reconstructed from a file.
    uint32_t cached_num_elements_ = 0;

    // ── Persistent forward-path scratch ──────────────────────────────────────
    // Allocated lazily on the first forward execute(); grown if n increases.
    // All device arrays are allocated from the pool when one is available,
    // otherwise via cudaMalloc (tracked by fwd_from_pool_).
    uint8_t*  d_is_boundary_      = nullptr;
    uint32_t* d_boundary_scan_    = nullptr;
    uint32_t* d_boundary_positions_ = nullptr;  // worst-case n elements
    T*        d_values_scratch_   = nullptr;
    uint32_t* d_lengths_scratch_  = nullptr;
    size_t    fwd_scratch_n_      = 0;           // current capacity (elements)
    MemoryPool* fwd_scratch_pool_ = nullptr;
    bool      fwd_from_pool_      = false;

    // Pinned host buffer for async D2H of num_runs.
    // mutable so getActualOutputSizesByName() can complete the pending
    // readback even when called on a const Stage reference.
    mutable uint32_t*    h_num_runs_       = nullptr;
    mutable bool         fwd_sync_pending_ = false;
    mutable cudaStream_t fwd_last_stream_  = nullptr;
    mutable std::vector<size_t> actual_output_sizes_;  // promoted to mutable

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

// Explicit instantiations for common types
extern template class RLEStage<uint8_t>;
extern template class RLEStage<uint16_t>;
extern template class RLEStage<uint32_t>;
extern template class RLEStage<int32_t>;

} // namespace fz
