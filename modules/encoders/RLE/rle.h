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
    
    std::string getName() const override { return "RLE"; }
    size_t getNumInputs() const override { return 1; }
    size_t getNumOutputs() const override { return 1; }
    
    std::vector<size_t> estimateOutputSizes(
        const std::vector<size_t>& input_sizes
    ) const override {
        if (is_inverse_) {
            // Decompression: conservative estimate (assume worst case expansion)
            // Read num_runs from input to get exact size
            return {input_sizes[0] * 2};  // Conservative upper bound
        } else {
            // Compression: worst case is every element is unique
            // Format: sizeof(uint32_t) + n * (sizeof(T) + sizeof(uint32_t))
            size_t n = input_sizes[0] / sizeof(T);
            return {sizeof(uint32_t) + n * (sizeof(T) + sizeof(uint32_t))};
        }
    }
    
    std::unordered_map<std::string, size_t> getActualOutputSizesByName() const override {
        return {{"output", actual_output_sizes_.empty() ? 0 : actual_output_sizes_[0]}};
    }
    size_t getActualOutputSize(int index) const override {
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
        if (max_size < 1) return 0;
        DataType dt = getDataTypeEnum();
        std::memcpy(header_buffer, &dt, sizeof(DataType));
        return sizeof(DataType);
    }
    
    void deserializeHeader(const uint8_t* header_buffer, size_t size) override {
        (void)header_buffer; (void)size;
    }
    
    size_t getMaxHeaderSize(size_t output_index) const override {
        (void)output_index;
        return sizeof(DataType);
    }
    
private:
    std::vector<size_t> actual_output_sizes_;
    bool is_inverse_;  // True = decompression (expand runs), False = compression (encode runs)
    
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
