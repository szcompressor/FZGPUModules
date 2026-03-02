#pragma once

#include "stage/stage.h"
#include "fzm_format.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace fz {

/**
 * Difference coding stage
 * 
 * Forward (compression): Computes first-order differences:
 *   output[0] = input[0]
 *   output[i] = input[i] - input[i-1]
 * 
 * Inverse (decompression): Computes cumulative sum:
 *   output[0] = input[0]
 *   output[i] = input[i] + output[i-1]
 * 
 * This is a lossless transform that preserves data size.
 * Useful for data with smooth variations (reduces magnitude).
 * 
 * Template parameter T determines data type (float, double, int, etc.)
 * Default is float.
 */
template<typename T = float>
class DifferenceStage : public Stage {
public:
    DifferenceStage() : actual_output_size_(0), is_inverse_(false) {}
    
    /**
     * Set inverse mode for decompression
     * When true, performs cumulative sum instead of difference
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
    
    std::string getName() const override { return "Difference"; }
    size_t getNumInputs() const override { return 1; }
    size_t getNumOutputs() const override { return 1; }
    
    std::vector<size_t> estimateOutputSizes(
        const std::vector<size_t>& input_sizes
    ) const override {
        // Difference coding preserves size
        return {input_sizes[0]};
    }
    
    std::unordered_map<std::string, size_t> getActualOutputSizesByName() const override {
        return {{"output", actual_output_size_}};
    }
    
    uint16_t getStageTypeId() const override {
        return static_cast<uint16_t>(StageType::DIFFERENCE);
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
        // DataType is baked into the template; factory uses it to pick the right instantiation.
    }
    
    size_t getMaxHeaderSize(size_t output_index) const override {
        (void)output_index;
        return sizeof(DataType);
    }
    
private:
    size_t actual_output_size_;
    bool is_inverse_;  // True = decompression (cumulative sum), False = compression (difference)
    
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
extern template class DifferenceStage<float>;
extern template class DifferenceStage<double>;
extern template class DifferenceStage<int32_t>;
extern template class DifferenceStage<int64_t>;
extern template class DifferenceStage<uint16_t>;
extern template class DifferenceStage<uint8_t>;
extern template class DifferenceStage<uint32_t>;

} // namespace fz