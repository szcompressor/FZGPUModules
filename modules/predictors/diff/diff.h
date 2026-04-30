#pragma once

/**
 * @file diff.h
 * @brief First-order difference coding stage with optional negabinary fusion.
 */

#include "stage/stage.h"
#include "fzm_format.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace fz {

/**
 * Difference coding stage.
 *
 * Forward (compression): first-order differences with optional negabinary output
 *   output[0] = input[0]
 *   output[i] = input[i] - input[i-1]           (when TOut == T)
 *   output[i] = Negabinary<T>::encode(diff)      (when TOut != T)
 *
 * Inverse (decompression): cumulative sum with optional negabinary decode first
 *
 * Optional chunking (setChunkSize > 0):
 *   The difference/cumsum resets at each chunk boundary.  Each chunk is a
 *   fully independent context — the first element of each chunk is stored
 *   as-is (previous = 0 implied).  This enables parallel decompression and
 *   is required for the PFPL pipeline where 16 KB chunks flow independently
 *   through Bitshuffle and RZE.
 *
 * Template parameters:
 *   T    — input element type (signed/unsigned integer or float).
 *   TOut — output element type (defaults to T).
 *          When TOut != T:  T must be a signed integer and TOut its unsigned
 *          counterpart of the same width; negabinary encoding is fused at the
 *          final write of the forward kernel and the decode is the first step
 *          of the inverse kernel.
 *
 * Serialized header layout (6 bytes):
 *   [0]     DataType T     (1 byte)
 *   [1]     DataType TOut  (1 byte)
 *   [2..5]  chunk_size     (uint32_t, little-endian, 0 = no chunking)
 */
template<typename T = float, typename TOut = T>
class DifferenceStage : public Stage {
    static_assert(
        std::is_same_v<T, TOut> ||
        (std::is_integral_v<T> && std::is_signed_v<T> &&
         std::is_integral_v<TOut> && std::is_unsigned_v<TOut> &&
         sizeof(T) == sizeof(TOut)),
        "DifferenceStage: TOut must equal T, or T must be a signed integer "
        "and TOut its unsigned counterpart of the same width (negabinary fusion).");
public:
    DifferenceStage() : actual_output_size_(0), is_inverse_(false), chunk_size_(0) {}

    void setInverse(bool inverse) override { is_inverse_ = inverse; }
    bool isInverse() const override { return is_inverse_; }

    /**
     * Set chunk size in bytes (default 0 = no chunking).
     *
     * When > 0, differences and cumulative sums reset at each chunk boundary.
     * Must be a positive multiple of sizeof(T).  Pass 0 to disable chunking
     * and process the whole array as a single context (the previous default).
     */
    void setChunkSize(size_t bytes) { chunk_size_ = bytes; }
    size_t getChunkSize() const { return chunk_size_; }
    size_t getRequiredInputAlignment() const override {
        return chunk_size_ > 0 ? chunk_size_ : 1;
    }

    void execute(
        cudaStream_t stream,
        MemoryPool* pool,
        const std::vector<void*>& inputs,
        const std::vector<void*>& outputs,
        const std::vector<size_t>& sizes
    ) override;


    std::string getName() const override { return "Difference"; }
    size_t getNumInputs()  const override { return 1; }
    size_t getNumOutputs() const override { return 1; }

    std::vector<size_t> estimateOutputSizes(
        const std::vector<size_t>& input_sizes
    ) const override {
        return {input_sizes[0]};   // size-preserving (sizeof(T)==sizeof(TOut))
    }

    std::unordered_map<std::string, size_t> getActualOutputSizesByName() const override {
        return {{"output", actual_output_size_}};
    }
    size_t getActualOutputSize(int index) const override {
        return (index == 0) ? actual_output_size_ : 0;
    }

    uint16_t getStageTypeId() const override {
        return static_cast<uint16_t>(StageType::DIFFERENCE);
    }

    uint8_t getOutputDataType(size_t output_index) const override {
        (void)output_index;
        return static_cast<uint8_t>(getOutDataTypeEnum());
    }

    uint8_t getInputDataType(size_t /*input_index*/) const override {
        return static_cast<uint8_t>(getInDataTypeEnum());
    }

    size_t serializeHeader(size_t output_index, uint8_t* buf, size_t max_size) const override {
        (void)output_index;
        if (max_size < 6) return 0;
        buf[0] = static_cast<uint8_t>(getInDataTypeEnum());
        buf[1] = static_cast<uint8_t>(getOutDataTypeEnum());
        uint32_t cs = static_cast<uint32_t>(chunk_size_);
        std::memcpy(buf + 2, &cs, sizeof(uint32_t));
        return 6;
    }

    void deserializeHeader(const uint8_t* buf, size_t size) override {
        // DataTypes are baked into the template; factory picks the right instantiation.
        // Only chunk_size needs to be restored at runtime.
        if (size >= 6) {
            uint32_t cs = 0;
            std::memcpy(&cs, buf + 2, sizeof(uint32_t));
            chunk_size_ = cs;
        }
    }

    size_t getMaxHeaderSize(size_t output_index) const override {
        (void)output_index;
        return 6;
    }

    void saveState() override {
        saved_chunk_size_ = chunk_size_;
        saved_actual_output_size_ = actual_output_size_;
    }

    void restoreState() override {
        chunk_size_ = saved_chunk_size_;
        actual_output_size_ = saved_actual_output_size_;
    }

private:
    size_t actual_output_size_;
    size_t saved_actual_output_size_ = 0;
    bool   is_inverse_;
    size_t chunk_size_;
    size_t saved_chunk_size_ = 0;


    DataType getInDataTypeEnum() const {
        if (std::is_same_v<T, uint8_t>)  return DataType::UINT8;
        if (std::is_same_v<T, uint16_t>) return DataType::UINT16;
        if (std::is_same_v<T, uint32_t>) return DataType::UINT32;
        if (std::is_same_v<T, uint64_t>) return DataType::UINT64;
        if (std::is_same_v<T, int8_t>)   return DataType::INT8;
        if (std::is_same_v<T, int16_t>)  return DataType::INT16;
        if (std::is_same_v<T, int32_t>)  return DataType::INT32;
        if (std::is_same_v<T, int64_t>)  return DataType::INT64;
        if (std::is_same_v<T, float>)    return DataType::FLOAT32;
        if (std::is_same_v<T, double>)   return DataType::FLOAT64;
        return DataType::UINT8;
    }

    DataType getOutDataTypeEnum() const {
        if (std::is_same_v<TOut, uint8_t>)  return DataType::UINT8;
        if (std::is_same_v<TOut, uint16_t>) return DataType::UINT16;
        if (std::is_same_v<TOut, uint32_t>) return DataType::UINT32;
        if (std::is_same_v<TOut, uint64_t>) return DataType::UINT64;
        if (std::is_same_v<TOut, int8_t>)   return DataType::INT8;
        if (std::is_same_v<TOut, int16_t>)  return DataType::INT16;
        if (std::is_same_v<TOut, int32_t>)  return DataType::INT32;
        if (std::is_same_v<TOut, int64_t>)  return DataType::INT64;
        if (std::is_same_v<TOut, float>)    return DataType::FLOAT32;
        if (std::is_same_v<TOut, double>)   return DataType::FLOAT64;
        return DataType::UINT8;
    }
};

// ─── Same-type instantiations (original API, TOut = T) ───────────────────────
extern template class DifferenceStage<float>;
extern template class DifferenceStage<double>;
extern template class DifferenceStage<int32_t>;
extern template class DifferenceStage<int64_t>;
extern template class DifferenceStage<uint16_t>;
extern template class DifferenceStage<uint8_t>;
extern template class DifferenceStage<uint32_t>;

// ─── Negabinary-fused instantiations (TOut = unsigned counterpart of T) ───────
extern template class DifferenceStage<int8_t,  uint8_t>;
extern template class DifferenceStage<int16_t, uint16_t>;
extern template class DifferenceStage<int32_t, uint32_t>;
extern template class DifferenceStage<int64_t, uint64_t>;

} // namespace fz