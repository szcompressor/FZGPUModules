#pragma once

/**
 * modules/transforms/negabinary/negabinary_stage.h
 *
 * NegabinaryStage<TIn, TOut> — element-wise negabinary encode/decode stage.
 *
 * Forward (compression):  applies Negabinary<TIn>::encode element-wise
 *                          TIn[]  →  TOut[]
 * Inverse (decompression): applies Negabinary<TIn>::decode element-wise
 *                          TOut[] →  TIn[]
 *
 * Template constraints (enforced by static_assert):
 *   - TIn  must be a signed integer type
 *   - TOut must be the corresponding unsigned type (same byte width)
 *
 * Common instantiations:
 *   NegabinaryStage<int16_t, uint16_t>
 *   NegabinaryStage<int32_t, uint32_t>
 *   NegabinaryStage<int64_t, uint64_t>
 *
 * The stage is size-preserving: output byte count equals input byte count.
 * Serialized config: 2 bytes (TIn DataType, TOut DataType).
 *
 * Note on use in pipelines:
 *   As a standalone stage, NegabinaryStage is suitable when the previous stage
 *   already produces signed difference output (e.g. DifferenceStage<T,T>).
 *   When fusing with DifferenceStage, prefer DifferenceStage<T, TOut> which
 *   encodes in a single kernel pass.
 */

#include "stage/stage.h"
#include "fzm_format.h"
#include "transforms/negabinary/negabinary.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <type_traits>

namespace fz {

template<typename TIn, typename TOut = typename std::make_unsigned<TIn>::type>
class NegabinaryStage : public Stage {
    static_assert(std::is_integral<TIn>::value && std::is_signed<TIn>::value,
                  "NegabinaryStage: TIn must be a signed integer type "
                  "(int8_t, int16_t, int32_t, or int64_t).");
    static_assert(std::is_integral<TOut>::value && std::is_unsigned<TOut>::value,
                  "NegabinaryStage: TOut must be an unsigned integer type.");
    static_assert(sizeof(TIn) == sizeof(TOut),
                  "NegabinaryStage: TIn and TOut must have the same byte width.");

public:
    NegabinaryStage() : is_inverse_(false), actual_output_size_(0) {}

    // ── Stage control ──────────────────────────────────────────────────────
    void setInverse(bool inv) override { is_inverse_ = inv; }
    bool isInverse() const override { return is_inverse_; }

    // ── Stage metadata ─────────────────────────────────────────────────────
    std::string getName() const override { return "Negabinary"; }
    size_t getNumInputs()  const override { return 1; }
    size_t getNumOutputs() const override { return 1; }

    std::vector<size_t> estimateOutputSizes(
        const std::vector<size_t>& input_sizes
    ) const override {
        return {input_sizes[0]};   // size-preserving
    }

    std::unordered_map<std::string, size_t> getActualOutputSizesByName() const override {
        return {{"output", actual_output_size_}};
    }

    uint16_t getStageTypeId() const override {
        return static_cast<uint16_t>(StageType::NEGABINARY);
    }

    uint8_t getOutputDataType(size_t) const override {
        return static_cast<uint8_t>(getTOutDataTypeEnum());
    }

    // ── Serialization ──────────────────────────────────────────────────────
    // Header: 2 bytes — [0] TIn DataType, [1] TOut DataType
    size_t serializeHeader(size_t, uint8_t* buf, size_t max_size) const override {
        if (max_size < 2) return 0;
        buf[0] = static_cast<uint8_t>(getTInDataTypeEnum());
        buf[1] = static_cast<uint8_t>(getTOutDataTypeEnum());
        return 2;
    }

    void deserializeHeader(const uint8_t*, size_t) override {
        // Types are baked into the template; factory picks the right instantiation.
    }

    size_t getMaxHeaderSize(size_t) const override { return 2; }

    // ── Execution ──────────────────────────────────────────────────────────
    void execute(
        cudaStream_t stream,
        MemoryPool*  pool,
        const std::vector<void*>& inputs,
        const std::vector<void*>& outputs,
        const std::vector<size_t>& sizes
    ) override;

private:
    bool   is_inverse_;
    size_t actual_output_size_;

    DataType getTInDataTypeEnum() const {
        if (std::is_same_v<TIn, int8_t>)  return DataType::INT8;
        if (std::is_same_v<TIn, int16_t>) return DataType::INT16;
        if (std::is_same_v<TIn, int32_t>) return DataType::INT32;
        if (std::is_same_v<TIn, int64_t>) return DataType::INT64;
        return DataType::INT32;
    }

    DataType getTOutDataTypeEnum() const {
        if (std::is_same_v<TOut, uint8_t>)  return DataType::UINT8;
        if (std::is_same_v<TOut, uint16_t>) return DataType::UINT16;
        if (std::is_same_v<TOut, uint32_t>) return DataType::UINT32;
        if (std::is_same_v<TOut, uint64_t>) return DataType::UINT64;
        return DataType::UINT32;
    }
};

// Explicit instantiation declarations
extern template class NegabinaryStage<int8_t,  uint8_t>;
extern template class NegabinaryStage<int16_t, uint16_t>;
extern template class NegabinaryStage<int32_t, uint32_t>;
extern template class NegabinaryStage<int64_t, uint64_t>;

} // namespace fz
