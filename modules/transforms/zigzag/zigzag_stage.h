#pragma once

/**
 * modules/transforms/zigzag/zigzag_stage.h
 *
 * ZigzagStage<TIn, TOut> — element-wise zigzag encode/decode stage.
 *
 * Forward (compression):  applies Zigzag<TIn>::encode element-wise
 *                          TIn[]  →  TOut[]
 * Inverse (decompression): applies Zigzag<TIn>::decode element-wise
 *                          TOut[] →  TIn[]
 *
 * Template constraints (enforced by static_assert):
 *   - TIn must be a signed integer type
 *   - TOut must be the corresponding unsigned type (same byte width)
 *
 * Common instantiations:
 *   ZigzagStage<int16_t, uint16_t>
 *   ZigzagStage<int32_t, uint32_t>
 *   ZigzagStage<int64_t, uint64_t>
 *
 * The stage is size-preserving: output byte count equals input byte count.
 * Serialized config: 2 bytes (TIn DataType, TOut DataType).
 */

#include "stage/stage.h"
#include "fzm_format.h"
#include "transforms/zigzag/zigzag.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <type_traits>

namespace fz {

template<typename TIn, typename TOut = typename std::make_unsigned<TIn>::type>
class ZigzagStage : public Stage {
    static_assert(std::is_integral<TIn>::value && std::is_signed<TIn>::value,
                  "ZigzagStage: TIn must be a signed integer type "
                  "(int8_t, int16_t, int32_t, or int64_t).");
    static_assert(std::is_integral<TOut>::value && std::is_unsigned<TOut>::value,
                  "ZigzagStage: TOut must be an unsigned integer type.");
    static_assert(sizeof(TIn) == sizeof(TOut),
                  "ZigzagStage: TIn and TOut must have the same byte width.");

public:
    ZigzagStage() : is_inverse_(false), actual_output_size_(0) {}

    // ── Stage control ──────────────────────────────────────────────────────
    void setInverse(bool inv) override { is_inverse_ = inv; }
    bool isInverse() const override    { return is_inverse_; }

    // ── Execution ──────────────────────────────────────────────────────────
    void execute(
        cudaStream_t stream,
        MemoryPool* pool,
        const std::vector<void*>& inputs,
        const std::vector<void*>& outputs,
        const std::vector<size_t>& sizes
    ) override;

    // ── Metadata ───────────────────────────────────────────────────────────
    std::string getName() const override { return "Zigzag"; }
    size_t getNumInputs()  const override { return 1; }
    size_t getNumOutputs() const override { return 1; }

    std::vector<size_t> estimateOutputSizes(
        const std::vector<size_t>& input_sizes
    ) const override {
        // Size-preserving transform
        return {input_sizes[0]};
    }

    std::unordered_map<std::string, size_t>
    getActualOutputSizesByName() const override {
        return {{"output", actual_output_size_}};
    }

    uint16_t getStageTypeId() const override {
        return static_cast<uint16_t>(StageType::ZIGZAG);
    }

    uint8_t getOutputDataType(size_t output_index) const override {
        (void)output_index;
        // Forward output is TOut (unsigned); inverse output is TIn (signed).
        return is_inverse_
            ? static_cast<uint8_t>(dataTypeOf<TIn>())
            : static_cast<uint8_t>(dataTypeOf<TOut>());
    }

    // ── Serialization ──────────────────────────────────────────────────────
    size_t serializeHeader(
        size_t output_index, uint8_t* buf, size_t max_size
    ) const override {
        (void)output_index;
        if (max_size < 2) return 0;
        buf[0] = static_cast<uint8_t>(dataTypeOf<TIn>());
        buf[1] = static_cast<uint8_t>(dataTypeOf<TOut>());
        return 2;
    }

    void deserializeHeader(const uint8_t* buf, size_t size) override {
        (void)buf; (void)size;
        // TIn/TOut are baked into the template.
        // The factory selects the right instantiation before calling this.
    }

    size_t getMaxHeaderSize(size_t) const override { return 2; }

private:
    bool   is_inverse_;
    size_t actual_output_size_;

    template<typename U>
    static constexpr DataType dataTypeOf() {
        if (std::is_same<U,  int8_t>::value)  return DataType::INT8;
        if (std::is_same<U, int16_t>::value)  return DataType::INT16;
        if (std::is_same<U, int32_t>::value)  return DataType::INT32;
        if (std::is_same<U, int64_t>::value)  return DataType::INT64;
        if (std::is_same<U,  uint8_t>::value) return DataType::UINT8;
        if (std::is_same<U, uint16_t>::value) return DataType::UINT16;
        if (std::is_same<U, uint32_t>::value) return DataType::UINT32;
        if (std::is_same<U, uint64_t>::value) return DataType::UINT64;
        return DataType::UINT8; // unreachable
    }
};

// Explicit instantiation declarations (definitions are in zigzag_stage.cu)
extern template class ZigzagStage<int8_t,  uint8_t>;
extern template class ZigzagStage<int16_t, uint16_t>;
extern template class ZigzagStage<int32_t, uint32_t>;
extern template class ZigzagStage<int64_t, uint64_t>;

} // namespace fz
