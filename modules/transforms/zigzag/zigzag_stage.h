#pragma once

/**
 * @file zigzag_stage.h
 * @brief Element-wise zigzag encode/decode stage (`TIn[]` ↔ `TOut[]`).
 *
 * Size-preserving. `TIn` must be a signed integer; `TOut` its unsigned
 * counterpart of the same width.
 * Serialized header: 2 bytes — `[0]` TIn DataType, `[1]` TOut DataType.
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

/**
 * Element-wise zigzag encode/decode stage.
 * Forward: `TIn[] → TOut[]`; inverse: `TOut[] → TIn[]`.
 * @tparam TIn   Signed integer type.
 * @tparam TOut  Corresponding unsigned type (same width as `TIn`).
 */
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

    /**
     * Byte-transparent mode (default off).  When enabled, the stage opts out of
     * finalize() type checking by reporting DataType::UNKNOWN on both ports,
     * while still applying the same `sizeof(TIn)`-word zigzag transform.
     *
     * This is what lets ZigzagStage act as the LC framework's `TCMS` component
     * (Two's-Complement → Magnitude-Sign) inside a byte-stream chain: LC's
     * TCMS_N operates on a raw byte stream as N-byte words regardless of the
     * upstream element type, so the typed default (which would reject, e.g.,
     * a uint16 codes port feeding `Zigzag<int8_t>`, or a UINT8 byte stream
     * feeding `Zigzag<int64_t>`) must be relaxed for the chain to connect.
     */
    void setByteTransparent(bool on) { byte_transparent_ = on; }
    bool isByteTransparent() const   { return byte_transparent_; }

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
        return {input_sizes[0]};
    }

    std::unordered_map<std::string, size_t>
    getActualOutputSizesByName() const override {
        return {{"output", actual_output_size_}};
    }
    size_t getActualOutputSize(int index) const override {
        return (index == 0) ? actual_output_size_ : 0;
    }

    uint16_t getStageTypeId() const override {
        return static_cast<uint16_t>(StageType::ZIGZAG);
    }

    uint8_t getOutputDataType(size_t output_index) const override {
        (void)output_index;
        if (byte_transparent_) return static_cast<uint8_t>(DataType::UNKNOWN);
        // Forward output is TOut (unsigned); inverse output is TIn (signed).
        return is_inverse_
            ? static_cast<uint8_t>(dataTypeOf<TIn>())
            : static_cast<uint8_t>(dataTypeOf<TOut>());
    }

    uint8_t getInputDataType(size_t /*input_index*/) const override {
        if (byte_transparent_) return static_cast<uint8_t>(DataType::UNKNOWN);
        // Forward input is TIn (signed); inverse input is TOut (unsigned).
        return is_inverse_
            ? static_cast<uint8_t>(dataTypeOf<TOut>())
            : static_cast<uint8_t>(dataTypeOf<TIn>());
    }

    // ── Serialization ──────────────────────────────────────────────────────
    size_t serializeHeader(
        size_t output_index, uint8_t* buf, size_t max_size
    ) const override {
        (void)output_index;
        if (max_size < 3) return 0;
        buf[0] = static_cast<uint8_t>(dataTypeOf<TIn>());
        buf[1] = static_cast<uint8_t>(dataTypeOf<TOut>());
        buf[2] = byte_transparent_ ? 1 : 0;   // LC TCMS mode
        return 3;
    }

    void deserializeHeader(const uint8_t* buf, size_t size) override {
        // TIn/TOut are baked into the template (factory selects the right
        // instantiation before calling this); byte 2 carries byte-transparent.
        if (size >= 3) byte_transparent_ = (buf[2] != 0);
    }

    size_t getMaxHeaderSize(size_t) const override { return 3; }

private:
    bool   is_inverse_;
    bool   byte_transparent_ = false;  // LC TCMS mode: opt out of type checking
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

extern template class ZigzagStage<int8_t,  uint8_t>;
extern template class ZigzagStage<int16_t, uint16_t>;
extern template class ZigzagStage<int32_t, uint32_t>;
extern template class ZigzagStage<int64_t, uint64_t>;

} // namespace fz
