#pragma once

/**
 * @file bitpack_stage.h
 * @brief Bit-packing stage: packs N-bit integers into a dense byte stream.
 *
 * Supported input types: `uint8_t`, `uint16_t`, `uint32_t`.
 * Output is always `uint8_t[]` (byte-transparent to downstream stages).
 *
 * `nbits` must be a power of two and satisfy `1 <= nbits <= 8*sizeof(T)`.
 * Allowed values per type:
 *   uint8_t  : 1, 2, 4, 8
 *   uint16_t : 1, 2, 4, 8, 16
 *   uint32_t : 1, 2, 4, 8, 16, 32
 *
 * Serialized header layout (10 bytes):
 *   [0]    DataType of T          (1 byte)
 *   [1]    nbits                  (1 byte)
 *   [2..9] num_elements           (uint64_t, little-endian)
 *
 * `num_elements` is written during forward compression and used by the inverse
 * to know how many elements to unpack (byte count alone is ambiguous).
 */

#include "stage/stage.h"
#include "fzm_format.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace fz {

/**
 * Bit-packing stage.
 *
 * Forward: `T[] → uint8_t[]`  Pack each element using only the low `nbits` bits.
 * Inverse: `uint8_t[] → T[]`  Unpack elements, zero-extending to full width.
 *
 * @tparam T  Input element type: `uint8_t`, `uint16_t`, or `uint32_t`.
 */
template<typename T>
class BitpackStage : public Stage {
    static_assert(
        std::is_same_v<T, uint8_t> ||
        std::is_same_v<T, uint16_t> ||
        std::is_same_v<T, uint32_t>,
        "BitpackStage: T must be uint8_t, uint16_t, or uint32_t.");

public:
    BitpackStage() = default;

    // ── Stage control ──────────────────────────────────────────────────────────
    void setInverse(bool inv) override { is_inverse_ = inv; }
    bool isInverse() const override    { return is_inverse_; }

    // ── Configuration ──────────────────────────────────────────────────────────

    /**
     * Set the number of bits per element.
     *
     * Must be a power of two between 1 and 8*sizeof(T) inclusive.
     * Allowed values:
     *   uint8_t  : 1, 2, 4, 8
     *   uint16_t : 1, 2, 4, 8, 16
     *   uint32_t : 1, 2, 4, 8, 16, 32
     */
    void setNBits(uint8_t nbits) {
        if (nbits == 0 || nbits > 8 * sizeof(T) || (nbits & (nbits - 1)) != 0)
            throw std::invalid_argument(
                "BitpackStage::setNBits: nbits must be a power of two "
                "in [1, " + std::to_string(8 * sizeof(T)) + "], got "
                + std::to_string(nbits));
        nbits_ = nbits;
    }
    uint8_t getNBits() const { return nbits_; }

    // ── Execution ──────────────────────────────────────────────────────────────
    void execute(
        cudaStream_t stream,
        MemoryPool* pool,
        const std::vector<void*>& inputs,
        const std::vector<void*>& outputs,
        const std::vector<size_t>& sizes
    ) override;

    // ── Metadata ───────────────────────────────────────────────────────────────
    std::string getName()      const override { return "Bitpack"; }
    size_t getNumInputs()      const override { return 1; }
    size_t getNumOutputs()     const override { return 1; }

    std::vector<size_t> estimateOutputSizes(
        const std::vector<size_t>& input_sizes
    ) const override {
        if (input_sizes.empty()) return {0};
        if (!is_inverse_) {
            // Forward: packed output is ceil(n * nbits / 8) bytes.
            const size_t n = input_sizes[0] / sizeof(T);
            return {(n * nbits_ + 7) / 8};
        } else {
            // Inverse: worst case — every packed bit expands to a full element.
            // input_sizes[0] is the packed byte count; max elements = bytes * (8/nbits).
            const size_t max_elems = (input_sizes[0] * 8 + nbits_ - 1) / nbits_;
            return {max_elems * sizeof(T)};
        }
    }

    std::unordered_map<std::string, size_t>
    getActualOutputSizesByName() const override {
        return {{"output", actual_output_size_}};
    }

    size_t getActualOutputSize(int index) const override {
        return (index == 0) ? actual_output_size_ : 0;
    }

    // ── Type system ────────────────────────────────────────────────────────────

    uint16_t getStageTypeId() const override {
        return static_cast<uint16_t>(StageType::BITPACK);
    }

    // Packed byte stream has no meaningful element type; opt out of type checking.
    uint8_t getOutputDataType(size_t /*output_index*/) const override {
        return static_cast<uint8_t>(DataType::UNKNOWN);
    }
    uint8_t getInputDataType(size_t /*input_index*/) const override {
        return static_cast<uint8_t>(DataType::UNKNOWN);
    }

    // ── Serialization ──────────────────────────────────────────────────────────

    size_t serializeHeader(
        size_t /*output_index*/, uint8_t* buf, size_t max_size
    ) const override {
        if (max_size < 10) return 0;
        buf[0] = static_cast<uint8_t>(dataTypeOf<T>());
        buf[1] = nbits_;
        std::memcpy(buf + 2, &num_elements_, sizeof(uint64_t));
        return 10;
    }

    void deserializeHeader(const uint8_t* buf, size_t size) override {
        // buf[0] (DataType) is used by the factory to pick the right instantiation.
        // We only need nbits and num_elements here.
        if (size >= 2)  nbits_ = buf[1];
        if (size >= 10) std::memcpy(&num_elements_, buf + 2, sizeof(uint64_t));
    }

    size_t getMaxHeaderSize(size_t /*output_index*/) const override { return 10; }

    // saveState/restoreState: deserializeHeader (called during decompression
    // setup) overwrites num_elements with the value from the file header.
    // Save the forward-pass values so they can be restored afterward.
    void saveState() override {
        saved_nbits_        = nbits_;
        saved_num_elements_ = num_elements_;
        saved_output_size_  = actual_output_size_;
    }

    void restoreState() override {
        nbits_               = saved_nbits_;
        num_elements_        = saved_num_elements_;
        actual_output_size_  = saved_output_size_;
    }

    bool isGraphCompatible() const override { return true; }

private:
    bool     is_inverse_        = false;
    uint8_t  nbits_             = 8 * sizeof(T);   // default: keep all bits (identity)
    uint64_t num_elements_      = 0;               // set by forward execute; used by inverse
    size_t   actual_output_size_ = 0;

    // saveState snapshots
    uint8_t  saved_nbits_        = 8 * sizeof(T);
    uint64_t saved_num_elements_ = 0;
    size_t   saved_output_size_  = 0;

    template<typename U>
    static constexpr DataType dataTypeOf() {
        if (std::is_same_v<U,  uint8_t>) return DataType::UINT8;
        if (std::is_same_v<U, uint16_t>) return DataType::UINT16;
        if (std::is_same_v<U, uint32_t>) return DataType::UINT32;
        return DataType::UINT8; // unreachable
    }
};

extern template class BitpackStage<uint8_t>;
extern template class BitpackStage<uint16_t>;
extern template class BitpackStage<uint32_t>;

} // namespace fz
