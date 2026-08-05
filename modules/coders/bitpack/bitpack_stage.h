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
 * ## Shift (frame-of-reference base + low-bit right shift)
 *
 * Each element is transformed before packing and inverted on unpack:
 *
 *     forward:  packed = (v - base) >> shift        (low `nbits` bits kept)
 *     inverse:  v      = (packed << shift) + base
 *
 * The two knobs attack opposite ends of the word and compose:
 *   - `base` removes a constant offset, i.e. dead **high** bits (values clustered
 *     far from zero — a "frame of reference" transform). Always lossless.
 *   - `shift` removes dead **low** bits. Lossless only when every value has
 *     `shift` trailing zeros after the base subtraction; otherwise it is a lossy
 *     truncation (`v` is restored to a multiple of `1 << shift` plus base).
 *
 * Both default to 0, which is the previous behaviour exactly.
 *
 * Serialized header layout (15 bytes):
 *   [0]      DataType of T        (1 byte)
 *   [1]      nbits                (1 byte)
 *   [2..9]   num_elements         (uint64_t, little-endian)
 *   [10]     shift                (1 byte)
 *   [11..14] base                 (uint32_t, little-endian; zero-extended T)
 *
 * `num_elements` is written during forward compression and used by the inverse
 * to know how many elements to unpack (byte count alone is ambiguous).
 * Headers shorter than 15 bytes (pre-shift archives) decode with shift = base = 0.
 */

#include "stage/stage.h"
#include "fzm_format.h"
#include "backend/types.h"
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
 * Forward: `T[] → uint8_t[]`  Pack `(v - base) >> shift` using only the low `nbits` bits.
 * Inverse: `uint8_t[] → T[]`  Unpack, then restore `(packed << shift) + base`.
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
     *
     * Ignored during forward execute when setAutoDetect(true) is active.
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

    /**
     * Frame-of-reference base: subtracted from every element before packing
     * and added back on unpack.  Removes dead high bits when the values are
     * clustered away from zero.  Always lossless.
     *
     * Ignored during forward execute when setAutoBase(true) is active.
     */
    void setBase(T base) { base_ = base; }
    T getBase() const { return base_; }

    /**
     * Right-shift applied after the base subtraction: `(v - base) >> shift`.
     * Removes dead low bits.  Must be in `[0, 8*sizeof(T) - 1]`.
     *
     * @warning **Lossy unless every `(v - base)` has `shift` trailing zeros.**
     *          The inverse restores `(packed << shift) + base`, so any dropped
     *          low bits are gone.  Use setAutoShift(true) to pick the largest
     *          shift that is provably lossless for the data at hand.
     *
     * Ignored during forward execute when setAutoShift(true) is active.
     */
    void setShift(uint8_t shift) {
        if (shift >= 8 * sizeof(T))
            throw std::invalid_argument(
                "BitpackStage::setShift: shift must be in [0, "
                + std::to_string(8 * sizeof(T) - 1) + "], got "
                + std::to_string(shift));
        shift_ = shift;
    }
    uint8_t getShift() const { return shift_; }

    /**
     * Enable automatic bit-width detection.
     *
     * When true, forward execute scans the input for its maximum value and
     * selects the smallest valid power-of-two nbits that covers the shifted,
     * base-subtracted range.  The chosen nbits is stored in the serialized
     * header so the inverse pass can unpack correctly.
     *
     * After compress(), getNBits() reflects the detected value.
     *
     * Incompatible with CUDA Graph capture: isGraphCompatible() returns false
     * while any auto-detect mode is enabled.
     */
    void setAutoDetect(bool enable) { auto_detect_ = enable; }
    bool isAutoDetect() const { return auto_detect_; }

    /**
     * Enable automatic frame-of-reference base selection: forward execute
     * min-reduces the input and uses that minimum as `base`.  Lossless.
     * After compress(), getBase() reflects the detected value.
     */
    void setAutoBase(bool enable) { auto_base_ = enable; }
    bool isAutoBase() const { return auto_base_; }

    /**
     * Enable automatic shift selection: forward execute OR-reduces every
     * `(v - base)` and uses the trailing-zero count of that OR as `shift` —
     * i.e. the largest shift that drops no information.  **Always lossless**,
     * unlike a hand-set setShift().  After compress(), getShift() reflects the
     * detected value.
     */
    void setAutoShift(bool enable) { auto_shift_ = enable; }
    bool isAutoShift() const { return auto_shift_; }

    /**
     * Convenience: enable auto base, auto shift, and auto nbits together —
     * the fully adaptive lossless mode.  Each element becomes
     * `(v - min) >> ctz(OR of (v - min))`, packed at the tightest power-of-two
     * width that fits.
     */
    void setAdaptive(bool enable) {
        auto_base_ = auto_shift_ = auto_detect_ = enable;
    }

    // ── Execution ──────────────────────────────────────────────────────────────
    void execute(
        fz::stream_t stream,
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
            if (auto_detect_) {
                // nbits is unknown until execute() scans the data; return worst
                // case (full-width, no compression) so PREALLOCATE has enough room.
                return {input_sizes[0]};
            }
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
        if (max_size < 15) return 0;
        buf[0] = static_cast<uint8_t>(dataTypeOf<T>());
        buf[1] = nbits_;
        std::memcpy(buf + 2, &num_elements_, sizeof(uint64_t));
        buf[10] = shift_;
        const uint32_t base32 = static_cast<uint32_t>(base_);
        std::memcpy(buf + 11, &base32, sizeof(uint32_t));
        return 15;
    }

    void deserializeHeader(const uint8_t* buf, size_t size) override {
        // buf[0] (DataType) is used by the factory to pick the right instantiation.
        // We only need nbits, num_elements, shift, and base here.
        if (size >= 2)  nbits_ = buf[1];
        if (size >= 10) std::memcpy(&num_elements_, buf + 2, sizeof(uint64_t));
        // Pre-shift archives stop at 10 bytes; leave shift/base at their defaults.
        if (size >= 15) {
            shift_ = buf[10];
            uint32_t base32 = 0;
            std::memcpy(&base32, buf + 11, sizeof(uint32_t));
            base_ = static_cast<T>(base32);
        }
    }

    size_t getMaxHeaderSize(size_t /*output_index*/) const override { return 15; }

    // saveState/restoreState: deserializeHeader (called during decompression
    // setup) overwrites num_elements with the value from the file header.
    // Save the forward-pass values so they can be restored afterward.
    void saveState() override {
        saved_nbits_        = nbits_;
        saved_num_elements_ = num_elements_;
        saved_output_size_  = actual_output_size_;
        saved_shift_        = shift_;
        saved_base_         = base_;
    }

    void restoreState() override {
        nbits_               = saved_nbits_;
        num_elements_        = saved_num_elements_;
        actual_output_size_  = saved_output_size_;
        shift_               = saved_shift_;
        base_                = saved_base_;
    }

    // Auto-detect requires a D2H sync to read the scanned min/max/OR, so it
    // cannot be recorded inside a CUDA Graph.
    bool isGraphCompatible() const override {
        return !(auto_detect_ || auto_base_ || auto_shift_);
    }

private:
    bool     is_inverse_        = false;
    bool     auto_detect_       = false;
    bool     auto_base_         = false;
    bool     auto_shift_        = false;
    uint8_t  nbits_             = 8 * sizeof(T);   // default: keep all bits (identity)
    uint8_t  shift_             = 0;               // low bits dropped before packing
    T        base_              = T(0);            // frame-of-reference offset
    uint64_t num_elements_      = 0;               // set by forward execute; used by inverse
    size_t   actual_output_size_ = 0;

    // saveState snapshots
    uint8_t  saved_nbits_        = 8 * sizeof(T);
    uint8_t  saved_shift_        = 0;
    T        saved_base_         = T(0);
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
