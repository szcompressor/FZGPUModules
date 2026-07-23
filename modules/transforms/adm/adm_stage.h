#pragma once

/**
 * @file adm_stage.h
 * @brief Adaptive Data Mapping (ADM) stage — remaps u16/u32 integer streams into
 *        a compact 8-bit symbol domain before entropy coding.
 *
 * Forward: `uint16_t[]` or `uint32_t[]` → opaque ADM payload (`uint8_t[]`).
 * Inverse: ADM payload → original integer array.
 *
 * ADM partitions the input into 512-element warp blocks, computes a per-warp
 * center (mean), and encodes each element as a compact (code, unary-signal) pair.
 * The resulting byte codes are compressible by ANSStage with a much better ratio
 * than the raw integer stream.
 *
 * D2H copies occur in both directions, so isGraphCompatible() returns false.
 *
 * Serialized FZM header layout (12 bytes):
 *   [0]     dtype         (0=U16, 1=U32)
 *   [1..3]  reserved      (zero)
 *   [4..11] num_elements  (uint64_t LE; element count of the uncompressed input)
 *
 * @note ADM algorithm adapted from the MANS project (Huang et al., BSD-3-Clause).
 *       See THIRD_PARTY.md for the full license text and attribution.
 */

#include "stage/stage.h"
#include "fzm_format.h"
#include "backend/types.h"
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace fz {

/** Input element type for ADMStage. */
enum class ADMDtype : uint8_t { U16 = 0, U32 = 1 };

class ADMStage : public Stage {
public:
    ADMStage()  = default;
    ~ADMStage() override = default;

    // ── Configuration ─────────────────────────────────────────────────────────

    /**
     * Set the input element type (default U16).
     * Must match the upstream stage's output type.
     * Must be called before finalize().
     */
    void     setDtype(ADMDtype dt) { dtype_ = dt; }
    ADMDtype getDtype()    const   { return dtype_; }

    // ── Stage control ─────────────────────────────────────────────────────────
    void setInverse(bool inv) override { is_inverse_ = inv; }
    bool isInverse()    const override { return is_inverse_; }

    // D2H copies occur in compress (payload-size readback) and decompress (header peek).
    bool isGraphCompatible() const override { return false; }

    // ── Pool lifecycle ────────────────────────────────────────────────────────
    void   onFinalize(size_t estimated_inlen, MemoryPool* pool) override;
    size_t estimateDeviceFootprintBytes(size_t inlen) const override;
    size_t estimateScratchBytes(const std::vector<size_t>& input_sizes) const override;

    // ── Execution ─────────────────────────────────────────────────────────────
    void execute(
        fz::stream_t stream,
        MemoryPool*  pool,
        const std::vector<void*>&  inputs,
        const std::vector<void*>&  outputs,
        const std::vector<size_t>& sizes
    ) override;

    // ── Metadata ──────────────────────────────────────────────────────────────
    std::string getName()       const override { return "ADM"; }
    size_t      getNumInputs()  const override { return 1; }
    size_t      getNumOutputs() const override { return 1; }

    std::vector<size_t> estimateOutputSizes(
        const std::vector<size_t>& input_sizes
    ) const override;

    std::unordered_map<std::string, size_t>
    getActualOutputSizesByName() const override {
        return {{"output", actual_output_size_}};
    }

    size_t getActualOutputSize(int index) const override {
        return (index == 0) ? actual_output_size_ : 0;
    }

    // ── Type system ───────────────────────────────────────────────────────────
    uint16_t getStageTypeId() const override {
        return static_cast<uint16_t>(StageType::ADM);
    }

    // Input type: U16 or U32 (used by pipeline finalize() for type checking).
    uint8_t getInputDataType(size_t /*idx*/) const override {
        return dtype_ == ADMDtype::U16
            ? static_cast<uint8_t>(DataType::UINT16)
            : static_cast<uint8_t>(DataType::UINT32);
    }

    // Output is an opaque ADM payload — opt out of downstream type checking.
    uint8_t getOutputDataType(size_t /*idx*/) const override {
        return static_cast<uint8_t>(DataType::UNKNOWN);
    }

    // ── Serialization ─────────────────────────────────────────────────────────
    size_t serializeHeader(
        size_t /*output_index*/, uint8_t* buf, size_t max_size
    ) const override {
        if (max_size < 12) return 0;
        buf[0] = static_cast<uint8_t>(dtype_);
        buf[1] = buf[2] = buf[3] = 0;
        std::memcpy(buf + 4, &num_elements_, sizeof(uint64_t));
        return 12;
    }

    void deserializeHeader(const uint8_t* buf, size_t size) override {
        if (size >= 1) dtype_        = static_cast<ADMDtype>(buf[0]);
        if (size >= 12) std::memcpy(&num_elements_, buf + 4, sizeof(uint64_t));
    }

    size_t getMaxHeaderSize(size_t /*output_index*/) const override { return 12; }

    void saveState() override {
        saved_dtype_        = dtype_;
        saved_num_elements_ = num_elements_;
        saved_output_size_  = actual_output_size_;
    }

    void restoreState() override {
        dtype_               = saved_dtype_;
        num_elements_        = saved_num_elements_;
        actual_output_size_  = saved_output_size_;
    }

private:
    ADMDtype dtype_          = ADMDtype::U16;
    bool     is_inverse_     = false;
    uint64_t num_elements_   = 0;   // set by forward execute; restored by deserializeHeader
    size_t   actual_output_size_ = 0;

    // Capacity (elements) of current scratch allocation. Grow-only.
    size_t cap_elements_ = 0;

    // Persistent scratch device pointers.
    int*      d_signal_length_  = nullptr;  // gsize × sizeof(int)
    int*      d_output_lengths_ = nullptr;  // (gsize+1) × sizeof(int)
    void*     d_centers_        = nullptr;  // gsize × sizeof(dtype)
    uint32_t* d_block_flags_    = nullptr;  // flags_words × sizeof(uint32_t)
    uint8_t*  d_codes_          = nullptr;  // num_elements × 1
    uint8_t*  d_concat_signals_ = nullptr;  // num_elements × kMaxSignalBytes
    uint8_t*  d_bit_signals_    = nullptr;  // num_elements × kMaxSignalBytes (thrust path)
    int*      d_loc_offset_     = nullptr;  // (gsize+1) × sizeof(int)
    int*      d_prefix_state_   = nullptr;  // (gsize+1) × sizeof(int)
    int*      d_block_resolved_ = nullptr;  // (num_blocks+1) × sizeof(int)
    unsigned int* d_overflow_flag_ = nullptr; // 1 word; checked after kernels in debug builds

    // saveState / restoreState snapshots.
    ADMDtype saved_dtype_          = ADMDtype::U16;
    uint64_t saved_num_elements_   = 0;
    size_t   saved_output_size_    = 0;

    // Allocates / reallocates all 9 scratch buffers from pool.
    void initScratch(size_t num_elements, MemoryPool* pool);

    // Returns the number of signal bytes per element for the current dtype.
    int maxSignalBytes() const;
    // Returns sizeof of the center element for the current dtype.
    size_t centerElemBytes() const;
};

} // namespace fz
