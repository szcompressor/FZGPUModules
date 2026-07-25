#pragma once

/**
 * @file tupl_stage.h
 * @brief GPU tuple deinterleave stage (AoS -> SoA transpose over fixed-size blocks).
 *
 * Given a block of `tuples` structs, each `dim` fields wide with fields of
 * `word_size` bytes (word_size in {1,2,4,8}), the forward pass regroups the
 * data field-major (SoA): all field 0 words, then all field 1 words, etc.
 * The inverse pass recombines SoA back into the original AoS layout. Output
 * is the same byte size as input (pure permutation, no compression).
 *
 * Any leftover bytes at the tail of a block that don't form a complete tuple
 * (block_size not evenly divisible by dim * word_size) are copied verbatim,
 * unchanged by either direction.
 *
 * Serialized header (6 bytes):
 *   `[0..3]` block_size (uint32_t LE), `[4]` word_size (uint8_t), `[5]` dim (uint8_t).
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

/**
 * GPU tuple deinterleave (AoS <-> SoA) stage.
 *
 * `setBlockSize(bytes)` — block size in bytes (default 16384; must be a
 * positive multiple of `word_size`).
 * `setWordSize(bytes)` — field width: 1, 2, 4, or 8 (default 1).
 * `setDim(n)` — number of fields per tuple, i.e. LC's `TUPLk` (default 2).
 *
 * @note **Prior work:** ported from `d_TUPL` / `d_iTUPL` in the LC framework
 *       (Burtscher et al., BSD-3-Clause). Upstream generates one fixed
 *       `(dim, word_size)` instantiation per component (`TUPL2_1`, `TUPL6_8`,
 *       ...) over a hardcoded 16 KB chunk; here `dim`/`word_size`/`block_size`
 *       are all independent runtime parameters. See `THIRD_PARTY.md`.
 */
class TUPLStage : public Stage {
public:
    TUPLStage()
        : is_inverse_(false)
        , block_size_(16384)
        , word_size_(1)
        , dim_(2)
        , actual_output_size_(0)
    {}

    // ── Stage control ──────────────────────────────────────────────────────
    void setInverse(bool inv) override { is_inverse_ = inv; }
    bool isInverse() const override    { return is_inverse_; }

    void setBlockSize(size_t bytes) { block_size_ = static_cast<uint32_t>(bytes); }
    void setWordSize(size_t bytes)  { word_size_  = static_cast<uint8_t>(bytes);  }
    void setDim(size_t dim)         { dim_        = static_cast<uint8_t>(dim);    }

    size_t getBlockSize() const { return block_size_; }
    size_t getRequiredInputAlignment() const override { return block_size_; }
    size_t getWordSize() const { return word_size_; }
    size_t getDim() const { return dim_; }

    // ── Execution ──────────────────────────────────────────────────────────
    void execute(
        fz::stream_t stream,
        MemoryPool* pool,
        const std::vector<void*>& inputs,
        const std::vector<void*>& outputs,
        const std::vector<size_t>& sizes
    ) override;

    // ── Metadata ───────────────────────────────────────────────────────────
    std::string getName() const override { return "TUPL"; }
    size_t getNumInputs()  const override { return 1; }
    size_t getNumOutputs() const override { return 1; }

    std::vector<size_t> estimateOutputSizes(
        const std::vector<size_t>& input_sizes
    ) const override {
        // Size-preserving transform.
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
        return static_cast<uint16_t>(StageType::TUPL);
    }

    uint8_t getOutputDataType(size_t) const override {
        // Raw byte stream — report as UINT8.
        return static_cast<uint8_t>(DataType::UINT8);
    }

    // ── Serialization ──────────────────────────────────────────────────────
    // Header: [0..3] block_size (uint32_t LE), [4] word_size (uint8_t), [5] dim (uint8_t)
    size_t serializeHeader(
        size_t output_index, uint8_t* buf, size_t max_size
    ) const override {
        (void)output_index;
        if (max_size < 6) return 0;
        std::memcpy(buf, &block_size_, sizeof(uint32_t));
        buf[4] = word_size_;
        buf[5] = dim_;
        return 6;
    }

    void deserializeHeader(const uint8_t* buf, size_t size) override {
        if (size >= 4) std::memcpy(&block_size_, buf, sizeof(uint32_t));
        if (size >= 5) word_size_ = buf[4];
        if (size >= 6) dim_ = buf[5];
    }

    size_t getMaxHeaderSize(size_t) const override { return 6; }

    void saveState() override {
        saved_block_size_ = block_size_;
        saved_word_size_ = word_size_;
        saved_dim_ = dim_;
        saved_actual_output_size_ = actual_output_size_;
    }

    void restoreState() override {
        block_size_ = saved_block_size_;
        word_size_ = saved_word_size_;
        dim_ = saved_dim_;
        actual_output_size_ = saved_actual_output_size_;
    }

private:
    bool     is_inverse_;
    uint32_t block_size_;              ///< Bytes per block.
    uint32_t saved_block_size_ = 0;
    uint8_t  word_size_;               ///< Bytes per tuple field (1, 2, 4, or 8).
    uint8_t  saved_word_size_ = 0;
    uint8_t  dim_;                     ///< Fields per tuple (LC's TUPLk, k = dim).
    uint8_t  saved_dim_ = 0;
    size_t   actual_output_size_ = 0;
    size_t   saved_actual_output_size_ = 0;

    // Validate config. Unlike LC (fixed 16 KB chunk shared across all
    // (dim, word_size) combos, so a chunk frequently doesn't divide evenly
    // into whole tuples), block_size here is caller-chosen -- we only require
    // it to be a whole number of words so per-block byte offsets stay
    // word-aligned; leftover bytes that don't form a whole tuple within a
    // block are still handled generically (see tupl_stage.cu), not banned.
    void validateConfig() const {
        if (word_size_ != 1 && word_size_ != 2 && word_size_ != 4 && word_size_ != 8)
            throw std::invalid_argument(
                "TUPLStage: word_size must be 1, 2, 4, or 8");
        if (dim_ < 2)
            throw std::invalid_argument("TUPLStage: dim must be >= 2");
        if (block_size_ == 0 || block_size_ % word_size_ != 0)
            throw std::invalid_argument(
                "TUPLStage: block_size must be a positive multiple of word_size");
    }
};

} // namespace fz
