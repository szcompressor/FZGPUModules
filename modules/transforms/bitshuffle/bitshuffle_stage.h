#pragma once

/**
 * modules/transforms/bitshuffle/bitshuffle_stage.h
 *
 * BitshuffleStage — GPU bit-matrix transpose stage.
 *
 * Given a chunk of N elements each W bits wide, the forward pass produces W
 * groups each containing the k-th bit of all N elements (a W × N bit-matrix
 * transpose).  The output is the same byte size as the input.
 *
 * Processing model:
 *   - Data is processed in fixed-size chunks (default 16 KB).
 *   - Within each chunk, a CUDA block executes the transpose using
 *     __ballot_sync to gather column bits across a warp of 32 threads.
 *   - Output layout: bit-plane W-1 of all elements, then W-2, ..., then 0.
 *
 * Configuration:
 *   setBlockSize(bytes)   — chunk size in bytes (default 16384, must be a
 *                           multiple of 1024 * element_width)
 *   setElementWidth(bytes)— element width: 1, 2, 4, or 8 (default 4)
 *   setInverse(bool)      — transpose in the reverse direction
 *
 * Output layout: MSB-first — bit-plane W-1 (MSBit) is at plane index 0;
 *   bit-plane 0 (LSBit) is at index W-1.
 *   Plane p occupies words p*(N_chunk/32) .. (p+1)*(N_chunk/32)-1
 *   where N_chunk = block_size_bytes / element_width.
 *
 * Serialized config: 5 bytes
 *   [0..3] block_size   (uint32_t, little-endian)
 *   [4]    element_width (uint8_t)
 */

#include "stage/stage.h"
#include "fzm_format.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace fz {

class BitshuffleStage : public Stage {
public:
    BitshuffleStage()
        : is_inverse_(false)
        , block_size_(16384)
        , element_width_(4)
        , actual_output_size_(0)
    {}

    // ── Stage control ──────────────────────────────────────────────────────
    void setInverse(bool inv) override { is_inverse_ = inv; }
    bool isInverse() const override    { return is_inverse_; }

    void setBlockSize(size_t bytes)   { block_size_    = static_cast<uint32_t>(bytes); }
    void setElementWidth(size_t bytes){ element_width_ = static_cast<uint8_t>(bytes);  }

    size_t getBlockSize()    const { return block_size_;    }
    size_t getRequiredInputAlignment() const override { return block_size_; }
    size_t getElementWidth() const { return element_width_; }

    // ── Execution ──────────────────────────────────────────────────────────
    void execute(
        cudaStream_t stream,
        MemoryPool* pool,
        const std::vector<void*>& inputs,
        const std::vector<void*>& outputs,
        const std::vector<size_t>& sizes
    ) override;

    // ── Metadata ───────────────────────────────────────────────────────────
    std::string getName() const override { return "Bitshuffle"; }
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

    uint16_t getStageTypeId() const override {
        return static_cast<uint16_t>(StageType::BITSHUFFLE);
    }

    uint8_t getOutputDataType(size_t) const override {
        // Raw byte stream — report as UINT8.
        return static_cast<uint8_t>(DataType::UINT8);
    }

    // ── Serialization ──────────────────────────────────────────────────────
    // Header: [0..3] block_size (uint32_t LE), [4] element_width (uint8_t)
    size_t serializeHeader(
        size_t output_index, uint8_t* buf, size_t max_size
    ) const override {
        (void)output_index;
        if (max_size < 5) return 0;
        std::memcpy(buf, &block_size_, sizeof(uint32_t));
        buf[4] = element_width_;
        return 5;
    }

    void deserializeHeader(const uint8_t* buf, size_t size) override {
        if (size >= 4) std::memcpy(&block_size_, buf, sizeof(uint32_t));
        if (size >= 5) element_width_ = buf[4];
    }

    size_t getMaxHeaderSize(size_t) const override { return 5; }

private:
    bool     is_inverse_;
    uint32_t block_size_;    // bytes per chunk
    uint8_t  element_width_; // bytes per element (1, 2, 4, or 8)
    size_t   actual_output_size_;

    // Validate config and return N_chunk (elements per chunk).
    // block_size must be a multiple of 1024*element_width so that butterfly
    // kernels always have full warps in every __shfl_xor_sync call.
    size_t validateConfig() const {
        if (element_width_ != 1 && element_width_ != 2 &&
            element_width_ != 4 && element_width_ != 8)
            throw std::invalid_argument(
                "BitshuffleStage: element_width must be 1, 2, 4, or 8");
        if (block_size_ == 0 || block_size_ % (1024u * element_width_) != 0)
            throw std::invalid_argument(
                "BitshuffleStage: block_size must be a positive multiple of "
                "1024 * element_width (default 16384 satisfies this for all "
                "supported element widths)");
        return block_size_ / element_width_;
    }
};

} // namespace fz
