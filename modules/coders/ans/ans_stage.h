#pragma once

/**
 * @file ans_stage.h
 * @brief rANS entropy coding stage (GPU, via vendored dietGPU kernel templates).
 *
 * Forward: uint8_t[] → ANS compressed bitstream (ANSCoalescedHeader + rANS data).
 * Inverse: ANS bitstream → uint8_t[].
 *
 * D2H copies occur in both directions, so isGraphCompatible() returns false.
 *
 * Only prob_bits=10 is supported in this build (baked into GpuANS.cu template
 * instantiations).  setProbBits() is reserved for future use; calling execute()
 * after setProbBits(n != 10) throws std::runtime_error.
 *
 * @note GPU rANS kernels adapted from dietGPU (Meta Platforms, Inc., MIT License).
 *       See THIRD_PARTY.md for the full license text and attribution.
 *
 * Serialized FZM header layout (12 bytes):
 *   [0]     prob_bits       (uint8_t)
 *   [1..3]  reserved        (3 bytes, zero)
 *   [4..11] original_bytes_ (uint64_t LE, uncompressed byte count)
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

class ANSStage : public Stage {
public:
    ANSStage()  = default;
    ~ANSStage() override = default;

    // ── Configuration ─────────────────────────────────────────────────────────

    /**
     * Set the probability resolution in bits (default 10 → 1024 buckets).
     * Must be in [1, 11].  Higher values give better compression at the cost
     * of a larger decode table.  Change before the first execute() call.
     */
    void    setProbBits(uint8_t pb) { prob_bits_ = pb; }
    uint8_t getProbBits() const     { return prob_bits_; }

    // ── Stage control ─────────────────────────────────────────────────────────
    void setInverse(bool inv) override { is_inverse_ = inv; }
    bool isInverse() const override    { return is_inverse_; }

    // D2H copies occur in both encode (header readback) and decode (header peek).
    bool isGraphCompatible() const override { return false; }

    // dietGPU requires input aligned to 4 bytes (kANSRequiredAlignment).
    size_t getRequiredInputAlignment() const override { return 4; }

    // ── Pool lifecycle ────────────────────────────────────────────────────────

    /**
     * Pre-allocates all 7 persistent scratch buffers from the pool so that
     * PREALLOCATE mode commits all memory at finalize time.  If estimated_inlen
     * is 0, allocation is deferred to the first execute() call.
     */
    void onFinalize(size_t estimated_inlen, MemoryPool* pool) override;

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
    std::string getName()       const override { return "ANS"; }
    size_t      getNumInputs()  const override { return 1; }
    size_t      getNumOutputs() const override { return 1; }

    std::vector<size_t> estimateOutputSizes(
        const std::vector<size_t>& input_sizes
    ) const override {
        if (input_sizes.empty()) return {0};
        if (!is_inverse_) {
            // ANS worst-case expansion is ~1.25× per block plus fixed header overhead.
            // 2× + 8 KiB is a conservative but safe bound for all realistic inputs.
            return {input_sizes[0] * 2 + 8192};
        }
        // original_bytes_ is restored from the serialized FZM header before execute();
        // fall back to input size if deserializeHeader() has not yet been called.
        return {original_bytes_ > 0 ? original_bytes_ : input_sizes[0]};
    }

    std::unordered_map<std::string, size_t>
    getActualOutputSizesByName() const override {
        return {{"output", actual_output_size_}};
    }

    size_t getActualOutputSize(int index) const override {
        return (index == 0) ? actual_output_size_ : 0;
    }

    // ── Type system ───────────────────────────────────────────────────────────
    uint16_t getStageTypeId() const override {
        return static_cast<uint16_t>(StageType::ANS);
    }

    // Byte-transparent: opt out of pipeline type-compatibility checking.
    uint8_t getOutputDataType(size_t /*output_index*/) const override {
        return static_cast<uint8_t>(DataType::UNKNOWN);
    }
    uint8_t getInputDataType(size_t /*input_index*/) const override {
        return static_cast<uint8_t>(DataType::UNKNOWN);
    }

    // ── Serialization ─────────────────────────────────────────────────────────
    size_t serializeHeader(
        size_t /*output_index*/, uint8_t* buf, size_t max_size
    ) const override {
        if (max_size < 12) return 0;
        buf[0] = prob_bits_;
        buf[1] = buf[2] = buf[3] = 0;
        std::memcpy(buf + 4, &original_bytes_, sizeof(uint64_t));
        return 12;
    }

    void deserializeHeader(const uint8_t* buf, size_t size) override {
        if (size >= 1)
            prob_bits_ = buf[0];
        if (size >= 12)
            std::memcpy(&original_bytes_, buf + 4, sizeof(uint64_t));
    }

    size_t getMaxHeaderSize(size_t /*output_index*/) const override { return 12; }

    void saveState() override {
        saved_prob_bits_      = prob_bits_;
        saved_original_bytes_ = original_bytes_;
        saved_output_size_    = actual_output_size_;
    }

    void restoreState() override {
        prob_bits_           = saved_prob_bits_;
        original_bytes_      = saved_original_bytes_;
        actual_output_size_  = saved_output_size_;
    }

private:
    bool     is_inverse_         = false;
    uint8_t  prob_bits_          = 10;  // kANSDefaultProbBits
    uint64_t original_bytes_     = 0;   // set by forward execute; used by inverse estimateOutputSizes
    size_t   actual_output_size_ = 0;

    // Capacity (input bytes) of current scratch allocation. Grow-only.
    size_t cap_bytes_ = 0;

    // Persistent scratch device pointers, sub-allocated from MemoryPool.
    // All are null until initScratch() is called.
    uint32_t* d_temp_histogram_    = nullptr;  // uint32_t[256]
    void*     d_table_             = nullptr;  // uint4[256] (encode table: pdf/cdf/mul/shift)
    uint8_t*  d_compressed_blocks_ = nullptr;  // uint8_t[max_blocks * kUncoalescedStride]
    uint32_t* d_compressed_words_  = nullptr;  // uint32_t[max_blocks]
    uint32_t* d_comp_words_prefix_ = nullptr;  // uint32_t[max_blocks]
    void*     d_temp_prefix_sum_   = nullptr;  // CUB temp storage (nullptr when blocks ≤ 512)
    uint32_t* d_decode_table_      = nullptr;  // uint32_t[1 << prob_bits_]

    // D2H readback buffer for ANSCoalescedHeader after forward encode.
    // Stored as raw bytes to avoid pulling the dietgpu headers into this header.
    // sizeof(ANSCoalescedHeader) == 32.
    uint8_t last_header_bytes_[32] = {};

    // Histogram launch params — computed in initScratch(), reused every execute().
    int hist_grid_dim_    = 0;
    int hist_block_dim_   = 0;
    int hist_shmem_use_   = 0;
    int hist_r_per_block_ = 0;

    // saveState / restoreState snapshots
    uint8_t  saved_prob_bits_      = 10;
    uint64_t saved_original_bytes_ = 0;
    size_t   saved_output_size_    = 0;

    // Allocates all 7 scratch buffers from pool and computes histogram launch
    // params for the given input capacity.  Replaces any previous allocation.
    void initScratch(size_t inlen, MemoryPool* pool);

    // execute()'s forward/inverse bodies, split into separate functions so that
    // no single function holds both branches' local variables, kernel launches,
    // and conditional (vGPU) code paths at once — see ans_stage.cu for why this
    // matters (an nvc++ host-codegen bug producing a misaligned stack-argument
    // store for CUDA kernel launches in the combined function).
    void executeForward(fz::stream_t stream, MemoryPool* pool,
                        uint8_t* in, uint8_t* out, size_t byte_size);
    void executeInverse(fz::stream_t stream, MemoryPool* pool,
                        uint8_t* in, uint8_t* out);

    // Per-block scratch stride (bytes): ANSWarpState (128 B) + max raw compressed
    // block (5120 B = roundUp(4096 + 4096/4, 16)).  Derived from dietGPU constants.
    static constexpr size_t kUncoalescedStride = 128 + 5120;  // 5248
};

} // namespace fz
