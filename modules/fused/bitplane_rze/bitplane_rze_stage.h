#pragma once

/**
 * @file bitplane_rze_stage.h
 * @brief Fused bitplane-transpose + zero-group RZE stage — the FZ-GPU lossless
 *        encoder, ported as a single-kernel pipeline stage.
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

/**
 * Fused bitplane-transpose + zero-group run-zero encoder (FZ-GPU's lossless
 * back-end), as a pipeline stage.
 *
 * In one shared-memory pass per 4096-byte chunk the forward kernel:
 *   1. bit-transposes a 32×32 matrix of `uint32_t` words (a bitshuffle at
 *      4-byte element width), then
 *   2. eliminates all-zero 4-byte groups, reserving compacted output space with
 *      a per-block `atomicAdd`.
 * Input is interpreted as `uint16_t` symbols (two per word), matching FZ-GPU's
 * quantizer codes; the typical pipeline is `QuantizerStage<float,uint16_t> →
 * BitplaneRZEStage`. The transpose is built in, so a separate `BitshuffleStage`
 * is **not** needed in front of it.
 *
 * ## Relationship to BitshuffleStage + RZEStage
 *
 * This stage is functionally close to `BitshuffleStage(element_width=4) →
 * RZEStage(levels=1)`, but it is **not** equivalent: (1) the transpose result
 * never leaves shared memory, so it avoids a full global-memory round-trip
 * (faster); (2) zero elimination is at 4-byte-group granularity rather than
 * per-byte; (3) it emits FZ-GPU's exact self-describing archive (per-block
 * atomic start positions + 128-byte header). It exists for FZ-GPU fidelity and
 * throughput, not to add new functionality. See `docs/stages/bitplane_rze.md`.
 *
 * Output is a self-describing byte archive (`uint8_t`): a 128-byte header
 * followed by the bitflag bitmap, per-block start positions, and the compacted
 * bitstream.
 *
 * @note **Not graph-compatible.** Both directions do a host-blocking
 *       `cudaStreamSynchronize` + D2H copy (the forward to read the compacted
 *       length, the inverse to read the archive header).
 *
 * @note Fixed `uint16_t` input — no template parameter (matches the reference
 *       kernels). u8/u32 support would need new kernels.
 *
 * @note **Prior work:** the encode/decode kernels are the FZ-GPU lossless codec
 *       (Boyuan Zhang et al., HPDC '23), BSD-3-Clause. The stage wrapper and
 *       memory-pool integration are FZGPUModules code. See `THIRD_PARTY.md`.
 */
class BitplaneRZEStage : public Stage {
public:
    BitplaneRZEStage() = default;
    ~BitplaneRZEStage() override;

    // ── Stage control ──────────────────────────────────────────────────────
    void setInverse(bool inv) override { is_inverse_ = inv; }
    bool isInverse() const override    { return is_inverse_; }

    // Both directions do a blocking D2H sync; never graph-capturable.
    bool isGraphCompatible() const override { return false; }

    // ── Execution ──────────────────────────────────────────────────────────
    void execute(
        cudaStream_t stream,
        MemoryPool* pool,
        const std::vector<void*>& inputs,
        const std::vector<void*>& outputs,
        const std::vector<size_t>& sizes
    ) override;

    // ── Metadata ───────────────────────────────────────────────────────────
    std::string getName() const override { return "BitplaneRZE"; }
    size_t getNumInputs()  const override { return 1; }
    size_t getNumOutputs() const override { return 1; }

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

    uint16_t getStageTypeId() const override {
        return static_cast<uint16_t>(StageType::BITPLANE_RZE);
    }

    // Forward output is a raw byte archive; inverse output is the uint16 codes.
    uint8_t getOutputDataType(size_t) const override {
        return static_cast<uint8_t>(is_inverse_ ? DataType::UINT16
                                                 : DataType::UINT8);
    }
    // Forward input is the uint16 code stream.
    uint8_t getInputDataType(size_t) const override {
        return static_cast<uint8_t>(is_inverse_ ? DataType::UINT8
                                                 : DataType::UINT16);
    }

    // ── Serialization ──────────────────────────────────────────────────────
    // Header (8 bytes): [0..7] original_len (uint64_t LE) — the pre-pad uint16
    // symbol count. The archive itself is self-describing, but the pipeline
    // needs this to size the inverse output buffer for a cold decompress.
    size_t serializeHeader(size_t, uint8_t* buf, size_t max_size) const override {
        if (max_size < sizeof(uint64_t)) return 0;
        uint64_t len = static_cast<uint64_t>(cached_orig_len_);
        std::memcpy(buf, &len, sizeof(uint64_t));
        return sizeof(uint64_t);
    }
    void deserializeHeader(const uint8_t* buf, size_t size) override {
        if (size >= sizeof(uint64_t)) {
            uint64_t len = 0;
            std::memcpy(&len, buf, sizeof(uint64_t));
            cached_orig_len_ = static_cast<size_t>(len);
        }
    }
    size_t getMaxHeaderSize(size_t) const override { return sizeof(uint64_t); }

    void saveState() override    { saved_orig_len_ = cached_orig_len_; }
    void restoreState() override { cached_orig_len_ = saved_orig_len_; }

    size_t getCachedOrigLen() const { return cached_orig_len_; }

private:
    bool   is_inverse_         = false;
    size_t actual_output_size_ = 0;
    // Pre-pad uint16 symbol count of the original input. Set by the forward
    // execute() and persisted in the FZM header so a cold inverse can size its
    // output buffer.
    size_t cached_orig_len_    = 0;
    size_t saved_orig_len_     = 0;
};

} // namespace fz
