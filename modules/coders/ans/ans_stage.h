#pragma once

/**
 * @file ans_stage.h
 * @brief ANS stage — TODO: one-line description.
 */

#include "stage/stage.h"
#include "fzm_format.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace fz {

/**
 * ANS stage.
 *
 * TODO: describe what this stage does, its input/output types,
 * and any configuration parameters.
 */
class ANSStage : public Stage {
public:
    ANSStage() = default;

    // ── Stage control ─────────────────────────────────────────────────────────
    void setInverse(bool inv) override { is_inverse_ = inv; }
    bool isInverse() const override    { return is_inverse_; }

    // ── Execution ─────────────────────────────────────────────────────────────
    void execute(
        cudaStream_t stream,
        MemoryPool* pool,
        const std::vector<void*>& inputs,
        const std::vector<void*>& outputs,
        const std::vector<size_t>& sizes
    ) override;

    // ── Metadata ──────────────────────────────────────────────────────────────
    std::string getName() const override { return "ANS"; }
    size_t getNumInputs()  const override { return 1; }
    size_t getNumOutputs() const override { return 1; }

    std::vector<size_t> estimateOutputSizes(
        const std::vector<size_t>& input_sizes
    ) const override {
        // TODO: return a safe upper bound for both forward AND inverse directions.
        // Non-size-preserving stages must check is_inverse_ and return the correct
        // bound for each direction — the DAG allocates output buffers before execute().
        return {input_sizes.empty() ? 0 : input_sizes[0]};
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

    uint8_t getOutputDataType(size_t /*output_index*/) const override {
        // TODO: return the DataType of the output, or DataType::UNKNOWN
        // for byte-transparent stages.
        return static_cast<uint8_t>(DataType::UNKNOWN);
    }

    uint8_t getInputDataType(size_t /*input_index*/) const override {
        // TODO: return the expected DataType of the input, or DataType::UNKNOWN
        // to opt out of finalize() type-checking.
        return static_cast<uint8_t>(DataType::UNKNOWN);
    }

    // ── Serialization ─────────────────────────────────────────────────────────
    size_t serializeHeader(
        size_t output_index, uint8_t* buf, size_t max_size
    ) const override {
        // TODO: write config bytes into buf (max 128 bytes). Return bytes written.
        (void)output_index; (void)buf; (void)max_size;
        return 0;
    }

    void deserializeHeader(const uint8_t* buf, size_t size) override {
        // TODO: restore config from buf.
        (void)buf; (void)size;
    }

    size_t getMaxHeaderSize(size_t /*output_index*/) const override {
        return 0; // TODO: update to match bytes written in serializeHeader()
    }

    // Uncomment if deserializeHeader() overwrites fields also used by forward passes:
    // void saveState()    override { saved_config_ = config_; }
    // void restoreState() override { config_ = saved_config_; }

    // Uncomment if this stage holds persistent pool allocations:
    // size_t estimateScratchBytes(const std::vector<size_t>& input_sizes) const override;

    // Uncomment if execute() does D2H copies or host-side branching on device data:
    // bool isGraphCompatible() const override { return false; }

    // Uncomment if input must be aligned to a chunk boundary:
    // size_t getRequiredInputAlignment() const override { return chunk_bytes_; }

private:
    bool   is_inverse_         = false;
    size_t actual_output_size_ = 0;
    // TODO: add config fields here
};

} // namespace fz
