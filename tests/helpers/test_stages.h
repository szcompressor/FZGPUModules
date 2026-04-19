#pragma once

/**
 * tests/helpers/test_stages.h
 *
 * Lightweight scaffold stages used by pipeline tests and dev examples.
 * These are NOT production stages — they have no serializeHeader/deserializeHeader
 * and cannot round-trip through .fzm files.  They exist solely to construct
 * interesting DAG topologies without depending on real compression logic.
 *
 * Stages provided:
 *   PassThroughStage — identity D2D copy (1 input → 1 output)
 *   ScaleStage       — multiplies float data by 2.0 (or ÷2 in inverse)
 *   SplitStage       — copies input to 2 outputs ("copy1", "copy2")
 *   Split3Stage      — copies input to 3 outputs
 *   MergeStage       — concatenates 2 inputs into 1 output
 */

#include "stage/stage.h"
#include "fzm_format.h"
#include <cuda_runtime.h>
#include <cstdint>

namespace fz_test {

// ─────────────────────────────────────────────────────────────────────────────
// PassThroughStage — identity D2D copy
// ─────────────────────────────────────────────────────────────────────────────
class PassThroughStage : public fz::Stage {
public:
    PassThroughStage() : actual_output_size_(0), is_inverse_(false) {}

    void execute(
        cudaStream_t stream, fz::MemoryPool*,
        const std::vector<void*>& inputs,
        const std::vector<void*>& outputs,
        const std::vector<size_t>& sizes
    ) override {
        cudaMemcpyAsync(outputs[0], inputs[0], sizes[0], cudaMemcpyDeviceToDevice, stream);
        actual_output_size_ = sizes[0];
    }

    std::string getName()      const override { return "PassThrough"; }
    size_t getNumInputs()      const override { return 1; }
    size_t getNumOutputs()     const override { return 1; }
    void   setInverse(bool v)        override { is_inverse_ = v; }
    bool   isInverse()         const override { return is_inverse_; }

    std::vector<size_t> estimateOutputSizes(
        const std::vector<size_t>& in) const override { return {in[0]}; }

    std::unordered_map<std::string, size_t>
    getActualOutputSizesByName() const override {
        return {{"output", actual_output_size_}};
    }

    uint16_t getStageTypeId() const override {
        return static_cast<uint16_t>(fz::StageType::PASSTHROUGH);
    }
    uint8_t getOutputDataType(size_t) const override {
        return static_cast<uint8_t>(fz::DataType::UINT8);
    }

private:
    size_t actual_output_size_;
    bool   is_inverse_;
};

// ─────────────────────────────────────────────────────────────────────────────
// ScaleStage — multiply float data by 2.0 (÷2 in inverse mode)
// ─────────────────────────────────────────────────────────────────────────────

// Forward-declared kernel; defined in test_stages.cu
void launch_scale_kernel(float* out, const float* in, size_t n, float factor,
                         cudaStream_t stream);

class ScaleStage : public fz::Stage {
public:
    ScaleStage() : actual_output_size_(0), is_inverse_(false) {}

    void execute(
        cudaStream_t stream, fz::MemoryPool*,
        const std::vector<void*>& inputs,
        const std::vector<void*>& outputs,
        const std::vector<size_t>& sizes
    ) override {
        size_t n      = sizes[0] / sizeof(float);
        float  factor = is_inverse_ ? 0.5f : 2.0f;
        launch_scale_kernel(static_cast<float*>(outputs[0]),
                            static_cast<const float*>(inputs[0]),
                            n, factor, stream);
        actual_output_size_ = sizes[0];
    }

    std::string getName()      const override { return "Scale"; }
    size_t getNumInputs()      const override { return 1; }
    size_t getNumOutputs()     const override { return 1; }
    void   setInverse(bool v)        override { is_inverse_ = v; }
    bool   isInverse()         const override { return is_inverse_; }

    std::vector<size_t> estimateOutputSizes(
        const std::vector<size_t>& in) const override { return {in[0]}; }

    std::unordered_map<std::string, size_t>
    getActualOutputSizesByName() const override {
        return {{"output", actual_output_size_}};
    }

    uint16_t getStageTypeId() const override {
        return static_cast<uint16_t>(fz::StageType::SCALE);
    }
    uint8_t getOutputDataType(size_t) const override {
        return static_cast<uint8_t>(fz::DataType::FLOAT32);
    }

private:
    size_t actual_output_size_;
    bool   is_inverse_;
};

// ─────────────────────────────────────────────────────────────────────────────
// SplitStage — copy input to two outputs
// ─────────────────────────────────────────────────────────────────────────────
class SplitStage : public fz::Stage {
public:
    SplitStage() : actual_size_(0) {}

    void execute(
        cudaStream_t stream, fz::MemoryPool*,
        const std::vector<void*>& inputs,
        const std::vector<void*>& outputs,
        const std::vector<size_t>& sizes
    ) override {
        cudaMemcpyAsync(outputs[0], inputs[0], sizes[0], cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(outputs[1], inputs[0], sizes[0], cudaMemcpyDeviceToDevice, stream);
        actual_size_ = sizes[0];
    }

    std::string getName()     const override { return "Split"; }
    size_t getNumInputs()     const override { return 1; }
    size_t getNumOutputs()    const override { return 2; }

    std::vector<std::string> getOutputNames() const override {
        return {"copy1", "copy2"};
    }

    std::vector<size_t> estimateOutputSizes(
        const std::vector<size_t>& in) const override { return {in[0], in[0]}; }

    std::unordered_map<std::string, size_t>
    getActualOutputSizesByName() const override {
        return {{"copy1", actual_size_}, {"copy2", actual_size_}};
    }

    uint16_t getStageTypeId() const override {
        return static_cast<uint16_t>(fz::StageType::SPLIT);
    }
    uint8_t getOutputDataType(size_t) const override {
        return static_cast<uint8_t>(fz::DataType::UINT8);
    }

private:
    size_t actual_size_;
};

// ─────────────────────────────────────────────────────────────────────────────
// Split3Stage — copy input to three outputs
// ─────────────────────────────────────────────────────────────────────────────
class Split3Stage : public fz::Stage {
public:
    Split3Stage() : actual_size_(0) {}

    void execute(
        cudaStream_t stream, fz::MemoryPool*,
        const std::vector<void*>& inputs,
        const std::vector<void*>& outputs,
        const std::vector<size_t>& sizes
    ) override {
        cudaMemcpyAsync(outputs[0], inputs[0], sizes[0], cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(outputs[1], inputs[0], sizes[0], cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(outputs[2], inputs[0], sizes[0], cudaMemcpyDeviceToDevice, stream);
        actual_size_ = sizes[0];
    }

    std::string getName()     const override { return "Split3"; }
    size_t getNumInputs()     const override { return 1; }
    size_t getNumOutputs()    const override { return 3; }

    std::vector<std::string> getOutputNames() const override {
        return {"copy1", "copy2", "copy3"};
    }

    std::vector<size_t> estimateOutputSizes(
        const std::vector<size_t>& in) const override { return {in[0], in[0], in[0]}; }

    std::unordered_map<std::string, size_t>
    getActualOutputSizesByName() const override {
        return {{"copy1", actual_size_}, {"copy2", actual_size_}, {"copy3", actual_size_}};
    }

    uint16_t getStageTypeId() const override {
        return static_cast<uint16_t>(fz::StageType::SPLIT);
    }
    uint8_t getOutputDataType(size_t) const override {
        return static_cast<uint8_t>(fz::DataType::UINT8);
    }

private:
    size_t actual_size_;
};

// ─────────────────────────────────────────────────────────────────────────────
// MergeStage — concatenate two inputs into one output
// ─────────────────────────────────────────────────────────────────────────────
class MergeStage : public fz::Stage {
public:
    MergeStage() : actual_output_size_(0) {}

    void execute(
        cudaStream_t stream, fz::MemoryPool*,
        const std::vector<void*>& inputs,
        const std::vector<void*>& outputs,
        const std::vector<size_t>& sizes
    ) override {
        uint8_t* out = static_cast<uint8_t*>(outputs[0]);
        cudaMemcpyAsync(out,           inputs[0], sizes[0], cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(out + sizes[0], inputs[1], sizes[1], cudaMemcpyDeviceToDevice, stream);
        actual_output_size_ = sizes[0] + sizes[1];
    }

    std::string getName()     const override { return "Merge"; }
    size_t getNumInputs()     const override { return 2; }
    size_t getNumOutputs()    const override { return 1; }

    std::vector<size_t> estimateOutputSizes(
        const std::vector<size_t>& in) const override {
        size_t total = 0;
        for (size_t s : in) total += s;
        return {total};
    }

    std::unordered_map<std::string, size_t>
    getActualOutputSizesByName() const override {
        return {{"output", actual_output_size_}};
    }

    uint16_t getStageTypeId() const override {
        return static_cast<uint16_t>(fz::StageType::MERGE);
    }
    uint8_t getOutputDataType(size_t) const override {
        return static_cast<uint8_t>(fz::DataType::UINT8);
    }

private:
    size_t actual_output_size_;
};

} // namespace fz_test
