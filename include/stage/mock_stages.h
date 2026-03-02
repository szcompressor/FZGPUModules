#pragma once

#include "stage/stage.h"
#include "fzm_format.h"
#include <cstdint>
#include <cstring>

namespace fz {

/**
 * Mock stage: Pass-through (identity)
 * Just copies input to output
 */
class PassThroughStage : public Stage {
public:
    PassThroughStage() : actual_output_size_(0), is_inverse_(false) {}
    
    void execute(
        cudaStream_t stream,
        const std::vector<void*>& inputs,
        const std::vector<void*>& outputs,
        const std::vector<size_t>& sizes
    ) override {
        // Simple device-to-device copy
        cudaMemcpyAsync(outputs[0], inputs[0], sizes[0],
                       cudaMemcpyDeviceToDevice, stream);
        actual_output_size_ = sizes[0];
    }
    
    std::string getName() const override { return "PassThrough"; }
    size_t getNumInputs() const override { return 1; }
    size_t getNumOutputs() const override { return 1; }
    
    std::vector<size_t> estimateOutputSizes(
        const std::vector<size_t>& input_sizes
    ) const override {
        return {input_sizes[0]};  // Same size
    }
    
    std::unordered_map<std::string, size_t> getActualOutputSizesByName() const override {
        return {{"output", actual_output_size_}};
    }
    
    uint16_t getStageTypeId() const override {
        return static_cast<uint16_t>(StageType::PASSTHROUGH);
    }
    
    uint8_t getOutputDataType(size_t output_index) const override {
        (void)output_index;
        return static_cast<uint8_t>(DataType::UINT8);  // Generic byte data
    }
    
    void setInverse(bool inverse) { is_inverse_ = inverse; }
    bool isInverse() const { return is_inverse_; }
    
private:
    size_t actual_output_size_;
    bool is_inverse_;
};

/**
 * Mock stage: Scale by 2
 * Multiplies float data by 2 (or divides in inverse mode)
 */
class ScaleStage : public Stage {
public:
    ScaleStage() : actual_output_size_(0), is_inverse_(false) {}
    
    void execute(
        cudaStream_t stream,
        const std::vector<void*>& inputs,
        const std::vector<void*>& outputs,
        const std::vector<size_t>& sizes
    ) override;
    
    std::string getName() const override { return "Scale"; }
    size_t getNumInputs() const override { return 1; }
    size_t getNumOutputs() const override { return 1; }
    
    std::vector<size_t> estimateOutputSizes(
        const std::vector<size_t>& input_sizes
    ) const override {
        return {input_sizes[0]};  // Same size
    }
    
    std::unordered_map<std::string, size_t> getActualOutputSizesByName() const override {
        return {{"output", actual_output_size_}};
    }
    
    uint16_t getStageTypeId() const override {
        return static_cast<uint16_t>(StageType::SCALE);
    }
    
    uint8_t getOutputDataType(size_t output_index) const override {
        (void)output_index;
        return static_cast<uint8_t>(DataType::FLOAT32);  // Scales float data
    }
    
    void setInverse(bool inverse) { is_inverse_ = inverse; }
    bool isInverse() const { return is_inverse_; }
    
private:
    size_t actual_output_size_;
    bool is_inverse_;
};

/**
 * Mock stage: Split into two copies
 * Demonstrates multiple outputs
 */
class SplitStage : public Stage {
public:
    SplitStage() : actual_size_(0) {}
    
    void execute(
        cudaStream_t stream,
        const std::vector<void*>& inputs,
        const std::vector<void*>& outputs,
        const std::vector<size_t>& sizes
    ) override {
        // Copy to both outputs
        cudaMemcpyAsync(outputs[0], inputs[0], sizes[0],
                       cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(outputs[1], inputs[0], sizes[0],
                       cudaMemcpyDeviceToDevice, stream);
        actual_size_ = sizes[0];
    }
    
    std::string getName() const override { return "Split"; }
    size_t getNumInputs() const override { return 1; }
    size_t getNumOutputs() const override { return 2; }
    
    std::vector<std::string> getOutputNames() const override {
        return {"copy1", "copy2"};
    }
    
    std::vector<size_t> estimateOutputSizes(
        const std::vector<size_t>& input_sizes
    ) const override {
        return {input_sizes[0], input_sizes[0]};  // Both same size
    }
    
    std::unordered_map<std::string, size_t> getActualOutputSizesByName() const override {
        return {{"copy1", actual_size_}, {"copy2", actual_size_}};
    }
    
    uint16_t getStageTypeId() const override {
        return static_cast<uint16_t>(StageType::SPLIT);
    }
    
    uint8_t getOutputDataType(size_t output_index) const override {
        (void)output_index;
        return static_cast<uint8_t>(DataType::UINT8);  // Generic byte data
    }
    
private:
    size_t actual_size_;
};

/**
 * Mock stage: Split into three copies
 * Demonstrates multiple outputs with more parallelism
 */
class Split3Stage : public Stage {
public:
    Split3Stage() : actual_size_(0) {}
    
    void execute(
        cudaStream_t stream,
        const std::vector<void*>& inputs,
        const std::vector<void*>& outputs,
        const std::vector<size_t>& sizes
    ) override {
        // Copy to all three outputs
        cudaMemcpyAsync(outputs[0], inputs[0], sizes[0],
                       cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(outputs[1], inputs[0], sizes[0],
                       cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(outputs[2], inputs[0], sizes[0],
                       cudaMemcpyDeviceToDevice, stream);
        actual_size_ = sizes[0];
    }
    
    std::string getName() const override { return "Split3"; }
    size_t getNumInputs() const override { return 1; }
    size_t getNumOutputs() const override { return 3; }
    
    std::vector<std::string> getOutputNames() const override {
        return {"copy1", "copy2", "copy3"};
    }
    
    std::vector<size_t> estimateOutputSizes(
        const std::vector<size_t>& input_sizes
    ) const override {
        return {input_sizes[0], input_sizes[0], input_sizes[0]};
    }
    
    std::unordered_map<std::string, size_t> getActualOutputSizesByName() const override {
        return {{"copy1", actual_size_}, {"copy2", actual_size_}, {"copy3", actual_size_}};
    }
    
    uint16_t getStageTypeId() const override {
        return static_cast<uint16_t>(StageType::SPLIT);  // Uses SPLIT type
    }
    
    uint8_t getOutputDataType(size_t output_index) const override {
        (void)output_index;
        return static_cast<uint8_t>(DataType::UINT8);  // Generic byte data
    }
    
private:
    size_t actual_size_;
};

/**
 * Mock stage: Merge two inputs
 * Simple concatenation
 */
class MergeStage : public Stage {
public:
    MergeStage() : actual_output_size_(0) {}
    
    void execute(
        cudaStream_t stream,
        const std::vector<void*>& inputs,
        const std::vector<void*>& outputs,
        const std::vector<size_t>& sizes
    ) override {
        // Copy both inputs to output sequentially
        size_t offset = 0;
        uint8_t* output = static_cast<uint8_t*>(outputs[0]);
        
        cudaMemcpyAsync(output + offset, inputs[0], sizes[0],
                       cudaMemcpyDeviceToDevice, stream);
        offset += sizes[0];
        
        cudaMemcpyAsync(output + offset, inputs[1], sizes[1],
                       cudaMemcpyDeviceToDevice, stream);
        offset += sizes[1];
        
        actual_output_size_ = offset;
    }
    
    std::string getName() const override { return "Merge"; }
    size_t getNumInputs() const override { return 2; }
    size_t getNumOutputs() const override { return 1; }
    
    std::vector<size_t> estimateOutputSizes(
        const std::vector<size_t>& input_sizes
    ) const override {
        size_t total = 0;
        for (size_t s : input_sizes) total += s;
        return {total};
    }
    
    std::unordered_map<std::string, size_t> getActualOutputSizesByName() const override {
        return {{"output", actual_output_size_}};
    }
    
    uint16_t getStageTypeId() const override {
        return static_cast<uint16_t>(StageType::MERGE);
    }
    
    uint8_t getOutputDataType(size_t output_index) const override {
        (void)output_index;
        return static_cast<uint8_t>(DataType::UINT8);  // Generic byte data
    }
    
private:
    size_t actual_output_size_;
};

} // namespace fz
