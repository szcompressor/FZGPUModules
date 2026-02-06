#pragma once

#include "stage/stage.h"
#include <cuda_runtime.h>
#include <cstddef>
#include <type_traits>

namespace fz {

/**
 * Configuration for Run-Length Encoding stage
 */
template<typename TInput>
struct RLEConfig {
    static_assert(std::is_unsigned<TInput>::value, "TInput must be unsigned integer");
    
    float run_capacity_ratio;   // Expected ratio of runs to input elements
    
    RLEConfig(float ratio = 0.5f)
        : run_capacity_ratio(ratio) {}
};

/**
 * Run-Length Encoding Stage
 * 
 * **GPU Algorithm: Two-Pass Parallel RLE**
 * 
 * Pass 1 - Identify Run Boundaries (Parallel):
 *   - Each thread compares its element with the next
 *   - Mark positions where value changes (run boundary)
 *   - Parallel prefix sum to get run indices
 * 
 * Pass 2 - Encode Runs (Parallel):
 *   - Each thread handles one run
 *   - Read run start position and scan forward to count length
 *   - Write (value, count) pair to output
 * 
 * This approach parallelizes the inherently sequential RLE operation
 * by identifying all run boundaries first, then encoding in parallel.
 * 
 * Input: Quantization codes [uint8/uint16/uint32]
 * Output: [num_runs (4B)] [value1, count1, value2, count2, ...]
 * 
 * Example:
 *   Input:  [5, 5, 5, 5, 7, 7, 5]
 *   Flags:  [0, 0, 0, 1, 0, 1, 1]  (1 = run boundary)
 *   Scan:   [0, 0, 0, 1, 1, 2, 3]  (run indices)
 *   Output: [3 runs] [5, 4] [7, 2] [5, 1]
 */
template<typename TInput = uint16_t>
class RLEStage : public Stage {
public:
    explicit RLEStage(const RLEConfig<TInput>& config = RLEConfig<TInput>());
    ~RLEStage() override;
    
    int execute(void* input, size_t input_size,
               void* output, cudaStream_t stream) override;
    
    cudaGraphNode_t addToGraph(cudaGraph_t graph,
                              cudaGraphNode_t* dependencies,
                              size_t num_deps,
                              void* input, size_t input_size,
                              void* output,
                              const std::vector<void*>& aux_buffers,
                              cudaStream_t stream) override;
    
    StageMemoryRequirements getMemoryRequirements(size_t input_size) const override;
    
    size_t getMaxOutputSize(size_t input_size) const override {
        size_t num_elements = input_size / sizeof(TInput);
        // Worst case: every element is a run
        return sizeof(uint32_t) + (num_elements * (sizeof(TInput) + sizeof(uint32_t)));
    }
    
    size_t getAverageOutputSize(size_t input_size) const override {
        return input_size / 2 + sizeof(uint32_t);
    }
    
    uint32_t getLastNumRuns() const { return last_num_runs_; }
    float getCompressionRatio() const { return last_compression_ratio_; }
    
    StageMetadata getMetadata() const override {
        StageMetadata meta(name_);
        meta.register_pressure = 16;
        meta.shared_memory_bytes = 0;
        meta.is_memory_bound = true;
        meta.produces_variable_output = true;
        return meta;
    }
    
private:
    RLEConfig<TInput> config_;
    uint32_t last_num_runs_;
    float last_compression_ratio_;
    
    // Graph mode: persistent storage for kernel parameters
    const TInput* graph_d_input_;
    uint8_t* graph_d_output_;
    uint32_t* graph_d_run_counter_;
    size_t graph_num_elements_;
};

} // namespace fz
