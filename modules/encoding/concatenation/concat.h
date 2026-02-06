#pragma once

#include "stage/stage.h"
#include <cstdint>
#include <vector>

namespace fz {

/**
 * Concatenation Stage - Merges multiple input buffers into a single output
 * 
 * This stage takes multiple dependency outputs and concatenates them with size headers.
 * Useful for combining different compression streams (e.g., codes + indices + values)
 * 
 * Output format:
 *   [num_inputs: uint32_t]
 *   [size_0: uint32_t][size_1: uint32_t]...[size_n: uint32_t]
 *   [data_0][data_1]...[data_n]
 * 
 * Usage in DAG:
 *   - Specify multiple dependencies
 *   - Concatenate stage reads each dependency's output buffer and size
 *   - Supports parallel execution of dependencies
 */
class ConcatenationStage : public Stage {
public:
    ConcatenationStage();
    virtual ~ConcatenationStage() = default;
    
    // ========== Stage Interface ==========
    
    /**
     * Execute concatenation
     * NOTE: This stage requires special handling in DAG - it reads from multiple dependencies
     * @param input Not used (reads from DAG dependencies instead)
     * @param input_size Not used
     * @param output Output buffer for concatenated data
     * @param stream CUDA stream
     * @return Total output size in bytes
     */
    int execute(void* input, size_t input_size,
               void* output, cudaStream_t stream) override;
    
    /**
     * Add concatenation to CUDA graph
     */
    cudaGraphNode_t addToGraph(
        cudaGraph_t graph,
        cudaGraphNode_t* dependencies,
        size_t num_deps,
        void* input, size_t input_size,
        void* output,
        const std::vector<void*>& aux_buffers,
        cudaStream_t stream) override;
    
    /**
     * Get memory requirements
     */
    StageMemoryRequirements getMemoryRequirements(size_t input_size) const override;
    
    /**
     * Get maximum output size
     * Assumes worst case: all inputs at max size
     */
    size_t getMaxOutputSize(size_t input_size) const override;
    
    /**
     * Get average output size
     */
    size_t getAverageOutputSize(size_t input_size) const override;
    
    /**
     * Set dependency information (called by DAG)
     * This allows the concatenation stage to read from multiple dependency outputs
     */
    void setDependencyInfo(const std::vector<void*>& dep_outputs,
                          const std::vector<size_t>& dep_sizes);
    
private:
    // Dependency information (set by DAG before execution)
    std::vector<void*> dependency_outputs_;
    std::vector<size_t> dependency_sizes_;
};

} // namespace fz
