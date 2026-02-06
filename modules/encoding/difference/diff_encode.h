#pragma once

#include "stage/stage.h"
#include <cuda_runtime.h>

namespace fz {

/**
 * Difference encoder configuration
 */
template<typename T>
struct DifferenceConfig {
    DifferenceConfig() {}
};

/**
 * Difference encoding stage
 * Converts values to differences: output[i] = input[i] - input[i-1]
 * First element is stored as-is
 */
template<typename T>
class DifferenceStage : public Stage {
public:
    DifferenceStage(const DifferenceConfig<T>& config = DifferenceConfig<T>());
    ~DifferenceStage() override;
    
    StageMemoryRequirements getMemoryRequirements(size_t input_size) const override;
    size_t getMaxOutputSize(size_t input_size) const override;
    size_t getAverageOutputSize(size_t input_size) const override;
    
    int execute(void* input, size_t input_size,
                void* output, cudaStream_t stream) override;
    
    cudaGraphNode_t addToGraph(cudaGraph_t graph,
                              cudaGraphNode_t* dependencies,
                              size_t num_deps,
                              void* input, size_t input_size,
                              void* output,
                              const std::vector<void*>& aux_buffers,
                              cudaStream_t stream) override;
    
private:
    DifferenceConfig<T> config_;
};

} // namespace fz
