#pragma once

#include "stage/stage.h"
#include <cuda_runtime.h>
#include <cstddef>
#include <type_traits>

namespace fz {

/**
 * Configuration for Bitpacking stage
 */
template<typename TInput>
struct BitpackingConfig {
    static_assert(std::is_unsigned<TInput>::value, "TInput must be unsigned integer");
    
    size_t block_size;  // Number of elements per block
    
    BitpackingConfig(size_t blk_size = 1024)
        : block_size(blk_size) {}
};

/**
 * Bitpacking Stage
 * 
 * **Algorithm: Block-Based Bitpacking**
 * 
 * 1. Divide input into fixed-size blocks (e.g., 1024 elements)
 * 2. For each block:
 *    - Find min and max values
 *    - Calculate bits needed: log2(max - min + 1)
 *    - Store block header: [min_value, num_bits]
 *    - Pack (value - min) using exactly num_bits per element
 * 
 * This exploits local value clustering (e.g., Lorenzo codes cluster around 512)
 * 
 * Output Format:
 *   [num_blocks (4B)]
 *   [block_headers: min_value (2B) | num_bits (1B) | reserved (1B)] * num_blocks
 *   [packed_data: tightly packed bits for all blocks]
 * 
 * Example:
 *   Block of 1024 values in range [510, 515]
 *   - Header: min=510, bits=3 (range 0-5 needs 3 bits)
 *   - Packed: 1024 * 3 = 3072 bits = 384 bytes
 *   - Original: 1024 * 2 bytes = 2048 bytes
 *   - Compression: 5.33x
 */
template<typename TInput = uint16_t>
class BitpackingStage : public Stage {
public:
    explicit BitpackingStage(const BitpackingConfig<TInput>& config = BitpackingConfig<TInput>());
    ~BitpackingStage() override;
    
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
        size_t num_blocks = (num_elements + config_.block_size - 1) / config_.block_size;
        
        // Worst case: all blocks need full bits (16 for uint16_t)
        size_t header_size = sizeof(uint32_t) + num_blocks * 4;  // num_blocks + block headers
        size_t data_size = input_size;  // Worst case = original size
        
        return header_size + data_size;
    }
    
    size_t getAverageOutputSize(size_t input_size) const override {
        // Assume average 10 bits per value for Lorenzo codes
        return input_size * 10 / 16 + 4096;  // ~60% of input + headers
    }
    
    StageMetadata getMetadata() const override {
        StageMetadata meta(name_);
        meta.register_pressure = 20;
        meta.shared_memory_bytes = 0;
        meta.is_memory_bound = true;
        meta.produces_variable_output = true;
        return meta;
    }
    
private:
    BitpackingConfig<TInput> config_;
    
    // Graph mode: persistent storage for kernel parameters
    const TInput* graph_d_input_;
    uint8_t* graph_d_output_;
    size_t graph_num_elements_;
};

} // namespace fz
