#include "pipeline/compressor.h"
#include "predictors/lorenzo/lorenzo.h"
#include "encoding/run_length_encoding/RLE.h"
#include "encoding/bitpacking/bitpacking.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <set>
#include <map>
#include <algorithm>

using namespace fz;

// Helper structure matching the bitpacking format
struct BlockHeader {
    uint16_t min_value;
    uint8_t num_bits;
    uint8_t reserved;
};

/**
 * Decode and print bitpacked data
 */
void printBitpackedData(const uint8_t* data, size_t max_values_to_print = 100) {
    // Read number of blocks
    uint32_t num_blocks = *reinterpret_cast<const uint32_t*>(data);
    std::cout << "Bitpacking format:\n";
    std::cout << "  Number of blocks: " << num_blocks << "\n\n";
    
    // Read block headers
    const BlockHeader* headers = reinterpret_cast<const BlockHeader*>(data + sizeof(uint32_t));
    
    // Packed data starts after all headers
    const uint8_t* packed_data = data + sizeof(uint32_t) + num_blocks * sizeof(BlockHeader);
    
    std::cout << "Block headers (first 10 blocks):\n";
    std::cout << "  Block | Min Value | Bits/Element | Elements | Compression\n";
    std::cout << "  ------|-----------|--------------|----------|------------\n";
    
    size_t total_bits = 0;
    size_t total_elements = 0;
    
    for (uint32_t i = 0; i < std::min(num_blocks, 10u); i++) {
        size_t block_size = 1024;  // From config
        std::cout << "  " << std::setw(5) << i 
                  << " | " << std::setw(9) << headers[i].min_value
                  << " | " << std::setw(12) << (int)headers[i].num_bits
                  << " | " << std::setw(8) << block_size
                  << " | " << std::setw(6) << std::fixed << std::setprecision(2)
                  << (16.0f / headers[i].num_bits) << "x\n";
    }
    if (num_blocks > 10) {
        std::cout << "  ... (" << (num_blocks - 10) << " more blocks)\n";
    }
    std::cout << "\n";
    
    // Decode and print first N values
    std::cout << "Decoded values (first " << max_values_to_print << "):\n";
    
    size_t bit_offset = 0;
    size_t values_printed = 0;
    
    for (uint32_t block_idx = 0; block_idx < num_blocks && values_printed < max_values_to_print; block_idx++) {
        uint16_t min_val = headers[block_idx].min_value;
        uint8_t num_bits = headers[block_idx].num_bits;
        size_t block_size = 1024;
        
        for (size_t i = 0; i < block_size && values_printed < max_values_to_print; i++) {
            // Decode value from packed bits
            uint32_t value = 0;
            for (uint8_t bit = 0; bit < num_bits; bit++) {
                size_t byte_pos = bit_offset / 8;
                size_t bit_pos = bit_offset % 8;
                
                uint8_t bit_val = (packed_data[byte_pos] >> bit_pos) & 1;
                value |= (bit_val << bit);
                
                bit_offset++;
            }
            
            // Add back the minimum value
            uint16_t decoded_value = value + min_val;
            
            std::cout << decoded_value << " ";
            if ((values_printed + 1) % 20 == 0) std::cout << "\n";
            
            values_printed++;
            total_elements++;
            total_bits += num_bits;
        }
    }
    std::cout << "\n\n";
    
    std::cout << "Bitpacking statistics:\n";
    std::cout << "  Average bits per element: " << std::fixed << std::setprecision(2) 
              << ((double)total_bits / total_elements) << "\n";
    std::cout << "  Effective compression: " << (16.0 / ((double)total_bits / total_elements)) << "x\n";
}

/**
 * Example: Simple 1D Lorenzo compression pipeline
 * 
 * This demonstrates the minimal code needed to compress data
 * with the high-level Pipeline API.
 */
int main() {
    // ========== Generate Test Data ==========
    
    const size_t n = 1024 * 1024;  // 1M elements
    std::vector<float> h_input(n);
    
    // Generate smooth signal (highly compressible)
    for (size_t i = 0; i < n; i++) {
        h_input[i] = sin(i * 0.001) + cos(i * 0.0001);
    }
    
    std::cout << "Input: " << n << " floats (" 
              << (n * sizeof(float) / (1024.0 * 1024.0)) << " MB)\n" << std::endl;
    
    // print first 100 values
    std::cout << "First 100 input values:\n";
    for (size_t i = 0; i < 100; i++) {
        std::cout << std::fixed << std::setprecision(6) << h_input[i] << " ";
        if ((i + 1) % 20 == 0) std::cout << "\n";
    }
    std::cout << "\n";
    
    // Test both modes
    for (int mode = 0; mode < 2; mode++) {
        bool use_graph = (mode == 1);
        std::cout << "\n========================================\n";
        std::cout << "Testing " << (use_graph ? "GRAPH" : "STREAM") << " mode\n";
        std::cout << "========================================\n\n";
        
        // ========== Configure Pipeline ==========
        
        PipelineConfig config;
        config.use_cuda_graph = use_graph;
        config.use_parallel_streams = true;
        config.max_streams = 4;
        config.memory_pool_multiplier = 3.0f;
        config.enable_profiling = true;
        // Try CONSERVATIVE mode in stream mode, SAFE in graph mode
        config.memory_mode = use_graph ? MemoryMode::SAFE : MemoryMode::CONSERVATIVE;
        
        std::cout << "Memory mode: " 
                  << (config.memory_mode == MemoryMode::CONSERVATIVE ? "CONSERVATIVE" : "SAFE") 
                  << "\n\n";
        
        Pipeline pipeline(config);
        
        // ========== Define Compression Stages ==========
        
        // Stage 1: Lorenzo predictor + quantization
        LorenzoConfig<float, uint16_t> lorenzo_config(
            1e-2,    // error_bound: 1% absolute error
            512,      // quant_radius: 512 (creates 1024 bins)
            0.2f      // outlier_capacity: 20% max outliers
        );
        
        LorenzoStage<float, uint16_t> lorenzo(lorenzo_config);
        pipeline.addStage(&lorenzo);
        
        // Stage 2: RLE on Lorenzo codes
        RLEConfig<uint16_t> rle_config;
        RLEStage<uint16_t> rle(rle_config);
        pipeline.addStage(&rle);
        
        // Stage 3: Bitpacking on RLE output
        BitpackingConfig<uint16_t> bitpack_config(1024);  // 1024 elements per block
        BitpackingStage<uint16_t> bitpack(bitpack_config);
        pipeline.addStage(&bitpack);
        
        // ========== Build Pipeline ==========
        
        size_t input_size = n * sizeof(float);
        pipeline.build(input_size);
        
        std::cout << "Pipeline Structure:\n";
        pipeline.printStructure();
        
        // Print buffer allocations
        pipeline.getDag()->printBufferAllocations();
        
        // Print memory pool stats
        std::cout << "\n";
        pipeline.printMemoryStats();
        
        // ========== Compress Data ==========
        
        std::vector<uint8_t> h_output(input_size);
        
        std::cout << "\nCompressing...\n";
        size_t compressed_size = pipeline.compressFromHost(
            h_input.data(), 
            input_size, 
            h_output.data()
        );
        
        pipeline.synchronize();
        
        // ========== Memory Pool Analysis ==========
        std::cout << "\n========== Memory Pool After Compression ==========\n";
        pipeline.printMemoryStats();
        
        std::cout << "\nMemory Reuse Analysis:\n";
        std::cout << "  Total allocated: " << pipeline.getMemoryPool()->getPeakUsage() / 1024.0 / 1024.0 << " MB\n";
        std::cout << "  Currently in use: " << pipeline.getMemoryPool()->getCurrentUsage() / 1024.0 / 1024.0 << " MB\n";
        std::cout << "  Active allocations: " << pipeline.getMemoryPool()->getAllocationCount() << "\n";
        std::cout << "\n  Note: All stage buffers are allocated from the same memory pool.\n";
        std::cout << "  The pool manages reuse automatically via CUDA's stream-ordered allocator.\n";
        std::cout << "  Since stages run sequentially, intermediate buffers could potentially\n";
        std::cout << "  be reused, but graph mode keeps all allocations live for replay.\n";
        std::cout << "====================================================\n\n";
        
        // ========== Inspect Lorenzo Output ==========
        std::cout << "\n========== Lorenzo Stage Output ==========\n";
        
        // Get Lorenzo output buffer (codes after quantization)
        void* lorenzo_output = pipeline.getStageOutput(0);  // Stage 0 = Lorenzo
        
        if (lorenzo_output) {
            if (use_graph) {
                // Graph mode: buffers are persistent, can read anytime
                std::vector<uint16_t> h_lorenzo_codes(100);
                cudaMemcpy(h_lorenzo_codes.data(), lorenzo_output, 
                          100 * sizeof(uint16_t), 
                          cudaMemcpyDeviceToHost);
                
                std::cout << "First 100 Lorenzo codes:\n";
                for (size_t i = 0; i < 100; i++) {
                    std::cout << h_lorenzo_codes[i] << " ";
                    if ((i + 1) % 20 == 0) std::cout << "\n";
                }
                std::cout << "\n";
            } else {
                // Stream mode: buffer was freed after RLE stage for memory reuse
                std::cout << "Lorenzo buffer freed in STREAM mode (memory reuse enabled).\n";
                std::cout << "Use GRAPH mode to inspect intermediate buffers.\n";
            }
        } else {
            std::cout << "Could not access Lorenzo output buffer\n";
        }
        
        std::cout << "\n========== Bitpacking Output Analysis ==========\n";
        printBitpackedData(h_output.data(), 100);
        
        std::cout << "\n========== Compression Results ==========\n";
        std::cout << "Compressed size: " << (compressed_size / 1048576.0) << " MB\n";
        std::cout << "Compression ratio: " << std::fixed << std::setprecision(2) << (static_cast<float>(input_size) / compressed_size) << "x\n";
        pipeline.printStats();
        
        pipeline.synchronize();
    }
    
    std::cout << "\nDone!" << std::endl;
    
    return 0;
}
