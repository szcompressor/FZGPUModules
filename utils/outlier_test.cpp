#include "fzmodules.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>

using namespace fz;

int main() {
    std::cout << "\n========================================\n";
    std::cout << "Outlier Pipeline Test\n";
    std::cout << "========================================\n\n";
    
    // Create synthetic data with outliers
    const size_t nx = 1000;
    const size_t ny = 1000;
    const size_t num_elements = nx * ny;
    
    std::vector<float> h_input(num_elements);
    
    // Generate smooth data with some outliers
    for (size_t y = 0; y < ny; y++) {
        for (size_t x = 0; x < nx; x++) {
            size_t idx = y * nx + x;
            // Smooth base pattern
            h_input[idx] = std::sin(x * 0.01f) * std::cos(y * 0.01f) * 100.0f;
            
            // Add outliers (~5% of data)
            if (idx % 20 == 0) {
                h_input[idx] += 500.0f;  // Large outlier
            }
        }
    }
    
    size_t input_size = num_elements * sizeof(float);
    std::cout << "Dataset: " << nx << " x " << ny << " = " << num_elements 
              << " floats (" << (input_size / 1024.0 / 1024.0) << " MB)\n\n";
    
    // Allocate host output buffer
    size_t max_output = input_size * 2;  // Conservative estimate
    std::vector<uint8_t> h_output(max_output);
    
    // ========================================
    // Test STREAM mode with outlier pipeline
    // ========================================
    
    std::cout << "========================================\n";
    std::cout << "Testing STREAM mode - Outlier Pipeline\n";
    std::cout << "========================================\n\n";
    std::cout << "Memory mode: SAFE\n\n";
    std::cout << "Full outlier pipeline with difference coding:\n";
    std::cout << "  Lorenzo (error=1e-3, 20% outliers)\n";
    std::cout << "    ├─> codes (uint16) -> Bitpack\n";
    std::cout << "    ├─> indices (uint32) -> Difference -> Bitpack  } Parallel\n";
    std::cout << "    ├─> errors (float) -> raw\n";
    std::cout << "    └─> Concatenate all 3 outputs\n\n";
    std::cout << "NOTE: Difference coding converts sparse indices to small deltas\n";
    std::cout << "      that compress much better with bitpacking.\n\n";
    std::cout << "NOTE: GRAPH mode is not compatible with WSL due to CUDA memory\n";
    std::cout << "      pool limitations. STREAM mode provides robust execution.\n\n";
    
    // Create pipeline config
    PipelineConfig config;
    config.memory_mode = MemoryMode::SAFE;
    config.use_cuda_graph = false;  // STREAM mode (GRAPH not compatible with WSL)
    config.enable_profiling = true;
    
    Pipeline pipeline(config);
    
    // Lorenzo predictor with outlier handling
    LorenzoConfig<float, uint16_t> lorenzo_cfg(
        1e-3f,    // error_bound
        512,      // radius (1024 quantization bins)
        0.2f      // outlier_capacity: 20% max outliers
    );
    auto* lorenzo = new LorenzoStage<float, uint16_t>(lorenzo_cfg);
    
    // Bitpacking for codes (primary output from Lorenzo)
    BitpackingConfig<uint16_t> bitpack_codes_cfg;
    bitpack_codes_cfg.block_size = 1024;
    auto* bitpack_codes = new BitpackingStage<uint16_t>(bitpack_codes_cfg);
    
    // Difference encoding for indices (converts sparse indices to small deltas)
    DifferenceConfig<uint32_t> diff_cfg;
    auto* diff_indices = new DifferenceStage<uint32_t>(diff_cfg);
    
    // Bitpacking for difference-coded indices
    BitpackingConfig<uint32_t> bitpack_indices_cfg;
    bitpack_indices_cfg.block_size = 512;
    auto* bitpack_indices = new BitpackingStage<uint32_t>(bitpack_indices_cfg);
    
    // Concatenation stage to merge all outputs
    auto* concat = new ConcatenationStage();
    
    // Build the DAG with branching
    // Lorenzo outputs: primary=codes, aux0=errors, aux1=indices, aux2=count
    auto* lorenzo_node = pipeline.addStage(lorenzo);
    auto* bitpack_codes_node = pipeline.addStage(bitpack_codes);
    auto* diff_indices_node = pipeline.addStage(diff_indices);
    auto* bitpack_indices_node = pipeline.addStage(bitpack_indices);
    auto* concat_node = pipeline.addStage(concat);
    
    // Set up dependencies:
    // - bitpack_codes depends on Lorenzo's primary output (codes)
    pipeline.addDependency(bitpack_codes_node, lorenzo_node, -1);  // -1 = primary output
    
    // - diff_indices depends on Lorenzo's aux1 (raw indices)
    pipeline.addDependency(diff_indices_node, lorenzo_node, 1);  // aux1 = indices
    
    // - bitpack_indices depends on diff_indices output
    pipeline.addDependency(bitpack_indices_node, diff_indices_node, -1);
    
    // - concat depends on all three outputs
    pipeline.addDependency(concat_node, bitpack_codes_node, -1);     // compressed codes
    pipeline.addDependency(concat_node, bitpack_indices_node, -1);   // compressed indices
    pipeline.addDependency(concat_node, lorenzo_node, 0);            // aux0 = raw errors
    
    // Build pipeline
    std::cout << "Building pipeline with input size: " << input_size << " bytes\n" << std::endl;
    pipeline.build(input_size);
    pipeline.printStructure();
    pipeline.printMemoryStats();
    
    // Compress
    std::cout << "\nCompressing..." << std::endl;
    size_t compressed_size = pipeline.compressFromHost(h_input.data(), input_size, h_output.data());
    pipeline.synchronize();
    
    std::cout << "\n========== Compression Results ==========\n";
    std::cout << "Original size:     " << std::fixed << std::setprecision(2) 
              << (input_size / 1024.0 / 1024.0) << " MB\n";
    std::cout << "Compressed size:   " << (compressed_size / 1024.0 / 1024.0) << " MB\n";
    std::cout << "Compression ratio: " << std::setprecision(2) 
              << (static_cast<float>(input_size) / compressed_size) << "x\n";
    std::cout << "=========================================\n\n";
    
    // Print DAG stats
    pipeline.printStats();
    
    std::cout << "\n✓ Full outlier pipeline completed successfully!\n";
    
    return 0;
}
