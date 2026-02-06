#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <fstream>

#include "pipeline/compressor.h"
#include "predictors/lorenzo/lorenzo.h"
#include "encoding/run_length_encoding/RLE.h"
#include "encoding/bitpacking/bitpacking.h"

using namespace fz;

int main() {
  std::cout << "\n========================================" << std::endl;
  std::cout << "CESM-ATM CLDHGH Compression Test" << std::endl;
  std::cout << "========================================\n" << std::endl;
  
  const char* filename = "/home/skyler/data/SDRB/CESM_ATM_1800x3600/CLDHGH.f32";
  const int width = 3600;
  const int height = 1800;
  const int n1 = width * height;
  
  // Read binary file
  std::ifstream file(filename, std::ios::binary);
  if (!file) {
    std::cout << "✗ Could not open file: " << filename << std::endl;
    return 1;
  }
  
  float* h_input1 = new float[n1];
  file.read(reinterpret_cast<char*>(h_input1), n1 * sizeof(float));
  file.close();
  
  if (!file.good() && !file.eof()) {
    std::cout << "✗ Error reading file" << std::endl;
    delete[] h_input1;
    return 1;
  }
  
  std::cout << "Dataset: " << width << " x " << height << " = " << n1 << " floats (" 
            << std::fixed << std::setprecision(2) << (n1 * sizeof(float)) / 1048576.0f << " MB)\n" << std::endl;
      
  size_t input_size = n1 * sizeof(float);
  uint8_t* h_output = new uint8_t[input_size];
  
  // Test both modes
  for (int mode = 0; mode < 2; mode++) {
    bool use_graph = (mode == 1);
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "Testing " << (use_graph ? "GRAPH" : "STREAM") << " mode" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    // Setup compression pipeline
    PipelineConfig config;
    config.use_cuda_graph = use_graph;
    config.use_parallel_streams = true;
    config.max_streams = 4;
    config.memory_pool_multiplier = 3.0f;
    config.enable_profiling = true;
    config.memory_mode = MemoryMode::SAFE;
    
    std::cout << "Memory mode: SAFE\n" << std::endl;
    
    Pipeline pipeline(config);
    
    // Configure Lorenzo stage
    LorenzoConfig<float, uint16_t> lorenzo_config(
        1e-3f,    // error_bound: 0.001 absolute error
        512,      // quant_radius: 512 (1024 bins)
        0.2f      // outlier_capacity: 20% max
    );
    
    LorenzoStage<float, uint16_t> lorenzo(lorenzo_config);
    pipeline.addStage(&lorenzo);
    
    // Add RLE stage on Lorenzo codes
    RLEConfig<uint16_t> rle_config;
    RLEStage<uint16_t> rle(rle_config);
    pipeline.addStage(&rle);
    
    // Add Bitpacking stage to compress RLE output
    BitpackingConfig<uint16_t> bitpack_config(1024);  // 1024 elements per block
    BitpackingStage<uint16_t> bitpack(bitpack_config);
    pipeline.addStage(&bitpack);
    
    std::cout << "Pipeline: Lorenzo (1e-3 error) -> RLE -> Bitpacking\n" << std::endl;
    
    // Build pipeline
    pipeline.build(input_size);
    
    std::cout << "Pipeline Structure:" << std::endl;
    pipeline.printStructure();
    std::cout << std::endl;
    pipeline.printMemoryStats();
    
    // Compress
    std::cout << "\nCompressing..." << std::endl;
    size_t compressed_size = pipeline.compressFromHost(h_input1, input_size, h_output);
    pipeline.synchronize();
    
    // Run a second time to test reuse/replay
    std::cout << "Running second compression..." << std::endl;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    size_t compressed_size2 = pipeline.compressFromHost(h_input1, input_size, h_output);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time_ms = 0;
    cudaEventElapsedTime(&time_ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    // Memory analysis
    std::cout << "\n========== Memory Pool After Compression ==========" << std::endl;
    pipeline.printMemoryStats();
    
    std::cout << "\nMemory Analysis:" << std::endl;
    std::cout << "  Peak usage:        " << std::fixed << std::setprecision(2) 
              << pipeline.getMemoryPool()->getPeakUsage() / 1048576.0 << " MB" << std::endl;
    std::cout << "  Current usage:     " 
              << pipeline.getMemoryPool()->getCurrentUsage() / 1048576.0 << " MB" << std::endl;
    std::cout << "  Active allocations: " 
              << pipeline.getMemoryPool()->getAllocationCount() << std::endl;
    std::cout << "  Memory freed:      " 
              << (pipeline.getMemoryPool()->getPeakUsage() - pipeline.getMemoryPool()->getCurrentUsage()) / 1048576.0 
              << " MB (" << std::setprecision(0)
              << (1.0 - (double)pipeline.getMemoryPool()->getCurrentUsage() / pipeline.getMemoryPool()->getPeakUsage()) * 100.0
              << "%)" << std::endl;
    
    // Compression results
    std::cout << "\n========== Compression Results ==========" << std::endl;
    std::cout << "Original size:     " << std::fixed << std::setprecision(2) 
              << (input_size / 1048576.0) << " MB" << std::endl;
    std::cout << "Compressed size:   " << (compressed_size / 1048576.0) << " MB" << std::endl;
    std::cout << "Compression ratio: " << ((float)input_size / compressed_size) << "x" << std::endl;
    std::cout << "Second run time:   " << std::setprecision(3) << time_ms << " ms" << std::endl;
    std::cout << "Throughput:        " << std::setprecision(2) 
              << ((input_size / 1048576.0) / (time_ms / 1000.0) / 1024.0) << " GB/s" << std::endl;
    
    pipeline.printStats();
  }
  
  delete[] h_output;
  delete[] h_input1;
  
  std::cout << "\n✓ All tests completed successfully.\n" << std::endl;
  
  return 0;
}