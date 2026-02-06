#include "compressor.h"
#include <iostream>
#include <iomanip>
#include <stdexcept>

namespace fz {

// ========== Pipeline Implementation ==========

Pipeline::Pipeline(const PipelineConfig& config)
    : config_(config),
      stream_(nullptr),
      d_input_buffer_(nullptr),
      d_output_buffer_(nullptr),
      d_buffer_size_(0),
      built_(false),
      input_size_(0),
      last_input_size_(0),
      last_compressed_size_(0) {
    
    initializeDevice();
    
    // Create CUDA stream
    cudaStreamCreate(&stream_);
}

Pipeline::~Pipeline() {
    cleanupDevice();
    
    if (stream_) {
        cudaStreamDestroy(stream_);
    }
}

void Pipeline::initializeDevice() {
    cudaSetDevice(config_.device_id);
    
    // Optionally warm up device
    cudaFree(0);
}

void Pipeline::cleanupDevice() {
    if (d_input_buffer_) {
        cudaFree(d_input_buffer_);
        d_input_buffer_ = nullptr;
    }
    if (d_output_buffer_) {
        cudaFree(d_output_buffer_);
        d_output_buffer_ = nullptr;
    }
}

DagNode* Pipeline::addStage(Stage* stage) {
    if (!stage) {
        throw std::runtime_error("Cannot add null stage to pipeline");
    }
    
    // Just store the stage - create DAG in build() when we have memory pool
    if (!dag_) {
        pending_stages_.push_back(stage);
        // Return a pseudo-pointer using the index (shifted to avoid null)
        // This allows dependencies to be tracked before build()
        return reinterpret_cast<DagNode*>(pending_stages_.size());
    }
    
    return dag_->addStage(stage);
}

void Pipeline::addDependency(DagNode* dependent, DagNode* dependency) {
    addDependency(dependent, dependency, -1);  // -1 means use primary output
}

void Pipeline::addDependency(DagNode* dependent, DagNode* dependency, int aux_index) {
    if (!dependent || !dependency) {
        throw std::runtime_error("Cannot add null dependency");
    }
    
    if (!dag_) {
        // Extract indices from pseudo-pointers
        size_t dep_idx = reinterpret_cast<size_t>(dependent) - 1;
        size_t dependency_idx = reinterpret_cast<size_t>(dependency) - 1;
        pending_dependencies_.push_back({dep_idx, dependency_idx, aux_index});
        return;
    }
    dag_->addDependency(dependent, dependency, aux_index);
}

void Pipeline::build(size_t input_size) {
    if (built_) {
        std::cout << "Pipeline already built - rebuilding..." << std::endl;
        reset();
    }
    
    input_size_ = input_size;
    
    // Create memory pool based on input size
    MemoryPoolConfig pool_config(
        input_size,
        config_.memory_pool_multiplier,
        config_.device_id,
        true  // enable reuse
    );
    memory_pool_ = std::make_unique<MemoryPool>(pool_config);
    
    // Create DAG with the memory pool
    DagConfig dag_config;
    dag_config.use_cuda_graph = config_.use_cuda_graph;
    dag_config.parallel_streams = config_.use_parallel_streams;
    dag_config.max_streams = config_.max_streams;
    dag_config.enable_profiling = config_.enable_profiling;
    dag_config.memory_mode = config_.memory_mode;
    dag_config.conservative_alloc_factor = config_.conservative_alloc_factor;
    
    dag_ = std::make_unique<Dag>(memory_pool_.get(), dag_config);
    
    // Add all pending stages
    std::vector<DagNode*> nodes;
    for (auto* stage : pending_stages_) {
        nodes.push_back(dag_->addStage(stage));
    }
    
    // For a linear pipeline (no explicit dependencies), auto-create sequential dependencies
    if (pending_dependencies_.empty() && nodes.size() > 1) {
        for (size_t i = 1; i < nodes.size(); i++) {
            dag_->addDependency(nodes[i], nodes[i-1]);
        }
    }
    
    // Add pending dependencies using resolved node pointers
    for (auto& dep : pending_dependencies_) {
        if (dep.dependent_idx >= nodes.size() || dep.dependency_idx >= nodes.size()) {
            throw std::runtime_error("Invalid dependency index");
        }
        dag_->addDependency(nodes[dep.dependent_idx], nodes[dep.dependency_idx], dep.aux_index);
    }
    
    pending_stages_.clear();
    pending_dependencies_.clear();
    
    // Build DAG
    dag_->build(input_size);
    
    built_ = true;
    
    std::cout << "Pipeline built for input size: " 
              << (input_size / (1024.0 * 1024.0)) << " MB" << std::endl;
    std::cout << "Memory pool size: " 
              << (memory_pool_->getMemPool() ? 
                  pool_config.getPoolSize() / (1024.0 * 1024.0) : 0.0) << " MB" 
              << std::endl;
}

size_t Pipeline::compress(void* input, size_t input_size, void* output) {
    if (!built_) {
        throw std::runtime_error("Pipeline not built - call build() first");
    }
    
    if (input_size > input_size_) {
        throw std::runtime_error("Input size exceeds pipeline capacity - rebuild with larger size");
    }
    
    // Execute DAG
    size_t compressed_size = dag_->execute(input, input_size, output, stream_);
    
    // Update stats
    last_input_size_ = input_size;
    last_compressed_size_ = compressed_size;
    
    return compressed_size;
}

size_t Pipeline::compressFromHost(const void* h_input, size_t input_size, void* h_output) {
    printf("[Pipeline] compressFromHost called\n");
    
    if (!built_) {
        throw std::runtime_error("Pipeline not built - call build() first");
    }
    
    printf("[Pipeline] Allocating device buffers if needed\n");
    
    // Allocate device buffers if needed
    if (d_buffer_size_ < input_size) {
        allocateDeviceBuffers(input_size);
    }
    
    printf("[Pipeline] Copying input to device\n");
    
    // Copy input to device
    cudaMemcpyAsync(d_input_buffer_, h_input, input_size, 
                    cudaMemcpyHostToDevice, stream_);
    
    printf("[Pipeline] Calling compress\n");
    
    // Compress on device
    size_t compressed_size = compress(d_input_buffer_, input_size, d_output_buffer_);
    
    // Check if compression failed (buffer overflow or other error)
    if (compressed_size == 0) {
        printf("ERROR: Compression failed (likely buffer overflow in CONSERVATIVE mode)\n");
        return 0;
    }
    
    // Copy output back to host
    cudaError_t err = cudaMemcpy(h_output, d_output_buffer_, compressed_size,
                                 cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("ERROR: D2H copy failed: %s\n", cudaGetErrorString(err));
        return 0;
    }
    
    return compressed_size;
}

size_t Pipeline::decompress(void* input, size_t input_size, void* output) {
    // TODO: Implement decompression
    // Would execute stages in reverse order with inverse operations
    throw std::runtime_error("Decompression not yet implemented");
}

void Pipeline::reset() {
    if (dag_) {
        dag_->reset();
    }
    if (memory_pool_) {
        memory_pool_->reset(stream_);
    }
}

DagNode* Pipeline::getNode(size_t index) {
    if (!dag_) return nullptr;
    return dag_->getNode(index);
}

void Pipeline::synchronize() {
    cudaStreamSynchronize(stream_);
}

void Pipeline::allocateDeviceBuffers(size_t size) {
    // Free old buffers
    if (d_input_buffer_) cudaFree(d_input_buffer_);
    if (d_output_buffer_) cudaFree(d_output_buffer_);
    
    // Allocate new buffers (input size + output size)
    // Output buffer = input size (worst case: incompressible data)
    cudaMalloc(&d_input_buffer_, size);
    cudaMalloc(&d_output_buffer_, size);
    
    d_buffer_size_ = size;
}

void Pipeline::printStructure() const {
    if (dag_) {
        dag_->printStructure();
    } else {
        std::cout << "Pipeline: No stages added yet" << std::endl;
    }
}

void Pipeline::printStats() const {
    std::cout << "\n========== Pipeline Statistics ==========\n";
    std::cout << "Last input size: " << (last_input_size_ / 1048576.0) << " MB\n";
    std::cout << "Last compressed size: " << (last_compressed_size_ / 1048576.0) << " MB\n";
    std::cout << "Compression ratio: " << std::fixed << std::setprecision(2) << getCompressionRatio() << "x\n";
    
    // Calculate total pipeline time and throughput
    if (dag_) {
        float total_time_ms = 0.0f;
        for (auto* node : dag_->getNodes()) {
            total_time_ms += node->stage->getLastExecutionTime();
        }
        
        if (total_time_ms > 0.0f) {
            // Pipeline throughput based on total input processed
            double total_bytes = static_cast<double>(last_input_size_ + last_compressed_size_);
            float pipeline_throughput = static_cast<float>(total_bytes / (total_time_ms * 1e6));
            
            std::cout << "Total execution time: " << total_time_ms << " ms\n";
            std::cout << "Pipeline throughput: " << pipeline_throughput << " GB/s\n";
        }
    }
    
    std::cout << "=========================================\n" << std::endl;
    
    if (dag_) {
        dag_->printStats();
    }
}

void Pipeline::printMemoryStats() const {
    if (memory_pool_) {
        memory_pool_->printStats();
    } else {
        std::cout << "Memory pool not initialized yet" << std::endl;
    }
}

float Pipeline::getCompressionRatio() const {
    if (last_compressed_size_ == 0) return 0.0f;
    return static_cast<float>(last_input_size_) / static_cast<float>(last_compressed_size_);
}

void* Pipeline::getStageOutput(size_t stage_index) {
    if (!dag_) return nullptr;
    
    const auto& nodes = dag_->getNodes();
    if (stage_index >= nodes.size()) return nullptr;
    
    return nodes[stage_index]->output_buffer;
}

void* Pipeline::getStageAuxOutput(size_t stage_index) {
    if (!dag_) return nullptr;
    
    const auto& nodes = dag_->getNodes();
    if (stage_index >= nodes.size()) return nullptr;
    
    // Return first auxiliary buffer if it exists
    if (!nodes[stage_index]->aux_buffers.empty()) {
        return nodes[stage_index]->aux_buffers[0];
    }
    
    return nullptr;
}

} // namespace fz
