#include "dag.h"
#include "mem/memory_pool.h"
#include "compressor.h"  // For MemoryMode enum
#include "encoding/concatenation/concat.h"  // For ConcatenationStage detection
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <queue>
#include <unordered_set>
#include <set>

namespace fz {

// ========== DagConfig Implementation ==========

DagConfig::DagConfig()
    : use_cuda_graph(true),
      parallel_streams(true),
      max_streams(4),
      enable_profiling(false),
      memory_mode(MemoryMode::SAFE),
      conservative_alloc_factor(0.5f) {
}

// ========== Dag Implementation ==========

Dag::Dag(MemoryPool* memory_pool, const DagConfig& config)
    : memory_pool_(memory_pool),
      config_(config),
      cuda_graph_(nullptr),
      cuda_graph_exec_(nullptr),
      graph_built_(false) {
    
    if (!memory_pool) {
        throw std::runtime_error("MemoryPool cannot be null");
    }
    
    // Create streams if parallel execution enabled
    if (config_.parallel_streams) {
        for (int i = 0; i < config_.max_streams; i++) {
            cudaStream_t stream;
            cudaStreamCreate(&stream);
            streams_.push_back(stream);
        }
    }
}

Dag::~Dag() {
    // Clean up nodes
    for (auto* node : nodes_) {
        delete node;
    }
    
    // Destroy CUDA graph
    if (cuda_graph_exec_) {
        cudaGraphExecDestroy(cuda_graph_exec_);
    }
    if (cuda_graph_) {
        cudaGraphDestroy(cuda_graph_);
    }
    
    // Destroy streams
    for (auto stream : streams_) {
        cudaStreamDestroy(stream);
    }
}

DagNode* Dag::addStage(Stage* stage) {
    if (!stage) {
        throw std::runtime_error("Cannot add null stage to DAG");
    }
    
    auto* node = new DagNode(stage);
    nodes_.push_back(node);
    return node;
}

void Dag::addDependency(DagNode* dependent, DagNode* dependency) {
    addDependency(dependent, dependency, -1);  // -1 means use primary output
}

void Dag::addDependency(DagNode* dependent, DagNode* dependency, int aux_index) {
    if (!dependent || !dependency) {
        throw std::runtime_error("Cannot add null dependency");
    }
    
    dependent->dependencies.push_back(dependency);
    dependent->dependency_aux_indices.push_back(aux_index);
    dependency->dependents.push_back(dependent);
}

void Dag::build(size_t input_size) {    
    if (config_.enable_profiling) {
        for (auto* node : nodes_) {
            node->stage->enableProfiling(true);
        }
    }
    
    // 1. Assign parallelism levels (based on dependencies)
    assignLevels();
    
    // 2. Assign streams for parallel execution
    if (config_.parallel_streams) {
        assignStreams();
    }
    
    // 3. Allocate buffers
    allocateBuffers(input_size);
    
    // 4. Build CUDA graph if enabled
    if (config_.use_cuda_graph) {
        // Graph building requires a dummy execution to capture
        // We'll do this on first execute() call
        graph_built_ = false;
    }
    
    std::cout << "DAG built successfully: " << nodes_.size() << " stages" << std::endl;
}

void Dag::assignLevels() {
    // Assign levels for parallel execution based on dependencies
    // Level 0 = no dependencies, Level N = max(dep levels) + 1
    
    for (size_t i = 0; i < nodes_.size(); i++) {
        auto* node = nodes_[i];
        node->execution_order = static_cast<int>(i);
        
        if (node->dependencies.empty()) {
            node->level = 0;
        } else {
            int max_level = -1;
            for (auto* dep : node->dependencies) {
                max_level = std::max(max_level, dep->level);
            }
            node->level = max_level + 1;
        }
    }
}

void Dag::assignStreams() {
    if (streams_.empty()) return;
    
    // Assign streams based on levels to maximize parallelism
    // Nodes at the same level can run in parallel on different streams
    
    std::unordered_map<int, int> level_stream_counter;
    
    for (auto* node : nodes_) {
        // Round-robin assignment within each level
        int stream_idx = level_stream_counter[node->level] % streams_.size();
        node->stream = streams_[stream_idx];
        level_stream_counter[node->level]++;
    }
}

void Dag::allocateBuffers(size_t input_size) {
    // Allocate buffers for each stage based on memory requirements
    
    cudaStream_t alloc_stream = streams_.empty() ? 0 : streams_[0];
    
    // Choose allocation method based on mode
    bool use_graph_alloc = config_.use_cuda_graph;
    
    std::cout << "Allocating buffers in " 
              << (use_graph_alloc ? "GRAPH" : "STREAM") 
              << " mode...\n";
    
    for (auto* node : nodes_) {
        // Get memory requirements from stage
        auto req = node->stage->getMemoryRequirements(input_size);
        
        // Apply memory mode: conservative uses configurable factor for compressors
        float alloc_factor = 1.0f;
        if (config_.memory_mode == MemoryMode::CONSERVATIVE) {
            // For encoding stages, use the configured allocation factor
            if (node->stage->getName().find("RLE") != std::string::npos ||
                node->stage->getName().find("Bitpack") != std::string::npos) {
                alloc_factor = config_.conservative_alloc_factor;
                std::cout << "  [" << node->stage->getName() << "] Using CONSERVATIVE mode (" 
                         << std::fixed << std::setprecision(0) << (alloc_factor * 100) 
                         << "% allocation)\n";
            }
        }
        
        // Allocate primary output
        if (req.output_size > 0) {
            std::string tag = node->stage->getName() + "_output";
            size_t alloc_size = static_cast<size_t>(req.output_size * alloc_factor);
            
            if (use_graph_alloc) {
                // Graph mode: persistent allocations
                node->output_buffer = memory_pool_->allocateGraphMemory(
                    alloc_size, alloc_stream, tag);
            } else {
                // Stream mode: allow reuse
                node->output_buffer = memory_pool_->allocate(
                    alloc_size, alloc_stream, tag);
            }
            
            if (!node->output_buffer) {
                throw std::runtime_error("Failed to allocate output buffer for " + 
                                       node->stage->getName());
            }
            
            node->allocated_output_size = alloc_size;
        }
        
        // Allocate auxiliary outputs (for multi-output stages or stages needing temp buffers)
        // Even non-MultiOutputStage stages may need aux buffers for graph mode
        if (req.aux_output_size > 0) {
            // Check if it's a MultiOutputStage first
            if (auto* multi_stage = dynamic_cast<MultiOutputStage*>(node->stage)) {
                size_t num_aux = multi_stage->getNumAuxiliaryOutputs();
                node->aux_buffers.resize(num_aux);
                node->allocated_aux_sizes.resize(num_aux);
                
                // Split aux_output_size equally
                if (num_aux > 0) {
                    size_t per_aux = req.aux_output_size / num_aux;
                    for (size_t i = 0; i < num_aux; i++) {
                        std::string tag = node->stage->getName() + "_aux" + std::to_string(i);
                        if (use_graph_alloc) {
                            node->aux_buffers[i] = memory_pool_->allocateGraphMemory(
                                per_aux, alloc_stream, tag);
                        } else {
                            node->aux_buffers[i] = memory_pool_->allocate(
                                per_aux, alloc_stream, tag);
                        }
                        node->allocated_aux_sizes[i] = per_aux;
                    }
                }
            } else {
                // Regular stage needing temp buffers for graph mode
                if (node->stage->getName().find("RLE") == 0) {
                    // Simplified RLE: just needs counter buffer
                    node->aux_buffers.resize(1);
                    node->allocated_aux_sizes.resize(1);
                    std::string tag = node->stage->getName() + "_counter";
                    if (use_graph_alloc) {
                        node->aux_buffers[0] = memory_pool_->allocateGraphMemory(
                            sizeof(uint32_t), alloc_stream, tag);
                    } else {
                        node->aux_buffers[0] = memory_pool_->allocate(
                            sizeof(uint32_t), alloc_stream, tag);
                    }
                    node->allocated_aux_sizes[0] = sizeof(uint32_t);
                } else {
                    // Generic: split equally into 3 buffers
                    node->aux_buffers.resize(3);
                    node->allocated_aux_sizes.resize(3);
                    size_t per_aux = req.aux_output_size / 3;
                    for (size_t i = 0; i < 3; i++) {
                        std::string tag = node->stage->getName() + "_aux" + std::to_string(i);
                        if (use_graph_alloc) {
                            node->aux_buffers[i] = memory_pool_->allocateGraphMemory(
                                per_aux, alloc_stream, tag);
                        } else {
                            node->aux_buffers[i] = memory_pool_->allocate(
                                per_aux, alloc_stream, tag);
                        }
                        node->allocated_aux_sizes[i] = per_aux;
                    }
                }
            }
        }
        
        // Update input_size for next stage (cascade through pipeline)
        input_size = req.output_size;
    }
}

size_t Dag::execute(void* input, size_t input_size, void* output, cudaStream_t stream) {
    printf("[DAG] execute called: use_cuda_graph=%d, graph_built=%d\n", 
           config_.use_cuda_graph, graph_built_);
           
    // If CUDA graph is enabled and already built, use graph execution
    if (config_.use_cuda_graph && graph_built_ && cuda_graph_exec_) {
        printf("[DAG] Using pre-built graph\n");
        // Fast path: just launch the pre-built graph
        cudaGraphLaunch(cuda_graph_exec_, stream);
        cudaStreamSynchronize(stream);
        
        // Update actual output sizes for variable-output stages
        size_t current_size = input_size;
        for (auto* node : nodes_) {
            size_t input_bytes = current_size;
            
            // Check if this is RLE stage - read actual num_runs
            if (node->stage->getName().find("RLE") == 0 && !node->aux_buffers.empty()) {
                uint32_t num_runs;
                cudaMemcpyAsync(&num_runs, node->aux_buffers[0], sizeof(uint32_t),
                               cudaMemcpyDeviceToHost, stream);
                cudaStreamSynchronize(stream);
                
                // Calculate actual output size
                size_t actual_size = sizeof(uint32_t) + num_runs * (sizeof(uint16_t) + sizeof(uint32_t));
                node->output_size = actual_size;
                current_size = actual_size;
            } else {
                current_size = node->output_size;
            }
            
            node->stage->updateThroughputStats(input_bytes, node->output_size);
        }
        
        // Return final output size from last node
        if (!nodes_.empty()) {
            return nodes_.back()->output_size;
        }
        return 0;
    }
    
    // Build CUDA graph manually using addToGraph if enabled
    if (config_.use_cuda_graph && !graph_built_) {
        printf("[DAG] Building CUDA graph with %zu stages\n", nodes_.size());
        
        cudaGraph_t new_graph;
        cudaGraphCreate(&new_graph, 0);
        cudaError_t err = cudaSuccess;
        
        // Add each stage to the graph in execution order
        for (size_t i = 0; i < nodes_.size(); i++) {
            auto* node = nodes_[i];
            
            // Collect graph dependencies from DAG dependencies
            std::vector<cudaGraphNode_t> dep_graph_nodes;
            for (auto* dep : node->dependencies) {
                if (dep->graph_node) {
                    dep_graph_nodes.push_back(dep->graph_node);
                }
            }
            
            // Determine input for this stage
            void* stage_input = input;
            size_t stage_input_size = input_size;
            
            if (!node->dependencies.empty()) {
                // Use first dependency's output as input
                auto* dep = node->dependencies[0];
                int aux_idx = node->dependency_aux_indices[0];
                
                if (aux_idx == -1) {
                    stage_input = dep->output_buffer;
                    stage_input_size = dep->output_size;
                } else {
                    if (aux_idx >= 0 && aux_idx < (int)dep->aux_buffers.size()) {
                        stage_input = dep->aux_buffers[aux_idx];
                        stage_input_size = dep->aux_sizes[aux_idx];
                    }
                }
            }
            
            // Calculate output size for this stage
            node->output_size = node->stage->getMaxOutputSize(stage_input_size);
            
            // Special handling for concatenation stages
            if (auto* concat_stage = dynamic_cast<ConcatenationStage*>(node->stage)) {
                // Gather all dependency outputs and sizes
                std::vector<void*> dep_outputs;
                std::vector<size_t> dep_sizes;
                
                for (size_t dep_idx = 0; dep_idx < node->dependencies.size(); dep_idx++) {
                    auto* dep = node->dependencies[dep_idx];
                    int aux_idx = node->dependency_aux_indices[dep_idx];
                    
                    if (aux_idx == -1) {
                        dep_outputs.push_back(dep->output_buffer);
                        dep_sizes.push_back(dep->output_size);
                    } else {
                        if (aux_idx >= 0 && aux_idx < (int)dep->aux_buffers.size()) {
                            dep_outputs.push_back(dep->aux_buffers[aux_idx]);
                            dep_sizes.push_back(dep->aux_sizes[aux_idx]);
                        }
                    }
                }
                
                // Set dependency info before adding to graph
                concat_stage->setDependencyInfo(dep_outputs, dep_sizes);
            }
            
            // Use stage's addToGraph to create kernel/memcpy nodes
            node->graph_node = node->stage->addToGraph(
                new_graph,
                dep_graph_nodes.data(),
                dep_graph_nodes.size(),
                stage_input,
                stage_input_size,
                node->output_buffer,
                node->aux_buffers,
                stream
            );
            
            if (!node->graph_node) {
                fprintf(stderr, "Failed to add %s to graph\n", node->stage->getName().c_str());
                cudaGraphDestroy(new_graph);
                // Fall back to stream execution
                goto stream_execution;
            }
        }
        
        // Instantiate the graph
        cudaGraphInstantiate(&cuda_graph_exec_, new_graph, nullptr, nullptr, 0);
        cuda_graph_ = new_graph;
        graph_built_ = true;
        
        std::cout << "CUDA graph built manually (" << nodes_.size() << " stages)" << std::endl;
        
        // Launch the newly built graph
        cudaGraphLaunch(cuda_graph_exec_, stream);
        cudaStreamSynchronize(stream);
        
        // Update actual output sizes for variable-output stages (like RLE, Bitpacking)
        // Read back size information from output buffers to get real sizes
        size_t current_size = input_size;  // Reset for stats tracking
        for (auto* node : nodes_) {
            size_t input_bytes = current_size;
            
            // Check if this is RLE stage - it has actual size in aux_buffers[0]
            if (node->stage->getName().find("RLE") == 0 && !node->aux_buffers.empty()) {
                // RLE stores num_runs in aux_buffers[0]
                uint32_t num_runs;
                cudaMemcpyAsync(&num_runs, node->aux_buffers[0], sizeof(uint32_t),
                               cudaMemcpyDeviceToHost, stream);
                cudaStreamSynchronize(stream);
                
                // Calculate actual output size: header + (value,count) pairs
                // For uint16_t: 4 bytes + num_runs * (2 + 4) bytes
                size_t actual_size = sizeof(uint32_t) + num_runs * (sizeof(uint16_t) + sizeof(uint32_t));
                node->output_size = actual_size;
                current_size = actual_size;
            } else if (node->stage->getName().find("Bitpack") == 0) {
                // Bitpacking stores num_blocks in first 4 bytes, then headers, then data
                // We need to read the actual compressed size from the output
                uint32_t num_blocks;
                cudaMemcpyAsync(&num_blocks, node->output_buffer, sizeof(uint32_t),
                               cudaMemcpyDeviceToHost, stream);
                cudaStreamSynchronize(stream);
                
                // Read all block headers to calculate total compressed size
                struct BlockHeader {
                    uint16_t min_value;
                    uint8_t num_bits;
                    uint8_t reserved;
                };
                
                std::vector<BlockHeader> headers(num_blocks);
                cudaMemcpyAsync(headers.data(), 
                               (uint8_t*)node->output_buffer + sizeof(uint32_t),
                               num_blocks * sizeof(BlockHeader),
                               cudaMemcpyDeviceToHost, stream);
                cudaStreamSynchronize(stream);
                
                // Calculate total bits needed
                size_t total_bits = 0;
                size_t block_size = 1024;  // TODO: get from config
                size_t num_elements = current_size / sizeof(uint16_t);
                for (uint32_t i = 0; i < num_blocks; i++) {
                    size_t elements_in_block = (i == num_blocks - 1) ? 
                        (num_elements - i * block_size) : block_size;
                    total_bits += elements_in_block * headers[i].num_bits;
                }
                
                size_t packed_bytes = (total_bits + 7) / 8;
                size_t actual_size = sizeof(uint32_t) + num_blocks * sizeof(BlockHeader) + packed_bytes;
                node->output_size = actual_size;
                current_size = actual_size;
            } else {
                current_size = node->output_size;
            }
            
            node->stage->updateThroughputStats(input_bytes, node->output_size);
        }
        
        // Now copy final stage output to caller's output buffer with correct size
        auto* last_node = nodes_.back();
        cudaMemcpyAsync(output, last_node->output_buffer, last_node->output_size,
                       cudaMemcpyDeviceToDevice, stream);
        cudaStreamSynchronize(stream);
        
        return nodes_.back()->output_size;
    }
    
stream_execution:
    // Stream-based execution fallback
    void* current_input = input;
    size_t current_size = input_size;
    
    // Re-allocate buffers that were freed during previous compression
    for (auto* node : nodes_) {
        if (node->freed) {
            // Re-allocate output buffer
            if (node->allocated_output_size > 0 && !node->output_buffer) {
                node->output_buffer = memory_pool_->allocate(
                    node->allocated_output_size,
                    stream,
                    node->stage->getName() + std::string("_output")
                );
            }
            // Re-allocate aux buffers
            for (size_t j = 0; j < node->aux_buffers.size(); j++) {
                if (node->allocated_aux_sizes[j] > 0 && !node->aux_buffers[j]) {
                    node->aux_buffers[j] = memory_pool_->allocate(
                        node->allocated_aux_sizes[j],
                        stream,
                        node->stage->getName() + std::string("_aux") + std::to_string(j)
                    );
                }
            }
            node->freed = false;
        }
    }
    
    for (size_t i = 0; i < nodes_.size(); i++) {
        auto* node = nodes_[i];
        cudaStream_t exec_stream = node->stream ? node->stream : stream;
        
        // Determine input for this stage based on dependencies
        void* stage_input = current_input;
        size_t stage_input_size = current_size;
        
        if (!node->dependencies.empty()) {
            // Use the first dependency's specified output
            auto* dep = node->dependencies[0];
            int aux_idx = node->dependency_aux_indices[0];
            
            if (aux_idx == -1) {
                // Use primary output
                stage_input = dep->output_buffer;
                stage_input_size = dep->output_size;
            } else {
                // Use auxiliary output
                if (aux_idx >= 0 && aux_idx < (int)dep->aux_buffers.size()) {
                    stage_input = dep->aux_buffers[aux_idx];
                    stage_input_size = dep->aux_sizes[aux_idx];
                } else {
                    throw std::runtime_error("Invalid aux output index " + std::to_string(aux_idx) +
                                           " for stage " + dep->stage->getName());
                }
            }
        }
        
        // Wait for dependencies if using multiple streams
        if (config_.parallel_streams && !node->dependencies.empty()) {
            for (auto* dep : node->dependencies) {
                if (dep->stream != exec_stream) {
                    // Stream synchronization
                    cudaEvent_t event;
                    cudaEventCreate(&event);
                    cudaEventRecord(event, dep->stream);
                    cudaStreamWaitEvent(exec_stream, event);
                    cudaEventDestroy(event);
                }
            }
        }
        
        // Execute stage
        size_t input_size_for_stage = stage_input_size;
        
        // Check if this is a concatenation stage that needs multi-input setup
        if (auto* concat_stage = dynamic_cast<ConcatenationStage*>(node->stage)) {
            // Gather all dependency outputs and sizes based on aux_indices
            std::vector<void*> dep_outputs;
            std::vector<size_t> dep_sizes;
            
            for (size_t dep_idx = 0; dep_idx < node->dependencies.size(); dep_idx++) {
                auto* dep = node->dependencies[dep_idx];
                int aux_idx = node->dependency_aux_indices[dep_idx];
                
                if (aux_idx == -1) {
                    // Use primary output
                    dep_outputs.push_back(dep->output_buffer);
                    dep_sizes.push_back(dep->output_size);
                } else {
                    // Use auxiliary output
                    if (aux_idx >= 0 && aux_idx < (int)dep->aux_buffers.size()) {
                        dep_outputs.push_back(dep->aux_buffers[aux_idx]);
                        dep_sizes.push_back(dep->aux_sizes[aux_idx]);
                    } else {
                        throw std::runtime_error("Invalid aux output index " + std::to_string(aux_idx) +
                                               " for concatenation dependency " + dep->stage->getName());
                    }
                }
            }
            
            // Set dependency info before execution
            concat_stage->setDependencyInfo(dep_outputs, dep_sizes);
            
            // Execute concatenation
            current_size = concat_stage->execute(
                nullptr, 0,  // Input not used for concatenation
                node->output_buffer,
                exec_stream
            );
            concat_stage->updateThroughputStats(input_size_for_stage, current_size);
        } else if (auto* multi_stage = dynamic_cast<MultiOutputStage*>(node->stage)) {
            // Multi-output execution
            current_size = multi_stage->executeMulti(
                stage_input, stage_input_size,
                node->output_buffer,
                node->aux_buffers,
                node->aux_sizes,
                exec_stream
            );
            // Update throughput stats
            multi_stage->updateThroughputStats(input_size_for_stage, current_size);
        } else {
            // Single-output execution
            current_size = node->stage->execute(
                stage_input, stage_input_size,
                node->output_buffer,
                exec_stream
            );
            // Update throughput stats
            node->stage->updateThroughputStats(input_size_for_stage, current_size);
        }
        
        // Check if output exceeded allocated buffer (abort on overflow to prevent corruption)
        if (config_.memory_mode == MemoryMode::CONSERVATIVE && 
            current_size > node->allocated_output_size) {
            std::cerr << "\n⚠️  ERROR: " << node->stage->getName() 
                     << " output (" << (current_size / 1024.0 / 1024.0) << " MB) "
                     << "exceeded allocated buffer (" 
                     << (node->allocated_output_size / 1024.0 / 1024.0) << " MB)!\n"
                     << "   Buffer overflow detected - aborting to prevent memory corruption.\n"
                     << "   Solution: Use MemoryMode::SAFE or increase conservative_alloc_factor.\n\n";
            return 0;  // Return 0 to indicate compression failure
        }
        
        node->output_size = current_size;
        node->completed = true;
        
        // In STREAM mode, free intermediate buffers to reduce peak memory usage
        // Buffers will be re-allocated from the pool on next compression (pool reuses freed memory)
        if (!config_.use_cuda_graph && !node->dependencies.empty()) {
            for (auto* dep : node->dependencies) {
                // Check if any future stages still need this dependency
                bool dep_still_needed = false;
                for (size_t j = i + 1; j < nodes_.size(); j++) {
                    for (auto* future_dep : nodes_[j]->dependencies) {
                        if (future_dep == dep) {
                            dep_still_needed = true;
                            break;
                        }
                    }
                    if (dep_still_needed) break;
                }
                
                // Free dependency buffers if no longer needed by future stages
                if (!dep_still_needed && !dep->freed) {
                    if (dep->output_buffer) {
                        memory_pool_->free(dep->output_buffer, exec_stream);
                        dep->output_buffer = nullptr;
                    }
                    for (auto* aux_buf : dep->aux_buffers) {
                        if (aux_buf) {
                            memory_pool_->free(aux_buf, exec_stream);
                        }
                    }
                    // Clear aux buffer pointers
                    for (size_t j = 0; j < dep->aux_buffers.size(); j++) {
                        dep->aux_buffers[j] = nullptr;
                    }
                    dep->freed = true;
                }
            }
        }
        
        // Next stage's input is this stage's output
        current_input = node->output_buffer;
    }
    
    // Copy final output
    DagNode* final_node = nodes_.back();
    cudaMemcpyAsync(output, final_node->output_buffer, final_node->output_size,
                    cudaMemcpyDeviceToDevice, stream);
    cudaStreamSynchronize(stream);
    

    return final_node->output_size;
}

void Dag::reset() {
    for (auto* node : nodes_) {
        node->completed = false;
        node->output_size = 0;
        node->aux_sizes.clear();
    }
}

void Dag::printStructure() const {
    std::cout << "\n========== DAG Structure ==========\n";
    std::cout << "Total stages: " << nodes_.size() << "\n";
    std::cout << "Execution order:\n";
    
    for (size_t i = 0; i < nodes_.size(); i++) {
        auto* node = nodes_[i];
        std::cout << "  [" << i << "] "
                  << node->stage->getName()
                  << " (level " << node->level << ")";
        
        if (!node->dependencies.empty()) {
            std::cout << " <- depends on: ";
            for (size_t j = 0; j < node->dependencies.size(); j++) {
                if (j > 0) std::cout << ", ";
                std::cout << node->dependencies[j]->stage->getName();
            }
        }
        std::cout << "\n";
    }
    
    std::cout << "===================================\n" << std::endl;
}

void Dag::printStats() const {
    std::cout << "\n========== DAG Execution Statistics ==========\n";
    
    for (auto* node : nodes_) {
        std::cout << node->stage->getName() << ":\n";
        std::cout << "  Input size: " << std::fixed << std::setprecision(2) << (node->stage->getLastInputBytes() / 1048576.0) << " MB\n";
        std::cout << "  Output size: " << std::fixed << std::setprecision(2) << (node->output_size / 1048576.0) << " MB\n";
        
        if (node->stage->getExecutionCount() > 0) {
            std::cout << "  Execution time: " 
                     << node->stage->getLastExecutionTime() << " ms\n";
            std::cout << "  Throughput: " 
                     << node->stage->getThroughput() << " GB/s\n";
        }
    }
    
    std::cout << "=============================================\n" << std::endl;
}

void Dag::printBufferAllocations() const {
    std::cout << "\n========== Buffer Allocations ==========\n";
    
    for (size_t i = 0; i < nodes_.size(); i++) {
        auto* node = nodes_[i];
        std::cout << "[" << i << "] " << node->stage->getName() << ":\n";
        
        // Print output buffer
        if (node->output_buffer) {
            std::cout << "  Output buffer:    " << node->output_buffer 
                     << " (allocated: " << std::fixed << std::setprecision(2)
                     << (node->allocated_output_size / 1024.0) << " KB)\n";
        } else {
            std::cout << "  Output buffer:    (null)\n";
        }
        
        // Print auxiliary buffers
        if (!node->aux_buffers.empty()) {
            for (size_t j = 0; j < node->aux_buffers.size(); j++) {
                if (node->aux_buffers[j]) {
                    std::cout << "  Aux buffer [" << j << "]:  " 
                             << node->aux_buffers[j];
                    if (j < node->allocated_aux_sizes.size()) {
                        std::cout << " (allocated: " << std::fixed << std::setprecision(2)
                                 << (node->allocated_aux_sizes[j] / 1024.0) << " KB)";
                    }
                    std::cout << "\n";
                } else {
                    std::cout << "  Aux buffer [" << j << "]:  (null)\n";
                }
            }
        }
        
        std::cout << "\n";
    }
    
    std::cout << "=========================================\n" << std::endl;
}

} // namespace fz

