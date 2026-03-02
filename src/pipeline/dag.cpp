#include "pipeline/dag.h"
#include "stage/stage.h"
#include "mem/mempool.h"
#include "log.h"

#include <algorithm>
#include <iostream>
#include <queue>
#include <stdexcept>

/**
 * DAG
 * 
 * Key Features:
 * - Dependencies automatically create buffers
 * - Stages assigned to levels based on dependencies
 * - Parallel execution of stages at same level (different streams)
 * - Reference-counted buffer lifetimes (automatic free when consumers done)
 * - Three memory strategies: MINIMAL, PIPELINE, PREALLOCATE
 * 
 * Simplified API:
 *   auto* stage1 = dag.addStage(&stage);
 *   auto* stage2 = dag.addStage(&stage);
 *   dag.addDependency(stage2, stage1, buffer_size);  // Auto-creates buffer!
 * 
 * Execution Model:
 *   for each level (0 to max_level):
 *       for each node in level:
 *           execute node on assigned stream (parallel!)
 *           decrement input buffer ref counts
 *           free buffers when ref count reaches 0
 */

namespace fz {

CompressionDAG::CompressionDAG(MemoryPool* mem_pool, MemoryStrategy strategy)
    : mem_pool_(mem_pool),
      strategy_(strategy),
      next_buffer_id_(0),
      is_finalized_(false),
      owns_streams_(false),
      max_level_(0),
      current_memory_usage_(0),
      peak_memory_usage_(0),
      profiling_enabled_(false) {
    
    if (!mem_pool_) {
        throw std::invalid_argument("MemoryPool cannot be null");
    }
}

CompressionDAG::~CompressionDAG() {
    // Clean up nodes
    for (auto* node : nodes_) {
        if (node->completion_event) {
            cudaEventDestroy(node->completion_event);
        }
        if (node->start_event) {
            cudaEventDestroy(node->start_event);
        }
        delete node;
    }
    
    if (owns_streams_) {
        for (auto stream : streams_) {
            cudaStreamDestroy(stream);
        }
    }
}

// ========== DAG Construction ==========

DAGNode* CompressionDAG::addStage(Stage* stage, const std::string& name) {
    if (is_finalized_) {
        throw std::runtime_error("Cannot modify DAG after finalization");
    }
    
    cudaError_t preErr = cudaGetLastError();
    if (preErr != cudaSuccess) {
        std::cerr << "[DAG] WARNING: Pre-existing CUDA error in addStage (before cudaEventCreate): " 
                  << cudaGetErrorString(preErr) << std::endl;
        // Error is now cleared
    }
    
    auto* node = new DAGNode(stage);
    node->id = static_cast<int>(nodes_.size());
    node->name = name.empty() ? "stage_" + std::to_string(node->id) : name;
    
    // Create completion event for synchronization
    cudaError_t err = cudaEventCreate(&node->completion_event);
    if (err != cudaSuccess) {
        delete node;
        throw std::runtime_error(
            std::string("Failed to create CUDA event for stage '") + node->name + 
            "': " + cudaGetErrorString(err));
    }

    // Create start event for profiling if already enabled
    if (profiling_enabled_) {
        err = cudaEventCreate(&node->start_event);
        if (err != cudaSuccess) {
            cudaEventDestroy(node->completion_event);
            delete node;
            throw std::runtime_error(
                std::string("Failed to create profiling start event for stage '") +
                node->name + "': " + cudaGetErrorString(err));
        }
    }
    
    nodes_.push_back(node);
    return node;
}

int CompressionDAG::addDependency(DAGNode* dependent, DAGNode* dependency, 
                                  size_t buffer_size, int output_index) {
    if (is_finalized_) {
        throw std::runtime_error("Cannot modify DAG after finalization");
    }
    
    if (!dependent || !dependency) {
        throw std::runtime_error("Cannot add null dependency");
    }
    
    // Add dependency relationship
    dependent->dependencies.push_back(dependency);
    dependency->dependents.push_back(dependent);
    
    // Auto-create intermediate buffer between stages
    int buffer_id = next_buffer_id_++;
    BufferInfo& buffer = buffers_[buffer_id];
    
    // Estimate size if not provided
    if (buffer_size == 0 && dependency->stage) {
        // TODO: Could call dependency->stage->estimateOutputSizes() here
        // For now, caller must provide size or we throw
        throw std::runtime_error("Buffer size must be provided for dependency");
    }
    
    buffer.size = buffer_size;
    buffer.tag = dependency->name + "_to_" + dependent->name;
    buffer.is_persistent = false;
    buffer.producer_output_index = output_index;  // Track which output
    
    // Connect buffer to stages
    dependency->output_buffer_ids.push_back(buffer_id);
    dependency->output_index_to_buffer_id[output_index] = buffer_id;  // Track mapping
    buffer.producer_stage_id = dependency->id;
    
    dependent->input_buffer_ids.push_back(buffer_id);
    buffer.consumer_stage_ids.push_back(dependent->id);
    
    return buffer_id;
}

void CompressionDAG::setInputBuffer(DAGNode* node, size_t size, const std::string& tag) {
    if (is_finalized_) {
        throw std::runtime_error("Cannot modify DAG after finalization");
    }
    
    int buffer_id = next_buffer_id_++;
    BufferInfo& buffer = buffers_[buffer_id];
    buffer.size = size;
    buffer.tag = tag;
    buffer.is_persistent = true;  // Input is persistent
    buffer.producer_stage_id = -1;  // External input
    
    node->input_buffer_ids.push_back(buffer_id);
    buffer.consumer_stage_ids.push_back(node->id);
}

void CompressionDAG::setOutputBuffer(DAGNode* node, size_t size, const std::string& tag) {
    if (is_finalized_) {
        throw std::runtime_error("Cannot modify DAG after finalization");
    }
    
    int buffer_id = next_buffer_id_++;
    BufferInfo& buffer = buffers_[buffer_id];
    buffer.size = size;
    buffer.tag = tag;
    buffer.is_persistent = true;  // Output is persistent
    buffer.producer_output_index = static_cast<int>(node->output_buffer_ids.size());  // Assign sequential index
    
    node->output_buffer_ids.push_back(buffer_id);
    node->output_index_to_buffer_id[buffer.producer_output_index] = buffer_id;  // Track mapping
    buffer.producer_stage_id = node->id;
}

int CompressionDAG::addUnconnectedOutput(DAGNode* node, size_t size, int output_index, const std::string& tag) {
    if (is_finalized_) {
        throw std::runtime_error("Cannot modify DAG after finalization");
    }
    
    int buffer_id = next_buffer_id_++;
    BufferInfo& buffer = buffers_[buffer_id];
    buffer.size = size;
    buffer.tag = tag;
    buffer.is_persistent = false;  // Temporary buffer (not pipeline output)
    buffer.producer_stage_id = node->id;
    buffer.producer_output_index = output_index;
    buffer.consumer_stage_ids.clear();  // No consumers
    
    node->output_buffer_ids.push_back(buffer_id);
    node->output_index_to_buffer_id[output_index] = buffer_id;  // Track mapping
    
    return buffer_id;
}

bool CompressionDAG::connectExistingOutput(DAGNode* producer, DAGNode* consumer, int output_index) {
    if (is_finalized_) {
        throw std::runtime_error("Cannot modify DAG after finalization");
    }
    
    if (!producer || !consumer) {
        throw std::runtime_error("Cannot connect null nodes");
    }
    
    // Find the buffer for this output index
    auto it = producer->output_index_to_buffer_id.find(output_index);
    if (it == producer->output_index_to_buffer_id.end()) {
        return false;  // Buffer doesn't exist
    }
    
    int buffer_id = it->second;
    BufferInfo& buffer = buffers_[buffer_id];
    
    // Add consumer relationship
    consumer->input_buffer_ids.push_back(buffer_id);
    buffer.consumer_stage_ids.push_back(consumer->id);
    
    // Add dependency relationship
    consumer->dependencies.push_back(producer);
    producer->dependents.push_back(consumer);
    
    // Update buffer tag to reflect connection
    buffer.tag = producer->name + "_to_" + consumer->name;
    buffer.is_persistent = false;  // Now it's an intermediate buffer
    
    return true;
}

void CompressionDAG::updateBufferTag(int buffer_id, const std::string& tag) {
    if (buffers_.find(buffer_id) == buffers_.end()) {
        throw std::runtime_error("Buffer " + std::to_string(buffer_id) + " does not exist");
    }
    buffers_[buffer_id].tag = tag;
}

void CompressionDAG::setBufferPersistent(int buffer_id, bool persistent) {
    if (buffers_.find(buffer_id) == buffers_.end()) {
        throw std::runtime_error("Buffer " + std::to_string(buffer_id) + " does not exist");
    }
    buffers_[buffer_id].is_persistent = persistent;
}

void CompressionDAG::configureStreams(int num_streams) {
    if (is_finalized_) {
        throw std::runtime_error("Cannot configure streams after finalization");
    }
    
    // Destroy old streams if we own them
    if (owns_streams_) {
        for (auto stream : streams_) {
            cudaStreamDestroy(stream);
        }
        streams_.clear();
    }
    
    // Create new streams
    for (int i = 0; i < num_streams; i++) {
        cudaStream_t stream;
        cudaError_t err = cudaStreamCreate(&stream);
        if (err != cudaSuccess) {
            // Clean up already created streams
            for (auto s : streams_) {
                cudaStreamDestroy(s);
            }
            streams_.clear();
            throw std::runtime_error(
                std::string("Failed to create CUDA stream: ") + cudaGetErrorString(err));
        }
        streams_.push_back(stream);
    }
    owns_streams_ = true;
}

void CompressionDAG::finalize() {
    // Initialize consumer ref counts
    for (auto& [buffer_id, buffer] : buffers_) {
        buffer.remaining_consumers = buffer.consumer_stage_ids.size();
    }
    
#ifdef FZ_DAG_VALIDATE
    // Kahn's algorithm cycle detection: verify this is a valid DAG
    {
        // Build in-degree map
        std::unordered_map<int, int> in_degree;
        for (auto* node : nodes_) {
            if (in_degree.find(node->id) == in_degree.end()) {
                in_degree[node->id] = 0;
            }
            for (auto* dep : node->dependents) {
                in_degree[dep->id]++;
            }
        }
        
        // Seed queue with zero in-degree nodes (sources)
        std::queue<int> q;
        for (auto& [id, deg] : in_degree) {
            if (deg == 0) q.push(id);
        }
        
        int visited = 0;
        while (!q.empty()) {
            int cur = q.front(); q.pop();
            visited++;
            // Find the node
            for (auto* node : nodes_) {
                if (node->id == cur) {
                    for (auto* dep : node->dependents) {
                        if (--in_degree[dep->id] == 0) {
                            q.push(dep->id);
                        }
                    }
                    break;
                }
            }
        }
        
        if (visited != static_cast<int>(nodes_.size())) {
            throw std::runtime_error(
                "Cycle detected in DAG! Topological sort visited " +
                std::to_string(visited) + " of " +
                std::to_string(nodes_.size()) + " nodes"
            );
        }
        FZ_LOG(DEBUG, "DAG validation passed: %zu nodes, no cycles", nodes_.size());
    }
#endif
    
    // Assign execution levels based on dependencies
    assignLevels();
    
    // Auto-configure streams if not already set up
    if (streams_.empty()) {
        int max_parallelism = getMaxParallelism();
        if (max_parallelism > 1) {
            // Create streams for parallel execution
            for (int i = 0; i < max_parallelism; i++) {
                cudaStream_t stream;
                cudaError_t err = cudaStreamCreate(&stream);
                if (err != cudaSuccess) {
                    std::cerr << "[DAG] ERROR: Failed to create stream " << i 
                              << ": " << cudaGetErrorString(err) << std::endl;
                    // Clean up already created streams
                    for (auto s : streams_) {
                        cudaStreamDestroy(s);
                    }
                    streams_.clear();
                    throw std::runtime_error(
                        std::string("Failed to auto-create CUDA stream: ") + cudaGetErrorString(err));
                }
                streams_.push_back(stream);
            }
            owns_streams_ = true;
        } else {
            // Even with no parallelism, track that we use default stream 0
            // This makes reporting more consistent (stream count = 1)
            streams_.push_back(0);  // Default stream
            owns_streams_ = false;  // We don't own stream 0, don't destroy it
        }
    }
    
    // Assign streams to nodes for parallel execution
    if (!streams_.empty()) {
        assignStreams();
    }
    
    is_finalized_ = true;
}

// ========== Execution ==========

void CompressionDAG::execute(cudaStream_t stream) {
    if (!is_finalized_) {
        throw std::runtime_error("DAG must be finalized before execution");
    }
    
    // Execute level by level (enables parallelism within levels)
    for (int level = 0; level <= max_level_; level++) {
        for (auto* node : levels_[level]) {
            // Use node's assigned stream, or fallback to provided stream
            cudaStream_t exec_stream = node->stream ? node->stream : stream;
            
            // Wait for all dependencies to complete
            for (auto* dep : node->dependencies) {
                cudaStreamWaitEvent(exec_stream, dep->completion_event);
            }
            
            // Allocate output buffers based on strategy
            if (strategy_ != MemoryStrategy::PREALLOCATE) {
                for (int buffer_id : node->output_buffer_ids) {
                    allocateBuffer(buffer_id, exec_stream);
                }
            }
            
            // Execute stage
            // Build input/output pointer arrays from buffer IDs
            std::vector<void*> inputs, outputs;
            std::vector<size_t> sizes;
            
            for (int buf_id : node->input_buffer_ids) {
                inputs.push_back(buffers_[buf_id].d_ptr);
                sizes.push_back(buffers_[buf_id].size);
            }
            
            for (int buf_id : node->output_buffer_ids) {
                outputs.push_back(buffers_[buf_id].d_ptr);
            }
            
            // Record stage start for profiling (before kernel launch)
            if (profiling_enabled_ && node->start_event) {
                cudaEventRecord(node->start_event, exec_stream);
            }

            // Call stage execute
            if (node->stage) {
                node->stage->execute(exec_stream, mem_pool_, inputs, outputs, sizes);
            }
            
            // Record completion for dependent stages (and profiling end)
            cudaEventRecord(node->completion_event, exec_stream);
            node->is_executed = true;
            
            // Decrement ref count on input buffers and free if needed
            for (int buffer_id : node->input_buffer_ids) {
                BufferInfo& buffer = buffers_[buffer_id];
                buffer.remaining_consumers--;
                
                // Free when all consumers done (unless PREALLOCATE mode)
                if (buffer.remaining_consumers == 0 && !buffer.is_persistent) {
                    if (strategy_ != MemoryStrategy::PREALLOCATE) {
                        freeBuffer(buffer_id, exec_stream);
                    }
                }
            }
        }
    }
}

void CompressionDAG::reset(cudaStream_t stream) {
    // Free all non-persistent buffers
    for (auto& [buffer_id, buffer] : buffers_) {
        if (buffer.is_allocated && !buffer.is_persistent && !buffer.is_external) {
            mem_pool_->free(buffer.d_ptr, stream);
            current_memory_usage_ -= buffer.allocated_size;
            buffer.is_allocated = false;
            buffer.d_ptr = nullptr;
            buffer.allocated_size = 0;
        }
        
        // Reset consumer count
        buffer.remaining_consumers = buffer.consumer_stage_ids.size();
    }
    
    // Reset node execution state
    for (auto* node : nodes_) {
        node->is_executed = false;
    }
    
    current_memory_usage_ = 0;
}

// ========== Buffer Management ==========

void* CompressionDAG::getBuffer(int buffer_id) const {
    auto it = buffers_.find(buffer_id);
    if (it == buffers_.end()) {
        throw std::runtime_error("Invalid buffer ID: " + std::to_string(buffer_id));
    }
    return it->second.d_ptr;
}

size_t CompressionDAG::getBufferSize(int buffer_id) const {
    auto it = buffers_.find(buffer_id);
    if (it == buffers_.end()) {
        throw std::runtime_error("Invalid buffer ID: " + std::to_string(buffer_id));
    }
    return it->second.size;
}

const BufferInfo& CompressionDAG::getBufferInfo(int buffer_id) const {
    auto it = buffers_.find(buffer_id);
    if (it == buffers_.end()) {
        throw std::runtime_error("Invalid buffer ID: " + std::to_string(buffer_id));
    }
    return it->second;
}

void CompressionDAG::setExternalPointer(int buffer_id, void* external_ptr) {
    auto it = buffers_.find(buffer_id);
    if (it == buffers_.end()) {
        throw std::runtime_error("Invalid buffer ID: " + std::to_string(buffer_id));
    }
    
    BufferInfo& buffer = it->second;
    
    if (buffer.is_allocated && !buffer.is_external) {
        // Free previously allocated buffer before switching to external
        mem_pool_->free(buffer.d_ptr, 0);
        current_memory_usage_ -= buffer.size;
    }
    
    buffer.d_ptr = external_ptr;
    buffer.is_external = true;
    buffer.is_allocated = true;  // Mark as "allocated" (pointer is set)
    
    FZ_LOG(TRACE, "Set external pointer for %s (%.1f KB)",
           buffer.tag.c_str(), buffer.size / 1024.0);
}

void CompressionDAG::updateBufferSize(int buffer_id, size_t new_size) {
    auto it = buffers_.find(buffer_id);
    if (it == buffers_.end()) {
        throw std::runtime_error("Invalid buffer ID: " + std::to_string(buffer_id));
    }
    
    it->second.size = new_size;
}

// ========== Internal Implementation ==========

void CompressionDAG::allocateBuffer(int buffer_id, cudaStream_t stream) {
    BufferInfo& buffer = buffers_[buffer_id];

    if (buffer.is_external) {
        return;  // External buffer, don't allocate/free
    }

    if (buffer.is_allocated) {
        // Already allocated — keep if large enough, grow if target size has increased
        if (buffer.size <= buffer.allocated_size) {
            return;
        }
        // Buffer needs to grow: free the old allocation and fall through to re-allocate
        FZ_LOG(DEBUG, "Growing buffer %s: %.1f KB -> %.1f KB",
               buffer.tag.c_str(), buffer.allocated_size / 1024.0, buffer.size / 1024.0);
        mem_pool_->free(buffer.d_ptr, stream);
        current_memory_usage_ -= buffer.allocated_size;
        buffer.is_allocated = false;
        buffer.d_ptr = nullptr;
        buffer.allocated_size = 0;
    }
    
    // Allocate from memory pool
    buffer.d_ptr = mem_pool_->allocate(
        buffer.size, 
        stream, 
        buffer.tag, 
        buffer.is_persistent
    );
    
    if (!buffer.d_ptr) {
        throw std::runtime_error("Failed to allocate buffer: " + buffer.tag);
    }
    
    buffer.is_allocated = true;
    buffer.allocated_size = buffer.size;
    current_memory_usage_ += buffer.size;
    peak_memory_usage_ = std::max(peak_memory_usage_, current_memory_usage_);

    FZ_LOG(TRACE, "Allocated %s (%.1f KB) - Current usage: %.2f MB",
           buffer.tag.c_str(), buffer.size / 1024.0, current_memory_usage_ / (1024.0 * 1024.0));
}

void CompressionDAG::freeBuffer(int buffer_id, cudaStream_t stream) {
    BufferInfo& buffer = buffers_[buffer_id];
    
    if (buffer.is_external) {
        return;  // External buffer, don't free
    }
    
    if (buffer.is_allocated && buffer.d_ptr) {
        mem_pool_->free(buffer.d_ptr, stream);
        current_memory_usage_ -= buffer.allocated_size;
        buffer.is_allocated = false;
        buffer.d_ptr = nullptr;
        buffer.allocated_size = 0;

        FZ_LOG(TRACE, "Freed %s - Current usage: %.2f MB",
               buffer.tag.c_str(), current_memory_usage_ / (1024.0 * 1024.0));
    }
}

void CompressionDAG::planPreallocation() {
    // Allocate all buffers upfront (use default stream 0 or first available)
    cudaStream_t alloc_stream = streams_.empty() ? 0 : streams_[0];
    preallocateBuffers(alloc_stream);
}

void CompressionDAG::preallocateBuffers(cudaStream_t stream) {
    for (auto& [buffer_id, buffer] : buffers_) {
        allocateBuffer(buffer_id, stream);
    }
}

void CompressionDAG::assignLevels() {
    // Assign levels for parallel execution based on dependencies
    // Level 0 = no dependencies, Level N = max(dep levels) + 1
    
    levels_.clear();
    max_level_ = 0;
    
    for (size_t i = 0; i < nodes_.size(); i++) {
        auto* node = nodes_[i];
        node->execution_order = static_cast<int>(i);
        
        if (node->dependencies.empty()) {
            node->level = 0;
        } else {
            int max_dep_level = -1;
            for (auto* dep : node->dependencies) {
                max_dep_level = std::max(max_dep_level, dep->level);
            }
            node->level = max_dep_level + 1;
        }
        
        max_level_ = std::max(max_level_, node->level);
    }
    
    // Group nodes by level
    levels_.resize(max_level_ + 1);
    for (auto* node : nodes_) {
        levels_[node->level].push_back(node);
    }
}

void CompressionDAG::assignStreams() {
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

// ========== Profiling ==========

void CompressionDAG::enableProfiling(bool enable) {
    profiling_enabled_ = enable;

    if (enable) {
        // Create start_event for all nodes that were added before this call
        for (auto* node : nodes_) {
            if (!node->start_event) {
                cudaError_t err = cudaEventCreate(&node->start_event);
                if (err != cudaSuccess) {
                    // Non-fatal: log and leave start_event null (timing skipped for this node)
                    std::cerr << "[DAG] WARNING: Could not create profiling event for '"
                              << node->name << "': " << cudaGetErrorString(err) << "\n";
                }
            }
        }
    } else {
        // Destroy and null out all start events to reclaim resources
        for (auto* node : nodes_) {
            if (node->start_event) {
                cudaEventDestroy(node->start_event);
                node->start_event = nullptr;
            }
        }
    }
}

std::vector<StageTimingResult> CompressionDAG::collectTimings() {
    if (!profiling_enabled_) return {};

    // Sync all owned streams so that CUDA event queries are valid.
    // (node streams may differ from the fallback stream passed to execute())
    for (auto s : streams_) {
        if (s) cudaStreamSynchronize(s);
    }

    std::vector<StageTimingResult> results;
    results.reserve(nodes_.size());

    for (auto* node : nodes_) {
        if (!node->start_event || !node->completion_event) continue;

        StageTimingResult r;
        r.name       = node->name;
        r.level      = node->level;
        r.elapsed_ms = 0.0f;

        cudaError_t err = cudaEventElapsedTime(&r.elapsed_ms,
                                               node->start_event,
                                               node->completion_event);
        if (err != cudaSuccess) {
            std::cerr << "[DAG] WARNING: cudaEventElapsedTime failed for '"
                      << node->name << "': " << cudaGetErrorString(err) << "\n";
            r.elapsed_ms = -1.0f;
        }

        // Sum up buffer sizes for input/output byte counts
        r.input_bytes = 0;
        for (int buf_id : node->input_buffer_ids) {
            auto it = buffers_.find(buf_id);
            if (it != buffers_.end()) r.input_bytes += it->second.size;
        }
        r.output_bytes = 0;
        for (int buf_id : node->output_buffer_ids) {
            auto it = buffers_.find(buf_id);
            if (it != buffers_.end()) r.output_bytes += it->second.size;
        }

        results.push_back(r);
    }

    return results;
}

// ========== Query & Debug ==========

size_t CompressionDAG::getTotalBufferSize() const {
    size_t total = 0;
    for (const auto& [buffer_id, buffer] : buffers_) {
        total += buffer.size;
    }
    return total;
}

int CompressionDAG::getMaxParallelism() const {
    // Returns the maximum number of nodes at any single level
    int max_width = 0;
    for (const auto& level : levels_) {
        max_width = std::max(max_width, static_cast<int>(level.size()));
    }
    return max_width;
}

void CompressionDAG::printDAG() const {
    std::cout << "\n========== Compression DAG ==========\n";
    std::cout << "Strategy: ";
    switch (strategy_) {
        case MemoryStrategy::MINIMAL: std::cout << "MINIMAL\n"; break;
        case MemoryStrategy::PIPELINE: std::cout << "PIPELINE\n"; break;
        case MemoryStrategy::PREALLOCATE: std::cout << "PREALLOCATE\n"; break;
    }
    std::cout << "Parallel streams: " << streams_.size() << "\n";
    std::cout << "Max level: " << max_level_ << "\n";
    
    std::cout << "\nStages (" << nodes_.size() << "):\n";
    for (const auto* node : nodes_) {
        std::cout << "  [" << node->id << "] " << node->name 
                  << " (Level " << node->level << ", Stream " 
                  << (node->stream ? "assigned" : "default") << ")";
        if (!node->dependencies.empty()) {
            std::cout << " <- deps: [";
            for (size_t i = 0; i < node->dependencies.size(); i++) {
                std::cout << node->dependencies[i]->id;
                if (i < node->dependencies.size() - 1) std::cout << ", ";
            }
            std::cout << "]";
        }
        if (!node->input_buffer_ids.empty()) {
            std::cout << ", inputs: [";
            for (size_t i = 0; i < node->input_buffer_ids.size(); i++) {
                std::cout << node->input_buffer_ids[i];
                if (i < node->input_buffer_ids.size() - 1) std::cout << ", ";
            }
            std::cout << "]";
        }
        if (!node->output_buffer_ids.empty()) {
            std::cout << " -> outputs: [";
            for (size_t i = 0; i < node->output_buffer_ids.size(); i++) {
                std::cout << node->output_buffer_ids[i];
                if (i < node->output_buffer_ids.size() - 1) std::cout << ", ";
            }
            std::cout << "]";
        }
        std::cout << "\n";
    }
    
    std::cout << "\nBuffers (" << buffers_.size() << "):\n";
    for (const auto& [buffer_id, buffer] : buffers_) {
        std::cout << "  [" << buffer_id << "] " << buffer.tag 
                  << " (" << (buffer.size / 1024.0) << " KB)";
        if (buffer.is_persistent) std::cout << " [PERSISTENT]";
        std::cout << "\n";
        std::cout << "      producer: " << buffer.producer_stage_id 
                  << ", consumers: [";
        for (size_t i = 0; i < buffer.consumer_stage_ids.size(); i++) {
            std::cout << buffer.consumer_stage_ids[i];
            if (i < buffer.consumer_stage_ids.size() - 1) std::cout << ", ";
        }
        std::cout << "]\n";
    }
    
    if (!levels_.empty()) {
        std::cout << "\nExecution levels (parallel within level):\n";
        for (int level = 0; level <= max_level_; level++) {
            std::cout << "  Level " << level << ": [";
            for (size_t i = 0; i < levels_[level].size(); i++) {
                std::cout << levels_[level][i]->id;
                if (i < levels_[level].size() - 1) std::cout << ", ";
            }
            std::cout << "]\n";
        }
    }
    
    std::cout << "\nMemory usage:\n";
    std::cout << "  Total buffer capacity: " << (getTotalBufferSize() / (1024.0 * 1024.0)) << " MB\n";
    std::cout << "=====================================\n\n";
}

void CompressionDAG::printBufferLifetimes() const {
    std::cout << "\n========== Buffer Lifetimes ==========\n";
    for (const auto& [buffer_id, buffer] : buffers_) {
        std::cout << "[" << buffer_id << "] " << buffer.tag << ":\n";
        std::cout << "  Size: " << (buffer.size / 1024.0) << " KB\n";
        std::cout << "  Producer stage: " << buffer.producer_stage_id << "\n";
        std::cout << "  Consumer stages: [";
        for (size_t i = 0; i < buffer.consumer_stage_ids.size(); i++) {
            std::cout << buffer.consumer_stage_ids[i];
            if (i < buffer.consumer_stage_ids.size() - 1) std::cout << ", ";
        }
        std::cout << "]\n";
        std::cout << "  Remaining consumers: " << buffer.remaining_consumers << "\n";
        std::cout << "  Persistent: " << (buffer.is_persistent ? "yes" : "no") << "\n";
        std::cout << "  Allocated: " << (buffer.is_allocated ? "yes" : "no") << "\n";
    }
    std::cout << "======================================\n\n";
}

} // namespace fz
