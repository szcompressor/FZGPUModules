#include "pipeline/dag.h"
#include "stage/stage.h"
#include "mem/mempool.h"
#include "log.h"
#include "cuda_check.h"

#include <algorithm>
#include <queue>
#include <string>
#include <stdexcept>

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
      profiling_enabled_(false),
      bounds_check_enabled_(false),
      capture_mode_(false),
      coloring_disabled_(false),
      coloring_applied_(false) {
    
    if (!mem_pool_) {
        throw std::invalid_argument("MemoryPool cannot be null");
    }
}

CompressionDAG::~CompressionDAG() {
    // Free all outstanding buffer allocations before the MemoryPool is destroyed.
    //
    // With PREALLOCATE, buffers are allocated once in preallocateBuffers() and
    // intentionally kept alive across compress() calls (reset() skips freeing them).
    // With MINIMAL mode, any buffers still live here are edge-case leftovers.
    // Either way we must cudaFreeAsync them before calling cudaMemPoolDestroy, otherwise
    // the pool destructor will hang waiting for the stream-ordered allocations to be freed.
    //
    // Use stream 0 then synchronize so the frees complete before the pool is torn down.
    if (coloring_applied_) {
        // Each color region is a single allocation shared by multiple BufferInfo entries.
        // Free by region pointer to avoid double-free of aliased d_ptrs.
        for (void* ptr : color_region_ptrs_) {
            if (ptr) mem_pool_->free(ptr, /*stream=*/0);
        }
        color_region_ptrs_.clear();
    } else {
        for (auto& [buffer_id, buffer] : buffers_) {
            if (buffer.is_allocated && !buffer.is_external && buffer.d_ptr) {
                mem_pool_->free(buffer.d_ptr, /*stream=*/0);
                buffer.is_allocated = false;
                buffer.d_ptr = nullptr;
            }
        }
    }
    cudaDeviceSynchronize();

    for (auto* node : nodes_) {
        if (node->completion_event) {
            FZ_CUDA_CHECK_WARN(cudaEventDestroy(node->completion_event));
        }
        if (node->start_event) {
            FZ_CUDA_CHECK_WARN(cudaEventDestroy(node->start_event));
        }
        delete node;
    }
    
    if (owns_streams_) {
        for (auto stream : streams_) {
            FZ_CUDA_CHECK_WARN(cudaStreamDestroy(stream));
        }
    }
}

DAGNode* CompressionDAG::addStage(Stage* stage, const std::string& name) {
    if (is_finalized_) {
        throw std::runtime_error("Cannot modify DAG after finalization");
    }
    
    cudaError_t preErr = cudaGetLastError();
    if (preErr != cudaSuccess) {
        FZ_LOG(WARN, "Pre-existing CUDA error in addStage (before cudaEventCreate): %s",
               cudaGetErrorString(preErr));
        // Error is now cleared
    }
    
    auto* node = new DAGNode(stage);
    node->id = static_cast<int>(nodes_.size());
    node->name = name.empty() ? "stage_" + std::to_string(node->id) : name;
    
    cudaError_t err = cudaEventCreate(&node->completion_event);
    if (err != cudaSuccess) {
        delete node;
        throw std::runtime_error(
            std::string("Failed to create CUDA event for stage '") + node->name + 
            "': " + cudaGetErrorString(err));
    }

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
    
    dependent->dependencies.push_back(dependency);
    dependency->dependents.push_back(dependent);
    
    int buffer_id = next_buffer_id_++;
    BufferInfo& buffer = buffers_[buffer_id];
    
    if (buffer_size == 0 && dependency->stage) {
        // TODO: Could call dependency->stage->estimateOutputSizes() here
        // For now, caller must provide size or we throw
        throw std::runtime_error("Buffer size must be provided for dependency");
    }
    buffer.size = buffer_size;
    buffer.initial_size = buffer_size;
    buffer.tag = dependency->name + "_to_" + dependent->name;
    buffer.is_persistent = false;
    buffer.producer_output_index = output_index;  // Track which output
    
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
    buffer.initial_size = size;
    buffer.tag = tag;
    buffer.is_persistent = true;  // Input is persistent
    buffer.producer_stage_id = -1;  // External input
    // Mark external immediately: the pointer is owned by the caller and must
    // not be freed or counted in pool-sizing calculations.  setExternalPointer()
    // will supply the actual device pointer before execution.
    buffer.is_external = true;

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
    buffer.initial_size = size;
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
    buffer.initial_size = size;
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
    
    auto it = producer->output_index_to_buffer_id.find(output_index);
    if (it == producer->output_index_to_buffer_id.end()) {
        return false;  // Buffer doesn't exist
    }
    
    int buffer_id = it->second;
    BufferInfo& buffer = buffers_[buffer_id];
    
    consumer->input_buffer_ids.push_back(buffer_id);
    buffer.consumer_stage_ids.push_back(consumer->id);
    
    consumer->dependencies.push_back(producer);
    producer->dependents.push_back(consumer);
    
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
    
    if (owns_streams_) {
        for (auto stream : streams_) {
            cudaStreamDestroy(stream);
        }
        streams_.clear();
    }
    
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
    for (auto& [buffer_id, buffer] : buffers_) {
        buffer.remaining_consumers = buffer.consumer_stage_ids.size();
    }
    
    // Kahn's algorithm cycle detection: verify this is a valid DAG.
    // Always enabled — a cycle is always a programming error.
    {
        std::unordered_map<int, int> in_degree;
        for (auto* node : nodes_) {
            if (in_degree.find(node->id) == in_degree.end()) {
                in_degree[node->id] = 0;
            }
            for (auto* dep : node->dependents) {
                in_degree[dep->id]++;
            }
        }
        
        std::queue<int> q;
        for (auto& [id, deg] : in_degree) {
            if (deg == 0) q.push(id);
        }
        
        int visited = 0;
        while (!q.empty()) {
            int cur = q.front(); q.pop();
            visited++;
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
                "Cyclic dependency detected in pipeline graph! "
                "Topological sort visited " +
                std::to_string(visited) + " of " +
                std::to_string(nodes_.size()) + " nodes"
            );
        }
        FZ_LOG(DEBUG, "DAG validation passed: %zu nodes, no cycles", nodes_.size());
    }
    
    assignLevels();

    if (streams_.empty()) {
        int max_parallelism = getMaxParallelism();
        if (max_parallelism > 1) {
            for (int i = 0; i < max_parallelism; i++) {
                cudaStream_t stream;
                cudaError_t err = cudaStreamCreate(&stream);
                if (err != cudaSuccess) {
                    FZ_LOG(WARN, "Failed to create CUDA stream %d: %s", i, cudaGetErrorString(err));
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
            streams_.push_back(0);  // Default stream
            owns_streams_ = false;  // We don't own stream 0, don't destroy it
        }
    }
    
    if (!streams_.empty()) {
        assignStreams();
    }

    // Pre-size per-node execution scratch vectors (§5).
    // Sized once here; reused on every execute() call with no heap allocation.
    for (auto* node : nodes_) {
        node->exec_inputs.resize(node->input_buffer_ids.size());
        node->exec_outputs.resize(node->output_buffer_ids.size());
        node->exec_sizes.resize(node->input_buffer_ids.size());
    }

    is_finalized_ = true;
}

void CompressionDAG::execute(cudaStream_t stream) {
    if (!is_finalized_) {
        throw std::runtime_error("DAG must be finalized before execution");
    }
    
    for (int level = 0; level <= max_level_; level++) {
        for (auto* node : levels_[level]) {
            // Use node's assigned stream, or fallback to provided stream.
            // During graph capture, force all work onto the single capture stream
            // so the entire execute() body is recorded in one graph regardless of
            // internal multi-stream assignments.
            cudaStream_t exec_stream = (capture_mode_ || !node->stream) ? stream : node->stream;
            
            for (auto* dep : node->dependencies) {
                FZ_CUDA_CHECK(cudaStreamWaitEvent(exec_stream, dep->completion_event));
            }
            
            if (strategy_ != MemoryStrategy::PREALLOCATE) {
                for (int buffer_id : node->output_buffer_ids) {
                    allocateBuffer(buffer_id, exec_stream);
                }
            }
            
            auto& inputs  = node->exec_inputs;
            auto& outputs = node->exec_outputs;
            auto& sizes   = node->exec_sizes;

            for (size_t j = 0; j < node->input_buffer_ids.size(); j++) {
                const auto& buf = buffers_[node->input_buffer_ids[j]];
                inputs[j] = buf.d_ptr;
                sizes[j]  = buf.size;
            }
            for (size_t j = 0; j < node->output_buffer_ids.size(); j++) {
                outputs[j] = buffers_[node->output_buffer_ids[j]].d_ptr;
            }
            
            if (profiling_enabled_ && node->start_event) {
                FZ_CUDA_CHECK(cudaEventRecord(node->start_event, exec_stream));
            }

            if (node->stage) {
                node->stage->execute(exec_stream, mem_pool_, inputs, outputs, sizes);

                // Propagate actual output sizes and run bounds checks.
                // Skipped during graph capture: getActualOutputSize() calls
                // completePendingSync() which would cudaStreamSynchronize(),
                // breaking the open capture bracket.  Sizes were already set
                // at finalize() time from estimateOutputSizes() worst-case
                // values, which is sufficient for PREALLOCATE graph mode.
                if (!capture_mode_) {
                    // §6: use index-based accessor — no unordered_map allocation.
                    for (size_t k = 0; k < node->output_buffer_ids.size(); k++) {
                        size_t sz = node->stage->getActualOutputSize(static_cast<int>(k));
                        if (sz > 0) buffers_[node->output_buffer_ids[k]].size = sz;
                    }

                    // Buffer overwrite bounds check.
                    auto do_bounds_check = [&]() {
                        auto out_names = node->stage->getOutputNames();
                        for (size_t k = 0; k < node->output_buffer_ids.size(); k++) {
                            int    bid  = node->output_buffer_ids[k];
                            size_t cap  = buffers_[bid].allocated_size;
                            if (cap == 0) continue;  // not yet allocated (MINIMAL, first run)
                            size_t sz   = node->stage->getActualOutputSize(static_cast<int>(k));
                            std::string name = (k < out_names.size())
                                ? out_names[k] : std::to_string(k);
                            if (sz > cap) {
                                throw std::runtime_error(
                                    "Buffer overwrite detected: stage '" + node->name +
                                    "' output '" + name + "' wrote " +
                                    std::to_string(sz) + " bytes into a " +
                                    std::to_string(cap) + "-byte buffer");
                            }
                        }
                    };

#ifndef NDEBUG
                    do_bounds_check();  // Always check in debug builds
#else
                    if (bounds_check_enabled_) do_bounds_check();  // Opt-in in release
#endif
                }
            }
            
            FZ_CUDA_CHECK(cudaEventRecord(node->completion_event, exec_stream));
            node->is_executed = true;
            
            for (int buffer_id : node->input_buffer_ids) {
                BufferInfo& buffer = buffers_[buffer_id];
                buffer.remaining_consumers--;

                if (buffer.remaining_consumers == 0 && !buffer.is_persistent) {
                    if (strategy_ != MemoryStrategy::PREALLOCATE) {
                        freeBuffer(buffer_id, exec_stream);
                    }
                }
            }
        }
    }

    // Synchronize all internal-stream work back to the provided stream.
    //
    // When the DAG has parallel nodes (max_parallelism > 1), each node runs on
    // its own internally-created CUDA stream rather than the caller-provided one.
    // Without this barrier, cudaStreamSynchronize(stream) in the caller returns
    // as soon as the provided stream is empty — which may be *before* the internal
    // streams have finished writing their result buffers.  This causes a race: any
    // work the caller submits to `stream` after execute() (e.g. a D2D memcpy to
    // copy out the result) can run concurrently with the DAG nodes still writing
    // to those buffers.
    //
    // We fix this by making `stream` wait for every node's completion_event.
    // completion_event was recorded on exec_stream immediately after the node's
    // stage execute(), so the waitEvent inserts a happens-before edge from
    // "node finishes" to "any subsequent work on stream".  This is a no-op for
    // single-source pipelines (all nodes already use `stream`).
    if (!capture_mode_) {
        for (int level = 0; level <= max_level_; level++) {
            for (auto* node : levels_[level]) {
                if (!node->is_executed) continue;
                cudaStream_t exec_stream =
                    (capture_mode_ || !node->stream) ? stream : node->stream;
                if (exec_stream != stream) {
                    FZ_CUDA_CHECK(cudaStreamWaitEvent(stream, node->completion_event));
                }
            }
        }
    }
}

void CompressionDAG::setCaptureMode(bool capture) {
    if (capture) {
        // Validate all stages before entering capture mode so the developer
        // gets a clear error at setup time rather than a silent broken graph.
        std::string incompatible;
        for (const auto* node : nodes_) {
            if (node->stage && !node->stage->isGraphCompatible()) {
                if (!incompatible.empty()) incompatible += ", ";
                incompatible += "'" + node->name + "' (" + node->stage->getName() + ")";
            }
        }
        if (!incompatible.empty()) {
            throw std::runtime_error(
                "Cannot enable CUDA Graph capture mode: the following stages are "
                "not graph-compatible: " + incompatible +
                ". Override Stage::isGraphCompatible() to return false to suppress "
                "this error and opt the stage out of graph capture.");
        }
    }
    capture_mode_ = capture;
}

void CompressionDAG::reset(cudaStream_t stream) {
    for (auto& [buffer_id, buffer] : buffers_) {
        // PREALLOCATE: buffers are owned for the lifetime of the DAG (they were
        // allocated once in preallocateBuffers() and execute() never re-allocates
        // them).  Freeing them here would leave every stage with a dangling pointer
        // on the next execute() call.  Skip the free; just restore the ref-count.
        if (strategy_ != MemoryStrategy::PREALLOCATE) {
            if (buffer.is_allocated && !buffer.is_persistent && !buffer.is_external) {
                if (buffer.d_ptr) {
                    mem_pool_->free(buffer.d_ptr, stream);
                    current_memory_usage_ -= buffer.allocated_size;
                }
                buffer.is_allocated = false;
                buffer.d_ptr = nullptr;
                buffer.allocated_size = 0;
            }
        }

        // Always restore consumer ref-count so the next execute() pass can
        // decrement it correctly.
        buffer.remaining_consumers = buffer.consumer_stage_ids.size();

        // Restore the functional size back to the allocated capacity.
        // During execute(), stages (like RZE) may shrink buffer.size to
        // reflect their actual output. If we don't reset this, subsequent 
        // executions will use the shrunk size, causing data truncation.
        // We use initial_size to ensure we even restore buffers that shrunk to 0.
        if (buffer.is_allocated && !buffer.is_external) {
             buffer.size = buffer.initial_size;
             // We intentionally do NOT reset buffer.allocated_size because
             // in PREALLOCATE mode, the pointer is kept and the capacity
             // must remain valid for do_bounds_check().
        }
    }
    
    // Reset node execution state
    for (auto* node : nodes_) {
        node->is_executed = false;
    }

    if (strategy_ != MemoryStrategy::PREALLOCATE) {
        current_memory_usage_ = 0;
    }
}

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
        if (coloring_applied_ && buffer_color_.count(buffer_id)) {
            // Buffer is part of a color group — its d_ptr is a shared color region
            // that may be aliased to other buffers in the same group.  Do NOT free
            // it here; the destructor releases each region once via color_region_ptrs_.
            // current_memory_usage_ is NOT decremented because the region is still
            // live and will be tracked until the DAG is destroyed.
        } else {
            // Individually-allocated buffer — safe to free now.
            mem_pool_->free(buffer.d_ptr, 0);
            current_memory_usage_ -= buffer.size;
        }
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
    // When propagateBufferSizes sets a new size, it serves as the newly established baseline envelope
    it->second.initial_size = new_size;
}

void CompressionDAG::allocateBuffer(int buffer_id, cudaStream_t stream) {
    BufferInfo& buffer = buffers_[buffer_id];

    if (buffer.is_external) {
        return;  // External buffer, don't allocate/free
    }

    // Zero-size buffers: mark as "allocated" with a null device pointer.
    // This arises when a stage legitimately produces no output (e.g.
    // outlier arrays when outlier_capacity == 0.0f).  Stages must handle
    // nullptr gracefully when their capacity indicates zero elements.
    if (buffer.size == 0) {
        if (!buffer.is_allocated) {
            buffer.is_allocated   = true;
            buffer.allocated_size = 0;
            buffer.d_ptr          = nullptr;
        }
        return;
    }

    if (buffer.is_allocated) {
        // Already allocated — keep if large enough, grow if target size has increased.
        // Re-add bytes to the usage counter so getCurrentMemoryUsage() is consistent
        // across compress() calls regardless of whether the buffer was freshly
        // allocated or reused from a previous call (e.g. persistent output buffers
        // after reset() zeroes the counter).
        if (buffer.size <= buffer.allocated_size) {
            current_memory_usage_ += buffer.size;
            peak_memory_usage_ = std::max(peak_memory_usage_, current_memory_usage_);
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


void CompressionDAG::assignLevels() {
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
    
    levels_.resize(max_level_ + 1);
    for (auto* node : nodes_) {
        levels_[node->level].push_back(node);
    }
}

void CompressionDAG::assignStreams() {
    if (streams_.empty()) return;
    
    std::unordered_map<int, int> level_stream_counter;
    
    for (auto* node : nodes_) {
        int stream_idx = level_stream_counter[node->level] % streams_.size();
        node->stream = streams_[stream_idx];
        level_stream_counter[node->level]++;
    }
}

} // namespace fz
