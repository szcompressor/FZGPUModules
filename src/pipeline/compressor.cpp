#include "pipeline/compressor.h"
#include "fzm_format.h"
#include "log.h"
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <cstdint>
#include <vector>
#include <unordered_set>

namespace fz {

Pipeline::Pipeline(size_t input_data_size, MemoryStrategy strategy, float pool_multiplier)
    : strategy_(strategy),
      is_inverse_mode_(false),
      num_streams_(1),
      soft_run_enabled_(false),
      is_finalized_(false),
      input_node_(nullptr),
      input_buffer_id_(-1),
      d_concat_buffer_(nullptr),
      concat_buffer_capacity_(0),
      needs_concat_(false),
      input_size_(0),
      input_size_hint_(input_data_size) {
    
    // Create memory pool with configuration
    MemoryPoolConfig pool_config(input_data_size, pool_multiplier);
    mem_pool_ = std::make_unique<MemoryPool>(pool_config);
    
    // Create DAG with the memory pool
    dag_ = std::make_unique<CompressionDAG>(mem_pool_.get(), strategy);
}

Pipeline::~Pipeline() {
    // Free concat buffer before pool is destroyed
    if (d_concat_buffer_ && mem_pool_) {
        mem_pool_->free(d_concat_buffer_, 0);
        d_concat_buffer_ = nullptr;
    }
}

// ========== Configuration ==========

void Pipeline::setMemoryStrategy(MemoryStrategy strategy) {
    if (is_finalized_) {
        throw std::runtime_error("Cannot change strategy after finalization");
    }
    strategy_ = strategy;
    // Recreate DAG with new strategy
    dag_ = std::make_unique<CompressionDAG>(mem_pool_.get(), strategy);
}

void Pipeline::setNumStreams(int num_streams) {
    if (is_finalized_) {
        throw std::runtime_error("Cannot configure streams after finalization");
    }
    num_streams_ = num_streams;
}

void Pipeline::enableSoftRun(bool enable) {
    if (is_finalized_) {
        throw std::runtime_error("Cannot enable soft-run after finalization");
    }
    soft_run_enabled_ = enable;
}

void Pipeline::setInverseMode(bool inverse) {
    // Toggle all stages to inverse mode
    for (auto& stage_ptr : stages_) {
        stage_ptr->setInverse(inverse);
    }
    
    is_inverse_mode_ = inverse;
    
    // Must re-finalize to rebuild DAG with reversed connections
    if (is_finalized_) {
        is_finalized_ = false;
        
        FZ_LOG(INFO, "Switched to %s mode, pipeline needs re-finalization",
               inverse ? "decompression" : "compression");
    }
}

// ========== Builder API ===========

int Pipeline::connect(Stage* dependent, Stage* producer, const std::string& output_name) {
    if (is_finalized_) {
        throw std::runtime_error("Cannot connect stages after finalization");
    }
    
    // Get DAG nodes for stages
    auto dep_it = stage_to_node_.find(dependent);
    auto prod_it = stage_to_node_.find(producer);
    
    if (dep_it == stage_to_node_.end() || prod_it == stage_to_node_.end()) {
        throw std::runtime_error("Cannot connect stages not added to pipeline");
    }
    
    DAGNode* dep_node = dep_it->second;
    DAGNode* prod_node = prod_it->second;
    
    // Validate output name and get output index
    int output_index = producer->getOutputIndex(output_name);
    if (output_index < 0) {
        throw std::runtime_error(
            "Stage '" + producer->getName() + "' does not have output '" + output_name + "'");
    }
    
    // Store connection for potential reversal
    ConnectionInfo conn_info;
    conn_info.dependent = dependent;
    conn_info.producer = producer;
    conn_info.output_name = output_name;
    conn_info.output_index = output_index;
    connections_.push_back(conn_info);
    
    // Try to reuse existing pre-allocated buffer (created in addStage)
    bool connected = dag_->connectExistingOutput(prod_node, dep_node, output_index);
    
    if (!connected) {
        // Shouldn't happen since addStage pre-allocates all outputs
        int buffer_id = dag_->addDependency(dep_node, prod_node, 1, output_index);
        FZ_LOG(WARN, "Had to create new buffer for %s.%s (should have been pre-allocated)",
               producer->getName().c_str(), output_name.c_str());
    }
    
    FZ_LOG(DEBUG, "Connected %s.%s -> %s",
           producer->getName().c_str(), output_name.c_str(), dependent->getName().c_str());
    
    // Return the buffer ID (look it up from the nodes mapping)
    auto it = prod_node->output_index_to_buffer_id.find(output_index);
    return (it != prod_node->output_index_to_buffer_id.end()) ? it->second : -1;
}

int Pipeline::connect(Stage* dependent, const std::vector<Stage*>& producers) {
    int last_buffer_id = -1;
    for (Stage* producer : producers) {
        last_buffer_id = connect(dependent, producer);
    }
    return last_buffer_id;
}

Stage* Pipeline::addRawStage(Stage* stage) {
    if (is_finalized_) {
        throw std::runtime_error("Cannot add stages after finalization");
    }
    
    // Wrap in unique_ptr for ownership
    auto stage_ptr = std::unique_ptr<Stage>(stage);
    
    // Add to internal DAG
    DAGNode* node = dag_->addStage(stage, stage->getName());
    
    // Pre-allocate all output buffers as unconnected
    size_t num_outputs = stage->getNumOutputs();
    auto output_names = stage->getOutputNames();
    
    for (size_t i = 0; i < num_outputs; i++) {
        std::string output_name = i < output_names.size() ? output_names[i] : std::to_string(i);
        std::string tag = stage->getName() + "." + output_name + "_unconnected";
        dag_->addUnconnectedOutput(node, 1, i, tag);
    }
    
    stage_to_node_[stage] = node;
    stages_.push_back(std::move(stage_ptr));
    
    return stage;
}

void Pipeline::finalize() {
    if (is_finalized_) {
        throw std::runtime_error("Pipeline already finalized");
    }
    
    // Rebuild DAG connections if in inverse mode
    if (is_inverse_mode_) {
        rebuildInverseConnections();
    }
    
    validate();
    
    auto [sources, sinks] = identifyTopology();
    setupInputBuffers(sources);
    
    // Unified output detection: ALL unconnected outputs become pipeline outputs
    int pipeline_outputs = autoDetectUnconnectedOutputs();
    detectMultiOutputScenario(pipeline_outputs);
    
    configureStreamsIfNeeded();
    
    dag_->finalize();
    propagateBufferSizes();

    if (strategy_ == MemoryStrategy::PREALLOCATE) {
        dag_->preallocateBuffers();
    }

    num_streams_ = std::max(1, static_cast<int>(dag_->getStreamCount()));
    is_finalized_ = true;
    
    FZ_LOG(INFO, "Finalized with %zu stages, strategy=%s",
           stages_.size(),
           strategy_ == MemoryStrategy::MINIMAL ? "MINIMAL" : 
           strategy_ == MemoryStrategy::PIPELINE ? "PIPELINE" : "PREALLOCATE");
}

// ========== Execution ==========

// ===== Unified Inverse Execution Engine =====

std::pair<void*, size_t> Pipeline::runInversePipeline(
    const std::vector<InverseStageSpec>& specs,
    std::unordered_map<int, std::pair<void*, size_t>>& live_bufs,
    size_t uncompressed_size,
    MemoryPool& pool,
    cudaStream_t stream
) {
    if (specs.empty()) {
        throw std::runtime_error("runInversePipeline: no stages to execute");
    }

    // Track which live_bufs entries were pool-allocated here
    // (vs. externally provided leaf buffers that must not be freed)
    std::unordered_set<int> pool_owned;

    // Process stages in reverse forward order (last compression stage first)
    for (int i = static_cast<int>(specs.size()) - 1; i >= 0; i--) {
        const auto& spec = specs[i];
        Stage* stage = spec.stage;

        // Gather inverse inputs: these are the forward outputs of this stage
        std::vector<void*> inv_inputs;
        std::vector<size_t> inv_sizes;
        for (int buf_id : spec.fwd_output_ids) {
            auto it = live_bufs.find(buf_id);
            if (it != live_bufs.end() && it->second.first) {
                inv_inputs.push_back(it->second.first);
                inv_sizes.push_back(it->second.second);
            } else {
                FZ_LOG(WARN, "Stage '%s': inverse input buffer %d not found in live_bufs",
                       stage->getName().c_str(), buf_id);
                inv_inputs.push_back(nullptr);
                inv_sizes.push_back(0);
            }
        }

        // Allocate inverse output buffer (conservative upper bound)
        // TODO: support fan-out inverse (multiple fwd_input_ids -> multiple inv outputs)
        void* d_inv_out = pool.allocate(uncompressed_size, stream, "inv_stage_out");
        if (!d_inv_out) {
            throw std::runtime_error(
                "MemoryPool allocation failed for inverse stage '" + stage->getName() + "'");
        }

        std::vector<void*> inv_outputs = { d_inv_out };
        stage->execute(stream, inv_inputs, inv_outputs, inv_sizes);
        cudaStreamSynchronize(stream);

        // Determine actual output size
        size_t actual_size = uncompressed_size;
        auto out_sizes = stage->getActualOutputSizesByName();
        auto out_names = stage->getOutputNames();
        if (!out_names.empty()) {
            auto sz_it = out_sizes.find(out_names[0]);
            if (sz_it != out_sizes.end()) actual_size = sz_it->second;
        }

        // Store result keyed by the stage's first forward input buffer ID
        if (!spec.fwd_input_ids.empty()) {
            int result_id = spec.fwd_input_ids[0];
            live_bufs[result_id] = { d_inv_out, actual_size };
            pool_owned.insert(result_id);
        }

        FZ_LOG(DEBUG, "Inverse '%s': %zu inputs -> %.2f KB",
               stage->getName().c_str(), inv_inputs.size(), actual_size / 1024.0);

        // Free pool-owned intermediate inputs that are no longer needed
        // (forward outputs of this stage that were computed by earlier inverse stages)
        for (int buf_id : spec.fwd_output_ids) {
            if (pool_owned.count(buf_id)) {
                pool.free(live_bufs.at(buf_id).first, stream);
                live_bufs.erase(buf_id);
                pool_owned.erase(buf_id);
            }
        }
    }

    // The result is at specs[0].fwd_input_ids[0] (the forward pipeline source's input)
    if (specs[0].fwd_input_ids.empty()) {
        throw std::runtime_error("runInversePipeline: source stage has no forward input IDs");
    }
    int final_id = specs[0].fwd_input_ids[0];
    auto it = live_bufs.find(final_id);
    if (it == live_bufs.end() || !it->second.first) {
        throw std::runtime_error("runInversePipeline: no result produced");
    }
    return it->second;
}

// =====

void Pipeline::compress(
    const void* d_input,
    size_t input_size,
    void** d_output,
    size_t* output_size,
    cudaStream_t stream
) {
    if (!is_finalized_) {
        throw std::runtime_error("Pipeline must be finalized before execution");
    }
    
    input_size_ = input_size;
    
    // Set external input pointer (zero-copy from user's buffer)
    dag_->setExternalPointer(input_buffer_id_, const_cast<void*>(d_input));
    
    // Execute DAG
    // For MINIMAL/PIPELINE with soft-run:
    if (soft_run_enabled_ && strategy_ != MemoryStrategy::PREALLOCATE) {
        // Two-phase execution:
        // 1. Execute with soft-run to determine sizes
        // 2. Re-execute with correct sizes
        
        // TODO: Implement soft-run logic
        // For each stage that supports it:
        //   - Run softRun() to get actual output size
        //   - Update buffer size in DAG
        //   - Re-allocate with correct size
        
        // For now, just execute normally
        dag_->execute(stream);
    } else {
        // Single execution with estimated sizes
        dag_->execute(stream);
    }
    
    // Capture buffer metadata for file serialization
    buffer_metadata_.clear();
    for (size_t i = 0; i < output_buffer_ids_.size(); i++) {
        int buffer_id = output_buffer_ids_[i];
        const auto& buffer_info = dag_->getBufferInfo(buffer_id);
        DAGNode* producer = output_nodes_[i];
        
        BufferMetadata meta;
        meta.buffer_id = buffer_id;
        meta.allocated_size = buffer_info.size;
        meta.producer = producer;
        meta.output_index = buffer_info.producer_output_index;
        
        // Get output name
        auto output_names = producer->stage->getOutputNames();
        int output_idx = buffer_info.producer_output_index;
        meta.name = (output_idx >= 0 && output_idx < static_cast<int>(output_names.size())) 
                    ? output_names[output_idx] 
                    : "output";
        
        // Get actual size from stage
        auto sizes_by_name = producer->stage->getActualOutputSizesByName();
        auto it = sizes_by_name.find(meta.name);
        meta.actual_size = (it != sizes_by_name.end()) ? it->second : buffer_info.size;
        
        buffer_metadata_.push_back(meta);
    }
    
    // Get output buffer and size
    if (needs_concat_) {
        // Multiple sinks: Concatenate all outputs
        concatOutputs(d_output, output_size, stream);
    } else {
        // Single sink: Return output directly
        *d_output = dag_->getBuffer(output_buffer_ids_[0]);
        
        // Get actual output size from sink stage using name-based lookup
        auto sizes_by_name = output_nodes_[0]->stage->getActualOutputSizesByName();
        auto output_names = output_nodes_[0]->stage->getOutputNames();
        
        // Get first output's size
        *output_size = 0;
        if (!output_names.empty() && sizes_by_name.count(output_names[0])) {
            *output_size = sizes_by_name.at(output_names[0]);
        }
    }
    
    FZ_LOG(INFO, "Compress complete: %zu -> %zu bytes", input_size, *output_size);
}

void Pipeline::decompress(
    const void* d_input,
    size_t input_size,
    void** d_output,
    size_t* output_size,
    cudaStream_t stream
) {
    if (!is_finalized_) {
        throw std::runtime_error("Pipeline not finalized");
    }
    if (buffer_metadata_.empty()) {
        throw std::runtime_error("decompress() requires compress() to have been called first");
    }

    FZ_LOG(INFO, "Decompressing in-memory (buffer-id-aware inverse)");

    // Build InverseStageSpec list from the current (forward) DAG topology.
    // The buffer IDs recorded in each node precisely describe data flow — no
    // name heuristics needed.
    std::vector<InverseStageSpec> specs;
    for (const auto& level_nodes : dag_->getLevels()) {
        for (auto* node : level_nodes) {
            InverseStageSpec spec;
            spec.stage = node->stage;
            spec.fwd_input_ids.assign(node->input_buffer_ids.begin(),
                                      node->input_buffer_ids.end());
            spec.fwd_output_ids.assign(node->output_buffer_ids.begin(),
                                       node->output_buffer_ids.end());
            specs.push_back(std::move(spec));
        }
    }

    // Seed live_bufs with the pipeline's compressed output buffers
    std::unordered_map<int, std::pair<void*, size_t>> live_bufs;
    for (const auto& meta : buffer_metadata_) {
        live_bufs[meta.buffer_id] = { dag_->getBuffer(meta.buffer_id), meta.actual_size };
        FZ_LOG(DEBUG, "Compressed buffer id=%d '%s': %zu bytes",
               meta.buffer_id, meta.name.c_str(), meta.actual_size);
    }

    // Temporarily set all stages to inverse mode
    for (auto& s : stages_) s->setInverse(true);

    std::pair<void*, size_t> result;
    try {
        result = runInversePipeline(specs, live_bufs, input_size, *mem_pool_, stream);
    } catch (...) {
        for (auto& s : stages_) s->setInverse(false);
        throw;
    }
    for (auto& s : stages_) s->setInverse(false);

    // Copy from pool-managed buffer to a plain cudaMalloc buffer the caller can cudaFree
    void* d_final = nullptr;
    cudaError_t err = cudaMalloc(&d_final, result.second);
    if (err != cudaSuccess) {
        mem_pool_->free(result.first, stream);
        throw std::runtime_error("cudaMalloc for decompress output failed");
    }
    cudaMemcpyAsync(d_final, result.first, result.second, cudaMemcpyDeviceToDevice, stream);
    cudaStreamSynchronize(stream);
    mem_pool_->free(result.first, stream);

    *d_output = d_final;
    *output_size = result.second;
    FZ_LOG(INFO, "Decompress complete: %zu bytes", result.second);
}

void Pipeline::reset(cudaStream_t stream) {
    if (!is_finalized_) {
        return;
    }
    
    dag_->reset(stream);
    
    FZ_LOG(DEBUG, "Reset complete");
}

// ========== Query & Debug ==========

size_t Pipeline::getPeakMemoryUsage() const {
    return dag_->getPeakMemoryUsage();
}

size_t Pipeline::getCurrentMemoryUsage() const {
    return dag_->getCurrentMemoryUsage();
}

void Pipeline::printPipeline() const {
    std::cout << "\n========== Pipeline Configuration ==========\n";
    std::cout << "Stages: " << stages_.size() << "\n";
    std::cout << "Strategy: ";
    switch (strategy_) {
        case MemoryStrategy::MINIMAL: std::cout << "MINIMAL\n"; break;
        case MemoryStrategy::PIPELINE: std::cout << "PIPELINE\n"; break;
        case MemoryStrategy::PREALLOCATE: std::cout << "PREALLOCATE\n"; break;
    }
    std::cout << "Parallel streams: " << num_streams_ << "\n";
    std::cout << "Soft-run enabled: " << (soft_run_enabled_ ? "yes" : "no") << "\n";
    std::cout << "Finalized: " << (is_finalized_ ? "yes" : "no") << "\n";
    
    std::cout << "\nStages:\n";
    for (const auto& stage : stages_) {
        std::cout << "  - " << stage->getName() 
                  << " (inputs=" << stage->getNumInputs()
                  << ", outputs=" << stage->getNumOutputs() << ")\n";
    }
    
    std::cout << "\n";
    if (is_finalized_) {
        dag_->printDAG();
    }
    std::cout << "============================================\n\n";
}

// ========== Internal Implementation ==========

// ===== Finalize Helpers =====

std::pair<std::vector<Stage*>, std::vector<Stage*>> Pipeline::identifyTopology() {
    auto sources = getSourceStages();
    auto sinks = getSinkStages();
    
    if (sources.empty() || sinks.empty()) {
        throw std::runtime_error("Pipeline has no source or sink stages");
    }
    
    if (sources.size() != 1) {
        throw std::runtime_error("Pipeline must have exactly one source stage");
    }
    
    return {sources, sinks};
}

void Pipeline::setupInputBuffers(const std::vector<Stage*>& sources) {
    input_node_ = stage_to_node_[sources[0]];
    dag_->setInputBuffer(input_node_, 1, "pipeline_input");
    input_buffer_id_ = input_node_->input_buffer_ids.back();
}

int Pipeline::autoDetectUnconnectedOutputs() {
    int pipeline_outputs_added = 0;
    
    // Iterate ALL stages and convert unconnected outputs to pipeline outputs
    for (const auto& [stage, node] : stage_to_node_) {
        size_t num_outputs = stage->getNumOutputs();
        auto output_names = stage->getOutputNames();
        
        for (size_t i = 0; i < num_outputs; i++) {
            // Find the pre-allocated buffer for this output
            auto it = node->output_index_to_buffer_id.find(static_cast<int>(i));
            if (it == node->output_index_to_buffer_id.end()) {
                throw std::runtime_error("Missing pre-allocated buffer for output " + std::to_string(i));
            }
            
            int buffer_id = it->second;
            const auto& buffer_info = dag_->getBufferInfo(buffer_id);
            
            // Check if this output is connected to a downstream stage
            bool is_connected = !buffer_info.consumer_stage_ids.empty();
            
            if (!is_connected) {
                // Unconnected output → convert to pipeline output
                std::string output_name = i < output_names.size() ? output_names[i] : std::to_string(i);
                std::string tag = "pipeline_output_" + stage->getName() + "." + output_name;
                
                // Reuse pre-allocated buffer and update its properties
                dag_->updateBufferTag(buffer_id, tag);
                dag_->setBufferPersistent(buffer_id, true);
                
                output_buffer_ids_.push_back(buffer_id);
                output_nodes_.push_back(node);
                pipeline_outputs_added++;
                
                FZ_LOG(DEBUG, "Unconnected output -> pipeline output: %s.%s (buffer %d)",
                       stage->getName().c_str(), output_name.c_str(), buffer_id);
            }
        }
    }
    
    return pipeline_outputs_added;
}

void Pipeline::detectMultiOutputScenario(int pipeline_outputs) {
    size_t total_outputs = output_buffer_ids_.size();
    if (total_outputs > 1) {
        needs_concat_ = true;
        
        FZ_LOG(DEBUG, "Multiple pipeline outputs detected (%zu), will auto-concat", total_outputs);
    }
}

void Pipeline::configureStreamsIfNeeded() {
    if (num_streams_ > 1) {
        dag_->configureStreams(num_streams_);
    }
}

void Pipeline::propagateBufferSizes() {
    if (input_size_hint_ == 0) {
        FZ_LOG(DEBUG, "No input size hint, using placeholder buffer sizes");
        return;
    }
    
    dag_->updateBufferSize(input_buffer_id_, input_size_hint_);
    
    FZ_LOG(TRACE, "Propagating sizes through %zu levels", dag_->getLevels().size());
    
    for (const auto& level_nodes : dag_->getLevels()) {
        for (auto* node : level_nodes) {
            FZ_LOG(TRACE, "  Processing stage '%s' (id=%d, outputs=%zu)",
                   node->name.c_str(), node->id, node->output_buffer_ids.size());
            std::vector<size_t> input_sizes;
            for (int buf_id : node->input_buffer_ids) {
                size_t size = dag_->getBufferSize(buf_id);
                input_sizes.push_back(size);
                FZ_LOG(TRACE, "    Input buffer [%d] = %.1f KB", buf_id, size / 1024.0);
            }
            
            auto estimated_outputs = node->stage->estimateOutputSizes(input_sizes);
            
            for (size_t i = 0; i < node->output_buffer_ids.size() && i < estimated_outputs.size(); i++) {
                dag_->updateBufferSize(node->output_buffer_ids[i], estimated_outputs[i]);
                FZ_LOG(TRACE, "    Output buffer [%d] = %.1f KB",
                       node->output_buffer_ids[i], estimated_outputs[i] / 1024.0);
            }
        }
    }
    
    FZ_LOG(DEBUG, "Buffer sizes estimated from input hint (%.2f MB)",
           input_size_hint_ / (1024.0 * 1024.0));
}

void Pipeline::rebuildInverseConnections() {    
    FZ_LOG(DEBUG, "Rebuilding connections for inverse mode");
    FZ_LOG(DEBUG, "Original connections: %zu", connections_.size());
    
    // Recreate DAG (this clears connections but keeps stage nodes)
    dag_ = std::make_unique<CompressionDAG>(mem_pool_.get(), strategy_);
    stage_to_node_.clear();
    
    // Re-add all stages to DAG (stages are already in inverse mode)
    for (auto& stage_ptr : stages_) {
        Stage* stage = stage_ptr.get();
        DAGNode* node = dag_->addStage(stage, stage->getName());
        stage_to_node_[stage] = node;
        
        // Pre-allocate outputs for inverse stage
        size_t num_outputs = stage->getNumOutputs();
        auto output_names = stage->getOutputNames();
        
        for (size_t i = 0; i < num_outputs; i++) {
            std::string output_name = i < output_names.size() ? output_names[i] : std::to_string(i);
            std::string tag = stage->getName() + "." + output_name + "_unconnected";
            dag_->addUnconnectedOutput(node, 1, i, tag);
        }
    }
    
    // Reverse connections: producer → dependent becomes dependent → producer
    // In compression: connect(diff, lorenzo, "codes") means diff depends on lorenzo
    // In decompression: this reverses to lorenzo depends on diff
    for (auto it = connections_.rbegin(); it != connections_.rend(); ++it) {
        const auto& orig_conn = *it;
        
        // Swap roles: original producer becomes new dependent
        Stage* new_dependent = orig_conn.producer;
        Stage* new_producer = orig_conn.dependent;
        
        // In inverse mode, stages swap input/output counts
        // The output that was connected in forward mode should map to the 
        // corresponding input in reverse mode
        // For now, use default output name "output" for simple cases
        std::string new_output_name = "output";
        
        // Get nodes
        auto new_dep_node = stage_to_node_[new_dependent];
        auto new_prod_node = stage_to_node_[new_producer];
        
        // Get output index (use 0 for simple cases)
        int new_output_idx = new_producer->getOutputIndex(new_output_name);
        if (new_output_idx < 0) {
            new_output_idx = 0;  // Default to first output
        }
        
        // Connect
        dag_->connectExistingOutput(new_prod_node, new_dep_node, new_output_idx);
        
        FZ_LOG(TRACE, "Reversed: %s <- %s.%s ==> %s <- %s.%s",
               orig_conn.dependent->getName().c_str(),
               orig_conn.producer->getName().c_str(), orig_conn.output_name.c_str(),
               new_dependent->getName().c_str(),
               new_producer->getName().c_str(), new_output_name.c_str());
    }
    
    FZ_LOG(DEBUG, "Inverse connections rebuilt");
}
// =====

void Pipeline::validate() {
    if (stages_.empty()) {
        throw std::runtime_error("Pipeline has no stages");
    }
    
    // TODO: More comprehensive validation:
    // - Check for cycles (DAG will do this in assignLevels)
    // - Verify all stages have correct input/output counts
    // - Ensure there's at least one source and one sink
    
    FZ_LOG(DEBUG, "Validation passed");
}

std::vector<Stage*> Pipeline::getSourceStages() const {
    std::vector<Stage*> sources;
    
    for (const auto& [stage, node] : stage_to_node_) {
        if (node->dependencies.empty()) {
            sources.push_back(stage);
        }
    }
    
    return sources;
}

std::vector<Stage*> Pipeline::getSinkStages() const {
    std::vector<Stage*> sinks;
    
    for (const auto& [stage, node] : stage_to_node_) {
        if (node->dependents.empty()) {
            sinks.push_back(stage);
        }
    }
    
    return sinks;
}

// ===== Output Concatenation Helpers =====

std::vector<Pipeline::OutputBufferInfo> Pipeline::collectOutputBuffers() const {
    std::vector<OutputBufferInfo> outputs;
    
    for (int buffer_id : output_buffer_ids_) {
        const auto& buffer_info = dag_->getBufferInfo(buffer_id);
        
        // Find producer node
        DAGNode* producer_node = nullptr;
        for (const auto& node : dag_->getNodes()) {
            if (node->id == buffer_info.producer_stage_id) {
                producer_node = node;
                break;
            }
        }
        
        if (!producer_node) {
            throw std::runtime_error("Producer stage not found for buffer " + std::to_string(buffer_id));
        }
        
        // Get output name from index
        auto stage_output_names = producer_node->stage->getOutputNames();
        int output_idx = buffer_info.producer_output_index;
        std::string output_name = (output_idx >= 0 && output_idx < static_cast<int>(stage_output_names.size())) 
                                  ? stage_output_names[output_idx] 
                                  : "output";
        
        // Look up actual size by name
        auto sizes_by_name = producer_node->stage->getActualOutputSizesByName();
        size_t actual_size = 0;
        auto it = sizes_by_name.find(output_name);
        if (it != sizes_by_name.end()) {
            actual_size = it->second;
        }
        
        outputs.push_back({
            buffer_id,
            dag_->getBuffer(buffer_id),
            actual_size,
            producer_node->stage->getName(),
            output_name
        });
    }
    
    return outputs;
}

size_t Pipeline::calculateConcatSize(const std::vector<OutputBufferInfo>& outputs) const {
    size_t total_size = sizeof(uint32_t);  // num_buffers header
    for (const auto& output : outputs) {
        total_size += sizeof(uint64_t);  // size header per buffer
        total_size += output.actual_size; // actual data
    }
    return total_size;
}

size_t Pipeline::writeConcatBuffer(
    const std::vector<OutputBufferInfo>& outputs,
    uint8_t* d_concat_bytes,
    cudaStream_t stream
) const {
    size_t offset = 0;
    
    // Write num_buffers header
    uint32_t num_buffers = static_cast<uint32_t>(outputs.size());
    cudaMemcpyAsync(d_concat_bytes + offset, &num_buffers, sizeof(uint32_t),
                   cudaMemcpyHostToDevice, stream);
    offset += sizeof(uint32_t);
    
    // Write each buffer with size header
    for (const auto& output : outputs) {
        // Write size header
        uint64_t size_header = static_cast<uint64_t>(output.actual_size);
        cudaMemcpyAsync(d_concat_bytes + offset, &size_header, sizeof(uint64_t),
                       cudaMemcpyHostToDevice, stream);
        offset += sizeof(uint64_t);
        
        // Write buffer data
        if (output.actual_size > 0) {
            cudaMemcpyAsync(d_concat_bytes + offset, output.d_ptr, output.actual_size,
                           cudaMemcpyDeviceToDevice, stream);
            offset += output.actual_size;
        }
        
        FZ_LOG(TRACE, "Concat: %s - %.1f KB at offset %zu",
               output.stage_name.c_str(), output.actual_size / 1024.0, offset - output.actual_size);
    }
    
    return offset;
}

void Pipeline::concatOutputs(void** d_output, size_t* output_size, cudaStream_t stream) {
    // Collect all output information
    auto outputs = collectOutputBuffers();
    
    // Calculate total size needed
    size_t total_size = calculateConcatSize(outputs);
    
    // Allocate or reuse concat buffer
    if (d_concat_buffer_ == nullptr || concat_buffer_capacity_ < total_size) {
        if (d_concat_buffer_) {
            mem_pool_->free(d_concat_buffer_, stream);
        }
        
        d_concat_buffer_ = mem_pool_->allocate(total_size, stream, "concat_output", true);
        if (!d_concat_buffer_) {
            throw std::runtime_error("Failed to allocate concat buffer");
        }
        concat_buffer_capacity_ = total_size;
        
        FZ_LOG(DEBUG, "Allocated concat buffer: %.1f KB for %zu outputs",
               total_size / 1024.0, outputs.size());
    }
    
    // Write concatenated data
    *d_output = d_concat_buffer_;
    *output_size = writeConcatBuffer(outputs, static_cast<uint8_t*>(d_concat_buffer_), stream);
    
    FZ_LOG(DEBUG, "Concatenation complete: %zu buffers -> %.1f KB",
           outputs.size(), total_size / 1024.0);
}

// ========== File Serialization ==========

std::vector<Pipeline::OutputBuffer> Pipeline::getOutputBuffers() const {
    std::vector<OutputBuffer> buffers;
    
    for (const auto& meta : buffer_metadata_) {
        OutputBuffer buf;
        buf.d_ptr = dag_->getBuffer(meta.buffer_id);
        buf.actual_size = meta.actual_size;
        buf.allocated_size = meta.allocated_size;
        buf.name = meta.name;
        buf.buffer_id = meta.buffer_id;
        buffers.push_back(buf);
    }
    
    return buffers;
}

Pipeline::FZMFileHeader Pipeline::buildHeader() const {
    if (!is_finalized_) {
        throw std::runtime_error("Cannot build header before finalization");
    }
    
    if (buffer_metadata_.empty()) {
        throw std::runtime_error("Cannot build header before compress() is called");
    }
    
    FZMFileHeader fh;
    fh.core.uncompressed_size = input_size_;
    
    // Build stage information from DAG topology (in execution order)
    const auto& levels = dag_->getLevels();
    
    for (const auto& level : levels) {
        for (auto* node : level) {
            FZMStageInfo stage_info;
            stage_info.stage_type = static_cast<StageType>(node->stage->getStageTypeId());
            stage_info.stage_version = 1;
            stage_info.num_inputs = static_cast<uint8_t>(node->input_buffer_ids.size());
            stage_info.num_outputs = static_cast<uint8_t>(node->output_buffer_ids.size());
            
            for (size_t i = 0; i < node->input_buffer_ids.size() && i < FZM_MAX_STAGE_INPUTS; i++) {
                stage_info.input_buffer_ids[i] = static_cast<uint16_t>(node->input_buffer_ids[i]);
            }
            for (size_t i = 0; i < node->output_buffer_ids.size() && i < FZM_MAX_STAGE_OUTPUTS; i++) {
                stage_info.output_buffer_ids[i] = static_cast<uint16_t>(node->output_buffer_ids[i]);
            }
            
            if (stage_info.num_outputs > 0) {
                stage_info.config_size = static_cast<uint32_t>(
                    node->stage->serializeHeader(0, stage_info.stage_config, FZM_STAGE_CONFIG_SIZE)
                );
            }
            fh.stages.push_back(stage_info);
        }
    }
    fh.core.num_stages = static_cast<uint32_t>(fh.stages.size());
    
    // Build buffer entries
    fh.core.num_buffers = static_cast<uint16_t>(buffer_metadata_.size());
    uint64_t byte_offset = 0;
    
    for (uint16_t i = 0; i < buffer_metadata_.size(); i++) {
        const auto& meta = buffer_metadata_[i];
        FZMBufferEntry entry;
        
        entry.stage_type = static_cast<StageType>(meta.producer->stage->getStageTypeId());
        entry.stage_version = 1;
        entry.data_type = static_cast<DataType>(
            meta.producer->stage->getOutputDataType(meta.output_index)
        );
        entry.producer_output_idx = static_cast<uint8_t>(meta.output_index);
        entry.dag_buffer_id = static_cast<uint16_t>(meta.buffer_id);  // for inverse routing
        strncpy(entry.name, meta.name.c_str(), FZM_MAX_NAME_LEN - 1);
        entry.name[FZM_MAX_NAME_LEN - 1] = '\0';
        entry.data_size = meta.actual_size;
        entry.allocated_size = meta.allocated_size;
        entry.uncompressed_size = meta.actual_size;
        entry.byte_offset = byte_offset;
        entry.config_size = static_cast<uint32_t>(
            meta.producer->stage->serializeHeader(meta.output_index, entry.stage_config, FZM_STAGE_CONFIG_SIZE)
        );
        
        byte_offset += meta.actual_size;
        fh.buffers.push_back(entry);
    }
    
    fh.core.compressed_size = byte_offset;
    fh.core.header_size = fh.core.computeHeaderSize();
    
    FZ_LOG(INFO, "Built FZM header: %u stages, %u buffers, %.2f MB compressed, header %llu bytes",
           fh.core.num_stages, fh.core.num_buffers,
           fh.core.compressed_size / (1024.0 * 1024.0),
           (unsigned long long)fh.core.header_size);
    
    return fh;
}

void Pipeline::writeToFile(const std::string& filename, cudaStream_t stream) {
    if (!is_finalized_) {
        throw std::runtime_error("Cannot write to file before finalization");
    }
    
    if (buffer_metadata_.empty()) {
        throw std::runtime_error("Cannot write to file before compress() is called");
    }
    
    FZMFileHeader fh = buildHeader();
    
    size_t total_data_size = fh.core.compressed_size;
    void* h_data = malloc(total_data_size);
    if (!h_data) {
        throw std::runtime_error("Failed to allocate host buffer for file write");
    }
    
    size_t offset = 0;
    for (const auto& meta : buffer_metadata_) {
        void* d_buffer = dag_->getBuffer(meta.buffer_id);
        cudaError_t err = cudaMemcpyAsync(
            static_cast<uint8_t*>(h_data) + offset,
            d_buffer, meta.actual_size,
            cudaMemcpyDeviceToHost, stream
        );
        if (err != cudaSuccess) {
            free(h_data);
            throw std::runtime_error("cudaMemcpyAsync failed: " + std::string(cudaGetErrorString(err)));
        }
        offset += meta.actual_size;
    }
    
    cudaError_t err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        free(h_data);
        throw std::runtime_error("cudaStreamSynchronize failed: " + std::string(cudaGetErrorString(err)));
    }
    
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        free(h_data);
        throw std::runtime_error("Failed to open file for writing: " + filename);
    }
    
    // Write header core + stage array + buffer array
    file.write(reinterpret_cast<const char*>(&fh.core), sizeof(FZMHeaderCore));
    if (!fh.stages.empty()) {
        file.write(reinterpret_cast<const char*>(fh.stages.data()),
                   fh.stages.size() * sizeof(FZMStageInfo));
    }
    if (!fh.buffers.empty()) {
        file.write(reinterpret_cast<const char*>(fh.buffers.data()),
                   fh.buffers.size() * sizeof(FZMBufferEntry));
    }
    if (!file) {
        free(h_data);
        throw std::runtime_error("Failed to write header to file");
    }
    
    file.write(reinterpret_cast<const char*>(h_data), total_data_size);
    if (!file) {
        free(h_data);
        throw std::runtime_error("Failed to write data to file");
    }
    
    file.close();
    free(h_data);
    
    size_t total_file_size = fh.core.header_size + total_data_size;
    FZ_LOG(INFO, "Wrote %.2f MB to %s (Header: %llu bytes, Data: %.2f MB)",
           total_file_size / (1024.0 * 1024.0), filename.c_str(),
           (unsigned long long)fh.core.header_size,
           total_data_size / (1024.0 * 1024.0));
}

Pipeline::FZMFileHeader Pipeline::readHeader(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file for reading: " + filename);
    }
    
    FZMFileHeader fh;
    file.read(reinterpret_cast<char*>(&fh.core), sizeof(FZMHeaderCore));
    if (!file) {
        throw std::runtime_error("Failed to read header from file: " + filename);
    }
    
    if (fh.core.magic != FZM_MAGIC) {
        throw std::runtime_error("Invalid FZM file format (bad magic number)");
    }
    
    if (fh.core.version != FZM_VERSION) {
        throw std::runtime_error(
            "Unsupported FZM version: " + std::to_string(fh.core.version) +
            " (expected " + std::to_string(FZM_VERSION) + ")"
        );
    }
    
    if (fh.core.num_stages > FZM_MAX_BUFFERS || fh.core.num_buffers > FZM_MAX_BUFFERS) {
        throw std::runtime_error("FZM header has too many stages/buffers");
    }
    
    // Read stage array
    fh.stages.resize(fh.core.num_stages);
    if (fh.core.num_stages > 0) {
        file.read(reinterpret_cast<char*>(fh.stages.data()),
                  fh.core.num_stages * sizeof(FZMStageInfo));
        if (!file) {
            throw std::runtime_error("Failed to read stage data from file");
        }
    }
    
    // Read buffer array
    fh.buffers.resize(fh.core.num_buffers);
    if (fh.core.num_buffers > 0) {
        file.read(reinterpret_cast<char*>(fh.buffers.data()),
                  fh.core.num_buffers * sizeof(FZMBufferEntry));
        if (!file) {
            throw std::runtime_error("Failed to read buffer data from file");
        }
    }
    
    FZ_LOG(INFO, "Read FZM header from %s (v%u, %.2f MB uncompressed, %.2f MB compressed, %u stages, %u buffers)",
           filename.c_str(), fh.core.version,
           fh.core.uncompressed_size / (1024.0 * 1024.0),
           fh.core.compressed_size / (1024.0 * 1024.0),
           fh.core.num_stages, fh.core.num_buffers);
    
    file.close();
    return fh;
}

void* Pipeline::loadCompressedData(
    const std::string& filename,
    const FZMFileHeader& fh,
    cudaStream_t stream,
    MemoryPool* pool
) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file for reading: " + filename);
    }
    
    // Seek past the variable-length header using header_size
    file.seekg(fh.core.header_size, std::ios::beg);
    if (!file) {
        throw std::runtime_error("Failed to seek past header in file");
    }
    
    size_t data_size = fh.core.compressed_size;
    void* h_data = malloc(data_size);
    if (!h_data) {
        throw std::runtime_error("Failed to allocate host buffer for compressed data");
    }
    
    file.read(reinterpret_cast<char*>(h_data), data_size);
    if (!file) {
        free(h_data);
        throw std::runtime_error("Failed to read compressed data from file");
    }
    file.close();
    
    void* d_data = nullptr;
    cudaError_t err;
    if (pool) {
        d_data = pool->allocate(data_size, stream, "compressed_data_load");
        if (!d_data) {
            free(h_data);
            throw std::runtime_error("MemoryPool::allocate failed for compressed data load");
        }
    } else {
        err = cudaMalloc(&d_data, data_size);
        if (err != cudaSuccess) {
            free(h_data);
            throw std::runtime_error("cudaMalloc failed: " + std::string(cudaGetErrorString(err)));
        }
    }

    err = cudaMemcpyAsync(d_data, h_data, data_size, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        if (pool) pool->free(d_data, stream);
        else cudaFree(d_data);
        free(h_data);
        throw std::runtime_error("cudaMemcpyAsync failed: " + std::string(cudaGetErrorString(err)));
    }

    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        if (pool) pool->free(d_data, stream);
        else cudaFree(d_data);
        free(h_data);
        throw std::runtime_error("cudaStreamSynchronize failed: " + std::string(cudaGetErrorString(err)));
    }
    
    free(h_data);
    
    FZ_LOG(DEBUG, "Loaded %.2f MB compressed data to GPU", data_size / (1024.0 * 1024.0));
    
    return d_data;
}

void Pipeline::decompressFromFile(
    const std::string& filename,
    void** d_output,
    size_t* output_size,
    cudaStream_t stream
) {
    FZ_LOG(INFO, "Decompressing from file: %s", filename.c_str());

    // 1. Read header
    FZMFileHeader fh = readHeader(filename);
    FZ_LOG(DEBUG, "Header: %u stages, %u buffers, %.2f MB compressed, %.2f MB uncompressed",
           fh.core.num_stages, fh.core.num_buffers,
           fh.core.compressed_size / (1024.0 * 1024.0),
           fh.core.uncompressed_size / (1024.0 * 1024.0));

    // 2. Local memory pool for GPU temporaries
    MemoryPoolConfig local_pool_cfg(
        fh.core.uncompressed_size + fh.core.compressed_size, 2.0f);
    MemoryPool local_pool(local_pool_cfg);

    // 3. Load compressed blob to GPU
    void* d_compressed = loadCompressedData(filename, fh, stream, &local_pool);

    // 4. Build InverseStageSpec list from file header (forward order).
    //    FZMStageInfo records the exact forward DAG buffer IDs for each stage,
    //    so we can route buffers precisely without name heuristics.
    std::vector<std::unique_ptr<Stage>> owned_stages;
    std::vector<InverseStageSpec> specs;

    for (uint32_t i = 0; i < fh.core.num_stages; i++) {
        const auto& si = fh.stages[i];
        Stage* stage = createStage(si.stage_type, si.stage_config, si.config_size);
        stage->setInverse(true);
        owned_stages.emplace_back(stage);

        FZ_LOG(DEBUG, "Reconstructed inverse stage %u: %s (config_size=%u)",
               i, stageTypeToString(si.stage_type).c_str(), si.config_size);

        InverseStageSpec spec;
        spec.stage = stage;
        for (uint8_t j = 0; j < si.num_inputs && j < FZM_MAX_STAGE_INPUTS; j++) {
            uint16_t id = si.input_buffer_ids[j];
            if (id != 0xFFFF) spec.fwd_input_ids.push_back(static_cast<int>(id));
        }
        for (uint8_t j = 0; j < si.num_outputs && j < FZM_MAX_STAGE_OUTPUTS; j++) {
            uint16_t id = si.output_buffer_ids[j];
            if (id != 0xFFFF) spec.fwd_output_ids.push_back(static_cast<int>(id));
        }
        specs.push_back(std::move(spec));
    }

    // 5. Seed live_bufs with file buffers.
    //    Each FZMBufferEntry records the original DAG buffer ID (dag_buffer_id) so we
    //    can key directly into live_bufs — no name-based matching required.
    std::unordered_map<int, std::pair<void*, size_t>> live_bufs;
    for (uint16_t i = 0; i < fh.core.num_buffers; i++) {
        const auto& entry = fh.buffers[i];
        void* d_buf = static_cast<uint8_t*>(d_compressed) + entry.byte_offset;
        int buf_id = static_cast<int>(entry.dag_buffer_id);
        live_bufs[buf_id] = { d_buf, static_cast<size_t>(entry.data_size) };
        FZ_LOG(DEBUG, "File buffer[%u] '%s': dag_buf_id=%d, %.2f KB",
               i, entry.name, buf_id, entry.data_size / 1024.0);
    }

    // 6. Run unified inverse pipeline
    std::pair<void*, size_t> result;
    try {
        result = runInversePipeline(specs, live_bufs,
                                    fh.core.uncompressed_size, local_pool, stream);
    } catch (...) {
        local_pool.free(d_compressed, stream);
        throw;
    }

    // 7. Copy from pool buffer to a plain cudaMalloc buffer the caller can cudaFree
    size_t final_size = result.second > 0 ? result.second : fh.core.uncompressed_size;
    void* d_final = nullptr;
    cudaError_t err = cudaMalloc(&d_final, final_size);
    if (err != cudaSuccess) {
        local_pool.free(d_compressed, stream);
        throw std::runtime_error("cudaMalloc for decompressed output failed");
    }
    cudaMemcpyAsync(d_final, result.first, final_size, cudaMemcpyDeviceToDevice, stream);
    cudaStreamSynchronize(stream);
    local_pool.free(d_compressed, stream);

    *d_output = d_final;
    *output_size = final_size;
    FZ_LOG(INFO, "Decompression complete: %.2f MB -> %zu bytes",
           fh.core.compressed_size / (1024.0 * 1024.0), final_size);
}



} // namespace fz

