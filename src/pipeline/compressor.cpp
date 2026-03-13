#include "pipeline/compressor.h"
#include "fzm_format.h"
#include "log.h"
#include "cuda_check.h"
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <algorithm>
#include <cstdint>
#include <unordered_set>
#include <vector>

namespace fz {

Pipeline::Pipeline(size_t input_data_size, MemoryStrategy strategy, float pool_multiplier)
    : strategy_(strategy),
      num_streams_(1),
      is_finalized_(false),
      soft_run_enabled_(false),
      is_compressed_(false),
      was_compressed_(false),
      profiling_enabled_(false),
      d_concat_buffer_(nullptr),
      concat_buffer_capacity_(0),
      needs_concat_(false),
      input_size_(0),
      input_size_hint_(input_data_size),
      input_alignment_bytes_(1),
      d_pad_buf_(nullptr),
      d_pad_buf_size_(0),
      original_input_size_(0),
      dims_({0, 1, 1}) {
    
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
    if (d_pad_buf_) {
        cudaFree(d_pad_buf_);
        d_pad_buf_ = nullptr;
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

void Pipeline::enableProfiling(bool enable) {
    profiling_enabled_ = enable;
    dag_->enableProfiling(enable);
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
        (void)buffer_id;
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
    
    // Push current dims to all stages so late setDims() calls are honoured
    for (const auto& stage_ptr : stages_) {
        stage_ptr->setDims(dims_);
    }

    validate();
    
    auto [sources, sinks] = identifyTopology();
    setupInputBuffers(sources);
    
    // Unified output detection: ALL unconnected outputs become pipeline outputs
    int pipeline_outputs = autoDetectUnconnectedOutputs();
    detectMultiOutputScenario(pipeline_outputs);
    
    configureStreamsIfNeeded();
    
    dag_->finalize();

    // ── Compute required input alignment ─────────────────────────────────────
    // Walk all stages and take the LCM of their alignment requirements.
    // Chunked stages (Bitshuffle, Difference, RZE) return their chunk size;
    // other stages return 1 (no requirement).  compress() will transparently
    // zero-pad the user's input to this boundary when needed.
    // IMPORTANT: this must run BEFORE propagateBufferSizes() so that the
    // rounded-up hint drives all downstream buffer size estimates.
    {
        size_t align = 1;
        int chunked_count = 0;
        for (const auto& stage_ptr : stages_) {
            const size_t a = stage_ptr->getRequiredInputAlignment();
            if (a > 1) { align = std::lcm(align, a); ++chunked_count; }
        }
        input_alignment_bytes_ = align;
        if (align > 1) {
            // Round up the buffer-sizing hint so that propagateBufferSizes()
            // allocates intermediate buffers large enough for the padded input,
            // and so the compress() guard check doesn't reject it.
            input_size_hint_ = ((input_size_hint_ + align - 1) / align) * align;
            FZ_LOG(INFO,
                "Input alignment: %zu bytes (LCM of %d chunked stage(s)); "
                "buffer hint rounded up to %zu bytes",
                align, chunked_count, input_size_hint_);
        }
    }

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

void Pipeline::reset(cudaStream_t stream) {
    if (!is_finalized_) {
        return;
    }

    dag_->reset(stream);
    was_compressed_ = false;

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
    
    return {sources, sinks};
}

void Pipeline::setupInputBuffers(const std::vector<Stage*>& sources) {
    input_nodes_.clear();
    input_buffer_ids_.clear();
    for (size_t i = 0; i < sources.size(); i++) {
        DAGNode* src_node = stage_to_node_[sources[i]];
        std::string tag = sources.size() == 1
            ? "pipeline_input"
            : "pipeline_input_" + std::to_string(i) + "_" + sources[i]->getName();
        dag_->setInputBuffer(src_node, 1, tag);
        input_nodes_.push_back(src_node);
        input_buffer_ids_.push_back(src_node->input_buffer_ids.back());
    }
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

        // Remove stale pre-allocated buffers for output indices that no longer
        // exist (e.g. stage's getNumOutputs() decreased after addStage() due to
        // a config change like setInplaceOutliers(true)).  These "zombie" buffers
        // would otherwise appear in node->output_buffer_ids and confuse
        // buildInverseDAG(), which expects every fwd output to be either an
        // intermediate buffer (consumed by another stage) or a pipeline output.
        {
            std::vector<int> stale_indices;
            for (const auto& [out_idx, buf_id] : node->output_index_to_buffer_id) {
                if (static_cast<size_t>(out_idx) >= num_outputs)
                    stale_indices.push_back(out_idx);
            }
            for (int out_idx : stale_indices) {
                int buf_id = node->output_index_to_buffer_id.at(out_idx);
                node->output_buffer_ids.erase(
                    std::remove(node->output_buffer_ids.begin(),
                                node->output_buffer_ids.end(), buf_id),
                    node->output_buffer_ids.end());
                node->output_index_to_buffer_id.erase(out_idx);
                FZ_LOG(DEBUG,
                    "autoDetect: removed stale output buf %d from stage '%s'"
                    " (idx=%d >= num_outputs=%zu)",
                    buf_id, stage->getName().c_str(), out_idx, num_outputs);
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

void Pipeline::propagateBufferSizes(bool force_from_current_inputs) {
    // Check whether any static hint is available
    bool has_hint = (input_size_hint_ > 0) || !per_source_hints_.empty();
    if (!has_hint && !force_from_current_inputs) {
        FZ_LOG(DEBUG, "No input size hint, using placeholder buffer sizes");
        return;
    }

    if (!force_from_current_inputs) {
        // Seed each source's external input buffer with its per-source hint,
        // falling back to the global constructor hint when none is set.
        for (size_t i = 0; i < input_nodes_.size(); i++) {
            Stage* src_stage = input_nodes_[i]->stage;
            auto it = per_source_hints_.find(src_stage);
            size_t hint = (it != per_source_hints_.end()) ? it->second : input_size_hint_;
            if (hint > 0) {
                dag_->updateBufferSize(input_buffer_ids_[i], hint);
            }
        }
    }

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
    
    if (force_from_current_inputs) {
        FZ_LOG(DEBUG, "Buffer sizes estimated from runtime input sizes");
    } else {
        FZ_LOG(DEBUG, "Buffer sizes estimated from input hint (%.2f MB)",
               input_size_hint_ / (1024.0 * 1024.0));
    }
}

void Pipeline::validate() {
    if (stages_.empty()) {
        throw std::runtime_error("Pipeline has no stages");
    }

    // E7: If any connect() calls were made, every stage must participate in at
    // least one connection as either a producer or a consumer.  A stage that
    // was added to the pipeline but never connected while others are connected
    // is almost certainly a programming error (forgot to wire it up).
    if (!connections_.empty()) {
        std::unordered_set<Stage*> connected_stages;
        for (const auto& conn : connections_) {
            connected_stages.insert(conn.dependent);
            connected_stages.insert(conn.producer);
        }
        for (const auto& stage_ptr : stages_) {
            Stage* s = stage_ptr.get();
            if (connected_stages.find(s) == connected_stages.end()) {
                throw std::runtime_error(
                    "Stage '" + s->getName() + "' has no connections; "
                    "every stage in a connected pipeline must be linked "
                    "to at least one other stage via connect()");
            }
        }
    }

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
} // namespace fz
