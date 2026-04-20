#include "pipeline/compressor.h"
#include "pipeline/concat_kernel.h"
#include "fzm_format.h"
#include "log.h"
#include "cuda_check.h"
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <unordered_set>
#include <vector>

namespace fz {

Pipeline::Pipeline(size_t input_data_size, MemoryStrategy strategy, float pool_multiplier)
    : strategy_(strategy),
      num_streams_(1),
      is_finalized_(false),
      warmup_on_finalize_(false),
      pool_managed_decomp_(false),
      is_compressed_(false),
      was_compressed_(false),
      profiling_enabled_(false),
      d_concat_buffer_(nullptr),
      concat_buffer_capacity_(0),
      needs_concat_(false),
      h_concat_header_(nullptr),
      h_concat_header_capacity_(0),
      h_copy_descs_(nullptr),
      d_copy_descs_(nullptr),
      copy_descs_capacity_(0),
      input_size_(0),
      input_alignment_bytes_(1),
      d_pad_buf_(nullptr),
      d_pad_buf_size_(0),
      original_input_size_(0),
      input_size_hint_(input_data_size),
      pool_multiplier_(pool_multiplier),
      dims_({0, 1, 1}),
      graph_mode_enabled_(false),
      graph_captured_(false),
      d_graph_input_(nullptr),
      d_graph_input_size_(0),
      captured_graph_(nullptr),
      graph_exec_(nullptr) {
    
    MemoryPoolConfig pool_config(input_data_size, pool_multiplier);
    mem_pool_ = std::make_unique<MemoryPool>(pool_config);
    
    dag_ = std::make_unique<CompressionDAG>(mem_pool_.get(), strategy);
}

Pipeline::~Pipeline() {
    // Destroy CUDA Graph handles before the pool is torn down.
    if (graph_exec_)     { cudaGraphExecDestroy(graph_exec_);  graph_exec_     = nullptr; }
    if (captured_graph_) { cudaGraphDestroy(captured_graph_);  captured_graph_ = nullptr; }
    if (d_graph_input_ && mem_pool_) {
        mem_pool_->free(d_graph_input_, 0);
        d_graph_input_ = nullptr;
    }

    // Free pool-managed buffers before pool is destroyed
    if (d_concat_buffer_ && mem_pool_) {
        mem_pool_->free(d_concat_buffer_, 0);
        d_concat_buffer_ = nullptr;
    }
    for (void* p : d_decomp_outputs_) {
        if (p && mem_pool_) mem_pool_->free(p, 0);
    }
    d_decomp_outputs_.clear();
    if (d_pad_buf_ && mem_pool_) {
        mem_pool_->free(d_pad_buf_, 0);
        d_pad_buf_ = nullptr;
    }
    if (h_concat_header_) {
        cudaFreeHost(h_concat_header_);
        h_concat_header_ = nullptr;
    }
    if (h_copy_descs_) {
        cudaFreeHost(h_copy_descs_);
        h_copy_descs_ = nullptr;
    }
    if (d_copy_descs_) {
        cudaFree(d_copy_descs_);
        d_copy_descs_ = nullptr;
    }
}

void Pipeline::setMemoryStrategy(MemoryStrategy strategy) {
    if (is_finalized_) {
        throw std::runtime_error("Cannot change strategy after finalization");
    }
    strategy_ = strategy;
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

void Pipeline::enableGraphMode(bool enable) {
    if (is_finalized_) {
        throw std::runtime_error("Cannot change graph mode after finalization");
    }
    graph_mode_enabled_ = enable;
}

int Pipeline::connect(Stage* dependent, Stage* producer, const std::string& output_name) {
    if (is_finalized_) {
        throw std::runtime_error("Cannot connect stages after finalization");
    }
    
    auto dep_it = stage_to_node_.find(dependent);
    auto prod_it = stage_to_node_.find(producer);
    
    if (dep_it == stage_to_node_.end() || prod_it == stage_to_node_.end()) {
        throw std::runtime_error("Cannot connect stages not added to pipeline");
    }
    
    DAGNode* dep_node = dep_it->second;
    DAGNode* prod_node = prod_it->second;
    
    int output_index = producer->getOutputIndex(output_name);
    if (output_index < 0) {
        throw std::runtime_error(
            "Stage '" + producer->getName() + "' does not have output '" + output_name + "'");
    }
    
    ConnectionInfo conn_info;
    conn_info.dependent = dependent;
    conn_info.producer = producer;
    conn_info.output_name = output_name;
    conn_info.output_index = output_index;
    connections_.push_back(conn_info);
    
    bool connected = dag_->connectExistingOutput(prod_node, dep_node, output_index);
    
    if (!connected) {
        int buffer_id = dag_->addDependency(dep_node, prod_node, 1, output_index);
        (void)buffer_id;
        FZ_LOG(WARN, "Had to create new buffer for %s.%s (should have been pre-allocated)",
               producer->getName().c_str(), output_name.c_str());
    }
    
    FZ_LOG(DEBUG, "Connected %s.%s -> %s",
           producer->getName().c_str(), output_name.c_str(), dependent->getName().c_str());
    
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
    
    auto stage_ptr = std::unique_ptr<Stage>(stage);
    
    DAGNode* node = dag_->addStage(stage, stage->getName());
    
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
    
    for (const auto& stage_ptr : stages_) {
        stage_ptr->setDims(dims_);
        auto it = stage_to_node_.find(stage_ptr.get());
        if (it != stage_to_node_.end() && it->second) {
            it->second->name = stage_ptr->getName();
        }
    }

    validate();

    // ── Connection type-checking ──────────────────────────────────────────────
    // Verify that each connection's producer output type is compatible with the
    // consumer input type.  Byte-transparent stages (Bitshuffle, RZE) return
    // DataType::UNKNOWN and are silently skipped.
    //
    // Compatibility rule:
    //   • Exact match → always OK.
    //   • Same element size + both integral → OK.  DifferenceStage accepts
    //     signed-reinterpreted unsigned input (uint16 ↔ int16, uint32 ↔ int32)
    //     which is safe because the kernel just reinterprets the bit pattern.
    //   • Float ↔ integer of any size → error (semantically wrong).
    //   • Different sizes → error (reads wrong number of elements).
    {
        auto typesCompatible = [](uint8_t a, uint8_t b) -> bool {
            if (a == b) return true;
            const auto da = static_cast<DataType>(a);
            const auto db = static_cast<DataType>(b);
            const bool a_float = (da == DataType::FLOAT32 || da == DataType::FLOAT64);
            const bool b_float = (db == DataType::FLOAT32 || db == DataType::FLOAT64);
            if (a_float || b_float) return false;  // float↔int is always wrong
            return getDataTypeSize(da) == getDataTypeSize(db);
        };

        constexpr uint8_t kUnknown = static_cast<uint8_t>(DataType::UNKNOWN);
        for (const auto& conn : connections_) {
            const uint8_t prod_type = conn.producer->getOutputDataType(
                static_cast<size_t>(conn.output_index));
            // Single-input stages use port 0; multi-input merge stages return
            // UNKNOWN and are skipped automatically.
            const uint8_t cons_type = conn.dependent->getInputDataType(0);

            if (prod_type == kUnknown || cons_type == kUnknown) continue;

            if (!typesCompatible(prod_type, cons_type)) {
                throw std::runtime_error(
                    "Type mismatch: '" + conn.producer->getName() +
                    "' output '" + conn.output_name + "' has type " +
                    dataTypeToString(static_cast<DataType>(prod_type)) +
                    " but '" + conn.dependent->getName() + "' expects " +
                    dataTypeToString(static_cast<DataType>(cons_type)));
            }
        }
    }

    auto [sources, sinks] = identifyTopology();
    setupInputBuffers(sources);
    
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

    // ── Topology-aware pool sizing ────────────────────────────────────────────
    // Replace the blunt input_size*multiplier estimate from construction time
    // with one derived from the actual DAG buffer layout:
    //   PREALLOCATE → sum of all non-external buffers (all live simultaneously)
    //   MINIMAL → peak concurrent live bytes across execution levels
    // pool_multiplier_ is applied on top as headroom for stage-internal scratch
    // (CUB temp storage, RZE per-chunk scratch, etc.).
    // Skip when there is no input size hint — buffer sizes will be 1-byte
    // placeholders and the topology-derived value would be meaningless.
    if (input_size_hint_ > 0 || !per_source_hints_.empty()) {
        // computeTopoPoolSize() already accounts for all intermediate buffers
        // and persistent stage scratch, so it is the accurate peak requirement.
        // Apply a small safety margin (10%) for transient CUB allocations that
        // happen inside execute() calls but are too short-lived to appear in the
        // topo simulation.  The old pool_multiplier_ (typically 3x) is kept for
        // the initial pool construction at Pipeline() time only, where we have
        // no topology yet and must size conservatively from input_size alone.
        constexpr float kTopoSafetyMargin = 1.1f;
        const size_t topo_base  = dag_->computeTopoPoolSize();
        // d_pad_buf_ is allocated from the pool at the first compress() call
        // (size = input_size_hint_, already rounded to alignment boundary).
        // It is persistent and live throughout every execution, so it must be
        // added to the topo estimate even though it never appears in buffers_.
        // Only add it when chunked stages require alignment (otherwise no pad
        // buffer is ever created).
        const size_t pad_bytes  = (input_alignment_bytes_ > 1) ? input_size_hint_ : 0;
        const size_t topo_sized = static_cast<size_t>((topo_base + pad_bytes) * kTopoSafetyMargin);
        // Pin the CUDA pool release threshold to UINT64_MAX regardless of
        // strategy.  The topo-derived value tells CUDA the *capacity* target,
        // but a low threshold causes the driver to trim freed pages back to the
        // OS between compress() calls, forcing re-page-faulting on the next
        // call — visible as random latency spikes in MINIMAL mode.
        // UINT64_MAX means "never trim": pages freed inside a call stay warm
        // in the pool for the next call.  This does NOT change the peak
        // in-flight footprint (MINIMAL still allocates/frees lazily); it only
        // keeps the OS-level pages hot across calls.
        mem_pool_->setReleaseThreshold(std::numeric_limits<uint64_t>::max());
        FZ_LOG(INFO, "Pool threshold: %.2f MB (topo %.2f MB + pad %.2f MB, ×1.1 margin) "
               "[release threshold pinned to UINT64_MAX]",
               topo_sized / (1024.0 * 1024.0),
               topo_base  / (1024.0 * 1024.0),
               pad_bytes  / (1024.0 * 1024.0));
    }

    if (strategy_ == MemoryStrategy::PREALLOCATE) {
        dag_->preallocateBuffers();
    }

    // ── Graph mode: validate requirements and allocate fixed input slot ───────
    if (graph_mode_enabled_) {
        if (strategy_ != MemoryStrategy::PREALLOCATE) {
            throw std::runtime_error(
                "Graph mode requires PREALLOCATE memory strategy. "
                "Call setMemoryStrategy(MemoryStrategy::PREALLOCATE) before finalize().");
        }
        if (input_size_hint_ == 0 && per_source_hints_.empty()) {
            throw std::runtime_error(
                "Graph mode requires a non-zero input size hint. "
                "Pass input_data_size to the Pipeline constructor.");
        }
        if (input_nodes_.size() != 1) {
            throw std::runtime_error(
                "Graph mode currently supports single-source pipelines only "
                "(" + std::to_string(input_nodes_.size()) + " source(s) detected).");
        }

        // The graph's input pointer must be stable across launches, so allocate
        // a fixed device buffer at the padded hint size.  compress() will D2D-copy
        // the user's input here before calling cudaGraphLaunch().
        const size_t padded = (input_alignment_bytes_ > 1)
            ? ((input_size_hint_ + input_alignment_bytes_ - 1) / input_alignment_bytes_)
              * input_alignment_bytes_
            : input_size_hint_;

        if (d_graph_input_) { mem_pool_->free(d_graph_input_, 0); d_graph_input_ = nullptr; }
        d_graph_input_ = mem_pool_->allocate(padded, 0, "graph_input_slot", /*persistent=*/true);
        if (!d_graph_input_) {
            throw std::runtime_error(
                "Failed to allocate graph input slot (" + std::to_string(padded) +
                " bytes); pool may be exhausted — increase pool_size_multiplier or input_data_size");
        }
        d_graph_input_size_ = padded;
        FZ_LOG(INFO, "Graph mode: fixed input slot allocated (%.2f MB)",
               padded / (1024.0 * 1024.0));
    }

    // Pre-allocate input padding scratch from the memory pool so the first
    // compress() call does not pay a raw cudaMalloc path for alignment padding.
    if (input_alignment_bytes_ > 1 && input_size_hint_ > 0) {
        const size_t padded = ((input_size_hint_ + input_alignment_bytes_ - 1)
                               / input_alignment_bytes_) * input_alignment_bytes_;
        if (padded > d_pad_buf_size_) {
            if (d_pad_buf_) {
                mem_pool_->free(d_pad_buf_, /*stream=*/0);
                d_pad_buf_ = nullptr;
                d_pad_buf_size_ = 0;
            }
            d_pad_buf_ = mem_pool_->allocate(padded, /*stream=*/0,
                                             "pipeline_input_pad", /*persistent=*/true);
            if (!d_pad_buf_) {
                throw std::runtime_error(
                    "Failed to preallocate pipeline input pad buffer (" +
                    std::to_string(padded) + " bytes); pool may be exhausted");
            }
            d_pad_buf_size_ = padded;
            mem_pool_->synchronize(/*stream=*/0);
        }
    }

    num_streams_ = std::max(1, static_cast<int>(dag_->getStreamCount()));

    // Pre-allocate pinned concat header buffer.
    // Size = sizeof(uint32_t) + num_outputs * sizeof(uint64_t).
    // We use output_buffer_ids_.size() as the output count; if it is still 0
    // here (single-sink, no concat needed) we skip — needs_concat_ is false.
    if (needs_concat_ && !output_buffer_ids_.empty()) {
        const size_t n_outputs = output_buffer_ids_.size();

        // Header buffer: [num_buffers: u32][size_0..N-1: u64]
        const size_t hdr_bytes  = sizeof(uint32_t) + n_outputs * sizeof(uint64_t);
        if (h_concat_header_ == nullptr || h_concat_header_capacity_ < hdr_bytes) {
            if (h_concat_header_) cudaFreeHost(h_concat_header_);
            FZ_CUDA_CHECK(cudaHostAlloc(&h_concat_header_, hdr_bytes, cudaHostAllocDefault));
            h_concat_header_capacity_ = hdr_bytes;
        }

        // Gather kernel descriptor buffers: one CopyDesc per output segment.
        // Pinned host buffer for CPU packing; device mirror for kernel access.
        const size_t desc_bytes = n_outputs * sizeof(CopyDesc);
        if (h_copy_descs_ == nullptr || copy_descs_capacity_ < desc_bytes) {
            if (h_copy_descs_) cudaFreeHost(h_copy_descs_);
            if (d_copy_descs_) cudaFree(d_copy_descs_);
            FZ_CUDA_CHECK(cudaHostAlloc(&h_copy_descs_, desc_bytes, cudaHostAllocDefault));
            FZ_CUDA_CHECK(cudaMalloc(&d_copy_descs_, desc_bytes));
            copy_descs_capacity_ = desc_bytes;
        }
    }

    is_finalized_ = true;

    FZ_LOG(INFO, "Finalized with %zu stages, strategy=%s",
           stages_.size(),
           strategy_ == MemoryStrategy::MINIMAL ? "MINIMAL" : "PREALLOCATE");

    if (warmup_on_finalize_ && input_size_hint_ > 0) {
        FZ_LOG(INFO, "Auto-warmup triggered by setWarmupOnFinalize(true)");
        warmup(/*stream=*/0);
    }
}

void Pipeline::warmup(cudaStream_t stream) {
    if (!is_finalized_) {
        throw std::runtime_error("Pipeline must be finalized before warmup()");
    }
    if (input_size_hint_ == 0) {
        FZ_LOG(INFO, "warmup() skipped: no input_size_hint (construct Pipeline with a non-zero size)");
        return;
    }

    // Allocate a temporary zero-filled dummy input outside the pool so we
    // don't disturb the pre-sized pool reservation.
    void* d_dummy = nullptr;
    FZ_CUDA_CHECK(cudaMalloc(&d_dummy, input_size_hint_));
    FZ_CUDA_CHECK(cudaMemsetAsync(d_dummy, 0, input_size_hint_, stream));

    // Suppress profiling so the warmup pass doesn't pollute last_perf_result_.
    const bool saved_profiling = profiling_enabled_;
    enableProfiling(false);

    void*  d_out  = nullptr;
    size_t out_sz = 0;
    compress(d_dummy, input_size_hint_, &d_out, &out_sz, stream);
    FZ_CUDA_CHECK(cudaStreamSynchronize(stream));

    // Also warm up the decompression path so inverse kernels are JIT-compiled.
    // Must be called before dag_->reset(): d_out is pool-owned and only valid
    // until the forward DAG is reset.
    {
        void*  d_decomp  = nullptr;
        size_t decomp_sz = 0;
        try {
            decompress(d_out, out_sz, &d_decomp, &decomp_sz, stream);
            FZ_CUDA_CHECK(cudaStreamSynchronize(stream));
        } catch (const std::exception& e) {
            FZ_LOG(INFO, "Warmup: decompress pass skipped (%s) — "
                         "inverse kernels will JIT on first decompress()", e.what());
        }
        if (!pool_managed_decomp_ && d_decomp) {
            cudaFree(d_decomp);
        }
        // Discard warmup-specific inverse DAG state.  The real first decompress()
        // call will rebuild both from scratch with the correct input data.
        for (void* p : d_decomp_outputs_) {
            if (p && mem_pool_) mem_pool_->free(p, 0);
        }
        d_decomp_outputs_.clear();
        inv_cache_.reset();
    }

    // Restore pipeline to a clean state — identical to just after finalize().
    dag_->reset(stream);
    was_compressed_ = false;
    is_compressed_  = false;
    buffer_metadata_.clear();

    enableProfiling(saved_profiling);
    cudaFree(d_dummy);

    FZ_LOG(INFO, "Warmup complete — compress and decompress kernels JIT-compiled (input %.2f MB)",
           input_size_hint_ / (1024.0 * 1024.0));
}

void Pipeline::captureGraph(cudaStream_t stream) {
    if (!is_finalized_) {
        throw std::runtime_error("Pipeline must be finalized before captureGraph()");
    }
    if (!graph_mode_enabled_) {
        throw std::runtime_error(
            "Call enableGraphMode(true) before finalize() to use captureGraph()");
    }
    if (was_compressed_) {
        throw std::runtime_error(
            "captureGraph() must be called before the first compress(). "
            "Reconstruct the pipeline or call reset() first.");
    }

    // Destroy any previously captured graph so we start clean.
    if (graph_exec_)     { cudaGraphExecDestroy(graph_exec_);  graph_exec_     = nullptr; }
    if (captured_graph_) { cudaGraphDestroy(captured_graph_);  captured_graph_ = nullptr; }
    graph_captured_ = false;

    // Wire the fixed input slot as the DAG's stable source pointer.
    // This pointer is baked into the graph; compress() will D2D-copy user
    // data here before each cudaGraphLaunch().
    dag_->setExternalPointer(input_buffer_ids_[0], d_graph_input_);
    dag_->updateBufferSize(input_buffer_ids_[0], d_graph_input_size_);

    // setCaptureMode(true) validates that every stage in the DAG is
    // graph-compatible; throws with a descriptive message if not.
    dag_->setCaptureMode(true);

    // Record: BeginCapture → execute (all work forced onto `stream` by
    // capture_mode_ flag) → EndCapture → Instantiate.
    //
    // NOTE: The legacy default stream (stream 0 / cudaStreamLegacy) is NOT
    // capturable.  Callers must pass a non-default stream created with
    // cudaStreamCreate().  Passing stream 0 will cause cudaStreamBeginCapture
    // to return cudaErrorStreamCaptureUnsupported.
    //
    // Exception safety: if dag_->execute() throws, EndCapture is still called
    // so the stream is never left stranded in capture mode.
    FZ_CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    try {
        dag_->execute(stream);
    } catch (...) {
        // Attempt to end the (now-invalidated) capture so the stream reverts
        // to normal mode.  Errors here are suppressed — the original exception
        // is the one the caller needs to see.
        cudaGraph_t tmp = nullptr;
        cudaStreamEndCapture(stream, &tmp);
        if (tmp) cudaGraphDestroy(tmp);
        dag_->setCaptureMode(false);
        throw;
    }
    FZ_CUDA_CHECK(cudaStreamEndCapture(stream, &captured_graph_));
    FZ_CUDA_CHECK(cudaGraphInstantiate(&graph_exec_, captured_graph_,
                                       nullptr, nullptr, 0));

    dag_->setCaptureMode(false);
    graph_captured_ = true;

    FZ_LOG(INFO, "CUDA Graph captured and instantiated successfully");
}

void Pipeline::reset(cudaStream_t stream) {
    if (!is_finalized_) {
        return;
    }

    dag_->reset(stream);
    was_compressed_ = false;

    FZ_LOG(DEBUG, "Reset complete");
}

size_t Pipeline::getPeakMemoryUsage() const {
    const size_t dag_peak  = dag_ ? dag_->getPeakMemoryUsage() : 0;
    const size_t pool_peak = mem_pool_ ? mem_pool_->getPeakUsage() : 0;
    return std::max(dag_peak, pool_peak);
}

size_t Pipeline::getPoolThreshold() const {
    return mem_pool_ ? mem_pool_->getConfiguredSize() : 0;
}

size_t Pipeline::getCurrentMemoryUsage() const {
    return dag_ ? dag_->getCurrentMemoryUsage() : 0;
}

void Pipeline::printPipeline() const {
    FZ_PRINT("========== Pipeline Configuration ==========");
    FZ_PRINT("Stages: %zu  Strategy: %s  Streams: %d  Finalized: %s",
             stages_.size(),
             strategy_ == MemoryStrategy::MINIMAL ? "MINIMAL" : "PREALLOCATE",
             num_streams_,
             is_finalized_ ? "yes" : "no");
    FZ_PRINT("Stages:");
    for (const auto& stage : stages_) {
        FZ_PRINT("  - %s (inputs=%zu, outputs=%zu)",
                 stage->getName().c_str(),
                 stage->getNumInputs(), stage->getNumOutputs());
    }
    if (is_finalized_) {
        dag_->printDAG();
    }
    FZ_PRINT("============================================");
}

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
    
    for (const auto& [stage, node] : stage_to_node_) {
        size_t num_outputs = stage->getNumOutputs();
        auto output_names = stage->getOutputNames();
        
        for (size_t i = 0; i < num_outputs; i++) {
            // Find the pre-allocated buffer for this output
            auto it = node->output_index_to_buffer_id.find(static_cast<int>(i));
            if (it == node->output_index_to_buffer_id.end()) {
                throw std::runtime_error(
                    "Missing pre-allocated buffer for output " + std::to_string(i) +
                    " of stage '" + node->name + "' — this is an internal pipeline bug");
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

std::vector<Pipeline::OutputBufferInfo> Pipeline::collectOutputBuffers() const {
    std::vector<OutputBufferInfo> outputs;
    
    for (int buffer_id : output_buffer_ids_) {
        const auto& buffer_info = dag_->getBufferInfo(buffer_id);
        
        DAGNode* producer_node = nullptr;
        for (const auto& node : dag_->getNodes()) {
            if (node->id == buffer_info.producer_stage_id) {
                producer_node = node;
                break;
            }
        }
        
        if (!producer_node) {
            throw std::runtime_error(
                "Producer stage not found for buffer " + std::to_string(buffer_id) +
                " — this is an internal pipeline bug");
        }
        
        auto stage_output_names = producer_node->stage->getOutputNames();
        int output_idx = buffer_info.producer_output_index;
        std::string output_name = (output_idx >= 0 && output_idx < static_cast<int>(stage_output_names.size())) 
                                  ? stage_output_names[output_idx] 
                                  : "output";
        
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

// Round up to the next multiple of 16 for concat slot alignment.
static inline size_t align16(size_t x) { return (x + 15u) & ~15u; }

size_t Pipeline::calculateConcatSize(const std::vector<OutputBufferInfo>& outputs) const {
    const size_t n = outputs.size();
    // Header: [num_buffers: u32][size_0..N-1: u64], padded to 16-byte boundary
    // so the first data segment's dst pointer is 16-byte aligned.
    const size_t hdr_raw     = sizeof(uint32_t) + n * sizeof(uint64_t);
    const size_t hdr_padded  = align16(hdr_raw);
    size_t total_size = hdr_padded;
    for (const auto& output : outputs) {
        // Each segment's slot is padded to 16 bytes so subsequent dst pointers
        // stay aligned.  The header stores the actual (unpadded) size.
        total_size += align16(output.actual_size);
    }
    return total_size;
}

size_t Pipeline::writeConcatBuffer(
    const std::vector<OutputBufferInfo>& outputs,
    uint8_t* d_concat_bytes,
    cudaStream_t stream
) const {
    const size_t n = outputs.size();

    // ── Pack the entire header into the pinned host buffer (no API calls) ──
    // Layout: [num_buffers: u32][size_0: u64][size_1: u64]...[size_N-1: u64]
    const size_t hdr_bytes = sizeof(uint32_t) + n * sizeof(uint64_t);

    uint8_t* h = static_cast<uint8_t*>(h_concat_header_);
    *reinterpret_cast<uint32_t*>(h) = static_cast<uint32_t>(n);
    h += sizeof(uint32_t);
    for (const auto& out : outputs) {
        *reinterpret_cast<uint64_t*>(h) = static_cast<uint64_t>(out.actual_size);
        h += sizeof(uint64_t);
    }

    // ── Build gather descriptors on the CPU (no API calls) ──────────────────
    // Offsets use padded slot sizes so each dst pointer is 16-byte aligned;
    // the bytes field carries the actual (unpadded) size for the kernel.
    CopyDesc* h_descs = static_cast<CopyDesc*>(h_copy_descs_);
    size_t offset = align16(hdr_bytes);  // first segment starts after padded header
    for (size_t i = 0; i < n; i++) {
        h_descs[i].src   = static_cast<const uint8_t*>(outputs[i].d_ptr);
        h_descs[i].dst   = d_concat_bytes + offset;
        h_descs[i].bytes = outputs[i].actual_size;
        FZ_LOG(TRACE, "Concat desc[%zu]: %s  %.1f KB -> offset %zu",
               i, outputs[i].stage_name.c_str(), outputs[i].actual_size / 1024.0, offset);
        offset += align16(outputs[i].actual_size);  // advance by padded slot size
    }

    // ── Two H2D copies + one kernel launch (replaces 1 H2D + N D2D) ─────────
    // 1. Header (num_buffers + per-segment sizes)
    FZ_CUDA_CHECK(cudaMemcpyAsync(d_concat_bytes, h_concat_header_, hdr_bytes,
                                  cudaMemcpyHostToDevice, stream));
    // 2. Gather descriptors
    const size_t desc_bytes = n * sizeof(CopyDesc);
    FZ_CUDA_CHECK(cudaMemcpyAsync(d_copy_descs_, h_copy_descs_, desc_bytes,
                                  cudaMemcpyHostToDevice, stream));
    // 3. Gather kernel: one block per segment, 256 threads each
    launch_gather_kernel(static_cast<const CopyDesc*>(d_copy_descs_),
                         static_cast<int>(n), 256, stream);

    return offset;
}

void Pipeline::concatOutputs(void** d_output, size_t* output_size, cudaStream_t stream) {
    auto outputs = collectOutputBuffers();
    
    size_t total_size = calculateConcatSize(outputs);
    
    if (d_concat_buffer_ == nullptr || concat_buffer_capacity_ < total_size) {
        if (d_concat_buffer_) {
            mem_pool_->free(d_concat_buffer_, stream);
        }
        
        d_concat_buffer_ = mem_pool_->allocate(total_size, stream, "concat_output", true);
        if (!d_concat_buffer_) {
            throw std::runtime_error(
                "Failed to allocate concat output buffer (" + std::to_string(total_size) +
                " bytes for " + std::to_string(outputs.size()) +
                " outputs); pool may be exhausted");
        }
        concat_buffer_capacity_ = total_size;
        
        FZ_LOG(DEBUG, "Allocated concat buffer: %.1f KB for %zu outputs",
               total_size / 1024.0, outputs.size());
    }
    
    *d_output = d_concat_buffer_;
    *output_size = writeConcatBuffer(outputs, static_cast<uint8_t*>(d_concat_buffer_), stream);
    
    FZ_LOG(DEBUG, "Concatenation complete: %zu buffers -> %.1f KB",
           outputs.size(), total_size / 1024.0);
}
} // namespace fz
