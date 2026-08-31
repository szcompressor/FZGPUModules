#include "pipeline/compressor.h"
#include "pipeline/concat_kernel.h"
#include "advanced/fusion_planner.h"
#include "advanced/fusion_registry.h"
#include "pipeline_utils.h"
#include "fzm_format.h"
#include "log.h"
#include "cuda_check.h"
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <cstring>

namespace fz {

Pipeline::Pipeline(size_t input_data_size, MemoryStrategy strategy, float pool_multiplier)
    : strategy_(strategy),
      num_streams_(1),
      is_finalized_(false),
      warmup_on_finalize_(false),
      pool_managed_decomp_(true),
      is_compressed_(false),
      was_compressed_(false),
      profiling_enabled_(false),
      needs_concat_(false),
      input_size_(0),
      input_alignment_bytes_(1),
      original_input_size_(0),
      input_size_hint_(input_data_size),
      pool_multiplier_(pool_multiplier),
      dims_({0, 1, 1}),
      graph_mode_enabled_(false),
      graph_captured_(false),
      d_graph_input_size_(0),
      captured_graph_(nullptr),
      graph_exec_(nullptr) {
    
    MemoryPoolConfig pool_config(input_data_size, pool_multiplier);
    mem_pool_ = std::make_unique<MemoryPool>(pool_config);
    
    dag_ = std::make_unique<CompressionDAG>(mem_pool_.get(), strategy);
}

Pipeline::~Pipeline() {
    // Destroy graph handles before the pool is torn down.
    if (graph_exec_)     { cudaGraphExecDestroy(graph_exec_);  graph_exec_     = nullptr; }
    if (captured_graph_) { cudaGraphDestroy(captured_graph_);  captured_graph_ = nullptr; }

    // Free pool-managed decompress outputs (vector is variable-length; no RAII wrapper).
    for (void* p : d_decomp_outputs_) {
        if (p && mem_pool_) mem_pool_->free(p, 0);
    }

    // All other buffers (d_concat_buffer_, d_graph_input_, d_pad_buf_,
    // h_concat_header_, h_copy_descs_, d_copy_descs_) are freed by their
    // RAII wrappers — no manual cleanup needed here.
}

void Pipeline::setMemoryStrategy(MemoryStrategy strategy) {
    if (is_finalized_) {
        throw std::runtime_error("Cannot change strategy after finalization");
    }
    strategy_ = strategy;
    dag_ = std::make_unique<CompressionDAG>(mem_pool_.get(), strategy);
    // Re-apply Pipeline-level DAG state to the replacement.  Constructing a fresh
    // CompressionDAG discards everything previously configured on the old one, and
    // profiling is the piece that matters: callers set it via enableProfiling()
    // *before* loadConfig(), and every TOML preset carries a `memory_strategy` key,
    // so loadConfig() lands here and silently brought the new DAG up with profiling
    // off.  collectTimings() then returns {} and the whole compress-phase per-stage
    // breakdown disappears -- from the stdout perf table and from `stages[]` in
    // --report-json alike -- while decompress kept working, because its inverse DAG
    // is built separately at decompress() time.  That asymmetry is exactly what the
    // 2026-07-29 full-corpus sweep recorded: 4,860 FZGM rows, every one of them
    // decompress-only.  Anything else added to CompressionDAG that is configured
    // before finalize() must be re-applied here too.
    dag_->enableProfiling(profiling_enabled_);
    // Buffer coloring is the second such setting, and it went missing the same way:
    // setColoringEnabled(false) before loadConfig() was silently discarded, because
    // every TOML preset carries a `memory_strategy` key and so routes through here.
    // The symptom was a *plausible wrong number* rather than an error -- an
    // uncolored-vs-colored peak-memory comparison reported identical peaks for all
    // 15 presets, i.e. "liveness coloring saves nothing", when in fact coloring had
    // simply never been turned off.
    dag_->setColoringEnabled(coloring_enabled_);
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

void Pipeline::planAndInstallFusion() {
    // Runtime override (FZ_FUSION=off|auto|force) wins over the programmatic mode.
    FusionPolicy mode = fusion_policy_;
    if (const char* e = std::getenv("FZ_FUSION")) {
        const std::string v(e);
        if (v == "off" || v == "0")      mode = FusionPolicy::Off;
        else if (v == "auto" || v == "1") mode = FusionPolicy::Auto;
        else if (v == "force" || v == "experimental" || v == "2")
            mode = FusionPolicy::Force;
    }
    fusion_info_ = {};
    fusion_info_.policy = mode;
    if (mode == FusionPolicy::Off) {
        fusion_info_.fallback_reason = "policy_off";
        return;
    }

    std::vector<CompressionDAG::FusedGroupExec> installed;
    const bool include_experimental = mode == FusionPolicy::Force;
    auto maximal_groups = planFusionGroups(*dag_);
    fusion_info_.legal_group_count = maximal_groups.size();
    for (auto& maximal : maximal_groups) {
        struct Candidate {
            size_t begin = 0;
            size_t end = 0; // exclusive
            const FusedImpl* impl = nullptr;
        };
        std::vector<Candidate> candidates;
        const size_t count = maximal.stages.size();
        // A maximal legality chain is only the search domain. Registry matches
        // decide which overlapping partial/full spans have executable plans.
        for (size_t begin = 0; begin + 1 < count; ++begin) {
            for (size_t end = count; end >= begin + 2; --end) {
                std::vector<Stage*> span(
                    maximal.stages.begin() + static_cast<std::ptrdiff_t>(begin),
                    maximal.stages.begin() + static_cast<std::ptrdiff_t>(end));
                if (const FusedImpl* impl = findFusedImpl(span, include_experimental))
                    candidates.push_back(Candidate{begin, end, impl});
                if (end == begin + 2) break; // avoid size_t underflow
            }
        }

        // Weighted interval selection. Weight = staged launches removed; every
        // candidate was separately admitted by the evidence-gated registry.
        std::vector<size_t> best(count + 1, 0);
        std::vector<const Candidate*> choice(count, nullptr);
        for (size_t ri = count; ri-- > 0;) {
            best[ri] = best[ri + 1];
            for (const Candidate& c : candidates) {
                if (c.begin != ri) continue;
                const size_t score = (c.end - c.begin - 1) + best[c.end];
                if (score > best[ri]) {
                    best[ri] = score;
                    choice[ri] = &c;
                }
            }
        }

        for (size_t i = 0; i < count;) {
            const Candidate* c = choice[i];
            if (!c) { ++i; continue; }
            CompressionDAG::FusedGroupExec fg;
            fg.impl = c->impl;
            fg.stages.assign(maximal.stages.begin() + static_cast<std::ptrdiff_t>(c->begin),
                             maximal.stages.begin() + static_cast<std::ptrdiff_t>(c->end));
            bool ok = true;
            for (Stage* s : fg.stages) {
                auto it = stage_to_node_.find(s);
                if (it == stage_to_node_.end() || !it->second) { ok = false; break; }
                fg.members.push_back(it->second);
            }
            if (ok) {
                fg.head = fg.members.front();
                fg.tail = fg.members.back();
                FusionGroupInfo info;
                info.implementation = c->impl->name;
                for (Stage* stage : fg.stages) info.stages.push_back(stage->getName());
                fusion_info_.installed_groups.push_back(std::move(info));
                FZ_LOG(INFO, "Fusion: installed '%s' over %zu/%zu compatible stages",
                       c->impl->name, fg.stages.size(), count);
                installed.push_back(std::move(fg));
            }
            i = c->end;
        }
    }
    if (installed.empty()) {
        fusion_info_.fallback_reason = fusion_info_.legal_group_count == 0
            ? "no_legal_group" : "no_profitable_implementation";
        return;
    }

    dag_->setFusedGroups(std::move(installed));
    if (graph_mode_enabled_) {
        FZ_LOG(WARN, "Fusion active — disabling CUDA graph mode (fused runner synchronises).");
        graph_mode_enabled_ = false;
    }
    // Buffer coloring aliases buffers by the STAGED liveness (the group's input
    // dies after the first stage). A fused kernel keeps that input live across
    // the whole group while writing the group's output, so an aliased
    // input/output region would corrupt. Disable coloring for fused pipelines.
    // (planAndInstallFusion runs before preallocateBuffers(), where coloring is
    // applied, so this takes effect.)
    dag_->setColoringEnabled(false);
}

void Pipeline::bindSemanticContracts() {
    // Semantic binding is intentionally independent of FusionPolicy: the staged
    // path must make the same algorithmic choice as a future fused path. For the
    // initial local contract, require one unambiguous direct consumer. Broader
    // side-output topology is handled by the later auxiliary-output checkpoint.
    for (DAGNode* producer : dag_->getNodes()) {
        if (!producer || !producer->stage || producer->dependents.size() != 1) continue;
        DAGNode* consumer = producer->dependents.front();
        if (!consumer || !consumer->stage) continue;
        const EncodingOracleDecl decl = consumer->stage->getEncodingOracle();
        if (!decl.valid()) continue;
        if (producer->stage->bindDownstreamEncodingOracle(decl)) {
            FZ_LOG(INFO, "Semantic binding: %s <- %s encoded-size oracle '%s'",
                   producer->stage->getName().c_str(),
                   consumer->stage->getName().c_str(), decl.op_name.c_str());
        }
    }
}

void Pipeline::finalize() {
    if (is_finalized_) {
        throw std::runtime_error("Pipeline already finalized");
    }

    for (const auto& stage_ptr : stages_) {
        stage_ptr->setDims(dims_);
        auto it = stage_to_node_.find(stage_ptr.get());
        if (it != stage_to_node_.end() && it->second)
            it->second->name = stage_ptr->getName();
    }

    validate();
    typeCheckConnections();
    bindSemanticContracts();

    auto [sources, sinks] = identifyTopology();
    setupInputBuffers(sources);
    detectMultiOutputScenario(autoDetectUnconnectedOutputs());
    configureStreamsIfNeeded();
    dag_->finalize();

    planAndInstallFusion();    // may disable graph mode if any group fuses

    computeInputAlignment();   // must run before propagateBufferSizes
    propagateBufferSizes();
    notifyStagesFinalizeHooks();  // let stages pre-allocate using propagated sizes
    refinePoolSize();

    if (strategy_ == MemoryStrategy::PREALLOCATE)
        dag_->preallocateBuffers();
    if (graph_mode_enabled_)
        setupGraphModeInput();

    preallocatePadBuffer();

    num_streams_ = std::max(1, static_cast<int>(dag_->getStreamCount()));
    preallocateConcatBuffers();

    is_finalized_ = true;
    FZ_LOG(INFO, "Finalized with %zu stages, strategy=%s",
           stages_.size(),
           strategy_ == MemoryStrategy::MINIMAL ? "MINIMAL" : "PREALLOCATE");

    if (warmup_on_finalize_ && input_size_hint_ > 0) {
        FZ_LOG(INFO, "Auto-warmup triggered by setWarmupOnFinalize(true)");
        warmup(/*stream=*/0);
    }
}

void Pipeline::typeCheckConnections() {
    // Byte-transparent stages (Bitshuffle, RZE) return UNKNOWN and are skipped.
    // Compatibility: exact match, or same element size + both integral (allows
    // uint16↔int16 reinterpret in DifferenceStage). Float↔integer is always wrong.
    auto typesCompatible = [](uint8_t a, uint8_t b) -> bool {
        if (a == b) return true;
        const auto da = static_cast<DataType>(a);
        const auto db = static_cast<DataType>(b);
        const bool a_float = (da == DataType::FLOAT32 || da == DataType::FLOAT64);
        const bool b_float = (db == DataType::FLOAT32 || db == DataType::FLOAT64);
        if (a_float || b_float) return false;
        return getDataTypeSize(da) == getDataTypeSize(db);
    };

    constexpr uint8_t kUnknown = static_cast<uint8_t>(DataType::UNKNOWN);

    for (const auto& conn : connections_) {
        // A connection's input port isn't stored explicitly -- connect() and
        // bindExternalInput() both append to the dependent's DAGNode::
        // input_buffer_ids in call order, so that vector's index of this
        // connection's buffer id IS its real port position. Counting entries
        // in connections_ alone (an earlier version of this fix did that)
        // undercounts whenever bindExternalInput() also consumed a port on
        // the same stage without adding to connections_ -- exactly
        // Cdf97OutlierCorrectStage's shape (external field on port 0, this
        // connection on port 1).
        DAGNode* dep_node  = stage_to_node_.at(conn.dependent);
        DAGNode* prod_node = stage_to_node_.at(conn.producer);
        int buf_id = prod_node->output_index_to_buffer_id.at(conn.output_index);
        auto pos_it = std::find(dep_node->input_buffer_ids.begin(), dep_node->input_buffer_ids.end(), buf_id);
        if (pos_it == dep_node->input_buffer_ids.end())
            throw std::runtime_error("typeCheckConnections: internal error -- connection buffer not found on dependent node");
        const size_t input_index = static_cast<size_t>(pos_it - dep_node->input_buffer_ids.begin());

        const uint8_t prod_type = conn.producer->getOutputDataType(
            static_cast<size_t>(conn.output_index));
        const uint8_t cons_type = conn.dependent->getInputDataType(input_index);

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

void Pipeline::computeInputAlignment() {
    // LCM of all stage alignment requirements. Must run before propagateBufferSizes()
    // so the rounded-up hint drives all downstream buffer size estimates.
    size_t align = 1;
    int chunked_count = 0;
    for (const auto& stage_ptr : stages_) {
        const size_t a = stage_ptr->getRequiredInputAlignment();
        if (a > 1) { align = std::lcm(align, a); ++chunked_count; }
    }
    input_alignment_bytes_ = align;
    if (align > 1) {
        input_size_hint_ = ((input_size_hint_ + align - 1) / align) * align;
        FZ_LOG(INFO, "Input alignment: %zu bytes (LCM of %d chunked stage(s)); "
               "buffer hint rounded up to %zu bytes",
               align, chunked_count, input_size_hint_);
    }
}

void Pipeline::refinePoolSize() {
    // Skip without an input hint — buffer sizes are 1-byte placeholders.
    if (input_size_hint_ == 0) return;

    constexpr float kTopoSafetyMargin = 1.1f;  // 10% headroom for transient CUB allocations
    const size_t topo_base  = dag_->computeTopoPoolSize();
    // d_pad_buf_ is pool-allocated but never appears in buffers_, so add it manually.
    const size_t pad_bytes  = (input_alignment_bytes_ > 1) ? input_size_hint_ : 0;
    const size_t topo_sized = static_cast<size_t>((topo_base + pad_bytes) * kTopoSafetyMargin);
    // UINT64_MAX: never trim pool pages between calls — avoids re-page-fault latency spikes.
    // Does not affect peak in-flight footprint; MINIMAL still allocates/frees lazily.
    mem_pool_->setReleaseThreshold(std::numeric_limits<uint64_t>::max());
    FZ_LOG(INFO, "Pool threshold: %.2f MB (topo %.2f MB + pad %.2f MB, ×1.1 margin) "
           "[release threshold pinned to UINT64_MAX]",
           topo_sized / (1024.0 * 1024.0),
           topo_base  / (1024.0 * 1024.0),
           pad_bytes  / (1024.0 * 1024.0));
}

void Pipeline::setupGraphModeInput() {
    if (strategy_ != MemoryStrategy::PREALLOCATE) {
        throw std::runtime_error(
            "Graph mode requires PREALLOCATE memory strategy. "
            "Call setMemoryStrategy(MemoryStrategy::PREALLOCATE) before finalize().");
    }
    if (mem_pool_ && mem_pool_->isFallbackMode()) {
        throw std::runtime_error(
            "Graph mode is not supported when the CUDA memory pool is unavailable "
            "(cudaMalloc fallback mode; common on vGPU or when FZ_FORCE_MEMPOOL_FALLBACK is set). "
            "Disable graph mode or use a GPU that supports cudaMemPool.");
    }
    if (input_size_hint_ == 0) {
        throw std::runtime_error(
            "Graph mode requires a non-zero input size hint. "
            "Pass input_data_size to the Pipeline constructor.");
    }
    if (input_nodes_.size() != 1) {
        throw std::runtime_error(
            "Graph mode currently supports single-source pipelines only "
            "(" + std::to_string(input_nodes_.size()) + " source(s) detected).");
    }

    // Allocate a fixed device buffer at the padded hint size. compress() will
    // D2D-copy the user's input here before calling cudaGraphLaunch().
    const size_t padded = (input_alignment_bytes_ > 1)
        ? ((input_size_hint_ + input_alignment_bytes_ - 1) / input_alignment_bytes_)
          * input_alignment_bytes_
        : input_size_hint_;

    if (!d_graph_input_.allocate(mem_pool_.get(), padded, 0,
                                 "graph_input_slot", /*persistent=*/true)) {
        throw std::runtime_error(
            "Failed to allocate graph input slot (" + std::to_string(padded) +
            " bytes); pool may be exhausted — increase pool_size_multiplier or input_data_size");
    }
    d_graph_input_size_ = padded;
    FZ_LOG(INFO, "Graph mode: fixed input slot allocated (%.2f MB)",
           padded / (1024.0 * 1024.0));
}

void Pipeline::preallocatePadBuffer() {
    if (input_alignment_bytes_ <= 1 || input_size_hint_ == 0) return;

    const size_t padded = ((input_size_hint_ + input_alignment_bytes_ - 1)
                           / input_alignment_bytes_) * input_alignment_bytes_;
    if (padded <= d_pad_buf_.capacity) return;

    if (!d_pad_buf_.allocate(mem_pool_.get(), padded, /*stream=*/0,
                             "pipeline_input_pad", /*persistent=*/true)) {
        throw std::runtime_error(
            "Failed to preallocate pipeline input pad buffer (" +
            std::to_string(padded) + " bytes); pool may be exhausted");
    }
    mem_pool_->synchronize(/*stream=*/0);
}

void Pipeline::preallocateConcatBuffers() {
    if (!needs_concat_ || output_buffer_ids_.empty()) return;

    const size_t n_outputs  = output_buffer_ids_.size();
    const size_t hdr_bytes  = sizeof(uint32_t) + n_outputs * sizeof(uint64_t);
    const size_t desc_bytes = n_outputs * sizeof(CopyDesc);

    if (!h_concat_header_.ensureCapacity(hdr_bytes))
        throw std::runtime_error("Failed to allocate pinned concat header buffer");
    if (!h_copy_descs_.ensureCapacity(desc_bytes))
        throw std::runtime_error("Failed to allocate pinned copy descriptor buffer");
    if (!d_copy_descs_.ensureCapacity(desc_bytes))
        throw std::runtime_error("Failed to allocate device copy descriptor buffer");
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
    dag_->setExternalPointer(input_buffer_ids_[0], d_graph_input_.ptr);
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

bool Pipeline::isMemPoolFallbackMode() const {
    return mem_pool_ && mem_pool_->isFallbackMode();
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

    // A pipeline wired entirely through bindExternalInput() (every stage has
    // some real dependency, e.g. a single mixed source+consumer stage) has
    // no pure (dependencies.empty()) source -- that's still a valid topology.
    if (sources.empty() && explicit_external_bindings_.empty()) {
        throw std::runtime_error("Pipeline has no source stages");
    }
    if (sinks.empty()) {
        throw std::runtime_error("Pipeline has no sink stages");
    }

    return {sources, sinks};
}

void Pipeline::setupInputBuffers(const std::vector<Stage*>& sources) {
    input_nodes_.clear();
    input_buffer_ids_.clear();
    const bool single = (sources.size() + explicit_external_bindings_.size()) == 1;
    for (size_t i = 0; i < sources.size(); i++) {
        DAGNode* src_node = stage_to_node_[sources[i]];
        // A stage that's both a pure source (dependencies.empty(), so
        // getSourceStages() found it) AND was separately given an explicit
        // bindExternalInput() call already has its external buffer -- binding
        // it again here would give the same node two input-buffer entries
        // for what's really one external input, corrupting its port count.
        bool already_bound = std::any_of(explicit_external_bindings_.begin(), explicit_external_bindings_.end(),
            [&](const auto& p) { return p.first == src_node; });
        if (already_bound) continue;

        std::string tag = single
            ? "pipeline_input"
            : "pipeline_input_" + std::to_string(i) + "_" + sources[i]->getName();
        dag_->setInputBuffer(src_node, 1, tag);
        input_nodes_.push_back(src_node);
        input_buffer_ids_.push_back(src_node->input_buffer_ids.back());
    }
    // Stages bound via bindExternalInput() -- may have other real
    // dependencies too, so getSourceStages() (dependencies.empty()) never
    // finds them; the buffer id was already created at bindExternalInput()
    // call time, just append it here.
    for (const auto& [node, buf_id] : explicit_external_bindings_) {
        input_nodes_.push_back(node);
        input_buffer_ids_.push_back(buf_id);
    }
}

void Pipeline::bindExternalInput(Stage* stage) {
    if (is_finalized_) {
        throw std::runtime_error("bindExternalInput(): cannot call after finalize()");
    }
    auto it = stage_to_node_.find(stage);
    if (it == stage_to_node_.end()) {
        throw std::runtime_error("bindExternalInput(): stage not added to this pipeline (call addStage() first)");
    }
    DAGNode* node = it->second;
    std::string tag = "pipeline_input_ext_" + stage->getName();
    dag_->setInputBuffer(node, 1, tag);   // size fixed up later via propagateBufferSizes
    explicit_external_bindings_.emplace_back(node, node->input_buffer_ids.back());
}

int Pipeline::autoDetectUnconnectedOutputs() {
    int pipeline_outputs_added = 0;
    
    // Iterate nodes in deterministic insertion order (nodes_), NOT stage_to_node_
    // — the latter is an unordered_map keyed by Stage* pointers, whose iteration
    // order depends on heap addresses (ASLR) and thus varies run-to-run. That
    // order determines the sequence in which unconnected outputs are appended to
    // output_buffer_ids_, i.e. the concat layout of multi-leaf pipelines, making
    // the compressed blob non-deterministic (same size, permuted segments) even
    // though every stage output is identical. Single-leaf pipelines were immune
    // (one output, no ordering). Deterministic node order fixes reproducibility.
    for (DAGNode* node : dag_->getNodes()) {
        Stage* stage = node->stage;
        if (!stage) continue;
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
    bool has_hint = (input_size_hint_ > 0);
    if (!has_hint && !force_from_current_inputs) {
        FZ_LOG(DEBUG, "No input size hint, using placeholder buffer sizes");
        return;
    }

    if (!force_from_current_inputs) {
        // Seed the source's external input buffer with the constructor hint.
        for (size_t i = 0; i < input_nodes_.size(); i++) {
            if (input_size_hint_ > 0) {
                dag_->updateBufferSize(input_buffer_ids_[i], input_size_hint_);
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

void Pipeline::notifyStagesFinalizeHooks() {
    // Call Stage::onFinalize for each stage with its estimated input size (bytes)
    // so stages with complex internal memory (e.g. HuffmanStage) can pre-allocate
    // their persistent scratch from the pool rather than lazily via cudaMalloc.
    // This makes PREALLOCATE mode semantically correct and makes pool-bypass
    // allocations visible to pool footprint reporting.
    MemoryPool* pool = mem_pool_.get();
    for (const auto& level_nodes : dag_->getLevels()) {
        for (auto* node : level_nodes) {
            size_t estimated_inlen = 0;
            if (!node->input_buffer_ids.empty())
                estimated_inlen = dag_->getBufferSize(node->input_buffer_ids[0]);
            node->stage->onFinalize(estimated_inlen, pool);
        }
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

    for (const auto& [stage, node] : stage_to_node_)
        stage->setTerminalOutput(node->dependents.empty());

    FZ_LOG(DEBUG, "Validation passed");
}

std::vector<Stage*> Pipeline::getSourceStages() const {
    std::vector<Stage*> sources;

    // Iterate dag_->getNodes() (deterministic insertion order), NOT
    // stage_to_node_ (an unordered_map keyed by Stage*, whose iteration order
    // depends on heap addresses / ASLR). With exactly one source this never
    // mattered -- order of one element is irrelevant -- but with multiple
    // source stages (see Pipeline::setPrimarySource()) this order becomes
    // input_nodes_'s order, and index 0 is the default primary source when
    // the caller hasn't called setPrimarySource(). An ASLR-dependent default
    // would silently pick a different "primary" answer on different runs of
    // the identical program. Same class of bug, same fix, as
    // autoDetectUnconnectedOutputs()'s existing comment on this file.
    for (DAGNode* node : dag_->getNodes()) {
        if (node->stage && node->dependencies.empty()) {
            sources.push_back(node->stage);
        }
    }

    return sources;
}

std::vector<Stage*> Pipeline::getSinkStages() const {
    std::vector<Stage*> sinks;

    for (DAGNode* node : dag_->getNodes()) {
        if (node->stage && node->dependents.empty()) {
            sinks.push_back(node->stage);
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

size_t Pipeline::calculateConcatSize(const std::vector<OutputBufferInfo>& outputs) const {
    size_t total = ConcatLayout::headerSize(outputs.size());
    for (const auto& output : outputs)
        total += ConcatLayout::slotSize(output.actual_size);
    return total;
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

    uint8_t* h = static_cast<uint8_t*>(h_concat_header_.ptr);
    uint32_t n_u32 = static_cast<uint32_t>(n);
    std::memcpy(h, &n_u32, sizeof(uint32_t));
    h += sizeof(uint32_t);
    for (const auto& out : outputs) {
        uint64_t sz_u64 = static_cast<uint64_t>(out.actual_size);
        std::memcpy(h, &sz_u64, sizeof(uint64_t));
        h += sizeof(uint64_t);
    }

    // ── Build gather descriptors on the CPU (no API calls) ──────────────────
    CopyDesc* h_descs = static_cast<CopyDesc*>(h_copy_descs_.ptr);
    size_t offset = ConcatLayout::headerSize(n);
    for (size_t i = 0; i < n; i++) {
        h_descs[i].src   = static_cast<const uint8_t*>(outputs[i].d_ptr);
        h_descs[i].dst   = d_concat_bytes + offset;
        h_descs[i].bytes = outputs[i].actual_size;
        FZ_LOG(TRACE, "Concat desc[%zu]: %s  %.1f KB -> offset %zu",
               i, outputs[i].stage_name.c_str(), outputs[i].actual_size / 1024.0, offset);
        offset += ConcatLayout::slotSize(outputs[i].actual_size);
    }

    // ── Two H2D copies + one kernel launch (replaces 1 H2D + N D2D) ─────────
    // 1. Header (num_buffers + per-segment sizes)
    FZ_CUDA_CHECK(cudaMemcpyAsync(d_concat_bytes, h_concat_header_.ptr, hdr_bytes,
                                  cudaMemcpyHostToDevice, stream));
    // 2. Gather descriptors
    const size_t desc_bytes = n * sizeof(CopyDesc);
    FZ_CUDA_CHECK(cudaMemcpyAsync(d_copy_descs_.ptr, h_copy_descs_.ptr, desc_bytes,
                                  cudaMemcpyHostToDevice, stream));
    // 3. Gather kernel: one block per segment, 256 threads each
    launch_gather_kernel(static_cast<const CopyDesc*>(d_copy_descs_.ptr),
                         static_cast<int>(n), 256, stream);

    return offset;
}

void Pipeline::concatOutputs(void** d_output, size_t* output_size, cudaStream_t stream) {
    auto outputs = collectOutputBuffers();
    
    size_t total_size = calculateConcatSize(outputs);
    
    if (d_concat_buffer_.capacity < total_size) {
        if (!d_concat_buffer_.allocate(mem_pool_.get(), total_size, stream,
                                       "concat_output", /*persistent=*/true)) {
            throw std::runtime_error(
                "Failed to allocate concat output buffer (" + std::to_string(total_size) +
                " bytes for " + std::to_string(outputs.size()) +
                " outputs); pool may be exhausted");
        }
        FZ_LOG(DEBUG, "Allocated concat buffer: %.1f KB for %zu outputs",
               total_size / 1024.0, outputs.size());
    }

    // Zero the concat buffer before the gather. Each segment slot is padded to
    // 16 bytes (ConcatLayout::slotSize = align16), but the gather kernel writes
    // only actual_size bytes per slot — the [actual_size, align16) padding is
    // left as whatever this persistent, recycled pool buffer held before. That
    // stale padding makes the compressed blob non-deterministic across runs
    // (identical total size, differing padding bytes; decode ignores it since
    // it reads actual_size per segment from the header). Only multi-leaf
    // pipelines concat, which is why single-output pipelines were unaffected.
    FZ_CUDA_CHECK(cudaMemsetAsync(d_concat_buffer_.ptr, 0, total_size, stream));

    *d_output    = d_concat_buffer_.ptr;
    *output_size = writeConcatBuffer(outputs, static_cast<uint8_t*>(d_concat_buffer_.ptr), stream);


    FZ_LOG(DEBUG, "Concatenation complete: %zu buffers -> %.1f KB",
           outputs.size(), total_size / 1024.0);
}
} // namespace fz
