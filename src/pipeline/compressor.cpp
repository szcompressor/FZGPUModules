#include "pipeline/compressor.h"
#include "fzm_format.h"
#include "log.h"
#include "cuda_check.h"
#include <chrono>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <cstdint>
#include <vector>
#include <unordered_set>
#include <nvtx3/nvtx3.hpp>

namespace fz {

Pipeline::Pipeline(size_t input_data_size, MemoryStrategy strategy, float pool_multiplier)
    : strategy_(strategy),
      num_streams_(1),
      is_finalized_(false),
      soft_run_enabled_(false),
      profiling_enabled_(false),
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
    cudaStream_t fallback_stream,
    int num_streams,
    std::vector<StageTimingResult>* timing_out
) {
    if (specs.empty()) {
        throw std::runtime_error("runInversePipeline: no stages to execute");
    }

    // ── Step 1: Build fwd_output_buf → spec_index map ─────────────────────────
    // Used to identify which spec produced a given buffer (for level computation).
    std::unordered_map<int, int> output_buf_to_spec;
    for (int i = 0; i < static_cast<int>(specs.size()); i++) {
        for (int buf_id : specs[i].fwd_output_ids) {
            output_buf_to_spec[buf_id] = i;
        }
    }

    // ── Step 2: Compute forward level for each spec ───────────────────────────
    // Specs arrive in forward topological order, so a simple forward pass suffices.
    // level[i] = max(level[j] + 1) for all j whose fwd_output_ids feed spec i.
    std::vector<int> spec_level(specs.size(), 0);
    for (int i = 0; i < static_cast<int>(specs.size()); i++) {
        for (int buf_id : specs[i].fwd_input_ids) {
            auto it = output_buf_to_spec.find(buf_id);
            if (it != output_buf_to_spec.end()) {
                int producer = it->second;
                spec_level[i] = std::max(spec_level[i], spec_level[producer] + 1);
            }
        }
    }

    int max_fwd_level = *std::max_element(spec_level.begin(), spec_level.end());

    // Group spec indices by forward level
    std::vector<std::vector<int>> fwd_groups(max_fwd_level + 1);
    for (int i = 0; i < static_cast<int>(specs.size()); i++) {
        fwd_groups[spec_level[i]].push_back(i);
    }

    // Max inverse parallelism = widest forward level
    int max_par = 0;
    for (const auto& g : fwd_groups) {
        max_par = std::max(max_par, static_cast<int>(g.size()));
    }

    // ── Step 3: Create CUDA streams and per-stream events ─────────────────────
    int n_streams = (num_streams > 0) ? num_streams : max_par;
    n_streams = std::max(1, std::min(n_streams, max_par));

    std::vector<cudaStream_t> inv_streams;
    bool owns_streams = (n_streams > 1);
    if (owns_streams) {
        inv_streams.resize(n_streams);
        for (auto& s : inv_streams) {
            FZ_CUDA_CHECK(cudaStreamCreate(&s));
        }
    } else {
        inv_streams.push_back(fallback_stream);
    }

    // One event per stream — records when the most recent work on that stream finished.
    // Used to synchronise the next batch: each stream waits for ALL events from the
    // previous batch before starting (conservative level barrier, same pattern as
    // the forward DAG's cudaStreamWaitEvent per dependency).
    std::vector<cudaEvent_t> batch_events(n_streams, nullptr);
    for (auto& e : batch_events) {
        FZ_CUDA_CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
    }
    // Track which events carry valid signals (false before first batch)
    std::vector<bool> event_has_work(n_streams, false);

    // ── Step 4: Track pool-owned buffer IDs for safe freeing ──────────────────
    std::unordered_set<int> pool_owned;

    // ── Timing infrastructure ─────────────────────────────────────────────────
    struct LevelTiming {
        int   fwd_level;
        int   n_specs;
        float wait_ms;    // cudaStreamWaitEvent setup
        float alloc_ms;   // pool.allocate() across all stages in level
        float exec_ms;    // stage->execute() across all stages in level
        float sizes_ms;   // getActualOutputSizesByName() across all stages
        float sync_ms;    // cudaStreamSynchronize at end of level
        float free_ms;    // pool.free() across all freed buffers
        float total_ms;   // wall-clock for the whole level
    };
    std::vector<LevelTiming> level_timings;
    level_timings.reserve(max_fwd_level + 1);

    float grand_wait_ms = 0, grand_alloc_ms = 0, grand_exec_ms = 0;
    float grand_sizes_ms = 0, grand_sync_ms = 0, grand_free_ms = 0;

    // ── Per-stage CUDA event timing ───────────────────────────────────────────
    struct StageEventInfo {
        cudaEvent_t cuda_start, cuda_end;
        Stage*      stage_ptr;
        int         inv_level;
        size_t      input_bytes;
        int         spec_idx;
    };
    std::vector<StageEventInfo> level_stage_cuda_infos; // reset each level
    std::vector<StageTimingResult> collected_stage_timings;

    auto t_inv_start = std::chrono::steady_clock::now();

    // ── Step 5: Execute in reverse level order (last fwd level first) ─────────
    for (int fwd_level = max_fwd_level; fwd_level >= 0; fwd_level--) {
        const auto& group = fwd_groups[fwd_level];
        auto t_level_start = std::chrono::steady_clock::now();

        // ── Phase A: stream barrier setup ────────────────────────────────────
        auto t_wait_start = std::chrono::steady_clock::now();
        // Before this batch: each stream must wait for ALL events from the previous
        // batch to ensure every upstream inverse result is ready in live_bufs.
        if (fwd_level < max_fwd_level) {
            for (int si = 0; si < n_streams; si++) {
                for (int ei = 0; ei < n_streams; ei++) {
                    if (event_has_work[ei]) {
                        FZ_CUDA_CHECK(cudaStreamWaitEvent(inv_streams[si], batch_events[ei]));
                    }
                }
            }
        }
        auto t_wait_end = std::chrono::steady_clock::now();

        float level_alloc_ms = 0, level_exec_ms = 0, level_sizes_ms = 0;

        // ── Phase B: launch all specs in this group ───────────────────────────
        // Launch all specs in this group concurrently (round-robin over streams)
        for (int gi = 0; gi < static_cast<int>(group.size()); gi++) {
            int spec_idx = group[gi];
            const auto& spec = specs[spec_idx];
            Stage* stage = spec.stage;
            int si = gi % n_streams;
            cudaStream_t exec_stream = inv_streams[si];

            // Inverse inputs = forward outputs of this stage
            std::vector<void*> inv_inputs;
            std::vector<size_t> inv_sizes;
            for (int buf_id : spec.fwd_output_ids) {
                auto it = live_bufs.find(buf_id);
                if (it != live_bufs.end() && it->second.first) {
                    inv_inputs.push_back(it->second.first);
                    inv_sizes.push_back(it->second.second);
                } else {
                    FZ_LOG(WARN, "Stage '%s' (fwd_level=%d): inv input buf %d missing",
                           stage->getName().c_str(), fwd_level, buf_id);
                    inv_inputs.push_back(nullptr);
                    inv_sizes.push_back(0);
                }
            }

            // Inverse outputs = one buffer per forward input of this stage.
            // For a single-input forward stage (most common): one output.
            // For a fan-in forward stage (merge): one output per forward input,
            //   all allocated and passed to the stage so it can reconstruct all branches.
            std::vector<void*> inv_outputs;

            // ── Time pool allocation ──────────────────────────────────────────
            auto t_alloc_start = std::chrono::steady_clock::now();
            for (int buf_id : spec.fwd_input_ids) {
                void* buf = pool.allocate(uncompressed_size, exec_stream, "inv_out");
                if (!buf) {
                    // Cleanup before throwing
                    for (int id : pool_owned) {
                        if (live_bufs.count(id) && live_bufs[id].first) {
                            pool.free(live_bufs[id].first, fallback_stream);
                        }
                    }
                    if (owns_streams) {
                        for (auto e : batch_events) cudaEventDestroy(e);
                        for (auto s : inv_streams) cudaStreamDestroy(s);
                    }
                    throw std::runtime_error(
                        "MemoryPool alloc failed for inverse stage '" + stage->getName() + "'");
                }
                inv_outputs.push_back(buf);
                live_bufs[buf_id] = { buf, 0 };  // size filled in after execute
                pool_owned.insert(buf_id);
            }
            auto t_alloc_end = std::chrono::steady_clock::now();
            level_alloc_ms += std::chrono::duration<float, std::milli>(t_alloc_end - t_alloc_start).count();

            // ── Time stage execute ────────────────────────────────────────────
            StageEventInfo sei;
            sei.stage_ptr   = stage;
            sei.inv_level   = max_fwd_level - fwd_level;
            sei.spec_idx    = spec_idx;
            sei.input_bytes = 0;
            for (size_t sz : inv_sizes) sei.input_bytes += sz;
            FZ_CUDA_CHECK(cudaEventCreate(&sei.cuda_start));
            FZ_CUDA_CHECK(cudaEventCreate(&sei.cuda_end));
            FZ_CUDA_CHECK(cudaEventRecord(sei.cuda_start, exec_stream));
            auto t_exec_start = std::chrono::steady_clock::now();
            stage->execute(exec_stream, &pool, inv_inputs, inv_outputs, inv_sizes);
            auto t_exec_end = std::chrono::steady_clock::now();
            FZ_CUDA_CHECK(cudaEventRecord(sei.cuda_end, exec_stream));
            level_stage_cuda_infos.push_back(std::move(sei));
            level_exec_ms += std::chrono::duration<float, std::milli>(t_exec_end - t_exec_start).count();

            // ── Time size retrieval ───────────────────────────────────────────
            auto t_sizes_start = std::chrono::steady_clock::now();
            // Retrieve actual output sizes by name and update live_bufs
            auto out_sizes = stage->getActualOutputSizesByName();
            auto out_names = stage->getOutputNames();
            for (size_t j = 0; j < spec.fwd_input_ids.size(); j++) {
                int buf_id = spec.fwd_input_ids[j];
                size_t actual = uncompressed_size;
                if (j < out_names.size()) {
                    auto sz_it = out_sizes.find(out_names[j]);
                    if (sz_it != out_sizes.end()) actual = sz_it->second;
                }
                live_bufs[buf_id].second = actual;
            }
            auto t_sizes_end = std::chrono::steady_clock::now();
            level_sizes_ms += std::chrono::duration<float, std::milli>(t_sizes_end - t_sizes_start).count();

            FZ_LOG(DEBUG,
                   "Inverse '%s' (fwd_level=%d, stream=%d): %zu in -> %zu out, "
                   "first_out=%.2f KB  [alloc=%.3f ms, exec=%.3f ms]",
                   stage->getName().c_str(), fwd_level, si,
                   inv_inputs.size(), inv_outputs.size(),
                   (inv_outputs.empty() ? 0.0
                    : live_bufs[spec.fwd_input_ids[0]].second / 1024.0),
                   std::chrono::duration<float, std::milli>(t_alloc_end - t_alloc_start).count(),
                   std::chrono::duration<float, std::milli>(t_exec_end - t_exec_start).count());

            // Record this stream's completion for the next batch barrier
            FZ_CUDA_CHECK(cudaEventRecord(batch_events[si], exec_stream));
            event_has_work[si] = true;
        }

        // ── Phase C: sync — forces CPU to wait for GPU kernels ────────────────
        auto t_sync_start = std::chrono::steady_clock::now();
        // After this batch completes (sync before freeing consumed inputs)
        for (int si = 0; si < n_streams; si++) {
            if (event_has_work[si]) {
                FZ_CUDA_CHECK(cudaStreamSynchronize(inv_streams[si]));
            }
        }
        auto t_sync_end = std::chrono::steady_clock::now();

        // ── Collect CUDA event timings now that all streams have synced ────────
        for (auto& ci : level_stage_cuda_infos) {
            float gpu_ms = 0.0f;
            FZ_CUDA_CHECK_WARN(cudaEventElapsedTime(&gpu_ms, ci.cuda_start, ci.cuda_end));
            FZ_CUDA_CHECK_WARN(cudaEventDestroy(ci.cuda_start));
            FZ_CUDA_CHECK_WARN(cudaEventDestroy(ci.cuda_end));

            size_t output_bytes = 0;
            for (int buf_id : specs[ci.spec_idx].fwd_input_ids) {
                auto oit = live_bufs.find(buf_id);
                if (oit != live_bufs.end()) output_bytes += oit->second.second;
            }

            StageTimingResult str;
            str.name         = ci.stage_ptr->getName();
            str.level        = ci.inv_level;
            str.elapsed_ms   = gpu_ms;
            str.input_bytes  = ci.input_bytes;
            str.output_bytes = output_bytes;
            collected_stage_timings.push_back(std::move(str));
        }
        level_stage_cuda_infos.clear();

        // ── Phase D: free consumed input buffers ──────────────────────────────
        auto t_free_start = std::chrono::steady_clock::now();
        // Free pool-owned buffers that were consumed by this batch
        // (= fwd_output_ids of specs in this group — produced by the previous batch)
        for (int gi = 0; gi < static_cast<int>(group.size()); gi++) {
            for (int buf_id : specs[group[gi]].fwd_output_ids) {
                if (pool_owned.count(buf_id)) {
                    pool.free(live_bufs.at(buf_id).first, fallback_stream);
                    live_bufs.erase(buf_id);
                    pool_owned.erase(buf_id);
                }
            }
        }
        auto t_free_end = std::chrono::steady_clock::now();

        auto t_level_end = std::chrono::steady_clock::now();

        LevelTiming lt;
        lt.fwd_level  = fwd_level;
        lt.n_specs    = static_cast<int>(group.size());
        lt.wait_ms    = std::chrono::duration<float, std::milli>(t_wait_end  - t_wait_start ).count();
        lt.alloc_ms   = level_alloc_ms;
        lt.exec_ms    = level_exec_ms;
        lt.sizes_ms   = level_sizes_ms;
        lt.sync_ms    = std::chrono::duration<float, std::milli>(t_sync_end  - t_sync_start ).count();
        lt.free_ms    = std::chrono::duration<float, std::milli>(t_free_end  - t_free_start ).count();
        lt.total_ms   = std::chrono::duration<float, std::milli>(t_level_end - t_level_start).count();
        level_timings.push_back(lt);

        grand_wait_ms  += lt.wait_ms;
        grand_alloc_ms += lt.alloc_ms;
        grand_exec_ms  += lt.exec_ms;
        grand_sizes_ms += lt.sizes_ms;
        grand_sync_ms  += lt.sync_ms;
        grand_free_ms  += lt.free_ms;
    }

    float grand_total_ms = std::chrono::duration<float, std::milli>(
        std::chrono::steady_clock::now() - t_inv_start).count();

    // ── Export stage timings to caller ────────────────────────────────────────
    if (timing_out) {
        *timing_out = std::move(collected_stage_timings);
    }

    // ── Print per-level timing summary ────────────────────────────────────────
    FZ_LOG(INFO, "");
    FZ_LOG(INFO, "===== runInversePipeline timing breakdown =====");
    FZ_LOG(INFO, "  %-10s  %-6s  %-10s  %-10s  %-10s  %-10s  %-10s  %-10s  %-10s",
           "inv-level", "specs",
           "wait(ms)", "alloc(ms)", "exec(ms)", "sizes(ms)", "sync(ms)", "free(ms)", "total(ms)");
    FZ_LOG(INFO, "  %s", std::string(97, '-').c_str());
    for (const auto& lt : level_timings) {
        // Inverse level = max_fwd_level - lt.fwd_level  (0 = first to run)
        int inv_lev = max_fwd_level - lt.fwd_level;
        FZ_LOG(INFO, "  %-10d  %-6d  %-10.3f  %-10.3f  %-10.3f  %-10.3f  %-10.3f  %-10.3f  %-10.3f",
               inv_lev, lt.n_specs,
               lt.wait_ms, lt.alloc_ms, lt.exec_ms,
               lt.sizes_ms, lt.sync_ms, lt.free_ms, lt.total_ms);
    }
    FZ_LOG(INFO, "  %s", std::string(97, '-').c_str());
    FZ_LOG(INFO, "  %-10s  %-6s  %-10.3f  %-10.3f  %-10.3f  %-10.3f  %-10.3f  %-10.3f  %-10.3f",
           "TOTAL", "",
           grand_wait_ms, grand_alloc_ms, grand_exec_ms,
           grand_sizes_ms, grand_sync_ms, grand_free_ms, grand_total_ms);
    FZ_LOG(INFO, "");
    FZ_LOG(INFO, "  Phase breakdown:  wait=%.1f%%  alloc=%.1f%%  exec=%.1f%%  sizes=%.1f%%  sync=%.1f%%  free=%.1f%%",
           100.f * grand_wait_ms  / grand_total_ms,
           100.f * grand_alloc_ms / grand_total_ms,
           100.f * grand_exec_ms  / grand_total_ms,
           100.f * grand_sizes_ms / grand_total_ms,
           100.f * grand_sync_ms  / grand_total_ms,
           100.f * grand_free_ms  / grand_total_ms);
    FZ_LOG(INFO, "===============================================");

    // ── Step 6: Destroy streams and events ────────────────────────────────────
    for (auto e : batch_events) FZ_CUDA_CHECK_WARN(cudaEventDestroy(e));
    if (owns_streams) {
        for (auto s : inv_streams) FZ_CUDA_CHECK_WARN(cudaStreamDestroy(s));
    }

    // ── Step 7: Return the final result ───────────────────────────────────────
    // The source stage is at fwd_level 0. Its forward input (the pipeline input)
    // is now overwritten in live_bufs with the inverse output = reconstructed data.
    if (fwd_groups[0].empty() || specs[fwd_groups[0][0]].fwd_input_ids.empty()) {
        throw std::runtime_error("runInversePipeline: cannot locate final result buffer");
    }
    int final_id = specs[fwd_groups[0][0]].fwd_input_ids[0];
    auto it = live_bufs.find(final_id);
    if (it == live_bufs.end() || !it->second.first) {
        throw std::runtime_error("runInversePipeline: no result produced");
    }
    return it->second;
}

// ── Helper: build per-level timing summary from per-stage CUDA event timings ──
static std::vector<LevelTimingResult> buildLevelTimings(
    const std::vector<StageTimingResult>& stages
) {
    std::unordered_map<int, LevelTimingResult> level_map;
    for (const auto& st : stages) {
        auto& lv = level_map[st.level];
        lv.level = st.level;
        lv.parallelism++;
        lv.elapsed_ms = std::max(lv.elapsed_ms, st.elapsed_ms);
    }
    std::vector<LevelTimingResult> levels;
    for (auto& [lvl, lv] : level_map) levels.push_back(lv);
    std::sort(levels.begin(), levels.end(),
              [](const LevelTimingResult& a, const LevelTimingResult& b) {
                  return a.level < b.level;
              });
    return levels;
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

    // Host-side wall-clock start (covers everything including output gathering)
    auto t_host_start = std::chrono::steady_clock::now();
    
    // Set external input pointer (zero-copy from user's buffer)
    dag_->setExternalPointer(input_buffer_id_, const_cast<void*>(d_input));
    
    // Execute DAG — time this portion separately for dag_elapsed_ms
    auto t_dag_start = std::chrono::steady_clock::now();
    dag_->execute(stream);
    auto t_dag_end = std::chrono::steady_clock::now();

    // required for postStreamSync() and cuda events
    FZ_CUDA_CHECK(cudaStreamSynchronize(stream));

    // Allow stages to finalize host-side state from GPU results
    // (e.g. Lorenzo reads back actual outlier count and trims output sizes).
    for (auto& stage_ptr : stages_) {
        stage_ptr->postStreamSync(stream);
    }

    // If profiling: collect CUDA event timings (stream already synced above)
    auto stage_timings = profiling_enabled_ ? dag_->collectTimings() : std::vector<StageTimingResult>{};
    
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
        concatOutputs(d_output, output_size, stream);
    } else {
        *d_output = dag_->getBuffer(output_buffer_ids_[0]);
        
        auto sizes_by_name = output_nodes_[0]->stage->getActualOutputSizesByName();
        auto output_names = output_nodes_[0]->stage->getOutputNames();
        
        *output_size = 0;
        if (!output_names.empty() && sizes_by_name.count(output_names[0])) {
            *output_size = sizes_by_name.at(output_names[0]);
        }
    }

    // Host-side wall-clock end
    auto t_host_end = std::chrono::steady_clock::now();

    float host_ms = std::chrono::duration<float, std::milli>(t_host_end - t_host_start).count();
    float dag_ms  = std::chrono::duration<float, std::milli>(t_dag_end  - t_dag_start ).count();

    // Build profiling result
    if (profiling_enabled_) {
        PipelinePerfResult r;
        r.is_compress     = true;
        r.host_elapsed_ms = host_ms;
        r.dag_elapsed_ms  = dag_ms;
        r.input_bytes     = input_size;
        r.output_bytes    = *output_size;
        r.stages          = std::move(stage_timings);

        // Build per-level aggregates
        r.levels = buildLevelTimings(r.stages);

        last_perf_result_ = std::move(r);
    }
    
    FZ_LOG(INFO, "Compress complete: %zu -> %zu bytes (host=%.2f ms, dag=%.2f ms, DAG=%.2f GB/s, pipeline=%.2f GB/s)",
           input_size, *output_size, host_ms, dag_ms,
           profiling_enabled_ ? last_perf_result_.throughput_gbs() : 0.0f,
           profiling_enabled_ ? last_perf_result_.pipeline_throughput_gbs() : 0.0f);
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

    auto t_host_start = std::chrono::steady_clock::now();

    FZ_LOG(INFO, "Decompressing");

    // ── Lookup tables built from the forward DAG state ───────────────────────

    // fwd output buffer ID → its compressed-data metadata
    std::unordered_map<int, const BufferMetadata*> fwd_buf_to_meta;
    for (const auto& meta : buffer_metadata_) {
        fwd_buf_to_meta[meta.buffer_id] = &meta;
    }

    // fwd output buffer ID → the forward DAGNode that *consumed* it (if any).
    // Outputs not in this map are direct pipeline outputs (no downstream stage).
    std::unordered_map<int, DAGNode*> connected_fwd_bufs;
    for (const auto& conn : connections_) {
        DAGNode* prod_node = stage_to_node_.at(conn.producer);
        int buf_id = prod_node->output_index_to_buffer_id.at(conn.output_index);
        connected_fwd_bufs[buf_id] = stage_to_node_.at(conn.dependent);
    }

    // ── Build compressed-data pointer map ────────────────────────────────────
    // Determines which device pointer feeds each compressed buffer in the
    // inverse DAG.  Three cases:
    //
    //  (a) d_input != nullptr, single output (no concat):
    //        d_input IS the compressed buffer — use it directly.
    //
    //  (b) d_input != nullptr, multi-output (concat format):
    //        Parse the concat layout using the sizes already stored in
    //        buffer_metadata_ — no GPU read-back needed.
    //        Format: [num_bufs:u32][size:u64][data]... (see writeConcatBuffer)
    //
    //  (c) d_input == nullptr (or not provided):
    //        Fall back to the forward DAG's own live output buffers.  This is
    //        the standard in-memory path: call compress() then decompress()
    //        on the same pipeline instance without resetting it.
    std::unordered_map<int, void*> compressed_ptrs;
    if (d_input != nullptr) {
        if (!needs_concat_) {
            // Single output — d_input is exactly the one compressed buffer
            if (!buffer_metadata_.empty()) {
                compressed_ptrs[buffer_metadata_[0].buffer_id] =
                    const_cast<void*>(d_input);
            }
        } else {
            // Multi-output concat: compute per-buffer slice offsets
            size_t byte_offset = sizeof(uint32_t); // skip num_buffers header
            for (const auto& meta : buffer_metadata_) {
                byte_offset += sizeof(uint64_t);    // skip per-buffer size field
                compressed_ptrs[meta.buffer_id] =
                    static_cast<uint8_t*>(const_cast<void*>(d_input)) + byte_offset;
                byte_offset += meta.actual_size;
            }
        }
    } else {
        // Fall back: read directly from the forward DAG's live buffers
        for (const auto& meta : buffer_metadata_) {
            compressed_ptrs[meta.buffer_id] = dag_->getBuffer(meta.buffer_id);
        }
    }

    // ── Switch all stages to inverse mode ────────────────────────────────────
    for (auto& s : stages_) s->setInverse(true);

    // ── Build the inverse CompressionDAG ─────────────────────────────────────
    auto inv_dag = std::make_unique<CompressionDAG>(mem_pool_.get(), strategy_);
    std::unordered_map<Stage*, DAGNode*> inv_nodes;

    // Step 1: Add stages in REVERSE forward topological order.
    // CompressionDAG::assignLevels() does a single forward pass over nodes_ and
    // requires parents to appear before children — so the inverse-DAG parents
    // (= forward-DAG leaves) must be added first.
    const auto& fwd_levels = dag_->getLevels();
    for (int fwd_lev = static_cast<int>(fwd_levels.size()) - 1; fwd_lev >= 0; fwd_lev--) {
        for (auto* fwd_node : fwd_levels[fwd_lev]) {
            Stage* stage = fwd_node->stage;
            DAGNode* node = inv_dag->addStage(stage, stage->getName());

            // Pre-allocate one unconnected output slot per inverse output.
            // (In inverse mode getNumOutputs() returns the correct inverse count.)
            size_t num_out = stage->getNumOutputs();
            auto out_names = stage->getOutputNames();
            for (size_t i = 0; i < num_out; i++) {
                std::string n = (i < out_names.size()) ? out_names[i] : std::to_string(i);
                inv_dag->addUnconnectedOutput(node, input_size_,
                                              static_cast<int>(i),
                                              stage->getName() + "." + n);
            }
            inv_nodes[stage] = node;
        }
    }

    // Step 2: Wire inverse inputs in the exact ORDER of the forward node's
    // output_buffer_ids — this guarantees stage->execute() receives buffers in
    // the same positional order that fwd_output_ids implied in runInversePipeline.
    for (int fwd_lev = 0; fwd_lev < static_cast<int>(fwd_levels.size()); fwd_lev++) {
        for (auto* fwd_node : fwd_levels[fwd_lev]) {
            Stage* stage = fwd_node->stage;
            DAGNode* inv_node = inv_nodes[stage];

            for (int fwd_out_buf_id : fwd_node->output_buffer_ids) {
                auto conn_it = connected_fwd_bufs.find(fwd_out_buf_id);

                if (conn_it != connected_fwd_bufs.end()) {
                    // This forward output fed a downstream stage.
                    // In inverse: the downstream stage's inverse PRODUCES this buffer.
                    DAGNode* fwd_consumer = conn_it->second;
                    DAGNode* inv_producer = inv_nodes[fwd_consumer->stage];

                    // Determine which output index of inv_producer carries this buffer.
                    // inv_producer's output[k] reconstructs fwd_consumer's input[k].
                    const auto& cons_inputs = fwd_consumer->input_buffer_ids;
                    int pos = -1;
                    for (int k = 0; k < static_cast<int>(cons_inputs.size()); k++) {
                        if (cons_inputs[k] == fwd_out_buf_id) { pos = k; break; }
                    }
                    if (pos < 0) {
                        for (auto& s : stages_) s->setInverse(false);
                        throw std::runtime_error(
                            "Inverse DAG: buffer " + std::to_string(fwd_out_buf_id) +
                            " not found in consumer '" + fwd_consumer->name + "' inputs");
                    }

                    bool ok = inv_dag->connectExistingOutput(inv_producer, inv_node, pos);
                    if (!ok) {
                        for (auto& s : stages_) s->setInverse(false);
                        throw std::runtime_error(
                            "Inverse DAG: connectExistingOutput failed for output " +
                            std::to_string(pos) + " of stage '" + inv_producer->name + "'");
                    }

                    FZ_LOG(DEBUG, "Inverse edge: %s.out[%d] -> %s (fwd_buf=%d)",
                           inv_producer->name.c_str(), pos,
                           inv_node->name.c_str(), fwd_out_buf_id);
                } else {
                    // Direct pipeline output → inject as external input to inv_node.
                    auto meta_it = fwd_buf_to_meta.find(fwd_out_buf_id);
                    if (meta_it == fwd_buf_to_meta.end()) {
                        for (auto& s : stages_) s->setInverse(false);
                        throw std::runtime_error(
                            "Inverse DAG: buf " + std::to_string(fwd_out_buf_id) +
                            " not found in buffer_metadata_");
                    }
                    const BufferMetadata* meta = meta_it->second;

                    inv_dag->setInputBuffer(inv_node, meta->actual_size,
                                            "inv_ext_" + meta->name);
                    int ext_buf_id = inv_node->input_buffer_ids.back();
                    auto cptr_it = compressed_ptrs.find(fwd_out_buf_id);
                    void* compressed_buf = (cptr_it != compressed_ptrs.end())
                        ? cptr_it->second
                        : dag_->getBuffer(meta->buffer_id); // safety fallback
                    inv_dag->setExternalPointer(ext_buf_id, compressed_buf);

                    FZ_LOG(DEBUG, "Inverse external input: '%s' %zu bytes -> stage '%s' (ptr=%p)",
                           meta->name.c_str(), meta->actual_size, stage->getName().c_str(),
                           compressed_buf);
                }
            }
        }
    }

    // Step 3: The inverse sink = former forward source (input_node_->stage).
    // Its first output buffer holds the fully-reconstructed data — mark it
    // persistent so DAG::execute() doesn't free it before we copy it out.
    Stage* fwd_source = input_node_->stage;
    DAGNode* inv_sink = inv_nodes.at(fwd_source);
    if (inv_sink->output_buffer_ids.empty()) {
        for (auto& s : stages_) s->setInverse(false);
        throw std::runtime_error("Inverse DAG: sink stage has no output buffers");
    }
    int inv_result_buf_id = inv_sink->output_buffer_ids[0];
    inv_dag->setBufferPersistent(inv_result_buf_id, true);

    // Step 4: Finalize the inverse DAG (assigns levels and streams).
    if (profiling_enabled_) {
        inv_dag->enableProfiling(true);
    }
    inv_dag->finalize();

    // Propagate estimated buffer sizes forward through the inverse DAG's levels.
    // External input sizes were set during setInputBuffer; this propagates them
    // through internal intermediate buffers using estimateOutputSizes().
    for (const auto& inv_level_nodes : inv_dag->getLevels()) {
        for (auto* node : inv_level_nodes) {
            std::vector<size_t> in_sizes;
            for (int buf_id : node->input_buffer_ids) {
                in_sizes.push_back(inv_dag->getBufferSize(buf_id));
            }
            auto est = node->stage->estimateOutputSizes(in_sizes);
            for (size_t i = 0;
                 i < node->output_buffer_ids.size() && i < est.size(); i++) {
                inv_dag->updateBufferSize(node->output_buffer_ids[i], est[i]);
            }
        }
    }

    // The inverse sink's output is the reconstructed original data.
    // Override any estimation with the exact known input_size_ — the inverse
    // predictor (e.g. Lorenzo) may report codes-size instead of float-data-size.
    inv_dag->updateBufferSize(inv_result_buf_id, input_size_);

    if (strategy_ == MemoryStrategy::PREALLOCATE) {
        inv_dag->preallocateBuffers();
    }

    FZ_LOG(DEBUG, "Inverse DAG: %zu levels, max_parallelism=%d, strategy=%s",
           inv_dag->getLevels().size(),
           inv_dag->getMaxParallelism(),
           strategy_ == MemoryStrategy::MINIMAL ? "MINIMAL" :
           strategy_ == MemoryStrategy::PIPELINE ? "PIPELINE" : "PREALLOCATE");

    // Step 5: Execute — uses fine-grained per-dependency cudaStreamWaitEvent,
    // stream assignment from assignStreams(), and strategy-aware buffer lifetimes.
    auto t_dag_start = std::chrono::steady_clock::now();
    inv_dag->execute(stream);
    auto t_dag_end = std::chrono::steady_clock::now();
    FZ_CUDA_CHECK(cudaStreamSynchronize(stream));

    // Collect per-stage timings (collectTimings() syncs all owned streams internally)
    auto stage_timings = profiling_enabled_ ? inv_dag->collectTimings() : std::vector<StageTimingResult>{};

    // Step 6: Extract result, copy to a caller-owned cudaMalloc buffer.
    void* d_inv_result = inv_dag->getBuffer(inv_result_buf_id);

    // Prefer the stage's post-execution actual size over the estimated DAG size.
    size_t actual_size = inv_dag->getBufferSize(inv_result_buf_id);
    {
        auto post_sizes = fwd_source->getActualOutputSizesByName();
        auto post_names = fwd_source->getOutputNames();
        if (!post_names.empty() && post_sizes.count(post_names[0])) {
            actual_size = post_sizes.at(post_names[0]);
        }
    }

    void* d_final = nullptr;
    cudaError_t err = cudaMalloc(&d_final, actual_size);
    if (err != cudaSuccess) {
        inv_dag->reset(stream);
        mem_pool_->free(d_inv_result, stream);
        for (auto& s : stages_) s->setInverse(false);
        throw std::runtime_error("cudaMalloc for decompress output failed");
    }
    FZ_CUDA_CHECK(cudaMemcpyAsync(d_final, d_inv_result, actual_size,
                    cudaMemcpyDeviceToDevice, stream));
    FZ_CUDA_CHECK(cudaStreamSynchronize(stream));

    // Release inverse DAG's pool memory: reset() frees non-persistent
    // intermediate buffers; the persistent result buffer needs explicit free.
    inv_dag->reset(stream);
    mem_pool_->free(d_inv_result, stream);
    // External input buffers are is_external=true → reset() skips them (correct,
    // they are owned by the forward DAG and must remain live).

    // Restore forward mode for subsequent compress() calls.
    for (auto& s : stages_) s->setInverse(false);

    *d_output = d_final;
    *output_size = actual_size;

    // Host-side wall-clock end
    auto t_host_end = std::chrono::steady_clock::now();
    float host_ms = std::chrono::duration<float, std::milli>(t_host_end - t_host_start).count();
    float dag_ms  = std::chrono::duration<float, std::milli>(t_dag_end  - t_dag_start ).count();

    // Build profiling result
    if (profiling_enabled_) {
        PipelinePerfResult r;
        r.is_compress     = false;
        r.host_elapsed_ms = host_ms;
        r.dag_elapsed_ms  = dag_ms;
        r.input_bytes     = input_size;
        r.output_bytes    = actual_size;
        r.stages          = std::move(stage_timings);

        // Build per-level aggregates
        r.levels = buildLevelTimings(r.stages);

        last_perf_result_ = std::move(r);
    }

    FZ_LOG(INFO, "Decompress complete (DAG-native): %zu bytes (host=%.2f ms, dag=%.2f ms, DAG=%.2f GB/s, pipeline=%.2f GB/s)",
           actual_size, host_ms, dag_ms,
           profiling_enabled_ ? last_perf_result_.throughput_gbs() : 0.0f,
           profiling_enabled_ ? last_perf_result_.pipeline_throughput_gbs() : 0.0f);
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
    cudaStream_t stream,
    PipelinePerfResult* perf_out
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

    // ── Compute-only timing (excludes file I/O and H→D copy) ─────────────────
    // runInversePipeline synchronizes at the end of every level internally, so
    // chrono is accurate here — the host blocks on the GPU inside the call.
    auto t_compute_start = std::chrono::steady_clock::now();

    std::vector<StageTimingResult> inv_stage_timings;
    std::pair<void*, size_t> result;
    {
        nvtx3::scoped_range compute_range{"decompress::compute"};
        try {
            result = runInversePipeline(specs, live_bufs,
                                        fh.core.uncompressed_size, local_pool, stream,
                                        0, &inv_stage_timings);
        } catch (...) {
            local_pool.free(d_compressed, stream);
            throw;
        }
    }

    // 7. Copy from pool buffer to a plain cudaMalloc buffer the caller can cudaFree
    size_t final_size = result.second > 0 ? result.second : fh.core.uncompressed_size;
    void* d_final = nullptr;
    cudaError_t err = cudaMalloc(&d_final, final_size);
    if (err != cudaSuccess) {
        local_pool.free(d_compressed, stream);
        throw std::runtime_error("cudaMalloc for decompressed output failed");
    }
    FZ_CUDA_CHECK(cudaMemcpyAsync(d_final, result.first, final_size, cudaMemcpyDeviceToDevice, stream));
    FZ_CUDA_CHECK(cudaStreamSynchronize(stream));

    auto t_compute_end = std::chrono::steady_clock::now();
    float compute_ms = std::chrono::duration<float, std::milli>(t_compute_end - t_compute_start).count();

    local_pool.free(d_compressed, stream);

    *d_output = d_final;
    *output_size = final_size;

    // Compute DAG elapsed from CUDA event stage timings (sum of critical-path levels);
    // fall back to compute_ms if no event data was collected.
    auto log_levels = buildLevelTimings(inv_stage_timings);
    float dag_ms_log = 0.0f;
    for (const auto& lv : log_levels) dag_ms_log += lv.elapsed_ms;
    if (dag_ms_log <= 0.0f) dag_ms_log = compute_ms;

    if (perf_out) {
        perf_out->is_compress     = false;
        perf_out->host_elapsed_ms = compute_ms;
        perf_out->dag_elapsed_ms  = dag_ms_log;
        perf_out->input_bytes     = fh.core.compressed_size;
        perf_out->output_bytes    = final_size;
        perf_out->stages          = std::move(inv_stage_timings);
        perf_out->levels          = std::move(log_levels);
    }

    float dag_tput  = static_cast<float>(final_size) / (dag_ms_log  * 1e-3f) / 1e9f;
    float pipe_tput = static_cast<float>(final_size) / (compute_ms  * 1e-3f) / 1e9f;
    FZ_LOG(INFO, "Decompression complete: %.2f MB -> %zu bytes (compute=%.2f ms, dag=%.2f ms, DAG=%.2f GB/s, pipeline=%.2f GB/s)",
           fh.core.compressed_size / (1024.0 * 1024.0), final_size,
           compute_ms, dag_ms_log, dag_tput, pipe_tput);
}



} // namespace fz

