#pragma once

#include "pipeline/perf.h"

#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace fz {

// Forward declarations
class Stage;
class MemoryPool;

/** Memory allocation strategy for pipeline execution. */
enum class MemoryStrategy {
    MINIMAL,      ///< Allocate on-demand, free at last consumer. Lowest peak memory.
    PREALLOCATE   ///< Allocate everything upfront at finalize(). Required for graph mode.
};

/** Buffer metadata and lifetime tracking. */
struct BufferInfo {
    size_t size;
    size_t initial_size;
    size_t allocated_size;
    void*  d_ptr;
    std::string tag;

    int remaining_consumers;
    std::vector<int> consumer_stage_ids;
    int producer_stage_id;
    int producer_output_index;

    bool is_allocated;
    bool is_persistent;  ///< If true, survives reset() until DAG destruction.
    bool is_external;    ///< If true, pointer is caller-owned — DAG never allocs or frees.

    BufferInfo()
        : size(0), initial_size(0), allocated_size(0), d_ptr(nullptr), tag(""),
          remaining_consumers(0), producer_stage_id(-1), producer_output_index(0),
          is_allocated(false), is_persistent(false), is_external(false) {}
};

/** DAG node representing one compression stage. */
struct DAGNode {
    int     id;
    Stage*  stage;
    std::string name;

    std::vector<int>     input_buffer_ids;
    std::vector<int>     output_buffer_ids;
    std::unordered_map<int, int> output_index_to_buffer_id;

    std::vector<DAGNode*> dependencies;
    std::vector<DAGNode*> dependents;

    int          level;
    int          execution_order;
    cudaStream_t stream;

    bool         is_executed;
    cudaEvent_t  completion_event;
    cudaEvent_t  start_event;  ///< Non-null only when profiling is enabled.

    // Pre-sized vectors for execute() — allocated at finalize(), reused every call
    // to avoid per-call heap allocations of the input/output/sizes arrays.
    std::vector<void*>   exec_inputs;
    std::vector<void*>   exec_outputs;
    std::vector<size_t>  exec_sizes;

    DAGNode(Stage* s = nullptr)
        : id(-1), stage(s), level(-1), execution_order(-1),
          stream(nullptr), is_executed(false), completion_event(nullptr), start_event(nullptr) {}
};

/**
 * Execution DAG for compression pipelines.
 *
 * Manages buffer lifetimes, stream assignment, and level-based parallel execution.
 * Pipeline uses this internally; prefer the Pipeline API over direct DAG access
 * unless you need low-level control.
 *
 * @note Not thread-safe. All calls must originate from the same host thread.
 */
class CompressionDAG {
public:
    CompressionDAG(MemoryPool* mem_pool, MemoryStrategy strategy = MemoryStrategy::MINIMAL);
    ~CompressionDAG();

    // ── Construction ──────────────────────────────────────────────────────────

    /** Add a stage and return its node for wiring dependencies. */
    DAGNode* addStage(Stage* stage, const std::string& name = "");

    /**
     * Add a dependency between two nodes, creating an intermediate buffer.
     * @param dependent    The node that consumes the output.
     * @param dependency   The node that produces the output.
     * @param buffer_size  Byte capacity of the intermediate buffer (0 = infer later).
     * @param output_index Which output port of `dependency` to connect.
     * @return Buffer ID of the created intermediate buffer.
     */
    int addDependency(DAGNode* dependent, DAGNode* dependency,
                      size_t buffer_size = 0, int output_index = 0);

    void setInputBuffer(DAGNode* node, size_t size, const std::string& tag = "input");
    void setOutputBuffer(DAGNode* node, size_t size, const std::string& tag = "output");

    /**
     * Add a placeholder buffer for an output port that has no downstream consumer.
     * The stage still needs a buffer for every declared output even if it's unused.
     */
    int addUnconnectedOutput(DAGNode* node, size_t size, int output_index, const std::string& tag);

    /**
     * Promote an unconnected output buffer to a connected one when a consumer is wired.
     * Reuses the existing allocation rather than creating a new buffer.
     */
    bool connectExistingOutput(DAGNode* producer, DAGNode* consumer, int output_index);

    void updateBufferTag(int buffer_id, const std::string& tag);
    void setBufferPersistent(int buffer_id, bool persistent);

    /** Assign execution levels and streams. Must be called before execute(). */
    void finalize();

    /** Set number of CUDA streams for parallel level execution. */
    void configureStreams(int num_streams);

    // ── Execution ─────────────────────────────────────────────────────────────

    void execute(cudaStream_t stream);

    /**
     * Pre-allocate all buffers upfront. Called automatically by finalize() for
     * PREALLOCATE strategy; call explicitly when input sizes change between runs.
     */
    void preallocateBuffers(cudaStream_t stream = 0);

    /** Free non-persistent buffers and reset execution state. */
    void reset(cudaStream_t stream = 0);

    // ── Buffer access ─────────────────────────────────────────────────────────

    void* getBuffer(int buffer_id) const;

    /**
     * Mark a buffer as externally managed — DAG will not allocate or free it.
     * Use to pass user-owned device pointers directly into the DAG (zero-copy input).
     */
    void setExternalPointer(int buffer_id, void* external_ptr);

    void updateBufferSize(int buffer_id, size_t new_size);

    // ── Query & debug ─────────────────────────────────────────────────────────

    /** @internal Used by printDAG(); not part of the stable public API. */
    size_t getTotalBufferSize() const;

    /**
     * Peak bytes that must be held simultaneously in the pool.
     *
     * PREALLOCATE: sum of all non-external buffer sizes (all live at once).
     * MINIMAL: simulates level-by-level alloc/free to find peak concurrent live bytes.
     *
     * Must be called after finalize() and propagateBufferSizes().
     */
    size_t computeTopoPoolSize() const;

    size_t getPeakMemoryUsage()    const { return peak_memory_usage_; }
    size_t getCurrentMemoryUsage() const { return current_memory_usage_; }
    size_t getBufferSize(int buffer_id) const;
    const BufferInfo& getBufferInfo(int buffer_id) const;
    const std::vector<std::vector<DAGNode*>>& getLevels() const { return levels_; }
    const std::vector<DAGNode*>& getNodes() const { return nodes_; }

    /** Maximum nodes at any single level — useful for choosing stream count. */
    int getMaxParallelism() const;

    /** @internal Used by Pipeline::finalize(); not part of the stable public API. */
    size_t getStreamCount() const { return streams_.size(); }

    void printDAG() const;
    void printBufferLifetimes() const;

    /**
     * Enable runtime buffer-overwrite detection.
     * After each stage executes, checks actual output size ≤ allocated capacity.
     * Always active in debug builds regardless of this flag.
     */
    void enableBoundsCheck(bool enable) { bounds_check_enabled_ = enable; }
    bool isBoundsCheckEnabled() const   { return bounds_check_enabled_; }

    /**
     * Enable or disable buffer coloring for PREALLOCATE mode (default: enabled).
     * Coloring aliases non-overlapping buffers to reduce peak pool size.
     * Must be called before finalize().
     */
    void setColoringEnabled(bool enable) { coloring_disabled_ = !enable; }
    bool isColoringEnabled() const       { return coloring_applied_; }
    size_t getColorRegionCount() const   { return color_region_sizes_.size(); }

    /**
     * Enable or disable CUDA Graph capture mode.
     *
     * When true, execute() suppresses host-synchronous operations so the call
     * is safe inside a cudaStreamBeginCapture bracket. Throws if any stage
     * returns false from isGraphCompatible().
     */
    void setCaptureMode(bool capture);
    bool isCaptureMode() const { return capture_mode_; }

    // ── Profiling ─────────────────────────────────────────────────────────────

    /**
     * Enable per-stage CUDA event profiling. Zero overhead when disabled.
     * Call collectTimings() after execute() + stream sync to read results.
     */
    void enableProfiling(bool enable);
    bool isProfilingEnabled() const { return profiling_enabled_; }

    /** Sync all DAG streams and collect per-stage timing results. */
    std::vector<StageTimingResult> collectTimings();

private:
    MemoryPool*     mem_pool_;
    MemoryStrategy  strategy_;

    std::vector<DAGNode*>           nodes_;
    std::unordered_map<int, BufferInfo> buffers_;

    int  next_buffer_id_;
    bool is_finalized_;

    std::vector<cudaStream_t> streams_;
    bool owns_streams_;

    std::vector<std::vector<DAGNode*>> levels_;
    int max_level_;

    size_t current_memory_usage_;
    size_t peak_memory_usage_;

    bool profiling_enabled_;
    bool bounds_check_enabled_;
    bool capture_mode_;

    // Buffer coloring (PREALLOCATE only). Non-overlapping buffers share a color
    // and are aliased into one pool region. color_region_ptrs_ owns the allocations.
    bool coloring_disabled_;
    bool coloring_applied_;
    std::unordered_map<int, int> buffer_color_;
    std::vector<size_t>          color_region_sizes_;
    std::vector<void*>           color_region_ptrs_;

    void assignLevels();
    void assignStreams();
    void allocateBuffer(int buffer_id, cudaStream_t stream);
    void freeBuffer(int buffer_id, cudaStream_t stream);
    void planPreallocation();
    void colorBuffers();
};

} // namespace fz
