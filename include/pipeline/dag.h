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

/**
 * Memory management strategy for pipeline execution
 */
enum class MemoryStrategy {
    MINIMAL,      // Allocate on-demand, free ASAP (lowest peak memory)
    PREALLOCATE   // Allocate everything upfront at finalize() (fastest, required for graph mode)
};

/**
 * Buffer metadata and lifetime tracking
 */
struct BufferInfo {
    size_t size;                        // Currently required size
    size_t initial_size;                // Original size requested during setup
    size_t allocated_size;              // Actual reserved size in memory pool
    void* d_ptr;                        // Device pointer
    std::string tag;                    // Debug tag
    
    // Lifecycle management
    int remaining_consumers;            // Current number of stages that still need to read this
    std::vector<int> consumer_stage_ids;// IDs of all stages that consume this buffer
    int producer_stage_id;              // ID of stage that produced it (-1 if input)
    int producer_output_index;          // Which output index of producer this corresponds to
    
    bool is_allocated;
    bool is_persistent;                 // If true, don't free until DAG destruction
    bool is_external;                   // If true, pointer managed externally
    
    BufferInfo() 
        : size(0), initial_size(0), allocated_size(0), d_ptr(nullptr), tag(""),
          remaining_consumers(0), producer_stage_id(-1), producer_output_index(0),
          is_allocated(false), is_persistent(false), is_external(false) {}
};

/**
 * DAG node representing a compression stage
 */
struct DAGNode {
    int id;                             // Unique node ID
    Stage* stage;                       // Pointer to stage implementation
    std::string name;                   // Debug name
    
    std::vector<int> input_buffer_ids;  // Input buffer IDs
    std::vector<int> output_buffer_ids; // Output buffer IDs (sequential)
    std::unordered_map<int, int> output_index_to_buffer_id; // output_idx -> buffer_id (fast lookup)
    
    // Dependencies
    std::vector<DAGNode*> dependencies; // Nodes that must execute before this
    std::vector<DAGNode*> dependents;   // Nodes that depend on this
    
    // Execution planning
    int level;                          // Execution level (for parallelism)
    int execution_order;                // Order within level
    cudaStream_t stream;                // Assigned stream
    
    // Execution state
    bool is_executed;
    cudaEvent_t completion_event;       // When this stage completes
    cudaEvent_t start_event;            // When this stage starts (profiling only, nullptr = disabled)

    // Pre-sized scratch vectors for execute() — sized at finalize(), reused
    // every call to avoid per-call heap allocation of the input/output/sizes arrays.
    std::vector<void*>  exec_inputs;
    std::vector<void*>  exec_outputs;
    std::vector<size_t> exec_sizes;
    
    DAGNode(Stage* s = nullptr) 
        : id(-1), stage(s), level(-1), execution_order(-1), 
          stream(nullptr), is_executed(false), completion_event(nullptr), start_event(nullptr) {}
};

/**
 * DAG for managing compression pipeline execution and buffer lifetimes
 *
 * @note Thread Safety: CompressionDAG is NOT thread-safe. All construction,
 * finalization, and execution calls must originate from the same host thread.
 * Intra-level parallelism is achieved via multiple CUDA streams managed
 * internally, not via host-thread concurrency.
 */
class CompressionDAG {
public:
    CompressionDAG(MemoryPool* mem_pool, MemoryStrategy strategy = MemoryStrategy::MINIMAL);
    ~CompressionDAG();
    
    // ========== DAG Construction ==========
    
    /**
     * Add a stage to the DAG
     * @return Node pointer for connecting dependencies
     */
    DAGNode* addStage(Stage* stage, const std::string& name = "");
    
    /**
     * Add execution dependency between stages
     * Automatically creates intermediate buffer between them
     * @param dependent Node that depends on another
     * @param dependency Node that must complete first
     * @param buffer_size Size of intermediate buffer (0 = estimate from stage)
     * @param output_index Which output of dependency to connect (for multi-output stages)
     * @return Buffer ID of created intermediate buffer
     */
    int addDependency(DAGNode* dependent, DAGNode* dependency, 
                     size_t buffer_size = 0, int output_index = 0);
    
    /**
     * Set input buffer for the pipeline (persistent)
     */
    void setInputBuffer(DAGNode* node, size_t size, const std::string& tag = "input");
    
    /**
     * Set output buffer for the pipeline (persistent)
     */
    void setOutputBuffer(DAGNode* node, size_t size, const std::string& tag = "output");
    
    /**
     * Add an unconnected output buffer for a stage
     * Used when a stage declares more outputs than are connected to downstream stages
     * The stage still expects buffers for all outputs, even if not consumed
     * 
     * @param node Stage node
     * @param size Buffer size
     * @param output_index Which output index this buffer represents
     * @param tag Debug tag
     * @return Buffer ID
     */
    int addUnconnectedOutput(DAGNode* node, size_t size, int output_index, const std::string& tag);
    
    /**
     * Convert an unconnected output buffer to a connected one
     * Reuses existing pre-allocated buffer and adds consumer relationship
     * 
     * @param producer Stage producing the output
     * @param consumer Stage consuming the output
     * @param output_index Which output of producer to connect
     * @return true if connection succeeded, false if buffer doesn't exist
     */
    bool connectExistingOutput(DAGNode* producer, DAGNode* consumer, int output_index);
    
    /**
     * Update a buffer's tag (for renaming during topology finalization)
     */
    void updateBufferTag(int buffer_id, const std::string& tag);
    
    /**
     * Mark a buffer as persistent or non-persistent
     */
    void setBufferPersistent(int buffer_id, bool persistent);
    
    /**
     * Finalize DAG - assigns levels and streams for execution
     */
    void finalize();
    
    /**
     * Configure streams for parallel execution
     * @param num_streams Number of streams to create
     */
    void configureStreams(int num_streams);
    
    // ========== Execution ==========
    
    /**
     * Execute the DAG using the configured memory strategy
     */
    void execute(cudaStream_t stream);

    /**
     * Pre-allocate all buffers upfront (called automatically by finalize() for
     * PREALLOCATE strategy, or explicitly when input sizes change between runs).
     *
     * @param stream CUDA stream for async pool allocations (default: 0)
     */
    void preallocateBuffers(cudaStream_t stream = 0);
    
    /**
     * Reset DAG for next execution (frees non-persistent buffers)
     */
    void reset(cudaStream_t stream = 0);
    
    // ========== Buffer Access ==========
    
    /**
     * Get device pointer for a buffer (must be allocated first)
     */
    void* getBuffer(int buffer_id) const;
    
    /**
     * Set external pointer for a buffer (zero-copy input)
     * 
     * Marks buffer as externally managed - DAG won't allocate or free it.
     * Useful for passing user's input buffer directly without copying.
     * 
     * @param buffer_id Buffer ID to set external pointer for
     * @param external_ptr User-provided device pointer
     */
    void setExternalPointer(int buffer_id, void* external_ptr);
    
    /**
     * Update buffer size (e.g., when actual input size is known)
     * 
     * @param buffer_id Buffer ID to update
     * @param new_size New size in bytes
     */
    void updateBufferSize(int buffer_id, size_t new_size);
    
    // ========== Query & Debug ==========
    
    size_t getTotalBufferSize() const;

    /**
     * Compute a topology-aware pool size from the current buffer size estimates.
     *
     * PREALLOCATE: returns the sum of all non-external buffer sizes, since all
     *   buffers are live simultaneously for the lifetime of the pipeline.
     *
     * MINIMAL: simulates level-by-level allocation and deallocation
     *   to find the peak concurrent live bytes.  A buffer becomes live when its
     *   producer executes and is freed after its last consumer finishes.
     *
     * Must be called after finalize() and propagateBufferSizes() so that both
     * the level structure and buffer sizes are populated.
     *
     * @return Peak bytes that must be held in the pool at any one time.
     */
    size_t computeTopoPoolSize() const;
    size_t getPeakMemoryUsage() const { return peak_memory_usage_; }
    size_t getCurrentMemoryUsage() const { return current_memory_usage_; }
    size_t getBufferSize(int buffer_id) const;
    const BufferInfo& getBufferInfo(int buffer_id) const;
    const std::vector<std::vector<DAGNode*>>& getLevels() const { return levels_; }
    const std::vector<DAGNode*>& getNodes() const { return nodes_; }
    int getMaxParallelism() const;  // Max nodes at any level (optimal stream count)
    size_t getStreamCount() const { return streams_.size(); }
    void printDAG() const;
    void printBufferLifetimes() const;

    /**
     * Enable or disable runtime buffer overwrite detection.
     *
     * When enabled, after each stage's execute() call the DAG compares the
     * stage's reported actual output sizes against the allocated buffer
     * capacities and throws std::runtime_error if any output exceeded its
     * buffer.  This check is also always active in debug builds (NDEBUG not
     * defined) regardless of this flag.
     *
     * @param enable  true to activate, false to deactivate (default: false)
     */
    void enableBoundsCheck(bool enable) { bounds_check_enabled_ = enable; }
    bool isBoundsCheckEnabled() const   { return bounds_check_enabled_; }

    /**
     * Disable buffer coloring for PREALLOCATE mode (default: enabled).
     *
     * Useful when inspecting individual buffer contents with cuda-memcheck or
     * the bounds-check system — aliased pointers make per-buffer inspection
     * ambiguous.  Must be called before finalize().
     */
    void disableColoring(bool disable) { coloring_disabled_ = disable; }
    bool isColoringApplied() const     { return coloring_applied_; }
    size_t getColorRegionCount() const { return color_region_sizes_.size(); }

    /**
     * Enable or disable CUDA Graph capture mode.
     *
     * When true, DAG::execute() suppresses any host-synchronous operations
     * (getActualOutputSize() readbacks, bounds checks) so that the call can
     * occur inside a cudaStreamBeginCapture / cudaStreamEndCapture bracket
     * without breaking the capture.  All device work is still enqueued; only
     * the CPU-side bookkeeping that would require a D2H sync is skipped.
     *
     * Throws std::runtime_error if capture=true and any stage in the DAG
     * returns false from Stage::isGraphCompatible().  This catches incompatible
     * stages (e.g. inverse-mode RZE) at setup time rather than producing a
     * silently broken graph.
     *
     * The Compressor sets this automatically around its capture call; do not
     * set it manually unless you are managing graph capture yourself.
     */
    void setCaptureMode(bool capture);
    bool isCaptureMode() const { return capture_mode_; }

    // ========== Profiling ==========

    /**
     * Enable or disable per-stage CUDA event profiling.
     *
     * When enabled, a pair of CUDA events is recorded around each stage's
     * execute() call during DAG::execute(). Call collectTimings() after
     * execute() (and after synchronizing all streams) to retrieve results.
     *
     * Calling enableProfiling(true) creates start_event for every node that
     * has already been added to the DAG; nodes added afterward gain start_event
     * automatically if profiling is still enabled.
     *
     * Zero overhead when disabled: no events are created or recorded.
     */
    void enableProfiling(bool enable);

    bool isProfilingEnabled() const { return profiling_enabled_; }

    /**
     * Collect per-stage timing results.
     *
     * Synchronizes all DAG-owned CUDA streams so that CUDA event queries are
     * valid, then reads elapsed time for each node's [start → completion] pair.
     * Buffer byte counts are taken from the DAG's buffer size table.
     *
     * Returns an empty vector if profiling was not enabled.
     */
    std::vector<StageTimingResult> collectTimings();
    
private:
    MemoryPool* mem_pool_;
    MemoryStrategy strategy_;
    
    // DAG structure
    std::vector<DAGNode*> nodes_;  // Owned pointers
    std::unordered_map<int, BufferInfo> buffers_;
    
    int next_buffer_id_;
    bool is_finalized_;
    
    // Streams for parallel execution
    std::vector<cudaStream_t> streams_;
    bool owns_streams_;  // Whether we created the streams
    
    // Execution plan (computed during finalize)
    std::vector<std::vector<DAGNode*>> levels_;  // Nodes grouped by level
    int max_level_;
    
    // Memory tracking
    size_t current_memory_usage_;
    size_t peak_memory_usage_;

    // Profiling
    bool profiling_enabled_;

    // Runtime buffer overwrite detection (always on in debug builds)
    bool bounds_check_enabled_;

    // CUDA Graph capture mode: suppresses D2H-triggering CPU-side bookkeeping
    // during execute() so the call is safe inside a stream-capture bracket.
    bool capture_mode_;

    // Buffer coloring (PREALLOCATE mode only).
    // Non-overlapping buffers are assigned the same color and share one pool
    // region, reducing peak pool size.  color_region_ptrs_ owns the allocations;
    // BufferInfo::d_ptr points into one of these regions (aliased, not owned).
    bool coloring_disabled_;
    bool coloring_applied_;
    std::unordered_map<int, int> buffer_color_;       // buffer_id  -> color index
    std::vector<size_t>          color_region_sizes_; // color index -> region byte size
    std::vector<void*>           color_region_ptrs_;  // color index -> allocated ptr (owned)

    // ========== Internal Helpers ==========

    // Assign execution levels based on dependencies
    void assignLevels();

    // Assign streams to nodes for parallel execution
    void assignStreams();

    // Allocate buffer
    void allocateBuffer(int buffer_id, cudaStream_t stream);

    // Free buffer based on reference counting
    void freeBuffer(int buffer_id, cudaStream_t stream);

    // Strategy-specific allocation planning
    void planPreallocation();

    // Compute buffer live ranges, build interference graph, and assign colors.
    // Called from finalize() when strategy == PREALLOCATE and coloring is not disabled.
    void colorBuffers();
};

} // namespace fz
