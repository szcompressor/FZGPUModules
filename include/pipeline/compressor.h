#pragma once

#include "pipeline/dag.h"
#include "pipeline/perf.h"
#include "stage/stage.h"
#include "stage/stage_factory.h"
#include "mem/mempool.h"
#include "fzm_format.h"

#include <memory>
#include <stdexcept>
#include <unordered_map>

namespace fz {

/**
 * High-level Pipeline API for compression workflows
 * 
 * Builder-style interface for constructing complex compression pipelines.
 * Handles buffer sizing, memory management, and execution automatically.
 * 
 * Example:
 *   // Create pipeline (memory pool handled internally)
 *   Pipeline pipeline(input_size, MemoryStrategy::MINIMAL);
 *   
 *   auto* lorenzo = pipeline.addStage<LorenzoPredictor>(lorenzo_config);
 *   auto* rle = pipeline.addStage<RLEEncoder>();
 *   auto* bitpack = pipeline.addStage<BitpackEncoder>();
 *   
 *   pipeline.connect(rle, lorenzo);
 *   pipeline.connect(bitpack, rle);
 *   pipeline.finalize();
 *   
 *   void* output;
 *   size_t output_size;
 *   pipeline.compress(d_input, input_size, &output, &output_size, stream);
 *
 * @note Thread Safety: Pipeline is NOT thread-safe. A single Pipeline instance
 * must not be accessed concurrently from multiple host threads. Each thread
 * should use its own Pipeline instance. The underlying CUDA operations are
 * serialized per-stream; multi-stream parallelism happens within a single
 * host thread via the DAG's level-based execution.
 */
class Pipeline {
public:
    /**
     * Construct pipeline with optional configuration
     * 
     * @param input_data_size Expected input data size for pool sizing (0 = use default)
     * @param strategy Memory allocation strategy (default: MINIMAL)
     * @param pool_multiplier Pool size multiplier (pool = input * multiplier)
     */
    explicit Pipeline(
        size_t input_data_size = 0,
        MemoryStrategy strategy = MemoryStrategy::MINIMAL,
        float pool_multiplier = 3.0f
    );
    
    ~Pipeline();
    
    // ========== Configuration ==========
    
    /**
     * Set memory allocation strategy
     * Must be called before finalize()
     */
    void setMemoryStrategy(MemoryStrategy strategy);
    
    /**
     * Configure number of parallel streams for execution
     * Must be called before finalize()
     */
    void setNumStreams(int num_streams);
    
    // ========== Builder API ==========
    
    /**
     * Add a stage to the pipeline
     * 
     * @param args Arguments forwarded to Stage constructor
     * @return Pointer to created stage (owned by Pipeline)
     * 
     * Example:
     *   auto* lorenzo = pipeline.addStage<LorenzoPredictor>(config);
     */
    template<typename StageT, typename... Args>
    StageT* addStage(Args&&... args);
    
    /**
     * Add a pre-created stage to the pipeline (takes ownership)
     * Used internally during pipeline reconstruction from file headers.
     *
     * @param stage Raw stage pointer — Pipeline takes ownership
     * @return The same pointer, for chaining
     */
    Stage* addRawStage(Stage* stage);
    
    /**
     * Connect two stages with automatic buffer sizing
     * 
     * Creates a dependency: dependent waits for producer to complete
     * Buffer size will be determined based on:
     * - PREALLOCATE: producer->estimateOutputSizes()
     * - MINIMAL/PIPELINE: producer->getActualOutputSizes() or soft-run
     * 
     * @param dependent Stage that consumes output
     * @param producer Stage that produces output
     * @param output_name Name of producer's output to connect (default: "output")
     *                    For multi-output stages like Lorenzo: "codes", "outliers", etc.
     * @return Buffer ID for advanced use cases
     * 
     * Example:
     *   pipeline.connect(rle, lorenzo, "codes");       // RLE consumes codes
     *   pipeline.connect(formatter, lorenzo, "outliers"); // Formatter consumes outliers
     */
    int connect(Stage* dependent, Stage* producer, const std::string& output_name = "output");
    
    /**
     * Connect stage to multiple producers (multiple inputs)
     * 
     * Example: Bitpack stage consuming 3 RLE outputs
     *   pipeline.connect(bitpack, {rle1, rle2, rle3});
     */
    int connect(Stage* dependent, const std::vector<Stage*>& producers);
    
    /**
     * Mark a stage output as a sink (include in final buffer)
     * 
     * Use this to include outputs in the final concatenated buffer without
     * needing to create PassThrough stages for outputs that don't need processing.
     * 
     * @param stage Stage whose output should be included
     * @param output_name Name of the output to include (default: "output")
     * 
     * Example:
     *   pipeline.addSink(lorenzo, "outlier_errors");  // Include outliers directly
     *   pipeline.addSink(lorenzo, "outlier_indices"); // Include indices directly
     */
    void addSink(Stage* stage, const std::string& output_name = "output");
    
    /**
     * Finalize the pipeline for execution
     * 
     * - Validates topology (no cycles, all stages connected)
     * - Assigns execution levels
     * - For PREALLOCATE mode: estimates and allocates all buffers
     * 
     * Must be called before compress/decompress
     */
    void finalize();
    
    // ========== Execution ==========
    
    /**
     * Execute compression pipeline
     * 
     * @param d_input Device pointer to input data
     * @param input_size Size of input data in bytes
     * @param d_output [out] Device pointer to output (may be allocated)
     * @param output_size [out] Actual output size in bytes
     * @param stream CUDA stream for execution (default: 0)
     * 
     * Buffer sizing behavior:
     * - PREALLOCATE: All buffers pre-allocated, d_output set to final buffer
     * - MINIMAL/PIPELINE: Buffers allocated dynamically, d_output may change
     * 
     * Serialization architecture:
     * - Pipeline builds FZMHeaderCore + stage/buffer arrays on CPU using Stage::serializeHeader()
     * - Each stage serializes its own config (Lorenzo, Huffman, etc.)
     * - GPU buffer contains ONLY data (no headers intermixed)
     * - Headers stored separately in Pipeline::header_ for file I/O
     * - This follows cuSZ pattern: header on CPU, data on GPU
     */
    void compress(
        const void* d_input,
        size_t input_size,
        void** d_output,
        size_t* output_size,
        cudaStream_t stream = 0
    );
    
    /**
     * Execute in-memory decompression (inverse of compress())
     *
     * Reconstructs the original data by running the pipeline stages in reverse.
     * Must be called on the same Pipeline instance that ran compress(), without
     * an intervening reset().
     *
     * @param d_input  Device pointer to compressed data (may be nullptr).
     *                 - nullptr / omitted: reads compressed buffers directly
     *                   from the forward DAG's live memory (simplest path —
     *                   safe immediately after compress()).
     *                 - non-null, single output: treated as the one compressed
     *                   buffer (useful when the caller holds its own copy).
     *                 - non-null, multi-output: must point to a concat buffer
     *                   in the format produced by compress() with multiple sinks
     *                   ([num_bufs:u32][size:u64][data]...).
     * @param input_size  Size of d_input in bytes (ignored when d_input is nullptr).
     * @param d_output    [out] Newly cudaMalloc'd device pointer; caller must
     *                    cudaFree() the result.
     * @param output_size [out] Bytes in *d_output (== original uncompressed size).
     * @param stream      CUDA stream to use (default: 0).
     *
     * Typical in-memory round-trip (benchmarking):
     * @code
     *   pipeline.compress(d_in, n, &d_comp, &comp_sz, stream);
     *   pipeline.decompress(nullptr, 0, &d_out, &out_sz, stream);
     *   // or equivalently:
     *   pipeline.decompress(d_comp, comp_sz, &d_out, &out_sz, stream);
     * @endcode
     */
    void decompress(
        const void* d_input,
        size_t input_size,
        void** d_output,
        size_t* output_size,
        cudaStream_t stream = 0
    );
    
    /**
     * Reset pipeline for re-execution
     * Frees non-persistent buffers and resets state
     */
    void reset(cudaStream_t stream = 0);
    
    // ========== Verification & Stats ==========
    
    // ========== Profiling ==========

    /**
     * Enable or disable per-stage CUDA event profiling.
     *
     * When enabled:
     *   - A CUDA start/completion event pair is recorded around every stage
     *     execute() call inside the DAG.
     *   - compress() and decompress() measure host-side wall time with
     *     std::chrono::steady_clock.
     *   - After each call, results are available via getLastPerfResult().
     *
     * Has zero overhead when disabled (no events created or recorded).
     * Can be toggled between calls without re-finalizing the pipeline.
     */
    void enableProfiling(bool enable);

    bool isProfilingEnabled() const { return profiling_enabled_; }

    /**
     * Return the performance snapshot captured during the most recent
     * compress() or decompress() call.
     *
     * The result is only valid after at least one compress()/decompress()
     * call with profiling enabled.  Use PipelinePerfResult::print() for a
     * human-readable summary.
     */
    const PipelinePerfResult& getLastPerfResult() const { return last_perf_result_; }

    /**
     * Get the underlying DAG (for advanced use cases)
     */
    CompressionDAG* getDAG() { return dag_.get(); }
    
    /**
     * Get memory usage statistics
     */
    size_t getPeakMemoryUsage() const;
    size_t getCurrentMemoryUsage() const;
    
    /**
     * Print pipeline topology and configuration
     */
    void printPipeline() const;
    
    // ========== File Serialization ==========
    
    /**
     * Output buffer information for direct access
     */
    struct OutputBuffer {
        void* d_ptr;           // GPU pointer to buffer
        size_t actual_size;    // Bytes after compression
        size_t allocated_size; // Buffer capacity
        std::string name;      // e.g., "codes", "outlier_errors"
        int buffer_id;         // DAG buffer ID
    };
    
    /**
     * Get individual output buffers (no concatenation)
     * Call after compress() to access compressed data buffers
     * 
     * @return Vector of output buffer info
     */
    std::vector<OutputBuffer> getOutputBuffers() const;
    
    /**
     * Build FZM header from current pipeline state
     * Must be called after compress() to have accurate sizes
     * 
     * @return Complete FZM header core, with stage and buffer arrays
     */
    struct FZMFileHeader {
        FZMHeaderCore core;
        std::vector<FZMStageInfo> stages;
        std::vector<FZMBufferEntry> buffers;
    };
    FZMFileHeader buildHeader() const;
    
    /**
     * Write compressed data to file
     * 
     * Format: [FZMHeaderCore:48B][stages...][buffers...][data...]
     * 
     * @param filename Output file path
     * @param stream CUDA stream for GPU->CPU transfer (default: 0)
     * 
     * Note: Must call compress() first. This method synchronizes the stream.
     */
    void writeToFile(const std::string& filename, cudaStream_t stream = 0);
    
    /**
     * Read FZM header from file
     * 
     * @param filename Input file path
     * @return Parsed FZM header with stage and buffer arrays
     */
    static FZMFileHeader readHeader(const std::string& filename);
    
    /**
     * Load compressed data from file to GPU
     * 
     * @param filename Input file path
     * @param header FZM header (from readHeader)
     * @param stream CUDA stream for CPU->GPU transfer
     * @return Device pointer to compressed data
     */
    static void* loadCompressedData(
        const std::string& filename,
        const FZMFileHeader& header,
        cudaStream_t stream = 0,
        MemoryPool* pool = nullptr  ///< If non-null, GPU buffer is allocated from this pool
    );
    
    /**
     * Decompress data from file
     *
     * One-shot decompression: reads file, reconstructs pipeline, and decompresses.
     *
     * @param filename    Input FZM file path
     * @param d_output    [out] Device pointer to decompressed data
     * @param output_size [out] Size of decompressed data
     * @param stream      CUDA stream for operations (default: 0)
     * @param perf_out    Optional: if non-null, populated with GPU compute timing
     *                    (excludes file I/O and H→D copy; dag_elapsed_ms covers
     *                    only runInversePipeline + the final D→D output copy).
     */
    static void decompressFromFile(
        const std::string& filename,
        void** d_output,
        size_t* output_size,
        cudaStream_t stream = 0,
        PipelinePerfResult* perf_out = nullptr
    );

private:
    // ========== Internal Implementation ==========
    
    /**
     * Validate pipeline topology
     * - Check for cycles
     * - Ensure all stages are reachable from input
     * - Verify buffer counts match stage expectations
     */
    void validate();
    
    // ===== Finalize Helpers (split from monolithic finalize()) =====
    
    /**
     * Identify and validate topology (sources and sinks)
     * @return pair of (sources, sinks)
     */
    std::pair<std::vector<Stage*>, std::vector<Stage*>> identifyTopology();
    
    /**
     * Set up input buffer for source stage
     */
    void setupInputBuffers(const std::vector<Stage*>& sources);
    
    /**
     * Auto-detect and convert all unconnected outputs to pipeline outputs
     * This is the unified mechanism for determining what goes in final compressed output
     * @return number of pipeline outputs created
     */
    int autoDetectUnconnectedOutputs();
    
    /**
     * Detect if concatenation is needed (multiple pipeline outputs)
     */
    void detectMultiOutputScenario(int pipeline_outputs);
    
    /**
     * Configure streams in DAG if explicitly set by user
     */
    void configureStreamsIfNeeded();
    
    /**
     * Propagate buffer sizes through DAG based on input hint
     */
    void propagateBufferSizes();
    
    /**
     * Rebuild DAG connections in reverse for decompression
     * Uses stored connection info to reverse dependency edges
     */
    void rebuildInverseConnections();

    // ===== Unified Inverse Execution =====

    /**
     * Describes a single stage's position in the forward DAG topology.
     * Used by runInversePipeline to route buffers correctly without name heuristics.
     */
    struct InverseStageSpec {
        Stage* stage;                    ///< Stage to execute (must be in inverse mode)
        std::vector<int> fwd_input_ids;  ///< Forward-pass input buffer IDs  (= inverse outputs)
        std::vector<int> fwd_output_ids; ///< Forward-pass output buffer IDs (= inverse inputs)
    };

    /**
     * Unified inverse execution engine shared by decompress() and decompressFromFile().
     *
     * Processes specs in REVERSE forward order. For each stage:
     *   - Gathers inverse inputs from live_bufs keyed by fwd_output_ids
     *   - Executes the stage in inverse mode
     *   - Stores the result in live_bufs keyed by fwd_input_ids[0]
     *   - Frees pool-owned intermediate buffers as they are consumed
     *
     * @param specs             Forward-ordered stage specifications
     * @param live_bufs         Mutable map of buffer_id -> {d_ptr, actual_size}.
     *                          Must be pre-seeded with all leaf (compressed) buffers.
     * @param uncompressed_size Upper bound for inverse output allocation
     * @param pool              Memory pool for scratch allocations
     * @param stream            Fallback CUDA stream (used when num_streams <= 1)
     * @param num_streams       Number of CUDA streams for intra-level parallelism.
     *                          0 = auto-detect from topology (recommended).
     * @return {d_ptr, size} of final result (pool-allocated — caller must copy and free)
     *
     * Parallelism model (mirrors the forward DAG):
     *   - Specs are grouped by their forward execution level (computed from the
     *     dependency graph encoded in fwd_input_ids / fwd_output_ids).
     *   - Groups are processed in reverse level order (last compression level first).
     *   - Within a group, specs are launched concurrently across dedicated CUDA streams
     *     and synchronized with cudaStreamWaitEvent between groups.
     *   - Fan-in inverse (forward merge ↔ inverse split): allocates one output buffer
     *     per fwd_input_id so every forward input is reconstructed in parallel.
     */
    static std::pair<void*, size_t> runInversePipeline(
        const std::vector<InverseStageSpec>& specs,
        std::unordered_map<int, std::pair<void*, size_t>>& live_bufs,
        size_t uncompressed_size,
        MemoryPool& pool,
        cudaStream_t stream,
        int num_streams = 0,
        std::vector<StageTimingResult>* timing_out = nullptr
    );

    // =====
    
    /**
     * Find source stages (no dependencies)
     */
    std::vector<Stage*> getSourceStages() const;
    
    /**
     * Find sink stages (no dependents)
     */
    std::vector<Stage*> getSinkStages() const;
    
    // ===== Output Concatenation Helpers =====
    
    /**
     * Information about a single output buffer for concatenation
     */
    struct OutputBufferInfo {
        int buffer_id;
        void* d_ptr;
        size_t actual_size;
        std::string stage_name;
        std::string output_name;
    };
    
    /**
     * Collect all output buffer information using name-based size lookup
     * @return Vector of output buffer info
     */
    std::vector<OutputBufferInfo> collectOutputBuffers() const;
    
    /**
     * Calculate total size needed for concatenation
     * Format: [num_buffers:4B][size1:8B][data1][size2:8B][data2]...
     */
    size_t calculateConcatSize(const std::vector<OutputBufferInfo>& outputs) const;
    
    /**
     * Write concatenated buffer to device memory
     * @param outputs Output buffer information
     * @param d_concat_bytes Device buffer to write to
     * @param stream CUDA stream for operations
     * @return Total bytes written
     */
    size_t writeConcatBuffer(
        const std::vector<OutputBufferInfo>& outputs,
        uint8_t* d_concat_bytes,
        cudaStream_t stream
    ) const;
    
    /**
     * Concatenate multiple sink outputs into a single buffer
     * Format: [num_buffers:4B][size1:8B][data1][size2:8B][data2]...
     * 
     * @param d_output Output pointer to concatenated buffer
     * @param output_size Total size of concatenated buffer
     * @param stream CUDA stream for memory operations
     */
    void concatOutputs(void** d_output, size_t* output_size, cudaStream_t stream);
    
    // ========== Member Variables ==========
    
    std::unique_ptr<MemoryPool> mem_pool_;       // Owned memory pool
    std::unique_ptr<CompressionDAG> dag_;        // Internal DAG
    MemoryStrategy strategy_;
    
    // Stage management
    std::vector<std::unique_ptr<Stage>> stages_; // Owned stages
    std::unordered_map<Stage*, DAGNode*> stage_to_node_; // Mapping
    
    // Connection tracking (for DAG reversal)
    struct ConnectionInfo {
        Stage* dependent;
        Stage* producer;
        std::string output_name;
        int output_index;
    };
    std::vector<ConnectionInfo> connections_;
    
    // Configuration
    int num_streams_;
    bool is_finalized_;
    bool soft_run_enabled_;   ///< Future: enable soft-run buffer sizing pass

    // Profiling
    bool profiling_enabled_;
    PipelinePerfResult last_perf_result_;
    
    // Input/output tracking
    DAGNode* input_node_;                        // Source stage
    std::vector<DAGNode*> output_nodes_;         // Sink stages (may be multiple)
    int input_buffer_id_;
    std::vector<int> output_buffer_ids_;         // One per sink stage
    void* d_concat_buffer_;                      // Concatenated output buffer (if multi-sink)
    size_t concat_buffer_capacity_;              // Allocated size of concat buffer
    bool needs_concat_;                          // True if multiple sinks detected
    
    // Current input size (set during compress)
    size_t input_size_;
    
    // Input size hint from constructor (for initial buffer estimation)
    size_t input_size_hint_;
    
    // ===== File Serialization State =====
    
    /**
     * Metadata for each output buffer (populated during compress)
     */
    struct BufferMetadata {
        int buffer_id;
        size_t actual_size;      // After compression (from getActualOutputSizes)
        size_t allocated_size;   // Buffer capacity
        std::string name;        // Output name
        DAGNode* producer;       // Producing stage node
        int output_index;        // Which output of producer
    };
    std::vector<BufferMetadata> buffer_metadata_;
};

// ========== Template Implementation ==========

template<typename StageT, typename... Args>
StageT* Pipeline::addStage(Args&&... args) {
    if (is_finalized_) {
        throw std::runtime_error("Cannot add stages after finalization");
    }
    
    // Create stage
    auto stage_ptr = std::make_unique<StageT>(std::forward<Args>(args)...);
    StageT* stage = stage_ptr.get();
    
    // Add to internal DAG
    DAGNode* node = dag_->addStage(stage, stage->getName());
    
    // Pre-allocate ALL output buffers as unconnected
    // This ensures the stage always has all expected buffers
    size_t num_outputs = stage->getNumOutputs();
    auto output_names = stage->getOutputNames();
    
    for (size_t i = 0; i < num_outputs; i++) {
        std::string output_name = i < output_names.size() ? output_names[i] : std::to_string(i);
        std::string tag = stage->getName() + "." + output_name + "_unconnected";
        dag_->addUnconnectedOutput(node, 1, i, tag);  // size=1 placeholder
    }
    
    // Track mapping
    stage_to_node_[stage] = node;
    stages_.push_back(std::move(stage_ptr));
    
    return stage;
}

} // namespace fz
