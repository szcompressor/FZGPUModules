#pragma once

#include "pipeline/dag.h"
#include "pipeline/perf.h"
#include "stage/stage.h"
#include "stage/stage_factory.h"
#include "mem/mempool.h"
#include "fzm_format.h"
#include "predictors/lorenzo/lorenzo.h"

#include <array>
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

    /**
     * Set spatial dimensions of the dataset (default: 1-D, {n, 1, 1}).
     * Controls which Lorenzo predictor variant is automatically selected by
     * addLorenzo() and is forwarded to LorenzoStage::Config::dims.
     *
     * @param x  Size of the fastest (x) dimension (must be > 0)
     * @param y  Size of the y dimension (1 = 1-D or 2-D collapsed)
     * @param z  Size of the z dimension (1 = 1-D or 2-D)
     *
     * Examples:
     *   pipeline.setDims(1000000);        // 1-D, 1 M elements
     *   pipeline.setDims(1024, 512);      // 2-D, 512 x 1024
     *   pipeline.setDims(256, 256, 128);  // 3-D
     */
    void setDims(size_t x, size_t y = 1, size_t z = 1) {
        dims_ = {x, y, z};
    }
    void setDims(std::array<size_t, 3> dims) { dims_ = dims; }

    std::array<size_t, 3> getDims() const { return dims_; }
    
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

    // ========== Convenience Stage Builders ==========

    /**
     * Add a Lorenzo predictor stage with automatic dimensionality selection.
     *
     * The correct 1-D / 2-D / 3-D variant is chosen from the pipeline's
     * current dims (set via setDims()).  If setDims() was never called the
     * stage defaults to 1-D Lorenzo.
     *
     * @param error_bound      Absolute pointwise error tolerance
     * @param quant_radius     Quantization radius (default 32768 for uint16_t)
     * @param outlier_capacity Fraction of data to reserve for outliers (default 0.2)
     * @return Pointer to the created stage (owned by Pipeline)
     *
     * Example:
     *   pipeline.setDims(nx, ny);           // 2-D dataset
     *   auto* lrz = pipeline.addLorenzo(1e-4f);
     *   pipeline.connect(rle, lrz, "codes");
     */
    template<typename TInput = float, typename TCode = uint16_t>
    LorenzoStage<TInput, TCode>* addLorenzo(
        float error_bound,
        int   quant_radius     = 32768,
        float outlier_capacity = 0.2f
    ) {
        typename LorenzoStage<TInput, TCode>::Config cfg;
        cfg.error_bound      = error_bound;
        cfg.quant_radius     = quant_radius;
        cfg.outlier_capacity = outlier_capacity;
        cfg.dims             = dims_;
        return addStage<LorenzoStage<TInput, TCode>>(cfg);
    }
    
    // ========== Execution ==========

    /**
     * Input specification for multi-source compression.
     *
     * @param source  Pointer to the source stage that will receive this data.
     *                Must be a stage already added to the pipeline that has no
     *                upstream producers (i.e. a DAG root).
     * @param d_data  Device pointer to the raw input for this source.
     * @param size    Size of d_data in bytes.
     */
    struct InputSpec {
        Stage*      source;
        const void* d_data;
        size_t      size;
    };

    /**
     * Execute compression pipeline — single-source convenience overload.
     *
     * Equivalent to:
     *   compress({InputSpec{sources[0], d_input, input_size}}, d_output, output_size, stream)
     *
     * Throws if the pipeline has more than one source stage.
     *
     * OWNERSHIP: *d_output points into the Pipeline's internal memory pool.
     * The caller must NOT call cudaFree() on it.  The pointer is valid until
     * the next compress() / reset() call on this Pipeline, or until the
     * Pipeline is destroyed.  This is intentionally zero-copy (contrast with
     * decompress(), which always returns a freshly cudaMalloc'd buffer that
     * the caller IS responsible for freeing).
     *
     * @param d_input     Device pointer to input data
     * @param input_size  Size of input data in bytes
     * @param d_output    [out] Pool-owned pointer to compressed output — do NOT cudaFree
     * @param output_size [out] Actual compressed size in bytes
     * @param stream      CUDA stream for execution (default: 0)
     */
    void compress(
        const void* d_input,
        size_t input_size,
        void** d_output,
        size_t* output_size,
        cudaStream_t stream = 0
    );

    /**
     * Execute compression pipeline — multi-source overload.
     *
     * Provide one InputSpec per source stage in the pipeline.  Order within
     * the vector does not matter; each spec is matched to its source stage by
     * pointer identity.
     *
     * OWNERSHIP: same as the single-source overload — *d_output points into
     * the Pipeline's internal pool.  Do NOT call cudaFree() on it.
     *
     * @param inputs      One InputSpec per pipeline source stage.
     * @param d_output    [out] Pool-owned pointer to compressed output — do NOT cudaFree
     * @param output_size [out] Actual compressed size in bytes
     * @param stream      CUDA stream for execution (default: 0)
     */
    void compress(
        const std::vector<InputSpec>& inputs,
        void** d_output,
        size_t* output_size,
        cudaStream_t stream = 0
    );

    /**
     * Set a per-source input size hint for buffer pre-sizing.
     *
     * For multi-source pipelines, call this once per source stage after
     * adding stages but before finalize().  The hint seeds
     * propagateBufferSizes() so that downstream buffer estimates are
     * accurate.  Falls back to the constructor's input_data_size for any
     * source without an explicit hint.
     *
     * @param source  Source stage pointer (must be a DAG root).
     * @param size    Expected input size in bytes.
     */
    void setInputSizeHint(Stage* source, size_t size) {
        per_source_hints_[source] = size;
    }
    
    /**
     * Execute in-memory decompression (inverse of compress()).
     *
     * Reconstructs the original data by running the pipeline stages in reverse.
     * Must be called on the same Pipeline instance that ran compress(), without
     * an intervening reset().
     *
     * For single-source pipelines:
     *   *d_output is a newly cudaMalloc'd buffer containing the raw decompressed
     *   data.  *output_size is the exact decompressed size in bytes.
     *
     * For multi-source pipelines:
     *   *d_output is a newly cudaMalloc'd buffer containing all decompressed
     *   sources concatenated in the same format as the multi-output compress()
     *   result: [num_bufs:u32][size1:u64][data1][size2:u64][data2]...
     *   Sources are ordered to match the order of input_nodes_ (the forward
     *   source discovery order from finalize()).
     *   Use decompressMulti() to receive individual per-source buffers instead.
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
     * @param output_size [out] Bytes in *d_output.
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
     * In-memory decompression — multi-source variant.
     *
     * Identical to decompress() but returns one {device_ptr, size} pair per
     * source stage instead of concatenating.  The order matches the forward
     * source discovery order (same as the InputSpec vector passed to compress()).
     * Each returned device pointer is a newly cudaMalloc'd allocation; the
     * caller is responsible for calling cudaFree() on each one.
     *
     * For single-source pipelines this is equivalent to the scalar decompress()
     * overload and returns a vector of size 1.
     *
     * @param d_input     Same semantics as decompress().
     * @param input_size  Same semantics as decompress().
     * @param stream      CUDA stream to use (default: 0).
     * @return Vector of {device pointer, byte count}, one entry per source.
     */
    std::vector<std::pair<void*, size_t>> decompressMulti(
        const void* d_input = nullptr,
        size_t      input_size = 0,
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
     * Enable runtime buffer-overwrite detection.
     *
     * After each stage executes, the DAG checks that the stage's reported
     * actual output size does not exceed the allocated buffer capacity.  A
     * violation throws std::runtime_error immediately with a descriptive
     * message naming the stage and output slot.
     *
     * In debug builds (-DNDEBUG absent) the check runs unconditionally
     * regardless of this flag.  This opt-in lets you activate it in release
     * builds for production debug sessions without a full recompile.
     *
     * Zero overhead when disabled in release builds.
     */
    void enableBoundsCheck(bool enable) { dag_->enableBoundsCheck(enable); }
    bool isBoundsCheckEnabled() const   { return dag_->isBoundsCheckEnabled(); }
    
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
     * @param filename          Input FZM file path
     * @param d_output          [out] Device pointer to decompressed data
     * @param output_size       [out] Size of decompressed data
     * @param stream            CUDA stream for operations (default: 0)
     * @param perf_out          Optional: if non-null, populated with GPU compute timing
     *                          (excludes file I/O and H→D copy).
     * @param pool_override_bytes  If non-zero, skip the automatic pool-size calculation
     *                             and use this value directly. Use only when the pipeline
     *                             topology has unusual characteristics (e.g. extreme fanout)
     *                             that the header-derived estimate cannot account for.
     */
    static void decompressFromFile(
        const std::string& filename,
        void** d_output,
        size_t* output_size,
        cudaStream_t stream = 0,
        PipelinePerfResult* perf_out = nullptr,
        size_t pool_override_bytes = 0
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
     * Find source stages (no dependencies)
     */
    std::vector<Stage*> getSourceStages() const;
    
    /**
     * Find sink stages (no dependents)
     */
    std::vector<Stage*> getSinkStages() const;

    // ===== Inverse DAG Helpers =====

    /**
     * Compact description of one forward stage used by buildInverseDAG().
     * Callers populate this from either the live DAGNode (decompress path)
     * or the serialized FZMStageInfo (decompressFromFile path).
     */
    struct FwdStageDesc {
        Stage*           stage;
        std::vector<int> output_buf_ids;  ///< forward output buffer IDs (positional)
        std::vector<int> input_buf_ids;   ///< forward input buffer IDs (positional)
    };

    /// maps fwd_buf_id → {device pointer, size in bytes} for each pipeline-output buffer
    using PipelineOutputMap = std::unordered_map<int, std::pair<void*, size_t>>;

    /**
     * Builds, wires, and finalizes an inverse CompressionDAG from a compact
     * forward-topology description.  Supports pipelines with one or more
     * source stages (as produced by Phase 1 multi-source compress()).
     *
     * Encapsulates Steps 1-5 of the inverse path (shared between decompress()
     * and decompressFromFile()):
     *   1. Add stages in reverse forward order, pre-allocate output slots.
     *   2. Wire: intermediate outputs → connectExistingOutput;
     *            pipeline-output buffers → setInputBuffer + setExternalPointer.
     *   3. For every forward source (fwd_stages entry with no input_buf_ids),
     *      mark its first inverse output as a result buffer and persistent.
     *   4. Finalize (assigns levels and streams).
     *   5. Propagate estimated buffer sizes; override each result buffer with
     *      its exact size from source_sizes.
     *
     * The caller is responsible for executing the returned DAG, running
     * postStreamSync() on all stages, extracting the results, and resetting.
     *
     * @param fwd_stages       Stages in FORWARD topological order (source first).
     * @param pipeline_outputs fwd_buf_id → {d_ptr, size} for every pipeline-output buffer.
     * @param pool             MemoryPool for inv_dag allocations.
     * @param strategy         Memory strategy for the inv_dag.
     * @param source_sizes     Maps each forward source Stage* to its exact
     *                         uncompressed size in bytes.
     * @param enable_profiling Enable CUDA event profiling inside the inv_dag.
     * @return {inv_dag (finalized, ready to execute),
     *          map of source Stage* → inv result buffer ID}
     */
    static std::pair<std::unique_ptr<CompressionDAG>,
                     std::unordered_map<Stage*, int>>
    buildInverseDAG(
        const std::vector<FwdStageDesc>&        fwd_stages,
        const PipelineOutputMap&                pipeline_outputs,
        MemoryPool*                             pool,
        MemoryStrategy                          strategy,
        const std::unordered_map<Stage*, size_t>& source_sizes,
        bool                                    enable_profiling
    );
    
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

    // Track whether compress() has been called at least once so that
    // writeToFile() can give a clear error if called before compressing, and so
    // that a subsequent compress() call can implicitly reset the DAG (RC4).
    bool is_compressed_;  ///< True after the first successful compress() call
    bool was_compressed_; ///< True between compress() and the next reset()

    // Profiling
    bool profiling_enabled_;
    PipelinePerfResult last_perf_result_;
    
    // Input/output tracking
    std::vector<DAGNode*> input_nodes_;          // Source stages (one per pipeline input)
    std::vector<DAGNode*> output_nodes_;         // Sink stages (may be multiple)
    std::vector<int> input_buffer_ids_;          // External input buffer per source
    std::vector<int> output_buffer_ids_;         // One per sink stage
    void* d_concat_buffer_;                      // Concatenated output buffer (if multi-sink)
    size_t concat_buffer_capacity_;              // Allocated size of concat buffer
    bool needs_concat_;                          // True if multiple sinks detected
    
    // Current input size (set during compress; sum over all sources)
    size_t input_size_;

    // Per-source actual input sizes recorded during the most recent compress().
    // Ordered to match input_nodes_ (source discovery order from finalize()).
    // Used by decompress()/decompressMulti() to size each inverse result buffer.
    std::vector<size_t> source_input_sizes_;

    // Input size hint from constructor (for initial buffer estimation).
    // Applied to all sources that lack a per_source_hints_ entry.
    size_t input_size_hint_;

    // Per-source size hints set via setInputSizeHint().  Overrides
    // input_size_hint_ for the matching source stage during propagateBufferSizes().
    std::unordered_map<Stage*, size_t> per_source_hints_;

    // Spatial dimensions of the dataset (x=fast, y, z).
    // Used by addLorenzo() to select 1-D/2-D/3-D automatically.
    // Default {0,1,1} means "1-D, infer x from input size".
    std::array<size_t, 3> dims_;
    
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
