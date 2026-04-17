#pragma once

#include "pipeline/concat_kernel.h"
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
     * - MINIMAL: producer->getActualOutputSizes() (sized at runtime)
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
     * Finalize the pipeline for execution
     *
     * - Validates topology (no cycles, all stages connected)
     * - Assigns execution levels
     * - For PREALLOCATE mode: estimates and allocates all buffers
     * - If setWarmupOnFinalize(true) was called and input_size_hint > 0,
     *   automatically runs warmup() to force JIT compilation of all kernels.
     *
     * Must be called before compress/decompress
     */
    void finalize();

    /**
     * Force JIT compilation of all pipeline kernels by running a dummy
     * compress() pass on a zero-filled device buffer.
     *
     * On first kernel launch CUDA lazily compiles PTX→SASS; this shows up
     * as a large one-time latency spike (e.g. 35× overhead for stages that
     * instantiate CUB DeviceScan).  warmup() pays that cost up front so that
     * the first timed compress() call sees the same latency as steady-state.
     *
     * Requires input_size_hint > 0 (passed to the Pipeline constructor) so
     * the dummy buffer can be sized correctly.  Profiling is suppressed
     * during the warmup pass and restored afterwards.  The pipeline is left
     * in a clean state identical to just after finalize() — as if the warmup
     * call never happened.
     *
     * Can also be triggered automatically by calling setWarmupOnFinalize(true)
     * before finalize().
     *
     * @param stream  CUDA stream to use for the warmup pass (default: 0).
     */
    void warmup(cudaStream_t stream = 0);

    /**
     * Control whether finalize() automatically calls warmup().
     *
     * When set to true (default: false), finalize() will call warmup(0) after
     * all buffers have been allocated.  No-op if input_size_hint is 0.
     *
     * Must be called before finalize().
     */
    void setWarmupOnFinalize(bool enable) { warmup_on_finalize_ = enable; }
    bool isWarmupOnFinalizeEnabled() const { return warmup_on_finalize_; }

    /**
     * Control whether decompress() allocates output from the internal pool.
     *
     * When false (default), *d_output is a freshly cudaMalloc'd buffer and
     * the caller is responsible for calling cudaFree() on it.
     *
     * When true, *d_output points into the Pipeline's memory pool.  The pointer
     * is valid until the next decompress() call or Pipeline destruction.  The
     * caller must NOT call cudaFree() on it.  This eliminates one cudaMalloc /
     * cudaFree round trip per decompress call, which is useful when decompress
     * is called in a tight loop or alongside other memory-intensive GPU work.
     */
    void setPoolManagedDecompOutput(bool enable) { pool_managed_decomp_ = enable; }
    bool isPoolManagedDecompOutput() const { return pool_managed_decomp_; }

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
     * @param d_output    [out] Device pointer to decompressed output.
     *                    Ownership depends on setPoolManagedDecompOutput():
     *                    - false (default): freshly cudaMalloc'd; caller must cudaFree().
     *                    - true: pool-managed; do NOT cudaFree(). Valid until the next
     *                      decompress() call or Pipeline destruction.
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
     * Return the pool release threshold currently configured, in bytes.
     *
     * After finalize() this reflects the topology-aware value set by
     * Pipeline::finalize() (topo base × safety margin).  Before finalize()
     * it reflects the initial estimate (input_size × pool_multiplier).
     */
    size_t getPoolThreshold() const;

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
     * Disable buffer coloring for PREALLOCATE mode (default: enabled).
     *
     * Coloring reduces peak pool size by reusing memory regions across buffers
     * whose live ranges do not overlap.  Disable for debugging when you need
     * each buffer to occupy a distinct, non-aliased memory region (e.g. when
     * using cuda-memcheck or the bounds-check system).
     *
     * Must be called before finalize().
     */
    void disableColoring(bool disable) { dag_->disableColoring(disable); }
    bool isColoringApplied() const     { return dag_->isColoringApplied(); }
    size_t getColorRegionCount() const { return dag_->getColorRegionCount(); }

    // ========== CUDA Graph Capture (compression-only) ==========

    /**
     * Enable CUDA Graph capture mode.
     *
     * When enabled, captureGraph() records the entire forward compression pass
     * (all kernel launches and D2D copies inside dag_->execute()) as a CUDA
     * Graph.  Subsequent compress() calls use cudaGraphLaunch() instead of
     * dag_->execute(), eliminating per-call CPU dispatch overhead.
     *
     * Requirements enforced at finalize() / captureGraph() time:
     *   - Strategy must be PREALLOCATE (buffers must be stable across launches).
     *   - A non-zero input_size_hint must be provided to the constructor.
     *   - All stages must return true from Stage::isGraphCompatible()
     *     (checked by DAG::setCaptureMode(), which is called inside captureGraph()).
     *   - Single-source pipelines only (multi-source not yet supported).
     *
     * Must be called before finalize().
     */
    void enableGraphMode(bool enable);
    bool isGraphModeEnabled() const { return graph_mode_enabled_; }

    /**
     * Record the forward compression pass as a CUDA Graph.
     *
     * Internally:
     *   1. Points the DAG's input buffer at a fixed device slot (d_graph_input_)
     *      so the captured graph has a stable, reusable input pointer.
     *   2. Calls DAG::setCaptureMode(true), which validates all stages are
     *      graph-compatible (throws if any are not).
     *   3. cudaStreamBeginCapture → dag_->execute() → cudaStreamEndCapture
     *   4. cudaGraphInstantiate — produces the replayable executable graph.
     *   5. Restores capture_mode_ = false.
     *
     * After this call, compress() copies the user's input into d_graph_input_
     * then calls cudaGraphLaunch() instead of dag_->execute().  All post-graph
     * work (postStreamSync, output collection, concat) runs as normal on the CPU.
     *
     * Can be called again to re-capture (destroys the previous graph first).
     * Must be called after finalize() and before the first compress().
     *
     * @param stream  CUDA stream for capture.  Must not already be in a capture
     *                bracket.  All recorded work will be on this stream.
     */
    void captureGraph(cudaStream_t stream = 0);
    bool isGraphCaptured() const { return graph_captured_; }

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
     * Propagate buffer sizes through DAG from source sizes.
     *
     * By default this uses constructor/per-source hints (finalize-time path).
     * When force_from_current_inputs=true, it uses the current source buffer
     * sizes already written into the DAG (compress-time path for zero-hint
     * pipelines).
     */
    void propagateBufferSizes(bool force_from_current_inputs = false);
    
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
    bool soft_run_enabled_;     ///< Future: enable soft-run buffer sizing pass
    bool warmup_on_finalize_;   ///< If true, finalize() auto-calls warmup()
    bool pool_managed_decomp_;  ///< If true, decompress() returns a pool-owned pointer

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

    // Pool-managed decompress output buffers (used when pool_managed_decomp_ == true).
    // Freed and reallocated on each decompressMulti() call; persistent in the pool.
    // One entry per source stage (vector of size 1 for single-source pipelines).
    std::vector<void*> d_decomp_outputs_;  ///< Pool-persistent decompressed output(s)

    // Pinned host buffer for concat header (num_buffers + per-buffer sizes).
    // Allocated once at finalize() time; avoids N+1 tiny H2D cudaMemcpyAsync
    // calls — instead we pack the header on the CPU then issue a single H2D copy.
    void*  h_concat_header_;           ///< cudaHostAlloc'd pinned buffer
    size_t h_concat_header_capacity_;  ///< allocated size in bytes

    // §2B gather kernel — persistent pinned host + device descriptor buffers.
    // Sized at finalize() for max_outputs descriptors; reused every compress call.
    // CPU packs {src, dst, bytes} into h_copy_descs_, one H2D copy delivers
    // all descriptors, then gather_kernel replaces N D2D cudaMemcpyAsync calls.
    void*  h_copy_descs_;              ///< pinned host CopyDesc array
    void*  d_copy_descs_;              ///< device  CopyDesc array
    size_t copy_descs_capacity_;       ///< allocated capacity in bytes
    
    // Current input size (set during compress; sum over all sources)
    size_t input_size_;

    // Per-source actual input sizes recorded during the most recent compress().
    // Ordered to match input_nodes_ (source discovery order from finalize()).
    // Used by decompress()/decompressMulti() to size each inverse result buffer.
    std::vector<size_t> source_input_sizes_;

    // Required input alignment in bytes, computed at finalize() as the LCM of
    // getRequiredInputAlignment() across all stages.  compress() pads the input
    // to this boundary transparently using d_pad_buf_.
    size_t input_alignment_bytes_;
    void*  d_pad_buf_;       ///< Lazily-allocated padding scratch buffer (MemoryPool)
    size_t d_pad_buf_size_;  ///< Current allocated size of d_pad_buf_

    // When compress() transparently pads the input, this holds the original
    // (unpadded) byte count.  decompress() uses it to trim the reported output
    // size back to what the caller originally provided.  0 when no padding.
    size_t original_input_size_;

    // Input size hint from constructor (for initial buffer estimation).
    // Applied to all sources that lack a per_source_hints_ entry.
    size_t input_size_hint_;

    // Pool size multiplier from constructor, stored so finalize() can apply it
    // as a headroom factor on top of the topology-derived base pool size.
    float pool_multiplier_;

    // Per-source size hints set via setInputSizeHint().  Overrides
    // input_size_hint_ for the matching source stage during propagateBufferSizes().
    std::unordered_map<Stage*, size_t> per_source_hints_;

    // Spatial dimensions of the dataset (x=fast, y, z).
    // Used by addLorenzo() to select 1-D/2-D/3-D automatically.
    // Default {0,1,1} means "1-D, infer x from input size".
    std::array<size_t, 3> dims_;
    
    // ===== Inverse DAG Cache =====

    /**
     * Cached inverse DAG for repeated decompress() calls.
     *
     * Built lazily on the first decompressMulti() call and reused on every
     * subsequent call.  On reuse only the external compressed-data pointers are
     * updated; the topology, CUDA events, and (in PREALLOCATE mode) all buffer
     * allocations are preserved, eliminating the per-call rebuild overhead.
     *
     * Invalidated when source sizes change between compress() calls (different
     * input sizes).  Since finalize() cannot be called twice, the DAG topology
     * itself never changes after the first build.
     */
    struct InvDAGCache {
        std::unique_ptr<CompressionDAG>    inv_dag;
        std::unordered_map<Stage*, int>    inv_result_map;       ///< source Stage* → result buf id
        std::unordered_map<int, int>       fwd_to_inv_ext_buf;   ///< fwd_buf_id → inv external buf id
        std::unordered_map<Stage*, size_t> source_sizes;         ///< source sizes at build time (for invalidation)
    };
    std::unique_ptr<InvDAGCache> inv_cache_;

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

    // ===== CUDA Graph State =====

    bool graph_mode_enabled_;  ///< Set by enableGraphMode() before finalize()
    bool graph_captured_;      ///< True after a successful captureGraph()

    /// Fixed device input buffer whose pointer is baked into the captured graph.
    /// compress() copies the user's input here before cudaGraphLaunch().
    /// Allocated from the pool at finalize() time when graph_mode_enabled_.
    void*  d_graph_input_;
    size_t d_graph_input_size_;

    cudaGraph_t     captured_graph_;  ///< Raw graph (kept for introspection / re-capture)
    cudaGraphExec_t graph_exec_;      ///< Instantiated executable graph
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
