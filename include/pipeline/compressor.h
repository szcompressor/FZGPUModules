#pragma once

#include "pipeline/dag.h"
#include "pipeline/perf.h"
#include "pipeline/config.h"
#include "stage/stage.h"
#include "stage/stage_factory.h"
#include "mem/mempool.h"
#include "fzm_format.h"

#include <array>
#include <memory>
#include <stdexcept>
#include <unordered_map>

namespace fz {

/**
 * High-level pipeline API for building and executing compression workflows.
 *
 * Stages are added with addStage<T>(), wired with connect(), then the
 * pipeline is finalized and ready for compress()/decompress().
 *
 * Ownership:
 *  - compress() output is pool-owned — do NOT cudaFree it. Valid until the
 *    next compress()/reset() or Pipeline destruction.
 *  - decompress() output is caller-owned by default (freshly cudaMalloc'd).
 *    Set setPoolManagedDecompOutput(true) to receive a pool-owned pointer instead.
 *
 * @note Not thread-safe. Use one Pipeline per host thread.
 */
class Pipeline {
public:
    /**
     * @param input_data_size  Expected input size in bytes for pool sizing (0 = default).
     * @param strategy         MINIMAL (on-demand alloc) or PREALLOCATE (upfront, required for graph mode).
     * @param pool_multiplier  Pool size = input_size × multiplier.
     */
    explicit Pipeline(
        size_t input_data_size = 0,
        MemoryStrategy strategy = MemoryStrategy::MINIMAL,
        float pool_multiplier = 3.0f
    );

    /**
     * Construct directly from a TOML config file.
     * Equivalent to the default constructor followed by loadConfig(path).
     * The pipeline is finalized on return.
     *
     * @param config_path  Path to the .toml config file.
     */
    explicit Pipeline(const std::string& config_path);

    ~Pipeline();

    // ── Configuration ─────────────────────────────────────────────────────────

    /** Must be called before finalize(). */
    void setMemoryStrategy(MemoryStrategy strategy);

    /** Number of parallel CUDA streams for level-based execution. Must be called before finalize(). */
    void setNumStreams(int num_streams);

    /**
     * Dataset spatial dimensions. Controls which Lorenzo variant is selected by
     * addLorenzo() and is forwarded to LorenzoStage at finalize().
     * Default: 1-D ({n, 1, 1}).
     */
    void setDims(size_t x, size_t y = 1, size_t z = 1) { dims_ = {x, y, z}; }
    void setDims(std::array<size_t, 3> dims)             { dims_ = dims; }
    std::array<size_t, 3> getDims() const                { return dims_; }

    // ── Builder API ───────────────────────────────────────────────────────────

    /**
     * Add a stage to the pipeline.
     * @return Raw pointer owned by the Pipeline.
     */
    template<typename StageT, typename... Args>
    StageT* addStage(Args&&... args);

    /**
     * Connect two stages (dependent consumes an output of producer).
     * @param dependent    The downstream stage that reads the output.
     * @param producer     The upstream stage that writes the output.
     * @param output_name  Named output port of producer (default: "output").
     * @return Buffer ID (rarely needed directly).
     */
    int connect(Stage* dependent, Stage* producer, const std::string& output_name = "output");

    /** Connect a stage to multiple producers (one input per producer). */
    int connect(Stage* dependent, const std::vector<Stage*>& producers);

    /**
     * Finalize the pipeline: validate topology, assign execution levels, and
     * (for PREALLOCATE) allocate all buffers. Must be called before compress/decompress.
     * If setWarmupOnFinalize(true) was set and input_size_hint > 0, runs warmup() automatically.
     */
    void finalize();

    /**
     * JIT-compile all pipeline kernels by running a dummy compress+decompress pass.
     * Eliminates the first-call latency spike from CUDA's lazy PTX→SASS compilation.
     * Requires a non-zero input_size_hint in the constructor.
     */
    void warmup(cudaStream_t stream = 0);

    /** When true, finalize() automatically calls warmup(). Must be set before finalize(). */
    void setWarmupOnFinalize(bool enable) { warmup_on_finalize_ = enable; }
    bool isWarmupOnFinalizeEnabled() const { return warmup_on_finalize_; }

    /**
     * When true (default), decompress() returns a pool-owned pointer (do NOT cudaFree).
     * Valid until the next decompress() call or Pipeline destruction.
     * When false, decompress() returns a freshly cudaMalloc'd pointer
     * that the caller must cudaFree().
     */
    void setPoolManagedDecompOutput(bool enable) { pool_managed_decomp_ = enable; }
    bool isPoolManagedDecompOutput() const { return pool_managed_decomp_; }

    /**
     * Return the worst-case compressed output size in bytes for the given input.
     *
     * Must be called after finalize(). Use this to pre-allocate a caller-owned
     * output buffer before passing it to the user-owned compress() overload.
     *
     * The returned value is a tight upper bound derived from each stage's
     * estimateOutputSizes() chain — it should rarely exceed ~110% of the actual
     * compressed size for typical data.
     *
     * @param input_bytes  Number of bytes you intend to compress.
     * @throws std::runtime_error if the pipeline is not yet finalized.
     */
    size_t getMaxCompressedSize(size_t input_bytes) const;

    // ── Execution ─────────────────────────────────────────────────────────────

    /**
     * Compress (pool-owned output). The pool retains the output buffer.
     *
     * @param d_input     Device pointer to raw input data.
     * @param input_size  Size of d_input in bytes.
     * @param d_output    Receives a pool-owned pointer to the compressed output.
     *                    Do NOT call cudaFree() — valid until the next compress(),
     *                    reset(), or Pipeline destruction.
     * @param output_size Receives the exact compressed size in bytes.
     * @param stream      CUDA stream for all GPU operations.
     */
    void compress(
        const void* d_input,
        size_t      input_size,
        void**      d_output,
        size_t*     output_size,
        cudaStream_t stream = 0
    );

    /**
     * Compress (user-owned output). The compressed data is written into the
     * caller-provided device buffer.
     *
     * The buffer just needs to be large enough for the actual compressed output
     * of this specific call — which depends on the data. If the actual output
     * exceeds `output_buf_capacity` a `std::runtime_error` is thrown with the
     * actual and capacity sizes so the caller can retry with a larger buffer.
     *
     * Use `getMaxCompressedSize(input_bytes)` for a guaranteed safe upper bound.
     * Alternatively, if you know empirically that your data compresses to at most
     * X bytes for your workload, you can pass X directly and accept the small
     * risk of a runtime error on unusually incompressible inputs.
     *
     * Incompatible with CUDA Graph mode (the output address cannot be baked into
     * a captured graph). Throws if enableGraphMode(true) was set.
     *
     * @param d_input              Device pointer to raw input data.
     * @param input_size           Size of d_input in bytes.
     * @param d_output_buf         Caller-allocated device buffer to write compressed
     *                             data into.
     * @param output_buf_capacity  Capacity of d_output_buf in bytes. Must fit the
     *                             actual compressed output for this call.
     * @param actual_output_size   Receives the exact compressed bytes written.
     * @param stream               CUDA stream for all GPU operations.
     */
    void compress(
        const void* d_input,
        size_t      input_size,
        void*       d_output_buf,
        size_t      output_buf_capacity,
        size_t*     actual_output_size,
        cudaStream_t stream = 0
    );

    /**
     * Decompress. Inverse of compress().
     *
     * @param d_input     nullptr to read from the forward DAG's live buffers
     *                    (simplest path, valid immediately after compress()).
     *                    Non-null for an external compressed buffer.
     * @param input_size  Byte size of d_input (ignored when d_input is nullptr).
     * @param d_output    Receives the decompressed device pointer.
     *                    Ownership depends on setPoolManagedDecompOutput():
     *                    false           → caller-owned, must cudaFree.
     *                    true (default)  → pool-owned, do NOT cudaFree.
     * @param output_size Receives the exact decompressed size in bytes.
     * @param stream      CUDA stream for all GPU operations.
     */
    void decompress(
        const void* d_input,
        size_t      input_size,
        void**      d_output,
        size_t*     output_size,
        cudaStream_t stream = 0
    );

    /**
     * Decompress into a caller-provided device buffer (user-owned output).
     *
     * The decompressed data is written directly into d_output_buf. No cudaMalloc
     * or pool allocation is performed — the caller owns the buffer entirely.
     *
     * The buffer just needs to be large enough for the actual decompressed output
     * of this call. If it is too small a `std::runtime_error` is thrown with the
     * actual size so the caller can retry. Typically the uncompressed size is
     * known from the file header (`FZMHeaderCore::uncompressed_size`) or from
     * the original compress() call.
     *
     * @param d_input              See decompress() above.
     * @param input_size           See decompress() above.
     * @param d_output_buf         Caller-allocated device buffer to receive
     *                             decompressed data.
     * @param output_buf_capacity  Capacity of d_output_buf in bytes.
     * @param actual_output_size   Receives the exact bytes written.
     * @param stream               CUDA stream for all GPU operations.
     */
    void decompress(
        const void* d_input,
        size_t      input_size,
        void*       d_output_buf,
        size_t      output_buf_capacity,
        size_t*     actual_output_size,
        cudaStream_t stream = 0
    );

    /** Free non-persistent buffers and reset execution state for re-use. */
    void reset(cudaStream_t stream = 0);

    // ── Profiling ─────────────────────────────────────────────────────────────

    /**
     * Enable per-stage CUDA event profiling. Zero overhead when disabled.
     * Results available via getLastPerfResult() after each compress()/decompress().
     */
    void enableProfiling(bool enable);
    bool isProfilingEnabled() const { return profiling_enabled_; }

    /** Performance snapshot from the most recent compress() or decompress() call. */
    const PipelinePerfResult& getLastPerfResult() const { return last_perf_result_; }

    /** The underlying DAG (for advanced/diagnostic use). */
    CompressionDAG* getDAG() { return dag_.get(); }

    /** Pool release threshold in bytes as configured by finalize(). */
    size_t getPoolThreshold() const;

    /**
     * Enable runtime buffer-overwrite detection.
     * After each stage executes, checks that actual output size ≤ allocated capacity.
     * Always active in debug builds regardless of this flag.
     */
    void enableBoundsCheck(bool enable) { dag_->enableBoundsCheck(enable); }
    bool isBoundsCheckEnabled() const   { return dag_->isBoundsCheckEnabled(); }

    /**
     * Enable or disable buffer coloring for PREALLOCATE mode (default: enabled).
     * Disable when per-buffer memory inspection is needed (e.g. cuda-memcheck).
     * Must be called before finalize().
     */
    void setColoringEnabled(bool enable) { dag_->setColoringEnabled(enable); }
    bool isColoringEnabled() const       { return dag_->isColoringEnabled(); }
    size_t getColorRegionCount() const   { return dag_->getColorRegionCount(); }

    // ── CUDA Graph Capture (compression-only) ─────────────────────────────────

    /**
     * Enable CUDA Graph mode. captureGraph() will record the forward compression
     * pass as a replayable CUDA Graph, eliminating per-call CPU dispatch overhead.
     *
     * Requirements: PREALLOCATE strategy, non-zero input_size_hint, all stages
     * graph-compatible, single-source pipeline. Must be set before finalize().
     */
    void enableGraphMode(bool enable);
    bool isGraphModeEnabled() const { return graph_mode_enabled_; }

    /**
     * Record the forward compression pass as a CUDA Graph.
     *
     * After this call compress() uses cudaGraphLaunch() instead of dag_->execute().
     * The input pointer is baked into the graph via a stable internal buffer
     * (d_graph_input_); compress() copies user input there before each launch.
     *
     * Can be called again to re-capture. Must be called after finalize() and
     * before the first compress().
     */
    void captureGraph(cudaStream_t stream = 0);
    bool isGraphCaptured() const { return graph_captured_; }

    size_t getPeakMemoryUsage() const;
    size_t getCurrentMemoryUsage() const;
    void printPipeline() const;

    // ── File Serialization ────────────────────────────────────────────────────

    /** Parsed FZM file header (returned by readHeader()). */
    struct FZMFileHeader {
        FZMHeaderCore               core;
        std::vector<FZMStageInfo>   stages;
        std::vector<FZMBufferEntry> buffers;
    };

    /** Write compressed data to an FZM file. compress() must have been called first. */
    void writeToFile(const std::string& filename, cudaStream_t stream = 0);

    /** Parse the FZM header from a file without decompressing the payload. */
    static FZMFileHeader readHeader(const std::string& filename);

    /** Build the FZM header from current pipeline state. Requires a prior compress(). */
    FZMFileHeader buildHeader() const;

    /**
     * One-shot decompress from an FZM file. Reconstructs the pipeline from the
     * file header, allocates a pool, and runs decompression.
     *
     * Output is always caller-owned (caller must cudaFree *d_output).
     *
     * @param filename             Path to the `.fzm` file.
     * @param d_output             Receives the decompressed device pointer (caller must `cudaFree`).
     * @param output_size          Receives the decompressed size in bytes.
     * @param stream               CUDA stream for all GPU operations.
     * @param perf_out             Optional timing result (GPU compute only, excludes I/O).
     * @param pool_override_bytes  Override automatic pool sizing (0 = automatic).
     *                             Formula: C + 2.5×max_stage_uncompressed + 32 MiB.
     */
    static void decompressFromFile(
        const std::string&  filename,
        void**              d_output,
        size_t*             output_size,
        cudaStream_t        stream             = 0,
        PipelinePerfResult* perf_out           = nullptr,
        size_t              pool_override_bytes = 0
    );

    /**
     * One-shot decompress from an FZM file (instance overload).
     *
     * Behaves identically to the static `decompressFromFile()` overload but
     * respects the setPoolManagedDecompOutput() flag on this instance:
     *   false           → caller must `cudaFree(*d_output)`.
     *   true (default)  → *d_output is pool-owned; do NOT `cudaFree`.
     *
     * The distinct name avoids overload-resolution ambiguity at call sites
     * that are not member functions.
     *
     * @param filename    Path to the `.fzm` file.
     * @param d_output    Receives the decompressed device pointer.
     * @param output_size Receives the decompressed size in bytes.
     * @param stream      CUDA stream for all GPU operations.
     * @param perf_out    Optional timing result.
     */
    void decompressFromFileInstance(
        const std::string&  filename,
        void**              d_output,
        size_t*             output_size,
        cudaStream_t        stream   = 0,
        PipelinePerfResult* perf_out = nullptr
    );

    // ── Config File ───────────────────────────────────────────────────────────

    /**
     * Build and finalize the pipeline from a TOML config file.
     *
     * Adds stages, wires connections, applies pipeline-level settings,
     * then calls finalize() internally. The pipeline must not be finalized
     * before this call.
     *
     * Recognized stage types: Lorenzo1D/2D/3D, Bitshuffle, RZE, RLE,
     * Difference, Zigzag, Negabinary.
     *
     * @throws std::runtime_error  File not found, parse error, unknown stage
     *                             type, bad wiring reference, or already finalized.
     */
    void loadConfig(const std::string& path);

    /**
     * Serialize the current pipeline to a TOML config file.
     *
     * Requires finalize() to have been called. The written file can be passed
     * back to loadConfig() to reconstruct an equivalent pipeline.
     *
     * @throws std::runtime_error  Pipeline not finalized.
     */
    void saveConfig(const std::string& path) const;

private:
    // ── Internal helpers ──────────────────────────────────────────────────────

    Stage* addRawStage(Stage* stage);

    struct OutputBuffer {
        void*       d_ptr;
        size_t      actual_size;
        size_t      allocated_size;
        std::string name;
        int         buffer_id;
    };
    std::vector<OutputBuffer> getOutputBuffers() const;

    static void* loadCompressedData(
        const std::string&   filename,
        const FZMFileHeader& header,
        cudaStream_t         stream = 0,
        MemoryPool*          pool   = nullptr
    );

    void validate();
    std::pair<std::vector<Stage*>, std::vector<Stage*>> identifyTopology();
    void setupInputBuffers(const std::vector<Stage*>& sources);
    int  autoDetectUnconnectedOutputs();
    void detectMultiOutputScenario(int pipeline_outputs);
    void configureStreamsIfNeeded();

    /**
     * Propagate buffer sizes through the DAG from source sizes.
     * force_from_current_inputs=true uses live source buffer sizes (compress-time path
     * for zero-hint pipelines); false uses constructor/per-source hints (finalize path).
     */
    void propagateBufferSizes(bool force_from_current_inputs = false);

    std::vector<Stage*> getSourceStages() const;
    std::vector<Stage*> getSinkStages() const;

    // ── Inverse DAG helpers ───────────────────────────────────────────────────

    /** Compact description of one forward stage used by buildInverseDAG(). */
    struct FwdStageDesc {
        Stage*           stage;
        std::vector<int> output_buf_ids;
        std::vector<int> input_buf_ids;
    };

    /** fwd_buf_id → {device pointer, size in bytes} for each pipeline-output buffer. */
    using PipelineOutputMap = std::unordered_map<int, std::pair<void*, size_t>>;

    /**
     * Build, wire, and finalize an inverse DAG from a forward topology description.
     * Shared between decompress() and decompressFromFile().
     * Returns {inv_dag (finalized, ready to execute), source Stage* → result buffer ID}.
     */
    static std::pair<std::unique_ptr<CompressionDAG>,
                     std::unordered_map<Stage*, int>>
    buildInverseDAG(
        const std::vector<FwdStageDesc>&          fwd_stages,
        const PipelineOutputMap&                  pipeline_outputs,
        MemoryPool*                               pool,
        MemoryStrategy                            strategy,
        const std::unordered_map<Stage*, size_t>& source_sizes,
        bool                                      enable_profiling
    );

    // ── Concat helpers ────────────────────────────────────────────────────────

    struct OutputBufferInfo {
        int         buffer_id;
        void*       d_ptr;
        size_t      actual_size;
        std::string stage_name;
        std::string output_name;
    };

    std::vector<OutputBufferInfo> collectOutputBuffers() const;

    /** Total bytes for concat format: [num_bufs:4B][size:8B][data]... */
    size_t calculateConcatSize(const std::vector<OutputBufferInfo>& outputs) const;

    size_t writeConcatBuffer(
        const std::vector<OutputBufferInfo>& outputs,
        uint8_t*     d_concat_bytes,
        cudaStream_t stream
    ) const;

    void concatOutputs(void** d_output, size_t* output_size, cudaStream_t stream);

    // ── Member variables ──────────────────────────────────────────────────────

    std::unique_ptr<MemoryPool>      mem_pool_;
    std::unique_ptr<CompressionDAG>  dag_;
    MemoryStrategy                   strategy_;

    std::vector<std::unique_ptr<Stage>> stages_;
    std::unordered_map<Stage*, DAGNode*> stage_to_node_;

    struct ConnectionInfo {
        Stage*      dependent;
        Stage*      producer;
        std::string output_name;
        int         output_index;
    };
    std::vector<ConnectionInfo> connections_;

    int  num_streams_;
    bool is_finalized_;
    bool warmup_on_finalize_;
    bool pool_managed_decomp_;

    // is_compressed_: true after the first successful compress() (gates writeToFile).
    // was_compressed_: true between compress() and the next reset() (gates captureGraph).
    bool is_compressed_;
    bool was_compressed_;

    bool profiling_enabled_;
    PipelinePerfResult last_perf_result_;

    std::vector<DAGNode*> input_nodes_;
    std::vector<DAGNode*> output_nodes_;
    std::vector<int>      input_buffer_ids_;
    std::vector<int>      output_buffer_ids_;

    void*  d_concat_buffer_;
    size_t concat_buffer_capacity_;
    bool   needs_concat_;

    // Pool-persistent decompress output buffers (one per source stage).
    // Only used when pool_managed_decomp_ == true.
    std::vector<void*> d_decomp_outputs_;

    // Pinned host buffer for concat header. Allocated once at finalize() to
    // avoid N+1 tiny H2D copies — CPU packs the header then issues one H2D copy.
    void*  h_concat_header_;
    size_t h_concat_header_capacity_;

    // Persistent pinned host + device descriptor buffers for the gather kernel.
    // Sized at finalize() for max_outputs; reused every compress call.
    void*  h_copy_descs_;
    void*  d_copy_descs_;
    size_t copy_descs_capacity_;

    size_t input_size_;

    // Per-source input sizes from the most recent compress(), ordered to match
    // input_nodes_. Used by decompress() to size each inverse result buffer.
    std::vector<size_t> source_input_sizes_;

    // Input alignment in bytes — LCM of all stage getRequiredInputAlignment() values,
    // computed at finalize(). compress() zero-pads to this boundary transparently.
    size_t input_alignment_bytes_;
    void*  d_pad_buf_;
    size_t d_pad_buf_size_;

    // Original (pre-padding) input size. decompress() uses this to trim the
    // reported output back to what the caller provided. 0 when no padding.
    size_t original_input_size_;

    size_t input_size_hint_;
    float  pool_multiplier_;

    // Dataset dimensions (x=fast, y, z). Used by convenience.h addLorenzo() to
    // select 1-D/2-D/3-D automatically. Default {0,1,1} = 1-D, infer x from input.
    std::array<size_t, 3> dims_;

    /**
     * Cached inverse DAG for repeated decompress() calls.
     *
     * Built lazily on the first decompressMulti() call. On reuse, only the
     * external compressed-data pointers are updated; topology, events, and
     * (in PREALLOCATE mode) buffer allocations are preserved. Invalidated when
     * source sizes change between compress() calls.
     */
    struct InvDAGCache {
        std::unique_ptr<CompressionDAG>    inv_dag;
        std::unordered_map<Stage*, int>    inv_result_map;
        std::unordered_map<int, int>       fwd_to_inv_ext_buf;
        std::unordered_map<Stage*, size_t> source_sizes;
    };
    std::unique_ptr<InvDAGCache> inv_cache_;

    struct BufferMetadata {
        int         buffer_id;
        size_t      actual_size;
        size_t      allocated_size;
        std::string name;
        DAGNode*    producer;
        int         output_index;
    };
    std::vector<BufferMetadata> buffer_metadata_;

    bool graph_mode_enabled_;
    bool graph_captured_;

    // Fixed device input buffer whose address is baked into the captured graph.
    // compress() copies user input here before cudaGraphLaunch().
    void*  d_graph_input_;
    size_t d_graph_input_size_;

    cudaGraph_t     captured_graph_;
    cudaGraphExec_t graph_exec_;
};

// ── Template implementation ───────────────────────────────────────────────────

template<typename StageT, typename... Args>
StageT* Pipeline::addStage(Args&&... args) {
    if (is_finalized_) {
        throw std::runtime_error("Cannot add stages after finalization");
    }

    auto stage_ptr = std::make_unique<StageT>(std::forward<Args>(args)...);
    StageT* stage  = stage_ptr.get();

    DAGNode* node        = dag_->addStage(stage, stage->getName());
    size_t   num_outputs = stage->getNumOutputs();
    auto     output_names = stage->getOutputNames();

    // Pre-allocate all output slots as unconnected (size=1 placeholder).
    // connect() will promote any that get wired to downstream stages.
    for (size_t i = 0; i < num_outputs; i++) {
        std::string out_name = i < output_names.size() ? output_names[i] : std::to_string(i);
        dag_->addUnconnectedOutput(node, 1, i, stage->getName() + "." + out_name + "_unconnected");
    }

    stage_to_node_[stage] = node;
    stages_.push_back(std::move(stage_ptr));
    return stage;
}

} // namespace fz
