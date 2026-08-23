/**
 * @file include/pipeline/compressor.h
 * @brief Pipeline builder and execution API.
 */
#pragma once

#include "backend/types.h"
#include "advanced/dag.h"
#include "pipeline/device_buffer.h"
#include "pipeline/perf.h"
#include "pipeline/config.h"
#include "stage/stage.h"
#include "stage/stage_factory.h"
#include "mem/mempool.h"
#include "fzm_format.h"

#include <array>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

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
 *  - decompress() output is pool-owned by default - do NOT cudaFree it.
 *    Call setPoolManagedDecompOutput(false) to receive a caller-owned pointer instead.
 */
/**
 * Kernel-fusion policy for compress. `Off` (default) runs every stage staged.
 * `Auto` runs the fusion planner at finalize() and, for each fusable chain that
 * has a registered fused implementation, replaces the group's staged execute()s
 * with one fused kernel — a compress-only optimization that leaves the archive
 * byte-identical (decompress is unaffected). Groups with no registered impl stay
 * staged, so `Auto` is always safe. Enabling fusion disables CUDA graph capture.
 * Overridable at runtime with FZ_FUSION=off|auto. See CN-FUSE-PROOF/PLAN.
 */
enum class FusionPolicy { Off, Auto };

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

    /** Kernel-fusion policy (default Off). Must be called before finalize(). */
    void setFusionPolicy(FusionPolicy mode) { fusion_policy_ = mode; }
    FusionPolicy getFusionPolicy() const { return fusion_policy_; }
    /** Number of fused groups installed at finalize() (0 unless Auto matched). */
    size_t getFusedGroupCount() const { return dag_ ? dag_->getFusedGroupCount() : 0; }

    /** Number of parallel CUDA streams for level-based execution. Must be called before finalize(). */
    void setNumStreams(int num_streams);

    /**
     * Dataset spatial dimensions. Forwarded to every stage immediately on addStage()
     * and again at finalize(). Call setDims() before addStage() so that dimension-
     * aware stages (e.g. LorenzoQuantStage) have the correct ndim() from the moment
     * they are added. Default: 1-D ({n, 1, 1}).
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
    void warmup(fz::stream_t stream = 0);

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

    /**
     * Return the uncompressed byte count from the most recent compress() call.
     *
     * Returns the original (pre-padding) input size so the value always matches
     * what the caller passed in, even when the pipeline transparently padded the
     * input for alignment. Returns 0 if compress() has never been called.
     *
     * Useful for pre-allocating a decompression output buffer without needing
     * out-of-band metadata:
     * @code
     *   pipeline.compress(d_in, in_bytes, &d_comp, &comp_sz, stream);
     *   size_t decomp_bytes = pipeline.getLastUncompressedSize();
     *   // allocate d_out of size decomp_bytes, then:
     *   pipeline.decompress(d_comp, comp_sz, d_out, decomp_bytes, &actual, stream);
     * @endcode
     *
     * The value persists across reset() — reset() invalidates the compressed
     * output pointer but the size remains meaningful for planning the next call.
     */
    size_t getLastUncompressedSize() const {
        return original_input_size_ > 0 ? original_input_size_ : input_size_;
    }

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
        fz::stream_t stream = 0
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
        fz::stream_t stream = 0
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
        fz::stream_t stream = 0
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
        fz::stream_t stream = 0
    );

    /**
     * Fully stream-asynchronous decompress into a caller-provided buffer — for
     * overlapped / double-buffered decode loops where decompress() calls on
     * distinct streams must run concurrently.
     *
     * Unlike the synchronous `decompress()` overloads, this:
     *   - writes the inverse result straight into `d_output_buf` (no internal
     *     `cudaMalloc`, no device→device copy, no `cudaFree` — those are all
     *     device-wide barriers that prevent cross-stream overlap), and
     *   - does **not** call `cudaStreamSynchronize()` before returning.
     *
     * `*actual_output_size` is the planned output size (known host-side from the
     * inverse DAG); the decompressed bytes are valid only after the caller
     * synchronizes `stream` itself. A capacity check is done host-side and throws
     * before any GPU work if `d_output_buf` is too small.
     *
     * Requirements:
     *   - **PREALLOCATE** memory strategy (internal inverse buffers are allocated
     *     once at finalize, never per call). Throws under MINIMAL.
     *   - Each concurrent stream uses its **own** Pipeline instance (independent
     *     pools/state); this matches the cached-inverse-DAG reuse model.
     *
     * Caveat: some inverse coder stages (e.g. RZE/RRE/Huffman/AdaptiveBitpack) do a
     * blocking device→host header read **inside** their `execute()`. Those readbacks
     * are internal to the stage and are NOT removed by this method, so a pipeline
     * built from them still serializes at that point on a single host thread. This
     * method removes only the Pipeline-level serializers; for full overlap of such
     * pipelines, drive each slot from its own host thread (so one slot's readback
     * does not stall another's GPU work).
     *
     * @param d_input             Compressed blob (device pointer).
     * @param input_size          Size of the compressed blob in bytes.
     * @param d_output_buf        Caller-owned device buffer for the result.
     * @param output_buf_capacity Capacity of d_output_buf in bytes.
     * @param actual_output_size  Receives the planned decompressed size.
     * @param stream              CUDA stream for all GPU operations.
     */
    void decompressInto(
        const void* d_input,
        size_t      input_size,
        void*       d_output_buf,
        size_t      output_buf_capacity,
        size_t*     actual_output_size,
        fz::stream_t stream = 0
    );

    // ── Explicit-ownership execution API ──────────────────────────────────────
    //
    // Span-based wrappers over the pointer overloads above. Behavior is
    // identical — these exist so ownership is visible in the signature instead
    // of depending on `void**` vs `void*` and on setPoolManagedDecompOutput().
    // Prefer these in new code; the pointer overloads remain supported.

    /**
     * Compress into pool memory. The returned buffer is borrowed from the pool:
     * do NOT free it, and treat it as invalidated by the next compress(),
     * reset(), or Pipeline destruction.
     */
    BorrowedDeviceBuffer compress(ConstDeviceSpan input, fz::stream_t stream = 0);

    /**
     * Compress into a caller-provided buffer. Returns the exact bytes written.
     * Throws if `output` is too small (see the pointer overload for sizing);
     * incompatible with CUDA Graph mode.
     */
    size_t compressInto(ConstDeviceSpan input, DeviceSpan output, fz::stream_t stream = 0);

    /**
     * Decompress, returning a buffer borrowed from the pool: do NOT free it.
     * Valid until the next decompress() or Pipeline destruction.
     *
     * An empty `input` means "read from the forward DAG's live buffers", the
     * same as passing nullptr to the pointer overload. Independent of
     * setPoolManagedDecompOutput() — this call always borrows.
     */
    BorrowedDeviceBuffer decompressBorrowed(ConstDeviceSpan input, fz::stream_t stream = 0);

    /**
     * Decompress into a fresh caller-owned allocation, released when the
     * returned buffer is destroyed. Independent of
     * setPoolManagedDecompOutput() — this call always owns.
     */
    OwnedDeviceBuffer decompressOwned(ConstDeviceSpan input, fz::stream_t stream = 0);

    /**
     * Decompress into a caller-provided buffer. Returns the exact bytes
     * written. Synchronous, like the `decompress()` pointer overload it wraps
     * — for the fully stream-asynchronous form use decompressIntoAsync().
     */
    size_t decompressInto(ConstDeviceSpan input, DeviceSpan output, fz::stream_t stream = 0);

    /**
     * Stream-asynchronous decompress into a caller-provided buffer; wraps the
     * `decompressInto()` pointer overload, including its PREALLOCATE
     * requirement and its "bytes are valid only after you synchronize"
     * contract. Returns the planned output size.
     */
    size_t decompressIntoAsync(ConstDeviceSpan input, DeviceSpan output, fz::stream_t stream = 0);

    /**
     * Make a finalized pipeline ready to decompress() external blobs without a
     * prior compress() call.
     *
     * Normally decompress() relies on state populated by compress() (the list of
     * pipeline-output buffers and the source's uncompressed size). When a pipeline
     * instance only ever decodes blobs produced elsewhere — e.g. K decode-only
     * slots reading archives off disk — this would otherwise force a throwaway
     * "warmup" compress() over dummy data purely to populate that state.
     *
     * prepareInverse() builds the pipeline-output buffer metadata directly from the
     * finalized forward topology and records the source uncompressed size, so the
     * next decompress() works straight away. It launches no kernels and touches no
     * device memory; the inverse DAG is still built lazily on the first decompress()
     * exactly as it is after a real compress().
     *
     * Per-segment compressed sizes are NOT needed here — decompress() reads them
     * from the blob's own self-describing concat header, so blobs of differing
     * sizes all decode correctly. Call this again to update the uncompressed size
     * if a later blob decodes to a different element count.
     *
     * @param uncompressed_size  Byte size of the original (decompressed) data.
     *                           Used to size the decompress output buffer.
     */
    void prepareInverse(size_t uncompressed_size);

    /** Free non-persistent buffers and reset execution state for re-use. */
    void reset(fz::stream_t stream = 0);

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
     * Returns true if the internal memory pool is running in cudaMalloc fallback mode.
     *
     * Fallback mode is active when pool creation failed at construction time (e.g. vGPU
     * environments), when `FZ_FORCE_MEMPOOL_FALLBACK` is set in the environment, or when
     * `MemoryPoolConfig::force_fallback` was passed to the Pipeline constructor.
     * In fallback mode all allocations use `cudaMalloc`/`cudaFree` with explicit stream
     * synchronization rather than stream-ordered pool allocations.
     */
    bool isMemPoolFallbackMode() const;

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
    void setColoringEnabled(bool enable) {
        coloring_enabled_ = enable;              // survives a setMemoryStrategy() DAG swap
        dag_->setColoringEnabled(enable);
    }
    bool isColoringEnabled() const       { return dag_->isColoringEnabled(); }
    size_t getColorRegionCount() const   { return dag_->getColorRegionCount(); }

    /**
     * What coloring was *asked for*, as opposed to isColoringEnabled(), which
     * reports whether the DAG actually applied it.  The two differ under
     * MINIMAL (nothing to color) and before finalize().  A benchmark row wants
     * the requested value, so a colored/uncolored pair can be identified even
     * when a topology gave coloring nothing to do.
     */
    bool isColoringRequested() const     { return coloring_enabled_; }

    /**
     * Per-stage notes from the last run — see Stage::getRunNotes().  Keyed by
     * stage name; stages with nothing to report are omitted, so an empty map is
     * the normal case and means "nothing surprising happened."
     *
     * Exists so benchmark rows can record events that change how a result should
     * be compared (currently: a Huffman codebook fallback) rather than losing
     * them to a log line.
     */
    std::unordered_map<std::string, std::vector<std::string>> collectRunNotes() const {
        std::unordered_map<std::string, std::vector<std::string>> notes;
        for (const auto& s : stages_) {
            if (!s) continue;
            auto n = s->getRunNotes();
            if (!n.empty()) notes.emplace(s->getName(), std::move(n));
        }
        return notes;
    }

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
    void captureGraph(fz::stream_t stream = 0);
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
    void writeToFile(const std::string& filename, fz::stream_t stream = 0);

    /** Parse the FZM header from a file without decompressing the payload. */
    static FZMFileHeader readHeader(const std::string& filename);

    /** Build the FZM header from current pipeline state. Requires a prior compress(). */
    FZMFileHeader buildHeader() const;

    // ── In-memory metadata header (decode without a prior compress) ────────────

    /**
     * Serialize this pipeline's FZM metadata header to a host byte buffer.
     *
     * Requires a prior compress(). The returned bytes are the same core + stage +
     * buffer header that writeToFile() prepends to an `.fzm` file, WITHOUT the
     * compressed payload. They carry everything the inverse path needs that is not
     * recoverable from the raw compressed blob — most importantly the data-dependent
     * per-stage inverse metadata (e.g. HuffmanStage's symbol count, the quantizer
     * outlier count). Store these bytes alongside the compress() output blob.
     *
     * The companion is primeInverseFromHeader(): a fresh, finalized, decode-only
     * pipeline of the same topology can primeInverseFromHeader(these_bytes) and then
     * decompress() the blob directly — no throwaway "warmup" compress() required.
     */
    std::vector<uint8_t> serializeHeaderToMemory() const;

    /**
     * Prepare a finalized pipeline to decompress() an external blob whose metadata
     * header was produced by serializeHeaderToMemory(), without a prior compress().
     *
     * Restores each stage's inverse-side configuration from the header (the analogue
     * of what decompressFromFile() does via createStage()), records the source
     * uncompressed size, and builds the pipeline-output buffer metadata. After this
     * call, decompress(blob, blob_size, ...) works and reuses the cached inverse DAG
     * across successive blocks. Call once per blob to refresh the data-dependent
     * metadata (e.g. a new outlier count); the cached inverse DAG is kept when the
     * source size is unchanged.
     *
     * This pipeline must have been built with the SAME stage topology as the one
     * that produced the header (same addStage()/connect() sequence). A stage-type
     * mismatch throws.
     *
     * @param header_bytes  Bytes returned by serializeHeaderToMemory() on the producer.
     * @param header_size   Length of header_bytes.
     */
    void primeInverseFromHeader(const void* header_bytes, size_t header_size);

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
        fz::stream_t        stream             = 0,
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
        fz::stream_t        stream   = 0,
        PipelinePerfResult* perf_out = nullptr
    );

    /**
     * One-shot decompress of an external in-memory blob, given its metadata header
     * — the fused convenience for decode-only pipelines.
     *
     * Equivalent to primeInverseFromHeader(header...) followed by
     * decompress(d_blob...). This is the single per-blob call for a streaming
     * decode loop: it restores the data-dependent inverse metadata from the header
     * and decodes the blob, reusing this instance's cached inverse DAG across calls
     * (so there is no per-blob DAG rebuild — unlike the static decompressFromFile()
     * path which reconstructs everything each call).
     *
     * `header_bytes` come from serializeHeaderToMemory() on the producer; `d_blob`
     * is the producer's compress() output (the two may be stored separately, or
     * concatenated and sliced by the caller). Output ownership follows
     * setPoolManagedDecompOutput() as for decompress().
     *
     * @param header_bytes  Bytes from serializeHeaderToMemory().
     * @param header_size   Length of header_bytes.
     * @param d_blob        Device pointer to the compressed blob.
     * @param blob_size     Size of the compressed blob in bytes.
     * @param d_output      Receives the decompressed device pointer.
     * @param output_size   Receives the decompressed size in bytes.
     * @param stream        CUDA stream for all GPU operations.
     */
    void decompressFromMemory(
        const void*  header_bytes,
        size_t       header_size,
        const void*  d_blob,
        size_t       blob_size,
        void**       d_output,
        size_t*      output_size,
        fz::stream_t stream = 0
    );

    // ── Config File ───────────────────────────────────────────────────────────

    /**
     * Build and finalize the pipeline from a TOML config file.
     *
     * Adds stages, wires connections, applies pipeline-level settings,
     * then calls finalize() internally. The pipeline must not be finalized
     * before this call.
     *
     * Recognized stage types: LorenzoQuant, Lorenzo, Quantizer,
     * Bitshuffle, RZE, RRE, RLE, Bitpack, Difference, Zigzag, Negabinary.
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
    // ── RAII buffer wrappers (private implementation detail) ─────────────────

    // Pool-allocated persistent device buffer.
    struct PoolBuffer {
        void*       ptr      = nullptr;
        size_t      capacity = 0;
        MemoryPool* pool     = nullptr;

        ~PoolBuffer()                         { free(0); }
        PoolBuffer()                          = default;
        PoolBuffer(const PoolBuffer&)         = delete;
        PoolBuffer& operator=(const PoolBuffer&) = delete;

        void free(fz::stream_t s) {
            if (ptr && pool) { pool->free(ptr, s); ptr = nullptr; capacity = 0; }
        }
        bool allocate(MemoryPool* p, size_t bytes, fz::stream_t s,
                      const char* tag, bool persistent = false) {
            free(s);
            pool = p;
            ptr  = pool->allocate(bytes, s, tag, persistent);
            if (ptr) capacity = bytes;
            return ptr != nullptr;
        }
    };

    // cudaHostAlloc pinned host buffer — grows on demand, never shrinks.
    struct PinnedBuffer {
        void*  ptr      = nullptr;
        size_t capacity = 0;

        ~PinnedBuffer()                           { if (ptr) cudaFreeHost(ptr); }
        PinnedBuffer()                            = default;
        PinnedBuffer(const PinnedBuffer&)         = delete;
        PinnedBuffer& operator=(const PinnedBuffer&) = delete;

        // Returns false on CUDA allocation failure.
        bool ensureCapacity(size_t bytes) {
            if (capacity >= bytes) return true;
            if (ptr) { cudaFreeHost(ptr); ptr = nullptr; capacity = 0; }
            if (cudaHostAlloc(&ptr, bytes, cudaHostAllocDefault) != cudaSuccess) return false;
            capacity = bytes;
            return true;
        }
    };

    // cudaMalloc device buffer — grows on demand, never shrinks.
    struct DeviceBuffer {
        void*  ptr      = nullptr;
        size_t capacity = 0;

        ~DeviceBuffer()                           { if (ptr) cudaFree(ptr); }
        DeviceBuffer()                            = default;
        DeviceBuffer(const DeviceBuffer&)         = delete;
        DeviceBuffer& operator=(const DeviceBuffer&) = delete;

        // Returns false on CUDA allocation failure.
        bool ensureCapacity(size_t bytes) {
            if (capacity >= bytes) return true;
            if (ptr) { cudaFree(ptr); ptr = nullptr; capacity = 0; }
            if (cudaMalloc(&ptr, bytes) != cudaSuccess) return false;
            capacity = bytes;
            return true;
        }
    };

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
        fz::stream_t         stream = 0,
        MemoryPool*          pool   = nullptr
    );

    void validate();
    std::pair<std::vector<Stage*>, std::vector<Stage*>> identifyTopology();
    void setupInputBuffers(const std::vector<Stage*>& sources);
    int  autoDetectUnconnectedOutputs();
    void detectMultiOutputScenario(int pipeline_outputs);
    void configureStreamsIfNeeded();

    // finalize() sub-steps
    void typeCheckConnections();
    void computeInputAlignment();
    /// Run the fusion planner and install matched fused groups on the DAG
    /// (Auto mode / FZ_FUSION=auto). No-op otherwise. May disable graph mode.
    void planAndInstallFusion();
    void notifyStagesFinalizeHooks();
    void refinePoolSize();
    void setupGraphModeInput();
    void preallocatePadBuffer();
    void preallocateConcatBuffers();

    // compress() helper: handles graph-mode copy or alignment padding.
    // Returns the effective source pointer and padded source size.
    std::pair<const void*, size_t> prepareInputSource(
        const void* d_input, size_t input_size, fz::stream_t stream);

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

    // decompress() helper: builds or reuses the inverse DAG cache.
    void buildOrReuseInvCache(
        const PipelineOutputMap& po_map,
        Stage*       src_stage,
        size_t       src_sz,
        fz::stream_t stream);

    /**
     * Shared inverse-execution core behind all decompress() overloads.
     * @param caller_output  if non-null, the inverse result is written here (no
     *                       internal allocation/copy); ownership stays with the
     *                       caller. If null, the library allocates (pool or
     *                       cudaMalloc per setPoolManagedDecompOutput()).
     * @param caller_capacity capacity of caller_output (checked host-side).
     * @param synchronize    if false, skip the post-execute cudaStreamSynchronize,
     *                       postStreamSync, size-refine and profiling collection
     *                       (the planned size from the inverse DAG is reported);
     *                       caller must synchronize the stream before reading.
     */
    void decompressCore(
        const void* d_input,
        size_t      input_size,
        void*       caller_output,
        size_t      caller_capacity,
        bool        synchronize,
        void**      d_output,
        size_t*     output_size,
        fz::stream_t stream);

    /**
     * Populate buffer_metadata_ from the finalized forward topology
     * (output_buffer_ids_ / output_nodes_) with placeholder sizes. Lets
     * decompress() route an external blob without a prior compress(). The
     * authoritative per-segment compressed sizes come from the blob's concat
     * header at decompress time, so the placeholders here are never used as sizes.
     */
    void buildStaticBufferMetadata();

    /**
     * Read the `n` per-segment compressed sizes from an external blob's concat
     * header (`[count:4B][size:8B * n]`), validating that the embedded count
     * matches the pipeline's pipeline-output count. Single D2H copy of the header.
     */
    std::vector<size_t> readConcatSegmentSizes(
        const void* d_blob, size_t n, fz::stream_t stream) const;

    // decompressFromFile() helpers.
    /** Parse + validate an FZM header from an in-memory byte buffer (mirrors readHeader()). */
    static FZMFileHeader parseHeaderFromMemory(const void* data, size_t size);
    static size_t computeFilePoolSize(const FZMFileHeader& fh, size_t pool_override_bytes);
    static std::pair<std::vector<std::unique_ptr<Stage>>, std::vector<FwdStageDesc>>
        reconstructForwardTopology(const FZMFileHeader& fh);
    static std::unordered_map<Stage*, size_t> buildSourceSizesFromHeader(
        const FZMFileHeader& fh, const std::vector<FwdStageDesc>& fwd_topology);

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
        fz::stream_t stream
    ) const;

    void concatOutputs(void** d_output, size_t* output_size, fz::stream_t stream);

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
    /// Pipeline-level mirror of the DAG's coloring flag. Needed because
    /// setMemoryStrategy() replaces dag_ wholesale and must restore it.
    bool coloring_enabled_ = true;
    PipelinePerfResult last_perf_result_;

    std::vector<DAGNode*> input_nodes_;
    std::vector<DAGNode*> output_nodes_;
    std::vector<int>      input_buffer_ids_;
    std::vector<int>      output_buffer_ids_;

    PoolBuffer   d_concat_buffer_;
    bool         needs_concat_;

    // Pool-persistent decompress output buffers (one per source stage).
    // Only used when pool_managed_decomp_ == true.
    std::vector<void*> d_decomp_outputs_;

    // Pinned host buffer for concat header (one H2D copy instead of N).
    PinnedBuffer h_concat_header_;
    // Persistent pinned host + device descriptor buffers for the gather kernel.
    PinnedBuffer h_copy_descs_;
    DeviceBuffer d_copy_descs_;

    size_t input_size_;

    // Per-source input sizes from the most recent compress(), ordered to match
    // input_nodes_. Used by decompress() to size each inverse result buffer.
    std::vector<size_t> source_input_sizes_;

    // Input alignment in bytes — LCM of all stage getRequiredInputAlignment() values.
    // compress() zero-pads to this boundary transparently.
    size_t     input_alignment_bytes_;
    PoolBuffer d_pad_buf_;

    // Original (pre-padding) input size. decompress() uses this to trim the
    // reported output back to what the caller provided. 0 when no padding.
    size_t original_input_size_;

    size_t input_size_hint_;
    float  pool_multiplier_;

    // Dataset dimensions (x=fast, y, z). Pushed to each stage on addStage() and
    // again at finalize(). Default {0,1,1} = 1-D, infer x from input size.
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
    FusionPolicy fusion_policy_ = FusionPolicy::Off;

    // Fixed device input buffer whose address is baked into the captured graph.
    // compress() copies user input here before cudaGraphLaunch().
    PoolBuffer d_graph_input_;
    size_t     d_graph_input_size_;

    fz::graph_t      captured_graph_;
    fz::graph_exec_t graph_exec_;
};

// ── Template implementation ───────────────────────────────────────────────────

template<typename StageT, typename... Args>
StageT* Pipeline::addStage(Args&&... args) {
    if (is_finalized_) {
        throw std::runtime_error("Cannot add stages after finalization");
    }

    // if constexpr, not a runtime check: on an unsupported backend, StageT's
    // constructor may not exist in the build at all (its .cu translation
    // unit excluded — see Stage::isSupportedOnBackend()'s doc comment), so
    // every line below that references `new StageT()` must never be
    // instantiated at all, not merely never executed — hence the whole rest
    // of the function lives in the `if constexpr` branch rather than after
    // a standalone early-throw.
    if constexpr (!StageT::isSupportedOnBackend()) {
        throw std::runtime_error(
            "addStage(): this stage type is not supported on the current "
            "GPU backend (FZGMOD_BACKEND) this library was built for");
    } else {
        auto stage_ptr = std::make_unique<StageT>(std::forward<Args>(args)...);
        StageT* stage  = stage_ptr.get();

        stage->setDims(dims_);

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
}

} // namespace fz
