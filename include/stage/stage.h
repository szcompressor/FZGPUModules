/**
 * @file include/stage/stage.h
 * @brief Base class interface for all compression stages.
 */
#pragma once

#include "backend/types.h"
#include "fzm_format.h"
#include "stage/fusion.h"
#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace fz {

// Forward declaration — avoids requiring mempool.h in every stage header
class MemoryPool;

/**
 * Base class for all compression/decompression stages.
 *
 * A stage is a single transformation in the pipeline (e.g. Lorenzo predictor,
 * RLE encoder, bitshuffle).  The pipeline interacts with stages exclusively
 * through this interface — no downcasting or type-name branching anywhere in
 * the pipeline or DAG code.
 *
 */
class Stage {
public:
    virtual ~Stage() = default;

    /**
     * Whether this stage type is supported on the backend the library was
     * built for. Default true (every stage supports every backend) — a
     * stage that doesn't (e.g. ANSStage on HIP/SYCL, whose vendored PTX
     * lanemask assembly has no translation) hides this with its own
     * `static constexpr bool isSupportedOnBackend()`.
     *
     * Deliberately `static constexpr`, not `virtual`: `Pipeline::addStage<T>()`
     * must be able to check this *before* `T` is ever constructed, via
     * `if constexpr`, so that on an unsupported backend the `new T()` branch
     * is never instantiated at all — not merely never executed. That matters
     * because an unsupported stage's own translation unit may be excluded
     * from the build entirely (see CMakeLists.txt's `HEADER_FILE_ONLY`
     * handling for ans_stage.cu on HIP); a virtual/instance method can't be
     * called without an object to call it on, which would require the very
     * constructor this mechanism exists to avoid referencing.
     */
    static constexpr bool isSupportedOnBackend() { return true; }

    /**
     * Execute the stage. Inputs, outputs, and sizes are device pointers/bytes.
     *
     * Stages may call cudaStreamSynchronize(stream) or issue blocking D2H copies
     * when the algorithm requires it (e.g. Huffman histogram readback for codebook
     * construction, ANS renormalization tables).  Such stages must return false from
     * isGraphCompatible() and must document the sync points.
     *
     * Note: the DAG dispatches sibling nodes (same topological level) via a
     * sequential CPU loop, each enqueuing to its own stream.  A sync inside
     * execute() blocks the CPU from dispatching subsequent siblings until the
     * synced stream is idle — this delays parallel branches in wide DAGs.
     * In a linear pipeline there are no siblings and no extra cost.
     */
    virtual void execute(
        fz::stream_t stream,
        MemoryPool* pool,
        const std::vector<void*>& inputs,
        const std::vector<void*>& outputs,
        const std::vector<size_t>& sizes
    ) = 0;

    /** Human-readable name used in error messages and debug output. */
    virtual std::string getName() const = 0;

    virtual size_t getNumInputs()  const = 0;
    virtual size_t getNumOutputs() const = 0;

    /**
     * Minimum input size alignment in bytes.
     * Chunked stages return their chunk size; the pipeline uses the LCM of all
     * stage alignments at finalize() to transparently zero-pad the input.
     * Default: 1 (no alignment requirement).
     */
    virtual size_t getRequiredInputAlignment() const { return 1; }

    /**
     * Output port names in order. Default: single port named "output".
     * Multi-output stages (e.g. Lorenzo: "codes", "outliers") override this.
     */
    virtual std::vector<std::string> getOutputNames() const {
        return {"output"};
    }

    /** Returns the index of a named output port, or -1 if not found. */
    int getOutputIndex(const std::string& name) const {
        auto names = getOutputNames();
        for (size_t i = 0; i < names.size(); i++) {
            if (names[i] == name) return static_cast<int>(i);
        }
        return -1;
    }

    /**
     * Estimate output buffer sizes given input sizes.
     * Used for buffer allocation planning in PREALLOCATE mode — must be
     * a safe upper bound; under-estimation causes buffer overruns.
     */
    virtual std::vector<size_t> estimateOutputSizes(
        const std::vector<size_t>& input_sizes
    ) const = 0;

    /** Actual output sizes after execute(), keyed by output port name. */
    virtual std::unordered_map<std::string, size_t> getActualOutputSizesByName() const = 0;

    /**
     * Actual size of a single output by index after execute().
     * Avoids constructing the map for the common single-output case.
     * Default delegates to getActualOutputSizesByName(); override to return
     * directly from an internal field.
     */
    virtual size_t getActualOutputSize(int index) const {
        auto names = getOutputNames();
        if (index < 0 || index >= static_cast<int>(names.size())) return 0;
        auto m  = getActualOutputSizesByName();
        auto it = m.find(names[index]);
        return (it != m.end()) ? it->second : 0;
    }

    /**
     * Switch between forward (compression) and inverse (decompression) mode.
     * Affects getNumInputs()/getNumOutputs() for stages with asymmetric port counts.
     */
    virtual void setInverse(bool inverse) { (void)inverse; }
    virtual bool isInverse() const { return false; }

    /** Stage type identifier written into the FZM file header. */
    virtual uint16_t getStageTypeId() const = 0;

    /** DataType enum of the given output port. */
    virtual uint8_t getOutputDataType(size_t output_index) const = 0;

    /**
     * Expected DataType of the given input port.
     *
     * Used by Pipeline::finalize() to detect type mismatches between connected
     * stages before any execution.  Return DataType::UNKNOWN to opt out of
     * checking — byte-transparent stages (Bitshuffle, RZE, RRE) and mock stages
     * must return UNKNOWN; finalize() skips any connection where either side
     * is UNKNOWN.
     */
    virtual uint8_t getInputDataType(size_t /*input_index*/) const {
        return static_cast<uint8_t>(DataType::UNKNOWN);
    }

    /**
     * Serialize stage config into header_buffer (max 128 bytes) for the FZM file.
     * Return the number of bytes written, or 0 if the stage has no config.
     */
    virtual size_t serializeHeader(size_t output_index, uint8_t* header_buffer, size_t max_size) const {
        (void)output_index; (void)header_buffer; (void)max_size;
        return 0;
    }

    /** Restore stage config from header_buffer during decompression. */
    virtual void deserializeHeader(const uint8_t* header_buffer, size_t size) {
        (void)header_buffer; (void)size;
    }

    /**
     * Save/restore config state around a decompression pass.
     * deserializeHeader() overwrites the stage's forward-pass config; saveState()
     * is called before and restoreState() after so the stage returns to its
     * original configuration.
     */
    virtual void saveState()    {}
    virtual void restoreState() {}

    /**
     * Notes about *what this stage actually did* on the last run, when that
     * differs from what was configured in a way that affects comparability.
     *
     * Motivating case: `HuffmanStage` silently falls back to an Adaptive book
     * when a `PerBlock`/`Fixed` build drives a symbol past the 27-bit code
     * field.  The fallback is correct — it does not relax the error bound — but
     * a field encoded with a different codebook is not compression-ratio
     * comparable to one that was not, and `getBookSource()` deliberately keeps
     * reporting what was *asked for*.  Without a channel like this, a benchmark
     * row records the two cases identically and the difference is unrecoverable
     * after the fact.
     *
     * Returns short stable machine-readable tokens (e.g. `"adaptive_fallback"`),
     * not prose — these are meant to land in a benchmark row and be grouped on.
     * Empty by default; a stage that never surprises its caller need not
     * implement it.
     */
    virtual std::vector<std::string> getRunNotes() const { return {}; }

    /**
     * Called once by Pipeline::finalize() so stages can react to the dataset
     * dimensions set via Pipeline::setDims() after construction.
     * @param dims  {x, y, z} extents (z==1 → 2-D; y==z==1 → 1-D)
     */
    virtual void setDims(const std::array<size_t, 3>& dims) { (void)dims; }

    /**
     * Called once by Pipeline::finalize() after buffer-size propagation,
     * with this stage's estimated input size (bytes) and the pipeline pool.
     *
     * Implement this to pre-allocate persistent stage-internal scratch (e.g.
     * Huffman codebook/histogram buffers) via `pool->allocatePersistentDevice`
     * and `pool->allocatePersistentPinned` rather than via `cudaMalloc` directly.
     * Pre-allocating here makes PREALLOCATE mode semantically correct
     * (all memory committed at finalize time) and makes the stage footprint
     * visible via `pool->getPersistentDeviceBytes()` / `getPersistentPinnedBytes()`.
     *
     * Stages that also allow lazy allocation (e.g. for capacity-growth realloc in
     * execute()) should check whether `pool` was already used to allocate here and
     * skip the lazy path if so.
     *
     * Default: no-op.
     */
    virtual void onFinalize(size_t /*estimated_inlen*/, MemoryPool* /*pool*/) {}

    /**
     * Estimated persistent device memory this stage allocates outside the pool
     * (via `pool->allocatePersistentDevice`).  Used for total footprint reporting.
     * Default: 0.
     */
    virtual size_t estimateDeviceFootprintBytes(size_t /*inlen*/) const { return 0; }

    /**
     * Estimated persistent pinned-host memory this stage allocates outside the pool
     * (via `pool->allocatePersistentPinned`).  Used for total footprint reporting.
     * Default: 0.
     */
    virtual size_t estimatePinnedFootprintBytes(size_t /*inlen*/) const { return 0; }

    /**
     * Called after dag->execute() and stream sync, before compress() returns.
     * Use for D2H transfers that must not block mid-pipeline (e.g. Lorenzo's
     * outlier count readback).  The stream is already idle so a plain
     * cudaMemcpy is safe here.
     */
    virtual void postStreamSync(fz::stream_t stream) { (void)stream; }

    /** Maximum bytes this stage writes into its per-output FZM header slot. */
    virtual size_t getMaxHeaderSize(size_t output_index) const {
        (void)output_index;
        return 0;
    }

    /**
     * Whether this stage is safe inside a CUDA Graph capture.
     *
     * A stage is graph-compatible if execute() enqueues only device-side work
     * (kernel launches, cudaMemcpyAsync D2D/H2D) and makes no host-synchronous
     * calls.  Override and return false if execute() contains D2H copies or
     * dynamic decisions based on device data — the DAG will throw at
     * setCaptureMode(true) time rather than producing a broken graph.
     *
     * Default: true. Inverse-mode stages that do D2H reads (e.g. RZE inverse)
     * must return false.
     */
    virtual bool isGraphCompatible() const { return true; }

    /**
     * Inform the stage whether all of its forward outputs are pipeline outputs.
     *
     * A stage with deferred exact sizing may keep readback in postStreamSync when
     * terminal, but must publish an exact size before returning from execute when
     * a downstream stage will consume its output. Default: no placement-sensitive
     * behavior.
     */
    virtual void setTerminalOutput(bool terminal) { (void)terminal; }

    /**
     * Peak persistent scratch bytes this stage holds in the MemoryPool.
     *
     * Only count allocations that are drawn from the pool and kept alive across
     * execute() calls.  Transient scratch freed within execute() is already
     * captured by the pool's high-water mark and must not be included.
     * Used by CompressionDAG::computeTopoPoolSize() to size the release threshold.
     */
    virtual size_t estimateScratchBytes(const std::vector<size_t>& input_sizes) const {
        (void)input_sizes;
        return 0;
    }

    /**
     * Fusion contract: how this stage accesses its input, which decides whether
     * the fusion planner may fold it into a single kernel with its neighbours
     * (see include/stage/fusion.h and docs/codebase_notes.md CN-FUSE-PROOF).
     *
     * Default is `Unfusable` — a stage is only ever fused if it opts in by
     * overriding this. Stages whose fusability depends on configuration (e.g. a
     * quantizer is a pure Map only in linear mode) must reflect that here.
     * Forward-mode only; an inverse stage should report `Unfusable`.
     */
    virtual FusionSpec getFusionSpec() const { return {}; }

    /**
     * Fused-kernel identity: the device-op this stage maps to, plus its runtime
     * parameter bytes. The generic fused runner collects these across a fused
     * group (after priming) and hands the ordered op list to the codegen, so no
     * per-pipeline shape is hard-coded. Default (empty `op_name`) = not a fused
     * op. Must agree with `getFusionSpec()` (a stage returning a fusable spec but
     * no op cannot actually be composed). See include/stage/fusion.h and
     * docs/codebase_notes.md CN-NVRTC-FUSE.
     */
    virtual FusedOpDecl getFusedOp() const { return {}; }

    /**
     * Inverse-mode fusion role + geometry (valid only when isInverse()). The
     * decompress-DAG analogue of getFusionSpec(): a stage that can participate in
     * a fused inverse kernel overrides this to return a non-Unfusable spec.
     * Kept a SEPARATE surface from getFusionSpec() (rather than un-gating it) so
     * the forward fusion planner can never see an inverse stage. Default {} =
     * not inverse-fusable.
     */
    virtual FusionSpec getInverseFusionSpec() const { return {}; }

    /**
     * Inverse-mode device-op identity (valid only when isInverse()). Mirrors
     * getFusedOp() for the decompress DAG: op_name is the device policy the
     * inverse harness composes — the SAME policy class as the forward op, which
     * carries decode()/undelta()/invert() alongside cost()/pack()/delta()/apply().
     * The generic inverse matcher/runner read these declarations by role, so a new
     * warp predictor/coder that declares forward+inverse ops fuses in BOTH
     * directions with no matcher edits. Default {} = not an inverse fused op.
     */
    virtual FusedOpDecl getInverseFusedOp() const { return {}; }

    /**
     * Coder (Cooperative) inverse hook: the element count the archive
     * reconstructs to, so a generic inverse runner never downcasts the coder
     * stage for it. Default 0; the warp coder stage overrides it.
     */
    virtual size_t getFusedInverseElementCount() const { return 0; }

    /**
     * Quant (Map) inverse hook: the linear dequant step (2*abs_eb) the warp
     * inverse harness multiplies reconstructed codes by, so a generic inverse
     * runner never downcasts the quantizer stage for it. Linear-quant only —
     * the same generality ceiling as the forward warp path. Default 0; the
     * linear quantizer stage overrides it.
     */
    virtual double getFusedInverseDequantStep() const { return 0.0; }

    /**
     * Exact local encoded-size policy exposed to a directly connected adaptive
     * producer. This is an algorithmic semantic contract, not a fusion-cost
     * estimate: staged and fused paths must make the same decision from it.
     * Default = no oracle.
     */
    virtual EncodingOracleDecl getEncodingOracle() const { return {}; }

    /**
     * Bind an immediate downstream encoder's exact oracle during finalize().
     * Adaptive stages override this and return true only when type, unit size,
     * exactness, and additivity are compatible. Default rejects the contract.
     */
    virtual bool bindDownstreamEncodingOracle(const EncodingOracleDecl& /*decl*/) {
        return false;
    }

    /** Escaping outputs a generated fused op can produce beside its main port. */
    virtual std::vector<FusedAuxOutputDecl> getFusedAuxOutputs() const { return {}; }

    /**
     * Establish forward-computed state this stage's own INVERSE will read, for a
     * fused runner that bypasses forward `execute()`. Default no-op; a quantizer
     * overrides it to run its value-range scan. Called once per fused group member
     * before the fused kernel is generated (op params are read afterwards, so any
     * param derived from primed state is valid).
     */
    virtual void primeFusedForwardState(const FusedPrimeContext& /*ctx*/) {}

    /**
     * Tail-coder hook: a fused runner that produced this stage's archive without
     * calling `execute()` reports the archive size and the ORIGINAL (uncompressed)
     * input size, so the stage's inverse can size its output buffer. Without it a
     * variable-length coder's inverse falls back to the compressed size and its
     * decode overruns (see CN-CHUNK-WIRE). Distinct from any `setFusedResult`
     * overload to avoid colliding with unrelated `(size_t,size_t)` signatures.
     */
    virtual void setFusedArchiveResult(size_t /*archive_bytes*/, size_t /*orig_bytes*/) {}

    /// A fused inverse runner bypasses execute(); let its tail stage publish the
    /// reconstructed byte count for postStreamSync/output-size refinement.
    virtual void setFusedInverseResult(size_t /*output_bytes*/) {}

    /**
     * Side-output hook: a fused runner that filled one of this stage's escaping
     * output ports (a pipeline leaf, e.g. an outlier list) without calling
     * `execute()` reports how many BYTES it wrote to port `output_index`. The stage
     * updates whatever forward-computed state its `serializeHeader` depends on (e.g.
     * the quantizer's outlier count) so the archive matches the fused result. Default
     * no-op — single-output stages and stages the runner didn't feed ignore it.
     */
    virtual void setFusedSideOutput(int /*output_index*/, size_t /*bytes*/) {}
};

} // namespace fz
