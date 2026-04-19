#pragma once

#include "fzm_format.h"
#include <array>
#include <cuda_runtime.h>
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
 * @note Thread Safety: Stage instances are not thread-safe. Each pipeline
 * (and its stages) must be used from a single host thread.
 */
class Stage {
public:
    virtual ~Stage() = default;

    /** Execute the stage. Inputs, outputs, and sizes are device pointers/bytes. */
    virtual void execute(
        cudaStream_t stream,
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
     * checking — byte-transparent stages (Bitshuffle, RZE) and mock stages
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
     * Called once by Pipeline::finalize() so stages can react to the dataset
     * dimensions set via Pipeline::setDims() after construction.
     * @param dims  {x, y, z} extents (z==1 → 2-D; y==z==1 → 1-D)
     */
    virtual void setDims(const std::array<size_t, 3>& dims) { (void)dims; }

    /**
     * Called after dag->execute() and stream sync, before compress() returns.
     * Use for D2H transfers that must not block mid-pipeline (e.g. Lorenzo's
     * outlier count readback).  The stream is already idle so a plain
     * cudaMemcpy is safe here.
     */
    virtual void postStreamSync(cudaStream_t stream) { (void)stream; }

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
};

} // namespace fz
