#pragma once

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
 * Base class for compression/decompression stages
 * 
 * Each stage represents a single transformation in the pipeline
 * (e.g. Lorenzo predictor, RLE encoder, bitpacking)
 */
class Stage {
public:
    virtual ~Stage() = default;
    
    /**
     * Execute the stage
     * 
     * @param stream CUDA stream for execution
     * @param inputs Array of input device pointers
     * @param outputs Array of output device pointers
     * @param sizes Array of buffer sizes (input/output)
     */
    virtual void execute(
        cudaStream_t stream,
        MemoryPool* pool,
        const std::vector<void*>& inputs,
        const std::vector<void*>& outputs,
        const std::vector<size_t>& sizes
    ) = 0;
    
    /**
     * Get stage name for debugging
     */
    virtual std::string getName() const = 0;
    
    /**
     * Get expected number of input buffers
     */
    virtual size_t getNumInputs() const = 0;
    
    /**
     * Get expected number of output buffers
     */
    virtual size_t getNumOutputs() const = 0;
    
    /**
     * Get names of all outputs (in order)
     * 
     * For stages with multiple outputs (e.g., Lorenzo predictor):
     * - Output 0: "codes"
     * - Output 1: "outliers"
     * 
     * Default: Single unnamed output
     */
    virtual std::vector<std::string> getOutputNames() const {
        return {"output"};  // Default single output
    }
    
    /**
     * Get index of named output
     * 
     * @param name Output name
     * @return Output index, or -1 if not found
     */
    int getOutputIndex(const std::string& name) const {
        auto names = getOutputNames();
        for (size_t i = 0; i < names.size(); i++) {
            if (names[i] == name) {
                return static_cast<int>(i);
            }
        }
        return -1;
    }
    
    /**
     * Estimate output size given input size
     * Used for buffer allocation planning in PREALLOCATE mode
     * 
     * @param input_sizes Sizes of input buffers
     * @return Estimated sizes of output buffers
     * 
     * Examples:
     * - Lorenzo: 120% of input (codes + ~20% outliers)
     * - Difference: 100% of input (same size)
     * - Bitpacking: Conservative estimate or soft-run needed
     */
    virtual std::vector<size_t> estimateOutputSizes(
        const std::vector<size_t>& input_sizes
    ) const = 0;
    
    /**
     * Get actual output sizes after execution (by name)
     * Used for dynamic buffer sizing in MINIMAL/PIPELINE modes
     * 
     * @return Map of output_name -> actual size written
     */
    virtual std::unordered_map<std::string, size_t> getActualOutputSizesByName() const = 0;
    
    // ===== Decompression Support =====
    
    /**
     * Set inverse mode for decompression
     * 
     * When true, the stage performs the inverse operation:
     * - Lorenzo: Reconstruction instead of prediction
     * - Difference: Cumulative sum instead of differencing
     * - RLE: Expansion instead of encoding
     * 
     * This affects getNumInputs() and getNumOutputs() for stages
     * with asymmetric input/output counts.
     * 
     * @param inverse True for decompression, false for compression
     */
    virtual void setInverse(bool inverse) {
        (void)inverse;
        // Stages that support bidirectional operation should override this
    }
    
    /**
     * Check if stage is in inverse mode
     * 
     * @return True if stage is configured for decompression
     */
    virtual bool isInverse() const {
        return false;  // Default: compression mode
    }
    
    /**
     * Get stage type ID for file format
     * Used to identify which stage produced a buffer during decompression
     * 
     * @return Stage type enum from fzm_format.h
     */
    virtual uint16_t getStageTypeId() const = 0;
    
    /**
     * Get data type for a specific output
     * 
     * @param output_index Index of output buffer
     * @return Data type enum from fzm_format.h
     */
    virtual uint8_t getOutputDataType(size_t output_index) const = 0;
    
    /**
     * Serialize stage-specific configuration for decompression
     * 
     * Each stage defines its own config structure (e.g., LorenzoConfig in lorenzo.h)
     * and serializes it into the provided buffer. The DAG/Pipeline doesn't need to
     * know the specific structure - it just passes the buffer to serializeHeader().
     * 
     * Architecture:
     * - Stage defines XxxConfig struct in its own header file
     * - Stage implements serializeHeader() to write config to buffer
     * - Config written into FZMBufferEntry.stage_config[128] in the file header
     * 
     * Examples:
     * - Lorenzo: Writes LorenzoConfig (error_bound, quant_radius, outlier_count)
     * - Difference: Writes DifferenceConfig (element_type)
     * 
     * @param output_index Which output to write header for (stages may have different headers per output)
     * @param header_buffer Host buffer to write config data to (size >= 128 bytes)
     * @param max_size Maximum size available in header_buffer (typically 128)
     * @return Number of bytes written (must be <= 128)
     * 
     * Note: Return 0 if stage has no additional config data (default)
     */
    virtual size_t serializeHeader(size_t output_index, uint8_t* header_buffer, size_t max_size) const {
        (void)output_index;
        (void)header_buffer;
        (void)max_size;
        return 0;  // Default: no stage-specific header
    }
    
    /**
     * Deserialize stage-specific configuration from buffer
     * 
     * Called during decompression to restore stage configuration from file header.
     * Each stage reads its own XxxConfig structure and updates internal state.
     * 
     * @param header_buffer Buffer containing serialized config (from FZMBufferEntry.stage_config)
     * @param size Size of config data (from FZMBufferEntry.config_size)
     * 
     * Note: Default no-op if stage has no config to read
     */
    virtual void deserializeHeader(const uint8_t* header_buffer, size_t size) {
        (void)header_buffer;
        (void)size;
        // Default: no config to deserialize
    }

    /**
     * Called once by Pipeline::finalize() after all stages are connected and
     * before the first compress/decompress execution.  Gives stages a chance
     * to update their internal dimensionality — useful when setDims() is
     * called on the Pipeline after stages are already constructed.
     *
     * The default implementation is a no-op.  Stages that depend on spatial
     * dimensions (e.g. LorenzoStage) should override this and update their
     * internal config so the correct kernel variant is selected at execute().
     *
     * @param dims  {x, y, z} extents of the dataset (z==1 → 2-D; y==z==1 → 1-D)
     */
    virtual void setDims(const std::array<size_t, 3>& dims) { (void)dims; }

    /**
     * Called once by Pipeline::compress() after dag->execute() and the stream
     * has been fully synchronized.
     *
     * Stages that need to transfer device-side results back to the host
     * (e.g. Lorenzo's actual outlier count) should do so here rather than
     * blocking mid-pipeline inside execute(). The stream is already idle when
     * this is called, so a plain cudaMemcpy (no async) is safe and cheap.
     *
     * Default: no-op.
     */
    virtual void postStreamSync(cudaStream_t stream) { (void)stream; }

    /**
     * Estimate maximum header size for buffer allocation
     * 
     * @param output_index Which output to estimate for
     * @return Maximum bytes needed for this stage's header
     */
    virtual size_t getMaxHeaderSize(size_t output_index) const {
        (void)output_index;
        return 0;  // Default: no header
    }
};

} // namespace fz
