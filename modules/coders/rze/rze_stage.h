#pragma once

/**
 * @file rze_stage.h
 * @brief Recursive Zero-byte Elimination stage — lossless byte-stream compressor.
 *
 * Operates on raw byte streams (e.g. BitshuffleStage output). Each chunk is
 * processed with up to 4 recursive levels:
 * - Level 1 (ZE): compact non-zero bytes; emit N/8-byte bitmap.
 * - Levels 2–4 (RE): compact non-repeated bytes of the previous bitmap.
 *
 * Output stream layout:
 * @code
 *   [uint32_t: original byte count]
 *   [uint32_t: num_chunks]
 *   [uint32_t × n_chunks: per-chunk compressed sizes (high bit → stored raw)]
 *   [compressed chunk data...]
 * @endcode
 *
 * Serialized header (9 bytes):
 *   `[0..3]` chunk_size (uint32_t LE), `[4]` levels (uint8_t),
 *   `[5..8]` cached_orig_bytes (uint32_t LE).
 */

#include "stage/stage.h"
#include "fzm_format.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace fz {

/**
 * Recursive Zero-byte Elimination stage.
 *
 * `setChunkSize(bytes)` — chunk size (default 16384; must be a multiple of 4096).
 * `setLevels(n)` — recursion depth 1–4 (default 4).
 *
 * @note CUDA Graph capture is supported for compression only. The inverse path
 *       requires two blocking D2H copies to read the stream header before the
 *       decode kernel can be launched.
 */
class RZEStage : public Stage {
public:
    RZEStage()
        : is_inverse_(false)
        , chunk_size_(16384)
        , levels_(4)
        , actual_output_size_(0)
        , cached_orig_bytes_(0)
        // Persistent scratch buffers — allocated on first use, reused thereafter.
        // Eliminates the blocking cudaMalloc/cudaFree pair that appeared in every
        // execute() call and was clearly visible as the two cudaMalloc events in nsys.
        , d_scratch_(nullptr)
        , d_sizes_dev_(nullptr)
        , d_clean_dev_(nullptr)
        , d_dst_off_dev_(nullptr)
        , d_inv_in_off_(nullptr)
        , d_inv_comp_sz_(nullptr)
        , d_inv_out_off_(nullptr)
        , d_inv_orig_sz_(nullptr)
        , scratch_capacity_(0)     // # chunks the current scratch allocation can hold
        , inv_capacity_(0)         // # chunks the current inverse table allocation holds
    {}

    ~RZEStage() override;

    // ── Stage control ──────────────────────────────────────────────────────
    void setInverse(bool inv) override { is_inverse_ = inv; }
    bool isInverse() const override    { return is_inverse_; }

    /**
     * CUDA Graph capture is supported for compression (forward pass) only.
     *
     * The inverse path reads the stream header (orig_bytes, per-chunk sizes)
     * with two blocking D2H cudaMemcpy calls before it can compute per-chunk
     * decode offsets and launch the decode kernel.  These calls prevent the
     * inverse path from being recorded into a CUDA Graph.
     *
     * This is intentional by design, not a fixable limitation: graph-compatible
     * decompression would only help a "repeatedly decompress the same compressed
     * buffer" workflow, which has no practical use case.  The compression path
     * (new data every iteration) is where graph capture provides real value.
     */
    bool isGraphCompatible() const override { return !is_inverse_; }

    void setChunkSize(size_t bytes) { chunk_size_ = static_cast<uint32_t>(bytes); }
    void setLevels(int n)           { levels_     = static_cast<uint8_t>(n);      }

    size_t getChunkSize()       const { return chunk_size_; }
    size_t getRequiredInputAlignment() const override { return chunk_size_; }
    int    getLevels()          const { return static_cast<int>(levels_); }
    uint32_t getCachedOrigBytes() const { return cached_orig_bytes_; }

    // ── Execution ──────────────────────────────────────────────────────────
    void execute(
        cudaStream_t stream,
        MemoryPool* pool,
        const std::vector<void*>& inputs,
        const std::vector<void*>& outputs,
        const std::vector<size_t>& sizes
    ) override;
    void postStreamSync(cudaStream_t stream) override;

    // ── Metadata ───────────────────────────────────────────────────────────
    std::string getName() const override { return "RZE"; }
    size_t getNumInputs()  const override { return 1; }
    size_t getNumOutputs() const override { return 1; }

    std::vector<size_t> estimateOutputSizes(
        const std::vector<size_t>& input_sizes
    ) const override {
        if (is_inverse_) {
            // The original byte count is stored in the first 4 bytes of the
            // compressed stream and cached here after the forward pass (or
            // restored via deserializeHeader for cold decompression).
            if (cached_orig_bytes_ > 0)
                return {static_cast<size_t>(cached_orig_bytes_)};
            // Fallback: should not normally be reached; return compressed
            // size as a lower bound (will likely trigger a buffer overwrite
            // warning — indicates a missing forward pass or header restore).
            return {input_sizes.empty() ? 0 : input_sizes[0]};
        }
        // Forward: worst case = original data + stream header.
        // Header = 4 (orig_bytes) + 4 (num_chunks) + 4*n_chunks.
        const size_t n_bytes  = input_sizes.empty() ? 0 : input_sizes[0];
        const size_t n_chunks = (n_bytes + chunk_size_ - 1) / chunk_size_;
        const size_t hdr      = 4 + 4 + 4 * n_chunks;
        return {n_bytes + hdr};
    }

    std::unordered_map<std::string, size_t>
    getActualOutputSizesByName() const override;
    size_t getActualOutputSize(int index) const override;

    /**
     * Forward pass allocates four persistent pool arrays proportional to
     * n_chunks = ceil(input_bytes / chunk_size_):
     *   d_scratch_    : n_chunks * chunk_size_   (per-chunk worst-case output)
     *   d_sizes_dev_  : n_chunks * 4             (raw compressed sizes)
     *   d_clean_dev_  : n_chunks * 4             (flag-stripped sizes)
     *   d_dst_off_dev_: n_chunks * 4             (exclusive prefix-sum offsets)
     *
     * Inverse path scratch is transient (allocated and freed within execute),
     * so it is not reported here.
     */
    size_t estimateScratchBytes(
        const std::vector<size_t>& input_sizes
    ) const override {
        if (is_inverse_ || input_sizes.empty()) return 0;
        const size_t in_bytes = input_sizes[0];
        const size_t n_chunks = (in_bytes + chunk_size_ - 1) / chunk_size_;
        return n_chunks * (static_cast<size_t>(chunk_size_) + 3 * sizeof(uint32_t));
    }

    uint16_t getStageTypeId() const override {
        return static_cast<uint16_t>(StageType::RZE);
    }

    uint8_t getOutputDataType(size_t) const override {
        return static_cast<uint8_t>(DataType::UINT8);
    }

    // ── Serialization ──────────────────────────────────────────────────────
    // Header layout (9 bytes):
    //   [0..3]  chunk_size         (uint32_t LE)
    //   [4]     levels             (uint8_t)
    //   [5..8]  cached_orig_bytes_ (uint32_t LE)  — original uncompressed size,
    //                              used by the inverse estimateOutputSizes() to
    //                              pre-allocate the correct output buffer.
    size_t serializeHeader(
        size_t output_index, uint8_t* buf, size_t max_size
    ) const override {
        (void)output_index;
        if (max_size < 9) return 0;
        std::memcpy(buf,     &chunk_size_,        sizeof(uint32_t));
        buf[4] = levels_;
        std::memcpy(buf + 5, &cached_orig_bytes_, sizeof(uint32_t));
        return 9;
    }

    void deserializeHeader(const uint8_t* buf, size_t size) override {
        if (size >= 4) std::memcpy(&chunk_size_,        buf,     sizeof(uint32_t));
        if (size >= 5) levels_ = buf[4];
        if (size >= 9) std::memcpy(&cached_orig_bytes_, buf + 5, sizeof(uint32_t));
    }

    size_t getMaxHeaderSize(size_t) const override { return 9; }

    void saveState() override {
        saved_chunk_size_ = chunk_size_;
        saved_levels_ = levels_;
        saved_cached_orig_bytes_ = cached_orig_bytes_;
    }

    void restoreState() override {
        chunk_size_ = saved_chunk_size_;
        levels_ = saved_levels_;
        cached_orig_bytes_ = saved_cached_orig_bytes_;
    }

private:
    bool     is_inverse_;
    uint32_t chunk_size_;
    uint32_t saved_chunk_size_ = 0;
    uint8_t  levels_;
    uint8_t  saved_levels_ = 0;
    size_t   actual_output_size_;
    // Cached original (uncompressed) byte count.  Set by the forward execute()
    // and persisted in the serialized header so that the inverse
    // estimateOutputSizes() can return the right buffer size even when the
    // inverse stage is constructed cold from a file-based pipeline.
    uint32_t cached_orig_bytes_ = 0;
    uint32_t saved_cached_orig_bytes_ = 0;

    // ── Persistent scratch buffers ──────────────────────────────────────────
    // Forward path:
    //   d_scratch_    : per-chunk worst-case output (n_chunks * chunk_size bytes)
    //   d_sizes_dev_  : raw compressed sizes from rzeEncodeKernel (with flag bit)
    //   d_clean_dev_  : compressed sizes with flag stripped (for pack offsets)
    //   d_dst_off_dev_: exclusive-prefix-sum of clean sizes + header offset
    // Inverse path:
    //   d_inv_{in_off,comp_sz,out_off,orig_sz}_: per-chunk decode tables
    // All are allocated once (or grown on demand) and freed in the destructor.
    uint8_t*  d_scratch_;
    uint32_t* d_sizes_dev_;
    uint32_t* d_clean_dev_;
    uint32_t* d_dst_off_dev_;
    uint32_t* d_inv_in_off_;
    uint32_t* d_inv_comp_sz_;
    uint32_t* d_inv_out_off_;
    uint32_t* d_inv_orig_sz_;
    mutable bool         tail_readback_pending_ = false;
    mutable cudaStream_t tail_readback_stream_ = nullptr;
    mutable uint32_t     tail_last_index_ = 0;
    // Output pointer saved by forward execute() so postStreamSync() can zero
    // the trailing alignment padding (0–3 bytes at actual_output_size_ - total_out).
    mutable uint8_t*     tail_output_ptr_ = nullptr;
    size_t    scratch_capacity_;  // # chunks forward scratch can hold
    size_t    inv_capacity_;      // # chunks inverse tables can hold
    MemoryPool* scratch_pool_owner_ = nullptr;
    MemoryPool* inv_pool_owner_ = nullptr;
    bool        scratch_from_pool_ = false;
    bool        inv_from_pool_ = false;
};

} // namespace fz
