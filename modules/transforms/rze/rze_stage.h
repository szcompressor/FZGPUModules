#pragma once

/**
 * modules/transforms/rze/rze_stage.h
 *
 * RZEStage — Recursive Zero-byte Elimination stage.
 *
 * Operates on a raw byte stream (e.g. the output of BitshuffleStage).
 * The forward pass compresses the stream by eliminating zero bytes and
 * recursively compressing the resulting bitmaps.  The inverse pass
 * decompresses back to the original byte stream.
 *
 * Algorithm (T=uint8_t, one pass per chunk):
 *   Level 1 (ZE):  compact non-zero bytes from the input N-byte chunk;
 *                  produce a bitmap of N/8 bytes (bm1).
 *   Level 2 (RE):  compact non-repeated bytes of bm1 (N/8 bytes);
 *                  produce a bitmap of N/64 bytes (bm2).
 *   Level 3 (RE):  compact non-repeated bytes of bm2;
 *                  produce a bitmap of N/512 bytes (bm3).
 *   Level 4 (RE):  compact non-repeated bytes of bm3;
 *                  produce a bitmap of N/4096 bytes (bm4, stored raw).
 *
 * Compressed chunk layout (written smallest-address first):
 *
 *   [non-zero data bytes from level-1 ZE]
 *   [non-repeated bytes from level-2 RE on bm1]
 *   [non-repeated bytes from level-3 RE on bm2]
 *   [non-repeated bytes from level-4 RE on bm3]  (if levels >= 4)
 *   [raw bm4  (N/4096 bytes, 4 bytes for N=16384)] (if levels >= 4)
 *   [original chunk size, 2 bytes, little-endian]
 *
 * For the "all-zeros" case the chunk is stored as just the 2-byte size tag
 * with no data or bitmaps.  For incompressible chunks the original data is
 * stored and the high bit of the per-chunk header field is set.
 *
 * Stream layout (output buffer):
 *
 *   [uint32_t: original total byte count]
 *   [uint32_t: number of chunks]
 *   [uint32_t × n_chunks: per-chunk compressed sizes (high bit set → uncompressed)]
 *   [compressed chunk 0 bytes]
 *   [compressed chunk 1 bytes]
 *   ...
 *
 * Configuration:
 *   setChunkSize(bytes)  — chunk size in bytes, must be a power of 8 ≥ 64 and
 *                          a multiple of 4096 × blockDim (default 16384).
 *                          Currently only 16384 is tested / supported.
 *   setLevels(n)         — recursion depth 1–4 (default 4)
 *   setInverse(bool)     — decompress instead of compress
 *
 * Serialized header: 9 bytes
 *   [0..3] chunk_size         (uint32_t LE)
 *   [4]    levels             (uint8_t)
 *   [5..8] cached_orig_bytes  (uint32_t LE) — original uncompressed size
 *
 * Stage type ID: StageType::RZE (18)
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

    ~RZEStage() {
        cudaFree(d_scratch_);
        cudaFree(d_sizes_dev_);
        cudaFree(d_clean_dev_);
        cudaFree(d_dst_off_dev_);
        cudaFree(d_inv_in_off_);
        cudaFree(d_inv_comp_sz_);
        cudaFree(d_inv_out_off_);
        cudaFree(d_inv_orig_sz_);
    }

    // ── Stage control ──────────────────────────────────────────────────────
    void setInverse(bool inv) override { is_inverse_ = inv; }
    bool isInverse() const override    { return is_inverse_; }

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
    getActualOutputSizesByName() const override {
        return {{"output", actual_output_size_}};
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
    size_t    scratch_capacity_;  // # chunks forward scratch can hold
    size_t    inv_capacity_;      // # chunks inverse tables can hold
};

} // namespace fz
