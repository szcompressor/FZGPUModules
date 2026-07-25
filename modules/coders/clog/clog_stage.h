#pragma once

/**
 * @file clog_stage.h
 * @brief Compressed-Logarithm (adaptive bit-width) coding stage — lossless byte-stream compressor.
 *
 * Standalone port of the LC framework `CLOG` component. Splits each chunk
 * into a fixed **32 subchunks**; each subchunk finds its own max value and
 * bit-packs every element in it to the minimum bit-width needed to represent
 * that max losslessly (no element exceeds the subchunk's own max, so
 * truncating every element to that width loses nothing). Unlike
 * `RREStage`/`RAREStage`/`RZEStage`/`RAZEStage`, there is no auxiliary
 * bitmap and no per-element full/dropped decision — every element in a
 * subchunk is packed at the same fixed width. `T` must be unsigned
 * (`uint8/16/32/64` only).
 *
 * Output stream layout (identical container to RREStage/RZEStage):
 * @code
 *   [uint32_t: original byte count]
 *   [uint32_t: num_chunks]
 *   [uint32_t × n_chunks: per-chunk compressed sizes (high bit → stored raw)]
 *   [compressed chunk data...]
 * @endcode
 *
 * Serialized header (9 bytes):
 *   `[0..3]` chunk_size (uint32_t LE), `[4]` word_size (uint8_t),
 *   `[5..8]` cached_orig_bytes (uint32_t LE).
 */

#include "stage/stage.h"
#include "fzm_format.h"
#include "backend/types.h"
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace fz {

/**
 * Compressed-Logarithm adaptive bit-width coding stage.
 *
 * `setChunkSize(bytes)` — chunk size (default 16384; one of 4096/8192/16384).
 * `setWordSize(bytes)`  — word granularity 1/2/4/8 (default 1), unsigned only.
 *
 * @note **Prior work:** GPU kernels are a faithful port of `d_CLOG.h` from
 *       the LC framework (Burtscher et al., BSD-3-Clause), sharing the
 *       encode/decode device code with `HCLOGStage` via
 *       `d_CLOGencode`/`d_CLOGdecode<T, CLogMode>` in
 *       `modules/coders/lc_common/lc_clog_components.cuh`. See
 *       `THIRD_PARTY.md`.
 *
 * @note CUDA Graph capture is supported for compression only. The inverse
 *       path reads the stream header with blocking D2H copies before it can
 *       launch the decode kernel (same constraint as RREStage/RZEStage).
 */
class CLOGStage : public Stage {
public:
    CLOGStage()
        : is_inverse_(false)
        , chunk_size_(16384)
        , word_size_(1)
        , actual_output_size_(0)
        , cached_orig_bytes_(0)
        , d_scratch_(nullptr)
        , d_sizes_dev_(nullptr)
        , d_clean_dev_(nullptr)
        , d_dst_off_dev_(nullptr)
        , scratch_capacity_(0)
    {}

    ~CLOGStage() override;

    // ── Stage control ──────────────────────────────────────────────────────
    void setInverse(bool inv) override { is_inverse_ = inv; }
    bool isInverse() const override    { return is_inverse_; }

    /// Forward (compress) is graph-capturable; inverse reads the stream header
    /// with blocking D2H copies and therefore is not. See RREStage for the
    /// rationale.
    bool isGraphCompatible() const override { return !is_inverse_; }

    void setChunkSize(size_t bytes) { chunk_size_ = static_cast<uint32_t>(bytes); }
    void setWordSize(size_t bytes)  { word_size_  = static_cast<uint8_t>(bytes);  }

    size_t getChunkSize()       const { return chunk_size_; }
    size_t getRequiredInputAlignment() const override { return chunk_size_; }
    int    getWordSize()        const { return static_cast<int>(word_size_); }
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
    std::string getName() const override { return "CLOG"; }
    size_t getNumInputs()  const override { return 1; }
    size_t getNumOutputs() const override { return 1; }

    std::vector<size_t> estimateOutputSizes(
        const std::vector<size_t>& input_sizes
    ) const override {
        if (is_inverse_) {
            if (cached_orig_bytes_ > 0)
                return {static_cast<size_t>(cached_orig_bytes_)};
            return {input_sizes.empty() ? 0 : input_sizes[0]};
        }
        // Forward: worst case = original data + stream header.
        const size_t n_bytes  = input_sizes.empty() ? 0 : input_sizes[0];
        const size_t n_chunks = (n_bytes + chunk_size_ - 1) / chunk_size_;
        const size_t hdr      = 4 + 4 + 4 * n_chunks;
        // postStreamSync()/getActualOutputSizesByName() always round the final
        // size up to a 4-byte boundary and zero-fill the pad, even when the
        // real total isn't already aligned (e.g. a partial final chunk stored
        // raw at a byte count that isn't a multiple of 4) -- reserve that pad
        // here too, or the caller's allocation is up to 3 bytes short and the
        // memset in postStreamSync writes out of bounds.
        const size_t worst    = n_bytes + hdr;
        return {(worst + 3) & ~size_t(3)};
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
        return static_cast<uint16_t>(StageType::CLOG);
    }

    uint8_t getOutputDataType(size_t) const override {
        return static_cast<uint8_t>(DataType::UINT8);
    }

    // ── Serialization ──────────────────────────────────────────────────────
    size_t serializeHeader(
        size_t output_index, uint8_t* buf, size_t max_size
    ) const override {
        (void)output_index;
        if (max_size < 9) return 0;
        std::memcpy(buf,     &chunk_size_,        sizeof(uint32_t));
        buf[4] = word_size_;
        std::memcpy(buf + 5, &cached_orig_bytes_, sizeof(uint32_t));
        return 9;
    }

    void deserializeHeader(const uint8_t* buf, size_t size) override {
        if (size >= 4) std::memcpy(&chunk_size_,        buf,     sizeof(uint32_t));
        if (size >= 5) word_size_ = buf[4];
        if (size >= 9) std::memcpy(&cached_orig_bytes_, buf + 5, sizeof(uint32_t));
    }

    size_t getMaxHeaderSize(size_t) const override { return 9; }

    void saveState() override {
        saved_chunk_size_ = chunk_size_;
        saved_word_size_ = word_size_;
        saved_cached_orig_bytes_ = cached_orig_bytes_;
    }

    void restoreState() override {
        chunk_size_ = saved_chunk_size_;
        word_size_ = saved_word_size_;
        cached_orig_bytes_ = saved_cached_orig_bytes_;
    }

private:
    bool     is_inverse_;
    uint32_t chunk_size_;
    uint32_t saved_chunk_size_ = 0;
    uint8_t  word_size_;
    uint8_t  saved_word_size_ = 0;
    size_t   actual_output_size_;
    uint32_t cached_orig_bytes_ = 0;
    uint32_t saved_cached_orig_bytes_ = 0;

    // ── Persistent forward scratch buffers ───────────────────────────────────
    uint8_t*  d_scratch_;
    uint32_t* d_sizes_dev_;
    uint32_t* d_clean_dev_;
    uint32_t* d_dst_off_dev_;
    mutable bool         tail_readback_pending_ = false;
    mutable cudaStream_t tail_readback_stream_ = nullptr;
    mutable uint32_t     tail_last_index_ = 0;
    mutable uint8_t*     tail_output_ptr_ = nullptr;
    size_t    scratch_capacity_;
    MemoryPool* scratch_pool_owner_ = nullptr;
    bool        scratch_from_pool_ = false;
};

} // namespace fz
