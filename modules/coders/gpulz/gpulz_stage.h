#pragma once

/**
 * @file gpulz_stage.h
 * @brief GPULZ stage — GPU LZSS (LZ77 + flag-bit literal/match coding) lossless
 *        byte-stream compressor.
 *
 * Operates on a raw byte stream treated as `word_size`-byte words (1, 2, 4, or
 * 8). The stream is split into fixed-size chunks (`chunk_size`, default 2048
 * bytes); each chunk is compressed independently by a single CUDA thread
 * block that keeps the whole chunk resident in shared memory and searches a
 * sliding window (32 words) for repeated word sequences, exactly as in the
 * upstream GPULZ paper/reference implementation.
 *
 * Output stream layout:
 * @code
 *   [uint32_t: original byte count]
 *   [uint32_t: num_chunks]
 *   [ (uint32_t flag_size, uint32_t data_size) x n_chunks ]   // flag_size high bit -> chunk stored raw
 *   [ per-chunk payload: flag bytes, then compressed-data bytes (or raw bytes if flagged) ... ]
 * @endcode
 *
 * Serialized header (10 bytes):
 *   `[0..3]` chunk_size (uint32_t LE), `[4]` word_size (uint8_t),
 *   `[5..8]` cached_orig_bytes (uint32_t LE), `[9]` reserved.
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
 * GPU LZSS (GPULZ) coder stage.
 *
 * `setChunkSize(bytes)` — chunk size in bytes (default 2048; supported: 1024,
 * 2048, 4096 — must yield `chunk_size / word_size >= 128` and a power of two).
 * `setWordSize(bytes)`  — word granularity 1/2/4/8 (default 4, matching the
 * upstream reference's `uint32_t` default).
 *
 * @note **Prior work:** GPU kernels are a direct port of the compression and
 *       decompression kernels in `gpulz.cu` from **GPULZ**
 *       (Zhang, Tian, Di, Yu, Swany, Tao, Cappello — ICS '23; BSD-3-Clause-style
 *       license, see repository). Upstream:
 *       https://github.com/hpdps-group/ICS23-GPULZ — see `THIRD_PARTY.md`.
 *       The per-chunk container/offset-scan plumbing (raw-fallback flag,
 *       CUB exclusive scan for packing offsets, deferred tail-size readback)
 *       is FZGM's own, following the same pattern as `RREStage`/`RZEStage`.
 *
 * @note CUDA Graph capture is supported for compression only (the final
 *       output-size readback is deferred to `postStreamSync()`). The inverse
 *       path reads the stream header with blocking D2H copies before it can
 *       launch the decode kernel, so it is not graph-capturable.
 */
class GPULZStage : public Stage {
public:
    GPULZStage()
        : is_inverse_(false)
        , chunk_size_(2048)
        , word_size_(4)
        , actual_output_size_(0)
        , cached_orig_bytes_(0)
    {}

    ~GPULZStage() override;

    // ── Stage control ──────────────────────────────────────────────────────
    void setInverse(bool inv) override { is_inverse_ = inv; }
    bool isInverse() const override    { return is_inverse_; }

    bool isGraphCompatible() const override { return !is_inverse_; }

    void setChunkSize(size_t bytes) { chunk_size_ = static_cast<uint32_t>(bytes); }
    void setWordSize(size_t bytes)  { word_size_  = static_cast<uint8_t>(bytes);  }

    size_t getChunkSize()              const { return chunk_size_; }
    size_t getRequiredInputAlignment() const override { return chunk_size_; }
    int    getWordSize()               const { return static_cast<int>(word_size_); }
    uint32_t getCachedOrigBytes()      const { return cached_orig_bytes_; }

    // ── Execution ─────────────────────────────────────────────────────────
    void execute(
        fz::stream_t stream,
        MemoryPool* pool,
        const std::vector<void*>& inputs,
        const std::vector<void*>& outputs,
        const std::vector<size_t>& sizes
    ) override;
    void postStreamSync(fz::stream_t stream) override;

    // ── Metadata ──────────────────────────────────────────────────────────
    std::string getName() const override { return "GPULZ"; }
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
        // Forward: worst case = original data (every chunk falls back to raw
        // storage) + stream header (two uint32_t per chunk).
        const size_t n_bytes  = input_sizes.empty() ? 0 : input_sizes[0];
        const size_t n_chunks = (n_bytes + chunk_size_ - 1) / chunk_size_;
        const size_t hdr      = 4 + 4 + 8 * n_chunks;
        const size_t worst    = n_bytes + hdr;
        // postStreamSync() rounds the final size up to a 4-byte boundary and
        // zero-fills the pad; reserve that pad here too (see RREStage).
        return {(worst + 3) & ~size_t(3)};
    }

    std::unordered_map<std::string, size_t>
    getActualOutputSizesByName() const override;
    size_t getActualOutputSize(int index) const override;

    /**
     * Forward pass allocates six persistent pool arrays proportional to
     * n_chunks = ceil(input_bytes / chunk_size_):
     *   d_data_scratch_ : n_chunks * chunk_size_                (per-chunk worst-case compressed data)
     *   d_flag_scratch_ : n_chunks * (chunk_size_/word_size_/8) (per-chunk worst-case flag bits)
     *   d_flag_size_    : n_chunks * 4  (actual flag-array bytes per chunk)
     *   d_data_size_    : n_chunks * 4  (actual compressed-data bytes per chunk)
     *   d_clean_dev_    : n_chunks * 4  (raw-fallback-adjusted total size, scan input)
     *   d_dst_off_dev_  : n_chunks * 4  (exclusive prefix-sum packing offsets)
     */
    size_t estimateScratchBytes(
        const std::vector<size_t>& input_sizes
    ) const override {
        if (is_inverse_ || input_sizes.empty()) return 0;
        const size_t in_bytes       = input_sizes[0];
        const size_t n_chunks       = (in_bytes + chunk_size_ - 1) / chunk_size_;
        const size_t block_elems    = chunk_size_ / word_size_;
        const size_t flag_bytes_max = (block_elems + 7) / 8;
        return n_chunks * (static_cast<size_t>(chunk_size_) + flag_bytes_max + 4 * sizeof(uint32_t));
    }

    uint16_t getStageTypeId() const override {
        return static_cast<uint16_t>(StageType::GPULZ);
    }

    uint8_t getOutputDataType(size_t) const override {
        return static_cast<uint8_t>(DataType::UINT8);
    }
    uint8_t getInputDataType(size_t) const override {
        return static_cast<uint8_t>(DataType::UNKNOWN);
    }

    // ── Serialization ─────────────────────────────────────────────────────
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
        saved_chunk_size_        = chunk_size_;
        saved_word_size_         = word_size_;
        saved_cached_orig_bytes_ = cached_orig_bytes_;
    }

    void restoreState() override {
        chunk_size_        = saved_chunk_size_;
        word_size_         = saved_word_size_;
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

    // ── Persistent forward scratch buffers ─────────────────────────────────
    uint8_t*  d_data_scratch_ = nullptr;
    uint8_t*  d_flag_scratch_ = nullptr;
    uint32_t* d_flag_size_    = nullptr;
    uint32_t* d_data_size_    = nullptr;
    uint32_t* d_clean_dev_    = nullptr;
    uint32_t* d_dst_off_dev_  = nullptr;
    mutable bool         tail_readback_pending_ = false;
    mutable fz::stream_t tail_readback_stream_  = nullptr;
    mutable uint32_t     tail_last_index_       = 0;
    mutable uint8_t*     tail_output_ptr_       = nullptr;
    size_t      scratch_capacity_ = 0;
    MemoryPool* scratch_pool_owner_ = nullptr;
    bool        scratch_from_pool_ = false;
};

} // namespace fz
