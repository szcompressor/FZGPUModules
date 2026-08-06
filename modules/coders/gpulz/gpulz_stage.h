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
 *       (Zhang, Tian, Di, Yu, Swany, Tao, Cappello — ICS '23; upstream
 *       repository declares no explicit license, see `THIRD_PARTY.md`).
 *       Upstream: https://github.com/hpdps-group/ICS23-GPULZ.
 *       The per-chunk container/offset-scan plumbing (raw-fallback flag,
 *       CUB exclusive scan for packing offsets, deferred tail-size readback)
 *       is FZGM's own, following the same pattern as `RREStage`/`RZEStage`.
 *       The all-zero-chunk fast path (skip encode entirely for chunks that
 *       are entirely zero) is adapted from the "sparse" GPULZ variant in
 *       `boyuanzhang62/AIZ_VLDB26` (`test/gpulz.cuh`'s `notEmptyFlagArr`).
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

    /**
     * Match-search effort, 0 or 1 (default 1). Encode-side only — the stream
     * format is identical either way, so a stream produced at one level
     * decodes the same as at the other and the level is not serialized.
     *
     *  0 — exact longest match over the 32-element near window only.
     *  1 — additionally consults a hashed table of two-word keys for
     *      long-range candidates (offsets up to 255).
     *
     * Measured on an H100 over 24.7 MB of Lorenzo-quantized `CLDHGH`
     * residuals at chunk_size=2048, word_size=4: level 0 gives 170 GB/s at
     * 4.36x, level 1 gives 126 GB/s at 5.13x.
     */
    void setMatchLevel(int level) { match_level_ = static_cast<uint8_t>(level); }
    int  getMatchLevel() const    { return static_cast<int>(match_level_); }

    /**
     * Split mode (default off) — emit the compressed stream as four separate
     * output ports instead of one interleaved stream:
     *
     *   `literals`  the literal words, back to back (raw-fallback chunks land
     *               here too, since such a chunk is by definition all literal)
     *   `lengths`   one match-length byte per match token
     *   `offsets`   one match-offset byte per match token
     *   `meta`      stream header + per-chunk size table + the flag bitmaps
     *
     * This is the Zstandard split (literals separate from sequences), for the
     * same reason: the parts have very different symbol distributions, and
     * interleaving them into one byte stream raises the entropy a downstream
     * coder sees. Measured across six SDRB fields, coding the four ports
     * separately beats the single-stream form by 23-43% compression ratio.
     *
     * The `literals` port keeps the data's natural word alphabet, so it should
     * be fed to a symbol-width-matched coder (`HuffmanStage<uint16_t>` for
     * uint16 quant codes) rather than a byte coder -- that alphabet effect is
     * the larger half of the gain.
     *
     * Every port must be entropy coded and all four re-merged: unlike the
     * single-stream form, which codes the whole payload by construction, a
     * split leaks any byte left out. Both the raw-fallback chunks and the
     * per-chunk size table are folded into ports above for exactly that
     * reason (leaving them out cost 28% and 67% respectively in testing).
     */
    void setSplitMode(bool on) { split_mode_ = on; }
    bool getSplitMode() const  { return split_mode_; }

    size_t getChunkSize()              const { return chunk_size_; }

    /**
     * Deliberately 1, not `chunk_size`: this stage zero-pads its own tail chunk.
     *
     * Requesting pipeline-level alignment does not actually work for a coder
     * sitting behind a width-changing stage. `Pipeline::finalize()` pads the
     * *pipeline input* to the LCM of every stage's alignment, but LorenzoQuant
     * turns float32 into uint16 codes, so a 2048-aligned input arrives here as
     * half as many bytes and need not be aligned at all. Worse, forcing the
     * pipeline input up to a chunk multiple grows the upstream stage's output
     * past its own estimate and trips the buffer-overwrite check. Padding
     * internally is both simpler and correct for any upstream wiring.
     */
    size_t getRequiredInputAlignment() const override { return 1; }
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
    size_t getNumInputs()  const override {
        return (is_inverse_ && split_mode_) ? 4 : 1;
    }
    size_t getNumOutputs() const override {
        return (!is_inverse_ && split_mode_) ? 4 : 1;
    }

    std::vector<std::string> getOutputNames() const override {
        if (!is_inverse_ && split_mode_)
            return {"literals", "lengths", "offsets", "meta"};
        return {"output"};
    }

    std::vector<size_t> estimateOutputSizes(
        const std::vector<size_t>& input_sizes
    ) const override {
        if (is_inverse_) {
            // Capacity must cover the padded extent the decode kernel writes,
            // even though execute() reports the unpadded size afterwards.
            if (cached_orig_bytes_ > 0)
                return {static_cast<size_t>(cached_orig_bytes_)};
            return {input_sizes.empty() ? 0 : input_sizes[0]};
        }
        const size_t n_bytes  = input_sizes.empty() ? 0 : input_sizes[0];
        const size_t n_chunks = (n_bytes + chunk_size_ - 1) / chunk_size_;
        const size_t hdr      = 4 + 4 + 8 * n_chunks;
        // Every bound below is against the PADDED extent, not the input size.
        // execute() zero-pads a partial tail chunk up to chunk_size_ and encodes
        // it as a full chunk, so the tail can contribute chunk_size_ bytes of
        // output from fewer than chunk_size_ bytes of input. Bounding by
        // `n_bytes` therefore under-reserves by exactly the tail padding, and
        // the encode writes past the buffer the DAG allocated (E22 in the
        // benchmarking repo: overruns of 4-40 B observed, silent when this
        // stage is mid-pipeline).
        const size_t padded = n_chunks * chunk_size_;

        if (split_mode_) {
            const size_t block_elems = chunk_size_ / word_size_;
            const size_t flag_stride = (block_elems + 7) / 8;
            // literals: every element a literal (or every chunk raw) -> padded.
            // lengths/offsets: one byte per match, at most one match per element.
            // meta: header + every chunk's full-width bitmap.
            return {align4(padded),
                    align4(n_chunks * block_elems),
                    align4(n_chunks * block_elems),
                    align4(hdr + n_chunks * flag_stride)};
        }
        // Forward: worst case = padded data (every chunk falls back to raw
        // storage) + stream header (two uint32_t per chunk).
        // postStreamSync() rounds the final size up to a 4-byte boundary and
        // zero-fills the pad; reserve that pad here too (see RREStage).
        return {align4(padded + hdr)};
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
        if (input_sizes.empty()) return 0;
        const size_t in_bytes       = input_sizes[0];
        const size_t n_chunks       = (in_bytes + chunk_size_ - 1) / chunk_size_;
        const size_t block_elems    = chunk_size_ / word_size_;
        const size_t flag_bytes_max = (block_elems + 7) / 8;
        if (is_inverse_) {
            // Split inverse restripes the four ports back into the packed
            // single-stream form before running the normal decode path.
            return split_mode_ ? (in_bytes + n_chunks * flag_bytes_max
                                  + 4 * n_chunks * sizeof(uint32_t))
                               : 0;
        }
        size_t bytes = n_chunks * (static_cast<size_t>(chunk_size_)
                                   + flag_bytes_max + 4 * sizeof(uint32_t));
        if (split_mode_) bytes += n_chunks * 5 * sizeof(uint32_t) + 16;
        return bytes;
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
        if (max_size < 14) return 0;
        std::memcpy(buf,      &chunk_size_,          sizeof(uint32_t));
        buf[4] = word_size_;
        std::memcpy(buf + 5,  &cached_orig_bytes_,   sizeof(uint32_t));
        buf[9] = split_mode_ ? 1u : 0u;
        std::memcpy(buf + 10, &orig_unpadded_bytes_, sizeof(uint32_t));
        return 14;
    }

    void deserializeHeader(const uint8_t* buf, size_t size) override {
        if (size >= 4)  std::memcpy(&chunk_size_,        buf,     sizeof(uint32_t));
        if (size >= 5)  word_size_ = buf[4];
        if (size >= 9)  std::memcpy(&cached_orig_bytes_, buf + 5, sizeof(uint32_t));
        if (size >= 10) split_mode_ = (buf[9] != 0);
        if (size >= 14) std::memcpy(&orig_unpadded_bytes_, buf + 10, sizeof(uint32_t));
    }

    size_t getMaxHeaderSize(size_t) const override { return 14; }

    void saveState() override {
        saved_chunk_size_        = chunk_size_;
        saved_word_size_         = word_size_;
        saved_cached_orig_bytes_ = cached_orig_bytes_;
        saved_split_mode_        = split_mode_;
        saved_orig_unpadded_bytes_ = orig_unpadded_bytes_;
    }

    void restoreState() override {
        chunk_size_        = saved_chunk_size_;
        word_size_         = saved_word_size_;
        cached_orig_bytes_ = saved_cached_orig_bytes_;
        split_mode_        = saved_split_mode_;
        orig_unpadded_bytes_ = saved_orig_unpadded_bytes_;
    }

private:
    static constexpr size_t align4(size_t n) { return (n + 3) & ~size_t(3); }

    /// Completes the deferred 4-entry per-port size readback used by split mode.
    void finishSplitReadback(fz::stream_t stream) const;

    bool     is_inverse_;
    uint32_t chunk_size_;
    uint32_t saved_chunk_size_ = 0;
    uint8_t  word_size_;
    uint8_t  saved_word_size_ = 0;
    uint8_t  match_level_ = 1;
    bool     split_mode_ = false;
    bool     saved_split_mode_ = false;
    size_t   actual_output_size_;
    // Split mode: per-port actual sizes, in getOutputNames() order.
    size_t   actual_split_sizes_[4] = {0, 0, 0, 0};
    uint32_t cached_orig_bytes_ = 0;          // chunk-padded extent the codec works on
    uint32_t saved_cached_orig_bytes_ = 0;
    // True input size before tail-chunk padding. The inverse decodes the full
    // padded extent but must *report* this, or a downstream stage that derives
    // an element count from its input size (LorenzoQuantStage's inverse does
    // exactly that) inflates its own output past the allocated buffer.
    uint32_t orig_unpadded_bytes_ = 0;
    uint32_t saved_orig_unpadded_bytes_ = 0;

    // ── Persistent forward scratch buffers ─────────────────────────────────
    uint8_t*  d_data_scratch_ = nullptr;
    uint8_t*  d_flag_scratch_ = nullptr;
    uint32_t* d_flag_size_    = nullptr;
    uint32_t* d_data_size_    = nullptr;
    uint32_t* d_clean_dev_    = nullptr;
    uint32_t* d_dst_off_dev_  = nullptr;
    // Split mode: per-chunk destination offsets into the literals / token
    // streams, plus a 4-entry device totals array read back in postStreamSync.
    uint32_t* d_lit_off_dev_  = nullptr;
    uint32_t* d_tok_off_dev_  = nullptr;
    uint32_t* d_meta_off_dev_ = nullptr;
    uint32_t* d_lit_cnt_dev_  = nullptr;
    uint32_t* d_tok_cnt_dev_  = nullptr;
    uint32_t* d_totals_dev_   = nullptr;
    mutable uint8_t*     split_out_ptr_[4] = {nullptr, nullptr, nullptr, nullptr};
    mutable bool         split_readback_pending_ = false;
    mutable bool         tail_readback_pending_ = false;
    mutable fz::stream_t tail_readback_stream_  = nullptr;
    mutable uint32_t     tail_last_index_       = 0;
    mutable uint8_t*     tail_output_ptr_       = nullptr;
    size_t      scratch_capacity_ = 0;
    MemoryPool* scratch_pool_owner_ = nullptr;
    bool        scratch_from_pool_ = false;
};

} // namespace fz
