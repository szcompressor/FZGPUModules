#pragma once

/**
 * @file rle.h
 * @brief Run-Length Encoding stage (lossless, stream-ordered).
 */

#include "stage/stage.h"
#include "fzm_format.h"
#include "log.h"
#include "backend/types.h"
#include <cstdint>
#include <cstring>
#include <memory>
#include <type_traits>

namespace fz {

/**
 * Byte offset of the values section within the packed RLE wire format,
 * rounded up to `alignof(T)`.  The 4-byte `num_runs` header alone only
 * guarantees 4-byte alignment; for 8-byte `T` (`int64_t`/`uint64_t`) the
 * values section must start on an 8-byte boundary or the `reinterpret_cast<T*>`
 * reads/writes in `rle_pack_kernel`/`execute()` fault with an unaligned
 * 64-bit load (found via the RLE_8 word-size round-trip test).
 */
template<typename T>
constexpr size_t rleValuesOffset() {
    return (alignof(T) > sizeof(uint32_t)) ? alignof(T) : sizeof(uint32_t);
}

/**
 * Byte offset of the values section within the *chunked* wire format, given the
 * chunk count.  The offset table is `num_chunks + 1` `uint32_t` entries
 * following the `num_chunks` header word; the values section is then rounded up
 * to `alignof(T)` for the same reason as `rleValuesOffset<T>()`.
 */
template<typename T>
constexpr size_t rleChunkedValuesOffset(size_t num_chunks) {
    const size_t hdr = (num_chunks + 2) * sizeof(uint32_t);
    return (alignof(T) > sizeof(uint32_t))
               ? ((hdr + alignof(T) - 1) & ~(alignof(T) - 1))
               : hdr;
}

/**
 * Run-Length Encoding stage. Lossless; effective when data has long runs of
 * identical values (e.g. quantized codes).
 *
 * Forward wire format: `[num_runs:u32][pad to alignof(T)][values:T×n (4B-aligned)][lengths:u32×n]`
 * The header-to-values pad is 0 bytes for T ≤ 4 bytes, 4 bytes for 8-byte T.
 *
 * ### Chunked mode (`setChunkSize(bytes)`, opt-in)
 *
 * With a non-zero chunk size the input is cut into independent chunks of
 * `chunk_size / sizeof(T)` elements, each encoded by a single thread block with
 * a block-local scan.  This removes the device-wide CUB scan and the
 * serialising dependency chain of the global path (5 kernels + a full-length
 * scan collapse to 2 kernels + a scan over `num_chunks` elements), which is
 * where the throughput comes from; the cost is a forced run boundary at every
 * chunk start plus a `4 × (num_chunks + 1)` byte offset table, so the
 * compression ratio drops slightly as chunks get smaller.
 *
 * Chunked wire format:
 * `[num_chunks:u32][run_offsets:u32×(num_chunks+1)][pad to alignof(T)]`
 * `[values:T×total_runs (4B-aligned)][lengths:u32×total_runs]`
 * where `run_offsets[c]` is the index of chunk `c`'s first run and
 * `run_offsets[num_chunks] == total_runs`.
 *
 * Chunked decode needs no device-to-host readback at all (the element count and
 * chunk size both come from the serialized stage header), so both directions are
 * CUDA Graph-capturable — the global path's inverse is not.
 *
 * Worst-case output is 2× input + 4 bytes (no repeated values), so RLE should
 * follow a predictor/quantizer stage that creates repetition.
 *
 * @tparam T  Element type (`uint8_t`/`uint16_t`/`uint32_t`/`uint64_t`,
 *            `int8_t`/`int16_t`/`int32_t`/`int64_t` — full 1/2/4/8-byte
 *            word-size coverage, matching the LC framework's RLE_1/2/4/8).
 *            Run counts are always `uint32_t`.
 */
template<typename T = uint16_t>
class RLEStage : public Stage {
public:
    RLEStage() : is_inverse_(false) {}
    ~RLEStage() override;

    void setInverse(bool inverse) override { is_inverse_ = inverse; }
    bool isInverse() const override { return is_inverse_; }

    void execute(
        fz::stream_t stream,
        MemoryPool* pool,
        const std::vector<void*>& inputs,
        const std::vector<void*>& outputs,
        const std::vector<size_t>& sizes
    ) override;

    /**
     * Completes the async D2H readback of num_runs started during forward
     * execute() and sets actual_output_sizes_.  Must be called after the
     * stream passed to execute() has been synchronized.
     */
    void postStreamSync(fz::stream_t stream) override;
    
    std::string getName() const override { return "RLE"; }
    size_t getNumInputs() const override { return 1; }
    size_t getNumOutputs() const override { return 1; }

    /**
     * Chunk size in **bytes**; 0 (the default) selects the whole-array path.
     * Must be a multiple of `sizeof(T)` and of 4; values are rounded down to
     * the nearest multiple of `sizeof(T)`.  Typical: 4096–65536.
     */
    void setChunkSize(size_t bytes) {
        chunk_size_ = static_cast<uint32_t>(bytes - (bytes % sizeof(T)));
    }
    size_t getChunkSize() const { return chunk_size_; }
    bool   isChunked()    const { return chunk_size_ >= sizeof(T); }

    /// Chunked mode needs whole chunks; the pipeline zero-pads the input to suit.
    size_t getRequiredInputAlignment() const override {
        return isChunked() ? chunk_size_ : 1;
    }

    /**
     * Persistent forward-path scratch:
     *   d_is_boundary_     : n bytes
     *   d_boundary_scan_   : n × u32
     *   d_boundary_positions_: n × u32  (worst-case, avoids D2H for num_runs)
     *   d_values_scratch_  : n × T
     *   d_lengths_scratch_ : n × u32
     * All five arrays are sized to the largest n seen so far and reused
     * across calls, eliminating per-call cudaMallocAsync overhead.
     */
    size_t estimateScratchBytes(
        const std::vector<size_t>& input_sizes
    ) const override {
        if (is_inverse_ || input_sizes.empty()) return 0;
        const size_t n = input_sizes[0] / sizeof(T);
        // is_boundary(1B) + boundary_scan(4B) + boundary_positions(4B)
        //   + values_scratch(sizeof(T)) + lengths_scratch(4B)
        // Chunked mode reuses the same arrays: boundary_positions holds the
        // per-chunk run start positions and boundary_scan holds the per-chunk
        // run counts, so the bound is unchanged (and is_boundary goes unused).
        return n * (1 + 4 + 4 + sizeof(T) + 4);
    }
    
    std::vector<size_t> estimateOutputSizes(
        const std::vector<size_t>& input_sizes
    ) const override {
        if (is_inverse_) {
            // Use the element count cached from the forward pass (or deserialized
            // from the file header) for an exact estimate.  Falls back to a
            // conservative 2× bound only when no prior forward pass has run.
            if (cached_num_elements_ > 0)
                return {static_cast<size_t>(cached_num_elements_) * sizeof(T)};
            return {input_sizes[0] * 2};
        } else {
            // Compression: worst case is every element is unique.
            // Wire format: [num_runs:u32][values:T×n, 4B-aligned][lengths:u32×n]
            // The values section is padded to a 4-byte boundary (matching
            // rle_pack_kernel), so the estimate must include that padding or
            // the allocated buffer will be too small and the lengths write OOBs.
            size_t n = input_sizes[0] / sizeof(T);
            size_t values_bytes   = n * sizeof(T);
            size_t values_aligned = (values_bytes + 3u) & ~3u;
            if (isChunked()) {
                // Same worst case (every element unique) plus the offset table.
                const size_t nc = numChunks(input_sizes[0]);
                return {rleChunkedValuesOffset<T>(nc) + values_aligned
                        + n * sizeof(uint32_t)};
            }
            return {rleValuesOffset<T>() + values_aligned + n * sizeof(uint32_t)};
        }
    }
    
    std::unordered_map<std::string, size_t> getActualOutputSizesByName() const override {
        completePendingSync();
        completePendingDecodeSync();
        return {{"output", actual_output_sizes_.empty() ? 0 : actual_output_sizes_[0]}};
    }
    size_t getActualOutputSize(int index) const override {
        completePendingSync();
        completePendingDecodeSync();
        return (index == 0 && !actual_output_sizes_.empty()) ? actual_output_sizes_[0] : 0;
    }

    uint16_t getStageTypeId() const override {
        return static_cast<uint16_t>(StageType::RLE);
    }
    
    uint8_t getOutputDataType(size_t output_index) const override {
        (void)output_index;
        return static_cast<uint8_t>(getDataTypeEnum());
    }

    uint8_t getInputDataType(size_t /*input_index*/) const override {
        return static_cast<uint8_t>(getDataTypeEnum());
    }
    
    size_t serializeHeader(size_t output_index, uint8_t* header_buffer, size_t max_size) const override {
        (void)output_index;
        const size_t needed = sizeof(DataType) + 2 * sizeof(uint32_t);
        if (max_size < needed) return 0;
        DataType dt = getDataTypeEnum();
        std::memcpy(header_buffer, &dt, sizeof(DataType));
        std::memcpy(header_buffer + sizeof(DataType), &cached_num_elements_, sizeof(uint32_t));
        std::memcpy(header_buffer + sizeof(DataType) + sizeof(uint32_t),
                    &chunk_size_, sizeof(uint32_t));
        return needed;
    }

    void deserializeHeader(const uint8_t* header_buffer, size_t size) override {
        if (size >= sizeof(DataType) + sizeof(uint32_t))
            std::memcpy(&cached_num_elements_, header_buffer + sizeof(DataType), sizeof(uint32_t));
        // chunk_size_ is absent in streams written before chunked mode existed;
        // leaving it at its default keeps those decoding on the global path.
        if (size >= sizeof(DataType) + 2 * sizeof(uint32_t))
            std::memcpy(&chunk_size_, header_buffer + sizeof(DataType) + sizeof(uint32_t),
                        sizeof(uint32_t));
    }

    size_t getMaxHeaderSize(size_t output_index) const override {
        (void)output_index;
        return sizeof(DataType) + 2 * sizeof(uint32_t);
    }

private:
    bool is_inverse_;

    /// Chunk size in bytes; 0 → whole-array (non-chunked) path.
    uint32_t chunk_size_ = 0;

    /// Elements per chunk, and the chunk count covering `bytes` of input.
    uint32_t elemsPerChunk() const {
        return static_cast<uint32_t>(chunk_size_ / sizeof(T));
    }
    size_t numChunks(size_t in_bytes) const {
        const size_t epc = elemsPerChunk();
        const size_t n   = in_bytes / sizeof(T);
        return epc ? (n + epc - 1) / epc : 0;
    }

    /// Cached original element count from the most recent forward pass.
    /// Persisted in the serialized header so inverse `estimateOutputSizes()`
    /// returns an exact bound even for cold decompression from file.
    uint32_t cached_num_elements_ = 0;

    // ── Persistent forward-path scratch ──────────────────────────────────────
    // Allocated lazily on the first forward execute(); grown if n increases.
    uint8_t*    d_is_boundary_        = nullptr;
    uint32_t*   d_boundary_scan_      = nullptr;
    uint32_t*   d_boundary_positions_ = nullptr; ///< Worst-case n elements.
    T*          d_values_scratch_     = nullptr;
    uint32_t*   d_lengths_scratch_    = nullptr;
    size_t      fwd_scratch_n_        = 0;        ///< Current scratch capacity (elements).
    MemoryPool* fwd_scratch_pool_     = nullptr;
    bool        fwd_from_pool_        = false;
    std::weak_ptr<const void> fwd_scratch_alive_; ///< See MemoryPool::lifetimeToken().

    // Pinned host buffer for async D2H of num_runs.
    // mutable so getActualOutputSizesByName() can complete the pending
    // readback even when called on a const Stage reference.
    mutable uint32_t*           h_num_runs_          = nullptr;
    mutable MemoryPool*         h_num_runs_pool_     = nullptr;
    mutable std::weak_ptr<const void> h_num_runs_alive_; ///< See MemoryPool::lifetimeToken().
    mutable bool                fwd_sync_pending_    = false;
    mutable fz::stream_t        fwd_last_stream_     = nullptr;
    mutable std::vector<size_t> actual_output_sizes_;

    // Pinned host buffer for the decode path's deferred total-output-size
    // readback (the decompress kernel launch itself only needs num_runs, which
    // is already synced earlier — this second D2H was previously blocking
    // *before* that kernel launch for no operational reason; deferring it here
    // removes a host stall from the decode critical path, same pattern as the
    // forward path's h_num_runs_ above).
    mutable uint32_t*           h_dec_total_size_       = nullptr;
    mutable MemoryPool*         h_dec_total_size_pool_  = nullptr;
    mutable std::weak_ptr<const void> h_dec_total_size_alive_; ///< See MemoryPool::lifetimeToken().
    mutable bool                dec_sync_pending_       = false;
    mutable fz::stream_t        dec_last_stream_        = nullptr;

    void completePendingDecodeSync() const {
        if (!dec_sync_pending_) return;
        cudaStreamSynchronize(dec_last_stream_);
        const uint32_t total_output_size = *h_dec_total_size_;
        actual_output_sizes_ = {static_cast<size_t>(total_output_size) * sizeof(T)};
        dec_sync_pending_ = false;
    }

    // Complete a pending forward-path readback (if any) by syncing the stream
    // that was used and computing actual_output_sizes_.  Safe to call from
    // const methods; all state it touches is mutable.
    void completePendingSync() const {
        if (!fwd_sync_pending_) return;
        cudaStreamSynchronize(fwd_last_stream_);
        const uint32_t num_runs      = *h_num_runs_;
        const size_t   values_bytes  = static_cast<size_t>(num_runs) * sizeof(T);
        const size_t   values_aligned = (values_bytes + 3) & ~3;
        const size_t   values_offset = isChunked()
            ? rleChunkedValuesOffset<T>(
                  numChunks(static_cast<size_t>(cached_num_elements_) * sizeof(T)))
            : rleValuesOffset<T>();
        actual_output_sizes_ = {
            values_offset + values_aligned + num_runs * sizeof(uint32_t)
        };
        fwd_sync_pending_ = false;
        // Log run count and effective compression ratio.
        const size_t in_bytes  = static_cast<size_t>(cached_num_elements_) * sizeof(T);
        const size_t out_bytes = actual_output_sizes_[0];
        const float  ratio     = in_bytes > 0
            ? static_cast<float>(in_bytes) / static_cast<float>(out_bytes) : 0.0f;
        FZ_LOG(DEBUG, "RLE encode: %u runs / %u elems  %.1f KB -> %.1f KB  ratio %.2fx",
               num_runs, cached_num_elements_,
               in_bytes / 1024.0f, out_bytes / 1024.0f, ratio);
    }

    // Helper to map template type T to DataType enum
    DataType getDataTypeEnum() const {
        if (std::is_same<T, uint8_t>::value) return DataType::UINT8;
        if (std::is_same<T, uint16_t>::value) return DataType::UINT16;
        if (std::is_same<T, uint32_t>::value) return DataType::UINT32;
        if (std::is_same<T, uint64_t>::value) return DataType::UINT64;
        if (std::is_same<T, int8_t>::value) return DataType::INT8;
        if (std::is_same<T, int16_t>::value) return DataType::INT16;
        if (std::is_same<T, int32_t>::value) return DataType::INT32;
        if (std::is_same<T, int64_t>::value) return DataType::INT64;
        if (std::is_same<T, float>::value) return DataType::FLOAT32;
        if (std::is_same<T, double>::value) return DataType::FLOAT64;
        return DataType::UINT8;  // Fallback
    }
};

extern template class RLEStage<uint8_t>;
extern template class RLEStage<uint16_t>;
extern template class RLEStage<uint32_t>;
extern template class RLEStage<uint64_t>;
extern template class RLEStage<int8_t>;
extern template class RLEStage<int16_t>;
extern template class RLEStage<int32_t>;
extern template class RLEStage<int64_t>;

} // namespace fz
