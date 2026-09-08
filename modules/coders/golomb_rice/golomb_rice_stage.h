#pragma once

/**
 * @file golomb_rice_stage.h
 * @brief Chunk-local Golomb-Rice entropy coder — lossless byte-stream compressor.
 *
 * A genuine entropy coder (variable-length codes sized to each value's
 * magnitude) whose parameter is computed PER 16 KB CHUNK rather than from a
 * global histogram. That is what makes it eligible for the chunk-cooperative
 * fusion path in principle (see the Fusion readiness note below) — the same
 * property that excludes ANS/Huffman (global codebook) from that path today.
 *
 * Rice coding is the classical choice for Laplacian-distributed values, which
 * is exactly the shape of a predictor's residuals (Lorenzo / AdaptiveLorenzo
 * output): most values near zero, a long thin tail. Encode maps a signed
 * value to unsigned via zigzag, splits it into quotient/remainder by a
 * per-chunk parameter k (q = u >> k, r = u & ((1<<k)-1)), and emits `q` one
 * bits, a zero stop bit, then k raw bits of r — so small values cost close to
 * `k+1` bits and the code gets exponentially more expensive as magnitude
 * grows past 2^k.
 *
 * **k selection is exact, not estimated.** For each candidate k in
 * `[0, kMaxCandidate()]`, the encoder computes the EXACT total bit cost the
 * pack step would emit (including the escape below) via a block-wide
 * reduction, and picks the argmin — the same "encoded-size oracle, not a
 * heuristic" discipline used elsewhere in this codebase (see
 * `include/stage/fusion.h`'s `EncodingOracleDecl`).
 *
 * **Escape for outliers.** A per-element unary run is capped at
 * `kEscapeQ` (24) ones: hitting the cap with no stop bit is unambiguous
 * ("escape") and is followed by the full zigzag value as `bitWidth<T>()` raw
 * bits instead of an unbounded unary run. This bounds worst-case per-element
 * cost to `kEscapeQ + bitWidth<T>()` bits regardless of how large one value
 * is, so a single outlier can never blow up a chunk's packed size — the
 * raw-storage fallback below only ever triggers on genuinely incompressible
 * chunks, never on a rare large value.
 *
 * Output stream layout (identical container convention to RZEStage/RREStage):
 * @code
 *   [uint32_t: original byte count]
 *   [uint32_t: num_chunks]
 *   [uint32_t × n_chunks: per-chunk compressed sizes (high bit → stored raw)]
 *   [per-chunk data: [uint32_t: k]
 *                     [uint32_t × kIntervalsPerChunk: restart-interval byte offsets]
 *                     [packed bits, kIntervalsPerChunk independently byte-aligned
 *                      segments...] , or raw bytes if flagged]
 * @endcode
 *
 * **Decode parallelism via restart intervals.** Each chunk's k-selection
 * still uses the whole chunk (no ratio cost), but the packed payload is split
 * into `kIntervalsPerChunk` segments — one per encode-side warp — each
 * independently rounded up to a byte boundary. Decode launches one WARP per
 * chunk with one lane per restart interval, so `kIntervalsPerChunk` Rice
 * decode streams run in parallel per chunk instead of one sequential stream
 * for the whole 16 KB chunk. See golomb_rice_stage.cu for the mechanism.
 *
 * Serialized header (5 bytes): `[0..3]` chunk_size (uint32_t LE), `[4]`
 * element DataType (INT16 or INT32) — needed so `GolombRice_fromHeader` can
 * reconstruct the correct `GolombRiceStage<T>` template instantiation when
 * replaying an archive whose T is a runtime (not caller-known) fact.
 *
 * ## Fusion (compress-only, SHIPPED 2026-09-06)
 * `getFusionSpec()`/`getFusedOp()` declare `FusionAccess::Cooperative` /
 * `FusionStrategy::ChunkCooperative` (T=int32_t, chunk_size=16384 only — the
 * chunk_fusion.cuh harness's shared buffers are a fixed uint32_t[4096]
 * shape). The device op (`GolombRiceCoder` in
 * `modules/fused/chunk_fusion/chunk_fusion.cuh`) reproduces this stage's own
 * `golombRiceEncodeKernel` byte-for-byte (same k selection, same
 * kIntervalsPerChunk restart-interval offsets, same header layout), so the
 * ordinary unfused inverse below decodes a fused-produced archive unchanged
 * — this coder never gets its own inverse, matching RRE/RARE/RAZE/CLOG/HCLOG
 * (only RZE among the chunk-cooperative coders has a fused inverse today).
 * Proven chain: `Quantizer(inplace,zigzag,NOA) -> Difference(plain, same-
 * type int32) -> GolombRice` (`examples/presets/diffplain_golomb_rice.toml`)
 * — Difference must be PLAIN (T==TOut, no fused encode step): this coder
 * zigzags its residuals internally, so an upstream transform that also
 * zigzagged would double-encode (caught by the byte-identity test during
 * development — the fused archive came out smaller than staged, the
 * fingerprint of doubling an always-nonnegative value's magnitude).
 * Byte-identity AND round-trip verified via
 * `FusionPlanner.DiffPlainGolombRiceEndToEndFusedMatchesStaged`
 * (tests/pipeline/test_fusion_planner.cpp), `compute-sanitizer`
 * memcheck/racecheck clean. Measured compress win (single fused kernel vs.
 * 3 staged DRAM round-trips), same DAG-throughput metric both times: CLDHGH
 * (25 MB) 131.8 → 142.5 GB/s (+8%); NYX/baryon_density (512 MB) 180.3 →
 * 280.7 GB/s (+56%) — the win grows with field size, as expected for a
 * DRAM-round-trip-elimination fusion (small fields have little round-trip
 * cost to delete). Compressed size matched exactly (byte-identical) on both.
 *
 * DECODE fusion is still not attempted — see
 * `memory/chunk_local_entropy_coder_design.md` for why (the decode side is
 * necessarily sequential per chunk, which the existing chunk-cooperative
 * INVERSE runner has not needed to support yet; the standalone decode's own
 * throughput investigation is also there, including why several plausible
 * fixes did not pan out).
 *
 * @tparam T Signed integer element type: int16_t or int32_t.
 */

#include "stage/stage.h"
#include "fzm_format.h"
#include "backend/types.h"
#include "fused/common/data_type_of.h"
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace fz {

template<typename T>
class GolombRiceStage : public Stage {
    static_assert(std::is_integral<T>::value && std::is_signed<T>::value,
                  "GolombRiceStage requires a signed integer type");
public:
    GolombRiceStage() = default;
    ~GolombRiceStage() override;

    // ── Stage control ──────────────────────────────────────────────────────
    void setInverse(bool inv) override { is_inverse_ = inv; }
    bool isInverse() const override    { return is_inverse_; }

    /// Decode is inherently sequential per chunk (a Rice codeword's length
    /// isn't known until decoded) and reads the stream header with blocking
    /// D2H copies before it can launch, like RZE/RRE's inverse.
    bool isGraphCompatible() const override { return !is_inverse_; }

    void setChunkSize(size_t bytes) { chunk_size_ = static_cast<uint32_t>(bytes); }
    size_t getChunkSize() const { return chunk_size_; }
    size_t getRequiredInputAlignment() const override { return chunk_size_; }
    uint32_t getCachedOrigBytes() const { return cached_orig_bytes_; }

    /// Variable-length coder = the sink that terminates a chunk-cooperative fused
    /// chain (compress only — see the "Fusion readiness" note in the class
    /// doc-comment above for why decode fusion is deliberately not attempted
    /// yet). Gated to T=int32_t: the chunk_fusion.cuh harness's shared buffers
    /// are a fixed uint32_t[4096] (16 KB) shape; GolombRiceCoder is written
    /// against that width, matching the other chunk-cooperative coders' int32/
    /// byte-word conventions. block_size is the chunk in bytes.
    FusionSpec getFusionSpec() const override {
        if (is_inverse_ || chunk_size_ != 16384u || !std::is_same<T, int32_t>::value)
            return {};
        return FusionSpec{FusionAccess::Cooperative, chunk_size_};
    }

    /// Chunk-cooperative coder op (the swappable variable-length sink). Stateless.
    FusedOpDecl getFusedOp() const override {
        if (!getFusionSpec().fusable()) return {};
        return FusedOpDecl{FusionStrategy::ChunkCooperative, "GolombRiceCoder",
                           "fused/chunk_fusion/chunk_fusion.cuh", {}};
    }
    /// Base-class tail hook -> the existing coder result setter (archive, orig).
    void setFusedArchiveResult(size_t archive_bytes, size_t orig_bytes) override {
        setFusedResult(archive_bytes, orig_bytes);
    }
    /// Set by a fused runner that produced this coder's archive without execute():
    /// the fused kernel wrote the identical archive blob. Sets the forward output
    /// size AND `cached_orig_bytes_` (the uncompressed input size) — the inverse
    /// output buffer is sized from the latter (else it defaults to the compressed
    /// size and the decode writes out of bounds).
    void setFusedResult(size_t archive_bytes, size_t orig_bytes) {
        actual_output_size_    = archive_bytes;
        cached_orig_bytes_     = static_cast<uint32_t>(orig_bytes);
        tail_readback_pending_ = false;
    }

    // ── Execution ──────────────────────────────────────────────────────────
    void execute(
        fz::stream_t stream,
        MemoryPool* pool,
        const std::vector<void*>& inputs,
        const std::vector<void*>& outputs,
        const std::vector<size_t>& sizes
    ) override;
    void postStreamSync(fz::stream_t stream) override;

    // ── Metadata ───────────────────────────────────────────────────────────
    std::string getName() const override { return "GolombRice"; }
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
        // Forward worst case: every chunk falls back to raw storage (original
        // bytes + 1 flagged size word each) plus the stream header, plus the
        // tail safety pad postStreamSync may need to zero (see execute()).
        const size_t n_bytes  = input_sizes.empty() ? 0 : input_sizes[0];
        const size_t n_chunks = (n_bytes + chunk_size_ - 1) / chunk_size_;
        const size_t hdr      = 4 + 4 + 4 * n_chunks;
        const size_t worst    = n_bytes + hdr + kTailPad;
        return {(worst + 3) & ~size_t(3)};
    }

    std::unordered_map<std::string, size_t>
    getActualOutputSizesByName() const override {
        return {{"output", actual_output_size_}};
    }
    size_t getActualOutputSize(int index) const override {
        return index == 0 ? actual_output_size_ : 0;
    }

    /**
     * Forward pass allocates three persistent pool arrays proportional to
     * n_chunks = ceil(input_bytes / chunk_size_):
     *   d_scratch_  : n_chunks * chunkScratchStride()  (per-chunk worst-case packed output)
     *   d_sizes_    : n_chunks * 4                     (flagged compressed sizes)
     *   d_dst_off_  : n_chunks * 4                      (exclusive prefix-sum offsets)
     * chunkScratchStride() bounds the packed size assuming EVERY element in
     * the chunk escapes (kEscapeQ + bitWidth<T>() bits each) plus the 1-byte
     * k header plus the tail pad — the true worst case, so pack() can never
     * write out of bounds regardless of data.
     */
    size_t estimateScratchBytes(
        const std::vector<size_t>& input_sizes
    ) const override {
        if (is_inverse_ || input_sizes.empty()) return 0;
        const size_t in_bytes = input_sizes[0];
        const size_t n_chunks = (in_bytes + chunk_size_ - 1) / chunk_size_;
        return n_chunks * (chunkScratchStride() + 3 * sizeof(uint32_t));
    }

    uint16_t getStageTypeId() const override {
        return static_cast<uint16_t>(StageType::GOLOMB_RICE);
    }

    uint8_t getOutputDataType(size_t) const override {
        return static_cast<uint8_t>(DataType::UINT8);
    }

    // ── Serialization ──────────────────────────────────────────────────────
    static DataType getElementDataType() { return fused::dataTypeOf<T>(); }

    size_t serializeHeader(size_t, uint8_t* buf, size_t max_size) const override {
        if (max_size < 5) return 0;
        std::memcpy(buf, &chunk_size_, sizeof(uint32_t));
        buf[4] = static_cast<uint8_t>(getElementDataType());
        return 5;
    }
    void deserializeHeader(const uint8_t* buf, size_t size) override {
        if (size >= 4) std::memcpy(&chunk_size_, buf, sizeof(uint32_t));
    }
    size_t getMaxHeaderSize(size_t) const override { return 5; }

    void saveState() override { saved_chunk_size_ = chunk_size_; }
    void restoreState() override { chunk_size_ = saved_chunk_size_; }

    /// Elements per chunk (chunk_size_ is in bytes, matching RZE's convention).
    uint32_t elemsPerChunk() const { return chunk_size_ / static_cast<uint32_t>(sizeof(T)); }

    /// Escape threshold: a unary run this long with no stop bit means "the
    /// value didn't fit the k-bit model — the next bitWidth<T>() bits are the
    /// raw zigzag value instead." Bounds worst-case per-element cost.
    static constexpr uint32_t kEscapeQ = 24;
    static constexpr uint32_t bitWidth() { return 8u * sizeof(T); }
    /// Candidate k range the exact-cost search tries, [0, kMaxCandidate()].
    static constexpr uint32_t kMaxCandidate() { return bitWidth() > 24 ? 24u : bitWidth() - 1; }

    /// Decode-parallelism restart intervals: each chunk's payload is split
    /// into kIntervalsPerChunk independently byte-aligned segments (one per
    /// encode-side warp — see golomb_rice_stage.cu's GR_TPB=512=32*16), so
    /// decode can run one warp per chunk with one lane per interval instead
    /// of one thread per chunk. Kept at 16 (half the warp), NOT 32: measured
    /// both on 2026-09-05. Going to 32 (GR_TPB=1024) roughly halves the
    /// per-warp serial iteration count but bought only a noise-level ~7%
    /// kernel-time change (ncu: 138.78us -> 128.86us, and long-scoreboard
    /// stall% rose 39.8% -> 66.3%, i.e. it made the kernel MORE
    /// memory-stall-bound, not less) while doubling the offset-table header
    /// (64 -> 128 bytes/chunk) for a real ~3% CR cost -- not worth it. Root
    /// cause (profiler-confirmed, not guessed): launch__waves_per_multi
    /// processor is ~0.19 on realistic chunk counts -- total decode warps
    /// equal num_chunks regardless of this constant, so occupancy is capped
    /// by DATASET SIZE, not by interval width; no amount of intra-warp
    /// lane-packing fixes an occupancy ceiling set by too few total warps.
    /// See chunk_local_entropy_coder_design.md's decode-parallelism section.
    /// Must equal GR_TPB/32 in the .cu.
    static constexpr uint32_t kIntervalsPerChunk = 16;

    /// Worst-case packed bytes for one full chunk (every element escapes),
    /// rounded up to a 4-byte boundary for safe atomicOr word access, plus a
    /// small pad so a decode-side 64-bit sliding-window read at the very tail
    /// of a chunk never reads past this stride (see chunk_local_entropy_coder
    /// _design.md's "tail padding" note — same class of fix as the ADM-stage
    /// 16-byte vectorized-read padding in cuszp_h100_optimization memory).
    /// The header now also carries a kIntervalsPerChunk-entry byte-offset
    /// table (one uint32 per restart interval) instead of just the k word,
    /// and each interval independently rounds its own payload up to a byte
    /// boundary, which can waste up to kIntervalsPerChunk bytes worst case.
    size_t chunkScratchStride() const {
        const size_t worst_bits  = static_cast<size_t>(elemsPerChunk()) * (kEscapeQ + bitWidth());
        const size_t worst_bytes = (4 + 4 * kIntervalsPerChunk) /*k + offset table*/
                                 + kIntervalsPerChunk /*per-interval rounding slack*/
                                 + (worst_bits + 7) / 8;
        return ((worst_bytes + 3) & ~size_t(3)) + kTailPad;
    }
    static constexpr size_t kTailPad = 16;

private:
    bool     is_inverse_ = false;
    uint32_t chunk_size_ = 16384;
    uint32_t saved_chunk_size_ = 0;
    size_t   actual_output_size_ = 0;
    uint32_t cached_orig_bytes_ = 0;
    uint32_t saved_cached_orig_bytes_ = 0;

    // ── Persistent forward scratch buffers ───────────────────────────────────
    uint8_t*  d_scratch_ = nullptr;
    uint32_t* d_sizes_dev_ = nullptr;
    uint32_t* d_dst_off_dev_ = nullptr;
    mutable bool         tail_readback_pending_ = false;
    mutable uint32_t     tail_last_index_ = 0;
    mutable uint32_t     tail_header_size_ = 0;
    mutable uint8_t*     tail_output_ptr_ = nullptr;
    size_t    scratch_capacity_ = 0;
    MemoryPool* scratch_pool_owner_ = nullptr;
    bool        scratch_from_pool_ = false;
    std::weak_ptr<const void> scratch_alive_;
};

} // namespace fz
