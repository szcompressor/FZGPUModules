#pragma once

/**
 * @file huffman_stage.h
 * @brief Huffman entropy coding stage with selectable encode mode.
 *
 * Forward: `T[]` → variable-length PHF bitstream (inline phf_header prepended).
 * Inverse: PHF bitstream → `T[]`.
 *
 * Two encode modes are available via setEncodeMode():
 *   - HuffmanEncodeMode::Coarse (default): multi-kernel coarse-grained path with a
 *     CPU prefix-sum sync in the middle of encode. Stable and well-tested.
 *   - HuffmanEncodeMode::Fine: single ReVISIT-lite kernel path with a fully GPU-async
 *     phase 3 (CUB ExclusiveSum + custom reduce kernel). No mid-encode CPU sync;
 *     preferred for latency-sensitive or graph-capture use cases.
 *
 * Note: the histogram D2H is a CPU-sync operation in both modes.  It disappears
 * entirely under HuffmanBookSource::Fixed, which builds the codebook once up front
 * instead of per call; see setBookSource().  The encoded stream is unchanged, so
 * fixed-book output decodes with a stock decoder.
 *
 * Supported input types: `uint8_t`, `uint16_t`, `uint32_t`.
 *
 * Serialized header layout (11 bytes):
 *   [0]     DataType of T   (1 byte)
 *   [1..2]  bklen_          (uint16_t LE)
 *   [3..10] original_len_   (uint64_t LE, element count)
 */

#include "stage/stage.h"
#include "fzm_format.h"
#include "coders/huffman/phf/hf.h"   // phf_header, phf_stream_t

#include "backend/types.h"
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

// Forward-declare phf::Buf<T> to avoid pulling hf_buf.h (CUDA-only) into this header.
// HuffmanStage<T>::~HuffmanStage() is defined in huffman_stage.cu where the type is complete.
namespace phf { template<typename E> struct Buf; }

namespace fz {

/** Selects the PHF encode algorithm used by HuffmanStage on the forward path. */
enum class HuffmanEncodeMode {
    Coarse, ///< Multi-kernel coarse path; CPU prefix-sum sync in phase 3 (default).
    Fine,   ///< ReVISIT-lite single kernel; fully GPU-async phase 3, no mid-encode CPU sync.
};

/**
 * Selects where the Huffman codebook comes from on the forward path.
 *
 * `PerBlock` is the classic path: histogram the input, copy the frequencies to the
 * host, build a canonical tree, copy the codebook back.  `Fixed` builds one book up
 * front and reuses it for every call, which removes the histogram kernel, the
 * frequency D2H, the host stream sync, and the host tree build from every compress.
 *
 * Prior work: CEAZ (Xiong et al., ICS'22) generates canonical codewords offline from
 * representative scientific data; Shah et al., *Lightweight Huffman Coding for
 * Efficient GPU Compression* (ICS'23) precomputes a dictionary of codebooks fitted to
 * cuSZ's quantization-code distribution and selects one at runtime.  The win grows as
 * the per-call payload shrinks, so `Fixed` matters most for small-chunk workloads.
 */
enum class HuffmanBookSource {
    PerBlock,  ///< Histogram + build a fresh codebook on every forward call (default).
    Fixed,     ///< Build one codebook up front and reuse it for every forward call.
    Adaptive,  ///< Histogram the *first* call only, then reuse that codebook forever.
};

/** Analytic symbol distribution used to synthesize a fixed codebook. */
enum class HuffmanBookModel {
    Gaussian,           ///< exp(-((i-center)/scale)^2 / 2)
    Laplace,            ///< exp(-|i-center|/scale)
    GeneralizedNormal,  ///< exp(-(|i-center|/scale)^shape)
    Uniform,            ///< flat; every symbol equally likely
};

/**
 * Parameters for a model-derived fixed codebook.
 *
 * The model is evaluated over symbols `[0, bklen)` and produces a strictly positive
 * frequency for every symbol, so every symbol is guaranteed to be codable.  This is
 * the important difference from a book trained on a sample of real data, where an
 * unseen symbol gets no code at all.
 */
struct HuffmanBookSpec {
    HuffmanBookModel model  = HuffmanBookModel::Gaussian;
    /// Distribution center in symbol space.  Negative means "use bklen/2", which is
    /// where an unsigned quantizer with radius = bklen/2 puts the zero-error code.
    double           center = -1.0;
    /// Distribution width in symbols (sigma for Gaussian, b for Laplace, alpha for
    /// GeneralizedNormal).  Ignored by Uniform.
    double           scale  = 32.0;
    /// Exponent for GeneralizedNormal only (2.0 == Gaussian, 1.0 == Laplace).
    double           shape  = 2.0;
};

/**
 * Huffman entropy coding stage.
 *
 * Forward: `T[] → uint8_t[]`  PHF-encoded bitstream with embedded phf_header.
 * Inverse: `uint8_t[] → T[]`  Decoded symbol stream.
 *
 * @note **Prior work:** PHF source files (`hf.h`, `hf_bk*.cc`, `hf_buf.cc`,
 *       `hf_canon.cc`, `hf_hl.cc`, `hf_kernels.cu`, `hf_impl.hh`) are vendored
 *       and adapted from the cuSZ PHF codec (`origin/v1.1.0_dev`), by the cuSZ
 *       team (BSD-3-Clause). Changes are documented at the top of each file.
 *       See `THIRD_PARTY.md`.
 *
 * @tparam T  Input element type: `uint8_t`, `uint16_t`, or `uint32_t`.
 */
template <typename T>
class HuffmanStage : public Stage {
    static_assert(
        std::is_same_v<T, uint8_t>  ||
        std::is_same_v<T, uint16_t> ||
        std::is_same_v<T, uint32_t>,
        "HuffmanStage: T must be uint8_t, uint16_t, or uint32_t.");

public:
    // __host__-only: HIP's clang infers __host__ __device__ for defaulted
    // special members by default, but ~unique_ptr<phf::Buf<T>> (the type this
    // destructor implicitly invokes, defined in huffman_stage.cu) is
    // host-only, so that inference fails to compile under HIP. Neither
    // function touches device code at all — pin them host-only explicitly.
    __host__ HuffmanStage();
    __host__ ~HuffmanStage() override;

    // ── Configuration ─────────────────────────────────────────────────────────

    /**
     * Set the Huffman codebook length (number of distinct symbols).
     *
     * Must be ≤ 2^(8*sizeof(T)).  Typical values:
     *   uint8_t  : 256  (covers all possible byte values)
     *   uint16_t : 1024 (covers quantization codes in [-512, 511])
     *   uint32_t : 1024 (must be set explicitly; 2^32 is too large for a codebook)
     *
     * Set before the first compress() call.  Changing bklen after the first
     * execute() forces Buf reallocation on the next call (old buffers returned to
     * pool, new ones allocated).  Buf is also reallocated when inlen grows past
     * the previously allocated capacity; shrinking inlen reuses the existing
     * allocation.
     * Default: 256 for uint8_t, 1024 for uint16_t/uint32_t.
     */
    /**
     * An **odd** bklen is rounded up to the next even value. The reverse
     * codebook occupies `4*(2*32) + sizeof(SYM)*bklen` bytes, so with a 2-byte
     * symbol an odd bklen makes it an odd number of 16-bit words; that shifts
     * the bitstream section to a 2-mod-4 offset, and the decode kernel reads
     * the bitstream as `uint32*` and faults on the misaligned load. Rounding up
     * is always safe -- a larger codebook only adds unused symbol slots.
     */
    void     setBklen(uint32_t bklen) { bklen_ = bklen + (bklen & 1u); }
    uint32_t getBklen() const         { return bklen_; }

    /**
     * Select the encode algorithm for the forward path.
     *
     * Must be called before the first compress() / execute() call (or before
     * the next one if changing mode at runtime — triggers Buf reallocation).
     * Default: HuffmanEncodeMode::Coarse.
     */
    void             setEncodeMode(HuffmanEncodeMode mode) { encode_mode_ = mode; }
    HuffmanEncodeMode getEncodeMode() const                { return encode_mode_; }

    // ── Pre-built codebooks ───────────────────────────────────────────────────

    /**
     * Select where the forward path gets its codebook.
     *
     * Switching to `Fixed` without first calling setFixedBookFromFreq() or
     * setFixedBookFromModel() throws on the next forward execute(): there is no
     * sensible default distribution to guess.
     *
     * The encoded bitstream still carries its reverse codebook in every mode, so a
     * `Fixed`-mode stream decompresses with a stock decoder and no extra
     * configuration.  Nothing about the file format changes.
     */
    void              setBookSource(HuffmanBookSource src) { book_source_ = src; }
    HuffmanBookSource getBookSource() const                { return book_source_; }

    /**
     * Supply the frequency table that defines the fixed codebook and switch to
     * `HuffmanBookSource::Fixed`.
     *
     * `n` becomes the codebook length (subject to the same odd-value rounding as
     * setBklen(); the padding slot gets frequency 1).  Every entry must be non-zero
     * — a zero-frequency symbol receives no code, and encoding it would corrupt the
     * stream rather than fail loudly.
     *
     * The table is held in host memory only.  It is not written to the stage header
     * (the per-stage config slot is 128 bytes; a reverse codebook is far larger), so
     * a pipeline reconstructed from a saved config gets `PerBlock` back unless the
     * caller re-supplies the table.  Decompression is unaffected — see
     * setBookSource().
     *
     * @throws std::invalid_argument if `n` is 0 or any frequency is 0.
     */
    void setFixedBookFromFreq(const uint32_t* h_freq, uint32_t n);

    /**
     * Synthesize the fixed codebook from an analytic distribution and switch to
     * `HuffmanBookSource::Fixed`.
     *
     * Call setBklen() first: the model is evaluated over `[0, bklen)`.  Every symbol
     * gets a strictly positive frequency, so the resulting book can encode any
     * in-range symbol.
     */
    void setFixedBookFromModel(const HuffmanBookSpec& spec);

    /// Frequency table backing the fixed codebook; empty when none has been set.
    const std::vector<uint32_t>& getFixedBookFreq() const { return fixed_freq_; }

    /**
     * Dynamic-range clamp used by `HuffmanBookSource::Adaptive`, as a right shift of
     * the largest sampled frequency: every symbol's frequency is floored at
     * `max(1, max_freq >> shift)`.
     *
     * The floor does two jobs.  It guarantees a code for every symbol in `[0, bklen)`,
     * including ones absent from the sampled block — without it, a symbol that shows
     * up only in a later block would have no code at all.  And it bounds the
     * frequency dynamic range, which bounds Huffman depth: an unbounded range can
     * demand codes wider than the 27 bits `HuffmanWord<4>` holds.
     *
     * A larger shift tracks the sampled distribution more closely (better ratio); a
     * smaller one is flatter and safer.  The build halves the shift and retries when
     * the resulting book does not fit, so this is a starting point, not a hard
     * setting — shift 0 is a uniform book, which always fits.  Default 24.
     */
    void    setAdaptiveFloorShift(uint8_t shift) { adaptive_floor_shift_ = shift; }
    uint8_t getAdaptiveFloorShift() const        { return adaptive_floor_shift_; }

    /// Shift actually used by the last Adaptive build, after any retries.  Equals
    /// getAdaptiveFloorShift() unless the book had to be flattened to fit.
    uint8_t getAdaptiveFloorShiftUsed() const    { return adaptive_shift_used_; }

    /**
     * Refit trigger for `Adaptive`: rebuild the codebook once the encoded bit rate
     * degrades past `ratio` times the rate the book achieved when it was fitted.
     *
     * A pinned book goes stale when the symbol distribution moves — measured at
     * 4.7% mean ratio loss across 23 CESM levels, 8.3% across differing fields, and
     * far worse if the book was fitted to an unrepresentative block.  Detecting
     * that would normally cost the histogram this mode exists to avoid, but encode
     * already reports `total_nbit`, so bits-per-symbol is free.  When it degrades,
     * the *next* call histograms and re-pins; the current call is already encoded
     * and is not retroactively improved.
     *
     * 1.2 (the default) refits after a 20% bit-rate regression.  Lower refits more
     * eagerly, trading throughput for ratio; `PerBlock` is the limit of that trade.
     * 0 disables refitting, pinning the first book forever.
     */
    void  setRefitThreshold(float ratio) { refit_threshold_ = ratio; }
    float getRefitThreshold() const      { return refit_threshold_; }

    /**
     * Refit unconditionally every `n` compress calls (0 = never).
     *
     * The bit-rate trigger above only fires when the rate gets *worse*, so it
     * catches a distribution drifting toward less compressible — but not a block
     * that is more compressible than the fitted one while still being badly served
     * by its codebook.  Measured: stepping through five different CESM-ATM
     * variables, `LHFLX` lost 27.6% ratio to a codebook fitted on `CLDHGH`, yet its
     * absolute bit rate *fell*, so the threshold never fired.  Deciding that case
     * properly needs the fresh book's rate for comparison, which means the
     * histogram this mode exists to avoid.
     *
     * A periodic refit sidesteps the question: it costs one histogram every `n`
     * calls, bounding how long a stale book can persist regardless of which
     * direction the rate moved.  `n = 1` is exactly `PerBlock`.
     *
     * Default 0.  Set it when successive calls carry genuinely different data
     * (separate variables rather than successive slabs of one field); leave it off
     * when they are slabs of one field, where the bit-rate trigger suffices.
     */
    void     setRefitInterval(uint32_t n) { refit_interval_ = n; }
    uint32_t getRefitInterval() const     { return refit_interval_; }

    /// Number of times `Adaptive` has re-fitted its codebook.  Diagnostic: a
    /// steadily climbing count means the data is drifting faster than one book can
    /// track, and `PerBlock` may be the better choice.
    uint32_t getRefitCount() const { return refit_count_; }

    /// Bits per symbol the resident Adaptive book achieved on the block it was
    /// fitted to; 0 before the first fit.  This is the baseline the refit trigger
    /// compares against.
    double getFitBitsPerSymbol() const { return fit_bits_per_sym_; }

    /// True when the fixed book came from setFixedBookFromModel(), i.e. when it is
    /// described by a handful of numbers and so can be written to a TOML config.
    /// A book set from a raw frequency table is not reproducible this way.
    bool                   hasBookSpec() const { return has_book_spec_; }
    const HuffmanBookSpec& getBookSpec() const { return book_spec_; }

    // ── Stage control ─────────────────────────────────────────────────────────
    void setInverse(bool inv) override { is_inverse_ = inv; }
    bool isInverse() const override    { return is_inverse_; }

    // Not graph-compatible in any configuration.  HuffmanBookSource::Fixed removes
    // the histogram D2H, but encode still returns total_nbit/total_ncell to the host
    // to assemble phf_header before the H2D merge, in both encode modes.
    bool isGraphCompatible() const override { return false; }

    // ── Pool lifecycle ────────────────────────────────────────────────────────

    /**
     * Called by Pipeline::finalize() after buffer-size propagation.
     *
     * Pre-allocates phf::Buf<T> from the pool using the estimated input size so
     * PREALLOCATE mode commits all memory at finalize time.  If estimated_inlen
     * is 0 (no size hint available), allocation is deferred to the first execute()
     * call.
     */
    void onFinalize(size_t estimated_inlen, MemoryPool* pool) override;

    size_t estimateDeviceFootprintBytes(size_t inlen) const override;
    size_t estimatePinnedFootprintBytes(size_t inlen) const override;

    // ── Execution ─────────────────────────────────────────────────────────────
    void execute(
        fz::stream_t stream,
        MemoryPool*  pool,
        const std::vector<void*>&   inputs,
        const std::vector<void*>&   outputs,
        const std::vector<size_t>&  sizes
    ) override;

    // ── Metadata ──────────────────────────────────────────────────────────────
    std::string getName()      const override { return "Huffman"; }
    size_t      getNumInputs() const override { return 1; }
    size_t      getNumOutputs()const override { return 1; }

    std::vector<size_t> estimateOutputSizes(
        const std::vector<size_t>& input_sizes
    ) const override {
        if (input_sizes.empty()) return {0};
        if (!is_inverse_) {
            // Upper bound: generous 2× input for worst-case bitstream + header overhead.
            return {input_sizes[0] * 2 + 4096};
        }
        // Inverse: exact decoded size restored from the serialized header.
        return {original_len_ * sizeof(T)};
    }

    std::unordered_map<std::string, size_t>
    getActualOutputSizesByName() const override {
        return {{"output", actual_output_size_}};
    }

    size_t getActualOutputSize(int index) const override {
        return (index == 0) ? actual_output_size_ : 0;
    }

    // ── Type system ───────────────────────────────────────────────────────────
    uint16_t getStageTypeId() const override {
        return static_cast<uint16_t>(StageType::HUFFMAN);
    }

    // Byte-transparent output: opt out of pipeline type-compatibility checking.
    uint8_t getOutputDataType(size_t /*output_index*/) const override {
        return static_cast<uint8_t>(DataType::UNKNOWN);
    }
    uint8_t getInputDataType(size_t /*input_index*/) const override {
        return static_cast<uint8_t>(DataType::UNKNOWN);
    }

    // ── Serialization ─────────────────────────────────────────────────────────
    size_t serializeHeader(
        size_t /*output_index*/, uint8_t* buf, size_t max_size
    ) const override {
        if (max_size < 11) return 0;
        buf[0] = static_cast<uint8_t>(dataTypeOf<T>());
        uint16_t bk = static_cast<uint16_t>(bklen_);
        std::memcpy(buf + 1, &bk,           sizeof(uint16_t));
        std::memcpy(buf + 3, &original_len_, sizeof(uint64_t));
        return 11;
    }

    void deserializeHeader(const uint8_t* buf, size_t size) override {
        if (size >= 3) {
            uint16_t bk;
            std::memcpy(&bk, buf + 1, sizeof(uint16_t));
            bklen_ = bk;
        }
        if (size >= 11)
            std::memcpy(&original_len_, buf + 3, sizeof(uint64_t));
    }

    size_t getMaxHeaderSize(size_t /*output_index*/) const override { return 11; }

    void saveState() override {
        saved_bklen_        = bklen_;
        saved_original_len_ = original_len_;
        saved_output_size_  = actual_output_size_;
    }

    void restoreState() override {
        bklen_              = saved_bklen_;
        original_len_       = saved_original_len_;
        actual_output_size_ = saved_output_size_;
    }

private:
    bool              is_inverse_   = false;
    uint32_t          bklen_        = defaultBklen();
    HuffmanEncodeMode encode_mode_  = HuffmanEncodeMode::Coarse;
    HuffmanBookSource book_source_  = HuffmanBookSource::PerBlock;

    // Frequency table defining the fixed codebook (host-only; see setFixedBookFromFreq).
    std::vector<uint32_t> fixed_freq_;
    // Model that generated fixed_freq_, when it came from setFixedBookFromModel().
    HuffmanBookSpec       book_spec_     {};
    bool                  has_book_spec_ = false;
    uint8_t               adaptive_floor_shift_ = 24;
    uint8_t               adaptive_shift_used_  = 24;

    // Refit bookkeeping (Adaptive only).
    float    refit_threshold_  = 1.2f;
    uint32_t refit_interval_   = 0;     // 0 = only the bit-rate trigger
    uint32_t calls_since_fit_  = 0;
    double   fit_bits_per_sym_ = 0.0;   // rate the resident book achieved when fitted
    bool     just_fitted_      = false; // next encode establishes fit_bits_per_sym_
    uint32_t refit_count_      = 0;
    // True when buf_->d_bk4 / d_revbk4 currently hold the fixed book.  Cleared by
    // initBuf(), which allocates fresh (uninitialized) codebook buffers.
    bool                  fixed_book_resident_ = false;

    uint64_t original_len_       = 0;   // element count set by forward execute
    size_t   actual_output_size_ = 0;
    size_t            cap_inlen_        = 0;                          // allocated capacity (elements); grow-only
    uint32_t          last_bklen_       = 0;                          // bklen_ when buf_ was last allocated
    HuffmanEncodeMode last_encode_mode_ = HuffmanEncodeMode::Coarse;  // encode_mode_ when buf_ was last allocated

    // Histogram launch params — computed once in initBuf(), reused every execute()
    int hist_grid_dim_    = 0;
    int hist_block_dim_   = 0;
    int hist_shmem_use_   = 0;
    int hist_r_per_block_ = 0;

    // PHF working buffers — allocated from pool_ on first execute() or in onFinalize()
    std::unique_ptr<phf::Buf<T>> buf_;
    phf_header header_ {};

    // Pool used for buf_ allocations.  Set by onFinalize() or captured from the
    // pool parameter on the first execute() call.  Raw non-owning pointer; the
    // pool outlives the stage when used inside a Pipeline.
    MemoryPool* pool_ = nullptr;

    // saveState / restoreState snapshots
    uint32_t saved_bklen_        = defaultBklen();
    uint64_t saved_original_len_ = 0;
    size_t   saved_output_size_  = 0;

    static constexpr uint32_t defaultBklen() {
        if constexpr (std::is_same_v<T, uint8_t>) return 256;
        return 1024;
    }

    template<typename U>
    static constexpr DataType dataTypeOf() {
        if constexpr (std::is_same_v<U,  uint8_t>)  return DataType::UINT8;
        if constexpr (std::is_same_v<U, uint16_t>)  return DataType::UINT16;
        return DataType::UINT32;
    }

    // Allocates buf_ from pool and computes histogram launch params for the given
    // element count.  Must be in huffman_stage.cu: calls cudaFuncSetAttribute with
    // a __global__ pointer.  If buf_ already exists, destroys it first (returning
    // its allocations to the pool) before creating the new one.
    void initBuf(size_t inlen, MemoryPool* pool);

    // Builds the canonical codebook from fixed_freq_ into buf_ and H2Ds it.
    // Requires buf_ to exist.  Must be in huffman_stage.cu (needs phf::Buf<T>).
    void buildFixedBook(fz::stream_t stream);

    // Builds a reusable codebook from a sampled histogram, flooring frequencies and
    // retrying with a flatter floor until the book fits the 27-bit code field.
    void buildAdaptiveBook(const uint32_t* h_hist, fz::stream_t stream);

    // Index of the first symbol with freq > 0 whose built code is unusable, or -1.
    int findUnusableCode(const uint32_t* freq) const;
};

extern template class HuffmanStage<uint8_t>;
extern template class HuffmanStage<uint16_t>;
extern template class HuffmanStage<uint32_t>;

} // namespace fz
