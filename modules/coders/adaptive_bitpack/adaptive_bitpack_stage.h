#pragma once

/**
 * @file adaptive_bitpack_stage.h
 * @brief Per-block adaptive fixed-rate bit-plane coder (cuSZp-style plain mode).
 *
 * Supported input types: signed `int16_t`, `int32_t` (quantizer codes or block
 * deltas). Output is a self-describing `uint8_t[]` archive.
 */

#include "stage/stage.h"
#include "fzm_format.h"
#include "backend/types.h"
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace fz {

/**
 * Serialized AdaptiveBitpack configuration (FZMBufferEntry.stage_config).
 * 16 bytes — fits within `FZM_STAGE_CONFIG_SIZE`.
 */
struct AdaptiveBitpackConfig {
    DataType data_type;        ///< Signed element type (1B): INT16 / INT32.
    uint8_t  outlier_selection;///< 1 = cuSZp2 per-block plain/outlier selection.
    uint8_t  _pad[2];          ///< Must be zero.
    uint32_t block_size;       ///< Elements per logical block (reset period).
    uint64_t num_elements;     ///< Original element count (sizes the inverse output).

    AdaptiveBitpackConfig()
        : data_type(DataType::INT32), outlier_selection(0), _pad{},
          block_size(32), num_elements(0) {}
};
static_assert(sizeof(AdaptiveBitpackConfig) <= FZM_STAGE_CONFIG_SIZE,
              "AdaptiveBitpackConfig must fit in FZM_STAGE_CONFIG_SIZE");

/**
 * Per-block adaptive fixed-rate bit-plane coder — the cuSZp lossless back-end's
 * "plain" mode, as a modular stage.
 *
 * For each block of `block_size` signed elements it emits one rate byte (the bit
 * width of the largest magnitude in the block) followed, when that rate is
 * non-zero, by a per-block sign bitmap and that many bit-planes. Per-block byte
 * offsets are resolved with a plain device-wide exclusive scan (CUB) rather than
 * the cuSZp decoupled look-back scan, which is a fusion-time optimization left to
 * the downstream compiler.
 *
 * Pair with `QuantizerStage(linear) → LorenzoStage(setBlockSize)` for the cuSZp
 * pipeline; `block_size` typically matches the Lorenzo block (32) but need not.
 *
 * @note **Forward is graph-compatible; inverse is not.** The forward path's
 *       data-dependent compressed-size readback (cuSZp's `cmpSize` D2H) is
 *       deferred to `postStreamSync()` — run after the launch, outside any
 *       capture window — so `execute()` enqueues only stream-ordered device
 *       work. Per-block cost/offset scratch is kept persistent so the readback
 *       can happen post-sync and no allocation occurs during graph replay
 *       (mirrors `RZEStage`'s forward). The inverse keeps a per-execute layout
 *       and stays out of capture.
 *
 * @note **Prior work:** the per-block fixed-rate bit-plane encoding is the cuSZp
 *       lossless scheme (Yafan Huang et al., SC'23/SC'24, BSD-3-Clause). This
 *       stage is a **direct port of the cuSZp encode/decode kernel logic**; the
 *       byte-granular layout, CUB offset scan, and FZM/MemoryPool scaffolding are
 *       FZGPUModules code. The cuSZp BSD-3-Clause copyright is reproduced in
 *       `THIRD_PARTY.md`. See also `memory/cuszp_stages.md`.
 *
 * @tparam T  Signed element type: `int16_t` or `int32_t`.
 */
template<typename T>
class AdaptiveBitpackStage : public Stage {
    static_assert(std::is_same_v<T, int16_t> || std::is_same_v<T, int32_t>,
                  "AdaptiveBitpackStage: T must be int16_t or int32_t.");
public:
    AdaptiveBitpackStage() = default;
    ~AdaptiveBitpackStage() override;

    // ── Stage control ──────────────────────────────────────────────────────
    void setInverse(bool inv) override { is_inverse_ = inv; }
    bool isInverse() const override    { return is_inverse_; }
    // Forward (compress) is graph-capturable: execute() enqueues only
    // device-side work and the data-dependent compressed-size readback is
    // deferred to postStreamSync() (run after the launch, outside any capture
    // window).  The inverse path keeps a per-execute layout and is left out of
    // graph capture, mirroring RZEStage.
    bool isGraphCompatible() const override { return !is_inverse_; }

    /// Elements per logical block (the fixed-rate granularity). Must be in
    /// [1, 1024]. cuSZp uses 32.
    void setBlockSize(uint32_t n) {
        if (n == 0 || n > 1024)
            throw std::invalid_argument(
                "AdaptiveBitpackStage::setBlockSize: n must be in [1, 1024], got "
                + std::to_string(n));
        block_size_ = n;
    }
    uint32_t getBlockSize() const { return block_size_; }

    /// Enable cuSZp2 per-block plain/outlier selection: each block may instead
    /// store element 0 as a raw 1..sizeof(T)-byte outlier and pack only the rest,
    /// whichever is smaller. Helps non-sparse, high-smoothness data (the first
    /// element of each block is delta-vs-0, i.e. a full magnitude). Off by default.
    void setOutlierSelection(bool enable) { outlier_selection_ = enable; }
    bool getOutlierSelection() const { return outlier_selection_; }

    // ── Execution ──────────────────────────────────────────────────────────
    void execute(
        fz::stream_t stream,
        MemoryPool* pool,
        const std::vector<void*>& inputs,
        const std::vector<void*>& outputs,
        const std::vector<size_t>& sizes
    ) override;

    /// Forward only: read the scanned payload length (D2H of two tail words from
    /// the persistent cost/offset scratch) once the stream is idle, and finalize
    /// actual_output_size_.  Kept out of execute() so the forward path stays
    /// graph-capturable (no host sync inside the capture window).
    void postStreamSync(fz::stream_t stream) override;

    // ── Metadata ───────────────────────────────────────────────────────────
    std::string getName() const override { return "AdaptiveBitpack"; }
    size_t getNumInputs()  const override { return 1; }
    size_t getNumOutputs() const override { return 1; }

    std::vector<size_t> estimateOutputSizes(
        const std::vector<size_t>& input_sizes
    ) const override;

    /// Transient pool scratch held during execute() — two uint32 per-block arrays
    /// (cost + offset) plus the CUB exclusive-scan temp storage. Reported so the
    /// pipeline sizes the pool (PREALLOCATE) / peak (MINIMAL) for it and per-stage
    /// memory accounting reflects this stage. Defined in the .cu (needs CUB).
    size_t estimateScratchBytes(
        const std::vector<size_t>& input_sizes
    ) const override;

    std::unordered_map<std::string, size_t>
    getActualOutputSizesByName() const override {
        return {{"output", actual_output_size_}};
    }
    size_t getActualOutputSize(int index) const override {
        return (index == 0) ? actual_output_size_ : 0;
    }

    uint16_t getStageTypeId() const override {
        return static_cast<uint16_t>(StageType::ADAPTIVE_BITPACK);
    }

    // Forward: signed codes -> uint8 archive. Inverse: archive -> signed codes.
    uint8_t getOutputDataType(size_t) const override {
        return static_cast<uint8_t>(is_inverse_ ? getElementDataType()
                                                 : DataType::UINT8);
    }
    uint8_t getInputDataType(size_t) const override {
        return static_cast<uint8_t>(is_inverse_ ? DataType::UINT8
                                                 : getElementDataType());
    }

    // ── Serialization ──────────────────────────────────────────────────────
    size_t serializeHeader(size_t, uint8_t* buf, size_t max_size) const override {
        if (max_size < sizeof(AdaptiveBitpackConfig)) return 0;
        AdaptiveBitpackConfig cfg;
        cfg.data_type         = getElementDataType();
        cfg.outlier_selection = outlier_selection_ ? uint8_t{1} : uint8_t{0};
        cfg.block_size        = block_size_;
        cfg.num_elements      = static_cast<uint64_t>(num_elements_);
        std::memcpy(buf, &cfg, sizeof(cfg));
        return sizeof(cfg);
    }
    void deserializeHeader(const uint8_t* buf, size_t size) override {
        if (size < sizeof(AdaptiveBitpackConfig))
            throw std::runtime_error("AdaptiveBitpackStage: header too small");
        AdaptiveBitpackConfig cfg;
        std::memcpy(&cfg, buf, sizeof(cfg));
        block_size_        = cfg.block_size ? cfg.block_size : 32u;
        num_elements_      = static_cast<size_t>(cfg.num_elements);
        outlier_selection_ = (cfg.outlier_selection != 0);
    }
    size_t getMaxHeaderSize(size_t) const override {
        return sizeof(AdaptiveBitpackConfig);
    }

    void saveState() override {
        saved_block_size_       = block_size_;
        saved_num_elements_     = num_elements_;
        saved_actual_size_      = actual_output_size_;
        saved_outlier_select_   = outlier_selection_;
    }
    void restoreState() override {
        block_size_         = saved_block_size_;
        num_elements_       = saved_num_elements_;
        actual_output_size_ = saved_actual_size_;
        outlier_selection_  = saved_outlier_select_;
    }

    size_t getNumElements() const { return num_elements_; }

private:
    bool     is_inverse_         = false;
    uint32_t block_size_         = 32;
    bool     outlier_selection_  = false;
    size_t   num_elements_       = 0;
    size_t   actual_output_size_ = 0;

    // Forward-path persistent scratch (kept alive across execute() so the
    // compressed-size readback can be deferred to postStreamSync(), and so no
    // allocation happens inside a captured graph replay). Grown lazily when a
    // larger input is seen; freed in the destructor. Pool-managed (persistent).
    uint32_t*   d_cost_          = nullptr;  ///< per-block payload byte cost
    uint32_t*   d_offset_        = nullptr;  ///< exclusive scan of d_cost_
    size_t      scratch_blocks_  = 0;        ///< capacity of d_cost_/d_offset_ in blocks
    MemoryPool* scratch_pool_    = nullptr;  ///< owner of d_cost_/d_offset_
    size_t      fwd_num_blocks_  = 0;        ///< block count of the last forward encode
    size_t      fwd_meta_region_ = 0;        ///< metadata-region bytes of the last forward encode

    uint32_t saved_block_size_   = 32;
    bool     saved_outlier_select_ = false;
    size_t   saved_num_elements_ = 0;
    size_t   saved_actual_size_  = 0;

    static DataType getElementDataType() {
        if (std::is_same<T, int16_t>::value) return DataType::INT16;
        return DataType::INT32;
    }
};

extern template class AdaptiveBitpackStage<int16_t>;
extern template class AdaptiveBitpackStage<int32_t>;

} // namespace fz
