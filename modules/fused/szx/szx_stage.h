#pragma once

/**
 * @file szx_stage.h
 * @brief SZx — ultrafast error-bounded lossy compressor, as a fused stage.
 *
 * Reference: "SZx: an Ultra-fast Error-bounded Lossy Compressor for Scientific
 * Datasets" (Yu, Di, et al.). This is a **from-the-paper** reimplementation of
 * the SZx forward/inverse pipeline as a single fused FZGPUModules stage; no SZx
 * source is vendored. Third-party attribution belongs in THIRD_PARTY.md.
 *
 * SZx is a *whole compressor*: it consumes raw floats and emits a self-describing
 * byte archive. Unlike the cuSZ-style chain (Lorenzo → Quantizer → coder), SZx
 * has no Lorenzo prediction and no entropy coder — that is what makes it fast.
 * Its one distinguishing move is **per-block constant/non-constant
 * classification**, which is why it lives here as a fused stage rather than being
 * composed from existing stages.
 *
 * Forward:  float[]/double[]  → uint8[] archive
 * Inverse:  uint8[] archive   → float[]/double[]  (error-bounded approximation)
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
 * Error-bound interpretation. SZx quantizes residuals against one global
 * tolerance per block, so it supports ABS and value-range-relative (NOA) bounds
 * only — no exact per-element REL path (use QuantizerStage for that).
 */
enum class SZxErrorMode : uint8_t { ABS = 0, NOA = 2 };

/**
 * Serialized SZx configuration (FZMBufferEntry.stage_config). Fits in 128 B.
 */
struct SZxConfig {
    DataType data_type;        ///< FLOAT32 / FLOAT64 (1B).
    uint8_t  eb_mode;          ///< SZxErrorMode cast to uint8_t.
    uint8_t  _pad[2];          ///< Must be zero.
    uint32_t block_size;       ///< Elements per block (SZx default 128).
    uint64_t num_elements;     ///< Original element count (sizes the inverse).
    double   error_bound;      ///< Absolute bound after mode conversion (f64-safe).
    double   value_base;       ///< value_range used for NOA→ABS conversion; else 0.

    SZxConfig()
        : data_type(DataType::FLOAT32), eb_mode(0), _pad{},
          block_size(128), num_elements(0), error_bound(0.0), value_base(0.0) {}
};
static_assert(sizeof(SZxConfig) <= FZM_STAGE_CONFIG_SIZE,
              "SZxConfig must fit in FZM_STAGE_CONFIG_SIZE");

/**
 * SZx ultrafast error-bounded compressor.
 *
 * Per block of `block_size` elements the forward pass:
 *   1. scans block min/max;
 *   2. **classifies** the block as *constant* when `max - min <= 2*eb`
 *      (the whole block is representable by one value within the bound) or
 *      *non-constant* otherwise — a 2-bit type code per block;
 *   3. **constant** blocks emit only a block reference value (the midpoint of the
 *      block's min/max range, stored at T precision — this minimises worst-case
 *      reconstruction error, unlike the arithmetic mean);
 *   4. **non-constant** blocks subtract the same block reference, quantize the
 *      residuals to fixed-length integers within `[-2^b, 2^b)` where `b` is the
 *      block's required bit width, and bit-pack them (no Huffman).
 *
 * The archive is: [meta region: 2-bit type codes + per-block bit widths] followed
 * by [payload region: reference values + packed residuals]. Output size is data
 * dependent (constant blocks are ~one value), so `estimateOutputSizes()` returns
 * a safe upper bound and `postStreamSync()` reads the true size back.
 *
 * @tparam T  float or double.
 */
template<typename T>
class SZxStage : public Stage {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "SZxStage: T must be float or double.");
public:
    SZxStage() = default;
    ~SZxStage() override;

    // ── Stage control ──────────────────────────────────────────────────────
    void setInverse(bool inv) override { is_inverse_ = inv; }
    bool isInverse() const override    { return is_inverse_; }
    // Forward defers its data-dependent compressed-size readback to
    // postStreamSync() (see AdaptiveBitpackStage), so ABS forward is
    // graph-capturable. NOA forward needs a range reduce + host read inside
    // execute(), so it is not; the inverse keeps a per-execute layout either way.
    bool isGraphCompatible() const override {
        return !is_inverse_ && eb_mode_ != SZxErrorMode::NOA;
    }

    void setBlockSize(uint32_t n) {
        if (n == 0 || n > 4096)
            throw std::invalid_argument(
                "SZxStage::setBlockSize: n must be in [1, 4096], got "
                + std::to_string(n));
        block_size_ = n;
    }
    uint32_t getBlockSize() const { return block_size_; }

    void   setErrorBound(double eb) { user_eb_ = eb; }
    double getErrorBound() const    { return user_eb_; }
    void   setErrorMode(SZxErrorMode m) { eb_mode_ = m; }
    SZxErrorMode getErrorMode() const   { return eb_mode_; }

    /// Fraction of blocks classified constant on the last forward encode — a
    /// cheap compressibility probe, reported through getRunNotes().
    double getConstantBlockFraction() const { return const_block_frac_; }

    // ── Execution ──────────────────────────────────────────────────────────
    void execute(fz::stream_t stream, MemoryPool* pool,
                 const std::vector<void*>& inputs,
                 const std::vector<void*>& outputs,
                 const std::vector<size_t>& sizes) override;

    void postStreamSync(fz::stream_t stream) override;

private:
    /// Resolve the requested error mode to an absolute bound. ABS returns
    /// user_eb directly; NOA reduces the data range (device) and returns
    /// user_eb * range, setting value_base_. Defined in the .cu (needs CUB).
    double resolveAbsEb(fz::stream_t stream, MemoryPool* pool, const T* d_in, size_t n);

public:

    // ── Metadata ───────────────────────────────────────────────────────────
    std::string getName() const override { return "SZx"; }
    size_t getNumInputs()  const override { return 1; }
    size_t getNumOutputs() const override { return 1; }

    std::vector<size_t> estimateOutputSizes(
        const std::vector<size_t>& input_sizes) const override;
    size_t estimateScratchBytes(
        const std::vector<size_t>& input_sizes) const override;

    std::unordered_map<std::string, size_t>
    getActualOutputSizesByName() const override {
        return {{"output", actual_output_size_}};
    }
    size_t getActualOutputSize(int index) const override {
        return (index == 0) ? actual_output_size_ : 0;
    }
    std::vector<std::string> getRunNotes() const override;

    uint16_t getStageTypeId() const override {
        return static_cast<uint16_t>(StageType::SZX);
    }
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
        if (max_size < sizeof(SZxConfig)) return 0;
        SZxConfig cfg;
        cfg.data_type    = getElementDataType();
        cfg.eb_mode      = static_cast<uint8_t>(eb_mode_);
        cfg.block_size   = block_size_;
        cfg.num_elements = static_cast<uint64_t>(num_elements_);
        cfg.error_bound  = abs_eb_;
        cfg.value_base   = value_base_;
        std::memcpy(buf, &cfg, sizeof(cfg));
        return sizeof(cfg);
    }
    void deserializeHeader(const uint8_t* buf, size_t size) override {
        if (size < sizeof(SZxConfig))
            throw std::runtime_error("SZxStage: header too small");
        SZxConfig cfg;
        std::memcpy(&cfg, buf, sizeof(cfg));
        block_size_   = cfg.block_size ? cfg.block_size : 128u;
        num_elements_ = static_cast<size_t>(cfg.num_elements);
        eb_mode_      = static_cast<SZxErrorMode>(cfg.eb_mode);
        abs_eb_       = cfg.error_bound;
        value_base_   = cfg.value_base;
    }
    size_t getMaxHeaderSize(size_t) const override { return sizeof(SZxConfig); }

    void saveState() override {
        saved_ = {block_size_, num_elements_, actual_output_size_, abs_eb_, value_base_};
    }
    void restoreState() override {
        block_size_         = saved_.block_size;
        num_elements_       = saved_.num_elements;
        actual_output_size_ = saved_.actual_size;
        abs_eb_             = saved_.abs_eb;
        value_base_         = saved_.value_base;
    }

    size_t getNumElements() const { return num_elements_; }

private:
    bool         is_inverse_        = false;
    uint32_t     block_size_        = 128;
    SZxErrorMode eb_mode_           = SZxErrorMode::ABS;
    double       user_eb_           = 1e-3;  ///< as requested by the user
    double       abs_eb_            = 0.0;    ///< resolved absolute bound (post scan)
    double       value_base_        = 0.0;    ///< value_range for NOA
    size_t       num_elements_      = 0;
    size_t       actual_output_size_= 0;
    double       const_block_frac_  = 0.0;

    // Forward-path persistent scratch, kept alive across execute() so the
    // compressed-size readback can be deferred to postStreamSync() (mirrors
    // AdaptiveBitpackStage). Grown lazily; freed in the destructor.
    uint32_t*   d_block_cost_   = nullptr;  ///< per-block payload byte cost
    uint32_t*   d_block_offset_ = nullptr;  ///< exclusive scan of d_block_cost_
    size_t      scratch_blocks_ = 0;
    MemoryPool* scratch_pool_   = nullptr;
    size_t      fwd_num_blocks_ = 0;
    size_t      fwd_meta_bytes_ = 0;

    struct Saved { uint32_t block_size; size_t num_elements; size_t actual_size;
                   double abs_eb; double value_base; };
    Saved saved_{128, 0, 0, 0.0, 0.0};

    static DataType getElementDataType() {
        return std::is_same<T, float>::value ? DataType::FLOAT32 : DataType::FLOAT64;
    }
};

extern template class SZxStage<float>;
extern template class SZxStage<double>;

} // namespace fz
