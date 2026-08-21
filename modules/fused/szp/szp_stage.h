#pragma once

/**
 * @file szp_stage.h
 * @brief SZp (a.k.a. fZ-light) — extreme-fast error-bounded compressor, as a
 *        fused stage.
 *
 * Reference: SZp / fZ-light (Huang, Di, et al., SC'24). The upstream CPU/OpenMP
 * reference at https://github.com/szcompressor/SZp is MIT-licensed; this stage
 * independently reimplements its forward/inverse and copies no upstream source.
 * The homomorphic-collectives variant is hZCCL. See `THIRD_PARTY.md`.
 *
 * SZp's inner loop is a *whole compressor* that decomposes almost exactly into
 * the FZGM chain `Quantizer(linear,ABS) → 1-D Lorenzo/diff → AdaptiveBitpack`
 * (its per-block fixed-length residual packing IS AdaptiveBitpack's plain mode).
 * This stage exists for (a) SZp byte-format parity and (b) single-launch
 * throughput. For a pure-composition equivalent that needs NO new code, see
 * `examples/presets/szp_composed.toml`.
 *
 * Forward:  float[]/double[] → uint8[] archive
 * Inverse:  uint8[] archive  → float[]/double[]  (error-bounded approximation)
 *
 * NOTE: this stage does NOT implement hZCCL's compressed-domain arithmetic
 * (add/reduce on compressed buffers for collectives). That is a separate
 * capability — a `HomomorphicOp` interface, not a stage — and is out of scope
 * here. See notes in the wire-in checklist in the .cu.
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

enum class SZpErrorMode : uint8_t { ABS = 0, NOA = 2 };

/** Serialized SZp configuration (FZMBufferEntry.stage_config). Fits in 128 B. */
struct SZpConfig {
    DataType data_type;    ///< FLOAT32 / FLOAT64.
    uint8_t  eb_mode;      ///< SZpErrorMode.
    uint8_t  _pad[2];
    uint32_t block_size;   ///< Elements per block (SZp default 128).
    uint64_t num_elements;
    double   error_bound;  ///< Absolute bound after mode conversion (f64-safe).
    double   value_base;   ///< value_range for NOA; else 0.

    SZpConfig()
        : data_type(DataType::FLOAT32), eb_mode(0), _pad{},
          block_size(128), num_elements(0), error_bound(0.0), value_base(0.0) {}
};
static_assert(sizeof(SZpConfig) <= FZM_STAGE_CONFIG_SIZE,
              "SZpConfig must fit in FZM_STAGE_CONFIG_SIZE");

/**
 * SZp / fZ-light fused compressor.
 *
 * Per block of `block_size` elements the forward pass: (1) 1-D Lorenzo predicts
 * each element from its predecessor; (2) the residual is error-bound quantized
 * with step `2*eb`; (3) residual codes are fixed-length bit-packed with one
 * leading bit-width byte per block. No entropy coder, no outlier stream — the
 * fixed-length packing subsumes both. This is exactly the shape hZCCL relies on
 * so that compressed buffers can be operated on without full decompression.
 *
 * @tparam T  float or double.
 */
template<typename T>
class SZpStage : public Stage {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "SZpStage: T must be float or double.");
public:
    SZpStage() = default;
    ~SZpStage() override;

    void setInverse(bool inv) override { is_inverse_ = inv; }
    bool isInverse() const override    { return is_inverse_; }
    // ABS forward is graph-capturable (size readback deferred to
    // postStreamSync()); NOA forward needs a range reduce + host read inside
    // execute(), so it is not.
    bool isGraphCompatible() const override {
        return !is_inverse_ && eb_mode_ != SZpErrorMode::NOA;
    }

    void setBlockSize(uint32_t n) {
        if (n == 0 || n > 4096)
            throw std::invalid_argument("SZpStage::setBlockSize: n in [1,4096]");
        block_size_ = n;
    }
    uint32_t getBlockSize() const { return block_size_; }
    void   setErrorBound(double eb) { user_eb_ = eb; }
    double getErrorBound() const    { return user_eb_; }
    void   setErrorMode(SZpErrorMode m) { eb_mode_ = m; }
    SZpErrorMode getErrorMode() const   { return eb_mode_; }

    void execute(fz::stream_t stream, MemoryPool* pool,
                 const std::vector<void*>& inputs,
                 const std::vector<void*>& outputs,
                 const std::vector<size_t>& sizes) override;
    void postStreamSync(fz::stream_t stream) override;

private:
    /// Resolve the requested error mode to an absolute bound (ABS: user_eb;
    /// NOA: user_eb * device-reduced range, sets value_base_). Defined in .cu.
    double resolveAbsEb(fz::stream_t stream, MemoryPool* pool, const T* d_in, size_t n);
public:

    std::string getName() const override { return "SZp"; }
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

    uint16_t getStageTypeId() const override {
        return static_cast<uint16_t>(StageType::SZP);
    }
    uint8_t getOutputDataType(size_t) const override {
        return static_cast<uint8_t>(is_inverse_ ? getElementDataType()
                                                 : DataType::UINT8);
    }
    uint8_t getInputDataType(size_t) const override {
        return static_cast<uint8_t>(is_inverse_ ? DataType::UINT8
                                                 : getElementDataType());
    }

    size_t serializeHeader(size_t, uint8_t* buf, size_t max_size) const override {
        if (max_size < sizeof(SZpConfig)) return 0;
        SZpConfig cfg;
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
        if (size < sizeof(SZpConfig))
            throw std::runtime_error("SZpStage: header too small");
        SZpConfig cfg; std::memcpy(&cfg, buf, sizeof(cfg));
        block_size_   = cfg.block_size ? cfg.block_size : 128u;
        num_elements_ = static_cast<size_t>(cfg.num_elements);
        eb_mode_      = static_cast<SZpErrorMode>(cfg.eb_mode);
        abs_eb_       = cfg.error_bound;
        value_base_   = cfg.value_base;
    }
    size_t getMaxHeaderSize(size_t) const override { return sizeof(SZpConfig); }

    void saveState() override {
        saved_ = {block_size_, num_elements_, actual_output_size_, abs_eb_, value_base_};
    }
    void restoreState() override {
        block_size_ = saved_.block_size; num_elements_ = saved_.num_elements;
        actual_output_size_ = saved_.actual_size; abs_eb_ = saved_.abs_eb;
        value_base_ = saved_.value_base;
    }
    size_t getNumElements() const { return num_elements_; }

private:
    bool         is_inverse_        = false;
    uint32_t     block_size_        = 128;
    SZpErrorMode eb_mode_           = SZpErrorMode::ABS;
    double       user_eb_           = 1e-3;
    double       abs_eb_            = 0.0;
    double       value_base_        = 0.0;
    size_t       num_elements_      = 0;
    size_t       actual_output_size_= 0;

    uint32_t*   d_block_cost_   = nullptr;
    uint32_t*   d_block_offset_ = nullptr;
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

extern template class SZpStage<float>;
extern template class SZpStage<double>;

} // namespace fz
