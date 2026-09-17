#pragma once

/**
 * @file quant_adaptive_lorenzo_stage.h
 * @brief AdaptiveLorenzo with the upstream linear Quantizer folded into its
 *        forward kernel (the "M1" partial fusion). Its own module directory;
 *        reuses AdaptiveLorenzoStage's machinery from the header below.
 */

#include "fused/adaptive_lorenzo/adaptive_lorenzo_stage.h"
#include "fused/lorenzo_quant/lorenzo_quant.h"  // ErrorBoundMode, resolveApproxRelMode
#include "stage/stage.h"
#include "fzm_format.h"
#include "backend/types.h"
#include "fused/common/data_type_of.h"
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace fz {

/// Serialized config for FusedQuantAdaptiveLorenzoStage, stored in
/// FZMBufferEntry.stage_config. `abs_error_bound_f64` is the resolved absolute
/// bound the forward pass actually quantized against (ABS: config_.error_bound
/// unchanged; NOA/PREL: eb * value_base) — the inverse reads it back to
/// dequant with `ebx2 = 2 * abs_error_bound_f64`, exactly mirroring
/// LorenzoQuantConfig's own error_bound_f64 field and the reasons given there
/// for keeping it double-precision (a narrow float copy of a NOA-resolved
/// bound already carries ~6e-08 relative error before the quantizer sees it).
struct FusedQuantAdaptiveLorenzoConfig {
    uint8_t  coder_block_size;  ///< Fixed at 32.
    uint8_t  blocks_per_tile;
    uint8_t  enable_order2;
    uint8_t  enable_centering;
    uint8_t  eb_mode;           ///< ErrorBoundMode cast to uint8_t.
    uint8_t  reserved[3];
    uint32_t num_elements;
    double   abs_error_bound_f64;
    double   value_base_f64;    ///< NOA/PREL scan result; 0 for ABS.
    float    user_error_bound;  ///< Original config_.error_bound, narrow copy (debug only).

    FusedQuantAdaptiveLorenzoConfig()
        : coder_block_size(32), blocks_per_tile(8), enable_order2(1),
          enable_centering(1), eb_mode(0), reserved{0, 0, 0}, num_elements(0),
          abs_error_bound_f64(0.0), value_base_f64(0.0), user_error_bound(0.0f) {}
};
static_assert(sizeof(FusedQuantAdaptiveLorenzoConfig) <= FZM_STAGE_CONFIG_SIZE,
              "FusedQuantAdaptiveLorenzoConfig must fit in FZM_STAGE_CONFIG_SIZE");

/**
 * AdaptiveLorenzo with the upstream linear Quantizer folded into its forward
 * kernel — the "M1" partial fusion (see FZGPUModules memory
 * `generic_fusion_plan.md`'s ARCHITECTURAL TENSION section, approach C, and
 * `quant_al_partial_fusion_probe.md` for the validated probe this promotes to
 * a real stage). Deletes the separate `QuantizerStage` kernel and its `codes`
 * DRAM round-trip on **compress only** — decompress still runs two kernels
 * (the existing cross-warp inverse scan, then a plain elementwise dequant),
 * matching every other fusion result in this codebase (compress-side only).
 *
 * Takes raw `float` input directly (no upstream Quantizer stage in the
 * pipeline) and owns its own error-bound resolution — `ABS`/`NOA`/`PREL`,
 * see `ErrorBoundMode` in `fused/lorenzo_quant/lorenzo_quant.h`, reusing the
 * exact same `computeValueBase()` scan `LorenzoQuantStage`/`QuantizerStage`
 * use. Quantization is the simple unbounded linear form (`round(x * ebx2_r)`,
 * matching `QuantizerStage`'s `linear_mode=true, linear_high_precision=false`
 * path bit-for-bit) — **no overflow guard**, unlike `QuantizerStage`: this is
 * an explicit, narrower scope for an opt-in fused stage, not a regression in
 * the staged path's safety net. Use `AdaptiveLorenzoStage` + a separate
 * `QuantizerStage` instead if overflow protection matters for your data.
 *
 * Forward outputs (compression):
 * - [0] `output` — residuals for the selected variant (`T`, one per element)
 * - [1] `modes`  — one byte per tile: bit 0 = order 2, bit 1 = centering
 * - [2] `means`  — one `T` per tile (meaningful only where bit 1 is set)
 *
 * Inverse (decompression): takes the three forward outputs, reconstructs the
 * original `float` data directly (no downstream `QuantizerStage` needed).
 *
 * @tparam T  Signed integer residual/code type: currently int32_t only.
 */
template<typename T = int32_t>
class FusedQuantAdaptiveLorenzoStage : public Stage {
    static_assert(std::is_integral<T>::value && std::is_signed<T>::value,
                  "FusedQuantAdaptiveLorenzoStage requires a signed integer residual type");
public:
    struct Config {
        uint32_t coder_block_size = 32;
        uint32_t blocks_per_tile  = 8;
        bool enable_order2    = true;
        bool enable_centering = true;
        /// Error bound (interpretation set by `eb_mode`).
        double error_bound = 1e-3;
        /// `ABS`, `NOA`, or `PREL`. `REL` is accepted as a deprecated alias for
        /// `PREL` (see `resolveApproxRelMode`) — this stage has no exact
        /// per-element relative path, same limitation as `LorenzoQuantStage`.
        ErrorBoundMode eb_mode = ErrorBoundMode::ABS;
        /// Pre-computed value_range (NOA) or max(|data|) (PREL) to skip the
        /// device scan. Leave at 0 to auto-compute.
        float precomputed_value_base = 0.0f;
        Config() = default;
    };

    FusedQuantAdaptiveLorenzoStage() { validate(); }
    explicit FusedQuantAdaptiveLorenzoStage(const Config& config) : config_(config) { validate(); }

    void setInverse(bool inv) override { is_inverse_ = inv; }
    bool isInverse() const override    { return is_inverse_; }

    uint32_t getTileSize() const {
        return config_.coder_block_size * config_.blocks_per_tile;
    }

    void setErrorBound(double eb) { config_.error_bound = eb; }
    void setErrorBoundMode(ErrorBoundMode mode) {
        config_.eb_mode = resolveApproxRelMode(mode, "FusedQuantAdaptiveLorenzoStage");
    }
    void setValueBase(float value_base) { config_.precomputed_value_base = value_base; }
    double getErrorBound() const { return config_.error_bound; }
    ErrorBoundMode getErrorBoundMode() const { return config_.eb_mode; }
    /// The absolute bound the forward pass actually quantized against, after
    /// NOA/PREL conversion. Valid only after compress(); 0 before that.
    double getComputedAbsErrorBound() const { return computed_abs_eb_; }

    /// Bind an exact additive fixed-rate policy exposed by a directly connected
    /// AdaptiveBitpack stage. See AdaptiveLorenzoStage's identical method.
    bool bindDownstreamEncodingOracle(const EncodingOracleDecl& decl) override {
        if (!decl.valid() || !decl.additive ||
            (decl.kind != EncodingOracleKind::PlainFixedRateBitpack &&
             decl.kind != EncodingOracleKind::AdaptiveFixedRateBitpack) ||
            decl.input_data_type != static_cast<uint8_t>(getElementDataType()) ||
            decl.unit_elems != config_.coder_block_size) {
            return false;
        }
        bound_oracle_ = decl;
        has_bound_oracle_ = true;
        return true;
    }

    bool hasBoundEncodingOracle() const { return has_bound_oracle_; }
    EncodingOracleKind getBoundEncodingOracleKind() const {
        return has_bound_oracle_ ? bound_oracle_.kind
                                 : EncodingOracleKind::PlainFixedRateBitpack;
    }

    FusionSpec getFusionSpec() const override {
        if (is_inverse_ || !has_bound_oracle_) return {};
        return FusionSpec{FusionAccess::TileSelector, getTileSize(),
                          config_.coder_block_size};
    }

    std::vector<FusedAuxOutputDecl> getFusedAuxOutputs() const override {
        if (!getFusionSpec().fusable()) return {};
        return {
            FusedAuxOutputDecl{1, "modes", FusedAuxSizeKind::FixedBitsPerUnit,
                               static_cast<uint8_t>(DataType::UINT8), getTileSize(),
                               2u, 0u},
            FusedAuxOutputDecl{2, "means", FusedAuxSizeKind::CompactedElements,
                               static_cast<uint8_t>(getElementDataType()), getTileSize(),
                               0u, 1u},
        };
    }

    void execute(
        fz::stream_t stream,
        MemoryPool* pool,
        const std::vector<void*>& inputs,
        const std::vector<void*>& outputs,
        const std::vector<size_t>& sizes
    ) override;

    std::string getName() const override { return "FusedQuantAdaptiveLorenzo"; }
    size_t getNumInputs()  const override { return is_inverse_ ? 3 : 1; }
    size_t getNumOutputs() const override { return is_inverse_ ? 1 : 3; }

    std::vector<std::string> getOutputNames() const override {
        return {"output", "modes", "means"};
    }

    ~FusedQuantAdaptiveLorenzoStage() override { releaseScratch(); }

    void postStreamSync(fz::stream_t stream) override;

    /// The `means` length is data-dependent (postStreamSync trim) and the
    /// forward pass does a D2H for NOA/PREL bound resolution — not graph-safe,
    /// same as AdaptiveLorenzoStage.
    bool isGraphCompatible() const override { return false; }

    size_t estimateScratchBytes(
        const std::vector<size_t>& input_sizes
    ) const override;

    std::vector<size_t> estimateOutputSizes(
        const std::vector<size_t>& input_sizes
    ) const override {
        if (input_sizes.empty()) return is_inverse_ ? std::vector<size_t>{0}
                                                     : std::vector<size_t>{0, 0, 0};
        if (is_inverse_) return {input_sizes[0] / sizeof(T) * sizeof(float)};
        const size_t n     = input_sizes[0] / sizeof(float);
        const size_t tiles = numTiles(n);
        return {n * sizeof(T), (tiles + 3) / 4, tiles * sizeof(T)};
    }

    void saveState() override    { saved_output_sizes_ = actual_output_sizes_; }
    void restoreState() override {
        if (!saved_output_sizes_.empty()) actual_output_sizes_ = saved_output_sizes_;
    }

    std::unordered_map<std::string, size_t>
    getActualOutputSizesByName() const override {
        auto names = getOutputNames();
        std::unordered_map<std::string, size_t> r;
        for (size_t i = 0; i < names.size() && i < actual_output_sizes_.size(); ++i)
            r[names[i]] = actual_output_sizes_[i];
        return r;
    }

    size_t getActualOutputSize(int index) const override {
        return (index >= 0 && index < static_cast<int>(actual_output_sizes_.size()))
            ? actual_output_sizes_[index] : 0;
    }

    void setFusedSideOutput(int output_index, size_t bytes) override {
        if (actual_output_sizes_.size() < 3) actual_output_sizes_.resize(3, 0);
        if (output_index == 1 || output_index == 2)
            actual_output_sizes_[static_cast<size_t>(output_index)] = bytes;
    }

    uint16_t getStageTypeId() const override {
        return static_cast<uint16_t>(StageType::FUSED_QUANT_ADAPTIVE_LORENZO);
    }

    uint8_t getOutputDataType(size_t output_index) const override {
        if (is_inverse_) return static_cast<uint8_t>(DataType::FLOAT32);
        return static_cast<uint8_t>(output_index == 1 ? DataType::UINT8
                                                       : getElementDataType());
    }
    uint8_t getInputDataType(size_t input_index) const override {
        if (!is_inverse_) return static_cast<uint8_t>(DataType::FLOAT32);
        return static_cast<uint8_t>(input_index == 1 ? DataType::UINT8
                                                      : getElementDataType());
    }

    size_t serializeHeader(size_t /*output_index*/, uint8_t* buf, size_t max_size) const override {
        if (max_size < sizeof(FusedQuantAdaptiveLorenzoConfig))
            throw std::runtime_error("FusedQuantAdaptiveLorenzoStage: header buffer too small");
        FusedQuantAdaptiveLorenzoConfig cfg;
        cfg.coder_block_size    = static_cast<uint8_t>(config_.coder_block_size);
        cfg.blocks_per_tile     = static_cast<uint8_t>(config_.blocks_per_tile);
        cfg.enable_order2       = config_.enable_order2 ? 1u : 0u;
        cfg.enable_centering    = config_.enable_centering ? 1u : 0u;
        cfg.eb_mode             = static_cast<uint8_t>(config_.eb_mode);
        cfg.num_elements        = static_cast<uint32_t>(num_elements_);
        cfg.abs_error_bound_f64 = computed_abs_eb_;
        cfg.value_base_f64      = computed_value_base_;
        cfg.user_error_bound    = static_cast<float>(config_.error_bound);
        std::memcpy(buf, &cfg, sizeof(cfg));
        return sizeof(cfg);
    }

    void deserializeHeader(const uint8_t* buf, size_t size) override {
        if (size < sizeof(FusedQuantAdaptiveLorenzoConfig))
            throw std::runtime_error("FusedQuantAdaptiveLorenzoStage: header too small");
        FusedQuantAdaptiveLorenzoConfig cfg;
        std::memcpy(&cfg, buf, sizeof(cfg));
        config_.coder_block_size = cfg.coder_block_size;
        config_.blocks_per_tile  = cfg.blocks_per_tile;
        config_.enable_order2    = (cfg.enable_order2 != 0);
        config_.enable_centering = (cfg.enable_centering != 0);
        config_.eb_mode          = static_cast<ErrorBoundMode>(cfg.eb_mode);
        num_elements_            = cfg.num_elements;
        computed_abs_eb_         = cfg.abs_error_bound_f64;
        computed_value_base_     = cfg.value_base_f64;
        validate();
    }

    size_t getMaxHeaderSize(size_t /*output_index*/) const override {
        return sizeof(FusedQuantAdaptiveLorenzoConfig);
    }

private:
    Config config_;
    EncodingOracleDecl bound_oracle_;
    bool has_bound_oracle_ = false;
    bool   is_inverse_ = false;
    size_t num_elements_ = 0;
    /// Resolved absolute bound (ABS: config_.error_bound; NOA/PREL: eb *
    /// value_base). 0 before the first forward execute() or header deserialize.
    double computed_abs_eb_ = 0.0;
    double computed_value_base_ = 0.0;
    std::vector<size_t> actual_output_sizes_{0, 0, 0};
    std::vector<size_t> saved_output_sizes_;

    // Forward scratch — identical role to AdaptiveLorenzoStage's, reused for
    // the inverse's offset recomputation too (same pattern as that class).
    uint8_t*    d_modes_dense_ = nullptr;
    T*          d_means_dense_ = nullptr;
    uint32_t*   d_flags_       = nullptr;
    uint32_t*   d_offsets_     = nullptr;
    size_t      scratch_tiles_ = 0;
    MemoryPool* scratch_pool_  = nullptr;
    size_t      pending_tiles_ = 0;

    // Inverse-only scratch: the reconstructed integer codes before the final
    // dequant pass (this stage's inverse always ends in float, unlike
    // AdaptiveLorenzoStage's, which stops at T).
    T*          d_inverse_codes_       = nullptr;
    size_t      inverse_codes_elems_   = 0;
    MemoryPool* inverse_scratch_pool_  = nullptr;

    size_t ensureScratch(size_t num_tiles, MemoryPool* pool, fz::stream_t stream);
    size_t ensureInverseScratch(size_t n, MemoryPool* pool, fz::stream_t stream);
    void   releaseScratch();

    size_t numTiles(size_t n) const {
        const size_t t = getTileSize();
        return (n + t - 1) / t;
    }

    void validate() const {
        if (config_.coder_block_size != 32)
            throw std::invalid_argument(
                "FusedQuantAdaptiveLorenzoStage: coder_block_size must be 32");
        if (config_.blocks_per_tile < 1 || config_.blocks_per_tile > 32)
            throw std::invalid_argument(
                "FusedQuantAdaptiveLorenzoStage: blocks_per_tile must be in [1, 32], got "
                + std::to_string(config_.blocks_per_tile));
    }

    static DataType getElementDataType() { return fused::dataTypeOf<T>(); }
};

extern template class FusedQuantAdaptiveLorenzoStage<int32_t>;

}  // namespace fz
