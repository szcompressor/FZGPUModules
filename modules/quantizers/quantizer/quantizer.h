#pragma once

/**
 * @file quantizer.h
 * @brief Direct-value quantizer stage with error-bounded coding and lossless outlier fallback.
 */

#include "stage/stage.h"
#include "fzm_format.h"
#include "fused/lorenzo_quant/lorenzo_quant.h"  // for ErrorBoundMode
#include "fused/chunk_fusion/chunk_op_params.h" // shared fused-op Params (host packs)
#include "backend/types.h"
#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <cstring>
#include <limits>
#include <type_traits>

namespace fz {

/**
 * Serialized quantizer configuration stored in FZMBufferEntry.stage_config.
 * Written by `serializeHeader()`; read back by `deserializeHeader()`.
 * 72 bytes — fits within the 128-byte `FZM_STAGE_CONFIG_SIZE` limit.
 */
struct QuantizerConfig {
    /// @deprecated Narrow copy of the absolute EB, kept so old readers still parse.
    /// New code must read `abs_error_bound_f64` — see the f64 note on that field.
    float    abs_error_bound;   ///< Absolute EB after mode conversion (0 for REL).
    float    user_error_bound;  ///< Original user-specified EB.
    /// @deprecated Narrow copy; new code reads `value_base_f64`.
    float    value_base;        ///< value_range (NOA); 0 for ABS/REL.
    uint32_t quant_radius;      ///< Quantization radius.
    uint32_t num_elements;      ///< Total element count.
    uint32_t outlier_count;     ///< Actual number of outliers.
    DataType input_type;        ///< Original input type (1B).
    DataType code_type;         ///< Quantization code type (1B).
    uint8_t  eb_mode;           ///< ErrorBoundMode cast to uint8_t.
    uint8_t  zigzag_codes;      ///< 1 if ABS/NOA codes are zigzag-encoded.
    float    outlier_threshold; ///< ABS/NOA: |x| >= threshold → forced outlier (inf = disabled).
    uint8_t  inplace_outliers;  ///< 1 if outliers are encoded in-place in the codes array.
    uint8_t  linear_mode;       ///< 1 if linear/no-outlier mode (signed codes, no outlier ports).
    uint8_t  dither;            ///< 1 if "_R"-style dithered reconstruction is enabled (LC QUANT_*_R).
    uint8_t  linear_high_precision; ///< 1 if linear coordinates are evaluated in double precision.
    uint8_t  power_of_two_bound;    ///< 1 if the uniform absolute EB was rounded down to a power of two.
    uint8_t  _pad[3];           ///< Alignment padding (dither_seed needs 8-byte alignment) — must be zero.
    uint64_t dither_seed;       ///< Deterministic per-element dither seed; meaningful only when dither.
    float    dither_strength;   ///< Dither offset amplitude as a fraction of abs_eb, in (0,1]; meaningful only when dither.
    uint32_t _pad2;             ///< Alignment padding (the f64 fields need 8-byte alignment) — must be zero.
    /// Full-precision absolute EB. The float copy above cannot carry an f64
    /// bound: decode computes `recon = q * 2*abs_eb`, and for a field like
    /// S3D/N2 (q ~ 3.3e8) a 6e-08 *relative* error in abs_eb lands as a 4.4e-08
    /// absolute error — 40x the bound. Headers written before 2026-08-07 leave
    /// this 0 and the reader falls back to the float field.
    double   abs_error_bound_f64;
    double   value_base_f64;    ///< Full-precision value_base; 0 in pre-2026-08-07 headers.

    QuantizerConfig()
        : abs_error_bound(0.0f), user_error_bound(0.0f), value_base(0.0f),
          quant_radius(0), num_elements(0), outlier_count(0),
          input_type(DataType::FLOAT32), code_type(DataType::UINT16),
          eb_mode(0), zigzag_codes(0),
          outlier_threshold(std::numeric_limits<float>::infinity()),
          inplace_outliers(0), linear_mode(0), dither(0),
          linear_high_precision(0), power_of_two_bound(0), _pad{}, dither_seed(0),
          dither_strength(1.0f), _pad2(0),
          abs_error_bound_f64(0.0), value_base_f64(0.0) {}
};
static_assert(sizeof(QuantizerConfig) <= FZM_STAGE_CONFIG_SIZE,
              "QuantizerConfig must fit in FZM_STAGE_CONFIG_SIZE");
static_assert(sizeof(QuantizerConfig) == 72,
              "QuantizerConfig archive layout changed; add an explicit compatibility path");

/**
 * Direct-value quantizer with error-bounded coding and lossless outlier fallback.
 *
 * @note **Prior work:** ABS/NOA/REL quantization scheme, outlier handling, and
 *       log-space REL encoding follow the LC/PFPL framework (Burtscher et al.,
 *       BSD-3-Clause). See `THIRD_PARTY.md`.
 *
 * Unlike LorenzoQuantStage (which quantizes prediction *differences*), this stage
 * quantizes the input *values* directly.  It supports all three error-bound
 * modes:
 *
 *   ABS — absolute error bound:  |x - x_hat| <= eb
 *         Uniform quantization with step = 2*eb.
 *         Works with any TCode type.
 *
 *   NOA — norm-of-absolute (PFPL): abs_eb = eb * (max(data) - min(data))
 *         Scans the data once to find value_range, then falls through to ABS.
 *         Works with any TCode type.
 *
 *   PREL — pseudo-relative: abs_eb = eb * max(|data|), then falls through to
 *         ABS.  Supported here only so the mode is uniform across stages; on
 *         this stage REL is strictly better (exact, same cost class), so PREL
 *         is really for the predictor-fused stages that cannot do REL at all.
 *
 *   REL — pointwise relative error bound (PFPL exact definition):
 *             |x - x_hat| / |x| <= eb
 *         Implemented via log2-space quantization (see PFPL paper):
 *           bin = round(log2(|x|) / log2eb),  log2eb = 2 * log2(1 + eb)
 *           x_hat = sign(x) * 2^(bin * log2eb)
 *         Zeros, denormals, infinities and NaNs are stored losslessly as
 *         outliers.  Reconstruction is also verified against the exact bounds;
 *         if the fast log2/pow2 approximation causes a violation the value is
 *         stored losslessly instead.
 *
 *         NOTE: REL mode uses a 4-byte code per element (bit-packed: sign of x,
 *         sign of log_bin, magnitude of log_bin).  You must use a 4-byte code
 *         type: QuantizerStage<float, uint32_t>.  An exception is thrown at
 *         runtime if TCode is narrower and the required stored value overflows.
 *         For epsilon >= 0.01 with float32, uint16_t codes are sufficient in
 *         practice (max |log_bin| ≈ 4460 << 16383 max for uint16 REL).
 *
 * Outputs (compression mode, scatter path):
 *   [0] codes         — quantization codes (TCode[n])
 *   [1] outlier_vals  — original values at outlier positions (TInput[k])
 *   [2] outlier_idxs  — indices of outlier positions (uint32_t[k])
 *
 * The outlier *count* is not a DAG output port — it lives in a stage-private
 * 4-byte device scratch (allocated via `pool->allocatePersistentDevice` in
 * `onFinalize()`), is D2H'd in `postStreamSync()`, and is serialized into the
 * FZM stage header. The inverse path receives it as a `uint32_t` kernel
 * argument read from the deserialized header.
 *
 * Inplace mode (`setInplaceOutliers(true)`, ABS/NOA only) emits 1 output —
 * the codes array with raw float bits encoded in-place; no scatter buffers,
 * no count scratch.
 *
 * Inputs (decompression mode): same 3 buffers (or 1 in inplace mode) →
 *   reconstructed TInput[n].
 */
template<typename TInput = float, typename TCode = uint16_t>
class QuantizerStage : public Stage {
public:
    /** Construction parameters. */
    struct Config {
        double error_bound           = 1e-4;    ///< Error bound (interpretation set by `eb_mode`).
        int    quant_radius          = 32768;   ///< Quantization radius.
        float  outlier_capacity      = 0.05f;   ///< Fraction of input size reserved for outliers.
        ErrorBoundMode eb_mode       = ErrorBoundMode::ABS;
        /// Pre-computed value_base > 0 to skip the NOA data scan; 0 = auto.
        float precomputed_value_base = 0.0f;
        /// ABS/NOA: zigzag-encode codes before storage to improve compressibility.
        /// No effect in REL mode (log-space codes are already unsigned).
        bool  zigzag_codes           = false;
        /// ABS/NOA: |x| >= threshold → lossless outlier (LC reference `threshold`). Default: ∞.
        float outlier_threshold      = std::numeric_limits<float>::infinity();
        /// ABS/NOA: write outlier raw float bits in-place in the codes array.
        /// Removes the scatter buffers; inverse checks `(code >> 1) >= quant_radius`.
        /// Must NOT be used with REL mode.
        bool  inplace_outliers       = false;
        /// ABS/NOA: linear / no-outlier mode (cuSZp-style). Emits raw signed codes
        /// (q = round(x / 2·eb), stored two's-complement in TCode and *declared* as the
        /// signed DataType), with NO radius clamp, NO outlier ports, NO zigzag — a value
        /// outside TCode range is rejected, so size TCode wide enough (use uint32_t).
        /// Intended front-end for `→ LorenzoStage(blockSize=32) → AdaptiveBitpackStage`.
        /// Mutually exclusive with REL, inplace_outliers, and zigzag_codes.
        bool  linear_mode            = false;
        /// Linear mode only: evaluate x/(2*abs_eb) with a precomputed double
        /// reciprocal and double multiply. For non-power-of-two bounds the stage
        /// also tightens its internal bound enough to absorb final TInput
        /// reconstruction rounding, making the user bound a strict guarantee.
        bool  linear_high_precision  = false;
        /// Uniform ABS/NOA/PREL modes: round the resolved absolute half-bound
        /// downward to the nearest power of two. This is between 1x and 2x
        /// tighter than requested and makes scaling an exact binary exponent
        /// shift. Inspired by SLEEK (IPDPS 2026), Sec. III-A. REL is unsupported
        /// because it quantizes in log space.
        bool  power_of_two_bound     = false;
        /// ABS/NOA/REL: reconstruct to a deterministic pseudo-random point within
        /// the bin/error-bound interval instead of always the bin center (LC's
        /// QUANT_*_R vs. QUANT_*_0). Decorrelates reconstruction error from the
        /// signal at no extra storage cost — the offset is a pure function of
        /// (element index, dither_seed), reproduced identically on decode. Any
        /// element whose dithered reconstruction would violate the error bound
        /// is escalated to a lossless outlier instead (same mechanism used for
        /// out-of-radius values). Mutually exclusive with linear_mode and
        /// inplace_outliers (both lack a per-element outlier-escalation path).
        bool     dither              = false;
        /// Seed for the deterministic dither offset. Persisted in the serialized
        /// header so decode reproduces identical offsets with no extra storage.
        uint64_t dither_seed         = 0;
        /// Dither offset amplitude as a fraction of abs_eb (or, in REL mode, of
        /// half a log-bin width), in (0, 1]. 1.0 (default) matches LC's literal
        /// "_R" definition — offset spans the full bin — and empirically
        /// escalates ~25% of elements to lossless outliers for smooth data.
        /// Lower values trade decorrelation strength for fewer outliers; 0.0
        /// disables the offset entirely (bit-identical to dither=false).
        float    dither_strength     = 1.0f;

        Config() = default;
        Config(TInput eb, ErrorBoundMode mode = ErrorBoundMode::ABS,
               int radius = 32768, float outlier_cap = 0.05f)
            : error_bound(static_cast<double>(eb)), quant_radius(radius),
              outlier_capacity(outlier_cap), eb_mode(mode) {}
    };

    explicit QuantizerStage(const Config& config = Config());
    ~QuantizerStage() override;

    void execute(
        fz::stream_t stream,
        MemoryPool* pool,
        const std::vector<void*>& inputs,
        const std::vector<void*>& outputs,
        const std::vector<size_t>& sizes
    ) override;

    void postStreamSync(fz::stream_t stream) override;

    /// Pre-allocate the stage-private 4-byte outlier-count device scratch
    /// (via `pool->allocatePersistentDevice`) in PREALLOCATE mode. In MINIMAL
    /// mode this is deferred to the first compress execute(). Inplace mode
    /// skips the allocation entirely.
    void onFinalize(size_t estimated_inlen, MemoryPool* pool) override;

    size_t estimateDeviceFootprintBytes(size_t /*estimated_inlen*/) const override {
        return (isLinearMode() || isInplaceMode()) ? 0 : sizeof(uint32_t);
    }

    std::string getName() const override { return "Quantizer"; }

    size_t getNumInputs() const override {
        if (!is_inverse_) return 1;
        return (isLinearMode() || isInplaceMode()) ? 1 : 3;
    }
    size_t getNumOutputs() const override {
        if (is_inverse_) return 1;
        return (isLinearMode() || isInplaceMode()) ? 1 : 3;
    }

    std::vector<std::string> getOutputNames() const override {
        if (is_inverse_) return {"reconstructed"};
        if (isLinearMode() || isInplaceMode()) return {"codes"};
        return {"codes", "outlier_vals", "outlier_idxs"};
    }

    std::vector<size_t> estimateOutputSizes(
        const std::vector<size_t>& input_sizes
    ) const override;

    std::unordered_map<std::string, size_t> getActualOutputSizesByName() const override {
        auto names = getOutputNames();
        std::unordered_map<std::string, size_t> result;
        for (size_t i = 0; i < names.size() && i < actual_output_sizes_.size(); i++)
            result[names[i]] = actual_output_sizes_[i];
        return result;
    }
    size_t getActualOutputSize(int index) const override {
        return (index >= 0 && index < static_cast<int>(actual_output_sizes_.size()))
            ? actual_output_sizes_[index] : 0;
    }

    void setInverse(bool inverse) override { is_inverse_ = inverse; }
    bool isInverse() const override        { return is_inverse_; }

    /// Fusable as a pure single-"codes"-port Map in either single-output forward
    /// mode: linear/no-outlier (cuSZp, warp-register strategy) or in-place outlier
    /// + zigzag under ABS/NOA (PFPL, chunk-cooperative strategy). The default
    /// 3-port outlier mode scatters to side buffers and is not a map.
    FusionSpec getFusionSpec() const override {
        if (is_inverse_) return {};
        if (isLinearMode() && !config_.linear_high_precision)
            return FusionSpec{FusionAccess::Map, 0};
        if (isInplaceMode() && config_.zigzag_codes &&
            (config_.eb_mode == ErrorBoundMode::ABS || config_.eb_mode == ErrorBoundMode::NOA))
            return FusionSpec{FusionAccess::Map, 0};
        // Split-outlier (3-port) ABS/NOA zigzag quant is a Map too: codes flow to the
        // next stage (port 0), outlier_vals/idxs (ports 1,2) escape as pipeline leaves
        // that the fused runner fills and the pipeline auto-concatenates. Dither is
        // excluded (the fused op has no per-element dither path).
        if (isSplitOutlierFusable()) return FusionSpec{FusionAccess::Map, 0};
        return {};
    }

    /// Establish the forward-computed absolute error bound for a fused runner
    /// that bypasses execute(). The inverse quant reconstructs with
    /// computed_abs_eb_, normally set during forward execute(); a fused pipeline
    /// reuses this stage object for decompress, so it must be primed. ABS only.
    void primeAbsEbForFusion() {
        if (config_.eb_mode == ErrorBoundMode::ABS) {
            computed_abs_eb_ = resolveUniformBound();
            if (config_.power_of_two_bound)
                computed_abs_eb_ = floorPowerOfTwo(computed_abs_eb_);
        }
    }

    /// Resolve computed_abs_eb_ (and value_base) for a fused runner, covering NOA:
    /// runs the value-range scan (or uses the precomputed base) exactly as
    /// execute() does, so both the fused kernel's scale and the inverse/serialized
    /// header see the right bound. `scan_n` is the logical element count.
    void primeComputedAbsEb(const void* d_in, size_t scan_n,
                            MemoryPool* pool, fz::stream_t stream);
    TInput getComputedAbsEb() const { return computed_abs_eb_; }
    uint32_t getActualOutlierCount() const { return actual_outlier_count_; }
    bool supportsChunkInverseFusion() const {
        return std::is_same<TInput, float>::value &&
               std::is_same<TCode, uint32_t>::value &&
               !isLinearMode() && config_.zigzag_codes && !config_.dither &&
               (config_.eb_mode == ErrorBoundMode::ABS ||
                config_.eb_mode == ErrorBoundMode::NOA);
    }
    /// The warp-register inverse tail: linear (cuSZp / SZp) dequant is just
    /// `code * 2*abs_eb` in float, which the fused unpack kernel does inline.
    /// High-precision (double) linear reconstruction stays staged.
    bool supportsWarpInverseFusion() const {
        return std::is_same<TInput, float>::value &&
               std::is_same<TCode, uint32_t>::value &&
               isLinearMode() && !config_.linear_high_precision &&
               (config_.eb_mode == ErrorBoundMode::ABS ||
                config_.eb_mode == ErrorBoundMode::NOA);
    }
    void setFusedInverseResult(size_t output_bytes) override {
        actual_output_sizes_ = {output_bytes};
    }

    /// Inverse-mode warp quant declaration — the Map role (linear dequant tail)
    /// of the warp decompress chain. Gated exactly on supportsWarpInverseFusion()
    /// (linear, non-high-precision, ABS/NOA, float, uint32 code), the inverse
    /// mirror of the forward getFusionSpec() Map declaration.
    FusionSpec getInverseFusionSpec() const override {
        if (!is_inverse_ || !supportsWarpInverseFusion()) return {};
        return FusionSpec{FusionAccess::Map, 0};
    }
    FusedOpDecl getInverseFusedOp() const override {
        if (!getInverseFusionSpec().fusable()) return {};
        // Marker op: the linear `code * 2*abs_eb` dequant is a built-in tail of the
        // warp inverse harness, not a composed device policy. The generic runner
        // reads the step from getFusedInverseDequantStep(), not from this name.
        FusedOpDecl d;
        d.strategy = FusionStrategy::WarpRegister;
        d.op_name  = "LinearDequant";
        return d;
    }
    double getFusedInverseDequantStep() const override {
        return 2.0 * static_cast<double>(computed_abs_eb_);
    }

    /// Fused-op identity for the chunk-cooperative harness: the inplace+zigzag
    /// ABS/NOA float quant maps to the `QuantInplaceZigzag` Map op. Params are
    /// packed from the primed bound (`primeFusedForwardState` must run first).
    FusedOpDecl getFusedOp() const override {
        if (!std::is_same<TInput, float>::value || is_inverse_) return {};  // device ops read float
        // Warp-register (cuSZp): linear float quant is the Map loader, for ABS or NOA
        // (both resolve to one uniform-step absolute bound — the runner primes the NOA
        // range scan and passes the resolved abs_eb). REL is excluded: it is log-domain
        // per-value quant, not a single abs_eb, so it can't ride the fused kernel.
        if (isLinearMode() && !config_.linear_high_precision &&
            (config_.eb_mode == ErrorBoundMode::ABS || config_.eb_mode == ErrorBoundMode::NOA))
            return FusedOpDecl{FusionStrategy::WarpRegister, "LinearQuant", "", {}};
        // Chunk-cooperative (PFPL): inplace+zigzag ABS/NOA float quant.
        if (isInplaceMode() && config_.zigzag_codes &&
            (config_.eb_mode == ErrorBoundMode::ABS || config_.eb_mode == ErrorBoundMode::NOA)) {
            fused::chunk::QuantInplaceZigzagParams p;
            p.ebx2_r    = 1.0f / (2.0f * static_cast<float>(computed_abs_eb_));
            p.radius    = static_cast<uint32_t>(config_.quant_radius);
            p.threshold = config_.outlier_threshold;
            FusedOpDecl d;
            d.strategy       = FusionStrategy::ChunkCooperative;
            d.op_name        = "QuantInplaceZigzag";
            d.include_header = "fused/chunk_fusion/chunk_fusion.cuh";
            d.params.resize(sizeof(p));
            std::memcpy(d.params.data(), &p, sizeof(p));
            return d;
        }
        // Chunk-cooperative (3-port): split-outlier ABS/NOA zigzag float quant. Same
        // uniform-step params as inplace; the op appends outliers to side buffers
        // instead of inlining raw bits (see QuantSplitOutlier in chunk_fusion.cuh).
        if (isSplitOutlierFusable()) {
            fused::chunk::QuantSplitOutlierParams p;
            p.ebx2_r    = 1.0f / (2.0f * static_cast<float>(computed_abs_eb_));
            p.radius    = static_cast<uint32_t>(config_.quant_radius);
            p.threshold = config_.outlier_threshold;
            FusedOpDecl d;
            d.strategy       = FusionStrategy::ChunkCooperative;
            d.op_name        = "QuantSplitOutlier";
            d.include_header = "fused/chunk_fusion/chunk_fusion.cuh";
            d.params.resize(sizeof(p));
            std::memcpy(d.params.data(), &p, sizeof(p));
            return d;
        }
        return {};
    }

    std::vector<FusedAuxOutputDecl> getFusedAuxOutputs() const override {
        if (is_inverse_ || !isSplitOutlierFusable()) return {};
        return {
            FusedAuxOutputDecl{1, "outlier_vals", FusedAuxSizeKind::CompactedElements,
                               static_cast<uint8_t>(getInputDataType()), 1u, 0u, 1u},
            FusedAuxOutputDecl{2, "outlier_idxs", FusedAuxSizeKind::CompactedElements,
                               static_cast<uint8_t>(DataType::UINT32), 1u, 0u, 1u},
        };
    }

    /// The fused runner bypasses forward execute(); prime the value-range scan so
    /// the fused kernel's scale and the reused inverse/header agree (covers NOA).
    void primeFusedForwardState(const FusedPrimeContext& c) override {
        fused_outlier_count_set_ = false;   // reset per fused compress; set by setFusedSideOutput
        num_elements_ = c.input_bytes / sizeof(TInput);
        primeComputedAbsEb(c.d_input, c.input_bytes / sizeof(TInput), c.pool, c.stream);
    }

    /// The fused split-outlier runner fills the outlier ports (1 = vals TInput,
    /// 2 = idxs uint32) but bypasses execute()/postStreamSync, so `actual_outlier_count_`
    /// — which serializeHeader writes so the inverse knows how many outliers to scatter
    /// — is never set. Recover it from the byte count the runner reports. Both ports
    /// encode the same count; either sets it, and both set the matching output size.
    void setFusedSideOutput(int output_index, size_t bytes) override {
        if (actual_output_sizes_.size() < 3) actual_output_sizes_.resize(3, 0);
        if (output_index == 1) {
            actual_outlier_count_   = static_cast<uint32_t>(bytes / sizeof(TInput));
            actual_output_sizes_[1] = bytes;
            fused_outlier_count_set_ = true;
        } else if (output_index == 2) {
            actual_outlier_count_   = static_cast<uint32_t>(bytes / sizeof(uint32_t));
            actual_output_sizes_[2] = bytes;
            fused_outlier_count_set_ = true;
        }
    }

    /// Store the logical grid so the NOA value-range scan can exclude the
    /// LC-chunk zero-padding tail of the input buffer (E16 over-loosening on
    /// all-positive fields). Runtime-only hint; not serialized (decode never
    /// scans). Base default is a no-op that discards dims.
    void setDims(const std::array<size_t, 3>& dims) override { dims_ = dims; }

    uint16_t getStageTypeId() const override {
        return static_cast<uint16_t>(StageType::QUANTIZER);
    }

    uint8_t getOutputDataType(size_t output_index) const override {
        if (is_inverse_) return static_cast<uint8_t>(getInputDataType());
        // Linear mode: codes hold two's-complement signed q — declare the signed type
        // so the DAG connects cleanly to LorenzoStage<intN>.
        if (isLinearMode()) return static_cast<uint8_t>(signedOf(getCodeDataType()));
        if (isInplaceMode()) return static_cast<uint8_t>(getCodeDataType()); // only codes
        switch (output_index) {
            case 0: return static_cast<uint8_t>(getCodeDataType());
            case 1: return static_cast<uint8_t>(getInputDataType());
            case 2: return static_cast<uint8_t>(DataType::UINT32);
            default: return static_cast<uint8_t>(DataType::UINT8);
        }
    }

    uint8_t getInputDataType(size_t /*input_index*/) const override {
        return static_cast<uint8_t>(getInputDataType());
    }

    size_t serializeHeader(size_t output_index, uint8_t* buf, size_t max_size) const override;
    size_t getMaxHeaderSize(size_t) const override { return sizeof(QuantizerConfig); }
    void deserializeHeader(const uint8_t* buf, size_t size) override;

    void saveState() override {
        saved_config_ = config_;
        saved_num_elements_ = num_elements_;
        saved_actual_outlier_count_ = actual_outlier_count_;
        saved_computed_abs_eb_ = computed_abs_eb_;
        saved_computed_value_base_ = computed_value_base_;
        saved_actual_output_sizes_ = actual_output_sizes_;
    }

    void restoreState() override {
        config_ = saved_config_;
        num_elements_ = saved_num_elements_;
        actual_outlier_count_ = saved_actual_outlier_count_;
        computed_abs_eb_ = saved_computed_abs_eb_;
        computed_value_base_ = saved_computed_value_base_;
        actual_output_sizes_ = saved_actual_output_sizes_;
    }

    void setErrorBound(double eb)             { config_.error_bound = eb; }
    void setQuantRadius(int r)               { config_.quant_radius = r; }
    void setOutlierCapacity(float c)         { config_.outlier_capacity = c; }
    void setErrorBoundMode(ErrorBoundMode m) { config_.eb_mode = m; }
    void setValueBase(float vb)              { config_.precomputed_value_base = vb; }
    void setZigzagCodes(bool enable)         { config_.zigzag_codes = enable; }
    /// ABS/NOA: |x| >= threshold → lossless outlier regardless of bin (LC reference parameter).
    void setOutlierThreshold(float t)        { config_.outlier_threshold = t; }
    /// ABS/NOA: encode outliers in-place (raw float bits in codes array; no scatter buffers).
    void setInplaceOutliers(bool enable)     { config_.inplace_outliers = enable; }
    /// ABS/NOA: linear / no-outlier mode (cuSZp-style signed codes; see Config::linear_mode).
    void setLinearMode(bool enable)          { config_.linear_mode = enable; }
    /// Linear mode: use double coordinate arithmetic and a strict rounding reserve.
    void setLinearHighPrecision(bool enable) { config_.linear_high_precision = enable; }
    /// Uniform modes: tighten the resolved absolute EB to the next lower power of two.
    void setPowerOfTwoBound(bool enable)     { config_.power_of_two_bound = enable; }
    /// ABS/NOA/REL: dithered ("_R"-style) reconstruction; see Config::dither.
    /// Throws at execute() if combined with linear_mode or inplace_outliers.
    void setDither(bool enable)              { config_.dither = enable; }
    /// Seed for the deterministic per-element dither offset (see Config::dither_seed).
    void setDitherSeed(uint64_t seed)        { config_.dither_seed = seed; }
    /// Dither offset amplitude as a fraction of abs_eb, in (0,1]; see Config::dither_strength.
    void setDitherStrength(float strength)   { config_.dither_strength = strength; }

    TInput         getErrorBound()        const { return static_cast<TInput>(config_.error_bound); }
    int            getQuantRadius()       const { return config_.quant_radius; }
    ErrorBoundMode getErrorBoundMode()    const { return config_.eb_mode; }
    float          getValueBase()         const { return config_.precomputed_value_base; }
    float          getOutlierCapacity()   const { return config_.outlier_capacity; }
    bool           getZigzagCodes()       const { return config_.zigzag_codes; }
    float          getOutlierThreshold()  const { return config_.outlier_threshold; }
    bool           getInplaceOutliers()   const { return config_.inplace_outliers; }
    bool           getLinearMode()        const { return config_.linear_mode; }
    bool           getLinearHighPrecision() const { return config_.linear_high_precision; }
    bool           getPowerOfTwoBound()   const { return config_.power_of_two_bound; }
    bool           getDither()            const { return config_.dither; }
    uint64_t       getDitherSeed()        const { return config_.dither_seed; }
    float          getDitherStrength()    const { return config_.dither_strength; }

private:
    Config config_;
    Config saved_config_;
    std::array<size_t, 3> dims_ = {0, 1, 1};  ///< Logical grid (setDims); NOA scan hint only.
    std::vector<size_t> actual_output_sizes_;
    std::vector<size_t> saved_actual_output_sizes_;
    size_t   num_elements_        = 0;
    size_t   saved_num_elements_  = 0;
    uint32_t actual_outlier_count_= 0;
    uint32_t saved_actual_outlier_count_ = 0;
    /// Set when a fused split-outlier runner reported the count via setFusedSideOutput.
    /// The fused kernel appends into its OWN counter, not this stage's scratch, so
    /// postStreamSync must not read the (untouched) scratch and clobber the count.
    /// Reset at the start of each fused compress (primeFusedForwardState).
    bool     fused_outlier_count_set_ = false;
    bool     is_inverse_          = false;
    TInput   computed_abs_eb_     = static_cast<TInput>(1e-4);
    TInput   saved_computed_abs_eb_ = static_cast<TInput>(1e-4);
    TInput   computed_value_base_ = static_cast<TInput>(0);
    TInput   saved_computed_value_base_ = static_cast<TInput>(0);
    /// Stage-private 4-byte device scratch holding the live outlier count.
    /// Allocated lazily via `pool->allocatePersistentDevice(4, ...)` — see
    /// `initOutlierCountScratch()`. Used by the forward kernel as the atomic
    /// counter and D2H'd in `postStreamSync()`. The inverse path does NOT
    /// touch this — it reads the count from the deserialized FZM header and
    /// passes it as a `uint32_t` kernel-launch argument. Not used in inplace
    /// mode (which has no separate scatter path).
    uint32_t* d_outlier_count_scratch_ = nullptr;
    /// Set by linear forward kernels when a quantization bin is outside the
    /// signed TCode range; postStreamSync() converts it to a hard failure.
    uint32_t* d_linear_overflow_scratch_ = nullptr;
    /// Pool that owns `d_outlier_count_scratch_` — captured at allocation
    /// time so the destructor returns it to the right pool.
    MemoryPool* persistent_pool_ = nullptr;
    /// Expires if the pool is destroyed before this stage. `persistent_pool_` is a
    /// raw borrow used in the destructor, and only Pipeline's declaration order
    /// (mem_pool_ before stages_) makes that safe — a stage built against a
    /// caller-owned pool has no such guarantee. See MemoryPool::lifetimeToken().
    std::weak_ptr<const void> persistent_pool_alive_;


    /// Lazily allocate the 4-byte outlier-count scratch via the pool's
    /// persistent allocator. Idempotent; no-op if already allocated, or if
    /// the stage is configured for inplace-outlier mode.
    void initOutlierCountScratch(MemoryPool* pool);
    void initLinearOverflowScratch(MemoryPool* pool);

    bool isInplaceMode() const {
        return config_.inplace_outliers
            && config_.eb_mode != ErrorBoundMode::REL;
    }

    bool isLinearMode() const { return config_.linear_mode; }

    static TInput floorPowerOfTwo(TInput value) {
        if (!(value > TInput(0)) || !std::isfinite(value))
            throw std::runtime_error(
                "QuantizerStage: power_of_two_bound requires a finite positive absolute bound");
        int exponent = 0;
        std::frexp(value, &exponent);
        return std::ldexp(TInput(1), exponent - 1);
    }

    TInput resolveUniformBound(TInput scale = TInput(1)) const {
        const double resolved = config_.error_bound * static_cast<double>(scale);
        TInput bound = static_cast<TInput>(resolved);
        // The strict path must never loosen a decimal/TOML request merely because
        // its TInput representation rounded upward. f64 compares equal here;
        // this primarily protects f32.
        if (config_.linear_high_precision && static_cast<double>(bound) > resolved)
            bound = std::nextafter(bound, TInput(0));
        return bound;
    }

    void applyUniformBoundPolicy(TInput data_abs_max, bool have_data_abs_max);

    /// True when this stage is a fusable 3-port split-outlier Map: standard outlier
    /// mode (not inplace, not linear), zigzag codes, ABS or NOA (uniform step), no
    /// dither. Its codes stream matches quantizer_abs_fwd_kernel<...,Zigzag=true> and
    /// the outliers become escaping side outputs. Forward only (device op reads float).
    bool isSplitOutlierFusable() const {
        return std::is_same<TInput, float>::value && !is_inverse_
            && !isInplaceMode() && !isLinearMode()
            && config_.zigzag_codes && !config_.dither
            && (config_.eb_mode == ErrorBoundMode::ABS ||
                config_.eb_mode == ErrorBoundMode::NOA);
    }

    /// Signed DataType corresponding to an unsigned code type (UINT16→INT16, etc.).
    /// Linear-mode codes are two's-complement signed values stored in an unsigned TCode.
    static DataType signedOf(DataType d) {
        switch (d) {
            case DataType::UINT8:  return DataType::INT8;
            case DataType::UINT16: return DataType::INT16;
            case DataType::UINT32: return DataType::INT32;
            default:               return d;
        }
    }

    DataType getInputDataType() const {
        if (std::is_same<TInput, float>::value)  return DataType::FLOAT32;
        if (std::is_same<TInput, double>::value) return DataType::FLOAT64;
        return DataType::FLOAT32;
    }
    DataType getCodeDataType() const {
        if (std::is_same<TCode, uint8_t>::value)  return DataType::UINT8;
        if (std::is_same<TCode, uint16_t>::value) return DataType::UINT16;
        if (std::is_same<TCode, uint32_t>::value) return DataType::UINT32;
        return DataType::UINT16;
    }
    size_t getMaxOutlierCount(size_t n) const {
        return static_cast<size_t>(std::ceil(n * config_.outlier_capacity));
    }
};

extern template class QuantizerStage<float,  uint16_t>;
extern template class QuantizerStage<float,  uint32_t>;
extern template class QuantizerStage<double, uint16_t>;
extern template class QuantizerStage<double, uint32_t>;

} // namespace fz
