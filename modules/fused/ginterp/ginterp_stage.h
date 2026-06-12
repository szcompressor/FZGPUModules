#pragma once

/**
 * @file ginterp_stage.h
 * @brief G-Interp spline-interpolation predictor + quantizer (cuSZ-Hi port).
 *
 * Supports 2-D and 3-D inputs. 1-D is rejected (cuSZ-Hi has no 1-D path).
 * `INTERPOLATION_PARAMS` auto-tuning (modes 1 and 3) is currently wired for
 * the 3-D path only; in 2-D the stage logs a warning and falls back to the
 * deterministic baseline.
 */

#include "stage/stage.h"
#include "fzm_format.h"
#include "fused/lorenzo_quant/lorenzo_quant.h"  // for ErrorBoundMode

#include <cuda_runtime.h>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <vector>

// Forward-declared in the public header so callers don't pull in the
// cuSZ-Hi type subset. The full definition lives in cusz_type_subset.h,
// included only inside the stage TU.
struct INTERPOLATION_PARAMS;

namespace fz {

/**
 * Serialized G-Interp configuration stored in FZMBufferEntry.stage_config.
 * Fits within the 128-byte `FZM_STAGE_CONFIG_SIZE` limit.
 */
struct GInterpConfig {
    // ── identity / dims (40 B) ──────────────────────────────────────────────
    float    error_bound;       ///< Absolute bound used by the decompressor.
    uint32_t quant_radius;      ///< Quantization radius (codes lie in [0, 2*radius)).
    uint32_t num_elements;      ///< Total element count (= dim_x*dim_y*dim_z).
    uint32_t outlier_count;     ///< Actual outlier count (post-execute).
    DataType input_type;        ///< Float input type (1 B).
    DataType code_type;         ///< Quant code type (1 B).
    uint8_t  ndim;              ///< Spatial dimensionality (3 in MVP).
    uint8_t  eb_mode;           ///< ErrorBoundMode cast to uint8_t.
    uint32_t dim_x;             ///< X (fast) dimension.
    uint32_t dim_y;             ///< Y dimension.
    uint32_t dim_z;             ///< Z dimension.
    uint32_t anchor_dim_x;      ///< Anchor grid X extent.
    uint32_t anchor_dim_y;      ///< Anchor grid Y extent.
    uint32_t anchor_dim_z;      ///< Anchor grid Z extent.
    float    user_eb;           ///< Original user-specified bound (before mode conversion).
    float    value_base;        ///< value_range (NOA) / max(|data|) (REL) used in conversion.

    // ── resolved INTERPOLATION_PARAMS (phase 2 auto-tune output, 39 B) ──────
    // Both encoder and decoder must use the exact same intp_param values, so
    // the resolved params are written here on compress and consumed on decompress.
    // Layout chosen to mirror `INTERPOLATION_PARAMS` field-for-field; we don't
    // memcpy the struct directly because its padding is implementation-defined.
    double   intp_alpha;        ///< Resolved alpha (auto-tuned from rel_eb or fixed).
    double   intp_beta;         ///< Resolved beta (default 4.0).
    uint8_t  intp_use_md[6];    ///< Resolved use_md[level], booleans as u8.
    uint8_t  intp_use_natural[6];
    uint8_t  intp_reverse[6];
    uint8_t  auto_tuning_mode;  ///< 0=off, 1=cheap, 3=full, 4=full+alpha sweep, 5+=manual α/β.
    uint8_t  pad[8];            ///< Reserved for future fields (alignment also).

    // Total: 88 bytes. Comfortable margin under FZM_STAGE_CONFIG_SIZE (128 B).

    GInterpConfig()
        : error_bound(0.0f), quant_radius(0), num_elements(0), outlier_count(0),
          input_type(DataType::FLOAT32), code_type(DataType::UINT16),
          ndim(3), eb_mode(0),
          dim_x(0), dim_y(1), dim_z(1),
          anchor_dim_x(0), anchor_dim_y(0), anchor_dim_z(0),
          user_eb(0.0f), value_base(0.0f),
          intp_alpha(1.75), intp_beta(4.0),
          intp_use_md{1, 1, 0, 0, 0, 0},
          intp_use_natural{0, 0, 0, 0, 0, 0},
          intp_reverse{0, 0, 0, 0, 0, 0},
          auto_tuning_mode(0), pad{} {}
};
static_assert(sizeof(GInterpConfig) <= FZM_STAGE_CONFIG_SIZE,
              "GInterpConfig must fit in FZM_STAGE_CONFIG_SIZE");

/**
 * G-Interp predictor with error-bounded quantization (3-D, MVP).
 *
 * @note **Prior work:** the underlying spline kernels are adapted from the
 *       cuSZ-Hi compressor (Indiana University, Argonne National Laboratory),
 *       BSD-3-Clause. The host-side wrapper, memory-pool integration, and
 *       outlier-fusion contract are FZGPUModules code. See `THIRD_PARTY.md`.
 *
 * Forward outputs (compression):
 *   - [0] codes          — quantization codes (`TCode`, full N elements)
 *   - [1] anchor         — corner anchor values (`TInput`, ~N/4096 elements)
 *   - [2] outlier_vals   — out-of-range residuals (`TInput`)
 *   - [3] outlier_idxs   — outlier element indices (`uint32_t`)
 *
 * The outlier count is **not** a DAG output port — it lives in a stage-private
 * 4-byte device scratch (allocated via `pool->allocatePersistentDevice` in
 * `onFinalize()`), is D2H'd in `postStreamSync()`, and is serialized in the
 * FZM header. The inverse path consumes it as a `uint32_t` kernel-launch
 * argument (read from the deserialized header), so the inverse kernel never
 * has to dereference a device pointer to know its loop bound.
 *
 * Inverse: takes the four forward outputs, produces the reconstructed `TInput`
 * volume.
 *
 * ## Error bound and limitations
 *
 * The error bound `eb` is a **target**, not a hard guarantee. The multi-level
 * interpolation tree predicts finer-level values from already-lossy coarser-
 * level reconstructions, so prediction errors accumulate across the four
 * levels. In practice the maximum element-wise error is:
 *   - typically `<= 1.1 * eb` on smooth data
 *   - up to `~2 * eb` on data with many outliers (large spikes that the spline
 *     can't predict — these are stored exactly via the outlier triplet, but
 *     their neighbours still see compounded interpolation error).
 *
 * Other limitations to be aware of:
 *   - **2-D and 3-D only**; `setDims()` throws for 1-D input.
 *   - 3-D path: best results when each `dim` is a multiple of 16 (the 3-D
 *     anchor tile size). 2-D path: anchor tile is 32×8, so best results when
 *     `dim_x` is a multiple of 32 and `dim_y` of 8. Ragged dims still work
 *     but edge elements see slightly worse prediction.
 *   - `INTERPOLATION_PARAMS` auto-tuning (`setAutoTuning(1)` / `(3)`) is wired
 *     for the 3-D path only. In 2-D the stage logs a warning and falls back
 *     to the deterministic baseline (`alpha=1.75`, `beta=4.0`,
 *     `use_md={t,t,f,f,f,f}`). 2-D auto-tune (cuSZ-Hi `auto_tuning_mode == 2`)
 *     is a follow-up.
 *
 * ## Radius auto-tune (default behaviour)
 *
 * `setQuantRadius(0)` (the default) means "auto": on first `execute()`, the
 * stage scans the input min/max and picks the largest radius that fits the
 * data range, capped at the TCode bit-width's maximum. This minimises outlier
 * count for unknown data ranges and is the recommended setting.
 *
 * For **CUDA graph capture** or strict determinism, set the radius explicitly
 * to any positive value to skip the scan (e.g. `setQuantRadius(512)` for
 * climate-style data where the user wants extremes routed to the outlier
 * triplet for downstream handling).
 *
 * @tparam TInput  Floating-point input type (`float` only in MVP).
 * @tparam TCode   Quantization code type (`uint8_t`, `uint16_t`, or `uint32_t`).
 */
template <typename TInput = float, typename TCode = uint16_t>
class GInterpStage : public Stage {
public:
    struct Config {
        float error_bound        = 1e-3f;
        /// Quantization radius.
        ///   0 = auto-tune (default): on first execute() the stage scans the
        ///       input min/max and picks the largest radius that fits the data
        ///       range, capped at the TCode bit-width's maximum. Minimises
        ///       outlier count and works on any data range out of the box.
        ///   > 0 = manual: use this radius directly, skip the scan. Required
        ///       for CUDA graph capture (the scan does a D2H sync). Manual
        ///       values like 512 are useful when the user wants residuals
        ///       beyond `radius * eb / 2` routed to the separate outlier
        ///       triplet (e.g. climate data where extremes are handled
        ///       downstream).
        int   quant_radius       = 0;
        float outlier_capacity   = 0.10f;
        /// Spatial dimensions `{x, y, z}` where x is fastest. 2-D inputs use
        /// `dims[2] == 1` (the stage automatically picks the 2-D kernel path).
        /// 1-D (`dims[1] == 1`) is rejected by `setDims()`.
        std::array<size_t, 3> dims = {0, 0, 0};
        ErrorBoundMode eb_mode   = ErrorBoundMode::ABS;
        /// Pre-computed value_range (NOA) or max(|data|) (REL). Set to skip
        /// the on-device scan during execute() (required for graph capture).
        float precomputed_value_base = 0.0f;
        /// `INTERPOLATION_PARAMS` auto-tuning mode (cuSZ-Hi).
        ///   0 = off (default, deterministic baseline)
        ///   1 = cheap profiling — sets `reverse[]` only (~1 ms)
        ///   3 = full structural profiling — sets `use_md`, `use_natural`,
        ///       `reverse` per level (~5–15 ms; matches the cuSZ-Hi paper)
        ///   4 = full structural + alpha/beta sweep (~15–30 ms; on top of
        ///       mode 3 results, scans 8 (alpha, beta) combos and picks the
        ///       lowest error)
        ///   5 = manual alpha/beta override (no profiling). Set `manual_alpha`
        ///       / `manual_beta` to use them verbatim; otherwise the
        ///       piecewise-linear `alpha` from `rel_eb` (cuSZ-Hi recipe) is
        ///       used with `beta=4.0`. The structural flags stay at baseline.
        /// Modes 1, 3, 4 are 3-D-only; on 2-D inputs they fall back to
        /// baseline. Mode 5 is dim-agnostic.
        /// Mode 2 from cuSZ-Hi is not yet wired (3-D-only alternate cheap
        /// probe — same value as mode 1).
        /// Any non-zero mode that does profiling (1/3/4) forces a host-blocking
        /// D2H sync inside execute() and is **incompatible with CUDA graph
        /// capture**. Mode 5 (manual) is graph-safe.
        uint8_t auto_tuning_mode = 0;

        /// Manual alpha override for `auto_tuning_mode == 5`. Set to a value
        /// > 0 to use this alpha verbatim; 0.0 (default) defers to the
        /// cuSZ-Hi piecewise-linear schedule keyed on `rel_eb`.
        double  manual_alpha = 0.0;
        /// Manual beta override for `auto_tuning_mode == 5`. Same convention.
        double  manual_beta  = 0.0;

        Config() = default;
    };

    explicit GInterpStage(const Config& cfg = Config()) : config_(cfg) {
        actual_output_sizes_.resize(4, 0);
    }
    ~GInterpStage() override;

    // ── Stage interface ──────────────────────────────────────────────────────
    void execute(
        cudaStream_t stream,
        MemoryPool* pool,
        const std::vector<void*>& inputs,
        const std::vector<void*>& outputs,
        const std::vector<size_t>& sizes
    ) override;

    void postStreamSync(cudaStream_t stream) override;

    /// In PREALLOCATE mode + auto-tuning > 0, pre-allocate the persistent
    /// profiling-errors scratch (36 floats device + pinned host) via the pool's
    /// persistent allocators so they aren't on the per-call stream-ordered
    /// path. In MINIMAL mode this is deferred to first execute(). Auto-tune off
    /// is a no-op.
    void onFinalize(size_t estimated_inlen, MemoryPool* pool) override;

    size_t estimateDeviceFootprintBytes(size_t /*estimated_inlen*/) const override {
        return needsProfilingScratch() ? kProfilingErrCount * sizeof(float) : 0;
    }
    size_t estimatePinnedFootprintBytes(size_t /*estimated_inlen*/) const override {
        return needsProfilingScratch() ? kProfilingErrCount * sizeof(float) : 0;
    }

    std::string getName() const override { return "GInterp"; }
    size_t getNumInputs()  const override { return is_inverse_ ? 4 : 1; }
    size_t getNumOutputs() const override { return is_inverse_ ? 1 : 4; }

    std::vector<std::string> getOutputNames() const override {
        return {"codes", "anchor", "outlier_vals", "outlier_idxs"};
    }

    std::vector<size_t> estimateOutputSizes(
        const std::vector<size_t>& input_sizes
    ) const override;

    std::unordered_map<std::string, size_t> getActualOutputSizesByName() const override {
        auto names = getOutputNames();
        std::unordered_map<std::string, size_t> r;
        for (size_t i = 0; i < names.size() && i < actual_output_sizes_.size(); i++)
            r[names[i]] = actual_output_sizes_[i];
        return r;
    }
    size_t getActualOutputSize(int index) const override {
        return (index >= 0 && index < static_cast<int>(actual_output_sizes_.size()))
            ? actual_output_sizes_[index] : 0;
    }

    void saveState()    override { saved_output_sizes_ = actual_output_sizes_; }
    void restoreState() override { actual_output_sizes_ = saved_output_sizes_; }

    // ── Setters ──────────────────────────────────────────────────────────────
    void setErrorBound(float eb)              { config_.error_bound = eb; }
    void setQuantRadius(int radius)           { config_.quant_radius = radius; }
    void setOutlierCapacity(float cap)        { config_.outlier_capacity = cap; }
    void setErrorBoundMode(ErrorBoundMode m)  { config_.eb_mode = m; }
    void setValueBase(float v)                { config_.precomputed_value_base = v; }
    /// Enable cuSZ-Hi's `INTERPOLATION_PARAMS` auto-tuning. See
    /// `Config::auto_tuning_mode` for mode semantics. Default is 0 (off).
    /// Profiling modes (1/3/4) disable CUDA graph capture for this stage;
    /// mode 5 (manual override) is graph-safe.
    void setAutoTuning(uint8_t mode)          { config_.auto_tuning_mode = mode; }
    /// Set the manual alpha/beta override pair used when
    /// `auto_tuning_mode == 5`. Both must be > 0 to take effect; passing 0
    /// for either field defers to the cuSZ-Hi piecewise-linear schedule
    /// (alpha from rel_eb) or the upstream default (beta = 4.0).
    void setManualAlphaBeta(double alpha, double beta) {
        config_.manual_alpha = alpha;
        config_.manual_beta  = beta;
    }
    void setDims(const std::array<size_t, 3>& dims) override;
    void setDims(size_t x, size_t y, size_t z) {
        setDims(std::array<size_t, 3>{x, y, z});
    }

    float          getErrorBound()       const { return config_.error_bound; }
    int            getQuantRadius()      const { return config_.quant_radius; }
    float          getOutlierCapacity()  const { return config_.outlier_capacity; }
    ErrorBoundMode getErrorBoundMode()   const { return config_.eb_mode; }
    float          getValueBase()        const { return config_.precomputed_value_base; }
    uint8_t        getAutoTuningMode()   const { return config_.auto_tuning_mode; }
    std::array<size_t, 3> getDims()      const { return config_.dims; }

    void setInverse(bool inv) override { is_inverse_ = inv; }
    bool isInverse()           const override { return is_inverse_; }

    /// Returns the effective spatial dimensionality (2 or 3). 1-D inputs are
    /// rejected by `setDims()` so this method never returns 1 once the stage
    /// has been configured.
    int ndim() const {
        if (config_.dims[2] > 1) return 3;
        if (config_.dims[1] > 1) return 2;
        return 1;
    }

    // ── Type / Serialization ─────────────────────────────────────────────────
    uint16_t getStageTypeId() const override {
        return static_cast<uint16_t>(StageType::G_INTERP);
    }

    uint8_t getOutputDataType(size_t output_index) const override {
        switch (output_index) {
            case 0: return static_cast<uint8_t>(codeDataType());    // codes
            case 1: return static_cast<uint8_t>(inputDataType());   // anchor
            case 2: return static_cast<uint8_t>(inputDataType());   // outlier_vals
            case 3: return static_cast<uint8_t>(DataType::UINT32);  // outlier_idxs
            default: return static_cast<uint8_t>(DataType::UINT8);
        }
    }
    uint8_t getInputDataType(size_t /*input_index*/) const override {
        return static_cast<uint8_t>(inputDataType());
    }

    size_t serializeHeader(size_t output_index, uint8_t* buf, size_t max_size) const override;
    void deserializeHeader(const uint8_t* buf, size_t size) override;
    size_t getMaxHeaderSize(size_t /*output_index*/) const override {
        return sizeof(GInterpConfig);
    }

    /// MVP: false. The forward path does no D2H during execute(), and the
    /// inverse outlier scatter reads `*d_outlier_count` on-device. The only
    /// host-blocking call is `postStreamSync()` for the outlier count, which
    /// runs outside the captured region (same pattern as LorenzoQuantStage).
    /// Keeping this false for the first PR until graph compatibility is
    /// verified end-to-end with a real capture+replay test.
    bool isGraphCompatible() const override { return false; }

private:
    Config config_;
    std::vector<size_t> actual_output_sizes_;
    std::vector<size_t> saved_output_sizes_;

    bool   is_inverse_ = false;
    size_t num_elements_ = 0;
    uint32_t actual_outlier_count_ = 0;
    /// Stage-private 4-byte device scratch holding the live outlier count.
    /// Allocated lazily in `initOutlierCountScratch()` via
    /// `pool->allocatePersistentDevice(4, ...)`, used by the forward kernel
    /// as the atomic counter, and D2H'd in `postStreamSync()`. The inverse
    /// path does NOT touch this — it reads the count from the deserialized
    /// FZM header and passes it as a `uint32_t` kernel-launch argument.
    uint32_t* d_outlier_count_scratch_ = nullptr;

    /// Absolute error bound actually used in kernel launches. For ABS this
    /// equals `config_.error_bound`; for REL/NOA it is the converted value.
    TInput computed_abs_eb_ = 0;
    /// value_range (NOA) or max(|data|) (REL) used in conversion. Stored so
    /// `serializeHeader()` can embed it for the decompressor.
    float computed_value_base_ = 0.0f;

    /// Cached anchor grid extent (set by setDims).
    std::array<size_t, 3> anchor_dims_ = {0, 0, 0};

    // ── Phase 2: auto-tuning state ──────────────────────────────────────────
    /// Persistent profiling scratch (pool-allocated when auto_tuning_mode > 0).
    /// Lives for the stage lifetime; freed in the destructor.
    static constexpr size_t kProfilingErrCount = 36;
    float*   d_profiling_errors_ = nullptr;
    float*   h_profiling_errors_ = nullptr;
    /// Pointer to the MemoryPool that owns the persistent scratch above —
    /// captured in onFinalize() / initProfilingScratch() so the destructor
    /// can return them to the right pool.
    MemoryPool* persistent_pool_ = nullptr;

    /// Resolved INTERPOLATION_PARAMS as POD fields. Initialised to cuSZ-Hi
    /// baseline (matches `INTERPOLATION_PARAMS()` default ctor); overwritten
    /// by execute() when `auto_tuning_mode > 0` and consumed by both forward
    /// and inverse launchers.
    double  resolved_alpha_ = 1.75;
    double  resolved_beta_  = 4.0;
    uint8_t resolved_use_md_[6]      = {1, 1, 0, 0, 0, 0};
    uint8_t resolved_use_natural_[6] = {0, 0, 0, 0, 0, 0};
    uint8_t resolved_reverse_[6]     = {0, 0, 0, 0, 0, 0};

    /// True when the configured auto_tuning_mode needs the 36-float
    /// profiling scratch (modes 1/2/3/4). Modes 0 and 5 don't.
    bool needsProfilingScratch() const {
        const uint8_t m = config_.auto_tuning_mode;
        return m == 1 || m == 2 || m == 3 || m == 4;
    }

    /// Lazily allocate the persistent profiling scratch via the pool's
    /// persistent allocators. No-op if already allocated or auto-tune is off.
    void initProfilingScratch(MemoryPool* pool);
    /// Lazily allocate the 4-byte outlier-count device scratch. Idempotent.
    void initOutlierCountScratch(MemoryPool* pool);
    /// Build a cuSZ-Hi `INTERPOLATION_PARAMS` from the resolved POD fields,
    /// for passing to the kernel launchers. Defined in the .cu (which has
    /// the full struct definition via cusz_type_subset.h).
    INTERPOLATION_PARAMS buildIntpParam() const;

    static DataType inputDataType() {
        if (std::is_same<TInput, float>::value)  return DataType::FLOAT32;
        if (std::is_same<TInput, double>::value) return DataType::FLOAT64;
        return DataType::FLOAT32;
    }
    static DataType codeDataType() {
        if (std::is_same<TCode, uint8_t>::value)  return DataType::UINT8;
        if (std::is_same<TCode, uint16_t>::value) return DataType::UINT16;
        if (std::is_same<TCode, uint32_t>::value) return DataType::UINT32;
        return DataType::UINT16;
    }
    size_t getMaxOutlierCount(size_t n) const {
        return static_cast<size_t>(std::ceil(n * config_.outlier_capacity));
    }
};

extern template class GInterpStage<float, uint8_t>;
extern template class GInterpStage<float, uint16_t>;
extern template class GInterpStage<float, uint32_t>;

} // namespace fz
