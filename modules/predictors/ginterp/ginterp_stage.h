#pragma once

/**
 * @file ginterp_stage.h
 * @brief G-Interp spline-interpolation predictor + quantizer (cuSZ-Hi port).
 *
 * MVP (phase 1): 3D-only, no auto-tuning, hard-coded baseline INTERPOLATION_PARAMS.
 * Phase 2 will add `setAutoTuning()` and the profiling pre-pass; phase 3 will
 * add the 2D path. See `memory/new_stages.md` for the full plan.
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

namespace fz {

/**
 * Serialized G-Interp configuration stored in FZMBufferEntry.stage_config.
 * Fits within the 128-byte `FZM_STAGE_CONFIG_SIZE` limit.
 */
struct GInterpConfig {
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
    uint8_t  reserved[28];      ///< Future auto-tuning params (alpha/beta/use_md/...).

    // Total: 64 bytes. Comfortable margin under FZM_STAGE_CONFIG_SIZE (128 B).

    GInterpConfig()
        : error_bound(0.0f), quant_radius(0), num_elements(0), outlier_count(0),
          input_type(DataType::FLOAT32), code_type(DataType::UINT16),
          ndim(3), eb_mode(0),
          dim_x(0), dim_y(1), dim_z(1),
          anchor_dim_x(0), anchor_dim_y(0), anchor_dim_z(0),
          user_eb(0.0f), value_base(0.0f), reserved{} {}
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
 *   - [4] outlier_count  — outlier count (`uint32_t`, 4 bytes)
 *
 * Inverse: takes the five forward outputs, produces the reconstructed `TInput`
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
 *   - **3-D only** in this MVP; `setDims()` throws for 1-D or 2-D.
 *   - Best results when each `dim` is a multiple of 16 (the anchor tile size).
 *     For ragged dims, edge voxels see slightly worse prediction because
 *     boundary anchors are unavailable.
 *   - cuSZ-Hi's `INTERPOLATION_PARAMS` auto-tuning is not yet ported; this MVP
 *     uses the upstream deterministic baseline (`alpha=1.75`, `beta=4.0`,
 *     `use_md={t,t,f,f,f,f}`). Real CR may be 10–30% worse than the cuSZ-Hi
 *     paper until phase 2 lands.
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
        /// Spatial dimensions `{x, y, z}` where x is fastest. MVP requires
        /// `dims[2] > 1` — `setDims()` throws otherwise.
        std::array<size_t, 3> dims = {0, 0, 0};
        ErrorBoundMode eb_mode   = ErrorBoundMode::ABS;
        /// Pre-computed value_range (NOA) or max(|data|) (REL). Set to skip
        /// the on-device scan during execute() (required for graph capture).
        float precomputed_value_base = 0.0f;

        Config() = default;
    };

    explicit GInterpStage(const Config& cfg = Config()) : config_(cfg) {
        actual_output_sizes_.resize(5, 0);
    }

    // ── Stage interface ──────────────────────────────────────────────────────
    void execute(
        cudaStream_t stream,
        MemoryPool* pool,
        const std::vector<void*>& inputs,
        const std::vector<void*>& outputs,
        const std::vector<size_t>& sizes
    ) override;

    void postStreamSync(cudaStream_t stream) override;

    std::string getName() const override { return "GInterp"; }
    size_t getNumInputs()  const override { return is_inverse_ ? 5 : 1; }
    size_t getNumOutputs() const override { return is_inverse_ ? 1 : 5; }

    std::vector<std::string> getOutputNames() const override {
        return {"codes", "anchor", "outlier_vals", "outlier_idxs", "outlier_count"};
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
    void setDims(const std::array<size_t, 3>& dims) override;
    void setDims(size_t x, size_t y, size_t z) {
        setDims(std::array<size_t, 3>{x, y, z});
    }

    float          getErrorBound()       const { return config_.error_bound; }
    int            getQuantRadius()      const { return config_.quant_radius; }
    float          getOutlierCapacity()  const { return config_.outlier_capacity; }
    ErrorBoundMode getErrorBoundMode()   const { return config_.eb_mode; }
    float          getValueBase()        const { return config_.precomputed_value_base; }
    std::array<size_t, 3> getDims()      const { return config_.dims; }

    void setInverse(bool inv) override { is_inverse_ = inv; }
    bool isInverse()           const override { return is_inverse_; }

    /// Returns the effective spatial dimensionality (3 in MVP).
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
            case 4: return static_cast<uint8_t>(DataType::UINT32);  // outlier_count
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
    /// Stashed during execute() (compress); consumed once by postStreamSync().
    const void* d_outlier_count_ptr_ = nullptr;

    /// Absolute error bound actually used in kernel launches. For ABS this
    /// equals `config_.error_bound`; for REL/NOA it is the converted value.
    TInput computed_abs_eb_ = 0;
    /// value_range (NOA) or max(|data|) (REL) used in conversion. Stored so
    /// `serializeHeader()` can embed it for the decompressor.
    float computed_value_base_ = 0.0f;

    /// Cached anchor grid extent (set by setDims).
    std::array<size_t, 3> anchor_dims_ = {0, 0, 0};

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
