/**
 * @file modules/fused/lorenzo_quant/lorenzo_quant.h
 * @brief Fused Lorenzo predictor and quantizer stage.
 */
#pragma once

#include "stage/stage.h"
#include "fzm_format.h"
#include "backend/types.h"
#include <array>
#include <cstdint>
#include <cmath>
#include <cstring>

namespace fz {


/**
 * Interpretation of the user-specified error bound.
 *
 * - **ABS** — `|x_orig - x_recon| <= eb` (default).
 * - **REL** — point-wise relative (PFPL): `|error| / |x_orig| <= eb`.
 *   For Lorenzo this is a *global* approximation: `abs_eb = eb × max(|data|)`.
 *   Values much smaller than max(|data|) may exceed the per-element ratio.
 *   For an exact per-element REL bound use `QuantizerStage` with REL mode.
 * - **NOA** — norm-of-absolute / value-range relative (PFPL):
 *   `abs_eb = eb × (max(data) - min(data))`.  Equivalent to what most other
 *   compressors call "relative".
 */
enum class ErrorBoundMode : uint8_t {
    ABS = 0, ///< Absolute error bound.
    REL = 1, ///< Global-approximate point-wise relative bound.
    NOA = 2, ///< Value-range relative bound (norm-of-absolute).
};

/**
 * Serialized Lorenzo predictor configuration stored in FZMBufferEntry.stage_config.
 *
 * Written by `serializeHeader()` at compression time; read back by
 * `deserializeHeader()` to reconstruct decompressor state. Fits within the
 * 128-byte `FZM_STAGE_CONFIG_SIZE` limit.
 */
struct LorenzoQuantConfig {
    float    error_bound;   ///< Absolute bound after mode conversion (used by decompressor).
    uint32_t quant_radius;  ///< Quantization radius.
    uint32_t num_elements;  ///< Total element count.
    uint32_t outlier_count; ///< Actual number of outliers.
    DataType input_type;    ///< Original input type (1B).
    DataType code_type;     ///< Quantization code type (1B).
    uint8_t  ndim;          ///< Spatial dimensionality 1/2/3 (0 treated as 1).
    uint8_t  eb_mode;       ///< ErrorBoundMode cast to uint8_t.
    uint32_t dim_x;         ///< X (fast) dimension; 0 = infer from num_elements.
    uint32_t dim_y;         ///< Y dimension (1 for 1-D).
    uint32_t dim_z;         ///< Z dimension (1 for 1-D/2-D).
    float    user_eb;       ///< Original user-specified error bound value.
    float    value_base;    ///< value_range (NOA) or max(|data|) (REL) used in conversion.
    uint8_t  zigzag_codes;  ///< 1 if codes are zigzag-encoded, else 0.
    uint8_t  reserved[3];   ///< Must be zero.

    // Total: 44 bytes (fits easily in 128B stage_config)

    LorenzoQuantConfig()
        : error_bound(0.0f), quant_radius(0), num_elements(0), outlier_count(0),
          input_type(DataType::FLOAT32), code_type(DataType::UINT16),
          ndim(1), eb_mode(0), dim_x(0), dim_y(1), dim_z(1),
          user_eb(0.0f), value_base(0.0f), zigzag_codes(0), reserved{0, 0, 0} {}
};
static_assert(sizeof(LorenzoQuantConfig) <= FZM_STAGE_CONFIG_SIZE, "LorenzoQuantConfig must fit in FZM_STAGE_CONFIG_SIZE");

/**
 * Lorenzo predictor with error-bounded quantization (1-D, 2-D, 3-D).
 *
 * @note **Prior work:** fused predictor+quantizer kernels and the multi-output
 *       design follow the cuSZ Lorenzo implementation (`lrz_c.cuhip.inl`,
 *       `lrz_x.cuhip.inl`) by the cuSZ team (BSD-3-Clause). See `THIRD_PARTY.md`.
 *
 * Forward outputs (compression):
 * - [0] codes         — quantization codes for all elements (`TCode`)
 * - [1] outlier_errors — prediction errors for outliers (`TInput`)
 * - [2] outlier_indices — outlier element indices (`uint32_t`)
 *
 * The outlier count is **not** a DAG output port — it lives in a stage-private
 * 4-byte device scratch (allocated via `pool->allocatePersistentDevice` in
 * `onFinalize()`), is D2H'd in `postStreamSync()`, and is serialized in the
 * FZM header. The inverse path receives it as a `uint32_t` kernel-launch
 * argument (read from the deserialized header), so the inverse kernel never
 * has to dereference a device pointer to know its loop bound.
 *
 * Inverse (decompression): takes the three forward outputs, produces the
 * reconstructed data as a single `TInput` array.
 *
 * @tparam TInput  Floating-point input type (`float` or `double`).
 * @tparam TCode   Quantization code type (`uint8_t`, `uint16_t`, `uint32_t`).
 */
template<typename TInput = float, typename TCode = uint16_t>
class LorenzoQuantStage : public Stage {
public:
    /** Construction parameters. */
    struct Config {
        float  error_bound       = 1e-3;    ///< Error bound (interpretation depends on `eb_mode`).
        int    quant_radius      = 32768;   ///< Quantization radius (2^15 for uint16_t).
        float  outlier_capacity  = 0.2f;    ///< Fraction of input size reserved for outliers.
        /// Spatial dimensions `{x, y, z}` where x is fastest.
        /// `dims[0]==0` → infer x from num_elements at runtime (valid for 1-D).
        /// `dims[1]==dims[2]==1` → 1-D; `dims[2]==1` → 2-D; otherwise 3-D.
        std::array<size_t, 3> dims = {0, 1, 1};
        ErrorBoundMode eb_mode = ErrorBoundMode::ABS;
        /// Pre-computed value_range (NOA) or max(|data|) (REL) to skip the
        /// device scan in execute(). Leave at 0 to auto-compute.
        float precomputed_value_base = 0.0f;
        /// Zigzag-encode codes before storage to improve compressibility
        /// when codes cluster near zero (`−2→3, −1→1, 0→0, 1→2, …`).
        bool zigzag_codes = false;
        Config() = default;
        Config(TInput eb, TCode radius = 32768, float outlier_cap = 0.2f,
               std::array<size_t, 3> d = {0, 1, 1})
            : error_bound(eb), quant_radius(radius), outlier_capacity(outlier_cap),
              dims(d) {}
    };
    
    explicit LorenzoQuantStage(const Config& config = Config());
    ~LorenzoQuantStage() override;

    void execute(
        fz::stream_t stream,
        MemoryPool* pool,
        const std::vector<void*>& inputs,
        const std::vector<void*>& outputs,
        const std::vector<size_t>& sizes
    ) override;

    /**
     * Reads back the actual outlier count from the device (4 bytes) and trims
     * actual_output_sizes_ to the real values.  Called by Pipeline::compress()
     * after the stream is synchronized — avoids a mid-pipeline stall.
     */
    void postStreamSync(fz::stream_t stream) override;

    /// Pre-allocate the stage-private 4-byte outlier-count device scratch
    /// (via `pool->allocatePersistentDevice`) in PREALLOCATE mode. In MINIMAL
    /// mode this is deferred to the first compress execute(). The 4-byte
    /// scratch lives for the stage's lifetime.
    void onFinalize(size_t estimated_inlen, MemoryPool* pool) override;

    size_t estimateDeviceFootprintBytes(size_t /*estimated_inlen*/) const override {
        return sizeof(uint32_t);
    }

    std::string getName() const override { return "LorenzoQuant"; }
    size_t getNumInputs()  const override { return is_inverse_ ? 3 : 1; }
    size_t getNumOutputs() const override { return is_inverse_ ? 1 : 3; }

    std::vector<std::string> getOutputNames() const override {
        return {"codes", "outlier_errors", "outlier_indices"};
    }
    
    std::vector<size_t> estimateOutputSizes(
        const std::vector<size_t>& input_sizes
    ) const override;
    
    std::unordered_map<std::string, size_t> getActualOutputSizesByName() const override {
        auto names = getOutputNames();
        std::unordered_map<std::string, size_t> result;
        for (size_t i = 0; i < names.size() && i < actual_output_sizes_.size(); i++) {
            result[names[i]] = actual_output_sizes_[i];
        }
        return result;
    }
    size_t getActualOutputSize(int index) const override {
        return (index >= 0 && index < static_cast<int>(actual_output_sizes_.size()))
            ? actual_output_sizes_[index] : 0;
    }

    // Preserve the forward-mode actual_output_sizes_ across decompression passes.
    // decompressMulti() calls saveState()/restoreState() around each inverse
    // execute() to prevent the inverse pass from permanently corrupting the
    // 4-element forward output-size vector (inverse sets it to a 1-element vector).
    void saveState()    override { saved_output_sizes_ = actual_output_sizes_; }
    void restoreState() override { actual_output_sizes_ = saved_output_sizes_; }

    // Configuration accessors
    void setErrorBound(TInput error_bound) { config_.error_bound = error_bound; }
    void setQuantRadius(TCode radius) { config_.quant_radius = radius; }
    void setOutlierCapacity(float capacity) { config_.outlier_capacity = capacity; }
    void setDims(const std::array<size_t, 3>& dims) override { config_.dims = dims; }
    /// REL here is *global-approximate* (`abs_eb = eb * max(|data|)`), NOT the
    /// exact per-element PFPL bound — use `QuantizerStage` REL for that. See the
    /// error-bound mode notes in the file-level doc.
    void setErrorBoundMode(ErrorBoundMode mode) { config_.eb_mode = mode; }
    // Provide a pre-computed value_range (NOA) or max(|data|) (REL) to skip
    // the internal data scan during execute().  Pass 0 to re-enable auto-scan.
    void setValueBase(float value_base) { config_.precomputed_value_base = value_base; }
    void setZigzagCodes(bool enable) { config_.zigzag_codes = enable; }
    void setDims(size_t x, size_t y = 1, size_t z = 1) { config_.dims = {x, y, z}; }

    TInput getErrorBound() const { return config_.error_bound; }
    TCode  getQuantRadius() const { return config_.quant_radius; }
    float  getOutlierCapacity() const { return config_.outlier_capacity; }
    std::array<size_t, 3> getDims() const { return config_.dims; }
    ErrorBoundMode getErrorBoundMode() const { return config_.eb_mode; }
    float getValueBase() const { return config_.precomputed_value_base; }
    bool  getZigzagCodes() const { return config_.zigzag_codes; }

    /// Returns the effective spatial dimensionality (1, 2, or 3).
    int ndim() const {
        if (config_.dims[2] > 1) return 3;
        if (config_.dims[1] > 1) return 2;
        return 1;
    }
    
    void setInverse(bool inverse) { is_inverse_ = inverse; }
    bool isInverse() const { return is_inverse_; }

    // ── Serialization ─────────────────────────────────────────────────────────
    
    uint16_t getStageTypeId() const override {
        return static_cast<uint16_t>(StageType::LORENZO_QUANT);
    }
    
    uint8_t getOutputDataType(size_t output_index) const override {
        switch (output_index) {
            case 0: return static_cast<uint8_t>(getCodeDataType());      // codes
            case 1: return static_cast<uint8_t>(getInputDataType());     // outlier_errors
            case 2: return static_cast<uint8_t>(DataType::UINT32);       // outlier_indices
            default: return static_cast<uint8_t>(DataType::UINT8);
        }
    }

    uint8_t getInputDataType(size_t /*input_index*/) const override {
        return static_cast<uint8_t>(getInputDataType());
    }
    
    size_t serializeHeader(size_t output_index, uint8_t* header_buffer, size_t max_size) const override {
        (void)output_index;  // Lorenzo uses same header for all outputs

        if (max_size < sizeof(LorenzoQuantConfig)) {
            throw std::runtime_error("Insufficient buffer for Lorenzo config");
        }

        LorenzoQuantConfig config;
        config.error_bound   = static_cast<float>(computed_abs_eb_);  // abs bound used by decompressor
        config.quant_radius  = static_cast<uint32_t>(config_.quant_radius);
        config.num_elements  = static_cast<uint32_t>(num_elements_);
        config.outlier_count = actual_outlier_count_;
        config.input_type    = getInputDataType();
        config.code_type     = getCodeDataType();
        config.ndim          = static_cast<uint8_t>(ndim());
        config.eb_mode       = static_cast<uint8_t>(config_.eb_mode);
        config.dim_x         = static_cast<uint32_t>(config_.dims[0]);
        config.dim_y         = static_cast<uint32_t>(config_.dims[1]);
        config.dim_z         = static_cast<uint32_t>(config_.dims[2]);
        config.user_eb       = static_cast<float>(config_.error_bound);  // original user-specified value
        config.value_base    = computed_value_base_;
        config.zigzag_codes  = config_.zigzag_codes ? uint8_t{1} : uint8_t{0};
        config.reserved[0]   = 0; config.reserved[1] = 0; config.reserved[2] = 0;

        std::memcpy(header_buffer, &config, sizeof(LorenzoQuantConfig));
        return sizeof(LorenzoQuantConfig);
    }
    
    size_t getMaxHeaderSize(size_t output_index) const override {
        (void)output_index;
        return sizeof(LorenzoQuantConfig);
    }
    
    void deserializeHeader(const uint8_t* header_buffer, size_t size) override {
        // Minimum size is the original 32-byte layout (before user_eb/value_base were added).
        constexpr size_t kLegacySize = 32;
        if (size < kLegacySize) {
            throw std::runtime_error("Invalid Lorenzo config size");
        }

        LorenzoQuantConfig config;
        std::memcpy(&config, header_buffer, std::min(size, sizeof(LorenzoQuantConfig)));

        // error_bound in the header is always the absolute bound used at compression.
        config_.error_bound  = config.error_bound;
        computed_abs_eb_     = static_cast<TInput>(config.error_bound);
        config_.quant_radius = static_cast<TCode>(config.quant_radius);
        num_elements_        = config.num_elements;
        actual_outlier_count_= config.outlier_count;
        // New fields: present only in headers written by v1+ (≥40B, added user_eb/value_base/eb_mode).
        constexpr size_t kV1Size = 40;
        if (size >= kV1Size) {
            config_.eb_mode                = static_cast<ErrorBoundMode>(config.eb_mode);
            config_.precomputed_value_base = config.value_base;
            computed_value_base_           = config.value_base;
        } else {
            config_.eb_mode                = ErrorBoundMode::ABS;
            config_.precomputed_value_base = 0.0f;
            computed_value_base_           = 0.0f;
        }
        // zigzag_codes field added in v2 (≥44B).
        if (size >= sizeof(LorenzoQuantConfig)) {
            config_.zigzag_codes = (config.zigzag_codes != 0);
        } else {
            config_.zigzag_codes = false;
        }

        // Restore spatial dimensions; handle old (pre-dims) files gracefully
        int eff_ndim = (config.ndim == 0) ? 1 : static_cast<int>(config.ndim);
        // dim_x: stored explicitly; fall back to derivation for old files
        if (config.dim_x > 0) {
            config_.dims[0] = config.dim_x;
        } else if (config.num_elements > 0) {
            size_t yz = std::max<size_t>(1, config.dim_y) * std::max<size_t>(1, config.dim_z);
            config_.dims[0] = config.num_elements / yz;
        } else {
            config_.dims[0] = 0;
        }
        if (eff_ndim >= 2) {
            config_.dims[1] = (config.dim_y > 0) ? config.dim_y : 1;
        } else {
            config_.dims[1] = 1;
        }
        if (eff_ndim >= 3) {
            config_.dims[2] = (config.dim_z > 0) ? config.dim_z : 1;
        } else {
            config_.dims[2] = 1;
        }
    }
    
private:
    Config config_;
    std::vector<size_t> actual_output_sizes_;
    std::vector<size_t> saved_output_sizes_;  // saved by saveState(), restored by restoreState()
    size_t num_elements_ = 0;              // Track for header
    uint32_t actual_outlier_count_ = 0;    // Track for header
    bool is_inverse_ = false;              // false = compress, true = decompress
    /// Actual absolute error bound used in kernel launches.
    /// For ABS mode this equals config_.error_bound.  For REL/NOA modes it is
    /// the converted value computed during execute() after the data scan.
    TInput computed_abs_eb_ = 0;
    /// Scaling factor used in the conversion: value_range (NOA) or max(|data|) (REL).
    /// Stored so serializeHeader() can embed it in the output stream.
    float computed_value_base_ = 0.0f;
    /// Stage-private 4-byte device scratch holding the live outlier count.
    /// Allocated lazily via `pool->allocatePersistentDevice(4, ...)` — see
    /// `initOutlierCountScratch()`. Used by the forward kernel as the atomic
    /// counter and D2H'd in `postStreamSync()`. The inverse path does NOT
    /// touch this — it reads the count from the deserialized FZM header and
    /// passes it as a `uint32_t` kernel-launch argument.
    uint32_t* d_outlier_count_scratch_ = nullptr;
    /// Pool that owns `d_outlier_count_scratch_` — captured at allocation
    /// time so the destructor can return it to the right pool.
    MemoryPool* persistent_pool_ = nullptr;

    /// Lazily allocate the 4-byte outlier-count scratch via the pool's
    /// persistent allocator. Idempotent; no-op if already allocated.
    void initOutlierCountScratch(MemoryPool* pool);
    
    DataType getInputDataType() const {
        if (std::is_same<TInput, float>::value) return DataType::FLOAT32;
        if (std::is_same<TInput, double>::value) return DataType::FLOAT64;
        return DataType::FLOAT32;
    }
    
    DataType getCodeDataType() const {
        if (std::is_same<TCode, uint8_t>::value) return DataType::UINT8;
        if (std::is_same<TCode, uint16_t>::value) return DataType::UINT16;
        if (std::is_same<TCode, uint32_t>::value) return DataType::UINT32;
        return DataType::UINT16;
    }
    
    size_t getMaxOutlierCount(size_t num_elements) const {
        return static_cast<size_t>(std::ceil(num_elements * config_.outlier_capacity));
    }
};

extern template class LorenzoQuantStage<float, uint16_t>;
extern template class LorenzoQuantStage<float, uint8_t>;
extern template class LorenzoQuantStage<double, uint16_t>;
extern template class LorenzoQuantStage<double, uint32_t>;

// Kernel launcher declarations — defined in lorenzo.cu.

template<typename TInput, typename TCode>
void launchLorenzoKernel(
    const TInput* d_input, size_t n,
    TInput ebx2_r, TCode quant_radius,
    TCode* d_codes, TInput* d_outlier_errors,
    uint32_t* d_outlier_indices, uint32_t* d_outlier_count,
    size_t max_outliers, int grid_size,
    bool zigzag_codes,
    fz::stream_t stream
);

template<typename TInput, typename TCode>
void launchLorenzoInverseKernel(
    const TCode* d_codes,
    const TInput* d_outlier_errors, const uint32_t* d_outlier_indices,
    uint32_t outlier_n,
    size_t n,
    TInput ebx2, TCode quant_radius,
    TInput* d_output,
    bool zigzag_codes,
    fz::stream_t stream, MemoryPool* pool
);

/// 2-D forward Lorenzo kernel launcher. `nx` is the fast (x) dimension.
template<typename TInput, typename TCode>
void launchLorenzoKernel2D(
    const TInput* d_input, size_t nx, size_t ny,
    TInput ebx2_r, TCode quant_radius,
    TCode* d_codes, TInput* d_outlier_errors,
    uint32_t* d_outlier_indices, uint32_t* d_outlier_count,
    size_t max_outliers,
    bool zigzag_codes,
    fz::stream_t stream
);

/// 2-D inverse Lorenzo kernel launcher.
template<typename TInput, typename TCode>
void launchLorenzoInverseKernel2D(
    const TCode* d_codes,
    const TInput* d_outlier_errors, const uint32_t* d_outlier_indices,
    uint32_t outlier_n,
    size_t nx, size_t ny,
    TInput ebx2, TCode quant_radius,
    TInput* d_output,
    bool zigzag_codes,
    fz::stream_t stream, MemoryPool* pool
);

/// 3-D forward Lorenzo kernel launcher.
template<typename TInput, typename TCode>
void launchLorenzoKernel3D(
    const TInput* d_input, size_t nx, size_t ny, size_t nz,
    TInput ebx2_r, TCode quant_radius,
    TCode* d_codes, TInput* d_outlier_errors,
    uint32_t* d_outlier_indices, uint32_t* d_outlier_count,
    size_t max_outliers,
    bool zigzag_codes,
    fz::stream_t stream
);

/// 3-D inverse Lorenzo kernel launcher.
template<typename TInput, typename TCode>
void launchLorenzoInverseKernel3D(
    const TCode* d_codes,
    const TInput* d_outlier_errors, const uint32_t* d_outlier_indices,
    uint32_t outlier_n,
    size_t nx, size_t ny, size_t nz,
    TInput ebx2, TCode quant_radius,
    TInput* d_output,
    bool zigzag_codes,
    fz::stream_t stream, MemoryPool* pool
);

} // namespace fz