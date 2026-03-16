#pragma once

#include "stage/stage.h"
#include "fzm_format.h"
#include <cuda_runtime.h>
#include <array>
#include <cstdint>
#include <cmath>
#include <cstring>

namespace fz {


// ===== Error Bound Mode =====

/**
 * How the user-specified error bound is interpreted.
 *
 * ABS  — absolute error bound.  The error for every element satisfies
 *         |x_orig - x_recon| <= eb.  This is the default.
 *
 * REL  — point-wise relative error bound (PFPL definition).
 *         e_rel = |x_orig - x_recon| / |x_orig| <= eb.
 *
 *         For LorenzoStage this is a GLOBAL approximation only:
 *           abs_eb = eb * max(|data|)
 *         Because Lorenzo quantises prediction *differences* with a fixed
 *         step size, a true per-element guarantee cannot be made.  Values
 *         much smaller than max(|data|) may exceed the per-element ratio.
 *
 *         For an exact per-element REL bound (PFPL log2-space algorithm),
 *         use QuantizerStage<TInput, uint32_t> with REL mode instead.
 *
 * NOA  — norm-of-absolute, a.k.a. value-range relative (PFPL definition).
 *         abs_eb = eb * (max(data) - min(data))
 *         Equivalent to what most other compressors call "relative".
 */
enum class ErrorBoundMode : uint8_t {
    ABS = 0,
    REL = 1,
    NOA = 2,
};

// ===== Lorenzo Stage-Specific Config =====
// Serialized into FZMBufferEntry.stage_config[128]

/**
 * Lorenzo predictor configuration for decompression
 * 
 * This structure is serialized by LorenzoStage::serializeHeader()
 * and fits into the generic 128-byte stage_config buffer in FZMBufferEntry.
 */
struct LorenzoConfig {
    float error_bound;        // Absolute error bound after mode conversion (4B)
    uint32_t quant_radius;    // Quantization radius (4B)
    uint32_t num_elements;    // Number of elements (4B)
    uint32_t outlier_count;   // Actual number of outliers (4B)
    DataType input_type;      // Original input type (1B)
    DataType code_type;       // Quantization code type (1B)
    uint8_t  ndim;            // Spatial dimensionality: 1, 2, or 3 (0 treated as 1) (1B)
    uint8_t  eb_mode;         // ErrorBoundMode (was reserved0) (1B)
    uint32_t dim_x;           // X dimension length (fastest, 0 = infer from num_elements) (4B)
    uint32_t dim_y;           // Y dimension length (1 for 1D) (4B)
    uint32_t dim_z;           // Z dimension length (1 for 1D/2D) (4B)
    float    user_eb;         // Original user-specified error bound (4B)
    float    value_base;      // value_range (NOA) or max(|data|) (REL) used in conversion (4B)
    uint8_t  zigzag_codes;    // 1 if quantization codes are zigzag-encoded, else 0 (1B)
    uint8_t  reserved[3];     // Reserved for future use, must be zero (3B)

    // Total: 44 bytes (fits easily in 128B stage_config)

    LorenzoConfig()
        : error_bound(0.0f), quant_radius(0), num_elements(0), outlier_count(0),
          input_type(DataType::FLOAT32), code_type(DataType::UINT16),
          ndim(1), eb_mode(0), dim_x(0), dim_y(1), dim_z(1),
          user_eb(0.0f), value_base(0.0f), zigzag_codes(0), reserved{0, 0, 0} {}
};
static_assert(sizeof(LorenzoConfig) <= FZM_STAGE_CONFIG_SIZE, "LorenzoConfig must fit in FZM_STAGE_CONFIG_SIZE");

/**
 * Lorenzo 1D predictor with quantization
 * 
 * Applies Lorenzo prediction (1D differences) with error-bounded quantization.
 * Produces quantization codes for predictable values and stores outliers separately.
 * 
 * Outputs:
 *   [0] codes: Quantization codes for all elements (TCode type)
 *   [1] outlier_errors: Prediction errors for outliers (TInput type)
 *   [2] outlier_indices: Indices of outlier elements (uint32_t)
 *   [3] outlier_count: Number of outliers (uint32_t)
 * 
 * Template parameters:
 *   TInput: Input data type (float, double)
 *   TCode: Quantization code type (uint8_t, uint16_t, uint32_t)
 */
template<typename TInput = float, typename TCode = uint16_t>
class LorenzoStage : public Stage {
public:
    /**
     * Configuration for Lorenzo predictor
     */
    struct Config {
        float error_bound = 1e-3;           // Error bound value (interpretation depends on eb_mode)
        int quant_radius = 32768;           // Quantization radius (2^15 for uint16_t)
        float outlier_capacity = 0.2f;      // Fraction of input size to allocate for outliers (0-1)
        // Spatial dimensions: dims[0]=x (fastest), dims[1]=y, dims[2]=z.
        // dims[0]==0 means "infer from num_elements at runtime" (valid for 1-D use).
        // dims[1]==1, dims[2]==1 → 1-D Lorenzo
        // dims[2]==1             → 2-D Lorenzo
        // otherwise             → 3-D Lorenzo
        std::array<size_t, 3> dims = {0, 1, 1};
        ErrorBoundMode eb_mode = ErrorBoundMode::ABS;
        // Pre-computed value_range (NOA) or max(|data|) (REL).
        // When 0 (default), execute() auto-computes it via a device scan.
        // Set to a positive value to skip the scan (e.g. when the caller
        // already knows value_range from a prior compression pass).
        float precomputed_value_base = 0.0f;        // When true, quantization codes are zigzag-encoded before storage.
        // Zigzag maps signed integers to unsigned: ..., -2→3, -1→1, 0→0, 1→2, 2→4, ...
        // This improves compressibility when codes cluster near zero.
        // Supported for 1-D/2-D/3-D Lorenzo.
        bool zigzag_codes = false;
        Config() = default;
        Config(TInput eb, TCode radius = 32768, float outlier_cap = 0.2f,
               std::array<size_t, 3> d = {0, 1, 1})
            : error_bound(eb), quant_radius(radius), outlier_capacity(outlier_cap),
              dims(d) {}
    };
    
    explicit LorenzoStage(const Config& config = Config());

    void execute(
        cudaStream_t stream,
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
    void postStreamSync(cudaStream_t stream) override;
    
    std::string getName() const override {
        switch (ndim()) {
            case 2:  return "Lorenzo2D";
            case 3:  return "Lorenzo3D";
            default: return "Lorenzo1D";
        }
    }
    size_t getNumInputs()  const override { return is_inverse_ ? 4 : 1; }
    size_t getNumOutputs() const override { return is_inverse_ ? 1 : 4; }
    
    std::vector<std::string> getOutputNames() const override {
        return {"codes", "outlier_errors", "outlier_indices", "outlier_count"};
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
    
    // Configuration accessors
    void setErrorBound(TInput error_bound) { config_.error_bound = error_bound; }
    void setQuantRadius(TCode radius) { config_.quant_radius = radius; }
    void setOutlierCapacity(float capacity) { config_.outlier_capacity = capacity; }
    void setDims(const std::array<size_t, 3>& dims) override { config_.dims = dims; }
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
    
    // Inverse mode: toggle between compression (false) and decompression (true)
    void setInverse(bool inverse) { is_inverse_ = inverse; }
    bool isInverse() const { return is_inverse_; }
    
    // ===== Decompression Support =====
    
    uint16_t getStageTypeId() const override {
        switch (ndim()) {
            case 2:  return static_cast<uint16_t>(StageType::LORENZO_2D);
            case 3:  return static_cast<uint16_t>(StageType::LORENZO_3D);
            default: return static_cast<uint16_t>(StageType::LORENZO_1D);
        }
    }
    
    uint8_t getOutputDataType(size_t output_index) const override {
        switch (output_index) {
            case 0: return static_cast<uint8_t>(getCodeDataType());      // codes
            case 1: return static_cast<uint8_t>(getInputDataType());     // outlier_errors
            case 2: return static_cast<uint8_t>(DataType::UINT32);       // outlier_indices
            case 3: return static_cast<uint8_t>(DataType::UINT32);       // outlier_count
            default: return static_cast<uint8_t>(DataType::UINT8);
        }
    }
    
    size_t serializeHeader(size_t output_index, uint8_t* header_buffer, size_t max_size) const override {
        (void)output_index;  // Lorenzo uses same header for all outputs

        if (max_size < sizeof(LorenzoConfig)) {
            throw std::runtime_error("Insufficient buffer for Lorenzo config");
        }

        LorenzoConfig config;
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

        std::memcpy(header_buffer, &config, sizeof(LorenzoConfig));
        return sizeof(LorenzoConfig);
    }
    
    size_t getMaxHeaderSize(size_t output_index) const override {
        (void)output_index;
        return sizeof(LorenzoConfig);
    }
    
    void deserializeHeader(const uint8_t* header_buffer, size_t size) override {
        // Minimum size is the original 32-byte layout (before user_eb/value_base were added).
        constexpr size_t kLegacySize = 32;
        if (size < kLegacySize) {
            throw std::runtime_error("Invalid Lorenzo config size");
        }

        LorenzoConfig config;
        std::memcpy(&config, header_buffer, std::min(size, sizeof(LorenzoConfig)));

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
        if (size >= sizeof(LorenzoConfig)) {
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
    /// Device pointer to the outlier_count output buffer.  Set during
    /// execute() (compress mode) and consumed once by postStreamSync().
    const void* d_outlier_count_ptr_ = nullptr;
    
    // Helper to get data type enums
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
    
    // Helper to get max outlier count based on capacity
    size_t getMaxOutlierCount(size_t num_elements) const {
        return static_cast<size_t>(std::ceil(num_elements * config_.outlier_capacity));
    }
};

// Explicit instantiations for common type combinations
extern template class LorenzoStage<float, uint16_t>;
extern template class LorenzoStage<float, uint8_t>;
extern template class LorenzoStage<double, uint16_t>;
extern template class LorenzoStage<double, uint32_t>;

// ========== Kernel Launcher Declarations ==========
// Forward declarations for the kernel launchers defined in lorenzo.cu.
// 1-D launchers already exist; 2-D and 3-D launchers are stubs until the
// cuSZ kernels are wired in.
// (All declarations are inside the already-open namespace fz.)

template<typename TInput, typename TCode>
void launchLorenzoKernel(
    const TInput* d_input, size_t n,
    TInput ebx2_r, TCode quant_radius,
    TCode* d_codes, TInput* d_outlier_errors,
    uint32_t* d_outlier_indices, uint32_t* d_outlier_count,
    size_t max_outliers, int grid_size,
    bool zigzag_codes,
    cudaStream_t stream
);

template<typename TInput, typename TCode>
void launchLorenzoInverseKernel(
    const TCode* d_codes,
    const TInput* d_outlier_errors, const uint32_t* d_outlier_indices,
    const uint32_t* d_outlier_count,
    size_t n, size_t max_outliers,
    TInput ebx2, TCode quant_radius,
    TInput* d_output,
    bool zigzag_codes,
    cudaStream_t stream, MemoryPool* pool
);

/// 2-D forward Lorenzo kernel launcher.
/// @param len3  {nx, ny, 1}  where nx is the fast (x) dimension.
template<typename TInput, typename TCode>
void launchLorenzoKernel2D(
    const TInput* d_input, size_t nx, size_t ny,
    TInput ebx2_r, TCode quant_radius,
    TCode* d_codes, TInput* d_outlier_errors,
    uint32_t* d_outlier_indices, uint32_t* d_outlier_count,
    size_t max_outliers,
    bool zigzag_codes,
    cudaStream_t stream
);

/// 2-D inverse Lorenzo kernel launcher.
template<typename TInput, typename TCode>
void launchLorenzoInverseKernel2D(
    const TCode* d_codes,
    const TInput* d_outlier_errors, const uint32_t* d_outlier_indices,
    const uint32_t* d_outlier_count,
    size_t nx, size_t ny, size_t max_outliers,
    TInput ebx2, TCode quant_radius,
    TInput* d_output,
    bool zigzag_codes,
    cudaStream_t stream, MemoryPool* pool
);

/// 3-D forward Lorenzo kernel launcher.
/// @param len3  {nx, ny, nz}.
template<typename TInput, typename TCode>
void launchLorenzoKernel3D(
    const TInput* d_input, size_t nx, size_t ny, size_t nz,
    TInput ebx2_r, TCode quant_radius,
    TCode* d_codes, TInput* d_outlier_errors,
    uint32_t* d_outlier_indices, uint32_t* d_outlier_count,
    size_t max_outliers,
    bool zigzag_codes,
    cudaStream_t stream
);

/// 3-D inverse Lorenzo kernel launcher.
template<typename TInput, typename TCode>
void launchLorenzoInverseKernel3D(
    const TCode* d_codes,
    const TInput* d_outlier_errors, const uint32_t* d_outlier_indices,
    const uint32_t* d_outlier_count,
    size_t nx, size_t ny, size_t nz, size_t max_outliers,
    TInput ebx2, TCode quant_radius,
    TInput* d_output,
    bool zigzag_codes,
    cudaStream_t stream, MemoryPool* pool
);

} // namespace fz