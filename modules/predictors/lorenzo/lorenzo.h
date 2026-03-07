#pragma once

#include "stage/stage.h"
#include "fzm_format.h"
#include <cuda_runtime.h>
#include <array>
#include <cstdint>
#include <cmath>
#include <cstring>

namespace fz {
    

// ===== Lorenzo Stage-Specific Config =====
// Serialized into FZMBufferEntry.stage_config[128]

/**
 * Lorenzo predictor configuration for decompression
 * 
 * This structure is serialized by LorenzoStage::serializeHeader()
 * and fits into the generic 128-byte stage_config buffer in FZMBufferEntry.
 */
struct LorenzoConfig {
    float error_bound;        // Error bound used in compression (4B)
    uint32_t quant_radius;    // Quantization radius (4B)
    uint32_t num_elements;    // Number of elements (4B)
    uint32_t outlier_count;   // Actual number of outliers (4B)
    DataType input_type;      // Original input type (1B)
    DataType code_type;       // Quantization code type (1B)
    uint8_t  ndim;            // Spatial dimensionality: 1, 2, or 3 (0 treated as 1) (1B)
    uint8_t  reserved0;       // Padding (1B)
    uint32_t dim_x;           // X dimension length (fastest, 0 = infer from num_elements) (4B)
    uint32_t dim_y;           // Y dimension length (1 for 1D) (4B)
    uint32_t dim_z;           // Z dimension length (1 for 1D/2D) (4B)

    // Total: 32 bytes (fits easily in 128B stage_config)

    LorenzoConfig()
        : error_bound(0.0f), quant_radius(0), num_elements(0), outlier_count(0),
          input_type(DataType::FLOAT32), code_type(DataType::UINT16),
          ndim(1), reserved0(0), dim_x(0), dim_y(1), dim_z(1) {}
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
        float error_bound = 1e-3;           // Error bound for quantization
        int quant_radius = 32768;           // Quantization radius (2^15 for uint16_t)
        float outlier_capacity = 0.2f;      // Fraction of input size to allocate for outliers (0-1)
        // Spatial dimensions: dims[0]=x (fastest), dims[1]=y, dims[2]=z.
        // dims[0]==0 means "infer from num_elements at runtime" (valid for 1-D use).
        // dims[1]==1, dims[2]==1 → 1-D Lorenzo
        // dims[2]==1             → 2-D Lorenzo
        // otherwise             → 3-D Lorenzo
        std::array<size_t, 3> dims = {0, 1, 1};

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
    void setDims(std::array<size_t, 3> dims) { config_.dims = dims; }
    void setDims(size_t x, size_t y = 1, size_t z = 1) { config_.dims = {x, y, z}; }

    TInput getErrorBound() const { return config_.error_bound; }
    TCode  getQuantRadius() const { return config_.quant_radius; }
    float  getOutlierCapacity() const { return config_.outlier_capacity; }
    std::array<size_t, 3> getDims() const { return config_.dims; }

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
        config.error_bound   = config_.error_bound;
        config.quant_radius  = static_cast<uint32_t>(config_.quant_radius);
        config.num_elements  = static_cast<uint32_t>(num_elements_);
        config.outlier_count = actual_outlier_count_;
        config.input_type    = getInputDataType();
        config.code_type     = getCodeDataType();
        config.ndim          = static_cast<uint8_t>(ndim());
        config.dim_x         = static_cast<uint32_t>(config_.dims[0]);
        config.dim_y         = static_cast<uint32_t>(config_.dims[1]);
        config.dim_z         = static_cast<uint32_t>(config_.dims[2]);

        std::memcpy(header_buffer, &config, sizeof(LorenzoConfig));
        return sizeof(LorenzoConfig);
    }
    
    size_t getMaxHeaderSize(size_t output_index) const override {
        (void)output_index;
        return sizeof(LorenzoConfig);
    }
    
    void deserializeHeader(const uint8_t* header_buffer, size_t size) override {
        if (size < sizeof(LorenzoConfig)) {
            throw std::runtime_error("Invalid Lorenzo config size");
        }

        LorenzoConfig config;
        std::memcpy(&config, header_buffer, sizeof(LorenzoConfig));

        config_.error_bound  = config.error_bound;
        config_.quant_radius = static_cast<TCode>(config.quant_radius);
        num_elements_        = config.num_elements;
        actual_outlier_count_= config.outlier_count;

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
    size_t num_elements_ = 0;           // Track for header
    uint32_t actual_outlier_count_ = 0; // Track for header
    bool is_inverse_ = false;           // false = compress, true = decompress
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
    size_t max_outliers, cudaStream_t stream
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
    size_t max_outliers, cudaStream_t stream
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
    cudaStream_t stream, MemoryPool* pool
);

} // namespace fz