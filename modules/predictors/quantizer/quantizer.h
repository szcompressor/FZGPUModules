#pragma once

/**
 * @file quantizer.h
 * @brief Direct-value quantizer stage with error-bounded coding and lossless outlier fallback.
 */

#include "stage/stage.h"
#include "fzm_format.h"
#include "predictors/lorenzo/lorenzo.h"  // for ErrorBoundMode
#include <cuda_runtime.h>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

namespace fz {

/**
 * Serialized quantizer configuration stored in FZMBufferEntry.stage_config.
 * Written by `serializeHeader()`; read back by `deserializeHeader()`.
 * 36 bytes — fits within the 128-byte `FZM_STAGE_CONFIG_SIZE` limit.
 */
struct QuantizerConfig {
    float    abs_error_bound;   ///< Absolute EB after mode conversion (0 for REL).
    float    user_error_bound;  ///< Original user-specified EB.
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
    uint8_t  _pad[3];           ///< Alignment padding — must be zero.

    QuantizerConfig()
        : abs_error_bound(0.0f), user_error_bound(0.0f), value_base(0.0f),
          quant_radius(0), num_elements(0), outlier_count(0),
          input_type(DataType::FLOAT32), code_type(DataType::UINT16),
          eb_mode(0), zigzag_codes(0),
          outlier_threshold(std::numeric_limits<float>::infinity()),
          inplace_outliers(0), _pad{} {}
};
static_assert(sizeof(QuantizerConfig) <= FZM_STAGE_CONFIG_SIZE,
              "QuantizerConfig must fit in FZM_STAGE_CONFIG_SIZE");

/**
 * Direct-value quantizer with error-bounded coding and lossless outlier fallback.
 *
 * Unlike LorenzoQuantizerStage (which quantizes prediction *differences*), this stage
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
 * Outputs (compression mode):
 *   [0] codes         — quantization codes (TCode[n])
 *   [1] outlier_vals  — original values at outlier positions (TInput[k])
 *   [2] outlier_idxs  — indices of outlier positions (uint32_t[k])
 *   [3] outlier_count — number of outliers (uint32_t scalar)
 *
 * Inputs (decompression mode):
 *   same 4 buffers → reconstructed TInput[n]
 */
template<typename TInput = float, typename TCode = uint16_t>
class QuantizerStage : public Stage {
public:
    /** Construction parameters. */
    struct Config {
        float  error_bound           = 1e-4f;   ///< Error bound (interpretation set by `eb_mode`).
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

        Config() = default;
        Config(TInput eb, ErrorBoundMode mode = ErrorBoundMode::ABS,
               int radius = 32768, float outlier_cap = 0.05f)
            : error_bound(static_cast<float>(eb)), quant_radius(radius),
              outlier_capacity(outlier_cap), eb_mode(mode) {}
    };

    explicit QuantizerStage(const Config& config = Config());

    void execute(
        cudaStream_t stream,
        MemoryPool* pool,
        const std::vector<void*>& inputs,
        const std::vector<void*>& outputs,
        const std::vector<size_t>& sizes
    ) override;

    void postStreamSync(cudaStream_t stream) override;

    std::string getName() const override { return "Quantizer"; }

    size_t getNumInputs() const override {
        if (!is_inverse_) return 1;
        return isInplaceMode() ? 1 : 4;
    }
    size_t getNumOutputs() const override {
        if (is_inverse_) return 1;
        return isInplaceMode() ? 1 : 4;
    }

    std::vector<std::string> getOutputNames() const override {
        if (is_inverse_) return {"reconstructed"};
        if (isInplaceMode()) return {"codes"};
        return {"codes", "outlier_vals", "outlier_idxs", "outlier_count"};
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

    uint16_t getStageTypeId() const override {
        return static_cast<uint16_t>(StageType::QUANTIZER);
    }

    uint8_t getOutputDataType(size_t output_index) const override {
        if (is_inverse_) return static_cast<uint8_t>(getInputDataType());
        if (isInplaceMode()) return static_cast<uint8_t>(getCodeDataType()); // only codes
        switch (output_index) {
            case 0: return static_cast<uint8_t>(getCodeDataType());
            case 1: return static_cast<uint8_t>(getInputDataType());
            case 2: return static_cast<uint8_t>(DataType::UINT32);
            case 3: return static_cast<uint8_t>(DataType::UINT32);
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

    void setErrorBound(TInput eb)            { config_.error_bound = static_cast<float>(eb); }
    void setQuantRadius(int r)               { config_.quant_radius = r; }
    void setOutlierCapacity(float c)         { config_.outlier_capacity = c; }
    void setErrorBoundMode(ErrorBoundMode m) { config_.eb_mode = m; }
    void setValueBase(float vb)              { config_.precomputed_value_base = vb; }
    void setZigzagCodes(bool enable)         { config_.zigzag_codes = enable; }
    /// ABS/NOA: |x| >= threshold → lossless outlier regardless of bin (LC reference parameter).
    void setOutlierThreshold(float t)        { config_.outlier_threshold = t; }
    /// ABS/NOA: encode outliers in-place (raw float bits in codes array; no scatter buffers).
    void setInplaceOutliers(bool enable)     { config_.inplace_outliers = enable; }

    TInput         getErrorBound()        const { return static_cast<TInput>(config_.error_bound); }
    int            getQuantRadius()       const { return config_.quant_radius; }
    ErrorBoundMode getErrorBoundMode()    const { return config_.eb_mode; }
    float          getValueBase()         const { return config_.precomputed_value_base; }
    bool           getZigzagCodes()       const { return config_.zigzag_codes; }
    float          getOutlierThreshold()  const { return config_.outlier_threshold; }
    bool           getInplaceOutliers()   const { return config_.inplace_outliers; }

private:
    Config config_;
    Config saved_config_;
    std::vector<size_t> actual_output_sizes_;
    std::vector<size_t> saved_actual_output_sizes_;
    size_t   num_elements_        = 0;
    size_t   saved_num_elements_  = 0;
    uint32_t actual_outlier_count_= 0;
    uint32_t saved_actual_outlier_count_ = 0;
    bool     is_inverse_          = false;
    TInput   computed_abs_eb_     = static_cast<TInput>(1e-4);
    TInput   saved_computed_abs_eb_ = static_cast<TInput>(1e-4);
    float    computed_value_base_ = 0.0f;
    float    saved_computed_value_base_ = 0.0f;
    const void* d_outlier_count_ptr_ = nullptr;

    bool isInplaceMode() const {
        return config_.inplace_outliers
            && config_.eb_mode != ErrorBoundMode::REL;
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
