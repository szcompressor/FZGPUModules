#pragma once

/**
 * @file lorenzo_stage.h
 * @brief Plain integer Lorenzo predictor (delta coding / prefix sum). Lossless.
 */

#include "stage/stage.h"
#include "fzm_format.h"
#include <cuda_runtime.h>
#include <array>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace fz {

/**
 * Serialized Lorenzo predictor config stored in FZMBufferEntry.stage_config.
 *
 * Fits in the 128-byte FZM_STAGE_CONFIG_SIZE limit (currently 16 bytes).
 */
struct LorenzoConfig {
    DataType data_type;   ///< Signed integer element type (1B).
    uint8_t  ndim;        ///< Spatial dimensionality 1/2/3 (0 treated as 1).
    uint8_t  reserved[2]; ///< Must be zero.
    uint32_t dim_x;       ///< X (fast) dimension.
    uint32_t dim_y;       ///< Y dimension (1 for 1-D).
    uint32_t dim_z;       ///< Z dimension (1 for 1-D/2-D).

    LorenzoConfig()
        : data_type(DataType::INT32), ndim(1), reserved{0, 0},
          dim_x(0), dim_y(1), dim_z(1) {}
};
static_assert(sizeof(LorenzoConfig) <= FZM_STAGE_CONFIG_SIZE,
              "LorenzoConfig must fit in FZM_STAGE_CONFIG_SIZE");

/**
 * Plain integer Lorenzo predictor (1-D, 2-D, 3-D). Lossless.
 *
 * Forward (compression): compute per-element delta from its neighbor(s).
 * Inverse (decompression): prefix sum to reconstruct original values.
 *
 * @tparam T  Signed integer element type: int8_t, int16_t, int32_t, int64_t.
 */
template<typename T>
class LorenzoStage : public Stage {
    static_assert(std::is_integral<T>::value && std::is_signed<T>::value,
                  "LorenzoStage requires a signed integer type");
public:
    LorenzoStage() = default;

    void setInverse(bool inv) override { is_inverse_ = inv; }
    bool isInverse() const override    { return is_inverse_; }

    void setDims(const std::array<size_t, 3>& dims) override { dims_ = dims; }
    void setDims(size_t x, size_t y = 1, size_t z = 1) { dims_ = {x, y, z}; }
    std::array<size_t, 3> getDims() const { return dims_; }

    int ndim() const {
        if (dims_[2] > 1) return 3;
        if (dims_[1] > 1) return 2;
        return 1;
    }

    void execute(
        cudaStream_t stream,
        MemoryPool* pool,
        const std::vector<void*>& inputs,
        const std::vector<void*>& outputs,
        const std::vector<size_t>& sizes
    ) override;

    std::string getName() const override { return "Lorenzo"; }
    size_t getNumInputs()  const override { return 1; }
    size_t getNumOutputs() const override { return 1; }

    std::vector<size_t> estimateOutputSizes(
        const std::vector<size_t>& input_sizes
    ) const override {
        return {input_sizes.empty() ? 0 : input_sizes[0]};
    }

    std::unordered_map<std::string, size_t>
    getActualOutputSizesByName() const override {
        return {{"output", actual_output_size_}};
    }

    size_t getActualOutputSize(int index) const override {
        return (index == 0) ? actual_output_size_ : 0;
    }

    uint16_t getStageTypeId() const override {
        return static_cast<uint16_t>(StageType::LORENZO);
    }

    uint8_t getOutputDataType(size_t /*output_index*/) const override {
        return static_cast<uint8_t>(getElementDataType());
    }

    uint8_t getInputDataType(size_t /*input_index*/) const override {
        return static_cast<uint8_t>(getElementDataType());
    }

    size_t serializeHeader(size_t /*output_index*/, uint8_t* buf, size_t max_size) const override {
        if (max_size < sizeof(LorenzoConfig))
            throw std::runtime_error("LorenzoStage: header buffer too small");
        LorenzoConfig cfg;
        cfg.data_type = getElementDataType();
        cfg.ndim      = static_cast<uint8_t>(ndim());
        cfg.dim_x     = static_cast<uint32_t>(dims_[0]);
        cfg.dim_y     = static_cast<uint32_t>(dims_[1]);
        cfg.dim_z     = static_cast<uint32_t>(dims_[2]);
        std::memcpy(buf, &cfg, sizeof(LorenzoConfig));
        return sizeof(LorenzoConfig);
    }

    void deserializeHeader(const uint8_t* buf, size_t size) override {
        if (size < sizeof(LorenzoConfig))
            throw std::runtime_error("LorenzoStage: header too small");
        LorenzoConfig cfg;
        std::memcpy(&cfg, buf, sizeof(LorenzoConfig));
        int eff_ndim = (cfg.ndim == 0) ? 1 : static_cast<int>(cfg.ndim);
        dims_[0] = cfg.dim_x;
        dims_[1] = (eff_ndim >= 2) ? cfg.dim_y : 1;
        dims_[2] = (eff_ndim >= 3) ? cfg.dim_z : 1;
    }

    size_t getMaxHeaderSize(size_t /*output_index*/) const override {
        return sizeof(LorenzoConfig);
    }

private:
    bool is_inverse_         = false;
    size_t actual_output_size_ = 0;
    std::array<size_t, 3> dims_ = {0, 1, 1};

    static DataType getElementDataType() {
        if (std::is_same<T, int8_t>::value)  return DataType::INT8;
        if (std::is_same<T, int16_t>::value) return DataType::INT16;
        if (std::is_same<T, int32_t>::value) return DataType::INT32;
        if (std::is_same<T, int64_t>::value) return DataType::INT64;
        return DataType::INT32;
    }
};

extern template class LorenzoStage<int8_t>;
extern template class LorenzoStage<int16_t>;
extern template class LorenzoStage<int32_t>;
extern template class LorenzoStage<int64_t>;

// Kernel launcher declarations — defined in lorenzo_stage.cu.

template<typename T>
void launchLorenzoDeltaKernel1D(
    const T* d_input, T* d_output, size_t n, cudaStream_t stream);

template<typename T>
void launchLorenzoPrefixSumKernel1D(
    const T* d_input, T* d_output, size_t n, cudaStream_t stream);

template<typename T>
void launchLorenzoDeltaKernel2D(
    const T* d_input, T* d_output, size_t nx, size_t ny, cudaStream_t stream);

template<typename T>
void launchLorenzoPrefixSumKernel2D(
    const T* d_input, T* d_output, size_t nx, size_t ny, cudaStream_t stream);

template<typename T>
void launchLorenzoDeltaKernel3D(
    const T* d_input, T* d_output, size_t nx, size_t ny, size_t nz, cudaStream_t stream);

template<typename T>
void launchLorenzoPrefixSumKernel3D(
    const T* d_input, T* d_output, size_t nx, size_t ny, size_t nz, cudaStream_t stream);

} // namespace fz
