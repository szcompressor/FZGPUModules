#pragma once

/**
 * @file cdf97_stage.h
 * @brief CDF 9/7 biorthogonal wavelet transform stage (SPERR's DWT front-half).
 *
 * A dimension-aware, size-preserving, lossless, invertible floating-point basis
 * change: it replaces a field with its multi-level CDF 9/7 wavelet coefficients
 * (same element type, same element count). It is the decorrelating transform of
 * the SPERR compressor (CDF 9/7 DWT -> SPECK); this stage is the DWT only.
 *
 * @note **Prior work.** The transform, its lifting constants, the symmetric
 *       boundary handling, the dyadic level count, and the 3-D dyadic vs
 *       wavelet-packet selection are ported from NCAR/SPERR
 *       (https://github.com/NCAR/SPERR). The `double` path is validated
 *       bit-exact against `sperr::CDF97`. See `THIRD_PARTY.md` and the kernel
 *       headers (`cdf97_lifting.cuh`, `cdf97_kernels.cuh`).
 *
 * ### Precision
 *
 * `TInput = double` reproduces SPERR's coefficients bit-for-bit (SPERR does the
 * whole transform in double). `TInput = float` is a faster, deliberately NOT
 * bit-exact variant (constants derived in double, arithmetic in float).
 *
 * ### Ports & shape
 *
 * One input, one output, identical type and size. Dimensionality (1/2/3-D) comes
 * from `setDims()` — the pipeline pushes its dims into every stage at add-time —
 * and is serialized into the FZM header so the inverse can reconstruct it.
 *
 * ### Current limitation
 *
 * Each transform line is processed in shared memory, so the largest dimension
 * must satisfy `maxdim * sizeof(TInput) <= 48 KiB` (6144 doubles / 12288 floats).
 * A larger extent throws; the long-line fallback is future work.
 */

#include "stage/stage.h"
#include "fzm_format.h"
#include "backend/types.h"
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
 * Serialized CDF97Stage config (FZMBufferEntry.stage_config). 16 bytes.
 */
struct Cdf97Config {
    DataType data_type;   ///< Floating-point element type (1B): FLOAT32 or FLOAT64.
    uint8_t  ndim;        ///< Spatial dimensionality 1/2/3 (0 treated as 1).
    uint8_t  reserved[2]; ///< Must be zero.
    uint32_t dim_x;       ///< X (fast) dimension.
    uint32_t dim_y;       ///< Y dimension (1 for 1-D).
    uint32_t dim_z;       ///< Z dimension (1 for 1-D/2-D).

    Cdf97Config()
        : data_type(DataType::FLOAT64), ndim(1), reserved{0, 0},
          dim_x(0), dim_y(1), dim_z(1) {}
};
static_assert(sizeof(Cdf97Config) <= FZM_STAGE_CONFIG_SIZE,
              "Cdf97Config must fit in FZM_STAGE_CONFIG_SIZE");

/**
 * CDF 9/7 wavelet transform stage. Lossless, invertible, size-preserving.
 *
 * @tparam TInput Floating-point element type (`float` or `double`). `double` is
 *         bit-exact with SPERR; `float` is a faster non-bit-exact variant.
 */
template <typename TInput = double>
class Cdf97Stage : public Stage {
    static_assert(std::is_floating_point<TInput>::value,
                  "Cdf97Stage: TInput must be a floating-point type.");

public:
    /// Largest transform line that fits the shared-memory scheme, in elements.
    static constexpr size_t kMaxLineElems = (48u * 1024u) / sizeof(TInput);

    Cdf97Stage() = default;

    // ── Stage control ─────────────────────────────────────────────────────────
    void setInverse(bool inv) override { is_inverse_ = inv; }
    bool isInverse() const override    { return is_inverse_; }

    void setDims(const std::array<size_t, 3>& dims) override { dims_ = dims; }
    void setDims(size_t x, size_t y = 1, size_t z = 1)       { dims_ = {x, y, z}; }
    std::array<size_t, 3> getDims() const                    { return dims_; }

    // ── Execution ─────────────────────────────────────────────────────────────
    void execute(cudaStream_t stream, MemoryPool* pool,
                 const std::vector<void*>& inputs,
                 const std::vector<void*>& outputs,
                 const std::vector<size_t>& sizes) override;

    /// No mid-execute sync or D2H: every kernel is enqueued on `stream`.
    bool isGraphCompatible() const override { return true; }

    // ── Metadata ──────────────────────────────────────────────────────────────
    std::string getName() const override { return "CDF97"; }
    size_t getNumInputs()  const override { return 1; }
    size_t getNumOutputs() const override { return 1; }

    std::vector<size_t> estimateOutputSizes(
        const std::vector<size_t>& input_sizes) const override {
        return { input_sizes.empty() ? 0 : input_sizes[0] };  // size-preserving
    }

    std::unordered_map<std::string, size_t>
    getActualOutputSizesByName() const override {
        return {{"output", actual_output_size_}};
    }
    size_t getActualOutputSize(int index) const override {
        return index == 0 ? actual_output_size_ : 0;
    }

    // ── Type system ───────────────────────────────────────────────────────────
    uint16_t getStageTypeId() const override {
        return static_cast<uint16_t>(StageType::CDF97);
    }
    uint8_t getOutputDataType(size_t /*output_index*/) const override {
        return static_cast<uint8_t>(elementDataType());
    }
    uint8_t getInputDataType(size_t /*input_index*/) const override {
        return static_cast<uint8_t>(elementDataType());
    }

    // ── Serialization ─────────────────────────────────────────────────────────
    size_t serializeHeader(size_t /*output_index*/, uint8_t* buf, size_t max_size) const override {
        if (max_size < sizeof(Cdf97Config))
            throw std::runtime_error("Cdf97Stage: header buffer too small");
        Cdf97Config cfg;
        cfg.data_type = elementDataType();
        cfg.ndim      = static_cast<uint8_t>(ndim());
        cfg.dim_x     = static_cast<uint32_t>(dims_[0]);
        cfg.dim_y     = static_cast<uint32_t>(dims_[1]);
        cfg.dim_z     = static_cast<uint32_t>(dims_[2]);
        std::memcpy(buf, &cfg, sizeof(cfg));
        return sizeof(cfg);
    }
    void deserializeHeader(const uint8_t* buf, size_t size) override {
        if (size < sizeof(Cdf97Config))
            throw std::runtime_error("Cdf97Stage: header too small");
        Cdf97Config cfg;
        std::memcpy(&cfg, buf, sizeof(cfg));
        const int eff = (cfg.ndim == 0) ? 1 : static_cast<int>(cfg.ndim);
        dims_[0] = cfg.dim_x;
        dims_[1] = (eff >= 2) ? cfg.dim_y : 1;
        dims_[2] = (eff >= 3) ? cfg.dim_z : 1;
    }
    size_t getMaxHeaderSize(size_t /*output_index*/) const override {
        return sizeof(Cdf97Config);
    }

    void saveState()    override { saved_dims_ = dims_; }
    void restoreState() override { dims_ = saved_dims_; }

private:
    bool   is_inverse_         = false;
    size_t actual_output_size_ = 0;
    std::array<size_t, 3> dims_       = {0, 1, 1};
    std::array<size_t, 3> saved_dims_ = {0, 1, 1};

    int ndim() const {
        if (dims_[2] > 1) return 3;
        if (dims_[1] > 1) return 2;
        return 1;
    }
    static DataType elementDataType() {
        return std::is_same<TInput, double>::value ? DataType::FLOAT64
                                                   : DataType::FLOAT32;
    }
};

extern template class Cdf97Stage<float>;
extern template class Cdf97Stage<double>;

} // namespace fz
