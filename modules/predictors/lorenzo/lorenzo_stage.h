#pragma once

/**
 * @file lorenzo_stage.h
 * @brief Plain integer Lorenzo predictor (delta coding / prefix sum). Lossless.
 */

#include "stage/stage.h"
#include "fzm_format.h"
#include "backend/types.h"
#include "fused/fused_block/warp_op_params.h"
#include <algorithm>
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
    uint8_t  centering;   ///< 1 if per-block mean centering is enabled, else 0.
    uint8_t  order;       ///< Prediction order: 0/1 = first, 2 = second. 0 reads as 1.
    uint32_t dim_x;       ///< X (fast) dimension.
    uint32_t dim_y;       ///< Y dimension (1 for 1-D).
    uint32_t dim_z;       ///< Z dimension (1 for 1-D/2-D).
    uint32_t block_size;  ///< 1-D block-local reset period; 0 = default N-D behavior.

    LorenzoConfig()
        : data_type(DataType::INT32), ndim(1), centering(0), order(1),
          dim_x(0), dim_y(1), dim_z(1), block_size(0) {}
};
static_assert(sizeof(LorenzoConfig) <= FZM_STAGE_CONFIG_SIZE,
              "LorenzoConfig must fit in FZM_STAGE_CONFIG_SIZE");

/**
 * Plain integer Lorenzo predictor (1-D, 2-D, 3-D). Lossless.
 *
 * Forward (compression): compute per-element delta from its neighbor(s).
 * Inverse (decompression): prefix sum to reconstruct original values.
 *
 * @note The optional block centering follows the FSZ centered-prediction
 *       component. This implementation was written independently from the FSZ
 *       paper; no FSZ source was copied. See `THIRD_PARTY.md`.
 *
 * @tparam T  Signed integer element type: int8_t, int16_t, int32_t, int64_t.
 */
template<typename T>
class LorenzoStage : public Stage {
    static_assert(std::is_integral<T>::value && std::is_signed<T>::value,
                  "LorenzoStage requires a signed integer type");
public:
    LorenzoStage() = default;

    /**
     * Construct with the block-mode parameters already set.
     *
     * Prefer this over `setBlockSize`/`setCentering` when enabling centering:
     * `Pipeline::addStage()` captures the stage's port count at add-time, and
     * centering adds a `"means"` port, so it must be known before the stage
     * joins the DAG. (Same reason `setDims()` must precede `addStage()`.)
     */
    explicit LorenzoStage(uint32_t block_size, bool centering = false,
                          uint8_t order = 1)
        : block_size_(block_size), centering_(centering), order_(order) {
        if (block_size > 1024)
            throw std::invalid_argument(
                "LorenzoStage: block_size must be in [0, 1024], got "
                + std::to_string(block_size));
        if (centering && block_size == 0)
            throw std::invalid_argument(
                "LorenzoStage: centering requires block_size > 0");
        if (order != 1 && order != 2)
            throw std::invalid_argument(
                "LorenzoStage: order must be 1 or 2, got " + std::to_string(order));
        if (order == 2 && block_size == 0)
            throw std::invalid_argument(
                "LorenzoStage: order 2 requires block_size > 0");
    }

    void setInverse(bool inv) override { is_inverse_ = inv; }
    bool isInverse() const override    { return is_inverse_; }

    void setDims(const std::array<size_t, 3>& dims) override { dims_ = dims; }
    void setDims(size_t x, size_t y = 1, size_t z = 1) { dims_ = {x, y, z}; }
    std::array<size_t, 3> getDims() const { return dims_; }

    /**
     * Set an explicit 1-D block-local reset period (cuSZp-style).
     *
     * `n == 0` (default): keep the current behavior — 1-D delta resets every
     * launch block (256), 2-D/3-D use the N-D inclusion-exclusion delta.
     * `n > 0`: force the **1-D** path over the flattened array, restarting the
     * prediction chain (`prev = 0`) every `n` elements, independent of the launch
     * configuration and of `dims_`. cuSZp uses `n = 32`.
     *
     * Must be in [1, 1024] (CUDA block limit; the inverse scans one segment per
     * CUDA block of `n` threads).
     */
    void setBlockSize(uint32_t n) {
        if (n > 1024)
            throw std::invalid_argument(
                "LorenzoStage::setBlockSize: n must be in [0, 1024], got "
                + std::to_string(n));
        block_size_ = n;
    }
    uint32_t getBlockSize() const { return block_size_; }

    /// Block-local only in 1-D block-reset mode (block_size_ > 0): each
    /// block_size_-element segment is a self-contained delta chain. The N-D
    /// default (block_size_ == 0) is a multi-dimensional stencil handled by a
    /// different driver and is not fused yet.
    FusionSpec getFusionSpec() const override {
        if (isInverse() || block_size_ == 0) return {};
        return FusionSpec{FusionAccess::BlockLocal, block_size_};
    }

    /// Warp-register predictor op (cuSZp2): 1-D Lorenzo, EPL=1 (block 32). Declares
    /// the device policy type + its packed params so the generic NVRTC warp runner
    /// composes the kernel with no per-predictor code. `inv2eb` is left 0 — the
    /// runner fills it from the resolved quantizer bound (see warp_op_params.h). n_ab
    /// left 0 ⇒ the runner uses the input element count (1-D needs no padding).
    FusedOpDecl getFusedOp() const override {
        if (isInverse() || block_size_ != 32u) return {};
        FusedOpDecl d;
        d.strategy       = FusionStrategy::WarpRegister;
        d.op_name        = "Lorenzo1DPredictor";
        d.include_header = "fused/fused_block/warp_fusion.cuh";
        d.elems_per_lane = 1;
        d.n_ab           = 0;
        fused::warp::Lorenzo1DParams p{0.0f};
        d.params.resize(sizeof(p));
        std::memcpy(d.params.data(), &p, sizeof(p));
        return d;
    }

    /**
     * Enable per-block mean centering (FSZ-style adaptive centering).
     *
     * Subtracts the block's integer mean `mu` from the values before predicting.
     * Because the k-th order difference of a constant is zero
     * (`delta(q - mu) == delta(q)`), this changes **only the first residual of
     * each block** — the one element with no predecessor, which would otherwise
     * be a raw value. On data with a large constant offset (pressure near
     * 1000 hPa, temperature in Kelvin) that raw value dominates the block's
     * magnitude and inflates a downstream fixed-rate coder's bit width for the
     * whole block; centering drops it to `q_0 - mu`.
     *
     * Requires block mode (`setBlockSize(n)` with `n > 0`) — there is no
     * per-block mean without blocks. Adds a second output port, `"means"`
     * (one `T` per block), which the inverse consumes as its second input.
     *
     * @note The `sizeof(T)` bytes per block of `mu` are only worth paying when
     *       the block is long: at `block_size = 32` the overhead is
     *       `8/32 = 0.25` bits per element to fix 1 residual in 32, which
     *       generally loses. Pair with the large block sizes that cross-block
     *       prediction wants (256+), where the overhead is under 0.03 bits per
     *       element.
     *
     * @warning Must be set **before** `Pipeline::addStage()` captures the port
     *          count — pass it to the `LorenzoStage(block_size, centering)`
     *          constructor rather than calling this on an already-added stage.
     */
    void setCentering(bool enable) { centering_ = enable; }
    bool getCentering() const { return centering_; }

    /**
     * Prediction order: 1 (first difference, the default) or 2 (second
     * difference, FSZ's LZ2).
     *
     * Second order predicts each element from the *trend* of the two before it
     * rather than from the previous value alone, so it annihilates a linear
     * ramp exactly where first order leaves a constant non-zero residual. That
     * makes it the right choice on fields with a smooth gradient (geopotential
     * height, temperature profiles, pressure gradients) and the wrong choice on
     * piecewise-constant fields, where it roughly doubles the residual.
     *
     * Requires block mode (`setBlockSize(n)` with `n > 0`): the second
     * difference is taken within a reset segment, and the N-D
     * inclusion-exclusion path has no second-order analogue here. Costs one
     * extra element of raw seed per block (both `e_0` and `e_1` lack full
     * predecessors), so it pairs with long blocks and with `setCentering()`.
     */
    void setOrder(uint8_t k) {
        if (k != 1 && k != 2)
            throw std::invalid_argument(
                "LorenzoStage::setOrder: order must be 1 or 2, got " + std::to_string(k));
        order_ = k;
    }
    uint8_t getOrder() const { return order_; }

    int ndim() const {
        if (dims_[2] > 1) return 3;
        if (dims_[1] > 1) return 2;
        return 1;
    }

    void execute(
        fz::stream_t stream,
        MemoryPool* pool,
        const std::vector<void*>& inputs,
        const std::vector<void*>& outputs,
        const std::vector<size_t>& sizes
    ) override;

    std::string getName() const override { return "Lorenzo"; }

    /// Centering adds a `"means"` port: a second output forward, a second input inverse.
    size_t getNumInputs()  const override {
        return (is_inverse_ && centeringActive()) ? 2 : 1;
    }
    size_t getNumOutputs() const override {
        return (!is_inverse_ && centeringActive()) ? 2 : 1;
    }

    std::vector<std::string> getOutputNames() const override {
        if (centeringActive()) return {"output", "means"};
        return {"output"};
    }

    std::vector<size_t> estimateOutputSizes(
        const std::vector<size_t>& input_sizes
    ) const override {
        const size_t in = input_sizes.empty() ? 0 : input_sizes[0];
        if (!centeringActive()) return {in};
        return {in, numBlocks(in / sizeof(T)) * sizeof(T)};
    }

    std::unordered_map<std::string, size_t>
    getActualOutputSizesByName() const override {
        if (centeringActive())
            return {{"output", actual_output_size_}, {"means", actual_means_size_}};
        return {{"output", actual_output_size_}};
    }

    size_t getActualOutputSize(int index) const override {
        if (index == 0) return actual_output_size_;
        if (index == 1 && centeringActive()) return actual_means_size_;
        return 0;
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
        cfg.block_size = block_size_;
        cfg.centering  = centeringActive() ? 1u : 0u;
        cfg.order      = order_;
        std::memcpy(buf, &cfg, sizeof(LorenzoConfig));
        return sizeof(LorenzoConfig);
    }

    void deserializeHeader(const uint8_t* buf, size_t size) override {
        // Accept legacy 16-byte headers (no block_size field).
        constexpr size_t kMinSize = 16;
        if (size < kMinSize)
            throw std::runtime_error("LorenzoStage: header too small");
        LorenzoConfig cfg;  // default-constructed: block_size = 0
        std::memcpy(&cfg, buf, std::min(size, sizeof(LorenzoConfig)));
        int eff_ndim = (cfg.ndim == 0) ? 1 : static_cast<int>(cfg.ndim);
        dims_[0] = cfg.dim_x;
        dims_[1] = (eff_ndim >= 2) ? cfg.dim_y : 1;
        dims_[2] = (eff_ndim >= 3) ? cfg.dim_z : 1;
        block_size_ = (size >= sizeof(LorenzoConfig)) ? cfg.block_size : 0;
        // `centering` occupies a byte that legacy writers zeroed as `reserved`,
        // so old archives decode as centering-off without a version bump.
        centering_  = (cfg.centering != 0);
        // `order` reuses a byte legacy writers zeroed as `reserved`; 0 reads as
        // first order, so pre-LZ2 archives decode unchanged.
        order_      = (cfg.order == 2) ? 2u : 1u;
    }

    size_t getMaxHeaderSize(size_t /*output_index*/) const override {
        return sizeof(LorenzoConfig);
    }

private:
    bool is_inverse_         = false;
    size_t actual_output_size_ = 0;
    size_t actual_means_size_  = 0;
    std::array<size_t, 3> dims_ = {0, 1, 1};
    uint32_t block_size_ = 0;  ///< 0 = default N-D behavior; >0 = 1-D block-local reset.
    bool centering_ = false;   ///< Per-block mean centering (block mode only).
    uint8_t order_ = 1;        ///< Prediction order: 1 or 2 (block mode only).

    /// Centering is only meaningful in block mode — there is no per-block mean
    /// without blocks. Keeps the port count consistent if `setCentering(true)`
    /// is called without `setBlockSize`; `execute()` rejects that combination.
    bool centeringActive() const { return centering_ && block_size_ > 0; }

    /// Number of `block_size_`-element reset segments covering `n` elements.
    size_t numBlocks(size_t n) const {
        return (block_size_ == 0) ? 0 : (n + block_size_ - 1) / block_size_;
    }

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
    const T* d_input, T* d_output, size_t n, fz::stream_t stream,
    unsigned block_threads = 256);

template<typename T>
void launchLorenzoPrefixSumKernel1D(
    const T* d_input, T* d_output, size_t n, fz::stream_t stream,
    unsigned block_threads = 256);

/// Block-mode forward with per-block mean centering. Writes one mean per block
/// to `d_means` (`ceil(n / block_threads)` elements) and centers only the first
/// residual of each block.
template<typename T>
void launchLorenzoDeltaCentered1D(
    const T* d_input, T* d_output, T* d_means, size_t n, fz::stream_t stream,
    unsigned block_threads);

/// Unified block-mode inverse: `passes` segmented prefix sums (1 = LZ1,
/// 2 = LZ2) followed by a uniform `+ mu` when `d_means` is non-null. One CTA per
/// reset segment with several elements per thread, so the CTA width no longer
/// tracks the segment length.
template<typename T>
void launchLorenzoSegmentedScan(
    const T* d_input, const T* d_means, T* d_output, size_t n, fz::stream_t stream,
    unsigned block_threads, int passes);

/// Block-mode second-order (LZ2) forward. `d_means` may be nullptr (no
/// centering); when non-null it also writes one mean per block.
template<typename T>
void launchLorenzo2Delta1D(
    const T* d_input, T* d_output, T* d_means, size_t n, fz::stream_t stream,
    unsigned block_threads);

template<typename T>
void launchLorenzoDeltaKernel2D(
    const T* d_input, T* d_output, size_t nx, size_t ny, fz::stream_t stream);

template<typename T>
void launchLorenzoPrefixSumKernel2D(
    const T* d_input, T* d_output, size_t nx, size_t ny, fz::stream_t stream);

template<typename T>
void launchLorenzoDeltaKernel3D(
    const T* d_input, T* d_output, size_t nx, size_t ny, size_t nz, fz::stream_t stream);

template<typename T>
void launchLorenzoPrefixSumKernel3D(
    const T* d_input, T* d_output, size_t nx, size_t ny, size_t nz, fz::stream_t stream);

} // namespace fz
