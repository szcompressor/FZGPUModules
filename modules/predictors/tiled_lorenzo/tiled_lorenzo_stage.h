#pragma once

/**
 * @file tiled_lorenzo_stage.h
 * @brief Dimension-aware (tiled separable) Lorenzo predictor — cuSZp3 delta. Lossless.
 *
 * Within each fixed-size tile (8x8 in 2-D, 4x4x4 in 3-D by default) the predictor
 * applies a *separable* integer Lorenzo: the leading x-column predicts down y, the
 * leading y/z edges predict down z, and every other element predicts from its x
 * neighbor. The forward pass emits its output in **tile-major** order (each tile's
 * elements contiguous, edge tiles zero-padded) so a downstream
 * `AdaptiveBitpackStage(block_size = tile_elems)` packs each tile as exactly one
 * block — reproducing cuSZp3's compression ratio. The inverse un-tiles back to the
 * original natural (row-major) order.
 *
 * Supported element types: signed `int16_t`, `int32_t` (linear quantizer codes).
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
 * Serialized TiledLorenzo config stored in FZMBufferEntry.stage_config.
 * 20 bytes — fits within FZM_STAGE_CONFIG_SIZE.
 */
struct TiledLorenzoConfig {
    DataType data_type;   ///< Signed integer element type (1B): INT16 / INT32.
    uint8_t  ndim;        ///< Spatial dimensionality 1/2/3.
    uint8_t  tile_x;      ///< Tile extent in x (fast dim).
    uint8_t  tile_y;      ///< Tile extent in y (1 for 1-D).
    uint8_t  tile_z;      ///< Tile extent in z (1 for 1-D/2-D).
    uint8_t  reserved[3]; ///< Must be zero.
    uint32_t dim_x;       ///< X (fast) dimension.
    uint32_t dim_y;       ///< Y dimension (1 for 1-D).
    uint32_t dim_z;       ///< Z dimension (1 for 1-D/2-D).

    TiledLorenzoConfig()
        : data_type(DataType::INT32), ndim(2),
          tile_x(8), tile_y(8), tile_z(1), reserved{0, 0, 0},
          dim_x(0), dim_y(1), dim_z(1) {}
};
static_assert(sizeof(TiledLorenzoConfig) <= FZM_STAGE_CONFIG_SIZE,
              "TiledLorenzoConfig must fit in FZM_STAGE_CONFIG_SIZE");

/**
 * Dimension-aware (tiled separable) Lorenzo predictor (cuSZp3). Lossless.
 *
 * @note **Prior work:** the dimension-aware separable delta is the cuSZp3 design
 *       (Yafan Huang et al., SC'25, BSD-3-Clause). This stage is a **direct port
 *       of the cuSZp3 delta kernel logic** (cuSZp_kernels_{2D,3D}_f32.cu); the
 *       tile-major modular decomposition, FZM header, and MemoryPool integration
 *       are FZGPUModules code. The cuSZp BSD-3-Clause copyright is reproduced in
 *       `THIRD_PARTY.md`. See also `memory/cuszp_stages.md` (Part 8).
 *
 * @tparam T  Signed element type: `int16_t` or `int32_t`.
 */
template<typename T>
class TiledLorenzoStage : public Stage {
    static_assert(std::is_same_v<T, int16_t> || std::is_same_v<T, int32_t>,
                  "TiledLorenzoStage: T must be int16_t or int32_t.");
public:
    TiledLorenzoStage() = default;
    ~TiledLorenzoStage() override = default;

    void setInverse(bool inv) override { is_inverse_ = inv; }
    bool isInverse() const override    { return is_inverse_; }

    /// Pipeline-driven dims push (at addStage and again at finalize). Ignored once
    /// setDimsOverride() has pinned explicit dims — see that method for why.
    void setDims(const std::array<size_t, 3>& dims) override {
        if (!dims_pinned_) dims_ = dims;
    }
    void setDims(size_t x, size_t y = 1, size_t z = 1) {
        if (!dims_pinned_) dims_ = {x, y, z};
    }
    std::array<size_t, 3> getDims() const { return dims_; }
    bool hasDimsOverride() const { return dims_pinned_; }

    /**
     * Pin this stage's dimensions independently of the pipeline's.
     *
     * A branch may carry data whose shape is not the pipeline's input shape —
     * the binned background produced by ROIBinSplitStage is `ceil(nx/b) x
     * ceil(ny/b) x nz`, not `nx x ny x nz`. Without pinning, `Pipeline::finalize()`
     * re-pushes the global dims to every stage and silently overwrites anything
     * set at construction, so the predictor would decorrelate the binned image
     * using the full-resolution row stride and produce garbage residuals that
     * still round-trip (the inverse makes the same mistake) — wrong ratio, no error.
     * The pinned dims are serialized in this stage's own header, so the inverse
     * pass recovers them from the archive.
     */
    void setDimsOverride(size_t x, size_t y, size_t z) {
        dims_ = {x, y, z};
        dims_pinned_ = true;
    }

    /**
     * Set the tile shape. `tz == 1` ⇒ 2-D tile; `ty == tz == 1` ⇒ 1-D tile.
     * The product `tx*ty*tz` (= tile_elems, the natural AdaptiveBitpack block size)
     * must be in [1, 1024]. Each extent must be in [1, 255]. Pass 0 for any extent
     * to keep the ndim-derived default (2-D 8x8, 3-D 4x4x4, 1-D 64).
     */
    void setTileShape(uint32_t tx, uint32_t ty = 1, uint32_t tz = 1) {
        auto chk = [](uint32_t v, const char* nm) {
            if (v > 255)
                throw std::invalid_argument(
                    std::string("TiledLorenzoStage::setTileShape: ") + nm
                    + " must be in [0, 255], got " + std::to_string(v));
        };
        chk(tx, "tx"); chk(ty, "ty"); chk(tz, "tz");
        const uint32_t prod = (tx ? tx : 1) * (ty ? ty : 1) * (tz ? tz : 1);
        if (prod > 1024)
            throw std::invalid_argument(
                "TiledLorenzoStage::setTileShape: tx*ty*tz must be in [1, 1024], got "
                + std::to_string(prod));
        tile_ = {tx, ty, tz};
        tile_set_ = true;
    }
    std::array<uint32_t, 3> getTileShape() const { return effectiveTile(); }

    /// Elements per tile = the AdaptiveBitpack block_size that aligns blocks to tiles.
    uint32_t getTileElems() const {
        auto t = effectiveTile();
        return t[0] * t[1] * t[2];
    }

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

    std::string getName() const override { return "TiledLorenzo"; }
    size_t getNumInputs()  const override { return 1; }
    size_t getNumOutputs() const override { return 1; }

    // Block-local for fusion: forward, 2-D tile (tz == 1). The block a downstream
    // coder sees is one tile (tile_elems); the fused driver (cuSZp3) recomputes
    // the separable delta per element. tz > 1 is not fused yet.
    FusionSpec getFusionSpec() const override {
        auto t = effectiveTile();
        if (is_inverse_ || t[2] != 1) return {};
        return FusionSpec{FusionAccess::BlockLocal, t[0] * t[1]};
    }

    /// Warp-register predictor op (cuSZp3): 2-D separable tiled Lorenzo, EPL=2 (tile
    /// elems == 64). Declares the device policy type + its packed geometry (dims, tile,
    /// tiles-along-x) and the padded tile-major count `n_ab`, so the generic NVRTC warp
    /// runner composes the kernel with no per-predictor code. `inv2eb` is left 0 — the
    /// runner fills it from the resolved quantizer bound (see warp_op_params.h).
    FusedOpDecl getFusedOp() const override {
        const auto t = effectiveTile();
        if (is_inverse_ || t[2] != 1u || t[0] * t[1] != 64u) return {};
        const uint32_t dx = static_cast<uint32_t>(dims_[0]);
        const uint32_t dy = static_cast<uint32_t>(dims_[1]);
        const uint32_t tx = t[0], ty = t[1];
        const uint32_t ntx = (dx + tx - 1u) / tx;
        const uint32_t nty = (dy + ty - 1u) / ty;
        FusedOpDecl d;
        d.strategy       = FusionStrategy::WarpRegister;
        d.op_name        = "TiledLorenzo2DPredictor";
        d.include_header = "fused/fused_block/warp_fusion.cuh";
        d.elems_per_lane = 2;
        d.n_ab           = static_cast<size_t>(ntx) * nty * tx * ty;
        fused::warp::TiledLorenzo2DParams p{0.0f, dx, dy, tx, ty, ntx};
        d.params.resize(sizeof(p));
        std::memcpy(d.params.data(), &p, sizeof(p));
        return d;
    }

    std::vector<size_t> estimateOutputSizes(
        const std::vector<size_t>& input_sizes
    ) const override {
        // Forward: natural n -> padded tile-major (num_tiles * tile_elems).
        // Inverse: padded tile-major -> natural n.
        const size_t n = naturalElems(input_sizes);
        const size_t out_elems = is_inverse_ ? n : paddedElems();
        return {out_elems * sizeof(T)};
    }

    std::unordered_map<std::string, size_t>
    getActualOutputSizesByName() const override {
        return {{"output", actual_output_size_}};
    }
    size_t getActualOutputSize(int index) const override {
        return (index == 0) ? actual_output_size_ : 0;
    }

    uint16_t getStageTypeId() const override {
        return static_cast<uint16_t>(StageType::TILED_LORENZO);
    }

    uint8_t getOutputDataType(size_t /*output_index*/) const override {
        return static_cast<uint8_t>(getElementDataType());
    }
    uint8_t getInputDataType(size_t /*input_index*/) const override {
        return static_cast<uint8_t>(getElementDataType());
    }

    size_t serializeHeader(size_t /*output_index*/, uint8_t* buf, size_t max_size) const override {
        if (max_size < sizeof(TiledLorenzoConfig))
            throw std::runtime_error("TiledLorenzoStage: header buffer too small");
        auto t = effectiveTile();
        TiledLorenzoConfig cfg;
        cfg.data_type = getElementDataType();
        cfg.ndim      = static_cast<uint8_t>(ndim());
        cfg.tile_x    = static_cast<uint8_t>(t[0]);
        cfg.tile_y    = static_cast<uint8_t>(t[1]);
        cfg.tile_z    = static_cast<uint8_t>(t[2]);
        cfg.dim_x     = static_cast<uint32_t>(dims_[0]);
        cfg.dim_y     = static_cast<uint32_t>(dims_[1]);
        cfg.dim_z     = static_cast<uint32_t>(dims_[2]);
        std::memcpy(buf, &cfg, sizeof(cfg));
        return sizeof(cfg);
    }

    void deserializeHeader(const uint8_t* buf, size_t size) override {
        if (size < sizeof(TiledLorenzoConfig))
            throw std::runtime_error("TiledLorenzoStage: header too small");
        TiledLorenzoConfig cfg;
        std::memcpy(&cfg, buf, sizeof(cfg));
        int eff_ndim = (cfg.ndim == 0) ? 1 : static_cast<int>(cfg.ndim);
        dims_[0] = cfg.dim_x;
        dims_[1] = (eff_ndim >= 2) ? cfg.dim_y : 1;
        dims_[2] = (eff_ndim >= 3) ? cfg.dim_z : 1;
        tile_ = {cfg.tile_x, cfg.tile_y, cfg.tile_z};
        tile_set_ = (cfg.tile_x != 0);
    }

    size_t getMaxHeaderSize(size_t /*output_index*/) const override {
        return sizeof(TiledLorenzoConfig);
    }

private:
    bool is_inverse_           = false;
    bool dims_pinned_          = false;  ///< set by setDimsOverride(); blocks pipeline dim pushes
    size_t actual_output_size_ = 0;
    std::array<size_t, 3>   dims_     = {0, 1, 1};
    std::array<uint32_t, 3> tile_     = {8, 8, 1};
    bool                    tile_set_ = false;

    /// Effective tile shape: explicit if set, else ndim-derived default.
    std::array<uint32_t, 3> effectiveTile() const {
        if (tile_set_) {
            return {tile_[0] ? tile_[0] : 1u,
                    tile_[1] ? tile_[1] : 1u,
                    tile_[2] ? tile_[2] : 1u};
        }
        switch (ndim()) {
            case 3:  return {4, 4, 4};
            case 2:  return {8, 8, 1};
            default: return {64, 1, 1};
        }
    }

    /// Natural element count (dims if known, else inferred from input bytes).
    size_t naturalElems(const std::vector<size_t>& input_sizes) const {
        if (dims_[0] > 0) return dims_[0] * dims_[1] * dims_[2];
        return input_sizes.empty() ? 0 : input_sizes[0] / sizeof(T);
    }

    /// Padded tile-major element count = num_tiles * tile_elems.
    size_t paddedElems() const {
        auto t = effectiveTile();
        const size_t dx = (dims_[0] > 0) ? dims_[0] : 0;
        const size_t dy = dims_[1], dz = dims_[2];
        if (dx == 0) return 0;
        const size_t ntx = (dx + t[0] - 1) / t[0];
        const size_t nty = (dy + t[1] - 1) / t[1];
        const size_t ntz = (dz + t[2] - 1) / t[2];
        return ntx * nty * ntz * (size_t)t[0] * t[1] * t[2];
    }

    static DataType getElementDataType() {
        if (std::is_same<T, int16_t>::value) return DataType::INT16;
        return DataType::INT32;
    }
};

extern template class TiledLorenzoStage<int16_t>;
extern template class TiledLorenzoStage<int32_t>;

} // namespace fz
