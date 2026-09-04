#pragma once

/**
 * @file adaptive_lorenzo_stage.h
 * @brief Per-tile adaptive multi-order Lorenzo predictor with centering. Lossless.
 */

#include "stage/stage.h"
#include "fzm_format.h"
#include "backend/types.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include "fused/common/data_type_of.h"

#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace fz {

/// Serialized config stored in FZMBufferEntry.stage_config.
struct AdaptiveLorenzoConfig {
    DataType data_type;        ///< Signed integer element type (1B).
    uint8_t  coder_block_size; ///< Downstream coder's block size (fixed at 32).
    uint8_t  blocks_per_tile;  ///< Coder blocks per adaptation tile.
    uint8_t  enable_order2;    ///< 1 if LZ2 is a candidate variant.
    uint8_t  enable_centering; ///< 1 if centering is a candidate variant.
    uint8_t  reserved[3];      ///< Must be zero.
    uint32_t num_elements;     ///< Element count (for tile-count recovery).

    AdaptiveLorenzoConfig()
        : data_type(DataType::INT32), coder_block_size(32), blocks_per_tile(8),
          enable_order2(1), enable_centering(1), reserved{0, 0, 0},
          num_elements(0) {}
};
static_assert(sizeof(AdaptiveLorenzoConfig) <= FZM_STAGE_CONFIG_SIZE,
              "AdaptiveLorenzoConfig must fit in FZM_STAGE_CONFIG_SIZE");

/**
 * Per-tile adaptive multi-order Lorenzo predictor (FSZ prediction stage). Lossless.
 *
 * Splits the flattened input into tiles of `coder_block_size * blocks_per_tile`
 * elements and, for each tile independently, picks whichever of four prediction
 * variants encodes smallest:
 *
 *   | variant | residual |
 *   |---|---|
 *   | LZ1              | `q_i - q_{i-1}` |
 *   | LZ2              | `q_i - 2q_{i-1} + q_{i-2}` |
 *   | LZ1 + centering  | LZ1 on `q_i - mu` |
 *   | LZ2 + centering  | LZ2 on `q_i - mu` |
 *
 * The prediction chain runs the length of the whole tile, not the coder block —
 * that is the cross-block prediction state of FSZ, obtained here by making the
 * tile a multiple of the coder's block. Only the tile's first one or two
 * elements lack predecessors instead of one per coder block.
 *
 * **Selection is by exact encoded size, not entropy.** For a coder block with
 * maximum residual magnitude of `r` bits, `AdaptiveBitpackStage` emits
 * `0` bytes when `r == 0` and `word_bytes * (r + 1)` otherwise (a sign bitmap
 * plus `r` bit-planes). The stage sums that over the tile's blocks for each
 * variant and takes the minimum, adding `sizeof(T)` for the mean when a centered
 * variant is in the running. This makes the selection exact **only for a
 * downstream `AdaptiveBitpackStage` with a matching `block_size`** — routing the
 * residuals into an entropy coder instead leaves the choice merely reasonable,
 * not optimal.
 *
 * **Single-pass evaluation.** All four variants are derived from one read of the
 * tile, because a constant offset cancels out of a k-th order difference for
 * every element with `k` predecessors (`delta^k(q - mu) == delta^k(q)`). The
 * centered variants therefore differ from the uncentered ones only in the tile's
 * first one (LZ1) or two (LZ2) residuals, so their costs come from adjusting the
 * first coder block's rate rather than recomputing anything.
 *
 * @note **Prior work:** the cross-block prediction state, the four-variant
 *       adaptive selection, and the finite-difference cancellation that makes a
 *       single-pass evaluation possible are the design of FSZ (Jiajun Huang,
 *       "FSZ: Breaking the Prediction-Throughput Trade-off in GPU Lossy
 *       Compression", SC'26, arXiv:2607.15413). This is an independent
 *       reimplementation as a modular DAG stage — FSZ fuses prediction,
 *       quantization and encoding into one kernel, whereas this stage is the
 *       prediction step alone and pairs with `QuantizerStage` upstream and
 *       `AdaptiveBitpackStage` downstream.
 *
 * Forward outputs:
 * - [0] `output` — residuals for the selected variant (`T`, one per element)
 * - [1] `modes`  — one byte per tile: bit 0 = order 2, bit 1 = centering
 * - [2] `means`  — one `T` per tile (meaningful only where bit 1 is set)
 *
 * @tparam T  Signed integer element type: int8_t, int16_t, int32_t, int64_t.
 */
template<typename T>
class AdaptiveLorenzoStage : public Stage {
    static_assert(std::is_integral<T>::value && std::is_signed<T>::value,
                  "AdaptiveLorenzoStage requires a signed integer type");
public:
    struct Config {
        /// Downstream coder block size. Fixed at 32: the cost model and the
        /// one-warp-per-block reduction both assume a 32-element block, which is
        /// also the only size at which `AdaptiveBitpackStage` packs each
        /// bit-plane into exactly one 32-bit word.
        uint32_t coder_block_size = 32;
        /// Coder blocks per adaptation tile. Tile length is
        /// `coder_block_size * blocks_per_tile`; longer tiles mean a longer
        /// prediction chain and cheaper per-tile metadata, but a coarser
        /// adaptation granularity. Must be in [1, 32] (tile <= 1024).
        uint32_t blocks_per_tile = 8;
        bool enable_order2    = true;   ///< Include the LZ2 variants.
        bool enable_centering = true;   ///< Include the centered variants.
        Config() = default;
    };

    AdaptiveLorenzoStage() { validate(); }
    explicit AdaptiveLorenzoStage(const Config& config) : config_(config) { validate(); }

    void setInverse(bool inv) override { is_inverse_ = inv; }
    bool isInverse() const override    { return is_inverse_; }

    uint32_t getTileSize() const {
        return config_.coder_block_size * config_.blocks_per_tile;
    }

    /// Bind an exact additive fixed-rate policy exposed by a directly connected
    /// AdaptiveBitpack stage. Both plain and element-0-outlier selection operate
    /// independently on the same 32-element coder units.
    bool bindDownstreamEncodingOracle(const EncodingOracleDecl& decl) override {
        if (!decl.valid() || !decl.additive ||
            (decl.kind != EncodingOracleKind::PlainFixedRateBitpack &&
             decl.kind != EncodingOracleKind::AdaptiveFixedRateBitpack) ||
            decl.input_data_type != static_cast<uint8_t>(getElementDataType()) ||
            decl.unit_elems != config_.coder_block_size) {
            return false;
        }
        bound_oracle_ = decl;
        has_bound_oracle_ = true;
        return true;
    }

    bool hasBoundEncodingOracle() const { return has_bound_oracle_; }
    EncodingOracleKind getBoundEncodingOracleKind() const {
        return has_bound_oracle_ ? bound_oracle_.kind
                                 : EncodingOracleKind::PlainFixedRateBitpack;
    }

    FusionSpec getFusionSpec() const override {
        if (is_inverse_ || !has_bound_oracle_) return {};
        return FusionSpec{FusionAccess::TileAdaptive, getTileSize(),
                          config_.coder_block_size};
    }

    std::vector<FusedAuxOutputDecl> getFusedAuxOutputs() const override {
        if (!getFusionSpec().fusable()) return {};
        return {
            FusedAuxOutputDecl{1, "modes", FusedAuxSizeKind::FixedBitsPerUnit,
                               static_cast<uint8_t>(DataType::UINT8), getTileSize(),
                               2u, 0u},
            FusedAuxOutputDecl{2, "means", FusedAuxSizeKind::CompactedElements,
                               static_cast<uint8_t>(getElementDataType()), getTileSize(),
                               0u, 1u},
        };
    }

    void execute(
        fz::stream_t stream,
        MemoryPool* pool,
        const std::vector<void*>& inputs,
        const std::vector<void*>& outputs,
        const std::vector<size_t>& sizes
    ) override;

    std::string getName() const override { return "AdaptiveLorenzo"; }
    size_t getNumInputs()  const override { return is_inverse_ ? 3 : 1; }
    size_t getNumOutputs() const override { return is_inverse_ ? 1 : 3; }

    std::vector<std::string> getOutputNames() const override {
        return {"output", "modes", "means"};
    }

    ~AdaptiveLorenzoStage() override { releaseScratch(); }

    /// Reads back the number of tiles that actually chose centering and trims
    /// the `means` port to it. Called once the stream is idle, so the D2H never
    /// stalls the pipeline mid-flight.
    void postStreamSync(fz::stream_t stream) override;

    /// The `means` length is data-dependent and resolved by a D2H in
    /// postStreamSync(), so the forward pass cannot be captured in a graph.
    bool isGraphCompatible() const override { return false; }

    /// Defined in the .cu — sizing the CUB scan temp needs a CUDA translation unit.
    size_t estimateScratchBytes(
        const std::vector<size_t>& input_sizes
    ) const override;

    std::vector<size_t> estimateOutputSizes(
        const std::vector<size_t>& input_sizes
    ) const override {
        if (input_sizes.empty()) return {0, 0, 0};
        if (is_inverse_) return {input_sizes[0]};
        const size_t n     = input_sizes[0] / sizeof(T);
        const size_t tiles = numTiles(n);
        // modes: 2 bits per tile. means: worst case one per tile — the real
        // length is trimmed in postStreamSync() once the scan total is known.
        return {input_sizes[0], (tiles + 3) / 4, tiles * sizeof(T)};
    }

    void saveState() override    { saved_output_sizes_ = actual_output_sizes_; }
    void restoreState() override {
        if (!saved_output_sizes_.empty()) actual_output_sizes_ = saved_output_sizes_;
    }

    std::unordered_map<std::string, size_t>
    getActualOutputSizesByName() const override {
        auto names = getOutputNames();
        std::unordered_map<std::string, size_t> r;
        for (size_t i = 0; i < names.size() && i < actual_output_sizes_.size(); ++i)
            r[names[i]] = actual_output_sizes_[i];
        return r;
    }

    size_t getActualOutputSize(int index) const override {
        return (index >= 0 && index < static_cast<int>(actual_output_sizes_.size()))
            ? actual_output_sizes_[index] : 0;
    }

    void setFusedSideOutput(int output_index, size_t bytes) override {
        if (actual_output_sizes_.size() < 3) actual_output_sizes_.resize(3, 0);
        if (output_index == 1 || output_index == 2)
            actual_output_sizes_[static_cast<size_t>(output_index)] = bytes;
    }

    uint16_t getStageTypeId() const override {
        return static_cast<uint16_t>(StageType::ADAPTIVE_LORENZO);
    }

    uint8_t getOutputDataType(size_t output_index) const override {
        // The mode map is a byte stream; the residuals and means are T.
        return static_cast<uint8_t>(output_index == 1 ? DataType::UINT8
                                                      : getElementDataType());
    }
    uint8_t getInputDataType(size_t input_index) const override {
        return static_cast<uint8_t>(input_index == 1 ? DataType::UINT8
                                                     : getElementDataType());
    }

    size_t serializeHeader(size_t /*output_index*/, uint8_t* buf, size_t max_size) const override {
        if (max_size < sizeof(AdaptiveLorenzoConfig))
            throw std::runtime_error("AdaptiveLorenzoStage: header buffer too small");
        AdaptiveLorenzoConfig cfg;
        cfg.data_type        = getElementDataType();
        cfg.coder_block_size = static_cast<uint8_t>(config_.coder_block_size);
        cfg.blocks_per_tile  = static_cast<uint8_t>(config_.blocks_per_tile);
        cfg.enable_order2    = config_.enable_order2 ? 1u : 0u;
        cfg.enable_centering = config_.enable_centering ? 1u : 0u;
        cfg.num_elements     = static_cast<uint32_t>(num_elements_);
        std::memcpy(buf, &cfg, sizeof(cfg));
        return sizeof(cfg);
    }

    void deserializeHeader(const uint8_t* buf, size_t size) override {
        if (size < sizeof(AdaptiveLorenzoConfig))
            throw std::runtime_error("AdaptiveLorenzoStage: header too small");
        AdaptiveLorenzoConfig cfg;
        std::memcpy(&cfg, buf, sizeof(cfg));
        config_.coder_block_size = cfg.coder_block_size;
        config_.blocks_per_tile  = cfg.blocks_per_tile;
        config_.enable_order2    = (cfg.enable_order2 != 0);
        config_.enable_centering = (cfg.enable_centering != 0);
        num_elements_            = cfg.num_elements;
        validate();
    }

    size_t getMaxHeaderSize(size_t /*output_index*/) const override {
        return sizeof(AdaptiveLorenzoConfig);
    }

private:
    Config config_;
    EncodingOracleDecl bound_oracle_;
    bool has_bound_oracle_ = false;
    bool   is_inverse_ = false;
    size_t num_elements_ = 0;
    std::vector<size_t> actual_output_sizes_{0, 0, 0};
    std::vector<size_t> saved_output_sizes_;

    // Persistent compaction scratch, grown when a larger input is seen. Held
    // across calls so postStreamSync() can read the scan total after the fact.
    uint8_t*    d_modes_dense_ = nullptr;   ///< 1 byte per tile, pre-packing
    T*          d_means_dense_ = nullptr;   ///< 1 mean per tile, pre-compaction
    uint32_t*   d_flags_       = nullptr;   ///< centering bit per tile, +1 slot
    uint32_t*   d_offsets_     = nullptr;   ///< exclusive scan of d_flags_
    size_t      scratch_tiles_ = 0;
    MemoryPool* scratch_pool_  = nullptr;
    size_t      pending_tiles_ = 0;         ///< tiles awaiting a postStreamSync trim

    size_t ensureScratch(size_t num_tiles, MemoryPool* pool, fz::stream_t stream);
    void   releaseScratch();

    size_t numTiles(size_t n) const {
        const size_t t = getTileSize();
        return (n + t - 1) / t;
    }

    void validate() const {
        if (config_.coder_block_size != 32)
            throw std::invalid_argument(
                "AdaptiveLorenzoStage: coder_block_size must be 32 (the cost model "
                "and the per-block warp reduction both assume a 32-element block)");
        if (config_.blocks_per_tile < 1 || config_.blocks_per_tile > 32)
            throw std::invalid_argument(
                "AdaptiveLorenzoStage: blocks_per_tile must be in [1, 32] so the "
                "tile fits one CUDA block, got "
                + std::to_string(config_.blocks_per_tile));
    }

    static DataType getElementDataType() { return fused::dataTypeOf<T>(); }
};

extern template class AdaptiveLorenzoStage<int16_t>;
extern template class AdaptiveLorenzoStage<int32_t>;

/// Forward: select the best variant per tile and emit its residuals.
template<typename T>
void launchAdaptiveLorenzoForward(
    const T* d_input, T* d_residuals, uint8_t* d_modes, T* d_means,
    uint32_t* d_flags, size_t n, uint32_t tile_size, bool enable_order2,
    bool enable_centering, EncodingOracleKind oracle_kind, fz::stream_t stream);

/// Inverse: replay each tile's recorded variant.
template<typename T>
void launchAdaptiveLorenzoInverse(
    const T* d_residuals, const uint8_t* d_modes, const T* d_means, T* d_output,
    size_t n, uint32_t tile_size, fz::stream_t stream);

} // namespace fz
