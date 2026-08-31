#pragma once

/**
 * @file speck2d_stage.h
 * @brief GPU-parallel "wavefront" SPECK-like coder (2-D), see speck2d_kernels.cuh
 *        for the algorithm/format and the level-fusion optimization.
 *
 * A lossless, dimension-aware CODER: input is a 2-D field of signed 32-bit codes
 * (e.g. quantized DWT coefficients out of `Cdf97Stage` -> a quantizer in linear/
 * signed mode); output is a variable-length compressed bitstream, smaller in the
 * common case (worst case ~3x input -- see `estimateOutputSizes()`). Losslessly
 * invertible: `decompress(compress(x)) == x` exactly, for any int32 input,
 * including degenerate (all-zero) fields.
 *
 * ### Precision / scope
 *
 * `int32_t` codes only (v1). Extending to `int16_t`/`int64_t` is straightforward
 * (the kernels are not int32-specific beyond the sign/magnitude split and the
 * `31-__clz` msb call) but not yet done -- see speck2d_kernels.cuh.
 *
 * ### Not size-preserving
 *
 * Unlike `Cdf97Stage`, output size is DATA-DEPENDENT and only known after the
 * device kernels run. Like `RLEStage`, the actual size is read back
 * asynchronously during `execute()` and completed in `postStreamSync()` --
 * `getActualOutputSize()`/`getActualOutputSizesByName()` must not be called
 * before the stream passed to `execute()` has been synchronized.
 *
 * ### Current limitation
 *
 * 2-D only (`ndim` must resolve to 2 from `setDims()`); 3-D SPECK is future work
 * (see memory/speck_gpu_design.md P4).
 */

#include "stage/stage.h"
#include "fzm_format.h"
#include "backend/types.h"
#include <array>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace fz {

/// Serialized Speck2DStage config (FZMBufferEntry.stage_config). 24 bytes.
struct Speck2DConfig {
    uint32_t dim_x;      ///< X (fast) dimension.
    uint32_t dim_y;      ///< Y dimension.
    int32_t  B;          ///< Root onset (max msb over the whole field); -1 = all-zero.
    uint64_t nbits_a;    ///< Section A bit length (locates Section B in the payload).

    Speck2DConfig() : dim_x(0), dim_y(0), B(-1), nbits_a(0) {}
};
static_assert(sizeof(Speck2DConfig) <= FZM_STAGE_CONFIG_SIZE,
              "Speck2DConfig must fit in FZM_STAGE_CONFIG_SIZE");

class Speck2DStage : public Stage {
public:
    Speck2DStage() = default;
    ~Speck2DStage() override;

    // ── Stage control ─────────────────────────────────────────────────────────
    void setInverse(bool inv) override { is_inverse_ = inv; }
    bool isInverse() const override    { return is_inverse_; }

    void setDims(const std::array<size_t, 3>& dims) override { dims_ = dims; }
    void setDims(size_t x, size_t y) { dims_ = {x, y, 1}; }

    // ── Execution ─────────────────────────────────────────────────────────────
    void execute(cudaStream_t stream, MemoryPool* pool,
                 const std::vector<void*>& inputs,
                 const std::vector<void*>& outputs,
                 const std::vector<size_t>& sizes) override;

    /// Completes the async D2H readback of (B, nbitsA, nbitsB) started during a
    /// forward execute(). No-op (and unnecessary) on the inverse direction,
    /// whose output size is exact and known from the deserialized header.
    void postStreamSync(cudaStream_t stream) override;

    /// Forward path reads scalars back mid-pipeline (data-dependent output
    /// size); not CUDA-Graph-capturable, unlike Cdf97Stage.
    bool isGraphCompatible() const override { return false; }

    // ── Metadata ──────────────────────────────────────────────────────────────
    std::string getName() const override { return "SPECK2D"; }
    size_t getNumInputs()  const override { return 1; }
    size_t getNumOutputs() const override { return 1; }

    std::vector<size_t> estimateOutputSizes(
        const std::vector<size_t>& input_sizes) const override {
        if (input_sizes.empty()) return {0};
        if (is_inverse_) {
            // Exact: n int32 codes, from dims_ (nx*ny) -- NOT from input_sizes,
            // which for the inverse direction is the COMPRESSED payload size and
            // has no fixed relationship to the element count. dims_ is reliable
            // in both cases that matter: a cold object reconstructed via
            // deserializeHeader() (dims come from the header), and the SAME
            // object reused in-memory across a compress-then-decompress cycle
            // (dims_ was already set by setDims() at pipeline-build time and is
            // never cleared). Falls back to input_sizes only if dims_ somehow
            // isn't set yet, which estimateOutputSizes() can be called before
            // execute() validates -- not reachable in normal pipeline use.
            size_t n = (dims_[0] && dims_[1]) ? dims_[0] * dims_[1]
                                              : input_sizes[0] / sizeof(int32_t);
            return { n * sizeof(int32_t) };
        }
        // Worst case: nn (tree node count) <= 2n-1 (every internal node has
        // >=2 children, a general property of this quadtree's partition -- see
        // speck2d_kernels.cuh), nl == n exactly (every pixel is a leaf), and
        // every present node/leaf costs at most 1 word (32 bits) -- see the
        // words_ub comment in speck2d_stage.cu. So (2n + n) words is a safe
        // upper bound; +margin for rounding.
        size_t n = input_sizes[0] / sizeof(int32_t);
        size_t words_ub = 3 * n + 8;
        return { words_ub * sizeof(uint32_t) };
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
        return static_cast<uint16_t>(StageType::SPECK2D);
    }
    uint8_t getOutputDataType(size_t /*output_index*/) const override {
        // Compressed payload is opaque bytes; decompressed output is INT32.
        return is_inverse_ ? static_cast<uint8_t>(DataType::INT32)
                           : static_cast<uint8_t>(DataType::UNKNOWN);
    }
    uint8_t getInputDataType(size_t /*input_index*/) const override {
        return is_inverse_ ? static_cast<uint8_t>(DataType::UNKNOWN)
                           : static_cast<uint8_t>(DataType::INT32);
    }

    // ── Serialization ─────────────────────────────────────────────────────────
    size_t serializeHeader(size_t /*output_index*/, uint8_t* buf, size_t max_size) const override {
        if (max_size < sizeof(Speck2DConfig))
            throw std::runtime_error("Speck2DStage: header buffer too small");
        Speck2DConfig cfg;
        cfg.dim_x   = static_cast<uint32_t>(dims_[0]);
        cfg.dim_y   = static_cast<uint32_t>(dims_[1]);
        cfg.B       = last_B_;
        cfg.nbits_a = last_nbitsA_;
        std::memcpy(buf, &cfg, sizeof(cfg));
        return sizeof(cfg);
    }
    void deserializeHeader(const uint8_t* buf, size_t size) override {
        if (size < sizeof(Speck2DConfig))
            throw std::runtime_error("Speck2DStage: header too small");
        Speck2DConfig cfg;
        std::memcpy(&cfg, buf, sizeof(cfg));
        dims_[0] = cfg.dim_x; dims_[1] = cfg.dim_y; dims_[2] = 1;
        last_B_ = cfg.B;
        last_nbitsA_ = cfg.nbits_a;
    }
    size_t getMaxHeaderSize(size_t /*output_index*/) const override {
        return sizeof(Speck2DConfig);
    }

    void saveState()    override { saved_dims_ = dims_; }
    void restoreState() override { dims_ = saved_dims_; }

private:
    bool   is_inverse_         = false;
    size_t actual_output_size_ = 0;
    std::array<size_t, 3> dims_       = {0, 0, 1};
    std::array<size_t, 3> saved_dims_ = {0, 0, 1};
    int32_t  last_B_      = -1;
    uint64_t last_nbitsA_ = 0;
    uint64_t last_nbitsB_ = 0;

    // Per-shape resident state (tree geometry + device buffers); opaque here,
    // defined in speck2d_stage.cu to keep CUDA types out of this public header.
    struct Impl;
    Impl* impl_ = nullptr;

    // Pending async scalar reads from the most recent forward execute(); valid
    // only after postStreamSync() completes them.
    bool pending_ = false;
};

} // namespace fz
