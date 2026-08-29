#pragma once

/**
 * @file cdf97_outlier_correct_stage.h
 * @brief Cdf97OutlierCorrectStage — sparse outlier correction that turns the
 *        GPU SPERR pipeline's reported quantization bound into an actually
 *        GUARANTEED reconstructed-domain pointwise error bound.
 *
 * ## The problem this fixes
 *
 * `Cdf97Stage -> QuantizerStage(linear/ABS) -> Speck2DStage` quantizes DWT
 * COEFFICIENTS directly. Because the CDF 9/7 synthesis filter's gain differs
 * by decomposition level, a uniform coefficient-domain threshold does NOT
 * translate to a uniform bound on the RECONSTRUCTED FIELD's pointwise error
 * — measured misses up to 2.7x the requested bound on real data (see
 * `memory/speck_gpu_design.md` sec.9). A candidate fix — scale each
 * coefficient's quantization step by its level's synthesis-filter gain — was
 * tried and REJECTED: it makes the max error WORSE, because many
 * coefficients across levels jointly influence any given pixel, so bounding
 * each one's isolated worst case does not bound their sum.
 *
 * This stage implements the fix that actually works, matching what native
 * SPERR's own `Outlier_Coder` does: quantize normally; separately compute
 * what the reconstruction WOULD be (dequantize + inverse-DWT a copy — cheap,
 * since `Speck2DStage` is lossless w.r.t. the codes, so this needs no actual
 * SPECK2D encode/decode to know it); every pixel whose error exceeds the
 * bound gets an EXACT correction value in a sparse (index, value) list,
 * applied as the final step of decompress. This gives a mathematically
 * exact guarantee, not a calibrated approximation. Validated standalone in
 * `examples/sperr_gpu_bounded.cu` before this DAG-integrated version;
 * measured cost is data-dependent (0.06%-8.2% of pixels on CLDHGH/CLDLOW
 * across bounds 1e-2..1e-5, markedly more on FLDSC at the tightest bound —
 * the guarantee itself is never violated, only its cost varies).
 *
 * ## Why this needs a Tee, and a 2-in/2-out (not 1-in/1-out) shape
 *
 * This stage needs BOTH the original raw field (to compute corrections
 * against, at compress time) AND the reconstructed field several stages
 * later (to apply corrections to, at decompress time) — a value from
 * upstream of the whole DWT/quantize/code chain AND a value computed by
 * that chain. `Pipeline::compress()` allows only one true "source" stage
 * (see `TeeStage`'s doc comment), and `buildInverseDAG()`'s wiring requires
 * `inverse_input_count == forward_output_count` / `inverse_output_count ==
 * forward_input_count` for EVERY stage (not just this one) — so getting a
 * genuine second, independently-sourced value into this stage's inverse
 * requires giving it a second FORWARD OUTPUT that some other stage consumes
 * and whose OWN inverse hands the value back. Concretely:
 *
 *   Tee1 (source) --out0--> Cdf97Stage -> QuantizerStage."codes"
 *        \--out1----------> CorrectStage.input[0]  (raw field)
 *
 *   QuantizerStage."codes" --> CorrectStage.input[1]  (codes)
 *
 *   CorrectStage.output[0] = correction stream   (archived leaf)
 *   CorrectStage.output[1] = codes passthrough --> Speck2DStage
 *
 * Forward inputs are [raw_field, codes]; per the contract, inverse OUTPUTS
 * reconstruct each: output[0] = the corrected final field (a genuine
 * reconstruction of "what raw_field was"), output[1] = the SAME codes
 * passed straight through (trivial — this stage never modifies them).
 * Forward outputs are [correction_stream, codes_passthrough]; per the
 * contract, inverse INPUTS come from whoever consumed each: input[0] = the
 * archived correction stream (codes_passthrough's consumer is Speck2DStage,
 * so at decompress input[1] = Speck2DStage-inverse's decoded codes).
 * `examples/sperr_gpu_bounded.cu`'s doc comment and
 * `memory/speck_gpu_design.md` sec.9's "DAG-integration follow-up" note walk
 * through the exact edge-by-edge trace that arrived at this shape — read
 * those before changing this stage's port structure, the wiring is not
 * arbitrary.
 *
 * The inverse-DWT that recovers the trial/candidate reconstruction runs
 * TWICE per decompress (once here internally, once in `Cdf97Stage`'s own
 * ordinary inverse step feeding the OTHER, discarded Tee1 branch) — an
 * accepted redundancy, since CDF 9/7 inverse is cheap relative to everything
 * else and avoiding it would need a structurally different (and much more
 * invasive) DAG topology.
 *
 * ## Scope
 *
 * `float` DWT coefficients only (matches the `sperr_gpu.toml` preset).
 * 2-D only. ABS-mode linear quantization only (the only mode this pipeline
 * uses); `error_bound` here MUST equal the `QuantizerStage`'s own
 * `error_bound` — set both from the same value when building the pipeline.
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

/// Serialized Cdf97OutlierCorrectStage config. 12 bytes.
struct Cdf97OutlierCorrectConfig {
    uint32_t dim_x;
    uint32_t dim_y;
    float    error_bound;

    Cdf97OutlierCorrectConfig() : dim_x(0), dim_y(0), error_bound(1e-4f) {}
};
static_assert(sizeof(Cdf97OutlierCorrectConfig) <= FZM_STAGE_CONFIG_SIZE,
              "Cdf97OutlierCorrectConfig must fit in FZM_STAGE_CONFIG_SIZE");

class Cdf97OutlierCorrectStage : public Stage {
public:
    Cdf97OutlierCorrectStage() = default;
    ~Cdf97OutlierCorrectStage() override;

    // ── Stage control ─────────────────────────────────────────────────────────
    void setInverse(bool inv) override { is_inverse_ = inv; }
    bool isInverse() const override    { return is_inverse_; }

    void setDims(const std::array<size_t, 3>& dims) override { dims_ = dims; }
    void setDims(size_t x, size_t y) { dims_ = {x, y, 1}; }

    /// MUST match the paired QuantizerStage's own error_bound (ABS mode).
    void setErrorBound(float b) { error_bound_ = b; }
    float getErrorBound() const { return error_bound_; }

    // ── Execution ─────────────────────────────────────────────────────────────
    void execute(cudaStream_t stream, MemoryPool* pool,
                 const std::vector<void*>& inputs,
                 const std::vector<void*>& outputs,
                 const std::vector<size_t>& sizes) override;

    /// Forward reads a device-side outlier count back to size its own output
    /// (data-dependent); completed here, not deferred to postStreamSync(), to
    /// keep this stage's port/host-sync shape simple -- one accepted sync per
    /// direction, matching the mid-pipeline sync `Speck2DStage` also accepts.
    bool isGraphCompatible() const override { return false; }

    // ── Metadata ──────────────────────────────────────────────────────────────
    std::string getName() const override { return "Cdf97OutlierCorrect"; }
    size_t getNumInputs()  const override { return 2; }
    size_t getNumOutputs() const override { return 2; }
    std::vector<std::string> getOutputNames() const override {
        return is_inverse_ ? std::vector<std::string>{"field", "codes"}
                           : std::vector<std::string>{"correction", "codes"};
    }

    std::vector<size_t> estimateOutputSizes(
        const std::vector<size_t>& input_sizes) const override {
        if (input_sizes.size() < 2) return {0, 0};
        if (is_inverse_) {
            size_t n = (dims_[0] && dims_[1]) ? dims_[0] * dims_[1] : input_sizes[1] / sizeof(int32_t);
            return { n * sizeof(float), n * sizeof(int32_t) };
        }
        // Worst case: every pixel is an outlier -- 4 (count) + 8 bytes/pixel
        // (index:u32 + value:f32) -- same proven-bound style as Speck2DStage.
        size_t n = input_sizes[0] / sizeof(float);
        return { 4 + n * 8, n * sizeof(int32_t) };
    }

    std::unordered_map<std::string, size_t>
    getActualOutputSizesByName() const override {
        auto names = getOutputNames();
        return {{names[0], actual_size0_}, {names[1], actual_size1_}};
    }
    size_t getActualOutputSize(int index) const override {
        if (index == 0) return actual_size0_;
        if (index == 1) return actual_size1_;
        return 0;
    }

    // ── Type system ───────────────────────────────────────────────────────────
    uint16_t getStageTypeId() const override {
        return static_cast<uint16_t>(StageType::CDF97_OUTLIER_CORRECT);
    }
    uint8_t getOutputDataType(size_t output_index) const override {
        if (output_index == 1) return static_cast<uint8_t>(DataType::INT32);   // codes passthrough
        return is_inverse_ ? static_cast<uint8_t>(DataType::FLOAT32)           // corrected field
                           : static_cast<uint8_t>(DataType::UNKNOWN);          // correction stream (opaque)
    }
    // Position- and direction-aware (Pipeline::typeCheckConnections() now checks
    // each connection against the input port it actually lands on -- see
    // Pipeline::typeCheckConnections() in compressor.cpp).
    //   forward: input0 = raw field passthrough (float32), input1 = quant codes (int32)
    //   inverse: input0 = correction stream (opaque byte blob), input1 = quant codes (int32)
    uint8_t getInputDataType(size_t input_index) const override {
        if (input_index == 1) return static_cast<uint8_t>(DataType::INT32);    // quant codes
        return is_inverse_ ? static_cast<uint8_t>(DataType::UNKNOWN)           // correction stream (opaque)
                           : static_cast<uint8_t>(DataType::FLOAT32);          // raw field
    }

    // ── Serialization ─────────────────────────────────────────────────────────
    size_t serializeHeader(size_t /*output_index*/, uint8_t* buf, size_t max_size) const override {
        if (max_size < sizeof(Cdf97OutlierCorrectConfig))
            throw std::runtime_error("Cdf97OutlierCorrectStage: header buffer too small");
        Cdf97OutlierCorrectConfig cfg;
        cfg.dim_x = static_cast<uint32_t>(dims_[0]);
        cfg.dim_y = static_cast<uint32_t>(dims_[1]);
        cfg.error_bound = error_bound_;
        std::memcpy(buf, &cfg, sizeof(cfg));
        return sizeof(cfg);
    }
    void deserializeHeader(const uint8_t* buf, size_t size) override {
        if (size < sizeof(Cdf97OutlierCorrectConfig))
            throw std::runtime_error("Cdf97OutlierCorrectStage: header too small");
        Cdf97OutlierCorrectConfig cfg;
        std::memcpy(&cfg, buf, sizeof(cfg));
        dims_[0] = cfg.dim_x; dims_[1] = cfg.dim_y; dims_[2] = 1;
        error_bound_ = cfg.error_bound;
    }
    size_t getMaxHeaderSize(size_t) const override { return sizeof(Cdf97OutlierCorrectConfig); }

    void saveState()    override { saved_dims_ = dims_; saved_eb_ = error_bound_; }
    void restoreState() override { dims_ = saved_dims_; error_bound_ = saved_eb_; }

private:
    bool   is_inverse_ = false;
    std::array<size_t, 3> dims_       = {0, 0, 1};
    std::array<size_t, 3> saved_dims_ = {0, 0, 1};
    float  error_bound_ = 1e-4f;
    float  saved_eb_    = 1e-4f;
    size_t actual_size0_ = 0, actual_size1_ = 0;

    struct Impl;
    Impl* impl_ = nullptr;
};

} // namespace fz
