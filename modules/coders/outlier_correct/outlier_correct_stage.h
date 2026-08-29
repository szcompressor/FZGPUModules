#pragma once

/**
 * @file outlier_correct_stage.h
 * @brief OutlierCorrectStage<Reconstructor> — transform-agnostic sparse
 *        outlier correction that turns a coefficient-domain quantization
 *        bound into an actually GUARANTEED reconstructed-domain pointwise
 *        error bound, for any reversible transform.
 *
 * ## The problem this fixes
 *
 * `Transform -> QuantizerStage(linear/ABS) -> Coder` quantizes transform
 * COEFFICIENTS directly. For transforms whose synthesis gain varies (CDF 9/7
 * is the motivating case: gain differs by decomposition level), a uniform
 * coefficient-domain threshold does NOT translate to a uniform bound on the
 * RECONSTRUCTED FIELD's pointwise error -- measured misses up to 2.7x the
 * requested bound on real CDF 9/7 data (see `memory/speck_gpu_design.md`
 * sec.9). A candidate fix -- scale each coefficient's quantization step by
 * its level's synthesis-filter gain -- was tried and REJECTED for CDF 9/7:
 * it makes the max error WORSE, because many coefficients across levels
 * jointly influence any given pixel, so bounding each one's isolated worst
 * case does not bound their sum. That reasoning is transform-shape-dependent
 * in general, so this stage does not assume a scaling fix exists for
 * whatever transform it's paired with either.
 *
 * The fix that actually works, matching what native SPERR's own
 * `Outlier_Coder` does for CDF 9/7 (and generalizes to any reversible
 * transform): quantize normally; separately compute what the reconstruction
 * WOULD be (dequantize + inverse-transform a copy); every pixel whose error
 * exceeds the bound gets an EXACT correction value in a sparse (index,
 * value) list, applied as the final step of decompress. This gives a
 * mathematically exact guarantee, not a calibrated approximation.
 *
 * ## Genericity: the `Reconstructor` policy
 *
 * Everything in this class -- diffing, sparse pack/apply, config,
 * serialization, port shape -- is transform-agnostic. The ONLY
 * transform-specific step is "given dequantized coefficients, produce the
 * trial reconstruction" -- that's `Reconstructor::applyInverseTransform()`.
 * A `Reconstructor` must provide:
 *
 * @code
 * struct MyReconstructor {
 *     static constexpr StageType kStageType = StageType::...;
 *     static std::string name() { return "..."; }
 *     // In-place: d_coeffs_inout holds dequantized coefficients on entry,
 *     // the trial reconstruction on return. n = nx*ny*max(nz,1) elements.
 *     // Defined in a .cu file (may call CUDA device kernels) -- keep it out
 *     // of this header, which stays host-safe and is included broadly
 *     // (fzgpumodules.h, .cpp translation units).
 *     static void applyInverseTransform(float* d_coeffs_inout, int nx, int ny, int nz,
 *                                        cudaStream_t stream);
 * };
 * @endcode
 *
 * `Cdf97Reconstructor` (`modules/coders/cdf97_outlier_correct/`) is the one
 * instantiation that ships today; adding another reversible transform's
 * bound-guarantee pipeline is writing one small policy struct like it, not a
 * new stage. This is the split found while pushing back on `TeeStage` +
 * `Cdf97OutlierCorrectStage` as too SPERR-specific to justify as Pipeline
 * primitives (see `memory/speck_gpu_design.md` sec.9's design-flaw
 * discussion) -- reusability was the actual bar, not a naming change.
 *
 * ## Port shape and why it needs the raw field bound directly (not via Tee)
 *
 * This stage needs BOTH the original raw field (to compute corrections
 * against, at compress time) AND the dequantized codes (to reconstruct a
 * trial value from, in both directions). The raw field is bound directly to
 * input port 0 via `Pipeline::bindExternalInput()` -- no duplicate-copy node
 * needed; see `Pipeline::bindExternalInput()`'s doc comment for why a
 * dedicated fan-out stage (`TeeStage`, now removed) turned out to be
 * unnecessary scaffolding once `buildInverseDAG()`'s actual mirroring rules
 * were traced precisely. Forward inputs: [raw_field, codes]. Forward
 * outputs: [correction_stream (archived leaf), codes_passthrough -> coder].
 * Per the DAG's bijective inverse contract (inverse output k reconstructs
 * forward input k), inverse outputs are: [corrected field, codes
 * passthrough]; inverse inputs are: [archived correction stream, coder's
 * decoded codes].
 *
 * The inverse-transform that recovers the trial/candidate reconstruction
 * runs identically in both directions (compress-time to detect outliers,
 * decompress-time to reconstruct before applying corrections) -- it is NOT
 * a forward/inverse pair in the Stage sense, which is why it's called
 * directly here via the `Reconstructor` policy rather than delegated to
 * another DAG-wired stage's own inverse: `buildInverseDAG()`'s mirroring
 * assumes a stage's inverse output reconstructs its OWN forward input, and
 * "run the same forward-direction computation again in both pipeline
 * phases" doesn't fit that shape for any stage, transform-specific or not.
 *
 * ## Scope
 *
 * `float` coefficients only. ABS-mode linear quantization only (the only
 * mode any current pipeline pairs this with); `error_bound` here MUST equal
 * the paired `QuantizerStage`'s own `error_bound` -- set both from the same
 * value when building the pipeline.
 *
 * ## File layout (mirrors Cdf97Stage's own .h/.cu split)
 *
 * This header declares the class only -- no CUDA device code, safe to
 * include from a plain .cpp translation unit (fzgpumodules.h does).
 * Member definitions (execute(), the pimpl Impl) live in
 * `outlier_correct_stage_impl.cuh`, included only from .cu files that
 * explicit-instantiate a concrete Reconstructor (see
 * `cdf97_outlier_correct_stage.cu`).
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

/// Serialized OutlierCorrectStage config. 12 bytes.
struct OutlierCorrectConfig {
    uint32_t dim_x;
    uint32_t dim_y;
    float    error_bound;

    OutlierCorrectConfig() : dim_x(0), dim_y(0), error_bound(1e-4f) {}
};
static_assert(sizeof(OutlierCorrectConfig) <= FZM_STAGE_CONFIG_SIZE,
              "OutlierCorrectConfig must fit in FZM_STAGE_CONFIG_SIZE");

template <typename Reconstructor>
class OutlierCorrectStage : public Stage {
public:
    OutlierCorrectStage() = default;
    ~OutlierCorrectStage() override;

    // ── Stage control ─────────────────────────────────────────────────────────
    void setInverse(bool inv) override { is_inverse_ = inv; }
    bool isInverse() const override    { return is_inverse_; }

    void setDims(const std::array<size_t, 3>& dims) override { dims_ = dims; }
    void setDims(size_t x, size_t y, size_t z = 1) { dims_ = {x, y, z}; }

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
    /// direction, matching the mid-pipeline sync Speck2DStage also accepts.
    bool isGraphCompatible() const override { return false; }

    // ── Metadata ──────────────────────────────────────────────────────────────
    std::string getName() const override { return Reconstructor::name() + "OutlierCorrect"; }
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
            size_t n = (dims_[0] && dims_[1])
                ? dims_[0] * dims_[1] * (dims_[2] > 0 ? dims_[2] : 1)
                : input_sizes[1] / sizeof(int32_t);
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
        return static_cast<uint16_t>(Reconstructor::kStageType);
    }
    uint8_t getOutputDataType(size_t output_index) const override {
        if (output_index == 1) return static_cast<uint8_t>(DataType::INT32);   // codes passthrough
        return is_inverse_ ? static_cast<uint8_t>(DataType::FLOAT32)           // corrected field
                           : static_cast<uint8_t>(DataType::UNKNOWN);          // correction stream (opaque)
    }
    // Position- and direction-aware (Pipeline::typeCheckConnections() checks
    // each connection against the input port it actually lands on).
    //   forward: input0 = raw field passthrough (float32), input1 = quant codes (int32)
    //   inverse: input0 = correction stream (opaque byte blob), input1 = quant codes (int32)
    uint8_t getInputDataType(size_t input_index) const override {
        if (input_index == 1) return static_cast<uint8_t>(DataType::INT32);    // quant codes
        return is_inverse_ ? static_cast<uint8_t>(DataType::UNKNOWN)           // correction stream (opaque)
                           : static_cast<uint8_t>(DataType::FLOAT32);          // raw field
    }

    // ── Serialization ─────────────────────────────────────────────────────────
    size_t serializeHeader(size_t /*output_index*/, uint8_t* buf, size_t max_size) const override {
        if (max_size < sizeof(OutlierCorrectConfig))
            throw std::runtime_error(getName() + ": header buffer too small");
        OutlierCorrectConfig cfg;
        cfg.dim_x = static_cast<uint32_t>(dims_[0]);
        cfg.dim_y = static_cast<uint32_t>(dims_[1]);
        cfg.error_bound = error_bound_;
        std::memcpy(buf, &cfg, sizeof(cfg));
        return sizeof(cfg);
    }
    void deserializeHeader(const uint8_t* buf, size_t size) override {
        if (size < sizeof(OutlierCorrectConfig))
            throw std::runtime_error(getName() + ": header too small");
        OutlierCorrectConfig cfg;
        std::memcpy(&cfg, buf, sizeof(cfg));
        dims_[0] = cfg.dim_x; dims_[1] = cfg.dim_y; dims_[2] = 1;
        error_bound_ = cfg.error_bound;
    }
    size_t getMaxHeaderSize(size_t) const override { return sizeof(OutlierCorrectConfig); }

    void saveState()    override { saved_dims_ = dims_; saved_eb_ = error_bound_; }
    void restoreState() override { dims_ = saved_dims_; error_bound_ = saved_eb_; }

private:
    bool   is_inverse_ = false;
    std::array<size_t, 3> dims_       = {0, 0, 1};
    std::array<size_t, 3> saved_dims_ = {0, 0, 1};
    float  error_bound_ = 1e-4f;
    float  saved_eb_    = 1e-4f;
    size_t actual_size0_ = 0, actual_size1_ = 0;

    struct Impl;   // defined in outlier_correct_stage_impl.cuh
    Impl* impl_ = nullptr;
};

} // namespace fz
