#pragma once

/**
 * @file log_transform_stage.h
 * @brief Log-space transform for point-wise relative error bounds.
 *
 * Implements the transformation scheme of X. Liang, S. Di, D. Tao, Z. Chen and
 * F. Cappello, "An efficient transformation scheme for lossy data compression
 * with point-wise relative error bound", IEEE CLUSTER 2018, pp. 179-189.
 */

#include "stage/stage.h"
#include "fzm_format.h"
#include "backend/types.h"
#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace fz {

/**
 * Serialized LogTransformStage configuration (FZMBufferEntry.stage_config).
 * 20 bytes; fits easily in the 128-byte stage_config limit.
 */
struct LogTransformConfig {
    float    error_bound;    ///< delta — the target point-wise relative bound.
    float    threshold;      ///< |x| < threshold => lossless outlier. 0 = only specials.
    uint32_t num_elements;   ///< Total element count.
    uint32_t outlier_count;  ///< Actual number of outliers written.
    DataType input_type;     ///< Original input type (1B).
    uint8_t  reserved[3];    ///< Must be zero.

    LogTransformConfig()
        : error_bound(1e-3f), threshold(0.0f), num_elements(0), outlier_count(0),
          input_type(DataType::FLOAT32), reserved{0, 0, 0} {}
};
static_assert(sizeof(LogTransformConfig) == 20,
              "LogTransformConfig must be 20 bytes");

/**
 * Log-space transform: turns a point-wise **relative** bound into a plain
 * **absolute** bound, so an ordinary ABS quantizer downstream delivers the
 * relative guarantee.
 *
 * @note **Prior work:** the transformation scheme, the IEEE-754 exponent
 *       shortcut for `log2`, and the near-zero threshold follow Liang et al.,
 *       IEEE CLUSTER 2018. See `THIRD_PARTY.md`.
 *
 * ### The identity
 *
 * ```
 *   |x - x_hat| / |x| <= delta
 *     <=>  x_hat / x         in [1-delta, 1+delta]
 *     <=>  log2|x_hat| - log2|x|  in [log2(1-delta), log2(1+delta)]
 * ```
 *
 * A multiplicative bound on `x` is an additive bound on `log2|x|`. The interval
 * is asymmetric and `|log2(1-delta)| > log2(1+delta)`, so the binding side is
 * the upper one and the absolute bound to use downstream is
 *
 * ```
 *   e = log2(1 + delta) - kLogRoundTripSlack     // quantizerErrorBound()
 * ```
 *
 * (the slack pays for the transform's own float32 rounding — see below)
 *
 * Symmetric bins of half-width `e` leave the negative side tighter than
 * required, wasting roughly `delta^2 / 2` of achievable relative error.
 * Negligible for small delta.
 *
 * ### Why this stage exists
 *
 * `QuantizerStage` with `ErrorBoundMode::REL` already does log-space
 * quantization and already gives an exact per-element bound — but it quantizes
 * **raw values**, with no predictor in front, so its codes still carry the
 * field's full spatial redundancy and barely compress. `LorenzoQuantStage` and
 * `GInterpStage` compress well but cannot honour a per-element relative bound
 * at all (see `ErrorBoundMode::PREL`).
 *
 * Putting the log transform *upstream of the predictor* is what gets both:
 *
 * ```
 *   LogTransformStage -> LorenzoStage -> QuantizerStage(ABS, eb=log2(1+delta))
 *                     \-> signs ------------------------------------------\
 *                     \-> outlier_vals/outlier_idxs ----------------------> MergeStage
 * ```
 *
 * ### Downstream error bound is the caller's job
 *
 * The quantizer's `eb` is a function of *this* stage's `eb`, and a stage cannot
 * reach across the DAG to set it. Do it explicitly:
 *
 * ```cpp
 * auto* lg = p.addStage<LogTransformStage<float>>();
 * lg->setErrorBound(1e-3f);
 * ...
 * quant->setErrorBound(lg->quantizerErrorBound());   // ~log2(1 + 1e-3)
 * quant->setErrorBoundMode(ErrorBoundMode::ABS);
 * ```
 *
 * Passing the raw delta to the quantizer instead produces a *far* looser
 * relative bound (by a factor of `1/log2(1+delta)`, ~693x at delta=1e-3) with
 * no error and no warning. There is no cross-stage check for this — see the
 * "Limitations" section of `docs/stages/log_transform.md`.
 *
 * ### Preconditions worth knowing before you reach for this
 *
 * - **Sign changes hurt.** The sign is stripped into a separate bit-plane, so a
 *   sign flip between neighbours is invisible to the downstream predictor and
 *   costs a raw bit per element. Single-signed fields (density, pressure,
 *   magnitude) are the good case; fields oscillating about zero are not.
 * - **Near-zero values are outliers.** `log2|x| -> -inf`, so zeros, denormals,
 *   inf/NaN and anything below `threshold` are stored losslessly. Fields with a
 *   lot of near-zero mass pay for it in the outlier list.
 * - **float32 log/exp2 round-trip costs ~1 ULP.** `quantizerErrorBound()`
 *   already subtracts `kLogRoundTripSlack` to pay for it, and the forward pass
 *   escalates any element whose actual round-trip is worse than that. Below
 *   `minimumErrorBound()` (~1.4e-6) the slack would consume the whole budget
 *   and `execute()` throws, which is the honest answer: float32 log space
 *   cannot deliver that bound.
 *
 * ### Ports
 *
 * Forward (4 outputs):
 * ```
 *   [0] output        - TInput[n]              log2(|x|), or log_floor at outliers
 *   [1] signs         - uint8[ceil(n/8)]       bit i set => element i is negative
 *   [2] outlier_vals  - TInput[k]              original values at outlier positions
 *   [3] outlier_idxs  - uint32[k]              indices of outlier positions
 * ```
 *
 * Inverse: those same 4 buffers -> `TInput[n]`.
 *
 * The outlier *count* is not a port — it lives in a stage-private 4-byte device
 * scratch (`pool->allocatePersistentDevice` in `onFinalize()`), is D2H'd in
 * `postStreamSync()`, and is serialized into the FZM stage header. The inverse
 * path receives it as a kernel argument read from the deserialized header. This
 * mirrors `QuantizerStage` exactly.
 *
 * @tparam TInput Floating-point input type. Only `float` is instantiated;
 *         float64 support is deliberately deferred (see the stage docs).
 */
template<typename TInput = float>
class LogTransformStage : public Stage {
    static_assert(std::is_floating_point<TInput>::value,
                  "LogTransformStage: TInput must be a floating-point type.");

public:
    /**
     * Slack reserved for the transform's own float32 round-trip error, in log2
     * units.
     *
     * The total log-space deviation is the quantizer's bound *plus* whatever
     * `exp2(log2(x))` loses to rounding. Handing the quantizer the full
     * `log2(1+delta)` spends the entire budget before that second term exists,
     * which puts a handful of elements marginally over the bound. So the
     * quantizer gets `log2(1+delta) - kLogRoundTripSlack` and the transform
     * keeps the remainder.
     *
     * float32 log2/exp2 round-trips within ~1 ULP, i.e. ~1.2e-7 relative in
     * value space, which is ~1.7e-7 in log2 units. 1e-6 is a comfortable
     * several-x margin over that while costing <0.1% of the bound at
     * delta = 1e-3 — an unmeasurable compression-ratio difference.
     */
    static constexpr float kLogRoundTripSlack = 1e-6f;

    /// Minimum outlier slots reserved regardless of `outlier_capacity`, so a
    /// small input cannot round its reserve down to zero. See `maxOutlierCount`.
    static constexpr size_t kMinOutlierSlots = 8;

    /** Construction parameters. */
    struct Config {
        /// delta — the target point-wise relative error bound.
        float error_bound = 1e-3f;
        /// `|x| < threshold` => lossless outlier. 0 disables the threshold, so
        /// only zeros/denormals/inf/NaN are escalated. Raising it trades a
        /// larger outlier list for a narrower (more compressible) log range.
        float threshold = 0.0f;
        /// Fraction of the input element count reserved for outliers.
        float outlier_capacity = 0.05f;
        Config() = default;
        explicit Config(float eb, float thr = 0.0f, float cap = 0.05f)
            : error_bound(eb), threshold(thr), outlier_capacity(cap) {}
    };

    LogTransformStage() = default;
    explicit LogTransformStage(const Config& config) : config_(config) {}
    ~LogTransformStage() override;

    // ── Execution ────────────────────────────────────────────────────────────
    void execute(cudaStream_t stream, MemoryPool* pool,
                 const std::vector<void*>& inputs,
                 const std::vector<void*>& outputs,
                 const std::vector<size_t>& sizes) override;

    void postStreamSync(fz::stream_t stream) override;
    void onFinalize(size_t estimated_inlen, MemoryPool* pool) override;

    size_t estimateDeviceFootprintBytes(size_t /*estimated_inlen*/) const override {
        return sizeof(uint32_t);
    }

    /// No mid-execute sync: the outlier-count D2H happens in `postStreamSync()`.
    bool isGraphCompatible() const override { return true; }

    // ── Metadata ─────────────────────────────────────────────────────────────
    std::string getName() const override { return "LogTransform"; }

    size_t getNumInputs()  const override { return is_inverse_ ? 4 : 1; }
    size_t getNumOutputs() const override { return is_inverse_ ? 1 : 4; }

    std::vector<std::string> getOutputNames() const override {
        if (is_inverse_) return {"reconstructed"};
        return {"output", "signs", "outlier_vals", "outlier_idxs"};
    }

    std::vector<size_t> estimateOutputSizes(
        const std::vector<size_t>& input_sizes) const override;

    std::unordered_map<std::string, size_t>
    getActualOutputSizesByName() const override {
        auto names = getOutputNames();
        std::unordered_map<std::string, size_t> result;
        for (size_t i = 0; i < names.size() && i < actual_output_sizes_.size(); ++i)
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
        return static_cast<uint16_t>(StageType::LOG_TRANSFORM);
    }

    uint8_t getOutputDataType(size_t output_index) const override {
        if (is_inverse_) return static_cast<uint8_t>(inputDataType());
        switch (output_index) {
            case 0:  return static_cast<uint8_t>(inputDataType());   // log values
            case 1:  return static_cast<uint8_t>(DataType::UINT8);   // sign bitmap
            case 2:  return static_cast<uint8_t>(inputDataType());   // outlier vals
            case 3:  return static_cast<uint8_t>(DataType::UINT32);  // outlier idxs
            default: return static_cast<uint8_t>(DataType::UINT8);
        }
    }
    /// Index-aware: the inverse path consumes the same four heterogeneous
    /// buffers the forward path produced, so reporting FLOAT32 for all of them
    /// mis-describes the sign bitmap and the outlier indices to the DAG's
    /// type check and buffer sizing.
    uint8_t getInputDataType(size_t input_index) const override {
        if (!is_inverse_) return static_cast<uint8_t>(inputDataType());
        switch (input_index) {
            case 0:  return static_cast<uint8_t>(inputDataType());   // log values
            case 1:  return static_cast<uint8_t>(DataType::UINT8);   // sign bitmap
            case 2:  return static_cast<uint8_t>(inputDataType());   // outlier vals
            case 3:  return static_cast<uint8_t>(DataType::UINT32);  // outlier idxs
            default: return static_cast<uint8_t>(DataType::UINT8);
        }
    }

    // ── Serialization ────────────────────────────────────────────────────────
    size_t serializeHeader(size_t output_index, uint8_t* buf, size_t max_size) const override;
    size_t getMaxHeaderSize(size_t) const override { return sizeof(LogTransformConfig); }
    void   deserializeHeader(const uint8_t* buf, size_t size) override;

    void saveState() override {
        saved_config_               = config_;
        saved_num_elements_         = num_elements_;
        saved_actual_outlier_count_ = actual_outlier_count_;
        saved_actual_output_sizes_  = actual_output_sizes_;
    }
    void restoreState() override {
        config_               = saved_config_;
        num_elements_         = saved_num_elements_;
        actual_outlier_count_ = saved_actual_outlier_count_;
        actual_output_sizes_  = saved_actual_output_sizes_;
    }

    // ── Configuration ────────────────────────────────────────────────────────
    /// Set delta, the target point-wise relative error bound.
    void setErrorBound(float delta)      { config_.error_bound = delta; }
    /// `|x| < threshold` => lossless outlier; 0 disables (specials only).
    void setThreshold(float t)           { config_.threshold = t; }
    void setOutlierCapacity(float c)     { config_.outlier_capacity = c; }

    float getErrorBound()      const { return config_.error_bound; }
    float getThreshold()       const { return config_.threshold; }
    float getOutlierCapacity() const { return config_.outlier_capacity; }
    uint32_t getOutlierCount() const { return actual_outlier_count_; }

    /**
     * The **absolute** error bound the downstream quantizer must use:
     * `log2(1 + delta) - kLogRoundTripSlack`.
     *
     * This is the whole point of the stage, and wiring it up is the caller's
     * responsibility — see the class doc.
     */
    float quantizerErrorBound() const {
        return quantizerErrorBoundFor(config_.error_bound);
    }

    /// `quantizerErrorBound()` as a free-standing calculation, for callers that
    /// want the number before building the stage.
    static float quantizerErrorBoundFor(float delta) {
        return std::log2(1.0f + delta) - kLogRoundTripSlack;
    }

    /**
     * Smallest delta this stage can honour in float32.
     *
     * Below this the round-trip slack swallows the whole budget and there is
     * nothing left for the quantizer — the honest answer is that float32 log
     * space cannot deliver the bound, so `execute()` throws rather than
     * quietly returning a stream that violates it.
     */
    static float minimumErrorBound() {
        return std::exp2(2.0f * kLogRoundTripSlack) - 1.0f;
    }

private:
    Config config_;
    Config saved_config_;
    bool   is_inverse_ = false;

    std::vector<size_t> actual_output_sizes_{0, 0, 0, 0};
    std::vector<size_t> saved_actual_output_sizes_{0, 0, 0, 0};

    size_t   num_elements_               = 0;
    size_t   saved_num_elements_         = 0;
    uint32_t actual_outlier_count_       = 0;
    uint32_t saved_actual_outlier_count_ = 0;

    /// Stage-private 4-byte device scratch holding the live outlier count.
    /// Same lifecycle as `QuantizerStage::d_outlier_count_scratch_`.
    uint32_t*   d_outlier_count_scratch_ = nullptr;
    MemoryPool* persistent_pool_         = nullptr;
    /// Expires if the pool is destroyed before this stage. `persistent_pool_` is a
    /// raw borrow used in the destructor, and only Pipeline's declaration order
    /// (mem_pool_ before stages_) makes that safe — a stage built against a
    /// caller-owned pool has no such guarantee. See MemoryPool::lifetimeToken().
    std::weak_ptr<const void> persistent_pool_alive_;


    void initOutlierCountScratch(MemoryPool* pool);

    /**
     * Outlier slots to reserve for `n` elements.
     *
     * Floored at `kMinOutlierSlots` (unless `n` itself is smaller). A dropped
     * outlier here is *not* a bounded-error event the way it is in a quantizer
     * — the element's value is simply gone and the reconstruction is wrong at
     * that position. Truncating `n * capacity` toward zero would silently do
     * exactly that for any input smaller than `1 / capacity` elements, so the
     * floor is cheap insurance against a class of bug that produces garbage
     * rather than a loose bound.
     */
    size_t maxOutlierCount(size_t n) const {
        const size_t scaled =
            static_cast<size_t>(static_cast<double>(n) * config_.outlier_capacity);
        const size_t floored = (scaled < kMinOutlierSlots) ? kMinOutlierSlots : scaled;
        return (floored > n) ? n : floored;
    }

    static size_t signBytes(size_t n) { return (n + 7) / 8; }

    static DataType inputDataType() {
        return std::is_same<TInput, double>::value ? DataType::FLOAT64
                                                   : DataType::FLOAT32;
    }
};

extern template class LogTransformStage<float>;

} // namespace fz
