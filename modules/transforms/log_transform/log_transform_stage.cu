// Independent implementation of the scheme described by X. Liang, S. Di,
// D. Tao, Z. Chen and
// F. Cappello, "An efficient transformation scheme for lossy data compression
// with point-wise relative error bound", IEEE CLUSTER 2018. No source was copied;
// see docs/acknowledgements.md.
#include "transforms/log_transform/log_transform_stage.h"
#include "stage/stage_registry.h"
#include <cstring>
#include <algorithm>
#include <stdexcept>
#include <string>
#include "predictors/predictor_utils.cuh"   // scatter_assign_kernel
#include "backend/api.h"
#include "mem/mempool.h"
#include "cuda_check.h"
#include "log.h"

#include <cmath>
#include <limits>
#include <algorithm>
#include <cstdio>
#include <stdexcept>

namespace fz {

// =============================================================================
// Forward kernels
// =============================================================================

/**
 * Forward transform: x -> log2(|x|), with the sign split off and non-loggable
 * values escalated to lossless outliers.
 *
 * An element is escalated when any of the following holds:
 *   - it is zero, denormal, infinite or NaN (no usable log2)
 *   - `|x| < threshold` (the near-zero regularization from the paper — without
 *     it, values approaching zero drag log2|x| toward -inf and blow up the
 *     range the downstream quantizer has to cover)
 *   - the transform's own float32 round-trip error exceeds the slack reserved
 *     for it. exp2(log2(x)) is not exact in float32, and the quantizer's bound
 *     has already been reduced by `kLogRoundTripSlack` to pay for it — an
 *     element whose round-trip is worse than budgeted would blow the composed
 *     bound, so it is stored losslessly instead.
 *
 * Escalated elements still write `log_floor` to the value stream rather than a
 * sentinel. The decode path overwrites them via scatter, so the value is
 * irrelevant to correctness — but writing a value adjacent to the low end of
 * the real log range keeps the stream smooth for the downstream predictor,
 * whereas a 0 (i.e. |x| = 1) would be a spike in the middle of it.
 */
template<typename TInput>
__global__ void logTransformFwdKernel(
    const TInput* __restrict__ in,
    size_t n,
    float threshold,          // |x| < threshold -> outlier (0 disables)
    float rel_tolerance,      // kLogRoundTripSlack, in value-space relative terms
    float log_floor,          // value written at escalated positions
    TInput* __restrict__      out,
    TInput* __restrict__      outlier_vals,
    uint32_t* __restrict__    outlier_idxs,
    uint32_t* __restrict__    outlier_count,
    size_t max_outliers)
{
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const TInput xin = in[i];
    const float  x   = static_cast<float>(xin);

    // Classify via the raw exponent field: expo == 0 covers zero and denormals,
    // expo == 255 covers inf and NaN. Both are outside the log domain.
    uint32_t bits;
    __builtin_memcpy(&bits, &x, sizeof(float));
    const int expo = static_cast<int>((bits >> 23) & 0xFFu);

    const float ax = fabsf(x);

    bool escalate = (expo == 0 || expo == 255) || (threshold > 0.0f && ax < threshold);

    float y = log_floor;
    if (!escalate) {
        y = __log2f(ax);
        // Verify the transform's own round-trip before committing to it.
        const float back = exp2f(y);
        if (fabsf(back - ax) > rel_tolerance * ax) escalate = true;
    }

    out[i] = static_cast<TInput>(escalate ? log_floor : y);

    if (escalate) {
        const uint32_t slot = atomicAdd(outlier_count, 1u);
        if (slot < static_cast<uint32_t>(max_outliers)) {
            outlier_vals[slot] = xin;
            outlier_idxs[slot] = static_cast<uint32_t>(i);
        }
    }
}

/**
 * Pack one sign bit per element into a bitmap, one thread per output *byte*.
 *
 * Pattern A from `docs/how_to_add_a_stage.md`: each thread exclusively owns its
 * output byte and reads the 8 elements that map to it, so no atomics and no
 * read-modify-write races on shared bytes.
 *
 * Bit `j` of byte `b` corresponds to element `8*b + j`, set when that element is
 * negative. The sign of zero is not preserved here — zeros are outliers and are
 * restored bit-exactly by the scatter, negative zero included.
 */
template<typename TInput>
__global__ void logTransformSignPackKernel(
    const TInput* __restrict__ in,
    size_t n,
    uint8_t* __restrict__ signs,
    size_t n_bytes)
{
    const size_t b = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (b >= n_bytes) return;

    uint8_t packed = 0;
    const size_t base = b * 8;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        const size_t idx = base + j;
        if (idx < n && static_cast<float>(in[idx]) < 0.0f) {
            packed |= static_cast<uint8_t>(1u << j);
        }
    }
    signs[b] = packed;
}

// =============================================================================
// Inverse kernel
// =============================================================================

/**
 * Inverse transform: `x = sign * 2^y`.
 *
 * Outlier positions are left as whatever this produces and are then overwritten
 * by `scatter_assign_kernel`, so no sentinel handling is needed here.
 */
template<typename TInput>
__global__ void logTransformInvKernel(
    const TInput* __restrict__  y_in,
    const uint8_t* __restrict__ signs,
    size_t n,
    TInput* __restrict__ out)
{
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const float mag = exp2f(static_cast<float>(y_in[i]));
    const bool  neg = (signs[i >> 3] >> (i & 7)) & 1u;
    out[i] = static_cast<TInput>(neg ? -mag : mag);
}

// =============================================================================
// Stage implementation
// =============================================================================

template<typename TInput>
LogTransformStage<TInput>::~LogTransformStage()
{
    // Skip if the pool died first: `persistent_pool_` is a raw borrow, and only
    // Pipeline's declaration order (mem_pool_ before stages_) makes using it here
    // safe. A destroyed pool has already released everything it owned, so this is
    // correct rather than a leak. See MemoryPool::lifetimeToken().
    if (persistent_pool_alive_.expired()) { persistent_pool_ = nullptr; return; }
    if (persistent_pool_ != nullptr && d_outlier_count_scratch_ != nullptr) {
        persistent_pool_->freePersistentDevice(d_outlier_count_scratch_);
    }
    d_outlier_count_scratch_ = nullptr;
    persistent_pool_         = nullptr;
}

template<typename TInput>
void LogTransformStage<TInput>::initOutlierCountScratch(MemoryPool* pool)
{
    if (d_outlier_count_scratch_ != nullptr) return;
    if (pool == nullptr) {
        throw std::runtime_error(
            "LogTransformStage: outlier-count scratch requires a MemoryPool");
    }
    persistent_pool_ = pool;
    persistent_pool_alive_ = pool->lifetimeToken();
    d_outlier_count_scratch_ = static_cast<uint32_t*>(
        pool->allocatePersistentDevice(sizeof(uint32_t), "log_transform_outlier_count"));
}

template<typename TInput>
void LogTransformStage<TInput>::onFinalize(size_t /*estimated_inlen*/, MemoryPool* pool)
{
    initOutlierCountScratch(pool);
}

template<typename TInput>
std::vector<size_t> LogTransformStage<TInput>::estimateOutputSizes(
    const std::vector<size_t>& input_sizes) const
{
    if (is_inverse_) {
        // input[0] is the log-value stream; it is one element per output element.
        const size_t n = input_sizes.empty() ? 0 : input_sizes[0] / sizeof(TInput);
        return {n * sizeof(TInput)};
    }
    const size_t n   = input_sizes.empty() ? 0 : input_sizes[0] / sizeof(TInput);
    const size_t max_out = maxOutlierCount(n);
    return {
        n * sizeof(TInput),          // output (log values)
        signBytes(n),                // signs
        max_out * sizeof(TInput),    // outlier_vals
        max_out * sizeof(uint32_t)   // outlier_idxs
    };
}

template<typename TInput>
void LogTransformStage<TInput>::execute(
    cudaStream_t stream,
    MemoryPool* pool,
    const std::vector<void*>& inputs,
    const std::vector<void*>& outputs,
    const std::vector<size_t>& sizes)
{
    if (inputs.empty() || outputs.empty() || sizes.empty()) {
        throw std::runtime_error(
            "LogTransformStage: inputs, outputs, and sizes must be non-empty");
    }
    if (config_.error_bound <= 0.0f) {
        throw std::runtime_error(
            "LogTransformStage: error_bound (delta) must be > 0");
    }
    if (!is_inverse_ && config_.error_bound < minimumErrorBound()) {
        // Below this the round-trip slack consumes the entire budget, leaving
        // the quantizer a non-positive bound. Refusing is the honest answer:
        // float32 log space cannot represent the requested relative tolerance.
        throw std::runtime_error(
            "LogTransformStage: error_bound " +
            std::to_string(config_.error_bound) +
            " is below the float32 floor of " +
            std::to_string(minimumErrorBound()) +
            " — log-space round-trip error alone would exceed it. Use "
            "QuantizerStage REL, or a looser bound.");
    }

    constexpr int kBlock = 256;

    if (!is_inverse_) {
        // ── Forward ──────────────────────────────────────────────────────────
        const size_t n = sizes[0] / sizeof(TInput);
        num_elements_  = n;

        actual_output_sizes_.assign(4, 0);
        if (n == 0) {
            actual_outlier_count_ = 0;
            return;
        }
        if (outputs.size() < 4) {
            throw std::runtime_error(
                "LogTransformStage: forward pass requires 4 output buffers");
        }

        initOutlierCountScratch(pool);
        FZ_CUDA_CHECK(cudaMemsetAsync(d_outlier_count_scratch_, 0,
                                      sizeof(uint32_t), stream));

        const size_t max_out = maxOutlierCount(n);

        // Escalated elements are parked at the bottom of the log range so they
        // do not spike the stream the downstream predictor sees. With no
        // threshold set there is no meaningful floor, so use the smallest
        // normal float's exponent.
        const float log_floor = (config_.threshold > 0.0f)
            ? std::log2(config_.threshold)
            : static_cast<float>(std::numeric_limits<float>::min_exponent - 1);

        // The escalation threshold is the round-trip slack expressed in value
        // space: a log2-unit deviation of s corresponds to a relative deviation
        // of about s * ln(2) in |x|.
        const float rel_tolerance =
            kLogRoundTripSlack * static_cast<float>(M_LN2);

        const int grid = static_cast<int>((n + kBlock - 1) / kBlock);
        logTransformFwdKernel<TInput><<<grid, kBlock, 0, stream>>>(
            static_cast<const TInput*>(inputs[0]), n,
            config_.threshold, rel_tolerance, log_floor,
            static_cast<TInput*>(outputs[0]),
            static_cast<TInput*>(outputs[2]),
            static_cast<uint32_t*>(outputs[3]),
            d_outlier_count_scratch_,
            max_out);
        FZ_CUDA_CHECK(cudaGetLastError());

        const size_t n_sign_bytes = signBytes(n);
        const int sign_grid = static_cast<int>((n_sign_bytes + kBlock - 1) / kBlock);
        logTransformSignPackKernel<TInput><<<sign_grid, kBlock, 0, stream>>>(
            static_cast<const TInput*>(inputs[0]), n,
            static_cast<uint8_t*>(outputs[1]), n_sign_bytes);
        FZ_CUDA_CHECK(cudaGetLastError());

        // The log-value and sign streams are fully determined here. The outlier
        // sizes are not — they depend on the device counter, which is read back
        // in postStreamSync().
        actual_output_sizes_[0] = n * sizeof(TInput);
        actual_output_sizes_[1] = n_sign_bytes;
        actual_output_sizes_[2] = max_out * sizeof(TInput);
        actual_output_sizes_[3] = max_out * sizeof(uint32_t);

        FZ_LOG(DEBUG,
            "LogTransformStage forward: n=%zu delta=%.6g -> quantizer abs_eb=%.6g "
            "(threshold=%.6g)",
            n, static_cast<double>(config_.error_bound),
            static_cast<double>(quantizerErrorBound()),
            static_cast<double>(config_.threshold));
    } else {
        // ── Inverse ──────────────────────────────────────────────────────────
        if (inputs.size() < 4) {
            throw std::runtime_error(
                "LogTransformStage: inverse pass requires 4 input buffers");
        }
        const size_t n = (num_elements_ > 0) ? num_elements_
                                             : sizes[0] / sizeof(TInput);
        actual_output_sizes_.assign(1, 0);
        if (n == 0) return;

        const int grid = static_cast<int>((n + kBlock - 1) / kBlock);
        logTransformInvKernel<TInput><<<grid, kBlock, 0, stream>>>(
            static_cast<const TInput*>(inputs[0]),
            static_cast<const uint8_t*>(inputs[1]),
            n,
            static_cast<TInput*>(outputs[0]));
        FZ_CUDA_CHECK(cudaGetLastError());

        // Restore the losslessly-stored originals over the reconstructed values.
        // The count is a register argument read from the deserialized header —
        // the kernel never dereferences a device pointer to find its bound.
        if (actual_outlier_count_ > 0) {
            const int sgrid = static_cast<int>(
                (actual_outlier_count_ + kBlock - 1) / kBlock);
            scatter_assign_kernel<TInput><<<sgrid, kBlock, 0, stream>>>(
                static_cast<const TInput*>(inputs[2]),
                static_cast<const uint32_t*>(inputs[3]),
                actual_outlier_count_,
                static_cast<TInput*>(outputs[0]));
            FZ_CUDA_CHECK(cudaGetLastError());
        }

        actual_output_sizes_[0] = n * sizeof(TInput);
    }
}

template<typename TInput>
void LogTransformStage<TInput>::postStreamSync(cudaStream_t stream)
{
    if (is_inverse_ || d_outlier_count_scratch_ == nullptr) return;

    uint32_t h_count = 0;
    FZ_CUDA_CHECK(cudaMemcpyAsync(&h_count, d_outlier_count_scratch_,
                                  sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
    FZ_CUDA_CHECK(cudaStreamSynchronize(stream));

    // atomicAdd increments even when the buffer is full, so h_count may exceed
    // capacity; only min(h_count, cap) entries were actually written.
    const uint32_t cap = static_cast<uint32_t>(maxOutlierCount(num_elements_));
    const bool overflowed = (h_count > cap);
    actual_outlier_count_ = overflowed ? cap : h_count;

    actual_output_sizes_[2] = actual_outlier_count_ * sizeof(TInput);
    actual_output_sizes_[3] = actual_outlier_count_ * sizeof(uint32_t);

    if (overflowed) {
        // Unlike a quantizer, dropping an outlier here is not a bounded-error
        // event — the element's value is simply lost. That makes failing *more*
        // clearly correct here than elsewhere, not less.
        //
        // Was FZ_LOG(WARN), which is invisible at the default log level: the
        // "Reconstruction will be WRONG" warning nobody saw. Matches the policy
        // in LorenzoQuantStage / QuantizerStage / GInterpStage.
        char msg[512];
        std::snprintf(msg, sizeof(msg),
            "LogTransformStage: outlier buffer overflow — %u needed, capacity %u "
            "(%.1f%% of %zu elements). Reconstruction would be WRONG at the "
            "dropped positions — the values are lost, not merely coarsened. Raise "
            "setOutlierCapacity() to at least %.2f, or lower setThreshold().",
            h_count, cap,
            num_elements_ > 0 ? 100.0f * h_count / static_cast<float>(num_elements_) : 0.0f,
            num_elements_,
            num_elements_ > 0
                ? std::min(1.0f, 1.1f * h_count / static_cast<float>(num_elements_))
                : 1.0f);
        throw std::runtime_error(msg);
    }

    FZ_LOG(DEBUG, "LogTransformStage: %u / %zu outliers (%.1f%%)",
           actual_outlier_count_, num_elements_,
           num_elements_ > 0
               ? static_cast<double>(actual_outlier_count_) * 100.0
                     / static_cast<double>(num_elements_)
               : 0.0);
}

template<typename TInput>
size_t LogTransformStage<TInput>::serializeHeader(
    size_t /*output_index*/, uint8_t* buf, size_t max_size) const
{
    if (max_size < sizeof(LogTransformConfig)) {
        throw std::runtime_error(
            "Insufficient buffer for LogTransformConfig: need " +
            std::to_string(sizeof(LogTransformConfig)) + " bytes, got " +
            std::to_string(max_size));
    }

    LogTransformConfig cfg;
    cfg.error_bound   = config_.error_bound;
    cfg.threshold     = config_.threshold;
    cfg.num_elements  = static_cast<uint32_t>(num_elements_);
    cfg.outlier_count = actual_outlier_count_;
    cfg.input_type    = inputDataType();

    std::memcpy(buf, &cfg, sizeof(LogTransformConfig));
    return sizeof(LogTransformConfig);
}

template<typename TInput>
void LogTransformStage<TInput>::deserializeHeader(const uint8_t* buf, size_t size)
{
    if (size < sizeof(LogTransformConfig)) {
        throw std::runtime_error(
            "LogTransformConfig header too small: got " + std::to_string(size) +
            " bytes, expected " + std::to_string(sizeof(LogTransformConfig)));
    }

    LogTransformConfig cfg;
    std::memcpy(&cfg, buf, sizeof(LogTransformConfig));

    config_.error_bound   = cfg.error_bound;
    config_.threshold     = cfg.threshold;
    num_elements_         = cfg.num_elements;
    actual_outlier_count_ = cfg.outlier_count;
}

// =============================================================================
// Explicit instantiations
// =============================================================================
template class LogTransformStage<float>;

} // namespace fz

// ── FZM-header reconstruction (self-registered; see stage_registry.h) ─────────
namespace {
fz::Stage* LogTransform_fromHeader(const uint8_t* config, size_t config_size) {
    auto* s = new fz::LogTransformStage<float>();
    s->deserializeHeader(config, config_size);
    return s;
}
}  // namespace
FZ_REGISTER_STAGE_FACTORY(fz::StageType::LOG_TRANSFORM, LogTransform_fromHeader);
