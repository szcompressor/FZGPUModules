#include "predictors/quantizer/quantizer.h"
#include "predictors/predictor_utils.cuh"
#include "transforms/zigzag/zigzag.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include "mem/mempool.h"
#include "cuda_check.h"
#include "log.h"
#include <limits>

namespace fz {

// =============================================================================
// ABS / NOA kernels — uniform quantization in value space
// =============================================================================

/**
 * Forward: value → quantization code (ABS/NOA modes).
 *
 * Step size = 2 * abs_eb.
 * code = round(value / step) + quant_radius,  stored in [1, 2*quant_radius-1].
 * stored = 0 is the outlier sentinel (never a valid quantised value because
 * round(value/step) = -quant_radius would require |value| = quant_radius * step,
 * which exceeds the outlier threshold anyway).
 */
template<typename TInput, typename TCode, bool ZigzagCodes = false>
__global__ void quantizer_abs_fwd_kernel(
    const TInput* __restrict__ in, size_t n,
    TInput ebx2_r,          // 1 / (2 * abs_eb)
    float  threshold,       // |x| >= threshold → forced outlier (pass inf to disable)
    TCode  quant_radius,
    TCode* __restrict__     codes,
    TInput* __restrict__    outlier_vals,
    uint32_t* __restrict__  outlier_idxs,
    uint32_t* __restrict__  outlier_count,
    size_t max_outliers
) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n) return;

    TInput x = in[i];
    // Round to nearest quantization bin (signed, centred at 0)
    int q = __float2int_rn((float)x * (float)ebx2_r);

    // |q| < quant_radius and |x| < threshold → representable
    if (q > -(int)quant_radius && q < (int)quant_radius && fabsf((float)x) < threshold) {
        if constexpr (ZigzagCodes) {
            // Zigzag-encode: signed q → unsigned code clustering near 0
            using SCode = typename std::make_signed<TCode>::type;
            codes[i] = static_cast<TCode>(Zigzag<SCode>::encode(static_cast<SCode>(q)));
        } else {
            codes[i] = static_cast<TCode>(q);  // signed stored as two's-complement in TCode
        }
    } else {
        // Outlier: code value doesn't matter — scatter_assign_kernel overwrites the
        // position with the true value.  Store 0 for consistency.
        codes[i] = static_cast<TCode>(0);
        uint32_t slot = atomicAdd(outlier_count, 1u);
        if (slot < static_cast<uint32_t>(max_outliers)) {
            outlier_vals[slot] = x;
            outlier_idxs[slot] = static_cast<uint32_t>(i);
        }
    }
}

/**
 * Forward ABS/NOA in-place outlier variant (LC-reference encoding style).
 *
 * Outliers (|bin| >= quant_radius  OR  |x| >= threshold) are stored as their
 * raw IEEE-754 bit pattern directly in the codes array.  Valid codes are
 * TCMS (zigzag) encoded so the inverse sentinel check (code >> 1) >= quant_radius
 * is unambiguous:
 *   - Valid TCMS codes are in [0, 2 * quant_radius)
 *   - Normal float bits are always >= 0x00800000, which exceeds 2 * quant_radius
 *     for all practical values (quant_radius <= 1 << 22).
 *
 * REQUIREMENT: sizeof(TCode) == sizeof(TInput)  (use float / uint32_t pair).
 */
template<typename TInput, typename TCode>
__global__ void quantizer_abs_fwd_inplace_kernel(
    const TInput* __restrict__ in, size_t n,
    TInput ebx2_r,
    float  threshold,
    TCode  quant_radius,
    TCode* __restrict__ codes
) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n) return;

    TInput x  = in[i];
    float  fx = static_cast<float>(x);
    int    q  = __float2int_rn(fx * static_cast<float>(ebx2_r));

    if (q > -(int)quant_radius && q < (int)quant_radius && fabsf(fx) < threshold) {
        // TCMS (zigzag) encode: valid codes are in [0, 2 * quant_radius)
        uint32_t uq = static_cast<uint32_t>((q << 1) ^ (q >> 31));
        codes[i] = static_cast<TCode>(uq);
    } else {
        // In-place outlier: store raw IEEE-754 bit pattern of x in codes[i]
        TCode raw;
        __builtin_memcpy(&raw, &x, sizeof(TCode));
        codes[i] = raw;
    }
}

/**
 * Inverse: quantization code → value (ABS/NOA modes).
 *
 * code = 0 means outlier — those positions are left as 0 (from the preceding
 * cudaMemset) and will be overwritten by scatter_assign_kernel.
 */
template<typename TInput, typename TCode, bool ZigzagCodes = false>
__global__ void quantizer_abs_inv_kernel(
    const TCode* __restrict__ codes, size_t n,
    TInput ebx2,            // 2 * abs_eb
    TCode  quant_radius,
    TInput* __restrict__ out
) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n) return;

    int q;
    if constexpr (ZigzagCodes) {
        // Zigzag-decode: unsigned code → signed quantization index
        using SCode = typename std::make_signed<TCode>::type;
        q = static_cast<int>(Zigzag<SCode>::decode(codes[i]));
    } else {
        // Reinterpret the unsigned code as signed two's-complement to recover q.
        // Outlier positions will be overwritten by scatter_assign_kernel afterward,
        // so whatever dequant writes there is harmless.
        q = static_cast<int>(static_cast<typename std::make_signed<TCode>::type>(codes[i]));
    }
    out[i] = static_cast<TInput>(q) * ebx2;
}

/**
 * Inverse ABS/NOA in-place outlier variant.
 *
 * Detects outliers via (code >> 1) >= quant_radius (provably safe — see
 * quantizer_abs_fwd_inplace_kernel comment).  No scatter pass required.
 * TCMS decode used for valid codes (requires zigzag encoding on forward pass).
 */
template<typename TInput, typename TCode>
__global__ void quantizer_abs_inv_inplace_kernel(
    const TCode* __restrict__ codes, size_t n,
    TInput ebx2,
    TCode  quant_radius,
    TInput* __restrict__ out
) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n) return;

    TCode code = codes[i];
    if ((code >> 1) >= quant_radius) {
        // Outlier: reinterpret raw bits as TInput
        TInput x;
        __builtin_memcpy(&x, &code, sizeof(TInput));
        out[i] = x;
    } else {
        // TCMS (zigzag) decode: even code → q = code/2, odd code → q = -(code+1)/2
        int32_t c = static_cast<int32_t>(static_cast<uint32_t>(code));
        int q = (c & 1) ? -((c + 1) >> 1) : (c >> 1);
        out[i] = static_cast<TInput>(q) * ebx2;
    }
}

// =============================================================================
// REL kernels — log2-space quantization (PFPL exact pointwise relative bound)
// =============================================================================

/**
 * Fast IEEE-754 log2 approximation.
 * Valid only for normal positive finite floats.
 *
 * Uses the identity:   log2(f) = exponent + log2(mantissa)
 *                              ≈ exponent + (mantissa - 1)
 * Error ≤ 0.086 bits.  The explicit bounds check in the kernel catches the
 * rare cases where this error causes a bound violation.
 */
__device__ __forceinline__ float log2approx(float x) {
    uint32_t bits;
    __builtin_memcpy(&bits, &x, sizeof(float));
    int   expo = static_cast<int>((bits >> 23) & 0xFFu) - 127;
    float frac = __uint_as_float((bits & 0x7FFFFFu) | 0x3F800000u) - 1.0f;
    return static_cast<float>(expo) + frac;
}

/**
 * Fast IEEE-754 pow2 approximation.
 * Inverse of log2approx.
 *
 * Uses:  2^x = 2^floor(x) * 2^frac(x)  ≈  2^floor(x) * (1 + frac(x))
 */
__device__ __forceinline__ float pow2approx(float x) {
    int   xi   = __float2int_rd(x);   // floor(x)
    float frac = x - static_cast<float>(xi);
    uint32_t bits = static_cast<uint32_t>(xi + 127) << 23;
    float base;
    __builtin_memcpy(&base, &bits, sizeof(float));
    return base * (1.0f + frac);
}

/**
 * REL code packing for a single element.
 *
 * Stored word layout (fits in uint16_t for |log_bin| ≤ 16382, uint32_t otherwise):
 *   bit  0      : sign of x        (0 = positive)
 *   bit  1      : sign of log_bin  (0 = non-negative)
 *   bits 2..N   : |log_bin|
 *   stored = 0  : outlier sentinel (never emitted for valid codes)
 *
 * Decoding:
 *   code_val    = stored - 1
 *   is_neg_x    = code_val & 1
 *   is_neg_lb   = (code_val >> 1) & 1
 *   abs_log_bin = code_val >> 2
 *   log_bin     = is_neg_lb ? -abs_log_bin : abs_log_bin
 *   abs_recon   = pow2(log_bin * log2eb)
 *   x_recon     = is_neg_x ? -abs_recon : abs_recon
 */
__device__ __forceinline__ uint32_t pack_rel_code(int log_bin, bool x_negative) {
    uint32_t abs_lb  = static_cast<uint32_t>(log_bin < 0 ? -log_bin : log_bin);
    uint32_t lb_sign = (log_bin < 0) ? 1u : 0u;
    uint32_t x_sign  = x_negative    ? 1u : 0u;
    uint32_t code_val = (abs_lb << 2) | (lb_sign << 1) | x_sign;
    return code_val + 1u;  // +1 so that 0 is free as the outlier sentinel
}

template<typename TInput, typename TCode>
__global__ void quantizer_rel_fwd_kernel(
    const TInput* __restrict__ in, size_t n,
    float log2eb,               // 2 * log2(1 + epsilon)
    float log2eb_r,             // 1 / log2eb
    float one_plus_eb,          // 1 + epsilon
    float one_over_one_plus_eb, // 1 / (1 + epsilon)
    float threshold,            // |x| >= threshold → forced outlier (pass inf to disable)
    TCode quant_radius,
    TCode* __restrict__    codes,
    TInput* __restrict__   outlier_vals,
    uint32_t* __restrict__ outlier_idxs,
    uint32_t* __restrict__ outlier_count,
    size_t max_outliers
) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n) return;

    float x = static_cast<float>(in[i]);

    // ── Threshold check: force outlier if |x| >= threshold ─────────────────
    if (fabsf(x) >= threshold) {
        codes[i] = 0;
        uint32_t slot = atomicAdd(outlier_count, 1u);
        if (slot < static_cast<uint32_t>(max_outliers)) {
            outlier_vals[slot] = in[i];
            outlier_idxs[slot] = static_cast<uint32_t>(i);
        }
        return;
    }

    // ── Special-case detection ──────────────────────────────────────────────
    // Zero, denormals (expo == 0), inf/NaN (expo == 255) → lossless outlier.
    uint32_t xbits;
    __builtin_memcpy(&xbits, &x, sizeof(float));
    int expo = static_cast<int>((xbits >> 23) & 0xFFu);

    if (expo == 0 || expo == 255) {
        // Zero, denormal, inf, NaN
        codes[i] = 0;
        uint32_t slot = atomicAdd(outlier_count, 1u);
        if (slot < static_cast<uint32_t>(max_outliers)) {
            outlier_vals[slot] = in[i];
            outlier_idxs[slot] = static_cast<uint32_t>(i);
        }
        return;
    }

    // ── Log2-space quantisation ─────────────────────────────────────────────
    bool  is_neg = (x < 0.0f);
    float abs_x  = is_neg ? -x : x;

    // Use hardware log2 (2 ULP error) instead of log2approx so that bin
    // selection is accurate even for tight error bounds like 1e-3.
    // log2approx has up to 0.086-bit error; for eb=1e-3, log2eb_r≈346, so
    // the approximation error shifts the bin by ~30 positions and pushes
    // ~7% of valid elements into the outlier path.
    float log_f   = __log2f(abs_x);
    int   log_bin = __float2int_rn(log_f * log2eb_r);  // round to nearest bin

    uint32_t stored = pack_rel_code(log_bin, is_neg);
    uint32_t abs_lb = static_cast<uint32_t>(log_bin < 0 ? -log_bin : log_bin);

    // Check it fits in TCode and within quant_radius
    bool fits = (abs_lb < static_cast<uint32_t>(quant_radius))
             && (stored <= static_cast<uint32_t>(std::numeric_limits<TCode>::max()));

    if (fits) {
        // ── Verification: check actual reconstruction is within error bound ─────
        // Use __exp2f (hardware, ~2 ULP) instead of pow2approx: the linear
        // 1+frac approximation has ~5% error at frac≈0.6, far exceeding
        // the ±0.1% window at eb=1e-3.
        float log_recon = static_cast<float>(log_bin) * log2eb;
        float abs_recon = exp2f(log_recon);
        float lower     = abs_x * one_over_one_plus_eb;
        float upper     = abs_x * one_plus_eb;

        if (abs_recon >= lower && abs_recon <= upper) {
            codes[i] = static_cast<TCode>(stored);
            return;
        }
    }

    // ── Outlier path ────────────────────────────────────────────────────────
    codes[i] = 0;
    uint32_t slot = atomicAdd(outlier_count, 1u);
    if (slot < static_cast<uint32_t>(max_outliers)) {
        outlier_vals[slot] = in[i];
        outlier_idxs[slot] = static_cast<uint32_t>(i);
    }
}

/**
 * REL inverse: decode the packed log-bin code back to a value.
 *
 * code == 0 → outlier placeholder (position already 0 from memset;
 *             scatter_assign_kernel will overwrite with the original).
 */
template<typename TInput, typename TCode>
__global__ void quantizer_rel_inv_kernel(
    const TCode* __restrict__ codes, size_t n,
    float log2eb,
    TInput* __restrict__ out
) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n) return;

    uint32_t stored = static_cast<uint32_t>(codes[i]);
    if (stored == 0) return;  // outlier: leave as 0, scatter handles it

    uint32_t code_val  = stored - 1u;
    bool     is_neg_x  = (code_val & 1u) != 0u;
    bool     is_neg_lb = ((code_val >> 1) & 1u) != 0u;
    int      abs_lb    = static_cast<int>(code_val >> 2);
    int      log_bin   = is_neg_lb ? -abs_lb : abs_lb;

    // Use __exp2f (hardware, ~2 ULP) for accurate reconstruction.
    // pow2approx has ~5% error at fractional parts near 0.6, which exceeds
    // even loose error bounds and corrupts the decompressed data.
    float abs_recon = exp2f(static_cast<float>(log_bin) * log2eb);
    out[i] = static_cast<TInput>(is_neg_x ? -abs_recon : abs_recon);
}

// =============================================================================
// QuantizerStage implementation
// =============================================================================

template<typename TInput, typename TCode>
QuantizerStage<TInput, TCode>::QuantizerStage(const Config& config)
    : config_(config),
      computed_abs_eb_(static_cast<TInput>(config.error_bound)),
      computed_value_base_(config.precomputed_value_base) {
    actual_output_sizes_.resize(4, 0);
}

template<typename TInput, typename TCode>
std::vector<size_t> QuantizerStage<TInput, TCode>::estimateOutputSizes(
    const std::vector<size_t>& input_sizes
) const {
    if (is_inverse_) {
        // inputs: codes, outlier_vals, outlier_idxs, outlier_count
        size_t num_elements = input_sizes.empty() ? 0 : input_sizes[0] / sizeof(TCode);
        return {num_elements * sizeof(TInput)};
    }
    size_t n = input_sizes.empty() ? 0 : input_sizes[0] / sizeof(TInput);
    if (isInplaceMode()) return {n * sizeof(TCode)};  // codes only, no scatter buffers
    size_t max_outliers = getMaxOutlierCount(n);
    return {
        n            * sizeof(TCode),    // codes
        max_outliers * sizeof(TInput),   // outlier_vals
        max_outliers * sizeof(uint32_t), // outlier_idxs
        sizeof(uint32_t)                 // outlier_count
    };
}

template<typename TInput, typename TCode>
void QuantizerStage<TInput, TCode>::execute(
    cudaStream_t stream,
    MemoryPool* pool,
    const std::vector<void*>& inputs,
    const std::vector<void*>& outputs,
    const std::vector<size_t>& sizes
) {
    // =========================================================================
    // DECOMPRESSION MODE — 4 inputs → 1 output (1 input in inplace mode)
    // =========================================================================
    if (is_inverse_) {
        const size_t expected_inputs = isInplaceMode() ? 1 : 4;
        if (inputs.size() < expected_inputs || outputs.empty() || sizes.empty()) {
            throw std::runtime_error(
                isInplaceMode()
                    ? "QuantizerStage (inverse, inplace): requires 1 input and 1 output"
                    : "QuantizerStage (inverse): requires 4 inputs and 1 output");
        }

        size_t num_elements = sizes[0] / sizeof(TCode);
        if (num_elements == 0) {
            actual_output_sizes_ = {0};
            return;
        }

        size_t max_outliers = (!isInplaceMode() && sizes.size() > 1)
                              ? sizes[1] / sizeof(TInput) : 0;

        // Zero output for scatter-based path (outlier positions overwritten by scatter).
        // In-place path writes every element directly — no memset needed.
        if (!isInplaceMode()) {
            FZ_CUDA_CHECK(cudaMemsetAsync(
                outputs[0], 0, num_elements * sizeof(TInput), stream));
        }

        constexpr int kBlock = 256;
        int grid = static_cast<int>((num_elements + kBlock - 1) / kBlock);

        if (config_.eb_mode == ErrorBoundMode::REL) {
            // REL inverse: decode packed log-bin codes
            float log2eb = 2.0f * std::log2(1.0f + config_.error_bound);

            quantizer_rel_inv_kernel<TInput, TCode><<<grid, kBlock, 0, stream>>>(
                static_cast<const TCode*>(inputs[0]),
                num_elements,
                log2eb,
                static_cast<TInput*>(outputs[0])
            );
        } else {
            // ABS / NOA inverse: dequantize with stored abs_eb
            TInput ebx2 = static_cast<TInput>(2) * computed_abs_eb_;

            if (isInplaceMode()) {
                // In-place: every element is written directly (no scatter needed)
                quantizer_abs_inv_inplace_kernel<TInput, TCode><<<grid, kBlock, 0, stream>>>(
                    static_cast<const TCode*>(inputs[0]),
                    num_elements, ebx2,
                    static_cast<TCode>(config_.quant_radius),
                    static_cast<TInput*>(outputs[0])
                );
                FZ_CUDA_CHECK(cudaGetLastError());
                actual_output_sizes_ = {num_elements * sizeof(TInput)};
                return;
            }

            if (config_.zigzag_codes) {
                quantizer_abs_inv_kernel<TInput, TCode, true><<<grid, kBlock, 0, stream>>>(
                    static_cast<const TCode*>(inputs[0]),
                    num_elements, ebx2,
                    static_cast<TCode>(config_.quant_radius),
                    static_cast<TInput*>(outputs[0])
                );
            } else {
                quantizer_abs_inv_kernel<TInput, TCode, false><<<grid, kBlock, 0, stream>>>(
                    static_cast<const TCode*>(inputs[0]),
                    num_elements, ebx2,
                    static_cast<TCode>(config_.quant_radius),
                    static_cast<TInput*>(outputs[0])
                );
            }
        }

        FZ_CUDA_CHECK(cudaGetLastError());

        // Scatter outliers back to their original positions
        if (max_outliers > 0) {
            int sblk  = 256;
            int sgrid = static_cast<int>(
                (max_outliers + sblk - 1) / sblk);

            scatter_assign_kernel<TInput><<<sgrid, sblk, 0, stream>>>(
                static_cast<const TInput*>(inputs[1]),
                static_cast<const uint32_t*>(inputs[2]),
                static_cast<const uint32_t*>(inputs[3]),
                static_cast<TInput*>(outputs[0])
            );
            FZ_CUDA_CHECK(cudaGetLastError());
        }

        actual_output_sizes_ = {num_elements * sizeof(TInput)};
        return;
    }

    // =========================================================================
    // COMPRESSION MODE — 1 input → 4 outputs  (or 1 in inplace mode)
    // =========================================================================
    {
        const size_t expected_outputs = isInplaceMode() ? 1 : 4;
        if (inputs.empty() || outputs.size() < expected_outputs || sizes.empty()) {
            throw std::runtime_error(
                isInplaceMode()
                    ? "QuantizerStage (inplace): requires 1 input and 1 output"
                    : "QuantizerStage: requires 1 input and 4 outputs");
        }
    }

    size_t input_size   = sizes[0];
    size_t num_elements = input_size / sizeof(TInput);
    size_t max_outliers = getMaxOutlierCount(num_elements);

    num_elements_ = num_elements;

    if (num_elements == 0) {
        if (isInplaceMode()) {
            actual_output_sizes_ = {0};
        } else {
            for (size_t j = 0; j < 4; j++) actual_output_sizes_[j] = 0;
        }
        actual_outlier_count_ = 0;
        return;
    }

    // Zero the outlier_count scalar so atomic increments start from 0.
    // Not needed in inplace mode (no separate outlier counter buffer).
    if (!isInplaceMode()) {
        FZ_CUDA_CHECK(cudaMemsetAsync(outputs[3], 0, sizeof(uint32_t), stream));
    }

    // ── Resolve absolute error bound ──────────────────────────────────────────
    if (config_.eb_mode == ErrorBoundMode::ABS) {
        computed_abs_eb_     = static_cast<TInput>(config_.error_bound);
        computed_value_base_ = 0.0f;
    } else if (config_.eb_mode == ErrorBoundMode::REL) {
        // REL: no abs_eb needed (log-space kernels use user_eb directly)
        computed_abs_eb_     = static_cast<TInput>(0);
        computed_value_base_ = 0.0f;
    } else {
        // NOA: scan for value_range
        float value_base = config_.precomputed_value_base;
        if (value_base <= 0.0f) {
            value_base = computeValueBase<TInput>(
                static_cast<const TInput*>(inputs[0]),
                num_elements, ErrorBoundMode::NOA, stream, pool);
        }
        computed_value_base_ = value_base;
        if (value_base <= 0.0f) {
            FZ_LOG(WARN,
                "QuantizerStage NOA: value_range is zero (constant/empty data?); "
                "falling back to ABS with user_eb");
            computed_abs_eb_ = static_cast<TInput>(config_.error_bound);
        } else {
            computed_abs_eb_ = static_cast<TInput>(config_.error_bound)
                               * static_cast<TInput>(value_base);
        }
        FZ_LOG(DEBUG,
            "QuantizerStage NOA: user_eb=%.6g value_range=%.6g -> abs_eb=%.6g",
            static_cast<double>(config_.error_bound),
            static_cast<double>(value_base),
            static_cast<double>(computed_abs_eb_));
    }

    // ── Launch forward kernel ─────────────────────────────────────────────────
    constexpr int kBlock = 256;
    int grid = static_cast<int>((num_elements + kBlock - 1) / kBlock);

    if (config_.eb_mode == ErrorBoundMode::REL) {
        float epsilon    = config_.error_bound;
        float log2eb     = 2.0f * std::log2(1.0f + epsilon);
        float log2eb_r   = 1.0f / log2eb;
        float opp_eb     = 1.0f + epsilon;
        float oopp_eb    = 1.0f / opp_eb;

        FZ_LOG(INFO, "QuantizerStage REL: param epsilon=%.6g log2eb=%.6g max_outliers=%zu num_elements=%zu",
               static_cast<double>(epsilon), static_cast<double>(log2eb), max_outliers, num_elements);

        quantizer_rel_fwd_kernel<TInput, TCode><<<grid, kBlock, 0, stream>>>(
            static_cast<const TInput*>(inputs[0]),
            num_elements,
            log2eb, log2eb_r, opp_eb, oopp_eb,
            config_.outlier_threshold,
            static_cast<TCode>(config_.quant_radius),
            static_cast<TCode*>(outputs[0]),
            static_cast<TInput*>(outputs[1]),
            static_cast<uint32_t*>(outputs[2]),
            static_cast<uint32_t*>(outputs[3]),
            max_outliers
        );
    } else {
        // ABS / NOA
        TInput ebx2_r = static_cast<TInput>(1)
                        / (static_cast<TInput>(2) * computed_abs_eb_);

        if (isInplaceMode()) {
            // In-place outlier encoding: raw float bits stored in codes array.
            // Requires zigzag_codes=true and sizeof(TCode)==sizeof(TInput).
            if (!config_.zigzag_codes)
                throw std::runtime_error(
                    "QuantizerStage: setInplaceOutliers(true) requires setZigzagCodes(true)");
            if (sizeof(TCode) != sizeof(TInput))
                throw std::runtime_error(
                    "QuantizerStage: setInplaceOutliers(true) requires sizeof(TCode)==sizeof(TInput) "
                    "(use QuantizerStage<float, uint32_t>)");

            quantizer_abs_fwd_inplace_kernel<TInput, TCode><<<grid, kBlock, 0, stream>>>(
                static_cast<const TInput*>(inputs[0]),
                num_elements, ebx2_r,
                config_.outlier_threshold,
                static_cast<TCode>(config_.quant_radius),
                static_cast<TCode*>(outputs[0])
            );
            FZ_CUDA_CHECK(cudaGetLastError());
            d_outlier_count_ptr_ = nullptr;
            actual_output_sizes_ = {num_elements * sizeof(TCode)};
            return;
        }

        if (config_.zigzag_codes) {
            quantizer_abs_fwd_kernel<TInput, TCode, true><<<grid, kBlock, 0, stream>>>(
                static_cast<const TInput*>(inputs[0]),
                num_elements, ebx2_r, config_.outlier_threshold,
                static_cast<TCode>(config_.quant_radius),
                static_cast<TCode*>(outputs[0]),
                static_cast<TInput*>(outputs[1]),
                static_cast<uint32_t*>(outputs[2]),
                static_cast<uint32_t*>(outputs[3]),
                max_outliers
            );
        } else {
            quantizer_abs_fwd_kernel<TInput, TCode, false><<<grid, kBlock, 0, stream>>>(
                static_cast<const TInput*>(inputs[0]),
                num_elements, ebx2_r, config_.outlier_threshold,
                static_cast<TCode>(config_.quant_radius),
                static_cast<TCode*>(outputs[0]),
                static_cast<TInput*>(outputs[1]),
                static_cast<uint32_t*>(outputs[2]),
                static_cast<uint32_t*>(outputs[3]),
                max_outliers
            );
        }
    }

    FZ_CUDA_CHECK(cudaGetLastError());

    d_outlier_count_ptr_ = outputs[3];

    actual_outlier_count_    = 0;
    actual_output_sizes_[0]  = num_elements   * sizeof(TCode);
    actual_output_sizes_[1]  = max_outliers   * sizeof(TInput);
    actual_output_sizes_[2]  = max_outliers   * sizeof(uint32_t);
    actual_output_sizes_[3]  = sizeof(uint32_t);
}

template<typename TInput, typename TCode>
void QuantizerStage<TInput, TCode>::postStreamSync(cudaStream_t /*stream*/) {
    if (is_inverse_ || d_outlier_count_ptr_ == nullptr) return;

    uint32_t h_count = 0;
    FZ_CUDA_CHECK(cudaMemcpy(&h_count, d_outlier_count_ptr_,
                              sizeof(uint32_t), cudaMemcpyDeviceToHost));

    // Cap at the allocated capacity: atomicAdd always increments the device
    // counter even when the buffer is full, so h_count may exceed max_outliers.
    // Only min(h_count, max_outliers) entries were actually written.  Write the
    // capped value back to the device so the inverse scatter_assign_kernel reads
    // the correct count (if h_count > cap, the uncapped value would cause OOB).
    uint32_t cap = static_cast<uint32_t>(getMaxOutlierCount(num_elements_));
    actual_outlier_count_ = (h_count > cap) ? cap : h_count;
    if (h_count > cap) {
        FZ_CUDA_CHECK(cudaMemcpy(
            const_cast<void*>(d_outlier_count_ptr_),
            &actual_outlier_count_, sizeof(uint32_t),
            cudaMemcpyHostToDevice));
    }
    d_outlier_count_ptr_ = nullptr;

    actual_output_sizes_[1] = actual_outlier_count_ * sizeof(TInput);
    actual_output_sizes_[2] = actual_outlier_count_ * sizeof(uint32_t);

    FZ_LOG(DEBUG, "QuantizerStage: %u / %zu outliers (%.1f%%)%s",
           actual_outlier_count_, num_elements_,
           num_elements_ > 0
               ? static_cast<double>(actual_outlier_count_) * 100.0 / static_cast<double>(num_elements_)
               : 0.0,
           h_count > cap ? " [outlier buffer full — excess dropped]" : "");
}

template<typename TInput, typename TCode>
size_t QuantizerStage<TInput, TCode>::serializeHeader(
    size_t /*output_index*/, uint8_t* buf, size_t max_size
) const {
    if (max_size < sizeof(QuantizerConfig))
        throw std::runtime_error(
            "Insufficient buffer for QuantizerConfig: need " +
            std::to_string(sizeof(QuantizerConfig)) + " bytes, got " +
            std::to_string(max_size));

    QuantizerConfig cfg;
    cfg.abs_error_bound  = static_cast<float>(computed_abs_eb_);
    cfg.user_error_bound = config_.error_bound;
    cfg.value_base       = computed_value_base_;
    cfg.quant_radius     = static_cast<uint32_t>(config_.quant_radius);
    cfg.num_elements     = static_cast<uint32_t>(num_elements_);
    cfg.outlier_count    = actual_outlier_count_;
    cfg.input_type       = getInputDataType();
    cfg.code_type        = getCodeDataType();
    cfg.eb_mode           = static_cast<uint8_t>(config_.eb_mode);
    cfg.zigzag_codes      = config_.zigzag_codes ? uint8_t{1} : uint8_t{0};
    cfg.outlier_threshold = config_.outlier_threshold;
    cfg.inplace_outliers  = config_.inplace_outliers ? uint8_t{1} : uint8_t{0};

    std::memcpy(buf, &cfg, sizeof(QuantizerConfig));
    return sizeof(QuantizerConfig);
}

template<typename TInput, typename TCode>
void QuantizerStage<TInput, TCode>::deserializeHeader(
    const uint8_t* buf, size_t size
) {
    // Accept both old (28-byte) and new (36-byte) header formats.
    constexpr size_t kMinSize = 28;  // size before outlier_threshold/inplace_outliers fields
    if (size < kMinSize)
        throw std::runtime_error(
            "QuantizerConfig header too small: got " + std::to_string(size) +
            " bytes, minimum is " + std::to_string(kMinSize) +
            " — file may be from an incompatible older version");

    QuantizerConfig cfg;
    std::memcpy(&cfg, buf, std::min(size, sizeof(QuantizerConfig)));

    config_.error_bound   = cfg.user_error_bound;
    config_.quant_radius  = static_cast<int>(cfg.quant_radius);
    config_.eb_mode       = static_cast<ErrorBoundMode>(cfg.eb_mode);
    config_.zigzag_codes  = (cfg.zigzag_codes != 0);
    num_elements_         = cfg.num_elements;
    actual_outlier_count_ = cfg.outlier_count;
    computed_abs_eb_      = static_cast<TInput>(cfg.abs_error_bound);
    computed_value_base_  = cfg.value_base;
    // New fields: default to safe values when reading old-format headers
    if (size >= sizeof(QuantizerConfig)) {
        config_.outlier_threshold = cfg.outlier_threshold;
        config_.inplace_outliers  = (cfg.inplace_outliers != 0);
    } else {
        config_.outlier_threshold = std::numeric_limits<float>::infinity();
        config_.inplace_outliers  = false;
    }
}

// =============================================================================
// Explicit instantiations
// =============================================================================
template class QuantizerStage<float,  uint16_t>;
template class QuantizerStage<float,  uint32_t>;
template class QuantizerStage<double, uint16_t>;
template class QuantizerStage<double, uint32_t>;

} // namespace fz
