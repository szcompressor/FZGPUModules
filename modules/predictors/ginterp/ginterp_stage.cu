// Algorithm adapted from cuSZ-Hi (Indiana University, Argonne National Laboratory,
// https://github.com/shixun404/cuSZ-Hi), BSD-3-Clause. See THIRD_PARTY.md.

#include "predictors/ginterp/ginterp_stage.h"
#include "predictors/ginterp/ginterp_kernels.h"
#include "predictors/predictor_utils.cuh"
#include "mem/mempool.h"
#include "cuda_check.h"
#include "log.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace fz {

// ─── pickAutoRadius ───────────────────────────────────────────────────────────
// Choose a quantizer radius based on observed data range. The constraint is
// that residuals at the coarsest interpolation level (where `cur_ebx2 = ebx2/beta`
// with default beta=4) must satisfy `|err| < radius * cur_ebx2` to be
// quantizable; anything beyond goes to the outlier triplet.
//
// We size `radius = ceil(data_range / cur_ebx2_min)` so that even a residual
// equal to the full data range stays quantizable, then clamp to the TCode
// bit-width's maximum (codes are stored as TCode). The minimum of 32 keeps
// constant or near-constant data from degenerating to radius=0.
template <typename TCode>
static int pickAutoRadius(float data_range, float ebx2)
{
    constexpr int kMaxForCode =
        (sizeof(TCode) == 1) ?    127 :   // uint8: codes in [0, 255]
        (sizeof(TCode) == 2) ?  32767 :   // uint16: codes in [0, 65535]
                                32767;    // uint32: cap at uint16-max — past
                                          // this, float precision in the
                                          // outlier-as-float path dominates.
    constexpr int kMinRadius = 32;
    if (data_range <= 0.0f || ebx2 <= 0.0f) return kMinRadius;
    const float cur_ebx2_min = ebx2 / 4.0f;  // beta=4 default
    const double needed =
        std::ceil(static_cast<double>(data_range) / cur_ebx2_min);
    if (needed >= static_cast<double>(kMaxForCode)) return kMaxForCode;
    return std::max(kMinRadius, static_cast<int>(needed));
}

// ─── setDims ─────────────────────────────────────────────────────────────────
template <typename TInput, typename TCode>
void GInterpStage<TInput, TCode>::setDims(const std::array<size_t, 3>& dims) {
    if (dims[0] == 0 || dims[1] == 0 || dims[2] == 0) {
        throw std::runtime_error(
            "GInterpStage::setDims: all three dimensions must be > 0 "
            "(MVP requires 3-D input — 2-D/1-D in later phases)");
    }
    if (dims[2] <= 1) {
        throw std::runtime_error(
            "GInterpStage::setDims: dims[2] must be > 1; only 3-D input is "
            "supported in the MVP (got z=" + std::to_string(dims[2]) + ")");
    }
    config_.dims = dims;
    auto anchor = ginterp::ginterpAnchorLen3(dims[0], dims[1], dims[2]);
    anchor_dims_ = {anchor.x, anchor.y, anchor.z};
}

// ─── estimateOutputSizes ─────────────────────────────────────────────────────
template <typename TInput, typename TCode>
std::vector<size_t> GInterpStage<TInput, TCode>::estimateOutputSizes(
    const std::vector<size_t>& input_sizes) const
{
    if (input_sizes.empty()) return {0, 0, 0, 0, 0};
    size_t input_bytes = input_sizes[0];
    size_t N           = input_bytes / sizeof(TInput);
    size_t max_outliers = getMaxOutlierCount(N);

    // Anchor count from cached dims (set by setDims). If setDims wasn't
    // called (e.g. estimate is being called before finalize from a pre-finalize
    // path), fall back to a conservative 1/64 estimate (worst case for the
    // smallest legal volume: nx=ny=nz=2 → anchor 1×1×1, but we use a
    // generous upper bound here since the DAG only sizes once).
    size_t anchor_N = (anchor_dims_[0] && anchor_dims_[1] && anchor_dims_[2])
        ? (anchor_dims_[0] * anchor_dims_[1] * anchor_dims_[2])
        : ((N + 4095) / 4096);  // ≤ 1/4096 of N for any valid setDims()

    return {
        N           * sizeof(TCode),    // codes
        anchor_N    * sizeof(TInput),   // anchor
        max_outliers * sizeof(TInput),  // outlier_vals
        max_outliers * sizeof(uint32_t),// outlier_idxs
        sizeof(uint32_t),               // outlier_count
    };
}

// ─── execute ─────────────────────────────────────────────────────────────────
template <typename TInput, typename TCode>
void GInterpStage<TInput, TCode>::execute(
    cudaStream_t stream,
    MemoryPool* pool,
    const std::vector<void*>& inputs,
    const std::vector<void*>& outputs,
    const std::vector<size_t>& sizes)
{
    if (is_inverse_) {
        // ───── DECOMPRESS: 5 inputs → 1 output ────────────────────────────
        if (inputs.size() < 5 || outputs.empty() || sizes.size() < 5) {
            throw std::runtime_error(
                "GInterpStage (inverse): requires 5 inputs and 1 output");
        }
        if (config_.dims[0] == 0 || config_.dims[2] <= 1) {
            throw std::runtime_error(
                "GInterpStage (inverse): dims not set — deserializeHeader() "
                "should have populated them");
        }

        size_t nx = config_.dims[0];
        size_t ny = config_.dims[1];
        size_t nz = config_.dims[2];
        size_t N  = nx * ny * nz;
        if (N == 0) {
            actual_output_sizes_.assign(1, 0);
            return;
        }

        // Conversion factors (computed_abs_eb_ set by deserializeHeader).
        // cuSZ-Hi convention: eb_r = 1/eb (NOT 1/(2*eb)). The kernel's
        // quantizer at ginterp_md.inl:865 does `int(code/2)`, which combined
        // with `eb_r = 1/eb` gives the canonical `round(err / (2*eb))` mapping.
        TInput abs_eb = computed_abs_eb_;
        float  ebx2   = static_cast<float>(2.0 * abs_eb);
        float  eb_r   = (abs_eb > TInput(0)) ? (1.0f / static_cast<float>(abs_eb)) : 0.0f;

        // Capacity from inputs[2] size (outlier_vals byte count).
        size_t max_outliers = sizes[2] / sizeof(TInput);

        // 1. Allocate full-N outlier_tmp from the pool (stream-ordered).
        TInput* d_outlier_tmp = static_cast<TInput*>(
            pool->allocate(N * sizeof(TInput), stream,
                           "ginterp_outlier_tmp", /*persistent=*/false));
        FZ_CUDA_CHECK(cudaMemsetAsync(d_outlier_tmp, 0,
                                       N * sizeof(TInput), stream));

        // 2. Scatter outliers into the temp buffer using on-device count.
        ginterp::launchScatterOutliers<TInput>(
            static_cast<const TInput*>(inputs[2]),
            static_cast<const uint32_t*>(inputs[3]),
            static_cast<const uint32_t*>(inputs[4]),
            d_outlier_tmp,
            max_outliers,
            stream);
        FZ_CUDA_CHECK(cudaGetLastError());

        // 3. Launch the decode kernel.
        dim3 data_len3(static_cast<unsigned>(nx),
                       static_cast<unsigned>(ny),
                       static_cast<unsigned>(nz));
        dim3 anchor_len3(static_cast<unsigned>(anchor_dims_[0]),
                         static_cast<unsigned>(anchor_dims_[1]),
                         static_cast<unsigned>(anchor_dims_[2]));

        ginterp::launchGInterpInverse3D<TInput, TCode>(
            static_cast<const TCode*>(inputs[0]),  data_len3,
            static_cast<const TInput*>(inputs[1]), anchor_len3,
            d_outlier_tmp,
            static_cast<TInput*>(outputs[0]),
            eb_r, ebx2, static_cast<int>(config_.quant_radius),
            stream);
        FZ_CUDA_CHECK(cudaGetLastError());

        // 4. Return outlier_tmp to the pool (stream-ordered free).
        pool->free(d_outlier_tmp, stream);

        actual_output_sizes_.assign(1, N * sizeof(TInput));
        return;
    }

    // ───── COMPRESS: 1 input → 5 outputs ─────────────────────────────────────
    if (inputs.empty() || outputs.size() < 5 || sizes.empty()) {
        throw std::runtime_error(
            "GInterpStage: requires 1 input and 5 outputs");
    }
    if (config_.dims[0] == 0 || config_.dims[2] <= 1) {
        throw std::runtime_error(
            "GInterpStage: dims not set — call Pipeline::setDims() "
            "before addStage() (3-D only in MVP)");
    }

    size_t input_bytes  = sizes[0];
    size_t N            = input_bytes / sizeof(TInput);
    size_t max_outliers = getMaxOutlierCount(N);
    num_elements_       = N;

    if (N == 0) {
        for (size_t i = 0; i < 5; i++) actual_output_sizes_[i] = 0;
        actual_outlier_count_ = 0;
        return;
    }

    // Zero outlier_count before kernel launch (kernel uses atomicAdd).
    FZ_CUDA_CHECK(cudaMemsetAsync(outputs[4], 0, sizeof(uint32_t), stream));

    // Resolve absolute error bound (ABS direct; REL/NOA via scan unless caller
    // pre-computed value_base for graph-safe operation).
    if (config_.eb_mode == ErrorBoundMode::ABS) {
        computed_abs_eb_     = static_cast<TInput>(config_.error_bound);
        computed_value_base_ = 0.0f;
    } else {
        float value_base = config_.precomputed_value_base;
        if (value_base <= 0.0f) {
            value_base = computeValueBase<TInput>(
                static_cast<const TInput*>(inputs[0]),
                N, config_.eb_mode, stream, pool);
        }
        computed_value_base_ = value_base;
        if (value_base <= 0.0f) {
            FZ_LOG(WARN,
                "GInterpStage: value_base is zero for %s mode "
                "(constant or empty data?); falling back to ABS",
                config_.eb_mode == ErrorBoundMode::NOA ? "NOA" : "REL");
            computed_abs_eb_ = static_cast<TInput>(config_.error_bound);
        } else {
            computed_abs_eb_ = static_cast<TInput>(config_.error_bound)
                                * static_cast<TInput>(value_base);
        }
    }

    // cuSZ-Hi convention: eb_r = 1/eb (see inverse path comment above).
    float ebx2 = static_cast<float>(2.0 * computed_abs_eb_);
    float eb_r = (computed_abs_eb_ > TInput(0))
                 ? (1.0f / static_cast<float>(computed_abs_eb_)) : 0.0f;

    // ── Auto-tune radius if user left it at the sentinel (0) ──
    // Manual override (any radius > 0) skips the scan — required for CUDA
    // graph capture, since the scan does a D2H sync inside computeValueBase.
    if (config_.quant_radius == 0 && ebx2 > 0.0f) {
        float data_range = 0.0f;
        if (config_.eb_mode == ErrorBoundMode::NOA && computed_value_base_ > 0.0f) {
            // NOA's value_base is already (max - min); reuse it.
            data_range = computed_value_base_;
        } else if (config_.eb_mode == ErrorBoundMode::REL && computed_value_base_ > 0.0f) {
            // REL's value_base is max(|data|); the full range is at most 2x.
            data_range = 2.0f * computed_value_base_;
        } else {
            // ABS, or NOA/REL without pre-computed value_base: do the scan.
            // We always invoke NOA mode here because we want the actual data
            // range (max - min), not a magnitude.
            data_range = computeValueBase<TInput>(
                static_cast<const TInput*>(inputs[0]),
                N, ErrorBoundMode::NOA, stream, pool);
        }
        int auto_r = pickAutoRadius<TCode>(data_range, ebx2);
        config_.quant_radius = auto_r;
        FZ_LOG(DEBUG,
               "GInterpStage: auto-tuned radius=%d (data_range=%.6g, ebx2=%.6g)",
               auto_r, static_cast<double>(data_range),
               static_cast<double>(ebx2));
    }

    dim3 data_len3(static_cast<unsigned>(config_.dims[0]),
                   static_cast<unsigned>(config_.dims[1]),
                   static_cast<unsigned>(config_.dims[2]));
    dim3 anchor_len3(static_cast<unsigned>(anchor_dims_[0]),
                     static_cast<unsigned>(anchor_dims_[1]),
                     static_cast<unsigned>(anchor_dims_[2]));

    ginterp::launchGInterpForward3D<TInput, TCode>(
        static_cast<const TInput*>(inputs[0]), data_len3,
        static_cast<TCode*>(outputs[0]),
        static_cast<TInput*>(outputs[1]), anchor_len3,
        static_cast<TInput*>(outputs[2]),
        static_cast<uint32_t*>(outputs[3]),
        static_cast<uint32_t*>(outputs[4]),
        eb_r, ebx2, static_cast<int>(config_.quant_radius),
        stream);
    FZ_CUDA_CHECK(cudaGetLastError());

    // Stash for postStreamSync(); using cudaMemcpy here would stall the DAG.
    d_outlier_count_ptr_ = outputs[4];

    // Max-capacity placeholders; postStreamSync() trims to actual count.
    actual_outlier_count_       = 0;
    size_t anchor_N             = anchor_dims_[0] * anchor_dims_[1] * anchor_dims_[2];
    actual_output_sizes_[0]     = N           * sizeof(TCode);
    actual_output_sizes_[1]     = anchor_N    * sizeof(TInput);
    actual_output_sizes_[2]     = max_outliers * sizeof(TInput);
    actual_output_sizes_[3]     = max_outliers * sizeof(uint32_t);
    actual_output_sizes_[4]     = sizeof(uint32_t);
}

// ─── postStreamSync ──────────────────────────────────────────────────────────
template <typename TInput, typename TCode>
void GInterpStage<TInput, TCode>::postStreamSync(cudaStream_t /*stream*/) {
    if (is_inverse_ || d_outlier_count_ptr_ == nullptr) return;

    uint32_t h_outlier_count = 0;
    FZ_CUDA_CHECK(cudaMemcpy(&h_outlier_count, d_outlier_count_ptr_,
                              sizeof(uint32_t), cudaMemcpyDeviceToHost));
    d_outlier_count_ptr_ = nullptr;

    size_t max_outliers = getMaxOutlierCount(num_elements_);
    if (h_outlier_count > max_outliers) {
        float actual_pct   = 100.0f * h_outlier_count
                              / static_cast<float>(num_elements_);
        float capacity_pct = 100.0f * max_outliers
                              / static_cast<float>(num_elements_);
        FZ_LOG(WARN,
               "GInterp outlier overflow! Detected %u (%.1f%%) outliers but "
               "only %.1f%% capacity allocated. Outliers beyond capacity were "
               "DROPPED — data will be corrupted for those elements. "
               "Increase outlier_capacity to at least %.1f%%.",
               h_outlier_count, actual_pct, capacity_pct, actual_pct * 1.1f);
        h_outlier_count = static_cast<uint32_t>(max_outliers);
    }

    actual_outlier_count_   = h_outlier_count;
    actual_output_sizes_[2] = h_outlier_count * sizeof(TInput);
    actual_output_sizes_[3] = h_outlier_count * sizeof(uint32_t);
    // [0] codes, [1] anchor, [4] outlier_count already correct.
}

// ─── serialization ───────────────────────────────────────────────────────────
template <typename TInput, typename TCode>
size_t GInterpStage<TInput, TCode>::serializeHeader(
    size_t /*output_index*/, uint8_t* buf, size_t max_size) const
{
    if (max_size < sizeof(GInterpConfig)) {
        throw std::runtime_error(
            "GInterpStage::serializeHeader: insufficient buffer");
    }
    GInterpConfig c;
    c.error_bound   = static_cast<float>(computed_abs_eb_);
    c.quant_radius  = static_cast<uint32_t>(config_.quant_radius);
    c.num_elements  = static_cast<uint32_t>(num_elements_);
    c.outlier_count = actual_outlier_count_;
    c.input_type    = inputDataType();
    c.code_type     = codeDataType();
    c.ndim          = static_cast<uint8_t>(ndim());
    c.eb_mode       = static_cast<uint8_t>(config_.eb_mode);
    c.dim_x         = static_cast<uint32_t>(config_.dims[0]);
    c.dim_y         = static_cast<uint32_t>(config_.dims[1]);
    c.dim_z         = static_cast<uint32_t>(config_.dims[2]);
    c.anchor_dim_x  = static_cast<uint32_t>(anchor_dims_[0]);
    c.anchor_dim_y  = static_cast<uint32_t>(anchor_dims_[1]);
    c.anchor_dim_z  = static_cast<uint32_t>(anchor_dims_[2]);
    c.user_eb       = static_cast<float>(config_.error_bound);
    c.value_base    = computed_value_base_;
    std::memcpy(buf, &c, sizeof(c));
    return sizeof(c);
}

template <typename TInput, typename TCode>
void GInterpStage<TInput, TCode>::deserializeHeader(
    const uint8_t* buf, size_t size)
{
    if (size < sizeof(GInterpConfig)) {
        throw std::runtime_error(
            "GInterpStage::deserializeHeader: invalid config size");
    }
    GInterpConfig c;
    std::memcpy(&c, buf, sizeof(c));

    computed_abs_eb_       = static_cast<TInput>(c.error_bound);
    config_.error_bound    = c.user_eb;
    config_.quant_radius   = static_cast<int>(c.quant_radius);
    config_.eb_mode        = static_cast<ErrorBoundMode>(c.eb_mode);
    config_.precomputed_value_base = c.value_base;
    computed_value_base_   = c.value_base;
    num_elements_          = c.num_elements;
    actual_outlier_count_  = c.outlier_count;

    config_.dims[0] = c.dim_x;
    config_.dims[1] = c.dim_y;
    config_.dims[2] = c.dim_z;
    anchor_dims_[0] = c.anchor_dim_x;
    anchor_dims_[1] = c.anchor_dim_y;
    anchor_dims_[2] = c.anchor_dim_z;
}

// ─── Explicit instantiations ─────────────────────────────────────────────────
template class GInterpStage<float, uint8_t>;
template class GInterpStage<float, uint16_t>;
template class GInterpStage<float, uint32_t>;

} // namespace fz
