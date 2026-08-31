#pragma once

/**
 * @file cdf97_outlier_correct_stage.h
 * @brief Cdf97OutlierCorrectStage — the CDF 9/7 instantiation of the generic
 *        `OutlierCorrectStage<Reconstructor>` (see `modules/coders/
 *        outlier_correct/outlier_correct_stage.h` for the mechanism, the
 *        Reconstructor policy contract, and why this needs the raw field
 *        bound directly rather than through a fan-out node).
 *
 * `Cdf97Reconstructor` is the entire CDF97-specific surface: one function
 * that runs the existing, tested `cdf97::dwt2d<float>` kernel in place.
 * Everything else (diffing, sparse pack/apply, config, ports, serialization)
 * lives once in the generic template.
 *
 * `applyInverseTransform()`'s body is defined in the .cu file, not here --
 * it calls into `transforms/cdf97/cdf97_kernels.cuh`, which is raw CUDA
 * device code and must stay out of this header (included broadly, including
 * by plain .cpp translation units via fzgpumodules.h).
 */

#include "coders/outlier_correct/outlier_correct_stage.h"
#include <string>

namespace fz {

struct Cdf97Reconstructor {
    static constexpr StageType kStageType = StageType::CDF97_OUTLIER_CORRECT;
    static std::string name() { return "Cdf97"; }

    /// In-place: d_coeffs_inout holds dequantized CDF 9/7 coefficients on
    /// entry, the trial reconstruction on return. 2-D only (nz must be 1) --
    /// matches the only shape `sperr_gpu.toml` uses; a 3-D reconstructor
    /// would call `cdf97::dwt3d_*` instead, as its own separate policy.
    static void applyInverseTransform(float* d_coeffs_inout, int nx, int ny, int nz, cudaStream_t stream);
};

using Cdf97OutlierCorrectStage = OutlierCorrectStage<Cdf97Reconstructor>;

extern template class OutlierCorrectStage<Cdf97Reconstructor>;

} // namespace fz
