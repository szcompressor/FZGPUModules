/**
 * @file cdf97_outlier_correct_stage.cu
 * @brief Cdf97Reconstructor::applyInverseTransform() + explicit instantiation
 *        of OutlierCorrectStage<Cdf97Reconstructor>. All generic logic lives
 *        in `outlier_correct_stage_impl.cuh`; this file supplies only the
 *        CDF97-specific step.
 */

#include "coders/cdf97_outlier_correct/cdf97_outlier_correct_stage.h"
#include "coders/outlier_correct/outlier_correct_stage_impl.cuh"
#include "transforms/cdf97/cdf97_kernels.cuh"
#include "stage/stage_registry.h"
#include <stdexcept>

namespace fz {

void Cdf97Reconstructor::applyInverseTransform(float* d_coeffs_inout, int nx, int ny, int nz, cudaStream_t stream) {
    if (nz > 1)
        throw std::runtime_error("Cdf97Reconstructor: 3-D not supported");
    cdf97::dwt2d<float>(d_coeffs_inout, nx, ny, /*inverse=*/true, stream);
}

template class OutlierCorrectStage<Cdf97Reconstructor>;

} // namespace fz

// ── FZM-header reconstruction (self-registered; see stage_registry.h) ─────────
FZ_REGISTER_SIMPLE_STAGE(fz::StageType::CDF97_OUTLIER_CORRECT, fz::Cdf97OutlierCorrectStage);
