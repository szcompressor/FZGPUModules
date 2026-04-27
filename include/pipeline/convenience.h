#pragma once

/**
 * @file convenience.h
 * @brief Optional stage-specific convenience builders layered on top of Pipeline.
 *
 * Not included by `fzgpumodules.h` — opt in explicitly:
 * @code
 *   #include "pipeline/convenience.h"
 *   pipeline.setDims(nx, ny);
 *   auto* lrz = addLorenzoQuant(pipeline, 1e-4f);
 * @endcode
 *
 * Note: LorenzoStage does not need a convenience wrapper — it defaults to 1-D
 * when setDims() is not called, matching the behaviour of all other stages.
 */

#include "pipeline/compressor.h"
#include "predictors/lorenzo_quant/lorenzo_quant.h"

namespace fz {

/**
 * Add a Lorenzo predictor stage with automatic dimensionality selection.
 *
 * Chooses the correct 1-D / 2-D / 3-D variant based on the pipeline's current
 * dims (set via Pipeline::setDims()).  Defaults to 1-D if setDims() was never
 * called.
 *
 * @param pipeline         The pipeline to add the stage to.
 * @param error_bound      Absolute pointwise error tolerance.
 * @param quant_radius     Quantization radius (default 32768 for uint16_t).
 * @param outlier_capacity Fraction of data to reserve for outliers (default 0.2).
 * @return Pointer to the created stage (owned by the pipeline).
 */
template <typename TInput = float, typename TCode = uint16_t>
inline LorenzoQuantStage<TInput, TCode>* addLorenzoQuant(
    Pipeline& pipeline,
    float     error_bound,
    int       quant_radius     = 32768,
    float     outlier_capacity = 0.2f
) {
    typename LorenzoQuantStage<TInput, TCode>::Config cfg;
    cfg.error_bound      = error_bound;
    cfg.quant_radius     = quant_radius;
    cfg.outlier_capacity = outlier_capacity;
    cfg.dims             = pipeline.getDims();
    return pipeline.addStage<LorenzoQuantStage<TInput, TCode>>(cfg);
}

} // namespace fz
