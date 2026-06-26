# Predictor stages {#stage_predictors}

| Stage | Description |
|---|---|
| \subpage stage_lorenzo | Plain integer delta predictor / prefix-sum (lossless) |
| \subpage stage_tiled_lorenzo | Dimension-aware (tiled separable) Lorenzo predictor (lossless, 2D/3D, cuSZp3 delta) |
| \subpage stage_diff | First-order difference / cumulative-sum coding |

> **Note:** `GInterpStage` lives in [Fused stages](\ref stage_fused) because the
> spline predictor and quantizer are intrinsically fused — see the GInterp doc
> for why a standalone predictor is not feasible.
