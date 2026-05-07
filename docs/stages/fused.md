# Fused stages {#stage_fused}

| Stage | Description |
|---|---|
| \subpage stage_lorenzo_quant | Fused float predictor + quantizer (1-D/2-D/3-D) |
| \ref stage_diff "DifferenceStage (negabinary-fused)" | First-order difference with inline negabinary encoding (`DifferenceStage<T, TOut>` where `TOut != T`) |

> **Note:** `DifferenceStage` with a `TOut != T` template argument fuses a negabinary
> encoding step into the differencing kernel (equivalent to `DifferenceStage<T>` followed
> by `NegabinaryStage`, but in a single pass).
> This stage lives in `modules/predictors/diff/`, not in `modules/fused/`, but is listed
> here so all fused-operation options appear in one place.
