# Fused stages {#stage_fused}

| Stage | Description |
|---|---|
| \subpage stage_lorenzo_quant | Fused float predictor + quantizer (1-D/2-D/3-D) |
| \subpage stage_adaptive_lorenzo | Per-tile adaptive multi-order Lorenzo + centering (FSZ prediction stage) |
| \subpage stage_ginterp | Multi-level spline interpolation predictor + quantizer (3-D, cuSZ-Hi port) |
| \subpage stage_bitplane_rze | Fused bitplane transpose + zero-group RZE — FZ-GPU's lossless encoder |
| \subpage stage_szx | SZx ultrafast EB compressor — per-block constant/non-constant classification + fixed-length residuals (whole compressor) |
| \ref stage_diff "DifferenceStage (negabinary-fused)" | First-order difference with inline negabinary encoding (`DifferenceStage<T, TOut>` where `TOut != T`) |

> **Note:** `DifferenceStage` with a `TOut != T` template argument fuses a negabinary
> encoding step into the differencing kernel (equivalent to `DifferenceStage<T>` followed
> by `NegabinaryStage`, but in a single pass).
> This stage lives in `modules/predictors/diff/`, not in `modules/fused/`, but is listed
> here so all fused-operation options appear in one place.

## Experimental / reference compressors (not supported modules)

Some whole-compressor implementations are kept only as GPU points of comparison and
are **not** part of the composable module set: they are absent from
`<fzgpumodules.h>`, from the stage catalog, and from the automatic-fusion planner.
They live under `experimental/reference_compressors/`.

| Reference compressor | Supported modular equivalent |
|---|---|
| `SZpStage<T>` (`experimental/reference_compressors/szp/`) — SZp / fZ-light: quantize + 1-D Lorenzo delta + fixed-length bitpack. See \ref experimental_szp. | `examples/presets/szp_composed.toml` — `Quantizer(linear) → Lorenzo(block_size=128) → AdaptiveBitpack(block_size=128)` |

Its `StageType::SZP` FZM factory stays linked so archives written before the
quarantine still decode, and `type = "SZp"` still loads from legacy TOML configs.
