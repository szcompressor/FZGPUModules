# Stage Reference {#stages_overview}

Detailed documentation for each stage in FZGPUModules, including API constraints,
behavioral rules, and usage notes that go beyond what the class-level Doxygen docs
can capture inline.

## Predictors / Quantizers

| Stage | Description |
|---|---|
| \subpage stage_lorenzo_quant | Fused float predictor + quantizer (cuSZ-style, 1-D/2-D/3-D) |
| \subpage stage_lorenzo | Plain integer delta predictor / prefix-sum (lossless) |
| \subpage stage_quantizer | Standalone direct-value quantizer (ABS / REL / NOA) |

## Coders

| Stage | Description |
|---|---|
| \subpage stage_rle | Run-length encoding |
| \subpage stage_rze | Recursive zero-byte elimination |
| \subpage stage_bitpack | Dense bit-packing of fixed-width integers |

## Transforms / Shufflers

| Stage | Description |
|---|---|
| \subpage stage_diff | First-order difference / cumulative-sum coding |
| \subpage stage_bitshuffle | GPU bit-matrix transpose (bit-plane separation) |
