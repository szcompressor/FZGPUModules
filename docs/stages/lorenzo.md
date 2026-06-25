# LorenzoStage {#stage_lorenzo}

**Header:** `modules/predictors/lorenzo/lorenzo_stage.h`  
**Class:** `fz::LorenzoStage<T>`  
**Category:** Predictor (lossless)

---

## What it does

Plain integer Lorenzo predictor.  Lossless, size-preserving.

- **Forward (compression):** compute per-element deltas from spatial neighbors.
- **Inverse (decompression):** prefix-sum to reconstruct original values.

Supports 1-D, 2-D, and 3-D layouts.  Typically placed **after** a `QuantizerStage`
in cuSZp-style pipelines (float → quant → Lorenzo → bitpack), where it operates on
the quantization codes rather than raw floating-point data.

---

## Template parameter

| Parameter | Constraint |
|---|---|
| `T` | Signed integer (see available instantiations below) |

## Available instantiations

Only these types are compiled and linked:
- `LorenzoStage<int8_t>`
- `LorenzoStage<int16_t>`
- `LorenzoStage<int32_t>`
- `LorenzoStage<int64_t>`

Using any other type will result in a linker error. Common choice: `LorenzoStage<int32_t>` (to match quantizer code width).

---

## Stage settings

| Setting | Purpose | Notes |
|---|---|---|
| `setDims(x[,y,z])` | Spatial dimensions | Or via `Pipeline::setDims()`; selects 1-/2-/3-D delta |
| `setBlockSize(n)` | 1-D block-local reset period | `0` = default; `n>0` = cuSZp-style, see below |

By default (`block_size == 0`) the predictor uses the N-D inclusion-exclusion delta
selected by `dims_` (and the 1-D path already resets per launch block of 256).

`setBlockSize(n)` with `n > 0` forces the **1-D** path over the flattened array and
restarts the prediction chain (`prev = 0`) every `n` elements, independent of the
launch configuration and of `dims_`. This is the cuSZp predictor (it uses `n = 32`).
`n` must be in `[1, 1024]` (the inverse scans one segment per CUDA block of `n`
threads). Block mode is graph-compatible. The `block_size` is serialized in the FZM
stage header (legacy 16-byte headers default it to 0).

```cpp
auto* lrz = p.addStage<LorenzoStage<int32_t>>();
lrz->setBlockSize(32);   // cuSZp block-local 1-D delta
```

---

## Ports

Single input → single output; element type and size are unchanged.

| Direction | Port | Type |
|---|---|---|
| Input | `"output"` (default) | `T[n]` |
| Output | `"output"` | `T[n]` |

Connection from a quantizer upstream uses the `"codes"` port of the quantizer:

```cpp
p.connect(lrz, quant, "codes");
```

---

## Dimension setup — critical ordering rule

Same rule as `LorenzoQuantStage`: call `p.setDims()` **before** `addStage()`, or
call `stage->setDims()` directly after adding.

```cpp
p.setDims(nx, ny, nz);
auto* lrz = p.addStage<LorenzoStage<int32_t>>();
```

---

## Typical pipeline (cuSZp-style)

The cuSZp front-end pairs the linear quantizer (signed codes, no outliers) with a
block-local Lorenzo predictor:

```cpp
auto* quant = p.addStage<QuantizerStage<float, uint32_t>>();
quant->setErrorBound(1e-3f);
quant->setErrorBoundMode(ErrorBoundMode::ABS);
quant->setLinearMode(true);          // signed INT32 codes, no outliers

auto* lrz = p.addStage<LorenzoStage<int32_t>>();
lrz->setBlockSize(32);               // block-local 1-D delta (cuSZp)

p.connect(lrz, quant, "codes");
// downstream coder (AdaptiveBitpackStage, forthcoming) connects to lrz
p.finalize();
```
