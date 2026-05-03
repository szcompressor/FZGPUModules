# LorenzoQuantStage {#stage_lorenzo_quant}

**Header:** `modules/fused/lorenzo_quant/lorenzo_quant.h`  
**Class:** `fz::LorenzoQuantStage<TInput, TCode>`  
**Category:** Fused predictor + quantizer

---

## What it does

Computes a Lorenzo prediction (each element minus its spatial neighbor(s)), then
immediately quantizes the prediction error into integer codes.  The fused kernel
avoids writing the raw residuals to device memory.

Supports 1-D, 2-D, and 3-D data.  Dimensionality is controlled by `setDims()` and
must be set **before** `pipeline.addStage()` so the pipeline can push the correct
dims at add-time.

Outliers (errors that fall outside `[-quant_radius, quant_radius)`) are scattered
to separate `outlier_errors` and `outlier_indices` buffers.

---

## Template parameters

| Parameter | Constraint | Typical value |
|---|---|---|
| `TInput` | `float` or `double` | `float` |
| `TCode` | `uint8_t`, `uint16_t`, `uint32_t` | `uint16_t` |

Common instantiation: `LorenzoQuantStage<float, uint16_t>`.

---

## Output ports (compression)

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | `"codes"` | `TCode[n]` | Quantized prediction errors |
| 1 | `"outlier_errors"` | `TInput[k]` | Original values at outlier positions |
| 2 | `"outlier_indices"` | `uint32_t[k]` | Linear indices of outlier positions |
| 3 | `"outlier_count"` | `uint32_t` | Number of outliers (scalar) |

Connect downstream stages to the `"codes"` port:

```cpp
p.connect(next_stage, lorenzo, "codes");
```

---

## Error bound modes

| Mode | Interpretation | Note |
|---|---|---|
| `ABS` | `\|error\| <= eb` | Default |
| `NOA` | `abs_eb = eb × (max - min)` | Scans data once; use `setValueBase()` to skip |
| `REL` | `abs_eb = eb × max(\|data\|)` | Global approximation — not exact per-element |

For exact pointwise relative bounds, use `QuantizerStage` with `ErrorBoundMode::REL`
instead.

---

## Key setters

```cpp
lorenzo->setErrorBound(1e-4f);
lorenzo->setErrorBoundMode(ErrorBoundMode::ABS);
lorenzo->setQuantRadius(32768);          // must fit in TCode range
lorenzo->setOutlierCapacity(0.10f);      // fraction of N reserved for outliers
lorenzo->setZigzagCodes(true);           // zigzag-encode codes for better compressibility
lorenzo->setValueBase(vmax - vmin);      // NOA: skip internal data scan
lorenzo->setValueBase(max_abs);          // REL: skip internal data scan
```

---

## Dimension setup — critical ordering rule

`addStage()` pushes the pipeline's current dims into the stage immediately.
`finalize()` pushes them again as a safety net.  If dims are set after `addStage()`,
call `stage->setDims()` directly.

```cpp
// Correct
p.setDims(nx, ny);
auto* lrz = p.addStage<LorenzoQuantStage<float, uint16_t>>();

// Also correct (set after addStage)
auto* lrz = p.addStage<LorenzoQuantStage<float, uint16_t>>();
lrz->setDims(nx, ny);   // call directly on the stage

// Wrong — dims may not propagate in time
auto* lrz = p.addStage<LorenzoQuantStage<float, uint16_t>>();
p.setDims(nx, ny);      // too late; addStage already ran
```

---

## CUDA Graph capture and NOA/REL modes

NOA and REL modes perform an internal `cudaStreamSynchronize` + D2H copy to
determine the value range.  This is illegal during graph capture.

**Fix:** call `setValueBase()` with a host-computed value before `captureGraph()`.

```cpp
// NOA
float value_base = vmax - vmin;
// REL
float value_base = std::max(std::abs(vmin), std::abs(vmax));

lorenzo->setValueBase(value_base);
pipeline.captureGraph(stream);
```

ABS mode needs no `setValueBase()` call.

---

## Typical pipeline

```cpp
p.setDims(nx, ny);
auto* lrz   = p.addStage<LorenzoQuantStage<float, uint16_t>>();
auto* bshuf = p.addStage<BitshuffleStage>();
auto* rze   = p.addStage<RZEStage>();

lrz->setErrorBound(1e-4f);
lrz->setZigzagCodes(true);
bshuf->setElementWidth(sizeof(uint16_t));

p.connect(bshuf, lrz, "codes");
p.connect(rze,   bshuf);
p.finalize();
```
