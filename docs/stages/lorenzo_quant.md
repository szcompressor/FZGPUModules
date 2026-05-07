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

| Parameter | Constraint |
|---|---|
| `TInput` | `float` or `double` |
| `TCode` | Unsigned integer (see available instantiations below) |

## Available instantiations

Only these combinations are compiled and linked:
- `LorenzoQuantStage<float, uint8_t>`
- `LorenzoQuantStage<float, uint16_t>`
- `LorenzoQuantStage<double, uint16_t>`
- `LorenzoQuantStage<double, uint32_t>`

Using any other combination will result in a linker error. Most common: `LorenzoQuantStage<float, uint16_t>` (cuSZ-style pipelines).

---

## Stage settings

| Setting | Purpose | Notes |
|---|---|---|
| `setErrorBound(eb)` | User error bound | Interpreted by `setErrorBoundMode()` |
| `setErrorBoundMode(mode)` | ABS / NOA / REL | REL is a global approximation (see below) |
| `setQuantRadius(r)` | Quantization radius | Must fit in `TCode` range |
| `setOutlierCapacity(f)` | Outlier reserve fraction | 0.0-1.0x of element count |
| `setZigzagCodes(enable)` | Zigzag-encode codes | Can improve compressibility |
| `setValueBase(v)` | Precomputed scale | NOA: `(max - min)`, REL: `abs(max)`; optional |

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
| `ABS` | `abs(error) <= eb` | Default |
| `NOA` | `abs_eb = eb × (max - min)` | Uses value range; can be precomputed via `setValueBase()` |
| `REL` | `abs_eb = eb × max(abs(data))` | Global approximation (not exact per-element) |

REL is supported, but because it uses a single global scale (`max(abs(x))`), small
values can exceed the per-element relative bound. For exact pointwise REL bounds,
use `QuantizerStage` with `ErrorBoundMode::REL`.

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

## Value base and CUDA Graph capture

NOA and REL modes need a data-dependent scale:

- NOA: `value_base = max - min`
- REL: `value_base = max(|x|)`

If `setValueBase()` is not called, the stage scans the data to compute the
value base internally. For CUDA Graph capture, you must provide the value base
up front to avoid a device sync and D2H read.

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
