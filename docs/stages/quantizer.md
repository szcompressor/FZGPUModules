# QuantizerStage {#stage_quantizer}

**Header:** `modules/quantizers/quantizer/quantizer.h`  
**Class:** `fz::QuantizerStage<TInput, TCode>`  
**Category:** Quantizer (lossy)

---

## What it does

Quantizes floating-point input values directly (not prediction residuals).
Values that fall outside the representable range are stored losslessly as
outliers in separate scatter buffers, unless inplace mode is active.

Three error-bound modes are supported: ABS, NOA, and REL.

---

## Template parameters

| Parameter | Constraint | Typical value |
|---|---|---|
| `TInput` | `float` or `double` | `float` |
| `TCode` | `uint8_t`, `uint16_t`, `uint32_t` | `uint16_t` or `uint32_t` |

---

## Output ports (compression)

### Normal mode (4 outputs)

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | `"codes"` | `TCode[n]` | Quantization codes |
| 1 | `"outlier_vals"` | `TInput[k]` | Original values at outlier positions |
| 2 | `"outlier_idxs"` | `uint32_t[k]` | Linear indices of outlier positions |
| 3 | `"outlier_count"` | `uint32_t` | Number of outliers (scalar) |

Connect downstream stages to `"codes"`:

```cpp
p.connect(next_stage, quant, "codes");
```

### Inplace outlier mode (1 output)

When `setInplaceOutliers(true)` is active, outliers are embedded directly in
the codes array using their raw IEEE-754 bit pattern.  Only the `"codes"` port
exists; the three outlier scatter ports are absent.

---

## Error bound modes

| Mode | Formula | `TCode` requirement |
|---|---|---|
| `ABS` | `\|x - x̂\| <= eb` | Any |
| `NOA` | `abs_eb = eb × (max - min)` | Any |
| `REL` | `\|x - x̂\| / \|x\| <= eb` (exact per-element) | 4-byte recommended (`uint32_t`) |

**REL mode notes:**
- Uses log₂-space quantization; zeros, denormals, inf, and NaN become outliers.
- A `uint16_t` code type is sufficient for `eb >= 0.01` with `float32` inputs
  in practice (max `|log_bin|` ≈ 4460 << 16383), but `uint32_t` is safe for
  all cases.
- Inplace outlier mode (`setInplaceOutliers`) is **not** compatible with REL
  mode and will be silently ignored.

---

## Key setters

```cpp
quant->setErrorBound(1e-4f);
quant->setErrorBoundMode(ErrorBoundMode::ABS);
quant->setQuantRadius(32768);
quant->setOutlierCapacity(0.05f);      // fraction of N reserved for outliers
quant->setZigzagCodes(true);           // improves compressibility (ABS/NOA only)
quant->setOutlierThreshold(threshold); // |x| >= threshold → forced outlier (default: ∞)
quant->setInplaceOutliers(true);       // ABS/NOA: embed outliers in codes array
quant->setValueBase(range_or_maxabs);  // NOA/REL: skip internal data scan
```

---

## Inplace outlier mode constraints

Both of the following are **required** when `setInplaceOutliers(true)` is set.
Violations throw at runtime during the first `compress()` call.

### 1. Zigzag encoding must be enabled

```cpp
quant->setZigzagCodes(true);    // required
quant->setInplaceOutliers(true);
```

**Why:** the inverse kernel distinguishes valid codes from embedded outlier floats
via the sentinel `(code >> 1) >= quant_radius`.  With zigzag encoding (TCMS), valid
codes are in `[0, 2 × quant_radius)`.  Normal float bit patterns are always
`>= 0x00800000`, which exceeds `2 × quant_radius` for any practical radius
(<= 2²²), making the sentinel check unambiguous.  Without zigzag, signed
two's-complement codes overlap with float bit patterns and the sentinel fails.

### 2. `sizeof(TCode) == sizeof(TInput)`

```cpp
// Correct: float (4B) paired with uint32_t (4B)
auto* quant = p.addStage<QuantizerStage<float, uint32_t>>();
quant->setInplaceOutliers(true);

// Wrong: uint16_t is 2B, float is 4B — throws at runtime
auto* quant = p.addStage<QuantizerStage<float, uint16_t>>();
quant->setInplaceOutliers(true);  // runtime error
```

**Why:** the inplace kernel stores outlier raw bits with `__builtin_memcpy(&raw, &x, sizeof(TCode))`.
If the sizes differ the copy is truncated or out-of-bounds.

---

## CUDA Graph capture and NOA/REL modes

NOA and REL modes perform an internal data scan that requires a device sync.
This is illegal during graph capture.  Provide a precomputed value with
`setValueBase()` before `captureGraph()`:

```cpp
// NOA: pass value_range = max - min
quant->setValueBase(vmax - vmin);
// REL: pass max absolute value
quant->setValueBase(std::max(std::abs(vmin), std::abs(vmax)));

pipeline.captureGraph(stream);
```

ABS mode needs no `setValueBase()` call.

---

## Typical pipelines

### PFPL-style (standalone quantizer)

```cpp
auto* quant = p.addStage<QuantizerStage<float, uint32_t>>();
auto* diff  = p.addStage<DifferenceStage<int32_t, uint32_t>>();
auto* bshuf = p.addStage<BitshuffleStage>();
auto* rze   = p.addStage<RZEStage>();

quant->setErrorBound(1e-4f);
quant->setZigzagCodes(true);

p.connect(diff,  quant, "codes");
p.connect(bshuf, diff);
p.connect(rze,   bshuf);
p.finalize();
```

### Inplace outlier mode

```cpp
auto* quant = p.addStage<QuantizerStage<float, uint32_t>>();
quant->setErrorBound(1e-4f);
quant->setZigzagCodes(true);      // required
quant->setInplaceOutliers(true);  // requires sizeof(TCode)==sizeof(TInput)

// Only "codes" port exists; no scatter buffers
p.connect(next, quant, "codes");
p.finalize();
```
