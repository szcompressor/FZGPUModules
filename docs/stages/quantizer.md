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

| Parameter | Constraint |
|---|---|
| `TInput` | `float` or `double` |
| `TCode` | Unsigned integer (see available instantiations below) |

## Available instantiations

Only these combinations are compiled and linked:
- `QuantizerStage<float, uint16_t>`
- `QuantizerStage<float, uint32_t>`
- `QuantizerStage<double, uint16_t>`
- `QuantizerStage<double, uint32_t>`

Using any other combination will result in a linker error. Most common: `QuantizerStage<float, uint16_t>`.

---

## Stage settings

| Setting | Purpose | Notes |
|---|---|---|
| `setErrorBound(eb)` | User error bound | Interpreted by `setErrorBoundMode()` |
| `setErrorBoundMode(mode)` | ABS / NOA / REL | REL is exact pointwise relative (log-space) |
| `setQuantRadius(r)` | Quantization radius | Used by ABS/NOA modes |
| `setOutlierCapacity(f)` | Outlier reserve fraction | 0.0-1.0 of element count |
| `setZigzagCodes(enable)` | Zigzag-encode codes | ABS/NOA only; improves compressibility |
| `setOutlierThreshold(t)` | Force outliers | ABS/NOA only; `|x| >= t` -> outlier |
| `setInplaceOutliers(enable)` | Embed outliers in codes | ABS/NOA only; see constraints below |
| `setValueBase(v)` | Precomputed value range | NOA only; optional, see below |

```cpp
quant->setErrorBound(1e-4f);
quant->setErrorBoundMode(ErrorBoundMode::ABS);
quant->setQuantRadius(32768);
quant->setOutlierCapacity(0.05f);      // fraction of N reserved for outliers
quant->setZigzagCodes(true);           // improves compressibility (ABS/NOA only)
quant->setOutlierThreshold(threshold); // |x| >= threshold -> forced outlier
quant->setInplaceOutliers(true);       // ABS/NOA: embed outliers in codes array
quant->setValueBase(range);            // NOA: skip internal data scan
```

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

| Mode | Formula | Notes |
|---|---|---|
| `ABS` | `\|x - x̂\| <= eb` | Uniform quantization with step `2 * eb` |
| `NOA` | `abs_eb = eb × (max - min)` | Scales ABS by the data range |
| `REL` | `\|x - x̂\| / \|x\| <= eb` | Exact per-element, log2-space quantization |

**REL mode details:**
- Encodes magnitude in log2 space (PFPL), then reconstructs `x_hat` from the log bin.
- Zeros, denormals, infinities, and NaNs are stored as outliers to preserve exact values.
- Uses a packed sign + log-bin representation. `uint32_t` is safe for all cases;
  `uint16_t` works for `eb >= 0.01` with `float32` in practice.

---

---

## Inplace outlier constraints (ABS/NOA only)

Both of the following are required when `setInplaceOutliers(true)` is set.
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

### 2. sizeof(TCode) == sizeof(TInput)

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

### Why REL does not support inplace outliers

REL mode packs sign + log-bin into the code word and uses a sentinel value for
outliers. There is no unused range large enough to safely embed raw IEEE-754
bit patterns without collisions, and REL already needs the scatter buffers to
preserve special values (zero, denormals, inf, NaN) exactly. For REL, outliers
must remain in the explicit scatter buffers.

---

## Value base and CUDA Graph capture

Only NOA needs a data-dependent value base (`max - min`). If `setValueBase()` is
not called, the stage scans the data once to compute it. For CUDA Graph capture,
provide the precomputed value base to avoid a device sync:

```cpp
quant->setValueBase(vmax - vmin);  // NOA only
pipeline.captureGraph(stream);
```

ABS and REL modes do not require `setValueBase()`.

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

### Inplace outlier pipeline

```cpp
auto* quant = p.addStage<QuantizerStage<float, uint32_t>>();
quant->setErrorBound(1e-4f);
quant->setZigzagCodes(true);      // required
quant->setInplaceOutliers(true);  // requires sizeof(TCode)==sizeof(TInput)

// Only "codes" port exists; no scatter buffers
p.connect(next, quant, "codes");
p.finalize();
```
