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
| `setErrorBoundMode(mode)` | ABS / NOA / REL / PREL | **`REL` is the exact pointwise relative bound** (log-space) — this is the only stage that provides it |
| `setQuantRadius(r)` | Quantization radius | Used by ABS/NOA modes |
| `setOutlierCapacity(f)` | Outlier reserve fraction | 0.0-1.0 of element count |
| `setZigzagCodes(enable)` | Zigzag-encode codes | ABS/NOA only; improves compressibility |
| `setOutlierThreshold(t)` | Force outliers | ABS/NOA only; `abs(x) >= t` -> outlier |
| `setInplaceOutliers(enable)` | Embed outliers in codes | ABS/NOA only; see constraints below |
| `setLinearMode(enable)` | Signed codes, no outliers | ABS/NOA only; cuSZp-style; see below |
| `setValueBase(v)` | Precomputed value range | NOA only; optional, see below |
| `setDither(enable)` | Dithered ("_R"-style) reconstruction | ABS/NOA/REL; incompatible with linear/inplace modes; see below |
| `setDitherSeed(seed)` | Seed for the dither offset | Persisted in the header; default 0 |
| `setDitherStrength(s)` | Dither offset amplitude, fraction of abs_eb | Range `(0,1]`; default `1.0`; see below |

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

### Normal mode (3 outputs)

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | `"codes"` | `TCode[n]` | Quantization codes |
| 1 | `"outlier_vals"` | `TInput[k]` | Original values at outlier positions |
| 2 | `"outlier_idxs"` | `uint32_t[k]` | Linear indices of outlier positions |

The outlier **count** is *not* a DAG output port. It lives in a stage-private
4-byte device scratch (allocated in `onFinalize()` via
`pool->allocatePersistentDevice`), is D2H'd in `postStreamSync()`, and is
serialized into the FZM stage header. The inverse path receives it as a
`uint32_t` kernel-launch argument — read from the deserialized header — so
the scatter kernel never has to dereference a device pointer to know its
loop bound. The count is also retrievable post-compress via
`getActualOutputSizesByName().at("outlier_idxs") / sizeof(uint32_t)`,
since `postStreamSync()` trims the indices size to the real count.

Connect downstream stages to `"codes"`:

```cpp
p.connect(next_stage, quant, "codes");
```

### Inplace outlier mode (1 output)

When `setInplaceOutliers(true)` is active, outliers are embedded directly in
the codes array using their raw IEEE-754 bit pattern.  Only the `"codes"` port
exists; the scatter buffers and outlier-count scratch are absent.

### Linear / no-outlier mode (1 output)

When `setLinearMode(true)` is active there is a single `"codes"` port and **no
outlier mechanism at all**.

---

## Linear / no-outlier mode (ABS/NOA only)

`setLinearMode(true)` selects a cuSZp-style quantizer: each value is mapped to
`q = round(x / (2 · eb))` and stored as two's-complement in `TCode`, with **no
radius clamp, no outlier ports, no zigzag**. The forward kernel is a pure
memory-bound map — the only atomic and the only divergent branch of the regular
forward kernel (the outlier path) are removed — so it is strictly faster and is
fully **graph-compatible** in ABS mode (no D2H, no outlier-count readback).

Because there is no outlier fallback, a bin that overflows `TCode` simply wraps;
**size `TCode` wide enough for the data** (use `uint32_t`). The codes are
*declared* by the stage as the **signed** DataType (`UINT16→INT16`,
`UINT32→INT32`) so they connect directly to a downstream `LorenzoStage<intN>`.

Constraints (each throws at the first `compress()` if violated):
- Valid only with `ABS` and `NOA` error-bound modes (not `REL`).
- Mutually exclusive with `setInplaceOutliers(true)` and `setZigzagCodes(true)`.

Intended front-end for the cuSZp-style modular pipeline:

```cpp
auto* quant = p.addStage<QuantizerStage<float, uint32_t>>();
quant->setErrorBound(1e-3f);
quant->setErrorBoundMode(ErrorBoundMode::ABS);
quant->setLinearMode(true);                 // signed INT32 codes, no outliers
auto* lrz = p.addStage<LorenzoStage<int32_t>>();
// lrz->setBlockSize(32);                    // (7.2) block-local 1-D delta, cuSZp uses 32
p.connect(lrz, quant, "codes");
```

---

## Error bound modes

| Mode | Formula | Notes |
|---|---|---|
| `ABS` | `abs(x_orig - x_recon) ≤ eb` | Uniform quantization with step `2 * eb` |
| `NOA` | `abs(error) / value_range ≤ eb` | Scales ABS by the data range |
| `REL` | `abs(error) / abs(x_orig) ≤ eb` | Ratio of error to original value |

**REL mode details:**
<!-- - Encodes magnitude in log2 space (PFPL), then reconstructs `x_hat` from the log bin. -->
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

## Dithered ("_R"-style) reconstruction

By default (`setDither(false)`, matching LC's `QUANT_*_0` modes), every value in
a given quantization bin reconstructs to the same deterministic point (the bin
center for ABS/NOA, the log-bin midpoint for REL). That makes the reconstruction
error *correlated with the signal* — smooth input regions produce smooth,
structured error, which shows up as visible banding/contouring and is worse for
downstream spectral analysis than the same amount of error spread out as noise.

`setDither(true)` (matching LC's `QUANT_*_R` modes) reconstructs each value to a
deterministic pseudo-random point within the bin/error-bound interval instead —
same worst-case error-bound guarantee, decorrelated error. The offset is a pure
function of `(element index, dither_seed)` — no extra bytes are stored per
element; `dither_seed` (persisted in the header) is all decode needs to
reproduce the identical offsets.

```cpp
quant->setDither(true);
quant->setDitherSeed(0x1234ABCDULL);  // any fixed value; default 0
```

**Cost: outlier rate increases substantially.** Since the dither offset spans
the *full* bin width (the same range the deterministic reconstruction could
have landed at LC's error-bound edge), any element whose original quantization
error already used up a meaningful fraction of the bound has a real chance of
being pushed over it once dithered — and any such element is escalated to a
lossless outlier (the encoder verifies the dithered reconstruction against the
true error bound *before* accepting a code; only the resulting compressed
stream — not the input value — is available to decode, so there is no way to
"try again" with a smaller offset). Empirically this lands around **~25% of
elements** for a smooth signal (the sum of two independent roughly-uniform
errors — the original quantization error and the dither offset — exceeding the
bound is a geometrically-derivable ~25% tail probability). Size
`outlier_capacity` accordingly (e.g. `0.3`–`0.5` rather than the `0.05`–`0.1`
typical for non-dithered use) — an undersized capacity silently drops excess
outliers past the buffer, corrupting reconstruction for those elements.

Incompatible with `setLinearMode(true)` and `setInplaceOutliers(true)` (both
throw at `compress()` if combined with `setDither(true)`) — neither has a
per-element outlier-escalation path: linear mode has no outlier ports at all,
and inplace mode's raw-bit escalation mechanism isn't wired up for dithering.

### Dialing back the outlier rate: `setDitherStrength()`

The ~25% outlier rate above comes from dithering across the *full* bin width.
`setDitherStrength(s)` scales the offset amplitude to `s × abs_eb` (or, in REL
mode, `s ×` half a log-bin width) for `s` in `(0, 1]` — the error-bound check
itself is never scaled, only how far the offset can reach. Lower `s` trades
decorrelation strength for fewer outliers, smoothly:

```cpp
quant->setDither(true);
quant->setDitherStrength(0.3f);  // less aggressive than the LC-literal default of 1.0
```

`s = 1.0` (default) exactly matches LC's `_R` definition and the ~25% figure
above. `s = 0.0` disables the offset entirely — reconstruction becomes
bit-identical to `dither=false` (bin center every time), so it's not a useful
end state, just the bottom of the dial. There's no way to make the offset
*data-adaptive* (e.g. only as large as whatever headroom a specific value has
left in its bin) — decode only ever sees the bin index, not where the original
value sat within it, so the amplitude must be this same fixed quantity on both
sides regardless of `s`.

Not currently available on `LorenzoQuantStage` — see that stage's docs for why
(the fused predictor+quantizer reconstructs via a running prefix-sum, so an
individual element's error depends on every prior dithered residual in its
block, not just its own bin).

---

## Value base and CUDA Graph capture

Only NOA needs a data-dependent value base (`max - min`). If `setValueBase()` is
not called, the stage scans the data once to compute it. For CUDA Graph capture,
provide the precomputed value base to avoid a device sync:

```cpp
quant->setValueBase(vmax - vmin);  // NOA only
// after enableGraphMode(true) + finalize()
pipeline.warmup(stream);
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

---

## Acknowledgements

The ABS/NOA/REL quantization scheme, outlier handling, and log-space REL encoding
in `QuantizerStage` follow the **LC/PFPL framework**
(Burtscher et al., Texas State University, BSD-3-Clause).

> Noushin Azami, Alex Fallin, Brandon Burtchell, Andrew Rodriguez,
> Benila Jerald, Yiqian Liu, Anju Mongandampulath Akathoott, and Martin Burtscher.
> *LC framework for synthesizing high-speed parallel lossless and
> error-bounded lossy data compression and decompression algorithms for CPUs and GPUs.*
> https://github.com/burtscher/LC-framework

See `THIRD_PARTY.md` for the full license text.
