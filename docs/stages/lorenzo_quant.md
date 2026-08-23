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
| `setErrorBoundMode(mode)` | ABS / NOA / PREL | `REL` warns and maps to `PREL` (see below) |
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

The outlier **count** is *not* a DAG output port. It lives in a stage-private
4-byte device scratch (allocated in `onFinalize()` via
`pool->allocatePersistentDevice`), is D2H'd in `postStreamSync()`, and is
serialized into the FZM stage header. The inverse path receives it as a
`uint32_t` kernel-launch argument — read from the deserialized header — so
the scatter kernel never has to dereference a device pointer to know its
loop bound. The count is also retrievable post-compress via
`getActualOutputSizesByName().at("outlier_indices") / sizeof(uint32_t)`,
since `postStreamSync()` trims the indices size to the real count.

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
| `PREL` | `abs_eb = eb × max(abs(data))` | Pseudo-relative; can be precomputed via `setValueBase()` |
| `REL` | — | **Not supported here.** Deprecated alias for `PREL`; warns and maps. |

All three supported modes resolve to a **single absolute bound** before
quantizing. That is inherent to the stage: it quantizes prediction *residuals*
against one global tolerance, and reconstruction is a running prefix-sum over
dequantized residuals, so a per-element varying bound cannot be threaded through
it. For an exact pointwise relative bound use `QuantizerStage` with
`ErrorBoundMode::REL`.

### Why PREL is not REL

`PREL` sets `abs_eb = eb × max(abs(x))` and then behaves exactly like `ABS`. It
therefore bounds

```
|error| / max(|x|)  <=  eb          (what PREL gives you)
|error| / |x|       <=  eb          (what REL means)
```

These agree only for elements at the peak magnitude. The effective per-element
relative error degrades in direct proportion to how far below peak an element
sits — roughly 10× looser per decade — and elements at or near zero are
unbounded in relative terms.

Measured on `CLDHGH.f32` (3600×1800) at `eb = 1e-3`, via
`examples/eb_mode_analysis.cpp`:

| abs(x) / peak | count | worst abs(e)/abs(x) | vs. requested `eb` |
|---|---|---|---|
| `[1e-1, 1e-0)` | 5,417,230 | 9.90e-03 | 9.9× |
| `[1e-2, 1e-1)` | 1,050,426 | 9.09e-02 | 90.9× |
| `[1e-3, 1e-2)` | 12,344 | 3.33e-01 | 333.3× |

74% of elements exceed the requested relative bound. `QuantizerStage` REL on the
same data stays at 1.0× in every decade with zero violations.

**When PREL is nevertheless the right choice:** when what you actually care
about is fidelity relative to the *field's* scale rather than each element's —
which is the common case for PSNR-driven work, and is why the SZ family reports
against the data range. `PREL` and `NOA` differ only in the scan statistic
(`max|x|` vs `max−min`) and coincide within 2× for data straddling zero.

Run `examples/eb_mode_analysis.cpp` on your own data to see the profile before
committing to a mode.

---

## No dithered ("_R"-style) reconstruction

Unlike `QuantizerStage`, `LorenzoQuantStage` does not support `setDither()`.
This stage quantizes prediction *residuals*, and reconstruction is a running
prefix-sum over dequantized residuals within each block — an individual
element's reconstructed value depends on the accumulated sum of every prior
residual in its block, not just its own bin. Verifying (and, when needed,
escalating to a lossless outlier) a dithered residual against the true
*per-element* error bound would require accounting for how that perturbation
propagates through every subsequent prefix-sum step in the block, which is a
materially harder problem than `QuantizerStage`'s independent per-element
verification. Use `QuantizerStage` (direct-value quantization) if dithering is
needed.

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
// after enableGraphMode(true) + finalize()
pipeline.warmup(stream);
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

---

## Acknowledgements

The fused predictor+quantizer kernels and multi-output design in
`LorenzoQuantStage` follow the **cuSZ** Lorenzo implementation
(`lrz_c.cuhip.inl`, `lrz_x.cuhip.inl`) by the cuSZ team (BSD-3-Clause).

> cuSZ team (UChicago Argonne National Laboratory, Indiana University, and others).
> *pSZ/cuSZ: A GPU-Based Error-Bounded Lossy Compressor for Scientific Data.*
> https://github.com/szcompressor/cuSZ

See `THIRD_PARTY.md` for the full license text.

The optional per-tile mean-centering component follows the FSZ design:

> Jiajun Huang. *FSZ: Breaking the Prediction-Throughput Trade-off in GPU Lossy
> Compression.* SC '26, arXiv:2607.15413.

Centering was implemented from the paper before the FSZ source release; no FSZ
source was copied. The fused cuSZ attribution above continues to apply to the
predictor/quantizer kernels themselves.
