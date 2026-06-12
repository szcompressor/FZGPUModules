# GInterpStage {#stage_ginterp}

**Header:** `modules/fused/ginterp/ginterp_stage.h`
**Class:** `fz::GInterpStage<TInput, TCode>`
**Category:** Fused predictor + quantizer (lossy)

**Common instantiation:**
```cpp
auto* g = p.addStage<fz::GInterpStage<float, uint16_t>>();
g->setErrorBound(1e-2f);
// No setQuantRadius() — let the stage auto-tune to the data range.
```

---

## What it does

Multi-level spline-interpolation predictor with error-bounded quantization,
ported from the [cuSZ-Hi](https://github.com/shixun404/cuSZ-Hi) compressor.

- **Forward (compression):** four-level interpolation pyramid; each level predicts
  finer samples from already-decoded coarser anchors. Residuals are quantized to
  `TCode` codes; anything that overflows the `[-radius, radius)` window is routed
  to a separate outlier triplet.
- **Inverse (decompression):** scatter outliers back into place, then run the
  inverse pyramid (anchors → level 4 → level 3 → level 2 → level 1).

G-Interp typically yields a higher compression ratio than `LorenzoQuantStage` on
**smooth scientific data** (climate fields, simulation snapshots, etc.) at the
same error bound. It is the prediction stage used in the cuSZ-Hi compressor.

### Why this stage is fused (no standalone predictor)

`LorenzoStage` ships in two flavours — the lossless plain predictor and the
fused `LorenzoQuantStage` — because the Lorenzo prediction at each cell reads
only the *original input values* of its neighbours. The quantizer is a clean
post-processing step that can be applied (or not) independently.

G-Interp does not work that way. The forward pass is an interpolation
**pyramid**: level 4 anchors are exact, then level 3 samples are predicted
from level-4 anchors, level 2 from the *quantized-and-reconstructed* level-3
samples, level 1 from the reconstructed level-2 samples, and so on. Each
finer level depends on the **lossy reconstruction** of every coarser level —
exactly the reconstruction the decoder will see — because that's the only
way the encoder can guarantee its error bound matches what the decoder
produces. The decoder mirrors this: it must walk the same tree using the
same quantizer to recover each level before moving to the next.

If you removed the quantizer, there'd be no value to feed into the next
level's prediction, so the kernel can't run. Splitting into a "pure G-Interp
predictor" + "separate quantizer" stage would either (a) require the
predictor to re-implement the quantizer internally just to feed itself,
making the split cosmetic, or (b) reduce the algorithm to a single-level
predictor, which throws away the multi-level CR gain that motivates using
G-Interp in the first place.

So unlike Lorenzo, the prediction-quantization coupling here is intrinsic
to the algorithm — G-Interp ships only as the fused stage.

---

## Template parameters

| Parameter | Constraint |
|---|---|
| `TInput` | `float` (MVP — `double` not yet wired) |
| `TCode`  | `uint8_t`, `uint16_t`, or `uint32_t` |

## Available instantiations

Only these types are compiled and linked:
- `GInterpStage<float, uint8_t>`
- `GInterpStage<float, uint16_t>` — most common
- `GInterpStage<float, uint32_t>`

---

## Stage settings

| Setting | Type | Default | Purpose |
|---|---|---|---|
| `setErrorBound(eb)` | `float` | `1e-3` | Target absolute bound (see "Error bound" below) |
| `setErrorBoundMode(mode)` | `ErrorBoundMode` | `ABS` | `ABS`, `REL`, or `NOA` (same semantics as `QuantizerStage`) |
| `setQuantRadius(r)` | `int` | `0` (auto) | Quantization radius — see "Radius auto-tune" below |
| `setOutlierCapacity(c)` | `float` | `0.10` | Fraction of `N` reserved for outliers (0.10 ⇒ 10%) |
| `setValueBase(v)` | `float` | `0` | Pre-computed `value_range` (NOA) or `max(abs(data))` (REL); set before graph capture |
| `setAutoTuning(mode)` | `uint8_t` | `0` | Enable `INTERPOLATION_PARAMS` auto-tuning — see "Auto-tuning" below |

### Radius auto-tune

`setQuantRadius(0)` (the default) means **auto-tune**. On first `execute()`, the
stage scans `min`/`max` of the input and picks the smallest radius such that
every residual fits in `[-radius, radius)` at the worst-case multi-level error.
The result is clamped to the `TCode` bit-width's maximum (`127` for `uint8_t`,
`32767` for `uint16_t` / `uint32_t`).

If the upstream mode is `REL` or `NOA`, the same scan that already happens for
the error-bound conversion is reused — auto-tune adds no extra D2H.

Set the radius explicitly to **any positive value** to skip the scan:

```cpp
g->setQuantRadius(512);   // climate-style — route outliers to the triplet
```

The manual path is required for:
- **CUDA Graph capture** (the auto scan does a `cudaStreamSynchronize` + D2H).
- Datasets where the user *wants* extremes pushed into the outlier triplet for
  separate downstream handling (e.g. climate fields with rare spikes).

The auto-tuned radius is cached in the serialized header on first compress, so
the decompressor never needs to know which mode was used.

### Auto-tuning

cuSZ-Hi's compression ratio depends heavily on the per-level interpolation
choices encoded by `INTERPOLATION_PARAMS` (alpha, beta, and the three
`use_md` / `use_natural` / `reverse` boolean arrays). By default the stage
runs the deterministic baseline (`alpha=1.75`, `beta=4.0`,
`use_md={true,true,false,false,false,false}`, `use_natural`/`reverse` all
false), which is the safe choice when the data distribution is unknown.

Enable auto-tuning to pick those flags per-dataset:

```cpp
g->setAutoTuning(3);   // recommended: full structural profiling
```

| Mode | Probe kernel | What it sets |
|---|---|---|
| `0` | none | (off; baseline) |
| `1` | `c_spline_profiling_data` (2 errors) | `reverse[0..3]` only (one global bool replicated) |
| `3` | `pa_spline_infprecis_data` (18 errors) | `use_md` / `use_natural` / `reverse` per level (matches the cuSZ-Hi paper) |

Modes `2` (2-D-only structural probe), `4` (full + alpha/beta sweep), and
`5+` (manual alpha/beta override) from cuSZ-Hi are not yet wired —
straightforward follow-up scope.

Common patterns:

```cpp
// alpha is always interpolated from rel_eb (see cuSZ-Hi spline3.cu:80-103).
// For ABS mode, the stage scans the data range once to derive rel_eb;
// REL/NOA reuses the user-supplied eb directly.
g->setErrorBoundMode(ErrorBoundMode::REL);
g->setErrorBound(1e-3f);
g->setAutoTuning(3);
```

**Auto-tuning is incompatible with CUDA graph capture.** Each profiling
kernel ends with a D2H of the error array and a `cudaStreamSynchronize`,
which would error out inside a captured region. `isGraphCompatible()` is
`false` whether auto-tuning is on or off in the MVP.

The resolved `INTERPOLATION_PARAMS` are embedded in the FZM stage header at
compress time, so the decompressor reuses them verbatim — there is no
re-tuning on the inverse path.

### Error bound and limitations

The error bound `eb` is a **target**, not a hard guarantee. The multi-level
interpolation tree predicts finer-level values from already-lossy coarser-level
reconstructions, so prediction errors accumulate across the four levels. In
practice the maximum element-wise error is:

- typically `≤ 1.1 × eb` on smooth data;
- up to `~2 × eb` on data with many outliers (large spikes that the spline
  can't predict — these are stored exactly via the outlier triplet, but their
  neighbours still see compounded interpolation error).

Other limitations:

- **2-D and 3-D only.** `setDims()` throws for 1-D input. 2-D inputs set
  `dims[2] = 1` and pick the 2-D launcher automatically.
- Best results when each `dim` is a multiple of 16 (the anchor tile size,
  used by both the 3-D and 2-D paths). Ragged dims still work but edge
  voxels see slightly worse prediction.
- **Auto-tuning is 3-D only.** `setAutoTuning(1)` / `(3)` wrap the cuSZ-Hi
  3-D profiling kernels. On 2-D inputs they log a warning and fall through
  to the deterministic baseline. cuSZ-Hi `auto_tuning_mode == 2` is the
  2-D-targeted probe and is a follow-up.
- cuSZ-Hi `auto_tuning` modes `2`/`4`/`5+` are **not yet ported**. Modes `1`
  (cheap reverse-only profile) and `3` (full structural — the cuSZ-Hi paper
  mode) are available via `setAutoTuning()` on 3-D data; see the
  "Auto-tuning" section above. Mode `4` (alpha/beta sweep on top of mode 3)
  and the 2-D-targeted mode `2` are straightforward follow-ups.
- `isGraphCompatible()` returns `false` in the MVP. The forward path does no
  D2H during `execute()`, but the auto-tune scan and `postStreamSync()` for the
  outlier count do. End-to-end graph compatibility will be enabled after the
  manual-radius graph capture path is tested.

---

## Ports

### Forward

| Index | Port | Type | Size |
|---|---|---|---|
| 0 | `codes` | `TCode` | `N` |
| 1 | `anchor` | `TInput` | `~N / 4096` |
| 2 | `outlier_vals` | `TInput` | up to `outlier_capacity * N` |
| 3 | `outlier_idxs` | `uint32_t` | up to `outlier_capacity * N` |

The outlier **count** is *not* a DAG output port. It lives in a stage-private
4-byte device scratch (allocated in `onFinalize()` via
`pool->allocatePersistentDevice`), is D2H'd in `postStreamSync()`, and is
serialized into the FZM stage header. The inverse path receives it as a
`uint32_t` kernel-launch argument — read from the deserialized header — so
the scatter kernel never has to dereference a device pointer to know its
loop bound. The count is also retrievable post-compress via
`getActualOutputSizesByName().at("outlier_idxs") / sizeof(uint32_t)`,
since `postStreamSync()` trims the indices size to the real count.

Connect downstream stages to the `codes` port:

```cpp
p.connect(next, g, "codes");
```

### Inverse

Four inputs (in the order above) → one output (reconstructed `TInput[N]`).

---

## Typical pipeline

```cpp
Pipeline p(in_bytes, MemoryStrategy::PREALLOCATE);
p.setDims(nx, ny, nz);   // 2-D: pass nz=1. 1-D is not supported.

auto* g = p.addStage<GInterpStage<float, uint16_t>>();
g->setErrorBound(1e-2f);
g->setErrorBoundMode(ErrorBoundMode::ABS);
// (No setQuantRadius — auto-tune.)

// Optional: feed codes into a coder.
// auto* huf = p.addStage<HuffmanStage<uint16_t>>();
// p.connect(huf, g, "codes");

p.finalize();
p.compress(d_in, in_bytes, &d_out, &out_sz, stream);
```

---

## TOML configuration

```toml
[[stage]]
name         = "ginterp"
type         = "GInterp"
input_type   = "float32"
code_type    = "uint16"
error_bound  = 1e-2
error_bound_mode = "ABS"    # "ABS", "REL", or "NOA"
quant_radius = 0            # 0 = auto-tune (default); positive = manual override
outlier_capacity = 0.10
auto_tuning  = 0            # 0=off, 1=cheap, 3=full structural
```

---

## Serialized header

64-byte `GInterpConfig` — fits comfortably in `FZM_STAGE_CONFIG_SIZE` (128 B).
Stores `error_bound`, `quant_radius` (the resolved value, never 0 by the time
it lands here), `dim_x/y/z`, anchor extents, `eb_mode`, `input_type` /
`code_type`, the user-specified `user_eb`, and the resolved `value_base` for
NOA/REL.

---

## Acknowledgements

`GInterpStage` ports the spline interpolation kernels from the
[cuSZ-Hi](https://github.com/shixun404/cuSZ-Hi) compressor (Indiana University,
Argonne National Laboratory), BSD-3-Clause. The host-side wrapper, memory-pool
integration, outlier-fusion contract, and radius auto-tune are FZGPUModules
code. See `THIRD_PARTY.md` for the full license text.

> Liu, S., Tao, D., et al. *cuSZ-Hi: High-Ratio GPU-Based Error-Bounded Lossy
> Compression for Scientific Data.* https://github.com/shixun404/cuSZ-Hi
