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
| `TInput` | `float` — `double` is not supported. Upstream cuSZ-Hi has `INIT(f8, …)` instantiations commented out as a placeholder in `src/kernel/spline3.cu`, so this matches upstream. |
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
| `setManualAlphaBeta(α, β)` | `double, double` | `0, 0` | Mode 5 only. Either value `0` falls back to the cuSZ-Hi piecewise-linear α schedule / `β = 4.0` default. Both `> 0` is required for graph-safe mode 5. |

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

| Mode | Probe kernel | What it sets | Dim | Graph-safe |
|---|---|---|---|---|
| `0` | none | (off; baseline) | 2-D + 3-D | yes |
| `1` | `c_spline_profiling_data` (2 errors, ~1 ms) | `reverse[0..3]` only (one global bool replicated) | 3-D only | no |
| `2` | `c_spline_profiling_data_2` (6 errors, ~1 ms) | single `use_natural` × `reverse` replicated across all levels; clears `use_md` | 2-D + 3-D | no |
| `3` | `pa_spline_infprecis_data` workflow=true (18 errors 3-D / 27 errors 2-D, ~5–15 ms) | `use_md` / `use_natural` / `reverse` per level (cuSZ-Hi paper mode) | 2-D + 3-D | no |
| `4` | mode 3 + `pa_spline_infprecis_data` workflow=false (+11 errors, ~10–20 ms total) | mode-3 flags **plus** sweeps 11 (alpha, beta) combos and picks the lowest-error | 2-D + 3-D | no |
| `5` | none (manual override) | resolved `alpha` / `beta` (user-supplied or piecewise-linear default); structural flags stay at baseline | 2-D + 3-D | yes |

**Mode 2 — alternate cheap probe.** Runs `c_spline_profiling_data_2`, which
writes 6 errors covering forward/reverse × {cubic, natural} on a tiny
shared-mem sample. Picks one global `use_natural` (sum-based vote) and one
global `reverse` (margin-based vote: 3× in 3-D, 2× in 2-D) and replicates both
across all levels, clearing `use_md`. Cheaper than mode 3 by ~10× and works on
both 2-D and 3-D inputs (unlike modes 1/3/4). Good middle ground when mode 1's
single decision feels too coarse but mode 3's cost is too high.

**Mode 4 — alpha/beta sweep.** Adds a second pass on top of mode 3 that probes
11 (α, β) combinations enumerated by cuSZ-Hi `pre_compute_att` (SPLINE3_AB_ATT):
`α ∈ {1.0, 1.25, 1.5, 1.75, 2.0}`, `β ∈ {2.0, 3.0, 4.0}` (full grid for α ≥ 1.5;
β=2.0 only for the lower α values). The combo with the lowest error becomes
the resolved `α/β`. Use mode 4 for offline workflows where the extra ~10 ms is
worth the CR improvement vs mode 3 (typically 2–5%).

**Mode 5 — manual alpha/beta override.** No profiling kernel runs, so the path
is graph-safe and works on both 2-D and 3-D inputs. Set via
`setManualAlphaBeta(alpha, beta)`. Passing 0 for either field falls back to the
cuSZ-Hi piecewise-linear `alpha` schedule (keyed on `rel_eb`) or `beta = 4.0`.
The structural flags (`use_md` / `use_natural` / `reverse`) stay at baseline.

Common patterns:

```cpp
// Recommended for offline CR-critical workflows:
g->setAutoTuning(4);   // structural probe + alpha/beta sweep

// User has prior knowledge of optimal params — skip profiling entirely:
g->setAutoTuning(5);
g->setManualAlphaBeta(1.5, 3.0);

// alpha is always interpolated from rel_eb (see cuSZ-Hi spline3.cu:80-103)
// in modes 1/3/4. For ABS mode, the stage scans the data range once to derive
// rel_eb; REL/NOA reuses the user-supplied eb directly.
g->setErrorBoundMode(ErrorBoundMode::REL);
g->setErrorBound(1e-3f);
g->setAutoTuning(3);
```

**Profiling modes (1/2/3/4) are incompatible with CUDA graph capture.** Modes
0 and 5 are graph-safe when combined with a manual `setQuantRadius(...)` and
either ABS mode or a `setValueBase(...)` (REL/NOA). For mode 5, also
`setManualAlphaBeta(α>0, β>0)` to skip the `rel_eb`-based α picker. Each
probe ends with a D2H of the error array and a `cudaStreamSynchronize`, which
would error out inside a captured region. **Mode 5 is graph-safe** since it
never launches a profile kernel. `isGraphCompatible()` returns `false` in
either case in the MVP — it'll be relaxed for mode 5 + manual radius once
end-to-end graph capture is tested.

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
- **Profiling auto-tune mode 1 is 3-D only.** On 2-D inputs it logs a warning
  and falls through to the deterministic baseline. Modes 2, 3, 4, 5 all work
  on both 2-D and 3-D inputs.
- `isGraphCompatible()` is **mode-aware**: returns `true` for modes 0 and 5
  when `quant_radius > 0`, the error-bound mode is ABS or a `value_base` is
  pre-set, and (for mode 5) `manual_alpha > 0`. Modes 1/2/3/4 each end with a
  D2H + `cudaStreamSynchronize` of the error array, so those remain
  graph-incompatible.
- **Fixed `LEVEL = 4` and anchor tile size `16` on every axis** for both
  3-D (16×16×16) and 2-D (16×16) — these stay aligned with cuSZ-Hi's hardcoded
  choices for the 3-D path. The 2-D path uses the same `LEVEL=4 / 16×16` tile
  rather than upstream cuSZ-Hi's `LEVEL=6 / 8×8 × 4`-numAnchorBlocks default
  (which has a known `c_gather_anchor` off-by-numAnchorBlock bug). Varying
  LEVEL would explode templated kernel instantiations and require encoding
  the choice in the FZM header. No planned work here unless a workload
  demonstrates a CR gap.
- **2-D auto-tune mode 3** uses a locally patched offset in
  `pa_spline_infprecis_data`: upstream cuSZ-Hi writes the SPLINE_DIM==2
  level==0 atomic at `errors+15+BIY` (BIY=5..10 → slots 20..25), which
  collides with the level==1 BIY=4 write at slot 20. Our copy writes at
  `errors+16+BIY` to match the host-side analysis loop's expected layout
  (level_id=0 at errors[21..26]). See the adapter-changes block at the top of
  `modules/fused/ginterp/ginterp_md.inl`.

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
auto_tuning  = 0            # 0=off, 1=cheap, 2=alt-cheap, 3=full, 4=full+a/b sweep, 5=manual
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
