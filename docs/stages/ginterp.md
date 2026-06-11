# GInterpStage {#stage_ginterp}

**Header:** `modules/predictors/ginterp/ginterp_stage.h`
**Class:** `fz::GInterpStage<TInput, TCode>`
**Category:** Predictor + quantizer (lossy)

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
| `setValueBase(v)` | `float` | `0` | Pre-computed `value_range` (NOA) or `max(|data|)` (REL); set before graph capture |

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

### Error bound and limitations

The error bound `eb` is a **target**, not a hard guarantee. The multi-level
interpolation tree predicts finer-level values from already-lossy coarser-level
reconstructions, so prediction errors accumulate across the four levels. In
practice the maximum element-wise error is:

- typically `≤ 1.1 × eb` on smooth data;
- up to `~2 × eb` on data with many outliers (large spikes that the spline
  can't predict — these are stored exactly via the outlier triplet, but their
  neighbours still see compounded interpolation error).

Other MVP limitations:

- **3-D only.** `setDims()` throws for 1-D or 2-D inputs.
- Best results when each `dim` is a multiple of 16 (the anchor tile size).
  Ragged dims still work but edge voxels see slightly worse prediction.
- cuSZ-Hi's `INTERPOLATION_PARAMS` auto-tuning is **not yet ported**. This MVP
  uses the upstream deterministic baseline (`alpha=1.75`, `beta=4.0`,
  `use_md={true,true,false,false,false,false}`). Real-world compression ratio
  may be 10–30 % off the cuSZ-Hi paper until phase 2 lands.
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
| 4 | `outlier_count` | `uint32_t` | 4 bytes |

Connect downstream stages to the `codes` port:

```cpp
p.connect(next, g, "codes");
```

### Inverse

Five inputs (in the order above) → one output (reconstructed `TInput[N]`).

---

## Typical pipeline

```cpp
Pipeline p(in_bytes, MemoryStrategy::PREALLOCATE);
p.setDims(nx, ny, nz);   // 3-D only — call before addStage()

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
eb_mode      = "ABS"        # "ABS", "REL", or "NOA"
quant_radius = 0            # 0 = auto-tune (default); positive = manual override
outlier_capacity = 0.10
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
