# SZpStage (experimental reference compressor) {#experimental_szp}

> **NOT a supported composable module.** `SZpStage` is a quarantined GPU
> reference implementation kept only as a point of comparison. It is **absent**
> from `<fzgpumodules.h>`, from the stage catalog, and from the automatic-fusion
> planner, and lives under `experimental/reference_compressors/szp/`. Its
> `StageType::SZP = 37` FZM factory stays linked so pre-existing archives still
> decode, and `type = "SZp"` still loads from legacy TOML configs, but neither is
> a supported surface.
>
> **The supported expression of the SZp algorithm is
> `examples/presets/szp_composed.toml`:**
> `Quantizer(linear) → Lorenzo(block_size=128) → AdaptiveBitpack(block_size=128)`.
> That chain reproduces SZp behaviour with zero new code and byte-identical
> compressed sizes (verified on CLDHGH).

**Header:** `experimental/reference_compressors/szp/szp_stage.h` (direct-include only)
**Class:** `fz::SZpStage<TData>` — `TData` is `float` or `double`
**Category:** Experimental / reference compressor (lossy, whole compressor)

**Common instantiation:**
<!-- doc-check: skip — quarantined stage, intentionally absent from <fzgpumodules.h>; direct-include only -->
```cpp
#include "reference_compressors/szp/szp_stage.h"   // not in <fzgpumodules.h>

auto* szp = p.addStage<fz::SZpStage<float>>();
szp->setBlockSize(128);                 // elements per block (SZp default)
szp->setErrorBound(1e-3);
szp->setErrorMode(fz::SZpErrorMode::ABS);
```

---

## What it does

SZp (also published as **fZ-light**, SC '24) is an extreme-fast error-bounded
lossy compressor. Like \ref stage_szx "SZx" it is a whole compressor in one
fused stage — raw floats in, self-describing byte archive out, no entropy coder.

- **Forward:** `float[]`/`double[]` → opaque SZp archive (`uint8_t[]`)
- **Inverse:** archive → reconstructed values, error-bounded per element

Per block of `block_size` elements:

1. **quantize** `q_i = round(x_i / (2·eb))` (linear, signed);
2. **predict** `d_i = q_i − q_{i−1}` with `d_0 = q_0` — a 1-D Lorenzo delta that
   resets at each block boundary;
3. **pack** `zigzag(d_i)` at the block's fixed bit width, one width byte per
   block.

## Relationship to the composable chain

SZp's inner loop is exactly the FZGM chain
`Quantizer(linear, ABS) → Lorenzo(block reset) → AdaptiveBitpack` —
`AdaptiveBitpack`'s plain mode *is* per-block fixed-length residual packing. The
native `SZpStage` exists for (a) a single-launch whole compressor and (b) a
named, format-stable SZp target; the composed preset
`examples/presets/szp_composed.toml` reproduces the same behaviour with **zero
new code**, and the two produce **byte-identical compressed sizes** (verified on
CLDHGH), which is how the native stage is validated against the composition.

Unlike \ref stage_szx "SZx", SZp has **no constant-block escape** — every element
pays the block's bit width even when its delta is zero — so it is fully
composable and is provided as a stage only for convenience and throughput.

## Archive layout

The 40-byte `SZpConfig` travels in the FZM stage-config slot. The output buffer
is:

```
[ meta region : 1 byte/block = width ] [ payload : packed zigzag deltas per block ]
```

Per-block byte offsets are a device-wide exclusive scan of the per-block cost,
recomputed from the width meta on decode (no offset table stored). The layout
round-trips against itself but is **not byte-compatible with the reference SZp
container**. This stage is a GPU reimplementation of the upstream CPU/OpenMP
compressor's forward/inverse.

## Error bounds

| `error_bound_mode` | meaning | graph-capturable forward |
|---|---|---|
| `ABS` (default) | abs(x − x̂) ≤ eb per element | yes |
| `NOA` | value-range relative: `abs_eb = eb · (max − min)` | no |

Reconstruction prefix-sums the per-block deltas to recover `q_i`, then scales by
`2·eb`, so the per-element bound is `eb` (or the resolved `abs_eb` under `NOA`).
`NOA` needs a device range reduce + host read in `execute()` and is therefore not
graph-capturable; `ABS` defers its size readback to `postStreamSync()` and stays
capturable. SZp is lossy: a resolved `abs_eb ≤ 0` throws.

> **Ratio note:** the block-reset delta makes each block's first residual the
> absolute quantized level at the block head, which sets the block width and caps
> the ratio on ramps and high-offset data. This is inherent to SZp's block-local
> 1-D prediction, not a defect — and it is exactly what \ref stage_szx "SZx"'s
> reference-value scheme avoids on flat regions.

## Stage settings

| setter | TOML key | default | meaning |
|---|---|---|---|
| `setBlockSize(n)` | `block_size` | 128 | elements per block; `n ∈ [1, 4096]` |
| `setErrorBound(eb)` | `error_bound` | 1e-3 | bound value (interpreted per mode) |
| `setErrorMode(m)` | `error_bound_mode` | `ABS` | `ABS` or `NOA` |
| — | `data_type` | `float32` | `float32` or `float64` |

## TOML configuration

```toml
[[stage]]
name = "szp"
type = "SZp"
data_type = "float32"
block_size = 128
error_bound = 1e-3
error_bound_mode = "ABS"
```

`examples/presets/szp_composed.toml` (the supported zero-new-code equivalent) is
shipped under `examples/presets/`. The legacy native preset is retained only at
`experimental/reference_compressors/szp/szp.toml`.

## hZCCL (not implemented)

SZp is the container hZCCL uses to run **collective communication in the
compressed domain** (add/reduce on compressed buffers without full
decompression). That capability is **out of scope for this stage** — it is a
separate `HomomorphicOp` interface, not a `Stage` (a `Stage` is a single-buffer
transform). See the future-work scoping note
`docs/szp_homomorphic_collectives.md`.

## Prior work

SZp / fZ-light: Jiajun Huang, Sheng Di, et al., SC '24. The upstream CPU/OpenMP
reference is available at https://github.com/szcompressor/SZp under the MIT
license. This stage is a GPU reimplementation of its predict-quantize-pack
forward/inverse; no upstream source was copied. The FZM archive layout, CUB
offset scan, and MemoryPool scaffolding are FZGPUModules code. See
`THIRD_PARTY.md`.
