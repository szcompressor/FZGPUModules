# SZpStage (experimental reference compressor) {#experimental_szp}

> **NOT a supported composable module.** `SZpStage` is a quarantined GPU
> reference implementation kept only as a point of comparison. It is **absent**
> from `<fzgpumodules.h>`, from the stage catalog, and from the automatic-fusion
> planner, and lives under `experimental/reference_compressors/szp/`. Its
> `StageType::SZP = 37` FZM factory stays linked so pre-existing archives still
> decode, and `type = "SZp"` still loads from legacy TOML configs, but neither is
> a supported surface.
>
> **The supported SZp-inspired modular composition is
> `examples/presets/szp_composed.toml`:**
> `Quantizer(linear) → Lorenzo(block_size=128) → AdaptiveBitpack(block_size=128)`.
> It matches the upstream algorithm's high-level stages but does not reproduce
> its exact quantization, partition seeding, predictor boundaries, or container.

**Header:** `experimental/reference_compressors/szp/szp_stage.h` (direct-include only)
**Class:** `fz::SZpStage<TData>` — `TData` is `float` or `double`
**Category:** Experimental / reference compressor (lossy, whole compressor)

**Common instantiation:**
<!-- doc-check: skip — quarantined stage, intentionally absent from <fzgpumodules.h>; direct-include only -->
```cpp
#include "reference_compressors/szp/szp_stage.h"   // not in <fzgpumodules.h>

auto* szp = p.addStage<fz::SZpStage<float>>();
szp->setBlockSize(128);                 // elements per block (FZGM default)
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

SZp and the FZGM composition share the same broad structure:
`quantize → 1-D Lorenzo differences → sign bits + per-block fixed-width packing`.
The upstream CPU/OpenMP implementation truncates `x/eb`, stores an initial seed
per OpenMP partition, and carries its predictor across the smaller packing-block
boundaries inside that partition. The FZGM composition instead uses the generic
`round(x/(2·eb))` quantizer and resets Lorenzo at every 128-value block. The
quarantined `SZpStage` mirrors the FZGM convention and validates the composition;
neither is a behavioral or byte-compatible reproduction of upstream SZp.

Unlike \ref stage_szx "SZx", the quarantined implementation has **no
constant-block escape** — every element pays the block's bit width even when its
delta is zero. Its high-level operations can be expressed as the supported
composition above; the monolithic stage itself remains unsupported.

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

> **Ratio note:** in the quarantined FZGM stage and supported composition, the
> block-reset delta makes each block's first residual the absolute quantized level
> at the block head, which sets the block width and caps the ratio on ramps and
> high-offset data. This is an FZGM composition detail, not a claim about the
> upstream CPU/OpenMP partitioning; \ref stage_szx "SZx" instead uses a
> reference-value scheme for flat regions.

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

`examples/presets/szp_composed.toml` (the supported modular analogue) is
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
license. This stage is a GPU adaptation of its predict-quantize-pack structure;
no upstream source was copied. The FZM archive layout, quantization convention, CUB
offset scan, and MemoryPool scaffolding are FZGPUModules code. See
`THIRD_PARTY.md`.
