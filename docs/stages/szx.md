# SZxStage {#stage_szx}

**Header:** `modules/fused/szx/szx_stage.h`
**Class:** `fz::SZxStage<TData>` — `TData` is `float` or `double`
**Category:** Fused (lossy, whole compressor)

**Common instantiation:**
```cpp
auto* szx = p.addStage<fz::SZxStage<float>>();
szx->setBlockSize(128);                 // elements per block (SZx default)
szx->setErrorBound(1e-3);
szx->setErrorMode(fz::SZxErrorMode::ABS);
```

---

## What it does

SZx is an **ultrafast error-bounded lossy compressor**: it consumes raw floats
and emits a self-describing byte archive in a single fused stage. Unlike the
cuSZ-style chain (`Lorenzo → Quantizer → coder`), SZx has **no prediction stage
and no entropy coder** — that is what makes it fast.

- **Forward:** `float[]`/`double[]` → opaque SZx archive (`uint8_t[]`)
- **Inverse:** archive → reconstructed values, error-bounded per element

Per block of `block_size` elements the forward pass:

1. scans block min/max;
2. **classifies** the block as *constant* when `max − min ≤ 2·eb` (the whole
   block is representable by one value within the bound) or *non-constant*
   otherwise — a per-block type code;
3. **constant** blocks emit only a block reference value (the midpoint),
   broadcast on decode;
4. **non-constant** blocks subtract the reference, quantize the residuals to
   fixed-length integers at the block's required bit width, and bit-pack them.

## Why it is a stage, not a composition

Every other whole-compressor behaviour in this library is expressible as a DAG
of smaller stages. SZx is the exception: the **constant-block escape** is a
data-dependent branch *inside* the per-block loop — a block collapses to one
value or expands to fixed-length residuals depending on its own range — and no
composition of the existing block-local stages reproduces that branch. The
classification is also SZx's whole point: on smooth or piecewise-flat fields it
is where the compression comes from. See
\ref experimental_szp "SZpStage" for the sibling that has no constant-block path (and is
therefore composable).

## Archive layout

The 40-byte `SZxConfig` (dtype, block size, resolved absolute bound, value base)
travels in the FZM stage-config slot, not the output buffer. The output buffer
is:

```
[ meta region : 2 bytes/block = {type, width} ] [ payload region ]
  payload per block, at its scanned byte offset:
    constant     : reference value (sizeof(TData) bytes)
    non-constant : reference value + block_len residual codes, width bits each
```

Per-block byte offsets are a device-wide exclusive scan of the per-block cost;
the decoder recomputes them from the meta region, so no offset table is stored.
The layout **round-trips against itself but is not byte-compatible with the
reference SZx container** — this is an algorithm-faithful reimplementation, not a
format port.

## Error bounds

| `error_bound_mode` | meaning | graph-capturable forward |
|---|---|---|
| `ABS` (default) | abs(x − x̂) ≤ eb per element | yes |
| `NOA` | value-range relative: `abs_eb = eb · (max − min)` | no |

The **ROI branch invariant** holds in both: reconstruction error is bounded per
element by the resolved absolute bound. `NOA` first reduces the data range on the
device and reads it back on the host inside `execute()`, so the forward path is
not CUDA-graph-capturable under `NOA` (`isGraphCompatible()` reports this);
`ABS` defers its only host read (the compressed-size readback) to
`postStreamSync()` and stays capturable. SZx is lossy: a resolved `abs_eb ≤ 0`
throws.

## Stage settings

| setter | TOML key | default | meaning |
|---|---|---|---|
| `setBlockSize(n)` | `block_size` | 128 | elements per block; `n ∈ [1, 4096]` |
| `setErrorBound(eb)` | `error_bound` | 1e-3 | bound value (interpreted per mode) |
| `setErrorMode(m)` | `error_bound_mode` | `ABS` | `ABS` or `NOA` |
| — | `data_type` | `float32` | `float32` or `float64` |

`getConstantBlockFraction()` reports the measured fraction of constant blocks on
the last forward encode (also emitted through `getRunNotes()`), a cheap
compressibility probe.

## TOML configuration

```toml
[[stage]]
name = "szx"
type = "SZx"
data_type = "float32"
block_size = 128
error_bound = 1e-3
error_bound_mode = "ABS"
```

SZx is a whole compressor, so it is normally the only stage in the pipeline
(`examples/presets/szx.toml`).

## Measured behaviour

On SDRBench CLDHGH (`3600×1800` f32, `eb = 1e-3` ABS) SZx reaches **5.38×** vs.
the composable \ref experimental_szp "SZp" configuration's 3.98× at the same bound — the
gap is entirely the constant-block classification paying off on the smooth
field. The reconstruction max-abs-error equals the bound in both.

## Prior work

The block-classification / fixed-length residual scheme is SZx (Xiaodong Yu,
Sheng Di, et al.). This stage is an algorithm-faithful reimplementation — no SZx
source is vendored — with the FZM archive layout, CUB offset scan, and
MemoryPool scaffolding being FZGPUModules code. The upstream implementation is
at https://github.com/szcompressor/SZx under the Argonne OPEN SOURCE LICENSE
SF-16-105 (four-condition BSD style); because no source was copied, that license
is recorded for provenance rather than applied to this implementation. See
`THIRD_PARTY.md`.
