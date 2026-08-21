# LorenzoStage {#stage_lorenzo}

**Header:** `modules/predictors/lorenzo/lorenzo_stage.h`  
**Class:** `fz::LorenzoStage<T>`  
**Category:** Predictor (lossless)

---

## What it does

Plain integer Lorenzo predictor.  Lossless, size-preserving.

- **Forward (compression):** compute per-element deltas from spatial neighbors.
- **Inverse (decompression):** prefix-sum to reconstruct original values.

Supports 1-D, 2-D, and 3-D layouts.  Typically placed **after** a `QuantizerStage`
in cuSZp-style pipelines (float → quant → Lorenzo → bitpack), where it operates on
the quantization codes rather than raw floating-point data.

\image html lorenzo.svg "Compression-side Lorenzo prediction in one and two dimensions."

---

## Template parameter

| Parameter | Constraint |
|---|---|
| `T` | Signed integer (see available instantiations below) |

## Available instantiations

Only these types are compiled and linked:
- `LorenzoStage<int8_t>`
- `LorenzoStage<int16_t>`
- `LorenzoStage<int32_t>`
- `LorenzoStage<int64_t>`

Using any other type will result in a linker error. Common choice: `LorenzoStage<int32_t>` (to match quantizer code width).

---

## Stage settings

| Setting | Purpose | Notes |
|---|---|---|
| `setDims(x[,y,z])` | Spatial dimensions | Or via `Pipeline::setDims()`; selects 1-/2-/3-D delta |
| `setBlockSize(n)` | 1-D block-local reset period | `0` = default; `n>0` = cuSZp-style, see below |
| `setCentering(b)` | Per-block mean centering | Requires `block_size > 0`; adds a `"means"` port, see below |
| `setOrder(k)` | Prediction order, 1 or 2 | Requires `block_size > 0`; `2` = FSZ's LZ2, see below |

By default (`block_size == 0`) the predictor uses the N-D inclusion-exclusion delta
selected by `dims_` (and the 1-D path already resets per launch block of 256).

`setBlockSize(n)` with `n > 0` forces the **1-D** path over the flattened array and
restarts the prediction chain (`prev = 0`) every `n` elements, independent of the
launch configuration and of `dims_`. This is the cuSZp predictor (it uses `n = 32`).
`n` must be in `[1, 1024]`. Block mode is graph-compatible.

The inverse assigns one CTA per reset segment with several elements per thread,
so the CTA width no longer tracks `n` — a 1024-element segment uses 256 threads
of 4 elements rather than 1024 threads, and the scan costs 2 barriers per pass
regardless of segment length. Reset periods that are not a multiple of 32 use a
barrier-based fallback that handles any width. Longer segments therefore no
longer cost decompression throughput: on CESM `Z3` at REL 1E-3, `block_size`
1024 decompresses at 120 GB/s against 122 at 512 and 101 at 256. The `block_size` is serialized in the FZM
stage header (legacy 16-byte headers default it to 0).

```cpp
auto* lrz = p.addStage<LorenzoStage<int32_t>>();
lrz->setBlockSize(32);   // cuSZp block-local 1-D delta
```

### Per-block mean centering

`setCentering(true)` subtracts each block's integer mean `mu` from the values
before predicting. Because the difference of a constant is zero, `delta(q - mu)`
equals `delta(q)` for every element that has a predecessor — so centering changes
**only the first residual of each block**, the chain seed that would otherwise be
emitted as a raw value. On a field with a large constant offset (temperature in
Kelvin, pressure in hPa) that seed is the largest magnitude in the block and sets
a downstream fixed-rate coder's bit width for all `block_size` elements.

Centering requires block mode and emits a second port, `"means"` (one `T` per
block), which the inverse takes as its second input. It is lossless: the
reconstruction is bit-identical to the uncentered path.

```cpp
// Centering must be a constructor argument: it adds a port, and addStage()
// captures the port count at add-time.
auto* lrz = p.addStage<LorenzoStage<int32_t>>(/*block_size=*/512, /*centering=*/true);
p.connect(coder, lrz);                       // residuals
// "means" left unconnected becomes a pipeline output stored in the .fzm
```

TOML: `block_size = 512`, `centering = true`.

**When it pays.** The gain scales with the number of chain restarts, since that
is how many raw seeds there are to fix, while the cost is a fixed
`sizeof(T)` bytes per block. Measured geometric-mean CR gain over 8 SDRBench
fields x 3 error bounds:

| `block_size` | 32 | 256 | 512 | 1024 |
|---|---|---|---|---|
| CR gain | 2.01x | 1.29x | 1.17x | 1.09x |

**When it loses.** Centering here is unconditional, and on sparse data where most
blocks already encode to zero bytes the per-block `mu` is pure overhead — HURR
`QRAIN` drops to 0.30x at `block_size = 32`. Enable it per dataset rather than by
default, or measure both ways.

---

## Ports

Single input → single output; element type and size are unchanged.

| Direction | Port | Type |
|---|---|---|
| Input | `"output"` (default) | `T[n]` |
| Output | `"output"` | `T[n]` |

Connection from a quantizer upstream uses the `"codes"` port of the quantizer:

```cpp
p.connect(lrz, quant, "codes");
```

---

## Dimension setup — critical ordering rule

Same rule as `LorenzoQuantStage`: call `p.setDims()` **before** `addStage()`, or
call `stage->setDims()` directly after adding.

```cpp
p.setDims(nx, ny, nz);
auto* lrz = p.addStage<LorenzoStage<int32_t>>();
```

---

## Typical pipeline (cuSZp-style)

The cuSZp front-end pairs the linear quantizer (signed codes, no outliers) with a
block-local Lorenzo predictor:

```cpp
auto* quant = p.addStage<QuantizerStage<float, uint32_t>>();
quant->setErrorBound(1e-3f);
quant->setErrorBoundMode(ErrorBoundMode::ABS);
quant->setLinearMode(true);          // signed INT32 codes, no outliers

auto* lrz = p.addStage<LorenzoStage<int32_t>>();
lrz->setBlockSize(32);               // block-local 1-D delta (cuSZp)

p.connect(lrz, quant, "codes");
// downstream coder (AdaptiveBitpackStage, forthcoming) connects to lrz
p.finalize();
```


### Second-order prediction (LZ2)

`setOrder(2)` predicts each element from the *trend* of the two before it rather
than from the previous value alone. Block-local LZ2 is the first difference
applied twice under the same zero-padding convention, which gives FSZ's

    e_0 = q_0,  e_1 = q_1 - 2*q_0,  e_i = q_i - 2*q_{i-1} + q_{i-2}   (i >= 2)

so a linear ramp collapses to all zeros past the two seeds, where first order
would leave a constant non-zero stride in every residual. Both difference passes
run in shared memory, so it stays a single trip through global memory.

Requires block mode; composes with `setCentering()`. TOML: `order = 2`.

**When it pays.** Second order costs one extra raw seed per block and doubles
the residual on piecewise-constant data, so it is not a free upgrade. Measured
geometric-mean CR against first order over 8 SDRBench fields:

| error bound | REL 1E-2 | REL 1E-3 | REL 1E-4 |
|---|---|---|---|
| LZ2 vs LZ1 | 0.97x | 0.97x | 1.03x |

At loose bounds most blocks already encode to nothing whatever the prediction
order, so the extra seed is pure cost; the win only appears once the bound is
tight enough that curvature drives the residual. At REL 1E-4, LZ2 combined with
centering is the best of the four variants on 6 of 8 fields (CESM `FICE` 1.75x,
`Z3` 1.52x over plain LZ1).

Because the right choice flips with both the field and the error bound, prefer
\ref stage_adaptive_lorenzo "AdaptiveLorenzoStage", which makes it per tile.

## Acknowledgements

The block-local first-order Lorenzo predictor is an independent implementation
of a standard finite-difference predictor; its `setBlockSize` mode composes the
same block-reset step used by cuSZp. The cross-block prediction state,
second-order option, and mean-centering variants follow the FSZ design:

> Jiajun Huang. *FSZ: Breaking the Prediction-Throughput Trade-off in GPU Lossy
> Compression.* SC '26, arXiv:2607.15413.

These FSZ-derived modes were implemented from the paper before the FSZ source
release; no FSZ source was copied. See `THIRD_PARTY.md`.
