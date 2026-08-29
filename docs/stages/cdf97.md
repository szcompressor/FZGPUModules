# Cdf97Stage {#stage_cdf97}

**Header:** `modules/transforms/cdf97/cdf97_stage.h`
**Class:** `fz::Cdf97Stage<TInput>`
**Category:** Transform (lossless, invertible, size-preserving)

---

## What it does

The CDF 9/7 biorthogonal wavelet transform — the DWT front-half of the SPERR
compressor (CDF 9/7 DWT -> SPECK bit-plane coding). A dimension-aware,
size-preserving, lossless, invertible float -> float basis change: it replaces
a field with its multi-level wavelet coefficients (same element type, same
element count), separable per axis with dyadic (Mallat) recursion (3-D also
supports wavelet-packet decomposition for anisotropic volumes, selected
automatically). See [THIRD_PARTY.md](../../THIRD_PARTY.md) for the port
relationship to NCAR/SPERR (lifting constants, boundary handling, level-count
rule, and the 3-D dyadic/wavelet-packet selection rule are direct ports; the
GPU kernels are FZGPUModules' own).

Feeds a quantizer, then (optionally) `Speck2DStage` — see
[stage_speck2d](speck2d.md) and `examples/presets/sperr_gpu.toml` for the full
SPERR-style pipeline. Quantizing this stage's coefficients directly does
**not** guarantee a pointwise bound on the reconstructed field — CDF 9/7's
synthesis-filter gain differs by level, so a uniform coefficient threshold
doesn't translate to a uniform reconstructed-domain error. For an actual
guarantee, pair with [`Cdf97OutlierCorrectStage`](outlier_correct.md) — see
that page and `examples/presets/sperr_gpu.toml` for the complete,
bound-guaranteed pipeline.

---

## Template parameter

| Parameter | Constraint |
|---|---|
| `TInput` | `float` or `double` |

`double` reproduces SPERR's coefficients **bit-for-bit** (`sperr::CDF97` runs
entirely in double). `float` is a faster, deliberately non-bit-exact variant
(lifting constants are derived in double, but the arithmetic itself is
`float`).

---

## Current limitation

Each transform line is processed in shared memory, so the largest dimension
must satisfy `maxdim * sizeof(TInput) <= 48 KiB` (6144 elements for `double`,
12288 for `float`). A larger extent throws at `execute()`, rather than
producing wrong output; the long-line (global-memory) fallback is future work.

---

## Typical pipeline

```cpp
p.setDims(nx, ny, 1);                          // before addStage — dims-aware
auto* dwt   = p.addStage<Cdf97Stage<double>>();
auto* quant = p.addStage<QuantizerStage<double, uint32_t>>();
quant->setLinearMode(true);                     // signed codes -> SPECK2D
quant->setErrorBoundMode(ErrorBoundMode::ABS);
quant->setErrorBound(1e-4);

p.connect(quant, dwt);
p.finalize();
```

---

## Performance

Two non-numerical (bit-exact-preserving) optimizations are implemented,
tried automatically in this order, each falling back cleanly when it can't
be used:

1. **Persistent grid-sync kernel** — fuses every level/axis of one transform
   into a single cooperative-groups launch, avoiding the fixed per-launch
   latency of a naive per-level scheme (SPERR's level cap is 6, so a naive
   2-D transform is 12 sequential small launches regardless of field size).
   Falls back transparently (via `cudaOccupancyMaxActiveBlocksPerMultiprocessor`,
   not a hardcoded compute-capability gate) whenever the needed grid doesn't
   fit one cooperative wave on the current GPU. Measured on H100: +45% at
   256², +29% at 512², ~0% at 1024² and above (occupancy-limited).
2. **Coalesced-tile kernel for strided passes** (2-D Y-axis; 3-D Y/Z-axis) —
   a strided pass (`elem_stride != 1`) ran 2.2–2.9x slower than a contiguous
   pass of the same size purely from memory coalescing. One block now owns
   several adjacent lines and loads/stores them via coalesced row-major
   sweeps before calling the same, unchanged lifting arithmetic. Complements
   (doesn't overlap) the persistent kernel — measured 2048² 265->350 GB/s
   (+32%), 4096² 320->345 GB/s (+8%).

Still ~6-9x behind a single-launch elementwise stage even where both
optimizations engage — this is the normal throughput range for a "heavy"
per-line transform, faster than `GInterpStage`, ~2.6-5x behind
`TiledLorenzoStage`.

---

## TOML

```toml
[[stage]]
type = "CDF97"
data_type = "float64"      # or "float32" — see Template parameter above
```

No other configuration: level count, dyadic-vs-packet selection, and boundary
handling are all derived automatically from the pipeline's dims.
