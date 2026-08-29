# Speck2DStage {#stage_speck2d}

**Header:** `modules/coders/speck2d/speck2d_stage.h`
**Kernels:** `modules/coders/speck2d/speck2d_kernels.cuh`
**Class:** `fz::Speck2DStage`
**Category:** Coder (lossless, variable-length output)

---

## What it does

A GPU-parallel "wavefront" SPECK-like coder: a hierarchical significance-map
bit-plane coder over a 2-D field of signed integer codes. Codes a coefficient's
magnitude one bit-plane at a time (MSB-first), using a quadtree significance
map to skip whole spatially-clustered near-zero regions for very few bits —
the same mechanism SPERR's SPECK coder uses, but reformulated so that **both
encode and decode are fully data-parallel** (SPIHT/SPECK decode is usually
considered inherently serial; this format is co-designed specifically to avoid
that).

**This is not a port of SPERR's SPECK bitstream** — see
[THIRD_PARTY.md](../../THIRD_PARTY.md) and
`memory/speck_algorithm_writeup.md` for the full algorithm derivation, what
was kept/changed/dropped versus the reference, and a calibrated novelty
statement.

### Why "2D" and not just "SPECK"

SPERR itself splits SPECK by dimensionality into separate classes
(`SPECK2D_INT`, `SPECK3D_INT`) — the set-partitioning geometry is different in
each case (quadtree vs. octree), not a parameter of one shared algorithm. This
stage implements only the 2-D (quadtree) case so far; a `Speck3DStage` would
be a distinct implementation, not a flag on this one, so the name says exactly
what's supported rather than promising 3-D and throwing at `execute()`.

---

## Input / output contract

- **Input:** `int32_t`, two's-complement signed codes, `nx * ny` elements —
  the same convention `QuantizerStage::setLinearMode(true)` emits (declares
  `DataType::INT32`, so the two stages connect without a cast). Split into
  sign + magnitude on-device; no dependency on the input having come from a
  DWT specifically (see **Beyond DWT coefficients** below).
- **Output:** a variable-length packed bitstream, smaller than the input in
  the common case. Worst case is a **proven** bound, not a guess: every
  internal quadtree node has >= 2 non-empty children (even a degenerate 1xN
  rectangle bisects into exactly 2), so node count `nn <= 2n - 1` for `n`
  leaves; combined with every present node/leaf costing at most 32 bits (one
  `uint32_t` word), `(3n + 8)` words is a safe upper bound with no tree build
  required at estimate time.
- **Not size-preserving**, and **not CUDA-Graph-compatible**
  (`isGraphCompatible()` is `false`): output size is data-dependent, read back
  asynchronously during `execute()` and completed in `postStreamSync()` — the
  same pattern `RLEStage` uses. One additional small mid-pipeline sync (to
  learn `nbitsA`, an absolute bit offset, before packing Section B) is
  accepted rather than fully re-plumbed; every other scalar read is async.
- **2-D only.** `execute()` throws if the pipeline's third dimension is > 1.

---

## Typical pipeline (SPERR-style)

```cpp
p.setDims(nx, ny, 1);
auto* dwt   = p.addStage<Cdf97Stage<double>>();
auto* quant = p.addStage<QuantizerStage<double, uint32_t>>();
quant->setLinearMode(true);                 // REQUIRED: SPECK2D needs signed int32 codes
quant->setErrorBoundMode(ErrorBoundMode::ABS);
quant->setErrorBound(1e-4);
auto* speck = p.addStage<Speck2DStage>();

p.connect(quant, dwt);
p.connect(speck, quant, "codes");           // Quantizer's linear-mode output port
p.finalize();
```

Or via TOML — see `examples/presets/sperr_gpu.toml` for the complete,
runnable preset (`fzgmod-cli -c examples/presets/sperr_gpu.toml -i
data/CLDHGH.f32 -l 3600x1800x1 -b --report`).

---

## Format (why decode is parallel here)

The reference SPECK coder maintains linked LIP/LIS/LSP lists and a DFS-serial
traversal, whose bit order is designed to be embedded/progressive (any prefix
of the stream is a valid coarser approximation) — that property is what makes
it inherently serial, on both encode and decode.

This stage instead precomputes a **significance pyramid**
(`onset[node] = max msb over its pixels`, a plain max-reduction over the
quadtree — embarrassingly parallel to build) and packs it into two sections
whose positions are each independently knowable:

- **Section A (tree):** nodes grouped by level, root first. Each present
  node's onset is a *terminated-unary* gap (`parent_onset - onset` zero bits,
  then a `1`) — the terminator makes a level's codewords self-delimiting, so
  decode ranks the `1`-bits in one parallel pass, then assigns onsets in an
  `O(depth)` (not `O(n)`) wavefront, one level at a time.
- **Section B (magnitude):** significant leaves, sign + mantissa bits,
  positions from a prefix sum over leaf lengths — a final parallel scatter.

Dropped versus the reference: embedded/progressive bit ordering entirely
(unused by FZGM's error-bounded model), and `need_decide` elision (its
parallel form has a within-sibling dependency that doesn't fit a
level-independent decode). Measured rate cost vs. SPERR's own SPECK payload on
real DWT coefficients: **1.10x-1.31x** (need_decide ~4-6% of that, the rest
from the terminator, worst when the field is sparse).

### Throughput lever: shallow-level fusion

A quadtree's level sizes are geometric in depth (`1, 4, 16, ..., 4^L`,
confirmed empirically at every field size tested) — most levels are tiny, and
a naive per-level-launch implementation is launch-latency-bound, not
bandwidth-bound. The shallow prefix of levels (count `<= 1024`, i.e. fits one
thread block) fuses into a single-block kernel per side
(`__syncthreads()` between levels, no cooperative-launch machinery needed);
the few genuinely large deep levels keep ordinary per-level launches. Measured
+7% to +70% (decode) / +4% to +17% (encode) depending on field size, byte-
identical/lossless before and after.

---

## Measured performance (H100, real DWT coefficients, CLDHGH 3600x1800)

| | GPU (this stage) | SPERR (single-thread CPU) | Speedup |
|---|---|---|---|
| Encode | 28.5-29.4 GB/s | — | 66x-349x |
| Decode | 20.9-37.7 GB/s | — | 27x-254x |

Correctness: byte-identical to a from-scratch CPU reference model at every
tested shape (8² to 2048²); full GPU-encode -> GPU-decode round trip lossless.
`compute-sanitizer` memcheck/racecheck/synccheck clean. Full numbers, the
rate/entropy investigation (including one dead end — an order-0 entropy
estimate that overstated a potential arithmetic-coding win, corrected by a
zstd measurement that found none), and the phase-by-phase build log are in
`memory/speck_gpu_design.md`.

**These throughput numbers are unaffected by, and are a separate concern
from, the pipeline's error-bound guarantee:** feeding this stage from
`Cdf97Stage` through a plain (non-subband-scaled) `QuantizerStage` does
**not**, by itself, reproduce SPERR's error-BOUND guarantee — only its
pipeline structure. `examples/presets/sperr_gpu.toml` closes that gap with
[`Cdf97OutlierCorrectStage`](outlier_correct.md) between `Quantizer` and this
stage: a sparse, exact correction pass verified end to end (16/16
`eb_ok=True` across real fields and bounds `1e-2`..`1e-5`, see
`memory/speck_gpu_design.md` sec.9). This stage itself was never the
cause — `Speck2DStage` codes exactly the coefficients it's handed, correctly
— the gap was in the naive coefficient-domain quantization step upstream,
and it's now fixed at that layer, not here.

---

## Beyond DWT coefficients

Nothing in the algorithm depends on the input having come from a wavelet
transform — it's a magnitude-quadtree significance coder over any signed 2-D
integer array, and this is proven, not assumed: the correctness suite includes
pure uniform-random data (zero DWT-like structure) alongside DWT-derived
data, and both round-trip losslessly.

The *compression ratio* benefit, however, is conditional: it comes entirely
from spatially-clustered near-zero regions letting the quadtree prune cheaply.
Dense, unstructured data gets no benefit (measured: v2's payload on
uniform-random input is *larger* than a flat unary encoding — pure overhead,
no pruning). The natural non-DWT candidates are **prediction residuals**
(`LorenzoStage`/`TiledLorenzoStage`/`GInterpStage` output), which share the
same "small in smooth regions, large near features" shape that makes DWT
detail subbands SPECK-friendly — an untested but plausible next experiment,
positioning this stage as a 2-D-block-aware alternative to `RLEStage`/
`RZEStage` (which only see 1-D sequential runs) rather than a DWT-only stage.

---

## TOML

```toml
[[stage]]
type = "SPECK2D"
```

No user-facing configuration — dims, the significance threshold, and the
Section A/B split point are all pipeline/data-derived and round-trip via the
stage's own serialized header (`Speck2DConfig`: `dim_x`, `dim_y`, `B`,
`nbits_a`).
