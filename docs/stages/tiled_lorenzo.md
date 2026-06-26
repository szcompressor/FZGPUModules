# TiledLorenzoStage {#stage_tiled_lorenzo}

**Header:** `modules/predictors/tiled_lorenzo/tiled_lorenzo_stage.h`
**Class:** `fz::TiledLorenzoStage<T>`
**Category:** Predictor (lossless)

---

## What it does

The cuSZp3 **dimension-aware (tiled separable) delta** predictor, as a modular
stage. The input (signed quantizer codes) is partitioned into fixed-size tiles —
8×8 in 2-D, 4×4×4 in 3-D by default — and within each tile a *separable* integer
Lorenzo is applied:

- the **leading x-column** (`x == 0`) predicts down **y** (and the leading y/z
  edge predicts down **z**),
- every other element predicts from its **x** neighbour,
- the tile origin `(0,0,0)` is stored as a delta-vs-0 (its own full magnitude).

This is the prediction used by the cuSZp3 `plain` and `outlier` kernels. It is a
lossless integer delta on already-quantized codes, so the inverse is an exact
separable prefix-sum.

### Tile-major output

Unlike `LorenzoStage` (which is size- and layout-preserving), the forward pass
emits its output in **tile-major** order: each tile's `tile_elems = tx*ty*tz`
elements are written contiguously, and edge tiles are **zero-padded** to a full
tile. The forward output therefore has `num_tiles * tile_elems` elements
(≥ the input count when the dims aren't exact multiples of the tile).

The point of tile-major output is to line the tiles up with a downstream
fixed-rate coder: pair this stage with
`AdaptiveBitpackStage(block_size = tile_elems)` so each tile becomes exactly one
coder block — reproducing cuSZp3's per-tile fixed-rate layout with the coder
unchanged. The inverse un-tiles back to the original natural (row-major) order.

This stage is a *separate* stage rather than a `LorenzoStage` mode precisely
because the tile-major reshape + padding break LorenzoStage's "same size, same
layout" contract, and the separable formula differs from LorenzoStage's N-D
inclusion-exclusion delta. (Same reasoning as `AdaptiveBitpackStage` vs
`BitpackStage`.)

---

## Template parameter

`T` — signed element type: `int16_t` or `int32_t` (linear quantizer codes).

---

## Stage settings

| Setting | Purpose | Notes |
|---|---|---|
| `setDims(x, y, z)` | Spatial dimensions (call before `addStage`, or use `Pipeline::setDims`) | ndim inferred from non-unit dims |
| `setTileShape(tx, ty, tz)` | Tile extents | each ≤ 255, product `tx*ty*tz` ∈ [1,1024]; default 2-D 8×8, 3-D 4×4×4, 1-D 64 |

```cpp
p.setDims(NX, NY);
auto* tl = p.addStage<TiledLorenzoStage<int32_t>>();
tl->setTileShape(8, 8);            // cuSZp3 2-D tile
```

The downstream `AdaptiveBitpackStage` block size should equal `getTileElems()`
(64 for both 8×8 and 4×4×4) so coder blocks align with tiles.

---

## Ports

Single input → single output (both signed `T`). The forward output is the padded
tile-major delta stream; the inverse output is the natural-order reconstruction.

| Direction | Port | Type |
|---|---|---|
| Forward in / inverse out | `"output"` | `T[n]` (natural order) |
| Forward out / inverse in | `"output"` | `T[num_tiles*tile_elems]` (tile-major, padded) |

---

## Graph compatibility

`isGraphCompatible()` is **true** — pure kernels with deterministic output sizes
(the padded size is known from dims at finalize; no host-blocking D2H).

---

## Typical pipeline (cuSZp3)

The three cuSZp3 modes decompose as:

```cpp
// plain: Quantizer(linear) -> TiledLorenzo -> AdaptiveBitpack
auto* quant = p.addStage<QuantizerStage<float, uint32_t>>();
quant->setErrorBound(1e-3f);
quant->setErrorBoundMode(ErrorBoundMode::ABS);
quant->setLinearMode(true);                 // signed INT32 codes

auto* tl = p.addStage<TiledLorenzoStage<int32_t>>();
tl->setTileShape(8, 8);
p.connect(tl, quant, "codes");

auto* ab = p.addStage<AdaptiveBitpackStage<int32_t>>();
ab->setBlockSize(64);                        // = tile_elems (8*8)
// ab->setOutlierSelection(true);            // -> cuSZp3 "outlier" mode
p.connect(ab, tl);
p.finalize();
```

- **fixed** mode = drop `TiledLorenzo` (`Quantizer(linear) → AdaptiveBitpack`).
- **plain** mode = the chain above.
- **outlier** mode = plain + `ab->setOutlierSelection(true)`.

For 1-D data, `LorenzoStage::setBlockSize(32)` already provides the cuSZp/cuSZp3
1-D block-local delta; `TiledLorenzoStage` targets the 2-D/3-D tiled case.

---

## TOML

```toml
[[stage]]
type = "TiledLorenzo"
data_type = "int32"   # or "int16"
tile_x = 8
tile_y = 8
tile_z = 1            # omit / 1 for 2-D
```

Dimensions come from the pipeline (`setDims` / the CLI `--dims`), not the stage block.

---

## Acknowledgements

The dimension-aware separable delta is the cuSZp3 / VGC design (Yafan Huang,
Sheng Di, Guanpeng Li, Franck Cappello, "GPU Lossy Compression for HPC Can Be
Versatile and Ultra-Fast", SC'25). This is an independent reimplementation of the
published scheme — no cuSZp source is copied. cuSZp3's memory-efficient
compression and selective decompression features are not ported. Repo:
https://github.com/szcompressor/cuSZp. See `THIRD_PARTY.md` and
`memory/cuszp_stages.md` (Part 8).
