# ROIBinSplitStage {#stage_roibin_split}

**Header:** `modules/structural/roibin_split/roibin_split_stage.h`
**Class:** `fz::ROIBinSplitStage<TData>` — `TData` is `float` or `double`
**Category:** Structural

**Common instantiation:**
```cpp
p.setDims(1552, 1480, 1);                       // before addStage
auto* split = p.addStage<fz::ROIBinSplitStage<float>>();
split->setRoiHalfWidth(4);                       // 9x9 box per peak
split->setBinFactor(2);                          // 2x2 background binning; 1 = off
split->setPeaksFile("frame.roi");                // compress side only
```

---

## What it does

Splits a detector field into two independently compressible streams plus the
metadata that ties them back together:

| direction | ports |
|---|---|
| forward (`1 → 3`) | field → `roi`, `bg`, `peaks` |
| inverse (`3 → 1`) | `roi`, `bg`, `peaks` → field |

- **roi** — the `(2*hw+1)^2` box around every peak, concatenated in peak order,
  at full resolution and the field's element type.
- **bg** — the background, box-averaged by `bin_factor` within each z-slice:
  `ceil(nx/b) * ceil(ny/b) * nz` values. `b = 1` copies the field through.
- **peaks** — the peak record table, `8 * npeaks` bytes (`UINT8`).

On the inverse, the background is un-binned first and the ROI boxes are then
pasted over it, so the tight-bound values win wherever the two overlap.

## Why it exists

Serial-crystallography frames are almost all background; the science lives in a
few hundred Bragg peaks covering a small percentage of the pixels. A single-bound
compressor must apply the tight ROI bound to the entire frame, and pays for the
other 99% at that bound.

This stage turns that into a graph problem. Because it is `1 → 3`, the two data
streams become two DAG branches, and each branch can carry **its own Quantizer at
its own error bound** and its own coder chain:

```
              ┌── roi ──> Quantizer(eb_tight) ──> AdaptiveBitpack ──┐
  input ──> split ── bg ──> Quantizer(eb_loose) ──> TiledLorenzo ──> AdaptiveBitpack ─┤──> archive
              └── peaks ─────────────────────────────────────────────┘
```

There is no way to say "compress this part tightly and that part loosely" to a
monolithic compressor; expressing it is the point. The design follows ROIBIN-SZ
(Underwood et al.).

## Where the peak list comes from

It is **not** derived from the data. It is the output of the experiment's own peak
finder, which in a real light-source pipeline has already run upstream — the same
assumption ROIBIN-SZ makes. At compress time `setPeaksFile()` reads it; the stage
validates every record against the field bounds and **throws if the file's
`nx/ny/nz` disagree with the pipeline's dimensions**, because a silent mismatch
would relocate every ROI box and still produce output that round-trips.

The table is then re-emitted on the `peaks` port, so it lands **inside the archive
and is counted in the compressed size**. The decompressor never needs the `.roi`
file. At 8 B/peak this is ≈0.01 % of a frame.

### .roi file format

```
magic   char[8]  "FZROI1\0\0"
nx      uint32   fast axis
ny      uint32   slow axis
nz      uint32   frames
npeaks  uint32
records npeaks x { uint32 z; uint16 x; uint16 y; }   (8 bytes each)
```

## Binning can introduce unbounded error

`bin_factor = b > 1` replaces each `b x b` block with its mean. **This is a
resolution reduction, not an error bound.** The background reconstruction error is
the binning error *plus* the quantization error, and it is **not** bounded by the
background branch's error bound.

| `bin_factor` | background guarantee | how to report background fidelity |
|---|---|---|
| `1` | satisfies its stated bound pixel-wise | max abs error and PSNR; the bound gate is meaningful |
| `> 1` | **no bound** | PSNR only; never claim a satisfied error bound |

The stage emits a `getRunNotes()` warning whenever `bin_factor > 1` so this cannot
be lost between the run and the write-up. The **ROI branch satisfies its bound in
both cases** — that is the invariant the science depends on.

---

## Stage settings

| setter | TOML key | default | meaning |
|---|---|---|---|
| `setRoiHalfWidth(hw)` | `roi_half_width` | 4 | box is `(2*hw+1)^2` |
| `setBinFactor(b)` | `bin_factor` | 1 | background binning; 1 disables |
| `setPeaksFile(path)` | `peaks_file` | — | compress-side peak list |
| — | `data_type` | `float32` | `float32` or `float64` |

## TOML configuration

```toml
[[stage]]
name = "split"
type = "ROIBinSplit"
data_type = "float32"
roi_half_width = 4
bin_factor = 2
peaks_file = "frame.roi"
```

Downstream stages connect to the named ports:

```toml
[[stage]]
type   = "TiledLorenzo"
inputs = [{from = "split", port = "roi"}]   # or "bg", or "peaks"
```

> **If a downstream predictor is dimension-aware and `bin_factor > 1`**, give it an
> explicit `dim_x`/`dim_y`/`dim_z` override. The binned background is not the
> pipeline's input shape, and `finalize()` re-pushes the global dims over anything
> set at construction — the predictor would then use the wrong row stride and
> **still round-trip**, silently costing ratio. See
> \ref stage_tiled_lorenzo "TiledLorenzoStage".

## Serialized config header

```
[0..3]   uint32  nx
[4..7]   uint32  ny
[8..11]  uint32  nz
[12..15] uint32  npeaks
[16..17] uint16  roi_half_width
[18..19] uint16  bin_factor
[20]     uint8   element DataType
[21]     uint8   reserved
```
22 bytes. Peak *values* travel on the `peaks` port, not here — the 128-byte stage
config slot cannot hold a few thousand records.

## Acknowledgements

The separation of supplied Bragg-peak regions from a spatially binned
background follows ROIBIN-SZ:

> Robert Underwood, Chun Hong Yoon, Ali Murat Gok, Sheng Di, and Franck
> Cappello. *ROIBIN-SZ: Fast and Science-Preserving Compression for Serial
> Crystallography.* Synchrotron Radiation News 36(4), 17–22, 2023.
> https://doi.org/10.1080/08940886.2023.2245722

The public ROIBIN-SZ integration and examples are distributed with
[SZ2](https://github.com/szcompressor/SZ2/tree/master/example/roibin_example).
This stage is an independent GPU/DAG implementation: no SZ2 or LibPressio
source was copied. Its CUDA kernels, FZROI1 peak-table format, fixed-output
geometry, three named ports, archive integration, and inverse scatter are
FZGPUModules code. See `THIRD_PARTY.md` for the provenance and license record.
