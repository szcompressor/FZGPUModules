# AdaptiveLorenzoStage {#stage_adaptive_lorenzo}

**Header:** `modules/fused/adaptive_lorenzo/adaptive_lorenzo_stage.h`
**Class:** `fz::AdaptiveLorenzoStage<T>`
**Category:** Fused predictor (per-tile adaptive)

---

## What it does

Splits the flattened input into **tiles** of `coder_block_size * blocks_per_tile`
elements (default 32 x 8 = 256) and, for each tile independently, emits whichever
of four prediction variants encodes smallest:

| variant | residual |
|---|---|
| LZ1 | `q_i - q_{i-1}` |
| LZ2 | `q_i - 2q_{i-1} + q_{i-2}` |
| LZ1 + centering | LZ1 on `q_i - mu` |
| LZ2 + centering | LZ2 on `q_i - mu` |

`mu` is the tile's integer mean. The choice is recorded as one byte per tile.

Three things happen at once here:

**Cross-block prediction state.** The prediction chain runs the length of the
whole tile, not the coder's 32-element block. Only the tile's first one or two
elements lack predecessors, instead of one per coder block. With a fixed-rate
coder that matters a lot: a single raw seed sets the bit width for its whole
block, so eliminating 7 of 8 of them is most of the compression win.

**Exact-cost selection.** The variant is chosen by the number of bytes
`AdaptiveBitpackStage` will actually emit — `0` for a block whose residuals are
all zero, else `4 * (r + 1)` for a maximum magnitude of `r` bits — summed over
the tile's blocks. Not entropy, not a heuristic.

**Single-pass evaluation.** All four variants come from one read of the tile.
A constant offset cancels out of a k-th order difference for every element with
`k` predecessors (`delta^k(q - mu) == delta^k(q)`), so the centered variants are
identical to the uncentered ones except in the tile's first one (LZ1) or two
(LZ2) residuals — all inside coder block 0. Their costs follow from re-rating
that single block rather than recomputing anything.

Lossless: the reconstruction is bit-identical to any fixed variant.

---

## Ports

| Port | Type | Size |
|---|---|---|
| `output` | `T` | one per element |
| `modes` | `uint8_t` | **2 bits per tile**, 4 tiles per byte — bit 0 = order 2, bit 1 = centering |
| `means` | `T` | **one per centered tile only** (compacted) |

Ports left unconnected become pipeline outputs and are stored in the `.fzm`.

Both side channels are kept small on purpose, because on a sparse field they
would otherwise dominate the output. The mode map is packed 4 tiles to a byte,
and `means` is compacted down to just the tiles that chose centering: the slot
index is an exclusive scan over the centering bits, which the inverse recomputes
from `modes` alone, so no slot table is stored. Together they cost 0.008
bits/element at a 256-element tile when nothing centers, against 0.156
bits/element for a naive dense layout.

Because the compacted length is data-dependent, the stage resolves it with a
device-to-host readback in `postStreamSync()` (the same pattern `QuantizerStage`
uses for its outlier arrays) and therefore reports `isGraphCompatible() == false`.

---

## Settings

| Setting | Default | Notes |
|---|---|---|
| `coder_block_size` | 32 | **Must be 32.** The cost model and the one-warp-per-block reduction both assume it, and it is the only size at which `AdaptiveBitpackStage` packs each bit-plane into exactly one 32-bit word. |
| `blocks_per_tile` | 8 | `[1, 32]`, so the tile fits one CUDA block. Longer tiles mean a longer prediction chain and less per-tile metadata, but coarser adaptation. |
| `enable_order2` | true | Include the LZ2 variants. |
| `enable_centering` | true | Include the centered variants. |

All four are **constructor arguments**: the port count and tile geometry are
fixed when `Pipeline::addStage()` runs.

```cpp
AdaptiveLorenzoStage<int32_t>::Config c;
c.blocks_per_tile = 16;                       // 512-element tiles
auto* al = p.addStage<AdaptiveLorenzoStage<int32_t>>(c);
p.connect(al, quant, "codes");
p.connect(bitpack, al);                       // "modes" and "means" -> .fzm
```

TOML: see `examples/presets/fsz.toml`.

---

## Cost model coupling

The selection is exact **only when the downstream coder is
`AdaptiveBitpackStage` with `block_size = 32`.** Routing the residuals into
Huffman, ANS, or a different block size leaves the choice reasonable but no
longer optimal, because the modelled byte cost is not the one being paid.

---

## Measured

8 SDRBench fields x 3 error bounds, RTX 3080 Ti, against a fixed-LZ1 predictor
at the same tile length:

See the measured table in `CHANGELOG.md` for the current per-bound geometric
means. Up to 1.95x on CESM `T` and 1.77x on `Z3` — fields with a large constant
offset and a smooth vertical gradient respectively — and the per-tile choice
beats the best *fixed* variant chosen per field, which is the point of adapting
at tile granularity rather than per dataset.

---

## Acknowledgements

The cross-block prediction state, the four-variant adaptive selection, and the
finite-difference cancellation that makes a single-pass evaluation possible are
the design of **FSZ**:

> Jiajun Huang. *FSZ: Breaking the Prediction-Throughput Trade-off in GPU Lossy
> Compression.* SC'26. arXiv:2607.15413.

This stage is an independent reimplementation for the FZGPUModules DAG model,
written from the paper alone before FSZ had a source release; no FSZ source was
used. FSZ 1.0.0 was released 2026-08 under BSD-3-Clause at
https://github.com/JiajunHuang1999/FSZ. The reference implementation fuses
prediction, quantization, and encoding into a single CUDA kernel, whereas this
is the prediction step alone and composes with `QuantizerStage` upstream and
`AdaptiveBitpackStage` downstream.

**Checked against the reference (2026-08-07, H100, 20 cells).** PSNR is identical
on every cell and the bitpacked payload is byte-for-byte the same size, so the
two make identical per-tile decisions. CR is 0.9928 of the reference (geomean);
99.6% of that deficit is the `modes` port above — FSZ carries the same two flags
free in spare bits of its per-block rate byte (its rate needs 5 of 8), which a
predictor decoupled from its coder cannot do. That is the exact price of the
port/DAG boundary: 0.0078 bits/element, invisible at CR 3 and 3% at CR 128.
See `compression_benchmarking/docs/adapters/fsz.md`.
