# Codebase Notes {#codebase_notes}

Longform engineering notes that back up decisions in the source: measurement
tables, before/after numbers, bug postmortems, and "why not the obvious thing"
arguments.

## Why this file exists

A tuning constant or an unusual kernel launch shape is only defensible with the
evidence behind it, and that evidence is bulky — measurement tables across eight
fields, a bug's full symptom trail, the three approaches that were tried first.
Keeping all of it inline pushes the actual code off the screen: several stages
had 40-110 line comment blocks where the contract worth reading at the call site
was four lines of it.

So the split is by **audience and lifetime**:

| Stays in the source | Lives here |
|---|---|
| The invariant or contract a caller must not violate | The measurements that justified the constant |
| The rule ("target pardeg 131072, floor at 768") | The table showing what each value scored |
| A one-line warning ("do not change without re-measuring") | Why it is not safe to extrapolate, with the curve |
| Why this line is not the obvious thing, in a sentence | The full postmortem of the bug that made it non-obvious |

A note is worth writing when someone would otherwise **redo the work** — repeat
a benchmark sweep, re-derive a race, or "simplify" a shape back into a bug.

## Conventions

Every note has a stable ID (`CN-<AREA>-<n>`) used as its anchor. The source
carries a short comment ending in a pointer:

```cpp
// Grid is (kGatherBlocksPerSegment, n_segs), not one block per segment.
// Rationale and measurements: docs/codebase_notes.md CN-CONCAT-1
```

Rules that keep this from rotting:

- **Never move a contract here.** If violating it breaks the build or corrupts
  data, it belongs at the call site.
- **IDs are permanent.** Superseded notes are marked, not deleted or renumbered,
  so an old pointer still resolves.
- **Date and pin measurements** to hardware, input, and configuration. An
  unattributed number cannot be reproduced or refuted.
- **Deleting the code deletes the note.** A note about a kernel that no longer
  exists is a trap.

---

## CN-CONCAT-1 — concat gather kernel grid shape

**Source:** `src/pipeline/concat_kernel.cu`, `src/pipeline/concat_kernel.h`
**Measured:** 2026-08, H100 (sm_90), CESM-2D `CLDHGH`, `gpu_zstd_lossless.toml`

`launch_gather_kernel` originally launched `<<<n_segs, 256>>>` — one block per
output segment, regardless of segment size. That collapses whenever segments are
unevenly sized, which is precisely the split-mode GPU-Zstd case the kernel exists
to serve: the `literals` port is ~99% of the archive, so one 256-thread block was
copying ~23 MB.

| | before | after |
|---|---|---|
| gather_kernel | 2.27 ms | 0.020 ms (112x) |
| share of pipeline GPU time | 60.7% | negligible |
| effective bandwidth | ~20 GB/s (on a 3 TB/s part) | — |
| CLDHGH split-mode compress, host wall | 3.60 ms | 1.33 ms |
| end-to-end | 7.2 GB/s | 19.5 GB/s |

Output was bit-identical across the change.

**Why it hid so well:** `concatOutputs()` runs *after* the `DagEventTimer`
bracket, so this never appeared in `dag_elapsed_ms` or in the per-stage table.
The only visible symptom was a 2.9x host/device ratio on `gpu_zstd_lossless`
against 1.02x for the single-stream control — and nothing was looking at that.

**Why the block count is a compile-time constant:** the launch configuration must
not depend on input data, or a captured CUDA Graph stops being valid when
replayed on a differently-sized input. Blocks with nothing to do exit after one
descriptor read, which costs nothing next to the copy they enable.

---

## CN-HF-1 — Huffman coarse-encode partition length (`sublen`)

**Source:** `modules/coders/huffman/phf/hf_bk.cc`, `capi_phf_coarse_tune_sublen()`
**Measured:** 2026-08, H100 (sm_90), `LorenzoQuant -> Huffman`, `bklen` 1024,
`book_source = Adaptive`, `encode_mode = Coarse`

`sublen` (elements per coarse-encode partition, with
`pardeg = ceil(inlen/sublen)`) drives cost in three directions at once:

- encode phase2/phase4 and the decode kernel launch **pardeg-sized grids**, so
  too large a `sublen` starves the GPU of parallelism;
- `GPU_coarse_encode_phase3_sync` D2Hs two pardeg-sized arrays, runs a **serial
  host prefix-sum** plus two accumulates over pardeg, and H2Ds the result behind
  two stream syncs — all O(pardeg), so too small a `sublen` makes that host
  barrier dominate;
- two pardeg-sized arrays ship **inside the encoded stream**, so too small a
  `sublen` also costs compression ratio.

It was previously the constant 768 (original cuSZ v1.x tuning), ignoring `inlen`
despite taking it as a parameter — leaving throughput on the table in both
directions at once.

Compress GB/s / decompress GB/s / ratio:

| input | n | sublen=768 (old) | current rule | change |
|---|---|---|---|---|
| NYX | 134,217,728 | 216.9 / 139.1 / 29.03 | 246.3 / 151.2 / 29.67 | +14% +9% +2.2% |
| CESM-ATM | 168,480,000 | 143.7 / 131.9 / 7.97 | 202.8 / 137.9 / 8.02 | +41% +5% +0.6% |
| HACC | 280,953,867 | 80.8 / 80.8 / 3.62 | 155.4 / 130.1 / 3.64 | +92% +61% +0.6% |
| HURR | 25,000,000 | 169.8 / 94.8 / 16.13 | unchanged (floor) | — |
| CESM | 6,480,000 | 76.3 / 30.4 / 6.19 | unchanged (floor) | — |

The rule targets `pardeg ~= 131072` and **floors at the historical 768**, which
makes every change a strict Pareto improvement: compress, decompress and ratio
all improve or stay equal on every field measured. The floor is deliberate —
below ~64M elements the trade stops being free, because a smaller `sublen` buys
throughput by spending ratio.

That smaller-`sublen` regime is real and sometimes worth taking, but it *is* a
trade, so it is opt-in via `FZ_HF_SUBLEN`. At `sublen=256` against the 768
default: CESM 26 MB +27% compress / +161% decompress for -3.9% ratio; EXAALT
11.5 MB +30% / +168% for -1.1% ratio.

**Do not extrapolate these constants without re-measuring.** The curve is not
monotonic: parallelism starvation past ~4096 is a steep cliff (CESM-ATM
decompress falls 138 -> 92 GB/s going 1024 -> 4096), which is why `kMaxSublen`
sits at the top of the measured range rather than being open-ended.

`sublen` is recorded in `phf_header`, so streams stay self-describing: archives
written before this change still decode with their own geometry. Not a format
change.

---

## CN-GPULZ-1 — match level and split mode

**Source:** `modules/coders/gpulz/gpulz_stage.h`
**Measured:** H100, 24.7 MB of Lorenzo-quantized CESM `CLDHGH` residuals,
`chunk_size = 2048`, `word_size = 4`

**Match level** (`setMatchLevel`) is encode-side only — the stream format is
identical either way, so a stream produced at one level decodes the same at the
other and the level is not serialized.

| level | throughput | ratio |
|---|---|---|
| 0 — exact longest match over the 32-element near window only | 170 GB/s | 4.36x |
| 1 — additionally consults a hashed two-word-key table (offsets to 255) | 126 GB/s | 5.13x |

**Split mode** (`setSplitMode`) emits `literals` / `lengths` / `offsets` / `meta`
as four ports instead of one interleaved stream. This is the Zstandard split, for
the same reason: the parts have very different symbol distributions, and
interleaving them raises the entropy a downstream coder sees. Measured across six
SDRB fields, coding the four ports separately beats the single-stream form by
**23-43% compression ratio**.

The `literals` port keeps the data's natural word alphabet, so it should feed a
symbol-width-matched coder (`HuffmanStage<uint16_t>` for uint16 quant codes)
rather than a byte coder — that alphabet effect is the larger half of the gain.

**Every port must be entropy coded and all four re-merged.** Unlike the
single-stream form, which codes the whole payload by construction, a split leaks
any byte left out. Both the raw-fallback chunks and the per-chunk size table are
folded into the ports above for exactly that reason: leaving them out cost 28%
and 67% respectively in testing.

---

## CN-HF-2 — why `HuffmanEncodeMode::Fine` does not engage on real data

**Source:** `modules/coders/huffman/huffman_stage.h`, `setEncodeMode()`
**Measured:** CESM-ATM `CLDHGH`/`CLDLOW`/`FLDSC`/`PRECT`/`TS`, `eb` 1e-2 … 1e-5,
`LorenzoQuant -> Huffman`, bklen 1024, radius 512

The fine path packs four codes per 32-bit shard, so it requires **every** code in
the book to fit in 8 bits, and silently falls back to `Coarse` otherwise.

The barrier is structural, not a tuning matter. An 8-bit ceiling admits at most
256 codewords by Kraft's inequality, and the quantized fields carry 322–1025
distinct symbols at all but the coarsest bound. Across all 20 (field, eb) cells
the longest code was **12–24 bits — never ≤ 8**, so the fine path never ran.

Only near-uniform distributions stay inside the ceiling, and those are exactly
the distributions Huffman cannot compress. Where a ≤ 8-bit book is even
constructible, package-merge puts its cost at **+1.8% to +14.6% bits/symbol**.

Restricting to **≤ 16 bits** instead costs at most **+0.31%** across the same
cells and is always constructible. So the change that would make a fine path
reachable is a 2x16-bit shard geometry, not length-limiting to 8. Not
implemented.

Use `getLastUsedFineEncode()` to check which path a call actually took rather
than assuming the requested one ran.

---

## CN-HF-3 — Adaptive codebook staleness and the refit triggers

**Source:** `modules/coders/huffman/huffman_stage.h`, `setRefitThreshold()` /
`setRefitInterval()`
**Measured:** `LorenzoQuant -> Huffman`, eb 1e-4 NOA, radius 512, via
`profiling/huffman_book_drift.cpp`

A codebook pinned by `HuffmanBookSource::Adaptive` goes stale as the symbol
distribution moves. Drift in ratio against `PerBlock`:

| corpus | mean | worst |
|---|---|---|
| 23 CESM `CLOUD` levels | 4.7% | 9.6% |
| 5 different CESM-ATM fields | 8.3% | 27.6% |
| 20 HURR slabs | 9.2% | 22.6% |

Real but bounded — and well under a fixed model book's 12.6–53.2%.

**The sharp edge is a degenerate first block.** A constant (zero-range) fitting
block poisons the whole run: the same CESM field measures **42.0% mean when the
fit lands on a constant level versus 4.7% when it does not**, and constant slabs
are common (CESM `CLOUD` levels 0-2, HURR `CLOUDf48` slabs 18-19). That is the
case that made `Fixed` look competitive (19.5% vs 42.0%) — a generic shape
degrades gracefully where an overfitted one collapses.

**Bit-rate trigger** (`setRefitThreshold`, default 1.2) is free: `encode()`
already reports `total_nbit`, so bits-per-symbol needs no histogram. With the
degeneracy guard, stepping 26 CESM `CLOUD` levels where the first is constant
goes from **42.0% mean / 52.5% worst to 0.86% / 3.06%**.

**Periodic refit** (`setRefitInterval`, default 0 = off) exists because the rate
trigger only fires when the rate gets *worse*. Across 5 different CESM-ATM
variables the default is unchanged at 8.3%, because `LHFLX`'s absolute bit rate
*fell* under a mismatched book even while losing 27.6% ratio — a rate trigger
structurally cannot see that. `--refit-interval 2` brings it to 6.9%.

Deciding that case properly needs the fresh book's rate, i.e. the histogram this
mode exists to avoid. **When every call carries a genuinely different variable,
`PerBlock` remains the right choice.**

---

## CN-HF-4 — why an unencodable book falls back instead of throwing

**Source:** `modules/coders/huffman/huffman_stage.cu`
**Measured:** HACC, the two fields that hard-failed

A histogram spanning a wide enough dynamic range drives the rarest symbol past
the 27-bit code field, which the builder clamps silently and then emits a stream
nothing can decode.

Falling back to an Adaptive book **is not a relaxation of the error bound**: the
bound belongs to the quantizer, and flooring frequencies changes how symbols are
spelled, not what they mean. On the two fields that hard-failed:

| field | eb | CR | PSNR | max abs err |
|---|---|---|---|---|
| HACC `vy` | 1e-2 | 15.27x | 44.78 dB | exactly the bound |
| HACC `xx` | 1e-3 | 13.79x | 64.77 dB | exactly the bound |

Throwing instead cost the `cusz` preset **every wide-dynamic-range field in the
corpus**, which is a worse answer than a book the format can actually hold.
`buildAdaptiveBook` halves the floor shift until the book fits, and shift 0 is
uniform, so the fallback always terminates.

---

## CN-LRZ-1 — Lorenzo block-mode inverse: CTA width vs reset period

**Source:** `modules/predictors/lorenzo/lorenzo_stage.cu`,
`lorenzo_segmented_scan_kernel`

The original block-mode inverse launched `blockDim == block_size`, tying CTA
width to the reset period: a 1024-element segment meant 1024-thread blocks
running Hillis-Steele with ~2·log2(1024) barriers. `ncu` flagged it as
barrier-bound, and decompression fell off a cliff exactly there — geometric mean
**102 GB/s at `block_size = 512` down to 69 at 1024**.

The replacement gives each thread `Seq` consecutive elements: serial scan in
registers, then a warp shuffle scan, then one pass over the warp totals — **2
barriers per scan pass regardless of segment length**.

| field | before | after |
|---|---|---|
| CESM `Z3` decompress @ block_size 1024 | 86 GB/s | 120 GB/s |
| CESM `U` decompress @ block_size 1024 | 68 GB/s | 107 GB/s |

Compression ratio and PSNR bit-identical. The point is not just the speedup:
1024 now matches or beats 256/512 instead of trailing them, which unlocks the
highest-ratio configuration. Reset periods that are not a multiple of 32 fall
back to the barrier-based scan, which handles any width.

---

## CN-TLRZ-1 — tiled Lorenzo inverse: why per-row, not phased per-tile

**Source:** `modules/predictors/tiled_lorenzo/tiled_lorenzo_stage.cu`,
`tiled_lorenzo_scan_kernel_rows`

The earlier phased one-block-per-tile inverse was barrier-bound: `ncu` showed
`sm__throughput` 51% at 91% resident warps, because it left most of its 64
threads idle across two `__syncthreads` — even the busiest phase used only
`ty*tz` of `tile_elems` threads.

The per-row kernel removes barriers and idle lanes entirely: every thread is
self-contained and owns one x-row. Exploiting the separable structure, a row
thread re-derives its own seed by walking the tile's tiny x=0 "spine" — the
z-column prefix then the y-column prefix — and then runs its own tx-length
x-chain.

The spine walks are a handful of adds over L1-resident bytes (re-read across the
tile's rows, but `tile_elems` is tiny); the dominant traffic is the coalesced
read of each row's contiguous tx deltas plus the inherent tile→natural scatter
store. The correctness identity is stated at the kernel itself, since it is what
makes the rewrite legitimate rather than merely faster.

---

## CN-QUANT-1 — rounding must happen in the input's precision

**Source:** `modules/quantizers/quantizer/quantizer.cu`, `quantRound()`
**Measured:** 2026-08-07, S3D `N2`; see the FZGM paper notes

The ABS/NOA kernels rounded with an unconditional `__float2int_rn`, capping
quantization at float32 precision no matter what `TInput` was. That silently
broke every f64 field whose values sit far from zero relative to their own
range.

S3D `N2` spans 1.1e-5 about 0.7369. The float32 spacing there is 5.96e-08 — **54x
the `abs_eb` that a 1e-4 NOA bound asks for** (1.103e-09). The float32 rounding
error alone blew the bound, so round-trips came back at **48-53 dB reporting
`status: ok`**.

`int` remains the bin type: codes are at most 32-bit, and widening `q` would
change the wrap semantics that linear mode documents.

---

## CN-QUANT-2 — why outlier-capacity overflow throws instead of logging

**Source:** `modules/quantizers/quantizer/quantizer.cu`
**Measured:** 2026-08-07, S3D `N2`; see the FZGM paper notes

Overflow means excess outliers were **dropped**, and a dropped outlier
reconstructs to whatever the code path leaves behind — for a field whose whole
range sits far outside the radius, effectively zero. The caller gets a stream
that decodes cleanly into wrong values.

It used to be a DEBUG log line, and the corruption was silent on real data: S3D
`N2` (range 1.1e-5 on an offset of 0.74) drives every element outside a 32768
radius, overflows a 10% capacity, and round-trips at **-96 dB PSNR with
`status: ok`**.

Policy matches `LorenzoQuantStage`: `outlier_capacity == 0` is an explicit opt-in
to the lossy trade-off and stays a quiet drop; any other capacity that overflows
is a failure to honour the error bound and must say so.

---

## CN-TIMER-1 — why DAG timing uses CUDA events, not a host clock

**Source:** `src/pipeline/dag_event_timer.h`

`DagEventTimer` brackets `dag->execute()` (or a `cudaGraphLaunch`) with a
start/stop CUDA event pair, so `elapsedMs()` is the **device wall time** of the
pipeline: from when the stream reaches the start marker until every kernel —
including those joined back from internal streams at the end of `execute()` — has
finished.

This replaced a host `steady_clock` measurement around `execute()`. That was
measuring the wrong thing: `execute()` returns to the host after merely
*enqueuing* kernels (it ends in an async `cudaStreamWaitEvent` barrier), so a
host timer there captured **launch latency, not GPU compute**.

The event bracket deliberately excludes host setup and PCIe transfers issued
outside it. Note the corollary that bit CN-CONCAT-1: work performed outside the
bracket — such as `concatOutputs()` — is invisible to `dag_elapsed_ms`, so a
regression there shows up only in the host/device wall-time ratio.
