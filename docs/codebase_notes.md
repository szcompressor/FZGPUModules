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

## CN-AB-TR — transpose-based encode: the rate-dependent crossover and threshold 4

**Source:** `modules/coders/adaptive_bitpack/adaptive_bitpack_kernels.cu`
(`encode_pack_kernel_warp_tr`, `encode_pack_outlier_kernel_warp_tr`,
`encodeTransposeThreshold`)

The warp-cooperative pack kernel is ALU-bound, not memory-bound (ncu on CLDHGH,
H100: pack at ~72% ALU pipe, ~20% DRAM, store efficiency already ~1.1
sectors/request — the old scalar path's 88%-excessive-sectors store problem is
gone in the warp kernels). The ALU cost is the O(rate) per-plane `__ballot_sync`
+ bit-extract loop. `encode_pack_*_warp_tr` replaces that loop with a single
32×32 `warpBitTranspose32` per 32-element half — the exact inverse of the
transpose *decode* (`decode_unpack_*_warp_tr`), and since the transpose is an
involution, feeding lane l its magnitude yields plane word l at lane l. The
archive format is unchanged; the sign region still uses one ballot per half.

The win is entirely rate-dependent, because the gather is O(rate) and the
transpose is O(1). Per-kernel `gpu__time_duration.sum` (ncu, CLDHGH 3600×1800,
cuszp2/outlier, isolated pack kernel):

| eb    | gather | transpose (thr 4) | speedup |
|-------|--------|-------------------|---------|
| 1e-3  | 51.8µs | 52.3µs            | ~1.00×  |
| 1e-4  | 64.3µs | 57.4µs            | 1.12×   |
| 1e-5  | 78.3µs | 58.5µs            | 1.34×   |

The gather climbs with tightening bounds (higher residual magnitudes → higher
rate); the transpose stays flat at ~58µs. End-to-end **compress** DAG throughput
(non-graph, matched clock): 1e-3 unchanged, 1e-4 +2.7%, 1e-5 +7.3% — diluted
because the pack kernel is ~¼ of the compress DAG (Lorenzo + quant + rate + scan
+ pack + concat). Decompress is untouched.

**Threshold 4, not the decode side's 6.** Blocks below the threshold fall back to
the gather (cheaper at trivial rate). The encode crossover measured lower than
decode's: at eb=1e-4 threshold 4 (57.4µs) beat threshold 6 (60.6µs) *and* beat
transpose-all (58.4µs) — a rate-4 block's four ballots already cost more than the
fixed 5-shuffle transpose, but forcing the transpose onto rate-1..3 blocks loses.
`FZ_ENCODE_TR` overrides it (`=0` disables entirely for A/B).

**Store coalescing.** The transpose has each of lanes 0..r-1 write one 4-byte
plane word at `base + word_bytes*(1+lane)` (consecutive words → coalesced). A
naïve byte-wise store desequences that across lanes (byte 0 of each lane is
strided by 4) and raised store sectors ~50%. `storePlaneWord` does a single
aligned `uint32` store when the block's payload base is 4-byte aligned — a
warp-uniform property, since every plane slot is base + a multiple of 4 — and
falls back to the byte-wise `store32le` otherwise (outlier `ob_bytes` of 1–3 can
misalign the base). The alignment branch is warp-uniform, so it never diverges.

## CN-AB-FUSE — why single-pass fused encode (decoupled look-back) was rejected

**Prototyped and reverted (2026-08).** The 3-kernel outlier encode path
(`encode_rate` → CUB `ExclusiveSum` → `encode_pack`) reads the quantization codes
twice and recomputes `absU` in both kernels. The obvious fix is to fuse all three
into one kernel that reads the codes once (keeping them in registers between the
rate reduce and the pack) and computes the per-block byte offsets inline with a
Merrill/Garland decoupled look-back at CTA granularity.

It was implemented, validated **bit-exact** against the 3-kernel path (identical
compressed size and PSNR, 27 stage + 6 cuszp tests, compute-sanitizer memcheck +
synccheck clean at 25k CTAs), and it was **2.4x slower** — a clean loss. Two real
concurrency bugs surfaced en route and are worth recording: (1) the tile id must
be `blockIdx.x`, not a dynamic atomic ticket — decode recomputes offsets by
scanning costs in natural block order, so a ticket-scrambled payload layout is
undecodable; (2) aggregate and inclusive prefixes must share a single 64-bit
`{ready,value}` descriptor read/written atomically — separate flag/value words
tear at scale and over-count the offset (→ OOB write).

Why it loses (ncu, CLDHGH, H100): the fused kernel runs at **97% occupancy but
17% SM throughput, with 82% of stall cycles at the CTA `__syncthreads` barrier**.
The pack cannot start until it knows its byte offset, so all 256 threads block on
the look-back while only thread 0 does a serial, global-memory-latency-bound scan
of predecessor descriptors. Embedding the prefix scan into the heavy pack kernel
entangles the scan's serial dependency with the expensive work. The slowdown is a
constant ~2.4x across 32/64/128 MB (not superlinear, so not O(n²) degeneration —
just a fixed structural cost). CUB's standalone scan avoids all of this: it is a
lightweight kernel where the prefix wavefront propagates over trivial per-tile
work in ~5 µs, then the pack runs separately at full occupancy. The ~16% saving
from eliminating the redundant read is dwarfed by the barrier/look-back overhead.

Conclusion: the 3-kernel design is the right architecture for this pipeline. A
warp-cooperative look-back (32 lanes reducing predecessor descriptors in
parallel) would cut the barrier stall but its best case is ~parity for a lot of
added lock-free-concurrency risk — not pursued. Related: the L2-blocked tiling
probe (`profiling/l2tile_profile.cpp`) is the other rejected traffic/locality
lever; this pipeline is compute/latency-bound (see CN-AB-TR), not traffic-bound.

## CN-FUSE-PROOF — full block-local fusion reaches native-class throughput (go decision)

**Go/no-go experiment (`profiling/fuse_proof.cu`), 2026-08.** Counterpart to the
CN-AB-FUSE *negative* result. That one fused only rate+scan+pack and lost 2.4x to
an in-kernel look-back barrier. This one fuses the **entire block-local chain** —
`Quantizer(linear) + Lorenzo(32) + AdaptiveBitpack(32,outlier)` — into per-warp
work: one warp owns one 32-element block and computes quant + Lorenzo delta +
adaptive-bitpack in registers, so the int32 codes are never written to or read
from DRAM.

Measured (H100, compress, best-of-10), fused vs the staged FZGM pipeline:

| field            | staged   | fused    | speedup |
|------------------|----------|----------|---------|
| CLDHGH 256 MB    | 114 GB/s | 358 GB/s | 3.14x   |
| HACC/xx 512 MB   | 113 GB/s | 280 GB/s | 2.49x   |
| NYX 512 MB       | 114 GB/s | 225 GB/s | 1.97x   |

That is native-cuSZp2-class throughput (~200-400 GB/s on H100). Correctness:
**byte-identical archive to the staged path** on all three fields (exact
compressed-size match to the byte over 134M elements — any bug in
quant/Lorenzo/rate/pack would shift a block cost and diverge the size), and
round-trips within the error bound where the linear quantizer itself is in range
(CLDHGH; NYX/HACC exceed int32 code range at eb=1e-3 in *both* paths, a
pre-existing linear-quantizer property, not a fusion bug).

**Why it worked where CN-AB-FUSE failed.** Two design choices: (1) fuse the
*whole* chain so the front-end quant+Lorenzo compute is hidden — ncu shows the
fused rate kernel at 74% SM / 74% ALU (healthy), 360 us, vs the staged rate-only
kernel's 385 us: quant+Lorenzo folded in for ~free while deleting their two
separate kernels (~482 us) and the code round-trips. (2) Keep the lightweight CUB
offset scan and **recompute** quant+Lorenzo in the pack kernel rather than
materialise deltas or run an in-kernel look-back — this sidesteps the barrier
that sank CN-AB-FUSE, at the cost of reading the input floats twice (still a big
net win). A true single-kernel variant with an amortised look-back is the
remaining headroom toward the top of the native range.

Speedup scales inversely with rate: low-rate/high-CR data (CLDHGH) benefits most
(cheap pack, so front-end elimination dominates); high-rate data (NYX) least (the
pack's inherent bit-packing ALU dominates). Floor observed ~2x.

**Decision: GO on the auto-fusion framework.** The runtime-modularity tax is
recoverable and large. This hand-fused kernel is the reference implementation and
the first "block-local predict+quant+fixedcoder" driver skeleton for the planner;
the staged path is its byte-exact validation oracle. Aligns with the FZ group's
domain-specific-compiler-for-lossy-compression direction (WSE case study) — same
codegen-of-fused-kernels idea, retargeted to CUDA. Next: the `FusionSpec` stage
contract + a DAG fusion planner (see CN-FUSE-PLAN once landed).

## CN-FUSE-PLAN — the fusion planner: what it identifies and why

**Source:** `include/stage/fusion.h`, `src/pipeline/fusion_planner.cpp`

Step 1 of the auto-fusion framework (motivation: CN-FUSE-PROOF). Pure analysis —
it changes no execution; it finds the stage chains a fused kernel *could*
collapse, so a later pass can substitute a fused (registered or NVRTC-generated)
implementation and leave everything else staged.

`Stage::getFusionSpec()` classifies a stage's data-access pattern — `Map`
(element-wise), `BlockLocal` (resettable fixed-size neighbourhood), `Cooperative`
(a per-block fixed-length coder), or `Unfusable` (default; opaque or genuinely
global). Access pattern, not computation, is what decides fusability, so the
enum is deliberately tiny. Config-dependent fusability is reported honestly:
`QuantizerStage` is `Map` only in linear mode (the outlier/in-place paths scatter
to side buffers), `LorenzoStage` is `BlockLocal` only in 1-D block-reset mode
(the N-D stencil needs a different driver), `AdaptiveBitpackStage` is
`Cooperative` only at the warp block sizes 32/64. Inverse stages are always
`Unfusable`.

`planFusionGroups()` returns maximal groups under four rules, each load-bearing:
(1) **strictly linear** — a group edge needs the producer to feed exactly one
stage and the consumer to be fed by exactly one; fan-in/out inside a group would
break register-resident composition. (2) **one block size** — block-local and
cooperative members must agree (Map imposes none); mixed periods can't share a
tile. (3) **coder terminates** — nothing fuses past a `Cooperative` variable-
length coder, so it's always the tail. (4) **size ≥ 2** — a lone stage is not a
fusion. The cuszp2 pipeline resolves to exactly one group
`{Quantizer, Lorenzo, AdaptiveBitpack}`, block 32, coder-terminated — the chain
the CN-FUSE-PROOF kernel hand-fuses. Tests: `tests/pipeline/test_fusion_planner.cpp`.

Not yet built (next steps): a fused-implementation registry keyed by the
stage-chain fingerprint (`scripts/gen_stage_fingerprints.py` gives the key), DAG
substitution of a `FusedStage` for a matched group, and NVRTC codegen that
assembles a driver skeleton from the members' device functors for chains with no
hand-written implementation.

## CN-FUSE-EXEC — wiring fused execution into the DAG (Step 2)

**Source:** `include/pipeline/fusion_registry.h`, `src/pipeline/fusion_registry.cpp`,
`modules/fused/fused_block/`, the fusion hook in `CompressionDAG::execute()`,
`Pipeline::planAndInstallFusion()`, `FusionPolicy`.

Turns the planner (CN-FUSE-PLAN) into measured speedup. Measured end-to-end
through the real Pipeline on CLDHGH (H100): staged 107→**234 GB/s** at eb=1e-3
(2.18x), 105→**220** at 1e-4 (2.10x), identical PSNR — native-class.

**Integration choice: swap execution, not DAG structure.** `buildHeader()` and
`buildInverseDAG()` read the DAG *nodes*, so keeping all three stage nodes intact
(only replacing forward execution) leaves the archive byte-identical and
decompress completely unchanged. The registry (`findFusedImpl`) matches a group's
exact shape; `CompressionDAG::execute()` runs the matched `FusedImpl` at the
group's head node — writing the tail node's output buffer — and records every
member's completion event so downstream waits are satisfied, skipping the members'
individual `execute()`s. `FusionPolicy::Auto` (default `Off`, `FZ_FUSION=off|auto`
override) installs groups at finalize; unmatched groups stay staged, so Auto is
always safe.

**Two non-obvious correctness requirements, both from decompress reusing the
forward stage objects** (`setInverse(true)`; the in-memory decompress path does
*not* call `deserializeHeader`, so it relies on forward-execute side effects):

1. **Prime forward-computed state the inverse needs.** The quantizer computes
   `computed_abs_eb_` during forward `execute()`; fusion skips that, so the reused
   inverse quant reconstructed with the *default* bound (1e-4) — a silent **10x
   too small** reconstruction with a byte-identical archive. The fused runner must
   call `QuantizerStage::primeAbsEbForFusion()` (ABS only). The tail coder
   likewise needs `num_elements_` via `setFusedResult()`. General rule: a fused
   runner must establish whatever forward state the group's reused stages read on
   the inverse path.
2. **Disable buffer coloring.** Coloring aliases buffers by *staged* liveness,
   where a group's input dies after its first stage. A fused kernel keeps that
   input live across the whole group while writing the group's output, so an
   aliased input/output region corrupts. `planAndInstallFusion()` disables
   coloring (and CUDA graph mode — the fused runner synchronises to read the
   archive length) when any group installs.

Validated bit-exact (byte-identical archive to staged, round-trip within bound,
`rs == rf`) and compute-sanitizer clean: `tests/pipeline/test_fusion_planner.cpp`.
Next: NVRTC codegen keyed by the stage-chain fingerprint (see CN-FUSE-DRIVER for
the parametric-driver step that generalised the runner to cuSZp3).

## CN-FUSE-DRIVER — parametric block-local fused driver (cuSZp2 + cuSZp3) (Step 3)

**Source:** `modules/fused/fused_block/` (the templated kernels + two predictor
policies), `matchesCuszp3`/`runCuszp3` in `src/pipeline/fusion_registry.cpp`,
`TiledLorenzoStage::getFusionSpec()`.

Generalises the single hand-written cuSZp2 runner (CN-FUSE-EXEC) into one driver
for the whole *predict+quant+fixed-rate-outlier-coder* family. Key realisation:
**the fused rate/pack kernels ARE the AdaptiveBitpack outlier warp kernels**
(`encode_{rate,pack}_outlier_kernel_warp`, CN-AB-TR area), with the materialised
int-codes array `in[start+idx]` replaced by an inline `pred.delta(lane,b,m)` call.
So the two kernels are templated on `<int ElemsPerLane, class Pred>` (block_size =
32*EPL, EPL≤2 keeps the per-lane register arrays from spilling) and a predictor
policy struct passed by value:
- `Lorenzo1DPredictor` (EPL=1, cuSZp2): 1-D Lorenzo via intra-warp `__shfl_up`.
- `TiledLorenzo2DPredictor` (EPL=2, cuSZp3): 2-D separable tiled Lorenzo (tz==1),
  each element re-quantises its own left/up predecessor from the float field — a
  pure map, so the neighbour code equals what the staged quantizer produced.
Because the pack path is byte-for-byte the shipped AB outlier warp encode, and EPL=1
reduces to the old cuSZp2 kernel exactly, both archives stay byte-identical (the
cuSZp2 end-to-end test still passes unchanged).

**cuSZp3 element bookkeeping (the non-obvious part):** the AB stage's element count
is the *padded tile-major* count `num_tiles*tile_elems` (TiledLorenzo emits
zero-padded edge tiles), so every block is full (count==64) and padding elements are
`ab_active` with delta 0 — the driver passes `n_ab` (not the natural field size) and
`setFusedResult(n_ab, …)`. The predictor, by contrast, keys off the *field* bounds
(`gx<dx && gy<dy`) and returns 0 for padding, matching `tiled_lorenzo_delta_kernel`.
Matcher requires 2-D (tz==1) tile_elems==64 and AB block 64 + outlier; other shapes
return 0 and fall back to staged.

Measured end-to-end on CLDHGH 3600x1800 (H100, eb=1e-3), compress steady-state:
staged 178.9 GB/s dag / 151.5 host → fused **222.7 dag / 196.0 host** (~1.25x dag,
1.29x host), ratio 8.31x and PSNR 63.7997 identical. Smaller multiple than cuSZp2's
2.1x because the cuSZp3 staged baseline (efficient per-row TiledLorenzo) is already
faster; the win is still removing two int32-width DRAM round-trips. Validated
byte-identical + round-trip on non-tile-aligned dims (300x180, exercises edge-tile
padding) and compute-sanitizer clean: `tests/pipeline/test_fusion_planner.cpp`
(`Cuszp3*`). fzgpu is a different family (BitplaneRZE coder, radius/zigzag quant with
its own outliers) — deferred; the natural next step is NVRTC codegen keyed by the
stage-chain fingerprint.

## CN-BSHUF-SMEM — shared-memory staging fixes the bitshuffle uncoalesced-store bug

**Source:** `modules/shufflers/bitshuffle/bitshuffle_stage.cu` — `bitshuffle{Encode,Decode}Kernel32Smem`, `butterfly32`, `FZ_BITSHUF_SMEM`.

The 32-bit butterfly bitshuffle (`bitshuffleEncodeKernel32`) writes each bit-plane
word to global at stride `npp = N_chunk/32` (512 B for the default 16 KB chunk):
`out[i/32 + sublane*npp]`. The 32 lanes of a warp share `i/32` and differ only in
`sublane`, so a warp's stores hit 32 addresses 512 B apart — **fully uncoalesced**.

Measured (PFPL on CLDHGH 3600x1800, eb=1e-4 NOA, H100, ncu):

| | scattered | smem-staged |
|---|---|---|
| store sectors/request | **32** (max) | **4** (coalesced) |
| store byte efficiency | 12.5% | — |
| Compute (SM) | 10% | — |
| DRAM throughput | **9%** | **31%** |
| L2 pipeline | 79% (bound) | — |
| kernel time (steady) | 122 µs | **33 µs** (3.7x) |

The kernel was neither compute-bound (10% SM) nor DRAM-bound (9%) — it was
**L2-transaction bound**: the scattered stores generated 8x the necessary sectors.
Because it is one CTA per 16 KB chunk, the fix stages the permuted chunk in shared
memory and flushes it coalesced: coalesced global read → register butterfly →
**conflict-free smem scatter** → `__syncthreads()` → **coalesced global flush**.
The smem plane layout is padded to `npp + 1` words per plane; a warp writes
`s[sublane*pstride + col]` with `col` warp-uniform, so stride `pstride = npp+1`
(coprime to 32, since `npp` is a multiple of 32) puts all 32 lanes in distinct
banks. Decode is the mirror (coalesced load into padded smem → butterfly → coalesced
store); `s[sublane*pstride + i/32]` is exactly the value the scattered decode
gathered, so the butterfly (self-inverse) is unchanged and the output byte-identical.

Whole PFPL DAG **0.302 → 0.214 ms (1.41x)** from this one kernel; PSNR unchanged
(84.7564 dB), 16 bitshuffle stage tests + 35 stage tests pass, compute-sanitizer
clean. Default on; `FZ_BITSHUF_SMEM=0` forces the scattered kernel for A/B. Dynamic
smem is opt-in raised past 48 KB via `cudaFuncAttributeMaxDynamicSharedMemorySize`,
with a fallback to the scattered kernel if a chunk's `32*(npp+1)*4` bytes exceed the
device ceiling. Only the 4-byte (primary) path is staged; 1/2/8-byte are unchanged.

This is the "optimize the bottleneck before fusing" result from the PFPL roofline
study: bitshuffle was 38% of PFPL runtime and looked compute-bound from the stage
timing, but was actually a coalescing bug. It also validates the chunk-cooperative
(CTA-per-chunk, smem-resident) kernel shape a flexible fusion framework would target.

## CN-RZE-HELPERS — folding away the RZE strip/add-offset helper kernels

**Source:** `modules/coders/rze/rze_stage.cu` — `RzeStripFlagOp`, `rzePackKernel`.

The RZE forward path launched six device operations serially on one stream:
`rzeEncodeKernel` → sizes memcpy → `rzeStripFlagKernel` (`& 0x7FFFFFFF`) → CUB
`ExclusiveSum` → `rzeAddOffsetKernel` (`+ header_size`) → `rzePackKernel`. Per-kernel
duration (PFPL on CLDHGH, 1583 chunks, H100, ncu):

| kernel | µs | note |
|---|---|---|
| rzeEncodeKernel | 42.5 | compute-bound (72% SM) — the RZE compaction, hard to shrink |
| rzeStripFlagKernel | 3.1 | 7 blocks — trivial, pure overhead |
| DeviceScanInit + Scan | 6.6 | CUB offset scan (1583 elems) |
| rzeAddOffsetKernel | 2.9 | 7 blocks — trivial, pure overhead |
| rzePackKernel | 12.7 | copies the 4.5 MB packed payload — real memory work |
| **Σ GPU** | **67.8** | vs a measured 92 µs stage |

The strip and add-offset kernels are folded away: the flag strip becomes a
`thrust::transform_iterator` feeding `ExclusiveSum` (so the scan reads
flag-stripped sizes with no separate kernel and no `d_clean_dev_` array), and the
header offset is added inside `rzePackKernel` (`dst = header_off + offset[cid]`,
`sz = sizes[cid] & 0x7FFFFFFF`). The tail-size readback reads `d_sizes_dev_` and
masks/adds `header_size` on the host. Six device ops → four; one device scratch
array removed; byte-identical archive (22 RZE + 35 stage + 16 pipeline tests pass,
compute-sanitizer clean, PFPL PSNR unchanged 84.7564 dB).

**Honest payoff: small.** RZE stage 92 → 87 µs, PFPL DAG 0.214 → 0.212 ms (~1%).
The naive "stage 92 − Σ GPU 68 = 24 µs of launch latency" estimate was optimistic:
kernels queue asynchronously on the stream, so most of that gap is event-timing
granularity and encode-tail overlap, not exposed launch stalls — removing two tiny
kernels only recovers their ~6 µs of GPU time. The change stands as a
**simplification** (fewer kernels/arrays, cleaner offset handling) with a marginal
perf bonus, not a bottleneck fix. The RZE stage remains dominated by the
compute-bound encode (42 µs) and the memory-bound pack (13 µs); shrinking it
further means a faster compaction algorithm or fusing encode+pack, not trimming
helpers. Contrast CN-BSHUF-SMEM, where the bottleneck really was addressable.

## CN-PFPL-FUSE — chunk-cooperative fused PFPL proof (the ~2.2× ceiling holds)

**Source:** `profiling/pfpl_fuse_proof.cu` (`fzgmod-profile-pfplfuse`). Go/no-go for
the *second* fusion strategy — chunk-cooperative (one CTA per chunk, intermediates
in shared memory), the LC-style counterpart to the warp-register driver (CN-FUSE-*).

PFPL = `Quantizer(NOA,zigzag) → Difference(1-D chunk, negabinary) → Bitshuffle(ew4)
→ RZE(w1)`. The three post-quant stages all chunk at 16384 B = 4096 int32 = one
CTA, so one kernel does quant → diff+negabinary → bitshuffle → RZE-encode per chunk
with every intermediate in smem; the cross-chunk `scan + pack` tail is kept (RZE
offsets depend on all chunks). Device functions are reused verbatim
(`Zigzag::encode`, `Negabinary::encode`, `butterfly32` copied, and
`lc_detail::d_RZE`), and the
quantizer's `ebx2_r` is matched via a forced `value_base`, so the fused archive is
**byte-identical** to a manually-run staged 4-stage forward.

Measured (CLDHGH, n truncated to 1582 full chunks, eb=1e-4 NOA, H100):

| | staged (4-stage) | fused |
|---|---|---|
| archive / ratio | 4.458 MB / 5.54x | 4.458 MB / 5.54x (**byte-identical**) |
| compress wall | 0.208 ms | **0.103 ms (2.01x)** |
| compress DAG | 0.172 ms | — (fused path 0.103 wall) |
| fused encode kernel | — | 70.5 µs, **47% SM**, 11% DRAM |
| **achieved occupancy** | — | **92.6% warps active** |
| smem / block | — | **36 KB (= RZE-encode alone)** |

**The occupancy worry was unfounded.** The fused kernel reuses RZE's own two 16 KB
buffers (codes→bitshuffled in one, negabinary→RZE-`s_out` in the other) plus its
4 KB temp, so its footprint equals `rzeEncodeKernel` alone — 4 blocks/SM, 92.6%
achieved occupancy, no collapse. The roofline prediction (~2.2×, from CN reassessment
after the bitshuffle fix) holds up at ~2× measured; the fused encode is now
compute-bound (47% SM — the RZE compaction + quant, the irreducible floor), DRAM
down to 11% (the three intermediate round-trips are gone). compute-sanitizer clean.

**Why this matters for the framework:** it is the first working chunk-cooperative
fusion, and it confirms the general recipe — a fusable chunk-aligned chain
`predict/transform* → variable-length coder` collapses into `[CTA-per-chunk fused
kernel] + [scan + pack tail]`, reusing the coder's smem budget for free. Prototype
limits (not blockers): outlier-free quant fast path (verified `outlier_count==0` for
this data/config via the byte compare), whole-chunk only (n truncated to a multiple
of 4096; the partial-tail chunk needs the staged path's guard), and the fused-path
timing still pays a per-call CUB-temp `cudaMalloc` a persistent buffer would remove.

## CN-CHUNK-FUSE — composable device-op harness for chunk-cooperative fusion

**Source:** `modules/fused/chunk_fusion/chunk_fusion.cuh` (harness + ops),
`chunk_fusion.cu` (launcher), proof `profiling/pfpl_fuse_proof.cu`.

Generalizes the hand-coded PFPL fused kernel (CN-PFPL-FUSE) into a *composable*
mechanism — the first step toward automatic, pipeline-agnostic fusion (goal: a
future NVRTC path that generates the glue for any compatible chain).

**The taxonomy it encodes.** Fusion has two orthogonal axes:
- *Strategy* (execution template): **warp-register** (cuSZp — warp=block, registers,
  shuffles, no barriers; `fused_block/`) vs **chunk-cooperative** (LC/PFPL — CTA=chunk,
  shared memory, `__syncthreads`; this file). Determines the harness.
- *Role* (access pattern = `FusionAccess`): Map (quant), stencil/block-local (diff,
  lorenzo), fixed cooperative (bitshuffle), variable-length coder / sink (RZE, RRE,
  AdaptiveBitpack), unfusable. Determines how a stage's device-op slots in.

**The mechanism.** Each stage contributes a small `__device__` OP — `QuantInplaceZigzag`
(Map: global floats → codes), `DiffNegabinary` / `Bitshuffle32` (transforms:
smem→smem, ping-ponged), `RZECoder` / `RRECoder` (the swappable sink, uniform LC
signature `d_XXX(csize, in[CS], out[CS], temp[CS])`). The **harness** `chunk_fused_kernel
<QuantOp, Coder, Transforms...>` is stage-agnostic glue: load chunk → `Chain<Transforms...>`
ping-pongs the ops through two 16 KB smem buffers with the right syncs → coder sink →
emit per-chunk bytes+size. The cross-chunk scan+pack tail (RZEStage-format archive)
lives in the launcher. **Swapping the coder or transform set re-composes with no new
glue** — that is the whole point, and what an NVRTC path would later emit as source
(the ops stay hand-written headers; only the harness is generated).

**Generalization proven (CLDHGH, 1582 chunks, eb=1e-4 NOA, H100):** the *same* harness,
given `RZECoder` vs `RRECoder`, fuses PFPL-with-RZE and PFPL-with-RRE, each
**byte-identical** to its staged pipeline (RZE 4.458 MB @ 243 GB/s; RRE 4.695 MB @
239 GB/s — different ratio, correctly a different coder), compute-sanitizer clean.
Requires inplace-outlier quant (single "codes" port) so the chain is strictly linear;
the op encodes out-of-radius values as raw float bits inline. Partial-tail chunk is
handled (`Bitshuffle32` copies the sub-chunk through, matching the staged bitshuffle's
tail memcpy). Not yet wired to the planner/registry — that (chunk-cooperative group
detection + mapping a matched chain to the harness instantiation) is the next step,
then NVRTC codegen.

## CN-CHUNK-WIRE — wiring chunk-cooperative fusion into the planner/registry

**Source:** `getFusionSpec()` on Quantizer(inplace)/Difference/Bitshuffle/RZE/RRE,
`matchesPfpl`/`runPfpl` in `src/pipeline/fusion_registry.cpp`,
`QuantizerStage::primeComputedAbsEb`, `RZEStage::setFusedResult`, and
`RREStage::setFusedResult`.

Turns the chunk-fusion harness (CN-CHUNK-FUSE) into automatic fusion via
`FusionPolicy::Auto` — PFPL now fuses inside a real `Pipeline`, byte-identical and
round-trip-correct, for both RZE and RRE (`tests/pipeline/test_fusion_planner.cpp`
`Pfpl*`, 10/10 fusion tests, sanitizer clean, 16/16 pipeline + 35/35 stages).

Reuses the existing four-layer machinery (FusionSpec → planFusionGroups → registry
→ execution substitution; see CN-FUSE-EXEC). New pieces specific to the
chunk-cooperative strategy:
- **FusionSpec roles, block_size in BYTES.** Quantizer(inplace+zigzag, ABS/NOA) is
  `Map`; Difference(negabinary chunk) and Bitshuffle(ew4) are `BlockLocal{16384}`;
  RZE/RRE are `Cooperative{16384}` (the sink — bitshuffle must NOT report Cooperative
  or the planner terminates before the coder). The planner unifies block_size at
  16384 (bytes), a different unit from cuSZp's element counts (32/64) — fine because
  the registry `dynamic_cast` disambiguates the strategy.
- **NOA priming.** The fused runner bypasses `Quantizer::execute()`, so it calls
  `primeComputedAbsEb()` (runs the value-range scan, sets `computed_abs_eb_`/
  `computed_value_base_`) — the fused kernel's `ebx2_r` and the reused inverse/header
  both need it. (ABS-only `primeAbsEbForFusion` was the cuSZp version.)
- **Coder axis in the runner.** `runPfpl` detects RZE vs RRE by `dynamic_cast` and
  passes the matching `ChunkCoderKind` — the same registry entry fuses either.

**The decompress-sizing gotcha (cost an hour; general to any fused variable-length
coder).** `RZEStage::estimateOutputSizes()` in the inverse falls back to the
*compressed* input size when `cached_orig_bytes_ == 0`, and the fused runner had
skipped the forward `execute()` that normally sets it — so the inverse RZE output
buffer was sized to the compressed length (~393 KB) and the decode wrote the full
4 MB one byte past, a heap overflow only compute-sanitizer caught (the sticky CUDA
error surfaced two kernels later as a bogus "DifferenceStage launch: invalid
argument"). Fix: `setFusedResult(archive_bytes, orig_bytes)` also sets
`cached_orig_bytes_`. **General rule (extends CN-FUSE-EXEC's "prime forward state the
inverse needs"): a fused coder runner must set every field the *inverse* stage reads
to size its output, not just the forward output size.** Next: NVRTC codegen to emit
the harness for arbitrary compatible chains (done — CN-NVRTC-FUSE).

## CN-NVRTC-FUSE — runtime codegen of the chunk-cooperative harness (NVRTC)

**Source:** `modules/fused/chunk_fusion/nvrtc_chunk_fusion.{h,cpp}`,
`chunk_geometry.h`, the `chunk_fused_body` split in `chunk_fusion.cuh`, the
`FZ_FUSION_NVRTC` toggle + shared `packChunks` tail in `chunk_fusion.cu`.

Replaces the *compile-time* template instantiation of the fused kernel
(CN-CHUNK-FUSE picks `chunk_fused_kernel<Quant, Coder, Transforms...>` in C++) with
a *runtime* one: a `ChunkFusionSpec` (the stage-chain fingerprint — quant op,
transform ops, coder op, all by device-op **type name**) is turned into CUDA source
that wraps `chunk_fused_body<...>`, compiled with NVRTC, JIT-loaded via the CUDA
driver API, cached by (source, arch), and launched with `cuLaunchKernel`. Swapping
the coder or adding/removing a transform is now a **data change to the spec** — no
new template instantiation, no registry glue. The per-stage `__device__` ops stay
hand-written in `chunk_fusion.cuh`; only the composing glue is generated. Selected
by env `FZ_FUSION_NVRTC=1` inside `launchFusedChunkPfpl()` (off by default; the
template path is the fallback). Both paths share `packChunks` (the cross-chunk
scan/pack tail), so only the encode-kernel launch differs.

**Validated:** byte-identical archives to the template path for RZE **and** RRE
(`test_fusion_planner` `Pfpl*` under `FZ_FUSION_NVRTC=1`, 11/11), compute-sanitizer
clean on the JIT'd kernel, throughput parity (242 vs 248 GB/s RZE, 231 vs 240 GB/s
RRE — within run-to-run noise), 52/52 ctest unregressed on the default path. The
codegen contract (spec → template-arg list) is locked host-only by
`NvrtcCodegenComposesSpecOps`.

**What it took to make the op stack NVRTC-compilable (the real work):**
- **`backend/api.h` no-ops under `__CUDACC_RTC__`** — NVRTC compiles device code
  only and provides the intrinsics itself; pulling in the host `<cuda_runtime.h>`
  would fail. `atomics.h`/`warp.h` then reduce to plain `atomicAdd_block`/
  `__shfl_*_sync`/`__ballot_sync`, all NVRTC-native.
- **Split `chunk_geometry.h` out of `chunk_fusion.h`.** The device path must not pull
  `backend/types.h` (it names `cudaStream_t`/`cudaMemPool_t`, host-only). The geometry
  constants (CHUNK_BYTES/NELEM/TPB…) now live in a dependency-free header shared by the
  device harness, the host launcher, and the NVRTC codegen.
- **Minimal in-memory stdlib stubs.** NVRTC ships no `<cstdint>`/`<type_traits>`/
  `<cstddef>` on an include path; the op headers use only a tiny surface
  (`is_integral`/`is_signed`/`is_unsigned`/`make_unsigned` + the int aliases + `size_t`),
  supplied as three `nvrtcCreateProgram` headers. `assert` is an NVRTC builtin, so
  `<cassert>` is an empty shim.
- **`__builtin_memcpy` → `__float_as_uint`** in the quant op's outlier bit-cast — the
  former is not an NVRTC builtin; the latter is native to both nvcc and NVRTC and is
  bit-identical.
- **`__builtin_clz{,ll}` → guarded `FZ_CLZ32/64`** in `d_PRencode` (RARE/RAZE): under
  `__CUDACC_RTC__` use the `__clz`/`__clzll` device intrinsics NVRTC provides, else keep
  `__builtin_*` (which nvcc accepts in host+device passes). Bit-identical for the
  non-zero inputs used, so a fused RARE/RAZE chain matches the staged kernel. Same class
  of fix — surfaced when the RARE/RAZE coders became fusable (Phase D).

**Params ABI (Phase B, generalized).** The generated kernel takes one packed
`const uint8_t* params` blob — each op declares a POD `Params` (stateless ops use
`EmptyParams` → 0 bytes), `chunk_fused_body` hands each op its slice at a compile-time
offset (`OpParamBytes`/`SumParamBytes`), and the host packs the blob. So the codegen no
longer bakes in a per-pipeline param shape; any op set with any params composes. The
one canonical `chunk_fused_body` still backs both the template and NVRTC paths (no
drift). Interim: `launchFusedChunkPfpl` hardcodes PFPL's quant `Params` when packing;
the generic runner (Phase C) assembles the blob from each stage's `getFusedOp().params`.

**Prototype constraints (future work).** NVRTC resolves the op headers from the source
tree via `-I` paths baked in at configure time (`FZGMOD_NVRTC_INC_*`); a shipped build
would embed the headers instead. The registry `matchesPfpl` still hardcodes the 4-stage
shape — the codegen dissolves the *kernel* per-shape glue, not yet the *matcher*
(Phase C: route by declared `FusionStrategy`, assemble the chain from `getFusedOp()`).
