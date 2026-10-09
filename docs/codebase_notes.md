# Codebase notes

## CN-FSZ-TI-1: thread-independent AdaptiveLorenzo forward kernels (negative result)

2026-09-09, H100. Two attempts to make `FusedQuantAdaptiveLorenzoStage`'s forward
kernel thread-independent (TI), mirroring native FSZ's execution shape. Both were
byte-identical to the shipped CTA-cooperative kernel and both were 1.5-1.7x slower,
so neither was merged. The branches (`ti_predictor_wp1` at `f2e9919`,
`ti_predictor_arrays` at `c1a8b26`) were deleted on 2026-10-09; this entry is the
record.

**Motivation.** Native FSZ's compress kernel uses one-warp CTAs
(`FSZ_TBLOCK_SZ=32`) in which each thread serially owns 4 whole 256-element tiles in
local registers, with no shuffles and no shared memory beyond one per-CTA offset
counter. FZGM's kernel is CTA-cooperative: 256 threads per tile, warp shuffles,
`__syncthreads()`, 9 shared arrays, and a single-threaded final 4-variant cost
selection. ncu showed ours compute-bound (77% SM, 96% occupancy), so native wins by
doing less total work. The remaining FZGM-vs-native FSZ gap was ~3.7x after M1
(`FusedQuantAdaptiveLorenzo`) shipped.

**Common scaffolding.** A new kernel sat alongside the CTA kernel and wrote the same
dense per-tile buffers (`modes_dense`, `means_dense`, `flags`, `residuals`), so the
downstream CUB scan, compaction and AdaptiveBitpack needed no change and the archive
format was untouched. Dispatch was gated by `FZ_AL_TI=1` (default off) with
`FZ_AL_TI_TPT` for tiles per thread, parsed once in a magic-static env config. A
non-serialized `Config::TIDispatch {Auto, ForceCTA, ForceTI}` and
`ti_tiles_per_thread` let one gtest binary compare both kernels without racing the
env cache. Tests (`MatchesCTAByteIdentical`, `MatchesCTAAcrossTilesPerThread`,
`TIForwardRoundTrip`) swept `blocks_per_tile` in {1,2,4,8} x order-2 on/off x
centering on/off x TPT {2,4,8}. `blocks_per_tile = 16` is excluded because the CTA
kernel's `__launch_bounds__(256,8)` already forbids 512-thread tiles. The kernels were
derived from this file's own CTA kernel; no FSZ kernel body was copied (only the
thread mapping idea).

**WP1: scalar-state, three-pass TI kernel**
(`fused_quant_adaptive_lorenzo_forward_kernel_ti<T, TILES_PER_THREAD>`). Per-block
`CoderStats` (`all`/`rest`/`first` for LZ1 and LZ2) were kept as scalars, costed with
`blockCost()` and reset at each 32-element block, so register use did not scale with
`blocks_per_tile`. The cost was three passes per tile: (1) a full-tile scan for
per-block stats and the mean sum, (2) a 32-element reread of block 0 to cost the
centered variants once the mean was known, (3) a full-tile pass to emit the chosen
residuals.

| NYX/temperature, ncu `--set full` | CTA | TI TPT=2 | TI TPT=4 | TI TPT=8 |
|---|---:|---:|---:|---:|
| Duration | 1.84 ms | 3.67 ms | 3.45 ms | 3.43 ms |
| Compute (SM) | 77.40% | 9.86% | 10.26% | 10.26% |
| Memory throughput | 64.45% | 63.57% | 66.07% | 64.85% |
| DRAM throughput | 17.23% | 17.69% | 18.03% | 17.59% |
| Registers/thread | 32 | 48 | 52 | 53 |
| Achieved occupancy | 95.99% | 52.45% | 46.30% | 24.36% |

End to end (`fzgmod-cli -b --report-json`, `device_ms.min`, 20 reps x 3 invocations,
`fsz_fused_quant.toml`): NYX/temperature CTA 2.906-2.908 ms vs TI 4.585-4.604 ms
(1.58x slower); HACC/vx 6.545-6.550 ms vs 9.925-9.994 ms (1.52x slower). Compressed
sizes were identical (NYX 24,421,280 B, CR 21.98).

**WP1b: register-array TI kernel**
(`fused_quant_adaptive_lorenzo_forward_kernel_ti_arrays<T, TPT, BPT>`, one-warp CTAs,
`BPT` = `blocks_per_tile` in {1,2,4,8} as a compile-time argument). Both predictors'
`CoderStats` were held in `CoderStats[BPT]` arrays (block loop fully unrolled so they
scalarize; tile loop `#pragma unroll 1` so the arrays are reused, not multiplied by
TPT). WP1's block-0 reread was removed algebraically, because centering changes only
residual 0 (LZ1) or residuals 0 and 1 (LZ2):

- quantized value `q0`, mean `mu`, `c0 = q0 - mu`, `cm0 = |c0|`;
- centered LZ1 block 0 = `{rest1 | cm0, rest1, cm0}`;
- keep `d1[1]` and the OR of LZ2 magnitudes over elements 2..31 (`tail2`); with
  `cm1 = |d1[1] - c0|`, centered LZ2 block 0 = `{tail2 | cm1 | cm0, tail2 | cm1, cm0}`;
- centered variant cost = uncentered total - block-0 cost + centered block-0 cost +
  `sizeof(T)` for the mean; the minimum of the four variants picks the mode.

Prediction and mode analysis then read each input once; a second traversal emits the
selected residual stream (the modular stage must materialize it for the separate
AdaptiveBitpack launch).

| NYX, TPT=8, ncu | WP1 scalar TI | WP1b array TI | CTA |
|---|---:|---:|---:|
| Kernel duration | 3.43 ms | 3.24 ms | 1.84 ms |
| Compute (SM) | 10.26% | 8.66% | 77.40% |
| Memory throughput | 64.85% | 68.50% | 64.45% |
| DRAM throughput | 17.59% | 17.85% | 17.23% |
| Registers/thread | 53 | 76 | 32 |
| Local load/store | n/a | 0 / 0 | n/a |
| Achieved occupancy | 24.36% | 23.83% | 95.99% |

End to end: NYX 2.904-2.910 ms (CTA) vs 4.420-4.440 ms (1.53x slower); HACC
6.546-6.549 ms vs 10.705 ms (1.64x slower). NYX TPT sweep: 4.670 ms (2), 4.975-5.020
ms (4), 4.420-4.440 ms (8). `cuobjdump` confirmed 76 registers and zero local bytes,
which rules out spilling. Two variants were also tried and reverted, both
byte-identical: `float4` loads (NYX 4.548 ms, 78 registers) and a one-raw-input-pass
hybrid that stored LZ1 residuals during analysis and transformed them in place for LZ2
(NYX 4.778 ms).

**Root cause.** Both TI kernels are latency-bound on uncoalesced per-thread global
access: each thread walks its own contiguous tile, so a warp touches 32 different
tiles per instruction. Loads issued 8,388,608 requests for 268,435,456 sectors, i.e.
32 sectors per request with 4 useful bytes per 32-byte sector. Native FSZ has the
same uncoalesced pattern (19 sectors/request measured) but tolerates it, because its
single-pass, 67-register kernel does enough arithmetic per load to hide the latency
and fuses prediction, selection and final coding in one kernel. FZGM's modular
boundary forces a dense residual stream between AdaptiveLorenzo and AdaptiveBitpack,
which adds traffic without adding compute. Removing WP1's block-0 reread gained only
~5.5% of kernel time.

**What a future attempt would need.** More TPT or register tuning will not help.
Either (a) a coalescing-aware thread-to-data mapping that is no longer "thread owns
a contiguous tile", or (b) predictor+coder fusion that consumes thread-local predictor
state directly instead of materializing residuals. Either must keep byte identity with
the CTA kernel and pass the same ncu checks (sectors/request, local memory, SM%).
The same lesson appears in AdaptiveBitpack's scalar-coder experiment
(`FZ_AB_FORCE_SCALAR`, removed): matching native's thread independence without its
coalescing and predictor-coder locality loses.

## CN-WARP-PROBE-1: tiled plain-chain adaptive-probe threshold

2026-09-09, H100, large-data corpus at rel_range 1e-3. The TI-vs-warp-cooperative
probe threshold `adaptive_thresh = 16.0` was a 2-point fit on 1-D cuSZp2 HACC data
(xx r~14: TI wins 393 vs 236 GB/s; vx r~17.6: warp-cooperative wins). The probe always
estimates rate with a flattened serial Lorenzo1D kernel (`ti_rate_probe_kernel`), so for
tiled 2-D/3-D plain (non-outlier) chains it sent CESMATM-3D/T and SCALE-LETKF/T to TI at
about half the single-pass throughput (25-26% of native cuSZp3 instead of 48-50%).

`adaptive_thresh_tiled = 1.4` was calibrated from `FZ_DEBUG_PROBE=1` avg_r on the 10 tiled
corpus fields. The data are not separable by one global cut: CESMATM-3D/CLOUD wants
single-pass at avg_r = 1.749, below EXAFEL/data, which wants TI at avg_r = 3.641. 1.4
routes 9/10 fields correctly, costs EXAFEL/data ~5% (331.9 -> 315.6 GB/s), and fixes
CESMATM-3D/{T,U,CLOUD} and SCALE-LETKF/{T,U,QV} (+15-96%). Corpus-wide (24 cells, cuSZp2 +
cuSZp3 plain/outlier vs native): compress throughput range 25.0-84.2% -> 45.7-80.5% of
native, geomean 0.558 -> 0.601; output byte-identical (dispatch only). Resolving the
EXAFEL/CLOUD ambiguity needs a dimension-aware or tile-shape-aware probe, not a constant.
Full analysis: paper_organizer `projects/FZGM/investigations/fusion/fused_execution_paths_map.md`.

## CN-WARP-TILESHAPE-1: compile-time tile shape in the tiled warp predictors

2026-09-08, H100, cuSZp3 2-D/3-D warp-cooperative path. Native-vs-FZGM instruction
profiling showed the fused path executing 10.8x more instructions per element than native
(14.53 vs 1.35), ALU-bound (65.9% ALU pipe) where native is memory-bound (68% DRAM).
`lx = local % tx` and `ly = local / tx` are per-element and were runtime divisions against
a `uint32_t` field. Making the tile shape a template argument (both shipped shapes are
powers of two: 8x8 2-D, 4x4x4 3-D) turns them into mask/shift: 1.95B -> 1.73B executed
instructions (-11%), NYX/temperature 177.7 -> 191.6 GB/s (+7.8%), byte-identical.
Corpus-wide (95 cells vs native, geomean): cuszp3_plain compress unchanged at 0.45x,
decompress 0.28x -> 0.39x; cuszp3_outlier 0.64x/0.46x -> 0.66x/0.56x compress/decompress.
A per-tile `tile_ctx()` hoist of the tile-index divisions was measured separately and had
no effect (<0.1% instructions; the compiler already CSEs it), so it was not kept. These
numbers predate f64 specialization (`2f5fe00`); the port onto the `Real`-templated
predictors (2026-10-09) was re-verified for correctness, not re-timed.

## CN-F64-SPECIALIZATION-1: f64 and predictor-free fixed-mode diagnostics

2026-10-05, H100. These are local performance diagnostics, not publication
rows. ABS bounds were precomputed from the full-field range at relative bound
1e-3. Each CLI invocation ran 20 in-process iterations; the diagnostic reducer
discarded the first three, then compared device-time medians. The discarded
inverse iterations include cold NVRTC compilation. This differs from Benchkit's
five retained repetitions plus one warmup with independent phase executions.

| Field / pipeline | Auto compression speedup over Staged | Auto decompression speedup over Staged |
|---|---:|---:|
| brown-p2-plain | 1.382x | 1.736x |
| brown-p2-outlier | 1.737x | 2.098x |
| miranda-p2-plain | 1.369x | 1.674x |
| miranda-p2-outlier | 1.727x | 2.044x |
| miranda-p3-plain | 0.800x | 1.025x |
| miranda-p3-outlier | 0.884x | 1.147x |
| miranda-p3-fixed | 0.921x | 1.504x |

The new fixed 1-D route has the following warmed diagnostic results:

| Field / pipeline | Compression | Decompression | Selected forward schedule |
|---|---:|---:|---|
| brown-fixed-f64 | 0.606x | 1.384x | single_pass |
| exaal-fixed-f32 | 0.511x | 1.096x | two_pass |

Both fixed 1-D compression cases regress. The registry admits legal
implementations and does not choose by measured profitability. This extension
therefore closes an implementation gap without establishing a compression
performance benefit; include these regressions if reporting its coverage.

The cuSZp2 results show the intended recovery of staged boundary costs. Tiled
cuSZp3 compression is slower on this MIRANDA field; removing boundaries does
not guarantee a win. Extra double quantization work, the separate input
validation scan/synchronization, and the selected warp schedule are plausible
costs, but this diagnostic does not establish their individual contributions.
Float64 thread-independent dispatch remains unavailable. No native-relative
claim follows from these Auto/Staged ratios. Some cuSZp2 inverse sample CVs
are 6–8%; run the publication timing gate before including any measurements.

Raw samples and the reduction script are in
`build/f64-auto/performance-check/`. Fixed 1-D follow-up artifacts are in
`build/f64-auto/fixed-validation/` and `fixed-performance-check/`.

## CN-F64-SPECIALIZATION-2: targeted publication rerun scope

The historical Auto source has 330 successful, bound-satisfying, timing-reliable
f64 cuSZp rows: MIRANDA 7, S3D 11, NWCHEM 1 and BROWN 3 fields, times three
bounds and five variants. Add 36 f32 fixed 1-D cells for six HACC and six
EXAALT fields. Each cell reports both compression and decompression.

Existing native/FZGM complete-case compression coverage is 278 coordinates:
p2 plain/outlier 66 each, p3 fixed 33, plain 62, outlier 51. Native S3D p3
fixed severe bound failures exclude all 33 coordinates; native plain/outlier
validity removes another 4/15. Extending FZGM does not repair native failures.
Keep coverage and valid populations distinct, and do not cherry-pick faster
new Auto rows. Twelve of the 278 previous fallbacks were fixed 1-D structural
mismatches, rather than dtype mismatches.

Historical precomputed-range H100 cell cadence is 6.6–7.2 seconds, including
process, artifact, and quality work. Estimate 36–40 minutes raw for 330 Auto
cells, budget 45–75 minutes practical; the extra 36 f32 cells add about 5–10
minutes. Matched Off+Auto gives 732 cells including f32 fixed: budget roughly
1.5–2.5 hours, subject to current per-process JIT and timing retries. This is an
estimate, not a reserved runtime or completed campaign.

Recommend an additive campaign pinned to a clean new source revision, starting
with a 12–24-cell calibration overlap. Preserve dataset hashes, effective
bounds, coder geometry, and original manifests. Reuse native only where
protocol/provenance and quality/timing gates permit; refresh Off for the small
affected matrix to establish current-source comparisons. Apply Benchkit verify
and all validity/timing/complete-case gates before regenerating paper tables.
Do not combine old NOA timings with new precomputed-range ABS timings without
explicit protocol reconciliation. Historical staged/native populations remain
useful but are not automatically interchangeable with this new build.

Correctness gate: 60/60 CTest executables passed. Fixed-mode memcheck and
racecheck passed under ordinary and forced single-pass (`FZ_SP_BPW=16`) dispatch;
the f64 suite also passed memcheck after the shared validation change. Full
BROWN f64 and EXAALT f32 checks have identical codec payloads, normalized codec
metadata, and reconstructions between Staged and Auto. Full-file hashes differ
in capacity metadata, as with the original f64 checks. EXAALT has the same
12 ppm maximum-bound overshoot in both arms; this satisfies Benchkit's existing
0.1% relative acceptance tolerance and is recorded explicitly in the artifact.
It is not a new error introduced by specialization.

Publication protocol source: `compression_benchmarking/configs/experiments/
range_precomputed_spec.yaml`; FZGM sessions are
`rq2-range-precomputed-merged-h100-{off,auto}-20260930b`, as pinned by
`specialization_native_performance_full_corpus.json`. Native cuSZp2/3 evidence
comes from Sep. 9 with dimension-matched fixed-mode native additions on Sep. 28;
those bounds were resolved outside timing and reconciled in the merged session.
