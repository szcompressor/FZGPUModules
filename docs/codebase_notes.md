# Codebase notes

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
