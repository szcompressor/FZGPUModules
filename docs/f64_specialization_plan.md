# Float64 support for existing warp specializations

## Goal and measured scope

Extend the registered warp-register strategy to execute its existing cuSZp-shaped
pipelines on `float64` inputs. Keep the current `float32` path byte-identical.
This closes a specific Auto coverage gap; it does not promise parity with native
cuSZp throughput or add a new predictor, coder, or execution strategy.

The full-corpus H100 artifact records **278 valid compression field/bound
coordinates** falling back across the five cuSZp variants:

| Variant | f64 compression coordinates |
|---|---:|
| cuSZp2 plain | 66 |
| cuSZp2 outlier | 66 |
| cuSZp3 fixed | 33 |
| cuSZp3 outlier | 51 |
| cuSZp3 plain | 62 |
| **Total** | **278** |

These counts use the valid, available rows in
`specialization_native_performance_full_corpus.csv`; unavailable and invalid
rows are excluded. Of these, 266 have the previously supported three-stage
graph and fell back for dtype; 12 fixed-mode BROWN/NWCHEM coordinates instead
have a two-stage 1-D graph that needed the extension below. They describe missing
implementations, not cases where a performance model rejected a specialization. The source artifact and the
published summary are in the paper evidence tree (see [evidence pointers](#evidence-pointers)).
Native-relative performance remains an empirical question after correctness and
coverage are established.

The first implementation covers only spans already recognized by the
warp-register matcher: linear ABS/NOA quantization, an existing 1-D Lorenzo or
64-element tiled Lorenzo predictor, and the existing int32 AdaptiveBitpack
coder. This includes cuSZp2 plain/outlier and cuSZp3 plain/outlier shapes. The
cuSZp3 fixed 2-D/3-D identity-predictor form is covered. The follow-up extension
also admits its natural two-stage 1-D Quantizer→AdaptiveBitpack graph in both
precisions: an internal identity policy emits the quantized code and performs no
inverse prediction. It derives geometry from the coder and adds no DAG stage.
The existing supported coder blocks are 32, 64, and 128 elements; the paper preset
retains block 32. FSZ, cuSZ, and cuSZ-Hi remain out of scope because no registered
strategy currently matches those pipelines.

## Implementation boundary

The warp policies quantize input values inline while producing int32 predictor
deltas. Before this change, the registered implementation assumed float32 input and
int32 codes. Add a dtype flag (`use_double`) to `WarpFusionSpec`, and have the
generated warp policy choose typed input loaders and the corresponding
quantization and dequantization arithmetic. Preserve the float32 arithmetic and its
existing policy interface.

Keep predictor geometry and float32 parameter POD layouts unchanged. For
float64, pass the resolved reciprocal as a separate `double` launcher argument
and use it directly in the typed policy; do not serialize it through the
existing float `inv2eb` field. The inverse path likewise needs a typed double
output and a double dequantization step (`2 * computed_abs_eb`). This avoids
rounding either scale through float while allowing the predictor POD to remain
the same geometry contract.

Fused quantization must preserve the staged `QuantizerStage<double,uint32_t>`
linear semantics: use a double product and round-to-nearest-even; reject a
non-finite scaled value and reject a rounded integer outside signed int32 range
before converting it. Never rely on a narrowing cast or wraparound. The coder
still consumes int32 residuals, so this plan does not widen predictor codes.
The opt-in `linear_high_precision` policy remains staged in this implementation;
the fusion declarations already exclude it, and its tighter-bound scan and
reconstruction reserve require separate parity work.

The thread-independent schedule stays disabled for float64. Its rate probe,
policy parameters, and memory layout are tuned for the existing float32
implementation; f64 correctness and the warp-cooperative schedules come first.
Do not infer performance parity from successful specialization installation.

## Work phases and gates

### 1. Extend the existing warp strategy

Generalize the strategy matcher to admit float64 input while retaining the
int32-code requirement. Add typed compression launch plumbing, double scale
passing, generated float64 quant/predictor policies, and typed inverse
dequantization/output. Keep thread-independent dispatch unavailable for f64.
Reject non-finite or out-of-int32-range quantized values with the same explicit
failure behavior as staged linear quantization.

### 2. Establish correctness and preserve float32 behavior

Before broad benchmark use, add and run focused checks for:

- fused versus staged codec payload byte identity on supported 1-D and tiled 2-D/3-D
  inputs, including partial final regions and quantization values near rounding
  and int32 limits;
- forward and inverse specialization installation for each supported f64
  graph shape, with expected staged fallback for strict high precision and
  unsupported code widths;
- round-trip reconstruction and serialized-file decode using the ordinary
  pipeline/file path;
- existing float32 byte-identity and behavior checks; and
- Compute Sanitizer on the device-code changes.

Installation and byte identity are correctness gates, not performance claims.
Any changed archive, bound violation, unexpected installation, or sanitizer
finding blocks the benchmark phase until resolved.

### 3. Validate on a small real f64 set

Run a small, representative set of real f64 fields spanning 1-D and tiled
multidimensional paths, with matched Staged and Auto runs. Confirm identical
archives/reconstructions and valid error bounds, and inspect timing reliability
and specialization metadata. This is a smoke/validation step; it does not
support a native-parity claim or replace corpus-scale evidence.

### 4. Add a provenance-pinned benchmark follow-up

Only after the real-field gate passes, define an additive benchmark manifest
covering all 330 f64 field/bound/variant coordinates and the corresponding
decompression population (278 have existing native-valid complete cases),
plus 36 f32 fixed 1-D coordinates if updating that coverage. See
[CN-F64SPEC-2](codebase_notes.md)
for the protocol and wall-time estimate. Pin source revisions, binaries, host/GPU provenance,
and logical cell identities; retain historical manifests and results. Apply the
usual validity, error-bound, timing, and verification gates before interpreting
Auto/native ratios. Report valid coverage separately from completed execution
and publication-valid results.

## Explicit exclusions

- **PFPL:** current PFPL evidence excludes f64 MIRANDA/NWCHEM because its
  in-place outlier representation needs 64-bit codes, while the quantizer has
  no `double`→`uint64_t` instantiation. The chunk runner and quant operation
  also assume float input and raw float-bit outliers. Supporting it requires a
  separate decision about 64-bit sentinel representation, chunk intermediates,
  bitshuffle/coder widths, and file compatibility; it is not part of this warp
  extension.
- **FSZ, cuSZ, cuSZ-Hi:** these have no registered matching execution strategy
  today (FSZ uses tile selection; cuSZ and cuSZ-Hi include global coding paths).
  Float64 alone cannot make them eligible.
- **Strict linear high precision:** keep `linear_high_precision=true` staged
  until its scan, tightened bound, reconstruction arithmetic, and inverse
  semantics are separately integrated and checked.
- **Performance selection or native parity:** Auto installs a legal registered
  implementation; this work does not add cost prediction or promise that the
  resulting kernel matches native cuSZp.

## Current status

The first implementation is complete on branch
`feature/f64-auto-specializations`, based on `6bb4613`, with changes left
uncommitted for review. Three Luna agents handled the backend, regression tests,
and scope audit; the parent integrated and validated their work on H100.

- Release/sm_90 build succeeded with CUDA 12.9.
- The full 59-test CTest suite passed; the final f32/f64 fusion rechecks also
  passed after preserving the custom float predictor interface.
- Seven focused f64 tests cover ordinary and forced single-pass execution,
  archive/reconstruction byte identity, partial regions, high bins, ties to even,
  nonfinite/overflow rejection, staged fallbacks, file decode, and the legacy
  custom float inverse policy interface.
- Compute Sanitizer memcheck passed for the f64 suite and all 51 existing fusion
  tests. F64 memcheck and racecheck also passed with `FZ_SP_BPW=16` forcing the
  single-pass path for eligible shapes; ordinary f64 racecheck passed too.
- Seven matched real-field graph pairs passed at NOA `1e-3`: BROWN
  `sample_r_B_0.5_26` (268,435,464 bytes) with cuSZp2 plain/outlier, and MIRANDA
  `density` (301,989,888 bytes) with cuSZp2 plain/outlier and cuSZp3
  plain/outlier/fixed. Every Auto pair installed forward and inverse groups,
  met the bound, and matched Staged codec payload and reconstructed bytes.

The complete FZM files may differ in recorded buffer capacity and the resulting
header checksum. Those execution-specific fields were normalized for the
real-field metadata comparison; the remaining header bytes, codec payloads,
and reconstructed data were compared exactly. This work does not claim full
container byte identity across execution policies.

The tests also exposed an existing priming bug: the fused path bypasses the
staged quantizer's overflow-flag reset but still runs its post-sync hook. Fused
priming now clears that flag; the f64 validation prepass remains authoritative.

Real-field reports, exact hashes, the validation script, and source provenance
are retained in `build/f64-auto/real-validation/`. No compressed or reconstructed
field files are retained. These are correctness smoke results with three timing
repetitions per arm, not a publication campaign or native-relative performance
claim. Historical manifests, results, and manuscript text were not changed.

The f64 validation prepass adds an input scan and synchronization. A next
performance task is to measure its share of total cost and consider reporting
range failures from an existing fused pass instead, while retaining rejection
semantics. The additive provenance-pinned corpus campaign remains pending.

## Evidence pointers

- Full-corpus source: `paper_organizer/projects/FZGM/evidence/publication/specialization/specialization_native_performance_full_corpus.csv`
- Full-corpus interpretation and older mixed-dtype counts:
  `paper_organizer/projects/FZGM/evidence/publication/specialization/specialization_table_v1.md`
- Dataset dtypes and shapes:
  `paper_organizer/papers/FZGM/tables/evaluation_datasets.tex`
- Warp strategy matcher and typed runner boundary: `src/pipeline/fusion_registry.cpp`
- Warp generated policies and current quantization representation:
  `modules/fused/fused_block/warp_fusion.cuh`
- Warp spec and launcher interface:
  `modules/fused/fused_block/nvrtc_warp_fusion.h` and
  `modules/fused/fused_block/nvrtc_warp_fusion.cu`
- Staged linear quantization, overflow handling, and dequantization:
  `modules/quantizers/quantizer/quantizer.cu`
- PFPL f64 exclusion and reason:
  `paper_organizer/projects/FZGM/EVIDENCE.md`
- Chunk fusion's float32/outlier-width assumptions:
  `modules/fused/chunk_fusion/chunk_fusion.cuh` and
  `src/pipeline/fusion_registry.cpp`

## Predictor-free fixed 1-D follow-up

The two-stage forward/inverse extension is complete for f32/f64 and ordinary
ABS/NOA int32 quantization. It preserves coder geometry and adds no predictor
stage. Strict high precision, uint16 codes and block sizes outside the existing
32/64/128 warp range remain staged. The full suite now passes 60/60 executables;
fixed-mode ordinary/forced-single-pass memcheck and racecheck are clean. Full
BROWN f64 and EXAALT f32 payload/reconstruction comparisons pass. Artifacts and
provenance are in `build/f64-auto/fixed-validation/`; measured performance
limitations and the targeted paper rerun scope are in
[CN-F64SPEC-1/2](codebase_notes.md). The campaign and manuscript update
remain pending.
