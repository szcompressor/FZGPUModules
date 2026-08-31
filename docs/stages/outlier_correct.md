# OutlierCorrectStage {#stage_outlier_correct}

**Header:** `modules/coders/outlier_correct/outlier_correct_stage.h`
**Class:** `fz::OutlierCorrectStage<Reconstructor>`
**Shipped instantiation:** `fz::Cdf97OutlierCorrectStage` (`modules/coders/cdf97_outlier_correct/`)
**Category:** Coder (lossless w.r.t. the codes it's paired with; turns a
reported error bound into a guaranteed one)

---

## What it does

`Transform -> QuantizerStage(linear/ABS) -> Coder` quantizes transform
COEFFICIENTS directly. For a transform whose synthesis gain varies by
decomposition level — CDF 9/7 is the motivating case — a uniform
coefficient-domain threshold does **not** translate to a uniform bound on the
RECONSTRUCTED FIELD's pointwise error: measured misses of up to 2.7x the
requested bound on real CDF 9/7 data (see `memory/speck_gpu_design.md` sec.9
in the repo, not shipped in docs). A candidate fix — scale each coefficient's
quantization step by its level's synthesis-filter gain — was tried and
**rejected**: it makes the max error *worse*, because many coefficients
across levels jointly influence any given pixel, so bounding each one's
isolated worst case does not bound their sum.

The fix that actually works, matching what native SPERR's own `Outlier_Coder`
does for CDF 9/7 (and generalizes to any reversible transform): quantize
normally; separately compute what the reconstruction WOULD be (dequantize +
inverse-transform a copy); every pixel whose error exceeds the bound gets an
EXACT correction value in a sparse (index, value) list, applied as the final
step of decompress. This gives a mathematically exact guarantee, not a
calibrated approximation. See [THIRD_PARTY.md](../../THIRD_PARTY.md) for the
port relationship to NCAR/SPERR (algorithmic attribution only — no SPERR code
was used).

---

## Genericity: the `Reconstructor` policy

Everything in this class — diffing, sparse pack/apply, config, serialization,
port shape — is transform-agnostic. The only transform-specific step is
"given dequantized coefficients, produce the trial reconstruction" — that's
`Reconstructor::applyInverseTransform()`. A `Reconstructor` must provide:

```cpp
struct MyReconstructor {
    static constexpr StageType kStageType = StageType::...;
    static std::string name() { return "..."; }
    // In-place: d_coeffs_inout holds dequantized coefficients on entry, the
    // trial reconstruction on return. n = nx*ny*max(nz,1) elements.
    static void applyInverseTransform(float* d_coeffs_inout, int nx, int ny, int nz,
                                       cudaStream_t stream);
};
```

`Cdf97Reconstructor` (wrapping [`Cdf97Stage`](cdf97.md)'s existing inverse-DWT
kernel) is the one instantiation that ships. Adding a bound-guarantee
pipeline for another reversible transform — a Lorenzo or interpolation
predictor, for instance — is writing one small policy struct like it, not a
new stage. See **When to use this in a different pipeline** below.

---

## Port shape and why it needs `Pipeline::bindExternalInput()`

This stage needs BOTH the original raw field (to compute corrections
against, at compress time) AND the dequantized codes (to reconstruct a trial
value from, in both directions). The raw field is bound directly to input
port 0 via `Pipeline::bindExternalInput()` — no duplicate-copy node needed.

- **Forward inputs:** `[raw_field, codes]`
- **Forward outputs:** `[correction (archived leaf), codes (passthrough -> coder)]`
- **Inverse outputs:** `[corrected field, codes (passthrough)]` — per the DAG's
  bijective inverse contract, inverse output *k* reconstructs forward input *k*
- **Inverse inputs:** `[archived correction stream, coder's decoded codes]`

The inverse-transform that recovers the trial/candidate reconstruction runs
identically in both directions (compress-time to detect outliers,
decompress-time to reconstruct before applying corrections) — it is **not**
a forward/inverse pair in the `Stage` sense, which is why it's called
directly via the `Reconstructor` policy rather than delegated to another
DAG-wired stage's own inverse.

---

## Scope

`float` coefficients only. ABS-mode linear quantization only; `error_bound`
here **must equal** the paired `QuantizerStage`'s own `error_bound` — set
both from the same value when building the pipeline. `Cdf97Reconstructor`
is 2-D only (throws if the pipeline's third dimension is > 1).

---

## Typical pipeline (SPERR-style, `Cdf97OutlierCorrectStage`)

```cpp
p.setDims(nx, ny, 1);

auto* dwt = p.addStage<Cdf97Stage<float>>();          // pure source, auto-discovered

auto* quant = p.addStage<QuantizerStage<float, uint32_t>>();
quant->setErrorBound(bound);
quant->setErrorBoundMode(ErrorBoundMode::ABS);
quant->setLinearMode(true);

auto* corr = p.addStage<Cdf97OutlierCorrectStage>();
corr->setErrorBound(bound);         // MUST match quant's bound
p.bindExternalInput(corr);          // corr.input[0] = raw field, before the connect() below

auto* speck = p.addStage<Speck2DStage>();

p.connect(quant, dwt);
p.connect(corr, quant, "codes");    // corr.input[1] = codes
p.connect(speck, corr, "codes");    // Speck2D consumes corr's codes passthrough

p.setPrimarySource(corr);           // decompress() returns corr's corrected field, not dwt's own
p.finalize();
```

Or via TOML — see `examples/presets/sperr_gpu.toml` for the complete,
runnable preset.

---

## When to use this in a different pipeline/context

Any pipeline where a quantized, coefficient-domain (or residual-domain)
threshold does not tightly bound the FINAL reconstructed value is a
candidate — not just wavelet transforms. The concrete condition is
**cascading/multi-level reconstruction**: whenever a later value's
reconstruction depends on an *already-approximated* neighbor rather than
directly on the original signal, per-coefficient error can compound in a way
a flat quantizer threshold doesn't capture.

`GInterpStage` (multi-level spline interpolation prediction, the codebase's
cuSZ-Hi-style predictor) has exactly this shape: coarse-level points are
reconstructed first and used to predict finer-level points, so a finer
level's prediction error is relative to an *already-reconstructed*
(imperfect) coarse value, not the untouched original — the same "errors
propagate through multiple levels" property that makes CDF 9/7 need this
mechanism, for a structurally different reason (auto-regressive prediction
vs. wavelet synthesis-filter gain). A `GInterpReconstructor` policy
(dequantize the codes and re-run `GInterpStage`'s own multi-level
reconstruction pass) would give that pipeline the same kind of exact,
worst-case guarantee this page describes for CDF 9/7 — genuinely useful
whenever a paper claims a bound but the actual multi-level pipeline can only
report one after the fact.

By contrast, a plain single-order `LorenzoStage`/`LorenzoQuantStage` predicts
every point directly from ALREADY-DECODED (not approximated) neighbors within
one pass — its coefficient-domain quantization step already bounds pointwise
error exactly, by construction, with no compounding across levels. This
mechanism would add cost there for no correctness benefit; it's specifically
for pipelines whose reconstruction is genuinely multi-pass/hierarchical.

---

## TOML

```toml
[[stage]]
name = "correct"
type = "Cdf97OutlierCorrect"
error_bound = 1e-4
inputs = [
  { from = "__external__" },
  { from = "quant", port = "codes" }
]
```

`error_bound` is the only user-facing key. `dims` come from the pipeline
(`setDims()`/the TOML `[pipeline]` table), not from this stage's own config.
