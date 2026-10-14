# OutlierCorrectStage {#stage_outlier_correct}

**Header:** `modules/coders/outlier_correct/outlier_correct_stage.h`
**Class:** `fz::OutlierCorrectStage<Reconstructor>`
**Shipped instantiation:** `fz::Cdf97OutlierCorrectStage` (`modules/coders/cdf97_outlier_correct/`)
**Category:** Coder (sparse reconstructed-domain residual correction)
**Release status:** Experimental in v2.0.

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

The correction approach computes a trial reconstruction (dequantize and
inverse-transform a copy), then stores a sparse list of indices and `float`
residuals for pixels whose error exceeds the requested bound. Decompression
adds those residuals to the trial values. This follows the reconstructed-domain
correction idea used by SPERR; see [THIRD_PARTY.md](../../THIRD_PARTY.md) for
algorithmic attribution.

Floating-point residual storage and addition can lose low bits through
cancellation. The implementation has no final error verification or fallback,
and nonfinite values are not covered by a supported accuracy contract. Current
tests therefore do not establish a mathematically exact or unrestricted strict
pointwise guarantee. The complete wavelet composition remains experimental.

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
kernel) is the one instantiation that ships. Adding a correction
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

A different reconstruction policy is possible, but requires a separate accuracy
contract and validation of its quantization, trial reconstruction, and correction
arithmetic. Structural similarity to a wavelet or predictive pipeline alone does
not establish an exact bound or justify adding this correction stage.

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
