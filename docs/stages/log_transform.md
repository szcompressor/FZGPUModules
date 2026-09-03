# LogTransformStage {#stage_log_transform}

`LogTransformStage<TInput>` — `modules/transforms/log_transform/`

Turns a **point-wise relative** error bound into a plain **absolute** one, so an
ordinary ABS quantizer downstream delivers the relative guarantee. Implements
the transformation scheme of X. Liang, S. Di, D. Tao, Z. Chen and F. Cappello,
"An efficient transformation scheme for lossy data compression with point-wise
relative error bound", IEEE CLUSTER 2018, pp. 179–189.

---

## The identity

```
|x - x_hat| / |x| <= delta
  <=>  x_hat / x                 in [1-delta, 1+delta]
  <=>  log2|x_hat| - log2|x|     in [log2(1-delta), log2(1+delta)]
```

A **multiplicative** bound on `x` is an **additive** bound on `log2|x|`. The
interval is asymmetric and `|log2(1-delta)| > log2(1+delta)`, so the binding
side is the upper one:

```
e = log2(1 + delta)
```

Symmetric bins of half-width `e` leave the negative side tighter than required,
wasting roughly `delta² / 2` of achievable relative error. Negligible for small
`delta`; reclaiming it would need asymmetric bins.

---

## Why this stage exists

Three ways to ask for a relative bound, and until this stage none of them gave
you both the guarantee and good compression:

| Approach | Per-element guarantee? | Compresses? |
|---|---|---|
| `LorenzoQuantStage` / `GInterpStage` with `PREL` | **No** | Yes |
| `QuantizerStage` with `REL` | Yes | **No** — no predictor in front |
| `LogTransformStage` → predictor → ABS quantizer | Yes | Yes |

`QuantizerStage` REL already does log-space quantization, but *at* the
quantizer, on raw values. Its codes still carry the field's full spatial
redundancy. Putting the log **upstream of the predictor** is what gets both —
that is the paper's contribution, and it is why this is a transform rather than
a new compressor: it bolts onto an existing absolute-error pipeline.

---

## Pipeline

```cpp
Pipeline p(bytes, MemoryStrategy::PREALLOCATE, 4.0f);
p.setDims(nx, ny, nz);

auto* lg = p.addStage<LogTransformStage<float>>();
lg->setErrorBound(1e-3f);          // delta — the relative bound you want

auto* lq = p.addStage<LorenzoQuantStage<float, uint16_t>>();
lq->setErrorBound(lg->quantizerErrorBound());   // NOT delta — see below
lq->setErrorBoundMode(ErrorBoundMode::ABS);
lq->setQuantRadius(4096);
lq->setZigzagCodes(true);
p.connect(lq, lg, "output");

auto* huf = p.addStage<HuffmanStage<uint16_t>>();
huf->setBklen(8192);
p.connect(huf, lq, "codes");
p.finalize();
```

The `signs`, `outlier_vals` and `outlier_idxs` ports are left unconnected and
become pipeline outputs automatically, exactly like `QuantizerStage`'s outlier
ports.

---

## Ports

Forward (4 outputs):

| Index | Name | Type | Contents |
|---|---|---|---|
| 0 | `output` | `TInput[n]` | `log2(abs(x))`, or the log floor at outlier positions |
| 1 | `signs` | `uint8[ceil(n/8)]` | bit `i` set ⇒ element `i` is negative |
| 2 | `outlier_vals` | `TInput[k]` | original values at outlier positions |
| 3 | `outlier_idxs` | `uint32[k]` | indices of outlier positions |

Inverse: those same four buffers → `TInput[n]`.

The outlier **count** is not a port. It lives in a stage-private 4-byte device
scratch, is D2H'd in `postStreamSync()`, and is serialized into the FZM stage
header — the same mechanism `QuantizerStage` uses.

---

## Settings

| Setting | Default | Purpose |
|---|---|---|
| `setErrorBound(delta)` | `1e-3` | The point-wise relative bound you want |
| `setThreshold(t)` | `0` (off) | `abs(x) < t` ⇒ lossless outlier. Trades a bigger outlier list for a narrower, more compressible log range |
| `setOutlierCapacity(c)` | `0.05` | Fraction of `n` reserved for outliers |
| `quantizerErrorBound()` | — | **Read this and hand it to the downstream quantizer** |
| `minimumErrorBound()` | — | Smallest `delta` float32 log space can honour (~1.4e-6) |

---

## Limitations

**The downstream eb is not wired automatically.** A stage cannot reach across
the DAG to configure another stage, so `quantizerErrorBound()` is yours to
propagate. Passing the raw `delta` to the quantizer instead yields a far looser
relative bound (by roughly `1/log2(1+delta)`, ~693× at `delta = 1e-3`) with no
error and no warning. This is the single easiest way to misuse the stage.

**Sign changes hurt.** The sign is stripped into a separate bit-plane, so a sign
flip between neighbours is invisible to the downstream predictor and costs a raw
bit per element. Single-signed fields (density, pressure, magnitude) are the good
case; fields oscillating about zero are not. This is intrinsic to the approach.

**Near-zero values become outliers.** `log2|x| → -∞`, so zeros, denormals,
inf/NaN and anything below `threshold` are stored losslessly. Fields with a lot
of near-zero mass pay for it in the outlier list — watch the outlier percentage
in the stage's `DEBUG` log.

**float32 only.** `LogTransformStage<float>` is the only instantiation. Below
`minimumErrorBound()` the log-space round-trip slack would consume the entire
budget, and `execute()` throws rather than emit a stream that violates its bound.

**Outlier overflow is not a bounded-error event.** Unlike a quantizer, a dropped
outlier here means the value is simply gone and the reconstruction is wrong at
that position. The stage `WARN`s loudly; the outlier reserve is also floored at
8 slots so small inputs cannot silently round their reserve to zero.

---

## Error budget

The composed bound is the quantizer's plus the transform's own float32 rounding.
`quantizerErrorBound()` therefore returns `log2(1+delta) - kLogRoundTripSlack`
(`1e-6` in log2 units), reserving the remainder for `exp2(log2(x))`. The forward
pass verifies each element's actual round-trip and escalates anything worse than
budgeted to a lossless outlier — a stronger guarantee than the paper's
analysis-based one, and the reason the measured violation count is zero rather
than merely small. The slack costs under 0.1% of the bound at `delta = 1e-3`.

---

## TOML

```toml
[[stage]]
type = "LogTransform"
error_bound = 1e-3
threshold = 0.0
outlier_capacity = 0.05
```

---

## See also

- \ref stage_quantizer — `ErrorBoundMode::REL`, the exact-but-uncompressed alternative
- \ref stage_lorenzo_quant — `ErrorBoundMode::PREL`, and why it is not a relative bound
- `examples/eb_mode_analysis.cpp` — measures all of the above on your own data

## Acknowledgements

The log-space transformation follows:

> Xin Liang, Sheng Di, Dingwen Tao, Zizhong Chen, and Franck Cappello.
> *An efficient transformation scheme for lossy data compression with
> point-wise relative error bound.* IEEE CLUSTER 2018, pp. 179–189.

This is an independent implementation of the paper's mathematical transform.
No reference implementation was vendored or copied; the sign plane, outlier
handling, round-trip verification, and DAG integration are FZGPUModules code.
See `docs/acknowledgements.md` for the provenance summary.
