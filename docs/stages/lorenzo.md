# LorenzoStage {#stage_lorenzo}

**Header:** `modules/predictors/lorenzo/lorenzo_stage.h`  
**Class:** `fz::LorenzoStage<T>`  
**Category:** Predictor (lossless)

---

## What it does

Plain integer Lorenzo predictor.  Lossless, size-preserving.

- **Forward (compression):** compute per-element deltas from spatial neighbors.
- **Inverse (decompression):** prefix-sum to reconstruct original values.

Supports 1-D, 2-D, and 3-D layouts.  Typically placed **after** a `QuantizerStage`
in cuSZp-style pipelines (float → quant → Lorenzo → bitpack), where it operates on
the quantization codes rather than raw floating-point data.

---

## Template parameter

| Parameter | Constraint |
|---|---|
| `T` | Signed integer: `int8_t`, `int16_t`, `int32_t`, `int64_t` |

Common instantiation: `LorenzoStage<int32_t>`.

---

## Ports

Single input → single output; element type and size are unchanged.

| Direction | Port | Type |
|---|---|---|
| Input | `"output"` (default) | `T[n]` |
| Output | `"output"` | `T[n]` |

Connection from a quantizer upstream uses the `"codes"` port of the quantizer:

```cpp
p.connect(lrz, quant, "codes");
```

---

## Dimension setup — critical ordering rule

Same rule as `LorenzoQuantStage`: call `p.setDims()` **before** `addStage()`, or
call `stage->setDims()` directly after adding.

```cpp
p.setDims(nx, ny, nz);
auto* lrz = p.addStage<LorenzoStage<int32_t>>();
```

---

## Typical pipeline (cuSZp-style)

```cpp
p.setDims(nx);
auto* quant = p.addStage<QuantizerStage<float, int32_t>>();
auto* lrz   = p.addStage<LorenzoStage<int32_t>>();
auto* bpack = p.addStage<BitpackStage<uint32_t>>();

p.connect(lrz,   quant, "codes");
p.connect(bpack, lrz);
p.finalize();
```
