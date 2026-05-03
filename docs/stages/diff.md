# DifferenceStage {#stage_diff}

**Header:** `modules/predictors/diff/diff.h`  
**Class:** `fz::DifferenceStage<T, TOut = T>`  
**Category:** Predictor / transform (lossless)

---

## What it does

- **Forward (compression):** first-order differences — `output[i] = input[i] - input[i-1]`,
  `output[0] = input[0]`.
- **Inverse (decompression):** cumulative sum.

When `TOut != T`, negabinary encoding is fused into the final write of the forward
kernel, and negabinary decoding is the first step of the inverse kernel.  This avoids
a separate `ZigzagStage` or `NegabinaryStage` in the pipeline.

Output is the same byte size as input (`sizeof(T) == sizeof(TOut)` is enforced).

---

## Template parameters

| Parameter | Constraint |
|---|---|
| `T` | Any numeric type (input / output when `TOut == T`) |
| `TOut` | Defaults to `T`. When different: must be the unsigned counterpart of a signed `T` of the same width (negabinary fusion). |

Valid negabinary-fused pairs: `<int8_t, uint8_t>`, `<int16_t, uint16_t>`,
`<int32_t, uint32_t>`, `<int64_t, uint64_t>`.

---

## Chunking

`setChunkSize(bytes)` makes differences and cumulative sums reset at each chunk
boundary.  Each chunk is independent: `output[chunk_start] = input[chunk_start]`
(previous = 0 implied).

This is required for the PFPL pipeline where 16 KB chunks flow independently
through `BitshuffleStage` and `RZEStage`.  Chunk size must be a positive multiple
of `sizeof(T)`.  Default is 0 (no chunking — whole array is one context).

---

## Key setters

```cpp
diff->setChunkSize(16384);   // bytes; 0 = no chunking
```

---

## Common instantiations

| Instantiation | Use case |
|---|---|
| `DifferenceStage<int32_t, uint32_t>` | After `QuantizerStage` codes (negabinary fused) |
| `DifferenceStage<int32_t>` | After `QuantizerStage` codes (plain delta, no negabinary) |
| `DifferenceStage<float>` | Delta coding of raw float data |

---

## Typical pipeline

```cpp
auto* quant = p.addStage<QuantizerStage<float, uint32_t>>();
auto* diff  = p.addStage<DifferenceStage<int32_t, uint32_t>>();
auto* bshuf = p.addStage<BitshuffleStage>();

diff->setChunkSize(16384);
bshuf->setElementWidth(sizeof(uint32_t));

p.connect(diff,  quant, "codes");
p.connect(bshuf, diff);
p.finalize();
```
