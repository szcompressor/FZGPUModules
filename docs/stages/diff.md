# DifferenceStage {#stage_diff}

**Header:** `modules/predictors/diff/diff.h`  
**Class:** `fz::DifferenceStage<T, TOut = T>`  
**Category:** Predictor / transform (lossless)

---

## What it does

- **Forward (compression):** first-order differences — `output[i] = input[i] - input[i-1]`,
  `output[0] = input[0]`.
- **Inverse (decompression):** cumulative sum.

When `TOut != T`, the stage writes the forward deltas in negabinary form into the
unsigned output type, and the inverse path decodes negabinary before the prefix sum.
This is equivalent to `DifferenceStage<T>` followed by a `NegabinaryStage`, but
fused into one kernel.

Output is the same byte size as input (`sizeof(T) == sizeof(TOut)` is enforced).

---

## Template parameters

| Parameter | Constraint |
|---|---|
| `T` | Numeric type (input / output when `TOut == T`, see available instantiations) |
| `TOut` | Defaults to `T`. When different: unsigned counterpart of signed `T` (negabinary fusion) |

## Available instantiations

Single-parameter (no negabinary fusion):
- `DifferenceStage<float>`
- `DifferenceStage<double>`
- `DifferenceStage<uint8_t>`
- `DifferenceStage<uint16_t>`
- `DifferenceStage<uint32_t>`
- `DifferenceStage<int32_t>`
- `DifferenceStage<int64_t>`

Negabinary-fused pairs (`<signed, unsigned>`):
- `DifferenceStage<int8_t, uint8_t>`
- `DifferenceStage<int16_t, uint16_t>`
- `DifferenceStage<int32_t, uint32_t>`
- `DifferenceStage<int64_t, uint64_t>`

Using any other combination will result in a linker error. Common choices: `DifferenceStage<int32_t, uint32_t>` (after quantizer codes), or `DifferenceStage<int32_t>` (plain delta coding).

---

## Stage settings

| Setting | Purpose | Notes |
|---|---|---|
| `setChunkSize(bytes)` | Reset delta at chunk boundaries | 0 = no chunking |

```cpp
diff->setChunkSize(16384);   // bytes; 0 = no chunking
```

---

## Chunking

`setChunkSize(bytes)` makes differences and cumulative sums reset at each chunk
boundary.  Each chunk is independent: `output[chunk_start] = input[chunk_start]`
(previous = 0 implied).

This is required for the PFPL pipeline where 16 KB chunks flow independently
through `BitshuffleStage` and `RZEStage`.  Chunk size must be a positive multiple
of `sizeof(T)`.  Default is 0 (no chunking — whole array is one context).


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
