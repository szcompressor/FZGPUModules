# DifferenceStage {#stage_diff}

**Header:** `modules/predictors/diff/diff.h`  
**Class:** `fz::DifferenceStage<T, TOut = T, Mode = FusionMode::NEGABINARY>`  
**Category:** Predictor / transform (lossless)

---

## What it does

- **Forward (compression):** first-order differences — `output[i] = input[i] - input[i-1]`,
  `output[0] = input[0]`.
- **Inverse (decompression):** cumulative sum.

When `TOut != T`, the stage writes the forward deltas in fused form into the
unsigned output type — negabinary (`FusionMode::NEGABINARY`, LC's DIFFNB) or
zigzag/sign-magnitude (`FusionMode::ZIGZAG`, LC's DIFFMS) — and the inverse path
decodes that transform before the prefix sum. This is equivalent to
`DifferenceStage<T>` followed by a `NegabinaryStage`/`ZigzagStage`, but fused into
one kernel. Neither mode dominates universally: negabinary tends to produce denser
zero runs at high bit-planes for smooth, symmetric-around-zero residuals, but
zigzag can win on other residual distributions — both are exposed so a pipeline
search can pick per-dataset, mirroring LC's own DIFFNB/DIFFMS split.

Output is the same byte size as input (`sizeof(T) == sizeof(TOut)` is enforced).

---

## Template parameters

| Parameter | Constraint |
|---|---|
| `T` | Numeric type (input / output when `TOut == T`, see available instantiations) |
| `TOut` | Defaults to `T`. When different: unsigned counterpart of signed `T` (fused transform) |
| `Mode` | `FusionMode::NEGABINARY` (default) or `FusionMode::ZIGZAG`. Ignored when `TOut == T` |

## Available instantiations

Single-parameter (no fusion):
- `DifferenceStage<float>`
- `DifferenceStage<double>`
- `DifferenceStage<uint8_t>`
- `DifferenceStage<uint16_t>`
- `DifferenceStage<uint32_t>`
- `DifferenceStage<int32_t>`
- `DifferenceStage<int64_t>`

Negabinary-fused pairs (`<signed, unsigned>`, `Mode` defaults to `NEGABINARY`):
- `DifferenceStage<int8_t, uint8_t>`
- `DifferenceStage<int16_t, uint16_t>`
- `DifferenceStage<int32_t, uint32_t>`
- `DifferenceStage<int64_t, uint64_t>`

Zigzag-fused pairs (`<signed, unsigned, FusionMode::ZIGZAG>`):
- `DifferenceStage<int8_t, uint8_t, FusionMode::ZIGZAG>`
- `DifferenceStage<int16_t, uint16_t, FusionMode::ZIGZAG>`
- `DifferenceStage<int32_t, uint32_t, FusionMode::ZIGZAG>`
- `DifferenceStage<int64_t, uint64_t, FusionMode::ZIGZAG>`

Using any other combination will result in a linker error. Common choices: `DifferenceStage<int32_t, uint32_t>` (after quantizer codes, negabinary fusion), or `DifferenceStage<int32_t>` (plain delta coding).

### TOML

```toml
[[stage]]
name        = "diff"
type        = "Difference"
input_type  = "int32"
output_type = "uint32"
fusion_mode = "zigzag"    # or "negabinary" (default); ignored when input_type == output_type
chunk_size  = 16384
```

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

## Acknowledgements

The `DifferenceStage` kernel follows the `d_DIFFNB` algorithm from the
**LC/PFPL framework** (Burtscher et al., Texas State University, BSD-3-Clause).

> Noushin Azami, Alex Fallin, Brandon Burtchell, Andrew Rodriguez,
> Benila Jerald, Yiqian Liu, Anju Mongandampulath Akathoott, and Martin Burtscher.
> *LC framework for synthesizing high-speed parallel lossless and
> error-bounded lossy data compression and decompression algorithms for CPUs and GPUs.*
> https://github.com/burtscher/LC-framework

See `THIRD_PARTY.md` for the full license text.

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
