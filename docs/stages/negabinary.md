# NegabinaryStage {#stage_negabinary}

**Header:** `modules/transforms/negabinary/negabinary_stage.h`
**Class:** `fz::NegabinaryStage<TIn, TOut>`
**Category:** Transform (lossless)

---

## What it does

Element-wise negabinary encoding (base -2). Converts a signed integer stream
into unsigned integers of the same width. Compared to zigzag, negabinary tends
to push more high-order zeros after differencing, which can help bitshuffle +
byte-oriented coders.

- Forward: signed -> unsigned negabinary codes
- Inverse: unsigned negabinary codes -> signed values

Output is the same byte size as input.

---

## Template parameters

| Parameter | Constraint |
|---|---|
| `TIn` | Signed integer (see available instantiations below) |
| `TOut` | Unsigned counterpart of `TIn` |

## Available instantiations

Only these pairs are compiled and linked:
- `NegabinaryStage<int8_t, uint8_t>`
- `NegabinaryStage<int16_t, uint16_t>`
- `NegabinaryStage<int32_t, uint32_t>`
- `NegabinaryStage<int64_t, uint64_t>`

Using any other pair will result in a linker error. Most common: `NegabinaryStage<int32_t, uint32_t>` (to match typical code widths).

---

## Stage settings

No stage-specific setters. This stage is purely type-driven.

---

## Ports

Single input -> single output; element count and byte size are unchanged.

| Direction | Port | Type |
|---|---|---|
| Input | "output" (default) | `TIn[n]` |
| Output | "output" | `TOut[n]` |

---

## Typical pipeline

```cpp
auto* diff = p.addStage<DifferenceStage<int32_t>>();
auto* nb   = p.addStage<NegabinaryStage<int32_t, uint32_t>>();
auto* bsh  = p.addStage<BitshuffleStage>();

bsh->setElementWidth(sizeof(uint32_t));

p.connect(nb,  diff);
p.connect(bsh, nb);
p.finalize();
```
