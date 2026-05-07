# ZigzagStage {#stage_zigzag}

**Header:** `modules/transforms/zigzag/zigzag_stage.h`
**Class:** `fz::ZigzagStage<TIn, TOut>`
**Category:** Transform (lossless)

---

## What it does

Element-wise zigzag encoding (two's complement to magnitude-sign). Converts a
signed integer stream into unsigned integers of the same width so small
magnitudes map to small codes.

- Forward: signed -> unsigned zigzag codes
- Inverse: unsigned zigzag codes -> signed values

Output is the same byte size as input.

---

## Template parameters

| Parameter | Constraint |
|---|---|
| `TIn` | Signed integer (see available instantiations below) |
| `TOut` | Unsigned counterpart of `TIn` |

## Available instantiations

Only these pairs are compiled and linked:
- `ZigzagStage<int8_t, uint8_t>`
- `ZigzagStage<int16_t, uint16_t>`
- `ZigzagStage<int32_t, uint32_t>`
- `ZigzagStage<int64_t, uint64_t>`

Using any other pair will result in a linker error. Most common: `ZigzagStage<int32_t, uint32_t>` (to match typical code widths).

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
auto* zz  = p.addStage<ZigzagStage<int32_t, uint32_t>>();
auto* rle = p.addStage<RLEStage<uint32_t>>();

p.connect(rle, zz);
p.finalize();
```
