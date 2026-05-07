# BitpackStage {#stage_bitpack}

**Header:** `modules/coders/bitpack/bitpack_stage.h`  
**Class:** `fz::BitpackStage<T>`  
**Category:** Coder (lossless)

---

## What it does

Packs each element using only its low `nbits` bits into a dense byte stream.

- **Forward:** `T[] → uint8_t[]` — ceil(n × nbits / 8) bytes.
- **Inverse:** `uint8_t[] → T[]` — unpacks elements, zero-extending to full width.

Most useful after a Lorenzo predictor stage where small delta values only need a few
bits of representation.

---

## Template parameter

| Parameter | Constraint |
|---|---|
| `T` | Unsigned integer type (see available instantiations below) |

## Available instantiations

Only these types are compiled and linked:
- `BitpackStage<uint8_t>`
- `BitpackStage<uint16_t>`
- `BitpackStage<uint32_t>`

Using any other type will result in a linker error. Most common: `BitpackStage<uint32_t>` (to match typical quantizer code width).

---

## Stage settings

| Setting | Purpose | Notes |
|---|---|---|
| `setNBits(nbits)` | Bits per element | Power of two, 1..`8 * sizeof(T)` |

```cpp
pack->setNBits(nbits);
```

`nbits` must be a power of two in `[1, 8 x sizeof(T)]`.
Default is `8 * sizeof(T)` (identity, no compression). There is no automatic
bit-width selection; choose a value that matches your code range.

| `T` | Allowed `nbits` |
|---|---|
| `uint8_t` | 1, 2, 4, 8 |
| `uint16_t` | 1, 2, 4, 8, 16 |
| `uint32_t` | 1, 2, 4, 8, 16, 32 |

Violations throw `std::invalid_argument` at `setNBits()` time.

---

## Typical pipeline

```cpp
p.setDims(nx);
auto* quant = p.addStage<QuantizerStage<float, int32_t>>();
auto* lrz   = p.addStage<LorenzoStage<int32_t>>();
auto* bpack = p.addStage<BitpackStage<uint32_t>>();

bpack->setNBits(16);   // pack small Lorenzo deltas into 16 bits

p.connect(lrz,   quant, "codes");
p.connect(bpack, lrz);
p.finalize();
```
