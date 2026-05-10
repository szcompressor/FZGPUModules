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
| `setNBits(nbits)` | Bits per element | Power of two, 1..`8 * sizeof(T)`; ignored when auto-detect is on |
| `setAutoDetect(bool)` | GPU scan to pick `nbits` automatically | Disables CUDA Graph compatibility while active |

### Manual bit-width

```cpp
pack->setNBits(nbits);
```

`nbits` must be a power of two in `[1, 8 × sizeof(T)]`.
Default is `8 * sizeof(T)` (identity, no compression).

| `T` | Allowed `nbits` |
|---|---|
| `uint8_t` | 1, 2, 4, 8 |
| `uint16_t` | 1, 2, 4, 8, 16 |
| `uint32_t` | 1, 2, 4, 8, 16, 32 |

Violations throw `std::invalid_argument` at `setNBits()` time.

### Auto-detect mode

```cpp
pack->setAutoDetect(true);
```

When enabled, forward execute scans the input for its maximum value using
`cub::DeviceReduce::Max` and selects the smallest valid power-of-two `nbits`
that covers it.  The chosen `nbits` is written into the compressed header so
the inverse pass unpacks correctly without any out-of-band configuration.

After `compress()`, `getNBits()` reflects the detected value.

Scratch buffers for the scan are allocated through the pipeline's memory pool
(with a transparent `cudaMalloc` fallback in vGPU / pool-fallback mode), so all
device memory remains tracked by the pipeline.

**CUDA Graph incompatibility:** auto-detect requires a device-to-host transfer
and stream synchronization to read the max value, making it incompatible with
CUDA Graph capture. `isGraphCompatible()` returns `false` while auto-detect is
on.  If you know the bit-width ahead of time, use `setNBits()` instead to keep
graph capture available.

**Output size estimate:** `estimateOutputSizes()` returns the worst-case (full
input size) when auto-detect is enabled, so `PREALLOCATE` mode reserves
sufficient space regardless of the detected `nbits`.

---

## Typical pipeline

### Manual `nbits`

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

### Auto-detect `nbits`

```cpp
p.setDims(nx);
auto* quant = p.addStage<QuantizerStage<float, uint16_t>>();
auto* lrz   = p.addStage<LorenzoStage<int16_t>>();
auto* bpack = p.addStage<BitpackStage<uint16_t>>();

bpack->setAutoDetect(true);   // let the stage pick the tightest nbits at runtime

p.connect(lrz,   quant, "codes");
p.connect(bpack, lrz);
p.finalize();

p.compress(d_in, n_bytes, stream);
// bpack->getNBits() now holds the detected value (e.g. 4 for small deltas)
```
