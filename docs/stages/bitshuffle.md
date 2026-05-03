# BitshuffleStage {#stage_bitshuffle}

**Header:** `modules/shufflers/bitshuffle/bitshuffle_stage.h`  
**Class:** `fz::BitshuffleStage`  
**Category:** Transform / shuffler (lossless)

---

## What it does

GPU bit-matrix transpose over fixed-size chunks.  Given a chunk of N elements each
W bits wide, the forward pass groups all N values' bit-plane k together, producing
W bit-planes of N bits each.  This concentrates sign bits and exponent bits into
contiguous regions, dramatically improving the compressibility of floating-point or
integer data for downstream byte-oriented coders like `RZEStage`.

Output is the same byte size as input (size-preserving transform).

---

## Key setters

```cpp
bshuf->setBlockSize(16384);          // chunk size in bytes (default 16384)
bshuf->setElementWidth(sizeof(T));   // element width: 1, 2, 4, or 8 (default 4)
```

**Constraints:**
- `block_size` must be a positive multiple of `1024 × element_width`.  The default
  of 16384 satisfies this for all supported element widths.
- `element_width` must be 1, 2, 4, or 8.  Both are enforced at `execute()` time.

---

## Alignment requirement

`BitshuffleStage` requires its input to be a multiple of `block_size` bytes.  The
pipeline pads automatically when connected to a chunked upstream stage
(`DifferenceStage` with matching `chunk_size`, or `RZEStage`).

---

## Typical pipeline

```cpp
auto* bshuf = p.addStage<BitshuffleStage>();
auto* rze   = p.addStage<RZEStage>();

bshuf->setElementWidth(sizeof(uint16_t));   // match upstream code type

p.connect(bshuf, upstream_stage);
p.connect(rze,   bshuf);
p.finalize();
```

Set `element_width` to match the element type flowing in from upstream:
- Codes from `LorenzoQuantStage<float, uint16_t>` → `setElementWidth(2)`
- Codes from `QuantizerStage<float, uint32_t>` → `setElementWidth(4)`
