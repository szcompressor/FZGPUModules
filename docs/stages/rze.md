# RZEStage {#stage_rze}

**Header:** `modules/coders/rze/rze_stage.h`  
**Class:** `fz::RZEStage` — no template parameters  
**Category:** Coder (lossless)

**Common instantiation:**
```cpp
auto* rze = p.addStage<fz::RZEStage>();
```

---

## What it does

Recursive Zero-byte Elimination.  Operates on raw byte streams (typically
`BitshuffleStage` output).  Each chunk is processed in up to 4 recursive levels:

- **Level 1 (ZE):** compact non-zero bytes; emit an N/8-byte zero bitmap.
- **Levels 2–4 (RE):** compact non-repeated bytes of the previous bitmap.

Because bit-shuffled scientific data can have many zero byte-planes, RZE can compress
those planes very aggressively.

---

## Stage settings

```cpp
rze->setChunkSize(16384);   // bytes; must be a multiple of 4096 (default 16384)
rze->setLevels(4);          // recursion depth 1–4 (default 4)
```

---

## Alignment requirement

Requires input to be a multiple of `chunk_size` bytes.  The pipeline pads
automatically when `BitshuffleStage` upstream uses a matching `block_size`.

---

## Typical pipeline

```cpp
auto* bshuf = p.addStage<BitshuffleStage>();
auto* rze   = p.addStage<RZEStage>();

rze->setChunkSize(16384);
rze->setLevels(4);

p.connect(rze, bshuf);
p.finalize();
```

---

## Stream layout (forward output)

```
[uint32_t: original byte count]
[uint32_t: num_chunks]
[uint32_t x num_chunks: per-chunk compressed sizes (high bit set -> chunk stored raw)]
[compressed chunk data ...]
```
