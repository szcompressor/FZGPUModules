# RZEStage {#stage_rze}

**Header:** `modules/coders/rze/rze_stage.h`  
**Class:** `fz::RZEStage`  
**Category:** Coder (lossless)

---

## What it does

Recursive Zero-byte Elimination.  Operates on raw byte streams (typically
`BitshuffleStage` output).  Each chunk is processed in up to 4 recursive levels:

- **Level 1 (ZE):** compact non-zero bytes; emit an N/8-byte zero bitmap.
- **Levels 2–4 (RE):** compact non-repeated bytes of the previous bitmap.

Because bit-shuffled scientific data has many zero byte-planes, RZE can compress
those planes very aggressively.

---

## Output stream layout

```
[uint32_t: original byte count]
[uint32_t: num_chunks]
[uint32_t × num_chunks: per-chunk compressed sizes (high bit set → chunk stored raw)]
[compressed chunk data ...]
```

---

## Key setters

```cpp
rze->setChunkSize(16384);   // bytes; must be a multiple of 4096 (default 16384)
rze->setLevels(4);          // recursion depth 1–4 (default 4)
```

---

## CUDA Graph compatibility

| Pass | Graph-compatible? |
|---|---|
| Compression (forward) | Yes |
| Decompression (inverse) | **No** |

The inverse path reads the stream header with two blocking D2H `cudaMemcpy` calls
before it can compute per-chunk decode offsets.  This is intentional: graph-captured
decompression of the same buffer every iteration has no practical use case, while
graph-captured compression (new data each iteration) provides meaningful latency savings.

Do not include `RZEStage` in a pipeline with `enableGraphMode(true)` if the
pipeline will be used for decompression.

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
