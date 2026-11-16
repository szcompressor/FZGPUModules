# GolombRiceStage {#stage_golomb_rice}

**Header:** `modules/coders/golomb_rice/golomb_rice_stage.h`  
**Class:** `fz::GolombRiceStage<T>` — `T` is `int16_t` or `int32_t`  
**Category:** Coder (lossless)

**Common instantiation:**
```cpp
auto* gr = p.addStage<fz::GolombRiceStage<int32_t>>();
gr->setChunkSize(16384);   // the only supported value
```

---

## What it does

A chunk-local Golomb-Rice entropy coder for signed integer streams, aimed at
predictor residuals (most values near zero, a long thin tail). Each value is
zigzag-mapped to unsigned and split by a per-chunk parameter `k` into a quotient
(`u >> k`, written as that many one bits and a zero stop bit) and `k` raw remainder
bits, so small values cost about `k + 1` bits.

- **Exact per-chunk `k`.** For every candidate `k`, the encoder computes the exact
  bit cost of the chunk and picks the smallest. There is no global histogram or
  codebook, unlike Huffman and ANS.
- **Escape for outliers.** A unary run is capped at 24 ones, then the full zigzag
  value follows as raw bits. One large value can't blow up a chunk.
- **Raw fallback.** A chunk that wouldn't shrink is stored raw and flagged.
- **Parallel decode.** Each chunk's payload is split into 16 byte-aligned restart
  intervals, decoded by one warp per chunk with one lane per interval.

The stage zigzags internally, so feed it signed residuals directly; an upstream
zigzag transform would double-encode.

---

## Ports

| Direction | Input | Output |
|---|---|---|
| Forward | `T` residual stream | byte stream (`output`) |
| Inverse | byte stream | `T` residual stream |

---

## Stage settings

```cpp
gr->setChunkSize(16384);   // bytes; only 16384 is supported (execute() throws otherwise)
```

## TOML

```toml
[[stage]]
name = "coder"
type = "GolombRice"
data_type = "int32"     # "int16" or "int32" (default "int32")
chunk_size = 16384      # default 16384; the only supported value
```

Presets: `examples/presets/diffplain_golomb_rice.toml` (fused chain),
`examples/presets/lorenzo_golomb_rice.toml` (experimental).

---

## Alignment requirement

Requires input to be a multiple of `chunk_size` bytes; the pipeline pads
automatically.

---

## Stream layout

```
[uint32 original byte count][uint32 num_chunks]
[uint32 × num_chunks: compressed size per chunk; high bit = stored raw]
[per chunk: uint32 k][uint32 × 16 restart-interval offsets][packed bits]   (or raw bytes)
```

The 5-byte stage header holds `chunk_size` (uint32) and the element `DataType`
(INT16 or INT32), so an archive decodes without knowing `T` in advance.

---

## Graph capture

Forward (compress) is CUDA-graph capturable. The inverse is not: decoding reads the
stream header with blocking device-to-host copies before it launches.

---

## Pipeline Specialization

Compress fuses as a chunk-cooperative segment codec for `T = int32_t` and
`chunk_size = 16384`, for example `Quantizer → Difference (plain) → GolombRice`.
The fused encoder produces the same bytes as the staged one, so the ordinary
inverse decodes either. Decode is not fused.
