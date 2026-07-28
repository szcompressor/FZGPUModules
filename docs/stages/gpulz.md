# GPULZStage {#stage_gpulz}

**Header:** `modules/coders/gpulz/gpulz_stage.h`
**Class:** `fz::GPULZStage` — no template parameters
**Category:** Coder (lossless)

**Common instantiation:**
```cpp
auto* gpulz = p.addStage<fz::GPULZStage>();
gpulz->setChunkSize(2048);  // bytes; 1024, 2048, or 4096 (default 2048)
gpulz->setWordSize(4);      // 1, 2, 4, or 8 (default 4)
```

---

## What it does

GPU LZSS (LZ77 + a literal/match flag bitmap) — a direct port of the
**GPULZ** reference kernels. Each fixed-size chunk (`chunk_size` bytes) is
compressed independently by one CUDA thread block:

- The chunk is loaded into shared memory as `word_size`-byte elements.
- Every element searches a 32-element sliding window behind it for the
  longest repeated run (a classic LZSS match search, done in parallel — one
  thread per element).
- A single thread then serially walks the per-element match lengths to build
  a literal/match decision for each "encode item" and a 1-bit-per-item flag
  bitmap (this walk is inherently sequential, matching the upstream
  algorithm).
- A block-wide prefix sum (Blelloch scan) turns per-item byte sizes into
  packing offsets, and the block writes literal bytes (`word_size` bytes) or
  match tokens (`[length, offset]`, 2 bytes) into a compact per-chunk buffer.

This is the first LZ77-family stage in FZGPUModules — pairing it with
`HuffmanStage` or `ANSStage` reproduces the LZ77 + entropy-coding structure of
DEFLATE/Zstandard, e.g. `GPULZ -> Huffman` or `GPULZ -> ANS`.

---

## Stage settings

```cpp
gpulz->setChunkSize(2048);  // bytes: 1024, 2048, or 4096 (default 2048)
gpulz->setWordSize(4);      // word granularity in bytes: 1, 2, 4, or 8 (default 4)
```

`chunk_size` is restricted to this set because the encode kernel keeps the
whole chunk (plus per-element length/offset/prefix-sum scratch) in **static**
`__shared__` memory, and the algorithm's Blelloch scan requires
`chunk_size / word_size` to be a power of two `>= 128` (the fixed
128-thread block size) — all three supported sizes satisfy this for every
supported `word_size`.

The 32-element sliding window and the 128-thread block size are fixed,
matching the upstream reference's defaults; they are not currently exposed as
stage settings.

---

## Alignment requirement

Requires input to be a multiple of `chunk_size` bytes
(`getRequiredInputAlignment()`); `Pipeline::finalize()` pads automatically.

---

## Raw-chunk fallback

If a chunk's LZSS-encoded form (flag bitmap + literal/match bytes) would not
be smaller than the chunk itself, the chunk is stored verbatim instead (a
high bit on its header entry marks this). This bounds worst-case output to
`original_size + header_bytes` regardless of input entropy.

---

## All-zero-chunk fast path

Before running the match search, the encode kernel does a warp-vote check
(`fz::backend::anySync32`) for whether the whole chunk is zero. If so, the
match search and flag/data encode are skipped entirely — the chunk
contributes 0 bytes to the compressed payload (a `(flag_size=0, data_size=0)`
sentinel, distinct from the raw-fallback sentinel above). On decode, the
corresponding output span is zero-filled directly rather than walking a flag
bitmap. This matters for sparse inputs (e.g. quantized neural-compressor
latents, which are often mostly zero) where whole 1-4 KB chunks are
frequently all-zero — those chunks cost nothing beyond the 8-byte header
entry, on both the encode and decode side.

---

## Graph capture

Forward (compress) is CUDA-graph capturable — the final output-size readback
is deferred to `postStreamSync()`. The inverse (decompress) path is not — it
reads the stream header (original size, per-chunk flag/data sizes) with
blocking device-to-host copies before it can compute per-chunk input offsets
and launch the decode kernel. This mirrors `RREStage`/`RZEStage`.

---

## Output stream format

```
[uint32_t: original byte count]
[uint32_t: num_chunks]
[ (uint32_t flag_size, uint32_t data_size) x num_chunks ]   // flag_size high bit -> chunk stored raw
[ per-chunk payload: flag bytes, then compressed-data bytes (or raw bytes if flagged) ... ]
```

---

## Acknowledgements

The GPU kernels in `GPULZStage` are a direct port of `compressKernelI` and
`decompressKernel` from **GPULZ**.

> Boyuan Zhang, Jiannan Tian, Sheng Di, Xiaodong Yu, Martin Swany, Dingwen Tao,
> and Franck Cappello.
> *GPULZ: Optimizing LZSS Lossless Compression for Multi-byte Data on Modern GPUs.*
> ICS '23. https://github.com/hpdps-group/ICS23-GPULZ

The all-zero-chunk fast path is adapted from the "sparse" GPULZ variant in
**AIZ_VLDB26** (Boyuan Zhang, `test/gpulz.cuh`'s `notEmptyFlagArr`):
https://github.com/boyuanzhang62/AIZ_VLDB26

See `THIRD_PARTY.md` — neither upstream repository declares an explicit
license.
