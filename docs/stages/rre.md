# RREStage {#stage_rre}

**Header:** `modules/coders/rre/rre_stage.h`  
**Class:** `fz::RREStage` — no template parameters  
**Category:** Coder (lossless)

**Common instantiation:**
```cpp
auto* rre = p.addStage<fz::RREStage>();
rre->setWordSize(1);   // 1, 2, 4, or 8 (default 1)
```

---

## What it does

`RRE` is the upstream name of this lossless **LC framework** component; LC
describes its behavior without expanding the token. It is used by cuSZ-Hi's LC pipelines (the speed chain
`TCMS1 → BIT1 → RRE1` and the compression-ratio chain `RRE4 → TCMS8 → RZE1`).

Operates on a raw byte stream treated as `word_size`-byte words. Each chunk is
processed in shared memory by one CUDA block:

- **Level 1 (RE):** compact non-repeated words — a word is kept only if it
  differs from its predecessor (word 0 is compared against an implicit `0`);
  emit a 1-bit-per-word bitmap.
- **Bitmap recursion:** the level-1 bitmap is itself RE-compressed through the
  hierarchical 2048 / 256 / 32 / 4-byte levels (the same machinery as `RZEStage`'s
  levels 2–4), so long runs of a repeated value collapse to a few bytes.

RRE is the repetition-eliminating sibling of `RZEStage` (which eliminates
**zeros**). Use RRE where the stream contains long runs of an arbitrary repeated
value, not just zeros.

---

## Stage settings

```cpp
rre->setChunkSize(16384);   // bytes; 4096, 8192, or 16384 (default 16384)
rre->setWordSize(1);        // word granularity in bytes: 1, 2, 4, or 8 (default 1)
```

`chunk_size` is restricted to this small set because each CUDA block holds the
whole chunk (`in` + `out` + a fixed 4 KB scratch buffer) in **static**
`__shared__` memory — the three supported sizes (12 KB / 20 KB / 36 KB total)
all fit comfortably under the 48 KB static cap. Larger chunk sizes would need
the dynamic-shared-memory opt-in `GInterpStage` uses for its 3-D `double`
path; not implemented here.

`word_size` selects the LC `RRE_1` / `RRE_2` / `RRE_4` / `RRE_8` variant. The
cuSZ-Hi chains use `RRE1` (speed, on quant codes), `RRE2` (anchor/outlier), and
`RRE4` (compression-ratio).

---

## Alignment requirement

Requires input to be a multiple of `chunk_size` bytes. The pipeline pads
automatically when an upstream byte-oriented stage uses a matching block size.

---

## Graph capture

Forward (compress) is CUDA-graph capturable. The inverse (decompress) path is
not — it reads the stream header (original size, per-chunk sizes) with blocking
device-to-host copies before launching the decode kernel. This mirrors
`RZEStage`; decompress-only graph capture has no practical use case.

---

## Typical pipeline

```cpp
// LC speed chain on quantization codes: TCMS1 -> BIT1 -> RRE1
auto* tcms = p.addStage<ZigzagStage<int8_t>>();      // TCMS == zigzag
auto* bit  = p.addStage<BitshuffleStage>();          // BIT
bit->setElementWidth(1);
auto* rre  = p.addStage<RREStage>();                 // RRE
rre->setWordSize(1);

p.connect(bit, tcms);
p.connect(rre, bit);
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

A chunk is stored verbatim (high-bit flag) when RRE fails to shrink it. A
constant (all-repeating) chunk collapses to a 2-byte size tag.

---

## Acknowledgements

The GPU kernels in `RREStage` (`modules/coders/lc_common/lc_chunk_components.cuh`) are a faithful
port of `d_RRE.h`, `d_repetition_elimination.h`, and `prefix_sum.h` from the
**LC framework** (Burtscher et al., Texas State University, BSD-3-Clause).

> Noushin Azami, Alex Fallin, Brandon Burtchell, Andrew Rodriguez,
> Benila Jerald, Yiqian Liu, Anju Mongandampulath Akathoott, and Martin Burtscher.
> *LC framework for synthesizing high-speed parallel lossless and
> error-bounded lossy data compression and decompression algorithms for CPUs and GPUs.*
> https://github.com/burtscher/LC-framework

See `THIRD_PARTY.md` for the full license text.
