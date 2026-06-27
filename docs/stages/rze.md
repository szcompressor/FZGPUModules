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

Zero-Elimination Encoding — the `RZE` lossless component of the **LC framework**
(`RZE_1/2/4/8`, used by cuSZ-Hi's LC pipelines). Operates on a raw byte stream
treated as `word_size`-byte words. Each chunk is processed in shared memory:

- **Level 1 (ZE):** compact non-zero words; emit a 1-bit-per-word zero bitmap.
- **Bitmap recursion:** the level-1 bitmap is itself RE-compressed through the
  hierarchical 2048 / 256 / 32 / 4-byte levels.

Because bit-shuffled scientific data can have many zero byte-planes, RZE can compress
those planes very aggressively. RZE is the zero-eliminating sibling of
\ref stage_rre "RREStage" (which eliminates repeated values); the two share the
vendored LC chunk kernels (`modules/coders/lc_common/lc_chunk_components.cuh`).

---

## Stage settings

```cpp
rze->setChunkSize(16384);   // bytes; only 16384 is currently supported (default)
rze->setWordSize(1);        // word granularity: 1, 2, 4, or 8 (default 1 = LC RZE_1)
```

`word_size` selects the LC `RZE_1` / `RZE_2` / `RZE_4` / `RZE_8` variant — the
`_N` suffix is the **word size** (not a recursion-level count). The cuSZ-Hi
chains use `RZE_1`.

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
rze->setWordSize(1);

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

---

## Acknowledgements

The GPU kernels in `RZEStage` are a faithful port of `d_RZE.h`,
`d_zero_elimination.h`, and `d_repetition_elimination.h` from the **LC framework**
(Burtscher et al., Texas State University, BSD-3-Clause), shared with `RREStage`
via `modules/coders/lc_common/lc_chunk_components.cuh`.

> Noushin Azami, Alex Fallin, Brandon Burtchell, Andrew Rodriguez,
> Benila Jerald, Yiqian Liu, Anju Mongandampulath Akathoott, and Martin Burtscher.
> *LC framework for synthesizing high-speed parallel lossless and
> error-bounded lossy data compression and decompression algorithms for CPUs and GPUs.*
> https://github.com/burtscher/LC-framework

See `THIRD_PARTY.md` for the full license text.
