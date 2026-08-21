# RAZEStage {#stage_raze}

**Header:** `modules/coders/raze/raze_stage.h`  
**Class:** `fz::RAZEStage` — no template parameters  
**Category:** Coder (lossless)

**Common instantiation:**
```cpp
auto* raze = p.addStage<fz::RAZEStage>();
raze->setWordSize(1);   // 1, 2, 4, or 8 (default 1)
```

---

## What it does

`RAZE` is the upstream name of this lossless **LC framework** component; LC
describes its behavior without expanding the token. It generalizes `RZEStage`:
rather than
RZE's binary "word is exactly zero, or it's kept in full" test, it histograms
how many top bits of each word (on its own — not compared against a
predecessor) are zero across the whole chunk, picks **one global cut `keep`**
(`0 <= keep < word_size*8`) that maximizes total bit savings, then:

- Words whose top `bits - keep` bits are all zero store only their bottom
  `keep` bits, **bit-packed** (not byte-aligned) across word boundaries.
- Words with a nonzero top-bit region are stored in full, exactly as in RZE.
- The 1-bit-per-word bitmap distinguishing the two cases is itself
  RE-compressed through the same hierarchical 2048/256/32/4-byte recursion
  RZE uses.

Where RZE can only exploit exact zeros, RAZE also captures small-magnitude
nonzero values — e.g. a mostly-small-integer stream with a long tail of large
outliers — without needing those small values to be literally zero to benefit.

---

## Stage settings

```cpp
raze->setChunkSize(16384);   // bytes; 4096, 8192, or 16384 (default 16384)
raze->setWordSize(1);        // word granularity in bytes: 1, 2, 4, or 8 (default 1)
```

Same chunk-size restriction and rationale as `RZEStage` (static `__shared__`
budget under the 48 KB cap). `word_size` selects the LC `RAZE_1` / `RAZE_2` /
`RAZE_4` / `RAZE_8` variant.

---

## Alignment requirement

Requires input to be a multiple of `chunk_size` bytes. The pipeline pads
automatically when an upstream byte-oriented stage uses a matching block size.

---

## Graph capture

Forward (compress) is CUDA-graph capturable. The inverse (decompress) path is
not — it reads the stream header (original size, per-chunk sizes) with
blocking device-to-host copies before launching the decode kernel. This
mirrors `RREStage`/`RZEStage`/`RAREStage`.

---

## Typical pipeline

```cpp
auto* raze = p.addStage<RAZEStage>();
raze->setWordSize(1);
p.connect(raze, upstream);
p.finalize();
```

---

## Stream layout (forward output)

Identical container to `RZEStage` — the per-chunk `keep` value RAZE adds
lives **inside** each chunk's own compressed bytes (accounted for by that
chunk's stored size), not in the container header, so the two stages share
byte-for-byte the same host-side framing:

```
[uint32_t: original byte count]
[uint32_t: num_chunks]
[uint32_t x num_chunks: per-chunk compressed sizes (high bit set -> chunk stored raw)]
[compressed chunk data ...]
```

A chunk is stored verbatim (high-bit flag) when RAZE fails to shrink it. An
all-zero chunk collapses to a 3-byte tag (RZE's equivalent fast path uses 2
bytes — RAZE's tag carries one extra byte because the general path's trailer
is `[keep][size_lo][size_hi]` vs. RZE's `[size_lo][size_hi]`).

---

## Acknowledgements

The GPU kernels in `RAZEStage` (`modules/coders/lc_common/lc_chunk_components.cuh`,
`d_PRencode`/`d_PRdecode<T, PartialReduceMode::ZERO>`) are a faithful port of
`d_RAZE.h` from the **LC framework** (Burtscher et al., Texas State
University, BSD-3-Clause), sharing the histogram/keep-selection/bit-pack
device code with `RAREStage` via a single template parameterized on the match
predicate.

> Noushin Azami, Alex Fallin, Brandon Burtchell, Andrew Rodriguez,
> Benila Jerald, Yiqian Liu, Anju Mongandampulath Akathoott, and Martin Burtscher.
> *LC framework for synthesizing high-speed parallel lossless and
> error-bounded lossy data compression and decompression algorithms for CPUs and GPUs.*
> https://github.com/burtscher/LC-framework

See `THIRD_PARTY.md` for the full license text.
