# RAREStage {#stage_rare}

**Header:** `modules/coders/rare/rare_stage.h`  
**Class:** `fz::RAREStage` — no template parameters  
**Category:** Coder (lossless)

**Common instantiation:**
```cpp
auto* rare = p.addStage<fz::RAREStage>();
rare->setWordSize(1);   // 1, 2, 4, or 8 (default 1)
```

---

## What it does

`RARE` is the upstream name of this lossless **LC framework** component; LC
describes its behavior without expanding the token. It generalizes `RREStage`:
rather than
RRE's binary "word repeats its predecessor in full, or is dropped entirely"
test, it histograms how many top bits of `word ^ predecessor` are zero across
the whole chunk, picks **one global cut keep** (`0 <= keep < word_size*8`)
that maximizes total bit savings, then:

- Words whose top `bits - keep` bits match their predecessor store only their
  bottom `keep` bits, **bit-packed** (not byte-aligned) across word
  boundaries.
- Words that don't match are stored in full, exactly as in RRE.
- The 1-bit-per-word bitmap distinguishing the two cases is itself
  RE-compressed through the same hierarchical 2048/256/32/4-byte recursion
  RRE uses.

Where RRE can only ever represent "identical to predecessor" or "unrelated,"
RARE also captures the common case of values that are *close* to their
predecessor in their high bits — e.g. a slowly-drifting counter or a signal
with small excursions around a locally-stable baseline — without needing an
upstream predictor stage to remove that structure first.

---

## Stage settings

```cpp
rare->setChunkSize(16384);   // bytes; 4096, 8192, or 16384 (default 16384)
rare->setWordSize(1);        // word granularity in bytes: 1, 2, 4, or 8 (default 1)
```

Same chunk-size restriction and rationale as `RREStage` (static `__shared__`
budget under the 48 KB cap). `word_size` selects the LC `RARE_1` / `RARE_2` /
`RARE_4` / `RARE_8` variant.

---

## Alignment requirement

Requires input to be a multiple of `chunk_size` bytes. The pipeline pads
automatically when an upstream byte-oriented stage uses a matching block size.

---

## Graph capture

Forward (compress) is CUDA-graph capturable. The inverse (decompress) path is
not — it reads the stream header (original size, per-chunk sizes) with
blocking device-to-host copies before launching the decode kernel. This
mirrors `RREStage`/`RZEStage`.

---

## Typical pipeline

```cpp
auto* rare = p.addStage<RAREStage>();
rare->setWordSize(1);
p.connect(rare, upstream);
p.finalize();
```

---

## Stream layout (forward output)

Identical container to `RREStage` — the per-chunk `keep` value RARE adds
lives **inside** each chunk's own compressed bytes (accounted for by that
chunk's stored size), not in the container header, so the two stages share
byte-for-byte the same host-side framing:

```
[uint32_t: original byte count]
[uint32_t: num_chunks]
[uint32_t x num_chunks: per-chunk compressed sizes (high bit set -> chunk stored raw)]
[compressed chunk data ...]
```

A chunk is stored verbatim (high-bit flag) when RARE fails to shrink it. A
chunk where every word matches its predecessor collapses to a 3-byte tag
(RRE's equivalent all-repeat fast path uses 2 bytes — RARE's tag carries one
extra byte because the general path's trailer is `[keep][size_lo][size_hi]`
vs. RRE's `[size_lo][size_hi]`).

---

## Acknowledgements

The GPU kernels in `RAREStage` (`modules/coders/lc_common/lc_chunk_components.cuh`,
`d_PRencode`/`d_PRdecode<T, PartialReduceMode::REPEAT>`) are a faithful port of
`d_RARE.h` from the **LC framework** (Burtscher et al., Texas State
University, BSD-3-Clause), sharing the histogram/keep-selection/bit-pack
device code with `RAZEStage` via a single template parameterized on the match
predicate.

> Noushin Azami, Alex Fallin, Brandon Burtchell, Andrew Rodriguez,
> Benila Jerald, Yiqian Liu, Anju Mongandampulath Akathoott, and Martin Burtscher.
> *LC framework for synthesizing high-speed parallel lossless and
> error-bounded lossy data compression and decompression algorithms for CPUs and GPUs.*
> https://github.com/burtscher/LC-framework

See `THIRD_PARTY.md` for the full license text.
