# TUPLStage {#stage_tupl}

**Header:** `modules/shufflers/tupl/tupl_stage.h`  
**Class:** `fz::TUPLStage` — no template parameters  
**Category:** Transform / shuffler (lossless)

**Common instantiation:**
```cpp
auto* tupl = p.addStage<fz::TUPLStage>();
```

---

## What it does

GPU tuple deinterleave (AoS → SoA) transpose over fixed-size blocks. Given a block
of `tuples` structs, each `dim` fields wide (fields of `word_size` bytes), the
forward pass regroups the data field-major: all field-0 words, then all field-1
words, and so on. This is LC's `TUPLk` component — separating interleaved fields
(e.g. real/imaginary pairs, RGB triples, or any struct-of-arrays candidate) lets
downstream byte-oriented coders (`RZEStage`, `RREStage`, `BitshuffleStage`, ...)
see each field's own value distribution instead of an interleaved mixture.

Output is the same byte size as input (pure permutation, no compression on its
own — it's a decorrelation step for whatever coder follows it).

---

## Stage settings

```cpp
tupl->setBlockSize(16384);   // block size in bytes (default 16384)
tupl->setWordSize(1);        // field width: 1, 2, 4, or 8 (default 1)
tupl->setDim(2);             // fields per tuple, i.e. LC's TUPLk (default 2)
```

**Constraints:**
- `word_size` must be 1, 2, 4, or 8.
- `dim` must be >= 2 (a tuple needs at least two fields).
- `block_size` must be a positive multiple of `word_size` (so per-block byte
  offsets stay word-aligned). Unlike LC — which shares one hardcoded 16 KB
  chunk across all `(dim, word_size)` combinations, so a chunk frequently
  doesn't divide evenly into whole tuples — `block_size` here is caller-chosen
  and independent of `dim`.

Any leftover bytes at the tail of a block that don't form a complete tuple
(`block_size` not evenly divisible by `dim * word_size`) are copied verbatim,
unchanged by either direction. This is the common case for `dim` in {3, 6, 12}
at the default 16 KB block size, not an edge case — e.g. `dim=3, word_size=8`
leaves 16 bytes of tail per 16 KB block.

---

## Alignment requirement

`TUPLStage` requires its input to be a multiple of `block_size` bytes for the
transpose to apply uniformly; a final partial block shorter than `block_size`
is passed through unchanged (a plain device-to-device copy), mirroring
`BitshuffleStage`'s tail handling.

---

## Typical pipeline

```cpp
auto* tupl = p.addStage<TUPLStage>();
auto* rze  = p.addStage<RZEStage>();

tupl->setWordSize(sizeof(uint16_t));   // match upstream element type
tupl->setDim(2);                       // e.g. deinterleave real/imag pairs

p.connect(tupl, upstream_stage);
p.connect(rze,  tupl);
p.finalize();
```

---

## Acknowledgements

`TUPLStage` is a faithful GPU port of `d_TUPL` / `d_iTUPL` from the
**LC framework** (Burtscher et al., Texas State University, BSD-3-Clause).
Upstream generates one fixed `(dim, word_size)` instantiation per component
(`TUPL2_1`, `TUPL6_8`, `TUPL12_1`, ...) over a hardcoded 16 KB chunk; here
`dim`, `word_size`, and `block_size` are all independent runtime parameters.

> Noushin Azami, Alex Fallin, Brandon Burtchell, Andrew Rodriguez,
> Benila Jerald, Yiqian Liu, Anju Mongandampulath Akathoott, and Martin Burtscher.
> *LC framework for synthesizing high-speed parallel lossless and
> error-bounded lossy data compression and decompression algorithms for CPUs and GPUs.*
> https://github.com/burtscher/LC-framework

See `THIRD_PARTY.md` for the full license text.
