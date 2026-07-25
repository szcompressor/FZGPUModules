# HCLOGStage {#stage_hclog}

**Header:** `modules/coders/hclog/hclog_stage.h`  
**Class:** `fz::HCLOGStage` — no template parameters  
**Category:** Coder (lossless)

**Common instantiation:**
```cpp
auto* hclog = p.addStage<fz::HCLOGStage>();
hclog->setWordSize(1);   // 1, 2, 4, or 8 (default 1)
```

---

## What it does

Compressed-Logarithm coding with a per-subchunk TCMS fallback — the `HCLOG`
lossless component of the **LC framework**, the auto-selecting sibling of
`CLOGStage`. Uses the same fixed-32-subchunk, minimum-bit-width truncation
scheme as CLOG, but for **each subchunk** it also computes the minimum
bit-width needed after reinterpreting every value via TCMS (the same
two's-complement → sign-magnitude / zigzag transform `ZigzagStage` uses) and
picks whichever of the two is smaller — recording the choice as one flag bit
per subchunk (32 bits total, stored as a single `int` at the very front of
the chunk).

This does meaningfully better than plain CLOG on **bipolar-looking data** —
e.g. a stream where a bit pattern's raw unsigned magnitude is large (values
near the top of `T`'s range, as any negative two's-complement value looks
when read unsigned) but its *signed* magnitude is small. Raw bit-packing
alone would need close to the full word width for such data (no
compression); TCMS/zigzag maps small `|value|` regardless of sign to a small
code, often halving the packed width. Like CLOG, `T` must be **unsigned**
(`uint8/16/32/64` only) and there is no auxiliary bitmap or per-element
full/dropped decision — every element in a subchunk shares the same packed
width, after the shared TCMS-or-not choice for that subchunk.

---

## Stage settings

```cpp
hclog->setChunkSize(16384);   // bytes; 4096, 8192, or 16384 (default 16384)
hclog->setWordSize(1);        // word granularity in bytes: 1, 2, 4, or 8 (default 1)
```

Same chunk-size restriction and rationale as `RREStage` (static `__shared__`
budget under the 48 KB cap). `word_size` selects the LC `HCLOG_1` / `HCLOG_2`
/ `HCLOG_4` / `HCLOG_8` variant.

---

## Alignment requirement

Requires input to be a multiple of `chunk_size` bytes. The pipeline pads
automatically when an upstream byte-oriented stage uses a matching block size.

---

## Graph capture

Forward (compress) is CUDA-graph capturable. The inverse (decompress) path is
not — it reads the stream header (original size, per-chunk sizes) with
blocking device-to-host copies before launching the decode kernel. This
mirrors `RREStage`/`RZEStage`/`RAREStage`/`RAZEStage`/`CLOGStage`.

---

## Typical pipeline

```cpp
auto* hclog = p.addStage<HCLOGStage>();
hclog->setWordSize(1);
p.connect(hclog, upstream);
p.finalize();
```

---

## Stream layout (forward output)

Identical container to `CLOGStage`/`RREStage`/`RZEStage` — HCLOG's internal
per-chunk header (a 32-bit per-subchunk TCMS-flag word, a `short`
original-size tag, then the 32-entry bit-width table) lives **inside** each
chunk's own compressed bytes (accounted for by that chunk's stored size), not
in the container header:

```
[uint32_t: original byte count]
[uint32_t: num_chunks]
[uint32_t x num_chunks: per-chunk compressed sizes (high bit set -> chunk stored raw)]
[compressed chunk data ...]
```

A chunk is stored verbatim (high-bit flag) when HCLOG fails to shrink it.

---

## Acknowledgements

The GPU kernels in `HCLOGStage` (`modules/coders/lc_common/lc_clog_components.cuh`,
`d_CLOGencode`/`d_CLOGdecode<T, CLogMode::WITH_TCMS_FALLBACK>`) are a
faithful port of `d_HCLOG.h` from the **LC framework** (Burtscher et al.,
Texas State University, BSD-3-Clause), sharing the encode/decode device code
with `CLOGStage` via a single template parameterized on whether the TCMS
fallback is enabled.

> Noushin Azami, Alex Fallin, Brandon Burtchell, Andrew Rodriguez,
> Benila Jerald, Yiqian Liu, Anju Mongandampulath Akathoott, and Martin Burtscher.
> *LC framework for synthesizing high-speed parallel lossless and
> error-bounded lossy data compression and decompression algorithms for CPUs and GPUs.*
> https://github.com/burtscher/LC-framework

See `THIRD_PARTY.md` for the full license text.
