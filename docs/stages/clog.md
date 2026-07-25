# CLOGStage {#stage_clog}

**Header:** `modules/coders/clog/clog_stage.h`  
**Class:** `fz::CLOGStage` — no template parameters  
**Category:** Coder (lossless)

**Common instantiation:**
```cpp
auto* clog = p.addStage<fz::CLOGStage>();
clog->setWordSize(1);   // 1, 2, 4, or 8 (default 1)
```

---

## What it does

Compressed-Logarithm coding — the `CLOG` lossless component of the **LC
framework**. Splits each chunk into a **fixed 32 subchunks** (not
configurable — tied to the 32-lane warp used to parallelize the per-subchunk
reduction). Each subchunk independently finds its own max value and computes
the minimum bit-width needed to represent it, then every element in that
subchunk is truncated to exactly that width. This is lossless: no element in
a subchunk exceeds its own max, so none needs more bits than the width
chosen for it.

Unlike `RREStage`/`RAREStage`/`RZEStage`/`RAZEStage`, CLOG has **no auxiliary
bitmap and no per-element full/dropped decision** — every element in a
subchunk shares the same packed width, decided purely from that subchunk's
own data. `T` must be **unsigned** (`uint8/16/32/64` only — CLOG has no
signed word-size variants).

`HCLOGStage` is the auto-selecting sibling: it additionally tries a
TCMS(zigzag) reinterpretation per subchunk and picks whichever needs fewer
bits, at the cost of one flag bit per subchunk.

---

## Stage settings

```cpp
clog->setChunkSize(16384);   // bytes; 4096, 8192, or 16384 (default 16384)
clog->setWordSize(1);        // word granularity in bytes: 1, 2, 4, or 8 (default 1)
```

Same chunk-size restriction and rationale as `RREStage` (static `__shared__`
budget under the 48 KB cap). `word_size` selects the LC `CLOG_1` / `CLOG_2` /
`CLOG_4` / `CLOG_8` variant.

---

## Alignment requirement

Requires input to be a multiple of `chunk_size` bytes. The pipeline pads
automatically when an upstream byte-oriented stage uses a matching block size.

---

## Graph capture

Forward (compress) is CUDA-graph capturable. The inverse (decompress) path is
not — it reads the stream header (original size, per-chunk sizes) with
blocking device-to-host copies before launching the decode kernel. This
mirrors `RREStage`/`RZEStage`/`RAREStage`/`RAZEStage`.

---

## Typical pipeline

```cpp
auto* clog = p.addStage<CLOGStage>();
clog->setWordSize(1);
p.connect(clog, upstream);
p.finalize();
```

---

## Stream layout (forward output)

Identical container to `RREStage`/`RZEStage` — CLOG's internal per-chunk
header (a `short` original-size tag followed by the 32-entry bit-width table)
lives **inside** each chunk's own compressed bytes (accounted for by that
chunk's stored size), not in the container header:

```
[uint32_t: original byte count]
[uint32_t: num_chunks]
[uint32_t x num_chunks: per-chunk compressed sizes (high bit set -> chunk stored raw)]
[compressed chunk data ...]
```

A chunk is stored verbatim (high-bit flag) when CLOG fails to shrink it —
notably including any chunk where the per-subchunk bit-width table overhead
outweighs the savings, which is common for high-entropy data spanning the
full width of `T` (see `RandomBytesRoundTrip` in the test suite).

---

## Acknowledgements

The GPU kernels in `CLOGStage` (`modules/coders/lc_common/lc_clog_components.cuh`,
`d_CLOGencode`/`d_CLOGdecode<T, CLogMode::PLAIN>`) are a faithful port of
`d_CLOG.h` from the **LC framework** (Burtscher et al., Texas State
University, BSD-3-Clause), sharing the encode/decode device code with
`HCLOGStage` via a single template parameterized on whether the TCMS
fallback is enabled.

> Noushin Azami, Alex Fallin, Brandon Burtchell, Andrew Rodriguez,
> Benila Jerald, Yiqian Liu, Anju Mongandampulath Akathoott, and Martin Burtscher.
> *LC framework for synthesizing high-speed parallel lossless and
> error-bounded lossy data compression and decompression algorithms for CPUs and GPUs.*
> https://github.com/burtscher/LC-framework

See `THIRD_PARTY.md` for the full license text.
