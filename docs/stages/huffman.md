# HuffmanStage {#stage_huffman}

**Header:** `modules/coders/huffman/huffman_stage.h`  
**Class:** `fz::HuffmanStage<T>`  
**Category:** Coder (lossless)

**Common instantiation:**
```cpp
auto* huf = p.addStage<fz::HuffmanStage<uint16_t>>();
huf->setBklen(1024);
```

---

## What it does

Entropy-encodes a flat symbol stream using GPU-accelerated Huffman coding (PHF
coarse-grained encoding).  Forward pass produces a variable-length bitstream with
an embedded self-describing header; inverse pass reconstructs the original symbol
array exactly.

- **Forward:** `T[] → uint8_t[]`  PHF bitstream with embedded `phf_header`
- **Inverse:** `uint8_t[] → T[]`  Exact symbol reconstruction

---

## Template parameter

| Parameter | Constraint |
|---|---|
| `T` | Symbol type: `uint8_t`, `uint16_t`, or `uint32_t` |

## Available instantiations

Only these types are compiled and linked:
- `HuffmanStage<uint8_t>`
- `HuffmanStage<uint16_t>` — most common (quantization codes)
- `HuffmanStage<uint32_t>`

---

## Stage settings

| Setting | Type | Default | Purpose |
|---|---|---|---|
| `setBklen(n)` | `uint32_t` | 256 (U8), 1024 (U16/U32) | Codebook length — number of distinct symbols |
| `setEncodeMode(m)` | `HuffmanEncodeMode` | `Coarse` | `Fine` removes the mid-encode CPU sync |
| `setBookSource(s)` | `HuffmanBookSource` | `PerBlock` | `Adaptive`/`Fixed` reuse one codebook — see [Pre-built codebooks](#huffman-prebuilt-book) |
| `setAdaptiveFloorShift(k)` | `uint8_t` | 24 | `Adaptive` frequency floor, as `max_freq >> k` |
| `setRefitThreshold(r)` | `float` | 1.2 | `Adaptive` refits when the bit rate degrades past `r`x the fitted rate (0 = never) |
| `setRefitInterval(n)` | `uint32_t` | 0 | `Adaptive` refits every `n` calls (0 = off) |
| `setFixedBookFromModel(spec)` | `HuffmanBookSpec` | — | Build a `Fixed` book from an analytic distribution |
| `setFixedBookFromFreq(f, n)` | `const uint32_t*`, `uint32_t` | — | Build a `Fixed` book from a frequency table |

### Setting `bklen`

`bklen` is the size of the Huffman codebook and must cover the full range of
symbols that will appear in the input.  **All input symbols must be in `[0, bklen)`.**
Symbols outside this range are detected after the histogram D2H and throw a
`std::runtime_error` naming the count of out-of-range symbols — but only on calls that
actually histogram, so the check does not run once a codebook has been pinned by
`Adaptive` or `Fixed` (see [Pre-built codebooks](#huffman-prebuilt-book)).

```cpp
huf->setBklen(1024);
```

Typical values:

| Upstream stage | Input type | Recommended `bklen` |
|---|---|---|
| `LorenzoQuantStage` with `zigzag_codes=true`, `quant_radius=r` | `uint16_t` | `2 * r` |
| `LorenzoQuantStage` with `zigzag_codes=false`, `quant_radius=r` | `uint16_t` | `65536` |
| `QuantizerStage` codes | `uint16_t` | `2 * radius` |
| Generic byte data | `uint8_t` | `256` (default) |

Set `bklen` before the first `compress()` call.  Changing `bklen` after the first
`execute()` forces a full reallocation of all PHF internal buffers on the next call.

---

## Typical pipeline

### Standalone (symbol array input)

```cpp
Pipeline p(in_bytes, MemoryStrategy::PREALLOCATE);
auto* huf = p.addStage<HuffmanStage<uint16_t>>();
huf->setBklen(1024);
p.finalize();

p.compress(d_in, in_bytes, stream);
```

### cuSZ-style Lorenzo + Huffman

```cpp
Pipeline p(in_bytes, MemoryStrategy::MINIMAL);

auto* lrz = p.addStage<LorenzoQuantStage<float, uint16_t>>();
lrz->setErrorBound(1e-3f);
lrz->setQuantRadius(512);
lrz->setZigzagCodes(true);   // required: keeps codes in [0, 2*radius-2]

auto* huf = p.addStage<HuffmanStage<uint16_t>>();
huf->setBklen(1024);          // must equal 2 * quant_radius
p.connect(huf, lrz, "codes");
p.finalize();
```

**Why `zigzag_codes=true` is required here:**  With `zigzag_codes=false` (raw delta),
positive deltas map to `[0, radius-1]` and negative deltas wrap to
`[65537-radius, 65535]` in uint16.  These two lobes require `bklen=65536` for
the PHF codebook to cover the full symbol range.  With zigzag, all codes land in
the contiguous range `[0, 2*radius-2]`, so `bklen=2*radius` is sufficient.

See `examples/presets/cusz.toml` for the corresponding TOML configuration.

---

## TOML configuration

```toml
[[stage]]
name        = "huf"
type        = "Huffman"
input_type  = "uint16"   # "uint8", "uint16", or "uint32"
bklen       = 1024       # optional; defaults to 256 (uint8) or 1024 (uint16/uint32)
encode_mode = "Coarse"   # optional; "Coarse" (default) or "Fine"
book_source = "PerBlock" # optional; "PerBlock" (default), "Adaptive", or "Fixed"
book_floor_shift = 24    # Adaptive only; frequency floor as max_freq >> shift
book_refit_threshold = 1.2  # Adaptive only; refit when bit rate degrades past this (0 = never)
book_refit_interval  = 0    # Adaptive only; refit every N calls (0 = off)
book_model  = "Gaussian" # Fixed only; "Gaussian", "Laplace", "GeneralizedNormal", "Uniform"
book_center = -1.0       # Fixed only; negative means bklen/2
book_scale  = 32.0       # Fixed only; distribution width in symbols
book_shape  = 2.0        # Fixed only; GeneralizedNormal exponent
inputs      = [{from = "lrz", port = "codes"}]
```

---

## Pre-built codebooks {#huffman-prebuilt-book}

By default the stage histograms its input and builds a fresh Huffman tree on every
forward call.  That costs a full extra pass over the input, a device-to-host copy, a
host stream synchronization, and a serial tree build — **every time**.  Two modes
build **one** codebook and reuse it instead.

The idea comes from CEAZ (Xiong et al., ICS'22), which generates canonical codewords
offline from representative scientific data, and from Shah et al., *Lightweight
Huffman Coding for Efficient GPU Compression* (ICS'23), which fits distributions to
cuSZ's quantization-code stream and selects a precomputed codebook at runtime.

### `Adaptive` — sample once, reuse (recommended)

Histograms the **first** block only, then builds a codebook from that histogram and
pins it.  One histogram for the lifetime of the stage instead of one per call:

```cpp
auto* huf = p.addStage<HuffmanStage<uint16_t>>();
huf->setBklen(1024);
huf->setBookSource(HuffmanBookSource::Adaptive);
```

Frequencies are floored at `max_freq >> book_floor_shift` before the tree is built.
The floor is what makes the book safe to reuse: every symbol in `[0, bklen)` gets a
code, including symbols the sampled block never contained, so a later block
containing them still encodes correctly.  It also bounds Huffman depth — see the
27-bit limit below.  If the book still does not fit, the shift is halved and the
build retried; shift 0 is a uniform book, which always fits.

Compression ratio is essentially unchanged, because the book is fitted to the data
rather than guessed.  Geometric mean ratio against `PerBlock` over 26 (preset, field)
combinations — `cusz.toml`, `cusz_hi_cr.toml` and `gpu_zstd.toml` across 5 CESM-ATM
fields, 3 Hurricane-ISABEL fields and EXAALT:

| preset | upstream of Huffman | `Adaptive` ratio vs `PerBlock` |
|---|---|---|
| `cusz.toml`       | `LorenzoQuant` codes, bklen 1024 | **-0.0%** |
| `cusz_hi_cr.toml` | `GInterp` codes, bklen 4096      | **-0.5%** |
| `gpu_zstd.toml`   | `GPULZ` literals, bklen 4096     | **-0.0%** |

Worst single field: -1.3%.  Sweeping the error bound over `1e-2 … 1e-5` on three
fields moves it by less than 0.01%.  Reconstruction PSNR is identical to `PerBlock`
in every cell, as it must be — the codebook affects only how the symbols are spelled.

Throughput gains are real but not yet well characterized: roughly 1.1x–1.4x geometric
mean per preset on a development box (WSL2, RTX 3080 Ti, single run per cell), and
1.5x–2.2x on repeated small-to-medium compresses of one field.  Run-to-run spread on
that machine is about ±20%, so treat all of these as provisional.

### `Fixed` — a book chosen up front

Skips the histogram entirely, using a codebook derived from an analytic distribution
or supplied directly:

```cpp
// Quantization codes cluster around the zero-error code; a Laplace body fits them
// well.  center < 0 means bklen/2, which is where an unsigned quantizer with
// radius = bklen/2 puts zero error.  Use center = 0 for zigzagged codes.
huf->setFixedBookFromModel({HuffmanBookModel::Laplace, /*center=*/-1.0,
                            /*scale=*/32.0, /*shape=*/2.0});

huf->setFixedBookFromFreq(freq.data(), 1024);   // or a table; every entry non-zero
```

`Fixed` is the right choice only when you must know the codebook before seeing any
data — offline-generated books in the CEAZ sense, or a workflow where the same book
has to be shared across independently compressed chunks.  Otherwise prefer
`Adaptive`: it reaches the same steady-state throughput without the ratio risk.

**That ratio risk is large.**  Over the same 26 combinations, a reasonably configured
`Fixed` model book (Laplace, centered per preset, `scale = bklen/64`) lost a geometric
mean of **-31%** on `cusz.toml`, **-83%** on `cusz_hi_cr.toml` and **-11%** on
`gpu_zstd.toml`, with a worst case of -93%.  On `cusz_hi_cr.toml` it pinned the ratio
near 4.5 on every field, from ones `PerBlock` compressed 12x to ones it compressed
61x — at that point the model's own entropy, not the data, sets the output size.

The knobs are sharp in both directions.  On CESM `CLDHGH` at `1e-4` the same book
costs only 3%, which is what makes `Fixed` easy to over-trust from a single
measurement; at `1e-2` on the same field it costs 68%.  If you use `Fixed`, measure it
on your own data, upstream stage, *and* error bound.

### Things to know about both

- **`Adaptive` does not eliminate the ratio/tuning question — it answers it.** A model
  book cannot match a bimodal or otherwise irregular code distribution; a sampled one
  tracks whatever the data actually does.
- **`book_floor_shift` matters much more at large `bklen`.**  Dropping it from 24 to 12
  costs about 1–3% ratio at `bklen = 1024` but **23%** at `bklen = 4096`, because the
  floor swallows a larger share of a wider distribution.  Leave it at the default
  unless a book fails to build.
- **The book is fitted to the first block only, so ratio drifts across a long run.**
  Measured with `fzgmod-profile-huffman-drift` (one pipeline, many successive
  different steps): **4.7% mean / 9.6% worst** across 23 CESM `CLOUD` levels, **8.3%
  / 27.6%** across 5 different CESM-ATM fields, **9.2% / 22.6%** across 20 Hurricane
  slabs.  Real but bounded, and still well under what a fixed model book gives up.
  Nothing detects drift automatically; `PerBlock` is the refit-every-call bound.
- **Refitting.**  `setRefitThreshold(r)` (default 1.2) rebuilds the book once the
  encoded bit rate degrades past `r` times the rate it achieved when fitted — free,
  because `encode()` already reports `total_nbit`.  The refit lands on the *next*
  call; the current one is already encoded.  This handles a distribution drifting
  toward less compressible: over 26 CESM `CLOUD` levels it cuts the loss from 42% to
  0.9%.  It cannot see a block that is *more* compressible than the fitted one while
  still being poorly served by its book — `LHFLX` lost 27.6% ratio to a `CLDHGH`
  codebook while its bit rate fell.  `setRefitInterval(n)` refits unconditionally
  every `n` calls and covers that case at the cost of one histogram per `n`.
  **When every call carries a genuinely different variable, prefer `PerBlock`**: a
  refit always lands one call late, so it cannot match a book fitted to the block
  being encoded.
- **A constant first block no longer poisons the book.**  A step whose values
  are all identical produces a single-symbol sample; the frequency floor turns that
  into a book where the dominant symbol costs a full bit, and it is then pinned
  forever.  The same CESM field measures **42% mean loss** when the fit lands on a
  constant level against **4.7%** when it does not.  This is not exotic — CESM
  `CLOUD` levels 0–2 and Hurricane `CLOUDf48` slabs 18–19 are all exactly constant.
  A degeneracy guard now refuses to pin a book fitted to a block with fewer than two
  distinct symbols, or with over 99.9% of its mass on one; the block is still encoded
  with that book, but the next call re-histograms.  This removed the failure mode —
  the same CESM run went from 42% mean loss to 0.86%.

- **The bitstream is unchanged.**  The reverse codebook still travels in every
  encoded stream, so output from either mode decodes through the stock path with no
  extra configuration and no file-format change.
- **The out-of-range symbol check goes away with the histogram.**  In `PerBlock` mode
  a symbol `>= bklen` is caught by the `sum(h_freq) == inlen` test.  Once a book is
  pinned nothing histograms the input, so the encode kernel simply indexes past the
  codebook.  The caller owns that invariant — `Adaptive` still checks on its one
  sampling call, `Fixed` never checks at all.
- **Codes cannot exceed 27 bits.**  `HuffmanWord<4>` holds the code in a 27-bit
  field.  A frequency distribution skewed enough to need a wider code is rejected with
  an exception in `PerBlock` and `Fixed` mode (the reference builder clamped it to an
  unusable `prefix_code = 0` and printed to stdout, which produced an undecodable
  stream).  `Adaptive` instead flattens its frequency floor until the book fits.
- **Ratio is data-dependent.**  Measure against `PerBlock` for your upstream stage
  before adopting either mode — the code distributions produced by `GInterp`,
  `AdaptiveLorenzo` and the LC stages differ from cuSZ-style dual quantization.

`setFixedBookFromFreq()` rejects a zero-frequency bin, and both setters reject a
book whose codes will not fit `HuffmanWord<4>`'s 27-bit code field (widen the model
scale or reduce the frequency dynamic range).  Model-derived books give every symbol
a strictly positive frequency, so they can always encode any in-range symbol; a book
trained on a sample of real data cannot make that promise.

Only the model form round-trips through TOML — a raw frequency table exceeds the
128-byte per-stage config slot and has to be re-supplied through the C++ API.

Fixed books do **not** make the stage graph-capturable on their own; see below.

---

## Execution flow (CPU–GPU movement pattern) {#huffman-execution}

HuffmanStage is unusual among FZGPUModules stages in that its default configuration
requires two host-synchronous operations inside each forward execute call — one to
transfer the histogram to the CPU for codebook construction, and one to synchronize
partition metadata for prefix-sum computation.

Each has its own opt-out. `setBookSource()` with `Adaptive` or `Fixed` removes steps
1–4 below (barrier 1) from every call after the first;
`setEncodeMode(HuffmanEncodeMode::Fine)` removes barrier 2 inside step 6. The flow
below is the default `PerBlock` + `Coarse` path.

### Forward pass

```
GPU  ←input T[]                                       output uint8_t[]→
  1. GPU histogram (p2013 shared-mem atomics)         d_freq[bklen]
     └─ cudaStreamSynchronize() ◄── HOST BARRIER 1
  2. cudaMemcpy D2H (blocking)                        h_freq[bklen]
     └─ CPU: sum(h_freq) == inlen check — throws if any symbol ≥ bklen
  3. CPU: canonical Huffman tree build                h_bk4[], h_revbk4[]
  4. cudaMemcpy H2D                                   d_bk4[], d_revbk4[]
  5. GPU encode phase 1 — fill from codebook          per-thread bitwords
  6. GPU encode phase 2 — deflate into partitions     d_par_nbit[], d_par_ncell[]
     └─ [inside encode()] HOST BARRIER 2:
        cudaMemcpy D2H(h_par_nbit, h_par_ncell)
        CPU: prefix-sum over pardeg partitions
        cudaMemcpy H2D(h_par_entry)
  7. GPU encode phase 4 — concatenate partitions      d_bitstream
  8. GPU memcpy_merge — assemble full PHF blob        d_encoded
  9. cudaMemcpyAsync D2D → pipeline output buffer
```

**Consequence:** the CPU-visible barriers per compress call make this stage
latency-bound. `Adaptive`/`Fixed` + `Fine` removes both of the barriers above, but
`isGraphCompatible()` still returns `false`: `encode()` returns `total_nbit` /
`total_ncell` to the host to assemble `phf_header` before the H2D merge, in both
encode modes. Making the stage capturable additionally requires assembling that
header and running the merge on the device.

### Inverse pass

```
GPU  ←input uint8_t[]                                 output T[]→
  1. cudaMemcpy D2H (blocking) — read phf_header      128-byte header
  2. GPU decode — revbk lookup → symbol reconstruction
```

---

## Internal buffer layout

`HuffmanStage<T>` holds a `phf::Buf<T>` object (lazily allocated on first execute,
reused as long as input length stays within the allocated capacity).  This object manages all PHF
internal device and host allocations directly via `cudaMalloc`/`cudaMallocHost`
**outside** the pipeline memory pool.  The pool is not used.

Approximate device footprint for a stream of `N` elements with codebook of length `B`:

| Buffer | Size |
|---|---|
| Histogram `d_freq` | `B × 4` bytes |
| Codebook `d_bk4` | `B × 4` bytes |
| Reverse codebook `d_revbk4` | `~4 × B × sizeof(T)` bytes |
| Partition metadata (3 arrays) | `pardeg × 4 × 3` bytes |
| Bitstream scratch | `N × 4` bytes (worst case) |
| Output `d_encoded` (alias of scratch) | same |

The stage output buffer (pipeline-managed) receives a D2D copy of `d_encoded`; the
pipeline pool provides that buffer.

---

## Serialized header

The FZM stage header is 11 bytes and stores only the configuration needed to
reconstruct the stage for decompression:

```
[0]      DataType of T   (1 byte)
[1..2]   bklen_          (uint16_t, little-endian)
[3..10]  original_len_   (uint64_t, little-endian; element count)
```

The PHF bitstream is self-describing: the 128-byte `phf_header` is embedded at
offset 0 of the encoded output and contains the codebook and partition layout.

---

## Limitations {#huffman-limitations}

**Symbol range is validated, but the check occurs after the GPU histogram D2H.**
All input symbols must be in `[0, bklen)`.  The histogram kernel skips out-of-range
symbols — they are not counted in `d_freq`.  `HuffmanStage` detects this by comparing
`sum(h_freq)` against `inlen` after the D2H copy and throws `std::runtime_error`
naming the out-of-range count.  The check adds negligible CPU overhead (one
O(bklen) accumulation) but cannot fire before the first host barrier.

Consequence: **when pairing with `LorenzoQuantStage`, `zigzag_codes=true` is
required** unless you set `bklen=65536`.  Raw signed-delta codes are not contiguous
in `[0, bklen)` for any `bklen < 65536`.

**Not CUDA Graph compatible in any configuration.**  In the default path two
device-to-host synchronization points exist in every forward call (histogram D2H for
codebook construction; partition metadata D2H for prefix-sum computation);
`HuffmanBookSource::Adaptive` / `::Fixed` and `HuffmanEncodeMode::Fine` remove them
respectively,
but the encoded-size round trip in `encode()` remains.  The stage cannot be included
in a graph-captured pipeline.

**Latency-bound, not throughput-bound.**  The CPU codebook build and the D2H syncs
are serial barriers.  Kernel execution time is small relative to round-trip PCIe
latency.  HuffmanStage performs poorly on very small inputs (< ~100 KB), which is
where a reused codebook helps most in relative terms: the cost it removes is per-call
and does not shrink with the payload.

**PHF scratch is pool-managed, not stream-ordered.**  `phf::Buf<T>` allocates all
PHF internal scratch via `MemoryPool::allocatePersistentDevice` /
`allocatePersistentPinned` (backed by `cudaMalloc` / `cudaMallocHost`).  These
allocations are persistent — they survive for the lifetime of the stage and are
returned to the pool when `phf::Buf<T>` is destroyed.  They are reported in
`pool->getPersistentDeviceBytes()` / `getPersistentPinnedBytes()` for total
footprint accounting.  They are not stream-ordered and do not participate in buffer
coloring.  Pool sizing (`MemoryPoolConfig::multiplier`) controls the stream-ordered
I/O buffer pool only; persistent PHF scratch is additional.

**Reallocation on capacity growth.**  `phf::Buf<T>` is reallocated only when the
input element count grows past the previously allocated capacity (`cap_inlen_`), or
when `bklen` changes.  Calls with smaller input reuse the existing buffer without
reallocating.  The `phf_header` embedded in the output always records the actual
element count (not the allocation capacity), so encode and decode are always
consistent.  Initial allocation and capacity-growth events incur full GPU allocator
overhead; steady-state or shrinking workloads do not.

---

## Acknowledgements

`HuffmanStage` incorporates PHF source files (`hf.h`, `hf_bk*.cc`, `hf_buf.cc`,
`hf_canon.cc`, `hf_hl.cc`, `hf_kernels.cu`, `hf_impl.hh`) vendored and adapted
from the **cuSZ** project PHF codec (`origin/v1.1.0_dev`), by the cuSZ team
(BSD-3-Clause). Changes are documented at the top of each adapted file.

> cuSZ team (UChicago Argonne National Laboratory, Indiana University, and others).
> *pSZ/cuSZ: A GPU-Based Error-Bounded Lossy Compressor for Scientific Data.*
> https://github.com/szcompressor/cuSZ

See `THIRD_PARTY.md` for the full license text.
