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

Entropy-encodes a flat symbol stream using cuSZ's GPU-accelerated Huffman coder.
The cuSZ sources use `phf` as an internal namespace and type prefix; it is not a
separate library or algorithm. The forward pass produces a variable-length bitstream
with an embedded self-describing header; inverse pass reconstructs the original
symbol array exactly.

- **Forward:** `T[] → uint8_t[]`  cuSZ Huffman bitstream with embedded `phf_header`
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
| `setBookSource(s)` | `HuffmanBookSource` | `PerBlock` | `Adaptive`/`Fixed` reuse one codebook — see [Pre-built codebooks](#huffman-prebuilt-book) |
| `setAdaptiveFloorShift(k)` | `uint8_t` | 24 | `Adaptive` frequency floor, as `max_freq >> k` |
| `setRefitThreshold(r)` | `float` | 1.2 | `Adaptive` refits when the bit rate degrades past `r`x the fitted rate (0 = never) |
| `setRefitInterval(n)` | `uint32_t` | 0 | `Adaptive` refits every `n` calls (0 = off) |
| `setValidateSymbolRange(b)` | `bool` | `true` | GPU check that symbols stay in `[0, bklen)` when a book is pinned |
| `setFixedBookFromModel(spec)` | `HuffmanBookSpec` | — | Build a `Fixed` book from an analytic distribution |
| `setFixedBookFromFreq(f, n)` | `const uint32_t*`, `uint32_t` | — | Build a `Fixed` book from a frequency table |

### Setting bklen

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
`execute()` forces a full reallocation of all Huffman internal buffers on the next call.

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
the Huffman codebook to cover the full symbol range.  With zigzag, all codes land in
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
book_source = "PerBlock" # optional; "PerBlock" (default), "Adaptive", or "Fixed"
book_floor_shift = 24    # Adaptive only; frequency floor as max_freq >> shift
book_refit_threshold = 1.2  # Adaptive only; refit when bit rate degrades past this (0 = never)
book_refit_interval  = 0    # Adaptive only; refit every N calls (0 = off)
validate_symbol_range = true # check symbols stay in [0, bklen) when a book is pinned
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

### Adaptive — sample once, reuse (recommended)

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

`Adaptive` is not a fully GPU-resident encoder. On a fitting call, the histogram is
computed on the GPU but copied to the host, and the canonical tree is built on the
CPU. Reused-book calls skip that work, but the default symbol-range guard still
copies a verdict to the host, and cuSZ's coarse encode path always copies partition
metadata to the host for its prefix sum. Disabling the guard removes only the first
of those reused-book synchronizations; the partition-prefix synchronization remains.

**There is no fully GPU-only Huffman mode.** `PerBlock`, `Adaptive`, and `Fixed`
all use the host-coordinated coarse encoding path. The book source changes when
the codebook is built or reused, not how encoded partitions are assembled.

### Fixed — a book chosen up front

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

### Things to know about both

- The book is fitted to the first block only, so ratio drifts across a long run.
- **Refitting.**  `setRefitThreshold(r)` (default 1.2) rebuilds the book once the
  encoded bit rate degrades past `r` times the rate it achieved when fitted. The refit lands on the *next*
  call; the current one is already encoded.  This handles a distribution drifting
  toward less compressible: over 26 CESM `CLOUD` levels it cuts the loss from 42% to
  0.9%.  It cannot see a block that is *more* compressible than the fitted one while
  still being poorly served by its book. `setRefitInterval(n)` refits unconditionally
  every `n` calls and covers that case at the cost of one histogram per `n`.
  **When every call carries a genuinely different variable, prefer `PerBlock`**: a
  refit always lands one call late, so it cannot match a book fitted to the block
  being encoded.
- **Codes cannot exceed 27 bits.**  `HuffmanWord<4>` holds the code in a 27-bit
  field. A `PerBlock` book that exceeds it falls back to a reusable frequency-floored
  Adaptive book; a caller-supplied `Fixed` book is rejected. `Adaptive` flattens its
  frequency floor until the book fits.

`setFixedBookFromFreq()` rejects a zero-frequency bin, and both setters reject a
book whose codes will not fit `HuffmanWord<4>`'s 27-bit code field (widen the model
scale or reduce the frequency dynamic range).  Model-derived books give every symbol
a strictly positive frequency, so they can always encode any in-range symbol; a book
trained on a sample of real data cannot make that promise.

Only the model form round-trips through TOML — a raw frequency table exceeds the
128-byte per-stage config slot and has to be re-supplied through the C++ API.

Fixed books do **not** make the stage graph-capturable; see below.

---

## Execution flow (CPU–GPU movement pattern) {#huffman-execution}

HuffmanStage is unusual among FZGPUModules stages in that its default configuration
requires two host-synchronous operations inside each forward execute call — one to
transfer the histogram to the CPU for codebook construction, and one to synchronize
partition metadata for prefix-sum computation.

`Adaptive` removes steps 1–4 below after a book has been fitted, and `Fixed` omits
them once its configured book is resident. Barrier 2 remains in every call.

### Forward pass

```
GPU  ←input T[]                                       output uint8_t[]→
  1. GPU histogram (p2013 shared-mem atomics)         d_freq[bklen]
  2. cudaMemcpyAsync D2H                              h_freq[bklen]
     └─ cudaStreamSynchronize() ◄── HOST BARRIER 1
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
  8. GPU memcpy_merge — assemble Huffman blob         d_encoded
  9. cudaMemcpyAsync D2D → pipeline output buffer
```

**Consequence:** the CPU-visible barriers per compress call make this stage
latency-bound. `Adaptive`/`Fixed` removes barrier 1; barrier 2 remains in every
supported configuration. `isGraphCompatible()` returns `false` regardless, because
`encode()` returns `total_nbit` / `total_ncell` to the host to assemble `phf_header`
before the H2D merge. Making the stage capturable would additionally require
computing partition offsets, assembling the header, and merging the stream on the
device. A device-resident option is under consideration but is not currently
implemented.

### Inverse pass

```
GPU  ←input uint8_t[]                                 output T[]→
  1. cudaMemcpy D2H (blocking) — read phf_header      128-byte header
  2. GPU decode — revbk lookup → symbol reconstruction
```

---

## Internal buffer layout

`HuffmanStage<T>` holds a `phf::Buf<T>` object (the internal cuSZ name), lazily
allocated on first execute and reused while the input stays within capacity. Its
device and pinned-host scratch is allocated persistently through the pipeline memory
pool.

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

The cuSZ Huffman bitstream is self-describing: the 128-byte `phf_header` is embedded at
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

**Huffman scratch is pool-owned, but not allocated from the stream-ordered arena.**
`phf::Buf<T>` asks `MemoryPool` for persistent device and pinned-host allocations.
Those APIs call `cudaMalloc` and `cudaMallocHost` directly; unlike the pool's normal
`allocate(size, stream)` path, they take no stream and are not ordered after earlier
work on one. `MemoryPool` supplies lifetime ownership, accounting, and cleanup here,
not suballocation or asynchronous reuse.

The allocations keep stable addresses for the lifetime of `phf::Buf<T>`. A pool
`reset()` does not release them, and buffer coloring cannot alias them with stage I/O
or another stage's scratch. When `phf::Buf<T>` is destroyed, `MemoryPool` untracks
them and calls `cudaFree` / `cudaFreeHost`; they are not returned to the reusable
`cudaMemPool_t` arena. Their bytes appear in `getPersistentDeviceBytes()` and
`getPersistentPinnedBytes()`, but do not count against the stream-ordered arena's
`pool_size_multiplier`. This allocation property does not make Huffman graph
compatible: its execution still contains host synchronizations.

**Reallocation on capacity growth.**  `phf::Buf<T>` is reallocated only when the
input element count grows past the previously allocated capacity (`cap_inlen_`), or
when `bklen` changes.  Calls with smaller input reuse the existing buffer without
reallocating.  The `phf_header` embedded in the output always records the actual
element count (not the allocation capacity), so encode and decode are always
consistent.  Initial allocation and capacity-growth events incur full GPU allocator
overhead; steady-state or shrinking workloads do not.

---

## Acknowledgements

`HuffmanStage` incorporates cuSZ's Huffman source files (`hf.h`, `hf_bk*.cc`,
`hf_buf.cc`, `hf_canon.cc`, `hf_hl.cc`, `hf_kernels.cu`, `hf_impl.hh`) vendored
and adapted from `origin/v1.1.0_dev` of the **cuSZ** repository (BSD-3-Clause).
cuSZ calls this implementation `phf` internally; that name is retained in its
namespace, structures, and file-format types. Changes are documented at the top
of each adapted file.

> cuSZ team (UChicago Argonne National Laboratory, Indiana University, and others).
> *pSZ/cuSZ: A GPU-Based Error-Bounded Lossy Compressor for Scientific Data.*
> https://github.com/szcompressor/cuSZ

See `THIRD_PARTY.md` for the full license text.
