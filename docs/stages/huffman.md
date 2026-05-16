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

### Setting `bklen`

`bklen` is the size of the Huffman codebook and must cover the full range of
symbols that will appear in the input.  **All input symbols must be in `[0, bklen)`.**
Symbols outside this range are detected after the histogram D2H and throw a
`std::runtime_error` naming the count of out-of-range symbols.

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
inputs      = [{from = "lrz", port = "codes"}]
```

---

## Execution flow (CPU–GPU movement pattern) {#huffman-execution}

HuffmanStage is unusual among FZGPUModules stages in that it requires two
host-synchronous operations inside each forward execute call — one to transfer the
histogram to the CPU for codebook construction, and one to synchronize partition
metadata for prefix-sum computation.

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

**Consequence:** two CPU-visible barriers per compress call make this stage
fundamentally latency-bound and incompatible with CUDA Graph capture.
`isGraphCompatible()` returns `false`.

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

**Not CUDA Graph compatible.**  Two device-to-host synchronization points exist in
every forward call (histogram D2H for codebook construction; partition metadata D2H
for prefix-sum computation).  The stage cannot be included in a graph-captured
pipeline.

**Latency-bound, not throughput-bound.**  The CPU codebook build and two D2H syncs
are serial barriers.  Kernel execution time is small relative to round-trip PCIe
latency.  HuffmanStage performs poorly on very small inputs (< ~100 KB).

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
