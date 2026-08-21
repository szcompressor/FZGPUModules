# ANSStage {#stage_ans}

**Header:** `modules/coders/ans/ans_stage.h`  
**Class:** `fz::ANSStage`  
**Category:** Coder (lossless)

**Common instantiation:**
```cpp
auto* ans = p.addStage<fz::ANSStage>();
// default prob_bits=10, no further configuration required
```

---

## What it does

Entropy-encodes a byte stream using GPU-accelerated rANS (range Asymmetric
Numeral Systems) coding via the vendored dietGPU kernel library.  Forward pass
produces a self-describing ANS bitstream (`ANSCoalescedHeader` + rANS data);
inverse pass reconstructs the original bytes exactly.

- **Forward:** `uint8_t[] → ANS bitstream`  (`ANSCoalescedHeader` + rANS data)
- **Inverse:** `ANS bitstream → uint8_t[]`  Exact reconstruction

The stage is **byte-transparent**: its input and output `DataType` is `UNKNOWN`,
so the pipeline does not enforce type compatibility at the connection point.  This
allows `ANSStage` to accept any upstream output as raw bytes — most commonly the
`uint16_t` quantization codes produced by `LorenzoQuantStage` or `QuantizerStage`.

---

## Stage settings {#ans_stage_settings}

| Setting | Type | Default | Purpose |
|---|---|---|---|
| `setProbBits(n)` | `uint8_t` | `10` | ANS probability resolution (table size = 2^n) |

### prob_bits limitation

**Only `prob_bits=10` is supported in this build.**  The dietGPU rANS kernels are
compiled as explicit template instantiations for `kANSDefaultProbBits=10` in
`modules/coders/ans/dietgpu/GpuANS.cu`.  Calling `setProbBits(n)` with `n != 10`
and then calling `execute()` throws `std::runtime_error`.  The setter is reserved
for a future build that adds instantiations for `prob_bits` 9 and 11.

---

## Typical pipeline

### Standalone (byte array input)

```cpp
Pipeline p(in_bytes, MemoryStrategy::PREALLOCATE);
auto* ans = p.addStage<ANSStage>();
p.finalize();

p.compress(d_in, in_bytes, stream);
```

### cuSZ-style Lorenzo + ANS

```cpp
Pipeline p(in_bytes, MemoryStrategy::PREALLOCATE);

auto* lrz = p.addStage<LorenzoQuantStage<float, uint16_t>>();
lrz->setErrorBound(1e-3f);
lrz->setQuantRadius(512);
lrz->setZigzagCodes(true);   // maps codes to [0, 1024] — compact symbol range

auto* ans = p.addStage<ANSStage>();
p.connect(ans, lrz, "codes");  // must connect to "codes" port, not "output"
p.finalize();
```

**Why `zigzag_codes=true`:**  Raw signed-delta codes from `LorenzoQuantStage`
span a sparse subset of the full uint16 range.  Zigzag remaps them to a compact
`[0, 2*radius-2]` range, concentrating the probability mass and improving rANS
compression ratio.  Without zigzag, most of the 256-bucket histogram will be near
zero, wasting table entries.

---

## TOML configuration

```toml
[[stage]]
name      = "ans"
type      = "ANS"
prob_bits = 10        # optional; only 10 is currently supported
inputs    = [{from = "lrz", port = "codes"}]
```

---

## Execution flow (CPU–GPU movement pattern) {#ans-execution}

### Forward pass

```
GPU  ←input uint8_t[]                                  output ANS bitstream→
  1. cudaMemsetAsync — zero histogram (256 × uint32_t)
  2. GPU_histogram_generic<uint8_t> — 256-bucket privatized histogram
  3. ansCalcWeights — normalize histogram → encode table (uint4[256])
  4. ansEncodeBatch<10, 4096> — warp-parallel rANS encode, 1 warp/block
  5. batchExclusivePrefixSum — align compressed word counts
  6. ansEncodeCoalesceBatch<64> — pack blocks into contiguous output stream
     └─ cudaMemcpyAsync D2H + cudaStreamSynchronize ◄── HOST BARRIER
        (reads ANSCoalescedHeader to determine actual_output_size_)
```

### Inverse pass

```
GPU  ←input ANS bitstream                              output uint8_t[]→
  1. cudaMemcpyAsync D2H + cudaStreamSynchronize ◄── HOST BARRIER
     (reads ANSCoalescedHeader: prob_bits, num_blocks, uncompressed word count)
  2. ansDecodeTable<256> — build 1024-entry decode lookup table on GPU
  3. ansDecodeKernel<128, 10, 4096> — occupancy-tuned warp-parallel decode
```

**Consequence:** one host barrier per compress call, one per decompress call —
the stage is not CUDA Graph compatible.  `isGraphCompatible()` returns `false`.

---

## Scratch buffers / device footprint

`ANSStage` pre-allocates 7 persistent device buffers from the pipeline
`MemoryPool` at `finalize()` time (PREALLOCATE mode) or on the first `execute()`
call (MINIMAL mode).  All buffers are grow-only — if a subsequent call arrives
with a larger input, a reallocation occurs; smaller inputs reuse the existing
capacity.

| Buffer | Size formula |
|---|---|
| `d_temp_histogram_` | `256 × 4` = 1 KiB |
| `d_table_` | `256 × 16` = 4 KiB (uint4 per symbol) |
| `d_compressed_blocks_` | `max_blocks × 5248` B (5120 B data + 128 B ANSWarpState) |
| `d_compressed_words_` | `max_blocks × 4` B |
| `d_comp_words_prefix_` | `max_blocks × 4` B |
| `d_temp_prefix_sum_` | CUB temp storage (negligible for ≤512 blocks) |
| `d_decode_table_` | `1024 × 4` = 4 KiB (2^prob_bits entries) |

where `max_blocks = ceil(inlen / 4096)`.

The histogram launch parameters (grid size, block size, shared memory, elements
per block) are precomputed at scratch-allocation time by
`GPU_histogram_generic_optimizer_on_initialization` and reused every call —
there is no per-call optimizer overhead.

---

## Serialized header

The FZM stage header is 12 bytes:

```
[0]     prob_bits       (uint8_t)
[1..3]  reserved        (zero)
[4..11] original_bytes_ (uint64_t LE; uncompressed byte count of the forward input)
```

`original_bytes_` is stored so that `estimateOutputSizes()` can return the exact
decompressed size for output buffer allocation without needing a device-to-host
header peek inside `estimateOutputSizes()`.

---

## Limitations {#ans-limitations}

**Only `prob_bits=10` is supported.** See \ref ans_stage_settings "Stage settings".

**Not CUDA Graph compatible.** One device-to-host synchronization point exists in
every forward call (to read the `ANSCoalescedHeader` for the compressed size) and
one in every inverse call (to read the header before decoding).

**Byte-level encoding only.** `ANSStage` operates on `uint8_t` symbols (256-entry
alphabet).  For multi-byte integer streams (e.g., `uint16_t` quantization codes),
pair it with `ADMStage` (\ref stage_adm) which remaps the wide symbol space into
the 8-bit domain before ANS coding.

**Compression ratio depends on upstream symbol compactness.** ANS achieves its
theoretical Shannon entropy bound only when the symbol distribution is known at
encode time.  Highly uniform or sparse input byte streams compress poorly.  When
connecting to `LorenzoQuantStage`, `setZigzagCodes(true)` is strongly recommended.

---

## Acknowledgements

`ANSStage` incorporates rANS kernel headers from **dietGPU** (Meta Platforms, Inc.,
and affiliates) vendored under `modules/coders/ans/dietgpu/`.  Copyright notices
are retained verbatim in each vendored file.

> Meta Platforms, Inc. and affiliates.
> *dietGPU: GPU-based lossless compression for numerical data.*
> https://github.com/facebookresearch/dietgpu

See `THIRD_PARTY.md` for the full license text.
