# ADMStage {#stage_adm}

**Header:** `transforms/adm/adm_stage.h`  
**Class:** `fz::ADMStage`  
**Category:** Transform (lossless)

**Common instantiation:**
```cpp
auto* adm = p.addStage<fz::ADMStage>();
adm->setDtype(fz::ADMDtype::U16);  // match upstream output type
```

---

## What it does

Remaps a `uint16_t[]` or `uint32_t[]` integer stream into a compact 8-bit symbol
domain, dramatically improving the compression ratio of a downstream entropy coder
(typically `ANSStage`).

- **Forward:** `uint16_t[]` or `uint32_t[]` → opaque ADM payload (`uint8_t[]`)
- **Inverse:** ADM payload → original integer array (exact reconstruction)

**Algorithm:**  ADM partitions the input into 512-element warp blocks and computes
a per-block center (mean).  Each element is then encoded as a (code, unary-signal)
pair relative to the center.  The resulting byte codes have a highly skewed,
low-entropy distribution ideal for GPU rANS (`ANSStage`) or Huffman coding.

The output is an opaque ADM payload — `getOutputDataType()` returns `UNKNOWN` to
opt out of downstream type checking.

---

## Stage settings

| Setting | Type | Default | Purpose |
|---|---|---|---|
| `setDtype(ADMDtype)` | `ADMDtype` | `U16` | Input element type: `U16` or `U32` |

`ADMDtype::U16` expects `uint16_t` input; `ADMDtype::U32` expects `uint32_t` input.
The dtype must match the upstream stage's output element type.

---

## Typical pipeline

### Standalone (integer array input)

```cpp
Pipeline p(in_bytes, MemoryStrategy::PREALLOCATE);

auto* adm = p.addStage<ADMStage>();
adm->setDtype(ADMDtype::U16);
p.finalize();

p.compress(d_in, in_bytes, stream);
```

### cuSZ-style Lorenzo + ADM + ANS (recommended)

```cpp
Pipeline p(in_bytes, MemoryStrategy::PREALLOCATE);

auto* lrz = p.addStage<LorenzoQuantStage<float, uint16_t>>();
lrz->setErrorBound(1e-3f);
lrz->setQuantRadius(512);
lrz->setZigzagCodes(true);

auto* adm = p.addStage<ADMStage>();
adm->setDtype(ADMDtype::U16);
p.connect(adm, lrz, "codes");  // connect to "codes" port

auto* ans = p.addStage<ANSStage>();
p.connect(ans, adm);

p.finalize();
```

**Why ADM before ANS:**  `ANSStage` operates on 8-bit symbols (256-entry alphabet).
Raw `uint16_t` quantization codes have a 65536-symbol alphabet, which ANS cannot
encode directly.  ADM acts as a symbol-domain adapter — it remaps the wide integer
stream into the 8-bit domain while preserving all information losslessly.

---

## TOML configuration

```toml
[[stage]]
name   = "adm"
type   = "ADM"
dtype  = "uint16"   # "uint16" (default) or "uint32"
inputs = [{from = "lrz", port = "codes"}]
```

---

## Execution flow (CPU–GPU movement pattern) {#adm-execution}

### Forward pass

ADM uses one of two prefix-sum strategies depending on the number of warp blocks
(`gsize = ⌈N / 512⌉`):

**Decoupled look-back path** (`gsize ≤ 1024`):

```
GPU  ←input uint16_t[]                            output ADM payload uint8_t[]→
  1. adm_map_decoupled_u16 — per-warp center + encode, decoupled prefix sum
  2. adm_concat_u16 — pack (code, signals) into contiguous payload
     └─ cudaMemcpyAsync D2H + cudaStreamSynchronize ◄── HOST BARRIER
        (reads actual output_lengths to determine actual_output_size_)
```

**Thrust fallback path** (`gsize > 1024`, ~512 K+ elements for uint16_t):

```
GPU  ←input uint16_t[]                            output ADM payload uint8_t[]→
  1. adm_map_thrust_u16 — per-warp center + encode, Thrust exclusive_scan
  2. adm_concat_u16 — pack (code, signals) into contiguous payload
     └─ cudaMemcpyAsync D2H + cudaStreamSynchronize ◄── HOST BARRIER
```

### Inverse pass

```
GPU  ←input ADM payload uint8_t[]            output uint16_t[]→
  1. adm_decompress_u16 — decode (code, signals) → original values using stored centers
```

**Consequence:** one host barrier per compress call — the stage is not CUDA Graph
compatible.  `isGraphCompatible()` returns `false`.

---

## Scratch buffers / device footprint

`ADMStage` pre-allocates 10 persistent device buffers from the pipeline
`MemoryPool` at `finalize()` time (PREALLOCATE mode) or on the first `execute()`
call (MINIMAL mode).  All buffers are grow-only.

Let `gsize = ⌈N / 512⌉` and `sig = kMaxSignalBytes` (2 for U16, 4 for U32):

| Buffer | Size formula |
|---|---|
| `d_signal_length_` | `gsize × 4` B |
| `d_output_lengths_` | `(gsize + 1) × 4` B |
| `d_centers_` | `gsize × sizeof(T)` (2 or 4 B per element) |
| `d_block_flags_` | `⌈gsize × 512 / 32⌉ × 4` B |
| `d_codes_` | `N` B |
| `d_concat_signals_` | `N × sig` B |
| `d_bit_signals_` | `N × sig` B (Thrust fallback path) |
| `d_loc_offset_` | `(gsize + 1) × 4` B |
| `d_prefix_state_` | `(gsize + 1) × 4` B |
| `d_overflow_flag_` | `4` B (debug overflow sentinel; always allocated) |

For 4096 uint16_t elements: ~68 KiB total device footprint.
For 1M uint16_t elements: ~10 MiB total device footprint.

---

## Serialized header

The FZM stage header is 12 bytes:

```
[0]     dtype         (0 = U16, 1 = U32)
[1..3]  reserved      (zero)
[4..11] num_elements  (uint64_t LE; element count of the uncompressed input)
```

`num_elements` is stored so that `estimateOutputSizes()` can return the exact
decompressed byte count for inverse-pass output buffer allocation.

---

## Limitations {#adm-limitations}

**Not CUDA Graph compatible.** A device-to-host synchronization occurs in every
forward call to read the actual payload size.  `isGraphCompatible()` returns `false`.

**Bounded-diff constraint.** Each GPU thread encodes `kChunk = 16` elements into a
fixed `kChunk × kMaxSignalBytes` byte buffer (`32` B for U16, `64` B for U32).  The
number of signal bits required per element is `⌈diff / 126⌉`, where `diff` is the
element's absolute deviation from the warp-block center.  If the total per-thread
signal bits exceed the buffer size, the kernel writes out-of-bounds.  In debug
builds (`NDEBUG` not defined) an `atomicOr` sentinel detects this condition and
`compress_u16`/`compress_u32` throw `std::runtime_error` rather than silently
corrupting data.  ADM is designed for bounded quantization codes (output of
`LorenzoQuantStage` or `QuantizerStage`) — arbitrary integer arrays with large
value ranges are not supported.

**Opaque output.** The ADM payload is not a self-describing format — it cannot be
decoded without the FZM header (which stores `num_elements`).  The payload must
always be wrapped in a `Pipeline` with `writeToFile`/`decompressFromFile` or paired
with the pipeline `decompress()` that restores the header.

**Dtype must be set before finalize().** `setDtype()` must be called before
`pipeline.finalize()` so the correct element width is used for scratch sizing and
type checking.

---

## Acknowledgements

`ADMStage` is a direct port of the ADM encode/decode kernels from **MANS**
(Wenjing Huang, Jinwu Yang, JingKai Huang, Haoquan Long, Dingwen Tao,
Guangming Tan) from `nv/adm/` in the MANS repository.  Kernel logic is
unchanged; changes from the original are documented at the top of
`modules/transforms/adm/mapping_uint16.cu` and `mapping_uint32.cu`.

> Wenjing Huang, Jinwu Yang, JingKai Huang, Haoquan Long, Dingwen Tao,
> Guangming Tan. *MANS: Multidimensional Adaptive Numerical Compressor for
> Scientific Data.* https://github.com/hpdps-group/MANS

See `THIRD_PARTY.md` for the full BSD-3-Clause license text.
