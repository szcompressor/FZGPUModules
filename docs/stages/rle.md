# RLEStage {#stage_rle}

**Header:** `modules/coders/rle/rle.h`  
**Class:** `fz::RLEStage<T>`  
**Category:** Coder (lossless)

---

## What it does

Run-length encoding.  Lossless.  Effective when data contains long runs of identical
values — most useful after a predictor or quantizer stage that creates repetition.

Worst-case output is `sizeof(uint32_t) + 2 × input_bytes` (all elements are unique).
`RLEStage` should only be used when the upstream stage reliably produces runs.

---

## Template parameter

| Parameter | Constraint |
|---|---|
| `T` | Element type (see available instantiations below) |

## Available instantiations

Only these types are compiled and linked — full 1/2/4/8-byte word-size coverage,
matching the LC framework's RLE_1/2/4/8:
- `RLEStage<uint8_t>` / `RLEStage<int8_t>`
- `RLEStage<uint16_t>` / `RLEStage<int16_t>`
- `RLEStage<uint32_t>` / `RLEStage<int32_t>`
- `RLEStage<uint64_t>` / `RLEStage<int64_t>`

Using any other type will result in a linker error. Common choice: `RLEStage<uint16_t>` (after quantizer codes).
The CLI's `--stages` selects the unsigned widths via `rle1`/`rle2`/`rle4`/`rle8` (default `rle` = `rle2`,
i.e. `uint16_t`); signed widths are reachable via the TOML `data_type` key or the typed API directly.

---

## Stage settings

| Setter | Default | Meaning |
|---|---|---|
| `setChunkSize(bytes)` | `0` | `0` = whole-array path. Non-zero cuts the input into independent chunks of `bytes / sizeof(T)` elements. Rounded down to a multiple of `sizeof(T)`. |

Run detection and output packing are otherwise managed internally.

### Chunked mode

Without a chunk size, encoding is a device-wide CUB scan over the whole array plus a
four-kernel dependency chain. With a chunk size, one thread block owns one chunk end
to end using a block-local scan, which collapses that to two kernels plus a scan over
`num_chunks` (not `n`). Decode gains the same block-per-chunk parallelism and needs no
device-to-host readback at all — the element count and chunk size come from the
serialized stage header — so unlike the whole-array path, chunked *decode* is also
CUDA Graph-capturable.

Measured on 128 MB of quantizer-like `uint16_t` codes (88% zeros):

| Mode | Encode | Decode | CR |
|---|---|---|---|
| whole-array | 52 GB/s | 102 GB/s | 1.50x |
| `chunk_size = 4096` | 136 GB/s | 177 GB/s | 1.50x |
| `chunk_size = 8192` | 136 GB/s | 173 GB/s | 1.50x |
| `chunk_size = 65536` | 119 GB/s | 157 GB/s | 1.50x |

**4096–8192 bytes is the sweet spot.** The ratio cost is a forced run boundary at every
chunk head plus a `4 × (num_chunks + 1)` byte offset table, which is negligible for data
with short runs but real for data with runs much longer than a chunk — a constant array
encodes to one run per chunk rather than one run total.

Chunked mode sets `getRequiredInputAlignment()` to the chunk size, so the pipeline
zero-pads the input to whole chunks transparently.

---

## Typical pipeline

```cpp
auto* lrz = p.addStage<LorenzoQuantStage<float, uint16_t>>();
auto* rle  = p.addStage<RLEStage<uint16_t>>();

p.connect(rle, lrz, "codes");
p.finalize();
```

---

## Stream layout (forward output)

Whole-array mode:

```
[uint32_t: num_runs]
[T x num_runs: run values (4-byte aligned)]
[uint32_t x num_runs: run lengths]
```

Chunked mode (`setChunkSize`):

```
[uint32_t: num_chunks]
[uint32_t x (num_chunks + 1): run offsets, run_offsets[num_chunks] == total_runs]
[T x total_runs: run values (aligned to alignof(T), then 4-byte aligned)]
[uint32_t x total_runs: run lengths]
```

## TOML

```toml
[[stage]]
type = "RLE"
data_type = "uint16"
chunk_size = 8192      # 0 (default) = whole-array mode
```
