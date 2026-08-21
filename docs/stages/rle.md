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

---

## Stage settings

| Setter | Default | Meaning |
|---|---|---|
| `setChunkSize(bytes)` | `0` | `0` = whole-array path. Non-zero cuts the input into independent chunks of `bytes / sizeof(T)` elements. Rounded down to a multiple of `sizeof(T)`. |

Run detection and output packing are otherwise managed internally.

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

---

## Acknowledgements

Run-length encoding is a standard lossless coding technique. This stage and its
archive layout are original FZGPUModules code; no third-party implementation was
copied. It is distributed under the repository's BSD-3-Clause license.
