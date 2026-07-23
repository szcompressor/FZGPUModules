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

`RLEStage` does not expose `setChunkSize()` or other tuning knobs. Run detection
and output packing are managed internally.

---

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

```
[uint32_t: num_runs]
[T x num_runs: run values (4-byte aligned)]
[uint32_t x num_runs: run lengths]
```
