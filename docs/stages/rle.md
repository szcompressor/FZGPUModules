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
| `T` | Element type: `uint8_t`, `uint16_t`, `uint32_t`, `int32_t` |

**`RLEStage` is a class template — the type parameter is required.**

```cpp
// Correct
auto* rle = p.addStage<RLEStage<uint16_t>>();

// Wrong — does not compile
auto* rle = p.addStage<RLEStage>();
```

---

## Wire format

```
[uint32_t: num_runs]
[T × num_runs: run values  (4-byte aligned)]
[uint32_t × num_runs: run lengths]
```

---

## No chunk size setter

`RLEStage` does not expose `setChunkSize()`.  Run detection and output packing are
managed internally.

---

## Typical pipeline

```cpp
auto* lrz = p.addStage<LorenzoQuantStage<float, uint16_t>>();
auto* rle  = p.addStage<RLEStage<uint16_t>>();

p.connect(rle, lrz, "codes");
p.finalize();
```
