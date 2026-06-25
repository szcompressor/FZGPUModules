# AdaptiveBitpackStage {#stage_adaptive_bitpack}

**Header:** `modules/coders/adaptive_bitpack/adaptive_bitpack_stage.h`
**Class:** `fz::AdaptiveBitpackStage<T>`
**Category:** Coder (lossless)

---

## What it does

Per-block adaptive fixed-rate bit-plane coder — the cuSZp lossless back-end's
"plain" mode, as a modular stage. It partitions the signed input into fixed-size
blocks and, for each block, stores only as many bit-planes as the largest
magnitude in that block requires:

- **rate byte** `r` — the bit width of the largest `|value|` in the block (0 if
  the block is all zeros).
- when `r > 0`: a **sign bitmap** (`ceil(block_size/8)` bytes) followed by `r`
  **bit-planes** of `ceil(block_size/8)` bytes each. Bit `j` of byte `k` of
  plane `p` is bit `p` of `|element 8k+j|`.

Per-block payloads are concatenated using a device-wide exclusive scan of the
per-block byte costs, preceded by the array of rate bytes. The archive carries no
internal header of its own — `block_size` and `num_elements` live in the FZM
stage header.

Unlike cuSZp's fused single kernel, the offset scan here is an ordinary CUB
`DeviceScan` rather than the cuSZp decoupled look-back scan: fusing predictor +
quantizer + coder into one kernel (and lowering the scan to a single pass) is a
job for the downstream compiler, not the stage.

---

## Template parameter

`T` — signed element type: `int16_t` or `int32_t` (quantizer codes or block
deltas).

---

## Stage settings

| Setting | Purpose | Notes |
|---|---|---|
| `setBlockSize(n)` | Elements per block (fixed-rate granularity) | `[1, 1024]`; default 32 (cuSZp) |
| `setOutlierSelection(b)` | cuSZp2 per-block plain/outlier selection | off by default; see below |

```cpp
auto* ab = p.addStage<AdaptiveBitpackStage<int32_t>>();
ab->setBlockSize(32);
ab->setOutlierSelection(true);   // cuSZp2 mode (optional)
```

### Outlier selection (cuSZp2)

With `setOutlierSelection(true)`, each block independently chooses the cheaper of
two encodings:

- **plain** — pack all elements (as above), or
- **outlier** — store element 0 separately as a raw 1..`sizeof(T)`-byte magnitude
  and pack only elements 1..n-1.

This targets non-sparse, high-smoothness data: with a block-local predictor the
first element of each block is a delta-vs-0 (a full magnitude) that would inflate
the whole block's bit width. Per-block metadata grows from 1 to 2 bytes
(`[rate][sel]`, where `sel` bit 0 = is-outlier and bits 1-2 = outlier byte count
− 1). The mode is recorded in the FZM header, so a cold decompress selects the
right path automatically. This is the cuSZp **outlier** mode (SC'24); cuSZp packs
the same flags into a single rate byte, which we widen to two bytes so the full
`int32` rate range stays representable.

---

## Ports

Single input → single output.

| Direction | Port | Type |
|---|---|---|
| Forward in / inverse out | `"output"` | `T[n]` (signed codes) |
| Forward out / inverse in | `"output"` | `uint8_t[]` (archive) |

---

## Graph compatibility

`isGraphCompatible()` is **false** — the forward path does a host-blocking D2H to
read the scanned total payload length (same pattern as `BitplaneRLEStage`).

---

## Typical pipeline (cuSZp-style)

```cpp
auto* quant = p.addStage<QuantizerStage<float, uint32_t>>();
quant->setErrorBound(1e-3f);
quant->setErrorBoundMode(ErrorBoundMode::ABS);
quant->setLinearMode(true);          // signed INT32 codes, no outliers

auto* lrz = p.addStage<LorenzoStage<int32_t>>();
lrz->setBlockSize(32);               // block-local 1-D delta

auto* ab = p.addStage<AdaptiveBitpackStage<int32_t>>();
ab->setBlockSize(32);

p.connect(lrz, quant, "codes");
p.connect(ab,  lrz);
p.finalize();
```

`block_size` on the coder need not match the Lorenzo block, but for faithful
cuSZp both are 32.

---

## TOML

```toml
[[stage]]
type = "AdaptiveBitpack"
input_type = "int32"   # or "int16"
block_size = 32
outlier_selection = false   # true = cuSZp2 per-block plain/outlier selection
```

---

## Acknowledgements

The per-block adaptive fixed-rate bit-plane scheme is from cuSZp (Yafan Huang et
al., SC'23/SC'24). This is an independent reimplementation of the published
scheme — no cuSZp source is copied. See `THIRD_PARTY.md` and
`memory/cuszp_stages.md`.
