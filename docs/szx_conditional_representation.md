# SZx as a conditional per-block representation {#szx_conditional_representation}

**Status:** design note — Checkpoint 1 (semantic specification) and Checkpoint 2
(general structural contract) of the SZx modularization. The monolithic
`SZxStage` (`modules/fused/szx/szx_stage.{h,cu}`, `StageType::SZX = 36`) is
**still the shipped, supported implementation and the correctness/performance
oracle**. Nothing here is wired into the pipeline yet. Do not claim SZx is
"modular" until the conditional archive + inverse routing described in
Checkpoint 3 actually round-trips against this oracle (Checkpoint 4).

This note supersedes nothing; it is the working spec that the
`BlockClassifier → ConditionalSplit → branches → ConditionalMerge` graph
(Checkpoint 3) must satisfy. Related: `memory/fsz_szx_szp_generalization_handover.md`
(cost-oracle / execution-layout generalization, a *separate* axis), and
`docs/experimental/szp.md` (SZp, which — unlike SZx — *is* a plain linear chain
and was quarantined rather than decomposed).

---

## Checkpoint 1 — Semantic specification

Everything in this section is read directly from the current implementation and
is the behaviour the modular graph must reproduce.

### 1.1 Geometry

* Input: `float[]` or `double[]` (`T`), `n` elements, 1-D logical order. SZx does
  **not** use `setDims()` — block `b` is elements `[b·B, min((b+1)·B, n))` in
  linear order regardless of `nx/ny/nz`.
* `B = block_size_`, default **128**, constrained to `[1, 4096]`
  (`setBlockSize` throws otherwise).
* `nb = ceil(n / B)`. The **final block is partial** when `n % B != 0`; its
  `len < B`. All per-block loops use `len`, and per-block cost uses `len`, so
  partial blocks are already handled — the modular graph must preserve this
  (no zero-padding of the tail; a padded tail would change classification and
  bit width).
* `n == 0` ⇒ empty archive, `actual_output_size_ = 0`.

### 1.2 Error-bound resolution (`resolveAbsEb`)

* `SZxErrorMode::ABS` (0): `abs_eb = user_eb`, `value_base = 0`.
* `SZxErrorMode::NOA` (2): device `Min`/`Max` reduce over **all `n` elements**
  (no dim clamp — SZx has no padding), `value_base = max - min`,
  `abs_eb = user_eb · value_base`. Host readback ⇒ NOA forward is **not**
  graph-capturable; ABS forward is (size readback deferred to `postStreamSync`).
* There is **no exact per-element REL path**. `abs_eb <= 0` after resolution
  throws ("SZx is lossy and requires error_bound > 0").
* Both `abs_eb_` and `value_base_` are serialized (see 1.5); the inverse uses the
  stored `abs_eb_` only and never re-derives it.
* Quantisation step is `2·abs_eb` everywhere (`q = llround((x - ref) / (2·eb))`),
  double-precision intermediate even for `T = float`.

### 1.3 Per-block classification (`classifyKernel`, one thread per block)

1. Scan block min `mn` / max `mx` in `double`.
2. **Reference value** `refv = (T)(0.5·(mn + mx))` — the **midpoint of the block
   range, stored at `T` precision**. (The header comment historically said
   "the mean"; that is wrong — it is the midpoint. Fixed in this pass.) `ref =
   (double)refv` is what both branches actually subtract.
3. Classify:
   * **constant** ⇔ `(mx - mn) <= 2·abs_eb`. The whole block is then within the
     bound of `refv`, so only `refv` is stored. Boundary is inclusive
     (`range == 2·eb` ⇒ constant).
   * **non-constant** otherwise.
4. Emit **2 meta bytes per block**: `meta[2b+0] = type` (`0` constant / `1`
   non-constant), `meta[2b+1] = width` (`0` for constant; else `w`).
5. **Bit width** `w` for a non-constant block: `w = bitWidth(max_b zigzag(q_i))`
   where `q_i = llround((x_i - ref) / (2·eb))` over the whole block, `zigzag`
   is the standard `(q<<1) ^ (q>>63)` map, and `bitWidth(u) = 64 - clzll(u)`
   (`0` when `u == 0`). `w` can be `0` (all residuals zero but range still
   `> 2·eb` — impossible in practice, but well-defined). `w` ranges `[0, 64]`.
6. **Per-block payload byte cost** (`cost[b]`, drives the offset scan):
   * constant: `sizeof(T)`.
   * non-constant: `sizeof(T) + ceil(len·w / 8)`.

`getConstantBlockFraction()` / `getRunNotes()` report the constant fraction, but
note: **`const_block_frac_` is never assigned in the current `.cu`** — it stays
`0.0`, so the run note never fires. (Latent bug; the modular
`BlockClassifier` should actually populate it. Recorded here, not fixed in this
pass since it changes reported output.)

### 1.4 Archive layout (`outputs[0]`)

```
[ meta region : 2·nb bytes = nb × {type u8, width u8} ]
[ payload region : per block, at exclusive-scan(cost) byte offset ]
    constant block b     :  refv                      (sizeof(T) bytes)
    non-constant block b :  refv (sizeof(T))  ||  residual codes (len·w bits,
                                                   LSB-first within each code,
                                                   codes concatenated, byte-
                                                   aligned only at block start)
```

* Payload offsets are **not stored**: the inverse recomputes `cost[b]` from the
  meta region (`costFromMetaKernel`) and re-runs the identical
  `cub::DeviceScan::ExclusiveSum`. This makes the meta region authoritative and
  the offset scan a pure function of it — a property the modular selector stream
  must keep.
* The `SZxConfig` (48 bytes: `data_type`, `eb_mode`, `block_size`,
  `num_elements`, `error_bound = abs_eb`, `value_base`) travels in the **FZM
  stage-config slot**, not in `outputs[0]`. `num_elements` is what sizes the
  inverse; `block_size` defaults back to 128 if a zero is deserialized.
* Output size is data-dependent ⇒ `estimateOutputSizes` returns a safe upper
  bound (`2·nb + nb·sizeof(T) + n·8 + 64`) and `postStreamSync` reads the true
  size back from `d_block_offset_[nb-1] + d_block_cost_[nb-1] + meta_bytes`.
* Round-trips against itself; **NOT byte-compatible with the upstream SZx
  container**.

### 1.5 Reconstruction (`decodeKernel`)

Per block, from `refv = loadRef<T>(payload + off[b])`:

* constant (`type == 0`): `out[i] = refv` for all `i` in the block (exact
  broadcast, no arithmetic).
* non-constant (`type == 1`): `out[i] = (T)(ref + (double)unzigzag(getBits(w)) ·
  2·eb)` where `ref = (double)refv`. `eb` here is the stored `abs_eb_`.

Error characteristics the graph must match:
* constant block: `|x_i - refv| <= (mx - mn)/2 <= abs_eb` (why the classifier
  uses the midpoint, not the mean — the midpoint minimises worst-case error).
* non-constant block: standard `2·eb` quantiser ⇒ `|x_i - x̂_i| <= abs_eb`
  (plus one `float` rounding when `T = float`).

### 1.6 What is SZx-specific vs. general

| Aspect | General (reusable contract) | SZx-specific (a branch/policy detail) |
|---|---|---|
| Fixed block geometry, linear order, partial final block | ✅ block geometry contract | block size 128 default |
| Per-block **classification** producing a small tag | ✅ `BlockClassifier` → selector stream | the predicate `(max-min) <= 2·eb` |
| **Selector/tag stream** owned by the split, meta-authoritative, offsets derived not stored | ✅ selector ownership + derived-offset rule | 2 bytes/block encoding `{type,width}` (width is really a *branch* datum) |
| Routing blocks to one of **k** representation branches | ✅ `ConditionalSplit` (k=2 here) | which 2 branches |
| **Constant/reference-value branch**: emit one value per block, broadcast on inverse | ✅ a reusable `BlockVariant` | midpoint-of-range reference; `sizeof(T)` bytes |
| **Predictive-residual branch**: subtract per-block reference, quantise to `2·eb`, zigzag, fixed-width bit-pack | ✅ a reusable `BlockVariant` (this is ~`Quantizer(linear)` + per-block bitpack) | reference = range midpoint (not a Lorenzo prediction); no outlier stream; no entropy coder |
| Variable per-branch, per-block payload size; merge into one self-describing buffer | ✅ `ConditionalMerge` + offset scan | payload = `[refs...][packed residuals...]` interleaved per block |
| Error-bound modes | ✅ ABS + range-relative resolve-to-abs, stored, inverse reads stored value | no REL; NOA range over all n |
| f32/f64 | ✅ `T` threaded through classifier + both branches + merge | ref stored at `T` precision, residual math in `double` |

**Not general / explicitly out of scope:** arbitrary control flow, data-dependent
branch *counts*, nested conditionals, cross-block dependencies. This is *bounded,
block-local, k-way* selection with `k` fixed at graph-build time.

---

## Checkpoint 2 — General structural contract

The smallest abstraction that expresses SZx (and is reusable by, e.g., a future
"constant-or-Lorenzo" or "smooth-or-noisy" compressor) is **three cooperating
concepts plus a `BlockVariant` interface**, *not* an `SZxConditional` stage and
*not* anything the fusion planner needs to know about.

### 2.1 Concepts

```
BlockClassifier<T>                       (a normal Stage; 1 input, 2 outputs)
  in : T[n]  (+ block geometry from config: block_size, and eb params)
  out: "selector"  — u8[nb], one tag per block, values in [0, k)
       "field"     — passthrough T[n]  (so downstream branches read the raw field)

ConditionalSplit<T>                       (structural; 2 inputs, k outputs)
  in : "selector" u8[nb], "field" T[n]
  out: "branch_0" ... "branch_{k-1}"
  Semantics: logically assigns each block to exactly one branch. Physically it
  may (a) hand every branch the full field + the selector (branch masks itself),
  or (b) compact each branch's blocks. (a) is the Checkpoint-3 default — simplest,
  no gather/scatter, matches how SZx kernels already self-mask by reading meta.

<branch i> : any subgraph ending in a BlockVariant-compatible encoder
  Must consume (field | compacted blocks) + selector, and produce a
  self-describing, per-block-addressable byte buffer + a per-block byte-size
  vector (or enough to reconstruct one).

ConditionalMerge                          (structural; k+1 inputs, 1 output)
  in : "selector" u8[nb], "branch_0"..."branch_{k-1}" (bytes + per-block sizes)
  out: "output" — one archive:
         [ selector region (authoritative) ]
         [ per-block payload, ordered by block index, at exclusive-scan offsets ]
  Offsets are DERIVED from (selector, per-branch per-block sizes), never stored.

ConditionalUnmerge / inverse routing      (the ConditionalMerge inverse)
  in : archive + selector region + config (nb, block geometry, k)
  out: routes each block's payload slice to branch i's inverse, then the branch
       inverses write their blocks into the output field; selector says which.
```

### 2.2 The `BlockVariant` interface (what each branch must satisfy)

```
concept BlockVariant<T>:
  // forward
  encodeBlock(const T* block, int len, BlockParams p) -> bytes         // variable length
  blockCost(const T* block, int len, BlockParams p)   -> size_t        // == encoded length, exact
  // inverse
  decodeBlock(const uint8_t* slice, int len, BlockParams p, T* out)    // writes len elements

  // memory-size estimation for the pipeline pre-allocator
  maxBlockBytes(int len, BlockParams p) -> size_t                      // safe upper bound
```

* `BlockParams` carries `abs_eb`, `value_base`, and any branch-private scalars
  (SZx: none beyond `abs_eb`). It is **resolved once by the classifier / a front
  pass** and travels in the archive config, exactly as `SZxConfig` does now.
* `blockCost` **is** the exact-encoded-size oracle already discussed in
  `memory/fsz_szx_szp_generalization_handover.md` — the same contract, reused.
* SZx's two variants:
  * `ConstantRefVariant<T>`: `encodeBlock` = store `refv` (midpoint of block
    min/max); `blockCost` = `sizeof(T)`; `decodeBlock` = broadcast.
  * `RangeMidResidualVariant<T>`: `encodeBlock` = store `refv` then
    `bitpack(zigzag(llround((x-ref)/(2eb))), w)` with
    `w = bitWidth(max zigzag)`; `blockCost` = `sizeof(T) + ceil(len·w/8)`;
    `decodeBlock` = inverse of that. `w` is written into the selector region's
    second byte (it is branch metadata that the merge must carry per block).

### 2.3 Ownership / rules (decisions, so Checkpoint 3 has no ambiguity)

1. **Selector ownership:** the `BlockClassifier` produces the selector; the
   `ConditionalSplit` and `ConditionalMerge` treat it as read-only;
   `ConditionalMerge` writes it (verbatim) into the archive as the authoritative
   region. On inverse it is read back before any payload.
2. **Selector width:** 1 byte/block for the tag. Branch-private per-block
   metadata (SZx's width byte) is a **second selector-region lane**, sized
   `k`-max bytes/block (SZx: 1). Keep it in the selector region (not per-branch)
   so offset derivation needs only the selector region — preserving SZx's
   "meta region is a pure function generating all offsets" property.
3. **Block geometry** is fixed and identical across all branches, carried in
   config (`block_size`, `num_elements`). No branch may re-block.
4. **Variable branch sizes:** each branch reports a `u32[nb]` per-block byte
   count (0 for blocks it does not own). `ConditionalMerge` sums the owned
   counts per block, exclusive-scans, and lays payload out in **block-index
   order** (not branch order) so the inverse can seek to block `b` with only the
   selector + scan.
5. **Archive serialization:** `[selector region (2·nb for SZx)] [payload]`.
   Byte-identical to today's SZx `outputs[0]` when `k=2` and the two variants
   above are used — this is the Checkpoint-4 parity target.
6. **Inverse routing:** `ConditionalUnmerge` recomputes offsets (selector →
   `costFromMeta` per branch → scan), then for each block calls
   `branch[selector[b]].decodeBlock(slice, len, params, out + b·B)`.
7. **Memory-size estimation:** pipeline pre-allocator asks each branch
   `maxBlockBytes` and sums; `ConditionalMerge` output upper bound =
   `selector_region + Σ_b max_i maxBlockBytes_i(len_b)`. Matches today's
   `estimateOutputSizes`.
8. **Fusion:** none of these stages declare a `FusionSpec` initially. The
   automatic-fusion planner must **not** learn about `ConditionalSplit` — it is a
   fan-out point (multiple real outputs), which the linear-chain planner already
   refuses to cross. This is deliberate: keep conditional representation and
   kernel fusion orthogonal.

### 2.4 Why not just a stage

A single `SZxConditionalStage` (or teaching the planner about `SZxStage`) would
re-hide the one genuinely reusable thing — *per-block k-way representation
selection with a derived-offset self-describing archive* — inside an
SZx-specific blob, which is exactly the anti-goal in the task brief. The split
above costs one extra structural stage and a `u8[nb]` buffer versus the fused
kernel, which Checkpoint 5 measures; if the DAG is too slow, Checkpoint 5's
disposition is to register the existing fused `SZxStage` as an *internal
optimized implementation behind the DAG* (same pattern as the warp-register
cuSZp fusion: swap execution, keep the DAG nodes and the archive), **not** to
abandon the decomposition.

---

## Handoff — remaining checkpoints

**Done this pass:**
* Checkpoint 1 (this doc §1) — full semantic spec from the oracle.
* Checkpoint 2 (this doc §2) — `BlockClassifier` / `ConditionalSplit` /
  `ConditionalMerge` / `BlockVariant` contract with all ownership decisions.
* Fixed the "mean" → "midpoint" comment error in `szx_stage.h`/`.cu`.
* Characterization test `tests/stages/test_szx_semantics.cpp` pinning the
  classification boundary (`range == 2·eb` ⇒ constant), the midpoint reference
  value, the 2-byte/block meta layout, partial-block handling, and f32/f64 +
  ABS/NOA, as the Checkpoint-4 parity oracle.
* SZx left **fully intact and supported** — no code path changed.

**Not started (next session):**
* **Checkpoint 3** — implement the four stages under
  `modules/structural/conditional/` (`ConditionalSplit`/`ConditionalMerge`) and
  `modules/predictors/block_classifier/` + the two `BlockVariant`s
  (`ConstantRefVariant`, `RangeMidResidualVariant`) under
  `modules/coders/block_variant/`. Wire an `szx_composed.toml` preset:
  `BlockClassifier → ConditionalSplit → {const branch, residual branch} →
  ConditionalMerge`. Preserve: error bounds, partial final block, f32/f64,
  ABS/NOA, exact inverse.
* **Checkpoint 4** — parity vs `SZxStage` over constant / smooth / noisy / mixed
  / partial blocks, f32/f64, ABS/NOA, and full-corpus samples. Require
  round-trip validity **and** byte-identical archive (achievable per §2.3 rule
  5) before any performance work.
* **Checkpoint 5** — measure the DAG; if too slow, register `SZxStage` as an
  internal optimized implementation *behind* the conditional DAG (execution
  swap, DAG/archive unchanged), then quarantine the directly-exposed `SZxStage`
  from the umbrella header + catalog exactly as SZp was (keep `StageType::SZX =
  36` and its factory for archive back-compat; add `"SZX"` to the
  `EXPERIMENTAL` set in `scripts/check_stage_integration.py`; move docs/card/
  preset; `kLegacyStageRegistry` entry for `type = "SZx"`).

**Constraints to keep in view:** do not create an SZx-specific pseudo-stage; do
not teach automatic fusion about the monolith; do not claim SZx is modular until
Checkpoint 3's conditional archive + inverse actually round-trips against the
oracle.
