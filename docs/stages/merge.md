# MergeStage {#stage_merge}

**Header:** `modules/structural/merge/merge_stage.h`  
**Class:** `fz::MergeStage` — no template parameters  
**Category:** Structural

**Common instantiation:**
```cpp
auto* merge = p.addStage<fz::MergeStage>();
merge->setSegmentNames({"vle", "anchor", "outliers"});  // N inputs, in order
```

---

## What it does

`MergeStage` concatenates **N input buffers into one contiguous buffer** (forward)
and splits that buffer back into the original N segments (inverse). It is the
structural mirror of the predictors' port asymmetry (e.g. `LorenzoQuantStage` is
`1 → 3` outputs forward / `3 → 1` inputs inverse); `MergeStage` is `N → 1`
forward / `1 → N` inverse.

## Why it exists

FZGM's DAG runs each stage on its own buffer, and the final `.fzm` archive is
assembled from the pipeline's **leaf** outputs *after* all stages have run. That
is the right model when each port is compressed independently — but some codecs
are defined to run a **single lossless pass over the concatenation of several
producer outputs**. The motivating case is cuSZ-Hi's LC back-end:

- **throughput (tp) mode** runs one LC chain over `[anchor | outliers]`;
- **ratio (cr) mode** runs one LC chain over `[Huffman(codes) | anchor | outliers]`.

There is no way to express "compress the concatenation of these ports as one
stream" by wiring ports straight into a coder: a coder takes a single input
buffer, and the archive-level concatenation happens too late (post-compression)
to feed a downstream chain. `MergeStage` materialises that concatenation *inside*
the pipeline so a downstream stage compresses it as one stream, and on decompress
it splits the reconstructed blob back into the original segments to feed the
inverse producers. Without it, a 1-to-1 port of cuSZ-Hi's merged-blob layout
would be impossible.

It is also a general-purpose building block — any time you need several producer
ports compressed together as one stream.

---

## Stage settings

```cpp
merge->setSegmentNames({"a", "b", "c"});  // defines N (=3) and the inverse
                                          // output-port names, in concat order
```
`setSegmentNames` must be called **before** `connect()` so `getNumInputs()`
reports the right count. Connect the producers in segment order:
```cpp
p.connect(merge, prodA, "portA");   // input 0  → segment "a"
p.connect(merge, prodB, "portB");   // input 1  → segment "b"
p.connect(merge, prodC, "portC");   // input 2  → segment "c"
```
Maximum 16 segments.

---

## Typical pipeline (cuSZ-Hi cr-mode merged blob)

```cpp
auto* lq  = p.addStage<LorenzoQuantStage<float, uint16_t>>();
auto* huf = p.addStage<HuffmanStage<uint16_t>>();
p.connect(huf, lq, "codes");

auto* merge = p.addStage<MergeStage>();
merge->setSegmentNames({"vle", "outlier_errors", "outlier_indices"});
p.connect(merge, huf, "output");
p.connect(merge, lq,  "outlier_errors");
p.connect(merge, lq,  "outlier_indices");

// one LC chain over the whole blob
auto* rre = p.addStage<RREStage>();   rre->setWordSize(4);
p.connect(rre, merge);
// ... → TCMS8 → RZE → archive
```

See `examples/lc_lossless_pipeline.cpp` for the full tp and cr pipelines.

---

## TOML configuration

```toml
[[stage]]
name     = "merge"
type     = "Merge"
segments = ["vle", "outlier_errors", "outlier_indices"]
inputs   = [
  { from = "huf" },                              # → segment 0
  { from = "lq", port = "outlier_errors" },      # → segment 1
  { from = "lq", port = "outlier_indices" },     # → segment 2
]
```
Input order must match `segments` order.

---

## Serialized config header

```
[0]        uint8   num_segments (N)
[1 .. 4N]  uint32  segment_size × N   (LE, connection order)
[4N+1 ..]  name table: per segment [uint8 len][len bytes]
```

---

## Acknowledgements

This structural stage and its serialized layout are original FZGPUModules
code; no third-party implementation was copied. It is distributed under the
repository's BSD-3-Clause license.
