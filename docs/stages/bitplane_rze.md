# BitplaneRZEStage {#stage_bitplane_rze}

`BitplaneRZEStage` is the **FZ-GPU lossless encoder**, ported as a single fused
pipeline stage. It is the lossless back-end of the FZ-GPU compressor (Zhang et
al., HPDC '23): a fused **bitplane transpose + zero-group elimination**
over `uint16_t` quantizer codes.

Header: [bitplane_rze_stage.h](../../modules/fused/bitplane_rze/bitplane_rze_stage.h) ·
Type id: `StageType::BITPLANE_RZE = 23`

## What it does

In one shared-memory pass per 4096-byte chunk (a 32×32 grid of `uint32_t`
words), the forward kernel:

1. **Bitplane transpose** — `__ballot_sync` transposes the 32×32 bit matrix of
   each chunk. This is a bitshuffle at 4-byte element width: bit *i* of all 32
   words in a row is gathered into one output word.
2. **Zero-group elimination** — a per-byte *byteflag* and per-32-byte *bitflag*
   bitmap mark which bytes survive; a block-local prefix sum compacts
   the non-zero 4-byte groups, and a single global `atomicAdd` reserves each
   block's slice of the output bitstream.

Input is interpreted as `uint16_t` symbols packed two per word, matching
FZ-GPU's quantizer codes. The transpose is **built in**, so a separate
`BitshuffleStage` is *not* placed in front of it.

The output is a self-describing byte archive (`uint8_t`).

## Relationship to BitshuffleStage + RZEStage

This stage is functionally close to
`BitshuffleStage(element_width=4) → RZEStage(levels=1)`, but it is **not**
equivalent, and it adds no new functionality the existing stages lack. It earns
its place on two grounds — **fidelity** (it reproduces FZ-GPU's exact codec and
archive) and **throughput**:

| | `BitplaneRZEStage` (fused) | `Bitshuffle(ew=4) → RZE(levels=1)` |
|---|---|---|
| **Memory traffic** | Transpose result stays in shared memory; only the compacted bitstream reaches DRAM. One pass. | Full transposed buffer is materialized in global memory, read back, then zero-eliminated. ~2× the global-memory traffic for this segment. |
| **Zero-elim granularity** | Compacts **4-byte groups** | RZE compacts **single bytes**, and can recurse (levels 2–4). Different CR on identical input. |
| **Archive format** | FZ-GPU's exact self-describing archive (per-block atomic start positions + 128-byte header) | The generic Bitshuffle + RZE stream layout |

## Template parameters

None. The stage is hard-wired to `uint16_t` input, matching the reference
kernels (`using E = uint16_t`). `uint8_t` / `uint32_t` support would require new
kernels.

## Stage settings

None. The chunk layout (`pad_len`, `chunk_size`, `grid_x`) is fully derived from
the input length. Inputs shorter than a 4096-byte multiple are zero-padded
inside the stage; the original length is recorded so decode trims back to it.

## Graph compatibility

**Not graph-compatible** in either direction. The forward pass does a blocking
`cudaStreamSynchronize` + 4-byte D2H to read the compacted bitstream length
before writing the archive header; the inverse pass does a 128-byte D2H to read
the archive header before launching the decode kernel.

## Ports

- **Forward:** 1 input (`uint16_t` codes) → 1 output (`uint8_t` archive).
- **Inverse:** 1 input (archive) → 1 output (`uint16_t` codes).

Wire it downstream of a quantizer's `codes` port:
`p.connect(bprze, quant, "codes")`.

## Typical pipeline

The faithful FZ-GPU pipeline:

```cpp
auto* quant = p.addStage<QuantizerStage<float, uint16_t>>();
quant->setErrorBound(1e-2f);
quant->setErrorBoundMode(ErrorBoundMode::ABS);
quant->setQuantRadius(32768);
auto* bprze = p.addStage<BitplaneRZEStage>();
p.connect(bprze, quant, "codes");
p.finalize();
```

## TOML configuration

```toml
[[stage]]
type = "BitplaneRZE"
inputs = [{ from = "quant", port = "codes" }]
```

No tunable keys — the stage has no parameters.

## Archive layout

Self-describing, with a 128-byte header at byte 0:

```
[0..127]              ArchiveHeader (original_len + entry[0..4] byte offsets)
[entry[1]..]          bitflag array  (uint32 × chunk_size)
[entry[2]..]          start positions (uint32 × grid_x)
[entry[3]..]          compacted bitstream (uint32 × total compacted words)
[entry[4]]            = total archive bytes
```

The FZM stage header stores only `original_len` (8 bytes, the pre-pad `uint16`
symbol count) so a cold decompress can size its output buffer.

## Acknowledgements

The encode/decode kernels are the FZ-GPU lossless codec (Boyuan Zhang, Jiannan
Tian, et al., "FZ-GPU", HPDC '23), BSD-3-Clause, as vendored in the cuSZ
repository. The stage wrapper, memory-pool integration, and padded-input
handling are FZGPUModules code. See \ref third_party_notices "Third-party notices".
