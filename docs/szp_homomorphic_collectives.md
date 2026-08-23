# Future work: SZp compressed-domain collectives (hZCCL)

Scoping note for adding **homomorphic operations on SZp-compressed buffers** —
the capability behind hZCCL ("Accelerating Collective Communication with
Co-Designed Homomorphic Compression", Huang/Di et al., SC '24). This is a design
sketch and a phased plan, **not** an implemented feature. It is the natural
follow-on to \ref stage_szp "SZpStage".

Status: **not started.** Owner: TBD. Companion: `docs/stages/szp.md`,
`THIRD_PARTY.md` (SZp entry).

---

## 1. The idea, and why it is not a stage

A distributed reduction (`MPI_Allreduce` with `MPI_SUM`, the workhorse of data-
parallel training and many solvers) moves a buffer across ranks and sums it. When
the buffer is compressed to save bandwidth, the naïve pipeline is
**decompress → add → recompress at every hop** of the reduction tree. hZCCL's
observation is that SZp's format is *structured enough to add two compressed
buffers without fully decompressing them* — you touch quantized integers, not
floats, and re-emit a compressed result. The decompress/recompress round-trips
disappear from the critical path.

This does not fit the `Stage` abstraction. A `Stage` is a **single-buffer
transform** (`execute(inputs, outputs)` with one logical data stream per port,
compress *or* decompress). A homomorphic add is a **binary operator on two
archives producing a third** — different arity, different lifecycle, and it sits
*between* network hops rather than inside a compress pipeline. It needs its own
small abstraction, proposed here as `HomomorphicOp`, living alongside `Pipeline`,
not inside the DAG.

## 2. Why SZp specifically

SZp's archive (see \ref stage_szp "SZpStage") is, per block: a width byte plus
fixed-length zigzag-packed **1-D Lorenzo deltas of quantized integers**. Three
properties make it homomorphic-friendly:

- **Quantized integers add exactly.** If `q^A_i` and `q^B_i` are the quantized
  codes of two inputs at the same `eb` and block layout, then `q^A_i + q^B_i` is
  the quantized code of the sum *up to the bounded quantization error*. Addition
  commutes with the linear quantizer.
- **The 1-D delta is linear.** `delta(q^A + q^B) = delta(q^A) + delta(q^B)`, so
  the prediction step composes without a full un-delta/re-delta — the deltas add
  directly within a block.
- **The layout is block-local.** Blocks are independent, so the whole operation
  is embarrassingly parallel: one CUDA block (or warp) per SZp block, no
  cross-block dependency except recomputing per-block byte offsets (one CUB
  scan, exactly as in encode/decode today).

The catch is **width growth** (§5): the sum's residuals can need one more bit
than either input, so the result block may be wider and the archive larger.

## 3. Proposed interface

<!-- doc-check: skip forward-looking API sketch; these types are not implemented yet -->
```cpp
// NOT IMPLEMENTED — design sketch.
namespace fz {

// A binary operator over two SZp archives with identical (dtype, dims, eb,
// block_size). Partial-decodes each block to quantized integers in registers,
// applies the op, re-derives the block width, and re-packs — no float round-trip.
class SZpHomomorphicOp {
public:
    enum class Kind { AddSaturate, AddRenorm, ScaleByInt };

    // Both operands must share the SZpConfig fields that define the layout;
    // throws otherwise. Output is a fresh archive (offsets change, so in-place
    // is not attempted in phase 1).
    void apply(Kind kind,
               const void* d_archive_a, size_t bytes_a,
               const void* d_archive_b, size_t bytes_b,
               void** d_archive_out, size_t* bytes_out,
               MemoryPool* pool, fz::stream_t stream);
};

} // namespace fz
```

The operator reuses SZp's own device helpers (block cost/offset scan,
`getBits`/`putBits`, zigzag) verbatim; only the middle "combine two blocks of
quantized deltas" kernel is new.

## 4. Where it plugs into a collective

Two integration depths, in increasing order of effort:

1. **Single-GPU primitive (phase 1).** `apply(AddSaturate, A, B, …)` on device
   buffers. Enough to validate correctness and the error model with no network.
2. **MPI_Op over compressed buffers (phase 2).** Register a user reduction op
   (`MPI_Op_create`) whose combine step is `apply(...)`, then drive a
   **ring- or tree-allreduce** where each rank holds a compressed partial and
   every hop combines compressed buffers. The MPI dependency is optional and
   behind a build flag (`FZGMOD_WITH_MPI`); the primitive in (1) has no MPI
   dependency and is the unit-testable core.

hZCCL's headline result is the co-design: the compressor's block size is tuned to
the collective's message-chunking so a hop never straddles a block. That tuning
is a §6 concern, not a §3 one.

## 5. The two hard parts (do not hand-wave these)

**Error growth is real and must be documented, not hidden.** Adding two
`eb`-bounded reconstructions yields a result bounded by `2·eb`; an `n`-way
allreduce accumulates to `n·eb` in the worst case. **Homomorphic sum does not
preserve the single-`eb` bound** — same class of caveat as ROIBIN binning "is not
a bound" (\ref stage_roibin_split "ROIBinSplitStage"). The deliverable must ship
an error model (`bound(result) = Σ bound(operands)`) and a validation harness
that measures realized error against it, and the API must make the growing bound
queryable. A reduction that silently reports `eb` after 1000 additions is wrong
in exactly the way that is hard to catch downstream.

**Width growth forces periodic renormalization.** Each add can grow a block's bit
width by 1, so archives inflate along a long reduction chain and the "no
decompress" win erodes. Two modes:
- `AddSaturate` — keep the width, saturate codes that overflow (fast, lossy
  beyond the error model; only safe when the running bound already dominates).
- `AddRenorm` — allow width growth, and periodically (every `k` hops, or when
  mean width crosses a threshold) do a full decompress→recompress to reset
  widths and re-center. Choosing `k` is the real performance knob and is the
  first thing to measure.

Neither is free; the phase-0 prototype exists to quantify the trade.

## 6. Phasing

| Phase | Deliverable | Gate |
|---|---|---|
| 0 | Prototype `AddSaturate`/`AddRenorm` on two device archives; measure realized error vs. the `Σ bound` model and width growth over a synthetic reduction chain | Error model holds; width-growth curve understood |
| 1 | `SZpHomomorphicOp` interface + `AddSaturate`, `AddRenorm`, `ScaleByInt`; unit tests (round-trip, bound-accumulation, width-growth); **no MPI** | Tests green, compute-sanitizer clean |
| 2 | `MPI_Op` integration + ring-allreduce behind `FZGMOD_WITH_MPI`; multi-rank validation on a real reduction | Matches uncompressed allreduce within the accumulated bound; bandwidth win measured |

**Non-goals:** homomorphic multiply of two compressed operands (only
integer-scalar scale is in scope); homomorphic ops on any other compressor;
lossless operation. If a workload needs those, re-scope — do not stretch this.

## 7. Application driver (keep it honest)

Per the project's application-driven discipline, this should be pulled by a
**named collective-bound workload**, not built speculatively. Candidate drivers:
data-parallel training allreduce, or a distributed iterative solver whose
halo/reduction traffic is the bottleneck. The success metric is end-to-end
(time-to-solution or achieved bandwidth at fixed accuracy), and the accuracy
budget must be stated against the accumulated-bound model from §5 — a CR or
device-`ms` number in isolation is not evidence, for the same reasons recorded
elsewhere in this repo's measurement notes.

## 8. Rough effort

Phase 0: ~days (the SZp device helpers already exist; the new kernel is "combine
two blocks of deltas"). Phase 1: ~1–2 weeks with tests and the error harness.
Phase 2: gated on an MPI test environment and a driver workload; estimate after
phase 0 quantifies the renormalization cadence.
