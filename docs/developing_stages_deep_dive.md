# Developing new stages: design & optimization decisions {#developing_stages_deep_dive}

**STATUS: WORK IN PROGRESS — outline only.** This page is a scaffold for a future
deep-dive. The mechanical "how to add a stage" steps already live in
[how_to_add_a_stage.md](how_to_add_a_stage.md); the specialization declaration
contract lives in
[pipeline_specialization_internals.md](pipeline_specialization_internals.md).
This page is meant to capture the *judgment* those two don't: how to decide a
stage's shape, ports, access pattern, and how it connects to and is optimized by
the rest of the pipeline. Sections below are stubs — fill them in with worked
examples and measured evidence as the design stabilizes. Do not treat unfilled
sections as authoritative.

---

## 1. Before you write a stage: is it a stage at all?

- [ ] When a new capability is a *composition of existing stages* vs. a genuinely
      new primitive. (Case study: SZp was retired to a composed
      `Quantizer → Lorenzo → AdaptiveBitpack` chain rather than kept as a
      monolithic stage; SZx could not be, because of its per-block conditional
      representation. Cross-link: `docs/szx_conditional_representation.md`,
      `docs/experimental/szp.md`.)
- [ ] Stage vs. structural node vs. policy-on-an-existing-stage (e.g. a coder
      variant is often a `setX()` flag, not a new class).
- [ ] The "whole compressor as a stage" anti-pattern and when it's justified
      (reference/experimental only).

## 2. Choosing the stage's shape

- [ ] Ports: number of inputs/outputs, named ports, why (`"codes"` vs `"output"`).
- [ ] Data types in/out; templating over element type; when to specialize.
- [ ] Size behavior: size-preserving vs. expanding vs. compacting; the
      bidirectional `estimateOutputSizes` contract and its worst-case bound.
- [ ] Dimension awareness (`setDims`) and when a stage needs it before `addStage`.
- [ ] Multi-output stages and escaping side outputs (outlier lists, means).

## 3. The access pattern decision (this is the load-bearing one)

- [ ] Mapping the algorithm to a `FusionAccess` role (Map / BlockLocal /
      Cooperative / TileAdaptive / Unfusable) — and why that choice determines
      everything downstream about how (and whether) the stage can be optimized.
- [ ] Block geometry: choosing a reset period / block size; alignment to warps
      (32·EPL) vs. chunks (16 KB); how the choice interacts with occupancy.
- [ ] What makes a stage a genuine fusion *barrier* (global codebook, whole-array
      scan) vs. a chain member.
- [ ] TODO: a decision flowchart from "what does my kernel read per output element"
      → role.

## 4. Connections & the DAG

- [ ] How a stage's ports become DAG edges; the `connect(down, up, port)` model.
- [ ] Buffer lifetime, coloring/aliasing, and why fused stages disable coloring.
- [ ] External input binding (`bindExternalInput`, multi-source pipelines).
- [ ] Inverse DAG construction: how forward ports map to inverse edges; what a stage
      must expose so its inverse can be rebuilt from the FZM header alone.

## 5. Optimization decisions

- [ ] Staged first, always: the staged kernel is the correctness oracle and the
      throughput baseline. Never optimize before it round-trips.
- [ ] Per-kernel optimization before fusion (coalescing, barrier removal, warp
      cooperation) — with the profiling method (ncu sectors/request, pipe
      analysis). Cross-link: `docs/decompress_kernel_optimizations.md`.
- [ ] The roofline "should I fuse?" cost model: predicting the fused ceiling from
      per-stage traffic; "knows when not to fuse". Why optimizing a bottleneck can
      *raise* the fusion ceiling (bitshuffle case study).
- [ ] Specialization declaration: forward + inverse ops, priming, the shared POD
      params contract. Cross-link:
      [pipeline_specialization_internals.md](pipeline_specialization_internals.md).
- [ ] Strategy selection: warp-register (≤64–128 elem blocks, registers) vs.
      chunk-cooperative (16 KB, shared memory) vs. thread-independent — the measured
      crossovers and when each wins.
- [ ] When NOT to add a fused backend (unprofitable; the FSZ tile-execution
      prototype was built, measured below staged, and removed — keep the semantic
      declarations, drop the kernel).

## 6. Correctness & attribution obligations

- [ ] The validation matrix: byte-identical archive, byte-exact round-trip,
      compute-sanitizer (memcheck + racecheck + synccheck for shuffles), partial
      final blocks, f32/f64, ABS/NOA/REL, full-corpus samples.
- [ ] Where longform evidence goes (`docs/codebase_notes.md`, `CN-*` IDs) vs. inline
      contracts. Cross-link: AGENTS.md "Longform notes vs. inline comments".
- [ ] Prior-work attribution (`THIRD_PARTY.md`, acknowledgements, module cards).

## 7. Worked examples (to be written)

- [ ] A Map stage from scratch (e.g. a transform), staged → declared → fused.
- [ ] A BlockLocal predictor, including its forward+inverse device policy.
- [ ] A Cooperative coder, including the exact encoded-size oracle contract.
- [ ] A structural stage (no compression op) and why it exists.
