# Performance Tuning {#performance_tuning}

FZGPUModules exposes performance levers at three levels: pipeline-level API options
(memory strategy, CUDA Graph capture, stream count), per-stage TOML/setter
configuration, and build-time flags. This page catalogs every lever found in the
codebase, what it trades off, and the measured effect size where one exists.

---

## Pipeline-Level Levers

### Memory Strategy {#pt_memory_strategy}

| Strategy | Behavior | Effect |
|---|---|---|
| `MemoryStrategy::MINIMAL` | Allocate on demand, free at last consumer | Low peak memory |
| `MemoryStrategy::PREALLOCATE` | Allocate everything at `finalize()` | Enables buffer coloring allowing for memory reuse |

Buffer coloring under `PREALLOCATE` is **specialization-aware**: with
Pipeline Specialization on, a fused group's
fully-internal intermediate buffers are not allocated at all and the group is one
liveness point, so `Auto` reduces peak pool memory as well as runtime (on both the
compress and decompress DAG). `setColoringEnabled(false)` disables coloring
entirely. Useful for easier `compute-sanitizer` debugging.

`pool_multiplier` (third `Pipeline` constructor argument, default `3.0f`) sizes the
memory pool as `input_size × multiplier`. Too low risks a mid-run pool growth
reallocation; too high wastes memory.

### CUDA Graph Mode {#pt_graph_mode}

```cpp
pipeline.enableGraphMode(true);   // before finalize()
pipeline.finalize();
pipeline.warmup(stream);          // JIT-compiles all kernels once
pipeline.captureGraph(stream);
```

Eliminates CPU-side kernel launch/dispatch overhead on repeated `compress()` calls
by replaying a recorded CUDA graph. Requires `MemoryStrategy::PREALLOCATE`, a
non-default stream, and every stage in the pipeline to be graph-compatible,
(`isGraphCompatible()`) — `HuffmanStage` and `ANSStage` are not graph-compatible.

**Effect size:** testing has shown minimal improvement through the use of graph capture, but more experiments are needed to determine the effect of graph capture on different pipelines and workloads. The main benefit of graph capture is to reduce CPU overhead for repeated calls to `compress()`, which may be more significant in certain scenarios.

### Pipeline Specialization {#pipeline_specialization}

FZGPUModules' runtime optimization layer. You build a pipeline (a modular DAG,
one kernel per stage), and `finalize()` identifies compatible subgraphs and binds
their specialization strategies. The first execution generates and compiles the
corresponding kernels with NVRTC; later executions reuse the cache. Specialization
changes execution while retaining the declared DAG and reconstruction semantics.
Deterministic paths also retain the staged archive bytes. Paths with atomically
appended side outputs may order equivalent records differently and are validated by
their reconstructed result.

Kernel fusion is the main strategy it applies today (compatible stages collapsed
into a single kernel, keeping intermediates in registers/shared memory instead of
round-tripping each one through DRAM). The name is deliberately broader than
"fusion" because a specialization is more than a fused kernel — it also carries
in-kernel optimizations such as single-pass decoupled-lookback and an NVRTC code
generator. Auto currently admits implementations through an evidence-gated
registry and resolves overlaps by launches removed; it does not yet evaluate a
general predictive per-input performance cost model. The warp strategy does choose
among its supported execution paths per call using bounded size/rate rules. Further
runtime optimizations fit under the same umbrella.

Specialization and CUDA Graph mode are mutually exclusive (see above) — pick one.

#### Why it exists

A modular pipeline pays a "modularity tax": each stage is its own kernel, and every
intermediate buffer is written to and re-read from DRAM between stages. For a short,
memory-bound chain like the cuSZp / SZp family (`Quantizer → Lorenzo →
AdaptiveBitpack`) that inter-stage traffic is most of the runtime. Specialization
removes it by fusing the chain into one kernel — which cuts both the **time**
(no DRAM round-trip for intermediates) and the **memory** (those intermediate
buffers are never allocated; see "What it guarantees" below). Ratio, PSNR, and
NRMSE are unchanged; deterministic specializations are also archive-byte-identical.

#### Enabling it

Specialization is **off by default** — you opt in.

**From C++:**

```cpp
#include "fzgpumodules.h"
using namespace fz;

void build(Pipeline& p) {
    // ... setDims / addStage / connect ...
    p.setSpecializationPolicy(SpecializationPolicy::Auto);   // must precede finalize()
    p.finalize();
}
```

`SpecializationPolicy`:

| Value | Meaning |
|---|---|
| `Off` (default) | Every stage runs staged. No specialization. |
| `Auto` | Install every auto-enabled registered specialization that matches a chain. Registration is the current evidence gate; there is not yet a per-input predictive profitability test. This is the production setting. |
| `Force` | Also admit *experimental* specializations that have not yet cleared the gate. For correctness/perf diagnostics only — not a production default. |

`PREALLOCATE` is recommended (and required for the fused path's persistent scratch).

**From the environment** (overrides the programmatic policy):

```
FZ_SPECIALIZE=off|auto|force
```

The policy can also be declared with the pipeline in TOML:

```toml
[pipeline]
specialization = "auto"
```

`FZ_SPECIALIZE` wins over the TOML or programmatic policy, so you can flip
specialization on or off for an already-built pipeline without editing its config
or recompiling. Without an environment override, TOML and
`setSpecializationPolicy()` set the same pipeline policy before `finalize()`.

**From the CLI:** put the policy in the loaded TOML as above. There is no
dedicated command-line flag; use the environment variable when you need to
override a config for an A/B run:

```bash
FZ_SPECIALIZE=auto fzgmod-cli -c examples/presets/szp_composed.toml \
    -i data/CLDHGH.f32 -l 3600x1800 -e 1e-3 -b --report-json out.json
```

#### What it guarantees

- **Equivalent results.** Deterministic specializations produce the exact same
  archive or reconstruction bytes as staged execution. Specializations with
  nondeterministically ordered side outputs preserve reconstruction and the error
  bound, although their archive bytes may differ in record order.
- **Both directions, independently.** Compress and decompress are specialized
  separately under the same policy; a pipeline may get one, both, or neither.
- **Silent, safe fallback.** Any chain without a matching auto-enabled implementation
  simply runs staged. Registration is evidence-based, but `Auto` does not yet predict
  per-input runtime and can occasionally be slower than staged. It never changes the
  result or makes an otherwise working pipeline fail.
- **The DAG and archive are unchanged.** Specialization swaps *execution*; the DAG
  nodes, port wiring, and FZM header are built exactly as in the staged path. That
  is why decode of a specialized archive is unaffected and old archives are
  unaffected.
- **Region boundaries remain ordinary graph edges.** A declared auxiliary output
  may leave a specialized main path and continue through staged consumers. Only
  edges wholly internal to the selected region are eliminated; boundary buffers
  remain materialized, sized by the specialized runner, and consumed through the
  original port wiring in both forward and inverse execution.
- **Lower peak memory.** Under `PREALLOCATE`, specialization also shrinks the peak
  memory usage: an intermediate buffer that lives entirely inside a fused group is
  never allocated (the kernel keeps it in registers/shared memory), and the group
  counts as one liveness point so the rest of the DAG can alias around it. This
  applies to both the compress and decompress DAG. See the memory-management
  section of \ref architecture "Architecture Overview" for the mechanism.

#### Seeing what happened

**From C++:** `getSpecializationInfo()`

```cpp
const SpecializationInfo& info = p.getSpecializationInfo();
// info.policy                    — resolved policy (after any FZ_SPECIALIZE override)
// info.legal_group_count         — how many fusable chains the planner found
// info.installed_groups          — the specializations actually installed (compress)
// info.installed_inverse_groups  — installed on decompress (lazily filled after the
//                                  first decompress builds the inverse DAG)
// info.fallback_reason           — why nothing was installed (empty on a hit)
for (const auto& g : info.installed_groups)
    printf("installed %s over %zu stages\n", g.implementation.c_str(), g.stages.size());
```

`getSpecializedGroupCount()` is a shortcut for `installed_groups.size()`.

**From the CLI:** `--report-json`

The JSON report carries a `specialization` block (and, for back-compatibility, an
identical `fusion` block — prefer `specialization` in new tooling):

```json
"specialization": {
  "policy": "auto",
  "legal_group_count": 1,
  "installed_group_count": 1,
  "installed_stage_count": 3,
  "inverse_installed_group_count": 1,
  "inverse_installed_stage_count": 3,
  "fallback_reason": null,
  "groups":         [ { "implementation": "warp-register",         "stages": ["Quantizer","Lorenzo","AdaptiveBitpack"] } ],
  "inverse_groups": [ { "implementation": "warp-register-inverse", "stages": ["AdaptiveBitpack","Lorenzo","Quantizer"] } ]
}
```

This is how a benchmark sweep proves whether `Auto` actually specialized a given
cell rather than silently falling back.

> **Note.** The standalone decompress path (`fzgmod-cli -x`) reconstructs the
> pipeline from the file header and does not emit a specialization block in its
> JSON, though it still specializes the inverse when `FZ_SPECIALIZE=auto`. Use the
> benchmark path (`-b`) to observe `inverse_installed_group_count`.

**`fallback_reason` values:**

| Reason | Meaning |
|---|---|
| `policy_off` | Policy resolved to `Off` — specialization disabled. |
| `no_legal_group` | No fusable chain in the DAG (nothing to specialize). |
| `no_profitable_implementation` | Legacy token: legal chains exist, but no auto-enabled registered specialization matched their exact shape. It does not currently mean a runtime cost model measured or predicted a slowdown. |
| (empty) | At least one specialization was installed. |

#### When it does not engage

- **Policy is Off** (the default). Opt in.
- **No matching specialization.** A chain must match a registered strategy's shape.
  Today that means the warp-register family (below) or the chunk-cooperative family.
  A chain outside those falls back to staged.
- **Registry admission.** Under `Auto`, a chain must match an auto-enabled
  implementation. Admission is based on prior evidence, but the current planner
  does not predict a crossover for each input. `Force` additionally exposes
  experimental implementations for diagnostics.
- **CUDA Graph mode.** Specialization and graph capture are mutually exclusive: the
  fused runner synchronizes to read data-dependent archive lengths, so enabling
  `Auto`/`Force` disables graph mode (with a log warning). Pick one.
- **MINIMAL memory strategy** does not support the fused path's persistent scratch;
  use `PREALLOCATE`.

#### Specialization strategies

Two execution models are registered. You don't choose between them — the planner
routes each chain to the one that fits its geometry. Both are generated at runtime
via NVRTC. The first use of a new generated shape pays the one-time JIT cost; use a
warmup call before measuring repeated-use performance. The performance results in
this guide concern warm, cached execution unless stated otherwise.

| Strategy | Execution model | Pipelines it covers |
|---|---|---|
| **warp-register** | One warp owns a small block (`block_size = 32·EPL`, EPL ≤ 4), intermediates in registers + warp shuffles, no barriers. | The cuSZp / SZp family: `Quantizer(linear) → Lorenzo (or TiledLorenzo) → AdaptiveBitpack`, any block 32–128, plain or outlier. Compress **and** decompress. |
| **chunk-cooperative** | One CTA owns a byte-chunk (4096, 8192, or 16384 bytes — a per-pipeline choice, default 16 KB), intermediates ping-ponged in shared memory, barriers between ops. | The PFPL / LC family: `Quantizer(inplace) → Difference → Bitshuffle → {RZE, RRE, RARE, RAZE}`. Compress (and PFPL/RZE decompress). |

Both are **general within their family**: a pipeline you assemble from stages that
declare the right fusion ops fuses with no per-pipeline code.

The planner enumerates the maximal fusable chains, then installs a maximum
launch-removal set of non-overlapping specializations over each, so a long chain
that has a fused implementation for only part of it still gets that part specialized,
with the remainder staged.

**warp-register has three internal execution paths**, auto-selected per call (not
user-chosen), all byte-identical to each other and to staged:

| Path | When it's used | How it works |
|---|---|---|
| Thread-independent (TI) | Any chain where both the predictor and coder declare a TI policy (`Lorenzo1DPredictor` **and** `TiledLorenzo{2,3}DPredictor`, paired with `AdaptiveBitpackCoder`/`PlainRateCoder`) — not 1-D-only, despite an older assumption to the contrary. Forced via `FZ_TI=1`, or selected by `FZ_ADAPTIVE`'s runtime rate probe (average fixed-rate ≤ threshold — favors compressible data). | Each thread owns a fixed number of blocks (`FZ_TI_BPT`, default 8) with **no cross-thread dependency at all** — no lookback, no scan. The fastest path when applicable, and (as of 2026-09-08) the path most real-world large fields actually land on at representative error bounds. |
| Single-pass decoupled-lookback | Default fallback when the probe doesn't pick TI, when `elems_per_lane ≤ 2` and the field is large enough (≥ 2²⁰ blocks, or forced smaller via `FZ_SP_BPW`). | One kernel: each warp owns a block of consecutive chunks (`BlocksPerWarp`, auto-picked to keep ~32k warps in flight) and propagates a running prefix sum to the next warp via a lock-free decoupled-lookback protocol — poll a per-warp state flag, no host round-trip, no second kernel launch. |
| Two-pass (CUB scan) | Fallback for small fields, or `elems_per_lane > 2` (the single-pass body holds `BlocksPerWarp × elems_per_lane` deltas per lane in local memory across the look-back, which spills registers badly past EPL 2). | Classic two-kernel structure: a "rate" kernel computes each block's byte count, a host-orchestrated CUB exclusive scan turns that into offsets, then a "pack" kernel writes the final bitstream using those offsets. |

**The TI/warp-cooperative threshold is chain-aware, not a single global constant.**
The probe (`ti_rate_probe_kernel`) always estimates rate via a flattened serial-Lorenzo1D
model, which is an accurate proxy for 1-D chains (threshold 16.0, `FZ_ADAPTIVE_THRESH`)
but a poor one for the tiled 2-D/3-D **non-outlier** chain — a separate, lower threshold
applies there (1.4, `FZ_ADAPTIVE_THRESH_TILED`), calibrated against measured throughput
on the large-data corpus after the 16.0 default was found to misroute ~half those fields
to TI at up to 2x slower than single-pass would give. Even the chain-aware threshold is
an approximation — the real per-field data isn't perfectly separable by any single cut in
`avg_r` for that chain, so a small residual misroute (~5% on one field) is an accepted
trade-off, not a bug. Set `FZ_DEBUG_PROBE=1` to print `predictor/coder/avg_r/threshold/decision`
to stderr for any compress call — use this before trusting a profile or re-tuning either
threshold. Full derivation, the raw `avg_r` measurements, and the corpus-wide before/after
numbers: `paper_organizer/papers/FZGM/reports/fused_execution_paths_map.md` (external to
this repo, on the same host).

**Before profiling or optimizing anything in this dispatch path, check which kernel
actually ran** (the `execution_path` field in `--report-json` output / the harness's
`fusion_groups`), not which one you intended to force. Two gotchas already cost a full
day of misdirected profiling here: (1) static pipeline TOMLs under `configs/pipelines/`
hardcode an absolute error bound independent of `-m`/`-e`, so testing at your intended
relative bound requires either driving it through a harness that resolves relative bounds
itself, or hand-computing the equivalent absolute value; (2) `FZ_TI`/`FZ_ADAPTIVE`/
`FZ_SINGLEPASS` are silent no-ops unless `FZ_SPECIALIZE=auto` (or `FZ_FUSION=auto`) is
*also* set, since specialization's programmatic default is `Off`.

All these knobs are tuned via `FZ_*` environment variables (`FZ_TI`, `FZ_ADAPTIVE`,
`FZ_ADAPTIVE_THRESH`, `FZ_ADAPTIVE_THRESH_TILED`, `FZ_TI_BPT`, `FZ_SINGLEPASS`,
`FZ_SP_BPW`, `FZ_DEBUG_PROBE`) — see `WarpFusionEnvConfig` in
`modules/fused/fused_block/nvrtc_warp_fusion.cu` for the full list, defaults, and parsing
rules. These are tuning/diagnostic knobs for experiments, not something a normal caller
needs to set.

**chunk-cooperative's chunk size is a per-pipeline choice among {4096, 8192,
16384} bytes** (default 16384) — not a single hardcoded constant. The supported
set lives once, as the single source of truth, in
`modules/fused/chunk_fusion/chunk_geometry.h` (`kSupportedChunkBytes`); a
`Geom<Bytes>` template derives the rest (element count, bitshuffle plane stride)
per size, and the device fusion harness, host launcher, and runtime-generated
NVRTC source are all instantiated once per supported size — the same pattern
the RRE-family coders already used internally for their own `word_size`/
`chunk_size` kernel dispatch. Every RRE-family coder's `getFusionSpec()` gates
on chunk size being one of that set (e.g. `modules/coders/rze/rze_stage.h`); a
pipeline built with any other chunk size for those coders is still fully
correct, it just runs staged (no fusion) instead of chunk-cooperative. All
participating stages (the coder plus `BitshuffleStage`/`DifferenceStage`
upstream) must agree on the same chunk size — the planner's fusion matcher
enforces this and falls back to staged on a mismatch.

#### Making your own stages specialization-compatible

If you write a new stage and want pipelines that use it to specialize
automatically, the stage declares its fusion identity through a small set of
`Stage` virtuals (`getFusionSpec` / `getFusedOp` and their inverse counterparts),
and — for the warp family — you add the device op's forward and inverse policy
methods to the harness. The matcher and runner are declaration-driven, so a
correctly-declaring stage joins the fast path with **no changes to the planner,
matcher, or runner**.

That contract, with worked examples, is documented separately for stage authors:
\ref pipeline_specialization_internals "Making a stage specialization-compatible".

### Stream Count

`pipeline.setNumStreams(int)` (default 4, `include/pipeline/compressor.h`) sets how
many concurrent CUDA streams the DAG scheduler uses for stages at the same
dependency level.

### decompressInto() for Double-Buffered Decode Loops

The caller-allocated decompress overload (`compressor.h:306`) skips the internal
`cudaMalloc`/D2D copy/`cudaStreamSynchronize` that the pool-owned decompress path
does, allowing cross-stream overlap in a double-buffered decode loop. Caveat:
`RZEStage`/`RREStage`/`HuffmanStage`/`AdaptiveBitpackStage`'s inverse paths still
block on internal D2H reads regardless of which decompress overload is used — this
lever removes framework-level overhead, not stage-internal syncs.

### prepareInverse() for Decode-Only Pipelines

`compressor.h:339` — caches the inverse DAG across calls and builds output-buffer
metadata directly from a known uncompressed size, avoiding a throwaway warmup
`compress()` call when a pipeline is only ever used for decoding (e.g. a reader
that never compresses).

### Diagnostics Toggles

| Setter | Effect |
|---|---|
| `enableProfiling(bool)` | Per-stage CUDA event timing when on; documented as zero-overhead when disabled *(unverified here)*. |
| `enableBoundsCheck(bool)` | Runtime buffer-overwrite check; adds per-stage overhead. Always on in debug builds. |
| `setColoringEnabled(bool)` | Buffer coloring (PREALLOCATE-only) aliases non-overlapping buffer lifetimes to reduce footprint; disabling trades memory for easier `compute-sanitizer` debugging. |

### vGPU / FZ_FORCE_MEMPOOL_FALLBACK

On a vGPU, or with `FZ_FORCE_MEMPOOL_FALLBACK` set in the environment, CUDA's
stream-ordered mempool allocator is unavailable and the pipeline falls back to
`cudaMalloc`+sync — this works correctly but loses the stream-ordered allocator's
performance and disables CUDA Graph capture entirely (`mainpage.md:30`).

---

## Build-Time Levers

### CMAKE_BUILD_TYPE

`Release` (`-O3`) vs `Debug` (`-O0`, `-Wall`) — use `Release` for anything
performance-sensitive; `Debug` only for development.

### CMAKE_CUDA_ARCHITECTURES

**Defaults to "86"** (`CMakeLists.txt:45`) if not explicitly set. On any other
GPU generation — including the H100 (`sm_90`) this session's measurements were
taken on — this silently builds PTX for the wrong architecture, either failing to
run or falling back to JIT compilation at first launch (extra startup latency,
and potentially missing architecture-specific instruction selection). Always pass
`-DCMAKE_CUDA_ARCHITECTURES=<your arch>` explicitly.

### FZ_LOG_MIN_LEVEL

Compiles out `FZ_LOG()` call sites below the threshold entirely. Arguments are
never evaluated, zero-overhead for suppressed levels. Default `2` (INFO); set to `255` (SILENT) to strip all
logging for production/embedded builds where every cycle counts.

### BUILD_PROFILING / FZ_PROFILING_ENABLED

Gates NVTX3 annotations (used in `decompressFromFile` and the `profiling/`
harnesses) and `-fno-omit-frame-pointer` for complete `nsys` CPU stack walks. `OFF` compiles the instrumentation out entirely (zero cost). When `ON` but not
captured under `nsys`, the per-call NVTX range-push/pop cost is *(unmeasured)*.

### USE_SANITIZER

`ASanUbsan`, `TSan`, or `Compute` — all are development-only. Tests run
**10-100x slower** under Compute Sanitizer (`docs/building.md`); use
`--gtest_filter` to scope down when iterating. `COMPUTE_SANITIZER_DEVICE_DEBUG`
adds `-G` for source-level correlation, which is much slower still.
