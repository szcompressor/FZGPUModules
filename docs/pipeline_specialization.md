# Pipeline Specialization {#pipeline_specialization}

**Pipeline Specialization** is FZGPUModules' finalize-time optimization layer. You
build a pipeline the usual way — a modular DAG, one kernel per stage — and the
library, at `finalize()`, inspects that DAG and *transparently replaces the
staged execution with an optimized specialized implementation* wherever a fast
path exists. It changes **how** the pipeline runs, never **what** it produces: the
DAG, the reconstruction, and the compressed archive bytes are all identical to the
staged run.

Kernel fusion is the specialization strategy it applies today (compatible stages
collapsed into a single kernel, keeping intermediates in registers/shared memory
instead of round-tripping each one through DRAM). The name is deliberately broader
than "fusion" because a specialization is more than a fused kernel — it also
carries in-kernel optimizations such as single-pass decoupled-lookback, an NVRTC
code generator, and a roofline-aware profitability gate that *declines* to
specialize when it would not pay off. Further runtime optimizations fit under the
same umbrella.

> **TL;DR** — add one line, `p.setSpecializationPolicy(SpecializationPolicy::Auto)`
> (or `FZ_SPECIALIZE=auto`), and eligible pipelines run their fused fast path on
> both compress and decompress, byte-identical to before. Everything else is
> diagnostics.

---

## Why it exists

A modular pipeline pays a "modularity tax": each stage is its own kernel, and every
intermediate buffer is written to and re-read from DRAM between stages. For a short,
memory-bound chain like the cuSZp / SZp family (`Quantizer → Lorenzo →
AdaptiveBitpack`) that inter-stage traffic is most of the runtime. Specialization
removes it by fusing the chain into one kernel — which cuts both the **time**
(no DRAM round-trip for intermediates) and the **memory** (those intermediate
buffers are never allocated; see "Lower peak memory" below).

Measured (CLDHGH 3600×1800, H100, eb 1e-3; staged → `Auto`):

| Pipeline | Compress | Decompress |
|---|---|---|
| `szp_composed.toml` (Quantizer→Lorenzo(128)→AdaptiveBitpack) | 97 → **268 GB/s** (2.8×) | 94 → **437 GB/s** (4.7×) |
| `cuszp2.toml` (block-32, outlier) | 109 → **206 GB/s** (1.9×) | 233 → **393 GB/s** (1.7×) |

Ratio, PSNR, and NRMSE are unchanged (the archive is byte-identical). The fused SZp
chain is also ~11× faster on compress than the standalone reference SZp kernel it
replaces.

---

## Enabling it

Specialization is **off by default** — you opt in.

### From C++

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
| `Auto` | Install every registered specialization that matches a chain **and** clears its profitability gate. This is the production setting. |
| `Force` | Also admit *experimental* specializations that have not yet cleared the gate. For correctness/perf diagnostics only — not a production default. |

`PREALLOCATE` is recommended (and required for the fused path's persistent
scratch); build the pipeline with `MemoryStrategy::PREALLOCATE`.

### From the environment (overrides the programmatic policy)

```
FZ_SPECIALIZE=off|auto|force
```

`FZ_SPECIALIZE` wins over whatever `setSpecializationPolicy()` requested, so you can
flip specialization on or off for an already-built binary without recompiling — the
basis for A/B measurement. `FZ_FUSION` is a deprecated alias kept for existing
scripts; prefer `FZ_SPECIALIZE`.

### From the CLI

The CLI has no policy flag — drive it with the environment variable:

```bash
FZ_SPECIALIZE=auto fzgmod-cli -c examples/presets/szp_composed.toml \
    -i data/CLDHGH.f32 -l 3600x1800 -e 1e-3 -b --report-json out.json
```

---

## What it guarantees

- **Byte-identical archive.** A specialized compress produces the exact same bytes
  as the staged compress. The staged path is the correctness oracle every
  specialization is validated against.
- **Byte-exact reconstruction.** A specialized decompress reconstructs the exact
  same values as the staged decompress.
- **Both directions, independently.** Compress and decompress are specialized
  separately under the same policy; a pipeline may get one, both, or neither.
- **Silent, safe fallback.** Any chain that isn't eligible, or doesn't clear the
  profitability gate, simply runs staged. Turning `Auto` on never makes a pipeline
  slower-than-staged in a way that changes results, and never fails a pipeline that
  worked staged.
- **The DAG and archive are unchanged.** Specialization swaps *execution*; the DAG
  nodes, port wiring, and FZM header are built exactly as in the staged path. That
  is why decode of a specialized archive is unaffected and old archives are
  unaffected. (Buffer *allocation* is optimized — see "Memory" below — but the
  logical layout the header records is identical.)
- **Lower peak memory, not just higher throughput.** Under `PREALLOCATE`,
  specialization also shrinks the pool: an intermediate buffer that lives entirely
  inside a fused group is never allocated (the kernel keeps it in
  registers/shared memory), and the group counts as one liveness point so the rest
  of the DAG can alias around it. This applies to both the compress and decompress
  DAG. See the memory-management section of [architecture.md](architecture.md) for
  the mechanism.

---

## Seeing what happened

### From C++: `getSpecializationInfo()`

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

### From the CLI: `--report-json`

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

### `fallback_reason` values

| Reason | Meaning |
|---|---|
| `policy_off` | Policy resolved to `Off` — specialization disabled. |
| `no_legal_group` | No fusable chain in the DAG (nothing to specialize). |
| `no_profitable_implementation` | Legal chains exist, but no registered specialization matched their exact shape, or none cleared the profitability gate. |
| *(empty)* | At least one specialization was installed. |

---

## When it does *not* engage

- **Policy is `Off`** (the default). Opt in.
- **No matching specialization.** A chain must match a registered strategy's shape.
  Today that means the warp-register family (below) or the chunk-cooperative family.
  A chain outside those falls back to staged.
- **Profitability gate.** Under `Auto`, a matched specialization still has to clear
  its gate. The gate exists so specialization "knows when *not* to fuse" — e.g. a
  chain whose fused ceiling is below the staged throughput. `Force` bypasses the
  gate for diagnostics.
- **CUDA Graph mode.** Specialization and graph capture are mutually exclusive: the
  fused runner synchronizes to read data-dependent archive lengths, so enabling
  `Auto`/`Force` disables graph mode (with a log warning). Pick one.
- **`MINIMAL` memory strategy** does not support the fused path's persistent scratch;
  use `PREALLOCATE`.

---

## Specialization strategies

Two execution models are registered. You don't choose between them — the planner
routes each chain to the one that fits its geometry. Both are byte-identical to
staged and generated at runtime via NVRTC (so only the first compress of a given
chain shape pays the one-time JIT).

| Strategy | Execution model | Pipelines it covers |
|---|---|---|
| **warp-register** | One warp owns a small block (`block_size = 32·EPL`, EPL ≤ 4), intermediates in registers + warp shuffles, no barriers. | The cuSZp / SZp family: `Quantizer(linear) → Lorenzo (or TiledLorenzo) → AdaptiveBitpack`, any block 32–128, plain or outlier. Compress **and** decompress. |
| **chunk-cooperative** | One CTA owns a fixed 16 KB byte-chunk, intermediates ping-ponged in shared memory, barriers between ops. | The PFPL / LC family: `Quantizer(inplace) → Difference → Bitshuffle → {RZE, RRE, RARE, RAZE}`. Compress (and PFPL/RZE decompress). |

Both are **general within their family**: a pipeline you assemble from stages that
declare the right fusion ops fuses with no per-pipeline code, even a shape that was
never hand-written. Adding a new predictor or coder to a family is a stage-level
declaration, not a planner edit — see the internals guide below.

The planner enumerates the maximal fusable chains, then installs a maximum
launch-removal set of non-overlapping specializations over each — so a long chain
that has a fused implementation for only part of it still gets that part specialized,
with the remainder staged.

---

## Making your own stages specialization-compatible

If you write a new stage and want pipelines that use it to specialize
automatically, the stage declares its fusion identity through a small set of
`Stage` virtuals (`getFusionSpec` / `getFusedOp` and their inverse counterparts),
and — for the warp family — you add the device op's forward and inverse policy
methods to the harness. The matcher and runner are declaration-driven, so a
correctly-declaring stage joins the fast path with **no changes to the planner,
matcher, or runner**.

That contract, with worked examples, is documented separately for stage authors:
**[docs/pipeline_specialization_internals.md](pipeline_specialization_internals.md)**.

---

## Relationship to other levers

- **CUDA Graph capture** removes CPU launch latency by replaying a captured graph;
  it is orthogonal to *what* runs but mutually exclusive with specialization (see
  above). For a short fused pipeline, specialization is usually the bigger win.
- **Buffer coloring** (`PREALLOCATE`) is not a separate opt-in for specialization —
  it is *specialization-aware*: fully-internal fused edges are elided and each
  group is one liveness point, so `Auto` reduces peak pool memory as well as
  runtime. Coloring stays conservative only for a fused group in a multi-stream
  DAG.
- **Memory strategy** and **per-stage config** are separate levers covered in
  [performance_tuning.md](performance_tuning.md).
- Longform design notes, the roofline cost model, and the measured evidence behind
  each strategy live in [docs/codebase_notes.md](codebase_notes.md)
  (`CN-FUSE-*`, `CN-CHUNK-*`, `CN-NVRTC-*`).
