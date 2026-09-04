# Architecture Overview {#architecture}

This page explains how FZGPUModules is structured internally: how stages, pipelines,
and memory management fit together.

---

## Design Goals

- **Composability** — users build arbitrary DAGs of stages rather than choosing from fixed compression schemes
- **Parallelism** — independent DAG branches execute on separate CUDA streams simultaneously
- **Memory efficiency** — buffer lifetimes are tracked so memory is reused; peak usage is minimized
- **Self-describing files** — the FZM format embeds the full pipeline configuration so decompression needs no external metadata

---

## Layer Model

The library has three layers. Each outer layer owns and orchestrates the one below it.

```
┌──────────────────────────────────────────────┐
│            Pipeline  (public API)            │
│   addStage<T>()  connect()  finalize()       │
│   compress()     decompress()  writeToFile() │
└─────────────────────┬────────────────────────┘
                      │ owns
┌─────────────────────▼────────────────────────┐
│          CompressionDAG  (internal)          │
│   topological sort · level assignment        │
│   stream pool · buffer lifetime tracking     │
└──────────┬──────────────────┬────────────────┘
           │ executes         │ allocates from
┌──────────▼───────┐   ┌──────▼────────────────┐
│  Stage (virtual) │   │     MemoryPool        │
│  (many impls)    │   │  CUDA pool + strategy │
└──────────────────┘   └───────────────────────┘
```

---

## Key Abstractions

### Stage

`Stage` (`include/stage/stage.h`) is the pure-virtual base class that every compression
operation inherits. Implementations live under `modules/`. From the outside a stage is
a black box: it takes one or more device buffers as input and produces one or more named
outputs.

Every stage implements a small set of virtual methods — `execute()` (dispatch the
kernel), `estimateOutputSizes()` / `estimateScratchBytes()` (buffer sizing), `setInverse()`
(forward vs. inverse), `serializeHeader()` / `deserializeHeader()` (FZM round-trips), and
`setDims()` (dimension-aware stages). The full contract, and how to implement each, is in
\ref how_to_add_a_stage "How to Add a New Stage"; \ref extending "Extending FZGPUModules"
collects that page together with the out-of-tree build pattern, the specialization
declaration contract, and the design notes behind the non-linear stages.

Stages declare *named* outputs (e.g. `"codes"`, `"outlier_errors"`) so the pipeline
can route individual outputs by name rather than by position. Most stages have a single
`"output"` port; Lorenzo-family stages have several because outlier data must travel
through the DAG separately.

### Pipeline

`Pipeline` (`include/pipeline/compressor.h`) is the user-facing API. It wraps
`CompressionDAG` and `MemoryPool` and hides buffer-ID bookkeeping behind a
named-output wiring model. The lifecycle is four steps — construct (input size sizes
the pool), `addStage<T>()` + `connect(downstream, upstream, "port")` to build the graph,
`finalize()` to validate topology and allocate buffers, then `compress()` / `decompress()`.

For the full lifecycle with code — including caller-allocated output, CUDA Graph capture,
and file I/O — see the \ref mainpage "Quick Start", the \ref api_reference "API Reference",
and `examples/`.

### CompressionDAG

`CompressionDAG` (`include/advanced/dag.h`) holds the graph topology: a set of
`DAGNode` objects (one per stage), directed edges representing data flow, and a
`BufferInfo` metadata table tracking each buffer's producer, consumer count, size,
and allocation state.

**Execution scheduling:**
1. A topological sort groups nodes into *levels* — nodes at the same level have no
   data dependency on each other.
2. Each node is assigned a CUDA stream from a round-robin pool.
3. At runtime, all nodes in a level launch concurrently on their assigned streams.
4. Before starting the next level, `cudaStreamWaitEvent()` ensures every node's
   output is ready before its consumers read it.

This means a wide DAG (many parallel branches) runs faster than a linear chain,
at no extra API cost.

### MemoryPool

`MemoryPool` (`include/mem/mempool.h`) is a thin wrapper over CUDA's stream-ordered
pool API (`cudaMallocAsync` / `cudaFreeAsync`). All intermediate pipeline buffers
are allocated from and returned to this pool during a compress or decompress call.

Two strategies control when allocations happen:

| Strategy | Behavior | Best for |
|----------|----------|----------|
| `MINIMAL` | Allocate on demand; free each buffer immediately after its last consumer reads it | Lowest peak GPU memory |
| `PREALLOCATE` | Allocate all buffers during `finalize()`; reuse them across calls | CUDA Graph capture; repeated compression of same-shape data |

`PREALLOCATE` also enables *buffer coloring*: the DAG scheduler detects which
buffers have non-overlapping lifetimes and assigns them to the same backing memory,
reducing total allocation footprint without affecting correctness.

Coloring is **specialization-aware** (see
[pipeline_specialization.md](pipeline_specialization.md)). When a fused group is
installed, (1) any intermediate buffer produced *and* consumed entirely inside the
group is never allocated at all — the fused kernel keeps it in registers/shared
memory — and (2) the whole group is treated as a single synthetic operation for
liveness, so its inputs and final output are the only live buffers and anything
outside the group can alias around it. Net effect: specialization *lowers* peak
pool memory on both the compress and decompress DAG, on top of the throughput win.
The one exception is a multi-stream DAG that also has a fused group, where coloring
stays conservative to avoid cross-stream anti-dependencies.

**Persistent allocations** are a second tier for stage-internal scratch that must live
for the stage's full lifetime (codebook tables, histograms, partition metadata), via
`pool->allocatePersistentDevice()` / `allocatePersistentPinned()`. They are not
stream-ordered, not subject to MINIMAL/PREALLOCATE policy, and not colored, but are
tracked for footprint reporting. Stages allocate them from `Stage::onFinalize()`.

---

## Execution Flow

### Compression

```
Pipeline::compress()
  └── CompressionDAG::execute(forward)
        for each level (sequential):
          for each node in level (concurrent CUDA streams):
            allocate output buffers from MemoryPool
            Stage::execute()
            record cudaEvent
          synchronize level boundary via cudaStreamWaitEvent
        free buffers consumed and never read again (MINIMAL strategy)
  └── gather all sink buffers → concatenate into one output allocation
      layout: [num_bufs : u32][size_0 : u64][data_0][size_1 : u64][data_1] ...
```

### Decompression

```
Pipeline::decompress()
  └── parse multi-buffer header → map each buffer back to its stage input
  └── CompressionDAG::execute(inverse DAG)
        same level-parallel scheduling, stages run in setInverse(true) mode
  └── return pointer to reconstructed data buffer
```

The inverse DAG is rebuilt from the FZM file header (or from the live forward DAG)
on every decompression call — it is not separately cached.

---

## Memory Ownership

| Buffer | Owner | Rule |
|--------|-------|------|
| Input data (`d_input`) | Caller | Pipeline borrows it; caller retains ownership |
| Compressed output | Pool | Do **not** `cudaFree` — valid until next `compress()` or `Pipeline` destruction |
| Decompressed output (default) | Pool | Do **not** `cudaFree` — valid until next `decompress()` or `Pipeline` destruction |
| Decompressed output (opt-out) | Caller | Call `setPoolManagedDecompOutput(false)` — caller must `cudaFree` |
| Scratch buffers | Pool | Internal; never exposed to the caller |

---

## Logging

All library output goes through two macros defined in `include/log.h`:

| Macro | Behavior |
|-------|----------|
| `FZ_LOG(LEVEL, fmt, ...)` | Compile-time filtered — calls below the threshold compile away entirely |
| `FZ_PRINT(fmt, ...)` | Always emits — used by diagnostic functions like `printDAG()`, `printStats()` |

Log levels: `TRACE=0`, `DEBUG=1`, `INFO=2` (default), `WARN=3`, `SILENT=255`.

The compile-time threshold is set via CMake:

```bash
cmake -DFZ_LOG_MIN_LEVEL=1 ..   # include DEBUG calls
cmake -DFZ_LOG_MIN_LEVEL=255 .. # strip all logging (benchmarking / production)
```

At runtime, `Logger::setMinLevel()` can filter within the compiled-in range, and
`Logger::setLogCallback()` redirects output to a user-provided function.

---

## Related Pages

| Topic | Page |
|-------|------|
| Full stage list with constraints and options | \ref stages_overview "Stage Reference" |
| Writing a new stage (in-tree, out-of-tree, fusion, design notes) | \ref extending "Extending FZGPUModules" |
| FZM binary file format specification | \ref fzm_format "FZM File Format" |
| Build options and CMake presets | \ref building_from_source "Building from Source" |
| CLI usage and TOML config syntax | \ref cli_overview "CLI & Config File" |
| Performance tuning levers and measured effect sizes | \ref performance_tuning "Performance Tuning" |
