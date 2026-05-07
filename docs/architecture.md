# Architecture Overview {#architecture}

This page explains how FZGPUModules is structured internally: how stages, pipelines,
and memory management fit together. You don't need to understand all of this to use
the library — the \ref mainpage "Quick Start" is a better first stop. Read this if
you want to understand *why* the API works the way it does, or if you're extending
the library.

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

Key methods every stage implements:

| Method | Purpose |
|--------|---------|
| `execute(stream, pool, inputs, outputs, sizes)` | Dispatch the GPU kernel |
| `estimateOutputSizes(input_sizes)` | Predict output buffer sizes for pre-allocation |
| `estimateScratchBytes(input_sizes)` | Request persistent scratch memory at finalize-time |
| `setInverse(bool)` | Switch between compression (forward) and decompression (inverse) |
| `serializeHeader()` / `deserializeHeader()` | Save/restore config for FZM file round-trips |
| `setDims(array<size_t,3>)` | Pass spatial dimensions to dimension-aware stages |

Stages declare *named* outputs (e.g. `"codes"`, `"outlier_errors"`) so the pipeline
can route individual outputs by name rather than by position. Most stages have a single
`"output"` port; Lorenzo-family stages have several because outlier data must travel
through the DAG separately.

### Pipeline

`Pipeline` (`include/pipeline/compressor.h`) is the user-facing API. It wraps
`CompressionDAG` and `MemoryPool` and hides buffer-ID bookkeeping behind a
named-output wiring model:

```cpp
// 1. Construct (input size is used for memory pool sizing)
fz::Pipeline pipeline(n * sizeof(float), fz::MemoryStrategy::PREALLOCATE);
pipeline.setDims(nx, ny);

// 2. Add stages and wire them
auto* lrz = pipeline.addStage<fz::LorenzoQuantStage<float, uint16_t>>(
    fz::LorenzoQuantStage<float, uint16_t>::Config{1e-4f});
auto* rle = pipeline.addStage<fz::RLEStage<uint16_t>>();
pipeline.connect(rle, lrz, "codes");   // route the "codes" output to RLE

// 3. Finalize — validates topology, assigns execution levels, allocates buffers
pipeline.finalize();

// 4. Compress / decompress
void* d_compressed; size_t compressed_size;
pipeline.compress(d_input, n * sizeof(float), &d_compressed, &compressed_size, stream);

void* d_output; size_t output_size;
pipeline.decompress(d_compressed, compressed_size, &d_output, &output_size, stream);
```

For more usage patterns — caller-allocated output, CUDA Graph capture, file I/O, and
multi-branch pipelines — see the \ref mainpage "Quick Start" and `examples/`.

### CompressionDAG

`CompressionDAG` (`include/pipeline/dag.h`) holds the graph topology: a set of
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
| FZM binary file format specification | \ref fzm_format "FZM File Format" |
| Build options and CMake presets | \ref building_from_source "Building from Source" |
| CLI usage and TOML config syntax | \ref cli_overview "CLI & Config File" |
