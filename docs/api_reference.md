# API Reference {#api_reference}

This page is a concise reference for the `fz::Pipeline` class — the primary public interface.
For context on how these pieces fit together see the \ref architecture "Architecture Overview".
For per-stage configuration options see the \ref stages_overview "Stage Reference".

---

## Lifecycle at a Glance

Every pipeline follows the same call sequence:

```
Pipeline(input_size)          ← size the memory pool
  setDims(x, y, z)            ← spatial dims before addStage (if needed)
  addStage<T>(...)            ← add one or more stages
  connect(downstream, upstream, "port")  ← wire the data flow
  finalize()                  ← validate, sort, (pre-)allocate
  compress(d_in, ...)         ← compress on GPU
  decompress(d_comp, ...)     ← decompress on GPU
```

`finalize()` is the dividing line. Configuration calls go before it; execution calls go after.

---

## Enums

### fz::MemoryStrategy
Defined in `include/pipeline/dag.h`.

| Value | Behavior |
|-------|----------|
| `MINIMAL` | Allocate buffers on demand; free each one as soon as its last consumer reads it. Lowest peak GPU memory. |
| `PREALLOCATE` | Allocate all buffers during `finalize()`. Required for CUDA Graph capture. Enables buffer coloring for lower memory footprint. |

### fz::ErrorBoundMode
Defined in `modules/fused/lorenzo_quant/lorenzo_quant.h`. Used by `LorenzoQuantStage` and `QuantizerStage`.

| Value | Meaning | Notes |
|-------|---------|-------|
| `ABS` | Absolute error — `abs(x_orig - x_recon) ≤ eb` | Useful when data is homogenous in magnitude (preserve big picture) |
| `REL` | Global approximate point-wise relative — `abs(error) / abs(x_orig) ≤ eb` (approximately) | High level detail close to zero, but higher error with larger values |
| `NOA` | Value-range relative — `abs(error) / value_range ≤ eb` (norm-of-absolute) | Useful for single bounds over multiple datasets |

---

## Construction

```cpp
// Default: MINIMAL strategy, pool = 1 GiB
fz::Pipeline p;

// Sized: pool = input_size × multiplier (default 3.0)
fz::Pipeline p(input_bytes);
fz::Pipeline p(input_bytes, fz::MemoryStrategy::PREALLOCATE);
fz::Pipeline p(input_bytes, fz::MemoryStrategy::PREALLOCATE, /*multiplier=*/4.0f);

// From a TOML config file (adds stages + calls finalize() internally)
fz::Pipeline p("pipeline.toml");
```

---

## Configuration (before finalize())

| Call | Purpose |
|------|---------|
| `setDims(x)`<br>`setDims(x, y, z)` | Spatial dimensions of the input data. Push dims before `addStage()` for Lorenzo-family stages. |
| `setMemoryStrategy(strategy)` | Switch between `MINIMAL` and `PREALLOCATE`. |
| `setNumStreams(n)` | Number of parallel CUDA streams for level-based execution (default: 4). |
| `enableGraphMode(true)` | Enable CUDA Graph capture mode. Requires `PREALLOCATE`. |
| `setWarmupOnFinalize(true)` | Auto-run `warmup()` at the end of `finalize()`. |
| `setColoringEnabled(false)` | Disable buffer coloring (useful when inspecting buffers with a memory checker). |
| `enableBoundsCheck(true)` | Enable runtime buffer-overwrite detection (always active in debug builds). |

---

## Building the Graph

```cpp
// Add a stage — returns a raw pointer owned by the Pipeline
StageT* stage = pipeline.addStage<StageT>(/* stage constructor args */);

// Wire two stages (downstream reads from upstream's named output port)
pipeline.connect(downstream, upstream);              // uses "output" port
pipeline.connect(downstream, upstream, "codes");     // named port
pipeline.connect(downstream, {upstream_a, upstream_b}); // multi-input

// Finalize: validate, sort, allocate
pipeline.finalize();
```

**Important:** Call `setDims()` before `addStage()` for any Lorenzo-family stage.
The dims are pushed into the stage at add-time and again at `finalize()`.

---

## Compression

### Pool-owned output (default)
The pipeline holds the output buffer. Do **not** `cudaFree` it.

```cpp
void*  d_compressed  = nullptr;
size_t compressed_sz = 0;
pipeline.compress(d_input, input_bytes, &d_compressed, &compressed_sz, stream);
// d_compressed is valid until the next compress(), reset(), or Pipeline destruction
```

### Caller-owned output
Pre-allocate the buffer; the pipeline writes into it.

```cpp
size_t capacity = pipeline.getMaxCompressedSize(input_bytes);
void*  d_buf    = nullptr;
cudaMalloc(&d_buf, capacity);

size_t actual_sz = 0;
pipeline.compress(d_input, input_bytes, d_buf, capacity, &actual_sz, stream);
// caller owns d_buf; cudaFree when done
```

`getMaxCompressedSize(input_bytes)` returns a tight upper bound safe to use as the buffer capacity.

---

## Decompression

### Pool-owned output (default)
```cpp
void*  d_output  = nullptr;
size_t output_sz = 0;
pipeline.decompress(d_compressed, compressed_sz, &d_output, &output_sz, stream);
// d_output is valid until the next decompress() or Pipeline destruction
```

### Caller-owned output
```cpp
pipeline.setPoolManagedDecompOutput(false);

void*  d_output  = nullptr;
size_t output_sz = 0;
pipeline.decompress(d_compressed, compressed_sz, &d_output, &output_sz, stream);
// caller owns d_output; cudaFree when done
```

### Caller-allocated buffer (no internal allocation)
```cpp
size_t decomp_capacity = pipeline.getLastUncompressedSize();
void*  d_buf = nullptr;
cudaMalloc(&d_buf, decomp_capacity);

size_t actual_sz = 0;
pipeline.decompress(d_compressed, compressed_sz,
                    d_buf, decomp_capacity, &actual_sz, stream);
// caller owns d_buf; cudaFree when done
```

**Sizing helpers:**

| Call | Returns |
|------|---------|
| `getMaxCompressedSize(input_bytes)` | Upper-bound on compressed output size |
| `getLastUncompressedSize()` | Original input size from the most recent `compress()` call |

---

## Memory Ownership Summary

| Buffer | Owner | Rule |
|--------|-------|------|
| Input (`d_input`) | Caller | Pipeline borrows; never freed by the library |
| Compressed output (pool-owned) | Pipeline | Do **not** `cudaFree` |
| Decompressed output (pool-owned, default) | Pipeline | Do **not** `cudaFree` |
| Decompressed output (caller-owned) | Caller | Must `cudaFree` |
| File decompress (`decompressFromFile` static) | Caller | Must `cudaFree` |
| File decompress (`decompressFromFileInstance`) | Depends on `setPoolManagedDecompOutput()` | Same rules as `decompress()` |

---

## File I/O

```cpp
// Write compressed data to a .fzm file (compress() must be called first)
pipeline.writeToFile("output.fzm", stream);

// One-shot decompress from file — no pipeline setup needed (static)
// Output is always caller-owned; caller must cudaFree *d_output
void*  d_output  = nullptr;
size_t output_sz = 0;
fz::Pipeline::decompressFromFile("output.fzm", &d_output, &output_sz, stream);
cudaStreamSynchronize(stream);
cudaFree(d_output);

// Instance decompress from file — output ownership follows setPoolManagedDecompOutput()
pipeline.decompressFromFileInstance("output.fzm", &d_output, &output_sz, stream);

// Read the header without decompressing
auto header = fz::Pipeline::readHeader("output.fzm");
size_t original_size = header.core.uncompressed_size;

// Load / save pipeline config as TOML
pipeline.loadConfig("pipeline.toml");   // also calls finalize()
pipeline.saveConfig("pipeline.toml");   // requires finalize() first
```

See the \ref fzm_format "FZM File Format" page for the full file header specification.

---

## CUDA Graph Capture

CUDA Graph capture records the entire compression pass as a replayable graph,
eliminating CPU kernel-launch overhead on repeated calls with the same pipeline.

```cpp
fz::Pipeline pipeline(input_bytes, fz::MemoryStrategy::PREALLOCATE);
// ... addStage, connect ...
pipeline.enableGraphMode(true);
pipeline.finalize();

pipeline.warmup(stream);         // JIT-compile kernels
pipeline.captureGraph(stream);   // record once

// Subsequent compress() calls replay the graph
pipeline.compress(d_input, input_bytes, &d_compressed, &compressed_sz, stream);
```

Call `compress()` only after `captureGraph()`; use the same stream for capture and replay.

Requirements: `PREALLOCATE` strategy, non-zero input size at construction, all stages
graph-compatible, single-source pipeline. Incompatible with the caller-owned `compress()` overload.

---

## Diagnostics

| Call | Purpose |
|------|---------|
| `pipeline.printPipeline()` | Print stage graph, buffer assignments, and execution levels |
| `pipeline.enableProfiling(true)` | Enable per-stage GPU timing |
| `pipeline.getLastPerfResult()` | Per-stage timing from the last compress/decompress |
| `pipeline.getPeakMemoryUsage()` | Peak pool bytes from the last run |
| `pipeline.getCurrentMemoryUsage()` | Live pool bytes right now |
| `pipeline.isMemPoolFallbackMode()` | True if the CUDA pool fell back to `cudaMalloc` (e.g. vGPU) |
| `pipeline.reset(stream)` | Free non-persistent buffers and reset state for re-use |

---

## Common Gotchas

- **Lorenzo downstream port** — connect to `"codes"`, not the default `"output"`.
- **setDims() before addStage()** — dimensions are pushed into the stage at add-time.
- **Pool-owned pointers** — never `cudaFree` compress output or default decompress output.
- **finalize() divides the world** — no configuration changes after `finalize()`.

---

## API Stability and Versioning {#api_stability}

### Public API boundary

The stable public API (source-compatible across 1.x) is everything reachable from `include/fzgpumodules.h`:
`Pipeline`, `Stage`, `StageFactory`, FZM file structs, `MemoryPool` public interface, and the `MemoryStrategy` / `ErrorBoundMode` enums.

Anything under `src/`, kernel implementations in `modules/*.cu`, allocation heuristics, pool sizing, buffer-coloring details, and logging output text may change in any release without being treated as breaking.

### Versioning policy (SemVer)

A **major** version bump is required when:
- Removing or renaming a public class, method, enum value, or field
- Changing ownership or lifetime rules for a returned pointer
- Changing a public function's signature
- Adding, removing, or retyping a `Stage` virtual method in a way that breaks custom stages
- Breaking FZM file compatibility beyond version-negotiation rules

**Minor** bumps cover backward-compatible additions (new methods, overloads, optional fields with safe defaults).
**Patch** bumps cover bug fixes, documentation fixes, and non-behavioral cleanup.

No ABI compatibility guarantee is made across any release — recompile downstream code against the library version in use.

### Stage interface stability

`StageType` enum values are serialized in `.fzm` files — existing values must never be renumbered or reused, even after a stage is removed.
Adding, removing, or changing any `Stage` virtual method signature is a breaking change and requires a major-version bump.

### API change checklist

Use when opening a PR that touches a public header:

- [ ] Does this change a public header in `include/`?
- [ ] Does it alter pointer ownership, lifetime, or free semantics?
- [ ] Does it affect `Stage` virtual method signatures or behavioral contracts?
- [ ] Does it change FZM file format compatibility or version-negotiation behavior?
- [ ] Are docs and tests updated to match the new behavior?

If any answer is "yes" and the change is not backward-compatible, schedule it as a major-version bump.
