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
Defined in `include/advanced/dag.h`.

| Value | Behavior |
|-------|----------|
| `MINIMAL` | Allocate buffers on demand; free each one as soon as its last consumer reads it. Lowest peak GPU memory. |
| `PREALLOCATE` | Allocate all buffers during `finalize()`. Required for CUDA Graph capture. Enables buffer coloring for lower memory footprint. |

### fz::ErrorBoundMode
Defined in `modules/fused/lorenzo_quant/lorenzo_quant.h`. Used by `LorenzoQuantStage`, `QuantizerStage`, and `GInterpStage`.

| Value | Meaning | Notes |
|-------|---------|-------|
| `ABS` | Absolute error — `abs(x_orig - x_recon) ≤ eb` | Useful when data is homogenous in magnitude (preserve big picture) |
| `REL` | **Guaranteed** point-wise relative — `abs(error) / abs(x_orig) ≤ eb` for **every** element | Implemented **only** by `QuantizerStage`, via log-space quantization. `LorenzoQuantStage` / `GInterpStage` accept it as a deprecated alias for `PREL` and warn. |
| `NOA` | Value-range relative — `abs(error) / value_range ≤ eb` (norm-of-absolute) | Useful for single bounds over multiple datasets |
| `PREL` | **Pseudo**-relative — `abs_eb = eb × max(abs(data))`, then applied as a plain `ABS` bound | Bounds `error / max(abs(x))`, **not** `error / abs(x)`. The cheap approximation used by the predictor-fused stages, which cannot vary the bound per element. |

**REL vs PREL — the distinction that matters.** `PREL` is only as tight as
`REL` for elements at the peak magnitude. An element at 1% of peak sees an
effective relative error 100× looser than the `eb` you asked for, and elements
near zero are unbounded in relative terms. If you need a per-element relative
guarantee, `QuantizerStage` with `REL` is the only stage that provides it.
`examples/eb_mode_analysis.cpp` measures this on your own data.

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

## Compression and Decompression

New code should prefer the span-based \ref explicit_ownership_api "Explicit-ownership API"
below, which states ownership in the return type. The pointer overloads documented
here remain fully supported; ownership for each is summarized in the
\ref memory_ownership_summary "Memory Ownership Summary".

The default (pool-owned) path is the simplest — the pipeline holds the buffer, so
you never `cudaFree` it:

```cpp
void*  d_compressed  = nullptr;
size_t compressed_sz = 0;
pipeline.compress(d_input, input_bytes, &d_compressed, &compressed_sz, stream);
// valid until the next compress()/reset() or Pipeline destruction

void*  d_output  = nullptr;
size_t output_sz = 0;
pipeline.decompress(d_compressed, compressed_sz, &d_output, &output_sz, stream);
// valid until the next decompress() or Pipeline destruction
```

The other overloads write into a caller-provided buffer (or hand back a caller-owned
allocation). Each throws if a supplied buffer is too small, reporting the size needed:

| Overload | Ownership | Notes |
|---|---|---|
| `compress(in, n, d_buf, capacity, &actual, stream)` | caller's buffer | size with `getMaxCompressedSize(n)` |
| `decompress(in, n, &d_out, &sz, stream)` after `setPoolManagedDecompOutput(false)` | caller-owned (`cudaFree`) | fresh allocation each call |
| `decompress(in, n, d_buf, capacity, &actual, stream)` | caller's buffer | synchronous; no temp alloc/copy/free |
| `decompressInto(in, n, d_buf, capacity, &actual, stream)` | caller's buffer | fully async for overlapped decode — see [Performance Tuning](\ref performance_tuning) |

`decompressInto()` requires `PREALLOCATE`, leaves `*actual` as the *planned* size, and
does **not** synchronize — the bytes are valid only after you synchronize the stream
yourself. Use one Pipeline per concurrent stream. (Some inverse coders — RZE/RRE/Huffman/
AdaptiveBitpack — still do a blocking device→host header read inside `execute()`; for full
overlap of those, drive each slot from its own host thread.)

**Sizing helpers:** `getMaxCompressedSize(input_bytes)` (tight upper bound for a compress
buffer) and `getLastUncompressedSize()` (original size from the most recent `compress()`).

---

## Explicit-ownership API (preferred in new code) {#explicit_ownership_api}

The pointer overloads above encode ownership in *how* you call them (`void**`
vs `void*`) and in mutable pipeline state (`setPoolManagedDecompOutput()`). The
span-based API states it in the type instead, and is a thin wrapper over the
same execution core — identical behavior, no performance difference.

| Call | Returns | Ownership |
|---|---|---|
| `compress(ConstDeviceSpan, stream)` | `BorrowedDeviceBuffer` | Pool-owned; do not free |
| `compressInto(ConstDeviceSpan, DeviceSpan, stream)` | `size_t` bytes written | Caller's buffer |
| `decompressBorrowed(ConstDeviceSpan, stream)` | `BorrowedDeviceBuffer` | Pool-owned; do not free |
| `decompressOwned(ConstDeviceSpan, stream)` | `OwnedDeviceBuffer` | Caller-owned; freed on destruction |
| `decompressInto(ConstDeviceSpan, DeviceSpan, stream)` | `size_t` bytes written | Caller's buffer (synchronous) |
| `decompressIntoAsync(ConstDeviceSpan, DeviceSpan, stream)` | `size_t` planned bytes | Caller's buffer (async; PREALLOCATE only) |

```cpp
fz::BorrowedDeviceBuffer comp = pipeline.compress({d_in, in_bytes}, stream);
// comp is pool memory: no cudaFree, invalidated by the next compress()/reset()

fz::OwnedDeviceBuffer out = pipeline.decompressOwned(comp.cspan(), stream);
// out.data() / out.bytes(); freed when `out` goes out of scope
```

`decompressBorrowed()` and `decompressOwned()` ignore
`setPoolManagedDecompOutput()` — the call site decides, and the flag is left
exactly as the caller set it. `OwnedDeviceBuffer` is move-only and records the
device it was allocated on, so it frees through the backend the library was
built against (never a hard-coded `cudaFree`) and on the right device.
`release()` hands the raw pointer back if you need to manage it yourself.

The types live in `include/pipeline/device_buffer.h` and allocate nothing
themselves; `DeviceSpan` / `ConstDeviceSpan` are plain non-owning views.

---

## Memory Ownership Summary {#memory_ownership_summary}

| Buffer | Owner | Rule |
|--------|-------|------|
| Input (`d_input`) | Caller | Pipeline borrows; never freed by the library |
| Compressed output (pool-owned) | Pipeline | Do **not** `cudaFree` |
| Decompressed output (pool-owned, default) | Pipeline | Do **not** `cudaFree` |
| Decompressed output (caller-owned) | Caller | Must `cudaFree` |
| File decompress (`decompressFromFile` static) | Caller | Must `cudaFree` |
| File decompress (`decompressFromFileInstance`) | Depends on `setPoolManagedDecompOutput()` | Same rules as `decompress()` |
| Memory decompress (`decompressFromMemory`) | Depends on `setPoolManagedDecompOutput()` | Same rules as `decompress()` |
| `BorrowedDeviceBuffer` (from `compress()` / `decompressBorrowed()`) | Pipeline | Do **not** free; invalidated by the next call reusing the slot |
| `OwnedDeviceBuffer` (from `decompressOwned()`) | Caller | Freed on destruction; `release()` to take the raw pointer |

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

## Decode-only pipelines (no warmup compress)

A pipeline that only ever decompresses blobs produced elsewhere — e.g. K streaming
slots reading independently-compressed blocks — does **not** need a throwaway
`compress()` over dummy data to become ready. The in-memory `decompress()` path
otherwise depends on state a forward `compress()` leaves on the instance: the
archive layout *and* the data-dependent inverse metadata that is **not** in the raw
blob (the `HuffmanStage` symbol count, the quantizer outlier count — which changes
block to block). Carry that metadata in a small in-memory header instead:

```cpp
// PRODUCER (after compress()): grab the metadata header, store it with the blob.
std::vector<uint8_t> header = producer.serializeHeaderToMemory();   // ~1 KB, no payload

// CONSUMER: a fresh, finalized pipeline of the SAME topology, never compress()ed.
Pipeline slot(block_bytes);
/* addStage/connect identically */ slot.finalize();

// One call per blob: restores this blob's metadata, decodes it, and reuses the
// slot's cached inverse DAG across calls (no per-blob DAG rebuild).
slot.decompressFromMemory(header.data(), header.size(),
                          d_blob, blob_size, &d_output, &output_sz, stream);
```

- `serializeHeaderToMemory()` requires a prior `compress()`; returns the FZM
  core+stage+buffer header as host bytes (no payload). The header is per-blob
  because the outlier count varies block to block.
- `decompressFromMemory()` fuses `primeInverseFromHeader()` + `decompress()`; output
  ownership follows `setPoolManagedDecompOutput()` exactly as `decompress()` does.
- The two steps are also available separately (`primeInverseFromHeader(header...)`
  then `decompress(blob...)`).
- For fully **self-describing** pipelines (pure lossless chains, linear-mode
  quantizer — no Huffman, no outliers) the lighter `prepareInverse(uncompressed_size)`
  needs no header at all.
- Worked example: `examples/decode_only_slots.cpp`.

---

## CUDA Graph Capture

Records the compression pass as a replayable graph, eliminating CPU kernel-launch
overhead on repeated calls. See [Performance Tuning](\ref performance_tuning) for when
this is worth it.

```cpp
fz::Pipeline pipeline(input_bytes, fz::MemoryStrategy::PREALLOCATE);
// ... addStage, connect ...
pipeline.enableGraphMode(true);
pipeline.finalize();
pipeline.warmup(stream);         // JIT-compile kernels
pipeline.captureGraph(stream);   // record once
pipeline.compress(d_input, input_bytes, &d_compressed, &compressed_sz, stream);  // replays
```

Requirements: `PREALLOCATE`, non-zero input size at construction, a single-source
pipeline of graph-compatible stages, and the same stream for capture and replay.
Incompatible with the caller-owned `compress()` overload.

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

### API tiers

The public headers are organized into three tiers by how stable they are:

| Tier | What | Headers | Guarantee |
|---|---|---|---|
| **Stable** | The API most users need | `fzgpumodules.h`, `pipeline/compressor.h` (`Pipeline`), `pipeline/device_buffer.h`, `pipeline/config.h`, `pipeline/perf.h`, `pipeline/stat.h`, `fzm_format.h`, the `modules/**/*_stage.h` stage headers, the `MemoryStrategy` / `ErrorBoundMode` enums | source-compatible within a major version |
| **Extension** | For custom-stage authors | `stage/stage.h`, `stage/stage_registry.h`, `stage/fusion.h`, `mem/mempool.h` (the `MemoryPool` interface a stage's `execute()` uses) | may grow at minor versions via backward-compatible defaults |
| **Advanced** (`include/advanced/`) | Pipeline internals exposed for experimentation | `advanced/dag.h` (`CompressionDAG` / `DAGNode` / `BufferInfo`), `advanced/fusion_planner.h`, `advanced/fusion_registry.h` | **no** source-compatibility promise; may change any release |

The umbrella `fzgpumodules.h` no longer advertises the Advanced headers. `Pipeline`
still depends on `CompressionDAG` internally, so those types remain reachable
transitively — but reach for them deliberately (`#include "advanced/dag.h"`), knowing
they are unstable. Anything under `src/`, the kernel implementations in `modules/*.cu`,
allocation heuristics, pool sizing, buffer-coloring details, and logging output text may
change in any release without being treated as breaking.

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
