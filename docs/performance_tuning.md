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
