# Performance Tuning {#performance_tuning}

FZGPUModules exposes performance levers at three levels: pipeline-level API options
(memory strategy, CUDA Graph capture, stream count), per-stage TOML/setter
configuration, and build-time flags. This page catalogs every lever found in the
codebase, what it trades off, and the measured effect size where one exists. Where
no controlled measurement exists yet, the table says so explicitly rather than
guessing — treat those rows as open questions, not settled defaults.

---

## Quick Reference

| Lever | Level | Effect (measured) |
|---|---|---|
| `AdaptiveBitpack`/`TiledLorenzo` warp-cooperative kernels | built-in (always on for `block_size` 32/64) | cuszp2 decompress **1.47x**, cuszp3 compress **1.31x** — see below |
| `MemoryStrategy::PREALLOCATE` vs `MINIMAL` | pipeline | Not yet quantified — see [Memory Strategy](#pt_memory_strategy) |
| `enableGraphMode` + `captureGraph` | pipeline | ~3% on one measured pipeline — see [CUDA Graph Mode](#pt_graph_mode) (caveat below) |
| `AdaptiveBitpack` block size (32/64 vs. other) | stage | Selects the warp-cooperative fast path — see [AdaptiveBitpack](#pt_adaptive_bitpack) |
| Quantizer `dither` | stage | ~25% higher outlier rate, tunable via `dither_strength` |
| `GInterp` `auto_tuning` mode | stage | +1 ms to +20 ms per call depending on mode — see [GInterp](#pt_ginterp) |
| `CMAKE_CUDA_ARCHITECTURES` | build | Defaults to `86` — mismatches newer GPUs (see footgun below) |
| `USE_SANITIZER` | build | 10-100x slower (intentional, dev-only) |

---

## Pipeline-Level Levers

### Memory Strategy {#pt_memory_strategy}

| Strategy | Behavior | Effect (measured) |
|---|---|---|
| `MemoryStrategy::MINIMAL` | Allocate on demand, free at last consumer | Lowest peak memory. Not yet quantified against PREALLOCATE. |
| `MemoryStrategy::PREALLOCATE` | Allocate everything at `finalize()`, enables buffer coloring | Required for CUDA Graph capture (`MINIMAL` throws — see `CLAUDE.md`). Not yet quantified. |

`pool_multiplier` (third `Pipeline` constructor argument, default `3.0f`) sizes the
memory pool as `input_size × multiplier`. Too low risks a mid-run pool growth
reallocation; too high wastes memory. No sweep has been run to characterize the
tradeoff curve — treat the default as a starting point, not a verified optimum.

### CUDA Graph Mode {#pt_graph_mode}

```cpp
pipeline.enableGraphMode(true);   // before finalize()
pipeline.finalize();
pipeline.warmup(stream);          // JIT-compiles all kernels once
pipeline.captureGraph(stream);
```

Eliminates CPU-side kernel launch/dispatch overhead on repeated `compress()` calls
by replaying a recorded CUDA graph. Requires `MemoryStrategy::PREALLOCATE`, a
non-default stream, and every stage in the pipeline to be graph-compatible
(`isGraphCompatible()`) — `HuffmanStage` and `ANSStage` are not (see below), and
`AdaptiveBitpackStage`'s inverse path is not (it reads its header size back from
device to host before allocating the output).

**Caveat on the effect size:** an early measurement on the cuszp2 pipeline (CLDHGH
dataset) found CUDA Graph mode worth only ~3% over ungraphed PREALLOCATE — small
enough that the dominant cost there is elsewhere (extra HBM round-trips between
DAG stages, not per-call launch overhead). That figure comes from a separate
benchmarking harness, not from a controlled measurement in this repository — treat
it as a plausible order of magnitude, not a verified constant, and expect the win
to be larger for pipelines with many small/cheap stages (where launch overhead is
proportionally bigger) and smaller for pipelines dominated by one or two expensive
kernels.

### Stream Count

`pipeline.setNumStreams(int)` (default 4, `include/pipeline/compressor.h`) sets how
many concurrent CUDA streams the DAG scheduler uses for stages at the same
dependency level. No guidance yet on tuning this against DAG width — the default
of 4 has not been swept against 1, 2, or 8.

### `decompressInto()` for Double-Buffered Decode Loops

The caller-allocated decompress overload (`compressor.h:306`) skips the internal
`cudaMalloc`/D2D copy/`cudaStreamSynchronize` that the pool-owned decompress path
does, allowing cross-stream overlap in a double-buffered decode loop. Caveat:
`RZEStage`/`RREStage`/`HuffmanStage`/`AdaptiveBitpackStage`'s inverse paths still
block on internal D2H reads regardless of which decompress overload is used — this
lever removes framework-level overhead, not stage-internal syncs.

### `prepareInverse()` for Decode-Only Pipelines

`compressor.h:339` — caches the inverse DAG across calls and builds output-buffer
metadata directly from a known uncompressed size, avoiding a throwaway warmup
`compress()` call when a pipeline is only ever used for decoding (e.g. a reader
that never compresses).

### Diagnostics Toggles

| Setter | Effect |
|---|---|
| `enableProfiling(bool)` | Per-stage CUDA event timing when on. Header comment states zero overhead when disabled — not independently re-measured in this repo. |
| `enableBoundsCheck(bool)` | Runtime buffer-overwrite check; adds per-stage overhead. Always on in debug builds. |
| `setColoringEnabled(bool)` | Buffer coloring (PREALLOCATE-only) aliases non-overlapping buffer lifetimes to reduce footprint; disabling trades memory for easier `compute-sanitizer` debugging. |

### vGPU / `FZ_FORCE_MEMPOOL_FALLBACK`

On a vGPU, or with `FZ_FORCE_MEMPOOL_FALLBACK` set in the environment, CUDA's
stream-ordered mempool allocator is unavailable and the pipeline falls back to
`cudaMalloc`+sync — this works correctly but loses the stream-ordered allocator's
performance and disables CUDA Graph capture entirely (`mainpage.md:30`).

---

## Per-Stage TOML Knobs

### Predictor block/tile size — `Lorenzo`, `TiledLorenzo` {#pt_predictor_block}

`Lorenzo::setBlockSize` and `TiledLorenzo::setTileShape` set the block-local reset
period / tile shape a downstream `AdaptiveBitpackStage` packs. **As of the
warp-cooperative kernel rewrite below, this choice now also determines whether
`AdaptiveBitpackStage` takes its fast path**: `block_size` (`tile_x*tile_y*tile_z`
for `TiledLorenzo`) must be exactly 32 or 64 to hit the warp-cooperative kernels;
any other value — including other multiples of 32 — silently falls back to the
original (slower) scalar kernels. The two shipped presets already use these values
(`cuszp2.toml`: Lorenzo block=32; `cuszp3*.toml`: TiledLorenzo 8x8/4x4x4 = 64), so
this only matters if you set a custom tile shape or block size.

### Quantizer: dither, radius, outlier capacity

| Setting | Effect (measured) |
|---|---|
| `dither = true` | Decorrelates reconstruction error from the signal at no storage cost, but raises the outlier-escalation rate to **~25%** on smooth data (full-bin-width dithering forces roughly a quarter of elements to violate the error bound and fall back to lossless scatter storage) — size `outlier_capacity` accordingly. |
| `dither_strength` (`0,1]`, default `1.0`) | Scales the dither offset as a fraction of the bin width, trading decorrelation strength for a lower outlier rate; `0.0` is bit-identical to `dither=false`. |
| `linear_mode` / `inplace_outliers` | Incompatible with dithering — neither has a per-element outlier-escalation path. |

### Bitshuffle: block size / element width

`setBlockSize`/`setElementWidth` control the bit-matrix transpose granularity.
Purely a config-surface description in the docs today — no throughput sweep exists.

### RZE / RRE: chunk size, word size

`setChunkSize` accepts only `4096`, `8192`, or `16384` bytes (default `16384`) —
restricted to this set because each CUDA block holds one chunk in shared memory,
and larger values would blow the shared-memory budget. `setWordSize` (1/2/4/8)
selects the LC `RZE_1`/`RZE_2`/`RZE_4`/`RZE_8` variant matching the upstream data
width. No controlled throughput comparison across chunk sizes exists yet — this is
a real documentation gap, not a "no effect" finding.

### AdaptiveBitpack: block size, outlier selection {#pt_adaptive_bitpack}

- **`block_size` 32 or 64 selects the warp-cooperative kernel path** (see the
  Changed section of `CHANGELOG.md` and the built-in optimization below); any
  other value uses the original scalar kernels, which are still correct but do not
  benefit from the coalescing fix.
- `outlier_selection = true` (cuSZp2/cuSZp3's per-block outlier mode): measured
  compression-ratio comparison on CLDHGH at abs eb=1e-3 gives **8.49x here vs.
  9.09x for reference cuSZp2** (plain mode matches the reference at 3.88x) — the
  gap is metadata-byte overhead from the 2-byte-per-block selection tag, not a
  correctness or throughput issue.

### GInterp: auto-tuning mode {#pt_ginterp}

`auto_tuning` (0-5) is the richest quantitative lever in the library:

| Mode | Cost per call | Effect |
|---|---|---|
| `0` (off) | none | Manual alpha/beta, no probing |
| `1` | ~1 ms | Cheap 2-error profiling pass, 3-D only |
| `2` | ~1 ms | Alternate cheap pass, 2-D + 3-D |
| `3`–`4` | ~5-15 ms, +10-20 ms total for the full sweep | Deeper probing; mode 4's full alpha/beta sweep typically improves CR by ~2-5% |
| `5` (manual) | none | Caller supplies alpha/beta directly |

Error bound is typically `≤1.1×eb` on smooth data, up to ~2×eb in adversarial cases
— see `docs/stages/ginterp.md` for the full breakdown.

### Huffman: `bklen`

Not CUDA Graph compatible — two device-to-host synchronizations per forward call
(histogram D2H for codebook construction, partition-metadata D2H for the
prefix-sum) are serial barriers that dominate over kernel execution time. The
stage is **latency-bound, not throughput-bound**, and "performs poorly on very
small inputs (< ~100 KB)" per its own documentation — batch small inputs together
if you need Huffman coding on them.

### ANS: `prob_bits`

Only `prob_bits=10` is supported in this build (the dietGPU kernels are compiled
as explicit template instantiations for `kANSDefaultProbBits=10`) — calling
`setProbBits(n)` with any other value throws. Like Huffman, ANS has one
device-to-host synchronization point per call and is not CUDA Graph compatible.

---

## Build-Time Levers

### `CMAKE_BUILD_TYPE`

`Release` (`-O3`) vs `Debug` (`-O0`, `-Wall`) — use `Release` for anything
performance-sensitive; `Debug` only for development.

### `CMAKE_CUDA_ARCHITECTURES` — a real footgun

**Defaults to `"86"`** (`CMakeLists.txt:45`) if not explicitly set. On any other
GPU generation — including the H100 (`sm_90`) this session's measurements were
taken on — this silently builds PTX for the wrong architecture, either failing to
run or falling back to JIT compilation at first launch (extra startup latency,
and potentially missing architecture-specific instruction selection). Always pass
`-DCMAKE_CUDA_ARCHITECTURES=<your arch>` explicitly. This is the same class of
"silently wrong default" issue as the NVTX3/`CUDAToolkit_ROOT_DIR` CMake bug fixed
in `profiling/CMakeLists.txt` this session (see `CHANGELOG.md`) — CMake's
find-module defaults are not always what you'd expect on a given machine, so it's
worth double-checking `cmake -B <dir> ... 2>&1 | grep -i "CUDA architectures"`
after any fresh configure.

### `FZ_LOG_MIN_LEVEL`

Compiles out `FZ_LOG()` call sites below the threshold entirely — arguments are
never evaluated, so this is genuinely zero-overhead for suppressed levels, not
just a runtime filter. Default `2` (INFO); set to `255` (SILENT) to strip all
logging for production/embedded builds where every cycle counts.

### `BUILD_PROFILING` / `FZ_PROFILING_ENABLED`

Gates NVTX3 annotations (used in `decompressFromFile` and the `profiling/`
harnesses) and `-fno-omit-frame-pointer` for complete `nsys` CPU stack walks. When
`BUILD_PROFILING=OFF`, there is zero instrumentation cost (the code doesn't exist
in the binary). When `ON` but not actively captured under `nsys`, the per-call
NVTX range-push/pop cost is not quantified in this repo — a real gap if you're
deciding whether to ship a profiling-enabled build to production.

### `USE_SANITIZER`

`ASanUbsan`, `TSan`, or `Compute` — all are development-only. Tests run
**10-100x slower** under Compute Sanitizer (`docs/building.md`); use
`--gtest_filter` to scope down when iterating. `COMPUTE_SANITIZER_DEVICE_DEBUG`
adds `-G` for source-level correlation, which is much slower still.

---

## Built-in Kernel Optimizations

These aren't user-facing levers — they're always active — but they materially
change what throughput to expect from the affected stages, so they belong here.

### `AdaptiveBitpackStage`: warp-cooperative encode/decode

Every encode/decode kernel previously assigned **one CUDA thread to an entire
data block**, with a strided, non-coalesced access pattern confirmed via `ncu` at
32.0 sectors-per-request on global loads (32 separate cache lines touched per warp
— the literal worst case). Redesigned so one warp's 32 lanes cooperate per block
via `__ballot_sync`; the on-disk archive format is bit-for-bit unchanged. The fast
path applies when `block_size` is 32 or 64 (every shipped preset); other values
fall back to the original, unchanged scalar kernels.

Measured on CLDHGH (3600×1800, H100):

| Pipeline | Compress | Decompress |
|---|---|---|
| cuszp2 | 103.4 → 110.2 GB/s (1.07x) | 84.0 → 123.2 GB/s (**1.47x**) |
| cuszp3 | 144.4 → 189.6 GB/s (**1.31x**) | 94.0 → 121.2 GB/s (1.29x) |

### `TiledLorenzoStage`: phased inverse scan

The inverse kernel previously ran a single, fully-serial 64-long dependency chain
on one thread per tile. Redesigned as one CUDA block per tile, decomposing the
chain into three parallel phases (Z-chain → per-row Y-chains → per-cell X-chains).
Applies to both 2-D (8x8) and 3-D (4x4x4) tile shapes and any custom
`setTileShape()`; no fallback needed since `tile_elems` is already capped at 1024
by the existing tile-shape validation (CUDA's own max threads/block). Measured as
part of the cuszp3 numbers above (`TiledLorenzoStage` is cuszp3-only; cuszp2 uses
plain `LorenzoStage`).

Both changes are verified via the expanded `AdaptiveBitpackStage`/`TiledLorenzoStage`
unit test suites and `compute-sanitizer --tool memcheck`/`racecheck`/`synccheck`
(0 errors/hazards) — see `CHANGELOG.md` for the full verification record.
