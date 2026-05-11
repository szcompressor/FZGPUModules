# Changelog

All notable changes to FZGPUModules are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Version numbers follow [Semantic Versioning](https://semver.org/).

---

## [Unreleased] — 2.0.0

### Changed
- Extracted duplicated `align16`, `buildLevelTimings`, and concat buffer layout arithmetic into `src/pipeline/pipeline_utils.h`; introduced `ConcatLayout` struct with `headerSize`/`slotSize` helpers replacing open-coded offset calculations in `compressor.cpp` and `compressor_exec.cpp`
- Added `PoolBuffer`, `PinnedBuffer`, and `DeviceBuffer` RAII wrappers (private nested structs in `Pipeline`) replacing raw pointer+capacity member pairs; destructor simplified to graph handle teardown only
- Decomposed `finalize()` into six focused sub-methods: `typeCheckConnections`, `computeInputAlignment`, `refinePoolSize`, `setupGraphModeInput`, `preallocatePadBuffer`, `preallocateConcatBuffers`
- Extracted `prepareInputSource()` from `compress()` (graph-mode copy and alignment padding logic) and `buildOrReuseInvCache()` from `decompress()` (inverse DAG cache build/reuse)
- Extracted `computeFilePoolSize`, `reconstructForwardTopology`, and `buildSourceSizesFromHeader` as private static members from `decompressFromFile()`
- Added Doxygen class-level descriptions to `Logger`, `Zigzag<T>`, and `Negabinary<T>` which were previously undocumented
- Added "no template parameters" note and common instantiation snippet to `BitshuffleStage` and `RZEStage` stage docs
- Expanded requirements table in README and docs mainpage to include host compiler guidance (GCC 7+ / Clang 5+, upper bound set by CUDA version; NVHPC 23.11 tested in CI)
- Migrated `.github/ISSUE_TEMPLATE.md` to `.github/ISSUE_TEMPLATE/bug_report.yml` (modern GitHub Forms format)
- Reorganized `modules/` into six semantic categories: `predictors/` (Lorenzo, Diff/delta, interpolation), `transforms/` (zigzag, negabinary), `quantizers/` (quantizer), `coders/` (RLE, RZE, bitpack), `shufflers/` (bitshuffle), `fused/` (lorenzo_quant); all include paths and CMake source lists updated accordingly — this is a **breaking change** for any code that includes stage headers directly (e.g. `"encoders/diff/diff.h"` → `"predictors/diff/diff.h"`)
- Merged `fzgmod_encoders` and `fzgmod_predictors` CMake targets into a single `fzgmod_modules` target; downstream CMakeLists linking either old target must switch to `fzgmod_modules`

### Added

**Stages**
- `BitpackStage`: added `setAutoDetect(bool)` — when enabled, forward execute scans the input for its maximum value via CUB `DeviceReduce::Max` and selects the tightest valid power-of-two `nbits` automatically; sets `isGraphCompatible()` to `false` while active

**Tests**
- Added 9 auto-detect test cases to `test_bitpack.cpp` covering graph incompatibility, worst-case size estimate, nbits selection for uint16_t and uint32_t, all-zero input, full-width fallback, and pipeline integration
- Unified stage test suite: standardized all 12 stage test files with file-level docstrings listing every test by short ID (ZZ, NB, ZS, NS, RL, DD, LZ, BP, TM, RZ, QZ, QD), full-width section dividers, and ID-prefixed headers before each `TEST`/`TYPED_TEST` block
- Added `RLEStage/HeaderSerialization` (RL7) to `test_rle.cpp` — RLE was the only stage without a `serializeHeader`/`deserializeHeader` round-trip test
- Replaced inline sine/cosine data generators in `test_bitshuffle_stage.cpp` (BS16 PipelineIntegration), `test_rze_stage.cpp` (RZ19 PipelineIntegration), and `test_quantizer.cpp` (QD1, QD2) with shared `make_smooth_data<T>()` helper
- Corrected `test_rze_stage.cpp` docstring: file has 20 tests (RZ1–RZ20), not 18; added RZ19 (`PipelineIntegration`) and RZ20 (`PipelineCompressionRatio`) entries

**CI**
- Added `docker-publish.yml` — builds and pushes the FZGPUModules image to GHCR (`ghcr.io/szcompressor/fzgpumodules`) on every push to main
- Updated `build-check.yml` to use the GHCR image instead of the upstream NVIDIA base, removing the inline `apt-get install` step

**Docker**
- Added Dockerfile with FZGPUModules pre-built and installed to `/usr/local` (headers, libs, CMake package config); supports local dev, CI/CD, and distribution from a single image
- Added `.dockerignore` to exclude build artifacts, git history, and test dependencies from the build context
- Added `docs/docker.md` covering image build, pre-installed library usage (`find_package` + `nvcc`), local source development, CI/CD patterns, and troubleshooting

**Documentation**
- Added `docs/stages/` — per-stage Doxygen pages covering constraints, behavioral rules, mode details, and usage examples for all eight stages (`LorenzoQuantStage`, `LorenzoStage`, `QuantizerStage`, `DifferenceStage`, `BitshuffleStage`, `RZEStage`, `RLEStage`, `BitpackStage`)
- Updated `Doxyfile` to include `docs/stages/` in `INPUT`; added `\ref stages_overview` link from the mainpage

**Stages**
- `BitpackStage` — sub-byte and multi-byte integer bit-packing; supports `uint8`/`uint16`/`uint32`, all power-of-two `nbits` values; graph-compatible; registered in `StageFactory`; 14 tests in `test_bitpack.cpp`
- `LorenzoStage<T>` — plain integer delta predictor (lossless, non-fused); accepts `int8_t`/`int16_t`/`int32_t`/`int64_t`; supports 1-D/2-D/3-D via `setDims()`; distinct from the fused `LorenzoQuantStage`
- Renamed `LorenzoStage` → `LorenzoQuantStage` to make the fused quantizer/predictor nature explicit; all callsites, headers, TOML configs, and tests updated

**Build & configuration**
- `examples/presets/quantizer_lorenzo_bitpack.toml` — new TOML preset for `QuantizerStage → LorenzoStage → BitpackStage` (cuSZp-style) pipeline
- `examples/presets/lorenzo_bitpack.toml` — new TOML preset for `LorenzoQuantStage → BitpackStage` pipeline

**Pipeline features**
- `Pipeline::getLastUncompressedSize()` — returns the original input byte count from the most recent `compress()` call (0 before first call); useful for sizing a decompression output buffer without out-of-band metadata; persists across `reset()`
- Multi-source pipeline support: `InputSpec` API, `compress(std::vector<InputSpec>)`, `decompressMulti()`, `setInputSizeHint()` per source
- `Pipeline::warmup(stream)` — forces PTX→SASS JIT compilation before timing-sensitive work
- `Pipeline::enableBoundsCheck(bool)` — runtime toggle for buffer overwrite detection (always on in Debug builds)
- `Pipeline::setCaptureMode(bool)` — CUDA Graph stream capture for steady-state compression
- `Pipeline::setPoolManagedDecompOutput(bool)` — opt-in pool-owned decompression output (avoids D2D copy)
- Cached inverse DAG: `buildInverseDAG()` result cached after first `decompress()` call; eliminates ~200–500 µs per-call DAG rebuild overhead
- Logging system: `FZ_LOG(LEVEL, ...)` with compile-time gating; `FZ_LOG_MIN_LEVEL` CMake option (0=TRACE … 255=SILENT); `Logger::setLogCallback()` for custom sinks

**Memory & DAG**
- `MemoryPool` cudaMalloc fallback: when `cudaMemPoolCreate()` fails (e.g. vGPU), the pool transparently falls back to `cudaMalloc`/`cudaFree` with stream synchronization; same for `MemoryPoolConfig::force_fallback` or the `FZ_FORCE_MEMPOOL_FALLBACK` env var — allows running the full test suite in fallback mode on any GPU
- `Pipeline::isMemPoolFallbackMode()` — query whether the internal pool is running in fallback mode
- `MemoryPool::isFallbackMode()` — low-level query on the pool handle directly
- `test_mempool_fallback.cpp` — 11 tests covering `isMemPoolFallbackMode()` detection, MINIMAL/PREALLOCATE round-trips, Lorenzo→RLE, Lorenzo→Bitshuffle→RZE (exercises stage-level scratch), usage tracking, no-leak across 5 compress+reset cycles, RLE scratch reuse, and file IO; all in forced fallback mode
- Buffer coloring: non-overlapping buffers in PREALLOCATE mode are aliased to reduce peak GPU memory
- Pinned concat header buffer: reduces H2D API calls from `1+N` to 1 per `compress()` call
- Custom gather kernel (`launch_gather_kernel`) for D2D segment copies: replaces N individual `cudaMemcpyAsync` calls with a single kernel dispatch
- `getActualOutputSize(int index)` index-based accessor on `Stage` — eliminates per-call `unordered_map` allocations in the inner execute loop
- Pool auto-sizing: `computeTopoPoolSize()` + `setReleaseThreshold()` for topology-aware pool configuration
- `setExternalPointer()` zero-copy path: user-owned device buffer passed directly into DAG

**File format**
- FZM version bumped to 3 (`FZM_VERSION = 0x0300`); `FZMHeaderCore` extended to 80 bytes with `num_sources` and `source_uncompressed_sizes[4]` fields
- CRC32 (IEEE 802.3) checksums on payload and header
- `Pipeline::writeToFile()` / `Pipeline::decompressFromFile()` static utility

**Stages**
- `BitshuffleStage` — GPU bit-matrix transpose
- `RZEStage` — recursive zero-byte elimination with optimized kernel
- `ZigzagStage<TIn, TOut>` — zigzag encode/decode
- `NegabinaryStage<TIn, TOut>` — negabinary encode/decode
- `DifferenceStage<T, TOut>` — first-order difference / cumulative-sum coding
- `QuantizerStage<TInput, TCode>` — direct-value quantizer with ABS/REL/NOA error modes
- Multi-dimensional Lorenzo (2-D and 3-D predictor kernels)

**Build & distribution**
- `find_package(FZGPUModules REQUIRED)` support via `cmake/FZGPUModulesConfig.cmake.in`
- Versioned shared library symlinks (`libfzgmod.so → .so.2 → .so.2.0.0`) via `VERSION`/`SOVERSION` target properties
- Relocatable RPATH (`$ORIGIN/../lib` on Linux, `@loader_path/../lib` on macOS)
- CUDA 11.2+ version floor check at CMake configure time
- Little-endian host check at CMake configure time
- `profiling/` directory for profiling programs (separated from `examples/`)
- `scripts/new_stage.sh` scaffold script for adding new stages
- CMakePresets with `asan` and `compute-san` presets for sanitizer builds
- Doxygen CI GitHub Actions workflow publishing to GitHub Pages

**Testing**
- Comprehensive test suite: 20 test binaries covering pipeline, stages, file I/O, memory strategies, buffer coloring, CUDA Graphs, bounds checking, and error handling
- All tests pass under CUDA Compute Sanitizer (memcheck, initcheck, racecheck, synccheck) and host ASan+UBSan
- `LorenzoQuantStage.DeterministicReconstruction` — verifies the fused kernel produces element-wise identical output across two independently constructed pipelines on the same input

### Removed
- `LorenzoStage` (the old fused predictor+quantizer) removed and replaced by `LorenzoQuantStage`; `LorenzoStage` now refers exclusively to the plain integer delta predictor

### Changed
- `BitpackStage` auto-detect: scratch buffers (`d_max`, CUB temp) now allocated through the pipeline `MemoryPool` (with transparent `cudaMalloc` fallback when the pool returns null) so all device memory remains tracked by the pipeline; prior implementation allocated directly via `cudaMalloc` outside the pool
- Refactored all pipeline test files to remove duplicate local data-generator functions (`make_smooth`, `make_smooth_data`, `make_test_data`) in favor of shared `make_smooth_data<T>()` from `fz_test_utils.h`; replaced manual compress/decompress boilerplate in `test_data_patterns.cpp`, `test_memory_strategies.cpp`, and `test_mempool_fallback.cpp` with `pipeline_round_trip<T>()` from `stage_harness.h`; added P15 (`Lorenzo2DRoundTrip`) and P16 (`Lorenzo3DRoundTrip`) tests to `test_pipeline.cpp`; unified comment structure (file-level docstrings with test IDs, section divider lines) across all pipeline test files

### Fixed
- ASan: avoid a use-after-free in `CompressionDAG::addStage` by taking the stage name by value
- CI: make CUDA module loading optional when lmod/module are unavailable so non-Jetstream runners do not fail early
- vGPU compatibility: added fallback from `cudaMallocAsync`/`cudaFreeAsync` to `cudaMalloc`/`cudaFree` when memory pools are unavailable; `MemoryPool` gracefully degrades to regular malloc mode with warning log; fixes "operation not supported" errors on virtualized GPUs (e.g., Jetstream NVIDIA Virtual Compute Server)
- vGPU stream synchronization: fallback code paths in `MemoryPool`, `DifferenceStage`, `RLEStage`, and `RZEStage` now synchronize streams before calling `cudaFree()` to prevent use-after-free race conditions when kernels are still using freed memory
- Race condition in `CompressionDAG::execute()` for multi-source pipelines: internal per-branch streams now have a GPU-side happens-before edge into the caller stream via `cudaStreamWaitEvent`
- `DifferenceStage` inverse: replaced `cub::DeviceScan` with a custom `cumsumChunkedKernel` that uses only shared memory (no device temp allocation, sanitizer-clean)
- `decompressFromFile` cleanup frees use `stream=0` to avoid pool-destructor race with `cudaMemPoolDestroy`
- Removed spurious `find_dependency(CCCL REQUIRED)` from installed CMake config file (would break downstream `find_package` for users without CCCL)
- Removed C language from `project()` (only CUDA and CXX required)

### Changed
- Project version set to `2.0.0` in `CMakeLists.txt`
- `buildInverseDAG()` return type changed from `{inv_dag, int}` to `{inv_dag, unordered_map<Stage*, int>}` for multi-source support (breaking for internal callers)
- `Stage::saveState()` / `restoreState()` contract: any stage that modifies `actual_output_sizes_` in its inverse `execute()` must override both methods

---

## [1.0.0] — 2026-04-14

Initial tagged release. cuSZ based compressor with modular design and experimental CUDASTF support.

[Unreleased]: https://github.com/szcompressor/FZGPUModules/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/szcompressor/FZGPUModules/releases/tag/release
