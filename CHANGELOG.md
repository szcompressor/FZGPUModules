# Changelog

All notable changes to FZGPUModules are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Version numbers follow [Semantic Versioning](https://semver.org/).

---

## [Unreleased] — 2.0.0

### Added
- CLI `--stages` now accepts `huffman` (alias `huf`): adds `HuffmanStage<uint16_t>` with `bklen` auto-derived from `2 * quant_radius` when following a predictor, or 1024 otherwise
- `HuffmanStage<T>::setEncodeMode(HuffmanEncodeMode)`: selects between `Coarse` (default, multi-kernel with CPU prefix-sum sync in phase 3) and `Fine` (ReVISIT-lite single kernel with fully GPU-async phase 3 — no mid-encode CPU sync, preferred for latency-sensitive workloads)
- `HuffmanEncodeMode` enum (`Coarse`, `Fine`) in `huffman_stage.h`; `getEncodeMode()` getter
- Fine encode path (`HuffmanEncodeMode::Fine`): replaces CPU prefix-sum with `cub::DeviceScan::ExclusiveSum` (`GPU_encode_scan`), a combined nbit+ncell reduction kernel accumulating in `uint64_t` (`GPU_encode_finalize_totals`), and async D→H copy of the two total scalars to pinned memory; totals are read by the caller after the natural stream sync — no additional synchronization required
- Four new Huffman tests covering the fine encode path: `FineEncode_RoundTrip_U16`, `FineEncode_RoundTrip_U8`, `FineEncode_CompressedSmaller`, `FineEncode_ModeSwitch`
- Five additional fine-path tests matching coarse-path coverage: `FineEncode_RoundTrip_U32`, `FineEncode_ReuseAfterSizeChange`, `FineEncode_OutOfRangeSymbolThrows`, `FineEncode_PipelineIntegration_U16`, `FineEncode_LorenzoQuantPipeline`
- `loadConfig`/`saveConfig`: `encode_mode` TOML key for `HuffmanStage` (`"Coarse"` or `"Fine"`); omitted on save when default (`Coarse`) to keep existing configs minimal
- `hf_hl.cc`: Fine-path max-codelen guard — after `build_book`, scans `h_bk4` for max code length; if `max_codelen > 8` (four symbols would overflow the 32-bit shard accumulator in the ReVISIT-lite kernel), silently falls back to Coarse
- `examples/presets/cusz.toml`: set `encode_mode = "Fine"` on the Huffman stage

### Fixed
- `KERNEL_CUHIP_Huffman_ReVISIT_lite` break handler: read `s_book[MaxBkLen]` one past the end of the `MaxBkLen`-element array, aliasing uninitialized `s_reduced[0]` and corrupting `par_ncell`; removed the break handler entirely — the fine path is restricted to `max_codelen ≤ 8` bits by the guard in `hf_hl.cc::encode()`, so the shard accumulator (ShardSize=4, BITWIDTH=32) never overflows and the handler was unreachable
- `GPU_scatter` / `KERNEL_CUHIP_scatter`: removed dead code — the scatter re-integration step (second half of break handling) was never called from `GPU_fine_encode`; also removed associated `d_brval`/`d_bridx`/`d_brnum`/`h_brnum` buffer allocations from `phf::Buf<E>`

### Changed
- `HuffmanStage<T>`: `phf::Buf<T>` is now reallocated only on capacity growth (inlen > cap_inlen_) or bklen change; shrinking inputs reuse the existing allocation; `phf_header.original_len` and `pardeg` now reflect the actual encode length (not the allocated capacity) — required `make_metadata` in `hf_hl.cc` to derive `pardeg` from `data_len` rather than `buf->len`
- `HuffmanStage<T>`: added symbol range validation after the histogram D2H — `sum(h_freq)` is compared against `inlen`; a mismatch means out-of-range symbols were skipped by the histogram kernel, and a `std::runtime_error` is thrown naming the count; turns the previously silent bitstream corruption into an immediate hard error
- `KERNEL_CUHIP_p2013Histogram`: fixed defective bounds check (`d <= 0 && d >= bins_len` was logically impossible and never fired); now correctly skips out-of-range symbols via unsigned comparison (`sym >= bins_len`) instead of clamping to a sentinel; also fixes potential UB for `uint32_t` inputs where large values overflow `int` before the comparison
- `Stage::execute()` contract: documented that sync calls inside `execute()` are allowed for algorithms that require host-side synchronization (D2H reads, CPU renormalization tables); stages using sync must return `false` from `isGraphCompatible()`
- `HuffmanStage<T>` / `phf::Buf<T>`: refactored all PHF internal scratch (device + pinned host) to allocate through `MemoryPool::allocatePersistentDevice` / `allocatePersistentPinned` instead of direct `cudaMalloc`/`cudaMallocHost`; `Buf<T>` destructor returns all allocations to the pool; removes the pool-bypass pattern and makes PHF footprint visible via `pool->getPersistentDeviceBytes()` / `getPersistentPinnedBytes()`
- `MemoryPool`: added `allocatePersistentDevice`, `allocatePersistentPinned`, `freePersistentDevice`, `freePersistentPinned`, and `getPersistentDeviceBytes()`/`getPersistentPinnedBytes()` footprint reporting; destructor frees any remaining persistent allocations as a safety net
- `Stage` interface: added `onFinalize(size_t estimated_inlen, MemoryPool*)` hook called by `Pipeline::finalize()` after buffer-size propagation — allows stages to pre-allocate persistent scratch at finalize time for PREALLOCATE semantic correctness; added `estimateDeviceFootprintBytes()` / `estimatePinnedFootprintBytes()` for total footprint reporting; `HuffmanStage` implements all three
- `Pipeline::finalize()`: added `notifyStagesFinalizeHooks()` sub-step that calls `onFinalize` for each stage with its estimated input size
- `docs/how_to_add_a_stage.md`: corrected execute() sync restriction — `cudaStreamSynchronize(stream)` on the stage's own stream is permitted for algorithms that require it (Huffman, ANS); clarified CUDA Graph compatibility requirement and sibling-dispatch cost in wide DAGs
- Extracted duplicated `align16`, `buildLevelTimings`, and concat buffer layout arithmetic into `src/pipeline/pipeline_utils.h`; introduced `ConcatLayout` struct with `headerSize`/`slotSize` helpers replacing open-coded offset calculations in `compressor.cpp` and `compressor_exec.cpp`
- Added `PoolBuffer`, `PinnedBuffer`, and `DeviceBuffer` RAII wrappers (private nested structs in `Pipeline`) replacing raw pointer+capacity member pairs; destructor simplified to graph handle teardown only
- Decomposed `finalize()` into six focused sub-methods: `typeCheckConnections`, `computeInputAlignment`, `refinePoolSize`, `setupGraphModeInput`, `preallocatePadBuffer`, `preallocateConcatBuffers`
- Extracted `prepareInputSource()` from `compress()` (graph-mode copy and alignment padding logic) and `buildOrReuseInvCache()` from `decompress()` (inverse DAG cache build/reuse)
- Extracted `computeFilePoolSize`, `reconstructForwardTopology`, and `buildSourceSizesFromHeader` as private static members from `decompressFromFile()`
- Centralized TOML config stage dispatch into a `kStageRegistry[]` table in `config.cpp` (`StageEntry` with `type_name`, `enum_val`, `load_fn`, `save_fn`); replaced the `if/else` load chain and `switch` save block with registry loops — adding a new stage now requires one `#include` and one registry entry instead of 3+ scattered edits; also added missing `saveQuantizerStage` (Quantizer stages were previously silently omitted from `saveConfig` output) and fixed `saveConfig` writing `"BitPack"` while `loadConfig` expected `"Bitpack"` (broken roundtrip for Bitpack stages); added `QuantizerStage::getOutlierCapacity()` getter to support the save function
- Added Doxygen class-level descriptions to `Logger`, `Zigzag<T>`, and `Negabinary<T>` which were previously undocumented
- Added "no template parameters" note and common instantiation snippet to `BitshuffleStage` and `RZEStage` stage docs
- Expanded requirements table in README and docs mainpage to include host compiler guidance (GCC 7+ / Clang 5+, upper bound set by CUDA version; NVHPC 23.11 tested in CI)
- Migrated `.github/ISSUE_TEMPLATE.md` to `.github/ISSUE_TEMPLATE/bug_report.yml` (modern GitHub Forms format)
- Reorganized `modules/` into six semantic categories: `predictors/` (Lorenzo, Diff/delta, interpolation), `transforms/` (zigzag, negabinary), `quantizers/` (quantizer), `coders/` (RLE, RZE, bitpack), `shufflers/` (bitshuffle), `fused/` (lorenzo_quant); all include paths and CMake source lists updated accordingly — this is a **breaking change** for any code that includes stage headers directly (e.g. `"encoders/diff/diff.h"` → `"predictors/diff/diff.h"`)
- Merged `fzgmod_encoders` and `fzgmod_predictors` CMake targets into a single `fzgmod_modules` target; downstream CMakeLists linking either old target must switch to `fzgmod_modules`

### Added

**Stages**
- `HuffmanStage<T>` (in progress): added `modules/util/histogram/histogram.h` and `histogram.cu` — internal GPU histogram utility (`KERNEL_CUHIP_p2013Histogram` + launch wrapper + optimizer); exposed as `fzgmod_utils` STATIC CMake target linked `PRIVATE` into `fzgmod_modules`; `phf::Buf<E>` extended with pre-allocated `d_freq`/`h_freq` (device + host histogram buffers, size = bklen) to avoid per-call `cudaMalloc` in execute; histogram launch params (`grid_dim`, `block_dim`, `shmem_use`, `r_per_block`) will be stored in `HuffmanStage<T>` and computed once in `finalize()`
- `HuffmanStage<T>` (in progress): added PHF CPU source layer — `hf_buf.cc`, `hf_hl.cc`, `hf_bk.cc`, `hf_bk_impl1.cc`, `hf_bk_internal.cc`, `hf_canon.cc` — adapted from PHF reference with `err.hh`/`timer.hh` stripped, RAII wrappers defined locally, and missing `capi_phf_encoded_bytes`/`capi_phf_coarse_tune_sublen`/`capi_phf_coarse_tune` implemented
- `HuffmanStage<T>` (in progress): added `hf_kernels.cu` (adapted from `hf_kernels.cuhip.inl`): fixed self-include bug, removed `timer.hh`, replaced `CHECK_GPU` → `FZ_CUDA_CHECK`, added explicit instantiations for `uint8_t`/`uint16_t`/`uint32_t`
- `HuffmanStage<T>`: full stage header and implementation — `huffman_stage.h` declares template with `setBklen()`, `serializeHeader`/`deserializeHeader` (11-byte header: DataType + bklen + original_len), `saveState`/`restoreState`, and `isGraphCompatible()=false`; `huffman_stage.cu` implements lazy `initBuf()` (creates `phf::Buf<T>` + runs histogram optimizer), forward execute (histogram → D2H → `build_book` → `encode` → D2D copy to pipeline output), and inverse execute (D2H phf_header read → `decode`); all six PHF CPU source files and `hf_kernels.cu` added to `fzgmod_modules` CMake sources; `HuffmanStage` registered in `stage_factory.h` and exposed in `fzgpumodules.h`; 9 tests (HF1–HF9) in `test_huffman.cpp` — HF9 is an end-to-end `LorenzoQuantStage<float,uint16_t>` → `HuffmanStage<uint16_t>` pipeline round-trip (requires `setZigzagCodes(true)` so Lorenzo codes land in [0, bklen) rather than the default signed two's-complement layout)
- `BitpackStage`: added `setAutoDetect(bool)` — when enabled, forward execute scans the input for its maximum value via CUB `DeviceReduce::Max` and selects the tightest valid power-of-two `nbits` automatically; sets `isGraphCompatible()` to `false` while active
- `HuffmanStage<T>` TOML support: added `addHuffmanStage`/`saveHuffmanStage` helpers and `{ "Huffman", ... }` entry in `kStageRegistry[]` in `config.cpp`; TOML keys are `input_type` (`"uint8"`, `"uint16"`, `"uint32"`) and optional `bklen` (default 1024)
- `examples/presets/cusz.toml`: cuSZ-style `LorenzoQuant → Huffman` preset; uses `zigzag_codes=true` with `quant_radius=512` and `bklen=1024`; inline comments document both zigzag (MODE A, recommended) and raw two's-complement (MODE B, requires `bklen=65536`) pairing modes
- `examples/compare_lorenzo_modes.cpp`: standalone example that runs `LorenzoQuantStage<float,uint16_t>` twice (zigzag=false and zigzag=true) on a float32 file, prints per-mode stats (uint16 range, signed range, outlier count, Shannon entropy, estimated coded size), and writes both uint16 code arrays to binary files for visualization; companion `plot_lorenzo_codes.py` plots a 2×2 histogram figure showing storage domain (full uint16 range for raw-delta, compact [0,2r] for zigzag) and semantic domain (signed delta, identical for both) to illustrate why bklen requirements differ by 64×

**Tests**
- Added `HuffmanStage/RoundTrip_U32` (HF10) and `HuffmanStage/ReuseAfterSizeChange` (HF11) to `test_huffman.cpp` — previously only uint8/uint16 were tested; HF11 exercises shrink-reuse (N1=8192 allocates cap, N2=2048 reuses without realloc) and verifies both passes produce exact round-trips
- Added `HuffmanStage/OutOfRangeSymbolThrows` (HF12) to `test_huffman.cpp` — verifies that symbols ≥ bklen throw `std::runtime_error` rather than silently corrupting the bitstream
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
- Added `docs/stages/huffman.md` — full `HuffmanStage<T>` reference covering CPU–GPU movement pattern (7-step forward flow with both host barriers annotated, 2-step inverse flow), internal buffer layout, serialized header format, TOML config keys, limitations (silent bklen corruption, not graph-compatible, pool bypass, reallocation on size change), and the zigzag pairing requirement; registered in `docs/stages/coders.md`
- Added `docs/stages/` — per-stage Doxygen pages covering constraints, behavioral rules, mode details, and usage examples for all eight stages (`LorenzoQuantStage`, `LorenzoStage`, `QuantizerStage`, `DifferenceStage`, `BitshuffleStage`, `RZEStage`, `RLEStage`, `BitpackStage`)
- Updated `Doxyfile` to include `docs/stages/` in `INPUT`; added `\ref stages_overview` link from the mainpage
- Populated `docs/libpressio_python.md` with full libpressio Python bindings guide: setup, quick start, `from_config` structure, all pipeline/stage options, metrics, common recipes, CUDA graph mode, stage output exposure, TOML config, and error handling

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
