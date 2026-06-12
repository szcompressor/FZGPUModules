# Changelog

All notable changes to FZGPUModules are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Version numbers follow [Semantic Versioning](https://semver.org/).

---

## [Unreleased] — 2.0.0

### Changed
- `QuantizerStage` — dropped the `outlier_count` DAG output port (same refactor as the fused stages). Forward ports in the normal scatter path: 4 → 3 (`codes`, `outlier_vals`, `outlier_idxs`). Inplace mode (`setInplaceOutliers(true)`) is unchanged — still a single `codes` port. Added `~QuantizerStage()`, `onFinalize()`, `estimateDeviceFootprintBytes()` reporting `0` (inplace) or `4` bytes; replaced the `const void* d_outlier_count_ptr_` mirror with a stage-private `uint32_t* d_outlier_count_scratch_` allocated via `pool->allocatePersistentDevice("quantizer_outlier_count")`. `scatter_assign_kernel` (shared `predictor_utils.cuh`) now takes `uint32_t n` by value instead of `const uint32_t* count_ptr`. Inverse path reads count from `actual_outlier_count_` (populated by `deserializeHeader()` on the .fzm path, or `postStreamSync()` for in-memory round-trips). `tests/stages/test_quantizer.cpp` helpers no longer download/upload a 4th `count` buffer — `QuantizerForwardResult.outlier_count` is derived from the trimmed `outlier_idxs` size after `postStreamSync()`.
- `LorenzoQuantStage` / `GInterpStage` — dropped the `outlier_count` DAG output port. The 4-byte counter now lives in a stage-private device scratch allocated in `onFinalize()` via `pool->allocatePersistentDevice`; it is D2H'd in `postStreamSync()` and is already serialized in the FZM stage header. The inverse path receives the count as a `uint32_t` kernel-launch argument (read from the deserialized header) — the scatter kernel no longer dereferences a device pointer to know its loop bound. **Breaking change** for any code that connected to / read from the `outlier_count` port name or invoked the stages standalone with a `d_outlier_count` output slot. Forward ports for LorenzoQuant: 4 → 3 (`codes`, `outlier_errors`, `outlier_indices`); forward ports for GInterp: 5 → 4 (`codes`, `anchor`, `outlier_vals`, `outlier_idxs`).
- `LorenzoQuantStage` / `GInterpStage` — added `onFinalize()` override, destructor, `estimateDeviceFootprintBytes()` reporting the 4-byte scratch. `LorenzoQuantStage` inverse launchers (`launchLorenzoInverseKernel`, `…2D`, `…3D`) and `GInterpStage`'s `launchScatterOutliers` now take `uint32_t n` instead of `const uint32_t* count_ptr`. `scatter_outliers_kernel` (shared cuh) and `ginterp::scatterOutliersKernel` lost their device-pointer count argument.
- `examples/cusz_standalone.cpp` — dropped `d_outlier_count` cudaMalloc/cudaFree pair; added `lq.onFinalize(input_bytes, &pool)` call; compress now passes 3 outputs (was 4); decompress passes 3 inputs (was 4). Doc block updated to reflect the new contract.
- `examples/analyze_lorenzo.cpp`, `examples/compare_lorenzo_modes.cpp` — dropped `d_count` allocation; added `stage.onFinalize(in_bytes, &pool)` call; outlier count derived from the trimmed `outlier_indices` port size after `postStreamSync()`.
- Forward execute paths in `LorenzoQuantStage` and `GInterpStage` no longer zero `actual_outlier_count_` — this method may run during `cudaStreamBeginCapture`/`EndCapture` (graph recording), where `postStreamSync()` isn't called and clobbering the count would break the subsequent inverse pass.

### Fixed
- `src/mem/mempool.cpp` `MemoryPool::setReleaseThreshold()`: skip `config_` update when `bytes == UINT64_MAX` (the "never trim" sentinel); the previous code set `config_.input_data_size = UINT64_MAX`, causing `getConfiguredSize()` → `getPoolSize()` to overflow a float cast and return 0, which made `Pipeline::getPoolThreshold()` always return 0 post-finalize
- `examples/analyze_lorenzo.cpp`, `examples/compare_lorenzo_modes.cpp`: replaced internal `#include "fused/lorenzo_quant/lorenzo_quant.h"` and `#include "mem/mempool.h"` with `#include "fzgpumodules.h"` to comply with the public-API-only rule for example code
- `examples/simple_api_lorenzo_dual_branch.cpp` → `examples/lorenzo_intro.cpp`: renamed to match new catalog naming convention; CMake target renamed `simple_api_lorenzo_dual_branch` → `lorenzo_intro`; doc block updated; fixed misplaced `#include <fstream>` stranded after `using namespace fz;`; updated output header label
- `examples/pfpl_pipeline.cpp` → `examples/pfpl_memory_strategies.cpp`: renamed for clarity; CMake target renamed `pfpl_example` → `pfpl_memory_strategies`; doc block reframed around the memory-strategy comparison concept; removed `getPoolThreshold()` call and table row (returns 0 post-finalize)
- `examples/manual_pipeline.cpp` → `examples/pfpl_manual_vs_dag.cpp`: renamed to match the existing binary name; doc block updated to frame the "when to bypass the DAG" angle
- `examples/pfpl_graph_capture.cpp`: removed `getPoolThreshold()` calls and "Pool threshold" row from comparison table (same post-finalize = 0 issue); added Build reminder to doc block
- `examples/file_io_example.cpp`: fixed Usage binary path in doc block; added Build reminder; verified all four decompress paths against current API
- `examples/ownership_example.cpp`: fixed Usage binary path in doc block; added Build reminder; verified all four ownership sections
- `examples/debug_logging.cpp`: fixed Usage binary path in doc block; added Build reminder; removed policy-violating `#include "log.h"` (already available via `fzgpumodules.h`)
- `examples/minimal_intro.cpp`: new hello-world example — LorenzoQuantStage→HuffmanStage compress+decompress+verify, synthetic data, ~80 lines
- `examples/toml_config.cpp`: new TOML config example — loadConfig() from preset, saveConfig(), loadConfig() round-trip verification, Pipeline::readHeader() on .fzm file; documents the correct Pattern of Pipeline(data_bytes)+loadConfig() for PREALLOCATE presets
- `examples/cusz_standalone.cpp`: new standalone stage execution example — LorenzoQuantStage+HuffmanStage driven via execute() without Pipeline; covers pool construction, onFinalize() scratch pre-allocation, postStreamSync() for outlier count, setInverse() for decompress direction
- `adm_map_decoupled_u16`/`adm_map_thrust_u16`/`adm_map_decoupled_u32`/`adm_map_thrust_u32`: added `#ifndef NDEBUG` overflow sentinel — kernels call `atomicOr(d_overflow_flag, 1)` when a thread's `bit_offset` exceeds `kChunk × kMaxSignalBytes × 8`; host checks the flag after `cudaStreamSynchronize` and throws a `std::runtime_error` to catch inputs that violate the algorithm's bounded-diff assumption
- `adm_map_decoupled_u16`/`adm_map_decoupled_u32`: `__shared__ excl_sum` was uninitialized for the first warp block (warp=0), causing non-deterministic writes to `d_concat_signals`; initialize to 0 at kernel entry
- `tests/stages/test_adm.cpp` AD2 (`U32RoundTrip`): `make_u32_data` amplitude (±12000) exceeded the algorithm's per-thread `local_bits` capacity (64 bytes, supports max diff ≤ 4032); reduced amplitude to ±250
- `tests/stages/test_adm.cpp` AD7 (`SerializeDeserialize`): used `adm_encode<uint16_t>` with `dtype=U32`, causing the U32 kernel to mis-interpret uint16_t bytes as uint32_t values with huge diffs; changed to `adm_encode<uint32_t>` with matching `make_u32_data`

- `GInterpStage` phase 2: `INTERPOLATION_PARAMS` auto-tuning via `setAutoTuning(uint8_t mode)`; supports mode `1` (cheap profiling, ~1 ms — sets per-level `reverse[]` from a 2-error probe via `c_spline_profiling_data`) and mode `3` (full structural, ~5–15 ms — sets `use_md`/`use_natural`/`reverse` per level from an 18-error probe via `pa_spline_infprecis_data` with workflow=true, matching the cuSZ-Hi paper); alpha is interpolated piecewise-linearly from `rel_eb` (cuSZ-Hi recipe; beta fixed at 4.0); persistent 36-float device + pinned-host profiling scratch allocated via `pool->allocatePersistentDevice` / `allocatePersistentPinned` in `onFinalize()` (only when `auto_tuning_mode > 0`); resolved `INTERPOLATION_PARAMS` (alpha, beta, three uint8[6] flag arrays) embedded in the FZM stage header so the decompressor uses the same configuration without re-tuning
- `GInterpConfig` grown from 64 → 88 bytes: added explicit `intp_alpha`/`intp_beta` (f64) + `intp_use_md`/`intp_use_natural`/`intp_reverse` (uint8[6] each) + `auto_tuning_mode` (u8) fields, replacing the phase-1 `reserved[28]` slot. Still well under `FZM_STAGE_CONFIG_SIZE`. This is a **breaking change** for any phase-1 `.fzm` file with a `GInterpStage` entry (the dev branch is unreleased so no compatibility shim was added)
- `ginterp_kernels.{h,cu}`: launcher signatures `launchGInterpForward3D` / `launchGInterpInverse3D` now take `const INTERPOLATION_PARAMS&` (was default-constructed inside the launcher). Added `launchGInterpResetErrors` (zeroes the 36-float scratch), `launchGInterpProfileMode1` (cheap reverse-only profile via `c_spline_profiling_data`), and `launchGInterpProfileMode3` (full structural profile via `pa_spline_infprecis_data` with workflow=true). `INTERPOLATION_PARAMS` is forward-declared in the public `ginterp_kernels.h` header so callers don't pull in `cusz_type_subset.h`
- `tests/stages/test_ginterp.cpp`: added GI10 (`AutoTuneMode1` round-trip), GI11 (`AutoTuneMode3` round-trip on 64³), GI12 (`AutoTuneFileRoundTrip` — verifies resolved params survive `.fzm` round-trip), GI13 (`AutoTuneSerializeHeader` — raw serialize/deserialize round-trip preserves `auto_tuning_mode`). All 14 tests pass.

### Fixed
- `modules/fused/ginterp/ginterp_md.inl` line 1777: `global2shmem_profiling_data<>` call was missing the `SPLINE_DIM` template argument (upstream cuSZ-Hi bug), silently shifting the rest of the int pack by one — `PROFILE_NUM_BLOCK_Z` ended up taking `LINEAR_BLOCK_SIZE`'s value (384) and producing massive shared-memory OOB writes. Passing `SPLINE_DIM` explicitly restores the intended template alignment. Required to make `setAutoTuning(1)` and `setAutoTuning(3)` not crash; surfaced via compute-sanitizer

### Added
- `GInterpStage<TInput, TCode>`: multi-level spline-interpolation predictor + quantizer adapted from cuSZ-Hi (Indiana University / Argonne National Laboratory, BSD-3-Clause); 3-D MVP using cuSZ-Hi's deterministic baseline (`LEVEL=4`, `AnchorBlockSize=16³`, `NumAnchorBlock=1`, `alpha=1.75`, `beta=4.0`); `isGraphCompatible()=false`; five output ports (`codes`, `anchor`, `outlier_vals`, `outlier_idxs`, `outlier_count`); 88-byte FZM header (`GInterpConfig`); supports `ABS`/`REL`/`NOA` error modes via the shared `ErrorBoundMode` enum; instantiated for `<float, uint8_t>`, `<float, uint16_t>`, `<float, uint32_t>`
- `GInterpStage` kernel and host files: `modules/fused/ginterp/ginterp_md.inl` (TU-private spline kernels adapted from cuSZ-Hi `spline3_md.inl` with `namespace cusz` → `namespace fz::ginterp` and `err.hh`/`timer.hh` stripped), `ginterp_kernels.cu` (forward/inverse launchers + outlier scatter helper, MVP constants), `ginterp_kernels.h` (launcher declarations), `cusz_type_subset.h` (minimal `INTERPOLATION_PARAMS` subset of `cusz/type.h`)
- `GInterpStage` radius auto-tune: `Config::quant_radius` default `0` is a sentinel for "auto"; on first compress `execute()` the stage scans `min`/`max` of the input and picks the smallest radius that fits the data range, clamped to the `TCode` bit-width's maximum (127 for u8, 32767 for u16/u32); reuses the existing NOA/REL `value_base` scan when one already happens; explicit `setQuantRadius(>0)` skips the scan (required for CUDA graph capture and for routing residuals beyond `radius*eb/2` to the outlier triplet, e.g. for climate-style data)
- `GInterpStage` registered in `stage_factory.h` (`case StageType::G_INTERP:` dispatching on `code_type`) and `config.cpp` (`addGInterpStage` / `saveGInterpStage` / `kStageRegistry` entry with TOML type `"GInterp"`, keys `input_type` / `code_type` / `error_bound` / `eb_mode` / `quant_radius` / `outlier_capacity`)
- `GInterpStage` added to `fzgpumodules.h` public include
- `StageType::G_INTERP = 22` in `fzm_format.h` plus `stageTypeToString` case
- `tests/stages/test_ginterp.cpp`: 10 tests (GI1–GI9, with GI6/GI6b split) covering ABS round-trip with auto-tune, non-cube dims, outlier handling within the documented `~2 × eb` operational envelope, file round-trip, header serialize/deserialize, stage-type id, graph-compat = false, dim-rejection of non-3D, auto-radius default behaviour, and manual radius override
- `docs/stages/ginterp.md`: full `GInterpStage` reference covering radius auto-tune, error bound and MVP limitations, port layout, TOML keys, and the cuSZ-Hi acknowledgement
- `docs/stages/predictors.md`: added `GInterpStage` entry
- `THIRD_PARTY.md`: cuSZ-Hi (BSD-3-Clause) section with full license text
- CLI `--stages`: added `ans` and `adm` as recognized stage tokens in the dynamic pipeline builder; updated help text and error message to list both
- `fzm_format.h`: replaced stale `TODO` comments on `StageType::ANS` and `StageType::ADM` with accurate descriptions
- `examples/cusz_huffman_vs_ans.cpp`: added third pipeline variant (ADM+ANS) showing `LorenzoQuantStage → ADMStage → ANSStage`; summary table now shows three data columns with `ANS/Huf` and `ADM+ANS/ANS` relative-delta columns
- CLI `-v`/`-vv`/`-vvv` and `--verbose[=N]` flags: route library log output (INFO/DEBUG/TRACE) to stderr via `fz::Logger::enableStderr()`
- CLI `--profile`: now prints the full per-stage GPU timing table (`PipelinePerfResult::print()`) in compress, decompress, and benchmark modes; benchmark captures both compress and decompress stage breakdowns from the last timed run
- CLI `--print-pipeline`: calls `pipeline->printPipeline()` after finalize to display stage topology and connections
- CLI `--bounds-check`: enables `pipeline->enableBoundsCheck(true)` for runtime buffer overrun detection
- CLI `--report` now includes peak device memory usage (`pipeline->getPeakMemoryUsage()`) for compress and benchmark modes
- CLI: TOML config path now respects `--warmup`, `--profile`, `--bounds-check`, and `--print-pipeline` flags (previously only the dynamic builder path applied these settings)
- `ADMStage`: Adaptive Data Mapping stage adapted from the MANS project (Huang et al., BSD-3-Clause); remaps `uint16_t[]`/`uint32_t[]` streams into a compact 8-bit symbol domain before entropy coding; `isGraphCompatible()=false`; 12-byte FZM header stores dtype + `num_elements`; two encode paths: decoupled look-back prefix sum for gsize ≤ 1024 and Thrust `exclusive_scan` fallback for larger arrays; 9 persistent scratch device buffers managed via `MemoryPool::allocatePersistentDevice`
- `ADMStage` kernel files: `modules/transforms/adm/mapping_uint16.cu` and `mapping_uint32.cu` — TU-private `__global__` kernels with exported host wrappers (`compress_u16`/`decompress_u16`/`get_max_u16_payload_bytes` and u32 equivalents); all per-call `cudaMalloc`/`cudaFree` from the MANS reference replaced with `AdmScratch` pool pointers
- `ADMStage` registered in `stage_factory.h` (`case StageType::ADM:`) and `config.cpp` (`addADMStage`/`saveADMStage`/`kStageRegistry` entry with TOML type `"ADM"` and `dtype` key)
- `ADMStage` added to `fzgpumodules.h` public include
- `tests/stages/test_adm.cpp`: 12 tests (AD1–AD12) covering u16/u32 round-trip, small input (< one warp block), large input (Thrust fallback path), zero input, compression ratio, header serialization, save/restore state, graph-compatibility, Pipeline integration, file round-trip, and LorenzoQuant→ADM→ANS end-to-end pipeline
- `modules/coders/ans/dietgpu/`: vendored dietGPU rANS headers (Meta Platforms, MIT license); namespace adapted to `fz::ans`, histogram functions stripped in favor of shared `fz::module::GPU_histogram_generic`
- `ANSStage`: full class definition in `ans_stage.h` — `ANSConfig` (12-byte FZM header with prob_bits + original_bytes_), 7 persistent scratch pointer fields, `isGraphCompatible()=false`, `estimateOutputSizes()`, `serializeHeader()`/`deserializeHeader()`, `saveState()`/`restoreState()`, `getRequiredInputAlignment()=4`, `estimateDeviceFootprintBytes()`, `estimateScratchBytes()`, `onFinalize()` pre-allocation path
- `ANSStage::execute()` in `ans_stage.cu`: forward path (histogram → `ansCalcWeights` → `ansEncodeBatch<10,4096>` → `batchExclusivePrefixSum` → `ansEncodeCoalesceBatch<64>` → D2H header readback); inverse path (D2H header peek → `ansDecodeTable<256>` → occupancy-based `ansDecodeKernel<128,10,4096>`); `initScratch()`/`onFinalize()`/`estimateDeviceFootprintBytes()`/`estimateScratchBytes()` implementations
- `ANSStage` registered in `stage_factory.h` (`case StageType::ANS:`) and `config.cpp` (`addANSStage`/`saveANSStage`/`kStageRegistry` entry with TOML type `"ANS"` and `prob_bits` key)
- `tests/stages/test_ans.cpp`: 12 tests (AN1–AN12) covering round-trip, zero input, compression ratio, header serialization, save/restore state, graph-compatibility, Pipeline integration, reuse-after-size-change, file round-trip, LorenzoQuant→ANS end-to-end pipeline, partial-block input (< 4096 bytes), and unsupported prob_bits validation
- `examples/cusz_huffman_vs_ans.cpp`: side-by-side throughput and compression-ratio comparison of `LorenzoQuantStage<float,uint16_t>→HuffmanStage<uint16_t>` vs `→ANSStage`; reports compress/decompress GB/s (host and DAG), CR, peak memory, and error stats per run with a final summary table
- `ANSStage` added to `fzgpumodules.h` public include
- `docs/stages/ans.md`: full stage documentation for `ANSStage` (execution flow, scratch buffers, header layout, limitations, acknowledgements)
- `docs/stages/coders.md`: added `ANSStage` entry to coder stage index
- `THIRD_PARTY.md`: added dietGPU (MIT, Meta Platforms) and MANS (BSD-3, Huang et al.) sections with verbatim license texts
- `THIRD_PARTY.md`: full copyright notices and per-module attribution for LC framework (RZE, Bitshuffle, Difference, Quantizer) and cuSZ/PHF (LorenzoQuant, Huffman)
- Attribution `@note` in Doxygen class comments for all six derived stages
- `## Acknowledgements` section added to stage docs for all six derived stages
- `README.md`: Acknowledgements section crediting LC framework and cuSZ
- `docs/how_to_add_a_stage.md`: Step 8b — attribution guide for new stages based on prior work; checklist item added
- `scripts/new_stage.sh`: attribution reminder appended to the post-run summary
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
