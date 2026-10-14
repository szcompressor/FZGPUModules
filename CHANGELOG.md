# Changelog

All notable changes to FZGPUModules are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Version numbers follow [Semantic Versioning](https://semver.org/).

---

## [Unreleased] — 2.0.0

### Added
- Composable `fz::Pipeline` graphs with named ports, automatic inverse construction, multiple sources, TOML configuration, and a command-line interface.
- CUDA stage implementations covering prediction, quantization, transforms, shuffling, entropy coding, and structural graph operations; see the supported catalog in `docs/stages/index.md`.
- Forward and inverse specialization for supported stage chains, with installed-group diagnostics and staged fallback for unsupported configurations.
- Self-describing FZM 3.1 archives with serialized stage configuration and checksums; the reader accepts supported FZM 3.0 archives with a warning.
- Explicit borrowed/owned device-buffer interfaces, graph-wide memory planning, reusable output buffers, and CUDA Graph execution for supported staged pipelines.
- Installed CMake targets, embedded NVRTC device headers, out-of-tree stage support, module scaffolding, and documentation/integration checks.

### Changed
- Replaced the historical `fz::Config<T>` / `fz::Compressor<T>` interface with `fz::Pipeline`; the shared-library ABI major version is 2. Applications and external stages must be rebuilt against the v2 headers.
- Specialization uses `SpecializationPolicy`, `setSpecializationPolicy()`, and `FZ_SPECIALIZE`. `Off` remains the default; `Auto` selects matching admitted implementations without a per-input speedup prediction. Earlier C++/environment fusion names remain compatibility aliases.
- CUDA JIT supports one primary context on one device per process, including independent pipelines and concurrent first use. Custom contexts, device/context switching, and resetting the owning context are unsupported.
- Installed specialization no longer requires the source checkout; device-header dependencies are embedded in the library.
- Module-local CMake declarations and an explicit catalog replace root-level module source lists.
- Stable main contains the supported CUDA catalog. Experimental modules, AMD/backend development, research artifacts, profiling programs, cards, and demonstrations are preserved on `development`.
- Libpressio integration uses `fzgpumodules:specialization` without a `fusion` option alias. The adapter probes optional stage APIs and owns/installs the Python `Chain` helper; rebuild the adapter against the installed SDK.
- Docker images compile for SM 80, 86, and 90 by default, with configurable architecture lists and build parallelism.

### Fixed
- Archives and decoded outputs preserve logical input lengths for partial chunks. Older archives retain their stored lengths; missing original lengths cannot be recovered.
- JIT kernel ownership and caching enforce the supported primary-context contract and handle concurrent lookup safely.
- File decompression keeps inverse results under DAG ownership and transfers caller-owned output only after successful completion, preventing duplicate frees.
- Sanitizer checks corrected ANS header alignment/padding, Huffman copy lengths, outlier initialization, empty-input counter state, and GInterp shared-memory ordering.
- Generated forward/inverse specialization handles declared side outputs and staged boundary inputs while preserving the supported graph/archive contracts.
- Static installs include utility dependencies; sanitizer builds instrument host code in CUDA sources and target the runner GPU architecture.

### Removed
- Experimental CDF97/SPECK/wavelet correction, SZx, monolithic SZp, and combined quantizer/AdaptiveLorenzo implementations from stable main. Their archive IDs remain reserved; decoding those archives requires a compatible development build.
- Active HIP/SYCL build configurations from stable main pending backend validation.
- Research/demo materials and the duplicate Python helper from the core release tree. Release-preparation records remain local internal documentation.

---

## [1.0.0] — 2026-04-14

Initial tagged release. cuSZ based compressor with modular design and experimental CUDASTF support.

[Unreleased]: https://github.com/szcompressor/FZGPUModules/compare/release...HEAD
[1.0.0]: https://github.com/szcompressor/FZGPUModules/releases/tag/release
