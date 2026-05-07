# Contributing to FZGPUModules

Thanks for your interest in improving FZGPUModules. Contributions of all kinds
are welcome: bug reports, documentation, examples, tests, and code changes.

---

## Ways to contribute

- **Bug reports** — file an issue with environment details and a minimal reproducer
- **Documentation** — fix typos, clarify confusing sections, add usage examples
- **Tests** — add regression tests for existing behavior or cover edge cases
- **New stages** — see [How to Add a Stage](docs/how_to_add_a_stage.md)
- **Bug fixes and features** — open an issue to discuss first for non-trivial changes

---

## Build and test

```bash
# Clone with submodules (googletest)
git clone https://github.com/szcompressor/FZGPUModules.git
cd FZGPUModules
git submodule update --init --recursive

# Build (tests disabled by default; enable explicitly)
cmake --preset release -DBUILD_TESTING=ON
cmake --build build/release -j$(nproc)
```

Run the test suite:

```bash
ctest --preset default           # all tests
ctest --preset stages            # stage unit tests only
ctest --preset pipeline          # pipeline integration tests only
```

Run with sanitizers before submitting:

```bash
cmake --preset asan
cmake --build --preset asan -j$(nproc)
LD_PRELOAD=$(gcc --print-file-name=libasan.so) \
  ASAN_OPTIONS=detect_leaks=0:abort_on_error=0:protect_shadow_gap=0 \
  UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=0 \
  ctest --preset asan
```

See [Building from Source](docs/building.md) for preset details and sanitizer flags.

---

## Code style and conventions

- **C++17**, CUDA 11.2+. Match the style of nearby files.
- **No bare printf/cout/cerr** in library code — use `FZ_LOG(LEVEL, ...)` or `FZ_PRINT(...)`.
- **No `cudaDeviceSynchronize()`** inside `Stage::execute()` — enqueue all work on the provided `stream`.
- **Public API only in user-facing code** — include `fzgpumodules.h`; do not include `cuda_check.h` or internal headers in examples.
- **`connect()` argument order** — `pipeline.connect(downstream, upstream, "port")`.
- **Lorenzo downstream port** — connect to `"codes"`, not `"output"`.
- **Template instantiations** — many stages are templates with explicit instantiations (e.g., `RLEStage<uint16_t>`). Check the stage's documentation (e.g., `docs/stages/rle.md`) for available types before using a template stage. Adding a new type requires adding an `extern template` declaration in the header and instantiation in the `.cu` file.

---

## Changelog

For any code change (fix, feature, refactor, removal), add a one-line entry to
`CHANGELOG.md` under the appropriate `[Unreleased]` subsection (`Added`, `Changed`,
`Fixed`, or `Removed`) before finishing the change. Documentation-only edits do
not need a changelog entry.

---

## Adding a new stage

Use the scaffold script and follow the step-by-step guide:

```bash
scripts/new_stage.sh MyStageName <category>   # category: predictors, quantizers, coders, shufflers, transforms, fused
```

Full instructions: [docs/how_to_add_a_stage.md](docs/how_to_add_a_stage.md)

Key requirements for any new stage:
- All required `Stage` virtual methods implemented
- `StageType` enum value chosen (unique integer, never reuse or renumber existing values)
- Registered in `StageFactory`, `config.cpp` (TOML), and root `CMakeLists.txt`
- Tests: ForwardRoundTrip, ZeroInput, SerializeDeserialize, PipelineIntegration, SaveRestoreState

---

## Submitting a pull request

1. Fork and create a branch from `main`.
2. Make your changes and confirm tests pass (including sanitizers for code changes).
3. Add a `CHANGELOG.md` entry for code changes.
4. Open a PR against `main` with a clear description and rationale.
5. Link any relevant issues; include reproduction steps for bug fixes.
6. Mention any build flags or environment details required to validate the change.

For non-trivial new features or API additions, open an issue first to discuss the
approach before writing code — this avoids wasted effort if the design needs revision.

---

## API compatibility

Avoid breaking public API unless the change warrants a major version bump. See
[API Reference — Stability and Versioning](docs/api_reference.md#api_stability) for the full policy and a
per-PR checklist.
