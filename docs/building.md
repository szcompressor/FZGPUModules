# Building from Source {#building_from_source}

This page covers everything needed to build FZGPUModules: getting the source,
configuring with CMake, running tests, installing, and integrating into your own project.

---

## Prerequisites

| Requirement | Minimum |
|---|---|
| CUDA Toolkit | 11.2+ (stream-ordered allocator) |
| C++ Standard | C++17 |
| CMake | 3.24+ |
| Compiler | GCC 9+ or Clang 10+ |
| Host byte order | Little-endian |

---

## Getting the Source

```bash
git clone https://github.com/szcompressor/FZGPUModules.git
cd FZGPUModules
git submodule update --init --recursive   # pulls in googletest
```

The googletest testing framework is a git submodule under `third_party/googletest`.
The tomlplusplus library (used for config file parsing) is vendored in `third_party/tomlplusplus`
and requires no extra steps.

---

## Quick Build

The fastest path to a working build uses the `release` preset:

```bash
cmake --preset release
cmake --build build/release -j$(nproc)
```

This builds the library (`libfzgmod.so`) and the `fzgmod-cli` command-line tool.
Each preset gets its own build directory under `build/<preset>/`, so presets coexist.

To build with tests:
```bash
cmake --preset release -DBUILD_TESTING=ON
cmake --build build/release -j$(nproc)
```

---

## CMake Presets

| Preset | Build type | Purpose |
|---|---|---|
| `release` | Release | Normal development and testing |
| `debug` | Debug | Unoptimized build with `-Wall` |
| `asan` | Debug | AddressSanitizer + UndefinedBehaviorSanitizer |
| `compute-san` | RelWithDebInfo | CUDA Compute Sanitizer (auto-wraps ctest with memcheck) |

Configure and build any preset:

```bash
cmake --preset <preset>
cmake --build --preset <preset> -j$(nproc)
```

---

## CMake Options

| Option | Default | Description |
|---|---|---|
| `CMAKE_BUILD_TYPE` | `Release` | Build type: `Release` or `Debug` |
| `BUILD_SHARED_LIBS` | `ON` | Build as a shared library; set `OFF` for static |
| `BUILD_TESTING` | `OFF` | Build the test suite |
| `BUILD_EXAMPLES` | `OFF` | Build example executables |
| `BUILD_PROFILING` | `OFF` | Build profiling targets (requires Nsight Systems / NVTX3) |
| `BUILD_CLI` | `ON` | Build the `fzgmod-cli` command-line tool |
| `USE_SANITIZER` | — | Sanitizer mode: `ASanUbsan`, `TSan`, or `Compute` |
| `COMPUTE_SANITIZER_DEVICE_DEBUG` | `OFF` | Add `-G` to CUDA builds for source-level Compute Sanitizer (much slower) |
| `CMAKE_CUDA_ARCHITECTURES` | — | CUDA compute capability (e.g., `75`, `80`, `90`); comma-separated for multiple |
| `CMAKE_INSTALL_PREFIX` | `/usr/local` | Installation prefix for `cmake --install` |

---

## Common Build Examples

**Development build** (with tests, profiling, examples, and sanitizer support):

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Debug \
  -DBUILD_TESTING=ON \
  -DBUILD_PROFILING=ON \
  -DBUILD_EXAMPLES=ON \
  -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DCMAKE_INSTALL_PREFIX=$HOME/.local

cmake --build build -j$(nproc)
```

Replace `80` with your GPU's compute capability (e.g., `75` for V100, `80` for A100, `90` for H100).
See [NVIDIA CUDA Compute Capability](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability)
for your GPU.

**Release build** (optimized, with examples):

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_EXAMPLES=ON \
  -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DCMAKE_INSTALL_PREFIX=/opt/fzgmod

cmake --build build -j$(nproc)
cmake --install build
```

---

## Binary Output

| Target | Location |
|---|---|
| Library (`libfzgmod.so` / `.a`) | `build/release/` |
| CLI (`fzgmod-cli`) | `build/release/bin/` |
| Examples | `build/release/bin/examples/` |
| Tests | `build/release/tests/` |
| Profiling | `build/release/bin/profiling/` |

---

## Testing

Build tests with `-DBUILD_TESTING=ON`:

```bash
cmake --preset release -DBUILD_TESTING=ON
cmake --build build/release -j$(nproc)
```

Then run the full suite with ctest:

```bash
ctest --preset default           # all tests (release build)
ctest --preset stages            # stage unit tests only
ctest --preset pipeline          # pipeline integration tests only
```

Or directly against the build tree:

```bash
ctest --test-dir build/release --output-on-failure
```

### Host Sanitizers (ASan + UBSan)

```bash
cmake --preset asan
cmake --build --preset asan -j$(nproc)

LD_PRELOAD=$(gcc --print-file-name=libasan.so) \
  ASAN_OPTIONS=detect_leaks=0:abort_on_error=0:protect_shadow_gap=0 \
  UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=0 \
  ctest --preset asan
```

Two flags are required when running with CUDA:

- `protect_shadow_gap=0` — ASan maps a large shadow-memory region that overlaps the
  VA range CUDA uses for its pool allocator. Without this flag `cudaMemPoolCreate` fails
  with *out of memory* on every test.
- `detect_leaks=0` — CUDA driver internals allocate persistent host memory that is
  never freed; suppressing this avoids false positives from `libcuda.so`.

### CUDA Compute Sanitizer

The `compute-san` preset wraps every `ctest` run with
`compute-sanitizer --tool memcheck` automatically:

```bash
cmake --preset compute-san
cmake --build --preset compute-san -j$(nproc)
ctest --preset compute-san               # all tests under memcheck
ctest --preset compute-san-stages        # stage tests only
ctest --preset compute-san-pipeline      # pipeline tests only
```

For other tools (initcheck, racecheck, synccheck), run test binaries directly:

```bash
compute-sanitizer --tool racecheck --error-exitcode=1 ./build/compute-san/tests/test_pipeline
```

Note: tests run 10–100× slower under Compute Sanitizer. Use `--gtest_filter` to
target specific tests when iterating on a fix.

### ThreadSanitizer

```bash
cmake --preset release -DUSE_SANITIZER=TSan -DCMAKE_BUILD_TYPE=Debug -B build/tsan
cmake --build build/tsan -j$(nproc)
ctest --test-dir build/tsan --output-on-failure
```

Do not combine TSan with ASan in the same build.

---

## Install

```bash
cmake --install build/release --prefix /your/install/prefix
```

This installs headers, the library, and CMake package config files under the prefix.

---

## Using from CMake

After installation (or by pointing `CMAKE_PREFIX_PATH` at the build tree):

```cmake
find_package(FZGPUModules REQUIRED)
target_link_libraries(my_target PRIVATE FZGMOD::fzgmod)
```

---

## Generating Documentation

Build the Doxygen HTML documentation (requires Doxygen 1.9.8+ and Graphviz `dot`):

```bash
cmake --build build/release --target docs
```

Output is written to `docs/doxygen/html/`. Open `index.html` in a browser.
