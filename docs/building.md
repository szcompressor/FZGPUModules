# Building from Source {#building_from_source}

This page covers everything needed to build FZGPUModules: getting the source,
configuring with CMake, running tests, installing, and integrating into your own project.

---

## Prerequisites

| Requirement | Requirement or scope |
|---|---|
| CUDA Toolkit | 11.2+ configure-time API floor (stream-ordered allocator) |
| AMD backend (experimental) | HIP, hipCUB, and rocThrust packages required; backend coverage remains experimental |
| C++ Standard | C++17 |
| CMake | 3.24+ |
| Python | Python 3 (build-time header embedding and source fingerprints) |
| Compiler | C++17 compiler and standard library supported by the selected GPU toolkit |
| Host byte order | Little-endian |

The compiler must provide C++17 language and library facilities, including
`<charconv>` used by the vendored TOML parser. Select a host compiler supported by
your CUDA toolkit; a compiler's C++17 support alone does not establish CUDA
compatibility. For HIP, use ROCm's supported compiler toolchain. The repository
does not establish a tested minimum GCC or Clang version, and the CUDA 11.2
configure-time floor is not a guarantee that every older toolkit builds current
sources. The build-check container specifies CUDA 12.6; the GPU CI workflow loads
the NVHPC 23.11 CUDA 12 toolchain when available.

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

CUDA specialization embeds the shipped device harnesses and their transitive
project headers into the library at build time. Installed libraries can compile
generated kernels after the source checkout is removed or the installation is
relocated; no source-path setting is required. CUDA/NVRTC and driver runtime
dependencies are still required. Header changes are picked up on the next build.

To build with tests:
```bash
cmake --preset release -DBUILD_TESTING=ON
cmake --build build/release -j$(nproc)
```

To validate source-independent installed specialization on a CUDA machine:

```bash
python3 scripts/check_installed_specialization.py --jobs 2
# Also instrument the generated kernels with Compute Sanitizer:
python3 scripts/check_installed_specialization.py --jobs 2 --compute-sanitizer
```

This check builds a temporary source snapshot, installs it, builds an external
consumer, removes the snapshot/build directories, and relocates the installation.
It then checks warp and chunk forward/inverse specialization, staged-versus-Auto
payload and reconstruction identity, aligned and partial inputs, error bounds,
repeated decode, and archive reload in a fresh process. The temporary files are
cleaned up afterward.

---

## CMake Presets

| Preset | Build type | Purpose |
|---|---|---|
| `release` | Release | Normal development and testing |
| `debug` | Debug | Unoptimized build with `-Wall` |
| `asan` | Debug | AddressSanitizer + UndefinedBehaviorSanitizer |
| `compute-san` | RelWithDebInfo | CUDA Compute Sanitizer (auto-wraps ctest with memcheck) |
| `hip` | Release | Experimental AMD HIP/ROCm build |

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
| `FZGMOD_BACKEND` | `CUDA` | Backend: supported `CUDA`, experimental `HIP`, or unimplemented `SYCL` |
| `USE_SANITIZER` | — | Sanitizer mode: `ASanUbsan`, `TSan`, or `Compute` |
| `COMPUTE_SANITIZER_DEVICE_DEBUG` | `OFF` | Add `-G` to CUDA builds for source-level Compute Sanitizer (much slower) |
| `CMAKE_CUDA_ARCHITECTURES` | — | CUDA compute capability (e.g., `75`, `80`, `90`); comma-separated for multiple |
| `CMAKE_INSTALL_PREFIX` | `/usr/local` | Installation prefix for `cmake --install` |

## Module source declarations

`modules/CMakeLists.txt` explicitly catalogs public stages, shared infrastructure,
and legacy archive implementations. Each directory has a local `CMakeLists.txt`
declaring its sources with `target_sources()`. Backend exclusions and private
dependencies belong in the affected module's file. The root creates and links the
library targets, then loads this catalog.

`scripts/new_stage.sh` creates the local declaration and registers its directory;
adding a stage does not require editing the root source list. Run
`python3 scripts/check_stage_integration.py` to check catalog membership, source
paths, factories and the public registration surfaces. See
[Adding a Stage](how_to_add_a_stage.md) for the remaining stage contracts.

---

## Common Build Examples

### Experimental HIP/ROCm backend

HIP support is experimental: it does not yet have the same CI and stage coverage as
CUDA, and some CUDA-specific stages such as ANS are unavailable. Configure ROCm's
package root and select the GPU architecture explicitly when the `gfx908` preset
default is not appropriate:

```bash
export CMAKE_PREFIX_PATH=/opt/rocm:$CMAKE_PREFIX_PATH
cmake --preset hip -DCMAKE_HIP_ARCHITECTURES=gfx90a
cmake --build --preset hip -j$(nproc)
ctest --preset hip
```

An installed HIP build records its backend in `FZGPUModulesConfig.cmake`; downstream
`find_package(FZGPUModules)` calls resolve ROCm dependencies rather than requiring a
CUDA Toolkit.

### CUDA examples

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

Run the documentation and stage integration checks with the same entry point as CI:

```bash
python3 scripts/check_docs.py
# Include per-check detail:
python3 scripts/check_docs.py --verbose
```

These checks do not run GPU code. C++ snippet checking needs a host compiler and
CUDA headers; its existing skip behavior is retained when those are unavailable.
Host-only checker regression fixtures run with
`python3 -m unittest discover -s scripts/tests -p 'test_check_*.py'`.

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
