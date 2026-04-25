# FZGPUModules

GPU-accelerated modular lossy compression pipeline for scientific floating-point data.

## Overview

FZGPUModules is a CUDA library for building composable, high-throughput compression
pipelines. Each pipeline is a directed acyclic graph (DAG) of stages — predictors,
quantizers, encoders, and transforms — connected and executed entirely on the GPU
with stream-ordered memory management.

**Key properties:**
- **Modular** — mix and match stages (Lorenzo, Quantizer, RLE, RZE, Bitshuffle, …)
- **High throughput** — parallel level execution, persistent scratch, CUDA Graph support
- **Memory-efficient** — MINIMAL and PREALLOCATE strategies; buffer coloring to alias non-overlapping allocations

---

## Requirements

| Requirement | Minimum |
|---|---|
| CUDA Toolkit | 11.2+ (stream-ordered allocator) |
| C++ Standard | C++17 |
| CMake | 3.24+ |
| Host byte order | Little-endian |

---

## Quick Start

```cpp
#include "fzgpumodules.h"

// 1. Build a pipeline
fz::Pipeline pipeline(fz::MemoryPoolConfig(input_bytes));

auto* lrz = pipeline.addStage<fz::LorenzoQuantizerStage<float, uint16_t>>(
    fz::LorenzoQuantizerStage<float, uint16_t>::Config{1e-4f});
auto* rle = pipeline.addStage<fz::RLEStage<uint16_t>>();

pipeline.connect(rle, lrz, "codes");
pipeline.finalize();

// 2. Compress
void* d_compressed = nullptr;
size_t compressed_size = 0;
pipeline.compress(d_input, n * sizeof(float), &d_compressed, &compressed_size, stream);

// 3. Decompress
void* d_output = nullptr;
size_t output_size = 0;
pipeline.decompress(d_compressed, compressed_size, &d_output, &output_size, stream);
cudaStreamSynchronize(stream);
// d_output is caller-owned — call cudaFree() when done.
cudaFree(d_output);
```

See `examples/` for more usage patterns including multi-branch pipelines, CUDA Graph
capture, and the low-level DAG API.

---

## Command Line Interface (CLI)

FZGPUModules provides a fully-featured CLI (`fzgmod-cli`) for testing, comparing, and benchmarking pipelines without writing C++ code.

**Dynamic Linear Pipelines**
Chain stages together using `--stages` with `->` separators — any number of stages. 
*(Note: Always wrap the stage list in quotes to prevent your shell from interpreting `->` as file redirection)*:
```bash
# Lorenzo -> Bitshuffle -> RZE (the default)
fzgmod-cli -z -i data.f32 -o compressed.fzm --stages "lorenzo->bitshuffle->rze" -m rel -e 1e-3

# Four-stage pipeline
fzgmod-cli -z -i data.f32 --stages "lorenzo->diff->bitshuffle->rze" -e 1e-4
```

**Decompress, Compare, and Report**
Decompress an `.fzm` file and compare it against the original to verify correctness and compute error metrics:
```bash
fzgmod-cli -x -i compressed.fzm -o decompressed.f32 --compare data.f32 --report
# Prints: Output size, Time, Throughput, Value Range, Max Abs Error, PSNR, NRMSE
```

**Complex Branched Pipelines via TOML**
For non-linear pipelines or per-stage configuration, use a TOML config file:
```bash
fzgmod-cli -z -i data.f32 -c examples/presets/pfpl.toml -o compressed.fzm --report
```
Ready-to-use presets are in `examples/presets/`:
- `pfpl.toml` — Quantizer (REL) → Difference → Bitshuffle → RZE
- `speed.toml` — Quantizer → Bitshuffle → RZE

The CLI passes the input file size to the `Pipeline` constructor before calling `loadConfig()`,
so presets that use `memory_strategy = "PREALLOCATE"` work correctly even without an
`input_size` key in the TOML file.

**Benchmarking**
Run a pipeline multiple times to measure host time, DAG time, and throughput (GB/s):
```bash
fzgmod-cli -b -i data.f32 --stages lorenzo->bitshuffle->rze -m rel -e 1e-3 --runs 10
```

**Key Flags**

| Flag | Description |
|---|---|
| `-z` / `-x` / `-b` | Compress / Decompress / Benchmark mode |
| `-i <file>` | Input file |
| `-o <file>` | Output file |
| `-c <file.toml>` | Load pipeline from TOML config |
| `--stages <s1->s2->...>` | Ordered stage chain (lorenzo, quantizer, bitshuffle, rze, diff, rle) |
| `-t <f32\|f64>` | Data type (default: f32) |
| `-m <rel\|abs\|noa>` | Error bound mode (default: rel) |
| `-e <val>` | Error bound value (default: 1e-3) |
| `-r <val>` | Quantization radius (default: 32768) |
| `-l <x>x<y>x<z>` | Dimensions (inferred if omitted) |
| `-R` / `--report` | Print compression ratio and throughput |
| `--compare <file>` | Compare decompressed vs original (MaxErr, PSNR, NRMSE) |
| `--runs <n>` | Benchmark iteration count (default: 10) |

---

## Caller-Allocated Output (With Size Query)

If you want full memory control, use the caller-allocated `compress` and `decompress` overloads and ask the pipeline for max output sizes before allocating:

```cpp
// After finalize()
size_t comp_capacity = pipeline.getMaxCompressedSize(input_bytes);
// typically the decompressed size is known or equals input_bytes
size_t decomp_capacity = input_bytes; 

void* d_comp_user = nullptr;
void* d_decomp_user = nullptr;
cudaMalloc(&d_comp_user, comp_capacity);
cudaMalloc(&d_decomp_user, decomp_capacity);

size_t comp_size = 0;
pipeline.compress(d_input, input_bytes,
                  d_comp_user, comp_capacity,
                  &comp_size, stream);

size_t decomp_size = 0;
pipeline.decompress(d_comp_user, comp_size,
                    d_decomp_user, decomp_capacity,
                    &decomp_size, stream);
```

For `.fzm` files, you can query the exact decompressed size from the header metadata before allocating:
- `fz::Pipeline::readHeader(path)` -> `header.core.uncompressed_size`

See `examples/caller_allocated_output.cpp` for a minimal end-to-end example.

---

## Building from Source

```bash
git clone https://github.com/szcompressor/FZGPUModules.git
git submodule update --init --recursive
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

**CMake options:**

| Option | Default | Description |
|---|---|---|
| `BUILD_SHARED_LIBS` | `ON` | Build shared libraries |
| `BUILD_EXAMPLES` | `OFF` | Build example programs |
| `BUILD_PROFILING` | `OFF` | Build profiling programs |
| `BUILD_TESTING` | `OFF` | Build test suite |
| `FZ_LOG_MIN_LEVEL` | `2` (INFO) | 0=TRACE 1=DEBUG 2=INFO 3=WARN 255=SILENT |

---

## Installing

```bash
cmake --install build --prefix /your/install/prefix
```

This installs headers, shared libraries with versioned symlinks
(`libfzgmod.so → libfzgmod.so.2 → libfzgmod.so.2.0.0`), and CMake
package config files.

### Downstream CMake usage

```cmake
find_package(FZGPUModules REQUIRED)
target_link_libraries(my_target PRIVATE FZGMOD::fzgmod)
```

All targets: `FZGMOD::fzgmod`, `FZGMOD::fzgmod_mem`, `FZGMOD::fzgmod_encoders`,
`FZGMOD::fzgmod_predictors`, `FZGMOD::fzgmod_pipeline`.

---

## Available Stages

| Stage | Description |
|---|---|
| `LorenzoQuantizerStage<TInput, TCode>` | 1-D/2-D/3-D Lorenzo predictor |
| `QuantizerStage<TInput, TCode>` | Direct-value quantizer (ABS/REL/NOA error modes) |
| `RLEStage<T>` | Run-length encoding |
| `DifferenceStage<T, TOut>` | First-order difference / cumulative-sum coding |
| `BitshuffleStage` | GPU bit-matrix transpose |
| `RZEStage` | Recursive zero-byte elimination |
| `ZigzagStage<TIn, TOut>` | Zigzag encode/decode |
| `NegabinaryStage<TIn, TOut>` | Negabinary encode/decode |
| `BitpackStage<T>` | Pack/unpack power-of-two value streams |

---

## Memory Strategies

| Strategy | Description |
|---|---|
| `MINIMAL` | Allocate on demand, free at last consumer. Lowest peak GPU memory. |
| `PREALLOCATE` | Allocate everything at `finalize()`. Required for CUDA Graph capture. Enables buffer coloring. |

---

## File I/O

```cpp
// Write to file after compressing
pipeline.writeToFile("output.fzm", stream);

// Decompress directly from file (no pipeline setup needed)
void* d_out = nullptr;
size_t out_size = 0;
fz::Pipeline::decompressFromFile("output.fzm", &d_out, &out_size, stream);
cudaStreamSynchronize(stream);
cudaFree(d_out);
```

FZM files embed the full stage configuration and compressed payload with CRC32
checksums. See `include/fzm_format.h` for the format specification.

---

## CUDA Graph Support

For throughput-critical workloads, enable CUDA Graph capture to eliminate
CPU-side kernel launch overhead on repeated compress calls:

```cpp
pipeline.setCaptureMode(true);           // requires PREALLOCATE strategy
pipeline.finalize();
pipeline.warmup(stream);                 // JIT-compiles all kernels once
// subsequent compress() calls replay the captured graph
```

---

## Key API Reference

| Class | Header | Description |
|---|---|---|
| `fz::Pipeline` | `pipeline/compressor.h` | High-level builder and executor |
| `fz::Stage` | `stage/stage.h` | Base class for all compression stages |
| `fz::CompressionDAG` | `pipeline/dag.h` | Low-level DAG wiring and execution |
| `fz::MemoryPool` | `mem/mempool.h` | Stream-ordered CUDA memory pool |
| `fz::PipelinePerfResult` | `pipeline/perf.h` | Per-stage profiling results |

Full API documentation is available on the [Doxygen site](https://szcompressor.github.io/FZGPUModules/).

---

## Citation

If you reference this work, please cite the following publication.

> Note: This citation corresponds to the v1.0.0 release of this codebase, not the current version.

- **[DRBSD-11]** FZModules: A Heterogeneous Computing Framework for Customizable Scientific Data Compression Pipelines
    ```bibtex
    @inproceedings{ruiter2025fzmodules,
        author = {Ruiter, Skyler and Tian, Jiannan and Song, Fengguang},
        title = {FZModules: A Heterogeneous Computing Framework for Customizable Scientific Data Compression Pipelines},
        year = {2025},
        url = {https://doi.org/10.1145/3731599.3767376},
        booktitle = {Proceedings of the SC '25 Workshops of the International Conference for High Performance Computing, Networking, Storage and Analysis},
        pages = {332-338},
        series = {SC Workshops '25}
    }
    ```

---

## Adding a Custom Stage

See `memory/how_to_add_a_stage.md` for the step-by-step guide, or use the
scaffold script:

```bash
scripts/new_stage.sh MyStageName transforms
```
