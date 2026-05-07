# FZGPUModules

GPU-accelerated composable compression pipeline library for scientific floating-point data.

## Overview

FZGPUModules is a CUDA library for building composable, high-throughput compression pipelines. Each pipeline is a directed acyclic graph (DAG) of stages - coders, predictors, quantizers, shufflers, transforms, fused stages, and external stages - connected and executed entirely on the GPU with stream-ordered memory management.



**Key properties:**
- **Modular** — mix and match stages (Lorenzo, Quantizer, RLE, RZE, Bitshuffle, …)
- **High throughput** — parallel level execution, persistent scratch, CUDA Graph support
- **Memory-efficient** — MINIMAL and PREALLOCATE strategies; buffer coloring to alias non-overlapping allocations
- **Self-describing files** — FZM format embeds full stage config with CRC32 checksums

---

## Requirements

| Requirement | Minimum |
|---|---|
| CUDA Toolkit | 11.2+ |
| C++ Standard | C++17 |
| CMake | 3.24+ |
| Host byte order | Little-endian |

---

## Building and Installing

```bash
git clone https://github.com/szcompressor/FZGPUModules.git
cd FZGPUModules
git submodule update --init --recursive

cmake --preset release
cmake --build build/release -j$(nproc)
cmake --install build/release --prefix /your/install/prefix
```

**Downstream CMake:**
```cmake
find_package(FZGPUModules REQUIRED)
target_link_libraries(my_target PRIVATE FZGMOD::fzgmod)
```

---

## Quick Start

```cpp
#include "fzgpumodules.h"

// 1. Build a pipeline
fz::Pipeline pipeline(n * sizeof(float));

auto* lrz = pipeline.addStage<fz::LorenzoQuantStage<float, uint16_t>>(
    fz::LorenzoQuantStage<float, uint16_t>::Config{1e-4f});
auto* rle = pipeline.addStage<fz::RLEStage<uint16_t>>();

pipeline.connect(rle, lrz, "codes");
pipeline.finalize();

// 2. Compress
void* d_compressed = nullptr;
size_t compressed_size = 0;
pipeline.compress(d_input, n * sizeof(float), &d_compressed, &compressed_size, stream);
// d_compressed is pool-owned — do NOT cudaFree it.

// 3. Decompress
void* d_output = nullptr;
size_t output_size = 0;
pipeline.decompress(d_compressed, compressed_size, &d_output, &output_size, stream);
cudaStreamSynchronize(stream);
// d_output is pool-owned — do NOT cudaFree it.
// Call pipeline.setPoolManagedDecompOutput(false) for caller-owned output.
```

See `examples/` for more patterns: caller-allocated output, CUDA Graph capture, file I/O, multi-branch pipelines.

---

## Available Stages

| Stage | Description |
|---|---|
| `LorenzoQuantStage<TInput, TCode>` | Fused float predictor + quantizer (lossy, 1D/2D/3D) |
| `LorenzoStage<T>` | Plain integer Lorenzo predictor (lossless) |
| `QuantizerStage<TInput, TCode>` | Direct-value quantizer (ABS/REL/NOA error modes) |
| `DifferenceStage<T, TOut>` | First-order difference / cumulative-sum coding |
| `RLEStage<T>` | Run-length encoding |
| `BitshuffleStage` | GPU bit-matrix transpose |
| `RZEStage` | Recursive zero-byte elimination |
| `ZigzagStage<TIn, TOut>` | Zigzag encode/decode |
| `NegabinaryStage<TIn, TOut>` | Negabinary encode/decode |
| `BitpackStage<T>` | Pack/unpack power-of-two value streams |

---

## Memory Strategies

| Strategy | Description |
|---|---|
| `MINIMAL` | Allocate on demand, free at last consumer. Lowest peak GPU memory. Default. |
| `PREALLOCATE` | Allocate everything at `finalize()`. Required for CUDA Graph capture. Enables buffer coloring. |

```cpp
fz::Pipeline pipeline(input_bytes, fz::MemoryStrategy::PREALLOCATE);
```

---

## CLI

```bash
# Compress using a stage chain
fzgmod-cli -z -i data.f32 -o compressed.fzm --stages "lorenzo->bitshuffle->rze" -m rel -e 1e-3

# Decompress and compare
fzgmod-cli -x -i compressed.fzm -o decompressed.f32 --compare data.f32 --report

# Compress using a TOML pipeline config
fzgmod-cli -z -i data.f32 -c examples/presets/pfpl.toml -o compressed.fzm --report
```

---

## Citation

If you reference this work, please cite:

> Note: This citation corresponds to the v1.0 release; the 2.0 API may differ.

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

For full documentation — API reference, stage details, build options, file format, and contributor guides — see the [official docs](https://szcompressor.github.io/FZGPUModules/).
