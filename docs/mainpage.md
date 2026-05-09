# FZGPUModules {#mainpage}

GPU-accelerated graph composable compression pipeline builder for analytical workflows.

## Overview

FZGPUModules is a CUDA library for building composable, high-throughput compression
pipelines. Each pipeline is a directed acyclic graph (DAG) of stages - coders,
predictors, quantizers, shufflers, transforms, fused stages, and external stages -
connected and executed entirely on the GPU with stream-ordered memory management.

**Key properties:**
- **Modular** — mix and match stages (Lorenzo, Quantizer, RLE, RZE, Bitshuffle, …)
- **High throughput** — parallel level execution, persistent scratch, CUDA Graph support
- **Memory-efficient** — MINIMAL and PREALLOCATE strategies; buffer coloring to alias non-overlapping allocations
- **File format** — FZM format with CRC32 checksums and full stage config serialization

---

### Requirements

| Requirement | Minimum | Notes |
|---|---|---|
| CUDA Toolkit | 11.2+ | Stream-ordered allocator required |
| Host Compiler | GCC 7+ or Clang 5+ | Upper bound set by CUDA version — see [NVIDIA release notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/); NVHPC 23.11 tested in CI |
| C++ Standard | C++17 | |
| CMake | 3.24+ | |
| Host byte order | Little-endian | |

**Note:** using a vGPU will result in the CUDA mempool creation to fail, resulting in an automatic fallback allocation using `cudaMalloc`. This will work correctly but without the performance benefits of the stream-ordered allocator. For perfomance critical workloads avoid vGPU setups. The lack of stream-ordered allocator support also prevents CUDA Graph capture on vGPUs so this feature is unavailable in those environments.

---

## Quick Start

### Building from Source

For full build options (presets, examples/tests, install), see
the \ref building_from_source "Building from Source" page.

```bash
git clone https://github.com/szcompressor/FZGPUModules.git
git submodule update --init --recursive
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### C++ API Usage

```cpp
#include "fzgpumodules.h"

// 1. Build a pipeline
fz::Pipeline pipeline(input_bytes);

auto* lrz = pipeline.addStage<fz::LorenzoQuantStage<float, uint16_t>>(
    fz::LorenzoQuantStage<float, uint16_t>::Config{1e-4f});
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
// d_output is pool-owned by default; do not cudaFree it.
// Call pipeline.setPoolManagedDecompOutput(false) for caller-owned output.
```

See `examples/` for more usage patterns including multi-branch pipelines, CUDA Graph
capture, and the low-level DAG API.

---

### Available Stages {#mainpage_stages}

For detailed per-stage documentation — constraints, behavioral rules, and extended
usage notes — see the \ref stages_overview "Stage Reference".


| Stage                              | Header                                             | Description                                    |
| ---------------------------------- | -------------------------------------------------- | ---------------------------------------------- |
| `LorenzoQuantStage<TInput, TCode>` | `modules/fused/lorenzo_quant/lorenzo_quant.h`      | Fused float predictor + quantizer (lossy)      |
| `LorenzoStage<T>`                  | `modules/predictors/lorenzo/lorenzo_stage.h`       | Plain integer Lorenzo predictor (lossless)     |
| `QuantizerStage<TInput, TCode>`    | `modules/quantizers/quantizer/quantizer.h`         | Direct-value quantizer (ABS/REL/NOA)           |
| `RLEStage<T>`                      | `modules/coders/rle/rle.h`                         | Run-length encoding                            |
| `DifferenceStage<T, TOut>`         | `modules/predictors/diff/diff.h`                   | First-order difference / cumulative-sum coding |
| `BitshuffleStage`                  | `modules/shufflers/bitshuffle/bitshuffle_stage.h`  | Bit-matrix transpose                           |
| `RZEStage`                         | `modules/coders/rze/rze_stage.h`                   | Recursive zero-byte elimination                |
| `ZigzagStage<TIn, TOut>`           | `modules/transforms/zigzag/zigzag_stage.h`         | Zigzag encode/decode                           |
| `NegabinaryStage<TIn, TOut>`       | `modules/transforms/negabinary/negabinary_stage.h` | Negabinary encode/decode                       |
| `BitpackStage<T>`                  | `modules/coders/bitpack/bitpack_stage.h`           | Pack/unpack power-of-two value streams         |

### Memory Strategies

| Strategy      | Description                                                                                                               |
| ------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `MINIMAL`     | Allocate on demand, free at last consumer. Lowest peak GPU memory.                                                        |
| `PREALLOCATE` | Allocate everything at `finalize()`. Required for CUDA Graph capture. Enables buffer coloring for efficient buffer reuse. |


---

### Caller-Allocated Output

If you want full memory control, use the caller-allocated overloads. This mirrors
nvcomp-style APIs: you pre-allocate an output buffer and pass its capacity; the
API returns the actual size.

```cpp
// After finalize()
size_t comp_capacity = pipeline.getMaxCompressedSize(input_bytes);
void* d_comp_user = nullptr;
cudaMalloc(&d_comp_user, comp_capacity);

size_t comp_size = 0;
pipeline.compress(d_input, input_bytes,
                  d_comp_user, comp_capacity,
                  &comp_size, stream);
```

For decompression, size the output from the original input or from the FZM header:
```cpp
auto header = fz::Pipeline::readHeader("output.fzm");
size_t decomp_capacity = header.core.uncompressed_size;

void* d_decomp_user = nullptr;
cudaMalloc(&d_decomp_user, decomp_capacity);

size_t decomp_size = 0;
pipeline.decompress(d_comp_user, comp_size,
                    d_decomp_user, decomp_capacity,
                    &decomp_size, stream);
```

See `examples/ownership_example.cpp` for a minimal end-to-end example.

---

### CUDA Graph Support

For throughput-critical workloads, enable CUDA Graph capture to eliminate
CPU-side kernel launch overhead on repeated compress calls:

```cpp
fz::Pipeline pipeline(input_bytes, fz::MemoryStrategy::PREALLOCATE, 2.0f);
// ... addStage, connect ...
pipeline.enableGraphMode(true);
pipeline.finalize();
pipeline.warmup(stream);      // JIT-compiles all kernels once
pipeline.captureGraph(stream);

// subsequent compress() calls replay the captured graph
pipeline.compress(d_input, input_bytes, &d_compressed, &compressed_sz, stream);
```

Call `compress()` only after `captureGraph()`; use the same stream for capture and replay.

---

### Compressor Config File 

For complex pipelines, you can also load the stage graph from a TOML config file:

```bash
fzgmod-cli -z -i data.f32 -c examples/presets/pfpl.toml -o compressed.fzm --report
```

You can also use the `Pipeline::loadFromConfig()` API to load a config file from C++. The config schema supports arbitrary DAGs.

See `examples/presets/` for reference and pre-built pipeline configurations and the \ref config_file_overview "Config File Reference" for the full config schema.

---

### File I/O

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
checksums. See the \ref fzm_format "FZM File Format" page for the full specification.

---

### Thread Safety

Each `Pipeline` must be used from a single host thread. There is no internal locking.

**Safe** — run one independent pipeline per thread:

```cpp
std::thread t1([&] {
    fz::Pipeline p1(input_size);
    // build, finalize, compress, decompress ...
});
std::thread t2([&] {
    fz::Pipeline p2(input_size);
    // build, finalize, compress, decompress ...
});
t1.join(); t2.join();
```

**Not safe** — two threads sharing one pipeline:

```cpp
fz::Pipeline shared;
std::thread t1([&] { shared.compress(...); });  // data race
std::thread t2([&] { shared.compress(...); });  // data race
```

The library has no global mutable state. The `FZ_LOG` logger singleton is set once
at startup; do not change log level or callback while pipelines are running on other threads.

---

## Citation

If you reference this work, please cite:

Note: this paper describes the 1.0 release of the library; the 2.0 API and
documentation may differ.

> **[DRBSD-11]** FZModules: A Heterogeneous Computing Framework for Customizable Scientific Data Compression Pipelines

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
