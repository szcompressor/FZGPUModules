# FZGPUModules {#mainpage}

GPU-accelerated modular lossy compression pipeline for scientific floating-point data.

## Overview

FZGPUModules is a CUDA library for building composable, high-throughput compression
pipelines. Each pipeline is a directed acyclic graph (DAG) of stages — predictors,
quantizers, encoders, and transforms — connected and executed on the GPU with
stream-ordered memory management.

Key properties:
- **Modular** — mix and match stages (Lorenzo, Quantizer, RLE, RZE, Bitshuffle, …)
- **High throughput** — parallel level execution, persistent scratch, CUDA Graph support
- **Memory-efficient** — MINIMAL and PREALLOCATE strategies; buffer coloring to alias non-overlapping allocations
- **File format** — FZM v3.1 with CRC32 checksums and full stage config serialization

## Quick Start

```cpp
#include "fzgpumodules.h"

// 1. Build a pipeline
fz::Pipeline pipeline(fz::MemoryPoolConfig(input_bytes));

auto* lrz = pipeline.addStage<fz::LorenzoStage<float, uint16_t>>(
    fz::LorenzoStage<float, uint16_t>::Config{1e-4f});
auto* rle = pipeline.addStage<fz::RLEStage<uint16_t>>();

pipeline.connect(lrz, "codes", rle);
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
```

## Key Classes

| Class | Header | Description |
|---|---|---|
| `fz::Pipeline` | `pipeline/compressor.h` | High-level builder and executor |
| `fz::Stage` | `stage/stage.h` | Base class for all compression stages |
| `fz::CompressionDAG` | `pipeline/dag.h` | Low-level DAG wiring and execution |
| `fz::MemoryPool` | `mem/mempool.h` | Stream-ordered CUDA memory pool |
| `fz::PipelinePerfResult` | `pipeline/perf.h` | Per-stage profiling results |

## Available Stages

| Stage | Header | Description |
|---|---|---|
| `LorenzoStage<TInput, TCode>` | `predictors/lorenzo/lorenzo.h` | 1-D/2-D/3-D Lorenzo predictor |
| `QuantizerStage<TInput, TCode>` | `predictors/quantizer/quantizer.h` | Direct-value quantizer (ABS/REL/NOA) |
| `RLEStage<T>` | `encoders/RLE/rle.h` | Run-length encoding |
| `DifferenceStage<T, TOut>` | `encoders/diff/diff.h` | First-order difference coding |
| `BitshuffleStage` | `transforms/bitshuffle/bitshuffle_stage.h` | GPU bit-matrix transpose |
| `RZEStage` | `transforms/rze/rze_stage.h` | Recursive zero-byte elimination |
| `ZigzagStage<TIn, TOut>` | `transforms/zigzag/zigzag_stage.h` | Zigzag encode/decode |
| `NegabinaryStage<TIn, TOut>` | `transforms/negabinary/negabinary_stage.h` | Negabinary encode/decode |

## Memory Strategies

- **`MINIMAL`** — allocate on demand, free at last consumer. Lowest peak GPU memory.
- **`PREALLOCATE`** — allocate everything at `finalize()`. Required for CUDA Graph capture.
  Enables buffer coloring to alias non-overlapping allocations.

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

## Requirements

- CUDA Toolkit 11.2+ (stream-ordered allocator)
- C++17
- CMake 3.24+
