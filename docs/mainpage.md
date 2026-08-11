# FZGPUModules {#mainpage}

GPU-accelerated graph composable compression pipeline builder for analytical workflows.

## Overview

FZGPUModules is a CUDA library for building composable, high-throughput compression
pipelines. Each pipeline is a directed acyclic graph (DAG) of stages - coders,
predictors, quantizers, shufflers, transforms, fused stages, and external stages -
connected and executed entirely on the GPU with stream-ordered memory management.

**Key properties:**
- **Modular** — mix and match stages (Lorenzo, G-Interp, Quantizer, ADM, RLE, RZE, RRE, Bitshuffle, TUPL, Huffman, ANS, …)
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
| \ref stage_lorenzo_quant "LorenzoQuantStage<TInput, TCode>" | `modules/fused/lorenzo_quant/lorenzo_quant.h`      | Fused float predictor + quantizer (lossy)      |
| \ref stage_lorenzo "LorenzoStage<T>"                  | `modules/predictors/lorenzo/lorenzo_stage.h`       | Plain integer Lorenzo predictor (lossless)     |
| \ref stage_tiled_lorenzo "TiledLorenzoStage<T>"       | `modules/predictors/tiled_lorenzo/tiled_lorenzo_stage.h` | Dimension-aware (tiled separable) Lorenzo predictor (lossless, 2D/3D, cuSZp3 delta) |
| \ref stage_ginterp "GInterpStage<TInput, TCode>"      | `modules/fused/ginterp/ginterp_stage.h`       | Multi-level spline interpolation predictor + quantizer (lossy, 3D, cuSZ-Hi port) |
| \ref stage_quantizer "QuantizerStage<TInput, TCode>"    | `modules/quantizers/quantizer/quantizer.h`         | Direct-value quantizer (ABS/REL/NOA)           |
| \ref stage_rle "RLEStage<T>"                      | `modules/coders/rle/rle.h`                         | Run-length encoding                            |
| \ref stage_diff "DifferenceStage<T, TOut>"         | `modules/predictors/diff/diff.h`                   | First-order difference / cumulative-sum coding |
| \ref stage_adm "ADMStage"                         | `modules/transforms/adm/adm_stage.h`               | Adaptive data mapping — uint16/32 → 8-bit symbol domain (MANS port) |
| \ref stage_bitshuffle "BitshuffleStage"                  | `modules/shufflers/bitshuffle/bitshuffle_stage.h`  | Bit-matrix transpose                           |
| \ref stage_tupl "TUPLStage"                       | `modules/shufflers/tupl/tupl_stage.h`              | Tuple deinterleave / AoS-to-SoA transpose (LC component) |
| \ref stage_rze "RZEStage"                         | `modules/coders/rze/rze_stage.h`                   | Recursive zero-byte elimination                |
| \ref stage_rre "RREStage"                         | `modules/coders/rre/rre_stage.h`                   | Repetition-reduction encoding (LC component)   |
| \ref stage_rare "RAREStage"                       | `modules/coders/rare/rare_stage.h`                 | Repetition-adaptive reduction encoding (LC component, auto-k RRE) |
| \ref stage_raze "RAZEStage"                       | `modules/coders/raze/raze_stage.h`                 | Zero-adaptive reduction encoding (LC component, auto-k RZE) |
| \ref stage_clog "CLOGStage"                       | `modules/coders/clog/clog_stage.h`                 | Compressed-Logarithm adaptive bit-width coding (LC component) |
| \ref stage_hclog "HCLOGStage"                     | `modules/coders/hclog/hclog_stage.h`               | Compressed-Logarithm coding with per-subchunk TCMS fallback (LC component) |
| \ref stage_zigzag "ZigzagStage<TIn, TOut>"           | `modules/transforms/zigzag/zigzag_stage.h`         | Zigzag encode/decode                           |
| \ref stage_negabinary "NegabinaryStage<TIn, TOut>"       | `modules/transforms/negabinary/negabinary_stage.h` | Negabinary encode/decode                       |
| \ref stage_bitpack "BitpackStage<T>"                  | `modules/coders/bitpack/bitpack_stage.h`           | Pack/unpack power-of-two value streams         |
| \ref stage_adaptive_bitpack "AdaptiveBitpackStage<T>" | `modules/coders/adaptive_bitpack/adaptive_bitpack_stage.h` | Per-block adaptive fixed-rate bit-plane coding (cuSZp/cuSZp2 port) |
| \ref stage_huffman "HuffmanStage<T>"                  | `modules/coders/huffman/huffman_stage.h`           | GPU Huffman entropy coding (PHF, cuSZ port)    |
| \ref stage_ans "ANSStage"                         | `modules/coders/ans/ans_stage.h`                   | GPU rANS entropy coding (dietGPU port)         |
| \ref stage_bitplane_rze "BitplaneRZEStage"        | `modules/fused/bitplane_rze/bitplane_rze_stage.h`  | Fused bitplane transpose + zero-group RZE lossless encoder (FZ-GPU port) |
| \ref stage_merge "MergeStage"                     | `modules/structural/merge/merge_stage.h`           | Concatenate N producer ports into one buffer / split back (structural) |
| \ref stage_roibin_split "ROIBinSplitStage"        | `modules/structural/roibin_split/roibin_split_stage.h` | Split a field into ROI boxes + binned background for dual-error-bound branches (structural) |

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

See the \ref performance_tuning "Performance Tuning" page for this and every other
performance lever (memory strategy, stage-level config, build flags) with measured
effect sizes where available.

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

### Decode-only pipelines (no warmup compress)

For streaming decode loops that only ever decompress blobs produced elsewhere, a
decode-only pipeline can decode an in-memory blob with **no** prior `compress()` by
carrying a small metadata header:

```cpp
// Producer (after compress()): store the header alongside the blob.
std::vector<uint8_t> header = producer.serializeHeaderToMemory();   // ~1 KB, no payload

// Consumer (fresh, finalized, same topology, never compress()ed): one call per blob.
slot.decompressFromMemory(header.data(), header.size(),
                          d_blob, blob_size, &d_out, &out_size, stream);
```

The header carries the data-dependent inverse metadata that is not in the raw blob
(Huffman symbol count, quantizer outlier count). See `examples/decode_only_slots.cpp`
and the \ref api_reference "API Reference".

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

## Coding with AI agents

If you're using Claude Code (or another agent-driven IDE assistant), point it at
[`CLAUDE.md`](https://github.com/szcompressor/FZGPUModules/blob/main/CLAUDE.md) at the
repo root for an LLM-tailored overview of the pipeline model, the stage catalog, build/test
recipes, and the ownership rules and gotchas that are easy to get wrong from source alone.

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
