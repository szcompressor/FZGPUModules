# FZGPUModules

[![CI](https://github.com/szcompressor/FZGPUModules/actions/workflows/ci.yml/badge.svg)](https://github.com/szcompressor/FZGPUModules/actions/workflows/ci.yml)
[![Docs](https://github.com/szcompressor/FZGPUModules/actions/workflows/docs.yml/badge.svg)](https://github.com/szcompressor/FZGPUModules/actions/workflows/docs.yml)
[![License](https://img.shields.io/badge/license-BSD%203--Clause-blue)](LICENSE)

GPU-accelerated graph composable compression pipeline builder for analytical workflows.

The CUDA backend is supported. The AMD HIP/ROCm backend is currently
**experimental**: most stages build and run, but backend coverage, CI, and
installed-package compatibility are still being completed, and unsupported stages
are rejected explicitly.

## Overview

FZGPUModules is a CUDA library for building composable, high-throughput compression pipelines. Each pipeline is a directed acyclic graph (DAG) of stages - coders, predictors, quantizers, shufflers, transforms, fused stages, and external stages - connected and executed entirely on the GPU with stream-ordered memory management.



**Key properties:**
- **Modular** — mix and match stages (Lorenzo, G-Interp, Quantizer, ADM, RLE, RZE, Bitshuffle, Huffman, ANS, …)
- **Pipeline Specialization** — at `finalize()` the library transparently replaces staged execution with an optimized specialized implementation (kernel fusion + runtime optimizations), byte-identical to staged on both compress and decompress. Opt in with `setSpecializationPolicy(Auto)` / `FZ_SPECIALIZE=auto`. See [docs/pipeline_specialization.md](docs/pipeline_specialization.md).
- **High throughput** — parallel level execution, persistent scratch, stream-ordered allocation (plus optional CUDA Graph capture)
- **Memory-efficient** — MINIMAL and PREALLOCATE strategies; buffer coloring to alias non-overlapping allocations
- **Self-describing files** — FZM format embeds full stage config with CRC32 checksums

---

## Documentation for LLM users

- **[AGENTS.md](AGENTS.md)** — repo guide for AI coding agents: pipeline model, stage catalog, build/test recipes, ownership rules, and key gotchas.

---

## Requirements

| Requirement | Minimum | Notes |
|---|---|---|
| CUDA Toolkit | 11.2+ | |
| Host Compiler | GCC 7+ or Clang 5+ | Upper bound set by CUDA version — see [NVIDIA release notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/); NVHPC 23.11 tested in CI |
| C++ Standard | C++17 | |
| CMake | 3.24+ | |
| Host byte order | Little-endian | |

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
| `TiledLorenzoStage<T>` | Dimension-aware (tiled separable) Lorenzo predictor (lossless, 2D/3D, cuSZp3 delta) |
| `GInterpStage<TInput, TCode>` | Multi-level spline interpolation predictor + quantizer (lossy, 3D, cuSZ-Hi port) |
| `AdaptiveLorenzoStage<T>` | Per-tile adaptive Lorenzo — picks the cheapest of LZ1/LZ2 x centering by exact encoded cost (lossless, FSZ) |
| `QuantizerStage<TInput, TCode>` | Direct-value quantizer (ABS/REL/NOA error modes) |
| `DifferenceStage<T, TOut>` | First-order difference / cumulative-sum coding |
| `LogTransformStage<TInput>` | Log transform — turns a point-wise relative bound into an absolute one for a downstream ABS quantizer (Liang et al., CLUSTER'18) |
| `ADMStage` | Adaptive data mapping — remaps uint16/uint32 streams to a compact 8-bit symbol domain (MANS port) |
| `Cdf97Stage<TInput>` | CDF 9/7 biorthogonal wavelet transform — SPERR's DWT front-half (lifting constants/boundary handling ported; GPU kernels FZGPUModules' own) |
| `Speck2DStage` | GPU-parallel "wavefront" SPECK-like bit-plane coder, 2-D — codes the same information as SPERR's `SPECK2D_INT`, independent decode-parallel design |
| `OutlierCorrectStage<Reconstructor>` / `Cdf97OutlierCorrectStage` | Sparse exact outlier correction — turns a coefficient-domain quantization bound into a GUARANTEED reconstructed-domain pointwise bound (SPERR `Outlier_Coder` mechanism, transform-agnostic via the `Reconstructor` policy) |
| `RLEStage<T>` | Run-length encoding |
| `BitshuffleStage` | GPU bit-matrix transpose |
| `TUPLStage` | AoS <-> SoA tuple transpose — regroups interleaved `dim`-field records by field (LC `TUPLk` port) |
| `RZEStage` | LC zero-word bitmap reducer with recursive bitmap compression |
| `RREStage` | LC repeated-word bitmap reducer with recursive bitmap compression |
| `RAZEStage` | Applies RZE to an automatically selected number of upper bits per chunk; stores lower bits verbatim (LC port) |
| `RAREStage` | Applies RRE to an automatically selected number of upper bits per chunk; stores lower bits verbatim (LC port) |
| `CLOGStage` | Per-subchunk adaptive bit-width coding, 32 subchunks per chunk (LC port) |
| `HCLOGStage` | CLOG plus a per-subchunk TCMS/zigzag reinterpretation fallback (LC port) |
| `GPULZStage` | GPU LZSS dictionary coder derived from GPULZ (substantially rewritten kernels); optional split mode emits literals/lengths/offsets/meta as four ports for a GPU-ZSTD-style chain |
| `ZigzagStage<TIn, TOut>` | Zigzag encode/decode |
| `NegabinaryStage<TIn, TOut>` | Negabinary encode/decode |
| `BitpackStage<T>` | Pack/unpack power-of-two value streams |
| `AdaptiveBitpackStage<T>` | Per-block adaptive fixed-rate bit-plane coding (cuSZp/cuSZp2 port) |
| `HuffmanStage<T>` | GPU Huffman entropy coding (cuSZ port) |
| `ANSStage` | GPU rANS entropy coding (dietGPU port) |
| `BitplaneRZEStage` | Fused bitplane transpose + zero-group RZE lossless encoder (FZ-GPU port) |
| `SZxStage<T>` | SZx constant-block classification + fixed-length residual whole compressor |
| `MergeStage` | Concatenate N producer ports into one buffer / split back (structural) |
| `ROIBinSplitStage<T>` | Split detector fields into full-resolution ROI, binned background, and peak-table branches |

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

## CUDA Graph Capture

For throughput-critical workloads, capture the forward compression pass into a
CUDA Graph. The correct sequence is: enable graph mode, finalize, warm up, then
capture. Only after capture can you call `compress()` to replay the graph.

```cpp
fz::Pipeline pipeline(input_bytes, fz::MemoryStrategy::PREALLOCATE, 2.0f);
// ... addStage, connect ...
pipeline.enableGraphMode(true);
pipeline.finalize();
pipeline.warmup(stream);      // JIT-compile kernels
pipeline.captureGraph(stream);

// Graph replay
pipeline.compress(d_input, input_bytes, &d_compressed, &compressed_sz, stream);
```

Use the same stream for capture and replay.

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

## Acknowledgements

FZGPUModules incorporates algorithms and GPU kernels ported or reimplemented from the following projects. Most are BSD-3-Clause licensed; the exceptions are dietGPU (MIT), SPERR (Apache License 2.0), and GPULZ and AIZ_VLDB26 (no license declared upstream).

| Project | Stages |
|---|---|
| [LC framework](https://github.com/burtscher/LC-framework) — Burtscher et al., Texas State University | `RZEStage`, `RREStage`, `RAZEStage`, `RAREStage`, `CLOGStage`, `HCLOGStage`, `TUPLStage`, `BitshuffleStage`, `DifferenceStage`, `QuantizerStage` |
| [cuSZ](https://github.com/szcompressor/cuSZ) — Argonne NL, Indiana U, et al. | `LorenzoQuantStage`, `HuffmanStage` |
| [FZ-GPU](https://github.com/szcompressor/cuSZ) — Zhang, Tian et al. (via cuSZ repo) | `BitplaneRZEStage` |
| [cuSZ-Hi](https://github.com/shixun404/cuSZ-Hi) — Indiana U, Argonne NL | `GInterpStage` |
| [cuSZp / cuSZp2 / cuSZp3](https://github.com/szcompressor/cuSZp) — Huang, Di et al., Argonne NL | `AdaptiveBitpackStage`, `TiledLorenzoStage` |
| [MANS](https://github.com/hpdps-group/MANS) — Huang, Yang et al. | `ADMStage` |
| [SPERR](https://github.com/NCAR/SPERR) — Li, Lindstrom, Clyne, NCAR (Apache License 2.0) | `Cdf97Stage` (direct port: lifting constants, boundary handling); `Speck2DStage`, `OutlierCorrectStage`/`Cdf97OutlierCorrectStage` (algorithmic attribution only, no code ported — see THIRD_PARTY.md) |
| [dietGPU](https://github.com/facebookresearch/dietgpu) — Meta Platforms (MIT) | `ANSStage` |
| [GPULZ](https://github.com/hpdps-group/ICS23-GPULZ) — Zhang, Tian, Di et al. (ICS '23; *no license declared upstream*) | `GPULZStage` |
| [AIZ_VLDB26](https://github.com/boyuanzhang62/AIZ_VLDB26) — Boyuan Zhang (*no license declared upstream*) | `GPULZStage` all-zero-chunk fast path |
| [FSZ](https://github.com/JiajunHuang1999/FSZ) — Jiajun Huang, SC '26 (arXiv:2607.15413) — *algorithmic attribution only; written from the paper, before FSZ 1.0.0 was released* | `AdaptiveLorenzoStage`, `LorenzoStage` centering / order-2, `LorenzoQuantStage` centering |
| Liang, Di, Tao, Chen, Cappello — IEEE CLUSTER 2018 — *algorithmic attribution only* | `LogTransformStage` |
| [ROIBIN-SZ](https://arxiv.org/abs/2206.11297) — Underwood, Yoon, Gok, Di, Cappello — *independent GPU/DAG implementation of the published design* | `ROIBinSplitStage` |
| [SZp / fZ-light](https://github.com/szcompressor/SZp) — Huang, Di et al. (MIT) | `szp_composed.toml` (`Quantizer → Lorenzo → AdaptiveBitpack`); `experimental/reference_compressors/szp` (quarantined reference impl) |
| [SZx](https://github.com/szcompressor/SZx) — Yu, Di et al. — *algorithmic attribution only; no source copied* | `SZxStage` |

For per-stage attribution details, copyright notices, relationship types (direct port, algorithmic reimplementation, or vendored), and paper citations, see [`docs/acknowledgements.md`](docs/acknowledgements.md) and [`THIRD_PARTY.md`](THIRD_PARTY.md).

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
