# Acknowledgements {#acknowledgements}

FZGPUModules builds on algorithms and GPU kernels from several open-source scientific
compression projects. This page documents what we took from each, how we adapted it,
and the associated licenses and citations.

For full verbatim copyright notices required by BSD-3-Clause binary redistribution,
see [`THIRD_PARTY.md`](../THIRD_PARTY.md) at the repository root.

---

## Summary

| Project | License | Relationship | Stages |
|---|---|---|---|
| [LC framework](#lc-framework) | BSD-3-Clause | Direct port / algorithm-faithful reimpl | `RZEStage`, `RREStage`, `BitshuffleStage`, `DifferenceStage`, `QuantizerStage` |
| [cuSZ / PHF](#cusz--phf) | BSD-3-Clause | Algorithm follow / vendored PHF headers | `LorenzoQuantStage`, `HuffmanStage` |
| [FZ-GPU](#fz-gpu) | BSD-3-Clause | Direct port of fused kernels | `BitplaneRLEStage` |
| [cuSZ-Hi](#cusz-hi) | BSD-3-Clause | Adapted spline kernels | `GInterpStage` |
| [cuSZp / cuSZp2 / cuSZp3](#cuszp--cuszp2--cuszp3) | BSD-3-Clause | Direct kernel port (`AdaptiveBitpackStage`, `TiledLorenzoStage`) + algorithmic reimpl (`LorenzoStage` block, `QuantizerStage` linear) | `AdaptiveBitpackStage`, `TiledLorenzoStage` |
| [MANS](#mans) | BSD-3-Clause | Direct port of kernels | `ADMStage` |
| [dietGPU](#dietgpu) | MIT | Vendored headers | `ANSStage` |

---

## LC Framework

**Repository:** https://github.com/burtscher/LC-framework  
**License:** BSD-3-Clause  
**Authors:** Noushin Azami, Alex Fallin, Brandon Burtchell, Andrew Rodriguez, Benila Jerald,
Yiqian Liu, Anju Mongandampulath Akathoott, and Martin Burtscher (Texas State University)  
**Funding:** U.S. Department of Energy, Office of Science, ASCR, contract DE-SC0022223

**Stages:**

- **`RREStage` + `RZEStage`** (`modules/coders/{rre,rze}/`) — GPU kernels are a faithful port of
  `d_RRE.h`, `d_RZE.h`, `d_repetition_elimination.h`, `d_zero_elimination.h`, and `prefix_sum.h`
  (the LC `RRE` and `RZE` lossless components used by cuSZ-Hi's LC pipelines), vendored together in
  `modules/coders/lc_common/lc_chunk_components.cuh`. Both support LC word sizes 1/2/4/8.
- **`BitshuffleStage`** (`modules/shufflers/bitshuffle/`) — the 4- and 8-byte butterfly
  shuffle kernels are adapted directly from `d_BIT_4` / `d_BIT_8`. The 1- and 2-byte paths
  use a standard `__ballot_sync` approach and are not LC-derived.
- **`DifferenceStage`** (`modules/predictors/diff/`) — independently written CUDA kernel
  following the `d_DIFFNB` algorithm described in the LC/PFPL framework.
- **`QuantizerStage`** (`modules/quantizers/quantizer/`) — independently written CUDA kernel
  following the LC/PFPL quantization scheme including ABS/NOA/REL error-bound modes, outlier
  handling, and log-space REL encoding.

---

## cuSZ / PHF

**Repository:** https://github.com/szcompressor/cuSZ  
**License:** BSD-3-Clause  
**Authors/Affiliations:** cuSZ team — UChicago Argonne LLC, Washington State University,
Indiana University, University of Kentucky, Oakland University (see copyright notices in
`THIRD_PARTY.md` for year-by-year breakdown)

**Stages:**

- **`LorenzoQuantStage`** (`modules/fused/lorenzo_quant/`) — GPU kernels and the fused
  predictor+quantizer design follow the cuSZ Lorenzo implementation (`lrz_c.cuhip.inl`,
  `lrz_x.cuhip.inl`).
- **`HuffmanStage`** (`modules/coders/huffman/`) — the PHF source files (`hf.h`,
  `hf_bk*.cc`, `hf_buf.cc`, `hf_canon.cc`, `hf_hl.cc`, `hf_kernels.cu`, `hf_impl.hh`)
  are vendored copies adapted from `origin/v1.1.0_dev` of the PHF codec in the cuSZ
  repository, with modifications documented at the top of each file.

**Citation:**

Jiannan Tian, Sheng Di, Xiaodong Yu, Cody Rivera, Kai Zhao, Shixun Wu, Algis Averbuch,
Jon Calhoun, Dingwen Tao, Franck Cappello.
*pSZ/cuSZ: A GPU-Based Error-Bounded Lossy Compressor for Scientific Data.*
SC '21. https://doi.org/10.1145/3458817.3476173

---

## FZ-GPU

**Repository:** https://github.com/szcompressor/cuSZ (vendored under `modules/codec/fzg/`)  
**License:** BSD-3-Clause (same as cuSZ)  
**Original authors:** Boyuan Zhang (kernel), Jiannan Tian (refactor)

**Stages:**

- **`BitplaneRLEStage`** (`modules/fused/bitplane_rle/`) — the fused bitplane-transpose +
  zero-byte run-length encode/decode kernels (`bitplane_rle_encode.inl`,
  `bitplane_rle_decode.inl`) are adapted from `KERNEL_CUHIP_fz_fused_encode` /
  `KERNEL_CUHIP_fz_fused_decode` as vendored in `origin/v1.1.0_dev` of cuSZ.

  **Changes from original:** `namespace fzgpu` → `namespace fz::bitplane_rle`; `err.hh` /
  `CHECK_GPU` stripped; `fzgpu::Buf` allocation replaced by FZGPUModules `MemoryPool`; the
  128-byte `fzg_header` is reproduced as `ArchiveHeader`. Kernel bodies preserved verbatim.
  Host-side wrapper and memory-pool integration are FZGPUModules code.

**Citation:**

Boyuan Zhang, Jiannan Tian, Sheng Di, Xiaodong Yu, Yunhe Feng, Xin Liang, Dingwen Tao,
Franck Cappello.
*FZ-GPU: A Fast and High-Ratio Lossy Compressor for Scientific Computing Applications on GPUs.*
HPDC '23. https://doi.org/10.1145/3588195.3592994

---

## cuSZ-Hi

**Repository:** https://github.com/shixun404/cuSZ-Hi  
**License:** BSD-3-Clause  
**Authors/Affiliations:** Indiana University and UChicago Argonne LLC

**Stages:**

- **`GInterpStage`** (`modules/fused/ginterp/`) — the multi-level spline interpolation
  kernels are adapted from `spline3.cu` and `spline3_md.inl`.

  **Changes from original:** `namespace cusz` → `namespace fz::ginterp`; `err.hh` /
  `timer.hh` includes stripped; `pszmem_cxx<T>` buffer abstraction replaced by raw device
  pointers via FZGPUModules `MemoryPool`; `CompactDram` outlier triplet replaced by separate
  `outlier_vals` / `outlier_idxs` / `outlier_count` output ports; minimal
  `cusz_type_subset.h` reproduces only `INTERPOLATION_PARAMS` and `u4` from upstream
  `cusz/type.h`. Host-side wrapper, memory-pool integration, radius auto-tune, and all five
  auto-tune modes are FZGPUModules code.

  **Bug fix patched locally:** `pa_spline_infprecis_data`'s SPLINE_DIM==2 level==0 atomic
  offset corrected from `errors + 15 + BIY` (upstream) to `errors + 16 + BIY` to avoid a
  slot collision with the level==1 BIY=4 write. Full analysis is in
  [docs/stages/ginterp.md](stages/ginterp.md).

  An unmodified reference copy of the upstream kernels lives in
  `memory/references/spline_cuszhi/` for cross-checking.

**Citation:**

Shixun Wu, Jinwen Pan, Jinyang Liu, Jiannan Tian, Ziwei Qiu, Jiajun Huang, Kai Zhao,
Xin Liang, Sheng Di, Zizhong Chen, Franck Cappello.
*Boosting Scientific Error-Bounded Lossy Compression through Optimized Synergistic
Lossy-Lossless Orchestration* (cuSZ-Hi).
SC '25. https://doi.org/10.1145/3712285.3759798

---

## cuSZp / cuSZp2 / cuSZp3

**Repository:** https://github.com/szcompressor/cuSZp  
**License:** BSD-3-Clause (verbatim copyright reproduced in `THIRD_PARTY.md`)  
**Authors/Affiliations:** Yafan Huang, Sheng Di, Guanpeng Li, Franck Cappello
(Argonne National Laboratory / University of Iowa)

**Mixed relationship:** two stages contain **direct ports of cuSZp kernel source**
(`AdaptiveBitpackStage`, `TiledLorenzoStage`); two are **independent
reimplementations** with no source copied (`LorenzoStage` block mode,
`QuantizerStage` linear mode).

**Stages:**

- **`AdaptiveBitpackStage`** (`modules/coders/adaptive_bitpack/`) — **direct port**
  of the cuSZp fixed-length (per-block fixed-rate bit-plane) encode/decode kernel
  logic from cuSZp (SC'23), plus the plain vs. outlier selection mode from cuSZp2
  (SC'24). Re-expressed one-thread-per-block with a byte-granular layout and an
  ordinary CUB `DeviceScan` for per-block offsets where cuSZp fuses a decoupled
  look-back scan (that fusion is not reproduced); `MemoryPool` integration and FZM
  scaffolding are FZGPUModules code.
- **`TiledLorenzoStage`** (`modules/predictors/tiled_lorenzo/`) — **direct port** of
  the cuSZp3 / VGC (SC'25) dimension-aware (2-D/3-D tiled separable) delta kernel
  logic, re-expressed as a standalone integer predictor with a tile-major output
  reshape; the tile-major decomposition, FZM header, and `MemoryPool` integration
  are FZGPUModules code.
- **`LorenzoStage::setBlockSize`** — **independent reimplementation** of the
  block-local 1-D delta from cuSZp (SC'23).
- **`QuantizerStage` linear mode** — **independent reimplementation** of
  `q = round(x / 2·eb)` from cuSZp (SC'23).

cuSZp3's **memory-efficient compression** and **selective decompression** features are not
ported; they don't map cleanly onto the staged pipeline model.

**Citations:**

Yafan Huang, Sheng Di, Xiaodong Yu, Guanpeng Li, Franck Cappello.
*cuSZp: An Ultra-fast GPU Error-bounded Lossy Compression Framework with Optimized End-to-End Performance.*
SC '23. https://doi.org/10.1145/3581784.3607048

Yafan Huang, Sheng Di, Guanpeng Li, Franck Cappello.
*cuSZp2: A GPU Lossy Compressor with Extreme Throughput and Optimized Compression Ratio.*
SC '24.

Yafan Huang, Sheng Di, Guanpeng Li, Franck Cappello.
*GPU Lossy Compression for HPC Can Be Versatile and Ultra-Fast* (cuSZp3 / VGC).
SC '25. https://doi.org/10.1145/3712285.3759817

---

## MANS

**Repository:** https://github.com/hpdps-group/MANS  
**License:** BSD-3-Clause  
**Authors:** Wenjing Huang, Jinwu Yang, JingKai Huang, Haoquan Long;
Advisors: Dingwen Tao, Guangming Tan

**Stages:**

- **`ADMStage`** (`modules/transforms/adm/`) — GPU kernels (`mapping_uint16.cu`,
  `mapping_uint32.cu`) are a direct port of `nv/adm/mapping_uint16.cu` and
  `nv/adm/mapping_uint32.cu` from MANS. Kernel logic is unchanged.

  **Changes from original:** unused `MansParams` parameter removed; per-call
  `cudaMalloc`/`cudaFree` replaced by pool-allocated `AdmScratch`; `check_cuda()` replaced
  by `FZ_CUDA_CHECK`; namespace changed from `mans::nv::adm` to `fz::adm`; kernels renamed
  with `_u16`/`_u32` suffix to avoid TU-level naming conflicts; inline comments translated
  to English.

---

## dietGPU

**Repository:** https://github.com/facebookresearch/dietgpu  
**License:** MIT  
**Authors:** Meta Platforms, Inc. and affiliates

**Stages:**

- **`ANSStage`** (`modules/coders/ans/`) — the rANS kernel headers
  (`GpuANSCodec.h`, `GpuANSEncode.h`, `GpuANSDecode.h`, `GpuANSStatistics.h`,
  `BatchPrefixSum.h`, and `utils/`) are vendored copies placed under
  `modules/coders/ans/dietgpu/`.

  **Changes from original:** `histogramBatch`/`histogramSingle` functions in
  `GpuANSStatistics.h` removed and replaced by the shared
  `fz::module::GPU_histogram_generic<uint8_t>` utility; namespace `multibyte_ans` adapted
  to `fz::ans`. All other kernel logic is unchanged from the original.
