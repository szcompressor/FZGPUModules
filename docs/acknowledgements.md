# Acknowledgements {#acknowledgements}

FZGPUModules builds on algorithms and GPU kernels from several open-source scientific
compression projects. This page documents what we took from each, how we adapted it,
and the associated licenses and citations.

For full verbatim copyright notices required by BSD-3-Clause binary redistribution,
see \ref third_party_notices "Third-party notices".

---

## Summary

| Project | License | Relationship | Stages |
|---|---|---|---|
| \ref ack_lc "LC framework" | BSD-3-Clause | Direct port / algorithm-faithful reimpl | `RZEStage`, `RREStage`, `RAREStage`, `RAZEStage`, `CLOGStage`, `HCLOGStage`, `BitshuffleStage`, `TUPLStage`, `DifferenceStage`, `QuantizerStage` |
| \ref ack_cusz "cuSZ" | BSD-3-Clause | Algorithm follow / vendored Huffman sources | `LorenzoQuantStage`, `HuffmanStage` |
| \ref ack_fzgpu "FZ-GPU" | BSD-3-Clause | Direct port of fused kernels | `BitplaneRZEStage` |
| \ref ack_cusz_hi "cuSZ-Hi" | BSD-3-Clause | Adapted spline kernels | `GInterpStage` |
| \ref ack_cuszp "cuSZp / cuSZp2 / cuSZp3" | BSD-3-Clause | Direct kernel port (`AdaptiveBitpackStage`, `TiledLorenzoStage`) + algorithmic reimpl (`LorenzoStage` block, `QuantizerStage` linear) | `AdaptiveBitpackStage`, `TiledLorenzoStage` |
| \ref ack_mans "MANS" | BSD-3-Clause | Direct port of kernels | `ADMStage` |
| \ref ack_sperr "SPERR" | Apache License 2.0 | Direct port (lifting constants, boundary handling) + algorithmic attribution only (`Speck2DStage`, `OutlierCorrectStage`) | `Cdf97Stage`, `Speck2DStage`, `OutlierCorrectStage`/`Cdf97OutlierCorrectStage` |
| \ref ack_dietgpu "dietGPU" | MIT | Vendored headers | `ANSStage` |
| \ref ack_gpulz "GPULZ" | **None declared upstream** | Substantially rewritten derivative | `GPULZStage` |
| \ref ack_aiz "AIZ_VLDB26" | **None declared upstream** | Adapted optimization | `GPULZStage` all-zero-chunk fast path |
| \ref ack_fsz "FSZ" | BSD-3-Clause | **Algorithmic attribution only; no code used** (written from the paper before FSZ 1.0.0 was released) | `AdaptiveLorenzoStage`, `LorenzoStage` centering / order-2, `LorenzoQuantStage` centering |
| \ref ack_log_transform "Point-wise relative transform" | n/a — paper only | **Algorithmic attribution only; no code used** | `LogTransformStage` |
| \ref ack_roibin "ROIBIN-SZ" | SZ2 BSD-style license; no code copied | **Algorithmic attribution only; independent GPU/DAG implementation** | `ROIBinSplitStage` |
| \ref ack_szp "SZp / fZ-light" | MIT | GPU reimplementation; no source copied | `SZpStage` |
| \ref ack_szx "SZx" | Argonne BSD-style license; no code copied | **Algorithmic attribution only; paper-derived** | `SZxStage` |

---

## LC Framework {#ack_lc}

**Repository:** https://github.com/burtscher/LC-framework  
**License:** BSD-3-Clause  
**Authors:** Noushin Azami, Alex Fallin, Brandon Burtchell, Andrew Rodriguez, Benila Jerald,
Yiqian Liu, Anju Mongandampulath Akathoott, and Martin Burtscher (Texas State University)  
**Funding:** U.S. Department of Energy, Office of Science, ASCR, contract DE-SC0022223

**Stages:**

- **RREStage + RZEStage** (`modules/coders/{rre,rze}/`) — GPU kernels are a faithful port of
  `d_RRE.h`, `d_RZE.h`, `d_repetition_elimination.h`, `d_zero_elimination.h`, and `prefix_sum.h`
  (the LC `RRE` and `RZE` lossless components used by cuSZ-Hi's LC pipelines), vendored together in
  `modules/coders/lc_common/lc_chunk_components.cuh`. Both support LC word sizes 1/2/4/8.
- **RAREStage + RAZEStage** (`modules/coders/{rare,raze}/`) — GPU kernels are a faithful
  port of `d_RARE.h` and `d_RAZE.h`, the auto-k generalizations of `RRE`/`RZE` (one global
  bit-width cut chosen per chunk instead of a binary match/no-match test), sharing a single
  merged `d_PRencode`/`d_PRdecode<T, PartialReduceMode>` template in the same vendored header.
  Both support LC word sizes 1/2/4/8.
- **CLOGStage + HCLOGStage** (`modules/coders/{clog,hclog}/`) — GPU kernels are a faithful
  port of `d_CLOG.h` and `d_HCLOG.h`: each chunk is split into a fixed 32 subchunks, each
  bit-packed to the minimum width needed for its own max value (`T` unsigned only); HCLOG
  additionally tries a per-subchunk TCMS(zigzag) reinterpretation and keeps whichever is
  smaller. Sharing a single merged `d_CLOGencode`/`d_CLOGdecode<T, CLogMode>` template in
  `modules/coders/lc_common/lc_clog_components.cuh`. Both support LC word sizes 1/2/4/8.
- **BitshuffleStage** (`modules/shufflers/bitshuffle/`) — the 4- and 8-byte butterfly
  shuffle kernels are adapted directly from `d_BIT_4` / `d_BIT_8`. The 1- and 2-byte paths
  use a standard `__ballot_sync` approach and are not LC-derived.
- **TUPLStage** (`modules/shufflers/tupl/`) — GPU kernels are a faithful port of
  `d_TUPL` / `d_iTUPL` (LC's `TUPLk` tuple deinterleave / AoS-to-SoA transpose). Upstream
  generates one fixed `(dim, word_size)` instantiation per component over a hardcoded
  16 KB chunk; here `dim`, `word_size`, and `block_size` are independent runtime
  parameters instead.
- **DifferenceStage** (`modules/predictors/diff/`) — independently written CUDA kernel
  following the `d_DIFFNB` algorithm described in the LC/PFPL framework.
- **QuantizerStage** (`modules/quantizers/quantizer/`) — independently written CUDA kernel
  following the LC/PFPL quantization scheme including ABS/NOA/REL error-bound modes, outlier
  handling, and log-space REL encoding.

---

## cuSZ {#ack_cusz}

**Repository:** https://github.com/szcompressor/cuSZ  
**License:** BSD-3-Clause  
**Authors/Affiliations:** cuSZ team — UChicago Argonne LLC, Washington State University,
Indiana University, University of Kentucky, Oakland University (see copyright notices in
`THIRD_PARTY.md` for year-by-year breakdown)

**Stages:**

- **LorenzoQuantStage** (`modules/fused/lorenzo_quant/`) — GPU kernels and the fused
  predictor+quantizer design follow the cuSZ Lorenzo implementation (`lrz_c.cuhip.inl`,
  `lrz_x.cuhip.inl`).
- **HuffmanStage** (`modules/coders/huffman/`) — cuSZ's Huffman source files (`hf.h`,
  `hf_bk*.cc`, `hf_buf.cc`, `hf_canon.cc`, `hf_hl.cc`, `hf_kernels.cu`, `hf_impl.hh`)
  are vendored copies adapted from `origin/v1.1.0_dev` of the cuSZ repository, with
  modifications documented at the top of each file. cuSZ uses `phf` as this
  implementation's internal namespace/type prefix.

**Citation:**

Jiannan Tian, Sheng Di, Xiaodong Yu, Cody Rivera, Kai Zhao, Shixun Wu, Algis Averbuch,
Jon Calhoun, Dingwen Tao, Franck Cappello.
*pSZ/cuSZ: A GPU-Based Error-Bounded Lossy Compressor for Scientific Data.*
SC '21. https://doi.org/10.1145/3458817.3476173

---

## FZ-GPU {#ack_fzgpu}

**Repository:** https://github.com/szcompressor/cuSZ (vendored under `modules/codec/fzg/`)  
**License:** BSD-3-Clause (same as cuSZ)  
**Original authors:** Boyuan Zhang (kernel), Jiannan Tian (refactor)

**Stages:**

- **BitplaneRZEStage** (`modules/fused/bitplane_rze/`) — the fused bitplane-transpose +
  zero-group encode/decode kernels (`bitplane_rze_encode.inl`,
  `bitplane_rze_decode.inl`) are adapted from `KERNEL_CUHIP_fz_fused_encode` /
  `KERNEL_CUHIP_fz_fused_decode` as vendored in `origin/v1.1.0_dev` of cuSZ.

  **Changes from original:** `namespace fzgpu` → `namespace fz::bitplane_rze`; `err.hh` /
  `CHECK_GPU` stripped; `fzgpu::Buf` allocation replaced by FZGPUModules `MemoryPool`; the
  128-byte `fzg_header` is reproduced as `ArchiveHeader`. Kernel bodies preserved verbatim.
  Host-side wrapper and memory-pool integration are FZGPUModules code.

**Citation:**

Boyuan Zhang, Jiannan Tian, Sheng Di, Xiaodong Yu, Yunhe Feng, Xin Liang, Dingwen Tao,
Franck Cappello.
*FZ-GPU: A Fast and High-Ratio Lossy Compressor for Scientific Computing Applications on GPUs.*
HPDC '23. https://doi.org/10.1145/3588195.3592994

---

## cuSZ-Hi {#ack_cusz_hi}

**Repository:** https://github.com/shixun404/cuSZ-Hi  
**License:** BSD-3-Clause  
**Upstream copyright holder:** UChicago Argonne, LLC and Washington State University

**Stages:**

- **GInterpStage** (`modules/fused/ginterp/`) — the multi-level spline interpolation
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
  \ref stage_ginterp "GInterpStage documentation".

**Citation:**

Shixun Wu, Jinwen Pan, Jinyang Liu, Jiannan Tian, Ziwei Qiu, Jiajun Huang, Kai Zhao,
Xin Liang, Sheng Di, Zizhong Chen, Franck Cappello.
*Boosting Scientific Error-Bounded Lossy Compression through Optimized Synergistic
Lossy-Lossless Orchestration* (cuSZ-Hi).
SC '25. https://doi.org/10.1145/3712285.3759798

---

## cuSZp / cuSZp2 / cuSZp3 {#ack_cuszp}

**Repository:** https://github.com/szcompressor/cuSZp  
**License:** BSD-3-Clause (verbatim copyright reproduced in `THIRD_PARTY.md`)  
**Authors/Affiliations:** Yafan Huang, Sheng Di, Guanpeng Li, Franck Cappello
(Argonne National Laboratory / University of Iowa)

**Mixed relationship:** two stages contain **direct ports of cuSZp kernel source**
(`AdaptiveBitpackStage`, `TiledLorenzoStage`); two are **independent
reimplementations** with no source copied (`LorenzoStage` block mode,
`QuantizerStage` linear mode).

**Stages:**

- **AdaptiveBitpackStage** (`modules/coders/adaptive_bitpack/`) — **direct port**
  of the cuSZp fixed-length (per-block fixed-rate bit-plane) encode/decode kernel
  logic from cuSZp (SC'23), plus the plain vs. outlier selection mode from cuSZp2
  (SC'24). Re-expressed one-thread-per-block with a byte-granular layout and an
  ordinary CUB `DeviceScan` for per-block offsets where cuSZp fuses a decoupled
  look-back scan (that fusion is not reproduced); `MemoryPool` integration and FZM
  scaffolding are FZGPUModules code.
- **TiledLorenzoStage** (`modules/predictors/tiled_lorenzo/`) — **direct port** of
  the cuSZp3 / VGC (SC'25) dimension-aware (2-D/3-D tiled separable) delta kernel
  logic, re-expressed as a standalone integer predictor with a tile-major output
  reshape; the tile-major decomposition, FZM header, and `MemoryPool` integration
  are FZGPUModules code.
- **LorenzoStage::setBlockSize** — **independent reimplementation** of the
  block-local 1-D delta from cuSZp (SC'23).
- **QuantizerStage linear mode** — **independent reimplementation** of
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

## MANS {#ack_mans}

**Repository:** https://github.com/hpdps-group/MANS  
**License:** BSD-3-Clause  
**Developers:** Wenjing Huang, Jinwu Yang, and Dingwen Tao (Institute of
Computing Technology, Chinese Academy of Sciences), and colleagues  
**Paper:** Wenjing Huang, Jinwu Yang, Dingwen Tao, et al. "MANS: Efficient and
Portable ANS Encoding for Multi-Byte Integer Data on CPUs and GPUs." SC '25.
(Full author list / BibTeX: see `THIRD_PARTY.md`.)

**Stages:**

- **ADMStage** (`modules/transforms/adm/`) — GPU kernels (`mapping_uint16.cu`,
  `mapping_uint32.cu`) are a direct port of `nv/adm/mapping_uint16.cu` and
  `nv/adm/mapping_uint32.cu` from MANS. Kernel logic is unchanged.

  **Changes from original:** unused `MansParams` parameter removed; per-call
  `cudaMalloc`/`cudaFree` replaced by pool-allocated `AdmScratch`; `check_cuda()` replaced
  by `FZ_CUDA_CHECK`; namespace changed from `mans::nv::adm` to `fz::adm`; kernels renamed
  with `_u16`/`_u32` suffix to avoid TU-level naming conflicts; inline comments translated
  to English.

---

## SPERR {#ack_sperr}

**Repository:** https://github.com/NCAR/SPERR
**License:** Apache License 2.0
**Authors:** Shaomeng Li, Peter Lindstrom, John Clyne (NCAR)
**Citation:** Li, S., Lindstrom, P., Clyne, J. *Lossy Scientific Data Compression With
SPERR*. IPDPS 2023.

**Stages:**

- **`Cdf97Stage`** (`modules/transforms/cdf97/`) — direct port of the numerically
  load-bearing constants and rules from `sperr::CDF97` (`include/CDF97.h`,
  `src/CDF97.cpp`): the lifting constants (computed from the filter bank
  coefficients, not the commented QccPack literal values), the symmetric
  boundary-extension rule, the level-count rule, and the 3-D dyadic/wavelet-packet
  selection rule. Validated **bit-exact** against `sperr::CDF97` in double
  precision. GPU kernels (the axis kernel, coalesced-tile kernel, persistent
  cooperative-groups level fusion) are FZGPUModules' own — none of that exists in
  SPERR, which runs single-threaded CPU.

- **`Speck2DStage`** (`modules/coders/speck2d/`) — **algorithmic attribution only,
  no SPERR code used.** Codes the same kind of information SPERR's `SPECK2D_INT`
  bit-plane coder does, but the bitstream, data structures, and encode/decode
  algorithms are an independent, from-scratch, GPU-parallel-decodable design.
  SPERR's `SPECK_INT`/`SPECK2D_INT` use linked LIP/LIS/LSP lists and a DFS-serial
  traversal; none of that structure appears here. See
  `memory/speck_algorithm_writeup.md` for the full derivation and novelty
  statement.

- **`OutlierCorrectStage<Reconstructor>` / `Cdf97OutlierCorrectStage`**
  (`modules/coders/outlier_correct/`, `modules/coders/cdf97_outlier_correct/`) —
  **algorithmic attribution only, no SPERR code used.** Implements the same
  *mechanism* as SPERR's `Outlier_Coder` (dequantize + inverse-transform a trial
  reconstruction, diff against the original, exact-correct every point over
  bound), arrived at independently from `Outlier_Coder.h`/`.cpp` and
  `SPECK_FLT.cpp`'s call site — not by porting SPERR's code. Generalized beyond
  SPERR's own design via a transform-agnostic `Reconstructor` policy.

See [`THIRD_PARTY.md`](../THIRD_PARTY.md) for the full Apache License 2.0 text
required by upstream and the complete relationship writeup for each stage.

---

## dietGPU {#ack_dietgpu}

**Repository:** https://github.com/facebookresearch/dietgpu  
**License:** MIT  
**Authors:** Meta Platforms, Inc. and affiliates

**Stages:**

- **ANSStage** (`modules/coders/ans/`) — the rANS kernel headers
  (`GpuANSCodec.h`, `GpuANSEncode.h`, `GpuANSDecode.h`, `GpuANSStatistics.h`,
  `BatchPrefixSum.h`, and `utils/`) are vendored copies placed under
  `modules/coders/ans/dietgpu/`.

  **Changes from original:** `histogramBatch`/`histogramSingle` functions in
  `GpuANSStatistics.h` removed and replaced by the shared
  `fz::module::GPU_histogram_generic<uint8_t>` utility; namespace `multibyte_ans` adapted
  to `fz::ans`. All other kernel logic is unchanged from the original.

---

## GPULZ {#ack_gpulz}

**Repository:** https://github.com/hpdps-group/ICS23-GPULZ
**License:** **none declared upstream** — see the caveat below
**Copyright notice:** `(C) 2023 by Indiana University and Argonne National Laboratory.`
**Authors:** Boyuan Zhang, Jiannan Tian, Sheng Di, Xiaodong Yu, Martin Swany,
Dingwen Tao, Franck Cappello
**Paper:** "GPULZ: Optimizing LZSS Lossless Compression for Multi-byte Data on Modern
GPUs", ICS '23

**Stages:**

- **GPULZStage** (`modules/coders/gpulz/`) — a substantially rewritten derivative
  of upstream `gpulz.cu`. It retains `compressKernelI`'s per-chunk flag-bitmap/token
  stream grammar and sequential literal/match parse. Its exact/hashed match search,
  block prefix sum, staged writes, and block-parallel decoder are FZGPUModules
  implementations; the decoder no longer follows upstream's single-thread
  `decompressKernel`.

  The per-chunk container format (raw-fallback flag, CUB
  exclusive-scan packing offsets, deferred tail-size readback via `postStreamSync()`) is
  FZGPUModules' own, following the same pattern as `RREStage`/`RZEStage`. Upstream's
  separate flag/data pack-out step (`compressKernelIII`) is folded into FZGPUModules'
  `gpulzPackKernel`. Split mode (emitting literals, lengths, offsets, and metadata as
  separate output ports) has no upstream counterpart.

  **License caveat:** the upstream repository publishes no `LICENSE` file and declares no
  license. Its README copyright notice does not grant redistribution permission. Anyone
  redistributing FZGPUModules with `GPULZStage` should obtain permission or licensing
  terms from the GPULZ copyright holders first.

---

## AIZ_VLDB26 {#ack_aiz}

**Repository:** https://github.com/boyuanzhang62/AIZ_VLDB26
**License:** **none declared upstream** — same situation as GPULZ above
**Author:** Boyuan Zhang

**Stages:**

- **GPULZStage all-zero-chunk fast path** (`modules/coders/gpulz/`) — skipping the match
  search and the flag/data encode entirely for chunks that are wholly zero, gated on a
  warp-vote check, is adapted from the `notEmptyFlagArr` optimization in the "sparse"
  GPULZ variant at `test/gpulz.cuh` upstream.

  **Changes from original:** retargeted to `GPULZStage`'s compile-time-templated kernel
  structure, using `fz::backend::anySync32` for the warp vote and FZGPUModules' own
  container format — empty chunks are marked with a `(flag_size=0, data_size=0)` sentinel,
  distinct from the raw-fallback sentinel, and the corresponding output span is zero-filled
  on decode.

---

## FSZ {#ack_fsz}

**License:** BSD-3-Clause (FSZ 1.0.0, released 2026-08).
**Repository:** https://github.com/JiajunHuang1999/FSZ
**Author:** Jiajun Huang (University of South Florida)
**Paper:** "FSZ: Breaking the Prediction-Throughput Trade-off in GPU Lossy Compression",
SC '26, arXiv:2607.15413

**Relationship: algorithmic attribution only — no code was used.** These stages were
written from the paper's description alone, *before* any source release existed. FSZ
1.0.0 was published afterwards, on 2026-08; it has not been consulted for
implementation, only used as a reference build for validation (see below).

**Validated against the reference implementation (2026-08-07, H100, 20 cells).** The
reconstruction is faithful: PSNR is identical to the reference on every cell, and the
bitpacked payload is byte-for-byte the same size (4,277,784 B on NYX/`baryon_density`),
so the two make identical per-tile decisions. Compression ratio is 0.9928 of the
reference (geomean), and 99.6% of that deficit is FZGPUModules' separate `modes` port:
FSZ packs the same two flags free into spare bits of its per-block rate byte, which a
predictor stage decoupled from its coder cannot do. See
`compression_benchmarking/docs/adapters/fsz.md`.

**Stages:** `AdaptiveLorenzoStage` (`modules/fused/adaptive_lorenzo/`),
`LorenzoStage::setCentering()` / `setOrder(2)` (`modules/predictors/lorenzo/`),
`LorenzoQuantStage::Config::centering` (`modules/fused/lorenzo_quant/`).

The ideas taken from the paper are:

- **Cross-block prediction state** — running the prediction chain across the encoding
  blocks within a tile rather than restarting at every block, so a tile carries one raw
  seed instead of one per block.
- **Per-tile adaptive multi-order prediction and centering** — selecting, per tile, among
  first- and second-order Lorenzo with and without subtracting the tile mean, by exact
  encoded size rather than by an entropy proxy.
- **Single-pass four-way evaluation** — costing all four variants from one data read, using
  the fact that a constant offset cancels exactly in k-th order finite differences
  (`delta^k(q - mu) == delta^k(q)`) for every element with `k` predecessors, so centering
  perturbs only a tile's first one or two residuals.

**Differences from the paper:** the reference fuses prediction, quantization and encoding
into a single CUDA kernel with a decoupled-lookback prefix sum; FZGPUModules implements the
prediction step alone as a DAG stage that composes with `QuantizerStage` and
`AdaptiveBitpackStage`. Kernel structure, the cost model's coupling to
`AdaptiveBitpackStage`'s rate formula, port layout, side-channel compaction, serialization,
and all host-side plumbing are FZGPUModules code.

---

## Point-wise relative error transform {#ack_log_transform}

**License:** not applicable — **algorithmic attribution only**; the stage was written from
the paper's description, no reference implementation was used.
**Authors:** Xin Liang, Sheng Di, Dingwen Tao, Zizhong Chen, Franck Cappello
**Paper:** "An efficient transformation scheme for lossy data compression with point-wise
relative error bound", IEEE CLUSTER 2018, pp. 179–189

**Stages:**

- **LogTransformStage** (`modules/transforms/log_transform/`) — implements the paper's
  transformation scheme: mapping data into log space so that a point-wise *relative* error
  bound becomes a plain *absolute* bound, letting an ordinary ABS quantizer downstream
  deliver the relative guarantee. Kernel implementation, the sign/zero/near-zero outlier
  handling, and the DAG stage plumbing are FZGPUModules code.

---

## ROIBIN-SZ {#ack_roibin}

**Paper:** Robert Underwood, Chun Hong Yoon, Ali Murat Gok, Sheng Di, and Franck
Cappello, “ROIBIN-SZ: Fast and Science-Preserving Compression for Serial
Crystallography,” *Synchrotron Radiation News* 36(4), 17–22, 2023.
https://doi.org/10.1080/08940886.2023.2245722

**Public implementation:**
https://github.com/szcompressor/SZ2/tree/master/example/roibin_example

**Relationship:** `ROIBinSplitStage` is an independent GPU/DAG implementation
of the published ROI/background separation design. No SZ2 or LibPressio source
is copied. The kernels, FZROI1 peak-table format, fixed-output geometry,
three-port layout, and inverse scatter are FZGPUModules code.

---

## SZp / fZ-light {#ack_szp}

**Repository:** https://github.com/szcompressor/SZp

**License:** MIT

**Relationship:** `SZpStage` and `szp_composed.toml` are GPU adaptations of the
published predict-quantize-pack structure. No upstream source is copied. FZGM
uses different quantization and predictor-partition conventions, and its archive
is not byte-compatible with the upstream container.

Jiajun Huang, Sheng Di, et al. *hZCCL: Accelerating Collective Communication
with Co-Designed Homomorphic Compression*, SC '24.

---

## SZx {#ack_szx}

**Repository:** https://github.com/szcompressor/SZx

**License:** Argonne OPEN SOURCE LICENSE SF-16-105 (four-condition BSD style)

**Relationship:** `SZxStage` was implemented from the paper; no SZx source was
copied and the FZGM archive is not byte-compatible with the upstream format.

Xiaodong Yu, Sheng Di, Kai Zhao, Jiannan Tian, Dingwen Tao, Xin Liang, and
Franck Cappello. *Ultrafast Error-bounded Lossy Compression for Scientific
Datasets*, HPDC '22.
