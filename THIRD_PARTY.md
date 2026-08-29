# Third-Party Licenses

FZGPUModules incorporates code from the following open-source projects.
Each section names the affected modules, states the relationship to the
upstream code (direct port, algorithm-faithful reimplementation, or
vendored with modification), and reproduces the required copyright notice
verbatim to satisfy BSD-3-Clause condition 2 (binary redistribution).

---

## GPULZ

**Repository:** https://github.com/hpdps-group/ICS23-GPULZ

**Used by:** `GPULZStage` (`modules/coders/gpulz/`)

**Relationship:** `GPULZStage`'s encode/decode kernels are a direct port of
`compressKernelI` / `decompressKernel` from the upstream `gpulz.cu` reference
implementation: the per-chunk sliding-window match search (shared-memory
lookahead buffer + window, Blelloch prefix sum over per-item byte sizes,
literal/match flag-bitmap construction) is preserved verbatim, retargeted from
a fixed `BLOCK_SIZE`/`WINDOW_SIZE`/`INPUT_TYPE` macro configuration to
compile-time template parameters (`T`, `CS`) dispatched at runtime. The
per-chunk container format (raw-fallback flag, CUB exclusive-scan packing
offsets, deferred tail-size readback via `postStreamSync()`) is FZGM's own,
following the same pattern as `RREStage`/`RZEStage`; `compressKernelIII`
(upstream's separate flag/data pack-out step) is folded into FZGM's own
`gpulzPackKernel`.

Original authors: Boyuan Zhang, Jiannan Tian, Sheng Di, Xiaodong Yu, Martin
Swany, Dingwen Tao, Franck Cappello.
Paper: "GPULZ: Optimizing LZSS Lossless Compression for Multi-byte Data on
Modern GPUs", ICS '23.

**License:** the upstream repository does not include a `LICENSE` file or
declare a license (GitHub reports `license: null` as of this writing). No
license terms are reproduced here because none are published upstream — this
differs from every other entry in this document, which vendor code under an
explicit permissive license. Anyone redistributing FZGPUModules (including
`GPULZStage`) should contact the GPULZ authors to confirm terms before
relying on this component outside of research/internal use.

---

## AIZ_VLDB26

**Repository:** https://github.com/boyuanzhang62/AIZ_VLDB26

**Used by:** `GPULZStage` (`modules/coders/gpulz/`)

**Relationship:** `GPULZStage`'s all-zero-chunk fast path (skip the match
search and flag/data encode entirely for chunks that are entirely zero,
gated on a warp-vote `fz::backend::anySync32` check) is adapted from the
`notEmptyFlagArr` optimization in the "sparse" GPULZ variant at
`test/gpulz.cuh` in this repository — a research artifact for a GPU-based AI
lossless compression pipeline that pipes quantized neural-compressor latents
through a modified GPULZ. The optimization idea (and the `notEmptyFlag`
warp-vote pattern) is reused; FZGM's implementation is retargeted to the
compile-time-templated kernel structure of `GPULZStage` and its own
container format (empty chunks are marked via a `(flag_size=0, data_size=0)`
sentinel, distinct from the raw-fallback sentinel, with the corresponding
output span zero-filled on decode).

Original author: Boyuan Zhang.

**License:** the upstream repository does not include a `LICENSE` file or
declare a license (GitHub reports `license: null` as of this writing) — same
situation as the GPULZ entry above.

---

## LC Framework

**Used by:** `RZEStage`, `RREStage`, `RAREStage`, `RAZEStage`, `CLOGStage`,
`HCLOGStage`, `BitshuffleStage` (4- and 8-byte butterfly kernels), `TUPLStage`,
`DifferenceStage`, `QuantizerStage`

**Relationship:**
- `RREStage` + `RZEStage` (`modules/coders/{rre,rze}/`) — GPU kernels are a
  faithful port of `d_RRE.h`, `d_RZE.h`, `d_repetition_elimination.h`,
  `d_zero_elimination.h`, and `prefix_sum.h` from the LC framework (the `RRE` and
  `RZE` lossless components used by cuSZ-Hi's LC pipelines), vendored together in
  `modules/coders/lc_common/lc_chunk_components.cuh`.  Both support LC word sizes
  1/2/4/8 (`RRE_N` / `RZE_N`).
- `RAREStage` + `RAZEStage` (`modules/coders/{rare,raze}/`) — GPU kernels are a
  faithful port of `d_RARE.h` and `d_RAZE.h` from the LC framework (the auto-k
  generalizations of `RRE`/`RZE`), sharing a single merged
  `d_PRencode`/`d_PRdecode<T, PartialReduceMode>` template in
  `modules/coders/lc_common/lc_chunk_components.cuh` — the two upstream files are
  textually identical apart from their match predicate (repetition vs.
  leading-zero-count). Both support LC word sizes 1/2/4/8 (`RARE_N` / `RAZE_N`).
- `CLOGStage` + `HCLOGStage` (`modules/coders/{clog,hclog}/`) — GPU kernels are a
  faithful port of `d_CLOG.h` and `d_HCLOG.h` from the LC framework (fixed
  32-subchunk adaptive bit-width truncation, `T` unsigned only; HCLOG adds a
  per-subchunk TCMS/zigzag fallback), sharing a single merged
  `d_CLOGencode`/`d_CLOGdecode<T, CLogMode>` template in
  `modules/coders/lc_common/lc_clog_components.cuh`. Both support LC word sizes
  1/2/4/8 (`CLOG_N` / `HCLOG_N`).
- `BitshuffleStage` (`modules/shufflers/bitshuffle/`) — the 4- and 8-byte
  butterfly shuffle kernels are adapted directly from `d_BIT_4` / `d_BIT_8`
  in the LC framework; the 1- and 2-byte paths use a standard `__ballot_sync`
  approach and are not LC-derived.
- `TUPLStage` (`modules/shufflers/tupl/`) — GPU kernels are a faithful port of
  `d_TUPL` / `d_iTUPL` from the LC framework (the `TUPLk` tuple deinterleave /
  AoS-to-SoA transpose component). Upstream generates one fixed
  `(dim, word_size)` instantiation per component over a hardcoded 16 KB chunk
  (`TUPL2_1`, `TUPL6_8`, `TUPL12_1`, ...); here `dim`, `word_size`, and
  `block_size` are independent runtime parameters instead.
- `DifferenceStage` (`modules/predictors/diff/`) — independently written CUDA
  kernel following the `d_DIFFNB` algorithm described in the LC/PFPL framework.
- `QuantizerStage` (`modules/quantizers/quantizer/`) — independently written
  CUDA kernel following the LC/PFPL quantization scheme including the ABS/NOA/REL
  error-bound modes, outlier handling, and log-space REL encoding.

**License:**

```
This file is part of the LC framework for synthesizing high-speed parallel
lossless and error-bounded lossy data compression and decompression algorithms
for CPUs and GPUs.

BSD 3-Clause License

Copyright (c) 2021-2025, Noushin Azami, Alex Fallin, Brandon Burtchell,
Andrew Rodriguez, Benila Jerald, Yiqian Liu,
Anju Mongandampulath Akathoott, and Martin Burtscher
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

URL: The latest version of this code is available at
     https://github.com/burtscher/LC-framework.

Sponsor: This code is based upon work supported by the U.S. Department of
Energy, Office of Science, Office of Advanced Scientific Research (ASCR),
under contract DE-SC0022223.
```

---

## cuSZ / PHF

**Used by:** `LorenzoQuantStage`, `HuffmanStage`

**Relationship:**
- `LorenzoQuantStage` (`modules/fused/lorenzo_quant/`) — GPU kernels and
  the fused predictor+quantizer design follow the cuSZ Lorenzo implementation
  (`lrz_c.cuhip.inl`, `lrz_x.cuhip.inl`).
- `HuffmanStage` (`modules/coders/huffman/`) — the PHF source files
  (`hf.h`, `hf_bk*.cc`, `hf_buf.cc`, `hf_canon.cc`, `hf_hl.cc`,
  `hf_kernels.cu`, `hf_impl.hh`) are vendored copies adapted from
  `origin/v1.1.0_dev` of the PHF codec in the cuSZ repository, with
  modifications documented at the top of each file.

**License:**

```
Copyright (c) 2020-2022, UChicago Argonne, LLC and Washington State University
Copyright (c) 2022-2024, UChicago Argonne, LLC and Indiana University
Copyright (c) 2024, UChicago Argonne, LLC and University of Kentucky
Copyright (c) 2025, UChicago Argonne, LLC and Oakland University

All Rights Reserved

Software Name: pSZ/cuSZ: A GPU-Based Error-Bounded Lossy Compressor for
               Scientific Data

OPEN SOURCE LICENSE

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Contact: SZ Team (szlossycompressor@gmail.com)
```

---

## FZ-GPU

**Used by:** `BitplaneRZEStage` (`modules/fused/bitplane_rze/`)

**Relationship:**
- `BitplaneRZEStage` (`modules/fused/bitplane_rze/`) — the fused
  bitplane-transpose + zero-group encode/decode kernels
  (`bitplane_rze_encode.inl`, `bitplane_rze_decode.inl`) are adapted from
  `KERNEL_CUHIP_fz_fused_encode` / `KERNEL_CUHIP_fz_fused_decode` of the
  FZ-GPU lossless codec, as vendored in `origin/v1.1.0_dev` of the cuSZ
  repository (`modules/codec/fzg/`). Changes from the original:
  `namespace fzgpu` → `namespace fz::bitplane_rze`; `err.hh` / `CHECK_GPU`
  stripped; the `fzgpu::Buf` allocation/`alloc_test_buf` path is dropped and
  all device memory is routed through the FZGPUModules `MemoryPool`; the
  128-byte self-describing archive header (`fzg_header`) is reproduced as
  `ArchiveHeader`. The kernel bodies are preserved verbatim. Host-side
  wrapper, memory-pool integration, and the padded-input handling are
  FZGPUModules code.

  Original authors: Boyuan Zhang (kernel), Jiannan Tian (refactor).
  Paper: Boyuan Zhang, Jiannan Tian, Sheng Di, Xiaodong Yu, Yunhe Feng,
  Xin Liang, Dingwen Tao, Franck Cappello, "FZ-GPU: A Fast and High-Ratio
  Lossy Compressor for Scientific Computing Applications on GPUs", HPDC '23.

**Reference copy** of the upstream files (unmodified) is in
`memory/references/dictionary/` for cross-checking.

**License:** vendored from the cuSZ repository — same OPEN SOURCE LICENSE as
the [cuSZ / PHF](#cusz--phf) section above.

---

## cuSZp / cuSZp2 / cuSZp3

**Repository:** https://github.com/szcompressor/cuSZp (the single repo hosts all
three generations; the current `main` is the cuSZp3 / VGC generation).

**Used by:** `AdaptiveBitpackStage` (`modules/coders/adaptive_bitpack/`),
`TiledLorenzoStage` (`modules/predictors/tiled_lorenzo/`), and the `linear` mode
of `QuantizerStage` + the `setBlockSize` option of `LorenzoStage`.

**Relationship:** Mixed — **two components contain direct copies/ports of cuSZp
kernel source** (`AdaptiveBitpackStage`, `TiledLorenzoStage`), while two are
**independent reimplementations** with no source copied (`QuantizerStage` linear
mode, `LorenzoStage` block mode). The BSD-3-Clause copyright notice is reproduced
verbatim below to satisfy the source-redistribution condition for the copied
parts. Mapping our pieces to the papers:

- **cuSZp (SC'23)** — the family's core: linear error-bounded quantization,
  block-local 1-D Lorenzo, fixed-length (per-block fixed-rate bit-plane)
  encoding, and a block bit-shuffle. `QuantizerStage`'s linear mode is an
  **independent reimplementation** of `q = round(x / 2·eb)` (no radius/outlier
  fallback); `LorenzoStage::setBlockSize` is an **independent reimplementation**
  of the block-local 1-D delta; **`AdaptiveBitpackStage` is a direct port of the
  cuSZp fixed-length encode/decode kernel logic**, re-expressed
  one-thread-per-block with a byte-granular layout and an ordinary CUB
  `DeviceScan` for per-block offsets (cuSZp fuses a decoupled look-back scan into
  one kernel — that fusion is left to a downstream compiler), wrapped with
  FZGPUModules `MemoryPool` integration and the FZM header/stage scaffolding. The
  SC'23 **block bit-shuffle is not reproduced** as a cuSZp stage (FZGPUModules has
  a separate LC-framework `BitshuffleStage`).
- **cuSZp2 (SC'24)** — adds the per-block **plain vs. outlier** selection over
  the fixed-length backend. `AdaptiveBitpackStage`'s default plain mode and its
  `setOutlierSelection(true)` reproduce these two modes.
- **cuSZp3 / VGC (SC'25)** — adds **dimension-aware (1-D/2-D/3-D) delta** with
  three modes (fixed = no delta, plain = delta, outlier = delta + outlier).
  **`TiledLorenzoStage` is a direct port of the cuSZp3 2-D/3-D tiled separable
  delta kernel logic** (from `cuSZp_kernels_{2D,3D}_f32.cu`), re-expressed as a
  standalone integer predictor with a tile-major output reshape + zero-padding so
  it composes with `AdaptiveBitpackStage`; the tile-major decomposition, FZM
  header, and `MemoryPool` integration are FZGPUModules code. Combined with the
  stages above it yields all three modes (1-D delta is `LorenzoStage`'s block
  mode). cuSZp3's **memory-efficient compression** and **selective decompression**
  features are **not ported** (they don't map cleanly onto the staged pipeline).

The reference codebases live at `compressors/cuSZp2/` and `compressors/cuSZp3/`;
design notes are in `memory/cuszp_stages.md`.

**Papers** (all Argonne National Laboratory / University of Iowa):
- Yafan Huang, Sheng Di, Xiaodong Yu, Guanpeng Li, Franck Cappello, "cuSZp: An
  Ultra-fast GPU Error-bounded Lossy Compression Framework with Optimized
  End-to-End Performance", SC '23.
- Yafan Huang, Sheng Di, Guanpeng Li, Franck Cappello, "cuSZp2: A GPU Lossy
  Compressor with Extreme Throughput and Optimized Compression Ratio", SC '24.
- Yafan Huang, Sheng Di, Guanpeng Li, Franck Cappello, "GPU Lossy Compression for
  HPC Can Be Versatile and Ultra-Fast" (cuSZp3 / VGC), SC '25.
  https://doi.org/10.1145/3712285.3759817

**License:**

```
Copyright © 2024, UChicago Argonne and University of Iowa

All Rights Reserved

Software Name: cuSZp: A Fast and High-ratio GPU Error-bounded Lossy Compressor

By: Argonne National Laboratory, University of Iowa

OPEN SOURCE LICENSE

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this
   list of conditions and the following disclaimer in the documentation and/or
   other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors may
   be used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

Contact: SZ Team (szlossycompressor@gmail.com)
```

---

## FSZ

**Used by:** `AdaptiveLorenzoStage` (`modules/fused/adaptive_lorenzo/`),
`LorenzoStage::setCentering()` / `setOrder(2)`
(`modules/predictors/lorenzo/`), `LorenzoQuantStage::Config::centering`
(`modules/fused/lorenzo_quant/`)

**Relationship: algorithmic attribution only — no code was used.** These stages
were written from the paper's description alone, before FSZ had a published
source release. FSZ 1.0.0 was released afterwards, on 2026-08, under
BSD-3-Clause at https://github.com/JiajunHuang1999/FSZ; it has been used since
only as a reference build for validation, never consulted for implementation.
The ideas taken are:

- **Cross-block prediction state** — running the prediction chain across the
  encoding blocks within a tile rather than restarting at every block, so a
  tile has one raw seed instead of one per block.
- **Per-tile adaptive multi-order prediction and centering** — selecting per
  tile among first/second-order Lorenzo with and without subtracting the tile
  mean, by exact encoded size.
- **Single-pass four-way evaluation** — costing all four variants from one data
  read, using the fact that a constant offset cancels exactly in k-th order
  finite differences (`delta^k(q - mu) == delta^k(q)`) for every element with
  `k` predecessors, so centering perturbs only a tile's first one or two
  residuals.

The reference fuses prediction, quantization and encoding into one CUDA kernel
with a decoupled-lookback prefix sum; FZGPUModules implements the prediction
step alone as a DAG stage composing with `QuantizerStage` and
`AdaptiveBitpackStage`. Kernel structure, the cost model's coupling to
`AdaptiveBitpackStage`'s rate formula, port layout, serialization, and all
host-side plumbing are FZGPUModules code.

**Citation:**
```
Jiajun Huang. "FSZ: Breaking the Prediction-Throughput Trade-off in GPU Lossy
Compression." SC'26. arXiv:2607.15413.
University of South Florida, Tampa, FL, USA.
```

---

## cuSZ-Hi

**Used by:** `GInterpStage` (`modules/fused/ginterp/`)

**Relationship:**
- `GInterpStage` (`modules/fused/ginterp/`) — the multi-level spline
  interpolation kernels are adapted from `spline3.cu` and `spline3_md.inl`
  in the cuSZ-Hi repository. Changes from the original: `namespace cusz` →
  `namespace fz::ginterp`; `err.hh` / `timer.hh` includes stripped; the
  `pszmem_cxx<T>` buffer abstraction is replaced by raw device pointers
  routed through the FZGPUModules `MemoryPool`; the `CompactDram` outlier
  triplet is replaced by separate `outlier_vals` / `outlier_idxs` /
  `outlier_count` output ports; minimal `cusz_type_subset.h` reproduces
  only the `INTERPOLATION_PARAMS` struct and `u4` typedef from upstream
  `cusz/type.h`. Host-side wrapper, memory-pool integration, outlier-
  fusion contract, radius auto-tune, and all five auto-tune modes
  (`setAutoTuning(0..5)`) are FZGPUModules code wrapping the upstream
  device kernels.

**Bug fix patched locally:** `pa_spline_infprecis_data`'s SPLINE_DIM==2
level==0 atomic offset was `errors + 15 + BIY` upstream, placing BIY=5
at slot 20 which collided with the level==1 BIY=4 write to the same
slot. The host-side analysis loop `for(level=3; level<LEVEL; ++level)
errors[level*6-9 .. level*6-4]` expects level=5 (level_id=0) at
`errors[21..26]`, so our copy uses `errors + 16 + BIY`. The fix is
documented in the adapter-changes comment block at the top of
`ginterp_md.inl` and in [docs/stages/ginterp.md](docs/stages/ginterp.md).

**Reference copy** of the upstream kernels (unmodified) is in
`memory/references/spline_cuszhi/` (`spline3.cu`, `spline3_md.inl`,
`type.h`) for cross-checking against the patched local copy.

**Paper:**
- Shixun Wu, Jinwen Pan, Jinyang Liu, Jiannan Tian, Ziwei Qiu, Jiajun Huang,
  Kai Zhao, Xin Liang, Sheng Di, Zizhong Chen, Franck Cappello, "Boosting
  Scientific Error-Bounded Lossy Compression through Optimized Synergistic
  Lossy-Lossless Orchestration" (cuSZ-Hi), SC '25.
  https://doi.org/10.1145/3712285.3759798

**License:**

```
BSD 3-Clause License

Copyright (c) 2024-2025, Indiana University and UChicago Argonne, LLC
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

URL: https://github.com/shixun404/cuSZ-Hi

---

## dietGPU

**Used by:** `ANSStage` (`modules/coders/ans/`)

**Relationship:**
- The rANS kernel headers (`GpuANSCodec.h`, `GpuANSEncode.h`, `GpuANSDecode.h`,
  `GpuANSStatistics.h`, `BatchPrefixSum.h`, and utils/) are vendored copies
  adapted from the dietGPU repository and placed under `modules/coders/ans/dietgpu/`.
  The `histogramBatch`/`histogramSingle` functions in `GpuANSStatistics.h` were
  removed and replaced by the shared `fz::module::GPU_histogram_generic<uint8_t>`
  utility; the namespace `multibyte_ans` was adapted to `fz::ans`.
  All other kernel logic is unchanged from the original.

**License:**

```
MIT License

Copyright (c) Meta Platforms, Inc. and affiliates.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

URL: https://github.com/facebookresearch/dietgpu

---

## MANS

**Used by:** `ADMStage` (`modules/transforms/adm/`), `MANSStage` (`modules/fused/mans/`)

**Relationship:**
- `ADMStage` (`modules/transforms/adm/mapping_uint16.cu`,
  `mapping_uint32.cu`) — GPU kernels are a direct port of
  `nv/adm/mapping_uint16.cu` and `nv/adm/mapping_uint32.cu` from the MANS
  repository.  Kernel logic is unchanged.  Changes from the original: unused
  `MansParams` parameter removed; per-call `cudaMalloc`/`cudaFree` replaced
  by pool-allocated `AdmScratch`; `check_cuda()` replaced by `FZ_CUDA_CHECK`;
  namespace changed from `mans::nv::adm` to `fz::adm`; kernels renamed with
  `_u16`/`_u32` suffix to avoid TU-level naming conflicts; inline Chinese
  comments translated to English.
- `MANSStage` (`modules/fused/mans/`) — to be added in a future release;
  will follow the fused ADM+rANS design from the MANS repository.  The GPU
  rANS component is covered by the dietGPU entry above.

**License:**

```
BSD 3-Clause License

Copyright (c) 2025
Developers: Wenjing Huang, Jinwu Yang, JingKai Huang, Haoquan Long
Advisors: Dingwen Tao, Guangming Tan
All contributors to the MANS project

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

URL: https://github.com/hpdps-group/MANS

---

## SPERR (CDF97Stage)

**Repository:** https://github.com/NCAR/SPERR

**Used by:** `Cdf97Stage` (`modules/transforms/cdf97/`)

**Relationship: direct port of the numerically load-bearing constants and
rules, GPU kernels are FZGPUModules' own.** `Cdf97Stage` reimplements SPERR's
CDF 9/7 biorthogonal wavelet transform (the DWT front-half of the SPERR
compressor) as a separable, multi-level, GPU lifting-scheme transform. Ported
verbatim from `sperr::CDF97`:

- The lifting constants (ALPHA/BETA/GAMMA/DELTA/EPSILON), computed from the
  filter bank coefficients `h[]` — **not** the commented QccPack literal
  values, which do not reproduce SPERR's coefficients bit-for-bit.
- The symmetric boundary-extension rule at signal edges (SPERR's four
  length-parity special cases, restated as branch-free clamped-index reads —
  proven equivalent, not just similar).
- The level-count rule (`num_of_xforms`: `while (len >= 9) len -= len/2`,
  capped at 6, computed once from the governing/minimum dimension).
- The 3-D dyadic-vs-wavelet-packet selection rule (`can_use_dyadic`).

The `double` path is validated **bit-exact** against `sperr::CDF97` (forward
coefficients differ by exactly 0.0) across 2-D and 3-D dyadic/wavelet-packet
shapes. Everything GPU-specific — the axis kernel, the coalesced-tile kernel
for strided passes, the cooperative-groups persistent-kernel level fusion, the
occupancy-based fallback logic, and the `Cdf97Stage` DAG wrapper — is
FZGPUModules' own; none of it exists in SPERR, which runs single-threaded CPU.

SPERR is licensed under the Apache License, Version 2.0. Full license text:

```
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS
```

URL: https://github.com/NCAR/SPERR

---

## SPECK2DStage — algorithmic attribution, not a port

**Used by:** `Speck2DStage` (`modules/coders/speck2d/`)

**Relationship: algorithmic attribution only — no SPERR code was used.**
`Speck2DStage` codes the same *kind* of information SPERR's `SPECK2D_INT`
bit-plane coder does (a hierarchical significance map over DWT-coefficient
magnitudes, coarsest bit-plane first), but its bitstream, data structures, and
encode/decode algorithms are an independent, from-scratch design, built by
reading the SPERR source to understand the reference algorithm and then
deriving a different, GPU-parallel-decodable formulation — not by porting or
adapting SPERR's code. SPERR's `SPECK_INT`/`SPECK2D_INT`/`SPECK2D_INT_ENC`/
`SPECK2D_INT_DEC` use linked LIP/LIS/LSP lists and a DFS-serial encode/decode
with an embedded/progressive bit order; none of that structure, nor any code
from it, appears here. See `memory/speck_algorithm_writeup.md` for the full
derivation and a calibrated novelty statement — "listless SPECK"/"no-list
SPIHT (NLS)"/GPU-SPIHT/EBCOT are the relevant prior art for the general idea
of replacing list-based bookkeeping with precomputed positional structure, and
should be read before any publication claim building on this stage.

SPERR (Apache License 2.0, see the CDF97 entry above) was used only as a
*reference oracle* during development — to measure this stage's compression
rate against, not to derive its implementation from.
