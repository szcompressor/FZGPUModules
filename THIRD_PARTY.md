# Third-Party Licenses

FZGPUModules incorporates code from the following open-source projects.
Each section names the affected modules, states the relationship to the
upstream code (direct port, algorithm-faithful reimplementation, or
vendored with modification), and reproduces the required copyright notice
verbatim to satisfy BSD-3-Clause condition 2 (binary redistribution).

---

## LC Framework

**Used by:** `RZEStage`, `BitshuffleStage` (4- and 8-byte butterfly kernels),
`DifferenceStage`, `QuantizerStage`

**Relationship:**
- `RZEStage` (`modules/coders/rze/`) — GPU kernels are a direct port of
  `zero_elim.h`, `repeated_elim.h`, and `rze.h` from the LC framework.
- `BitshuffleStage` (`modules/shufflers/bitshuffle/`) — the 4- and 8-byte
  butterfly shuffle kernels are adapted directly from `d_BIT_4` / `d_BIT_8`
  in the LC framework; the 1- and 2-byte paths use a standard `__ballot_sync`
  approach and are not LC-derived.
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

**Used by:** `BitplaneRLEStage` (`modules/fused/bitplane_rle/`)

**Relationship:**
- `BitplaneRLEStage` (`modules/fused/bitplane_rle/`) — the fused
  bitplane-transpose + zero-byte run-length encode/decode kernels
  (`bitplane_rle_encode.inl`, `bitplane_rle_decode.inl`) are adapted from
  `KERNEL_CUHIP_fz_fused_encode` / `KERNEL_CUHIP_fz_fused_decode` of the
  FZ-GPU lossless codec, as vendored in `origin/v1.1.0_dev` of the cuSZ
  repository (`modules/codec/fzg/`). Changes from the original:
  `namespace fzgpu` → `namespace fz::bitplane_rle`; `err.hh` / `CHECK_GPU`
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

**Relationship:** These are **independent reimplementations of published cuSZp
schemes — no cuSZp source code is copied.** Mapping our pieces to the papers:

- **cuSZp (SC'23)** — the family's core: linear error-bounded quantization,
  block-local 1-D Lorenzo, fixed-length (per-block fixed-rate bit-plane)
  encoding, and a block bit-shuffle. We reimplement the first three:
  `QuantizerStage`'s linear mode reproduces `q = round(x / 2·eb)` (no
  radius/outlier fallback); `LorenzoStage::setBlockSize` reproduces the
  block-local 1-D delta; `AdaptiveBitpackStage` reproduces the fixed-length
  encoding (byte-granular layout, one-thread-per-block kernels, an ordinary CUB
  `DeviceScan` for per-block offsets where cuSZp fuses a decoupled look-back scan
  into one kernel — that fusion is left to a downstream compiler). The SC'23
  **block bit-shuffle is not reproduced** as a cuSZp stage (FZGPUModules has a
  separate LC-framework `BitshuffleStage`).
- **cuSZp2 (SC'24)** — adds the per-block **plain vs. outlier** selection over
  the fixed-length backend. `AdaptiveBitpackStage`'s default plain mode and its
  `setOutlierSelection(true)` reproduce these two modes.
- **cuSZp3 / VGC (SC'25)** — adds **dimension-aware (1-D/2-D/3-D) delta** with
  three modes (fixed = no delta, plain = delta, outlier = delta + outlier).
  `TiledLorenzoStage` reproduces the 2-D/3-D tiled separable delta; combined with
  the stages above it yields all three modes (1-D delta is `LorenzoStage`'s block
  mode). cuSZp3's **memory-efficient compression** and **selective decompression**
  features are **not ported** (they don't map cleanly onto the staged pipeline).

The reference codebases (for cross-checking only, not vendored) live at
`compressors/cuSZp2/` and `compressors/cuSZp3/`; design notes are in
`memory/cuszp_stages.md`.

**Papers** (all Argonne National Laboratory / University of Iowa):
- Yafan Huang, Sheng Di, Xiaodong Yu, Guanpeng Li, Franck Cappello, "cuSZp: An
  Ultra-fast GPU Error-bounded Lossy Compression Framework with Optimized
  End-to-End Performance", SC '23.
- Yafan Huang, Sheng Di, Guanpeng Li, Franck Cappello, "cuSZp2: A GPU Lossy
  Compressor with Extreme Throughput and Optimized Compression Ratio", SC '24.
- Yafan Huang, Sheng Di, Guanpeng Li, Franck Cappello, "GPU Lossy Compression for
  HPC Can Be Versatile and Ultra-Fast" (cuSZp3 / VGC), SC '25.

**License:** cuSZp is BSD-3-Clause. As no source is copied, this is an
algorithmic attribution.

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
