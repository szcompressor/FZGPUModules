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

## cuSZ-Hi

**Used by:** `GInterpStage` (`modules/predictors/ginterp/`)

**Relationship:**
- `GInterpStage` (`modules/predictors/ginterp/`) — the multi-level spline
  interpolation kernels are adapted from `spline3.cu` and `spline3_md.inl`
  in the cuSZ-Hi repository. Changes from the original: `namespace cusz` →
  `namespace fz::ginterp`; `err.hh` / `timer.hh` includes stripped; the
  `pszmem_cxx<T>` buffer abstraction is replaced by raw device pointers
  routed through the FZGPUModules `MemoryPool`; the `CompactDram` outlier
  triplet is replaced by separate `outlier_vals` / `outlier_idxs` /
  `outlier_count` output ports; minimal `cusz_type_subset.h` reproduces
  only the `INTERPOLATION_PARAMS` struct and `u4` typedef from upstream
  `cusz/type.h`. Host-side wrapper, memory-pool integration, outlier-
  fusion contract, and radius auto-tune are FZGPUModules code.

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
