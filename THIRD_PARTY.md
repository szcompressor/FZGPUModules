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
