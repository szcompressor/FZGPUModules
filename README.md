<h2 align="center">
FZ GPU Module Library (FZMod): A GPU-accelerated Module Library for Prediction-based Error-bounded Lossy Compression
</h2>

<p align="center">
<a href="./LICENSE"><img src="https://img.shields.io/badge/License-BSD%203--Clause-blue.svg"></a>
</p>

FZMod is a modular refactor of [cuSZ/pSZ](https://github.com/szcompressor/cuSZ), a GPU implementation of the [SZ algorithm](https://github.com/szcompressor/SZ). It aims to create a flexible and extensible framework for lossy GPU accelerated data-reduction modules. 

(C) 2025 by Argonne National Laboratory, and Indiana University. See [COPYRIGHT](https://github.com/szcompressor/FZModules/blob/main/LICENSE) in top-level directory.

- Developers: (primary) Skyler Ruiter, Pushkal Mudhapaka, Jiannan Tian.
- Special thanks to Fengguang Song for advising this project.

<br>

<p align="center", style="font-size: 2em">
<a href="https://github.com/szcompressor/FZModules/wiki/Build-and-Install"><b>Build and Install Wiki Page</b></a>
</p>

<p align="center", style="font-size: 2em">
<a href="https://github.com/szcompressor/FZModules/wiki/Command-Line-Interface-(CLI)-Usage"><b>Command Line Interface (CLI) Wiki Page</b></a>
</p>

<p align="center", style="font-size: 2em">
<a href="https://github.com/szcompressor/FZModules/wiki/Modules-Explained"><b>Modules Explained Wiki Page</b></a>


# FAQ 

<details>
<summary>
How do SZ, pSZ/cuSZ, and FZMod work?
</summary>

Prediction-based SZ algorithm comprises four major parts,

0. User specifies error-mode (e.g., absolute value (`abs`), or relative to data value magnitude (`r2r`) and error-bound.)
1. Prediction errors are quantized in units of input error-bound (*quant-code*). Range-limited quant-codes are stored, whereas the out-of-range codes are otherwise gathered as *outlier*.
3. The in-range quant-codes are fed into Huffman encoder. A Huffman symbol may be represented in multiple bytes.
4. (CPU-only) additional lossless encoding pass can be applied to the Huffman-encoded data, e.g., DEFLATE, ZSTD, etc.

</details>

<details>
<summary>
What datasets are used?
</summary>

We tested using datasets from [Scientific Data Reduction Benchmarks](https://sdrbench.github.io/) (SDRBench).

| dataset                                                                 | dim. | description                                                  |
| ----------------------------------------------------------------------- | ---- | ------------------------------------------------------------ |
| [EXAALT](https://gitlab.com/exaalt/exaalt/-/wikis/home)                 | 1D   | molecular dynamics simulation                                |
| [HACC](https://www.alcf.anl.gov/files/theta_2017_workshop_heitmann.pdf) | 1D   | cosmology: particle simulation                               |
| [CESM-ATM](https://www.cesm.ucar.edu)                                   | 2D   | climate simulation                                           |
| [EXAFEL](https://lcls.slac.stanford.edu/exafel)                         | 2D   | images from the LCLS instrument                              |
| [Hurricane ISABEL](http://vis.computer.org/vis2004contest/data.html)    | 3D   | weather simulation                                           |
| [NYX](https://amrex-astro.github.io/Nyx/)                               | 3D   | adaptive mesh hydrodynamics + N-body cosmological simulation |

To download more SDRBench datasets, please use [`scripts/download-sdrb-data.sh`](scripts/download-sdrb-data.sh). 

</details>