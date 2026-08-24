# LibPressio Python Bindings {#libpressio_python}

FZGPUModules experimentally exposes a [libpressio](https://github.com/robertu94/libpressio) plugin (`fzgpumodules`)
that lets you build and run GPU compression pipelines from Python (or C++) through libpressio. 
The plugin translates libpressio's generic compressor interface into FZGPUModules pipeline
calls, handles GPU memory management, and surfaces metrics after each compress.

---

## Setup

### Prerequisites

- CUDA toolkit and drivers (11.7 or later)
- C++ compiler supported by your CUDA version (GCC 7+ or Clang 5+)
- CMake 3.18+

### Install spack

```bash
git clone --depth=2 https://github.com/spack/spack.git ~/spack
. ~/spack/share/spack/setup-env.sh   # add to ~/.bashrc to persist
```

### Add the spack package repos

FZGPUModules' explicit-ownership pipeline API (`OwnedDeviceBuffer`/`BorrowedDeviceBuffer`,
`Pipeline::decompressOwned()`) has landed on `szcompressor/FZModules` `main` — no PR to
track there anymore. The libpressio `fzgpumodules` plugin itself, and the spack package
definitions that build it, are not yet merged into upstream `robertu94/libpressio` or
`robertu94/spack_packages`. Until those two land, use the fork which contains the package
definitions for both `fzgpumodules` and the updated `libpressio`:

```bash
# Provides fzgpumodules package + libpressio with +fzgpumodules variant support.
# Once the upstream PR is merged, replace with: robertu94/spack_packages
spack repo add --name robert_pkgs https://github.com/skyler-ruiter/spack_packages.git
spack repo list   # should show robert_pkgs → spack_repo/robertu94
```

### Create and activate a spack environment

```bash
spack env create fzgm-env
spack env activate fzgm-env
```

### Point spack at the libpressio source fork

The plugin code lives in a fork of libpressio (PR not yet merged upstream). Use
`spack develop` so spack builds directly from the fork source:

```bash
git clone https://github.com/skyler-ruiter/libpressio ~/libpressio-fork
spack develop --path ~/libpressio-fork libpressio@master
```

### Install

```bash
spack add libpressio +cuda +python +fzgpumodules cuda_arch=<your_arch>
spack concretize
spack install
```

Replace `<your_arch>` with your GPU's compute capability (e.g. `80` for A100, `86` for RTX 3090,
`90` for H100).

**Gotcha:** if spack resolves `swig@4.1` or newer, the build fails inside
`pressioPYTHON_wrap.cxx` with `too few arguments to function 'SWIG_Python_AppendOutput'` —
libpressio's vendored `numpy.i` calls the pre-4.1 2-argument form, and SWIG 4.1 added a
required third argument to that runtime function. Pin swig in the environment before
concretizing:

```bash
spack config add "packages:swig:require:@=4.0.2"
```

### Activate in Python

```bash
spack env activate fzgm-env   # run in each new shell
```

---

## Quick Start

```python
import numpy as np
import libpressio as lp

data = np.random.rand(256, 256).astype(np.float32)

comp = lp.PressioCompressor.from_config({
    "compressor_id": "fzgpumodules",
    "early_config": {
        "fzgpumodules:stages":      ["lorenzo:float:uint16", "rle:uint16"],
        "fzgpumodules:connections": ["s1 <- s0:codes"],
    },
    "compressor_config": {
        "pressio:abs": 1e-3,
    },
})

compressed   = comp.encode(data)
decompressed = comp.decode(compressed, data.copy())

print(f"max error: {float(abs(data - decompressed).max()):.3e}")  # <= 1e-3
```

For linear pipelines, `python/fzgm_helper.py`'s `Chain` builder avoids hand-writing
`"sN <- sM:port"` connection strings and `"fzgpumodules:sN:<option>"` keys:

```python
import numpy as np
from fzgm_helper import Chain

data = np.random.rand(256, 256).astype(np.float32)

comp = (Chain()
    .add("lorenzo:float:uint16", quant_radius=999, outlier_capacity=0.25)
    .add("rze", from_port="codes", chunk_size=8192)
    .configure(fusion="auto")
    .compressor(eb=1e-3))

compressed   = comp.encode(data)
decompressed = comp.decode(compressed, data.copy())
```

It's a thin builder, not a libpressio feature — for non-linear graphs (multiple
branches into one stage, direction-dependent stages, config-file mode) build
`early_config` by hand as above.

---

## GPU-resident data (zero-copy)

`encode()`/`decode()` accept any object exposing `__cuda_array_interface__`
version 3 (CuPy arrays, contiguous PyTorch CUDA tensors, etc.), not just NumPy
arrays. Passed a device-resident input, libpressio constructs the underlying
`pressio_data` directly in the `"cudamalloc"` domain — no host round-trip — and
`encode()`/`decode()` hand back a `PressioDataCuda` (itself exposing
`__cuda_array_interface__`) instead of a NumPy array:

```python
import cupy as cp

data = cp.random.rand(256, 256).astype(cp.float32)   # already on the GPU

comp = lp.PressioCompressor.from_config({
    "compressor_id": "fzgpumodules",
    "early_config": {
        "fzgpumodules:stages":      ["lorenzo:float:uint16", "rze"],
        "fzgpumodules:connections": ["s1 <- s0:codes"],
    },
    "compressor_config": {"pressio:abs": 1e-3},
})

compressed   = comp.encode(data)                      # stays on the GPU
out          = cp.empty_like(data)
decompressed = comp.decode(compressed, out)            # stays on the GPU too
```

Passing a device-resident `out` to `decode()` is what keeps the result on the
GPU — passing `None` or a NumPy array (as in Quick Start above) still copies
back to host, unchanged from before. `test/test_fzgm_python.py`'s
GPU-resident-round-trip checks cover this without requiring CuPy (a minimal
`__cuda_array_interface__` wrapper via `ctypes` + `libcudart` stands in).

Constraints: only contiguous, unmasked, interface-version-3 arrays — strided or
masked device arrays raise `NotImplementedError`. Compression still stages
through the plugin's own `cudamalloc` domain either way; a *different* CUDA
context's pointer, or a pointer libpressio's domain manager can't otherwise
reach, would fail rather than silently copy.

---

## from_config Structure

`PressioCompressor.from_config` takes a dict with three sections:

```python
comp = lp.PressioCompressor.from_config({
    "compressor_id": "fzgpumodules",

    # early_config: applied BEFORE the options template is built.
    # Put stages, connections, and graph_mode here so per-stage option keys
    # (fzgpumodules:s0:...) are registered when compressor_config runs.
    "early_config": {
        "fzgpumodules:stages":      [...],
        "fzgpumodules:connections": [...],
        "fzgpumodules:graph_mode":  False,       # optional, default False
        "fzgpumodules:metric":      "composite", # optional
        "composite:plugins":        ["size", "error_stat"],
    },

    # compressor_config: applied after template; per-stage keys are safe here.
    "compressor_config": {
        "pressio:abs":                  1e-3,
        "fzgpumodules:memory_strategy": "minimal",
        "fzgpumodules:s0:quant_radius": 32768,
        # ...
    },
})
```

Per-stage keys like `fzgpumodules:s0:quant_radius` do not exist until the stages list has been
processed. Anything that creates those keys must go in `early_config`.

---

## Encode and Decode

```python
compressed   = comp.encode(data)              # returns a numpy byte array
decompressed = comp.decode(compressed, data.copy())
```

`decode` needs a pre-allocated output buffer (second argument) to know the output shape and dtype:

```python
out = np.empty_like(data)
decompressed = comp.decode(compressed, out)
```

---

## Pipeline Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `pressio:abs` | double | 1e-3 | Absolute error bound |
| `pressio:rel` | double | 1e-3 | Relative error bound (corresponds to noa to fzgpumodules) |
| `fzgpumodules:error_bound_mode` | str | `"abs"` | `"abs"`, `"rel"`, `"noa"`, or `"prel"`. On predictor stages `"rel"` warns and maps to `"prel"`; only `QuantizerStage` honours exact `"rel"`. |
| `fzgpumodules:stages` | list[str] | `["lorenzo:float:uint16", "diff:uint16"]` | Ordered stage tokens |
| `fzgpumodules:connections` | list[str] | `["s1 <- s0:codes"]` | Stage wiring strings |
| `fzgpumodules:dims` | list[int] | `[0, 1, 1]` | Spatial dims `[nx, ny, nz]`; `nx=0` infers 1-D |
| `fzgpumodules:memory_strategy` | str | `"minimal"` | `"minimal"` or `"preallocate"` |
| `fzgpumodules:memory_multiplier` | float | 3.0 | GPU pool size multiplier |
| `fzgpumodules:num_streams` | int | 1 | Parallel CUDA streams |
| `fzgpumodules:graph_mode` | bool | False | CUDA graph capture (see below) |
| `fzgpumodules:fusion` | str | `"off"` | `"off"` or `"auto"` — kernel fusion (see below) |
| `fzgpumodules:config_file` | str | `""` | Path to TOML pipeline config file (see below) |
| `fzgpumodules:expose_stage_outputs` | bool | False | Expose terminal stage outputs as metrics after `encode` |

### Error bound modes

| Value | Meaning |
|-------|---------|
| `ABS` | Absolute error — `abs(x_orig - x_recon) ≤ eb` |
| `REL` | Exact point-wise relative bound, `abs(error) / abs(x_orig) ≤ eb` — honoured exactly only by `QuantizerStage`. On stages that quantize a fused prediction residual against one global tolerance (`lorenzo:float:*`/`lorenzo:double:*`, i.e. `LorenzoQuantStage`) it is a deprecated alias for `PREL`: the engine logs a warning and resolves it to `PREL` rather than failing, since it cannot honour an exact per-element bound there. |
| `NOA` | Value-range relative — `abs(error) / value_range ≤ eb` (norm-of-absolute), uses `pressio:abs` as the fraction |
| `PREL` | Pseudo-relative — `abs_eb = eb * max(abs(data))`, applied as a single absolute bound, uses `pressio:abs` as the fraction. This is the honest name for what `REL` silently did on fused prediction stages before the REL/PREL split; prefer it explicitly there to silence the deprecation warning. |

Note: `pressio:abs` doubles as the bound fraction for `NOA` and `PREL` (both are ABS-family
bounds under the hood); `pressio:rel` is only read when `error_bound_mode = "rel"`. See
[Fast and Effective Lossy Compression on GPUs and CPUs with Guaranteed Error Bounds](https://doi.org/10.1109/IPDPS64566.2025.00083)
for details on the error bound definitions and their implications for compression.

### Connections format

```python
"fzgpumodules:connections": [
    "s1 <- s0",        # connect default output of s0 → input of s1
    "s1 <- s0:codes",  # connect the :codes port of s0 → input of s1
]
```

Stage IDs are assigned left-to-right from the `stages` list: `s0`, `s1`, `s2`, …
Unconnected stage outputs become pipeline outputs and are included in the compressed buffer
automatically.

---

## Stage Tokens

Each entry in `fzgpumodules:stages` is a token of the form `<kind>[:<type>[:<type2>]]`.
Per-stage parameters use the key `fzgpumodules:<sid>:<param>` where `<sid>` is `s0`, `s1`, etc.

### Lorenzo Predictor + Quantizer

**Quantizing variants** (lossy, float/double input):

```python
"lorenzo:float:uint16"   # float → uint16 codes
"lorenzo:float:uint8"    # float → uint8
"lorenzo:double:uint16"  # double → uint16
"lorenzo:double:uint32"  # double → uint32
```

Per-stage options (prefix `fzgpumodules:sN:`):

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `quant_radius` | int | 32768 | Bin count / 2 |
| `outlier_capacity` | float | 0.2 | Outlier buffer as fraction of N |
| `zigzag_codes` | bool | False | Zigzag-encode codes for better RLE/entropy |
| `value_base` | float | 0.0 | Pre-scanned value range (NOA) or max(abs(data)) (PREL); 0 = auto-scan |
| `centering` | bool | False | Per-tile mean centering (FSZ): predict each 1024-element tile's first element from the tile mean instead of 0. Helps fields with a large constant offset (temperature in Kelvin, pressure in hPa). 1-D only. |

**Integer variants** (lossless, no per-stage options):

```python
"lorenzo:int8", "lorenzo:int16", "lorenzo:int32", "lorenzo:int64"
```

Typically followed by `zigzag:intN` + `bitpack:uintN`.

### Standalone Quantizer

```python
"quantizer:float:uint16", "quantizer:float:uint32"
"quantizer:double:uint16", "quantizer:double:uint32"
```

All Lorenzo options above, plus:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `outlier_threshold` | float | inf | `abs(x) >= threshold` stored losslessly |
| `inplace_outliers` | bool | False | Inline outliers in code array (requires `zigzag_codes=True` and `quantizer:float:uint32`) |
| `linear_mode` | bool | False | ABS/NOA: no-outlier cuSZp-style raw signed codes, no radius clamp, no zigzag. Mutually exclusive with `rel`, `inplace_outliers`, `zigzag_codes`. A value whose bin exceeds the signed code range raises a clean exception at compress time rather than silently wrapping — widen the code type, center the data first, or use an outlier-capable quantizer instead. |
| `dither` | bool | False | Reconstruct to a deterministic pseudo-random point within the error-bound interval instead of the bin center, decorrelating reconstruction error from the signal. Elements that would violate the bound are escalated to lossless outliers — raise `outlier_capacity` accordingly (dither commonly escalates ~25% of elements at `dither_strength=1.0`). Mutually exclusive with `linear_mode`, `inplace_outliers`. |
| `dither_seed` | int | 0 | Seed for the deterministic dither offset; persisted in the header |
| `dither_strength` | float | 1.0 | Dither amplitude as a fraction of the error bound, in `(0, 1]`; `0.0` disables the offset |

### Difference Stage

```python
# Single-type:
"diff:float", "diff:double", "diff:uint8", "diff:uint16", "diff:uint32",
"diff:int32", "diff:int64"

# Negabinary-fused (int → uint):
"diff:int8:uint8", "diff:int16:uint16", "diff:int32:uint32", "diff:int64:uint64"
```

Note: `diff:int8` and `diff:int16` (same-type signed) are not available in the v2.0 library.

### Zigzag and Negabinary Transforms

```python
"zigzag:int8", "zigzag:int16", "zigzag:int32", "zigzag:int64"     # signed → unsigned
"negabinary:int8", "negabinary:int16", "negabinary:int32", "negabinary:int64"
```

No per-stage options.

### Run-Length Encoding (RLE)

```python
"rle:uint8", "rle:uint16", "rle:uint32", "rle:int32"
```

No per-stage options.

### Bitpacking

```python
"bitpack:uint8", "bitpack:uint16", "bitpack:uint32"
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `nbits` | int | 0 | Bits per element; 0 = full width; must be a power of 2 ≤ element bits |

Valid `nbits` values: `uint8` → 1/2/4/8; `uint16` → 1/2/4/8/16; `uint32` → 1/2/4/8/16/32.

### Bitshuffle

```python
"fzgpumodules:stages": ["bitshuffle"]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `element_width` | int | 4 | Element size in bytes: 1, 2, 4, or 8 |
| `block_size` | int | 16384 | Chunk size in bytes; must be multiple of `1024 * element_width` |

`element_width` must match the actual dtype of the incoming data (e.g. 2 for uint16, 4 for float32).

### RZE zero-word reducer

```python
"fzgpumodules:stages": ["rze"]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `chunk_size` | int | 16384 | Chunk size in bytes; one of 4096, 8192, 16384 |
| `word_size` | int | 1 | Word granularity 1/2/4/8 = LC RZE_1/2/4/8 |

Incompatible with `graph_mode=True`.

### Tiled Lorenzo (dimension-aware, lossless)

```python
"tiled_lorenzo:int16", "tiled_lorenzo:int32"
```

Separable Lorenzo predictor over fixed-size tiles (8x8 in 2-D, 4x4x4 in 3-D by default),
emitted tile-major so a downstream `adaptive_bitpack` block aligns one block per tile.
Signed integer input (quantizer codes or block deltas) only.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `tile_x` | int | 0 | Tile extent in x (fast dim); 0 = ndim-derived default |
| `tile_y` | int | 0 | Tile extent in y; 0 = default (1 for 1-D) |
| `tile_z` | int | 0 | Tile extent in z; 0 = default (1 for 1-D/2-D) |

`tile_x*tile_y*tile_z` must be in `[1, 1024]`, each extent in `[1, 255]`. Set
`fzgpumodules:dims` for 2-D/3-D data — leaving it at the default `[0,1,1]` treats the
input as flat 1-D, which is valid but won't exploit 2-D/3-D locality.

### Adaptive Bitpack

```python
"adaptive_bitpack:int16", "adaptive_bitpack:int32"
```

Per-block adaptive fixed-rate bit-plane coder (the cuSZp lossless back-end's "plain"
mode). Pair with `quantizer` (`linear_mode=True`) or `tiled_lorenzo` upstream.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `block_size` | int | 32 | Elements per fixed-rate block, in `[1, 1024]`. Match a `tiled_lorenzo` tile_elems upstream to align blocks to tiles. |
| `outlier_selection` | bool | False | cuSZp2 per-block plain/outlier selection: store element 0 as a raw outlier and pack only the rest, whichever is smaller |

### LC Byte-Stream Reducers (RZE, RRE, RARE, RAZE, CLOG, HCLOG)

```python
"rze", "rre", "rare", "raze", "clog", "hclog"
```

Lossless byte-stream reducers from the LC framework, all sharing the same shape:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `chunk_size` | int | 16384 | Bytes; `rze`/`rre` only support 16384, the others accept 4096/8192/16384 |
| `word_size` | int | 1 | Word granularity 1/2/4/8; `clog`/`hclog` are unsigned-words only |

`rze` is additionally incompatible with `graph_mode=True`.

### GPULZ (LZ77-style, lossless)

```python
"gpulz"
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `chunk_size` | int | 2048 | Bytes |
| `word_size` | int | 4 | Match word size in bytes |
| `match_level` | int | 1 | Encode-side effort: `0` = exact longest match over the near window only; `1` = additionally consults a hashed long-range table (trades throughput for ratio) |

The stage's `split_mode` (four-port literal/length/offset/meta output) is not exposed —
it changes the stage's output port count, which must be known before construction, and
`GPULZStage` has no constructor that takes it up front.

### TUPL (AoS→SoA transpose)

```python
"tupl"
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `block_size` | int | 16384 | Bytes; chunk over which the transpose runs |
| `word_size` | int | 1 | Bytes per tuple element |
| `dim` | int | 2 | Tuple width: elements per interleaved group |

### BitplaneRZE (fused, lossless)

```python
"bitplane_rze"
```

Fixed `uint16_t` input, no configurable options — bitplane-transpose + zero-group RZE
fused into one stage (the FZ-GPU lossless codec).

### ANS Entropy Coding

```python
"ans"
```

No configurable options in this build — the vendored dietGPU kernels only support
`prob_bits=10`. Incompatible with `graph_mode=True` (D2H copies in both directions).

### ADM — Adaptive Data Mapping

```python
"adm:uint16", "adm:uint32"
```

Assumes bounded quantization codes (small diffs from a per-block center) — see the
`local_bits overflow` note in `mapping_uint16.cu` / `mapping_uint32.cu`. Input whose
diffs exceed that capacity raises a clean exception (this used to corrupt device
memory instead, since the guard was compiled out in Release builds; fixed).

### SZx / SZp (fused lossy compressors)

```python
"szx:float", "szx:double", "szp:float", "szp:double"
```

Ultrafast block-local error-bounded compressors (SZx and SZp/fZ-light), each a
self-contained fused stage rather than a Lorenzo+quantize+code pipeline.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `block_size` | int | 128 | Elements per block, in `[1, 4096]` |

Use `fzgpumodules:error_bound_mode` / `pressio:abs` as usual — **only `abs` and `noa` are
supported** (`rel`/`prel` raise a clear error at pipeline construction, since neither
stage has a point-wise or pseudo-relative path).

### Direction-dependent port count stages

The four stages below have a port count that *swaps* with direction — more outputs than
inputs going forward (compress), and the mirror image going backward (decompress).
`Pipeline::buildInverseDAG()` reconciles this transparently; there is nothing to
configure for it. (A related but different case — a stage whose forward port count
needs to *grow* based on a config value, like `GPULZStage::setSplitMode(true)` taking
it from 1 output to 4 — is not supported, because `addRawStage()` captures the port
count once, immediately after `addStage()` returns, before the plugin's factory
function gets the pointer back to call any setter on it.)

#### AdaptiveLorenzo (lossless)

```python
"adaptive_lorenzo:int16", "adaptive_lorenzo:int32"
```

Per-tile adaptive Lorenzo predictor variant search (order-1/order-2, centered/uncentered).
Forward: 1 input -> 3 outputs (`output`, `modes`, `means`). Inverse: 3 inputs -> 1 output.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `coder_block_size` | int | 32 | Downstream coder block size — fixed at 32 by the cost model; don't change |
| `blocks_per_tile` | int | 8 | Coder blocks per adaptation tile, in `[1, 32]` (tile size <= 1024) |
| `enable_order2` | bool | True | Include order-2 (LZ2) prediction variants in the per-tile search |
| `enable_centering` | bool | True | Include centered prediction variants in the per-tile search |

Pair with `adaptive_bitpack` downstream (`block_size` = `coder_block_size`, i.e. 32).

#### LogTransform

```python
"log_transform"
```

Float32 only. Forward: 1 input -> 4 outputs. Inverse: 4 inputs -> 1 output. **Always
interprets the pipeline's error bound as an exact point-wise relative bound** regardless
of `fzgpumodules:error_bound_mode` — set `error_bound_mode=rel` and `pressio:rel` so the
value you set matches what the stage does with it.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `threshold` | float | 0.0 | `\|x\| < threshold` => lossless outlier; 0 disables (only zeros/denormals/inf/NaN escalate) |
| `outlier_capacity` | float | 0.05 | Fraction of input element count reserved for outliers |

#### GInterp

```python
"ginterp"
```

cuSZ-Hi-style interpolation predictor, float32 codes to uint16. **2-D/3-D only** —
set `fzgpumodules:dims`; 1-D is rejected. Forward: 1 input -> 4 outputs
(`codes`, `anchor`, `outlier_vals`, `outlier_idxs`). Inverse: 4 inputs -> 1 output.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `quant_radius` | int | 0 | 0 = auto-tune (scans data min/max on first execute); >0 = manual, required for CUDA graph capture |
| `outlier_capacity` | float | 0.10 | Outlier buffer reserve as fraction of total elements |
| `auto_tuning_mode` | int | 0 | 0 off, 1 cheap profiling (3-D only, ~1ms), 3 full structural (3-D only, ~5-15ms), 4 full+alpha/beta sweep (3-D only, ~15-30ms), 5 manual override (dim-agnostic, graph-safe) |
| `manual_alpha` | float | 0.0 | Manual alpha for `auto_tuning_mode=5`; 0.0 defers to the cuSZ-Hi schedule |
| `manual_beta` | float | 0.0 | Manual beta for `auto_tuning_mode=5`; 0.0 -> beta=4.0 |

Modes 1/3/4 force a host-blocking D2H sync and are incompatible with `graph_mode=True`.

#### ROIBinSplit

```python
"roibin_split:float", "roibin_split:double"
```

Splits a field into per-peak regions of interest plus a (optionally binned) background.
Forward: 1 input -> 3 outputs (`roi`, `background`, `meta`). Inverse: 3 inputs -> 1 output.
Requires `fzgpumodules:dims` (2-D/3-D).

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `peaks_file` | str | `""` | Path to a `.roi` peak-list file (magic `FZROI1`, header `nx/ny/nz/npeaks` as little-endian uint32, then `npeaks` records of `{z:uint32, x:uint16, y:uint16}` = 8 bytes each). **Required** — compression throws if unset. Its geometry must match `fzgpumodules:dims` if both are set. |
| `roi_half_width` | int | 4 | ROI box half-width in pixels; the box is `(2*hw+1)^2` |
| `bin_factor` | int | 1 | Background binning factor; 1 disables binning |

### Huffman Entropy Coding

```python
"huffman:uint8", "huffman:uint16", "huffman:uint32"
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `bklen` | int | 0 (= type's built-in default: 256 for uint8, 1024 for uint16/uint32) | Codebook length; all input symbols must be in `[0, bklen)` |
| `device_resident` | bool | False | Forward execution path: False = HostCoordinated (cuSZ coarse path), True = DeviceResident (scan/header assembly stays on GPU) |
| `validate_symbol_range` | bool | True | Verify every symbol is in `[0, bklen)` on the GPU when a codebook is pinned. Safe to disable only when the range is guaranteed upstream (e.g. Lorenzo/Quantizer zigzag codes with `bklen == 2 * quant_radius`). |
| `adaptive_book` | bool | False | Codebook source: False = PerBlock (fresh histogram+book every call), True = Adaptive (histogram the first call only, reuse forever). `Fixed` book source is not exposed — it needs a caller-supplied frequency table with no scalar option equivalent. |

When following a Lorenzo/Quantizer stage with `zigzag_codes=True` and radius `r`, set `bklen = 2 * r`.
Incompatible with `graph_mode=True` (two D2H syncs per forward call).

`bklen` values close to 65536 raise a clean error rather than crash — `bklen_` is
`uint16_t` end to end (serialized header and histogram kernel API), and the histogram
kernel also needs `(bklen+1)*4` bytes of shared memory for one privatized replica, which
caps the practical maximum well below 65536 on most devices (~58000 on an H100).
`setBklen()` and the histogram optimizer both validate and throw instead of launching a
kernel that would corrupt memory or fault. Keep `bklen` well under 65536, or size the
upstream `quant_radius` so codes fit the default `bklen` (1024 for uint16).

---

## Metrics

Read metrics after each `encode` call:

```python
metrics = comp.get_metrics()

# Plugin-specific:
peak_mem   = metrics.get("fzgpumodules:peak_memory",         None)  # bytes
exec_us    = metrics.get("fzgpumodules:execution_time_us",   None)  # microseconds
rebuilt    = metrics.get("fzgpumodules:rebuilt",              None)  # bool
n_outliers = metrics.get("fzgpumodules:s0:outlier_count",    None)  # Lorenzo/Quantizer

# Composite metrics (requires "size" and "error_stat" in composite:plugins):
cr      = metrics.get("size:compression_ratio", None)
max_err = metrics.get("error_stat:max_error",   None)
```

To enable size and error metrics:

```python
"early_config": {
    "fzgpumodules:metric":   "composite",
    "composite:plugins":     ["size", "error_stat"],
    # ... stages, connections ...
},
```

Full metrics reference:

| Key | Type | Description |
|-----|------|-------------|
| `fzgpumodules:peak_memory` | int | Peak GPU memory in bytes |
| `fzgpumodules:execution_time_us` | int | Stream-synced device execution time (microseconds). Excludes host<->device staging and pipeline (re)build cost — see `rebuilt` below |
| `fzgpumodules:rebuilt` | bool | True if this `encode()` call (re)built the pipeline (first call, dirty options, or a changed input size in manual-pipeline mode) rather than reusing the cached one. `execution_time_us` excludes build time regardless — in a timing loop, discard measurements where this is true, or warm up first |
| `fzgpumodules:sN:outlier_count` | int | Outlier count for stage N (Lorenzo/Quantizer) |
| `size:compression_ratio` | float | Uncompressed / compressed size |
| `size:compressed_size` | int | Compressed size in bytes |
| `error_stat:max_error` | float | Maximum pointwise error |

---

## Common Recipes

### Lorenzo + RLE (default)

```python
comp = lp.PressioCompressor.from_config({
    "compressor_id": "fzgpumodules",
    "early_config": {
        "fzgpumodules:stages":      ["lorenzo:float:uint16", "rle:uint16"],
        "fzgpumodules:connections": ["s1 <- s0:codes"],
    },
    "compressor_config": {"pressio:abs": 1e-4},
})
```

### Lorenzo + RZE (best ratio on smooth data)

```python
comp = lp.PressioCompressor.from_config({
    "compressor_id": "fzgpumodules",
    "early_config": {
        "fzgpumodules:stages":      ["lorenzo:float:uint16", "rze"],
        "fzgpumodules:connections": ["s1 <- s0:codes"],
    },
    "compressor_config": {"pressio:abs": 1e-4},
})
```

### Lorenzo + Bitshuffle

```python
comp = lp.PressioCompressor.from_config({
    "compressor_id": "fzgpumodules",
    "early_config": {
        "fzgpumodules:stages":      ["lorenzo:float:uint16", "bitshuffle"],
        "fzgpumodules:connections": ["s1 <- s0:codes"],
    },
    "compressor_config": {
        "pressio:abs":                   1e-4,
        "fzgpumodules:s1:element_width": 2,   # uint16 = 2 bytes
    },
})
```

### Quantizer with Inplace Outliers (float32 only)

```python
comp = lp.PressioCompressor.from_config({
    "compressor_id": "fzgpumodules",
    "early_config": {
        "fzgpumodules:stages":      ["quantizer:float:uint32", "rle:uint32"],
        "fzgpumodules:connections": ["s1 <- s0:codes"],
    },
    "compressor_config": {
        "pressio:abs":                      1e-4,
        "fzgpumodules:s0:zigzag_codes":     True,
        "fzgpumodules:s0:inplace_outliers": True,
    },
})
```

### Lossless Integer Lorenzo

```python
codes = np.array([...], dtype=np.int32)

comp = lp.PressioCompressor.from_config({
    "compressor_id": "fzgpumodules",
    "early_config": {
        "fzgpumodules:stages":      ["lorenzo:int32", "zigzag:int32", "bitpack:uint32"],
        "fzgpumodules:connections": ["s1 <- s0", "s2 <- s1"],
    },
    "compressor_config": {},
})

compressed   = comp.encode(codes)
decompressed = comp.decode(compressed, codes.copy())
assert np.array_equal(codes, decompressed)
```

### Standalone Fused Lossy Compressor (SZx)

```python
comp = lp.PressioCompressor.from_config({
    "compressor_id": "fzgpumodules",
    "early_config": {
        "fzgpumodules:stages":      ["szx:float"],
        "fzgpumodules:connections": [],
    },
    "compressor_config": {"pressio:abs": 1e-3},
})
```

### Fusion-Enabled Pipeline

```python
comp = lp.PressioCompressor.from_config({
    "compressor_id": "fzgpumodules",
    "early_config": {
        "fzgpumodules:stages":      ["lorenzo:float:uint16", "rze"],
        "fzgpumodules:connections": ["s1 <- s0:codes"],
        "fzgpumodules:fusion":      "auto",
    },
    "compressor_config": {"pressio:abs": 1e-4},
})
```

### 3-D Structured Grid

```python
data = np.random.rand(128, 256, 256).astype(np.float32)
nz, ny, nx = data.shape  # numpy shape is [z, y, x]; FZ wants [nx, ny, nz]

comp = lp.PressioCompressor.from_config({
    "compressor_id": "fzgpumodules",
    "early_config": {
        "fzgpumodules:stages":      ["lorenzo:float:uint16", "rle:uint16"],
        "fzgpumodules:connections": ["s1 <- s0:codes"],
    },
    "compressor_config": {
        "pressio:abs":         1e-4,
        "fzgpumodules:dims":   [nx, ny, nz],
    },
})
```

---

## CUDA Graph Mode

Graph mode eliminates CPU dispatch overhead after a one-time warmup. Use for benchmarking or
throughput-critical applications.

```python
comp = lp.PressioCompressor.from_config({
    "compressor_id": "fzgpumodules",
    "early_config": {
        "fzgpumodules:stages":      ["lorenzo:float:uint16", "rle:uint16"],
        "fzgpumodules:connections": ["s1 <- s0:codes"],
        "fzgpumodules:graph_mode":  True,
    },
    "compressor_config": {"pressio:abs": 1e-4},
})

for i in range(100):
    compressed   = comp.encode(data)           # first call: warmup + capture; rest: graph replay
    decompressed = comp.decode(compressed, data.copy())
```

**Constraints:**
- `decompress` reads from live GPU state of the most recent `compress` call; compressed bytes from
  `encode` are not used for decompression.
- Cross-machine or cross-process decompression is not supported in graph mode.
- Incompatible with the `rze` stage.
- Memory strategy is forced to `preallocate`.

---

## Kernel Fusion

Set `fzgpumodules:fusion = "auto"` to let the pipeline planner fuse eligible adjacent
stages into single kernels — lower launch overhead and less intermediate GPU traffic.
Off by default.

```python
comp = lp.PressioCompressor.from_config({
    "compressor_id": "fzgpumodules",
    "early_config": {
        "fzgpumodules:stages":      ["lorenzo:float:uint16", "rze"],
        "fzgpumodules:connections": ["s1 <- s0:codes"],
        "fzgpumodules:fusion":      "auto",
    },
    "compressor_config": {"pressio:abs": 1e-3},
})
```

Fusion may disable CUDA graph mode and buffer coloring for the fused groups (the fused
runner synchronises, and the fused kernel keeps a group's input live across the whole
group rather than the per-stage liveness coloring assumes). The environment variable
`FZ_FUSION=off|auto`, if set, overrides this option at pipeline-build time — this is the
same knob the underlying C++ `Pipeline::setFusionPolicy()` / `FZ_FUSION` env override use.

---

## Exposing Stage Outputs

Set `fzgpumodules:expose_stage_outputs = True` in `early_config` to retrieve intermediate pipeline
data as numpy arrays after `encode`. All unconnected (terminal) output ports are exposed as metrics.

```python
comp = lp.PressioCompressor.from_config({
    "compressor_id": "fzgpumodules",
    "early_config": {
        "fzgpumodules:stages":               ["lorenzo:float:uint16"],
        "fzgpumodules:connections":          [],
        "fzgpumodules:expose_stage_outputs": True,
    },
    "compressor_config": {"pressio:abs": 1e-3},
})

compressed = comp.encode(data)
metrics    = comp.get_metrics()
codes           = metrics["fzgpumodules:s0:output:codes"]           # uint16
outlier_indices = metrics["fzgpumodules:s0:output:outlier_indices"] # uint32
```

For multi-stage pipelines only unconnected outputs are exposed:

```python
# codes port is connected; outlier_indices is terminal
comp = lp.PressioCompressor.from_config({
    "compressor_id": "fzgpumodules",
    "early_config": {
        "fzgpumodules:stages":               ["lorenzo:float:uint16", "rle:uint16"],
        "fzgpumodules:connections":          ["s1 <- s0:codes"],
        "fzgpumodules:expose_stage_outputs": True,
    },
    "compressor_config": {"pressio:abs": 1e-3},
})
comp.encode(data)
metrics = comp.get_metrics()
# fzgpumodules:s0:output:outlier_indices  (uint32)
# fzgpumodules:s1:output:output           (uint8 RLE bytes)
```

### Stage output port names

| Stage | Output port(s) | dtype |
|-------|----------------|-------|
| `lorenzo:float:*`, `lorenzo:double:*` | `codes`, `outlier_indices` (+ `means` if `centering=True`) | code type, uint32 (+ float/double) |
| `lorenzo:intN` (lossless) | `output` | same as input |
| `tiled_lorenzo:*`, `adaptive_bitpack:*` | `output` | uint8 (adaptive_bitpack), same as input (tiled_lorenzo) |
| `quantizer:*` | `codes`, `outlier_idxs` | code type, uint32 |
| `diff:*`, `zigzag:*`, `negabinary:*`, `adm:*` | `output` | same as output type |
| `rle:*`, `bitpack:*`, `bitshuffle`, `rze`, `rre`, `rare`, `raze`, `clog`, `hclog`, `gpulz`, `tupl`, `bitplane_rze`, `ans`, `huffman:*` | `output` | uint8 |
| `szx:*`, `szp:*` | `output` | uint8 |

Note: `quantizer` uses `outlier_idxs`; quantizing Lorenzo uses `outlier_indices`.

Not available in config-file mode.

---

## TOML Config File

Load a pipeline from a TOML file instead of specifying stages and connections inline. The file
controls stages, connections, dims, and error bounds.

```python
comp = lp.PressioCompressor.from_config({
    "compressor_id": "fzgpumodules",
    "compressor_config": {
        "fzgpumodules:config_file": "/path/to/my_pipeline.toml",
        "fzgpumodules:graph_mode": False,   # graph_mode and num_streams still apply on top
    },
})
```

No `early_config` is needed — the TOML file bootstraps per-stage key registration.

Example TOML:

```toml
[pipeline]
dims       = [256, 256, 1]
input_size = 262144       # 256*256*4 bytes (float32)

[[stage]]
name             = "lorenzo"
type             = "LorenzoQuant"
input_type       = "float32"
code_type        = "uint16"
error_bound      = 1e-3
error_bound_mode = "ABS"
quant_radius     = 32768
outlier_capacity = 0.1

[[stage]]
name   = "rle"
type   = "RLE"
inputs = [{ from = "lorenzo", port = "codes" }]
```

See \ref config_file_overview "Config file reference" for the full TOML format and stage type names.

Per-stage outlier count metrics (`fzgpumodules:sN:outlier_count`) are not available in config-file mode.

---

## Error Handling

`encode` and `decode` raise `libpressio.PressioException` on failure:

```python
try:
    compressed = comp.encode(data)
except lp.PressioException as e:
    print(f"Compression failed: {e}")
```

Common causes:
- Stage token not recognized (typo or unsupported type combination).
- `inplace_outliers=True` with a stage other than `quantizer:float:uint32`.
- `graph_mode=True` combined with the `rze` stage.
- Input dtype does not match what the first pipeline stage expects.
