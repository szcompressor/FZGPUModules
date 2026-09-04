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

The libpressio `fzgpumodules` plugin itself, and the spack package
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

compressed   = comp.encode(data)              # PressioDataCuda, even for this host `data` -- see below
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
`pressio_data` directly in the `"cudamalloc"` domain — no host round-trip.

`encode()`'s return is always device-resident for `fzgpumodules`, regardless
of input. `compress_device()` (`fzgpumodules.cc`) unconditionally builds its
output `pressio_data` in the `"cudamalloc"` domain, so `comp.encode(data)`
hands back a `PressioDataCuda` — not a NumPy byte array — even when `data` is
a plain host NumPy array. `PressioDataCuda` exposes `__cuda_array_interface__`
plus `.shape`/`.dtype`; there is no `.nbytes`, so get the compressed size via
`int(np.prod(compressed.shape)) * compressed.dtype.itemsize` (dtype is
`uint8`, so this is just `compressed.shape[0]`) rather than
`compressed.nbytes`. Passing it straight into `decode()` needs no conversion —
`decode()` reads it via the same `cudamalloc` domain either way. `decode()`,
by contrast, honors the domain of the `out` buffer *you* pass in — a
device-resident `out` (CuPy, etc.) keeps the result on the GPU; a NumPy `out`
(or `None`) copies back to host, as in Quick Start above:

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

`test/test_fzgm_python.py`'s GPU-resident-round-trip checks cover this without
requiring CuPy (a minimal `__cuda_array_interface__` wrapper via `ctypes` +
`libcudart` stands in).

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
compressed   = comp.encode(data)              # returns a PressioDataCuda, not a numpy array -- see "GPU-resident data" above
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
| `fzgpumodules:connections` | list[str] | `["s1 <- s0:codes"]` | Stage wiring strings (`__external__` sentinel: see below) |
| `fzgpumodules:primary_source` | str | `""` | Stage id whose reconstruction `decode` returns; only needed with a `__external__` connection (see SPERR recipe) |
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
    "s1 <- s0",           # connect default output of s0 → input of s1
    "s1 <- s0:codes",     # connect the :codes port of s0 → input of s1
    "s2 <- __external__", # bind the pipeline's raw input directly to s2's next port
]
```

Stage IDs are assigned left-to-right from the `stages` list: `s0`, `s1`, `s2`, …
Unconnected stage outputs become pipeline outputs and are included in the compressed buffer
automatically.

`__external__` is a reserved source name (not a stage id) for
`Pipeline::bindExternalInput()` — it binds the untouched pipeline input to a
specific stage port even when that stage also has other real connections
(needed by `cdf97_outlier_correct`; see the SPERR recipe below). List it in
`connections` at the position matching the port it should occupy, same as
any other connection. When it creates more than one source stage in the
pipeline, set `fzgpumodules:primary_source` (see Pipeline Options) to say
which one's reconstruction `decode` should return.

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

### Lorenzo + RZE

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

**Constraints:**
- `decompress` reads from live GPU state of the most recent `compress` call; compressed bytes from
  `encode` are not used for decompression.
- Memory strategy is forced to `preallocate`.

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
- Input dtype does not match what the first pipeline stage expects.
