# LibPressio Python Bindings {#libpressio_python}

FZGPUModules experimentally exposes a [libpressio](https://github.com/robertu94/libpressio) plugin (`fzgpumodules`)
that lets you build and run GPU compression pipelines from Python (or C++) through libpressio. 
The plugin translates libpressio's generic compressor interface into FZGPUModules pipeline
calls, handles GPU memory management, and surfaces metrics after each compress.

Use a v2-compatible adapter and rebuild it against the installed SDK.
Public-header probes omit registrations for unavailable ADM, legacy SZp,
CDF97, SPECK2D, Cdf97OutlierCorrect, and SZx stages. The unpatched fork
requires the v2 compatibility changes before following the setup below.

The plugin metadata `fzgpumodules:optional_stages` reports availability of CDF97,
SPECK2D, wavelet correction, and SZx wrappers in the installed SDK. The v2-compatible
adapter supports both the stable catalog and this development catalog.

The adapter currently requires a prior `compress()` on the same compressor
instance before `decompress()`. Its raw compressed output does not independently
restore a fresh decoder's pipeline and stage metadata. This is an existing
adapter limitation, separate from the library's self-describing FZM file API.

---

## Setup

### Verified v2 installation

The tested path uses an installed FZGPUModules SDK plus a separately built
libpressio adapter in a dedicated Python virtual environment. Spack supplied the
host `std_compat` dependency in our validation; it did not build the SDK or adapter.

Use FZGPUModules `v2.0.0` (`01f40d5`) and the compatible adapter snapshot
`b330cbe7a125fcf79194068b823b51323335b7d7`. The prepared adapter source archive includes the Python Chain helper and
optional stage probes. As checked on 2026-10-14, the public v2.0.0 release only
attaches the library source archive, and the fork's remote `master` does not
contain the tested adapter revision. The verified local adapter archive is
`/home/exouser/release_candidates/fzgpumodules-2.0.0/libpressio-fzgmod-2.0.0-source.tar.gz`.
Public installation requires making that adapter snapshot available through
the fork or a source archive; do not substitute an unpinned `master` checkout.
Retain the source pins and archive provenance when rebuilding.

Prerequisites are a CUDA toolkit/driver, a supported C++17 compiler, CMake 3.24+,
Python with development headers and venv support, SWIG, and an installed
`std_compat` CMake package. The validated host uses Python 3.12.3, CUDA 12.9,
GCC 13.3, SWIG 4.2.0, and std_compat 0.0.22. These are tested versions, not a
claim that all older combinations are supported. See [build requirements](building.md#prerequisites).

Set local paths to the stable library source and compatible adapter snapshot:

```bash
export FZGM_PY_ROOT="$HOME/fzgm-python-v2"
export FZGM_SOURCE=/path/to/FZGPUModules-2.0.0
export LIBPRESSIO_SOURCE=/path/to/libpressio-fzgmod-2.0.0
export STD_COMPAT_CONFIG=/path/to/libstdcompat/share/std_compat/cmake
export CUDA_ROOT=/usr/local/cuda-12.9

python3 -m venv "$FZGM_PY_ROOT/venv"
"$FZGM_PY_ROOT/venv/bin/python" -m pip install \
  numpy==2.5.2 numcodecs==0.16.5 typing_extensions==4.16.0
```

These Python pins were validated on CPython 3.12, Linux x86_64. For offline
reproduction, use the matching retained wheels with `--no-index --find-links`.
Do not use `pip install libpressio` as a substitute for building this adapter.

Build and install the SDK, or point the adapter at an existing compatible install:

```bash
cmake -S "$FZGM_SOURCE" -B "$FZGM_PY_ROOT/build-fzgm" \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_CUDA_ARCHITECTURES=native -DBUILD_TESTING=OFF \
  -DCMAKE_INSTALL_PREFIX="$FZGM_PY_ROOT/fzgpumodules"
cmake --build "$FZGM_PY_ROOT/build-fzgm" --parallel 4
cmake --install "$FZGM_PY_ROOT/build-fzgm"
```

`native` targets the visible GPU. For a GPU-less build host, set the target
explicitly: `80` for A100 or `90` for H100. The prepared H100 demo SDK targets
SM 90 and must be rebuilt for an A100 host.

Install the adapter's native libraries and Python modules into explicit destinations:

```bash
FZGM_PYTHON_SITE=$("$FZGM_PY_ROOT/venv/bin/python" -c \
  'import sysconfig; print(sysconfig.get_path("purelib"))')
cmake -S "$LIBPRESSIO_SOURCE" -B "$FZGM_PY_ROOT/build-libpressio" \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
  -DLIBPRESSIO_BUILD_MODE=CORE -DBUILD_PYTHON_WRAPPER=ON \
  -DLIBPRESSIO_HAS_FZGPUMODULES=ON \
  -DLIBPRESSIO_HAS_SIZE=ON -DLIBPRESSIO_HAS_TIME=ON \
  -DCMAKE_INSTALL_PREFIX="$FZGM_PY_ROOT/libpressio" \
  -DPython3_EXECUTABLE="$FZGM_PY_ROOT/venv/bin/python" \
  -DLIBPRESSIO_PYTHON_SITELIB="$FZGM_PYTHON_SITE" \
  -DFZGPUModules_DIR="$FZGM_PY_ROOT/fzgpumodules/lib/cmake/FZGPUModules" \
  -Dstd_compat_DIR="$STD_COMPAT_CONFIG"
cmake --build "$FZGM_PY_ROOT/build-libpressio" --parallel 4
cmake --install "$FZGM_PY_ROOT/build-libpressio"
```

`LIBPRESSIO_PYTHON_SITELIB` is separate from `CMAKE_INSTALL_PREFIX`; setting only
the native prefix does not select the venv's Python installation directory.
The plugin-enabled Python install includes `fzgm_helper.Chain` automatically.

### Use the installed bindings

Run in a dedicated shell:

```bash
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH="$FZGM_PY_ROOT/libpressio/lib:$FZGM_PY_ROOT/fzgpumodules/lib:$CUDA_ROOT/lib64:${LD_LIBRARY_PATH:-}"
unset PYTHONPATH FZ_SPECIALIZE FZ_FUSION
"$FZGM_PY_ROOT/venv/bin/python" -I -c \
  'import libpressio; from fzgm_helper import Chain; print(libpressio.__file__)'
```

Select `fzgpumodules:specialization` with `off` or `auto`; the plugin has no
`fusion` alias. The ready demo environment's `env.sh` also sets these paths and
clears policy overrides. Its preflight checks the installed import/library origins,
visible GPU, helper availability, and stable catalog before presentation.

### Spack and SWIG compatibility

The earlier Spack recipe is an alternative requiring validation against the same
SDK and adapter revisions. `spack develop ... libpressio@master` alone does not
pin the v2-compatible adapter. Verify the custom package's feature variants and
selected sources before using it as a v2 installation recipe.

Our snapshot built with SWIG 4.2.0. The older blanket warning that every SWIG
version from 4.1 onward fails is too broad. If an older source/recipe fails with
`SWIG_Python_AppendOutput` argument errors, check its vendored NumPy interface
against the selected SWIG version; the former 4.0.2 workaround is not required
by this verified 4.2.0 setup.

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

For linear pipelines, the plugin's Python bindings install `fzgm_helper.Chain`,
which avoids hand-writing
`"sN <- sM:port"` connection strings and `"fzgpumodules:sN:<option>"` keys:

```python
import numpy as np
from fzgm_helper import Chain

data = np.random.rand(256, 256).astype(np.float32)

comp = (Chain()
    .add("lorenzo:float:uint16", quant_radius=999, outlier_capacity=0.25)
    .add("rze", from_port="codes", chunk_size=8192)
    .configure(specialization="auto")
    .compressor(eb=1e-3))

compressed   = comp.encode(data)
decompressed = comp.decode(compressed, data.copy())
```

The builder supports linear pipelines. For non-linear graphs (multiple
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
| `fzgpumodules:specialization` | str | `"off"` | `"off"` or `"auto"` — Pipeline Specialization (see below) |
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

## Specialization demo

The [installed demo runbook](demos/python-specialization.md) records the verified
environment, dependency pins, setup commands, live presentation sequence, and
rehearsal checks.

The adapter's `fzgpumodules:specialization` option enables Pipeline
Specialization. The former `fzgpumodules:fusion` key was removed and is rejected
with a migration message. `off` selects staged execution and `auto` installs admitted
implementations where supported. Auto can fall back; selecting the policy alone
does not establish that a specialization was installed. CUDA Graph mode is
mutually exclusive with specialization, and this demo leaves graph mode disabled.

The Python demo builds a cuSZp-style linear Quantizer → integer Lorenzo →
AdaptiveBitpack pipeline with matching 32-element blocks. The local adapter patch
adds the previously missing integer Lorenzo `block_size` option and exposes
installation diagnostics. With a patched adapter installed:

```bash
python3 scripts/libpressio_specialization_demo.py
```

For a build-tree adapter, pass `--python-path /path/to/libpressio-build/tools/swig`.
The script checks exact logical shape, finite reconstruction, maximum absolute
error against the requested bound, zero installed groups for Off, and positive
forward/inverse counts for Auto. It warms both directions before repeated timing.
Timing includes Python and host/device transfers; it is not kernel-only throughput
or a claim that Auto always wins. An optional flat f32 field can be supplied with
`--input field.f32 --error-bound 1e-3`.

After `encode()`, `get_metrics()` exposes these additional diagnostics:

| Key suffix after `fzgpumodules:` | Meaning |
|---|---|
| `specialization_forward_groups` | Installed forward group count |
| `specialization_inverse_groups` | Installed inverse count; available after first `decode()` |
| `specialization_forward_implementations` | Forward implementation names |
| `specialization_inverse_implementations` | Inverse implementation names |
| `specialization_fallback_reason` | Empty on a forward installation; otherwise the library fallback reason |

These metrics require a v2-compatible adapter.
Use the same compressor instance for encode and decode. This demo does not provide
standalone decoder restoration or independently portable archive files.

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

### Specialized Pipeline

```python
comp = lp.PressioCompressor.from_config({
    "compressor_id": "fzgpumodules",
    "early_config": {
        "fzgpumodules:stages":      ["lorenzo:float:uint16", "rze"],
        "fzgpumodules:connections": ["s1 <- s0:codes"],
        "fzgpumodules:specialization":      "auto",
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
