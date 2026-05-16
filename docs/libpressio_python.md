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

The `fzgpumodules` plugin has not yet been merged into the upstream libpressio or
robertu94/spack_packages. Until the PRs land, use the fork which contains the package
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

Replace `<your_arch>` with your GPU's compute capability (e.g. `80` for A100, `86` for RTX 3090).

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
| `fzgpumodules:error_bound_mode` | str | `"abs"` | `"abs"`, `"rel"`, or `"noa"` |
| `fzgpumodules:stages` | list[str] | `["lorenzo:float:uint16", "diff:uint16"]` | Ordered stage tokens |
| `fzgpumodules:connections` | list[str] | `["s1 <- s0:codes"]` | Stage wiring strings |
| `fzgpumodules:dims` | list[int] | `[0, 1, 1]` | Spatial dims `[nx, ny, nz]`; `nx=0` infers 1-D |
| `fzgpumodules:memory_strategy` | str | `"minimal"` | `"minimal"` or `"preallocate"` |
| `fzgpumodules:memory_multiplier` | float | 3.0 | GPU pool size multiplier |
| `fzgpumodules:num_streams` | int | 1 | Parallel CUDA streams |
| `fzgpumodules:graph_mode` | bool | False | CUDA graph capture (see below) |
| `fzgpumodules:config_file` | str | `""` | Path to TOML pipeline config file (see below) |
| `fzgpumodules:expose_stage_outputs` | bool | False | Expose terminal stage outputs as metrics after `encode` |

### Error bound modes

| Value | Meaning |
|-------|---------|
| `ABS` | Absolute error — `abs(x_orig - x_recon) ≤ eb` |
| `REL` | Global approximate point-wise relative — `abs(error) / abs(x_orig) ≤ eb` (approximately) |
| `NOA` | Value-range relative — `abs(error) / value_range ≤ eb` (norm-of-absolute) | 

Note: `pressio:rel` is interpreted as NOA for the plugin, as it follows more semantically distinct definitions of relative error. See [Fast and Effective Lossy Compression on GPUs  and CPUs with Guaranteed Error Bounds](https://doi.org/10.1109/IPDPS64566.2025.00083) for details on the error bound definitions and their implications for compression.

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
| `value_base` | float | 0.0 | Pre-scanned value range (NOA/REL); 0 = auto-scan |

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

### Repeated Zero Elimination (RZE)

```python
"fzgpumodules:stages": ["rze"]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `chunk_size` | int | 16384 | Chunk size in bytes; must be multiple of 4096 |
| `levels` | int | 4 | Recursion depth; must be in [1, 4] |

Incompatible with `graph_mode=True`.

### Huffman Entropy Coding

```python
"huffman:uint8", "huffman:uint16", "huffman:uint32"
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `bklen` | int | 256 (uint8) / 1024 (uint16, uint32) | Codebook length; all input symbols must be in `[0, bklen)` |

When following a Lorenzo/Quantizer stage with `zigzag_codes=True` and radius `r`, set `bklen = 2 * r`.
Incompatible with `graph_mode=True` (two D2H syncs per forward call).

---

## Metrics

Read metrics after each `encode` call:

```python
metrics = comp.get_metrics()

# Plugin-specific:
peak_mem   = metrics.get("fzgpumodules:peak_memory",         None)  # bytes
exec_us    = metrics.get("fzgpumodules:execution_time_us",   None)  # microseconds
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
| `fzgpumodules:execution_time_us` | int | Compress wall time in microseconds |
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
| `lorenzo:float:*`, `lorenzo:double:*` | `codes`, `outlier_indices` | code type, uint32 |
| `lorenzo:intN` (lossless) | `output` | same as input |
| `quantizer:*` | `codes`, `outlier_idxs` | code type, uint32 |
| `diff:*`, `zigzag:*`, `negabinary:*` | `output` | same as output type |
| `rle:*`, `bitpack:*`, `bitshuffle`, `rze` | `output` | uint8 |

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

See [config_file.md](config_file.md) for the full TOML format reference and stage type names.

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