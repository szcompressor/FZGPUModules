# Pipeline Configuration Files {#config_file_overview}

**Status:** Implemented (v2.0)

Human-readable TOML files that fully describe a compression pipeline: the DAG
(topology), stage types and parameters, and pipeline-level settings. A config
file can reconstruct an identical pipeline without writing any C++ code.

---

## API

### Methods

```cpp
// Build and finalize from a config file.
// Throws std::runtime_error on parse errors, unknown stage types, or bad wiring.
// Equivalent to manually calling addStage() + connect() + finalize().
void Pipeline::loadConfig(const std::string& path);

// Serialize the current (finalized) pipeline to a config file.
// The file can be passed back to loadConfig() to reconstruct an equivalent pipeline.
// Throws std::runtime_error if the pipeline is not finalized.
void Pipeline::saveConfig(const std::string& path) const;

// Constructor overload -- delegates to the default constructor + loadConfig().
// The pipeline is finalized on return.
explicit Pipeline::Pipeline(const std::string& config_path);
```

### Usage patterns

**Load a config and compress data:**

For best results -- especially when using `memory_strategy = "PREALLOCATE"` --
pass the input size to the constructor before calling `loadConfig()`. This lets
`finalize()` size buffers correctly rather than relying on a 1-byte placeholder.

```cpp
// Recommended: pass input size so PREALLOCATE buffers are correctly sized
fz::Pipeline pipeline(input_bytes);
pipeline.loadConfig("my_compressor.toml");  // calls finalize() internally

void* d_compressed = nullptr;
size_t compressed_sz = 0;
pipeline.compress(d_input, input_bytes, &d_compressed, &compressed_sz, stream);
```

Alternatively, the single-argument constructor can be used when `MINIMAL`
strategy is sufficient and pool sizing from the .toml is acceptable:

```cpp
fz::Pipeline pipeline("my_compressor.toml");  // finalized on return
pipeline.compress(d_input, input_bytes, &d_compressed, &compressed_sz, stream);
```

> [!IMPORTANT]
> When using `memory_strategy = "PREALLOCATE"` (required for CUDA Graph capture),
> always use the constructor + `loadConfig()` pattern so the pipeline receives
> the real `input_bytes` before `finalize()` runs preallocations.

**Build programmatically, then save for later reuse:**

```cpp
fz::Pipeline pipeline(input_bytes, fz::MemoryStrategy::PREALLOCATE);

auto* lrz = pipeline.addStage<fz::LorenzoQuantStage<float, uint16_t>>();
lrz->setErrorBound(1e-4f);
lrz->setQuantRadius(32768);
lrz->setOutlierCapacity(0.10f);
lrz->setZigzagCodes(true);

auto* bs = pipeline.addStage<fz::BitshuffleStage>();
bs->setBlockSize(16384);
bs->setElementWidth(sizeof(uint16_t));
pipeline.connect(bs, lrz, "codes");

auto* rze = pipeline.addStage<fz::RZEStage>();
rze->setChunkSize(16384);
rze->setWordSize(1);
pipeline.connect(rze, bs);

pipeline.finalize();
pipeline.saveConfig("my_compressor.toml");
```

**Load an existing config and update a parameter before reuse:**
Not supported -- `loadConfig()` calls `finalize()` internally, and finalized
pipelines are immutable. Edit the .toml file directly to change parameters.

---

## TOML Schema

A config file has one `[pipeline]` table and one or more `[[stage]]` entries
(an array of tables).

### [pipeline] -- pipeline-level settings

All keys are optional. Absent keys use the pipeline constructor defaults.

| Key | Type | Default | Description |
|---|---|---|---|
| input_size | integer | 0 | Input buffer size hint in bytes. Used for pool sizing at finalize(). |
| dims | array of 3 integers | [0, 1, 1] | Spatial dimensions [x, y, z]. x=0 means infer from input_size. Used by LorenzoND kernels. |
| memory_strategy | string | "MINIMAL" | "MINIMAL" or "PREALLOCATE". |
| pool_multiplier | float | 3.0 | Pool capacity = input_size x pool_multiplier. Relevant for PREALLOCATE. |
| num_streams | integer | 1 | Number of CUDA streams for multi-stream execution. |
| primary_source | string | "" (unset) | Stage `name` whose inverse output `decompress()` returns. Only needed when `__external__` (see below) creates more than one source stage; unset uses the sole/first-discovered source. |

### [[stage]] -- one entry per stage

Stages are processed in file order. Each [[stage]] table describes one node
in the pipeline DAG.

**Required keys (all stages):**

| Key | Type | Description |
|---|---|---|
| name | string | A unique local identifier used in inputs[].from references. |
| type | string | Stage class to instantiate (see Stage Types below). |

**Optional key (non-source stages):**

| Key | Type | Description |
|---|---|---|
| inputs | array of inline tables | Upstream connections. Each element is { from = "<name>" } or { from = "<name>", port = "<output_name>" }. Stages with no inputs key are pipeline sources. |

If port is omitted it defaults to "output" (the single-output port name for all
stages except Lorenzo, which uses named ports "codes", "outlier_errors",
"outlier_indices", and "outlier_count").

`{ from = "__external__" }` is a reserved entry, not a stage reference: it
binds the pipeline's raw input directly to this port
(`Pipeline::bindExternalInput()`), even when the same stage's `inputs` array
also has a normal `{ from = "<name>" }` entry at another position — needed by
[Cdf97OutlierCorrect](#cdf97outliercorrect) below, whose raw-field port and
codes port come from two different places. Position in the array matters,
same as for a real connection. When this creates more than one source stage
in the pipeline, set `[pipeline].primary_source = "<name>"` to say which
stage's inverse output `decompress()` should return (see the
[SPERR pipeline example](#sperr-pipeline-bound-guaranteed) below).

---

## Stage Types

The `type` string in a `[[stage]]` table selects the stage class. Each stage's
settings, ports, and constraints — including every key accepted in its
`[stage.settings]` table — are documented on that stage's own reference page;
the TOML key for a setting is the snake_case form of its C++ setter (for
example `setBlockSize(32)` is `block_size = 32`).

| TOML `type` | Stage class | Reference |
|---|---|---|
| `Lorenzo` | `LorenzoStage` | \ref stage_lorenzo "LorenzoStage" |
| `LorenzoQuant` | `LorenzoQuantStage` | \ref stage_lorenzo_quant "LorenzoQuantStage" |
| `AdaptiveLorenzo` | `AdaptiveLorenzoStage` | \ref stage_adaptive_lorenzo "AdaptiveLorenzoStage" |
| `TiledLorenzo` | `TiledLorenzoStage` | \ref stage_tiled_lorenzo "TiledLorenzoStage" |
| `Difference` | `DifferenceStage` | \ref stage_diff "DifferenceStage" |
| `Quantizer` | `QuantizerStage` | \ref stage_quantizer "QuantizerStage" |
| `Zigzag` | `ZigzagStage` | \ref stage_zigzag "ZigzagStage" |
| `Negabinary` | `NegabinaryStage` | \ref stage_negabinary "NegabinaryStage" |
| `ADM` | `ADMStage` | \ref stage_adm "ADMStage" |
| `LogTransform` | `LogTransformStage` | \ref stage_log_transform "LogTransformStage" |
| `Bitshuffle` | `BitshuffleStage` | \ref stage_bitshuffle "BitshuffleStage" |
| `TUPL` | `TUPLStage` | \ref stage_tupl "TUPLStage" |
| `RLE` | `RLEStage` | \ref stage_rle "RLEStage" |
| `RZE` | `RZEStage` | \ref stage_rze "RZEStage" |
| `RRE` | `RREStage` | \ref stage_rre "RREStage" |
| `RARE` | `RAREStage` | \ref stage_rare "RAREStage" |
| `RAZE` | `RAZEStage` | \ref stage_raze "RAZEStage" |
| `CLOG` | `CLOGStage` | \ref stage_clog "CLOGStage" |
| `HCLOG` | `HCLOGStage` | \ref stage_hclog "HCLOGStage" |
| `Bitpack` | `BitpackStage` | \ref stage_bitpack "BitpackStage" |
| `AdaptiveBitpack` | `AdaptiveBitpackStage` | \ref stage_adaptive_bitpack "AdaptiveBitpackStage" |
| `Huffman` | `HuffmanStage` | \ref stage_huffman "HuffmanStage" |
| `ANS` | `ANSStage` | \ref stage_ans "ANSStage" |
| `GPULZ` | `GPULZStage` | \ref stage_gpulz "GPULZStage" |
| `GInterp` | `GInterpStage` | \ref stage_ginterp "GInterpStage" |
| `BitplaneRZE` | `BitplaneRZEStage` | \ref stage_bitplane_rze "BitplaneRZEStage" |
| `SZx` | `SZxStage` | \ref stage_szx "SZxStage" |
| `Merge` | `MergeStage` | \ref stage_merge "MergeStage" |
| `ROIBinSplit` | `ROIBinSplitStage` | \ref stage_roibin_split "ROIBinSplitStage" |

> **Legacy / experimental types.** `type = "SZp"` still loads (and re-saves) for
> backward compatibility with configs and archives written before SZp was
> quarantined as an experimental reference compressor, but it is not a supported
> module and is absent from the catalog above. New pipelines can use the
> SZp-inspired modular composition in `examples/presets/szp_composed.toml`
> (`Quantizer(linear) → Lorenzo(block_size=128) → AdaptiveBitpack(block_size=128)`).

For a cuSZp-style `Quantizer` with a strict requested bound, set
`linear_mode = true` and `linear_high_precision = true`. The optional
`power_of_two_bound = true` rounds the resolved ABS/NOA/PREL absolute bound
downward to a power of two; it is therefore a tighter, separately labelled
rate-distortion configuration. See the Quantizer reference for constraints and
effective-bound semantics.

### CDF97

CDF 9/7 biorthogonal wavelet transform (SPERR's DWT front-half). Lossless,
invertible, size-preserving `float -> float` basis change -- not itself
lossy, feeds a `Quantizer`. No `inputs` key needed when it's the pipeline's
only true source (the common case).

| Key | Type | Default | Description |
|---|---|---|---|
| data_type | string | "float32" | "float32" or "float64". `float64` reproduces SPERR's coefficients bit-for-bit. |

### SPECK2D

GPU-parallel "wavefront" SPECK-like bit-plane coder (2-D only). Takes signed
`int32` codes -- the same convention `Quantizer`'s `linear_mode = true`
emits.

No tunable keys -- dims, threshold, and the tree/magnitude split point are
all pipeline/data-derived.

### Cdf97OutlierCorrect

Sparse exact outlier correction: turns `CDF97`+`Quantizer`'s reported error
bound into an actually guaranteed one (quantizing DWT coefficients alone
does not bound the reconstructed field's pointwise error -- see
[stage_outlier_correct](stages/outlier_correct.md)). Needs the raw field
bound to one input port via `{ from = "__external__" }` and the paired
`Quantizer`'s codes on the other, in that order (see the `inputs` note
above) -- see the [SPERR pipeline example](#sperr-pipeline-bound-guaranteed)
below.

| Key | Type | Default | Description |
|---|---|---|---|
| error_bound | float | 1e-4 | MUST equal the paired `Quantizer` stage's own `error_bound` exactly (ABS mode only -- `rel_range`/NOA bounds must be converted to absolute externally before setting this). |

---

## Complete Examples

### Lorenzo-based pipeline (ABS error)

Lorenzo predictor with zigzag codes feeding into Bitshuffle and RZE.

```toml
# my_compressor.toml
# FZGPUModules pipeline config -- float32 input, Lorenzo topology.

[pipeline]
input_size       = 25920000    # 3600 x 1800 x float32 = 12.96 MB
dims             = [3600, 1800, 1]
memory_strategy  = "PREALLOCATE"
pool_multiplier  = 4.0
num_streams      = 1

[[stage]]
name             = "lorenzo"
type             = "LorenzoQuant"
input_type       = "float32"
code_type        = "uint16"
error_bound      = 1e-4
error_bound_mode = "ABS"
quant_radius     = 32768
outlier_capacity = 0.10
zigzag_codes     = true

# Bitshuffle the codes branch from Lorenzo
[[stage]]
name          = "bshuf_codes"
type          = "Bitshuffle"
block_size    = 16384
element_width = 2
inputs = [{ from = "lorenzo", port = "codes" }]

# RZE compresses the bitshuffle output
[[stage]]
name       = "rze_codes"
type       = "RZE"
chunk_size = 16384
word_size  = 1
inputs = [{ from = "bshuf_codes" }]

# Lorenzo outlier_errors, outlier_indices, outlier_count are unconnected
# -> they become pipeline outputs stored directly in the .fzm file.
```

### PFPL pipeline (Quantizer, REL error)

The PFPL (Predictor-Free Pipeline) preset -- direct-value quantizer with
relative error bound, followed by Difference -> Bitshuffle -> RZE.
This is the examples/presets/pfpl.toml configuration.

```toml
[pipeline]
memory_strategy = "PREALLOCATE"

[[stage]]
name             = "quant"
type             = "Quantizer"
input_type       = "float32"
code_type        = "uint32"
error_bound      = 1e-4
error_bound_mode = "NOA"
quant_radius     = 32768
outlier_capacity = 0.1
zigzag_codes     = true

[[stage]]
name        = "diff"
type        = "Difference"
input_type  = "int32"
output_type = "uint32"
chunk_size  = 16384
inputs = [{ from = "quant", port = "codes" }]

[[stage]]
name          = "bshuf"
type          = "Bitshuffle"
element_width = 4
block_size    = 16384
inputs = [{ from = "diff", port = "output" }]

[[stage]]
name      = "rze"
type      = "RZE"
word_size = 1
inputs = [{ from = "bshuf", port = "output" }]
```

Load it via the CLI:

```bash
fzgmod-cli -b -c examples/presets/pfpl.toml -i data.f32
```

### SPERR pipeline (bound-guaranteed)

CDF 9/7 DWT -> Quantizer -> sparse outlier correction -> SPECK2D. Unlike the
two examples above, this pipeline has **two** source stages (`dwt`, a pure
source with no `inputs` key, and `correct`, bound to the raw input via
`{ from = "__external__" }` alongside its normal `quant` connection) --
`primary_source` says which one's reconstruction `decompress()` returns.
This is `examples/presets/sperr_gpu.toml`; see
[stage_outlier_correct](stages/outlier_correct.md) for why the pipeline
needs this stage at all (a plain `CDF97 -> Quantizer -> SPECK2D` chain does
not guarantee its reported error bound).

```toml
[pipeline]
memory_strategy = "PREALLOCATE"
pool_multiplier = 8.0
primary_source  = "correct"    # decompress() returns correct's corrected
                                # field, not dwt's own uncorrected inverse

[[stage]]
name = "dwt"
type = "CDF97"
data_type = "float32"

[[stage]]
name = "quant"
type = "Quantizer"
input_type       = "float32"
code_type        = "uint32"
error_bound      = 1e-4
error_bound_mode = "ABS"       # NOA/rel_range is NOT valid for this pipeline -- see below
linear_mode      = true
inputs = [{ from = "dwt" }]

[[stage]]
name = "correct"
type = "Cdf97OutlierCorrect"
error_bound = 1e-4              # MUST match quant's error_bound exactly
inputs = [
  { from = "__external__" },    # correct.input[0] = raw field
  { from = "quant", port = "codes" }   # correct.input[1] = codes
]

[[stage]]
name = "speck"
type = "SPECK2D"
inputs = [{ from = "correct", port = "codes" }]
```

```bash
fzgmod-cli -c examples/presets/sperr_gpu.toml \
    -i data/CLDHGH.f32 -l 3600x1800x1 -b --report --compare data/CLDHGH.f32
```

---

## Limitations

- No post-load parameter editing. Because loadConfig() calls finalize()
  internally, stages are immutable after loading. Change parameters by editing
  the .toml file.
- Supported stage types only. The factory handles exactly the types documented
  above; run `fzgmod-cli --list-stages` for the authoritative list from this
  build. Custom stages written outside the library require a manual addStage()
  / connect() / finalize() call chain (or a PR to add the type to
  `kStageRegistry` in config.cpp).
- Multi-source pipelines (`{ from = "__external__" }`, `primary_source`) are
  representable, but the `[pipeline]` table still has exactly one
  `input_size` and one `dims` triple -- every source reads the same external
  buffer (see `Pipeline::bindExternalInput()`), there is no way to express
  multiple *different* external inputs in this format.
