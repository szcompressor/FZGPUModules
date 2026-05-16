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
rze->setLevels(4);
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

---

## Stage Types

### Lorenzo1D / Lorenzo2D / Lorenzo3D

Error-bounded prediction and quantization. Dimensionality is encoded in the type
string; runtime spatial dimensions come from [pipeline].dims.

| Key | Type | Default | Description |
|---|---|---|---|
| input_type | string | "float32" | Input element type. "float32" or "float64". |
| code_type | string | "uint16" | Quantization code type. "uint8", "uint16", or "uint32". |
| error_bound | float | 1e-3 | Error bound value. Interpretation depends on error_bound_mode. |
| error_bound_mode | string | "ABS" | "ABS" (absolute), "REL" (point-wise relative), or "NOA" (value-range relative). |
| quant_radius | integer | 32768 | Quantization radius. Must match the range of code_type (e.g. 32768 for uint16). |
| outlier_capacity | float | 0.2 | Fraction of elements reserved as outlier capacity (0.0-1.0). |
| zigzag_codes | boolean | false | Zigzag-encode codes before output to improve downstream compressibility. |

**Output ports:** "codes", "outlier_errors", "outlier_indices", "outlier_count".
Ports not referenced in any downstream inputs become pipeline outputs and are
stored in the .fzm file.

### Bitshuffle

GPU bit-matrix transpose. Size-preserving; improves entropy coder performance
on integer data.

| Key | Type | Default | Description |
|---|---|---|---|
| block_size | integer | 16384 | Chunk size in bytes. Must be a positive multiple of 1024 x element_width. |
| element_width | integer | 4 | Element width in bytes: 1, 2, 4, or 8. |

### RZE

Recursive Zero-byte Elimination -- lossless byte-stream compressor operating on
Bitshuffle output.

| Key | Type | Default | Description |
|---|---|---|---|
| chunk_size | integer | 16384 | Chunk size in bytes. Must be a positive multiple of 4096. |
| levels | integer | 4 | Recursion depth 1-4. Level 1 = ZE only; levels 2-4 add RE passes. |

### RLE

Run-Length Encoding. Effective on quantization code streams with long runs of
identical values.

| Key | Type | Default | Description |
|---|---|---|---|
| data_type | string | "uint16" | Element type. One of "uint8", "uint16", "uint32", "int32". |

### Difference

First-order difference coding with optional negabinary fusion.

| Key | Type | Default | Description |
|---|---|---|---|
| input_type | string | "float32" | Input element type. |
| output_type | string | (same as input_type) | Output element type. When output_type is the unsigned counterpart of a signed input_type, negabinary encoding is fused into the forward pass. |
| chunk_size | integer | 0 | Chunk size in bytes (0 = no chunking, process whole array as one context). When > 0, differences reset at each chunk boundary, enabling parallel decompression. |

**Negabinary-fused instantiations** (when input_type != output_type):

| input_type | output_type |
|---|---|
| "int8" | "uint8" |
| "int16" | "uint16" |
| "int32" | "uint32" |
| "int64" | "uint64" |

### Zigzag

Element-wise zigzag encode/decode (signed integer -> unsigned integer of same
width).

| Key | Type | Description |
|---|---|---|
| input_type | string | Signed integer type: "int8", "int16", "int32", "int64". |
| output_type | string | Corresponding unsigned type: "uint8", "uint16", "uint32", "uint64". |

### Quantizer

Direct-value error-bounded quantizer with lossless outlier fallback. Unlike
LorenzoND, this stage quantizes input values directly (no prediction step) and
supports ABS, NOA, and REL (log-space) error bound modes.

| Key | Type | Default | Description |
|---|---|---|---|
| input_type | string | "float32" | Input element type. "float32" or "float64". |
| code_type | string | "uint32" | Quantization code type. "uint16" or "uint32". |
| error_bound | float | 1e-3 | Error bound value. Interpretation depends on error_bound_mode. |
| error_bound_mode | string | "REL" | "ABS" (absolute), "REL" (pointwise relative log-space), or "NOA" (value-range relative). |
| quant_radius | integer | 32768 | Quantization radius. |
| outlier_capacity | float | 0.05 | Fraction of elements reserved as outlier capacity (0.0-1.0). |
| zigzag_codes | boolean | true | Zigzag-encode codes before output to improve downstream compressibility. No effect in REL mode. |
| outlier_threshold | float | inf | ABS/NOA: values with |x| >= threshold are forced to lossless outlier regardless of bin. Omit (default) to disable. |
| inplace_outliers | boolean | false | ABS/NOA: encode outlier raw bits in-place in the codes array (no scatter buffers). Cannot be used with REL mode. |

**Output ports:** "codes", "outlier_vals", "outlier_idxs", "outlier_count".
In inplace-outlier mode only "codes" is produced; the other three outputs
are omitted.

> [!NOTE]
> REL mode requires a 4-byte code type ("uint32") because it stores sign +
> log-bin packed into 32 bits. Using "uint16" in REL mode will raise a runtime
> error if the bin magnitude overflows 15 bits (rare in practice for eb >= 0.01).

### Negabinary

Element-wise negabinary encode/decode (same signed/unsigned pairing as Zigzag).

| Key | Type | Description |
|---|---|---|
| input_type | string | Signed integer type. |
| output_type | string | Corresponding unsigned type. |

### Bitpack

Packs N-bit unsigned integers into a dense byte stream. Output is ceil(n * nbits / 8)
bytes -- smaller than the input when nbits < 8*sizeof(T). nbits must be a power of two.

> [!NOTE]
> nbits must fit the actual code range. If codes span more bits than nbits, the
> upper bits are silently truncated and decompression will produce wrong values.
> The combination Lorenzo (small quant_radius, zigzag_codes=true) -> Bitpack works
> well because zigzag residuals cluster near zero. Adding a Difference stage between
> Lorenzo and Bitpack does not help: unsigned difference deltas wrap across the full
> uint16 range even when source values are small, so nbits=16 (identity) is required
> to round-trip correctly through a Difference stage.

| Key | Type | Default | Description |
|---|---|---|---|
| input_type | string | "uint16" | Element type of the input codes. One of "uint8", "uint16", "uint32". |
| nbits | integer | 16 | Bits per element. Must be a power of two: 1, 2, 4, 8 for uint8; 1-16 for uint16; 1-32 for uint32. |

### Huffman

GPU Huffman entropy coding (PHF coarse-grained). Encodes a flat symbol stream
into a variable-length bitstream with an embedded self-describing header.

> [!NOTE]
> All input symbols must be in `[0, bklen)`. When pairing with Lorenzo/Quantizer
> using `zigzag_codes=true`, set `bklen = 2 * quant_radius` to cover the exact
> symbol range without over-allocating the codebook.
> HuffmanStage is not CUDA Graph compatible (two D2H syncs per forward call).

| Key | Type | Default | Description |
|---|---|---|---|
| input_type | string | "uint16" | Symbol element type. One of "uint8", "uint16", "uint32". |
| bklen | integer | 256 (uint8) / 1024 (uint16, uint32) | Codebook length. Must cover all symbols: all inputs must be in `[0, bklen)`. |

```toml
[[stage]]
name       = "huf"
type       = "Huffman"
input_type = "uint16"
bklen      = 1024
inputs     = [{from = "lrz", port = "codes"}]
```

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
type             = "Lorenzo2D"
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
levels     = 4
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
error_bound      = 1e-3
error_bound_mode = "REL"
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
element_width = 2
block_size    = 16384
inputs = [{ from = "diff", port = "output" }]

[[stage]]
name   = "rze"
type   = "RZE"
levels = 4
inputs = [{ from = "bshuf", port = "output" }]
```

Load it via the CLI:

```bash
fzgmod-cli -b -c examples/presets/pfpl.toml -i data.f32
```

---

## Limitations

- No post-load parameter editing. Because loadConfig() calls finalize()
  internally, stages are immutable after loading. Change parameters by editing
  the .toml file.
- Supported stage types only. The factory handles the types listed above
  (Lorenzo1D/2D/3D, Quantizer, Bitshuffle, RZE, RLE, Difference, Zigzag,
  Negabinary, Bitpack, Huffman). Custom stages written outside the library
  require a manual addStage() / connect() / finalize() call chain (or a PR
  to add the type to the dispatch table in config.cpp).
- Single-source pipelines only. The [pipeline] table has one input_size and
  one dims triple. Multi-source pipelines are not currently representable in
  the config format and must be constructed manually.