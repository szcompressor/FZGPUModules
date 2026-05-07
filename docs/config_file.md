# Pipeline Configuration Files {#config_file_overview}

**Status:** Implemented (v2.0)

Human-readable TOML files that fully describe a compression pipeline: the DAG
(topology), stage types and parameters, and pipeline-level settings. A config
file can reconstruct an identical pipeline without writing any C++ code.

---

## Relationship to .fzm Files

| Artifact | Format | Purpose | When used |
|---|---|---|---|
| .toml config | Human-readable text | Define the pipeline setup | Authoring and CI reproducibility |
| .fzm binary | Binary | Store compressed data | Runtime output of compress() / writeToFile() |

The two formats are complementary and independent. A .toml config describes
how to build the compressor; a .fzm file is what the compressor produced.
Loading a config file does not load compressed data, and writing a compressed
file does not save pipeline parameters.

---

## API

### Header

```cpp
#include "pipeline/config.h"   // pulled in automatically via compressor.h
#include "pipeline/compressor.h"
```

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

---

## Data Type Strings

Used in input_type, output_type, code_type, and data_type keys:

| String | C type |
|---|---|
| "float32" | float |
| "float64" | double |
| "uint8" | uint8_t |
| "uint16" | uint16_t |
| "uint32" | uint32_t |
| "uint64" | uint64_t |
| "int8" | int8_t |
| "int16" | int16_t |
| "int32" | int32_t |
| "int64" | int64_t |

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

## Behavior and Guarantees

### loadConfig() sequence

1. Parse the TOML file. Throws on any syntax error.
2. Apply [pipeline] settings (input size, dims, strategy, etc.).
3. Stage construction pass -- iterate [[stage]] entries in file order, call
   the appropriate addStage<T>() and configure the stage via public setters.
   Collect a name -> Stage* map.
4. Wiring pass -- for each stage with an inputs array, call connect() using
   the resolved Stage* pointers. Throws if a from name is not found.
5. Call finalize(). The pipeline is ready to compress on return.

### saveConfig() sequence

1. Throw if not finalized.
2. Build a Stage* -> name map using getName() with numeric suffixes for
   duplicate names (e.g. "Lorenzo1D", "Lorenzo1D1").
3. For each stage, emit a [[stage]] table with type string and per-type
   parameters read via public getters.
4. For each stage, emit an inputs array by scanning connections_ for entries
   where that stage is the dependent.
5. Write the TOML document to the file.

### Invariants

- loadConfig() throws if the pipeline is already finalized (is_finalized_ == true).
- saveConfig() throws if the pipeline is not yet finalized.
- finalize() is always called internally by loadConfig(); callers must not call it separately.
- A pipeline built with loadConfig() is semantically equivalent to one built with
  the corresponding manual addStage() / connect() / finalize() calls.
- The error_bound stored in the config is always the user-specified value
  (not the converted absolute bound used internally for REL/NOA modes), so
  reloading preserves the original user intent.

---

## Error Handling

All errors are reported as std::runtime_error with a descriptive message.

| Condition | Error message prefix |
|---|---|
| File not found / unreadable | "loadConfig: failed to parse \"...\"" |
| TOML syntax error | "loadConfig: failed to parse \"...\": ..." |
| No [[stage]] entries | "loadConfig: no [[stage]] entries found in \"...\"" |
| Stage missing name or type | "loadConfig: each [[stage]] must have 'name' and 'type'" |
| Duplicate stage name | "loadConfig: duplicate stage name \"...\"" |
| Unknown type string | "loadConfig: unknown stage type \"...\"" |
| Unknown type combination | "loadConfig: unsupported ... type combination" |
| from reference not found | "loadConfig: stage \"...\" references unknown stage \"...\"" |
| Called on finalized pipeline | "loadConfig: pipeline is already finalized" |
| saveConfig before finalize | "saveConfig: pipeline must be finalized first" |
| Output file unwritable | "saveConfig: cannot open \"...\" for writing" |

---

## Dependency

Toml++ v3.4.0 (vendored, header-only). It is included only in
src/pipeline/config.cpp and never leaks into public headers. Users of
fzgpumodules.h do not need to know it exists.

```
third_party/tomlplusplus/include/toml++/toml.hpp
```

The include path is added to fzgmod_compile_settings in CMakeLists.txt.
No additional linking step is required.

---

## Limitations

- No post-load parameter editing. Because loadConfig() calls finalize()
  internally, stages are immutable after loading. Change parameters by editing
  the .toml file.
- Supported stage types only. The factory handles the 10 types listed above
  (Lorenzo1D/2D/3D, Quantizer, Bitshuffle, RZE, RLE, Difference, Zigzag,
  Negabinary). Custom stages written outside the library require a manual
  addStage() / connect() / finalize() call chain (or a PR to add the type to
  the dispatch table in config.cpp).
- Single-source pipelines only. The [pipeline] table has one input_size and
  one dims triple. Multi-source pipelines are not currently representable in
  the config format and must be constructed manually.
- Not thread-safe. Consistent with Pipeline; all config I/O must happen on the
  host thread that owns the pipeline.

---

## Testing

tests/pipeline/test_config.cpp -- label config.

```bash
ctest --test-dir build -L config --output-on-failure
```

| Test | What it verifies |
|---|---|
| ConfigLoad/LorenzoOnly | Minimal 1-stage config round-trips within error bound |
| ConfigLoad/FullPipeline | 3-stage Lorenzo->Bitshuffle->RZE, PREALLOCATE, round-trips |
| ConfigLoad/ConstructorOverload | Pipeline(path) == Pipeline() + loadConfig(path) |
| ConfigLoad/AlreadyFinalized | Throws on already-finalized pipeline |
| ConfigLoad/MissingFile | Throws on nonexistent path |
| ConfigLoad/BadStageType | Throws on unknown type string |
| ConfigLoad/BadWiringRef | Throws when from name not found |
| ConfigLoad/DuplicateStageName | Throws on two stages with same name |
| ConfigLoad/AllSupportedStageTypes | All supported stage types in one chain, save+load |
| ConfigLoad/ZigzagAndNegabinary | Zigzag and Negabinary type-pair variants |
| ConfigSave/RoundTrip | Programmatic build -> save -> load -> compress within EB |
| ConfigSave/PreservesParams | Saved TOML contains correct EB, QR, block sizes, etc. |
| ConfigSave/RequiresFinalized | Throws when pipeline not yet finalized |
