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

### LorenzoQuant

Fused Lorenzo predictor + error-bounded quantizer. Dimensionality is not part of
the type string -- the stage adapts to the runtime spatial dimensions from
`[pipeline].dims`.

| Key | Type | Default | Description |
|---|---|---|---|
| input_type | string | "float32" | Input element type. "float32" or "float64". |
| code_type | string | "uint16" | Quantization code type. "uint8", "uint16", or "uint32". |
| error_bound | float | 1e-3 | Error bound value. Interpretation depends on error_bound_mode. |
| error_bound_mode | string | "ABS" | "ABS" (absolute), "NOA" (value-range relative), or "PREL" (pseudo-relative, `eb × max\|data\|`). "REL" is accepted but warns and maps to "PREL" — this stage has no exact per-element relative bound. |
| quant_radius | integer | 32768 | Quantization radius. Must match the range of code_type (e.g. 32768 for uint16). |
| outlier_capacity | float | 0.2 | Fraction of elements reserved as outlier capacity (0.0-1.0). |
| zigzag_codes | boolean | false | Zigzag-encode codes before output to improve downstream compressibility. |
| centering | boolean | false | Per-tile mean centering: seed each 1024-element tile's prediction chain with the tile mean instead of 0, so its first residual is `q0 - mu` rather than the raw `q0`. **1-D only** — loading this with 2-D/3-D dims throws. Adds a "means" output port. Helps most on fields with a large constant offset; can hurt on sparse data where blocks already encode to nothing. |

**Output ports:** "codes", "outlier_errors", "outlier_indices", "outlier_count"
(plus "means" when `centering = true`).
Ports not referenced in any downstream inputs become pipeline outputs and are
stored in the .fzm file.

### Lorenzo

Plain integer Lorenzo predictor (lossless delta / prefix sum).

| Key | Type | Default | Description |
|---|---|---|---|
| data_type | string | "int32" | Signed integer element type. "int8", "int16", "int32", or "int64". |
| block_size | integer | 0 | 1-D block-local reset period. `0` = N-D inclusion-exclusion delta; `n > 0` forces the 1-D path resetting every `n` elements (cuSZp uses 32). Must be in [0, 1024]. |
| centering | boolean | false | Per-block mean centering. Requires `block_size > 0`; adds a "means" output port. |
| order | integer | 1 | Prediction order: `1` (first difference) or `2` (second difference, FSZ's LZ2). `2` requires `block_size > 0`. |

**Output ports:** "output", plus "means" when `centering = true`.

### TiledLorenzo

Dimension-aware tiled separable Lorenzo predictor (lossless, cuSZp3 delta).
Applies the delta along each axis within a tile rather than over the flat
array, so the prediction respects 2-D/3-D locality. Requires
`pipeline.setDims()` (or `-l`) to be set.

| Key | Type | Default | Description |
|---|---|---|---|
| data_type | string | "int32" | Signed integer element type. "int16" or "int32". |
| tile_x | integer | 0 | Tile extent along x. `0` on all three = stage-chosen default shape. |
| tile_y | integer | 0 | Tile extent along y. |
| tile_z | integer | 0 | Tile extent along z. |

Setting any one of `tile_x`/`tile_y`/`tile_z` sets the whole shape; the unset
axes fall back to `1`, not to the default shape.

### AdaptiveLorenzo

Per-tile adaptive multi-order Lorenzo predictor with centering (FSZ prediction
stage). Picks LZ1 / LZ2 / LZ1+centering / LZ2+centering per tile by exact encoded
byte cost. Pair with `AdaptiveBitpack` at `block_size = 32`, which is what the
cost model assumes.

| Key | Type | Default | Description |
|---|---|---|---|
| data_type | string | "int32" | Signed integer element type. "int16" or "int32". |
| coder_block_size | integer | 32 | Downstream coder block size. Must be 32. |
| blocks_per_tile | integer | 8 | Coder blocks per adaptation tile; tile = 32 x this. Must be in [1, 32]. |
| enable_order2 | boolean | true | Include the second-order (LZ2) variants. |
| enable_centering | boolean | true | Include the mean-centered variants. |

**Output ports:** "output" (residuals), "modes" (1 byte per tile), "means"
(one element per tile). Ports not referenced downstream become pipeline outputs.

### Bitshuffle

GPU bit-matrix transpose. Size-preserving; improves entropy coder performance
on integer data.

| Key | Type | Default | Description |
|---|---|---|---|
| block_size | integer | 16384 | Chunk size in bytes. Must be a positive multiple of 1024 x element_width. |
| element_width | integer | 4 | Element width in bytes: 1, 2, 4, or 8. |

### TUPL

Tuple deinterleave (AoS -> SoA) transpose -- lossless byte-stream shuffler (LC
framework `TUPLk` component). Regroups a block of `dim`-field tuples
field-major. Size-preserving; a decorrelation step for a downstream coder, not
a compressor on its own.

| Key | Type | Default | Description |
|---|---|---|---|
| block_size | integer | 16384 | Block size in bytes. Must be a positive multiple of word_size. |
| word_size | integer | 1 | Field width in bytes: 1, 2, 4, or 8. |
| dim | integer | 2 | Fields per tuple (LC's TUPLk, k = dim). Must be >= 2. |

### RZE

Zero-Elimination Encoding -- lossless byte-stream compressor (LC framework
component). Eliminates zero words (the sibling of RRE, which eliminates repeats).

| Key | Type | Default | Description |
|---|---|---|---|
| chunk_size | integer | 16384 | Chunk size in bytes. Only 16384 is currently supported. |
| word_size | integer | 1 | Word granularity in bytes: 1, 2, 4, or 8 (LC RZE_1/RZE_2/RZE_4/RZE_8). |

### RRE

Repetition-Reduction Encoding -- lossless byte-stream compressor (LC framework
component used by cuSZ-Hi's LC pipelines). Eliminates runs of a repeated value
(the sibling of RZE, which eliminates zeros).

| Key | Type | Default | Description |
|---|---|---|---|
| chunk_size | integer | 16384 | Chunk size in bytes. Only 16384 is currently supported. |
| word_size | integer | 1 | Word granularity in bytes: 1, 2, 4, or 8 (LC RRE_1/RRE_2/RRE_4/RRE_8). |

### RARE

Repetition-Adaptive Reduction Encoding -- lossless byte-stream compressor (LC
framework component). The auto-k generalization of RRE: instead of a binary
repeat-or-drop test, picks one global bit-width `keep` that maximizes savings
and bit-packs every word whose top bits match its predecessor at that width
(the sibling of RAZE, which generalizes RZE the same way).

| Key | Type | Default | Description |
|---|---|---|---|
| chunk_size | integer | 16384 | Chunk size in bytes: 4096, 8192, or 16384. |
| word_size | integer | 1 | Word granularity in bytes: 1, 2, 4, or 8 (LC RARE_1/RARE_2/RARE_4/RARE_8). |

### RAZE

Zero-Adaptive Reduction Encoding -- lossless byte-stream compressor (LC
framework component). The auto-k generalization of RZE: instead of a binary
zero-or-full test, picks one global bit-width `keep` that maximizes savings
and bit-packs every word whose top bits are all zero at that width (the
sibling of RARE, which generalizes RRE the same way).

| Key | Type | Default | Description |
|---|---|---|---|
| chunk_size | integer | 16384 | Chunk size in bytes: 4096, 8192, or 16384. |
| word_size | integer | 1 | Word granularity in bytes: 1, 2, 4, or 8 (LC RAZE_1/RAZE_2/RAZE_4/RAZE_8). |

### CLOG

Compressed-Logarithm adaptive bit-width coding -- lossless byte-stream
compressor (LC framework component). Splits each chunk into a fixed 32
subchunks; each subchunk is bit-packed to the minimum width needed to
represent its own max value losslessly. `word_size` selects an unsigned type
only.

| Key | Type | Default | Description |
|---|---|---|---|
| chunk_size | integer | 16384 | Chunk size in bytes: 4096, 8192, or 16384. |
| word_size | integer | 1 | Word granularity in bytes: 1, 2, 4, or 8 (LC CLOG_1/CLOG_2/CLOG_4/CLOG_8). |

### HCLOG

Compressed-Logarithm coding with a per-subchunk TCMS fallback -- lossless
byte-stream compressor (LC framework component, the auto-selecting sibling of
CLOG). For each subchunk, additionally tries a TCMS(zigzag) reinterpretation
and picks whichever needs fewer bits, recording the choice as one flag bit
per subchunk. `word_size` selects an unsigned type only.

| Key | Type | Default | Description |
|---|---|---|---|
| chunk_size | integer | 16384 | Chunk size in bytes: 4096, 8192, or 16384. |
| word_size | integer | 1 | Word granularity in bytes: 1, 2, 4, or 8 (LC HCLOG_1/HCLOG_2/HCLOG_4/HCLOG_8). |

### GPULZ

LZSS dictionary coder -- lossless (GPULZ, ICS '23). Per-chunk sliding-window
match search; chunks that do not compress fall back to raw, and all-zero chunks
are skipped entirely.

| Key | Type | Default | Description |
|---|---|---|---|
| chunk_size | integer | 2048 | Chunk size in bytes: 1024, 2048, or 4096. |
| word_size | integer | 4 | Word granularity in bytes: 1, 2, 4, or 8. Match the upstream symbol width. |
| match_level | integer | 1 | `0` = exact longest match over the 32-element near window only; `1` additionally consults a hashed two-word table for long-range candidates. Level 0 is faster, level 1 compresses better; the container format is identical and the level is not serialized. |
| split_mode | boolean | false | Emit four separate ports instead of one interleaved stream (see below). |

**Output ports:** "output" when `split_mode = false`; otherwise "literals",
"lengths", "offsets", and "meta".

Split mode is the Zstandard split -- the parts have very different symbol
distributions, and interleaving them raises the entropy a downstream coder
sees. **All four ports must be entropy coded and re-merged**: unlike the
single-stream form, a split leaks any byte left out. Feed "literals" to a
symbol-width-matched coder (e.g. `Huffman` with `input_type = "uint16"` for
uint16 quant codes) rather than a byte coder; that alphabet effect is the
larger half of the gain. See `examples/presets/gpu_zstd.toml`.

### Merge

Structural stage: concatenates N producer ports into one buffer (forward) and
splits the reconstructed buffer back into N segments (inverse). Used to run a
single lossless chain over the concatenation of several ports (e.g. cuSZ-Hi's
merged `[Huffman | outliers]` blob). List one `inputs` entry per segment, in
`segments` order.

| Key | Type | Default | Description |
|---|---|---|---|
| segments | string[] | (required) | Segment names, in connection order (max 16). Defines N. |

### RLE

Run-Length Encoding. Effective on quantization code streams with long runs of
identical values.

| Key | Type | Default | Description |
|---|---|---|---|
| data_type | string | "uint16" | Element type. One of "uint8", "uint16", "uint32", "uint64", "int8", "int16", "int32", "int64". |

### Difference

First-order difference coding with optional negabinary or zigzag fusion.

| Key | Type | Default | Description |
|---|---|---|---|
| input_type | string | "float32" | Input element type. |
| output_type | string | (same as input_type) | Output element type. When output_type is the unsigned counterpart of a signed input_type, the transform selected by fusion_mode is fused into the forward pass. |
| fusion_mode | string | "negabinary" | "negabinary" (LC's DIFFNB) or "zigzag" (LC's DIFFMS, sign-magnitude/TCMS). Ignored when input_type == output_type. |
| chunk_size | integer | 0 | Chunk size in bytes (0 = no chunking, process whole array as one context). When > 0, differences reset at each chunk boundary, enabling parallel decompression. |

**Fused instantiations** (when input_type != output_type; fusion_mode selects negabinary vs. zigzag):

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
| error_bound_mode | string | "REL" | "ABS" (absolute), "REL" (exact pointwise relative, log-space), "NOA" (value-range relative), or "PREL" (pseudo-relative, `eb × max\|data\|`). |
| quant_radius | integer | 32768 | Quantization radius. |
| outlier_capacity | float | 0.05 | Fraction of elements reserved as outlier capacity (0.0-1.0). |
| zigzag_codes | boolean | true | Zigzag-encode codes before output to improve downstream compressibility. No effect in REL mode. |
| outlier_threshold | float | inf | ABS/NOA: values with |x| >= threshold are forced to lossless outlier regardless of bin. Omit (default) to disable. |
| inplace_outliers | boolean | false | ABS/NOA: encode outlier raw bits in-place in the codes array (no scatter buffers). Cannot be used with REL mode. |
| dither | boolean | false | ABS/NOA/REL: reconstruct to a deterministic pseudo-random point within the bin/bound instead of always the bin center (LC's QUANT_*_R). Decorrelates reconstruction error from the signal. Roughly quadruples the outlier rate (~25% for smooth data) — size outlier_capacity accordingly. Cannot be used with linear_mode or inplace_outliers. |
| dither_seed | integer | 0 | Seed for the deterministic per-element dither offset. Only meaningful when dither = true. |
| dither_strength | float | 1.0 | Dither offset amplitude as a fraction of abs_eb, in (0,1]. 1.0 matches LC's literal definition (~25% outlier rate); lower values trade decorrelation strength for fewer outliers. Only meaningful when dither = true. |

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
| base | integer | 0 | Frame-of-reference offset. Packs (v - base) and restores v = packed + base, removing dead high bits when values cluster away from zero. Always lossless. |
| shift | integer | 0 | Right shift applied after the base subtraction, removing dead low bits. Must be in [0, 8*sizeof(T)-1]. **Lossy** unless every (v - base) has that many trailing zeros -- prefer auto_shift. |
| auto_base | boolean | false | Min-reduce the input and use the minimum as base. Lossless. Disables CUDA Graph capture. |
| auto_shift | boolean | false | OR-reduce every (v - base) and use its trailing-zero count as shift -- the largest shift that drops no information. Always lossless. Disables CUDA Graph capture. |
| auto_detect | boolean | false | Scan for the maximum and pick the smallest power-of-two nbits covering the shifted range. Disables CUDA Graph capture. |
| adaptive | boolean | false | Shorthand for auto_base + auto_shift + auto_detect: fully adaptive, lossless. |

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

### ANS

GPU rANS entropy coder (dietGPU port). Byte-alphabet coder -- feed it a byte
stream, not wide symbols.

| Key | Type | Default | Description |
|---|---|---|---|
| prob_bits | integer | 10 | Symbol probability precision in bits. |

Not available on the HIP backend; adding the stage there throws.

### AdaptiveBitpack

Per-block adaptive fixed-rate bit-plane coding (cuSZp / cuSZp2 port). Each
block is packed to the minimum bit width its own max magnitude needs.

| Key | Type | Default | Description |
|---|---|---|---|
| input_type | string | "int32" | Signed integer element type. "int16" or "int32". |
| block_size | integer | 32 | Elements per block. Use 32 when pairing with `AdaptiveLorenzo`, whose cost model assumes it. |
| outlier_selection | boolean | false | cuSZp2 per-block plain/outlier selection: a block may instead store element 0 as a raw outlier and pack only the rest, whichever is smaller. Helps non-sparse, high-smoothness data. |

### ADM

Adaptive data mapping (MANS port). Remaps a uint16/uint32 stream into a compact
8-bit symbol domain so a byte coder downstream sees a denser alphabet.

| Key | Type | Default | Description |
|---|---|---|---|
| dtype | string | "uint16" | Input element type: "uint16" or "uint32". |

### LogTransform

Log transform for point-wise relative error (Liang et al., CLUSTER '18).
Converts a point-wise *relative* bound into a plain *absolute* one, so an
ordinary ABS quantizer downstream delivers the relative guarantee. `float`
input only.

| Key | Type | Default | Description |
|---|---|---|---|
| error_bound | float | 1e-3 | The point-wise relative error bound to be realized. |
| threshold | float | 0.0 | Magnitudes at or below this are treated as near-zero and routed to the outlier channel rather than transformed. |
| outlier_capacity | float | 0.05 | Fraction of elements reserved for outliers (zeros, sign changes, sub-threshold values). |

### GInterp

Multi-level spline interpolation predictor with fused error-bounded
quantization (cuSZ-Hi port). 3-D primary path; 2-D supported with a
deterministic parameter fallback. Requires `pipeline.setDims()`.

| Key | Type | Default | Description |
|---|---|---|---|
| input_type | string | "float32" | "float32" or "float64". |
| code_type | string | "uint16" | Quantization code type: "uint8", "uint16", or "uint32". |
| error_bound | float | 1e-3 | Error bound value. |
| error_bound_mode | string | "ABS" | "ABS", "REL", "PREL", or "NOA". |
| quant_radius | integer | 0 | Codes lie in `[0, 2*radius)`. `0` enables the radius auto-tune. |
| outlier_capacity | float | 0.10 | Fraction of elements reserved for out-of-radius outliers. |
| auto_tuning | integer | 0 | cuSZ-Hi `INTERPOLATION_PARAMS` auto-tuning: `0` off, `1` cheap, `3` full, `4` full + alpha sweep, `5` manual alpha/beta. Modes 1/3/4 disable CUDA Graph capture for this stage; mode 5 is graph-safe. 3-D only. |

**Output ports:** "codes" (connect downstream stages here, not "output"), plus
the fused outlier channel.

### BitplaneRZE

Fused bitplane transpose + zero-group RZE lossless encoder (FZ-GPU port).

No tunable keys -- the configuration is derived from the input length.

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
- Single-source pipelines only. The [pipeline] table has one input_size and
  one dims triple. Multi-source pipelines are not currently representable in
  the config format and must be constructed manually.