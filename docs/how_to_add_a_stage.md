# How to Add a New Stage {#how_to_add_a_stage}

<!-- doc-check: skip-file — this page is a template walkthrough built around
     placeholders (`MyStage`, `MY_STAGE`, `<category>/<name>/`), so its snippets
     are not compilable as written. See scripts/check_doc_cpp.py. -->

Complete walkthrough for adding a new compression/decompression stage to FZGPUModules.
Use `scripts/new_stage.sh` to generate the file skeleton automatically:

```bash
scripts/new_stage.sh MyStageName <category>            # auto-assign StageType ID
scripts/new_stage.sh MyStageName <category> 42         # pin StageType ID explicitly
```

Prototyping a stage in your own repository, or not ready to publish it yet? See
\ref out_of_tree_stage "Adding a stage outside the main repo" — the class is
written the same way, but you link against an installed FZGPUModules instead of
editing its shared files.

---

## Overview

A stage is a single transformation in the pipeline (predictor, coder, transform, etc.).
The pipeline interacts with every stage exclusively through the `Stage` base class interface —
there is no casting or type-name branching anywhere in pipeline or DAG code.

To make pipelines that use your stage eligible for \ref pipeline_specialization "Pipeline Specialization" (automatic kernel fusion + optimizations, both compress and decompress), implement the declaration contract in \ref pipeline_specialization_internals "the specialization-compatibility contract" after the staged stage round-trips.

**Files you will touch for a new stage.** `scripts/new_stage.sh` scaffolds and
edits the ones marked *(script)*, so a hand-written stage really only fills in the
kernels and the TOML support:

| File | What you do | Automation |
|------|-------------|------------|
| `modules/<category>/<name>/<name>_stage.h` | Stage class declaration | |
| `modules/<category>/<name>/<name>_stage.cu` | CUDA kernels + `execute()`, plus one `FZ_REGISTER_SIMPLE_STAGE` / `FZ_REGISTER_STAGE_FACTORY` line that self-registers FZM-header reconstruction | *(script emits the line)* |
| `include/fzm_format.h` | Add `StageType` enum value + `stageTypeToString()` case | *(script)* |
| `modules/<category>/<name>/CMakeLists.txt` | Declare the module's sources on `fzgmod_modules` | *(script)* |
| `modules/CMakeLists.txt` | Add the module directory to `FZGMOD_STAGE_MODULES` | *(script)* |
| `tests/stages/test_<name>.cpp` + `tests/stages/CMakeLists.txt` | Standard test set, registered | *(script)* |
| `src/pipeline/config.cpp` | `#include`, `addXxxStage` + `saveXxxStage` helpers, one `kStageRegistry[]` entry — TOML load/save, the only shared file still edited by hand | |
| `include/fzgpumodules.h` | Add the stage header include (public API export) | |
| `src/utils/cli/cli.cpp` | *(Optional)* add the name to the `--stages` dynamic builder | |

There is **no central factory switch to edit** — stage reconstruction self-registers
from the stage's own `.cu` (Step 5). The one shared registration that remains is TOML
load/save in `config.cpp`, because toml++ is deliberately confined to that translation
unit.

---

## Step 1 — Choose a location

Stages live under `modules/` in one of these categories:

| Category | Path | Existing examples |
|----------|------|-------------------|
| Predictors | `modules/predictors/<name>/` | `lorenzo/`, `diff/` |
| Quantizers | `modules/quantizers/<name>/` | `quantizer/` |
| Coders | `modules/coders/<name>/` | `rle/`, `rze/`, `bitpack/` |
| Shufflers | `modules/shufflers/<name>/` | `bitshuffle/` |
| Transforms | `modules/transforms/<name>/` | `zigzag/`, `negabinary/` |
| Fused | `modules/fused/<name>/` | `lorenzo_quant/` |

### Category definitions

**Predictors** decorrelate the data by computing residuals from a prediction model.
The forward pass subtracts predicted values from actual values, producing a residual
stream that has much smaller magnitude and is far more compressible.  The inverse
reconstructs the original by applying the cumulative prediction.  The operation is
always lossless.  Use this category when your stage consumes raw data values and
produces signed residuals (e.g. delta coding, Lorenzo predictor, interpolation).

**Quantizers** perform the lossy step: they map a continuous (floating-point) or
fine-grained value to a discrete integer code within a user-specified error bound.
Output codes are integers, typically `uint16_t` or `uint32_t`, and the stage emits
a separate outlier stream for values that fall outside the representable range.
The inverse reconstructs an approximation of the original values.  Use this
category when your stage introduces controlled, bounded loss to reduce dynamic range.

**Coders** compress an integer or byte stream losslessly by exploiting statistical
redundancy — repeated values, long zero runs, or skewed symbol distributions.
They do not reorder, predict, or transform the data; they only compact it.
Output byte size is variable and must be smaller than input in the common case
(or stored raw if not).  The inverse exactly recovers the input stream.  Use this
category for entropy coders, run-length schemes, or any stage whose sole job is
symbol-to-bitstream encoding.

**Shufflers** restructure the bytes of a stream without changing any values —
they are size-preserving, lossless rearrangements designed to improve the
compressibility of downstream coders.  The canonical example is bit-matrix
transposition (bitshuffle): grouping bit-plane k of all elements together so
that sign bits and exponent bits form long runs of identical bytes.  Use this
category when your stage only reorders bytes/bits and the output is the same size
as the input.

**Transforms** apply an invertible, element-wise mathematical mapping — every
input element maps to exactly one output element of a (possibly different) type,
and the mapping is exactly reversible.  Examples: zigzag encoding maps signed
integers to unsigned integers preserving magnitude order; negabinary maps signed
integers to base-(-2) unsigned codes.  Use this category when your stage is a
bijective point-wise function with no inter-element dependencies, no size change,
and no loss.

**Fused** stages combine two or more logically distinct operations (typically a
predictor and a quantizer, or a predictor and a transform) into a single kernel
to reduce memory round-trips.  A fused stage is always a performance optimization:
it is semantically equivalent to the un-fused stages wired in sequence, but avoids
the intermediate buffer reads and writes.  Use this category only when profiling
shows the unfused version is memory-bandwidth limited and the stages are always
used together.

If your stage does not fit cleanly into one category, prefer the one that best
describes its **primary** effect on the data.  A stage that both predicts and
quantizes but is not performance-critical belongs in `predictors/` or `quantizers/`
rather than `fused/`.

Create the directory: `modules/<category>/<name>/`

---

## Step 2 — Write the header (<name>_stage.h)

Copy the pattern from a nearby existing stage (e.g. `modules/transforms/zigzag/zigzag_stage.h`
for a size-preserving transform, or `modules/coders/rle/rle.h` for a coding stage).

Required overrides:

```cpp
#pragma once
#include "stage/stage.h"
#include "fzm_format.h"
#include "backend/types.h"   // NOT <cuda_runtime.h> — see Step 3b
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace fz {

class MyStage : public Stage {
public:
    // ── Stage control ─────────────────────────────────────────────────────────
    void setInverse(bool inv) override { is_inverse_ = inv; }
    bool isInverse() const override    { return is_inverse_; }

    // ── Execution ─────────────────────────────────────────────────────────────
    void execute(cudaStream_t stream, MemoryPool* pool,
                 const std::vector<void*>& inputs,
                 const std::vector<void*>& outputs,
                 const std::vector<size_t>& sizes) override;

    // ── Metadata ──────────────────────────────────────────────────────────────
    std::string getName() const override { return "MyStage"; }
    size_t getNumInputs()  const override { return 1; }
    size_t getNumOutputs() const override { return 1; }

    // Upper bound on output size — must be safe to over-estimate; never under.
    // Must return correct bounds for BOTH forward and inverse directions.
    std::vector<size_t> estimateOutputSizes(
        const std::vector<size_t>& input_sizes) const override {
        return {input_sizes[0]};   // size-preserving example
    }

    std::unordered_map<std::string, size_t>
    getActualOutputSizesByName() const override {
        return {{"output", actual_output_size_}};
    }

    // ── Type system ───────────────────────────────────────────────────────────
    uint16_t getStageTypeId() const override {
        return static_cast<uint16_t>(StageType::MY_STAGE);
    }

    // Return DataType::UNKNOWN to opt out of finalize()-time type checking.
    uint8_t getOutputDataType(size_t /*output_index*/) const override {
        return static_cast<uint8_t>(DataType::UINT16);
    }
    uint8_t getInputDataType(size_t /*input_index*/) const override {
        return static_cast<uint8_t>(DataType::UNKNOWN);
    }

    // ── Serialization ─────────────────────────────────────────────────────────
    // max_size is always 128 bytes. Return 0 if the stage has no config.
    size_t serializeHeader(size_t output_index,
                           uint8_t* buf, size_t max_size) const override;
    void   deserializeHeader(const uint8_t* buf, size_t size) override;
    size_t getMaxHeaderSize(size_t) const override { return 8; }

    // saveState / restoreState: implement these if deserializeHeader overwrites
    // fields also used by the forward pass (e.g. a value_range computed at
    // compress-time and stored in the header for decompression).
    void saveState()    override { saved_config_ = config_; }
    void restoreState() override { config_ = saved_config_; }

private:
    bool   is_inverse_         = false;
    size_t actual_output_size_ = 0;
    SomeConfig config_;
    SomeConfig saved_config_;
};

} // namespace fz
```

### Multi-output stages

For a single output, the default `getOutputNames()` returning `{"output"}` is fine.
Multi-output stages override it:

```cpp
std::vector<std::string> getOutputNames() const override {
    return {"codes", "outliers"};
}
```

Users connect to named ports: `pipeline.connect(downstream, myStage, "codes")`.

### Non-size-preserving stages: bidirectional estimateOutputSizes

If your stage changes the data size (encoding, packing, compression), handle both
directions:

```cpp
std::vector<size_t> estimateOutputSizes(
    const std::vector<size_t>& input_sizes) const override
{
    if (input_sizes.empty()) return {0};
    if (!is_inverse_) {
        return {encodedSize(input_sizes[0])};   // forward: encoded output size
    } else {
        return {decodedSize(input_sizes[0])};   // inverse: decoded output size
    }
}
```

A forward-only implementation silently under-allocates the inverse output buffer.
The pipeline bounds checker will catch this; without bounds checking it is silent
memory corruption.

### Persistent scratch memory

If your stage needs a reusable buffer across calls, override `estimateScratchBytes()`
so the pool accounts for it, then allocate with `persistent = true` in `execute()`:

```cpp
size_t estimateScratchBytes(const std::vector<size_t>& input_sizes) const override {
    return input_sizes.empty() ? 0 : input_sizes[0] * 2;
}

// Inside execute():
void* scratch = pool->allocate(scratch_bytes, stream, "my_scratch", /*persistent=*/true);
```

### CUDA Graph compatibility

Override `isGraphCompatible()` to return `false` if `execute()` contains any of:
- `cudaStreamSynchronize()` on its own stream
- Blocking D2H copies (`cudaMemcpy` with `DeviceToHost`)
- CPU decisions based on device data

```cpp
bool isGraphCompatible() const override { return false; }
```

If you need a D2H transfer only *after* the whole pipeline finishes, use
`postStreamSync()` instead — the stream is idle there and graph capture is
unaffected.

Stages that sync mid-execute are valid and supported. The pattern is used by
`HuffmanStage` (histogram D2H for codebook build + partition metadata D2H)
and any future ANS/arithmetic coder that needs CPU-side renormalization. Document
the sync points and return `false` from `isGraphCompatible()`.

### Input alignment

Stages that require input sizes to be a multiple of a chunk size override:

```cpp
size_t getRequiredInputAlignment() const override { return chunk_size_bytes; }
```

`Pipeline::finalize()` computes the LCM of all stage alignments and pads the input
transparently.

---

## Step 3 — Write the implementation (<name>_stage.cu)

```cpp
#include "<category>/<name>/<name>_stage.h"
#include "mem/mempool.h"
#include "cuda_check.h"   // FZ_CUDA_CHECK macro (internal use only — do not use in examples)
#include "backend/api.h"  // NOT <cuda_runtime.h> — see Step 3b
#include <stdexcept>

namespace fz {

__global__ void myKernel(const T* __restrict__ in, U* __restrict__ out, size_t n) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = /* transform */ in[idx];
}

void MyStage::execute(cudaStream_t stream, MemoryPool* pool,
                      const std::vector<void*>& inputs,
                      const std::vector<void*>& outputs,
                      const std::vector<size_t>& sizes)
{
    if (inputs.empty() || outputs.empty() || sizes.empty())
        throw std::runtime_error("MyStage: inputs, outputs, and sizes must be non-empty");

    const size_t n = sizes[0] / sizeof(T);
    if (n == 0) { actual_output_size_ = 0; return; }

    constexpr int kBlock = 256;
    const int grid = static_cast<int>((n + kBlock - 1) / kBlock);
    myKernel<<<grid, kBlock, 0, stream>>>(
        static_cast<const T*>(inputs[0]),
        static_cast<U*>(outputs[0]), n);

    FZ_CUDA_CHECK(cudaGetLastError());
    actual_output_size_ = n * sizeof(U);
}

} // namespace fz
```

**Enqueue all GPU work on stream** — the pipeline manages inter-stage ordering
through CUDA events. Do not call `cudaDeviceSynchronize()` or synchronize a
*different* stream inside `execute()`.

**Calling cudaStreamSynchronize(stream) on your own stream is permitted**
when the algorithm inherently requires it (e.g. reading a GPU histogram to the
host to build a CPU-side codebook, as in Huffman or ANS). When you do this:

- Override `isGraphCompatible()` to return `false` — CUDA Graph capture records
  a snapshot of the command stream; a mid-execute sync prevents capture.
- Document the sync points in your class header and in `docs/stages/<name>.md`.
- Be aware that the DAG dispatches nodes within a level sequentially on the CPU.
  A sync inside your `execute()` blocks the CPU from dispatching sibling nodes
  (stages at the same DAG level) until your stream is idle. In a linear chain
  this has no impact; in a wide DAG it delays parallel branches.

### Shared output locations

If multiple input elements contribute to the same output location (packing,
reduction), each thread must exclusively own its output slot or use atomics.
Plain `|=` / `+=` on a shared address without atomics is a data race even after
a `cudaMemsetAsync` pre-zero.

Two patterns that avoid atomics for packing:

```
// Pattern A: one thread per output slot.
// Each thread reads all inputs that map to its output and packs them itself.
// Works for regular mappings like bit-packing (k inputs → 1 byte).

// Pattern B: atomicOr / atomicCAS.
// More flexible, slower. Prefer Pattern A when the mapping is regular.
```

---

## Step 3b — Backend facade conventions

Stable main builds CUDA only. Use `backend/types.h`, `backend/api.h`, and
`backend/cub.h` in stage code so runtime types, operations, and CUB access follow
the library's existing abstraction. Keep CUDA-only headers out of public stage
headers when a facade provides the required type.

State logical warp widths explicitly for warp-local algorithms. Backend work and
AMD validation are preserved on `development`; a stable-main CUDA test does not
establish HIP compatibility. Stage/backend-specific restrictions should remain
local to the module build declaration.

---

## Step 4 — Register the StageType

In `include/fzm_format.h`, add to the `StageType` enum:

```cpp
enum class StageType : uint16_t {
    // ... existing entries (do NOT renumber or reuse) ...
    MY_STAGE = 19,   // next available value after RZE = 18
};
```

Also add to `stageTypeToString()` in the same file:

```cpp
case StageType::MY_STAGE: return "MyStage";
```

**Never renumber or reuse existing values.** They are serialized in `.fzm` files;
reusing a value corrupts files that contain the old stage type.

---

## Step 5 — Self-register the FZM-header factory

`decompressFromFile()` rebuilds each stage from its serialized header through
`createStage()` (`include/stage/stage_registry.h`). There is **no central switch**
to edit — a stage registers its own reconstruction from its `.cu` at file scope,
so this step stays inside the stage's own directory.

For a stage with no template dispatch, one line at the bottom of `<name>_stage.cu`
(after `#include "stage/stage_registry.h"`) is enough:

```cpp
FZ_REGISTER_SIMPLE_STAGE(fz::StageType::MY_STAGE, fz::MyStage);
```

If your stage is templated on a type, write a small factory that dispatches on the
`DataType` byte(s) stored in the config header, then register it:

```cpp
namespace {
fz::Stage* MyStage_fromHeader(const uint8_t* config, size_t config_size) {
    using fz::DataType;
    DataType dt = (config_size > 0) ? static_cast<DataType>(config[0])
                                    : DataType::INT32;
    fz::Stage* s = (dt == DataType::INT16) ? static_cast<fz::Stage*>(new fz::MyStage<int16_t>())
                                           : static_cast<fz::Stage*>(new fz::MyStage<int32_t>());
    s->deserializeHeader(config, config_size);
    return s;
}
}  // namespace
FZ_REGISTER_STAGE_FACTORY(fz::StageType::MY_STAGE, MyStage_fromHeader);
```

The registrar runs at static-init when `libfzgmod_modules` loads. `scripts/new_stage.sh`
emits the `FZ_REGISTER_SIMPLE_STAGE` line for you.

**Static-archive builds:** registration relies on the stage's object file being
linked in. In the default shared-library build (`BUILD_SHARED_LIBS=ON`) every
module object is present in `libfzgmod_modules.so`, so registrars always run. If
you link the modules as a static archive, link it with `--whole-archive` (or an
equivalent `KEEP`) so the registrars are not stripped. `test_stage_registry`
asserts every shipped `StageType` has a registered factory.

---

## Step 6 — Declare module sources and register the directory

Create `modules/<category>/<name>/CMakeLists.txt` with an explicit source list:

```cmake
target_sources(fzgmod_modules PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/<name>_stage.cu
)
```

Add `<category>/<name>` to `FZGMOD_STAGE_MODULES` in `modules/CMakeLists.txt`.
The catalog calls `add_subdirectory()` for each entry; the root build file needs
no edits. `scripts/new_stage.sh` performs both steps automatically.

Keep extra source files, dependencies, and backend restrictions in the module's
own build declaration. Do not glob source files: experimental files should not
become shipped implementations simply because they were added to a directory.

---

## Step 6b — Export in the public header

Add the stage header include to `include/fzgpumodules.h` (the main public API header)
so users can access it with `#include "fzgpumodules.h"`:

```cpp
#include "<category>/<name>/<name>_stage.h"
```

Organize includes alphabetically within each category for consistency.

---

## Step 7 — Register in the TOML config loader

To make your stage constructable from a `.toml` pipeline file via
`Pipeline::loadConfig()` / `saveConfig()`, edit `src/pipeline/config.cpp`.
The file uses a central `kStageRegistry[]` table — there are **no scattered
`if/else` chains or `switch` blocks to hunt down**:

**1. Add the header include** at the top alongside the other stage includes:

```cpp
#include "<category>/<name>/<name>_stage.h"
```

**2. Write a load helper** (reads TOML keys, adds stage to the pipeline):

```cpp
static Stage* addMyStage(Pipeline& p, const toml::table& t) {
    auto* s = p.addStage<MyStage>();
    s->setChunkSize(static_cast<size_t>(optInt(t, "chunk_size", 16384)));
    return s;
}
```

**3. Write a save helper** (writes TOML keys for `saveConfig()`):

```cpp
static void saveMyStage(Stage* s, std::ostringstream& out) {
    auto* ms = static_cast<MyStage*>(s);
    out << "chunk_size = " << static_cast<int64_t>(ms->getChunkSize()) << "\n";
}
```

**4. Add one entry to kStageRegistry[]**:

```cpp
{ "MyStage", StageType::MY_STAGE, addMyStage, saveMyStage },
```

That's it. The `type` string (first field) is what appears in TOML files.
Convention: class name without the `Stage` suffix (e.g. `"Bitshuffle"`, `"RZE"`, `"Quantizer"`).

If your stage is templated, dispatch on `input_type` / `code_type` TOML keys in
the helpers — see `addLorenzoQuantStage` / `saveLorenzoQuantStage` for the pattern.

---

## Step 8 — Register in the CLI dynamic builder (optional)

If the stage makes sense as a general-purpose pipeline step, add it to the
`--stages` builder in `src/utils/cli/cli.cpp` and update the help text.

This step is optional — stages that only make sense with specific type
instantiations or unusual wiring can be TOML-only.

---

## Step 8b — Attribution (required when based on prior work)

If your stage ports, adapts, or closely follows an algorithm from another
project, you must acknowledge it in three places:

**1. Source file comment** — top of the `.cu` file:

```cpp
// Algorithm adapted from <Project Name> (<Authors>, <License>).
// Upstream: <URL> — see THIRD_PARTY.md.
```

For a direct port (kernel logic transliterated from upstream source), use:

```cpp
// GPU kernels are a direct port of <filename(s)> from <Project Name>
// (<Authors>, <License>). Upstream: <URL> — see THIRD_PARTY.md.
```

**2. Doxygen class comment** — in the `_stage.h` header, inside the class-level
`/** ... */` block:

```cpp
/**
 * My stage description.
 *
 * @note **Prior work:** <one-line description of what was adapted and from where>
 *       (<Authors>, <License>). See `THIRD_PARTY.md`.
 * ...
 */
```

**3. Stage documentation** — at the end of `docs/stages/<name>.md`, add an
`## Acknowledgements` section:

```markdown
## Acknowledgements

<Stage name> <relationship — e.g. "kernels are a direct port of X from"> the
**<Project Name>** (<Authors>, <License>).

> <Author list.>
> *<Paper or project title.>*
> <URL>

See `THIRD_PARTY.md` for the full license text.
```

**4. THIRD_PARTY.md** — if the upstream project is not already listed there,
add a new entry with:
- The modules that use it and the relationship (direct port / algorithm-based /
  vendored).
- The verbatim copyright notice copied from the upstream LICENSE file.

You do **not** need to do any of this for algorithms that are textbook-standard
and not derived from a specific project's code (e.g., zigzag, negabinary, basic
prefix sums).

---

## Step 9 — Write tests

Create `tests/stages/test_<name>.cpp` with at minimum:

| Test | What it checks |
|------|----------------|
| `ForwardRoundTrip` | Forward + inverse produces exact or within-error output |
| `ZeroInput` | `n=0` does not crash or corrupt |
| `SerializeDeserialize` | `serializeHeader` → `deserializeHeader` restores identical config |
| `PipelineIntegration` | Stage wired into a `Pipeline`, compress + decompress round-trip |
| `SaveRestoreState` | `saveState` + `deserializeHeader` + `restoreState` returns to original config |
| `GraphCompatible` | `isGraphCompatible()` returns expected value |

Use `tests/helpers/stage_harness.h`. Pipeline integration tests must use a
**single** `Pipeline` instance for both compress and decompress:

```cpp
#include "helpers/stage_harness.h"

TEST(MyStage, PipelineIntegration) {
    const size_t N = 1024;
    auto h_input = fz_test::make_smooth_data<float>(N);
    const size_t in_bytes = N * sizeof(float);

    fz::Pipeline p(in_bytes, fz::MemoryStrategy::PREALLOCATE);
    auto* s = p.addStage<fz::MyStage>();
    s->setSomeParam(42);
    p.finalize();

    fz::CudaStream cs;
    auto res = fz_test::pipeline_round_trip<float>(p, h_input, cs.stream);
    EXPECT_LT(res.max_error, 1e-4f);
}
```

Do **not** create separate `Pipeline` objects for compress and decompress — `decompress()`
builds the inverse DAG from the state of the same forward pipeline. The two-pipeline
pattern only works via `writeToFile`/`decompressFromFile`.

Register in `tests/stages/CMakeLists.txt`:
```cmake
fz_add_test(test_my_stage test_my_stage.cpp LABELS stages gpu)
```

---

## Checklist

- [ ] `<name>_stage.h` — all required overrides implemented
- [ ] `<name>_stage.cu` — `execute()` enqueues on `stream`; if it calls `cudaStreamSynchronize(stream)`, `isGraphCompatible()` returns `false` and sync points are documented
- [ ] Backend facade conventions (Step 3b): runtime types/API and CUB access use the existing facades; CUDA validation covers the declared logical warp widths.
- [ ] `StageType` enum value added (unique integer, never reuse old values)
- [ ] `stageTypeToString()` case added
- [ ] FZM-header factory self-registered in the `.cu` (`FZ_REGISTER_SIMPLE_STAGE` or `FZ_REGISTER_STAGE_FACTORY`)
- [ ] Module-local `CMakeLists.txt` declares sources on `fzgmod_modules`; directory added to `FZGMOD_STAGE_MODULES` in `modules/CMakeLists.txt`
- [ ] Stage header include added to `include/fzgpumodules.h` (public API export)
- [ ] `config.cpp` — `#include` header, `addXxxStage` / `saveXxxStage` helpers, one entry in `kStageRegistry[]`
- [ ] `cli.cpp` — `--stages` name + help text *(if applicable)*
- [ ] Tests: ForwardRoundTrip, ZeroInput, SerializeDeserialize, PipelineIntegration, SaveRestoreState
- [ ] `saveState`/`restoreState` implemented if `deserializeHeader` overwrites forward-pass config
- [ ] `estimateScratchBytes()` overridden if stage holds persistent pool allocations
- [ ] `getRequiredInputAlignment()` overridden if stage requires chunk-aligned input
- [ ] `isGraphCompatible()` returns `false` if `execute()` does any D2H transfer
- [ ] If based on prior work: attribution comment in `.cu`, `@note` in `.h`, `## Acknowledgements` in `docs/stages/<name>.md`, entry in `THIRD_PARTY.md` (if upstream not already listed)
