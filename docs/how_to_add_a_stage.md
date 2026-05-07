# How to Add a New Stage {#how_to_add_a_stage}

Complete walkthrough for adding a new compression/decompression stage to FZGPUModules.
Use `scripts/new_stage.sh` to generate the file skeleton automatically:

```bash
scripts/new_stage.sh MyStageName <category>            # auto-assign StageType ID
scripts/new_stage.sh MyStageName <category> 42         # pin StageType ID explicitly
```

---

## Overview

A stage is a single transformation in the pipeline (predictor, coder, transform, etc.).
The pipeline interacts with every stage exclusively through the `Stage` base class interface —
there is no casting or type-name branching anywhere in pipeline or DAG code.

**Files you will touch for a new stage:**

| File | What you do |
|------|-------------|
| `modules/<category>/<name>/<name>_stage.h` | Stage class declaration |
| `modules/<category>/<name>/<name>_stage.cu` | CUDA kernels + `execute()` |
| `include/fzm_format.h` | Add `StageType` enum value |
| `include/stage/stage_factory.h` | Add reconstruction case in `createStage()` |
| `CMakeLists.txt` (root) | Add `.cu` to `fzgmod_modules` library target |
| `src/pipeline/config.cpp` | Add factory case for TOML `loadConfig()` support |
| `src/utils/cli/cli.cpp` | *(Optional)* Add name to `--stages` dynamic builder |
| `tests/stages/test_<name>.cpp` | Standard test set |
| `tests/stages/CMakeLists.txt` | Register the test |

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
#include <cuda_runtime.h>
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

If `execute()` reads any data back from the device to the host (D2H copy) to make
a branch decision, override:

```cpp
bool isGraphCompatible() const override { return false; }
```

If you need a D2H transfer only after the whole pipeline finishes, use
`postStreamSync()` instead — the stream is idle there and graph capture is unaffected.

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
#include <cuda_runtime.h>
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

**Do not call `cudaDeviceSynchronize()` inside `execute()`** — the pipeline manages
stream ordering through CUDA events. All work must be enqueued on `stream`.

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

## Step 5 — Register in the factory

In `include/stage/stage_factory.h`, add a case to `createStage()`:

```cpp
case StageType::MY_STAGE: {
    stage = new MyStage();
    if (config_size > 0)
        stage->deserializeHeader(config, config_size);
    break;
}
```

This is the reconstruction path used by `decompressFromFile()`. If your stage is
templated on a type, dispatch on the `DataType` byte(s) stored in the config header
(see the `ZIGZAG` case as a reference).

---

## Step 6 — Add to CMakeLists.txt

All stage `.cu` files belong to the `fzgmod_modules` target in the root `CMakeLists.txt`:

```cmake
add_library(fzgmod_modules
    ...
    modules/<category>/<name>/<name>_stage.cu   # add here
)
```

---

## Step 7 — Register in the TOML config loader

To make your stage constructable from a `.toml` pipeline file via
`Pipeline::loadConfig()`, add to `src/pipeline/config.cpp`.

First include your stage header at the top alongside the other stage headers:

```cpp
#include "<category>/<name>/<name>_stage.h"
```

Then add a factory helper (follow the pattern of existing helpers in that file):

```cpp
static Stage* addMyStage(Pipeline& p, const toml::table& t) {
    auto* s = p.addStage<MyStage>();
    s->setChunkSize(static_cast<size_t>(optInt(t, "chunk_size", 16384)));
    return s;
}
```

And add a branch in the stage-construction pass inside `loadConfig()`:

```cpp
} else if (type == "MyStage") {
    s = addMyStage(*this, *t);
```

The `type` string must match exactly. Convention: use the class name without the
`Stage` suffix (e.g. `"Bitshuffle"`, `"RZE"`, `"Quantizer"`).

If your stage is templated, dispatch on `input_type` / `code_type` TOML keys —
see `addLorenzoQuantStage` or `addQuantizerStage` for the pattern.

Also add the corresponding `saveConfig()` serialization case (search for
`stage->getName()` in `config.cpp` to find the save path).

---

## Step 8 — Register in the CLI dynamic builder *(optional)*

If the stage makes sense as a general-purpose pipeline step, add it to the
`--stages` builder in `src/utils/cli/cli.cpp` and update the help text.

This step is optional — stages that only make sense with specific type
instantiations or unusual wiring can be TOML-only.

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
- [ ] `<name>_stage.cu` — `execute()` enqueues on `stream`, never calls `cudaDeviceSynchronize()`
- [ ] `StageType` enum value added (unique integer, never reuse old values)
- [ ] `stageTypeToString()` case added
- [ ] `createStage()` factory case added
- [ ] `.cu` file added to `fzgmod_modules` in root `CMakeLists.txt`
- [ ] `config.cpp` — factory helper + `loadConfig()` branch + `saveConfig()` case
- [ ] `cli.cpp` — `--stages` name + help text *(if applicable)*
- [ ] Tests: ForwardRoundTrip, ZeroInput, SerializeDeserialize, PipelineIntegration, SaveRestoreState
- [ ] `saveState`/`restoreState` implemented if `deserializeHeader` overwrites forward-pass config
- [ ] `estimateScratchBytes()` overridden if stage holds persistent pool allocations
- [ ] `getRequiredInputAlignment()` overridden if stage requires chunk-aligned input
- [ ] `isGraphCompatible()` returns `false` if `execute()` does any D2H transfer
