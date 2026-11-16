---
name: fzgm-add-stage
description: Add a new compression stage (predictor, quantizer, transform, coder, shuffler, or fused kernel) to FZGPUModules, from design through scaffold, implementation, TOML/CLI/docs integration, tests, and verification. Use when asked to create, add, implement, or port a new stage or module into this library, or to finish a half-integrated stage.
---

# Add a new FZGPUModules stage

The canonical reference is `docs/how_to_add_a_stage.md` (Steps 1-9 + Checklist).
This skill gives the order, the decisions to make up front, what the docs miss, and
one script that says whether the stage is done. Read the doc section named in each
phase when you reach it.

Prototyping outside the repo, or not ready to publish? Use `docs/out_of_tree_stage.md`
instead; the stage class is the same, but no shared files are edited.

## Phase 0 — Workspace

- Work on a new branch, ideally in its own worktree, not a shared checkout:
  `git worktree add -b stage/<name> ../FZGPUModules-<name> main`
- A fresh clone or worktree needs submodules, or CMake fails on googletest:
  `git submodule update --init --recursive`
- Don't build yet. The scaffold (Phase 2) edits `include/fzm_format.h`, which forces
  a near-full rebuild, so build once after scaffolding.
- Build parallelism: `-j$(nproc)` on your own machine; cap it (e.g. `-j16`) on a
  shared host. `verify_stage.sh` reads `JOBS`.

## Phase 1 — Design card (before any code)

Answer these in a short note to the user. If a question can't be answered from the
request, ask; don't guess a file format.

| Question | Why it matters | Where it lands |
|---|---|---|
| Name (PascalCase, no `Stage` suffix) and category | directory, StageType, TOML name | Step 1 |
| Input/output element types; templated? | `getInput/OutputDataType`, factory, TOML type keys | Steps 2, 5, 7 |
| Size-preserving or variable-size? Forward *and* inverse size bounds | `estimateOutputSizes` must be safe in both directions | Step 2 |
| Ports: one output, or named outputs (`codes`, `outliers`, ...)? | `getOutputNames`, downstream wiring | Step 2 |
| What config goes in the header (<= 128 bytes)? Does decode overwrite forward config? | `serializeHeader`, `saveState/restoreState` | Step 2 |
| Scratch memory, input alignment, any host sync / D2H in `execute()`? | `estimateScratchBytes`, `getRequiredInputAlignment`, documented sync points | Step 2 |
| Inputs that don't divide evenly (trailing partial word/chunk): handle, pass through, or reject? | silent corruption at the tail; a stage that handles a partial last chunk returns `getRequiredInputAlignment() = 1` (otherwise the pipeline pads every input) | Steps 2, 3 |
| Based on prior work (code or a specific paper's algorithm)? | attribution in 4 places | Step 8b |
| Fuse with neighbours later? | specialization contract, after it round-trips | `docs/pipeline_specialization_internals.md` |
| Add to the CLI `--stages` builder? | optional | Step 8 |

**Category tie-break.** Decide by what the operation needs, not what it's called:
- any element depends on others (delta, scan, prediction) → `predictors`, even for
  integer code streams;
- strictly element-wise and bijective → `transforms`;
- changes size losslessly → `coders`;
- reorders bytes/bits only → `shufflers`.
If still unsure, use the category of the closest existing stage and say why.

**TOML key convention.** Single-type stages use `data_type` (RLE, Lorenzo); stages
with distinct input/output types use `input_type` / `output_type` (Zigzag, Bitpack).

**Template to copy** — say which you picked:

| Shape of the new stage | Copy from |
|---|---|
| Size-preserving, element-wise, templated | `modules/transforms/zigzag/` (factory on header byte 0, explicit + `extern template` instantiations) |
| Size-preserving with inter-element dependency (scan / prefix op) | `modules/predictors/diff/`; for a device-wide scan use `backend::withTempStorage` + `cub::DeviceScan` as in `modules/coders/rle/` (transient scratch, not counted in `estimateScratchBytes`) |
| Variable-size lossless coder | `modules/coders/rze/` (chunked bytes), `modules/coders/rle/` (templated), `modules/coders/bitpack/`; see Phase 3, "Variable-size output" |
| Multi-output (codes + outliers) | `modules/quantizers/quantizer/` |
| Dimension-aware predictor | `modules/predictors/lorenzo/` |

## Phase 2 — Scaffold, then first build

```bash
scripts/new_stage.sh <Name> <category>        # StageType = highest existing + 1
cmake --preset release -DBUILD_TESTING=ON -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build/release -j<N>             # first build: a few minutes
```

Don't edit the stage's files while that first build runs; a half-built tree gives
confusing link errors.

The scaffold creates the module dir (`.h`, `.cu`, `CMakeLists.txt`), the test file,
the `StageType` entry, and the catalog and test registration, then prints every
remaining manual step. Never pass an ID that fills a gap in the enum: retired IDs
live on in old `.fzm` files.

The scaffold is a **plain (non-template) class** with `FZ_REGISTER_SIMPLE_STAGE`. For a
templated stage, convert it like `zigzag_stage.h/.cu`: class template, a header
factory that dispatches on the stored `DataType` registered with
`FZ_REGISTER_STAGE_FACTORY`, explicit instantiations in the `.cu`, and matching
`extern template` declarations in the header. This is real work; budget for it.

## Phase 3 — Implement (Steps 2-3b)

Fill every `TODO` the script left. Contracts that break silently if missed:

- `estimateOutputSizes` covers **both** directions; forward-only under-allocates the
  inverse buffer. Over-estimating is fine; under-estimating corrupts memory.
- Set `actual_output_size_` (or each named output's size) to the bytes really written.
- Enqueue everything on `stream`. No `cudaDeviceSynchronize`, no other streams. A sync
  on your own stream is allowed; document it.
- Allocate device memory through the `MemoryPool`, never `cudaMalloc`; persistent
  scratch needs `estimateScratchBytes`.
- Use the backend facades (`backend/types.h`, `backend/api.h`, `backend/cub.h`), not
  `<cuda_runtime.h>`, in stage code.
- **No buffer starts zeroed.** Pool memory is reused across calls and, with buffer
  coloring, shared with other stages' earlier buffers. So:
  - write every byte anything reads (padding, unused bits of the last word, tails);
  - clear before any read-modify-write (`|=`, `atomicOr`, counters, histograms) or
    sparse write, with `cudaMemsetAsync` on `stream`;
  - never write to an input buffer; persistent scratch keeps last call's contents.
- Shared output slots (packing, reductions) need exclusive ownership or atomics. An
  `atomicOr` on a 32-bit word needs a 4-byte-aligned address.
- Integer promotion: int8/int16 promote to `int`. Mask or cast back before shifting
  or zigzag-encoding, or negative values corrupt.
- Floating point: keep error bounds, ranges and bases in the input precision. Never
  narrow `double` through `float` (setters, header, TOML, kernels).
- Handle `n == 0` and sizes that aren't a multiple of the block/tile/word size.
- Persistent memory has two mechanisms: `pool->allocate(..., /*persistent=*/true)`
  (count it in `estimateScratchBytes`), or `pool->allocatePersistentDevice` /
  `allocatePersistentPinned` in `onFinalize()` (count it in
  `estimateDeviceFootprintBytes` / `estimatePinnedFootprintBytes`). Pinned readback
  slots belong in `onFinalize()`.

### Variable-size output (coders, anything whose output size depends on the data)

- How the size reaches the host: the DAG asks `getActualOutputSize*()` right after
  `execute()`, and `compress()` calls `postStreamSync()` at the end. The usual pattern
  (RZE, RLE): enqueue an async D2H copy of the total into a pinned slot, then sync and
  read it lazily in the size getters and in `postStreamSync()`.
- If a downstream stage consumes this output (`setTerminalOutput(false)`), publish
  the exact size before `execute()` returns; see `setTerminalOutput` in
  `include/stage/stage.h`.

Size bounds:
- Forward: the worst case including all headers and metadata. With a raw fallback,
  that's metadata + n.
- Inverse: cache the original size in the stage header (and in `saveState` /
  `restoreState`). Never use the compressed size as the decoded-size bound.

Raw fallback (coders): a per-chunk flag in the stream, store raw when encoded >= raw,
and include it in the worst-case bound. Test it with incompressible input.

## Phase 4 — Integrate (Steps 5-8b)

The scaffold's summary lists these; do all of them.

- `src/pipeline/config.cpp`: include, `add<Name>Stage` + `save<Name>Stage`, and one
  `kStageRegistry[]` entry with **five** fields:
  `{ "<Name>", StageType::<UPPER>, add<Name>Stage, save<Name>Stage, "modules/<category>/<lower>" },`
  Save every parameter the load helper reads.
- `include/fzgpumodules.h`: header include (alphabetical within category).
- `tests/pipeline/test_stage_registry.cpp`: add `StageType::<UPPER>` to `shipped[]`.
- `include/fzm_format.h`: replace the scaffold's `TODO` comment on the enum value.
- Catalog tables that list every stage: a row in `docs/mainpage.md` (stage table) and
  in `docs/config_file.md` (TOML type → class table).
- Docs: `docs/stages/<lower>.md` starting `# <Name>Stage {#stage_<lower>}`. Use
  `docs/stages/diff.md` as the layout (header, class, ports, types, constraints, TOML
  block, example), plus a `\subpage stage_<lower>` row in `docs/stages/<category>.md`.
- Attribution when based on prior work: `.cu` comment, `@note` in the class Doxygen,
  `## Acknowledgements` in the stage doc, `THIRD_PARTY.md` entry (Step 8b).
- `CHANGELOG.md` under `[Unreleased]` → `Added`: one line naming the class.
- CLI `--stages` builder (`src/utils/cli/cli.cpp`): only if you decided yes in Phase 1.

## Phase 5 — Tests (Step 9)

Fill the five generated tests (RoundTrip, ZeroInput, SerializeDeserialize,
SaveRestoreState, PipelineIntegration). ZeroInput checks `n == 0` on
the stage directly, because a `Pipeline` rejects an empty input. Templated stages:
make the tests typed (`TYPED_TEST`) over every type the factory and TOML accept.

Then add these, which past bugs showed are needed:

- **Fresh-decoder file round trip**: `fz_test::pipeline_file_round_trip` (writes the
  archive, decodes in a new pipeline). It catches header and factory bugs.
- **A reference check**: compare the forward output with a simple host
  implementation, not only round-trip equality (a wrong-but-invertible kernel
  round-trips).
- **Awkward sizes**: n = 1, an odd n, a size not a multiple of the block/chunk/word,
  and one large size (a few million elements, crosses grid limits).
- **Edge inputs**: all zeros, a constant non-zero value, all ones.
- **Second call**: two different inputs through one pipeline; the second result must
  equal a fresh pipeline's (catches stale buffers and persistent scratch).
- **Coders**: incompressible input (random bytes) must round-trip even if it expands.
  The scaffold's tests use float data with a tolerance; for a byte coder, compare
  bytes exactly.
- **Floating-point / lossy stages**: f64 as well as f32 with a relative error bound,
  through the file round trip.
- **TOML**: `saveConfig` → `loadConfig` with non-default parameters. Check the saved
  text, then compare the loaded pipeline's compressed bytes with a pipeline built in
  code (a `Pipeline` doesn't expose the stages it loaded). Pattern:
  `tests/pipeline/test_config.cpp` (`ConfigSave.PreservesParams`).

Lossless stages: compare bytes exactly (`EXPECT_EQ`), not with a tolerance.

## Phase 6 — Verify

```bash
.claude/skills/fzgm-add-stage/verify_stage.sh --static <lower_name>   # file checks only, seconds
.claude/skills/fzgm-add-stage/verify_stage.sh <lower_name>            # everything, e.g. xor_delta
cmake --build build/release -j<N> && ctest --preset default            # then the full suite
```

`verify_stage.sh` runs the docs/integration checks, finds leftover TODOs (including
the enum line), checks the registry test, umbrella header, docs, CHANGELOG and the
file/TOML/second-call tests, builds the stage's tests plus the registry, config and
file-I/O tests, runs them, and runs the stage's tests under Compute Sanitizer
(memcheck, racecheck, initcheck). When the build supports `FZ_POISON_ALLOCATIONS`,
it also runs the stage's tests with pool memory poisoned to `0xA5` and `0x00`
(otherwise it prints a WARN that this check is unavailable). It also warns on skipped
tests, and ends with
`RESULT: PASS` or a list of what to fix.

The full rebuild before `ctest --preset default` matters: shared headers changed, and
`ctest` does not rebuild stale test binaries.

On a shared GPU, prefix GPU commands with a lock: `FZ_GPU_WRAPPER="<lock command> --"`
(for example compression_benchmarking's `python -m benchkit lock --`).

For a CUDA "illegal address" error: it is reported at the next sync, not in the
faulting kernel. Use `CUDA_LAUNCH_BLOCKING=1` and Compute Sanitizer to localize it
before reading code (the `systematic-debugging` skill applies).

## Done means

Report, with the actual output: `verify_stage.sh` ended in `RESULT: PASS`, the full
rebuild + `ctest --preset default` passed (counts), the design card (including the
questions you'd ask the user), and anything skipped or deferred (CLI, specialization).
Don't claim done from a build alone.
