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

- Work on a new branch, ideally in its own worktree, not on a shared checkout:
  `git worktree add -b stage/<name> ../FZGPUModules-<name> main`
- A fresh clone or worktree needs submodules, or CMake fails on googletest:
  `git submodule update --init --recursive`
- Configure with tests for the GPU you have (the default architecture is 86):
  `cmake --preset release -DBUILD_TESTING=ON -DCMAKE_CUDA_ARCHITECTURES=native`
  then `cmake --build build/release -j$(nproc)`. The first build takes a while;
  start it now and design while it runs.

## Phase 1 — Design card (before any code)

Answer these in a short note to the user (or yourself). If a question can't be
answered from the request, ask; don't guess a file format.

| Question | Why it matters | Where it lands |
|---|---|---|
| Name (PascalCase, no `Stage` suffix) and category | directory, StageType, TOML name | Step 1 category definitions |
| Input/output element types; templated? | `getInput/OutputDataType`, factory dispatch, TOML type keys | Steps 2, 5, 7 |
| Size-preserving or variable-size? Forward *and* inverse size bounds | `estimateOutputSizes` must be safe in both directions | Step 2 |
| Ports: one output, or named outputs (`codes`, `outliers`, ...)? | `getOutputNames`, downstream wiring | Step 2 |
| What config goes in the header (<= 128 bytes)? Does decode overwrite forward config? | `serializeHeader`, `saveState/restoreState` | Step 2 |
| Scratch memory, input alignment, any host sync / D2H in `execute()`? | `estimateScratchBytes`, `getRequiredInputAlignment`, `isGraphCompatible` | Step 2 |
| Based on prior work (code or a specific paper's algorithm)? | attribution in 4 places | Step 8b |
| Does it need to fuse with neighbours later? | specialization contract, after it round-trips | `docs/pipeline_specialization_internals.md` |

Pick the nearest existing stage as a template and say which:

| Shape of the new stage | Copy from |
|---|---|
| Size-preserving, element-wise | `modules/transforms/zigzag/` |
| Variable-size lossless coder | `modules/coders/rle/` (templated), `modules/coders/rze/` (chunked bytes), `modules/coders/bitpack/` |
| Multi-output (codes + outliers) | `modules/quantizers/quantizer/` |
| Dimension-aware predictor | `modules/predictors/lorenzo/`, `modules/predictors/diff/` |

## Phase 2 — Scaffold

```bash
scripts/new_stage.sh <Name> <category>        # StageType ID auto-assigned
```

It creates the module dir (`.h`, `.cu`, `CMakeLists.txt`), `tests/stages/test_<lower>.cpp`,
the `StageType` entry, the catalog and test registration, and prints what's left.
Never renumber or reuse a `StageType` value: they're stored in `.fzm` files.

## Phase 3 — Implement (Steps 2-3b)

Fill every `TODO` the script left. Contracts that break silently if missed:

- `estimateOutputSizes` covers **both** directions; forward-only under-allocates the
  inverse buffer. Over-estimating is fine; under-estimating corrupts memory.
- Set `actual_output_size_` (or each named output's size) to the bytes really written.
- Enqueue everything on `stream`. No `cudaDeviceSynchronize`, no other streams. A sync
  on your own stream is allowed only with `isGraphCompatible() == false`, documented.
- Allocate device memory through the `MemoryPool`, never `cudaMalloc`; persistent
  scratch needs `estimateScratchBytes`.
- Use the backend facades (`backend/types.h`, `backend/api.h`, `backend/cub.h`), not
  `<cuda_runtime.h>`, in stage code.
- Shared output slots (packing, reductions) need exclusive ownership or atomics; `|=`
  after a memset is still a race. `atomicOr` on a 32-bit word needs a 4-byte-aligned
  address: don't atomically OR into a byte offset.
- Integer promotion: small signed types (int8/int16) promote to `int` in arithmetic.
  Mask or cast back before shifting or zigzag-encoding, or negative values corrupt.
- Floating point: keep error bounds, ranges, and bases in the input precision. Don't
  narrow `double` through `float` anywhere (setters, header, TOML, kernels).
- Handle `n == 0` and sizes that aren't a multiple of the block/tile/word size.

## Phase 4 — Integrate (Steps 5-8b)

- `src/pipeline/config.cpp`: include, `add<Name>Stage` + `save<Name>Stage`, one
  `kStageRegistry[]` entry. Save every parameter the load helper reads.
- `include/fzgpumodules.h`: add the header include (alphabetical within category).
- `tests/pipeline/test_stage_registry.cpp`: add `StageType::<UPPER>` to `shipped[]`
  (**not mentioned in the doc**; the integration script doesn't catch it either).
- `src/utils/cli/cli.cpp` `--stages` builder: optional, only for general-purpose stages.
- Docs: `docs/stages/<lower>.md` starting `# <Name>Stage {#stage_<lower>}` (copy the
  layout of `docs/stages/zigzag.md`: header, class, category, ports, types,
  constraints, TOML keys, example), plus a `\subpage stage_<lower>` row in the
  category page `docs/stages/<category>.md`.
- Attribution when based on prior work: `.cu` comment, `@note` in the class Doxygen,
  `## Acknowledgements` in the stage doc, `THIRD_PARTY.md` entry (Step 8b).
- `CHANGELOG.md` under `[Unreleased]` → `Added`: one line.

## Phase 5 — Tests (Step 9)

Fill the six generated tests (RoundTrip, ZeroInput, SerializeDeserialize,
SaveRestoreState, PipelineIntegration, GraphCompatible), then add these, which past
bugs showed are needed:

- **Fresh-decoder file round trip**: `fz_test::pipeline_file_round_trip` (writes the
  archive, decodes in a new pipeline). This is what catches header/factory bugs.
- **Awkward sizes**: n = 1, an odd n, and n not a multiple of the block/chunk size.
- **Edge inputs**: all zeros, and a constant non-zero value. A constant input once
  crashed the default Huffman pipeline on real data.
- **Coders**: incompressible input (random bytes) must round-trip even if it expands.
- **Floating-point / lossy stages**: f64 as well as f32, and a relative error bound,
  through the file round trip. f64 bound bugs only showed up that way.
- **Templated stages**: every type the factory and TOML loader accept.
- **TOML**: build a pipeline with the stage, `saveConfig` → `loadConfig`, and check
  parameters survive and the loaded pipeline round-trips.

Lossless stages: compare bytes exactly (`EXPECT_EQ`), not with a tolerance.

## Phase 6 — Verify

```bash
.claude/skills/fzgm-add-stage/verify_stage.sh --static <lower_name>   # file checks only, seconds
.claude/skills/fzgm-add-stage/verify_stage.sh <lower_name>            # everything, e.g. xor_delta
```

On a shared GPU, prefix GPU commands with a lock: `FZ_GPU_WRAPPER="<lock command> --"`
(for example compression_benchmarking's `python -m benchkit lock --`).

It checks integration (`scripts/check_docs.py`), leftover TODOs, registry/umbrella/docs/
CHANGELOG presence, builds, runs the stage's tests plus registry/config/file-I/O tests,
and runs the stage's tests under Compute Sanitizer (memcheck, racecheck, initcheck).
It ends with `RESULT: PASS` or a list of failures. Then run the full suite once:
`ctest --preset default`.

For a CUDA "illegal address" error: it is reported at the next sync, not in the
faulting kernel. Use `CUDA_LAUNCH_BLOCKING=1` and Compute Sanitizer to localize it
before reading code (the `systematic-debugging` skill applies).

## Done means

Report, with the actual output: `verify_stage.sh` ended in `RESULT: PASS`,
`ctest --preset default` passed (counts), the design card, and anything skipped or
deferred (CLI, specialization). Don't claim done from a build alone.
