# CLAUDE.md

Guidance for Claude Code (or any other AI coding agent) when working in this repository.

## What is FZGPUModules

FZGPUModules is a **CUDA library for building composable, high-throughput error-bounded
compression pipelines**. Rather than picking one monolithic compressor, you assemble a
directed acyclic graph (DAG) of independent GPU stages — predictors, quantizers, coders,
shufflers, transforms, and fused kernels — connected and executed entirely on the GPU with
stream-ordered memory management. It vendors/reimplements ideas from cuSZ, cuSZp/cuSZp2/cuSZp3,
cuSZ-Hi (LC), FZ-GPU, and PFPL as modular, mix-and-matchable stages.

## The pipeline model

Every pipeline is built with `fz::Pipeline` (`include/pipeline/compressor.h`):

```cpp
fz::Pipeline p(input_bytes, fz::MemoryStrategy::PREALLOCATE, /*pool_mult=*/4.0f);
p.setDims(nx, ny, nz);                 // before addStage, for dimension-aware stages
auto* stage = p.addStage<StageType>();
p.connect(downstream, upstream, "codes");   // named port, not always "output"
p.finalize();                          // validates, sorts into levels, (pre-)allocates
p.compress(d_input, input_bytes, &d_comp, &comp_sz, stream);
p.decompress(d_comp, comp_sz, &d_decomp, &decomp_sz, stream);
```

Stages at the same DAG level run concurrently on separate CUDA streams. See
[docs/architecture.md](docs/architecture.md) for the full layer model (`Pipeline` →
`CompressionDAG` → `Stage`/`MemoryPool`) and [docs/api_reference.md](docs/api_reference.md)
for the complete lifecycle/enum reference.

## Stage categories (`modules/`)

| Category | Directory | What it does | Docs |
|---|---|---|---|
| Predictors | `modules/predictors/` | Model-based decorrelation (Lorenzo, tiled Lorenzo, diff/delta) | [docs/stages/predictors.md](docs/stages/predictors.md) |
| Quantizers | `modules/quantizers/` | Lossy discrete mapping (error-bounded quantizer) | [docs/stages/quantizers.md](docs/stages/quantizers.md) |
| Transforms | `modules/transforms/` | Invertible basis changes (zigzag, negabinary, ADM, log) | [docs/stages/transforms.md](docs/stages/transforms.md) |
| Coders | `modules/coders/` | Symbol-to-bitstream encoding (RLE, RZE, RRE, bitpack, adaptive bitpack, Huffman, ANS) | [docs/stages/coders.md](docs/stages/coders.md) |
| Shufflers | `modules/shufflers/` | Lossless data restructuring for better compressibility (bitshuffle) | [docs/stages/shufflers.md](docs/stages/shufflers.md) |
| Fused | `modules/fused/` | Combined multi-operation kernels (lorenzo_quant, GInterp, bitplane_rze) | [docs/stages/fused.md](docs/stages/fused.md) |
| Structural | `modules/structural/` | DAG plumbing, not a compression operation (merge N ports → 1) | [docs/stages/structural.md](docs/stages/structural.md) |

Full per-stage constraints, ports, and TOML keys: [docs/stages/index.md](docs/stages/index.md).
How to add a brand-new stage: [docs/how_to_add_a_stage.md](docs/how_to_add_a_stage.md).

## Build

```bash
cmake --preset release                          # default (no examples/tests/profiling)
cmake --preset release -DBUILD_EXAMPLES=ON      # with examples
cmake --preset release -DBUILD_TESTING=ON       # with tests
cmake --preset release -DBUILD_PROFILING=ON     # with profiling
cmake --build build/release -j$(nproc)
```

Test binaries go to `build/release/tests/`, example binaries to `build/release/bin/examples/`,
profiling binaries to `build/release/bin/profiling/`. Full option/preset list:
[docs/building.md](docs/building.md).

## Testing

```bash
ctest --preset default            # all tests, release build
ctest --preset stages             # stage unit tests only
ctest --preset pipeline           # pipeline integration tests only
ctest --preset asan               # full suite, ASan + UBSan     — see note below
ctest --preset compute-san        # full suite, compute-sanitizer — see note below
```

**Both sanitizer presets need environment setup, and both fail misleadingly without
it** — `compute-san` reports every test as `Not Run` behind a single "Unable to find
executable: compute-sanitizer" line, and `asan` aborts all 48 at startup with "ASan
runtime does not come first in initial library list". Neither looks like a missing
tool. On the JetStream2 H100 box, `source ~/load-env` puts `compute-sanitizer` on
PATH and defines `fz-asan`:

```bash
source ~/load-env
ctest --preset compute-san        # works once compute-sanitizer is on PATH
fz-asan ctest --preset asan       # wraps with LD_PRELOAD=libasan + ASAN_OPTIONS
```

`LD_PRELOAD` must not be exported globally — it would inject ASan into `nvcc` and
every other process. `ASAN_OPTIONS=protect_shadow_gap=0` is required under CUDA, and
`detect_leaks=0` because the CUDA runtime holds allocations for process lifetime by
design. Run `compute-san` on any change to device code: it catches out-of-bounds
device writes that land in pool slack, which a normal test run reports as passing.

`tests/` layout: `stages/` (per-stage unit tests), `pipeline/` (integration tests),
`golden/` (reference data), `helpers/`.

## Directory layout

```
include/            public headers (only these in user/example code)
modules/             stage implementations, see table above
src/pipeline/        Pipeline, DAG, MemoryPool internals
examples/            example programs (-DBUILD_EXAMPLES=ON)
profiling/           profiling programs (-DBUILD_PROFILING=ON)
tests/               test suite (-DBUILD_TESTING=ON)
docs/                doxygen-driven documentation site (mainpage, architecture, stages/, api_reference)
memory/              local working documents and release notes — gitignored, not part of the shipped repo
scripts/             helper scripts
```

## Key rules

- Never include `cuda_check.h` or use `FZ_CUDA_CHECK` in example/user code — use plain CUDA calls.
- `RLEStage` is a template: `RLEStage<uint16_t>`, never `RLEStage`.
- Lorenzo/Quantizer downstream must connect to the `"codes"` port, not the default `"output"`.
- Pool-owned pointers (compress output, default decompress output) must NOT be `cudaFree`'d;
  see the Memory Ownership table in [docs/architecture.md](docs/architecture.md).
- Call `pipeline.setDims()` before `addStage()` for dimension-aware stages (Lorenzo,
  GInterp, etc.) — dims are pushed into the stage at add-time (and again at finalize as
  a safety net).
- `MemoryStrategy::PREALLOCATE` is required for CUDA Graph capture; `MINIMAL` throws.

## Changelog

Whenever you make a code change — new feature, fix, refactor, or removal — add an entry to
`CHANGELOG.md` under the appropriate `[Unreleased]` subsection (`Added`, `Changed`, `Fixed`,
or `Removed`) before finishing the task. Keep entries concise (one line per logical change).
Do not add entries for documentation-only or comment-only edits.

## Where to go deeper

| Topic | Page |
|---|---|
| Architecture (layers, DAG scheduling, memory ownership) | [docs/architecture.md](docs/architecture.md) |
| Performance tuning (every lever, when to use each, measured effect sizes) | [docs/performance_tuning.md](docs/performance_tuning.md) |
| Full API reference (lifecycle, enums, all setters) | [docs/api_reference.md](docs/api_reference.md) |
| Per-stage reference (ports, constraints, TOML keys) | [docs/stages/index.md](docs/stages/index.md) |
| Adding a new stage | [docs/how_to_add_a_stage.md](docs/how_to_add_a_stage.md) |
| FZM binary file format | [docs/fzm_format.md](docs/fzm_format.md) |
| CLI and TOML config syntax | [docs/cli.md](docs/cli.md), [docs/config_file.md](docs/config_file.md) |
| Building from source (all presets/options) | [docs/building.md](docs/building.md) |
