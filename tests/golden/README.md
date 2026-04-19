# Golden Reference Files

This directory contains committed reference FZM files used by
`tests/pipeline/test_golden.cpp` to catch regressions in the FZM file format.

## Files

| File | Pipeline | N | Error bound | Notes |
|---|---|---|---|---|
| `ref_lorenzo_bitshuffle.fzm` | Lorenzo(float,uint16) → Bitshuffle | 4096 | 1e-2 | single-source, v3.1 |

## What the golden tests check

1. **Version currency** — the committed file's `FZM_VERSION` field matches the
   current `FZM_VERSION` constant.  If the version was bumped without updating
   the golden file this test fails, prompting a regeneration.

2. **Decompression correctness** — `decompressFromFile` on the committed file
   produces output within the original error bound.  This verifies backward
   compatibility: the current code can still read files written at that version.

3. **Compressed size stability** — re-compressing the same input produces a
   compressed size within ±5% of the reference.  A large deviation indicates a
   regression in the compression pipeline that may not show up in round-trip
   tests (which only check correctness, not output stability).

## Regenerating after a format change

If `FZM_VERSION` is bumped or a stage's serialization format changes, regenerate
the golden files:

```bash
cmake --preset release
cmake --build --preset release --target generate_golden_files
```

This runs `tests/golden/regenerate` which writes fresh reference files to this
directory.  Commit the new files alongside the version bump.

## Design note

Golden files are intentionally small (N=4096 floats = 16 KB raw) so they are
cheap to commit and fast to load in tests.  The regression value comes from
the *stability* of the format across code changes, not from large-scale coverage.
