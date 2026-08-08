# Baseline Provenance

Every third-party compressor the FZGM paper reports numbers for: where it came
from, what we changed, and why. This discharges the disclosure obligation in the
paper's Experimental Setup section — baselines we modified are held to a higher
standard than our own code, because we benefit from the modifications.

Regenerate and verify everything here with:

```bash
./artifact/capture_provenance.sh           # rebuild patches/
./artifact/capture_provenance.sh --check   # fail if committed patches drifted
```

Patches live in `patches/`; `patches/MANIFEST.txt` lists the exact commit of
every baseline.

---

## 1. Read this first: the tree named `cuSZp-V2.0.1` is not upstream V2.0.1

The local trees under `~/compressors/` are laid out as:

| Tree | What it actually is |
|---|---|
| `cuSZp-V2.0.1` | upstream tag **+ our shared fixes** — *not* pristine |
| `cuSZp-V2.0.1_optimized` | base + third-party H100 kernel optimizations |
| `cuSZp-V2.0.1_split` | base + third-party restructuring (count→scan→pack) |
| `cuSZp-V3.0.0` | upstream tag **+ our shared fixes** — *not* pristine |
| `cuSZp-V3.0.0_optimized` | base + third-party H100 kernel optimizations |

The naming invites the reading that `cuSZp-V2.0.1` is stock upstream. It is not.
Anywhere the paper says "unmodified cuSZp," it must mean the upstream tag, not
this tree. There is no local git history in any of these directories, so the
patches in `patches/cuszp/` were reconstructed by diffing against the release
tags fetched fresh from GitHub.

**Upstream:** `https://github.com/szcompressor/cuSZp`

| Tag | Commit | Date |
|---|---|---|
| `cuSZp-V2.0.1` | `240240b2f8cc3984236c6cea1f18658ef535a81c` | 2024-11-17 |
| `cuSZp-V3.0.0` | `da90afb42770fcef6129cb586cf185c7d54c614c` | 2025-10-24 |

---

## 2. Which variant produces the paper's numbers

**The headline native-vs-FZGM comparison uses the base trees only.** The
benchkit adapter resolves `CUSZP2_CLI` / `CUSZP3_CLI`, and both env scripts
(`scripts/env-jetstream2.sh`, `scripts/env-bigred200.sh`) point at
`cuSZp-V2.0.1/` and `cuSZp-V3.0.0/`. The `_optimized` and `_split` trees are
referenced *only* by the dedicated `configs/experiments/cuszp{2,3}_variants.yaml`
studies.

This matters: the most sensitive modifications (§4, third-party kernel
optimizations) are **not** in the headline comparison. What the headline
comparison carries is the §3 set — a correctness backport and two measurement
artifact removals, all of which help cuSZp rather than handicap it.

---

## 3. Base-tree changes (upstream tag → `cuSZp-V{2.0.1,3.0.0}`)

Patches: `patches/cuszp/v2.0.1_upstream-to-base.patch` (594 lines),
`patches/cuszp/v3.0.0_upstream-to-base.patch` (1,405 lines).
Full narrative writeup: `compression_benchmarking/docs/adapters/cuszp.md` §§169–334.

Four changes, three categories. **None handicaps cuSZp; three actively help it.**

### 3a. `excl_sum` uninitialized-shared-memory fix — *correctness* (V2 only)

`src/cuSZp_kernels_f32.cu` / `_f64.cu`, four call sites each. In the decoupled
look-back scan, `excl_sum` is `__shared__` and assigned only inside
`if (warp > 0) {...}`, but every warp then reads it at
`base_idx = excl_sum + rate_ofs`. Warp 0 therefore read whatever was last in that
shared-memory bank — undefined behavior. On this H100 it corrupted output on
roughly 11 of 24 test cells.

Warp 0 has no predecessor, so its exclusive prefix sum is 0 by definition; the fix
sets `excl_sum = 0` on that path.

**Why this is the right call:** the bug is already fixed upstream in V3 and simply
was not backported to V2. We are not improving cuSZp — we are applying the
author's own later fix so V2 produces correct output at all. Without it, cuSZp2
numbers on this hardware would be meaningless. Disclosed as a backport, with the
V3 code as the reference.

### 3b. Scratch-buffer caching — *measurement artifact* (V2 and V3)

`src/cuSZp_entry_*.cu`. Upstream `cudaMalloc`s and `cudaFree`s three small scratch
arrays (`d_cmpOffset`, `d_locOffset`, `d_flag`) on **every call**. That is cheap on
bare metal but pathological on the JetStream2 GPU-passthrough VM, where a single
`cudaMalloc` has been observed to take hundreds of ms against a ~145 µs kernel.
Replaced with a grow-once cache keyed on `cmpOffSize`, still `memset` to 0 every
call, freed at process exit.

**Why it doesn't advantage us:** it removes a platform artifact from *cuSZp's*
timing, making cuSZp faster. Correctness is unchanged.

### 3c. `TIMING_REPEATS` batching — *measurement artifact* (V2 and V3)

`examples/cuSZp.cpp`. Times an average over 100 back-to-back launches inside one
event pair instead of a single launch. On the same VM, single-shot
`cudaEventElapsedTime` reported ~3.2 ms for a call whose true kernel time was
~145 µs by nsys/hardware counters — a fixed per-launch dispatch tax, not execution
time. Batching amortizes it. The algorithm is deterministic, so `cmpSize` is
identical every rep.

**Also makes cuSZp faster.** Note this is the same class of hazard the paper's own
measurement rules cover (cold-run and dispatch-latency effects).

### 3d. `CMAKE_CUDA_ARCHITECTURES += 90` — *build only*

Upstream's list stops at 86; the H100 is sm_90. Without this there is no native
cubin for the test hardware. No source change.

---

## 4. Variant trees — third-party contributions (attribution required)

**These optimizations were not written by the paper's authors.** The
`_optimized` and `_split` trees under `cuSZp-V2.0.1` were produced by an outside
collaborator (Yuxi Hong, `yuxilab`) on their own cluster, using an agentic coding
tool; the raw session transcripts were vendored into the trees as
`.yuxi_notes/` (four files, ~540 KB).

**Two actions this forces:**

1. **Attribution.** If any variant number appears in the paper, the contribution
   must be credited — acknowledgements at minimum, and the AI-assisted origin
   noted if the venue's policy requires it (several now do). Check the target
   venue's disclosure rules before submission.
2. **`.yuxi_notes/` must never ship.** The transcripts contain a third party's
   email address, organization name, and internal cluster paths. They are
   excluded by `capture_provenance.sh`, and the committed patches are verified
   free of that content. Re-check after any regeneration.

### What the variants change

**`_optimized`** (`patches/cuszp/v2.0.1_base-to-optimized.patch`,
`v3.0.0_base-to-optimized.patch`) — two H100 kernel changes:

- **Register/local-memory spill fix.** The per-thread `absQuant[thread_chunk]`
  array (2–4 KB/thread, spilled to local memory) becomes `absQuant[32]`, with
  values recomputed on the fly rather than held.
- **Scan restructuring.** The single `excl_sum` decoupled look-back is replaced by
  per-warp totals plus a block-level scan.

Both are pure performance work on cuSZp's own kernels — they make the *baseline*
faster, which is the safe direction for us.

> ⚠️ The spill-vs-recompute trade is hardware- and codebase-specific and has been
> observed to regress elsewhere. Do not present it as a general result.

**`_split`** (`patches/cuszp/v2.0.1_base-to-split.patch`) — the `_optimized`
changes **plus** a restructuring of compress into distinct count → scan → pack
phases.

> ⚠️ **`_split` is the one to be careful with.** It changes cuSZp's algorithm
> structure, not just its implementation. If a headline comparison ever uses
> `_split` rather than `_optimized`, justify it explicitly as the structural
> equivalent of the FZGM pipeline and report both numbers. As of this writing the
> headline comparison uses neither (§2), which is the cleanest position — keep it
> that way if possible.

---

## 5. Other baselines

Commits and dirty state: `patches/MANIFEST.txt`. Local modifications:
`patches/baselines/<name>_local.patch`. All are git checkouts, so upstream
identity is recoverable; the patches capture uncommitted working-tree changes.

| Baseline | Tag / commit | Local changes | Category |
|---|---|---|---|
| cuSZ | `v0.17.3` (`e1c0135f`) | 6 files, 98+/22− | CLI + executor plumbing |
| cuSZ-Hi | `e16b75b` | 5 files, 46+/13− | context + comparison plumbing |
| FZ-GPU | `v.0.1-3-gc7e83f7` | 2 files, 53+/12− | adds compressed-output file I/O + stream param |
| PFPL | `36f5aae` | makefile, 2 lines | build |
| lsCOMP | `d920dd9` | CMakeLists, 1 line | build |
| MANS | `sc-ae-21-g7e9265f` | CMakeLists, 3 lines | build |
| SZ3 | `v3.3.2-10-ge08c0ba` | none | — |
| FSZ | `v1.0.0` (`43240ed`) | none | — |
| zfp | `0.5.0-1195-g5bf9376` | none | — |
| MGARD | `1.6.0-3-gb40d1a73` | none (untracked build only) | — |
| SPERR | `v0.8.5-1-gb801258` | none (untracked build only) | — |
| tthresh | `2dbad4e` | none (untracked build only) | — |

The FZ-GPU change is the only non-build source edit here: upstream's tool never
writes the compressed buffer to disk, so round-trip and compression-ratio
measurement were impossible without adding it. It adds output plumbing and a
caller-supplied stream; the compression path itself is untouched.

**nvCOMP** (`nvcomp-5.2.0.10`, `nvcomp-5.3.0.16`) is a closed-source vendor
binary — no source, no patch. Record the exact release used. Any nvCOMP number
must carry the caveats already established: the vendor-default chunk size cost
2.1×, and the lossless-gate carve-outs are load-bearing.

---

## 6. Open items

- [ ] Decide whether any `_optimized` / `_split` number appears in the paper. If
      yes, add the acknowledgement and check the venue's AI-disclosure policy (§4).
- [x] **Confirmed 2026-08-07: base-tree binaries are not stale.** Every binary
      post-dates the newest source file in its tree, so the §3 fixes are compiled
      in. `cuSZp-V2.0.1`: newest source 07-19 21:49, `build/` binary 07-19 21:49
      (kernels with the `excl_sum` fix are 21:45), `build_sm90/` 07-21 18:06.
      `cuSZp-V3.0.0`: newest source 07-19 15:34, `build/` 07-19 15:34,
      `build_sm90/` 07-22 01:08. The `CUSZP{2,3}_CLI` env vars point at `build/`,
      both of which also post-date the 07-19 13:03 `CMakeLists.txt` sm_90 change.
      Re-check after any baseline rebuild.
- [ ] Ideally report unmodified-upstream cuSZp2 alongside the fixed base for at
      least one field, to show §3 helps rather than hurts. Note this is only
      meaningful where the `excl_sum` bug does not corrupt output (§3a).
- [ ] Pin the nvCOMP release actually used in reported numbers (two are present).
