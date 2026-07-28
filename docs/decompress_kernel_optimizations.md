# Decompress kernel optimizations (H100, 2026-07)

Reference notes for the round of decompress-path kernel work on the cuSZp2/cuSZp3
pipelines: what was slow, how it was diagnosed, what changed, why, and what's left.
All changes are **decompress-only** and **bit-exact** (archive format unchanged);
compress kernels were not touched.

Companion material: [performance_tuning.md](performance_tuning.md) for the full lever
catalog; the profiling harness is described in the repo's profiling/ setup.

---

## TL;DR — measured results (H100, NYX temperature 512³ unless noted)

| pipeline | stage | before | after | speedup |
|---|---|---:|---:|---|
| cuszp3 | TiledLorenzo inverse | 2.34 ms | 0.74 ms | **3.2×** |
| cuszp3 (plain) | AdaptiveBitpack decode | 1.54 ms | 0.61 ms | **2.5×** |
| cuszp3 (outlier) | AdaptiveBitpack decode | 1.45 ms | 0.74 ms | **2.0×** |
| cuszp2 | Lorenzo 1-D inverse | 2.52 ms | 0.50 ms | **5.1×** |
| cuszp2 | AdaptiveBitpack decode | 1.66 ms | 1.13 ms | **1.5×** |

**Pipeline-level decompress DAG:** cuszp3 3-D 4.34 → 1.80 ms (**2.4×**); cuszp2
4.11 → 2.09 ms (**2.0×**). End-to-end fraction of native decompress throughput
roughly doubled for cuszp3 (17.8% → 29% plain, 23.3% → 36% outlier).

---

## The diagnostic method (reusable)

Two techniques did all the work of locating the bottlenecks. Use them before writing
any kernel code.

### 1. Per-stage bandwidth decomposition, with the Quantizer as a control

`fzgmod-cli -b --profile` prints a per-stage **decompress** DAG timing table (the
compress side only prints the aggregate — a known limitation). In the cuszp3/cuszp2
decompress DAGs the **Quantizer inverse is a trivial elementwise dequant that runs at
~1200 GB/s(in) ≈ HBM bandwidth**. Use it as a control:

- It proves the DAG / MemoryPool / stream machinery imposes ~zero overhead — any stage
  far below the Quantizer's rate is **algorithmically** inefficient, not wiring-bound.
- A stage running 4–5× below the Quantizer is the target.

The gap to *native* (single fused kernel) decomposes as **~3× irreducible
materialization** (each modular stage does a full DRAM round-trip where native keeps
intermediates in registers/shared) **+ per-stage inefficiency**. Only the second part
is recoverable without fusion — and it was the larger part here.

**The gap is non-uniform by shape/size.** On small/2-D arrays (e.g. CESM-2D, 26 MB)
FZGM is near parity because native itself isn't bandwidth-bound there; the modular tax
only bites once native saturates HBM on large 3-D/1-D arrays. Do not trust the aggregate
"% of native" — profile per shape class.

### 2. ncu pipe analysis to pick the fix

The two slow stages had **different** root causes; ncu was essential to tell them apart:

- **Barrier-bound** (TiledLorenzo, plain Lorenzo): `sm__throughput` ~50% at ~90%
  *resident* warps → warps resident but stalled at `__syncthreads`. Fix = remove
  barriers / raise parallelism.
- **ALU-bound** (AdaptiveBitpack decode): `sm__inst_executed_pipe_alu` 85% vs
  `pipe_lsu` 29%, DRAM 15%, `l1tex...sectors_per_request` = 1 (loads already optimal).
  Fix = cut ALU instruction count; touching memory would do nothing.

ncu needs `sudo -E env "PATH=$PATH" "LD_LIBRARY_PATH=$LD_LIBRARY_PATH" ncu ...` on this
VM (`RmProfilingAdminOnly=1` restricts counters to root).

---

## Optimization 1 — TiledLorenzo inverse (cuszp3, 2-D/3-D)

**File:** `modules/predictors/tiled_lorenzo/tiled_lorenzo_stage.cu`
(`tiled_lorenzo_scan_kernel_rows`).

**Root cause.** The prior "phased" kernel used one CUDA block per tile and split the
separable inverse into three `__syncthreads`-separated phases (Z-chain → Y-chains →
X-chains). Even its busiest phase used only `ty*tz` of `tile_elems` threads (16 of 64
for a 4×4×4 tile), so a 64-thread block sat mostly idle across two barriers. ncu:
`sm__throughput` 51% at 91% resident warps — classic barrier stall.

**Fix.** One thread per **x-row** (= one `(tile, ly, lz)`), fully parallel and
self-contained. Exploiting the separable structure, each row thread re-derives its own
seed by walking the tile's tiny x=0 spine (`Σ d(0,0,k)` then `Σ d(0,k,lz)` — a handful
of L1-resident adds), then runs its own `tx`-length x-chain. No barriers, no idle lanes,
coalesced reads; writes are the inherent tile→natural scatter (unchanged).

**Reasoning / dead end.** The first attempt — one thread per tile doing the whole scan
in shared memory — *regressed* to 4.2 ms because it serialized the 64-element scan across
2M blocks at ~1 warp/SM. The lesson: the fix was never "put it in shared to hide global
latency"; it was "give every thread independent work." The per-row mapping is what
matters; the tiny redundant spine re-reads across a tile's rows are L1-cached and free.

**Result.** NYX 3-D 2.34 → 0.74 ms (3.2×, 229 → 727 GB/s); CESM-2D 0.143 → 0.069 ms
(2.1×). Output bit-identical to the phased kernel (git-stash A/B). The phased kernel is
retained but unreferenced. Correct for any tile shape (tests cover custom 16×16 and
partial-edge dims).

---

## Optimization 2 — AdaptiveBitpack decode (cuszp2/cuszp3, plain + outlier)

**File:** `modules/coders/adaptive_bitpack/adaptive_bitpack_kernels.cu`
(`warpBitTranspose32`, `decode_unpack_kernel_warp_tr`,
`decode_unpack_outlier_kernel_warp_tr`).

**Root cause.** The bit-plane decode reconstructs each element's magnitude by gathering
one bit per plane: `for p in 0..rate: av |= ((plane_p >> lane) & 1) << p` — **O(rate)**
ALU ops per element. ncu: ALU pipe 85%, LSU 29%, DRAM 15% → purely ALU-bound. On
high-entropy data (NYX temperature, rate ≈ 21) that's ~8·rate ≈ 170 ALU ops/lane.
Memory-side changes (e.g. staging the redundant byte loads to shared) would do nothing —
the loads were already 1 sector/request.

**Fix.** Reconstruct all 32 magnitudes of a warp's 32-element half with a single **32×32
warp bit-matrix transpose** (`warpBitTranspose32`): load each plane's 32-bit half-word
as one transpose row (0 past the rate), run a 5-step `__shfl_xor` butterfly, and lane *l*
ends up holding its rate-bit magnitude. Fixed cost, **independent of rate** — replaces
O(rate) per lane with O(log 32) per warp.

- **Hybrid, per block:** blocks with `rate < FZ_DECODE_TR` (default **6**, warp-uniform
  so no divergence) keep the cheaper O(rate) gather; the transpose wins above that.
  `FZ_DECODE_TR=0` disables it (debug/A-B). Env is read once.
- **Outlier mode** (`..._outlier_kernel_warp_tr`) unifies plain and outlier blocks under
  a `sign_off` offset (0 for plain, `ob_bytes` for outlier); element 0 of an outlier
  block is read from the outlier-byte prefix (its plane bits are 0, so its transposed
  magnitude is 0 and is simply overwritten). Covers cuszp2 and cuszp3_outlier.

**Reasoning / de-risking.** Hand-rolled warp-shuffle bit code is a silent-corruption
class. The butterfly was **verified in a standalone harness against a CPU transpose over
random matrices before integration**, then confirmed bit-exact against the gather path on
real NYX/HACC data via `cmp`.

**Result.** Plain: NYX bs=64 1.54 → 0.61 ms (2.5×), HACC bs=32 2.78 → 2.00 ms (1.39×),
CESM 0.049 → 0.043 ms (no low-rate regression). Outlier: NYX cuszp3_outlier 1.45 →
0.74 ms (2.0×). Bit-exact on both plain and outlier paths (bs=32 and bs=64).

**Watch-out.** `FZ_DECODE_TR=` (empty string) is **not** unset — `atoi("")=0` disables
the transpose. Use `env -u FZ_DECODE_TR` when A-B'ing the default in shell loops, or you
measure gather-vs-gather and wrongly conclude "no speedup."

---

## Optimization 3 — plain Lorenzo 1-D inverse (cuszp2)

**File:** `modules/predictors/lorenzo/lorenzo_stage.cu`
(`lorenzo_scan_1d_warp32_kernel`).

**Root cause.** After the decode transpose, cuszp2's aggregate decompress *still* hadn't
moved — because decode wasn't its bottleneck. The stage breakdown showed the plain
(non-tiled) `LorenzoStage` block-local inverse dominating: `lorenzo_scan_1d_kernel` did a
shared-memory Hillis-Steele scan (5 log-steps × 2 `__syncthreads` = **10 barriers**)
launched with **32-thread blocks** — one warp per 32-element reset segment, capped at
~50% occupancy on the blocks-per-SM limit, paying block barriers for a pure intra-warp
scan.

**Fix.** A 32-element reset segment is exactly one warp, so the prefix sum is a
barrier-free `__shfl_up` warp scan (5 shuffles, no shared memory). Launch wide 256-thread
blocks (8 segments each) for full occupancy; segments stay independent because the
shuffle width is 32 and each warp covers one 32-aligned segment. Only `block_size==32`
takes the fast path; other reset periods keep the Hillis-Steele kernel.

**Result.** NYX cuszp2 Lorenzo inverse 2.52 → 0.50 ms (5.1×, 212 → 1084 GB/s ≈ HBM
bandwidth); cuszp2 decompress DAG 4.11 → 2.09 ms (2.0×). Bit-exact (git-stash A/B). All
three cuszp2 decompress stages are now bandwidth-efficient.

---

## Verification (all three)

- **Bit-exact:** git-stash A/B (rebuild original kernel, `cmp` decompressed output) for
  TiledLorenzo and Lorenzo; gather-vs-transpose `cmp` for AdaptiveBitpack (plain + outlier,
  bs=32 + bs=64). All identical.
- **Unit tests:** `test_tiled_lorenzo` (15/15, incl. custom tiles + partial edges),
  `test_adaptive_bitpack` / `test_cuszp_block_sizes` (with transpose forced),
  `test_lorenzo` (14/14). Full suite passes (the only intermittent failure,
  `test_mempool_fallback`, is a pre-existing GPU-contention flake that passes serially).
- **compute-sanitizer:** memcheck + synccheck clean on all three (synccheck specifically
  because of the new warp-shuffle code).

---

## What's left: the materialization floor (fusion)

After these fixes, every cuszp2/cuszp3 decompress stage runs at or near HBM bandwidth
individually. The remaining ~3× gap to native is **structural**: native cuSZp does
unpack + inverse-Lorenzo + dequant in one fused kernel keeping intermediates in
registers/shared, while the modular pipeline round-trips each intermediate through DRAM
between stages. The Quantizer control run at ~1200 GB/s means three back-to-back
bandwidth-bound stages have a hard floor of ~3× the single fused kernel.

The only lever left with meaningful headroom is an **opt-in fused decompress stage** for
the cuszp hot path — trading some modularity (for that path) to eliminate the
inter-stage DRAM round-trips. This is a deliberate design tension (fusion vs.
recomposability) and the natural next investigation.

Separately unaddressed: the **1-D plain-Lorenzo path used by HACC** goes through the same
`lorenzo_scan_1d_warp32_kernel` now, but HACC's cuszp3_1d pipeline also has other stages;
and **cuSZ-Hi cr** remains gated on host-blocking syncs in GInterpStage (a different root
cause — see performance_tuning.md).
