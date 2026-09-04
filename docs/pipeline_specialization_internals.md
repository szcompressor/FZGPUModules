# Making a stage specialization-compatible {#pipeline_specialization_internals}

Specialization is **declaration-driven**. A stage declares
its fused identity; the planner walks the DAG matching those declarations by role;
the runner builds the fused kernel from the declared op names. A correctly
declared stage joins the fast path the same way an existing one does.

---

## The two layers

Every specializable stage spans two layers that must agree exactly:

1. **Host declaration** (`Stage` virtuals in the stage's `.h`) — *what* device op
   this stage maps to, its role, geometry, and runtime parameter bytes.
2. **Device policy** (a small POD in a harness `.cuh`) — the *how*: the actual
   register/shared-memory kernel code, carrying **both** the forward and inverse
   methods. Forward and inverse are two methods on one policy class, so a stage
   that fuses on compress can fuse on decompress from the same declaration.

The host packs a parameter blob whose layout the device op `reinterpret_cast`s, so
the POD `Params` struct is **shared verbatim** between host and device (see
`modules/fused/fused_block/warp_op_params.h`,
`modules/fused/chunk_fusion/chunk_op_params.h`).

---

## Roles and strategies

`include/stage/fusion.h` defines the vocabulary.

`FusionAccess` — a stage's data-access pattern, which decides *how* it can fuse:

| Role | Meaning | Example |
|---|---|---|
| `Map` | element-wise `out[i] = f(in[i])` | linear Quantizer |
| `BlockLocal` | bounded, resettable neighbourhood in a fixed block | 1-D Lorenzo (block reset) |
| `Cooperative` | warp/CTA reduce+scan producing variable-length output (a coder) | AdaptiveBitpack |
| `TileAdaptive` | one selector tile containing N coder units | FSZ selector |
| `Unfusable` (default) | opaque / global dependency — a fusion barrier | Huffman (global codebook) |

`FusionStrategy` — which execution model the op belongs to. A fused group is
composed of ops that all share one strategy:

| Strategy | Execution model |
|---|---|
| `WarpRegister` | one warp owns a `≤ 32·kMaxWarpElemsPerLane`-element block, intermediates in registers + shuffles, no barriers (cuSZp / SZp) |
| `ChunkCooperative` | one CTA owns a compatible size byte-chunk, intermediates in shared memory, `__syncthreads` between ops (LC / PFPL) |

The rest of this guide works the **warp-register** path end to end. The chunk-cooperative path uses the same declaration surface with a different harness.

---

## Forward declaration

<!-- doc-check: skip — class-member fragments, not standalone TUs -->
```cpp
// Predictor example (LorenzoStage). Role = BlockLocal.
FusionSpec getFusionSpec() const override {
    if (isInverse() || block_size_ == 0) return {};            // Unfusable
    return FusionSpec{FusionAccess::BlockLocal, block_size_};   // role + reset period
}

FusedOpDecl getFusedOp() const override {
    if (isInverse() || block_size_ % 32u != 0 ||
        block_size_ / 32u > fused::warp::kMaxWarpElemsPerLane) return {};
    FusedOpDecl d;
    d.strategy       = FusionStrategy::WarpRegister;
    d.op_name        = "Lorenzo1DPredictor";               // the device policy TYPE name
    d.include_header = "fused/fused_block/warp_fusion.cuh"; // where the policy lives
    d.elems_per_lane = block_size_ / 32u;                  // block_size = 32 * EPL
    d.n_ab           = 0;                                  // 0 => runner uses input elem count
    fused::warp::Lorenzo1DParams p{0.0f, block_size_ / 32u};
    d.params.resize(sizeof(p));                            // raw POD bytes
    std::memcpy(d.params.data(), &p, sizeof(p));
    return d;
}
```

Conventions:

- **op_name is the device policy type name.** The NVRTC codegen instantiates the
  harness with exactly this identifier, so it must name a type in `include_header`.
- **params is the POD's raw bytes.** The generated kernel casts the packed blob to
  the POD type; host and device layouts must match. Stateless ops leave it empty.
- **The `inv2eb` slot convention.** Every warp predictor's `Params` begins with
  `float inv2eb` at offset 0. The predictor stage cannot know the error bound (the
  quantizer owns it), so it packs `0` there; the runner overwrites those 4 bytes
  with `1/(2·abs_eb)` resolved from the primed quantizer bound before uploading. The
  quantizer is absorbed into the predictor (it quantizes inline in `delta()`), which
  is why the Map/quant stage declares op `"LinearQuant"` with empty params.
- **elems_per_lane** (= `block_size / 32`) is the harness's compile-time template
  arg; **n_ab** is the padded block-covering element count (0 = 1-D, no padding).

The coder (Cooperative, the group tail) declares similarly with `op_name` naming its
coder policy (`"PlainRateCoder"` / `"AdaptiveBitpackCoder"`).

---

## Inverse declaration

The mirror surface, gated on `isInverse()`. This is what makes a stage fuse on
**decompress**, not just compress. It reuses the same `FusionSpec` / `FusedOpDecl`
structs and the same policy type names (the policy carries the inverse methods).

<!-- doc-check: skip -->
```cpp
FusionSpec getInverseFusionSpec() const override {
    if (!isInverse() || centeringActive() || block_size_ % 32u != 0 ||
        block_size_ / 32u > fused::warp::kMaxWarpElemsPerLane) return {};
    return FusionSpec{FusionAccess::BlockLocal, block_size_};   // same role as forward
}
FusedOpDecl getInverseFusedOp() const override {
    if (!getInverseFusionSpec().fusable()) return {};
    FusedOpDecl d;
    d.strategy       = FusionStrategy::WarpRegister;
    d.op_name        = "Lorenzo1DPredictor";   // same policy; its undelta() is used
    d.include_header = "fused/fused_block/warp_fusion.cuh";
    d.elems_per_lane = block_size_ / 32u;
    return d;                                  // inverse needs no params (dequant step from quant)
}
```

Two generic scalar hooks let the inverse runner pull what it needs **without
`dynamic_cast`ing to a concrete stage type** — so a *new* coder/quant works too:

<!-- doc-check: skip -->
```cpp
// On the Cooperative/coder stage: how many elements the archive reconstructs to.
size_t getFusedInverseElementCount() const override { return num_elements_; }

// On the Map/quant stage: the linear dequant step (2*abs_eb) the harness multiplies by.
double getFusedInverseDequantStep() const override {
    return 2.0 * static_cast<double>(computed_abs_eb_);
}
```

Keep forward and inverse eligibility in lockstep (reuse the same block-size gating),
so a pipeline that fuses on compress also fuses on decompress. Guard anything the
inverse harness can't undo — e.g. Lorenzo excludes `centeringActive()` because the
`undelta` policy is plain first-difference.

---

## Priming: the fused runner bypasses execute()

A fused runner replaces the group's per-stage `execute()` calls with one kernel, so
any state a stage would normally compute in `execute()` — and that its **own
inverse** later reads — must be established explicitly. These `Stage` hooks
(all default no-ops) exist for that:

| Hook | Who overrides it | Why |
|---|---|---|
| `primeFusedForwardState(ctx)` | Quantizer | Run the value-range scan / bound resolution the fused kernel needs (and the inverse reads back). Called once per group member before codegen. |
| `setFusedArchiveResult(archive, orig)` | variable-length coders | Report archive + original sizes so the coder's inverse can size its output (else it falls back to the compressed size and overruns). |
| `setFusedInverseResult(bytes)` | inverse tail (quant) | Publish the reconstructed byte count for output-size refinement. |
| `setFusedSideOutput(port, bytes)` | outlier-producing quant | Report bytes written to an escaping side port (e.g. an outlier list) so `serializeHeader` matches the fused result. |

Without `primeFusedForwardState`, a fused
quantizer's inverse read the *default* bound (1e-4) instead of the resolved one —
producing a byte-identical archive that reconstructed 10× too small. If
your stage computes anything in `execute()` that its inverse depends on, it must be
primed. See `docs/codebase_notes.md` CN-FUSE-DRIVER / CN-CHUNK-WIRE.

**Allocation is handled for you.** When the group is installed, the DAG
automatically (a) skips the device allocation for any intermediate buffer produced
*and* consumed only inside the group — the fused kernel keeps it in
registers/shared memory — and (b) collapses the group to one liveness point so
`PREALLOCATE` buffer coloring can alias the rest of the DAG around it. Your stage
does nothing for this; just make sure `getFusedAuxOutputs()` / the side-output
hooks correctly identify any port that *escapes* the group (an outlier list, a
means stream) so it stays materialized.

---

## The device policy contract (warp-register harness)

`modules/fused/fused_block/warp_fusion.cuh`. Each policy is a small POD with static
device methods. The harness bodies (`fused_rate_body` / `fused_pack_body` /
`fused_unpack_body`) call these; they are the *only* thing that changes per op.

**Predictor** (produces per-lane int codes, forward and inverse):
<!-- doc-check: skip -->
```cpp
struct MyPredictor {
    /* fields */;
    __device__ static MyPredictor fromParams(const float* in, size_t n, const void* pp);
    __device__ int  delta(uint32_t lane, size_t b, int m) const;      // forward: float -> code delta
    template<int EPL>
    __device__ static void undelta(int (&d)[EPL], uint32_t lane);     // inverse: deltas -> codes
};
```

**Coder** (the Cooperative sink; forward cost/pack + inverse decode):
<!-- doc-check: skip -->
```cpp
struct MyCoder {
    static constexpr uint32_t meta_bytes = 1;   // per-block metadata stride
    template<int EPL> __device__ static void cost(const int (&d)[EPL], /*...*/);   // forward pass A
    template<int EPL> __device__ static void pack(const int (&d)[EPL], /*...*/);   // forward pass B
    template<int EPL> __device__ static void decode(const uint8_t* meta,          // inverse
                       const uint8_t* base, uint32_t word_bytes, size_t count,
                       uint32_t lane, int (&d)[EPL]);
};
```

**Transform** (optional register→register map between predictor and coder):
`apply<EPL>()` forward, `invert<EPL>()` inverse (applied in reverse order on
decode). *Current limitation:* the inverse harness does not yet compose interior
transforms, so a chain with an interior transform (e.g. Zigzag) fuses on compress
but stays staged on decompress until `invert()` + `applyInverseTransforms` land.

The forward and inverse methods of one policy **must** be exact inverses that also
match the staged stage's kernels bit-for-bit.

---

## Adding a new warp predictor — end-to-end checklist

1. **Device policy** in `warp_fusion.cuh`: add `MyPredictor` with `fromParams`,
   `delta`, and `undelta`. Add its `Params` POD to `warp_op_params.h` (leading
   `float inv2eb`).
2. **Forward declaration** on your stage: `getFusionSpec` → `BlockLocal`;
   `getFusedOp` → `{WarpRegister, "MyPredictor", elems_per_lane, params}`.
3. **Inverse declaration**: `getInverseFusionSpec` → `BlockLocal`;
   `getInverseFusedOp` → `{WarpRegister, "MyPredictor", elems_per_lane}`.
4. **Nothing else.** `matchesWarpRegister` / `runWarpRegister` and their inverse
   counterparts are role-based over the declarations — they build the
   `WarpFusionSpec` from your `op_name`, so a new predictor needs no registry,
   planner, matcher, or runner edit. The NVRTC codegen instantiates the harness with
   your policy name.
5. **Validate against the staged oracle** (see below) and add a test.

Adding a new **coder** is the same, at the Cooperative role, with `decode` alongside
`cost`/`pack`. Adding a new **quantizer dequant** is currently linear-only (the
harness hard-codes `code · 2·eb`); a non-linear dequant would extend the harness with
a dequant policy, the same way a predictor is added.

---

## How the matcher/runner consume declarations (why no planner edit)

`src/pipeline/fusion_registry.cpp` — role-based, mirrored for forward and inverse:

- **Matcher** (`matchesWarpRegister`): every stage declares a `WarpRegister` op;
  `front` is Map, one `BlockLocal` predictor, interior Map/BlockLocal transforms,
  `back` is Cooperative. No concrete types named.
- **Runner** (`runWarpRegister`): primes each stage, then builds
  `WarpFusionSpec{predictor = the BlockLocal op's name, coder = the Cooperative op's
  name, transforms = interior op names, elems_per_lane}`, patches `inv2eb` from the
  primed quant bound, and calls `launchNvrtcWarpFused`.

The inverse pair (`matchesWarpRegisterInverse` / `runWarpRegisterInverse`) is the
same, over `getInverseFusionSpec`/`getInverseFusedOp`, with roles reversed
(Cooperative coder → BlockLocal predictor → Map quant) and the two scalar hooks for
element count and dequant step.

The planner (`planFusionGroups`) only enumerates maximal fusable chains from
`getFusionSpec()` roles — it never needs to know your op exists.

---

## Validation requirements (non-negotiable)

A specialization is only correct if it is indistinguishable from staged:

1. **Byte-identical compress.** Compress the same input with `SpecializationPolicy::Off`
   and `::Auto`; the archives must be `memcmp`-equal. (For nondeterministic side
   outputs like an atomic-appended outlier list, compare *reconstruction* instead.)
2. **Byte-exact decompress.** Decompress the identical archive off vs auto; the
   reconstructions must be equal.
3. **compute-sanitizer** memcheck + racecheck clean on the fused kernels (warp
   shuffles and cooperative writes are easy to get subtly wrong).
4. **Partial final block.** Test an element count that is *not* a multiple of the
   block size — the most common source of fused-vs-staged divergence.
5. Add the cases to `tests/pipeline/test_fusion_planner.cpp` (see
   `Warp1DGeneralEplFusesMatchesStaged` for the pattern).

---

## Chunk-cooperative path (brief)

Same declaration surface (`getFusionSpec`/`getFusedOp` with
`FusionStrategy::ChunkCooperative`), different harness
(`modules/fused/chunk_fusion/`). Device ops are templated on chunk size
(`Op<ChunkBytes>`, one of `{4096, 8192, 16384}` — see `chunk_geometry.h`'s
`Geom<Bytes>`) with shared-memory ping-pong; the POD params live in
`chunk_op_params.h`. The harness is NVRTC-composed from an op list, with the
chunk size baked into the generated template args (`ChunkFusionSpec::chunk_bytes`),
so a novel `Map → Transform* → Coder` chunk chain fuses with zero new glue —
proven for `Quant → Diff → Bitshuffle → {RZE, RRE, RARE, RAZE}` at all three
sizes. All participating ops in a group must declare the same `block_size`
(chunk size) or the planner rejects the group. The NVRTC surface needs the op
headers to compile without the CUDA runtime; see CN-NVRTC-FUSE for the stub
requirements — note this also means any host-only helper in a header pulled
into the NVRTC translation unit must be `#ifndef __CUDACC_RTC__`-guarded, since
NVRTC rejects an unannotated function merely being *present*, not just called
(see `chunk_geometry.h`'s `isSupportedChunkBytes`). Decompress generalization
for this path beyond PFPL/RZE is future work.

---

## Deeper reading

- User guide: \ref pipeline_specialization "Pipeline Specialization" (in Performance Tuning)
- Adding a stage at all: [how_to_add_a_stage.md](how_to_add_a_stage.md)
- Future deep-dive on stage design + optimization decisions (WIP outline):
  [developing_stages_deep_dive.md](developing_stages_deep_dive.md)
