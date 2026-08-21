#pragma once

/**
 * @file modules/fused/chunk_fusion/nvrtc_chunk_fusion.h
 * @brief Runtime NVRTC codegen for the chunk-cooperative fusion harness.
 *
 * The compile-time path (chunk_fusion.cuh) instantiates `chunk_fused_kernel<...>`
 * for a fixed op list chosen in C++. This path does the same thing at RUNTIME:
 * a ChunkFusionSpec (the stage-chain fingerprint — quant op, transform ops,
 * coder op, all by device-op type name) is turned into CUDA source that wraps
 * `chunk_fused_body<...>`, compiled with NVRTC, JIT-loaded via the CUDA driver
 * API, and launched. Swapping the coder or the transform set is now a data
 * change to the spec — no new C++ template instantiation, no registry glue.
 *
 * The per-stage __device__ ops stay hand-written in chunk_fusion.cuh; only the
 * composing glue is generated. See docs/codebase_notes.md CN-NVRTC-FUSE.
 *
 * Selected by env FZ_FUSION_NVRTC=1 inside launchFusedChunkPfpl(); off by
 * default (the template path is the fallback). Requires the source tree to be
 * present at the configured include paths (FZGMOD_NVRTC_INC_*) — a prototype
 * constraint; a shipped build would embed the op headers instead.
 */

#include "fused/chunk_fusion/chunk_fusion.h"   // ChunkCoderKind, chunk geometry
#include "backend/types.h"
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace fz {
namespace fused {

/// The stage-chain fingerprint the codegen composes into one fused kernel. Each
/// field is a device-op *type name* defined in chunk_fusion.cuh. Defaults spell
/// the PFPL body; a planner would fill these from the fused group's stages.
struct ChunkFusionSpec {
    std::string              quant_op   = "QuantInplaceZigzag";
    std::vector<std::string> transforms = {"DiffNegabinary", "Bitshuffle32"};
    std::string              coder      = "RZECoder";
};

/// Map a coder enum to its device-op name (the swappable sink).
inline const char* chunkCoderOpName(ChunkCoderKind k) {
    return (k == ChunkCoderKind::RRE) ? "RRECoder" : "RZECoder";
}

/// True if NVRTC + the CUDA driver are usable in this process (they are, in any
/// normal CUDA runtime — kept as a hook for graceful fallback).
bool nvrtcChunkFusionAvailable();

/// The CUDA source the codegen emits for `spec` (exposed for tests/inspection).
std::string generateChunkFusionSource(const ChunkFusionSpec& spec);

/**
 * Compile (once, then cached) and launch the fused encode kernel for `spec`,
 * filling `d_scratch` (nc * CHUNK_BYTES) and `d_sizes` (nc). `d_params` is the
 * device-resident packed per-op Params blob (ops in execution order); the kernel
 * hands each op its slice. Mirrors the launch of chunk_fused_kernel<...>; the
 * cross-chunk scan/pack tail is shared with the template path in chunk_fusion.cu.
 * Throws std::runtime_error on compile failure.
 */
/// `side_*` feed the harness Map op's escaping outputs (e.g. an outlier list). Pass
/// nullptr/0 when the composed op produces none — the generated kernel takes the args
/// unconditionally (an unused `ChunkSideCtx{}` is harmless). `d_side_count` must be a
/// device counter the caller has zeroed; the Map op atomically appends into it.
void launchNvrtcChunkFusedEncode(
    const ChunkFusionSpec& spec, const float* d_in, size_t n, const uint8_t* d_params,
    uint8_t* d_scratch, uint32_t* d_sizes, unsigned nc, fz::stream_t stream,
    uint32_t* d_side_idxs = nullptr, float* d_side_vals = nullptr,
    uint32_t* d_side_count = nullptr, uint32_t side_max = 0);

/**
 * Generic chunk-cooperative fused compress — the entry the generic registry runner
 * uses, with no per-pipeline shape. NVRTC-composes `spec` (any linear
 * Map -> Transform* -> Coder chain of ChunkCooperative ops), uploads the packed
 * per-op params blob (`host_params`/`params_bytes`, ops in execution order), runs
 * the encode kernel plus the shared cross-chunk scan/pack tail, and returns the
 * archive byte length. Always uses NVRTC — the only way to compose an arbitrary
 * runtime op list — but the compiled module is cached by (source, arch), so only
 * the first compress of a given chain pays the JIT cost.
 */
/// When `d_side_idxs`/`d_side_vals` are non-null the composed Map op is a split-outlier
/// producer: this allocates and zeroes a device append-counter, runs the encode so the
/// op fills the two side buffers (capacity `side_max` elements each), and writes the
/// final outlier count back to `*out_side_count` (a stream sync happens before it
/// returns, so the count and the packed archive are both ready).
size_t launchGenericChunkFusion(
    const ChunkFusionSpec& spec, const float* d_in, size_t n,
    const uint8_t* host_params, size_t params_bytes,
    uint8_t* d_out, MemoryPool* pool, fz::stream_t stream,
    uint32_t* d_side_idxs = nullptr, float* d_side_vals = nullptr,
    uint32_t side_max = 0, uint32_t* out_side_count = nullptr);

} // namespace fused
} // namespace fz
