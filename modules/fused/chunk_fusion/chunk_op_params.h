#pragma once

/**
 * @file modules/fused/chunk_fusion/chunk_op_params.h
 * @brief POD parameter blocks for the chunk-cooperative device-ops.
 *
 * Shared VERBATIM between the device op definitions (chunk_fusion.cuh, which
 * `using Params = ...`) and the host stages that pack these bytes into the fused
 * params blob. The generated kernel `reinterpret_cast`s the packed blob to these
 * types, so the two sides MUST see identical layouts — keep them plain POD and
 * dependency-free (no backend/runtime headers), so both the device compile
 * (nvcc/NVRTC) and the host stage compile agree.
 *
 * Stateless ops (Difference, Bitshuffle, the coders) have no entry here — they
 * contribute zero params bytes.
 */

#include <cstdint>

namespace fz {
namespace fused {
namespace chunk {

/// Params for the `QuantInplaceZigzag` Map op (inplace-outlier NOA/ABS quant).
/// `ebx2_r = 1/(2*abs_eb)`; out-of-radius / over-threshold values become raw
/// IEEE-754 bits. Filled from the quantizer's primed `computed_abs_eb_`.
struct QuantInplaceZigzagParams {
    float    ebx2_r;
    uint32_t radius;
    float    threshold;
};

/// Params for the `QuantSplitOutlier` Map op (3-port outlier NOA/ABS quant). Same
/// uniform-step layout as the inplace variant — the only difference is where the
/// outliers go (a side list vs. inline raw bits), not how codes are computed.
using QuantSplitOutlierParams = QuantInplaceZigzagParams;

/// Side-output context threaded to the harness Map op, so a Map that produces
/// escaping outputs (e.g. `QuantSplitOutlier`) can append them. `out_count` is a
/// GLOBAL device counter shared across every chunk's CTA (the outlier list is one
/// pipeline output spanning all chunks); the runner allocates and zeroes it. Ops
/// that produce no side output ignore this entirely (all fields may be null/0).
struct ChunkSideCtx {
    uint32_t* out_idxs  = nullptr;  ///< global element index of each outlier
    float*    out_vals  = nullptr;  ///< raw value of each outlier
    uint32_t* out_count = nullptr;  ///< global append counter (device, zeroed pre-launch)
    uint32_t  max       = 0;        ///< outlier list capacity (elements); appends past it drop
};

/// Marker Params for stateless ops (Difference, Bitshuffle, the coders). Contributes
/// zero bytes to the packed params blob (see OpParamBytes in chunk_fusion.cuh).
struct EmptyParams {};

/// Dependency-free type-equality (no <type_traits> — must compile under NVRTC, whose
/// stdlib is stubbed). Used to size a stateless op's params blob contribution to 0.
template <class A, class B> struct FzSame { static constexpr bool value = false; };
template <class A>          struct FzSame<A, A> { static constexpr bool value = true; };

} // namespace chunk
} // namespace fused
} // namespace fz
