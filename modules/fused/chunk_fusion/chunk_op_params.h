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

} // namespace chunk
} // namespace fused
} // namespace fz
