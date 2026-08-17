#pragma once

/**
 * @file modules/fused/chunk_fusion/chunk_geometry.h
 * @brief Chunk-cooperative fusion geometry — single source of truth.
 *
 * Deliberately dependency-free (no backend/runtime headers): it is included by
 * the device harness (chunk_fusion.cuh), the host launcher (chunk_fusion.h/.cu),
 * AND the runtime-generated NVRTC source. The NVRTC device compile must NOT pull
 * host runtime types (backend/types.h names cudaStream_t etc., which NVRTC does
 * not provide), so the constants live here, apart from the launcher declarations.
 */

namespace fz {
namespace fused {
namespace chunk {

constexpr int CHUNK_BYTES = 16384;             // one chunk (bytes)
constexpr int NELEM       = CHUNK_BYTES / 4;   // 4096 uint32 codes
constexpr int NPP         = NELEM / 32;        // 128 bitshuffle words per plane
constexpr int TEMP_BYTES  = 4096;              // LC coder scratch (RZE/RRE both 4096)
constexpr int TPB         = 512;               // = LC coder TPB

} // namespace chunk
} // namespace fused
} // namespace fz
