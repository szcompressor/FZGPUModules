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
 *
 * Chunk size is a per-instantiation template parameter (see `chunk_fused_body`/
 * `chunk_inverse_pfpl_body` in chunk_fusion.cuh), not a single fixed constant —
 * every supported size gets its own compiled kernel, the same way the RRE-family
 * coders already dispatch `word_size`/`chunk_size` combinations via a switch over
 * pre-instantiated template kernels. `TEMP_BYTES` and `TPB` stay plain constants:
 * both are chunk-size-independent (TEMP_BYTES is sized for the largest supported
 * chunk and reused unscaled for smaller ones, matching the convention already
 * used by RZE/RRE's own `RZE_TEMP`/`RRE_TEMP`; see lc_chunk_components.cuh).
 */

namespace fz {
namespace fused {
namespace chunk {

constexpr int TEMP_BYTES = 4096;   // LC coder scratch (RZE/RRE/RARE/RAZE/CLOG/HCLOG all 4096)
constexpr int TPB        = 512;    // = LC coder TPB

/// Chunk sizes the chunk-cooperative fusion harness is instantiated for. Must
/// each be a power of 2 >= 4096 (matches lc_chunk_components.cuh's own
/// `d_RZE`/`d_RRE`/... static_assert) and a multiple of `1024 * element_width`
/// for whatever Bitshuffle element width composes with them (4-byte codes here,
/// so a multiple of 4096 — see BitshuffleStage::getRequiredInputAlignment()).
/// Extending this set requires no harness redesign: add the value here, add a
/// dispatch case in chunk_fusion.cu's per-size switches, and widen each
/// participating stage's getFusionSpec() gate to accept it.
constexpr int kSupportedChunkBytes[] = {4096, 8192, 16384};
constexpr int kNumSupportedChunkBytes =
    sizeof(kSupportedChunkBytes) / sizeof(kSupportedChunkBytes[0]);

// Host-only (launcher validation). Excluded from NVRTC's device-JIT compile
// (__CUDACC_RTC__, defined automatically by NVRTC) via #ifndef below: NVRTC
// rejects any unannotated function merely being *present* in the translation
// unit in JIT mode (not just called), and this header can't use
// __host__/__device__ either, since it's also included from plain host C++
// translation units where those tokens aren't defined at all (no CUDA
// compiler). Geom<>'s static_assert below inlines the same check as a literal
// expression instead of calling this, so it works unmodified under NVRTC too.
#ifndef __CUDACC_RTC__
constexpr bool isSupportedChunkBytes(int bytes) {
    return bytes == 4096 || bytes == 8192 || bytes == 16384;
}
#endif

/// Per-chunk-size derived geometry, instantiated once per supported size.
template <int Bytes>
struct Geom {
    static_assert(Bytes == 4096 || Bytes == 8192 || Bytes == 16384,
                 "chunk-cooperative chunk size must be one of kSupportedChunkBytes");
    static constexpr int CHUNK_BYTES = Bytes;
    static constexpr int NELEM       = Bytes / 4;   // uint32 codes per chunk
    static constexpr int NPP         = NELEM / 32;  // bitshuffle words per plane
};

} // namespace chunk
} // namespace fused
} // namespace fz
