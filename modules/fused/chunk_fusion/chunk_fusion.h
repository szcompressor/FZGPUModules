#pragma once

/**
 * @file modules/fused/chunk_fusion/chunk_fusion.h
 * @brief Public launcher for chunk-cooperative fused compress (see chunk_fusion.cuh).
 *
 * Runs the composable CTA-per-chunk harness for a PFPL-shaped chain
 * (Quantizer(inplace,zigzag) -> Difference(negabinary) -> Bitshuffle -> <Coder>)
 * plus the cross-chunk scan+pack tail, producing a raw LC-coder archive that is
 * byte-identical to the staged pipeline. The coder is selected at call time — the
 * same harness composes RZE, RRE, etc. Not graph-capturable (reads back the
 * archive length). See docs/codebase_notes.md CN-CHUNK-FUSE.
 */

#include "backend/types.h"
#include <cstddef>
#include <cstdint>

namespace fz {
class MemoryPool;

namespace fused {

/// The variable-length coder that terminates a chunk-fused chain (the swappable sink).
enum class ChunkCoderKind { RZE, RRE };

/**
 * Run a PFPL-shaped chunk-cooperative fused compress into a raw coder archive.
 * @param coder   which LC coder terminates the chain (RZE or RRE)
 * @param d_in    device float field (n elements)
 * @param n       element count
 * @param ebx2_r  1/(2*abs_eb) quant scale (abs_eb = eb*value_range for NOA)
 * @param radius  quant radius (out-of-radius -> inplace raw-float-bits outlier)
 * @param threshold |x| >= threshold forces an outlier (pass +inf to disable)
 * @param d_out   device archive buffer (LC format: [orig][n_chunks][sizes][payloads])
 * @param pool    scratch pool
 * @param stream  CUDA stream (synchronised internally to read the archive length)
 * @return total archive length in bytes (4-byte rounded, matching the staged coder)
 */
size_t launchFusedChunkPfpl(
    ChunkCoderKind coder, const float* d_in, size_t n, float ebx2_r,
    uint32_t radius, float threshold, uint8_t* d_out, MemoryPool* pool, fz::stream_t stream);

} // namespace fused
} // namespace fz
