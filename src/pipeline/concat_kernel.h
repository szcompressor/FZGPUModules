#pragma once

#include "backend/api.h"
#include <cstddef>
#include <cstdint>

namespace fz {

/**
 * Descriptor for one segment in the gather kernel.
 * CPU packs an array of these into a pinned buffer; one H2D copy delivers
 * all descriptors to the device before the kernel is launched.
 */
struct CopyDesc {
    const uint8_t* src;   ///< source device pointer (pool buffer)
    uint8_t*       dst;   ///< destination device pointer (concat buffer + offset)
    size_t         bytes; ///< bytes to copy for this segment
};

/// Block-rows assigned to each segment by launch_gather_kernel().
///
/// Constant, not derived from segment sizes: a CUDA Graph bakes the launch
/// configuration in at capture time, so making the grid depend on the data would
/// silently invalidate a captured graph the first time a differently-sized input
/// was replayed through it.
constexpr unsigned kGatherBlocksPerSegment = 256;

/**
 * Gather kernel: copies N independent device-side segments in one launch.
 *
 * Grid:    (kGatherBlocksPerSegment, n_segs)
 * Block:   block_dim threads (typically 256)
 * All blocks in a segment's row cooperate, striding through it in 16-byte
 * (uint4) chunks for coalesced access, then handling tail bytes with a scalar
 * loop.
 *
 * Replaces N individual cudaMemcpyAsync D2D calls with a single kernel
 * launch, eliminating (N-1) CUDA API roundtrips on the CPU hot path and
 * enabling all segments to copy in parallel on the SMs.
 *
 * The block-row (rather than one block per segment) matters whenever segment
 * sizes are uneven. GPULZ split mode puts ~99% of the output in `literals`, so
 * a one-block-per-segment grid had a single 256-thread block copying ~23 MB —
 * 2.27 ms and 60.7% of the pipeline's total GPU time, at ~20 GB/s on hardware
 * capable of 3 TB/s.
 *
 * @param descs    Device pointer to array of CopyDesc (one per segment)
 * @param n_segs   Number of segments (== gridDim.y)
 */
void launch_gather_kernel(
    const CopyDesc* d_descs,
    int             n_segs,
    int             block_dim,
    cudaStream_t    stream
);

} // namespace fz
