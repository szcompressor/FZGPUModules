#pragma once

#include <cuda_runtime.h>
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

/**
 * Gather kernel: copies N independent device-side segments in one launch.
 *
 * Grid:    one block per segment  (gridDim.x == n_segs)
 * Block:   block_dim threads      (typically 256)
 * Each block strides through its segment in 16-byte (uint4) chunks for
 * coalesced access, then handles the tail bytes with a scalar loop.
 *
 * Replaces N individual cudaMemcpyAsync D2D calls with a single kernel
 * launch, eliminating (N-1) CUDA API roundtrips on the CPU hot path and
 * enabling all segments to copy in parallel on the SMs.
 *
 * @param descs    Device pointer to array of CopyDesc (one per segment)
 * @param n_segs   Number of segments (== gridDim.x)
 */
void launch_gather_kernel(
    const CopyDesc* d_descs,
    int             n_segs,
    int             block_dim,
    cudaStream_t    stream
);

} // namespace fz
