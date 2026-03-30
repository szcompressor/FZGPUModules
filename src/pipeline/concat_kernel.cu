// concat_kernel.cu — GPU gather kernel for pipeline output concatenation (§2B)
//
// Replaces N individual cudaMemcpyAsync D2D calls in writeConcatBuffer() with
// a single kernel launch.  One block per segment, threads stride in 16-byte
// uint4 chunks for coalesced access; scalar tail loop handles remainder bytes.
//
// This is the prerequisite for CUDA Graph capture (§7): with a single kernel
// node at the concat boundary, the entire compress path becomes capturable as
// one graph with a fixed, O(1) update pattern instead of N copy-node patches.

#include "pipeline/concat_kernel.h"
#include <cuda_runtime.h>

namespace fz {

__global__ void gather_kernel(const CopyDesc* __restrict__ descs, int n_segs) {
    const int seg = blockIdx.x;
    if (seg >= n_segs) return;

    const uint8_t* __restrict__ src   = descs[seg].src;
    uint8_t* __restrict__       dst   = descs[seg].dst;
    const size_t                bytes = descs[seg].bytes;

    if (src == nullptr || dst == nullptr || bytes == 0) return;

    // Fast path: both pointers and the byte count are 16-byte aligned →
    // use uint4 (128-bit) loads/stores for peak memory bandwidth.
    const bool aligned = (reinterpret_cast<uintptr_t>(src) % 16 == 0) &&
                         (reinterpret_cast<uintptr_t>(dst) % 16 == 0) &&
                         (bytes % 16 == 0);
    if (aligned) {
        const size_t n_wide = bytes / 16;
        for (size_t i = threadIdx.x; i < n_wide; i += blockDim.x) {
            reinterpret_cast<uint4*>(dst)[i] =
                reinterpret_cast<const uint4*>(src)[i];
        }
        return;
    }

    // Fallback path: byte-at-a-time copy, still fully parallel across threads.
    // Handles arbitrary alignment of src, dst, and segment size.
    // This path is taken for head/tail segments whose offsets in the concat
    // buffer are not 16-byte aligned (e.g. after a 4-byte num_buffers header).
    for (size_t i = threadIdx.x; i < bytes; i += blockDim.x) {
        dst[i] = src[i];
    }
}

void launch_gather_kernel(
    const CopyDesc* d_descs,
    int             n_segs,
    int             block_dim,
    cudaStream_t    stream
) {
    if (n_segs <= 0) return;
    gather_kernel<<<n_segs, block_dim, 0, stream>>>(d_descs, n_segs);
}

} // namespace fz
