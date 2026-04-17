// concat_kernel.cu — GPU gather kernel for pipeline output concatenation (§2B)
//
// Replaces N individual cudaMemcpyAsync D2D calls in writeConcatBuffer() with
// a single kernel launch.  One block per segment, threads stride in 16-byte
// uint4 chunks (bulk path) then handle the remaining <16 bytes in a scalar tail.
//
// Alignment contract (enforced by calculateConcatSize / writeConcatBuffer):
//   - src: pool-allocated, always 256-byte aligned (cudaMallocFromPoolAsync).
//   - dst: concat buffer base is pool-allocated (256-byte aligned), and each
//     segment's slot start is padded to a 16-byte boundary, so dst is always
//     16-byte aligned.
//   - bytes: actual (unpadded) size; may be any value.
//
// With src and dst guaranteed to be 16-byte aligned, the bulk path always
// fires for the leading floor(bytes/16)*16 bytes.  Only the tail (<16 bytes)
// falls to scalar.  This makes the fast path the common path for all segments.
//
// This is the prerequisite for CUDA Graph capture (§7): with a single kernel
// node at the concat boundary, the entire compress path becomes capturable as
// one graph with a fixed, O(1) update pattern.

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

    // Bulk path: src and dst are both 16-byte aligned (see alignment contract
    // above).  Copy as many uint4 (128-bit) chunks as possible for peak
    // memory bandwidth, then handle the tail with a byte-at-a-time loop.
    const bool aligned = (reinterpret_cast<uintptr_t>(src) % 16 == 0) &&
                         (reinterpret_cast<uintptr_t>(dst) % 16 == 0);
    if (aligned) {
        const size_t n_wide = bytes / 16;
        for (size_t i = threadIdx.x; i < n_wide; i += blockDim.x) {
            reinterpret_cast<uint4*>(dst)[i] =
                reinterpret_cast<const uint4*>(src)[i];
        }
        // Scalar tail: at most 15 bytes, handled by the first few threads.
        const size_t tail_start = n_wide * 16;
        for (size_t i = threadIdx.x; i < bytes - tail_start; i += blockDim.x) {
            dst[tail_start + i] = src[tail_start + i];
        }
        return;
    }

    // Fallback: unaligned pointers (should not occur under the normal allocation
    // path, but handles edge cases safely).
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
