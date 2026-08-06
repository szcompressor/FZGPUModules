// concat_kernel.cu — GPU gather kernel for pipeline output concatenation (§2B)
//
// Replaces N individual cudaMemcpyAsync D2D calls in writeConcatBuffer() with
// a single kernel launch.  Threads stride in 16-byte uint4 chunks (bulk path)
// then handle the remaining <16 bytes in a scalar tail.
//
// Grid shape: (kBlocksPerSegment, n_segs).  The original version launched ONE
// block per segment, which collapsed on any pipeline whose segments are very
// unevenly sized — precisely the split-mode GPU-Zstd case it exists for.  There
// the `literals` port is ~99% of the output, so a single 256-thread block was
// copying ~23 MB: measured 2.27 ms, 60.7% of the whole pipeline's GPU time, at
// ~20 GB/s on a 3 TB/s H100.  Worse, it sat AFTER the DAG event bracket, so it
// was invisible in dag_elapsed_ms and only showed up as a 2.9x host/device gap.
//
// kBlocksPerSegment is a compile-time constant rather than a function of the
// segment size on purpose: the launch configuration must not depend on the data
// for CUDA Graph capture to stay valid across replays with different inputs
// (see the graph note below).  Blocks with nothing to do exit after one
// descriptor read, which costs nothing next to the copy they enable.
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
#include "backend/api.h"

namespace fz {

__global__ void gather_kernel(const CopyDesc* __restrict__ descs, int n_segs) {
    const int seg = blockIdx.y;
    if (seg >= n_segs) return;

    const uint8_t* __restrict__ src   = descs[seg].src;
    uint8_t* __restrict__       dst   = descs[seg].dst;
    const size_t                bytes = descs[seg].bytes;

    if (src == nullptr || dst == nullptr || bytes == 0) return;

    // Every block in this segment's block-row participates, striding by the full
    // row width.  A segment smaller than one stride is finished by block 0 and
    // the rest fall out of the loop immediately.
    const size_t tid    = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;

    // Bulk path: src and dst are both 16-byte aligned (see alignment contract
    // above).  Copy as many uint4 (128-bit) chunks as possible for peak
    // memory bandwidth, then handle the tail with a byte-at-a-time loop.
    const bool aligned = (reinterpret_cast<uintptr_t>(src) % 16 == 0) &&
                         (reinterpret_cast<uintptr_t>(dst) % 16 == 0);
    if (aligned) {
        const size_t n_wide = bytes / 16;
        for (size_t i = tid; i < n_wide; i += stride) {
            reinterpret_cast<uint4*>(dst)[i] =
                reinterpret_cast<const uint4*>(src)[i];
        }
        // Scalar tail: at most 15 bytes.  `tid` is unique across the whole block
        // row, so only threads with tid < 15 enter the loop and each tail byte is
        // still written exactly once.
        const size_t tail_start = n_wide * 16;
        for (size_t i = tid; i < bytes - tail_start; i += stride) {
            dst[tail_start + i] = src[tail_start + i];
        }
        return;
    }

    // Fallback: unaligned pointers (should not occur under the normal allocation
    // path, but handles edge cases safely).
    for (size_t i = tid; i < bytes; i += stride) {
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
    // Data-independent launch config — see the graph note in the file header.
    // 256 block-rows x 256 threads gives 64K threads per segment, enough to
    // saturate an H100 on the one large segment while the small ones cost a
    // descriptor read each.
    const dim3 grid(kGatherBlocksPerSegment, static_cast<unsigned>(n_segs));
    gather_kernel<<<grid, block_dim, 0, stream>>>(d_descs, n_segs);
}

} // namespace fz
