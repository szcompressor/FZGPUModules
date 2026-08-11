#include "predictors/lorenzo/lorenzo_stage.h"
#include "mem/mempool.h"
#include "cuda_check.h"
#include "backend/api.h"
#include "backend/warp.h"
#include <stdexcept>
#include <string>

namespace fz {

// ─────────────────────────────────────────────────────────────────────────────
// 1-D kernels
// ─────────────────────────────────────────────────────────────────────────────

// Forward: d_output[i] = d_input[i] - d_input[i-1]  (d_input[-1] == 0)
// Blocks are independent — each block restarts its chain at 0.
// This matches the block-local model used by LorenzoQuantStage so that the
// inverse (prefix sum) is fully self-contained per block.
template<typename T>
__global__ void lorenzo_delta_1d_kernel(
    const T* __restrict__ in,
    T*       __restrict__ out,
    size_t n)
{
    const size_t block_offset = static_cast<size_t>(blockIdx.x) * blockDim.x;
    const size_t gid = block_offset + threadIdx.x;
    if (gid >= n) return;

    // Previous element: 0 at the start of each block.
    T prev = (threadIdx.x > 0) ? in[gid - 1] : static_cast<T>(0);
    out[gid] = in[gid] - prev;
}

// Inverse: parallel prefix sum (exclusive scan within each block)
// Same block-local model: each block's input is already self-contained.
template<typename T>
__global__ void lorenzo_scan_1d_kernel(
    const T* __restrict__ in,
    T*       __restrict__ out,
    size_t n)
{
    extern __shared__ char smem[];
    T* s = reinterpret_cast<T*>(smem);

    const size_t block_offset = static_cast<size_t>(blockIdx.x) * blockDim.x;
    const size_t gid = block_offset + threadIdx.x;
    const int   tid = static_cast<int>(threadIdx.x);

    s[tid] = (gid < n) ? in[gid] : static_cast<T>(0);
    __syncthreads();

    // Inclusive scan (Hillis-Steele)
    for (int stride = 1; stride < static_cast<int>(blockDim.x); stride <<= 1) {
        T val = (tid >= stride) ? s[tid - stride] : static_cast<T>(0);
        __syncthreads();
        s[tid] += val;
        __syncthreads();
    }

    if (gid < n) out[gid] = s[tid];
}

// Inverse for the block_size==32 case (cuSZp2/cuszp2): each 32-element reset
// segment is exactly one warp, so the prefix sum is a barrier-free warp scan via
// __shfl_up — no shared memory and none of the Hillis-Steele kernel's 10
// __syncthreads. Segments stay independent because the shuffle width is 32 and
// each warp covers one 32-aligned segment; this lets us launch wide CUDA blocks
// (many segments each) for full occupancy instead of one 32-thread block per
// segment (which caps at ~50% occupancy on the blocks-per-SM limit). ncu on the
// Hillis-Steele kernel flagged it as barrier-bound; this is the same class of fix
// as the TiledLorenzo per-row rewrite.
template<typename T>
__global__ void lorenzo_scan_1d_warp32_kernel(
    const T* __restrict__ in, T* __restrict__ out, size_t n)
{
    const size_t   gid  = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned lane = threadIdx.x & 31u;
    T v = (gid < n) ? in[gid] : static_cast<T>(0);
    #pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
        const T y = fz::backend::shflUp(v, static_cast<unsigned>(off), 32);
        if (lane >= static_cast<unsigned>(off)) v = static_cast<T>(v + y);
    }
    if (gid < n) out[gid] = v;
}

// `block_threads` is both the launch block size and the prediction-reset period:
// the delta/scan kernels reset per CUDA block, so launching with blockDim == n
// makes every n-element segment an independent chain (cuSZp uses n = 32). The
// default 256 reproduces the historical launch-block-local behavior.
template<typename T>
void launchLorenzoDeltaKernel1D(
    const T* d_input, T* d_output, size_t n, cudaStream_t stream,
    unsigned block_threads)
{
    if (n == 0) return;
    const int kBlock = static_cast<int>(block_threads);
    const int grid = static_cast<int>((n + kBlock - 1) / kBlock);
    lorenzo_delta_1d_kernel<T><<<grid, kBlock, 0, stream>>>(d_input, d_output, n);
    FZ_CUDA_CHECK(cudaGetLastError());
}

template<typename T>
void launchLorenzoPrefixSumKernel1D(
    const T* d_input, T* d_output, size_t n, cudaStream_t stream,
    unsigned block_threads)
{
    if (n == 0) return;
    // Fast path: 32-element reset segments == one warp → barrier-free warp scan,
    // launched with wide blocks (8 segments each) for occupancy.
    if (block_threads == 32u) {
        const int kBlock = 256;   // 8 warps = 8 independent 32-element segments
        const int grid = static_cast<int>((n + kBlock - 1) / kBlock);
        lorenzo_scan_1d_warp32_kernel<T><<<grid, kBlock, 0, stream>>>(d_input, d_output, n);
        FZ_CUDA_CHECK(cudaGetLastError());
        return;
    }
    const int kBlock = static_cast<int>(block_threads);
    const int grid = static_cast<int>((n + kBlock - 1) / kBlock);
    lorenzo_scan_1d_kernel<T>
        <<<grid, kBlock, kBlock * sizeof(T), stream>>>(d_input, d_output, n);
    FZ_CUDA_CHECK(cudaGetLastError());
}


// Barrier-based fallback for reset periods that are not a multiple of 32.
template<typename T>
__global__ void lorenzo_scan_any_kernel(
    const T* __restrict__ in,
    const T* __restrict__ means,
    T*       __restrict__ out,
    size_t n,
    int passes)
{
    extern __shared__ char smem[];
    T* s = reinterpret_cast<T*>(smem);

    const size_t gid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int    tid = static_cast<int>(threadIdx.x);

    s[tid] = (gid < n) ? in[gid] : static_cast<T>(0);
    __syncthreads();

    for (int pass = 0; pass < passes; ++pass) {
        for (int stride = 1; stride < static_cast<int>(blockDim.x); stride <<= 1) {
            T val = (tid >= stride) ? s[tid - stride] : static_cast<T>(0);
            __syncthreads();
            s[tid] += val;
            __syncthreads();
        }
    }

    T q = s[tid];
    if (means != nullptr) q = static_cast<T>(q + means[blockIdx.x]);
    if (gid < n) out[gid] = q;
}

// ─────────────────────────────────────────────────────────────────────────────
// Segmented inverse scan — one CTA per reset segment, Seq elements per thread
//
// Do NOT tie the CTA width to the reset period (`blockDim == block_size`, as the
// original did): that made the inverse barrier-bound and got *slower* at the
// highest-ratio block sizes. Here each thread owns `Seq` consecutive elements
// and the scan is serial-in-registers -> warp shuffle -> one pass over the warp
// totals, which is 2 barriers per scan pass regardless of segment length.
// Measurements: docs/codebase_notes.md CN-LRZ-1
//
// Handles both prediction orders and centering in one kernel: `passes` scans
// invert `passes` differences, and centering is undone by a uniform `+ mu`
// (for a single scan, seeding with mu and adding it at the end are equivalent;
// for two scans only the trailing add is correct, since a seeded mu would be
// summed i+1 times by the second scan).
// ─────────────────────────────────────────────────────────────────────────────
template<typename T, int Seq>
__global__ void lorenzo_segmented_scan_kernel(
    const T* __restrict__ in,
    const T* __restrict__ means,   // nullptr = centering off
    T*       __restrict__ out,
    size_t n,
    int passes)
{
    __shared__ T warp_totals[32];

    const unsigned seg_len = blockDim.x * Seq;
    const size_t   base    = static_cast<size_t>(blockIdx.x) * seg_len;
    const unsigned tid     = threadIdx.x;
    const unsigned lane    = tid & 31u;
    const unsigned warp    = tid >> 5;

    T v[Seq];
    #pragma unroll
    for (int i = 0; i < Seq; ++i) {
        const size_t g = base + static_cast<size_t>(tid) * Seq + i;
        v[i] = (g < n) ? in[g] : static_cast<T>(0);
    }

    for (int pass = 0; pass < passes; ++pass) {
        // 1. Serial inclusive scan over this thread's own elements.
        #pragma unroll
        for (int i = 1; i < Seq; ++i) v[i] = static_cast<T>(v[i] + v[i - 1]);

        // 2. Inclusive scan of thread totals across the warp.
        T tsum = v[Seq - 1];
        #pragma unroll
        for (int off = 1; off < 32; off <<= 1) {
            const T y = fz::backend::shflUp(tsum, static_cast<unsigned>(off), 32);
            if (lane >= static_cast<unsigned>(off)) tsum = static_cast<T>(tsum + y);
        }
        const T warp_excl = static_cast<T>(tsum - v[Seq - 1]);

        // 3. Exclusive scan across warps, through shared memory.
        if (lane == 31u) warp_totals[warp] = tsum;
        __syncthreads();
        T block_excl = static_cast<T>(0);
        for (unsigned w = 0; w < warp; ++w) block_excl = static_cast<T>(block_excl + warp_totals[w]);

        const T add = static_cast<T>(warp_excl + block_excl);
        #pragma unroll
        for (int i = 0; i < Seq; ++i) v[i] = static_cast<T>(v[i] + add);

        // Guard warp_totals against the next pass overwriting it mid-read.
        __syncthreads();
    }

    const T mu = (means != nullptr) ? means[blockIdx.x] : static_cast<T>(0);
    #pragma unroll
    for (int i = 0; i < Seq; ++i) {
        const size_t g = base + static_cast<size_t>(tid) * Seq + i;
        if (g < n) out[g] = static_cast<T>(v[i] + mu);
    }
}

// Pick (threads, Seq) for a reset period: keep the CTA at or below 256 threads
// where possible so occupancy does not track segment length. Requires
// `block_size % 32 == 0`; callers fall back to the generic path otherwise.
inline void segmentedScanShape(unsigned block_size, unsigned& threads, int& seq) {
    if (block_size <= 256u)                        { threads = block_size;      seq = 1; }
    else if ((block_size / 4u) % 32u == 0u)        { threads = block_size / 4u; seq = 4; }
    else if ((block_size / 2u) % 32u == 0u)        { threads = block_size / 2u; seq = 2; }
    else                                           { threads = block_size;      seq = 1; }
}

// Unified block-mode inverse: any order, with or without centering.
template<typename T>
void launchLorenzoSegmentedScan(
    const T* d_input, const T* d_means, T* d_output, size_t n, cudaStream_t stream,
    unsigned block_threads, int passes)
{
    if (n == 0) return;
    const int grid = static_cast<int>((n + block_threads - 1) / block_threads);

    if (block_threads % 32u != 0u) {
        // Non-warp-multiple reset period (e.g. 100): the warp-shuffle scan does
        // not apply, so use the barrier-based scan, which handles any width.
        lorenzo_scan_any_kernel<T><<<grid, block_threads,
                                     block_threads * sizeof(T), stream>>>(
            d_input, d_means, d_output, n, passes);
        FZ_CUDA_CHECK(cudaGetLastError());
        return;
    }

    unsigned threads; int seq;
    segmentedScanShape(block_threads, threads, seq);
    switch (seq) {
        case 4: lorenzo_segmented_scan_kernel<T, 4><<<grid, threads, 0, stream>>>(
                    d_input, d_means, d_output, n, passes); break;
        case 2: lorenzo_segmented_scan_kernel<T, 2><<<grid, threads, 0, stream>>>(
                    d_input, d_means, d_output, n, passes); break;
        default: lorenzo_segmented_scan_kernel<T, 1><<<grid, threads, 0, stream>>>(
                    d_input, d_means, d_output, n, passes); break;
    }
    FZ_CUDA_CHECK(cudaGetLastError());
}

// ─────────────────────────────────────────────────────────────────────────────
// 1-D block mode with per-block mean centering (FSZ adaptive centering)
//
// Only the *first* residual of each block changes. For k-th order differences
// delta^k(q - mu) == delta^k(q) for every element that has a predecessor, so
// subtracting a per-block constant can only affect the chain seed — the one
// element that would otherwise be emitted as a raw value. One CUDA block owns
// one reset segment, so the mean is a block-wide reduction in shared memory.
// ─────────────────────────────────────────────────────────────────────────────

// Forward: out[0] = in[0] - mu, out[i] = in[i] - in[i-1] (i > 0), means[b] = mu.
template<typename T>
__global__ void lorenzo_delta_1d_centered_kernel(
    const T* __restrict__ in,
    T*       __restrict__ out,
    T*       __restrict__ means,
    size_t n)
{
    extern __shared__ char smem[];
    // Accumulate in 64 bits: up to 1024 elements of T summed without overflow
    // for every T narrower than int64_t.
    long long* s = reinterpret_cast<long long*>(smem);

    const size_t   base = static_cast<size_t>(blockIdx.x) * blockDim.x;
    const size_t   gid  = base + threadIdx.x;
    const unsigned tid  = threadIdx.x;
    const bool     live = (gid < n);

    const T v = live ? in[gid] : static_cast<T>(0);
    s[tid] = live ? static_cast<long long>(v) : 0LL;
    __syncthreads();

    // Reduction over a possibly non-power-of-two blockDim: start the stride at
    // the next power of two and guard the upper partner index.
    unsigned p = 1u;
    while (p < blockDim.x) p <<= 1;
    for (unsigned stride = p >> 1; stride > 0u; stride >>= 1) {
        if (tid < stride && tid + stride < blockDim.x) s[tid] += s[tid + stride];
        __syncthreads();
    }

    // Trailing partial block: only the live elements are in the sum, so divide
    // by the live count, not by blockDim.
    const long long count = static_cast<long long>(min(static_cast<size_t>(blockDim.x), n - base));
    const long long tot   = s[0];
    // Round half away from zero, matching round() on the equivalent float mean.
    const T mu = static_cast<T>((tot >= 0) ? (tot + count / 2) / count
                                           : (tot - count / 2) / count);
    if (tid == 0) means[blockIdx.x] = mu;

    if (!live) return;
    out[gid] = (tid > 0u) ? static_cast<T>(v - in[gid - 1])
                          : static_cast<T>(v - mu);
}

template<typename T>
void launchLorenzoDeltaCentered1D(
    const T* d_input, T* d_output, T* d_means, size_t n, cudaStream_t stream,
    unsigned block_threads)
{
    if (n == 0) return;
    const int kBlock = static_cast<int>(block_threads);
    const int grid   = static_cast<int>((n + kBlock - 1) / kBlock);
    lorenzo_delta_1d_centered_kernel<T>
        <<<grid, kBlock, kBlock * sizeof(long long), stream>>>(d_input, d_output, d_means, n);
    FZ_CUDA_CHECK(cudaGetLastError());
}

// ─────────────────────────────────────────────────────────────────────────────
// 1-D block mode, second-order (LZ2)
//
// Block-local LZ2 is the LZ1 delta applied twice under the same zero-padding
// convention: with d = delta(q) (d_0 = q_0) and e = delta(d),
//   e_0 = q_0,  e_1 = q_1 - 2*q_0,  e_i = q_i - 2*q_{i-1} + q_{i-2} (i >= 2)
// which is FSZ's LZ2 with the two missing predecessors read as zero. Doing both
// passes in shared memory keeps it to a single trip through global memory and
// needs no scratch buffer, unlike chaining two delta kernels.
//
// LZ2 annihilates a linear ramp, so it wins where the field has a smooth
// gradient and LZ1 leaves a constant non-zero residual. It costs one extra
// element of raw seed per block (e_0 and e_1 both lack full predecessors),
// which is why it pairs with long blocks and with centering.
//
// Centering here subtracts mu from the *whole* segment rather than just the
// seed. Since mu cancels out of every second difference from i >= 2 on, only
// e_0 and e_1 change, and the inverse is a uniform "+ mu" after the two scans.
// ─────────────────────────────────────────────────────────────────────────────

template<typename T>
__global__ void lorenzo2_delta_1d_kernel(
    const T* __restrict__ in,
    T*       __restrict__ out,
    T*       __restrict__ means,   // nullptr = centering off
    size_t n)
{
    // Two shared regions: the data staging area (T) and a 64-bit accumulator
    // for the mean reduction, which must not overflow on a 1024-element sum.
    extern __shared__ char smem[];
    T*         s   = reinterpret_cast<T*>(smem);
    long long* red = reinterpret_cast<long long*>(smem + blockDim.x * sizeof(T));

    const size_t   base = static_cast<size_t>(blockIdx.x) * blockDim.x;
    const size_t   gid  = base + threadIdx.x;
    const unsigned tid  = threadIdx.x;
    const bool     live = (gid < n);

    T v = live ? in[gid] : static_cast<T>(0);

    if (means != nullptr) {
        red[tid] = live ? static_cast<long long>(v) : 0LL;
        __syncthreads();
        unsigned p = 1u;
        while (p < blockDim.x) p <<= 1;
        for (unsigned stride = p >> 1; stride > 0u; stride >>= 1) {
            if (tid < stride && tid + stride < blockDim.x) red[tid] += red[tid + stride];
            __syncthreads();
        }
        const long long count =
            static_cast<long long>(min(static_cast<size_t>(blockDim.x), n - base));
        const long long tot = red[0];
        const T mu = static_cast<T>((tot >= 0) ? (tot + count / 2) / count
                                               : (tot - count / 2) / count);
        if (tid == 0) means[blockIdx.x] = mu;
        v = static_cast<T>(v - mu);
        __syncthreads();
    }

    // Pass 1: first difference.
    s[tid] = v;
    __syncthreads();
    const T d = static_cast<T>(s[tid] - ((tid > 0u) ? s[tid - 1] : static_cast<T>(0)));
    __syncthreads();

    // Pass 2: difference of the differences.
    s[tid] = d;
    __syncthreads();
    const T e = static_cast<T>(s[tid] - ((tid > 0u) ? s[tid - 1] : static_cast<T>(0)));

    if (live) out[gid] = e;
}

template<typename T>
__global__ void lorenzo2_scan_1d_kernel(
    const T* __restrict__ in,
    const T* __restrict__ means,   // nullptr = centering off
    T*       __restrict__ out,
    size_t n)
{
    extern __shared__ char smem[];
    T* s = reinterpret_cast<T*>(smem);

    const size_t gid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int    tid = static_cast<int>(threadIdx.x);

    s[tid] = (gid < n) ? in[gid] : static_cast<T>(0);
    __syncthreads();

    // Two inclusive scans invert the two differences, in the same order.
    for (int pass = 0; pass < 2; ++pass) {
        for (int stride = 1; stride < static_cast<int>(blockDim.x); stride <<= 1) {
            T val = (tid >= stride) ? s[tid - stride] : static_cast<T>(0);
            __syncthreads();
            s[tid] += val;
            __syncthreads();
        }
    }

    T q = s[tid];
    // Centering subtracted mu from every element, so restore it on every element.
    if (means != nullptr) q = static_cast<T>(q + means[blockIdx.x]);
    if (gid < n) out[gid] = q;
}

template<typename T>
void launchLorenzo2Delta1D(
    const T* d_input, T* d_output, T* d_means, size_t n, cudaStream_t stream,
    unsigned block_threads)
{
    if (n == 0) return;
    const int kBlock = static_cast<int>(block_threads);
    const int grid   = static_cast<int>((n + kBlock - 1) / kBlock);
    const size_t shmem = kBlock * (sizeof(T) + sizeof(long long));
    lorenzo2_delta_1d_kernel<T>
        <<<grid, kBlock, shmem, stream>>>(d_input, d_output, d_means, n);
    FZ_CUDA_CHECK(cudaGetLastError());
}

// ─────────────────────────────────────────────────────────────────────────────
// 2-D kernels
// ─────────────────────────────────────────────────────────────────────────────

// Forward 2-D: d[x,y] - d[x-1,y] - d[x,y-1] + d[x-1,y-1]
template<typename T>
__global__ void lorenzo_delta_2d_kernel(
    const T* __restrict__ in,
    T*       __restrict__ out,
    size_t nx, size_t ny)
{
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;  // fast dim
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) return;

    const size_t idx = y * nx + x;
    T v   = in[idx];
    T vx  = (x > 0) ? in[idx - 1]      : static_cast<T>(0);
    T vy  = (y > 0) ? in[idx - nx]     : static_cast<T>(0);
    T vxy = (x > 0 && y > 0) ? in[idx - nx - 1] : static_cast<T>(0);
    out[idx] = v - vx - vy + vxy;
}

// ─────────────────────────────────────────────────────────────────────────────
// Inverse (prefix-sum) scan, 2-D and 3-D
//
// The multi-dimensional inverse is a sequence of independent 1-D inclusive
// scans, one axis at a time: 2-D is a row pass then a column pass; 3-D is an
// x, then y, then z pass. Every one of those is the same operation — "scan
// `len` elements spaced `elem_stride` apart, once per line" — so they all share
// `lorenzo_scan_line_kernel` below and differ only in launch parameters.
//
// One block owns one whole line and walks it in tiles of `blockDim.x`, carrying
// a running total between tiles. That keeps the line length independent of the
// maximum block dimension and of shared-memory capacity: shared memory is sized
// by the block, not by the extent. The earlier per-axis kernels launched
// `blockDim = extent` and so failed with `invalid configuration argument` for
// any dimension above 1024 (e.g. a 3600x1800 field).
//
// `in` and `out` may alias: each thread reads and writes only its own index,
// and tiles touch disjoint ranges.
//
// The line's base offset is `(line / inner) * outer_mult + (line % inner) * inner_mult`,
// which expresses all five axis passes — see the launchers for the mapping.
// ─────────────────────────────────────────────────────────────────────────────

template<typename T>
__global__ void lorenzo_scan_line_kernel(
    const T* __restrict__ in,
    T*       __restrict__ out,
    size_t len,          // elements along the scanned axis
    size_t elem_stride,  // distance between consecutive elements of a line
    size_t n_lines,      // number of independent lines (one per block)
    size_t inner,        // see base-offset formula above
    size_t outer_mult,
    size_t inner_mult)
{
    const size_t line = blockIdx.x;
    if (line >= n_lines) return;

    const size_t base = (line / inner) * outer_mult + (line % inner) * inner_mult;

    extern __shared__ char smem[];
    T* s = reinterpret_cast<T*>(smem);

    const int tid = static_cast<int>(threadIdx.x);
    T carry = static_cast<T>(0);

    for (size_t tile = 0; tile < len; tile += blockDim.x) {
        const size_t i = tile + static_cast<size_t>(tid);

        s[tid] = (i < len) ? in[base + i * elem_stride] : static_cast<T>(0);
        __syncthreads();

        for (int stride = 1; stride < static_cast<int>(blockDim.x); stride <<= 1) {
            T val = (tid >= stride) ? s[tid - stride] : static_cast<T>(0);
            __syncthreads();
            s[tid] += val;
            __syncthreads();
        }

        if (i < len) out[base + i * elem_stride] = s[tid] + carry;

        // Tile total: the padded tail is zero, so the last slot is the true sum.
        // Read it before the next iteration overwrites shared memory.
        const T tile_total = s[blockDim.x - 1];
        __syncthreads();
        carry += tile_total;
    }
}

/// Threads per block for the line scan. Shared memory is `kScanBlock * sizeof(T)`,
/// independent of the field dimensions.
static constexpr int kScanBlock = 256;

template<typename T>
void launchLorenzoDeltaKernel2D(
    const T* d_input, T* d_output, size_t nx, size_t ny, cudaStream_t stream)
{
    if (nx == 0 || ny == 0) return;
    dim3 block(16, 16);
    dim3 grid(static_cast<unsigned>((nx + 15) / 16),
              static_cast<unsigned>((ny + 15) / 16));
    lorenzo_delta_2d_kernel<T><<<grid, block, 0, stream>>>(d_input, d_output, nx, ny);
    FZ_CUDA_CHECK(cudaGetLastError());
}

template<typename T>
void launchLorenzoPrefixSumKernel2D(
    const T* d_input, T* d_output, size_t nx, size_t ny, cudaStream_t stream)
{
    if (nx == 0 || ny == 0) return;
    const size_t smem = kScanBlock * sizeof(T);

    // Row scan: line y covers [y*nx, y*nx + nx), contiguous.
    //   base = y * nx  →  inner = n_lines (so line/inner == 0), inner_mult = nx
    lorenzo_scan_line_kernel<T>
        <<<static_cast<unsigned>(ny), kScanBlock, smem, stream>>>(
            d_input, d_output, /*len=*/nx, /*elem_stride=*/1,
            /*n_lines=*/ny, /*inner=*/ny, /*outer_mult=*/0, /*inner_mult=*/nx);
    FZ_CUDA_CHECK(cudaGetLastError());

    // Column scan on the row-scan output: line x covers x, x+nx, x+2nx, ...
    //   base = x  →  inner = n_lines, inner_mult = 1
    lorenzo_scan_line_kernel<T>
        <<<static_cast<unsigned>(nx), kScanBlock, smem, stream>>>(
            d_output, d_output, /*len=*/ny, /*elem_stride=*/nx,
            /*n_lines=*/nx, /*inner=*/nx, /*outer_mult=*/0, /*inner_mult=*/1);
    FZ_CUDA_CHECK(cudaGetLastError());
}

// ─────────────────────────────────────────────────────────────────────────────
// 3-D kernels
// ─────────────────────────────────────────────────────────────────────────────

// Forward 3-D inclusion-exclusion delta (8-neighbor formula)
template<typename T>
__global__ void lorenzo_delta_3d_kernel(
    const T* __restrict__ in,
    T*       __restrict__ out,
    size_t nx, size_t ny, size_t nz)
{
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) return;

    const size_t idx = z * ny * nx + y * nx + x;
    auto get = [&](ptrdiff_t dx, ptrdiff_t dy, ptrdiff_t dz) -> T {
        ptrdiff_t xx = static_cast<ptrdiff_t>(x) + dx;
        ptrdiff_t yy = static_cast<ptrdiff_t>(y) + dy;
        ptrdiff_t zz = static_cast<ptrdiff_t>(z) + dz;
        if (xx < 0 || yy < 0 || zz < 0) return static_cast<T>(0);
        return in[zz * static_cast<ptrdiff_t>(ny * nx)
                + yy * static_cast<ptrdiff_t>(nx) + xx];
    };

    out[idx] =  get(0,0,0) - get(-1,0,0) - get(0,-1,0) - get(0,0,-1)
              + get(-1,-1,0) + get(-1,0,-1) + get(0,-1,-1)
              - get(-1,-1,-1);
}

template<typename T>
void launchLorenzoDeltaKernel3D(
    const T* d_input, T* d_output, size_t nx, size_t ny, size_t nz,
    cudaStream_t stream)
{
    if (nx == 0 || ny == 0 || nz == 0) return;
    dim3 block(8, 8, 8);
    dim3 grid(static_cast<unsigned>((nx + 7) / 8),
              static_cast<unsigned>((ny + 7) / 8),
              static_cast<unsigned>((nz + 7) / 8));
    lorenzo_delta_3d_kernel<T><<<grid, block, 0, stream>>>(d_input, d_output, nx, ny, nz);
    FZ_CUDA_CHECK(cudaGetLastError());
}

template<typename T>
void launchLorenzoPrefixSumKernel3D(
    const T* d_input, T* d_output, size_t nx, size_t ny, size_t nz,
    cudaStream_t stream)
{
    if (nx == 0 || ny == 0 || nz == 0) return;
    const size_t smem = kScanBlock * sizeof(T);

    // Inverse 3-D is three sequential 1-D prefix-sum passes (x → y → z).

    // X-pass: one line per (y, z), contiguous. For line i = y + z*ny,
    //   base = z*ny*nx + y*nx = i*nx  →  inner = n_lines, inner_mult = nx
    lorenzo_scan_line_kernel<T>
        <<<static_cast<unsigned>(ny * nz), kScanBlock, smem, stream>>>(
            d_input, d_output, /*len=*/nx, /*elem_stride=*/1,
            /*n_lines=*/ny * nz, /*inner=*/ny * nz,
            /*outer_mult=*/0, /*inner_mult=*/nx);
    FZ_CUDA_CHECK(cudaGetLastError());

    // Y-pass on the X output: one line per (x, z). For line i = x + z*nx,
    //   base = z*ny*nx + x  →  inner = nx, outer_mult = ny*nx, inner_mult = 1
    lorenzo_scan_line_kernel<T>
        <<<static_cast<unsigned>(nx * nz), kScanBlock, smem, stream>>>(
            d_output, d_output, /*len=*/ny, /*elem_stride=*/nx,
            /*n_lines=*/nx * nz, /*inner=*/nx,
            /*outer_mult=*/ny * nx, /*inner_mult=*/1);
    FZ_CUDA_CHECK(cudaGetLastError());

    // Z-pass on the Y output: one line per (x, y). For line i = x + y*nx,
    //   base = y*nx + x = i  →  inner = n_lines, inner_mult = 1
    lorenzo_scan_line_kernel<T>
        <<<static_cast<unsigned>(nx * ny), kScanBlock, smem, stream>>>(
            d_output, d_output, /*len=*/nz, /*elem_stride=*/ny * nx,
            /*n_lines=*/nx * ny, /*inner=*/nx * ny,
            /*outer_mult=*/0, /*inner_mult=*/1);
    FZ_CUDA_CHECK(cudaGetLastError());
}

// ─────────────────────────────────────────────────────────────────────────────
// LorenzoStage::execute
// ─────────────────────────────────────────────────────────────────────────────

template<typename T>
void LorenzoStage<T>::execute(
    cudaStream_t stream,
    MemoryPool* /*pool*/,
    const std::vector<void*>& inputs,
    const std::vector<void*>& outputs,
    const std::vector<size_t>& sizes)
{
    if (inputs.empty() || outputs.empty() || sizes.empty())
        throw std::runtime_error("LorenzoStage: inputs, outputs, and sizes must be non-empty");

    const size_t byte_size = sizes[0];
    if (byte_size == 0) {
        actual_output_size_ = 0;
        return;
    }

    const size_t n = byte_size / sizeof(T);
    const T* in    = static_cast<const T*>(inputs[0]);
    T*       out   = static_cast<T*>(outputs[0]);

    // Resolve effective dims: if dim_x is 0, treat as flat 1-D of n elements.
    size_t nx = (dims_[0] > 0) ? dims_[0] : n;
    size_t ny = dims_[1];
    size_t nz = dims_[2];

    // Explicit block mode (cuSZp-style): force the 1-D path over the flattened
    // array, resetting the prediction chain every block_size_ elements.
    if (block_size_ > 0) {
        const unsigned bt = block_size_;
        const size_t nblocks = (n + bt - 1) / bt;

        // The means port is optional plumbing shared by both orders; resolve it
        // once here so the order dispatch below only picks a kernel.
        T*       means_out = nullptr;
        const T* means_in  = nullptr;
        if (centering_) {
            if (!is_inverse_) {
                if (outputs.size() < 2 || outputs[1] == nullptr)
                    throw std::runtime_error(
                        "LorenzoStage: centering enabled but the 'means' output port "
                        "is not connected");
                means_out = static_cast<T*>(outputs[1]);
                actual_means_size_ = nblocks * sizeof(T);
            } else {
                if (inputs.size() < 2 || inputs[1] == nullptr)
                    throw std::runtime_error(
                        "LorenzoStage: centering enabled but the 'means' input port "
                        "is not connected");
                means_in = static_cast<const T*>(inputs[1]);
            }
        }

        if (!is_inverse_) {
            if      (order_ == 2) launchLorenzo2Delta1D<T>(in, out, means_out, n, stream, bt);
            else if (centering_)  launchLorenzoDeltaCentered1D<T>(in, out, means_out, n, stream, bt);
            else                  launchLorenzoDeltaKernel1D<T>(in, out, n, stream, bt);
        } else if (order_ == 1 && !centering_ && bt == 32u) {
            // 32-element segments are exactly one warp: the barrier-free warp
            // scan with wide CTAs still beats the segmented kernel here.
            launchLorenzoPrefixSumKernel1D<T>(in, out, n, stream, bt);
        } else {
            launchLorenzoSegmentedScan<T>(in, means_in, out, n, stream, bt,
                                          (order_ == 2) ? 2 : 1);
        }
        actual_output_size_ = byte_size;
        return;
    }

    if (centering_)
        throw std::runtime_error(
            "LorenzoStage: setCentering(true) requires block mode — call "
            "setBlockSize(n) with n > 0 (there is no per-block mean without blocks)");
    if (order_ == 2)
        throw std::runtime_error(
            "LorenzoStage: setOrder(2) requires block mode — call setBlockSize(n) "
            "with n > 0 (the N-D path has no second-order form)");

    int eff_ndim = ndim();

    if (!is_inverse_) {
        if      (eff_ndim == 3) launchLorenzoDeltaKernel3D<T>(in, out, nx, ny, nz, stream);
        else if (eff_ndim == 2) launchLorenzoDeltaKernel2D<T>(in, out, nx, ny, stream);
        else                    launchLorenzoDeltaKernel1D<T>(in, out, n, stream);
    } else {
        if      (eff_ndim == 3) launchLorenzoPrefixSumKernel3D<T>(in, out, nx, ny, nz, stream);
        else if (eff_ndim == 2) launchLorenzoPrefixSumKernel2D<T>(in, out, nx, ny, stream);
        else                    launchLorenzoPrefixSumKernel1D<T>(in, out, n, stream);
    }

    actual_output_size_ = byte_size;
}

// ─────────────────────────────────────────────────────────────────────────────
// Explicit instantiations
// ─────────────────────────────────────────────────────────────────────────────

template class LorenzoStage<int8_t>;
template class LorenzoStage<int16_t>;
template class LorenzoStage<int32_t>;
template class LorenzoStage<int64_t>;

template void launchLorenzoDeltaKernel1D<int8_t> (const int8_t*,  int8_t*,  size_t, cudaStream_t, unsigned);
template void launchLorenzoDeltaKernel1D<int16_t>(const int16_t*, int16_t*, size_t, cudaStream_t, unsigned);
template void launchLorenzoDeltaKernel1D<int32_t>(const int32_t*, int32_t*, size_t, cudaStream_t, unsigned);
template void launchLorenzoDeltaKernel1D<int64_t>(const int64_t*, int64_t*, size_t, cudaStream_t, unsigned);

template void launchLorenzoPrefixSumKernel1D<int8_t> (const int8_t*,  int8_t*,  size_t, cudaStream_t, unsigned);
template void launchLorenzoPrefixSumKernel1D<int16_t>(const int16_t*, int16_t*, size_t, cudaStream_t, unsigned);
template void launchLorenzoPrefixSumKernel1D<int32_t>(const int32_t*, int32_t*, size_t, cudaStream_t, unsigned);
template void launchLorenzoPrefixSumKernel1D<int64_t>(const int64_t*, int64_t*, size_t, cudaStream_t, unsigned);

template void launchLorenzoDeltaCentered1D<int8_t> (const int8_t*,  int8_t*,  int8_t*,  size_t, cudaStream_t, unsigned);
template void launchLorenzoDeltaCentered1D<int16_t>(const int16_t*, int16_t*, int16_t*, size_t, cudaStream_t, unsigned);
template void launchLorenzoDeltaCentered1D<int32_t>(const int32_t*, int32_t*, int32_t*, size_t, cudaStream_t, unsigned);
template void launchLorenzoDeltaCentered1D<int64_t>(const int64_t*, int64_t*, int64_t*, size_t, cudaStream_t, unsigned);


template void launchLorenzoSegmentedScan<int8_t> (const int8_t*,  const int8_t*,  int8_t*,  size_t, cudaStream_t, unsigned, int);
template void launchLorenzoSegmentedScan<int16_t>(const int16_t*, const int16_t*, int16_t*, size_t, cudaStream_t, unsigned, int);
template void launchLorenzoSegmentedScan<int32_t>(const int32_t*, const int32_t*, int32_t*, size_t, cudaStream_t, unsigned, int);
template void launchLorenzoSegmentedScan<int64_t>(const int64_t*, const int64_t*, int64_t*, size_t, cudaStream_t, unsigned, int);

template void launchLorenzo2Delta1D<int8_t> (const int8_t*,  int8_t*,  int8_t*,  size_t, cudaStream_t, unsigned);
template void launchLorenzo2Delta1D<int16_t>(const int16_t*, int16_t*, int16_t*, size_t, cudaStream_t, unsigned);
template void launchLorenzo2Delta1D<int32_t>(const int32_t*, int32_t*, int32_t*, size_t, cudaStream_t, unsigned);
template void launchLorenzo2Delta1D<int64_t>(const int64_t*, int64_t*, int64_t*, size_t, cudaStream_t, unsigned);


template void launchLorenzoDeltaKernel2D<int8_t> (const int8_t*,  int8_t*,  size_t, size_t, cudaStream_t);
template void launchLorenzoDeltaKernel2D<int16_t>(const int16_t*, int16_t*, size_t, size_t, cudaStream_t);
template void launchLorenzoDeltaKernel2D<int32_t>(const int32_t*, int32_t*, size_t, size_t, cudaStream_t);
template void launchLorenzoDeltaKernel2D<int64_t>(const int64_t*, int64_t*, size_t, size_t, cudaStream_t);

template void launchLorenzoPrefixSumKernel2D<int8_t> (const int8_t*,  int8_t*,  size_t, size_t, cudaStream_t);
template void launchLorenzoPrefixSumKernel2D<int16_t>(const int16_t*, int16_t*, size_t, size_t, cudaStream_t);
template void launchLorenzoPrefixSumKernel2D<int32_t>(const int32_t*, int32_t*, size_t, size_t, cudaStream_t);
template void launchLorenzoPrefixSumKernel2D<int64_t>(const int64_t*, int64_t*, size_t, size_t, cudaStream_t);

template void launchLorenzoDeltaKernel3D<int8_t> (const int8_t*,  int8_t*,  size_t, size_t, size_t, cudaStream_t);
template void launchLorenzoDeltaKernel3D<int16_t>(const int16_t*, int16_t*, size_t, size_t, size_t, cudaStream_t);
template void launchLorenzoDeltaKernel3D<int32_t>(const int32_t*, int32_t*, size_t, size_t, size_t, cudaStream_t);
template void launchLorenzoDeltaKernel3D<int64_t>(const int64_t*, int64_t*, size_t, size_t, size_t, cudaStream_t);

template void launchLorenzoPrefixSumKernel3D<int8_t> (const int8_t*,  int8_t*,  size_t, size_t, size_t, cudaStream_t);
template void launchLorenzoPrefixSumKernel3D<int16_t>(const int16_t*, int16_t*, size_t, size_t, size_t, cudaStream_t);
template void launchLorenzoPrefixSumKernel3D<int32_t>(const int32_t*, int32_t*, size_t, size_t, size_t, cudaStream_t);
template void launchLorenzoPrefixSumKernel3D<int64_t>(const int64_t*, int64_t*, size_t, size_t, size_t, cudaStream_t);

} // namespace fz
