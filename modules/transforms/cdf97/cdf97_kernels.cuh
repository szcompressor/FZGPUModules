#pragma once

/**
 * @file cdf97_kernels.cuh
 * @brief Separable multi-level CDF 9/7 DWT built on the 1-D lifting primitive.
 *
 * STATUS: 1-D/2-D/3-D (both dyadic and wavelet-packet) implemented and validated
 * bit-exact vs SPERR (double). Long-line (> shared memory) fallback is TODO. See
 * memory/cdf97_dwt_design.md.
 *
 * Follows SPERR's separable dyadic (Mallat) decomposition: transform each axis,
 * recurse on the LL(L) corner. GPU line-in-shared-memory scheme follows Matela
 * (2009) / van der Laan et al.
 *
 * ### Persistent-kernel path (grid-sync level fusion)
 *
 * A first-pass throughput measurement (`memory/cdf97_dwt_design.md`, 2026-08-13)
 * found the naive per-level, per-axis kernel launch scheme (`*_multilaunch`
 * below) is launch-count-bound, not bandwidth-bound: SPERR's level cap tops out
 * at 6, so every transform issues a fixed ~6-18 sequential kernel launches
 * regardless of field size, and runtime was nearly flat across a 4x data-size
 * range — the signature of a fixed per-launch floor dominating.
 *
 * `cdf97_persistent_kernel` fixes this by keeping ALL levels/axes of ONE
 * transform call inside a single kernel launch, using a cooperative-groups
 * grid-wide barrier (`grid.sync()`) between passes instead of ending the kernel
 * and having the host re-launch. This requires every block used by any pass to
 * be co-resident simultaneously for the whole kernel's lifetime — a cooperative
 * launch (`cudaLaunchCooperativeKernel`), which the driver only accepts if the
 * needed grid actually fits in one wave on the current GPU (checked via
 * `cudaOccupancyMaxActiveBlocksPerMultiprocessor`).
 *
 * The pass sequence built by `cdf97_build_passes_*` is **identical, op-for-op,
 * to the sequence the `*_multilaunch` functions issue** — same order, same
 * arithmetic — so the persistent path is not a numerical change, only a launch-
 * count change. It stays bit-exact with SPERR wherever the multilaunch path is.
 *
 * Every top-level driver (`dwt1d`/`dwt2d`/`dwt3d_dyadic`/`dwt3d_packet`) tries
 * the persistent path first and **transparently falls back** to
 * `*_multilaunch` whenever it can't be used: cooperative launch unsupported on
 * this GPU/driver, or the field is large enough that its level-0 line count
 * doesn't fit in one cooperative wave (fewer SMs => smaller max wave, so this
 * naturally self-adjusts per-GPU rather than gating on a hardcoded compute
 * capability). Confirmed working via `cudaLaunchCooperativeKernel` on A100 —
 * not an H100-only feature; it degrades gracefully on anything smaller.
 */

#include "cdf97_lifting.cuh"
#include <cooperative_groups.h>
#include <cstddef>

namespace fz {
namespace cdf97 {

/**
 * One axis, one level. Each block transforms exactly one line of length
 * `line_len`; block threads cooperate over the line via the shared buffer.
 *
 * Lines are enumerated over a 2-D grid so the same kernel serves every axis of
 * both 2-D and 3-D. A line is addressed by (a = blockIdx.x, b = blockIdx.y):
 *
 *     base = b * stride_b + a * stride_a            (offset of the line's sample 0)
 *     sample k of the line is at  base + k * elem_stride
 *
 * e.g. for a row-major (nx,ny,nz) volume, plane_sz = nx*ny:
 *   X over all planes : line_len=cx, stride_a=nx,  n_a=cy, stride_b=plane_sz, n_b=cz, elem_stride=1
 *   Y over all planes : line_len=cy, stride_a=1,   n_a=cx, stride_b=plane_sz, n_b=cz, elem_stride=nx
 *   Z columns         : line_len=cz, stride_a=1,   n_a=cx, stride_b=nx,       n_b=cy, elem_stride=plane_sz
 *   pure 2-D          : n_b=1, stride_b=0
 *
 * Forward: gather (de-interleave) on load, analysis, identity write (result is
 * already in [low|high] subband order). Inverse: load subband order, synthesis,
 * scatter (re-interleave) on store. Requires `line_len*sizeof(T)` shared bytes.
 */
template <typename T>
__global__ void cdf97_axis_kernel(T* data,
                                  int line_len,
                                  long stride_a,
                                  long stride_b,
                                  long elem_stride,
                                  long n_a,
                                  bool inverse)
{
    extern __shared__ char smem_raw[];
    T* sm = reinterpret_cast<T*>(smem_raw);

    const long a = blockIdx.x;
    if (a >= n_a) return;
    const long base = (long)blockIdx.y * stride_b + a * stride_a;

    const int even_len = line_len - line_len / 2;
    const int odd_len  = line_len / 2;
    const int t = threadIdx.x, nt = blockDim.x;

    if (!inverse) {
        for (int e = t; e < even_len; e += nt)
            sm[e] = data[base + (long)(2 * e) * elem_stride];
        for (int o = t; o < odd_len; o += nt)
            sm[even_len + o] = data[base + (long)(2 * o + 1) * elem_stride];
        __syncthreads();

        analysis_line<T>(sm, line_len, t, nt);

        for (int k = t; k < line_len; k += nt)
            data[base + (long)k * elem_stride] = sm[k];
    } else {
        for (int k = t; k < line_len; k += nt)
            sm[k] = data[base + (long)k * elem_stride];
        __syncthreads();

        synthesis_line<T>(sm, line_len, t, nt);

        for (int e = t; e < even_len; e += nt)
            data[base + (long)(2 * e) * elem_stride] = sm[e];
        for (int o = t; o < odd_len; o += nt)
            data[base + (long)(2 * o + 1) * elem_stride] = sm[even_len + o];
    }
}

/**
 * Same transform as `cdf97_axis_kernel`, but for passes whose natural per-line
 * access is *strided* (`elem_stride != 1` — the Y-pass of a 2-D field, the
 * Y/Z-passes of a 3-D volume). Measured on H100: such passes ran 2.2-2.9x
 * slower than an `elem_stride==1` pass of the same size, purely from memory
 * coalescing — each thread in a warp was landing on a separate cache line.
 *
 * Every strided pass in this file has `stride_a == 1` (adjacent line indices
 * are adjacent in memory — e.g. the 2-D Y-pass's columns are `nx`-major, so
 * column `a` and `a+1` differ by one `double`). This kernel exploits exactly
 * that: instead of one block per line, one block owns `TW` *adjacent* lines
 * (`TW` chosen so all `TW` lines' shared-memory copies fit the budget) and
 * loads/stores them with row-major sweeps — for a fixed position `k` along the
 * lines, the `TW` lines' samples at `k` are `TW` contiguous elements, so a
 * `TW`-wide slice of the load is one coalesced transaction instead of `TW`
 * separate ones. The lifting itself is byte-for-byte identical to
 * `cdf97_axis_kernel` — same `analysis_line`/`synthesis_line` calls, called
 * once per line in the tile — so this is a pure memory-access-pattern change,
 * not a numerical one; it stays bit-exact with SPERR wherever the untiled
 * kernel is.
 *
 * Block shape is 2-D: `(TW, RPS)` where `RPS = blockDim.y` is how many "row
 * lanes" cooperate on the coalesced sweep. The lifting phase reinterprets the
 * same `TW*RPS` threads as one flat cooperative index (`analysis_line` doesn't
 * care how a block is shaped, only that `t`/`nt` cover `[0,nt)` once).
 *
 * `n_a` is not required to be a multiple of `TW`: the last tile is boundary-
 * clipped (`tw_eff <= TW`), and — since `tw_eff` is the same for every thread
 * in a block (it depends only on `blockIdx.x`, not on `threadIdx`) — every
 * thread in the block still takes the same number of trips through the
 * per-line loop below, so `__syncthreads()` inside `analysis_line`/
 * `synthesis_line` stays block-uniform even though it differs *between*
 * blocks (`__syncthreads()` is block-scoped, so that's fine).
 */
template <typename T>
__global__ void cdf97_axis_kernel_tiled(T* data,
                                        int line_len,
                                        long stride_b,
                                        long elem_stride,
                                        long n_a,
                                        int  TW,
                                        bool inverse)
{
    extern __shared__ char smem_raw[];
    T* sm = reinterpret_cast<T*>(smem_raw);   // TW segments of line_len, sm[la*line_len + pos]

    const long tile_start = (long)blockIdx.x * TW;
    if (tile_start >= n_a) return;
    const int  tw_eff = (int)((n_a - tile_start) < TW ? (n_a - tile_start) : TW);
    const long base_b = (long)blockIdx.y * stride_b;

    const int tx  = threadIdx.x;              // local line index within the tile, [0,TW)
    const int ty  = threadIdx.y;               // row lane, [0,RPS)
    const int rps = blockDim.y;
    const int flat_t  = tx + ty * blockDim.x;   // reinterpreted flat index for the lifting phase
    const int flat_nt = blockDim.x * blockDim.y;

    const int even_len = line_len - line_len / 2;
    const int odd_len  = line_len / 2;

    if (!inverse) {
        // Coalesced gather: for fixed source-row, TW adjacent lines read together.
        if (tx < tw_eff) {
            for (int e = ty; e < even_len; e += rps)
                sm[tx * line_len + e] = data[base_b + (tile_start + tx) * 1 + (long)(2 * e) * elem_stride];
            for (int o = ty; o < odd_len; o += rps)
                sm[tx * line_len + even_len + o] = data[base_b + (tile_start + tx) * 1 + (long)(2 * o + 1) * elem_stride];
        }
        __syncthreads();

        for (int la = 0; la < tw_eff; ++la)
            analysis_line<T>(sm + la * line_len, line_len, flat_t, flat_nt);

        if (tx < tw_eff) {
            for (int k = ty; k < line_len; k += rps)
                data[base_b + (tile_start + tx) * 1 + (long)k * elem_stride] = sm[tx * line_len + k];
        }
    } else {
        if (tx < tw_eff) {
            for (int k = ty; k < line_len; k += rps)
                sm[tx * line_len + k] = data[base_b + (tile_start + tx) * 1 + (long)k * elem_stride];
        }
        __syncthreads();

        for (int la = 0; la < tw_eff; ++la)
            synthesis_line<T>(sm + la * line_len, line_len, flat_t, flat_nt);

        if (tx < tw_eff) {
            for (int e = ty; e < even_len; e += rps)
                data[base_b + (tile_start + tx) * 1 + (long)(2 * e) * elem_stride] = sm[tx * line_len + e];
            for (int o = ty; o < odd_len; o += rps)
                data[base_b + (tile_start + tx) * 1 + (long)(2 * o + 1) * elem_stride] = sm[tx * line_len + even_len + o];
        }
    }
}

// ── Level-count helpers (mirror sperr::num_of_xforms / can_use_dyadic) ────────

/// SPERR's sperr::num_of_xforms(len): count approx-length shrinks (len -= len/2)
/// while len stays >= 9, capped at 6. Computed ONCE from the governing dimension
/// and applied to every axis. (floor(log2 N)-2 is WRONG: N=8->0, N=64->3.)
inline int cdf97_num_levels(size_t len)
{
    size_t num = 0;
    while (len >= 9) { ++num; len -= len / 2; }
    return (int)(num < 6 ? num : 6);
}

/// Approx (low-subband) extent of `n` after `lev` levels: repeated n -= n/2.
inline int cdf97_ext(int n, int lev) { int v = n; for (int i = 0; i < lev; ++i) v -= v / 2; return v; }

/// Threads per line: line half-length rounded up to a warp, clamped to [32,1024].
inline int cdf97_block_threads(int line_len)
{
    int need = (line_len + 1) / 2;
    int tpb  = ((need + 31) / 32) * 32;
    if (tpb < 32)   tpb = 32;
    if (tpb > 1024) tpb = 1024;
    return tpb;
}

inline int cdf97_max3(int a, int b, int c) { int m = a > b ? a : b; return m > c ? m : c; }

/**
 * Threads per block for the PERSISTENT kernel specifically — deliberately much
 * narrower than `cdf97_block_threads()`. A cooperative launch needs every block
 * co-resident at once, so occupancy (block *count*) is the scarce resource, not
 * per-block width: a 544-thread block (cdf97_block_threads(1024)) fits only ~3
 * per SM (thread-count-capped), while a 128-thread block fits ~16, multiplying
 * the max cooperative grid ~5x. Each thread just does proportionally more
 * `for (i=t; i<n; i+=nt)` iterations inside analysis_line/synthesis_line —
 * correctness is unaffected by the width.
 */
inline int cdf97_persistent_block_threads() { return 128; }

/**
 * Can this volume use SPERR's dyadic 3-D scheme? Mirrors sperr::can_use_dyadic:
 * requires a genuine 3-D volume, and either equal XY/Z level counts or both >= 5.
 * Returns the shared level count via `*out_levels`; false => wavelet-packet.
 */
inline bool cdf97_dyadic3d_levels(int nx, int ny, int nz, int* out_levels)
{
    if (nz < 2 || ny < 2) return false;              // 2-D/1-D
    const int xy = cdf97_num_levels(nx < ny ? nx : ny);
    const int z  = cdf97_num_levels(nz);
    if (xy == z || (xy >= 5 && z >= 5)) { *out_levels = xy < z ? xy : z; return true; }
    return false;                                    // anisotropic -> wavelet packet
}

/**
 * How many adjacent lines `cdf97_axis_kernel_tiled` should batch per block.
 *
 * Measured directly on H100 (`scratchpad/cdf97_tile_probe.cu`): TW=4 is the
 * empirical sweet spot where it fits shared memory (2.2-2.5x vs untiled at
 * n_a=1024/2048); TW=2 is the fallback when 4 doesn't fit. **Bigger is not
 * better** — maximizing TW against the shared-memory budget alone (the first
 * version of this function) was actively harmful: it shrinks the block count
 * (`n_a/TW`) along with it, and at small `n_a` that starves occupancy far more
 * than coalescing helps. Measured regression at n_a=64 with TW pushed to its
 * shared-memory ceiling (32): 5x *slower* than untiled (12.2 -> 2.6 GB/s) from
 * only 2 resident blocks on a 132-SM GPU.
 *
 * So this is two gates, not one: shared-memory fit picks a starting TW (4, or
 * 2 if 4 doesn't fit), then the occupancy guard below halves it until the
 * resulting block count (`n_a/TW`) is at least one wave across every SM —
 * falling back to TW=1 (untiled) entirely if even that doesn't leave enough
 * blocks to bother.
 */
inline int cdf97_choose_tile_width(int line_len, size_t elem_bytes, long n_a)
{
    constexpr size_t kSmemBudget = 48 * 1024;
    const size_t line_bytes = (size_t)line_len * elem_bytes;
    if (line_bytes == 0 || line_bytes * 2 > kSmemBudget) return 1;   // can't even fit TW=2

    int tw = (line_bytes * 4 <= kSmemBudget) ? 4 : 2;

    static int cachedSM = -1;                 // process-lifetime cache; single-GPU assumption
    if (cachedSM < 0) {
        int dev = 0, sm = 0;
        if (cudaGetDevice(&dev) == cudaSuccess &&
            cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, dev) == cudaSuccess && sm > 0)
            cachedSM = sm;
        else
            cachedSM = 1;   // query failed: degrade to "any occupancy is enough" rather than crash
    }
    while (tw > 1 && (n_a / tw) < cachedSM) tw /= 2;
    return tw;
}

// ── Launch helper (multilaunch path: one host-issued kernel per axis/level) ──
template <typename T>
inline void cdf97_launch_axis(T* d, int line_len, long stride_a, long stride_b,
                              long elem_stride, long n_a, long n_b, bool inverse,
                              cudaStream_t stream)
{
    // Strided passes (elem_stride != 1) route to the coalesced-tile kernel —
    // see cdf97_axis_kernel_tiled's doc comment (measured 2.2-2.9x faster on
    // H100 by fixing memory coalescing, not a numerical change). Every such
    // pass in this file has stride_a == 1, the tiled kernel's precondition.
    const int tw = (elem_stride != 1) ? cdf97_choose_tile_width(line_len, sizeof(T), n_a) : 1;

    if (tw > 1) {
        const int    tt  = cdf97_block_threads(line_len) < 256 ? cdf97_block_threads(line_len) : 256;
        const int    rps = tt / tw > 0 ? tt / tw : 1;
        const size_t sh  = (size_t)tw * line_len * sizeof(T);
        dim3 block(tw, rps);
        dim3 grid((unsigned)((n_a + tw - 1) / tw), (unsigned)n_b);
        cdf97_axis_kernel_tiled<T><<<grid, block, sh, stream>>>(
            d, line_len, stride_b, elem_stride, n_a, tw, inverse);
        return;
    }

    const int    tpb = cdf97_block_threads(line_len);
    const size_t sh  = (size_t)line_len * sizeof(T);
    dim3 grid((unsigned)n_a, (unsigned)n_b);
    cdf97_axis_kernel<T><<<grid, tpb, sh, stream>>>(
        d, line_len, stride_a, stride_b, elem_stride, n_a, inverse);
}

// ── Pass descriptor (shared by the persistent kernel and its host builders) ──

/// One (axis, level) sweep — same parameters cdf97_launch_axis takes, but data
/// to be consumed by the persistent kernel's internal loop instead of a launch.
struct AxisPass {
    int  line_len;
    long stride_a, stride_b, elem_stride;
    int  n_a, n_b;
};

/// Worst case is 3-D dyadic/packet: 6 levels x up to 3 passes/level = 18.
constexpr int kCdf97MaxPasses = 20;

struct Cdf97PassList {
    AxisPass passes[kCdf97MaxPasses];
    int n = 0;
    void push(int line_len, long sa, long sb, long es, int na, int nb) {
        passes[n++] = AxisPass{line_len, sa, sb, es, na, nb};
    }
};

// ── Pass-list builders ────────────────────────────────────────────────────────
// Each mirrors its *_multilaunch counterpart's loop op-for-op (same order, same
// parameters) so the persistent and multilaunch paths are numerically identical.

inline Cdf97PassList cdf97_build_passes_1d(int nx, bool inverse)
{
    Cdf97PassList L;
    const int levels = cdf97_num_levels(nx);
    if (!inverse) {
        for (int lv = 0; lv < levels; ++lv) L.push(cdf97_ext(nx, lv), 0, 0, 1, 1, 1);
    } else {
        for (int lv = levels - 1; lv >= 0; --lv) L.push(cdf97_ext(nx, lv), 0, 0, 1, 1, 1);
    }
    return L;
}

inline Cdf97PassList cdf97_build_passes_2d(int nx, int ny, bool inverse)
{
    Cdf97PassList L;
    const int levels = cdf97_num_levels(nx < ny ? nx : ny);
    if (!inverse) {
        for (int lv = 0; lv < levels; ++lv) {
            const int cx = cdf97_ext(nx, lv), cy = cdf97_ext(ny, lv);
            L.push(cx, nx, 0, 1,  cy, 1);   // X
            L.push(cy, 1,  0, nx, cx, 1);   // Y
        }
    } else {
        for (int lv = levels - 1; lv >= 0; --lv) {
            const int cx = cdf97_ext(nx, lv), cy = cdf97_ext(ny, lv);
            L.push(cy, 1,  0, nx, cx, 1);   // undo Y
            L.push(cx, nx, 0, 1,  cy, 1);   // undo X
        }
    }
    return L;
}

inline Cdf97PassList cdf97_build_passes_3d_dyadic(int nx, int ny, int nz, int levels, bool inverse)
{
    Cdf97PassList L;
    const long plane = (long)nx * ny;
    if (!inverse) {
        for (int lv = 0; lv < levels; ++lv) {
            const int cx = cdf97_ext(nx, lv), cy = cdf97_ext(ny, lv), cz = cdf97_ext(nz, lv);
            L.push(cx, nx, plane, 1,  cy, cz);    // X
            L.push(cy, 1,  plane, nx, cx, cz);    // Y
            L.push(cz, 1,  nx,    plane, cx, cy); // Z
        }
    } else {
        for (int lv = levels - 1; lv >= 0; --lv) {
            const int cx = cdf97_ext(nx, lv), cy = cdf97_ext(ny, lv), cz = cdf97_ext(nz, lv);
            L.push(cz, 1,  nx,    plane, cx, cy); // undo Z
            L.push(cy, 1,  plane, nx, cx, cz);    // undo Y
            L.push(cx, nx, plane, 1,  cy, cz);    // undo X
        }
    }
    return L;
}

inline Cdf97PassList cdf97_build_passes_3d_packet(int nx, int ny, int nz, bool inverse)
{
    Cdf97PassList L;
    const long plane = (long)nx * ny;
    const int levels_z  = cdf97_num_levels(nz);
    const int levels_xy = cdf97_num_levels(nx < ny ? nx : ny);
    if (!inverse) {
        for (int lv = 0; lv < levels_z; ++lv)
            L.push(cdf97_ext(nz, lv), 1, nx, plane, nx, ny);
        for (int lv = 0; lv < levels_xy; ++lv) {
            const int cx = cdf97_ext(nx, lv), cy = cdf97_ext(ny, lv);
            L.push(cx, nx, plane, 1,  cy, nz);   // X
            L.push(cy, 1,  plane, nx, cx, nz);   // Y
        }
    } else {
        for (int lv = levels_xy - 1; lv >= 0; --lv) {
            const int cx = cdf97_ext(nx, lv), cy = cdf97_ext(ny, lv);
            L.push(cy, 1,  plane, nx, cx, nz);   // undo Y
            L.push(cx, nx, plane, 1,  cy, nz);   // undo X
        }
        for (int lv = levels_z - 1; lv >= 0; --lv)
            L.push(cdf97_ext(nz, lv), 1, nx, plane, nx, ny);
    }
    return L;
}

// ── Persistent kernel: all passes of one transform call, one launch ──────────

/**
 * Runs every pass in `list` in order, using a grid-wide barrier between passes
 * instead of ending the kernel. Every block in the launch grid participates in
 * every `grid.sync()` unconditionally (required for grid-sync correctness) —
 * blocks outside a given pass's `(n_a, n_b)` extent simply skip the pass's work
 * but still call the barrier, so control flow stays uniform across the grid.
 *
 * Must be launched with `cudaLaunchCooperativeKernel` — see
 * `cdf97_try_persistent()`, which also verifies (via
 * `cudaOccupancyMaxActiveBlocksPerMultiprocessor`) that the grid this call needs
 * actually fits in one cooperative wave before attempting the launch.
 */
template <typename T>
__global__ void cdf97_persistent_kernel(T* data, Cdf97PassList list, bool inverse)
{
    extern __shared__ char smem_raw[];
    T* sm = reinterpret_cast<T*>(smem_raw);
    namespace cg = cooperative_groups;
    cg::grid_group grid = cg::this_grid();

    for (int p = 0; p < list.n; ++p) {
        const AxisPass pass = list.passes[p];
        if (blockIdx.x < (unsigned)pass.n_a && blockIdx.y < (unsigned)pass.n_b) {
            const long base = (long)blockIdx.y * pass.stride_b + (long)blockIdx.x * pass.stride_a;
            const int even_len = pass.line_len - pass.line_len / 2;
            const int odd_len  = pass.line_len / 2;
            const int t = threadIdx.x, nt = blockDim.x;

            if (!inverse) {
                for (int e = t; e < even_len; e += nt)
                    sm[e] = data[base + (long)(2 * e) * pass.elem_stride];
                for (int o = t; o < odd_len; o += nt)
                    sm[even_len + o] = data[base + (long)(2 * o + 1) * pass.elem_stride];
                __syncthreads();

                analysis_line<T>(sm, pass.line_len, t, nt);

                for (int k = t; k < pass.line_len; k += nt)
                    data[base + (long)k * pass.elem_stride] = sm[k];
            } else {
                for (int k = t; k < pass.line_len; k += nt)
                    sm[k] = data[base + (long)k * pass.elem_stride];
                __syncthreads();

                synthesis_line<T>(sm, pass.line_len, t, nt);

                for (int e = t; e < even_len; e += nt)
                    data[base + (long)(2 * e) * pass.elem_stride] = sm[e];
                for (int o = t; o < odd_len; o += nt)
                    data[base + (long)(2 * o + 1) * pass.elem_stride] = sm[even_len + o];
            }
        }
        grid.sync();   // unconditional: every block, every pass — data written this
                       // pass must be visible before the next pass reads it.
    }
}

/**
 * Attempt the persistent (single-launch, grid-sync) path for `list`. Returns
 * false — without side effects — whenever it can't be used, so the caller can
 * fall back to `*_multilaunch`:
 *   - the device/driver doesn't support cooperative launch, or
 *   - the grid this call needs (max n_a * n_b across all passes, since every
 *     pass's blocks must be co-resident for the whole launch) doesn't fit in
 *     one cooperative wave on this GPU.
 * The second condition is what makes this self-adjust per-GPU rather than
 * gating on a hardcoded compute capability: a GPU with fewer SMs has a smaller
 * max wave, so it falls back more readily for the same field size — exactly
 * the "graceful degrade on smaller/older hardware" behavior wanted here.
 */
template <typename T>
inline bool cdf97_try_persistent(T* d, const Cdf97PassList& list, int maxLineLen,
                                 bool inverse, cudaStream_t stream)
{
    if (list.n == 0) return true;   // nothing to do; trivially "succeeded"

    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) return false;

    int coopOk = 0;
    cudaDeviceGetAttribute(&coopOk, cudaDevAttrCooperativeLaunch, device);
    if (!coopOk) return false;

    const int    tpb     = cdf97_persistent_block_threads();
    const size_t shBytes = (size_t)maxLineLen * sizeof(T);

    int maxA = 0, maxB = 0;
    for (int i = 0; i < list.n; ++i) {
        maxA = maxA > list.passes[i].n_a ? maxA : list.passes[i].n_a;
        maxB = maxB > list.passes[i].n_b ? maxB : list.passes[i].n_b;
    }

    int blocksPerSM = 0;
    cudaError_t oe = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocksPerSM, cdf97_persistent_kernel<T>, tpb, shBytes);
    if (oe != cudaSuccess || blocksPerSM <= 0) return false;

    int numSM = 0;
    cudaDeviceGetAttribute(&numSM, cudaDevAttrMultiProcessorCount, device);
    const long maxCoopBlocks = (long)blocksPerSM * numSM;
    const long neededBlocks  = (long)maxA * maxB;
    if (neededBlocks == 0 || neededBlocks > maxCoopBlocks) return false;

    Cdf97PassList listCopy = list;   // stable, addressable storage for kernel args
    bool invCopy = inverse;
    dim3 grid((unsigned)maxA, (unsigned)maxB);
    void* args[] = { (void*)&d, (void*)&listCopy, (void*)&invCopy };
    cudaError_t le = cudaLaunchCooperativeKernel(
        (void*)cdf97_persistent_kernel<T>, grid, dim3(tpb), args, shBytes, stream);
    return le == cudaSuccess;
}

// ── 1-D driver ───────────────────────────────────────────────────────────────
/**
 * Forward/inverse 1-D CDF 9/7 DWT on a length-nx signal, in place. One line;
 * multi-level recursion on the approx half (mirrors SPERR m_dwt1d/m_idwt1d).
 */
template <typename T>
inline void dwt1d_multilaunch(T* d, int nx, bool inverse, cudaStream_t stream)
{
    const int levels = cdf97_num_levels(nx);
    if (!inverse) {
        for (int lv = 0; lv < levels; ++lv)
            cdf97_launch_axis<T>(d, cdf97_ext(nx, lv), 0, 0, 1, 1, 1, false, stream);
    } else {
        for (int lv = levels - 1; lv >= 0; --lv)
            cdf97_launch_axis<T>(d, cdf97_ext(nx, lv), 0, 0, 1, 1, 1, true, stream);
    }
}

template <typename T>
inline void dwt1d(T* d, int nx, bool inverse, cudaStream_t stream)
{
    Cdf97PassList L = cdf97_build_passes_1d(nx, inverse);
    if (cdf97_try_persistent<T>(d, L, nx, inverse, stream)) return;
    dwt1d_multilaunch<T>(d, nx, inverse, stream);
}

// ── 2-D dyadic driver ────────────────────────────────────────────────────────
/**
 * Forward/inverse 2-D CDF 9/7 DWT on a row-major (ny x nx) field, in place.
 * Forward: per level, X-rows then Y-columns, recursing on the top-left corner.
 * Inverse: exact reverse (Y then X, levels descending). Bit-exact vs SPERR
 * (double). Assumes max(nx,ny)*sizeof(T) fits in shared memory (long-line TODO).
 */
template <typename T>
inline void dwt2d_multilaunch(T* d, int nx, int ny, bool inverse, cudaStream_t stream)
{
    const int levels = cdf97_num_levels(nx < ny ? nx : ny);  // SPERR: from min dim
    if (!inverse) {
        for (int lv = 0; lv < levels; ++lv) {
            const int cx = cdf97_ext(nx, lv), cy = cdf97_ext(ny, lv);
            cdf97_launch_axis<T>(d, cx, /*a=row*/nx, 0, /*elem*/1,  cy, 1, false, stream); // X
            cdf97_launch_axis<T>(d, cy, /*a=col*/1,  0, /*elem*/nx, cx, 1, false, stream); // Y
        }
    } else {
        for (int lv = levels - 1; lv >= 0; --lv) {
            const int cx = cdf97_ext(nx, lv), cy = cdf97_ext(ny, lv);
            cdf97_launch_axis<T>(d, cy, 1,  0, nx, cx, 1, true, stream);  // undo Y
            cdf97_launch_axis<T>(d, cx, nx, 0, 1,  cy, 1, true, stream);  // undo X
        }
    }
}

template <typename T>
inline void dwt2d(T* d, int nx, int ny, bool inverse, cudaStream_t stream)
{
    Cdf97PassList L = cdf97_build_passes_2d(nx, ny, inverse);
    const int maxdim = nx > ny ? nx : ny;
    if (cdf97_try_persistent<T>(d, L, maxdim, inverse, stream)) return;
    dwt2d_multilaunch<T>(d, nx, ny, inverse, stream);
}

// ── 3-D dyadic driver ────────────────────────────────────────────────────────
/**
 * Forward/inverse 3-D CDF 9/7 DWT on a row-major (nz x ny x nx) volume, in place,
 * using SPERR's **dyadic** scheme. Forward: per level, X then Y on every z-plane,
 * then Z-columns. Inverse: Z, then Y then X per plane, levels descending.
 *
 * Precondition: cdf97_dyadic3d_levels(...) is true (equal XY/Z level counts or
 * both >= 5). For anisotropic volumes SPERR uses wavelet-packet — this function
 * no-ops and returns false in that case. Returns true if it ran.
 */
template <typename T>
inline bool dwt3d_dyadic_multilaunch(T* d, int nx, int ny, int nz, int levels,
                                     bool inverse, cudaStream_t stream)
{
    const long plane = (long)nx * ny;

    auto level_forward = [&](int cx, int cy, int cz) {
        cdf97_launch_axis<T>(d, cx, nx, plane, 1,  cy, cz, false, stream);  // X-rows
        cdf97_launch_axis<T>(d, cy, 1,  plane, nx, cx, cz, false, stream);  // Y-cols
        cdf97_launch_axis<T>(d, cz, 1,  nx,    plane, cx, cy, false, stream);
    };
    auto level_inverse = [&](int cx, int cy, int cz) {
        cdf97_launch_axis<T>(d, cz, 1,  nx,    plane, cx, cy, true, stream);   // undo Z
        cdf97_launch_axis<T>(d, cy, 1,  plane, nx, cx, cz, true, stream);      // undo Y
        cdf97_launch_axis<T>(d, cx, nx, plane, 1,  cy, cz, true, stream);      // undo X
    };

    if (!inverse) {
        for (int lv = 0; lv < levels; ++lv)
            level_forward(cdf97_ext(nx, lv), cdf97_ext(ny, lv), cdf97_ext(nz, lv));
    } else {
        for (int lv = levels - 1; lv >= 0; --lv)
            level_inverse(cdf97_ext(nx, lv), cdf97_ext(ny, lv), cdf97_ext(nz, lv));
    }
    return true;
}

template <typename T>
inline bool dwt3d_dyadic(T* d, int nx, int ny, int nz, bool inverse, cudaStream_t stream)
{
    int levels = 0;
    if (!cdf97_dyadic3d_levels(nx, ny, nz, &levels)) return false;  // wavelet packet, see dwt3d()

    Cdf97PassList L = cdf97_build_passes_3d_dyadic(nx, ny, nz, levels, inverse);
    const int maxdim = cdf97_max3(nx, ny, nz);
    if (cdf97_try_persistent<T>(d, L, maxdim, inverse, stream)) return true;
    return dwt3d_dyadic_multilaunch<T>(d, nx, ny, nz, levels, inverse, stream);
}

/**
 * Forward/inverse 3-D CDF 9/7 DWT using SPERR's **wavelet-packet** scheme, for
 * anisotropic volumes where the XY and Z level counts differ (and are not both
 * >= 5), i.e. cdf97_dyadic3d_levels() is false. In place, row-major.
 *
 * Unlike dyadic, the two axes are transformed to full depth independently, not
 * interleaved per level (that independence is what accommodates the mismatched
 * level counts). Forward: transform Z to its full depth over ALL nx*ny columns,
 * then transform XY to its full depth on EVERY one of the nz planes. Inverse:
 * XY planes first, then Z (mirrors SPERR m_idwt3d_wavelet_packet).
 *
 * The Z footprint stays the full nx*ny at every Z level (only the column length
 * cz shrinks); the plane count stays the full nz at every XY level (only the
 * XY corner shrinks).
 */
template <typename T>
inline void dwt3d_packet_multilaunch(T* d, int nx, int ny, int nz, bool inverse, cudaStream_t stream)
{
    const long plane     = (long)nx * ny;
    const int  levels_z  = cdf97_num_levels(nz);
    const int  levels_xy = cdf97_num_levels(nx < ny ? nx : ny);

    if (!inverse) {
        for (int lv = 0; lv < levels_z; ++lv) {
            const int cz = cdf97_ext(nz, lv);
            cdf97_launch_axis<T>(d, cz, 1, nx, plane, nx, ny, false, stream);
        }
        for (int lv = 0; lv < levels_xy; ++lv) {
            const int cx = cdf97_ext(nx, lv), cy = cdf97_ext(ny, lv);
            cdf97_launch_axis<T>(d, cx, nx, plane, 1,  cy, nz, false, stream);  // X-rows
            cdf97_launch_axis<T>(d, cy, 1,  plane, nx, cx, nz, false, stream);  // Y-cols
        }
    } else {
        for (int lv = levels_xy - 1; lv >= 0; --lv) {
            const int cx = cdf97_ext(nx, lv), cy = cdf97_ext(ny, lv);
            cdf97_launch_axis<T>(d, cy, 1,  plane, nx, cx, nz, true, stream);   // undo Y
            cdf97_launch_axis<T>(d, cx, nx, plane, 1,  cy, nz, true, stream);   // undo X
        }
        for (int lv = levels_z - 1; lv >= 0; --lv) {
            const int cz = cdf97_ext(nz, lv);
            cdf97_launch_axis<T>(d, cz, 1, nx, plane, nx, ny, true, stream);    // undo Z
        }
    }
}

template <typename T>
inline void dwt3d_packet(T* d, int nx, int ny, int nz, bool inverse, cudaStream_t stream)
{
    Cdf97PassList L = cdf97_build_passes_3d_packet(nx, ny, nz, inverse);
    const int maxdim = cdf97_max3(nx, ny, nz);
    if (cdf97_try_persistent<T>(d, L, maxdim, inverse, stream)) return;
    dwt3d_packet_multilaunch<T>(d, nx, ny, nz, inverse, stream);
}

/**
 * Top-level 3-D CDF 9/7 DWT: selects dyadic vs wavelet-packet exactly as SPERR's
 * CDF97::dwt3d()/idwt3d() do via can_use_dyadic. In place, row-major
 * (nz x ny x nx). Handles every volume shape. Each mode tries the persistent
 * (single-launch) path first and falls back to per-level launches transparently
 * — see the file-level doc comment.
 */
template <typename T>
inline void dwt3d(T* d, int nx, int ny, int nz, bool inverse, cudaStream_t stream)
{
    int levels = 0;
    if (cdf97_dyadic3d_levels(nx, ny, nz, &levels))
        dwt3d_dyadic<T>(d, nx, ny, nz, inverse, stream);
    else
        dwt3d_packet<T>(d, nx, ny, nz, inverse, stream);
}

// TODO(longline): line_len*sizeof(T) beyond the shared-memory budget (~6k
//   doubles at 48KB): opt into larger dynamic smem
//   (cudaFuncAttributeMaxDynamicSharedMemorySize) or a global-memory sweep.

}  // namespace cdf97
}  // namespace fz
