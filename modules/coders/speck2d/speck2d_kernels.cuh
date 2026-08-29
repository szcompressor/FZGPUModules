#pragma once

/**
 * @file speck2d_kernels.cuh
 * @brief Device kernels + host quadtree builder for the GPU-parallel "wavefront"
 *        SPECK-like coder (`Speck2DStage`).
 *
 * @note **Prior work / novelty.** This is NOT a port of SPERR's SPECK bitstream
 *       (SPERR uses linked LIP/LIS/LSP lists and a DFS-serial encode/decode with
 *       an embedded/progressive bit order). The design here is derived from first
 *       principles by reading the SPERR source (`SPECK_INT`/`SPECK2D_INT_*`) and
 *       recognizing that every significance decision SPECK makes is a pure
 *       function of a block-max quadtree pyramid (`onset[node] = max msb over its
 *       pixels`), which is embarrassingly parallel to build. Dropping SPECK's
 *       embedded/progressive bit ordering (unused by FZGM's error-bounded model)
 *       buys full data-parallelism on BOTH encode and decode, at a measured rate
 *       cost of ~1.10x-1.31x SPERR's SPECK payload (memory/speck_gpu_design.md).
 *       "Listless SPECK"/NLS/GPU-SPIHT/EBCOT are prior art for the general idea of
 *       replacing linked-list state with precomputed positional structure; the
 *       reorder-for-parallel-packing format and the GPU decode in particular are
 *       this implementation's own design (see memory/speck_gpu_design.md for the
 *       calibration).
 *
 * ### Format (v2 in the design doc)
 *
 * Given a 2-D field of (magnitude, sign) pairs, SPECK's quadtree partition
 * (`partition()` below, matching SPERR's `m_partition_S`: BR,BL,TR,TL split with
 * `len - len/2` / `len/2` halves) is built once per (nx,ny) SHAPE (data-
 * independent -- cached and reused across calls with the same dims). A node's
 * `onset` is the max `msb(magnitude)` over its pixels; the ENTIRE encode is then
 * two data-parallel prefix-sum passes:
 *
 *   Section A (tree): nodes in LEVEL-major order, present nodes only (present =
 *     "an ancestor's onset made this subtree worth describing"). Each present
 *     node emits a TERMINATED-UNARY gap = parent_onset - onset (`gap` zero bits
 *     then a '1'); a level's codewords are then self-delimiting, so decode is a
 *     parallel rank-the-ones pass, not a serial bit-by-bit walk.
 *   Section B (magnitude): significant leaves in leaf-array order, each a sign
 *     bit + `onset` mantissa bits, positions from a prefix-sum over leaf lengths.
 *
 * ### Level fusion (the throughput lever)
 *
 * A quadtree's level sizes are geometric in depth (1,4,16,...,4^L -- confirmed
 * empirically, not assumed), so most of a real field's ~11-13 levels are TINY
 * (<=1024 nodes) while only the last couple carry real data volume. The many
 * tiny levels are launch-latency-bound, not bandwidth-bound (measured: ~28 GB/s
 * plateau vs the CDF97 DWT's ~350 GB/s at the same field size, dominated by
 * kernel-launch count, not memory traffic) -- so `nFused` shallow levels (all
 * <=1024 nodes, hence <= one 1024-thread block) are FUSED into a single-block
 * kernel per side (`k_encode_shallow`/`k_decode_shallow`, `__syncthreads()`
 * between levels, no cooperative-launch machinery needed since it's one block).
 * The deep tail (real bandwidth work, not latency-bound) keeps ordinary
 * per-level launches. This measured +7% to +70% throughput depending on field
 * size (memory/speck_gpu_design.md, P3.6) with zero numerical change --
 * `k_encode_shallow`'s onset/visited arithmetic is byte-for-byte identical to
 * the multi-launch `k_onset`/`k_vis` kernels below, just fused into one launch.
 */

#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <algorithm>

namespace fz {
namespace speck2d {

// ── Host quadtree geometry (data-independent; built once per (nx,ny) shape) ───

struct Rect { int sx, sy, lx, ly; };
inline bool isPixel(const Rect& r) { return r.lx * r.ly == 1; }
inline bool isEmptyRect(const Rect& r) { return r.lx * r.ly == 0; }

/// SPERR's m_partition_S geometry: BR,BL,TR,TL, len split as (len-len/2, len/2).
inline void partitionRect(const Rect& s, Rect out[4]) {
    const int dx = s.lx / 2, dy = s.ly / 2;
    const int ax = s.lx - dx, ay = s.ly - dy;
    out[0] = {s.sx + ax, s.sy + ay, dx, dy};   // BR
    out[1] = {s.sx,      s.sy + ay, ax, dy};   // BL
    out[2] = {s.sx + ax, s.sy,      dx, ay};   // TR
    out[3] = {s.sx,      s.sy,      ax, ay};   // TL
}

/// Node index == DFS pre-order position (== level-major position after grouping
/// by `level[]`, since level.reserve/push_back below tracks it separately).
struct Tree {
    std::vector<int> parent, child[4], level, is_leaf, pixel;
    int max_level = 0;
    int nnodes() const { return (int)parent.size(); }
    int add(int par, int lvl) {
        int i = (int)parent.size();
        parent.push_back(par);
        for (int c = 0; c < 4; ++c) child[c].push_back(-1);
        level.push_back(lvl); is_leaf.push_back(0); pixel.push_back(-1);
        if (lvl > max_level) max_level = lvl;
        return i;
    }
};

inline int buildTreeRec(Tree& t, int nx, Rect r, int par, int lvl) {
    int idx = t.add(par, lvl);
    if (isPixel(r)) { t.is_leaf[idx] = 1; t.pixel[idx] = r.sy * nx + r.sx; return idx; }
    Rect ch[4]; partitionRect(r, ch);
    for (int c = 0; c < 4; ++c)
        if (!isEmptyRect(ch[c])) t.child[c][idx] = buildTreeRec(t, nx, ch[c], idx, lvl + 1);
    return idx;
}
inline Tree buildTree(int nx, int ny) {
    Tree t; buildTreeRec(t, nx, Rect{0, 0, nx, ny}, -1, 0); return t;
}

/// Fixed threshold: single-block fusion covers levels whose node count fits one
/// `kShallowCap`-thread block. Levels grow geometrically (~4x/depth), so this
/// covers most of the tree's DEPTH (the many-tiny-launches problem) while
/// leaving the (few) deep, genuinely large levels to ordinary per-level launches.
constexpr int kShallowCap = 1024;
inline int chooseShallowLevels(const std::vector<int>& level_count) {
    int L = 0;
    while (L < (int)level_count.size() && level_count[L] <= kShallowCap) ++L;
    return L;
}

// ── Elementwise: signed int32 code <-> (uint32 magnitude, uint8 sign) ─────────
// Two's-complement round trip via int64_t to avoid UB at INT32_MIN.
__global__ inline void k_split_sign_magnitude(const int32_t* code, int n, uint32_t* mag, uint8_t* sgn) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    int64_t v = code[i];
    sgn[i] = (v < 0) ? 1 : 0;
    mag[i] = (uint32_t)(v < 0 ? -v : v);
}
__global__ inline void k_join_sign_magnitude(const uint32_t* mag, const uint8_t* sgn, int n, int32_t* code) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    int64_t m = mag[i];
    code[i] = (int32_t)(sgn[i] ? -m : m);
}

__device__ __forceinline__ int d_getbit(const uint8_t* s, uint64_t pos) { return (s[pos >> 3] >> (pos & 7)) & 1; }

// ── Shared onset/visited kernels (multi-launch, deep-tail path) ───────────────
__global__ inline void k_msb(const uint32_t* c, int n, int* m) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    uint32_t v = c[i]; m[i] = v ? (31 - __clz(v)) : -1;
}
__global__ inline void k_onset(const int* nodes, int cnt, const int* lf, const int* px,
    const int* c0, const int* c1, const int* c2, const int* c3, const int* pm, int* on) {
    int t = blockIdx.x * blockDim.x + threadIdx.x; if (t >= cnt) return;
    int nd = nodes[t]; int o;
    if (lf[nd]) o = pm[px[nd]];
    else { o = -1; int cc[4] = {c0[nd], c1[nd], c2[nd], c3[nd]};
        for (int k = 0; k < 4; ++k) if (cc[k] >= 0) o = max(o, on[cc[k]]); }
    on[nd] = o;
}
__global__ inline void k_visited(const int* nodes, int cnt, const int* par, const int* on, uint8_t* vis) {
    int t = blockIdx.x * blockDim.x + threadIdx.x; if (t >= cnt) return;
    int nd = nodes[t]; int p = par[nd];
    vis[nd] = (p < 0) ? (uint8_t)(on[nd] >= 0) : (uint8_t)(vis[p] && on[p] >= 0);
}

// ── Fused single-block kernel: shallow levels [0,nLevels), encode side ────────
// Bottom-up onset then top-down visited, __syncthreads() between levels -- no
// cooperative launch needed (one block). Byte-identical arithmetic to
// k_onset/k_visited above, just fused. See file doc comment for why only the
// shallow prefix fuses.
__global__ inline void k_encode_shallow(int nLevels, const int* starts, const int* counts,
    const int* levelnodes, const int* is_leaf, const int* pixel,
    const int* c0, const int* c1, const int* c2, const int* c3,
    const int* parent, const int* msb, int* onset, uint8_t* visited)
{
    const int tid = threadIdx.x;
    for (int L = nLevels - 1; L >= 0; --L) {
        const int C = counts[L];
        if (tid < C) {
            int nd = levelnodes[starts[L] + tid];
            int on;
            if (is_leaf[nd]) on = msb[pixel[nd]];
            else {
                on = -1; int cc[4] = {c0[nd], c1[nd], c2[nd], c3[nd]};
                for (int k = 0; k < 4; ++k) if (cc[k] >= 0) on = max(on, onset[cc[k]]);
            }
            onset[nd] = on;
        }
        __syncthreads();
    }
    for (int L = 0; L < nLevels; ++L) {
        const int C = counts[L];
        if (tid < C) {
            int nd = levelnodes[starts[L] + tid];
            int p = parent[nd];
            visited[nd] = (p < 0) ? (uint8_t)(onset[nd] >= 0)
                                  : (uint8_t)(visited[p] && onset[p] >= 0);
        }
        __syncthreads();
    }
}

// ── Section A/B bit counts + packing (encode) ──────────────────────────────────
// Root's parent_onset == onset[0] identically (root's rect IS the whole image,
// so B == rect_onset(whole image) == onset[node 0]) -- reading on[0] in-kernel
// avoids a host round trip to learn B before these can launch.
__global__ inline void k_bitsA(int nn, const int* levelnodes, const int* par, const uint8_t* vis,
    const int* on, int* bitsA) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= nn) return;
    int nd = levelnodes[idx];
    if (!vis[nd]) { bitsA[idx] = 0; return; }
    int p = par[nd]; int po = (p < 0) ? on[0] : on[p]; int o = on[nd];
    bitsA[idx] = (po - o) + 1;
}
__global__ inline void k_packA(int nn, const int* levelnodes, const int* par, const uint8_t* vis,
    const int* on, const int* offA, uint32_t* out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= nn) return;
    int nd = levelnodes[idx]; if (!vis[nd]) return;
    int p = par[nd]; int po = (p < 0) ? on[0] : on[p]; int o = on[nd];
    uint64_t pos = (uint64_t)offA[idx] + (uint64_t)(po - o);
    atomicOr(&out[pos >> 5], 1u << (unsigned)(pos & 31));
}
__global__ inline void k_bitsB(int nl, const int* leaves, const uint8_t* vis, const int* on, int* bitsB) {
    int j = blockIdx.x * blockDim.x + threadIdx.x; if (j >= nl) return;
    int nd = leaves[j]; bitsB[j] = (vis[nd] && on[nd] >= 0) ? (1 + on[nd]) : 0;
}
__global__ inline void k_packB(int nl, const int* leaves, const uint8_t* vis, const int* on, const int* pixel,
    const uint32_t* coeff, const uint8_t* sgn, uint64_t nbitsA, const int* offB, uint32_t* out) {
    int j = blockIdx.x * blockDim.x + threadIdx.x; if (j >= nl) return;
    int nd = leaves[j]; if (!(vis[nd] && on[nd] >= 0)) return;
    int o = on[nd]; uint64_t pos = nbitsA + (uint64_t)offB[j];
    int px = pixel[nd];
    if (sgn[px]) atomicOr(&out[pos >> 5], 1u << (unsigned)(pos & 31));
    pos++;
    uint32_t c = coeff[px];
    for (int b = 0; b < o; ++b)
        if ((c >> b) & 1u) atomicOr(&out[(pos + b) >> 5], 1u << (unsigned)((pos + b) & 31));
}

// ── Decode: Phase A1 (parse Section A's terminated-unary codewords) ──────────
__global__ inline void k_bitflags(const uint8_t* stream, uint64_t nbitsA, int* flag) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nbitsA) flag[i] = d_getbit(stream, i);
}
__global__ inline void k_scatter_ones(const int* flag, const int* rank, uint64_t nbitsA, uint64_t* ones_pos) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nbitsA && flag[i]) ones_pos[rank[i]] = i;
}
__global__ inline void k_gaps(const uint64_t* ones_pos, int num_ones, int* gaps) {
    int k = blockIdx.x * blockDim.x + threadIdx.x; if (k >= num_ones) return;
    long long prev = k ? (long long)ones_pos[k - 1] : -1;
    gaps[k] = (int)((long long)ones_pos[k] - prev - 1);
}

// ── Decode: Phase A2 (wavefront onset assignment), deep-tail multi-launch ────
__global__ inline void k_present(const int* nodes, int C, const int* parent, const int* onset, int B,
    uint8_t* present, int* flag) {
    int j = blockIdx.x * blockDim.x + threadIdx.x; if (j >= C) return;
    int nd = nodes[j]; int p = parent[nd];
    uint8_t pr = (p < 0) ? (uint8_t)(B >= 0) : (uint8_t)(present[p] && onset[p] >= 0);
    present[nd] = pr; flag[j] = pr;
}
__global__ inline void k_assign(const int* nodes, int C, const uint8_t* present, const int* rank, int cursor,
    const int* parent, int* onset, int B, const int* gaps) {
    int j = blockIdx.x * blockDim.x + threadIdx.x; if (j >= C) return;
    int nd = nodes[j]; if (!present[nd]) return;
    int p = parent[nd]; int pon = (p < 0) ? B : onset[p];
    int k = cursor + rank[j];
    onset[nd] = pon - gaps[k];
}

// ── Fused single-block kernel: shallow levels [0,nLevels), decode side ────────
// Carries the codeword cursor in a __shared__ variable across levels; a
// block-wide Hillis-Steele exclusive scan replaces the per-level CUB
// DeviceScan + its 2 device->host reads that the deep-tail path still uses.
__global__ inline void k_decode_shallow(int nLevels, const int* starts, const int* counts,
    const int* levelnodes, const int* parent, int B,
    uint8_t* present, int* onset, const int* gaps, int* cursor_out)
{
    __shared__ int scan[kShallowCap];
    __shared__ int cursor;
    const int tid = threadIdx.x;
    if (tid == 0) cursor = 0;
    __syncthreads();

    for (int L = 0; L < nLevels; ++L) {
        const int C = counts[L];
        int nd = -1; uint8_t pr = 0;
        if (tid < C) {
            nd = levelnodes[starts[L] + tid];
            int p = parent[nd];
            pr = (p < 0) ? (uint8_t)(B >= 0) : (uint8_t)(present[p] && onset[p] >= 0);
            present[nd] = pr;
        }
        scan[tid] = (tid < C) ? (int)pr : 0;
        __syncthreads();
        for (int off = 1; off < blockDim.x; off <<= 1) {
            int v = (tid >= off) ? scan[tid - off] : 0;
            __syncthreads();
            scan[tid] += v;
            __syncthreads();
        }
        const int excl = scan[tid] - (int)pr;
        if (tid < C) {
            if (pr) {
                int k = cursor + excl;
                int p = parent[nd];
                int pon = (p < 0) ? B : onset[p];
                onset[nd] = pon - gaps[k];
            } else {
                onset[nd] = -1;
            }
        }
        __syncthreads();
        if (tid == 0 && C > 0) cursor += scan[C - 1];
        __syncthreads();
    }
    if (tid == 0) *cursor_out = cursor;
}

// ── Decode: Phase B (magnitude fill) ──────────────────────────────────────────
__global__ inline void k_leaflen(const int* leaves, int nl, const uint8_t* present, const int* onset, int* len) {
    int j = blockIdx.x * blockDim.x + threadIdx.x; if (j >= nl) return;
    int nd = leaves[j]; len[j] = (present[nd] && onset[nd] >= 0) ? (1 + onset[nd]) : 0;
}
__global__ inline void k_fill(const int* leaves, int nl, const uint8_t* present, const int* onset, const int* pixel,
    const uint8_t* stream, uint64_t nbitsA, const int* off, uint32_t* coeff, uint8_t* sgn) {
    int j = blockIdx.x * blockDim.x + threadIdx.x; if (j >= nl) return;
    int nd = leaves[j]; if (!(present[nd] && onset[nd] >= 0)) return;
    int on = onset[nd]; uint64_t pos = nbitsA + (uint64_t)off[j];
    int px = pixel[nd];
    uint8_t s = d_getbit(stream, pos); pos++;
    uint32_t mant = 0; for (int b = 0; b < on; ++b) if (d_getbit(stream, pos + b)) mant |= (1u << b);
    coeff[px] = (1u << on) | mant; sgn[px] = s;
}

}  // namespace speck2d
}  // namespace fz
