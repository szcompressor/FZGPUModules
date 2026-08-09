#include "predictors/tiled_lorenzo/tiled_lorenzo_stage.h"
#include "mem/mempool.h"
#include "cuda_check.h"
#include "backend/api.h"
#include <stdexcept>

namespace fz {

// Tile/grid geometry passed to both kernels.
struct TiledGeom {
    uint32_t dx, dy, dz;     // data dims
    uint32_t tx, ty, tz;     // tile dims
    uint32_t ntx, nty, ntz;  // tile counts per axis
    uint32_t tile_elems;     // tx*ty*tz
};

// ─────────────────────────────────────────────────────────────────────────────
// Forward: natural row-major codes -> tile-major separable deltas (cuSZp3).
// One thread per tile-major output element. Padding elements (data coord out of
// range) are written as 0. In-range elements always have in-range predecessors
// within the same tile, so neighbour reads are safe.
// ─────────────────────────────────────────────────────────────────────────────
template<typename T>
__global__ void tiled_lorenzo_delta_kernel(
    const T* __restrict__ in, T* __restrict__ out,
    TiledGeom g, size_t total_out)
{
    const size_t oidx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (oidx >= total_out) return;

    const uint32_t te    = g.tile_elems;
    const uint32_t t     = static_cast<uint32_t>(oidx / te);   // tile index
    const uint32_t local = static_cast<uint32_t>(oidx % te);   // element within tile

    // Decompose local (x fastest, then y, then z).
    const uint32_t lx = local % g.tx;
    const uint32_t ly = (local / g.tx) % g.ty;
    const uint32_t lz = local / (g.tx * g.ty);

    // Decompose tile index.
    const uint32_t tix = t % g.ntx;
    const uint32_t tiy = (t / g.ntx) % g.nty;
    const uint32_t tiz = t / (g.ntx * g.nty);

    const uint32_t gx = tix * g.tx + lx;
    const uint32_t gy = tiy * g.ty + ly;
    const uint32_t gz = tiz * g.tz + lz;

    if (gx >= g.dx || gy >= g.dy || gz >= g.dz) {  // padding
        out[oidx] = static_cast<T>(0);
        return;
    }

    const size_t gidx = (static_cast<size_t>(gz) * g.dy + gy) * g.dx + gx;
    const T cur = in[gidx];

    T pred;
    if (lx > 0)        pred = in[gidx - 1];                                 // X-delta
    else if (ly > 0)   pred = in[gidx - g.dx];                             // Y-delta
    else if (lz > 0)   pred = in[gidx - static_cast<size_t>(g.dx) * g.dy]; // Z-delta
    else               pred = static_cast<T>(0);                          // tile origin

    out[oidx] = static_cast<T>(cur - pred);
}

// ─────────────────────────────────────────────────────────────────────────────
// Inverse (phased, one CUDA block per tile): the same separable prefix-sum as
// tiled_lorenzo_scan_kernel above, but computed by tile_elems threads instead
// of one. The three chains traced from the serial kernel's prevQuant_z/_y/_x
// update rules are each independently parallelizable (addition is
// associative) — this just stops running them all on a single thread:
//   Phase 1 (thread 0):        Z-chain, length tz             -> s_zseed[tz]
//   Phase 2 (threads 0..tz-1): per-lz Y-chain, length ty       -> s_yseed[ty*tz]
//   Phase 3 (threads 0..ty*tz-1): per-(ly,lz) X-chain, length tx -> out[]
// Every (lx,ly,lz) grid point is written exactly once: (0,0,lz) in phase 1,
// (0,ly>0,lz) in phase 2, (lx>0,ly,lz) in phase 3 — matching the serial
// kernel's unconditional per-iteration out[gidx] write.
// ─────────────────────────────────────────────────────────────────────────────
template<typename T>
__global__ void tiled_lorenzo_scan_kernel_phased(
    const T* __restrict__ in, T* __restrict__ out,
    TiledGeom g, size_t num_tiles)
{
    extern __shared__ unsigned char s_raw[];
    T* s_zseed = reinterpret_cast<T*>(s_raw);   // [tz]
    T* s_yseed = s_zseed + g.tz;                // [ty*tz]

    const size_t t = blockIdx.x;
    if (t >= num_tiles) return;

    const uint32_t tix = static_cast<uint32_t>(t % g.ntx);
    const uint32_t tiy = static_cast<uint32_t>((t / g.ntx) % g.nty);
    const uint32_t tiz = static_cast<uint32_t>(t / (static_cast<size_t>(g.ntx) * g.nty));
    const size_t base = t * g.tile_elems;
    const uint32_t tid = threadIdx.x;

    auto writeIfInRange = [&](uint32_t lx, uint32_t ly, uint32_t lz, T cur) {
        const uint32_t gx = tix * g.tx + lx;
        const uint32_t gy = tiy * g.ty + ly;
        const uint32_t gz = tiz * g.tz + lz;
        if (gx < g.dx && gy < g.dy && gz < g.dz) {
            const size_t gidx = (static_cast<size_t>(gz) * g.dy + gy) * g.dx + gx;
            out[gidx] = cur;
        }
    };

    // Phase 1: Z-chain (thread 0 only; tz is small — 1 for 2-D, 4 for 3-D).
    if (tid == 0) {
        T prev = 0;
        for (uint32_t lz = 0; lz < g.tz; ++lz) {
            const uint32_t local = (lz * g.ty) * g.tx;  // (lx=0, ly=0, lz)
            const T d = in[base + local];
            const T cur = (lz > 0) ? static_cast<T>(d + prev) : d;
            prev = cur;
            s_zseed[lz] = cur;
            writeIfInRange(0, 0, lz, cur);
        }
    }
    __syncthreads();

    // Phase 2: per-lz Y-chain, one thread per lz.
    if (tid < g.tz) {
        const uint32_t lz = tid;
        T prev = s_zseed[lz];
        s_yseed[0 * g.tz + lz] = prev;  // row ly=0 reuses the Z-chain value directly
        for (uint32_t ly = 1; ly < g.ty; ++ly) {
            const uint32_t local = (lz * g.ty + ly) * g.tx;  // (lx=0, ly, lz)
            const T d = in[base + local];
            const T cur = static_cast<T>(d + prev);
            prev = cur;
            s_yseed[ly * g.tz + lz] = cur;
            writeIfInRange(0, ly, lz, cur);
        }
    }
    __syncthreads();

    // Phase 3: per-(ly,lz) X-chain, one thread per (ly,lz) pair.
    const uint32_t rows = g.ty * g.tz;
    if (tid < rows) {
        const uint32_t ly = tid % g.ty;
        const uint32_t lz = tid / g.ty;
        T prev = s_yseed[ly * g.tz + lz];
        for (uint32_t lx = 1; lx < g.tx; ++lx) {
            const uint32_t local = (lz * g.ty + ly) * g.tx + lx;
            const T d = in[base + local];
            const T cur = static_cast<T>(d + prev);
            prev = cur;
            writeIfInRange(lx, ly, lz, cur);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Inverse (per-row, fully parallel). One thread owns one x-row = one (tile,ly,lz).
// Replaces a phased one-block-per-tile kernel that was barrier-bound; do not go
// back to it. Every thread here is self-contained: exploiting the separable
// structure, a row thread re-derives its own seed by walking the tile's tiny x=0
// "spine" — the z-column prefix (delta(0,0,0..lz)) then the y-column prefix
// (delta(0,1..ly,lz)) — then runs its own tx-length x-chain, writing each
// in-range element.
// Profile numbers and the traffic argument: docs/codebase_notes.md CN-TLRZ-1
//
// Correctness matches the phased math exactly: reconstructed(lx,ly,lz) =
// Σ_{k≤lz} d(0,0,k) + Σ_{1≤k≤ly} d(0,k,lz) + Σ_{1≤k≤lx} d(k,ly,lz); padding deltas
// are 0, sit past the in-range extent, and are never stored.
// ─────────────────────────────────────────────────────────────────────────────
template<typename T>
__global__ void tiled_lorenzo_scan_kernel_rows(
    const T* __restrict__ in, T* __restrict__ out,
    TiledGeom g, size_t num_rows)
{
    const size_t r = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (r >= num_rows) return;

    const uint32_t tx = g.tx, ty = g.ty, tz = g.tz;
    const uint32_t rows_per_tile = ty * tz;

    const size_t   t   = r / rows_per_tile;          // tile index
    const uint32_t rr  = static_cast<uint32_t>(r % rows_per_tile);
    const uint32_t ly  = rr % ty;
    const uint32_t lz  = rr / ty;

    const size_t base = t * g.tile_elems;

    // Seed = reconstructed value at (lx=0, ly, lz): z-column prefix up to lz, then
    // y-column prefix (1..ly) at this lz. Both walk the tile's x=0 spine deltas.
    T seed = 0;
    for (uint32_t k = 0; k <= lz; ++k)
        seed = static_cast<T>(seed + in[base + (static_cast<size_t>(k) * ty) * tx]); // d(0,0,k)
    for (uint32_t k = 1; k <= ly; ++k)
        seed = static_cast<T>(seed + in[base + (static_cast<size_t>(lz) * ty + k) * tx]); // d(0,k,lz)

    // X-chain: reconstruct (lx,ly,lz) = seed + prefix of row deltas, write natural.
    const uint32_t tix = static_cast<uint32_t>(t % g.ntx);
    const uint32_t tiy = static_cast<uint32_t>((t / g.ntx) % g.nty);
    const uint32_t tiz = static_cast<uint32_t>(t / (static_cast<size_t>(g.ntx) * g.nty));
    const uint32_t gy = tiy * ty + ly;
    const uint32_t gz = tiz * tz + lz;
    if (gy >= g.dy || gz >= g.dz) return;   // whole row is padding — nothing to store

    const size_t rowbase = base + (static_cast<size_t>(lz) * ty + ly) * tx;
    const size_t out_row = (static_cast<size_t>(gz) * g.dy + gy) * g.dx + tix * tx;
    T cur = seed;
    const uint32_t gx0 = tix * tx;
    for (uint32_t lx = 0; lx < tx; ++lx) {
        if (lx > 0) cur = static_cast<T>(cur + in[rowbase + lx]);   // d(lx,ly,lz)
        if (gx0 + lx < g.dx) out[out_row + lx] = cur;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Launchers
// ─────────────────────────────────────────────────────────────────────────────
template<typename T>
static void launchDelta(const T* in, T* out, const TiledGeom& g,
                        size_t total_out, cudaStream_t stream)
{
    if (total_out == 0) return;
    const int kBlock = 256;
    const size_t grid = (total_out + kBlock - 1) / kBlock;
    tiled_lorenzo_delta_kernel<T>
        <<<static_cast<unsigned>(grid), kBlock, 0, stream>>>(in, out, g, total_out);
    FZ_CUDA_CHECK(cudaGetLastError());
}

template<typename T>
static void launchScan(const T* in, T* out, const TiledGeom& g,
                       size_t num_tiles, cudaStream_t stream)
{
    if (num_tiles == 0) return;

    // Fully-parallel per-row inverse: one thread per x-row (= one (tile,ly,lz)).
    // Correct for any tile shape; no barriers, no idle lanes. See kernel comment.
    const size_t num_rows = num_tiles * g.ty * g.tz;
    const unsigned kBlock = 256;
    const size_t grid = (num_rows + kBlock - 1) / kBlock;
    tiled_lorenzo_scan_kernel_rows<T>
        <<<static_cast<unsigned>(grid), kBlock, 0, stream>>>(in, out, g, num_rows);
    FZ_CUDA_CHECK(cudaGetLastError());
}

// ─────────────────────────────────────────────────────────────────────────────
// TiledLorenzoStage::execute
// ─────────────────────────────────────────────────────────────────────────────
template<typename T>
void TiledLorenzoStage<T>::execute(
    cudaStream_t stream,
    MemoryPool* /*pool*/,
    const std::vector<void*>& inputs,
    const std::vector<void*>& outputs,
    const std::vector<size_t>& sizes)
{
    if (inputs.empty() || outputs.empty() || sizes.empty())
        throw std::runtime_error("TiledLorenzoStage: inputs, outputs, and sizes must be non-empty");

    const size_t in_bytes = sizes[0];
    if (in_bytes == 0) { actual_output_size_ = 0; return; }

    // Resolve effective dims (flat 1-D fallback if dims unset).
    auto te = effectiveTile();
    size_t dx = dims_[0], dy = dims_[1], dz = dims_[2];
    if (dx == 0) { dx = in_bytes / sizeof(T); dy = 1; dz = 1; }

    TiledGeom g;
    g.dx = static_cast<uint32_t>(dx);
    g.dy = static_cast<uint32_t>(dy);
    g.dz = static_cast<uint32_t>(dz);
    g.tx = te[0]; g.ty = te[1]; g.tz = te[2];
    g.ntx = static_cast<uint32_t>((dx + g.tx - 1) / g.tx);
    g.nty = static_cast<uint32_t>((dy + g.ty - 1) / g.ty);
    g.ntz = static_cast<uint32_t>((dz + g.tz - 1) / g.tz);
    g.tile_elems = g.tx * g.ty * g.tz;

    const size_t num_tiles = static_cast<size_t>(g.ntx) * g.nty * g.ntz;
    const size_t padded    = num_tiles * g.tile_elems;
    const size_t natural   = dx * dy * dz;

    const T* in  = static_cast<const T*>(inputs[0]);
    T*       out = static_cast<T*>(outputs[0]);

    if (!is_inverse_) {
        launchDelta<T>(in, out, g, padded, stream);
        actual_output_size_ = padded * sizeof(T);
    } else {
        launchScan<T>(in, out, g, num_tiles, stream);
        actual_output_size_ = natural * sizeof(T);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Explicit instantiations
// ─────────────────────────────────────────────────────────────────────────────
template class TiledLorenzoStage<int16_t>;
template class TiledLorenzoStage<int32_t>;

} // namespace fz
