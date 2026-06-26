#include "predictors/tiled_lorenzo/tiled_lorenzo_stage.h"
#include "mem/mempool.h"
#include "cuda_check.h"
#include <cuda_runtime.h>
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
// Inverse: tile-major separable deltas -> natural row-major codes.
// One thread per tile; storage-free separable prefix-sum (3 running scalars),
// mirroring the cuSZp3 decompress update rules. Writes only in-range coords.
// ─────────────────────────────────────────────────────────────────────────────
template<typename T>
__global__ void tiled_lorenzo_scan_kernel(
    const T* __restrict__ in, T* __restrict__ out,
    TiledGeom g, size_t num_tiles)
{
    const size_t t = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= num_tiles) return;

    const uint32_t tix = static_cast<uint32_t>(t % g.ntx);
    const uint32_t tiy = static_cast<uint32_t>((t / g.ntx) % g.nty);
    const uint32_t tiz = static_cast<uint32_t>(t / (static_cast<size_t>(g.ntx) * g.nty));
    const size_t base = t * g.tile_elems;

    T prevQuant_z = 0;
    for (uint32_t lz = 0; lz < g.tz; ++lz) {
        const uint32_t gz = tiz * g.tz + lz;
        T prevQuant_y = 0;
        for (uint32_t ly = 0; ly < g.ty; ++ly) {
            const uint32_t gy = tiy * g.ty + ly;
            T prevQuant_x = 0;
            for (uint32_t lx = 0; lx < g.tx; ++lx) {
                const uint32_t local = (lz * g.ty + ly) * g.tx + lx;
                const T d = in[base + local];
                T cur;
                if (lx > 0) {
                    cur = static_cast<T>(d + prevQuant_x);
                } else if (ly > 0) {
                    cur = static_cast<T>(d + prevQuant_y);
                } else if (lz > 0) {
                    cur = static_cast<T>(d + prevQuant_z);
                    prevQuant_z = cur;
                } else {
                    cur = d;            // tile origin
                    prevQuant_z = cur;  // seed Z-chain at the (0,0,0) corner
                }
                if (lx == 0) prevQuant_y = cur;  // leading x-column feeds Y-chain
                prevQuant_x = cur;

                const uint32_t gx = tix * g.tx + lx;
                if (gx < g.dx && gy < g.dy && gz < g.dz) {
                    const size_t gidx =
                        (static_cast<size_t>(gz) * g.dy + gy) * g.dx + gx;
                    out[gidx] = cur;
                }
            }
        }
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
    const int kBlock = 128;
    const size_t grid = (num_tiles + kBlock - 1) / kBlock;
    tiled_lorenzo_scan_kernel<T>
        <<<static_cast<unsigned>(grid), kBlock, 0, stream>>>(in, out, g, num_tiles);
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
