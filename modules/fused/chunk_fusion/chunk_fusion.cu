// Launcher for the chunk-cooperative fusion harness — see chunk_fusion.h/.cuh.
// The harness (chunk_fused_kernel) composes the per-stage device ops; this file
// picks the coder instantiation and runs the shared cross-chunk scan+pack tail.

#include "fused/chunk_fusion/chunk_fusion.h"
#include "fused/chunk_fusion/chunk_fusion.cuh"
#include "mem/mempool.h"
#include "cuda_check.h"

#include "backend/cub.h"
#include "backend/algorithms.h"
#include <cub/device/device_scan.cuh>
#include <thrust/iterator/transform_iterator.h>

namespace fz {
namespace fused {

namespace {

using namespace fz::fused::chunk;

__global__ void chunk_pack(const byte* __restrict__ scratch, byte* __restrict__ outp,
                           const uint32_t* __restrict__ off, const uint32_t* __restrict__ sz,
                           uint32_t header) {
    const uint32_t cid = blockIdx.x;
    const uint32_t n   = sz[cid] & 0x7FFFFFFFu;
    const byte* s = scratch + (size_t)cid * CHUNK_BYTES;
    byte* d = outp + header + off[cid];
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) d[i] = s[i];
}

struct StripFlag { __host__ __device__ uint32_t operator()(uint32_t x) const { return x & 0x7FFFFFFFu; } };

// Encode kernel + header + exclusive-scan offsets + pack. Templated on the coder;
// the transform chain (Diff -> Bitshuffle) is the PFPL body. A different pipeline
// shape maps to a different Chain<...> / QuantOp here (that mapping is the registry).
template<class Coder>
size_t runChunkPfpl(const float* d_in, size_t n, QuantInplaceZigzag::Params qp,
                    uint8_t* d_out, MemoryPool* pool, cudaStream_t stream) {
    const size_t   nc     = (n + NELEM - 1) / NELEM;
    const uint32_t header = 8u + 4u * (uint32_t)nc;

    auto* d_scratch = static_cast<byte*>(pool->allocate(nc * CHUNK_BYTES, stream, "chunk_scratch"));
    auto* d_sizes   = static_cast<uint32_t*>(pool->allocate(nc * 4, stream, "chunk_sizes"));
    auto* d_off     = static_cast<uint32_t*>(pool->allocate(nc * 4, stream, "chunk_off"));

    chunk_fused_kernel<QuantInplaceZigzag, Coder, DiffNegabinary, Bitshuffle32>
        <<<(unsigned)nc, TPB, 0, stream>>>(d_in, n, qp, d_scratch, d_sizes);
    FZ_CUDA_CHECK(cudaGetLastError());

    const uint32_t hdr[2] = { (uint32_t)(n * 4), (uint32_t)nc };
    FZ_CUDA_CHECK(cudaMemcpyAsync(d_out, hdr, 8, cudaMemcpyHostToDevice, stream));
    FZ_CUDA_CHECK(cudaMemcpyAsync(d_out + 8, d_sizes, nc * 4, cudaMemcpyDeviceToDevice, stream));

    auto strip = thrust::make_transform_iterator((const uint32_t*)d_sizes, StripFlag{});
    auto tmp = fz::backend::withTempStorage(pool, stream, "chunk_cub",
        [&](void* t, size_t& b) { cub::DeviceScan::ExclusiveSum(t, b, strip, d_off, (int)nc, stream); });

    chunk_pack<<<(unsigned)nc, 256, 0, stream>>>(d_scratch, d_out, d_off, d_sizes, header);
    FZ_CUDA_CHECK(cudaGetLastError());

    uint32_t last_off = 0, last_sz = 0;
    FZ_CUDA_CHECK(cudaMemcpyAsync(&last_off, d_off + nc - 1, 4, cudaMemcpyDeviceToHost, stream));
    FZ_CUDA_CHECK(cudaMemcpyAsync(&last_sz,  d_sizes + nc - 1, 4, cudaMemcpyDeviceToHost, stream));
    FZ_CUDA_CHECK(cudaStreamSynchronize(stream));

    const size_t total  = header + last_off + (last_sz & 0x7FFFFFFFu);
    const size_t padded = (total + 3) & ~size_t(3);
    if (padded > total)
        FZ_CUDA_CHECK(cudaMemsetAsync(d_out + total, 0, padded - total, stream));
    FZ_CUDA_CHECK(cudaStreamSynchronize(stream));

    fz::backend::freeTempStorage(pool, tmp, stream);
    pool->free(d_off, stream);
    pool->free(d_sizes, stream);
    pool->free(d_scratch, stream);
    return padded;
}

} // namespace

size_t launchFusedChunkPfpl(
    ChunkCoderKind coder, const float* d_in, size_t n, float ebx2_r,
    uint32_t radius, float threshold, uint8_t* d_out, MemoryPool* pool, cudaStream_t stream)
{
    if (n == 0) return 0;
    QuantInplaceZigzag::Params qp{ ebx2_r, radius, threshold };
    switch (coder) {
        case ChunkCoderKind::RRE: return runChunkPfpl<RRECoder>(d_in, n, qp, d_out, pool, stream);
        case ChunkCoderKind::RZE:
        default:                  return runChunkPfpl<RZECoder>(d_in, n, qp, d_out, pool, stream);
    }
}

} // namespace fused
} // namespace fz
