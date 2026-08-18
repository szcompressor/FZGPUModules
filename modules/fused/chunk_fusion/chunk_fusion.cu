// Launcher for the chunk-cooperative fusion harness — see chunk_fusion.h/.cuh.
// The harness (chunk_fused_kernel) composes the per-stage device ops; this file
// picks the coder instantiation and runs the shared cross-chunk scan+pack tail.

#include "fused/chunk_fusion/chunk_fusion.h"
#include "fused/chunk_fusion/chunk_fusion.cuh"
#include "fused/chunk_fusion/nvrtc_chunk_fusion.h"
#include "mem/mempool.h"
#include "cuda_check.h"

#include "backend/cub.h"
#include "backend/algorithms.h"
#include <cub/device/device_scan.cuh>
#include <thrust/iterator/transform_iterator.h>

#include <cstdlib>   // getenv — FZ_FUSION_NVRTC toggle

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

bool useNvrtcFusion() {
    const char* e = std::getenv("FZ_FUSION_NVRTC");
    return e && e[0] == '1';
}

// Cross-chunk tail: header + exclusive-scan offsets + pack + 4-byte rounding.
// Shared by the template and NVRTC encode paths — given filled d_scratch +
// d_sizes it produces the byte-identical LC archive in d_out. Frees scratch/sizes.
size_t packChunks(const float* /*d_in*/, size_t n, size_t nc,
                  byte* d_scratch, uint32_t* d_sizes,
                  uint8_t* d_out, MemoryPool* pool, cudaStream_t stream) {
    const uint32_t header = 8u + 4u * (uint32_t)nc;
    auto* d_off = static_cast<uint32_t*>(pool->allocate(nc * 4, stream, "chunk_off"));

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

// Compile-time template encode (the default, non-NVRTC path). Picks the kernel
// instantiation for the coder; the transform chain (Diff -> Bitshuffle) is the
// PFPL body. A different pipeline shape maps to a different Chain<...> / QuantOp.
void encodeTemplate(ChunkCoderKind coder, const float* d_in, size_t n,
                    const byte* d_params, byte* d_scratch, uint32_t* d_sizes,
                    size_t nc, cudaStream_t stream) {
    switch (coder) {
        case ChunkCoderKind::RRE:
            chunk_fused_kernel<QuantInplaceZigzag, RRECoder, DiffNegabinary, Bitshuffle32>
                <<<(unsigned)nc, TPB, 0, stream>>>(d_in, n, d_params, d_scratch, d_sizes);
            break;
        case ChunkCoderKind::RZE:
        default:
            chunk_fused_kernel<QuantInplaceZigzag, RZECoder, DiffNegabinary, Bitshuffle32>
                <<<(unsigned)nc, TPB, 0, stream>>>(d_in, n, d_params, d_scratch, d_sizes);
            break;
    }
    FZ_CUDA_CHECK(cudaGetLastError());
}

} // namespace

size_t launchFusedChunkPfpl(
    ChunkCoderKind coder, const float* d_in, size_t n, float ebx2_r,
    uint32_t radius, float threshold, uint8_t* d_out, MemoryPool* pool, cudaStream_t stream)
{
    if (n == 0) return 0;
    const size_t nc = (n + NELEM - 1) / NELEM;

    auto* d_scratch = static_cast<byte*>(pool->allocate(nc * CHUNK_BYTES, stream, "chunk_scratch"));
    auto* d_sizes   = static_cast<uint32_t*>(pool->allocate(nc * 4, stream, "chunk_sizes"));

    // Build the packed params blob on device. For PFPL only the quant Map op is
    // parametric, so the blob is exactly its Params; the stateless diff/bitshuffle/
    // coder ops contribute nothing. (The generic runner of Phase C will assemble
    // this blob from each stage's getFusedOp().params instead of hardcoding it.)
    QuantInplaceZigzagParams qp{ ebx2_r, radius, threshold };
    auto* d_params = static_cast<byte*>(pool->allocate(sizeof(qp), stream, "chunk_params"));
    FZ_CUDA_CHECK(cudaMemcpyAsync(d_params, &qp, sizeof(qp), cudaMemcpyHostToDevice, stream));

    // Encode: the compile-time template kernel, or — when FZ_FUSION_NVRTC=1 — the
    // same harness body generated + JIT-compiled at runtime from a spec. Both
    // fill d_scratch/d_sizes identically from the same blob; the tail is shared.
    if (useNvrtcFusion()) {
        ChunkFusionSpec spec;                     // PFPL op names (defaults) + coder
        spec.coder = chunkCoderOpName(coder);
        launchNvrtcChunkFusedEncode(spec, d_in, n, d_params, d_scratch, d_sizes,
                                    (unsigned)nc, stream);
    } else {
        encodeTemplate(coder, d_in, n, d_params, d_scratch, d_sizes, nc, stream);
    }

    const size_t out_bytes = packChunks(d_in, n, nc, d_scratch, d_sizes, d_out, pool, stream);
    pool->free(d_params, stream);
    return out_bytes;
}

size_t launchGenericChunkFusion(
    const ChunkFusionSpec& spec, const float* d_in, size_t n,
    const uint8_t* host_params, size_t params_bytes,
    uint8_t* d_out, MemoryPool* pool, cudaStream_t stream)
{
    if (n == 0) return 0;
    const size_t nc = (n + NELEM - 1) / NELEM;

    auto* d_scratch = static_cast<byte*>(pool->allocate(nc * CHUNK_BYTES, stream, "chunk_scratch"));
    auto* d_sizes   = static_cast<uint32_t*>(pool->allocate(nc * 4, stream, "chunk_sizes"));

    // Upload the caller-assembled params blob (already ordered [Map][Trs...][Coder]).
    // Allocate >=1 byte so the device pointer is valid even with no parametric op.
    const size_t pbytes = params_bytes ? params_bytes : 1;
    auto* d_params = static_cast<byte*>(pool->allocate(pbytes, stream, "chunk_params"));
    if (params_bytes)
        FZ_CUDA_CHECK(cudaMemcpyAsync(d_params, host_params, params_bytes,
                                      cudaMemcpyHostToDevice, stream));

    // Compose + launch the fused encode from the spec (NVRTC), then the shared tail.
    launchNvrtcChunkFusedEncode(spec, d_in, n, d_params, d_scratch, d_sizes,
                                (unsigned)nc, stream);
    const size_t out_bytes = packChunks(d_in, n, nc, d_scratch, d_sizes, d_out, pool, stream);
    pool->free(d_params, stream);
    return out_bytes;
}

} // namespace fused
} // namespace fz
