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
#include <stdexcept>
#include <string>

namespace fz {
namespace fused {

namespace {

using namespace fz::fused::chunk;

// chunk_bytes is a runtime arg here (not a template param): the pack kernel only
// uses it for addressing stride, not shared-memory sizing, so it doesn't need a
// per-size instantiation the way the encode/decode harness kernels do.
__global__ void chunk_pack(const byte* __restrict__ scratch, byte* __restrict__ outp,
                           const uint32_t* __restrict__ off, const uint32_t* __restrict__ sz,
                           uint32_t header, uint32_t chunk_bytes) {
    const uint32_t cid = blockIdx.x;
    const uint32_t n   = sz[cid] & 0x7FFFFFFFu;
    const byte* s = scratch + (size_t)cid * chunk_bytes;
    byte* d = outp + header + off[cid];
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) d[i] = s[i];
}

struct StripFlag { __host__ __device__ uint32_t operator()(uint32_t x) const { return x & 0x7FFFFFFFu; } };

__global__ void scatter_pfpl_outliers(const float* __restrict__ vals,
                                      const uint32_t* __restrict__ idxs,
                                      uint32_t count, size_t n,
                                      float* __restrict__ out) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count && static_cast<size_t>(idxs[i]) < n) out[idxs[i]] = vals[i];
}

bool useNvrtcFusion() {
    const char* e = std::getenv("FZ_FUSION_NVRTC");
    return e && e[0] == '1';
}

// Cross-chunk tail: header + exclusive-scan offsets + pack + 4-byte rounding.
// Shared by the template and NVRTC encode paths — given filled d_scratch +
// d_sizes it produces the byte-identical LC archive in d_out. Frees scratch/sizes.
size_t packChunks(const float* /*d_in*/, size_t n, size_t nc,
                  byte* d_scratch, uint32_t* d_sizes,
                  uint8_t* d_out, MemoryPool* pool, cudaStream_t stream,
                  int chunk_bytes) {
    const uint32_t header = 8u + 4u * (uint32_t)nc;
    auto* d_off = static_cast<uint32_t*>(pool->allocate(nc * 4, stream, "chunk_off"));

    const uint32_t hdr[2] = { (uint32_t)(n * 4), (uint32_t)nc };
    FZ_CUDA_CHECK(cudaMemcpyAsync(d_out, hdr, 8, cudaMemcpyHostToDevice, stream));
    FZ_CUDA_CHECK(cudaMemcpyAsync(d_out + 8, d_sizes, nc * 4, cudaMemcpyDeviceToDevice, stream));

    auto strip = thrust::make_transform_iterator((const uint32_t*)d_sizes, StripFlag{});
    auto tmp = fz::backend::withTempStorage(pool, stream, "chunk_cub",
        [&](void* t, size_t& b) { cub::DeviceScan::ExclusiveSum(t, b, strip, d_off, (int)nc, stream); });

    chunk_pack<<<(unsigned)nc, 256, 0, stream>>>(d_scratch, d_out, d_off, d_sizes, header,
                                                  (uint32_t)chunk_bytes);
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
// instantiation for (coder, chunk_bytes); the transform chain (Diff -> Bitshuffle)
// is the PFPL body. A different pipeline shape maps to a different Chain<...> /
// QuantOp. One instantiation per (coder, chunk size) pair — mirrors the same
// switch-over-pre-instantiated-templates pattern RZEStage::launchEncode uses.
template <int ChunkBytes>
void encodeTemplateAt(ChunkCoderKind coder, const float* d_in, size_t n,
                      const byte* d_params, byte* d_scratch, uint32_t* d_sizes,
                      size_t nc, cudaStream_t stream) {
    switch (coder) {
        case ChunkCoderKind::RRE:
            chunk_fused_kernel<ChunkBytes, QuantInplaceZigzag, RRECoder<ChunkBytes>,
                               DiffNegabinary, Bitshuffle32<ChunkBytes>>
                <<<(unsigned)nc, TPB, 0, stream>>>(d_in, n, d_params, d_scratch, d_sizes);
            break;
        case ChunkCoderKind::RZE:
        default:
            chunk_fused_kernel<ChunkBytes, QuantInplaceZigzag, RZECoder<ChunkBytes>,
                               DiffNegabinary, Bitshuffle32<ChunkBytes>>
                <<<(unsigned)nc, TPB, 0, stream>>>(d_in, n, d_params, d_scratch, d_sizes);
            break;
    }
    FZ_CUDA_CHECK(cudaGetLastError());
}

void encodeTemplate(ChunkCoderKind coder, const float* d_in, size_t n,
                    const byte* d_params, byte* d_scratch, uint32_t* d_sizes,
                    size_t nc, cudaStream_t stream, int chunk_bytes) {
    switch (chunk_bytes) {
        case 4096:  encodeTemplateAt<4096>(coder, d_in, n, d_params, d_scratch, d_sizes, nc, stream); break;
        case 8192:  encodeTemplateAt<8192>(coder, d_in, n, d_params, d_scratch, d_sizes, nc, stream); break;
        case 16384: encodeTemplateAt<16384>(coder, d_in, n, d_params, d_scratch, d_sizes, nc, stream); break;
        default:
            throw std::runtime_error(
                "chunk_fusion: chunk_bytes must be 4096, 8192, or 16384; got "
                + std::to_string(chunk_bytes));
    }
}

} // namespace

size_t launchFusedChunkPfpl(
    ChunkCoderKind coder, const float* d_in, size_t n, float ebx2_r,
    uint32_t radius, float threshold, uint8_t* d_out, MemoryPool* pool, cudaStream_t stream,
    int chunk_bytes)
{
    if (n == 0) return 0;
    if (!isSupportedChunkBytes(chunk_bytes))
        throw std::runtime_error("launchFusedChunkPfpl: chunk_bytes must be 4096, 8192, or 16384");
    const size_t nelem = (size_t)chunk_bytes / 4;
    const size_t nc    = (n + nelem - 1) / nelem;

    auto* d_scratch = static_cast<byte*>(pool->allocate(nc * (size_t)chunk_bytes, stream, "chunk_scratch"));
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
        spec.coder       = chunkCoderOpName(coder);
        spec.chunk_bytes = chunk_bytes;
        launchNvrtcChunkFusedEncode(spec, d_in, n, d_params, d_scratch, d_sizes,
                                    (unsigned)nc, stream);
    } else {
        encodeTemplate(coder, d_in, n, d_params, d_scratch, d_sizes, nc, stream, chunk_bytes);
    }

    const size_t out_bytes = packChunks(d_in, n, nc, d_scratch, d_sizes, d_out, pool, stream, chunk_bytes);
    pool->free(d_params, stream);
    return out_bytes;
}

size_t launchGenericChunkFusion(
    const ChunkFusionSpec& spec, const float* d_in, size_t n,
    const uint8_t* host_params, size_t params_bytes,
    uint8_t* d_out, MemoryPool* pool, cudaStream_t stream,
    uint32_t* d_side_idxs, float* d_side_vals, uint32_t side_max, uint32_t* out_side_count)
{
    if (n == 0) {
        if (out_side_count) *out_side_count = 0;
        return 0;
    }
    if (!isSupportedChunkBytes(spec.chunk_bytes))
        throw std::runtime_error("launchGenericChunkFusion: spec.chunk_bytes must be 4096, 8192, or 16384");
    const size_t nelem = (size_t)spec.chunk_bytes / 4;
    const size_t nc    = (n + nelem - 1) / nelem;

    auto* d_scratch = static_cast<byte*>(pool->allocate(nc * (size_t)spec.chunk_bytes, stream, "chunk_scratch"));
    auto* d_sizes   = static_cast<uint32_t*>(pool->allocate(nc * 4, stream, "chunk_sizes"));

    // Upload the caller-assembled params blob (already ordered [Map][Trs...][Coder]).
    // Allocate >=1 byte so the device pointer is valid even with no parametric op.
    const size_t pbytes = params_bytes ? params_bytes : 1;
    auto* d_params = static_cast<byte*>(pool->allocate(pbytes, stream, "chunk_params"));
    if (params_bytes)
        FZ_CUDA_CHECK(cudaMemcpyAsync(d_params, host_params, params_bytes,
                                      cudaMemcpyHostToDevice, stream));

    // Split-outlier producer: a device append-counter, zeroed so the Map op's
    // atomicAdd starts at 0. The counter is GLOBAL across all chunk CTAs (the
    // outlier list is one pipeline output spanning every chunk).
    const bool side = (d_side_idxs != nullptr && d_side_vals != nullptr);
    uint32_t* d_side_count = nullptr;
    if (side) {
        d_side_count = static_cast<uint32_t*>(pool->allocate(sizeof(uint32_t), stream, "chunk_outlier_count"));
        FZ_CUDA_CHECK(cudaMemsetAsync(d_side_count, 0, sizeof(uint32_t), stream));
    }

    // Compose + launch the fused encode from the spec (NVRTC), then the shared tail.
    launchNvrtcChunkFusedEncode(spec, d_in, n, d_params, d_scratch, d_sizes,
                                (unsigned)nc, stream,
                                d_side_idxs, d_side_vals, d_side_count, side_max);
    const size_t out_bytes = packChunks(d_in, n, nc, d_scratch, d_sizes, d_out, pool, stream,
                                        spec.chunk_bytes);

    // Read back the outlier count (data-dependent, known only after the kernel). Sync
    // so the caller can size the side buffers from it immediately on return.
    if (side) {
        uint32_t count = 0;
        FZ_CUDA_CHECK(cudaMemcpyAsync(&count, d_side_count, sizeof(uint32_t),
                                      cudaMemcpyDeviceToHost, stream));
        FZ_CUDA_CHECK(cudaStreamSynchronize(stream));
        if (out_side_count) *out_side_count = count;
        pool->free(d_side_count, stream);
    } else if (out_side_count) {
        *out_side_count = 0;
    }
    pool->free(d_params, stream);
    return out_bytes;
}

// One instantiation per supported chunk size, mirroring encodeTemplateAt<>.
template <int ChunkBytes>
void launchInversePfplKernelAt(
    const uint8_t* d_archive, const uint32_t* d_entries, const uint32_t* d_offsets,
    size_t nc, size_t output_bytes, float ebx2, bool inplace_outliers,
    uint32_t quant_radius, float* d_out, cudaStream_t stream) {
    chunk_inverse_pfpl_kernel<ChunkBytes, RZECoder<ChunkBytes>>
        <<<static_cast<unsigned>(nc), TPB, 0, stream>>>(
            d_archive, d_entries, d_offsets, output_bytes, ebx2,
            inplace_outliers, quant_radius, d_out);
}

size_t launchFusedChunkPfplInverse(
    const uint8_t* d_archive, size_t archive_bytes, size_t output_bytes,
    float ebx2, bool inplace_outliers, uint32_t quant_radius,
    const float* d_outlier_vals, const uint32_t* d_outlier_idxs,
    uint32_t outlier_count, float* d_out, MemoryPool* pool, cudaStream_t stream,
    int chunk_bytes)
{
    if (output_bytes == 0) return 0;
    if (!d_archive || !d_out || !pool || output_bytes % sizeof(uint32_t) != 0)
        throw std::invalid_argument("launchFusedChunkPfplInverse: invalid buffer/size");
    if (!inplace_outliers && outlier_count > 0 &&
        (!d_outlier_vals || !d_outlier_idxs))
        throw std::runtime_error(
            "launchFusedChunkPfplInverse: missing split-outlier side inputs");
    if (!isSupportedChunkBytes(chunk_bytes))
        throw std::runtime_error("launchFusedChunkPfplInverse: chunk_bytes must be 4096, 8192, or 16384");

    const size_t nc = (output_bytes + (size_t)chunk_bytes - 1) / (size_t)chunk_bytes;
    const size_t header_bytes = 8u + nc * sizeof(uint32_t);
    if (archive_bytes < header_bytes)
        throw std::runtime_error("launchFusedChunkPfplInverse: truncated RZE header");

    const auto* d_entries = reinterpret_cast<const uint32_t*>(d_archive + 8);
    auto* d_offsets = static_cast<uint32_t*>(
        pool->allocate(nc * sizeof(uint32_t), stream, "chunk_inv_offsets"));
    auto strip = thrust::make_transform_iterator(d_entries, StripFlag{});
    auto tmp = fz::backend::withTempStorage(pool, stream, "chunk_inv_scan",
        [&](void* t, size_t& b) {
            cub::DeviceScan::ExclusiveSum(t, b, strip, d_offsets, static_cast<int>(nc), stream);
        });

    switch (chunk_bytes) {
        case 4096:
            launchInversePfplKernelAt<4096>(d_archive, d_entries, d_offsets, nc, output_bytes,
                                            ebx2, inplace_outliers, quant_radius, d_out, stream);
            break;
        case 8192:
            launchInversePfplKernelAt<8192>(d_archive, d_entries, d_offsets, nc, output_bytes,
                                            ebx2, inplace_outliers, quant_radius, d_out, stream);
            break;
        case 16384:
        default:
            launchInversePfplKernelAt<16384>(d_archive, d_entries, d_offsets, nc, output_bytes,
                                             ebx2, inplace_outliers, quant_radius, d_out, stream);
            break;
    }
    FZ_CUDA_CHECK(cudaGetLastError());

    if (!inplace_outliers && outlier_count > 0) {
        constexpr int kBlock = 256;
        const int grid = static_cast<int>((outlier_count + kBlock - 1) / kBlock);
        scatter_pfpl_outliers<<<grid, kBlock, 0, stream>>>(
            d_outlier_vals, d_outlier_idxs, outlier_count,
            output_bytes / sizeof(float), d_out);
        FZ_CUDA_CHECK(cudaGetLastError());
    }

    fz::backend::freeTempStorage(pool, tmp, stream);
    pool->free(d_offsets, stream);
    return output_bytes;
}

} // namespace fused
} // namespace fz
