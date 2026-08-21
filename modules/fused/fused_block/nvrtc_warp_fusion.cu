// Runtime NVRTC codegen + generic launcher for the warp-register fusion harness.
// Generates two `extern "C"` kernels wrapping fused_rate_body / fused_pack_body for a
// WarpFusionSpec, JITs them through the shared NVRTC cache, and runs them with the
// host-side CUB exclusive-scan in between (which is why this is a .cu — cub is
// host-orchestrated device code nvcc must compile). See nvrtc_warp_fusion.h.

#include "fused/fused_block/nvrtc_warp_fusion.h"
#include "fused/common/nvrtc_jit.h"
#include "coders/adaptive_bitpack/adaptive_bitpack_kernels.h"   // ab::configure (host)
#include "mem/mempool.h"
#include "cuda_check.h"

#include "backend/cub.h"
#include "backend/algorithms.h"
#include <cub/device/device_scan.cuh>

#include <cuda.h>
#include <stdexcept>
#include <string>

namespace fz {
namespace fused {

namespace ab = fz::adaptive_bitpack;

namespace {
[[noreturn]] void cuThrow(CUresult r, const char* what) {
    const char* name = nullptr; cuGetErrorName(r, &name);
    const char* str  = nullptr; cuGetErrorString(r, &str);
    throw std::runtime_error(std::string("NVRTC-warp: ") + what + ": " +
                             (name ? name : "?") + " (" + (str ? str : "") + ")");
}
#define CU_CHECK(call) do { CUresult _r = (call); if (_r != CUDA_SUCCESS) cuThrow(_r, #call); } while (0)
} // namespace

std::string generateWarpFusionSource(const WarpFusionSpec& spec) {
    // The only things that change per chain: the predictor policy type and EPL (a
    // compile-time template arg). Both kernels reconstruct the policy from the params
    // blob + the launch-time input via the predictor's own `fromParams` factory.
    const std::string P   = spec.predictor;
    const std::string EPL = std::to_string(spec.elems_per_lane);
    std::string src;
    src += "#include \"fused/fused_block/warp_fusion.cuh\"\n";
    src += "using namespace fz::fused::warp;\n";
    src += "extern \"C\" __global__ void fz_fused_warp_rate(\n";
    src += "    const float* in, unsigned long long n, const unsigned char* pp,\n";
    src += "    unsigned word_bytes, unsigned long long num_blocks,\n";
    src += "    unsigned char* meta, unsigned* cost) {\n";
    src += "  " + P + " pred = " + P + "::fromParams(in, (size_t)n, pp);\n";
    src += "  fused_rate_body<" + EPL + ">(pred, (size_t)n, word_bytes, (size_t)num_blocks, meta, cost);\n";
    src += "}\n";
    src += "extern \"C\" __global__ void fz_fused_warp_pack(\n";
    src += "    const float* in, unsigned long long n, const unsigned char* pp,\n";
    src += "    unsigned word_bytes, unsigned long long num_blocks,\n";
    src += "    const unsigned char* meta, const unsigned* offset, unsigned char* payload) {\n";
    src += "  " + P + " pred = " + P + "::fromParams(in, (size_t)n, pp);\n";
    src += "  fused_pack_body<" + EPL + ">(pred, (size_t)n, word_bytes, (size_t)num_blocks, meta, offset, payload);\n";
    src += "}\n";
    return src;
}

size_t launchNvrtcWarpFused(
    const WarpFusionSpec& spec, const float* d_in, size_t n_ab,
    const uint8_t* pred_params, size_t params_bytes,
    uint8_t* d_out, MemoryPool* pool, cudaStream_t stream)
{
    if (n_ab == 0) return 0;
    const uint32_t block_size = 32u * static_cast<uint32_t>(spec.elems_per_lane);
    const ab::Config cfg = ab::configure(n_ab, block_size, /*outlier=*/true);
    const size_t num_blocks = cfg.num_blocks;
    if (num_blocks == 0) return 0;
    const size_t meta_region = 2u * num_blocks;   // outlier meta_bytes == 2

    auto* d_cost   = static_cast<uint32_t*>(pool->allocate(sizeof(uint32_t)*num_blocks, stream, "warp_cost"));
    auto* d_offset = static_cast<uint32_t*>(pool->allocate(sizeof(uint32_t)*num_blocks, stream, "warp_offset"));
    const size_t pbytes = params_bytes ? params_bytes : 1;
    auto* d_params = static_cast<uint8_t*>(pool->allocate(pbytes, stream, "warp_params"));
    if (params_bytes)
        FZ_CUDA_CHECK(cudaMemcpyAsync(d_params, pred_params, params_bytes,
                                      cudaMemcpyHostToDevice, stream));

    uint8_t* d_meta    = d_out;
    uint8_t* d_payload = d_out + meta_region;

    const std::string src  = generateWarpFusionSource(spec);
    CUfunction rate = reinterpret_cast<CUfunction>(nvrtcGetKernel(src, "fz_fused_warp_rate"));
    CUfunction pack = reinterpret_cast<CUfunction>(nvrtcGetKernel(src, "fz_fused_warp_pack"));

    const int WPB = 8, THREADS = WPB * 32;
    const unsigned grid = static_cast<unsigned>((num_blocks + WPB - 1) / WPB);
    unsigned long long n_arg = n_ab, nb_arg = num_blocks;
    unsigned wb_arg = cfg.word_bytes;

    void* rate_args[] = { (void*)&d_in, (void*)&n_arg, (void*)&d_params,
                          (void*)&wb_arg, (void*)&nb_arg, (void*)&d_meta, (void*)&d_cost };
    CU_CHECK(cuLaunchKernel(rate, grid,1,1, (unsigned)THREADS,1,1, 0,
                            (CUstream)stream, rate_args, nullptr));

    auto d_tmp = fz::backend::withTempStorage(pool, stream, "warp_cub",
        [&](void* tmp, size_t& bytes) {
            cub::DeviceScan::ExclusiveSum(tmp, bytes, d_cost, d_offset, num_blocks, stream);
        });

    void* pack_args[] = { (void*)&d_in, (void*)&n_arg, (void*)&d_params,
                          (void*)&wb_arg, (void*)&nb_arg, (void*)&d_meta,
                          (void*)&d_offset, (void*)&d_payload };
    CU_CHECK(cuLaunchKernel(pack, grid,1,1, (unsigned)THREADS,1,1, 0,
                            (CUstream)stream, pack_args, nullptr));

    uint32_t h_off = 0, h_cost = 0;
    FZ_CUDA_CHECK(cudaMemcpyAsync(&h_off,  d_offset + num_blocks-1, 4, cudaMemcpyDeviceToHost, stream));
    FZ_CUDA_CHECK(cudaMemcpyAsync(&h_cost, d_cost   + num_blocks-1, 4, cudaMemcpyDeviceToHost, stream));
    FZ_CUDA_CHECK(cudaStreamSynchronize(stream));

    fz::backend::freeTempStorage(pool, d_tmp, stream);
    pool->free(d_params, stream);
    pool->free(d_offset, stream);
    pool->free(d_cost, stream);
    return meta_region + static_cast<size_t>(h_off) + h_cost;
}

} // namespace fused
} // namespace fz
