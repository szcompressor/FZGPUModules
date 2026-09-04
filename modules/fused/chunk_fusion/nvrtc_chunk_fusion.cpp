// Runtime NVRTC codegen for the chunk-cooperative fusion harness. Generates an
// `extern "C"` kernel wrapping chunk_fused_body<...> for a ChunkFusionSpec, then
// compiles+loads+launches it through the shared NVRTC JIT (nvrtc_jit.h). See
// nvrtc_chunk_fusion.h and docs/codebase_notes.md CN-NVRTC-FUSE.

#include "fused/chunk_fusion/nvrtc_chunk_fusion.h"
#include "fused/common/nvrtc_jit.h"

#include <cuda.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace fz {
namespace fused {

namespace {
#define CU_CHECK(call) FZ_CU_CHECK(call, "NVRTC-fusion")
} // namespace

std::string generateChunkFusionSource(const ChunkFusionSpec& spec) {
    // Template args: <ChunkBytes, QuantOp, Coder<ChunkBytes>, Transforms...>.
    // This is the "connecting code" — the only thing that changes per pipeline;
    // the ops are #included. Ops whose geometry depends on chunk size (the coder,
    // and Bitshuffle32's plane-stride math) are themselves templated on it; ops
    // that only ever touch a runtime element count (quantizers, difference) are
    // not, so only those two get the `<chunk_bytes>` suffix.
    const std::string cb = std::to_string(spec.chunk_bytes);
    std::string targs = cb + ", " + spec.quant_op + ", " + spec.coder + "<" + cb + ">";
    for (const auto& t : spec.transforms) {
        targs += ", " + t;
        if (t == "Bitshuffle32") targs += "<" + cb + ">";
    }

    std::string src;
    src += "#include \"fused/chunk_fusion/chunk_fusion.cuh\"\n";
    src += "extern \"C\" __global__ void __launch_bounds__(" +
           std::to_string(chunk::TPB) + ") fz_fused_chunk(\n";
    src += "    const float* in, unsigned long long n, const unsigned char* params,\n";
    src += "    unsigned char* scratch, unsigned int* sizes,\n";
    src += "    unsigned int* side_idxs, float* side_vals, unsigned int* side_count,\n";
    src += "    unsigned int side_max) {\n";
    src += "  using namespace fz::fused::chunk;\n";
    src += "  chunk_fused_body< " + targs + " >(\n";
    src += "      in, (size_t)n, params, scratch, sizes,\n";
    src += "      ChunkSideCtx{side_idxs, side_vals, side_count, side_max});\n";
    src += "}\n";
    return src;
}

bool nvrtcChunkFusionAvailable() { return nvrtcAvailable(); }

void launchNvrtcChunkFusedEncode(
    const ChunkFusionSpec& spec, const float* d_in, size_t n, const uint8_t* d_params,
    uint8_t* d_scratch, uint32_t* d_sizes, unsigned nc, fz::stream_t stream,
    uint32_t* d_side_idxs, float* d_side_vals, uint32_t* d_side_count, uint32_t side_max)
{
    const std::string src  = generateChunkFusionSource(spec);
    CUfunction        func = reinterpret_cast<CUfunction>(nvrtcGetKernel(src, "fz_fused_chunk"));

    unsigned long long n_arg = n;
    void* args[] = { (void*)&d_in, (void*)&n_arg, (void*)&d_params,
                     (void*)&d_scratch, (void*)&d_sizes,
                     (void*)&d_side_idxs, (void*)&d_side_vals,
                     (void*)&d_side_count, (void*)&side_max };

    CU_CHECK(cuLaunchKernel(func,
                            nc, 1, 1,                     // grid  = one CTA per chunk
                            (unsigned)chunk::TPB, 1, 1,   // block = TPB
                            0,                            // static smem only (36 KB)
                            (CUstream)stream, args, nullptr));
}

} // namespace fused
} // namespace fz
