// Runtime NVRTC codegen for the chunk-cooperative fusion harness. Generates an
// `extern "C"` kernel wrapping chunk_fused_body<...> for a ChunkFusionSpec,
// compiles it with NVRTC, JIT-loads it via the CUDA driver API, caches the
// module by (source, arch), and launches it. See nvrtc_chunk_fusion.h and
// docs/codebase_notes.md CN-NVRTC-FUSE.

#include "fused/chunk_fusion/nvrtc_chunk_fusion.h"

#include <nvrtc.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace fz {
namespace fused {

namespace {

// ── Minimal C++ stdlib stubs handed to NVRTC in-memory. The op headers use only
// a tiny surface (is_integral/is_signed/is_unsigned/make_unsigned + the int
// aliases); NVRTC ships neither <cstdint> nor <type_traits> on an include path,
// so we supply exactly what compiles. `assert` comes from NVRTC's builtin
// preinclude, so <cassert> is an empty shim (avoids a redefinition warning).
constexpr const char* kCstdint = R"H(#pragma once
typedef unsigned char      uint8_t;   typedef signed char   int8_t;
typedef unsigned short     uint16_t;  typedef short         int16_t;
typedef unsigned int       uint32_t;  typedef int           int32_t;
typedef unsigned long long uint64_t;  typedef long long     int64_t;
namespace std {
  using ::uint8_t;  using ::int8_t;  using ::uint16_t; using ::int16_t;
  using ::uint32_t; using ::int32_t; using ::uint64_t; using ::int64_t;
}
)H";

constexpr const char* kTypeTraits = R"H(#pragma once
namespace std {
template<class T, T v> struct integral_constant { static constexpr T value = v; };
using true_type  = integral_constant<bool, true>;
using false_type = integral_constant<bool, false>;
template<class T> struct is_integral                 : false_type {};
template<> struct is_integral<bool>                  : true_type {};
template<> struct is_integral<char>                  : true_type {};
template<> struct is_integral<signed char>           : true_type {};
template<> struct is_integral<unsigned char>         : true_type {};
template<> struct is_integral<short>                 : true_type {};
template<> struct is_integral<unsigned short>        : true_type {};
template<> struct is_integral<int>                   : true_type {};
template<> struct is_integral<unsigned int>          : true_type {};
template<> struct is_integral<long>                  : true_type {};
template<> struct is_integral<unsigned long>         : true_type {};
template<> struct is_integral<long long>             : true_type {};
template<> struct is_integral<unsigned long long>    : true_type {};
template<class T> struct is_signed   : integral_constant<bool, (T(-1) < T(0))> {};
template<class T> struct is_unsigned : integral_constant<bool, (T(0) < T(-1))> {};
template<class T> struct make_unsigned { using type = T; };
template<> struct make_unsigned<signed char>  { using type = unsigned char; };
template<> struct make_unsigned<char>         { using type = unsigned char; };
template<> struct make_unsigned<short>        { using type = unsigned short; };
template<> struct make_unsigned<int>          { using type = unsigned int; };
template<> struct make_unsigned<long>         { using type = unsigned long; };
template<> struct make_unsigned<long long>    { using type = unsigned long long; };
template<class T> using make_unsigned_t = typename make_unsigned<T>::type;
}
)H";

constexpr const char* kCassert = "#pragma once\n";  // assert() is NVRTC-builtin

[[noreturn]] void cuThrow(CUresult r, const char* what) {
    const char* name = nullptr; cuGetErrorName(r, &name);
    const char* str  = nullptr; cuGetErrorString(r, &str);
    throw std::runtime_error(std::string("NVRTC-fusion: ") + what + ": " +
                             (name ? name : "?") + " (" + (str ? str : "") + ")");
}
#define CU_CHECK(call) do { CUresult _r = (call); if (_r != CUDA_SUCCESS) cuThrow(_r, #call); } while (0)

// One thread's current context, or the device's primary context if none is set.
CUcontext ensureContext() {
    CUcontext ctx = nullptr;
    if (cuCtxGetCurrent(&ctx) == CUDA_SUCCESS && ctx) return ctx;
    cudaFree(nullptr);                      // force the runtime to init a primary ctx
    if (cuCtxGetCurrent(&ctx) == CUDA_SUCCESS && ctx) return ctx;
    CUdevice dev; CU_CHECK(cuDeviceGet(&dev, 0));
    CU_CHECK(cuDevicePrimaryCtxRetain(&ctx, dev));
    CU_CHECK(cuCtxSetCurrent(ctx));
    return ctx;
}

std::string deviceArch() {
    int dev = 0; cudaGetDevice(&dev);
    int major = 9, minor = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
    return "compute_" + std::to_string(major) + std::to_string(minor);
}

// Compiled + loaded kernel, cached by (source, arch). CUmodule is context-bound;
// the prototype assumes one context for the process lifetime.
struct CachedKernel { CUmodule module; CUfunction func; };

std::mutex                                     g_mtx;
std::unordered_map<std::string, CachedKernel>  g_cache;

CUfunction compileAndLoad(const std::string& src, const std::string& arch) {
    const std::string key = arch + "\n" + src;
    {
        std::lock_guard<std::mutex> lk(g_mtx);
        auto it = g_cache.find(key);
        if (it != g_cache.end()) return it->second.func;
    }

    const char* hdrSrc[]  = { kCstdint,   kTypeTraits,   kCassert };
    const char* hdrName[] = { "cstdint",  "type_traits", "cassert" };

    nvrtcProgram prog = nullptr;
    if (nvrtcCreateProgram(&prog, src.c_str(), "fz_fused_chunk.cu", 3, hdrSrc, hdrName)
            != NVRTC_SUCCESS)
        throw std::runtime_error("NVRTC-fusion: nvrtcCreateProgram failed");

    const std::string archOpt = "--gpu-architecture=" + arch;
    std::vector<std::string> optStore = {
        "--std=c++17", archOpt,
        std::string("-I") + FZGMOD_NVRTC_INC_INCLUDE,
        std::string("-I") + FZGMOD_NVRTC_INC_MODULES,
    };
    std::vector<const char*> opts;
    for (auto& s : optStore) opts.push_back(s.c_str());

    nvrtcResult cr = nvrtcCompileProgram(prog, (int)opts.size(), opts.data());
    if (cr != NVRTC_SUCCESS) {
        size_t logSize = 0; nvrtcGetProgramLogSize(prog, &logSize);
        std::string log(logSize, '\0'); nvrtcGetProgramLog(prog, &log[0]);
        nvrtcDestroyProgram(&prog);
        throw std::runtime_error(std::string("NVRTC-fusion: compile failed:\n") + log);
    }

    size_t ptxSize = 0; nvrtcGetPTXSize(prog, &ptxSize);
    std::string ptx(ptxSize, '\0'); nvrtcGetPTX(prog, &ptx[0]);
    nvrtcDestroyProgram(&prog);

    ensureContext();
    CUmodule module = nullptr;
    CU_CHECK(cuModuleLoadData(&module, ptx.c_str()));
    CUfunction func = nullptr;
    CU_CHECK(cuModuleGetFunction(&func, module, "fz_fused_chunk"));

    std::lock_guard<std::mutex> lk(g_mtx);
    auto it = g_cache.find(key);
    if (it != g_cache.end()) {          // lost a race — keep the winner, drop ours
        cuModuleUnload(module);
        return it->second.func;
    }
    g_cache.emplace(key, CachedKernel{module, func});
    return func;
}

} // namespace

std::string generateChunkFusionSource(const ChunkFusionSpec& spec) {
    // Template args: <QuantOp, Coder, Transforms...>. This is the "connecting
    // code" — the only thing that changes per pipeline; the ops are #included.
    std::string targs = spec.quant_op + ", " + spec.coder;
    for (const auto& t : spec.transforms) targs += ", " + t;

    std::string src;
    src += "#include \"fused/chunk_fusion/chunk_fusion.cuh\"\n";
    src += "extern \"C\" __global__ void __launch_bounds__(" +
           std::to_string(chunk::TPB) + ") fz_fused_chunk(\n";
    src += "    const float* in, unsigned long long n, float ebx2_r,\n";
    src += "    unsigned int radius, float threshold,\n";
    src += "    unsigned char* scratch, unsigned int* sizes) {\n";
    src += "  using namespace fz::fused::chunk;\n";
    src += "  " + spec.quant_op + "::Params qp{ ebx2_r, radius, threshold };\n";
    src += "  chunk_fused_body< " + targs + " >(\n";
    src += "      in, (size_t)n, qp, scratch, sizes);\n";
    src += "}\n";
    return src;
}

bool nvrtcChunkFusionAvailable() {
    int major = 0, minor = 0;
    return nvrtcVersion(&major, &minor) == NVRTC_SUCCESS;
}

void launchNvrtcChunkFusedEncode(
    const ChunkFusionSpec& spec, const float* d_in, size_t n,
    float ebx2_r, uint32_t radius, float threshold,
    uint8_t* d_scratch, uint32_t* d_sizes, unsigned nc, fz::stream_t stream)
{
    const std::string src  = generateChunkFusionSource(spec);
    CUfunction        func = compileAndLoad(src, deviceArch());

    unsigned long long n_arg = n;
    void* args[] = { (void*)&d_in, (void*)&n_arg, (void*)&ebx2_r,
                     (void*)&radius, (void*)&threshold,
                     (void*)&d_scratch, (void*)&d_sizes };

    CU_CHECK(cuLaunchKernel(func,
                            nc, 1, 1,               // grid  = one CTA per chunk
                            (unsigned)chunk::TPB, 1, 1,   // block = TPB
                            0,                       // static smem only (36 KB)
                            (CUstream)stream, args, nullptr));
}

} // namespace fused
} // namespace fz
