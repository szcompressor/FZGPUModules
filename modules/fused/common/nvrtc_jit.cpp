// Shared NVRTC compile/cache for the runtime-generated fusion kernels. Factored out
// of nvrtc_chunk_fusion.cpp so the warp-register path reuses the exact same machinery
// (compile straight to a CUBIN for the device SM, load via the driver API, cache the
// module by (arch, source), return device functions by name). See nvrtc_jit.h.

#include "fused/common/nvrtc_jit.h"

#include <nvrtc.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace fz {
namespace fused {

[[noreturn]] void cuThrow(int result, const char* prefix, const char* what) {
    CUresult    r    = static_cast<CUresult>(result);
    const char* name = nullptr; cuGetErrorName(r, &name);
    const char* str  = nullptr; cuGetErrorString(r, &str);
    throw std::runtime_error(std::string(prefix) + ": " + what + ": " +
                             (name ? name : "?") + " (" + (str ? str : "") + ")");
}
#define CU_CHECK(call) FZ_CU_CHECK(call, "NVRTC-JIT")

namespace {

// ── Minimal C++ stdlib stubs handed to NVRTC in-memory. The op headers use only a
// tiny surface (the fixed-width int aliases + a few type traits); NVRTC ships neither
// <cstdint> nor <type_traits> on an include path, so we supply exactly what compiles.
// `assert` comes from NVRTC's builtin preinclude, so <cassert> is an empty shim.
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
template<class T> struct make_signed { using type = T; };
template<> struct make_signed<unsigned char>      { using type = signed char; };
template<> struct make_signed<char>               { using type = signed char; };
template<> struct make_signed<unsigned short>     { using type = short; };
template<> struct make_signed<unsigned int>       { using type = int; };
template<> struct make_signed<unsigned long>      { using type = long; };
template<> struct make_signed<unsigned long long> { using type = long long; };
template<class T> using make_signed_t = typename make_signed<T>::type;
}
)H";

constexpr const char* kCassert = "#pragma once\n";  // assert() is NVRTC-builtin

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

// Real device arch (sm_XX, not compute_XX) so we compile straight to a CUBIN for the
// running GPU. Loading a cubin needs no driver-side PTX JIT — which is what failed as
// CUDA_ERROR_UNSUPPORTED_PTX_VERSION when the runner's driver was older than the
// toolkit's NVRTC. SASS for the device's own SM always loads on that driver.
std::string deviceArch() {
    int dev = 0; cudaGetDevice(&dev);
    int major = 9, minor = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
    return "sm_" + std::to_string(major) + std::to_string(minor);
}

// A compiled+loaded module and its resolved functions. CUmodule is context-bound; the
// prototype assumes one context for the process lifetime.
struct CachedModule {
    CUmodule module = nullptr;
    std::unordered_map<std::string, CUfunction> funcs;
};

std::mutex                                    g_mtx;
std::unordered_map<std::string, CachedModule> g_cache;   // key: arch + "\n" + src

// Compile `src` to a module for `arch` (or return the cached one). Caller holds no
// lock; this takes g_mtx around the cache reads/writes.
CachedModule& moduleFor(const std::string& src, const std::string& arch) {
    const std::string key = arch + "\n" + src;
    {
        std::lock_guard<std::mutex> lk(g_mtx);
        auto it = g_cache.find(key);
        if (it != g_cache.end()) return it->second;
    }

    const char* hdrSrc[]  = { kCstdint,   kTypeTraits,   kCassert };
    const char* hdrName[] = { "cstdint",  "type_traits", "cassert" };

    nvrtcProgram prog = nullptr;
    if (nvrtcCreateProgram(&prog, src.c_str(), "fz_fused.cu", 3, hdrSrc, hdrName)
            != NVRTC_SUCCESS)
        throw std::runtime_error("NVRTC-JIT: nvrtcCreateProgram failed");

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
        throw std::runtime_error(std::string("NVRTC-JIT: compile failed:\n") + log);
    }

    // Emit a CUBIN for the device SM (see deviceArch) rather than PTX — no driver JIT.
    size_t cubinSize = 0; nvrtcGetCUBINSize(prog, &cubinSize);
    std::string cubin(cubinSize, '\0'); nvrtcGetCUBIN(prog, &cubin[0]);
    nvrtcDestroyProgram(&prog);

    ensureContext();
    CUmodule module = nullptr;
    CU_CHECK(cuModuleLoadData(&module, cubin.data()));

    std::lock_guard<std::mutex> lk(g_mtx);
    auto it = g_cache.find(key);
    if (it != g_cache.end()) {          // lost a race — keep the winner, drop ours
        cuModuleUnload(module);
        return it->second;
    }
    return g_cache.emplace(key, CachedModule{module, {}}).first->second;
}

} // namespace

bool nvrtcAvailable() {
    int major = 0, minor = 0;
    return nvrtcVersion(&major, &minor) == NVRTC_SUCCESS;
}

void* nvrtcGetKernel(const std::string& src, const char* entry) {
    CachedModule& mod = moduleFor(src, deviceArch());
    std::lock_guard<std::mutex> lk(g_mtx);
    auto it = mod.funcs.find(entry);
    if (it != mod.funcs.end()) return reinterpret_cast<void*>(it->second);
    CUfunction func = nullptr;
    CU_CHECK(cuModuleGetFunction(&func, mod.module, entry));
    mod.funcs.emplace(entry, func);
    return reinterpret_cast<void*>(func);
}

} // namespace fused
} // namespace fz
