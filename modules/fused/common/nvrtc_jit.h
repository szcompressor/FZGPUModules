#pragma once

/**
 * @file modules/fused/common/nvrtc_jit.h
 * @brief Shared NVRTC compile/cache used by every runtime-generated fusion kernel.
 *
 * Both fusion strategies generate CUDA source at runtime and JIT it: the chunk-
 * cooperative path (one entry point) and the warp-register path (two entries in one
 * module). This is the common machinery — compile a source to a CUBIN for the
 * device's real SM (no driver-side PTX JIT), load it, and hand back a named device
 * function, caching the compiled module by (arch, source) so only the first compile
 * of a given kernel pays the cost. See nvrtc_chunk_fusion.cpp / nvrtc_warp_fusion.cpp.
 */

#include <string>

namespace fz {
namespace fused {

/// True if NVRTC + the CUDA driver are usable in this process.
bool nvrtcAvailable();

/// Compile `src` (cached by device arch + source text) and return the device
/// function named `entry` from the resulting module. Several entries in the same
/// source share one compiled module (compiled once, looked up per entry). Returned
/// as `void*` so the header stays free of the CUDA driver headers; callers that
/// launch cast it back to `CUfunction`. Throws std::runtime_error on failure.
void* nvrtcGetKernel(const std::string& src, const char* entry);

/**
 * Shared CUDA-driver error-to-exception helper for JIT call sites. `prefix`
 * labels which JIT path failed (e.g. "NVRTC-JIT", "NVRTC-fusion", "NVRTC-warp");
 * `result` is a `CUresult` passed as `int` so this header doesn't need
 * `<cuda.h>`. Use via `FZ_CU_CHECK` below rather than calling directly.
 */
[[noreturn]] void cuThrow(int result, const char* prefix, const char* what);

/// Checks a CUDA-driver call and throws via cuThrow() on failure. Requires
/// `<cuda.h>` already included at the call site (for `CUresult`/`CUDA_SUCCESS`).
#define FZ_CU_CHECK(call, prefix) \
    do { \
        CUresult _fz_cu_r = (call); \
        if (_fz_cu_r != CUDA_SUCCESS) ::fz::fused::cuThrow((int)_fz_cu_r, prefix, #call); \
    } while (0)

} // namespace fused
} // namespace fz
