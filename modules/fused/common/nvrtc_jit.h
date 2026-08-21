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

} // namespace fused
} // namespace fz
