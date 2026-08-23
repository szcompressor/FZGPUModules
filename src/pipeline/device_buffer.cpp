/**
 * @file src/pipeline/device_buffer.cpp
 * @brief Backend-correct deallocation for fz::OwnedDeviceBuffer.
 */
#include "pipeline/device_buffer.h"

#include "backend/api.h"
#include "log.h"

namespace fz {
namespace detail {

void freeDeviceBuffer(void* ptr, int device) noexcept {
    if (!ptr) return;

    // Free on the device the allocation came from: a caller may have switched
    // devices between decompressOwned() and the buffer going out of scope.
    int prev = 0;
    bool switched = false;
    if (cudaGetDevice(&prev) == cudaSuccess && prev != device) {
        switched = (cudaSetDevice(device) == cudaSuccess);
    }

    cudaError_t err = cudaFree(ptr);
    if (err != cudaSuccess) {
        // Destructor context — report, never throw.
        FZ_LOG(WARN, "OwnedDeviceBuffer: free failed: %s", cudaGetErrorString(err));
    }

    if (switched) cudaSetDevice(prev);
}

}  // namespace detail
}  // namespace fz
