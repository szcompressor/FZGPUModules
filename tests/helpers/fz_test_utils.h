#pragma once

/**
 * Shared test utilities for FZGPUModules unit tests.
 *
 * Provides:
 *   - CudaBuffer<T>   – RAII device allocation (cudaMalloc / cudaFree)
 *   - CudaStream      – RAII cudaStream_t
 *   - h2d / d2h       – convenience host↔device copy helpers
 *   - make_test_pool  – build a MemoryPool sized for a test with n bytes of data
 */

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "mem/mempool.h"

namespace fz_test {

// ─────────────────────────────────────────────────────────────────────────────
// Macro: assert a CUDA call succeeds and fail the current GTest if not.
// ─────────────────────────────────────────────────────────────────────────────
#define FZ_TEST_CUDA(expr)                                               \
    do {                                                                 \
        cudaError_t _e = (expr);                                         \
        ASSERT_EQ(_e, cudaSuccess)                                       \
            << "CUDA error in " #expr ": " << cudaGetErrorString(_e);   \
    } while (0)

// ─────────────────────────────────────────────────────────────────────────────
// RAII device buffer
// ─────────────────────────────────────────────────────────────────────────────
template <typename T>
class CudaBuffer {
public:
    explicit CudaBuffer(size_t n) : n_(n), ptr_(nullptr) {
        if (n > 0) {
            cudaError_t e = cudaMalloc(&ptr_, n * sizeof(T));
            if (e != cudaSuccess) {
                throw std::runtime_error(
                    std::string("CudaBuffer cudaMalloc failed: ") +
                    cudaGetErrorString(e));
            }
        }
    }
    ~CudaBuffer() { if (ptr_) cudaFree(ptr_); }

    // Non-copyable, movable
    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;
    CudaBuffer(CudaBuffer&& o) noexcept : n_(o.n_), ptr_(o.ptr_) {
        o.ptr_ = nullptr; o.n_ = 0;
    }

    T*     get()          const { return ptr_; }
    void*  void_ptr()     const { return ptr_; }
    size_t count()        const { return n_; }
    size_t bytes()        const { return n_ * sizeof(T); }

    // Upload a host vector
    void upload(const std::vector<T>& h, cudaStream_t s = 0) {
        if (h.size() != n_)
            throw std::runtime_error("CudaBuffer::upload size mismatch");
        cudaError_t e = cudaMemcpyAsync(ptr_, h.data(), bytes(),
                                         cudaMemcpyHostToDevice, s);
        if (e != cudaSuccess)
            throw std::runtime_error(std::string("CudaBuffer upload failed: ") +
                                     cudaGetErrorString(e));
    }

    // Download to a host vector
    std::vector<T> download(cudaStream_t s = 0) const {
        std::vector<T> h(n_);
        cudaError_t e = cudaMemcpyAsync(h.data(), ptr_, bytes(),
                                         cudaMemcpyDeviceToHost, s);
        if (e != cudaSuccess)
            throw std::runtime_error(std::string("CudaBuffer download failed: ") +
                                     cudaGetErrorString(e));
        cudaStreamSynchronize(s);
        return h;
    }

private:
    size_t n_;
    T*     ptr_;
};

// ─────────────────────────────────────────────────────────────────────────────
// RAII CUDA stream
// ─────────────────────────────────────────────────────────────────────────────
struct CudaStream {
    cudaStream_t stream = nullptr;
    CudaStream()  { cudaStreamCreate(&stream); }
    ~CudaStream() { if (stream) cudaStreamDestroy(stream); }
    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;
    operator cudaStream_t() const { return stream; }
    void sync() const { cudaStreamSynchronize(stream); }
};

// ─────────────────────────────────────────────────────────────────────────────
// Build a MemoryPool whose capacity is generous for a given data size.
// pool_multiplier defaults to 10× to comfortably cover temporaries in tests.
// ─────────────────────────────────────────────────────────────────────────────
inline std::unique_ptr<fz::MemoryPool> make_test_pool(size_t data_bytes,
                                                       float  multiplier = 10.0f) {
    fz::MemoryPoolConfig cfg(data_bytes, multiplier);
    return std::make_unique<fz::MemoryPool>(cfg);
}

} // namespace fz_test
