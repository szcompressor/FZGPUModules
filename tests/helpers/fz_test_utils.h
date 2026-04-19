#pragma once

/**
 * Shared test utilities for FZGPUModules unit tests.
 *
 * Provides:
 *   - CudaBuffer<T>        – RAII device allocation (cudaMalloc / cudaFree)
 *   - CudaStream           – RAII cudaStream_t
 *   - h2d / d2h            – convenience host↔device copy helpers
 *   - make_test_pool       – build a MemoryPool sized for a test with n bytes of data
 *   - make_random_floats   – reproducible seeded random float data
 *   - max_abs_error        – max absolute element-wise difference between two vectors
 *   - gpu_free_bytes       – current GPU free memory (for leak checks)
 */

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <fstream>
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

    // Download the full buffer to a host vector.
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

    // Download only the first byte_count bytes.  Returns a vector<T> of
    // byte_count/sizeof(T) elements.  Use this when the stage only wrote
    // byte_count bytes into a larger allocated buffer so that initcheck does
    // not flag reads of the unwritten trailing bytes.
    std::vector<T> download_bytes(size_t byte_count, cudaStream_t s = 0) const {
        size_t elem_count = byte_count / sizeof(T);
        std::vector<T> h(elem_count);
        if (elem_count == 0) return h;
        cudaError_t e = cudaMemcpyAsync(h.data(), ptr_, byte_count,
                                         cudaMemcpyDeviceToHost, s);
        if (e != cudaSuccess)
            throw std::runtime_error(std::string("CudaBuffer download_bytes failed: ") +
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

// ─────────────────────────────────────────────────────────────────────────────
// Generate a vector of N floats in [-1, 1] using a deterministic seed so
// tests are reproducible across runs.
// ─────────────────────────────────────────────────────────────────────────────
inline std::vector<float> make_random_floats(size_t n, uint64_t seed = 42) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> v(n);
    for (auto& x : v) x = dist(rng);
    return v;
}

// ─────────────────────────────────────────────────────────────────────────────
// Generate a vector of N doubles in [-1, 1] with a deterministic seed.
// ─────────────────────────────────────────────────────────────────────────────
inline std::vector<double> make_random_doubles(size_t n, uint64_t seed = 42) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<double> v(n);
    for (auto& x : v) x = dist(rng);
    return v;
}

// ─────────────────────────────────────────────────────────────────────────────
// Smooth sinusoidal float vector.
//   freq      — angular frequency increment per element (radians per step)
//   amplitude — peak value
// ─────────────────────────────────────────────────────────────────────────────
inline std::vector<float> make_sine_floats(size_t n,
                                            float  freq      = 0.01f,
                                            float  amplitude = 1.0f) {
    std::vector<float> v(n);
    for (size_t i = 0; i < n; i++)
        v[i] = amplitude * std::sin(static_cast<float>(i) * freq);
    return v;
}

// ─────────────────────────────────────────────────────────────────────────────
// Generic smooth data generator for type-parametric tests.
// Produces a two-component sinusoid scaled to amp1 and amp2, suitable as
// input to lossy predictors (Lorenzo, Quantizer) for any floating-point T.
// ─────────────────────────────────────────────────────────────────────────────
template <typename T>
inline std::vector<T> make_smooth_data(size_t n,
                                        double freq1 = 0.01,
                                        double freq2 = 0.003,
                                        double amp1  = 50.0,
                                        double amp2  = 20.0) {
    std::vector<T> v(n);
    for (size_t i = 0; i < n; i++) {
        double x = amp1 * std::sin(static_cast<double>(i) * freq1)
                 + amp2 * std::cos(static_cast<double>(i) * freq2);
        v[i] = static_cast<T>(x);
    }
    return v;
}

// ─────────────────────────────────────────────────────────────────────────────
// Linear ramp: v[i] = i * scale.  Scale defaults to 1.0.
// ─────────────────────────────────────────────────────────────────────────────
template <typename T = float>
inline std::vector<T> make_ramp(size_t n, T scale = T{1}) {
    std::vector<T> v(n);
    for (size_t i = 0; i < n; i++)
        v[i] = static_cast<T>(i) * scale;
    return v;
}

// ─────────────────────────────────────────────────────────────────────────────
// Constant-fill: every element is val.
// ─────────────────────────────────────────────────────────────────────────────
template <typename T = float>
inline std::vector<T> make_constant(size_t n, T val) {
    return std::vector<T>(n, val);
}

// ─────────────────────────────────────────────────────────────────────────────
// Max absolute element-wise difference.
// Asserts inside GTest if sizes differ.
// ─────────────────────────────────────────────────────────────────────────────
inline float max_abs_error(const std::vector<float>& orig,
                            const std::vector<float>& recon) {
    EXPECT_EQ(orig.size(), recon.size()) << "max_abs_error: size mismatch";
    float err = 0.0f;
    size_t n = std::min(orig.size(), recon.size());
    for (size_t i = 0; i < n; i++)
        err = std::max(err, std::abs(recon[i] - orig[i]));
    return err;
}

// ─────────────────────────────────────────────────────────────────────────────
// Query how many bytes are free on the current CUDA device.
// Useful as a before/after check in memory-leak tests.
// ─────────────────────────────────────────────────────────────────────────────
inline size_t gpu_free_bytes() {
    size_t free_bytes = 0, total_bytes = 0;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    return free_bytes;
}

// ─────────────────────────────────────────────────────────────────────────────
// Overwrite a single byte in an existing file at byte_offset.
// Used to corrupt magic numbers, version fields, etc. in FZM files.
// ─────────────────────────────────────────────────────────────────────────────
inline void make_corrupt_file(const std::string& path,
                               size_t             byte_offset,
                               uint8_t            new_value) {
    std::fstream f(path, std::ios::in | std::ios::out | std::ios::binary);
    if (!f)
        throw std::runtime_error("make_corrupt_file: cannot open " + path);
    f.seekp(static_cast<std::streamoff>(byte_offset));
    f.write(reinterpret_cast<const char*>(&new_value), 1);
}

// ─────────────────────────────────────────────────────────────────────────────
// Replace a file with a stub containing only n_bytes of zero bytes.
// Used to simulate truncated or partially-written files.
// ─────────────────────────────────────────────────────────────────────────────
inline void write_stub_file(const std::string& path, size_t n_bytes) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f)
        throw std::runtime_error("write_stub_file: cannot open " + path);
    std::vector<char> buf(n_bytes, '\0');
    f.write(buf.data(), static_cast<std::streamsize>(n_bytes));
}

} // namespace fz_test
