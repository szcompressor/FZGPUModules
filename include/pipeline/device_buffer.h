/**
 * @file include/pipeline/device_buffer.h
 * @brief Backend-neutral device span and buffer value types.
 *
 * These types make ownership part of the *type* rather than a documented
 * property of a `void*` plus mutable pipeline state. They are the vocabulary
 * of the explicit execution API on fz::Pipeline
 * (`compress()`/`compressInto()`/`decompressBorrowed()`/`decompressOwned()`/
 * `decompressInto()`), which wraps the historical pointer overloads without
 * changing their behavior.
 *
 * Nothing here allocates. The only owning type, OwnedDeviceBuffer, frees
 * through the backend facade in a .cpp, so this header pulls in no CUDA/HIP
 * headers and works unchanged in a HIP build (it never assumes `cudaFree`).
 */
#pragma once

#include <cstddef>
#include <utility>

namespace fz {

/** Mutable, non-owning view of device memory. */
struct DeviceSpan {
    void*  data  = nullptr;
    size_t bytes = 0;

    DeviceSpan() = default;
    DeviceSpan(void* d, size_t n) : data(d), bytes(n) {}

    bool empty() const { return bytes == 0; }
};

/** Read-only, non-owning view of device memory. */
struct ConstDeviceSpan {
    const void* data  = nullptr;
    size_t      bytes = 0;

    ConstDeviceSpan() = default;
    ConstDeviceSpan(const void* d, size_t n) : data(d), bytes(n) {}
    ConstDeviceSpan(const DeviceSpan& s) : data(s.data), bytes(s.bytes) {}

    bool empty() const { return bytes == 0; }
};

/**
 * A device pointer the Pipeline still owns.
 *
 * Returned by the borrowing execution calls. The caller must NOT free it, and
 * it is invalidated by the next call that reuses the same pool slot (see the
 * Memory Ownership table in docs/architecture.md). Deliberately has no
 * conversion to OwnedDeviceBuffer.
 */
class BorrowedDeviceBuffer {
public:
    BorrowedDeviceBuffer() = default;
    BorrowedDeviceBuffer(void* d, size_t n) : data_(d), bytes_(n) {}

    void*  data()  const { return data_; }
    size_t bytes() const { return bytes_; }
    bool   empty() const { return bytes_ == 0; }

    DeviceSpan      span()  const { return DeviceSpan(data_, bytes_); }
    ConstDeviceSpan cspan() const { return ConstDeviceSpan(data_, bytes_); }

private:
    void*  data_  = nullptr;
    size_t bytes_ = 0;
};

namespace detail {
/**
 * Free a device allocation on `device`, using whichever backend the library
 * was built against. Never throws; a failure is logged and swallowed because
 * it runs from a destructor.
 */
void freeDeviceBuffer(void* ptr, int device) noexcept;
}  // namespace detail

/**
 * A device allocation the caller owns. Move-only; frees on destruction using
 * the backend and device recorded at allocation time.
 */
class OwnedDeviceBuffer {
public:
    OwnedDeviceBuffer() = default;
    OwnedDeviceBuffer(void* d, size_t n, int device)
        : data_(d), bytes_(n), device_(device) {}

    OwnedDeviceBuffer(const OwnedDeviceBuffer&)            = delete;
    OwnedDeviceBuffer& operator=(const OwnedDeviceBuffer&) = delete;

    OwnedDeviceBuffer(OwnedDeviceBuffer&& other) noexcept
        : data_(other.data_), bytes_(other.bytes_), device_(other.device_) {
        other.data_  = nullptr;
        other.bytes_ = 0;
    }

    OwnedDeviceBuffer& operator=(OwnedDeviceBuffer&& other) noexcept {
        if (this != &other) {
            reset();
            data_        = other.data_;
            bytes_       = other.bytes_;
            device_      = other.device_;
            other.data_  = nullptr;
            other.bytes_ = 0;
        }
        return *this;
    }

    ~OwnedDeviceBuffer() { reset(); }

    void*  data()   const { return data_; }
    size_t bytes()  const { return bytes_; }
    int    device() const { return device_; }
    bool   empty()  const { return bytes_ == 0; }
    explicit operator bool() const { return data_ != nullptr; }

    DeviceSpan      span()  const { return DeviceSpan(data_, bytes_); }
    ConstDeviceSpan cspan() const { return ConstDeviceSpan(data_, bytes_); }

    /** Relinquish ownership; the caller becomes responsible for freeing. */
    void* release() {
        void* p = data_;
        data_   = nullptr;
        bytes_  = 0;
        return p;
    }

    /** Free the held allocation (if any) and become empty. */
    void reset() {
        if (data_) detail::freeDeviceBuffer(data_, device_);
        data_  = nullptr;
        bytes_ = 0;
    }

private:
    void*  data_   = nullptr;
    size_t bytes_  = 0;
    int    device_ = 0;
};

}  // namespace fz
