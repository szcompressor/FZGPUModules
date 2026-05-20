/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Adapted for FZGPUModules: namespace renamed from multibyte_ans to fz::ans;
 * removed profilerStart/profilerStop (cudaProfilerStart/Stop deprecated in CUDA 12).
 */
#ifndef FZ_ANS_DIETGPU_UTILS_DEVICEUTILS_H
#define FZ_ANS_DIETGPU_UTILS_DEVICEUTILS_H

#pragma once
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <vector>
#include <mutex>
#include <unordered_map>

#define CUDA_VERIFY(X)                                           \
  do {                                                           \
    auto err__ = (X);                                            \
    if (err__ != cudaSuccess) {                                  \
      std::cout << "CUDA error " << fz::ans::errorToName(err__) \
                << " " << fz::ans::errorToString(err__)          \
                << std::endl;                                    \
    }                                                            \
  } while (0)

#ifdef GPU_SYNC_ERROR
#define CUDA_TEST_ERROR()                 \
  do {                                    \
    CUDA_VERIFY(cudaDeviceSynchronize()); \
  } while (0)
#else
#define CUDA_TEST_ERROR()            \
  do {                               \
    CUDA_VERIFY(cudaGetLastError()); \
  } while (0)
#endif

namespace fz { namespace ans {

constexpr int kWarpSize = 32;

inline std::string errorToString(cudaError_t err) {
  return std::string(cudaGetErrorString(err));
}

inline std::string errorToName(cudaError_t err) {
  return std::string(cudaGetErrorName(err));
}

inline int getCurrentDevice() {
  int dev = -1;
  CUDA_VERIFY(cudaGetDevice(&dev));
  return dev;
}

inline void setCurrentDevice(int device) {
  CUDA_VERIFY(cudaSetDevice(device));
}

inline int getNumDevices() {
  int numDev = -1;
  cudaError_t err = cudaGetDeviceCount(&numDev);
  if (cudaErrorNoDevice == err) {
    numDev = 0;
  } else {
    CUDA_VERIFY(err);
  }
  return numDev;
}

inline void synchronizeAllDevices() {
  for (int i = 0; i < getNumDevices(); ++i) {
    CUDA_VERIFY(cudaSetDevice(i));
    CUDA_VERIFY(cudaDeviceSynchronize());
  }
}

inline const cudaDeviceProp& getDeviceProperties(int device) {
  static std::mutex mutex;
  static std::unordered_map<int, cudaDeviceProp> properties;

  std::lock_guard<std::mutex> guard(mutex);

  auto it = properties.find(device);
  if (it == properties.end()) {
    cudaDeviceProp prop;
    CUDA_VERIFY(cudaGetDeviceProperties(&prop, device));
    properties[device] = prop;
    it = properties.find(device);
  }
  return it->second;
}

inline const cudaDeviceProp& getCurrentDeviceProperties() {
  return getDeviceProperties(getCurrentDevice());
}

inline int getMaxThreads(int device) {
  return getDeviceProperties(device).maxThreadsPerBlock;
}

inline int getMaxThreadsCurrentDevice() {
  return getMaxThreads(getCurrentDevice());
}

inline size_t getMaxSharedMemPerBlock(int device) {
  return getDeviceProperties(device).sharedMemPerBlock;
}

inline size_t getMaxSharedMemPerBlockCurrentDevice() {
  return getMaxSharedMemPerBlock(getCurrentDevice());
}

inline int getDeviceForAddress(const void* p) {
  if (!p) {
    return -1;
  }
  cudaPointerAttributes att;
  cudaError_t err = cudaPointerGetAttributes(&att, p);
  if (err == cudaErrorInvalidValue) {
    err = cudaGetLastError();
    return -1;
  }
#if CUDA_VERSION < 10000
  if (att.memoryType == cudaMemoryTypeHost) {
    return -1;
  } else {
    return att.device;
  }
#else
  if (att.type == cudaMemoryTypeDevice) {
    return att.device;
  } else {
    return -1;
  }
#endif
}

inline bool getFullUnifiedMemSupport(int device) {
  const auto& prop = getDeviceProperties(device);
  return (prop.major >= 6);
}

inline bool getFullUnifiedMemSupportCurrentDevice() {
  return getFullUnifiedMemSupport(getCurrentDevice());
}

class DeviceScope {
 public:
  explicit DeviceScope(int device) {
    if (device >= 0) {
      int curDevice = getCurrentDevice();
      if (curDevice != device) {
        prevDevice_ = curDevice;
        setCurrentDevice(device);
        return;
      }
    }
    prevDevice_ = -1;
  }
  ~DeviceScope() {
    if (prevDevice_ != -1) {
      setCurrentDevice(prevDevice_);
    }
  }
 private:
  int prevDevice_;
};

class CudaEvent {
 public:
  explicit CudaEvent(cudaStream_t stream, bool timer = false) : event_(nullptr) {
    CUDA_VERIFY(cudaEventCreateWithFlags(
        &event_, timer ? cudaEventDefault : cudaEventDisableTiming));
    CUDA_VERIFY(cudaEventRecord(event_, stream));
  }
  CudaEvent(const CudaEvent&) = delete;
  CudaEvent(CudaEvent&& event) noexcept : event_(event.event_) {
    event.event_ = nullptr;
  }
  ~CudaEvent() {
    if (event_) CUDA_VERIFY(cudaEventDestroy(event_));
  }
  CudaEvent& operator=(CudaEvent&& event) noexcept {
    event_ = event.event_;
    event.event_ = nullptr;
    return *this;
  }
  CudaEvent& operator=(CudaEvent&) = delete;
  inline cudaEvent_t get() { return event_; }
  void streamWaitOnEvent(cudaStream_t stream) {
    CUDA_VERIFY(cudaStreamWaitEvent(stream, event_, 0));
  }
  void cpuWaitOnEvent() { CUDA_VERIFY(cudaEventSynchronize(event_)); }
  float timeFrom(CudaEvent& from) {
    cpuWaitOnEvent();
    float ms = 0;
    CUDA_VERIFY(cudaEventElapsedTime(&ms, from.event_, event_));
    return ms;
  }
 private:
  cudaEvent_t event_;
};

class CudaStream {
 public:
  explicit CudaStream(int flags = cudaStreamDefault) : stream_(nullptr) {
    CUDA_VERIFY(cudaStreamCreateWithFlags(&stream_, flags));
  }
  CudaStream(const CudaStream&) = delete;
  CudaStream(CudaStream&& stream) noexcept : stream_(stream.stream_) {
    stream.stream_ = nullptr;
  }
  ~CudaStream() {
    if (stream_) CUDA_VERIFY(cudaStreamDestroy(stream_));
  }
  CudaStream& operator=(CudaStream&& stream) noexcept {
    stream_ = stream.stream_;
    stream.stream_ = nullptr;
    return *this;
  }
  CudaStream& operator=(CudaStream&) = delete;
  inline cudaStream_t get() { return stream_; }
  operator cudaStream_t() { return stream_; }
  static CudaStream make() { return CudaStream(); }
  static CudaStream makeNonBlocking() { return CudaStream(cudaStreamNonBlocking); }
 private:
  cudaStream_t stream_;
};

}} // namespace fz::ans

#endif // FZ_ANS_DIETGPU_UTILS_DEVICEUTILS_H
