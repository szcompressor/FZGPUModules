#ifndef _PORTABLE_MEM_CXX_SMART_PTR_H
#define _PORTABLE_MEM_CXX_SMART_PTR_H

// Jianann Tian
// 24-12-22

#include <cstring>
#include <memory>

template <typename T>
T* malloc_device(size_t const len) {
  T* __a;
  cudaMalloc(&__a, len * sizeof(T));
  cudaMemset(__a, 0, len * sizeof(T));
  return __a;
}

template <typename T>
T* malloc_host(size_t const len) {
  T* __a;
  cudaMallocHost(&__a, len * sizeof(T));
  memset(__a, 0, len * sizeof(T));
  return __a;
}

template <typename T>
T* malloc_unified(size_t const len) {
  T* __a;
  cudaMallocManaged(&__a, len * sizeof(T));
  return __a;
}


#define event_create_pair(...)                   \
  ([]() -> std::pair<cudaEvent_t, cudaEvent_t> { \
    cudaEvent_t a, b;                            \
    cudaEventCreate(&a);                         \
    cudaEventCreate(&b);                         \
    return {a, b};                               \
  })(__VA_ARGS__);
#define event_destroy_pair(a, b) \
  cudaEventDestroy(a);           \
  cudaEventDestroy(b);
#define event_recording_start(E1, STREAM) \
  cudaEventRecord(E1, (cudaStream_t)STREAM);
#define event_recording_stop(E2, STREAM)     \
  cudaEventRecord(E2, (cudaStream_t)STREAM); \
  cudaEventSynchronize(E2);
#define event_time_elapsed(start, end, p_millisec) \
  cudaEventElapsedTime(p_millisec, start, end);

#define GPU_DELETER_D GPU_deleter_device
#define GPU_DELETER_H GPU_deleter_host
#define GPU_DELETER_U GPU_deleter_unified

#define MAKE_UNIQUE_HOST(TYPE, LEN) GPU_make_unique(malloc_host<TYPE>(LEN), GPU_DELETER_H())
#define MAKE_UNIQUE_DEVICE(TYPE, LEN) GPU_make_unique(malloc_device<TYPE>(LEN), GPU_DELETER_D())
#define MAKE_UNIQUE_UNIFIED(TYPE, LEN) GPU_make_unique(malloc_unified<TYPE>(LEN), GPU_DELETER_U())

// smart pointer deleter for on-device buffer
struct GPU_deleter_device {
  void* stream;
  GPU_deleter_device(void* s = nullptr) : stream(s) {}
  void operator()(void* ptr) const
  {
    if (ptr) cudaFree(ptr);
  }
};

// smart pointer deleter for on-host pinned memory buffer
struct GPU_deleter_host {
  void* stream;
  GPU_deleter_host(void* s = nullptr) : stream(s) {}
  void operator()(void* ptr) const
  {
    if (ptr) cudaFreeHost(ptr);
  }
};

// smart pointer deleter for unifed memory
struct GPU_deleter_unified : GPU_deleter_device {
  GPU_deleter_unified(void* s = nullptr) : GPU_deleter_device(s) {}
};

// GPU unique_ptr checker (template): default to false
template <typename T>
struct is_unique_ptr : std::false_type {};

// GPU unique_ptr checker: specialization with std::true_type
template <typename T, typename Deleter>
struct is_unique_ptr<std::unique_ptr<T, Deleter>> : std::true_type {};
// GPU unique_ptr checker: ::value alias (C++17 onward)
template <typename T>
inline constexpr bool is_unique_ptr_v = is_unique_ptr<T>::value;

// GPU shared_ptr checker (template): default to false
template <typename T>
struct is_shared_ptr : std::false_type {};

// GPU shared_ptr checker: specialization with std::true_type
template <typename T>
struct is_shared_ptr<std::shared_ptr<T>> : std::true_type {};
// GPU shared_ptr checker: ::value alias (C++17 onward)
template <typename T>
inline constexpr bool is_shared_ptr_v = is_shared_ptr<T>::value;

// GPU smart pointer checker: unique_ptr or shared_ptr
template <typename T>
struct is_smart_ptr : std::bool_constant<is_unique_ptr_v<T> || is_shared_ptr_v<T>> {};

// GPU smart pointer checker: ::value alias (C++17 onward)
template <typename T>
inline constexpr bool is_smart_ptr_v = is_smart_ptr<T>::value;

template <typename T, typename Deleter>
std::unique_ptr<T[], Deleter> GPU_make_unique(T* ptr, Deleter d)
{
  return std::unique_ptr<T[], Deleter>(ptr, d);
}

template <typename T, typename Deleter>
std::shared_ptr<T[]> GPU_make_shared(T* ptr, Deleter d)
{
  return std::shared_ptr<T[]>(ptr, d);
}

template <typename T>
using GPU_unique_dptr = std::unique_ptr<T, GPU_deleter_device>;
template <typename T>
using GPU_unique_hptr = std::unique_ptr<T, GPU_deleter_host>;
template <typename T>
using GPU_unique_uptr = std::unique_ptr<T, GPU_deleter_unified>;

#endif /* _PORTABLE_MEM_CXX_SMART_PTR_H */
