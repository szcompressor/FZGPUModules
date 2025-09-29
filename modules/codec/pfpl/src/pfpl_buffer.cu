#include "pfpl_buffer.hh"
#include "err.hh"
#include <cstdio>

namespace pfpl {

template <typename T>
void PFPL_Buf<T>::init(size_t data_len) {
  
  int device;
  cudaGetDevice(&device);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device);
  SMs = deviceProp.multiProcessorCount;
  mTpSM = deviceProp.maxThreadsPerMultiProcessor;
  blocks = SMs * (mTpSM / threads_per_block);

  int temp_chunks = (data_len + chunk_size - 1) / chunk_size;
  max_size = 3 * sizeof(int) + temp_chunks * sizeof(short) + temp_chunks * chunk_size;

  max_archive_bytes = sizeof(T) * data_len * 2;  // Temporary estimate

  CHECK_GPU(cudaMalloc(&d_archive, max_archive_bytes));
  CHECK_GPU(cudaMalloc(&d_comp_len, sizeof(size_t)));

  CHECK_GPU(cudaMalloc((void**)&d_fullcarry, sizeof(int) * temp_chunks));
}

template <typename T>
PFPL_Buf<T>::PFPL_Buf(size_t data_len) {
  init(data_len);
}

template <typename T>
PFPL_Buf<T>::~PFPL_Buf() {
  if (d_archive) {
    CHECK_GPU(cudaFree(d_archive));
    d_archive = nullptr;
  }
  if (d_comp_len) {
    CHECK_GPU(cudaFree(d_comp_len));
    d_comp_len = nullptr;
  }
  if (d_fullcarry) {
    CHECK_GPU(cudaFree(d_fullcarry));
    d_fullcarry = nullptr;
  }
}

template <typename T>
void PFPL_Buf<T>::clear_buffer() {
  if (d_archive) {
    CHECK_GPU(cudaMemset(d_archive, 0, max_archive_bytes));
  }
  if (d_comp_len) {
    CHECK_GPU(cudaMemset(d_comp_len, 0, sizeof(size_t)));
  }
  if (d_fullcarry) {
    CHECK_GPU(cudaMemset(d_fullcarry, 0, sizeof(int) * ((max_size + chunk_size - 1) / chunk_size)));
  }
}

template struct PFPL_Buf<uint8_t>;
template struct PFPL_Buf<uint16_t>;
template struct PFPL_Buf<uint32_t>;

}  // namespace pfpl