#include <cuda_runtime.h>

#include <stdexcept>

#include "err.hh"
#include "mem/cxx_smart_ptr.h"
#include "pfpl.hh"

namespace pfpl {
template <typename T>
void GPU_PFPL_encode(const T*, size_t, uint8_t*, size_t*, int*, int,
                     cudaStream_t);
template <typename T>
void GPU_PFPL_decode(const uint8_t*, size_t, T*, int*, int, cudaStream_t);
}

namespace fz {

template <typename T>
struct PFPL_Codec<T>::impl {
  pfpl::PFPL_Buf<T>* buf;

  explicit impl(size_t data_len, bool comp = true) : buf(new pfpl::PFPL_Buf<T>(data_len, comp)) {}
  ~impl() { delete buf; }

  impl(const impl&) = delete;
  impl& operator=(const impl&) = delete;
  impl(impl&&) noexcept = delete;
  impl& operator=(impl&&) noexcept = delete;

  void encode(T* in_data, size_t num_codes, uint8_t** out_data, size_t* out_len, cudaStream_t stream) {

    printf("Starting PFPL encoding for %zu codes...\n", num_codes);
    
    pfpl::GPU_PFPL_encode<T>(in_data, num_codes, buf->d_archive,
                             buf->d_comp_len, buf->d_fullcarry, buf->blocks, stream);

    printf("PFPL encoding completed.\n");
    
    *out_data = buf->d_archive;
    size_t pfpl_comp_len;
    CHECK_GPU(cudaMemcpy(&pfpl_comp_len, buf->d_comp_len, sizeof(size_t), cudaMemcpyDeviceToHost));
    printf("PFPL compressed length retrieved: %zu bytes\n", pfpl_comp_len);

    // align pfpl_comp_len to 8 byte alignment
    size_t aligned_size = (pfpl_comp_len + 7) & ~7;

    if (aligned_size > pfpl_comp_len) {
      size_t padding = aligned_size - pfpl_comp_len;
      cudaMemsetAsync((uint8_t*)(*out_data) + pfpl_comp_len, 0, padding, stream);
      pfpl_comp_len = aligned_size;
      printf("PFPL compressed length aligned to %zu bytes with %zu bytes padding\n", aligned_size, padding);
    }

    *out_len = pfpl_comp_len;

    printf("PFPL encoded data length: %zu bytes\n", *out_len);
  }

  void decode(uint8_t* in_data, T* out_data, size_t encoded_size, size_t const num_codes, cudaStream_t stream) {

    pfpl::GPU_PFPL_decode<T>(in_data, encoded_size, out_data,
                             buf->d_uncomp_len, buf->blocks, stream);

    printf("PFPL decoding completed for %zu bytes.\n", encoded_size);
  }
};

template <typename T>
PFPL_Codec<T>::PFPL_Codec(size_t data_len, bool is_comp)
    : pimpl(std::unique_ptr<impl>(new impl(data_len, is_comp))) {}

template <typename T>
PFPL_Codec<T>::~PFPL_Codec() = default;

template <typename T>
PFPL_Codec<T>* PFPL_Codec<T>::encode(T* in_data, size_t data_len,
                                     uint8_t** out_comp, size_t* comp_len,
                                     cudaStream_t stream) {
  pimpl->encode(in_data, data_len, out_comp, comp_len, stream);
  return this;
}

template <typename T>
PFPL_Codec<T>* PFPL_Codec<T>::decode(uint8_t* in_comp, T* out_data,
                                     size_t const encoded_size, size_t const num_codes, cudaStream_t stream) {
  pimpl->decode(in_comp, out_data, encoded_size, num_codes, stream);
  return this;
}

template <typename T>
void PFPL_Codec<T>::clear_buffer() {
  pimpl->buf->clear_buffer();
}

template <typename T>
size_t PFPL_Codec<T>::total_footprint_d() {
  return pimpl->buf->total_footprint_d;
}

template <typename T>
size_t PFPL_Codec<T>::total_footprint_h() {
  return pimpl->buf->total_footprint_h;
}

template <typename T>
void PFPL_Codec<T>::set_kernel_params(int len) {
  pimpl->buf->chunks = ((len * sizeof(T)) + pimpl->buf->chunk_size - 1) / pimpl->buf->chunk_size;
  pimpl->buf->max_size = 3 * sizeof(int) + pimpl->buf->chunks * sizeof(short) + pimpl->buf->chunks * pimpl->buf->chunk_size;
  pimpl->buf->num_codes = len;

  printf("PFPL set kernel params: num_codes=%d, chunks=%d, max_size=%zu bytes\n",
         pimpl->buf->num_codes, pimpl->buf->chunks, pimpl->buf->max_size);
}

template class PFPL_Codec<uint8_t>;
template class PFPL_Codec<uint16_t>;
template class PFPL_Codec<uint32_t>;

}  // namespace fz