#include <cuda_runtime.h>

#include <stdexcept>

#include "err.hh"
#include "pfpl.hh"

namespace pfpl {
template <typename T>
void GPU_PFPL_encode(const T*, size_t, uint8_t*, size_t*, int*, int,
                     cudaStream_t);
}

namespace fz {

template <typename T>
struct PFPL_Codec<T>::impl {
  pfpl::PFPL_Buf<T>* buf;

  explicit impl(size_t data_len) : buf(new pfpl::PFPL_Buf<T>(data_len)) {}
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
    *out_len = pfpl_comp_len;

    printf("PFPL encoded data length: %zu bytes\n", *out_len);
  }

  void decode(uint8_t* in_data, T* out_data, size_t data_len, cudaStream_t stream) {
    throw std::runtime_error("PFPL codec is not implemented yet");
  }
};

template <typename T>
PFPL_Codec<T>::PFPL_Codec(size_t data_len)
    : pimpl(std::unique_ptr<impl>(new impl(data_len))) {}

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
                                     size_t data_len, cudaStream_t stream) {
  pimpl->decode(in_comp, out_data, data_len, stream);
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
  pimpl->buf->chunks = (len + pimpl->buf->chunk_size - 1) / pimpl->buf->chunk_size;
  pimpl->buf->max_size = 3 * sizeof(int) + pimpl->buf->chunks * sizeof(short) + pimpl->buf->chunks * pimpl->buf->chunk_size;
  pimpl->buf->num_codes = len;
}

template class PFPL_Codec<uint8_t>;
template class PFPL_Codec<uint16_t>;
template class PFPL_Codec<uint32_t>;

}  // namespace fz