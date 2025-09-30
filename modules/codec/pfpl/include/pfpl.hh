#pragma once

#include <memory>

#include "pfpl_buffer.hh"

namespace fz {

template <typename T>
class PFPL_Codec {
  private:
  struct impl;
  std::unique_ptr<impl> pimpl;

  public:
  PFPL_Codec(size_t const data_len, bool is_comp);
  ~PFPL_Codec();

  PFPL_Codec* encode(T* in_data, size_t const data_len, uint8_t** out_comp, size_t* comp_len, cudaStream_t stream);

  PFPL_Codec* decode(uint8_t* in_comp, T* out_data, size_t const encoded_size, size_t const num_codes, cudaStream_t stream);

  void clear_buffer();

  size_t total_footprint_d();
  size_t total_footprint_h();

  void set_kernel_params(int len);
};

}