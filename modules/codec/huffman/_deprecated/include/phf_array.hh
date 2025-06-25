#ifndef BD7D510D_8AA0_4DA6_93F2_6F2CC27BF1A5
#define BD7D510D_8AA0_4DA6_93F2_6F2CC27BF1A5

#include <cstddef>
#include <cstdint>

// dense array, 1d
template <typename T>
struct array1 {
  T* buf;
  size_t len;
};

// sparse array, 1d
template <typename T>
struct compact_array1 {
  T* const val;
  uint32_t* idx;
  uint32_t* num;
  uint32_t* host_num;
  size_t reserved_len;
};

namespace phf {

template <typename T>
using array = array1<T>;

template <typename T>
using sparse = compact_array1<T>;

template <typename Hf>
struct book {
  Hf* bk;
  uint16_t bklen;
  Hf const alt_prefix_code;  // even if u8 can use short u4 internal
  uint32_t const alt_bitcount;
};

template <typename Hf>
struct dense {
  Hf* const out;
  uint32_t* bits;
  size_t n_part;
};

struct par_config {
  const size_t sublen;
  const size_t pardeg;
};

}  // namespace phf

#endif /* BD7D510D_8AA0_4DA6_93F2_6F2CC27BF1A5 */
