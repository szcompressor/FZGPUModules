#pragma once
#include <cstddef>
#include <cstdint>

namespace pfpl {

template <typename T>
struct PFPL_Buf {
  PFPL_Buf(size_t data_len, bool comp = true);
  ~PFPL_Buf();

  void init(size_t data_len, bool comp = true);
  void clear_buffer();

  const int chunk_size = 1024 * 16;
  const int threads_per_block = 512;
  const int warps_size = 32;

  int num_codes = 0;

  int SMs = 0;
  int mTpSM = 0;
  int blocks = 0;
  int chunks = 0;
  int max_size = 0;

  uint8_t* d_archive = nullptr;
  size_t max_archive_bytes = 0;

  int* d_fullcarry = nullptr;

  uint8_t* d_comp_out = nullptr;
  size_t* d_comp_len = nullptr;
  int* d_uncomp_len = nullptr;

  size_t total_footprint_d = 0;
  size_t total_footprint_h = 0;
};

}  // namespace pfpl