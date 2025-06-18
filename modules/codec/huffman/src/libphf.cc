#include <cuda_runtime.h>

#include <cstdio>

#include "hf.h"
#include "hf_hl.hh"
#include "hf_impl.hh"

namespace phf {

const char* BACKEND_TEXT = "cuHF";

const char* VERSION_TEXT = "coarse_2024-03-20, ReVISIT_2021-(TBD)";
// const int VERSION = 20240527;

const int COMPATIBILITY = 0;

}  // namespace phf

size_t capi_phf_coarse_tune_sublen(size_t len)
{
  using phf::HuffmanHelper;
  auto div = [](auto _l, auto _subl) { return (_l - 1) / _subl + 1; };

  // TODO ROCm GPUs should use different constants.
  int current_dev = 0;
  cudaSetDevice(current_dev);
  cudaDeviceProp dev_prop{};
  cudaGetDeviceProperties(&dev_prop, current_dev);

  auto nSM = dev_prop.multiProcessorCount;
  auto allowed_block_dim = dev_prop.maxThreadsPerBlock;
  auto deflate_nthread = allowed_block_dim * nSM / HuffmanHelper::DEFLATE_CONSTANT;

  auto optimal_sublen = div(len, deflate_nthread);
  // round up
  optimal_sublen =
      div(optimal_sublen, HuffmanHelper::BLOCK_DIM_DEFLATE) * HuffmanHelper::BLOCK_DIM_DEFLATE;

  return optimal_sublen;
};

void capi_phf_coarse_tune(size_t len, int* sublen, int* pardeg) {
  auto div = [](auto l, auto subl) { return (l - 1) / subl + 1; };
  *sublen = capi_phf_coarse_tune_sublen(len);
  *pardeg = div(len, *sublen);
}

uint32_t capi_phf_encoded_bytes(phf_header* h) {
  return h->entry[PHFHEADER_END];
}

void capi_phf_version() {
  printf("\n///  %s build: %s\n", phf::BACKEND_TEXT, phf::VERSION_TEXT);
}

void capi_phf_versioninfo() { capi_phf_version(); }