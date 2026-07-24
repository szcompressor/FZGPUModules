#include "fused/bitplane_rze/bitplane_rze_kernels.h"
#include "backend/warp.h"
#include "cuda_check.h"

#include "bitplane_rze_encode.inl"
#include "bitplane_rze_decode.inl"

namespace fz {
namespace bitplane_rze {

Config configure(size_t data_len) {
  // Pad byte count up to a 4096-byte chunk multiple, matching FZ-GPU.
  size_t data_bytes = data_len * sizeof(uint16_t);
  data_bytes = ((data_bytes + 4096 - 1) / 4096) * 4096;
  if (data_bytes == 0) data_bytes = 4096;  // one empty chunk for zero input
  Config c;
  c.data_len   = data_len;
  c.data_bytes = data_bytes;
  c.pad_len    = data_bytes / sizeof(uint16_t);
  c.chunk_size = data_bytes / 512;   // # uint32 bitflag words
  c.grid_x     = data_bytes / 4096;  // # CUDA blocks (one per 4096-byte chunk)
  return c;
}

size_t maxArchiveBytes(size_t data_len) {
  Config c = configure(data_len);
  // header + bitflag + start-pos + worst-case bitstream (every group retained).
  return kBitplaneRzeHeaderBytes
       + sizeof(uint32_t) * c.chunk_size
       + sizeof(uint32_t) * c.grid_x
       + c.data_bytes;
}

void launchEncode(
    const uint16_t* d_in, const Config& cfg,
    uint32_t* d_offset_counter, uint32_t* d_bitflag, uint32_t* d_start_pos,
    uint8_t* d_comp_out, uint32_t* d_comp_len, cudaStream_t stream) {
  dim3 grid(static_cast<unsigned>(cfg.grid_x));
  dim3 block(32, 32);
  KERNEL_fused_encode<<<grid, block, 0, stream>>>(
      reinterpret_cast<const uint32_t*>(d_in), cfg.pad_len / 2,
      d_offset_counter, d_bitflag, d_start_pos,
      reinterpret_cast<uint32_t*>(d_comp_out), d_comp_len);
  FZ_CUDA_CHECK(cudaGetLastError());
}

void launchDecode(
    const uint8_t* d_bitstream, const uint32_t* d_bitflag,
    const uint32_t* d_start_pos, uint16_t* d_out, const Config& cfg,
    cudaStream_t stream) {
  dim3 grid(static_cast<unsigned>(cfg.grid_x));
  dim3 block(32, 32);
  KERNEL_fused_decode<<<grid, block, 0, stream>>>(
      reinterpret_cast<uint32_t*>(const_cast<uint8_t*>(d_bitstream)),
      const_cast<uint32_t*>(d_bitflag), const_cast<uint32_t*>(d_start_pos),
      reinterpret_cast<uint32_t*>(d_out), cfg.pad_len / 2);
  FZ_CUDA_CHECK(cudaGetLastError());
}

}  // namespace bitplane_rze
}  // namespace fz
