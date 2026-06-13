#pragma once

/**
 * @file bitplane_rle_kernels.h
 * @brief Host-callable launchers + config helper for the fused bitplane-RLE
 *        (FZ-GPU lossless) encode/decode kernels.
 *
 * Internal interface — only `bitplane_rle_kernels.cu` and
 * `bitplane_rle_stage.cu` include it. The two device kernels live in
 * `bitplane_rle_encode.inl` / `bitplane_rle_decode.inl`, included privately by
 * `bitplane_rle_kernels.cu`.
 *
 * The codec is hard-wired to a 4096-byte data chunk (a 32×32 grid of
 * `uint32_t` words). Each CUDA block processes one chunk: it bit-transposes the
 * 32×32 bit matrix, then eliminates all-zero 4-byte groups, reserving output
 * space with a per-block `atomicAdd`. Input is interpreted as `uint16_t`
 * symbols packed two-per-word, matching FZ-GPU's quantizer codes.
 */

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace fz {
namespace bitplane_rle {

/// Self-describing 128-byte archive header, embedded at byte 0 of the output.
/// Mirrors FZ-GPU's `fzg_header`. `entry[]` holds cumulative byte offsets:
///   entry[0] = 0                  (header start)
///   entry[1] = 128                (bitflag array start)
///   entry[2] = entry[1] + bitflag bytes   (start-position array start)
///   entry[3] = entry[2] + startpos bytes  (compacted bitstream start)
///   entry[4] = entry[3] + bitstream bytes (= total archive size)
enum : int {
  BPRLE_HDR_HEADER    = 0,
  BPRLE_HDR_BITFLAG   = 1,
  BPRLE_HDR_STARTPOS  = 2,
  BPRLE_HDR_BITSTREAM = 3,
  BPRLE_HDR_END       = 4,
};

constexpr size_t kBitplaneRleHeaderBytes = 128;

struct ArchiveHeader {
  size_t   original_len;             ///< Pre-pad uint16 symbol count.
  uint32_t entry[BPRLE_HDR_END + 1]; ///< Cumulative byte offsets (see above).
  uint8_t  padding[kBitplaneRleHeaderBytes - sizeof(size_t) -
                   sizeof(uint32_t) * (BPRLE_HDR_END + 1)];
};
static_assert(sizeof(ArchiveHeader) == kBitplaneRleHeaderBytes,
              "ArchiveHeader must be exactly 128 bytes");

/// Sizes derived from the (unpadded) uint16 symbol count.
struct Config {
  size_t data_len;    ///< User-visible uint16 symbol count.
  size_t pad_len;     ///< Symbol count padded up to a 4096-byte multiple.
  size_t data_bytes;  ///< Padded byte count (= pad_len * 2).
  size_t chunk_size;  ///< Number of uint32 bitflag words (data_bytes / 512).
  size_t grid_x;      ///< Number of CUDA blocks (data_bytes / 4096).
};

/// Compute the padded layout for `data_len` uint16 symbols.
Config configure(size_t data_len);

/// Worst-case archive byte count for `data_len` symbols. Safe upper bound for
/// output-buffer allocation (header + bitflag + start-pos + a bitstream that
/// retains every 4-byte group).
size_t maxArchiveBytes(size_t data_len);

/**
 * Forward (compress). Bit-transpose + zero-byte RLE of `pad_len/2` uint32 words
 * read from `d_in` into the three archive sub-arrays. The kernel reads exactly
 * `cfg.data_bytes` bytes, so `d_in` must point to a buffer of at least that
 * size with any tail beyond the real input zeroed.
 *
 * `d_offset_counter` (1 uint32) must be zeroed on `stream` before the call; the
 * kernel atomically accumulates the total compacted word count into it.
 * `d_comp_len` (grid_x uint32) is per-block scratch written by the kernel.
 *
 * This launcher does NOT synchronize — the caller reads `d_offset_counter` D2H
 * afterwards to learn the bitstream length.
 */
void launchEncode(
    const uint16_t* d_in, const Config& cfg,
    uint32_t* d_offset_counter, uint32_t* d_bitflag, uint32_t* d_start_pos,
    uint8_t* d_comp_out, uint32_t* d_comp_len, cudaStream_t stream);

/**
 * Inverse (decompress). Reconstruct `pad_len` uint16 symbols into `d_out` from
 * the archive sub-arrays. `d_out` must hold at least `cfg.data_bytes` bytes
 * (the kernel writes the full padded length; trailing padding is zero).
 */
void launchDecode(
    const uint8_t* d_bitstream, const uint32_t* d_bitflag,
    const uint32_t* d_start_pos, uint16_t* d_out, const Config& cfg,
    cudaStream_t stream);

}  // namespace bitplane_rle
}  // namespace fz
