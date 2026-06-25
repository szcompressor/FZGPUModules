#pragma once

/**
 * @file adaptive_bitpack_kernels.h
 * @brief Per-block adaptive fixed-rate bit-plane packing kernels (cuSZp-style
 *        plain mode), decoupled from the offset scan.
 *
 * Layout (one logical "data block" of `block_size` signed elements):
 *   - rate byte `r` = number of bit-planes needed for max |value| in the block
 *     (0 if the whole block is zero).
 *   - if r > 0: a sign region of `word_bytes = ceil(block_size/8)` bytes
 *     (bit j of byte k = sign of element 8k+j), followed by `r` bit-planes of
 *     `word_bytes` bytes each (bit j of byte k of plane p = bit p of |element 8k+j|).
 *   - block payload cost = r > 0 ? word_bytes * (r + 1) : 0.
 *
 * Archive = [num_blocks rate bytes] [payloads packed by exclusive-scan offsets].
 * The stage carries `block_size` and `num_elements` in the FZM header, so the
 * archive needs no internal header of its own.
 */

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace fz {
namespace adaptive_bitpack {

/// Derived block geometry for a given element count and block size.
/// `meta_bytes` is the per-block metadata stride: 1 for plain mode (just the rate
/// byte), 2 for outlier-selection mode (rate byte + selection byte).
struct Config {
    size_t   num_elements = 0;
    uint32_t block_size   = 32;
    uint32_t word_bytes   = 4;   ///< ceil(block_size / 8)
    size_t   num_blocks   = 0;   ///< ceil(num_elements / block_size)
    uint32_t meta_bytes   = 1;   ///< per-block metadata stride (1 plain, 2 outlier)
};

inline Config configure(size_t num_elements, uint32_t block_size,
                        bool outlier_selection = false) {
    Config c;
    c.num_elements = num_elements;
    c.block_size   = block_size;
    c.word_bytes   = (block_size + 7u) / 8u;
    c.num_blocks   = (block_size == 0)
        ? 0
        : (num_elements + block_size - 1) / block_size;
    c.meta_bytes   = outlier_selection ? 2u : 1u;
    return c;
}

/// Worst-case archive size: every block stores its metadata plus a full-width
/// sign region and `bits_per_elem` bit-planes (the cheaper of the plain/outlier
/// candidates is always <= the plain candidate, so this bounds both modes).
inline size_t maxArchiveBytes(const Config& c, unsigned bits_per_elem) {
    return c.num_blocks * static_cast<size_t>(c.meta_bytes)
         + c.num_blocks * static_cast<size_t>(c.word_bytes) * (bits_per_elem + 1);
}

// ── Encode ────────────────────────────────────────────────────────────────
// Pass A: write the per-block rate byte into `d_rate` (the archive's rate
// region) and the per-block payload byte cost into `d_cost`.
template<typename T>
void launchEncodeRate(const T* d_in, const Config& c,
                      uint8_t* d_rate, uint32_t* d_cost, cudaStream_t stream);

// Pass B: pack sign + bit-planes for each block at `d_payload + d_offset[b]`,
// where `d_offset` is the exclusive scan of `d_cost`.
template<typename T>
void launchEncodePack(const T* d_in, const Config& c,
                      const uint8_t* d_rate, const uint32_t* d_offset,
                      uint8_t* d_payload, cudaStream_t stream);

// ── Decode ────────────────────────────────────────────────────────────────
// Pass A: per-block payload byte cost from the rate region (for the scan).
void launchDecodeCost(const uint8_t* d_rate, const Config& c,
                      uint32_t* d_cost, cudaStream_t stream);

// Pass B: unpack each block from `d_payload + d_offset[b]` into `d_out`.
template<typename T>
void launchDecodeUnpack(const uint8_t* d_rate, const uint32_t* d_offset,
                        const uint8_t* d_payload, const Config& c,
                        T* d_out, cudaStream_t stream);

// ── Outlier-selection mode (cuSZp2) ─────────────────────────────────────────
// Per block, choose the cheaper of (a) plain packing of all elements or
// (b) extracting element 0 as a 1..sizeof(T)-byte raw outlier and packing only
// elements 1..n-1. Metadata is 2 bytes per block: [rate][sel], where sel bit0 =
// is_outlier and (when set) sel bits1-2 = outlier_byte_num - 1.
template<typename T>
void launchEncodeRateOutlier(const T* d_in, const Config& c,
                             uint8_t* d_meta, uint32_t* d_cost, cudaStream_t stream);
template<typename T>
void launchEncodePackOutlier(const T* d_in, const Config& c,
                             const uint8_t* d_meta, const uint32_t* d_offset,
                             uint8_t* d_payload, cudaStream_t stream);
void launchDecodeCostOutlier(const uint8_t* d_meta, const Config& c,
                             uint32_t* d_cost, cudaStream_t stream);
template<typename T>
void launchDecodeUnpackOutlier(const uint8_t* d_meta, const uint32_t* d_offset,
                               const uint8_t* d_payload, const Config& c,
                               T* d_out, cudaStream_t stream);

} // namespace adaptive_bitpack
} // namespace fz
