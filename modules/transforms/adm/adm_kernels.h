#pragma once
// Internal header — not part of the public API.
// Shared constants, scratch-pointer struct, and host wrapper declarations
// for the ADM (Adaptive Data Mapping) encode/decode kernels.

#include "backend/types.h"
#include <cstdint>
#include <cstddef>

namespace fz {
namespace adm {

// ── Shared constants ──────────────────────────────────────────────────────────

// Block structure: 32 threads/warp × 16 elements/thread = 512 elements/warp-block.
static constexpr int kBlockThreads  = 32;
static constexpr int kChunk         = 16;
static constexpr int kDecmpChunk    = 32;
static constexpr int kBlockElems    = kBlockThreads * kChunk;  // 512

// Decoupled look-back prefix-sum limit (warps). Above this, Thrust fallback is used.
static constexpr int kDecoupledMaxGsize = 1024;

// Multi-warp-block redesign: group this many 32-thread "warp blocks" into one
// CUDA thread block, so the decoupled look-back chain walks blocks (length
// gsize/kWarpsPerBlock) instead of individual warps (length gsize). Profiling
// (ncu) showed the decoupled kernel was ~7% achieved occupancy against a 50%
// theoretical ceiling, with 63% of warp cycles stalled on the look-back's
// memory-barrier spin — a latency/serialization problem, not a register or
// local-memory-spill one (only 32 regs/thread, local_code/local_bits are
// tiny). That profile is why this lever is expected to help here, unlike the
// register-bound cuSZp3 compress kernels it's adapted from.
static constexpr int kWarpsPerBlock = 8;

inline size_t adm_num_blocks(size_t gsize) {
    return (gsize + static_cast<size_t>(kWarpsPerBlock) - 1)
         / static_cast<size_t>(kWarpsPerBlock);
}

// Center-relative shift (1 = code 1 means "equal to center").
static constexpr int kShift = 1;

// Maximum signal bytes per input element.
static constexpr int kMaxSignalBytesU16 = 2;
static constexpr int kMaxSignalBytesU32 = 4;

// ── Size helpers ──────────────────────────────────────────────────────────────

inline size_t adm_gsize(size_t n) {
    return (n + static_cast<size_t>(kBlockElems) - 1) / kBlockElems;
}

inline size_t adm_flags_bytes(size_t gsize) {
    return (gsize + 7) / 8;
}

inline size_t adm_flags_words(size_t gsize) {
    return (adm_flags_bytes(gsize) + sizeof(uint32_t) - 1) / sizeof(uint32_t);
}

// ── Scratch pointer bundle ────────────────────────────────────────────────────
// All pointers are pool-managed device allocations. The stage pre-allocates
// them in onFinalize()/initScratch() and reuses them every execute() call.

struct AdmScratch {
    int*      d_signal_length;    // gsize × sizeof(int)
    int*      d_output_lengths;   // (gsize+1) × sizeof(int)
    void*     d_centers;          // gsize × sizeof(uint16_t or uint32_t)
    uint32_t* d_block_flags;      // adm_flags_words(gsize) × sizeof(uint32_t)
    uint8_t*  d_codes;            // num_elements × 1
    uint8_t*  d_concat_signals;   // num_elements × kMaxSignalBytes
    uint8_t*  d_bit_signals;      // num_elements × kMaxSignalBytes (thrust path)
    int*      d_loc_offset;       // (gsize+1) × sizeof(int)  (decoupled path)
    int*      d_prefix_state;     // (gsize+1) × sizeof(int)  (decoupled path)
    int*      d_block_resolved;   // (num_blocks+1) × sizeof(int) (decoupled path,
                                   //  block-level look-back "resolved" value —
                                   //  see kWarpsPerBlock)
    unsigned int* d_overflow_flag; // 1 word; written by kernels only in debug builds
};

// ── u16 wrappers ──────────────────────────────────────────────────────────────

void compress_u16(
    const uint16_t* d_input, size_t num_elements,
    uint8_t* d_output, size_t& output_size,
    const AdmScratch& s, fz::stream_t stream);

void decompress_u16(
    const uint8_t* d_input, size_t input_size,
    uint16_t* d_output, size_t num_elements,
    const AdmScratch& s, fz::stream_t stream);

size_t get_max_u16_payload_bytes(size_t num_elements);

// ── u32 wrappers ──────────────────────────────────────────────────────────────

void compress_u32(
    const uint32_t* d_input, size_t num_elements,
    uint8_t* d_output, size_t& output_size,
    const AdmScratch& s, fz::stream_t stream);

void decompress_u32(
    const uint8_t* d_input, size_t input_size,
    uint32_t* d_output, size_t num_elements,
    const AdmScratch& s, fz::stream_t stream);

size_t get_max_u32_payload_bytes(size_t num_elements);

} // namespace adm
} // namespace fz
