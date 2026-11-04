#pragma once

/**
 * @file modules/fused/chunk_fusion/pfpl_stream.h
 * @brief Two-array PFPL: a float field compresses straight into one caller
 *        buffer and decompresses straight back, stream-ordered, with no host
 *        synchronization and no device temporaries.
 *
 * The stream is self-describing: a 16-byte StreamHeader, then the LC/RZE
 * archive that the PFPL preset (examples/presets/pfpl_specialization.toml with
 * error_bound_mode = "ABS": Quantizer in-place zigzag -> Difference negabinary
 * -> Bitshuffle -> RZE over 16 KiB chunks) produces for the same input and
 * bound, byte for byte. Encoding and decoding are one kernel each over the
 * chunk-cooperative harness (chunk_fusion.cuh): one block per chunk keeps every
 * intermediate in shared memory and finds the chunk's place in the archive by
 * decoupled look-back, so neither direction touches device memory beyond the
 * two arrays and a Plan's 8 bytes of look-back state per chunk.
 *
 * Archive bytes are a multiple of 4; a field whose length is not a whole number
 * of chunks is coded padded with zeros (as the staged pipeline pads it) and
 * decodes to exactly its own length. Both arrays must be 4-byte aligned.
 */

#include "backend/types.h"

#include <cstddef>
#include <cstdint>

namespace fz {
namespace pfpl {

/// Leading 16 bytes of every stream.
struct StreamHeader {
    uint32_t magic;          ///< kStreamMagic
    uint32_t archive_bytes;  ///< LC archive bytes after the header (a multiple of 4)
    float    abs_eb;         ///< absolute error bound every value was coded to
    uint32_t n;              ///< float elements of the field
};
static_assert(sizeof(StreamHeader) == 16, "the PFPL stream header is 16 bytes");

constexpr uint32_t kStreamMagic = 0x4C504650u;   ///< "PFPL" in little-endian byte order
constexpr int      kChunkBytes  = 16384;         ///< the preset's Difference/Bitshuffle/RZE chunk
constexpr uint32_t kQuantRadius = 4194304;       ///< the preset's quant_radius

/** Longest stream n floats can produce (every chunk stored raw). */
size_t maxStreamBytes(size_t n);

/** Length of the stream whose first bytes are at `h_stream` (host memory), or 0
 *  if they are not a PFPL stream header or claim more than `capacity` bytes. */
size_t streamBytes(const void* h_stream, size_t capacity = SIZE_MAX);

/**
 * Per-stream look-back state. Calls through one Plan must be ordered (one CUDA/HIP
 * stream, or synchronized between streams): use one Plan per concurrently used
 * stream. It grows to the largest field it has seen.
 */
class Plan {
public:
    Plan() = default;
    ~Plan();
    Plan(const Plan&)            = delete;
    Plan& operator=(const Plan&) = delete;

    /** Make room for fields of up to n floats. Allocating synchronizes the device. */
    void reserve(size_t n);

private:
    friend void compress(Plan&, const float*, size_t, float, void*, size_t, size_t*, fz::stream_t);
    friend void decompress(Plan&, const void*, float*, size_t, fz::stream_t);
    uint32_t nextEpoch(fz::stream_t stream);

    unsigned long long* state_   = nullptr;  ///< one look-back word per chunk, tagged with its call's epoch
    unsigned long long* ticket_  = nullptr;  ///< encode chunk tickets handed out on the device
    size_t              chunks_  = 0;        ///< chunks state_ holds
    unsigned long long  tickets_ = 0;        ///< the ticket counter's value between calls
    uint32_t            epoch_   = 0;        ///< last epoch used (0 = never: fresh state reads as empty)
};

/**
 * Compress n floats at d_in to within abs_eb into d_out, which holds `capacity`
 * bytes. Stream-ordered, no host synchronization. *d_stream_bytes (device
 * memory; may be null) receives the stream's length -- greater than `capacity`
 * when it does not fit. Nothing at or past d_out + capacity is ever written; a
 * stream that does not fit leaves its 16-byte header (when that fits), which
 * records the length it needed, so streamBytes() reports the overflow too.
 * abs_eb must be positive and finite; the field must code to under 4 GiB (the
 * LC archive's size fields are 32-bit).
 */
void compress(Plan& plan, const float* d_in, size_t n, float abs_eb,
              void* d_out, size_t capacity, size_t* d_stream_bytes, fz::stream_t stream);

/**
 * Decompress the compress() stream at d_stream, which must describe n floats,
 * into d_out[0, n). Stream-ordered, no host synchronization. A stream that is
 * not a PFPL stream of n floats stops the kernel (device trap) rather than
 * decoding garbage.
 */
void decompress(Plan& plan, const void* d_stream, float* d_out, size_t n, fz::stream_t stream);

} // namespace pfpl
} // namespace fz
