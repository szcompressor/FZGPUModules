#pragma once

/**
 * modules/transforms/negabinary/negabinary.h
 *
 * Negabinary (base −2) encoding.
 *
 * Maps a signed two's-complement integer to an unsigned negabinary
 * representation using the compact XOR-mask formula:
 *
 *   encode: u = (n + MASK) ^ MASK     // signed → unsigned
 *   decode: n = (SInt)((u ^ MASK) - MASK)   // unsigned → signed  (NOT self-inverse)
 *
 * where MASK = 0xAA…A (alternating bits: 1010…1010) for the given width.
 *
 * Properties:
 *   - After differencing smooth data the high-order bits of the negabinary
 *     representation vanish faster than zigzag encoding, producing denser
 *     zero runs at high bit-planes when followed by bitshuffle + RZE.
 *   - Size-preserving: input and output have the same byte width.
 *   - Lossless: decode(encode(n)) == n for all n.
 *
 * Design constraints (mirrors zigzag.h):
 *   - All functions are __host__ __device__ — callable from both CUDA device
 *     kernels and host-side C++ code.
 *   - Template parameter T must be a signed integer type (static_assert).
 *   - The matching unsigned type UInt is derived via make_unsigned_t.
 *   - MASK is a compile-time constant derived by truncating 0xAAAAAAAAAAAAAAAA
 *     to the required width (gives 0xAA for 8-bit, 0xAAAA for 16-bit, etc.)
 *
 * Usage:
 *   using NB = fz::Negabinary<int32_t>;
 *   uint32_t u = NB::encode(-3);   // → negabinary representation
 *   int32_t  n = NB::decode(u);    // → -3
 */

#include <cstdint>
#include <type_traits>

#ifdef __CUDACC__
#  define FZ_HOST_DEVICE __host__ __device__
#else
#  define FZ_HOST_DEVICE
#endif

namespace fz {

template <typename T>
struct Negabinary {
    static_assert(std::is_integral<T>::value && std::is_signed<T>::value,
                  "fz::Negabinary<T>: T must be a signed integer type "
                  "(int8_t, int16_t, int32_t, or int64_t).");

    using SInt = T;
    using UInt = typename std::make_unsigned<T>::type;

    // Alternating-bit mask 0xAA…A for the UInt width.
    // Truncating the 64-bit constant to UInt gives:
    //   uint8_t  → 0xAA
    //   uint16_t → 0xAAAA
    //   uint32_t → 0xAAAAAAAA
    //   uint64_t → 0xAAAAAAAAAAAAAAAA
    static constexpr UInt MASK =
        static_cast<UInt>(static_cast<uint64_t>(0xAAAAAAAAAAAAAAAAULL));

    /**
     * Negabinary encode: signed integer → unsigned negabinary.
     *
     * The formula (n + MASK) ^ MASK converts two's complement to base-(-2).
     */
    FZ_HOST_DEVICE static constexpr UInt encode(SInt n) noexcept {
        return (static_cast<UInt>(n) + MASK) ^ MASK;
    }

    /**
     * Negabinary decode: unsigned negabinary → signed integer.
     * Inverse of encode(); decode(encode(n)) == n for all n.
     *
     * Derivation from encode u = (n + MASK) ^ MASK:
     *   u ^ MASK = n + MASK
     *   n = (u ^ MASK) - MASK   (unsigned subtraction wraps, then reinterpret as SInt)
     */
    FZ_HOST_DEVICE static constexpr SInt decode(UInt u) noexcept {
        return static_cast<SInt>((u ^ MASK) - MASK);
    }
};

// ---------------------------------------------------------------------------
// Convenience aliases for the four standard widths
// ---------------------------------------------------------------------------
using Negabinary8  = Negabinary<int8_t>;
using Negabinary16 = Negabinary<int16_t>;
using Negabinary32 = Negabinary<int32_t>;
using Negabinary64 = Negabinary<int64_t>;

}  // namespace fz

#undef FZ_HOST_DEVICE
