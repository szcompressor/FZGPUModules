#pragma once

/**
 * modules/transforms/zigzag/zigzag.h
 *
 * Zigzag (TCMS — Two's Complement to Magnitude-Sign) encoding.
 *
 * Maps signed integers to unsigned integers so that small-magnitude values
 * (both positive and negative) map to small non-negative integers:
 *
 *   encode:  u = (x << 1) ^ (x >> (W-1))     // signed → unsigned
 *   decode:  x = (u >> 1) ^ -(u & 1)          // unsigned → signed
 *
 * Design constraints:
 *   - All functions are __host__ __device__ so they are callable from both
 *     CUDA device kernels and host-side C++ code.
 *   - The template is parameterised on the *signed* type T; the matching
 *     unsigned type UInt is derived automatically via make_unsigned_t.
 *   - T must be a signed integer type (enforced by static_assert).
 *   - encode()/decode() use the exact-bitwidth companion type to avoid
 *     undefined behaviour from signed right shift on non-matching types.
 *
 * Usage:
 *   using ZZ = fz::Zigzag<int32_t>;
 *   uint32_t u = ZZ::encode(-3);   // → 5
 *   int32_t  x = ZZ::decode(5u);   // → -3
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
struct Zigzag {
    static_assert(std::is_integral<T>::value && std::is_signed<T>::value,
                  "fz::Zigzag<T>: T must be a signed integer type "
                  "(int8_t, int16_t, int32_t, or int64_t).");

    using SInt = T;
    using UInt = typename std::make_unsigned<T>::type;

    static constexpr int W = sizeof(T) * 8;

    /**
     * Zigzag encode: signed integer → unsigned integer.
     *
     * Small-magnitude signed values map to small unsigned values:
     *   0 → 0,  -1 → 1,  1 → 2,  -2 → 3,  2 → 4, …
     */
    FZ_HOST_DEVICE static constexpr UInt encode(SInt x) noexcept {
        // Cast to unsigned before shift to avoid UB on signed overflow,
        // then XOR with the arithmetic right-shift mask.
        return (static_cast<UInt>(x) << 1) ^ static_cast<UInt>(x >> (W - 1));
    }

    /**
     * Zigzag decode: unsigned integer → signed integer.
     * Inverse of encode(); decode(encode(x)) == x for all x.
     */
    FZ_HOST_DEVICE static constexpr SInt decode(UInt u) noexcept {
        // (u >> 1) reverses the shift; -(u & 1) reconstructs the sign mask.
        // Cast to signed only at the final XOR to keep arithmetic well-defined.
        return static_cast<SInt>((u >> 1) ^ static_cast<UInt>(-(static_cast<SInt>(u & 1u))));
    }
};

// ---------------------------------------------------------------------------
// Convenience aliases for the four standard widths
// ---------------------------------------------------------------------------
using Zigzag8  = Zigzag<int8_t>;
using Zigzag16 = Zigzag<int16_t>;
using Zigzag32 = Zigzag<int32_t>;
using Zigzag64 = Zigzag<int64_t>;

}  // namespace fz

#undef FZ_HOST_DEVICE
