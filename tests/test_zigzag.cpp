/**
 * tests/test_zigzag.cpp
 *
 * Unit tests for fz::Zigzag<T> (transforms/zigzag/zigzag.h).
 *
 * All tests run host-side only; no CUDA device is required.
 * The __host__ __device__ functions compile fine as plain C++ when
 * __CUDACC__ is not defined, so this file links against GTest only.
 *
 * Properties verified:
 *   1. encode/decode are bijections (round-trip identity).
 *   2. encode maps signed magnitude correctly (known test vectors).
 *   3. decode(encode(x)) == x for all representable values (exhaustive
 *      for 8-bit and 16-bit; sampled for 32-bit and 64-bit).
 *   4. Zero encodes to zero.
 *   5. The most-negative value (INT_MIN) round-trips correctly.
 *   6. Convenience aliases compile and agree with the primary template.
 */

#include <gtest/gtest.h>
#include "transforms/zigzag/zigzag.h"

#include <cstdint>
#include <limits>
#include <vector>

using namespace fz;

// ─────────────────────────────────────────────────────────────────────────────
// 1. Known test vectors
// ─────────────────────────────────────────────────────────────────────────────

TEST(ZigzagKnownVectors, Int32) {
    //  x   →  encode(x)
    //  0   →  0
    // -1   →  1
    //  1   →  2
    // -2   →  3
    //  2   →  4
    // INT32_MIN → UINT32_MAX  (the most-negative maps to the largest uint)
    using ZZ = Zigzag<int32_t>;

    EXPECT_EQ(ZZ::encode( 0),  0u);
    EXPECT_EQ(ZZ::encode(-1),  1u);
    EXPECT_EQ(ZZ::encode( 1),  2u);
    EXPECT_EQ(ZZ::encode(-2),  3u);
    EXPECT_EQ(ZZ::encode( 2),  4u);
    EXPECT_EQ(ZZ::encode(-3),  5u);
    EXPECT_EQ(ZZ::encode( 3),  6u);
    EXPECT_EQ(ZZ::encode(std::numeric_limits<int32_t>::max()),
              static_cast<uint32_t>(std::numeric_limits<int32_t>::max()) * 2u);
    EXPECT_EQ(ZZ::encode(std::numeric_limits<int32_t>::min()),
              std::numeric_limits<uint32_t>::max());
}

TEST(ZigzagKnownVectors, Int16) {
    using ZZ = Zigzag<int16_t>;

    EXPECT_EQ(ZZ::encode(int16_t( 0)), uint16_t(0));
    EXPECT_EQ(ZZ::encode(int16_t(-1)), uint16_t(1));
    EXPECT_EQ(ZZ::encode(int16_t( 1)), uint16_t(2));
    EXPECT_EQ(ZZ::encode(int16_t(-2)), uint16_t(3));
    EXPECT_EQ(ZZ::encode(int16_t( 2)), uint16_t(4));
}

TEST(ZigzagKnownVectors, Int8) {
    using ZZ = Zigzag<int8_t>;

    EXPECT_EQ(ZZ::encode(int8_t( 0)), uint8_t(0));
    EXPECT_EQ(ZZ::encode(int8_t(-1)), uint8_t(1));
    EXPECT_EQ(ZZ::encode(int8_t( 1)), uint8_t(2));
    EXPECT_EQ(ZZ::encode(int8_t(-63)), uint8_t(125));
    EXPECT_EQ(ZZ::encode(int8_t( 63)), uint8_t(126));
    EXPECT_EQ(ZZ::encode(std::numeric_limits<int8_t>::min()),
              std::numeric_limits<uint8_t>::max());
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. Round-trip: exhaustive for 8-bit and 16-bit
// ─────────────────────────────────────────────────────────────────────────────

TEST(ZigzagRoundTrip, Exhaustive8Bit) {
    using ZZ = Zigzag<int8_t>;
    for (int i = std::numeric_limits<int8_t>::min();
         i    <= std::numeric_limits<int8_t>::max(); ++i) {
        int8_t x = static_cast<int8_t>(i);
        EXPECT_EQ(ZZ::decode(ZZ::encode(x)), x)
            << "round-trip failed for x=" << static_cast<int>(x);
    }
}

TEST(ZigzagRoundTrip, Exhaustive16Bit) {
    using ZZ = Zigzag<int16_t>;
    for (int i = std::numeric_limits<int16_t>::min();
         i    <= std::numeric_limits<int16_t>::max(); ++i) {
        int16_t x = static_cast<int16_t>(i);
        EXPECT_EQ(ZZ::decode(ZZ::encode(x)), x)
            << "round-trip failed for x=" << i;
    }
}

TEST(ZigzagRoundTrip, Sampled32Bit) {
    using ZZ = Zigzag<int32_t>;

    // Sample a large range including extremes and values near zero.
    std::vector<int32_t> cases = {
        0, 1, -1, 2, -2, 127, -127, 128, -128,
        32767, -32767, 32768, -32768,
        1'000'000, -1'000'000,
        std::numeric_limits<int32_t>::max(),
        std::numeric_limits<int32_t>::min(),
        std::numeric_limits<int32_t>::max() - 1,
        std::numeric_limits<int32_t>::min() + 1,
    };
    // Also step through every 65537th value to cover the midrange cheaply.
    for (int64_t v = std::numeric_limits<int32_t>::min();
         v        <= std::numeric_limits<int32_t>::max();
         v += 65537) {
        cases.push_back(static_cast<int32_t>(v));
    }

    for (int32_t x : cases) {
        EXPECT_EQ(ZZ::decode(ZZ::encode(x)), x)
            << "round-trip failed for x=" << x;
    }
}

TEST(ZigzagRoundTrip, Sampled64Bit) {
    using ZZ = Zigzag<int64_t>;

    std::vector<int64_t> cases = {
        0LL, 1LL, -1LL, 2LL, -2LL,
        INT64_MAX, INT64_MIN,
        INT64_MAX - 1, INT64_MIN + 1,
        static_cast<int64_t>(1) << 32,
        -(static_cast<int64_t>(1) << 32),
        static_cast<int64_t>(1e15),
        -static_cast<int64_t>(1e15),
    };

    for (int64_t x : cases) {
        EXPECT_EQ(ZZ::decode(ZZ::encode(x)), x)
            << "round-trip failed for x=" << x;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. Zero always encodes to zero
// ─────────────────────────────────────────────────────────────────────────────

TEST(ZigzagProperties, ZeroMapsToZero) {
    EXPECT_EQ(Zigzag<int8_t>::encode(0),  uint8_t(0));
    EXPECT_EQ(Zigzag<int16_t>::encode(0), uint16_t(0));
    EXPECT_EQ(Zigzag<int32_t>::encode(0), uint32_t(0));
    EXPECT_EQ(Zigzag<int64_t>::encode(0), uint64_t(0));
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. Monotone magnitude ordering: encode(k) < encode(k+1) for k >= 0
//    and encode(-k) < encode(-(k+1)) for k >= 1
// ─────────────────────────────────────────────────────────────────────────────

TEST(ZigzagProperties, MonotoneMagnitude32) {
    using ZZ = Zigzag<int32_t>;
    for (int32_t k = 0; k < 1000; ++k) {
        EXPECT_LT(ZZ::encode(k),  ZZ::encode(k + 1))
            << "non-monotone at k=" << k;
        EXPECT_LT(ZZ::encode(-k), ZZ::encode(-(k + 1)))
            << "non-monotone at -k=" << k;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. Convenience aliases agree with the primary template
// ─────────────────────────────────────────────────────────────────────────────

TEST(ZigzagAliases, AgreePrimaryTemplate) {
    EXPECT_EQ(Zigzag8::encode(int8_t(-1)),   Zigzag<int8_t>::encode(int8_t(-1)));
    EXPECT_EQ(Zigzag16::encode(int16_t(-1)), Zigzag<int16_t>::encode(int16_t(-1)));
    EXPECT_EQ(Zigzag32::encode(int32_t(-1)), Zigzag<int32_t>::encode(int32_t(-1)));
    EXPECT_EQ(Zigzag64::encode(int64_t(-1)), Zigzag<int64_t>::encode(int64_t(-1)));
}
