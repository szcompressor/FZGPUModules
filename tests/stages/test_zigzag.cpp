/**
 * tests/stages/test_zigzag.cpp
 *
 * Host-only unit tests for fz::Zigzag<T> (transforms/zigzag/zigzag.h).
 * No CUDA device required — __host__ __device__ functions compile as plain C++.
 *
 *   ZZ1  ZigzagKnownVectors/Int32              — known encoding table for int32
 *   ZZ2  ZigzagKnownVectors/Int16              — known spot values for int16
 *   ZZ3  ZigzagKnownVectors/Int8               — known spot values for int8
 *   ZZ4  ZigzagRoundTrip/Exhaustive8Bit        — exhaustive decode(encode(x))==x for int8
 *   ZZ5  ZigzagRoundTrip/Exhaustive16Bit       — exhaustive decode(encode(x))==x for int16
 *   ZZ6  ZigzagRoundTrip/Sampled32Bit          — sampled decode(encode(x))==x for int32
 *   ZZ7  ZigzagRoundTrip/Sampled64Bit          — sampled decode(encode(x))==x for int64
 *   ZZ8  ZigzagProperties/ZeroMapsToZero       — encode(0)==0 for all widths
 *   ZZ9  ZigzagProperties/MonotoneMagnitude32  — encode(k)<encode(k+1) for k≥0
 *   ZZ10 ZigzagAliases/AgreePrimaryTemplate    — Zigzag8/16/32/64 match primary template
 */

#include <gtest/gtest.h>
#include "transforms/zigzag/zigzag.h"

#include <cstdint>
#include <limits>
#include <vector>

using namespace fz;

// ─────────────────────────────────────────────────────────────────────────────
// ZZ1: KnownVectors/Int32 — known encoding table for int32 values
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

// ─────────────────────────────────────────────────────────────────────────────
// ZZ2: KnownVectors/Int16 — known spot values for int16 zigzag encoding
// ─────────────────────────────────────────────────────────────────────────────
TEST(ZigzagKnownVectors, Int16) {
    using ZZ = Zigzag<int16_t>;

    EXPECT_EQ(ZZ::encode(int16_t( 0)), uint16_t(0));
    EXPECT_EQ(ZZ::encode(int16_t(-1)), uint16_t(1));
    EXPECT_EQ(ZZ::encode(int16_t( 1)), uint16_t(2));
    EXPECT_EQ(ZZ::encode(int16_t(-2)), uint16_t(3));
    EXPECT_EQ(ZZ::encode(int16_t( 2)), uint16_t(4));
}

// ─────────────────────────────────────────────────────────────────────────────
// ZZ3: KnownVectors/Int8 — known spot values for int8 zigzag encoding
// ─────────────────────────────────────────────────────────────────────────────
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
// ZZ4: RoundTrip/Exhaustive8Bit — decode(encode(x))==x for all int8 values
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

// ─────────────────────────────────────────────────────────────────────────────
// ZZ5: RoundTrip/Exhaustive16Bit — decode(encode(x))==x for all int16 values
// ─────────────────────────────────────────────────────────────────────────────
TEST(ZigzagRoundTrip, Exhaustive16Bit) {
    using ZZ = Zigzag<int16_t>;
    for (int i = std::numeric_limits<int16_t>::min();
         i    <= std::numeric_limits<int16_t>::max(); ++i) {
        int16_t x = static_cast<int16_t>(i);
        EXPECT_EQ(ZZ::decode(ZZ::encode(x)), x)
            << "round-trip failed for x=" << i;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ZZ6: RoundTrip/Sampled32Bit — decode(encode(x))==x sampled across int32 range
// ─────────────────────────────────────────────────────────────────────────────
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

// ─────────────────────────────────────────────────────────────────────────────
// ZZ7: RoundTrip/Sampled64Bit — decode(encode(x))==x for int64 extremes and samples
// ─────────────────────────────────────────────────────────────────────────────
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
// ZZ8: Properties/ZeroMapsToZero — encode(0)==0 for all widths
// ─────────────────────────────────────────────────────────────────────────────

TEST(ZigzagProperties, ZeroMapsToZero) {
    EXPECT_EQ(Zigzag<int8_t>::encode(0),  uint8_t(0));
    EXPECT_EQ(Zigzag<int16_t>::encode(0), uint16_t(0));
    EXPECT_EQ(Zigzag<int32_t>::encode(0), uint32_t(0));
    EXPECT_EQ(Zigzag<int64_t>::encode(0), uint64_t(0));
}

// ─────────────────────────────────────────────────────────────────────────────
// ZZ9: Properties/MonotoneMagnitude32 — encode(k)<encode(k+1) for k≥0, both signs
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
// ZZ10: Aliases/AgreePrimaryTemplate — Zigzag8/16/32/64 produce same results as Zigzag<T>
// ─────────────────────────────────────────────────────────────────────────────

TEST(ZigzagAliases, AgreePrimaryTemplate) {
    EXPECT_EQ(Zigzag8::encode(int8_t(-1)),   Zigzag<int8_t>::encode(int8_t(-1)));
    EXPECT_EQ(Zigzag16::encode(int16_t(-1)), Zigzag<int16_t>::encode(int16_t(-1)));
    EXPECT_EQ(Zigzag32::encode(int32_t(-1)), Zigzag<int32_t>::encode(int32_t(-1)));
    EXPECT_EQ(Zigzag64::encode(int64_t(-1)), Zigzag<int64_t>::encode(int64_t(-1)));
}
