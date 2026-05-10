/**
 * tests/stages/test_negabinary.cpp
 *
 * Host-only unit tests for fz::Negabinary<T> (transforms/negabinary/negabinary.h).
 * No CUDA device required — encode/decode are __host__ __device__ functions.
 *
 *   NB1   NegabinaryKnownVectors/Int32Zero      — encode(0)==0, decode(0)==0
 *   NB2   NegabinaryKnownVectors/Int32One        — encode(1)==1 (base -2 identity)
 *   NB3   NegabinaryKnownVectors/Int32NegOne     — encode(-1)==3 (known bit pattern)
 *   NB4   NegabinaryKnownVectors/Int32Two        — encode(2)==6
 *   NB5   NegabinaryKnownVectors/Int32NegTwo     — encode(-2)==2
 *   NB6   NegabinaryRoundTrip/Exhaustive8Bit     — decode(encode(x))==x for all int8
 *   NB7   NegabinaryRoundTrip/Exhaustive16Bit    — decode(encode(x))==x for all int16
 *   NB8   NegabinaryRoundTrip/Sampled32Bit       — sampled round-trip for int32
 *   NB9   NegabinaryRoundTrip/Sampled64Bit       — sampled round-trip for int64 extremes
 *   NB10  NegabinaryProperties/ZeroMapsToZero    — encode(0)==0 for all widths
 *   NB11  NegabinaryAliases/AgreePrimaryTemplate — Negabinary8/16/32/64 match primary template
 */

#include <gtest/gtest.h>
#include "transforms/negabinary/negabinary.h"

#include <cstdint>
#include <limits>
#include <vector>

using namespace fz;

// ─────────────────────────────────────────────────────────────────────────────
// NB1-NB5: KnownVectors — spot values verified against the negabinary formula
//   (n + 0xAAAAAAAA) ^ 0xAAAAAAAA for int32_t; encode then decode recovers n.
// ─────────────────────────────────────────────────────────────────────────────

TEST(NegabinaryKnownVectors, Int32Zero) {
    EXPECT_EQ(Negabinary<int32_t>::encode(0), 0u);
    EXPECT_EQ(Negabinary<int32_t>::decode(0u), 0);
}

TEST(NegabinaryKnownVectors, Int32One) {
    // 1 in negabinary is still 1
    uint32_t u = Negabinary<int32_t>::encode(1);
    EXPECT_EQ(Negabinary<int32_t>::decode(u), 1);
    EXPECT_EQ(u, 1u);
}

TEST(NegabinaryKnownVectors, Int32NegOne) {
    // −1 in base −2: …1100 in bits from the standard negabinary table,
    // but we verify via the round-trip identity because the bit pattern
    // is implementation-defined by the formula.
    uint32_t u = Negabinary<int32_t>::encode(-1);
    EXPECT_EQ(Negabinary<int32_t>::decode(u), -1);
    // The formula says: u = (-1 + 0xAAAAAAAA) ^ 0xAAAAAAAA
    //                     = (0xAAAAAAA9)      ^ 0xAAAAAAAA = 0x00000003
    EXPECT_EQ(u, 3u);
}

TEST(NegabinaryKnownVectors, Int32Two) {
    // 2 in negabinary (base -2): 110 ((-2)^2 + (-2)^1 = 4 - 2 = 2), pattern = 0b110 = 6
    uint32_t u = Negabinary<int32_t>::encode(2);
    EXPECT_EQ(Negabinary<int32_t>::decode(u), 2);
    EXPECT_EQ(u, 6u);
}

TEST(NegabinaryKnownVectors, Int32NegTwo) {
    // -2 in negabinary: 1010 = 4+0-2+0 =2  — wait, let me just verify round-trip.
    // (-2 + 0xAAAAAAAA) ^ 0xAAAAAAAA = (0xAAAAAAA8) ^ 0xAAAAAAAA = 0x00000002
    uint32_t u = Negabinary<int32_t>::encode(-2);
    EXPECT_EQ(Negabinary<int32_t>::decode(u), -2);
    EXPECT_EQ(u, 2u);
}

// ─────────────────────────────────────────────────────────────────────────────
// NB6: RoundTrip/Exhaustive8Bit — decode(encode(x))==x for all int8 values
// ─────────────────────────────────────────────────────────────────────────────

TEST(NegabinaryRoundTrip, Exhaustive8Bit) {
    using NB = Negabinary<int8_t>;
    for (int i = std::numeric_limits<int8_t>::min();
         i    <= std::numeric_limits<int8_t>::max(); ++i) {
        int8_t n = static_cast<int8_t>(i);
        EXPECT_EQ(NB::decode(NB::encode(n)), n)
            << "round-trip failed for n=" << static_cast<int>(n);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// NB7: RoundTrip/Exhaustive16Bit — decode(encode(x))==x for all int16 values
// ─────────────────────────────────────────────────────────────────────────────
TEST(NegabinaryRoundTrip, Exhaustive16Bit) {
    using NB = Negabinary<int16_t>;
    for (int i = std::numeric_limits<int16_t>::min();
         i    <= std::numeric_limits<int16_t>::max(); ++i) {
        int16_t n = static_cast<int16_t>(i);
        EXPECT_EQ(NB::decode(NB::encode(n)), n)
            << "round-trip failed for n=" << i;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// NB8: RoundTrip/Sampled32Bit — decode(encode(x))==x sampled across int32 range
// ─────────────────────────────────────────────────────────────────────────────
TEST(NegabinaryRoundTrip, Sampled32Bit) {
    using NB = Negabinary<int32_t>;

    std::vector<int32_t> cases = {
        0, 1, -1, 2, -2, 127, -127, 128, -128,
        32767, -32767, 32768, -32768,
        1'000'000, -1'000'000,
        std::numeric_limits<int32_t>::max(),
        std::numeric_limits<int32_t>::min(),
        std::numeric_limits<int32_t>::max() - 1,
        std::numeric_limits<int32_t>::min() + 1,
    };
    for (int64_t v = std::numeric_limits<int32_t>::min();
         v        <= std::numeric_limits<int32_t>::max();
         v += 65537) {
        cases.push_back(static_cast<int32_t>(v));
    }

    for (int32_t n : cases) {
        EXPECT_EQ(NB::decode(NB::encode(n)), n)
            << "round-trip failed for n=" << n;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// NB9: RoundTrip/Sampled64Bit — decode(encode(x))==x for int64 extremes and samples
// ─────────────────────────────────────────────────────────────────────────────
TEST(NegabinaryRoundTrip, Sampled64Bit) {
    using NB = Negabinary<int64_t>;

    std::vector<int64_t> cases = {
        0LL, 1LL, -1LL, 2LL, -2LL,
        INT64_MAX, INT64_MIN,
        INT64_MAX - 1LL, INT64_MIN + 1LL,
        static_cast<int64_t>(1) << 32,
        -(static_cast<int64_t>(1) << 32),
        static_cast<int64_t>(1e15),
        -static_cast<int64_t>(1e15),
    };

    for (int64_t n : cases) {
        EXPECT_EQ(NB::decode(NB::encode(n)), n)
            << "round-trip failed for n=" << n;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// NB10: Properties/ZeroMapsToZero — encode(0)==0 for all integer widths
// ─────────────────────────────────────────────────────────────────────────────

TEST(NegabinaryProperties, ZeroMapsToZero) {
    EXPECT_EQ(Negabinary<int8_t>::encode(0),  uint8_t(0));
    EXPECT_EQ(Negabinary<int16_t>::encode(0), uint16_t(0));
    EXPECT_EQ(Negabinary<int32_t>::encode(0), uint32_t(0));
    EXPECT_EQ(Negabinary<int64_t>::encode(0), uint64_t(0));
}

// ─────────────────────────────────────────────────────────────────────────────
// NB11: Aliases/AgreePrimaryTemplate — Negabinary8/16/32/64 match primary template
// ─────────────────────────────────────────────────────────────────────────────

TEST(NegabinaryAliases, AgreePrimaryTemplate) {
    EXPECT_EQ(Negabinary8::encode(int8_t(-1)),   Negabinary<int8_t>::encode(int8_t(-1)));
    EXPECT_EQ(Negabinary16::encode(int16_t(-1)), Negabinary<int16_t>::encode(int16_t(-1)));
    EXPECT_EQ(Negabinary32::encode(int32_t(-1)), Negabinary<int32_t>::encode(int32_t(-1)));
    EXPECT_EQ(Negabinary64::encode(int64_t(-1)), Negabinary<int64_t>::encode(int64_t(-1)));
}
