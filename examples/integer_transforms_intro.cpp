/**
 * examples/integer_transforms_intro.cpp
 *
 * Side-by-side comparison of three signed -> unsigned integer transforms:
 *   - Zigzag        (see modules/transforms/zigzag/zigzag.h)
 *   - Negabinary    (see modules/transforms/negabinary/negabinary.h)
 *   - Sign-magnitude ("TCMS" read literally: magnitude bits + a trailing sign bit)
 *
 * This example does NOT include the library headers above -- it is fully
 * self-contained and re-derives each algorithm from its definition so the
 * three can be read and compared side by side in one file:
 *
 *   Zigzag:         u = 2*|x|, minus 1 if x is negative.
 *                   Interleaves codes by magnitude: 0,-1,1,-2,2,-3,3,...
 *                   -> codes: 0, 1, 2, 3, 4, 5, 6, ...
 *
 *   Negabinary:     repeatedly divide by -2, collecting remainders, exactly
 *                   the way you would convert a number to base 10 or base 2
 *                   -- just with a negative radix. No separate sign bit is
 *                   needed: the sign falls out of which digit positions are
 *                   odd/even powers of -2. This is the "textbook" long-
 *                   division form; the production NegabinaryStage uses an
 *                   equivalent constant-time XOR-mask formula for the same
 *                   result.
 *
 *   Sign-magnitude: u = (|x| << 1) | sign_bit, i.e. literally the magnitude
 *                   bits with the sign bit tacked on at the end (the low
 *                   bit). -> codes: 0(+), 2(+),3(-), 4(+),5(-), ...
 *
 * Zigzag and sign-magnitude look similar (both are magnitude*2 plus a
 * sign-ish bit) but they are NOT the same mapping: zigzag numbers negatives
 * *below* the positive of the same magnitude (-1 -> 1, 1 -> 2), while
 * sign-magnitude always numbers the positive of a magnitude before the
 * negative (1 -> 2, -1 -> 3). See the printed table below -- the "code"
 * columns diverge for every nonzero value.
 *
 * ON NAMING: "TCMS" (Two's-Complement to Magnitude-Sign) reads, on paper,
 * like the sign-magnitude scheme above. But the LC framework this codebase
 * ports components from (Burtscher et al., github.com/burtscher/LC-framework,
 * `components/include/d_TCMS.h`) names its zigzag-equivalent bit trick
 * "TCMS": `out = (data << 1) ^ (signed(data) >> (bits-1))`, byte-for-byte
 * the zigzag formula above, applied at 1/2/4/8-byte word granularity. So
 * `ZigzagStage::setByteTransparent(true)` (see modules/transforms/zigzag/
 * zigzag_stage.h and memory/lc_lossless_stages.md) really is a bit-exact
 * port of LC's TCMS component -- confirmed against the vendored source, not
 * a naming bug. The "sign-magnitude" transform in this example is a
 * genuinely different, non-LC scheme; it does not need a stage of its own
 * unless you want raw magnitude/sign separation for some other reason (e.g.
 * bit-plane experiments), and it should NOT be called "TCMS" in this
 * codebase since that name is already correctly claimed by zigzag.
 *
 * No GPU, no Pipeline -- plain host C++ over a toy set of small integers.
 *
 * Usage:
 *   ./build/bin/examples/integer_transforms_intro
 *
 * Build:
 *   cmake --preset release -DBUILD_EXAMPLES=ON && cmake --build build/release -j$(nproc)
 *   Binary: build/release/bin/examples/integer_transforms_intro
 */

#include <bitset>
#include <cstdint>
#include <cstdio>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────
// Zigzag: u = 2*|x|, minus 1 if x is negative.
// ─────────────────────────────────────────────────────────────────────────
uint32_t zigzagEncode(int32_t x) {
    uint32_t mag = (x < 0) ? static_cast<uint32_t>(-static_cast<int64_t>(x))
                           : static_cast<uint32_t>(x);
    uint32_t u = mag * 2u;
    if (x < 0) u -= 1u;
    return u;
}

int32_t zigzagDecode(uint32_t u) {
    if (u & 1u) {
        // odd code -> came from a negative value
        return -static_cast<int32_t>((u + 1u) / 2u);
    }
    return static_cast<int32_t>(u / 2u);
}

// ─────────────────────────────────────────────────────────────────────────
// Negabinary: convert to base -2 by repeated division, exactly like
// converting to any other base -- just with a negative radix.
//
// At each step we need a remainder r in {0, 1} such that (n - r) is evenly
// divisible by -2. C's `%` can return a negative remainder for negative n,
// so we normalize it into {0, 1} before dividing.
// ─────────────────────────────────────────────────────────────────────────
uint32_t negabinaryEncode(int32_t x) {
    int64_t n = x;
    uint32_t digits = 0;
    int bit = 0;
    while (n != 0) {
        int64_t r = n % 2;
        if (r < 0) r += 2;            // normalize remainder into {0, 1}
        digits |= static_cast<uint32_t>(r) << bit;
        n = (n - r) / -2;             // strip the digit, flip sign each step
        ++bit;
    }
    return digits;
}

int32_t negabinaryDecode(uint32_t digits) {
    int64_t n = 0;
    int64_t place_value = 1;          // (-2)^0, (-2)^1, (-2)^2, ...
    for (int bit = 0; bit < 32; ++bit) {
        if ((digits >> bit) & 1u) n += place_value;
        place_value *= -2;
    }
    return static_cast<int32_t>(n);
}

// ─────────────────────────────────────────────────────────────────────────
// Sign-magnitude: magnitude bits, then the sign bit appended as the new low
// bit. NOT the same thing as the LC framework's "TCMS" component -- see the
// file header comment above. Named signMagnitude*, not tcms*, on purpose.
// ─────────────────────────────────────────────────────────────────────────
uint32_t signMagnitudeEncode(int32_t x) {
    uint32_t mag  = (x < 0) ? static_cast<uint32_t>(-static_cast<int64_t>(x))
                            : static_cast<uint32_t>(x);
    uint32_t sign = (x < 0) ? 1u : 0u;
    return (mag << 1) | sign;
}

int32_t signMagnitudeDecode(uint32_t u) {
    uint32_t sign = u & 1u;
    uint32_t mag  = u >> 1;
    return sign ? -static_cast<int32_t>(mag) : static_cast<int32_t>(mag);
}

// ─────────────────────────────────────────────────────────────────────────

int main() {
    const std::vector<int32_t> data = {0, 1, -1, 5, -5, 42, -42, 100, -100, 127, -128};

    std::printf("%8s %10s | %10s %16s | %10s %20s | %10s %16s | %s\n",
                "value", "value(bin)", "zigzag", "zigzag(bin)", "negabin",
                "negabinary(bin)", "sign-mag", "sign-mag(bin)", "roundtrip");
    std::printf("---------------------+----------------------------------+"
                "------------------------------------+"
                "---------------------------------+----------\n");

    bool all_ok = true;
    for (int32_t x : data) {
        const uint32_t zz = zigzagEncode(x);
        const uint32_t nb = negabinaryEncode(x);
        const uint32_t sm = signMagnitudeEncode(x);

        const bool zz_ok = zigzagDecode(zz) == x;
        const bool nb_ok = negabinaryDecode(nb) == x;
        const bool sm_ok = signMagnitudeDecode(sm) == x;
        const bool ok = zz_ok && nb_ok && sm_ok;
        all_ok &= ok;

        // Original value's own two's-complement bit pattern (8-bit -- every
        // value in the toy set fits int8_t), for direct visual comparison
        // against the transformed codes below.
        const auto value_bits = static_cast<uint8_t>(static_cast<int8_t>(x));

        std::printf("%8d %10s | %10u %16s | %10u %20s | %10u %16s | %s\n", x,
                    std::bitset<8>(value_bits).to_string().c_str(), zz,
                    std::bitset<16>(zz).to_string().c_str(), nb,
                    std::bitset<20>(nb).to_string().c_str(), sm,
                    std::bitset<16>(sm).to_string().c_str(), ok ? "OK" : "FAIL");
    }

    std::printf("\nAll round-trips %s.\n", all_ok ? "passed" : "FAILED");

    std::printf(
        "\nNote how 1 and -1 land on different codes in zigzag (2, 1) vs "
        "sign-magnitude (2, 3):\n"
        "zigzag orders the negative of a magnitude before the positive; "
        "sign-magnitude always\n"
        "keeps the positive first and appends the sign bit afterwards. "
        "Negabinary needs no\n"
        "separate sign bit at all -- sign is encoded by which digit "
        "positions (even/odd\n"
        "powers of -2) are set. NOTE: this codebase's LC-framework port "
        "names its zigzag-\n"
        "equivalent bit trick \"TCMS\" -- see the file header comment for "
        "why sign-magnitude\n"
        "here is a different, non-LC transform despite the acronym.\n");

    return all_ok ? 0 : 1;
}
