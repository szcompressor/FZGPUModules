#pragma once

/**
 * @file adaptive_bitpack_oracle.cuh
 * @brief Exact, device-inline encoding quotes shared by staged and fused selectors.
 */

#include <cstdint>

namespace fz::adaptive_bitpack_oracle {

struct PlainFixedRateDecision {
    uint8_t rate = 0;
};

struct PlainFixedRateQuote {
    uint32_t payload_bytes = 0;
    PlainFixedRateDecision decision{};
};

struct AdaptiveFixedRateDecision {
    uint8_t rate = 0;
    uint8_t selector = 0; ///< bit0=outlier; bits1-2=outlier bytes minus one.
};

struct AdaptiveFixedRateQuote {
    uint32_t payload_bytes = 0;
    AdaptiveFixedRateDecision decision{};
};

/// Exact payload emitted by plain AdaptiveBitpack for one coder unit: an all-zero
/// unit emits nothing; otherwise it emits one sign word and `rate` plane words.
__device__ __forceinline__ PlainFixedRateQuote quotePlainFixedRate(
    uint32_t max_magnitude, uint32_t word_bytes)
{
    const uint8_t rate = max_magnitude
        ? static_cast<uint8_t>(32 - __clz(max_magnitude))
        : uint8_t{0};
    PlainFixedRateQuote q;
    q.decision.rate = rate;
    q.payload_bytes = rate > 0
        ? word_bytes * (static_cast<uint32_t>(rate) + 1u)
        : 0u;
    return q;
}

/// Exact plain-versus-element0-outlier choice used by AdaptiveBitpack's outlier
/// mode. Ties select plain, matching the staged and warp-register encoders.
__device__ __forceinline__ AdaptiveFixedRateQuote quoteAdaptiveFixedRate(
    uint32_t max_magnitude, uint32_t max_rest_magnitude,
    uint32_t first_magnitude, uint32_t word_bytes)
{
    const PlainFixedRateQuote plain = quotePlainFixedRate(max_magnitude, word_bytes);
    const uint8_t rest_rate = max_rest_magnitude
        ? static_cast<uint8_t>(32 - __clz(max_rest_magnitude))
        : uint8_t{0};
    const uint32_t outlier_bytes = first_magnitude
        ? static_cast<uint32_t>((32 - __clz(first_magnitude) + 7) / 8)
        : 0u;
    const uint32_t outlier_payload = outlier_bytes +
        (rest_rate > 0
            ? word_bytes * (static_cast<uint32_t>(rest_rate) + 1u)
            : word_bytes);

    AdaptiveFixedRateQuote q;
    if (plain.payload_bytes <= outlier_payload) {
        q.payload_bytes = plain.payload_bytes;
        q.decision.rate = plain.decision.rate;
        q.decision.selector = 0;
    } else {
        q.payload_bytes = outlier_payload;
        q.decision.rate = rest_rate;
        // outlier_payload can beat plain only when outlier_bytes is nonzero.
        q.decision.selector = static_cast<uint8_t>(
            1u | ((outlier_bytes - 1u) << 1u));
    }
    return q;
}

} // namespace fz::adaptive_bitpack_oracle
