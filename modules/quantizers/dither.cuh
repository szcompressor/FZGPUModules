#pragma once

/**
 * @file modules/quantizers/dither.cuh
 * @brief Deterministic, stateless per-element dither offset for "_R"-style
 *        quantizer reconstruction (LC's QUANT_*_R family).
 *
 * A deterministic quantizer always reconstructs every value in a given bin to
 * the same point (e.g. the bin center). That makes the reconstruction error
 * correlated with the signal — smooth input regions produce smooth, structured
 * error (visible banding/contouring), which is worse for downstream spectral
 * analysis than the same amount of error spread out as noise. Dithering fixes
 * this by reconstructing to a pseudo-random point within the bin instead of
 * always the same point — same worst-case error-bound guarantee, decorrelated
 * error.
 *
 * The randomness must be a *pure function* of (element index, seed) — no host
 * RNG state, no sequential dependency — so that:
 *   - encode (which must verify the dithered reconstruction stays within the
 *     error bound for THIS element, escalating to a lossless outlier if not)
 *     and decode (which must reproduce the identical offset with no access to
 *     the original value) always agree.
 *   - it is safe under arbitrary launch configurations and CUDA Graph capture.
 *
 * Usage:
 *   float offset = fz::ditherUnit(i, seed) * abs_eb;  // offset in [-abs_eb, abs_eb)
 *   float recon  = bin_center + offset;
 */

#include <cstdint>

// nvcc defines __CUDACC__; hipcc/amdclang++ defines __HIPCC__ instead. Without
// the second arm these helpers stay host-only under HIP and every __global__
// caller fails to resolve them.
#if defined(__CUDACC__) || defined(__HIPCC__)
#  define FZ_HOST_DEVICE __host__ __device__
#else
#  define FZ_HOST_DEVICE
#endif

namespace fz {

/**
 * SplitMix64 finalizer (Vigna) — single-pass bit mixer with good avalanche,
 * no lookup tables, no state. Used purely as a hash, not a sequential RNG.
 */
FZ_HOST_DEVICE inline uint64_t ditherHash64(uint64_t x) noexcept {
    x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ULL;
    x ^= x >> 27; x *= 0x94D049BB133111EBULL;
    x ^= x >> 31;
    return x;
}

/**
 * Deterministic dither value for element `index` given `seed`.
 *
 * Returns a value uniformly distributed in [-1, 1), reproducible for the same
 * (index, seed) pair on both host and device, in any launch configuration.
 *
 * @param index  Global element index (must match between encode and decode).
 * @param seed   Stage-level seed (persisted in the serialized header so
 *               decode can reproduce it without recomputation).
 */
FZ_HOST_DEVICE inline float ditherUnit(size_t index, uint64_t seed) noexcept {
    uint64_t mixed = (static_cast<uint64_t>(index) * 0x9E3779B97F4A7C15ULL) ^ seed;
    uint64_t h = ditherHash64(mixed);
    // Top 24 bits -> uniform integer in [0, 2^24), then map to [-1, 1).
    uint32_t top24 = static_cast<uint32_t>(h >> 40);
    float u01 = static_cast<float>(top24) * (1.0f / 16777216.0f);  // 1 / 2^24
    return u01 * 2.0f - 1.0f;
}

} // namespace fz

#undef FZ_HOST_DEVICE
