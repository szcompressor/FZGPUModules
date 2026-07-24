#pragma once

/**
 * @file include/backend/warp.h
 * @brief Backend-neutral warp/wavefront shuffle and ballot primitives.
 *
 * CUDA warps are 32 lanes; AMD CDNA wavefronts (MI100/gfx908) are 64. Code
 * written for CUDA's `__shfl_*_sync`/`__ballot_sync` falls into two failure
 * modes when ported naively to HIP:
 *
 *  1. **Mask width.** HIP's mask parameter is template-deduced and requires
 *     an 8-byte integral type (`static_assert(sizeof(MaskT) == 8)`), so a
 *     bare `0xffffffffu` (4 bytes) is a hard compile error, not a silent
 *     truncation. `kFullMask` below is the right width for each backend.
 *
 *  2. **Implicit `width`.** `__shfl_*_sync`'s `width` parameter defaults to
 *     `warpSize`, which is 32 under CUDA but 64 under HIP. Code that omits
 *     `width` and relies on that default silently changes which lanes
 *     participate — verified on MI100 hardware: a `__shfl_down_sync` with a
 *     32-lane grouping intent but no explicit width reads across the
 *     wrong lane boundary once compiled for a 64-wide wavefront. The
 *     `width` parameter is *required* here (no default) so this can't
 *     happen silently at a call site again.
 *
 * `shflUp/Down/Xor/plain` are width-only — the mask is functionally inert
 * for these under HIP (it only feeds a debug-only convergence assert,
 * compiled out under `-DNDEBUG`), so `kFullMask` is always correct to pass
 * regardless of the caller's actual lane grouping.
 *
 * `ballotSync32`/`anySync32` are different: cub/CUDA code that packs a
 * `__ballot_sync` result into a 32-bit wire-format bitmask assumes the
 * *lowest* 32 bits are the answer. Under HIP that's only true for the lower
 * half of the wavefront — verified on MI100 hardware that the upper 32
 * lanes' ballot bits land at [32:63], not auto-normalized to [0:31]. These
 * two functions restrict to the caller's own 32-lane half (by hardware lane
 * id, not any software-derived index) and normalize the result back to
 * [0:31], so callers get exactly what CUDA's native 32-lane `__ballot_sync`
 * already returned.
 *
 * Requires `-DHIP_ENABLE_WARP_SYNC_BUILTINS` (set globally for the HIP
 * backend in the root CMakeLists.txt) — ROCm does not declare
 * `__shfl_*_sync`/`__ballot_sync`/`__any_sync` at all without it.
 */

#include "backend/api.h"

#include <cstdint>

namespace fz {
namespace backend {

#if defined(FZGMOD_BACKEND_HIP)
using warp_mask_t = unsigned long long;
inline constexpr warp_mask_t kFullMask = ~0ULL;
#else
using warp_mask_t = unsigned;
inline constexpr warp_mask_t kFullMask = 0xffffffffu;
#endif

template <typename T>
__device__ inline T shflUp(T val, unsigned delta, int width) {
    return __shfl_up_sync(kFullMask, val, delta, width);
}

template <typename T>
__device__ inline T shflDown(T val, unsigned delta, int width) {
    return __shfl_down_sync(kFullMask, val, delta, width);
}

template <typename T>
__device__ inline T shflXor(T val, int laneMask, int width) {
    return __shfl_xor_sync(kFullMask, val, laneMask, width);
}

template <typename T>
__device__ inline T shfl(T val, int srcLane, int width) {
    return __shfl_sync(kFullMask, val, srcLane, width);
}

/** 32-lane-sub-group ballot, normalized to bits [0:31] regardless of which
 *  hardware half of a 64-lane wavefront the calling lane sits in. */
__device__ inline uint32_t ballotSync32(int pred) {
#if defined(FZGMOD_BACKEND_HIP)
    bool upper = (__lane_id() & 32u) != 0u;
    warp_mask_t mask = upper ? 0xFFFFFFFF00000000ULL : 0x00000000FFFFFFFFULL;
    unsigned long long b = __ballot_sync(mask, pred);
    return static_cast<uint32_t>(upper ? (b >> 32) : b);
#else
    return __ballot_sync(kFullMask, pred);
#endif
}

/** 32-lane-sub-group __any_sync — true if `pred` is true for any lane in the
 *  caller's own 32-lane half of the wavefront. */
__device__ inline bool anySync32(int pred) {
#if defined(FZGMOD_BACKEND_HIP)
    bool upper = (__lane_id() & 32u) != 0u;
    warp_mask_t mask = upper ? 0xFFFFFFFF00000000ULL : 0x00000000FFFFFFFFULL;
    return __any_sync(mask, pred) != 0;
#else
    return __any_sync(kFullMask, pred) != 0;
#endif
}

} // namespace backend
} // namespace fz
