// Per-block adaptive fixed-rate bit-plane packing kernels (cuSZp-style plain
// mode). One CUDA thread owns one logical data block — simple and verifiable;
// kernel fusion / warp-cooperative packing is left to the downstream compiler.

#include "coders/adaptive_bitpack/adaptive_bitpack_kernels.h"
#include "backend/warp.h"
#include "cuda_check.h"

#include <type_traits>

namespace fz {
namespace adaptive_bitpack {

// Two's-complement magnitude as an unsigned value (well-defined for INT_MIN).
template<typename T>
__device__ __forceinline__ typename std::make_unsigned<T>::type absU(T v) {
    using U = typename std::make_unsigned<T>::type;
    U uv = static_cast<U>(v);
    return (v < 0) ? static_cast<U>(~uv + static_cast<U>(1)) : uv;
}

__device__ __forceinline__ int bitWidth32(uint32_t x) {
    return x ? (32 - __clz(x)) : 0;
}

// Reconstruct a signed value from magnitude + sign (well-defined for INT_MIN).
template<typename T>
__device__ __forceinline__ T applySign(uint32_t mag, bool neg) {
    using U = typename std::make_unsigned<T>::type;
    U m = static_cast<U>(mag);
    return neg ? static_cast<T>(static_cast<U>(0) - m) : static_cast<T>(m);
}

// ── Encode pass A: rate byte + payload cost per block ───────────────────────
template<typename T>
__global__ void encode_rate_kernel(
    const T* __restrict__ in, size_t num_elements,
    uint32_t block_size, uint32_t word_bytes, size_t num_blocks,
    uint8_t* __restrict__ rate, uint32_t* __restrict__ cost)
{
    size_t b = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (b >= num_blocks) return;

    size_t start = b * block_size;
    size_t count = min(static_cast<size_t>(block_size), num_elements - start);

    uint32_t acc = 0;  // OR of all magnitudes → same top bit as the max
    for (size_t i = 0; i < count; ++i)
        acc |= static_cast<uint32_t>(absU<T>(in[start + i]));

    int r = bitWidth32(acc);
    rate[b] = static_cast<uint8_t>(r);
    cost[b] = (r > 0) ? word_bytes * (static_cast<uint32_t>(r) + 1u) : 0u;
}

// ── Encode pass B: pack sign + bit-planes ───────────────────────────────────
template<typename T>
__global__ void encode_pack_kernel(
    const T* __restrict__ in, size_t num_elements,
    uint32_t block_size, uint32_t word_bytes, size_t num_blocks,
    const uint8_t* __restrict__ rate, const uint32_t* __restrict__ offset,
    uint8_t* __restrict__ payload)
{
    size_t b = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (b >= num_blocks) return;

    int r = rate[b];
    if (r == 0) return;

    size_t start = b * block_size;
    size_t count = min(static_cast<size_t>(block_size), num_elements - start);
    uint8_t* base = payload + offset[b];

    for (uint32_t k = 0; k < word_bytes; ++k) {
        // sign byte
        uint8_t sgn = 0;
        for (int j = 0; j < 8; ++j) {
            uint32_t idx = k * 8u + j;
            if (idx < count && in[start + idx] < 0) sgn |= (1u << j);
        }
        base[k] = sgn;
        // bit-planes
        for (int p = 0; p < r; ++p) {
            uint8_t pl = 0;
            for (int j = 0; j < 8; ++j) {
                uint32_t idx = k * 8u + j;
                if (idx < count) {
                    uint32_t av = static_cast<uint32_t>(absU<T>(in[start + idx]));
                    pl |= static_cast<uint8_t>(((av >> p) & 1u) << j);
                }
            }
            base[word_bytes * (1u + p) + k] = pl;
        }
    }
}

// ── Decode pass A: cost per block from rate region ──────────────────────────
__global__ void decode_cost_kernel(
    const uint8_t* __restrict__ rate, uint32_t word_bytes, size_t num_blocks,
    uint32_t* __restrict__ cost)
{
    size_t b = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (b >= num_blocks) return;
    int r = rate[b];
    cost[b] = (r > 0) ? word_bytes * (static_cast<uint32_t>(r) + 1u) : 0u;
}

// ── Decode pass B: unpack each block ────────────────────────────────────────
template<typename T>
__global__ void decode_unpack_kernel(
    const uint8_t* __restrict__ rate, const uint32_t* __restrict__ offset,
    const uint8_t* __restrict__ payload,
    size_t num_elements, uint32_t block_size, uint32_t word_bytes,
    size_t num_blocks, T* __restrict__ out)
{
    using U = typename std::make_unsigned<T>::type;
    size_t b = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (b >= num_blocks) return;

    size_t start = b * block_size;
    size_t count = min(static_cast<size_t>(block_size), num_elements - start);
    int r = rate[b];

    if (r == 0) {
        for (size_t i = 0; i < count; ++i) out[start + i] = static_cast<T>(0);
        return;
    }

    const uint8_t* base = payload + offset[b];
    for (uint32_t k = 0; k < word_bytes; ++k) {
        uint8_t sgn = base[k];
        for (int j = 0; j < 8; ++j) {
            uint32_t idx = k * 8u + j;
            if (idx >= count) break;
            uint32_t av = 0;
            for (int p = 0; p < r; ++p)
                av |= ((static_cast<uint32_t>(base[word_bytes * (1u + p) + k]) >> j) & 1u) << p;
            U mag = static_cast<U>(av);
            T v = ((sgn >> j) & 1u)
                ? static_cast<T>(static_cast<U>(0) - mag)   // two's-complement negate
                : static_cast<T>(mag);
            out[start + idx] = v;
        }
    }
}

// ── Warp-cooperative plain-mode kernels ─────────────────────────────────────
// One warp (not one thread) owns one logical data block. Lane l, sub-index m
// (m = 0..ElemsPerLane-1) owns element `lane + 32*m` of the block. Byte k = 4m
// + (l/8), bit j = l%8 reproduces the scalar kernels' `bit j of byte k = elem
// 8k+j` layout exactly (substitute l=8q+j: 8(4m+q)+j = 32m+8q+j = 32m+l) — so
// the on-disk archive format is unchanged, only how it's computed. A
// __ballot_sync at fixed (p, m) yields a 32-bit mask whose byte b (bits
// 8b..8b+7) is exactly the scalar kernels' base[4m+b] for that plane.
// ElemsPerLane is a compile-time template parameter (not a runtime loop
// bound) specifically so the tiny per-lane v[]/av[] arrays register-allocate
// instead of spilling to local memory from dynamic indexing (the same issue
// that made native cuSZp3's absQuant[] spill-bound, per prior profiling).
// Only block_size 32 (ElemsPerLane=1, cuszp2) and 64 (ElemsPerLane=2, cuszp3)
// are shipped/tested; any other block_size falls back to the scalar kernels
// above, which are untouched.
template<typename T, int ElemsPerLane>
__global__ void encode_rate_kernel_warp(
    const T* __restrict__ in, size_t num_elements,
    uint32_t word_bytes, size_t num_blocks,
    uint8_t* __restrict__ rate, uint32_t* __restrict__ cost)
{
    const uint32_t lane          = threadIdx.x & 31u;
    const uint32_t warp_in_block = threadIdx.x >> 5;
    const uint32_t warps_per_cta = blockDim.x >> 5;
    const size_t b = static_cast<size_t>(blockIdx.x) * warps_per_cta + warp_in_block;
    if (b >= num_blocks) return;

    constexpr uint32_t block_size = static_cast<uint32_t>(ElemsPerLane) * 32u;
    const size_t start = b * block_size;
    const size_t count = min(static_cast<size_t>(block_size), num_elements - start);

    uint32_t acc = 0;
    #pragma unroll
    for (int m = 0; m < ElemsPerLane; ++m) {
        const size_t idx = static_cast<size_t>(lane) + 32u * m;
        if (idx < count) acc |= static_cast<uint32_t>(absU<T>(in[start + idx]));
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc |= fz::backend::shflXor(acc, off, 32);

    if (lane == 0) {
        const int r = bitWidth32(acc);
        rate[b] = static_cast<uint8_t>(r);
        cost[b] = (r > 0) ? word_bytes * (static_cast<uint32_t>(r) + 1u) : 0u;
    }
}

template<typename T, int ElemsPerLane>
__global__ void encode_pack_kernel_warp(
    const T* __restrict__ in, size_t num_elements,
    uint32_t word_bytes, size_t num_blocks,
    const uint8_t* __restrict__ rate, const uint32_t* __restrict__ offset,
    uint8_t* __restrict__ payload)
{
    const uint32_t lane          = threadIdx.x & 31u;
    const uint32_t warp_in_block = threadIdx.x >> 5;
    const uint32_t warps_per_cta = blockDim.x >> 5;
    const size_t b = static_cast<size_t>(blockIdx.x) * warps_per_cta + warp_in_block;
    if (b >= num_blocks) return;

    const int r = rate[b];
    if (r == 0) return;

    constexpr uint32_t block_size = static_cast<uint32_t>(ElemsPerLane) * 32u;
    const size_t start = b * block_size;
    const size_t count = min(static_cast<size_t>(block_size), num_elements - start);
    uint8_t* base = payload + offset[b];

    T v[ElemsPerLane];
    uint32_t av[ElemsPerLane];
    bool active[ElemsPerLane];
    #pragma unroll
    for (int m = 0; m < ElemsPerLane; ++m) {
        const size_t idx = static_cast<size_t>(lane) + 32u * m;
        active[m] = idx < count;
        v[m]  = active[m] ? in[start + idx] : static_cast<T>(0);
        av[m] = active[m] ? static_cast<uint32_t>(absU<T>(v[m])) : 0u;
    }

    #pragma unroll
    for (int m = 0; m < ElemsPerLane; ++m) {
        const uint32_t sign_mask = fz::backend::ballotSync32(active[m] && v[m] < 0);
        if (lane < 4) base[4u * m + lane] = static_cast<uint8_t>((sign_mask >> (8u * lane)) & 0xFFu);
    }
    for (int p = 0; p < r; ++p) {
        #pragma unroll
        for (int m = 0; m < ElemsPerLane; ++m) {
            const uint32_t plane_mask = fz::backend::ballotSync32(active[m] && ((av[m] >> p) & 1u));
            if (lane < 4)
                base[word_bytes * (1u + p) + 4u * m + lane] =
                    static_cast<uint8_t>((plane_mask >> (8u * lane)) & 0xFFu);
        }
    }
}

template<typename T, int ElemsPerLane>
__global__ void decode_unpack_kernel_warp(
    const uint8_t* __restrict__ rate, const uint32_t* __restrict__ offset,
    const uint8_t* __restrict__ payload,
    size_t num_elements, uint32_t word_bytes, size_t num_blocks,
    T* __restrict__ out)
{
    using U = typename std::make_unsigned<T>::type;
    const uint32_t lane          = threadIdx.x & 31u;
    const uint32_t warp_in_block = threadIdx.x >> 5;
    const uint32_t warps_per_cta = blockDim.x >> 5;
    const size_t b = static_cast<size_t>(blockIdx.x) * warps_per_cta + warp_in_block;
    if (b >= num_blocks) return;

    constexpr uint32_t block_size = static_cast<uint32_t>(ElemsPerLane) * 32u;
    const size_t start = b * block_size;
    const size_t count = min(static_cast<size_t>(block_size), num_elements - start);
    const int r = rate[b];

    if (r == 0) {
        #pragma unroll
        for (int m = 0; m < ElemsPerLane; ++m) {
            const size_t idx = static_cast<size_t>(lane) + 32u * m;
            if (idx < count) out[start + idx] = static_cast<T>(0);
        }
        return;
    }

    const uint8_t* base = payload + offset[b];
    const uint32_t q = lane >> 3;   // byte group 0..3 (matches k = 4m + q)
    const uint32_t j = lane & 7u;   // bit position within that byte

    #pragma unroll
    for (int m = 0; m < ElemsPerLane; ++m) {
        const size_t idx = static_cast<size_t>(lane) + 32u * m;
        if (idx >= count) continue;
        const uint32_t k = 4u * m + q;
        const uint8_t sgn = base[k];
        uint32_t av = 0;
        for (int p = 0; p < r; ++p)
            av |= ((static_cast<uint32_t>(base[word_bytes * (1u + p) + k]) >> j) & 1u) << p;
        out[start + idx] = applySign<T>(av, ((sgn >> j) & 1u) != 0);
    }
}

// ── Outlier-selection mode (cuSZp2) ─────────────────────────────────────────
// Metadata: 2 bytes per block — meta[2b]=rate, meta[2b+1]=sel
//   sel bit0 = is_outlier; if set, sel bits1-2 = outlier_byte_num - 1.

// Encode pass A: choose plain vs outlier per block; write 2-byte meta + cost.
template<typename T>
__global__ void encode_rate_outlier_kernel(
    const T* __restrict__ in, size_t num_elements,
    uint32_t block_size, uint32_t word_bytes, size_t num_blocks,
    uint8_t* __restrict__ meta, uint32_t* __restrict__ cost)
{
    size_t b = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (b >= num_blocks) return;

    size_t start = b * block_size;
    size_t count = min(static_cast<size_t>(block_size), num_elements - start);

    uint32_t acc_all = 0, acc_rest = 0;
    uint32_t mag0 = (count > 0) ? static_cast<uint32_t>(absU<T>(in[start])) : 0u;
    for (size_t i = 0; i < count; ++i) {
        uint32_t av = static_cast<uint32_t>(absU<T>(in[start + i]));
        acc_all |= av;
        if (i > 0) acc_rest |= av;
    }
    int fr_all  = bitWidth32(acc_all);
    int fr_rest = bitWidth32(acc_rest);
    uint32_t ob_bytes = static_cast<uint32_t>((bitWidth32(mag0) + 7) / 8);

    uint32_t cost_plain = (fr_all > 0) ? word_bytes * (fr_all + 1u) : 0u;
    uint32_t cost_out   = ob_bytes
                        + ((fr_rest > 0) ? word_bytes * (fr_rest + 1u) : word_bytes);

    if (cost_plain <= cost_out) {
        meta[2 * b]     = static_cast<uint8_t>(fr_all);
        meta[2 * b + 1] = 0;                 // plain
        cost[b]         = cost_plain;
    } else {
        meta[2 * b]     = static_cast<uint8_t>(fr_rest);
        meta[2 * b + 1] = static_cast<uint8_t>(1u | ((ob_bytes - 1u) << 1));
        cost[b]         = cost_out;
    }
}

// Encode pass B: pack each block (plain or outlier) at payload + offset[b].
template<typename T>
__global__ void encode_pack_outlier_kernel(
    const T* __restrict__ in, size_t num_elements,
    uint32_t block_size, uint32_t word_bytes, size_t num_blocks,
    const uint8_t* __restrict__ meta, const uint32_t* __restrict__ offset,
    uint8_t* __restrict__ payload)
{
    size_t b = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (b >= num_blocks) return;

    int     r       = meta[2 * b];
    uint8_t sel     = meta[2 * b + 1];
    bool    is_out  = (sel & 1u) != 0;
    size_t  start   = b * block_size;
    size_t  count   = min(static_cast<size_t>(block_size), num_elements - start);
    uint8_t* base   = payload + offset[b];

    if (!is_out) {
        // Plain: sign region + r planes over all elements (same as plain mode).
        if (r == 0) return;
        for (uint32_t k = 0; k < word_bytes; ++k) {
            uint8_t sgn = 0;
            for (int j = 0; j < 8; ++j) {
                uint32_t idx = k * 8u + j;
                if (idx < count && in[start + idx] < 0) sgn |= (1u << j);
            }
            base[k] = sgn;
            for (int p = 0; p < r; ++p) {
                uint8_t pl = 0;
                for (int j = 0; j < 8; ++j) {
                    uint32_t idx = k * 8u + j;
                    if (idx < count) {
                        uint32_t av = static_cast<uint32_t>(absU<T>(in[start + idx]));
                        pl |= static_cast<uint8_t>(((av >> p) & 1u) << j);
                    }
                }
                base[word_bytes * (1u + p) + k] = pl;
            }
        }
        return;
    }

    // Outlier: [ob_bytes elem0 magnitude LE][sign region][r planes for elems 1..]
    uint32_t ob_bytes = ((sel >> 1) & 3u) + 1u;
    uint32_t mag0 = static_cast<uint32_t>(absU<T>(in[start]));
    for (uint32_t k = 0; k < ob_bytes; ++k)
        base[k] = static_cast<uint8_t>((mag0 >> (8u * k)) & 0xffu);

    uint8_t* sign   = base + ob_bytes;
    uint8_t* planes = base + ob_bytes + word_bytes;
    for (uint32_t k = 0; k < word_bytes; ++k) {
        uint8_t sgn = 0;
        for (int j = 0; j < 8; ++j) {
            uint32_t idx = k * 8u + j;
            if (idx < count && in[start + idx] < 0) sgn |= (1u << j);
        }
        sign[k] = sgn;
        for (int p = 0; p < r; ++p) {
            uint8_t pl = 0;
            for (int j = 0; j < 8; ++j) {
                uint32_t idx = k * 8u + j;
                if (idx > 0 && idx < count) {  // element 0 lives in the outlier bytes
                    uint32_t av = static_cast<uint32_t>(absU<T>(in[start + idx]));
                    pl |= static_cast<uint8_t>(((av >> p) & 1u) << j);
                }
            }
            planes[word_bytes * p + k] = pl;
        }
    }
}

// Decode pass A: cost per block from the 2-byte metadata.
__global__ void decode_cost_outlier_kernel(
    const uint8_t* __restrict__ meta, uint32_t word_bytes, size_t num_blocks,
    uint32_t* __restrict__ cost)
{
    size_t b = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (b >= num_blocks) return;
    int     r      = meta[2 * b];
    uint8_t sel    = meta[2 * b + 1];
    if (sel & 1u) {
        uint32_t ob_bytes = ((sel >> 1) & 3u) + 1u;
        cost[b] = ob_bytes + ((r > 0) ? word_bytes * (r + 1u) : word_bytes);
    } else {
        cost[b] = (r > 0) ? word_bytes * (r + 1u) : 0u;
    }
}

// Decode pass B: unpack each block (plain or outlier).
template<typename T>
__global__ void decode_unpack_outlier_kernel(
    const uint8_t* __restrict__ meta, const uint32_t* __restrict__ offset,
    const uint8_t* __restrict__ payload,
    size_t num_elements, uint32_t block_size, uint32_t word_bytes,
    size_t num_blocks, T* __restrict__ out)
{
    size_t b = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (b >= num_blocks) return;

    int     r      = meta[2 * b];
    uint8_t sel    = meta[2 * b + 1];
    bool    is_out = (sel & 1u) != 0;
    size_t  start  = b * block_size;
    size_t  count  = min(static_cast<size_t>(block_size), num_elements - start);
    const uint8_t* base = payload + offset[b];

    if (!is_out) {
        if (r == 0) {
            for (size_t i = 0; i < count; ++i) out[start + i] = static_cast<T>(0);
            return;
        }
        for (uint32_t k = 0; k < word_bytes; ++k) {
            uint8_t sgn = base[k];
            for (int j = 0; j < 8; ++j) {
                uint32_t idx = k * 8u + j;
                if (idx >= count) break;
                uint32_t av = 0;
                for (int p = 0; p < r; ++p)
                    av |= ((static_cast<uint32_t>(base[word_bytes * (1u + p) + k]) >> j) & 1u) << p;
                out[start + idx] = applySign<T>(av, ((sgn >> j) & 1u) != 0);
            }
        }
        return;
    }

    // Outlier block.
    uint32_t ob_bytes = ((sel >> 1) & 3u) + 1u;
    uint32_t mag0 = 0;
    for (uint32_t k = 0; k < ob_bytes; ++k)
        mag0 |= static_cast<uint32_t>(base[k]) << (8u * k);
    const uint8_t* sign   = base + ob_bytes;
    const uint8_t* planes = base + ob_bytes + word_bytes;

    // element 0 (sign bit 0 of sign byte 0)
    out[start] = applySign<T>(mag0, (sign[0] & 1u) != 0);

    // elements 1 .. count-1
    for (uint32_t k = 0; k < word_bytes; ++k) {
        uint8_t sgn = sign[k];
        for (int j = 0; j < 8; ++j) {
            uint32_t idx = k * 8u + j;
            if (idx == 0) continue;       // handled above
            if (idx >= count) break;
            uint32_t av = 0;
            for (int p = 0; p < r; ++p)
                av |= ((static_cast<uint32_t>(planes[word_bytes * p + k]) >> j) & 1u) << p;
            out[start + idx] = applySign<T>(av, ((sgn >> j) & 1u) != 0);
        }
    }
}

// ── Warp-cooperative outlier-selection kernels ──────────────────────────────
// Same lane mapping as the plain-mode warp kernels above. Element 0 (lane 0,
// m=0) is excluded from the "rest" OR-accumulator and from the bit-plane
// ballot predicate (idx>0) but included in the sign-region ballot (matches
// the scalar kernels: sign[0] bit 0 legitimately holds element 0's sign,
// which decode reads directly rather than via the `mag0`/outlier-byte path).
template<typename T, int ElemsPerLane>
__global__ void encode_rate_outlier_kernel_warp(
    const T* __restrict__ in, size_t num_elements,
    uint32_t word_bytes, size_t num_blocks,
    uint8_t* __restrict__ meta, uint32_t* __restrict__ cost)
{
    const uint32_t lane          = threadIdx.x & 31u;
    const uint32_t warp_in_block = threadIdx.x >> 5;
    const uint32_t warps_per_cta = blockDim.x >> 5;
    const size_t b = static_cast<size_t>(blockIdx.x) * warps_per_cta + warp_in_block;
    if (b >= num_blocks) return;

    constexpr uint32_t block_size = static_cast<uint32_t>(ElemsPerLane) * 32u;
    const size_t start = b * block_size;
    const size_t count = min(static_cast<size_t>(block_size), num_elements - start);

    uint32_t acc_all = 0, acc_rest = 0;
    #pragma unroll
    for (int m = 0; m < ElemsPerLane; ++m) {
        const size_t idx = static_cast<size_t>(lane) + 32u * m;
        if (idx < count) {
            const uint32_t av = static_cast<uint32_t>(absU<T>(in[start + idx]));
            acc_all |= av;
            if (idx > 0) acc_rest |= av;
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc_all  |= fz::backend::shflXor(acc_all, off, 32);
        acc_rest |= fz::backend::shflXor(acc_rest, off, 32);
    }

    if (lane == 0) {
        const uint32_t mag0 = (count > 0) ? static_cast<uint32_t>(absU<T>(in[start])) : 0u;
        const int fr_all  = bitWidth32(acc_all);
        const int fr_rest = bitWidth32(acc_rest);
        const uint32_t ob_bytes = static_cast<uint32_t>((bitWidth32(mag0) + 7) / 8);
        const uint32_t cost_plain = (fr_all > 0) ? word_bytes * (fr_all + 1u) : 0u;
        const uint32_t cost_out   = ob_bytes
                                  + ((fr_rest > 0) ? word_bytes * (fr_rest + 1u) : word_bytes);
        if (cost_plain <= cost_out) {
            meta[2 * b]     = static_cast<uint8_t>(fr_all);
            meta[2 * b + 1] = 0;
            cost[b]         = cost_plain;
        } else {
            meta[2 * b]     = static_cast<uint8_t>(fr_rest);
            meta[2 * b + 1] = static_cast<uint8_t>(1u | ((ob_bytes - 1u) << 1));
            cost[b]         = cost_out;
        }
    }
}

template<typename T, int ElemsPerLane>
__global__ void encode_pack_outlier_kernel_warp(
    const T* __restrict__ in, size_t num_elements,
    uint32_t word_bytes, size_t num_blocks,
    const uint8_t* __restrict__ meta, const uint32_t* __restrict__ offset,
    uint8_t* __restrict__ payload)
{
    const uint32_t lane          = threadIdx.x & 31u;
    const uint32_t warp_in_block = threadIdx.x >> 5;
    const uint32_t warps_per_cta = blockDim.x >> 5;
    const size_t b = static_cast<size_t>(blockIdx.x) * warps_per_cta + warp_in_block;
    if (b >= num_blocks) return;

    const int     r   = meta[2 * b];
    const uint8_t sel = meta[2 * b + 1];
    const bool is_out = (sel & 1u) != 0;
    constexpr uint32_t block_size = static_cast<uint32_t>(ElemsPerLane) * 32u;
    const size_t start = b * block_size;
    const size_t count = min(static_cast<size_t>(block_size), num_elements - start);
    uint8_t* base = payload + offset[b];

    if (!is_out) {
        if (r == 0) return;
        T v[ElemsPerLane];
        uint32_t av[ElemsPerLane];
        bool active[ElemsPerLane];
        #pragma unroll
        for (int m = 0; m < ElemsPerLane; ++m) {
            const size_t idx = static_cast<size_t>(lane) + 32u * m;
            active[m] = idx < count;
            v[m]  = active[m] ? in[start + idx] : static_cast<T>(0);
            av[m] = active[m] ? static_cast<uint32_t>(absU<T>(v[m])) : 0u;
        }
        #pragma unroll
        for (int m = 0; m < ElemsPerLane; ++m) {
            const uint32_t sm = fz::backend::ballotSync32(active[m] && v[m] < 0);
            if (lane < 4) base[4u * m + lane] = static_cast<uint8_t>((sm >> (8u * lane)) & 0xFFu);
        }
        for (int p = 0; p < r; ++p) {
            #pragma unroll
            for (int m = 0; m < ElemsPerLane; ++m) {
                const uint32_t pm = fz::backend::ballotSync32(active[m] && ((av[m] >> p) & 1u));
                if (lane < 4)
                    base[word_bytes * (1u + p) + 4u * m + lane] =
                        static_cast<uint8_t>((pm >> (8u * lane)) & 0xFFu);
            }
        }
        return;
    }

    // Outlier block: [ob_bytes elem0 magnitude LE][sign region][r planes for elems 1..].
    const uint32_t ob_bytes = ((sel >> 1) & 3u) + 1u;
    if (lane == 0) {
        const uint32_t mag0 = static_cast<uint32_t>(absU<T>(in[start]));
        for (uint32_t k = 0; k < ob_bytes; ++k)
            base[k] = static_cast<uint8_t>((mag0 >> (8u * k)) & 0xffu);
    }
    uint8_t* sign   = base + ob_bytes;
    uint8_t* planes = base + ob_bytes + word_bytes;

    T v[ElemsPerLane];
    uint32_t av[ElemsPerLane];
    bool active[ElemsPerLane], plane_active[ElemsPerLane];
    #pragma unroll
    for (int m = 0; m < ElemsPerLane; ++m) {
        const size_t idx = static_cast<size_t>(lane) + 32u * m;
        active[m]       = idx < count;
        plane_active[m] = active[m] && idx > 0;
        v[m]  = active[m] ? in[start + idx] : static_cast<T>(0);
        av[m] = active[m] ? static_cast<uint32_t>(absU<T>(v[m])) : 0u;
    }
    #pragma unroll
    for (int m = 0; m < ElemsPerLane; ++m) {
        const uint32_t sm = fz::backend::ballotSync32(active[m] && v[m] < 0);
        if (lane < 4) sign[4u * m + lane] = static_cast<uint8_t>((sm >> (8u * lane)) & 0xFFu);
    }
    for (int p = 0; p < r; ++p) {
        #pragma unroll
        for (int m = 0; m < ElemsPerLane; ++m) {
            const uint32_t pm = fz::backend::ballotSync32(plane_active[m] && ((av[m] >> p) & 1u));
            if (lane < 4)
                planes[word_bytes * p + 4u * m + lane] =
                    static_cast<uint8_t>((pm >> (8u * lane)) & 0xFFu);
        }
    }
}

template<typename T, int ElemsPerLane>
__global__ void decode_unpack_outlier_kernel_warp(
    const uint8_t* __restrict__ meta, const uint32_t* __restrict__ offset,
    const uint8_t* __restrict__ payload,
    size_t num_elements, uint32_t word_bytes, size_t num_blocks,
    T* __restrict__ out)
{
    using U = typename std::make_unsigned<T>::type;
    const uint32_t lane          = threadIdx.x & 31u;
    const uint32_t warp_in_block = threadIdx.x >> 5;
    const uint32_t warps_per_cta = blockDim.x >> 5;
    const size_t b = static_cast<size_t>(blockIdx.x) * warps_per_cta + warp_in_block;
    if (b >= num_blocks) return;

    const int     r   = meta[2 * b];
    const uint8_t sel = meta[2 * b + 1];
    const bool is_out = (sel & 1u) != 0;
    constexpr uint32_t block_size = static_cast<uint32_t>(ElemsPerLane) * 32u;
    const size_t start = b * block_size;
    const size_t count = min(static_cast<size_t>(block_size), num_elements - start);
    const uint8_t* base = payload + offset[b];

    if (!is_out) {
        if (r == 0) {
            #pragma unroll
            for (int m = 0; m < ElemsPerLane; ++m) {
                const size_t idx = static_cast<size_t>(lane) + 32u * m;
                if (idx < count) out[start + idx] = static_cast<T>(0);
            }
            return;
        }
        const uint32_t q = lane >> 3, j = lane & 7u;
        #pragma unroll
        for (int m = 0; m < ElemsPerLane; ++m) {
            const size_t idx = static_cast<size_t>(lane) + 32u * m;
            if (idx >= count) continue;
            const uint32_t k = 4u * m + q;
            const uint8_t sgn = base[k];
            uint32_t av = 0;
            for (int p = 0; p < r; ++p)
                av |= ((static_cast<uint32_t>(base[word_bytes * (1u + p) + k]) >> j) & 1u) << p;
            out[start + idx] = applySign<T>(av, ((sgn >> j) & 1u) != 0);
        }
        return;
    }

    // Outlier block.
    const uint32_t ob_bytes = ((sel >> 1) & 3u) + 1u;
    const uint8_t* sign   = base + ob_bytes;
    const uint8_t* planes = base + ob_bytes + word_bytes;

    if (lane == 0) {
        uint32_t mag0 = 0;
        for (uint32_t k = 0; k < ob_bytes; ++k)
            mag0 |= static_cast<uint32_t>(base[k]) << (8u * k);
        out[start] = applySign<T>(mag0, (sign[0] & 1u) != 0);
    }

    const uint32_t q = lane >> 3, j = lane & 7u;
    #pragma unroll
    for (int m = 0; m < ElemsPerLane; ++m) {
        const size_t idx = static_cast<size_t>(lane) + 32u * m;
        if (idx == 0 || idx >= count) continue;
        const uint32_t k = 4u * m + q;
        const uint8_t sgn = sign[k];
        uint32_t av = 0;
        for (int p = 0; p < r; ++p)
            av |= ((static_cast<uint32_t>(planes[word_bytes * p + k]) >> j) & 1u) << p;
        out[start + idx] = applySign<T>(av, ((sgn >> j) & 1u) != 0);
    }
}

// ── Launchers ───────────────────────────────────────────────────────────────
static constexpr int kBlk = 256;
static constexpr int kWarpsPerCta = 8;
static constexpr int kWarpCtaThreads = kWarpsPerCta * 32;

template<typename T>
void launchEncodeRate(const T* d_in, const Config& c,
                      uint8_t* d_rate, uint32_t* d_cost, cudaStream_t stream) {
    if (c.num_blocks == 0) return;
    if (c.block_size == 32u) {
        int grid = static_cast<int>((c.num_blocks + kWarpsPerCta - 1) / kWarpsPerCta);
        encode_rate_kernel_warp<T, 1><<<grid, kWarpCtaThreads, 0, stream>>>(
            d_in, c.num_elements, c.word_bytes, c.num_blocks, d_rate, d_cost);
    } else if (c.block_size == 64u) {
        int grid = static_cast<int>((c.num_blocks + kWarpsPerCta - 1) / kWarpsPerCta);
        encode_rate_kernel_warp<T, 2><<<grid, kWarpCtaThreads, 0, stream>>>(
            d_in, c.num_elements, c.word_bytes, c.num_blocks, d_rate, d_cost);
    } else {
        int grid = static_cast<int>((c.num_blocks + kBlk - 1) / kBlk);
        encode_rate_kernel<T><<<grid, kBlk, 0, stream>>>(
            d_in, c.num_elements, c.block_size, c.word_bytes, c.num_blocks,
            d_rate, d_cost);
    }
    FZ_CUDA_CHECK(cudaGetLastError());
}

template<typename T>
void launchEncodePack(const T* d_in, const Config& c,
                      const uint8_t* d_rate, const uint32_t* d_offset,
                      uint8_t* d_payload, cudaStream_t stream) {
    if (c.num_blocks == 0) return;
    if (c.block_size == 32u) {
        int grid = static_cast<int>((c.num_blocks + kWarpsPerCta - 1) / kWarpsPerCta);
        encode_pack_kernel_warp<T, 1><<<grid, kWarpCtaThreads, 0, stream>>>(
            d_in, c.num_elements, c.word_bytes, c.num_blocks, d_rate, d_offset, d_payload);
    } else if (c.block_size == 64u) {
        int grid = static_cast<int>((c.num_blocks + kWarpsPerCta - 1) / kWarpsPerCta);
        encode_pack_kernel_warp<T, 2><<<grid, kWarpCtaThreads, 0, stream>>>(
            d_in, c.num_elements, c.word_bytes, c.num_blocks, d_rate, d_offset, d_payload);
    } else {
        int grid = static_cast<int>((c.num_blocks + kBlk - 1) / kBlk);
        encode_pack_kernel<T><<<grid, kBlk, 0, stream>>>(
            d_in, c.num_elements, c.block_size, c.word_bytes, c.num_blocks,
            d_rate, d_offset, d_payload);
    }
    FZ_CUDA_CHECK(cudaGetLastError());
}

void launchDecodeCost(const uint8_t* d_rate, const Config& c,
                      uint32_t* d_cost, cudaStream_t stream) {
    if (c.num_blocks == 0) return;
    int grid = static_cast<int>((c.num_blocks + kBlk - 1) / kBlk);
    decode_cost_kernel<<<grid, kBlk, 0, stream>>>(
        d_rate, c.word_bytes, c.num_blocks, d_cost);
    FZ_CUDA_CHECK(cudaGetLastError());
}

template<typename T>
void launchDecodeUnpack(const uint8_t* d_rate, const uint32_t* d_offset,
                        const uint8_t* d_payload, const Config& c,
                        T* d_out, cudaStream_t stream) {
    if (c.num_blocks == 0) return;
    if (c.block_size == 32u) {
        int grid = static_cast<int>((c.num_blocks + kWarpsPerCta - 1) / kWarpsPerCta);
        decode_unpack_kernel_warp<T, 1><<<grid, kWarpCtaThreads, 0, stream>>>(
            d_rate, d_offset, d_payload, c.num_elements, c.word_bytes, c.num_blocks, d_out);
    } else if (c.block_size == 64u) {
        int grid = static_cast<int>((c.num_blocks + kWarpsPerCta - 1) / kWarpsPerCta);
        decode_unpack_kernel_warp<T, 2><<<grid, kWarpCtaThreads, 0, stream>>>(
            d_rate, d_offset, d_payload, c.num_elements, c.word_bytes, c.num_blocks, d_out);
    } else {
        int grid = static_cast<int>((c.num_blocks + kBlk - 1) / kBlk);
        decode_unpack_kernel<T><<<grid, kBlk, 0, stream>>>(
            d_rate, d_offset, d_payload, c.num_elements, c.block_size,
            c.word_bytes, c.num_blocks, d_out);
    }
    FZ_CUDA_CHECK(cudaGetLastError());
}

template<typename T>
void launchEncodeRateOutlier(const T* d_in, const Config& c,
                             uint8_t* d_meta, uint32_t* d_cost, cudaStream_t stream) {
    if (c.num_blocks == 0) return;
    if (c.block_size == 32u) {
        int grid = static_cast<int>((c.num_blocks + kWarpsPerCta - 1) / kWarpsPerCta);
        encode_rate_outlier_kernel_warp<T, 1><<<grid, kWarpCtaThreads, 0, stream>>>(
            d_in, c.num_elements, c.word_bytes, c.num_blocks, d_meta, d_cost);
    } else if (c.block_size == 64u) {
        int grid = static_cast<int>((c.num_blocks + kWarpsPerCta - 1) / kWarpsPerCta);
        encode_rate_outlier_kernel_warp<T, 2><<<grid, kWarpCtaThreads, 0, stream>>>(
            d_in, c.num_elements, c.word_bytes, c.num_blocks, d_meta, d_cost);
    } else {
        int grid = static_cast<int>((c.num_blocks + kBlk - 1) / kBlk);
        encode_rate_outlier_kernel<T><<<grid, kBlk, 0, stream>>>(
            d_in, c.num_elements, c.block_size, c.word_bytes, c.num_blocks,
            d_meta, d_cost);
    }
    FZ_CUDA_CHECK(cudaGetLastError());
}

template<typename T>
void launchEncodePackOutlier(const T* d_in, const Config& c,
                             const uint8_t* d_meta, const uint32_t* d_offset,
                             uint8_t* d_payload, cudaStream_t stream) {
    if (c.num_blocks == 0) return;
    if (c.block_size == 32u) {
        int grid = static_cast<int>((c.num_blocks + kWarpsPerCta - 1) / kWarpsPerCta);
        encode_pack_outlier_kernel_warp<T, 1><<<grid, kWarpCtaThreads, 0, stream>>>(
            d_in, c.num_elements, c.word_bytes, c.num_blocks, d_meta, d_offset, d_payload);
    } else if (c.block_size == 64u) {
        int grid = static_cast<int>((c.num_blocks + kWarpsPerCta - 1) / kWarpsPerCta);
        encode_pack_outlier_kernel_warp<T, 2><<<grid, kWarpCtaThreads, 0, stream>>>(
            d_in, c.num_elements, c.word_bytes, c.num_blocks, d_meta, d_offset, d_payload);
    } else {
        int grid = static_cast<int>((c.num_blocks + kBlk - 1) / kBlk);
        encode_pack_outlier_kernel<T><<<grid, kBlk, 0, stream>>>(
            d_in, c.num_elements, c.block_size, c.word_bytes, c.num_blocks,
            d_meta, d_offset, d_payload);
    }
    FZ_CUDA_CHECK(cudaGetLastError());
}

void launchDecodeCostOutlier(const uint8_t* d_meta, const Config& c,
                             uint32_t* d_cost, cudaStream_t stream) {
    if (c.num_blocks == 0) return;
    int grid = static_cast<int>((c.num_blocks + kBlk - 1) / kBlk);
    decode_cost_outlier_kernel<<<grid, kBlk, 0, stream>>>(
        d_meta, c.word_bytes, c.num_blocks, d_cost);
    FZ_CUDA_CHECK(cudaGetLastError());
}

template<typename T>
void launchDecodeUnpackOutlier(const uint8_t* d_meta, const uint32_t* d_offset,
                               const uint8_t* d_payload, const Config& c,
                               T* d_out, cudaStream_t stream) {
    if (c.num_blocks == 0) return;
    if (c.block_size == 32u) {
        int grid = static_cast<int>((c.num_blocks + kWarpsPerCta - 1) / kWarpsPerCta);
        decode_unpack_outlier_kernel_warp<T, 1><<<grid, kWarpCtaThreads, 0, stream>>>(
            d_meta, d_offset, d_payload, c.num_elements, c.word_bytes, c.num_blocks, d_out);
    } else if (c.block_size == 64u) {
        int grid = static_cast<int>((c.num_blocks + kWarpsPerCta - 1) / kWarpsPerCta);
        decode_unpack_outlier_kernel_warp<T, 2><<<grid, kWarpCtaThreads, 0, stream>>>(
            d_meta, d_offset, d_payload, c.num_elements, c.word_bytes, c.num_blocks, d_out);
    } else {
        int grid = static_cast<int>((c.num_blocks + kBlk - 1) / kBlk);
        decode_unpack_outlier_kernel<T><<<grid, kBlk, 0, stream>>>(
            d_meta, d_offset, d_payload, c.num_elements, c.block_size,
            c.word_bytes, c.num_blocks, d_out);
    }
    FZ_CUDA_CHECK(cudaGetLastError());
}

// ── Explicit instantiations ─────────────────────────────────────────────────
template void launchEncodeRate<int16_t>(const int16_t*, const Config&, uint8_t*, uint32_t*, cudaStream_t);
template void launchEncodeRate<int32_t>(const int32_t*, const Config&, uint8_t*, uint32_t*, cudaStream_t);
template void launchEncodePack<int16_t>(const int16_t*, const Config&, const uint8_t*, const uint32_t*, uint8_t*, cudaStream_t);
template void launchEncodePack<int32_t>(const int32_t*, const Config&, const uint8_t*, const uint32_t*, uint8_t*, cudaStream_t);
template void launchDecodeUnpack<int16_t>(const uint8_t*, const uint32_t*, const uint8_t*, const Config&, int16_t*, cudaStream_t);
template void launchDecodeUnpack<int32_t>(const uint8_t*, const uint32_t*, const uint8_t*, const Config&, int32_t*, cudaStream_t);
template void launchEncodeRateOutlier<int16_t>(const int16_t*, const Config&, uint8_t*, uint32_t*, cudaStream_t);
template void launchEncodeRateOutlier<int32_t>(const int32_t*, const Config&, uint8_t*, uint32_t*, cudaStream_t);
template void launchEncodePackOutlier<int16_t>(const int16_t*, const Config&, const uint8_t*, const uint32_t*, uint8_t*, cudaStream_t);
template void launchEncodePackOutlier<int32_t>(const int32_t*, const Config&, const uint8_t*, const uint32_t*, uint8_t*, cudaStream_t);
template void launchDecodeUnpackOutlier<int16_t>(const uint8_t*, const uint32_t*, const uint8_t*, const Config&, int16_t*, cudaStream_t);
template void launchDecodeUnpackOutlier<int32_t>(const uint8_t*, const uint32_t*, const uint8_t*, const Config&, int32_t*, cudaStream_t);

} // namespace adaptive_bitpack
} // namespace fz
