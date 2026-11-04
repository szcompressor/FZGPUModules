// Included by pfpl_stream.cu (inside fz::pfpl::<anonymous>): the AMD (wave64) encode/decode kernels.
//
// The generic harness kernels keep a chunk in two 16 KiB shared buffers plus 4 KiB of coder scratch (36 KiB: one
// 512-thread block per CU on MI200's 64 KiB LDS) and run d_RZE's bitmap recursion with block-wide scans, which
// spend most of their time waiting at barriers (CN-PFPLSTREAM-1). These produce the same stream, byte for byte, from
// one padded 16.1 KiB buffer and 2 KiB of bitmap (3 blocks per CU), with 6-7 barriers:
//  - quantize, difference and bitshuffle in registers (the difference's cross-wavefront neighbours through a
//    64-word exchange), scattering the bitshuffled planes into a buffer padded to 129 words per plane so the
//    plane scatter hits 32 different banks (the unpadded layout puts all 32 lanes of a group in one bank);
//  - zero elimination with one block scan, then the bitmap levels L1 -> L4 (d_RZE's repetition eliminations of
//    2048, 256 and 32 bytes) by wavefront 0 alone with wave scans and ballots, and the look-back;
//  - the packed chunk compacted in place in shared memory and stored once.
// Decoding mirrors it: an element of a repetition-eliminated level is data[k - 1], k the set bitmap bits up to and
// including it, so each level decodes in parallel from popcounts; the chunk-local prefix sum runs as eight
// wave scans at once with one barrier, in the element order the inverse bitshuffle leaves in registers.
// Tuned for, and only used on, 64-lane wavefronts.

namespace amd {

constexpr int      kTPB      = 512;                  // 8 wavefronts; must equal the harness's TPB (storeChunkBytes)
constexpr int      kPerT     = kElems / kTPB;         // 8 elements per thread, and 8 bitshuffled words
constexpr int      kNPP      = kElems / 32;           // words per bitshuffle plane
constexpr int      kPS       = kNPP + 1;              // padded plane stride (words)
constexpr int      kBufWords = 32 * kPS;
constexpr int      kL1       = kChunkBytes / 8;       // zero-elimination bitmap bytes
constexpr uint32_t kAvail    = kChunkBytes - 2;       // d_RZE: room before its 2-byte trailer
static_assert(kTPB == TPB && kTPB == 8 * 64 && kPerT == 8, "one 512-thread block per 16 KiB chunk");
static_assert(kBufWords * 4 >= kChunkBytes + 4, "the packed chunk is compacted in place in the plane buffer");

__device__ __forceinline__ uint32_t inclMask(int j) { return j == 31 ? 0xffffffffu : (2u << j) - 1u; }

template <typename T>
__device__ __forceinline__ T waveInclusiveScan(T v) {
    const int lane = threadIdx.x & 63;
    #pragma unroll
    for (int d = 1; d < 64; d <<= 1) {
        const T up = fz::backend::shflUp(v, d, 64);
        if (lane >= d) v += up;
    }
    return v;
}

// Exclusive prefix of v over the block in thread order, and the total. One barrier; `waves` (8 words) must not be
// rewritten before a later barrier.
__device__ __forceinline__ uint32_t blockExclusiveScan(uint32_t v, uint32_t* waves, uint32_t& total) {
    const int lane = threadIdx.x & 63, wave = threadIdx.x >> 6;
    const uint32_t incl = waveInclusiveScan(v);
    if (lane == 63) waves[wave] = incl;
    __syncthreads();
    uint32_t before = 0, all = 0;
    #pragma unroll
    for (int w = 0; w < 8; ++w) {
        const uint32_t x = waves[w];
        before += w < wave ? x : 0u;
        all += x;
    }
    total = all;
    return before + incl - v;
}

// 32 bytes at an arbitrarily aligned global address (in a buffer whose start is 4-byte aligned and which extends to
// the next 4-byte boundary past them).
__device__ __forceinline__ void load32(const byte* __restrict__ src, uint32_t v[8]) {
    const uint32_t mis = static_cast<uint32_t>(reinterpret_cast<size_t>(src) & 3u);
    const uint32_t* a = reinterpret_cast<const uint32_t*>(src - mis);
    if (mis == 0) {
        #pragma unroll
        for (int r = 0; r < 8; ++r) v[r] = a[r];
    } else {
        uint32_t lo = a[0];
        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            const uint32_t hi = a[r + 1];
            v[r] = __funnelshift_r(lo, hi, 8u * mis);
            lo = hi;
        }
    }
}

__device__ __forceinline__ void store32(byte* __restrict__ dst, const uint32_t v[8]) {
    if ((reinterpret_cast<size_t>(dst) & 3u) == 0) {
        uint32_t* d = reinterpret_cast<uint32_t*>(dst);
        #pragma unroll
        for (int r = 0; r < 8; ++r) d[r] = v[r];
    } else {
        #pragma unroll
        for (int r = 0; r < 8; ++r)
            #pragma unroll
            for (int b = 0; b < 4; ++b) dst[4 * r + b] = static_cast<byte>(v[r] >> (8 * b));
    }
}

__global__ void __launch_bounds__(kTPB)
encodeKernel(const float* __restrict__ in, uint32_t n, QuantInplaceZigzagParams qp,
             byte* __restrict__ out, unsigned long long capacity, uint32_t chunks,
             unsigned long long* __restrict__ state, unsigned long long* __restrict__ ticket,
             unsigned long long ticket_base, uint32_t epoch, size_t* __restrict__ d_stream_bytes) {
    __shared__ __align__(16) uint32_t buf[kBufWords];   // bitshuffled planes (padded), then the packed chunk
    __shared__ __align__(16) uint32_t l1w[kL1 / 4];      // L1 bitmap, then its repetition-eliminated bytes
    __shared__ uint8_t  re2[kL1 / 8];                    // L2's repetition-eliminated bytes
    __shared__ uint8_t  re3[32];                         // L3's
    __shared__ uint32_t scratch[64];
    __shared__ uint32_t s_cid;
    __shared__ uint32_t s_info[8];                       // raw, csize, offset, n1, n2, n3, L4, wpos

    const int t = threadIdx.x, lane = t & 63, wave = t >> 6;
    if (t == 0) s_cid = static_cast<uint32_t>(atomicAdd(ticket, 1ull) - ticket_base);
    __syncthreads();
    const uint32_t cid  = s_cid;
    const size_t   base = static_cast<size_t>(cid) * kElems;

    // Quantize elements i = t + 512k (QuantInplaceZigzag, verify_reconstruction): zigzag codes, raw bits for outliers.
    uint32_t c[kPerT];
    #pragma unroll
    for (int k = 0; k < kPerT; ++k) {
        const size_t i = base + t + kTPB * k;
        const float  x = i < n ? in[i] : 0.0f;
        const int    q = static_cast<int>(roundf(x * qp.ebx2_r));
        bool repr = q > -static_cast<int>(qp.radius) && q < static_cast<int>(qp.radius) && fabsf(x) < qp.threshold;
        if (repr) repr = fabsf(x - fz::backend::fmulRounded(static_cast<float>(q), qp.ebx2)) <= qp.abs_eb;
        c[k] = repr ? static_cast<uint32_t>((q << 1) ^ (q >> 31)) : __float_as_uint(x);
    }

    // Difference with element i-1 (DiffNegabinary): lane-1 of this wavefront, or for lane 0 the previous
    // wavefront's lane 63 (wavefront 0: wavefront 7's of slab k-1; element 0 keeps its code).
    if (lane == 63) {
        #pragma unroll
        for (int k = 0; k < kPerT; ++k) scratch[wave * kPerT + k] = c[k];
    }
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < kPerT; ++k) {
        uint32_t prev = fz::backend::shflUp(c[k], 1, 64);
        if (lane == 0) prev = wave > 0 ? scratch[(wave - 1) * kPerT + k] : (k > 0 ? scratch[7 * kPerT + k - 1] : 0u);
        const int32_t d = (t == 0 && k == 0) ? static_cast<int32_t>(c[k]) : static_cast<int32_t>(c[k] - prev);
        // Bitshuffle32: the butterfly leaves plane (t & 31) of group (t >> 5) + 16k in this lane.
        buf[(t & 31) * kPS + (t >> 5) + 16 * k] =
            fused::chunk::butterfly32(Negabinary<int32_t>::encode(d), t & 31);
    }
    __syncthreads();

    // Zero elimination (d_ZEencodebyteshort): thread t owns bytes 32t..32t+31 of the bitshuffled chunk, i.e. plane
    // t >> 4, words 8(t & 15)..+7.
    uint32_t v[kPerT];
    uint32_t bmp = 0;
    {
        const uint32_t* row = buf + (t >> 4) * kPS + (t & 15) * 8;
        #pragma unroll
        for (int r = 0; r < kPerT; ++r) {
            v[r] = row[r];
            #pragma unroll
            for (int b = 0; b < 4; ++b) bmp |= static_cast<uint32_t>(((v[r] >> (8 * b)) & 0xffu) != 0u) << (4 * r + b);
        }
    }
    l1w[t] = bmp;
    uint32_t wpos;
    const uint32_t pos = blockExclusiveScan(__popc(bmp), scratch, wpos);

    // Wavefront 0: d_RZE's bitmap levels, its fall-back checks, then the look-back.
    if (wave == 0) {
        uint32_t raw = 0, csize = 2, n1 = 0, n2 = 0, n3 = 0, L4 = 0;
        if (wpos >= kAvail) {
            raw = 1;
        } else if (wpos != 0) {
            // L1 -> RE data + L2: lane owns L1 bytes 32*lane..+31; L2 bit e = (byte e != byte e-1), byte -1 = 0.
            uint32_t b1[kPerT];
            #pragma unroll
            for (int r = 0; r < kPerT; ++r) b1[r] = l1w[kPerT * lane + r];
            uint32_t prevByte = fz::backend::shflUp(b1[7], 1, 64) >> 24;
            if (lane == 0) prevByte = 0;
            uint32_t m1 = 0;
            #pragma unroll
            for (int r = 0; r < kPerT; ++r)
                #pragma unroll
                for (int b = 0; b < 4; ++b) {
                    const uint32_t cur = (b1[r] >> (8 * b)) & 0xffu;
                    m1 |= static_cast<uint32_t>(cur != prevByte) << (4 * r + b);
                    prevByte = cur;
                }
            const uint32_t c1 = __popc(m1);
            const uint32_t i1 = waveInclusiveScan(c1);
            n1 = fz::backend::shfl(i1, 63, 64);
            if (n1 > kAvail - wpos) {
                raw = 1;
            } else {
                byte* l1b = reinterpret_cast<byte*>(l1w);   // compacted in place: positions only move down
                uint32_t q = i1 - c1;
                #pragma unroll
                for (int r = 0; r < kPerT; ++r)
                    #pragma unroll
                    for (int b = 0; b < 4; ++b)
                        if ((m1 >> (4 * r + b)) & 1u) l1b[q++] = static_cast<byte>(b1[r] >> (8 * b));
                // L2 (lane owns bytes 4*lane..+3, i.e. m1) -> RE data + L3.
                uint32_t prev2 = fz::backend::shflUp(m1, 1, 64) >> 24;
                if (lane == 0) prev2 = 0;
                uint32_t m2 = 0;
                #pragma unroll
                for (int b = 0; b < 4; ++b) {
                    const uint32_t cur = (m1 >> (8 * b)) & 0xffu;
                    m2 |= static_cast<uint32_t>(cur != prev2) << b;
                    prev2 = cur;
                }
                const uint32_t c2 = __popc(m2);
                const uint32_t i2 = waveInclusiveScan(c2);
                n2 = fz::backend::shfl(i2, 63, 64);
                if (n2 > kAvail - wpos - n1) {
                    raw = 1;
                } else {
                    q = i2 - c2;
                    #pragma unroll
                    for (int b = 0; b < 4; ++b)
                        if ((m2 >> b) & 1u) re2[q++] = static_cast<uint8_t>(m1 >> (8 * b));
                    // L3 (32 bytes: byte j = the nibbles of lanes 2j, 2j+1) -> RE data + L4.
                    const uint32_t lo = fz::backend::shfl(m2, (2 * lane) & 63, 64);
                    const uint32_t hi = fz::backend::shfl(m2, (2 * lane + 1) & 63, 64);
                    const uint32_t l3 = lane < 32 ? (lo | (hi << 4)) : 0u;
                    uint32_t prev3 = fz::backend::shflUp(l3, 1, 64);
                    if (lane == 0) prev3 = 0;
                    const bool nr3 = lane < 32 && l3 != prev3;
                    L4 = static_cast<uint32_t>(fz::backend::ballotWave(nr3));
                    n3 = __popc(L4);
                    if (n3 > kAvail - wpos - n1 - n2) {
                        raw = 1;
                    } else {
                        if (nr3) re3[__popc(L4 & ((1u << lane) - 1u))] = static_cast<uint8_t>(l3);
                        const uint32_t w3 = wpos + n1 + n2 + n3;
                        if (w3 >= kAvail - 4) raw = 1;
                        else csize = w3 + 4 + 2;
                    }
                }
            }
        }
        const uint32_t nbytes = raw ? static_cast<uint32_t>(kChunkBytes) : csize;
        const uint32_t offset = lookbackWave(cid, nbytes, state, epoch, nullptr);
        if (lane == 0) {
            s_info[0] = raw; s_info[1] = csize; s_info[2] = offset; s_info[3] = n1;
            s_info[4] = n2;  s_info[5] = n3;    s_info[6] = L4;     s_info[7] = wpos;
        }
    }
    __syncthreads();
    const uint32_t raw = s_info[0], csize = s_info[1], offset = s_info[2];
    const uint32_t nbytes = raw ? static_cast<uint32_t>(kChunkBytes) : csize;

    const unsigned long long table = kStreamHeader + kLcHeader;
    const unsigned long long dst   = table + 4ull * chunks + offset;
    if (t == 0 && table + 4ull * (cid + 1u) <= capacity)
        reinterpret_cast<uint32_t*>(out + table)[cid] = raw ? (0x80000000u | kChunkBytes) : csize;
    const bool fits = dst + nbytes <= capacity;
    if (raw) {
        if (fits) store32(out + dst + 32u * t, v);   // the bitshuffled chunk itself, as LC stores it
    } else {
        // Compact in place: the zero-eliminated bytes, then [RE1][RE2][RE3][L4][trailer] from wavefront 0.
        byte* bb = reinterpret_cast<byte*>(buf);
        uint32_t q = pos;
        #pragma unroll
        for (int r = 0; r < kPerT; ++r)
            #pragma unroll
            for (int b = 0; b < 4; ++b)
                if ((bmp >> (4 * r + b)) & 1u) bb[q++] = static_cast<byte>(v[r] >> (8 * b));
        if (wave == 0) {
            const uint32_t n1 = s_info[3], n2 = s_info[4], n3 = s_info[5], L4 = s_info[6];
            const byte* l1b = reinterpret_cast<const byte*>(l1w);
            for (uint32_t j = lane; j < n1; j += 64) bb[wpos + j] = l1b[j];
            for (uint32_t j = lane; j < n2; j += 64) bb[wpos + n1 + j] = re2[j];
            if (lane < static_cast<int>(n3)) bb[wpos + n1 + n2 + lane] = re3[lane];
            if (wpos != 0 && lane < 4) bb[wpos + n1 + n2 + n3 + lane] = static_cast<byte>(L4 >> (8 * lane));
            if (lane == 0) {
                bb[csize - 2] = static_cast<byte>(kChunkBytes & 0xff);
                bb[csize - 1] = static_cast<byte>(kChunkBytes >> 8);
            }
        }
        __syncthreads();
        if (fits) fused::chunk::storeChunkBytes(out + dst, bb, csize);
    }

    if (cid == chunks - 1u && t == 0) {
        const unsigned long long archive = kLcHeader + 4ull * chunks + offset + nbytes;
        const unsigned long long padded  = (archive + 3ull) & ~3ull;
        const unsigned long long total   = kStreamHeader + padded;
        uint32_t* w = reinterpret_cast<uint32_t*>(out);
        if (total <= capacity) {
            for (unsigned long long b = kStreamHeader + archive; b < total; ++b) out[b] = 0;
            w[4] = static_cast<uint32_t>(chunks) * kElems * 4u;
            w[5] = chunks;
        }
        if (kStreamHeader <= capacity) {
            w[0] = kStreamMagic;
            w[1] = static_cast<uint32_t>(padded);
            w[2] = __float_as_uint(qp.abs_eb);
            w[3] = n;
        }
        if (d_stream_bytes != nullptr) *d_stream_bytes = static_cast<size_t>(total);
    }
}

// Wavefront 0: rebuild the L1 bitmap of a packed (not raw) chunk from its tail -- [L4][trailer] at the end, RE3,
// RE2 and RE1 data before them -- into l1w, and return where the RE1 data starts (the zero-eliminated bytes' length).
// `payload` is the chunk's copy in shared memory: each level's reads depend on the one before, so they are latency.
__device__ __forceinline__ uint32_t decodeBitmaps(const byte* payload, uint32_t stored, uint32_t* __restrict__ l1w) {
    const int lane = threadIdx.x & 63;
    uint32_t rpos = stored - 2;
    if (rpos == 0) {   // an all-zero chunk: the trailer alone
        for (int i = lane; i < kL1 / 4; i += 64) l1w[i] = 0;
        return 0;
    }
    rpos -= 4;
    const uint32_t L4 = static_cast<uint32_t>(payload[rpos]) | static_cast<uint32_t>(payload[rpos + 1]) << 8 |
                        static_cast<uint32_t>(payload[rpos + 2]) << 16 | static_cast<uint32_t>(payload[rpos + 3]) << 24;
    rpos -= __popc(L4);
    // L3: byte j (lanes < 32) = RE3[k - 1], k the set L4 bits in [0, j]; 0 before the first.
    uint32_t l3 = 0;
    if (lane < 32) {
        const uint32_t k = __popc(L4 & inclMask(lane));
        l3 = k ? payload[rpos + k - 1] : 0u;
    }
    rpos -= fz::backend::waveSum(static_cast<uint32_t>(__popc(l3)));
    // L2: lane owns bytes 4*lane..+3, their L3 bits the nibble (lane & 1) of L3 byte lane >> 1.
    const uint32_t nib = (fz::backend::shfl(l3, lane >> 1, 64) >> (4 * (lane & 1))) & 0xfu;
    const uint32_t c2  = __popc(nib);
    const uint32_t e2  = waveInclusiveScan(c2) - c2;
    uint32_t l2 = 0;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const uint32_t k = e2 + __popc(nib & inclMask(j));
        l2 |= (k ? static_cast<uint32_t>(payload[rpos + k - 1]) : 0u) << (8 * j);
    }
    rpos -= fz::backend::waveSum(static_cast<uint32_t>(__popc(l2)));
    // L1: lane owns bytes 32*lane..+31, their L2 bits are l2.
    const uint32_t c1 = __popc(l2);
    const uint32_t e1 = waveInclusiveScan(c1) - c1;
    #pragma unroll
    for (int r = 0; r < kPerT; ++r) {
        uint32_t word = 0;
        #pragma unroll
        for (int b = 0; b < 4; ++b) {
            const uint32_t k = e1 + __popc(l2 & inclMask(4 * r + b));
            word |= (k ? static_cast<uint32_t>(payload[rpos + k - 1]) : 0u) << (8 * b);
        }
        l1w[kPerT * lane + r] = word;
    }
    return rpos;
}

__global__ void __launch_bounds__(kTPB)
decodeKernel(const byte* __restrict__ stream, float* __restrict__ out, uint32_t n,
             unsigned long long* __restrict__ state, uint32_t epoch) {
    __shared__ __align__(16) uint32_t buf[kBufWords];   // the packed chunk, then the bitshuffled planes (padded)
    __shared__ __align__(16) uint32_t l1w[kL1 / 4];
    __shared__ uint32_t scratch[64];
    __shared__ uint32_t s_offset;

    const int t = threadIdx.x, lane = t & 63, wave = t >> 6;
    const uint32_t* w = reinterpret_cast<const uint32_t*>(stream);
    const uint32_t chunks = gridDim.x;
    if (w[0] != kStreamMagic || w[3] != n || w[4] != chunks * kElems * 4u || w[5] != chunks ||
        w[1] < kLcHeader + 4u * chunks || !(__uint_as_float(w[2]) > 0.0f)) {
        streamTrap();
    }
    const uint32_t  cid    = blockIdx.x;
    const uint32_t* table  = w + (kStreamHeader + kLcHeader) / 4u;
    const uint32_t  entry  = table[cid];
    const uint32_t  stored = entry & 0x7fffffffu;
    const bool      raw    = (entry & 0x80000000u) != 0u;
    if (wave == 0) {
        const uint32_t offset = lookbackWave(cid, stored, state, epoch, table);
        if (lane == 0) s_offset = offset;
    }
    __syncthreads();
    const byte* payload = stream + kStreamHeader + kLcHeader + 4ull * chunks + s_offset;

    uint32_t v[kPerT];
    if (raw) {
        load32(payload + 32u * t, v);
    } else {
        fused::chunk::loadChunkBytes(reinterpret_cast<byte*>(buf), payload, stored);
        __syncthreads();
        if (wave == 0) (void)decodeBitmaps(reinterpret_cast<const byte*>(buf), stored, l1w);
        __syncthreads();
        const uint32_t bmp = l1w[t];
        uint32_t total;
        uint32_t q = blockExclusiveScan(__popc(bmp), scratch, total);
        const byte* bb = reinterpret_cast<const byte*>(buf);
        #pragma unroll
        for (int r = 0; r < kPerT; ++r) {
            uint32_t word = 0;
            #pragma unroll
            for (int b = 0; b < 4; ++b)
                if ((bmp >> (4 * r + b)) & 1u) word |= static_cast<uint32_t>(bb[q++]) << (8 * b);
            v[r] = word;
        }
        __syncthreads();   // every gather done before the planes overwrite the packed bytes
    }
    {
        uint32_t* row = buf + (t >> 4) * kPS + (t & 15) * 8;
        #pragma unroll
        for (int r = 0; r < kPerT; ++r) row[r] = v[r];
    }
    __syncthreads();

    // Inverse bitshuffle and negabinary: element i = t + 512k; then its chunk-local inclusive prefix sum.
    uint32_t d[kPerT];
    #pragma unroll
    for (int k = 0; k < kPerT; ++k)
        d[k] = static_cast<uint32_t>(Negabinary<int32_t>::decode(
            fused::chunk::butterfly32(buf[(t & 31) * kPS + (t >> 5) + 16 * k], t & 31)));
    #pragma unroll
    for (int dd = 1; dd < 64; dd <<= 1)
        #pragma unroll
        for (int k = 0; k < kPerT; ++k) {
            const uint32_t up = fz::backend::shflUp(d[k], dd, 64);
            if (lane >= dd) d[k] += up;
        }
    if (lane == 63) {
        #pragma unroll
        for (int k = 0; k < kPerT; ++k) scratch[k * 8 + wave] = d[k];
    }
    __syncthreads();
    const size_t   base  = static_cast<size_t>(cid) * kElems;
    const uint32_t cnt   = min(static_cast<uint32_t>(kElems), n - cid * kElems);
    const float    ebx2  = 2.0f * __uint_as_float(w[2]);
    uint32_t carry = 0;
    #pragma unroll
    for (int k = 0; k < kPerT; ++k) {
        uint32_t before = 0, slab = 0;
        #pragma unroll
        for (int ww = 0; ww < 8; ++ww) {
            const uint32_t x = scratch[k * 8 + ww];
            before += ww < wave ? x : 0u;
            slab += x;
        }
        const uint32_t code = d[k] + before + carry;
        carry += slab;
        const uint32_t e = t + kTPB * k;
        if (e < cnt) {
            float y;
            if ((code >> 1) >= kQuantRadius) {
                y = __uint_as_float(code);
            } else {
                const int32_t qq = static_cast<int32_t>((code >> 1) ^ (0u - (code & 1u)));
                y = static_cast<float>(qq) * ebx2;
            }
            out[base + e] = y;
        }
    }
}

} // namespace amd
