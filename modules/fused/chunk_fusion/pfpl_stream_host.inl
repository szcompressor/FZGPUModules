// The coder on the host: the stream the kernels write, byte for byte, from plain C++. No device, stream or Plan, so any
// thread may call it, and it needs no GPU. One thread, one chunk at a time. Part of pfpl_stream.cu.
namespace host {

constexpr uint32_t kAvail  = kChunkBytes - 2;       // d_RZE: room before its 2-byte trailer
constexpr int      kL1     = kChunkBytes / 8;       // zero-elimination bitmap bytes
constexpr int      kL2     = kL1 / 8;
constexpr int      kL3     = kL2 / 8;
constexpr int      kL4     = kL3 / 8;
constexpr int      kWords  = kChunkBytes / 4;       // 32-bit words of a bitshuffled chunk

// What the kernels' butterfly32 computes over 32 lanes: word p's bit (31 - r) is word r's bit (31 - p). Its own inverse.
inline void transpose32(uint32_t* a) {
    uint32_t m = 0x0000ffffu;
    for (int j = 16; j != 0; j >>= 1, m ^= (m << j))
        for (int k = 0; k < 32; k = (k + j + 1) & ~j) {
            const uint32_t t = (a[k] ^ (a[k + j] >> j)) & m;
            a[k] ^= t;
            a[k + j] ^= (t << j);
        }
}

// The range check comes before the conversion: out of int's range it is undefined here, where the GPU saturates, and
// either way the value is stored raw. `volatile` keeps a compiler from fusing the product into x - q * ebx2: the bound
// check reads the rounded product, as fz::backend::fmulRounded makes the kernels do.
inline uint32_t codeOf(float x, const QuantInplaceZigzagParams& p) {
    const float r = std::round(x * p.ebx2_r);
    if (std::fabs(r) < static_cast<float>(p.radius) && std::fabs(x) < p.threshold) {
        const int q = static_cast<int>(r);
        volatile float rebuilt = static_cast<float>(q) * p.ebx2;
        const float value = rebuilt;
        if (std::fabs(x - value) <= p.abs_eb)
            return (static_cast<uint32_t>(q) << 1) ^ static_cast<uint32_t>(q >> 31);
    }
    uint32_t bits;
    std::memcpy(&bits, &x, sizeof(bits));
    return bits;
}

inline uint64_t codeOf(double x, const QuantParams64& p) {
    const double r = std::rint(x * p.ebx2_r);
    if (std::fabs(r) < static_cast<double>(kQuantRadius64)) {
        volatile double rebuilt = r * p.ebx2;
        const double value = rebuilt;
        if (std::fabs(x - value) <= p.abs_eb) {
            const long long q = static_cast<long long>(r);
            return (static_cast<uint64_t>(q) << 1) ^ static_cast<uint64_t>(q >> 63);
        }
    }
    uint64_t bits;
    std::memcpy(&bits, &x, sizeof(bits));
    return bits;
}

inline float valueOf(uint32_t code, float ebx2) {
    if ((code >> 1) >= kQuantRadius) {
        float x;
        std::memcpy(&x, &code, sizeof(x));
        return x;
    }
    return static_cast<float>(static_cast<int32_t>((code >> 1) ^ (0u - (code & 1u)))) * ebx2;
}

inline double valueOf(uint64_t code, double ebx2) {
    if ((code >> 1) >= kQuantRadius64) {
        double x;
        std::memcpy(&x, &code, sizeof(x));
        return x;
    }
    return static_cast<double>(static_cast<long long>((code >> 1) ^ (0ull - (code & 1ull)))) * ebx2;
}

// Negabinary codes to the bitshuffled chunk's words, and back: plane by plane, most significant first. A float chunk's
// plane is one word per group of 32 elements. A double's high words make planes 0..31 and its low words planes 32..63,
// and a plane holds its groups' words in pairs swapped (the staged 8-byte bitshuffle's order).
inline void shuffle(const uint32_t* nb, uint32_t* words) {
    for (int g = 0; g < kWords / 32; ++g) {
        uint32_t a[32];
        std::memcpy(a, nb + 32 * g, sizeof(a));
        transpose32(a);
        for (int p = 0; p < 32; ++p) words[p * (kWords / 32) + g] = a[p];
    }
}

inline void unshuffle(const uint32_t* words, uint32_t* nb) {
    for (int g = 0; g < kWords / 32; ++g) {
        uint32_t a[32];
        for (int p = 0; p < 32; ++p) a[p] = words[p * (kWords / 32) + g];
        transpose32(a);
        std::memcpy(nb + 32 * g, a, sizeof(a));
    }
}

inline void shuffle(const uint64_t* nb, uint32_t* words) {
    constexpr int kNpp = kWords / 64;
    for (int g = 0; g < kNpp; ++g) {
        uint32_t hi[32], lo[32];
        for (int r = 0; r < 32; ++r) {
            hi[r] = static_cast<uint32_t>(nb[32 * g + r] >> 32);
            lo[r] = static_cast<uint32_t>(nb[32 * g + r]);
        }
        transpose32(hi);
        transpose32(lo);
        for (int p = 0; p < 32; ++p) {
            words[p * kNpp + (g ^ 1)]        = hi[p];
            words[(32 + p) * kNpp + (g ^ 1)] = lo[p];
        }
    }
}

inline void unshuffle(const uint32_t* words, uint64_t* nb) {
    constexpr int kNpp = kWords / 64;
    for (int g = 0; g < kNpp; ++g) {
        uint32_t hi[32], lo[32];
        for (int p = 0; p < 32; ++p) {
            hi[p] = words[p * kNpp + (g ^ 1)];
            lo[p] = words[(32 + p) * kNpp + (g ^ 1)];
        }
        transpose32(hi);
        transpose32(lo);
        for (int r = 0; r < 32; ++r) nb[32 * g + r] = (static_cast<uint64_t>(hi[r]) << 32) | lo[r];
    }
}

inline uint32_t popcountBytes(const uint8_t* b, int n) {
    uint32_t c = 0;
    for (int i = 0; i < n; ++i) c += static_cast<uint32_t>(__builtin_popcount(b[i]));
    return c;
}

// One level of repetition elimination: bit e of `bitmap` says byte e differs from the one before it (byte -1 is 0), and
// the bytes so marked go to `kept`. Returns how many.
inline uint32_t repeatEliminate(const uint8_t* src, int n, uint8_t* bitmap, uint8_t* kept) {
    std::memset(bitmap, 0, static_cast<size_t>(n) / 8);
    uint32_t count = 0;
    uint8_t prev = 0;
    for (int e = 0; e < n; ++e) {
        if (src[e] != prev) {
            bitmap[e >> 3] = static_cast<uint8_t>(bitmap[e >> 3] | (1u << (e & 7)));
            kept[count++] = src[e];
            prev = src[e];
        }
    }
    return count;
}

// Its inverse.
inline void repeatRestore(const uint8_t* kept, const uint8_t* bitmap, int n, uint8_t* out) {
    uint8_t prev = 0;
    uint32_t k = 0;
    for (int e = 0; e < n; ++e) {
        if ((bitmap[e >> 3] >> (e & 7)) & 1u) prev = kept[k++];
        out[e] = prev;
    }
}

// d_RZE on one bitshuffled chunk: its packed form in `packed` and that length, with the kernels' fall-back checks in
// their order, or 0 if the chunk is stored raw.
inline uint32_t packChunk(const uint8_t* chunk, uint8_t* packed) {
    uint8_t l1[kL1] = {};
    uint32_t wpos = 0;
    for (int i = 0; i < kChunkBytes; ++i) {
        if (chunk[i] != 0) {
            l1[i >> 3] = static_cast<uint8_t>(l1[i >> 3] | (1u << (i & 7)));
            packed[wpos++] = chunk[i];
        }
    }
    if (wpos >= kAvail) return 0;
    uint32_t csize = 2;
    if (wpos != 0) {
        uint8_t l2[kL2], l3[kL3], l4[kL4], re1[kL1], re2[kL2], re3[kL3];
        const uint32_t n1 = repeatEliminate(l1, kL1, l2, re1);
        if (n1 > kAvail - wpos) return 0;
        const uint32_t n2 = repeatEliminate(l2, kL2, l3, re2);
        if (n2 > kAvail - wpos - n1) return 0;
        const uint32_t n3 = repeatEliminate(l3, kL3, l4, re3);
        if (n3 > kAvail - wpos - n1 - n2) return 0;
        const uint32_t w3 = wpos + n1 + n2 + n3;
        if (w3 >= kAvail - 4) return 0;
        std::memcpy(packed + wpos, re1, n1);
        std::memcpy(packed + wpos + n1, re2, n2);
        std::memcpy(packed + wpos + n1 + n2, re3, n3);
        std::memcpy(packed + w3, l4, kL4);
        csize = w3 + kL4 + 2;
    }
    packed[csize - 2] = static_cast<uint8_t>(kChunkBytes & 0xff);
    packed[csize - 1] = static_cast<uint8_t>(kChunkBytes >> 8);
    return csize;
}

// The bitshuffled chunk from its packed form. False if the lengths in it do not add up to `stored`.
inline bool unpackChunk(const uint8_t* payload, uint32_t stored, uint8_t* chunk) {
    if (stored < 2 || stored >= kChunkBytes) return false;
    uint32_t rpos = stored - 2;
    if (rpos == 0) {
        std::memset(chunk, 0, kChunkBytes);
        return true;
    }
    if (rpos < kL4) return false;
    rpos -= kL4;
    uint8_t l1[kL1], l2[kL2], l3[kL3];
    const uint8_t* l4 = payload + rpos;
    const uint32_t n3 = popcountBytes(l4, kL4);
    if (rpos < n3) return false;
    rpos -= n3;
    repeatRestore(payload + rpos, l4, kL3, l3);
    const uint32_t n2 = popcountBytes(l3, kL3);
    if (rpos < n2) return false;
    rpos -= n2;
    repeatRestore(payload + rpos, l3, kL2, l2);
    const uint32_t n1 = popcountBytes(l2, kL2);
    if (rpos < n1) return false;
    rpos -= n1;
    repeatRestore(payload + rpos, l2, kL1, l1);
    if (popcountBytes(l1, kL1) != rpos) return false;
    uint32_t j = 0;
    for (int i = 0; i < kChunkBytes; ++i) chunk[i] = ((l1[i >> 3] >> (i & 7)) & 1u) ? payload[j++] : uint8_t(0);
    return true;
}

template <typename Real>
size_t encode(const Real* in, size_t n, const typename Elem<Real>::Params& params, float abs_eb, void* out_, size_t capacity) {
    using Word  = typename Elem<Real>::Word;
    using SWord = typename Elem<Real>::SWord;
    constexpr size_t kPer = kChunkBytes / sizeof(Real);
    uint8_t* out = static_cast<uint8_t*>(out_);
    const uint32_t chunks = chunksOf<Real>(n);
    const uint32_t magic = Elem<Real>::kMagic;
    uint32_t boundBits;
    std::memcpy(&boundBits, &abs_eb, sizeof(boundBits));
    if (chunks == 0) {
        const uint32_t w[6] = {magic, kLcHeader, boundBits, 0u, 0u, 0u};
        if (kStreamHeader <= capacity) std::memcpy(out, w, kStreamHeader);
        if (kStreamHeader + kLcHeader <= capacity) std::memcpy(out + kStreamHeader, w + 4, kLcHeader);
        return kStreamHeader + kLcHeader;
    }
    const unsigned long long table = kStreamHeader + kLcHeader;
    const unsigned long long data  = table + 4ull * chunks;
    unsigned long long offset = 0;
    Word nb[kPer];
    uint32_t words[kWords];
    uint8_t packed[kChunkBytes];
    for (uint32_t cid = 0; cid < chunks; ++cid) {
        const size_t base = static_cast<size_t>(cid) * kPer;
        Word prev = 0;
        for (size_t i = 0; i < kPer; ++i) {
            const Word code = codeOf(base + i < n ? in[base + i] : static_cast<Real>(0), params);
            nb[i] = Negabinary<SWord>::encode(static_cast<SWord>(code - prev));
            prev = code;
        }
        shuffle(nb, words);
        const uint8_t* chunk = reinterpret_cast<const uint8_t*>(words);
        const uint32_t csize = packChunk(chunk, packed);
        const uint32_t entry = csize != 0 ? csize : (0x80000000u | static_cast<uint32_t>(kChunkBytes));
        const uint32_t nbytes = entry & 0x7fffffffu;
        if (table + 4ull * (cid + 1u) <= capacity) std::memcpy(out + table + 4ull * cid, &entry, sizeof(entry));
        if (data + offset + nbytes <= capacity) std::memcpy(out + data + offset, csize != 0 ? packed : chunk, nbytes);
        offset += nbytes;
    }
    const unsigned long long archive = kLcHeader + 4ull * chunks + offset;
    const unsigned long long padded  = (archive + 3ull) & ~3ull;
    const unsigned long long total   = kStreamHeader + padded;
    if (total <= capacity) {
        for (unsigned long long b = kStreamHeader + archive; b < total; ++b) out[b] = 0;
        const uint32_t lc[2] = {chunks * static_cast<uint32_t>(kChunkBytes), chunks};
        std::memcpy(out + kStreamHeader, lc, sizeof(lc));
    }
    if (kStreamHeader <= capacity) {
        const uint32_t w[4] = {magic, static_cast<uint32_t>(padded), boundBits, static_cast<uint32_t>(n)};
        std::memcpy(out, w, sizeof(w));
    }
    return static_cast<size_t>(total);
}

template <typename Real>
bool decode(const void* stream_, Real* out, size_t n, size_t outBytes) {
    using Word  = typename Elem<Real>::Word;
    using SWord = typename Elem<Real>::SWord;
    constexpr size_t kPer = kChunkBytes / sizeof(Real);
    const uint8_t* stream = static_cast<const uint8_t*>(stream_);
    if (stream == nullptr || streamBytes(stream) == 0 || !fitsArchive<Real>(n)) return false;
    const uint32_t chunks = chunksOf<Real>(n);
    uint32_t w[6];
    std::memcpy(w, stream, sizeof(w));
    float bound;
    std::memcpy(&bound, &w[2], sizeof(bound));
    if (w[0] != Elem<Real>::kMagic || w[3] != n || w[4] != chunks * static_cast<uint32_t>(kChunkBytes) || w[5] != chunks ||
        w[1] < kLcHeader + 4u * chunks || !(bound > 0.0f))
        return false;
    const Real ebx2 = static_cast<Real>(2) * static_cast<Real>(bound);
    const unsigned long long end = kLcHeader + 4ull * chunks;
    const size_t want = std::min(outBytes, n * sizeof(Real));
    uint8_t* to = reinterpret_cast<uint8_t*>(out);
    size_t written = 0;
    unsigned long long offset = 0;
    uint32_t words[kWords];
    Word nb[kPer];
    Real values[kPer];
    for (uint32_t cid = 0; cid < chunks && written < want; ++cid) {
        uint32_t entry;
        std::memcpy(&entry, stream + kStreamHeader + kLcHeader + 4ull * cid, sizeof(entry));
        const uint32_t stored = entry & 0x7fffffffu;
        const bool raw = (entry & 0x80000000u) != 0u;
        if (end + offset + stored > w[1]) return false;
        const uint8_t* payload = stream + kStreamHeader + end + offset;
        offset += stored;
        if (raw) {
            if (stored != kChunkBytes) return false;
            std::memcpy(words, payload, kChunkBytes);
        } else if (!unpackChunk(payload, stored, reinterpret_cast<uint8_t*>(words))) {
            return false;
        }
        unshuffle(words, nb);
        Word code = 0;
        for (size_t i = 0; i < kPer; ++i) {
            code += static_cast<Word>(Negabinary<SWord>::decode(nb[i]));
            values[i] = valueOf(code, ebx2);
        }
        const size_t bytes = std::min(want - written, kPer * sizeof(Real));
        std::memcpy(to + written, values, bytes);
        written += bytes;
    }
    return written == want;
}

} // namespace host
