/**
 * Adapted from PHF reference (origin/v1.1.0_dev:modules/codec/huffman/hf_canon.seq.cc)
 * Original authors: Jiannan Tian (Indiana University, Argonne National Laboratory)
 * Changes: none — reference has no external-project dependencies.
 */

#include "hf_impl.hh"

template <typename E, typename H>
int canonize(uint8_t* bin, uint32_t const bklen)
{
    constexpr auto TYPE_BITS = sizeof(E) * 8;

    // layout in bin: (1) in-cb, (2) out-cb, (3) canon,
    //                (4) numl, (5) iterby, (6) first, (7) entry, (8) keys
    // sections (6)–(8) form the "revbook"

    auto seg1 = sizeof(H) * (3 * bklen);
    auto seg2 = sizeof(uint32_t) * (4 * TYPE_BITS);

    H*         input_bk  = (H*)bin;
    H*         output_bk = input_bk + bklen;
    H*         canon     = input_bk + bklen * 2;
    auto       numl      = (uint32_t*)(bin + seg1);
    auto       iterby    = numl + TYPE_BITS;
    auto       first     = numl + TYPE_BITS * 2;
    auto       entry     = numl + TYPE_BITS * 3;
    auto       keys      = (E*)(bin + seg1 + seg2);

    using PW = HuffmanWord<sizeof(H)>;
    constexpr auto FILL = ~((H)0x0);

    int max_l = 0;

    for (auto c = input_bk; c < input_bk + bklen; c++) {
        auto pw = (PW*)c;
        int  l  = pw->bitcount;
        if (*c != FILL) {
            max_l = l > max_l ? l : max_l;
            numl[l] += 1;
        }
    }

    for (unsigned long i = 1; i < TYPE_BITS; i++) entry[i] = numl[i - 1];
    for (unsigned long i = 1; i < TYPE_BITS; i++) entry[i] += entry[i - 1];
    for (unsigned long i = 0; i < TYPE_BITS; i++) iterby[i] = entry[i];

    first[max_l] = 0;
    for (int l = max_l - 1; l >= 1; l--)
        first[l] = static_cast<int>((first[l + 1] + numl[l + 1]) / 2.0 + 0.5);
    first[0] = 0xff;

    for (uint32_t i = 0; i < bklen; i++) canon[i]     = FILL;
    for (uint32_t i = 0; i < bklen; i++) output_bk[i] = FILL;

    for (uint32_t i = 0; i < bklen; i++) {
        auto    c = input_bk[i];
        uint8_t l = reinterpret_cast<PW*>(&c)->bitcount;
        if (c != FILL) {
            canon[iterby[l]] = static_cast<H>(first[l] + iterby[l] - entry[l]);
            keys[iterby[l]]  = i;
            reinterpret_cast<PW*>(&(canon[iterby[l]]))->bitcount = l;
            iterby[l]++;
        }
    }

    for (uint32_t i = 0; i < bklen; i++)
        if (canon[i] != FILL) output_bk[keys[i]] = canon[i];

    return 0;
}

#define INIT(E, H) template int canonize<E, H>(uint8_t*, uint32_t const);

INIT(uint8_t,  uint32_t)
INIT(uint16_t, uint32_t)
INIT(uint32_t, uint32_t)
INIT(uint8_t,  uint64_t)
INIT(uint16_t, uint64_t)
INIT(uint32_t, uint64_t)

#undef INIT

// ─────────────────────────────────────────────────────────────────────────────

template <typename E, typename H>
int hf_canon_reference<E, H>::canonize()
{
    using Space = hf_canon_reference<E, H>;
    using PW    = HuffmanWord<sizeof(H)>;
    constexpr auto FILL = ~((H)0x0);

    int max_l = 0;

    for (auto c = input_bk(); c < input_bk() + booklen; c++) {
        auto pw = (PW*)c;
        int  l  = pw->bitcount;
        if (*c != FILL) {
            max_l = l > max_l ? l : max_l;
            numl(l) += 1;
        }
    }

    for (unsigned long i = 1; i < Space::TYPE_BITS; i++) entry(i)  = numl(i - 1);
    for (unsigned long i = 1; i < Space::TYPE_BITS; i++) entry(i)  += entry(i - 1);
    for (unsigned long i = 0; i < Space::TYPE_BITS; i++) iterby(i) = entry(i);

    first(max_l) = 0;
    for (int l = max_l - 1; l >= 1; l--)
        first(l) = static_cast<int>((first(l + 1) + numl(l + 1)) / 2.0 + 0.5);
    first(0) = 0xff;

    for (auto i = 0; i < booklen; i++) canon(i)     = ~((H)0x0);
    for (auto i = 0; i < booklen; i++) output_bk(i) = ~((H)0x0);

    for (auto i = 0; i < booklen; i++) {
        auto    c = input_bk(i);
        uint8_t l = reinterpret_cast<PW*>(&c)->bitcount;
        if (c != FILL) {
            canon(iterby(l)) = static_cast<H>(first(l) + iterby(l) - entry(l));
            keys(iterby(l))  = i;
            reinterpret_cast<PW*>(&(canon(iterby(l))))->bitcount = l;
            iterby(l)++;
        }
    }

    for (auto i = 0; i < booklen; i++)
        if (canon(i) != FILL) output_bk(keys(i)) = canon(i);

    return 0;
}

#define INIT(E, H) template class hf_canon_reference<E, H>;

INIT(uint8_t,  uint32_t)
INIT(uint16_t, uint32_t)
INIT(uint32_t, uint32_t)
INIT(uint8_t,  uint64_t)
INIT(uint16_t, uint64_t)
INIT(uint32_t, uint64_t)
INIT(uint8_t,  unsigned long long)
INIT(uint16_t, unsigned long long)
INIT(uint32_t, unsigned long long)

#undef INIT
