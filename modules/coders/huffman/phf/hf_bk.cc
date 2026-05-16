// Adapted from PHF reference (origin/v1.1.0_dev:modules/codec/huffman/hf_bk.seq.cc)
// Changes:
//   - Removed #include "timer.hh" and #include <bitset> (only in commented-out debug code).
//   - Removed all hires::now() / duration_t timing; kept float* milliseconds params for ABI
//     compatibility but only preserve the guard "if (milliseconds) *milliseconds = 0" idiom.
//   - Removed phf_allocate_reverse_book (internal allocation helper, not declared in hf.h).
//   - Added definitions for capi_phf_encoded_bytes, capi_phf_coarse_tune_sublen,
//     capi_phf_coarse_tune (declared in hf.h but absent from all reference files).

#include <cstdint>
#include <iostream>

#include "hf.h"
#include "hf_impl.hh"

// ── Missing helper definitions (declared in hf.h) ────────────────────────────

extern "C" {

// Total encoded-bitstream size in bytes.
uint32_t capi_phf_encoded_bytes(phf_header* h)
{
    return h->entry[PHFHEADER_END];
}

// Partition sub-length: number of input elements per coarse-encode partition.
// 768 matches the original cuSZ v1.x default tuning.
size_t capi_phf_coarse_tune_sublen(size_t /*inlen*/)
{
    return 768;
}

void capi_phf_coarse_tune(size_t len, int* sublen, int* pardeg)
{
    *sublen = (int)capi_phf_coarse_tune_sublen(len);
    *pardeg = (int)((len - 1) / (size_t)(*sublen) + 1);
}

}  // extern "C"

// ── phf_reverse_book_bytes ───────────────────────────────────────────────────

// Defined here; hf.h aliases phf_reverse_book_bytes → capi_phf_reverse_book_bytes.
size_t capi_phf_reverse_book_bytes(uint16_t bklen, size_t BK_UNIT_BYTES, size_t SYM_BYTES)
{
    static const int CELL_BITWIDTH = BK_UNIT_BYTES * 8;
    return BK_UNIT_BYTES * (2 * CELL_BITWIDTH) + SYM_BYTES * bklen;
}

// ── phf_CPU_build_canonized_codebook_v1 ─────────────────────────────────────

template <typename E, typename H>
void phf_CPU_build_canonized_codebook_v1(
    uint32_t* freq, int const bklen, H* book, uint8_t* revbook,
    int const /*revbook_bytes*/, float* milliseconds)
{
    constexpr auto TYPE_BITS = sizeof(H) * 8;
    auto bk_bytes  = sizeof(H) * bklen;
    auto space      = new hf_canon_reference<E, H>(bklen);
    if (milliseconds) *milliseconds = 0;

    memset(book, 0xff, bk_bytes);

    phf_CPU_build_codebook_v1<H>(freq, bklen, book);

    space->input_bk() = book;
    space->canonize();

    memcpy(book, space->output_bk(), bk_bytes);

    auto offset = 0;
    memcpy(revbook,          space->first(),  sizeof(int) * TYPE_BITS);
    offset += sizeof(int) * TYPE_BITS;
    memcpy(revbook + offset, space->entry(),  sizeof(int) * TYPE_BITS);
    offset += sizeof(int) * TYPE_BITS;
    memcpy(revbook + offset, space->keys(),   sizeof(E) * bklen);

    delete space;
}

// ── phf_CPU_build_canonized_codebook_v2 ─────────────────────────────────────

template <typename E, typename H>
void phf_CPU_build_canonized_codebook_v2(
    uint32_t* freq, int const bklen, uint32_t* bk4, uint8_t* revbook,
    float* milliseconds)
{
    using PW4 = HuffmanWord<4>;
    using PW8 = HuffmanWord<8>;

    constexpr auto TYPE_BITS = sizeof(H) * 8;
    auto bk_bytes = sizeof(H) * bklen;
    auto space    = new hf_canon_reference<E, H>(bklen);
    if (milliseconds) *milliseconds = 0;

    memset(bk4, 0xff, bk_bytes);

    auto bk8 = new uint64_t[bklen];
    memset(bk8, 0xff, sizeof(uint64_t) * bklen);

    // part 1: build 64-bit codebook then truncate to 32 bits
    phf_CPU_build_codebook_v1<uint64_t>(freq, bklen, bk8);

    for (auto i = 0; i < bklen; i++) {
        auto pw8 = reinterpret_cast<PW8*>(bk8 + i);
        auto pw4 = reinterpret_cast<PW4*>(bk4 + i);

        if (*(bk8 + i) == ~((uint64_t)0x0)) {
            // not meaningful — leave bk4[i] as 0xff...
        }
        else {
            if (pw8->bitcount > pw4->FIELD_CODE) {
                pw4->bitcount    = pw4->OUTLIER_CUTOFF;
                pw4->prefix_code = 0;
                std::cout << i << "\tlarger than FIELD_CODE" << std::endl;
            }
            else {
                pw4->bitcount    = pw8->bitcount;
                pw4->prefix_code = pw8->prefix_code;
            }
        }
    }

    space->input_bk() = bk4;

    // part 2: canonize
    space->canonize();

    memcpy(bk4, space->output_bk(), bk_bytes);

    auto offset = 0;
    memcpy(revbook,          space->first(),  sizeof(int) * TYPE_BITS);
    offset += sizeof(int) * TYPE_BITS;
    memcpy(revbook + offset, space->entry(),  sizeof(int) * TYPE_BITS);
    offset += sizeof(int) * TYPE_BITS;
    memcpy(revbook + offset, space->keys(),   sizeof(E) * bklen);

    delete space;
    delete[] bk8;
}

// ── Explicit instantiations ──────────────────────────────────────────────────

#define INSTANTIATE_PHF_CPU_BUILD_CANONICAL(E, H)                   \
    template void phf_CPU_build_canonized_codebook_v2<E, H>(        \
        uint32_t* freq, int const bklen, H* book, uint8_t* revbook, \
        float* milliseconds);

INSTANTIATE_PHF_CPU_BUILD_CANONICAL(uint8_t,  uint32_t)
INSTANTIATE_PHF_CPU_BUILD_CANONICAL(uint16_t, uint32_t)
INSTANTIATE_PHF_CPU_BUILD_CANONICAL(uint32_t, uint32_t)

#undef INSTANTIATE_PHF_CPU_BUILD_CANONICAL
