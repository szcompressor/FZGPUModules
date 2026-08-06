// Adapted from PHF reference (origin/v1.1.0_dev:modules/codec/huffman/hf_bk.seq.cc)
// Changes:
//   - Removed #include "timer.hh" and #include <bitset> (only in commented-out debug code).
//   - Removed all hires::now() / duration_t timing; kept float* milliseconds params for ABI
//     compatibility but only preserve the guard "if (milliseconds) *milliseconds = 0" idiom.
//   - Removed phf_allocate_reverse_book (internal allocation helper, not declared in hf.h).
//   - Added definitions for capi_phf_encoded_bytes, capi_phf_coarse_tune_sublen,
//     capi_phf_coarse_tune (declared in hf.h but absent from all reference files).
//   - Removed the "larger than FIELD_CODE" stdout print in the >27-bit clamp path;
//     HuffmanStage::findUnusableCode() detects the clamp marker and throws.

#include <cstdint>
#include <cstdlib>

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
// Together with pardeg = ceil(inlen/sublen) this is the coarse path's entire
// parallel decomposition, and it drives cost in three places at once:
//
//   - encode phase2/phase4 and the decode kernel launch pardeg-sized grids, so a
//     too-large sublen starves the GPU of parallelism;
//   - GPU_coarse_encode_phase3_sync D2Hs two pardeg-sized arrays, runs a SERIAL
//     host prefix-sum plus two accumulates over pardeg, and H2Ds the result,
//     behind two stream syncs -- all O(pardeg), so a too-small sublen makes that
//     host barrier dominate;
//   - two pardeg-sized arrays are written into the encoded stream, so a too-small
//     sublen also costs compression ratio.
//
// It was previously the constant 768 (the original cuSZ v1.x tuning), ignoring
// `inlen` despite taking it as a parameter. On large inputs that leaves a large
// amount on the table in BOTH directions at once.
//
// Measured on H100 (sm_90), LorenzoQuant->Huffman, bklen 1024, book_source
// Adaptive, encode_mode Coarse -- compress GB/s / decompress GB/s / ratio:
//
//   input          n        sublen=768 (old)     this rule        change
//   NYX     134,217,728   216.9/139.1/29.03   246.3/151.2/29.67  +14% +9%  +2.2%
//   CESMATM 168,480,000   143.7/131.9/ 7.97   202.8/137.9/ 8.02  +41% +5%  +0.6%
//   HACC    280,953,867    80.8/ 80.8/ 3.62   155.4/130.1/ 3.64  +92% +61% +0.6%
//   HURR     25,000,000   169.8/ 94.8/16.13   unchanged (floor)
//   CESM      6,480,000    76.3/ 30.4/ 6.19   unchanged (floor)
//
// The rule targets pardeg ~= 131072 and FLOORS at the historical 768, which makes
// every change a strict Pareto improvement -- compress, decompress and ratio all
// improve or stay equal, on every field measured. That floor is deliberate and
// conservative: below ~64M elements the trade stops being free, because a smaller
// sublen buys throughput by spending compression ratio (two pardeg-sized arrays
// ship inside the stream, and each partition pads to a cell boundary).
//
// That smaller-sublen regime is real and sometimes worth taking, but it is a
// trade, so it is opt-in through FZ_HF_SUBLEN rather than the default. Measured
// at sublen=256 against the 768 default: CESM 26 MB +27% compress / +161%
// decompress for -3.9% ratio; EXAALT 11.5 MB +30% / +168% for -1.1% ratio.
//
// Do not extrapolate the constants without re-measuring. The curve is not
// monotonic: parallelism starvation past ~4096 is a steep cliff (CESM-ATM
// decompress falls 138->92 GB/s going 1024->4096), which is why kMaxSublen sits
// at the top of the measured range rather than being open-ended.
//
// sublen is recorded in phf_header, so streams stay self-describing: archives
// written before this change still decode with their own geometry, and this is
// not a format change.
size_t capi_phf_coarse_tune_sublen(size_t inlen)
{
    // Escape hatch for tuning experiments, for the small-input throughput/ratio
    // trade described above, and for reproducing an older archive's encode
    // geometry. Not needed in normal use.
    if (const char* e = getenv("FZ_HF_SUBLEN")) {
        long v = atol(e);
        if (v > 0) return (size_t)v;
    }

    constexpr size_t kTargetPardeg = 131072;
    constexpr size_t kMinSublen    = 768;   // the historical constant; never regress below it
    constexpr size_t kMaxSublen    = 4096;  // top of the measured range

    const size_t want = inlen / kTargetPardeg;
    if (want <= kMinSublen) return kMinSublen;

    // Round down to a power of two. The measured curve is flat between adjacent
    // powers of two, and rounding down keeps pardeg above target rather than
    // below, which is the safer side of the cliff noted above.
    size_t p = 1024;  // first power of two above kMinSublen
    while ((p << 1) <= want && (p << 1) <= kMaxSublen) p <<= 1;
    return p;
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
                // Marker only: prefix_code 0 with bitcount == OUTLIER_CUTOFF is not a
                // usable code.  HuffmanStage::findUnusableCode() detects it and throws;
                // the reference printed a line per symbol to stdout instead, which both
                // let a corrupt book through and polluted the caller's stdout.
                pw4->bitcount    = pw4->OUTLIER_CUTOFF;
                pw4->prefix_code = 0;
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
