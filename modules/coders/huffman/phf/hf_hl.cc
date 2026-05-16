// Adapted from PHF reference (origin/v1.1.0_dev:modules/codec/huffman/hf_hl.cc)
// Changes:
//   - Replaced #include "hf_hl.hh" with local hf_buf.h.
//   - Removed #include <iostream> and related using-decls (not needed).
//   - Dropped encode_ReVISIT_lite (not called by HuffmanStage).
//   - Dropped float _time_book timing; pass nullptr to phf_CPU_build_canonized_codebook_v2.

#include "hf_buf.h"
#include "hf_impl.hh"

using H4 = uint32_t;
using M  = PHF_METADATA;

namespace phf {

template <typename E>
using phf_module = cuhip::modules<E, H4>;

template <typename E>
int high_level<E>::build_book(
    phf::Buf<E>* buf, uint32_t* h_hist, uint16_t const rt_bklen, HF_STREAM stream)
{
    buf->register_runtime_bklen(rt_bklen);

    phf_CPU_build_canonized_codebook_v2<E, H4>(
        h_hist, rt_bklen, buf->h_bk4, buf->h_revbk4, nullptr);

    cudaMemcpyAsync(
        buf->d_bk4, buf->h_bk4,
        rt_bklen * sizeof(H4), cudaMemcpyHostToDevice, (cudaStream_t)stream);
    cudaMemcpyAsync(
        buf->d_revbk4, buf->h_revbk4,
        buf->revbk4_bytes, cudaMemcpyHostToDevice, (cudaStream_t)stream);

    return 0;
}

template <typename E>
int high_level<E>::encode(
    HF_SPACE* buf, E* in, size_t const len, uint8_t** out, size_t* outlen,
    phf_header& header, HF_STREAM stream)
{
    // Uses actual data_len (not buf->len which is the allocated capacity) so that
    // capacity-based reallocation produces correct pardeg and original_len in the header.
    auto make_metadata = [](HF_SPACE* buf, size_t data_len, phf_header& header) {
        const size_t actual_pardeg = (data_len - 1) / buf->sublen + 1;
        header.bklen        = buf->rt_bklen;
        header.sublen       = buf->sublen;
        header.pardeg       = actual_pardeg;
        header.original_len = data_len;

        M nbyte[PHFHEADER_END];
        nbyte[PHFHEADER_HEADER]    = PHFHEADER_FORCED_ALIGN;
        nbyte[PHFHEADER_REVBK]     = buf->revbk4_bytes;
        nbyte[PHFHEADER_PAR_NBIT]  = actual_pardeg * sizeof(M);
        nbyte[PHFHEADER_PAR_ENTRY] = actual_pardeg * sizeof(M);
        nbyte[PHFHEADER_BITSTREAM] = 4 * header.total_ncell;

        header.entry[0] = 0;
        for (auto i = 1; i < PHFHEADER_END + 1; i++) header.entry[i] = nbyte[i - 1];
        for (auto i = 1; i < PHFHEADER_END + 1; i++) header.entry[i] += header.entry[i - 1];
    };

    phf_module<E>::GPU_coarse_encode(
        in, len, buf->d_bk4, buf->rt_bklen, buf->numSMs,
        {buf->sublen, buf->pardeg},
        buf->d_scratch4,
        buf->d_par_nbit,  buf->h_par_nbit,
        buf->d_par_ncell, buf->h_par_ncell,
        buf->d_par_entry, buf->h_par_entry,
        buf->d_bitstream4, buf->bitstream_max_len,
        &header.total_nbit, &header.total_ncell, stream);

    cudaStreamSynchronize((cudaStream_t)stream);

    make_metadata(buf, len, header);
    buf->memcpy_merge(header, stream);

    *out    = buf->d_encoded;
    *outlen = phf_encoded_bytes(&header);

    return 0;
}

#define PHF_ACCESSOR(SYM, TYPE) \
    reinterpret_cast<TYPE*>(in_encoded + header.entry[PHFHEADER_##SYM])

template <typename E>
int high_level<E>::decode(
    HF_SPACE* buf, phf_header& header, PHF_BYTE* in_encoded,
    E* out_decoded, HF_STREAM stream)
{
    phf_module<E>::GPU_coarse_decode(
        PHF_ACCESSOR(BITSTREAM, H4), PHF_ACCESSOR(REVBK, PHF_BYTE),
        buf->revbk4_bytes,
        PHF_ACCESSOR(PAR_NBIT, M), PHF_ACCESSOR(PAR_ENTRY, M),
        header.sublen, header.pardeg, out_decoded, stream);

    return 0;
}

#undef PHF_ACCESSOR

}  // namespace phf

template struct phf::high_level<uint8_t>;
template struct phf::high_level<uint16_t>;
template struct phf::high_level<uint32_t>;
