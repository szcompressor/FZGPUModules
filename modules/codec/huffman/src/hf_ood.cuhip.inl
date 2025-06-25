/**
 * @file hfclass.cuhip.inl
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2023-06-13
 * (created) 2020-04-24
 *
 * @copyright (C) 2020 by Washington State University, The University of
 * Alabama, Argonne National Laboratory
 * @copyright (C) 2021 by Washington State University, Argonne National
 * Laboratory See LICENSE in top-level directory
 *
 */

#ifndef ABBC78E4_3E65_4633_9BEA_27823AB7C398
#define ABBC78E4_3E65_4633_9BEA_27823AB7C398

#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <cuda_runtime.h>

#include "hf.h"
#include "hf_hl.hh"
#include "hf_impl.hh"
#include "mem/cxx_smart_ptr.h"

#include "io.hh"
#include "timer.hh"

using std::cout;
using std::endl;

#define PHF_TPL template <typename E>
#define PHF_CLASS HuffmanCodec<E>
#define PHF_ACCESSOR(SYM, TYPE) \
  reinterpret_cast<TYPE*>(in_encoded + header.entry[PHFHEADER_##SYM])

using _portable::utils::tofile;

namespace phf {

PHF_TPL PHF_CLASS::HuffmanCodec(size_t const inlen, int const _pardeg,
                                bool debug)
    : header{},
      buf{nullptr},
      _time_book{0.0}, _time_lossless{0.0},
      event_start{}, event_end{},
      pardeg{0}, sublen{0}, numSMs{0}, len{0},
      rt_bklen{0},
      h_hist{},
      in_dtype{std::is_same_v<E, uint8_t>   ? HF_U1
               : std::is_same_v<E, uint16_t> ? HF_U2
               : std::is_same_v<E, uint32_t> ? HF_U4
                                       : HF_INVALID} {
  cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);

  pardeg = _pardeg, len = inlen;
  sublen = (inlen - 1) / pardeg + 1;

  // TODO make unique_ptr; modify ctor
  buf = new Buf(inlen, max_bklen, _pardeg, false, debug);
  h_hist = MAKE_UNIQUE_HOST(uint32_t, max_bklen);

  std::tie(event_start, event_end) = event_create_pair();
};

// using CPU huffman
PHF_TPL PHF_CLASS* PHF_CLASS::buildbook(uint32_t* freq, uint16_t const _rt_bklen,
                                        phf_stream_t stream) {
  rt_bklen = _rt_bklen;
  // memcpy_allkinds<D2H>(h_hist.get(), freq, rt_bklen);
  cudaMemcpy(h_hist.get(), freq, rt_bklen * sizeof(uint32_t),
             cudaMemcpyDeviceToHost);

  phf_CPU_build_canonized_codebook_v2<E, H4>(
      h_hist.get(), rt_bklen, buf->h_bk4.get(), buf->h_revbk4.get(),
      &_time_book);
  // memcpy_allkinds_async<H2D>(buf->d_bk4.get(), buf->h_bk4.get(), rt_bklen,
  //                            (cudaStream_t)stream);
  // memcpy_allkinds_async<H2D>(buf->d_revbk4.get(), buf->h_revbk4.get(),
  //                            buf->revbk4_bytes, (cudaStream_t)stream);

  cudaMemcpyAsync(buf->d_bk4.get(), buf->h_bk4.get(),
                  rt_bklen * sizeof(H4), cudaMemcpyHostToDevice,
                  (cudaStream_t)stream);
  cudaMemcpyAsync(buf->d_revbk4.get(), buf->h_revbk4.get(),
                  buf->revbk4_bytes * sizeof(PHF_BYTE),
                  cudaMemcpyHostToDevice, (cudaStream_t)stream);

  return this;
}

PHF_TPL
PHF_CLASS* PHF_CLASS::encode(E* in, size_t const len, uint8_t** out,
                             size_t* outlen, phf_stream_t stream) {
  _time_lossless = 0;
  phf::par_config hfpar{sublen, pardeg};

  event_recording_start(event_start, stream);

  phf_module::GPU_coarse_encode(
      in, len, buf->d_bk4.get(), rt_bklen, numSMs, hfpar,
      // internal buffers
      buf->d_scratch4.get(), buf->d_par_nbit.get(), buf->h_par_nbit.get(),
      buf->d_par_ncell.get(), buf->h_par_ncell.get(), buf->d_par_entry.get(),
      buf->h_par_entry.get(), buf->d_bitstream4.get(), buf->bitstream_max_len,
      // output
      &header.total_nbit, &header.total_ncell, stream);

  event_recording_stop(event_end, stream);
  event_time_elapsed(event_start, event_end, &_time_lossless);

  cudaStreamSynchronize((cudaStream_t)stream);

  make_metadata();
  buf->memcpy_merge(header, stream);  // TODO externalize/make explicit

  *out = buf->d_encoded;
  *outlen = phf_encoded_bytes(&header);

  return this;
}

PHF_TPL PHF_CLASS* PHF_CLASS::decode(uint8_t* in_encoded, E* out_decoded,
                                     phf_stream_t stream,
                                     bool header_on_device) {
  Header header;
  if (header_on_device)
    cudaMemcpyAsync(&header, in_encoded, sizeof(header), cudaMemcpyDeviceToHost,
                    (cudaStream_t)stream);

  event_recording_start(event_start, stream);

  phf_module::GPU_coarse_decode(
      PHF_ACCESSOR(BITSTREAM, H4), PHF_ACCESSOR(REVBK, PHF_BYTE),
      buf->revbk4_bytes, PHF_ACCESSOR(PAR_NBIT, M), PHF_ACCESSOR(PAR_ENTRY, M),
      header.sublen, header.pardeg, out_decoded, stream);

  event_recording_stop(event_end, stream);
  event_time_elapsed(event_start, event_end, &_time_lossless);

  return this;
}

PHF_TPL PHF_CLASS* PHF_CLASS::clear_buffer() {
  buf->clear_buffer();
  return this;
}

PHF_TPL void PHF_CLASS::make_metadata() {
  // header.self_bytes = sizeof(Header);
  header.bklen = rt_bklen;
  header.sublen = sublen;
  header.pardeg = pardeg;
  header.original_len = len;

  M nbyte[PHFHEADER_END];
  nbyte[PHFHEADER_HEADER] = PHFHEADER_FORCED_ALIGN;
  nbyte[PHFHEADER_REVBK] = buf->revbk4_bytes;
  nbyte[PHFHEADER_PAR_NBIT] = buf->pardeg * sizeof(M);
  nbyte[PHFHEADER_PAR_ENTRY] = buf->pardeg * sizeof(M);
  nbyte[PHFHEADER_BITSTREAM] = 4 * header.total_ncell;

  header.entry[0] = 0;
  // *.END + 1: need to know the ending position
  for (auto i = 1; i < PHFHEADER_END + 1; i++) header.entry[i] = nbyte[i - 1];
  for (auto i = 1; i < PHFHEADER_END + 1; i++)
    header.entry[i] += header.entry[i - 1];
}

PHF_TPL PHF_CLASS* PHF_CLASS::dump_internal_data(std::string field,
                                                 std::string fname) {
  auto ofname = fname + ".book_u4";
  if (field == "book")
    _portable::utils::tofile(ofname, buf->h_bk4.get(), rt_bklen);
  return this;
}

PHF_TPL PHF_CLASS::~HuffmanCodec() {
  delete buf;
  event_destroy_pair(event_start, event_end);
}

}  // namespace phf

#undef PHF_ACCESSOR
#undef PHF_TPL
#undef PHF_CLASS

#endif /* ABBC78E4_3E65_4633_9BEA_27823AB7C398 */
