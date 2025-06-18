#include <cstddef>

#include "hf.h"
#include "hf_hl.hh"
#include "err.hh"

namespace phf {

template <typename E>
int Buf<E>::_revbk4_bytes(int bklen)
{
  return phf_reverse_book_bytes(bklen, 4, sizeof(SYM));
}

template <typename E>
int Buf<E>::_revbk8_bytes(int bklen) {
  return phf_reverse_book_bytes(bklen, 8, sizeof(SYM));
}

template <typename E>
Buf<E>::Buf(size_t inlen, size_t _bklen, int _pardeg, bool _use_HFR, bool debug)
    : len(inlen),
      pardeg((inlen - 1) / ((_use_HFR ? 1024 : capi_phf_coarse_tune_sublen(inlen))) + 1),
      sublen(_use_HFR ? 1024 : capi_phf_coarse_tune_sublen(inlen)),
      bklen(_bklen),
      use_HFR(_use_HFR),
      revbk4_bytes(_revbk4_bytes(_bklen)),
      bitstream_max_len(inlen / 2),
      rt_bklen(0),
      numSMs(0),
      d_scratch4(),
      h_scratch4(),
      d_encoded(nullptr),
      h_encoded(nullptr),
      d_bitstream4(),
      h_bitstream4(),
      d_bk4(),
      h_bk4(),
      d_revbk4(),
      h_revbk4(),
      d_par_nbit(),
      h_par_nbit(),
      d_par_ncell(),
      h_par_ncell(),
      d_par_entry(),
      h_par_entry(),
      d_brval(),
      d_bridx(),
      d_brnum(),
      h_brnum() {
  (void)_pardeg; //! unused parameter cast to void
  (void)debug;
  cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);

  h_scratch4 = MAKE_UNIQUE_HOST(H4, len);
  d_scratch4 = MAKE_UNIQUE_DEVICE(H4, len);
  h_bk4 = MAKE_UNIQUE_HOST(H4, bklen);
  d_bk4 = MAKE_UNIQUE_DEVICE(H4, bklen);
  h_revbk4 = MAKE_UNIQUE_HOST(PHF_BYTE, revbk4_bytes);
  d_revbk4 = MAKE_UNIQUE_DEVICE(PHF_BYTE, revbk4_bytes);
  d_bitstream4 = MAKE_UNIQUE_DEVICE(H4, bitstream_max_len);
  h_bitstream4 = MAKE_UNIQUE_HOST(H4, bitstream_max_len);
  h_par_nbit = MAKE_UNIQUE_HOST(M, pardeg);
  d_par_nbit = MAKE_UNIQUE_DEVICE(M, pardeg);
  h_par_ncell = MAKE_UNIQUE_HOST(M, pardeg);
  d_par_ncell = MAKE_UNIQUE_DEVICE(M, pardeg);
  h_par_entry = MAKE_UNIQUE_HOST(M, pardeg);
  d_par_entry = MAKE_UNIQUE_DEVICE(M, pardeg);

  // ReVISIT-lite specific
  d_brval = MAKE_UNIQUE_DEVICE(E, 100 + len / 10 + 1);  // len / 10 is a heuristic
  d_bridx = MAKE_UNIQUE_DEVICE(uint32_t, 100 + len / 10 + 1);
  d_brnum = MAKE_UNIQUE_DEVICE(uint32_t, 1);
  h_brnum = MAKE_UNIQUE_HOST(uint32_t, 1);

  // repurpose scratch after several substeps
  d_encoded = (uint8_t*)d_scratch4.get();
  h_encoded = (uint8_t*)h_scratch4.get();

  total_footprint_d += (sizeof(H4) * len) + (sizeof(H4) * bklen) +
                       (sizeof(PHF_BYTE) * revbk4_bytes) +
                       (sizeof(H4) * bitstream_max_len) +
                       (sizeof(M) * pardeg * 3);  // nbit, ncell, entry
  total_footprint_d += (sizeof(E) * (100 + len / 10 + 1)) +  // brval
                        (sizeof(uint32_t) * (100 + len / 10 + 1)) +  // bridx
                        (sizeof(uint32_t) * 1);  // brnum
  total_footprint_h = total_footprint_d;
}

template <typename E>
Buf<E>::~Buf() {}

template <typename E>
void Buf<E>::memcpy_merge(Header& header, phf_stream_t stream)
{
  auto memcpy_start = d_encoded;
  auto memcpy_adjust_to_start = 0;

  memcpy_helper _revbk{     d_revbk4.get(),     revbk4_bytes,                   header.entry[PHFHEADER_REVBK]};
  memcpy_helper _par_nbit{  d_par_nbit.get(),   pardeg * sizeof(M),             header.entry[PHFHEADER_PAR_NBIT]};
  memcpy_helper _par_entry{ d_par_entry.get(),  pardeg * sizeof(M),             header.entry[PHFHEADER_PAR_ENTRY]};
  memcpy_helper _bitstream{ d_bitstream4.get(), bitstream_max_len * sizeof(H4), header.entry[PHFHEADER_BITSTREAM]};

  auto start = ((uint8_t*)memcpy_start + memcpy_adjust_to_start);
  
  auto d2d_memcpy_merge = [&](memcpy_helper& var) {
    CHECK_GPU(cudaMemcpyAsync(start + var.dst, var.ptr, var.nbyte, cudaMemcpyDeviceToDevice, (cudaStream_t)stream));
  };

  CHECK_GPU(cudaMemcpyAsync(start, &header, sizeof(header), cudaMemcpyHostToDevice, (cudaStream_t)stream));

  // /* debug */ CHECK_GPU(cudaStreamSynchronize(stream));
  d2d_memcpy_merge(_revbk);
  d2d_memcpy_merge(_par_nbit);
  d2d_memcpy_merge(_par_entry);
  d2d_memcpy_merge(_bitstream);
  // /* debug */ CHECK_GPU(cudaStreamSynchronize(stream));
}

template <typename E>
void Buf<E>::clear_buffer() {
  cudaMemset(d_scratch4.get(), 0, len);
  cudaMemset(d_bk4.get(), 0, bklen);
  cudaMemset(d_revbk4.get(), 0, revbk4_bytes);
  cudaMemset(d_bitstream4.get(), 0, bitstream_max_len);
  cudaMemset(d_par_nbit.get(), 0, pardeg);
  cudaMemset(d_par_ncell.get(), 0, pardeg);
  cudaMemset(d_par_entry.get(), 0, pardeg);
}

}  // namespace phf

template struct phf::Buf<uint8_t>;
template struct phf::Buf<uint16_t>;
template struct phf::Buf<uint32_t>;