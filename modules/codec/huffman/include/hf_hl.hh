#ifndef HF_HL_HH
#define HF_HL_HH

#include <cstdint>
#include <memory>

#include "err.hh"
#include "hf.h"
#include "hf_impl.hh"
#include "mem/cxx_smart_ptr.h"

namespace phf {

template <typename E>
struct Buf {
  using H4 = uint32_t;
  using M = PHF_METADATA;

  typedef struct RC {
    static const int SCRATCH = 0;
    static const int FREQ = 1;
    static const int BK = 2;
    static const int REVBK = 3;
    static const int PAR_NBIT = 4;
    static const int PAR_NCELL = 5;
    static const int PAR_ENTRY = 6;
    static const int BITSTREAM = 7;
    static const int END = 8;
    // uint32_t nbyte[END];
  } RC;

  typedef struct {
    void* const ptr;
    size_t const nbyte;
    size_t const dst;
  } memcpy_helper;

  using SYM = E;
  using Header = phf_header;

  const size_t len;
  size_t pardeg;
  size_t sublen;
  const size_t bklen;
  const bool use_HFR;
  const size_t revbk4_bytes;
  const size_t bitstream_max_len;

  uint16_t rt_bklen;
  int numSMs;
  size_t total_footprint_d = 0;
  size_t total_footprint_h = 0;

  // array
  GPU_unique_dptr<H4[]> d_scratch4;
  GPU_unique_hptr<H4[]> h_scratch4;
  PHF_BYTE* d_encoded;
  PHF_BYTE* h_encoded;
  GPU_unique_dptr<H4[]> d_bitstream4;
  GPU_unique_hptr<H4[]> h_bitstream4;

  GPU_unique_dptr<H4[]> d_bk4;
  GPU_unique_hptr<H4[]> h_bk4;
  GPU_unique_dptr<PHF_BYTE[]> d_revbk4;
  GPU_unique_hptr<PHF_BYTE[]> h_revbk4;

  // data partition/embarrassingly parallelism description
  GPU_unique_dptr<M[]> d_par_nbit;
  GPU_unique_hptr<M[]> h_par_nbit;
  GPU_unique_dptr<M[]> d_par_ncell;
  GPU_unique_hptr<M[]> h_par_ncell;
  GPU_unique_dptr<M[]> d_par_entry;
  GPU_unique_hptr<M[]> h_par_entry;

  // ReVISIT-lite specific
  GPU_unique_dptr<E[]> d_brval;
  GPU_unique_dptr<uint32_t[]> d_bridx;
  GPU_unique_dptr<uint32_t[]> d_brnum;
  GPU_unique_hptr<uint32_t[]> h_brnum;

  // utils
  static int _revbk4_bytes(int bklen);
  static int _revbk8_bytes(int bklen);

  Buf(const Buf&) = delete;
  Buf& operator=(const Buf&) = delete;

  Buf(Buf&&) noexcept = delete;
  Buf& operator=(Buf&&) noexcept = delete;

  // ctor
  Buf(size_t inlen, size_t _bklen, int _pardeg = -1, bool _use_HFR = false,
      bool debug = false);
  ~Buf();

  // setter
  void register_runtime_bklen(int _rt_bklen) { rt_bklen = _rt_bklen; }

  void memcpy_merge(phf_header& header, phf_stream_t stream);
  void clear_buffer();
};

template <typename E>
class HuffmanCodec {
 private:
  using SYM = E;
  using Buf = phf::Buf<E>;

 public:
  using H4 = uint32_t;
  using H = H4;
  using M = PHF_METADATA;
  using module = phf::cuhip::modules<E, H>;
  using phf_module = phf::cuhip::modules<E, H>;
  using Header = phf_header;

  phf_header header;

  Buf* buf;

  float _time_book{0.0}, _time_lossless{0.0};
  float time_book() const { return _time_book; }
  float time_codec() const { return _time_lossless; }
  float time_lossless() const { return _time_lossless; }
  size_t inlen() const { return len; };

  cudaEvent_t event_start, event_end;

  size_t pardeg, sublen;
  int numSMs;
  size_t len;
  static constexpr uint16_t max_bklen = 1024;
  uint16_t rt_bklen;

  GPU_unique_hptr<uint32_t[]> h_hist;

  phf_dtype const in_dtype;

  // Prevent copying
  HuffmanCodec(const HuffmanCodec&) = delete;
  HuffmanCodec& operator=(const HuffmanCodec&) = delete;

  // TODO Is specifying inlen when constructing proper?
  HuffmanCodec(size_t const inlen, int const pardeg, bool debug = false);
  ~HuffmanCodec();
  HuffmanCodec* buildbook(uint32_t* d_hist_ext, uint16_t const rt_bklen, phf_stream_t);
  // alternatively, it can force check the input array
  HuffmanCodec* encode(E*, size_t const, PHF_BYTE**, size_t*, phf_stream_t);
  HuffmanCodec* decode(PHF_BYTE*, E*, phf_stream_t, bool = true);
  HuffmanCodec* clear_buffer();
  HuffmanCodec* dump_internal_data(std::string, std::string);

 private:
  void make_metadata();
};

struct HuffmanHelper {
  static const int BLOCK_DIM_ENCODE = 256;
  static const int BLOCK_DIM_DEFLATE = 256;

  static const int ENC_SEQUENTIALITY = 4;  // empirical
  static const int DEFLATE_CONSTANT = 4;   // deflate_chunk_constant
};

}

#define HF_SPACE phf::Buf<E>
#define HF_STREAM void*

namespace phf {

template <typename E>
struct high_level {
  static int build_book(HF_SPACE* buf, uint32_t* h_hist, uint16_t const runtime_bklen,
                        HF_STREAM stream);

  static int encode(HF_SPACE* buf, E* in_data, size_t const data_len,
                    uint8_t** out_encoded, size_t* encoded_len,
                    phf_header& header, HF_STREAM stream);

  static int encode_ReVISIT_lite(HF_SPACE* buf, E* in_data,
                                 size_t const data_len, uint8_t** out_encoded,
                                 size_t* encoded_len, phf_header& header,
                                 HF_STREAM stream);

  static int decode(HF_SPACE* buf, phf_header& header, PHF_BYTE* in_encoded,
                    E* out_decoded, HF_STREAM stream);
};

}  // namespace phf

#endif /* HF_HL_HH */