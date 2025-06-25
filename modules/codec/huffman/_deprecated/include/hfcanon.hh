/**
 * @file hfcanon.hh
 * @author Jiannan Tian
 * @brief Canonization of existing Huffman codebook (header).
 * @version 0.1
 * @date 2020-09-20
 * Created on 2020-04-10
 *
 * @copyright (C) 2020 by Washington State University, The University of
 * Alabama, Argonne National Laboratory See LICENSE in top-level directory
 *
 */

#ifndef B684F0FA_8869_4DDF_9467_2E28E967AC06
#define B684F0FA_8869_4DDF_9467_2E28E967AC06

#include <cstdint>
#include <cstring>

// #include "type.h"

template <typename E, typename H>
class hf_space {
 public:
  static const int TYPE_BITS = sizeof(H) * 8;

  static uint32_t space_bytes(int const bklen)
  {
    return sizeof(H) * (3 * bklen) + sizeof(uint32_t) * (4 * TYPE_BITS) +
           sizeof(E) * bklen;
  }

  static uint32_t revbook_bytes(int const bklen)
  {
    return sizeof(uint32_t) * (2 * TYPE_BITS) + sizeof(E) * bklen;
  }

  static uint32_t revbook_offset(int const bklen)
  {
    return sizeof(H) * (3 * bklen) + sizeof(uint32_t) * (2 * TYPE_BITS);
  }
};

template <typename E, typename H>
int canonize_on_gpu(uint8_t* bin, uint32_t bklen, void* stream);

template <typename E, typename H>
int canonize(uint8_t* bin, uint32_t const bklen);


///////////////////////////////////////////////////////////////////////////////


template <typename E = uint32_t, typename H = uint32_t>
class hf_canon_reference {
 private:
  uint16_t const booklen;  // Move booklen first to match initialization order
  H *_icb, *_ocb, *_canon;
  int *_numl, *_iterby, *_first, *_entry;
  E* _keys;

 public:
  static const auto TYPE_BITS = sizeof(H) * 8;

  // public fn
  hf_canon_reference(uint16_t booklen)
      : booklen(booklen),  // Now matches declaration order
        _icb(nullptr),
        _ocb(new H[booklen]),
        _canon(new H[booklen]),
        _numl(new int[TYPE_BITS]),
        _iterby(new int[TYPE_BITS]),
        _first(new int[TYPE_BITS]),
        _entry(new int[TYPE_BITS]),
        _keys(new E[booklen]) {
    // Zero out the memory
    memset(_ocb, 0, sizeof(H) * booklen);
    memset(_canon, 0, sizeof(H) * booklen);
    memset(_numl, 0, sizeof(int) * TYPE_BITS);
    memset(_iterby, 0, sizeof(int) * TYPE_BITS);
    memset(_first, 0, sizeof(int) * TYPE_BITS);
    memset(_entry, 0, sizeof(int) * TYPE_BITS);
    memset(_keys, 0, sizeof(E) * booklen);
  }

  ~hf_canon_reference() {
    // delete[] _icb;
    delete[] _ocb;
    delete[] _canon;
    delete[] _keys;
    delete[] _numl;
    delete[] _iterby;
    delete[] _first;
    delete[] _entry;
  }

  hf_canon_reference(const hf_canon_reference&) = delete;
  hf_canon_reference& operator=(const hf_canon_reference&) = delete;

  hf_canon_reference(hf_canon_reference&&) = delete;
  hf_canon_reference& operator=(hf_canon_reference&&) = delete;

  // accessor
  H*& input_bk() { return _icb; }
  H* output_bk() { return _ocb; }
  H* canon() { return _canon; }
  E* keys() { return _keys; }
  int* numl() { return _numl; }
  int* iterby() { return _iterby; }
  int* first() { return _first; }
  int* entry() { return _entry; }

  H& input_bk(int i) { return _icb[i]; }
  H& output_bk(int i) { return _ocb[i]; }
  H& canon(int i) { return _canon[i]; }
  E& keys(int i) { return _keys[i]; }
  int& numl(int i) { return _numl[i]; }
  int& iterby(int i) { return _iterby[i]; }
  int& first(int i) { return _first[i]; }
  int& entry(int i) { return _entry[i]; }
  // run
  int canonize();
};

#endif /* B684F0FA_8869_4DDF_9467_2E28E967AC06 */
