#pragma once 

#include <cstdint>
#include <cstddef>
#include <string>
#include <iostream>

#define HEADER_SIZE 128
#define MAGIC 0x465A4D4F // "FZMO" in ASCII
#define VERSION 100 // version 1.0.0

#define HEADER_OFFSET 0
#define HEADER_ANCHOR 1
#define HEADER_ENCODED 2
#define HEADER_SPFMT 3
#define HEADER_END 4

namespace fz {

enum class EB_TYPE : uint32_t { REL, ABS };
enum class ALGO : uint32_t { LORENZO, SPLINE };
enum class PRECISION : uint32_t { FLOAT, DOUBLE };
enum class CODEC : uint32_t { HUFFMAN, FZG };
enum class SECONDARY_CODEC : uint32_t { NONE, GZIP, LSTD };

typedef struct alignas(1) fzmod_header {
  uint32_t magic = MAGIC; // magic number for fzmod files
  uint32_t version = VERSION; // version of the fzmod file format
  PRECISION precision; // precision of the data
  ALGO algo; // algorithm used for compression
  uint32_t hist_type; // type of histogram used
  CODEC codec; // codec used for compression
  SECONDARY_CODEC lossless_codec_2;  // secondary cpu lossless codec, if any
  EB_TYPE eb_type; // type of error bound
  double eb; // error bound value
  uint16_t radius; // radius for the histogram
  int sublen; // sublength
  int pardeg; // degree of parallelism 
  uint32_t offsets[5]; // entries for the header
  uint32_t x; // x dimension of the data
  uint32_t y; // y dimension of the data  
  uint32_t z; // z dimension of the data
  uint32_t num_outliers; // number of outliers in the data
  double user_input_eb; // user input error bound
  double logging_max; // maximum value for logging
  double logging_min; // minimum value for logging
  uint8_t reserved[18]; // reserved to make header exactly 128 bytes

} __attribute__((packed)) fzmod_header;

// static assert 128 bytes
static_assert(sizeof(fzmod_header) == HEADER_SIZE, "fzmod_header must be exactly 128 bytes");

}