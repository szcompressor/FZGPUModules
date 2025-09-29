#pragma once

#include <array>
#include "header.hh"
#include "io.hh"

namespace utils = _portable::utils;

namespace fz {

  template <typename T>
  class Config {
    public:

      // #### FILE IO #### //
      fzmod_header* header;
      bool toFile = true;
      bool fromFile = false;
      std::string fname = "";

      // #### REPORTING #### //
      bool report = true;
      bool compare = false;
      bool dump = false;
      bool verbose = false;
      bool dryrun = false;

      // #### PIPELINE #### //
      bool comp = true;
      double eb = 1e-3;
      EB_TYPE eb_type = EB_TYPE::REL;
      ALGO algo = ALGO::LORENZO;
      PRECISION precision = PRECISION::FLOAT;
      CODEC codec = CODEC::HUFFMAN;
      SECONDARY_CODEC lossless_codec_2 = SECONDARY_CODEC::NONE;

      // #### DATA #### //
      uint32_t x;
      uint32_t y;
      uint32_t z;
      size_t len;
      size_t num_outliers = 0;
      float outlier_buffer_ratio = 0.2f;

      size_t orig_size;
      size_t comp_size;

      double logging_max, logging_min = 0.0;

      // #### HISTOGRAM #### //
      uint16_t radius = 512;
      bool use_histogram_sparse = false;
      int h_gen_grid_d, h_gen_block_d, h_gen_shmem_use, h_gen_repeat;

      // #### HUFFMAN #### //
      int sublen;
      int pardeg;
      bool use_huffman_reVISIT = false;

      // #### LORENZO #### //
      int lorenzo_prediction_dimension = 0;
      bool use_lorenzo_zigzag = false;

      // #### SPLINE #### //
      size_t anchor512_len;
      constexpr static size_t BLK = 8;

      // helper function to calculate the number of anchors
      static size_t _div(size_t _l, size_t _subl) { return (_l - 1) / _subl + 1; };

      Config(size_t _x, size_t _y = 1, size_t _z = 1) : x(_x), y(_y), z(_z),
        anchor512_len(_div(_x, BLK) * _div(_y, BLK) * _div(_z, BLK)) {
        header = new fzmod_header();
        len = x * y * z;
        orig_size = len * sizeof(T);
        comp_size = 0;
        precision = std::is_same<T, float>::value ? PRECISION::FLOAT : PRECISION::DOUBLE;
      }

      Config(std::string fname) : anchor512_len(0) {
        auto file = MAKE_UNIQUE_HOST(uint8_t, HEADER_SIZE);
        utils::fromfile(fname.c_str(), file.get(), HEADER_SIZE);
        header = new fzmod_header();
        std::memcpy(header, file.get(), sizeof(fzmod_header));

        // check magic number and version
        if (header->magic != MAGIC) {
          printf("Magic number: %x\n", header->magic);
          throw std::runtime_error("Invalid magic number in header");
        }
        if (header->version != VERSION) {
          printf("Version: %u\n", header->version);
          throw std::runtime_error("Unsupported version in header");
        }

        // Populate the config from the header
        x = header->x;
        y = header->y;
        z = header->z;
        len = x * y * z;
        orig_size = len * sizeof(T);
        comp_size = header->offsets[4];
        eb = header->eb;
        eb_type = header->eb_type;
        algo = header->algo;
        if (algo == ALGO::LORENZO_ZZ) {
          use_lorenzo_zigzag = true;
          algo = ALGO::LORENZO;
        }
        precision = header->precision;
        codec = header->codec;
        lossless_codec_2 = header->lossless_codec_2;
        radius = header->radius;
        sublen = header->sublen;
        pardeg = header->pardeg;
        num_outliers = header->num_outliers;
        logging_max = header->logging_max;
        logging_min = header->logging_min;
        anchor512_len = _div(x, BLK) * _div(y, BLK) * _div(z, BLK);
      }

      // anchor len for spline predictor anchor points
      std::array<size_t, 3> anchor_len3() const {
        return std::array<size_t, 3>{_div(x, BLK), _div(y, BLK), _div(z, BLK)};
      }

      void populate_header(uint32_t* offsets) {
        header->magic = MAGIC;
        header->version = VERSION;
        header->precision = precision;
        header->algo = algo;
        header->hist_type = use_histogram_sparse ? 0 : 1; // 0 for sparse, 1 for generic
        header->codec = codec;
        header->lossless_codec_2 = lossless_codec_2;
        header->eb_type = eb_type;
        header->eb = eb;
        header->radius = radius;
        header->sublen = sublen;
        header->pardeg = pardeg;

        // populate entries
        for (int i = 0; i < 5; ++i) {
          header->offsets[i] = offsets[i];
        }

        header->x = x;
        header->y = y;
        header->z = z;
        header->num_outliers = num_outliers;
        header->user_input_eb = eb; // or some other value
        header->logging_max = logging_max;
        header->logging_min = logging_min;
      }

      ~Config() {
        delete header;
      }
      Config(const Config&) = delete;
      Config& operator=(const Config&) = delete;

      void print() {
        printf("========================= FZMOD CONFIG ========================\n");
        printf("File Name: %s\n", fname.c_str());
        printf("Dimensions: %u x %u x %u\n", x, y, z);
        printf("Length: %zu\n", len);
        printf("Original Size: %zu bytes\n", orig_size);
        printf("Compression Enabled: %s\n", comp ? "Yes" : "No");
        printf("Error Bound: %.3e (%s)\n", eb, eb_type == EB_TYPE::REL ? "Relative" : "Absolute");
        printf("Algorithm: %s\n", algo == ALGO::LORENZO || algo == ALGO::LORENZO_ZZ ? "Lorenzo" : "Spline");
        if (use_lorenzo_zigzag) {
          printf("Lorenzo Zigzag: Enabled\n");
        } else {
          printf("Lorenzo Zigzag: Disabled\n");
        }
        printf("Use Histogram Sparse: %s\n", use_histogram_sparse ? "Yes" : "No");
        printf("Precision: %s\n", precision == PRECISION::FLOAT ? "Float" : "Double");
        std::string codec_str = (codec == CODEC::HUFFMAN) ? "Huffman" :
                                (codec == CODEC::FZG) ? "FZG" : "PFPL";
        printf("Codec: %s\n", codec_str.c_str());
        printf("Secondary Codec: %s\n", lossless_codec_2 == SECONDARY_CODEC::NONE ? "None" : 
               (lossless_codec_2 == SECONDARY_CODEC::GZIP ? "Gzip" : "LSTD"));
        printf("Radius: %u\n", radius);
        printf("Sublength: %d\n", sublen);
        printf("Parallel Degree: %d\n", pardeg);
        printf("Number of Outliers: %zu\n", num_outliers);
        printf("Outlier Buffer Ratio: %f\n", outlier_buffer_ratio);
        printf("Logging Max: %f\n", logging_max);
        printf("Logging Min: %f\n", logging_min);
        printf("Anchor Length: %zu\n", anchor512_len);
        printf("Offsets:\n");
        for (int i = 0; i < 5; ++i) {
          printf("  Offset %d: %u\n", i, header->offsets[i]);
        }
        printf("===============================================================\n\n");
      }

      void save(std::string fname) {
        auto file = MAKE_UNIQUE_HOST(uint8_t, HEADER_SIZE);
        std::memcpy(file.get(), header, sizeof(fzmod_header));
        utils::tofile(fname.c_str(), file.get(), HEADER_SIZE);
      }
  };

} // namespace fz
