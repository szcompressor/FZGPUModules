#pragma once

#include "codec.hh"
#include "config.hh"
#include "mem/buffer.hh"
#include "metrics.hh"
#include "predictor.hh"
#include "stat.hh"
#include "timer.hh"

namespace fz {

template <typename T>
struct Decompressor {

  Config<T>* conf = nullptr;
  CompressorBufferToggle* toggle = nullptr;
  InternalBuffers<T>* ibuffer = nullptr;
  fzmod_metrics* metrics = nullptr;

  Decompressor(std::string fname, bool toFile = true) {
    conf = new Config<T>(fname);
    conf->toFile = toFile;
    std::string basename = fname.substr(0, fname.rfind("."));
    conf->fname = basename;
    toggle = new CompressorBufferToggle();
    metrics = new fzmod_metrics();
    ibuffer = new InternalBuffers<T>(conf, toggle, false);
  }

  Decompressor(Config<T>& config) : conf(&config) {
    // check if config is from previous compression
    if (conf->hedear->offsets[4] == 0) {
      throw std::runtime_error("Invalid configuration: The header offsets are not set correctly. \
        likely this is not from a previous compression.");
    }
    toggle = new CompressorBufferToggle();
    metrics = new fzmod_metrics();
    ibuffer = new InternalBuffers<T>(conf, toggle, false);
  }

  ~Decompressor() {
    if (ibuffer) delete ibuffer;
    if (toggle) delete toggle;
    if (metrics) delete metrics;
    if (conf) delete conf;
  }

  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

  void decompress(uint8_t* compressed_data, T* out_data, cudaStream_t stream, T* orig_data = nullptr) {
    if (orig_data != nullptr) {
      conf->compare = true;
    }

    CREATE_CPU_TIMER(total)
    START_CPU_TIMER(total)

    // DECOMPRESSION SETUP
    cudaMemsetAsync(out_data, 0, conf->orig_size, stream);

    // ~~~~~~~~~~~~~~~~~~~~~~~ CODEC 2 ~~~~~~~~~~~~~~~~~~~~~~~ //

    if (conf->lossless_codec_2 == SECONDARY_CODEC::GZIP) {
      throw std::runtime_error("GZIP decompression is not implemented yet");
    } else if (conf->lossless_codec_2 == SECONDARY_CODEC::ZSTD) {
      zstd(compressed_data);
      compressed_data = ibuffer->d_internal_temp.get();
    } else if (conf->lossless_codec_2 == SECONDARY_CODEC::NONE) {
      // No secondary codec
    } else {
      throw std::runtime_error("Unsupported secondary codec type");
    }

    // static const int HEADER = 0;
    static const int ANCHOR = 1;
    static const int ENCODED = 2;
    static const int OUTLIER = 3;
    static const int END = 4;

    auto access = [&](int FIELD, size_t offset_nbyte = 0) {
      return (void*)(compressed_data + conf->header->offsets[FIELD] + offset_nbyte);
    };

    auto d_anchor = (T*)access(ANCHOR);
    auto d_out_vals = (T*)access(OUTLIER);
    auto d_out_idx = (uint32_t*)access(OUTLIER, conf->num_outliers * sizeof(T));

    if (conf->num_outliers != 0) {
      float ms = 0;
      fz::spv_scatter_naive<T, uint32_t>(d_out_vals, d_out_idx, conf->num_outliers, out_data, &ms, stream);
      metrics->scatter_time = ms;
    }

    // ~~~~~~~~~~~~~~~~~~~~~~~ CODEC 1 ~~~~~~~~~~~~~~~~~~~~~~~ //

    CREATE_CPU_TIMER(decoding)
    START_CPU_TIMER(decoding)
    if (conf->codec == CODEC::HUFFMAN) {
      huffman((uint8_t*)access(ENCODED), stream);
    } else if (conf->codec == CODEC::FZG) {
      fzg((uint8_t*)access(ENCODED), stream);
    } else {
      throw std::runtime_error("Unsupported codec type");
    }
    STOP_CPU_TIMER(decoding)
    TIME_ELAPSED_CPU_TIMER(decoding, metrics->decoding_time)

    // ~~~~~~~~~~~~~~~~~~~~~~~ PREDICTION ~~~~~~~~~~~~~~~~~~~~~~~ //

    CREATE_CPU_TIMER(prediction)
    START_CPU_TIMER(prediction)
    if (conf->algo == ALGO::LORENZO) {
      lorenzo(out_data, stream);
    } else if (conf->algo == ALGO::SPLINE) {
      spline(out_data, stream);
    } else {
      throw std::runtime_error("Unsupported algorithm type");
    }
    STOP_CPU_TIMER(prediction)
    TIME_ELAPSED_CPU_TIMER(prediction, metrics->prediction_reversing_time)

    STOP_CPU_TIMER(total)
    TIME_ELAPSED_CPU_TIMER(total, metrics->end_to_end_decomp_time)

    // ~~~~~~~~~~~~~~~~~~~~~~~ END DECOMPRESSION ~~~~~~~~~~~~~~~~~~~~~~~ //

    if (conf->verbose) {
      conf->print();
    }

    if (conf->toFile) {
      CREATE_CPU_TIMER(file_io)
      START_CPU_TIMER(file_io)

      auto decompressed_fname = conf->fname + ".fzmodx";
      auto decompressed_file_data_host = MAKE_UNIQUE_HOST(T, conf->len);
      cudaMemcpy(decompressed_file_data_host.get(), out_data, conf->orig_size, cudaMemcpyDeviceToHost);
      utils::tofile(decompressed_fname.c_str(), decompressed_file_data_host.get(), conf->len);

      STOP_CPU_TIMER(file_io)
      TIME_ELAPSED_CPU_TIMER(file_io, metrics->decomp_file_io_time);
    }

    if (conf->report) {
      metrics->precision = conf->precision;
      metrics->num_outliers = conf->num_outliers;
      metrics->data_len = conf->len;
      metrics->orig_bytes = conf->len * sizeof(T);
      metrics->comp_bytes = conf->header->offsets[END];
      metrics->compression_ratio = static_cast<double>(metrics->orig_bytes) / metrics->comp_bytes;
      metrics->final_eb = conf->eb;
      if (conf->compare) {
        metrics->compare<T>(orig_data, out_data, conf->len, stream);
      }
      metrics->print(false, conf->compare);
    }

  } // end decompress

  // ~~~~~~~~~~~~~~~~~~~~~~~ CODEC FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~~ //

  void zstd(uint8_t* src) {
    size_t src_len;
    src_len = conf->comp_size - sizeof(fzmod_header) - sizeof(size_t);
    
    // get original size from header and move past it
    size_t dst_len;
    cudaMemcpy(&dst_len, src, sizeof(size_t), cudaMemcpyDeviceToHost);
    src += sizeof(size_t); // move past src_len

    // allocate space for decompression
    ibuffer->d_internal_temp = MAKE_UNIQUE_DEVICE(uint8_t, dst_len);
    uint8_t* dst = ibuffer->d_internal_temp.get();

    // decompress using zstd
    ZSTD_decompress(dst, dst_len, src, src_len);
    
  } // end zstd

  void huffman(uint8_t* encoded, cudaStream_t stream) {
    phf_header h;
    cudaMemcpy((uint8_t*)&h, encoded, sizeof(phf_header), cudaMemcpyDeviceToHost);
    phf::high_level<uint16_t>::decode(ibuffer->buf_hf, h, encoded, ibuffer->codes(), stream);
  } // end huffman

  void fzg(uint8_t* encoded, cudaStream_t stream) {
    ibuffer->codec_fzg->decode(encoded, ibuffer->codes(), conf->len, stream);
  } // end fzg

  // ~~~~~~~~~~~~~~~~~~~~~~~ PREDICTOR FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~~ //

  void lorenzo(T* out_data, cudaStream_t stream) {
    std::array<size_t, 3> len3 = {conf->x, conf->y, conf->z};
    double eb = conf->eb;
    double ebx2 = eb * 2;
    double ebx2_r = 1 / ebx2;

    if (conf->use_lorenzo_zigzag) {
      fz::module::GPU_x_lorenzo_nd<T, true, uint16_t>(
        ibuffer->codes(), out_data, out_data, len3, ebx2, ebx2_r, conf->radius, stream);
    } else {
      fz::module::GPU_x_lorenzo_nd<T, false, uint16_t>(
        ibuffer->codes(), out_data, out_data, len3, ebx2, ebx2_r, conf->radius, stream);
    }
  } // end lorenzo

  void spline(T* out_data, cudaStream_t stream) {
    std::array<size_t, 3> len3 = {conf->x, conf->y, conf->z};
    double eb = conf->eb;
    double eb_r = 1 / eb;
    double ebx2 = eb * 2;

    fz::module::GPU_reverse_predict_spline(
      ibuffer->codes(), len3, ibuffer->anchor_points(),
      conf->anchor_len3(), out_data, len3, ebx2, eb_r, conf->radius, stream);
  } // end spline

}; // end Decompressor

} // namespace fz