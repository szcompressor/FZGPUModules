#pragma once

#include "codec.hh"
#include "config.hh"
#include "mem/buffer.hh"
#include "predictor.hh"
#include "stat.hh"
#include "timer.hh"
#include "metrics.hh"

namespace fz {

template <typename T>
struct Compressor {

  Config<T>* conf = nullptr;
  CompressorBufferToggle* toggle = nullptr;
  InternalBuffers<T>* ibuffer = nullptr;
  fzmod_metrics* metrics = nullptr;

  Compressor(Config<T>& config) : conf(&config) {
    toggle = new CompressorBufferToggle();
    if (conf->fromFile && conf->toFile) {
      toggle->_internal = true;
      if (conf->fname == "") {
        throw std::runtime_error("File name must be provided for from file and to file operations in config");
      }
    }
    module_optimizer();
    ibuffer = new InternalBuffers<T>(conf, toggle, true);
    metrics = new fzmod_metrics();
    metrics->total_footprint_d = ibuffer->total_footprint_d;
    metrics->total_footprint_h = ibuffer->total_footprint_h;
  }

  ~Compressor() {
    if (ibuffer) delete ibuffer;
    if (toggle) delete toggle;
    if (metrics) delete metrics;
  }

  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

  size_t compress(T* in_data, uint8_t** out_data, cudaStream_t stream) {
    
    CREATE_CPU_TIMER(total)
    START_CPU_TIMER(total)

    // PRE-COMPRESSION SETUP

    if (toggle->_internal) {
      in_data = ibuffer->internal_original;
    }

    if (conf->eb_type == EB_TYPE::REL) {
      CREATE_CPU_TIMER(preproc)
      START_CPU_TIMER(preproc)
      double _max_val, _min_val, _range;
      fz::analysis::GPU_probe_extrema<T>(in_data, conf->len, _max_val, _min_val, _range);
      conf->eb *= _range;
      conf->logging_max = _max_val;
      conf->logging_min = _min_val;
      
      metrics->max = _max_val;
      metrics->min = _min_val;
      metrics->range = _range;
      STOP_CPU_TIMER(preproc)
      TIME_ELAPSED_CPU_TIMER(preproc, metrics->preprocessing_time)
    }

    //~~~~~~~~~~~~~~~~~~~~~~~~~ Predictor ~~~~~~~~~~~~~~~~~~~~~~~~~~ //

    CREATE_CPU_TIMER(predictor)
    START_CPU_TIMER(predictor)
    if (conf->algo == ALGO::LORENZO || conf->algo == ALGO::LORENZO_ZZ) {
      lorenzo(in_data, stream);
    } else if (conf->algo == ALGO::SPLINE) {
      spline(in_data, stream);
    } else {
      throw std::runtime_error("Unsupported algorithm type");
    }
    STOP_CPU_TIMER(predictor)
    TIME_ELAPSED_CPU_TIMER(predictor, metrics->prediction_time)

    // get the number of outliers
    cudaStreamSynchronize(stream);
    cudaMemcpy(ibuffer->h_num.get(), ibuffer->d_num.get(), sizeof(uint32_t),
      cudaMemcpyDeviceToHost);
    conf->num_outliers = *(ibuffer->h_num.get());

    size_t max_num_outliers = conf->outlier_buffer_ratio * conf->len;
    if (conf->num_outliers > max_num_outliers) {
      printf("Number of outliers: %zu, Max allowed: %zu\n",
        conf->num_outliers, max_num_outliers);
      throw std::runtime_error("Number of outliers exceeds reserved buffer size try increasing outlier_buffer_ratio in the config (default 0.2x of input data len) or lower the error bound");
    }

    if (conf->dump) {
      // dump the outliers and quant codes to file
      std::string pred_dump_fname = conf->fname + ".pred_dump";
      size_t pred_dump_size = (conf->num_outliers * sizeof(T) * 2) + sizeof(uint32_t) + (conf->len * sizeof(uint16_t));
      auto pred_dump_file = MAKE_UNIQUE_HOST(uint8_t, pred_dump_size);
      size_t offset = 0;
      // copy number of outliers
      std::memcpy(pred_dump_file.get() + offset, &conf->num_outliers, sizeof(uint32_t));
      offset += sizeof(uint32_t);
      // copy outlier values
      cudaMemcpy(pred_dump_file.get() + offset, ibuffer->h_val.get(), conf->num_outliers * sizeof(T), cudaMemcpyDeviceToHost);
      offset += conf->num_outliers * sizeof(T);
      // copy outlier indices
      cudaMemcpy(pred_dump_file.get() + offset, ibuffer->h_idx.get(), conf->num_outliers * sizeof(uint32_t), cudaMemcpyDeviceToHost);
      offset += conf->num_outliers * sizeof(uint32_t);
      // copy quant codes
      if (toggle->quant_codes) {
        cudaMemcpy(pred_dump_file.get() + offset, ibuffer->codes(), conf->len * sizeof(uint16_t), cudaMemcpyDeviceToHost);
      } else {
        std::memset(pred_dump_file.get() + offset, 0, conf->len * sizeof(uint16_t));
      }
      utils::tofile(pred_dump_fname.c_str(), pred_dump_file.get(), pred_dump_size);
      printf("Dumped prediction data to %s\n", pred_dump_fname.c_str());
    }

    // // print first 100 quant codes for debugging
    // auto h_codes_owner = MAKE_UNIQUE_HOST(uint16_t, conf->len);
    // uint16_t* h_codes = h_codes_owner.get();
    // cudaMemcpy(h_codes, ibuffer->codes(), conf->len * sizeof(uint16_t), cudaMemcpyDeviceToHost);
    // printf("First 100 quant codes: ");
    // for (size_t i = 0; i < std::min<size_t>(1000, conf->len); i++) {
    //   if (i % 50 == 0) printf("\n");
    //   printf("%hu ", h_codes[i]);
    // }
    // printf("\n");

    //~~~~~~~~~~~~~~~~~~~~~~~~~ Lossless Encoder 1 ~~~~~~~~~~~~~~~~~~~~~~~~~~ //

    if (conf->codec == CODEC::HUFFMAN) {
      CREATE_CPU_TIMER(hist)
      START_CPU_TIMER(hist)  
      histogram(stream);
      STOP_CPU_TIMER(hist)
      TIME_ELAPSED_CPU_TIMER(hist, metrics->hist_time)

      if (conf->dump) {
        // dump the histogram to file
        std::string hist_dump_fname = conf->fname + ".hist_dump";
        size_t hist_dump_size = conf->radius * 2 * sizeof(uint32_t);
        auto hist_dump_file = MAKE_UNIQUE_HOST(uint8_t, hist_dump_size);
        cudaMemcpy(hist_dump_file.get(), ibuffer->h_hist.get(), hist_dump_size, cudaMemcpyDeviceToHost);
        utils::tofile(hist_dump_fname.c_str(), hist_dump_file.get(), hist_dump_size);
        printf("Dumped histogram data to %s\n", hist_dump_fname.c_str());
      }

      CREATE_CPU_TIMER(encoder)
      START_CPU_TIMER(encoder)
      huffman(stream);
      STOP_CPU_TIMER(encoder)
      TIME_ELAPSED_CPU_TIMER(encoder, metrics->encoder_time)
    } else if (conf->codec == CODEC::FZG) {
      CREATE_CPU_TIMER(encoder)
      START_CPU_TIMER(encoder)
      fzg(stream);
      STOP_CPU_TIMER(encoder)
      TIME_ELAPSED_CPU_TIMER(encoder, metrics->encoder_time)
    } else if (conf->codec == CODEC::PFPL) {
      CREATE_CPU_TIMER(encoder)
      START_CPU_TIMER(encoder)
      pfpl(stream);
      STOP_CPU_TIMER(encoder)
      TIME_ELAPSED_CPU_TIMER(encoder, metrics->encoder_time)
    } else {
      throw std::runtime_error("Unsupported codec type");
    }

    //~~~~~~~~~~~~~~~~~~~~~~~~~ Finalize Compression ~~~~~~~~~~~~~~~~~~~~~~~~~~ //

    // construct offsets array for file
    static const int HEADER = 0;
    static const int ANCHOR = 1;
    static const int ENCODED = 2;
    static const int OUTLIER = 3;
    static const int END = 4;

    uint32_t nbyte[END] = {0, 0, 0, 0};
    nbyte[HEADER] = sizeof(fzmod_header);
    nbyte[ANCHOR] =
        conf->algo == ALGO::SPLINE ? conf->anchor512_len * sizeof(float) : 0;
    nbyte[ENCODED] = ibuffer->codec_comp_output_len * sizeof(uint8_t);
    nbyte[OUTLIER] = conf->num_outliers * (sizeof(T) + sizeof(uint32_t));

    uint32_t offsets[END + 1] = {0, 0, 0, 0, 0};
    offsets[0] = 0;
    for (auto i = 1; i < END + 1; i++) offsets[i] = nbyte[i - 1];
    for (auto i = 1; i < END + 1; i++) offsets[i] += offsets[i - 1];

    #define DST(FIELD, OFFSET) ((void*)(ibuffer->compressed() + offsets[FIELD] + OFFSET))
    #define CONCAT_ON_DEVICE(dst, src, nbyte, stream) \
      if (nbyte != 0) cudaMemcpyAsync(dst, src, nbyte, cudaMemcpyDeviceToDevice, stream);

    // concatenate buffers on device
    CONCAT_ON_DEVICE(DST(ANCHOR, 0), ibuffer->anchor_points(), nbyte[ANCHOR], stream);
    CONCAT_ON_DEVICE(DST(ENCODED, 0), ibuffer->codec_comp_output, nbyte[ENCODED], stream);
    CONCAT_ON_DEVICE(DST(OUTLIER, 0), ibuffer->outlier_values(), conf->num_outliers * sizeof(T), stream);
    CONCAT_ON_DEVICE(DST(OUTLIER, conf->num_outliers * sizeof(T)), ibuffer->outlier_indices(), conf->num_outliers * sizeof(uint32_t), stream);

    // set the output data pointer
    *out_data = ibuffer->compressed();
    conf->comp_size = offsets[END];

    //~~~~~~~~~~~~~~~~~~~~~~~~~ Lossless Encoder 2 ~~~~~~~~~~~~~~~~~~~~~~~~~~ //

    if (conf->lossless_codec_2 == SECONDARY_CODEC::GZIP) {
      throw std::runtime_error("GZIP compression is not implemented yet");
    } else if (conf->lossless_codec_2 == SECONDARY_CODEC::ZSTD) {
      zstd();
    } else if (conf->lossless_codec_2 == SECONDARY_CODEC::NONE) {
      // No secondary codec
    } else {
      throw std::runtime_error("Unsupported secondary codec type");
    }

    STOP_CPU_TIMER(total)
    TIME_ELAPSED_CPU_TIMER(total, metrics->end_to_end_comp_time)

    // ~~~~~~~~~~~~~~~~~~~ COMPRESSION FINISHED ~~~~~~~~~~~~~~~~~~~ //

    conf->populate_header(offsets);
    if (conf->verbose) {
      conf->print();
    }

    // output compressed data to disk
    if (conf->toFile) {
      CREATE_CPU_TIMER(file_io)
      START_CPU_TIMER(file_io)
      auto compressed_fname = conf->fname + ".fzmod";
      auto file = MAKE_UNIQUE_HOST(uint8_t, conf->comp_size);

      // copy header info to beginning of file
      cudaMemcpy(file.get(), ibuffer->compressed(), conf->comp_size, cudaMemcpyDeviceToHost);
      std::memcpy(file.get(), conf->header, sizeof(fzmod_header));
      utils::tofile(compressed_fname.c_str(), file.get(), conf->comp_size);
      STOP_CPU_TIMER(file_io)
      TIME_ELAPSED_CPU_TIMER(file_io, metrics->file_io_time)
    }

    // report metrics about compression
    if (conf->report) {
      metrics->precision = conf->precision;
      metrics->num_outliers = conf->num_outliers;
      metrics->data_len = conf->len;
      metrics->orig_bytes = conf->len * sizeof(T);
      metrics->comp_bytes = conf->comp_size;
      metrics->compression_ratio = static_cast<double>(metrics->orig_bytes) / metrics->comp_bytes;
      metrics->final_eb = conf->eb;
      metrics->print();
    }

    return conf->comp_size;

  } // end compress

  // ~~~~~~~~~~~~~~~~~~~ PREPROCESSING ~~~~~~~~~~~~~~~~~~~ //

  void module_optimizer() {    
    if (conf->algo == ALGO::LORENZO) {
      toggle->quant_codes = true;
      toggle->top1 = true;
    } else if (conf->algo == ALGO::SPLINE) {
      toggle->quant_codes = true;
      toggle->anchor_points = true;
    }
    if (conf->codec == CODEC::HUFFMAN) {
      toggle->histogram = true;
      toggle->top1 = true;
    } else if (conf->codec == CODEC::FZG) {
      
    } else if (conf->codec == CODEC::PFPL) {

    }

    if (conf->codec == CODEC::HUFFMAN) {
      if (!conf->use_histogram_sparse) {
        fz::module::GPU_histogram_generic_optimizer_on_initialization<uint16_t>(
          conf->len, conf->radius * 2, conf->h_gen_grid_d, 
          conf->h_gen_block_d, conf->h_gen_shmem_use,
          conf->h_gen_repeat);
      }
      capi_phf_coarse_tune(conf->len, &conf->sublen, &conf->pardeg);
    } else if (conf->codec == CODEC::FZG) {
      // FZG specific optimizations can be added here
    } else if (conf->codec == CODEC::PFPL) {
      // PFPL specific optimizations can be added here
    } else {
      throw std::runtime_error("Unsupported codec type for optimization");
    }

  } // end module_optimizer

  // ################ PREDICTOR FUNCTIONS ################ //

  void lorenzo(T* input_data, cudaStream_t stream) {
    std::array<size_t, 3> len3 = std::array<size_t, 3>{conf->x, conf->y, conf->z};
    double eb = conf->eb;
    double ebx2 = eb * 2;
    double ebx2_r = 1 / ebx2;

    if (conf->use_lorenzo_zigzag) {
      fz::module::GPU_c_lorenzo_nd_with_outlier<T, true, uint16_t>(
          input_data, len3, ibuffer->codes(), ibuffer->outlier_values(), 
          ibuffer->outlier_indices(), ibuffer->num_outliers_d(), ibuffer->top1(),
          ebx2, ebx2_r, conf->radius, ibuffer->outlier_reserve_size,
          stream);
    } else {
      fz::module::GPU_c_lorenzo_nd_with_outlier<T, false, uint16_t>(
          input_data, len3, ibuffer->codes(), ibuffer->outlier_values(),
          ibuffer->outlier_indices(), ibuffer->num_outliers_d(),
          ibuffer->top1(), ebx2, ebx2_r, conf->radius, ibuffer->outlier_reserve_size,
          stream);
    }
  } // end lorenzo

  void spline(T* input_data, cudaStream_t stream) {
    std::array<size_t, 3> len3 = std::array<size_t, 3>{conf->x, conf->y, conf->z};
    double eb = conf->eb;
    double eb_r = 1 / eb;
    double ebx2 = eb * 2;
    fz::module::GPU_predict_spline(
        input_data, len3, ibuffer->codes(), len3, ibuffer->anchor_points(),
        conf->anchor_len3(), ibuffer->outlier_values(), ibuffer->outlier_indices(),
        ibuffer->num_outliers_d(), ebx2, eb_r, conf->radius, stream);
  } // end spline

  // ################ STAT FUNCTIONS ################ //

  void histogram(cudaStream_t stream) {
    if (conf->use_histogram_sparse) {
      fz::module::GPU_histogram_Cauchy<uint16_t>(
          ibuffer->codes(), conf->len, ibuffer->histogram(),
          conf->radius * 2, stream);
    } else {
      fz::module::GPU_histogram_generic<uint16_t>(
          ibuffer->codes(), conf->len, ibuffer->histogram(),
          static_cast<uint16_t>(conf->radius * 2),
          conf->h_gen_grid_d, conf->h_gen_block_d,
          conf->h_gen_shmem_use, conf->h_gen_repeat, stream);
    }
  } // end histogram

  // ################ CODEC 1 FUNCTIONS ################ //

  void huffman(cudaStream_t stream) {
    cudaMemcpy(ibuffer->h_hist.get(), ibuffer->d_hist.get(),
               conf->radius * 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    phf::high_level<uint16_t>::build_book(ibuffer->buf_hf, 
      ibuffer->h_hist.get(), conf->radius * 2, stream);
    phf_header dummy_header;
    if (conf->use_huffman_reVISIT) {
      phf::high_level<uint16_t>::encode_ReVISIT_lite(
        ibuffer->buf_hf, ibuffer->codes(), conf->len,
        &ibuffer->codec_comp_output, &ibuffer->codec_comp_output_len,
        dummy_header, stream);
    } else {
      phf::high_level<uint16_t>::encode(ibuffer->buf_hf, ibuffer->codes(),
        conf->len, &ibuffer->codec_comp_output, &ibuffer->codec_comp_output_len,
        dummy_header, stream);
    }
  } // end huffman

  void fzg(cudaStream_t stream) {
    ibuffer->codec_fzg->encode(
      ibuffer->codes(), conf->len, &ibuffer->codec_comp_output,
      &ibuffer->codec_comp_output_len, stream);
  } // end fzg

  void pfpl(cudaStream_t stream) {

    const int num_codes = conf->len - conf->num_outliers;

    // set PFPL kernel launch parameters
    ibuffer->codec_pfpl->set_kernel_params(num_codes);

    ibuffer->codec_pfpl->encode(
      ibuffer->codes(), num_codes, &ibuffer->codec_comp_output,
      &ibuffer->codec_comp_output_len, stream);

    printf("PFPL finished... Compressed size: %zu bytes\n", ibuffer->codec_comp_output_len);
  }

  // ################ CODEC 2 FUNCTIONS ################ //

  void zstd() {
    int compression_level = 3; // Default compression level
    GPU_unique_hptr<uint8_t[]> src = MAKE_UNIQUE_HOST(uint8_t, conf->comp_size);
    cudaMemcpy(src.get(), ibuffer->compressed(), conf->comp_size, cudaMemcpyDeviceToHost);

    // Allocate memory for the compressed output
    size_t max_compressed_size = ZSTD_compressBound(conf->comp_size);
    ibuffer->h_compressed = MAKE_UNIQUE_HOST(uint8_t, max_compressed_size + sizeof(size_t));
    uint8_t* dst = ibuffer->h_compressed.get();

    memcpy(dst, &conf->comp_size, sizeof(size_t));
    size_t dst_len = ZSTD_compress(dst + sizeof(size_t), max_compressed_size - sizeof(size_t),
                                  src.get(), conf->comp_size, compression_level);
    
    if (ZSTD_isError(dst_len)) {
      throw std::runtime_error("ZSTD compression failed: " + std::string(ZSTD_getErrorName(dst_len)));
    }

    if (dst_len >= conf->comp_size * 0.9) {
      size_t temp_comp_len = dst_len + sizeof(size_t) + 128;
      printf("ZSTD compression did not reduce size sufficiently, "
             "original size: %zu, compressed size: %zu, consider disabling it\n",
             conf->comp_size, temp_comp_len);
      throw std::runtime_error("ZSTD compression did not reduce size sufficiently");
    }

    ibuffer->d_compressed = MAKE_UNIQUE_DEVICE(uint8_t, dst_len + sizeof(size_t));
    cudaMemcpy(ibuffer->d_compressed.get(), dst, dst_len + sizeof(size_t), cudaMemcpyHostToDevice);

    conf->comp_size = dst_len + sizeof(size_t) + 128; // include header size
  } // end zstd

}; // end Compressor

} // namespace fz