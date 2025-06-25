#pragma once

#include <cuda/experimental/stf.cuh> // CUDASTF

#include "fzmod.hh" // Main FZMod API

// CUDASTF includes
#include "tools/ibuffer.hh"
#include "modules/hist_generic_stf.cu"
#include "modules/hist_sparse_stf.cu"
#include "modules/huffman_class.hh"
#include "modules/lorenzo_1d.cu"
#include "modules/spvn_stf.cu"

namespace utils = _portable::utils;
using namespace cuda::experimental::stf;

namespace fz {

template <typename T>
struct STF_Compressor {

  Config<T>* conf = nullptr;
  CompressorBufferToggle* toggle = nullptr;
  fzmod_metrics* metrics = nullptr;
  context* ctx = nullptr;
  HuffmanCodecSTF<T>* codec_hf = nullptr;
  stf_internal_buffers<T>* ibuffer = nullptr;

  STF_Compressor(Config<T>& config, context& ctx, T* input_data_host) : conf(&config), ctx(&ctx) {
    toggle = new CompressorBufferToggle();
    module_optimizer();
    metrics = new fzmod_metrics();
    codec_hf = new HuffmanCodecSTF<T>(conf->len, conf->pardeg, ctx);
    ibuffer = new stf_internal_buffers<T>(ctx, conf->len, input_data_host,
      codec_hf->revbk4_bytes, codec_hf->max_bklen * sizeof(uint32_t), conf->pardeg);
    if (conf->eb_type == EB_TYPE::REL) {
      double _max_val, _min_val, _range;
      GPU_unique_dptr<T[]> input_data_device = MAKE_UNIQUE_DEVICE(T, conf->len);
      cuda_safe_call(cudaMemcpy(input_data_device.get(), input_data_host, conf->len * sizeof(T), cudaMemcpyHostToDevice));
      // Probe extrema on the device
      fz::analysis::GPU_probe_extrema<T>(input_data_device.get(), conf->len, _max_val, _min_val, _range);
      conf->eb *= _range;
      conf->logging_max = _max_val;
      conf->logging_min = _min_val;
      metrics->max = _max_val;
      metrics->min = _min_val;
      metrics->range = _range;
    }
  } // end Compressor constructor

  ~STF_Compressor() {
    if (ibuffer) delete ibuffer;
    if (toggle) delete toggle;
    if (metrics) delete metrics;
    if (codec_hf) delete codec_hf;
  }  // end Compressor destructor

  size_t compress(cudaStream_t stream) {
    float ms_total = 0;
    cudaEvent_t start, stop;

    cuda_safe_call(cudaEventCreate(&start));
    cuda_safe_call(cudaEventCreate(&stop));

    cuda_safe_call(cudaEventRecord(start, ctx->task_fence()));

    //! Initialize the buffers
    ctx->task(
      ibuffer->l_compressed.write(), ibuffer->l_q_codes.write(),
      ibuffer->l_top1.write(), ibuffer->l_hist.write(),
      ibuffer->l_out_vals.write(), ibuffer->l_out_idxs.write(),
      ibuffer->l_out_num.write(), ibuffer->l_revbk4.write(),
      ibuffer->l_bk4.write(), ibuffer->codeword_hist.write(),
      ibuffer->l_scratch.write(), ibuffer->l_par_nbit.write(),
      ibuffer->l_par_ncell.write(), ibuffer->l_par_entry.write(),
      ibuffer->l_bitstream.write(), ibuffer->hf_header_entry.write(),
      ibuffer->out_entries.write())
      .set_symbol("init_memeory")->*[&]
      (cudaStream_t s, 
      auto compressed, auto q_c, auto l_t1, 
      auto l_h, auto o_v, auto o_i, 
      auto o_n, auto rev_bk4, auto bk4,
      auto codeword_hist, auto l_scratch, auto l_par_nbit,
      auto l_par_ncell, auto l_par_entry, auto l_bitstream,
      auto hf_header_entry, auto out_entries) 
    {
      int const END = 4;
      cuda_safe_call(
          cudaMemsetAsync(compressed.data_handle(), 0, conf->len * 4 / 2, s));
      cuda_safe_call(
        cudaMemsetAsync(q_c.data_handle(), 0, conf->len * sizeof(uint16_t), s));
      cuda_safe_call(
        cudaMemsetAsync(l_t1.data_handle(), 0, sizeof(uint32_t), s));
      cuda_safe_call(
        cudaMemsetAsync(l_h.data_handle(), 0, 1024 * sizeof(uint32_t), s));
      cuda_safe_call(cudaMemsetAsync(o_v.data_handle(), 0,
                                     (conf->len / 5) * sizeof(float), s));
      cuda_safe_call(cudaMemsetAsync(o_i.data_handle(), 0,
                                     (conf->len / 5) * sizeof(uint32_t), s));
      cuda_safe_call(
        cudaMemsetAsync(o_n.data_handle(), 0, sizeof(uint32_t), s));
      cuda_safe_call(cudaMemsetAsync(rev_bk4.data_handle(), 0xff, 
          codec_hf->revbk4_bytes, s));
      cuda_safe_call(cudaMemsetAsync(bk4.data_handle(), 0xff, 
          codec_hf->max_bklen * sizeof(uint32_t), s));
      cuda_safe_call(cudaMemsetAsync(codeword_hist.data_handle(), 0, 
          1024 * sizeof(uint32_t), s));
      cuda_safe_call(cudaMemsetAsync(l_scratch.data_handle(), 0,
                                     conf->len * sizeof(uint32_t), s));
      cuda_safe_call(cudaMemsetAsync(l_par_nbit.data_handle(), 0,
                                     conf->pardeg * sizeof(uint32_t), s));
      cuda_safe_call(cudaMemsetAsync(l_par_ncell.data_handle(), 0,
                                     conf->pardeg * sizeof(uint32_t), s));
      cuda_safe_call(cudaMemsetAsync(l_par_entry.data_handle(), 0,
                                     conf->pardeg * sizeof(uint32_t), s));
      cuda_safe_call(cudaMemsetAsync(l_bitstream.data_handle(), 0,
                                     (conf->len / 2) * sizeof(uint32_t), s));
      cuda_safe_call(
        cudaMemsetAsync(hf_header_entry.data_handle(), 0, 
          6 * sizeof(uint32_t), s));
      cuda_safe_call(
        cudaMemsetAsync(out_entries.data_handle(), 0, 
          (END + 1) * sizeof(uint32_t), s));
    };

    if (conf->algo == ALGO::SPLINE) {
      spline();
    } else if (conf->algo == ALGO::LORENZO) {
      lorenzo();
    } else {
      throw std::runtime_error("Unsupported compression algorithm");
    }

    if (conf->codec == CODEC::HUFFMAN) {
      histogram();
      huffman(stream);
    } else if (conf->codec == CODEC::FZG) {
      throw std::runtime_error("FZG codec is not implemented yet in CUDASTF.");
    } else {
      throw std::runtime_error("Unsupported codec type");
    }

    static const int HEADER = 0;
    static const int ANCHOR = 1;
    static const int ENCODED = 2;
    static const int SPFMT = 3;
    static const int END = 4;

    //! fill in the entry array for data segments using prefix sum
    ctx->host_launch(ibuffer->l_out_num.read()).set_symbol("h_entries_fill")->*[&]
      (auto out_num) 
    {
      conf->num_outliers = out_num(0);
      uint32_t nbyte[END] = {0, 0, 0, 0};
      nbyte[HEADER] = sizeof(fzmod_header);
      nbyte[ENCODED] = ibuffer->codec_comp_output_len * sizeof(uint8_t);
      nbyte[ANCHOR] = 0;
      nbyte[SPFMT] = conf->num_outliers * (sizeof(T) + sizeof(uint32_t));

      uint32_t offsets[END + 1] = {0, 0, 0, 0, 0};
      offsets[0] = 0;
      for (auto i = 1; i < END + 1; i++) offsets[i] = nbyte[i - 1];
      for (auto i = 1; i < END + 1; i++) offsets[i] += offsets[i - 1];

      int END = sizeof(offsets) / sizeof(offsets[0]);
      conf->comp_size = offsets[END - 1];
      conf->populate_header(offsets);
      if (conf->verbose) {
        conf->print();
      }
    };

    cuda_safe_call(cudaStreamSynchronize(ctx->task_fence()));

    //! concat data on device
    ctx->task(ibuffer->l_compressed.rw(), 
      ibuffer->l_out_vals.read(), ibuffer->l_out_idxs.read())
      .set_symbol("concat_bitstream")->*[&]
      (cudaStream_t s, auto compressed, auto out_vals, 
      auto out_idxs) 
    {
      cuda_safe_call(cudaMemcpyAsync(
          compressed.data_handle() + conf->header->offsets[SPFMT], 
          out_vals.data_handle(),
          conf->num_outliers * sizeof(T), 
          cudaMemcpyDeviceToDevice, s));
      cuda_safe_call(cudaMemcpyAsync(
          compressed.data_handle() + conf->header->offsets[SPFMT] + 
          (conf->num_outliers * sizeof(T)),
          out_idxs.data_handle(), 
          conf->num_outliers * sizeof(uint32_t),
          cudaMemcpyDeviceToDevice, s));
    };

    cuda_safe_call(cudaEventRecord(stop, ctx->task_fence()));

    cuda_safe_call(cudaStreamSynchronize(ctx->task_fence()));

    if (conf->toFile) {

      auto file = MAKE_UNIQUE_HOST(uint8_t, conf->comp_size);

      cuda_safe_call(cudaStreamSynchronize(ctx->task_fence()));
    
      //! copy header to compressed buffer
      ctx->task(ibuffer->l_compressed.read())
        .set_symbol("gpu_comp_to_file")->*[&]
        (cudaStream_t s, auto compressed) 
      {
        cuda_safe_call(
          cudaMemcpy(file.get(), compressed.data_handle(), conf->comp_size,
            cudaMemcpyDeviceToHost));
      };
    
      cuda_safe_call(cudaStreamSynchronize(ctx->task_fence()));

      //! Output Data To File
      ctx->host_launch().set_symbol("file_header_and_output")->*[&]() {
        auto compressed_fname = conf->fname + ".stf_compressed";

        // Copy compressed data from device to host
        std::memcpy(file.get(), conf->header, sizeof(fzmod_header));
        utils::tofile(compressed_fname.c_str(), file.get(), conf->comp_size);
      };
    }

    ctx->finalize();

    cuda_safe_call(cudaEventElapsedTime(&ms_total, start, stop));
    metrics->end_to_end_comp_time = ms_total;

    if (conf->report) {
      metrics->precision = conf->precision;
      metrics->num_outliers = conf->num_outliers;
      metrics->data_len = conf->len;
      metrics->orig_bytes = conf->len * sizeof(T);
      metrics->comp_bytes = conf->comp_size;
      metrics->compression_ratio =
          static_cast<double>(metrics->orig_bytes) / metrics->comp_bytes;
      metrics->final_eb = conf->eb;
      metrics->print();
    }

    return conf->comp_size;
  } // end compress

  // ###################################################### //

  void lorenzo() {
    if (conf->y != 1 || conf->z != 1) {
      throw std::runtime_error("Lorenzo compression only supports 1D data in STF.");
    }
    ctx->task(
      ibuffer->l_uncomp.rw(), ibuffer->l_q_codes.write(),
      ibuffer->l_out_vals.write(), ibuffer->l_out_idxs.write(), 
      ibuffer->l_out_num.write(), ibuffer->l_top1.write())
      .set_symbol("lorenzo_1d")->*[&]
      (cudaStream_t s, 
       auto l_u, auto q_c, auto o_v,
       auto o_i, auto o_n, auto l_t1) 
    {
      lorenzo_1d<T>(l_u, conf->len, q_c, o_v, o_i, o_n, l_t1, conf->eb * 2,
                  1 / (conf->eb * 2), 512, s);
    };
  } // end lorenzo

  void spline() {
    throw std::runtime_error("Spline compression is not implemented yet in CUDASTF.");
  } // end spline

  void histogram() {
    if (!conf->use_histogram_sparse) {
      //! Generic 2013 Histogram Kernel
      ctx->task(ibuffer->l_q_codes.read(), ibuffer->l_hist.rw())
        .set_symbol("hist_generic")->*[&]
        (cudaStream_t s, auto q_c, auto l_h) 
      {
        kernel_hist_generic<<<conf->h_gen_grid_d, conf->h_gen_block_d, conf->h_gen_shmem_use, s>>>(
          q_c, conf->len, l_h, conf->radius * 2, conf->h_gen_repeat);
      };
    } else {
      ctx->task(ibuffer->l_q_codes.read(), ibuffer->l_hist.rw())
        .set_symbol("hist_sparse")->*[&]
        (cudaStream_t s, auto q_c, auto l_h) 
      {
        throw std::runtime_error("Sparse histogram is not working yet in CUDASTF.");
        constexpr auto chunk = 32768;
        constexpr auto num_workers = 256;
        auto num_chunks = (conf->len - 1) / chunk + 1;
        kernel_hist_sparse<<<num_chunks, num_workers, sizeof(uint16_t) * conf->radius * 2, s>>>(
          q_c, conf->len, l_h, conf->radius * 2, chunk, conf->radius);
      };
    }
  } // end histogram

  void huffman(cudaStream_t stream) {
    codec_hf->buildbook(codec_hf->max_bklen, *ibuffer, *ctx, stream);
    codec_hf->encode(conf->pardeg, *ibuffer, *ctx, stream);
  } // end huffman

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
    }

    if (conf->codec == CODEC::HUFFMAN) {
      if (!conf->use_histogram_sparse) {
        histogram_optimizer(
            conf->len, conf->radius * 2, conf->h_gen_grid_d,
            conf->h_gen_block_d, conf->h_gen_shmem_use, conf->h_gen_repeat);
      }
      capi_phf_coarse_tune(conf->len, &conf->sublen, &conf->pardeg);
    } else if (conf->codec == CODEC::FZG) {
      // FZG specific optimizations can be added here
    } else {
      throw std::runtime_error("Unsupported codec type for optimization");
    }
  } // end module_optimizer

}; // end Compressor

template <typename T>
struct STF_Decompressor {

  Config<T>* conf = nullptr;
  CompressorBufferToggle* toggle = nullptr;
  fzmod_metrics* metrics = nullptr;
  context* ctx = nullptr;
  HuffmanCodecSTF<T>* codec_hf = nullptr;
  stf_internal_buffers<T>* ibuffer = nullptr;

  STF_Decompressor(std::string fname, context& ctx) : ctx(&ctx) {
    conf = new Config<T>(fname);
    toggle = new CompressorBufferToggle();
    std::string basename = fname.substr(0, fname.rfind("."));
    conf->fname = basename;
    metrics = new fzmod_metrics();
    ibuffer = new stf_internal_buffers<T>(ctx, conf->len, nullptr, 0, 0, 0, false);
    codec_hf = new HuffmanCodecSTF<T>(conf->len, conf->pardeg, ctx);
  } // end Decompressor constructor

  ~STF_Decompressor() {
    if (ibuffer) delete ibuffer;
    if (toggle) delete toggle;
    if (metrics) delete metrics;
    if (codec_hf) delete codec_hf;
    if (conf) delete conf;
  } // end Decompressor destructor

  void decompress(uint8_t* compressed_data_host, cudaStream_t stream, T* original_data_host = nullptr) {
    static const int ENCODED = 2;
    static const int SPFMT = 3;

    hf_header header_hf;

    float ms_total = 0;
    cudaEvent_t start, stop;

    cuda_safe_call(cudaEventCreate(&start));
    cuda_safe_call(cudaEventCreate(&stop));

    cuda_safe_call(cudaEventRecord(start, ctx->task_fence()));

    throw std::runtime_error("STF decompression is not implemented yet in CUDASTF.");

    // size_t encoded_size = conf->header->offsets[SPFMT] - 
    //                       conf->header->offsets[ENCODED];
    // auto l_encoded = ctx->logical_data(shape_of<slice<uint8_t>>(encoded_size));
    // auto l_spval = ctx->logical_data(shape_of<slice<T>>(conf->num_outliers));
    // auto l_spidx = ctx->logical_data(shape_of<slice<uint32_t>>(conf->num_outliers));

    // uint8_t* encoded_data_h;
    // cudaMallocHost(&encoded_data_h, encoded_size);
    // std::memcpy(encoded_data_h, 
    //             compressed_data_host + conf->header->offsets[ENCODED], 
    //             encoded_size);
    // T* outlier_vals_h = nullptr;
    // uint32_t* outlier_idx_h = nullptr;

    // if (conf->num_outliers > 0) {
    //   outlier_vals_h = (T*)malloc(conf->num_outliers * sizeof(T));
    //   outlier_idx_h = (uint32_t*)malloc(conf->num_outliers * sizeof(uint32_t));
    //   std::memcpy(outlier_vals_h, 
    //               compressed_data_host + conf->header->offsets[SPFMT],
    //               conf->num_outliers * sizeof(T));
    //   std::memcpy(outlier_idx_h, 
    //               compressed_data_host + conf->header->offsets[SPFMT] + 
    //               (conf->num_outliers * sizeof(T)),
    //               conf->num_outliers * sizeof(uint32_t));
    // }
    
    // //! Populate internal buffers with zero
    // ctx->task(ibuffer->l_uncompressed.write(), 
    //   l_encoded.write(), l_spval.write(), 
    //   l_spidx.write()).set_symbol("populate_outliers")->*[&]
    //   (cudaStream_t s, auto uncompressed, auto encoded, 
    //   auto spval, auto spidx) 
    // {
    //   cuda_safe_call(
    //     cudaMemsetAsync(uncompressed.data_handle(), 0, 
    //     conf->len * sizeof(T), s));
    //   cuda_safe_call(
    //     cudaMemcpyAsync(encoded.data_handle(), encoded_data_h, 
    //     encoded_size, cudaMemcpyHostToDevice, s));
    //   if (conf->num_outliers > 0) {
    //     cuda_safe_call(
    //       cudaMemcpyAsync(spval.data_handle(), outlier_vals_h, 
    //       conf->num_outliers * sizeof(T), cudaMemcpyHostToDevice, s));
    //     cuda_safe_call(
    //       cudaMemcpyAsync(spidx.data_handle(), outlier_idx_h, 
    //       conf->num_outliers * sizeof(uint32_t), cudaMemcpyHostToDevice, s));
    //   }
    // };

    // //! Populate the header with the entries
    // ctx->task(
    //   l_encoded.read())
    //   .set_symbol("hf_header_populate")
    //   ->*[&](cudaStream_t s, auto encoded) {
    //   cuda_safe_call(cudaMemcpyAsync(
    //     &header_hf, encoded.data_handle(), 
    //     sizeof(hf_header), cudaMemcpyDeviceToHost, s));
    // };

    // cuda_safe_call(cudaStreamSynchronize(ctx->task_fence()));

    // //! scatter outliers to uncompressed buffer
    // ctx->task(
    //   l_spval.read(), 
    //   l_spidx.read(), 
    //   ibuffer->l_uncompressed.rw())
    //   .set_symbol("spv_scatter")->*[&]
    //   (cudaStream_t s, 
    //   auto val,
    //   auto idx, 
    //   auto uncompressed) 
    // {
    //   if (conf->num_outliers > 0) {
    //     spvn_scatter<T><<<(conf->num_outliers - 1) / 128 + 1, 128, 0, s>>>(
    //         val, idx, conf->num_outliers, uncompressed);
    //   }
    // };

    // //! decode the huffman encoded data
    // ctx->task(
    //   l_encoded.rw(), 
    //   ibuffer->l_q_codes.write())
    //   .set_symbol("hf_decode")->*[&]
    //   (cudaStream_t s, 
    //   auto encoded, 
    //   auto q_codes) 
    // {
    //   auto div = [](auto l, auto subl) { return (l - 1) / subl + 1; };
    //   auto const block_dim = 256;
    //   auto const grid_dim = div(conf->pardeg, block_dim);

    //   #define ACCESSOR(SYM, TYPE) reinterpret_cast<TYPE*>( \
    //     encoded.data_handle() + header_hf.entry[HF_HEADER_##SYM])

    //   hf_kernel_decode<<<grid_dim, block_dim, codec_hf->revbk4_bytes, s>>>(
    //     ACCESSOR(BITSTREAM, uint32_t), ACCESSOR(REVBK, uint8_t), 
    //     ACCESSOR(PAR_NBIT, uint32_t), ACCESSOR(PAR_ENTRY, uint32_t), 
    //     codec_hf->revbk4_bytes, conf->sublen, conf->pardeg, q_codes);
    // };

    // //! decompress the lorenzo quant codes
    // ctx->task(
    //   ibuffer->l_q_codes.rw(), 
    //   ibuffer->l_uncompressed.rw())
    //   .set_symbol("lorenzo_decomp")->*[&]
    //   (cudaStream_t s, 
    //   auto q_codes, 
    //   auto uncompressed) 
    // {
    //   lorenzo_decomp_1d<T>(q_codes, uncompressed, conf->len, 
    //     conf->eb*2, 1/(conf->eb*2), conf->radius, s);
    // };

    cuda_safe_call(cudaEventRecord(stop, ctx->task_fence()));

    // ctx->host_launch(
    //   ibuffer->l_uncompressed.read())
    //   .set_symbol("decomp_to_file")->*[&]
    //   (auto uncompressed) 
    // {
    //   auto decompressed_fname = conf->fname + ".stf_decompressed";
    //   utils::tofile(decompressed_fname.c_str(), uncompressed.data_handle(), conf->len);
    // };
  
    // if (conf->num_outliers > 0) {
    //   free(outlier_vals_h);
    //   free(outlier_idx_h);
    // }
    // cudaFreeHost(encoded_data_h);

    ctx->finalize();

    cuda_safe_call(cudaEventElapsedTime(&ms_total, start, stop));
    metrics->end_to_end_decomp_time = ms_total;

    // if (conf->verbose) {
    //   conf->print();
    // }

    // if (conf->report) {
    //   metrics->precision = conf->precision;
    //   metrics->num_outliers = conf->num_outliers;
    //   metrics->data_len = conf->len;
    //   metrics->orig_bytes = conf->len * sizeof(T);
    //   metrics->comp_bytes = conf->header->offsets[SPFMT];
    //   metrics->compression_ratio =
    //       static_cast<double>(metrics->orig_bytes) / metrics->comp_bytes;
    //   metrics->final_eb = conf->eb;
    //   //TODO: compare logic here
    //   metrics->print(false);
    // }
  } // end decompress

  // ~~~~~~~~~~~~~~~~~~~~~~~~ Compare ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //

  // void compare(std::string fname, size_t len1, size_t len2, size_t len3) {

  //   float* original_data_device;
  //   float* decompressed_data_device;
  //   cudaMalloc(&original_data_device, len1 * len2 * len3 * sizeof(float));
  //   cudaMalloc(&decompressed_data_device, len1 * len2 * len3 *
  //   sizeof(float));

  //   float* original_data_host;
  //   float* decompressed_data_host;
  //   cudaMallocHost(&original_data_host, len1 * len2 * len3 * sizeof(float));
  //   cudaMallocHost(&decompressed_data_host, len1 * len2 * len3 *
  //   sizeof(float)); utils::fromfile(fname, original_data_host, len1 * len2 *
  //   len3); utils::fromfile(fname + ".stf_decompressed",
  //   decompressed_data_host, len1 * len2 * len3);

  //   cudaMemcpy(original_data_device, original_data_host,
  //     len1 * len2 * len3 * sizeof(float), cudaMemcpyHostToDevice);
  //   cudaMemcpy(decompressed_data_device, decompressed_data_host,
  //     len1 * len2 * len3 * sizeof(float), cudaMemcpyHostToDevice);

  //   // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //

  //   constexpr auto MINVAL = 0;
  //   constexpr auto MAXVAL = 1;
  //   constexpr auto AVGVAL = 2;

  //   constexpr auto SUM_CORR = 0;
  //   constexpr auto SUM_ERR_SQ = 1;
  //   constexpr auto SUM_VAR_ODATA = 2;
  //   constexpr auto SUM_VAR_XDATA = 3;

  //   float orig_data_res[4], decomp_data_res[4];

  //   fz::module::GPU_extrema(original_data_device,
  //     len1 * len2 * len3, orig_data_res);
  //   fz::module::GPU_extrema(decompressed_data_device,
  //     len1 * len2 * len3, decomp_data_res);

  //   float h_err[4];

  //   fz::module::GPU_calculate_errors<float>(
  //       original_data_device, orig_data_res[AVGVAL],
  //       decompressed_data_device, decomp_data_res[AVGVAL], len1 * len2 *
  //       len3, h_err);

  //   double std_orig_data = sqrt(h_err[SUM_VAR_ODATA] / (len1 * len2 * len3));
  //   double std_decomp_data = sqrt(h_err[SUM_VAR_XDATA] / (len1 * len2 *
  //   len3)); double ee = h_err[SUM_CORR] / (len1 * len2 * len3);

  //   float max_abserr{0};
  //   size_t max_abserr_index{0};
  //   fz::module::GPU_find_max_error<float>(
  //       decompressed_data_device, original_data_device,
  //       len1 * len2 * len3, max_abserr, max_abserr_index);

  //   printf("Original Data Min: %f\n", orig_data_res[MINVAL]);
  //   printf("Original Data Max: %f\n", orig_data_res[MAXVAL]);
  //   printf("Original Data Range: %f\n", orig_data_res[MAXVAL] -
  //   orig_data_res[MINVAL]); printf("Original Data Mean: %f\n",
  //   orig_data_res[AVGVAL]); printf("Original Data Stddev: %f\n",
  //   std_orig_data); printf("Decompressed Data Min: %f\n",
  //   decomp_data_res[MINVAL]); printf("Decompressed Data Max: %f\n",
  //   decomp_data_res[MAXVAL]); printf("Decompressed Data Range: %f\n",
  //   decomp_data_res[MAXVAL] - decomp_data_res[MINVAL]); printf("Decompressed
  //   Data Mean: %f\n", decomp_data_res[AVGVAL]); printf("Decompressed Data
  //   Stddev: %f\n", std_decomp_data);

  //   printf("Max Absolute Error: %f\n", max_abserr);
  //   printf("Max Absolute Error Index: %zu\n", max_abserr_index);

  //   printf("Correlation Coefficient: %f\n", ee);
  //   double mse = h_err[SUM_ERR_SQ] / (len1 * len2 * len3);
  //   printf("Mean Squared Error: %f\n", mse);
  //   printf("Normalized Root Mean Squared Error: %f\n", sqrt(mse) /
  //     (orig_data_res[MAXVAL] - orig_data_res[MINVAL]));
  //   printf("Peak Signal-to-Noise Ratio: %f\n", 20 *
  //     log10(orig_data_res[MAXVAL] - orig_data_res[MINVAL]) - 10 *
  //     log10(mse));

  //   // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //

  //   cudaFree(original_data_device);
  //   cudaFree(decompressed_data_device);
  //   cudaFreeHost(original_data_host);
  //   cudaFreeHost(decompressed_data_host);
  // }

}; // end Decompressor

} // namespace fz