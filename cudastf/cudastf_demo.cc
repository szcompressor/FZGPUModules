#include <cuda_runtime.h>
#include <sys/time.h>

#include <cuda/experimental/stf.cuh>
#include <fstream>
#include <iostream>
#include <vector>

#include "fzmod_compressor.hh"
#include "fzmod_decompressor.hh"
#include "ibuffer.hh"
#include "hist_generic_stf.cu"
#include "hist_sparse_stf.cu"
#include "huffman_class.hh"
#include "lorenzo_1d.cu"
#include "spvn_stf.cu"

namespace utils = _portable::utils;
using namespace cuda::experimental::stf;

static const int HEADER = 0;
static const int ANCHOR = 1;
static const int ENCODED = 2;
static const int SPFMT = 3;
static const int END = 4;

float* compressed_final;
float* decompressed_final;

//! Timing utility structure
struct Timer {
  struct timeval start_time;

  void start() { gettimeofday(&start_time, NULL); }

  double elapsed() {
    struct timeval current_time;
    gettimeofday(&current_time, NULL);
    return (current_time.tv_sec - start_time.tv_sec) +
           (current_time.tv_usec - start_time.tv_usec) * 1e-6;
  }
};

struct FZModPerformance {
  double prediction_time;
  double histogram_time;
  double huffman_time;
  double total_time;
  std::vector<double> comp_time;
  std::vector<double> communication_time;
};

// ~~~~~~~~~~~~~~~~~~~~~~~ Compress ~~~~~~~~~~~~~~~~~~~~~~~ //

void compress(std::string fname, size_t len1, size_t len2, size_t len3) {

  size_t data_len = len1 * len2 * len3;
  double eb = 2e-4;
  size_t compressed_len = 0;

  // make the cudastream
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // get dataset
  float* input_data_host;
  cudaMallocHost(&input_data_host, data_len * sizeof(float));
  utils::fromfile(fname, input_data_host, data_len);

  // histogram variables
  int hist_grid_d, hist_block_d, hist_shmem_use, hist_repeat;

  // huffman variables
  int sublen, pardeg;

  uint32_t h_entries[END + 1] = {0, 0, 0, 0, 0};
  uint32_t h_nbyte[END] = {0, 0, 0, 0};

  size_t num_outliers = 0;

  // histogram kernel optimization
  histogram_optimizer(data_len, 1024, hist_grid_d, 
    hist_block_d, hist_shmem_use, hist_repeat);

  // huffman kernel optimizer
  capi_phf_coarse_tune(data_len, &sublen, &pardeg);

  // CUDASTF context
  context ctx;

  // create Huffman codec
  HuffmanCodecSTF codec_hf(data_len, pardeg, ctx);

  // allocate internal memory buffers
  stf_internal_buffers ibuffer(ctx, 
    data_len, 
    input_data_host,
    codec_hf.revbk4_bytes,
    codec_hf.max_bklen * sizeof(uint32_t), pardeg);

  float ms_total = 0;
  cudaEvent_t start, stop;

  cuda_safe_call(cudaEventCreate(&start));
  cuda_safe_call(cudaEventCreate(&stop));

  cuda_safe_call(cudaEventRecord(start, ctx.task_fence()));
  
  //! Initialize the buffers
  ctx.task(
    ibuffer.l_compressed.write(),
    ibuffer.l_q_codes.write(),
    ibuffer.l_top1.write(),
    ibuffer.l_hist.write(),
    ibuffer.l_out_vals.write(),
    ibuffer.l_out_idxs.write(),
    ibuffer.l_out_num.write(),
    ibuffer.l_revbk4.write(),
    ibuffer.l_bk4.write(),
    ibuffer.codeword_hist.write(),
    ibuffer.l_scratch.write(),
    ibuffer.l_par_nbit.write(),
    ibuffer.l_par_ncell.write(),
    ibuffer.l_par_entry.write(),
    ibuffer.l_bitstream.write(),
    ibuffer.hf_header_entry.write(),
    ibuffer.out_entries.write())
    .set_symbol("init_memeory")->*[&]
    (cudaStream_t s, 
     auto compressed, 
     auto q_c, 
     auto l_t1, 
     auto l_h, 
     auto o_v, 
     auto o_i, 
     auto o_n, 
     auto rev_bk4, 
     auto bk4,
     auto codeword_hist,
     auto l_scratch,
     auto l_par_nbit,
     auto l_par_ncell,
     auto l_par_entry,
     auto l_bitstream,
     auto hf_header_entry,
     auto out_entries) 
  {
    cuda_safe_call(
      cudaMemsetAsync(compressed.data_handle(), 0, 
        data_len * 4 / 2, s));
    cuda_safe_call(
      cudaMemsetAsync(q_c.data_handle(), 0, data_len * sizeof(uint16_t), s));
    cuda_safe_call(
      cudaMemsetAsync(l_t1.data_handle(), 0, sizeof(uint32_t), s));
    cuda_safe_call(
      cudaMemsetAsync(l_h.data_handle(), 0, 1024 * sizeof(uint32_t), s));
    cuda_safe_call(
      cudaMemsetAsync(o_v.data_handle(), 0, 
        (data_len / 5) * sizeof(float), s));
    cuda_safe_call(
      cudaMemsetAsync(o_i.data_handle(), 0, 
        (data_len / 5) * sizeof(uint32_t), s));
    cuda_safe_call(
      cudaMemsetAsync(o_n.data_handle(), 0, sizeof(uint32_t), s));
    cuda_safe_call(
      cudaMemsetAsync(rev_bk4.data_handle(), 0xff, 
        codec_hf.revbk4_bytes, s));
    cuda_safe_call(
      cudaMemsetAsync(bk4.data_handle(), 0xff, 
        codec_hf.max_bklen * sizeof(uint32_t), s));
    cuda_safe_call(
      cudaMemsetAsync(codeword_hist.data_handle(), 0, 
        1024 * sizeof(uint32_t), s));
    cuda_safe_call(
      cudaMemsetAsync(l_scratch.data_handle(), 0, 
        data_len * sizeof(uint32_t), s));
    cuda_safe_call(
      cudaMemsetAsync(l_par_nbit.data_handle(), 0, 
        pardeg * sizeof(uint32_t), s));
    cuda_safe_call(
      cudaMemsetAsync(l_par_ncell.data_handle(), 0, 
        pardeg * sizeof(uint32_t), s));
    cuda_safe_call(
      cudaMemsetAsync(l_par_entry.data_handle(), 0, 
        pardeg * sizeof(uint32_t), s));
    cuda_safe_call(
      cudaMemsetAsync(l_bitstream.data_handle(), 0, 
        (data_len / 2) * sizeof(uint32_t), s));
    cuda_safe_call(
      cudaMemsetAsync(hf_header_entry.data_handle(), 0, 
        6 * sizeof(uint32_t), s));
    cuda_safe_call(
      cudaMemsetAsync(out_entries.data_handle(), 0, 
        (END + 1) * sizeof(uint32_t), s));
  };

  //! 1D Prototype Lorenzo Kernel
  ctx.task(
    ibuffer.l_uncomp.rw(), 
    ibuffer.l_q_codes.write(),
    ibuffer.l_out_vals.write(), 
    ibuffer.l_out_idxs.write(), 
    ibuffer.l_out_num.write(), 
    ibuffer.l_top1.write())
    .set_symbol("lorenzo_1d")->*[&]
    (cudaStream_t s, 
     auto l_u, 
     auto q_c, 
     auto o_v,
     auto o_i, 
     auto o_n, 
     auto l_t1) 
  {
    lorenzo_1d(l_u, data_len, q_c, o_v, o_i, o_n, l_t1, eb * 2,
                1 / (eb * 2), 512, s);
  };

  //! Generic 2013 Histogram Kernel
  ctx.task(
    ibuffer.l_q_codes.read(), 
    ibuffer.l_hist.rw())
    .set_symbol("hist_generic")->*[&]
    (cudaStream_t s, 
     auto q_c, 
     auto l_h) 
  {
    kernel_hist_generic<<<hist_grid_d, hist_block_d, hist_shmem_use, s>>>(
      q_c, data_len, l_h, 1024, hist_repeat);
  };

  //! Huffman Buildbook CPU Kernel
  codec_hf.buildbook(codec_hf.max_bklen, ibuffer, ctx, stream);

  //! Huffman Encode GPU Kernel
  codec_hf.encode(pardeg, ibuffer, ctx, stream);

  //! fill in the entry array for data segments using prefix sum
  ctx.host_launch(
    ibuffer.l_out_num.read())
    .set_symbol("h_entries_fill")->*[&]
    (auto out_num) 
  {
    num_outliers = out_num(0);
    h_nbyte[HEADER] = 128;
    h_nbyte[ENCODED] = ibuffer.codec_comp_output_len * sizeof(uint8_t);
    h_nbyte[ANCHOR] = 0;
    h_nbyte[SPFMT] = num_outliers * (sizeof(float) + sizeof(uint32_t));

    h_entries[0] = 0;
    for (auto i = 1; i < END + 1; i++) h_entries[i] = h_nbyte[i - 1];
    for (auto i = 1; i < END + 1; i++) h_entries[i] += h_entries[i - 1];

    int END = sizeof(h_entries) / sizeof(h_entries[0]);
    compressed_len = h_entries[END - 1];
  };

  cuda_safe_call(cudaStreamSynchronize(ctx.task_fence()));

  //! concat data on device
  ctx.task(
    ibuffer.l_compressed.rw(), 
    ibuffer.l_out_vals.read(), 
    ibuffer.l_out_idxs.read())
    .set_symbol("concat_bitstream")->*[&]
    (cudaStream_t s, 
     auto compressed, 
     auto out_vals, 
     auto out_idxs) 
  {
    cuda_safe_call(cudaMemcpyAsync(
        compressed.data_handle() + h_entries[SPFMT], 
        out_vals.data_handle(),
        num_outliers * sizeof(float), 
        cudaMemcpyDeviceToDevice, s));
    cuda_safe_call(cudaMemcpyAsync(
        compressed.data_handle() + h_entries[SPFMT] + 
        (num_outliers * sizeof(float)),
        out_idxs.data_handle(), 
        num_outliers * sizeof(uint32_t),
        cudaMemcpyDeviceToDevice, s));
  };

  cuda_safe_call(cudaEventRecord(stop, ctx.task_fence()));

  cuda_safe_call(cudaStreamSynchronize(ctx.task_fence()));

  // create memory to store compressed data for file output
  auto file = MAKE_UNIQUE_HOST(uint8_t, compressed_len);

  cuda_safe_call(cudaStreamSynchronize(ctx.task_fence()));

  //! copy header to compressed buffer
  ctx.task(
    ibuffer.l_compressed.read())
    .set_symbol("gpu_comp_to_file")->*[&]
    (cudaStream_t s, 
     auto compressed) 
  {
    cuda_safe_call(
      cudaMemcpy(file.get(), compressed.data_handle(), compressed_len,
        cudaMemcpyDeviceToHost));
  };

  cuda_safe_call(cudaStreamSynchronize(ctx.task_fence()));

  //! Output Data To File
  ctx.host_launch().set_symbol("file_header_and_output")->*[&]() 
  {
    auto compressed_fname = fname + ".stf_compressed";

    // Create a header with hardcoded enum values
    std::array<uint8_t, 128> header{};
    uint8_t* ptr = header.data();
    size_t offset = 0;

    auto writeField = [&ptr, &offset](
      const void* data, size_t dataSize) {
      
        if (offset + dataSize <= 128) {
        std::memcpy(ptr + offset, data, dataSize);
        offset += dataSize;
      }
    };

    // Use hardcoded values for enums
    uint32_t precision = 0;  // PRECISION::PRECISION_FLOAT
    uint32_t algo = 0;       // ALGO::ALGO_LORENZO
    uint32_t hist_type = 1;  // Generic histogram
    uint32_t codec = 0;      // CODEC::CODEC_HUFFMAN
    uint32_t future_codec = 0;
    uint32_t eb_type = 0;  // EB_TYPE::EB_ABS
    double eb = 2e-4;      // Your error bound
    uint16_t radius = 512;
    int one = 1;  // w = 1

    // Write the header fields
    writeField(&precision, sizeof(uint32_t));
    writeField(&algo, sizeof(uint32_t));
    writeField(&hist_type, sizeof(uint32_t));
    writeField(&codec, sizeof(uint32_t));
    writeField(&future_codec, sizeof(uint32_t));
    writeField(&eb_type, sizeof(uint32_t));
    writeField(&eb, sizeof(double));
    writeField(&radius, sizeof(uint16_t));
    writeField(&sublen, sizeof(int));
    writeField(&pardeg, sizeof(int));

    // Write host_entries
    writeField(h_entries, sizeof(h_entries));

    // Write dimensions
    writeField(&len1, sizeof(uint32_t));
    writeField(&len2, sizeof(uint32_t));
    writeField(&len3, sizeof(uint32_t));
    writeField(&one, sizeof(uint32_t));  // w = 1

    // Write outlier count
    writeField(&num_outliers, sizeof(size_t));

    // Write user input error bound
    writeField(&eb, sizeof(double));
    // Write logging max and min
    double logging_max = 0.0;  //! Replace with actual value
    double logging_min = 0.0;  //! Replace with actual value
    writeField(&logging_max, sizeof(double));
    writeField(&logging_min, sizeof(double));

    // Copy compressed data from device to host
    std::memcpy(file.get(), header.data(), 128);
    utils::tofile(compressed_fname.c_str(), file.get(), compressed_len);
  };

  // ~~~~~~~~~~~~~~~~~~~~~~~ Finalize Tasks ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //

  // finish the tasks
  ctx.finalize();

  cuda_safe_call(cudaEventElapsedTime(&ms_total, start, stop));
  printf("Total Time: %.6f ms\n", ms_total);

  // cleanup 
  cudaFreeHost(input_data_host);
  cudaStreamDestroy(stream);
}

// ~~~~~~~~~~~~~~~~~~~~~~~ Decompressor ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //

void decompress(std::string fname, size_t len1, size_t len2, size_t len3) {
  Timer total_timer;
  FZModPerformance perf;
  total_timer.start();

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  context ctx;

  size_t data_len = len1 * len2 * len3;
  size_t compressed_len = 0;
  uint32_t precision;
  uint32_t algo;
  uint32_t hist_type;
  uint32_t codec;
  uint32_t future_codec;
  uint32_t eb_type;
  double eb;
  uint16_t radius;
  int sublen;
  int pardeg;
  uint32_t entries_file[END + 1];
  uint32_t x, y, z, w;
  size_t splen;
  double user_input_eb;
  double logging_max;
  double logging_min;

  std::string compressed_fname = fname + ".stf_compressed";

  uint8_t* header;
  cudaMallocHost(&header, 128);
  utils::fromfile(compressed_fname, header, 128);

  stf_internal_buffers ibuffer(ctx, data_len, nullptr, 0, 0, 0, false);

  const uint8_t* ptr = header;
  size_t offset = 0;

  auto readField = [&ptr, &offset](void* data, size_t dataSize) {
    if (offset + dataSize > 128) {
      std::cerr << "Buffer overrun detected!" << std::endl;
      std::abort();
    }
    std::memcpy(data, ptr + offset, dataSize);
    offset += dataSize;
  };

  // Read the header fields
  readField(&precision, sizeof(uint32_t));
  readField(&algo, sizeof(uint32_t));
  readField(&hist_type, sizeof(uint32_t));
  readField(&codec, sizeof(uint32_t));
  readField(&future_codec, sizeof(uint32_t));
  readField(&eb_type, sizeof(uint32_t));
  readField(&eb, sizeof(double));
  readField(&radius, sizeof(uint16_t));
  readField(&sublen, sizeof(int));
  readField(&pardeg, sizeof(int));
  readField(&entries_file, sizeof(entries_file));
  readField(&x, sizeof(uint32_t));
  readField(&y, sizeof(uint32_t));
  readField(&z, sizeof(uint32_t));
  readField(&w, sizeof(uint32_t));
  readField(&splen, sizeof(size_t));
  readField(&user_input_eb, sizeof(double));
  readField(&logging_max, sizeof(double));
  readField(&logging_min, sizeof(double));

  // Calculate the compressed length based on entries_file
  compressed_len = entries_file[END];

  // create huffman codec object
  HuffmanCodecSTF codec_hf(data_len, pardeg, ctx);
  hf_header header_hf;

  cudaFreeHost(header);

  uint8_t* compressed_data_host;
  cudaMallocHost(&compressed_data_host, 
    compressed_len * sizeof(uint8_t));
  utils::fromfile(compressed_fname.c_str(), 
    compressed_data_host, compressed_len);

  int encoded_size = entries_file[SPFMT] - entries_file[ENCODED];

  auto l_encoded = ctx.logical_data(shape_of<slice<uint8_t>>(encoded_size));
  auto l_spval = ctx.logical_data(shape_of<slice<float>>(splen));
  auto l_spidx = ctx.logical_data(shape_of<slice<uint32_t>>(splen));

  uint8_t* encoded_data_h;
  cudaMallocHost(&encoded_data_h, encoded_size);
  std::memcpy(encoded_data_h, compressed_data_host + 
    entries_file[ENCODED], encoded_size);
  
  float* outlier_vals_h = nullptr;
  uint32_t* outlier_idx_h = nullptr;
  
  if (splen > 0) {
    outlier_vals_h = (float*)malloc(splen * sizeof(float));
    outlier_idx_h = (uint32_t*)malloc(splen * sizeof(uint32_t));
    
    std::memcpy(outlier_vals_h, 
                compressed_data_host + entries_file[SPFMT], 
                splen * sizeof(float));
                
    std::memcpy(outlier_idx_h, 
                compressed_data_host + entries_file[SPFMT] + splen * sizeof(float),
                splen * sizeof(uint32_t));
  }

  //! Populate internal buffers with zero
  ctx.task(
    ibuffer.l_uncompressed.write(), 
    l_encoded.write(), 
    l_spval.write(), 
    l_spidx.write())
    .set_symbol("populate_outliers")->*[&]
    (cudaStream_t s, 
     auto uncompressed, 
     auto encoded, 
     auto spval, 
     auto spidx) 
  {
    cuda_safe_call(
      cudaMemsetAsync(uncompressed.data_handle(), 0, 
      data_len * sizeof(float), s));
    cuda_safe_call(
      cudaMemcpyAsync(encoded.data_handle(), encoded_data_h, 
      encoded_size, cudaMemcpyHostToDevice, s));
    if (splen > 0) {
      cuda_safe_call(
        cudaMemcpyAsync(spval.data_handle(), outlier_vals_h, 
        splen * sizeof(float), cudaMemcpyHostToDevice, s));
      cuda_safe_call(
        cudaMemcpyAsync(spidx.data_handle(), outlier_idx_h, 
        splen * sizeof(uint32_t), cudaMemcpyHostToDevice, s));
    }
  };

  //! Populate the header with the entries
  ctx.task(
    l_encoded.read())
    .set_symbol("hf_header_populate")
    ->*[&](cudaStream_t s, auto encoded) {
    cuda_safe_call(cudaMemcpyAsync(
      &header_hf, encoded.data_handle(), 
      sizeof(hf_header), cudaMemcpyDeviceToHost, s));
  };

  cuda_safe_call(cudaStreamSynchronize(ctx.task_fence()));

  //! scatter outliers to uncompressed buffer
  ctx.task(
    l_spval.read(), 
    l_spidx.read(), 
    ibuffer.l_uncompressed.rw())
    .set_symbol("spv_scatter")->*[&]
    (cudaStream_t s, 
     auto val,
     auto idx, 
     auto uncompressed) 
  {
    if (splen > 0) {
      spvn_scatter<<<(splen-1)/128+1, 128, 0, s>>>(
        val, idx, splen, uncompressed);
    }
  };

  //! decode the huffman encoded data
  ctx.task(
    l_encoded.rw(), 
    ibuffer.l_q_codes.write())
    .set_symbol("hf_decode")->*[&]
    (cudaStream_t s, 
     auto encoded, 
     auto q_codes) 
  {
    auto div = [](auto l, auto subl) { return (l - 1) / subl + 1; };
    auto const block_dim = 256;
    auto const grid_dim = div(pardeg, block_dim);

    #define ACCESSOR(SYM, TYPE) reinterpret_cast<TYPE*>( \
      encoded.data_handle() + header_hf.entry[HF_HEADER_##SYM])

    hf_kernel_decode<<<grid_dim, block_dim, codec_hf.revbk4_bytes, s>>>(
      ACCESSOR(BITSTREAM, uint32_t), ACCESSOR(REVBK, uint8_t), 
      ACCESSOR(PAR_NBIT, uint32_t), ACCESSOR(PAR_ENTRY, uint32_t), 
      codec_hf.revbk4_bytes, sublen, pardeg, q_codes);
  };

  //! decompress the lorenzo quant codes
  ctx.task(
    ibuffer.l_q_codes.rw(), 
    ibuffer.l_uncompressed.rw())
    .set_symbol("lorenzo_decomp")->*[&]
    (cudaStream_t s, 
     auto q_codes, 
     auto uncompressed) 
  {
    lorenzo_decomp_1d(q_codes, uncompressed, data_len, eb*2, 1/(eb*2), 512, s);
  };

  //! copy decompressed data to file
  ctx.host_launch(
    ibuffer.l_uncompressed.read())
    .set_symbol("decomp_to_file")->*[&]
    (auto uncompressed) 
  {
    auto decompressed_fname = fname + ".stf_decompressed";
    utils::tofile(decompressed_fname.c_str(), uncompressed.data_handle(), data_len);
  };

  if (splen > 0) {
    free(outlier_vals_h);
    free(outlier_idx_h);
  }
  cudaFreeHost(encoded_data_h);
  cudaFreeHost(compressed_data_host);

  // ~~~~~~~~~~~~~~~~~~~~~~~ Finalize Tasks ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //

  ctx.finalize();

  cudaStreamDestroy(stream);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~ Compare ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //

void compare(std::string fname, size_t len1, size_t len2, size_t len3) {

  float* original_data_device;
  float* decompressed_data_device;
  cudaMalloc(&original_data_device, len1 * len2 * len3 * sizeof(float));
  cudaMalloc(&decompressed_data_device, len1 * len2 * len3 * sizeof(float));

  float* original_data_host;
  float* decompressed_data_host;
  cudaMallocHost(&original_data_host, len1 * len2 * len3 * sizeof(float));
  cudaMallocHost(&decompressed_data_host, len1 * len2 * len3 * sizeof(float));
  utils::fromfile(fname, original_data_host, len1 * len2 * len3);
  utils::fromfile(fname + ".stf_decompressed", decompressed_data_host, len1 * len2 * len3);

  cudaMemcpy(original_data_device, original_data_host, 
    len1 * len2 * len3 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(decompressed_data_device, decompressed_data_host, 
    len1 * len2 * len3 * sizeof(float), cudaMemcpyHostToDevice);

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //

  constexpr auto MINVAL = 0;
  constexpr auto MAXVAL = 1;
  constexpr auto AVGVAL = 2;

  constexpr auto SUM_CORR = 0;
  constexpr auto SUM_ERR_SQ = 1;
  constexpr auto SUM_VAR_ODATA = 2;
  constexpr auto SUM_VAR_XDATA = 3;

  float orig_data_res[4], decomp_data_res[4];

  fz::module::GPU_extrema(original_data_device, 
    len1 * len2 * len3, orig_data_res);
  fz::module::GPU_extrema(decompressed_data_device, 
    len1 * len2 * len3, decomp_data_res);

  float h_err[4];

  fz::module::GPU_calculate_errors<float>(
      original_data_device, orig_data_res[AVGVAL], 
      decompressed_data_device, decomp_data_res[AVGVAL], len1 * len2 * len3, h_err);
  
  double std_orig_data = sqrt(h_err[SUM_VAR_ODATA] / (len1 * len2 * len3));
  double std_decomp_data = sqrt(h_err[SUM_VAR_XDATA] / (len1 * len2 * len3));
  double ee = h_err[SUM_CORR] / (len1 * len2 * len3);

  float max_abserr{0};
  size_t max_abserr_index{0};
  fz::module::GPU_find_max_error<float>(
      decompressed_data_device, original_data_device, 
      len1 * len2 * len3, max_abserr, max_abserr_index);
  
  printf("Original Data Min: %f\n", orig_data_res[MINVAL]);
  printf("Original Data Max: %f\n", orig_data_res[MAXVAL]);
  printf("Original Data Range: %f\n", orig_data_res[MAXVAL] - orig_data_res[MINVAL]);
  printf("Original Data Mean: %f\n", orig_data_res[AVGVAL]);
  printf("Original Data Stddev: %f\n", std_orig_data);
  printf("Decompressed Data Min: %f\n", decomp_data_res[MINVAL]);
  printf("Decompressed Data Max: %f\n", decomp_data_res[MAXVAL]);
  printf("Decompressed Data Range: %f\n", decomp_data_res[MAXVAL] - decomp_data_res[MINVAL]);
  printf("Decompressed Data Mean: %f\n", decomp_data_res[AVGVAL]);
  printf("Decompressed Data Stddev: %f\n", std_decomp_data);

  printf("Max Absolute Error: %f\n", max_abserr);
  printf("Max Absolute Error Index: %zu\n", max_abserr_index);

  printf("Correlation Coefficient: %f\n", ee);
  double mse = h_err[SUM_ERR_SQ] / (len1 * len2 * len3);
  printf("Mean Squared Error: %f\n", mse);
  printf("Normalized Root Mean Squared Error: %f\n", sqrt(mse) / 
    (orig_data_res[MAXVAL] - orig_data_res[MINVAL]));
  printf("Peak Signal-to-Noise Ratio: %f\n", 20 * 
    log10(orig_data_res[MAXVAL] - orig_data_res[MINVAL]) - 10 * log10(mse));

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //

  cudaFree(original_data_device);
  cudaFree(decompressed_data_device);
  cudaFreeHost(original_data_host);
  cudaFreeHost(decompressed_data_host);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~ Main Function ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  
int main(int argc, char **argv) {
  
  auto fname = std::string(argv[1]);
  size_t len1 = std::stoi(argv[2]);
  size_t len2 = std::stoi(argv[3]);
  size_t len3 = std::stoi(argv[4]);

  printf("fname: %s, len1: %zu, len2: %zu, len3: %zu\n", fname.c_str(), len1, len2, len3);

  compress(fname, len1, len2, len3);

  decompress(fname, len1, len2, len3);

  compare(fname, len1, len2, len3);

  return 0;

}