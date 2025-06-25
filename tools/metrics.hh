#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <iostream>
#include <cmath>

#include "header.hh"
#include "compare.hh"

namespace fz {

struct fzmod_metrics {
  // timing data
  double end_to_end_comp_time = 0;
  double preprocessing_time = 0;
  double prediction_time = 0;
  double hist_time = 0;
  double encoder_time = 0;
  double file_io_time = 0;

  double end_to_end_decomp_time = 0;
  double scatter_time = 0;
  double decoding_time = 0;
  double prediction_reversing_time = 0;
  double decomp_file_io_time = 0;

  // data metrics
  double min = 0;
  double max = 0;
  double range = 0;
  double mean = 0;
  double stddev = 0;

  double decomp_min = 0;
  double decomp_max = 0;
  double decomp_range = 0;
  double decomp_mean = 0;
  double decomp_stddev = 0;

  double max_err = 0;
  size_t max_err_idx = 0;
  double max_abserr = 0;

  PRECISION precision = PRECISION::FLOAT;

  // compression metrics
  double compression_ratio = 0;
  double final_eb = 0;

  uint64_t num_outliers = 0;

  uint64_t data_len = 0;
  uint64_t orig_bytes = 0;
  uint64_t comp_bytes = 0;
  size_t total_footprint_d = 0;
  size_t total_footprint_h = 0;

  double bitrate = 0;
  double nrmse = 0;
  double coeff = 0;
  double psnr = 0;

  void print(bool comp = true, bool compare = false) {
    auto throughput = [](double n_bytes, double time_ms) {
      return n_bytes / (1.0 * 1024 * 1024 * 1024) / (time_ms * 1e-3);
    };

    printf("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
    printf("FZMod GPU Compression Library Metrics\n");
    printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n");

    if (comp) {
      printf("~~COMPRESSION~~\n");

      printf("Timing Metrics\n");
      printf("Preprocessing stage:\t %f ms \t %f Gib/s\n", preprocessing_time,
             throughput(orig_bytes, preprocessing_time));
      printf("Prediction stage:\t %f ms \t %f Gib/s\n",
             prediction_time,
             throughput(orig_bytes, prediction_time));
      printf("Histogram stage:\t %f ms \t %f Gib/s\n", hist_time,
             throughput(orig_bytes, hist_time));
      printf("Encoding stage:\t\t %f ms \t %f Gib/s\n", encoder_time,
             throughput(orig_bytes, encoder_time));
      printf("End-end compression:\t %f ms \t %f Gib/s\n",
             end_to_end_comp_time,
             throughput(orig_bytes, end_to_end_comp_time));
      printf("File IO stage:\t\t %f ms \t %f Gib/s\n", file_io_time,
             throughput(orig_bytes, file_io_time));

      printf("\n");

      printf("Compression Metrics\n");
      printf("Original size:\t\t %zu bytes\n", orig_bytes);
      printf("Compressed size:\t %zu bytes\n", comp_bytes);
      printf("Compression ratio:\t %f\n",
             (float)orig_bytes / (float)comp_bytes);
      printf("Num outliers:\t\t %zu\n", num_outliers);
      printf("Data Length:\t\t %zu\n", data_len);
      printf("Data Type:\t\t %s\n",
             precision == PRECISION::FLOAT ? "fp32" : "fp64");
      size_t total_footprint_d_mib =
          total_footprint_d / (1024 * 1024);
      size_t total_footprint_h_mib =
          total_footprint_h / (1024 * 1024);
      printf("Original File Size:\t\t %zu MiB\n",
             orig_bytes / (1024 * 1024));
      printf("Total Footprint (Device):\t %zu MiB\n", total_footprint_d_mib);
      printf("Total Footprint (Host):\t\t %zu MiB\n", total_footprint_h_mib);
    } else {
      printf("~~DECOMPRESSION~~\n");

      printf("Timing Metrics\n");
      printf("Decoding Stage:\t\t %f ms %f GiB/s\n", decoding_time,
             throughput(orig_bytes, decoding_time));
      printf("Pred-Reversing Stage:\t %f ms %f GiB/s\n",
             prediction_reversing_time,
             throughput(orig_bytes, prediction_reversing_time));
      printf("End-end decompression:\t %f ms %f GiB/s\n",
             end_to_end_decomp_time,
             throughput(orig_bytes, end_to_end_decomp_time));
      printf("File IO time:\t\t %f ms %f GiB/s\n", decomp_file_io_time,
             throughput(orig_bytes, decomp_file_io_time));

      printf("\n");
    }

    if (compare && !comp) {
      printf("~~COMPARISON~~\n");

      printf("Data Original Min:\t\t %f\n", min);
      printf("Data Original Max:\t\t %f\n", max);
      printf("Data Original Range:\t\t %f\n", range);
      printf("Data Original Mean:\t\t %f\n", mean);
      printf("Data Original Stddev:\t\t %f\n", stddev);
      printf("\n");
      printf("Data Decompressed Min:\t\t %f\n", decomp_min);
      printf("Data Decompressed Max:\t\t %f\n", decomp_max);
      printf("Data Decompressed Range:\t %f\n", decomp_range);
      printf("Data Decompressed Mean:\t\t %f\n", decomp_mean);
      printf("Data Decompressed Stddev:\t %f\n", decomp_stddev);
      printf("\n");
      printf("Compression Ratio:\t\t %f\n", compression_ratio);
      printf("Bitrate:\t\t\t %f\n", bitrate);
      printf("NRMSE:\t\t\t\t %f\n", nrmse);
      printf("PSNR:\t\t\t\t %f\n", psnr);
      printf("coeff:\t\t\t\t %f\n", coeff);
      printf("\n");
      printf("Max Error Index:\t\t %zu\n", max_err_idx);
      printf("Max Error Value:\t\t %f\n", max_err);
      printf("Max Abs Error:\t\t\t %f\n", max_abserr);
      printf("\n");
    }
    printf("\n");
  } // print

  template <typename T>
  void compare(T* orig_data, T* decomp_data, size_t len, cudaStream_t stream) {
    constexpr auto MINVAL = 0;
    constexpr auto MAXVAL = 1;
    constexpr auto AVGVAL = 2;

    constexpr auto SUM_CORR = 0;
    constexpr auto SUM_ERR_SQ = 1;
    constexpr auto SUM_VAR_ODATA = 2;
    constexpr auto SUM_VAR_XDATA = 3;

    T orig_data_res[4], decomp_data_res[4];

    fz::module::GPU_extrema(orig_data, len, orig_data_res);
    fz::module::GPU_extrema(decomp_data, len, decomp_data_res);

    T h_err[4];

    fz::module::GPU_calculate_errors<T>(
      orig_data, orig_data_res[AVGVAL], decomp_data, decomp_data_res[AVGVAL], len, h_err);

    double std_orig_data = sqrt(h_err[SUM_VAR_ODATA] / len);
    double std_decomp_data = sqrt(h_err[SUM_VAR_XDATA] / len);
    double ee = h_err[SUM_CORR] / len;

    T max_abserr{0};
    size_t max_abserr_index{0};
    fz::module::GPU_find_max_error<T>(
      decomp_data, orig_data, len, max_abserr, max_abserr_index, stream);

    min = orig_data_res[MINVAL];
    max = orig_data_res[MAXVAL];
    range = orig_data_res[MAXVAL] - orig_data_res[MINVAL];
    mean = orig_data_res[AVGVAL];
    stddev = std_orig_data;

    decomp_min = decomp_data_res[MINVAL];
    decomp_max = decomp_data_res[MAXVAL];
    decomp_range = decomp_data_res[MAXVAL] - decomp_data_res[MINVAL];
    decomp_mean = decomp_data_res[AVGVAL];
    decomp_stddev = std_decomp_data;

    max_err_idx = max_abserr_index;
    max_err = max_abserr;
    max_abserr = max_abserr / range;

    coeff = ee / std_orig_data / std_decomp_data;
    double mse = h_err[SUM_ERR_SQ] / len;
    nrmse = sqrt(mse) / range;
    psnr = 20 * log10(range) - 10 * log10(mse);

    double bytes = 1.0 * sizeof(T) * len;
    bitrate = 32.0 / (bytes / comp_bytes);
    compression_ratio = (float)orig_bytes / (float)comp_bytes;
  }

}; // fzmod_metrics

} // namespace fz