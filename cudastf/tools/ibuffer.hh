#pragma once

#include <cuda_runtime.h>
#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

// ############################# Memory Manager

//!TODO: implement taking a radius and outlier_buffer_ratio

template <typename T>
struct stf_internal_buffers {
  stf_internal_buffers(context& ctx, size_t len, T* input_data, size_t revbk4_bytes, size_t bk_bytes, int pardeg, bool comp = true) {
    if (comp) {
      l_uncomp = ctx.logical_data(input_data, {len}).set_symbol("l_uncomp"); 
      l_compressed = ctx.logical_data(shape_of<slice<uint8_t>>(len * 4 / 2)).set_symbol("l_compressed");
      l_q_codes =   ctx.logical_data(shape_of<slice<uint16_t>>(len)).set_symbol("l_q_codes");
      l_top1 =      ctx.logical_data(shape_of<slice<uint32_t>>(1)).set_symbol("l_top1");
      l_hist =      ctx.logical_data(shape_of<slice<uint32_t>>(1024)).set_symbol("l_hist");
      l_out_vals =  ctx.logical_data(shape_of<slice<T>>(len / 5)).set_symbol("l_out_vals");
      l_out_idxs =  ctx.logical_data(shape_of<slice<uint32_t>>(len / 5)).set_symbol("l_out_idxs");
      l_out_num =   ctx.logical_data(shape_of<slice<uint32_t>>(1)).set_symbol("l_out_num");
      l_revbk4 =    ctx.logical_data(shape_of<slice<uint8_t>>(revbk4_bytes)).set_symbol("l_revbk4");
      l_bk4 =       ctx.logical_data(shape_of<slice<uint32_t>>(bk_bytes)).set_symbol("l_bk4");
      codeword_hist = ctx.logical_data(shape_of<slice<uint32_t>>(1024)).set_symbol("codeword_hist");
      l_scratch =  ctx.logical_data(shape_of<slice<uint32_t>>(len)).set_symbol("l_scratch");
      l_par_nbit = ctx.logical_data(shape_of<slice<uint32_t>>(pardeg)).set_symbol("l_par_nbit");
      l_par_ncell = ctx.logical_data(shape_of<slice<uint32_t>>(pardeg)).set_symbol("l_par_ncell");
      l_par_entry = ctx.logical_data(shape_of<slice<uint32_t>>(pardeg)).set_symbol("l_par_entry");
      l_bitstream = ctx.logical_data(shape_of<slice<uint32_t>>(len / 2)).set_symbol("l_bitstream");
      hf_header_entry = ctx.logical_data(shape_of<slice<uint32_t>>(6)).set_symbol("hf_header_entry");
      out_entries = ctx.logical_data(shape_of<slice<uint32_t>>(5)).set_symbol("out_entries");
    } else {
      l_uncompressed = ctx.logical_data(shape_of<slice<T>>(len)).set_symbol("l_uncompressed");
      l_q_codes =   ctx.logical_data(shape_of<slice<uint16_t>>(len)).set_symbol("l_q_codes");
    }
  }
  mutable logical_data<slice<T>> l_uncomp;       // logical uncompressed data
  mutable logical_data<slice<uint8_t>> l_compressed; // logical compressed data

  mutable logical_data<slice<uint16_t>> l_q_codes;   // quantization codes
  mutable logical_data<slice<uint32_t>> l_top1;     // top 1 histogram data

  mutable logical_data<slice<uint32_t>> l_hist;      // histogram data

  mutable logical_data<slice<T>> l_out_vals;     // outlier values
  mutable logical_data<slice<uint32_t>> l_out_idxs;  // outlier indices
  mutable logical_data<slice<uint32_t>> l_out_num;   // outlier number

  mutable logical_data<slice<uint32_t>> l_bk4;     // logical book data
  mutable logical_data<slice<uint8_t>> l_revbk4;  // logical reverse book data
  mutable logical_data<slice<uint32_t>> codeword_hist; // logical codeword histogram data

  mutable logical_data<slice<uint32_t>> l_scratch; // logical scratch data
  mutable logical_data<slice<uint32_t>> l_par_nbit; // logical partitioned nbit data
  mutable logical_data<slice<uint32_t>> l_par_ncell; // logical partitioned ncell data
  mutable logical_data<slice<uint32_t>> l_par_entry; // logical partitioned entry data
  mutable logical_data<slice<uint32_t>> l_bitstream; // logical bitstream data
  mutable logical_data<slice<uint32_t>> hf_header_entry; // logical header entry data
  uint32_t codec_comp_output_len{0}; // codec compressed output length

  mutable logical_data<slice<uint32_t>> out_entries; // logical outlier entries data

  // ~~~ Decompression Buffers ~~~ //

  mutable logical_data<slice<T>> l_uncompressed; // logical compressed data
};