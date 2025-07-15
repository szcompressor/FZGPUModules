#pragma once
#include <memory>

#include "huffman/include/hf_impl.hh"
#include "huffman/src/hf_bk_impl1.seq.cc"

#include "huffman_kernels.cu"

using namespace cuda::experimental::stf;

// ~~~~~~~~~~~~~~~ HEADER DETAILS ~~~~~~~~~~~~~~~ //

#define HF_HEADER_FORCED_ALIGN 128
#define HF_HEADER_HEADER 0
#define HF_HEADER_REVBK 1
#define HF_HEADER_PAR_NBIT 2
#define HF_HEADER_PAR_ENTRY 3
#define HF_HEADER_BITSTREAM 4
#define HF_HEADER_END 5

#define COMPRESSION_PIPELINE_HEADER_SIZE 128

//! Header struct for Huffman codec
typedef struct {
  int bklen : 16;
  int sublen, pardeg;
  size_t original_len;
  size_t total_nbit, total_ncell;
  uint32_t entry[HF_HEADER_END + 1];
} hf_header;

// ~~~~~~~~~~~~~ KERNEL VARIABLES ~~~~~~~~~~~~~ //

//! helper variables for gpu kernels
struct HuffmanHelper {
  static const int BLOCK_DIM_ENCODE = 256;
  static const int BLOCK_DIM_DEFLATE = 256;

  static const int ENC_SEQUENTIALITY = 4;
  static const int DEFLATE_CONSTANT = 4;
};

//! Coarse tune the sublen for Huffman encoding
size_t capi_phf_coarse_tune_sublen(size_t len)
{
  auto div = [](auto _l, auto _subl) { return (_l - 1) / _subl + 1; };

  int current_dev = 0;
  cudaSetDevice(current_dev);
  cudaDeviceProp dev_prop{};
  cudaGetDeviceProperties(&dev_prop, current_dev);

  auto nSM = dev_prop.multiProcessorCount;
  auto allowed_block_dim = dev_prop.maxThreadsPerBlock;
  auto deflate_nthread = allowed_block_dim * nSM / 
    HuffmanHelper::DEFLATE_CONSTANT;

  auto optimal_sublen = div(len, deflate_nthread);
  // round up
  optimal_sublen =
      div(optimal_sublen, HuffmanHelper::BLOCK_DIM_DEFLATE) * 
        HuffmanHelper::BLOCK_DIM_DEFLATE;

  return optimal_sublen;
};

//! Coarse tune the sublen and pardeg for Huffman encoding
void capi_phf_coarse_tune(size_t len, int* sublen, int* pardeg) {
  auto div = [](auto l, auto subl) { 
    return (l - 1) / subl + 1; 
  };
  *sublen = capi_phf_coarse_tune_sublen(len);
  *pardeg = div(len, *sublen);
}

// ~~~~~~~~~~~~~~~ STF HUFFMAN CLASS ~~~~~~~~~~~~~~~ //

//! Huffman Codec (STF Version)
template <typename T>
class HuffmanCodecSTF {
  public:
  int num_SMs;
  size_t pardeg, len, sublen;
  static constexpr uint16_t max_bklen  = 1024;
  size_t revbk4_bytes;
  uint16_t rt_bklen = 1024;
  size_t total_nbit, total_ncell = 0;
  hf_header header;

  //! Constructor
  HuffmanCodecSTF(size_t const inlen, int const _pardeg, context& ctx) {
    cudaDeviceGetAttribute(&num_SMs, cudaDevAttrMultiProcessorCount, 0);
    pardeg = _pardeg;
    len = inlen;
    sublen = (len - 1) / pardeg + 1;
    revbk4_bytes = 4 * (2 * 4 * 8) + sizeof(uint16_t) * max_bklen;
  }

  // ~~~~~~~~~~~~~~~~~~~~~~~~~ Huffman Buildbook ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  void buildbook(uint16_t const _rt_bklen, 
                 stf_internal_buffers<T>& ibuffer, 
                 context& ctx, 
                 cudaStream_t stream) {

    //! Build the Huffman codebook on CPU
    ctx.host_launch(
      ibuffer.l_hist.read(), 
      ibuffer.l_revbk4.rw(), 
      ibuffer.l_bk4.rw())
      .set_symbol("hf_bb")->*[&]
      (auto l_hist, 
       auto rev_book, 
       auto l_bk4) 
    {
      using PW4 = HuffmanWord<4>;
      using PW8 = HuffmanWord<8>;

      auto bklen = rt_bklen;

      auto bk_bytes = sizeof(uint32_t) * bklen;
      constexpr auto TYPE_BITS = sizeof(uint32_t) * 8;
      auto space_bytes = hf_space<uint16_t, uint32_t>::space_bytes(bklen);
      auto revbook_ofst = hf_space<uint16_t, uint32_t>::revbook_offset(bklen);
      auto space = new hf_canon_reference<uint16_t, uint32_t>(bklen);

      auto bk8 = new uint64_t[bklen];
      memset(bk8, 0xff, sizeof(uint64_t) * bklen);
      auto bk4 = new uint32_t[bklen];
      memset(bk4, 0xff, bk_bytes);

      { // phf_CPU_build_codebook_v1<uint64_t>(freq, bklen, bk8);
        auto state_num = 2 * bklen;
        auto tree = create_tree_serial(state_num);
        {
          for (size_t i = 0; i < bklen; i++) {
            uint32_t freq = l_hist(i);
            if (freq) qinsert(tree, new_node(tree, freq, i, 0, 0));
          }
          while (tree->qend > 2) qinsert(tree, 
            new_node(tree, 0, 0, qremove(tree), qremove(tree)));
          phf_stack<node_t, sizeof(uint64_t)>::template 
            inorder_traverse<uint64_t>(tree->qq[1], bk8);
        }
        destroy_tree(tree);
      }

      for (auto i = 0; i < bklen; i++) {
        auto pw8 = reinterpret_cast<PW8*>(bk8 + i);
        auto pw4 = reinterpret_cast<PW4*>(bk4 + i);
        if (*(bk8 + i) == ~((uint64_t)0x0)) {
        } else {
          if (pw8->bitcount > pw4->FIELD_CODE) {
            pw4->bitcount = pw4->OUTLIER_CUTOFF;
            pw4->prefix_code = 0;
            std::cout << i << "\tlarger than FIELD_CODE" << std::endl;
          } else {
            pw4->bitcount = pw8->bitcount;
            pw4->prefix_code = pw8->prefix_code;
          }
        }
      }

      space->input_bk() = bk4;
      {
        space->canonize();
      }

      memcpy(bk4, space->output_bk(), bk_bytes);

      auto offset = 0;

      memcpy(rev_book.data_handle(), 
        space->first(), sizeof(int) * TYPE_BITS);
      offset += sizeof(int) * TYPE_BITS;

      memcpy(rev_book.data_handle() + offset, 
        space->entry(), sizeof(int) * TYPE_BITS);
      offset += sizeof(int) * TYPE_BITS;

      memcpy(rev_book.data_handle() + offset, 
        space->keys(), sizeof(uint16_t) * bklen);


      memcpy(l_bk4.data_handle(), bk4, bk_bytes);

      delete space;
      delete[] bk8;
      delete[] bk4;
    };
  }

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~ Encode ~~~~~~~~~~~~~~~~~~~~~~~~ //
  void encode(int pardeg, 
              stf_internal_buffers<T>& ibuffer, 
              context& ctx, 
              cudaStream_t stream) {

    auto div = [](auto whole, auto part) -> uint32_t {
      if (whole == 0) throw std::runtime_error("Dividend is zero.");
      if (part == 0) throw std::runtime_error("Divisor is zero.");
      return (whole - 1) / part + 1;
    };

    auto data_len = len;
    auto numSMs = num_SMs;
    auto blen = rt_bklen;
    auto t_sublen = sublen;

    //! GPU coarse encode phase 1
    ctx.task(
      ibuffer.l_q_codes.read(), 
      ibuffer.l_bk4.read(), 
      ibuffer.l_scratch.write())
      .set_symbol("hf_en_phase1")->*[div, data_len, blen, numSMs]
      (cudaStream_t s, 
       auto q_codes, 
       auto book, 
       auto scratch) 
    {
      auto block_dim = 256;
      auto grid_dim = div(data_len, block_dim);
      hf_encode_phase1_fill<<<8 * numSMs, 256, sizeof(uint32_t) * blen, s>>>(
        q_codes, data_len, book, blen, scratch);
    };

    //! GPU_coarse_encode_phase2
    ctx.task(
      ibuffer.l_scratch.rw(), 
      ibuffer.l_par_nbit.write(), 
      ibuffer.l_par_ncell.write())
      .set_symbol("hf_en_phase2")
      ->*[div, pardeg, data_len, t_sublen]
      (cudaStream_t s, 
       auto scratch, 
       auto par_nbit, 
       auto par_ncell) 
    {
      auto block_dim = 256;
      auto grid_dim = div(pardeg, block_dim);
      hf_encode_phase2_deflate<<<grid_dim, block_dim, 0, s>>>(
        scratch, data_len, par_nbit, par_ncell, t_sublen, pardeg);
    };

    //! Populate parentry with ncell
    ctx.task(
      ibuffer.l_par_entry.rw(), 
      ibuffer.l_par_ncell.read())
      .set_symbol("hf_en_par_entry_fill")->*[pardeg]
      (cudaStream_t s, 
       auto par_entry, 
       auto par_ncell) 
    {
      cuda_safe_call(
        cudaMemcpyAsync(par_entry.data_handle() + 1, par_ncell.data_handle(),
        (pardeg - 1) * sizeof(uint32_t), cudaMemcpyDeviceToDevice, s));
    };

    cuda_safe_call(cudaStreamSynchronize(ctx.fence()));

    //! GPU_coarse_encode_phase3_sync
    ctx.host_launch(
      ibuffer.l_par_nbit.read(), 
      ibuffer.l_par_ncell.read(), 
      ibuffer.l_par_entry.rw())
      .set_symbol("hf_en_phase3")->*[pardeg, this, stream]
      (auto par_nbit, 
       auto par_ncell, 
       auto par_entry) 
    {
      for (auto i = 1; i < pardeg; i++) {
        par_entry(i) += par_entry(i - 1);
      }
      total_nbit = std::accumulate(
        par_nbit.data_handle(), par_nbit.data_handle() + pardeg, size_t(0));
      total_ncell = std::accumulate(
        par_ncell.data_handle(), par_ncell.data_handle() + pardeg, size_t(0));
    };

    //! GPU_coarse_encode_phase4
    ctx.task(
      ibuffer.l_scratch.read(), 
      ibuffer.l_par_entry.read(), 
      ibuffer.l_par_ncell.read(), 
      ibuffer.l_bitstream.write())
      .set_symbol("hf_en_phase4")->*[pardeg, t_sublen]
      (cudaStream_t s, 
       auto scratch, 
       auto par_entry, 
       auto par_ncell, 
       auto bitstream) {
      hf_encode_phase4_concatenate<<<pardeg, 128, 0, s>>>(
        scratch, par_entry, par_ncell, t_sublen, bitstream);
    };

    //! Setup Header sections
    ctx.host_launch(
      ibuffer.hf_header_entry.write())
      .set_symbol("hf_en_bitstream_setup")->*[&](
      auto hf_h_entry)
    {
      // Reset header values
      header.bklen = rt_bklen;
      header.sublen = sublen;
      header.pardeg = pardeg;
      header.original_len = len;
      header.total_nbit = total_nbit;
      header.total_ncell = total_ncell;

      // Calculate section sizes
      uint32_t sizes[HF_HEADER_END + 1];
      sizes[HF_HEADER_HEADER] = HF_HEADER_FORCED_ALIGN;
      sizes[HF_HEADER_REVBK] = revbk4_bytes;
      sizes[HF_HEADER_PAR_NBIT] = pardeg * sizeof(uint32_t);
      sizes[HF_HEADER_PAR_ENTRY] = pardeg * sizeof(uint32_t);
      sizes[HF_HEADER_BITSTREAM] = 4 * total_ncell;

      // Calculate offsets properly (only once)
      header.entry[0] = 0;
      hf_h_entry(0) = 0;
      for (auto i = 1; i < HF_HEADER_END + 1; i++) {
        header.entry[i] = header.entry[i-1] + sizes[i-1];
        hf_h_entry(i) = header.entry[i];
      }

      ibuffer.codec_comp_output_len = header.entry[HF_HEADER_END];
    };

    cuda_safe_call(cudaStreamSynchronize(ctx.fence()));

    //! Copy the header to the compressed buffer
    ctx.task(
      ibuffer.l_compressed.rw())
      .set_symbol("hf_en_comp_hfheader")->*[this]
      (cudaStream_t s, 
       auto compressed) 
    {
      cuda_safe_call(cudaMemcpyAsync(
        compressed.data_handle() + COMPRESSION_PIPELINE_HEADER_SIZE, 
        &header, sizeof(hf_header), cudaMemcpyHostToDevice, s));
    };

    //! copy the reversebook to comp buffer
    ctx.task(
      ibuffer.l_compressed.rw(), 
      ibuffer.l_revbk4.read())
      .set_symbol("hf_en_comp_revbkheader")->*[this]
      (cudaStream_t s, 
       auto compressed, 
       auto revbook) 
    {
      cuda_safe_call(cudaMemcpyAsync(
        compressed.data_handle() + COMPRESSION_PIPELINE_HEADER_SIZE + 
        header.entry[HF_HEADER_REVBK], revbook.data_handle(),
        revbk4_bytes, cudaMemcpyDeviceToDevice, s));
    };

    //! copy par_nbit to comp buffer
    ctx.task(
      ibuffer.l_compressed.rw(), 
      ibuffer.l_par_nbit.read())
      .set_symbol("hf_en_comp_nbit")->*[this, pardeg]
      (cudaStream_t s, 
       auto compressed, 
       auto par_nbit) 
    {
      cuda_safe_call(cudaMemcpyAsync(
        compressed.data_handle() + COMPRESSION_PIPELINE_HEADER_SIZE + 
        header.entry[HF_HEADER_PAR_NBIT], par_nbit.data_handle(),
        pardeg * sizeof(uint32_t), cudaMemcpyDeviceToDevice, s));
    };

    //! copy par_entry to comp buffer
    ctx.task(
      ibuffer.l_compressed.rw(), 
      ibuffer.l_par_entry.read())
      .set_symbol("hf_en_comp_entry")->*[this, pardeg]
      (cudaStream_t s, 
       auto compressed, 
       auto par_entry) 
    {
      cuda_safe_call(cudaMemcpyAsync(
        compressed.data_handle() + COMPRESSION_PIPELINE_HEADER_SIZE + 
        header.entry[HF_HEADER_PAR_ENTRY], par_entry.data_handle(),
        pardeg * sizeof(uint32_t), cudaMemcpyDeviceToDevice, s));
    };

    //! copy bitstream to comp buffer
    ctx.task(
      ibuffer.l_compressed.rw(), 
      ibuffer.l_bitstream.read())
      .set_symbol("hf_en_comp_bitstream")->*[this, data_len]
      (cudaStream_t s, 
       auto compressed, 
       auto bitstream) 
    {
      cuda_safe_call(cudaMemcpyAsync(
        compressed.data_handle() + COMPRESSION_PIPELINE_HEADER_SIZE + 
        header.entry[HF_HEADER_BITSTREAM], bitstream.data_handle(),
        4 * total_ncell, cudaMemcpyDeviceToDevice, s));
    };

  } // encode

}; // HuffmanCodecSTF