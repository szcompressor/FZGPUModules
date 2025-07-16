#pragma once

#include "config.hh"
#include "hf_hl.hh"
#include "mem/cxx_smart_ptr.h"

#define ALIGN_4ki(len) (((len) + 4095) & ~4095)

namespace fz {

typedef struct CompressorBufferToggle {
  bool quant_codes = true;
  bool host_outliers = false;
  bool device_outliers = true;
  bool anchor_points = false;
  bool histogram = true;
  bool top1 = true;
  bool compressed = true;
  bool _internal = false;
} CompressorBufferToggle;

template <typename T>
class InternalBuffers {
  public:

  using CodecFZG = fz::FzgCodec;
  using hf_mem_t = phf::Buf<uint16_t>;
  hf_mem_t* buf_hf = nullptr;

  bool is_comp;

  size_t total_footprint_d = 0;
  size_t total_footprint_h = 0;

  Config<T>* conf;

  uint8_t* codec_comp_output{nullptr};
  size_t codec_comp_output_len{0};

  T* internal_original{nullptr};

  GPU_unique_dptr<uint16_t[]> d_codes = nullptr;
  GPU_unique_dptr<T[]> d_anchors = nullptr;

  GPU_unique_dptr<uint32_t[]> d_hist = nullptr;
  GPU_unique_hptr<uint32_t[]> h_hist = nullptr;
  GPU_unique_dptr<uint32_t[]> d_top1 = nullptr;
  GPU_unique_hptr<uint32_t[]> h_top1 = nullptr;

  GPU_unique_dptr<uint8_t[]> d_compressed = nullptr;
  GPU_unique_hptr<uint8_t[]> h_compressed = nullptr;

  GPU_unique_dptr<uint8_t[]> d_internal_temp = nullptr;

  // OUTLIERS
  size_t outlier_reserve_size = 0;
  GPU_unique_dptr<T[]> d_val = nullptr;
  GPU_unique_hptr<T[]> h_val = nullptr;
  GPU_unique_dptr<uint32_t[]> d_idx = nullptr;
  GPU_unique_hptr<uint32_t[]> h_idx = nullptr;
  GPU_unique_dptr<uint32_t[]> d_num = nullptr;
  GPU_unique_hptr<uint32_t[]> h_num = nullptr;

  CodecFZG* codec_fzg = nullptr;

  InternalBuffers(Config<T>* config, CompressorBufferToggle* toggle, bool is_comp = true) 
    : conf(config), is_comp(is_comp)
  {
    if (conf->codec == CODEC::HUFFMAN) {
      buf_hf = new phf::Buf<uint16_t>(conf->len, conf->radius * 2);
      total_footprint_d += buf_hf->total_footprint_d;
      total_footprint_h += buf_hf->total_footprint_h;
    } else if (conf->codec == CODEC::FZG) {
      codec_fzg = new CodecFZG(conf->len);
      total_footprint_d += codec_fzg->total_footprint_d();
      total_footprint_h += codec_fzg->total_footprint_h();
    } else {
      throw std::runtime_error("Unsupported codec type");
    }
    if (is_comp) {
      outlier_reserve_size = conf->outlier_buffer_ratio * conf->len;
      if (toggle->device_outliers) {
        d_val = MAKE_UNIQUE_DEVICE(T, outlier_reserve_size);
        d_idx = MAKE_UNIQUE_DEVICE(uint32_t, outlier_reserve_size);
        d_num = MAKE_UNIQUE_DEVICE(uint32_t, 1);
        h_num = MAKE_UNIQUE_HOST(uint32_t, 1); // still need this
        total_footprint_d += outlier_reserve_size * (sizeof(T) + sizeof(uint32_t)) + sizeof(uint32_t);
        total_footprint_h += sizeof(uint32_t);
      }
      if (toggle->host_outliers) {
        h_val = MAKE_UNIQUE_HOST(T, outlier_reserve_size);
        h_idx = MAKE_UNIQUE_HOST(uint32_t, outlier_reserve_size);
        total_footprint_h += outlier_reserve_size * (sizeof(T) + sizeof(uint32_t));
      }
      if (toggle->quant_codes) {
        d_codes = MAKE_UNIQUE_DEVICE(uint16_t, ALIGN_4ki(conf->len));
        total_footprint_d += ALIGN_4ki(conf->len) * sizeof(uint16_t);
      }
      if (toggle->anchor_points) {
        d_anchors = MAKE_UNIQUE_DEVICE(T, conf->anchor512_len);
        total_footprint_d += conf->anchor512_len * sizeof(T);
      }
      if (toggle->histogram) {
        d_hist = MAKE_UNIQUE_DEVICE(uint32_t, conf->radius * 2);
        h_hist = MAKE_UNIQUE_HOST(uint32_t, conf->radius * 2);
        total_footprint_d += conf->radius * 2 * sizeof(uint32_t);
        total_footprint_h += conf->radius * 2 * sizeof(uint32_t);
      }
      if (toggle->top1) {
        d_top1 = MAKE_UNIQUE_DEVICE(uint32_t, 1);
        h_top1 = MAKE_UNIQUE_HOST(uint32_t, 1);
        total_footprint_d += sizeof(uint32_t);
        total_footprint_h += sizeof(uint32_t);
      }
      if (toggle->compressed) {
        d_compressed = MAKE_UNIQUE_DEVICE(uint8_t, conf->len * 4 / 2);
        h_compressed = MAKE_UNIQUE_HOST(uint8_t, conf->len * 4 / 2);
        total_footprint_d += conf->len * 4 / 2 * sizeof(uint8_t);
        total_footprint_h += conf->len * 4 / 2 * sizeof(uint8_t);
      }
      if (toggle->_internal) {
        // cudaMalloc(&internal_compressed, conf->len * 4 / 2);
        CHECK_GPU(cudaMalloc(&internal_original, conf->len * sizeof(T)));
        auto file = MAKE_UNIQUE_HOST(T, conf->len);
        utils::fromfile(conf->fname.c_str(), file.get(), conf->len * sizeof(T));
        CHECK_GPU(cudaMemcpy(internal_original, file.get(), conf->len * sizeof(T), cudaMemcpyHostToDevice));
        total_footprint_d += conf->len * sizeof(T);
      }
    } else {
      d_codes = MAKE_UNIQUE_DEVICE(uint16_t, ALIGN_4ki(conf->len));
    }
  }

  ~InternalBuffers() {
    if (codec_fzg) delete codec_fzg;
    if (buf_hf) delete buf_hf;
    if (internal_original) cudaFree(internal_original);
  }

  uint16_t* codes() const { return d_codes.get(); }
  uint32_t* histogram() const { return d_hist.get(); }
  uint32_t* top1() const { return d_top1.get(); }
  uint32_t* top1_host() const {
    cudaMemcpy(h_top1.get(), d_top1.get(), sizeof(uint32_t), cudaMemcpyDeviceToHost);
    return h_top1.get();
  }
  uint8_t* compressed() { return d_compressed.get(); }
  uint8_t* compressed_host() { return h_compressed.get(); }
  T* anchor_points() const { return d_anchors.get(); }
  T* outlier_values() const { return d_val.get(); }
  uint32_t* outlier_indices() const { return d_idx.get(); }
  uint32_t* num_outliers() const {
    cudaMemcpy(h_num.get(), d_num.get(), sizeof(uint32_t), cudaMemcpyDeviceToHost);
    return h_num.get();
  } 
  uint32_t* num_outliers_d() const { return d_num.get(); }

};

}