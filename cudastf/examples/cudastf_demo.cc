#include "cudastf_api.hh" // CUDASTF API

namespace utils = _portable::utils;

float* compressed_final;
float* decompressed_final;

// ~~~~~~~~~~~~~~~~~~~~~~~ Compress ~~~~~~~~~~~~~~~~~~~~~~~ //

void compress(std::string fname, size_t len1, size_t len2, size_t len3, cudaStream_t stream) {
  fz::Config<float> conf(len1, len2, len3);
  conf.eb = 2e-4;
  conf.fname = fname;
  conf.use_histogram_sparse = false;

  float* input_data_host;
  cudaMallocHost(&input_data_host, conf.orig_size);
  utils::fromfile(fname, input_data_host, conf.orig_size);

  context ctx;
  uint8_t* compressed_data_device;
  fz::STF_Compressor<float> compressor(conf, ctx);
  compressor.compress(input_data_host, &compressed_data_device, stream);

  // get out compressed data if needed

  conf.print();

  cudaFreeHost(input_data_host);
}

// ~~~~~~~~~~~~~~~~~~~~~~~ Decompressor ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //

void decompress_file_demo(std::string fname, cudaStream_t stream) {
  context ctx;
  std::string compressed_fname = fname + ".stf_compressed";
  fz::STF_Decompressor<float> decompressor(compressed_fname, ctx);
  
  uint8_t* compressed_data_host;
  size_t original_size = decompressor.conf->orig_size;
  cudaMalloc(&compressed_data_host, decompressor.conf->comp_size);

  decompressor.conf->use_histogram_sparse = false;
  decompressor.conf->print();

  float* out_data_device;
  cudaMalloc(&out_data_device, original_size);

  float* original_data_host;
  cudaMallocHost(&original_data_host, original_size);
  utils::fromfile(fname, original_data_host, original_size);
  decompressor.decompress(compressed_data_host, out_data_device, stream, original_data_host);

  cudaFree(compressed_data_host);
  cudaFree(out_data_device);
  cudaFreeHost(original_data_host);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~ Main Function ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  
int main(int argc, char **argv) {
  
  auto fname = std::string(argv[1]);
  size_t len1 = std::stoi(argv[2]);
  size_t len2 = std::stoi(argv[3]);
  size_t len3 = std::stoi(argv[4]);

  printf("fname: %s, len1: %zu, len2: %zu, len3: %zu\n", fname.c_str(), len1, len2, len3);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  compress(fname, len1, len2, len3, stream);

  decompress_file_demo(fname, stream);

  cudaStreamDestroy(stream);

  return 0;

}