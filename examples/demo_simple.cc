/*
 * filename: demo_simple.cc
 * author: Skyler Ruiter
 * date: 06/23/2025
 *
 * simplified demonstration of fzmod compression and decompression for
 * getting up and running quickly.
 */

#include "fzmod.hh"
namespace utils = _portable::utils;  // for file I/O utilities

uint8_t* compressed_data_host;

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Compression Demo
void compress_demo(std::string fname, size_t x, size_t y, size_t z,
                   cudaStream_t stream) {
  fz::Config<float> conf(x, y, z);
  conf.fromFile = true;
  conf.fname = fname;

  fz::Compressor<float> compressor(conf);

  uint8_t* comp_data_d;
  compressor.compress(nullptr, &comp_data_d, stream);

  cudaMallocHost(&compressed_data_host, conf.comp_size);
  cudaMemcpy(compressed_data_host, comp_data_d, conf.comp_size,
             cudaMemcpyDeviceToHost);

  cudaStreamSynchronize(stream);
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Decompression Demo
void decompress_demo_file(std::string fname, cudaStream_t stream) {
  std::string compressed_fname = fname + ".fzmod";
  fz::Decompressor<float> decompressor(compressed_fname);

  float* decompressed;
  size_t original_size = decompressor.conf->orig_size;
  cudaMalloc(&decompressed, original_size);

  float* original_data_host;
  cudaMallocHost(&original_data_host, original_size);
  utils::fromfile(fname, original_data_host, original_size);

  decompressor.decompress(compressed_data_host, decompressed, stream,
                          original_data_host);

  cudaFree(decompressed);
  cudaFreeHost(original_data_host);
  cudaStreamSynchronize(stream);
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

int main(int argc, char** argv) {
  // USAGE: ./fzmod_demo <filename> <len1> <len2> <len3>

  if (argc != 5) {
    std::cerr << "Usage: " << argv[0] << " <filename> <len1> <len2> <len3>"
              << std::endl;
    return 1;
  }
  auto fname = std::string(argv[1]);
  size_t len1 = std::stoi(argv[2]);
  size_t len2 = std::stoi(argv[3]);
  size_t len3 = std::stoi(argv[4]);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  compress_demo(fname, len1, len2, len3, stream);

  decompress_demo_file(fname, stream);

  cudaStreamDestroy(stream);
  cudaFreeHost(compressed_data_host);
  return 0;
}
