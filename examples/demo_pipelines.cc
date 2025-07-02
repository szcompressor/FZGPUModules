/*
 * filename: demo_pipelines.cc
 * author: Skyler Ruiter
 * date: 06/23/2025
 *
 * 
 */

#include "fzmod.hh"
namespace utils = _portable::utils;  // for file I/O utilities

uint8_t* compressed_data_host;
float* decompressed_data_host;

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Compression Demo
void compress_demo(std::string fname, size_t x, size_t y, size_t z,
                   cudaStream_t stream) {
  
  fz::Config<float> conf(x, y, z);
  conf.eb = 2e-4;                                     // set error bound
  conf.eb_type = fz::EB_TYPE::ABS;                    // set error bound type
  conf.algo = fz::ALGO::LORENZO;                      // set algorithm type
  conf.codec = fz::CODEC::HUFFMAN;                    // set codec type
  conf.lossless_codec_2 = fz::SECONDARY_CODEC::NONE;  // no secondary codec
  conf.fromFile = false;  // not reading directly from file
  conf.fname = fname;     // filename for output

  float *input_data_device, *input_data_host;
  cudaMallocHost(&input_data_host, conf.orig_size);
  cudaMalloc(&input_data_device, conf.orig_size);
  utils::fromfile(fname, input_data_host, conf.orig_size);
  cudaMemcpy(input_data_device, input_data_host, conf.orig_size,
             cudaMemcpyHostToDevice);

  fz::Compressor<float> compressor(conf);

  uint8_t* comp_data_d;

  compressor.compress(input_data_device, &comp_data_d, stream);

  cudaMallocHost(&compressed_data_host, conf.comp_size);
  cudaMemcpy(compressed_data_host, comp_data_d, conf.comp_size,
             cudaMemcpyDeviceToHost);

  cudaFreeHost(input_data_host);
  cudaFree(input_data_device);
  cudaFree(comp_data_d);
  cudaStreamSynchronize(stream);
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Decompression Demo
void decompress_demo_file(std::string fname, cudaStream_t stream) {
  
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
