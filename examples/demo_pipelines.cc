/*
 * filename: demo_pipelines.cc
 * author: Skyler Ruiter
 * date: 06/23/2025
 *
 * A demo file that helps show how to compare pipelines
 * using the module framework of fzmod.
 */

#include "fzmod.hh"
namespace utils = _portable::utils;  // for file I/O utilities

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

float* input_data_device; // device pointer for input data (easy reuse)

// Function to run a compression and decompression pipeline and get the output
void run_pipeline(fz::Config<float>& conf, uint8_t* compressed_data_device, cudaStream_t stream) {

  // create a compressor object with the given configuration
  fz::Compressor<float> compressor(conf);

  // call compress to put the compressed data in the device pointer
  compressor.compress(input_data_device, &compressed_data_device, stream);

  // make a decompressor object with the same configuration
  fz::Decompressor<float> decompressor(conf);

  // allocate memory for the decompressed data on device
  float* decompressed;
  cudaMalloc(&decompressed, conf.orig_size);

  // call decompress on the data and give the original data for quality analysis
  decompressor.decompress(compressed_data_device, decompressed, stream, input_data_device);

  // free the decompressed data pointer
  cudaFree(decompressed);
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
  size_t x = std::stoi(argv[2]);
  size_t y = std::stoi(argv[3]);
  size_t z = std::stoi(argv[4]);

  // set the same eb and eb type for all pipelines for fair comparison
  double eb = 2e-4;
  fz::EB_TYPE eb_type = fz::EB_TYPE::REL;
  
  // make a stream to send to the compressor
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  fz::Config<float> default_conf(x, y, z);
  fz::Config<float> quality_conf(x, y, z);
  fz::Config<float> speed_conf(x, y, z);

  // ## Set the configurations for the pipelines ## //

  // default configuration
  default_conf.eb = eb;
  default_conf.eb_type = eb_type;
  default_conf.toFile = false;
  default_conf.verbose = false;
  
  // data-quality focused configuration
  quality_conf.eb = eb;
  quality_conf.eb_type = eb_type;
  quality_conf.toFile = false;
  quality_conf.algo = fz::ALGO::SPLINE;
  quality_conf.codec = fz::CODEC::HUFFMAN;
  quality_conf.lossless_codec_2 = fz::SECONDARY_CODEC::NONE;
  quality_conf.verbose = false;

  // speed optimized configuration
  speed_conf.eb = eb;
  speed_conf.eb_type = eb_type;
  speed_conf.toFile = false;
  speed_conf.algo = fz::ALGO::LORENZO;
  speed_conf.codec = fz::CODEC::FZG;
  speed_conf.lossless_codec_2 = fz::SECONDARY_CODEC::NONE;
  speed_conf.verbose = false;

  // setup the input data from given file
  float* input_data_host;
  cudaMallocHost(&input_data_host, default_conf.orig_size);
  utils::fromfile(fname, input_data_host, default_conf.orig_size);
  cudaMalloc(&input_data_device, default_conf.orig_size);
  cudaMemcpyAsync(input_data_device, input_data_host, default_conf.orig_size,
                  cudaMemcpyHostToDevice, stream);

  // create pointers for the compressed data to be placed
  uint8_t* comp_d_default = nullptr;
  uint8_t* comp_d_quality = nullptr;
  uint8_t* comp_d_speed = nullptr;

  // run the pipelines using each configuration and print results
  printf("Running default pipeline...\n");
  run_pipeline(default_conf, comp_d_default, stream);
  cudaStreamSynchronize(stream);
  printf("Running quality pipeline...\n");
  run_pipeline(quality_conf, comp_d_quality, stream);
  cudaStreamSynchronize(stream);
  printf("Running speed pipeline...\n");
  run_pipeline(speed_conf, comp_d_speed, stream);
  
  // cleanup
  cudaFree(input_data_device);
  cudaFreeHost(input_data_host);
  cudaStreamDestroy(stream);
  return 0;
}
