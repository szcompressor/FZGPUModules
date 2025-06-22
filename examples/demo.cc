#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>

#include "fzmod_compressor.hh"
#include "fzmod_decompressor.hh"
namespace utils = _portable::utils;

uint8_t* compressed_data_host;
float* decompressed_data_host;
// fz::Config<float> conf;

void compress_demo(std::string fname, size_t x, 
    size_t y, size_t z, cudaStream_t stream) {
    
    // Setup config with compression options
    fz::Config<float> conf(x, y, z);
    conf.eb = 2e-4;
    conf.eb_type = fz::EB_TYPE::ABS;
    conf.algo = fz::ALGO::LORENZO;
    conf.precision = fz::PRECISION::FLOAT;
    conf.codec = fz::CODEC::HUFFMAN;
    conf.fname = fname;

    // conf->use_huffman_reVISIT = true;

    // create memory for the data
    float* input_data_device, * input_data_host;
    uint8_t* internal_compressed;

    // allocate memory for the data
    cudaMallocHost(&input_data_host, conf.orig_size);
    cudaMalloc(&input_data_device, conf.orig_size);

    // read data from file
    utils::fromfile(fname, input_data_host, conf.orig_size);

    // copy data to device
    cudaMemcpy(input_data_device, 
        input_data_host, conf.orig_size, cudaMemcpyHostToDevice);

    // create compressor object
    fz::Compressor<float> compressor(conf);

    compressor.compress(input_data_device, &internal_compressed, stream);

    //! internal_compressed is a pointer to the compressed data on the gpu

    // copy out compressed data (if not dumped to file, can set in config)
    cudaMallocHost(&compressed_data_host, conf.comp_size);
    cudaMemcpy(compressed_data_host, 
        internal_compressed, conf.comp_size, cudaMemcpyDeviceToHost);

    // free memory
    cudaFreeHost(input_data_host);
    cudaFree(input_data_device);
    cudaFree(internal_compressed);
    cudaStreamSynchronize(stream);

    // output is dumped to fname.fzmod and used in decompression
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

void decompress_demo_file(std::string fname, cudaStream_t stream) {

    std::string compressed_fname = fname + ".fzmod";
    
    // create decompressor object
    fz::Decompressor<float> decompressor(compressed_fname);

    decompressor.conf->toFile = true;

    // decompress the data
    float* decompressed;
    size_t original_size = decompressor.conf->orig_size;
    cudaMalloc(&decompressed, original_size);

    // get original data
    float* original_data_host;
    cudaMallocHost(&original_data_host, original_size);
    utils::fromfile(fname, original_data_host, original_size);

    decompressor.decompress(compressed_data_host, 
        decompressed, stream, original_data_host);

    // Free memory
    cudaFree(decompressed);
    cudaFreeHost(original_data_host);
    cudaStreamSynchronize(stream);
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

int main(int argc, char **argv) {

    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << 
            " <filename> <len1> <len2> <len3> <eb>" << std::endl;
        return 1;
    }
    auto fname = std::string(argv[1]);
    size_t len1 = std::stoi(argv[2]);
    size_t len2 = std::stoi(argv[3]);
    size_t len3 = std::stoi(argv[4]);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // COMPRESSION
    compress_demo(fname, len1, len2, len3, stream);

    // DECOMPRESSION
    decompress_demo_file(fname, stream);

    cudaStreamDestroy(stream);
    cudaFreeHost(compressed_data_host);
    return 0;
}
