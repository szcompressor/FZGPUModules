/*
* filename: demo.cc
* author: Skyler Ruiter
* date: 06/23/2025
* 
* A demonstration of the fzmod compression library using CUDA.
* This example shows how to compress and decompress data using the fzmod library.
* It includes steps for setting up the configuration, compressing data, and decompressing it.
* This demonstrates a simple end-to-end compression and decompression workflow
* that writes and reads to disk.
*/

#include "fzmod.hh"
namespace utils = _portable::utils; // for file I/O utilities

// pointers for host versions of compressed and decompressed data
uint8_t* compressed_data_host;
float* decompressed_data_host;

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Compression Demo
void compress_demo(std::string fname, size_t x, 
    size_t y, size_t z, cudaStream_t stream) {
    
    //! STEP 2: Setup config with compression options
    //! these are settings that adjust the runtime/compression behavior
    fz::Config<float> conf(x, y, z);
    conf.eb = 5e-5; // set error bound
    conf.eb_type = fz::EB_TYPE::REL; // set error bound type
    conf.algo = fz::ALGO::LORENZO; // set algorithm type
    conf.codec = fz::CODEC::FZG; // set codec type
    conf.lossless_codec_2 = fz::SECONDARY_CODEC::NONE; // no secondary codec
    conf.fromFile = false; // not reading directly from file
    conf.fname = fname; // filename for output
    conf.verbose = true;

    //! STEP 3: get the input data on the device (file->host->device)
    float* input_data_device, * input_data_host;
    cudaMallocHost(&input_data_host, conf.orig_size);
    cudaMalloc(&input_data_device, conf.orig_size);
    utils::fromfile(fname, input_data_host, conf.orig_size);
    cudaMemcpy(input_data_device, input_data_host, conf.orig_size, cudaMemcpyHostToDevice);

    //! STEP 4: Create compressor object with the config
    //! this will do internal memory allocation and setup
    fz::Compressor<float> compressor(conf);

    //! STEP 5: Get pointer for the internal compressed data
    //! this will be the location of the compressed data on the GPU
    uint8_t* comp_data_d;

    //! STEP 6: Compress the data on the GPU
    compressor.compress(input_data_device, &comp_data_d, stream);

    //! STEP 7: copy out compressed data (if not dumped to file, can set in config)
    //! or leave it on the device for further processing
    cudaMallocHost(&compressed_data_host, conf.comp_size);
    cudaMemcpy(compressed_data_host, 
        comp_data_d, conf.comp_size, cudaMemcpyDeviceToHost);

    //! STEP 8: free memory
    cudaFreeHost(input_data_host);
    cudaFree(input_data_device);
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Decompression Demo
void decompress_demo_file(std::string fname, cudaStream_t stream) {

    //! STEP 10: Make a decompressor object with the compressed filename
    //! this will read the header to setup internal buffers and configuration
    std::string compressed_fname = fname + ".fzmod";
    fz::Decompressor<float> decompressor(compressed_fname);

    //! STEP 11: Make any adjustments to the decompressor config
    decompressor.conf->toFile = true;
    decompressor.conf->verbose = true;

    //! STEP 12: allocate memory for the decompressed data
    //! this will be the location of the decompressed data on the GPU
    float* decompressed;
    size_t original_size = decompressor.conf->orig_size;
    cudaMalloc(&decompressed, original_size);

    //! STEP 12.5: Get the original data for data quality analysis
    //! this is optional, if you want to compare the data
    float* original_data_host;
    cudaMallocHost(&original_data_host, original_size);
    utils::fromfile(fname, original_data_host, original_size);

    //! STEP 13: Decompress the data on the GPU
    decompressor.decompress(compressed_data_host, 
        decompressed, stream, original_data_host);

    //! STEP 14: Free the memory
    cudaFree(decompressed);
    cudaFreeHost(original_data_host);
    cudaStreamSynchronize(stream);
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

int main(int argc, char **argv) {

    // USAGE: ./fzmod_demo <filename> <len1> <len2> <len3>

    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << 
            " <filename> <len1> <len2> <len3>" << std::endl;
        return 1;
    }
    auto fname = std::string(argv[1]);
    size_t len1 = std::stoi(argv[2]);
    size_t len2 = std::stoi(argv[3]);
    size_t len3 = std::stoi(argv[4]);

    //! STEP 1: Create a cuda stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    //! COMPRESSION (STEPS 2-9)
    compress_demo(fname, len1, len2, len3, stream);

    printf("\n### COMPRESSION COMPLETE ###\n\n");

    //! DECOMPRESSION (STEPS 10-14)
    decompress_demo_file(fname, stream);

    //! STEP 15: Cleanup
    cudaStreamDestroy(stream);
    cudaFreeHost(compressed_data_host);
    return 0;
}
