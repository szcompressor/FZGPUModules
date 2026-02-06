#include "bitpacking.h"
#include <cstdio>
#include <algorithm>

namespace fz {

// ========== Helper Functions ==========

// Count leading zeros to determine bits needed
__device__ __host__ inline uint32_t clz(uint32_t x) {
#ifdef __CUDA_ARCH__
    return __clz(x);
#else
    if (x == 0) return 32;
    return __builtin_clz(x);
#endif
}

// Calculate bits needed to represent a value
__device__ __host__ inline uint32_t bits_needed(uint32_t val) {
    if (val == 0) return 1;
    return 32 - clz(val);
}

// ========== CUDA Kernels ==========

/**
 * Block header structure (4 bytes for compatibility)
 * For uint32_t inputs, min_value stores lower 16 bits
 */
struct BlockHeader {
    uint16_t min_value;  // Minimum value in block (or lower 16 bits for uint32)
    uint8_t num_bits;    // Bits needed per element
    uint8_t reserved;    // Padding for alignment
};

/**
 * Simple sequential bitpacking kernel - supports uint8/16/32/64
 * Process one block at a time for correctness
 */
template<typename TInput>
__global__ void bitpack_sequential_kernel(
    const TInput* __restrict__ input,
    uint8_t* __restrict__ output,
    size_t num_elements,
    size_t block_size
) {
    // Only thread 0 does the work
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    
    size_t num_blocks = (num_elements + block_size - 1) / block_size;
    
    // Write num_blocks header
    *reinterpret_cast<uint32_t*>(output) = num_blocks;
    
    // Pointer to block headers
    BlockHeader* headers = reinterpret_cast<BlockHeader*>(output + sizeof(uint32_t));
    
    // Pointer to packed data (after all headers)
    uint8_t* packed_data = output + sizeof(uint32_t) + num_blocks * sizeof(BlockHeader);
    size_t bit_offset = 0;  // Current bit position in packed_data
    
    // Process each block
    for (size_t block_idx = 0; block_idx < num_blocks; block_idx++) {
        size_t block_start = block_idx * block_size;
        size_t block_end = min(block_start + block_size, num_elements);
        size_t block_len = block_end - block_start;
        
        // Find min and max in this block
        TInput min_val = input[block_start];
        TInput max_val = input[block_start];
        
        for (size_t i = block_start + 1; i < block_end; i++) {
            TInput val = input[i];
            if (val < min_val) min_val = val;
            if (val > max_val) max_val = val;
        }
        
        // Calculate bits needed
        uint64_t range = (uint64_t)max_val - (uint64_t)min_val;  // Use uint64 to avoid overflow
        uint32_t num_bits = bits_needed(range > UINT32_MAX ? UINT32_MAX : (uint32_t)range);
        if (num_bits == 0) num_bits = 1;  // At least 1 bit
        
        // For types larger than uint16_t, we can't store full min_value in the header
        // So we skip delta encoding and just pack the raw values
        TInput actual_min = min_val;
        if (sizeof(TInput) > sizeof(uint16_t)) {
            // No delta encoding for uint32/uint64 - pack raw values
            actual_min = 0;
            num_bits = bits_needed(max_val);  // Bits needed for max value itself
        }
        
        // Write block header (store lower 16 bits of min for compatibility)
        headers[block_idx].min_value = (uint16_t)(actual_min & 0xFFFF);
        headers[block_idx].num_bits = num_bits;
        headers[block_idx].reserved = 0;
        
        // Pack the values in this block
        for (size_t i = block_start; i < block_end; i++) {
            uint64_t value = (uint64_t)input[i] - (uint64_t)actual_min;  // Subtract minimum
            
            // Write value bit by bit
            for (uint32_t bit = 0; bit < num_bits; bit++) {
                uint32_t bit_val = (value >> bit) & 1;
                
                // Calculate byte and bit position
                size_t byte_pos = bit_offset / 8;
                size_t bit_pos = bit_offset % 8;
                
                // Set or clear the bit
                if (bit_val) {
                    packed_data[byte_pos] |= (1 << bit_pos);
                } else {
                    packed_data[byte_pos] &= ~(1 << bit_pos);
                }
                
                bit_offset++;
            }
        }
    }
}

// Test kernel
__global__ void test_empty_kernel() {}

// ========== BitpackingStage Implementation ==========

template<typename TInput>
BitpackingStage<TInput>::BitpackingStage(const BitpackingConfig<TInput>& config)
    : Stage("Bitpack_" + std::string(
          sizeof(TInput) == 1 ? "u8" : 
          sizeof(TInput) == 2 ? "u16" : "u32")),
      config_(config),
      graph_d_input_(nullptr),
      graph_d_output_(nullptr),
      graph_num_elements_(0) {
}

template<typename TInput>
BitpackingStage<TInput>::~BitpackingStage() {
}

template<typename TInput>
StageMemoryRequirements BitpackingStage<TInput>::getMemoryRequirements(size_t input_size) const {
    size_t num_elements = input_size / sizeof(TInput);
    size_t num_blocks = (num_elements + config_.block_size - 1) / config_.block_size;
    
    // Output: worst case = input size + headers
    size_t output_size = sizeof(uint32_t) + num_blocks * sizeof(BlockHeader) + input_size;
    
    StageMemoryRequirements req;
    req.output_size = output_size;
    req.aux_output_size = 0;  // No auxiliary buffers needed
    req.temp_size = 0;
    
    return req;
}

template<typename TInput>
int BitpackingStage<TInput>::execute(void* input, size_t input_size,
                                     void* output, cudaStream_t stream) {
    const TInput* d_input = static_cast<const TInput*>(input);
    uint8_t* d_output = static_cast<uint8_t*>(output);
    size_t num_elements = input_size / sizeof(TInput);
    
    // Validate TInput is an unsigned integer type
    static_assert(std::is_unsigned<TInput>::value && std::is_integral<TInput>::value,
                  "TInput must be an unsigned integer type (uint8_t, uint16_t, uint32_t, uint64_t)");
    
    // Check for empty input
    if (num_elements == 0) {
        fprintf(stderr, "Bitpacking: Empty input (input_size=%zu)\n", input_size);
        // Write empty output: 0 blocks
        uint32_t zero = 0;
        cudaMemcpyAsync(output, &zero, sizeof(uint32_t), cudaMemcpyHostToDevice, stream);
        cudaStreamSynchronize(stream);
        return sizeof(uint32_t);  // Just the header
    }
    
    // Validate pointers
    if (d_input == nullptr || d_output == nullptr) {
        fprintf(stderr, "Bitpacking: null input or output pointer (input=%p, output=%p)\n", d_input, d_output);
        return -1;
    }
    
    recordProfilingStart(stream);
    
    // Clear any previous errors
    cudaGetLastError();
    
    // Clear output buffer to ensure unused bits are zero
    // Note: In CONSERVATIVE memory mode, the allocated buffer might be smaller than max_output
    // We'll memset only what we can, relying on the kernel to initialize properly
    size_t max_output = getMaxOutputSize(input_size);
    
    // Try to memset, but don't fail if buffer is too small (CONSERVATIVE mode)
    cudaError_t err = cudaMemsetAsync(d_output, 0, max_output, stream);
    if (err != cudaSuccess) {
        // Memset failed - likely buffer too small in CONSERVATIVE mode
        // Clear the error and continue without memset
        cudaGetLastError();
    }
    
    // Launch templated sequential kernel
    fprintf(stderr, "[BITPACK] Processing %zu elements (type size=%zu)\n", 
            num_elements, sizeof(TInput));
    bitpack_sequential_kernel<TInput><<<1, 1, 0, stream>>>(
        d_input, d_output, num_elements, config_.block_size
    );
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Bitpacking kernel launch failed: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    cudaStreamSynchronize(stream);
    
    // Calculate actual output size
    uint32_t num_blocks;
    cudaMemcpyAsync(&num_blocks, d_output, sizeof(uint32_t),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    // Read headers to calculate total packed size
    size_t header_offset = sizeof(uint32_t);
    BlockHeader* headers = new BlockHeader[num_blocks];
    cudaMemcpyAsync(headers, d_output + header_offset, 
                    num_blocks * sizeof(BlockHeader),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    // Calculate total bits used
    size_t total_bits = 0;
    size_t total_elements = 0;
    uint64_t bits_histogram[17] = {0};  // Count blocks by bits used (0-16)
    
    for (uint32_t i = 0; i < num_blocks; i++) {
        size_t block_len = (i == num_blocks - 1) ? 
            (num_elements - i * config_.block_size) : config_.block_size;
        total_bits += block_len * headers[i].num_bits;
        total_elements += block_len;
        bits_histogram[headers[i].num_bits]++;
    }
    
    // Print statistics
    // printf("Bitpacking stats: %u blocks, %zu elements\\n", num_blocks, total_elements);
    // printf("  Bits distribution: ");
    // for (int b = 0; b <= 16; b++) {
    //     if (bits_histogram[b] > 0) {
    //         printf("%db:%lu ", b, bits_histogram[b]);
    //     }
    // }
    // printf("\\n");
    // printf("  Avg bits/element: %.2f\\n", (double)total_bits / total_elements);
    
    delete[] headers;
    
    size_t packed_bytes = (total_bits + 7) / 8;  // Round up to bytes
    size_t output_size = header_offset + num_blocks * sizeof(BlockHeader) + packed_bytes;
    
    recordProfilingEnd(stream);
    updateProfilingStats();
    
    return output_size;
}

template<typename TInput>
cudaGraphNode_t BitpackingStage<TInput>::addToGraph(
    cudaGraph_t graph,
    cudaGraphNode_t* dependencies,
    size_t num_deps,
    void* input, size_t input_size,
    void* output,
    const std::vector<void*>& aux_buffers,
    cudaStream_t stream) {
    
    // Store parameters in member variables
    graph_d_input_ = static_cast<const TInput*>(input);
    graph_d_output_ = static_cast<uint8_t*>(output);
    graph_num_elements_ = input_size / sizeof(TInput);
    
    // Clear any previous CUDA errors
    cudaGetLastError();
    
    // Node 1: Memset output buffer to zero  
    cudaGraphNode_t memset_node;
    cudaMemsetParams memset_params = {};
    memset_params.dst = graph_d_output_;
    memset_params.value = 0;
    memset_params.pitch = 0;
    memset_params.elementSize = 1;
    size_t max_output = getMaxOutputSize(input_size);
    memset_params.width = max_output;
    memset_params.height = 1;
    
    // Try smaller memset or skip entirely for large buffers
    cudaGraphNode_t* deps_for_kernel = dependencies;
    size_t num_deps_for_kernel = num_deps;
    bool memset_added = false;
    
    if (max_output < 10000000) {  // Only memset if < 10MB
        cudaError_t err = cudaGraphAddMemsetNode(&memset_node, graph, dependencies, num_deps, &memset_params);
        if (err != cudaSuccess) {
            // Memset failed, skip and use original dependencies
        } else {
            deps_for_kernel = &memset_node;
            num_deps_for_kernel = 1;
            memset_added = true;
        }
    }
    
    // Node 2: Bitpacking kernel
    // Use the typed input pointer
    const TInput* d_input_typed = static_cast<const TInput*>((const void*)graph_d_input_);
    
    void* kernel_args[] = {
        (void*)&d_input_typed,
        (void*)&graph_d_output_,
        (void*)&graph_num_elements_,
        (void*)&config_.block_size
    };
    
    cudaKernelNodeParams kernel_params = {};
    kernel_params.func = (void*)bitpack_sequential_kernel<TInput>;
    kernel_params.gridDim = dim3(1);
    kernel_params.blockDim = dim3(1);
    kernel_params.sharedMemBytes = 0;
    kernel_params.kernelParams = kernel_args;
    kernel_params.extra = nullptr;
    
    cudaGraphNode_t kernel_node;
    cudaError_t err = cudaGraphAddKernelNode(&kernel_node, graph, deps_for_kernel, num_deps_for_kernel, &kernel_params);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to add Bitpacking kernel node: %s\n", cudaGetErrorString(err));
        return nullptr;
    }
    
    return kernel_node;
}

// Explicit template instantiations
template class BitpackingStage<uint8_t>;
template class BitpackingStage<uint16_t>;
template class BitpackingStage<uint32_t>;

} // namespace fz
