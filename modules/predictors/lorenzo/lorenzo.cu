#include "predictors/lorenzo/lorenzo.h"
#include "predictors/predictor_utils.cuh"
#include "transforms/zigzag/zigzag.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <type_traits>
#include "mem/mempool.h"
#include "cuda_check.h"
#include "log.h"

namespace fz {

// ===== Warp-level scan primitives for parallel prefix sum =====

template<typename T, int Seq>
__device__ void warp_inclusive_scan(T* thp_data) {
    // Inclusive scan within a warp (32 threads)
    // Each thread has Seq elements in registers
    
    // First, do sequential scan within thread's Seq elements
    for (int i = 1; i < Seq; i++) {
        thp_data[i] += thp_data[i - 1];
    }
    
    // Now scan across threads in warp
    T thread_sum = thp_data[Seq - 1];  // Last element of this thread
    
    // Get lane ID within warp (0-31)
    int lane_id = threadIdx.x % 32;
    
    // Warp-level scan using shuffle operations
    #pragma unroll
    for (int offset = 1; offset < 32; offset *= 2) {
        T val = __shfl_up_sync(0xffffffff, thread_sum, offset);
        if (lane_id >= offset) {
            thread_sum += val;
        }
    }
    
    // Get prefix sum from previous thread in warp
    T prefix = __shfl_up_sync(0xffffffff, thread_sum, 1);
    if (lane_id == 0) prefix = 0;
    
    // Add prefix to all elements in this thread
    for (int i = 0; i < Seq; i++) {
        thp_data[i] += prefix;
    }
}

template<typename T, int Seq, int NumThreads>
__device__ void block_exclusive_scan(T* thp_data, T* exch_in, T* exch_out) {
    constexpr int NumWarps = NumThreads / 32;
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    
    // Each warp writes its total to shared memory
    if (lane_id == 31) {
        exch_in[warp_id] = thp_data[Seq - 1];
    }
    __syncthreads();
    
    // First warp scans the per-warp totals
    if (warp_id == 0 && lane_id < NumWarps) {
        T val = exch_in[lane_id];
        
        // Create mask for only the participating threads
        unsigned mask = (1u << NumWarps) - 1;  // e.g., 0xFF for 8 warps
        
        #pragma unroll
        for (int offset = 1; offset < NumWarps; offset *= 2) {
            T tmp = __shfl_up_sync(mask, val, offset);
            if (lane_id >= offset) val += tmp;
        }
        exch_out[lane_id] = val;
    }
    __syncthreads();
    
    // Add the prefix from previous warps to all threads
    if (warp_id > 0) {
        T warp_prefix = exch_out[warp_id - 1];
        for (int i = 0; i < Seq; i++) {
            thp_data[i] += warp_prefix;
        }
    }
}

/**
 * Scatter kernel to restore outliers to their original positions
 * 
 * Takes sparse (value, index) pairs and ADDS outlier prediction errors
 * to the already-reconstructed values from quantized codes.
 * 
 * During compression, outlier positions store code=radius, 
 * so after scan they have just the cumulative sum up to that point.
 * This kernel scales the outlier errors (stored in quantized units) by ebx2
 * and adds them to the reconstructed values.
 */
template<typename T>
__global__ void scatter_outliers_kernel(
    const T* __restrict__ outlier_values,
    const uint32_t* __restrict__ outlier_indices,
    const uint32_t* __restrict__ outlier_count_ptr,  // device pointer — no D2H sync needed
    T* __restrict__ output
) {
    // Read count from device; blocks without real work exit cheaply.
    uint32_t outlier_count = *outlier_count_ptr;
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < outlier_count) {
        uint32_t idx = outlier_indices[tid];
        // Write the outlier prediction error in quantized units
        // The prefix sum kernel will read this and incorporate it into the sum
        output[idx] = outlier_values[tid];
    }
}

/**
 * Helper kernel to propagate block sums across blocks
 * After each block computes its local prefix sum, we need to add the total
 * from all previous blocks to maintain global cumulative sum.
 */
template<typename T>
__global__ void propagate_block_sums_kernel(
    T* __restrict__ data,
    T* __restrict__ block_sums,
    const size_t n,
    const int tile_dim
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Determine which original tile (dequantize block) this element belongs to
        int original_block = static_cast<int>(idx / tile_dim);
        
        // Add cumulative sum from all previous original blocks
        if (original_block > 0) {
            T prefix = 0;
            for (int b = 0; b < original_block; b++) {
                prefix += block_sums[b];
            }
            data[idx] += prefix;
        }
    }
}

/**
 * Helper kernel to extract last value from each block
 */
template<typename T>
__global__ void extract_block_sums_kernel(
    const T* __restrict__ data,
    T* __restrict__ block_sums,
    const size_t n,
    const int tile_dim
) {
    int block_id = blockIdx.x;
    
    if (threadIdx.x == 0) {
        // Get last valid element in this block
        size_t last_idx = min(static_cast<size_t>((block_id + 1) * tile_dim - 1), n - 1);
        block_sums[block_id] = data[last_idx];
    }
}

/**
 * Optimized Lorenzo 1D decompression kernel with parallel prefix sum
 * 
 * Based on cuSZ-style block-level scan algorithm:
 * 1. Load and dequantize: codes → prediction errors
 * 2. Parallel prefix sum: warp-level + block-level scan (inverse Lorenzo)
 * 3. Scale by 2*eb and write output
 * 
 * Note: This kernel only handles quantized codes. Outliers are scattered
 * separately using scatter_outliers_kernel.
 * 
 * Performance optimizations:
 * - Shared memory staging for coalesced access
 * - Warp shuffle for efficient intra-warp scan
 * - Block-level coordination for inter-warp scan
 * - Sequential processing per thread (Seq=4) for better occupancy
 */
template<typename TInput, typename TCode, int TileDim, int Seq, bool ZigzagCodes = false>
__global__ void lorenzo_dequantize_1d_kernel(
    const TCode* __restrict__ quant_codes,
    const size_t n,
    const TInput ebx2,
    const TCode quant_radius,
    TInput* __restrict__ output
) {
    constexpr int NumThreads = TileDim / Seq;
    
    __shared__ TInput scratch[TileDim];
    __shared__ TCode s_codes[TileDim];
    __shared__ TInput exch_in[NumThreads / 32];
    __shared__ TInput exch_out[NumThreads / 32];
    
    TInput thp_data[Seq];  // Private registers for each thread
    
    const size_t block_offset = blockIdx.x * TileDim;
    
    // ===== Stage 1: Load codes to shared memory (sequential per thread) =====
    #pragma unroll
    for (int i = 0; i < Seq; i++) {
        size_t local_id = threadIdx.x * Seq + i;  // Sequential: thread 0 reads 0,1,2,3
        size_t global_id = block_offset + local_id;
        if (global_id < n && local_id < TileDim) {
            s_codes[local_id] = quant_codes[global_id];
        }
    }
    __syncthreads();
    
    // ===== Stage 2: Dequantize codes to prediction errors =====
    // Write in sequential order for each thread's Seq elements
    #pragma unroll
    for (int i = 0; i < Seq; i++) {
        size_t local_id = threadIdx.x * Seq + i;  // Sequential: 0,1,2,3 for thread 0
        size_t global_id = block_offset + local_id;
        
        if (global_id < n && local_id < TileDim) {
            // Dequantize: reinterpret unsigned code as signed two's-complement
            // to recover the signed delta stored during compression (q, not q+radius).
            using SCode = typename std::make_signed<TCode>::type;
            TInput delta;
            if constexpr (ZigzagCodes) {
                // Zigzag-decode: unsigned code → signed prediction error
                delta = static_cast<TInput>(Zigzag<SCode>::decode(s_codes[local_id]));
            } else {
                delta = static_cast<TInput>(static_cast<SCode>(s_codes[local_id]));
            }
            
            // Inject any scattered outlier prediction errors BEFORE prefix sum
            delta += output[global_id];
            
            scratch[local_id] = delta;
        } else if (local_id < TileDim) {
            scratch[local_id] = 0;
        }
    }
    __syncthreads();
    
    // ===== Stage 3: Load to private registers =====
    #pragma unroll
    for (int i = 0; i < Seq; i++) {
        thp_data[i] = scratch[threadIdx.x * Seq + i];
    }
    __syncthreads();
    
    // ===== Stage 4: Parallel prefix sum (cumulative sum) =====
    // Intra-warp inclusive scan
    warp_inclusive_scan<TInput, Seq>(thp_data);
    
    // Inter-warp exclusive scan (coordinate across warps in block)
    block_exclusive_scan<TInput, Seq, NumThreads>(thp_data, exch_in, exch_out);
    
    // ===== Stage 5: Scale by 2*eb and write back to shared memory =====
    #pragma unroll
    for (int i = 0; i < Seq; i++) {
        scratch[threadIdx.x * Seq + i] = thp_data[i] * ebx2;
    }
    __syncthreads();
    
    // ===== Stage 6: Write output to global memory (sequential per thread) =====
    #pragma unroll
    for (int i = 0; i < Seq; i++) {
        size_t local_id = threadIdx.x * Seq + i;  // Sequential: thread 0 writes 0,1,2,3
        size_t global_id = block_offset + local_id;
        if (global_id < n && local_id < TileDim) {
            output[global_id] = scratch[local_id];
        }
    }
}

/**
 * Optimized Lorenzo 1D prediction + quantization kernel
 * 
 * Performance optimizations:
 * 1. Pre-quantization: Multiply by ebx2_r first to simplify all subsequent math
 * 2. Shared memory staging: Better memory coalescing for global loads/stores
 * 3. Sequential processing: Each thread processes Seq elements for better occupancy
 * 4. Proper thread boundaries: Uses shared memory to get previous element correctly
 * 5. Direct radius check: Simpler outlier detection (fabs(delta) < radius)
 * 6. Store prediction errors: More compressible than original values
 * 
 * Memory layout:
 * - TileDim = 1024 elements per block
 * - Seq = 4 elements per thread
 * - NumThreads = 256 (TileDim / Seq)
 */
template<typename TInput, typename TCode, int TileDim, int Seq, bool ZigzagCodes = false>
__global__ void lorenzo_quantize_1d_kernel(
    const TInput* __restrict__ input,
    const size_t n,
    const TInput ebx2_r,
    const TCode quant_radius,
    TCode* __restrict__ quant_codes,
    TInput* __restrict__ outlier_errors,
    uint32_t* __restrict__ outlier_indices,
    uint32_t* __restrict__ outlier_count,
    const size_t max_outliers
) {
    constexpr int NumThreads = TileDim / Seq;
    
    // Shared memory for staging and better memory access patterns
    __shared__ TInput s_data[TileDim];
    __shared__ TCode s_codes[TileDim];
    
    // Private thread data: current Seq elements + 1 previous element
    TInput thp_data[Seq + 1];
    
    const size_t block_offset = blockIdx.x * TileDim;
    
    // ========== Stage 1: Load input data to shared memory with pre-quantization ==========
    // Coalesced loads: each thread loads Seq elements
    #pragma unroll
    for (int i = 0; i < Seq; i++) {
        size_t global_idx = block_offset + threadIdx.x + i * NumThreads;
        if (global_idx < n) {
            // Pre-quantize: multiply by 1/(2*eb) to normalize to quantization units
            s_data[threadIdx.x + i * NumThreads] = round(input[global_idx] * ebx2_r);
        }
    }
    __syncthreads();
    
    // ========== Stage 2: Load to private registers ==========
    // Each thread loads its Seq elements
    #pragma unroll
    for (int i = 0; i < Seq; i++) {
        thp_data[i + 1] = s_data[threadIdx.x * Seq + i];
    }
    
    // Load previous element for Lorenzo prediction.
    // The first thread of every block restarts the chain at 0 (no cross-block
    // lookahead). This mirrors cuSZ and makes each block's prefix sum fully
    // independent, eliminating the inter-block propagation pass on decompress.
    if (threadIdx.x > 0) {
        thp_data[0] = s_data[threadIdx.x * Seq - 1];
    } else {
        thp_data[0] = static_cast<TInput>(0);
    }
    
    // ========== Stage 3: Lorenzo prediction + quantization ==========
    #pragma unroll
    for (int i = 0; i < Seq; i++) {
        size_t global_idx = block_offset + threadIdx.x * Seq + i;
        
        if (global_idx < n) {
            // Lorenzo 1D: predict from previous element
            TInput delta = thp_data[i + 1] - thp_data[i];
            
            // Check if quantizable (directly using radius)
            bool quantizable = fabs(delta) < static_cast<TInput>(quant_radius);
            
            if (quantizable) {
                // Store signed delta (two's-complement or zigzag-encoded in TCode).
                if constexpr (ZigzagCodes) {
                    using SCode = typename std::make_signed<TCode>::type;
                    s_codes[threadIdx.x * Seq + i] =
                        static_cast<TCode>(Zigzag<SCode>::encode(static_cast<SCode>(static_cast<int>(delta))));
                } else {
                    s_codes[threadIdx.x * Seq + i] = static_cast<TCode>(static_cast<int>(delta));
                }
            } else {
                // Outlier: store 0 so this position contributes nothing to the
                // prefix sum.  The true delta is replayed via scatter_outliers_kernel.
                s_codes[threadIdx.x * Seq + i] = static_cast<TCode>(0);
                
                // Store outlier prediction error (not original value!)
                uint32_t outlier_idx = atomicAdd(outlier_count, 1);
                if (outlier_idx < max_outliers) {
                    outlier_errors[outlier_idx] = delta;  // Store prediction error
                    outlier_indices[outlier_idx] = static_cast<uint32_t>(global_idx);
                }
                
                // Do NOT rewrite thp_data[i + 1]. We want the next element to 
                // predict from this element's true value, not the predicted value.
                // The decompression prefix sum will correctly reconstruct this.
            }
        }
    }
    __syncthreads();
    
    // ========== Stage 4: Write quantization codes to global memory ==========
    // Coalesced stores
    #pragma unroll
    for (int i = 0; i < Seq; i++) {
        size_t global_idx = block_offset + threadIdx.x + i * NumThreads;
        if (global_idx < n) {
            quant_codes[global_idx] = s_codes[threadIdx.x + i * NumThreads];
        }
    }
}



// Kernel launcher wrappers (force kernel instantiation)
template<typename TInput, typename TCode>
void launchLorenzoKernel(
    const TInput* input,
    size_t n,
    TInput ebx2_r,
    TCode quant_radius,
    TCode* quant_codes,
    TInput* outlier_errors,
    uint32_t* outlier_indices,
    uint32_t* outlier_count,
    size_t max_outliers,
    int grid_size,
    bool zigzag_codes,
    cudaStream_t stream
) {
    constexpr int TileDim = 1024;
    constexpr int Seq = 4;
    constexpr int NumThreads = TileDim / Seq;  // 256

    if (zigzag_codes) {
        lorenzo_quantize_1d_kernel<TInput, TCode, TileDim, Seq, true>
            <<<grid_size, NumThreads, 0, stream>>>(
                input, n, ebx2_r, quant_radius,
                quant_codes, outlier_errors, outlier_indices, outlier_count,
                max_outliers
            );
    } else {
        lorenzo_quantize_1d_kernel<TInput, TCode, TileDim, Seq, false>
            <<<grid_size, NumThreads, 0, stream>>>(
                input, n, ebx2_r, quant_radius,
                quant_codes, outlier_errors, outlier_indices, outlier_count,
                max_outliers
            );
    }
}

template<typename TInput, typename TCode>
void launchLorenzoInverseKernel(
    const TCode* quant_codes,
    const TInput* outlier_errors,
    const uint32_t* outlier_indices,
    const uint32_t* outlier_count_ptr,
    size_t n,
    size_t max_outliers,    // pre-allocated capacity — avoids D2H count read
    TInput ebx2,
    TCode quant_radius,
    TInput* output,
    bool zigzag_codes,
    cudaStream_t stream,
    MemoryPool* pool
) {
    constexpr int TileDim = 1024;
    constexpr int Seq = 4;
    constexpr int NumThreads = TileDim / Seq;  // 256

    const int grid_size = (n + TileDim - 1) / TileDim;

    // Step 0: Initialize output array to 0 because we will scatter outliers into it
    FZ_CUDA_CHECK(cudaMemsetAsync(output, 0, n * sizeof(TInput), stream));

    // Step 1: Scatter outliers BEFORE prefix sum.
    // Launch with the full allocation capacity so no host-side D2H is needed;
    // scatter_outliers_kernel reads the actual count from the device pointer and
    // each thread that exceeds it exits immediately.
    if (outlier_count_ptr != nullptr && max_outliers > 0) {
        int scatter_block_size = 256;
        int scatter_grid_size  = (static_cast<int>(max_outliers) + scatter_block_size - 1)
                                 / scatter_block_size;

        scatter_outliers_kernel<TInput>
            <<<scatter_grid_size, scatter_block_size, 0, stream>>>(
                outlier_errors, outlier_indices, outlier_count_ptr, output
            );

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("scatter_outliers_kernel launch failed: ") +
                cudaGetErrorString(err)
            );
        }
    }

    // Step 2: Launch decompression kernel (dequantize + intra-block prefix sum).
    // Because compression no longer crosses block boundaries, each block's scan
    // is fully self-contained — no inter-block propagation pass is needed.
    if (zigzag_codes) {
        lorenzo_dequantize_1d_kernel<TInput, TCode, TileDim, Seq, true>
            <<<grid_size, NumThreads, 0, stream>>>(
                quant_codes, n, ebx2, quant_radius, output
            );
    } else {
        lorenzo_dequantize_1d_kernel<TInput, TCode, TileDim, Seq, false>
            <<<grid_size, NumThreads, 0, stream>>>(
                quant_codes, n, ebx2, quant_radius, output
            );
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("lorenzo_dequantize_1d_kernel launch failed: ") +
            cudaGetErrorString(err)
        );
    }
}

// ========== LorenzoStage Implementation ==========

template<typename TInput, typename TCode>
LorenzoStage<TInput, TCode>::LorenzoStage(const Config& config)
    : config_(config),
      computed_abs_eb_(static_cast<TInput>(config.error_bound)),
      computed_value_base_(config.precomputed_value_base) {
    actual_output_sizes_.resize(4, 0);
}

template<typename TInput, typename TCode>
void LorenzoStage<TInput, TCode>::execute(
    cudaStream_t stream,
    MemoryPool* pool,
    const std::vector<void*>& inputs,
    const std::vector<void*>& outputs,
    const std::vector<size_t>& sizes
) {
    if (is_inverse_) {
        // ===== DECOMPRESSION MODE: 4 inputs → 1 output =====
        if (inputs.size() < 4 || outputs.empty() || sizes.size() < 4) {
            throw std::runtime_error("LorenzoStage (inverse): Requires 4 inputs and 1 output");
        }

        size_t codes_size   = sizes[0];
        size_t num_elements = codes_size / sizeof(TCode);

        if (num_elements == 0) {
            actual_output_sizes_.resize(1);
            actual_output_sizes_[0] = 0;
            return;
        }

        // Calculate ebx2 = 2 * abs_error_bound for dequantization.
        // computed_abs_eb_ was set by deserializeHeader (always the absolute bound).
        TInput ebx2 = static_cast<TInput>(2) * computed_abs_eb_;

        // Derive max outlier capacity from the outlier_errors buffer size.
        size_t max_outliers = (sizes.size() > 1) ? (sizes[1] / sizeof(TInput)) : 0;

        int eff_ndim = ndim();

        if (eff_ndim == 3) {
            // -- 3-D inverse Lorenzo --
            size_t nx = (config_.dims[0] > 0) ? config_.dims[0]
                        : num_elements / (config_.dims[1] * config_.dims[2]);
            size_t ny = config_.dims[1];
            size_t nz = config_.dims[2];
            launchLorenzoInverseKernel3D<TInput, TCode>(
                static_cast<const TCode*>(inputs[0]),
                static_cast<const TInput*>(inputs[1]),
                static_cast<const uint32_t*>(inputs[2]),
                static_cast<const uint32_t*>(inputs[3]),
                nx, ny, nz, max_outliers,
                ebx2, config_.quant_radius,
                static_cast<TInput*>(outputs[0]),
                config_.zigzag_codes,
                stream, pool
            );
        } else if (eff_ndim == 2) {
            // -- 2-D inverse Lorenzo --
            size_t nx = (config_.dims[0] > 0) ? config_.dims[0]
                        : num_elements / config_.dims[1];
            size_t ny = config_.dims[1];
            launchLorenzoInverseKernel2D<TInput, TCode>(
                static_cast<const TCode*>(inputs[0]),
                static_cast<const TInput*>(inputs[1]),
                static_cast<const uint32_t*>(inputs[2]),
                static_cast<const uint32_t*>(inputs[3]),
                nx, ny, max_outliers,
                ebx2, config_.quant_radius,
                static_cast<TInput*>(outputs[0]),
                config_.zigzag_codes,
                stream, pool
            );
        } else {
            // -- 1-D inverse Lorenzo (original path) --
            launchLorenzoInverseKernel<TInput, TCode>(
                static_cast<const TCode*>(inputs[0]),
                static_cast<const TInput*>(inputs[1]),
                static_cast<const uint32_t*>(inputs[2]),
                static_cast<const uint32_t*>(inputs[3]),
                num_elements,
                max_outliers,
                ebx2,
                config_.quant_radius,
                static_cast<TInput*>(outputs[0]),
                config_.zigzag_codes,
                stream,
                pool
            );
        }

        // Check for launch errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("LorenzoStage (inverse) kernel launch failed: ") +
                cudaGetErrorString(err)
            );
        }

        // Set output size
        actual_output_sizes_.resize(1);
        actual_output_sizes_[0] = num_elements * sizeof(TInput);
        
    } else {
        // ===== COMPRESSION MODE: 1 input → 4 outputs =====
        if (inputs.empty() || outputs.size() < 4 || sizes.empty()) {
            throw std::runtime_error("LorenzoStage: Requires 1 input and 4 outputs");
        }

        size_t input_size  = sizes[0];
        size_t num_elements = input_size / sizeof(TInput);
        size_t max_outliers = getMaxOutlierCount(num_elements);

        // Store for header generation
        num_elements_ = num_elements;

        if (num_elements == 0) {
            // Empty input
            for (size_t i = 0; i < 4; i++) {
                actual_output_sizes_[i] = 0;
            }
            actual_outlier_count_ = 0;
            return;
        }

        // Initialize outlier count to 0
        FZ_CUDA_CHECK(cudaMemsetAsync(outputs[3], 0, sizeof(uint32_t), stream));

        // ── Resolve absolute error bound ──────────────────────────────────────
        // For ABS mode: abs_eb = config_.error_bound (no scan needed).
        // For NOA/REL modes: perform a stream-synchronising min/max scan first,
        //   then derive the absolute bound.  If the caller pre-computed
        //   value_base (> 0), skip the scan and use that value instead.
        if (config_.eb_mode == ErrorBoundMode::ABS) {
            computed_abs_eb_    = static_cast<TInput>(config_.error_bound);
            computed_value_base_ = 0.0f;
        } else {
            float value_base = config_.precomputed_value_base;
            if (value_base <= 0.0f) {
                value_base = computeValueBase<TInput>(
                    static_cast<const TInput*>(inputs[0]),
                    num_elements, config_.eb_mode, stream, pool);
            }
            computed_value_base_ = value_base;

            if (value_base <= 0.0f) {
                FZ_LOG(WARN,
                    "LorenzoStage: value_base is zero for %s mode "
                    "(constant or empty data?); falling back to ABS",
                    config_.eb_mode == ErrorBoundMode::NOA ? "NOA" : "REL");
                computed_abs_eb_ = static_cast<TInput>(config_.error_bound);
            } else {
                computed_abs_eb_ = static_cast<TInput>(config_.error_bound)
                                   * static_cast<TInput>(value_base);
            }
            FZ_LOG(DEBUG,
                "LorenzoStage %s: user_eb=%.6g value_base=%.6g -> abs_eb=%.6g",
                config_.eb_mode == ErrorBoundMode::NOA ? "NOA" : "REL",
                static_cast<double>(config_.error_bound),
                static_cast<double>(value_base),
                static_cast<double>(computed_abs_eb_));
        }

        // Calculate ebx2_r = 1 / (2 * abs_error_bound) for pre-quantization
        TInput ebx2_r = static_cast<TInput>(1)
                        / (static_cast<TInput>(2) * computed_abs_eb_);

        int eff_ndim = ndim();

        if (eff_ndim == 3) {
            // -- 3-D forward Lorenzo --
            size_t nx = (config_.dims[0] > 0) ? config_.dims[0]
                        : num_elements / (config_.dims[1] * config_.dims[2]);
            size_t ny = config_.dims[1];
            size_t nz = config_.dims[2];
            launchLorenzoKernel3D<TInput, TCode>(
                static_cast<const TInput*>(inputs[0]), nx, ny, nz,
                ebx2_r, config_.quant_radius,
                static_cast<TCode*>(outputs[0]),
                static_cast<TInput*>(outputs[1]),
                static_cast<uint32_t*>(outputs[2]),
                static_cast<uint32_t*>(outputs[3]),
                max_outliers, config_.zigzag_codes, stream
            );
        } else if (eff_ndim == 2) {
            // -- 2-D forward Lorenzo --
            size_t nx = (config_.dims[0] > 0) ? config_.dims[0]
                        : num_elements / config_.dims[1];
            size_t ny = config_.dims[1];
            launchLorenzoKernel2D<TInput, TCode>(
                static_cast<const TInput*>(inputs[0]), nx, ny,
                ebx2_r, config_.quant_radius,
                static_cast<TCode*>(outputs[0]),
                static_cast<TInput*>(outputs[1]),
                static_cast<uint32_t*>(outputs[2]),
                static_cast<uint32_t*>(outputs[3]),
                max_outliers, config_.zigzag_codes, stream
            );
        } else {
            // -- 1-D forward Lorenzo (original path) --
            constexpr int TileDim = 1024;
            const int grid_size = (num_elements + TileDim - 1) / TileDim;

            launchLorenzoKernel<TInput, TCode>(
                static_cast<const TInput*>(inputs[0]),
                num_elements,
                ebx2_r,
                config_.quant_radius,
                static_cast<TCode*>(outputs[0]),
                static_cast<TInput*>(outputs[1]),
                static_cast<uint32_t*>(outputs[2]),
                static_cast<uint32_t*>(outputs[3]),
                max_outliers,
                grid_size,
                config_.zigzag_codes,
                stream
            );
        }

        // Check for launch errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("LorenzoStage kernel launch failed: ") +
                cudaGetErrorString(err)
            );
        }

        // Store device pointer to the outlier count output so postStreamSync()
        // can read it back after the stream is fully synchronized by compress().
        // We must NOT sync here — doing so stalls the entire DAG mid-pipeline.
        d_outlier_count_ptr_ = outputs[3];

        // Use max-capacity sizes for now; postStreamSync() will trim them to
        // the real outlier count once the stream is idle.
        actual_outlier_count_ = 0;
        actual_output_sizes_[0] = num_elements * sizeof(TCode);
        actual_output_sizes_[1] = max_outliers * sizeof(TInput);
        actual_output_sizes_[2] = max_outliers * sizeof(uint32_t);
        actual_output_sizes_[3] = sizeof(uint32_t);
    }
}

template<typename TInput, typename TCode>
void LorenzoStage<TInput, TCode>::postStreamSync(cudaStream_t /*stream*/) {
    // Only applies to compression mode and only when execute() set the ptr.
    if (is_inverse_ || d_outlier_count_ptr_ == nullptr) return;

    // The stream is fully synchronized by the time Pipeline::compress() calls
    // us, so a plain (synchronous) cudaMemcpy is safe and adds no extra stall.
    uint32_t h_outlier_count = 0;
    FZ_CUDA_CHECK(cudaMemcpy(&h_outlier_count, d_outlier_count_ptr_, sizeof(uint32_t),
               cudaMemcpyDeviceToHost));
    d_outlier_count_ptr_ = nullptr;

    size_t max_outliers = getMaxOutlierCount(num_elements_);

    if (h_outlier_count > max_outliers) {
        if (config_.outlier_capacity == 0.0f) {
            // Capacity intentionally zero: all outliers are silently dropped.
            // This is a deliberate lossy trade-off; no warning is needed.
            FZ_LOG(DEBUG,
                   "Lorenzo: outlier_capacity=0, silently dropped %u outlier(s) "
                   "(%.1f%% of data). Reconstruction error for these elements "
                   "may exceed the error bound.",
                   h_outlier_count,
                   100.0f * h_outlier_count / static_cast<float>(num_elements_));
        } else {
            // Non-zero capacity but still overflowed — unexpected, warn loudly.
            float actual_pct   = 100.0f * h_outlier_count
                                 / static_cast<float>(num_elements_);
            float capacity_pct = 100.0f * max_outliers
                                 / static_cast<float>(num_elements_);
            FZ_LOG(WARN,
                   "Lorenzo outlier overflow! Detected %u (%.1f%%) outliers but "
                   "only %.1f%% capacity allocated. Outliers beyond capacity were "
                   "DROPPED — data will be corrupted for those elements. "
                   "Increase outlier_capacity to at least %.1f%%.",
                   h_outlier_count, actual_pct, capacity_pct, actual_pct * 1.1f);
        }
        h_outlier_count = static_cast<uint32_t>(max_outliers);
    }

    actual_outlier_count_      = h_outlier_count;
    actual_output_sizes_[1]    = h_outlier_count * sizeof(TInput);
    actual_output_sizes_[2]    = h_outlier_count * sizeof(uint32_t);
    // [0] codes and [3] outlier_count are already correct from execute()
}

template<typename TInput, typename TCode>
std::vector<size_t> LorenzoStage<TInput, TCode>::estimateOutputSizes(
    const std::vector<size_t>& input_sizes
) const {
    size_t input_size = input_sizes[0];
    size_t num_elements = input_size / sizeof(TInput);
    size_t max_outliers = getMaxOutlierCount(num_elements);
    
    return {
        num_elements * sizeof(TCode),        // codes (fixed size)
        max_outliers * sizeof(TInput),       // outlier_errors (max capacity)
        max_outliers * sizeof(uint32_t),     // outlier_indices (max capacity)
        sizeof(uint32_t)                      // outlier_count (fixed)
    };
}

// ========== Explicit Template Instantiations ==========

// Instantiate classes
template class LorenzoStage<float, uint16_t>;
template class LorenzoStage<float, uint8_t>;
template class LorenzoStage<double, uint16_t>;
template class LorenzoStage<double, uint32_t>;

// Instantiate kernel launchers
template void launchLorenzoKernel<float, uint16_t>(
    const float*, size_t, float, uint16_t,
    uint16_t*, float*, uint32_t*, uint32_t*, size_t, int, bool, cudaStream_t);
template void launchLorenzoKernel<float, uint8_t>(
    const float*, size_t, float, uint8_t,
    uint8_t*, float*, uint32_t*, uint32_t*, size_t, int, bool, cudaStream_t);
template void launchLorenzoKernel<double, uint16_t>(
    const double*, size_t, double, uint16_t,
    uint16_t*, double*, uint32_t*, uint32_t*, size_t, int, bool, cudaStream_t);
template void launchLorenzoKernel<double, uint32_t>(
    const double*, size_t, double, uint32_t,
    uint32_t*, double*, uint32_t*, uint32_t*, size_t, int, bool, cudaStream_t);

template void launchLorenzoInverseKernel<float, uint16_t>(
    const uint16_t*, const float*, const uint32_t*, const uint32_t*,
    size_t, size_t, float, uint16_t, float*, bool, cudaStream_t, MemoryPool*);
template void launchLorenzoInverseKernel<float, uint8_t>(
    const uint8_t*, const float*, const uint32_t*, const uint32_t*,
    size_t, size_t, float, uint8_t, float*, bool, cudaStream_t, MemoryPool*);
template void launchLorenzoInverseKernel<double, uint16_t>(
    const uint16_t*, const double*, const uint32_t*, const uint32_t*,
    size_t, size_t, double, uint16_t, double*, bool, cudaStream_t, MemoryPool*);
template void launchLorenzoInverseKernel<double, uint32_t>(
    const uint32_t*, const double*, const uint32_t*, const uint32_t*,
    size_t, size_t, double, uint32_t, double*, bool, cudaStream_t, MemoryPool*);

// ========== 2-D Compression Kernel ==========
// Adapted from cuSZ KERNEL_CUHIP_c_lorenzo_2d__32x32.
// TileDim=32(x), Yseq=8, NumWarps=4(y).
// blockDim=(32,4,1), gridDim=(ceil(nx/32), ceil(ny/32), 1).
// Outlier format: separate errors/indices/count arrays (no ZigZag).

template<typename TInput, typename TCode, bool ZigzagCodes = false>
__global__ void lorenzo_quantize_2d_kernel(
    const TInput* __restrict__ in_data,
    uint32_t data_lenx, uint32_t data_leny, uint32_t data_leapy,
    TInput ebx2_r, TCode quant_radius,
    TCode*    __restrict__ out_codes,
    TInput*   __restrict__ out_outlier_errors,
    uint32_t* __restrict__ out_outlier_indices,
    uint32_t* __restrict__ out_outlier_count,
    size_t max_outliers
) {
    constexpr int TileDim  = 32;
    constexpr int Yseq     = 8;
    constexpr int NumWarps = 4;   // BDY = 4, each warp covers Yseq rows

    // Inter-warp y-boundary exchange: warp w provides its last row to warp w+1.
    __shared__ TInput exchange[NumWarps - 1][TileDim + 1];

    // center[0] = value from preceding y-thread (or 0 for warp 0)
    // center[1..Yseq] = this thread's Yseq rows
    TInput center[Yseq + 1] = {0};

    const auto gix      = blockIdx.x * TileDim + threadIdx.x;
    const auto giy_base = blockIdx.y * TileDim + threadIdx.y * Yseq;
    auto g_id = [&](int i) -> uint32_t { return (giy_base + i) * data_leapy + gix; };

    // Load + pre-quantize
#pragma unroll
    for (int iy = 0; iy < Yseq; iy++) {
        if (gix < data_lenx && (giy_base + iy) < data_leny)
            center[iy + 1] = round(in_data[g_id(iy)] * ebx2_r);
    }

    // Cross-warp boundary: pass last row down to next warp
    if (threadIdx.y < NumWarps - 1)
        exchange[threadIdx.y][threadIdx.x] = center[Yseq];
    __syncthreads();
    if (threadIdx.y > 0)
        center[0] = exchange[threadIdx.y - 1][threadIdx.x];
    __syncthreads();

    // Lorenzo 2D prediction + quantize (iterate y in reverse so center[i-1] is intact)
#pragma unroll
    for (int i = Yseq; i > 0; i--) {
        // y-direction: subtract previous row
        center[i] -= center[i - 1];
        // x-direction: subtract west neighbour via full-warp shuffle
        TInput west = __shfl_up_sync(0xffffffff, center[i], 1, 32);
        if (threadIdx.x > 0) center[i] -= west;

        auto gid       = g_id(i - 1);
        bool is_valid  = (gix < data_lenx && (giy_base + i - 1) < data_leny);

        if (is_valid) {
            bool quantizable = fabsf(center[i]) < static_cast<TInput>(quant_radius);
            if (quantizable) {
                if constexpr (ZigzagCodes) {
                    using SCode = typename std::make_signed<TCode>::type;
                    out_codes[gid] = static_cast<TCode>(
                        Zigzag<SCode>::encode(static_cast<SCode>(static_cast<int>(center[i]))));
                } else {
                    out_codes[gid] = static_cast<TCode>(static_cast<int>(center[i]));
                }
            } else {
                out_codes[gid] = static_cast<TCode>(0);   // 0 contributes 0 to prefix sum
            }

            if (!quantizable) {
                uint32_t cur_idx = atomicAdd(out_outlier_count, 1u);
                if (cur_idx < max_outliers) {
                    out_outlier_errors[cur_idx]  = center[i];   // pre-quantized delta
                    out_outlier_indices[cur_idx] = gid;
                }
            }
        }
    }
}

// ========== 2-D Decompression Kernel ==========
// Adapted from cuSZ KERNEL_CUHIP_x_lorenzo_2d__32x32.
// Outliers are pre-scattered into inout_data before this kernel is called.
// blockDim=(32,4,1), gridDim=(ceil(nx/32), ceil(ny/32), 1).

template<typename TInput, typename TCode, bool ZigzagCodes = false>
__global__ void lorenzo_dequantize_2d_kernel(
    const TCode* __restrict__ in_codes,
    TInput*      __restrict__ inout_data,   // holds scattered outlier deltas on entry, output on exit
    uint32_t data_lenx, uint32_t data_leny, uint32_t data_leapy,
    TCode quant_radius, TInput ebx2
) {
    constexpr int TileDim  = 32;
    constexpr int NumWarps = 4;
    constexpr int Yseq     = TileDim / NumWarps;   // 8

    __shared__ TInput scratch[NumWarps - 1][TileDim + 1];

    TInput thp_data[Yseq] = {0};

    const auto gix      = blockIdx.x * TileDim + threadIdx.x;
    const auto giy_base = blockIdx.y * TileDim + threadIdx.y * Yseq;
    auto get_gid = [&](int i) -> uint32_t { return (giy_base + i) * data_leapy + gix; };

    // Load: fuse scattered outlier + (code - radius)
#pragma unroll
    for (int i = 0; i < Yseq; i++) {
        auto gid = get_gid(i);
        if (gix < data_lenx && (giy_base + i) < data_leny) {
            using SCode = typename std::make_signed<TCode>::type;
            TInput decoded;
            if constexpr (ZigzagCodes) {
                decoded = static_cast<TInput>(Zigzag<SCode>::decode(in_codes[gid]));
            } else {
                decoded = static_cast<TInput>(static_cast<SCode>(in_codes[gid]));
            }
            thp_data[i] = inout_data[gid] + decoded;
        }
    }

    // Partial-sum along y sequentially (restores y-Lorenzo)
    for (int i = 1; i < Yseq; i++) thp_data[i] += thp_data[i - 1];

    // Cross-warp y scan (exclusive prefix across the 4 warps)
    if (threadIdx.y < NumWarps - 1)
        scratch[threadIdx.y][threadIdx.x] = thp_data[Yseq - 1];
    __syncthreads();

    if (threadIdx.y == 0) {
        TInput warp_accum[NumWarps - 1];
#pragma unroll
        for (int i = 0; i < NumWarps - 1; i++) warp_accum[i] = scratch[i][threadIdx.x];
#pragma unroll
        for (int i = 1; i < NumWarps - 1; i++) warp_accum[i] += warp_accum[i - 1];
#pragma unroll
        for (int i = 1; i < NumWarps - 1; i++) scratch[i][threadIdx.x] = warp_accum[i];
    }
    __syncthreads();

    if (threadIdx.y > 0) {
        TInput addend = scratch[threadIdx.y - 1][threadIdx.x];
#pragma unroll
        for (int i = 0; i < Yseq; i++) thp_data[i] += addend;
    }
    __syncthreads();

    // x-axis partial sum via full-warp shuffle, then scale by ebx2
#pragma unroll
    for (int i = 0; i < Yseq; i++) {
        for (int d = 1; d < TileDim; d *= 2) {
            TInput n = __shfl_up_sync(0xffffffff, thp_data[i], d, 32);
            if (threadIdx.x >= d) thp_data[i] += n;
        }
        thp_data[i] *= ebx2;
    }

    // Write output
#pragma unroll
    for (int i = 0; i < Yseq; i++) {
        auto gid = get_gid(i);
        if (gix < data_lenx && (giy_base + i) < data_leny)
            inout_data[gid] = thp_data[i];
    }
}

// ========== 3-D Compression Kernel ==========
// Adapted from cuSZ KERNEL_CUHIP_c_lorenzo_3d.
// blockDim=(32,8,1), gridDim=(ceil(nx/(4*8)), ceil(ny/8), ceil(nz/8)).
// threadIdx.x ∈ [0,32): covers 4 segments of 8 in the x-dimension.
// threadIdx.y ∈ [0,8): covers one TileDim slice in y.
// z is iterated sequentially per thread (TileDim=8 z-slices per block).

template<typename TInput, typename TCode, bool ZigzagCodes = false>
__global__ void lorenzo_quantize_3d_kernel(
    const TInput* __restrict__ in_data,
    uint32_t data_lenx, uint32_t data_leny, uint32_t data_leapy,
    uint32_t data_lenz, uint32_t data_leapz,
    TInput ebx2_r, TCode quant_radius,
    TCode*    __restrict__ out_codes,
    TInput*   __restrict__ out_outlier_errors,
    uint32_t* __restrict__ out_outlier_indices,
    uint32_t* __restrict__ out_outlier_count,
    size_t max_outliers
) {
    constexpr int TileDim = 8;

    // s[threadIdx.y + 1][threadIdx.x]: ghost row at index 0 for y-direction diff
    __shared__ TInput s[9][33];

    TInput delta[TileDim + 1] = {0};   // delta[0] = ghost (0), delta[1..TileDim] = slices

    const auto gix      = blockIdx.x * (TileDim * 4) + threadIdx.x;
    const auto giy      = blockIdx.y * TileDim + threadIdx.y;
    const auto giz_base = blockIdx.z * TileDim;
    const auto base_id  = gix + giy * data_leapy + giz_base * data_leapz;

    auto giz = [&](int z) { return giz_base + z; };
    auto gid = [&](int z) -> uint32_t { return base_id + z * data_leapz; };

    // Load + pre-quantize along z
    if (gix < data_lenx && giy < data_leny) {
        for (int z = 0; z < TileDim; z++)
            if ((uint32_t)giz(z) < data_lenz)
                delta[z + 1] = round(in_data[gid(z)] * ebx2_r);
    }
    __syncthreads();

    for (int z = TileDim; z > 0; z--) {
        // z-direction: subtract previous z-slice
        delta[z] -= delta[z - 1];

        // x-direction: subtract west neighbour within 8-thread segment
        auto seg_tix = threadIdx.x % TileDim;
        TInput prev_x = __shfl_up_sync(0xffffffff, delta[z], 1, 8);
        if (seg_tix > 0) delta[z] -= prev_x;

        // y-direction: exchange via shared memory (ghost row at index 0)
        s[threadIdx.y + 1][threadIdx.x] = delta[z];
        __syncthreads();
        delta[z] -= (threadIdx.y > 0) * s[threadIdx.y][threadIdx.x];

        // Quantize and write
        bool is_valid = (gix < data_lenx && giy < data_leny && (uint32_t)giz(z - 1) < data_lenz);
        if (is_valid) {
            bool quantizable = fabsf(delta[z]) < static_cast<TInput>(quant_radius);
            if (quantizable) {
                if constexpr (ZigzagCodes) {
                    using SCode = typename std::make_signed<TCode>::type;
                    out_codes[gid(z - 1)] = static_cast<TCode>(
                        Zigzag<SCode>::encode(static_cast<SCode>(static_cast<int>(delta[z]))));
                } else {
                    out_codes[gid(z - 1)] = static_cast<TCode>(static_cast<int>(delta[z]));
                }
            } else {
                out_codes[gid(z - 1)] = static_cast<TCode>(0);   // 0 contributes 0 to prefix sum
            }

            if (!quantizable) {
                uint32_t cur_idx = atomicAdd(out_outlier_count, 1u);
                if (cur_idx < max_outliers) {
                    out_outlier_errors[cur_idx]  = delta[z];   // pre-quantized delta
                    out_outlier_indices[cur_idx] = gid(z - 1);
                }
            }
        }
        __syncthreads();
    }
}

// ========== 3-D Decompression Kernel ==========
// Adapted from cuSZ KERNEL_CUHIP_x_lorenzo_3d.
// Outliers are pre-scattered into inout_data before this kernel is called.
// blockDim=(32,1,8), gridDim=(ceil(nx/(4*8)), ceil(ny/8), ceil(nz/8)).

template<typename TInput, typename TCode, bool ZigzagCodes = false>
__global__ void lorenzo_dequantize_3d_kernel(
    const TCode* __restrict__ in_codes,
    TInput*      __restrict__ inout_data,
    uint32_t data_lenx, uint32_t data_leny, uint32_t data_leapy,
    uint32_t data_lenz, uint32_t data_leapz,
    TCode quant_radius, TInput ebx2
) {
    constexpr int TileDim = 8;
    constexpr int Yseq    = TileDim;

    // scratch[TileDim][4][8] — used for x-z transpose during 3D partial sum
    __shared__ TInput scratch[TileDim][4][8];

    TInput thread_private[Yseq] = {0};

    const auto seg_id  = threadIdx.x / 8;
    const auto seg_tix = threadIdx.x % 8;

    const auto gix      = blockIdx.x * (4 * TileDim) + threadIdx.x;
    const auto giy_base = blockIdx.y * TileDim;
    const auto giz      = blockIdx.z * TileDim + threadIdx.z;

    auto giy = [&](int y) { return giy_base + y; };
    auto gid = [&](int y) -> uint32_t {
        return giz * data_leapz + (giy_base + y) * data_leapy + gix;
    };

    // Load: fuse scattered outlier + (code - radius)
#pragma unroll
    for (int y = 0; y < Yseq; y++) {
        if (gix < data_lenx && (uint32_t)giy(y) < data_leny && giz < data_lenz) {
            using SCode = typename std::make_signed<TCode>::type;
            TInput decoded;
            if constexpr (ZigzagCodes) {
                decoded = static_cast<TInput>(Zigzag<SCode>::decode(in_codes[gid(y)]));
            } else {
                decoded = static_cast<TInput>(static_cast<SCode>(in_codes[gid(y)]));
            }
            thread_private[y] = inout_data[gid(y)] + decoded;
        }
    }

    // Partial-sum along y sequentially (restores y-Lorenzo)
    for (int y = 1; y < Yseq; y++) thread_private[y] += thread_private[y - 1];

    // x and z partial sums via warp shuffle + shared-memory transpose
#pragma unroll
    for (int i = 0; i < TileDim; i++) {
        TInput val = thread_private[i];

        // x partial sum within 8-thread segment
        for (int dist = 1; dist < TileDim; dist *= 2) {
            TInput addend = __shfl_up_sync(0xffffffff, val, dist, 8);
            if (seg_tix >= dist) val += addend;
        }

        // Transpose x <-> z through shared memory
        scratch[threadIdx.z][seg_id][seg_tix] = val;
        __syncthreads();
        val = scratch[seg_tix][seg_id][threadIdx.z];
        __syncthreads();

        // z partial sum within 8-thread segment
        for (int dist = 1; dist < TileDim; dist *= 2) {
            TInput addend = __shfl_up_sync(0xffffffff, val, dist, 8);
            if (seg_tix >= dist) val += addend;
        }

        // Transpose back z <-> x
        scratch[threadIdx.z][seg_id][seg_tix] = val;
        __syncthreads();
        val = scratch[seg_tix][seg_id][threadIdx.z];
        __syncthreads();

        thread_private[i] = val;
    }

    // Write: scale by ebx2 and store
#pragma unroll
    for (int y = 0; y < Yseq; y++)
        if (gix < data_lenx && (uint32_t)giy(y) < data_leny && giz < data_lenz)
            inout_data[gid(y)] = thread_private[y] * ebx2;
}

// ========== 2-D Kernel Launchers ==========

template<typename TInput, typename TCode>
void launchLorenzoKernel2D(
    const TInput* d_input, size_t nx, size_t ny,
    TInput ebx2_r, TCode quant_radius,
    TCode* d_codes, TInput* d_outlier_errors,
    uint32_t* d_outlier_indices, uint32_t* d_outlier_count,
    size_t max_outliers,
    bool zigzag_codes,
    cudaStream_t stream
) {
    constexpr int TileDim = 32;
    constexpr int BDY     = 4;    // NumWarps in y

    dim3 block(TileDim, BDY, 1);
    dim3 grid(
        (static_cast<uint32_t>(nx) + TileDim - 1) / TileDim,
        (static_cast<uint32_t>(ny) + TileDim - 1) / TileDim,
        1
    );
    uint32_t leapy = static_cast<uint32_t>(nx);

    if (zigzag_codes) {
        lorenzo_quantize_2d_kernel<TInput, TCode, true><<<grid, block, 0, stream>>>(
            d_input,
            static_cast<uint32_t>(nx), static_cast<uint32_t>(ny), leapy,
            ebx2_r, quant_radius,
            d_codes, d_outlier_errors, d_outlier_indices, d_outlier_count,
            max_outliers
        );
    } else {
        lorenzo_quantize_2d_kernel<TInput, TCode, false><<<grid, block, 0, stream>>>(
            d_input,
            static_cast<uint32_t>(nx), static_cast<uint32_t>(ny), leapy,
            ebx2_r, quant_radius,
            d_codes, d_outlier_errors, d_outlier_indices, d_outlier_count,
            max_outliers
        );
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(
            std::string("lorenzo_quantize_2d_kernel launch failed: ") + cudaGetErrorString(err));
}

template<typename TInput, typename TCode>
void launchLorenzoInverseKernel2D(
    const TCode* d_codes,
    const TInput* d_outlier_errors, const uint32_t* d_outlier_indices,
    const uint32_t* d_outlier_count,
    size_t nx, size_t ny, size_t max_outliers,
    TInput ebx2, TCode quant_radius,
    TInput* d_output,
    bool zigzag_codes,
    cudaStream_t stream, MemoryPool* pool
) {
    (void)pool;

    const size_t total = nx * ny;
    constexpr int TileDim = 32;
    constexpr int BDY     = 4;

    // Step 0: zero output (scatter target)
    FZ_CUDA_CHECK(cudaMemsetAsync(d_output, 0, total * sizeof(TInput), stream));

    // Step 1: scatter outlier prediction errors into output
    if (d_outlier_count != nullptr && max_outliers > 0) {
        int sblk  = 256;
        int sgrid = (static_cast<int>(max_outliers) + sblk - 1) / sblk;
        scatter_outliers_kernel<TInput><<<sgrid, sblk, 0, stream>>>(
            d_outlier_errors, d_outlier_indices, d_outlier_count, d_output);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            throw std::runtime_error(
                std::string("scatter_outliers_kernel (2D) launch failed: ") +
                cudaGetErrorString(err));
    }

    // Step 2: 2D inverse Lorenzo (dequantize + 2D prefix sum)
    dim3 block(TileDim, BDY, 1);
    dim3 grid(
        (static_cast<uint32_t>(nx) + TileDim - 1) / TileDim,
        (static_cast<uint32_t>(ny) + TileDim - 1) / TileDim,
        1
    );
    uint32_t leapy = static_cast<uint32_t>(nx);

    if (zigzag_codes) {
        lorenzo_dequantize_2d_kernel<TInput, TCode, true><<<grid, block, 0, stream>>>(
            d_codes, d_output,
            static_cast<uint32_t>(nx), static_cast<uint32_t>(ny), leapy,
            quant_radius, ebx2
        );
    } else {
        lorenzo_dequantize_2d_kernel<TInput, TCode, false><<<grid, block, 0, stream>>>(
            d_codes, d_output,
            static_cast<uint32_t>(nx), static_cast<uint32_t>(ny), leapy,
            quant_radius, ebx2
        );
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(
            std::string("lorenzo_dequantize_2d_kernel launch failed: ") + cudaGetErrorString(err));
}

// ========== 3-D Kernel Launchers ==========

template<typename TInput, typename TCode>
void launchLorenzoKernel3D(
    const TInput* d_input, size_t nx, size_t ny, size_t nz,
    TInput ebx2_r, TCode quant_radius,
    TCode* d_codes, TInput* d_outlier_errors,
    uint32_t* d_outlier_indices, uint32_t* d_outlier_count,
    size_t max_outliers,
    bool zigzag_codes,
    cudaStream_t stream
) {
    constexpr int TileDim = 8;

    // blockDim=(32,8,1): 32 x-threads covering 4 x-segments, 8 y-threads per block
    dim3 block(32, TileDim, 1);
    dim3 grid(
        (static_cast<uint32_t>(nx) + (4 * TileDim) - 1) / (4 * TileDim),
        (static_cast<uint32_t>(ny) + TileDim - 1) / TileDim,
        (static_cast<uint32_t>(nz) + TileDim - 1) / TileDim
    );
    uint32_t leapy = static_cast<uint32_t>(nx);
    uint32_t leapz = static_cast<uint32_t>(nx) * static_cast<uint32_t>(ny);

    if (zigzag_codes) {
        lorenzo_quantize_3d_kernel<TInput, TCode, true><<<grid, block, 0, stream>>>(
            d_input,
            static_cast<uint32_t>(nx), static_cast<uint32_t>(ny), leapy,
            static_cast<uint32_t>(nz), leapz,
            ebx2_r, quant_radius,
            d_codes, d_outlier_errors, d_outlier_indices, d_outlier_count,
            max_outliers
        );
    } else {
        lorenzo_quantize_3d_kernel<TInput, TCode, false><<<grid, block, 0, stream>>>(
            d_input,
            static_cast<uint32_t>(nx), static_cast<uint32_t>(ny), leapy,
            static_cast<uint32_t>(nz), leapz,
            ebx2_r, quant_radius,
            d_codes, d_outlier_errors, d_outlier_indices, d_outlier_count,
            max_outliers
        );
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(
            std::string("lorenzo_quantize_3d_kernel launch failed: ") + cudaGetErrorString(err));
}

template<typename TInput, typename TCode>
void launchLorenzoInverseKernel3D(
    const TCode* d_codes,
    const TInput* d_outlier_errors, const uint32_t* d_outlier_indices,
    const uint32_t* d_outlier_count,
    size_t nx, size_t ny, size_t nz, size_t max_outliers,
    TInput ebx2, TCode quant_radius,
    TInput* d_output,
    bool zigzag_codes,
    cudaStream_t stream, MemoryPool* pool
) {
    (void)pool;

    const size_t total = nx * ny * nz;
    constexpr int TileDim = 8;

    // Step 0: zero output (scatter target)
    FZ_CUDA_CHECK(cudaMemsetAsync(d_output, 0, total * sizeof(TInput), stream));

    // Step 1: scatter outlier prediction errors into output
    if (d_outlier_count != nullptr && max_outliers > 0) {
        int sblk  = 256;
        int sgrid = (static_cast<int>(max_outliers) + sblk - 1) / sblk;
        scatter_outliers_kernel<TInput><<<sgrid, sblk, 0, stream>>>(
            d_outlier_errors, d_outlier_indices, d_outlier_count, d_output);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            throw std::runtime_error(
                std::string("scatter_outliers_kernel (3D) launch failed: ") +
                cudaGetErrorString(err));
    }

    // Step 2: 3D inverse Lorenzo (dequantize + 3D partial sum)
    // blockDim=(32,1,8): 32 x-threads covering 4 x-segments, 8 z-threads for z scan
    dim3 block(32, 1, TileDim);
    dim3 grid(
        (static_cast<uint32_t>(nx) + (4 * TileDim) - 1) / (4 * TileDim),
        (static_cast<uint32_t>(ny) + TileDim - 1) / TileDim,
        (static_cast<uint32_t>(nz) + TileDim - 1) / TileDim
    );
    uint32_t leapy = static_cast<uint32_t>(nx);
    uint32_t leapz = static_cast<uint32_t>(nx) * static_cast<uint32_t>(ny);

    if (zigzag_codes) {
        lorenzo_dequantize_3d_kernel<TInput, TCode, true><<<grid, block, 0, stream>>>(
            d_codes, d_output,
            static_cast<uint32_t>(nx), static_cast<uint32_t>(ny), leapy,
            static_cast<uint32_t>(nz), leapz,
            quant_radius, ebx2
        );
    } else {
        lorenzo_dequantize_3d_kernel<TInput, TCode, false><<<grid, block, 0, stream>>>(
            d_codes, d_output,
            static_cast<uint32_t>(nx), static_cast<uint32_t>(ny), leapy,
            static_cast<uint32_t>(nz), leapz,
            quant_radius, ebx2
        );
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(
            std::string("lorenzo_dequantize_3d_kernel launch failed: ") + cudaGetErrorString(err));
}

// Explicit instantiations for 2-D / 3-D stubs (same type combos as 1-D)
#define INSTANTIATE_LORENZO_ND(TInput, TCode) \
    template void launchLorenzoKernel2D<TInput, TCode>( \
        const TInput*, size_t, size_t, TInput, TCode, \
        TCode*, TInput*, uint32_t*, uint32_t*, size_t, bool, cudaStream_t); \
    template void launchLorenzoInverseKernel2D<TInput, TCode>( \
        const TCode*, const TInput*, const uint32_t*, const uint32_t*, \
        size_t, size_t, size_t, TInput, TCode, TInput*, bool, cudaStream_t, MemoryPool*); \
    template void launchLorenzoKernel3D<TInput, TCode>( \
        const TInput*, size_t, size_t, size_t, TInput, TCode, \
        TCode*, TInput*, uint32_t*, uint32_t*, size_t, bool, cudaStream_t); \
    template void launchLorenzoInverseKernel3D<TInput, TCode>( \
        const TCode*, const TInput*, const uint32_t*, const uint32_t*, \
        size_t, size_t, size_t, size_t, TInput, TCode, TInput*, bool, cudaStream_t, MemoryPool*);

INSTANTIATE_LORENZO_ND(float,  uint16_t)
INSTANTIATE_LORENZO_ND(float,  uint8_t)
INSTANTIATE_LORENZO_ND(double, uint16_t)
INSTANTIATE_LORENZO_ND(double, uint32_t)

#undef INSTANTIATE_LORENZO_ND

} // namespace fz
