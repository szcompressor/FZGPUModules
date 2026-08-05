// Algorithm adapted from the cuSZ Lorenzo implementation (cuSZ team, BSD-3-Clause).
// Upstream: https://github.com/szcompressor/cuSZ — see THIRD_PARTY.md.
#include "backend/warp.h"
#include "fused/lorenzo_quant/lorenzo_quant.h"
#include "predictors/predictor_utils.cuh"
#include "transforms/zigzag/zigzag.h"
#include "backend/api.h"
#include <cstdint>
#include <algorithm>
#include <cstdio>
#include <type_traits>
#include "mem/mempool.h"
#include "cuda_check.h"
#include "log.h"
#include "fused/lorenzo_quant/lorenzo_quant_kernels.cuh"

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
        T val = fz::backend::shflUp(thread_sum, offset, 32);
        if (lane_id >= offset) {
            thread_sum += val;
        }
    }
    
    // Get prefix sum from previous thread in warp
    T prefix = fz::backend::shflUp(thread_sum, 1, 32);
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
        
        // Mask for only the participating threads (lanes 0..NumWarps-1, always
        // within the lower 32-lane software-warp — warp_id==0 covers
        // threadIdx.x 0..31 unambiguously). Explicit width=32: unlike CUDA,
        // HIP's __shfl_up_sync width defaults to warpSize (64 on MI100), so
        // an unspecified width would pull from outside this software-warp.
        fz::backend::warp_mask_t mask =
            (static_cast<fz::backend::warp_mask_t>(1) << NumWarps) - 1;  // e.g., 0xFF for 8 warps

        #pragma unroll
        for (int offset = 1; offset < NumWarps; offset *= 2) {
            T tmp = __shfl_up_sync(mask, val, offset, 32);
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
    TInput* __restrict__ output,
    const TInput* __restrict__ means
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

            // Undo centering by seeding the tile's chain with mu; the prefix sum
            // carries it into every element, so this is one add per tile.
            if (means != nullptr && local_id == 0) delta += means[blockIdx.x];

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
    const size_t max_outliers,
    TInput* __restrict__ means
) {
    constexpr int NumThreads = TileDim / Seq;

    // Shared memory for staging and better memory access patterns
    __shared__ TInput s_data[TileDim];
    __shared__ TCode s_codes[TileDim];
    __shared__ TInput s_red[NumThreads];
    __shared__ TInput s_mu;

    // Private thread data: current Seq elements + 1 previous element
    TInput thp_data[Seq + 1];

    const size_t block_offset = blockIdx.x * TileDim;

    // ========== Stage 1: Load input data to shared memory with pre-quantization ==========
    // Coalesced loads: each thread loads Seq elements
    TInput thread_sum = static_cast<TInput>(0);
    #pragma unroll
    for (int i = 0; i < Seq; i++) {
        size_t global_idx = block_offset + threadIdx.x + i * NumThreads;
        if (global_idx < n) {
            // Pre-quantize: multiply by 1/(2*eb) to normalize to quantization units
            TInput q = round(input[global_idx] * ebx2_r);
            s_data[threadIdx.x + i * NumThreads] = q;
            thread_sum += q;
        }
    }
    __syncthreads();

    // ========== Stage 1b: Per-tile mean (adaptive centering) ==========
    // Accumulated in the load loop above rather than re-read from s_data: the
    // tail of a partial tile is never written, so summing s_data would read
    // uninitialized shared memory.
    if (means != nullptr) {
        s_red[threadIdx.x] = thread_sum;
        __syncthreads();
        for (unsigned stride = NumThreads >> 1; stride > 0u; stride >>= 1) {
            if (threadIdx.x < stride) s_red[threadIdx.x] += s_red[threadIdx.x + stride];
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            const TInput count =
                static_cast<TInput>(min(static_cast<size_t>(TileDim), n - block_offset));
            // mu stays an exact integer in quantized units so the inverse is lossless.
            s_mu = round(s_red[0] / count);
            means[blockIdx.x] = s_mu;
        }
        __syncthreads();
    }

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
        // Centering enters as the chain seed: predicting element 0 from mu
        // instead of 0 makes its residual (q_0 - mu) rather than the raw q_0,
        // and leaves every other residual — and the outlier path — untouched.
        thp_data[0] = (means != nullptr) ? s_mu : static_cast<TInput>(0);
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
    cudaStream_t stream,
    TInput* means
) {
    constexpr int TileDim = 1024;
    constexpr int Seq = 4;
    constexpr int NumThreads = TileDim / Seq;  // 256

    if (zigzag_codes) {
        lorenzo_quantize_1d_kernel<TInput, TCode, TileDim, Seq, true>
            <<<grid_size, NumThreads, 0, stream>>>(
                input, n, ebx2_r, quant_radius,
                quant_codes, outlier_errors, outlier_indices, outlier_count,
                max_outliers, means
            );
    } else {
        lorenzo_quantize_1d_kernel<TInput, TCode, TileDim, Seq, false>
            <<<grid_size, NumThreads, 0, stream>>>(
                input, n, ebx2_r, quant_radius,
                quant_codes, outlier_errors, outlier_indices, outlier_count,
                max_outliers, means
            );
    }
}

template<typename TInput, typename TCode>
void launchLorenzoInverseKernel(
    const TCode* quant_codes,
    const TInput* outlier_errors,
    const uint32_t* outlier_indices,
    uint32_t outlier_n,
    size_t n,
    TInput ebx2,
    TCode quant_radius,
    TInput* output,
    bool zigzag_codes,
    cudaStream_t stream,
    MemoryPool* pool,
    const TInput* means
) {
    constexpr int TileDim = 1024;
    constexpr int Seq = 4;
    constexpr int NumThreads = TileDim / Seq;  // 256

    const int grid_size = (n + TileDim - 1) / TileDim;

    // Step 0: Initialize output array to 0 because we will scatter outliers into it
    FZ_CUDA_CHECK(cudaMemsetAsync(output, 0, n * sizeof(TInput), stream));

    // Step 1: Scatter outliers BEFORE prefix sum.  The count is a register
    // arg (came from the deserialized header) — no device pointer deref.
    if (outlier_n > 0) {
        int scatter_block_size = 256;
        int scatter_grid_size  = (static_cast<int>(outlier_n) + scatter_block_size - 1)
                                 / scatter_block_size;

        scatter_outliers_kernel<TInput>
            <<<scatter_grid_size, scatter_block_size, 0, stream>>>(
                outlier_errors, outlier_indices, outlier_n, output
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
                quant_codes, n, ebx2, quant_radius, output, means
            );
    } else {
        lorenzo_dequantize_1d_kernel<TInput, TCode, TileDim, Seq, false>
            <<<grid_size, NumThreads, 0, stream>>>(
                quant_codes, n, ebx2, quant_radius, output, means
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

// ========== LorenzoQuantStage Implementation ==========

template<typename TInput, typename TCode>
LorenzoQuantStage<TInput, TCode>::LorenzoQuantStage(const Config& config)
    : config_(config),
      computed_abs_eb_(static_cast<TInput>(config.error_bound)),
      computed_value_base_(config.precomputed_value_base) {
    actual_output_sizes_.resize(config_.centering ? 4 : 3, 0);
}

template<typename TInput, typename TCode>
LorenzoQuantStage<TInput, TCode>::~LorenzoQuantStage()
{
    if (persistent_pool_ != nullptr && d_outlier_count_scratch_ != nullptr) {
        persistent_pool_->freePersistentDevice(d_outlier_count_scratch_);
    }
    d_outlier_count_scratch_ = nullptr;
    persistent_pool_         = nullptr;
}

template<typename TInput, typename TCode>
void LorenzoQuantStage<TInput, TCode>::initOutlierCountScratch(MemoryPool* pool)
{
    if (d_outlier_count_scratch_ != nullptr) return;
    if (pool == nullptr) {
        throw std::runtime_error(
            "LorenzoQuantStage: outlier-count scratch requires a MemoryPool");
    }
    persistent_pool_ = pool;
    d_outlier_count_scratch_ = static_cast<uint32_t*>(
        pool->allocatePersistentDevice(sizeof(uint32_t), "lorenzo_outlier_count"));
}

template<typename TInput, typename TCode>
void LorenzoQuantStage<TInput, TCode>::onFinalize(
    size_t /*estimated_inlen*/, MemoryPool* pool)
{
    initOutlierCountScratch(pool);
}

template<typename TInput, typename TCode>
void LorenzoQuantStage<TInput, TCode>::execute(
    cudaStream_t stream,
    MemoryPool* pool,
    const std::vector<void*>& inputs,
    const std::vector<void*>& outputs,
    const std::vector<size_t>& sizes
) {
    if (is_inverse_) {
        // ===== DECOMPRESSION MODE: 3 inputs → 1 output =====
        if (inputs.size() < 3 || outputs.empty() || sizes.size() < 3) {
            throw std::runtime_error("LorenzoQuantStage (inverse): Requires 3 inputs and 1 output");
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

        // Outlier count comes from the deserialized FZM header (set by
        // deserializeHeader() into actual_outlier_count_), not from a port.
        uint32_t outlier_n = actual_outlier_count_;

        int eff_ndim = ndim();

        if (config_.centering && eff_ndim != 1)
            throw std::runtime_error(
                "LorenzoQuantStage: centering is 1-D only — the 2-D/3-D Lorenzo "
                "paths have no per-tile prediction chain to seed with a mean");


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
                outlier_n,
                nx, ny, nz,
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
                outlier_n,
                nx, ny,
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
                outlier_n,
                num_elements,
                ebx2,
                config_.quant_radius,
                static_cast<TInput*>(outputs[0]),
                config_.zigzag_codes,
                stream,
                pool,
                config_.centering ? static_cast<const TInput*>(inputs[3]) : nullptr
            );
        }

        // Check for launch errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("LorenzoQuantStage (inverse) kernel launch failed: ") +
                cudaGetErrorString(err)
            );
        }

        // Set output size
        actual_output_sizes_.resize(1);
        actual_output_sizes_[0] = num_elements * sizeof(TInput);
        
    } else {
        // ===== COMPRESSION MODE: 1 input → 3 outputs =====
        if (inputs.empty() || outputs.size() < 3 || sizes.empty()) {
            throw std::runtime_error("LorenzoQuantStage: Requires 1 input and 3 outputs");
        }

        size_t input_size  = sizes[0];
        size_t num_elements = input_size / sizeof(TInput);
        size_t max_outliers = getMaxOutlierCount(num_elements);

        // Store for header generation
        num_elements_ = num_elements;

        // The input buffer is rounded up to the LC chunk alignment (see
        // Pipeline::computeInputAlignment); the tail is zero-padding. The
        // NOA/REL value-range scan below must cover only the real elements —
        // otherwise a padding zero becomes the array minimum and, for
        // all-positive fields, NOA collapses to eb*max instead of
        // eb*(max-min), over-loosening the bound by ~min/range (E16). Use the
        // logical grid (dims) when set; guard falls back to the full length.
        size_t scan_N = config_.dims[0] * config_.dims[1] * config_.dims[2];
        if (scan_N == 0 || scan_N > num_elements) scan_N = num_elements;

        if (num_elements == 0) {
            // Empty input
            for (size_t i = 0; i < actual_output_sizes_.size(); i++) {
                actual_output_sizes_[i] = 0;
            }
            actual_outlier_count_ = 0;
            return;
        }

        // Zero the stage-private outlier-count scratch (kernel uses atomicAdd).
        // Lazy-allocate it here for MINIMAL mode (PREALLOCATE allocated it in onFinalize).
        initOutlierCountScratch(pool);
        FZ_CUDA_CHECK(cudaMemsetAsync(d_outlier_count_scratch_, 0,
                                       sizeof(uint32_t), stream));

        // ── Resolve absolute error bound ──────────────────────────────────────
        // For ABS mode: abs_eb = config_.error_bound (no scan needed).
        // For NOA/PREL modes: perform a stream-synchronising min/max scan first,
        //   then derive the absolute bound.  If the caller pre-computed
        //   value_base (> 0), skip the scan and use that value instead.
        // A Config-struct-supplied REL never passed through setErrorBoundMode(),
        // so catch it here too.
        config_.eb_mode = resolveApproxRelMode(config_.eb_mode, "LorenzoQuantStage");

        if (config_.eb_mode == ErrorBoundMode::ABS) {
            computed_abs_eb_    = static_cast<TInput>(config_.error_bound);
            computed_value_base_ = 0.0f;
        } else {
            float value_base = config_.precomputed_value_base;
            if (value_base <= 0.0f) {
                value_base = computeValueBase<TInput>(
                    static_cast<const TInput*>(inputs[0]),
                    scan_N, config_.eb_mode, stream, pool);
            }
            computed_value_base_ = value_base;

            if (value_base <= 0.0f) {
                FZ_LOG(WARN,
                    "LorenzoQuantStage: value_base is zero for %s mode "
                    "(constant or empty data?); falling back to ABS",
                    config_.eb_mode == ErrorBoundMode::NOA ? "NOA" : "PREL");
                computed_abs_eb_ = static_cast<TInput>(config_.error_bound);
            } else {
                computed_abs_eb_ = static_cast<TInput>(config_.error_bound)
                                   * static_cast<TInput>(value_base);
            }
            FZ_LOG(DEBUG,
                "LorenzoQuantStage %s: user_eb=%.6g value_base=%.6g -> abs_eb=%.6g",
                config_.eb_mode == ErrorBoundMode::NOA ? "NOA" : "PREL",
                static_cast<double>(config_.error_bound),
                static_cast<double>(value_base),
                static_cast<double>(computed_abs_eb_));
        }

        // Calculate ebx2_r = 1 / (2 * abs_error_bound) for pre-quantization
        TInput ebx2_r = static_cast<TInput>(1)
                        / (static_cast<TInput>(2) * computed_abs_eb_);

        int eff_ndim = ndim();

        if (config_.centering && eff_ndim != 1)
            throw std::runtime_error(
                "LorenzoQuantStage: centering is 1-D only — the 2-D/3-D Lorenzo "
                "paths have no per-tile prediction chain to seed with a mean");


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
                d_outlier_count_scratch_,
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
                d_outlier_count_scratch_,
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
                d_outlier_count_scratch_,
                max_outliers,
                grid_size,
                config_.zigzag_codes,
                stream,
                config_.centering ? static_cast<TInput*>(outputs[3]) : nullptr
            );
        }

        // Check for launch errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("LorenzoQuantStage kernel launch failed: ") +
                cudaGetErrorString(err)
            );
        }

        // Use max-capacity sizes for now; postStreamSync() will trim them
        // (and refresh actual_outlier_count_) once the stream is idle. We do
        // NOT zero actual_outlier_count_ here: this method may be invoked
        // during cudaStreamBeginCapture/EndCapture (graph recording), in
        // which case postStreamSync is not called and we'd lose the count
        // from the previous real compress.
        actual_output_sizes_[0] = num_elements * sizeof(TCode);
        actual_output_sizes_[1] = max_outliers * sizeof(TInput);
        actual_output_sizes_[2] = max_outliers * sizeof(uint32_t);
        if (config_.centering) {
            // One mean per 1024-element tile; unlike the outlier ports this is
            // a fixed size, so postStreamSync() never has to trim it.
            constexpr size_t kTileDim = 1024;
            actual_output_sizes_.resize(4);
            actual_output_sizes_[3] =
                ((num_elements + kTileDim - 1) / kTileDim) * sizeof(TInput);
        }
    }
}

template<typename TInput, typename TCode>
void LorenzoQuantStage<TInput, TCode>::postStreamSync(cudaStream_t stream) {
    // Only applies to compression mode and only when execute() has run.
    if (is_inverse_ || d_outlier_count_scratch_ == nullptr) return;

    // D2H on the caller's stream — stream-scoped sync (stalls only this thread).
    uint32_t h_outlier_count = 0;
    FZ_CUDA_CHECK(cudaMemcpyAsync(&h_outlier_count, d_outlier_count_scratch_,
                                   sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
    FZ_CUDA_CHECK(cudaStreamSynchronize(stream));

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
            // Non-zero capacity but still overflowed.  Dropping the excess breaks
            // the error bound this stage exists to guarantee, and the result is
            // indistinguishable from a correct one at the API — the caller gets a
            // stream that decodes cleanly into wrong values.  This used to be an
            // FZ_LOG(WARN), which is invisible at the default log level, so the
            // corruption was silent in practice.  Fail instead: a compressor that
            // cannot honour its bound must say so.
            //
            // outlier_capacity == 0 stays a warning-free drop above, because that
            // is an explicit opt-in to the lossy trade-off.
            const float actual_pct   = 100.0f * h_outlier_count
                                       / static_cast<float>(num_elements_);
            const float capacity_pct = 100.0f * max_outliers
                                       / static_cast<float>(num_elements_);
            char msg[512];
            std::snprintf(msg, sizeof(msg),
                   "LorenzoQuantStage: outlier overflow — %u of %zu elements (%.1f%%) "
                   "fell outside the quantizer radius, but outlier_capacity reserves "
                   "only %.1f%%. Dropping the excess would violate the error bound. "
                   "Raise outlier_capacity to at least %.2f, widen quant_radius, or "
                   "loosen the error bound. Set outlier_capacity = 0 to opt into "
                   "dropping outliers deliberately.",
                   h_outlier_count, num_elements_, actual_pct, capacity_pct,
                   std::min(1.0f, actual_pct * 1.1f / 100.0f));
            throw std::runtime_error(msg);
        }
        h_outlier_count = static_cast<uint32_t>(max_outliers);
    }

    actual_outlier_count_      = h_outlier_count;
    actual_output_sizes_[1]    = h_outlier_count * sizeof(TInput);
    actual_output_sizes_[2]    = h_outlier_count * sizeof(uint32_t);
    // [0] codes already correct from execute()
}

template<typename TInput, typename TCode>
std::vector<size_t> LorenzoQuantStage<TInput, TCode>::estimateOutputSizes(
    const std::vector<size_t>& input_sizes
) const {
    if (input_sizes.empty())
        return config_.centering ? std::vector<size_t>{0, 0, 0, 0}
                                 : std::vector<size_t>{0, 0, 0};

    if (is_inverse_) {
        // Inverse: inputs are {codes, outlier_errors, outlier_indices} and the
        // single output is the reconstructed field. Sizing this from
        // sizeof(TInput) — as the forward branch does — under-allocates the
        // output by sizeof(TInput)/sizeof(TCode) (2x for the usual
        // float/uint16 pair) and the inverse kernel's first memset runs off the
        // end of the buffer.
        //
        // This is latent whenever LorenzoQuant is the pipeline's *source*
        // stage, because buildInverseDAG() then overrides its result buffer
        // with the exact known uncompressed size. It only bites when something
        // sits upstream of it — e.g. LogTransformStage.
        const size_t num_elements = input_sizes[0] / sizeof(TCode);
        return {num_elements * sizeof(TInput)};
    }

    size_t input_size = input_sizes[0];
    size_t num_elements = input_size / sizeof(TInput);
    size_t max_outliers = getMaxOutlierCount(num_elements);

    std::vector<size_t> sizes = {
        num_elements * sizeof(TCode),        // codes (fixed size)
        max_outliers * sizeof(TInput),       // outlier_errors (max capacity)
        max_outliers * sizeof(uint32_t),     // outlier_indices (max capacity)
    };
    if (config_.centering) {
        constexpr size_t kTileDim = 1024;    // matches launchLorenzoKernel
        sizes.push_back(((num_elements + kTileDim - 1) / kTileDim) * sizeof(TInput));
    }
    return sizes;
}

// ========== Explicit Template Instantiations ==========

// Instantiate classes
template class LorenzoQuantStage<float, uint16_t>;
template class LorenzoQuantStage<float, uint8_t>;
template class LorenzoQuantStage<double, uint16_t>;
template class LorenzoQuantStage<double, uint32_t>;

// Instantiate kernel launchers
template void launchLorenzoKernel<float, uint16_t>(
    const float*, size_t, float, uint16_t,
    uint16_t*, float*, uint32_t*, uint32_t*, size_t, int, bool, cudaStream_t, float*);
template void launchLorenzoKernel<float, uint8_t>(
    const float*, size_t, float, uint8_t,
    uint8_t*, float*, uint32_t*, uint32_t*, size_t, int, bool, cudaStream_t, float*);
template void launchLorenzoKernel<double, uint16_t>(
    const double*, size_t, double, uint16_t,
    uint16_t*, double*, uint32_t*, uint32_t*, size_t, int, bool, cudaStream_t, double*);
template void launchLorenzoKernel<double, uint32_t>(
    const double*, size_t, double, uint32_t,
    uint32_t*, double*, uint32_t*, uint32_t*, size_t, int, bool, cudaStream_t, double*);

template void launchLorenzoInverseKernel<float, uint16_t>(
    const uint16_t*, const float*, const uint32_t*, uint32_t,
    size_t, float, uint16_t, float*, bool, cudaStream_t, MemoryPool*, const float*);
template void launchLorenzoInverseKernel<float, uint8_t>(
    const uint8_t*, const float*, const uint32_t*, uint32_t,
    size_t, float, uint8_t, float*, bool, cudaStream_t, MemoryPool*, const float*);
template void launchLorenzoInverseKernel<double, uint16_t>(
    const uint16_t*, const double*, const uint32_t*, uint32_t,
    size_t, double, uint16_t, double*, bool, cudaStream_t, MemoryPool*, const double*);
template void launchLorenzoInverseKernel<double, uint32_t>(
    const uint32_t*, const double*, const uint32_t*, uint32_t,
    size_t, double, uint32_t, double*, bool, cudaStream_t, MemoryPool*, const double*);

} // namespace fz
// (2-D and 3-D kernels + launchers are in lorenzo_quant_nd.cu)

