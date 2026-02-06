#pragma once

#include "stage/stage.h"
#include <cuda_runtime.h>
#include <cstddef>
#include <limits>
#include <type_traits>

namespace fz {

/**
 * Configuration for Lorenzo predictor with quantization
 * @tparam TInput Input data type (float, double)
 * @tparam TCode Quantization code type (uint8_t, uint16_t, uint32_t)
 */
template<typename TInput, typename TCode>
struct LorenzoConfig {
    static_assert(std::is_floating_point<TInput>::value, "TInput must be floating point");
    static_assert(std::is_unsigned<TCode>::value, "TCode must be unsigned integer");
    
    TInput error_bound;          // Absolute error bound for quantization
    TCode quant_radius;          // Quantization radius
    float outlier_capacity;      // Max outliers as fraction of input (0.0-1.0)
    
    LorenzoConfig(TInput eb = static_cast<TInput>(1e-3), 
                  TCode radius = 512,
                  float outlier_frac = 0.2f)
        : error_bound(eb), 
          quant_radius(radius),
          outlier_capacity(outlier_frac) {}
};

/**
 * 1D Lorenzo predictor fused with quantization
 * 
 * @tparam TInput Input data type (float, double)
 * @tparam TCode Quantization code type (uint8_t, uint16_t, uint32_t)
 * 
 * Pipeline: TInput[] -> Lorenzo prediction -> Quantization -> TCode codes + outliers
 * 
 * Outputs:
 *   - Primary: Quantization codes (TCode array, same size as input)
 *   - Aux 0: Outlier prediction errors (TInput array, up to 20% of input size)
 *   - Aux 1: Outlier indices (uint32_t array, up to 20% of input size)  
 *   - Aux 2: Outlier count (single uint32_t)
 */
template<typename TInput = float, typename TCode = uint16_t>
class LorenzoStage : public MultiOutputStage {
public:
    explicit LorenzoStage(const LorenzoConfig<TInput, TCode>& config = LorenzoConfig<TInput, TCode>());
    virtual ~LorenzoStage() = default;
    
    // ========== Stage Interface Implementation ==========
    
    /**
     * Execute Lorenzo prediction + quantization
     * @param input TInput* to input data (device pointer)
     * @param input_size Size in bytes (num_elements * sizeof(TInput))
     * @param output TCode* for quantization codes (device pointer)
     * @param stream CUDA stream for execution
     * @return Size of output in bytes (num_elements * sizeof(TCode))
     */
    int execute(void* input, size_t input_size, 
               void* output, cudaStream_t stream) override;
    
    /**
     * Execute with outlier outputs
     * @param aux_outputs [0] = outlier_errors (TInput*, prediction errors not original values)
     *                    [1] = outlier_indices (uint32_t*)
     *                    [2] = outlier_count (uint32_t*)
     * @param aux_sizes Actual sizes written to auxiliary outputs
     */
    int executeMulti(void* input, size_t input_size,
                    void* primary_output,
                    std::vector<void*>& aux_outputs,
                    std::vector<size_t>& aux_sizes,
                    cudaStream_t stream) override;
    
    /**
     * Add to CUDA graph
     */
    cudaGraphNode_t addToGraph(cudaGraph_t graph,
                              cudaGraphNode_t* dependencies,
                              size_t num_deps,
                              void* input, size_t input_size,
                              void* output,
                              const std::vector<void*>& aux_buffers,
                              cudaStream_t stream) override;
    
    // ========== Memory Management ==========
    
    /**
     * Calculate memory requirements
     * - output_size: quantization codes (num_elements * sizeof(TCode))
     * - temp_size: working memory for kernel (minimal)
     * - aux_output_size: outlier data (values + indices + count)
     */
    StageMemoryRequirements getMemoryRequirements(size_t input_size) const override;
    
    /**
     * Maximum output is same as input (1:1 mapping)
     */
    size_t getMaxOutputSize(size_t input_size) const override {
        size_t num_elements = input_size / sizeof(TInput);
        return num_elements * sizeof(TCode);
    }
    
    /**
     * Average output size (for buffer allocation)
     * Lorenzo + quantization typically produces codes of same size
     */
    size_t getAverageOutputSize(size_t input_size) const override {
        return getMaxOutputSize(input_size);
    }
    
    size_t getNumAuxiliaryOutputs() const override { return 3; }
    
    // ========== Configuration ==========
    
    /**
     * Update error bound (useful for adaptive compression)
     */
    void setErrorBound(TInput error_bound);
    
    TInput getErrorBound() const { return config_.error_bound; }
    
    const LorenzoConfig<TInput, TCode>& getConfig() const { return config_; }
    
    // ========== Metadata ==========
    
    StageMetadata getMetadata() const override;
    
    void getOptimalLaunchConfig(size_t input_size,
                               dim3& block_size,
                               dim3& grid_size) const override;
    
private:
    LorenzoConfig<TInput, TCode> config_;
    
    // Helper to calculate outlier buffer sizes
    size_t getMaxOutlierCount(size_t num_elements) const {
        return static_cast<size_t>(num_elements * config_.outlier_capacity);
    }
};

// ========== CUDA Kernel Declarations ==========

/**
 * Fused Lorenzo prediction + quantization kernel (optimized with shared memory)
 * 
 * Uses shared memory staging and sequential processing for better performance.
 * Each thread processes SEQ elements using pre-quantization for efficiency.
 * 
 * Outlier positions are marked with code value 0. Use outlier_indices for reconstruction.
 * 
 * @tparam TInput Input data type
 * @tparam TCode Quantization code type
 * @tparam TileDim Total elements per block (e.g., 1024)
 * @tparam Seq Sequential elements per thread (e.g., 4)
 * @param input Input data
 * @param n Number of elements
 * @param ebx2_r Reciprocal of (2 * error_bound) for pre-quantization
 * @param quant_radius Quantization radius
 * @param quant_codes Output quantization codes (0 = outlier position)
 * @param outlier_errors Output outlier prediction errors (not original values)
 * @param outlier_indices Output outlier indices  
 * @param outlier_count Output outlier count (atomic counter)
 * @param max_outliers Maximum outliers allowed
 */
template<typename TInput, typename TCode, int TileDim = 1024, int Seq = 4>
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
);

// Common instantiations
using LorenzoStageF32U16 = LorenzoStage<float, uint16_t>;
using LorenzoStageF32U8 = LorenzoStage<float, uint8_t>;
using LorenzoStageF64U16 = LorenzoStage<double, uint16_t>;
using LorenzoStageF64U32 = LorenzoStage<double, uint32_t>;

} // namespace fz

