#pragma once

#include "dag.h"
#include "mem/memory_pool.h"
#include "stage/stage.h"
#include <cuda_runtime.h>
#include <memory>
#include <vector>
#include <string>

namespace fz {

/**
 * Memory allocation strategy
 */
enum class MemoryMode {
    CONSERVATIVE,  // Allocate based on expected compression (~50% of worst-case)
    SAFE          // Allocate worst-case (100% - current behavior)
};

/**
 * Pipeline configuration
 */
struct PipelineConfig {
    int device_id;                      // CUDA device to use
    bool use_cuda_graph;                // Enable CUDA graph execution
    bool use_parallel_streams;          // Enable multi-stream parallelism
    int max_streams;                    // Maximum concurrent streams
    float memory_pool_multiplier;       // Pool size = input_size * multiplier
    bool enable_profiling;              // Enable per-stage timing
    MemoryMode memory_mode;             // Memory allocation strategy
    float conservative_alloc_factor;    // Allocation factor for CONSERVATIVE mode (0.0-1.0, default 0.5)
    
    PipelineConfig()
        : device_id(0),
          use_cuda_graph(true),
          use_parallel_streams(true),
          max_streams(4),
          memory_pool_multiplier(3.0f),
          enable_profiling(false),
          memory_mode(MemoryMode::SAFE),
          conservative_alloc_factor(0.5f) {}
};

/**
 * High-level compression pipeline
 * 
 * Manages the complete lifecycle of compression:
 * - Stage composition
 * - Memory management
 * - DAG execution
 * - CUDA graph optimization
 * 
 * Usage:
 *   Pipeline pipeline(config);
 *   pipeline.addStage(&lorenzo_stage);
 *   pipeline.addStage(&rle_stage);
 *   pipeline.addStage(&bitpack_stage);
 *   pipeline.build(input_size);
 *   size_t compressed_size = pipeline.compress(input, output);
 */
class Pipeline {
public:
    /**
     * Create compression pipeline
     */
    explicit Pipeline(const PipelineConfig& config = PipelineConfig());
    
    /**
     * Destructor - cleans up resources
     */
    ~Pipeline();
    
    // ========== Pipeline Construction ==========
    
    /**
     * Add a stage to the pipeline (executed in order added)
     * @param stage Pointer to stage (must outlive Pipeline)
     * @return Node pointer for dependency tracking
     */
    DagNode* addStage(Stage* stage);
    
    /**
     * Add dependency between stages (for multi-path pipelines)
     * @param dependent Stage that waits
     * @param dependency Stage to wait for
     */
    void addDependency(DagNode* dependent, DagNode* dependency);
    void addDependency(DagNode* dependent, DagNode* dependency, int aux_index);
    
    /**
     * Build the pipeline for a given input size
     * Must be called before compress()
     * 
     * @param input_size Maximum input size in bytes
     */
    void build(size_t input_size);
    
    // ========== Compression ==========
    
    /**
     * Compress data
     * 
     * @param input Input data (device pointer)
     * @param input_size Input size in bytes
     * @param output Output buffer (device pointer, pre-allocated)
     * @return Compressed size in bytes
     */
    size_t compress(void* input, size_t input_size, void* output);
    
    /**
     * Compress data (host version - handles H2D/D2H)
     * 
     * @param h_input Host input data
     * @param input_size Input size in bytes
     * @param h_output Host output buffer (pre-allocated)
     * @return Compressed size in bytes
     */
    size_t compressFromHost(const void* h_input, size_t input_size, void* h_output);
    
    /**
     * Decompress data (TODO: implement)
     */
    size_t decompress(void* input, size_t input_size, void* output);
    
    // ========== Resource Management ==========
    
    /**
     * Reset pipeline state for next compression
     * Frees temporary allocations, keeps graph-persistent memory
     */
    void reset();
    
    /**
     * Get the memory pool (for advanced usage)
     */
    MemoryPool* getMemoryPool() { return memory_pool_.get(); }
    
    /**
     * Get the DAG (for inspection/debugging)
     */
    Dag* getDag() { return dag_.get(); }
    
    /**
     * Get a specific node by index (for debugging)
     */
    DagNode* getNode(size_t index);
    
    /**
     * Get CUDA stream
     */
    cudaStream_t getStream() const { return stream_; }
    
    /**
     * Synchronize with pipeline execution
     */
    void synchronize();
    
    /**
     * Get stage output buffer (for debugging/inspection)
     * @param stage_index Index of stage (0-based)
     * @return Device pointer to stage output, or nullptr if invalid
     */
    void* getStageOutput(size_t stage_index);
    
    /**
     * Get stage auxiliary output buffer (for debugging/inspection)
     * @param stage_index Index of stage (0-based)
     * @return Device pointer to stage auxiliary output, or nullptr if invalid
     */
    void* getStageAuxOutput(size_t stage_index);
    
    // ========== Statistics & Debugging ==========
    
    /**
     * Print pipeline structure
     */
    void printStructure() const;
    
    /**
     * Print execution statistics
     */
    void printStats() const;
    
    /**
     * Print memory pool statistics
     */
    void printMemoryStats() const;
    
    /**
     * Get compression ratio (output/input)
     */
    float getCompressionRatio() const;
    
    /**
     * Get last compression size
     */
    size_t getLastCompressedSize() const { return last_compressed_size_; }
    
    /**
     * Get last input size
     */
    size_t getLastInputSize() const { return last_input_size_; }
    
private:
    PipelineConfig config_;
    
    // Resources
    std::unique_ptr<MemoryPool> memory_pool_;
    std::unique_ptr<Dag> dag_;
    cudaStream_t stream_;
    
    // Device buffers for host compress
    void* d_input_buffer_;
    void* d_output_buffer_;
    size_t d_buffer_size_;
    
    // State
    bool built_;
    size_t input_size_;
    size_t last_input_size_;
    size_t last_compressed_size_;
    
    // Pending stages/dependencies (before build)
    std::vector<Stage*> pending_stages_;
    struct PendingDependency {
        size_t dependent_idx;
        size_t dependency_idx;
        int aux_index;  // -1 for primary, 0+ for aux outputs
    };
    std::vector<PendingDependency> pending_dependencies_;  // Store indices and aux info
    
    // Helpers
    void initializeDevice();
    void cleanupDevice();
    void allocateDeviceBuffers(size_t size);
};

} // namespace fz
