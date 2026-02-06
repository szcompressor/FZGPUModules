#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <vector>
#include <string>
#include <memory>

namespace fz {

// Forward declarations
class MemoryPool;
struct StageMemoryRequirements;
struct StageMetadata;

/**
 * Memory requirements for a stage execution
 */
struct StageMemoryRequirements {
    size_t output_size;        // Primary output buffer size
    size_t temp_size;          // Temporary/scratch memory needed
    size_t aux_output_size;    // Auxiliary output (e.g., outliers, metadata)
    
    StageMemoryRequirements(size_t out = 0, size_t temp = 0, size_t aux = 0)
        : output_size(out), temp_size(temp), aux_output_size(aux) {}
};

/**
 * Metadata about stage characteristics for optimization
 */
struct StageMetadata {
    std::string name;
    int register_pressure;      // Estimated register usage per thread
    int shared_memory_bytes;    // Shared memory per block
    bool is_memory_bound;       // vs compute bound
    bool produces_variable_output; // Output size depends on data
    
    StageMetadata(const std::string& n = "")
        : name(n), register_pressure(32), shared_memory_bytes(0),
          is_memory_bound(true), produces_variable_output(false) {}
};

/**
 * Base class for all compression pipeline stages
 * Supports CUDA graph integration, stream-ordered execution, and DAG composition
 */
class Stage {
public:
    Stage(const std::string& name);
    virtual ~Stage();
    
    // ========== Core Interface ==========
    
    /**
     * Execute the stage (non-graph mode)
     * @param input Pointer to input data on device
     * @param input_size Size of input in bytes
     * @param output Pointer to output buffer (pre-allocated)
     * @param stream CUDA stream for execution
     * @return Actual output size in bytes, or -1 on error
     */
    virtual int execute(void* input, size_t input_size, 
                       void* output, cudaStream_t stream) = 0;
    
    /**
     * Add this stage's operations to a CUDA graph
     * @param graph The graph to add to
     * @param dependencies Array of dependent graph nodes
     * @param num_deps Number of dependencies
     * @param input Input buffer pointer
     * @param input_size Input size in bytes
     * @param output Output buffer pointer
     * @param aux_buffers Pre-allocated auxiliary output buffers (from mempool)
     * @param stream Stream to associate with graph nodes
     * @return The graph node representing this stage's kernel
     */
    virtual cudaGraphNode_t addToGraph(cudaGraph_t graph,
                                       cudaGraphNode_t* dependencies,
                                       size_t num_deps,
                                       void* input, size_t input_size,
                                       void* output,
                                       const std::vector<void*>& aux_buffers,
                                       cudaStream_t stream) = 0;
    
    // ========== Memory Management ==========
    
    /**
     * Calculate memory requirements for given input size
     * @param input_size Size of input data in bytes
     * @return Memory requirements structure
     */
    virtual StageMemoryRequirements getMemoryRequirements(size_t input_size) const = 0;
    
    /**
     * Get maximum possible output size (worst case)
     * Used for buffer pre-allocation in graph mode
     */
    virtual size_t getMaxOutputSize(size_t input_size) const = 0;

    /**
     * Get average output size for given input size
     * If wanting to optimize memory usage
     */
    virtual size_t getAverageOutputSize(size_t input_size) const = 0;
    
    // ========== DAG Dependencies ==========
    
    /**
     * Set input stages that this stage depends on
     */
    void setDependencies(const std::vector<Stage*>& deps);
    
    /**
     * Get stages that this stage depends on
     */
    const std::vector<Stage*>& getDependencies() const { return dependencies_; }
    
    /**
     * Check if this stage can execute (all dependencies satisfied)
     */
    bool canExecute() const;
    
    // ========== Optimization Hints ==========
    
    /**
     * Get stage metadata for optimization decisions
     */
    virtual StageMetadata getMetadata() const;
    
    /**
     * Check if this stage can be fused with the next stage
     * Allows for kernel fusion optimization
     */
    virtual bool canFuseWith(const Stage* next) const { return false; }
    
    /**
     * Get optimal grid configuration for this stage
     * @param input_size Input data size
     * @param block_size Output: recommended block size
     * @param grid_size Output: recommended grid size
     */
    virtual void getOptimalLaunchConfig(size_t input_size,
                                       dim3& block_size,
                                       dim3& grid_size) const;
    
    // ========== Profiling Support ==========
    
    /**
     * Enable/disable profiling for this stage
     */
    void enableProfiling(bool enable);
    
    /**
     * Get last execution time in milliseconds
     * Only valid if profiling is enabled
     */
    float getLastExecutionTime() const { return last_execution_time_ms_; }
    
    /**
     * Get total accumulated execution time
     */
    float getTotalExecutionTime() const { return total_execution_time_ms_; }
    
    /**
     * Get number of times this stage has executed
     */
    int getExecutionCount() const { return execution_count_; }
    
    /**
     * Get throughput in GB/s (based on last execution)
     */
    float getThroughput() const;
    
    /**
     * Get average throughput in GB/s (across all executions)
     */
    float getAverageThroughput() const;
    
    /**
     * Get last input/output sizes
     */
    size_t getLastInputBytes() const { return last_input_bytes_; }
    size_t getLastOutputBytes() const { return last_output_bytes_; }
    
    /**
     * Update throughput stats (call after execute)
     */
    void updateThroughputStats(size_t input_bytes, size_t output_bytes);
    
    /**
     * Reset profiling statistics
     */
    void resetProfilingStats();
    
    // ========== Utilities ==========
    
    const std::string& getName() const { return name_; }
    
    /**
     * Mark this stage as completed (for DAG execution tracking)
     */
    void markCompleted() { completed_ = true; }
    
    /**
     * Reset completion status
     */
    void reset() { completed_ = false; }
    
    bool isCompleted() const { return completed_; }
    
protected:
    /**
     * Helper to record profiling events around kernel execution
     */
    void recordProfilingStart(cudaStream_t stream);
    void recordProfilingEnd(cudaStream_t stream);
    void updateProfilingStats();
    
    std::string name_;
    std::vector<Stage*> dependencies_;
    
    // Profiling state
    bool profiling_enabled_;
    cudaEvent_t start_event_;
    cudaEvent_t end_event_;
    float last_execution_time_ms_;
    float total_execution_time_ms_;
    int execution_count_;
    
    // Throughput tracking
    size_t last_input_bytes_;
    size_t last_output_bytes_;
    size_t total_input_bytes_;
    size_t total_output_bytes_;
    
    // Execution state
    bool completed_;
};

/**
 * Helper class for stages that produce multiple outputs (e.g., quantization -> codes + outliers)
 */
class MultiOutputStage : public Stage {
public:
    MultiOutputStage(const std::string& name) : Stage(name) {}
    virtual ~MultiOutputStage() = default;
    
    /**
     * Execute with multiple outputs
     * @param aux_outputs Vector of additional output buffers
     * @param aux_sizes Vector to store actual sizes of auxiliary outputs
     */
    virtual int executeMulti(void* input, size_t input_size,
                            void* primary_output,
                            std::vector<void*>& aux_outputs,
                            std::vector<size_t>& aux_sizes,
                            cudaStream_t stream) = 0;
    
    /**
     * Get number of auxiliary outputs this stage produces
     */
    virtual size_t getNumAuxiliaryOutputs() const = 0;
};

} // namespace fz

