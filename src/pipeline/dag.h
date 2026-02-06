#pragma once

#include "stage/stage.h"
#include "mem/memory_pool.h"
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <memory>
#include <string>

namespace fz {

// Forward declaration
enum class MemoryMode;

/**
 * DAG node representing a stage in the compression pipeline
 */
struct DagNode {
    Stage* stage;                           // The stage to execute
    std::vector<DagNode*> dependencies;     // Stages this depends on
    std::vector<DagNode*> dependents;       // Stages that depend on this
    std::vector<int> dependency_aux_indices; // Which output to use from each dependency (-1=primary, 0+=aux)
    
    cudaStream_t stream;                    // Assigned CUDA stream
    cudaGraphNode_t graph_node;             // CUDA graph node (for graph mode)
    
    // Memory management
    void* output_buffer;                    // Primary output buffer
    std::vector<void*> aux_buffers;         // Auxiliary output buffers
    size_t output_size;                     // Actual output size (runtime)
    std::vector<size_t> aux_sizes;          // Actual aux sizes (runtime)
    size_t allocated_output_size;           // Allocated output buffer size
    std::vector<size_t> allocated_aux_sizes; // Allocated aux buffer sizes
    
    // Execution state
    bool completed;                         // Has this node been executed?
    bool freed;                             // Have buffers been freed?
    int execution_order;                    // Topological order (-1 = not set)
    int level;                              // Parallelism level for stream assignment
    
    DagNode(Stage* s)
        : stage(s), stream(nullptr), graph_node(nullptr),
          output_buffer(nullptr), output_size(0), allocated_output_size(0),
          completed(false), freed(false), execution_order(-1), level(-1) {}
};

/**
 * DAG configuration
 */
struct DagConfig {
    bool use_cuda_graph;        // Use CUDA graph for execution
    bool parallel_streams;      // Use multiple streams for independent branches
    int max_streams;            // Maximum number of concurrent streams
    bool enable_profiling;      // Enable per-stage profiling
    MemoryMode memory_mode;     // Memory allocation strategy
    float conservative_alloc_factor;  // Allocation factor for CONSERVATIVE mode (0.0-1.0)
    
    DagConfig();
};

/**
 * DAG - Directed Acyclic Graph of compression stages
 */
class Dag {
public:
    Dag(MemoryPool* memory_pool, const DagConfig& config = DagConfig());
    ~Dag();
    
    // ========== DAG Construction ==========
    
    /**
     * @return Pointer to the created node
     */
    DagNode* addStage(Stage* stage);
    
    /**
     * Add dependency for stream synchronization
     * Only needed if you want parallel streams to sync properly.
     * dependent waits for dependency to complete before executing.
     */
    void addDependency(DagNode* dependent, DagNode* dependency);
    
    /**
     * Add dependency with specific output selection
     * @param dependent The node that depends on dependency's output
     * @param dependency The node producing the output
     * @param aux_index Which output to use: -1 for primary, 0+ for aux outputs
     */
    void addDependency(DagNode* dependent, DagNode* dependency, int aux_index);
    
    // ========== Execution ==========
    
    /**
     * Prepare for execution:
     * - Stream assignment
     * - Memory allocation
     * - CUDA graph construction (if enabled)
     */
    void build(size_t input_size);
    
    /**
     * Execute the DAG in the order stages were added
     * @param input Input data (device pointer)
     * @param input_size Input size in bytes
     * @param output Output buffer (device pointer, pre-allocated)
     * @param stream Main CUDA stream
     * @return Total output size in bytes
     */
    size_t execute(void* input, size_t input_size, void* output, cudaStream_t stream);
    
    /**
     * Reset DAG state for next execution
     */
    void reset();
    
    // ========== Graph Mode ==========
    
    /**
     * Get the CUDA graph (if built)
     */
    cudaGraph_t getGraph() const { return cuda_graph_; }
    
    /**
     * Get the CUDA graph executable (if built)
     */
    cudaGraphExec_t getGraphExec() const { return cuda_graph_exec_; }
    
    // ========== Utilities ==========
    
    /**
     * Get all nodes in execution order
     */
    const std::vector<DagNode*>& getNodes() const { return nodes_; }
    
    /**
     * Get a specific node by index
     */
    DagNode* getNode(size_t index) { return nodes_[index]; }
    
    /**
     * Print DAG structure for debugging
     */
    void printStructure() const;
    
    /**
     * Print execution statistics
     */
    void printStats() const;
    
    /**
     * Print buffer allocation details for each stage
     */
    void printBufferAllocations() const;
    
private:
    MemoryPool* memory_pool_;
    DagConfig config_;
    
    std::vector<DagNode*> nodes_;              // All nodes in execution order
    
    // CUDA graph resources
    cudaGraph_t cuda_graph_;
    cudaGraphExec_t cuda_graph_exec_;
    bool graph_built_;
    
    // Stream management
    std::vector<cudaStream_t> streams_;
    
    // Helpers
    void assignStreams();
    void allocateBuffers(size_t input_size);
    void buildCudaGraph(void* input, size_t input_size, cudaStream_t stream);
    void assignLevels();  // For parallel stream assignment
};

} // namespace fz
