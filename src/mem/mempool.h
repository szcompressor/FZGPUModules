#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace fz {

/**
 * Allocation metadata for tracking and debugging
 */
struct AllocationInfo {
    void* ptr;              // Device pointer
    size_t size;            // Size in bytes
    std::string tag;        // Debug tag (e.g., "lorenzo_output", "rle_temp")
    bool in_use;            // Currently allocated?
    
    AllocationInfo(void* p = nullptr, size_t s = 0, const std::string& t = "")
        : ptr(p), size(s), tag(t), in_use(true) {}
};

/**
 * Memory pool configuration
 */
struct MemoryPoolConfig {
    size_t input_data_size;             // Input data size to base pool sizing on
    float pool_size_multiplier;         // Pool = input_size * multiplier (e.g., 3x)
    int device_id;                      // CUDA device ID
    bool enable_reuse;                  // Enable opportunistic memory reuse
    
    MemoryPoolConfig(
        size_t input_size = 0,
        float multiplier = 3.0f,  // 3x input size is usually sufficient
        int device = 0,
        bool reuse = true
    ) : input_data_size(input_size),
        pool_size_multiplier(multiplier),
        device_id(device),
        enable_reuse(reuse) {}
    
    // Get actual pool size based on input
    size_t getPoolSize() const {
        if (input_data_size == 0) {
            // Fallback: 1GB default
            return 1024ULL * 1024 * 1024;
        }
        return static_cast<size_t>(input_data_size * pool_size_multiplier);
    }
};

/**
 * CUDA-graph compatible memory pool using stream-ordered allocation
 *
 * Features:
 * - Stream-ordered cudaMallocAsync/cudaFreeAsync
 * - CUDA mempool for efficient reuse
 * - Memory tracking for debugging
 * - Graph-compatible (allocations persist across graph execution)
 * - Opportunistic memory reuse to minimize footprint
 */
class MemoryPool {
public:
    /**
     * Create memory pool for specified device and stream
     */
    MemoryPool(const MemoryPoolConfig& config = MemoryPoolConfig());
    
    /**
     * Destructor - ensures all memory is freed
     */
    ~MemoryPool();
    
    // Disable copy/move for safety
    MemoryPool(const MemoryPool&) = delete;
    MemoryPool& operator=(const MemoryPool&) = delete;
    
    // ========== Stream-Ordered Allocation ==========
    
    /**
     * Allocate memory from pool (stream-ordered)
     * 
     * @param size Size in bytes
     * @param stream CUDA stream for allocation
     * @param tag Debug tag for tracking
     * @param persistent If true, allocation persists for graph replay (graph mode)
     *                   If false, can be freed/reused immediately (stream mode)
     * @return Device pointer (nullptr on failure)
     */
    void* allocate(size_t size, cudaStream_t stream, const std::string& tag = "", bool persistent = false);
    
    /**
     * Free memory back to pool (stream-ordered)
     * 
     * @param ptr Device pointer to free
     * @param stream CUDA stream for deallocation
     */
    void free(void* ptr, cudaStream_t stream);
    
    // ========== Memory Management ==========
    
    /**
     * Reset pool - free all tracked allocations
     * Should be called between compression runs
     */
    void reset(cudaStream_t stream);
    
    /**
     * Trim pool - release memory back to OS if exceeds threshold
     */
    void trim();

    /**
     * Update the CUDA pool's release threshold and keep the internal config
     * in sync so that overflow warnings fire at the new value.
     *
     * Called by Pipeline::finalize() after topology-aware pool sizing to
     * replace the blunt input_size*multiplier estimate with a tighter bound
     * derived from the actual DAG buffer layout.
     *
     * @param bytes  New release threshold in bytes.
     */
    void setReleaseThreshold(size_t bytes);
    
    /**
     * Synchronize with stream to ensure all operations complete
     */
    void synchronize(cudaStream_t stream);
    
    // ========== Statistics & Debugging ==========
    
    /**
     * Get current pool usage in bytes (queries CUDA memory pool)
     */
    size_t getCurrentUsage() const {
        uint64_t used_mem = 0;
        cudaMemPoolGetAttribute(mem_pool_, cudaMemPoolAttrUsedMemCurrent, &used_mem);
        return static_cast<size_t>(used_mem);
    }
    
    /**
     * Get peak pool usage in bytes (queries CUDA memory pool)
     */
    size_t getPeakUsage() const {
        uint64_t used_mem_high = 0;
        cudaMemPoolGetAttribute(mem_pool_, cudaMemPoolAttrUsedMemHigh, &used_mem_high);
        return static_cast<size_t>(used_mem_high);
    }
    
    /**
     * Get number of active allocations (stream + graph)
     */
    size_t getAllocationCount() const { return allocations_.size() + graph_allocations_.size(); }
    
    /**
     * Print allocation statistics
     */
    void printStats() const;
    
    /**
     * Get configured pool capacity (the size hint passed at construction).
     * The CUDA pool itself is soft-capped by ReleaseThreshold, not hard-capped,
     * so this is used only for overflow warnings.
     */
    size_t getConfiguredSize() const { return config_.getPoolSize(); }

    /**
     * Get CUDA mempool handle (for advanced usage)
     */
    cudaMemPool_t getMemPool() const { return mem_pool_; }
    
    /**
     * Get device ID
     */
    int getDeviceId() const { return config_.device_id; }
    
private:
    MemoryPoolConfig config_;
    cudaMemPool_t mem_pool_;           // CUDA memory pool handle
    
    // Allocation tracking
    std::unordered_map<void*, AllocationInfo> allocations_;
    std::unordered_map<void*, AllocationInfo> graph_allocations_;  // Persistent for graphs
    
    // Statistics
    size_t total_allocations_;
    size_t total_frees_;
    // Host-side running total of currently live bytes (matches what we asked for,
    // not the driver's internal bookkeeping).  Used for overflow detection without
    // querying a CUDA attribute on every hot-path allocation.
    size_t current_allocated_bytes_;
    // Set the first time current_allocated_bytes_ exceeds the configured pool size
    // so we only emit the overflow warning once per reset() cycle.
    bool overflow_warned_;
    
    bool initialized_;
    
    // Helper: Create and configure CUDA mempool
    void initializeMemPool();

}; // memory pool class

} // namespace fz