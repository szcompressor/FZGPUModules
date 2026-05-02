#pragma once

/**
 * @file mempool.h
 * @brief Stream-ordered CUDA memory pool for pipeline buffer management.
 */

#include <cuda_runtime.h>

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace fz {

/** Allocation metadata for debugging and lifetime tracking. */
struct AllocationInfo {
    void*       ptr;    ///< Device pointer.
    size_t      size;   ///< Size in bytes.
    std::string tag;    ///< Debug label (e.g. "lorenzo_output").
    bool        in_use; ///< True while allocated.

    AllocationInfo(void* p = nullptr, size_t s = 0, const std::string& t = "")
        : ptr(p), size(s), tag(t), in_use(true) {}
};

/** Construction parameters for MemoryPool. */
struct MemoryPoolConfig {
    size_t input_data_size;      ///< Input byte count used to size the pool.
    float  pool_size_multiplier; ///< Pool capacity = input_data_size × multiplier.
    int    device_id;            ///< CUDA device index.
    bool   enable_reuse;         ///< Enable opportunistic buffer reuse.

    MemoryPoolConfig(
        size_t input_size = 0,
        float  multiplier = 3.0f,
        int    device     = 0,
        bool   reuse      = true)
        : input_data_size(input_size),
          pool_size_multiplier(multiplier),
          device_id(device),
          enable_reuse(reuse) {}

    /** Returns `input_data_size × multiplier`, or 1 GiB if input_data_size is zero. */
    size_t getPoolSize() const {
        if (input_data_size == 0) return 1024ULL * 1024 * 1024;
        return static_cast<size_t>(input_data_size * pool_size_multiplier);
    }
};

/**
 * Stream-ordered CUDA memory pool.
 *
 * Uses `cudaMallocAsync`/`cudaFreeAsync` over a `cudaMemPool_t` for
 * efficient reuse and CUDA Graph compatibility. All allocations are tracked
 * for overflow warnings and debug printing.
 *
 * @note Non-copyable. Not thread-safe.
 */
class MemoryPool {
public:
    explicit MemoryPool(const MemoryPoolConfig& config = MemoryPoolConfig());
    ~MemoryPool();

    MemoryPool(const MemoryPool&)            = delete;
    MemoryPool& operator=(const MemoryPool&) = delete;

    // ── Allocation ────────────────────────────────────────────────────────────

    /**
     * Allocate `size` bytes from the pool on `stream`.
     *
     * @param size        Bytes to allocate.
     * @param stream      CUDA stream ordering the allocation.
     * @param tag         Debug label stored in `AllocationInfo`.
     * @param persistent  If true, allocation survives `reset()` (graph replay);
     *                    if false, `reset()` will free it.
     * @return Device pointer, or nullptr on failure.
     */
    void* allocate(size_t size, cudaStream_t stream,
                   const std::string& tag = "", bool persistent = false);

    /** Free `ptr` back to the pool, ordered on `stream`. */
    void free(void* ptr, cudaStream_t stream);

    // ── Lifecycle ─────────────────────────────────────────────────────────────

    /** Free all non-persistent allocations. Call between compression runs. */
    void reset(cudaStream_t stream);

    /** Release pool memory back to the OS if usage exceeds the release threshold. */
    void trim();

    /**
     * Update the CUDA pool's release threshold and keep config in sync.
     *
     * Called by Pipeline::finalize() after topology-aware sizing to replace
     * the blunt `input_size × multiplier` estimate with a tighter bound.
     *
     * @param bytes  New threshold in bytes.
     */
    void setReleaseThreshold(size_t bytes);

    /** Block until all stream-ordered operations on `stream` complete. */
    void synchronize(cudaStream_t stream);

    // ── Stats & debug ─────────────────────────────────────────────────────────

    /** Current live bytes (queries `cudaMemPoolAttrUsedMemCurrent`). */
    size_t getCurrentUsage() const {
        if (!mem_pool_) return current_allocated_bytes_;
        uint64_t used = 0;
        cudaMemPoolGetAttribute(mem_pool_, cudaMemPoolAttrUsedMemCurrent, &used);
        return static_cast<size_t>(used);
    }

    /** Peak live bytes since last reset (queries `cudaMemPoolAttrUsedMemHigh`). */
    size_t getPeakUsage() const {
        if (!mem_pool_) return current_allocated_bytes_;
        uint64_t high = 0;
        cudaMemPoolGetAttribute(mem_pool_, cudaMemPoolAttrUsedMemHigh, &high);
        return static_cast<size_t>(high);
    }

    /** Total number of currently live allocations (stream + graph). */
    size_t getAllocationCount() const { return allocations_.size() + graph_allocations_.size(); }

    void printStats() const;

    /**
     * Soft-capacity hint passed at construction (used only for overflow warnings;
     * the CUDA pool itself is not hard-capped).
     */
    size_t getConfiguredSize() const { return config_.getPoolSize(); }

    /** Raw `cudaMemPool_t` handle for advanced usage. */
    cudaMemPool_t getMemPool() const { return mem_pool_; }

    int getDeviceId() const { return config_.device_id; }

private:
    MemoryPoolConfig config_;
    cudaMemPool_t    mem_pool_;

    std::unordered_map<void*, AllocationInfo> allocations_;
    std::unordered_map<void*, AllocationInfo> graph_allocations_; ///< Persistent for graph replay.

    size_t total_allocations_;
    size_t total_frees_;
    // Host-side running total of live bytes — used for overflow detection without
    // querying a CUDA attribute on every hot-path allocation.
    size_t current_allocated_bytes_;
    // Set the first time current_allocated_bytes_ exceeds configured pool size
    // so the overflow warning fires only once per reset() cycle.
    bool   overflow_warned_;
    bool   initialized_;

    void initializeMemPool();
};

} // namespace fz
