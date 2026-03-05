#include <algorithm>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>

#include "mempool.h"
#include "log.h"
#include "cuda_check.h"

namespace fz {

// Helper function to format bytes with appropriate units
static std::string formatBytes(uint64_t bytes) {
    const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    int unit_idx = 0;
    double value = static_cast<double>(bytes);
    
    while (value >= 1024.0 && unit_idx < 4) {
        value /= 1024.0;
        unit_idx++;
    }
    
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << value << " " << units[unit_idx];
    return oss.str();
}

// ========== MemoryPool Implementation ==========

MemoryPool::MemoryPool(const MemoryPoolConfig& config)
    : config_(config),
      mem_pool_(nullptr),
      total_allocations_(0),
      total_frees_(0),
      initialized_(false) {
    
    // Set device
    cudaSetDevice(config_.device_id);
    
    // Initialize CUDA memory pool
    initializeMemPool();
    
    initialized_ = true;
}

MemoryPool::~MemoryPool() {
    if (!initialized_) return;
    
    // Destroy memory pool
    if (mem_pool_) {
        cudaMemPoolDestroy(mem_pool_);
    }
}

void MemoryPool::initializeMemPool() {
    // Get default mempool for device
    cudaMemPoolProps pool_props = {};
    pool_props.allocType = cudaMemAllocationTypePinned;
    pool_props.handleTypes = cudaMemHandleTypeNone;
    pool_props.location.type = cudaMemLocationTypeDevice;
    pool_props.location.id = config_.device_id;
    
    // Create memory pool
    cudaError_t err = cudaMemPoolCreate(&mem_pool_, &pool_props);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to create CUDA memory pool: " + 
                                std::string(cudaGetErrorString(err)));
    }
    
    // Set pool attributes for performance
    
    // 1. Set release threshold based on reuse configuration
    // If reuse is disabled, never release memory (keep it all in pool)
    // If reuse is enabled, release memory above pool size
    uint64_t threshold = config_.enable_reuse ? config_.getPoolSize() : UINT64_MAX;
    err = cudaMemPoolSetAttribute(mem_pool_, cudaMemPoolAttrReleaseThreshold, &threshold);
    if (err != cudaSuccess) {
        // Non-fatal - continue without custom threshold
        cudaGetLastError(); // Clear error
    }
    
    // 2. Enable opportunistic reuse if configured
    if (config_.enable_reuse) {
        int reuse = 1;
        err = cudaMemPoolSetAttribute(mem_pool_, cudaMemPoolReuseAllowOpportunistic, &reuse);
        if (err != cudaSuccess) {
            cudaGetLastError(); // Clear error
        }
        
        err = cudaMemPoolSetAttribute(mem_pool_, cudaMemPoolReuseAllowInternalDependencies, &reuse);
        if (err != cudaSuccess) {
            cudaGetLastError(); // Clear error
        }
    }
    
    // 3. Reserve initial pool memory (optional optimization)
    // Note: Many GPUs require reservations to be at least 2MB and aligned.
    // Small reservations may fail with "invalid argument".
    // On-demand allocation works fine, so we only reserve for larger pools.
    uint64_t reserve_size = config_.getPoolSize();
    const uint64_t MIN_RESERVE_SIZE = 2 * 1024 * 1024; // 2 MB
    
    if (reserve_size >= MIN_RESERVE_SIZE) {
        // Align to 2MB boundary for better compatibility
        reserve_size = (reserve_size + MIN_RESERVE_SIZE - 1) & ~(MIN_RESERVE_SIZE - 1);
        
        err = cudaMemPoolSetAttribute(mem_pool_, cudaMemPoolAttrReservedMemCurrent, &reserve_size);
        if (err != cudaSuccess) {
            // Non-fatal - memory will be allocated on-demand instead
            cudaGetLastError(); // Clear error
        }
    }
    // For small pools, skip pre-reservation and let memory allocate on-demand
}

void* MemoryPool::allocate(size_t size, cudaStream_t stream, const std::string& tag, bool persistent) {
    if (size == 0) return nullptr;
    
    void* ptr = nullptr;
    cudaError_t err = cudaMallocFromPoolAsync(&ptr, size, mem_pool_, stream);
    
    if (err != cudaSuccess) {
        FZ_LOG(WARN, "MemoryPool::allocate failed for size %zu bytes (tag: %s): %s",
               size, tag.c_str(), cudaGetErrorString(err));
        return nullptr;
    }
    
    // Track allocation in appropriate map (always track for stats, but can be disabled if needed)
    if (persistent) {
        graph_allocations_[ptr] = AllocationInfo(ptr, size, tag);
    } else {
        allocations_[ptr] = AllocationInfo(ptr, size, tag);
    }
    total_allocations_++;
    
    return ptr;
}

void MemoryPool::free(void* ptr, cudaStream_t stream) {
    if (!ptr) return;
    
    // Check stream allocations first
    auto it = allocations_.find(ptr);
    if (it != allocations_.end()) {
        FZ_CUDA_CHECK_WARN(cudaFreeAsync(ptr, stream));
        allocations_.erase(it);
        total_frees_++;
        return;
    }
    
    // Check graph allocations
    auto git = graph_allocations_.find(ptr);
    if (git != graph_allocations_.end()) {
        FZ_CUDA_CHECK_WARN(cudaFreeAsync(ptr, stream));
        graph_allocations_.erase(git);
        total_frees_++;
        return;
    }
    
    FZ_LOG(WARN, "MemoryPool::free - pointer not found in allocations");
}

void MemoryPool::reset(cudaStream_t stream) {
    // Free all non-graph allocations
    for (auto& pair : allocations_) {
        FZ_CUDA_CHECK_WARN(cudaFreeAsync(pair.first, stream));
        total_frees_++;
    }
    allocations_.clear();
}

void MemoryPool::trim() {
    // Trim pool - release memory back to OS
    cudaMemPoolTrimTo(mem_pool_, 0);
}

void MemoryPool::synchronize(cudaStream_t stream) {
    FZ_CUDA_CHECK(cudaStreamSynchronize(stream));
}

void MemoryPool::printStats() const {
    std::cout << "\n========== Memory Pool Statistics ==========\n";
    std::cout << "Device ID: " << config_.device_id << "\n";
    
    // Query CUDA memory pool attributes
    uint64_t reserved_mem_current = 0;
    uint64_t reserved_mem_high = 0;
    uint64_t used_mem_current = 0;
    uint64_t used_mem_high = 0;
    uint64_t release_threshold = 0;
    
    cudaMemPoolGetAttribute(mem_pool_, cudaMemPoolAttrReservedMemCurrent, &reserved_mem_current);
    cudaMemPoolGetAttribute(mem_pool_, cudaMemPoolAttrReservedMemHigh, &reserved_mem_high);
    cudaMemPoolGetAttribute(mem_pool_, cudaMemPoolAttrUsedMemCurrent, &used_mem_current);
    cudaMemPoolGetAttribute(mem_pool_, cudaMemPoolAttrUsedMemHigh, &used_mem_high);
    cudaMemPoolGetAttribute(mem_pool_, cudaMemPoolAttrReleaseThreshold, &release_threshold);
    
    // Pool-level statistics from CUDA
    std::cout << "\nMemory Pool Status:\n";
    std::cout << "  Reserved memory (current): " << formatBytes(reserved_mem_current) << "\n";
    std::cout << "  Reserved memory (peak):    " << formatBytes(reserved_mem_high) << "\n";
    std::cout << "  Used memory (current):     " << formatBytes(used_mem_current) << "\n";
    std::cout << "  Used memory (peak):        " << formatBytes(used_mem_high) << "\n";
    std::cout << "  Release threshold:         " << (release_threshold == UINT64_MAX ? "UNLIMITED" : formatBytes(release_threshold)) << "\n";
    
#ifndef NDEBUG
    std::cout << "\nAllocation Tracking:\n";
    std::cout << "  Active allocations:        " << allocations_.size() << "\n";
    std::cout << "  Graph allocations:         " << graph_allocations_.size() << "\n";
    std::cout << "  Total allocations:         " << total_allocations_ << "\n";
    std::cout << "  Total frees:               " << total_frees_ << "\n";
    std::cout << "\nActive Allocations:\n";
    
    for (const auto& pair : allocations_) {
        const auto& info = pair.second;
        std::cout << "  [" << info.tag << "] "
                  << formatBytes(info.size) << " at " 
                  << info.ptr << "\n";
    }
    
    if (!graph_allocations_.empty()) {
        std::cout << "\nGraph-Persistent Allocations:\n";
        for (const auto& pair : graph_allocations_) {
            const auto& info = pair.second;
            std::cout << "  [" << info.tag << "] "
                      << formatBytes(info.size) << " at " 
                      << info.ptr << "\n";
        }
    }
#endif
    
    std::cout << "==========================================\n" << std::endl;
}

} // namespace fz