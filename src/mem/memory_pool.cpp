#include "memory_pool.h"
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <iostream>
#include <iomanip>

namespace fz {

// ========== MemoryPool Implementation ==========

MemoryPool::MemoryPool(const MemoryPoolConfig& config)
    : config_(config),
      mem_pool_(nullptr),
      current_usage_(0),
      peak_usage_(0),
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
    
    // 1. Set release threshold based on pool size
    uint64_t threshold = config_.getPoolSize();
    cudaMemPoolSetAttribute(mem_pool_, cudaMemPoolAttrReleaseThreshold, &threshold);
    
    // 2. Enable opportunistic reuse if configured
    if (config_.enable_reuse) {
        int reuse = 1;
        cudaMemPoolSetAttribute(mem_pool_, cudaMemPoolReuseAllowOpportunistic, &reuse);
        cudaMemPoolSetAttribute(mem_pool_, cudaMemPoolReuseAllowInternalDependencies, &reuse);
    }
    
    // 3. Reserve initial pool memory
    uint64_t reserve_size = config_.getPoolSize();
    cudaMemPoolSetAttribute(mem_pool_, cudaMemPoolAttrReservedMemCurrent, &reserve_size);
}

void* MemoryPool::allocate(size_t size, cudaStream_t stream, const std::string& tag) {
    if (size == 0) return nullptr;
    
    void* ptr = nullptr;
    cudaError_t err = cudaMallocFromPoolAsync(&ptr, size, mem_pool_, stream);
    
    if (err != cudaSuccess) {
        std::cerr << "MemoryPool::allocate failed for size " << size 
                  << " bytes (tag: " << tag << "): "
                  << cudaGetErrorString(err) << std::endl;
        return nullptr;
    }
    
    // Track allocation
    allocations_[ptr] = AllocationInfo(ptr, size, tag);
    updateUsageStats(size, true);
    total_allocations_++;
    
    return ptr;
}

void MemoryPool::free(void* ptr, cudaStream_t stream) {
    if (!ptr) return;
    
    // Find allocation
    auto it = allocations_.find(ptr);
    if (it == allocations_.end()) {
        std::cerr << "MemoryPool::free - pointer not found in allocations" << std::endl;
        return;
    }
    
    size_t size = it->second.size;
    
    // Free memory (stream-ordered)
    cudaFreeAsync(ptr, stream);
    
    // Update tracking
    allocations_.erase(it);
    updateUsageStats(size, false);
    total_frees_++;
}

void* MemoryPool::allocateGraphMemory(size_t size, cudaStream_t stream, const std::string& tag) {
    if (size == 0) return nullptr;
    
    void* ptr = nullptr;
    cudaError_t err = cudaMallocFromPoolAsync(&ptr, size, mem_pool_, stream);
    
    if (err != cudaSuccess) {
        std::cerr << "MemoryPool::allocateGraphMemory failed for size " << size 
                  << " bytes (tag: " << tag << "): "
                  << cudaGetErrorString(err) << std::endl;
        return nullptr;
    }
    
    // Track as graph-persistent allocation
    graph_allocations_[ptr] = AllocationInfo(ptr, size, tag);
    updateUsageStats(size, true);
    total_allocations_++;
    
    return ptr;
}

void MemoryPool::freeGraphMemory(void* ptr, cudaStream_t stream) {
    if (!ptr) return;
    
    // Find allocation
    auto it = graph_allocations_.find(ptr);
    if (it == graph_allocations_.end()) {
        std::cerr << "MemoryPool::freeGraphMemory - pointer not found in graph allocations" << std::endl;
        return;
    }
    
    size_t size = it->second.size;
    
    // Free memory (stream-ordered)
    cudaFreeAsync(ptr, stream);
    
    // Update tracking
    graph_allocations_.erase(it);
    updateUsageStats(size, false);
    total_frees_++;
}

void MemoryPool::reset(cudaStream_t stream) {
    // Free all non-graph allocations
    for (auto& pair : allocations_) {
        cudaFreeAsync(pair.first, stream);
        updateUsageStats(pair.second.size, false);
        total_frees_++;
    }
    allocations_.clear();
}

void MemoryPool::trim() {
    // Trim pool - release memory back to OS
    cudaMemPoolTrimTo(mem_pool_, 0);
}

void MemoryPool::synchronize(cudaStream_t stream) {
    cudaStreamSynchronize(stream);
}

void MemoryPool::updateUsageStats(size_t size, bool allocating) {
    if (allocating) {
        current_usage_ += size;
        if (current_usage_ > peak_usage_) {
            peak_usage_ = current_usage_;
        }
    } else {
        current_usage_ -= size;
    }
}

void MemoryPool::printStats() const {
    std::cout << "\n========== Memory Pool Statistics ==========\n";
    std::cout << "Device ID: " << config_.device_id << "\n";
    std::cout << "Current usage: " << (current_usage_ / (1024.0 * 1024.0)) << " MB\n";
    std::cout << "Peak usage: " << (peak_usage_ / (1024.0 * 1024.0)) << " MB\n";
    std::cout << "Active allocations: " << allocations_.size() << "\n";
    std::cout << "Graph allocations: " << graph_allocations_.size() << "\n";
    std::cout << "Total allocations: " << total_allocations_ << "\n";
    std::cout << "Total frees: " << total_frees_ << "\n";
    std::cout << "\nActive Allocations:\n";
    
    for (const auto& pair : allocations_) {
        const auto& info = pair.second;
        std::cout << "  [" << info.tag << "] "
                  << (info.size / 1024.0) << " KB at " 
                  << info.ptr << "\n";
    }
    
    if (!graph_allocations_.empty()) {
        std::cout << "\nGraph-Persistent Allocations:\n";
        for (const auto& pair : graph_allocations_) {
            const auto& info = pair.second;
            std::cout << "  [" << info.tag << "] "
                      << (info.size / 1024.0) << " KB at " 
                      << info.ptr << "\n";
        }
    }
    
    std::cout << "==========================================\n" << std::endl;
}

} // namespace fz

