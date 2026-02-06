#include "stage.h"
#include "mem/memory_pool.h"
#include <stdexcept>

namespace fz {

Stage::Stage(const std::string& name)
    : name_(name),
      profiling_enabled_(false),
      start_event_(nullptr),
      end_event_(nullptr),
      last_execution_time_ms_(0.0f),
      total_execution_time_ms_(0.0f),
      execution_count_(0),
      completed_(false) {
}

Stage::~Stage() {
    if (start_event_) {
        cudaEventDestroy(start_event_);
    }
    if (end_event_) {
        cudaEventDestroy(end_event_);
    }
}

void Stage::setDependencies(const std::vector<Stage*>& deps) {
    dependencies_ = deps;
}

bool Stage::canExecute() const {
    // Check if all dependencies are completed
    for (const auto* dep : dependencies_) {
        if (!dep->isCompleted()) {
            return false;
        }
    }
    return true;
}

StageMetadata Stage::getMetadata() const {
    return StageMetadata(name_);
}

void Stage::getOptimalLaunchConfig(size_t input_size,
                                   dim3& block_size,
                                   dim3& grid_size) const {
    // Default configuration - stages should override for better performance
    block_size = dim3(256, 1, 1);
    
    // Assume working with float data by default
    size_t num_elements = input_size / sizeof(float);
    size_t num_blocks = (num_elements + block_size.x - 1) / block_size.x;
    grid_size = dim3(num_blocks, 1, 1);
}

void Stage::enableProfiling(bool enable) {
    if (enable && !profiling_enabled_) {
        // Create events if they don't exist
        if (!start_event_) {
            cudaEventCreate(&start_event_);
        }
        if (!end_event_) {
            cudaEventCreate(&end_event_);
        }
    }
    profiling_enabled_ = enable;
}

void Stage::resetProfilingStats() {
    last_execution_time_ms_ = 0.0f;
    total_execution_time_ms_ = 0.0f;
    execution_count_ = 0;
    last_input_bytes_ = 0;
    last_output_bytes_ = 0;
    total_input_bytes_ = 0;
    total_output_bytes_ = 0;
}

void Stage::updateThroughputStats(size_t input_bytes, size_t output_bytes) {
    last_input_bytes_ = input_bytes;
    last_output_bytes_ = output_bytes;
    total_input_bytes_ += input_bytes;
    total_output_bytes_ += output_bytes;
}

float Stage::getThroughput() const {
    if (last_execution_time_ms_ <= 0.0f) return 0.0f;
    
    // Throughput = (input + output) / time
    // Convert to GB/s: bytes / (ms * 1e-3) / 1e9 = bytes / (ms * 1e6)
    double total_bytes = static_cast<double>(last_input_bytes_ + last_output_bytes_);
    return static_cast<float>(total_bytes / (last_execution_time_ms_ * 1e6));
}

float Stage::getAverageThroughput() const {
    if (total_execution_time_ms_ <= 0.0f || execution_count_ == 0) return 0.0f;
    
    double total_bytes = static_cast<double>(total_input_bytes_ + total_output_bytes_);
    return static_cast<float>(total_bytes / (total_execution_time_ms_ * 1e6));
}

void Stage::recordProfilingStart(cudaStream_t stream) {
    if (profiling_enabled_ && start_event_) {
        cudaEventRecord(start_event_, stream);
    }
}

void Stage::recordProfilingEnd(cudaStream_t stream) {
    if (profiling_enabled_ && end_event_) {
        cudaEventRecord(end_event_, stream);
    }
}

void Stage::updateProfilingStats() {
    if (profiling_enabled_ && start_event_ && end_event_) {
        cudaEventSynchronize(end_event_);
        float elapsed_ms = 0.0f;
        cudaEventElapsedTime(&elapsed_ms, start_event_, end_event_);
        
        last_execution_time_ms_ = elapsed_ms;
        total_execution_time_ms_ += elapsed_ms;
        execution_count_++;
    }
}

} // namespace fz

