#include "pipeline/dag.h"
#include "stage/stage.h"
#include "mem/mempool.h"
#include "log.h"
#include "cuda_check.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace fz {

// ========== Profiling ==========

void CompressionDAG::enableProfiling(bool enable) {
    profiling_enabled_ = enable;

    if (enable) {
        // Create start_event for all nodes that were added before this call
        for (auto* node : nodes_) {
            if (!node->start_event) {
                FZ_CUDA_CHECK_WARN(cudaEventCreate(&node->start_event));
            }
        }
    } else {
        // Destroy and null out all start events to reclaim resources
        for (auto* node : nodes_) {
            if (node->start_event) {
                FZ_CUDA_CHECK_WARN(cudaEventDestroy(node->start_event));
                node->start_event = nullptr;
            }
        }
    }
}

std::vector<StageTimingResult> CompressionDAG::collectTimings() {
    if (!profiling_enabled_) return {};

    // Sync all owned streams so that CUDA event queries are valid.
    // (node streams may differ from the fallback stream passed to execute())
    for (auto s : streams_) {
        if (s) FZ_CUDA_CHECK(cudaStreamSynchronize(s));
    }

    std::vector<StageTimingResult> results;
    results.reserve(nodes_.size());

    for (auto* node : nodes_) {
        if (!node->start_event || !node->completion_event) continue;

        StageTimingResult r;
        r.name       = node->name;
        r.level      = node->level;
        r.elapsed_ms = 0.0f;

        cudaError_t err = cudaEventElapsedTime(&r.elapsed_ms,
                                               node->start_event,
                                               node->completion_event);
        if (err != cudaSuccess) {
            FZ_LOG(WARN, "cudaEventElapsedTime failed for '%s': %s",
                   node->name.c_str(), cudaGetErrorString(err));
            r.elapsed_ms = -1.0f;
        }

        // Sum up buffer sizes for input/output byte counts
        r.input_bytes = 0;
        for (int buf_id : node->input_buffer_ids) {
            auto it = buffers_.find(buf_id);
            if (it != buffers_.end()) r.input_bytes += it->second.size;
        }
        r.output_bytes = 0;
        for (int buf_id : node->output_buffer_ids) {
            auto it = buffers_.find(buf_id);
            if (it != buffers_.end()) r.output_bytes += it->second.size;
        }

        results.push_back(r);
    }

    return results;
}

// ========== Query & Debug ==========

size_t CompressionDAG::getTotalBufferSize() const {
    size_t total = 0;
    for (const auto& [buffer_id, buffer] : buffers_) {
        total += buffer.size;
    }
    return total;
}

size_t CompressionDAG::computeTopoPoolSize() const {
    // Helper: build input_sizes vector for a node from the buffer table.
    auto getInputSizes = [&](const DAGNode* node) {
        std::vector<size_t> sizes;
        sizes.reserve(node->input_buffer_ids.size());
        for (int bid : node->input_buffer_ids) {
            auto it = buffers_.find(bid);
            sizes.push_back(it != buffers_.end() ? it->second.size : 0);
        }
        return sizes;
    };

    if (strategy_ == MemoryStrategy::PREALLOCATE) {
        // Pool must be large enough to hold all live buffers simultaneously,
        // plus all stages' persistent scratch.
        // With coloring, the effective buffer footprint is the sum of color
        // region sizes (not the sum of individual buffer sizes).
        size_t total = 0;
        if (coloring_applied_) {
            for (size_t sz : color_region_sizes_) total += sz;
        } else {
            for (const auto& [buf_id, buf_info] : buffers_) {
                if (!buf_info.is_external)
                    total += buf_info.size;
            }
        }
        for (const auto* node : nodes_) {
            if (node->stage)
                total += node->stage->estimateScratchBytes(getInputSizes(node));
        }
        return total;
    }

    // MINIMAL: simulate level-by-level allocation and deallocation
    // to find the peak concurrent live bytes.

    // Build node_id -> level map for consumer lookup.
    std::unordered_map<int, int> node_level;
    node_level.reserve(nodes_.size());
    for (const auto* node : nodes_)
        node_level[node->id] = node->level;

    // For each non-external buffer, compute the level after which it is freed.
    // A buffer is freed when all its consumers have executed, i.e. after the
    // highest-level consumer.  Pipeline-output buffers (no consumers) survive
    // until the last level.
    std::unordered_map<int, int> free_after_level;
    for (const auto& [buf_id, buf_info] : buffers_) {
        if (buf_info.is_external) continue;
        if (buf_info.consumer_stage_ids.empty()) {
            free_after_level[buf_id] = max_level_;
        } else {
            int max_lvl = 0;
            for (int cid : buf_info.consumer_stage_ids) {
                auto it = node_level.find(cid);
                if (it != node_level.end())
                    max_lvl = std::max(max_lvl, it->second);
            }
            free_after_level[buf_id] = max_lvl;
        }
    }

    // Walk levels, tracking the running total of live bytes.
    // Persistent stage scratch is added when the stage executes and is never
    // subtracted (it lives until DAG destruction, not until consumers finish).
    size_t running = 0, peak = 0;
    for (int lvl = 0; lvl <= max_level_; ++lvl) {
        // Allocate: output buffers + persistent scratch of all stages at this level.
        if (lvl < static_cast<int>(levels_.size())) {
            for (const auto* node : levels_[lvl]) {
                for (int buf_id : node->output_buffer_ids) {
                    auto it = buffers_.find(buf_id);
                    if (it != buffers_.end() && !it->second.is_external)
                        running += it->second.size;
                }
                if (node->stage)
                    running += node->stage->estimateScratchBytes(getInputSizes(node));
            }
        }
        peak = std::max(peak, running);
        // Free: buffers whose last consumer executed at this level.
        for (const auto& [buf_id, free_lvl] : free_after_level) {
            if (free_lvl == lvl) {
                auto it = buffers_.find(buf_id);
                if (it != buffers_.end())
                    running -= it->second.size;
            }
        }
    }
    return peak;
}

int CompressionDAG::getMaxParallelism() const {
    // Returns the maximum number of nodes at any single level
    int max_width = 0;
    for (const auto& level : levels_) {
        max_width = std::max(max_width, static_cast<int>(level.size()));
    }
    return max_width;
}

void CompressionDAG::printDAG() const {
    // Build comma-separated id lists into a local buffer.
    auto fmt_ids = [](const auto& vec) -> std::string {
        std::string s;
        for (size_t i = 0; i < vec.size(); i++) {
            if (i) s += ", ";
            s += std::to_string(vec[i]);
        }
        return s;
    };
    auto fmt_node_ids = [](const auto& vec) -> std::string {
        std::string s;
        for (size_t i = 0; i < vec.size(); i++) {
            if (i) s += ", ";
            s += std::to_string(vec[i]->id);
        }
        return s;
    };

    FZ_PRINT("========== Compression DAG ==========");
    FZ_PRINT("Strategy: %s",
             strategy_ == MemoryStrategy::MINIMAL ? "MINIMAL" : "PREALLOCATE");
    FZ_PRINT("Parallel streams: %zu  Max level: %d",
             streams_.size(), max_level_);

    FZ_PRINT("Stages (%zu):", nodes_.size());
    for (const auto* node : nodes_) {
        std::string line = "  [" + std::to_string(node->id) + "] " + node->name
            + " (Level " + std::to_string(node->level) + ", Stream "
            + (node->stream ? "assigned" : "default") + ")";
        if (!node->dependencies.empty())
            line += " <- deps: [" + fmt_node_ids(node->dependencies) + "]";
        if (!node->input_buffer_ids.empty())
            line += ", inputs: [" + fmt_ids(node->input_buffer_ids) + "]";
        if (!node->output_buffer_ids.empty())
            line += " -> outputs: [" + fmt_ids(node->output_buffer_ids) + "]";
        FZ_PRINT("%s", line.c_str());
    }

    FZ_PRINT("Buffers (%zu):", buffers_.size());
    for (const auto& [buffer_id, buffer] : buffers_) {
        FZ_PRINT("  [%d] %s (%.1f KB)%s",
                 buffer_id, buffer.tag.c_str(), buffer.size / 1024.0,
                 buffer.is_persistent ? " [PERSISTENT]" : "");
        FZ_PRINT("      producer: %d, consumers: [%s]",
                 buffer.producer_stage_id,
                 fmt_ids(buffer.consumer_stage_ids).c_str());
    }

    if (!levels_.empty()) {
        FZ_PRINT("Execution levels (parallel within level):");
        for (int level = 0; level <= max_level_; level++) {
            FZ_PRINT("  Level %d: [%s]", level,
                     fmt_node_ids(levels_[level]).c_str());
        }
    }

    FZ_PRINT("Memory: total buffer capacity %.2f MB",
             getTotalBufferSize() / (1024.0 * 1024.0));
    FZ_PRINT("=====================================");
}

void CompressionDAG::printBufferLifetimes() const {
    FZ_PRINT("========== Buffer Lifetimes ==========");
    for (const auto& [buffer_id, buffer] : buffers_) {
        std::string consumers;
        for (size_t i = 0; i < buffer.consumer_stage_ids.size(); i++) {
            if (i) consumers += ", ";
            consumers += std::to_string(buffer.consumer_stage_ids[i]);
        }
        FZ_PRINT("[%d] %s", buffer_id, buffer.tag.c_str());
        FZ_PRINT("  size: %.1f KB  producer: %d  consumers: [%s]",
                 buffer.size / 1024.0, buffer.producer_stage_id,
                 consumers.c_str());
        FZ_PRINT("  remaining: %zu  persistent: %s  allocated: %s",
                 buffer.remaining_consumers,
                 buffer.is_persistent ? "yes" : "no",
                 buffer.is_allocated  ? "yes" : "no");
    }
    FZ_PRINT("======================================");
}

} // namespace fz
