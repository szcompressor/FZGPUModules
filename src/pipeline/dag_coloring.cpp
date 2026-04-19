#include "pipeline/dag.h"
#include "stage/stage.h"
#include "mem/mempool.h"
#include "log.h"
#include "cuda_check.h"

#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace fz {

// ========== Preallocation & Buffer Coloring ==========

void CompressionDAG::planPreallocation() {
    // Allocate all buffers upfront (use default stream 0 or first available)
    cudaStream_t alloc_stream = streams_.empty() ? 0 : streams_[0];
    preallocateBuffers(alloc_stream);
}

void CompressionDAG::preallocateBuffers(cudaStream_t stream) {
    // Color buffers now — sizes are guaranteed to be propagated by the time
    // the caller (Pipeline::finalize) invokes preallocateBuffers(), which
    // is always after propagateBufferSizes().  Running colorBuffers() here
    // rather than in finalize() ensures we never color 1-byte placeholders.
    if (strategy_ == MemoryStrategy::PREALLOCATE && !coloring_disabled_ && !coloring_applied_) {
        colorBuffers();
    }

    if (coloring_applied_) {
        // Allocate one region per color; assign d_ptr for each buffer from its region.
        color_region_ptrs_.assign(color_region_sizes_.size(), nullptr);
        for (size_t c = 0; c < color_region_sizes_.size(); c++) {
            if (color_region_sizes_[c] > 0)
                color_region_ptrs_[c] = mem_pool_->allocate(color_region_sizes_[c], stream);
        }
        for (auto& [buf_id, buf] : buffers_) {
            if (buf.is_external) continue;
            auto it = buffer_color_.find(buf_id);
            if (it == buffer_color_.end()) continue;
            int c = it->second;
            buf.d_ptr = color_region_ptrs_[c];
            buf.is_allocated = true;
            buf.allocated_size = color_region_sizes_[c];
        }
        size_t colored_total = 0;
        for (size_t sz : color_region_sizes_) colored_total += sz;
        current_memory_usage_ += colored_total;
        peak_memory_usage_ = std::max(peak_memory_usage_, current_memory_usage_);
        FZ_LOG(DEBUG, "PREALLOCATE (colored): %zu regions, peak %.1f KB",
               color_region_sizes_.size(), colored_total / 1024.0);
    } else {
        for (auto& [buffer_id, buffer] : buffers_) {
            allocateBuffer(buffer_id, stream);
        }
    }
}

void CompressionDAG::colorBuffers() {
    // Step 1: compute live ranges.
    // A buffer is born at the level of its producer and dies at the level of
    // its last consumer.  Pipeline-output buffers (no consumers) live until
    // max_level_ so they are never aliased with anything that writes after them.
    struct LiveRange { int born; int dies; };
    std::unordered_map<int, LiveRange> ranges;
    ranges.reserve(buffers_.size());

    std::unordered_map<int, int> node_level;
    node_level.reserve(nodes_.size());
    for (const auto* node : nodes_)
        node_level[node->id] = node->level;

    for (const auto& [buf_id, buf] : buffers_) {
        if (buf.is_external) continue;

        int born = (buf.producer_stage_id < 0) ? 0 : node_level[buf.producer_stage_id];

        int dies;
        if (buf.consumer_stage_ids.empty()) {
            dies = max_level_;
        } else {
            dies = 0;
            for (int cid : buf.consumer_stage_ids) {
                auto it = node_level.find(cid);
                if (it != node_level.end())
                    dies = std::max(dies, it->second);
            }
        }
        ranges[buf_id] = {born, dies};
    }

    // Collect colorable buffer IDs.
    std::vector<int> buf_ids;
    buf_ids.reserve(ranges.size());
    for (const auto& [buf_id, _] : ranges)
        buf_ids.push_back(buf_id);

    // Step 2: build interference graph.
    // A and B interfere iff their live ranges overlap:  born(A) <= dies(B) && born(B) <= dies(A)
    std::unordered_map<int, std::unordered_set<int>> interference;
    for (size_t i = 0; i < buf_ids.size(); i++) {
        for (size_t j = i + 1; j < buf_ids.size(); j++) {
            int a = buf_ids[i], b = buf_ids[j];
            const auto& ra = ranges[a];
            const auto& rb = ranges[b];
            if (ra.born <= rb.dies && rb.born <= ra.dies) {
                interference[a].insert(b);
                interference[b].insert(a);
            }
        }
    }

    // Step 3: greedy coloring — sort by live range length descending (longer
    // ranges first tends to minimize wasted space within each region).
    std::sort(buf_ids.begin(), buf_ids.end(), [&](int a, int b) {
        const auto& ra = ranges[a]; const auto& rb = ranges[b];
        return (ra.dies - ra.born) > (rb.dies - rb.born);
    });

    buffer_color_.clear();
    color_region_sizes_.clear();

    for (int buf_id : buf_ids) {
        // Find the lowest color not used by any already-colored neighbor.
        std::unordered_set<int> used;
        auto it = interference.find(buf_id);
        if (it != interference.end()) {
            for (int nb : it->second) {
                auto cit = buffer_color_.find(nb);
                if (cit != buffer_color_.end())
                    used.insert(cit->second);
            }
        }
        int c = 0;
        while (used.count(c)) c++;

        buffer_color_[buf_id] = c;
        if (c >= static_cast<int>(color_region_sizes_.size()))
            color_region_sizes_.resize(c + 1, 0);
        color_region_sizes_[c] = std::max(color_region_sizes_[c], buffers_[buf_id].size);
    }

    // Safety guard: multi-stream branching DAGs can create aliased buffers on
    // different streams, introducing anti-dependencies that require synthetic
    // stream wait events (not yet implemented).  If more than one stream is in
    // use and any color group contains buffers whose producer/consumer nodes
    // are on distinct streams, fall back to uncolored allocation.
    if (streams_.size() > 1) {
        bool cross_stream_alias = false;
        for (size_t c = 0; c < color_region_sizes_.size() && !cross_stream_alias; c++) {
            cudaStream_t first_stream = nullptr;
            bool first_set = false;
            for (const auto& [buf_id, color] : buffer_color_) {
                if (color != static_cast<int>(c)) continue;
                const BufferInfo& buf = buffers_[buf_id];
                // Check producer stream
                if (buf.producer_stage_id >= 0) {
                    for (const auto* node : nodes_) {
                        if (node->id == buf.producer_stage_id) {
                            if (!first_set) { first_stream = node->stream; first_set = true; }
                            else if (node->stream != first_stream) { cross_stream_alias = true; }
                            break;
                        }
                    }
                }
                if (cross_stream_alias) break;
            }
        }
        if (cross_stream_alias) {
            FZ_LOG(DEBUG, "Buffer coloring disabled: cross-stream aliases detected (multi-stream DAG)");
            buffer_color_.clear();
            color_region_sizes_.clear();
            return;  // coloring_applied_ stays false
        }
    }

    coloring_applied_ = true;

    size_t colored_total = 0;
    for (size_t sz : color_region_sizes_) colored_total += sz;
    size_t uncolored_total = 0;
    for (const auto& [buf_id, buf] : buffers_)
        if (!buf.is_external) uncolored_total += buf.size;

    FZ_LOG(DEBUG, "Buffer coloring: %zu buffers -> %zu regions (%.1f KB -> %.1f KB, %.0f%% reduction)",
           buf_ids.size(), color_region_sizes_.size(),
           uncolored_total / 1024.0, colored_total / 1024.0,
           uncolored_total > 0 ? (1.0 - (double)colored_total / uncolored_total) * 100.0 : 0.0);
}

} // namespace fz
