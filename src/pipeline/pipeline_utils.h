// pipeline_utils.h — shared helpers for pipeline translation units (not public API)
#pragma once
#include "pipeline/perf.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace fz {

// Rounds x up to the next multiple of 16.
static inline size_t align16(size_t x) { return (x + 15u) & ~15u; }

// Builds a per-level timing summary from per-stage CUDA event timings.
static inline std::vector<LevelTimingResult> buildLevelTimings(
    const std::vector<StageTimingResult>& stages)
{
    std::unordered_map<int, LevelTimingResult> level_map;
    for (const auto& st : stages) {
        auto& lv = level_map[st.level];
        lv.level = st.level;
        lv.parallelism++;
        lv.elapsed_ms = std::max(lv.elapsed_ms, st.elapsed_ms);
    }
    std::vector<LevelTimingResult> levels;
    for (auto& [lvl, lv] : level_map) levels.push_back(lv);
    std::sort(levels.begin(), levels.end(),
              [](const LevelTimingResult& a, const LevelTimingResult& b) {
                  return a.level < b.level;
              });
    return levels;
}

// Encapsulates the concat buffer layout produced by writeConcatBuffer():
//   [u32 n][u64 s0]..[u64 sN-1][padding→16B][slot0 (padded)][slot1]...
// Each slot is padded to 16 bytes so data pointers stay aligned.
// The header stores actual (unpadded) sizes; slots carry the data.
struct ConcatLayout {
    static size_t headerSize(size_t n) {
        return align16(sizeof(uint32_t) + n * sizeof(uint64_t));
    }
    static size_t slotSize(size_t actual_size) {
        return align16(actual_size);
    }
};

} // namespace fz
