#include "pipeline/perf.h"

#include <algorithm>
#include <iomanip>
#include <iostream>

namespace fz {

// ──────────────────────────────────────────────────────────────────────────────
// StageTimingResult
// ──────────────────────────────────────────────────────────────────────────────

float StageTimingResult::throughput_gbs() const noexcept {
    if (elapsed_ms <= 0.0f) return 0.0f;
    return static_cast<float>(input_bytes + output_bytes) / (elapsed_ms * 1e-3f) / 1e9f;
}

// ──────────────────────────────────────────────────────────────────────────────
// PipelinePerfResult
// ──────────────────────────────────────────────────────────────────────────────

float PipelinePerfResult::throughput_gbs() const noexcept {
    if (dag_elapsed_ms <= 0.0f) return 0.0f;
    size_t uncompressed = is_compress ? input_bytes : output_bytes;
    return static_cast<float>(uncompressed) / (dag_elapsed_ms * 1e-3f) / 1e9f;
}

float PipelinePerfResult::pipeline_throughput_gbs() const noexcept {
    if (host_elapsed_ms <= 0.0f) return 0.0f;
    size_t uncompressed = is_compress ? input_bytes : output_bytes;
    return static_cast<float>(uncompressed) / (host_elapsed_ms * 1e-3f) / 1e9f;
}

void PipelinePerfResult::print(std::ostream& os) const {
    const char* mode = is_compress ? "Compress" : "Decompress";
    size_t uncompressed = is_compress ? input_bytes : output_bytes;
    size_t compressed   = is_compress ? output_bytes : input_bytes;

    os << "\n======== FZGPUModules Performance Report (" << mode << ") ========\n";

    // ── Overall summary ──────────────────────────────────────────────────────
    os << std::fixed << std::setprecision(3);
    os << "  Host elapsed:      " << std::setw(9) << host_elapsed_ms << " ms  (includes host overhead)\n";
    os << "  GPU execute:       " << std::setw(9) << dag_elapsed_ms  << " ms  (dag->execute only)\n";
    os << std::setprecision(2);
    os << "  Uncompressed:      " << std::setw(9) << (uncompressed / 1048576.0) << " MB\n";
    os << "  Compressed:        " << std::setw(9) << (compressed   / 1048576.0) << " MB\n";
    os << std::setprecision(4);
    os << "  DAG throughput:    " << std::setw(9) << throughput_gbs()
       << " GB/s  (uncompressed / dag_elapsed_ms)\n";
    os << "  Pipeline thruput:  " << std::setw(9) << pipeline_throughput_gbs()
       << " GB/s  (uncompressed / host_elapsed_ms)\n";

    // ── Level breakdown ───────────────────────────────────────────────────────
    if (!levels.empty()) {
        os << "\n  Level breakdown:\n";
        os << "    " << std::left
           << std::setw(8)  << "Level"
           << std::setw(14) << "Elapsed(ms)"
           << std::setw(12) << "Parallelism" << "\n";
        os << "    " << std::string(34, '-') << "\n";
        for (const auto& lv : levels) {
            os << std::fixed << std::setprecision(3);
            os << "    " << std::left
               << std::setw(8)  << lv.level
               << std::setw(14) << lv.elapsed_ms
               << std::setw(12) << lv.parallelism << "\n";
        }
    }

    // ── Per-stage table ───────────────────────────────────────────────────────
    if (!stages.empty()) {
        size_t name_w = 20;
        for (const auto& st : stages) {
            name_w = std::max(name_w, st.name.size() + 2);
        }

        os << "\n  Stage timings:\n";
        os << "    " << std::left
           << std::setw(static_cast<int>(name_w)) << "Stage"
           << std::setw(7)  << "Level"
           << std::setw(13) << "Elapsed(ms)"
           << std::setw(12) << "Input(KB)"
           << std::setw(12) << "Output(KB)"
           << std::setw(11) << "GB/s(R+W)" << "\n";
        os << "    " << std::string(name_w + 55, '-') << "\n";

        for (const auto& st : stages) {
            os << std::fixed << std::setprecision(3);
            os << "    " << std::left
               << std::setw(static_cast<int>(name_w)) << st.name
               << std::setw(7)  << st.level
               << std::setw(13) << st.elapsed_ms;
            os << std::setprecision(1);
            os << std::setw(12) << (st.input_bytes  / 1024.0)
               << std::setw(12) << (st.output_bytes / 1024.0);
            os << std::setprecision(4)
               << std::setw(11) << st.throughput_gbs() << "\n";
        }
    }

    os << "=============================================================\n\n";
}

} // namespace fz
