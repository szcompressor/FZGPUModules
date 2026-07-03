#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace fz {
namespace cli {

/**
 * @file report_json.h
 * @brief Plain-data model + writer for the `--report-json` artifact.
 *
 * The CLI populates a ReportData from its internal types (CliSettings,
 * TimingSummary, PipelinePerfResult, Metrics) and calls write_report_json().
 * The writer is the single source of truth for *derived* values (ratio,
 * bitrate, throughput, median/min/max) so the harness and the JSON never
 * disagree.  Raw counts are always emitted so the harness can recompute.
 *
 * Schema is documented in memory/report_json_spec.md.  Optional blocks are
 * omitted entirely (not emitted as null) when their `has_*`/non-empty guard is
 * false, except `error_message` which is `null` on success.
 */

constexpr const char* kReportSchemaVersion = "1.1";

/// One stage's device time within a phase, for the `stages[]` block.
struct StageTimeJson {
    std::string name;
    std::string phase;       ///< "compress" | "decompress"
    double      device_ms = 0.0;
};

/// Per-rep timing arrays for one phase. `present` gates emission.
struct PhaseTimingJson {
    bool                present = false;
    std::vector<double> device_ms;      ///< per-rep device wall time (ms)
    std::vector<double> host_wall_ms;   ///< per-rep host wall time (ms)
};

/// Everything the JSON needs, in plain types decoupled from CLI internals.
struct ReportData {
    // ── tool ──
    std::string tool_name    = "fzgmod-cli";
    std::string tool_version;            ///< empty -> "unknown"
    std::string git_sha;                 ///< empty -> field omitted

    // ── status ──
    std::string status        = "ok";    ///< "ok" | "error"
    std::string error_message;           ///< empty -> null

    // ── config (echo of resolved invocation) ──
    std::string         operation;       ///< "compress" | "decompress" | "benchmark"
    std::string         dtype;
    std::vector<size_t> dims;            ///< {nx, ny, nz}
    std::string         dim_order = "fast-to-slow";
    size_t              num_elements = 0;
    std::string         error_mode;
    double              error_bound = 0.0;
    int                 radius = 0;
    std::string         pipeline;        ///< resolved stage chain or config name
    std::string         memory_strategy;
    size_t              chunk_size = 0;
    int                 n_runs = 1;

    // ── size ──
    bool   has_size = false;
    size_t original_bytes = 0;           ///< uncompressed byte count
    size_t compressed_bytes = 0;

    // ── timing ──
    std::string     timing_method = "cuda_events_dag";
    PhaseTimingJson compress;
    PhaseTimingJson decompress;

    // ── memory ──
    bool   has_memory = false;
    size_t peak_device_bytes = 0;

    // ── quality (only when an original is available) ──
    bool   has_quality = false;
    double val_min = 0.0, val_max = 0.0, val_range = 0.0;
    double max_abs_err = 0.0, psnr_db = 0.0, nrmse = 0.0;

    // ── stages (only when profiling produced per-stage timings) ──
    std::vector<StageTimeJson> stages;

    // ── graph mode (benchmark only; omitted entirely when graph_requested is false) ──
    bool        graph_requested = false;
    bool        graph_active    = false;
    std::string graph_incompatible_reason;   // empty → field omitted
};

/**
 * Serialize @p d to a standalone JSON file at @p path.
 * Throws std::runtime_error if the file cannot be opened/written.
 */
void write_report_json(const std::string& path, const ReportData& d);

}  // namespace cli
}  // namespace fz
