#include "report_json.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace fz {
namespace cli {

namespace {

// Escape a string for embedding in a JSON double-quoted value.
std::string esc(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

// Render a double as a JSON number, or `null` if not finite (JSON has no Inf/NaN).
std::string num(double v) {
    if (!std::isfinite(v)) return "null";
    std::ostringstream os;
    os.precision(8);
    os << v;
    return os.str();
}

std::string str_or_null(const std::string& s) {
    return s.empty() ? "null" : ("\"" + esc(s) + "\"");
}

struct Stats { double median, min, max; };

Stats compute_stats(std::vector<double> v) {
    Stats s{0.0, 0.0, 0.0};
    if (v.empty()) return s;
    std::sort(v.begin(), v.end());
    s.min = v.front();
    s.max = v.back();
    const size_t n = v.size();
    s.median = (n % 2 == 1) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
    return s;
}

// Emit {"median":..,"min":..,"max":..,"all":[..]} for a per-rep array.
void write_metric_block(std::ostream& o, const std::string& indent,
                        const std::vector<double>& all) {
    const Stats s = compute_stats(all);
    o << "{ \"median\": " << num(s.median)
      << ", \"min\": " << num(s.min)
      << ", \"max\": " << num(s.max)
      << ", \"all\": [";
    for (size_t i = 0; i < all.size(); ++i) {
        if (i) o << ", ";
        o << num(all[i]);
    }
    o << "] }";
    (void)indent;
}

void write_phase(std::ostream& o, const std::string& key, const PhaseTimingJson& p,
                 bool& first) {
    if (!p.present) return;
    if (!first) o << ",";
    first = false;
    o << "\n      \"" << key << "\": {\n";
    o << "        \"device_ms\":    ";
    write_metric_block(o, "        ", p.device_ms);
    o << ",\n        \"host_wall_ms\": ";
    write_metric_block(o, "        ", p.host_wall_ms);
    o << "\n      }";
}

// Throughput in GB/s (decimal, 1e9) from uncompressed bytes and a device-ms median.
double throughput_gbs(size_t uncompressed_bytes, const std::vector<double>& device_ms) {
    if (device_ms.empty()) return 0.0;
    const double med = compute_stats(device_ms).median;
    if (med <= 0.0) return 0.0;
    return static_cast<double>(uncompressed_bytes) / (med * 1e-3) / 1e9;
}

}  // namespace

void write_report_json(const std::string& path, const ReportData& d) {
    std::ostringstream o;

    o << "{\n";
    o << "  \"schema_version\": \"" << kReportSchemaVersion << "\",\n";

    // tool
    o << "  \"tool\": { \"name\": \"" << esc(d.tool_name) << "\", \"version\": \""
      << esc(d.tool_version.empty() ? "unknown" : d.tool_version) << "\"";
    if (!d.git_sha.empty()) o << ", \"git_sha\": \"" << esc(d.git_sha) << "\"";
    o << " },\n";

    // status
    o << "  \"status\": \"" << esc(d.status) << "\",\n";
    o << "  \"error_message\": " << str_or_null(d.error_message) << ",\n";

    // config
    o << "  \"config\": {\n";
    o << "    \"operation\": " << str_or_null(d.operation) << ",\n";
    o << "    \"dtype\": " << str_or_null(d.dtype) << ",\n";
    o << "    \"dims\": [";
    for (size_t i = 0; i < d.dims.size(); ++i) { if (i) o << ", "; o << d.dims[i]; }
    o << "], \"dim_order\": \"" << esc(d.dim_order) << "\", \"num_elements\": "
      << d.num_elements << ",\n";
    o << "    \"error_mode\": " << str_or_null(d.error_mode)
      << ", \"error_bound\": " << num(d.error_bound)
      << ", \"radius\": " << d.radius << ",\n";
    o << "    \"pipeline\": " << str_or_null(d.pipeline) << ",\n";
    o << "    \"memory_strategy\": " << str_or_null(d.memory_strategy)
      << ", \"coloring\": " << (d.coloring ? "true" : "false")
      << ", \"chunk_size\": " << d.chunk_size
      << ", \"n_runs\": " << d.n_runs << "\n";
    o << "  },\n";

    // size
    if (d.has_size) {
        const double ratio = d.compressed_bytes > 0
            ? static_cast<double>(d.original_bytes) / static_cast<double>(d.compressed_bytes)
            : 0.0;
        const double bitrate = d.num_elements > 0
            ? static_cast<double>(d.compressed_bytes) * 8.0 / static_cast<double>(d.num_elements)
            : 0.0;
        o << "  \"size\": {\n";
        o << "    \"original_bytes\": " << d.original_bytes << ",\n";
        o << "    \"compressed_bytes\": " << d.compressed_bytes << ",\n";
        o << "    \"ratio\": " << num(ratio) << ",\n";
        o << "    \"bitrate_bits_per_elem\": " << num(bitrate) << "\n";
        o << "  },\n";
    }

    // timing
    o << "  \"timing\": {\n";
    o << "    \"n_runs\": " << d.n_runs << ",\n";
    o << "    \"timing_method\": \"" << esc(d.timing_method) << "\"";
    bool first_phase = true;
    if (d.compress.present || d.decompress.present) o << ",";
    write_phase(o, "compress", d.compress, first_phase);
    write_phase(o, "decompress", d.decompress, first_phase);
    o << "\n  },\n";

    // throughput (derived from raw bytes / device-ms median)
    if (d.has_size && (d.compress.present || d.decompress.present)) {
        o << "  \"throughput\": {\n";
        o << "    \"compress_gbs\": "
          << num(d.compress.present ? throughput_gbs(d.original_bytes, d.compress.device_ms) : 0.0)
          << ",\n";
        o << "    \"decompress_gbs\": "
          << num(d.decompress.present ? throughput_gbs(d.original_bytes, d.decompress.device_ms) : 0.0)
          << ",\n";
        o << "    \"basis\": \"original_bytes / device_ms median (GB/s, 1e9)\"\n";
        o << "  },\n";
    }

    // memory
    if (d.has_memory) {
        o << "  \"memory\": { \"peak_device_bytes\": " << d.peak_device_bytes
          << ", \"method\": \"pipeline_pool_tracker\" },\n";
    }

    // quality
    if (d.has_quality) {
        o << "  \"quality\": {\n";
        o << "    \"val_min\": " << num(d.val_min)
          << ", \"val_max\": " << num(d.val_max)
          << ", \"val_range\": " << num(d.val_range) << ",\n";
        o << "    \"max_abs_err\": " << num(d.max_abs_err)
          << ", \"psnr_db\": " << num(d.psnr_db)
          << ", \"nrmse\": " << num(d.nrmse) << "\n";
        o << "  },\n";
    }

    // stages
    o << "  \"stages\": [";
    for (size_t i = 0; i < d.stages.size(); ++i) {
        const auto& s = d.stages[i];
        o << (i ? "," : "") << "\n    { \"name\": \"" << esc(s.name)
          << "\", \"phase\": \"" << esc(s.phase)
          << "\", \"device_ms\": " << num(s.device_ms) << " }";
    }
    o << (d.stages.empty() ? "]" : "\n  ]");
    if (d.graph_requested || !d.stage_versions.empty() || !d.run_notes.empty()) o << ",";
    o << "\n";

    // stage_versions: source fingerprint of each stage that ran.  Omitted entirely
    // when empty (a build without generated fingerprints), so a consumer can tell
    // "this build does not report versions" from "nothing changed".
    if (!d.stage_versions.empty()) {
        o << "  \"stage_versions\": {";
        bool first = true;
        for (const auto& kv : d.stage_versions) {
            o << (first ? "" : ",") << "\n    \"" << esc(kv.first) << "\": \""
              << esc(kv.second) << "\"";
            first = false;
        }
        o << "\n  }";
        if (d.graph_requested || !d.run_notes.empty()) o << ",";
        o << "\n";
    }

    // run_notes: what a stage actually did, when it differs from what was asked
    // for in a way that affects comparability (see Stage::getRunNotes()).  Omitted
    // when empty -- absence reads as "nothing surprising happened", which is the
    // overwhelmingly common case and should not cost bytes in every row.
    if (!d.run_notes.empty()) {
        o << "  \"run_notes\": {";
        bool first = true;
        for (const auto& kv : d.run_notes) {
            o << (first ? "" : ",") << "\n    \"" << esc(kv.first) << "\": [";
            for (size_t i = 0; i < kv.second.size(); ++i) {
                o << (i ? ", " : "") << "\"" << esc(kv.second[i]) << "\"";
            }
            o << "]";
            first = false;
        }
        o << "\n  }";
        if (d.graph_requested) o << ",";
        o << "\n";
    }

    // graph (benchmark only; emitted only when graph was requested)
    if (d.graph_requested) {
        o << "  \"graph\": {\n";
        o << "    \"requested\": " << (d.graph_requested ? "true" : "false") << ",\n";
        o << "    \"active\": " << (d.graph_active ? "true" : "false");
        if (!d.graph_incompatible_reason.empty()) {
            o << ",\n    \"incompatible_reason\": \"" << esc(d.graph_incompatible_reason) << "\"";
        }
        o << "\n  }\n";
    }

    o << "}\n";

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open report-json output file: " + path);
    }
    const std::string s = o.str();
    out.write(s.data(), static_cast<std::streamsize>(s.size()));
    if (!out) {
        throw std::runtime_error("Failed to write report-json file: " + path);
    }
}

}  // namespace cli
}  // namespace fz
