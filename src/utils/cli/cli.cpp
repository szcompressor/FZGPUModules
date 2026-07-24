#include "cuda_check.h"
#include "fzgpumodules.h"
#include "pipeline/config.h"
#include "report_json.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <cmath>
#include <memory>

using namespace fz;

namespace {

using OptionMap = std::unordered_map<std::string, std::string>;

constexpr size_t kDefaultChunkSize = 16384;
constexpr int kDefaultBenchmarkRuns = 10;

enum class CliOperation {
    None,
    Compress,
    Decompress,
    Benchmark
};

struct CliSettings {
    CliOperation operation = CliOperation::None;
    std::string input_path;
    std::string output_path;
    std::string config_path;
    std::string original_path;
    std::string report_json_path;   // --report-json <file>; empty = disabled

    size_t nx = 0;
    size_t ny = 1;
    size_t nz = 1;

    std::string type = "f32";
    // Ordered stage pipeline spec, e.g. "lorenzo->bitshuffle->rze"
    std::string stages = "lorenzo->bitshuffle->rze";

    float error_bound = 1e-3f;
    ErrorBoundMode error_mode = ErrorBoundMode::REL;
    int quant_radius = 32768;

    MemoryStrategy strategy = MemoryStrategy::PREALLOCATE;
    float pool_multiplier = 3.0f;

    bool warmup = false;
    bool profile = false;
    bool report = false;

    int  verbose_level = 0;    // 0=off, 1=INFO, 2=DEBUG, 3=TRACE
    bool print_pipeline = false;
    bool bounds_check = false;
    bool use_graph = false;

    size_t chunk_size = kDefaultChunkSize;

    int benchmark_runs = kDefaultBenchmarkRuns;
};

struct TimingSummary {
    std::vector<double> host_ms;
    std::vector<float> dag_ms;

    void add(double host, float dag) {
        host_ms.push_back(host);
        dag_ms.push_back(dag);
    }
    bool empty() const { return host_ms.empty(); }
};

static std::string trim(const std::string& s) {
    size_t begin = 0;
    while (begin < s.size() && std::isspace(static_cast<unsigned char>(s[begin])) != 0) {
        ++begin;
    }
    size_t end = s.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(s[end - 1])) != 0) {
        --end;
    }
    return s.substr(begin, end - begin);
}

static std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

static std::string normalize_key(std::string key) {
    key = to_lower(trim(std::move(key)));
    std::replace(key.begin(), key.end(), '_', '-');
    return key;
}

static size_t checked_mul(size_t a, size_t b, const char* label) {
    if (a == 0 || b == 0) return 0;
    if (a > (std::numeric_limits<size_t>::max() / b)) {
        throw std::runtime_error(std::string("Size overflow while computing ") + label);
    }
    return a * b;
}

template <typename T>
static T parse_integer(const std::string& text, const char* name) {
    try {
        size_t idx = 0;
        unsigned long long v = std::stoull(text, &idx);
        if (idx != text.size()) throw std::runtime_error("trailing characters");
        if (v > static_cast<unsigned long long>(std::numeric_limits<T>::max())) {
            throw std::runtime_error("value out of range");
        }
        return static_cast<T>(v);
    } catch (const std::exception&) {
        throw std::runtime_error(std::string("Invalid value for ") + name + ": '" + text + "'");
    }
}

static float parse_float(const std::string& text, const char* name) {
    try {
        size_t idx = 0;
        float v = std::stof(text, &idx);
        if (idx != text.size()) throw std::runtime_error("trailing characters");
        return v;
    } catch (const std::exception&) {
        throw std::runtime_error(std::string("Invalid value for ") + name + ": '" + text + "'");
    }
}

static bool parse_bool(const std::string& text, const char* name) {
    const std::string lower = to_lower(trim(text));
    if (lower == "1" || lower == "true" || lower == "yes" || lower == "on") return true;
    if (lower == "0" || lower == "false" || lower == "no" || lower == "off") return false;
    throw std::runtime_error(std::string("Invalid boolean for ") + name + ": '" + text + "'");
}

static ErrorBoundMode parse_error_mode(const std::string& text) {
    const std::string mode = to_lower(trim(text));
    if (mode == "rel") return ErrorBoundMode::REL;
    if (mode == "abs") return ErrorBoundMode::ABS;
    if (mode == "noa") return ErrorBoundMode::NOA;
    throw std::runtime_error("Unknown error mode: '" + text + "' (expected rel|abs|noa)");
}

static MemoryStrategy parse_strategy(const std::string& text) {
    const std::string mode = to_lower(trim(text));
    if (mode == "minimal") return MemoryStrategy::MINIMAL;
    if (mode == "preallocate") return MemoryStrategy::PREALLOCATE;
    throw std::runtime_error("Unknown strategy: '" + text + "' (expected minimal|preallocate)");
}

static OptionMap parse_option_tokens(int argc, char** argv, int start_index) {
    OptionMap opts;

    for (int i = start_index; i < argc; ++i) {
        std::string token = argv[i];

        // -v/-vv/-vvv: verbose level shortcuts (set opts directly, highest seen wins)
        if (token == "-v")   { if (opts.count("verbose") == 0 || opts["verbose"] < "1") opts["verbose"] = "1"; continue; }
        if (token == "-vv")  { if (opts.count("verbose") == 0 || opts["verbose"] < "2") opts["verbose"] = "2"; continue; }
        if (token == "-vvv") { opts["verbose"] = "3"; continue; }

        if (token == "-h" || token == "--help") {
            opts["help"] = "true";
            continue;
        }

        const size_t eq = token.find('=');
        std::string key = token;
        std::string value;
        bool has_value = false;

        if (eq != std::string::npos && token.rfind("-", 0) == 0) {
            key = token.substr(0, eq);
            value = token.substr(eq + 1);
            has_value = true;
        }

        if (key.rfind("--", 0) == 0) {
            key = key.substr(2);
            // Long flag aliases
            if (key == "len" || key == "xyz") key = "dims";
            else if (key == "eb") key = "error-bound";
            else if (key == "compress") key = "z";
            else if (key == "decompress") key = "x";
            else if (key == "origin") key = "compare";
            else if (key == "histogram") key = "hist";
            else if (key == "codec1") key = "codec";
            else if (key == "dtype") key = "type";
        } else if (key.rfind("-", 0) == 0) {
            key = key.substr(1);
            if (key == "i") key = "input";
            else if (key == "o") key = "output";
            else if (key == "c") key = "config";
            else if (key == "e") key = "error-bound";
            else if (key == "m") key = "mode";
            else if (key == "l") key = "dims";
            else if (key == "t") key = "type";
            else if (key == "r") key = "radius";
            else if (key == "R") key = "report";
        } else {
            throw std::runtime_error("Unexpected positional argument: '" + token + "'");
        }

        key = normalize_key(key);

        // --verbose / --verbose=N / --verbose=level : optional-value option
        if (key == "verbose") {
            opts[key] = has_value ? value : "1";
            continue;
        }

        const bool is_flag =
            (key == "z") ||
            (key == "x") ||
            (key == "b" || key == "benchmark") ||
            (key == "report") ||
            (key == "warmup") ||
            (key == "profile") ||
            (key == "print-pipeline") ||
            (key == "bounds-check") ||
            (key == "graph");

        if (is_flag) {
            opts[key] = "true";
            if (has_value && (value != "true" && value != "1")) {
                throw std::runtime_error("Flag '" + key + "' does not take a value");
            }
            continue;
        }

        if (!has_value) {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for option '" + token + "'");
            }
            value = trim(argv[++i]);
        }
        opts[key] = value;
    }

    return opts;
}

static bool contains(const OptionMap& opts, const std::string& key) {
    return opts.find(key) != opts.end();
}

static std::string get_optional(const OptionMap& opts, const std::string& key, const std::string& default_val = "") {
    auto it = opts.find(key);
    return it != opts.end() ? it->second : default_val;
}

static std::vector<uint8_t> read_binary_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in.is_open()) throw std::runtime_error("Failed to open input file: " + path);

    const std::streamsize end_pos = in.tellg();
    if (end_pos < 0) throw std::runtime_error("Failed to determine input size: " + path);

    std::vector<uint8_t> bytes(static_cast<size_t>(end_pos));
    in.seekg(0, std::ios::beg);
    if (!bytes.empty()) {
        in.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        if (!in) throw std::runtime_error("Failed to read full input file: " + path);
    }
    return bytes;
}

static void write_binary_file(const std::string& path, const uint8_t* data, size_t size) {
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) throw std::runtime_error("Failed to open output file: " + path);

    if (size > 0) {
        out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(size));
        if (!out) throw std::runtime_error("Failed to write output file: " + path);
    }
}

static void parse_dims(const std::string& dims_str, size_t& nx, size_t& ny, size_t& nz) {
    std::string s = dims_str;
    std::replace(s.begin(), s.end(), 'x', ' ');
    std::replace(s.begin(), s.end(), 'X', ' ');
    std::replace(s.begin(), s.end(), '-', ' ');
    std::replace(s.begin(), s.end(), '*', ' ');

    std::stringstream ss(s);
    nx = ny = nz = 1;
    if (!(ss >> nx)) throw std::runtime_error("Invalid dimensions format: '" + dims_str + "'");
    if (ss >> ny) {
        ss >> nz;
    } else {
        ny = nz = 1;
    }
}

static void apply_common_options(const OptionMap& opts, CliSettings* s) {
    s->input_path = get_optional(opts, "input");
    s->output_path = get_optional(opts, "output");
    s->config_path = get_optional(opts, "config");
    s->original_path = get_optional(opts, "compare");
    s->report_json_path = get_optional(opts, "report-json");

    if (contains(opts, "dims")) parse_dims(opts.at("dims"), s->nx, s->ny, s->nz);
    if (contains(opts, "type")) s->type = to_lower(opts.at("type"));
    if (contains(opts, "stages")) s->stages = opts.at("stages");
    if (contains(opts, "error-bound")) s->error_bound = parse_float(opts.at("error-bound"), "error-bound");
    if (contains(opts, "mode")) s->error_mode = parse_error_mode(opts.at("mode"));
    if (contains(opts, "radius")) s->quant_radius = parse_integer<int>(opts.at("radius"), "radius");
    if (contains(opts, "strategy")) s->strategy = parse_strategy(opts.at("strategy"));
    if (contains(opts, "pool-mult")) s->pool_multiplier = parse_float(opts.at("pool-mult"), "pool-mult");
    if (contains(opts, "chunk-size")) s->chunk_size = parse_integer<size_t>(opts.at("chunk-size"), "chunk-size");
    if (contains(opts, "runs")) s->benchmark_runs = parse_integer<int>(opts.at("runs"), "runs");

    s->warmup = contains(opts, "warmup") && parse_bool(opts.at("warmup"), "warmup");
    s->profile = contains(opts, "profile") && parse_bool(opts.at("profile"), "profile");
    s->report = contains(opts, "report") && parse_bool(opts.at("report"), "report");
    s->print_pipeline = contains(opts, "print-pipeline") && parse_bool(opts.at("print-pipeline"), "print-pipeline");
    s->bounds_check   = contains(opts, "bounds-check")   && parse_bool(opts.at("bounds-check"),   "bounds-check");
    s->use_graph      = contains(opts, "graph")          && parse_bool(opts.at("graph"),           "graph");

    if (contains(opts, "verbose")) {
        const std::string& v = opts.at("verbose");
        if (v == "true" || v == "1" || v == "info") s->verbose_level = 1;
        else if (v == "2" || v == "debug")           s->verbose_level = 2;
        else if (v == "3" || v == "trace")           s->verbose_level = 3;
        else s->verbose_level = parse_integer<int>(v, "verbose");
    }

    if (s->verbose_level > 0) {
        fz::LogLevel log_level = fz::LogLevel::INFO;
        if      (s->verbose_level >= 3) log_level = fz::LogLevel::TRACE;
        else if (s->verbose_level >= 2) log_level = fz::LogLevel::DEBUG;
        fz::Logger::enableStderr(log_level);
    }

    if (contains(opts, "z")) s->operation = CliOperation::Compress;
    if (contains(opts, "x")) s->operation = CliOperation::Decompress;
    if (contains(opts, "b") || contains(opts, "benchmark")) s->operation = CliOperation::Benchmark;
}

static size_t validate_or_infer_dims(CliSettings* s, size_t input_bytes, size_t element_size) {
    const size_t elements = input_bytes / element_size;
    if (input_bytes % element_size != 0) {
        throw std::runtime_error("Input file size is not a multiple of the element type size");
    }

    if (s->nx == 0) {
        s->nx = elements;
        s->ny = 1;
        s->nz = 1;
    } else {
        const size_t specified = checked_mul(s->nx, checked_mul(s->ny, s->nz, "ny * nz"), "nx * ny * nz");
        if (specified != elements) {
            throw std::runtime_error("Specified dimensions (" + std::to_string(s->nx) + "x" +
                                     std::to_string(s->ny) + "x" + std::to_string(s->nz) + " = " +
                                     std::to_string(specified) + ") do not match input file size (" +
                                     std::to_string(elements) + " elements)");
        }
    }
    return input_bytes;
}

struct Metrics {
    double max_err = 0.0;
    double psnr = 0.0;
    double nrmse = 0.0;
    double val_min = 0.0;
    double val_max = 0.0;
    double val_range = 0.0;
};

template <typename T>
static Metrics calc_metrics(const std::vector<uint8_t>& orig_bytes, const std::vector<uint8_t>& recon_bytes) {
    const T* orig = reinterpret_cast<const T*>(orig_bytes.data());
    const T* recon = reinterpret_cast<const T*>(recon_bytes.data());
    size_t n = orig_bytes.size() / sizeof(T);
    
    Metrics m;
    if (n == 0) return m;

    m.val_min = static_cast<double>(orig[0]);
    m.val_max = static_cast<double>(orig[0]);
    double mse_sum = 0.0;

    for (size_t i = 0; i < n; ++i) {
        double o = static_cast<double>(orig[i]);
        double r = static_cast<double>(recon[i]);
        double diff = std::abs(o - r);
        
        if (diff > m.max_err) m.max_err = diff;
        if (o < m.val_min) m.val_min = o;
        if (o > m.val_max) m.val_max = o;
        
        mse_sum += diff * diff;
    }
    
    m.val_range = m.val_max - m.val_min;
    double mse = mse_sum / n;
    
    if (mse == 0.0) {
        m.psnr = std::numeric_limits<double>::infinity();
        m.nrmse = 0.0;
    } else {
        if (m.val_range > 0) {
            m.psnr = 20.0 * std::log10(m.val_range) - 10.0 * std::log10(mse);
            m.nrmse = std::sqrt(mse) / m.val_range;
        } else {
            m.psnr = 0.0;
            m.nrmse = 0.0;
        }
    }
    
    return m;
}

// Split a "->" separated stages string into a vector of lowercased names.
static std::vector<std::string> parse_stages(const std::string& stages_str) {
    std::vector<std::string> result;
    std::string remaining = stages_str;
    const std::string sep = "->";
    size_t pos = 0;
    while ((pos = remaining.find(sep)) != std::string::npos) {
        std::string tok = trim(remaining.substr(0, pos));
        if (!tok.empty()) result.push_back(to_lower(tok));
        remaining = remaining.substr(pos + sep.size());
    }
    std::string last = trim(remaining);
    if (!last.empty()) result.push_back(to_lower(last));
    return result;
}

template <typename T>
static void build_dynamic_linear_pipeline(Pipeline* pipeline, const CliSettings& s) {
    pipeline->setDims(s.nx, s.ny, s.nz);
    pipeline->setWarmupOnFinalize(s.warmup);
    pipeline->enableProfiling(s.profile);

    Stage* last_stage = nullptr;
    // Track whether the previous stage emits via the "codes" port (predictors)
    // so codecs that care about element width can pick the right value.
    bool last_is_codes_port = false;

    auto connect_next = [&](Stage* next, bool emits_codes = false) {
        if (last_stage) {
            pipeline->connect(next, last_stage, last_is_codes_port ? "codes" : "output");
        }
        last_stage = next;
        last_is_codes_port = emits_codes;
    };

    const std::vector<std::string> stage_list = parse_stages(s.stages);
    if (stage_list.empty()) {
        throw std::runtime_error("--stages is empty; provide at least one stage name");
    }

    for (const std::string& name : stage_list) {
        if (name == "lorenzo") {
            auto* lrz = pipeline->addStage<LorenzoQuantStage<T, uint16_t>>();
            lrz->setErrorBound(s.error_bound);
            lrz->setErrorBoundMode(s.error_mode);
            lrz->setQuantRadius(s.quant_radius);
            lrz->setOutlierCapacity(0.10f);
            lrz->setZigzagCodes(true);
            connect_next(lrz, /*emits_codes=*/true);
        } else if (name == "quantizer") {
            auto* quant = pipeline->addStage<QuantizerStage<T, uint16_t>>();
            quant->setErrorBound(s.error_bound);
            quant->setErrorBoundMode(s.error_mode);
            quant->setQuantRadius(s.quant_radius);
            quant->setOutlierCapacity(0.05f);
            quant->setZigzagCodes(true);
            connect_next(quant, /*emits_codes=*/true);
        } else if (name == "bitshuffle" || name == "bshuf") {
            auto* bshuf = pipeline->addStage<BitshuffleStage>();
            bshuf->setBlockSize(s.chunk_size);
            // If the upstream stage was a predictor, codes are uint16_t (2 bytes);
            // otherwise fall back to the element width of the input type.
            bshuf->setElementWidth(last_is_codes_port ? 2 : static_cast<int>(sizeof(T)));
            connect_next(bshuf);
        } else if (name == "rze" || name == "rze1" || name == "rze2" ||
                   name == "rze4" || name == "rze8") {
            // Optional trailing digit selects the LC word granularity (default 1).
            auto* rze = pipeline->addStage<RZEStage>();
            rze->setChunkSize(s.chunk_size);
            rze->setWordSize(name.size() > 3 ? static_cast<size_t>(name[3] - '0') : 1);
            connect_next(rze);
        } else if (name == "rre" || name == "rre1" || name == "rre2" ||
                   name == "rre4" || name == "rre8") {
            // Optional trailing digit selects the LC word granularity (default 1).
            auto* rre = pipeline->addStage<RREStage>();
            rre->setChunkSize(s.chunk_size);
            rre->setWordSize(name.size() > 3 ? static_cast<size_t>(name[3] - '0') : 1);
            connect_next(rre);
        } else if (name == "diff" || name == "difference") {
            auto* diff = pipeline->addStage<DifferenceStage<uint16_t>>();
            diff->setChunkSize(s.chunk_size);
            connect_next(diff);
        } else if (name == "rle" || name == "rle1" || name == "rle2" ||
                   name == "rle4" || name == "rle8") {
            // Optional trailing digit selects the word size (default 2, matching
            // the historical uint16_t default); mirrors rze[1|2|4|8]/rre[1|2|4|8].
            Stage* rle = nullptr;
            const int width = name.size() > 3 ? (name[3] - '0') : 2;
            switch (width) {
                case 1: rle = pipeline->addStage<RLEStage<uint8_t>>();  break;
                case 2: rle = pipeline->addStage<RLEStage<uint16_t>>(); break;
                case 4: rle = pipeline->addStage<RLEStage<uint32_t>>(); break;
                case 8: rle = pipeline->addStage<RLEStage<uint64_t>>(); break;
            }
            connect_next(rle);
        } else if (name == "huffman" || name == "huf") {
            // When following a predictor with zigzag_codes=true, codes are in [0, 2*radius-2];
            // set bklen=2*quant_radius to cover the full symbol range exactly.
            // When not following a predictor, fall back to 1024; use TOML for custom bklen.
            const uint32_t bklen = last_is_codes_port
                ? static_cast<uint32_t>(2 * s.quant_radius)
                : 1024u;
            auto* huf = pipeline->addStage<HuffmanStage<uint16_t>>();
            huf->setBklen(bklen);
            connect_next(huf);
        } else if (name == "ans") {
#if !defined(FZGMOD_BACKEND_HIP)
            auto* ans = pipeline->addStage<ANSStage>();
            connect_next(ans);
#else
            // ANSStage (vendored dietgpu) is excluded on HIP -- see the
            // matching guard in src/pipeline/config.cpp.
            throw std::runtime_error("'ans' stage is not available on the HIP backend");
#endif
        } else if (name == "adm") {
            auto* adm = pipeline->addStage<ADMStage>();
            // If upstream emits uint16_t codes (predictor), use U16; otherwise U16 is the default.
            adm->setDtype(ADMDtype::U16);
            connect_next(adm);
        } else if (name == "none") {
            // explicit no-op
        } else {
            throw std::runtime_error(
                "Unknown stage '" + name + "' in --stages. "
                "Supported: lorenzo, quantizer, bitshuffle, rze[1|2|4|8], rre[1|2|4|8], diff, rle[1|2|4|8], huffman, ans, adm");
        }
    }

    if (s.bounds_check) pipeline->enableBoundsCheck(true);
    pipeline->finalize();
    if (s.print_pipeline) pipeline->printPipeline();
}

static void print_root_usage(const char* argv0) {
    std::cout
        << "Name: FZModules GPU Compression Library\n\n"
        << "Synopsis: (Basic usage)\n"
        << "  " << argv0 << " -t f32 -m rel -e 1e-3 -i {data} -l 300x100x200 -z --report\n"
        << "          ------ ------ ------- --------- -------------- -- --------\n"
        << "           Type   Mode   Error   Input     Dim-fast-slow  zip  Report\n"
        << "  " << argv0 << " -i {compressed} -x --compare {original} --report\n"
        << "          --------------- -- -------------------- -------------\n"
        << "           Input file   Unzip   Compare original     Report\n\n"
        << "Operation Modes (Pick One):\n"
        << "  -z, --compress            Compress mode\n"
        << "  -x, --decompress          Decompress mode\n"
        << "  -b, --benchmark           Benchmark mode\n\n"
        << "General Options:\n"
        << "  -h, --help                        Show this help message and exit\n"
        << "  -c, --config <file.toml>          Load pipeline from TOML config\n\n"
        << "Analysis Options:\n"
        << "  -R, --report                      Generate a compression/decompression report\n"
        << "  --report-json <file>              Write a machine-readable JSON report to <file>\n"
        << "  --compare <original>              Compare decompressed output with original\n"
        << "  --profile                         Print per-stage GPU timing table\n"
        << "  --graph                           Use CUDA Graph mode for benchmark (silently falls back if pipeline is incompatible)\n"
        << "  --print-pipeline                  Print pipeline stage graph after finalize\n\n"
        << "Diagnostic Options:\n"
        << "  -v                                Verbose: enable INFO-level library logging\n"
        << "  -vv                               Verbose: enable DEBUG-level library logging\n"
        << "  -vvv                              Verbose: enable TRACE-level library logging\n"
        << "  --verbose[=N]                     Verbose level (1=INFO, 2=DEBUG, 3=TRACE)\n"
        << "  --bounds-check                    Enable runtime buffer overrun detection\n\n"
        << "Compression Parameters (for dynamic linear pipelines):\n"
        << "  --stages \"<s1->s2->...>\"          Ordered pipeline stages (default: \"lorenzo->bitshuffle->rze\")\n"
        << "                                    NOTE: Wrap in quotes to prevent shell redirection ('->')\n"
        << "                                    Supported stages: lorenzo, quantizer, bitshuffle,\n"
        << "                                                      rze[1|2|4|8], rre[1|2|4|8], diff, rle[1|2|4|8], huffman, ans, adm\n"
        << "  -m, --mode <rel,abs,noa>          Error bound mode (default: rel)\n"
        << "  -e, --error-bound <val>           Error bound value (default: 1e-3)\n"
        << "  -t, --type <f32,f64>              Data type (default: f32)\n"
        << "  -r, --radius <value>              Quantization radius (default: 32768)\n"
        << "  --chunk-size <bytes>              Encoder chunk size (default: 16384)\n"
        << "  -l, --len <x>x<y>x<z>            Dimensions (e.g., 100x200x300)\n\n"
        << "Input/Output Options:\n"
        << "  -i, --input <filename>            Input file\n"
        << "  -o, --output <filename>           Output file\n";
}

static void print_summary(const std::string& label, const TimingSummary& stats, size_t bytes) {
    if (stats.empty()) return;
    const int n = static_cast<int>(stats.host_ms.size());
    const double mean_host = std::accumulate(stats.host_ms.begin(), stats.host_ms.end(), 0.0) / n;
    const float mean_dag = std::accumulate(stats.dag_ms.begin(), stats.dag_ms.end(), 0.0f) / n;
    const auto tput_gbs = [bytes](double ms) -> double {
        return ms > 0.0 ? static_cast<double>(bytes) / (ms * 1e-3) / 1e9 : 0.0;
    };

    std::cout << "\n[benchmark] " << label << " summary\n"
              << "  runs:           " << n << "\n"
              << "  host ms:        mean=" << std::fixed << std::setprecision(3) << mean_host << "\n"
              << "  dag ms:         mean=" << mean_dag << "\n"
              << "  throughput:     " << std::setprecision(2) << tput_gbs(mean_host) << " GB/s (host mean)\n";
}

// ── --report-json helpers ──────────────────────────────────────────────────

static std::string error_mode_str(ErrorBoundMode m) {
    switch (m) {
        case ErrorBoundMode::REL: return "rel";
        case ErrorBoundMode::ABS: return "abs";
        case ErrorBoundMode::NOA: return "noa";
    }
    return "rel";
}

static std::string strategy_str(MemoryStrategy s) {
    return s == MemoryStrategy::MINIMAL ? "MINIMAL" : "PREALLOCATE";
}

static const char* operation_str(CliOperation op) {
    switch (op) {
        case CliOperation::Compress:   return "compress";
        case CliOperation::Decompress: return "decompress";
        case CliOperation::Benchmark:  return "benchmark";
        default:                       return "";
    }
}

// Populate the config/tool fields shared by every operation's report.
static fz::cli::ReportData make_report_base(const CliSettings& s, const char* op) {
    fz::cli::ReportData d;
#ifdef FZGMOD_VERSION
    d.tool_version = FZGMOD_VERSION;
#endif
#ifdef FZGMOD_GIT_SHA
    d.git_sha = FZGMOD_GIT_SHA;
#endif
    d.operation       = op;
    d.dtype           = s.type;
    d.dims            = {s.nx, s.ny, s.nz};
    d.num_elements    = checked_mul(s.nx, checked_mul(s.ny, s.nz, "ny*nz"), "nx*ny*nz");
    d.error_mode      = error_mode_str(s.error_mode);
    d.error_bound     = s.error_bound;
    d.radius          = s.quant_radius;
    d.pipeline        = s.config_path.empty() ? s.stages : s.config_path;
    d.memory_strategy = strategy_str(s.strategy);
    d.chunk_size      = s.chunk_size;
    return d;
}

// Append per-stage device timings from a perf result into the report.
static void append_stages(std::vector<fz::cli::StageTimeJson>& out,
                          const fz::PipelinePerfResult& r, const char* phase) {
    for (const auto& st : r.stages) {
        out.push_back({st.name, phase, static_cast<double>(st.elapsed_ms)});
    }
}

// Copy quality metrics into the report.
static void set_quality(fz::cli::ReportData& d, const Metrics& m) {
    d.has_quality = true;
    d.val_min     = m.val_min;
    d.val_max     = m.val_max;
    d.val_range   = m.val_range;
    d.max_abs_err = m.max_err;
    d.psnr_db     = m.psnr;
    d.nrmse       = m.nrmse;
}

// Best-effort emit; report-json failures must not mask a successful compression.
static void try_write_report(const std::string& path, const fz::cli::ReportData& d) {
    if (path.empty()) return;
    try {
        fz::cli::write_report_json(path, d);
    } catch (const std::exception& e) {
        std::cerr << "[fzgmod-cli] warning: failed to write report-json: " << e.what() << "\n";
    }
}

// Emit a status:"error" report so the harness records a structured failure
// instead of inferring a crash from a missing file.  Best-effort; config echo
// reflects whatever was resolved before the failure.
static void emit_error_report(const std::string& path, const CliSettings& s,
                              const std::string& msg) {
    if (path.empty()) return;
    fz::cli::ReportData d = make_report_base(s, operation_str(s.operation));
    d.status        = "error";
    d.error_message = msg;
    try_write_report(path, d);
}

static int run_compress(CliSettings s) {
    if (s.input_path.empty()) throw std::runtime_error("-z (compress) requires -i/--input");
    if (s.output_path.empty()) s.output_path = s.input_path + ".fzm";

    size_t element_size = (s.type == "f64" || s.type == "i64") ? 8 : 4;
    std::vector<uint8_t> input_bytes = read_binary_file(s.input_path);
    if (input_bytes.empty()) throw std::runtime_error("Input file is empty: " + s.input_path);

    const size_t payload_bytes = validate_or_infer_dims(&s, input_bytes.size(), element_size);

    // --report-json needs device timing + per-stage breakdown, so force profiling
    // on (without enabling the stdout perf table, which stays gated on s.profile).
    const bool want_json = !s.report_json_path.empty();
    const bool prof = s.profile || want_json;

    void* d_input = nullptr;
    try {
        FZ_CUDA_CHECK(cudaMalloc(&d_input, payload_bytes));
        FZ_CUDA_CHECK(cudaMemcpy(d_input, input_bytes.data(), payload_bytes, cudaMemcpyHostToDevice));

        std::unique_ptr<Pipeline> pipeline;
        if (!s.config_path.empty()) {
            pipeline = std::make_unique<Pipeline>(payload_bytes, s.strategy, s.pool_multiplier);
            pipeline->setDims(s.nx, s.ny, s.nz);
            pipeline->setWarmupOnFinalize(s.warmup);
            pipeline->enableProfiling(prof);
            if (s.bounds_check) pipeline->enableBoundsCheck(true);
            pipeline->loadConfig(s.config_path);
            if (s.print_pipeline) pipeline->printPipeline();
        } else {
            pipeline = std::make_unique<Pipeline>(payload_bytes, s.strategy, s.pool_multiplier);
            CliSettings sb = s;
            sb.profile = prof;
            if (s.type == "f32") {
                build_dynamic_linear_pipeline<float>(pipeline.get(), sb);
            } else if (s.type == "f64") {
                build_dynamic_linear_pipeline<double>(pipeline.get(), sb);
            } else {
                throw std::runtime_error("Dynamic builder only supports f32/f64 currently. Use TOML for others.");
            }
        }

        const auto t0 = std::chrono::high_resolution_clock::now();
        void* d_compressed = nullptr;
        size_t compressed_size = 0;
        pipeline->compress(d_input, payload_bytes, &d_compressed, &compressed_size, 0);
        FZ_CUDA_CHECK(cudaDeviceSynchronize());
        const auto t1 = std::chrono::high_resolution_clock::now();
        double host_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        pipeline->writeToFile(s.output_path, 0);

        if (s.profile) {
            pipeline->getLastPerfResult().print(std::cout);
        }

        if (s.report) {
            double ratio = static_cast<double>(payload_bytes) / static_cast<double>(compressed_size);
            double tput = static_cast<double>(payload_bytes) / (host_ms * 1e-3) / 1e9;
            std::cout << "\n[Compress Report]\n"
                      << "  Input size:      " << payload_bytes << " bytes\n"
                      << "  Compressed size: " << compressed_size << " bytes\n"
                      << "  Ratio:           " << std::fixed << std::setprecision(2) << ratio << "x\n"
                      << "  Time:            " << host_ms << " ms\n"
                      << "  Throughput:      " << tput << " GB/s\n"
                      << "  Peak device mem: " << std::setprecision(1)
                      << pipeline->getPeakMemoryUsage() / 1024.0 / 1024.0 << " MB\n";
        }

        if (want_json) {
            const auto& perf = pipeline->getLastPerfResult();
            fz::cli::ReportData d = make_report_base(s, "compress");
            d.has_size          = true;
            d.original_bytes    = payload_bytes;
            d.compressed_bytes  = compressed_size;
            d.has_memory        = true;
            d.peak_device_bytes = pipeline->getPeakMemoryUsage();
            d.n_runs            = 1;
            d.compress.present      = true;
            d.compress.device_ms    = {static_cast<double>(perf.dag_elapsed_ms)};
            d.compress.host_wall_ms = {host_ms};
            append_stages(d.stages, perf, "compress");
            try_write_report(s.report_json_path, d);
        }
    } catch (...) {
        if (d_input) FZ_CUDA_CHECK_WARN(cudaFree(d_input));
        throw;
    }
    if (d_input) FZ_CUDA_CHECK(cudaFree(d_input));
    return 0;
}

static int run_decompress(CliSettings s) {
    if (s.input_path.empty()) throw std::runtime_error("-x (decompress) requires -i/--input");
    if (s.output_path.empty() && !s.report && s.original_path.empty()) {
        throw std::runtime_error("-x (decompress) requires -o/--output unless just comparing/reporting");
    }

    const bool want_json = !s.report_json_path.empty();

    void* d_output = nullptr;
    size_t output_size = 0;

    try {
        std::vector<uint8_t> orig;
        if (!s.original_path.empty()) {
            orig = read_binary_file(s.original_path);
        }

        PipelinePerfResult decomp_perf;
        const auto t0 = std::chrono::high_resolution_clock::now();
        Pipeline::decompressFromFile(s.input_path, &d_output, &output_size, 0,
                                     (s.profile || want_json) ? &decomp_perf : nullptr);
        FZ_CUDA_CHECK(cudaDeviceSynchronize());
        const auto t1 = std::chrono::high_resolution_clock::now();
        double host_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // Truncate to the original size if we have it (to remove chunk padding)
        size_t usable_size = output_size;
        if (!orig.empty() && orig.size() < output_size) {
            usable_size = orig.size();
        }

        std::vector<uint8_t> host(usable_size);
        if (usable_size > 0) {
            FZ_CUDA_CHECK(cudaMemcpy(host.data(), d_output, usable_size, cudaMemcpyDeviceToHost));
        }

        if (!s.output_path.empty()) {
            write_binary_file(s.output_path, host.data(), host.size());
        }

        if (s.profile) {
            decomp_perf.print(std::cout);
        }

        // Compute quality metrics once (used by both the text report and JSON).
        Metrics m;
        bool has_m = false;
        if (!orig.empty() && orig.size() == host.size()) {
            if (s.type == "f32") m = calc_metrics<float>(orig, host);
            else if (s.type == "f64") m = calc_metrics<double>(orig, host);
            else if (s.type == "i32") m = calc_metrics<int32_t>(orig, host);
            else if (s.type == "i64") m = calc_metrics<int64_t>(orig, host);
            has_m = true;
        }

        if (s.report || !s.original_path.empty()) {
            double tput = static_cast<double>(usable_size) / (host_ms * 1e-3) / 1e9;
            std::cout << "\n[Decompress Report]\n"
                      << "  Output size:     " << usable_size << " bytes\n";
            if (usable_size != output_size) {
                std::cout << "  (Padded size:    " << output_size << " bytes, truncated to match original)\n";
            }
            std::cout << "  Time:            " << std::fixed << std::setprecision(3) << host_ms << " ms\n"
                      << "  Throughput:      " << std::setprecision(2) << tput << " GB/s\n";

            if (!orig.empty()) {
                if (orig.size() != host.size()) {
                    std::cout << "  Compare error:   Size mismatch! Original=" << orig.size()
                              << ", Reconstructed=" << host.size() << "\n";
                } else {
                    std::cout << "  Value Range:     [" << std::scientific << m.val_min << ", " << m.val_max << "] (Span: " << m.val_range << ")\n"
                              << "  Max Abs Error:   " << std::scientific << m.max_err << "\n"
                              << "  PSNR:            " << std::fixed << std::setprecision(2) << m.psnr << " dB\n"
                              << "  NRMSE:           " << std::scientific << m.nrmse << "\n";
                }
            }
        }

        if (want_json) {
            const size_t element_size = (s.type == "f64" || s.type == "i64") ? 8 : 4;
            fz::cli::ReportData d = make_report_base(s, "decompress");
            d.has_size          = true;
            d.original_bytes    = usable_size;                  // uncompressed
            d.compressed_bytes  = decomp_perf.input_bytes;      // = header compressed_size
            d.num_elements      = element_size ? usable_size / element_size : 0;
            d.n_runs            = 1;
            d.decompress.present      = true;
            d.decompress.device_ms    = {static_cast<double>(decomp_perf.dag_elapsed_ms)};
            d.decompress.host_wall_ms = {host_ms};
            append_stages(d.stages, decomp_perf, "decompress");
            if (has_m) set_quality(d, m);
            try_write_report(s.report_json_path, d);
        }
    } catch (...) {
        if (d_output) FZ_CUDA_CHECK_WARN(cudaFree(d_output));
        throw;
    }
    if (d_output) FZ_CUDA_CHECK(cudaFree(d_output));
    return 0;
}

static int run_benchmark(CliSettings s) {
    if (s.input_path.empty()) throw std::runtime_error("-b (benchmark) requires -i/--input");

    size_t element_size = (s.type == "f64" || s.type == "i64") ? 8 : 4;
    std::vector<uint8_t> input_bytes = read_binary_file(s.input_path);
    if (input_bytes.empty()) throw std::runtime_error("Input file is empty: " + s.input_path);

    const size_t payload_bytes = validate_or_infer_dims(&s, input_bytes.size(), element_size);
    const bool want_json = !s.report_json_path.empty();
    void* d_input = nullptr;

    // Graph mode state — bench_stream stays 0 (default stream) unless graph capture succeeds.
    bool graph_active = false;
    std::string graph_reason;
    cudaStream_t bench_stream = 0;

    try {
        FZ_CUDA_CHECK(cudaMalloc(&d_input, payload_bytes));
        FZ_CUDA_CHECK(cudaMemcpy(d_input, input_bytes.data(), payload_bytes, cudaMemcpyHostToDevice));

        s.profile = true;
        std::unique_ptr<Pipeline> pipeline;

        // ── Try CUDA Graph capture ─────────────────────────────────────────────
        // Graph mode requires PREALLOCATE (enforced here regardless of -strategy).
        // enableGraphMode() must be called before finalize() (which loadConfig() does
        // internally, or build_dynamic_linear_pipeline() does at the end).
        // enableProfiling() must be on before warmup()/captureGraph() so the
        // whole-pipeline DagEventTimer (events recorded on `stream`, outside the
        // graph) yields dag_elapsed_ms.  Per-stage event timing is *not* available
        // during a graph replay — cudaEventElapsedTime() across graph-recorded
        // events is unsupported — so the compress `stages[]` breakdown is empty
        // under --graph (decompress, which runs the normal DAG, still reports it).
        // A real (non-default) stream is required because cudaStreamBeginCapture
        // returns cudaErrorStreamCaptureUnsupported on stream 0.
        if (s.use_graph) {
            try {
                CliSettings gs = s;
                gs.strategy    = MemoryStrategy::PREALLOCATE;
                gs.warmup      = false;   // manual warmup(stream) below; skip auto on stream 0
                gs.print_pipeline = false; // suppress during trial; print after confirmed success

                auto gp = std::make_unique<Pipeline>(payload_bytes, MemoryStrategy::PREALLOCATE,
                                                     s.pool_multiplier);
                if (!s.config_path.empty()) {
                    gp->setDims(s.nx, s.ny, s.nz);
                    gp->setWarmupOnFinalize(false);
                    if (s.bounds_check) gp->enableBoundsCheck(true);
                    gp->enableProfiling(true);
                    gp->enableGraphMode(true);
                    gp->loadConfig(s.config_path);   // calls finalize() internally
                } else {
                    gp->enableGraphMode(true);        // must precede finalize() inside build_*
                    if (gs.type == "f32") {
                        build_dynamic_linear_pipeline<float>(gp.get(), gs);
                    } else if (gs.type == "f64") {
                        build_dynamic_linear_pipeline<double>(gp.get(), gs);
                    } else {
                        throw std::runtime_error("Dynamic builder only supports f32/f64 currently.");
                    }
                }

                // Ensure profiling is on before warmup/captureGraph (calling again is harmless
                // if build_dynamic_linear_pipeline already set it).
                gp->enableProfiling(true);

                // CUDA graph capture requires a non-default stream.
                FZ_CUDA_CHECK(cudaStreamCreate(&bench_stream));

                gp->warmup(bench_stream);
                gp->captureGraph(bench_stream);
                FZ_CUDA_CHECK(cudaStreamSynchronize(bench_stream));

                if (s.print_pipeline) gp->printPipeline();

                pipeline = std::move(gp);
                graph_active = true;
                s.strategy = MemoryStrategy::PREALLOCATE;  // reflect actual strategy in JSON
                std::cout << "[benchmark] CUDA graph captured successfully\n";
            } catch (const std::exception& ex) {
                graph_reason = ex.what();
                if (bench_stream != 0) {
                    cudaStreamDestroy(bench_stream);
                    bench_stream = 0;
                }
                pipeline.reset();
                std::cerr << "[benchmark] Graph capture failed (falling back to normal pipeline): "
                          << ex.what() << "\n";
            }
        }

        // ── Normal pipeline (no --graph, or silent graph fallback) ─────────────
        if (!pipeline) {
            if (!s.config_path.empty()) {
                pipeline = std::make_unique<Pipeline>(payload_bytes, s.strategy, s.pool_multiplier);
                pipeline->setDims(s.nx, s.ny, s.nz);
                pipeline->setWarmupOnFinalize(s.warmup);
                pipeline->enableProfiling(true);
                if (s.bounds_check) pipeline->enableBoundsCheck(true);
                pipeline->loadConfig(s.config_path);
                if (s.print_pipeline) pipeline->printPipeline();
            } else {
                pipeline = std::make_unique<Pipeline>(payload_bytes, s.strategy, s.pool_multiplier);
                if (s.type == "f32") {
                    build_dynamic_linear_pipeline<float>(pipeline.get(), s);
                } else if (s.type == "f64") {
                    build_dynamic_linear_pipeline<double>(pipeline.get(), s);
                } else {
                    throw std::runtime_error("Dynamic builder only supports f32/f64 currently.");
                }
            }
        }

        void* d_compressed = nullptr;
        size_t compressed_size = 0;
        pipeline->compress(d_input, payload_bytes, &d_compressed, &compressed_size, bench_stream);
        FZ_CUDA_CHECK(cudaDeviceSynchronize());

        TimingSummary compress_stats, decompress_stats;
        std::vector<uint8_t> final_recon;
        PipelinePerfResult last_compress_perf, last_decompress_perf;

        for (int i = 0; i < s.benchmark_runs; ++i) {
            const bool is_last = (i == s.benchmark_runs - 1);

            const auto t0 = std::chrono::high_resolution_clock::now();
            pipeline->compress(d_input, payload_bytes, &d_compressed, &compressed_size, bench_stream);
            FZ_CUDA_CHECK(cudaDeviceSynchronize());
            const auto t1 = std::chrono::high_resolution_clock::now();
            compress_stats.add(std::chrono::duration<double, std::milli>(t1 - t0).count(), pipeline->getLastPerfResult().dag_elapsed_ms);
            if (is_last) last_compress_perf = pipeline->getLastPerfResult();

            void* d_recon = nullptr;
            size_t recon_size = 0;
            const auto t2 = std::chrono::high_resolution_clock::now();
            pipeline->decompress(d_compressed, compressed_size, &d_recon, &recon_size, bench_stream);
            FZ_CUDA_CHECK(cudaDeviceSynchronize());
            const auto t3 = std::chrono::high_resolution_clock::now();

            if (recon_size != payload_bytes) {
                if (d_recon) FZ_CUDA_CHECK_WARN(cudaFree(d_recon));
                throw std::runtime_error("Benchmark size mismatch");
            }
            decompress_stats.add(std::chrono::duration<double, std::milli>(t3 - t2).count(), pipeline->getLastPerfResult().dag_elapsed_ms);
            if (is_last) last_decompress_perf = pipeline->getLastPerfResult();

            if ((s.report || !s.original_path.empty() || want_json) && is_last) {
                final_recon.resize(recon_size);
                FZ_CUDA_CHECK(cudaMemcpy(final_recon.data(), d_recon, recon_size, cudaMemcpyDeviceToHost));
            }

            if (!pipeline->isPoolManagedDecompOutput() && d_recon) {
                FZ_CUDA_CHECK(cudaFree(d_recon));
            }
        }

        print_summary("compress", compress_stats, payload_bytes);
        print_summary("decompress", decompress_stats, payload_bytes);
        last_compress_perf.print(std::cout);
        last_decompress_perf.print(std::cout);

        // Build comparison source + quality metrics once (shared by text + JSON).
        const bool want_quality = s.report || !s.original_path.empty() || want_json;
        std::vector<uint8_t> orig;
        Metrics m;
        bool has_m = false;
        if (want_quality) {
            orig = s.original_path.empty() ? input_bytes : read_binary_file(s.original_path);
            if (!orig.empty() && orig.size() == final_recon.size()) {
                if (s.type == "f32") m = calc_metrics<float>(orig, final_recon);
                else if (s.type == "f64") m = calc_metrics<double>(orig, final_recon);
                else if (s.type == "i32") m = calc_metrics<int32_t>(orig, final_recon);
                else if (s.type == "i64") m = calc_metrics<int64_t>(orig, final_recon);
                has_m = true;
            }
        }

        if (s.report || !s.original_path.empty()) {
            double ratio = static_cast<double>(payload_bytes) / compressed_size;
            std::cout << "\n[Quality Report]\n"
                      << "  Input size:      " << payload_bytes << " bytes\n"
                      << "  Compressed size: " << compressed_size << " bytes\n"
                      << "  Ratio:           " << std::fixed << std::setprecision(2) << ratio << "x\n"
                      << "  Peak device mem: " << std::setprecision(1)
                      << pipeline->getPeakMemoryUsage() / 1024.0 / 1024.0 << " MB\n";

            if (has_m) {
                std::cout << "  Value Range:     [" << std::scientific << m.val_min << ", " << m.val_max << "] (Span: " << m.val_range << ")\n"
                          << "  Max Abs Error:   " << std::scientific << m.max_err << "\n"
                          << "  PSNR:            " << std::fixed << std::setprecision(2) << m.psnr << " dB\n"
                          << "  NRMSE:           " << std::scientific << m.nrmse << "\n";
            } else if (!orig.empty()) {
                std::cout << "  Compare error:   Size mismatch! Original=" << orig.size()
                          << ", Reconstructed=" << final_recon.size() << "\n";
            }
        }

        if (want_json) {
            fz::cli::ReportData d = make_report_base(s, "benchmark");
            d.has_size          = true;
            d.original_bytes    = payload_bytes;
            d.compressed_bytes  = compressed_size;
            d.has_memory        = true;
            d.peak_device_bytes = pipeline->getPeakMemoryUsage();
            d.n_runs            = s.benchmark_runs;
            d.compress.present      = true;
            d.compress.host_wall_ms = compress_stats.host_ms;
            d.compress.device_ms.assign(compress_stats.dag_ms.begin(), compress_stats.dag_ms.end());
            d.decompress.present      = true;
            d.decompress.host_wall_ms = decompress_stats.host_ms;
            d.decompress.device_ms.assign(decompress_stats.dag_ms.begin(), decompress_stats.dag_ms.end());
            append_stages(d.stages, last_compress_perf, "compress");
            append_stages(d.stages, last_decompress_perf, "decompress");
            if (has_m) set_quality(d, m);
            d.graph_requested = s.use_graph;
            d.graph_active    = graph_active;
            d.graph_incompatible_reason = graph_reason;
            try_write_report(s.report_json_path, d);
        }

    } catch (...) {
        if (bench_stream != 0) { cudaStreamDestroy(bench_stream); bench_stream = 0; }
        if (d_input) FZ_CUDA_CHECK_WARN(cudaFree(d_input));
        throw;
    }
    if (bench_stream != 0) cudaStreamDestroy(bench_stream);
    if (d_input) FZ_CUDA_CHECK(cudaFree(d_input));
    return 0;
}

} // namespace

int fzgmod_cli_main(int argc, char** argv) {
    CliSettings settings;  // hoisted so the error path can emit a report-json
    try {
        if (argc < 2) {
            print_root_usage(argv[0]);
            return 1;
        }

        OptionMap opts = parse_option_tokens(argc, argv, 1);
        if (contains(opts, "help")) {
            print_root_usage(argv[0]);
            return 0;
        }

        apply_common_options(opts, &settings);

        if (settings.operation == CliOperation::Compress) {
            return run_compress(settings);
        } else if (settings.operation == CliOperation::Decompress) {
            return run_decompress(settings);
        } else if (settings.operation == CliOperation::Benchmark) {
            return run_benchmark(settings);
        } else {
            throw std::runtime_error("Must specify operation mode: -z (compress), -x (decompress), or -b (benchmark)");
        }
    } catch (const std::exception& e) {
        std::cerr << "[fzgmod-cli] error: " << e.what() << "\n\n";
        emit_error_report(settings.report_json_path, settings, e.what());
        print_root_usage(argc > 0 ? argv[0] : "fzgmod-cli");
        return 1;
    }
}