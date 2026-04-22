#include "cuda_check.h"
#include "fzgpumodules.h"
#include "pipeline/config.h"

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

    size_t chunk_size = kDefaultChunkSize;
    int rze_levels = 4;

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

        const bool is_flag =
            (key == "z") ||
            (key == "x") ||
            (key == "b" || key == "benchmark") ||
            (key == "report") ||
            (key == "warmup") ||
            (key == "profile");

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

    if (contains(opts, "dims")) parse_dims(opts.at("dims"), s->nx, s->ny, s->nz);
    if (contains(opts, "type")) s->type = to_lower(opts.at("type"));
    if (contains(opts, "stages")) s->stages = opts.at("stages");
    if (contains(opts, "error-bound")) s->error_bound = parse_float(opts.at("error-bound"), "error-bound");
    if (contains(opts, "mode")) s->error_mode = parse_error_mode(opts.at("mode"));
    if (contains(opts, "radius")) s->quant_radius = parse_integer<int>(opts.at("radius"), "radius");
    if (contains(opts, "strategy")) s->strategy = parse_strategy(opts.at("strategy"));
    if (contains(opts, "pool-mult")) s->pool_multiplier = parse_float(opts.at("pool-mult"), "pool-mult");
    if (contains(opts, "chunk-size")) s->chunk_size = parse_integer<size_t>(opts.at("chunk-size"), "chunk-size");
    if (contains(opts, "rze-levels")) s->rze_levels = parse_integer<int>(opts.at("rze-levels"), "rze-levels");
    if (contains(opts, "runs")) s->benchmark_runs = parse_integer<int>(opts.at("runs"), "runs");

    s->warmup = contains(opts, "warmup") && parse_bool(opts.at("warmup"), "warmup");
    s->profile = contains(opts, "profile") && parse_bool(opts.at("profile"), "profile");
    s->report = contains(opts, "report") && parse_bool(opts.at("report"), "report");

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
            auto* lrz = pipeline->addStage<LorenzoStage<T, uint16_t>>();
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
        } else if (name == "rze") {
            auto* rze = pipeline->addStage<RZEStage>();
            rze->setChunkSize(s.chunk_size);
            rze->setLevels(s.rze_levels);
            connect_next(rze);
        } else if (name == "diff" || name == "difference") {
            auto* diff = pipeline->addStage<DifferenceStage<uint16_t>>();
            diff->setChunkSize(s.chunk_size);
            connect_next(diff);
        } else if (name == "rle") {
            auto* rle = pipeline->addStage<RLEStage<uint16_t>>();
            connect_next(rle);
        } else if (name == "none") {
            // explicit no-op
        } else {
            throw std::runtime_error(
                "Unknown stage '" + name + "' in --stages. "
                "Supported: lorenzo, quantizer, bitshuffle, rze, diff, rle");
        }
    }

    pipeline->finalize();
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
        << "  -R, --report                      Generate a report\n"
        << "  --compare <original>              Compare decompressed output with original\n\n"
        << "Compression Parameters (for dynamic linear pipelines):\n"
        << "  --stages \"<s1->s2->...>\"          Ordered pipeline stages (default: \"lorenzo->bitshuffle->rze\")\n"
        << "                                    NOTE: Wrap in quotes to prevent shell redirection ('->')\n"
        << "                                    Supported stages: lorenzo, quantizer, bitshuffle,\n"
        << "                                                      rze, diff, rle\n"
        << "  -m, --mode <rel,abs,noa>          Error bound mode (default: rel)\n"
        << "  -e, --error-bound <val>           Error bound value (default: 1e-3)\n"
        << "  -t, --type <f32,f64>              Data type (default: f32)\n"
        << "  -r, --radius <value>              Quantization radius (default: 32768)\n"
        << "  --chunk-size <bytes>              Encoder chunk size (default: 16384)\n"
        << "  --rze-levels <1-4>                RZE levels (default: 4)\n"
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

static int run_compress(CliSettings s) {
    if (s.input_path.empty()) throw std::runtime_error("-z (compress) requires -i/--input");
    if (s.output_path.empty()) s.output_path = s.input_path + ".fzm";

    size_t element_size = (s.type == "f64" || s.type == "i64") ? 8 : 4;
    std::vector<uint8_t> input_bytes = read_binary_file(s.input_path);
    if (input_bytes.empty()) throw std::runtime_error("Input file is empty: " + s.input_path);

    const size_t payload_bytes = validate_or_infer_dims(&s, input_bytes.size(), element_size);

    void* d_input = nullptr;
    try {
        FZ_CUDA_CHECK(cudaMalloc(&d_input, payload_bytes));
        FZ_CUDA_CHECK(cudaMemcpy(d_input, input_bytes.data(), payload_bytes, cudaMemcpyHostToDevice));

        std::unique_ptr<Pipeline> pipeline;
        if (!s.config_path.empty()) {
            pipeline = std::make_unique<Pipeline>(payload_bytes, s.strategy, s.pool_multiplier);
            pipeline->setDims(s.nx, s.ny, s.nz);
            pipeline->loadConfig(s.config_path);
        } else {
            pipeline = std::make_unique<Pipeline>(payload_bytes, s.strategy, s.pool_multiplier);
            if (s.type == "f32") {
                build_dynamic_linear_pipeline<float>(pipeline.get(), s);
            } else if (s.type == "f64") {
                build_dynamic_linear_pipeline<double>(pipeline.get(), s);
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

        if (s.report) {
            double ratio = static_cast<double>(payload_bytes) / static_cast<double>(compressed_size);
            double tput = static_cast<double>(payload_bytes) / (host_ms * 1e-3) / 1e9;
            std::cout << "\n[Compress Report]\n"
                      << "  Input size:      " << payload_bytes << " bytes\n"
                      << "  Compressed size: " << compressed_size << " bytes\n"
                      << "  Ratio:           " << std::fixed << std::setprecision(2) << ratio << "x\n"
                      << "  Time:            " << host_ms << " ms\n"
                      << "  Throughput:      " << tput << " GB/s\n";
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

    void* d_output = nullptr;
    size_t output_size = 0;

    try {
        std::vector<uint8_t> orig;
        if (!s.original_path.empty()) {
            orig = read_binary_file(s.original_path);
        }

        const auto t0 = std::chrono::high_resolution_clock::now();
        Pipeline::decompressFromFile(s.input_path, &d_output, &output_size, 0);
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
                    Metrics m;
                    if (s.type == "f32") m = calc_metrics<float>(orig, host);
                    else if (s.type == "f64") m = calc_metrics<double>(orig, host);
                    else if (s.type == "i32") m = calc_metrics<int32_t>(orig, host);
                    else if (s.type == "i64") m = calc_metrics<int64_t>(orig, host);
                    
                    std::cout << "  Value Range:     [" << std::scientific << m.val_min << ", " << m.val_max << "] (Span: " << m.val_range << ")\n"
                              << "  Max Abs Error:   " << std::scientific << m.max_err << "\n"
                              << "  PSNR:            " << std::fixed << std::setprecision(2) << m.psnr << " dB\n"
                              << "  NRMSE:           " << std::scientific << m.nrmse << "\n";
                }
            }
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
    void* d_input = nullptr;

    try {
        FZ_CUDA_CHECK(cudaMalloc(&d_input, payload_bytes));
        FZ_CUDA_CHECK(cudaMemcpy(d_input, input_bytes.data(), payload_bytes, cudaMemcpyHostToDevice));

        s.profile = true;
        std::unique_ptr<Pipeline> pipeline;
        if (!s.config_path.empty()) {
            pipeline = std::make_unique<Pipeline>(payload_bytes, s.strategy, s.pool_multiplier);
            pipeline->setDims(s.nx, s.ny, s.nz);
            pipeline->loadConfig(s.config_path);
            pipeline->enableProfiling(true);
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

        void* d_compressed = nullptr;
        size_t compressed_size = 0;
        pipeline->compress(d_input, payload_bytes, &d_compressed, &compressed_size, 0);
        FZ_CUDA_CHECK(cudaDeviceSynchronize());

        TimingSummary compress_stats, decompress_stats;
        std::vector<uint8_t> final_recon;

        for (int i = 0; i < s.benchmark_runs; ++i) {
            const auto t0 = std::chrono::high_resolution_clock::now();
            pipeline->compress(d_input, payload_bytes, &d_compressed, &compressed_size, 0);
            FZ_CUDA_CHECK(cudaDeviceSynchronize());
            const auto t1 = std::chrono::high_resolution_clock::now();
            compress_stats.add(std::chrono::duration<double, std::milli>(t1 - t0).count(), pipeline->getLastPerfResult().dag_elapsed_ms);

            void* d_recon = nullptr;
            size_t recon_size = 0;
            const auto t2 = std::chrono::high_resolution_clock::now();
            pipeline->decompress(d_compressed, compressed_size, &d_recon, &recon_size, 0);
            FZ_CUDA_CHECK(cudaDeviceSynchronize());
            const auto t3 = std::chrono::high_resolution_clock::now();
            
            if (recon_size != payload_bytes) {
                if (d_recon) FZ_CUDA_CHECK_WARN(cudaFree(d_recon));
                throw std::runtime_error("Benchmark size mismatch");
            }
            decompress_stats.add(std::chrono::duration<double, std::milli>(t3 - t2).count(), pipeline->getLastPerfResult().dag_elapsed_ms);
            
            if ((s.report || !s.original_path.empty()) && i == s.benchmark_runs - 1) {
                final_recon.resize(recon_size);
                FZ_CUDA_CHECK(cudaMemcpy(final_recon.data(), d_recon, recon_size, cudaMemcpyDeviceToHost));
            }
            
            if (!pipeline->isPoolManagedDecompOutput() && d_recon) {
                FZ_CUDA_CHECK(cudaFree(d_recon));
            }
        }

        print_summary("compress", compress_stats, payload_bytes);
        print_summary("decompress", decompress_stats, payload_bytes);
        
        if (s.report || !s.original_path.empty()) {
            double ratio = static_cast<double>(payload_bytes) / compressed_size;
            std::cout << "\n[Quality Report]\n"
                      << "  Input size:      " << payload_bytes << " bytes\n"
                      << "  Compressed size: " << compressed_size << " bytes\n"
                      << "  Ratio:           " << std::fixed << std::setprecision(2) << ratio << "x\n";
            
            std::vector<uint8_t> orig;
            if (!s.original_path.empty()) {
                orig = read_binary_file(s.original_path);
            } else {
                orig = input_bytes;
            }
            
            if (!orig.empty() && orig.size() == final_recon.size()) {
                Metrics m;
                if (s.type == "f32") m = calc_metrics<float>(orig, final_recon);
                else if (s.type == "f64") m = calc_metrics<double>(orig, final_recon);
                else if (s.type == "i32") m = calc_metrics<int32_t>(orig, final_recon);
                else if (s.type == "i64") m = calc_metrics<int64_t>(orig, final_recon);
                
                std::cout << "  Value Range:     [" << std::scientific << m.val_min << ", " << m.val_max << "] (Span: " << m.val_range << ")\n"
                          << "  Max Abs Error:   " << std::scientific << m.max_err << "\n"
                          << "  PSNR:            " << std::fixed << std::setprecision(2) << m.psnr << " dB\n"
                          << "  NRMSE:           " << std::scientific << m.nrmse << "\n";
            } else if (!orig.empty()) {
                std::cout << "  Compare error:   Size mismatch! Original=" << orig.size() 
                          << ", Reconstructed=" << final_recon.size() << "\n";
            }
        }
        
    } catch (...) {
        if (d_input) FZ_CUDA_CHECK_WARN(cudaFree(d_input));
        throw;
    }
    if (d_input) FZ_CUDA_CHECK(cudaFree(d_input));
    return 0;
}

} // namespace

int fzgmod_cli_main(int argc, char** argv) {
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

        CliSettings settings;
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
        print_root_usage(argc > 0 ? argv[0] : "fzgmod-cli");
        return 1;
    }
}