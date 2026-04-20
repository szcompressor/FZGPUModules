#include "cuda_check.h"
#include "fzgpumodules.h"

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

using namespace fz;

namespace {

using OptionMap = std::unordered_map<std::string, std::string>;

constexpr size_t kDefaultChunkSize = 16384;
constexpr int kDefaultBenchmarkRuns = 10;

enum class PipelineKind {
	PFPL,
	LorenzoRZE,
};

enum class BenchmarkPhase {
	Compress,
	Decompress,
	Both,
};

struct CliSettings {
	std::string input_path;
	std::string output_path;

	size_t nx = 0;
	size_t ny = 1;
	size_t nz = 1;

	float error_bound = 1e-3f;
	ErrorBoundMode error_mode = ErrorBoundMode::REL;

	MemoryStrategy strategy = MemoryStrategy::PREALLOCATE;
	float pool_multiplier = 3.0f;

	PipelineKind pipeline_kind = PipelineKind::PFPL;
	bool warmup = false;
	bool profile = false;

	size_t chunk_size = kDefaultChunkSize;
	int rze_levels = 4;

	int benchmark_runs = kDefaultBenchmarkRuns;
	BenchmarkPhase benchmark_phase = BenchmarkPhase::Compress;
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
	if (a == 0 || b == 0) {
		return 0;
	}
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
		if (idx != text.size()) {
			throw std::runtime_error("trailing characters");
		}
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
		if (idx != text.size()) {
			throw std::runtime_error("trailing characters");
		}
		return v;
	} catch (const std::exception&) {
		throw std::runtime_error(std::string("Invalid value for ") + name + ": '" + text + "'");
	}
}

static bool parse_bool(const std::string& text, const char* name) {
	const std::string lower = to_lower(trim(text));
	if (lower == "1" || lower == "true" || lower == "yes" || lower == "on") {
		return true;
	}
	if (lower == "0" || lower == "false" || lower == "no" || lower == "off") {
		return false;
	}
	throw std::runtime_error(std::string("Invalid value for ") + name + ": '" + text + "'");
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

static PipelineKind parse_pipeline_kind(const std::string& text) {
	const std::string value = to_lower(trim(text));
	if (value == "pfpl") return PipelineKind::PFPL;
	if (value == "lorenzo" || value == "lorenzo-rze") return PipelineKind::LorenzoRZE;
	throw std::runtime_error("Unknown pipeline kind: '" + text + "' (expected pfpl|lorenzo-rze)");
}

static BenchmarkPhase parse_benchmark_phase(const std::string& text) {
	const std::string value = to_lower(trim(text));
	if (value == "compress") return BenchmarkPhase::Compress;
	if (value == "decompress") return BenchmarkPhase::Decompress;
	if (value == "both") return BenchmarkPhase::Both;
	throw std::runtime_error("Unknown benchmark phase: '" + text + "' (expected compress|decompress|both)");
}

static std::string mode_to_string(ErrorBoundMode mode) {
	switch (mode) {
		case ErrorBoundMode::REL: return "rel";
		case ErrorBoundMode::ABS: return "abs";
		case ErrorBoundMode::NOA: return "noa";
	}
	return "unknown";
}

static std::string strategy_to_string(MemoryStrategy strategy) {
	switch (strategy) {
		case MemoryStrategy::MINIMAL: return "minimal";
		case MemoryStrategy::PREALLOCATE: return "preallocate";
	}
	return "unknown";
}

static std::string pipeline_to_string(PipelineKind kind) {
	switch (kind) {
		case PipelineKind::PFPL: return "pfpl";
		case PipelineKind::LorenzoRZE: return "lorenzo-rze";
	}
	return "unknown";
}

static OptionMap parse_option_tokens(int argc, char** argv, int start_index) {
	OptionMap opts;

	for (int i = start_index; i < argc; ++i) {
		std::string token = argv[i];
		if (token == "-h" || token == "--help") {
			opts["help"] = "true";
			continue;
		}

		if (token.rfind("--", 0) != 0) {
			throw std::runtime_error("Unexpected positional argument: '" + token + "'");
		}

		std::string key;
		std::string value;
		const size_t eq = token.find('=');
		if (eq != std::string::npos) {
			key = normalize_key(token.substr(2, eq - 2));
			value = trim(token.substr(eq + 1));
			if (key.empty()) {
				throw std::runtime_error("Malformed option token: '" + token + "'");
			}
			opts[key] = value;
			continue;
		}

		key = normalize_key(token.substr(2));
		if (key.empty()) {
			throw std::runtime_error("Malformed option token: '" + token + "'");
		}

		const bool is_flag =
			(key == "warmup") ||
			(key == "profile");

		if (is_flag) {
			opts[key] = "true";
			continue;
		}

		if (i + 1 >= argc) {
			throw std::runtime_error("Missing value for option '--" + key + "'");
		}

		value = trim(argv[++i]);
		opts[key] = value;
	}

	return opts;
}

static std::vector<uint8_t> read_binary_file(const std::string& path) {
	std::ifstream in(path, std::ios::binary | std::ios::ate);
	if (!in.is_open()) {
		throw std::runtime_error("Failed to open input file: " + path);
	}

	const std::streamsize end_pos = in.tellg();
	if (end_pos < 0) {
		throw std::runtime_error("Failed to determine input size: " + path);
	}

	std::vector<uint8_t> bytes(static_cast<size_t>(end_pos));
	in.seekg(0, std::ios::beg);
	if (!bytes.empty()) {
		in.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
		if (!in) {
			throw std::runtime_error("Failed to read full input file: " + path);
		}
	}

	return bytes;
}

static void write_binary_file(const std::string& path, const uint8_t* data, size_t size) {
	std::ofstream out(path, std::ios::binary);
	if (!out.is_open()) {
		throw std::runtime_error("Failed to open output file: " + path);
	}

	if (size > 0) {
		out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(size));
		if (!out) {
			throw std::runtime_error("Failed to write output file: " + path);
		}
	}
}

static OptionMap read_config_file(const std::string& path) {
	std::ifstream in(path);
	if (!in.is_open()) {
		throw std::runtime_error("Failed to open config file: " + path);
	}

	OptionMap config;
	std::string line;
	size_t line_no = 0;
	while (std::getline(in, line)) {
		++line_no;
		const std::string t = trim(line);
		if (t.empty() || t[0] == '#' || t[0] == ';') {
			continue;
		}

		const size_t eq = t.find('=');
		if (eq == std::string::npos) {
			throw std::runtime_error(
				"Config parse error in " + path + " at line " + std::to_string(line_no) +
				": expected key=value");
		}

		const std::string key = normalize_key(t.substr(0, eq));
		const std::string value = trim(t.substr(eq + 1));
		if (key.empty()) {
			throw std::runtime_error(
				"Config parse error in " + path + " at line " + std::to_string(line_no) +
				": empty key");
		}
		config[key] = value;
	}

	return config;
}

static bool contains(const OptionMap& opts, const std::string& key) {
	return opts.find(key) != opts.end();
}

static const std::string& get_required(const OptionMap& opts, const std::string& key) {
	auto it = opts.find(key);
	if (it == opts.end()) {
		throw std::runtime_error("Missing required option '--" + key + "'");
	}
	return it->second;
}

static void apply_common_options(const OptionMap& opts, CliSettings* out) {
	auto apply_size = [&](const std::string& key, size_t* dst) {
		auto it = opts.find(key);
		if (it != opts.end()) *dst = parse_integer<size_t>(it->second, key.c_str());
	};

	// auto apply_float = [&](const std::string& key, float* dst) {
	// 	auto it = opts.find(key);
	// 	if (it != opts.end()) *dst = parse_float(it->second, key.c_str());
	// };

	auto apply_bool = [&](const std::string& key, bool* dst) {
		auto it = opts.find(key);
		if (it != opts.end()) *dst = parse_bool(it->second, key.c_str());
	};

	if (contains(opts, "input")) out->input_path = get_required(opts, "input");
	if (contains(opts, "output")) out->output_path = get_required(opts, "output");

	if (contains(opts, "nx")) out->nx = parse_integer<size_t>(get_required(opts, "nx"), "nx");
	if (contains(opts, "ny")) out->ny = parse_integer<size_t>(get_required(opts, "ny"), "ny");
	if (contains(opts, "nz")) out->nz = parse_integer<size_t>(get_required(opts, "nz"), "nz");
	if (contains(opts, "dim-x")) out->nx = parse_integer<size_t>(get_required(opts, "dim-x"), "dim-x");
	if (contains(opts, "dim-y")) out->ny = parse_integer<size_t>(get_required(opts, "dim-y"), "dim-y");
	if (contains(opts, "dim-z")) out->nz = parse_integer<size_t>(get_required(opts, "dim-z"), "dim-z");

	if (contains(opts, "error-bound")) {
		out->error_bound = parse_float(get_required(opts, "error-bound"), "error-bound");
	}
	if (contains(opts, "eb")) {
		out->error_bound = parse_float(get_required(opts, "eb"), "eb");
	}
	if (contains(opts, "mode")) {
		out->error_mode = parse_error_mode(get_required(opts, "mode"));
	}
	if (contains(opts, "strategy")) {
		out->strategy = parse_strategy(get_required(opts, "strategy"));
	}
	if (contains(opts, "pool-mult")) {
		out->pool_multiplier = parse_float(get_required(opts, "pool-mult"), "pool-mult");
	}
	if (contains(opts, "pool-multiplier")) {
		out->pool_multiplier = parse_float(get_required(opts, "pool-multiplier"), "pool-multiplier");
	}
	if (contains(opts, "pipeline")) {
		out->pipeline_kind = parse_pipeline_kind(get_required(opts, "pipeline"));
	}

	apply_bool("warmup", &out->warmup);
	apply_bool("profile", &out->profile);

	apply_size("chunk-size", &out->chunk_size);
	if (contains(opts, "rze-levels")) {
		out->rze_levels = parse_integer<int>(get_required(opts, "rze-levels"), "rze-levels");
	}
}

static void validate_common_settings(const CliSettings& s) {
	if (s.error_bound <= 0.0f) {
		throw std::runtime_error("error-bound must be > 0");
	}
	if (s.pool_multiplier <= 0.0f) {
		throw std::runtime_error("pool-mult must be > 0");
	}
	if (s.chunk_size == 0) {
		throw std::runtime_error("chunk-size must be > 0");
	}
	if (s.rze_levels < 1 || s.rze_levels > 4) {
		throw std::runtime_error("rze-levels must be in [1, 4]");
	}
}

static size_t validate_or_infer_dims(CliSettings* settings, size_t input_bytes) {
	if ((input_bytes % sizeof(float)) != 0) {
		throw std::runtime_error(
			"Input byte size is not divisible by sizeof(float); this first-pass CLI expects float32 raw input");
	}

	if (settings->nx == 0) {
		settings->nx = input_bytes / sizeof(float);
		settings->ny = 1;
		settings->nz = 1;
	}

	if (settings->ny == 0 || settings->nz == 0) {
		throw std::runtime_error("ny and nz must be > 0");
	}

	const size_t n_xy = checked_mul(settings->nx, settings->ny, "nx*ny");
	const size_t n = checked_mul(n_xy, settings->nz, "nx*ny*nz");
	const size_t expected = checked_mul(n, sizeof(float), "elements*sizeof(float)");
	if (expected != input_bytes) {
		std::ostringstream oss;
		oss << "Input size mismatch: file has " << input_bytes
			<< " bytes, but nx*ny*nz*sizeof(float) = " << expected;
		throw std::runtime_error(oss.str());
	}
	return expected;
}

static void build_pipeline(Pipeline* pipeline, const CliSettings& s) {
	pipeline->setDims(s.nx, s.ny, s.nz);
	pipeline->setWarmupOnFinalize(s.warmup);
	pipeline->enableProfiling(s.profile);

	if (s.pipeline_kind == PipelineKind::PFPL) {
		auto* quant = pipeline->addStage<QuantizerStage<float, uint32_t>>();
		quant->setErrorBound(s.error_bound);
		quant->setErrorBoundMode(s.error_mode);
		quant->setQuantRadius(s.error_mode == ErrorBoundMode::ABS ? (1 << 22) : 32768);
		quant->setOutlierCapacity(0.05f);
		quant->setZigzagCodes(true);
		if (s.error_mode != ErrorBoundMode::REL) {
			quant->setInplaceOutliers(true);
		}

		auto* diff = pipeline->addStage<DifferenceStage<int32_t, uint32_t>>();
		diff->setChunkSize(s.chunk_size);
		pipeline->connect(diff, quant, "codes");

		auto* bshuf = pipeline->addStage<BitshuffleStage>();
		bshuf->setBlockSize(s.chunk_size);
		bshuf->setElementWidth(4);
		pipeline->connect(bshuf, diff);

		auto* rze = pipeline->addStage<RZEStage>();
		rze->setChunkSize(s.chunk_size);
		rze->setLevels(s.rze_levels);
		pipeline->connect(rze, bshuf);
	} else {
		auto* lrz = pipeline->addStage<LorenzoStage<float, uint16_t>>();
		lrz->setErrorBound(s.error_bound);
		lrz->setErrorBoundMode(s.error_mode);
		lrz->setQuantRadius(32768);
		lrz->setOutlierCapacity(0.10f);
		lrz->setZigzagCodes(true);

		auto* bshuf = pipeline->addStage<BitshuffleStage>();
		bshuf->setBlockSize(s.chunk_size);
		bshuf->setElementWidth(sizeof(uint16_t));
		pipeline->connect(bshuf, lrz, "codes");

		auto* rze = pipeline->addStage<RZEStage>();
		rze->setChunkSize(s.chunk_size);
		rze->setLevels(s.rze_levels);
		pipeline->connect(rze, bshuf);
	}

	pipeline->finalize();
}

static void print_root_usage(const char* argv0) {
	std::cout
		<< "FZGPUModules CLI (first pass)\n\n"
		<< "Usage:\n"
		<< "  " << argv0 << " compress --input <raw.f32> --output <out.fzm> [options]\n"
		<< "  " << argv0 << " decompress --input <in.fzm> --output <raw.out>\n"
		<< "  " << argv0 << " benchmark --input <raw.f32> [options]\n"
		<< "  " << argv0 << " pipeline-from-config --config <cfg.txt> [--input ... --output ...]\n\n"
		<< "Global:\n"
		<< "  --help                          Show help\n\n"
		<< "Shared options (compress/benchmark/pipeline-from-config):\n"
		<< "  --nx <n> --ny <n> --nz <n>      Input dimensions (default: infer nx from file, ny=nz=1)\n"
		<< "  --pipeline <pfpl|lorenzo-rze>   Pipeline topology (default: pfpl)\n"
		<< "  --error-bound <v>               Error bound (default: 1e-3)\n"
		<< "  --mode <rel|abs|noa>            Error mode (default: rel)\n"
		<< "  --strategy <minimal|preallocate>\n"
		<< "  --pool-mult <v>                 Pool multiplier (default: 3.0)\n"
		<< "  --chunk-size <n>                Chunk size (default: 16384)\n"
		<< "  --rze-levels <1..4>             RZE levels (default: 4)\n"
		<< "  --warmup                        Enable warmup on finalize\n";
}

static void print_compress_usage(const char* argv0) {
	std::cout
		<< "Usage:\n"
		<< "  " << argv0 << " compress --input <raw.f32> --output <out.fzm> [options]\n\n"
		<< "Notes:\n"
		<< "  - This first pass currently expects float32 raw input\n"
		<< "  - Output is FZM v3 format written via Pipeline::writeToFile()\n";
}

static void print_decompress_usage(const char* argv0) {
	std::cout
		<< "Usage:\n"
		<< "  " << argv0 << " decompress --input <in.fzm> --output <raw.out>\n\n"
		<< "Notes:\n"
		<< "  - Reconstructs pipeline from file and writes decompressed bytes as raw output\n";
}

static void print_benchmark_usage(const char* argv0) {
	std::cout
		<< "Usage:\n"
		<< "  " << argv0 << " benchmark --input <raw.f32> [options]\n\n"
		<< "Benchmark options:\n"
		<< "  --runs <n>                      Iterations (default: 10)\n"
		<< "  --phase <compress|decompress|both>\n";
}

static void print_config_usage(const char* argv0) {
	std::cout
		<< "Usage:\n"
		<< "  " << argv0 << " pipeline-from-config --config <cfg.txt> [overrides]\n\n"
		<< "Config file format:\n"
		<< "  key=value (one per line, '#' comments supported)\n"
		<< "  Keys match long option names, e.g. pipeline, error-bound, mode, nx, ny, nz\n";
}

static void print_summary(const std::string& label, const TimingSummary& stats, size_t bytes) {
	if (stats.empty()) {
		return;
	}

	const int n = static_cast<int>(stats.host_ms.size());
	const double mean_host = std::accumulate(stats.host_ms.begin(), stats.host_ms.end(), 0.0) / n;
	const float mean_dag = std::accumulate(stats.dag_ms.begin(), stats.dag_ms.end(), 0.0f) / n;
	const double min_host = *std::min_element(stats.host_ms.begin(), stats.host_ms.end());
	const double max_host = *std::max_element(stats.host_ms.begin(), stats.host_ms.end());

	const auto tput_gbs = [bytes](double ms) -> double {
		return ms > 0.0 ? static_cast<double>(bytes) / (ms * 1e-3) / 1e9 : 0.0;
	};

	std::cout << "\n[benchmark] " << label << " summary\n"
			  << "  runs:           " << n << "\n"
			  << "  host ms:        mean=" << std::fixed << std::setprecision(3) << mean_host
			  << " min=" << min_host << " max=" << max_host << "\n"
			  << "  dag ms:         mean=" << mean_dag << "\n"
			  << "  throughput:     " << std::setprecision(2) << tput_gbs(mean_host) << " GB/s (host mean)\n";
}

static int run_compress(CliSettings settings) {
	if (settings.input_path.empty()) {
		throw std::runtime_error("compress requires --input");
	}
	if (settings.output_path.empty()) {
		throw std::runtime_error("compress requires --output");
	}

	validate_common_settings(settings);

	std::vector<uint8_t> input_bytes = read_binary_file(settings.input_path);
	if (input_bytes.empty()) {
		throw std::runtime_error("Input file is empty: " + settings.input_path);
	}

	const size_t payload_bytes = validate_or_infer_dims(&settings, input_bytes.size());

	void* d_input = nullptr;
	try {
		FZ_CUDA_CHECK(cudaMalloc(&d_input, payload_bytes));
		FZ_CUDA_CHECK(cudaMemcpy(d_input, input_bytes.data(), payload_bytes, cudaMemcpyHostToDevice));

		Pipeline pipeline(payload_bytes, settings.strategy, settings.pool_multiplier);
		build_pipeline(&pipeline, settings);

		void* d_compressed = nullptr;
		size_t compressed_size = 0;
		pipeline.compress(d_input, payload_bytes, &d_compressed, &compressed_size, 0);
		pipeline.writeToFile(settings.output_path, 0);
		FZ_CUDA_CHECK(cudaDeviceSynchronize());

		const double ratio = compressed_size > 0
			? static_cast<double>(payload_bytes) / static_cast<double>(compressed_size)
			: 0.0;

		std::cout << "[compress] completed\n"
				  << "  input:          " << settings.input_path << "\n"
				  << "  output:         " << settings.output_path << "\n"
				  << "  dims:           " << settings.nx << "x" << settings.ny << "x" << settings.nz << "\n"
				  << "  pipeline:       " << pipeline_to_string(settings.pipeline_kind) << "\n"
				  << "  mode:           " << mode_to_string(settings.error_mode) << "\n"
				  << "  strategy:       " << strategy_to_string(settings.strategy) << "\n"
				  << "  error bound:    " << settings.error_bound << "\n"
				  << "  input bytes:    " << payload_bytes << "\n"
				  << "  compressed:     " << compressed_size << "\n"
				  << "  ratio:          " << std::fixed << std::setprecision(3) << ratio << "x\n";
	} catch (...) {
		if (d_input != nullptr) {
			FZ_CUDA_CHECK_WARN(cudaFree(d_input));
		}
		throw;
	}

	if (d_input != nullptr) {
		FZ_CUDA_CHECK(cudaFree(d_input));
	}

	return 0;
}

static int run_decompress(CliSettings settings) {
	if (settings.input_path.empty()) {
		throw std::runtime_error("decompress requires --input");
	}
	if (settings.output_path.empty()) {
		throw std::runtime_error("decompress requires --output");
	}

	void* d_output = nullptr;
	size_t output_size = 0;

	try {
		Pipeline::decompressFromFile(settings.input_path, &d_output, &output_size, 0);
		FZ_CUDA_CHECK(cudaDeviceSynchronize());

		std::vector<uint8_t> host(output_size);
		if (output_size > 0) {
			FZ_CUDA_CHECK(cudaMemcpy(host.data(), d_output, output_size, cudaMemcpyDeviceToHost));
		}

		write_binary_file(settings.output_path, host.data(), host.size());

		std::cout << "[decompress] completed\n"
				  << "  input:          " << settings.input_path << "\n"
				  << "  output:         " << settings.output_path << "\n"
				  << "  output bytes:   " << output_size << "\n";
	} catch (...) {
		if (d_output != nullptr) {
			FZ_CUDA_CHECK_WARN(cudaFree(d_output));
		}
		throw;
	}

	if (d_output != nullptr) {
		FZ_CUDA_CHECK(cudaFree(d_output));
	}

	return 0;
}

static int run_benchmark(CliSettings settings) {
	if (settings.input_path.empty()) {
		throw std::runtime_error("benchmark requires --input");
	}

	validate_common_settings(settings);
	if (settings.benchmark_runs <= 0) {
		throw std::runtime_error("runs must be > 0");
	}

	std::vector<uint8_t> input_bytes = read_binary_file(settings.input_path);
	if (input_bytes.empty()) {
		throw std::runtime_error("Input file is empty: " + settings.input_path);
	}

	const size_t payload_bytes = validate_or_infer_dims(&settings, input_bytes.size());

	void* d_input = nullptr;
	try {
		FZ_CUDA_CHECK(cudaMalloc(&d_input, payload_bytes));
		FZ_CUDA_CHECK(cudaMemcpy(d_input, input_bytes.data(), payload_bytes, cudaMemcpyHostToDevice));

		settings.profile = true; // benchmark always enables profiling so dag_ms is populated.
		Pipeline pipeline(payload_bytes, settings.strategy, settings.pool_multiplier);
		build_pipeline(&pipeline, settings);

		void* d_compressed = nullptr;
		size_t compressed_size = 0;
		pipeline.compress(d_input, payload_bytes, &d_compressed, &compressed_size, 0);
		FZ_CUDA_CHECK(cudaDeviceSynchronize());

		TimingSummary compress_stats;
		TimingSummary decompress_stats;

		for (int i = 0; i < settings.benchmark_runs; ++i) {
			if (settings.benchmark_phase == BenchmarkPhase::Compress ||
				settings.benchmark_phase == BenchmarkPhase::Both) {
				const auto t0 = std::chrono::high_resolution_clock::now();
				pipeline.compress(d_input, payload_bytes, &d_compressed, &compressed_size, 0);
				FZ_CUDA_CHECK(cudaDeviceSynchronize());
				const auto t1 = std::chrono::high_resolution_clock::now();
				const double host_ms =
					std::chrono::duration<double, std::milli>(t1 - t0).count();
				compress_stats.add(host_ms, pipeline.getLastPerfResult().dag_elapsed_ms);
			}

			if (settings.benchmark_phase == BenchmarkPhase::Decompress ||
				settings.benchmark_phase == BenchmarkPhase::Both) {
				void* d_recon = nullptr;
				size_t recon_size = 0;

				const auto t0 = std::chrono::high_resolution_clock::now();
				pipeline.decompress(d_compressed, compressed_size, &d_recon, &recon_size, 0);
				FZ_CUDA_CHECK(cudaDeviceSynchronize());
				const auto t1 = std::chrono::high_resolution_clock::now();

				if (recon_size != payload_bytes) {
					if (d_recon != nullptr) {
						FZ_CUDA_CHECK_WARN(cudaFree(d_recon));
					}
					throw std::runtime_error(
						"Benchmark decompression size mismatch: got " +
						std::to_string(recon_size) + " expected " +
						std::to_string(payload_bytes));
				}

				const double host_ms =
					std::chrono::duration<double, std::milli>(t1 - t0).count();
				decompress_stats.add(host_ms, pipeline.getLastPerfResult().dag_elapsed_ms);

				if (d_recon != nullptr) {
					FZ_CUDA_CHECK(cudaFree(d_recon));
				}
			}
		}

		std::cout << "[benchmark] completed\n"
				  << "  input:          " << settings.input_path << "\n"
				  << "  runs:           " << settings.benchmark_runs << "\n"
				  << "  dims:           " << settings.nx << "x" << settings.ny << "x" << settings.nz << "\n"
				  << "  pipeline:       " << pipeline_to_string(settings.pipeline_kind) << "\n"
				  << "  strategy:       " << strategy_to_string(settings.strategy) << "\n"
				  << "  mode:           " << mode_to_string(settings.error_mode) << "\n"
				  << "  error bound:    " << settings.error_bound << "\n"
				  << "  compressed:     " << compressed_size << " bytes\n";

		print_summary("compress", compress_stats, payload_bytes);
		print_summary("decompress", decompress_stats, payload_bytes);
	} catch (...) {
		if (d_input != nullptr) {
			FZ_CUDA_CHECK_WARN(cudaFree(d_input));
		}
		throw;
	}

	if (d_input != nullptr) {
		FZ_CUDA_CHECK(cudaFree(d_input));
	}

	return 0;
}

static CliSettings settings_from_options(const OptionMap& opts) {
	CliSettings settings;
	apply_common_options(opts, &settings);

	if (contains(opts, "runs")) {
		settings.benchmark_runs = parse_integer<int>(get_required(opts, "runs"), "runs");
	}
	if (contains(opts, "phase")) {
		settings.benchmark_phase = parse_benchmark_phase(get_required(opts, "phase"));
	}

	return settings;
}

} // namespace

int fzgmod_cli_main(int argc, char** argv) {
	try {
		if (argc < 2) {
			print_root_usage(argv[0]);
			return 1;
		}

		const std::string command = to_lower(argv[1]);
		if (command == "-h" || command == "--help" || command == "help") {
			print_root_usage(argv[0]);
			return 0;
		}

		if (command == "compress") {
			OptionMap opts = parse_option_tokens(argc, argv, 2);
			if (contains(opts, "help")) {
				print_compress_usage(argv[0]);
				return 0;
			}
			CliSettings settings = settings_from_options(opts);
			return run_compress(settings);
		}

		if (command == "decompress") {
			OptionMap opts = parse_option_tokens(argc, argv, 2);
			if (contains(opts, "help")) {
				print_decompress_usage(argv[0]);
				return 0;
			}
			CliSettings settings = settings_from_options(opts);
			return run_decompress(settings);
		}

		if (command == "benchmark") {
			OptionMap opts = parse_option_tokens(argc, argv, 2);
			if (contains(opts, "help")) {
				print_benchmark_usage(argv[0]);
				return 0;
			}
			CliSettings settings = settings_from_options(opts);
			return run_benchmark(settings);
		}

		if (command == "pipeline-from-config") {
			OptionMap opts = parse_option_tokens(argc, argv, 2);
			if (contains(opts, "help")) {
				print_config_usage(argv[0]);
				return 0;
			}

			const std::string cfg_path = get_required(opts, "config");
			OptionMap merged = read_config_file(cfg_path);

			// Command-line options override config-file options.
			for (const auto& kv : opts) {
				if (kv.first == "config") {
					continue;
				}
				merged[kv.first] = kv.second;
			}

			CliSettings settings = settings_from_options(merged);
			return run_compress(settings);
		}

		throw std::runtime_error(
			"Unknown command: '" + command + "' (expected compress|decompress|benchmark|pipeline-from-config)");
	} catch (const std::exception& e) {
		std::cerr << "[fzgmod-cli] error: " << e.what() << "\n\n";
		print_root_usage(argc > 0 ? argv[0] : "fzgmod-cli");
		return 1;
	}
}