#include <iostream>
#include <string>

#include "fzmod.hh"

#ifdef FZMOD_CUDASTF
#include "cudastf_api.hh"
#endif

namespace utils = _portable::utils;

struct CLIOptions {
  std::string input_file;
  bool skip_to_file = false;
  bool skip_huffman = false;
  bool dump = false;
  bool report = false;
  bool version = false;
  bool help = false;
  bool verbose = false;
  bool skip = false;
  bool stf = false; // use cudastf api
  std::string origin = "";

  std::vector<std::string> pipeline;
  uint32_t x = 0;
  uint32_t y = 0;
  uint32_t z = 0;
  size_t len = 0;
  int n_dims = 0;
  float outlier_buff_ratio = 0.2f;
  uint16_t radius = 512;

  bool comp = true;
  double eb = 0;
  bool float_precision = true;
};

static int constexpr PRECISION = 0; // index into pipeline
static int constexpr EB_TYPE = 1;          
static int constexpr PREDICTOR = 2;        
static int constexpr HISTOGRAM = 3;        
static int constexpr CODEC = 4;            
static int constexpr SECONDARY_CODEC = 5;  

void print_help(){
  std::cout
      << "Name: FZModules GPU Compression Library\n\n"
      << "Synopsis: (Basic usage) \n"
      << "  ./fzmod -t f32 -m rel -e 1e-3 -i {data} -l 300x100x200 -z --report\n"
      << "          ------ ------ ------- --------- -------------- -- --------\n"
      << "           Type   Mode   Error   Input     Dim-fast-slow  zip  Report\n"
      << "  ./fzmod -i {compressed} -x --compare {original} --report\n"
      << "          --------------- -- -------------------- -------------\n"
      << "           Input file   Unzip   Compare original     Report    \n\n"
      << "General Options:\n"
      << "  -h, --help                Show this help message and exit\n"
      << "  -v, --version             Show version information and exit\n\n"
      << "Analysis Options:\n"
      << "  -D, --dump                Dump the prediction/histogram output to "
         "file\n"
      << "  -R, --report              Generate a report\n"
      << "  -V, --verbose             Enable verbose output\n"
      << "  -S, --skip <write2disk>   Skip processing specified step\n"
      << "  --compare, --origin   Compare with original\n"
      << "  --stf                  Use cudastf API\n\n"
      << "Compression Parameters:\n"
      << "  -m, --mode <rel,abs>      Error bound mode (relative, absolute)\n"
      << "  -e, --eb, --error-bound <value>  Set error bound value\n"
      << "  -p, --pred, --predictor <lorenzo,lorenzo_zz,spline>  Set "
         "predictor\n"
      << "  --hist, --histogram <generic,sparse>  Histogram type\n"
      << "  -c1, --codec, --codec1 <huffman,huff_revisit,fzg>  Primary codec\n"
      << "  -c2, --codec2 <none,zstd,gzip>  Secondary codec\n"
      << "  -t, --type, --dtype <f32,f64>  Data type (float32, float64)\n"
      << "  -r, --radius <value>  Set histogram radius (default: 512)\n"
      << "  -b, --outlier-buffer-ratio <value>  Set outlier buffer ratio "
         "(default: 0.2)\n\n"
      << "Input/Output Options:\n"
      << "  -i, --input <filename>    Input file\n"
      << "  -l, --len, --xyz <val>x<val>x<val>  Dimensions (e.g., "
         "100x200x300)\n"
      << "  -z, --zip, --compress     Compress mode\n"
      << "  -x, --unzip, --decompress Decompress mode\n\n"
      << "Examples:\n"
      << "  Compression Pipelines:\n"
      << "    1. (default) Lorenzo -> Histogram -> Huffman -> None\n"
      << "    ./fzmod -t f32 -m rel -e 1e-4 -i data.f32 -l 500x500x500 -z \\ \n"
      << "        --pred lorenzo --hist sparse --codec huffman\n"
      << "    2. (quality) Spline 3D -> Histogram -> Huffman -> None\n"
      << "    ./fzmod -t f32 -m abs -e 1e-3 -i data.f32 -l 500x500x500 -z \\ \n"
      << "        --pred spline --hist generic --codec huffman\n"
      << "    3. (speed) Lorenzo -> FZG -> None\n"
      << "    ./fzmod -t f32 -m rel -e 1e-4 -i data.f32 -l 500x500x500 -z \\ \n"
      << "        --pred lorenzo --codec fzg\n"
      << "  Testing Data:\n"
      << "    Get testing data from Scientific Data Reduction Benchmarks "
         "(SDRB)\n"
      << "    at https://sdrbench.github.io\n\n"
      << "    2D CESM example (compression and decompression):\n"
      << "    ./fzmod -t f32 -m rel -e 1e-4 -i ${CESM} -l 3600-1800 -z --report\n"
      << "    ./fzmod -i ${CESM}.fzmod -x --report --compare ${CESM}\n";
  exit(0);
}

void print_version() {
  // print using definition in header (100 for version 1.0.0)
  printf("FZModules Version %d.%d.%d\n", (VERSION / 100) % 100,
         (VERSION / 10) % 10, VERSION % 10);
  printf("Github: https://github.com/skyler-ruiter/FZModules\n");
  exit(0);
}

void parse_args(int const argc, char** const argv, CLIOptions& options) {

  int i = 1;

  options.pipeline.push_back("f32");
  options.pipeline.push_back("rel");
  options.pipeline.push_back("lorenzo");
  options.pipeline.push_back("generic");
  options.pipeline.push_back("huffman");
  options.pipeline.push_back("none");

  auto check_next = [&]() {
    if (i + 1 >= argc) throw std::runtime_error("out-of-range at: " + std::string(argv[i]));
  };

  std::string opt;
  auto optmatch = [&](std::vector<std::string> vs) -> bool {
    for (auto& i : vs) {
      if (opt == i) return true;
    }
    return false;
  };

  while (i < argc) {
    if (argv[i][0] == '-') {
      opt = std::string(argv[i]);

      if (optmatch({"-h", "--help"})) {
        options.help = true;
        print_help();
      } else if (optmatch({"-v", "--version"})) {
        options.version = true;
        print_version();
      } else if (optmatch({"--stf"})) {
        options.stf = true;
      } else if (optmatch({"-D", "--dump"})) {
        options.dump = true;
      } else if (optmatch({"-R", "--report"})) {
        options.report = true;
      } else if (optmatch({"-V", "--verbose"})) {
        options.verbose = true;
      } else if (optmatch({"-S", "--skip"})) {
        check_next();
        options.skip = true;
        auto _ = std::string(argv[++i]);
        if (_ == "huffman") {
          options.skip_huffman = true;
          throw std::runtime_error("Skipping huffman is not supported yet.");
        } else if (_ == "write2disk") {
          options.skip_to_file = true;
        } else {
          throw std::runtime_error("Unknown skip option: " + _);
        }
      } else if (optmatch({"--compare", "--origin"})) {
        check_next();
        auto _ = std::string(argv[++i]);
        options.origin = _;
        if (options.origin.empty()) {
          throw std::runtime_error("Origin file cannot be empty.");
        }
      } else if (optmatch({"-m", "--mode"})) {
        check_next();
        auto _ = std::string(argv[++i]);
        if (_ == "rel") {
          options.pipeline[EB_TYPE] = "rel";
        } else if (_ == "abs") {
          options.pipeline[EB_TYPE] = "abs";
        } else {
          throw std::runtime_error("Unknown error bound mode: " + _);
        }
      } else if (optmatch({"-e", "--eb", "--error-bound"})) {
        check_next();
        char* end;
        options.eb = std::strtod(argv[++i], &end);
      } else if (optmatch({"-p", "--pred", "--predictor"})) {
        check_next();
        auto _ = std::string(argv[++i]);
        if (_ == "lorenzo") {
          options.pipeline[PREDICTOR] = "lorenzo";
        } else if (_ == "lorenzo_zz") {
          options.pipeline[PREDICTOR] = "lorenzo_zz";
        } else if (_ == "spline") {
          options.pipeline[PREDICTOR] = "spline";
        } else {
          throw std::runtime_error("Unknown predictor: " + _);
        }
      } else if (optmatch({"--hist", "--histogram"})) {
        check_next();
        auto _ = std::string(argv[++i]);
        if (_ == "generic") {
          options.pipeline[HISTOGRAM] = "generic";
        } else if (_ == "sparse") {
          options.pipeline[HISTOGRAM] = "sparse";
        } else {
          throw std::runtime_error("Unknown histogram type: " + _);
        }
      } else if (optmatch({"-c1", "--codec", "--codec1"})) {
        check_next();
        auto _ = std::string(argv[++i]);
        if (_ == "huffman") {
          options.pipeline[CODEC] = "huffman";
        } else if (_ == "huff_revisit") {
          options.pipeline[CODEC] = "huff_revisit";
        } else if (_ == "fzg") {
          options.pipeline[CODEC] = "fzg";
        } else {
          throw std::runtime_error("Unknown primary codec: " + _);
        }
      } else if (optmatch({"-c2", "--codec2"})) {
        check_next();
        auto _ = std::string(argv[++i]);
        if (_ == "none") {
          options.pipeline[SECONDARY_CODEC] = "none";
        } else if (_ == "zstd") {
          options.pipeline[SECONDARY_CODEC] = "zstd";
        } else if (_ == "gzip") {
          options.pipeline[SECONDARY_CODEC] = "gzip";
        } else {
          throw std::runtime_error("Unknown secondary codec: " + _);
        }
      } else if (optmatch({"-t", "--type", "-dtype"})) {
        check_next();
        auto _ = std::string(argv[++i]);
        if (_ == "f32") {
          options.pipeline[PRECISION] = "f32";
          options.float_precision = true;
        } else if (_ == "f64") {
          options.pipeline[PRECISION] = "f64";
          options.float_precision = false;
        } else {
          throw std::runtime_error("Unknown data type: " + _);
        }
      } else if (optmatch({"-r", "--radius"})) {
        check_next();
        char* end;
        options.radius = std::strtol(argv[++i], &end, 10);
        if (*end) {
          throw std::runtime_error("Invalid radius value: " + std::string(argv[i]));
        }
        if (options.radius == 0) {
          throw std::runtime_error("Radius cannot be zero.");
        }
      } else if (optmatch({"-b", "--outlier-buffer-ratio"})) {
        check_next();
        char* end;
        options.outlier_buff_ratio = std::strtof(argv[++i], &end);
        if (*end) {
          throw std::runtime_error("Invalid outlier buffer ratio value: " + std::string(argv[i]));
        }
        if (options.outlier_buff_ratio <= 0) {
          throw std::runtime_error("Outlier buffer ratio must be greater than zero.");
        }
      } else if (optmatch({"-i", "--input"})) {
        check_next();
        options.input_file = std::string(argv[++i]);
        if (options.input_file.empty()) {
          throw std::runtime_error("Input file cannot be empty.");
        }
      } else if (optmatch({"-l", "--len", "--xyz"})) {
        check_next();
        std::vector<std::string> dims;
        auto parse_len = [&](const std::string& str) {
          std::stringstream data_len_ss(str);
          auto data_len_literal = data_len_ss.str();
          auto checked = false;
          for (auto s : {"x", "*", "-", ",", "m"}) {
            if (checked) break;
            char delimiter = s[0];
            if (data_len_literal.find(delimiter) != std::string::npos) {
              while (data_len_ss.good()) {
                std::string substr;
                std::getline(data_len_ss, substr, delimiter);
                dims.push_back(substr);
              }
              checked = true;
            }
          }
          if (!checked) {
            dims.push_back(str);
          }
        };
        // lambda to convert string to int returning uint32_t
        auto str2int = [&](const char* s) -> uint32_t {
          char* end;
          auto res = std::strtol(s, &end, 10);
          if (*end) {
            throw std::runtime_error("Invalid dimension value: " + std::string(s));
          }
          return res;
        };
        parse_len(std::string(argv[++i]));
        options.n_dims = dims.size();
        options.x = str2int(dims[0].c_str());
        if (options.n_dims > 1) {
          options.y = str2int(dims[1].c_str());
          if (options.n_dims > 2) {
            options.z = str2int(dims[2].c_str());
          } else {
            options.z = 1; // default z dimension
          }
        } else {
          options.y = 1; // default y dimension
          options.z = 1; // default z dimension
        }
        options.len = options.x * options.y * options.z;
        if (options.x == 0 || options.y == 0 || options.z == 0) {
          throw std::runtime_error("Invalid dimensions: " + std::to_string(options.x) + "x" + 
                                   std::to_string(options.y) + "x" + std::to_string(options.z));
        }
        if (options.n_dims < 1 || options.n_dims > 3) {
          throw std::runtime_error("Invalid number of dimensions: " + std::to_string(options.n_dims) + 
                                   ". Expected 1, 2, or 3 dimensions.");
        }
      } else if (optmatch({"-z", "--zip", "--compress"})) {
        options.comp = true; // enable compression
      } else if (optmatch({"-x", "--unzip", "--decompress"})) {
        options.comp = false; // disable compression
      } else {
        throw std::runtime_error("Unknown option: " + opt);
      }
    } else {
      throw std::runtime_error("Invalid argument: " + std::string(argv[i]));
    }
    i++;
  } // while
} // parse_args

void validate_cli(const CLIOptions& options) {
  if (options.comp) {
    if (options.input_file.empty()) {
      throw std::runtime_error("Input file is required for compression.");
    }
    if (options.eb <= 0) {
      throw std::runtime_error("Error bound must be greater than zero.");
    }
    if (options.x == 0 || options.y == 0 || options.z == 0) {
      throw std::runtime_error("Invalid dimensions: " + std::to_string(options.x) + "x" + 
                               std::to_string(options.y) + "x" + std::to_string(options.z));
    }
  } else {
    if (options.input_file.empty()) {
      throw std::runtime_error("Input file is required for decompression.");
    }
  }
} // validate_cli

template <typename T>
void apply_cli_options(const CLIOptions& options, fz::Config<T>& config) {
  config.toFile = !options.skip_to_file;
  config.fromFile = true;  // always read from file
  config.fname = options.input_file;

  config.report = options.report;
  config.dump = options.dump;
  config.verbose = options.verbose;

  if (options.comp) {
    config.comp = options.comp;
    config.eb = options.eb;
    config.eb_type = (options.pipeline[EB_TYPE] == "rel") ? fz::EB_TYPE::REL
                                                          : fz::EB_TYPE::ABS;
    config.algo = (options.pipeline[PREDICTOR] == "lorenzo" ||
                   options.pipeline[PREDICTOR] == "lorenzo_zz")
                      ? fz::ALGO::LORENZO
                      : fz::ALGO::SPLINE;
    config.codec = (options.pipeline[CODEC] == "huffman" ||
                    options.pipeline[CODEC] == "huff_revisit")
                       ? fz::CODEC::HUFFMAN
                       : fz::CODEC::FZG;
    config.lossless_codec_2 = (options.pipeline[SECONDARY_CODEC] == "none")
                                  ? fz::SECONDARY_CODEC::NONE
                              : (options.pipeline[SECONDARY_CODEC] == "zstd")
                                  ? fz::SECONDARY_CODEC::ZSTD
                                  : fz::SECONDARY_CODEC::GZIP;
    config.outlier_buffer_ratio = options.outlier_buff_ratio;
    config.radius = options.radius;
    config.use_histogram_sparse = (options.pipeline[HISTOGRAM] == "sparse");
    config.use_huffman_reVISIT = options.pipeline[CODEC] == "huff_revisit";
    config.use_lorenzo_zigzag = (options.pipeline[PREDICTOR] == "lorenzo_zz");
  } else {
    config.compare = !options.origin.empty();
  }
} // apply_cli_options

template<typename T>
void run_compression(const CLIOptions& options, cudaStream_t stream) {
  if (!options.stf) {
    fz::Config<T> config(options.x, options.y, options.z);
    apply_cli_options<T>(options, config);
    fz::Compressor<T> compressor(config);
    uint8_t* out_data;
    compressor.compress(nullptr, &out_data, stream);
    cudaFree(out_data);
  } else {
    #ifdef FZMOD_CUDASTF
    fz::Config<T> config(options.x, options.y, options.z);
    apply_cli_options<T>(options, config);
    context ctx;
    T* input_data_host;
    cudaMallocHost(&input_data_host, config.orig_size);
    utils::fromfile(options.input_file.c_str(), input_data_host,
                    config.orig_size);
    fz::STF_Compressor<T> compressor(config, ctx, input_data_host);
    compressor.compress(stream);
    #else
    throw std::runtime_error("STF compression is not built/enabled.");
    #endif  // FZMOD_CUDASTF
  }
} // run_compression

template<typename T>
void run_decompression(const CLIOptions& options, cudaStream_t stream) {
  if (!options.stf) {
    fz::Decompressor<T> decompressor(options.input_file);
    apply_cli_options<T>(options, *decompressor.conf);

    T* out_data;
    size_t out_size = decompressor.conf->orig_size;
    cudaMalloc(&out_data, out_size);

    uint8_t* compressed_data;
    cudaMallocHost(&compressed_data, decompressor.conf->comp_size);
    utils::fromfile(options.input_file.c_str(), compressed_data, decompressor.conf->comp_size);

    T* orig_data;
    if (options.origin.empty()) {
      decompressor.decompress(compressed_data, out_data, stream);
    } else {
      cudaMallocHost(&orig_data, decompressor.conf->orig_size);
      utils::fromfile(options.origin.c_str(), orig_data, decompressor.conf->orig_size);
      decompressor.decompress(compressed_data, out_data, stream, orig_data);
    }

    cudaFree(out_data);
    cudaFree(compressed_data);
    if (!options.origin.empty()) {
      cudaFree(orig_data);
    }
  } else {
    #ifdef FZMOD_CUDASTF
    context ctx;
    fz::STF_Decompressor<T> decompressor(options.input_file, ctx);
    apply_cli_options<T>(options, *decompressor.conf);
    // T* out_data;
    // size_t out_size = decompressor.conf->orig_size;
    // cudaMalloc(&out_data, out_size);

    uint8_t* compressed_data;
    cudaMallocHost(&compressed_data, decompressor.conf->comp_size);
    utils::fromfile(options.input_file.c_str(), compressed_data, decompressor.conf->comp_size);

    T* orig_data;
    if (options.origin.empty()) {
      decompressor.decompress(compressed_data, stream);
    } else {
      cudaMallocHost(&orig_data, decompressor.conf->orig_size);
      utils::fromfile(options.origin.c_str(), orig_data, decompressor.conf->orig_size);
      decompressor.decompress(compressed_data, stream, orig_data);
    }
    // cudaFree(out_data);
    cudaFree(compressed_data);
    if (!options.origin.empty()) {
      cudaFree(orig_data);
    }
    #else
    throw std::runtime_error("STF decompression is not built/enabled.");
    #endif // FZMOD_CUDASTF
  }
} // run_decompression

void run_CLI(int argc, char** argv) {
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  CLIOptions options;

  if (argc < 2) {
    print_help();
    return;
  }

  parse_args(argc, argv, options);
  validate_cli(options);

  if (options.help || options.version) return;

  if (options.comp) {
    if (options.pipeline[0] == "f32") {
      run_compression<float>(options, stream);
    } else if (options.pipeline[0] == "f64") {
      run_compression<double>(options, stream);
    } else {
      throw std::runtime_error("Unsupported data type: " + options.pipeline[0]);
    }
  } else {
    if (options.pipeline[0] == "f32") {
      run_decompression<float>(options, stream);
    } else if (options.pipeline[0] == "f64") {
      run_decompression<double>(options, stream);
    } else {
      throw std::runtime_error("Unsupported data type: " + options.pipeline[0]);
    }
  }
  cudaStreamDestroy(stream);
} // run_CLI

int main(int argc, char** argv) {
  run_CLI(argc, argv);
  return 0;
} // main