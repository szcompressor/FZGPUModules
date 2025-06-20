#include <iostream>
#include <string>

#include "fzmod_compressor.hh"
#include "fzmod_decompressor.hh"

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
  // bool auto = false;
  std::string origin;

  std::vector<std::string> pipeline;
  uint32_t x = 0;
  uint32_t y = 0;
  uint32_t z = 0;
  size_t len = 0;
  int n_dims = 0;

  bool comp = true;
  double eb = 0;
  bool float_precision = true;
};

void print_help() {
  std::cout
      << "Usage: fzmod_cli [options] <input_file>\n\n"
      << "General Options:\n"
      << "  -h, --help                Show this help message and exit\n"
      << "  -v, --version             Show version information and exit\n\n"
      << "Analysis Options:\n"
      << "  -D, --dump                Dump the contents of the file\n"
      << "  -R, --report <time,cr>    Generate a report (time, compression "
         "ratio)\n"
      << "  -V, --verbose             Enable verbose output\n"
      << "  -S, --skip <huffman,write2disk>   Skip processing specified step\n"
      << "      --compare, --origin   Compare with original\n\n"
      << "Compression Parameters:\n"
      << "  -m, --mode <rel,abs>      Error bound mode (relative, absolute)\n"
      << "  -e, --eb, --error-bound <value>  Set error bound value\n"
      << "  -p, --pred, --predictor <lorenzo,lorenzo_zz,spline>  Set "
         "predictor\n"
      << "      --hist, --histogram <generic,sparse>  Histogram type\n"
      << "  -c1, --codec, --codec1 <huffman,huff_revisit,fzg>  Primary codec\n"
      << "  -c2, --codec2 <none,zstd,gzip>  Secondary codec\n"
      << "  -t, --type, --dtype <f32,f64>  Data type (float32, float64)\n\n"
      << "Input/Output Options:\n"
      << "  -i, --input <filename>    Input file\n"
      << "  -l, --len, --xyz <val>x<val>x<val>  Dimensions (e.g., "
         "100x200x300)\n"
      << "  -z, --zip, --compress     Compress mode\n"
      << "  -x, --unzip, --decompress Decompress mode\n";
}

void print_version() {
  // print using definition in header (100 for version 1.0.0)
  printf("FZModules Version %d.%d.%d\n", (VERSION / 100) % 100,
         (VERSION / 10) % 10, VERSION % 10);
  printf("Github: https://github.com/skyler-ruiter/FZModules\n");
}

void parse_args(int const argc, char** const argv, CLIOptions& options) {

  int i = 1;

  int constexpr PRECISION = 0; // index into pipeline
  int constexpr EB_TYPE = 1; // index into pipeline
  int constexpr PREDICTOR = 2; // index into pipeline
  int constexpr HISTOGRAM = 3; // index into pipeline
  int constexpr CODEC = 4; // index into pipeline
  int constexpr SECONDARY_CODEC = 5; // index into pipeline

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
      } else if (optmatch({"-D", "--dump"})) {
        options.dump = true;
      } else if (optmatch({"-R", "--report"})) {
        check_next();
        options.report = true;
      } else if (optmatch({"-V", "--verbose"})) {
        options.verbose = true;
      } else if (optmatch({"-S", "--skip"})) {
        check_next();
        options.skip = true;
        auto _ = std::string(argv[++i]);
        if (_ == "huffman") {
          options.skip_huffman = true;
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
        options.eb - std::strtod(argv[++i], &end);
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

template<typename T>
void run_compression(const CLIOptions& options, cudaStream_t stream) {

}

void run_CLI(int argc, char** argv) {
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  CLIOptions options;

  parse_args(argc, argv, options);

  if (options.help || options.version) {
    return;
  }

  if (options.comp) {
    if (options.pipeline[0] == "f32") {
      run_compression<float>(options, stream);
    } else if (options.pipeline[0] == "f64") {
      run_compression<double>(options, stream);
    } else {
      throw std::runtime_error("Unsupported data type: " + options.pipeline[0]);
    }
  } else {
    // Decompression logic here
    // For now, just a placeholder
    std::cout << "Decompression not implemented yet." << std::endl;
  }
}

int main(int argc, char** argv) {
  run_CLI(argc, argv);
  return 0;
}