#include <gtest/gtest.h>

#include "helpers/fz_test_utils.h"
#include "fzgpumodules.h"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

// CLI entrypoint implemented in utils/cli/cli.cpp and linked via libfzgmod.
int fzgmod_cli_main(int argc, char** argv);

using namespace fz;

namespace {

class TempWorkspace {
public:
    TempWorkspace() {
        const auto stamp = std::chrono::high_resolution_clock::now()
                               .time_since_epoch()
                               .count();
        dir_ = std::filesystem::temp_directory_path() /
               ("fzgmod_cli_test_" + std::to_string(stamp));
        std::filesystem::create_directories(dir_);
    }

    ~TempWorkspace() {
        std::error_code ec;
        std::filesystem::remove_all(dir_, ec);
    }

    std::filesystem::path file(const std::string& name) const {
        return dir_ / name;
    }

private:
    std::filesystem::path dir_;
};

static void write_float_file(const std::filesystem::path& path,
                             const std::vector<float>& values) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        throw std::runtime_error("write_float_file: cannot open " + path.string());
    }
    if (!values.empty()) {
        out.write(reinterpret_cast<const char*>(values.data()),
                  static_cast<std::streamsize>(values.size() * sizeof(float)));
    }
}

static std::vector<float> read_float_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in.is_open()) {
        throw std::runtime_error("read_float_file: cannot open " + path.string());
    }

    const std::streamsize bytes = in.tellg();
    if (bytes < 0 || (bytes % static_cast<std::streamsize>(sizeof(float))) != 0) {
        throw std::runtime_error("read_float_file: invalid byte count in " + path.string());
    }

    std::vector<float> values(static_cast<size_t>(bytes) / sizeof(float));
    in.seekg(0, std::ios::beg);
    if (!values.empty()) {
        in.read(reinterpret_cast<char*>(values.data()), bytes);
        if (!in) {
            throw std::runtime_error("read_float_file: short read from " + path.string());
        }
    }
    return values;
}

static int run_cli(std::vector<std::string> args) {
    std::vector<char*> argv;
    argv.reserve(args.size());
    for (auto& s : args) {
        argv.push_back(const_cast<char*>(s.c_str()));
    }
    return fzgmod_cli_main(static_cast<int>(argv.size()), argv.data());
}

} // namespace

TEST(CLI, CompressMissingRequiredArgsFails) {
    const int rc = run_cli({"fzgmod-cli", "compress"});
    EXPECT_NE(rc, 0);
}

TEST(CLI, VersionFlagWorks) {
    const int rc = run_cli({"fzgmod-cli", "-v"});
    EXPECT_EQ(rc, 0);
}

TEST(CLI, CompressCreatesFzmFile) {
    TempWorkspace tmp;

    constexpr size_t kN = 1 << 12;
    constexpr float kEb = 1e-3f;

    std::vector<float> input = fz_test::make_sine_floats(kN, 0.01f, 5.0f);
    const auto input_path = tmp.file("input.f32");
    const auto compressed_path = tmp.file("compressed.fzm");

    write_float_file(input_path, input);

    const int compress_rc = run_cli({
        "fzgmod-cli",
        "compress",
        "--input", input_path.string(),
        "--output", compressed_path.string(),
        "--pipeline", "pfpl",
        "--mode", "abs",
        "--error-bound", std::to_string(kEb),
        "--strategy", "minimal"
    });
    ASSERT_EQ(compress_rc, 0);
    ASSERT_TRUE(std::filesystem::exists(compressed_path));
    ASSERT_GT(std::filesystem::file_size(compressed_path), 0u);
}

TEST(CLI, ShorthandCompressCreatesFzmFile) {
    TempWorkspace tmp;

    constexpr size_t kN = 1 << 12;
    std::vector<float> input = fz_test::make_sine_floats(kN, 0.01f, 5.0f);
    const auto input_path = tmp.file("short_input.f32");
    const auto compressed_path = tmp.file("short_compressed.fzm");

    write_float_file(input_path, input);

    const int rc = run_cli({
        "fzgmod-cli",
        "-z",
        "-i", input_path.string(),
        "-o", compressed_path.string(),
        "-l", "4096x1x1",
        "-t", "f32",
        "-m", "abs",
        "-e", "1e-3",
        "--pipeline", "lorenzo-rze",
        "--strategy", "minimal"
    });

    ASSERT_EQ(rc, 0);
    ASSERT_TRUE(std::filesystem::exists(compressed_path));
    ASSERT_GT(std::filesystem::file_size(compressed_path), 0u);
}

TEST(CLI, DecompressCommandRoundTripsKnownGoodFile) {
    TempWorkspace tmp;

    constexpr size_t kN = 1 << 12;
    constexpr float kEb = 1e-2f;

    std::vector<float> input = fz_test::make_sine_floats(kN, 0.01f, 5.0f);
    const size_t in_bytes = kN * sizeof(float);

    const auto compressed_path = tmp.file("known_good.fzm");
    const auto output_path = tmp.file("output.f32");

    {
        fz_test::CudaStream stream;
        fz_test::CudaBuffer<float> d_in(kN);
        d_in.upload(input, stream);
        stream.sync();

        Pipeline pipeline(in_bytes, MemoryStrategy::MINIMAL);
        auto* lrz = pipeline.addStage<LorenzoStage<float, uint16_t>>();
        auto* diff = pipeline.addStage<DifferenceStage<uint16_t>>();
        lrz->setErrorBound(kEb);
        lrz->setQuantRadius(512);
        lrz->setOutlierCapacity(0.2f);
        pipeline.connect(diff, lrz, "codes");
        pipeline.finalize();

        void* d_comp = nullptr;
        size_t comp_sz = 0;
        pipeline.compress(d_in.void_ptr(), in_bytes, &d_comp, &comp_sz, stream);
        pipeline.writeToFile(compressed_path.string(), stream);
    }

    const int decompress_rc = run_cli({
        "fzgmod-cli",
        "decompress",
        "--input", compressed_path.string(),
        "--output", output_path.string()
    });
    ASSERT_EQ(decompress_rc, 0);
    ASSERT_TRUE(std::filesystem::exists(output_path));

    const std::vector<float> recon = read_float_file(output_path);
    ASSERT_EQ(recon.size(), input.size());

    const float max_err = fz_test::max_abs_error(input, recon);
    EXPECT_LE(max_err, kEb * 1.05f);
}

TEST(CLI, ShorthandDecompressWorks) {
    TempWorkspace tmp;

    constexpr size_t kN = 1 << 12;
    constexpr float kEb = 1e-2f;

    std::vector<float> input = fz_test::make_sine_floats(kN, 0.01f, 5.0f);
    const size_t in_bytes = kN * sizeof(float);

    const auto compressed_path = tmp.file("short_known_good.fzm");
    const auto output_path = tmp.file("short_output.f32");

    {
        fz_test::CudaStream stream;
        fz_test::CudaBuffer<float> d_in(kN);
        d_in.upload(input, stream);
        stream.sync();

        Pipeline pipeline(in_bytes, MemoryStrategy::MINIMAL);
        auto* lrz = pipeline.addStage<LorenzoStage<float, uint16_t>>();
        auto* diff = pipeline.addStage<DifferenceStage<uint16_t>>();
        lrz->setErrorBound(kEb);
        lrz->setQuantRadius(512);
        lrz->setOutlierCapacity(0.2f);
        pipeline.connect(diff, lrz, "codes");
        pipeline.finalize();

        void* d_comp = nullptr;
        size_t comp_sz = 0;
        pipeline.compress(d_in.void_ptr(), in_bytes, &d_comp, &comp_sz, stream);
        pipeline.writeToFile(compressed_path.string(), stream);
    }

    const int rc = run_cli({
        "fzgmod-cli",
        "-x",
        "-i", compressed_path.string(),
        "-o", output_path.string()
    });

    ASSERT_EQ(rc, 0);
    ASSERT_TRUE(std::filesystem::exists(output_path));
    ASSERT_EQ(std::filesystem::file_size(output_path), in_bytes);
}

TEST(CLI, PipelineFromConfigWorksWithOverride) {
    TempWorkspace tmp;

    constexpr size_t kN = 1 << 11;

    std::vector<float> input = fz_test::make_sine_floats(kN, 0.02f, 3.0f);
    const auto input_path = tmp.file("cfg_input.f32");
    const auto config_path = tmp.file("pipeline.cfg");
    const auto override_output_path = tmp.file("cfg_output_override.fzm");

    write_float_file(input_path, input);

    {
        std::ofstream cfg(config_path, std::ios::trunc);
        ASSERT_TRUE(cfg.is_open());
        cfg << "input=" << input_path.string() << "\n";
        cfg << "output=" << tmp.file("unused_from_config.fzm").string() << "\n";
        cfg << "pipeline=pfpl\n";
        cfg << "mode=abs\n";
        cfg << "error-bound=1e-3\n";
        cfg << "strategy=minimal\n";
    }

    const int rc = run_cli({
        "fzgmod-cli",
        "pipeline-from-config",
        "--config", config_path.string(),
        "--output", override_output_path.string()
    });

    ASSERT_EQ(rc, 0);
    ASSERT_TRUE(std::filesystem::exists(override_output_path));
    ASSERT_GT(std::filesystem::file_size(override_output_path), 0u);
}

TEST(CLI, BenchmarkCompressSingleRunWorks) {
    TempWorkspace tmp;

    constexpr size_t kN = 1 << 11;
    std::vector<float> input = fz_test::make_sine_floats(kN, 0.015f, 2.0f);
    const auto input_path = tmp.file("bench_input.f32");
    write_float_file(input_path, input);

    const int rc = run_cli({
        "fzgmod-cli",
        "benchmark",
        "--input", input_path.string(),
        "--pipeline", "pfpl",
        "--mode", "abs",
        "--error-bound", "1e-3",
        "--strategy", "minimal",
        "--runs", "1",
        "--phase", "compress"
    });

    EXPECT_EQ(rc, 0);
}
