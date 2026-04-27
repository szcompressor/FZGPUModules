/**
 * tests/pipeline/test_config.cpp
 *
 * Unit and integration tests for Pipeline::loadConfig() / saveConfig()
 * and the Pipeline(config_path) constructor overload.
 *
 * Tests:
 *   ConfigLoad/LorenzoOnly          — load minimal single-stage config, round-trip
 *   ConfigLoad/FullPipeline         — load multi-stage Lorenzo→Bitshuffle→RZE, round-trip
 *   ConfigLoad/ConstructorOverload  — Pipeline("path.toml") matches loadConfig()
 *   ConfigSave/RoundTrip            — build programmatically, saveConfig, loadConfig, compare
 *   ConfigSave/PreservesParams      — saved TOML contains correct parameter values
 *   ConfigSave/RequiresFinalized    — saveConfig throws when not finalized
 *   ConfigLoad/AlreadyFinalized     — loadConfig throws when already finalized
 *   ConfigLoad/MissingFile          — loadConfig throws on nonexistent path
 *   ConfigLoad/BadStageType         — loadConfig throws on unknown type string
 *   ConfigLoad/BadWiringRef         — loadConfig throws on missing 'from' stage name
 *   ConfigLoad/DuplicateStageName   — loadConfig throws on duplicate stage names
 */

#include <gtest/gtest.h>
#include "helpers/fz_test_utils.h"
#include "fzgpumodules.h"

// toml++ in header-only mode for reading back saved configs in assertions
#define TOML_HEADER_ONLY 1
#include <toml++/toml.hpp>

#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using namespace fz;
using namespace fz_test;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

static std::vector<float> make_smooth_data(size_t n) {
    std::vector<float> v(n);
    for (size_t i = 0; i < n; i++)
        v[i] = std::sin(static_cast<float>(i) * 0.01f) * 50.0f
             + std::cos(static_cast<float>(i) * 0.003f) * 20.0f;
    return v;
}

// Write a TOML string to a temp file and return the path.
static std::string write_toml(const std::string& name, const std::string& content) {
    std::string path = "/tmp/fzgmod_cfg_test_" + name + ".toml";
    std::ofstream f(path);
    f << content;
    return path;
}

// Decompress helper: returns max abs error vs h_input.
static float round_trip_error(Pipeline& p, const std::vector<float>& h_input,
                               CudaStream& stream) {
    size_t in_bytes = h_input.size() * sizeof(float);
    CudaBuffer<float> d_in(h_input.size());
    d_in.upload(h_input, stream);
    stream.sync();

    void* d_comp = nullptr; size_t comp_sz = 0;
    p.compress(d_in.void_ptr(), in_bytes, &d_comp, &comp_sz, stream);
    stream.sync();

    void* d_dec = nullptr; size_t dec_sz = 0;
    p.decompress(nullptr, comp_sz, &d_dec, &dec_sz, stream);
    stream.sync();

    std::vector<float> h_recon(h_input.size());
    cudaError_t cpy_err = cudaMemcpy(h_recon.data(), d_dec, dec_sz, cudaMemcpyDeviceToHost);
    EXPECT_EQ(cpy_err, cudaSuccess) << "cudaMemcpy failed: " << cudaGetErrorString(cpy_err);
    cudaFree(d_dec);

    return max_abs_error(h_input, h_recon);
}

// ─────────────────────────────────────────────────────────────────────────────
// ConfigLoad — loading from a hand-authored TOML string
// ─────────────────────────────────────────────────────────────────────────────

TEST(ConfigLoad, LorenzoOnly) {
    constexpr size_t N  = 1 << 13;
    constexpr float  EB = 1e-2f;

    std::string path = write_toml("lorenzo_only", R"(
[pipeline]
input_size = 32768
memory_strategy = "MINIMAL"

[[stage]]
name             = "lrz"
type             = "LorenzoQuant"
input_type       = "float32"
code_type        = "uint16"
error_bound      = 0.01
error_bound_mode = "ABS"
quant_radius     = 512
outlier_capacity = 0.2
zigzag_codes     = false
)");

    CudaStream stream;
    Pipeline p;
    p.loadConfig(path);

    auto h_input = make_smooth_data(N);
    float err = round_trip_error(p, h_input, stream);
    EXPECT_LE(err, EB * 1.01f) << "loadConfig round-trip error " << err << " > EB " << EB;

    std::remove(path.c_str());
}

TEST(ConfigLoad, FullPipeline) {
    // Lorenzo → Bitshuffle (codes branch) → RZE
    constexpr size_t N  = 1 << 14;
    constexpr float  EB = 1e-2f;
    size_t in_bytes = N * sizeof(float);

    std::string path = write_toml("full_pipeline", R"(
[pipeline]
input_size       = 65536
memory_strategy  = "PREALLOCATE"
pool_multiplier  = 4.0
num_streams      = 1

[[stage]]
name             = "lrz"
type             = "LorenzoQuant"
input_type       = "float32"
code_type        = "uint16"
error_bound      = 0.01
error_bound_mode = "ABS"
quant_radius     = 512
outlier_capacity = 0.2
zigzag_codes     = false

[[stage]]
name          = "bshuf"
type          = "Bitshuffle"
block_size    = 16384
element_width = 2
inputs = [{ from = "lrz", port = "codes" }]

[[stage]]
name       = "rze"
type       = "RZE"
chunk_size = 16384
levels     = 4
inputs = [{ from = "bshuf" }]
)");

    CudaStream stream;
    Pipeline p(in_bytes, MemoryStrategy::PREALLOCATE, 4.0f);
    p.loadConfig(path);

    auto h_input = make_smooth_data(N);
    float err = round_trip_error(p, h_input, stream);
    EXPECT_LE(err, EB * 1.01f) << "Full pipeline loadConfig round-trip error " << err;

    std::remove(path.c_str());
}

TEST(ConfigLoad, ConstructorOverload) {
    // Pipeline(path) should produce identical results to loadConfig(path).
    constexpr size_t N  = 1 << 13;
    constexpr float  EB = 1e-2f;

    std::string path = write_toml("ctor_overload", R"(
[pipeline]
input_size       = 32768
memory_strategy  = "MINIMAL"

[[stage]]
name             = "lrz"
type             = "LorenzoQuant"
input_type       = "float32"
code_type        = "uint16"
error_bound      = 0.01
error_bound_mode = "ABS"
quant_radius     = 512
outlier_capacity = 0.2
zigzag_codes     = false
)");

    CudaStream stream;
    auto h_input = make_smooth_data(N);

    // Via constructor
    Pipeline p1(path);
    float err1 = round_trip_error(p1, h_input, stream);

    // Via loadConfig (fresh pipeline)
    Pipeline p2;
    p2.loadConfig(path);
    float err2 = round_trip_error(p2, h_input, stream);

    EXPECT_LE(err1, EB * 1.01f);
    EXPECT_LE(err2, EB * 1.01f);
    // Both paths should produce the same reconstruction error.
    EXPECT_NEAR(err1, err2, 1e-5f);

    std::remove(path.c_str());
}

// ─────────────────────────────────────────────────────────────────────────────
// ConfigSave — saving a built pipeline to TOML
// ─────────────────────────────────────────────────────────────────────────────

TEST(ConfigSave, RoundTrip) {
    // Build a pipeline programmatically, save it, reload it, verify compress quality.
    constexpr size_t N  = 1 << 13;
    constexpr float  EB = 1e-2f;
    size_t in_bytes = N * sizeof(float);

    // ── Build & save ──────────────────────────────────────────────────────────
    Pipeline p1(in_bytes, MemoryStrategy::MINIMAL);
    auto* lrz = p1.addStage<LorenzoQuantStage<float, uint16_t>>();
    lrz->setErrorBound(EB);
    lrz->setQuantRadius(512);
    lrz->setOutlierCapacity(0.15f);
    lrz->setZigzagCodes(true);
    p1.setPoolManagedDecompOutput(false);
    p1.finalize();

    std::string cfg_path = "/tmp/fzgmod_save_roundtrip.toml";
    p1.saveConfig(cfg_path);

    // ── Reload & compare ──────────────────────────────────────────────────────
    Pipeline p2;
    p2.loadConfig(cfg_path);

    CudaStream stream;
    auto h_input = make_smooth_data(N);
    float err = round_trip_error(p2, h_input, stream);
    EXPECT_LE(err, EB * 1.01f)
        << "saveConfig → loadConfig round-trip error " << err << " > EB " << EB;

    std::remove(cfg_path.c_str());
}

TEST(ConfigSave, PreservesParams) {
    // Check that the saved TOML actually contains the right values by parsing it.
    constexpr float  EB      = 5e-3f;
    constexpr int    QR      = 1024;
    constexpr float  OUTCAP  = 0.12f;
    constexpr size_t CHUNK   = 8192;
    constexpr int    LEVELS  = 3;
    constexpr size_t BSBLOCK = 8192;
    constexpr size_t BSWIDTH = 2;

    size_t in_bytes = (1 << 13) * sizeof(float);
    Pipeline p(in_bytes, MemoryStrategy::PREALLOCATE, 5.0f);
    p.setNumStreams(2);

    auto* lrz = p.addStage<LorenzoQuantStage<float, uint16_t>>();
    lrz->setErrorBound(EB);
    lrz->setQuantRadius(QR);
    lrz->setOutlierCapacity(OUTCAP);
    lrz->setZigzagCodes(true);
    lrz->setErrorBoundMode(ErrorBoundMode::ABS);

    auto* bs = p.addStage<BitshuffleStage>();
    bs->setBlockSize(BSBLOCK);
    bs->setElementWidth(BSWIDTH);
    p.connect(bs, lrz, "codes");

    auto* rze = p.addStage<RZEStage>();
    rze->setChunkSize(CHUNK);
    rze->setLevels(LEVELS);
    p.connect(rze, bs);

    p.setPoolManagedDecompOutput(false);
    p.finalize();

    std::string cfg_path = "/tmp/fzgmod_preserve_params.toml";
    p.saveConfig(cfg_path);

    // Parse the saved TOML and assert key fields.
    auto doc = toml::parse_file(cfg_path);

    // Pipeline-level
    EXPECT_EQ(doc["pipeline"]["memory_strategy"].value_or<std::string>(""), "PREALLOCATE");
    EXPECT_EQ(doc["pipeline"]["num_streams"].value_or<int64_t>(0), 2);
    EXPECT_NEAR(doc["pipeline"]["pool_multiplier"].value_or<double>(0.0), 5.0, 1e-4);

    // Stage array
    auto* stages = doc["stage"].as_array();
    ASSERT_NE(stages, nullptr);
    ASSERT_EQ(stages->size(), 3u);

    // Stage 0: Lorenzo
    auto& s0 = *(*stages)[0].as_table();
    EXPECT_EQ(s0["type"].value_or<std::string>(""), "LorenzoQuant");
    EXPECT_NEAR(s0["error_bound"].value_or<double>(0.0), static_cast<double>(EB), 1e-6);
    EXPECT_EQ(s0["quant_radius"].value_or<int64_t>(0), QR);
    EXPECT_NEAR(s0["outlier_capacity"].value_or<double>(0.0), static_cast<double>(OUTCAP), 1e-4);
    EXPECT_EQ(s0["zigzag_codes"].value_or<bool>(false), true);
    EXPECT_EQ(s0["error_bound_mode"].value_or<std::string>(""), "ABS");
    // No inputs key on the source stage
    EXPECT_FALSE(s0.contains("inputs"));

    // Stage 1: Bitshuffle
    auto& s1 = *(*stages)[1].as_table();
    EXPECT_EQ(s1["type"].value_or<std::string>(""), "Bitshuffle");
    EXPECT_EQ(s1["block_size"].value_or<int64_t>(0), static_cast<int64_t>(BSBLOCK));
    EXPECT_EQ(s1["element_width"].value_or<int64_t>(0), static_cast<int64_t>(BSWIDTH));
    ASSERT_TRUE(s1.contains("inputs"));

    // Stage 2: RZE
    auto& s2 = *(*stages)[2].as_table();
    EXPECT_EQ(s2["type"].value_or<std::string>(""), "RZE");
    EXPECT_EQ(s2["chunk_size"].value_or<int64_t>(0), static_cast<int64_t>(CHUNK));
    EXPECT_EQ(s2["levels"].value_or<int64_t>(0), static_cast<int64_t>(LEVELS));
    ASSERT_TRUE(s2.contains("inputs"));

    // Bitshuffle inputs: from="LorenzoQuant" (getName() return value), port="codes"
    auto* bs_inputs = s1["inputs"].as_array();
    ASSERT_NE(bs_inputs, nullptr);
    ASSERT_GE(bs_inputs->size(), 1u);
    auto& inp0 = *(*bs_inputs)[0].as_table();
    EXPECT_EQ(inp0["port"].value_or<std::string>(""), "codes");

    std::remove(cfg_path.c_str());
}

TEST(ConfigSave, RequiresFinalized) {
    Pipeline p;
    p.addStage<LorenzoQuantStage<float, uint16_t>>();
    // Not finalized — saveConfig must throw.
    EXPECT_THROW(p.saveConfig("/tmp/fzgmod_should_not_write.toml"), std::runtime_error);
}

// ─────────────────────────────────────────────────────────────────────────────
// ConfigLoad — error-handling
// ─────────────────────────────────────────────────────────────────────────────

TEST(ConfigLoad, AlreadyFinalized) {
    std::string path = write_toml("already_final", R"(
[pipeline]
[[stage]]
name = "lrz"
type = "LorenzoQuant"
input_type = "float32"
code_type  = "uint16"
error_bound = 0.01
)");

    Pipeline p;
    p.addStage<LorenzoQuantStage<float, uint16_t>>();
    p.setPoolManagedDecompOutput(false);
    p.finalize();

    EXPECT_THROW(p.loadConfig(path), std::runtime_error);
    std::remove(path.c_str());
}

TEST(ConfigLoad, MissingFile) {
    Pipeline p;
    EXPECT_THROW(p.loadConfig("/tmp/fzgmod_this_does_not_exist_xyz.toml"), std::runtime_error);
}

TEST(ConfigLoad, BadStageType) {
    std::string path = write_toml("bad_type", R"(
[pipeline]
[[stage]]
name = "s1"
type = "NonExistentStageType"
)");

    Pipeline p;
    EXPECT_THROW(p.loadConfig(path), std::runtime_error);
    std::remove(path.c_str());
}

TEST(ConfigLoad, BadWiringRef) {
    std::string path = write_toml("bad_wiring", R"(
[pipeline]
[[stage]]
name = "lrz"
type = "LorenzoQuant"
input_type = "float32"
code_type  = "uint16"
error_bound = 0.01

[[stage]]
name   = "rze"
type   = "RZE"
inputs = [{ from = "does_not_exist" }]
)");

    Pipeline p;
    EXPECT_THROW(p.loadConfig(path), std::runtime_error);
    std::remove(path.c_str());
}

TEST(ConfigLoad, DuplicateStageName) {
    std::string path = write_toml("dup_name", R"(
[pipeline]
[[stage]]
name = "lrz"
type = "LorenzoQuant"
input_type = "float32"
code_type  = "uint16"
error_bound = 0.01

[[stage]]
name = "lrz"
type = "RZE"
)");

    Pipeline p;
    EXPECT_THROW(p.loadConfig(path), std::runtime_error);
    std::remove(path.c_str());
}

// ─────────────────────────────────────────────────────────────────────────────
// ConfigSave / ConfigLoad round-trip for every supported stage type combination
// ─────────────────────────────────────────────────────────────────────────────

TEST(ConfigLoad, AllSupportedStageTypes) {
    // Build a valid linear pipeline exercising Lorenzo, Difference, RLE,
    // Bitshuffle, and RZE in one chain.  saveConfig → loadConfig must not throw.
    // (We don't run compress here — topology correctness is what's under test.)
    size_t in_bytes = (1 << 13) * sizeof(float);

    Pipeline p1(in_bytes, MemoryStrategy::MINIMAL);

    auto* lrz  = p1.addStage<LorenzoQuantStage<float, uint16_t>>();
    lrz->setErrorBound(1e-2f);

    // codes branch: Diff → RLE → Bitshuffle → RZE
    auto* diff = p1.addStage<DifferenceStage<uint16_t>>();
    p1.connect(diff, lrz, "codes");

    auto* rle  = p1.addStage<RLEStage<uint16_t>>();
    p1.connect(rle, diff);

    auto* bs   = p1.addStage<BitshuffleStage>();
    bs->setBlockSize(16384); bs->setElementWidth(2);
    p1.connect(bs, rle);

    auto* rze  = p1.addStage<RZEStage>();
    rze->setChunkSize(16384); rze->setLevels(2);
    p1.connect(rze, bs);

    // Lorenzo outlier outputs (ports 1-3) are unconnected → pipeline outputs.

    p1.setPoolManagedDecompOutput(false);
    p1.finalize();

    std::string cfg_path = "/tmp/fzgmod_all_types.toml";
    ASSERT_NO_THROW(p1.saveConfig(cfg_path));

    Pipeline p2;
    ASSERT_NO_THROW(p2.loadConfig(cfg_path));

    std::remove(cfg_path.c_str());
}

TEST(ConfigLoad, ZigzagAndNegabinary) {
    // Standalone Zigzag and Negabinary stage types.
    {
        std::string path = write_toml("zigzag", R"(
[pipeline]
[[stage]]
name        = "zz"
type        = "Zigzag"
input_type  = "int32"
output_type = "uint32"
)");
        Pipeline p;
        EXPECT_NO_THROW(p.loadConfig(path));
        std::remove(path.c_str());
    }
    {
        std::string path = write_toml("negabinary", R"(
[pipeline]
[[stage]]
name        = "nb"
type        = "Negabinary"
input_type  = "int16"
output_type = "uint16"
)");
        Pipeline p;
        EXPECT_NO_THROW(p.loadConfig(path));
        std::remove(path.c_str());
    }
}

// ── Lorenzo standalone via TOML load ─────────────────────────────────────────
TEST(ConfigLoad, LorenzoStandaloneLoads) {
    // Verify the "Lorenzo" type string is accepted by loadConfig.
    std::string path = write_toml("lorenzo_standalone", R"(
[pipeline]
[[stage]]
name      = "lrz"
type      = "Lorenzo"
data_type = "int32"
)");
    Pipeline p;
    EXPECT_NO_THROW(p.loadConfig(path));
    std::remove(path.c_str());
}

// ── Lorenzo saveConfig round-trip ─────────────────────────────────────────────
TEST(ConfigSave, LorenzoStandaloneSaveLoad) {
    // Build a pipeline with a Lorenzo stage, save it, reload it.
    const std::string cfg_path = "/tmp/fzgmod_lorenzo_save.toml";

    Pipeline p1(1024 * sizeof(int32_t), MemoryStrategy::PREALLOCATE);
    auto* lrz = p1.addStage<LorenzoStage<int32_t>>();
    lrz->setDims(1024);
    p1.finalize();

    ASSERT_NO_THROW(p1.saveConfig(cfg_path));

    // Reload and verify the type string is preserved.
    auto tbl = toml::parse_file(cfg_path);
    auto* stages = tbl["stage"].as_array();
    ASSERT_NE(stages, nullptr);
    ASSERT_GE(stages->size(), 1u);
    auto* s0 = (*stages)[0].as_table();
    ASSERT_NE(s0, nullptr);
    EXPECT_EQ((*s0)["type"].value_or<std::string>(""), "Lorenzo");
    EXPECT_EQ((*s0)["data_type"].value_or<std::string>(""), "int32");

    Pipeline p2;
    EXPECT_NO_THROW(p2.loadConfig(cfg_path));

    std::remove(cfg_path.c_str());
}

// ── Quantizer → Lorenzo round-trip via TOML config ────────────────────────────
TEST(ConfigLoad, QuantizerLorenzoRoundTrip) {
    constexpr size_t N  = 1 << 13;
    constexpr float  EB = 1e-2f;

    std::string path = write_toml("quant_lrz_nopack", R"(
[pipeline]
input_size = 32768
dims = [8192, 1, 1]
memory_strategy = "PREALLOCATE"

[[stage]]
name             = "quant"
type             = "Quantizer"
input_type       = "float32"
code_type        = "uint32"
error_bound      = 0.01
error_bound_mode = "ABS"
quant_radius     = 32768
outlier_capacity = 0.1
zigzag_codes     = false

[[stage]]
name      = "lrz"
type      = "Lorenzo"
data_type = "int32"
inputs    = [{from = "quant", port = "codes"}]
)");

    auto h_input = make_smooth_data(N);

    Pipeline p;
    ASSERT_NO_THROW(p.loadConfig(path));

    CudaStream cs;
    float err = round_trip_error(p, h_input, cs);

    EXPECT_LT(err, EB * 1.1f);
    std::remove(path.c_str());
}
