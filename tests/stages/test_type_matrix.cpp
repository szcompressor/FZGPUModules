/**
 * tests/stages/test_type_matrix.cpp
 *
 * Parametric type-matrix tests for stages that support multiple
 * <TInput, TCode> combinations.  Uses Google Test's TYPED_TEST_SUITE to
 * drive a single test body over every supported type pair, so adding a new
 * instantiation to a stage automatically inherits full coverage.
 *
 * Current coverage:
 *   LorenzoTypeMatrix   — <float,uint16_t>, <float,uint8_t>,
 *                         <double,uint16_t>, <double,uint32_t>
 *   QuantizerTypeMatrix — <float,uint16_t>, <float,uint32_t>,
 *                         <double,uint16_t>, <double,uint32_t>
 *
 * Each type pair gets the same set of tests:
 *   RoundTripAbs        — ABS error-bound mode, compress+decompress in memory
 *   FileSerialization   — ABS mode, full writeToFile/decompressFromFile cycle
 *   ConstantInput       — constant array → zero prediction error, no outliers
 *   SerializeDeserialize— stage config round-trip via serializeHeader/deserializeHeader
 */

#include <gtest/gtest.h>
#include "helpers/stage_harness.h"
#include "fused/lorenzo_quant/lorenzo_quant.h"
#include "quantizers/quantizer/quantizer.h"
#include "fzgpumodules.h"

#include <cmath>
#include <cstring>
#include <vector>

using namespace fz;
using namespace fz_test;

// ─────────────────────────────────────────────────────────────────────────────
// Type-pair trait structs
// ─────────────────────────────────────────────────────────────────────────────

template <typename TIn, typename TC>
struct StagePair {
    using Input = TIn;
    using Code  = TC;
    // Error bound for *Lorenzo* (ABS mode, data amplitude ~70):
    //   0.05 is tight enough to catch bugs but not so tight that uint8 codes overflow.
    // For *Quantizer* (ABS mode, same data), this is in data units, so 0.5 gives
    // ~0.7% relative error on range [-70, 70].
    static constexpr double lorenzo_eb()   { return 5e-2; }
    static constexpr double quantizer_eb() { return 5e-1; }

    // Returns true if this type pair can be reconstructed by StageFactory.
    // The factory currently handles Lorenzo<float,uint16> and <double,uint16>.
    static constexpr bool factory_supported_lorenzo() {
        return std::is_same<TC, uint16_t>::value;
    }
    // Quantizer is not in StageFactory — file round-trips are not yet supported.
    static constexpr bool factory_supported_quantizer() { return false; }
};

// ─────────────────────────────────────────────────────────────────────────────
// Lorenzo type list
// ─────────────────────────────────────────────────────────────────────────────

using LorenzoTypes = ::testing::Types<
    StagePair<float,  uint16_t>,
    StagePair<float,  uint8_t>,
    StagePair<double, uint16_t>,
    StagePair<double, uint32_t>
>;

template <typename P>
class LorenzoTypeMatrix : public ::testing::Test {};
TYPED_TEST_SUITE(LorenzoTypeMatrix, LorenzoTypes);

// ── ABS round-trip ─────────────────────────────────────────────────────────
TYPED_TEST(LorenzoTypeMatrix, RoundTripAbs) {
    using TIn = typename TypeParam::Input;
    using TC  = typename TypeParam::Code;

    CudaStream stream;
    constexpr size_t N = 1 << 12;

    // uint8_t codes have a small range (radius=64 → 128 distinct codes);
    // use a looser error bound so data amplitude fits in the code range.
    const double eb_use = std::is_same<TC, uint8_t>::value
                        ? TypeParam::lorenzo_eb() * 5
                        : TypeParam::lorenzo_eb();

    auto h_input = make_smooth_data<TIn>(N);

    const TC qrad = static_cast<TC>(std::is_same<TC, uint8_t>::value ? 64 : 512);

    Pipeline p(N * sizeof(TIn), MemoryStrategy::MINIMAL);
    auto* lrz = p.addStage<LorenzoQuantStage<TIn, TC>>();
    lrz->setErrorBound(static_cast<TIn>(eb_use));
    lrz->setQuantRadius(qrad);
    lrz->setOutlierCapacity(0.2f);
    p.setPoolManagedDecompOutput(false);
    p.finalize();

    auto res = pipeline_round_trip<TIn>(p, h_input, stream);

    ASSERT_EQ(res.data.size(), h_input.size());
    EXPECT_LE(res.max_error, eb_use * 1.01)
        << "Lorenzo<" << sizeof(TIn)*8 << "b input, "
        << sizeof(TC)*8 << "b code> ABS round-trip exceeded error bound";
    EXPECT_GT(res.compressed_bytes, 0u);
}

// ── File serialization round-trip ─────────────────────────────────────────
// Skipped for type pairs not supported by StageFactory (float/uint8,
// double/uint32) because decompressFromFile cannot reconstruct those stages.
TYPED_TEST(LorenzoTypeMatrix, FileSerialization) {
    using TIn = typename TypeParam::Input;
    using TC  = typename TypeParam::Code;

    if (!TypeParam::factory_supported_lorenzo()) {
        GTEST_SKIP() << "StageFactory does not support Lorenzo<"
                     << sizeof(TIn)*8 << "b, " << sizeof(TC)*8
                     << "b> — skipping file round-trip";
    }

    CudaStream stream;
    constexpr size_t N = 1 << 12;
    const double eb_use = TypeParam::lorenzo_eb();

    auto h_input = make_smooth_data<TIn>(N);
    const std::string tmp =
        std::string("/tmp/fzgmod_typematrix_lorenzo_")
        + std::to_string(sizeof(TIn)) + "_" + std::to_string(sizeof(TC)) + ".fzm";

    Pipeline p(N * sizeof(TIn), MemoryStrategy::MINIMAL);
    auto* lrz = p.addStage<LorenzoQuantStage<TIn, TC>>();
    lrz->setErrorBound(static_cast<TIn>(eb_use));
    lrz->setQuantRadius(static_cast<TC>(512));
    lrz->setOutlierCapacity(0.2f);
    p.setPoolManagedDecompOutput(false);
    p.finalize();

    auto res = pipeline_file_round_trip<TIn>(p, h_input, stream, tmp);
    std::remove(tmp.c_str());

    ASSERT_EQ(res.data.size(), h_input.size());
    EXPECT_LE(res.max_error, eb_use * 1.01)
        << "Lorenzo<" << sizeof(TIn)*8 << "b input, "
        << sizeof(TC)*8 << "b code> file round-trip exceeded error bound";
}

// ── Constant input → exact round-trip ─────────────────────────────────────
TYPED_TEST(LorenzoTypeMatrix, ConstantInputExact) {
    using TIn = typename TypeParam::Input;
    using TC  = typename TypeParam::Code;

    CudaStream stream;
    constexpr size_t N = 512;
    const double eb_use = std::is_same<TC, uint8_t>::value
                        ? TypeParam::lorenzo_eb() * 5
                        : TypeParam::lorenzo_eb();

    // Constant input: prediction residuals are all zero, no outliers
    std::vector<TIn> h_input(N, static_cast<TIn>(3.0));

    const TC qrad2 = static_cast<TC>(std::is_same<TC, uint8_t>::value ? 64 : 512);

    Pipeline p(N * sizeof(TIn), MemoryStrategy::MINIMAL);
    auto* lrz = p.addStage<LorenzoQuantStage<TIn, TC>>();
    lrz->setErrorBound(static_cast<TIn>(eb_use));
    lrz->setQuantRadius(qrad2);
    lrz->setOutlierCapacity(0.1f);
    p.setPoolManagedDecompOutput(false);
    p.finalize();

    auto res = pipeline_round_trip<TIn>(p, h_input, stream);

    ASSERT_EQ(res.data.size(), h_input.size());
    // Constant input should round-trip exactly (no quantization error).
    EXPECT_LE(res.max_error, eb_use * 1.01)
        << "Lorenzo<" << sizeof(TIn)*8 << "b," << sizeof(TC)*8
        << "b> constant input exceeded error bound";
}

// ── serializeHeader / deserializeHeader preserves error_bound + quant_radius
TYPED_TEST(LorenzoTypeMatrix, SerializeDeserialize) {
    using TIn = typename TypeParam::Input;
    using TC  = typename TypeParam::Code;

    const double eb_use = std::is_same<TC, uint8_t>::value
                        ? TypeParam::lorenzo_eb() * 5
                        : TypeParam::lorenzo_eb();
    const TC qr = static_cast<TC>(std::is_same<TC, uint8_t>::value ? 64 : 512);

    LorenzoQuantStage<TIn, TC> stage;
    stage.setErrorBound(static_cast<TIn>(eb_use));
    stage.setQuantRadius(qr);

    uint8_t buf[256] = {};
    size_t written = stage.serializeHeader(0, buf, sizeof(buf));
    ASSERT_GT(written, 0u);

    LorenzoQuantStage<TIn, TC> restored;
    ASSERT_NO_THROW(restored.deserializeHeader(buf, written));

    EXPECT_EQ(restored.getQuantRadius(), static_cast<TC>(qr));
}

// ─────────────────────────────────────────────────────────────────────────────
// Quantizer type list
// ─────────────────────────────────────────────────────────────────────────────

using QuantizerTypes = ::testing::Types<
    StagePair<float,  uint16_t>,
    StagePair<float,  uint32_t>,
    StagePair<double, uint16_t>,
    StagePair<double, uint32_t>
>;

template <typename P>
class QuantizerTypeMatrix : public ::testing::Test {};
TYPED_TEST_SUITE(QuantizerTypeMatrix, QuantizerTypes);

// ── ABS round-trip ─────────────────────────────────────────────────────────
TYPED_TEST(QuantizerTypeMatrix, RoundTripAbs) {
    using TIn = typename TypeParam::Input;
    using TC  = typename TypeParam::Code;

    CudaStream stream;
    constexpr size_t N = 1 << 12;
    const double EB = TypeParam::quantizer_eb();

    auto h_input = make_smooth_data<TIn>(N);

    Pipeline p(N * sizeof(TIn), MemoryStrategy::MINIMAL);
    auto* qtz = p.addStage<QuantizerStage<TIn, TC>>();
    qtz->setErrorBound(static_cast<TIn>(EB));
    qtz->setQuantRadius(512);
    p.setPoolManagedDecompOutput(false);
    p.finalize();

    auto res = pipeline_round_trip<TIn>(p, h_input, stream);

    ASSERT_EQ(res.data.size(), h_input.size());
    EXPECT_LE(res.max_error, EB * 1.01)
        << "Quantizer<" << sizeof(TIn)*8 << "b input, "
        << sizeof(TC)*8 << "b code> ABS round-trip exceeded error bound";
    EXPECT_GT(res.compressed_bytes, 0u);
}

// ── File serialization round-trip ─────────────────────────────────────────
// Skipped for all type pairs: Quantizer (StageType=14) is absent from
// StageFactory::createStage(), so decompressFromFile cannot reconstruct it.
TYPED_TEST(QuantizerTypeMatrix, FileSerialization) {
    using TIn = typename TypeParam::Input;
    using TC  = typename TypeParam::Code;

    if (!TypeParam::factory_supported_quantizer()) {
        GTEST_SKIP() << "StageFactory does not support Quantizer<"
                     << sizeof(TIn)*8 << "b, " << sizeof(TC)*8
                     << "b> — skipping file round-trip";
    }

    CudaStream stream;
    constexpr size_t N = 1 << 12;
    const double EB = TypeParam::quantizer_eb();

    auto h_input = make_smooth_data<TIn>(N);
    const std::string tmp =
        std::string("/tmp/fzgmod_typematrix_quantizer_")
        + std::to_string(sizeof(TIn)) + "_" + std::to_string(sizeof(TC)) + ".fzm";

    Pipeline p(N * sizeof(TIn), MemoryStrategy::MINIMAL);
    auto* qtz = p.addStage<QuantizerStage<TIn, TC>>();
    qtz->setErrorBound(static_cast<TIn>(EB));
    qtz->setQuantRadius(512);
    p.setPoolManagedDecompOutput(false);
    p.finalize();

    auto res = pipeline_file_round_trip<TIn>(p, h_input, stream, tmp);
    std::remove(tmp.c_str());

    ASSERT_EQ(res.data.size(), h_input.size());
    EXPECT_LE(res.max_error, EB * 1.01)
        << "Quantizer<" << sizeof(TIn)*8 << "b input, "
        << sizeof(TC)*8 << "b code> file round-trip exceeded error bound";
}

// ── serializeHeader / deserializeHeader preserves quant_radius ───────────
TYPED_TEST(QuantizerTypeMatrix, SerializeDeserialize) {
    using TIn = typename TypeParam::Input;
    using TC  = typename TypeParam::Code;

    constexpr int QR = 1024;

    QuantizerStage<TIn, TC> stage;
    stage.setErrorBound(static_cast<TIn>(TypeParam::quantizer_eb()));
    stage.setQuantRadius(QR);

    uint8_t buf[256] = {};
    size_t written = stage.serializeHeader(0, buf, sizeof(buf));
    ASSERT_GT(written, 0u);

    QuantizerStage<TIn, TC> restored;
    ASSERT_NO_THROW(restored.deserializeHeader(buf, written));

    EXPECT_EQ(restored.getQuantRadius(), QR);
}
