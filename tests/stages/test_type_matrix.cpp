/**
 * tests/stages/test_type_matrix.cpp
 *
 * Parametric type-matrix tests for LorenzoQuantStage and QuantizerStage across
 * all supported <TInput, TCode> combinations.  Each TYPED_TEST runs once per
 * type pair, so adding a new instantiation inherits full coverage automatically.
 *
 * LorenzoTypeMatrix — <float,uint16_t>, <float,uint8_t>, <double,uint16_t>, <double,uint32_t>
 *   TM1  LorenzoTypeMatrix/RoundTripAbs          — ABS mode compress+decompress in memory
 *   TM2  LorenzoTypeMatrix/FileSerialization      — ABS mode writeToFile/decompressFromFile cycle
 *   TM3  LorenzoTypeMatrix/ConstantInputExact     — constant array → zero residuals, within error bound
 *   TM4  LorenzoTypeMatrix/SerializeDeserialize   — serializeHeader/deserializeHeader config round-trip
 *
 * QuantizerTypeMatrix — <float,uint16_t>, <float,uint32_t>, <double,uint16_t>, <double,uint32_t>
 *   TM5  QuantizerTypeMatrix/RoundTripAbs         — ABS mode compress+decompress in memory
 *   TM6  QuantizerTypeMatrix/FileSerialization    — file round-trip (skipped: not in StageFactory)
 *   TM7  QuantizerTypeMatrix/SerializeDeserialize — serializeHeader/deserializeHeader config round-trip
 */

#include <gtest/gtest.h>
#include "helpers/stage_harness.h"
#include "fused/lorenzo_quant/lorenzo_quant.h"
#include "quantizers/quantizer/quantizer.h"
#include "fzgpumodules.h"

#include <algorithm>
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

// ─────────────────────────────────────────────────────────────────────────────
// TM1: LorenzoTypeMatrix/RoundTripAbs — ABS mode compress+decompress in memory
// ─────────────────────────────────────────────────────────────────────────────
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

// ─────────────────────────────────────────────────────────────────────────────
// TM2: LorenzoTypeMatrix/FileSerialization — ABS mode writeToFile/decompressFromFile cycle
// Skipped for type pairs not in StageFactory (float/uint8, double/uint32).
// ─────────────────────────────────────────────────────────────────────────────
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

// ─────────────────────────────────────────────────────────────────────────────
// TM3: LorenzoTypeMatrix/ConstantInputExact — constant array → zero residuals, within error bound
// ─────────────────────────────────────────────────────────────────────────────
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

// ─────────────────────────────────────────────────────────────────────────────
// TM4: LorenzoTypeMatrix/SerializeDeserialize — serializeHeader/deserializeHeader config round-trip
// ─────────────────────────────────────────────────────────────────────────────
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
// TM4b: LorenzoQuant honours an f64 bound the header used to narrow.
//
// The header stored `error_bound` as a float, so a `double` stage round-tripped
// its own bound through float32 — a ~6e-08 relative error. That sounds harmless
// and is not: the inverse reconstructs by prefix-summing quantized residuals and
// scaling by 2*abs_eb, so the error is amplified by the magnitude of the sum. For
// S3D/N2 (values ~0.7369 at abs_eb 1.103e-09) it lands as ~4.4e-08 absolute —
// 40x the bound — and the cell round-tripped at 53 dB reporting `status: ok`.
//
// Driven through a full compress/decompress because the bound is only resolved
// during execute(); a header-only assertion would compare two zeroes.
// ─────────────────────────────────────────────────────────────────────────────
TEST(LorenzoQuantHeader, DoubleNoaTinyRangeLargeOffsetRespectsBound) {
    CudaStream stream;
    constexpr size_t N       = 1 << 14;
    constexpr double OFFSET  = 0.7369;
    constexpr double RANGE   = 1.10e-5;
    constexpr double USER_EB = 1.0e-4;   // NOA: abs_eb = USER_EB * range

    std::vector<double> h_input(N);
    for (size_t i = 0; i < N; i++)
        h_input[i] = OFFSET + RANGE * (0.5 + 0.5 * std::sin(i * 0.01));

    const double vmin = *std::min_element(h_input.begin(), h_input.end());
    const double vmax = *std::max_element(h_input.begin(), h_input.end());
    const double abs_eb = USER_EB * (vmax - vmin);

    // NOA, not ABS, and that distinction is the whole point. In ABS mode the
    // bound comes straight from `config_.error_bound`, which is already a float,
    // so the header narrowing is a no-op and an ABS test passes vacuously. Only a
    // relative mode computes abs_eb as a genuine double (user_eb * value_base),
    // which is the precision the header then threw away.
    ASSERT_NE(static_cast<double>(static_cast<float>(abs_eb)), abs_eb)
        << "pick a bound float32 cannot hold, or this test proves nothing";

    Pipeline p(N * sizeof(double), MemoryStrategy::MINIMAL, 8.0f);
    auto* lrz = p.addStage<LorenzoQuantStage<double, uint16_t>>();
    lrz->setErrorBound(static_cast<double>(USER_EB));
    lrz->setErrorBoundMode(ErrorBoundMode::NOA);
    lrz->setQuantRadius(static_cast<uint16_t>(32768));
    lrz->setOutlierCapacity(0.5f);
    p.setPoolManagedDecompOutput(false);
    p.finalize();

    // A FILE round-trip, deliberately. An in-process compress/decompress reuses
    // the live stage object, whose computed_abs_eb_ is already exact — it never
    // reads the serialized header, so it cannot see this bug at all.
    const std::string tmp = "/tmp/fzgmod_lq_f64_bound.fzm";
    auto res = pipeline_file_round_trip<double>(p, h_input, stream, tmp);
    std::remove(tmp.c_str());
    ASSERT_EQ(res.data.size(), h_input.size());

    EXPECT_LE(res.max_error, abs_eb * 1.01)
        << "f64 NOA LorenzoQuant file round-trip max_err=" << res.max_error
        << " exceeds abs_eb=" << abs_eb
        << " (ratio " << (res.max_error / abs_eb) << "x)";
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

// ─────────────────────────────────────────────────────────────────────────────
// TM5: QuantizerTypeMatrix/RoundTripAbs — ABS mode compress+decompress in memory
// ─────────────────────────────────────────────────────────────────────────────
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

// ─────────────────────────────────────────────────────────────────────────────
// TM6: QuantizerTypeMatrix/FileSerialization — file round-trip (skipped: not in StageFactory)
// ─────────────────────────────────────────────────────────────────────────────
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

// ─────────────────────────────────────────────────────────────────────────────
// TM7: QuantizerTypeMatrix/SerializeDeserialize — serializeHeader/deserializeHeader config round-trip
// ─────────────────────────────────────────────────────────────────────────────
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
