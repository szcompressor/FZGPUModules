/**
 * tests/stages/test_pfpl_stream.cpp
 *
 * Two-array PFPL (fz::pfpl) against the staged PFPL pipeline in ABS mode.
 *
 *   PS1  PfplStreamCase/MatchesStagedPipeline  — archive bytes and decoded values equal the staged
 *                                                pipeline's, within the bound, non-finite values restored
 *   PS2  PfplStream/OverflowStaysInCapacity    — a short capacity reports the full length and writes
 *                                                nothing past itself
 *   PS3  PfplStream/ConcurrentStreamsMatchSerial — streams run together produce the serial streams
 *   PS4  PfplStream/RepeatedCallsOnOnePlan     — look-back epochs and tickets survive many calls
 *   PS5  PfplStream/StreamBytesReadsTheHeader  — streamBytes() on valid, foreign and truncated data
 *   PS6  PfplStream/FreshPlanIgnoresStaleMemory — a new Plan on dirty pages produces the same stream
 *
 * Doubles have no staged quantizer with in-place outliers, so the quantizer's rule is computed here and the rest of
 * the chain is compared with the staged stages:
 *
 *   PD1  PfplStream64Case/MatchesStagedStages  — archive bytes equal the staged difference, 8-byte bitshuffle and RZE
 *                                                of the expected codes; decoded values within the bound, non-finite
 *                                                values restored
 *   PD2  PfplStream64/OverflowStaysInCapacity  — as PS2
 *   PD3  PfplStream64/ConcurrentStreamsMatchSerial — as PS3
 *   PD4  PfplStream64/SharesAPlanWithFloats    — float and double fields alternate through one Plan
 *   PD5  PfplStream64/StreamBytesReadsTheHeader — as PS5
 *   PD6  PfplStream64/BoundIsRecordedAsAFloat  — never rounded up; too small or not finite is refused
 *
 * On HIP, CTest runs this twice: with the AMD kernels (the default) and with the generic harness
 * kernels (FZ_PFPL_HARNESS=1), which are the kernels CUDA uses.
 */

#include <gtest/gtest.h>
#include "helpers/fz_test_utils.h"
#include "coders/rze/rze_stage.h"
#include "fused/chunk_fusion/pfpl_stream.h"
#include "pipeline/compressor.h"
#include "predictors/diff/diff.h"
#include "quantizers/quantizer/quantizer.h"
#include "shufflers/bitshuffle/bitshuffle_stage.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <tuple>
#include <vector>

using fz_test::CudaBuffer;

namespace {

constexpr size_t kChunkBytes = 16384;

// The PFPL preset in ABS mode (examples/presets/pfpl_specialization.toml), staged.
std::unique_ptr<fz::Pipeline> stagedPfpl(size_t rawBytes, double eb) {
    auto p = std::make_unique<fz::Pipeline>(rawBytes, fz::MemoryStrategy::PREALLOCATE);
    p->setSpecializationPolicy(fz::SpecializationPolicy::Off);
    auto* quant = p->addStage<fz::QuantizerStage<float, uint32_t>>();
    quant->setErrorBound(eb);
    quant->setErrorBoundMode(fz::ErrorBoundMode::ABS);
    quant->setQuantRadius(fz::pfpl::kQuantRadius);
    quant->setZigzagCodes(true);
    quant->setInplaceOutliers(true);
    quant->setVerifyReconstruction(true);
    auto* diff = p->addStage<fz::DifferenceStage<int32_t, uint32_t>>();
    diff->setChunkSize(kChunkBytes);
    auto* shuffle = p->addStage<fz::BitshuffleStage>();
    shuffle->setElementWidth(4);
    shuffle->setBlockSize(kChunkBytes);
    auto* rze = p->addStage<fz::RZEStage>();
    rze->setWordSize(1);
    rze->setChunkSize(kChunkBytes);
    p->connect(diff, quant, "codes");
    p->connect(shuffle, diff);
    p->connect(rze, shuffle);
    p->finalize();
    return p;
}

std::vector<float> makeField(const std::string& kind, size_t n, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> g(0.0, 1.0);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    std::vector<float> f(n);
    if (kind == "polar") {
        // log2 magnitude and phase of a random state vector, with 20% exact-zero sentinels.
        const size_t half = n / 2;
        const double scale = 1.0 / std::sqrt(2.0 * static_cast<double>(std::max<size_t>(half, 1)));
        for (size_t i = 0; i < half; ++i) {
            if (u(rng) < 0.2) continue;
            const double re = g(rng) * scale, im = g(rng) * scale;
            f[i] = static_cast<float>(std::log2(std::hypot(re, im)) - 1.0);
            f[half + i] = static_cast<float>(std::atan2(im, re) / std::log(2.0));
        }
        if (n % 2) f[n - 1] = 0.5f;
    } else if (kind == "smooth") {
        for (size_t i = 0; i < n; ++i)
            f[i] = static_cast<float>(std::sin(i * 1e-3) + 0.5 * std::cos(i * 7e-5));
    } else if (kind == "noise") {
        for (size_t i = 0; i < n; ++i) f[i] = static_cast<float>(2.0 * u(rng) - 1.0);
    } else if (kind == "spiky") {
        // Outliers past the quantization radius, and non-finite values.
        const float inf = std::numeric_limits<float>::infinity();
        for (size_t i = 0; i < n; ++i) {
            const double r = u(rng);
            f[i] = r < 0.001    ? static_cast<float>(1e9 * g(rng))
                   : r < 0.0015 ? std::numeric_limits<float>::quiet_NaN()
                   : r < 0.002  ? inf
                   : r < 0.0025 ? -inf
                                : static_cast<float>(1e-3 * g(rng));
        }
    }  // "zeros" stays all zero
    return f;
}

bool sameBits(float a, float b) {
    uint32_t x, y;
    std::memcpy(&x, &a, 4);
    std::memcpy(&y, &b, 4);
    return x == y;
}

// A stream that does not block host threads, as a concurrent caller would use it.
struct Stream {
    cudaStream_t s = nullptr;
    Stream() { cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking); }
    ~Stream() { cudaStreamDestroy(s); }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
    operator cudaStream_t() const { return s; }
};

size_t readDeviceSize(const CudaBuffer<size_t>& d, cudaStream_t s) {
    return d.download(s)[0];
}

void expectMatchesStaged(const std::string& kind, size_t n, double eb, cudaStream_t s, fz::pfpl::Plan& plan) {
    SCOPED_TRACE(kind + " n=" + std::to_string(n) + " eb=" + std::to_string(eb));
    const size_t raw = n * sizeof(float);
    const size_t padded = (raw + kChunkBytes - 1) / kChunkBytes * kChunkBytes;
    const uint64_t seed = n * 31 + static_cast<uint64_t>(eb * 1e9);
    std::vector<float> h = makeField(kind, n, seed);
    std::vector<float> hPadded = h;
    hPadded.resize(padded / sizeof(float), 0.0f);

    CudaBuffer<float> d_in(hPadded.size()), d_out(hPadded.size());
    CudaBuffer<uint8_t> d_stream(fz::pfpl::maxStreamBytes(n));
    CudaBuffer<size_t> d_len(1);
    d_in.upload(hPadded, s);

    fz::pfpl::compress(plan, d_in.get(), n, static_cast<float>(eb), d_stream.get(), d_stream.bytes(), d_len.get(), s);
    const size_t len = readDeviceSize(d_len, s);
    ASSERT_LE(len, d_stream.bytes());
    ASSERT_GE(len, sizeof(fz::pfpl::StreamHeader));
    const std::vector<uint8_t> stream = d_stream.download_bytes(len, s);
    EXPECT_EQ(fz::pfpl::streamBytes(stream.data(), len), len);
    fz::pfpl::StreamHeader hdr;
    std::memcpy(&hdr, stream.data(), sizeof(hdr));
    EXPECT_EQ(hdr.n, n);
    EXPECT_EQ(hdr.abs_eb, static_cast<float>(eb));

    // The staged pipeline codes the zero-padded input.
    auto pipe = stagedPfpl(padded, eb);
    const size_t cap = pipe->getMaxCompressedSize(padded);
    CudaBuffer<uint8_t> d_ref(cap);
    CudaBuffer<float> d_refOut(hPadded.size());
    size_t refLen = 0;
    pipe->compress(d_in.get(), raw, d_ref.get(), cap, &refLen, s);
    cudaStreamSynchronize(s);
    const std::vector<uint8_t> ref = d_ref.download_bytes(refLen, s);

    const size_t arc = len - sizeof(fz::pfpl::StreamHeader);
    ASSERT_EQ(arc, refLen) << "archive length differs from the staged pipeline's";
    EXPECT_EQ(std::memcmp(ref.data(), stream.data() + sizeof(fz::pfpl::StreamHeader), arc), 0)
        << "archive bytes differ from the staged pipeline's";

    size_t refDecoded = 0;
    pipe->decompressInto(d_ref.get(), refLen, d_refOut.get(), padded, &refDecoded, s);
    fz::pfpl::decompress(plan, d_stream.get(), d_out.get(), n, s);
    cudaStreamSynchronize(s);
    const std::vector<float> a = d_out.download_bytes(raw, s);
    const std::vector<float> b = d_refOut.download_bytes(raw, s);

    size_t differ = 0, overBound = 0, nonFinite = 0;
    double maxErr = 0.0;
    for (size_t i = 0; i < n; ++i) {
        if (!sameBits(a[i], b[i])) ++differ;
        if (!std::isfinite(h[i])) {
            if (!sameBits(a[i], h[i])) ++nonFinite;
            continue;
        }
        const double e = std::fabs(static_cast<double>(a[i]) - static_cast<double>(h[i]));
        maxErr = std::max(maxErr, e);
        if (e > eb) ++overBound;
    }
    EXPECT_EQ(differ, 0u) << "values differ from the staged decode";
    EXPECT_EQ(overBound, 0u) << "values past the bound; largest error " << maxErr;
    EXPECT_EQ(nonFinite, 0u) << "non-finite values not restored";
}

using CaseParam = std::tuple<const char*, size_t, double>;
class PfplStreamCase : public ::testing::TestWithParam<CaseParam> {};

}  // namespace

// PS1
TEST_P(PfplStreamCase, MatchesStagedPipeline) {
    Stream s;
    fz::pfpl::Plan plan;
    expectMatchesStaged(std::get<0>(GetParam()), std::get<1>(GetParam()), std::get<2>(GetParam()), s, plan);
}

INSTANTIATE_TEST_SUITE_P(
    Fields, PfplStreamCase,
    ::testing::Combine(::testing::Values("polar", "smooth", "noise", "zeros", "spiky"),
                       // under a chunk, one chunk, not a whole number of chunks, and many chunks
                       ::testing::Values(size_t{512}, size_t{4096}, size_t{3 * 4096 + 2}, size_t{1} << 18,
                                         size_t{1} << 23),
                       ::testing::Values(1e-2, 1e-4, 1e-6)),
    [](const ::testing::TestParamInfo<CaseParam>& info) {
        return std::string(std::get<0>(info.param)) + "_n" + std::to_string(std::get<1>(info.param)) + "_eb" +
               std::to_string(static_cast<int>(std::lround(-std::log10(std::get<2>(info.param)))));
    });

// PS2
TEST(PfplStream, OverflowStaysInCapacity) {
    Stream s;
    fz::pfpl::Plan plan;
    const size_t n = 3 * 4096 + 77;
    const std::vector<float> h = makeField("noise", n, 5);
    CudaBuffer<float> d_in(n);
    CudaBuffer<size_t> d_len(1);
    d_in.upload(h, s);

    size_t full = 0;
    {
        CudaBuffer<uint8_t> d_stream(fz::pfpl::maxStreamBytes(n));
        fz::pfpl::compress(plan, d_in.get(), n, 1e-7f, d_stream.get(), d_stream.bytes(), d_len.get(), s);
        full = readDeviceSize(d_len, s);
    }
    ASSERT_GT(full, size_t{100});

    for (const size_t cap : {full - 1, full / 2, size_t{100}, size_t{24}, size_t{8}}) {
        SCOPED_TRACE("capacity " + std::to_string(cap));
        const size_t guard = 4096;
        CudaBuffer<uint8_t> d_stream(cap + guard);
        ASSERT_EQ(cudaMemset(d_stream.get(), 0xA5, cap + guard), cudaSuccess);
        fz::pfpl::compress(plan, d_in.get(), n, 1e-7f, d_stream.get(), cap, d_len.get(), s);
        EXPECT_EQ(readDeviceSize(d_len, s), full);
        const std::vector<uint8_t> all = d_stream.download(s);
        const size_t touched = static_cast<size_t>(
            std::count_if(all.begin() + static_cast<std::ptrdiff_t>(cap), all.end(), [](uint8_t b) { return b != 0xA5; }));
        EXPECT_EQ(touched, 0u) << "bytes written past the capacity";
    }
}

// PS3
TEST(PfplStream, ConcurrentStreamsMatchSerial) {
    constexpr int kStreams = 4, kRounds = 20;
    constexpr size_t n = size_t{1} << 22;
    constexpr float eb = 1e-4f;

    std::vector<std::vector<float>> fields;
    std::vector<std::unique_ptr<Stream>> streams;
    std::vector<std::unique_ptr<fz::pfpl::Plan>> plans;
    std::vector<std::unique_ptr<CudaBuffer<float>>> in, out;
    std::vector<std::unique_ptr<CudaBuffer<uint8_t>>> st;
    std::vector<std::unique_ptr<CudaBuffer<size_t>>> len;
    std::vector<std::vector<uint8_t>> serial(kStreams);
    for (int i = 0; i < kStreams; ++i) {
        fields.push_back(makeField(i % 2 ? "polar" : "smooth", n, 100 + i));
        streams.emplace_back(new Stream);
        plans.emplace_back(new fz::pfpl::Plan);
        in.emplace_back(new CudaBuffer<float>(n));
        out.emplace_back(new CudaBuffer<float>(n));
        st.emplace_back(new CudaBuffer<uint8_t>(fz::pfpl::maxStreamBytes(n)));
        len.emplace_back(new CudaBuffer<size_t>(1));
        in[i]->upload(fields[i], *streams[i]);
    }
    for (int i = 0; i < kStreams; ++i) {
        fz::pfpl::compress(*plans[i], in[i]->get(), n, eb, st[i]->get(), st[i]->bytes(), len[i]->get(), *streams[i]);
        serial[i] = st[i]->download_bytes(readDeviceSize(*len[i], *streams[i]), *streams[i]);
    }

    for (int round = 0; round < kRounds; ++round) {
        SCOPED_TRACE("round " + std::to_string(round));
        for (int i = 0; i < kStreams; ++i) {
            ASSERT_EQ(cudaMemsetAsync(st[i]->get(), 0, st[i]->bytes(), *streams[i]), cudaSuccess);
        }
        for (int i = 0; i < kStreams; ++i) {
            fz::pfpl::compress(*plans[i], in[i]->get(), n, eb, st[i]->get(), st[i]->bytes(), len[i]->get(), *streams[i]);
        }
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        for (int i = 0; i < kStreams; ++i) {
            EXPECT_EQ(st[i]->download_bytes(serial[i].size(), *streams[i]), serial[i]) << "stream " << i;
        }
        for (int i = 0; i < kStreams; ++i) {
            fz::pfpl::decompress(*plans[i], st[i]->get(), out[i]->get(), n, *streams[i]);
        }
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        for (int i = 0; i < kStreams; ++i) {
            const std::vector<float> d = out[i]->download(*streams[i]);
            size_t overBound = 0;
            for (size_t j = 0; j < n; ++j) {
                overBound += std::fabs(static_cast<double>(d[j]) - static_cast<double>(fields[i][j])) > eb;
            }
            EXPECT_EQ(overBound, 0u) << "stream " << i;
        }
    }
}

// PS4
TEST(PfplStream, RepeatedCallsOnOnePlan) {
    Stream s;
    fz::pfpl::Plan plan;
    for (int r = 0; r < 300 && !::testing::Test::HasFatalFailure(); ++r) {
        expectMatchesStaged(r % 3 ? "polar" : "smooth", (size_t{1} << 20) + (r % 5) * 1000, 1e-3, s, plan);
    }
}

// PS5
TEST(PfplStream, StreamBytesReadsTheHeader) {
    const fz::pfpl::StreamHeader good{fz::pfpl::kStreamMagic, 1000, 1e-3f, 4096};
    const size_t length = sizeof(good) + 1000;
    EXPECT_EQ(fz::pfpl::streamBytes(&good), length);
    EXPECT_EQ(fz::pfpl::streamBytes(&good, length), length);
    EXPECT_EQ(fz::pfpl::streamBytes(&good, length - 1), 0u) << "claims more than the capacity";

    fz::pfpl::StreamHeader foreign = good;
    foreign.magic ^= 1u;
    EXPECT_EQ(fz::pfpl::streamBytes(&foreign), 0u);
}

// PS6: a Plan's counter and look-back state land on recycled pages, so their contents must not reach its first
// kernel. The pages are dirty, and the default stream is busy: a clear queued there would run after the fill, while
// the first kernel on this non-blocking stream goes ahead of it. A stale counter is a wild chunk id, and stale
// look-back words are a wrong offset.
TEST(PfplStream, FreshPlanIgnoresStaleMemory) {
    Stream s;
    const size_t n = 8 * 4096;
    const std::vector<float> h = makeField("smooth", n, 1);
    CudaBuffer<float> d_in(n);
    CudaBuffer<uint8_t> d_stream(fz::pfpl::maxStreamBytes(n));
    CudaBuffer<uint8_t> d_busy(size_t{1} << 30);
    CudaBuffer<size_t> d_len(1);
    d_in.upload(h, s);

    std::vector<uint8_t> expected;
    {
        fz::pfpl::Plan plan;
        fz::pfpl::compress(plan, d_in.get(), n, 1e-4f, d_stream.get(), d_stream.bytes(), d_len.get(), s);
        expected = d_stream.download_bytes(readDeviceSize(d_len, s), s);
    }

    for (int round = 0; round < 40 && !::testing::Test::HasFailure(); ++round) {
        SCOPED_TRACE("round " + std::to_string(round));
        {
            std::vector<std::unique_ptr<CudaBuffer<uint8_t>>> dirty;
            for (int i = 0; i < 64; ++i) {
                dirty.emplace_back(new CudaBuffer<uint8_t>(4096));
                ASSERT_EQ(cudaMemsetAsync(dirty.back()->get(), 0xFF, 4096, s), cudaSuccess);
            }
            ASSERT_EQ(cudaStreamSynchronize(s), cudaSuccess);
        }
        ASSERT_EQ(cudaMemsetAsync(d_busy.get(), 0, d_busy.bytes(), nullptr), cudaSuccess);
        fz::pfpl::Plan plan;
        fz::pfpl::compress(plan, d_in.get(), n, 1e-4f, d_stream.get(), d_stream.bytes(), d_len.get(), s);
        ASSERT_EQ(readDeviceSize(d_len, s), expected.size());
        EXPECT_EQ(d_stream.download_bytes(expected.size(), s), expected);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    }
}

// ─── Doubles ────────────────────────────────────────────────────────────────────────────────────────────────────────

namespace {

constexpr size_t kElems64 = kChunkBytes / sizeof(double);

std::vector<double> makeField64(const std::string& kind, size_t n, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> g(0.0, 1.0);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    std::vector<double> f(n);
    if (kind == "polar") {
        const size_t half = n / 2;
        const double scale = 1.0 / std::sqrt(2.0 * static_cast<double>(std::max<size_t>(half, 1)));
        for (size_t i = 0; i < half; ++i) {
            if (u(rng) < 0.2) continue;
            const double re = g(rng) * scale, im = g(rng) * scale;
            f[i] = std::log2(std::hypot(re, im)) - 1.0;
            f[half + i] = std::atan2(im, re) / std::log(2.0);
        }
        if (n % 2) f[n - 1] = 0.5;
    } else if (kind == "smooth") {
        for (size_t i = 0; i < n; ++i) f[i] = std::sin(i * 1e-3) + 0.5 * std::cos(i * 7e-5);
    } else if (kind == "noise") {
        for (size_t i = 0; i < n; ++i) f[i] = 2.0 * u(rng) - 1.0;
    } else if (kind == "spiky") {
        // Past the radius at small bounds, non-finite, subnormal, and values no index reconstructs within the bound.
        const double inf = std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < n; ++i) {
            const double r = u(rng);
            f[i] = r < 0.001    ? 1e300 * g(rng)
                   : r < 0.002  ? 1e9 * g(rng)
                   : r < 0.0025 ? std::numeric_limits<double>::quiet_NaN()
                   : r < 0.003  ? inf
                   : r < 0.0035 ? -inf
                   : r < 0.004  ? std::numeric_limits<double>::denorm_min() * (1.0 + 1000.0 * u(rng))
                   : r < 0.0045 ? -std::numeric_limits<double>::min() * u(rng)
                                : 1e-3 * g(rng);
        }
    }  // "zeros" stays all zero
    return f;
}

uint64_t bitsOf(double x) {
    uint64_t u;
    std::memcpy(&u, &x, 8);
    return u;
}

double doubleOf(uint64_t u) {
    double x;
    std::memcpy(&x, &u, 8);
    return x;
}

// The code fz::pfpl gives x at this bound. volatile keeps the host compiler from fusing the product into the
// subtraction: the decoder stores the rounded product, so that is what the bound applies to.
uint64_t expectedCode(double x, double eb) {
    const double r = std::rint(x * (0.5 / eb));
    if (!(std::fabs(r) < static_cast<double>(fz::pfpl::kQuantRadius64))) return bitsOf(x);
    const volatile double product = r * (2.0 * eb);
    if (!(std::fabs(x - product) <= eb)) return bitsOf(x);
    const int64_t q = static_cast<int64_t>(r);
    return (static_cast<uint64_t>(q) << 1) ^ static_cast<uint64_t>(q >> 63);
}

double expectedValue(uint64_t code, double eb) {
    if ((code >> 1) >= fz::pfpl::kQuantRadius64) return doubleOf(code);
    const int64_t q = static_cast<int64_t>((code >> 1) ^ (0ull - (code & 1ull)));
    return static_cast<double>(q) * (2.0 * eb);
}

// Everything after the quantizer, staged: difference, negabinary, 8-byte bitshuffle, RZE.
std::unique_ptr<fz::Pipeline> stagedTail64(size_t rawBytes) {
    auto p = std::make_unique<fz::Pipeline>(rawBytes, fz::MemoryStrategy::PREALLOCATE);
    p->setSpecializationPolicy(fz::SpecializationPolicy::Off);
    auto* diff = p->addStage<fz::DifferenceStage<int64_t, uint64_t>>();
    diff->setChunkSize(kChunkBytes);
    auto* shuffle = p->addStage<fz::BitshuffleStage>();
    shuffle->setElementWidth(8);
    shuffle->setBlockSize(kChunkBytes);
    auto* rze = p->addStage<fz::RZEStage>();
    rze->setWordSize(1);
    rze->setChunkSize(kChunkBytes);
    p->connect(shuffle, diff);
    p->connect(rze, shuffle);
    p->finalize();
    return p;
}

void expectMatches64(const std::string& kind, size_t n, double eb, cudaStream_t s, fz::pfpl::Plan& plan) {
    SCOPED_TRACE(kind + " n=" + std::to_string(n) + " eb=" + std::to_string(eb));
    const size_t raw = n * sizeof(double);
    const size_t padded = (raw + kChunkBytes - 1) / kChunkBytes * kChunkBytes;
    const std::vector<double> h = makeField64(kind, n, n * 31 + static_cast<uint64_t>(-std::log10(eb)));

    CudaBuffer<double> d_in(n), d_out(n);
    CudaBuffer<uint8_t> d_stream(fz::pfpl::maxStreamBytes<double>(n));
    CudaBuffer<size_t> d_len(1);
    d_in.upload(h, s);

    fz::pfpl::compress(plan, d_in.get(), n, eb, d_stream.get(), d_stream.bytes(), d_len.get(), s);
    const size_t len = readDeviceSize(d_len, s);
    ASSERT_LE(len, d_stream.bytes());
    ASSERT_GE(len, sizeof(fz::pfpl::StreamHeader));
    const std::vector<uint8_t> stream = d_stream.download_bytes(len, s);
    EXPECT_EQ(fz::pfpl::streamBytes(stream.data(), len), len);
    fz::pfpl::StreamHeader hdr;
    std::memcpy(&hdr, stream.data(), sizeof(hdr));
    EXPECT_EQ(hdr.magic, fz::pfpl::kStreamMagic64);
    EXPECT_EQ(hdr.n, n);
    EXPECT_LE(static_cast<double>(hdr.abs_eb), eb);
    const double used = static_cast<double>(hdr.abs_eb);

    // The expected codes, zero-padded to whole chunks as the field is coded.
    std::vector<uint64_t> codes(padded / sizeof(uint64_t), 0);
    for (size_t i = 0; i < n; ++i) codes[i] = expectedCode(h[i], used);
    CudaBuffer<uint64_t> d_codes(codes.size());
    d_codes.upload(codes, s);
    cudaStreamSynchronize(s);

    auto pipe = stagedTail64(padded);
    const size_t cap = pipe->getMaxCompressedSize(padded);
    CudaBuffer<uint8_t> d_ref(cap);
    size_t refLen = 0;
    pipe->compress(d_codes.get(), padded, d_ref.get(), cap, &refLen, s);
    cudaStreamSynchronize(s);
    const std::vector<uint8_t> ref = d_ref.download_bytes(refLen, s);

    const size_t arc = len - sizeof(fz::pfpl::StreamHeader);
    ASSERT_EQ(arc, refLen) << "archive length differs from the staged stages'";
    EXPECT_EQ(std::memcmp(ref.data(), stream.data() + sizeof(fz::pfpl::StreamHeader), arc), 0)
        << "archive bytes differ from the staged stages'";

    fz::pfpl::decompress(plan, d_stream.get(), d_out.get(), n, s);
    cudaStreamSynchronize(s);
    const std::vector<double> a = d_out.download_bytes(raw, s);

    size_t differ = 0, overBound = 0, nonFinite = 0;
    double maxErr = 0.0;
    for (size_t i = 0; i < n; ++i) {
        if (bitsOf(a[i]) != bitsOf(expectedValue(codes[i], used))) ++differ;
        if (!std::isfinite(h[i])) {
            if (bitsOf(a[i]) != bitsOf(h[i])) ++nonFinite;
            continue;
        }
        const double e = std::fabs(a[i] - h[i]);
        maxErr = std::max(maxErr, e);
        if (e > used) ++overBound;
    }
    EXPECT_EQ(differ, 0u) << "values differ from the codes' reconstruction";
    EXPECT_EQ(overBound, 0u) << "values past the bound; largest error " << maxErr;
    EXPECT_EQ(nonFinite, 0u) << "non-finite values not restored";
}

class PfplStream64Case : public ::testing::TestWithParam<CaseParam> {};

}  // namespace

// PD1
TEST_P(PfplStream64Case, MatchesStagedStages) {
    Stream s;
    fz::pfpl::Plan plan;
    expectMatches64(std::get<0>(GetParam()), std::get<1>(GetParam()), std::get<2>(GetParam()), s, plan);
}

INSTANTIATE_TEST_SUITE_P(
    Fields, PfplStream64Case,
    ::testing::Combine(::testing::Values("polar", "smooth", "noise", "zeros", "spiky"),
                       ::testing::Values(size_t{256}, kElems64, 3 * kElems64 + 2, size_t{1} << 17, size_t{1} << 22),
                       ::testing::Values(1e-3, 1e-8, 1e-13)),
    [](const ::testing::TestParamInfo<CaseParam>& info) {
        return std::string(std::get<0>(info.param)) + "_n" + std::to_string(std::get<1>(info.param)) + "_eb" +
               std::to_string(static_cast<int>(std::lround(-std::log10(std::get<2>(info.param)))));
    });

// PD2
TEST(PfplStream64, OverflowStaysInCapacity) {
    Stream s;
    fz::pfpl::Plan plan;
    const size_t n = 3 * kElems64 + 77;
    const std::vector<double> h = makeField64("noise", n, 5);
    CudaBuffer<double> d_in(n);
    CudaBuffer<size_t> d_len(1);
    d_in.upload(h, s);

    size_t full = 0;
    {
        CudaBuffer<uint8_t> d_stream(fz::pfpl::maxStreamBytes<double>(n));
        fz::pfpl::compress(plan, d_in.get(), n, 1e-12, d_stream.get(), d_stream.bytes(), d_len.get(), s);
        full = readDeviceSize(d_len, s);
    }
    ASSERT_GT(full, size_t{100});

    for (const size_t cap : {full - 1, full / 2, size_t{100}, size_t{24}, size_t{8}}) {
        SCOPED_TRACE("capacity " + std::to_string(cap));
        const size_t guard = 4096;
        CudaBuffer<uint8_t> d_stream(cap + guard);
        ASSERT_EQ(cudaMemset(d_stream.get(), 0xA5, cap + guard), cudaSuccess);
        fz::pfpl::compress(plan, d_in.get(), n, 1e-12, d_stream.get(), cap, d_len.get(), s);
        EXPECT_EQ(readDeviceSize(d_len, s), full);
        const std::vector<uint8_t> all = d_stream.download(s);
        const size_t touched = static_cast<size_t>(
            std::count_if(all.begin() + static_cast<std::ptrdiff_t>(cap), all.end(), [](uint8_t b) { return b != 0xA5; }));
        EXPECT_EQ(touched, 0u) << "bytes written past the capacity";
    }
}

// PD3
TEST(PfplStream64, ConcurrentStreamsMatchSerial) {
    constexpr int kStreams = 4, kRounds = 20;
    constexpr size_t n = size_t{1} << 21;
    constexpr double eb = 1e-7;

    std::vector<std::vector<double>> fields;
    std::vector<std::unique_ptr<Stream>> streams;
    std::vector<std::unique_ptr<fz::pfpl::Plan>> plans;
    std::vector<std::unique_ptr<CudaBuffer<double>>> in, out;
    std::vector<std::unique_ptr<CudaBuffer<uint8_t>>> st;
    std::vector<std::unique_ptr<CudaBuffer<size_t>>> len;
    std::vector<std::vector<uint8_t>> serial(kStreams);
    for (int i = 0; i < kStreams; ++i) {
        fields.push_back(makeField64(i % 2 ? "polar" : "smooth", n, 100 + i));
        streams.emplace_back(new Stream);
        plans.emplace_back(new fz::pfpl::Plan);
        in.emplace_back(new CudaBuffer<double>(n));
        out.emplace_back(new CudaBuffer<double>(n));
        st.emplace_back(new CudaBuffer<uint8_t>(fz::pfpl::maxStreamBytes<double>(n)));
        len.emplace_back(new CudaBuffer<size_t>(1));
        in[i]->upload(fields[i], *streams[i]);
    }
    for (int i = 0; i < kStreams; ++i) {
        fz::pfpl::compress(*plans[i], in[i]->get(), n, eb, st[i]->get(), st[i]->bytes(), len[i]->get(), *streams[i]);
        serial[i] = st[i]->download_bytes(readDeviceSize(*len[i], *streams[i]), *streams[i]);
    }

    for (int round = 0; round < kRounds; ++round) {
        SCOPED_TRACE("round " + std::to_string(round));
        for (int i = 0; i < kStreams; ++i) {
            ASSERT_EQ(cudaMemsetAsync(st[i]->get(), 0, st[i]->bytes(), *streams[i]), cudaSuccess);
        }
        for (int i = 0; i < kStreams; ++i) {
            fz::pfpl::compress(*plans[i], in[i]->get(), n, eb, st[i]->get(), st[i]->bytes(), len[i]->get(), *streams[i]);
        }
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        for (int i = 0; i < kStreams; ++i) {
            EXPECT_EQ(st[i]->download_bytes(serial[i].size(), *streams[i]), serial[i]) << "stream " << i;
        }
        for (int i = 0; i < kStreams; ++i) {
            fz::pfpl::decompress(*plans[i], st[i]->get(), out[i]->get(), n, *streams[i]);
        }
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        for (int i = 0; i < kStreams; ++i) {
            const std::vector<double> d = out[i]->download(*streams[i]);
            size_t overBound = 0;
            for (size_t j = 0; j < n; ++j) overBound += std::fabs(d[j] - fields[i][j]) > eb;
            EXPECT_EQ(overBound, 0u) << "stream " << i;
        }
    }
}

// PD4: a stream workspace's plan is sized in bytes, and one run could hold either type.
TEST(PfplStream64, SharesAPlanWithFloats) {
    Stream s;
    fz::pfpl::Plan plan;
    plan.reserveBytes(size_t{1} << 23);
    for (int r = 0; r < 30 && !::testing::Test::HasFatalFailure(); ++r) {
        if (r % 2) expectMatchesStaged("polar", (size_t{1} << 20) + (r % 5) * 1000, 1e-3, s, plan);
        else expectMatches64("polar", (size_t{1} << 19) + (r % 5) * 500, 1e-8, s, plan);
    }
}

// PD5
TEST(PfplStream64, StreamBytesReadsTheHeader) {
    const fz::pfpl::StreamHeader good{fz::pfpl::kStreamMagic64, 1000, 1e-3f, 2048};
    const size_t length = sizeof(good) + 1000;
    EXPECT_EQ(fz::pfpl::streamBytes(&good), length);
    EXPECT_EQ(fz::pfpl::streamBytes(&good, length), length);
    EXPECT_EQ(fz::pfpl::streamBytes(&good, length - 1), 0u) << "claims more than the capacity";

    // 8 chunks of doubles need a size table of 32 bytes, but as floats the same n is only 4 chunks.
    const fz::pfpl::StreamHeader doubles{fz::pfpl::kStreamMagic64, 8 + 4 * 4, 1e-3f, 8 * 2048};
    const fz::pfpl::StreamHeader floats{fz::pfpl::kStreamMagic, 8 + 4 * 4, 1e-3f, 8 * 2048};
    EXPECT_EQ(fz::pfpl::streamBytes(&doubles), 0u) << "shorter than its size table";
    EXPECT_NE(fz::pfpl::streamBytes(&floats), 0u);

    EXPECT_EQ(fz::pfpl::maxStreamBytes<double>(2048), fz::pfpl::maxStreamBytes<float>(4096));
}

// PD6
TEST(PfplStream64, BoundIsRecordedAsAFloat) {
    Stream s;
    fz::pfpl::Plan plan;
    const size_t n = 2048;
    const std::vector<double> h = makeField64("smooth", n, 3);
    CudaBuffer<double> d_in(n);
    CudaBuffer<uint8_t> d_stream(fz::pfpl::maxStreamBytes<double>(n));
    CudaBuffer<size_t> d_len(1);
    d_in.upload(h, s);

    for (const double eb : {1e-5, 0.1, 1.0 / 3.0, 3e-30}) {
        SCOPED_TRACE(eb);
        fz::pfpl::compress(plan, d_in.get(), n, eb, d_stream.get(), d_stream.bytes(), d_len.get(), s);
        const size_t len = readDeviceSize(d_len, s);
        fz::pfpl::StreamHeader hdr;
        const std::vector<uint8_t> head = d_stream.download_bytes(sizeof(hdr), s);
        std::memcpy(&hdr, head.data(), sizeof(hdr));
        EXPECT_GT(len, sizeof(hdr));
        EXPECT_LE(static_cast<double>(hdr.abs_eb), eb);
        EXPECT_GE(static_cast<double>(hdr.abs_eb), eb * (1.0 - 1.2e-7));
    }
    const auto attempt = [&](double eb) {
        fz::pfpl::compress(plan, d_in.get(), n, eb, d_stream.get(), d_stream.bytes(), d_len.get(), s);
    };
    EXPECT_THROW(attempt(1e-40), std::invalid_argument);
    EXPECT_THROW(attempt(0.0), std::invalid_argument);
    EXPECT_THROW(attempt(-1.0), std::invalid_argument);
    EXPECT_THROW(attempt(std::numeric_limits<double>::infinity()), std::invalid_argument);
    EXPECT_THROW(attempt(std::numeric_limits<double>::quiet_NaN()), std::invalid_argument);
}
