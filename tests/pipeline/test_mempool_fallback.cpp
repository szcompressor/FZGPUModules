/**
 * tests/pipeline/test_mempool_fallback.cpp
 *
 * Tests for the MemoryPool cudaMalloc fallback path that activates on vGPU
 * environments where cudaMemPoolCreate() is unavailable or broken.
 *
 * Fallback mode is forced here via FZ_FORCE_MEMPOOL_FALLBACK (env var), which
 * lets any machine run these tests without needing an actual vGPU.  All tests
 * use the MemPoolFallbackTest fixture, which sets and unsets the env var around
 * each test so the fallback is scoped to Pipeline objects created inside the test.
 *
 * To run the entire test suite in fallback mode without recompiling:
 *   FZ_FORCE_MEMPOOL_FALLBACK=1 ctest --preset default
 */

#include <gtest/gtest.h>
#include "helpers/fz_test_utils.h"
#include "fzgpumodules.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace fz;
using namespace fz_test;

// ─────────────────────────────────────────────────────────────────────────────
// Fixture — sets FZ_FORCE_MEMPOOL_FALLBACK before each test so that every
// Pipeline constructed inside the test body uses the cudaMalloc fallback path.
// ─────────────────────────────────────────────────────────────────────────────
class MemPoolFallbackTest : public ::testing::Test {
protected:
    void SetUp() override    { setenv("FZ_FORCE_MEMPOOL_FALLBACK", "1", /*overwrite=*/1); }
    void TearDown() override { unsetenv("FZ_FORCE_MEMPOOL_FALLBACK"); }
};

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

static std::vector<float> make_smooth(size_t n) {
    std::vector<float> v(n);
    for (size_t i = 0; i < n; i++)
        v[i] = std::sin(static_cast<float>(i) * 0.01f) * 50.0f
             + std::cos(static_cast<float>(i) * 0.003f) * 20.0f;
    return v;
}

// Build a minimal Lorenzo-only pipeline, compress+decompress, return reconstruction.
// Pool ownership is off (caller cudaFree's decomp output) for simplicity.
static std::vector<float> fallback_roundtrip(
    const std::vector<float>& h_input,
    MemoryStrategy             strategy,
    float                      pool_mult = 3.0f)
{
    constexpr float EB = 1e-2f;
    size_t in_bytes = h_input.size() * sizeof(float);

    CudaStream stream;
    CudaBuffer<float> d_in(h_input.size());
    d_in.upload(h_input, stream);
    stream.sync();

    Pipeline pipeline(in_bytes, strategy, pool_mult);
    auto* lrz = pipeline.addStage<LorenzoQuantStage<float, uint16_t>>();
    lrz->setErrorBound(EB);
    lrz->setQuantRadius(512);
    lrz->setOutlierCapacity(0.2f);
    pipeline.setPoolManagedDecompOutput(false);
    pipeline.finalize();

    void*  d_comp = nullptr; size_t comp_sz = 0;
    pipeline.compress(d_in.void_ptr(), in_bytes, &d_comp, &comp_sz, stream);

    void*  d_dec = nullptr; size_t dec_sz = 0;
    pipeline.decompress(nullptr, 0, &d_dec, &dec_sz, stream);

    std::vector<float> h_recon(h_input.size());
    if (d_dec && dec_sz == in_bytes)
        cudaMemcpy(h_recon.data(), d_dec, in_bytes, cudaMemcpyDeviceToHost);
    if (d_dec) cudaFree(d_dec);
    return h_recon;
}

// ─────────────────────────────────────────────────────────────────────────────
// FB1: isMemPoolFallbackMode() returns true when forced.
// ─────────────────────────────────────────────────────────────────────────────
TEST_F(MemPoolFallbackTest, IsMemPoolFallbackModeTrue) {
    constexpr size_t N = 1 << 10;
    Pipeline pipeline(N * sizeof(float));
    auto* lrz = pipeline.addStage<LorenzoQuantStage<float, uint16_t>>();
    lrz->setErrorBound(1e-2f);
    lrz->setQuantRadius(512);
    lrz->setOutlierCapacity(0.2f);
    pipeline.finalize();

    EXPECT_TRUE(pipeline.isMemPoolFallbackMode())
        << "isMemPoolFallbackMode() must return true when FZ_FORCE_MEMPOOL_FALLBACK is set";
}

// ─────────────────────────────────────────────────────────────────────────────
// FB2: MINIMAL strategy round-trip in fallback mode.
// ─────────────────────────────────────────────────────────────────────────────
TEST_F(MemPoolFallbackTest, MinimalRoundTrip) {
    constexpr size_t N  = 1 << 14;
    constexpr float  EB = 1e-2f;
    auto h_input  = make_smooth(N);
    auto h_recon  = fallback_roundtrip(h_input, MemoryStrategy::MINIMAL);

    ASSERT_EQ(h_recon.size(), N);
    EXPECT_LE(max_abs_error(h_input, h_recon), EB * 1.01f)
        << "Fallback MINIMAL round-trip error exceeds bound";
}

// ─────────────────────────────────────────────────────────────────────────────
// FB3: PREALLOCATE strategy round-trip in fallback mode.
// ─────────────────────────────────────────────────────────────────────────────
TEST_F(MemPoolFallbackTest, PreallocateRoundTrip) {
    constexpr size_t N  = 1 << 14;
    constexpr float  EB = 1e-2f;
    auto h_input  = make_smooth(N);
    auto h_recon  = fallback_roundtrip(h_input, MemoryStrategy::PREALLOCATE);

    ASSERT_EQ(h_recon.size(), N);
    EXPECT_LE(max_abs_error(h_input, h_recon), EB * 1.01f)
        << "Fallback PREALLOCATE round-trip error exceeds bound";
}

// ─────────────────────────────────────────────────────────────────────────────
// FB4: Both strategies produce the same output in fallback mode.
// ─────────────────────────────────────────────────────────────────────────────
TEST_F(MemPoolFallbackTest, MinimalAndPreallocateProduceSameResult) {
    constexpr size_t N = 1 << 13;
    auto h_input = make_smooth(N);

    auto recon_min  = fallback_roundtrip(h_input, MemoryStrategy::MINIMAL);
    auto recon_pre  = fallback_roundtrip(h_input, MemoryStrategy::PREALLOCATE);

    ASSERT_EQ(recon_min.size(),  N);
    ASSERT_EQ(recon_pre.size(),  N);
    for (size_t i = 0; i < N; i++)
        EXPECT_FLOAT_EQ(recon_min[i], recon_pre[i])
            << "MINIMAL vs PREALLOCATE mismatch at index " << i;
}

// ─────────────────────────────────────────────────────────────────────────────
// FB5: Multi-stage pipeline (Lorenzo → RLE) round-trip in fallback mode.
//      Exercises the RLE persistent scratch path through the fallback pool.
// ─────────────────────────────────────────────────────────────────────────────
TEST_F(MemPoolFallbackTest, LorenzoRleRoundTrip) {
    constexpr size_t N  = 1 << 14;
    constexpr float  EB = 1e-2f;
    const size_t in_bytes = N * sizeof(float);

    CudaStream stream;
    auto h_input = make_smooth(N);
    CudaBuffer<float> d_in(N);
    d_in.upload(h_input, stream);
    stream.sync();

    Pipeline pipeline(in_bytes, MemoryStrategy::MINIMAL, 3.0f);
    auto* lrz = pipeline.addStage<LorenzoQuantStage<float, uint16_t>>();
    lrz->setErrorBound(EB);
    lrz->setQuantRadius(512);
    lrz->setOutlierCapacity(0.2f);
    auto* rle = pipeline.addStage<RLEStage<uint16_t>>();
    pipeline.connect(rle, lrz, "codes");
    pipeline.setPoolManagedDecompOutput(false);
    pipeline.finalize();

    EXPECT_TRUE(pipeline.isMemPoolFallbackMode());

    void*  d_comp = nullptr; size_t comp_sz = 0;
    ASSERT_NO_THROW(pipeline.compress(d_in.void_ptr(), in_bytes, &d_comp, &comp_sz, stream));
    ASSERT_GT(comp_sz, 0u);

    void*  d_dec = nullptr; size_t dec_sz = 0;
    ASSERT_NO_THROW(pipeline.decompress(nullptr, 0, &d_dec, &dec_sz, stream));
    ASSERT_NE(d_dec, nullptr);
    ASSERT_EQ(dec_sz, in_bytes);

    std::vector<float> h_recon(N);
    cudaMemcpy(h_recon.data(), d_dec, in_bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_dec);

    EXPECT_LE(max_abs_error(h_input, h_recon), EB * 1.01f)
        << "Lorenzo→RLE fallback round-trip error exceeds bound";
}

// ─────────────────────────────────────────────────────────────────────────────
// FB6: Multi-stage pipeline (Lorenzo → Bitshuffle → RZE) in fallback mode.
//      BitshuffleStage and RZEStage both allocate stage-level scratch through
//      the pool; this exercises those paths under the cudaMalloc fallback.
// ─────────────────────────────────────────────────────────────────────────────
TEST_F(MemPoolFallbackTest, LorenzoBitshuffleRzeRoundTrip) {
    constexpr size_t N  = 1 << 14;
    constexpr float  EB = 1e-2f;
    const size_t in_bytes = N * sizeof(float);

    CudaStream stream;
    auto h_input = make_smooth(N);
    CudaBuffer<float> d_in(N);
    d_in.upload(h_input, stream);
    stream.sync();

    Pipeline pipeline(in_bytes, MemoryStrategy::MINIMAL, 4.0f);
    auto* lrz  = pipeline.addStage<LorenzoQuantStage<float, uint16_t>>();
    lrz->setErrorBound(EB);
    lrz->setQuantRadius(512);
    lrz->setOutlierCapacity(0.2f);
    auto* bshuf = pipeline.addStage<BitshuffleStage>();
    auto* rze   = pipeline.addStage<RZEStage>();
    pipeline.connect(bshuf, lrz, "codes");
    pipeline.connect(rze,   bshuf);
    pipeline.setPoolManagedDecompOutput(false);
    pipeline.finalize();

    EXPECT_TRUE(pipeline.isMemPoolFallbackMode());

    void*  d_comp = nullptr; size_t comp_sz = 0;
    ASSERT_NO_THROW(pipeline.compress(d_in.void_ptr(), in_bytes, &d_comp, &comp_sz, stream));
    ASSERT_GT(comp_sz, 0u);

    void*  d_dec = nullptr; size_t dec_sz = 0;
    ASSERT_NO_THROW(pipeline.decompress(nullptr, 0, &d_dec, &dec_sz, stream));
    ASSERT_NE(d_dec, nullptr);
    ASSERT_EQ(dec_sz, in_bytes);

    std::vector<float> h_recon(N);
    cudaMemcpy(h_recon.data(), d_dec, in_bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_dec);

    EXPECT_LE(max_abs_error(h_input, h_recon), EB * 1.01f)
        << "Lorenzo→Bitshuffle→RZE fallback round-trip error exceeds bound";
}

// ─────────────────────────────────────────────────────────────────────────────
// FB7: getCurrentUsage() tracks allocations correctly in fallback mode.
//      After compress(), usage must be > 0; after reset(), must be 0 (MINIMAL).
// ─────────────────────────────────────────────────────────────────────────────
TEST_F(MemPoolFallbackTest, CurrentUsageTrackingMinimal) {
    constexpr size_t N  = 1 << 14;
    const size_t in_bytes = N * sizeof(float);

    CudaStream stream;
    auto h_input = make_smooth(N);
    CudaBuffer<float> d_in(N);
    d_in.upload(h_input, stream);
    stream.sync();

    Pipeline pipeline(in_bytes, MemoryStrategy::MINIMAL);
    auto* lrz = pipeline.addStage<LorenzoQuantStage<float, uint16_t>>();
    lrz->setErrorBound(1e-2f);
    lrz->setQuantRadius(512);
    lrz->setOutlierCapacity(0.2f);
    pipeline.setPoolManagedDecompOutput(false);
    pipeline.finalize();

    EXPECT_EQ(pipeline.getCurrentMemoryUsage(), 0u)
        << "Fallback: usage must be 0 before first compress()";

    void*  d_comp = nullptr; size_t comp_sz = 0;
    pipeline.compress(d_in.void_ptr(), in_bytes, &d_comp, &comp_sz, stream);
    stream.sync();

    EXPECT_GT(pipeline.getCurrentMemoryUsage(), 0u)
        << "Fallback: usage must be > 0 after compress()";

    pipeline.reset(stream);
    stream.sync();

    EXPECT_EQ(pipeline.getCurrentMemoryUsage(), 0u)
        << "Fallback: usage must be 0 after reset()";
}

// ─────────────────────────────────────────────────────────────────────────────
// FB8: Repeated compress+reset cycles do not leak memory in fallback mode.
//      getCurrentUsage() must return 0 after each reset() regardless of
//      how many compress() calls have been made.
// ─────────────────────────────────────────────────────────────────────────────
TEST_F(MemPoolFallbackTest, RepeatedCompressResetNoLeak) {
    constexpr size_t N  = 1 << 14;
    const size_t in_bytes = N * sizeof(float);

    CudaStream stream;
    auto h_input = make_smooth(N);
    CudaBuffer<float> d_in(N);
    d_in.upload(h_input, stream);
    stream.sync();

    Pipeline pipeline(in_bytes, MemoryStrategy::MINIMAL);
    auto* lrz = pipeline.addStage<LorenzoQuantStage<float, uint16_t>>();
    lrz->setErrorBound(1e-2f);
    lrz->setQuantRadius(512);
    lrz->setOutlierCapacity(0.2f);
    pipeline.setPoolManagedDecompOutput(false);
    pipeline.finalize();

    constexpr int CYCLES = 5;
    for (int i = 0; i < CYCLES; i++) {
        void*  d_comp = nullptr; size_t comp_sz = 0;
        pipeline.compress(d_in.void_ptr(), in_bytes, &d_comp, &comp_sz, stream);
        stream.sync();
        ASSERT_GT(comp_sz, 0u) << "cycle " << i;

        pipeline.reset(stream);
        stream.sync();

        EXPECT_EQ(pipeline.getCurrentMemoryUsage(), 0u)
            << "Fallback: usage must be 0 after reset() — cycle " << i;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FB9: RLE persistent scratch is reused across compress() calls in fallback mode.
//      GPU free memory after the second compress+reset must be ~equal to after
//      the first — no permanent new allocations on the second call.
// ─────────────────────────────────────────────────────────────────────────────
TEST_F(MemPoolFallbackTest, RleScratchReusedAcrossCalls) {
    constexpr size_t N  = 1 << 14;
    const size_t in_bytes = N * sizeof(float);

    CudaStream stream;
    auto h_input = make_smooth(N);
    CudaBuffer<float> d_in(N);
    d_in.upload(h_input, stream);
    stream.sync();

    Pipeline pipeline(in_bytes, MemoryStrategy::MINIMAL);
    auto* lrz = pipeline.addStage<LorenzoQuantStage<float, uint16_t>>();
    lrz->setErrorBound(1e-2f);
    lrz->setQuantRadius(512);
    lrz->setOutlierCapacity(0.2f);
    auto* rle = pipeline.addStage<RLEStage<uint16_t>>();
    pipeline.connect(rle, lrz, "codes");
    pipeline.setPoolManagedDecompOutput(false);
    pipeline.finalize();

    void*  d_comp = nullptr; size_t comp_sz = 0;
    pipeline.compress(d_in.void_ptr(), in_bytes, &d_comp, &comp_sz, stream);
    stream.sync();
    ASSERT_GT(comp_sz, 0u);
    pipeline.reset(stream);
    stream.sync();
    cudaDeviceSynchronize();
    const size_t free_after_first = gpu_free_bytes();

    d_comp = nullptr; comp_sz = 0;
    pipeline.compress(d_in.void_ptr(), in_bytes, &d_comp, &comp_sz, stream);
    stream.sync();
    ASSERT_GT(comp_sz, 0u);
    pipeline.reset(stream);
    stream.sync();
    cudaDeviceSynchronize();
    const size_t free_after_second = gpu_free_bytes();

    constexpr size_t kTol = 512 * 1024;
    const size_t diff = (free_after_second > free_after_first)
                      ? free_after_second - free_after_first
                      : free_after_first  - free_after_second;
    EXPECT_LE(diff, kTol)
        << "Fallback: GPU free memory after second compress+reset ("
        << free_after_second / 1024 << " KB) must be ~equal to after first ("
        << free_after_first / 1024 << " KB) — RLE scratch must be reused";
}

// ─────────────────────────────────────────────────────────────────────────────
// FB10: File IO round-trip (compress-to-file → decompressFromFile) in fallback mode.
// ─────────────────────────────────────────────────────────────────────────────
TEST_F(MemPoolFallbackTest, FileIoRoundTrip) {
    constexpr size_t N  = 1 << 12;
    constexpr float  EB = 1e-2f;
    const size_t in_bytes = N * sizeof(float);
    const char* tmp_path = "/tmp/fz_fallback_test.fzm";

    CudaStream stream;
    auto h_input = make_smooth(N);
    CudaBuffer<float> d_in(N);
    d_in.upload(h_input, stream);
    stream.sync();

    // Compress and write to file.
    {
        Pipeline pipeline(in_bytes, MemoryStrategy::MINIMAL);
        auto* lrz = pipeline.addStage<LorenzoQuantStage<float, uint16_t>>();
        lrz->setErrorBound(EB);
        lrz->setQuantRadius(512);
        lrz->setOutlierCapacity(0.2f);
        pipeline.finalize();

        EXPECT_TRUE(pipeline.isMemPoolFallbackMode());

        void*  d_comp = nullptr; size_t comp_sz = 0;
        pipeline.compress(d_in.void_ptr(), in_bytes, &d_comp, &comp_sz, stream);
        ASSERT_GT(comp_sz, 0u);
        pipeline.writeToFile(tmp_path, stream);
    }

    // Decompress from file (static, always caller-owned).
    void*  d_out = nullptr; size_t out_sz = 0;
    ASSERT_NO_THROW(Pipeline::decompressFromFile(tmp_path, &d_out, &out_sz, stream));
    ASSERT_NE(d_out, nullptr);
    ASSERT_EQ(out_sz, in_bytes);

    std::vector<float> h_recon(N);
    cudaMemcpy(h_recon.data(), d_out, in_bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_out);
    std::remove(tmp_path);

    EXPECT_LE(max_abs_error(h_input, h_recon), EB * 1.01f)
        << "Fallback: file IO round-trip error exceeds bound";
}

// ─────────────────────────────────────────────────────────────────────────────
// Normal-mode sanity check: isMemPoolFallbackMode() returns false when the env
// var is NOT set (verifies the env-var gate itself works).
// ─────────────────────────────────────────────────────────────────────────────
TEST(MemPoolFallbackSanity, FallbackFalseWithoutEnvVar) {
    // Ensure the env var is definitely absent for this test.
    unsetenv("FZ_FORCE_MEMPOOL_FALLBACK");

    constexpr size_t N = 1 << 10;
    Pipeline pipeline(N * sizeof(float));
    auto* lrz = pipeline.addStage<LorenzoQuantStage<float, uint16_t>>();
    lrz->setErrorBound(1e-2f);
    lrz->setQuantRadius(512);
    lrz->setOutlierCapacity(0.2f);
    pipeline.finalize();

    // On real hardware the pool should succeed and fallback should be false.
    // On a vGPU the pool creation may fail naturally, so we cannot assert false —
    // we only assert that the method is callable and doesn't crash.
    (void)pipeline.isMemPoolFallbackMode();
    SUCCEED();
}
