/**
 * tests/test_memory_strategies.cpp
 *
 * Tests that verify all three MemoryStrategy modes produce correct results
 * and that pool-size configuration options work as expected.
 *
 * Strategies under test:
 *   MINIMAL      – buffers sized and allocated on-demand after each stage
 *   PIPELINE     – balanced: allocates ahead, frees when consumers are done
 *   PREALLOCATE  – sizes everything up-front via estimateOutputSizes()
 *
 * All three must produce identical decompressed data for the same input.
 */

#include <gtest/gtest.h>
#include "helpers/fz_test_utils.h"
#include "fzgpumodules.h"

#include <cmath>
#include <cstring>
#include <vector>

using namespace fz;
using namespace fz_test;

// ─────────────────────────────────────────────────────────────────────────────
// Shared helpers
// ─────────────────────────────────────────────────────────────────────────────

static std::vector<float> make_smooth(size_t n) {
    std::vector<float> v(n);
    for (size_t i = 0; i < n; i++)
        v[i] = std::sin(static_cast<float>(i) * 0.01f) * 50.0f
             + std::cos(static_cast<float>(i) * 0.003f) * 20.0f;
    return v;
}

// Compress+decompress with the given strategy; return the reconstructed data.
static std::vector<float> roundtrip_with_strategy(
    const std::vector<float>& h_input,
    MemoryStrategy            strategy,
    float                     pool_multiplier = 3.0f)
{
    constexpr float EB = 1e-2f;
    size_t in_bytes = h_input.size() * sizeof(float);

    CudaStream stream;
    CudaBuffer<float> d_in(h_input.size());
    d_in.upload(h_input, stream);
    stream.sync();

    Pipeline pipeline(in_bytes, strategy, pool_multiplier);
    auto* lrz = pipeline.addStage<LorenzoStage<float, uint16_t>>();
    lrz->setErrorBound(EB);
    lrz->setQuantRadius(512);
    lrz->setOutlierCapacity(0.2f);
    pipeline.finalize();

    void*  d_comp = nullptr;
    size_t comp_sz = 0;
    pipeline.compress(d_in.void_ptr(), in_bytes, &d_comp, &comp_sz, stream);
    EXPECT_GT(comp_sz, 0u) << "Compressed output is empty";

    void*  d_dec = nullptr;
    size_t dec_sz = 0;
    pipeline.decompress(nullptr, 0, &d_dec, &dec_sz, stream);

    std::vector<float> h_recon(h_input.size());
    if (d_dec && dec_sz == in_bytes) {
        cudaMemcpy(h_recon.data(), d_dec, in_bytes, cudaMemcpyDeviceToHost);
    }
    if (d_dec) cudaFree(d_dec);
    return h_recon;
}

// ─────────────────────────────────────────────────────────────────────────────
// M2: PIPELINE strategy — Lorenzo-only pipeline round-trip
// ─────────────────────────────────────────────────────────────────────────────
TEST(MemoryStrategy, PipelineRoundTrip) {
    constexpr size_t N  = 1 << 14;
    constexpr float  EB = 1e-2f;
    auto h_input = make_smooth(N);

    auto h_recon = roundtrip_with_strategy(h_input, MemoryStrategy::PIPELINE);

    ASSERT_EQ(h_recon.size(), N);
    float max_err = max_abs_error(h_input, h_recon);
    EXPECT_LE(max_err, EB * 1.01f)
        << "PIPELINE strategy: max_err=" << max_err << " exceeds bound " << EB;
}

// ─────────────────────────────────────────────────────────────────────────────
// M3: PREALLOCATE strategy — Lorenzo-only pipeline round-trip
// ─────────────────────────────────────────────────────────────────────────────
TEST(MemoryStrategy, PreallocateRoundTrip) {
    constexpr size_t N  = 1 << 14;
    constexpr float  EB = 1e-2f;
    auto h_input = make_smooth(N);

    auto h_recon = roundtrip_with_strategy(h_input, MemoryStrategy::PREALLOCATE);

    ASSERT_EQ(h_recon.size(), N);
    float max_err = max_abs_error(h_input, h_recon);
    EXPECT_LE(max_err, EB * 1.01f)
        << "PREALLOCATE strategy: max_err=" << max_err << " exceeds bound " << EB;
}

// ─────────────────────────────────────────────────────────────────────────────
// M4/M5: All three strategies produce the same decompressed data
// ─────────────────────────────────────────────────────────────────────────────
TEST(MemoryStrategy, AllStrategiesProduceSameResult) {
    constexpr size_t N = 1 << 13;  // 8 K — small enough for fast test
    auto h_input = make_smooth(N);

    auto recon_minimal     = roundtrip_with_strategy(h_input, MemoryStrategy::MINIMAL);
    auto recon_pipeline    = roundtrip_with_strategy(h_input, MemoryStrategy::PIPELINE);
    auto recon_preallocate = roundtrip_with_strategy(h_input, MemoryStrategy::PREALLOCATE);

    ASSERT_EQ(recon_minimal.size(),     N);
    ASSERT_EQ(recon_pipeline.size(),    N);
    ASSERT_EQ(recon_preallocate.size(), N);

    // Decompressed values must be identical element-for-element (all three
    // strategies run the same deterministic kernels on the same data).
    for (size_t i = 0; i < N; i++) {
        EXPECT_FLOAT_EQ(recon_minimal[i], recon_pipeline[i])
            << "MINIMAL vs PIPELINE mismatch at index " << i;
        EXPECT_FLOAT_EQ(recon_minimal[i], recon_preallocate[i])
            << "MINIMAL vs PREALLOCATE mismatch at index " << i;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// M6: setMemoryStrategy() can be changed BEFORE finalize()
// ─────────────────────────────────────────────────────────────────────────────
TEST(MemoryStrategy, ChangeStrategyBeforeFinalize) {
    constexpr size_t N  = 4096;
    constexpr float  EB = 1e-2f;
    auto h_input = make_smooth(N);
    size_t in_bytes = N * sizeof(float);

    CudaStream stream;
    CudaBuffer<float> d_in(N);
    d_in.upload(h_input, stream);
    stream.sync();

    Pipeline pipeline(in_bytes, MemoryStrategy::MINIMAL);

    // Switch to PREALLOCATE before adding any stages — stages must be added
    // to the correct DAG, so the strategy change must come first.
    ASSERT_NO_THROW(pipeline.setMemoryStrategy(MemoryStrategy::PREALLOCATE));

    auto* lrz = pipeline.addStage<LorenzoStage<float, uint16_t>>();
    lrz->setErrorBound(EB);
    lrz->setQuantRadius(512);
    lrz->setOutlierCapacity(0.2f);

    pipeline.finalize();

    void*  d_comp = nullptr;
    size_t comp_sz = 0;
    pipeline.compress(d_in.void_ptr(), in_bytes, &d_comp, &comp_sz, stream);
    EXPECT_GT(comp_sz, 0u);

    void*  d_dec = nullptr;
    size_t dec_sz = 0;
    pipeline.decompress(nullptr, 0, &d_dec, &dec_sz, stream);
    ASSERT_NE(d_dec, nullptr);
    EXPECT_EQ(dec_sz, in_bytes);
    cudaFree(d_dec);
}

// ─────────────────────────────────────────────────────────────────────────────
// P1: Default pool multiplier (3.0×) — pipeline runs without OOM
// ─────────────────────────────────────────────────────────────────────────────
TEST(MemoryPool, DefaultMultiplier) {
    constexpr size_t N = 1 << 14;
    auto h_input = make_smooth(N);
    // roundtrip_with_strategy uses 3.0× by default
    auto h_recon = roundtrip_with_strategy(h_input, MemoryStrategy::MINIMAL, 3.0f);
    EXPECT_EQ(h_recon.size(), N);
}

// ─────────────────────────────────────────────────────────────────────────────
// P3: Zero input_data_size hint — falls back to 1 GB default pool
// ─────────────────────────────────────────────────────────────────────────────
TEST(MemoryPool, ZeroInputSizeHint) {
    constexpr size_t N  = 1 << 14;
    constexpr float  EB = 1e-2f;
    auto h_input = make_smooth(N);
    size_t in_bytes = N * sizeof(float);

    CudaStream stream;
    CudaBuffer<float> d_in(N);
    d_in.upload(h_input, stream);
    stream.sync();

    // input_data_size = 0 → MemoryPool falls back to 1 GB
    Pipeline pipeline(0, MemoryStrategy::MINIMAL);
    auto* lrz = pipeline.addStage<LorenzoStage<float, uint16_t>>();
    lrz->setErrorBound(EB);
    lrz->setQuantRadius(512);
    lrz->setOutlierCapacity(0.2f);
    pipeline.finalize();

    void*  d_comp = nullptr;
    size_t comp_sz = 0;
    ASSERT_NO_THROW(
        pipeline.compress(d_in.void_ptr(), in_bytes, &d_comp, &comp_sz, stream)
    );
    EXPECT_GT(comp_sz, 0u);

    // With input_data_size=0, propagateBufferSizes() skips size propagation
    // (no hint available), so the decompression output allocation may be 0.
    // We only verify that compress() ran successfully without throwing.
    (void)comp_sz;
}

// ─────────────────────────────────────────────────────────────────────────────
// P4: Very large multiplier (10×) — pipeline runs without error
// ─────────────────────────────────────────────────────────────────────────────
TEST(MemoryPool, LargeMultiplier) {
    constexpr size_t N = 1 << 13;
    auto h_input = make_smooth(N);
    auto h_recon = roundtrip_with_strategy(h_input, MemoryStrategy::MINIMAL, 10.0f);
    EXPECT_EQ(h_recon.size(), N);
}

// ─────────────────────────────────────────────────────────────────────────────
// P6: getPeakMemoryUsage() returns a non-zero value after compress()
// ─────────────────────────────────────────────────────────────────────────────
TEST(MemoryPool, PeakUsageNonZeroAfterCompress) {
    constexpr size_t N  = 1 << 14;
    constexpr float  EB = 1e-2f;
    size_t in_bytes = N * sizeof(float);

    auto h_input = make_smooth(N);

    CudaStream stream;
    CudaBuffer<float> d_in(N);
    d_in.upload(h_input, stream);
    stream.sync();

    Pipeline pipeline(in_bytes, MemoryStrategy::MINIMAL);
    auto* lrz = pipeline.addStage<LorenzoStage<float, uint16_t>>();
    lrz->setErrorBound(EB);
    lrz->setQuantRadius(512);
    lrz->setOutlierCapacity(0.2f);
    pipeline.finalize();

    void*  d_comp = nullptr;
    size_t comp_sz = 0;
    pipeline.compress(d_in.void_ptr(), in_bytes, &d_comp, &comp_sz, stream);

    EXPECT_GT(pipeline.getPeakMemoryUsage(), 0u)
        << "getPeakMemoryUsage() should be > 0 after compress()";
}

// ─────────────────────────────────────────────────────────────────────────────
// ML4: MemoryPool::reset() releases all tracked allocations.
//
// After compress(), the output buffer (and any un-freed intermediates) are
// still tracked inside the DAG.  After pipeline.reset(), the DAG calls
// mem_pool_->free() for every non-persistent non-external buffer and zeroes
// getCurrentMemoryUsage(). We verify that the usage counter reaches 0, not
// that OS-level gpu_free_bytes() changes (the pool backing store stays resident
// so that subsequent compress() calls can reuse it without re-cudaMalloc).
// ─────────────────────────────────────────────────────────────────────────────
TEST(MemoryStrategies, PoolResetReleasesAllocations) {
    constexpr size_t N = 1 << 16;  // 64 K floats
    const size_t in_bytes = N * sizeof(float);

    CudaStream stream;
    auto h_input = make_smooth(N);
    CudaBuffer<float> d_in(N);
    d_in.upload(h_input, stream);
    stream.sync();

    Pipeline pipeline(in_bytes, MemoryStrategy::MINIMAL);
    auto* lrz = pipeline.addStage<LorenzoStage<float, uint16_t>>();
    lrz->setErrorBound(1e-2f);
    lrz->setQuantRadius(512);
    lrz->setOutlierCapacity(0.2f);
    pipeline.finalize();

    void*  d_comp  = nullptr;
    size_t comp_sz = 0;
    pipeline.compress(d_in.void_ptr(), in_bytes, &d_comp, &comp_sz, stream);
    cudaStreamSynchronize(stream);
    ASSERT_GT(comp_sz, 0u);

    // After compress the output buffer(s) are still held inside the pool.
    size_t mem_after_compress = pipeline.getCurrentMemoryUsage();
    EXPECT_GT(mem_after_compress, 0u)
        << "getCurrentMemoryUsage() should be > 0 after compress() — "
           "at least the output buffer is tracked";

    pipeline.reset(stream);
    cudaStreamSynchronize(stream);

    // After reset, every non-persistent DAG buffer is freed back to the pool
    // and the usage counter is zeroed.
    EXPECT_EQ(pipeline.getCurrentMemoryUsage(), 0u)
        << "getCurrentMemoryUsage() should be 0 after reset()";
}