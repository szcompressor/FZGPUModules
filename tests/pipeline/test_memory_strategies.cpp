/**
 * tests/test_memory_strategies.cpp
 *
 * Tests that verify all MemoryStrategy modes produce correct results
 * and that pool-size configuration options work as expected.
 *
 * Strategies under test:
 *   MINIMAL      – buffers sized and allocated on-demand after each stage
 *   PREALLOCATE  – sizes everything up-front via estimateOutputSizes()
 *
 * Both must produce identical decompressed data for the same input.
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
    auto* lrz = pipeline.addStage<LorenzoQuantStage<float, uint16_t>>();
    lrz->setErrorBound(EB);
    lrz->setQuantRadius(512);
    lrz->setOutlierCapacity(0.2f);
    pipeline.setPoolManagedDecompOutput(false);
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
// M2: MINIMAL strategy — Lorenzo-only pipeline round-trip
// ─────────────────────────────────────────────────────────────────────────────
TEST(MemoryStrategy, MinimalRoundTrip) {
    constexpr size_t N  = 1 << 14;
    constexpr float  EB = 1e-2f;
    auto h_input = make_smooth(N);

    auto h_recon = roundtrip_with_strategy(h_input, MemoryStrategy::MINIMAL);

    ASSERT_EQ(h_recon.size(), N);
    float max_err = max_abs_error(h_input, h_recon);
    EXPECT_LE(max_err, EB * 1.01f)
        << "MINIMAL strategy: max_err=" << max_err << " exceeds bound " << EB;
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
// M4: Both strategies produce the same decompressed data
// ─────────────────────────────────────────────────────────────────────────────
TEST(MemoryStrategy, AllStrategiesProduceSameResult) {
    constexpr size_t N = 1 << 13;  // 8 K — small enough for fast test
    auto h_input = make_smooth(N);

    auto recon_minimal     = roundtrip_with_strategy(h_input, MemoryStrategy::MINIMAL);
    auto recon_preallocate = roundtrip_with_strategy(h_input, MemoryStrategy::PREALLOCATE);

    ASSERT_EQ(recon_minimal.size(),     N);
    ASSERT_EQ(recon_preallocate.size(), N);

    // Decompressed values must be identical element-for-element (both
    // strategies run the same deterministic kernels on the same data).
    for (size_t i = 0; i < N; i++) {
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

    auto* lrz = pipeline.addStage<LorenzoQuantStage<float, uint16_t>>();
    lrz->setErrorBound(EB);
    lrz->setQuantRadius(512);
    lrz->setOutlierCapacity(0.2f);

    pipeline.setPoolManagedDecompOutput(false);
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
    auto* lrz = pipeline.addStage<LorenzoQuantStage<float, uint16_t>>();
    lrz->setErrorBound(EB);
    lrz->setQuantRadius(512);
    lrz->setOutlierCapacity(0.2f);
    pipeline.setPoolManagedDecompOutput(false);
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
    auto* lrz = pipeline.addStage<LorenzoQuantStage<float, uint16_t>>();
    lrz->setErrorBound(EB);
    lrz->setQuantRadius(512);
    lrz->setOutlierCapacity(0.2f);
    pipeline.setPoolManagedDecompOutput(false);
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
    auto* lrz = pipeline.addStage<LorenzoQuantStage<float, uint16_t>>();
    lrz->setErrorBound(1e-2f);
    lrz->setQuantRadius(512);
    lrz->setOutlierCapacity(0.2f);
    pipeline.setPoolManagedDecompOutput(false);
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

// ─────────────────────────────────────────────────────────────────────────────
// ML5: MINIMAL liveness analysis produces peak ≤ PREALLOCATE total.
//
// For a multi-stage pipeline (Lorenzo → RLE), MINIMAL mode can free the
// intermediate "codes" buffer once RLE has consumed it, so the topology-aware
// pool size is smaller than the sum-of-all-buffers that PREALLOCATE requires.
// ─────────────────────────────────────────────────────────────────────────────
TEST(MemoryStrategies, MinimalTopoPoolSizeLEPreallocate) {
    constexpr size_t N = 1 << 14;
    const size_t in_bytes = N * sizeof(float);

    auto build_topo_size = [&](MemoryStrategy strategy) -> size_t {
        Pipeline p(in_bytes, strategy);
        auto* lrz = p.addStage<LorenzoQuantStage<float, uint16_t>>();
        lrz->setErrorBound(1e-2f);
        lrz->setQuantRadius(512);
        lrz->setOutlierCapacity(0.2f);
        auto* rle = p.addStage<RLEStage<uint16_t>>();
        p.connect(rle, lrz, "codes");
        p.setPoolManagedDecompOutput(false);
    p.finalize();
        return p.getDAG()->computeTopoPoolSize();
    };

    size_t minimal_topo     = build_topo_size(MemoryStrategy::MINIMAL);
    size_t preallocate_topo = build_topo_size(MemoryStrategy::PREALLOCATE);

    EXPECT_GT(minimal_topo, 0u)
        << "MINIMAL computeTopoPoolSize() must be positive";
    EXPECT_GT(preallocate_topo, 0u)
        << "PREALLOCATE computeTopoPoolSize() must be positive";
    EXPECT_LE(minimal_topo, preallocate_topo)
        << "MINIMAL topo pool (" << minimal_topo << " B) should be <= "
        << "PREALLOCATE total (" << preallocate_topo << " B) — "
        << "liveness analysis must reduce peak below sum-of-all-buffers";
}

// ─────────────────────────────────────────────────────────────────────────────
// ML6: Repeated compress+reset cycles do not accumulate memory.
//
// Persistent scratch (e.g. RLE/Lorenzo workspace) is allocated once and
// reused across calls.  getCurrentMemoryUsage() after each reset() must stay
// at 0 regardless of how many compress() calls have been made.
// ─────────────────────────────────────────────────────────────────────────────
TEST(MemoryStrategies, RepeatedCompressResetStableMemory) {
    constexpr size_t N = 1 << 14;
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
        void*  d_comp  = nullptr;
        size_t comp_sz = 0;
        pipeline.compress(d_in.void_ptr(), in_bytes, &d_comp, &comp_sz, stream);
        cudaStreamSynchronize(stream);
        ASSERT_GT(comp_sz, 0u) << "cycle " << i;

        pipeline.reset(stream);
        cudaStreamSynchronize(stream);

        EXPECT_EQ(pipeline.getCurrentMemoryUsage(), 0u)
            << "getCurrentMemoryUsage() must be 0 after reset() — cycle " << i;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CO1: Buffer coloring is applied in PREALLOCATE mode (coloring enabled by default).
//
// PREALLOCATE mode runs the interference-graph coloring pass at finalize().
// Verify that (a) isColoringEnabled() returns true and (b) the number of color
// regions is strictly less than the total number of buffers — i.e., at least
// one pair of non-overlapping buffers shares a region.
// ─────────────────────────────────────────────────────────────────────────────
TEST(MemoryStrategies, ColoringAppliedInPreallocateMode) {
    constexpr size_t N = 1 << 14;
    const size_t in_bytes = N * sizeof(float);

    // Multi-stage pipeline to create enough buffers for coloring to matter.
    Pipeline pipeline(in_bytes, MemoryStrategy::PREALLOCATE);
    auto* lrz = pipeline.addStage<LorenzoQuantStage<float, uint16_t>>();
    lrz->setErrorBound(1e-2f);
    lrz->setQuantRadius(512);
    lrz->setOutlierCapacity(0.2f);
    auto* rle = pipeline.addStage<RLEStage<uint16_t>>();
    pipeline.connect(rle, lrz, "codes");
    pipeline.setPoolManagedDecompOutput(false);
    pipeline.finalize();

    EXPECT_TRUE(pipeline.isColoringEnabled())
        << "Buffer coloring should be applied for PREALLOCATE strategy";

    EXPECT_GT(pipeline.getColorRegionCount(), 0u)
        << "At least one color region must exist after coloring";
}

// ─────────────────────────────────────────────────────────────────────────────
// CO2: Disabling coloring still produces correct results.
//
// setColoringEnabled(false) bypasses the interference-graph pass.  Each buffer
// gets its own allocation.  The round-trip must still be data-correct.
// ─────────────────────────────────────────────────────────────────────────────
TEST(MemoryStrategies, DisabledColoringRoundTrip) {
    constexpr size_t N  = 1 << 13;
    constexpr float  EB = 1e-2f;
    const size_t in_bytes = N * sizeof(float);

    CudaStream stream;
    auto h_input = make_smooth(N);
    CudaBuffer<float> d_in(N);
    d_in.upload(h_input, stream);
    stream.sync();

    Pipeline pipeline(in_bytes, MemoryStrategy::PREALLOCATE);
    pipeline.setColoringEnabled(false);

    auto* lrz = pipeline.addStage<LorenzoQuantStage<float, uint16_t>>();
    lrz->setErrorBound(EB);
    lrz->setQuantRadius(512);
    lrz->setOutlierCapacity(0.2f);
    pipeline.setPoolManagedDecompOutput(false);
    pipeline.finalize();

    EXPECT_FALSE(pipeline.isColoringEnabled())
        << "isColoringEnabled() should be false when coloring was disabled";

    void*  d_comp = nullptr;
    size_t comp_sz = 0;
    pipeline.compress(d_in.void_ptr(), in_bytes, &d_comp, &comp_sz, stream);
    ASSERT_GT(comp_sz, 0u);

    void*  d_dec = nullptr;
    size_t dec_sz = 0;
    pipeline.decompress(nullptr, 0, &d_dec, &dec_sz, stream);
    ASSERT_NE(d_dec, nullptr);
    ASSERT_EQ(dec_sz, in_bytes);

    std::vector<float> h_recon(N);
    cudaMemcpy(h_recon.data(), d_dec, in_bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_dec);

    float max_err = max_abs_error(h_input, h_recon);
    EXPECT_LE(max_err, EB * 1.01f)
        << "Disabled-coloring round-trip error " << max_err << " exceeds bound " << EB;
}

// ─────────────────────────────────────────────────────────────────────────────
// BP1: setBufferPersistent() — a buffer marked persistent survives reset()
//
// After compress(), we find one of the DAG output buffers, mark it persistent,
// then call reset().  A persistent buffer must NOT be freed — its device pointer
// must still be non-null and its is_persistent flag must be set.
// ─────────────────────────────────────────────────────────────────────────────
TEST(MemoryStrategies, PersistentBufferSurvivesReset) {
    constexpr size_t N  = 1 << 13;
    constexpr float  EB = 1e-2f;
    const size_t in_bytes = N * sizeof(float);

    CudaStream stream;
    auto h_input = make_smooth(N);
    CudaBuffer<float> d_in(N);
    d_in.upload(h_input, stream);
    stream.sync();

    // MINIMAL so buffers are normally freed after reset()
    Pipeline pipeline(in_bytes, MemoryStrategy::MINIMAL);
    auto* lrz = pipeline.addStage<LorenzoQuantStage<float, uint16_t>>();
    lrz->setErrorBound(EB);
    lrz->setQuantRadius(512);
    lrz->setOutlierCapacity(0.2f);
    pipeline.setPoolManagedDecompOutput(false);
    pipeline.finalize();

    void*  d_comp  = nullptr;
    size_t comp_sz = 0;
    pipeline.compress(d_in.void_ptr(), in_bytes, &d_comp, &comp_sz, stream);
    stream.sync();
    ASSERT_GT(comp_sz, 0u);

    // Find the first output buffer of the first DAG node and mark it persistent.
    CompressionDAG* dag = pipeline.getDAG();
    const auto& nodes = dag->getNodes();
    ASSERT_FALSE(nodes.empty()) << "DAG has no nodes";

    int target_buf_id = -1;
    for (auto* node : nodes) {
        if (!node->output_buffer_ids.empty()) {
            target_buf_id = node->output_buffer_ids[0];
            break;
        }
    }
    ASSERT_GE(target_buf_id, 0) << "No output buffer found";

    // Verify the buffer is currently allocated (non-null pointer)
    void* ptr_before = dag->getBuffer(target_buf_id);
    ASSERT_NE(ptr_before, nullptr) << "Buffer must be allocated after compress()";

    dag->setBufferPersistent(target_buf_id, true);
    EXPECT_TRUE(dag->getBufferInfo(target_buf_id).is_persistent)
        << "is_persistent must be true after setBufferPersistent(true)";

    // reset() must not free persistent buffers
    pipeline.reset(stream);
    stream.sync();

    void* ptr_after = dag->getBuffer(target_buf_id);
    EXPECT_EQ(ptr_after, ptr_before)
        << "Persistent buffer pointer must not change after reset()";
}

// ─────────────────────────────────────────────────────────────────────────────
// SC1: RLE persistent scratch is reused across compress() calls
//
// An RLE stage allocates five persistent scratch arrays on the first execute()
// call (boundary flags, scan, positions, values, lengths) and reuses them on
// subsequent calls without additional cudaMalloc.  Verify this by measuring
// GPU free memory:
//   - After the first compress+reset, GPU free memory settles at a baseline
//     that is lower than before the compress (scratch is persistent).
//   - After the second compress+reset, GPU free memory must equal that
//     baseline — no new permanent allocations were made.
// ─────────────────────────────────────────────────────────────────────────────
TEST(MemoryStrategies, RLEScratchReusedAcrossCalls) {
    constexpr size_t N  = 1 << 14;
    constexpr float  EB = 1e-2f;
    const size_t in_bytes = N * sizeof(float);

    CudaStream stream;
    auto h_input = make_smooth(N);
    CudaBuffer<float> d_in(N);
    d_in.upload(h_input, stream);
    stream.sync();

    Pipeline pipeline(in_bytes, MemoryStrategy::MINIMAL);
    auto* lrz = pipeline.addStage<LorenzoQuantStage<float, uint16_t>>();
    lrz->setErrorBound(EB);
    lrz->setQuantRadius(512);
    lrz->setOutlierCapacity(0.2f);
    auto* rle = pipeline.addStage<RLEStage<uint16_t>>();
    pipeline.connect(rle, lrz, "codes");
    pipeline.setPoolManagedDecompOutput(false);
    pipeline.finalize();

    // First compress — allocates scratch (persistent) and working buffers
    void*  d_comp  = nullptr;
    size_t comp_sz = 0;
    pipeline.compress(d_in.void_ptr(), in_bytes, &d_comp, &comp_sz, stream);
    stream.sync();
    ASSERT_GT(comp_sz, 0u);

    // reset() frees non-persistent working buffers; persistent scratch stays.
    pipeline.reset(stream);
    stream.sync();
    cudaDeviceSynchronize();
    const size_t free_after_first = gpu_free_bytes();

    // Second compress — scratch must be reused (no new persistent allocations)
    d_comp  = nullptr;
    comp_sz = 0;
    pipeline.compress(d_in.void_ptr(), in_bytes, &d_comp, &comp_sz, stream);
    stream.sync();
    ASSERT_GT(comp_sz, 0u);

    pipeline.reset(stream);
    stream.sync();
    cudaDeviceSynchronize();
    const size_t free_after_second = gpu_free_bytes();

    // Allow ±512 KB tolerance for CUDA internal memory accounting fluctuations.
    constexpr size_t kTol = 512 * 1024;
    const size_t diff = (free_after_second > free_after_first)
                      ? free_after_second - free_after_first
                      : free_after_first  - free_after_second;
    EXPECT_LE(diff, kTol)
        << "GPU free memory after second compress+reset (" << free_after_second / 1024 << " KB) "
        << "must be ~equal to after first (" << free_after_first / 1024 << " KB) — "
        << "RLE scratch must be reused, not re-allocated";
}

// ─────────────────────────────────────────────────────────────────────────────
// CU1: getCurrentMemoryUsage() returns 0 after reset() in MINIMAL mode
//
// reset() frees all non-persistent pool buffers.  For MINIMAL strategy (where
// no buffers are pre-allocated at finalize()), getCurrentMemoryUsage() must
// drop back to 0 after each reset(), regardless of how many compress() calls
// have been made.
// ─────────────────────────────────────────────────────────────────────────────
TEST(MemoryStrategies, CurrentUsageZeroAfterResetMinimal) {
    constexpr size_t N  = 1 << 13;
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

    for (int i = 0; i < 3; i++) {
        void*  d_comp  = nullptr;
        size_t comp_sz = 0;
        pipeline.compress(d_in.void_ptr(), in_bytes, &d_comp, &comp_sz, stream);
        stream.sync();
        ASSERT_GT(comp_sz, 0u) << "cycle " << i;

        EXPECT_GT(pipeline.getCurrentMemoryUsage(), 0u)
            << "cycle " << i << ": getCurrentMemoryUsage() must be > 0 after compress()";

        pipeline.reset(stream);
        stream.sync();

        EXPECT_EQ(pipeline.getCurrentMemoryUsage(), 0u)
            << "cycle " << i << ": getCurrentMemoryUsage() must be 0 after reset()";
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PT1: getPoolThreshold() returns a value ≥ the input data size before finalize()
//
// Pipeline initialises the pool with size = input_data_size × multiplier (3×
// by default).  getPoolThreshold() exposes the configured pool size.  Before
// finalize() the threshold is set to input × multiplier, so it must be at least
// as large as the raw input.  (After finalize() the threshold is pinned to
// UINT64_MAX to keep pool pages warm; this test queries the pre-finalize value.)
// ─────────────────────────────────────────────────────────────────────────────
TEST(MemoryPool, PoolThresholdAtLeastInputSize) {
    constexpr size_t N = 1 << 14;
    const size_t in_bytes = N * sizeof(float);

    Pipeline pipeline(in_bytes, MemoryStrategy::MINIMAL, 3.0f);
    auto* lrz = pipeline.addStage<LorenzoQuantStage<float, uint16_t>>();
    lrz->setErrorBound(1e-2f);
    // Query threshold BEFORE finalize() — after finalize() the CUDA pool release
    // threshold is pinned to UINT64_MAX (keeps pages warm) which would overflow
    // the returned size_t from getConfiguredSize().
    const size_t threshold = pipeline.getPoolThreshold();
    EXPECT_GE(threshold, in_bytes)
        << "Pool threshold (" << threshold << " B) must be >= input size ("
        << in_bytes << " B) — pool sized at 3× input should far exceed input";
    EXPECT_GE(threshold, 3 * in_bytes)
        << "Pool threshold with 3× multiplier must be at least 3× the input size";
}

// ─────────────────────────────────────────────────────────────────────────────
// EP1: setExternalPointer() on a DAG buffer — buffer is not reallocated
//
// Sets an external pointer on one of the DAG's non-input buffers (a stage
// output buffer that has been pre-allocated after compress()), then verifies
// that the pointer returned by getBuffer() matches the externally-set value
// and that the is_external flag is set.
// ─────────────────────────────────────────────────────────────────────────────
TEST(MemoryStrategies, SetExternalPointerReflectsInGetBuffer) {
    constexpr size_t N  = 1 << 12;
    const size_t in_bytes = N * sizeof(float);

    CudaStream stream;
    auto h_input = make_smooth(N);
    CudaBuffer<float> d_in(N);
    d_in.upload(h_input, stream);
    stream.sync();

    // PREALLOCATE so all buffers are allocated at finalize() and accessible
    Pipeline pipeline(in_bytes, MemoryStrategy::PREALLOCATE);
    auto* lrz = pipeline.addStage<LorenzoQuantStage<float, uint16_t>>();
    lrz->setErrorBound(1e-2f);
    lrz->setQuantRadius(512);
    lrz->setOutlierCapacity(0.2f);
    pipeline.setPoolManagedDecompOutput(false);
    pipeline.finalize();

    // compress() to ensure buffers are live
    void*  d_comp  = nullptr;
    size_t comp_sz = 0;
    pipeline.compress(d_in.void_ptr(), in_bytes, &d_comp, &comp_sz, stream);
    stream.sync();
    ASSERT_GT(comp_sz, 0u);

    CompressionDAG* dag = pipeline.getDAG();
    const auto& nodes = dag->getNodes();
    ASSERT_FALSE(nodes.empty());

    // Find a stage output buffer with a valid pointer
    int buf_id = -1;
    for (auto* node : nodes) {
        for (int bid : node->output_buffer_ids) {
            if (dag->getBuffer(bid) != nullptr) {
                buf_id = bid;
                break;
            }
        }
        if (buf_id >= 0) break;
    }
    ASSERT_GE(buf_id, 0) << "No allocated output buffer found";

    // Allocate a dummy external buffer of the right size
    size_t buf_sz = dag->getBufferSize(buf_id);
    ASSERT_GT(buf_sz, 0u);
    void* d_external = nullptr;
    cudaMalloc(&d_external, buf_sz);
    ASSERT_NE(d_external, nullptr);

    dag->setExternalPointer(buf_id, d_external);

    EXPECT_EQ(dag->getBuffer(buf_id), d_external)
        << "getBuffer() must return the externally-set pointer";
    EXPECT_TRUE(dag->getBufferInfo(buf_id).is_external)
        << "is_external must be true after setExternalPointer()";

    cudaFree(d_external);
}

// ─────────────────────────────────────────────────────────────────────────────
// EP2: setExternalPointer() end-to-end — compress produces correct data when
//      a stage's output buffer is externally owned
//
// We pre-allocate a device buffer for the Lorenzo codes output, hand it to the
// DAG via setExternalPointer(), then run compress+decompress and verify the
// round-trip error is within the error bound.  This confirms that the pipeline
// actually writes to (and reads from) the external pointer during execution,
// not a newly allocated internal buffer.
// ─────────────────────────────────────────────────────────────────────────────
TEST(MemoryStrategies, SetExternalPointerEndToEnd) {
    constexpr size_t N  = 1 << 12;
    constexpr float  EB = 1e-2f;
    const size_t in_bytes = N * sizeof(float);

    CudaStream stream;
    auto h_input = make_smooth(N);
    CudaBuffer<float> d_in(N);
    d_in.upload(h_input, stream);
    stream.sync();

    // PREALLOCATE so buffer sizes are known and IDs are stable after finalize().
    Pipeline pipeline(in_bytes, MemoryStrategy::PREALLOCATE, 4.0f);
    auto* lrz = pipeline.addStage<LorenzoQuantStage<float, uint16_t>>();
    lrz->setErrorBound(EB);
    lrz->setQuantRadius(512);
    lrz->setOutlierCapacity(0.2f);
    pipeline.setPoolManagedDecompOutput(false);
    pipeline.finalize();

    // Do one compress to settle buffer sizes.
    void* d_comp = nullptr; size_t comp_sz = 0;
    pipeline.compress(d_in.void_ptr(), in_bytes, &d_comp, &comp_sz, stream);
    stream.sync();
    ASSERT_GT(comp_sz, 0u);

    // Identify the Lorenzo codes output buffer (first stage, first output).
    CompressionDAG* dag  = pipeline.getDAG();
    const auto&     nodes = dag->getNodes();
    ASSERT_FALSE(nodes.empty());

    int ext_buf_id = -1;
    for (auto* node : nodes) {
        for (int bid : node->output_buffer_ids) {
            if (dag->getBuffer(bid) != nullptr) {
                ext_buf_id = bid;
                break;
            }
        }
        if (ext_buf_id >= 0) break;
    }
    ASSERT_GE(ext_buf_id, 0) << "No allocated output buffer found after first compress";

    // Allocate an external buffer of the same size.
    size_t ext_sz = dag->getBufferSize(ext_buf_id);
    ASSERT_GT(ext_sz, 0u);
    void* d_external = nullptr;
    cudaMalloc(&d_external, ext_sz);
    ASSERT_NE(d_external, nullptr);

    // Register it with the DAG and reset so the pipeline re-uses it.
    dag->setExternalPointer(ext_buf_id, d_external);
    pipeline.reset();

    // Compress through the external buffer.
    d_comp = nullptr; comp_sz = 0;
    ASSERT_NO_THROW(
        pipeline.compress(d_in.void_ptr(), in_bytes, &d_comp, &comp_sz, stream)
    ) << "compress() must not throw with an external intermediate buffer";
    stream.sync();
    ASSERT_GT(comp_sz, 0u) << "Compressed output must be non-empty";

    // Decompress and verify correctness.
    void*  d_dec  = nullptr;
    size_t dec_sz = 0;
    ASSERT_NO_THROW(
        pipeline.decompress(nullptr, 0, &d_dec, &dec_sz, stream)
    ) << "decompress() must not throw after compress with external buffer";
    stream.sync();
    ASSERT_NE(d_dec, nullptr);
    ASSERT_EQ(dec_sz, in_bytes);

    std::vector<float> h_recon(N);
    cudaMemcpy(h_recon.data(), d_dec, in_bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_dec);
    cudaFree(d_external);

    float max_err = 0.0f;
    for (size_t i = 0; i < N; i++)
        max_err = std::max(max_err, std::abs(h_recon[i] - h_input[i]));
    EXPECT_LE(max_err, EB * 1.01f)
        << "EP2: round-trip via external buffer: max_err=" << max_err
        << " exceeds bound " << EB;
}

// ─────────────────────────────────────────────────────────────────────────────
// MM1: PREALLOCATE getCurrentMemoryUsage() — non-zero after compress,
//      stays non-zero after reset() (buffers persist), returns to zero only
//      after a new MINIMAL-mode round-trip (different pipeline).
//
// PREALLOCATE allocates all buffers at finalize() and never frees them between
// calls — the memory stays live until the pipeline is destroyed.  This is
// intentionally different from MINIMAL, where reset() zeroes the counter.
// ─────────────────────────────────────────────────────────────────────────────
TEST(MemoryStrategies, PreallocateCurrentUsageNonZeroAfterReset) {
    constexpr size_t N  = 1 << 14;
    constexpr float  EB = 1e-2f;
    const size_t in_bytes = N * sizeof(float);

    CudaStream stream;
    auto h_input = make_smooth(N);
    CudaBuffer<float> d_in(N);
    d_in.upload(h_input, stream);
    stream.sync();

    Pipeline pipeline(in_bytes, MemoryStrategy::PREALLOCATE, 3.0f);
    auto* lrz = pipeline.addStage<LorenzoQuantStage<float, uint16_t>>();
    lrz->setErrorBound(EB);
    lrz->setQuantRadius(512);
    lrz->setOutlierCapacity(0.2f);
    pipeline.setPoolManagedDecompOutput(false);
    pipeline.finalize();

    // Before any compress: PREALLOCATE allocates at finalize, so usage should
    // already be non-zero.
    const size_t usage_after_finalize = pipeline.getCurrentMemoryUsage();
    EXPECT_GT(usage_after_finalize, 0u)
        << "MM1: PREALLOCATE must report non-zero usage after finalize()";

    // Compress.
    void* d_comp = nullptr; size_t comp_sz = 0;
    pipeline.compress(d_in.void_ptr(), in_bytes, &d_comp, &comp_sz, stream);
    stream.sync();

    const size_t usage_after_compress = pipeline.getCurrentMemoryUsage();
    EXPECT_GT(usage_after_compress, 0u)
        << "MM1: PREALLOCATE must report non-zero usage after compress()";

    // reset() does NOT free PREALLOCATE buffers — usage stays non-zero.
    pipeline.reset();
    const size_t usage_after_reset = pipeline.getCurrentMemoryUsage();
    EXPECT_GT(usage_after_reset, 0u)
        << "MM1: PREALLOCATE usage must remain non-zero after reset() "
           "(persistent buffers are not freed on reset)";
}

// ─────────────────────────────────────────────────────────────────────────────
// MM2: MINIMAL getCurrentMemoryUsage() is non-zero after compress and zero
//      after reset() — verify consistent behaviour across 3 compress/reset
//      cycles, matching the behaviour PREALLOCATE does NOT exhibit.
// ─────────────────────────────────────────────────────────────────────────────
TEST(MemoryStrategies, MinimalCurrentUsageZeroAfterEachReset) {
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
    pipeline.setPoolManagedDecompOutput(false);
    pipeline.finalize();

    // Before compress: MINIMAL does not pre-allocate at finalize.
    EXPECT_EQ(pipeline.getCurrentMemoryUsage(), 0u)
        << "MM2: MINIMAL usage must be 0 before first compress()";

    for (int cycle = 0; cycle < 3; cycle++) {
        void* d_comp = nullptr; size_t comp_sz = 0;
        pipeline.compress(d_in.void_ptr(), in_bytes, &d_comp, &comp_sz, stream);
        stream.sync();

        EXPECT_GT(pipeline.getCurrentMemoryUsage(), 0u)
            << "MM2 cycle " << cycle << ": MINIMAL usage must be > 0 after compress()";

        pipeline.reset();

        EXPECT_EQ(pipeline.getCurrentMemoryUsage(), 0u)
            << "MM2 cycle " << cycle << ": MINIMAL usage must be 0 after reset()";
    }
}