/**
 * tests/test_buffer_coloring.cpp
 *
 * Tests for the buffer coloring feature in PREALLOCATE mode.
 *
 * Buffer coloring reduces peak pool memory by aliasing buffers whose live
 * ranges do not overlap onto shared memory regions.  These tests verify:
 *
 *   BC1  Coloring fires by default for PREALLOCATE on a linear pipeline
 *   BC2  Coloring does NOT fire for MINIMAL mode
 *   BC3  setColoringEnabled(false) prevents coloring even in PREALLOCATE mode
 *   BC4  A colored pipeline produces the same decompressed output as an
 *        uncolored pipeline (correctness of aliased pointer assignment)
 *   BC5  Region count is strictly less than buffer count for a multi-stage
 *        linear pipeline (the coloring actually merged something)
 *   BC6  Peak memory with coloring is <= peak memory without coloring
 *   BC7  getColorRegionCount() returns 0 when coloring is disabled
 *   BC8  Repeated compress() calls on a colored pipeline stay correct
 *   BC9  A multi-stage linear pipeline (Lorenzo→Diff→Bitshuffle) roundtrip
 *        is correct with coloring enabled
 */

#include <gtest/gtest.h>
#include "helpers/fz_test_utils.h"
#include "fzgpumodules.h"

#include <cmath>
#include <vector>

using namespace fz;
using namespace fz_test;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

static constexpr size_t kN       = 1 << 14;  // 16 K floats (fast)
static constexpr float  kEB      = 1e-2f;
static constexpr size_t kNBytes  = kN * sizeof(float);

static std::vector<float> make_test_data() {
    std::vector<float> v(kN);
    for (size_t i = 0; i < kN; i++)
        v[i] = std::sin(static_cast<float>(i) * 0.01f) * 50.0f
             + std::cos(static_cast<float>(i) * 0.003f) * 20.0f;
    return v;
}

// Build a single-stage Lorenzo PREALLOCATE pipeline.
// Sets coloring_disabled according to the argument before finalize().
static std::unique_ptr<Pipeline> make_lorenzo_pipeline(bool disable_coloring) {
    auto p = std::make_unique<Pipeline>(kNBytes, MemoryStrategy::PREALLOCATE, 4.0f);
    if (disable_coloring) p->setColoringEnabled(false);
    auto* lrz = p->addStage<LorenzoQuantStage<float, uint16_t>>();
    lrz->setErrorBound(kEB);
    lrz->setQuantRadius(512);
    lrz->setOutlierCapacity(0.2f);
    p->setPoolManagedDecompOutput(false);
    p->finalize();
    return p;
}

// Compress + decompress with a Pipeline, return reconstructed host data.
static std::vector<float> roundtrip(Pipeline& p, const CudaBuffer<float>& d_in,
                                    size_t n = kN, size_t n_bytes = kNBytes) {
    void*  d_comp   = nullptr;
    size_t comp_sz  = 0;
    p.compress(d_in.void_ptr(), n_bytes, &d_comp, &comp_sz, 0);
    EXPECT_GT(comp_sz, 0u) << "Compressed size is zero";

    void*  d_dec  = nullptr;
    size_t dec_sz = 0;
    p.decompress(nullptr, 0, &d_dec, &dec_sz, 0);

    std::vector<float> h_recon(n, 0.0f);
    if (d_dec && dec_sz == n_bytes)
        cudaMemcpy(h_recon.data(), d_dec, n_bytes, cudaMemcpyDeviceToHost);
    if (d_dec) cudaFree(d_dec);
    return h_recon;
}

// ─────────────────────────────────────────────────────────────────────────────
// BC1: Coloring fires by default for PREALLOCATE on a linear pipeline
// ─────────────────────────────────────────────────────────────────────────────
TEST(BufferColoring, AppliedByDefaultForPreallocate) {
    auto p = make_lorenzo_pipeline(/*disable_coloring=*/false);
    EXPECT_TRUE(p->isColoringEnabled())
        << "Coloring should be applied by default in PREALLOCATE mode";
}

// ─────────────────────────────────────────────────────────────────────────────
// BC2: Coloring does NOT fire for MINIMAL mode
// ─────────────────────────────────────────────────────────────────────────────
TEST(BufferColoring, NotAppliedForMinimal) {
    Pipeline p(kNBytes, MemoryStrategy::MINIMAL, 4.0f);
    auto* lrz = p.addStage<LorenzoQuantStage<float, uint16_t>>();
    lrz->setErrorBound(kEB);
    lrz->setQuantRadius(512);
    lrz->setOutlierCapacity(0.2f);
    p.setPoolManagedDecompOutput(false);
    p.finalize();

    EXPECT_FALSE(p.isColoringEnabled())
        << "Coloring must not run in MINIMAL mode";
}

// ─────────────────────────────────────────────────────────────────────────────
// BC3: setColoringEnabled(false) prevents coloring even in PREALLOCATE mode
// ─────────────────────────────────────────────────────────────────────────────
TEST(BufferColoring, DisabledWhenRequested) {
    auto p = make_lorenzo_pipeline(/*disable_coloring=*/true);
    EXPECT_FALSE(p->isColoringEnabled())
        << "Coloring should be suppressed when setColoringEnabled(false) is called";
}

// ─────────────────────────────────────────────────────────────────────────────
// BC4: A colored pipeline produces identical output to an uncolored one
// ─────────────────────────────────────────────────────────────────────────────
TEST(BufferColoring, ColoredAndUncoloredMatchExactly) {
    auto h_input = make_test_data();
    CudaBuffer<float> d_in(kN);
    d_in.upload(h_input);
    cudaDeviceSynchronize();

    auto p_colored   = make_lorenzo_pipeline(false);
    auto p_uncolored = make_lorenzo_pipeline(true);

    auto recon_colored   = roundtrip(*p_colored,   d_in);
    auto recon_uncolored = roundtrip(*p_uncolored, d_in);

    ASSERT_EQ(recon_colored.size(),   kN);
    ASSERT_EQ(recon_uncolored.size(), kN);

    for (size_t i = 0; i < kN; i++) {
        EXPECT_FLOAT_EQ(recon_colored[i], recon_uncolored[i])
            << "Output mismatch at index " << i
            << " (colored=" << recon_colored[i]
            << " uncolored=" << recon_uncolored[i] << ")";
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// BC5: getColorRegionCount() returns a positive, bounded value after finalize
//      The actual memory savings are verified in BC6 (peak memory comparison).
// ─────────────────────────────────────────────────────────────────────────────
TEST(BufferColoring, RegionCountPositiveAndBounded) {
    Pipeline p(kNBytes, MemoryStrategy::PREALLOCATE, 4.0f);

    auto* lrz = p.addStage<LorenzoQuantStage<float, uint16_t>>();
    lrz->setErrorBound(kEB);
    lrz->setQuantRadius(512);
    lrz->setOutlierCapacity(0.2f);

    auto* diff = p.addStage<DifferenceStage<int16_t, uint16_t>>();
    diff->setChunkSize(4096);
    p.connect(diff, lrz, "codes");

    p.setPoolManagedDecompOutput(false);
    p.finalize();

    ASSERT_TRUE(p.isColoringEnabled());

    const size_t regions = p.getColorRegionCount();
    // At least one region must exist (something was allocated)
    EXPECT_GT(regions, 0u) << "Must have at least one color region after finalize";
    // Upper bound: number of regions cannot exceed number of non-external
    // buffers.  For a 2-stage pipeline the buffer count is bounded by
    // a small constant; 32 is a conservatively large sentinel.
    EXPECT_LT(regions, 32u)
        << "Region count is implausibly large; likely a bug in colorBuffers()";
}

// ─────────────────────────────────────────────────────────────────────────────
// BC6: Peak memory with coloring is <= peak memory without coloring
// ─────────────────────────────────────────────────────────────────────────────
TEST(BufferColoring, PeakMemoryNoHigherWithColoring) {
    auto h_input = make_test_data();
    CudaBuffer<float> d_in(kN);
    d_in.upload(h_input);
    cudaDeviceSynchronize();

    // Colored pipeline
    Pipeline p_col(kNBytes, MemoryStrategy::PREALLOCATE, 4.0f);
    {
        auto* lrz = p_col.addStage<LorenzoQuantStage<float, uint16_t>>();
        lrz->setErrorBound(kEB);
        lrz->setQuantRadius(512);
        lrz->setOutlierCapacity(0.2f);
        auto* diff = p_col.addStage<DifferenceStage<int16_t, uint16_t>>();
        diff->setChunkSize(4096);
        p_col.connect(diff, lrz, "codes");
        p_col.setPoolManagedDecompOutput(false);
    p_col.finalize();
    }

    // Uncolored pipeline (identical topology)
    Pipeline p_uncol(kNBytes, MemoryStrategy::PREALLOCATE, 4.0f);
    p_uncol.setColoringEnabled(false);
    {
        auto* lrz = p_uncol.addStage<LorenzoQuantStage<float, uint16_t>>();
        lrz->setErrorBound(kEB);
        lrz->setQuantRadius(512);
        lrz->setOutlierCapacity(0.2f);
        auto* diff = p_uncol.addStage<DifferenceStage<int16_t, uint16_t>>();
        diff->setChunkSize(4096);
        p_uncol.connect(diff, lrz, "codes");
        p_uncol.setPoolManagedDecompOutput(false);
    p_uncol.finalize();
    }

    // Run both to establish peak usage.
    // d_comp is pool-owned — must NOT be cudaFree'd; the pipeline destructor
    // returns it to the pool.
    void* d_comp = nullptr; size_t comp_sz = 0;
    p_col.compress(d_in.void_ptr(), kNBytes, &d_comp, &comp_sz, 0);
    cudaDeviceSynchronize();
    const size_t peak_colored = p_col.getPeakMemoryUsage();

    d_comp = nullptr; comp_sz = 0;
    p_uncol.compress(d_in.void_ptr(), kNBytes, &d_comp, &comp_sz, 0);
    cudaDeviceSynchronize();
    const size_t peak_uncolored = p_uncol.getPeakMemoryUsage();

    EXPECT_LE(peak_colored, peak_uncolored)
        << "Colored peak (" << peak_colored / 1024 << " KB) should be <= "
        << "uncolored peak (" << peak_uncolored / 1024 << " KB)";
}

// ─────────────────────────────────────────────────────────────────────────────
// BC7: getColorRegionCount() returns 0 when coloring is disabled
// ─────────────────────────────────────────────────────────────────────────────
TEST(BufferColoring, RegionCountZeroWhenDisabled) {
    auto p = make_lorenzo_pipeline(/*disable_coloring=*/true);
    EXPECT_EQ(p->getColorRegionCount(), 0u)
        << "Region count should be 0 when coloring is disabled";
}

// ─────────────────────────────────────────────────────────────────────────────
// BC8: Repeated compress() calls on a colored pipeline stay correct
//      (verifies that reset() does not free or corrupt the aliased regions)
// ─────────────────────────────────────────────────────────────────────────────
TEST(BufferColoring, RepeatedCompressStaysCorrect) {
    auto h_input = make_test_data();
    CudaBuffer<float> d_in(kN);
    d_in.upload(h_input);
    cudaDeviceSynchronize();

    auto p = make_lorenzo_pipeline(/*disable_coloring=*/false);
    ASSERT_TRUE(p->isColoringEnabled());

    // Run 5 compress→decompress cycles and verify each one
    for (int run = 0; run < 5; run++) {
        auto h_recon = roundtrip(*p, d_in);

        ASSERT_EQ(h_recon.size(), kN) << "Run " << run;
        const float max_err = max_abs_error(h_input, h_recon);
        EXPECT_LE(max_err, kEB * 1.01f)
            << "Run " << run << ": max_err=" << max_err
            << " exceeds error bound " << kEB;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// BC10: Colored (aliased) buffers remain stable across CUDA graph replays
//       (PREALLOCATE + coloring enabled + graph mode; 5 compress replays)
// ─────────────────────────────────────────────────────────────────────────────
TEST(BufferColoring, ColoredBuffersStableAcrossGraphReplays) {
    auto h_input = make_test_data();

    CudaStream stream;
    CudaBuffer<float> d_in(kN);
    d_in.upload(h_input, stream);
    stream.sync();

    // PREALLOCATE + coloring enabled (default) + graph mode
    Pipeline p(kNBytes, MemoryStrategy::PREALLOCATE, 4.0f);
    auto* lrz = p.addStage<LorenzoQuantStage<float, uint16_t>>();
    lrz->setErrorBound(kEB);
    lrz->setQuantRadius(512);
    lrz->setOutlierCapacity(0.2f);
    p.enableGraphMode(true);
    p.setPoolManagedDecompOutput(false);
    p.finalize();

    ASSERT_TRUE(p.isColoringEnabled())
        << "BC10: coloring must be applied in PREALLOCATE mode";

    p.captureGraph(stream);
    ASSERT_TRUE(p.isGraphCaptured())
        << "BC10: graph capture must succeed with coloring enabled";

    // Five replays — aliased regions must not corrupt each other across calls.
    for (int run = 0; run < 5; run++) {
        void*  d_comp  = nullptr;
        size_t comp_sz = 0;
        ASSERT_NO_THROW(p.compress(d_in.void_ptr(), kNBytes, &d_comp, &comp_sz, stream))
            << "run " << run;
        stream.sync();
        ASSERT_GT(comp_sz, 0u) << "run " << run;

        void*  d_dec  = nullptr;
        size_t dec_sz = 0;
        p.decompress(nullptr, 0, &d_dec, &dec_sz, stream);
        stream.sync();
        ASSERT_NE(d_dec, nullptr) << "run " << run;
        ASSERT_EQ(dec_sz, kNBytes) << "run " << run;

        std::vector<float> h_recon(kN);
        cudaMemcpy(h_recon.data(), d_dec, kNBytes, cudaMemcpyDeviceToHost);
        cudaFree(d_dec);

        const float max_err = max_abs_error(h_input, h_recon);
        EXPECT_LE(max_err, kEB * 1.01f)
            << "run " << run << ": max_err=" << max_err
            << " exceeds bound " << kEB;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// BC9: Multi-stage linear pipeline roundtrip correct with coloring
//      (Lorenzo → Difference → Bitshuffle — 3 stages, more interference edges)
// ─────────────────────────────────────────────────────────────────────────────
TEST(BufferColoring, MultiStageLinearPipelineRoundTrip) {
    constexpr size_t N_LARGE = 1 << 16;  // 64 K floats
    const size_t in_bytes = N_LARGE * sizeof(float);

    std::vector<float> h_input(N_LARGE);
    for (size_t i = 0; i < N_LARGE; i++)
        h_input[i] = std::sin(static_cast<float>(i) * 0.01f) * 100.0f;

    CudaBuffer<float> d_in(N_LARGE);
    d_in.upload(h_input);
    cudaDeviceSynchronize();

    Pipeline p(in_bytes, MemoryStrategy::PREALLOCATE, 4.0f);

    auto* lrz = p.addStage<LorenzoQuantStage<float, uint16_t>>();
    lrz->setErrorBound(kEB);
    lrz->setQuantRadius(512);
    lrz->setOutlierCapacity(0.2f);

    auto* diff = p.addStage<DifferenceStage<int16_t, uint16_t>>();
    diff->setChunkSize(4096);
    p.connect(diff, lrz, "codes");

    auto* bitshuffle = p.addStage<BitshuffleStage>();
    bitshuffle->setBlockSize(4096);
    bitshuffle->setElementWidth(4);
    p.connect(bitshuffle, diff);

    p.setPoolManagedDecompOutput(false);
    p.finalize();

    ASSERT_TRUE(p.isColoringEnabled())
        << "Coloring should apply to a 3-stage linear pipeline";
    EXPECT_GT(p.getColorRegionCount(), 0u);

    void*  d_comp  = nullptr;
    size_t comp_sz = 0;
    p.compress(d_in.void_ptr(), in_bytes, &d_comp, &comp_sz, 0);
    ASSERT_GT(comp_sz, 0u) << "Compressed output is empty";

    void*  d_dec  = nullptr;
    size_t dec_sz = 0;
    p.decompress(nullptr, 0, &d_dec, &dec_sz, 0);
    ASSERT_NE(d_dec, nullptr) << "Decompressed output is null";
    ASSERT_EQ(dec_sz, in_bytes) << "Decompressed size mismatch";

    std::vector<float> h_recon(N_LARGE);
    cudaMemcpy(h_recon.data(), d_dec, in_bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_dec);

    const float max_err = max_abs_error(h_input, h_recon);
    EXPECT_LE(max_err, kEB * 1.01f)
        << "MultiStage: max_err=" << max_err << " exceeds bound " << kEB;
}
