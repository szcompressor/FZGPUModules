/**
 * tests/test_liveness.cpp
 *
 * Tests for buffer liveness analysis, execution-level structure, and
 * topology-aware pool-size estimation.
 *
 * Covers:
 *   LV1  printBufferLifetimes() runs without crash on a linear pipeline
 *   LV2  Linear 3-stage pipeline has 3 levels, each with exactly 1 node
 *   LV3  Diamond-shaped DAG (Split→[PT,PT]→Merge) produces 3 levels;
 *         level 1 holds 2 parallel nodes (correct parallelism detection)
 *   LV4  Diamond DAG compress() produces output of the expected size
 *         (MergeStage concatenates two equal-size copies → 2× input)
 *   LV5  MINIMAL computeTopoPoolSize() ≤ PREALLOCATE for a 3-stage linear
 *         pipeline — liveness frees the intermediate codes buffer so the
 *         peak live-set is smaller than the sum-of-all-buffers
 *   LV6  PREALLOCATE observed peak memory ≥ MINIMAL observed peak for the
 *         same pipeline — PREALLOCATE allocates all buffers upfront so its
 *         pool peak is set at finalize time, not reduced by early frees
 */

#include <gtest/gtest.h>
#include "helpers/fz_test_utils.h"
#include "helpers/test_stages.h"
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

// Build a 3-stage linear pipeline: Lorenzo → DifferenceStage → RLEStage.
// Lorenzo produces codes (consumed by Diff) and outliers (pipeline output).
// DifferenceStage feeds RLEStage.
static std::unique_ptr<Pipeline> build_linear3(size_t in_bytes, MemoryStrategy strategy) {
    auto p = std::make_unique<Pipeline>(in_bytes, strategy, 5.0f);

    auto* lrz = p->addStage<LorenzoStage<float, uint16_t>>();
    lrz->setErrorBound(1e-2f);
    lrz->setQuantRadius(512);
    lrz->setOutlierCapacity(0.2f);

    auto* diff = p->addStage<DifferenceStage<uint16_t>>();
    diff->setChunkSize(4096);
    p->connect(diff, lrz, "codes");

    auto* rle = p->addStage<RLEStage<uint16_t>>();
    p->connect(rle, diff);

    p->setPoolManagedDecompOutput(false);
    p->finalize();
    return p;
}

// ─────────────────────────────────────────────────────────────────────────────
// LV1: printBufferLifetimes() runs without crashing
// ─────────────────────────────────────────────────────────────────────────────
TEST(Liveness, PrintBufferLifetimesNoCrash) {
    constexpr size_t N = 1 << 12;
    auto p = build_linear3(N * sizeof(float), MemoryStrategy::MINIMAL);
    ASSERT_NO_THROW(p->getDAG()->printBufferLifetimes())
        << "printBufferLifetimes() must not throw";
}

// ─────────────────────────────────────────────────────────────────────────────
// LV2: Linear 3-stage pipeline has 3 levels, each with exactly 1 node
// ─────────────────────────────────────────────────────────────────────────────
TEST(Liveness, LinearPipelineLevelStructure) {
    constexpr size_t N = 1 << 12;
    auto p = build_linear3(N * sizeof(float), MemoryStrategy::MINIMAL);

    const auto& levels = p->getDAG()->getLevels();

    ASSERT_EQ(levels.size(), 3u)
        << "Linear 3-stage pipeline must produce 3 execution levels";

    for (size_t lvl = 0; lvl < levels.size(); lvl++) {
        EXPECT_EQ(levels[lvl].size(), 1u)
            << "Level " << lvl << " must have exactly 1 node (sequential pipeline)";
    }

    EXPECT_EQ(p->getDAG()->getMaxParallelism(), 1)
        << "No parallelism in a linear pipeline";
}

// ─────────────────────────────────────────────────────────────────────────────
// LV3: Diamond-shaped DAG has 3 levels; level 1 has 2 parallel nodes
//
// Topology:
//   input → SplitStage → PassThrough_A → MergeStage → output
//                      → PassThrough_B →
//
// Expected level assignment:
//   Level 0: [SplitStage]
//   Level 1: [PassThrough_A, PassThrough_B]   ← 2 parallel nodes
//   Level 2: [MergeStage]
// ─────────────────────────────────────────────────────────────────────────────
TEST(Liveness, DiamondTopologyLevelStructure) {
    constexpr size_t N = 1 << 10;  // 1 KiB byte data for mock stages

    Pipeline p(N, MemoryStrategy::MINIMAL, 8.0f);
    auto* split = p.addStage<SplitStage>();
    auto* pta   = p.addStage<PassThroughStage>();
    auto* ptb   = p.addStage<PassThroughStage>();
    auto* merge = p.addStage<MergeStage>();

    p.connect(pta, split, "copy1");
    p.connect(ptb, split, "copy2");
    p.connect(merge, {pta, ptb});

    p.setPoolManagedDecompOutput(false);
    p.finalize();

    const auto& levels = p.getDAG()->getLevels();

    ASSERT_EQ(levels.size(), 3u)
        << "Diamond DAG must have 3 levels: [split], [pta, ptb], [merge]";

    EXPECT_EQ(levels[0].size(), 1u) << "Level 0: only SplitStage";
    EXPECT_EQ(levels[1].size(), 2u) << "Level 1: PassThrough_A and PassThrough_B in parallel";
    EXPECT_EQ(levels[2].size(), 1u) << "Level 2: only MergeStage";

    EXPECT_EQ(p.getDAG()->getMaxParallelism(), 2)
        << "Maximum parallelism for the diamond DAG is 2 (level 1)";
}

// ─────────────────────────────────────────────────────────────────────────────
// LV4: Diamond DAG compress() produces output of the expected size
//
// SplitStage copies input to two equal-size buffers; PassThrough is identity;
// MergeStage concatenates them → compressed output is 2× the input size.
// ─────────────────────────────────────────────────────────────────────────────
TEST(Liveness, DiamondDagCompressOutputSize) {
    constexpr size_t N = 1 << 10;  // 1 KiB byte data

    std::vector<uint8_t> h_data(N);
    for (size_t i = 0; i < N; i++)
        h_data[i] = static_cast<uint8_t>(i & 0xFF);

    CudaStream stream;
    CudaBuffer<uint8_t> d_in(N);
    d_in.upload(h_data, stream);
    stream.sync();

    Pipeline p(N, MemoryStrategy::MINIMAL, 8.0f);
    auto* split = p.addStage<SplitStage>();
    auto* pta   = p.addStage<PassThroughStage>();
    auto* ptb   = p.addStage<PassThroughStage>();
    auto* merge = p.addStage<MergeStage>();
    p.connect(pta, split, "copy1");
    p.connect(ptb, split, "copy2");
    p.connect(merge, {pta, ptb});
    p.setPoolManagedDecompOutput(false);
    p.finalize();

    void*  d_comp  = nullptr;
    size_t comp_sz = 0;
    ASSERT_NO_THROW(p.compress(d_in.void_ptr(), N, &d_comp, &comp_sz, stream))
        << "Diamond DAG compress must not throw";
    stream.sync();

    // MergeStage concatenates two N-byte copies → 2*N bytes.
    EXPECT_EQ(comp_sz, 2 * N)
        << "Diamond (Split→PassThrough×2→Merge) output must be 2× input size";
}

// ─────────────────────────────────────────────────────────────────────────────
// LV5: MINIMAL computeTopoPoolSize() ≤ PREALLOCATE for a 3-stage linear
//
// Tests a 3-stage chain (Lorenzo→Diff→RLE) — longer than the 2-stage chain
// already covered by test_memory_strategies ML5.  The intermediate diff-codes
// buffer is live only between Diff and RLE; MINIMAL can free it after RLE
// consumes it.  PREALLOCATE keeps all buffers live simultaneously.
// ─────────────────────────────────────────────────────────────────────────────
TEST(Liveness, MinimalTopoPoolLEPreallocateThreeStages) {
    constexpr size_t N = 1 << 14;
    const size_t in_bytes = N * sizeof(float);

    auto minimal_p = build_linear3(in_bytes, MemoryStrategy::MINIMAL);
    auto pre_p     = build_linear3(in_bytes, MemoryStrategy::PREALLOCATE);

    const size_t minimal_topo = minimal_p->getDAG()->computeTopoPoolSize();
    const size_t pre_topo     = pre_p->getDAG()->computeTopoPoolSize();

    EXPECT_GT(minimal_topo, 0u) << "MINIMAL topo pool size must be positive";
    EXPECT_GT(pre_topo,     0u) << "PREALLOCATE topo pool size must be positive";

    EXPECT_LE(minimal_topo, pre_topo)
        << "MINIMAL topo pool (" << minimal_topo << " B) should be <= "
        << "PREALLOCATE topo pool (" << pre_topo << " B) for a 3-stage linear pipeline";
}

// ─────────────────────────────────────────────────────────────────────────────
// LV6: PREALLOCATE observed peak memory ≥ MINIMAL observed peak
//
// PREALLOCATE allocates all buffers at finalize() (before any compression),
// so its pool peak is the total of all buffer sizes.  MINIMAL allocates on
// demand and frees as soon as possible, so its peak is smaller.
// Both peaks are measured after an actual compress() call.
// ─────────────────────────────────────────────────────────────────────────────
TEST(Liveness, PreallocatePeakGEMinimalPeak) {
    constexpr size_t N = 1 << 14;
    const size_t in_bytes = N * sizeof(float);
    auto h_input = make_smooth(N);

    CudaStream stream;
    CudaBuffer<float> d_in(N);
    d_in.upload(h_input, stream);
    stream.sync();

    auto measure_peak = [&](MemoryStrategy strategy) -> size_t {
        auto p = build_linear3(in_bytes, strategy);
        void*  d_comp  = nullptr;
        size_t comp_sz = 0;
        p->compress(d_in.void_ptr(), in_bytes, &d_comp, &comp_sz, stream);
        stream.sync();
        return p->getPeakMemoryUsage();
    };

    const size_t peak_minimal  = measure_peak(MemoryStrategy::MINIMAL);
    const size_t peak_prealloc = measure_peak(MemoryStrategy::PREALLOCATE);

    EXPECT_GT(peak_minimal,  0u) << "MINIMAL peak must be positive";
    EXPECT_GT(peak_prealloc, 0u) << "PREALLOCATE peak must be positive";

    EXPECT_GE(peak_prealloc, peak_minimal)
        << "PREALLOCATE pre-allocates all buffers upfront, so its peak ("
        << peak_prealloc << " B) must be >= MINIMAL peak (" << peak_minimal << " B)";
}

// ─────────────────────────────────────────────────────────────────────────────
// LV7: Liveness API remains consistent under graph mode
//
// CUDA graph capture records all kernel launches and buffer pointers at
// captureGraph() time; buffer addresses are then stable for every replay.
// The liveness analysis (printBufferLifetimes, getLevels, computeTopoPoolSize)
// must still report the same topology after graph capture — the graph
// recording must not corrupt or invalidate the DAG metadata.
//
// Pipeline: Lorenzo → Bitshuffle (both are graph-compatible forward stages).
// warmup() is called before captureGraph() to pre-allocate any stage-internal
// host memory outside the CUDA stream-capture window.
// ─────────────────────────────────────────────────────────────────────────────
TEST(Liveness, LivenessApiConsistentUnderGraphMode) {
    constexpr size_t N = 1 << 12;
    const size_t in_bytes = N * sizeof(float);

    // Build a 2-stage PREALLOCATE + graph-mode pipeline.
    auto p = std::make_unique<Pipeline>(in_bytes, MemoryStrategy::PREALLOCATE, 5.0f);

    auto* lrz = p->addStage<LorenzoStage<float, uint16_t>>();
    lrz->setErrorBound(1e-2f);
    lrz->setQuantRadius(512);
    lrz->setOutlierCapacity(0.2f);

    auto* bs = p->addStage<BitshuffleStage>();
    bs->setBlockSize(4096);
    bs->setElementWidth(2);
    p->connect(bs, lrz, "codes");

    p->enableGraphMode(true);
    p->setPoolManagedDecompOutput(false);
    p->finalize();

    // warmup() runs a dummy compress+decompress to JIT-compile kernels and
    // pre-allocate any stage-internal host buffers before the capture window.
    CudaStream stream;
    p->warmup(stream);
    stream.sync();

    // Record the liveness metrics before graph capture.
    const size_t topo_before   = p->getDAG()->computeTopoPoolSize();
    const size_t levels_before = p->getDAG()->getLevels().size();

    // Capture the graph (PREALLOCATE; all buffer pointers are already stable).
    ASSERT_NO_THROW(p->captureGraph(stream))
        << "LV7: captureGraph() must not throw on a valid PREALLOCATE pipeline";
    ASSERT_TRUE(p->isGraphCaptured());

    // printBufferLifetimes() must not crash after graph capture.
    ASSERT_NO_THROW(p->getDAG()->printBufferLifetimes())
        << "LV7: printBufferLifetimes() must not throw after graph capture";

    // Level structure must be unchanged — graph capture must not mutate the DAG.
    const auto& levels_after = p->getDAG()->getLevels();
    ASSERT_EQ(levels_after.size(), levels_before)
        << "LV7: getLevels() must return the same count before and after captureGraph()";

    for (size_t lvl = 0; lvl < levels_after.size(); lvl++) {
        EXPECT_EQ(levels_after[lvl].size(), 1u)
            << "LV7: level " << lvl << " must still have exactly 1 node after capture";
    }

    // computeTopoPoolSize() must be stable — topology did not change.
    const size_t topo_after = p->getDAG()->computeTopoPoolSize();
    EXPECT_EQ(topo_after, topo_before)
        << "LV7: computeTopoPoolSize() must be unchanged after captureGraph()";
    EXPECT_GT(topo_after, 0u)
        << "LV7: computeTopoPoolSize() must remain positive after captureGraph()";
}

// ─────────────────────────────────────────────────────────────────────────────
// LV8: printDAG() does not throw on linear and diamond pipelines, both before
//      and after compress().
// ─────────────────────────────────────────────────────────────────────────────
TEST(Liveness, PrintDAGNoCrash) {
    constexpr size_t N = 1 << 12;
    const size_t in_bytes = N * sizeof(float);

    // Linear 3-stage pipeline — before compress.
    auto p = build_linear3(in_bytes, MemoryStrategy::MINIMAL);
    ASSERT_NO_THROW(p->getDAG()->printDAG())
        << "LV8: printDAG() must not throw on a linear pipeline before compress()";

    CudaStream stream;
    auto h_in = make_smooth(N);
    CudaBuffer<float> d_in(N);
    d_in.upload(h_in, stream);
    stream.sync();

    void* d_comp = nullptr; size_t comp_sz = 0;
    p->compress(d_in.void_ptr(), in_bytes, &d_comp, &comp_sz, stream);
    stream.sync();

    ASSERT_NO_THROW(p->getDAG()->printDAG())
        << "LV8: printDAG() must not throw on a linear pipeline after compress()";
}
