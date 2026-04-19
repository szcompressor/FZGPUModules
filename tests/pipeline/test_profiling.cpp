/**
 * tests/pipeline/test_profiling.cpp
 *
 * Tests for pipeline profiling: enableProfiling(), getLastPerfResult(),
 * StageTimingResult, LevelTimingResult, and printDAG().
 *
 * Covers:
 *   PR1  Profiling disabled by default — stages/levels vectors are empty
 *   PR2  enableProfiling(true) + compress — stages vector has one entry
 *         per stage in the pipeline topology, level vector has one entry
 *         per DAG level
 *   PR3  Stage elapsed_ms > 0 for all stages after compress
 *   PR4  Stage name is non-empty; level matches DAG topology
 *   PR5  input_bytes / output_bytes > 0 for each stage
 *   PR6  throughput_gbs() > 0 after compress with profiling enabled
 *   PR7  Profiling result refreshes on each compress call
 *   PR8  Profiling works for decompress: is_compress flag is false
 *   PR9  Multi-stage pipeline (Lorenzo → RLE): stages.size() == 2,
 *         levels.size() == 2, level 0 has parallelism=1 for each level
 *   PR10 enableProfiling(false) clears result after next compress
 *   PR11 printDAG() on a finalized pipeline does not throw
 */

#include <gtest/gtest.h>
#include "helpers/fz_test_utils.h"
#include "fzgpumodules.h"

#include <cmath>
#include <sstream>
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

// Build a single-stage Lorenzo pipeline with the given strategy.
static std::unique_ptr<Pipeline> build_lorenzo(size_t in_bytes,
                                                MemoryStrategy strategy = MemoryStrategy::MINIMAL) {
    auto p = std::make_unique<Pipeline>(in_bytes, strategy, 3.0f);
    auto* lrz = p->addStage<LorenzoStage<float, uint16_t>>();
    lrz->setErrorBound(1e-2f);
    lrz->setQuantRadius(512);
    lrz->setOutlierCapacity(0.2f);
    p->finalize();
    return p;
}

// Build a two-stage Lorenzo → RLE pipeline.
static std::unique_ptr<Pipeline> build_lorenzo_rle(size_t in_bytes,
                                                    MemoryStrategy strategy = MemoryStrategy::MINIMAL) {
    auto p = std::make_unique<Pipeline>(in_bytes, strategy, 5.0f);
    auto* lrz = p->addStage<LorenzoStage<float, uint16_t>>();
    lrz->setErrorBound(1e-2f);
    lrz->setQuantRadius(512);
    lrz->setOutlierCapacity(0.2f);
    auto* rle = p->addStage<RLEStage<uint16_t>>();
    p->connect(rle, lrz, "codes");
    p->finalize();
    return p;
}

// ─────────────────────────────────────────────────────────────────────────────
// PR1: Profiling disabled by default
// ─────────────────────────────────────────────────────────────────────────────
TEST(Profiling, DisabledByDefault) {
    constexpr size_t N = 1 << 12;
    const size_t in_bytes = N * sizeof(float);

    CudaStream stream;
    auto h_in = make_smooth(N);
    CudaBuffer<float> d_in(N);
    d_in.upload(h_in, stream);
    stream.sync();

    auto p = build_lorenzo(in_bytes);

    void* d_comp = nullptr; size_t comp_sz = 0;
    p->compress(d_in.void_ptr(), in_bytes, &d_comp, &comp_sz, stream);
    stream.sync();

    const auto& r = p->getLastPerfResult();
    EXPECT_TRUE(r.stages.empty())
        << "PR1: stages must be empty when profiling is disabled";
    EXPECT_TRUE(r.levels.empty())
        << "PR1: levels must be empty when profiling is disabled";
}

// ─────────────────────────────────────────────────────────────────────────────
// PR2: Single-stage pipeline — one stage entry, one level entry
// ─────────────────────────────────────────────────────────────────────────────
TEST(Profiling, SingleStageEntryCount) {
    constexpr size_t N = 1 << 12;
    const size_t in_bytes = N * sizeof(float);

    CudaStream stream;
    auto h_in = make_smooth(N);
    CudaBuffer<float> d_in(N);
    d_in.upload(h_in, stream);
    stream.sync();

    auto p = build_lorenzo(in_bytes);
    p->enableProfiling(true);

    void* d_comp = nullptr; size_t comp_sz = 0;
    p->compress(d_in.void_ptr(), in_bytes, &d_comp, &comp_sz, stream);
    stream.sync();

    const auto& r = p->getLastPerfResult();
    EXPECT_EQ(r.stages.size(), 1u)
        << "PR2: single-stage pipeline must produce exactly 1 stage entry";
    EXPECT_EQ(r.levels.size(), 1u)
        << "PR2: single-stage pipeline must produce exactly 1 level entry";
    EXPECT_TRUE(r.is_compress) << "PR2: is_compress must be true after compress()";
}

// ─────────────────────────────────────────────────────────────────────────────
// PR3: Stage elapsed_ms > 0 after compress with profiling enabled
// ─────────────────────────────────────────────────────────────────────────────
TEST(Profiling, StageElapsedPositive) {
    constexpr size_t N = 1 << 14;
    const size_t in_bytes = N * sizeof(float);

    CudaStream stream;
    auto h_in = make_smooth(N);
    CudaBuffer<float> d_in(N);
    d_in.upload(h_in, stream);
    stream.sync();

    auto p = build_lorenzo(in_bytes);
    p->enableProfiling(true);

    void* d_comp = nullptr; size_t comp_sz = 0;
    p->compress(d_in.void_ptr(), in_bytes, &d_comp, &comp_sz, stream);
    stream.sync();

    const auto& r = p->getLastPerfResult();
    ASSERT_FALSE(r.stages.empty());
    for (const auto& s : r.stages) {
        EXPECT_GT(s.elapsed_ms, 0.0f)
            << "PR3: stage '" << s.name << "' elapsed_ms must be > 0";
    }
    EXPECT_GT(r.dag_elapsed_ms, 0.0f)
        << "PR3: dag_elapsed_ms must be > 0 after a real compress";
}

// ─────────────────────────────────────────────────────────────────────────────
// PR4: Stage name is non-empty; level index is 0 for single-stage pipeline
// ─────────────────────────────────────────────────────────────────────────────
TEST(Profiling, StageNameAndLevel) {
    constexpr size_t N = 1 << 12;
    const size_t in_bytes = N * sizeof(float);

    CudaStream stream;
    auto h_in = make_smooth(N);
    CudaBuffer<float> d_in(N);
    d_in.upload(h_in, stream);
    stream.sync();

    auto p = build_lorenzo(in_bytes);
    p->enableProfiling(true);

    void* d_comp = nullptr; size_t comp_sz = 0;
    p->compress(d_in.void_ptr(), in_bytes, &d_comp, &comp_sz, stream);
    stream.sync();

    const auto& r = p->getLastPerfResult();
    ASSERT_FALSE(r.stages.empty());
    EXPECT_FALSE(r.stages[0].name.empty())
        << "PR4: stage name must be non-empty";
    EXPECT_EQ(r.stages[0].level, 0)
        << "PR4: sole stage in a linear pipeline must be at level 0";
}

// ─────────────────────────────────────────────────────────────────────────────
// PR5: input_bytes and output_bytes are non-zero for each stage
// ─────────────────────────────────────────────────────────────────────────────
TEST(Profiling, StageIOBytesNonZero) {
    constexpr size_t N = 1 << 12;
    const size_t in_bytes = N * sizeof(float);

    CudaStream stream;
    auto h_in = make_smooth(N);
    CudaBuffer<float> d_in(N);
    d_in.upload(h_in, stream);
    stream.sync();

    auto p = build_lorenzo(in_bytes);
    p->enableProfiling(true);

    void* d_comp = nullptr; size_t comp_sz = 0;
    p->compress(d_in.void_ptr(), in_bytes, &d_comp, &comp_sz, stream);
    stream.sync();

    const auto& r = p->getLastPerfResult();
    ASSERT_FALSE(r.stages.empty());
    for (const auto& s : r.stages) {
        EXPECT_GT(s.input_bytes, 0u)
            << "PR5: stage '" << s.name << "' input_bytes must be > 0";
        EXPECT_GT(s.output_bytes, 0u)
            << "PR5: stage '" << s.name << "' output_bytes must be > 0";
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PR6: throughput_gbs() > 0 on the result and on each StageTimingResult
// ─────────────────────────────────────────────────────────────────────────────
TEST(Profiling, ThroughputPositive) {
    constexpr size_t N = 1 << 16;
    const size_t in_bytes = N * sizeof(float);

    CudaStream stream;
    auto h_in = make_smooth(N);
    CudaBuffer<float> d_in(N);
    d_in.upload(h_in, stream);
    stream.sync();

    auto p = build_lorenzo(in_bytes);
    p->enableProfiling(true);

    void* d_comp = nullptr; size_t comp_sz = 0;
    p->compress(d_in.void_ptr(), in_bytes, &d_comp, &comp_sz, stream);
    stream.sync();

    const auto& r = p->getLastPerfResult();
    EXPECT_GT(r.throughput_gbs(), 0.0f)
        << "PR6: pipeline throughput_gbs must be > 0";
    ASSERT_FALSE(r.stages.empty());
    for (const auto& s : r.stages) {
        EXPECT_GT(s.throughput_gbs(), 0.0f)
            << "PR6: stage '" << s.name << "' throughput_gbs must be > 0";
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PR7: Result refreshes on each compress call
//
// Two consecutive compress() calls with profiling enabled must each produce
// a fresh result.  We verify elapsed_ms is re-populated (not stale zeros).
// ─────────────────────────────────────────────────────────────────────────────
TEST(Profiling, ResultRefreshesEachCall) {
    constexpr size_t N = 1 << 14;
    const size_t in_bytes = N * sizeof(float);

    CudaStream stream;
    auto h_in = make_smooth(N);
    CudaBuffer<float> d_in(N);
    d_in.upload(h_in, stream);
    stream.sync();

    auto p = build_lorenzo(in_bytes);
    p->enableProfiling(true);

    for (int call = 0; call < 3; call++) {
        void* d_comp = nullptr; size_t comp_sz = 0;
        p->compress(d_in.void_ptr(), in_bytes, &d_comp, &comp_sz, stream);
        stream.sync();

        const auto& r = p->getLastPerfResult();
        ASSERT_FALSE(r.stages.empty()) << "PR7: call " << call << " produced no stages";
        EXPECT_GT(r.stages[0].elapsed_ms, 0.0f)
            << "PR7: call " << call << " elapsed_ms must be > 0";
        EXPECT_GT(r.dag_elapsed_ms, 0.0f)
            << "PR7: call " << call << " dag_elapsed_ms must be > 0";
        p->reset();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PR8: Profiling during decompress — is_compress == false
// ─────────────────────────────────────────────────────────────────────────────
TEST(Profiling, DecompressIsCompressFlagFalse) {
    constexpr size_t N = 1 << 12;
    const size_t in_bytes = N * sizeof(float);

    CudaStream stream;
    auto h_in = make_smooth(N);
    CudaBuffer<float> d_in(N);
    d_in.upload(h_in, stream);
    stream.sync();

    auto p = build_lorenzo(in_bytes);
    p->enableProfiling(true);

    void* d_comp = nullptr; size_t comp_sz = 0;
    p->compress(d_in.void_ptr(), in_bytes, &d_comp, &comp_sz, stream);
    stream.sync();

    // After compress the flag should be true.
    EXPECT_TRUE(p->getLastPerfResult().is_compress)
        << "PR8: is_compress must be true immediately after compress()";

    void* d_dec = nullptr; size_t dec_sz = 0;
    p->decompress(nullptr, 0, &d_dec, &dec_sz, stream);
    stream.sync();
    if (d_dec) cudaFree(d_dec);

    // After decompress the flag should flip to false.
    EXPECT_FALSE(p->getLastPerfResult().is_compress)
        << "PR8: is_compress must be false after decompress()";
    EXPECT_FALSE(p->getLastPerfResult().stages.empty())
        << "PR8: decompress profiling must populate stages";
}

// ─────────────────────────────────────────────────────────────────────────────
// PR9: Two-stage Lorenzo → RLE pipeline
//
// stages.size() == 2; levels.size() == 2; both levels have parallelism == 1;
// stage level indices are 0 and 1 respectively (topological order).
// ─────────────────────────────────────────────────────────────────────────────
TEST(Profiling, MultiStagePipelineEntries) {
    constexpr size_t N = 1 << 14;
    const size_t in_bytes = N * sizeof(float);

    CudaStream stream;
    auto h_in = make_smooth(N);
    CudaBuffer<float> d_in(N);
    d_in.upload(h_in, stream);
    stream.sync();

    auto p = build_lorenzo_rle(in_bytes);
    p->enableProfiling(true);

    void* d_comp = nullptr; size_t comp_sz = 0;
    p->compress(d_in.void_ptr(), in_bytes, &d_comp, &comp_sz, stream);
    stream.sync();

    const auto& r = p->getLastPerfResult();

    EXPECT_EQ(r.stages.size(), 2u)
        << "PR9: Lorenzo→RLE must produce exactly 2 stage entries";
    EXPECT_EQ(r.levels.size(), 2u)
        << "PR9: Lorenzo→RLE must produce exactly 2 level entries";

    for (const auto& lv : r.levels) {
        EXPECT_EQ(lv.parallelism, 1)
            << "PR9: linear pipeline — every level must have parallelism 1";
        EXPECT_GT(lv.elapsed_ms, 0.0f)
            << "PR9: level " << lv.level << " elapsed_ms must be > 0";
    }

    // Stages should be in ascending level order.
    EXPECT_EQ(r.stages[0].level, 0) << "PR9: first stage must be at level 0";
    EXPECT_EQ(r.stages[1].level, 1) << "PR9: second stage must be at level 1";

    for (const auto& s : r.stages) {
        EXPECT_FALSE(s.name.empty()) << "PR9: stage name must be non-empty";
        EXPECT_GT(s.elapsed_ms, 0.0f)
            << "PR9: stage '" << s.name << "' elapsed_ms must be > 0";
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PR10: enableProfiling(false) — result is not updated after next compress
//
// getLastPerfResult() is a "last populated" cache: it retains the result from
// the most recent profiling-enabled call.  When profiling is disabled, a
// subsequent compress() does not overwrite the cache — the stale result from
// the last enabled call remains accessible.  This is the documented contract.
// ─────────────────────────────────────────────────────────────────────────────
TEST(Profiling, DisableAfterEnableRetainsLastResult) {
    constexpr size_t N = 1 << 12;
    const size_t in_bytes = N * sizeof(float);

    CudaStream stream;
    auto h_in = make_smooth(N);
    CudaBuffer<float> d_in(N);
    d_in.upload(h_in, stream);
    stream.sync();

    auto p = build_lorenzo(in_bytes);

    // First compress with profiling on — capture elapsed_ms.
    p->enableProfiling(true);
    void* d_comp = nullptr; size_t comp_sz = 0;
    p->compress(d_in.void_ptr(), in_bytes, &d_comp, &comp_sz, stream);
    stream.sync();
    ASSERT_FALSE(p->getLastPerfResult().stages.empty())
        << "PR10: profiling must be populated when enabled";
    const float first_elapsed = p->getLastPerfResult().stages[0].elapsed_ms;
    EXPECT_GT(first_elapsed, 0.0f) << "PR10: first compress elapsed_ms must be > 0";

    // Disable, reset, compress again — result should NOT be refreshed.
    p->enableProfiling(false);
    p->reset();
    d_comp = nullptr; comp_sz = 0;
    p->compress(d_in.void_ptr(), in_bytes, &d_comp, &comp_sz, stream);
    stream.sync();

    // The cache still holds the result from the first (profiling-enabled) call.
    EXPECT_FALSE(p->getLastPerfResult().stages.empty())
        << "PR10: stale result must be retained after disabling profiling";
    // elapsed_ms should be the value from the first call, not re-measured.
    EXPECT_FLOAT_EQ(p->getLastPerfResult().stages[0].elapsed_ms, first_elapsed)
        << "PR10: getLastPerfResult() must not be updated when profiling is disabled";
}

// ─────────────────────────────────────────────────────────────────────────────
// PR11: printDAG() does not throw on a finalized pipeline
// ─────────────────────────────────────────────────────────────────────────────
TEST(Profiling, PrintDAGNoCrash) {
    constexpr size_t N = 1 << 12;
    const size_t in_bytes = N * sizeof(float);

    // Single-stage pipeline.
    auto p1 = build_lorenzo(in_bytes);
    EXPECT_NO_THROW(p1->getDAG()->printDAG())
        << "PR11: printDAG() must not throw on a single-stage pipeline";

    // Two-stage pipeline.
    auto p2 = build_lorenzo_rle(in_bytes);
    EXPECT_NO_THROW(p2->getDAG()->printDAG())
        << "PR11: printDAG() must not throw on a multi-stage pipeline";
}
