/**
 * tests/test_graph_capture.cpp
 *
 * Tests for CUDA Graph capture mode (compress path only).
 *
 * Covers:
 *   GC1  - captureGraph() produces correct compressed output
 *   GC2  - graph replay decompresses correctly across 12 iterations
 *   GC3  - graph decompressed output matches non-graph PREALLOCATE within EB
 *   GC4  - MINIMAL strategy + enableGraphMode() throws at finalize()
 *   GC5  - zero input size hint + enableGraphMode() throws at finalize()
 *   GC6  - captureGraph() before finalize() throws
 *   GC7  - compress() before captureGraph() falls back to normal exec (warmup path)
 *   GC8  - RZE inverse in pipeline → setCaptureMode(true) throws at capture time
 *   GC9  - isGraphCompatible() false for RZE inverse, true for all forward stages
 *   GC10 - graph mode with coloring disabled still produces correct output
 *   GC11 - warmup() before captureGraph() does not break subsequent graph compress
 *   GC12 - graph mode with mempool fallback throws at finalize()
 *   GC13 - captureGraph() without enableGraphMode() throws
 *   GC14 - captureGraph() after compress() throws
 *   GC15 - enableGraphMode() after finalize() throws
 *   GC16 - graph capture works with a Lorenzo -> RLE pipeline
 */

#include <gtest/gtest.h>
#include "helpers/fz_test_utils.h"
#include "fzgpumodules.h"

#include <cstdlib>
#include <string>
#include <vector>

using namespace fz;
using namespace fz_test;

// ─────────────────────────────────────────────────────────────────────────────
// Shared helpers
// ─────────────────────────────────────────────────────────────────────────────

struct ScopedEnvVar {
    std::string name;
    bool had_old = false;
    std::string old_value;

    ScopedEnvVar(const char* var_name, const char* value) : name(var_name) {
        const char* old = std::getenv(var_name);
        if (old) {
            had_old = true;
            old_value = old;
        }
        setenv(var_name, value, /*overwrite=*/1);
    }

    ~ScopedEnvVar() {
        if (had_old) {
            setenv(name.c_str(), old_value.c_str(), /*overwrite=*/1);
        } else {
            unsetenv(name.c_str());
        }
    }
};

// Build a finalized graph-mode pipeline with a single Lorenzo stage.
// Caller owns the pipeline; the graph stream must be non-default.
static std::unique_ptr<Pipeline> make_graph_pipeline(size_t in_bytes, bool disable_coloring = false) {
    auto p = std::make_unique<Pipeline>(in_bytes, MemoryStrategy::PREALLOCATE, 4.0f);
    auto* lrz = p->addStage<LorenzoQuantStage<float, uint16_t>>();
    lrz->setErrorBound(1e-2f);
    lrz->setQuantRadius(512);
    lrz->setOutlierCapacity(0.2f);
    if (disable_coloring) p->setColoringEnabled(false);
    p->enableGraphMode(true);
    p->setPoolManagedDecompOutput(false);
    p->finalize();
    return p;
}

static bool is_graph_supported() {
    Pipeline p(1024, MemoryStrategy::PREALLOCATE, 1.0f);
    return !p.isMemPoolFallbackMode();
}

static std::unique_ptr<Pipeline> make_graph_pipeline_with_rle(size_t in_bytes) {
    auto p = std::make_unique<Pipeline>(in_bytes, MemoryStrategy::PREALLOCATE, 4.0f);
    auto* lrz = p->addStage<LorenzoQuantStage<float, uint16_t>>();
    lrz->setErrorBound(1e-2f);
    lrz->setQuantRadius(512);
    lrz->setOutlierCapacity(0.2f);

    auto* rle = p->addStage<RLEStage<uint16_t>>();
    p->connect(rle, lrz, "codes");

    p->enableGraphMode(true);
    p->setPoolManagedDecompOutput(false);
    p->finalize();
    return p;
}

// ─────────────────────────────────────────────────────────────────────────────
// GC1: captureGraph() produces correct compressed + decompressed output
// ─────────────────────────────────────────────────────────────────────────────
TEST(GraphCapture, CorrectOutput) {
    if (!is_graph_supported()) {
        GTEST_SKIP() << "Graph mode unsupported in cudaMalloc fallback mode";
    }
    constexpr size_t N  = 1 << 14;
    constexpr float  EB = 1e-2f;
    const size_t in_bytes = N * sizeof(float);

    auto h_input = make_smooth_data<float>(N);
    CudaStream stream;
    CudaBuffer<float> d_in(N);
    d_in.upload(h_input, stream);
    stream.sync();

    auto p = make_graph_pipeline(in_bytes);
    p->captureGraph(stream);
    ASSERT_TRUE(p->isGraphCaptured());

    void*  d_comp  = nullptr;
    size_t comp_sz = 0;
    ASSERT_NO_THROW(p->compress(d_in.void_ptr(), in_bytes, &d_comp, &comp_sz, stream));
    stream.sync();
    EXPECT_GT(comp_sz, 0u);

    void*  d_dec  = nullptr;
    size_t dec_sz = 0;
    p->decompress(nullptr, 0, &d_dec, &dec_sz, stream);
    stream.sync();
    ASSERT_NE(d_dec, nullptr);
    ASSERT_EQ(dec_sz, in_bytes);

    std::vector<float> h_recon(N);
    cudaMemcpy(h_recon.data(), d_dec, in_bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_dec);

    float max_err = max_abs_error(h_input, h_recon);
    EXPECT_LE(max_err, EB * 1.01f)
        << "GC1: graph-mode round-trip max_err=" << max_err;
}

// ─────────────────────────────────────────────────────────────────────────────
// GC2: Graph replay produces correct, consistent decompressed output across 12
//      iterations — the reconstructed data must be within error bound every time
//      and the decompressed sizes must be stable.
//
// Note: we compare decompressed floats (not raw compressed bytes) because the
// concat buffer contains alignment padding bytes that are not overwritten between
// calls and may differ from a freshly-allocated buffer used on run 1.  The
// semantically meaningful invariant is data correctness, not byte identity.
// ─────────────────────────────────────────────────────────────────────────────
TEST(GraphCapture, ReplayCorrect) {
    if (!is_graph_supported()) {
        GTEST_SKIP() << "Graph mode unsupported in cudaMalloc fallback mode";
    }
    constexpr size_t N     = 1 << 14;
    constexpr float  EB    = 1e-2f;
    constexpr int    ITERS = 12;
    const size_t in_bytes  = N * sizeof(float);

    auto h_input = make_smooth_data<float>(N);
    CudaStream stream;
    CudaBuffer<float> d_in(N);
    d_in.upload(h_input, stream);
    stream.sync();

    auto p = make_graph_pipeline(in_bytes);
    p->captureGraph(stream);

    for (int i = 0; i < ITERS; i++) {
        void*  d_comp  = nullptr;
        size_t comp_sz = 0;
        p->compress(d_in.void_ptr(), in_bytes, &d_comp, &comp_sz, stream);
        stream.sync();
        ASSERT_GT(comp_sz, 0u) << "GC2: empty compressed output on iteration " << i;

        void*  d_dec  = nullptr;
        size_t dec_sz = 0;
        p->decompress(nullptr, 0, &d_dec, &dec_sz, stream);
        stream.sync();
        ASSERT_NE(d_dec, nullptr) << "GC2: null decompressed pointer on iteration " << i;
        ASSERT_EQ(dec_sz, in_bytes)  << "GC2: wrong decompressed size on iteration " << i;

        std::vector<float> h_recon(N);
        cudaMemcpy(h_recon.data(), d_dec, in_bytes, cudaMemcpyDeviceToHost);
        cudaFree(d_dec);

        float max_err = max_abs_error(h_input, h_recon);
        EXPECT_LE(max_err, EB * 1.01f)
            << "GC2: iteration " << i << " max_err=" << max_err;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GC3: Graph output decompresses to same data as non-graph PREALLOCATE.
//
// We compare decompressed floats rather than raw compressed bytes because the
// two pipelines use independently allocated pool buffers whose alignment-padding
// filler may differ.  The meaningful invariant is data correctness.
// ─────────────────────────────────────────────────────────────────────────────
TEST(GraphCapture, MatchesPreallocate) {
    if (!is_graph_supported()) {
        GTEST_SKIP() << "Graph mode unsupported in cudaMalloc fallback mode";
    }
    constexpr size_t N  = 1 << 14;
    constexpr float  EB = 1e-2f;
    const size_t in_bytes = N * sizeof(float);

    auto h_input = make_smooth_data<float>(N);
    CudaStream stream;
    CudaBuffer<float> d_in(N);
    d_in.upload(h_input, stream);
    stream.sync();

    // ── Non-graph PREALLOCATE reference ──────────────────────────────────────
    std::vector<float> h_ref;
    {
        Pipeline ref(in_bytes, MemoryStrategy::PREALLOCATE, 4.0f);
        auto* lrz = ref.addStage<LorenzoQuantStage<float, uint16_t>>();
        lrz->setErrorBound(EB);
        lrz->setQuantRadius(512);
        lrz->setOutlierCapacity(0.2f);
        ref.setPoolManagedDecompOutput(false);
    ref.finalize();

        void*  d_comp = nullptr;
        size_t comp_sz = 0;
        ref.compress(d_in.void_ptr(), in_bytes, &d_comp, &comp_sz, stream);
        stream.sync();
        ASSERT_GT(comp_sz, 0u);

        void*  d_dec = nullptr;
        size_t dec_sz = 0;
        ref.decompress(nullptr, 0, &d_dec, &dec_sz, stream);
        stream.sync();
        ASSERT_EQ(dec_sz, in_bytes);
        h_ref.resize(N);
        cudaMemcpy(h_ref.data(), d_dec, in_bytes, cudaMemcpyDeviceToHost);
        cudaFree(d_dec);
    }

    // ── Graph mode ───────────────────────────────────────────────────────────
    {
        auto p = make_graph_pipeline(in_bytes);
        p->captureGraph(stream);

        void*  d_comp = nullptr;
        size_t comp_sz = 0;
        p->compress(d_in.void_ptr(), in_bytes, &d_comp, &comp_sz, stream);
        stream.sync();
        ASSERT_GT(comp_sz, 0u);

        void*  d_dec = nullptr;
        size_t dec_sz = 0;
        p->decompress(nullptr, 0, &d_dec, &dec_sz, stream);
        stream.sync();
        ASSERT_EQ(dec_sz, in_bytes);

        std::vector<float> h_graph(N);
        cudaMemcpy(h_graph.data(), d_dec, in_bytes, cudaMemcpyDeviceToHost);
        cudaFree(d_dec);

        float max_err = max_abs_error(h_ref, h_graph);
        EXPECT_LE(max_err, EB * 1.01f)
            << "GC3: graph vs PREALLOCATE decompressed data differs; max_err=" << max_err;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GC4: MINIMAL strategy + enableGraphMode() throws at finalize()
// ─────────────────────────────────────────────────────────────────────────────
TEST(GraphCapture, MinimalStrategyThrowsAtFinalize) {
    constexpr size_t N = 1024;
    const size_t in_bytes = N * sizeof(float);

    Pipeline p(in_bytes, MemoryStrategy::MINIMAL);
    auto* lrz = p.addStage<LorenzoQuantStage<float, uint16_t>>();
    lrz->setErrorBound(1e-2f);
    p.enableGraphMode(true);

    EXPECT_THROW(p.finalize(), std::runtime_error)
        << "GC4: MINIMAL + graph mode must throw at finalize()";
}

// ─────────────────────────────────────────────────────────────────────────────
// GC5: Zero input size hint + enableGraphMode() throws at finalize()
// ─────────────────────────────────────────────────────────────────────────────
TEST(GraphCapture, ZeroInputHintThrowsAtFinalize) {
    Pipeline p(0, MemoryStrategy::PREALLOCATE);  // hint = 0
    auto* lrz = p.addStage<LorenzoQuantStage<float, uint16_t>>();
    lrz->setErrorBound(1e-2f);
    p.enableGraphMode(true);

    EXPECT_THROW(p.finalize(), std::runtime_error)
        << "GC5: zero input size hint + graph mode must throw at finalize()";
}

// ─────────────────────────────────────────────────────────────────────────────
// GC6: captureGraph() before finalize() throws
// ─────────────────────────────────────────────────────────────────────────────
TEST(GraphCapture, CaptureBeforeFinalizeThrows) {
    constexpr size_t N = 1024;
    Pipeline p(N * sizeof(float), MemoryStrategy::PREALLOCATE);
    auto* lrz = p.addStage<LorenzoQuantStage<float, uint16_t>>();
    lrz->setErrorBound(1e-2f);
    p.enableGraphMode(true);
    // Do NOT call finalize()

    CudaStream stream;
    EXPECT_THROW(p.captureGraph(stream), std::runtime_error)
        << "GC6: captureGraph() before finalize() must throw";
}

// ─────────────────────────────────────────────────────────────────────────────
// GC7: compress() before captureGraph() (graph mode enabled) falls back to
//      normal DAG execution — this is the warmup() path, not an error.
//
// warmup() internally calls compress() before captureGraph() to JIT-compile
// kernels.  The pipeline must handle this gracefully and produce valid output.
// ─────────────────────────────────────────────────────────────────────────────
TEST(GraphCapture, CompressBeforeCaptureSucceeds) {
    if (!is_graph_supported()) {
        GTEST_SKIP() << "Graph mode unsupported in cudaMalloc fallback mode";
    }
    constexpr size_t N  = 1 << 13;
    constexpr float  EB = 1e-2f;
    const size_t in_bytes = N * sizeof(float);

    auto h_input = make_smooth_data<float>(N);
    CudaStream stream;
    CudaBuffer<float> d_in(N);
    d_in.upload(h_input, stream);
    stream.sync();

    auto p = make_graph_pipeline(in_bytes);
    // graph mode enabled + finalized, but captureGraph() not yet called

    void*  d_comp  = nullptr;
    size_t comp_sz = 0;
    ASSERT_NO_THROW(p->compress(d_in.void_ptr(), in_bytes, &d_comp, &comp_sz, stream))
        << "GC7: compress() before captureGraph() must not throw (warmup path)";
    stream.sync();
    EXPECT_GT(comp_sz, 0u);

    // reset() clears was_compressed_ so captureGraph() can proceed.
    // This mirrors what warmup() does internally after its pre-capture compress.
    p->reset(stream);
    stream.sync();

    // Now capture and verify graph mode still works after the pre-capture compress.
    p->captureGraph(stream);
    ASSERT_TRUE(p->isGraphCaptured());

    void*  d_dec  = nullptr;
    size_t dec_sz = 0;
    p->decompress(nullptr, 0, &d_dec, &dec_sz, stream);
    stream.sync();
    ASSERT_NE(d_dec, nullptr);
    ASSERT_EQ(dec_sz, in_bytes);

    std::vector<float> h_recon(N);
    cudaMemcpy(h_recon.data(), d_dec, in_bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_dec);

    EXPECT_LE(max_abs_error(h_input, h_recon), EB * 1.01f)
        << "GC7: decompressed result after pre-capture compress exceeds error bound";
}

// ─────────────────────────────────────────────────────────────────────────────
// GC8: RZE inverse in pipeline → setCaptureMode(true) throws at capture time
//
// Building a pipeline with RZEStage in inverse mode and then calling
// captureGraph() must throw because RZE inverse is not graph-compatible.
// ─────────────────────────────────────────────────────────────────────────────
TEST(GraphCapture, RZEInverseIncompatibleThrowsAtCapture) {
    constexpr size_t N = 1 << 14;
    const size_t in_bytes = N * sizeof(float);

    // Build a forward pipeline first to get valid compressed data,
    // then test that a pipeline containing an inverse-mode RZE stage
    // throws when setCaptureMode(true) is called (via captureGraph).
    //
    // The simplest way: use DAG::setCaptureMode directly on an inv-RZE pipeline.
    // The Pipeline::captureGraph path validates via DAG::setCaptureMode(true).

    // Build a pipeline with an explicitly inverse RZE stage.
    Pipeline p(in_bytes, MemoryStrategy::PREALLOCATE, 4.0f);
    auto* rze = p.addStage<RZEStage>();
    rze->setInverse(true);
    rze->setChunkSize(16384);
    rze->setLevels(4);
    p.enableGraphMode(true);

    // finalize() itself should throw because setCaptureMode validation
    // happens there for graph mode — or if not, captureGraph() will throw.
    // Either way an exception must be raised before any GPU work.
    CudaStream stream;
    bool threw = false;
    try {
        p.setPoolManagedDecompOutput(false);
    p.finalize();
        p.captureGraph(stream);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    EXPECT_TRUE(threw)
        << "GC8: pipeline with inverse RZE stage must throw when graph capture is attempted";
}

// ─────────────────────────────────────────────────────────────────────────────
// GC9: isGraphCompatible() returns correct values for each stage type
// ─────────────────────────────────────────────────────────────────────────────
TEST(GraphCapture, IsGraphCompatiblePerStage) {
    // Forward stages — must all return true
    { LorenzoQuantStage<float, uint16_t> s; EXPECT_TRUE(s.isGraphCompatible())  << "Lorenzo forward"; }
    { RLEStage<uint16_t> s;            EXPECT_TRUE(s.isGraphCompatible())  << "RLE forward"; }
    { DifferenceStage<int16_t, uint16_t> s; EXPECT_TRUE(s.isGraphCompatible()) << "Diff forward"; }
    { ZigzagStage<int16_t, uint16_t> s; EXPECT_TRUE(s.isGraphCompatible()) << "Zigzag forward"; }
    { NegabinaryStage<int16_t, uint16_t> s; EXPECT_TRUE(s.isGraphCompatible()) << "Negabinary forward"; }
    { BitshuffleStage s;               EXPECT_TRUE(s.isGraphCompatible())  << "Bitshuffle forward"; }

    // RZE forward — must return true
    { RZEStage s; EXPECT_TRUE(s.isGraphCompatible()) << "RZE forward"; }

    // RZE inverse — must return false
    {
        RZEStage s;
        s.setInverse(true);
        EXPECT_FALSE(s.isGraphCompatible()) << "RZE inverse must not be graph-compatible";
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GC10: Graph mode with buffer coloring disabled still produces correct output
// ─────────────────────────────────────────────────────────────────────────────
TEST(GraphCapture, ColoringDisabledCorrectOutput) {
    if (!is_graph_supported()) {
        GTEST_SKIP() << "Graph mode unsupported in cudaMalloc fallback mode";
    }
    constexpr size_t N  = 1 << 13;
    constexpr float  EB = 1e-2f;
    const size_t in_bytes = N * sizeof(float);

    auto h_input = make_smooth_data<float>(N);
    CudaStream stream;
    CudaBuffer<float> d_in(N);
    d_in.upload(h_input, stream);
    stream.sync();

    auto p = make_graph_pipeline(in_bytes, /*disable_coloring=*/true);
    EXPECT_FALSE(p->isColoringEnabled())
        << "GC10: coloring should be disabled";

    p->captureGraph(stream);
    ASSERT_TRUE(p->isGraphCaptured());

    void*  d_comp  = nullptr;
    size_t comp_sz = 0;
    p->compress(d_in.void_ptr(), in_bytes, &d_comp, &comp_sz, stream);
    stream.sync();
    ASSERT_GT(comp_sz, 0u);

    void*  d_dec  = nullptr;
    size_t dec_sz = 0;
    p->decompress(nullptr, 0, &d_dec, &dec_sz, stream);
    stream.sync();
    ASSERT_NE(d_dec, nullptr);
    ASSERT_EQ(dec_sz, in_bytes);

    std::vector<float> h_recon(N);
    cudaMemcpy(h_recon.data(), d_dec, in_bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_dec);

    EXPECT_LE(max_abs_error(h_input, h_recon), EB * 1.01f)
        << "GC10: coloring-disabled graph round-trip error exceeds bound";
}

// ─────────────────────────────────────────────────────────────────────────────
// GC11: warmup() before captureGraph() does not break subsequent graph compress
//
// The example code pattern is: finalize → warmup → captureGraph → compress.
// warmup() resets the DAG back to post-finalize state; captureGraph() must
// still succeed and compress must produce correct output.
// ─────────────────────────────────────────────────────────────────────────────
TEST(GraphCapture, WarmupBeforeCaptureWorks) {
    if (!is_graph_supported()) {
        GTEST_SKIP() << "Graph mode unsupported in cudaMalloc fallback mode";
    }
    constexpr size_t N  = 1 << 14;
    constexpr float  EB = 1e-2f;
    const size_t in_bytes = N * sizeof(float);

    auto h_input = make_smooth_data<float>(N);
    CudaStream stream;
    CudaBuffer<float> d_in(N);
    d_in.upload(h_input, stream);
    stream.sync();

    auto p = make_graph_pipeline(in_bytes);

    // warmup() — JIT-compile kernels; must leave DAG in clean state
    ASSERT_NO_THROW(p->warmup(stream));

    // captureGraph() — must succeed after warmup
    ASSERT_NO_THROW(p->captureGraph(stream));
    ASSERT_TRUE(p->isGraphCaptured());

    // compress + decompress must be correct
    void*  d_comp  = nullptr;
    size_t comp_sz = 0;
    p->compress(d_in.void_ptr(), in_bytes, &d_comp, &comp_sz, stream);
    stream.sync();
    ASSERT_GT(comp_sz, 0u);

    void*  d_dec  = nullptr;
    size_t dec_sz = 0;
    p->decompress(nullptr, 0, &d_dec, &dec_sz, stream);
    stream.sync();
    ASSERT_NE(d_dec, nullptr);
    ASSERT_EQ(dec_sz, in_bytes);

    std::vector<float> h_recon(N);
    cudaMemcpy(h_recon.data(), d_dec, in_bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_dec);

    EXPECT_LE(max_abs_error(h_input, h_recon), EB * 1.01f)
        << "GC11: warmup→capture round-trip error exceeds bound";
}

// ─────────────────────────────────────────────────────────────────────────────
// GC12: Graph mode must throw when mempool fallback is active (vGPU / forced)
// ─────────────────────────────────────────────────────────────────────────────
TEST(GraphCapture, MempoolFallbackThrowsAtFinalize) {
    ScopedEnvVar force_fallback("FZ_FORCE_MEMPOOL_FALLBACK", "1");
    constexpr size_t N = 1024;
    const size_t in_bytes = N * sizeof(float);

    Pipeline p(in_bytes, MemoryStrategy::PREALLOCATE, 4.0f);
    auto* lrz = p.addStage<LorenzoQuantStage<float, uint16_t>>();
    lrz->setErrorBound(1e-2f);
    p.enableGraphMode(true);

    EXPECT_THROW(p.finalize(), std::runtime_error)
        << "GC12: graph mode must be rejected when mempool fallback is active";
}

// ─────────────────────────────────────────────────────────────────────────────
// GC13: captureGraph() without enableGraphMode() throws
// ─────────────────────────────────────────────────────────────────────────────
TEST(GraphCapture, CaptureWithoutEnableThrows) {
    constexpr size_t N = 1024;
    const size_t in_bytes = N * sizeof(float);

    Pipeline p(in_bytes, MemoryStrategy::PREALLOCATE, 4.0f);
    auto* lrz = p.addStage<LorenzoQuantStage<float, uint16_t>>();
    lrz->setErrorBound(1e-2f);
    p.finalize();

    CudaStream stream;
    EXPECT_THROW(p.captureGraph(stream), std::runtime_error)
        << "GC13: captureGraph() without enableGraphMode() must throw";
}

// ─────────────────────────────────────────────────────────────────────────────
// GC14: captureGraph() after compress() throws (must capture before first run)
// ─────────────────────────────────────────────────────────────────────────────
TEST(GraphCapture, CaptureAfterCompressThrows) {
    if (!is_graph_supported()) {
        GTEST_SKIP() << "Graph mode unsupported in cudaMalloc fallback mode";
    }
    constexpr size_t N  = 1 << 12;
    const size_t in_bytes = N * sizeof(float);

    auto h_input = make_smooth_data<float>(N);
    CudaStream stream;
    CudaBuffer<float> d_in(N);
    d_in.upload(h_input, stream);
    stream.sync();

    auto p = make_graph_pipeline(in_bytes);

    void*  d_comp  = nullptr;
    size_t comp_sz = 0;
    p->compress(d_in.void_ptr(), in_bytes, &d_comp, &comp_sz, stream);
    stream.sync();
    ASSERT_GT(comp_sz, 0u);

    EXPECT_THROW(p->captureGraph(stream), std::runtime_error)
        << "GC14: captureGraph() after compress() must throw";
}

// ─────────────────────────────────────────────────────────────────────────────
// GC15: enableGraphMode() after finalize() throws
// ─────────────────────────────────────────────────────────────────────────────
TEST(GraphCapture, EnableAfterFinalizeThrows) {
    constexpr size_t N = 1024;
    Pipeline p(N * sizeof(float), MemoryStrategy::PREALLOCATE, 4.0f);
    auto* lrz = p.addStage<LorenzoQuantStage<float, uint16_t>>();
    lrz->setErrorBound(1e-2f);
    p.finalize();

    EXPECT_THROW(p.enableGraphMode(true), std::runtime_error)
        << "GC15: enableGraphMode() after finalize() must throw";
}

// ─────────────────────────────────────────────────────────────────────────────
// GC16: Graph capture works for a Lorenzo → RLE pipeline
// ─────────────────────────────────────────────────────────────────────────────
TEST(GraphCapture, RlePipelineCorrectOutput) {
    if (!is_graph_supported()) {
        GTEST_SKIP() << "Graph mode unsupported in cudaMalloc fallback mode";
    }
    constexpr size_t N  = 1 << 14;
    constexpr float  EB = 1e-2f;
    const size_t in_bytes = N * sizeof(float);

    auto h_input = make_smooth_data<float>(N);
    CudaStream stream;
    CudaBuffer<float> d_in(N);
    d_in.upload(h_input, stream);
    stream.sync();

    auto p = make_graph_pipeline_with_rle(in_bytes);

    // Warmup to pre-allocate stage scratch and host buffers outside the capture window.
    p->warmup(stream);
    stream.sync();

    p->captureGraph(stream);
    ASSERT_TRUE(p->isGraphCaptured());

    void*  d_comp  = nullptr;
    size_t comp_sz = 0;
    p->compress(d_in.void_ptr(), in_bytes, &d_comp, &comp_sz, stream);
    stream.sync();
    ASSERT_GT(comp_sz, 0u);

    void*  d_dec  = nullptr;
    size_t dec_sz = 0;
    p->decompress(nullptr, 0, &d_dec, &dec_sz, stream);
    stream.sync();
    ASSERT_NE(d_dec, nullptr);
    ASSERT_EQ(dec_sz, in_bytes);

    std::vector<float> h_recon(N);
    cudaMemcpy(h_recon.data(), d_dec, in_bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_dec);

    float max_err = max_abs_error(h_input, h_recon);
    EXPECT_LE(max_err, EB * 1.01f)
        << "GC16: Lorenzo->RLE graph round-trip max_err=" << max_err;
}
