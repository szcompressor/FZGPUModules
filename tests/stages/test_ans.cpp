#include <gtest/gtest.h>
#include <cstdint>
#include <vector>

#include "coders/ans/ans_stage.h"
#include "helpers/stage_harness.h"
#include "helpers/fz_test_utils.h"

using namespace fz;
using namespace fz_test;

// ── RoundTrip ────────────────────────────────────────────────────────────────
// Forward compression followed by inverse decompression produces correct output.
// Uses a single Pipeline instance — compress() and decompress() must share state.
TEST(ANSStage, RoundTrip) {
    const size_t N = 4096;
    const size_t in_bytes = N * sizeof(float);
    auto h_input = make_smooth_data<float>(N);

    Pipeline p(in_bytes, MemoryStrategy::PREALLOCATE);
    auto* stage = p.addStage<ANSStage>();
    // TODO: set stage parameters and connect if not the first stage
    p.finalize();

    CudaStream cs;
    auto res = pipeline_round_trip<float>(p, h_input, cs.stream);

    // TODO: adjust tolerance or use EXPECT_EQ for lossless stages
    EXPECT_LT(res.max_error, 1e-4f);
}

// ── ZeroInput ─────────────────────────────────────────────────────────────────
TEST(ANSStage, ZeroInput) {
    Pipeline p(0, MemoryStrategy::PREALLOCATE);
    p.addStage<ANSStage>();
    p.finalize();

    CudaStream cs;
    std::vector<float> empty;
    EXPECT_NO_THROW({
        auto res = pipeline_round_trip<float>(p, empty, cs.stream);
    });
}

// ── SerializeDeserialize ──────────────────────────────────────────────────────
TEST(ANSStage, SerializeDeserialize) {
    ANSStage original;
    // TODO: set parameters on original

    uint8_t buf[128] = {};
    size_t written = original.serializeHeader(0, buf, sizeof(buf));

    ANSStage restored;
    restored.deserializeHeader(buf, written);

    // TODO: EXPECT_EQ the relevant config fields between original and restored
    // e.g. EXPECT_EQ(original.getFoo(), restored.getFoo());
    SUCCEED(); // replace with real assertions
}

// ── SaveRestoreState ──────────────────────────────────────────────────────────
// Only needed if deserializeHeader() overwrites fields used by forward passes.
// The pipeline calls saveState() before and restoreState() after each decompress.
TEST(ANSStage, SaveRestoreState) {
    ANSStage s;
    // TODO: set a parameter, saveState, call deserializeHeader with different
    // bytes, restoreState, verify the original value is back.
    SUCCEED(); // replace with real assertions
}

// ── PipelineIntegration ───────────────────────────────────────────────────────
// Wires the stage into a full pipeline and verifies end-to-end round-trip.
// IMPORTANT: use one Pipeline instance — decompress() builds the inverse DAG
// from the same object that ran compress(). Two separate pipelines will throw.
TEST(ANSStage, PipelineIntegration) {
    const size_t N = 4096;
    const size_t in_bytes = N * sizeof(float);
    auto h_input = make_smooth_data<float>(N);

    Pipeline p(in_bytes, MemoryStrategy::PREALLOCATE);
    p.addStage<ANSStage>();
    // TODO: add other stages, connect ports
    p.finalize();

    CudaStream cs;
    auto res = pipeline_round_trip<float>(p, h_input, cs.stream);

    EXPECT_LT(res.max_error, 1e-4f);
    // TODO: EXPECT_LT(res.compressed_bytes, in_bytes) if stage is compressive
}

// ── GraphCompatible ───────────────────────────────────────────────────────────
TEST(ANSStage, GraphCompatible) {
    ANSStage stage;
    // TODO: change expected value to false if execute() does D2H transfers
    EXPECT_TRUE(stage.isGraphCompatible());
}
