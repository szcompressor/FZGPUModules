/**
 * tests/test_pipeline_errors.cpp
 *
 * Tests that the Pipeline API rejects invalid usage with clear exceptions.
 *
 * All operations that are documented as "must be called before finalize()"
 * are verified to throw std::runtime_error when called after finalize().
 * Operations that are permanently illegal (e.g., finalize twice) are also
 * verified.
 */

#include <gtest/gtest.h>
#include "helpers/fz_test_utils.h"
#include "fzgpumodules.h"

#include <stdexcept>

using namespace fz;
using namespace fz_test;

// ─────────────────────────────────────────────────────────────────────────────
// E1: addStage() after finalize() throws
// ─────────────────────────────────────────────────────────────────────────────
TEST(PipelineErrors, AddStageAfterFinalize) {
    Pipeline p(1024 * sizeof(float), MemoryStrategy::MINIMAL);
    auto* lrz = p.addStage<LorenzoStage<float, uint16_t>>();
    lrz->setErrorBound(1e-2f);
    p.finalize();

    EXPECT_THROW(
        (p.addStage<LorenzoStage<float, uint16_t>>()),
        std::runtime_error
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// E2: connect() after finalize() throws
//
// Two stages are added but NOT connected before finalize() so that both
// become independent sources.  The subsequent connect() call after finalize
// must throw regardless of whether the stages could form a valid graph.
// ─────────────────────────────────────────────────────────────────────────────
TEST(PipelineErrors, ConnectAfterFinalize) {
    Pipeline p(1024 * sizeof(float), MemoryStrategy::MINIMAL);
    auto* lrz  = p.addStage<LorenzoStage<float, uint16_t>>();
    auto* diff = p.addStage<DifferenceStage<uint16_t>>();
    lrz->setErrorBound(1e-2f);
    // Finalize with two unconnected stages (both become independent sources)
    p.finalize();

    EXPECT_THROW(
        (p.connect(diff, lrz, "codes")),
        std::runtime_error
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// E3: finalize() called twice throws
// ─────────────────────────────────────────────────────────────────────────────
TEST(PipelineErrors, FinalizeTwice) {
    Pipeline p(1024 * sizeof(float), MemoryStrategy::MINIMAL);
    auto* lrz = p.addStage<LorenzoStage<float, uint16_t>>();
    lrz->setErrorBound(1e-2f);
    p.finalize();

    EXPECT_THROW(p.finalize(), std::runtime_error);
}

// ─────────────────────────────────────────────────────────────────────────────
// E10 / M7: setMemoryStrategy() after finalize() throws
// ─────────────────────────────────────────────────────────────────────────────
TEST(PipelineErrors, SetStrategyAfterFinalize) {
    Pipeline p(1024 * sizeof(float), MemoryStrategy::MINIMAL);
    auto* lrz = p.addStage<LorenzoStage<float, uint16_t>>();
    lrz->setErrorBound(1e-2f);
    p.finalize();

    EXPECT_THROW(
        p.setMemoryStrategy(MemoryStrategy::PREALLOCATE),
        std::runtime_error
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// E4: compress() before finalize() throws
// ─────────────────────────────────────────────────────────────────────────────
TEST(PipelineErrors, CompressBeforeFinalize) {
    Pipeline p(1024 * sizeof(float), MemoryStrategy::MINIMAL);
    auto* lrz = p.addStage<LorenzoStage<float, uint16_t>>();
    lrz->setErrorBound(1e-2f);
    // Deliberately NOT calling p.finalize()

    CudaStream stream;
    std::vector<float> data(1024, 1.0f);
    CudaBuffer<float> d_in(1024);
    d_in.upload(data, stream);
    stream.sync();

    void*  d_comp  = nullptr;
    size_t comp_sz = 0;
    EXPECT_THROW(
        p.compress(d_in.void_ptr(), 1024 * sizeof(float), &d_comp, &comp_sz, stream),
        std::runtime_error
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// E5: decompress() before compress() throws
//
// The pipeline is properly finalized but compress() was never called, so
// buffer_metadata_ is empty and decompressMulti() will throw.
// ─────────────────────────────────────────────────────────────────────────────
TEST(PipelineErrors, DecompressBeforeCompress) {
    Pipeline p(1024 * sizeof(float), MemoryStrategy::MINIMAL);
    auto* lrz = p.addStage<LorenzoStage<float, uint16_t>>();
    lrz->setErrorBound(1e-2f);
    p.finalize();

    CudaStream stream;
    void*  d_decomp  = nullptr;
    size_t decomp_sz = 0;
    EXPECT_THROW(
        p.decompress(nullptr, 0, &d_decomp, &decomp_sz, stream),
        std::runtime_error
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// E8: connect() with an invalid output_name throws
// ─────────────────────────────────────────────────────────────────────────────
TEST(PipelineErrors, ConnectWithInvalidOutputName) {
    Pipeline p(1024 * sizeof(float), MemoryStrategy::MINIMAL);
    auto* lrz  = p.addStage<LorenzoStage<float, uint16_t>>();
    lrz->setErrorBound(1e-2f);
    auto* diff = p.addStage<DifferenceStage<uint16_t>>();

    // "nonexistent" is not a valid output of LorenzoStage
    EXPECT_THROW(
        (p.connect(diff, lrz, "nonexistent")),
        std::runtime_error
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Sanity: a correctly built pipeline throws nothing during construction
// ─────────────────────────────────────────────────────────────────────────────
TEST(PipelineErrors, ValidPipelineDoesNotThrow) {
    auto build = []() {
        Pipeline p(4096 * sizeof(float), MemoryStrategy::MINIMAL);
        auto* lrz = p.addStage<LorenzoStage<float, uint16_t>>();
        lrz->setErrorBound(1e-2f);
        lrz->setQuantRadius(512);
        p.finalize();
    };
    ASSERT_NO_THROW(build());
}

// ─────────────────────────────────────────────────────────────────────────────
// E6: Cyclic connection (A → B → A) detected at finalize() → throws
//
// Two DifferenceStage instances are connected in a cycle:
//   diff2 depends on diff1  (connect(diff2, diff1))
//   diff1 depends on diff2  (connect(diff1, diff2))
// DAG::finalize() runs Kahn's topological sort; a back-edge means not all
// nodes are visited, so it throws "Cyclic dependency detected".
// ─────────────────────────────────────────────────────────────────────────────
TEST(PipelineErrors, CyclicConnectionThrows) {
    Pipeline p(1024 * sizeof(float), MemoryStrategy::MINIMAL);
    auto* diff1 = p.addStage<DifferenceStage<uint16_t>>();
    auto* diff2 = p.addStage<DifferenceStage<uint16_t>>();
    // diff2 depends on diff1
    p.connect(diff2, diff1, "output");
    // diff1 depends on diff2 — creates a cycle
    p.connect(diff1, diff2, "output");

    EXPECT_THROW(p.finalize(), std::runtime_error)
        << "Cyclic pipeline should throw at finalize()";
}

// ─────────────────────────────────────────────────────────────────────────────
// E7: Stage added to a connected pipeline but never wired up → throws
//
// Three stages are added: lrz and diff are connected via connect(), but
// orphan is never involved in any connect() call.  finalize() should detect
// the isolated stage and raise a clear error.
// ─────────────────────────────────────────────────────────────────────────────
TEST(PipelineErrors, DisconnectedStageThrows) {
    Pipeline p(1024 * sizeof(float), MemoryStrategy::MINIMAL);
    auto* lrz    = p.addStage<LorenzoStage<float, uint16_t>>();
    auto* diff   = p.addStage<DifferenceStage<uint16_t>>();
    auto* orphan = p.addStage<LorenzoStage<float, uint16_t>>();  // never connected
    lrz->setErrorBound(1e-2f);
    orphan->setErrorBound(1e-2f);
    p.connect(diff, lrz, "codes");  // lrz and diff are connected; orphan is not

    EXPECT_THROW(p.finalize(), std::runtime_error)
        << "Stage with no connections in a connected pipeline should throw";
}

// ─────────────────────────────────────────────────────────────────────────────
// FE6: writeToFile() called before compress() → throws
//
// The pipeline is correctly finalized but compress() was never called.
// writeToFile() must detect this and raise a clear runtime_error.
// ─────────────────────────────────────────────────────────────────────────────
TEST(PipelineErrors, WriteToFileBeforeCompressThrows) {
    Pipeline p(1024 * sizeof(float), MemoryStrategy::MINIMAL);
    auto* lrz = p.addStage<LorenzoStage<float, uint16_t>>();
    lrz->setErrorBound(1e-2f);
    p.finalize();

    CudaStream stream;
    EXPECT_THROW(
        p.writeToFile("/tmp/fzgmod_test_fe6_no_compress.fzm", stream),
        std::runtime_error
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// RC4: compress() → decompress() → compress() without explicit reset()
//
// The second compress() must implicitly reset the DAG so that the
// remaining_consumers counters don't underflow and the result is correct.
// This is the "implicit reset" behaviour described in RC4/RC5.
// ─────────────────────────────────────────────────────────────────────────────
TEST(PipelineErrors, CompressDecompressCompressNoReset) {
    CudaStream stream;
    constexpr size_t N        = 1024;
    constexpr float  EB       = 1e-2f;
    const size_t     in_bytes = N * sizeof(float);

    auto h_input1 = make_random_floats(N, 42);
    auto h_input2 = make_random_floats(N, 99);

    CudaBuffer<float> d_in1(N), d_in2(N);
    d_in1.upload(h_input1, stream);
    d_in2.upload(h_input2, stream);
    stream.sync();

    Pipeline p(in_bytes, MemoryStrategy::MINIMAL);
    auto* lrz = p.addStage<LorenzoStage<float, uint16_t>>();
    lrz->setErrorBound(EB);
    p.finalize();

    // --- First compress + decompress ---
    void*  d_comp  = nullptr;
    size_t comp_sz = 0;
    ASSERT_NO_THROW(
        p.compress(d_in1.void_ptr(), in_bytes, &d_comp, &comp_sz, stream));
    ASSERT_GT(comp_sz, 0u);

    void*  d_dec  = nullptr;
    size_t dec_sz = 0;
    ASSERT_NO_THROW(
        p.decompress(nullptr, comp_sz, &d_dec, &dec_sz, stream));
    ASSERT_NE(d_dec, nullptr);
    cudaFree(d_dec);

    // --- Second compress WITHOUT explicit reset() — implicit reset (RC4) ---
    d_comp  = nullptr;
    comp_sz = 0;
    ASSERT_NO_THROW(
        p.compress(d_in2.void_ptr(), in_bytes, &d_comp, &comp_sz, stream))
        << "Second compress() without reset() should succeed via implicit reset";
    ASSERT_GT(comp_sz, 0u);

    // --- Decompress second result and verify against h_input2 ---
    d_dec  = nullptr;
    dec_sz = 0;
    ASSERT_NO_THROW(
        p.decompress(nullptr, comp_sz, &d_dec, &dec_sz, stream));
    ASSERT_NE(d_dec, nullptr);
    ASSERT_EQ(dec_sz, in_bytes);

    std::vector<float> h_recon(N);
    FZ_TEST_CUDA(
        cudaMemcpy(h_recon.data(), d_dec, in_bytes, cudaMemcpyDeviceToHost));
    cudaFree(d_dec);

    float max_err = max_abs_error(h_input2, h_recon);
    EXPECT_LE(max_err, EB * 1.01f)
        << "RC4: second-compress result max_err=" << max_err
        << " exceeds bound " << EB;
}
