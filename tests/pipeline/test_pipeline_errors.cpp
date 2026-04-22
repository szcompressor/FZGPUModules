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

// ─────────────────────────────────────────────────────────────────────────────
// OverreportingStage — local test-only stage used by E11.
//
// estimateOutputSizes() returns N/2 so PREALLOCATE allocates a small buffer.
// execute() only writes N/2 bytes (safe), but getActualOutputSizesByName()
// lies and reports N bytes written.  This triggers the bounds-check throw
// without actually corrupting GPU memory.
// ─────────────────────────────────────────────────────────────────────────────
namespace {
class OverreportingStage : public fz::Stage {
public:
    OverreportingStage() : reported_size_(0) {}

    void execute(
        cudaStream_t stream, fz::MemoryPool*,
        const std::vector<void*>& inputs,
        const std::vector<void*>& outputs,
        const std::vector<size_t>& sizes
    ) override {
        cudaMemcpyAsync(outputs[0], inputs[0], sizes[0] / 2,
                        cudaMemcpyDeviceToDevice, stream);
        reported_size_ = sizes[0];  // lie: claim full input size was written
    }

    std::string getName()     const override { return "OverreportingStage"; }
    size_t getNumInputs()     const override { return 1; }
    size_t getNumOutputs()    const override { return 1; }

    std::vector<size_t> estimateOutputSizes(
        const std::vector<size_t>& in) const override { return {in[0] / 2}; }

    std::unordered_map<std::string, size_t>
    getActualOutputSizesByName() const override {
        return {{"output", reported_size_}};
    }

    uint16_t getStageTypeId() const override {
        return static_cast<uint16_t>(fz::StageType::PASSTHROUGH);
    }
    uint8_t getOutputDataType(size_t) const override {
        return static_cast<uint8_t>(fz::DataType::UINT8);
    }

private:
    size_t reported_size_;
};
} // namespace

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
    p.setPoolManagedDecompOutput(false);
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
    p.setPoolManagedDecompOutput(false);
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
    p.setPoolManagedDecompOutput(false);
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
    p.setPoolManagedDecompOutput(false);
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
// buffer_metadata_ is empty and decompress() will throw.
// ─────────────────────────────────────────────────────────────────────────────
TEST(PipelineErrors, DecompressBeforeCompress) {
    Pipeline p(1024 * sizeof(float), MemoryStrategy::MINIMAL);
    auto* lrz = p.addStage<LorenzoStage<float, uint16_t>>();
    lrz->setErrorBound(1e-2f);
    p.setPoolManagedDecompOutput(false);
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
        p.setPoolManagedDecompOutput(false);
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
    p.setPoolManagedDecompOutput(false);
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
    p.setPoolManagedDecompOutput(false);
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

// ─────────────────────────────────────────────────────────────────────────────
// E9: compress() with a null device pointer → throws
// ─────────────────────────────────────────────────────────────────────────────
TEST(PipelineErrors, NullInputPointerThrows) {
    constexpr size_t N        = 1024;
    const size_t     in_bytes = N * sizeof(float);

    Pipeline p(in_bytes, MemoryStrategy::MINIMAL);
    auto* lrz = p.addStage<LorenzoStage<float, uint16_t>>();
    lrz->setErrorBound(1e-2f);
    p.setPoolManagedDecompOutput(false);
    p.finalize();

    CudaStream stream;
    void*  d_out  = nullptr;
    size_t out_sz = 0;

    // nullptr device pointer — the guard in compress() must reject this before
    // anything hits the GPU.
    EXPECT_THROW(
        p.compress(nullptr, in_bytes, &d_out, &out_sz, stream),
        std::runtime_error
    ) << "Null device pointer should throw";
}

// ─────────────────────────────────────────────────────────────────────────────
// E10: compress() with data size larger than the finalize-time hint → throws
//
// The pipeline is built with a 1 KB hint, but compress() is called with 4 KB.
// The guard must detect that buffer estimates from finalize() are stale and
// reject the call before any GPU work begins.
// ─────────────────────────────────────────────────────────────────────────────
TEST(PipelineErrors, InputSizeExceedsHintThrows) {
    constexpr size_t HINT_BYTES  = 1024 * sizeof(float);
    constexpr size_t LARGE_BYTES = 4096 * sizeof(float);

    Pipeline p(HINT_BYTES, MemoryStrategy::MINIMAL);
    auto* lrz = p.addStage<LorenzoStage<float, uint16_t>>();
    lrz->setErrorBound(1e-2f);
    p.setPoolManagedDecompOutput(false);
    p.finalize();

    CudaStream stream;
    CudaBuffer<float> d_in(4096);
    {
        auto data = make_random_floats(4096, 7);
        d_in.upload(data, stream);
        stream.sync();
    }

    void*  d_out  = nullptr;
    size_t out_sz = 0;
    EXPECT_THROW(
        p.compress(d_in.void_ptr(), LARGE_BYTES, &d_out, &out_sz, stream),
        std::runtime_error
    ) << "Data size exceeding finalize-time hint should throw";
}

// ─────────────────────────────────────────────────────────────────────────────
// E11a: enableGraphMode() + MINIMAL strategy throws at finalize()
//
// Graph mode requires PREALLOCATE so all buffer pointers are stable before the
// graph is recorded. Using MINIMAL must throw a clear error at finalize() time,
// before any GPU work is attempted.
// ─────────────────────────────────────────────────────────────────────────────
TEST(PipelineErrors, GraphModeWithMinimalStrategyThrows) {
    Pipeline p(1024 * sizeof(float), MemoryStrategy::MINIMAL);
    auto* lrz = p.addStage<LorenzoStage<float, uint16_t>>();
    lrz->setErrorBound(1e-2f);
    p.enableGraphMode(true);

    EXPECT_THROW(p.finalize(), std::runtime_error)
        << "Graph mode with MINIMAL strategy must throw at finalize()";
}

// ─────────────────────────────────────────────────────────────────────────────
// ER1: Pipeline is still usable after compress() throws
//
// We force a throw by calling compress() with a null input pointer.
// After the throw the pipeline must be in a consistent state — a second
// compress() with valid inputs must succeed and produce correct output.
// This exercises the try/catch + dag_->reset() error-recovery path.
// ─────────────────────────────────────────────────────────────────────────────
TEST(PipelineErrors, PipelineUsableAfterCompressThrow) {
    constexpr size_t N        = 1024;
    constexpr float  EB       = 1e-2f;
    const size_t     in_bytes = N * sizeof(float);

    Pipeline p(in_bytes, MemoryStrategy::MINIMAL);
    auto* lrz = p.addStage<LorenzoStage<float, uint16_t>>();
    lrz->setErrorBound(EB);
    p.setPoolManagedDecompOutput(false);
    p.finalize();

    CudaStream stream;

    // First call: force a throw with a null pointer.
    void* d_out  = nullptr;
    size_t out_sz = 0;
    EXPECT_THROW(
        p.compress(nullptr, in_bytes, &d_out, &out_sz, stream),
        std::runtime_error);

    // Second call: valid input — must succeed.
    auto h_input = make_random_floats(N, 77);
    CudaBuffer<float> d_in(N);
    d_in.upload(h_input, stream);
    stream.sync();

    d_out  = nullptr;
    out_sz = 0;
    ASSERT_NO_THROW(
        p.compress(d_in.void_ptr(), in_bytes, &d_out, &out_sz, stream))
        << "Pipeline must be usable after a previous compress() throw";
    EXPECT_GT(out_sz, 0u);

    // Decompress and verify correctness — proves internal state is clean.
    void*  d_dec  = nullptr;
    size_t dec_sz = 0;
    ASSERT_NO_THROW(p.decompress(nullptr, out_sz, &d_dec, &dec_sz, stream));
    ASSERT_NE(d_dec, nullptr);
    ASSERT_EQ(dec_sz, in_bytes);

    std::vector<float> h_recon(N);
    FZ_TEST_CUDA(cudaMemcpy(h_recon.data(), d_dec, in_bytes, cudaMemcpyDeviceToHost));
    cudaFree(d_dec);

    EXPECT_LE(max_abs_error(h_input, h_recon), EB * 1.01f)
        << "Reconstructed data after error-recovery compress must satisfy error bound";
}

// ─────────────────────────────────────────────────────────────────────────────
// ER2: GPU memory is not leaked after a failed compress()
//
// Measure free GPU bytes before and after a compress() that throws.
// The pool must return to (approximately) the same level — no persistent
// allocation must remain from the failed call.
//
// Note: "approximately" is important. The CUDA memory allocator has internal
// fragmentation and rounding. We check that any residual is small (< 1 MB)
// rather than requiring an exact match, which would be fragile.
// ─────────────────────────────────────────────────────────────────────────────
TEST(PipelineErrors, NoMemoryLeakAfterCompressThrow) {
    constexpr size_t N        = 4096;
    const size_t     in_bytes = N * sizeof(float);

    Pipeline p(in_bytes, MemoryStrategy::MINIMAL);
    auto* lrz = p.addStage<LorenzoStage<float, uint16_t>>();
    lrz->setErrorBound(1e-2f);
    p.setPoolManagedDecompOutput(false);
    p.finalize();

    // Warm up CUDA runtime allocator so its own internal bookkeeping doesn't
    // skew the before/after comparison.
    {
        CudaStream stream;
        CudaBuffer<float> d_dummy(N);
        auto data = make_random_floats(N, 1);
        d_dummy.upload(data, stream);
        void* d_out = nullptr; size_t out_sz = 0;
        p.compress(d_dummy.void_ptr(), in_bytes, &d_out, &out_sz, stream);
        stream.sync();
        p.reset();
    }

    cudaDeviceSynchronize();
    const size_t free_before = gpu_free_bytes();

    // Trigger a failing compress() — null pointer throws immediately so no
    // GPU allocations should escape.
    {
        CudaStream stream;
        void* d_out = nullptr; size_t out_sz = 0;
        EXPECT_THROW(p.compress(nullptr, in_bytes, &d_out, &out_sz, stream),
                     std::runtime_error);
        stream.sync();
    }

    cudaDeviceSynchronize();
    const size_t free_after = gpu_free_bytes();

    // Allow up to 1 MB of allocator-internal variance.
    constexpr size_t TOLERANCE = 1024 * 1024;
    EXPECT_GE(free_after + TOLERANCE, free_before)
        << "GPU memory decreased by more than 1 MB after a failed compress() — "
           "possible leak. Before: " << free_before << " B, After: " << free_after << " B";
}

// ─────────────────────────────────────────────────────────────────────────────
// E11: Buffer overwrite detection via enableBoundsCheck()
//
// OverreportingStage deliberately underestimates output size in
// estimateOutputSizes() (returns N/2) but then reports N as its actual output
// size via getActualOutputSizesByName().  With PREALLOCATE strategy the output
// buffer is allocated at N/2 bytes upfront; when the stage claims to have
// written N bytes the bounds-check logic in CompressionDAG::execute() must
// throw "Buffer overwrite detected".
// ─────────────────────────────────────────────────────────────────────────────
TEST(PipelineErrors, BoundsCheckDetectsOverreport) {
    constexpr size_t N        = 512;  // elements — divisible so N/2 is exact
    const size_t     in_bytes = N * sizeof(float);

    Pipeline p(in_bytes, MemoryStrategy::PREALLOCATE);
    p.addStage<OverreportingStage>();
    p.enableBoundsCheck(true);
    p.setPoolManagedDecompOutput(false);
    p.finalize();

    CudaStream stream;
    CudaBuffer<float> d_in(N);
    {
        auto data = make_random_floats(N, 13);
        d_in.upload(data, stream);
        stream.sync();
    }

    void*  d_out  = nullptr;
    size_t out_sz = 0;
    EXPECT_THROW(
        p.compress(d_in.void_ptr(), in_bytes, &d_out, &out_sz, stream),
        std::runtime_error
    ) << "OverreportingStage should trigger bounds-check throw";
}

// ─────────────────────────────────────────────────────────────────────────────
// ER2: Pipeline is still usable after decompress() throws
//
// decompress() before compress() throws (no buffer_metadata_).  After the
// throw, the pipeline must recover cleanly: a full compress → decompress cycle
// must produce correct output.  This exercises the inv_dag error-recovery path.
// ─────────────────────────────────────────────────────────────────────────────
TEST(PipelineErrors, PipelineUsableAfterDecompressThrow) {
    constexpr size_t N        = 1024;
    constexpr float  EB       = 1e-2f;
    const size_t     in_bytes = N * sizeof(float);

    auto h_in = make_random_floats(N, 99);

    Pipeline p(in_bytes, MemoryStrategy::MINIMAL);
    auto* lrz = p.addStage<LorenzoStage<float, uint16_t>>();
    lrz->setErrorBound(EB);
    p.setPoolManagedDecompOutput(false);
    p.finalize();

    CudaStream stream;

    // Force decompress() to throw — compress() was never called.
    {
        void*  d_out  = nullptr;
        size_t out_sz = 0;
        EXPECT_THROW(
            p.decompress(nullptr, 0, &d_out, &out_sz, stream),
            std::runtime_error
        ) << "decompress() before compress() must throw";
    }

    // Now run a full compress → decompress cycle — pipeline must be clean.
    CudaBuffer<float> d_in(N);
    d_in.upload(h_in, stream);
    stream.sync();

    void*  d_comp  = nullptr;
    size_t comp_sz = 0;
    ASSERT_NO_THROW(
        p.compress(d_in.void_ptr(), in_bytes, &d_comp, &comp_sz, stream)
    ) << "compress() after failed decompress() must not throw";
    stream.sync();
    ASSERT_GT(comp_sz, 0u);

    void*  d_dec  = nullptr;
    size_t dec_sz = 0;
    ASSERT_NO_THROW(
        p.decompress(nullptr, 0, &d_dec, &dec_sz, stream)
    ) << "decompress() after valid compress() must not throw";
    stream.sync();
    ASSERT_NE(d_dec, nullptr);
    ASSERT_EQ(dec_sz, in_bytes);

    std::vector<float> h_recon(N);
    FZ_TEST_CUDA(cudaMemcpy(h_recon.data(), d_dec, in_bytes, cudaMemcpyDeviceToHost));
    cudaFree(d_dec);

    EXPECT_LE(max_abs_error(h_in, h_recon), EB * 1.01f)
        << "Round-trip after failed decompress() must be data-correct";
}

// ─────────────────────────────────────────────────────────────────────────────
// ER3: PREALLOCATE strategy — pipeline still usable after compress() throws.
//
// PREALLOCATE allocates all stage buffers at finalize() time.  A compress()
// that throws due to a null input pointer must leave those pre-allocated
// buffers intact and the DAG properly reset.  A subsequent valid
// compress() + decompress() must produce correct output, proving the recovery
// path (dag_->reset()) does not corrupt or double-free PREALLOCATE buffers.
// ─────────────────────────────────────────────────────────────────────────────
TEST(PipelineErrors, PipelineUsableAfterCompressThrowPreallocate) {
    constexpr size_t N        = 1024;
    constexpr float  EB       = 1e-2f;
    const size_t     in_bytes = N * sizeof(float);

    Pipeline p(in_bytes, MemoryStrategy::PREALLOCATE);
    auto* lrz = p.addStage<LorenzoStage<float, uint16_t>>();
    lrz->setErrorBound(EB);
    p.setPoolManagedDecompOutput(false);
    p.finalize();

    CudaStream stream;

    // Force a throw: null device pointer fails before GPU work starts.
    void*  d_out  = nullptr;
    size_t out_sz = 0;
    EXPECT_THROW(
        p.compress(nullptr, in_bytes, &d_out, &out_sz, stream),
        std::runtime_error
    ) << "Null pointer with PREALLOCATE must throw";

    // Recovery: the PREALLOCATE buffers must still be valid after the throw.
    auto h_input = make_random_floats(N, 111);
    CudaBuffer<float> d_in(N);
    d_in.upload(h_input, stream);
    stream.sync();

    d_out  = nullptr;
    out_sz = 0;
    ASSERT_NO_THROW(
        p.compress(d_in.void_ptr(), in_bytes, &d_out, &out_sz, stream))
        << "PREALLOCATE pipeline must be usable after a previous compress() throw";
    ASSERT_GT(out_sz, 0u);

    // Full decompress + verify to confirm internal state is clean.
    void*  d_dec  = nullptr;
    size_t dec_sz = 0;
    ASSERT_NO_THROW(p.decompress(nullptr, out_sz, &d_dec, &dec_sz, stream));
    ASSERT_NE(d_dec, nullptr);
    ASSERT_EQ(dec_sz, in_bytes);
    stream.sync();

    std::vector<float> h_recon(N);
    FZ_TEST_CUDA(cudaMemcpy(h_recon.data(), d_dec, in_bytes, cudaMemcpyDeviceToHost));
    cudaFree(d_dec);

    EXPECT_LE(max_abs_error(h_input, h_recon), EB * 1.01f)
        << "Data must be correct after PREALLOCATE error-recovery compress";
}

// ─────────────────────────────────────────────────────────────────────────────
// ER4: Multiple consecutive compress() failures do not accumulate GPU memory.
//
// Three null-pointer compress() calls in a row each throw.  GPU free bytes
// after all three failures must be within 1 MB of the baseline recorded before
// the first failure — proving that each failed call cleans up any partial pool
// allocations and does not leak them into subsequent calls.
// ─────────────────────────────────────────────────────────────────────────────
TEST(PipelineErrors, NoLeakAfterMultipleConsecutiveCompressThrows) {
    constexpr size_t N        = 4096;
    const size_t     in_bytes = N * sizeof(float);

    Pipeline p(in_bytes, MemoryStrategy::MINIMAL);
    auto* lrz = p.addStage<LorenzoStage<float, uint16_t>>();
    lrz->setErrorBound(1e-2f);
    p.setPoolManagedDecompOutput(false);
    p.finalize();

    // Warm up the CUDA runtime allocator so its internal bookkeeping does not
    // skew the before/after comparison.
    {
        CudaStream stream;
        CudaBuffer<float> d_dummy(N);
        auto data = make_random_floats(N, 5);
        d_dummy.upload(data, stream);
        void* d_out = nullptr; size_t out_sz = 0;
        p.compress(d_dummy.void_ptr(), in_bytes, &d_out, &out_sz, stream);
        stream.sync();
        p.reset();
    }

    cudaDeviceSynchronize();
    const size_t free_before = gpu_free_bytes();

    {
        CudaStream stream;
        for (int i = 0; i < 3; ++i) {
            void*  d_out  = nullptr;
            size_t out_sz = 0;
            EXPECT_THROW(
                p.compress(nullptr, in_bytes, &d_out, &out_sz, stream),
                std::runtime_error
            ) << "Null-pointer compress() must throw (iteration " << i << ")";
        }
    }

    cudaDeviceSynchronize();
    const size_t free_after = gpu_free_bytes();

    constexpr size_t TOLERANCE = 1024 * 1024;
    EXPECT_GE(free_after + TOLERANCE, free_before)
        << "GPU memory decreased by more than 1 MB after 3 consecutive failed "
           "compress() calls. Before: " << free_before
        << " B, After: " << free_after << " B";
}

// ─────────────────────────────────────────────────────────────────────────────
// E15: Type-mismatch connection is caught at finalize() time
//
// Lorenzo<float,uint16_t> produces uint16 codes.
// RLEStage<float> expects float input.
// Connecting Lorenzo→RLE with mismatched types must throw std::runtime_error
// at finalize() — before any kernel is ever launched.
// ─────────────────────────────────────────────────────────────────────────────
TEST(PipelineErrors, TypeMismatchAtFinalize) {
    Pipeline p(1024 * sizeof(float), MemoryStrategy::MINIMAL);

    auto* lrz = p.addStage<LorenzoStage<float, uint16_t>>();
    lrz->setErrorBound(1e-2f);
    lrz->setQuantRadius(512);
    lrz->setOutlierCapacity(0.1f);

    // RLEStage<uint32_t> expects uint32 input; Lorenzo outputs uint16 codes.
    auto* rle = p.addStage<RLEStage<uint32_t>>();
    p.connect(rle, lrz, "codes");  // uint16 → uint32: type mismatch

    EXPECT_THROW(p.finalize(), std::runtime_error)
        << "E15: finalize() must throw on a uint16→uint32 type mismatch";
}

// ─────────────────────────────────────────────────────────────────────────────
// E16: Matching types pass finalize() without error
//
// Lorenzo<float,uint16_t> → DifferenceStage<uint16_t> → RLEStage<uint16_t>
// All connections are uint16→uint16; finalize() must succeed.
// ─────────────────────────────────────────────────────────────────────────────
TEST(PipelineErrors, TypeMatchAtFinalizeSucceeds) {
    Pipeline p(1024 * sizeof(float), MemoryStrategy::MINIMAL);

    auto* lrz = p.addStage<LorenzoStage<float, uint16_t>>();
    lrz->setErrorBound(1e-2f);
    lrz->setQuantRadius(512);
    lrz->setOutlierCapacity(0.1f);

    auto* diff = p.addStage<DifferenceStage<uint16_t>>();
    diff->setChunkSize(4096);
    p.connect(diff, lrz, "codes");  // uint16 → uint16: ok

    auto* rle = p.addStage<RLEStage<uint16_t>>();
    p.connect(rle, diff);           // uint16 → uint16: ok

    EXPECT_NO_THROW(p.finalize())
        << "E16: finalize() must not throw when all connection types match";
}
