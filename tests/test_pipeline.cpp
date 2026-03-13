/**
 * tests/test_pipeline.cpp
 *
 * Integration tests for the full compression pipeline.
 *
 * These tests build real Pipeline objects, run compress() and decompress(),
 * and verify the reconstructed data is within the configured error bound.
 *
 * Topologies tested:
 *   1. Minimal: Lorenzo only (1 stage).
 *   2. Standard: Lorenzo → DifferenceStage<uint16_t> (codes branch).
 *   3. Round-trip via file: writeToFile() → decompressFromFile().
 */

#include <gtest/gtest.h>
#include "helpers/fz_test_utils.h"
#include "fzgpumodules.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace fz;
using namespace fz_test;

// ─────────────────────────────────────────────────────────────────────────────
// Utility: generate smooth test data of N floats
// ─────────────────────────────────────────────────────────────────────────────
static std::vector<float> make_smooth_data(size_t n) {
    std::vector<float> v(n);
    for (size_t i = 0; i < n; i++)
        v[i] = std::sin(static_cast<float>(i) * 0.01f) * 50.0f
             + std::cos(static_cast<float>(i) * 0.003f) * 20.0f;
    return v;
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: Lorenzo-only pipeline, in-memory round-trip
// ─────────────────────────────────────────────────────────────────────────────
TEST(Pipeline, LorenzoOnlyRoundTrip) {
    CudaStream stream;
    constexpr size_t N  = 1 << 14;  // 16 K floats
    constexpr float  EB = 1e-2f;

    auto h_input = make_smooth_data(N);
    size_t in_bytes = N * sizeof(float);

    // Upload input
    CudaBuffer<float> d_in(N);
    d_in.upload(h_input, stream);
    stream.sync();

    // Build pipeline
    Pipeline pipeline(in_bytes, MemoryStrategy::MINIMAL);

    auto* lorenzo = pipeline.addStage<LorenzoStage<float, uint16_t>>();
    lorenzo->setErrorBound(EB);
    lorenzo->setQuantRadius(512);
    lorenzo->setOutlierCapacity(0.2f);

    pipeline.finalize();

    // Compress
    void*  d_compressed   = nullptr;
    size_t compressed_sz  = 0;
    pipeline.compress(d_in.void_ptr(), in_bytes, &d_compressed, &compressed_sz, stream);

    EXPECT_GT(compressed_sz, 0u) << "Compressed output is empty";
    EXPECT_LT(compressed_sz, in_bytes) << "Compressed size should be smaller for smooth data";

    // Decompress (in-memory, no d_input needed — uses live forward DAG buffers)
    void*  d_decompressed  = nullptr;
    size_t decompressed_sz = 0;
    pipeline.decompress(nullptr, compressed_sz, &d_decompressed, &decompressed_sz, stream);

    ASSERT_NE(d_decompressed, nullptr);
    EXPECT_EQ(decompressed_sz, in_bytes);

    // Copy back and check error
    std::vector<float> h_recon(N);
    FZ_TEST_CUDA(cudaMemcpy(h_recon.data(), d_decompressed, in_bytes, cudaMemcpyDeviceToHost));
    cudaFree(d_decompressed);

    float max_err = 0.0f;
    for (size_t i = 0; i < N; i++)
        max_err = std::max(max_err, std::abs(h_recon[i] - h_input[i]));

    EXPECT_LE(max_err, EB * 1.01f)
        << "Max reconstruction error " << max_err << " exceeds bound " << EB;
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: Lorenzo → Difference pipeline, in-memory round-trip
// ─────────────────────────────────────────────────────────────────────────────
TEST(Pipeline, LorenzoThenDiffRoundTrip) {
    CudaStream stream;
    constexpr size_t N  = 1 << 14;
    constexpr float  EB = 1e-2f;

    auto h_input = make_smooth_data(N);
    size_t in_bytes = N * sizeof(float);

    CudaBuffer<float> d_in(N);
    d_in.upload(h_input, stream);
    stream.sync();

    Pipeline pipeline(in_bytes, MemoryStrategy::MINIMAL);

    auto* lorenzo = pipeline.addStage<LorenzoStage<float, uint16_t>>();
    lorenzo->setErrorBound(EB);
    lorenzo->setQuantRadius(512);
    lorenzo->setOutlierCapacity(0.2f);

    auto* diff = pipeline.addStage<DifferenceStage<uint16_t>>();
    pipeline.connect(diff, lorenzo, "codes");

    pipeline.finalize();

    void* d_compressed  = nullptr;
    size_t cmp_sz       = 0;
    pipeline.compress(d_in.void_ptr(), in_bytes, &d_compressed, &cmp_sz, stream);

    EXPECT_GT(cmp_sz, 0u);

    void*  d_decompressed = nullptr;
    size_t dcmp_sz        = 0;
    pipeline.decompress(nullptr, cmp_sz, &d_decompressed, &dcmp_sz, stream);

    ASSERT_NE(d_decompressed, nullptr);
    EXPECT_EQ(dcmp_sz, in_bytes);

    std::vector<float> h_recon(N);
    FZ_TEST_CUDA(cudaMemcpy(h_recon.data(), d_decompressed, in_bytes, cudaMemcpyDeviceToHost));
    cudaFree(d_decompressed);

    float max_err = 0.0f;
    for (size_t i = 0; i < N; i++)
        max_err = std::max(max_err, std::abs(h_recon[i] - h_input[i]));

    EXPECT_LE(max_err, EB * 1.01f)
        << "Max reconstruction error " << max_err << " exceeds bound " << EB;
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: write to file, reload, decompress — file-format round-trip
// ─────────────────────────────────────────────────────────────────────────────
TEST(Pipeline, FileRoundTrip) {
    CudaStream stream;
    constexpr size_t N  = 1 << 13;  // 8 K floats — faster for file I/O test
    constexpr float  EB = 1e-2f;

    auto h_input = make_smooth_data(N);
    size_t in_bytes = N * sizeof(float);

    CudaBuffer<float> d_in(N);
    d_in.upload(h_input, stream);
    stream.sync();

    // ── Compress and write to a temp file ─────────────────────────────────
    const std::string tmp_file = "/tmp/fzgmod_test_roundtrip.fzm";

    Pipeline pipeline(in_bytes, MemoryStrategy::MINIMAL);
    auto* lorenzo = pipeline.addStage<LorenzoStage<float, uint16_t>>();
    lorenzo->setErrorBound(EB);
    lorenzo->setQuantRadius(512);
    lorenzo->setOutlierCapacity(0.2f);
    pipeline.finalize();

    void* d_compressed = nullptr;
    size_t cmp_sz      = 0;
    pipeline.compress(d_in.void_ptr(), in_bytes, &d_compressed, &cmp_sz, stream);
    EXPECT_GT(cmp_sz, 0u);

    pipeline.writeToFile(tmp_file, stream);

    // ── Decompress from file using a fresh pipeline ────────────────────────
    Pipeline pipeline2(in_bytes, MemoryStrategy::MINIMAL);
    // pipeline2 only needs to be finalized enough to call decompressFromFile
    // (decompressFromFile is static-ish — it reads the header and rebuilds stages)
    auto* lorenzo2 = pipeline2.addStage<LorenzoStage<float, uint16_t>>();
    lorenzo2->setErrorBound(EB);
    lorenzo2->setQuantRadius(512);
    pipeline2.finalize();

    void*  d_decompressed = nullptr;
    size_t dcmp_sz        = 0;

    pipeline2.decompressFromFile(tmp_file, &d_decompressed, &dcmp_sz, stream, nullptr);

    ASSERT_NE(d_decompressed, nullptr);
    EXPECT_EQ(dcmp_sz, in_bytes);

    std::vector<float> h_recon(N);
    FZ_TEST_CUDA(cudaMemcpy(h_recon.data(), d_decompressed, in_bytes, cudaMemcpyDeviceToHost));
    cudaFree(d_decompressed);

    float max_err = 0.0f;
    for (size_t i = 0; i < N; i++)
        max_err = std::max(max_err, std::abs(h_recon[i] - h_input[i]));

    EXPECT_LE(max_err, EB * 1.01f)
        << "File round-trip max error " << max_err << " exceeds bound " << EB;

    std::remove(tmp_file.c_str());
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: repeated compress+decompress with *different* data each iteration.
// Verifies that pipeline state from a previous run never bleeds into the next.
// ─────────────────────────────────────────────────────────────────────────────
TEST(Pipeline, RepeatCompressDifferentData) {
    CudaStream stream;
    constexpr size_t N  = 4096;
    constexpr float  EB = 1e-2f;
    size_t in_bytes = N * sizeof(float);

    Pipeline pipeline(in_bytes, MemoryStrategy::MINIMAL);
    auto* lorenzo = pipeline.addStage<LorenzoStage<float, uint16_t>>();
    lorenzo->setErrorBound(EB);
    lorenzo->setQuantRadius(512);
    pipeline.finalize();

    // Three datasets with different characteristics
    std::vector<std::vector<float>> datasets = {
        make_smooth_data(N),
        std::vector<float>(N, 3.14f),   // constant
        [&]{ auto v = make_smooth_data(N); for (auto& x:v) x *= 0.01f; return v; }(),  // small range
    };

    for (int iter = 0; iter < 3; iter++) {
        pipeline.reset(stream);

        CudaBuffer<float> d_in(N);
        d_in.upload(datasets[iter], stream);
        stream.sync();

        void*  d_comp = nullptr;
        size_t comp_sz = 0;
        pipeline.compress(d_in.void_ptr(), in_bytes, &d_comp, &comp_sz, stream);
        EXPECT_GT(comp_sz, 0u) << "Iteration " << iter << ": empty compressed output";

        void*  d_dec = nullptr;
        size_t dec_sz = 0;
        pipeline.decompress(nullptr, 0, &d_dec, &dec_sz, stream);
        ASSERT_NE(d_dec, nullptr) << "Iteration " << iter << ": null decompressed pointer";
        ASSERT_EQ(dec_sz, in_bytes)  << "Iteration " << iter << ": wrong decompressed size";

        std::vector<float> h_recon(N);
        FZ_TEST_CUDA(cudaMemcpy(h_recon.data(), d_dec, in_bytes, cudaMemcpyDeviceToHost));
        cudaFree(d_dec);

        float max_err = 0.0f;
        for (size_t i = 0; i < N; i++)
            max_err = std::max(max_err, std::abs(h_recon[i] - datasets[iter][i]));
        EXPECT_LE(max_err, EB * 1.01f)
            << "Iteration " << iter << ": max error " << max_err << " exceeds " << EB;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// I4: Lorenzo → RLE — codes branch fed into RLE encoder
// ─────────────────────────────────────────────────────────────────────────────
TEST(Pipeline, LorenzoPlusRLERoundTrip) {
    CudaStream stream;
    constexpr size_t N  = 1 << 14;  // 16 K floats
    constexpr float  EB = 1e-2f;

    auto h_input = make_smooth_data(N);
    size_t in_bytes = N * sizeof(float);

    CudaBuffer<float> d_in(N);
    d_in.upload(h_input, stream);
    stream.sync();

    Pipeline pipeline(in_bytes, MemoryStrategy::MINIMAL);

    auto* lorenzo = pipeline.addStage<LorenzoStage<float, uint16_t>>();
    lorenzo->setErrorBound(EB);
    lorenzo->setQuantRadius(512);
    lorenzo->setOutlierCapacity(0.2f);

    auto* rle = pipeline.addStage<RLEStage<uint16_t>>();
    pipeline.connect(rle, lorenzo, "codes");
    // outliers output of Lorenzo is left unconnected → becomes a pipeline output

    pipeline.finalize();

    void*  d_compressed = nullptr;
    size_t cmp_sz       = 0;
    pipeline.compress(d_in.void_ptr(), in_bytes, &d_compressed, &cmp_sz, stream);
    EXPECT_GT(cmp_sz, 0u);

    void*  d_decompressed = nullptr;
    size_t dcmp_sz        = 0;
    pipeline.decompress(nullptr, cmp_sz, &d_decompressed, &dcmp_sz, stream);
    ASSERT_NE(d_decompressed, nullptr);
    EXPECT_EQ(dcmp_sz, in_bytes);

    std::vector<float> h_recon(N);
    FZ_TEST_CUDA(cudaMemcpy(h_recon.data(), d_decompressed, in_bytes, cudaMemcpyDeviceToHost));
    cudaFree(d_decompressed);

    float max_err = 0.0f;
    for (size_t i = 0; i < N; i++)
        max_err = std::max(max_err, std::abs(h_recon[i] - h_input[i]));
    EXPECT_LE(max_err, EB * 1.01f)
        << "Lorenzo+RLE max error " << max_err << " exceeds bound " << EB;
}

// ─────────────────────────────────────────────────────────────────────────────
// I5: Lorenzo → Difference → RLE (3-stage pipeline)
// ─────────────────────────────────────────────────────────────────────────────
TEST(Pipeline, LorenzoDiffRLERoundTrip) {
    CudaStream stream;
    constexpr size_t N  = 1 << 14;
    constexpr float  EB = 1e-2f;

    auto h_input = make_smooth_data(N);
    size_t in_bytes = N * sizeof(float);

    CudaBuffer<float> d_in(N);
    d_in.upload(h_input, stream);
    stream.sync();

    Pipeline pipeline(in_bytes, MemoryStrategy::MINIMAL);

    auto* lorenzo = pipeline.addStage<LorenzoStage<float, uint16_t>>();
    lorenzo->setErrorBound(EB);
    lorenzo->setQuantRadius(512);
    lorenzo->setOutlierCapacity(0.2f);

    auto* diff = pipeline.addStage<DifferenceStage<uint16_t>>();
    pipeline.connect(diff, lorenzo, "codes");

    auto* rle = pipeline.addStage<RLEStage<uint16_t>>();
    pipeline.connect(rle, diff);
    // Lorenzo outliers remain unconnected → second pipeline output

    pipeline.finalize();

    void*  d_compressed = nullptr;
    size_t cmp_sz       = 0;
    pipeline.compress(d_in.void_ptr(), in_bytes, &d_compressed, &cmp_sz, stream);
    EXPECT_GT(cmp_sz, 0u);

    void*  d_decompressed = nullptr;
    size_t dcmp_sz        = 0;
    pipeline.decompress(nullptr, cmp_sz, &d_decompressed, &dcmp_sz, stream);
    ASSERT_NE(d_decompressed, nullptr);
    EXPECT_EQ(dcmp_sz, in_bytes);

    std::vector<float> h_recon(N);
    FZ_TEST_CUDA(cudaMemcpy(h_recon.data(), d_decompressed, in_bytes, cudaMemcpyDeviceToHost));
    cudaFree(d_decompressed);

    float max_err = 0.0f;
    for (size_t i = 0; i < N; i++)
        max_err = std::max(max_err, std::abs(h_recon[i] - h_input[i]));
    EXPECT_LE(max_err, EB * 1.01f)
        << "Lorenzo+Diff+RLE max error " << max_err << " exceeds bound " << EB;
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: compress, pipeline reset, compress again — state resets cleanly
// ─────────────────────────────────────────────────────────────────────────────
TEST(Pipeline, RepeatCompress) {
    CudaStream stream;
    constexpr size_t N  = 4096;
    constexpr float  EB = 1e-2f;

    auto h_input = make_smooth_data(N);
    size_t in_bytes = N * sizeof(float);

    CudaBuffer<float> d_in(N);
    d_in.upload(h_input, stream);
    stream.sync();

    Pipeline pipeline(in_bytes, MemoryStrategy::MINIMAL);
    auto* lorenzo = pipeline.addStage<LorenzoStage<float, uint16_t>>();
    lorenzo->setErrorBound(EB);
    lorenzo->setQuantRadius(512);
    pipeline.finalize();

    for (int iter = 0; iter < 3; iter++) {
        pipeline.reset(stream);

        void*  d_out = nullptr;
        size_t out_sz = 0;
        pipeline.compress(d_in.void_ptr(), in_bytes, &d_out, &out_sz, stream);

        EXPECT_GT(out_sz, 0u) << "Iteration " << iter << ": empty output";

        void*  d_dec = nullptr;
        size_t dec_sz = 0;
        pipeline.decompress(nullptr, out_sz, &d_dec, &dec_sz, stream);
        ASSERT_NE(d_dec, nullptr) << "Iteration " << iter;
        cudaFree(d_dec);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: PREALLOCATE pipeline reused across multiple compress+decompress calls
// without reset() between iterations.
//
// This mirrors ARM A of the repeated-profiling example (profile_repeat.cpp),
// which exposed reuse bugs. Verifies that compressed output and reconstruction
// accuracy are correct on every iteration when the same input is used.
// ─────────────────────────────────────────────────────────────────────────────
TEST(Pipeline, RepeatedCompressPreallocateSameData) {
    CudaStream stream;
    constexpr size_t N      = 1 << 14;  // 16 K floats
    constexpr float  EB     = 1e-2f;
    constexpr int    N_RUNS = 5;

    auto h_input = make_smooth_data(N);
    size_t in_bytes = N * sizeof(float);

    CudaBuffer<float> d_in(N);
    d_in.upload(h_input, stream);
    stream.sync();

    // Single PREALLOCATE pipeline — mirrors ARM A of profile_repeat
    Pipeline pipeline(in_bytes, MemoryStrategy::PREALLOCATE);

    auto* lorenzo = pipeline.addStage<LorenzoStage<float, uint16_t>>();
    lorenzo->setErrorBound(EB);
    lorenzo->setQuantRadius(512);
    lorenzo->setOutlierCapacity(0.2f);

    auto* diff = pipeline.addStage<DifferenceStage<uint16_t>>();
    pipeline.connect(diff, lorenzo, "codes");

    pipeline.finalize();

    for (int iter = 0; iter < N_RUNS; ++iter) {
        void*  d_comp  = nullptr;
        size_t comp_sz = 0;
        pipeline.compress(d_in.void_ptr(), in_bytes, &d_comp, &comp_sz, stream);

        EXPECT_GT(comp_sz, 0u) << "Iteration " << iter << ": empty compressed output";

        void*  d_dec  = nullptr;
        size_t dec_sz = 0;
        pipeline.decompress(nullptr, comp_sz, &d_dec, &dec_sz, stream);

        ASSERT_NE(d_dec, nullptr) << "Iteration " << iter << ": null decompressed pointer";
        ASSERT_EQ(dec_sz, in_bytes) << "Iteration " << iter << ": wrong decompressed size";

        std::vector<float> h_recon(N);
        FZ_TEST_CUDA(cudaMemcpy(h_recon.data(), d_dec, in_bytes, cudaMemcpyDeviceToHost));
        cudaFree(d_dec);

        float max_err = 0.0f;
        for (size_t i = 0; i < N; i++)
            max_err = std::max(max_err, std::abs(h_recon[i] - h_input[i]));
        EXPECT_LE(max_err, EB * 1.01f)
            << "Iteration " << iter << ": max error " << max_err << " exceeds bound " << EB;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: repeated compress+decompress on the same input produces identical
// reconstructed output across runs.
// ─────────────────────────────────────────────────────────────────────────────
TEST(Pipeline, RepeatedCompressDecompressStableOutput) {
    CudaStream stream;
    constexpr size_t N      = 1 << 14;
    constexpr float  EB     = 1e-2f;
    constexpr int    N_RUNS = 3;

    auto h_input = make_smooth_data(N);
    size_t in_bytes = N * sizeof(float);

    CudaBuffer<float> d_in(N);
    d_in.upload(h_input, stream);
    stream.sync();

    Pipeline pipeline(in_bytes, MemoryStrategy::PREALLOCATE);

    auto* lorenzo = pipeline.addStage<LorenzoStage<float, uint16_t>>();
    lorenzo->setErrorBound(EB);
    lorenzo->setQuantRadius(512);
    lorenzo->setOutlierCapacity(0.2f);

    auto* diff = pipeline.addStage<DifferenceStage<uint16_t>>();
    pipeline.connect(diff, lorenzo, "codes");

    pipeline.finalize();

    std::vector<float> baseline_recon;
    baseline_recon.reserve(N);

    for (int iter = 0; iter < N_RUNS; ++iter) {
        void*  d_comp  = nullptr;
        size_t comp_sz = 0;
        pipeline.compress(d_in.void_ptr(), in_bytes, &d_comp, &comp_sz, stream);
        EXPECT_GT(comp_sz, 0u) << "Iteration " << iter << ": empty compressed output";

        void*  d_dec  = nullptr;
        size_t dec_sz = 0;
        pipeline.decompress(nullptr, comp_sz, &d_dec, &dec_sz, stream);

        ASSERT_NE(d_dec, nullptr) << "Iteration " << iter << ": null decompressed pointer";
        ASSERT_EQ(dec_sz, in_bytes) << "Iteration " << iter << ": wrong decompressed size";

        std::vector<float> h_recon(N);
        FZ_TEST_CUDA(cudaMemcpy(h_recon.data(), d_dec, in_bytes, cudaMemcpyDeviceToHost));
        cudaFree(d_dec);

        float max_err = 0.0f;
        for (size_t i = 0; i < N; i++)
            max_err = std::max(max_err, std::abs(h_recon[i] - h_input[i]));
        EXPECT_LE(max_err, EB * 1.01f)
            << "Iteration " << iter << ": max error " << max_err << " exceeds bound " << EB;

        if (iter == 0) {
            baseline_recon = h_recon;
        } else {
            for (size_t i = 0; i < N; ++i) {
                EXPECT_FLOAT_EQ(h_recon[i], baseline_recon[i])
                    << "Mismatch at iteration " << iter << ", index " << i;
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: PREALLOCATE pipeline reused with different data on each call.
//
// Ensures that no stale compressed state from one iteration bleeds into the
// next decompression when the pipeline is reused without reset().
// ─────────────────────────────────────────────────────────────────────────────
TEST(Pipeline, RepeatedCompressPreallocateDifferentData) {
    CudaStream stream;
    constexpr size_t N  = 1 << 14;
    constexpr float  EB = 1e-2f;
    size_t in_bytes = N * sizeof(float);

    Pipeline pipeline(in_bytes, MemoryStrategy::PREALLOCATE);

    auto* lorenzo = pipeline.addStage<LorenzoStage<float, uint16_t>>();
    lorenzo->setErrorBound(EB);
    lorenzo->setQuantRadius(512);
    lorenzo->setOutlierCapacity(0.2f);

    auto* diff = pipeline.addStage<DifferenceStage<uint16_t>>();
    pipeline.connect(diff, lorenzo, "codes");

    pipeline.finalize();

    std::vector<std::vector<float>> datasets = {
        make_smooth_data(N),
        std::vector<float>(N, 3.14f),   // constant
        [&] { auto v = make_smooth_data(N); for (auto& x : v) x *= 0.5f;    return v; }(),
        [&] { auto v = make_smooth_data(N); for (auto& x : v) x += 100.0f;  return v; }(),
        std::vector<float>(N, 0.0f),    // all zeros
    };

    for (int iter = 0; iter < static_cast<int>(datasets.size()); ++iter) {
        CudaBuffer<float> d_in(N);
        d_in.upload(datasets[iter], stream);
        stream.sync();

        void*  d_comp  = nullptr;
        size_t comp_sz = 0;
        pipeline.compress(d_in.void_ptr(), in_bytes, &d_comp, &comp_sz, stream);

        EXPECT_GT(comp_sz, 0u) << "Iteration " << iter << ": empty compressed output";

        void*  d_dec  = nullptr;
        size_t dec_sz = 0;
        pipeline.decompress(nullptr, comp_sz, &d_dec, &dec_sz, stream);

        ASSERT_NE(d_dec, nullptr) << "Iteration " << iter << ": null decompressed pointer";
        ASSERT_EQ(dec_sz, in_bytes) << "Iteration " << iter << ": wrong decompressed size";

        std::vector<float> h_recon(N);
        FZ_TEST_CUDA(cudaMemcpy(h_recon.data(), d_dec, in_bytes, cudaMemcpyDeviceToHost));
        cudaFree(d_dec);

        float max_err = 0.0f;
        for (size_t i = 0; i < N; i++)
            max_err = std::max(max_err, std::abs(h_recon[i] - datasets[iter][i]));
        EXPECT_LE(max_err, EB * 1.01f)
            << "Iteration " << iter << ": max error " << max_err << " exceeds bound " << EB;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DIM4: addLorenzo() correctly forwards pipeline dims_ to stage config.
//
// Verifies that setDims() + addLorenzo() produces the same ndim() as manually
// calling setDims() on a LorenzoStage directly.
// ─────────────────────────────────────────────────────────────────────────────
TEST(Pipeline, AddLorenzoForwardsDims) {
    // 1D: default (no setDims call)
    {
        Pipeline p1(256 * sizeof(float), MemoryStrategy::MINIMAL);
        auto* lrz1 = p1.addLorenzo<float, uint16_t>(1e-2f);
        EXPECT_EQ(lrz1->ndim(), 1) << "Default should be 1D";
    }

    // 2D: setDims(nx, ny) before addLorenzo
    {
        constexpr size_t NX = 32, NY = 32;
        Pipeline p2(NX * NY * sizeof(float), MemoryStrategy::MINIMAL);
        p2.setDims(NX, NY);
        auto* lrz2 = p2.addLorenzo<float, uint16_t>(1e-2f);
        EXPECT_EQ(lrz2->ndim(), 2) << "setDims(nx,ny) should give 2D via addLorenzo()";
        auto dims2 = lrz2->getDims();
        EXPECT_EQ(dims2[0], NX);
        EXPECT_EQ(dims2[1], NY);
        EXPECT_EQ(dims2[2], 1u);
    }

    // 3D: setDims(nx, ny, nz) before addLorenzo
    {
        constexpr size_t NX = 16, NY = 16, NZ = 16;
        Pipeline p3(NX * NY * NZ * sizeof(float), MemoryStrategy::MINIMAL);
        p3.setDims(NX, NY, NZ);
        auto* lrz3 = p3.addLorenzo<float, uint16_t>(1e-2f);
        EXPECT_EQ(lrz3->ndim(), 3) << "setDims(nx,ny,nz) should give 3D via addLorenzo()";
    }
}
