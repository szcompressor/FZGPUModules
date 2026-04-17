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

// ─────────────────────────────────────────────────────────────────────────────
// MS1: Multi-source compress + decompressMulti round-trip.
//
// Two independent Lorenzo stages are roots (no shared connection) in one
// Pipeline.  compress() receives an InputSpec per source; decompressMulti()
// returns one reconstructed buffer per source.  Both must be within error
// bound on independent data.
// ─────────────────────────────────────────────────────────────────────────────
TEST(Pipeline, MultiSourceRoundTrip) {
    constexpr size_t N  = 1 << 13;   // 8 K floats per source
    constexpr float  EB = 1e-2f;
    const size_t     in_bytes = N * sizeof(float);

    // Use the same data for both sources so we don't need to know which result
    // index corresponds to which source (decompressMulti returns results in
    // input_nodes_ discovery order, which may not match InputSpec order).
    auto h_input1 = make_smooth_data(N);
    const auto& h_input2 = h_input1;

    CudaStream stream;
    CudaBuffer<float> d_in1(N), d_in2(N);
    d_in1.upload(h_input1, stream);
    d_in2.upload(h_input2, stream);
    stream.sync();

    // Build a pipeline with two independent Lorenzo sources.
    // Both are roots (no upstream); both are sinks (no downstream connection).
    // The pipeline total hint is 2× per-source size.
    Pipeline pipeline(2 * in_bytes, MemoryStrategy::MINIMAL, 5.0f);
    auto* lrz1 = pipeline.addStage<LorenzoStage<float, uint16_t>>();
    lrz1->setErrorBound(EB);
    lrz1->setQuantRadius(512);
    lrz1->setOutlierCapacity(0.2f);

    auto* lrz2 = pipeline.addStage<LorenzoStage<float, uint16_t>>();
    lrz2->setErrorBound(EB);
    lrz2->setQuantRadius(512);
    lrz2->setOutlierCapacity(0.2f);

    // Per-source hints so propagateBufferSizes() can estimate downstream buffers.
    pipeline.setInputSizeHint(lrz1, in_bytes);
    pipeline.setInputSizeHint(lrz2, in_bytes);

    pipeline.finalize();

    void*  d_comp  = nullptr;
    size_t comp_sz = 0;
    pipeline.compress(
        {{lrz1, d_in1.void_ptr(), in_bytes},
         {lrz2, d_in2.void_ptr(), in_bytes}},
        &d_comp, &comp_sz, stream
    );
    stream.sync();
    ASSERT_GT(comp_sz, 0u) << "Multi-source compress output is empty";

    auto results = pipeline.decompressMulti(nullptr, 0, stream);
    stream.sync();

    ASSERT_EQ(results.size(), 2u) << "Expected 2 decompressed outputs";

    // Each result must be the right size and within error bound.
    for (size_t src = 0; src < 2; src++) {
        auto [d_dec, dec_sz] = results[src];
        ASSERT_NE(d_dec, nullptr)  << "Source " << src << ": null decompressed pointer";
        ASSERT_EQ(dec_sz, in_bytes) << "Source " << src << ": wrong decompressed size";

        std::vector<float> h_recon(N);
        cudaMemcpy(h_recon.data(), d_dec, in_bytes, cudaMemcpyDeviceToHost);
        cudaFree(d_dec);

        const auto& h_orig = (src == 0) ? h_input1 : h_input2;
        float max_err = max_abs_error(h_orig, h_recon);
        EXPECT_LE(max_err, EB * 1.01f)
            << "Source " << src << ": max_err=" << max_err
            << " exceeds bound " << EB;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// RS1: Explicit reset() between compress cycles.
//
// After an explicit pipeline.reset(), was_compressed_ is cleared.  The next
// compress() must not auto-reset (which would cause a double-reset) and must
// produce correct data.  Also verifies that the pool usage counter returns
// to 0 after reset and rises again after the second compress.
// ─────────────────────────────────────────────────────────────────────────────
TEST(Pipeline, ExplicitResetThenCompress) {
    constexpr size_t N  = 1 << 14;
    constexpr float  EB = 1e-2f;
    const size_t in_bytes = N * sizeof(float);

    CudaStream stream;
    auto h_input = make_smooth_data(N);
    CudaBuffer<float> d_in(N);
    d_in.upload(h_input, stream);
    stream.sync();

    Pipeline pipeline(in_bytes, MemoryStrategy::MINIMAL);
    auto* lrz = pipeline.addStage<LorenzoStage<float, uint16_t>>();
    lrz->setErrorBound(EB);
    lrz->setQuantRadius(512);
    lrz->setOutlierCapacity(0.2f);
    pipeline.finalize();

    // First compress.
    void*  d_comp1 = nullptr;
    size_t comp_sz1 = 0;
    pipeline.compress(d_in.void_ptr(), in_bytes, &d_comp1, &comp_sz1, stream);
    stream.sync();
    ASSERT_GT(comp_sz1, 0u);
    EXPECT_GT(pipeline.getCurrentMemoryUsage(), 0u);

    // Explicit reset — pool usage must drop to 0.
    pipeline.reset(stream);
    stream.sync();
    EXPECT_EQ(pipeline.getCurrentMemoryUsage(), 0u)
        << "Memory usage should be 0 after explicit reset()";

    // Second compress after explicit reset must produce correct results.
    void*  d_comp2 = nullptr;
    size_t comp_sz2 = 0;
    pipeline.compress(d_in.void_ptr(), in_bytes, &d_comp2, &comp_sz2, stream);
    stream.sync();
    ASSERT_GT(comp_sz2, 0u);
    // After reset(), the counter reflects only the persistent output buffer
    // that was kept alive.  After the second compress(), intermediate buffers
    // are re-allocated and the counter rises again.
    EXPECT_GT(pipeline.getCurrentMemoryUsage(), 0u)
        << "Memory usage should be non-zero after second compress()";

    void*  d_dec = nullptr;
    size_t dec_sz = 0;
    pipeline.decompress(nullptr, 0, &d_dec, &dec_sz, stream);
    stream.sync();
    ASSERT_NE(d_dec, nullptr);
    ASSERT_EQ(dec_sz, in_bytes);

    std::vector<float> h_recon(N);
    cudaMemcpy(h_recon.data(), d_dec, in_bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_dec);

    float max_err = max_abs_error(h_input, h_recon);
    EXPECT_LE(max_err, EB * 1.01f)
        << "Post-reset compress/decompress max_err=" << max_err
        << " exceeds bound " << EB;
}

// ─────────────────────────────────────────────────────────────────────────────
// VP1: compress() with varying input sizes across calls produces correct output
//
// Buffer estimates are made at finalize() using the hint (N floats).  The
// pipeline must correctly handle successive compress() calls where the input
// size changes between calls (≤ the hint).  Each call must round-trip cleanly.
// ─────────────────────────────────────────────────────────────────────────────
TEST(Pipeline, VaryingInputSizesAcrossCalls) {
    constexpr size_t N  = 1 << 13;   // 8 K floats — finalize-time hint
    constexpr float  EB = 1e-2f;
    const size_t in_bytes = N * sizeof(float);

    CudaStream stream;
    auto h_large  = make_smooth_data(N);
    auto h_small  = make_smooth_data(N / 2);

    CudaBuffer<float> d_large(N);
    CudaBuffer<float> d_small(N / 2);
    d_large.upload(h_large, stream);
    d_small.upload(h_small, stream);
    stream.sync();

    // Hint = full N — large enough for all calls below.
    Pipeline pipeline(in_bytes, MemoryStrategy::MINIMAL);
    auto* lrz = pipeline.addStage<LorenzoStage<float, uint16_t>>();
    lrz->setErrorBound(EB);
    lrz->setQuantRadius(512);
    lrz->setOutlierCapacity(0.2f);
    pipeline.finalize();

    auto roundtrip = [&](void* d_in, size_t bytes, const std::vector<float>& h_ref,
                         const char* label) {
        void*  d_comp = nullptr;
        size_t comp_sz = 0;
        ASSERT_NO_THROW(
            pipeline.compress(d_in, bytes, &d_comp, &comp_sz, stream)
        ) << label << ": compress threw";
        stream.sync();
        ASSERT_GT(comp_sz, 0u) << label << ": empty compressed output";

        void*  d_dec  = nullptr;
        size_t dec_sz = 0;
        ASSERT_NO_THROW(
            pipeline.decompress(nullptr, 0, &d_dec, &dec_sz, stream)
        ) << label << ": decompress threw";
        stream.sync();
        ASSERT_NE(d_dec, nullptr) << label << ": null decompress output";
        ASSERT_EQ(dec_sz, bytes)  << label << ": wrong decompressed size";

        const size_t n = bytes / sizeof(float);
        std::vector<float> h_recon(n);
        cudaMemcpy(h_recon.data(), d_dec, bytes, cudaMemcpyDeviceToHost);
        cudaFree(d_dec);

        float max_err = max_abs_error(h_ref, h_recon);
        EXPECT_LE(max_err, EB * 1.01f)
            << label << ": max_err=" << max_err << " exceeds bound " << EB;
    };

    // Call 1: full N floats
    roundtrip(d_large.void_ptr(), in_bytes, h_large, "call1-large");

    // Call 2: half N floats (smaller than hint)
    roundtrip(d_small.void_ptr(), in_bytes / 2, h_small, "call2-small");

    // Call 3: full N floats again
    roundtrip(d_large.void_ptr(), in_bytes, h_large, "call3-large");
}

// ─────────────────────────────────────────────────────────────────────────────
// LD1: Large-scale compress + decompress (~100 MB)
//
// Exercises the pool sizing heuristics, buffer coloring, and multi-stage
// round-trip at a scale where small-N tests cannot surface memory pressure
// bugs (pool under-allocation, integer overflow in size calculations, etc.).
//
// Pipeline: Lorenzo → RLE (Lorenzo codes → RLE compressed codes)
// Data: smooth sinusoidal — high compression ratio, no outliers.
//
// Strategy: PREALLOCATE so pool sizing is fully exercised at finalize() time.
// Pool multiplier: 4× to give comfortable headroom at 100 MB scale.
// ─────────────────────────────────────────────────────────────────────────────
TEST(Pipeline, LargeScaleRoundTrip) {
    // 100 MB of float32 = 25 million elements
    constexpr size_t N        = 25 * 1024 * 1024;
    constexpr float  EB       = 1e-2f;
    const size_t     in_bytes = N * sizeof(float);

    // Generate smooth data host-side in chunks to avoid one 100 MB stack alloc.
    std::vector<float> h_input(N);
    for (size_t i = 0; i < N; i++)
        h_input[i] = std::sin(static_cast<float>(i) * 0.001f) * 50.0f
                   + std::cos(static_cast<float>(i) * 0.0003f) * 20.0f;

    CudaStream stream;

    // Upload to device
    CudaBuffer<float> d_in(N);
    d_in.upload(h_input, stream);
    stream.sync();

    // Build pipeline: Lorenzo → RLE
    Pipeline pipeline(in_bytes, MemoryStrategy::PREALLOCATE, 4.0f);
    auto* lrz = pipeline.addStage<LorenzoStage<float, uint16_t>>();
    lrz->setErrorBound(EB);
    lrz->setQuantRadius(512);
    lrz->setOutlierCapacity(0.05f);  // 5% outlier budget at 100 MB scale
    auto* rle = pipeline.addStage<RLEStage<uint16_t>>();
    pipeline.connect(rle, lrz, "codes");
    pipeline.finalize();

    // Compress
    void*  d_comp  = nullptr;
    size_t comp_sz = 0;
    ASSERT_NO_THROW(
        pipeline.compress(d_in.void_ptr(), in_bytes, &d_comp, &comp_sz, stream)
    ) << "Large-scale compress must not throw";
    stream.sync();
    ASSERT_GT(comp_sz, 0u) << "Compressed output is empty";
    EXPECT_LT(comp_sz, in_bytes) << "Smooth data must compress smaller than raw input";

    // Decompress
    void*  d_dec  = nullptr;
    size_t dec_sz = 0;
    ASSERT_NO_THROW(
        pipeline.decompress(nullptr, 0, &d_dec, &dec_sz, stream)
    ) << "Large-scale decompress must not throw";
    stream.sync();
    ASSERT_NE(d_dec, nullptr) << "Decompressed pointer is null";
    ASSERT_EQ(dec_sz, in_bytes) << "Decompressed size mismatch";

    // Download and verify — sample every 1024 elements to keep test time bounded
    // while still covering the full 100 MB address space.
    constexpr size_t STRIDE = 1024;
    std::vector<float> h_sample(N / STRIDE);
    std::vector<float> h_recon_sample(N / STRIDE);
    for (size_t i = 0; i < N / STRIDE; ++i)
        h_sample[i] = h_input[i * STRIDE];

    // D2H only the sampled elements (one cudaMemcpy per stride would be slow;
    // copy the full array and pick samples on the host).
    std::vector<float> h_recon(N);
    FZ_TEST_CUDA(cudaMemcpy(h_recon.data(), d_dec, in_bytes, cudaMemcpyDeviceToHost));
    cudaFree(d_dec);

    float max_err = 0.0f;
    for (size_t i = 0; i < N; i += STRIDE)
        max_err = std::max(max_err, std::abs(h_recon[i] - h_input[i]));

    EXPECT_LE(max_err, EB * 1.01f)
        << "LD1: large-scale round-trip max_err=" << max_err
        << " exceeds bound " << EB;
}
