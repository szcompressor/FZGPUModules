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
