/**
 * tests/pipeline/test_pipeline.cpp
 *
 * Integration tests for the full compression pipeline.
 *
 * These tests build real Pipeline objects, run compress() and decompress(),
 * and verify the reconstructed data is within the configured error bound.
 *
 * Tests:
 *   P1   LorenzoOnlyRoundTrip           — Lorenzo-only pipeline, in-memory round-trip
 *   P2   LorenzoThenDiffRoundTrip       — Lorenzo → DifferenceStage round-trip
 *   P3   FileRoundTrip                  — write to file, reload, decompress
 *   P4   RepeatCompressDifferentData    — repeated compress+decompress with different data each iter
 *   P5   LorenzoPlusRLERoundTrip        — Lorenzo → RLE codes branch round-trip
 *   P6   LorenzoDiffRLERoundTrip        — Lorenzo → Difference → RLE (3-stage)
 *   P7   RepeatCompress                 — compress, reset, compress again — state resets cleanly
 *   P8   RepeatedCompressPreallocateSameData   — PREALLOCATE reused across multiple calls, same data
 *   P9   RepeatedCompressDecompressStableOutput — repeated compress+decompress produces identical output
 *   P10  RepeatedCompressPreallocateDifferentData — PREALLOCATE reused with different data each call
 *   P11  AddStageForwardsDims           — setDims() before addStage() immediately gives correct ndim()
 *   P12  ExplicitResetThenCompress      — explicit reset() between compress cycles
 *   P13  VaryingInputSizesAcrossCalls   — compress() with varying input sizes across calls
 *   P14  LargeScaleRoundTrip            — large-scale compress + decompress (~100 MB)
 *   P15  Lorenzo2DRoundTrip             — 2D Lorenzo round-trip (64×64 grid)
 *   P16  Lorenzo3DRoundTrip             — 3D Lorenzo round-trip (16×16×16 grid)
 */

#include <gtest/gtest.h>
#include "helpers/fz_test_utils.h"
#include "helpers/stage_harness.h"
#include "fzgpumodules.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace fz;
using namespace fz_test;

// ─────────────────────────────────────────────────────────────────────────────
// P1: Lorenzo-only pipeline, in-memory round-trip
// ─────────────────────────────────────────────────────────────────────────────
TEST(Pipeline, LorenzoOnlyRoundTrip) {
    CudaStream stream;
    constexpr size_t N  = 1 << 14;  // 16 K floats
    constexpr float  EB = 1e-2f;

    auto h_input = make_smooth_data<float>(N);
    size_t in_bytes = N * sizeof(float);

    // Upload input
    CudaBuffer<float> d_in(N);
    d_in.upload(h_input, stream);
    stream.sync();

    // Build pipeline
    Pipeline pipeline(in_bytes, MemoryStrategy::MINIMAL);

    auto* lorenzo = pipeline.addStage<LorenzoQuantStage<float, uint16_t>>();
    lorenzo->setErrorBound(EB);
    lorenzo->setQuantRadius(512);
    lorenzo->setOutlierCapacity(0.2f);

    pipeline.setPoolManagedDecompOutput(false);
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
// P2: Lorenzo → Difference pipeline, in-memory round-trip
// ─────────────────────────────────────────────────────────────────────────────
TEST(Pipeline, LorenzoThenDiffRoundTrip) {
    CudaStream stream;
    constexpr size_t N  = 1 << 14;
    constexpr float  EB = 1e-2f;

    auto h_input = make_smooth_data<float>(N);
    size_t in_bytes = N * sizeof(float);

    CudaBuffer<float> d_in(N);
    d_in.upload(h_input, stream);
    stream.sync();

    Pipeline pipeline(in_bytes, MemoryStrategy::MINIMAL);

    auto* lorenzo = pipeline.addStage<LorenzoQuantStage<float, uint16_t>>();
    lorenzo->setErrorBound(EB);
    lorenzo->setQuantRadius(512);
    lorenzo->setOutlierCapacity(0.2f);

    auto* diff = pipeline.addStage<DifferenceStage<uint16_t>>();
    pipeline.connect(diff, lorenzo, "codes");

    pipeline.setPoolManagedDecompOutput(false);
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
// P3: write to file, reload, decompress — file-format round-trip
// ─────────────────────────────────────────────────────────────────────────────
TEST(Pipeline, FileRoundTrip) {
    CudaStream stream;
    constexpr size_t N  = 1 << 13;  // 8 K floats — faster for file I/O test
    constexpr float  EB = 1e-2f;

    auto h_input = make_smooth_data<float>(N);
    size_t in_bytes = N * sizeof(float);

    CudaBuffer<float> d_in(N);
    d_in.upload(h_input, stream);
    stream.sync();

    // ── Compress and write to a temp file ─────────────────────────────────
    const std::string tmp_file = "/tmp/fzgmod_test_roundtrip.fzm";

    Pipeline pipeline(in_bytes, MemoryStrategy::MINIMAL);
    auto* lorenzo = pipeline.addStage<LorenzoQuantStage<float, uint16_t>>();
    lorenzo->setErrorBound(EB);
    lorenzo->setQuantRadius(512);
    lorenzo->setOutlierCapacity(0.2f);
    pipeline.setPoolManagedDecompOutput(false);
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
    auto* lorenzo2 = pipeline2.addStage<LorenzoQuantStage<float, uint16_t>>();
    lorenzo2->setErrorBound(EB);
    lorenzo2->setQuantRadius(512);
    pipeline2.setPoolManagedDecompOutput(false);
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
// P4: repeated compress+decompress with different data each iteration.
// Verifies that pipeline state from a previous run never bleeds into the next.
// ─────────────────────────────────────────────────────────────────────────────
TEST(Pipeline, RepeatCompressDifferentData) {
    CudaStream stream;
    constexpr size_t N  = 4096;
    constexpr float  EB = 1e-2f;
    size_t in_bytes = N * sizeof(float);

    Pipeline pipeline(in_bytes, MemoryStrategy::MINIMAL);
    auto* lorenzo = pipeline.addStage<LorenzoQuantStage<float, uint16_t>>();
    lorenzo->setErrorBound(EB);
    lorenzo->setQuantRadius(512);
    pipeline.setPoolManagedDecompOutput(false);
    pipeline.finalize();

    // Three datasets with different characteristics
    std::vector<std::vector<float>> datasets = {
        make_smooth_data<float>(N),
        std::vector<float>(N, 3.14f),   // constant
        [&]{ auto v = make_smooth_data<float>(N); for (auto& x:v) x *= 0.01f; return v; }(),  // small range
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
// P5: Lorenzo → RLE — codes branch fed into RLE encoder
// ─────────────────────────────────────────────────────────────────────────────
TEST(Pipeline, LorenzoPlusRLERoundTrip) {
    CudaStream stream;
    constexpr size_t N  = 1 << 14;  // 16 K floats
    constexpr float  EB = 1e-2f;

    auto h_input = make_smooth_data<float>(N);
    size_t in_bytes = N * sizeof(float);

    CudaBuffer<float> d_in(N);
    d_in.upload(h_input, stream);
    stream.sync();

    Pipeline pipeline(in_bytes, MemoryStrategy::MINIMAL);

    auto* lorenzo = pipeline.addStage<LorenzoQuantStage<float, uint16_t>>();
    lorenzo->setErrorBound(EB);
    lorenzo->setQuantRadius(512);
    lorenzo->setOutlierCapacity(0.2f);

    auto* rle = pipeline.addStage<RLEStage<uint16_t>>();
    pipeline.connect(rle, lorenzo, "codes");
    // outliers output of Lorenzo is left unconnected → becomes a pipeline output

    pipeline.setPoolManagedDecompOutput(false);
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
// P6: Lorenzo → Difference → RLE (3-stage pipeline)
// ─────────────────────────────────────────────────────────────────────────────
TEST(Pipeline, LorenzoDiffRLERoundTrip) {
    CudaStream stream;
    constexpr size_t N  = 1 << 14;
    constexpr float  EB = 1e-2f;

    auto h_input = make_smooth_data<float>(N);
    size_t in_bytes = N * sizeof(float);

    CudaBuffer<float> d_in(N);
    d_in.upload(h_input, stream);
    stream.sync();

    Pipeline pipeline(in_bytes, MemoryStrategy::MINIMAL);

    auto* lorenzo = pipeline.addStage<LorenzoQuantStage<float, uint16_t>>();
    lorenzo->setErrorBound(EB);
    lorenzo->setQuantRadius(512);
    lorenzo->setOutlierCapacity(0.2f);

    auto* diff = pipeline.addStage<DifferenceStage<uint16_t>>();
    pipeline.connect(diff, lorenzo, "codes");

    auto* rle = pipeline.addStage<RLEStage<uint16_t>>();
    pipeline.connect(rle, diff);
    // Lorenzo outliers remain unconnected → second pipeline output

    pipeline.setPoolManagedDecompOutput(false);
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
// P7: compress, pipeline reset, compress again — state resets cleanly
// ─────────────────────────────────────────────────────────────────────────────
TEST(Pipeline, RepeatCompress) {
    CudaStream stream;
    constexpr size_t N  = 4096;
    constexpr float  EB = 1e-2f;

    auto h_input = make_smooth_data<float>(N);
    size_t in_bytes = N * sizeof(float);

    CudaBuffer<float> d_in(N);
    d_in.upload(h_input, stream);
    stream.sync();

    Pipeline pipeline(in_bytes, MemoryStrategy::MINIMAL);
    auto* lorenzo = pipeline.addStage<LorenzoQuantStage<float, uint16_t>>();
    lorenzo->setErrorBound(EB);
    lorenzo->setQuantRadius(512);
    pipeline.setPoolManagedDecompOutput(false);
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
// P8: PREALLOCATE pipeline reused across multiple compress+decompress calls
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

    auto h_input = make_smooth_data<float>(N);
    size_t in_bytes = N * sizeof(float);

    CudaBuffer<float> d_in(N);
    d_in.upload(h_input, stream);
    stream.sync();

    // Single PREALLOCATE pipeline — mirrors ARM A of profile_repeat
    Pipeline pipeline(in_bytes, MemoryStrategy::PREALLOCATE);

    auto* lorenzo = pipeline.addStage<LorenzoQuantStage<float, uint16_t>>();
    lorenzo->setErrorBound(EB);
    lorenzo->setQuantRadius(512);
    lorenzo->setOutlierCapacity(0.2f);

    auto* diff = pipeline.addStage<DifferenceStage<uint16_t>>();
    pipeline.connect(diff, lorenzo, "codes");

    pipeline.setPoolManagedDecompOutput(false);
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
// P9: repeated compress+decompress on the same input produces identical
// reconstructed output across runs.
// ─────────────────────────────────────────────────────────────────────────────
TEST(Pipeline, RepeatedCompressDecompressStableOutput) {
    CudaStream stream;
    constexpr size_t N      = 1 << 14;
    constexpr float  EB     = 1e-2f;
    constexpr int    N_RUNS = 3;

    auto h_input = make_smooth_data<float>(N);
    size_t in_bytes = N * sizeof(float);

    CudaBuffer<float> d_in(N);
    d_in.upload(h_input, stream);
    stream.sync();

    Pipeline pipeline(in_bytes, MemoryStrategy::PREALLOCATE);

    auto* lorenzo = pipeline.addStage<LorenzoQuantStage<float, uint16_t>>();
    lorenzo->setErrorBound(EB);
    lorenzo->setQuantRadius(512);
    lorenzo->setOutlierCapacity(0.2f);

    auto* diff = pipeline.addStage<DifferenceStage<uint16_t>>();
    pipeline.connect(diff, lorenzo, "codes");

    pipeline.setPoolManagedDecompOutput(false);
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
// P10: PREALLOCATE pipeline reused with different data on each call.
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

    auto* lorenzo = pipeline.addStage<LorenzoQuantStage<float, uint16_t>>();
    lorenzo->setErrorBound(EB);
    lorenzo->setQuantRadius(512);
    lorenzo->setOutlierCapacity(0.2f);

    auto* diff = pipeline.addStage<DifferenceStage<uint16_t>>();
    pipeline.connect(diff, lorenzo, "codes");

    pipeline.setPoolManagedDecompOutput(false);
    pipeline.finalize();

    std::vector<std::vector<float>> datasets = {
        make_smooth_data<float>(N),
        std::vector<float>(N, 3.14f),   // constant
        [&] { auto v = make_smooth_data<float>(N); for (auto& x : v) x *= 0.5f;    return v; }(),
        [&] { auto v = make_smooth_data<float>(N); for (auto& x : v) x += 100.0f;  return v; }(),
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
// P11: setDims() before addStage() immediately gives the correct ndim().
//
// addStage() now calls stage->setDims(dims_) before returning, so the stage
// reflects the pipeline's current dimensions without any helper wrapper.
// ─────────────────────────────────────────────────────────────────────────────
TEST(Pipeline, AddStageForwardsDims) {
    // 1D: default (no setDims call)
    {
        Pipeline p1(256 * sizeof(float), MemoryStrategy::MINIMAL);
        LorenzoQuantStage<float, uint16_t>::Config cfg1;
        cfg1.error_bound = 1e-2f;
        auto* lrz1 = p1.addStage<LorenzoQuantStage<float, uint16_t>>(cfg1);
        EXPECT_EQ(lrz1->ndim(), 1) << "Default should be 1D";
    }

    // 2D: setDims(nx, ny) before addStage
    {
        constexpr size_t NX = 32, NY = 32;
        Pipeline p2(NX * NY * sizeof(float), MemoryStrategy::MINIMAL);
        p2.setDims(NX, NY);
        LorenzoQuantStage<float, uint16_t>::Config cfg2;
        cfg2.error_bound = 1e-2f;
        auto* lrz2 = p2.addStage<LorenzoQuantStage<float, uint16_t>>(cfg2);
        EXPECT_EQ(lrz2->ndim(), 2) << "setDims(nx,ny) before addStage should give 2D";
        auto dims2 = lrz2->getDims();
        EXPECT_EQ(dims2[0], NX);
        EXPECT_EQ(dims2[1], NY);
        EXPECT_EQ(dims2[2], 1u);
    }

    // 3D: setDims(nx, ny, nz) before addStage
    {
        constexpr size_t NX = 16, NY = 16, NZ = 16;
        Pipeline p3(NX * NY * NZ * sizeof(float), MemoryStrategy::MINIMAL);
        p3.setDims(NX, NY, NZ);
        LorenzoQuantStage<float, uint16_t>::Config cfg3;
        cfg3.error_bound = 1e-2f;
        auto* lrz3 = p3.addStage<LorenzoQuantStage<float, uint16_t>>(cfg3);
        EXPECT_EQ(lrz3->ndim(), 3) << "setDims(nx,ny,nz) before addStage should give 3D";
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// P12: Explicit reset() between compress cycles.
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
    auto h_input = make_smooth_data<float>(N);
    CudaBuffer<float> d_in(N);
    d_in.upload(h_input, stream);
    stream.sync();

    Pipeline pipeline(in_bytes, MemoryStrategy::MINIMAL);
    auto* lrz = pipeline.addStage<LorenzoQuantStage<float, uint16_t>>();
    lrz->setErrorBound(EB);
    lrz->setQuantRadius(512);
    lrz->setOutlierCapacity(0.2f);
    pipeline.setPoolManagedDecompOutput(false);
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
// P13: compress() with varying input sizes across calls produces correct output
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
    auto h_large  = make_smooth_data<float>(N);
    auto h_small  = make_smooth_data<float>(N / 2);

    CudaBuffer<float> d_large(N);
    CudaBuffer<float> d_small(N / 2);
    d_large.upload(h_large, stream);
    d_small.upload(h_small, stream);
    stream.sync();

    // Hint = full N — large enough for all calls below.
    Pipeline pipeline(in_bytes, MemoryStrategy::MINIMAL);
    auto* lrz = pipeline.addStage<LorenzoQuantStage<float, uint16_t>>();
    lrz->setErrorBound(EB);
    lrz->setQuantRadius(512);
    lrz->setOutlierCapacity(0.2f);
    pipeline.setPoolManagedDecompOutput(false);
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
// P14: Large-scale compress + decompress (~100 MB)
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
    auto* lrz = pipeline.addStage<LorenzoQuantStage<float, uint16_t>>();
    lrz->setErrorBound(EB);
    lrz->setQuantRadius(512);
    lrz->setOutlierCapacity(0.05f);  // 5% outlier budget at 100 MB scale
    auto* rle = pipeline.addStage<RLEStage<uint16_t>>();
    pipeline.connect(rle, lrz, "codes");
    pipeline.setPoolManagedDecompOutput(false);
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
        << "P14: large-scale round-trip max_err=" << max_err
        << " exceeds bound " << EB;
}

// ─────────────────────────────────────────────────────────────────────────────
// P15: 2D Lorenzo round-trip (64×64 grid)
//
// Verifies that setDims(NX, NY) before addStage correctly configures the
// Lorenzo stage for 2D prediction, and that compress+decompress produces
// data within the error bound.
// ─────────────────────────────────────────────────────────────────────────────
TEST(Pipeline, Lorenzo2DRoundTrip) {
    constexpr size_t NX = 64, NY = 64;  // 4096 floats, 2D
    constexpr size_t N  = NX * NY;
    constexpr float  EB = 1e-2f;
    const size_t in_bytes = N * sizeof(float);

    CudaStream stream;

    // Generate smooth 2D data laid out row-major: element at (i,j) uses
    // sin(i*0.05)*30 + cos(j*0.07)*20
    std::vector<float> h_input(N);
    for (size_t i = 0; i < NX; i++)
        for (size_t j = 0; j < NY; j++)
            h_input[i * NY + j] = std::sin(static_cast<float>(i) * 0.05f) * 30.0f
                                 + std::cos(static_cast<float>(j) * 0.07f) * 20.0f;

    Pipeline pipeline(in_bytes, MemoryStrategy::MINIMAL);
    pipeline.setDims(NX, NY);
    auto* lrz = pipeline.addStage<LorenzoQuantStage<float, uint16_t>>();
    lrz->setErrorBound(EB);
    lrz->setQuantRadius(512);
    lrz->setOutlierCapacity(0.2f);
    pipeline.setPoolManagedDecompOutput(false);
    pipeline.finalize();

    auto res = pipeline_round_trip<float>(pipeline, h_input, stream);

    ASSERT_EQ(res.data.size(), N);
    EXPECT_LE(static_cast<float>(res.max_error), EB * 1.01f)
        << "P15: 2D Lorenzo round-trip max_error=" << res.max_error
        << " exceeds bound " << EB;
}

// ─────────────────────────────────────────────────────────────────────────────
// P16: 3D Lorenzo round-trip (16×16×16 grid)
//
// Verifies that setDims(NX, NY, NZ) before addStage correctly configures the
// Lorenzo stage for 3D prediction, and that compress+decompress produces
// data within the error bound.
// ─────────────────────────────────────────────────────────────────────────────
TEST(Pipeline, Lorenzo3DRoundTrip) {
    constexpr size_t NX = 16, NY = 16, NZ = 16;  // 4096 floats, 3D
    constexpr size_t N  = NX * NY * NZ;
    constexpr float  EB = 1e-2f;
    const size_t in_bytes = N * sizeof(float);

    CudaStream stream;

    // Generate smooth 3D data laid out row-major: element at (i,j,k) uses
    // sin(i*0.1)*20 + cos(j*0.07)*15 + sin(k*0.05)*10
    std::vector<float> h_input(N);
    for (size_t i = 0; i < NX; i++)
        for (size_t j = 0; j < NY; j++)
            for (size_t k = 0; k < NZ; k++)
                h_input[i * NY * NZ + j * NZ + k] =
                    std::sin(static_cast<float>(i) * 0.1f)  * 20.0f
                  + std::cos(static_cast<float>(j) * 0.07f) * 15.0f
                  + std::sin(static_cast<float>(k) * 0.05f) * 10.0f;

    Pipeline pipeline(in_bytes, MemoryStrategy::MINIMAL);
    pipeline.setDims(NX, NY, NZ);
    auto* lrz = pipeline.addStage<LorenzoQuantStage<float, uint16_t>>();
    lrz->setErrorBound(EB);
    lrz->setQuantRadius(512);
    lrz->setOutlierCapacity(0.2f);
    pipeline.setPoolManagedDecompOutput(false);
    pipeline.finalize();

    auto res = pipeline_round_trip<float>(pipeline, h_input, stream);

    ASSERT_EQ(res.data.size(), N);
    EXPECT_LE(static_cast<float>(res.max_error), EB * 1.01f)
        << "P16: 3D Lorenzo round-trip max_error=" << res.max_error
        << " exceeds bound " << EB;
}
