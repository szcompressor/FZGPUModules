/**
 * tests/test_file_io.cpp
 *
 * Tests for the FZM file format: writing, reading, round-trip correctness,
 * header integrity, and robust rejection of malformed or missing files.
 *
 * Also includes static-assertion style struct-size tests for format structs
 * (H2, H3) that require no GPU work.
 */

#include <gtest/gtest.h>
#include "helpers/fz_test_utils.h"
#include "fzgpumodules.h"
#include "fzm_format.h"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
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

// ─────────────────────────────────────────────────────────────────────────────
// H2: sizeof(FZMBufferEntry) == 256
// ─────────────────────────────────────────────────────────────────────────────
TEST(FZMFormat, BufferEntrySize) {
    // The static_assert in fzm_format.h already catches compile-time failures;
    // this runtime test makes the expectation explicit in the test suite output.
    EXPECT_EQ(sizeof(FZMBufferEntry), 256u);
}

// ─────────────────────────────────────────────────────────────────────────────
// H3: sizeof(FZMHeaderCore) == 72
// ─────────────────────────────────────────────────────────────────────────────
TEST(FZMFormat, HeaderCoreSize) {
    EXPECT_EQ(sizeof(FZMHeaderCore), 72u);
}

// ─────────────────────────────────────────────────────────────────────────────
// H5: Default-constructed FZMHeaderCore has the correct magic value
// ─────────────────────────────────────────────────────────────────────────────
TEST(FZMFormat, HeaderCoreMagic) {
    FZMHeaderCore h;
    EXPECT_EQ(h.magic,   FZM_MAGIC);
    EXPECT_EQ(h.version, FZM_VERSION);
}

// ─────────────────────────────────────────────────────────────────────────────
// F2: Lorenzo → Difference pipeline: writeToFile → decompressFromFile
// ─────────────────────────────────────────────────────────────────────────────
TEST(FileIO, LorenzoDiffRoundTrip) {
    CudaStream stream;
    constexpr size_t N  = 1 << 13;
    constexpr float  EB = 1e-2f;
    size_t in_bytes = N * sizeof(float);
    auto h_input = make_smooth(N);

    CudaBuffer<float> d_in(N);
    d_in.upload(h_input, stream);
    stream.sync();

    const std::string tmp = "/tmp/fzgmod_test_lorenzodiff.fzm";

    // ── Compress ────────────────────────────────────────────────────────────
    Pipeline pipeline(in_bytes, MemoryStrategy::MINIMAL);
    auto* lrz  = pipeline.addStage<LorenzoStage<float, uint16_t>>();
    auto* diff = pipeline.addStage<DifferenceStage<uint16_t>>();
    lrz->setErrorBound(EB);
    lrz->setQuantRadius(512);
    lrz->setOutlierCapacity(0.2f);
    pipeline.connect(diff, lrz, "codes");
    pipeline.finalize();

    void*  d_comp = nullptr;
    size_t comp_sz = 0;
    pipeline.compress(d_in.void_ptr(), in_bytes, &d_comp, &comp_sz, stream);
    ASSERT_GT(comp_sz, 0u);
    pipeline.writeToFile(tmp, stream);

    // ── Decompress from file ─────────────────────────────────────────────────
    void*  d_dec = nullptr;
    size_t dec_sz = 0;
    Pipeline::decompressFromFile(tmp, &d_dec, &dec_sz, stream);

    ASSERT_NE(d_dec, nullptr);
    EXPECT_EQ(dec_sz, in_bytes);

    std::vector<float> h_recon(N);
    FZ_TEST_CUDA(cudaMemcpy(h_recon.data(), d_dec, in_bytes, cudaMemcpyDeviceToHost));
    cudaFree(d_dec);

    float max_err = max_abs_error(h_input, h_recon);
    EXPECT_LE(max_err, EB * 1.01f)
        << "Lorenzo+Diff file round-trip max_err=" << max_err;

    std::remove(tmp.c_str());
}

// ─────────────────────────────────────────────────────────────────────────────
// F3: Compressed file size < raw input size for smooth sinusoidal data
// ─────────────────────────────────────────────────────────────────────────────
TEST(FileIO, CompressedFileSmallerThanRaw) {
    CudaStream stream;
    constexpr size_t N  = 1 << 14;
    constexpr float  EB = 1e-2f;
    size_t in_bytes = N * sizeof(float);
    auto h_input = make_smooth(N);

    CudaBuffer<float> d_in(N);
    d_in.upload(h_input, stream);
    stream.sync();

    const std::string tmp = "/tmp/fzgmod_test_filesize.fzm";

    Pipeline pipeline(in_bytes, MemoryStrategy::MINIMAL);
    auto* lrz = pipeline.addStage<LorenzoStage<float, uint16_t>>();
    lrz->setErrorBound(EB);
    lrz->setQuantRadius(512);
    lrz->setOutlierCapacity(0.2f);
    pipeline.finalize();

    void*  d_comp = nullptr;
    size_t comp_sz = 0;
    pipeline.compress(d_in.void_ptr(), in_bytes, &d_comp, &comp_sz, stream);
    pipeline.writeToFile(tmp, stream);

    // Measure on-disk file size
    std::ifstream f(tmp, std::ios::binary | std::ios::ate);
    ASSERT_TRUE(f.is_open()) << "Failed to open temp file: " << tmp;
    size_t file_sz = static_cast<size_t>(f.tellg());
    f.close();

    EXPECT_LT(file_sz, in_bytes)
        << "File (" << file_sz << " B) should be smaller than raw data ("
        << in_bytes << " B) for smooth sinusoidal input";

    std::remove(tmp.c_str());
}

// ─────────────────────────────────────────────────────────────────────────────
// F4: readHeader() returns the expected stage and buffer counts
// ─────────────────────────────────────────────────────────────────────────────
TEST(FileIO, ReadHeaderCounts) {
    CudaStream stream;
    constexpr size_t N  = 1 << 12;
    constexpr float  EB = 1e-2f;
    size_t in_bytes = N * sizeof(float);
    auto h_input = make_smooth(N);

    CudaBuffer<float> d_in(N);
    d_in.upload(h_input, stream);
    stream.sync();

    const std::string tmp = "/tmp/fzgmod_test_header.fzm";

    // Single-stage Lorenzo pipeline
    Pipeline pipeline(in_bytes, MemoryStrategy::MINIMAL);
    auto* lrz = pipeline.addStage<LorenzoStage<float, uint16_t>>();
    lrz->setErrorBound(EB);
    lrz->setQuantRadius(512);
    lrz->setOutlierCapacity(0.2f);
    pipeline.finalize();

    void*  d_comp = nullptr;
    size_t comp_sz = 0;
    pipeline.compress(d_in.void_ptr(), in_bytes, &d_comp, &comp_sz, stream);
    pipeline.writeToFile(tmp, stream);

    auto header = Pipeline::readHeader(tmp);

    EXPECT_EQ(header.core.magic, FZM_MAGIC)
        << "Magic number mismatch in readHeader";
    EXPECT_GE(header.core.num_stages,  1u)
        << "Expected at least 1 stage entry";
    EXPECT_GE(header.core.num_buffers, 1u)
        << "Expected at least 1 buffer entry";
    EXPECT_GT(header.core.uncompressed_size, 0u)
        << "uncompressed_size should be > 0";
    EXPECT_GT(header.core.compressed_size, 0u)
        << "compressed_size should be > 0";

    // Stage array and buffer array must match the header counts
    EXPECT_EQ(header.stages.size(),  header.core.num_stages);
    EXPECT_EQ(header.buffers.size(), header.core.num_buffers);

    std::remove(tmp.c_str());
}

// ─────────────────────────────────────────────────────────────────────────────
// FE3: decompressFromFile() on a file with a corrupted magic number throws
// ─────────────────────────────────────────────────────────────────────────────
TEST(FileIO, DecompressCorruptMagicThrows) {
    CudaStream stream;
    constexpr size_t N  = 1 << 12;
    constexpr float  EB = 1e-2f;
    const std::string tmp = "/tmp/fzgmod_test_corrupt_magic.fzm";

    // Build a valid compressed file first
    auto h_input = make_smooth(N);
    size_t in_bytes = N * sizeof(float);
    CudaBuffer<float> d_in(N);
    d_in.upload(h_input, stream);
    stream.sync();

    {
        Pipeline pipeline(in_bytes, MemoryStrategy::MINIMAL);
        auto* lrz = pipeline.addStage<LorenzoStage<float, uint16_t>>();
        lrz->setErrorBound(EB);
        lrz->setQuantRadius(512);
        lrz->setOutlierCapacity(0.2f);
        pipeline.finalize();

        void*  d_comp = nullptr;
        size_t cmp_sz = 0;
        pipeline.compress(d_in.void_ptr(), in_bytes, &d_comp, &cmp_sz, stream);
        pipeline.writeToFile(tmp, stream);
    }

    // Corrupt the first 4 bytes (the FZM_MAGIC field)
    make_corrupt_file(tmp, 0, 0xDE);
    make_corrupt_file(tmp, 1, 0xAD);
    make_corrupt_file(tmp, 2, 0xBE);
    make_corrupt_file(tmp, 3, 0xEF);

    void*  d_out  = nullptr;
    size_t out_sz = 0;
    EXPECT_THROW(
        Pipeline::decompressFromFile(tmp, &d_out, &out_sz),
        std::exception
    );

    if (d_out) cudaFree(d_out);
    std::remove(tmp.c_str());
}

// ─────────────────────────────────────────────────────────────────────────────
// FE4: decompressFromFile() on a file truncated to 10 bytes throws
// ─────────────────────────────────────────────────────────────────────────────
TEST(FileIO, DecompressTruncatedFileThrows) {
    const std::string tmp = "/tmp/fzgmod_test_truncated.fzm";

    // Write a 10-byte stub (far too small to be a valid FZM header)
    write_stub_file(tmp, 10);

    void*  d_out  = nullptr;
    size_t out_sz = 0;
    EXPECT_THROW(
        Pipeline::decompressFromFile(tmp, &d_out, &out_sz),
        std::exception
    );

    if (d_out) cudaFree(d_out);
    std::remove(tmp.c_str());
}

// ─────────────────────────────────────────────────────────────────────────────
// FE5a: decompressFromFile() on a file with a wrong MAJOR version throws
//
// The FZM version field is a uint16_t at byte-offset 4 of the header.
// On a little-endian host (byte[4]=low, byte[5]=high), patching byte[5] to a
// different value changes the major version, which must always throw.
// ─────────────────────────────────────────────────────────────────────────────
TEST(FileIO, DecompressWrongMajorVersionThrows) {
    CudaStream stream;
    constexpr size_t N      = 1 << 11;
    constexpr float  EB     = 1e-2f;
    const std::string tmp   = "/tmp/fzgmod_test_wrong_major.fzm";
    size_t in_bytes = N * sizeof(float);
    auto h_input = make_smooth(N);

    CudaBuffer<float> d_in(N);
    d_in.upload(h_input, stream);
    stream.sync();

    {
        Pipeline pipeline(in_bytes, MemoryStrategy::MINIMAL);
        auto* lrz = pipeline.addStage<LorenzoStage<float, uint16_t>>();
        lrz->setErrorBound(EB);
        lrz->setOutlierCapacity(0.2f);
        pipeline.finalize();

        void*  d_comp = nullptr;
        size_t cmp_sz = 0;
        pipeline.compress(d_in.void_ptr(), in_bytes, &d_comp, &cmp_sz, stream);
        pipeline.writeToFile(tmp, stream);
    }

    // uint16_t version is at byte offset 4 in FZMHeaderCore.
    // On a little-endian host: byte[4]=low (minor), byte[5]=high (major).
    // Patch byte[5] to change the major version to something wrong (0x05).
    make_corrupt_file(tmp, 5, 0x05);

    void*  d_out  = nullptr;
    size_t out_sz = 0;
    EXPECT_THROW(
        Pipeline::decompressFromFile(tmp, &d_out, &out_sz),
        std::runtime_error
    ) << "Wrong major version should throw";

    if (d_out) cudaFree(d_out);
    std::remove(tmp.c_str());
}

// ─────────────────────────────────────────────────────────────────────────────
// FE5b: decompressFromFile() on a file with a wrong MINOR version succeeds
//
// A minor version mismatch is backward/forward compatible: the reader issues
// a warning (via FZ_LOG) but does not throw.  The decompressed output must
// still be correct.
// ─────────────────────────────────────────────────────────────────────────────
TEST(FileIO, DecompressWrongMinorVersionSucceeds) {
    CudaStream stream;
    constexpr size_t N      = 1 << 11;
    constexpr float  EB     = 1e-2f;
    const std::string tmp   = "/tmp/fzgmod_test_wrong_minor.fzm";
    size_t in_bytes = N * sizeof(float);
    auto h_input = make_smooth(N);

    CudaBuffer<float> d_in(N);
    d_in.upload(h_input, stream);
    stream.sync();

    {
        Pipeline pipeline(in_bytes, MemoryStrategy::MINIMAL);
        auto* lrz = pipeline.addStage<LorenzoStage<float, uint16_t>>();
        lrz->setErrorBound(EB);
        lrz->setOutlierCapacity(0.2f);
        pipeline.finalize();

        void*  d_comp = nullptr;
        size_t cmp_sz = 0;
        pipeline.compress(d_in.void_ptr(), in_bytes, &d_comp, &cmp_sz, stream);
        pipeline.writeToFile(tmp, stream);
    }

    // Patch byte[4] (low byte of uint16_t version = minor) to a non-zero value.
    // This gives minor=0x01 while major stays 0x03 → minor mismatch → warn only.
    make_corrupt_file(tmp, 4, 0x01);

    void*  d_out  = nullptr;
    size_t out_sz = 0;
    ASSERT_NO_THROW(
        Pipeline::decompressFromFile(tmp, &d_out, &out_sz)
    ) << "Wrong minor version should NOT throw";

    // Output should still be valid and correctly sized.
    ASSERT_NE(d_out, nullptr);
    EXPECT_EQ(out_sz, in_bytes);

    std::vector<float> h_recon(N);
    FZ_TEST_CUDA(cudaMemcpy(h_recon.data(), d_out, in_bytes, cudaMemcpyDeviceToHost));
    cudaFree(d_out);

    float max_err = max_abs_error(h_input, h_recon);
    EXPECT_LE(max_err, EB * 1.01f)
        << "FE5b: round-trip after minor-version patch max_err=" << max_err;

    std::remove(tmp.c_str());
}
// ─────────────────────────────────────────────────────────────────────────────
TEST(FileIO, DecompressNonexistentPathThrows) {
    void*  d_out  = nullptr;
    size_t out_sz = 0;

    EXPECT_THROW(
        Pipeline::decompressFromFile(
            "/tmp/fzgmod_file_that_does_not_exist_abc123.fzm",
            &d_out, &out_sz
        ),
        std::exception  // runtime_error or ios_base::failure
    );

    if (d_out) cudaFree(d_out);
}

// ─────────────────────────────────────────────────────────────────────────────
// FE2: decompressFromFile() on a zero-byte file throws
// ─────────────────────────────────────────────────────────────────────────────
TEST(FileIO, DecompressEmptyFileThrows) {
    const std::string tmp = "/tmp/fzgmod_test_empty.fzm";

    // Create an empty file
    { std::ofstream f(tmp); }

    void*  d_out  = nullptr;
    size_t out_sz = 0;

    EXPECT_THROW(
        Pipeline::decompressFromFile(tmp, &d_out, &out_sz),
        std::exception
    );

    if (d_out) cudaFree(d_out);
    std::remove(tmp.c_str());
}
