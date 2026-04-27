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
// H3: sizeof(FZMHeaderCore) == 80 (grew from 72 in v3.1: +flags +data_checksum +header_checksum)
// ─────────────────────────────────────────────────────────────────────────────
TEST(FZMFormat, HeaderCoreSize) {
    EXPECT_EQ(sizeof(FZMHeaderCore), 80u);
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
    // Wrap in a nested scope so that stream/pipeline/pool destructors all fire
    // before the final cudaDeviceSynchronize().  Under compute-sanitizer the
    // CUB DeviceScan used by DifferenceStage inverse leaves deferred
    // instrumentation work; syncing after all destructors ensures that work
    // completes before the next test allocates from the device pool.
    {
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
    auto* lrz  = pipeline.addStage<LorenzoQuantStage<float, uint16_t>>();
    auto* diff = pipeline.addStage<DifferenceStage<uint16_t>>();
    lrz->setErrorBound(EB);
    lrz->setQuantRadius(512);
    lrz->setOutlierCapacity(0.2f);
    pipeline.connect(diff, lrz, "codes");
    pipeline.setPoolManagedDecompOutput(false);
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
    stream.sync();

    ASSERT_NE(d_dec, nullptr);
    EXPECT_EQ(dec_sz, in_bytes);

    std::vector<float> h_recon(N);
    FZ_TEST_CUDA(cudaMemcpy(h_recon.data(), d_dec, in_bytes, cudaMemcpyDeviceToHost));
    cudaFree(d_dec);

    float max_err = max_abs_error(h_input, h_recon);
    EXPECT_LE(max_err, EB * 1.01f)
        << "Lorenzo+Diff file round-trip max_err=" << max_err;

    std::remove(tmp.c_str());
    }  // stream, pipeline, pool all destroyed here
    // Flush any compute-sanitizer deferred work from the CUB-scan-based
    // DifferenceStage inverse path before the next test starts.
    cudaDeviceSynchronize();
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
    auto* lrz = pipeline.addStage<LorenzoQuantStage<float, uint16_t>>();
    lrz->setErrorBound(EB);
    lrz->setQuantRadius(512);
    lrz->setOutlierCapacity(0.2f);
    pipeline.setPoolManagedDecompOutput(false);
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
    auto* lrz = pipeline.addStage<LorenzoQuantStage<float, uint16_t>>();
    lrz->setErrorBound(EB);
    lrz->setQuantRadius(512);
    lrz->setOutlierCapacity(0.2f);
    pipeline.setPoolManagedDecompOutput(false);
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
        auto* lrz = pipeline.addStage<LorenzoQuantStage<float, uint16_t>>();
        lrz->setErrorBound(EB);
        lrz->setQuantRadius(512);
        lrz->setOutlierCapacity(0.2f);
        pipeline.setPoolManagedDecompOutput(false);
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
        auto* lrz = pipeline.addStage<LorenzoQuantStage<float, uint16_t>>();
        lrz->setErrorBound(EB);
        lrz->setOutlierCapacity(0.2f);
        pipeline.setPoolManagedDecompOutput(false);
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
        auto* lrz = pipeline.addStage<LorenzoQuantStage<float, uint16_t>>();
        lrz->setErrorBound(EB);
        lrz->setOutlierCapacity(0.2f);
        pipeline.setPoolManagedDecompOutput(false);
    pipeline.finalize();

        void*  d_comp = nullptr;
        size_t cmp_sz = 0;
        pipeline.compress(d_in.void_ptr(), in_bytes, &d_comp, &cmp_sz, stream);
        pipeline.writeToFile(tmp, stream);
    }

    // Patch byte[4] (low byte of uint16_t version = minor) to a value that
    // differs from FZM_VERSION_MINOR (currently 0x01).
    // Using 0x02 gives minor=0x02, major=0x03 → minor mismatch → warn only.
    // Note: corrupting the version byte also invalidates the header checksum,
    // but readHeader() skips header-checksum verification on minor mismatches.
    make_corrupt_file(tmp, 4, 0x02);

    void*  d_out  = nullptr;
    size_t out_sz = 0;
    ASSERT_NO_THROW(
        Pipeline::decompressFromFile(tmp, &d_out, &out_sz)
    ) << "Wrong minor version should NOT throw";
    // Flush compute-sanitizer's stream post-processing before the next test.
    cudaDeviceSynchronize();

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
// FC1: writeToFile() before compress() throws
// ─────────────────────────────────────────────────────────────────────────────
TEST(FileIO, WriteToFileBeforeCompressThrows) {
    constexpr size_t N = 1 << 12;
    const size_t in_bytes = N * sizeof(float);

    Pipeline pipeline(in_bytes, MemoryStrategy::MINIMAL);
    auto* lrz = pipeline.addStage<LorenzoQuantStage<float, uint16_t>>();
    lrz->setErrorBound(1e-2f);
    lrz->setQuantRadius(512);
    lrz->setOutlierCapacity(0.2f);
    pipeline.setPoolManagedDecompOutput(false);
    pipeline.finalize();

    // No compress() called — must throw
    EXPECT_THROW(
        pipeline.writeToFile("/tmp/fzgmod_should_not_exist.fzm"),
        std::runtime_error
    ) << "writeToFile() before compress() must throw";
}

// ─────────────────────────────────────────────────────────────────────────────
// FC2: buildHeader() before compress() throws
// ─────────────────────────────────────────────────────────────────────────────
TEST(FileIO, BuildHeaderBeforeCompressThrows) {
    constexpr size_t N = 1 << 12;
    const size_t in_bytes = N * sizeof(float);

    Pipeline pipeline(in_bytes, MemoryStrategy::MINIMAL);
    auto* lrz = pipeline.addStage<LorenzoQuantStage<float, uint16_t>>();
    lrz->setErrorBound(1e-2f);
    lrz->setQuantRadius(512);
    lrz->setOutlierCapacity(0.2f);
    pipeline.setPoolManagedDecompOutput(false);
    pipeline.finalize();

    EXPECT_THROW(
        pipeline.buildHeader(),
        std::runtime_error
    ) << "buildHeader() before compress() must throw";
}

// ─────────────────────────────────────────────────────────────────────────────
// FC3: Stage config preserved through writeToFile / readHeader
//
// After compress() + writeToFile(), readHeader() returns FZMStageInfo entries
// whose stage_config bytes can be deserialized back into stages with the same
// parameters (quant_radius, block_size, element_width, etc.).
// ─────────────────────────────────────────────────────────────────────────────
TEST(FileIO, StageConfigPreservedThroughFile) {
    CudaStream stream;
    constexpr size_t N  = 1 << 12;
    constexpr float  EB = 2e-3f;
    const size_t in_bytes = N * sizeof(float);
    auto h_input = make_smooth(N);

    CudaBuffer<float> d_in(N);
    d_in.upload(h_input, stream);
    stream.sync();

    const std::string tmp = "/tmp/fzgmod_test_stage_config.fzm";

    // Build: Lorenzo → Bitshuffle (two stages; Bitshuffle has serializable block+width)
    Pipeline pipeline(in_bytes, MemoryStrategy::MINIMAL, 5.0f);
    auto* lrz = pipeline.addStage<LorenzoQuantStage<float, uint16_t>>();
    lrz->setErrorBound(EB);
    lrz->setQuantRadius(256);
    lrz->setOutlierCapacity(0.15f);

    auto* bs = pipeline.addStage<BitshuffleStage>();
    bs->setBlockSize(8192);   // non-default block size
    bs->setElementWidth(2);   // non-default element width
    pipeline.connect(bs, lrz, "codes");

    pipeline.setPoolManagedDecompOutput(false);
    pipeline.finalize();

    void*  d_comp  = nullptr;
    size_t comp_sz = 0;
    pipeline.compress(d_in.void_ptr(), in_bytes, &d_comp, &comp_sz, stream);
    stream.sync();
    pipeline.writeToFile(tmp, stream);

    auto hdr = Pipeline::readHeader(tmp);

    // Find the Bitshuffle stage entry by type
    bool found_bs = false;
    for (const auto& si : hdr.stages) {
        if (si.stage_type == StageType::BITSHUFFLE && si.config_size >= 5) {
            uint32_t block_size_read = 0;
            uint8_t  elem_width_read = 0;
            std::memcpy(&block_size_read, si.stage_config, sizeof(uint32_t));
            elem_width_read = si.stage_config[4];

            EXPECT_EQ(block_size_read, 8192u)
                << "Bitshuffle block_size must survive writeToFile/readHeader";
            EXPECT_EQ(elem_width_read, 2u)
                << "Bitshuffle element_width must survive writeToFile/readHeader";
            found_bs = true;
            break;
        }
    }
    EXPECT_TRUE(found_bs) << "Bitshuffle stage not found in file header";

    // Find the Lorenzo buffer entry and verify quant_radius
    bool found_lrz = false;
    for (const auto& be : hdr.buffers) {
        if (be.stage_type == StageType::LORENZO_QUANT && be.config_size >= sizeof(LorenzoQuantConfig)) {
            LorenzoQuantConfig lc;
            std::memcpy(&lc, be.stage_config, sizeof(LorenzoQuantConfig));
            EXPECT_EQ(lc.quant_radius, 256u)
                << "Lorenzo quant_radius must survive writeToFile/readHeader";
            found_lrz = true;
            break;
        }
    }
    EXPECT_TRUE(found_lrz) << "Lorenzo buffer entry not found in file header";

    std::remove(tmp.c_str());
}

// ─────────────────────────────────────────────────────────────────────────────
// FC4: createStage() reconstructs stages with correct config from file header
//
// Reads the file header, calls createStage() for each stage entry, and
// verifies the reconstructed stage has the same parameters as the original.
// ─────────────────────────────────────────────────────────────────────────────
TEST(FileIO, CreateStageReconstructsCorrectConfig) {
    CudaStream stream;
    constexpr size_t N  = 1 << 12;
    constexpr float  EB = 5e-3f;
    const size_t in_bytes = N * sizeof(float);
    auto h_input = make_smooth(N);

    CudaBuffer<float> d_in(N);
    d_in.upload(h_input, stream);
    stream.sync();

    const std::string tmp = "/tmp/fzgmod_test_create_stage.fzm";

    // Build a Lorenzo pipeline with non-default settings
    Pipeline pipeline(in_bytes, MemoryStrategy::MINIMAL);
    auto* lrz = pipeline.addStage<LorenzoQuantStage<float, uint16_t>>();
    lrz->setErrorBound(EB);
    lrz->setQuantRadius(1024);
    lrz->setOutlierCapacity(0.1f);
    pipeline.setPoolManagedDecompOutput(false);
    pipeline.finalize();

    void*  d_comp  = nullptr;
    size_t comp_sz = 0;
    pipeline.compress(d_in.void_ptr(), in_bytes, &d_comp, &comp_sz, stream);
    stream.sync();
    pipeline.writeToFile(tmp, stream);

    auto hdr = Pipeline::readHeader(tmp);
    ASSERT_GE(hdr.stages.size(), 1u) << "Header must have at least one stage";

    // Reconstruct the Lorenzo stage via createStage()
    const auto& si = hdr.stages[0];
    ASSERT_EQ(si.stage_type, StageType::LORENZO_QUANT)
        << "First stage must be Lorenzo";

    std::unique_ptr<Stage> reconstructed(
        createStage(si.stage_type, si.stage_config, si.config_size)
    );
    ASSERT_NE(reconstructed.get(), nullptr)
        << "createStage() must return a valid stage";

    // Cast to access Lorenzo-specific config
    auto* recon_lrz = dynamic_cast<LorenzoQuantStage<float, uint16_t>*>(reconstructed.get());
    ASSERT_NE(recon_lrz, nullptr)
        << "createStage() for LORENZO must return LorenzoQuantStage<float,uint16_t>";

    EXPECT_EQ(recon_lrz->getQuantRadius(), static_cast<uint16_t>(1024))
        << "Reconstructed Lorenzo quant_radius mismatch";

    std::remove(tmp.c_str());
}

// ─────────────────────────────────────────────────────────────────────────────
// FC5: Lorenzo → RLE pipeline file round-trip
//
// Tests a different multi-stage topology (Lorenzo → RLE) through writeToFile /
// decompressFromFile to verify that RLE's cached_num_elements_ is correctly
// serialized and restored for correct inverse sizing.
// ─────────────────────────────────────────────────────────────────────────────
TEST(FileIO, LorenzoRLERoundTripThroughFile) {
    CudaStream stream;
    constexpr size_t N  = 1 << 13;
    constexpr float  EB = 1e-2f;
    const size_t in_bytes = N * sizeof(float);
    auto h_input = make_smooth(N);

    CudaBuffer<float> d_in(N);
    d_in.upload(h_input, stream);
    stream.sync();

    const std::string tmp = "/tmp/fzgmod_test_lorenzorle.fzm";

    Pipeline pipeline(in_bytes, MemoryStrategy::MINIMAL);
    auto* lrz = pipeline.addStage<LorenzoQuantStage<float, uint16_t>>();
    lrz->setErrorBound(EB);
    lrz->setQuantRadius(512);
    lrz->setOutlierCapacity(0.2f);
    auto* rle = pipeline.addStage<RLEStage<uint16_t>>();
    pipeline.connect(rle, lrz, "codes");
    pipeline.setPoolManagedDecompOutput(false);
    pipeline.finalize();

    void*  d_comp  = nullptr;
    size_t comp_sz = 0;
    pipeline.compress(d_in.void_ptr(), in_bytes, &d_comp, &comp_sz, stream);
    stream.sync();
    ASSERT_GT(comp_sz, 0u);
    pipeline.writeToFile(tmp, stream);

    void*  d_dec  = nullptr;
    size_t dec_sz = 0;
    ASSERT_NO_THROW(Pipeline::decompressFromFile(tmp, &d_dec, &dec_sz, stream));
    stream.sync();
    ASSERT_NE(d_dec, nullptr);
    ASSERT_EQ(dec_sz, in_bytes);

    std::vector<float> h_recon(N);
    FZ_TEST_CUDA(cudaMemcpy(h_recon.data(), d_dec, in_bytes, cudaMemcpyDeviceToHost));
    cudaFree(d_dec);

    float max_err = max_abs_error(h_input, h_recon);
    EXPECT_LE(max_err, EB * 1.01f)
        << "Lorenzo+RLE file round-trip max_err=" << max_err;

    std::remove(tmp.c_str());
}

// ─────────────────────────────────────────────────────────────────────────────
// FC6: decompressFromFile() with perf_out populates valid timing data
//
// When a non-null PipelinePerfResult* is passed, the struct is populated.
// We only verify that the fields are structurally valid (positive times, correct
// direction flag, non-zero byte counts) — exact timing values are hardware-dependent.
// ─────────────────────────────────────────────────────────────────────────────
TEST(FileIO, DecompressFromFilePerfOutPopulated) {
    CudaStream stream;
    constexpr size_t N  = 1 << 12;
    constexpr float  EB = 1e-2f;
    const size_t in_bytes = N * sizeof(float);
    auto h_input = make_smooth(N);

    CudaBuffer<float> d_in(N);
    d_in.upload(h_input, stream);
    stream.sync();

    const std::string tmp = "/tmp/fzgmod_test_perf.fzm";

    Pipeline pipeline(in_bytes, MemoryStrategy::MINIMAL);
    auto* lrz = pipeline.addStage<LorenzoQuantStage<float, uint16_t>>();
    lrz->setErrorBound(EB);
    lrz->setQuantRadius(512);
    lrz->setOutlierCapacity(0.2f);
    pipeline.setPoolManagedDecompOutput(false);
    pipeline.finalize();

    void*  d_comp  = nullptr;
    size_t comp_sz = 0;
    pipeline.compress(d_in.void_ptr(), in_bytes, &d_comp, &comp_sz, stream);
    stream.sync();
    pipeline.writeToFile(tmp, stream);

    void*  d_dec  = nullptr;
    size_t dec_sz = 0;
    PipelinePerfResult perf;
    ASSERT_NO_THROW(Pipeline::decompressFromFile(tmp, &d_dec, &dec_sz, stream, &perf));
    stream.sync();
    ASSERT_NE(d_dec, nullptr);
    cudaFree(d_dec);

    EXPECT_FALSE(perf.is_compress)
        << "perf.is_compress must be false for a decompress call";
    EXPECT_GT(perf.host_elapsed_ms, 0.0f)
        << "perf.host_elapsed_ms must be positive";
    EXPECT_GT(perf.output_bytes, 0u)
        << "perf.output_bytes must be positive (decompressed size)";

    std::remove(tmp.c_str());
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

// ─────────────────────────────────────────────────────────────────────────────
// CK1: Written FZM file has both checksum flags set and non-zero checksums
// ─────────────────────────────────────────────────────────────────────────────
TEST(FileIO, WrittenFileHasChecksums) {
    CudaStream stream;
    constexpr size_t N  = 1 << 12;
    constexpr float  EB = 1e-2f;
    const size_t in_bytes = N * sizeof(float);
    auto h_input = make_smooth(N);

    CudaBuffer<float> d_in(N);
    d_in.upload(h_input, stream);
    stream.sync();

    const std::string tmp = "/tmp/fzgmod_test_checksums.fzm";

    Pipeline pipeline(in_bytes, MemoryStrategy::MINIMAL);
    auto* lrz = pipeline.addStage<LorenzoQuantStage<float, uint16_t>>();
    lrz->setErrorBound(EB);
    lrz->setQuantRadius(512);
    lrz->setOutlierCapacity(0.2f);
    pipeline.setPoolManagedDecompOutput(false);
    pipeline.finalize();

    void*  d_comp = nullptr;
    size_t comp_sz = 0;
    pipeline.compress(d_in.void_ptr(), in_bytes, &d_comp, &comp_sz, stream);
    pipeline.writeToFile(tmp, stream);

    auto hdr = Pipeline::readHeader(tmp);

    EXPECT_TRUE(hdr.core.flags & FZM_FLAG_HAS_DATA_CHECKSUM)
        << "FZM_FLAG_HAS_DATA_CHECKSUM must be set in written file";
    EXPECT_TRUE(hdr.core.flags & FZM_FLAG_HAS_HEADER_CHECKSUM)
        << "FZM_FLAG_HAS_HEADER_CHECKSUM must be set in written file";
    EXPECT_NE(hdr.core.data_checksum,   0u) << "data_checksum must be non-zero";
    EXPECT_NE(hdr.core.header_checksum, 0u) << "header_checksum must be non-zero";

    // Round-trip still works (checksums pass verification)
    void*  d_dec = nullptr;
    size_t dec_sz = 0;
    EXPECT_NO_THROW(Pipeline::decompressFromFile(tmp, &d_dec, &dec_sz, stream));
    if (d_dec) cudaFree(d_dec);

    std::remove(tmp.c_str());
}

// ─────────────────────────────────────────────────────────────────────────────
// CK2: Corrupted compressed data payload → decompressFromFile throws with
//      a checksum error (not a silent bad decompression result).
// ─────────────────────────────────────────────────────────────────────────────
TEST(FileIO, CorruptedDataPayloadThrows) {
    CudaStream stream;
    constexpr size_t N  = 1 << 12;
    constexpr float  EB = 1e-2f;
    const size_t in_bytes = N * sizeof(float);
    auto h_input = make_smooth(N);

    CudaBuffer<float> d_in(N);
    d_in.upload(h_input, stream);
    stream.sync();

    const std::string tmp = "/tmp/fzgmod_test_corrupt_data.fzm";

    {
        Pipeline pipeline(in_bytes, MemoryStrategy::MINIMAL);
        auto* lrz = pipeline.addStage<LorenzoQuantStage<float, uint16_t>>();
        lrz->setErrorBound(EB);
        lrz->setQuantRadius(512);
        lrz->setOutlierCapacity(0.2f);
        pipeline.setPoolManagedDecompOutput(false);
    pipeline.finalize();

        void*  d_comp = nullptr;
        size_t comp_sz = 0;
        pipeline.compress(d_in.void_ptr(), in_bytes, &d_comp, &comp_sz, stream);
        pipeline.writeToFile(tmp, stream);
    }

    // Determine the data payload start (= header_size bytes into the file).
    auto hdr = Pipeline::readHeader(tmp);
    size_t payload_start = static_cast<size_t>(hdr.core.header_size);

    // Corrupt the first byte of the compressed data payload.
    {
        std::fstream f(tmp, std::ios::in | std::ios::out | std::ios::binary);
        ASSERT_TRUE(f.is_open());
        f.seekp(static_cast<std::streamoff>(payload_start));
        char c = 0;
        f.read(&c, 1);
        f.seekp(static_cast<std::streamoff>(payload_start));
        c ^= 0xFF;
        f.write(&c, 1);
    }

    void*  d_out  = nullptr;
    size_t out_sz = 0;
    EXPECT_THROW(
        Pipeline::decompressFromFile(tmp, &d_out, &out_sz),
        std::runtime_error
    ) << "Corrupted data payload must throw (checksum mismatch)";

    if (d_out) cudaFree(d_out);
    std::remove(tmp.c_str());
}

// ─────────────────────────────────────────────────────────────────────────────
// DI1: Lorenzo 2D dims survive writeToFile → decompressFromFile
//
// Compress a 64×64 float array with setDims(64, 64), write to file, and
// decompress from file.  Verifies that:
//   (a) dim_x / dim_y are embedded in the serialized LorenzoQuantConfig in the header
//   (b) decompressFromFile produces correct output (dims were restored via
//       deserializeHeader — if they weren't, the 2D kernel uses wrong strides
//       and produces garbage that fails the error-bound check).
// ─────────────────────────────────────────────────────────────────────────────
TEST(FileIO, LorenzoDimsSerializedThroughFile) {
    CudaStream stream;
    constexpr size_t DIM_X = 64;
    constexpr size_t DIM_Y = 64;
    constexpr size_t N     = DIM_X * DIM_Y;
    constexpr float  EB    = 1e-2f;
    const size_t in_bytes  = N * sizeof(float);
    auto h_input = make_smooth(N);

    CudaBuffer<float> d_in(N);
    d_in.upload(h_input, stream);
    stream.sync();

    const std::string tmp = "/tmp/fzgmod_test_dims_serial.fzm";

    // Compress with explicit 2D dims
    Pipeline pipeline(in_bytes, MemoryStrategy::MINIMAL);
    pipeline.setDims(DIM_X, DIM_Y);
    auto* lrz = pipeline.addStage<LorenzoQuantStage<float, uint16_t>>();
    lrz->setErrorBound(EB);
    lrz->setQuantRadius(512);
    lrz->setOutlierCapacity(0.2f);
    pipeline.setPoolManagedDecompOutput(false);
    pipeline.finalize();

    void*  d_comp  = nullptr;
    size_t comp_sz = 0;
    pipeline.compress(d_in.void_ptr(), in_bytes, &d_comp, &comp_sz, stream);
    stream.sync();
    ASSERT_GT(comp_sz, 0u);
    pipeline.writeToFile(tmp, stream);

    // (a) Verify dim_x / dim_y are in the serialized header.
    //     ndim is stored inside LorenzoQuantConfig, not encoded in the StageType.
    auto hdr = Pipeline::readHeader(tmp);
    bool found_2d = false;
    for (const auto& be : hdr.buffers) {
        const bool is_lorenzo = (be.stage_type == StageType::LORENZO_QUANT);
        if (is_lorenzo && be.config_size >= sizeof(LorenzoQuantConfig)) {
            LorenzoQuantConfig lc;
            std::memcpy(&lc, be.stage_config, sizeof(LorenzoQuantConfig));
            if (lc.dim_x == DIM_X && lc.dim_y == DIM_Y) {
                found_2d = true;
                break;
            }
        }
    }
    EXPECT_TRUE(found_2d)
        << "Lorenzo dim_x=" << DIM_X << " dim_y=" << DIM_Y
        << " must be stored in the file header LorenzoQuantConfig";

    // (b) Decompress from file and verify correctness
    void*  d_dec  = nullptr;
    size_t dec_sz = 0;
    ASSERT_NO_THROW(Pipeline::decompressFromFile(tmp, &d_dec, &dec_sz, stream))
        << "decompressFromFile on 2D Lorenzo file must not throw";
    stream.sync();
    ASSERT_NE(d_dec, nullptr);
    ASSERT_EQ(dec_sz, in_bytes);

    std::vector<float> h_recon(N);
    FZ_TEST_CUDA(cudaMemcpy(h_recon.data(), d_dec, in_bytes, cudaMemcpyDeviceToHost));
    cudaFree(d_dec);

    float max_err = max_abs_error(h_input, h_recon);
    EXPECT_LE(max_err, EB * 1.01f)
        << "Lorenzo 2D dims file round-trip max_err=" << max_err;

    std::remove(tmp.c_str());
}

// ─────────────────────────────────────────────────────────────────────────────
// PS1: decompressFromFile() with pool_override_bytes uses caller-supplied size
//
// Compress → write, read the header to compute the pool size using the same
// formula decompressFromFile uses internally (C + 2.5×max_U + 32 MiB), then
// call decompressFromFile() with that value via pool_override_bytes.
// This confirms the escape hatch is wired up correctly and the formula produces
// a valid pool size for a standard single-source pipeline.
// ─────────────────────────────────────────────────────────────────────────────
TEST(FileIO, PoolOverrideBytesWorks) {
    CudaStream stream;
    constexpr size_t N  = 1 << 13;
    constexpr float  EB = 1e-2f;
    const size_t in_bytes = N * sizeof(float);
    auto h_input = make_smooth(N);

    CudaBuffer<float> d_in(N);
    d_in.upload(h_input, stream);
    stream.sync();

    const std::string tmp = "/tmp/fzgmod_test_pool_override.fzm";

    Pipeline pipeline(in_bytes, MemoryStrategy::MINIMAL);
    auto* lrz = pipeline.addStage<LorenzoQuantStage<float, uint16_t>>();
    lrz->setErrorBound(EB);
    lrz->setQuantRadius(512);
    lrz->setOutlierCapacity(0.2f);
    pipeline.setPoolManagedDecompOutput(false);
    pipeline.finalize();

    void*  d_comp  = nullptr;
    size_t comp_sz = 0;
    pipeline.compress(d_in.void_ptr(), in_bytes, &d_comp, &comp_sz, stream);
    stream.sync();
    ASSERT_GT(comp_sz, 0u);
    pipeline.writeToFile(tmp, stream);

    // Compute pool size using the internal formula:
    //   C + 2.5 * max_stage_uncompressed + 32 MiB
    auto hdr = Pipeline::readHeader(tmp);
    const size_t C = static_cast<size_t>(hdr.core.compressed_size);
    size_t max_U   = static_cast<size_t>(hdr.core.uncompressed_size);
    for (const auto& buf : hdr.buffers)
        max_U = std::max(max_U, static_cast<size_t>(buf.uncompressed_size));
    const size_t pool_bytes = C
        + static_cast<size_t>(2.5 * static_cast<double>(max_U))
        + 32ULL * 1024 * 1024;

    void*  d_dec  = nullptr;
    size_t dec_sz = 0;
    ASSERT_NO_THROW(
        Pipeline::decompressFromFile(tmp, &d_dec, &dec_sz, stream,
                                     /*perf_out=*/nullptr, pool_bytes)
    ) << "decompressFromFile with formula-derived pool_override_bytes must succeed";
    stream.sync();
    ASSERT_NE(d_dec, nullptr);
    ASSERT_EQ(dec_sz, in_bytes);

    std::vector<float> h_recon(N);
    FZ_TEST_CUDA(cudaMemcpy(h_recon.data(), d_dec, in_bytes, cudaMemcpyDeviceToHost));
    cudaFree(d_dec);

    float max_err = max_abs_error(h_input, h_recon);
    EXPECT_LE(max_err, EB * 1.01f)
        << "Pool override round-trip max_err=" << max_err;

    std::remove(tmp.c_str());
}
