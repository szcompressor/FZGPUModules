/**
 * tests/pipeline/test_golden.cpp
 *
 * Golden-file regression tests for the FZM file format.
 *
 * These tests read committed reference files from tests/golden/ and verify:
 *
 *   GF1 VersionCurrent       — the golden file's version field matches the
 *                              current FZM_VERSION constant.  Fails when a
 *                              version bump is made without regenerating golden
 *                              files, prompting an explicit update.
 *
 *   GF2 DecompressesCorrectly — decompressFromFile on the reference file
 *                              produces output within the original error bound.
 *                              This is the backward-compatibility check: the
 *                              current code can read files written at that
 *                              version.
 *
 *   GF3 CompressedSizeStable — re-compressing the same input produces a
 *                              compressed size within ±5% of the golden file's
 *                              compressed size.  Large deviations indicate a
 *                              compression regression not caught by round-trip
 *                              correctness tests.
 *
 * If the golden file does not exist (e.g. on a fresh checkout before running
 * generate_golden_files), each test is skipped gracefully with GTEST_SKIP().
 *
 * Regenerate:
 *   cmake --build --preset release --target generate_golden_files
 *   git add tests/golden/ref_lorenzo_bitshuffle.fzm
 */

#include <gtest/gtest.h>
#include "helpers/stage_harness.h"
#include "fzgpumodules.h"
#include "fzm_format.h"
#include "predictors/lorenzo/lorenzo.h"
#include "transforms/bitshuffle/bitshuffle_stage.h"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

using namespace fz;
using namespace fz_test;

// ─────────────────────────────────────────────────────────────────────────────
// Parameters — must match tests/golden/regenerate.cpp exactly.
// ─────────────────────────────────────────────────────────────────────────────
static constexpr size_t N           = 1 << 12;
static constexpr float  EB          = 1e-2f;
static constexpr int    QRAD        = 512;
static constexpr float  OUTLIER_CAP = 0.2f;

// Path to golden file, relative to build tree — tests run with CWD = build dir,
// so we use an absolute source-tree path injected at configure time.
// If FZGM_GOLDEN_DIR is not defined (e.g. manual build), fall back to cwd.
#ifndef FZGM_GOLDEN_DIR
#  define FZGM_GOLDEN_DIR "tests/golden"
#endif

static const std::string kGoldenDir  = FZGM_GOLDEN_DIR;
static const std::string kGoldenFile = kGoldenDir + "/ref_lorenzo_bitshuffle.fzm";

// ─────────────────────────────────────────────────────────────────────────────
// Shared input: two-component sinusoid identical to regenerate.cpp.
// ─────────────────────────────────────────────────────────────────────────────
static std::vector<float> make_input() {
    return make_smooth_data<float>(N);
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper macro: build and finalize the reference pipeline inline.
// Pipeline is not copyable/movable, so we use a macro to stamp it out in-place.
// ─────────────────────────────────────────────────────────────────────────────
#define MAKE_REFERENCE_PIPELINE(name)                                    \
    Pipeline name(N * sizeof(float), MemoryStrategy::MINIMAL);          \
    do {                                                                  \
        auto* _lrz = name.addStage<LorenzoStage<float, uint16_t>>();    \
        _lrz->setErrorBound(EB);                                         \
        _lrz->setQuantRadius(QRAD);                                      \
        _lrz->setOutlierCapacity(OUTLIER_CAP);                           \
        auto* _bs = name.addStage<BitshuffleStage>();                    \
        name.connect(_bs, _lrz, "codes");                                \
        name.setPoolManagedDecompOutput(false);                          \
        name.finalize();                                                 \
    } while (0)

// ─────────────────────────────────────────────────────────────────────────────
// GF1: Golden file version matches current FZM_VERSION
// ─────────────────────────────────────────────────────────────────────────────
TEST(GoldenFile, VersionCurrent) {
    if (!std::filesystem::exists(kGoldenFile)) {
        GTEST_SKIP() << "Golden file not found: " << kGoldenFile
                     << "  Run: cmake --build --preset release"
                        " --target generate_golden_files";
    }

    auto hdr = Pipeline::readHeader(kGoldenFile);
    EXPECT_EQ(hdr.core.version, FZM_VERSION)
        << "Golden file version (0x" << std::hex << hdr.core.version
        << ") does not match current FZM_VERSION (0x" << FZM_VERSION << "). "
        << "Run generate_golden_files and commit the updated reference file.";
}

// ─────────────────────────────────────────────────────────────────────────────
// GF2: Decompressing the golden file produces output within the error bound
// ─────────────────────────────────────────────────────────────────────────────
TEST(GoldenFile, DecompressesCorrectly) {
    if (!std::filesystem::exists(kGoldenFile)) {
        GTEST_SKIP() << "Golden file not found: " << kGoldenFile;
    }

    CudaStream stream;
    auto h_input = make_input();

    void*  d_dec  = nullptr;
    size_t dec_sz = 0;
    ASSERT_NO_THROW(
        Pipeline::decompressFromFile(kGoldenFile, &d_dec, &dec_sz, stream)
    ) << "decompressFromFile failed on golden reference file";
    stream.sync();

    ASSERT_NE(d_dec, nullptr);
    ASSERT_EQ(dec_sz, N * sizeof(float))
        << "Decompressed size mismatch: expected " << N * sizeof(float)
        << " bytes, got " << dec_sz;

    std::vector<float> h_recon(N);
    cudaError_t cp_err = cudaMemcpy(h_recon.data(), d_dec, dec_sz, cudaMemcpyDeviceToHost);
    ASSERT_EQ(cp_err, cudaSuccess) << "cudaMemcpy: " << cudaGetErrorString(cp_err);
    cudaFree(d_dec);

    double max_err = max_abs_error_typed<float>(h_input, h_recon);
    EXPECT_LE(max_err, EB * 1.01)
        << "Golden file decompressed with max_error=" << max_err
        << " which exceeds the original error bound " << EB;
}

// ─────────────────────────────────────────────────────────────────────────────
// GF3: Re-compressing the same input produces a similar compressed size
// ─────────────────────────────────────────────────────────────────────────────
TEST(GoldenFile, CompressedSizeStable) {
    if (!std::filesystem::exists(kGoldenFile)) {
        GTEST_SKIP() << "Golden file not found: " << kGoldenFile;
    }

    CudaStream stream;
    auto h_input = make_input();

    // Reference compressed size from the golden file header
    auto hdr = Pipeline::readHeader(kGoldenFile);
    const size_t ref_compressed = static_cast<size_t>(hdr.core.compressed_size);
    ASSERT_GT(ref_compressed, 0u);

    // Re-compress with the identical pipeline configuration
    MAKE_REFERENCE_PIPELINE(p);
    auto res = pipeline_round_trip<float>(p, h_input, stream);

    const size_t new_compressed = res.compressed_bytes;
    ASSERT_GT(new_compressed, 0u);

    // Allow ±5% drift in compressed size (floating-point rounding differences
    // across CUDA versions are possible, but large changes indicate a regression).
    const double ratio = static_cast<double>(new_compressed) /
                         static_cast<double>(ref_compressed);
    EXPECT_GE(ratio, 0.95)
        << "Re-compressed size (" << new_compressed
        << " B) is more than 5% smaller than golden (" << ref_compressed << " B)";
    EXPECT_LE(ratio, 1.05)
        << "Re-compressed size (" << new_compressed
        << " B) is more than 5% larger than golden (" << ref_compressed << " B)";

    // Correctness is still required regardless of size
    EXPECT_LE(res.max_error, EB * 1.01)
        << "Re-compression round-trip error " << res.max_error
        << " exceeded error bound " << EB;
}
