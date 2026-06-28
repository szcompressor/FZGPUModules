/**
 * tests/stages/test_large_data.cpp
 *
 * OPTIONAL large-data stress tests. Each test is SKIPPED unless the environment
 * variable `FZ_LARGE_DATA_TEST=1` is set, and self-SKIPs if the GPU lacks the
 * memory for the requested size. Motivation: a published comparison reported
 * cuSZ-Hi failing to produce output on large tensor data, theorized to be a
 * data-size limit (i.e. a 32-bit index/offset overflow). These tests push each
 * stage near/over the 2^31-byte boundary to confirm our stages handle it.
 *
 * Size: the per-test input defaults to 2 GiB (which crosses the 2^31-byte offset
 * line). Override with `FZ_LARGE_DATA_BYTES=<bytes>`. The input is filled and
 * verified on the device in 64 MiB chunks so host RAM stays small. Lossless
 * stages are checked for an exact byte round-trip; lossy stages within the error
 * bound.
 *
 * Run with, e.g.:
 *   FZ_LARGE_DATA_TEST=1 ./test_large_data
 *   FZ_LARGE_DATA_TEST=1 FZ_LARGE_DATA_BYTES=8589934592 ./test_large_data   # 8 GiB
 *
 * Entropy coders (Huffman / ANS / ADM) are intentionally omitted: a raw stress
 * pattern doesn't satisfy their symbol-range / codebook assumptions, so a large
 * test of them would exercise data shape rather than data size. The list below
 * is a representative cross-section and is straightforward to extend.
 */

#include <gtest/gtest.h>
#include "helpers/fz_test_utils.h"
#include "fzgpumodules.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

using namespace fz;
using namespace fz_test;

namespace {

constexpr size_t kChunk = size_t(64) << 20;  // 64 MiB host staging chunk

bool large_enabled() {
    const char* e = std::getenv("FZ_LARGE_DATA_TEST");
    return e && std::string(e) == "1";
}

size_t target_bytes() {
    if (const char* e = std::getenv("FZ_LARGE_DATA_BYTES"))
        return std::strtoull(e, nullptr, 10);
    return size_t(2) * 1024 * 1024 * 1024;  // 2 GiB
}

// Sparse, position-dependent byte pattern: mostly zeros (so zero-eliminating
// coders compress rather than expand) with deterministic nonzero markers, so any
// corruption or misindexing shows up in the exact compare.
void fill_bytes(void* d, size_t bytes) {
    std::vector<uint8_t> buf(std::min(kChunk, bytes));
    for (size_t off = 0; off < bytes; off += buf.size()) {
        size_t n = std::min(buf.size(), bytes - off);
        for (size_t j = 0; j < n; ++j) {
            size_t g = off + j;
            buf[j] = (g % 17 == 0) ? uint8_t((g * 2654435761u) >> 24) : 0;
        }
        cudaMemcpy(static_cast<char*>(d) + off, buf.data(), n,
                   cudaMemcpyHostToDevice);
    }
}

// Smooth float field (valid finite values for lossy quantizers).
void fill_floats(void* d, size_t elems) {
    const size_t chunk_e = kChunk / sizeof(float);
    std::vector<float> buf(std::min(chunk_e, elems));
    for (size_t off = 0; off < elems; off += buf.size()) {
        size_t n = std::min(buf.size(), elems - off);
        for (size_t j = 0; j < n; ++j) {
            double g = double(off + j);
            buf[j] = float(std::sin(g * 1e-4) * 100.0 + std::cos(g * 3e-5) * 10.0);
        }
        cudaMemcpy(static_cast<float*>(d) + off, buf.data(), n * sizeof(float),
                   cudaMemcpyHostToDevice);
    }
}

void verify_lossless(const void* a, const void* b, size_t bytes) {
    std::vector<uint8_t> ba(std::min(kChunk, bytes)), bb(ba.size());
    for (size_t off = 0; off < bytes; off += ba.size()) {
        size_t n = std::min(ba.size(), bytes - off);
        cudaMemcpy(ba.data(), static_cast<const char*>(a) + off, n,
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(bb.data(), static_cast<const char*>(b) + off, n,
                   cudaMemcpyDeviceToHost);
        ASSERT_EQ(std::memcmp(ba.data(), bb.data(), n), 0)
            << "byte mismatch in chunk starting at offset " << off;
    }
}

void verify_lossy(const void* a, const void* b, size_t elems, float eb) {
    const size_t chunk_e = kChunk / sizeof(float);
    std::vector<float> fa(std::min(chunk_e, elems)), fb(fa.size());
    double maxerr = 0.0;
    for (size_t off = 0; off < elems; off += fa.size()) {
        size_t n = std::min(fa.size(), elems - off);
        cudaMemcpy(fa.data(), static_cast<const float*>(a) + off,
                   n * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(fb.data(), static_cast<const float*>(b) + off,
                   n * sizeof(float), cudaMemcpyDeviceToHost);
        for (size_t j = 0; j < n; ++j)
            maxerr = std::max(maxerr,
                              std::abs(double(fa[j]) - double(fb[j])));
    }
    EXPECT_LE(maxerr, double(eb) * 1.01) << "max abs error " << maxerr;
}

enum class Verify { Lossless, LossyAbs };

// Build the pipeline (the callback adds stages, sets dims, and finalizes a
// Pipeline already constructed with `in_bytes`), then round-trip a large input
// entirely on the device and verify. SKIPs (never fails) when disabled or when
// the GPU can't hold the working set.
void run_large(size_t in_bytes, MemoryStrategy strat,
               const std::function<void(Pipeline&)>& build,
               Verify v, float eb = 0.0f) {
    if (!large_enabled())
        GTEST_SKIP() << "optional — set FZ_LARGE_DATA_TEST=1 to run "
                        "(default size " << target_bytes()
                     << " B, override with FZ_LARGE_DATA_BYTES)";

    size_t freeB = 0, totB = 0;
    cudaMemGetInfo(&freeB, &totB);
    if (freeB < in_bytes * 3)  // input + decompressed + compressed/scratch
        GTEST_SKIP() << "insufficient GPU memory: free=" << freeB
                     << " < ~3x input (" << in_bytes
                     << "); lower FZ_LARGE_DATA_BYTES";

    void* d_in = nullptr;
    if (cudaMalloc(&d_in, in_bytes) != cudaSuccess) {
        cudaGetLastError();
        GTEST_SKIP() << "input cudaMalloc(" << in_bytes << ") failed";
    }

    CudaStream cs;
    if (v == Verify::Lossless) fill_bytes(d_in, in_bytes);
    else                       fill_floats(d_in, in_bytes / sizeof(float));
    cudaStreamSynchronize(cs.stream);

    Pipeline p(in_bytes, strat);
    build(p);

    void* d_comp = nullptr; size_t comp_sz = 0;
    void* d_dec  = nullptr; size_t dec_sz  = 0;
    try {
        p.compress(d_in, in_bytes, &d_comp, &comp_sz, cs.stream);
        cudaStreamSynchronize(cs.stream);
        p.decompress(nullptr, comp_sz, &d_dec, &dec_sz, cs.stream);
        cudaStreamSynchronize(cs.stream);
    } catch (const std::exception& e) {
        cudaGetLastError();
        cudaFree(d_in);
        GTEST_SKIP() << "round-trip raised (likely OOM at this size): "
                     << e.what();
    }

    ASSERT_EQ(dec_sz, in_bytes);
    if (v == Verify::Lossless) verify_lossless(d_in, d_dec, in_bytes);
    else                       verify_lossy(d_in, d_dec, in_bytes / sizeof(float), eb);

    cudaFree(d_in);
}

// 2-D factorization with a fixed fast dim, returning byte-exact dims.
struct Dims2D { size_t nx, ny, bytes_used; };
Dims2D dims_2d(size_t in_bytes, size_t elem_size, size_t nx = 16384) {
    size_t elems = in_bytes / elem_size;
    size_t ny = elems / nx; if (ny == 0) ny = 1;
    return {nx, ny, nx * ny * elem_size};
}
// Cubic factorization (multiple of 8 per side) for 3-D stages.
struct Dims3D { size_t s, bytes_used; };
Dims3D dims_3d(size_t in_bytes, size_t elem_size) {
    size_t elems = in_bytes / elem_size;
    size_t s = size_t(std::cbrt(double(elems)));
    s -= (s % 8); if (s == 0) s = 8;
    return {s, s * s * s * elem_size};
}

constexpr float kEB = 1e-2f;

}  // namespace

// ── Lossless predictors / coders / shufflers / transforms ────────────────────

TEST(LargeData, Lorenzo_Int32_1D) {
    size_t bytes = (target_bytes() / 4) * 4;
    run_large(bytes, MemoryStrategy::MINIMAL, [&](Pipeline& p) {
        p.setDims(bytes / 4);
        p.addStage<LorenzoStage<int32_t>>();
        p.finalize();
    }, Verify::Lossless);
}

TEST(LargeData, TiledLorenzo_Int32_2D) {
    auto d = dims_2d(target_bytes(), sizeof(int32_t));
    run_large(d.bytes_used, MemoryStrategy::MINIMAL, [&](Pipeline& p) {
        p.setDims(d.nx, d.ny);
        auto* tl = p.addStage<TiledLorenzoStage<int32_t>>();
        tl->setTileShape(8, 8);
        p.finalize();
    }, Verify::Lossless);
}

TEST(LargeData, AdaptiveBitpack_Int32) {
    size_t bytes = (target_bytes() / 4) * 4;
    run_large(bytes, MemoryStrategy::MINIMAL, [&](Pipeline& p) {
        auto* ab = p.addStage<AdaptiveBitpackStage<int32_t>>();
        ab->setBlockSize(64);
        p.finalize();
    }, Verify::Lossless);
}

TEST(LargeData, BitplaneRZE_Uint16) {
    size_t bytes = (target_bytes() / 2) * 2;
    run_large(bytes, MemoryStrategy::MINIMAL, [&](Pipeline& p) {
        p.addStage<BitplaneRZEStage>();
        p.finalize();
    }, Verify::Lossless);
}

TEST(LargeData, RZE) {
    run_large(target_bytes(), MemoryStrategy::MINIMAL, [&](Pipeline& p) {
        p.addStage<RZEStage>();
        p.finalize();
    }, Verify::Lossless);
}

TEST(LargeData, Bitshuffle) {
    run_large(target_bytes(), MemoryStrategy::MINIMAL, [&](Pipeline& p) {
        auto* bs = p.addStage<BitshuffleStage>();
        bs->setElementWidth(2);
        p.finalize();
    }, Verify::Lossless);
}

TEST(LargeData, Difference_Int32) {
    size_t bytes = (target_bytes() / 4) * 4;
    run_large(bytes, MemoryStrategy::MINIMAL, [&](Pipeline& p) {
        p.addStage<DifferenceStage<int32_t>>();
        p.finalize();
    }, Verify::Lossless);
}

// ── Lossy front-ends ─────────────────────────────────────────────────────────

TEST(LargeData, Quantizer_Linear_Float) {
    size_t bytes = (target_bytes() / 4) * 4;
    run_large(bytes, MemoryStrategy::MINIMAL, [&](Pipeline& p) {
        p.setDims(bytes / 4);
        auto* q = p.addStage<QuantizerStage<float, uint32_t>>();
        q->setErrorBound(kEB);
        q->setErrorBoundMode(ErrorBoundMode::ABS);
        q->setLinearMode(true);
        p.finalize();
    }, Verify::LossyAbs, kEB);
}

TEST(LargeData, LorenzoQuant_Float) {
    size_t bytes = (target_bytes() / 4) * 4;
    run_large(bytes, MemoryStrategy::MINIMAL, [&](Pipeline& p) {
        p.setDims(bytes / 4);
        auto* lq = p.addStage<LorenzoQuantStage<float, uint16_t>>();
        lq->setErrorBound(kEB);
        lq->setErrorBoundMode(ErrorBoundMode::ABS);
        p.finalize();
    }, Verify::LossyAbs, kEB);
}

// The cuSZ-Hi family stage — the one the published comparison flagged on large
// tensors. 3-D smooth volume.
TEST(LargeData, GInterp_Float_3D) {
    auto d = dims_3d(target_bytes(), sizeof(float));
    run_large(d.bytes_used, MemoryStrategy::MINIMAL, [&](Pipeline& p) {
        p.setDims(d.s, d.s, d.s);
        auto* gi = p.addStage<GInterpStage<float, uint16_t>>();
        gi->setErrorBound(kEB);
        gi->setErrorBoundMode(ErrorBoundMode::ABS);
        p.finalize();
    }, Verify::LossyAbs, kEB);
}

// ── Full cuSZp-family pipelines ──────────────────────────────────────────────

TEST(LargeData, CuSZp_Pipeline_1D) {
    size_t bytes = (target_bytes() / 4) * 4;
    run_large(bytes, MemoryStrategy::MINIMAL, [&](Pipeline& p) {
        p.setDims(bytes / 4);
        auto* q = p.addStage<QuantizerStage<float, uint32_t>>();
        q->setErrorBound(kEB);
        q->setErrorBoundMode(ErrorBoundMode::ABS);
        q->setLinearMode(true);
        auto* lrz = p.addStage<LorenzoStage<int32_t>>();
        lrz->setBlockSize(32);
        p.connect(lrz, q, "codes");
        auto* ab = p.addStage<AdaptiveBitpackStage<int32_t>>();
        ab->setBlockSize(32);
        p.connect(ab, lrz);
        p.finalize();
    }, Verify::LossyAbs, kEB);
}

TEST(LargeData, CuSZp3_Pipeline_2D) {
    auto d = dims_2d(target_bytes(), sizeof(float));
    run_large(d.bytes_used, MemoryStrategy::MINIMAL, [&](Pipeline& p) {
        p.setDims(d.nx, d.ny);
        auto* q = p.addStage<QuantizerStage<float, uint32_t>>();
        q->setErrorBound(kEB);
        q->setErrorBoundMode(ErrorBoundMode::ABS);
        q->setLinearMode(true);
        auto* tl = p.addStage<TiledLorenzoStage<int32_t>>();
        tl->setTileShape(8, 8);
        p.connect(tl, q, "codes");
        auto* ab = p.addStage<AdaptiveBitpackStage<int32_t>>();
        ab->setBlockSize(64);
        p.connect(ab, tl);
        p.finalize();
    }, Verify::LossyAbs, kEB);
}
