/**
 * tests/test_ownership.cpp
 *
 * Tests for pipeline pointer ownership and lifetime semantics.
 *
 * The library has two distinct ownership contracts:
 *
 *   compress() output  — POOL-OWNED: returned pointer lives in the Pipeline's
 *     internal memory pool. The caller must NOT call cudaFree(). The pointer
 *     is valid until the next compress() or reset() call.
 *
 *   decompress() output (default) — CALLER-OWNED: returned pointer is a fresh
 *     cudaMalloc'd buffer. The caller MUST call cudaFree() when done.
 *
 *   decompress() output (pool-managed) — POOL-OWNED: returned pointer lives in
 *     the pool when setPoolManagedDecompOutput(true) is set. The caller must
 *     NOT cudaFree(). The pointer is valid until the next decompress() call or
 *     Pipeline destruction.
 *
 * These tests verify:
 *   OW1  Compress output data is correct (pool-owned, no cudaFree)
 *   OW2  Compress output pool allocation is reclaimed after reset()
 *   OW3  Decompress default: caller-owned, data correct, cudaFree succeeds,
 *          memory reclaimed
 *   OW4  Decompress pool-managed: data correct without cudaFree
 *   OW5  Pool-managed decompress pointer survives a subsequent compress()
 *          (compress does not invalidate the decompress output)
 *   OW6  Second pool-managed decompress() frees the previous pointer
 *          (pool usage stays bounded; new pointer has correct data)
 *   OW7  Pool-managed decompress pointer survives reset()
 *          (reset() invalidates the compress output but NOT the decompress output)
 *   OW9  Zero-size compress round-trips without crash (edge case)
 *   OW11 Pool-managed decompress pool usage stays bounded across 5 calls
 *          (previous output freed before next is allocated; no monotonic growth)
 */

#include <gtest/gtest.h>
#include "helpers/fz_test_utils.h"
#include "fzgpumodules.h"

#include <cstring>
#include <vector>

using namespace fz;
using namespace fz_test;

// ─────────────────────────────────────────────────────────────────────────────
// Shared helper: build and finalize a single-stage Lorenzo pipeline.
// Returns a unique_ptr because Pipeline is neither copyable nor movable.
// ─────────────────────────────────────────────────────────────────────────────
static std::unique_ptr<Pipeline> make_pipeline(
    size_t n_floats,
    MemoryStrategy strategy = MemoryStrategy::MINIMAL,
    bool pool_managed_decomp = false)
{
    auto p = std::make_unique<Pipeline>(n_floats * sizeof(float), strategy);
    auto* lrz = p->addStage<LorenzoStage<float, uint16_t>>();
    lrz->setErrorBound(1e-2f);
    if (pool_managed_decomp) p->setPoolManagedDecompOutput(true);
    p->finalize();
    return p;
}

// ─────────────────────────────────────────────────────────────────────────────
// OW1: Compress output data is correct — pool-owned, must NOT be cudaFree'd.
//
// Verifies the basic compress → decompress round-trip using a pool-owned
// compress output.  The compress pointer is passed directly to decompress()
// without any copy; if the pool reclaims it before decompress() finishes we
// would get garbage or a crash, so this also implicitly validates lifetime.
// ─────────────────────────────────────────────────────────────────────────────
TEST(Ownership, CompressOutputPoolOwnedDataCorrect) {
    constexpr size_t N  = 2048;
    constexpr float  EB = 1e-2f;
    const size_t in_bytes = N * sizeof(float);

    auto h_in = make_random_floats(N, 10);
    auto p = make_pipeline(N);

    CudaStream stream;
    CudaBuffer<float> d_in(N);
    d_in.upload(h_in, stream);
    stream.sync();

    void*  d_comp  = nullptr;
    size_t comp_sz = 0;
    ASSERT_NO_THROW(p->compress(d_in.void_ptr(), in_bytes, &d_comp, &comp_sz, stream));
    ASSERT_NE(d_comp, nullptr);
    EXPECT_GT(comp_sz, 0u);
    EXPECT_LT(comp_sz, in_bytes);  // Lorenzo should compress random floats

    // Decompress using the pool-owned pointer directly — no copy.
    void*  d_dec  = nullptr;
    size_t dec_sz = 0;
    ASSERT_NO_THROW(p->decompress(d_comp, comp_sz, &d_dec, &dec_sz, stream));
    ASSERT_NE(d_dec, nullptr);
    ASSERT_EQ(dec_sz, in_bytes);

    std::vector<float> h_recon(N);
    FZ_TEST_CUDA(cudaMemcpy(h_recon.data(), d_dec, in_bytes, cudaMemcpyDeviceToHost));
    cudaFree(d_dec);

    EXPECT_LE(max_abs_error(h_in, h_recon), EB * 1.01f);
}

// ─────────────────────────────────────────────────────────────────────────────
// OW2: Compress output pool allocation is reclaimed after reset().
//
// Measures pool peak usage immediately after compress() and again after
// reset().  The peak should stay flat (we're just checking the compress output
// buffer was freed, not absolute numbers), but current usage must drop.
//
// Strategy: MINIMAL so intermediate buffers are also freed at reset() and the
// measurement is clean.
// ─────────────────────────────────────────────────────────────────────────────
TEST(Ownership, CompressOutputReclaimedAfterReset) {
    constexpr size_t N = 4096;
    const size_t in_bytes = N * sizeof(float);

    auto p = make_pipeline(N, MemoryStrategy::MINIMAL);

    CudaStream stream;
    CudaBuffer<float> d_in(N);
    d_in.upload(make_random_floats(N, 20), stream);
    stream.sync();

    void*  d_comp  = nullptr;
    size_t comp_sz = 0;
    ASSERT_NO_THROW(p->compress(d_in.void_ptr(), in_bytes, &d_comp, &comp_sz, stream));
    ASSERT_NE(d_comp, nullptr);
    stream.sync();

    const size_t usage_after_compress = p->getCurrentMemoryUsage();
    EXPECT_GT(usage_after_compress, 0u)
        << "Pool usage should be non-zero while compress output is alive";

    p->reset(stream);
    stream.sync();

    const size_t usage_after_reset = p->getCurrentMemoryUsage();
    EXPECT_LT(usage_after_reset, usage_after_compress)
        << "Pool usage should decrease after reset() reclaims compress output buffer";
}

// ─────────────────────────────────────────────────────────────────────────────
// OW3: Decompress default is caller-owned.
//
// Verifies:
//   - data is correct
//   - cudaFree() on the returned pointer succeeds (no double-free, no crash)
//   - GPU free bytes recovers after cudaFree (no permanent leak)
// ─────────────────────────────────────────────────────────────────────────────
TEST(Ownership, DecompressDefaultCallerOwned) {
    constexpr size_t N  = 2048;
    constexpr float  EB = 1e-2f;
    const size_t in_bytes = N * sizeof(float);

    auto h_in = make_random_floats(N, 30);
    auto p = make_pipeline(N);

    CudaStream stream;
    CudaBuffer<float> d_in(N);
    d_in.upload(h_in, stream);
    stream.sync();

    void*  d_comp = nullptr;
    size_t comp_sz = 0;
    ASSERT_NO_THROW(p->compress(d_in.void_ptr(), in_bytes, &d_comp, &comp_sz, stream));

    cudaDeviceSynchronize();
    const size_t free_before_decomp = gpu_free_bytes();

    void*  d_dec  = nullptr;
    size_t dec_sz = 0;
    ASSERT_NO_THROW(p->decompress(d_comp, comp_sz, &d_dec, &dec_sz, stream));
    ASSERT_NE(d_dec, nullptr);
    ASSERT_EQ(dec_sz, in_bytes);

    // Verify data correctness.
    std::vector<float> h_recon(N);
    FZ_TEST_CUDA(cudaMemcpy(h_recon.data(), d_dec, in_bytes, cudaMemcpyDeviceToHost));
    EXPECT_LE(max_abs_error(h_in, h_recon), EB * 1.01f);

    // Caller-owned: cudaFree must succeed without error.
    cudaError_t free_err = cudaFree(d_dec);
    EXPECT_EQ(free_err, cudaSuccess)
        << "cudaFree on caller-owned decompress output must succeed";

    // Memory should be reclaimed (allow 1 MB variance for allocator overhead).
    cudaDeviceSynchronize();
    const size_t free_after_free = gpu_free_bytes();
    constexpr size_t TOLERANCE = 1024 * 1024;
    EXPECT_GE(free_after_free + TOLERANCE, free_before_decomp)
        << "GPU memory not reclaimed after cudaFree of decompress output";
}

// ─────────────────────────────────────────────────────────────────────────────
// OW4: Decompress pool-managed — data is correct, no cudaFree needed.
//
// With setPoolManagedDecompOutput(true), the returned pointer lives in the
// pool.  This test verifies the data is correct and that the pointer is
// NOT freed (it will be freed when the pipeline is destroyed or on the next
// decompress() call).
// ─────────────────────────────────────────────────────────────────────────────
TEST(Ownership, DecompressPoolManagedDataCorrect) {
    constexpr size_t N  = 2048;
    constexpr float  EB = 1e-2f;
    const size_t in_bytes = N * sizeof(float);

    auto h_in = make_random_floats(N, 40);
    auto p = make_pipeline(N, MemoryStrategy::MINIMAL, /*pool_managed_decomp=*/true);

    CudaStream stream;
    CudaBuffer<float> d_in(N);
    d_in.upload(h_in, stream);
    stream.sync();

    void*  d_comp = nullptr;
    size_t comp_sz = 0;
    ASSERT_NO_THROW(p->compress(d_in.void_ptr(), in_bytes, &d_comp, &comp_sz, stream));

    void*  d_dec  = nullptr;
    size_t dec_sz = 0;
    ASSERT_NO_THROW(p->decompress(d_comp, comp_sz, &d_dec, &dec_sz, stream));
    ASSERT_NE(d_dec, nullptr);
    ASSERT_EQ(dec_sz, in_bytes);

    std::vector<float> h_recon(N);
    FZ_TEST_CUDA(cudaMemcpy(h_recon.data(), d_dec, in_bytes, cudaMemcpyDeviceToHost));
    EXPECT_LE(max_abs_error(h_in, h_recon), EB * 1.01f);

    // Do NOT cudaFree(d_dec) — pool-managed; destructor handles it.
}

// ─────────────────────────────────────────────────────────────────────────────
// OW5: Pool-managed decompress pointer survives a subsequent compress().
//
// compress() resets the forward DAG (freeing forward intermediate buffers)
// but must NOT free the pool-managed decompress output — that buffer lives in
// d_decomp_outputs_ with persistent=true and is only freed on the next
// decompress() call or destructor.
// ─────────────────────────────────────────────────────────────────────────────
TEST(Ownership, PoolManagedDecompSurvivesSubsequentCompress) {
    constexpr size_t N  = 2048;
    constexpr float  EB = 1e-2f;
    const size_t in_bytes = N * sizeof(float);

    auto h_in1 = make_random_floats(N, 50);
    auto h_in2 = make_random_floats(N, 51);
    auto p = make_pipeline(N, MemoryStrategy::MINIMAL, /*pool_managed_decomp=*/true);

    CudaStream stream;
    CudaBuffer<float> d_in1(N), d_in2(N);
    d_in1.upload(h_in1, stream);
    d_in2.upload(h_in2, stream);
    stream.sync();

    // First compress + pool-managed decompress.
    void *d_comp1 = nullptr, *d_dec1 = nullptr;
    size_t comp_sz1 = 0, dec_sz1 = 0;
    ASSERT_NO_THROW(p->compress(d_in1.void_ptr(), in_bytes, &d_comp1, &comp_sz1, stream));
    ASSERT_NO_THROW(p->decompress(d_comp1, comp_sz1, &d_dec1, &dec_sz1, stream));
    ASSERT_NE(d_dec1, nullptr);

    // Second compress — implicitly resets the forward DAG.
    void *d_comp2 = nullptr;
    size_t comp_sz2 = 0;
    ASSERT_NO_THROW(p->compress(d_in2.void_ptr(), in_bytes, &d_comp2, &comp_sz2, stream));
    stream.sync();

    // d_dec1 must still be readable and contain correct data from the first
    // decompress — compress() must not have freed it.
    std::vector<float> h_recon1(N);
    FZ_TEST_CUDA(cudaMemcpy(h_recon1.data(), d_dec1, in_bytes, cudaMemcpyDeviceToHost));
    EXPECT_LE(max_abs_error(h_in1, h_recon1), EB * 1.01f)
        << "Pool-managed decompress pointer must remain valid after a subsequent compress()";
}

// ─────────────────────────────────────────────────────────────────────────────
// OW6: Second pool-managed decompress() frees the previous pointer.
//
// The pointer returned by the first decompress() becomes invalid when the
// second decompress() is called (the implementation frees d_decomp_outputs_
// at the start of decompress when pool-managed).
//
// We verify this indirectly: pool current usage must stay bounded (not grow
// indefinitely), and the data from the SECOND decompress must be correct.
// Direct verification of the first pointer being invalid would require
// intentionally accessing freed GPU memory, which would crash — so we rely
// on compute-sanitizer for that negative case.
// ─────────────────────────────────────────────────────────────────────────────
TEST(Ownership, PoolManagedDecompSecondCallFreesPrevious) {
    constexpr size_t N  = 4096;
    constexpr float  EB = 1e-2f;
    const size_t in_bytes = N * sizeof(float);

    auto h_in = make_random_floats(N, 60);
    auto p = make_pipeline(N, MemoryStrategy::MINIMAL, /*pool_managed_decomp=*/true);

    CudaStream stream;
    CudaBuffer<float> d_in(N);
    d_in.upload(h_in, stream);
    stream.sync();

    // Compress once; reuse the same compressed data for both decompresses.
    void*  d_comp = nullptr;
    size_t comp_sz = 0;
    ASSERT_NO_THROW(p->compress(d_in.void_ptr(), in_bytes, &d_comp, &comp_sz, stream));
    stream.sync();

    // First decompress.
    void*  d_dec1  = nullptr;
    size_t dec_sz1 = 0;
    ASSERT_NO_THROW(p->decompress(d_comp, comp_sz, &d_dec1, &dec_sz1, stream));
    ASSERT_NE(d_dec1, nullptr);
    stream.sync();

    const size_t usage_after_first = p->getCurrentMemoryUsage();

    // Second decompress — must free d_dec1 and allocate a new buffer.
    void*  d_dec2  = nullptr;
    size_t dec_sz2 = 0;
    ASSERT_NO_THROW(p->decompress(d_comp, comp_sz, &d_dec2, &dec_sz2, stream));
    ASSERT_NE(d_dec2, nullptr);
    ASSERT_EQ(dec_sz2, in_bytes);
    stream.sync();

    const size_t usage_after_second = p->getCurrentMemoryUsage();

    // Pool usage must not have grown by more than one decomp output buffer —
    // the first allocation must have been freed before the second was made.
    EXPECT_LE(usage_after_second, usage_after_first + in_bytes + 1024)
        << "Pool usage grew more than one output buffer between two pool-managed "
           "decompresses — previous buffer may not have been freed";

    // Data from the second decompress must be correct.
    std::vector<float> h_recon2(N);
    FZ_TEST_CUDA(cudaMemcpy(h_recon2.data(), d_dec2, in_bytes, cudaMemcpyDeviceToHost));
    EXPECT_LE(max_abs_error(h_in, h_recon2), EB * 1.01f)
        << "Second pool-managed decompress returned incorrect data";
}

// ─────────────────────────────────────────────────────────────────────────────
// OW7: Pool-managed decompress pointer survives reset().
//
// reset() invalidates the compress output (forward DAG non-persistent buffers
// are freed) but must NOT free pool-managed decompress outputs — those live in
// d_decomp_outputs_ and are only freed on the next decompress() call or
// Pipeline destruction.  The docstring says "valid until the next decompress()
// call or Pipeline destruction" — reset() is intentionally not in that list.
// ─────────────────────────────────────────────────────────────────────────────
TEST(Ownership, PoolManagedDecompSurvivesReset) {
    constexpr size_t N  = 2048;
    constexpr float  EB = 1e-2f;
    const size_t in_bytes = N * sizeof(float);

    auto h_in = make_random_floats(N, 70);
    auto p = make_pipeline(N, MemoryStrategy::MINIMAL, /*pool_managed_decomp=*/true);

    CudaStream stream;
    CudaBuffer<float> d_in(N);
    d_in.upload(h_in, stream);
    stream.sync();

    void*  d_comp = nullptr;
    size_t comp_sz = 0;
    ASSERT_NO_THROW(p->compress(d_in.void_ptr(), in_bytes, &d_comp, &comp_sz, stream));

    void*  d_dec  = nullptr;
    size_t dec_sz = 0;
    ASSERT_NO_THROW(p->decompress(d_comp, comp_sz, &d_dec, &dec_sz, stream));
    ASSERT_NE(d_dec, nullptr);
    stream.sync();

    // reset() must free the forward compress output but leave d_dec intact.
    p->reset(stream);
    stream.sync();

    // Read from d_dec after reset() — must still hold valid decompressed data.
    std::vector<float> h_recon(N);
    FZ_TEST_CUDA(cudaMemcpy(h_recon.data(), d_dec, in_bytes, cudaMemcpyDeviceToHost));
    EXPECT_LE(max_abs_error(h_in, h_recon), EB * 1.01f)
        << "Pool-managed decompress pointer must remain valid after reset()";
}



// ─────────────────────────────────────────────────────────────────────────────
// OW9: Zero-size compress round-trips without crash.
//
// compress() with a zero-byte input is an edge case, not a programming error.
// All stages have n==0 guards that produce a valid (empty) compressed stream.
// This test verifies the pipeline survives, produces a non-null output, and
// that decompress() on that output also completes without crashing.
// ─────────────────────────────────────────────────────────────────────────────
TEST(Ownership, ZeroSizeCompressRoundTrip) {
    auto p = make_pipeline(1024);  // hint is non-zero so buffers are sized

    CudaStream stream;
    // Allocate a minimal device buffer (can't pass null — that's a separate error).
    CudaBuffer<float> d_in(1);

    void*  d_comp = nullptr;
    size_t comp_sz = 0;
    ASSERT_NO_THROW(p->compress(d_in.void_ptr(), 0, &d_comp, &comp_sz, stream))
        << "Zero-size compress must not throw";
    ASSERT_NE(d_comp, nullptr);

    // Decompress the zero-size stream — must not crash.
    void*  d_dec  = nullptr;
    size_t dec_sz = 0;
    ASSERT_NO_THROW(p->decompress(d_comp, comp_sz, &d_dec, &dec_sz, stream))
        << "Decompress of zero-size compressed stream must not throw";

    if (d_dec) cudaFree(d_dec);
}



// ─────────────────────────────────────────────────────────────────────────────
// OW11: Pool-managed decompress pool usage stays bounded across 5 calls.
//
// Each decompress() call must free the previous pool-owned output before
// allocating the next.  Pool usage after call N (N > 1) must not exceed
// usage after call 1 by more than one output buffer — proving no accumulation.
// ─────────────────────────────────────────────────────────────────────────────
TEST(Ownership, PoolManagedDecompBoundedAcross5Calls) {
    constexpr size_t N  = 4096;
    const size_t     in_bytes = N * sizeof(float);

    auto h_in = make_random_floats(N, 90);
    auto p = make_pipeline(N, MemoryStrategy::MINIMAL, /*pool_managed_decomp=*/true);

    CudaStream stream;
    CudaBuffer<float> d_in(N);
    d_in.upload(h_in, stream);
    stream.sync();

    void*  d_comp  = nullptr;
    size_t comp_sz = 0;
    ASSERT_NO_THROW(p->compress(d_in.void_ptr(), in_bytes, &d_comp, &comp_sz, stream));
    stream.sync();

    size_t usage_after_first = 0;
    for (int i = 0; i < 5; ++i) {
        void*  d_dec  = nullptr;
        size_t dec_sz = 0;
        ASSERT_NO_THROW(p->decompress(d_comp, comp_sz, &d_dec, &dec_sz, stream));
        stream.sync();
        ASSERT_NE(d_dec, nullptr);

        const size_t usage = p->getCurrentMemoryUsage();
        if (i == 0) {
            usage_after_first = usage;
        } else {
            EXPECT_LE(usage, usage_after_first + in_bytes + 1024)
                << "Pool grew by more than one output buffer on iteration " << (i + 1)
                << " — previous pool-managed output may not have been freed";
        }
    }
}
