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
 *   OW8  decompressMulti() returns one caller-owned pointer per source;
 *          each cudaFree succeeds and memory is reclaimed
 *   OW9  Zero-size compress round-trips without crash (edge case)
 *   OW10 Multi-source decompress() returns a single caller-owned concat buffer
 *          in [num_bufs:u32][sz:u64][data]... format; both sources parse and
 *          verify correctly; the concat pointer is caller-owned (cudaFree)
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
// at the start of decompressMulti).
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
// OW8: decompressMulti() returns one caller-owned pointer per source.
//
// Uses a single-source pipeline (decompressMulti still applies and returns a
// vector of size 1). Verifies:
//   - data is correct
//   - cudaFree on each returned pointer succeeds
//   - memory is reclaimed after freeing all pointers
// ─────────────────────────────────────────────────────────────────────────────
TEST(Ownership, DecompressMultiCallerOwned) {
    constexpr size_t N  = 2048;
    constexpr float  EB = 1e-2f;
    const size_t in_bytes = N * sizeof(float);

    auto h_in = make_random_floats(N, 80);
    auto p = make_pipeline(N);

    CudaStream stream;
    CudaBuffer<float> d_in(N);
    d_in.upload(h_in, stream);
    stream.sync();

    void*  d_comp = nullptr;
    size_t comp_sz = 0;
    ASSERT_NO_THROW(p->compress(d_in.void_ptr(), in_bytes, &d_comp, &comp_sz, stream));

    cudaDeviceSynchronize();
    const size_t free_before = gpu_free_bytes();

    auto results = p->decompressMulti(d_comp, comp_sz, stream);
    ASSERT_EQ(results.size(), 1u);
    ASSERT_NE(results[0].first, nullptr);
    ASSERT_EQ(results[0].second, in_bytes);

    std::vector<float> h_recon(N);
    FZ_TEST_CUDA(cudaMemcpy(h_recon.data(), results[0].first, in_bytes,
                            cudaMemcpyDeviceToHost));
    EXPECT_LE(max_abs_error(h_in, h_recon), EB * 1.01f);

    // Free all returned pointers — each is a fresh cudaMalloc'd buffer.
    for (auto& [ptr, _] : results) {
        cudaError_t err = cudaFree(ptr);
        EXPECT_EQ(err, cudaSuccess)
            << "cudaFree on decompressMulti result must succeed";
    }

    cudaDeviceSynchronize();
    const size_t free_after = gpu_free_bytes();
    constexpr size_t TOLERANCE = 1024 * 1024;
    EXPECT_GE(free_after + TOLERANCE, free_before)
        << "GPU memory not reclaimed after cudaFree of decompressMulti results";
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
// OW10: Multi-source decompress() returns a valid caller-owned concat buffer.
//
// A two-source pipeline is compressed via the vector<InputSpec> overload.
// decompress() (not decompressMulti()) is called — for a multi-source pipeline
// it must return a single caller-owned buffer packed as:
//   [num_bufs:u32][sz0:u64][data0...][sz1:u64][data1...]
// Both source regions are parsed host-side and verified against the originals.
// The single returned pointer must be freed with cudaFree (caller-owned).
//
// Both sources use the same smooth data so results can be verified without
// knowing which decompressMulti index maps to which InputSpec source.
// ─────────────────────────────────────────────────────────────────────────────
TEST(Ownership, MultiSourceDecompressReturnsConcatBuffer) {
    constexpr size_t N  = 4096;
    constexpr float  EB = 1e-2f;
    const size_t     in_bytes = N * sizeof(float);

    // Use smooth sinusoidal data — Lorenzo handles it well (few outliers).
    auto h_input = make_sine_floats(N, 0.05f, 5.0f);

    CudaStream stream;
    CudaBuffer<float> d_in1(N), d_in2(N);
    d_in1.upload(h_input, stream);
    d_in2.upload(h_input, stream);
    stream.sync();

    // Two independent Lorenzo sources; total hint = 2 × per-source.
    Pipeline pipeline(2 * in_bytes, MemoryStrategy::MINIMAL, 5.0f);
    auto* lrz1 = pipeline.addStage<LorenzoStage<float, uint16_t>>();
    lrz1->setErrorBound(EB);
    lrz1->setQuantRadius(512);
    lrz1->setOutlierCapacity(0.2f);

    auto* lrz2 = pipeline.addStage<LorenzoStage<float, uint16_t>>();
    lrz2->setErrorBound(EB);
    lrz2->setQuantRadius(512);
    lrz2->setOutlierCapacity(0.2f);

    pipeline.setInputSizeHint(lrz1, in_bytes);
    pipeline.setInputSizeHint(lrz2, in_bytes);
    pipeline.finalize();

    void*  d_comp  = nullptr;
    size_t comp_sz = 0;
    ASSERT_NO_THROW(pipeline.compress(
        {{lrz1, d_in1.void_ptr(), in_bytes},
         {lrz2, d_in2.void_ptr(), in_bytes}},
        &d_comp, &comp_sz, stream
    ));
    stream.sync();
    ASSERT_GT(comp_sz, 0u);

    // decompress() on a multi-source pipeline returns a single concat buffer.
    void*  d_dec  = nullptr;
    size_t dec_sz = 0;
    ASSERT_NO_THROW(pipeline.decompress(nullptr, 0, &d_dec, &dec_sz, stream));
    stream.sync();
    ASSERT_NE(d_dec, nullptr);
    ASSERT_GT(dec_sz, 0u);

    // D2H the entire concat buffer so we can parse it on the host.
    std::vector<uint8_t> h_concat(dec_sz);
    FZ_TEST_CUDA(cudaMemcpy(h_concat.data(), d_dec, dec_sz, cudaMemcpyDeviceToHost));

    // Caller-owned: free it now (data is in h_concat on the host).
    EXPECT_EQ(cudaFree(d_dec), cudaSuccess)
        << "cudaFree on multi-source decompress concat pointer must succeed";

    // Parse [num_bufs:u32][sz0:u64][data0...][sz1:u64][data1...]
    const uint8_t* ptr = h_concat.data();
    uint32_t num_bufs = 0;
    std::memcpy(&num_bufs, ptr, sizeof(uint32_t));
    ptr += sizeof(uint32_t);
    ASSERT_EQ(num_bufs, 2u) << "Multi-source concat must contain 2 buffers";

    for (uint32_t src = 0; src < num_bufs; ++src) {
        uint64_t sz = 0;
        std::memcpy(&sz, ptr, sizeof(uint64_t));
        ptr += sizeof(uint64_t);
        ASSERT_EQ(sz, in_bytes)
            << "Source " << src << ": wrong byte-count in concat header";

        std::vector<float> h_recon(N);
        std::memcpy(h_recon.data(), ptr, in_bytes);
        ptr += sz;

        EXPECT_LE(max_abs_error(h_input, h_recon), EB * 1.01f)
            << "Source " << src << ": reconstruction error exceeds bound";
    }
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
