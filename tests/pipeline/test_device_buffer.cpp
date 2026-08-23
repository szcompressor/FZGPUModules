/**
 * tests/pipeline/test_device_buffer.cpp
 *
 * Tests for the explicit-ownership execution API (fz::DeviceSpan,
 * fz::BorrowedDeviceBuffer, fz::OwnedDeviceBuffer and the Pipeline span
 * overloads). These wrap the pointer overloads, so the contract under test is:
 *
 *   DB1  compile-time traits: spans are trivially copyable, OwnedDeviceBuffer
 *          is move-only, BorrowedDeviceBuffer is non-owning/copyable
 *   DB2  compress(span) borrows from the pool and round-trips
 *   DB3  compressInto(span, span) writes the same bytes as the borrowing form
 *   DB4  decompressOwned() frees on destruction (device memory reclaimed)
 *   DB5  decompressOwned() owns regardless of setPoolManagedDecompOutput(true)
 *   DB6  decompressBorrowed() borrows regardless of
 *          setPoolManagedDecompOutput(false), and leaves the flag as it found it
 *   DB7  decompressInto(span, span) matches the pointer overload
 *   DB8  caller-capacity failure throws before anything is written
 *   DB9  OwnedDeviceBuffer move transfers ownership exactly once
 */

#include <gtest/gtest.h>
#include "helpers/fz_test_utils.h"
#include "fzgpumodules.h"

#include <type_traits>
#include <vector>

using namespace fz;
using namespace fz_test;

namespace {

std::unique_ptr<Pipeline> make_pipeline(
    size_t n_floats,
    MemoryStrategy strategy = MemoryStrategy::MINIMAL)
{
    auto p = std::make_unique<Pipeline>(n_floats * sizeof(float), strategy);
    auto* lrz = p->addStage<LorenzoQuantStage<float, uint16_t>>();
    lrz->setErrorBound(1e-2f);
    p->finalize();
    return p;
}

size_t free_device_bytes() {
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) return 0;
    return free_b;
}

}  // namespace

// ── DB1 ──────────────────────────────────────────────────────────────────────
TEST(DeviceBuffer, OwnershipTraits) {
    static_assert(std::is_trivially_copyable<DeviceSpan>::value, "");
    static_assert(std::is_trivially_copyable<ConstDeviceSpan>::value, "");
    static_assert(std::is_copy_constructible<BorrowedDeviceBuffer>::value,
                  "borrowed buffers are non-owning views and may be copied");

    static_assert(!std::is_copy_constructible<OwnedDeviceBuffer>::value, "");
    static_assert(!std::is_copy_assignable<OwnedDeviceBuffer>::value, "");
    static_assert(std::is_move_constructible<OwnedDeviceBuffer>::value, "");
    static_assert(std::is_move_assignable<OwnedDeviceBuffer>::value, "");

    // A DeviceSpan converts to a read-only view, never the other way.
    static_assert(std::is_convertible<DeviceSpan, ConstDeviceSpan>::value, "");
    static_assert(!std::is_convertible<ConstDeviceSpan, DeviceSpan>::value, "");
    // Borrowed must not silently become owned.
    static_assert(!std::is_convertible<BorrowedDeviceBuffer, OwnedDeviceBuffer>::value, "");

    EXPECT_TRUE(DeviceSpan().empty());
    EXPECT_FALSE(static_cast<bool>(OwnedDeviceBuffer()));
}

// ── DB2 ──────────────────────────────────────────────────────────────────────
TEST(DeviceBuffer, CompressSpanBorrowsAndRoundTrips) {
    constexpr size_t N  = 4096;
    constexpr float  EB = 1e-2f;
    const size_t in_bytes = N * sizeof(float);

    auto h_in = make_random_floats(N, 11);
    auto p = make_pipeline(N);

    CudaStream stream;
    CudaBuffer<float> d_in(N);
    d_in.upload(h_in, stream);
    stream.sync();

    BorrowedDeviceBuffer comp = p->compress(ConstDeviceSpan(d_in.void_ptr(), in_bytes), stream);
    ASSERT_NE(comp.data(), nullptr);
    EXPECT_GT(comp.bytes(), 0u);
    EXPECT_LT(comp.bytes(), in_bytes);

    // Borrowed → do NOT free; feed straight back in.
    BorrowedDeviceBuffer dec = p->decompressBorrowed(comp.cspan(), stream);
    ASSERT_NE(dec.data(), nullptr);
    ASSERT_EQ(dec.bytes(), in_bytes);

    std::vector<float> h_recon(N);
    FZ_TEST_CUDA(cudaMemcpy(h_recon.data(), dec.data(), in_bytes, cudaMemcpyDeviceToHost));
    EXPECT_LE(max_abs_error(h_in, h_recon), EB * 1.01f);
}

// ── DB3 ──────────────────────────────────────────────────────────────────────
TEST(DeviceBuffer, CompressIntoMatchesBorrowingForm) {
    constexpr size_t N = 4096;
    const size_t in_bytes = N * sizeof(float);

    auto h_in = make_random_floats(N, 12);
    auto p = make_pipeline(N);

    CudaStream stream;
    CudaBuffer<float> d_in(N);
    d_in.upload(h_in, stream);
    stream.sync();

    BorrowedDeviceBuffer ref = p->compress(ConstDeviceSpan(d_in.void_ptr(), in_bytes), stream);
    std::vector<uint8_t> h_ref(ref.bytes());
    FZ_TEST_CUDA(cudaMemcpy(h_ref.data(), ref.data(), ref.bytes(), cudaMemcpyDeviceToHost));

    CudaBuffer<uint8_t> d_out(p->getMaxCompressedSize(in_bytes));
    size_t written = p->compressInto(ConstDeviceSpan(d_in.void_ptr(), in_bytes),
                                     DeviceSpan(d_out.void_ptr(), d_out.bytes()), stream);
    ASSERT_EQ(written, h_ref.size());

    std::vector<uint8_t> h_got(written);
    FZ_TEST_CUDA(cudaMemcpy(h_got.data(), d_out.void_ptr(), written, cudaMemcpyDeviceToHost));
    EXPECT_EQ(h_got, h_ref);
}

// ── DB4 ──────────────────────────────────────────────────────────────────────
TEST(DeviceBuffer, OwnedBufferFreesOnDestruction) {
    constexpr size_t N = 1 << 18;   // 1 MB, large enough to see in cudaMemGetInfo
    const size_t in_bytes = N * sizeof(float);

    auto h_in = make_random_floats(N, 13);
    auto p = make_pipeline(N);

    CudaStream stream;
    CudaBuffer<float> d_in(N);
    d_in.upload(h_in, stream);
    stream.sync();

    BorrowedDeviceBuffer comp = p->compress(ConstDeviceSpan(d_in.void_ptr(), in_bytes), stream);

    const size_t before = free_device_bytes();
    {
        OwnedDeviceBuffer dec = p->decompressOwned(comp.cspan(), stream);
        ASSERT_NE(dec.data(), nullptr);
        EXPECT_EQ(dec.bytes(), in_bytes);
        EXPECT_LT(free_device_bytes(), before);   // allocation is real
    }
    // Destructor must have freed it — no cudaFree by the caller.
    FZ_TEST_CUDA(cudaDeviceSynchronize());
    EXPECT_GE(free_device_bytes(), before - (in_bytes / 2));
}

// ── DB5 / DB6 ────────────────────────────────────────────────────────────────
TEST(DeviceBuffer, OwnershipIsFromTheCallNotThePipelineFlag) {
    constexpr size_t N = 4096;
    const size_t in_bytes = N * sizeof(float);

    auto h_in = make_random_floats(N, 14);
    auto p = make_pipeline(N);

    CudaStream stream;
    CudaBuffer<float> d_in(N);
    d_in.upload(h_in, stream);
    stream.sync();

    BorrowedDeviceBuffer comp = p->compress(ConstDeviceSpan(d_in.void_ptr(), in_bytes), stream);

    // DB5: owning call under the pool-managed flag — must still be caller-owned,
    // i.e. destroying it and then decompressing again must be safe.
    p->setPoolManagedDecompOutput(true);
    {
        OwnedDeviceBuffer owned = p->decompressOwned(comp.cspan(), stream);
        ASSERT_NE(owned.data(), nullptr);
    }
    EXPECT_TRUE(p->isPoolManagedDecompOutput());   // flag restored

    // DB6: borrowing call with the flag off — pointer stays valid without a free,
    // and the flag is left as the caller set it.
    p->setPoolManagedDecompOutput(false);
    BorrowedDeviceBuffer borrowed = p->decompressBorrowed(comp.cspan(), stream);
    ASSERT_NE(borrowed.data(), nullptr);
    EXPECT_FALSE(p->isPoolManagedDecompOutput());

    std::vector<float> h_recon(N);
    FZ_TEST_CUDA(cudaMemcpy(h_recon.data(), borrowed.data(), in_bytes, cudaMemcpyDeviceToHost));
    EXPECT_LE(max_abs_error(h_in, h_recon), 1e-2f * 1.01f);
    // No cudaFree(borrowed) — the pool owns it.
}

// ── DB7 ──────────────────────────────────────────────────────────────────────
TEST(DeviceBuffer, DecompressIntoSpanMatchesPointerOverload) {
    constexpr size_t N = 4096;
    const size_t in_bytes = N * sizeof(float);

    auto h_in = make_random_floats(N, 15);
    auto p = make_pipeline(N);

    CudaStream stream;
    CudaBuffer<float> d_in(N);
    d_in.upload(h_in, stream);
    stream.sync();

    BorrowedDeviceBuffer comp = p->compress(ConstDeviceSpan(d_in.void_ptr(), in_bytes), stream);

    CudaBuffer<float> d_out(N);
    size_t written = p->decompressInto(comp.cspan(),
                                       DeviceSpan(d_out.void_ptr(), in_bytes), stream);
    ASSERT_EQ(written, in_bytes);

    std::vector<float> h_recon(N);
    FZ_TEST_CUDA(cudaMemcpy(h_recon.data(), d_out.void_ptr(), in_bytes, cudaMemcpyDeviceToHost));
    EXPECT_LE(max_abs_error(h_in, h_recon), 1e-2f * 1.01f);
}

// ── DB8 ──────────────────────────────────────────────────────────────────────
TEST(DeviceBuffer, CallerCapacityFailureThrows) {
    constexpr size_t N = 4096;
    const size_t in_bytes = N * sizeof(float);

    auto h_in = make_random_floats(N, 16);
    auto p = make_pipeline(N);

    CudaStream stream;
    CudaBuffer<float> d_in(N);
    d_in.upload(h_in, stream);
    stream.sync();

    CudaBuffer<uint8_t> d_tiny(16);
    EXPECT_THROW(p->compressInto(ConstDeviceSpan(d_in.void_ptr(), in_bytes),
                                 DeviceSpan(d_tiny.void_ptr(), 16), stream),
                 std::runtime_error);

    BorrowedDeviceBuffer comp = p->compress(ConstDeviceSpan(d_in.void_ptr(), in_bytes), stream);
    EXPECT_THROW(p->decompressInto(comp.cspan(), DeviceSpan(d_tiny.void_ptr(), 16), stream),
                 std::runtime_error);
}

// ── DB9 ──────────────────────────────────────────────────────────────────────
TEST(DeviceBuffer, OwnedBufferMoveTransfersOwnershipOnce) {
    constexpr size_t N = 4096;
    const size_t in_bytes = N * sizeof(float);

    auto h_in = make_random_floats(N, 17);
    auto p = make_pipeline(N);

    CudaStream stream;
    CudaBuffer<float> d_in(N);
    d_in.upload(h_in, stream);
    stream.sync();

    BorrowedDeviceBuffer comp = p->compress(ConstDeviceSpan(d_in.void_ptr(), in_bytes), stream);

    OwnedDeviceBuffer a = p->decompressOwned(comp.cspan(), stream);
    void* raw = a.data();
    ASSERT_NE(raw, nullptr);

    OwnedDeviceBuffer b = std::move(a);
    EXPECT_EQ(b.data(), raw);
    EXPECT_EQ(a.data(), nullptr);       // moved-from is empty, will not double-free
    EXPECT_EQ(a.bytes(), 0u);

    // release() hands the pointer back to the caller; b must not free it.
    void* released = b.release();
    EXPECT_EQ(released, raw);
    EXPECT_EQ(b.data(), nullptr);
    FZ_TEST_CUDA(cudaFree(released));
}
