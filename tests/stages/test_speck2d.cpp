/**
 * tests/stages/test_speck2d.cpp
 *
 * GPU unit tests for Speck2DStage — the GPU-parallel "wavefront" SPECK-like
 * coder (see modules/coders/speck2d/speck2d_kernels.cuh for the algorithm).
 *
 *   SK1  SPECK2D/RoundTripUniform    — lossless round trip, dense random codes
 *   SK2  SPECK2D/RoundTripClustered  — lossless round trip, DWT-like sparse codes
 *   SK3  SPECK2D/RoundTripNonSquare  — non-power-of-2, non-square dims
 *   SK4  SPECK2D/AllZero             — degenerate all-zero field, empty payload
 *   SK5  SPECK2D/HeaderRoundTrip     — dims/B/nbitsA survive serialize/deserialize
 *   SK6  SPECK2D/Metadata            — ports, type id, worst-case size estimate
 *   SK7  SPECK2D/DimsMismatchThrows  — input smaller than dims imply
 *   SK8  SPECK2D/ThreeDThrows        — 3-D not yet supported
 *   SK9  SPECK2D/Compresses          — actual output is smaller than raw input
 */

#include <gtest/gtest.h>
#include "helpers/fz_test_utils.h"
#include "coders/speck2d/speck2d_stage.h"
#include "fzgpumodules.h"

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

using namespace fz;
using namespace fz_test;

namespace {

std::vector<int32_t> make_uniform_codes(size_t n, uint64_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int32_t> d(-65535, 65535);
    std::vector<int32_t> v(n);
    for (auto& x : v) x = d(rng);
    return v;
}

// DWT-like: sparse, magnitude concentrated near one corner (mimics quantized
// wavelet coefficients, the intended real workload for this stage).
std::vector<int32_t> make_clustered_codes(int nx, int ny, uint64_t seed) {
    std::mt19937 rng(seed);
    std::vector<int32_t> v((size_t)nx * ny);
    for (int y = 0; y < ny; ++y)
        for (int x = 0; x < nx; ++x) {
            double s = std::exp(-6.0 * ((double)x / nx + (double)y / ny));
            std::exponential_distribution<double> ex(1.0 / (1.0 + 4000.0 * s));
            double m = ex(rng);
            if (s < 0.05 && std::uniform_real_distribution<double>(0, 1)(rng) < 0.7) m = 0;
            int32_t mag = (int32_t)m;
            bool neg = mag && (rng() & 1);
            v[(size_t)y * nx + x] = neg ? -mag : mag;
        }
    return v;
}

// Forward-compress then inverse-decompress; returns the reconstructed codes.
// Mirrors test_cdf97.cpp's round_trip() but with the async-readback pattern
// this stage requires (postStreamSync() after the stream is synced).
std::vector<int32_t> round_trip(const std::vector<int32_t>& in, int nx, int ny,
                                cudaStream_t stream, MemoryPool& pool,
                                size_t* out_compressed_bytes = nullptr)
{
    const size_t n = in.size();
    EXPECT_EQ(n, (size_t)nx * ny);

    Speck2DStage fwd;
    fwd.setDims((size_t)nx, (size_t)ny);
    CudaBuffer<int32_t> d_in(n);
    d_in.upload(in, stream);

    auto est = fwd.estimateOutputSizes({n * sizeof(int32_t)});
    CudaBuffer<uint8_t> d_comp(est[0]);

    fwd.execute(stream, &pool, {d_in.void_ptr()}, {d_comp.void_ptr()}, {n * sizeof(int32_t)});
    cudaStreamSynchronize(stream);
    fwd.postStreamSync(stream);
    const size_t comp_bytes = fwd.getActualOutputSize(0);
    if (out_compressed_bytes) *out_compressed_bytes = comp_bytes;

    uint8_t hdr[FZM_STAGE_CONFIG_SIZE] = {};
    const size_t hlen = fwd.serializeHeader(0, hdr, sizeof(hdr));

    Speck2DStage inv;
    inv.deserializeHeader(hdr, hlen);
    inv.setInverse(true);

    CudaBuffer<int32_t> d_out(n);
    inv.execute(stream, &pool, {d_comp.void_ptr()}, {d_out.void_ptr()}, {comp_bytes});
    cudaStreamSynchronize(stream);
    return d_out.download(stream);
}

} // namespace

TEST(SPECK2D, RoundTripUniform) {
    const int nx = 64, ny = 64;
    CudaStream s; auto pool = make_test_pool(nx * ny * sizeof(int32_t) * 4);
    auto in = make_uniform_codes((size_t)nx * ny, 21);
    auto out = round_trip(in, nx, ny, s, *pool);
    EXPECT_EQ(in, out);
}

TEST(SPECK2D, RoundTripClustered) {
    const int nx = 257, ny = 129;   // non-power-of-2, exercises the deep multi-launch tail
    CudaStream s; auto pool = make_test_pool(nx * ny * sizeof(int32_t) * 4);
    auto in = make_clustered_codes(nx, ny, 22);
    auto out = round_trip(in, nx, ny, s, *pool);
    EXPECT_EQ(in, out);
}

TEST(SPECK2D, RoundTripNonSquare) {
    const int nx = 500, ny = 300;
    CudaStream s; auto pool = make_test_pool(nx * ny * sizeof(int32_t) * 4);
    auto in = make_clustered_codes(nx, ny, 23);
    auto out = round_trip(in, nx, ny, s, *pool);
    EXPECT_EQ(in, out);
}

TEST(SPECK2D, AllZero) {
    const int nx = 128, ny = 128;
    CudaStream s; auto pool = make_test_pool(nx * ny * sizeof(int32_t) * 4);
    std::vector<int32_t> in((size_t)nx * ny, 0);
    size_t comp_bytes = 0;
    auto out = round_trip(in, nx, ny, s, *pool, &comp_bytes);
    EXPECT_EQ(in, out);
    EXPECT_EQ(comp_bytes, 0u);   // B<0 -> nothing to code
}

TEST(SPECK2D, HeaderRoundTrip) {
    const int nx = 64, ny = 64;
    CudaStream s; auto pool = make_test_pool(nx * ny * sizeof(int32_t) * 4);
    Speck2DStage fwd;
    fwd.setDims((size_t)nx, (size_t)ny);
    auto in = make_clustered_codes(nx, ny, 24);
    CudaBuffer<int32_t> d_in((size_t)nx * ny);
    d_in.upload(in, s);
    auto est = fwd.estimateOutputSizes({in.size() * sizeof(int32_t)});
    CudaBuffer<uint8_t> d_comp(est[0]);
    fwd.execute(s, pool.get(), {d_in.void_ptr()}, {d_comp.void_ptr()}, {in.size() * sizeof(int32_t)});
    cudaStreamSynchronize(s);
    fwd.postStreamSync(s);

    uint8_t hdr[FZM_STAGE_CONFIG_SIZE] = {};
    const size_t hlen = fwd.serializeHeader(0, hdr, sizeof(hdr));
    EXPECT_EQ(hlen, sizeof(Speck2DConfig));

    Speck2DStage st2;
    st2.deserializeHeader(hdr, hlen);
    st2.setInverse(true);
    // Exact inverse-direction size estimate comes from the deserialized dims.
    EXPECT_EQ(st2.estimateOutputSizes({0})[0], (size_t)nx * ny * sizeof(int32_t));
}

TEST(SPECK2D, Metadata) {
    Speck2DStage st;
    EXPECT_EQ(st.getName(), "SPECK2D");
    EXPECT_EQ(st.getNumInputs(), 1u);
    EXPECT_EQ(st.getNumOutputs(), 1u);
    EXPECT_EQ(st.getStageTypeId(), static_cast<uint16_t>(StageType::SPECK2D));
    EXPECT_EQ(st.getInputDataType(0), static_cast<uint8_t>(DataType::INT32));
    EXPECT_FALSE(st.isGraphCompatible());
    // Worst case: (3n+8) words -- see estimateOutputSizes() doc.
    const size_t n = 1000;
    size_t expect = (3 * n + 8) * sizeof(uint32_t);
    EXPECT_EQ(st.estimateOutputSizes({n * sizeof(int32_t)})[0], expect);
}

TEST(SPECK2D, DimsMismatchThrows) {
    CudaStream s; auto pool = make_test_pool(1024 * sizeof(int32_t));
    Speck2DStage st;
    st.setDims(100, 1);   // 100*1 != 1024/4 elements implied by the buffer below
    CudaBuffer<int32_t> d_in(1024), d_out(4096);
    EXPECT_THROW(st.execute(s, pool.get(), {d_in.void_ptr()}, {d_out.void_ptr()}, {50 * sizeof(int32_t)}),
                 std::runtime_error);
}

TEST(SPECK2D, ThreeDThrows) {
    CudaStream s; auto pool = make_test_pool(64 * sizeof(int32_t));
    Speck2DStage st;
    st.setDims({4, 4, 4});
    CudaBuffer<int32_t> d_in(64), d_out(256);
    EXPECT_THROW(st.execute(s, pool.get(), {d_in.void_ptr()}, {d_out.void_ptr()}, {64 * sizeof(int32_t)}),
                 std::runtime_error);
}

TEST(SPECK2D, Compresses) {
    // Clustered (DWT-like) data should compress well below raw size -- this is
    // the coder's whole purpose, not just correctness.
    const int nx = 512, ny = 512;
    CudaStream s; auto pool = make_test_pool(nx * ny * sizeof(int32_t) * 4);
    auto in = make_clustered_codes(nx, ny, 25);
    size_t comp_bytes = 0;
    auto out = round_trip(in, nx, ny, s, *pool, &comp_bytes);
    EXPECT_EQ(in, out);
    EXPECT_LT(comp_bytes, in.size() * sizeof(int32_t) / 2);
}
