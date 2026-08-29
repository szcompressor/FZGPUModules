/**
 * tests/stages/test_cdf97.cpp
 *
 * GPU unit tests for Cdf97Stage — the CDF 9/7 wavelet transform (SPERR's DWT).
 *
 *   CW1  CDF97/RoundTrip1D            — forward+inverse recovers a 1-D signal
 *   CW2  CDF97/RoundTrip2D            — recovers a 2-D field
 *   CW3  CDF97/RoundTrip3DDyadic      — recovers a cubic (dyadic) volume
 *   CW4  CDF97/RoundTrip3DPacket      — recovers an anisotropic (packet) volume
 *   CW5  CDF97/FloatRoundTrip         — float variant round-trips within f32 eps
 *   CW6  CDF97/HeaderRoundTrip        — dims survive serialize/deserialize
 *   CW7  CDF97/Metadata               — ports, type id, size-preserving estimate
 *   CW8  CDF97/DimsMismatchThrows     — dims product must equal element count
 *   CW9  CDF97/LongLineThrows         — an over-long line is rejected, not silent
 *   CW10 CDF97/NotIdentity            — the forward pass actually transforms
 */

#include <gtest/gtest.h>
#include "helpers/fz_test_utils.h"
#include "transforms/cdf97/cdf97_stage.h"
#include "fzgpumodules.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

using namespace fz;
using namespace fz_test;

namespace {

// Forward then inverse through two stage instances (the inverse instance is
// configured from the forward instance's serialized header, exercising the
// decompression reconstruction path). Returns the reconstructed field.
template <typename T>
std::vector<T> round_trip(const std::vector<T>& in,
                          std::array<size_t, 3> dims,
                          cudaStream_t stream, MemoryPool& pool)
{
    const size_t n = in.size();
    const size_t bytes = n * sizeof(T);

    Cdf97Stage<T> fwd;
    fwd.setDims(dims);
    CudaBuffer<T> d_in(n), d_coeff(n);
    d_in.upload(in, stream);

    fwd.execute(stream, &pool, {d_in.void_ptr()}, {d_coeff.void_ptr()}, {bytes});
    EXPECT_EQ(fwd.getActualOutputSize(0), bytes);  // size-preserving

    // Reconstruct the inverse stage from the serialized header.
    uint8_t hdr[FZM_STAGE_CONFIG_SIZE] = {};
    const size_t hlen = fwd.serializeHeader(0, hdr, sizeof(hdr));
    Cdf97Stage<T> inv;
    inv.deserializeHeader(hdr, hlen);
    inv.setInverse(true);
    EXPECT_EQ(inv.getDims(), dims);

    CudaBuffer<T> d_out(n);
    inv.execute(stream, &pool, {d_coeff.void_ptr()}, {d_out.void_ptr()}, {bytes});
    return d_out.download(stream);
}

template <typename T>
double max_abs_diff(const std::vector<T>& a, const std::vector<T>& b) {
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs((double)a[i] - (double)b[i]));
    return m;
}

} // namespace

TEST(CDF97, RoundTrip1D) {
    CudaStream s; auto pool = make_test_pool(4096 * sizeof(double));
    auto in = make_random_doubles(4096, 7);
    auto out = round_trip<double>(in, {4096, 1, 1}, s, *pool);
    EXPECT_LT(max_abs_diff(in, out), 1e-10);
}

TEST(CDF97, RoundTrip2D) {
    CudaStream s; auto pool = make_test_pool(129 * 257 * sizeof(double));
    auto in = make_random_doubles(129 * 257, 8);
    auto out = round_trip<double>(in, {257, 129, 1}, s, *pool);  // {nx, ny, 1}
    EXPECT_LT(max_abs_diff(in, out), 1e-10);
}

TEST(CDF97, RoundTrip3DDyadic) {
    CudaStream s; auto pool = make_test_pool(64 * 64 * 64 * sizeof(double));
    auto in = make_random_doubles(64 * 64 * 64, 9);
    auto out = round_trip<double>(in, {64, 64, 64}, s, *pool);
    EXPECT_LT(max_abs_diff(in, out), 1e-10);
}

TEST(CDF97, RoundTrip3DPacket) {
    CudaStream s; auto pool = make_test_pool(50 * 50 * 100 * sizeof(double));
    auto in = make_random_doubles(50 * 50 * 100, 10);
    auto out = round_trip<double>(in, {50, 50, 100}, s, *pool);  // anisotropic
    EXPECT_LT(max_abs_diff(in, out), 1e-10);
}

TEST(CDF97, FloatRoundTrip) {
    CudaStream s; auto pool = make_test_pool(128 * 96 * sizeof(float));
    auto in = make_random_floats(128 * 96, 11);
    auto out = round_trip<float>(in, {96, 128, 1}, s, *pool);
    EXPECT_LT(max_abs_diff(in, out), 1e-4);  // float path: looser, not bit-exact
}

TEST(CDF97, HeaderRoundTrip) {
    Cdf97Stage<double> st;
    st.setDims(129, 257, 33);
    uint8_t hdr[FZM_STAGE_CONFIG_SIZE] = {};
    const size_t hlen = st.serializeHeader(0, hdr, sizeof(hdr));
    EXPECT_EQ(hlen, sizeof(Cdf97Config));

    Cdf97Stage<double> st2;
    st2.deserializeHeader(hdr, hlen);
    EXPECT_EQ(st2.getDims(), (std::array<size_t, 3>{129, 257, 33}));
}

TEST(CDF97, Metadata) {
    Cdf97Stage<double> st;
    EXPECT_EQ(st.getName(), "CDF97");
    EXPECT_EQ(st.getNumInputs(), 1u);
    EXPECT_EQ(st.getNumOutputs(), 1u);
    EXPECT_EQ(st.getStageTypeId(), static_cast<uint16_t>(StageType::CDF97));
    EXPECT_EQ(st.getOutputDataType(0), static_cast<uint8_t>(DataType::FLOAT64));
    EXPECT_EQ(Cdf97Stage<float>().getOutputDataType(0),
              static_cast<uint8_t>(DataType::FLOAT32));
    // size-preserving in both directions
    EXPECT_EQ(st.estimateOutputSizes({4096})[0], 4096u);
}

TEST(CDF97, DimsMismatchThrows) {
    CudaStream s; auto pool = make_test_pool(1024 * sizeof(double));
    Cdf97Stage<double> st;
    st.setDims(100, 1, 1);  // 100 != 1024/8
    CudaBuffer<double> d_in(1024 / sizeof(double)), d_out(1024 / sizeof(double));
    EXPECT_THROW(st.execute(s, pool.get(), {d_in.void_ptr()}, {d_out.void_ptr()}, {1024}),
                 std::runtime_error);
}

TEST(CDF97, LongLineThrows) {
    // A line longer than the shared-memory limit must be rejected, not corrupt.
    const size_t nx = Cdf97Stage<double>::kMaxLineElems + 64;
    CudaStream s; auto pool = make_test_pool(nx * sizeof(double));
    Cdf97Stage<double> st;
    st.setDims(nx, 1, 1);
    CudaBuffer<double> d_in(nx), d_out(nx);
    EXPECT_THROW(
        st.execute(s, pool.get(), {d_in.void_ptr()}, {d_out.void_ptr()}, {nx * sizeof(double)}),
        std::runtime_error);
}

TEST(CDF97, NotIdentity) {
    CudaStream s; auto pool = make_test_pool(64 * 64 * sizeof(double));
    auto in = make_random_doubles(64 * 64, 12);
    Cdf97Stage<double> fwd; fwd.setDims(64, 64, 1);
    CudaBuffer<double> d_in(in.size()), d_coeff(in.size());
    d_in.upload(in, s);
    fwd.execute(s, pool.get(), {d_in.void_ptr()}, {d_coeff.void_ptr()}, {in.size() * sizeof(double)});
    auto coeff = d_coeff.download(s);
    EXPECT_GT(max_abs_diff(in, coeff), 1e-3);  // the transform did something
}
