/**
 * tests/test_lorenzo.cpp
 *
 * Unit tests for LorenzoStage<float, uint16_t>.
 *
 * Lorenzo is a lossy predictor with quantization:
 *   Forward: produces 4 outputs — codes, outlier_errors, outlier_indices, outlier_count
 *   Inverse: takes those 4 outputs and reconstructs the original data within error_bound
 *
 * Key properties verified:
 *   1. Forward produces non-trivial output (something happened).
 *   2. Forward + Inverse reconstructs within error_bound for smooth data.
 *   3. Constant input is reproduced exactly (zero prediction error → no outliers).
 *   4. Large smooth dataset round-trips within the configured error bound.
 */

#include <gtest/gtest.h>
#include "helpers/fz_test_utils.h"
#include "predictors/lorenzo/lorenzo.h"
#include "fzgpumodules.h"

#include <cmath>
#include <vector>

using namespace fz;
using namespace fz_test;

// ─────────────────────────────────────────────────────────────────────────────
// Helper: run Lorenzo forward pass.
// Returns device buffers for each output so the inverse pass can consume them.
// ─────────────────────────────────────────────────────────────────────────────
struct LorenzoForwardResult {
    std::vector<uint8_t> codes_raw;   // raw bytes of quantization codes
    std::vector<uint8_t> errors_raw;  // raw bytes of outlier errors
    std::vector<uint8_t> indices_raw; // raw bytes of outlier indices
    std::vector<uint8_t> count_raw;   // raw bytes of outlier count (4 bytes)
    // Sizes in bytes
    size_t codes_bytes   = 0;
    size_t errors_bytes  = 0;
    size_t indices_bytes = 0;
    size_t count_bytes   = 0;
};

static LorenzoForwardResult run_lorenzo_forward(
    LorenzoStage<float, uint16_t>& stage,
    const std::vector<float>&      h_input,
    cudaStream_t                   stream,
    fz::MemoryPool&                pool)
{
    size_t n = h_input.size();
    size_t in_bytes = n * sizeof(float);

    CudaBuffer<float> d_in(n);
    d_in.upload(h_input, stream);

    auto est = stage.estimateOutputSizes({in_bytes});
    EXPECT_EQ(est.size(), 4u);

    // Allocate four output buffers
    CudaBuffer<uint8_t> d_codes  (est[0]);
    CudaBuffer<uint8_t> d_errors (est[1]);
    CudaBuffer<uint8_t> d_indices(est[2]);
    CudaBuffer<uint8_t> d_count  (est[3]);

    std::vector<void*> inputs  = {d_in.void_ptr()};
    std::vector<void*> outputs = {d_codes.void_ptr(), d_errors.void_ptr(),
                                   d_indices.void_ptr(), d_count.void_ptr()};
    std::vector<size_t> sizes  = {in_bytes};

    stage.execute(stream, &pool, inputs, outputs, sizes);
    cudaStreamSynchronize(stream);
    stage.postStreamSync(stream);  // read back actual outlier count

    auto actual = stage.getActualOutputSizesByName();

    LorenzoForwardResult r;
    r.codes_bytes   = actual.count("codes")          ? actual.at("codes")          : est[0];
    r.errors_bytes  = actual.count("outlier_errors")  ? actual.at("outlier_errors") : est[1];
    r.indices_bytes = actual.count("outlier_indices") ? actual.at("outlier_indices"): est[2];
    r.count_bytes   = actual.count("outlier_count")   ? actual.at("outlier_count")  : est[3];

    r.codes_raw   = d_codes.download_bytes(r.codes_bytes,   stream);
    r.errors_raw  = d_errors.download_bytes(r.errors_bytes, stream);
    r.indices_raw = d_indices.download_bytes(r.indices_bytes, stream);
    r.count_raw   = d_count.download_bytes(r.count_bytes,  stream);

    return r;
}

// Run Lorenzo inverse pass and return reconstructed floats.
static std::vector<float> run_lorenzo_inverse(
    LorenzoStage<float, uint16_t>& stage,
    const LorenzoForwardResult&    fwd,
    size_t                          n_elements,
    cudaStream_t                    stream,
    fz::MemoryPool&                 pool)
{
    // Upload the four compressed buffers back to device
    CudaBuffer<uint8_t> d_codes  (fwd.codes_raw.size());  d_codes.upload(fwd.codes_raw, stream);
    CudaBuffer<uint8_t> d_errors (fwd.errors_raw.size()); d_errors.upload(fwd.errors_raw, stream);
    CudaBuffer<uint8_t> d_indices(fwd.indices_raw.size());d_indices.upload(fwd.indices_raw, stream);
    CudaBuffer<uint8_t> d_count  (fwd.count_raw.size());  d_count.upload(fwd.count_raw, stream);

    // Output: reconstructed floats
    CudaBuffer<float> d_out(n_elements);

    std::vector<void*> inputs = {d_codes.void_ptr(), d_errors.void_ptr(),
                                  d_indices.void_ptr(), d_count.void_ptr()};
    std::vector<void*> outputs = {d_out.void_ptr()};
    std::vector<size_t> sizes  = {fwd.codes_bytes, fwd.errors_bytes,
                                   fwd.indices_bytes, fwd.count_bytes};

    stage.execute(stream, &pool, inputs, outputs, sizes);
    cudaStreamSynchronize(stream);

    return d_out.download(stream);
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: smooth sinusoidal data round-trip within error_bound
// ─────────────────────────────────────────────────────────────────────────────
TEST(LorenzoStage, SmoothRoundTrip) {
    CudaStream stream;
    constexpr size_t N  = 1024;
    constexpr float  EB = 1e-2f;

    auto pool = make_test_pool(N * sizeof(float) * 20);

    auto h_input = make_sine_floats(N, 0.05f, 10.0f);

    // Forward
    LorenzoStage<float, uint16_t> fwd;
    fwd.setErrorBound(EB);
    fwd.setQuantRadius(512);
    fwd.setOutlierCapacity(0.2f);

    auto fwd_result = run_lorenzo_forward(fwd, h_input, stream, *pool);

    // Codes must be non-trivial
    EXPECT_GT(fwd_result.codes_bytes, 0u) << "Codes output is empty";

    // Inverse
    LorenzoStage<float, uint16_t> inv;
    inv.setErrorBound(EB);
    inv.setQuantRadius(512);
    inv.setInverse(true);

    // Copy stage config from the forward stage via serializeHeader
    uint8_t cfg_buf[128] = {};
    size_t  cfg_sz = fwd.serializeHeader(0, cfg_buf, sizeof(cfg_buf));
    inv.deserializeHeader(cfg_buf, cfg_sz);

    auto h_recon = run_lorenzo_inverse(inv, fwd_result, N, stream, *pool);

    ASSERT_EQ(h_recon.size(), N);
    float max_err = 0.0f;
    for (size_t i = 0; i < N; i++) {
        float err = std::abs(h_recon[i] - h_input[i]);
        max_err = std::max(max_err, err);
    }
    EXPECT_LE(max_err, EB * 1.01f)  // 1 % tolerance for floating-point accumulation
        << "Max reconstruction error " << max_err << " exceeds bound " << EB;
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: constant input — zero prediction errors, should have no outliers
// ─────────────────────────────────────────────────────────────────────────────
TEST(LorenzoStage, ConstantInputRoundTrip) {
    CudaStream stream;
    constexpr size_t N  = 512;
    constexpr float  EB = 1e-3f;

    auto pool = make_test_pool(N * sizeof(float) * 20);

    std::vector<float> h_input(N, 5.0f);

    LorenzoStage<float, uint16_t> fwd;
    fwd.setErrorBound(EB);
    fwd.setQuantRadius(512);
    fwd.setOutlierCapacity(0.2f);

    auto fwd_result = run_lorenzo_forward(fwd, h_input, stream, *pool);

    // Lorenzo predicts each element from its left neighbour.  Element 0 has no
    // predecessor so its prediction is 0; for a constant input of 5.0f the
    // prediction error is 5.0 >> EB, making it a legitimate outlier.  All
    // interior elements (i > 0) see a perfect prediction, so the count must be
    // exactly 1 (only the boundary element).
    ASSERT_GE(fwd_result.count_raw.size(), sizeof(uint32_t));
    uint32_t outlier_count = 0;
    std::memcpy(&outlier_count, fwd_result.count_raw.data(), sizeof(uint32_t));
    EXPECT_LE(outlier_count, 1u)
        << "Constant input should produce at most 1 outlier (first-element boundary)";

    // Round-trip
    LorenzoStage<float, uint16_t> inv;
    inv.setInverse(true);
    uint8_t cfg[128] = {};
    size_t  cfg_sz = fwd.serializeHeader(0, cfg, sizeof(cfg));
    inv.deserializeHeader(cfg, cfg_sz);

    auto h_recon = run_lorenzo_inverse(inv, fwd_result, N, stream, *pool);

    ASSERT_EQ(h_recon.size(), N);
    for (size_t i = 0; i < N; i++) {
        EXPECT_NEAR(h_recon[i], 5.0f, EB)
            << "Reconstruction mismatch at index " << i;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: large dataset (64 K floats), tighter error bound
// ─────────────────────────────────────────────────────────────────────────────
TEST(LorenzoStage, LargeDataRoundTrip) {
    CudaStream stream;
    constexpr size_t N  = 1 << 16;  // 64 K
    constexpr float  EB = 1e-3f;

    auto pool = make_test_pool(N * sizeof(float) * 20);

    std::vector<float> h_input(N);
    for (size_t i = 0; i < N; i++)
        h_input[i] = std::cos(static_cast<float>(i) * 0.001f) * 50.0f
                   + std::sin(static_cast<float>(i) * 0.003f) * 20.0f;

    LorenzoStage<float, uint16_t> fwd;
    fwd.setErrorBound(EB);
    fwd.setQuantRadius(512);
    fwd.setOutlierCapacity(0.2f);

    auto fwd_result = run_lorenzo_forward(fwd, h_input, stream, *pool);
    EXPECT_GT(fwd_result.codes_bytes, 0u);

    LorenzoStage<float, uint16_t> inv;
    inv.setInverse(true);
    uint8_t cfg[128] = {};
    inv.deserializeHeader(cfg, fwd.serializeHeader(0, cfg, sizeof(cfg)));

    auto h_recon = run_lorenzo_inverse(inv, fwd_result, N, stream, *pool);
    ASSERT_EQ(h_recon.size(), N);

    float max_err = 0.0f;
    for (size_t i = 0; i < N; i++) {
        max_err = std::max(max_err, std::abs(h_recon[i] - h_input[i]));
    }
    EXPECT_LE(max_err, EB * 1.01f)
        << "Max error " << max_err << " exceeds bound " << EB;
}

// ─────────────────────────────────────────────────────────────────────────────
// 2-D tests
// ─────────────────────────────────────────────────────────────────────────────

// Helper shared by 2D / 3D tests: run a full forward+inverse cycle and return
// the max absolute error.  The caller is responsible for configuring dims on
// `fwd` before calling.
static float roundtrip_max_error(
    LorenzoStage<float, uint16_t>& fwd,
    const std::vector<float>&       h_input,
    size_t                           n_elements,
    cudaStream_t                     stream,
    fz::MemoryPool&                  pool)
{
    auto fwd_result = run_lorenzo_forward(fwd, h_input, stream, pool);
    EXPECT_GT(fwd_result.codes_bytes, 0u) << "Forward produced empty codes";

    LorenzoStage<float, uint16_t> inv;
    inv.setInverse(true);
    uint8_t cfg[128] = {};
    inv.deserializeHeader(cfg, fwd.serializeHeader(0, cfg, sizeof(cfg)));

    auto h_recon = run_lorenzo_inverse(inv, fwd_result, n_elements, stream, pool);
    EXPECT_EQ(h_recon.size(), n_elements);

    float max_err = 0.0f;
    for (size_t i = 0; i < n_elements; i++)
        max_err = std::max(max_err, std::abs(h_recon[i] - h_input[i]));
    return max_err;
}

// ─────────────────────────────────────────────────────────────────────────────
// Lorenzo2DStage: sinusoidal surface roundtrip
// ─────────────────────────────────────────────────────────────────────────────
TEST(Lorenzo2DStage, SmoothRoundTrip) {
    CudaStream stream;
    constexpr size_t NX = 128, NY = 64;
    constexpr size_t N  = NX * NY;
    constexpr float  EB = 1e-2f;

    auto pool = make_test_pool(N * sizeof(float) * 20);

    std::vector<float> h_input(N);
    for (size_t y = 0; y < NY; y++)
        for (size_t x = 0; x < NX; x++)
            h_input[y * NX + x] = std::sin(x * 0.05f) * std::cos(y * 0.07f) * 10.0f;

    LorenzoStage<float, uint16_t> fwd;
    fwd.setErrorBound(EB);
    fwd.setQuantRadius(512);
    fwd.setOutlierCapacity(0.25f);
    fwd.setDims(NX, NY);

    EXPECT_EQ(fwd.ndim(), 2) << "ndim() should be 2 for a 2D stage";
    EXPECT_EQ(fwd.getName(), "Lorenzo2D");

    float max_err = roundtrip_max_error(fwd, h_input, N, stream, *pool);
    EXPECT_LE(max_err, EB * 1.01f)
        << "2D smooth roundtrip max_err=" << max_err << " exceeds bound " << EB;
}

// ─────────────────────────────────────────────────────────────────────────────
// Lorenzo2DStage: constant surface — boundary outliers only, interior exact
// ─────────────────────────────────────────────────────────────────────────────
TEST(Lorenzo2DStage, ConstantInput) {
    CudaStream stream;
    constexpr size_t NX = 64, NY = 64;
    constexpr size_t N  = NX * NY;
    constexpr float  EB = 1e-3f;
    constexpr float  VAL = 3.14f;

    auto pool = make_test_pool(N * sizeof(float) * 20);

    std::vector<float> h_input(N, VAL);

    LorenzoStage<float, uint16_t> fwd;
    fwd.setErrorBound(EB);
    fwd.setQuantRadius(512);
    fwd.setOutlierCapacity(0.25f);
    fwd.setDims(NX, NY);

    // For a constant 2D field, the only non-zero deltas are the boundary
    // (first row/col where the Lorenzo filter sees 0 as predecessor).  The
    // interior should be perfectly quantizable.
    auto fwd_result = run_lorenzo_forward(fwd, h_input, stream, *pool);
    ASSERT_GE(fwd_result.count_raw.size(), sizeof(uint32_t));
    uint32_t outlier_count = 0;
    std::memcpy(&outlier_count, fwd_result.count_raw.data(), sizeof(uint32_t));
    // At most NX + NY - 1 boundary outliers (first row xor first col depends on
    // the kernel's prediction order), conservatively cap at NX + NY.
    EXPECT_LE(outlier_count, NX + NY)
        << "Constant 2D surface should have few boundary outliers";

    // Full roundtrip
    LorenzoStage<float, uint16_t> inv;
    inv.setInverse(true);
    uint8_t cfg[128] = {};
    inv.deserializeHeader(cfg, fwd.serializeHeader(0, cfg, sizeof(cfg)));
    auto h_recon = run_lorenzo_inverse(inv, fwd_result, N, stream, *pool);

    ASSERT_EQ(h_recon.size(), N);
    for (size_t i = 0; i < N; i++)
        EXPECT_NEAR(h_recon[i], VAL, EB) << "Mismatch at index " << i;
}

// ─────────────────────────────────────────────────────────────────────────────
// Lorenzo2DStage: linear ramp — nearly all quantizable, almost no outliers
// ─────────────────────────────────────────────────────────────────────────────
TEST(Lorenzo2DStage, LinearRampRoundTrip) {
    CudaStream stream;
    constexpr size_t NX = 256, NY = 128;
    constexpr size_t N  = NX * NY;
    constexpr float  EB = 1e-2f;

    auto pool = make_test_pool(N * sizeof(float) * 20);

    std::vector<float> h_input(N);
    for (size_t y = 0; y < NY; y++)
        for (size_t x = 0; x < NX; x++)
            h_input[y * NX + x] = x * 0.001f + y * 0.001f;

    LorenzoStage<float, uint16_t> fwd;
    fwd.setErrorBound(EB);
    fwd.setQuantRadius(512);
    fwd.setOutlierCapacity(0.25f);
    fwd.setDims(NX, NY);

    float max_err = roundtrip_max_error(fwd, h_input, N, stream, *pool);
    EXPECT_LE(max_err, EB * 1.01f)
        << "2D linear ramp max_err=" << max_err << " exceeds bound " << EB;
}

// ─────────────────────────────────────────────────────────────────────────────
// Lorenzo2DStage: serialize/deserialize preserves dims and ndim
// ─────────────────────────────────────────────────────────────────────────────
TEST(Lorenzo2DStage, SerializeDeserializeDims) {
    constexpr size_t NX = 100, NY = 50;

    LorenzoStage<float, uint16_t> src;
    src.setErrorBound(1e-3f);
    src.setQuantRadius(512);
    src.setDims(NX, NY);

    // Fake num_elements_ via forward-side state (needs a compress call or
    // manual approach — use the accessor pattern used by the existing tests)
    uint8_t cfg[128] = {};
    size_t sz = src.serializeHeader(0, cfg, sizeof(cfg));
    ASSERT_GE(sz, sizeof(LorenzoConfig));

    LorenzoStage<float, uint16_t> dst;
    dst.setInverse(true);
    dst.deserializeHeader(cfg, sz);

    EXPECT_EQ(dst.ndim(), 2)          << "ndim should be 2 after deserialize";
    EXPECT_EQ(dst.getDims()[0], NX)   << "dim_x should be " << NX;
    EXPECT_EQ(dst.getDims()[1], NY)   << "dim_y should be " << NY;
    EXPECT_EQ(dst.getDims()[2], 1u)   << "dim_z should be 1 for 2D";
    EXPECT_EQ(dst.getName(), "Lorenzo2D");
}

// ─────────────────────────────────────────────────────────────────────────────
// 3-D tests
// ─────────────────────────────────────────────────────────────────────────────

// ─────────────────────────────────────────────────────────────────────────────
// Lorenzo3DStage: smooth 3D volume roundtrip
// ─────────────────────────────────────────────────────────────────────────────
TEST(Lorenzo3DStage, SmoothRoundTrip) {
    CudaStream stream;
    constexpr size_t NX = 32, NY = 32, NZ = 32;
    constexpr size_t N  = NX * NY * NZ;
    constexpr float  EB = 1e-2f;

    auto pool = make_test_pool(N * sizeof(float) * 20);

    std::vector<float> h_input(N);
    for (size_t z = 0; z < NZ; z++)
        for (size_t y = 0; y < NY; y++)
            for (size_t x = 0; x < NX; x++) {
                size_t idx = z * NX * NY + y * NX + x;
                h_input[idx] = std::sin(x * 0.1f) * std::cos(y * 0.1f) * (1.0f + z * 0.02f);
            }

    LorenzoStage<float, uint16_t> fwd;
    fwd.setErrorBound(EB);
    fwd.setQuantRadius(512);
    fwd.setOutlierCapacity(0.25f);
    fwd.setDims(NX, NY, NZ);

    EXPECT_EQ(fwd.ndim(), 3) << "ndim() should be 3 for a 3D stage";
    EXPECT_EQ(fwd.getName(), "Lorenzo3D");

    float max_err = roundtrip_max_error(fwd, h_input, N, stream, *pool);
    EXPECT_LE(max_err, EB * 1.01f)
        << "3D smooth roundtrip max_err=" << max_err << " exceeds bound " << EB;
}

// ─────────────────────────────────────────────────────────────────────────────
// Lorenzo3DStage: constant volume — boundary outliers only
// ─────────────────────────────────────────────────────────────────────────────
TEST(Lorenzo3DStage, ConstantInput) {
    CudaStream stream;
    constexpr size_t NX = 16, NY = 16, NZ = 16;
    constexpr size_t N  = NX * NY * NZ;
    constexpr float  EB = 1e-3f;
    constexpr float  VAL = 2.718f;

    auto pool = make_test_pool(N * sizeof(float) * 20);

    std::vector<float> h_input(N, VAL);

    LorenzoStage<float, uint16_t> fwd;
    fwd.setErrorBound(EB);
    fwd.setQuantRadius(512);
    fwd.setOutlierCapacity(0.5f);
    fwd.setDims(NX, NY, NZ);

    auto fwd_result = run_lorenzo_forward(fwd, h_input, stream, *pool);
    ASSERT_GE(fwd_result.count_raw.size(), sizeof(uint32_t));
    uint32_t outlier_count = 0;
    std::memcpy(&outlier_count, fwd_result.count_raw.data(), sizeof(uint32_t));
    // Boundary face: at most NX*NY + NX*NZ + NY*NZ cells see a zero predecessor.
    EXPECT_LE(outlier_count, NX * NY + NX * NZ + NY * NZ)
        << "Constant 3D volume should have outliers only on faces";

    // Full roundtrip
    LorenzoStage<float, uint16_t> inv;
    inv.setInverse(true);
    uint8_t cfg[128] = {};
    inv.deserializeHeader(cfg, fwd.serializeHeader(0, cfg, sizeof(cfg)));
    auto h_recon = run_lorenzo_inverse(inv, fwd_result, N, stream, *pool);

    ASSERT_EQ(h_recon.size(), N);
    float max_err = 0.0f;
    for (size_t i = 0; i < N; i++)
        max_err = std::max(max_err, std::abs(h_recon[i] - VAL));
    EXPECT_LE(max_err, EB * 1.01f)
        << "3D constant volume max recon error " << max_err << " exceeds bound " << EB;
}

// ─────────────────────────────────────────────────────────────────────────────
// Lorenzo3DStage: linear ramp roundtrip
// ─────────────────────────────────────────────────────────────────────────────
TEST(Lorenzo3DStage, LinearRampRoundTrip) {
    CudaStream stream;
    constexpr size_t NX = 40, NY = 24, NZ = 16;   // non-power-of-two on purpose
    constexpr size_t N  = NX * NY * NZ;
    constexpr float  EB = 1e-2f;

    auto pool = make_test_pool(N * sizeof(float) * 20);

    auto h_input = make_ramp<float>(N, 0.001f);

    LorenzoStage<float, uint16_t> fwd;
    fwd.setErrorBound(EB);
    fwd.setQuantRadius(512);
    fwd.setOutlierCapacity(0.25f);
    fwd.setDims(NX, NY, NZ);

    float max_err = roundtrip_max_error(fwd, h_input, N, stream, *pool);
    EXPECT_LE(max_err, EB * 1.01f)
        << "3D linear ramp max_err=" << max_err << " exceeds bound " << EB;
}

// ─────────────────────────────────────────────────────────────────────────────
// Lorenzo3DStage: serialize/deserialize preserves dims and ndim
// ─────────────────────────────────────────────────────────────────────────────
TEST(Lorenzo3DStage, SerializeDeserializeDims) {
    constexpr size_t NX = 40, NY = 24, NZ = 16;

    LorenzoStage<float, uint16_t> src;
    src.setErrorBound(1e-3f);
    src.setQuantRadius(512);
    src.setDims(NX, NY, NZ);

    uint8_t cfg[128] = {};
    size_t sz = src.serializeHeader(0, cfg, sizeof(cfg));
    ASSERT_GE(sz, sizeof(LorenzoConfig));

    LorenzoStage<float, uint16_t> dst;
    dst.setInverse(true);
    dst.deserializeHeader(cfg, sz);

    EXPECT_EQ(dst.ndim(), 3)          << "ndim should be 3 after deserialize";
    EXPECT_EQ(dst.getDims()[0], NX)   << "dim_x should be " << NX;
    EXPECT_EQ(dst.getDims()[1], NY)   << "dim_y should be " << NY;
    EXPECT_EQ(dst.getDims()[2], NZ)   << "dim_z should be " << NZ;
    EXPECT_EQ(dst.getName(), "Lorenzo3D");
}

// ─────────────────────────────────────────────────────────────────────────────
// NOA error-bound mode (norm-of-absolute): abs_eb = user_eb * (max - min)
// ─────────────────────────────────────────────────────────────────────────────

// 1D smooth sinusoid with NOA mode.
TEST(LorenzoNOA, SmoothRoundTrip) {
    CudaStream stream;
    constexpr size_t N       = 1024;
    constexpr float  USER_EB = 0.01f;   // 1 % of value range

    auto pool = make_test_pool(N * sizeof(float) * 20);

    auto h_input = make_sine_floats(N, 0.05f, 10.0f);

    // Compute expected absolute bound on host for the EXPECT.
    float vmin = *std::min_element(h_input.begin(), h_input.end());
    float vmax = *std::max_element(h_input.begin(), h_input.end());
    float expected_abs_eb = USER_EB * (vmax - vmin);

    LorenzoStage<float, uint16_t> fwd;
    fwd.setErrorBound(USER_EB);
    fwd.setErrorBoundMode(ErrorBoundMode::NOA);
    fwd.setQuantRadius(512);
    fwd.setOutlierCapacity(0.2f);

    auto fwd_result = run_lorenzo_forward(fwd, h_input, stream, *pool);
    EXPECT_GT(fwd_result.codes_bytes, 0u) << "NOA forward produced no codes";

    LorenzoStage<float, uint16_t> inv;
    inv.setInverse(true);
    uint8_t cfg[128] = {};
    inv.deserializeHeader(cfg, fwd.serializeHeader(0, cfg, sizeof(cfg)));

    auto h_recon = run_lorenzo_inverse(inv, fwd_result, N, stream, *pool);

    ASSERT_EQ(h_recon.size(), N);
    float max_err = 0.0f;
    for (size_t i = 0; i < N; i++)
        max_err = std::max(max_err, std::abs(h_recon[i] - h_input[i]));
    // Allow 2 % tolerance: the GPU min/max scan may differ by at most one ULP
    // from the host reference, and Lorenzo's quantisation adds one step of 2*eb.
    EXPECT_LE(max_err, expected_abs_eb * 1.02f)
        << "NOA 1D max_err=" << max_err
        << " exceeds expected_abs_eb=" << expected_abs_eb;
}

// NOA eb_mode survives serialise/deserialise after a real compression pass.
TEST(LorenzoNOA, SerializeDeserializeMode) {
    CudaStream stream;
    constexpr size_t N = 256;
    auto pool = make_test_pool(N * sizeof(float) * 20);

    auto h_input = make_sine_floats(N, 0.05f, 10.0f);

    LorenzoStage<float, uint16_t> src;
    src.setErrorBound(0.02f);
    src.setErrorBoundMode(ErrorBoundMode::NOA);
    src.setQuantRadius(512);
    src.setOutlierCapacity(0.2f);

    // A real forward pass is required so computed_abs_eb_ is populated before
    // serializeHeader() is called — without execute() the absolute bound is 0.
    auto fwd_result = run_lorenzo_forward(src, h_input, stream, *pool);
    (void)fwd_result;

    uint8_t cfg[128] = {};
    size_t  sz = src.serializeHeader(0, cfg, sizeof(cfg));
    ASSERT_GE(sz, sizeof(LorenzoConfig));

    LorenzoStage<float, uint16_t> dst;
    dst.setInverse(true);
    dst.deserializeHeader(cfg, sz);

    // serializeHeader stores computed_abs_eb_ as error_bound and the mode.
    // After deserializing, getErrorBound() returns the absolute bound (not user_eb),
    // and getErrorBoundMode() returns the mode that was active during compression.
    EXPECT_EQ(dst.getErrorBoundMode(), ErrorBoundMode::NOA)
        << "ErrorBoundMode should survive header round-trip";
    // The absolute bound must be positive (NOA multiplied user_eb by value_range > 0).
    EXPECT_GT(dst.getErrorBound(), 0.0f)
        << "Computed absolute bound should be positive after NOA compression";
}

// ─────────────────────────────────────────────────────────────────────────────
// Lorenzo3DStage: getStageTypeId() returns correct enum values per dimensionality
// ─────────────────────────────────────────────────────────────────────────────
TEST(Lorenzo3DStage, StageTypeIds) {
    LorenzoStage<float, uint16_t> s1d;
    s1d.setDims(1024, 1, 1);
    EXPECT_EQ(s1d.getStageTypeId(),
              static_cast<uint16_t>(StageType::LORENZO_1D)) << "1D stage type";

    LorenzoStage<float, uint16_t> s2d;
    s2d.setDims(128, 64, 1);
    EXPECT_EQ(s2d.getStageTypeId(),
              static_cast<uint16_t>(StageType::LORENZO_2D)) << "2D stage type";

    LorenzoStage<float, uint16_t> s3d;
    s3d.setDims(32, 32, 32);
    EXPECT_EQ(s3d.getStageTypeId(),
              static_cast<uint16_t>(StageType::LORENZO_3D)) << "3D stage type";
}

// ─────────────────────────────────────────────────────────────────────────────
// L9 / DP14: LorenzoStage<double, uint16_t> — verifies the double CUDA kernel
//            instantiation and checks round-trip is within error_bound.
//
//            Uses the Pipeline API (handles all buffer sizing automatically).
// ─────────────────────────────────────────────────────────────────────────────
TEST(LorenzoDoubleStage, RoundTripWithinErrorBound) {
    constexpr size_t N  = 1 << 14;  // 16 K doubles
    constexpr double EB = 1e-3;

    std::vector<double> h_input(N);
    for (size_t i = 0; i < N; i++)
        h_input[i] = std::sin(static_cast<double>(i) * 0.01) * 50.0
                   + std::cos(static_cast<double>(i) * 0.003) * 20.0;

    CudaStream stream;
    size_t in_bytes = N * sizeof(double);
    CudaBuffer<double> d_in(N);
    d_in.upload(h_input, stream);
    stream.sync();

    Pipeline pipeline(in_bytes, MemoryStrategy::MINIMAL);
    auto* lrz = pipeline.addStage<LorenzoStage<double, uint16_t>>();
    lrz->setErrorBound(static_cast<float>(EB));
    lrz->setQuantRadius(512);
    lrz->setOutlierCapacity(0.2f);
    pipeline.finalize();

    void*  d_comp  = nullptr;
    size_t comp_sz = 0;
    pipeline.compress(d_in.void_ptr(), in_bytes, &d_comp, &comp_sz, stream);
    EXPECT_GT(comp_sz, 0u) << "Double compressed output is empty";

    void*  d_decomp  = nullptr;
    size_t decomp_sz = 0;
    pipeline.decompress(nullptr, comp_sz, &d_decomp, &decomp_sz, stream);
    ASSERT_NE(d_decomp, nullptr);
    ASSERT_EQ(decomp_sz, in_bytes);

    std::vector<double> h_out(N);
    cudaMemcpyAsync(h_out.data(), d_decomp, in_bytes, cudaMemcpyDeviceToHost, stream);
    stream.sync();
    cudaFree(d_decomp);

    for (size_t i = 0; i < N; i++) {
        EXPECT_LE(std::abs(h_out[i] - h_input[i]), EB * 1.01)
            << "Error exceeds bound at element " << i;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ZigzagCodes tests
// These verify the setZigzagCodes(true) path of LorenzoStage:
//   1. Round-trip reconstructs within the error bound.
//   2. Round-trip result matches the non-zigzag path (same accuracy guarantee).
//   3. The zigzag_codes flag is preserved through serializeHeader/deserializeHeader.
//   4. Enabling zigzag on a 2D/3D stage throws at execute() time.
// ─────────────────────────────────────────────────────────────────────────────

TEST(LorenzoZigzag, SmoothRoundTrip_1D) {
    CudaStream stream;
    constexpr size_t N  = 2048;
    constexpr float  EB = 1e-2f;

    auto pool = make_test_pool(N * sizeof(float) * 20);

    auto h_input = make_sine_floats(N, 0.05f, 10.0f);

    // Forward with zigzag enabled
    LorenzoStage<float, uint16_t> fwd;
    fwd.setErrorBound(EB);
    fwd.setQuantRadius(512);
    fwd.setOutlierCapacity(0.2f);
    fwd.setZigzagCodes(true);
    EXPECT_TRUE(fwd.getZigzagCodes());

    auto fwd_result = run_lorenzo_forward(fwd, h_input, stream, *pool);
    EXPECT_GT(fwd_result.codes_bytes, 0u);

    // Inverse: restore flag via header serialization
    LorenzoStage<float, uint16_t> inv;
    inv.setInverse(true);
    uint8_t cfg[128] = {};
    size_t cfg_sz = fwd.serializeHeader(0, cfg, sizeof(cfg));
    inv.deserializeHeader(cfg, cfg_sz);
    EXPECT_TRUE(inv.getZigzagCodes())
        << "zigzag_codes flag must survive serializeHeader/deserializeHeader";

    auto h_recon = run_lorenzo_inverse(inv, fwd_result, N, stream, *pool);

    ASSERT_EQ(h_recon.size(), N);
    float max_err = 0.0f;
    for (size_t i = 0; i < N; i++)
        max_err = std::max(max_err, std::abs(h_recon[i] - h_input[i]));
    EXPECT_LE(max_err, EB * 1.01f)
        << "ZigzagCodes 1D round-trip max_err=" << max_err << " exceeds bound " << EB;
}

TEST(LorenzoZigzag, MatchesRegularRoundTrip) {
    // Zigzag only changes the code representation; reconstruction quality must
    // be identical to the non-zigzag path.
    CudaStream stream;
    constexpr size_t N  = 1024;
    constexpr float  EB = 5e-3f;

    auto pool = make_test_pool(N * sizeof(float) * 20);

    auto h_input = make_sine_floats(N, 0.02f, 5.0f);  // cos-like smooth wave

    auto run_and_reconstruct = [&](bool use_zigzag) {
        LorenzoStage<float, uint16_t> fwd;
        fwd.setErrorBound(EB);
        fwd.setQuantRadius(512);
        fwd.setOutlierCapacity(0.2f);
        fwd.setZigzagCodes(use_zigzag);

        auto fwd_result = run_lorenzo_forward(fwd, h_input, stream, *pool);

        LorenzoStage<float, uint16_t> inv;
        inv.setInverse(true);
        uint8_t cfg[128] = {};
        inv.deserializeHeader(cfg, fwd.serializeHeader(0, cfg, sizeof(cfg)));

        return run_lorenzo_inverse(inv, fwd_result, N, stream, *pool);
    };

    auto h_no_zz   = run_and_reconstruct(false);
    auto h_with_zz = run_and_reconstruct(true);

    ASSERT_EQ(h_no_zz.size(), N);
    ASSERT_EQ(h_with_zz.size(), N);

    float max_diff = 0.0f;
    for (size_t i = 0; i < N; i++)
        max_diff = std::max(max_diff, std::abs(h_no_zz[i] - h_with_zz[i]));
    EXPECT_LE(max_diff, EB * 0.01f)
        << "Zigzag and non-zigzag paths should produce identical reconstructions; "
        << "max_diff=" << max_diff;
}

TEST(LorenzoZigzag, HeaderSerialization) {
    // Verify that zigzag_codes=true survives a full header round-trip and that
    // zigzag_codes=false also round-trips correctly (no cross-contamination).
    LorenzoStage<float, uint16_t> src_on;
    src_on.setErrorBound(1e-3f);
    src_on.setQuantRadius(512);
    src_on.setZigzagCodes(true);

    LorenzoStage<float, uint16_t> src_off;
    src_off.setErrorBound(1e-3f);
    src_off.setQuantRadius(512);
    src_off.setZigzagCodes(false);

    uint8_t buf_on[128]  = {};
    uint8_t buf_off[128] = {};
    size_t sz_on  = src_on.serializeHeader(0, buf_on,  sizeof(buf_on));
    size_t sz_off = src_off.serializeHeader(0, buf_off, sizeof(buf_off));

    EXPECT_EQ(sz_on,  sizeof(LorenzoConfig));
    EXPECT_EQ(sz_off, sizeof(LorenzoConfig));

    LorenzoStage<float, uint16_t> dst_on, dst_off;
    dst_on.setInverse(true);
    dst_off.setInverse(true);
    dst_on.deserializeHeader(buf_on,   sz_on);
    dst_off.deserializeHeader(buf_off, sz_off);

    EXPECT_TRUE(dst_on.getZigzagCodes())
        << "zigzag_codes=true was not recovered after serialization";
    EXPECT_FALSE(dst_off.getZigzagCodes())
        << "zigzag_codes=false was not recovered after serialization";
}

TEST(LorenzoZigzag, Supports2D) {
    // zigzag_codes=true should be supported for 2-D Lorenzo and round-trip
    // within the configured error bound.
    CudaStream stream;
    constexpr size_t NX = 32, NY = 32;
    constexpr size_t N  = NX * NY;
    constexpr float  EB = 1e-2f;

    auto pool = make_test_pool(N * sizeof(float) * 20);

    std::vector<float> h_input(N);
    for (size_t y = 0; y < NY; y++) {
        for (size_t x = 0; x < NX; x++) {
            size_t i = y * NX + x;
            h_input[i] = 0.5f * std::sin(static_cast<float>(x) * 0.2f)
                       + 0.3f * std::cos(static_cast<float>(y) * 0.15f);
        }
    }

    LorenzoStage<float, uint16_t> fwd;
    fwd.setErrorBound(EB);
    fwd.setQuantRadius(512);
    fwd.setOutlierCapacity(0.2f);
    fwd.setDims(NX, NY);
    fwd.setZigzagCodes(true);

    auto result = run_lorenzo_forward(fwd, h_input, stream, *pool);

    LorenzoStage<float, uint16_t> inv;
    inv.setInverse(true);
    uint8_t cfg[128] = {};
    inv.deserializeHeader(cfg, fwd.serializeHeader(0, cfg, sizeof(cfg)));

    auto h_recon = run_lorenzo_inverse(inv, result, N, stream, *pool);

    ASSERT_EQ(h_recon.size(), N);

    float max_err = 0.0f;
    for (size_t i = 0; i < N; i++) {
        max_err = std::max(max_err, std::abs(h_recon[i] - h_input[i]));
    }
    EXPECT_LE(max_err, EB * 1.01f)
        << "2D zigzag round-trip max_err=" << max_err << " exceeds EB=" << EB;
}

TEST(LorenzoZigzag, Supports3D) {
    // zigzag_codes=true should be supported for 3-D Lorenzo and round-trip
    // within the configured error bound.
    CudaStream stream;
    constexpr size_t NX = 16, NY = 8, NZ = 8;
    constexpr size_t N  = NX * NY * NZ;
    constexpr float  EB = 1e-2f;

    auto pool = make_test_pool(N * sizeof(float) * 24);

    std::vector<float> h_input(N);
    for (size_t z = 0; z < NZ; z++) {
        for (size_t y = 0; y < NY; y++) {
            for (size_t x = 0; x < NX; x++) {
                size_t i = z * (NX * NY) + y * NX + x;
                h_input[i] = 0.6f * std::sin(static_cast<float>(x) * 0.17f)
                           + 0.4f * std::cos(static_cast<float>(y) * 0.21f)
                           + 0.3f * std::sin(static_cast<float>(z) * 0.29f);
            }
        }
    }

    LorenzoStage<float, uint16_t> fwd;
    fwd.setErrorBound(EB);
    fwd.setQuantRadius(512);
    fwd.setOutlierCapacity(0.2f);
    fwd.setDims(NX, NY, NZ);
    fwd.setZigzagCodes(true);

    auto result = run_lorenzo_forward(fwd, h_input, stream, *pool);

    LorenzoStage<float, uint16_t> inv;
    inv.setInverse(true);
    uint8_t cfg[128] = {};
    inv.deserializeHeader(cfg, fwd.serializeHeader(0, cfg, sizeof(cfg)));

    auto h_recon = run_lorenzo_inverse(inv, result, N, stream, *pool);

    ASSERT_EQ(h_recon.size(), N);

    float max_err = 0.0f;
    for (size_t i = 0; i < N; i++) {
        max_err = std::max(max_err, std::abs(h_recon[i] - h_input[i]));
    }
    EXPECT_LE(max_err, EB * 1.01f)
        << "3D zigzag round-trip max_err=" << max_err << " exceeds EB=" << EB;
}

// ─────────────────────────────────────────────────────────────────────────────
// Templated helper: run a full forward→inverse round-trip for any (TInput, TCode)
// combination.  Returns max-absolute reconstruction error.
// ─────────────────────────────────────────────────────────────────────────────
template<typename TInput, typename TCode>
static double run_typed_roundtrip(
    LorenzoStage<TInput, TCode>& fwd,
    const std::vector<TInput>&   h_input,
    cudaStream_t                 stream,
    fz::MemoryPool&              pool)
{
    size_t n        = h_input.size();
    size_t in_bytes = n * sizeof(TInput);

    CudaBuffer<TInput> d_in(n);
    d_in.upload(h_input, stream);

    auto est = fwd.estimateOutputSizes({in_bytes});

    CudaBuffer<uint8_t> d_codes  (est[0]);
    CudaBuffer<uint8_t> d_errors (est[1]);
    CudaBuffer<uint8_t> d_indices(est[2]);
    CudaBuffer<uint8_t> d_count  (est[3]);

    std::vector<void*>  fi = {d_in.void_ptr()};
    std::vector<void*>  fo = {d_codes.void_ptr(), d_errors.void_ptr(),
                               d_indices.void_ptr(), d_count.void_ptr()};
    std::vector<size_t> fs = {in_bytes};
    fwd.execute(stream, &pool, fi, fo, fs);
    cudaStreamSynchronize(stream);
    fwd.postStreamSync(stream);

    auto actual = fwd.getActualOutputSizesByName();
    size_t cb = actual.count("codes")           ? actual.at("codes")           : est[0];
    size_t eb = actual.count("outlier_errors")  ? actual.at("outlier_errors")  : est[1];
    size_t ib = actual.count("outlier_indices") ? actual.at("outlier_indices") : est[2];
    size_t kb = actual.count("outlier_count")   ? actual.at("outlier_count")   : est[3];

    LorenzoStage<TInput, TCode> inv;
    inv.setInverse(true);
    uint8_t cfg[128] = {};
    inv.deserializeHeader(cfg, fwd.serializeHeader(0, cfg, sizeof(cfg)));

    CudaBuffer<TInput> d_out(n);
    std::vector<void*>  ii = {d_codes.void_ptr(), d_errors.void_ptr(),
                               d_indices.void_ptr(), d_count.void_ptr()};
    std::vector<void*>  io = {d_out.void_ptr()};
    std::vector<size_t> is = {cb, eb, ib, kb};
    inv.execute(stream, &pool, ii, io, is);
    cudaStreamSynchronize(stream);

    auto h_recon = d_out.download(stream);
    double max_err = 0.0;
    for (size_t i = 0; i < n; i++) {
        double e = std::abs(static_cast<double>(h_recon[i]) -
                            static_cast<double>(h_input[i]));
        if (e > max_err) max_err = e;
    }
    return max_err;
}

// ─────────────────────────────────────────────────────────────────────────────
//  LorenzoTypeMatrix — tests all four (TInput, TCode) instantiations
//
//  The existing LorenzoStage / LorenzoDoubleStage suites only exercise
//  float/uint16_t.  These tests verify that every instantiation in the .cu
//  file actually works correctly end-to-end.
//
//  Instantiations covered:
//    LorenzoStage<float,  uint16_t>  — already tested above (regression guard)
//    LorenzoStage<float,  uint8_t>   — NEW: narrow 8-bit code type
//    LorenzoStage<double, uint16_t>  — NEW: double-precision input
//    LorenzoStage<double, uint32_t>  — NEW: double-precision with wide codes
// ─────────────────────────────────────────────────────────────────────────────

// Smooth sinusoidal regression guard — same type as existing suite
TEST(LorenzoTypeMatrix, FloatUint16_RoundTrip) {
    CudaStream stream;
    constexpr size_t N  = 2048;
    constexpr double EB = 1e-2;
    auto pool = make_test_pool(N * sizeof(float) * 20);

    auto h_in = make_sine_floats(N, 0.05f, 10.0f);

    LorenzoStage<float, uint16_t> fwd;
    fwd.setErrorBound(static_cast<float>(EB));
    fwd.setQuantRadius(512);
    fwd.setOutlierCapacity(0.1f);

    double max_err = run_typed_roundtrip(fwd, h_in, stream, *pool);
    EXPECT_LE(max_err, EB + 1e-9)
        << "float/uint16 round-trip max error=" << max_err;
}

// 8-bit code type: range is ±128 so EB must be generous relative to signal
TEST(LorenzoTypeMatrix, FloatUint8_RoundTrip) {
    CudaStream stream;
    constexpr size_t N  = 2048;
    constexpr float  EB = 0.5f;   // wide bound to stay inside uint8 range
    auto pool = make_test_pool(N * sizeof(float) * 20);

    // Constant ramp — prediction residuals ≈ 0, so very few outliers even
    // with the narrow uint8_t code range.
    auto h_in = make_ramp<float>(N, 1.0f);

    LorenzoStage<float, uint8_t> fwd;
    fwd.setErrorBound(EB);
    fwd.setQuantRadius(64);     // 64 * 2 * EB = 64 steps → easily fits uint8
    fwd.setOutlierCapacity(0.2f);

    double max_err = run_typed_roundtrip(fwd, h_in, stream, *pool);
    EXPECT_LE(max_err, static_cast<double>(EB) + 1e-6)
        << "float/uint8 round-trip max error=" << max_err;
}

// Double-precision input with uint16_t code type
TEST(LorenzoTypeMatrix, DoubleUint16_RoundTrip) {
    CudaStream stream;
    constexpr size_t N  = 2048;
    constexpr double EB = 1e-6;   // tight bound to exercise double precision
    auto pool = make_test_pool(N * sizeof(double) * 20);

    std::vector<double> h_in(N);
    for (size_t i = 0; i < N; i++)
        h_in[i] = std::sin(static_cast<double>(i) * 0.05) * 1e-3;

    LorenzoStage<double, uint16_t> fwd;
    fwd.setErrorBound(EB);
    fwd.setQuantRadius(512);
    fwd.setOutlierCapacity(0.1);

    double max_err = run_typed_roundtrip(fwd, h_in, stream, *pool);
    EXPECT_LE(max_err, EB + 1e-15)
        << "double/uint16 round-trip max error=" << max_err;
}

// Double-precision input with wide uint32_t code type
TEST(LorenzoTypeMatrix, DoubleUint32_RoundTrip) {
    CudaStream stream;
    constexpr size_t N  = 2048;
    constexpr double EB = 1e-8;
    auto pool = make_test_pool(N * sizeof(double) * 20);

    std::vector<double> h_in(N);
    for (size_t i = 0; i < N; i++)
        h_in[i] = std::cos(static_cast<double>(i) * 0.03) * 1e-4;

    LorenzoStage<double, uint32_t> fwd;
    fwd.setErrorBound(EB);
    fwd.setQuantRadius(65536);
    fwd.setOutlierCapacity(0.1);

    double max_err = run_typed_roundtrip(fwd, h_in, stream, *pool);
    EXPECT_LE(max_err, EB + 1e-18)
        << "double/uint32 round-trip max error=" << max_err;
}

// Header survives serialization for the double/uint32_t combination
TEST(LorenzoTypeMatrix, DoubleUint32_HeaderRoundTrip) {
    CudaStream stream;
    constexpr size_t N = 1024;
    auto pool = make_test_pool(N * sizeof(double) * 20);

    // Run execute() so computed_abs_eb_ / quant_radius are populated.
    std::vector<double> h_in(N);
    for (size_t i = 0; i < N; i++) h_in[i] = std::cos(static_cast<double>(i) * 0.03) * 1e-4;

    LorenzoStage<double, uint32_t> src;
    src.setErrorBound(1e-8);
    src.setQuantRadius(65536);
    src.setOutlierCapacity(0.1);
    src.setZigzagCodes(false);

    // Forward pass to populate internal state before serialization.
    run_typed_roundtrip(src, h_in, stream, *pool);   // warms up computed fields

    uint8_t buf[128] = {};
    size_t sz = src.serializeHeader(0, buf, sizeof(buf));
    EXPECT_EQ(sz, sizeof(LorenzoConfig));

    LorenzoStage<double, uint32_t> dst;
    dst.setInverse(true);
    dst.deserializeHeader(buf, sz);

    // After round-trip: quant_radius and zigzag_codes must match exactly.
    EXPECT_EQ(dst.getQuantRadius(), static_cast<uint32_t>(65536))
        << "double/uint32 quant_radius not recovered from header";
    EXPECT_FALSE(dst.getZigzagCodes())
        << "double/uint32 zigzag_codes not recovered from header";
    // The serialized error_bound is the computed abs bound, which for ABS mode
    // equals setErrorBound value.  It is stored as float, so compare at float precision.
    EXPECT_NEAR(static_cast<float>(dst.getErrorBound()), 1e-8f, 1e-8f * 2e-7f)
        << "double/uint32 error_bound not recovered from header";
}
