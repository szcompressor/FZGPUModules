/**
 * tests/pipeline/test_lc_presets.cpp
 *
 * Loads the cuSZ-Hi TOML presets (throughput + ratio modes) and verifies they
 * round-trip within the error bound, then verifies a saveConfig→loadConfig
 * round-trip preserves the pipeline (exercises the Zigzag `byte_transparent`
 * flag and MergeStage `segments` serialization).
 *
 * Both presets use GInterpStage (spline interpolation) as the predictor,
 * matching the cuSZ-Hi Spline path, followed by the LC lossless back-end.
 * GInterp accumulates interpolation errors across levels, so the tolerance is
 * 2× eb (the documented worst-case for data with many outliers; smooth data
 * typically stays within 1.1×).
 *
 *   LCP1  LCPresets/ThroughputRoundTrip   — cusz_hi_tp.toml loads + round-trips
 *   LCP2  LCPresets/RatioRoundTrip        — cusz_hi_cr.toml loads + round-trips
 *   LCP3  LCPresets/SaveLoadRoundTrip     — save→load of the tp pipeline round-trips
 */

#include <gtest/gtest.h>
#include "helpers/fz_test_utils.h"
#include "fzgpumodules.h"

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#ifndef FZ_PRESETS_DIR
#error "FZ_PRESETS_DIR must be defined by CMake (path to examples/presets)."
#endif

using namespace fz;
using namespace fz_test;

static constexpr size_t NX = 512, NY = 512;
// Matches error_bound in both cusz_hi_*.toml (NOA mode).
static constexpr float PRESET_EB = 1e-3f;

// Both presets use error_bound_mode = "NOA": abs_eb = preset_eb × (vmax - vmin).
// make_smooth_data defaults (amp1=50, amp2=20) give range ≈ 140, so abs_eb ≈ 0.14.
// GInterp accumulates multi-level errors; docs bound ≤ 2× abs_eb worst-case.
// Tolerance is computed per-test from the actual data range rather than hardcoded.

struct RoundTripOut {
    float max_error;
    float data_range;  // vmax - vmin of the input, needed for NOA tolerance
};

static RoundTripOut preset_round_trip(Pipeline& p) {
    auto h_in = make_smooth_data<float>(NX * NY);
    const float vmin = *std::min_element(h_in.begin(), h_in.end());
    const float vmax = *std::max_element(h_in.begin(), h_in.end());
    const size_t in_bytes = h_in.size() * sizeof(float);
    CudaStream stream;

    CudaBuffer<float> d_in(h_in.size());
    d_in.upload(h_in, stream);
    stream.sync();

    void* d_comp = nullptr; size_t comp_sz = 0;
    p.compress(d_in.void_ptr(), in_bytes, &d_comp, &comp_sz, stream);

    void* d_dec = nullptr; size_t dec_sz = 0;
    p.decompress(d_comp, comp_sz, &d_dec, &dec_sz, stream);
    stream.sync();

    std::vector<float> recon(dec_sz / sizeof(float));
    cudaMemcpy(recon.data(), d_dec, dec_sz, cudaMemcpyDeviceToHost);
    if (!p.isPoolManagedDecompOutput()) cudaFree(d_dec);
    return {max_abs_error(h_in, recon), vmax - vmin};
}

TEST(LCPresets, ThroughputRoundTrip) {
    Pipeline p;
    p.setDims(NX, NY, 1);  // presets carry no dims; caller must set them
    p.loadConfig(std::string(FZ_PRESETS_DIR) + "/cusz_hi_tp.toml");
    auto r = preset_round_trip(p);
    const float abs_eb  = PRESET_EB * r.data_range;
    const float eb_tol  = abs_eb * 2.0f;
    EXPECT_LE(r.max_error, eb_tol)
        << "abs_eb=" << abs_eb << " data_range=" << r.data_range;
}

TEST(LCPresets, RatioRoundTrip) {
    Pipeline p;
    p.setDims(NX, NY, 1);  // presets carry no dims; caller must set them
    p.loadConfig(std::string(FZ_PRESETS_DIR) + "/cusz_hi_cr.toml");
    auto r = preset_round_trip(p);
    const float abs_eb  = PRESET_EB * r.data_range;
    const float eb_tol  = abs_eb * 2.0f;
    EXPECT_LE(r.max_error, eb_tol)
        << "abs_eb=" << abs_eb << " data_range=" << r.data_range;
}

TEST(LCPresets, SaveLoadRoundTrip) {
    // Load tp, save it back out, reload the saved copy, and round-trip — this
    // exercises serialization of Zigzag byte_transparent + Merge segments.
    Pipeline p;
    p.setDims(NX, NY, 1);  // presets carry no dims; caller must set them
    p.loadConfig(std::string(FZ_PRESETS_DIR) + "/cusz_hi_tp.toml");
    const std::string saved = "/tmp/fzgmod_lc_tp_saved.toml";
    p.saveConfig(saved);

    // The saved config must mention the byte-transparent TCMS and the Merge segments.
    std::ifstream f(saved);
    std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    EXPECT_NE(text.find("byte_transparent = true"), std::string::npos)
        << "saved config lost Zigzag byte_transparent flag";
    EXPECT_NE(text.find("segments"), std::string::npos)
        << "saved config lost Merge segments";

    Pipeline p2;
    p2.loadConfig(saved);
    auto r = preset_round_trip(p2);
    const float abs_eb = PRESET_EB * r.data_range;
    EXPECT_LE(r.max_error, abs_eb * 1.05f)
        << "abs_eb=" << abs_eb << " data_range=" << r.data_range;
    std::remove(saved.c_str());
}
