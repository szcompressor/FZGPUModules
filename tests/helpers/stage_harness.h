#pragma once

/**
 * tests/helpers/stage_harness.h
 *
 * High-level test helpers that wrap the Pipeline API so individual test cases
 * can focus on stage setup and assertions instead of boilerplate.
 *
 * Two primary helpers:
 *
 *   pipeline_round_trip<T>(pipeline, input, stream)
 *     — compress → decompress in memory, return reconstructed data + stats.
 *
 *   pipeline_file_round_trip<T>(pipeline, input, stream, path)
 *     — compress → writeToFile → decompressFromFile, return reconstructed data.
 *
 * These replace ~30 lines of upload/compress/decompress/download boilerplate
 * with a single call.  Stage setup (addStage, connect, finalize) is left to
 * the caller so it remains explicit and readable in each test.
 *
 * Also provides:
 *
 *   max_abs_error_typed<T>()   — double-precision max absolute error for
 *                                 any arithmetic element type.
 *
 *   make_smooth_data<T>()      — smooth sinusoid for any floating-point T.
 */

#include "helpers/fz_test_utils.h"
#include "fzgpumodules.h"

#include <cmath>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

namespace fz_test {

// ─────────────────────────────────────────────────────────────────────────────
// RoundTripResult — returned by the two pipeline helpers below.
// ─────────────────────────────────────────────────────────────────────────────
template <typename T>
struct RoundTripResult {
    std::vector<T> data;            // reconstructed elements
    double         max_error;       // max |orig[i] - recon[i]|, double precision
    size_t         compressed_bytes;// byte count of the compressed buffer
};

// ─────────────────────────────────────────────────────────────────────────────
// Generic max absolute error — double precision regardless of T.
// Fails the enclosing GTest if sizes differ.
// ─────────────────────────────────────────────────────────────────────────────
template <typename T>
inline double max_abs_error_typed(const std::vector<T>& orig,
                                   const std::vector<T>& recon) {
    EXPECT_EQ(orig.size(), recon.size()) << "max_abs_error_typed: size mismatch";
    double err = 0.0;
    size_t n = std::min(orig.size(), recon.size());
    for (size_t i = 0; i < n; i++) {
        double d = std::abs(static_cast<double>(recon[i]) - static_cast<double>(orig[i]));
        if (d > err) err = d;
    }
    return err;
}

// ─────────────────────────────────────────────────────────────────────────────
// pipeline_round_trip
//
// Compresses h_input through a pre-finalized Pipeline, then decompresses the
// result back to host memory and returns the reconstructed data + stats.
//
// Preconditions:
//   - pipeline is finalized and ready for the first compress() call.
//   - h_input contains N elements of type T; in_bytes = N * sizeof(T).
//
// The decompressed buffer is cudaMalloc-owned by the caller per the default
// Pipeline ownership model; this function frees it before returning.
//
// Usage:
//   auto res = pipeline_round_trip<float>(pipeline, h_input, stream);
//   EXPECT_LE(res.max_error, error_bound);
// ─────────────────────────────────────────────────────────────────────────────
template <typename T>
inline RoundTripResult<T> pipeline_round_trip(
    fz::Pipeline&              pipeline,
    const std::vector<T>&      h_input,
    cudaStream_t               stream)
{
    const size_t in_bytes = h_input.size() * sizeof(T);

    CudaBuffer<T> d_in(h_input.size());
    d_in.upload(h_input, stream);
    cudaStreamSynchronize(stream);

    void*  d_comp  = nullptr;
    size_t comp_sz = 0;
    pipeline.compress(d_in.void_ptr(), in_bytes, &d_comp, &comp_sz, stream);

    void*  d_dec  = nullptr;
    size_t dec_sz = 0;
    pipeline.decompress(d_comp, comp_sz, &d_dec, &dec_sz, stream);
    cudaStreamSynchronize(stream);

    const size_t n_out = dec_sz / sizeof(T);
    std::vector<T> h_recon(n_out);
    cudaError_t cp_err = cudaMemcpy(h_recon.data(), d_dec, dec_sz, cudaMemcpyDeviceToHost);
    EXPECT_EQ(cp_err, cudaSuccess) << "cudaMemcpy (decompress): " << cudaGetErrorString(cp_err);
    cudaFree(d_dec);

    RoundTripResult<T> r;
    r.data             = std::move(h_recon);
    r.compressed_bytes = comp_sz;
    r.max_error        = max_abs_error_typed<T>(h_input, r.data);
    return r;
}

// ─────────────────────────────────────────────────────────────────────────────
// pipeline_file_round_trip
//
// Like pipeline_round_trip, but the compressed data is written to tmp_path via
// writeToFile() and reconstructed via decompressFromFile().  Exercises the full
// serialization path: stage configs, FZM header, checksums, StageFactory.
//
// The temporary file is left on disk; the caller is responsible for removing
// it (typically with std::remove in a test teardown or at the end of the test).
//
// Usage:
//   auto res = pipeline_file_round_trip<float>(pipeline, h_input, stream,
//                                              "/tmp/test.fzm");
//   EXPECT_LE(res.max_error, error_bound);
//   std::remove("/tmp/test.fzm");
// ─────────────────────────────────────────────────────────────────────────────
template <typename T>
inline RoundTripResult<T> pipeline_file_round_trip(
    fz::Pipeline&              pipeline,
    const std::vector<T>&      h_input,
    cudaStream_t               stream,
    const std::string&         tmp_path)
{
    const size_t in_bytes = h_input.size() * sizeof(T);

    CudaBuffer<T> d_in(h_input.size());
    d_in.upload(h_input, stream);
    cudaStreamSynchronize(stream);

    void*  d_comp  = nullptr;
    size_t comp_sz = 0;
    pipeline.compress(d_in.void_ptr(), in_bytes, &d_comp, &comp_sz, stream);
    cudaStreamSynchronize(stream);
    pipeline.writeToFile(tmp_path, stream);

    void*  d_dec  = nullptr;
    size_t dec_sz = 0;
    fz::Pipeline::decompressFromFile(tmp_path, &d_dec, &dec_sz, stream);
    cudaStreamSynchronize(stream);

    const size_t n_out = dec_sz / sizeof(T);
    std::vector<T> h_recon(n_out);
    cudaError_t cp_err = cudaMemcpy(h_recon.data(), d_dec, dec_sz, cudaMemcpyDeviceToHost);
    EXPECT_EQ(cp_err, cudaSuccess) << "cudaMemcpy (file decompress): " << cudaGetErrorString(cp_err);
    cudaFree(d_dec);

    RoundTripResult<T> r;
    r.data             = std::move(h_recon);
    r.compressed_bytes = comp_sz;
    r.max_error        = max_abs_error_typed<T>(h_input, r.data);
    return r;
}

} // namespace fz_test
