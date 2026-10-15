/**
 * tests/stages/test_lorenzo_quant_shared_init.cu
 *
 * Regression for the 3-D LorenzoQuant forward kernel reading uninitialized shared
 * memory.
 *
 *   LZQ1 LorenzoQuantStage/Forward3DIgnoresStaleSharedMemory
 *
 * The y-direction difference took `(threadIdx.y > 0) * s[threadIdx.y][...]`, which for
 * threadIdx.y == 0 reads ghost row 0 of the tile's shared array. That row is never
 * written, so the value is whatever an earlier kernel left there, and multiplying it by
 * zero does not discard it: 0 * NaN and 0 * Inf are NaN. In `fzgmod-cli -b` the Huffman
 * kernels of the first compress left such values, so the second compress turned
 * elements into outliers carrying NaN errors, which decoded to NaN (the report showed
 * psnr/nrmse as null). The first kernel in a process sees zeroed shared memory, which
 * is why one compress per process was correct.
 *
 * The test fills shared memory with NaN on every SM, then compresses a 3-D field and
 * requires every reconstructed value to be finite and within the bound.
 */

#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

#include "pipeline/compressor.h"
#include "fused/lorenzo_quant/lorenzo_quant.h"
#include "helpers/fz_test_utils.h"

using namespace fz;
using namespace fz_test;

namespace {

__global__ void fill_shared_with_nan(int n_words) {
    extern __shared__ unsigned smem[];
    for (int i = threadIdx.x; i < n_words; i += blockDim.x)
        smem[i] = 0x7fc00000u;   // quiet NaN as float
}

// Leave NaN in the shared memory of every SM, with the default 48 KB per block so
// the SM keeps the carveout the Lorenzo kernel runs with.
void poison_shared_memory() {
    int dev = 0, n_sm = 0;
    ASSERT_EQ(cudaGetDevice(&dev), cudaSuccess);
    ASSERT_EQ(cudaDeviceGetAttribute(&n_sm, cudaDevAttrMultiProcessorCount, dev), cudaSuccess);
    const int bytes = 48 * 1024;
    fill_shared_with_nan<<<32 * n_sm, 256, bytes>>>(bytes / 4);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
}

}  // namespace

TEST(LorenzoQuantStage, Forward3DIgnoresStaleSharedMemory) {
    const size_t nx = 37, ny = 13, nz = 11;   // none a multiple of the 32x8x8 tile
    const size_t n = nx * ny * nz;
    const size_t bytes = n * sizeof(float);
    const float eb = 1e-2f;
    auto h_in = make_smooth_data<float>(n);

    Pipeline p(bytes, MemoryStrategy::PREALLOCATE);
    p.setDims(nx, ny, nz);   // selects the 3-D kernel
    auto* lq = p.addStage<LorenzoQuantStage<float, uint16_t>>();
    lq->setErrorBound(eb);
    lq->setQuantRadius(512);
    lq->setOutlierCapacity(0.2f);
    p.finalize();

    float* d_in = nullptr;
    ASSERT_EQ(cudaMalloc(&d_in, bytes), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice), cudaSuccess);

    poison_shared_memory();
    CudaStream cs;
    void* d_comp = nullptr;
    size_t comp_size = 0;
    p.compress(d_in, bytes, &d_comp, &comp_size, cs.stream);
    void* d_out = nullptr;
    size_t out_size = 0;
    p.decompress(d_comp, comp_size, &d_out, &out_size, cs.stream);
    ASSERT_EQ(cudaStreamSynchronize(cs.stream), cudaSuccess);
    ASSERT_EQ(out_size, bytes);

    std::vector<float> h_out(n);
    ASSERT_EQ(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost), cudaSuccess);
    if (!p.isPoolManagedDecompOutput()) cudaFree(d_out);
    cudaFree(d_in);

    size_t bad = 0;
    for (size_t i = 0; i < n; ++i)
        if (!std::isfinite(h_out[i]) || std::fabs(h_out[i] - h_in[i]) > eb * 1.01f) ++bad;
    EXPECT_EQ(bad, 0u) << bad << " of " << n << " values are non-finite or out of bound";
}
