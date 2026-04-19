#include "helpers/test_stages.h"
#include <cuda_runtime.h>

namespace fz_test {

__global__ void scale_kernel_impl(float* out, const float* in, size_t n, float factor) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = in[idx] * factor;
}

void launch_scale_kernel(float* out, const float* in, size_t n, float factor,
                         cudaStream_t stream) {
    int block = 256;
    int grid  = static_cast<int>((n + block - 1) / block);
    scale_kernel_impl<<<grid, block, 0, stream>>>(out, in, n, factor);
}

} // namespace fz_test
