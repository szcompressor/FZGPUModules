/**
 * @file cdf97_outlier_correct_stage.cu
 * @brief Cdf97OutlierCorrectStage::execute() — see the header's doc comment
 *        for the mechanism and why the port shape is what it is.
 */

#include "coders/cdf97_outlier_correct/cdf97_outlier_correct_stage.h"
#include "transforms/cdf97/cdf97_kernels.cuh"
#include "cuda_check.h"
#include <cub/cub.cuh>
#include <stdexcept>
#include <string>

namespace fz {

__global__ static void oc_k_dequant_linear(const int32_t* code, int n, float ebx2, float* coeff) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    coeff[i] = (float)code[i] * ebx2;
}
__global__ static void oc_k_outlier_flag(const float* original, const float* trial, int n, float bound,
                                         int* flag, float* corr_val) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    float err = original[i] - trial[i];
    bool out = fabsf(err) > bound;
    flag[i] = out ? 1 : 0;
    corr_val[i] = out ? err : 0.0f;
}
__global__ static void oc_k_pack(const int* flag, const int* rank, const float* corr_val, int n,
                                 uint32_t* out_idx, float* out_val) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    if (flag[i]) { int r = rank[i]; out_idx[r] = (uint32_t)i; out_val[r] = corr_val[i]; }
}
__global__ static void oc_k_apply(const uint32_t* idx, const float* val, int count, float* field) {
    int j = blockIdx.x * blockDim.x + threadIdx.x; if (j >= count) return;
    field[idx[j]] += val[j];
}

struct Cdf97OutlierCorrectStage::Impl {
    int nx = 0, ny = 0; size_t n = 0;
    float* d_coeff = nullptr;
    int* d_flag = nullptr; float* d_corrval = nullptr; int* d_rank = nullptr;
    void* d_tmp = nullptr; size_t tmpb = 0;

    ~Impl() { freeAll(); }

    void freeAll() {
        void** ptrs[] = { (void**)&d_coeff, (void**)&d_flag, (void**)&d_corrval, (void**)&d_rank, &d_tmp };
        for (void** p : ptrs) { if (*p) FZ_CUDA_CHECK_WARN(cudaFree(*p)); *p = nullptr; }
        nx = ny = 0; n = 0; tmpb = 0;
    }

    bool sameShape(int nx_, int ny_) const { return nx == nx_ && ny == ny_ && n > 0; }

    void ensureShape(int nx_, int ny_) {
        if (sameShape(nx_, ny_)) return;
        freeAll();
        nx = nx_; ny = ny_; n = (size_t)nx * ny;
        FZ_CUDA_CHECK(cudaMalloc(&d_coeff, n * 4));
        FZ_CUDA_CHECK(cudaMalloc(&d_flag, n * 4));
        FZ_CUDA_CHECK(cudaMalloc(&d_corrval, n * 4));
        FZ_CUDA_CHECK(cudaMalloc(&d_rank, n * 4));
        cub::DeviceScan::ExclusiveSum(d_tmp, tmpb, d_flag, d_rank, (int)n);
        FZ_CUDA_CHECK(cudaMalloc(&d_tmp, tmpb));
    }
};

Cdf97OutlierCorrectStage::~Cdf97OutlierCorrectStage() { delete impl_; }

void Cdf97OutlierCorrectStage::execute(cudaStream_t stream, MemoryPool* /*pool*/,
                                       const std::vector<void*>& inputs,
                                       const std::vector<void*>& outputs,
                                       const std::vector<size_t>& sizes)
{
    if (inputs.size() < 2 || outputs.size() < 2 || sizes.size() < 2)
        throw std::runtime_error("Cdf97OutlierCorrectStage: needs 2 inputs/outputs/sizes");
    if (dims_[0] == 0 || dims_[1] == 0)
        throw std::runtime_error("Cdf97OutlierCorrectStage: dimensions not set — call setDims() first");
    if (dims_[2] > 1)
        throw std::runtime_error("Cdf97OutlierCorrectStage: 3-D not supported");

    const int nx = (int)dims_[0], ny = (int)dims_[1];
    const size_t n = (size_t)nx * ny;
    if (!impl_) impl_ = new Impl();
    impl_->ensureShape(nx, ny);
    Impl& I = *impl_;
    auto g = [&](int cnt) { return dim3((unsigned)((cnt + 255) / 256)); };

    // codes passthrough (output[1]) is identical in both directions: pure
    // copy of input[1] -- this stage never modifies the codes, it only reads
    // them (see the header's doc comment on why this port exists at all).
    if (outputs[1] != inputs[1])
        FZ_CUDA_CHECK(cudaMemcpyAsync(outputs[1], inputs[1], n * 4, cudaMemcpyDeviceToDevice, stream));
    actual_size1_ = n * 4;

    const float ebx2 = 2.0f * error_bound_;

    if (!is_inverse_) {
        // ── forward: compute the correction stream ──────────────────────────
        const float* d_field = static_cast<const float*>(inputs[0]);
        const int32_t* d_codes = static_cast<const int32_t*>(inputs[1]);

        oc_k_dequant_linear<<<g((int)n), 256, 0, stream>>>(d_codes, (int)n, ebx2, I.d_coeff);
        cdf97::dwt2d<float>(I.d_coeff, nx, ny, /*inverse=*/true, stream);

        oc_k_outlier_flag<<<g((int)n), 256, 0, stream>>>(d_field, I.d_coeff, (int)n, error_bound_, I.d_flag, I.d_corrval);
        cub::DeviceScan::ExclusiveSum(I.d_tmp, I.tmpb, I.d_flag, I.d_rank, (int)n, stream);
        int last_flag, last_rank;
        FZ_CUDA_CHECK(cudaMemcpyAsync(&last_flag, I.d_flag + n - 1, 4, cudaMemcpyDeviceToHost, stream));
        FZ_CUDA_CHECK(cudaMemcpyAsync(&last_rank, I.d_rank + n - 1, 4, cudaMemcpyDeviceToHost, stream));
        FZ_CUDA_CHECK(cudaStreamSynchronize(stream));   // accepted sync -- see isGraphCompatible() doc
        const uint32_t num_out = (uint32_t)(last_rank + last_flag);

        uint8_t* out = static_cast<uint8_t*>(outputs[0]);
        FZ_CUDA_CHECK(cudaMemcpyAsync(out, &num_out, 4, cudaMemcpyHostToDevice, stream));
        uint32_t* idx_ptr = reinterpret_cast<uint32_t*>(out + 4);
        float*    val_ptr = reinterpret_cast<float*>(out + 4 + (size_t)num_out * 4);
        oc_k_pack<<<g((int)n), 256, 0, stream>>>(I.d_flag, I.d_rank, I.d_corrval, (int)n, idx_ptr, val_ptr);

        actual_size0_ = 4 + (size_t)num_out * 8;
    } else {
        // ── inverse: recompute the trial reconstruction, apply corrections ──
        const uint8_t* corr = static_cast<const uint8_t*>(inputs[0]);
        const int32_t* d_codes = static_cast<const int32_t*>(inputs[1]);
        float* d_out = static_cast<float*>(outputs[0]);

        oc_k_dequant_linear<<<g((int)n), 256, 0, stream>>>(d_codes, (int)n, ebx2, d_out);
        cdf97::dwt2d<float>(d_out, nx, ny, /*inverse=*/true, stream);

        uint32_t num_out = 0;
        FZ_CUDA_CHECK(cudaMemcpy(&num_out, corr, 4, cudaMemcpyDeviceToHost));   // accepted sync
        if (num_out > 0) {
            const uint32_t* idx_ptr = reinterpret_cast<const uint32_t*>(corr + 4);
            const float*    val_ptr = reinterpret_cast<const float*>(corr + 4 + (size_t)num_out * 4);
            oc_k_apply<<<g((int)num_out), 256, 0, stream>>>(idx_ptr, val_ptr, (int)num_out, d_out);
        }
        actual_size0_ = n * 4;
    }
    FZ_CUDA_CHECK(cudaGetLastError());
}

} // namespace fz
