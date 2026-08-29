#pragma once

/**
 * @file outlier_correct_stage_impl.cuh
 * @brief Out-of-line member definitions for `OutlierCorrectStage<Reconstructor>`
 *        (declared in `outlier_correct_stage.h`). CUDA device code -- include
 *        only from a .cu file that also defines its Reconstructor's
 *        `applyInverseTransform()` and explicit-instantiates the template
 *        (see `cdf97_outlier_correct_stage.cu`), never from the plain header.
 */

#include "outlier_correct_stage.h"
#include "outlier_correct_kernels.cuh"
#include "cuda_check.h"
#include <cub/cub.cuh>
#include <algorithm>
#include <stdexcept>

namespace fz {

template <typename Reconstructor>
struct OutlierCorrectStage<Reconstructor>::Impl {
    size_t n = 0;
    float* d_coeff = nullptr;
    int* d_flag = nullptr; float* d_corrval = nullptr; int* d_rank = nullptr;
    void* d_tmp = nullptr; size_t tmpb = 0;

    ~Impl() { freeAll(); }

    void freeAll() {
        void** ptrs[] = { (void**)&d_coeff, (void**)&d_flag, (void**)&d_corrval, (void**)&d_rank, &d_tmp };
        for (void** p : ptrs) { if (*p) FZ_CUDA_CHECK_WARN(cudaFree(*p)); *p = nullptr; }
        n = 0; tmpb = 0;
    }

    void ensureShape(size_t n_) {
        if (n == n_ && n > 0) return;
        freeAll();
        n = n_;
        FZ_CUDA_CHECK(cudaMalloc(&d_coeff, n * 4));
        FZ_CUDA_CHECK(cudaMalloc(&d_flag, n * 4));
        FZ_CUDA_CHECK(cudaMalloc(&d_corrval, n * 4));
        FZ_CUDA_CHECK(cudaMalloc(&d_rank, n * 4));
        cub::DeviceScan::ExclusiveSum(d_tmp, tmpb, d_flag, d_rank, (int)n);
        FZ_CUDA_CHECK(cudaMalloc(&d_tmp, tmpb));
    }
};

template <typename Reconstructor>
OutlierCorrectStage<Reconstructor>::~OutlierCorrectStage() { delete impl_; }

template <typename Reconstructor>
void OutlierCorrectStage<Reconstructor>::execute(cudaStream_t stream, MemoryPool* /*pool*/,
                                                  const std::vector<void*>& inputs,
                                                  const std::vector<void*>& outputs,
                                                  const std::vector<size_t>& sizes)
{
    if (inputs.size() < 2 || outputs.size() < 2 || sizes.size() < 2)
        throw std::runtime_error(getName() + ": needs 2 inputs/outputs/sizes");
    if (dims_[0] == 0 || dims_[1] == 0)
        throw std::runtime_error(getName() + ": dimensions not set — call setDims() first");

    const int nx = (int)dims_[0], ny = (int)dims_[1], nz = (int)dims_[2];
    const size_t n = (size_t)nx * ny * (size_t)std::max(nz, 1);
    if (!impl_) impl_ = new Impl();
    Impl& I = *impl_;
    I.ensureShape(n);
    auto g = [&](int cnt) { return dim3((unsigned)((cnt + 255) / 256)); };

    // codes passthrough (output[1]) is identical in both directions: pure
    // copy of input[1] -- this stage never modifies the codes, it only
    // reads them (see the header's doc comment on why this port exists).
    if (outputs[1] != inputs[1])
        FZ_CUDA_CHECK(cudaMemcpyAsync(outputs[1], inputs[1], n * 4, cudaMemcpyDeviceToDevice, stream));
    actual_size1_ = n * 4;

    const float ebx2 = 2.0f * error_bound_;

    if (!is_inverse_) {
        // ── forward: compute the correction stream ──────────────────────
        const float*   d_field = static_cast<const float*>(inputs[0]);
        const int32_t* d_codes = static_cast<const int32_t*>(inputs[1]);

        oc_k_dequant_linear<<<g((int)n), 256, 0, stream>>>(d_codes, (int)n, ebx2, I.d_coeff);
        Reconstructor::applyInverseTransform(I.d_coeff, nx, ny, nz, stream);

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
        const uint8_t* corr    = static_cast<const uint8_t*>(inputs[0]);
        const int32_t* d_codes = static_cast<const int32_t*>(inputs[1]);
        float*          d_out  = static_cast<float*>(outputs[0]);

        oc_k_dequant_linear<<<g((int)n), 256, 0, stream>>>(d_codes, (int)n, ebx2, d_out);
        Reconstructor::applyInverseTransform(d_out, nx, ny, nz, stream);

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
