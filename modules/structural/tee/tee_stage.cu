/**
 * @file tee_stage.cu
 * @brief TeeStage::execute() — pure D2D duplication (forward) / selection (inverse).
 */

#include "structural/tee/tee_stage.h"
#include "cuda_check.h"
#include <stdexcept>

namespace fz {

void TeeStage::execute(cudaStream_t stream, MemoryPool* /*pool*/,
                       const std::vector<void*>& inputs,
                       const std::vector<void*>& outputs,
                       const std::vector<size_t>& sizes)
{
    if (!is_inverse_) {
        if (inputs.empty() || sizes.empty())
            throw std::runtime_error("TeeStage: forward needs a non-empty input");
        const size_t bytes = sizes[0];
        for (int i = 0; i < n_; ++i) {
            if (i >= (int)outputs.size()) throw std::runtime_error("TeeStage: missing output buffer");
            if (outputs[i] != inputs[0])
                FZ_CUDA_CHECK(cudaMemcpyAsync(outputs[i], inputs[0], bytes, cudaMemcpyDeviceToDevice, stream));
        }
        actual_output_size_ = bytes;
    } else {
        if (passthrough_idx_ < 0 || passthrough_idx_ >= (int)inputs.size())
            throw std::runtime_error("TeeStage: inverse passthrough_idx_ out of range");
        const void* src = inputs[passthrough_idx_];
        const size_t bytes = (passthrough_idx_ < (int)sizes.size()) ? sizes[passthrough_idx_] : 0;
        if (outputs.empty()) throw std::runtime_error("TeeStage: inverse needs an output buffer");
        if (outputs[0] != src)
            FZ_CUDA_CHECK(cudaMemcpyAsync(outputs[0], src, bytes, cudaMemcpyDeviceToDevice, stream));
        actual_output_size_ = bytes;
    }
}

} // namespace fz
