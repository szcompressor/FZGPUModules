/**
 * modules/structural/merge/merge_stage.cu
 *
 * MergeStage — concatenate N input buffers into one (forward), split one buffer
 * back into N segments (inverse).  Pure stream-ordered device-to-device copies;
 * no kernels, no host synchronisation, so it is graph-capturable in both
 * directions.  See merge_stage.h for the rationale (materialising the
 * concatenation so a downstream coder can compress several producer ports as a
 * single stream — e.g. cuSZ-Hi's LC merged-blob layout).
 */

#include "structural/merge/merge_stage.h"
#include "mem/mempool.h"
#include "cuda_check.h"

#include "backend/api.h"
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace fz {

void MergeStage::execute(
    cudaStream_t stream,
    MemoryPool* /*pool*/,
    const std::vector<void*>& inputs,
    const std::vector<void*>& outputs,
    const std::vector<size_t>& sizes)
{
    const size_t N = segment_names_.size();
    if (N == 0)
        throw std::runtime_error("MergeStage: setSegmentNames() must be called before execute()");

    if (!is_inverse_) {
        // ── Forward: concatenate N inputs → outputs[0] in connection order. ──
        if (inputs.size() != N || sizes.size() != N)
            throw std::runtime_error(
                "MergeStage: expected " + std::to_string(N) + " inputs, got "
                + std::to_string(inputs.size()));
        if (outputs.empty())
            throw std::runtime_error("MergeStage: missing output buffer");

        segment_sizes_.assign(N, 0);
        uint8_t* dst = static_cast<uint8_t*>(outputs[0]);
        size_t   off = 0;
        for (size_t i = 0; i < N; i++) {
            const size_t sz = sizes[i];
            segment_sizes_[i] = sz;
            if (sz > 0)
                FZ_CUDA_CHECK(cudaMemcpyAsync(dst + off, inputs[i], sz,
                                              cudaMemcpyDeviceToDevice, stream));
            off += sz;
        }
        merged_total_ = off;
        FZ_LOG(TRACE, "Merge: %zu segments -> %.1f KB blob", N, merged_total_ / 1024.0);

    } else {
        // ── Inverse: split inputs[0] → N outputs using the size table. ──
        if (inputs.empty())
            throw std::runtime_error("MergeStage: missing input buffer for split");
        if (outputs.size() != N)
            throw std::runtime_error(
                "MergeStage: expected " + std::to_string(N) + " outputs, got "
                + std::to_string(outputs.size()));
        if (segment_sizes_.size() != N)
            throw std::runtime_error(
                "MergeStage: segment size table missing/incomplete on inverse "
                "(needs deserializeHeader or a prior forward pass)");

        const uint8_t* src = static_cast<const uint8_t*>(inputs[0]);
        size_t off = 0;
        for (size_t i = 0; i < N; i++) {
            const size_t sz = segment_sizes_[i];
            if (sz > 0)
                FZ_CUDA_CHECK(cudaMemcpyAsync(outputs[i], src + off, sz,
                                              cudaMemcpyDeviceToDevice, stream));
            off += sz;
        }
        FZ_LOG(TRACE, "Merge: split %.1f KB blob -> %zu segments", off / 1024.0, N);
    }
}

} // namespace fz
