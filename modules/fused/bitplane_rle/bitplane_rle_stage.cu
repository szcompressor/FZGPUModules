// Fused bitplane-transpose + zero-byte RLE stage (FZ-GPU lossless encoder).
//
// Prior work: the encode/decode kernels are the FZ-GPU lossless codec
// (Boyuan Zhang et al., "FZ-GPU", HPDC '23), BSD-3-Clause. The stage wrapper,
// memory-pool integration, and padded-input handling are FZGPUModules code.
// See THIRD_PARTY.md.

#include "fused/bitplane_rle/bitplane_rle_stage.h"
#include "fused/bitplane_rle/bitplane_rle_kernels.h"
#include "mem/mempool.h"
#include "cuda_check.h"
#include "log.h"

#include <cuda_runtime.h>
#include <stdexcept>

namespace fz {

namespace bprle = bitplane_rle;

BitplaneRLEStage::~BitplaneRLEStage() = default;

std::vector<size_t> BitplaneRLEStage::estimateOutputSizes(
    const std::vector<size_t>& input_sizes) const {
    if (is_inverse_) {
        // Decode writes the full padded length; the trailing padding is zero.
        // cached_orig_len_ is restored from the FZM header (cold path) or set
        // by the forward pass (warm in-memory round-trip).
        const size_t data_len = (cached_orig_len_ > 0)
            ? cached_orig_len_
            : (input_sizes.empty() ? 0 : input_sizes[0] / sizeof(uint16_t));
        bprle::Config cfg = bprle::configure(data_len);
        return {cfg.pad_len * sizeof(uint16_t)};
    }
    // Forward: worst-case self-describing archive.
    const size_t in_bytes = input_sizes.empty() ? 0 : input_sizes[0];
    const size_t data_len = in_bytes / sizeof(uint16_t);
    return {bprle::maxArchiveBytes(data_len)};
}

void BitplaneRLEStage::execute(
    cudaStream_t stream,
    MemoryPool* pool,
    const std::vector<void*>& inputs,
    const std::vector<void*>& outputs,
    const std::vector<size_t>& sizes) {

    if (inputs.empty() || outputs.empty() || sizes.empty())
        throw std::runtime_error(
            "BitplaneRLEStage: inputs, outputs, and sizes must be non-empty");
    if (pool == nullptr)
        throw std::runtime_error("BitplaneRLEStage: requires a MemoryPool");

    if (is_inverse_) {
        // ── Decode ───────────────────────────────────────────────────────────
        // The archive is self-describing: read its 128-byte header to recover
        // the original symbol count and the sub-array byte offsets.
        bprle::ArchiveHeader header{};
        FZ_CUDA_CHECK(cudaMemcpy(&header, inputs[0], sizeof(header),
                                 cudaMemcpyDeviceToHost));
        const size_t data_len = header.original_len;
        cached_orig_len_ = data_len;
        bprle::Config cfg = bprle::configure(data_len);

        auto* archive = static_cast<uint8_t*>(inputs[0]);
        const auto* d_bitflag = reinterpret_cast<const uint32_t*>(
            archive + header.entry[bprle::BPRLE_HDR_BITFLAG]);
        const auto* d_start_pos = reinterpret_cast<const uint32_t*>(
            archive + header.entry[bprle::BPRLE_HDR_STARTPOS]);
        const uint8_t* d_bitstream =
            archive + header.entry[bprle::BPRLE_HDR_BITSTREAM];

        auto* d_out = static_cast<uint16_t*>(outputs[0]);
        bprle::launchDecode(d_bitstream, d_bitflag, d_start_pos, d_out, cfg,
                            stream);
        FZ_CUDA_CHECK(cudaStreamSynchronize(stream));

        // Report the pre-pad original size; the buffer holds pad_len symbols,
        // the tail beyond data_len being reconstructed padding zeros.
        actual_output_size_ = data_len * sizeof(uint16_t);
        return;
    }

    // ── Encode ─────────────────────────────────────────────────────────────
    const size_t in_bytes = sizes[0];
    const size_t data_len = in_bytes / sizeof(uint16_t);
    bprle::Config cfg = bprle::configure(data_len);

    // Lay out the three archive sub-arrays as views into the output buffer.
    auto* archive = static_cast<uint8_t*>(outputs[0]);
    auto* d_bitflag = reinterpret_cast<uint32_t*>(
        archive + bprle::kBitplaneRleHeaderBytes);
    auto* d_start_pos = reinterpret_cast<uint32_t*>(
        archive + bprle::kBitplaneRleHeaderBytes
                + sizeof(uint32_t) * cfg.chunk_size);
    auto* d_comp_out =
        archive + bprle::kBitplaneRleHeaderBytes
                + sizeof(uint32_t) * cfg.chunk_size
                + sizeof(uint32_t) * cfg.grid_x;

    // Transient scratch (freed before return).
    auto* d_offset_counter = static_cast<uint32_t*>(
        pool->allocate(sizeof(uint32_t), stream, "bprle_offset"));
    auto* d_comp_len = static_cast<uint32_t*>(
        pool->allocate(sizeof(uint32_t) * cfg.grid_x, stream, "bprle_comp_len"));
    FZ_CUDA_CHECK(cudaMemsetAsync(d_offset_counter, 0, sizeof(uint32_t), stream));

    // The kernel reads exactly cfg.data_bytes; if the input is shorter, copy it
    // into a zero-padded scratch buffer so the tail reads are well-defined.
    const uint16_t* d_in = nullptr;
    uint16_t* d_in_pad = nullptr;
    if (in_bytes == cfg.data_bytes) {
        d_in = static_cast<const uint16_t*>(inputs[0]);
    } else {
        d_in_pad = static_cast<uint16_t*>(
            pool->allocate(cfg.data_bytes, stream, "bprle_in_pad"));
        FZ_CUDA_CHECK(cudaMemsetAsync(d_in_pad, 0, cfg.data_bytes, stream));
        if (in_bytes > 0)
            FZ_CUDA_CHECK(cudaMemcpyAsync(d_in_pad, inputs[0], in_bytes,
                                          cudaMemcpyDeviceToDevice, stream));
        d_in = d_in_pad;
    }

    bprle::launchEncode(d_in, cfg, d_offset_counter, d_bitflag, d_start_pos,
                        d_comp_out, d_comp_len, stream);

    FZ_CUDA_CHECK(cudaStreamSynchronize(stream));
    uint32_t h_offset_sum = 0;
    FZ_CUDA_CHECK(cudaMemcpy(&h_offset_sum, d_offset_counter, sizeof(uint32_t),
                             cudaMemcpyDeviceToHost));

    // Build the self-describing header (cumulative byte offsets).
    bprle::ArchiveHeader header{};
    header.original_len = data_len;
    uint32_t nbyte[bprle::BPRLE_HDR_END];
    nbyte[bprle::BPRLE_HDR_HEADER]    = bprle::kBitplaneRleHeaderBytes;
    nbyte[bprle::BPRLE_HDR_BITFLAG]   = sizeof(uint32_t) * cfg.chunk_size;
    nbyte[bprle::BPRLE_HDR_STARTPOS]  = sizeof(uint32_t) * cfg.grid_x;
    nbyte[bprle::BPRLE_HDR_BITSTREAM] = h_offset_sum * sizeof(uint32_t);
    header.entry[0] = 0;
    for (int i = 1; i <= bprle::BPRLE_HDR_END; i++) header.entry[i] = nbyte[i - 1];
    for (int i = 1; i <= bprle::BPRLE_HDR_END; i++) header.entry[i] += header.entry[i - 1];

    FZ_CUDA_CHECK(cudaMemcpy(archive, &header, sizeof(header),
                             cudaMemcpyHostToDevice));

    actual_output_size_ = header.entry[bprle::BPRLE_HDR_END];
    cached_orig_len_ = data_len;

    pool->free(d_offset_counter, stream);
    pool->free(d_comp_len, stream);
    if (d_in_pad) pool->free(d_in_pad, stream);
}

} // namespace fz
