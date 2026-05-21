// Adapted from dietGPU (Meta Platforms, Inc.), MIT License — see THIRD_PARTY.md.
//
// Sync points (both make this stage graph-incompatible):
//   Forward: cudaMemcpyAsync D2H + cudaStreamSynchronize after coalesce
//            (reads ANSCoalescedHeader to determine actual compressed size).
//   Inverse: cudaMemcpyAsync D2H + cudaStreamSynchronize before decode
//            (reads ANSCoalescedHeader to determine prob_bits and uncompressed size).

#include "coders/ans/ans_stage.h"
#include "ans/GpuANSCodec.h"
#include "ans/GpuANSStatistics.h"
#include "ans/GpuANSEncode.h"
#include "ans/GpuANSDecode.h"
#include "ans/BatchPrefixSum.h"
#include "util/histogram/histogram.h"
#include "mem/mempool.h"
#include "cuda_check.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

namespace fz {

// ── initScratch ───────────────────────────────────────────────────────────────
// Allocates (or reallocates) all 7 persistent scratch buffers sized for inlen
// input bytes.  Computes histogram launch params for the uint8_t alphabet.

void ANSStage::initScratch(size_t inlen, MemoryPool* pool)
{
    const auto max_blocks = static_cast<uint32_t>(
        fz::ans::divUp(static_cast<uint64_t>(inlen),
                       static_cast<uint64_t>(fz::ans::kDefaultBlockSize)));

    // Release existing allocations before resizing.
    if (d_temp_histogram_)    pool->freePersistentDevice(d_temp_histogram_);
    if (d_table_)             pool->freePersistentDevice(d_table_);
    if (d_compressed_blocks_) pool->freePersistentDevice(d_compressed_blocks_);
    if (d_compressed_words_)  pool->freePersistentDevice(d_compressed_words_);
    if (d_comp_words_prefix_) pool->freePersistentDevice(d_comp_words_prefix_);
    if (d_temp_prefix_sum_)   pool->freePersistentDevice(d_temp_prefix_sum_);
    if (d_decode_table_)      pool->freePersistentDevice(d_decode_table_);

    d_temp_histogram_ = static_cast<uint32_t*>(pool->allocatePersistentDevice(
        fz::ans::kNumSymbols * sizeof(uint32_t), "ans.histogram"));

    d_table_ = pool->allocatePersistentDevice(
        fz::ans::kNumSymbols * sizeof(uint4), "ans.encode_table");

    d_compressed_blocks_ = static_cast<uint8_t*>(pool->allocatePersistentDevice(
        static_cast<size_t>(max_blocks) * kUncoalescedStride, "ans.comp_blocks"));

    d_compressed_words_ = static_cast<uint32_t*>(pool->allocatePersistentDevice(
        static_cast<size_t>(max_blocks) * sizeof(uint32_t), "ans.comp_words"));

    d_comp_words_prefix_ = static_cast<uint32_t*>(pool->allocatePersistentDevice(
        static_cast<size_t>(max_blocks) * sizeof(uint32_t), "ans.comp_words_prefix"));

    const size_t temp_elems = fz::ans::getBatchExclusivePrefixSumTempSize(max_blocks);
    if (temp_elems > 0) {
        d_temp_prefix_sum_ = pool->allocatePersistentDevice(
            temp_elems * sizeof(uint32_t), "ans.prefix_sum_temp");
    } else {
        d_temp_prefix_sum_ = nullptr;
    }

    d_decode_table_ = static_cast<uint32_t*>(pool->allocatePersistentDevice(
        (size_t(1) << prob_bits_) * sizeof(uint32_t), "ans.decode_table"));

    fz::module::GPU_histogram_generic_optimizer_on_initialization<uint8_t>(
        inlen, static_cast<uint16_t>(fz::ans::kNumSymbols),
        hist_grid_dim_, hist_block_dim_, hist_shmem_use_, hist_r_per_block_);

    cap_bytes_ = inlen;
}

// ── onFinalize ────────────────────────────────────────────────────────────────

void ANSStage::onFinalize(size_t estimated_inlen, MemoryPool* pool)
{
    if (estimated_inlen == 0) return;
    initScratch(estimated_inlen, pool);
}

// ── estimateDeviceFootprintBytes ──────────────────────────────────────────────

size_t ANSStage::estimateDeviceFootprintBytes(size_t inlen) const
{
    if (inlen == 0) return 0;
    const auto max_blocks = static_cast<uint32_t>(
        (static_cast<uint64_t>(inlen) + fz::ans::kDefaultBlockSize - 1)
        / fz::ans::kDefaultBlockSize);
    const size_t temp_bytes =
        fz::ans::getBatchExclusivePrefixSumTempSize(max_blocks) * sizeof(uint32_t);

    return fz::ans::kNumSymbols  * sizeof(uint32_t)               // d_temp_histogram_
         + fz::ans::kNumSymbols  * sizeof(uint4)                   // d_table_
         + static_cast<size_t>(max_blocks) * kUncoalescedStride    // d_compressed_blocks_
         + static_cast<size_t>(max_blocks) * sizeof(uint32_t)      // d_compressed_words_
         + static_cast<size_t>(max_blocks) * sizeof(uint32_t)      // d_comp_words_prefix_
         + temp_bytes                                               // d_temp_prefix_sum_
         + (size_t(1) << prob_bits_) * sizeof(uint32_t);           // d_decode_table_
}

// ── estimateScratchBytes ──────────────────────────────────────────────────────

size_t ANSStage::estimateScratchBytes(const std::vector<size_t>& input_sizes) const
{
    if (input_sizes.empty()) return 0;
    return estimateDeviceFootprintBytes(input_sizes[0]);
}

// ── execute ───────────────────────────────────────────────────────────────────

void ANSStage::execute(
    cudaStream_t stream,
    MemoryPool*  pool,
    const std::vector<void*>&  inputs,
    const std::vector<void*>&  outputs,
    const std::vector<size_t>& sizes)
{
    if (inputs.empty() || outputs.empty() || sizes.empty())
        throw std::runtime_error(
            "ANSStage: inputs, outputs, and sizes must be non-empty");

    const size_t byte_size = sizes[0];
    if (byte_size == 0) {
        actual_output_size_ = 0;
        return;
    }

    auto* in  = static_cast<uint8_t*>(inputs[0]);
    auto* out = static_cast<uint8_t*>(outputs[0]);

    if (!is_inverse_) {
        // ── Forward (compress) ────────────────────────────────────────────────
        if (prob_bits_ != static_cast<uint8_t>(fz::ans::kANSDefaultProbBits))
            throw std::runtime_error(
                "ANSStage: only prob_bits=10 is supported in this build");

        if (byte_size > cap_bytes_)
            initScratch(byte_size, pool);

        const auto inlen      = static_cast<uint32_t>(byte_size);
        const auto max_blocks = static_cast<uint32_t>(
            (static_cast<uint64_t>(inlen) + fz::ans::kDefaultBlockSize - 1)
            / fz::ans::kDefaultBlockSize);

        // Step 1: Build uint8_t histogram (256 bins).
        FZ_CUDA_CHECK(cudaMemsetAsync(
            d_temp_histogram_, 0,
            fz::ans::kNumSymbols * sizeof(uint32_t), stream));
        fz::module::GPU_histogram_generic<uint8_t>(
            in, byte_size, d_temp_histogram_,
            static_cast<uint16_t>(fz::ans::kNumSymbols),
            hist_grid_dim_, hist_block_dim_, hist_shmem_use_, hist_r_per_block_,
            stream);

        // Step 2: Normalize histogram → ANS encode table (pdf/cdf/mul/shift per symbol).
        fz::ans::ansCalcWeights(
            prob_bits_, inlen,
            d_temp_histogram_, static_cast<uint4*>(d_table_), stream);

        // Step 3: Encode each 4096-byte block into uncoalesced per-block buffers.
        // ansEncodeBatch requires blockDim >= kNumSymbols (256) to populate smemLookup.
        // 256 threads / 32 = 8 warps per CUDA block; one warp handles one data block.
        const uint32_t enc_grid = (max_blocks + 7u) / 8u;
        fz::ans::ansEncodeBatch<
            fz::ans::kANSDefaultProbBits,
            fz::ans::kDefaultBlockSize>
            <<<enc_grid, 256, 0, stream>>>(
                in, static_cast<int>(inlen), max_blocks,
                static_cast<uint32_t>(kUncoalescedStride),
                d_compressed_blocks_, d_compressed_words_,
                static_cast<const uint4*>(d_table_));
        FZ_CUDA_CHECK(cudaGetLastError());

        // Step 4: Prefix-sum per-block compressed word counts (aligned to kBlockAlignment).
        fz::ans::batchExclusivePrefixSum<uint32_t>(
            d_compressed_words_, d_comp_words_prefix_,
            d_temp_prefix_sum_, max_blocks,
            fz::ans::Align<fz::ans::ANSEncodedT, fz::ans::kBlockAlignment>{},
            stream);

        // Step 5: Coalesce per-block buffers into the output bitstream.
        // One CUDA block per data block; CUDA block 0 also writes the coalesced header.
        fz::ans::ansEncodeCoalesceBatch<64>
            <<<max_blocks, 64, 0, stream>>>(
                d_compressed_blocks_, static_cast<int>(inlen),
                max_blocks, static_cast<uint32_t>(kUncoalescedStride),
                d_compressed_words_, d_comp_words_prefix_,
                static_cast<const uint4*>(d_table_),
                static_cast<uint32_t>(prob_bits_),
                out, nullptr);
        FZ_CUDA_CHECK(cudaGetLastError());

        // Step 6: D2H readback of ANSCoalescedHeader to get actual compressed size.
        // Stream sync is required; stage is not graph-compatible.
        FZ_CUDA_CHECK(cudaMemcpyAsync(
            last_header_bytes_, out, 32, cudaMemcpyDeviceToHost, stream));
        FZ_CUDA_CHECK(cudaStreamSynchronize(stream));

        const auto* h =
            reinterpret_cast<const fz::ans::ANSCoalescedHeader*>(last_header_bytes_);
        actual_output_size_ = h->getTotalCompressedSize();
        original_bytes_     = byte_size;

    } else {
        // ── Inverse (decompress) ──────────────────────────────────────────────

        // Step 1: D2H header peek — reads prob_bits and total uncompressed word count.
        FZ_CUDA_CHECK(cudaMemcpyAsync(
            last_header_bytes_, in, 32, cudaMemcpyDeviceToHost, stream));
        FZ_CUDA_CHECK(cudaStreamSynchronize(stream));

        const auto* h =
            reinterpret_cast<const fz::ans::ANSCoalescedHeader*>(last_header_bytes_);
        h->checkMagicAndVersion();

        const uint32_t hdr_prob_bits = h->getProbBits();
        const uint32_t uncomp_words  = h->getTotalUncompressedWords();
        const uint32_t num_blocks    = h->getNumBlocks();

        if (hdr_prob_bits != static_cast<uint32_t>(fz::ans::kANSDefaultProbBits))
            throw std::runtime_error(
                "ANSStage: only prob_bits=10 is supported; file has prob_bits="
                + std::to_string(hdr_prob_bits));

        if (uncomp_words == 0) {
            actual_output_size_ = 0;
            return;
        }

        // Ensure decode scratch is allocated (sized by uncompressed byte count).
        if (uncomp_words > cap_bytes_)
            initScratch(static_cast<size_t>(uncomp_words), pool);

        // Step 2: Build decode lookup table from symbol probs in the bitstream header.
        fz::ans::ansDecodeTable<256>
            <<<1, 256, 0, stream>>>(in, hdr_prob_bits, d_decode_table_);
        FZ_CUDA_CHECK(cudaGetLastError());

        // Step 3: Decode with an occupancy-based grid so all SMs stay busy.
        // ansDecodeKernel loops over data blocks (block += warpsPerGrid), so a
        // grid smaller than numBlocks is valid — it just increases work per warp.
        int device = 0;
        FZ_CUDA_CHECK(cudaGetDevice(&device));
        int num_sms = 0;
        FZ_CUDA_CHECK(cudaDeviceGetAttribute(
            &num_sms, cudaDevAttrMultiProcessorCount, device));

        int max_blocks_per_sm = 0;
        FZ_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_blocks_per_sm,
            fz::ans::ansDecodeKernel<
                128,
                fz::ans::kANSDefaultProbBits,
                fz::ans::kDefaultBlockSize>,
            128, 0));

        // Warps needed to give each data block its own warp (upper bound).
        // 128 threads / 32 = 4 warps per CUDA block.
        const int warps_needed = static_cast<int>((num_blocks + 3u) / 4u);
        const int dec_grid = std::max(1, std::min(
            num_sms * max_blocks_per_sm, warps_needed));

        fz::ans::ansDecodeKernel<
            128,
            fz::ans::kANSDefaultProbBits,
            fz::ans::kDefaultBlockSize>
            <<<dec_grid, 128, 0, stream>>>(in, d_decode_table_, out);
        FZ_CUDA_CHECK(cudaGetLastError());

        // ANSDecodedT = uint8_t, so uncompressed word count equals byte count.
        actual_output_size_ = uncomp_words;
    }
}

} // namespace fz
