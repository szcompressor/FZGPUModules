// Adapted from PHF reference (origin/v1.1.0_dev:modules/codec/huffman/)
// Changes:
//   - PHF buffers (phf::Buf<T>) allocated from MemoryPool
//   - Buf reallocated only on capacity growth or bklen change (not on every inlen change).
//   - Symbol range validation: freq_sum check after D2H catches out-of-[0,bklen) symbols.
//   - onFinalize() pre-allocates buf_ from pool at finalize time for PREALLOCATE mode;
//     execute() falls back to lazy allocation from its pool parameter if not pre-allocated.

#include "coders/huffman/huffman_stage.h"
#include "coders/huffman/phf/hf_buf.h"
#include "util/histogram/histogram.h"
#include "mem/mempool.h"
#include "cuda_check.h"

#include <cuda_runtime.h>
#include <cstddef>
#include <stdexcept>

namespace fz {

// ── Constructor / destructor ───────────────────────────────────────────────────
// Defined here (not defaulted in the header) so that the unique_ptr<phf::Buf<T>>
// destructor can see the complete phf::Buf<T> type.

template <typename T>
HuffmanStage<T>::HuffmanStage() = default;

template <typename T>
HuffmanStage<T>::~HuffmanStage() = default;

// ── initBuf ───────────────────────────────────────────────────────────────────
// Allocates phf::Buf<T> from pool and runs the histogram optimizer.
// Destroys the existing buf_ first (returning its pool allocations) if present.
// Must live in a .cu file: the optimizer calls cudaFuncSetAttribute with a
// __global__ kernel pointer, which is not callable from CXX translation units.

template <typename T>
void HuffmanStage<T>::initBuf(size_t inlen, MemoryPool* pool)
{
    buf_.reset();  // destroy old Buf<T> first — returns allocations to pool
    bool use_hfr = (encode_mode_ == HuffmanEncodeMode::Fine);
    buf_               = std::make_unique<phf::Buf<T>>(inlen, bklen_, pool, -1, use_hfr);
    pool_              = pool;
    cap_inlen_         = inlen;
    last_bklen_        = bklen_;
    last_encode_mode_  = encode_mode_;
    fz::module::GPU_histogram_generic_optimizer_on_initialization<T>(
        inlen, static_cast<uint16_t>(bklen_),
        hist_grid_dim_, hist_block_dim_, hist_shmem_use_, hist_r_per_block_);
}

// ── onFinalize ────────────────────────────────────────────────────────────────

template <typename T>
void HuffmanStage<T>::onFinalize(size_t estimated_inlen, MemoryPool* pool)
{
    if (estimated_inlen == 0) return;  // no hint; defer to first execute()
    const size_t inlen = estimated_inlen / sizeof(T);
    if (inlen == 0) return;
    initBuf(inlen, pool);
}

// ── Footprint estimates ───────────────────────────────────────────────────────

template <typename T>
size_t HuffmanStage<T>::estimateDeviceFootprintBytes(size_t inlen) const
{
    if (inlen == 0) return 0;
    const size_t n = inlen / sizeof(T);
    const size_t pardeg = (n - 1) / 4096 + 1;  // approximate sublen=4096 default
    using M = PHF_METADATA;
    using H4 = uint32_t;
    size_t base = sizeof(H4) * n                          // d_scratch4
                + sizeof(H4) * bklen_                     // d_bk4
                + bklen_ * 4 * sizeof(T)                  // d_revbk4 (approx)
                + sizeof(H4) * (n / 2)                    // d_bitstream4
                + sizeof(M) * pardeg * 3                  // d_par_nbit/ncell/entry
                + sizeof(uint32_t) * bklen_               // d_freq
                + sizeof(T) * (100 + n / 10 + 1)          // d_brval
                + sizeof(uint32_t) * (100 + n / 10 + 1)   // d_bridx
                + sizeof(uint32_t);                        // d_brnum
    // Fine mode adds CUB temp storage (~few KB) and two uint64_t device scalars
    if (encode_mode_ == HuffmanEncodeMode::Fine)
        base += 65536 + 2 * sizeof(uint64_t);  // conservative CUB temp upper bound
    return base;
}

template <typename T>
size_t HuffmanStage<T>::estimatePinnedFootprintBytes(size_t inlen) const
{
    if (inlen == 0) return 0;
    const size_t n = inlen / sizeof(T);
    const size_t pardeg = (n - 1) / 4096 + 1;
    using M = PHF_METADATA;
    using H4 = uint32_t;
    size_t base = sizeof(H4) * n                          // h_scratch4
                + sizeof(H4) * bklen_                     // h_bk4
                + bklen_ * 4 * sizeof(T)                  // h_revbk4 (approx)
                + sizeof(H4) * (n / 2)                    // h_bitstream4
                + sizeof(M) * pardeg * 3                  // h_par_nbit/ncell/entry
                + sizeof(uint32_t) * bklen_               // h_freq
                + sizeof(uint32_t);                        // h_brnum
    if (encode_mode_ == HuffmanEncodeMode::Fine)
        base += 2 * sizeof(uint64_t);  // h_total_nbit, h_total_ncell
    return base;
}

// ── execute ───────────────────────────────────────────────────────────────────

template <typename T>
void HuffmanStage<T>::execute(
    cudaStream_t stream,
    MemoryPool*  pool,
    const std::vector<void*>&  inputs,
    const std::vector<void*>&  outputs,
    const std::vector<size_t>& sizes)
{
    if (inputs.empty() || outputs.empty() || sizes.empty())
        throw std::runtime_error(
            "HuffmanStage: inputs, outputs, and sizes must be non-empty");

    if (sizes[0] == 0) { actual_output_size_ = 0; return; }

    if (!is_inverse_) {
        // ── Forward: T[] → PHF bitstream ─────────────────────────────────────
        T*     d_input = static_cast<T*>(inputs[0]);
        size_t inlen   = sizes[0] / sizeof(T);

        // Reallocate when capacity exceeded, bklen changed, or encode mode changed.
        // Use pool from execute() parameter; on first call, this also sets pool_.
        if (!buf_ || inlen > cap_inlen_ || bklen_ != last_bklen_ || encode_mode_ != last_encode_mode_)
            initBuf(inlen, pool);

        // Zero frequency array (histogram kernel uses atomicAdd into d_freq)
        FZ_CUDA_CHECK(cudaMemsetAsync(
            buf_->d_freq, 0, bklen_ * sizeof(uint32_t), stream));

        // GPU histogram → d_freq
        fz::module::GPU_histogram_generic<T>(
            d_input, inlen,
            buf_->d_freq, static_cast<uint16_t>(bklen_),
            hist_grid_dim_, hist_block_dim_, hist_shmem_use_, hist_r_per_block_,
            stream);

        // Sync stream, then D2H copy of frequency table to host
        FZ_CUDA_CHECK(cudaStreamSynchronize(stream));
        FZ_CUDA_CHECK(cudaMemcpy(
            buf_->h_freq, buf_->d_freq,
            bklen_ * sizeof(uint32_t), cudaMemcpyDeviceToHost));

        // Symbol range validation: out-of-range symbols are skipped by the histogram
        // kernel (not counted), so freq_sum < inlen means bklen is too small.
        {
            uint64_t freq_sum = 0;
            const uint32_t* hf = buf_->h_freq;
            for (uint32_t i = 0; i < bklen_; ++i) freq_sum += hf[i];
            if (freq_sum != static_cast<uint64_t>(inlen))
                throw std::runtime_error(
                    "HuffmanStage: " +
                    std::to_string(inlen - static_cast<size_t>(freq_sum)) +
                    " out-of-range symbol(s) detected; all symbols must be in [0, " +
                    std::to_string(bklen_) + "). "
                    "Increase bklen or use setZigzagCodes(true) with LorenzoQuantStage.");
        }

        // Build Huffman codebook from host histogram (H2D copies codebook + revbook)
        phf::high_level<T>::build_book(
            buf_.get(), buf_->h_freq, static_cast<uint16_t>(bklen_), stream);

        // Encode: GPU_coarse_encode → stream sync → memcpy_merge into buf_->d_encoded
        uint8_t* d_out  = nullptr;
        size_t   outlen = 0;
        phf::high_level<T>::encode(
            buf_.get(), d_input, inlen, &d_out, &outlen, header_, stream);

        // Copy PHF output (buf_->d_encoded = buf_->d_scratch4) to pipeline output buffer.
        FZ_CUDA_CHECK(cudaMemcpyAsync(
            outputs[0], d_out, outlen, cudaMemcpyDeviceToDevice, stream));

        actual_output_size_ = outlen;
        original_len_       = inlen;

    } else {
        // ── Inverse: PHF bitstream → T[] ─────────────────────────────────────
        if (original_len_ == 0)
            throw std::runtime_error(
                "HuffmanStage: inverse called with original_len_=0; "
                "call deserializeHeader() before decompressing");

        auto*  d_encoded = static_cast<uint8_t*>(inputs[0]);
        size_t inlen     = original_len_;

        if (!buf_ || inlen > cap_inlen_ || bklen_ != last_bklen_) initBuf(inlen, pool);

        // Read the phf_header embedded at the start of the encoded buffer (D2H)
        phf_header hdr {};
        FZ_CUDA_CHECK(cudaMemcpy(
            &hdr, d_encoded, sizeof(hdr), cudaMemcpyDeviceToHost));

        phf::high_level<T>::decode(
            buf_.get(), hdr, d_encoded, static_cast<T*>(outputs[0]), stream);

        actual_output_size_ = inlen * sizeof(T);
    }
}

template class HuffmanStage<uint8_t>;
template class HuffmanStage<uint16_t>;
template class HuffmanStage<uint32_t>;

} // namespace fz
