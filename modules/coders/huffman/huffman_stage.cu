// Adapted from PHF reference (origin/v1.1.0_dev:modules/codec/huffman/)
// Changes:
//   - PHF buffers (phf::Buf<T>) allocated from MemoryPool
//   - Buf reallocated only on capacity growth or bklen change (not on every inlen change).
//   - Symbol range validation: freq_sum check after D2H catches out-of-[0,bklen) symbols.
//   - onFinalize() pre-allocates buf_ from pool at finalize time for PREALLOCATE mode;
//     execute() falls back to lazy allocation from its pool parameter if not pre-allocated.
//   - HuffmanBookSource::Fixed: codebook built once from a caller-supplied or
//     model-synthesized frequency table, skipping the per-call histogram + host tree build.

#include "coders/huffman/huffman_stage.h"
#include "coders/huffman/phf/hf_buf.h"
#include "util/histogram/histogram.h"
#include "mem/mempool.h"
#include "cuda_check.h"
#include "log.h"

#include "backend/api.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace fz {

// ── Device-side symbol-range check ────────────────────────────────────────────
//
// PerBlock catches out-of-range symbols for free: the histogram kernel skips any
// symbol >= bklen, so sum(h_freq) != inlen means some were dropped.  Once Adaptive
// or Fixed pins a codebook nothing histograms, and the encode kernel indexes d_bk4
// with the raw symbol — an out-of-range value reads past the codebook and produces a
// stream that cannot be decoded, with no diagnostic at all.
//
// This restores the guarantee.  The kernel records the largest offending symbol
// (biased by +1, so 0 means "none").  The verdict must be read *before* encode
// launches — a post-encode check is too late, because the out-of-range index has
// already faulted — so it costs a stream sync.  That sync is still far cheaper than
// the PerBlock path it replaces (a 4-byte D2H instead of bklen words, and no host
// tree build), but it is not free; see setValidateSymbolRange().
template <typename T>
__global__ void huffmanSymbolRangeKernel(
    const T* __restrict__ in, size_t n, uint32_t bklen, uint32_t* __restrict__ out)
{
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    uint32_t local = 0;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += stride) {
        const uint32_t s = static_cast<uint32_t>(in[i]);
        if (s >= bklen && s + 1u > local) local = s + 1u;
    }
    if (local != 0) atomicMax(out, local);
}

// ── Constructor / destructor ───────────────────────────────────────────────────
// Defined here (not defaulted in the header) so that the unique_ptr<phf::Buf<T>>
// destructor can see the complete phf::Buf<T> type.

template <typename T>
__host__ HuffmanStage<T>::HuffmanStage() = default;

template <typename T>
__host__ HuffmanStage<T>::~HuffmanStage() = default;

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
    // Fresh d_bk4 / d_revbk4 allocations: whatever fixed book was resident is gone.
    fixed_book_resident_ = false;
    pool_              = pool;
    cap_inlen_         = inlen;
    last_bklen_        = bklen_;
    last_encode_mode_  = encode_mode_;
    fz::module::GPU_histogram_generic_optimizer_on_initialization<T>(
        inlen, static_cast<uint16_t>(bklen_),
        hist_grid_dim_, hist_block_dim_, hist_shmem_use_, hist_r_per_block_);
}

// ── Fixed codebook ────────────────────────────────────────────────────────────

template <typename T>
void HuffmanStage<T>::setFixedBookFromFreq(const uint32_t* h_freq, uint32_t n)
{
    if (h_freq == nullptr || n == 0)
        throw std::invalid_argument(
            "HuffmanStage::setFixedBookFromFreq: frequency table must be non-empty");

    setBklen(n);  // applies the odd→even rounding the decode kernel's alignment needs

    fixed_freq_.assign(bklen_, 1u);  // padding slot (if any) keeps frequency 1
    for (uint32_t i = 0; i < n; ++i) {
        if (h_freq[i] == 0)
            throw std::invalid_argument(
                "HuffmanStage::setFixedBookFromFreq: frequency[" + std::to_string(i) +
                "] is 0; a zero-frequency symbol gets no code, so encoding it would "
                "silently corrupt the stream. Add 1 to every bin, or shrink bklen.");
        fixed_freq_[i] = h_freq[i];
    }

    book_source_         = HuffmanBookSource::Fixed;
    fixed_book_resident_ = false;
    has_book_spec_       = false;  // a raw table is not describable by a spec
}

template <typename T>
void HuffmanStage<T>::setFixedBookFromModel(const HuffmanBookSpec& spec)
{
    if (bklen_ == 0)
        throw std::invalid_argument(
            "HuffmanStage::setFixedBookFromModel: bklen is 0; call setBklen() first");
    if (spec.scale <= 0.0)
        throw std::invalid_argument(
            "HuffmanStage::setFixedBookFromModel: scale must be > 0");
    if (spec.shape <= 0.0)
        throw std::invalid_argument(
            "HuffmanStage::setFixedBookFromModel: shape must be > 0");

    const double center = (spec.center < 0.0) ? 0.5 * bklen_ : spec.center;

    // Peak frequency.  Deliberately modest: canonical Huffman depth grows with the
    // frequency dynamic range, and codes longer than HuffmanWord<4>::FIELD_CODE (27
    // bits) cannot be represented.  buildFixedBook() checks the built book anyway.
    constexpr double kPeak = 4096.0;

    std::vector<uint32_t> freq(bklen_, 1u);
    for (uint32_t i = 0; i < bklen_; ++i) {
        const double d = static_cast<double>(i) - center;
        double w;
        switch (spec.model) {
            case HuffmanBookModel::Gaussian:
                w = std::exp(-0.5 * (d / spec.scale) * (d / spec.scale));
                break;
            case HuffmanBookModel::Laplace:
                w = std::exp(-std::fabs(d) / spec.scale);
                break;
            case HuffmanBookModel::GeneralizedNormal:
                w = std::exp(-std::pow(std::fabs(d) / spec.scale, spec.shape));
                break;
            case HuffmanBookModel::Uniform:
            default:
                w = 1.0;
                break;
        }
        const double scaled = kPeak * w;
        // Floor of 1: every symbol stays codable, which is the whole point of using
        // a model rather than a histogram of a sample.
        freq[i] = (scaled < 1.0) ? 1u : static_cast<uint32_t>(scaled);
    }

    fixed_freq_          = std::move(freq);
    book_source_         = HuffmanBookSource::Fixed;
    fixed_book_resident_ = false;
    book_spec_           = spec;
    has_book_spec_       = true;
}

// Returns the index of the first symbol whose built code is unusable, or -1 if the
// book is sound.
//
// phf_CPU_build_canonized_codebook_v2 does not fail when a symbol needs a code longer
// than HuffmanWord<4>::FIELD_CODE (27 bits): it clamps bitcount to OUTLIER_CUTOFF,
// sets prefix_code to 0, and prints a line to stdout (hf_bk.cc).  A zero prefix code
// is not a valid code, so the encode kernel would emit a stream that cannot be
// decoded.  A bitcount in [28, 31] is always invalid — 27 is the widest the format
// represents — so the clamp marker and the 0xffffffff "no code built" sentinel are
// both caught by the same test.
//
// Only symbols that actually occur are checked: with a real histogram, an absent
// symbol legitimately gets no code, and encoding never indexes it.
template <typename T>
int HuffmanStage<T>::findUnusableCode(const uint32_t* freq) const
{
    using PW4 = HuffmanWord<4>;
    for (uint32_t i = 0; i < bklen_; ++i) {
        if (freq[i] == 0) continue;
        const uint32_t bits = buf_->h_bk4[i] >> PW4::FIELD_CODE;
        if (bits >= static_cast<uint32_t>(PW4::OUTLIER_CUTOFF)) return static_cast<int>(i);
    }
    return -1;
}

template <typename T>
void HuffmanStage<T>::buildFixedBook(fz::stream_t stream)
{
    if (fixed_freq_.empty())
        throw std::runtime_error(
            "HuffmanStage: book source is Fixed but no codebook has been supplied; "
            "call setFixedBookFromFreq() or setFixedBookFromModel() first");
    if (fixed_freq_.size() != bklen_)
        throw std::runtime_error(
            "HuffmanStage: fixed codebook has " + std::to_string(fixed_freq_.size()) +
            " entries but bklen is " + std::to_string(bklen_) +
            "; setBklen() was called after the codebook was set");

    phf::high_level<T>::build_book(
        buf_.get(), fixed_freq_.data(), static_cast<uint16_t>(bklen_), stream);

    const int bad = findUnusableCode(fixed_freq_.data());
    if (bad >= 0)
        throw std::runtime_error(
            "HuffmanStage: fixed codebook symbol " + std::to_string(bad) +
            " needs a code longer than " +
            std::to_string(HuffmanWord<4>::FIELD_CODE) +
            " bits, which the 32-bit codeword format cannot hold. Reduce the "
            "frequency dynamic range (a wider model scale, or fewer symbols).");

    fixed_book_resident_ = true;
}

// ── Adaptive codebook ─────────────────────────────────────────────────────────
//
// Builds one reusable codebook from a histogram of the first block, rather than from
// a guessed analytic shape.  This is the Fixed path's throughput with a book that
// actually fits the data: exactly one histogram for the lifetime of the stage
// instead of one per call.
//
// Frequencies are floored at max_freq >> shift before building.  The floor is what
// makes the book safe to reuse — every symbol in [0, bklen) gets a code, including
// ones the sampled block never contained — and it bounds Huffman depth.  If the book
// still does not fit the 27-bit code field, the shift is halved and the build
// retried; shift 0 is a uniform book, which always fits, so this terminates.

template <typename T>
void HuffmanStage<T>::buildAdaptiveBook(const uint32_t* h_hist, fz::stream_t stream)
{
    uint32_t max_freq = 0;
    uint64_t total    = 0;
    uint32_t distinct = 0;
    for (uint32_t i = 0; i < bklen_; ++i) {
        if (h_hist[i] > max_freq) max_freq = h_hist[i];
        if (h_hist[i] != 0) ++distinct;
        total += h_hist[i];
    }

    // Degeneracy guard.  A block that is constant (or all but constant) teaches the
    // codebook nothing except "one symbol", and the frequency floor then turns that
    // into a book where the dominant symbol costs a full bit.  Pinning it would
    // freeze that book for the rest of the run: measured at 42% mean ratio loss over
    // CESM CLOUD when the fit landed on a constant level, against 4.7% when it did
    // not.  Constant slabs are common in real data (CESM CLOUD levels 0-2, Hurricane
    // CLOUDf48 slabs 18-19), so this is a case worth refusing rather than absorbing.
    //
    // Build the book anyway — this call still has to encode — but leave it unpinned
    // so the next call histograms afresh and gets another chance at a real sample.
    // If every block is degenerate this degrades to PerBlock, which is correct.
    const bool degenerate =
        (distinct < 2) ||
        (total > 0 && static_cast<double>(max_freq) >= 0.999 * static_cast<double>(total));

    fixed_freq_.resize(bklen_);

    uint8_t shift = adaptive_floor_shift_;
    for (;;) {
        const uint32_t floor_v = std::max(1u, max_freq >> shift);
        for (uint32_t i = 0; i < bklen_; ++i)
            fixed_freq_[i] = std::max(h_hist[i], floor_v);

        phf::high_level<T>::build_book(
            buf_.get(), fixed_freq_.data(), static_cast<uint16_t>(bklen_), stream);

        if (findUnusableCode(fixed_freq_.data()) < 0) break;
        if (shift == 0)
            // Unreachable: shift 0 floors every symbol at max_freq, so the book is
            // uniform and every code is ceil(log2(bklen)) bits wide.
            throw std::runtime_error(
                "HuffmanStage: no codebook fits the 27-bit code field even at a "
                "uniform distribution; bklen (" + std::to_string(bklen_) +
                ") is implausibly large.");
        shift /= 2;
    }

    adaptive_shift_used_  = shift;
    has_book_spec_        = false;  // sampled, not model-described
    fixed_book_resident_  = !degenerate;
    just_fitted_          = !degenerate;
    if (degenerate)
        FZ_LOG(DEBUG,
               "HuffmanStage: sample block is degenerate (%u distinct symbol(s), "
               "%.4f%% on the most common); using its codebook for this call but "
               "not pinning it.",
               distinct,
               total ? 100.0 * static_cast<double>(max_freq) / static_cast<double>(total) : 0.0);
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

        // The histogram (and with it the D2H, the host sync, and the out-of-range
        // symbol check) runs every call under PerBlock, once under Adaptive, and
        // never under Fixed.
        const bool need_histogram =
            (book_source_ == HuffmanBookSource::PerBlock) ||
            (book_source_ == HuffmanBookSource::Adaptive && !fixed_book_resident_);

        if (!need_histogram) {
            // Reusing a book: no histogram kernel, no frequency D2H, no host stream
            // sync, no tree build.
            //
            // Note this also skips the out-of-range symbol check below, which was a
            // side effect of histogramming.  Symbols >= bklen index past d_bk4 in the
            // encode kernel, so the caller owns that invariant once a book is pinned.
            if (book_source_ == HuffmanBookSource::Fixed && !fixed_book_resident_)
                buildFixedBook(stream);

            if (validate_symbol_range_) {
                // The verdict has to be known *before* encode launches, not after:
                // an out-of-range symbol makes the encode kernel read past d_bk4,
                // and that illegal access has already happened by the time any
                // post-encode check could report it (it takes down the context).
                // So this costs a real stream sync — cheaper than PerBlock's
                // (a 4-byte D2H rather than bklen words, and no host tree build),
                // but a sync nonetheless.  setValidateSymbolRange(false) removes it
                // for callers who guarantee the range upstream.
                //
                // d_freq / h_freq are the histogram buffers, idle on this path —
                // reuse word 0 rather than allocating a flag of our own.
                FZ_CUDA_CHECK(cudaMemsetAsync(buf_->d_freq, 0, sizeof(uint32_t), stream));
                constexpr int kBlock = 256;
                const int grid = static_cast<int>(
                    std::min<size_t>((inlen + kBlock - 1) / kBlock, 4096));
                huffmanSymbolRangeKernel<T><<<grid, kBlock, 0, stream>>>(
                    d_input, inlen, bklen_, buf_->d_freq);
                FZ_CUDA_CHECK(cudaGetLastError());
                FZ_CUDA_CHECK(cudaMemcpyAsync(buf_->h_freq, buf_->d_freq,
                                              sizeof(uint32_t),
                                              cudaMemcpyDeviceToHost, stream));
                FZ_CUDA_CHECK(cudaStreamSynchronize(stream));

                if (buf_->h_freq[0] != 0) {
                    const uint32_t offender = buf_->h_freq[0] - 1u;
                    throw std::runtime_error(
                        "HuffmanStage: symbol " + std::to_string(offender) +
                        " is outside [0, " + std::to_string(bklen_) + "). With a "
                        "pinned codebook (Adaptive/Fixed) nothing histograms the "
                        "input, so the encode kernel would index past the codebook "
                        "and fault. Increase bklen, or use setZigzagCodes(true) "
                        "with LorenzoQuantStage.");
                }
            }
        } else {
            // Zero frequency array (histogram kernel uses atomicAdd into d_freq)
            FZ_CUDA_CHECK(cudaMemsetAsync(
                buf_->d_freq, 0, bklen_ * sizeof(uint32_t), stream));

            // GPU histogram → d_freq
            fz::module::GPU_histogram_generic<T>(
                d_input, inlen,
                buf_->d_freq, static_cast<uint16_t>(bklen_),
                hist_grid_dim_, hist_block_dim_, hist_shmem_use_, hist_r_per_block_,
                stream);

            // D2H: frequency table (async on caller's stream — stream-scoped sync,
            // not a device-wide default-stream barrier).
            FZ_CUDA_CHECK(cudaMemcpyAsync(
                buf_->h_freq, buf_->d_freq,
                bklen_ * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
            FZ_CUDA_CHECK(cudaStreamSynchronize(stream));

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

            if (book_source_ == HuffmanBookSource::Adaptive) {
                // First call only: build a floored, reusable book from this histogram
                // and pin it.  Every later call takes the !need_histogram branch.
                buildAdaptiveBook(buf_->h_freq, stream);
            } else {
                // Build Huffman codebook from host histogram (H2D copies codebook + revbook)
                phf::high_level<T>::build_book(
                    buf_.get(), buf_->h_freq, static_cast<uint16_t>(bklen_), stream);

                // A histogram spanning a wide enough dynamic range drives the rarest
                // symbol past the 27-bit code limit, which the builder clamps silently
                // (see findUnusableCode).  Fail instead of emitting an undecodable stream.
                const int bad = findUnusableCode(buf_->h_freq);
                if (bad >= 0)
                    throw std::runtime_error(
                        "HuffmanStage: symbol " + std::to_string(bad) +
                        " needs a code longer than " +
                        std::to_string(HuffmanWord<4>::FIELD_CODE) +
                        " bits, which the 32-bit codeword format cannot hold. The symbol "
                        "distribution is too skewed for this bklen; reduce bklen, or use "
                        "setBookSource(HuffmanBookSource::Adaptive), which flattens the "
                        "frequency range until the codebook fits.");
            }
        }

        // Encode: GPU_coarse_encode → stream sync → memcpy_merge into buf_->d_encoded
        uint8_t* d_out  = nullptr;
        size_t   outlen = 0;
        phf::high_level<T>::encode(
            buf_.get(), d_input, inlen, &d_out, &outlen, header_, stream);

        // Mirror out which encode path actually ran.  Fine falls back to coarse for
        // books with codes longer than 8 bits, and without this the fallback is
        // invisible to anyone benchmarking the two modes against each other.
        last_used_fine_   = buf_->last_used_fine;
        last_max_codelen_ = buf_->last_max_codelen;

        // Warn on the first fallback and again only if the code length moves, so a
        // resident Fixed/Adaptive book does not emit this once per compress call.
        if (encode_mode_ == HuffmanEncodeMode::Fine && !last_used_fine_ &&
            last_max_codelen_ != warned_max_codelen_) {
            warned_max_codelen_ = last_max_codelen_;
            FZ_LOG(WARN,
                   "HuffmanStage: Fine encode requested but max code length is %u bits "
                   "(> 8); fell back to the coarse path. Measurements from this stage "
                   "are coarse-path numbers.",
                   static_cast<unsigned>(last_max_codelen_));
        }

        // ── Adaptive refit trigger ───────────────────────────────────────────
        // encode() already reports total_nbit, so the achieved bit rate costs
        // nothing to observe — no histogram, no extra kernel, no sync beyond the
        // one encode performs anyway.  Compare it against the rate the resident
        // book achieved on the block it was fitted to; a regression past the
        // threshold means the distribution has moved out from under the book.
        //
        // Unpinning here makes the *next* call histogram and re-pin.  This call is
        // already encoded and is not retroactively improved, which is the price of
        // detecting drift without paying for a histogram to predict it.
        if (book_source_ == HuffmanBookSource::Adaptive && inlen > 0) {
            const double bits_per_sym =
                static_cast<double>(header_.total_nbit) / static_cast<double>(inlen);
            if (just_fitted_) {
                fit_bits_per_sym_ = bits_per_sym;
                just_fitted_      = false;
                calls_since_fit_  = 0;
            } else if (fixed_book_resident_ && refit_interval_ > 0 &&
                       ++calls_since_fit_ >= refit_interval_) {
                // Periodic refit: bounds how long a stale book can persist even when
                // the rate never regressed enough to trip the threshold.
                fixed_book_resident_ = false;
                ++refit_count_;
            } else if (fixed_book_resident_ && refit_threshold_ > 0.0f &&
                       fit_bits_per_sym_ > 0.0 &&
                       bits_per_sym > fit_bits_per_sym_ * refit_threshold_) {
                fixed_book_resident_ = false;   // next call re-histograms and re-fits
                ++refit_count_;
                FZ_LOG(DEBUG,
                       "HuffmanStage: bit rate %.3f b/sym exceeds %.2fx the fitted "
                       "%.3f b/sym — refitting the codebook on the next call (refit #%u).",
                       bits_per_sym, static_cast<double>(refit_threshold_),
                       fit_bits_per_sym_, refit_count_);
            }
        }

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

        // Read the phf_header embedded at the start of the encoded buffer (D2H).
        // Issue on the caller's stream (NOT a plain cudaMemcpy, which would run on
        // the legacy default stream and impose a device-wide barrier); the
        // following stream sync then stalls only this thread, so concurrent decodes
        // on other streams/instances still overlap.
        phf_header hdr {};
        FZ_CUDA_CHECK(cudaMemcpyAsync(
            &hdr, d_encoded, sizeof(hdr), cudaMemcpyDeviceToHost, stream));
        FZ_CUDA_CHECK(cudaStreamSynchronize(stream));

        phf::high_level<T>::decode(
            buf_.get(), hdr, d_encoded, static_cast<T*>(outputs[0]), stream);

        actual_output_size_ = inlen * sizeof(T);
    }
}

template class HuffmanStage<uint8_t>;
template class HuffmanStage<uint16_t>;
template class HuffmanStage<uint32_t>;

} // namespace fz
