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
#include "stage/stage_registry.h"
#include <cstring>
#include <algorithm>
#include <stdexcept>
#include <string>
#include "coders/huffman/phf/hf_buf.h"
#include "util/histogram/histogram.h"
#include "mem/mempool.h"
#include "cuda_check.h"
#include "log.h"

#include "backend/algorithms.h"
#include "backend/api.h"
#include "backend/cub.h"
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

// Device-resident phase 1. Unlike the vendored kernel, this path never indexes
// past the codebook: it records an invalid symbol and emits symbol zero's code so
// the remainder of the captured pipeline can finish safely. postStreamSync()
// reports the error after the pipeline's normal completion barrier.
template <typename T>
__global__ void huffmanDeviceFillKernel(
    const T* __restrict__ in, size_t n,
    const uint32_t* __restrict__ book, uint32_t bklen,
    uint32_t* __restrict__ encoded, uint32_t* __restrict__ range_error)
{
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += stride) {
        const uint32_t symbol = static_cast<uint32_t>(in[i]);
        if (symbol < bklen) {
            encoded[i] = book[symbol];
        } else {
            encoded[i] = book[0];
            const uint32_t marker = symbol == UINT32_MAX ? UINT32_MAX : symbol + 1u;
            atomicMax(range_error, marker);
        }
    }
}

__global__ void huffmanAssembleDeviceKernel(
    uint8_t* __restrict__ out,
    const uint8_t* __restrict__ revbook, size_t revbook_bytes,
    const PHF_METADATA* __restrict__ par_nbit,
    const PHF_METADATA* __restrict__ par_ncell,
    const PHF_METADATA* __restrict__ par_entry,
    uint32_t bklen, uint32_t sublen, uint32_t pardeg, size_t original_len)
{
    const size_t revbook_off = PHFHEADER_FORCED_ALIGN;
    const size_t nbit_off = revbook_off + revbook_bytes;
    const size_t entry_off = nbit_off + static_cast<size_t>(pardeg) * sizeof(PHF_METADATA);
    const size_t bitstream_off = entry_off + static_cast<size_t>(pardeg) * sizeof(PHF_METADATA);

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const size_t total_ncell = static_cast<size_t>(par_entry[pardeg - 1])
                                 + par_ncell[pardeg - 1];
        auto* header = reinterpret_cast<phf_header*>(out);
        header->bklen = static_cast<int>(bklen);
        header->sublen = static_cast<int>(sublen);
        header->pardeg = static_cast<int>(pardeg);
        header->original_len = original_len;
        header->total_nbit = 0;
        header->total_ncell = total_ncell;
        header->entry[PHFHEADER_HEADER] = 0;
        header->entry[PHFHEADER_REVBK] = static_cast<uint32_t>(revbook_off);
        header->entry[PHFHEADER_PAR_NBIT] = static_cast<uint32_t>(nbit_off);
        header->entry[PHFHEADER_PAR_ENTRY] = static_cast<uint32_t>(entry_off);
        header->entry[PHFHEADER_BITSTREAM] = static_cast<uint32_t>(bitstream_off);
        header->entry[PHFHEADER_END] =
            static_cast<uint32_t>(bitstream_off + total_ncell * sizeof(uint32_t));
    }

    const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = tid; i < revbook_bytes; i += stride)
        out[revbook_off + i] = revbook[i];
    for (size_t i = tid; i < pardeg; i += stride) {
        reinterpret_cast<PHF_METADATA*>(out + nbit_off)[i] = par_nbit[i];
        reinterpret_cast<PHF_METADATA*>(out + entry_off)[i] = par_entry[i];
    }
}

__global__ void huffmanTotalBitsDeviceKernel(
    const PHF_METADATA* __restrict__ par_nbit, uint32_t pardeg,
    uint8_t* __restrict__ out)
{
    __shared__ unsigned long long sums[256];
    unsigned long long local = 0;
    for (uint32_t i = threadIdx.x; i < pardeg; i += blockDim.x)
        local += par_nbit[i];
    sums[threadIdx.x] = local;
    __syncthreads();
    for (uint32_t offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) sums[threadIdx.x] += sums[threadIdx.x + offset];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        reinterpret_cast<phf_header*>(out)->total_nbit =
            static_cast<size_t>(sums[0]);
}

enum : uint32_t {
    HUF_BOOK_PER_BLOCK = 0,
    HUF_BOOK_ADAPTIVE  = 1,
    HUF_BOOK_FIXED     = 2,
};

enum : uint32_t {
    HUF_BOOK_STATUS_OK       = 0,
    HUF_BOOK_STATUS_RANGE    = 1,
    HUF_BOOK_STATUS_OVERLONG = 2,
};

enum : uint32_t {
    HUF_BOOK_FLAG_BUILT      = 1u << 0,
    HUF_BOOK_FLAG_ADAPTIVE   = 1u << 1,
    HUF_BOOK_FLAG_DEGENERATE = 1u << 2,
};

// Serial tree construction on one device thread. The heap operations intentionally
// mirror cuSZ's CPU builder, including its strict/non-strict frequency tie rules,
// so canonical output stays byte-identical. For the common <= 1024-symbol books,
// the working set lives in shared memory: the old global-memory heap made every
// dependent sift step pay device-memory latency. Larger books retain the global
// scratch fallback. Measurements and rationale: docs/codebase_notes.md CN-HUFFMAN-1.
template <typename T>
__global__ void huffmanBuildCanonicalBookDeviceKernel(
    const uint32_t* __restrict__ source_freq, uint32_t bklen,
    size_t expected_symbols, uint32_t source_mode, uint32_t requested_floor_shift,
    bool validate_range, uint32_t* __restrict__ book,
    uint8_t* __restrict__ reverse_book, uint32_t* __restrict__ meta,
    uint64_t* __restrict__ node_freq, int32_t* __restrict__ node_parent,
    int32_t* __restrict__ heap, int32_t* __restrict__ leaf_node,
    uint32_t* __restrict__ work_freq, bool use_shared_scratch)
{
    extern __shared__ __align__(8) uint8_t shared_scratch[];
    if (use_shared_scratch) {
        const size_t node_capacity = 2 * static_cast<size_t>(bklen);
        const size_t freq_bytes = node_capacity * sizeof(uint64_t);
        const size_t parent_bytes = node_capacity * sizeof(int32_t);
        const size_t heap_bytes = (node_capacity + 1) * sizeof(int32_t);
        const size_t leaf_bytes = static_cast<size_t>(bklen) * sizeof(int32_t);
        node_freq = reinterpret_cast<uint64_t*>(shared_scratch);
        node_parent = reinterpret_cast<int32_t*>(shared_scratch + freq_bytes);
        heap = reinterpret_cast<int32_t*>(shared_scratch + freq_bytes + parent_bytes);
        leaf_node = reinterpret_cast<int32_t*>(
            shared_scratch + freq_bytes + parent_bytes + heap_bytes);
        work_freq = reinterpret_cast<uint32_t*>(
            shared_scratch + freq_bytes + parent_bytes + heap_bytes + leaf_bytes);
    }
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    constexpr uint32_t kMaxCodeBits = HuffmanWord<4>::FIELD_CODE;
    constexpr uint32_t kTypeBits = 32;
    constexpr uint32_t kFill = 0xffffffffu;

    uint64_t source_total = 0;
    uint32_t max_freq = 0;
    uint32_t distinct = 0;
    for (uint32_t i = 0; i < bklen; ++i) {
        const uint32_t f = source_freq[i];
        source_total += f;
        max_freq = f > max_freq ? f : max_freq;
        distinct += f != 0;
    }

    uint32_t status = HUF_BOOK_STATUS_OK;
    bool force_uniform = false;
    if (source_mode != HUF_BOOK_FIXED && source_total != expected_symbols) {
        if (validate_range) status = HUF_BOOK_STATUS_RANGE;
        force_uniform = true;  // keep later encode kernels memory-safe
    }
    if (max_freq == 0) force_uniform = true;

    bool adaptive = source_mode == HUF_BOOK_ADAPTIVE;
    uint32_t shift = requested_floor_shift;
    uint32_t max_depth = 0;

    for (;;) {
        uint32_t floor_value = 0;
        if (adaptive && !force_uniform)
            floor_value = max(1u, max_freq >> shift);

        int32_t node_count = 0;
        int32_t heap_end = 1;
        for (uint32_t symbol = 0; symbol < bklen; ++symbol) {
            const uint32_t f = force_uniform ? 1u
                : (adaptive ? max(source_freq[symbol], floor_value)
                            : source_freq[symbol]);
            work_freq[symbol] = f;
            leaf_node[symbol] = -1;
            if (f == 0) continue;

            const int32_t node = node_count++;
            node_freq[node] = f;
            node_parent[node] = -1;
            leaf_node[symbol] = node;

            int32_t i = heap_end++;
            int32_t parent = i >> 1;
            while (parent != 0 && node_freq[heap[parent]] > node_freq[node]) {
                heap[i] = heap[parent];
                i = parent;
                parent = i >> 1;
            }
            heap[i] = node;
        }

        auto remove_min = [&]() {
            const int32_t result = heap[1];
            --heap_end;
            heap[1] = heap[heap_end];
            int32_t i = 1;
            while ((i << 1) < heap_end) {
                int32_t child = i << 1;
                if (child + 1 < heap_end
                    && node_freq[heap[child + 1]] < node_freq[heap[child]])
                    ++child;
                if (node_freq[heap[i]] > node_freq[heap[child]]) {
                    const int32_t tmp = heap[i];
                    heap[i] = heap[child];
                    heap[child] = tmp;
                    i = child;
                } else {
                    break;
                }
            }
            return result;
        };

        while (heap_end > 2) {
            const int32_t left = remove_min();
            const int32_t right = remove_min();
            const int32_t node = node_count++;
            node_freq[node] = node_freq[left] + node_freq[right];
            node_parent[node] = -1;
            node_parent[left] = node;
            node_parent[right] = node;

            int32_t i = heap_end++;
            int32_t parent = i >> 1;
            while (parent != 0 && node_freq[heap[parent]] > node_freq[node]) {
                heap[i] = heap[parent];
                i = parent;
                parent = i >> 1;
            }
            heap[i] = node;
        }

        max_depth = 0;
        for (uint32_t symbol = 0; symbol < bklen; ++symbol) {
            int32_t node = leaf_node[symbol];
            uint32_t depth = 0;
            if (node >= 0) {
                while (node_parent[node] >= 0) {
                    ++depth;
                    node = node_parent[node];
                }
                if (depth == 0) depth = 1;
            }
            work_freq[symbol] = depth;
            max_depth = depth > max_depth ? depth : max_depth;
        }

        if (max_depth <= kMaxCodeBits) break;

        if (source_mode == HUF_BOOK_FIXED) {
            status = HUF_BOOK_STATUS_OVERLONG;
            force_uniform = true;
            adaptive = false;
        } else {
            adaptive = true;
            shift /= 2;
            if (shift == 0) force_uniform = true;
        }
    }

    uint32_t counts[kTypeBits] = {};
    int32_t entries[kTypeBits] = {};
    int32_t first[kTypeBits] = {};
    int32_t iter[kTypeBits] = {};
    for (uint32_t symbol = 0; symbol < bklen; ++symbol) {
        const uint32_t depth = work_freq[symbol];
        if (depth != 0) ++counts[depth];
        book[symbol] = kFill;
    }
    for (uint32_t i = 1; i < kTypeBits; ++i)
        entries[i] = entries[i - 1] + static_cast<int32_t>(counts[i - 1]);
    for (uint32_t i = 0; i < kTypeBits; ++i) iter[i] = entries[i];
    first[max_depth] = 0;
    for (int32_t len = static_cast<int32_t>(max_depth) - 1; len >= 1; --len)
        first[len] = (first[len + 1] + static_cast<int32_t>(counts[len + 1]) + 1) / 2;
    first[0] = 0xff;

    auto* reverse_first = reinterpret_cast<int32_t*>(reverse_book);
    auto* reverse_entry = reverse_first + kTypeBits;
    auto* reverse_keys = reinterpret_cast<T*>(reverse_entry + kTypeBits);
    for (uint32_t i = 0; i < kTypeBits; ++i) {
        reverse_first[i] = first[i];
        reverse_entry[i] = entries[i];
    }
    for (uint32_t i = 0; i < bklen; ++i) reverse_keys[i] = T{};

    for (uint32_t symbol = 0; symbol < bklen; ++symbol) {
        const uint32_t len = work_freq[symbol];
        if (len == 0) continue;
        const int32_t position = iter[len]++;
        const uint32_t code = static_cast<uint32_t>(
            first[len] + position - entries[len]);
        book[symbol] = code | (len << kMaxCodeBits);
        reverse_keys[position] = static_cast<T>(symbol);
    }

    const bool degenerate = source_mode != HUF_BOOK_FIXED
        && (distinct < 2 || (source_total > 0
            && static_cast<double>(max_freq) >= 0.999 * static_cast<double>(source_total)));
    meta[0] = status;
    meta[1] = HUF_BOOK_FLAG_BUILT
            | (adaptive ? HUF_BOOK_FLAG_ADAPTIVE : 0u)
            | (degenerate ? HUF_BOOK_FLAG_DEGENERATE : 0u);
    meta[2] = shift;
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
    buf_               = std::make_unique<phf::Buf<T>>(inlen, bklen_, pool);
    // Fresh d_bk4 / d_revbk4 allocations: whatever fixed book was resident is gone.
    fixed_book_resident_ = false;
    pool_              = pool;
    cap_inlen_         = inlen;
    last_bklen_        = bklen_;
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

template <typename T>
void HuffmanStage<T>::buildDeviceBook(
    fz::stream_t stream, MemoryPool* pool, size_t expected_symbols,
    uint32_t source_mode)
{
    const size_t node_capacity = 2 * static_cast<size_t>(bklen_);
    const size_t freq_bytes = node_capacity * sizeof(uint64_t);
    const size_t parent_bytes = node_capacity * sizeof(int32_t);
    const size_t heap_bytes = (node_capacity + 1) * sizeof(int32_t);
    const size_t leaf_bytes = static_cast<size_t>(bklen_) * sizeof(int32_t);
    const size_t work_bytes = static_cast<size_t>(bklen_) * sizeof(uint32_t);
    const size_t scratch_bytes =
        freq_bytes + parent_bytes + heap_bytes + leaf_bytes + work_bytes;

    // 1024 symbols require 40,964 bytes, below CUDA's portable 48 KiB dynamic
    // shared-memory limit. Keep arbitrary larger bklen values supported through
    // the previous pool-backed path.
    const bool use_shared_scratch = bklen_ <= 1024;
    auto* scratch = use_shared_scratch ? nullptr : static_cast<uint8_t*>(
        pool->allocate(scratch_bytes, stream, "huf_device_book"));
    auto* node_freq = reinterpret_cast<uint64_t*>(scratch);
    auto* node_parent = scratch ? reinterpret_cast<int32_t*>(scratch + freq_bytes) : nullptr;
    auto* heap = scratch ? reinterpret_cast<int32_t*>(
        scratch + freq_bytes + parent_bytes) : nullptr;
    auto* leaf_node = scratch ? reinterpret_cast<int32_t*>(
        scratch + freq_bytes + parent_bytes + heap_bytes) : nullptr;
    auto* work_freq = scratch ? reinterpret_cast<uint32_t*>(
        scratch + freq_bytes + parent_bytes + heap_bytes + leaf_bytes) : nullptr;

    buf_->register_runtime_bklen(static_cast<uint16_t>(bklen_));
    FZ_CUDA_CHECK(cudaMemsetAsync(
        buf_->d_book_meta, 0, 4 * sizeof(uint32_t), stream));
    huffmanBuildCanonicalBookDeviceKernel<T><<<
        1, 1, use_shared_scratch ? scratch_bytes : 0, stream>>>(
        buf_->d_freq, bklen_, expected_symbols, source_mode,
        adaptive_floor_shift_, validate_symbol_range_, buf_->d_bk4,
        buf_->d_revbk4, buf_->d_book_meta, node_freq, node_parent,
        heap, leaf_node, work_freq, use_shared_scratch);
    FZ_CUDA_CHECK(cudaGetLastError());
    if (scratch) pool->free(scratch, stream);
}

template <typename T>
void HuffmanStage<T>::updateAdaptiveRate(size_t inlen, size_t total_nbit)
{
    if ((book_source_ != HuffmanBookSource::Adaptive && !adaptive_fallback_)
        || inlen == 0)
        return;

    const double bits_per_sym =
        static_cast<double>(total_nbit) / static_cast<double>(inlen);
    if (just_fitted_) {
        fit_bits_per_sym_ = bits_per_sym;
        just_fitted_      = false;
        calls_since_fit_  = 0;
    } else if (fixed_book_resident_ && refit_interval_ > 0
               && ++calls_since_fit_ >= refit_interval_) {
        fixed_book_resident_ = false;
        ++refit_count_;
    } else if (fixed_book_resident_ && refit_threshold_ > 0.0f
               && fit_bits_per_sym_ > 0.0
               && bits_per_sym > fit_bits_per_sym_ * refit_threshold_) {
        fixed_book_resident_ = false;
        ++refit_count_;
        FZ_LOG(DEBUG,
               "HuffmanStage: bit rate %.3f b/sym exceeds %.2fx the fitted "
               "%.3f b/sym — refitting the codebook on the next call (refit #%u).",
               bits_per_sym, static_cast<double>(refit_threshold_),
               fit_bits_per_sym_, refit_count_);
    }
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
    const size_t sublen = capi_phf_coarse_tune_sublen(n);
    const size_t pardeg = (n - 1) / sublen + 1;
    const size_t bitstream_words =
        pardeg * ((sublen * HuffmanWord<4>::FIELD_CODE + 31) / 32);
    const size_t reverse_bytes =
        phf_reverse_book_bytes(static_cast<uint16_t>(bklen_), 4, sizeof(T));
    using M = PHF_METADATA;
    using H4 = uint32_t;
    const size_t archive_words =
        (PHFHEADER_FORCED_ALIGN + reverse_bytes + 2 * pardeg * sizeof(M)
         + bitstream_words * sizeof(H4) + sizeof(H4) - 1) / sizeof(H4);
    size_t base = sizeof(H4) * std::max(n, archive_words) // d_scratch4
                + sizeof(H4) * bklen_                     // d_bk4
                + reverse_bytes                           // d_revbk4
                + sizeof(H4) * bitstream_words            // d_bitstream4
                + sizeof(M) * pardeg * 3                  // d_par_nbit/ncell/entry
                + sizeof(uint32_t) * bklen_               // d_freq
                + sizeof(uint32_t) * 4;                   // d_book_meta
    return base;
}

template <typename T>
size_t HuffmanStage<T>::estimatePinnedFootprintBytes(size_t inlen) const
{
    if (inlen == 0) return 0;
    const size_t n = inlen / sizeof(T);
    const size_t sublen = capi_phf_coarse_tune_sublen(n);
    const size_t pardeg = (n - 1) / sublen + 1;
    const size_t bitstream_words =
        pardeg * ((sublen * HuffmanWord<4>::FIELD_CODE + 31) / 32);
    const size_t reverse_bytes =
        phf_reverse_book_bytes(static_cast<uint16_t>(bklen_), 4, sizeof(T));
    using M = PHF_METADATA;
    using H4 = uint32_t;
    const size_t archive_words =
        (PHFHEADER_FORCED_ALIGN + reverse_bytes + 2 * pardeg * sizeof(M)
         + bitstream_words * sizeof(H4) + sizeof(H4) - 1) / sizeof(H4);
    size_t base = sizeof(H4) * std::max(n, archive_words) // h_scratch4
                + sizeof(H4) * bklen_                     // h_bk4
                + reverse_bytes                           // h_revbk4
                + sizeof(H4) * bitstream_words            // h_bitstream4
                + sizeof(M) * pardeg * 3                  // h_par_nbit/ncell/entry
                + sizeof(uint32_t) * bklen_               // h_freq
                + sizeof(uint32_t) * 4;                   // h_book_meta
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

    pending_device_readback_ = false;
    pending_device_output_ = nullptr;
    pending_device_inlen_ = 0;
    if (sizes[0] == 0) { actual_output_size_ = 0; return; }

    if (!is_inverse_) {
        // ── Forward: T[] → PHF bitstream ─────────────────────────────────────
        T*     d_input = static_cast<T*>(inputs[0]);
        size_t inlen   = sizes[0] / sizeof(T);

        // Reallocate when capacity is exceeded or bklen changes.
        // Use pool from execute() parameter; on first call, this also sets pool_.
        if (!buf_ || inlen > cap_inlen_ || bklen_ != last_bklen_)
            initBuf(inlen, pool);

        // The histogram (and with it the D2H, the host sync, and the out-of-range
        // symbol check) runs every call under PerBlock, once under Adaptive, and
        // never under Fixed.
        // A stage that fell back to Adaptive follows Adaptive's rule from then on:
        // histogram only while no book is pinned. Leaving it on PerBlock's rule would
        // rebuild the same floored book — and re-log the same warning — every call.
        const bool adaptive_rule =
            (book_source_ == HuffmanBookSource::Adaptive) || adaptive_fallback_;
        const bool need_histogram =
            (book_source_ == HuffmanBookSource::PerBlock && !adaptive_fallback_) ||
            (adaptive_rule && !fixed_book_resident_);

        if (execution_mode_ == HuffmanExecutionMode::DeviceResident) {
            FZ_CUDA_CHECK(cudaMemsetAsync(
                buf_->d_book_meta, 0, 4 * sizeof(uint32_t), stream));

            if (need_histogram) {
                FZ_CUDA_CHECK(cudaMemsetAsync(
                    buf_->d_freq, 0, bklen_ * sizeof(uint32_t), stream));
                fz::module::GPU_histogram_generic<T>(
                    d_input, inlen,
                    buf_->d_freq, static_cast<uint16_t>(bklen_),
                    hist_grid_dim_, hist_block_dim_, hist_shmem_use_, hist_r_per_block_,
                    stream);

                buildDeviceBook(
                    stream, pool, inlen,
                    adaptive_rule ? HUF_BOOK_ADAPTIVE : HUF_BOOK_PER_BLOCK);
            } else if (book_source_ == HuffmanBookSource::Fixed
                       && !fixed_book_resident_) {
                if (fixed_freq_.empty())
                    throw std::runtime_error(
                        "HuffmanStage: book source is Fixed but no codebook has been "
                        "supplied; call setFixedBookFromFreq() or "
                        "setFixedBookFromModel() first");
                if (fixed_freq_.size() != bklen_)
                    throw std::runtime_error(
                        "HuffmanStage: fixed codebook has "
                        + std::to_string(fixed_freq_.size())
                        + " entries but bklen is " + std::to_string(bklen_)
                        + "; setBklen() was called after the codebook was set");
                FZ_CUDA_CHECK(cudaMemcpyAsync(
                    buf_->d_freq, fixed_freq_.data(),
                    bklen_ * sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
                buildDeviceBook(stream, pool, inlen, HUF_BOOK_FIXED);
            }
        } else if (!need_histogram) {
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

            if (book_source_ == HuffmanBookSource::Adaptive || adaptive_fallback_) {
                // First call only: build a floored, reusable book from this histogram
                // and pin it.  Every later call takes the !need_histogram branch.
                buildAdaptiveBook(buf_->h_freq, stream);
            } else {
                // Build Huffman codebook from host histogram (H2D copies codebook + revbook)
                phf::high_level<T>::build_book(
                    buf_.get(), buf_->h_freq, static_cast<uint16_t>(bklen_), stream);

                // A histogram spanning a wide enough dynamic range drives the rarest
                // symbol past the 27-bit code limit, which the builder clamps silently
                // (see findUnusableCode) and then emits a stream nothing can decode.
                //
                // Fall back to an Adaptive book rather than fail. This is NOT a
                // relaxation of the error bound -- the bound belongs to the quantizer,
                // and flooring the frequencies only changes how symbols are spelled,
                // not what they mean. Throwing instead cost the cuSZ preset every
                // wide-dynamic-range field in the corpus.
                // Measurements: docs/codebase_notes.md CN-HF-4
                //
                // buildAdaptiveBook halves the floor shift until the book fits and
                // shift 0 is uniform, so the fallback always terminates.
                const int bad = findUnusableCode(buf_->h_freq);
                if (bad >= 0) {
                    FZ_LOG(WARN,
                        "HuffmanStage: symbol %d needs a code longer than %d bits, which "
                        "the 32-bit codeword format cannot hold; the symbol distribution "
                        "is too skewed for bklen=%u. Falling back to an Adaptive "
                        "(frequency-floored) codebook for the rest of this stage's life. "
                        "The error bound is unaffected. Set "
                        "setBookSource(HuffmanBookSource::Adaptive) explicitly, or reduce "
                        "bklen, to choose this deliberately.",
                        bad, static_cast<int>(HuffmanWord<4>::FIELD_CODE), bklen_);
                    adaptive_fallback_ = true;
                    buildAdaptiveBook(buf_->h_freq, stream);
                }
            }
        }

        if (execution_mode_ == HuffmanExecutionMode::DeviceResident) {
            const uint32_t actual_pardeg = static_cast<uint32_t>(
                (inlen - 1) / buf_->sublen + 1);
            const phf::par_config hfpar{
                buf_->sublen, static_cast<size_t>(actual_pardeg)};
            using Module = phf::cuhip::modules<T, uint32_t>;

            // Phase 1 also performs the pinned-book range check without ever
            // indexing outside d_bk4. Histogram-building sources have already
            // validated this input, but resetting the flag keeps the path uniform.
            FZ_CUDA_CHECK(cudaMemsetAsync(
                buf_->d_book_meta + 3, 0, sizeof(uint32_t), stream));
            const int fill_grid = static_cast<int>(
                std::min<size_t>((inlen + 255) / 256, 8 * static_cast<size_t>(buf_->numSMs)));
            huffmanDeviceFillKernel<T><<<fill_grid, 256, 0, stream>>>(
                d_input, inlen, buf_->d_bk4, bklen_, buf_->d_scratch4,
                buf_->d_book_meta + 3);
            FZ_CUDA_CHECK(cudaGetLastError());

            Module::GPU_coarse_encode_phase2(
                buf_->d_scratch4, inlen, hfpar, buf_->d_scratch4,
                buf_->d_par_nbit, buf_->d_par_ncell, stream);

            auto scan_tmp = fz::backend::withTempStorage(
                pool, stream, "huf_device_scan",
                [&](void* tmp, size_t& bytes) {
                    cub::DeviceScan::ExclusiveSum(
                        tmp, bytes, buf_->d_par_ncell, buf_->d_par_entry,
                        actual_pardeg, stream);
                });
            fz::backend::freeTempStorage(pool, scan_tmp, stream);

            const size_t revbook_off = PHFHEADER_FORCED_ALIGN;
            const size_t nbit_off = revbook_off + buf_->revbk4_bytes;
            const size_t entry_off = nbit_off
                + static_cast<size_t>(actual_pardeg) * sizeof(PHF_METADATA);
            const size_t bitstream_off = entry_off
                + static_cast<size_t>(actual_pardeg) * sizeof(PHF_METADATA);
            auto* d_output = static_cast<uint8_t*>(outputs[0]);
            auto* d_bitstream = reinterpret_cast<uint32_t*>(d_output + bitstream_off);

            FZ_CUDA_CHECK(cudaMemsetAsync(
                d_output, 0, PHFHEADER_FORCED_ALIGN, stream));

            Module::GPU_coarse_encode_phase4_device(
                buf_->d_scratch4, buf_->d_par_entry, buf_->d_par_ncell,
                hfpar, d_bitstream, stream);

            const size_t copy_items = std::max<size_t>(
                buf_->revbk4_bytes, actual_pardeg);
            const int copy_grid = static_cast<int>(
                std::min<size_t>((copy_items + 255) / 256, 4096));
            huffmanAssembleDeviceKernel<<<copy_grid, 256, 0, stream>>>(
                d_output, buf_->d_revbk4, buf_->revbk4_bytes,
                buf_->d_par_nbit, buf_->d_par_ncell, buf_->d_par_entry,
                bklen_, static_cast<uint32_t>(buf_->sublen), actual_pardeg, inlen);
            huffmanTotalBitsDeviceKernel<<<1, 256, 0, stream>>>(
                buf_->d_par_nbit, actual_pardeg, d_output);
            FZ_CUDA_CHECK(cudaGetLastError());

            // The exact end offset and any range error remain on the device until
            // the pipeline's normal completion barrier.
            pending_device_output_ = d_output;
            pending_device_readback_ = true;
            pending_device_inlen_ = inlen;
            actual_output_size_ = estimateOutputSizes({sizes[0]})[0];
            original_len_ = inlen;
            if (!is_terminal_output_)
                postStreamSync(stream);
            return;
        }

        // Encode: GPU_coarse_encode → stream sync → memcpy_merge into buf_->d_encoded
        uint8_t* d_out  = nullptr;
        size_t   outlen = 0;
        phf::high_level<T>::encode(
            buf_.get(), d_input, inlen, &d_out, &outlen, header_, stream);

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
        updateAdaptiveRate(inlen, header_.total_nbit);

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

        // DeviceResident parses the embedded header in the decode kernel. The
        // legacy path reads it on the caller's stream (rather than the default
        // stream) and synchronizes only that stream before launching decode.
        if (execution_mode_ == HuffmanExecutionMode::DeviceResident) {
            const size_t estimated_pardeg =
                (inlen - 1) / buf_->sublen + 1;
            phf::cuhip::modules<T, uint32_t>::GPU_coarse_decode_device(
                d_encoded, buf_->revbk4_bytes, estimated_pardeg, buf_->numSMs,
                static_cast<T*>(outputs[0]), stream);
            FZ_CUDA_CHECK(cudaGetLastError());
        } else {
            phf_header hdr {};
            FZ_CUDA_CHECK(cudaMemcpyAsync(
                &hdr, d_encoded, sizeof(hdr), cudaMemcpyDeviceToHost, stream));
            FZ_CUDA_CHECK(cudaStreamSynchronize(stream));

            phf::high_level<T>::decode(
                buf_.get(), hdr, d_encoded, static_cast<T*>(outputs[0]), stream);
        }

        actual_output_size_ = inlen * sizeof(T);
    }
}

template <typename T>
void HuffmanStage<T>::postStreamSync(cudaStream_t stream)
{
    if (!pending_device_readback_) return;

    FZ_CUDA_CHECK(cudaMemcpyAsync(
        &header_, pending_device_output_, sizeof(header_),
        cudaMemcpyDeviceToHost, stream));
    FZ_CUDA_CHECK(cudaMemcpyAsync(
        buf_->h_book_meta, buf_->d_book_meta, 4 * sizeof(uint32_t),
        cudaMemcpyDeviceToHost, stream));
    FZ_CUDA_CHECK(cudaStreamSynchronize(stream));

    pending_device_readback_ = false;
    pending_device_output_ = nullptr;
    actual_output_size_ = header_.entry[PHFHEADER_END];
    const size_t completed_inlen = pending_device_inlen_;
    pending_device_inlen_ = 0;

    const uint32_t book_status = buf_->h_book_meta[0];
    const uint32_t book_flags = buf_->h_book_meta[1];
    const uint32_t range_error = buf_->h_book_meta[3];

    if (validate_symbol_range_
        && (book_status == HUF_BOOK_STATUS_RANGE || range_error != 0))
        throw std::runtime_error(
            "HuffmanStage: DeviceResident input contains a symbol outside [0, "
            + std::to_string(bklen_) + ")");

    if (book_status == HUF_BOOK_STATUS_OVERLONG)
        throw std::runtime_error(
            "HuffmanStage: fixed codebook needs a code longer than "
            + std::to_string(HuffmanWord<4>::FIELD_CODE)
            + " bits, which the 32-bit codeword format cannot hold. Reduce the "
              "frequency dynamic range (a wider model scale, or fewer symbols).");

    if ((book_flags & HUF_BOOK_FLAG_BUILT) != 0) {
        if ((book_flags & HUF_BOOK_FLAG_ADAPTIVE) != 0) {
            adaptive_shift_used_ = static_cast<uint8_t>(buf_->h_book_meta[2]);
            has_book_spec_ = false;
            const bool degenerate =
                (book_flags & HUF_BOOK_FLAG_DEGENERATE) != 0;
            fixed_book_resident_ = !degenerate;
            just_fitted_ = !degenerate;
            if (book_source_ == HuffmanBookSource::PerBlock && !adaptive_fallback_) {
                adaptive_fallback_ = true;
                FZ_LOG(WARN,
                    "HuffmanStage: device-built PerBlock tree exceeded the %d-bit "
                    "code field; using a frequency-floored Adaptive book for the "
                    "rest of this stage's life. The error bound is unaffected.",
                    static_cast<int>(HuffmanWord<4>::FIELD_CODE));
            }
            if (degenerate)
                FZ_LOG(DEBUG,
                    "HuffmanStage: device-side Adaptive sample was degenerate; "
                    "using its book for this call but not pinning it.");
        } else if (book_source_ == HuffmanBookSource::Fixed) {
            fixed_book_resident_ = true;
        }
    }

    updateAdaptiveRate(completed_inlen, header_.total_nbit);
}

template class HuffmanStage<uint8_t>;
template class HuffmanStage<uint16_t>;
template class HuffmanStage<uint32_t>;

} // namespace fz

// ── FZM-header reconstruction (self-registered; see stage_registry.h) ─────────
namespace {
fz::Stage* Huffman_fromHeader(const uint8_t* config, size_t config_size) {
    using fz::DataType; using fz::HuffmanStage; using fz::Stage;
    DataType dt = (config_size > 0) ? static_cast<DataType>(config[0]) : DataType::UINT16;
    Stage* stage = nullptr;
    if      (dt == DataType::UINT8)  stage = new HuffmanStage<uint8_t>();
    else if (dt == DataType::UINT16) stage = new HuffmanStage<uint16_t>();
    else if (dt == DataType::UINT32) stage = new HuffmanStage<uint32_t>();
    else throw std::runtime_error("Unsupported HuffmanStage DataType: "
            + std::to_string(static_cast<int>(dt)));
    stage->deserializeHeader(config, config_size);
    return stage;
}
}  // namespace
FZ_REGISTER_STAGE_FACTORY(fz::StageType::HUFFMAN, Huffman_fromHeader);
