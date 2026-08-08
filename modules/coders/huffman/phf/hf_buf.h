// Adapted from PHF reference (origin/v1.1.0_dev:modules/codec/huffman/hf_hl.hh)
// Changes:
//   - Replaced #include "mem/cxx_smart_ptr.h" with local RAII wrappers below.
//   - Replaced #include "err.hh" (not needed in header; hf_buf.cc uses cuda_check.h).
//   - Removed HuffmanCodec<E> class (not used; we use Buf<E> + high_level<E> directly).
//   - Removed timer/io includes (not needed).
//   - Buf<E> refactored to use pool-managed raw pointers instead of unique_ptr members.
//     Constructor takes MemoryPool* and allocates via allocatePersistentDevice/Pinned.
//     Destructor returns all allocations to the pool via freePersistentDevice/Pinned.
//     The pool is the sole owner; Buf<E> holds non-owning raw pointers.

#pragma once

#include "backend/types.h"
#include <cstddef>
#include <cstdint>
#include <memory>

#include "hf.h"
#include "hf_impl.hh"
#include "mem/mempool.h"

// ── HuffmanHelper (used by hf_kernels.cu) ────────────────────────────────────

struct HuffmanHelper {
    static const int BLOCK_DIM_ENCODE   = 256;
    static const int BLOCK_DIM_DEFLATE  = 256;
    static const int ENC_SEQUENTIALITY  = 4;
    static const int DEFLATE_CONSTANT   = 4;
};

// ── HF_SPACE / HF_STREAM convenience macros ──────────────────────────────────

#define HF_SPACE phf::Buf<E>
#define HF_STREAM void*

// ── phf::Buf<E> ──────────────────────────────────────────────────────────────

namespace phf {

template <typename E>
struct Buf {
    using H4 = uint32_t;
    using M  = PHF_METADATA;

    struct RC {
        static const int SCRATCH    = 0;
        static const int FREQ       = 1;
        static const int BK         = 2;
        static const int REVBK      = 3;
        static const int PAR_NBIT   = 4;
        static const int PAR_NCELL  = 5;
        static const int PAR_ENTRY  = 6;
        static const int BITSTREAM  = 7;
        static const int END        = 8;
    };

    struct memcpy_helper {
        void*  const ptr;
        size_t const nbyte;
        size_t const dst;
    };

    using SYM    = E;
    using Header = phf_header;

    // ── Fields ────────────────────────────────────────────────────────────────
    const size_t len;
    size_t       pardeg;
    size_t       sublen;
    const size_t bklen;
    const bool   use_HFR;
    const size_t revbk4_bytes;
    const size_t bitstream_max_len;
    /// H4-word capacity of d_scratch4/h_scratch4.
    ///
    /// The scratch buffer serves two unrelated purposes: phase-1 fill needs one
    /// H4 word per input element (`len`), while `d_encoded` ALIASES it and must
    /// hold the whole merged blob — header + reverse codebook + the two pardeg
    /// arrays + the bitstream. The second is not bounded by the first: the
    /// reverse codebook is a fixed size set by `bklen` (2,304 bytes at
    /// bklen = 1024) and does not shrink with the input, so on small inputs it
    /// dominates. Sizing this at `len` alone therefore overran the allocation
    /// for `len` below ~1,224 elements at bklen = 1024, which `memcpy_merge`
    /// reported as a cudaMemcpyAsync "invalid argument". Hence: the max of both.
    const size_t scratch4_len;

    uint16_t rt_bklen;
    int      numSMs;

    // Diagnostics written by high_level<E>::encode() on every forward call.  The fine
    // path is only *requested* by use_HFR; encode() falls back to coarse when the built
    // book has a code longer than 8 bits, so these record what actually ran.
    uint8_t  last_max_codelen = 0;      ///< Max Huffman code length in the book just used.
    bool     last_used_fine   = false;  ///< True iff GPU_fine_encode ran on the last call.
    size_t   total_footprint_d = 0;
    size_t   total_footprint_h = 0;

    // Device scratch — raw pointers, allocated from pool_ on construction
    H4*       d_scratch4;
    H4*       h_scratch4;
    PHF_BYTE* d_encoded;   // alias into d_scratch4 (not a separate allocation)
    PHF_BYTE* h_encoded;   // alias into h_scratch4 (not a separate allocation)

    H4*       d_bitstream4;
    H4*       h_bitstream4;

    H4*       d_bk4;
    H4*       h_bk4;
    PHF_BYTE* d_revbk4;
    PHF_BYTE* h_revbk4;

    // Per-partition metadata
    M*  d_par_nbit;
    M*  h_par_nbit;
    M*  d_par_ncell;
    M*  h_par_ncell;
    M*  d_par_entry;
    M*  h_par_entry;

    // Histogram buffers — pre-allocated for forward execute; size = bklen each
    uint32_t* d_freq;
    uint32_t* h_freq;

    // Fine-path async totals: populated after GPU_fine_encode; read after caller sync.
    // Null when use_HFR is false.
    uint64_t* d_total_nbit;
    uint64_t* d_total_ncell;
    uint64_t* h_total_nbit;
    uint64_t* h_total_ncell;

    // CUB temp storage for GPU_encode_scan (ExclusiveSum). Null when use_HFR is false.
    uint8_t*  d_cub_temp;
    size_t    cub_temp_bytes;

    // ── Static helpers ────────────────────────────────────────────────────────
    static int _revbk4_bytes(int bklen);
    static int _revbk8_bytes(int bklen);

    /// Worst-case H4-word capacity of the concatenated bitstream.
    ///
    /// This used to be `inlen / 2`, which is not a bound at all — it silently
    /// assumes the encoder averages <= 16 bits per symbol. A Huffman code may be
    /// up to HuffmanWord<4>::FIELD_CODE (27) bits, and any book that is a poor
    /// fit for the data approaches that: a Gaussian book applied to bimodal data
    /// measured 18 bits/symbol and phase4_concatenate wrote 4 KB past the end of
    /// d_bitstream4 (9,216 cells written into an 8,192-cell buffer).
    ///
    /// The real bound is per partition: each of `pardeg` partitions holds at most
    /// `sublen` symbols of at most 27 bits, rounded up to whole 32-bit cells.
    static size_t _bitstream4_len(size_t pardeg, size_t sublen);

    // Non-copyable, non-movable
    Buf(const Buf&)            = delete;
    Buf& operator=(const Buf&) = delete;
    Buf(Buf&&)                 = delete;
    Buf& operator=(Buf&&)      = delete;

    // ── Constructor / destructor ──────────────────────────────────────────────

    /**
     * Allocate all PHF internal buffers from `pool` via
     * allocatePersistentDevice / allocatePersistentPinned.
     * Destructor returns all pointers to the same pool.
     */
    Buf(size_t inlen, size_t _bklen, fz::MemoryPool* pool,
        int _pardeg = -1, bool _use_HFR = false);
    ~Buf();

    // ── Mutators ──────────────────────────────────────────────────────────────
    void register_runtime_bklen(int _rt_bklen) { rt_bklen = _rt_bklen; }

    void memcpy_merge(phf_header& header, phf_stream_t stream);
    void clear_buffer();

private:
    fz::MemoryPool* pool_;  // non-owning; used only in destructor to return allocations
    /// Expires if the pool is destroyed before this Buf. See MemoryPool::lifetimeToken().
    std::weak_ptr<const void> pool_alive_;
};

// ── phf::high_level<E> ───────────────────────────────────────────────────────

template <typename E>
struct high_level {
    // Build codebook from host histogram; H2D copies codebook and revbook.
    static int build_book(Buf<E>* buf, uint32_t* h_hist,
                          uint16_t rt_bklen, HF_STREAM stream);

    // GPU coarse encode: histogram must already be done (histogram D2H happened
    // outside this function to fill h_hist before build_book was called).
    // Output lives at buf->d_encoded; *outlen = phf_encoded_bytes(&header).
    static int encode(Buf<E>* buf, E* in_data, size_t data_len,
                      uint8_t** out_encoded, size_t* encoded_len,
                      phf_header& header, HF_STREAM stream);

    // GPU coarse decode: reads phf_header from in_encoded[0..127],
    // reconstructs symbols into out_decoded.
    static int decode(Buf<E>* buf, phf_header& header,
                      PHF_BYTE* in_encoded, E* out_decoded, HF_STREAM stream);
};

}  // namespace phf
